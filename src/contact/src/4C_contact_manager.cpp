// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_manager.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_lagrange_strategy.hpp"
#include "4C_contact_lagrange_strategy_poro.hpp"
#include "4C_contact_lagrange_strategy_tsi.hpp"
#include "4C_contact_lagrange_strategy_wear.hpp"
#include "4C_contact_nitsche_strategy.hpp"
#include "4C_contact_nitsche_strategy_fpi.hpp"
#include "4C_contact_nitsche_strategy_fsi.hpp"
#include "4C_contact_nitsche_strategy_poro.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_penalty_strategy.hpp"
#include "4C_contact_strategy_factory.hpp"
#include "4C_contact_utils.hpp"
#include "4C_contact_utils_parallel.hpp"
#include "4C_contact_wear_interface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 03/08|
 *----------------------------------------------------------------------*/
CONTACT::Manager::Manager(Core::FE::Discretization& discret, double alphaf)
    : Mortar::ManagerBase(), discret_(discret)
{
  // overwrite base class communicator
  comm_ = discret.get_comm();

  // create some local variables (later to be stored in strategy)
  const int dim = Global::Problem::instance()->n_dim();
  if (dim != 2 && dim != 3) FOUR_C_THROW("Contact problem must be 2D or 3D");
  std::vector<std::shared_ptr<CONTACT::Interface>> interfaces;
  Teuchos::ParameterList contactParams;

  // read and check contact input parameters
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Checking contact input parameters..........." << std::endl;

  read_and_check_input(contactParams);
  if (Core::Communication::my_mpi_rank(get_comm()) == 0) std::cout << "done!" << std::endl;

  // check for fill_complete of discretization
  if (!discret.filled()) FOUR_C_THROW("discretization is not fillcomplete");

  // let's check for contact boundary conditions in the discretization and and detect groups of
  // matching conditions. For each group, create a contact interface and store it.
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "Building contact interface(s)..............." << std::endl;

  // Vector that contains solid-to-solid and beam-to-solid contact pairs
  std::vector<const Core::Conditions::Condition*> beamandsolidcontactconditions;
  discret.get_condition("Contact", beamandsolidcontactconditions);

  // Vector that solely contains solid-to-solid contact pairs
  std::vector<const Core::Conditions::Condition*> contactconditions;

  // Sort out beam-to-solid contact pairs, since these are treated in the beam3contact framework
  for (const auto& beamSolidCondition : beamandsolidcontactconditions)
  {
    if (beamSolidCondition->parameters().get<std::string>("Application") != "Beamtosolidcontact")
      contactconditions.push_back(beamSolidCondition);
  }

  // there must be more than one contact condition
  // unless we have a self contact problem!
  if (contactconditions.size() < 1) FOUR_C_THROW("Not enough contact conditions in discretization");
  if (contactconditions.size() == 1)
  {
    const std::string side = contactconditions[0]->parameters().get<std::string>("Side");
    FOUR_C_ASSERT_ALWAYS(side == "Selfcontact", "Not enough contact conditions in discretization");
  }

  // find all pairs of matching contact conditions
  // there is a maximum of (conditions / 2) groups
  std::vector<int> foundgroups;
  int numgroupsfound = 0;

  // maximum dof number in discretization
  // later we want to create NEW Lagrange multiplier degrees of
  // freedom, which of course must not overlap with existing displacement dofs
  int maxdof = discret.dof_row_map()->max_all_gid();

  // get input parameters
  auto stype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contactParams, "STRATEGY");
  auto wearLaw = Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(contactParams, "WEARLAW");
  auto wearType = Teuchos::getIntegralValue<Inpar::Wear::WearType>(contactParams, "WEARTYPE");
  auto constr_direction = Teuchos::getIntegralValue<CONTACT::ConstraintDirection>(
      contactParams, "CONSTRAINT_DIRECTIONS");
  auto frictionType = Teuchos::getIntegralValue<CONTACT::FrictionType>(contactParams, "FRICTION");
  auto adhesionType = Teuchos::getIntegralValue<CONTACT::AdhesionType>(contactParams, "ADHESION");
  const bool nurbs = contactParams.get<bool>("NURBS");
  auto algo = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(contactParams, "ALGORITHM");

  bool friplus = false;
  if ((wearLaw != Inpar::Wear::wear_none) ||
      (contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::tsi))
    friplus = true;

  // only for poro
  bool poromaster = false;
  bool poroslave = false;
  bool structmaster = false;
  bool structslave = false;
  int slavetype = -1;
  int mastertype = -1;  // 1 poro, 0 struct, -1 default
  bool isanyselfcontact = false;

  for (unsigned i = 0; i < contactconditions.size(); ++i)
  {
    // initialize vector for current group of conditions and temp condition
    std::vector<const Core::Conditions::Condition*> currentgroup;
    const Core::Conditions::Condition* tempcond = nullptr;

    // try to build contact group around this condition
    currentgroup.push_back(contactconditions[i]);
    const int groupid1 = currentgroup[0]->parameters().get<int>("InterfaceID");

    // In case of MultiScale contact this is the id of the interface's constitutive contact law
    auto maybe_contactconstitutivelawid =
        currentgroup[0]->parameters().get<std::optional<int>>("ConstitutiveLawID");
    auto contactconstitutivelawid = maybe_contactconstitutivelawid.value_or(-1);

    bool foundit = false;

    // only one surface per group is ok for self contact
    const std::string& side = contactconditions[i]->parameters().get<std::string>("Side");
    if (side == "Selfcontact") foundit = true;

    for (unsigned j = 0; j < contactconditions.size(); ++j)
    {
      if (j == i) continue;  // do not detect contactconditions[i] again
      tempcond = contactconditions[j];
      const int groupid2 = tempcond->parameters().get<int>("InterfaceID");
      if (groupid1 != groupid2) continue;  // not in the group
      foundit = true;                      // found a group entry
      currentgroup.push_back(tempcond);    // store it in currentgroup
    }

    // now we should have found a group of conds
    if (!foundit) FOUR_C_THROW("Cannot find matching contact condition for id {}", groupid1);

    // see whether we found this group before
    bool foundbefore = false;
    for (int j = 0; j < numgroupsfound; ++j)
    {
      if (groupid1 == foundgroups[j])
      {
        foundbefore = true;
        break;
      }
    }

    // if we have processed this group before, do nothing
    if (foundbefore) continue;

    // we have not found this group before, process it
    foundgroups.push_back(groupid1);
    ++numgroupsfound;

    // find out which sides are Master and Slave
    std::vector<bool> isslave;
    std::vector<bool> isself;
    CONTACT::Utils::get_master_slave_side_info(isslave, isself, currentgroup);
    for (const bool is : isself)
    {
      if (is)
      {
        isanyselfcontact = true;
        break;
      }
    }

    // find out which sides are initialized as In/Active and other initialization data
    std::vector<bool> isactive(currentgroup.size());
    bool Two_half_pass(false);
    bool Check_nonsmooth_selfcontactsurface(false);
    bool Searchele_AllProc(false);

    CONTACT::Utils::get_initialization_info(Two_half_pass, Check_nonsmooth_selfcontactsurface,
        Searchele_AllProc, isactive, isslave, isself, currentgroup);

    // create interface local parameter list (copy)
    Teuchos::ParameterList icparams = contactParams;

    // find out if interface-specific coefficients of friction are given
    if (frictionType == CONTACT::FrictionType::tresca ||
        frictionType == CONTACT::FrictionType::coulomb ||
        frictionType == CONTACT::FrictionType::stick)
    {
      // read interface COFs
      std::vector<double> frcoeff(currentgroup.size());

      for (unsigned j = 0; j < currentgroup.size(); ++j)
        frcoeff[j] = currentgroup[j]->parameters().get<double>("FrCoeffOrBound");

      // check consistency of interface COFs
      for (unsigned j = 1; j < currentgroup.size(); ++j)
      {
        if (frcoeff[j] != frcoeff[0])
          FOUR_C_THROW("Inconsistency in friction coefficients of interface {}", groupid1);
      }

      // check for infeasible value of COF
      if (frcoeff[0] < 0.0) FOUR_C_THROW("Negative FrCoeff / FrBound on interface {}", groupid1);

      // add COF locally to contact parameter list of this interface
      if (frictionType == CONTACT::FrictionType::tresca)
      {
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      else if (frictionType == CONTACT::FrictionType::coulomb)
      {
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      // dummy values for FRCOEFF and FRBOUND have to be set,
      // since entries are accessed regardless of the friction law
      else if (frictionType == CONTACT::FrictionType::stick)
      {
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(-1.0));
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
    }

    // find out if interface-specific coefficients of adhesion are given
    if (adhesionType == CONTACT::AdhesionType::bounded)
    {
      // read interface COFs
      std::vector<double> ad_bound(currentgroup.size());
      for (unsigned j = 0; j < currentgroup.size(); ++j)
        ad_bound[j] = currentgroup[j]->parameters().get<double>("AdhesionBound");

      // check consistency of interface COFs
      for (unsigned j = 1; j < currentgroup.size(); ++j)
      {
        if (ad_bound[j] != ad_bound[0])
          FOUR_C_THROW("Inconsistency in adhesion bounds of interface {}", groupid1);
      }

      // check for infeasible value of COF
      if (ad_bound[0] < 0.0) FOUR_C_THROW("Negative adhesion bound on interface {}", groupid1);

      // add COF locally to contact parameter list of this interface
      icparams.setEntry("ADHESION_BOUND", static_cast<Teuchos::ParameterEntry>(ad_bound[0]));
    }

    // add information to contact parameter list of this interface
    icparams.set<bool>("Two_half_pass", Two_half_pass);
    icparams.set<bool>("Check_nonsmooth_selfcontactsurface", Check_nonsmooth_selfcontactsurface);
    icparams.set<bool>("Searchele_AllProc", Searchele_AllProc);

    // Safety check for interface storage redundancy in case of self contact
    Inpar::Mortar::ExtendGhosting redundant =
        Teuchos::getIntegralValue<Inpar::Mortar::ExtendGhosting>(
            icparams.sublist("PARALLEL REDISTRIBUTION"), "GHOSTING_STRATEGY");
    if (isanyselfcontact == true && redundant != Inpar::Mortar::ExtendGhosting::redundant_all)
      FOUR_C_THROW("Manager: Self contact requires fully redundant slave and master storage");

    // Use factory to create an empty interface and store it in this Manager.
    std::shared_ptr<CONTACT::Interface> newinterface = STRATEGY::Factory::create_interface(
        groupid1, get_comm(), dim, icparams, isself[0], nullptr, contactconstitutivelawid);
    interfaces.push_back(newinterface);

    // Get the RCP to the last created interface
    std::shared_ptr<CONTACT::Interface> interface = interfaces.back();

    // note that the nodal ids are unique because they come from
    // one global problem discretization containing all nodes of the
    // contact interface.
    // We rely on this fact, therefore it is not possible to
    // do contact between two distinct discretizations here.

    // collect all initially active nodes
    std::vector<int> initialactive;

    //-------------------------------------------------- process nodes
    for (unsigned j = 0; j < currentgroup.size(); ++j)
    {
      // get all nodes and add them
      const std::vector<int>* nodeids = currentgroup[j]->get_nodes();
      if (!nodeids) FOUR_C_THROW("Condition does not have Node Ids");
      for (unsigned k = 0; k < (*nodeids).size(); ++k)
      {
        int gid = (*nodeids)[k];
        // do only nodes that I have in my discretization
        if (!discret.node_col_map()->my_gid(gid)) continue;
        Core::Nodes::Node* node = discret.g_node(gid);
        if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

        // store global IDs of initially active nodes
        if (isactive[j]) initialactive.push_back(gid);

        // find out if this node is initial active on another Condition
        // and do NOT overwrite this status then!
        bool foundinitialactive = false;
        if (!isactive[j])
        {
          for (unsigned k = 0; k < initialactive.size(); ++k)
          {
            if (gid == initialactive[k])
            {
              foundinitialactive = true;
              break;
            }
          }
        }

        // create Node object or FriNode object in the frictional case
        // for the boolean variable initactive we use isactive[j]+foundinitialactive,
        // as this is true for BOTH initial active nodes found for the first time
        // and found for the second, third, ... time!
        if (frictionType != CONTACT::FrictionType::none)
        {
          std::shared_ptr<CONTACT::FriNode> cnode =
              std::make_shared<CONTACT::FriNode>(node->id(), node->x(), node->owner(),
                  discret.dof(0, node), isslave[j], isactive[j] + foundinitialactive, friplus);
          //-------------------
          // get nurbs weight!
          if (nurbs) Mortar::Utils::prepare_nurbs_node(node, *cnode);

          // get edge and corner information:
          std::vector<const Core::Conditions::Condition*> contactcornercond;
          discret.get_condition("mrtrcorner", contactcornercond);
          for (unsigned j = 0; j < contactcornercond.size(); j++)
          {
            if (contactcornercond.at(j)->contains_node(node->id()))
            {
              cnode->set_on_corner() = true;
            }
          }
          std::vector<const Core::Conditions::Condition*> contactedgecond;
          discret.get_condition("mrtredge", contactedgecond);
          for (unsigned j = 0; j < contactedgecond.size(); j++)
          {
            if (contactedgecond.at(j)->contains_node(node->id()))
            {
              cnode->set_on_edge() = true;
            }
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<const Core::Conditions::Condition*> contactSymconditions;
          discret.get_condition("mrtrsym", contactSymconditions);

          for (unsigned l = 0; l < contactSymconditions.size(); l++)
            if (contactSymconditions.at(l)->contains_node(node->id()))
            {
              const std::vector<int>& onoff =
                  contactSymconditions.at(l)->parameters().get<std::vector<int>>("ONOFF");
              for (unsigned k = 0; k < onoff.size(); k++)
                if (onoff.at(k) == 1) cnode->dbc_dofs()[k] = true;
              if (stype == CONTACT::SolvingStrategy::lagmult &&
                  constr_direction != CONTACT::ConstraintDirection::xyz)
                FOUR_C_THROW(
                    "Contact symmetry with Lagrange multiplier method"
                    " only with contact constraints in xyz direction.\n"
                    "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
            }

          // note that we do not have to worry about double entries
          // as the add_node function can deal with this case!
          // the only problem would have occurred for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->add_node(cnode);
        }
        else
        {
          std::shared_ptr<CONTACT::Node> cnode =
              std::make_shared<CONTACT::Node>(node->id(), node->x(), node->owner(),
                  discret.dof(0, node), isslave[j], isactive[j] + foundinitialactive);
          //-------------------
          // get nurbs weight!
          if (nurbs)
          {
            Mortar::Utils::prepare_nurbs_node(node, *cnode);
          }

          // get edge and corner information:
          std::vector<const Core::Conditions::Condition*> contactcornercond;
          discret.get_condition("mrtrcorner", contactcornercond);
          for (unsigned j = 0; j < contactcornercond.size(); j++)
          {
            if (contactcornercond.at(j)->contains_node(node->id()))
            {
              cnode->set_on_corner() = true;
            }
          }
          std::vector<const Core::Conditions::Condition*> contactedgecond;
          discret.get_condition("mrtredge", contactedgecond);
          for (unsigned j = 0; j < contactedgecond.size(); j++)
          {
            if (contactedgecond.at(j)->contains_node(node->id()))
            {
              cnode->set_on_edge() = true;
            }
          }


          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<const Core::Conditions::Condition*> contactSymconditions;
          discret.get_condition("mrtrsym", contactSymconditions);

          for (unsigned l = 0; l < contactSymconditions.size(); l++)
            if (contactSymconditions.at(l)->contains_node(node->id()))
            {
              const std::vector<int>& onoff =
                  contactSymconditions.at(l)->parameters().get<std::vector<int>>("ONOFF");
              for (unsigned k = 0; k < onoff.size(); k++)
                if (onoff.at(k) == 1) cnode->dbc_dofs()[k] = true;
              if (stype == CONTACT::SolvingStrategy::lagmult &&
                  constr_direction != CONTACT::ConstraintDirection::xyz)
                FOUR_C_THROW(
                    "Contact symmetry with Lagrange multiplier method"
                    " only with contact constraints in xyz direction.\n"
                    "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
            }

          // note that we do not have to worry about double entries
          // as the add_node function can deal with this case!
          // the only problem would have occurred for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->add_node(cnode);
        }
      }
    }

    //----------------------------------------------- process elements
    int ggsize = 0;
    for (unsigned j = 0; j < currentgroup.size(); ++j)
    {
      // get elements from condition j of current group
      const std::map<int, std::shared_ptr<Core::Elements::Element>>& currele =
          currentgroup[j]->geometry();

      // elements in a boundary condition have a unique id
      // but ids are not unique among 2 distinct conditions
      // due to the way elements in conditions are build.
      // We therefore have to give the second, third,... set of elements
      // different ids. ids do not have to be continuous, we just add a large
      // enough number ggsize to all elements of cond2, cond3,... so they are
      // different from those in cond1!!!
      // note that elements in ele1/ele2 already are in column (overlapping) map
      int lsize = (int)currele.size();
      int gsize = 0;
      gsize = Core::Communication::sum_all(lsize, get_comm());

      for (const auto& ele : currele | std::views::values)
      {
        std::shared_ptr<CONTACT::Element> cele =
            std::make_shared<CONTACT::Element>(ele->id() + ggsize, ele->owner(), ele->shape(),
                ele->num_node(), ele->node_ids(), isslave[j], nurbs);

        if ((contactParams.get<CONTACT::Problemtype>("PROBTYPE") ==
                    CONTACT::Problemtype::poroelast ||
                contactParams.get<CONTACT::Problemtype>("PROBTYPE") ==
                    CONTACT::Problemtype::poroscatra) &&
            algo != Inpar::Mortar::algorithm_gpts)
          set_poro_parent_element(slavetype, mastertype, *cele, ele);

        if (algo == Inpar::Mortar::algorithm_gpts)
        {
          std::shared_ptr<Core::Elements::FaceElement> faceele =
              std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
          if (faceele == nullptr) FOUR_C_THROW("Cast to FaceElement failed!");
          if (faceele->parent_element() == nullptr) FOUR_C_THROW("face parent does not exist");
          if (discret.element_col_map()->lid(faceele->parent_element()->id()) == -1)
            FOUR_C_THROW("vol dis does not have parent ele");
          cele->set_parent_master_element(faceele->parent_element(), faceele->face_parent_number());
        }

        //------------------------------------------------------------------
        // get knotvector, normal factor and zero-size information for nurbs
        if (nurbs)
        {
          Mortar::Utils::prepare_nurbs_element(discret, ele, *cele, dim);
        }

        interface->add_element(cele);
      }  // for (fool=ele1.start(); fool != ele1.end(); ++fool)

      ggsize += gsize;  // update global element counter
    }

    /* Finalize the contact interface construction
     *
     * Always assign degrees of freedom here, because we need a valid column map for further contact
     * setup. This is an initial one time cost, that does not matter compared to the repeated
     * fill_complete calls due to dynamic redistribution.
     */
    if (CONTACT::Utils::use_safe_redistribute_and_ghosting(contactParams))
    {
      /* Finalize parallel layout of maps. Note: Do not redistribute here.
       *
       * Since this is the initial setup, we don't need redistribution here, just a proper extension
       * of the interface ghosting.
       */
      interface->update_parallel_layout_and_data_structures(false, true, maxdof, 0.0);
    }
    else
      interface->fill_complete(Global::Problem::instance()->discretization_map(),
          Global::Problem::instance()->binning_strategy_params(),
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type(), true, maxdof);

    if ((contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast ||
            contactParams.get<CONTACT::Problemtype>("PROBTYPE") ==
                CONTACT::Problemtype::poroscatra) &&
        algo != Inpar::Mortar::algorithm_gpts)
      find_poro_interface_types(
          poromaster, poroslave, structmaster, structslave, slavetype, mastertype);
  }
  if (Core::Communication::my_mpi_rank(get_comm()) == 0) std::cout << "done!" << std::endl;

  //**********************************************************************
  // create the solver strategy object and pass all necessary data to it
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Building contact strategy object............";
    fflush(stdout);
  }

  // build the correct data container
  std::shared_ptr<CONTACT::AbstractStrategyDataContainer> data_ptr =
      std::make_shared<CONTACT::AbstractStrategyDataContainer>();

  // create LagrangeStrategyWear for wear as non-distinct quantity
  if (stype == CONTACT::SolvingStrategy::lagmult && wearLaw != Inpar::Wear::wear_none &&
      (wearType == Inpar::Wear::wear_intstate || wearType == Inpar::Wear::wear_primvar))
  {
    strategy_ = std::make_shared<Wear::LagrangeStrategyWear>(data_ptr, discret.dof_row_map(),
        discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
  }
  else if (stype == CONTACT::SolvingStrategy::lagmult)
  {
    if (contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast ||
        contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroscatra)
    {
      strategy_ = std::make_shared<LagrangeStrategyPoro>(data_ptr, discret.dof_row_map(),
          discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof, poroslave,
          poromaster);
    }
    else if (contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::tsi)
    {
      strategy_ = std::make_shared<LagrangeStrategyTsi>(data_ptr, discret.dof_row_map(),
          discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
    }
    else
    {
      strategy_ = std::make_shared<LagrangeStrategy>(data_ptr, discret.dof_row_map(),
          discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
    }
  }
  else if (((stype == CONTACT::SolvingStrategy::penalty ||
                stype == CONTACT::SolvingStrategy::multiscale) &&
               algo != Inpar::Mortar::algorithm_gpts) ||
           stype == CONTACT::SolvingStrategy::uzawa)
  {
    strategy_ = std::make_shared<PenaltyStrategy>(data_ptr, discret.dof_row_map(),
        discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
  }
  else if (algo == Inpar::Mortar::algorithm_gpts && (stype == CONTACT::SolvingStrategy::nitsche ||
                                                        stype == CONTACT::SolvingStrategy::penalty))
  {
    if ((contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::poroelast ||
            contactParams.get<CONTACT::Problemtype>("PROBTYPE") ==
                CONTACT::Problemtype::poroscatra) &&
        stype == CONTACT::SolvingStrategy::nitsche)
    {
      strategy_ = std::make_shared<NitscheStrategyPoro>(data_ptr, discret.dof_row_map(),
          discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
    }
    else if (contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::fsi &&
             stype == CONTACT::SolvingStrategy::nitsche)
    {
      strategy_ = std::make_shared<NitscheStrategyFsi>(data_ptr, discret.dof_row_map(),
          discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
    }
    else if (contactParams.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::fpi &&
             stype == CONTACT::SolvingStrategy::nitsche)
    {
      strategy_ = std::make_shared<NitscheStrategyFpi>(data_ptr, discret.dof_row_map(),
          discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
    }
    else
    {
      strategy_ = std::make_shared<NitscheStrategy>(data_ptr, discret.dof_row_map(),
          discret.node_row_map(), contactParams, interfaces, dim, comm_, alphaf, maxdof);
    }
  }
  else
  {
    FOUR_C_THROW("Unrecognized contact strategy");
  }

  dynamic_cast<CONTACT::AbstractStrategy&>(*strategy_).setup(false, true);

  if (Core::Communication::my_mpi_rank(get_comm()) == 0) std::cout << "done!" << std::endl;
  //**********************************************************************

  // print friction information of interfaces
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    for (unsigned i = 0; i < interfaces.size(); ++i)
    {
      double checkfrcoeff = 0.0;
      if (frictionType == CONTACT::FrictionType::tresca)
      {
        checkfrcoeff = interfaces[i]->interface_params().get<double>("FRBOUND");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrBound (Tresca)  " << checkfrcoeff << std::endl;
      }
      else if (frictionType == CONTACT::FrictionType::coulomb)
      {
        checkfrcoeff = interfaces[i]->interface_params().get<double>("FRCOEFF");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrCoeff (Coulomb) " << checkfrcoeff << std::endl;
      }
    }
  }

  // print initial parallel redistribution
  if (Core::Communication::my_mpi_rank(get_comm()) == 0 &&
      Core::Communication::num_mpi_ranks(get_comm()) > 1)
    std::cout << "\nInitial parallel distribution of all contact interfaces:" << std::endl;
  for (auto& interface : interfaces) interface->print_parallel_distribution();

  // create binary search tree
  for (auto& interface : interfaces) interface->create_search_tree();

  return;
}


/*----------------------------------------------------------------------*
 |  read and check input parameters (public)                  popp 04/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Manager::read_and_check_input(Teuchos::ParameterList& cparams) const
{
  // read parameter lists from Global::Problem
  const Teuchos::ParameterList& mortar = Global::Problem::instance()->mortar_coupling_params();
  const Teuchos::ParameterList& contact = Global::Problem::instance()->contact_dynamic_params();
  const Teuchos::ParameterList& wearlist = Global::Problem::instance()->wear_params();
  const Teuchos::ParameterList& tsic = Global::Problem::instance()->tsi_contact_params();
  const Teuchos::ParameterList& stru = Global::Problem::instance()->structural_dynamic_params();

  // read Problem Type and Problem Dimension from Global::Problem
  const Core::ProblemType problemtype = Global::Problem::instance()->get_problem_type();
  Core::FE::ShapeFunctionType distype = Global::Problem::instance()->spatial_approximation_type();
  const int dim = Global::Problem::instance()->n_dim();

  // in case just System type system_condensed_lagmult
  if (Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") ==
      CONTACT::SystemType::condensed_lagmult)
    FOUR_C_THROW(
        "For Contact anyway just the lagrange multiplier can be condensed, choose SYSTEM = "
        "Condensed.");

  // *********************************************************************
  // invalid parallel strategies
  // *********************************************************************
  const Teuchos::ParameterList& mortarParallelRedistParams =
      mortar.sublist("PARALLEL REDISTRIBUTION");

  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none &&
      mortarParallelRedistParams.get<int>("MIN_ELEPROC") < 0)
    FOUR_C_THROW(
        "Minimum number of elements per processor for parallel redistribution must be >= 0");

  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") == Inpar::Mortar::ParallelRedist::redist_dynamic &&
      mortarParallelRedistParams.get<double>("MAX_BALANCE_EVAL_TIME") < 1.0)
    FOUR_C_THROW(
        "Maximum allowed value of load balance for dynamic parallel redistribution must be "
        ">= 1.0");

  if (problemtype == Core::ProblemType::tsi &&
      Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
    FOUR_C_THROW("Parallel redistribution not yet implemented for TSI problems");

  // *********************************************************************
  // adhesive contact
  // *********************************************************************
  if (Teuchos::getIntegralValue<CONTACT::AdhesionType>(contact, "ADHESION") !=
          CONTACT::AdhesionType::none and
      Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
          Inpar::Wear::wear_none)
    FOUR_C_THROW("Adhesion combined with wear not yet tested!");

  if (Teuchos::getIntegralValue<CONTACT::AdhesionType>(contact, "ADHESION") !=
          CONTACT::AdhesionType::none and
      Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none)
    FOUR_C_THROW("Adhesion combined with friction not yet tested!");

  // *********************************************************************
  // generally invalid combinations (nts/mortar)
  // *********************************************************************
  if ((Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::penalty ||
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::nitsche) &&
      contact.get<double>("PENALTYPARAM") <= 0.0)
    FOUR_C_THROW("Penalty parameter eps = 0, must be greater than 0");

  if ((Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::penalty ||
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::nitsche) &&
      Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    FOUR_C_THROW("Tangential penalty parameter eps = 0, must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      contact.get<double>("PENALTYPARAM") <= 0.0)
    FOUR_C_THROW("Penalty parameter eps = 0, must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    FOUR_C_THROW("Tangential penalty parameter eps = 0, must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      contact.get<int>("UZAWAMAXSTEPS") < 2)
    FOUR_C_THROW("Maximum number of Uzawa / Augmentation steps must be at least 2");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          CONTACT::SolvingStrategy::uzawa &&
      contact.get<double>("UZAWACONSTRTOL") <= 0.0)
    FOUR_C_THROW("Constraint tolerance for Uzawa / Augmentation scheme must be greater than 0");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      contact.get<double>("SEMI_SMOOTH_CT") == 0.0)
    FOUR_C_THROW("Parameter ct = 0, must be greater than 0 for frictional contact");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
          CONTACT::FrictionType::tresca &&
      dim == 3 &&
      Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
          CONTACT::SolvingStrategy::nitsche)
    FOUR_C_THROW("3D frictional contact with Tresca's law not yet implemented");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
          CONTACT::FrictionType::none &&
      not contact.get<bool>("SEMI_SMOOTH_NEWTON") && dim == 3)
    FOUR_C_THROW("3D frictional contact only implemented with Semi-smooth Newton");

  if (mortar.get<bool>("CROSSPOINTS") && dim == 3)
    FOUR_C_THROW("Crosspoints / edge node modification not yet implemented for 3D");

  if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
          CONTACT::FrictionType::tresca &&
      contact.get<bool>("FRLESS_FIRST"))
    // Hopefully coming soon, when Coulomb and Tresca are combined. Until then, throw error.
    FOUR_C_THROW("Frictionless first contact step with Tresca's law not yet implemented");

  // *********************************************************************
  // warnings
  // *********************************************************************
  if (mortar.get<double>("SEARCH_PARAM") == 0.0 &&
      Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << ("Warning: Contact search called without inflation of bounding volumes\n")
              << std::endl;

  if (Teuchos::getIntegralValue<Inpar::Wear::WearSide>(wearlist, "WEAR_SIDE") !=
      Inpar::Wear::wear_slave)
    std::cout << ("\n \n Warning: Contact with both-sided wear is still experimental !")
              << std::endl;


  // *********************************************************************
  //                       MORTAR-SPECIFIC CHECKS
  // *********************************************************************
  if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar, "ALGORITHM") ==
      Inpar::Mortar::algorithm_mortar)
  {
    // *********************************************************************
    // invalid parameter combinations
    // *********************************************************************
    if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::lagmult &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_petrovgalerkin)
      FOUR_C_THROW("Petrov-Galerkin approach for LM only with Lagrange multiplier strategy");

    if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
            CONTACT::SolvingStrategy::lagmult &&
        (Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
                Inpar::Mortar::shape_standard &&
            Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") !=
                Inpar::Mortar::lagmult_const) &&
        Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") ==
            CONTACT::SystemType::condensed)
      FOUR_C_THROW("Condensation of linear system only possible for dual Lagrange multipliers");

    if (Teuchos::getIntegralValue<Inpar::Mortar::ConsistentDualType>(
            mortar, "LM_DUAL_CONSISTENT") != Inpar::Mortar::consistent_none &&
        Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            CONTACT::SolvingStrategy::lagmult &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
            Inpar::Mortar::shape_standard)
      FOUR_C_THROW(
          "Consistent dual shape functions in boundary elements only for Lagrange "
          "multiplier strategy.");

    if (Teuchos::getIntegralValue<Inpar::Mortar::ConsistentDualType>(
            mortar, "LM_DUAL_CONSISTENT") != Inpar::Mortar::consistent_none &&
        Teuchos::getIntegralValue<Inpar::Mortar::IntType>(mortar, "INTTYPE") ==
            Inpar::Mortar::inttype_elements &&
        (Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_dual))
      FOUR_C_THROW(
          "Consistent dual shape functions in boundary elements not for purely "
          "element-based integration.");

    if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
            CONTACT::SolvingStrategy::nitsche &&
        Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar, "ALGORITHM") !=
            Inpar::Mortar::algorithm_gpts)
      FOUR_C_THROW("Nitsche contact only with GPTS algorithm.");


    // *********************************************************************
    // not (yet) implemented combinations
    // *********************************************************************

    if (mortar.get<bool>("CROSSPOINTS") && Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(
                                               mortar, "LM_QUAD") == Inpar::Mortar::lagmult_lin)
      FOUR_C_THROW("Crosspoints and linear LM interpolation for quadratic FE not yet compatible");

    // check for self contact
    bool self = false;
    {
      std::vector<const Core::Conditions::Condition*> contactCondition;
      discret().get_condition("Mortar", contactCondition);

      for (const auto& condition : contactCondition)
      {
        const std::string side = condition->parameters().get<std::string>("Side");
        if (side == "Selfcontact") self = true;
      }
    }

    if (self == true &&
        Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
            "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none)
      FOUR_C_THROW("Self contact and parallel redistribution not yet compatible");

    if (contact.get<bool>("INITCONTACTBYGAP") && contact.get<double>("INITCONTACTGAPVALUE") == 0.0)
      FOUR_C_THROW(
          "For initialization of init contact with gap, the INITCONTACTGAPVALUE is needed.");

    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none &&
        contact.get<bool>("FRLESS_FIRST"))
      FOUR_C_THROW("Frictionless first contact step with wear not yet implemented");

    if (problemtype != Core::ProblemType::ehl && contact.get<bool>("REGULARIZED_NORMAL_CONTACT"))
      FOUR_C_THROW("Regularized normal contact only implemented for EHL");

    // *********************************************************************
    // thermal-structure-interaction contact
    // *********************************************************************
    if (problemtype == Core::ProblemType::tsi &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_standard)
      FOUR_C_THROW("Thermal contact only for dual shape functions");

    if (problemtype == Core::ProblemType::tsi &&
        Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") !=
            CONTACT::SystemType::condensed)
      FOUR_C_THROW("Thermal contact only for dual shape functions with condensed system");

    // no nodal scaling in for thermal-structure-interaction
    if (problemtype == Core::ProblemType::tsi &&
        tsic.get<double>("TEMP_DAMAGE") <= tsic.get<double>("TEMP_REF"))
      FOUR_C_THROW("damage temperature must be greater than reference temperature");

    // *********************************************************************
    // contact with wear
    // *********************************************************************
    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") ==
            Inpar::Wear::wear_none &&
        wearlist.get<double>("WEARCOEFF") != 0.0)
      FOUR_C_THROW("Wear coefficient only necessary in the context of wear.");

    if (problemtype == Core::ProblemType::structure and
        Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none and
        Teuchos::getIntegralValue<Inpar::Wear::WearTimInt>(wearlist, "WEARTIMINT") !=
            Inpar::Wear::wear_expl)
      FOUR_C_THROW(
          "Wear calculation for pure structure problems only with explicit internal state "
          "variable approach reasonable!");

    if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
            CONTACT::FrictionType::none &&
        Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none)
      FOUR_C_THROW("Wear models only applicable to frictional contact.");

    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none &&
        wearlist.get<double>("WEARCOEFF") <= 0.0)
      FOUR_C_THROW("No valid wear coefficient provided, must be equal or greater 0.0");

    if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") ==
            CONTACT::FrictionType::tresca &&
        Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
            Inpar::Wear::wear_none)
      FOUR_C_THROW("Wear only for Coulomb friction!");

    // *********************************************************************
    // 3D quadratic mortar (choice of interpolation and testing fcts.)
    // *********************************************************************
    if (Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(mortar, "LM_QUAD") ==
            Inpar::Mortar::lagmult_pwlin &&
        Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            Inpar::Mortar::shape_dual)
      FOUR_C_THROW(
          "No piecewise linear approach (for LM) implemented for quadratic contact with "
          "DUAL shape fct.");

    // *********************************************************************
    // poroelastic contact
    // *********************************************************************
    if (problemtype == Core::ProblemType::poroelast ||
        problemtype == Core::ProblemType::poroscatra || problemtype == Core::ProblemType::fpsi ||
        problemtype == Core::ProblemType::fpsi_xfem)
    {
      const Teuchos::ParameterList& porodyn =
          Global::Problem::instance()->poroelast_dynamic_params();
      if ((Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
                  Inpar::Mortar::shape_dual &&
              Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
                  Inpar::Mortar::shape_petrovgalerkin) &&
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::lagmult)
        FOUR_C_THROW("POROCONTACT: Only dual and petrovgalerkin shape functions implemented yet!");

      if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
              "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none &&
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::lagmult)
        // Since we use Pointers to Parent Elements, which are not copied to other procs!
        FOUR_C_THROW("POROCONTACT: Parallel Redistribution not implemented yet!");

      if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
              CONTACT::SolvingStrategy::lagmult &&
          porodyn.get<bool>("CONTACT_NO_PENETRATION"))
        FOUR_C_THROW("POROCONTACT: Use Lagrangean Strategy for poro contact!");

      if (Teuchos::getIntegralValue<CONTACT::FrictionType>(contact, "FRICTION") !=
              CONTACT::FrictionType::none &&
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::lagmult)
        FOUR_C_THROW("POROCONTACT: is_friction for poro contact not implemented!");

      if (Teuchos::getIntegralValue<CONTACT::SystemType>(contact, "SYSTEM") !=
              CONTACT::SystemType::condensed &&
          Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              CONTACT::SolvingStrategy::lagmult)
        FOUR_C_THROW("POROCONTACT: System has to be condensed for poro contact!");

      if ((dim != 3) && (dim != 2))
      {
        const Teuchos::ParameterList& porodyn =
            Global::Problem::instance()->poroelast_dynamic_params();
        if (porodyn.get<bool>("CONTACT_NO_PENETRATION"))
          FOUR_C_THROW("POROCONTACT: PoroContact with no penetration just tested for 3d (and 2d)!");
      }
    }

    // *********************************************************************
    // element-based vs. segment-based mortar integration
    // *********************************************************************
    auto inttype = Teuchos::getIntegralValue<Inpar::Mortar::IntType>(mortar, "INTTYPE");

    if (inttype == Inpar::Mortar::inttype_elements && mortar.get<int>("NUMGP_PER_DIM") <= 0)
      FOUR_C_THROW("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");

    if (inttype == Inpar::Mortar::inttype_elements_BS && mortar.get<int>("NUMGP_PER_DIM") <= 0)
      FOUR_C_THROW(
          "Invalid Gauss point number NUMGP_PER_DIM for element-based integration with "
          "boundary segmentation."
          "\nPlease note that the value you have to provide only applies to the element-based "
          "integration"
          "\ndomain, while pre-defined default values will be used in the segment-based boundary "
          "domain.");

    if ((inttype == Inpar::Mortar::inttype_elements ||
            inttype == Inpar::Mortar::inttype_elements_BS) &&
        mortar.get<int>("NUMGP_PER_DIM") <= 1)
      FOUR_C_THROW("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");
  }  // END MORTAR CHECKS

  // *********************************************************************
  //                       NTS-SPECIFIC CHECKS
  // *********************************************************************
  else if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar, "ALGORITHM") ==
           Inpar::Mortar::algorithm_nts)
  {
    if (problemtype == Core::ProblemType::poroelast or problemtype == Core::ProblemType::fpsi or
        problemtype == Core::ProblemType::tsi)
      FOUR_C_THROW("NTS only for problem type: structure");
  }  // END NTS CHECKS

  // *********************************************************************
  //                       GPTS-SPECIFIC CHECKS
  // *********************************************************************
  else if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(mortar, "ALGORITHM") ==
           Inpar::Mortar::algorithm_gpts)
  {
    const_cast<Teuchos::ParameterList&>(Global::Problem::instance()->contact_dynamic_params())
        .set("SYSTEM", CONTACT::SystemType::none);

    if (contact.get<double>("PENALTYPARAM") <= 0.0)
      FOUR_C_THROW("Penalty parameter eps = 0, must be greater than 0");

    if (problemtype != Core::ProblemType::structure &&
        problemtype != Core::ProblemType::poroelast && problemtype != Core::ProblemType::fsi_xfem &&
        problemtype != Core::ProblemType::fpsi_xfem)
      FOUR_C_THROW(
          "GPTS algorithm only tested for structural, FSI-CutFEM, FPSI-CutFEM, and "
          "poroelastic problems");

    if (Teuchos::getIntegralValue<Inpar::Wear::WearLaw>(wearlist, "WEARLAW") !=
        Inpar::Wear::wear_none)
      FOUR_C_THROW("GPTS algorithm not implemented for wear");

  }  // END GPTS CHECKS

  // *********************************************************************
  // store contents of BOTH ParameterLists in local parameter list
  // *********************************************************************
  cparams.setParameters(mortar);
  cparams.setParameters(contact);
  cparams.setParameters(wearlist);
  cparams.setParameters(tsic);
  if (problemtype == Core::ProblemType::tsi)
    cparams.set<double>(
        "TIMESTEP", Global::Problem::instance()->tsi_dynamic_params().get<double>("TIMESTEP"));
  else if (problemtype != Core::ProblemType::structure)
  {
    // rauch 01/16
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      std::cout << "\n \n  Warning: CONTACT::Manager::read_and_check_input() reads TIMESTEP = "
                << stru.get<double>("TIMESTEP") << " from --STRUCTURAL DYNAMIC \n"
                << std::endl;
    cparams.set<double>("TIMESTEP", stru.get<double>("TIMESTEP"));
  }
  else
    cparams.set<double>("TIMESTEP", stru.get<double>("TIMESTEP"));

  // *********************************************************************
  // NURBS contact
  // *********************************************************************
  switch (distype)
  {
    case Core::FE::ShapeFunctionType::nurbs:
    {
      cparams.set<bool>("NURBS", true);
      break;
    }
    default:
    {
      cparams.set<bool>("NURBS", false);
      break;
    }
  }

  // *********************************************************************
  cparams.setName("CONTACT DYNAMIC / MORTAR COUPLING");

  // store relevant problem types
  if (problemtype == Core::ProblemType::structure)
  {
    cparams.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::structure);
  }
  else if (problemtype == Core::ProblemType::tsi)
  {
    cparams.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::tsi);
  }
  else if (problemtype == Core::ProblemType::poroelast or problemtype == Core::ProblemType::fpsi or
           problemtype == Core::ProblemType::poroscatra)
  {
    const Teuchos::ParameterList& porodyn = Global::Problem::instance()->poroelast_dynamic_params();
    if (problemtype == Core::ProblemType::poroelast or problemtype == Core::ProblemType::fpsi)
      cparams.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::poroelast);
    else if (problemtype == Core::ProblemType::poroscatra)
      cparams.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::poroscatra);
    // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
    double porotimefac =
        1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
    cparams.set<double>("porotimefac", porotimefac);
    cparams.set<bool>("CONTACT_NO_PENETRATION",
        porodyn.get<bool>("CONTACT_NO_PENETRATION"));  // used in the integrator
  }
  else if (problemtype == Core::ProblemType::fsi_xfem)
  {
    cparams.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::fsi);
  }
  else if (problemtype == Core::ProblemType::fpsi_xfem)
  {
    const Teuchos::ParameterList& porodyn = Global::Problem::instance()->poroelast_dynamic_params();
    cparams.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::fpi);
    // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
    double porotimefac =
        1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
    cparams.set<double>("porotimefac", porotimefac);
    cparams.set<bool>("CONTACT_NO_PENETRATION",
        porodyn.get<bool>("CONTACT_NO_PENETRATION"));  // used in the integrator
  }
  else
  {
    cparams.set<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::other);
  }

  // no parallel redistribution in the serial case
  if (Core::Communication::num_mpi_ranks(get_comm()) == 1)
    cparams.sublist("PARALLEL REDISTRIBUTION")
        .set<Inpar::Mortar::ParallelRedist>(
            "PARALLEL_REDIST", Inpar::Mortar::ParallelRedist::redist_none);

  // set dimension
  cparams.set<int>("DIMENSION", dim);
  return true;
}

/*----------------------------------------------------------------------*
 |  write restart information for contact (public)            popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::write_restart(Core::IO::DiscretizationWriter& output, bool forcedrestart)
{
  // clear cache of maps due to varying vector size
  output.clear_map_cache();

  // quantities to be written for restart
  std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>> restart_vectors;

  // quantities to be written for restart
  get_strategy().do_write_restart(restart_vectors, forcedrestart);

  if (get_strategy().lagrange_multiplier_old() != nullptr)
    output.write_vector("lagrmultold", get_strategy().lagrange_multiplier_old());

  // write all vectors specified by used strategy
  for (std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>::const_iterator p =
           restart_vectors.begin();
      p != restart_vectors.end(); ++p)
    output.write_vector(p->first, p->second);

  return;
}

/*----------------------------------------------------------------------*
 |  read restart information for contact (public)             popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::read_restart(Core::IO::DiscretizationReader& reader,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::Vector<double>> zero)
{
  // If Parent Elements are required, we need to reconnect them before contact restart!
  auto atype =
      Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(get_strategy().params(), "ALGORITHM");
  if (atype == Inpar::Mortar::algorithm_gpts)
  {
    for (unsigned i = 0;
        i < dynamic_cast<CONTACT::AbstractStrategy&>(get_strategy()).contact_interfaces().size();
        ++i)
      dynamic_cast<CONTACT::AbstractStrategy&>(get_strategy())
          .contact_interfaces()[i]
          ->create_volume_ghosting(Global::Problem::instance()->discretization_map());
  }

  // If Parent Elements are required, we need to reconnect them before contact restart!
  if ((get_strategy().params().get<CONTACT::Problemtype>("PROBTYPE") ==
              CONTACT::Problemtype::poroelast ||
          get_strategy().params().get<CONTACT::Problemtype>("PROBTYPE") ==
              CONTACT::Problemtype::poroscatra) ||
      get_strategy().params().get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::fpi)
    reconnect_parent_elements();

  // this is contact, thus we need the displacement state for restart
  // let strategy object do all the work
  get_strategy().do_read_restart(reader, dis);

  return;
}

/*----------------------------------------------------------------------*
 |  write interface tractions for postprocessing (public)     popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::postprocess_quantities(Core::IO::DiscretizationWriter& output)
{
  if (get_strategy().is_nitsche()) return;

  // *********************************************************************
  // active contact set and slip set
  // *********************************************************************

  // evaluate active set and slip set
  Core::LinAlg::Vector<double> activeset(*get_strategy().active_row_nodes());
  activeset.put_scalar(1.0);
  if (get_strategy().is_friction())
  {
    Core::LinAlg::Vector<double> slipset(*get_strategy().slip_row_nodes());
    slipset.put_scalar(1.0);
    Core::LinAlg::Vector<double> slipsetexp(*get_strategy().active_row_nodes());
    Core::LinAlg::export_to(slipset, slipsetexp);
    activeset.update(1.0, slipsetexp, 1.0);
  }

  // export to problem node row map
  std::shared_ptr<Core::LinAlg::Map> problemnodes = get_strategy().problem_nodes();
  std::shared_ptr<Core::LinAlg::Vector<double>> activesetexp =
      std::make_shared<Core::LinAlg::Vector<double>>(*problemnodes);
  Core::LinAlg::export_to(activeset, *activesetexp);

  if (get_strategy().wear_both_discrete())
  {
    Core::LinAlg::Vector<double> mactiveset(*get_strategy().master_active_nodes());
    mactiveset.put_scalar(1.0);
    Core::LinAlg::Vector<double> slipset(*get_strategy().master_slip_nodes());
    slipset.put_scalar(1.0);
    Core::LinAlg::Vector<double> slipsetexp(*get_strategy().master_active_nodes());
    Core::LinAlg::export_to(slipset, slipsetexp);
    mactiveset.update(1.0, slipsetexp, 1.0);

    Core::LinAlg::Vector<double> mactivesetexp(*problemnodes);
    Core::LinAlg::export_to(mactiveset, mactivesetexp);
    activesetexp->update(1.0, mactivesetexp, 1.0);
  }

  output.write_vector("activeset", activesetexp);

  // *********************************************************************
  //  weighted gap
  // *********************************************************************
  // export to problem dof row map
  std::shared_ptr<Core::LinAlg::Map> gapnodes = get_strategy().problem_nodes();
  std::shared_ptr<const Core::LinAlg::Vector<double>> gaps =
      std::dynamic_pointer_cast<CONTACT::AbstractStrategy>(strategy_)->contact_wgap();
  if (gaps != nullptr)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> gapsexp =
        std::make_shared<Core::LinAlg::Vector<double>>(*gapnodes);
    Core::LinAlg::export_to(*gaps, *gapsexp);

    output.write_vector("gap", gapsexp);
  }

  // *********************************************************************
  // contact tractions
  // *********************************************************************

  // evaluate contact tractions
  get_strategy().compute_contact_stresses();

  // export to problem dof row map
  std::shared_ptr<Core::LinAlg::Map> problemdofs = get_strategy().problem_dofs();

  // normal direction
  std::shared_ptr<const Core::LinAlg::Vector<double>> normalstresses =
      get_strategy().contact_normal_stress();
  std::shared_ptr<Core::LinAlg::Vector<double>> normalstressesexp =
      std::make_shared<Core::LinAlg::Vector<double>>(*problemdofs);
  Core::LinAlg::export_to(*normalstresses, *normalstressesexp);

  // tangential plane
  std::shared_ptr<const Core::LinAlg::Vector<double>> tangentialstresses =
      get_strategy().contact_tangential_stress();
  std::shared_ptr<Core::LinAlg::Vector<double>> tangentialstressesexp =
      std::make_shared<Core::LinAlg::Vector<double>>(*problemdofs);
  Core::LinAlg::export_to(*tangentialstresses, *tangentialstressesexp);

  // write to output
  // contact tractions in normal and tangential direction
  output.write_vector("norcontactstress", normalstressesexp);
  output.write_vector("tancontactstress", tangentialstressesexp);

  if (get_strategy().contact_normal_force() != nullptr)
  {
    // normal direction
    std::shared_ptr<const Core::LinAlg::Vector<double>> normalforce =
        get_strategy().contact_normal_force();
    std::shared_ptr<Core::LinAlg::Vector<double>> normalforceexp =
        std::make_shared<Core::LinAlg::Vector<double>>(*problemdofs);
    Core::LinAlg::export_to(*normalforce, *normalforceexp);

    // tangential plane
    std::shared_ptr<const Core::LinAlg::Vector<double>> tangentialforce =
        get_strategy().contact_tangential_force();
    std::shared_ptr<Core::LinAlg::Vector<double>> tangentialforceexp =
        std::make_shared<Core::LinAlg::Vector<double>>(*problemdofs);
    Core::LinAlg::export_to(*tangentialforce, *tangentialforceexp);

    // write to output
    // contact tractions in normal and tangential direction
    output.write_vector("norslaveforce", normalforceexp);
    output.write_vector("tanslaveforce", tangentialforceexp);
  }

  // *********************************************************************
  // wear with internal state variable approach
  // *********************************************************************
  bool wwear = get_strategy().weighted_wear();
  if (wwear)
  {
    // ***************************************************************************
    // we do not compute the non-weighted wear here. we just write    farah 06/13
    // the output. the non-weighted wear will be used as dirichlet-b.
    // for the ale problem. n.w.wear will be called in stru_ale_algorithm.cpp
    // and computed in GetStrategy().OutputWear();
    // ***************************************************************************

    // evaluate wear (not weighted)
    get_strategy().output_wear();

    // write output
    std::shared_ptr<const Core::LinAlg::Vector<double>> wearoutput = get_strategy().contact_wear();
    std::shared_ptr<Core::LinAlg::Vector<double>> wearoutputexp =
        std::make_shared<Core::LinAlg::Vector<double>>(*problemdofs);
    Core::LinAlg::export_to(*wearoutput, *wearoutputexp);
    output.write_vector("wear", wearoutputexp);
    get_strategy().reset_wear();
  }

  // *********************************************************************
  // poro contact
  // *********************************************************************
  bool poro = get_strategy().has_poro_no_penetration();
  if (poro)
  {
    // output of poro no penetration lagrange multiplier!
    CONTACT::LagrangeStrategyPoro& costrategy =
        dynamic_cast<CONTACT::LagrangeStrategyPoro&>(get_strategy());
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaout = costrategy.lambda_no_pen();
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdaoutexp =
        std::make_shared<Core::LinAlg::Vector<double>>(*problemdofs);
    Core::LinAlg::export_to(*lambdaout, *lambdaoutexp);
    output.write_vector("poronopen_lambda", lambdaoutexp);
  }
}

/*-----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Manager::postprocess_quantities_per_interface(
    std::shared_ptr<Teuchos::ParameterList> outputParams)
{
  get_strategy().compute_contact_stresses();
  get_strategy().postprocess_quantities_per_interface(outputParams);
}

/*----------------------------------------------------------------------------------------------*
 |  Reconnect Contact Element -- Parent Element Pointers (required for restart)       ager 04/16|
 *---------------------------------------------------------------------------------------------*/
void CONTACT::Manager::reconnect_parent_elements()
{
  {
    const Core::LinAlg::Map* elecolmap = discret_.element_col_map();

    CONTACT::AbstractStrategy& strategy = dynamic_cast<CONTACT::AbstractStrategy&>(get_strategy());

    for (auto& interface : strategy.contact_interfaces())
    {
      const Core::LinAlg::Map* ielecolmap = interface->discret().element_col_map();

      for (int i = 0; i < ielecolmap->num_my_elements(); ++i)
      {
        int gid = ielecolmap->gid(i);

        Core::Elements::Element* ele = interface->discret().g_element(gid);
        if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
        Core::Elements::FaceElement* faceele = dynamic_cast<Core::Elements::FaceElement*>(ele);

        int volgid = faceele->parent_element_id();
        if (elecolmap->lid(volgid) == -1)  // Volume discretization has not Element
          FOUR_C_THROW(
              "Manager::reconnect_parent_elements: Element {} does not exist on this Proc!",
              volgid);

        Core::Elements::Element* vele = discret_.g_element(volgid);
        if (!vele) FOUR_C_THROW("Cannot find element with gid %", volgid);

        faceele->set_parent_master_element(vele, faceele->face_parent_number());
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Set Parent Elements for Poro Face Elements                ager 11/15|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::set_poro_parent_element(int& slavetype, int& mastertype,
    CONTACT::Element& cele, std::shared_ptr<Core::Elements::Element> ele)
{
  // ints to communicate decision over poro bools between processors on every interface
  // safety check - because there may not be mixed interfaces and structural slave elements
  // slavetype ... 1 poro, 0 struct, -1 default
  // mastertype ... 1 poro, 0 struct, -1 default
  std::shared_ptr<Core::Elements::FaceElement> faceele =
      std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
  if (faceele == nullptr) FOUR_C_THROW("Cast to FaceElement failed!");
  cele.phys_type() = Mortar::Element::other;
  std::vector<const Core::Conditions::Condition*> porocondvec;
  discret_.get_condition("PoroCoupling", porocondvec);
  if (!cele.is_slave())  // treat an element as a master element if it is no slave element
  {
    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->geometry().begin();
          eleitergeometry != porocondvec[i]->geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->id() == eleitergeometry->second->id())
        {
          if (mastertype == 0)
            FOUR_C_THROW(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          cele.phys_type() = Mortar::Element::poro;
          mastertype = 1;
          break;
        }
      }
    }
    if (cele.phys_type() == Mortar::Element::other)
    {
      if (mastertype == 1)
        FOUR_C_THROW(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele.phys_type() = Mortar::Element::structure;
      mastertype = 0;
    }
  }
  else if (cele.is_slave())  // treat an element as slave element if it is one
  {
    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->geometry().begin();
          eleitergeometry != porocondvec[i]->geometry().end(); ++eleitergeometry)
      {
        if (faceele->parent_element()->id() == eleitergeometry->second->id())
        {
          if (slavetype == 0)
            FOUR_C_THROW(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          cele.phys_type() = Mortar::Element::poro;
          slavetype = 1;
          break;
        }
      }
    }
    if (cele.phys_type() == Mortar::Element::other)
    {
      if (slavetype == 1)
        FOUR_C_THROW(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele.phys_type() = Mortar::Element::structure;
      slavetype = 0;
    }
  }
  // store information about parent for porous contact (required for calculation of deformation
  // gradient!) in every contact element although only really needed for phystype poro
  cele.set_parent_master_element(faceele->parent_element(), faceele->face_parent_number());
  return;
}

/*----------------------------------------------------------------------*
 |  Find Physical Type (Poro or Structure) of Poro Interface  ager 11/15|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::find_poro_interface_types(bool& poromaster, bool& poroslave,
    bool& structmaster, bool& structslave, int& slavetype, int& mastertype) const
{
  // find poro and structure elements when a poro coupling condition is applied on an element
  // and restrict to pure poroelastic or pure structural interfaces' sides.
  //(only poro slave elements AND (only poro master elements or only structure master elements)
  // Tell the contact element which physical type it is to extract PhysType in contact integrator
  // bools to decide which side is structural and which side is poroelastic to manage all 4
  // constellations
  // s-s, p-s, s-p, p-p
  // wait for all processors to determine if they have poro or structural master or slave elements
  Core::Communication::barrier(comm_);
  std::vector<int> slaveTypeList(Core::Communication::num_mpi_ranks(comm_));
  std::vector<int> masterTypeList(Core::Communication::num_mpi_ranks(comm_));
  Core::Communication::gather_all(&slavetype, slaveTypeList.data(), 1, comm_);
  Core::Communication::gather_all(&mastertype, masterTypeList.data(), 1, comm_);
  Core::Communication::barrier(comm_);

  for (int i = 0; i < Core::Communication::num_mpi_ranks(comm_); ++i)
  {
    switch (slaveTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structslave)
          FOUR_C_THROW(
              "struct and poro slave elements in the same problem - no mixed interface "
              "constellations supported");
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poroslave = true;
        break;
      case 0:
        if (poroslave)
          FOUR_C_THROW(
              "struct and poro slave elements in the same problem - no mixed interface "
              "constellations supported");
        structslave = true;
        break;
      default:
        FOUR_C_THROW("this cannot happen");
        break;
    }
  }

  for (int i = 0; i < Core::Communication::num_mpi_ranks(comm_); ++i)
  {
    switch (masterTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structmaster)
          FOUR_C_THROW(
              "struct and poro master elements in the same problem - no mixed interface "
              "constellations supported");
        // adjust FOUR_C_THROW text, when more than one interface is supported
        poromaster = true;
        break;
      case 0:
        if (poromaster)
          FOUR_C_THROW(
              "struct and poro master elements in the same problem - no mixed interface "
              "constellations supported");
        structmaster = true;
        break;
      default:
        FOUR_C_THROW("this cannot happen");
        break;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
