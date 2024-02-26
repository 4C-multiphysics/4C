/*----------------------------------------------------------------------*/
/*! \file
\brief BACI implementation of main class to control all contact

\level 1


*/
/*----------------------------------------------------------------------*/

#include "baci_contact_manager.hpp"

#include "baci_contact_aug_interface.hpp"
#include "baci_contact_aug_strategy.hpp"
#include "baci_contact_element.hpp"
#include "baci_contact_friction_node.hpp"
#include "baci_contact_interface.hpp"
#include "baci_contact_lagrange_strategy.hpp"
#include "baci_contact_lagrange_strategy_poro.hpp"
#include "baci_contact_lagrange_strategy_tsi.hpp"
#include "baci_contact_lagrange_strategy_wear.hpp"
#include "baci_contact_nitsche_strategy.hpp"
#include "baci_contact_nitsche_strategy_fpi.hpp"
#include "baci_contact_nitsche_strategy_fsi.hpp"
#include "baci_contact_nitsche_strategy_poro.hpp"
#include "baci_contact_node.hpp"
#include "baci_contact_penalty_strategy.hpp"
#include "baci_contact_strategy_factory.hpp"
#include "baci_contact_utils.hpp"
#include "baci_contact_utils_parallel.hpp"
#include "baci_contact_wear_interface.hpp"
#include "baci_global_data.hpp"
#include "baci_inpar_contact.hpp"
#include "baci_inpar_mortar.hpp"
#include "baci_inpar_wear.hpp"
#include "baci_io.hpp"
#include "baci_io_control.hpp"
#include "baci_linalg_utils_sparse_algebra_manipulation.hpp"
#include "baci_mortar_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 03/08|
 *----------------------------------------------------------------------*/
CONTACT::Manager::Manager(DRT::Discretization& discret, double alphaf)
    : MORTAR::ManagerBase(), discret_(discret)
{
  // overwrite base class communicator
  comm_ = Teuchos::rcp(Discret().Comm().Clone());

  // create some local variables (later to be stored in strategy)
  const int dim = GLOBAL::Problem::Instance()->NDim();
  if (dim != 2 && dim != 3) dserror("Contact problem must be 2D or 3D");
  std::vector<Teuchos::RCP<CONTACT::Interface>> interfaces;
  Teuchos::ParameterList contactParams;

  // read and check contact input parameters
  if (Comm().MyPID() == 0) std::cout << "Checking contact input parameters..........." << std::endl;

  ReadAndCheckInput(contactParams);
  if (Comm().MyPID() == 0) std::cout << "done!" << std::endl;

  // check for FillComplete of discretization
  if (!Discret().Filled()) dserror("Discretization is not fillcomplete");

  // let's check for contact boundary conditions in the discretization and and detect groups of
  // matching conditions. For each group, create a contact interface and store it.
  if (Comm().MyPID() == 0) std::cout << "Building contact interface(s)..............." << std::endl;

  // Vector that contains solid-to-solid and beam-to-solid contact pairs
  std::vector<DRT::Condition*> beamandsolidcontactconditions(0);
  Discret().GetCondition("Contact", beamandsolidcontactconditions);

  // Vector that solely contains solid-to-solid contact pairs
  std::vector<DRT::Condition*> contactconditions(0);

  // Sort out beam-to-solid contact pairs, since these are treated in the beam3contact framework
  for (const auto& beamSolidCondition : beamandsolidcontactconditions)
  {
    if (*beamSolidCondition->Get<std::string>("Application") != "Beamtosolidcontact")
      contactconditions.push_back(beamSolidCondition);
  }

  // there must be more than one contact condition
  // unless we have a self contact problem!
  if (contactconditions.size() < 1) dserror("Not enough contact conditions in discretization");
  if (contactconditions.size() == 1)
  {
    const std::string* side = contactconditions[0]->Get<std::string>("Side");
    if (*side != "Selfcontact") dserror("Not enough contact conditions in discretization");
  }

  // find all pairs of matching contact conditions
  // there is a maximum of (conditions / 2) groups
  std::vector<int> foundgroups(0);
  int numgroupsfound = 0;

  // maximum dof number in discretization
  // later we want to create NEW Lagrange multiplier degrees of
  // freedom, which of course must not overlap with existing displacement dofs
  int maxdof = Discret().DofRowMap()->MaxAllGID();

  // get input parameters
  INPAR::CONTACT::SolvingStrategy stype =
      INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contactParams, "STRATEGY");
  INPAR::WEAR::WearLaw wearLaw =
      INPUT::IntegralValue<INPAR::WEAR::WearLaw>(contactParams, "WEARLAW");
  INPAR::WEAR::WearType wearType =
      INPUT::IntegralValue<INPAR::WEAR::WearType>(contactParams, "WEARTYPE");
  INPAR::CONTACT::ConstraintDirection constr_direction =
      INPUT::IntegralValue<INPAR::CONTACT::ConstraintDirection>(
          contactParams, "CONSTRAINT_DIRECTIONS");
  INPAR::CONTACT::FrictionType frictionType =
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contactParams, "FRICTION");
  INPAR::CONTACT::AdhesionType adhesionType =
      INPUT::IntegralValue<INPAR::CONTACT::AdhesionType>(contactParams, "ADHESION");
  const bool nurbs = contactParams.get<bool>("NURBS");
  INPAR::MORTAR::AlgorithmType algo =
      INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(contactParams, "ALGORITHM");

  bool friplus = false;
  if ((wearLaw != INPAR::WEAR::wear_none) ||
      (contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::tsi))
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
    std::vector<DRT::Condition*> currentgroup(0);
    DRT::Condition* tempcond = nullptr;

    // try to build contact group around this condition
    currentgroup.push_back(contactconditions[i]);
    const int groupid1 = *currentgroup[0]->Get<int>("Interface ID");

    // In case of MultiScale contact this is the id of the interface's constitutive contact law
    int contactconstitutivelawid = *currentgroup[0]->Get<int>("ConstitutiveLawID");

    bool foundit = false;

    // only one surface per group is ok for self contact
    const std::string* side = contactconditions[i]->Get<std::string>("Side");
    if (*side == "Selfcontact") foundit = true;

    for (unsigned j = 0; j < contactconditions.size(); ++j)
    {
      if (j == i) continue;  // do not detect contactconditions[i] again
      tempcond = contactconditions[j];
      const int groupid2 = *tempcond->Get<int>("Interface ID");
      if (groupid1 != groupid2) continue;  // not in the group
      foundit = true;                      // found a group entry
      currentgroup.push_back(tempcond);    // store it in currentgroup
    }

    // now we should have found a group of conds
    if (!foundit) dserror("Cannot find matching contact condition for id %d", groupid1);

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
    std::vector<bool> isslave(0);
    std::vector<bool> isself(0);
    CONTACT::UTILS::GetMasterSlaveSideInfo(isslave, isself, currentgroup);
    for (const bool is : isself)
    {
      if (is)
      {
        isanyselfcontact = true;
        break;
      }
    }

    // find out which sides are initialized as In/Active and other initalization data
    std::vector<bool> isactive(currentgroup.size());
    bool Two_half_pass(false);
    bool Check_nonsmooth_selfcontactsurface(false);
    bool Searchele_AllProc(false);

    CONTACT::UTILS::GetInitializationInfo(Two_half_pass, Check_nonsmooth_selfcontactsurface,
        Searchele_AllProc, isactive, isslave, isself, currentgroup);

    // create interface local parameter list (copy)
    Teuchos::ParameterList icparams = contactParams;

    // find out if interface-specific coefficients of friction are given
    if (frictionType == INPAR::CONTACT::friction_tresca ||
        frictionType == INPAR::CONTACT::friction_coulomb ||
        frictionType == INPAR::CONTACT::friction_stick)
    {
      // read interface COFs
      std::vector<double> frcoeff(currentgroup.size());

      for (unsigned j = 0; j < currentgroup.size(); ++j)
        frcoeff[j] = *currentgroup[j]->Get<double>("FrCoeffOrBound");

      // check consistency of interface COFs
      for (unsigned j = 1; j < currentgroup.size(); ++j)
      {
        if (frcoeff[j] != frcoeff[0])
          dserror("Inconsistency in friction coefficients of interface %i", groupid1);
      }

      // check for infeasible value of COF
      if (frcoeff[0] < 0.0) dserror("Negative FrCoeff / FrBound on interface %i", groupid1);

      // add COF locally to contact parameter list of this interface
      if (frictionType == INPAR::CONTACT::friction_tresca)
      {
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      else if (frictionType == INPAR::CONTACT::friction_coulomb)
      {
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(frcoeff[0]));
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
      // dummy values for FRCOEFF and FRBOUND have to be set,
      // since entries are accessed regardless of the friction law
      else if (frictionType == INPAR::CONTACT::friction_stick)
      {
        icparams.setEntry("FRCOEFF", static_cast<Teuchos::ParameterEntry>(-1.0));
        icparams.setEntry("FRBOUND", static_cast<Teuchos::ParameterEntry>(-1.0));
      }
    }

    // find out if interface-specific coefficients of adhesion are given
    if (adhesionType == INPAR::CONTACT::adhesion_bound)
    {
      // read interface COFs
      std::vector<double> ad_bound(currentgroup.size());
      for (unsigned j = 0; j < currentgroup.size(); ++j)
        ad_bound[j] = *currentgroup[j]->Get<double>("AdhesionBound");

      // check consistency of interface COFs
      for (unsigned j = 1; j < currentgroup.size(); ++j)
      {
        if (ad_bound[j] != ad_bound[0])
          dserror("Inconsistency in adhesion bounds of interface %i", groupid1);
      }

      // check for infeasible value of COF
      if (ad_bound[0] < 0.0) dserror("Negative adhesion bound on interface %i", groupid1);

      // add COF locally to contact parameter list of this interface
      icparams.setEntry("ADHESION_BOUND", static_cast<Teuchos::ParameterEntry>(ad_bound[0]));
    }

    // add information to contact parameter list of this interface
    icparams.set<bool>("Two_half_pass", Two_half_pass);
    icparams.set<bool>("Check_nonsmooth_selfcontactsurface", Check_nonsmooth_selfcontactsurface);
    icparams.set<bool>("Searchele_AllProc", Searchele_AllProc);

    // Safety check for interface storage redundancy in case of self contact
    INPAR::MORTAR::ExtendGhosting redundant =
        Teuchos::getIntegralValue<INPAR::MORTAR::ExtendGhosting>(
            icparams.sublist("PARALLEL REDISTRIBUTION"), "GHOSTING_STRATEGY");
    if (isanyselfcontact == true && redundant != INPAR::MORTAR::ExtendGhosting::redundant_all)
      dserror("Manager: Self contact requires fully redundant slave and master storage");

    // Use factory to create an empty interface and store it in this Manager.
    Teuchos::RCP<CONTACT::Interface> newinterface = STRATEGY::Factory::CreateInterface(groupid1,
        Comm(), dim, icparams, isself[0], Teuchos::null, Teuchos::null, contactconstitutivelawid);
    interfaces.push_back(newinterface);

    // Get the RCP to the last created interface
    Teuchos::RCP<CONTACT::Interface> interface = interfaces.back();

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
      const std::vector<int>* nodeids = currentgroup[j]->GetNodes();
      if (!nodeids) dserror("Condition does not have Node Ids");
      for (unsigned k = 0; k < (*nodeids).size(); ++k)
      {
        int gid = (*nodeids)[k];
        // do only nodes that I have in my discretization
        if (!Discret().NodeColMap()->MyGID(gid)) continue;
        DRT::Node* node = Discret().gNode(gid);
        if (!node) dserror("Cannot find node with gid %", gid);

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
        if (frictionType != INPAR::CONTACT::friction_none)
        {
          Teuchos::RCP<CONTACT::FriNode> cnode =
              Teuchos::rcp(new CONTACT::FriNode(node->Id(), node->X(), node->Owner(),
                  Discret().Dof(0, node), isslave[j], isactive[j] + foundinitialactive, friplus));
          //-------------------
          // get nurbs weight!
          if (nurbs) MORTAR::UTILS::PrepareNURBSNode(node, cnode);

          // get edge and corner information:
          std::vector<DRT::Condition*> contactcornercond(0);
          Discret().GetCondition("mrtrcorner", contactcornercond);
          for (unsigned j = 0; j < contactcornercond.size(); j++)
          {
            if (contactcornercond.at(j)->ContainsNode(node->Id()))
            {
              cnode->SetOnCorner() = true;
            }
          }
          std::vector<DRT::Condition*> contactedgecond(0);
          Discret().GetCondition("mrtredge", contactedgecond);
          for (unsigned j = 0; j < contactedgecond.size(); j++)
          {
            if (contactedgecond.at(j)->ContainsNode(node->Id()))
            {
              cnode->SetOnEdge() = true;
            }
          }

          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<DRT::Condition*> contactSymconditions(0);
          Discret().GetCondition("mrtrsym", contactSymconditions);

          for (unsigned l = 0; l < contactSymconditions.size(); l++)
            if (contactSymconditions.at(l)->ContainsNode(node->Id()))
            {
              const std::vector<int>* onoff =
                  contactSymconditions.at(l)->Get<std::vector<int>>("onoff");
              for (unsigned k = 0; k < onoff->size(); k++)
                if (onoff->at(k) == 1) cnode->DbcDofs()[k] = true;
              if (stype == INPAR::CONTACT::solution_lagmult &&
                  constr_direction != INPAR::CONTACT::constr_xyz)
                dserror(
                    "Contact symmetry with Lagrange multiplier method"
                    " only with contact constraints in xyz direction.\n"
                    "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
            }

          // note that we do not have to worry about double entries
          // as the AddNode function can deal with this case!
          // the only problem would have occurred for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->AddNode(cnode);
        }
        else
        {
          Teuchos::RCP<CONTACT::Node> cnode = Teuchos::rcp(new CONTACT::Node(node->Id(), node->X(),
              node->Owner(), Discret().Dof(0, node), isslave[j], isactive[j] + foundinitialactive));
          //-------------------
          // get nurbs weight!
          if (nurbs)
          {
            MORTAR::UTILS::PrepareNURBSNode(node, cnode);
          }

          // get edge and corner information:
          std::vector<DRT::Condition*> contactcornercond(0);
          Discret().GetCondition("mrtrcorner", contactcornercond);
          for (unsigned j = 0; j < contactcornercond.size(); j++)
          {
            if (contactcornercond.at(j)->ContainsNode(node->Id()))
            {
              cnode->SetOnCorner() = true;
            }
          }
          std::vector<DRT::Condition*> contactedgecond(0);
          Discret().GetCondition("mrtredge", contactedgecond);
          for (unsigned j = 0; j < contactedgecond.size(); j++)
          {
            if (contactedgecond.at(j)->ContainsNode(node->Id()))
            {
              cnode->SetOnEdge() = true;
            }
          }


          // Check, if this node (and, in case, which dofs) are in the contact symmetry condition
          std::vector<DRT::Condition*> contactSymconditions(0);
          Discret().GetCondition("mrtrsym", contactSymconditions);

          for (unsigned l = 0; l < contactSymconditions.size(); l++)
            if (contactSymconditions.at(l)->ContainsNode(node->Id()))
            {
              const std::vector<int>* onoff =
                  contactSymconditions.at(l)->Get<std::vector<int>>("onoff");
              for (unsigned k = 0; k < onoff->size(); k++)
                if (onoff->at(k) == 1) cnode->DbcDofs()[k] = true;
              if (stype == INPAR::CONTACT::solution_lagmult &&
                  constr_direction != INPAR::CONTACT::constr_xyz)
                dserror(
                    "Contact symmetry with Lagrange multiplier method"
                    " only with contact constraints in xyz direction.\n"
                    "Set CONSTRAINT_DIRECTIONS to xyz in CONTACT input section");
            }

          // note that we do not have to worry about double entries
          // as the AddNode function can deal with this case!
          // the only problem would have occured for the initial active nodes,
          // as their status could have been overwritten, but is prevented
          // by the "foundinitialactive" block above!
          interface->AddNode(cnode);
        }
      }
    }

    //----------------------------------------------- process elements
    int ggsize = 0;
    for (unsigned j = 0; j < currentgroup.size(); ++j)
    {
      // get elements from condition j of current group
      std::map<int, Teuchos::RCP<DRT::Element>>& currele = currentgroup[j]->Geometry();

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
      Comm().SumAll(&lsize, &gsize, 1);

      std::map<int, Teuchos::RCP<DRT::Element>>::iterator fool;
      for (fool = currele.begin(); fool != currele.end(); ++fool)
      {
        Teuchos::RCP<DRT::Element> ele = fool->second;
        Teuchos::RCP<CONTACT::Element> cele = Teuchos::rcp(new CONTACT::Element(ele->Id() + ggsize,
            ele->Owner(), ele->Shape(), ele->NumNode(), ele->NodeIds(), isslave[j], nurbs));

        if ((contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroelast ||
                contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroscatra) &&
            algo != INPAR::MORTAR::algorithm_gpts)
          SetPoroParentElement(slavetype, mastertype, cele, ele);

        if (algo == INPAR::MORTAR::algorithm_gpts)
        {
          Teuchos::RCP<DRT::FaceElement> faceele =
              Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
          if (faceele == Teuchos::null) dserror("Cast to FaceElement failed!");
          if (faceele->ParentElement() == nullptr) dserror("face parent does not exist");
          if (Discret().ElementColMap()->LID(faceele->ParentElement()->Id()) == -1)
            dserror("vol dis does not have parent ele");
          cele->SetParentMasterElement(faceele->ParentElement(), faceele->FaceParentNumber());
        }

        //------------------------------------------------------------------
        // get knotvector, normal factor and zero-size information for nurbs
        if (nurbs)
        {
          MORTAR::UTILS::PrepareNURBSElement(discret, ele, cele, dim);
        }

        interface->AddElement(cele);
      }  // for (fool=ele1.start(); fool != ele1.end(); ++fool)

      ggsize += gsize;  // update global element counter
    }

    /* Finalize the contact interface construction
     *
     * Always assign degrees of freedom here, because we need a valid column map for further contact
     * setup. This is an initial one time cost, that does not matter compared to the repeated
     * FillComplete calls due to dynamic redistribution.
     */
    if (CONTACT::UTILS::UseSafeRedistributeAndGhosting(contactParams))
    {
      /* Finalize parallel layout of maps. Note: Do not redistribute here.
       *
       * Since this is the initial setup, we don't need redistribution here, just a proper extension
       * of the interface ghosting.
       */
      interface->UpdateParallelLayoutAndDataStructures(false, true, maxdof, 0.0);
    }
    else
      interface->FillComplete(true, maxdof);

    if ((contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroelast ||
            contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroscatra) &&
        algo != INPAR::MORTAR::algorithm_gpts)
      FindPoroInterfaceTypes(
          poromaster, poroslave, structmaster, structslave, slavetype, mastertype);
  }
  if (Comm().MyPID() == 0) std::cout << "done!" << std::endl;

  //**********************************************************************
  // create the solver strategy object and pass all necessary data to it
  if (Comm().MyPID() == 0)
  {
    std::cout << "Building contact strategy object............";
    fflush(stdout);
  }

  // build the correct data container
  Teuchos::RCP<CONTACT::AbstractStratDataContainer> data_ptr =
      Teuchos::rcp(new CONTACT::AbstractStratDataContainer());

  // create LagrangeStrategyWear for wear as non-distinct quantity
  if (stype == INPAR::CONTACT::solution_lagmult && wearLaw != INPAR::WEAR::wear_none &&
      (wearType == INPAR::WEAR::wear_intstate || wearType == INPAR::WEAR::wear_primvar))
  {
    strategy_ = Teuchos::rcp(new WEAR::LagrangeStrategyWear(data_ptr, Discret().DofRowMap(),
        Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
  }
  else if (stype == INPAR::CONTACT::solution_lagmult)
  {
    if (contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroelast ||
        contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroscatra)
    {
      strategy_ = Teuchos::rcp(
          new LagrangeStrategyPoro(data_ptr, Discret().DofRowMap(), Discret().NodeRowMap(),
              contactParams, interfaces, dim, comm_, alphaf, maxdof, poroslave, poromaster));
    }
    else if (contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::tsi)
    {
      strategy_ = Teuchos::rcp(new LagrangeStrategyTsi(data_ptr, Discret().DofRowMap(),
          Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
    }
    else
    {
      strategy_ = Teuchos::rcp(new LagrangeStrategy(data_ptr, Discret().DofRowMap(),
          Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
    }
  }
  else if (((stype == INPAR::CONTACT::solution_penalty ||
                stype == INPAR::CONTACT::solution_multiscale) &&
               algo != INPAR::MORTAR::algorithm_gpts) ||
           stype == INPAR::CONTACT::solution_uzawa)
  {
    strategy_ = Teuchos::rcp(new PenaltyStrategy(data_ptr, Discret().DofRowMap(),
        Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
  }
  else if (algo == INPAR::MORTAR::algorithm_gpts &&
           (stype == INPAR::CONTACT::solution_nitsche || stype == INPAR::CONTACT::solution_penalty))
  {
    if ((contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroelast ||
            contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::poroscatra) &&
        stype == INPAR::CONTACT::solution_nitsche)
    {
      strategy_ = Teuchos::rcp(new NitscheStrategyPoro(data_ptr, Discret().DofRowMap(),
          Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
    }
    else if (contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::fsi &&
             stype == INPAR::CONTACT::solution_nitsche)
    {
      strategy_ = Teuchos::rcp(new NitscheStrategyFsi(data_ptr, Discret().DofRowMap(),
          Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
    }
    else if (contactParams.get<int>("PROBTYPE") == INPAR::CONTACT::fpi &&
             stype == INPAR::CONTACT::solution_nitsche)
    {
      strategy_ = Teuchos::rcp(new NitscheStrategyFpi(data_ptr, Discret().DofRowMap(),
          Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
    }
    else
    {
      strategy_ = Teuchos::rcp(new NitscheStrategy(data_ptr, Discret().DofRowMap(),
          Discret().NodeRowMap(), contactParams, interfaces, dim, comm_, alphaf, maxdof));
    }
  }
  else if (stype == INPAR::CONTACT::solution_augmented)
  {
    dserror(
        "The augmented contact formulation is no longer supported in the"
        " old structural time integrator!");
  }
  else
  {
    dserror("Unrecognized contact strategy");
  }

  dynamic_cast<CONTACT::AbstractStrategy&>(*strategy_).Setup(false, true);

  if (Comm().MyPID() == 0) std::cout << "done!" << std::endl;
  //**********************************************************************

  // print friction information of interfaces
  if (Comm().MyPID() == 0)
  {
    for (unsigned i = 0; i < interfaces.size(); ++i)
    {
      double checkfrcoeff = 0.0;
      if (frictionType == INPAR::CONTACT::friction_tresca)
      {
        checkfrcoeff = interfaces[i]->InterfaceParams().get<double>("FRBOUND");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrBound (Tresca)  " << checkfrcoeff << std::endl;
      }
      else if (frictionType == INPAR::CONTACT::friction_coulomb)
      {
        checkfrcoeff = interfaces[i]->InterfaceParams().get<double>("FRCOEFF");
        std::cout << std::endl << "Interface         " << i + 1 << std::endl;
        std::cout << "FrCoeff (Coulomb) " << checkfrcoeff << std::endl;
      }
    }
  }

  // print initial parallel redistribution
  if (Comm().MyPID() == 0 && Comm().NumProc() > 1)
    std::cout << "\nInitial parallel distribution of all contact interfaces:" << std::endl;
  for (auto& interface : interfaces) interface->PrintParallelDistribution();

  // create binary search tree
  for (auto& interface : interfaces) interface->CreateSearchTree();

  return;
}


/*----------------------------------------------------------------------*
 |  read and check input parameters (public)                  popp 04/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Manager::ReadAndCheckInput(Teuchos::ParameterList& cparams)
{
  // read parameter lists from GLOBAL::Problem
  const Teuchos::ParameterList& mortar = GLOBAL::Problem::Instance()->MortarCouplingParams();
  const Teuchos::ParameterList& contact = GLOBAL::Problem::Instance()->ContactDynamicParams();
  const Teuchos::ParameterList& wearlist = GLOBAL::Problem::Instance()->WearParams();
  const Teuchos::ParameterList& tsic = GLOBAL::Problem::Instance()->TSIContactParams();
  const Teuchos::ParameterList& stru = GLOBAL::Problem::Instance()->StructuralDynamicParams();

  // read Problem Type and Problem Dimension from GLOBAL::Problem
  const GLOBAL::ProblemType problemtype = GLOBAL::Problem::Instance()->GetProblemType();
  CORE::FE::ShapeFunctionType distype = GLOBAL::Problem::Instance()->SpatialApproximationType();
  const int dim = GLOBAL::Problem::Instance()->NDim();

  // in case just System type system_condensed_lagmult
  if (INPUT::IntegralValue<INPAR::CONTACT::SystemType>(contact, "SYSTEM") ==
      INPAR::CONTACT::system_condensed_lagmult)
    dserror(
        "For Contact anyway just the lagrange multiplier can be condensed, choose SYSTEM = "
        "Condensed.");

  // *********************************************************************
  // invalid parallel strategies
  // *********************************************************************
  const Teuchos::ParameterList& mortarParallelRedistParams =
      mortar.sublist("PARALLEL REDISTRIBUTION");

  if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != INPAR::MORTAR::ParallelRedist::redist_none &&
      mortarParallelRedistParams.get<int>("MIN_ELEPROC") < 0)
    dserror("Minimum number of elements per processor for parallel redistribution must be >= 0");

  if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") == INPAR::MORTAR::ParallelRedist::redist_dynamic &&
      mortarParallelRedistParams.get<double>("MAX_BALANCE_EVAL_TIME") < 1.0)
    dserror(
        "Maximum allowed value of load balance for dynamic parallel redistribution must be "
        ">= 1.0");

  if (problemtype == GLOBAL::ProblemType::tsi &&
      Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") != INPAR::MORTAR::ParallelRedist::redist_none)
    dserror("Parallel redistribution not yet implemented for TSI problems");

  // *********************************************************************
  // adhesive contact
  // *********************************************************************
  if (INPUT::IntegralValue<INPAR::CONTACT::AdhesionType>(contact, "ADHESION") !=
          INPAR::CONTACT::adhesion_none and
      INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none)
    dserror("Adhesion combined with wear not yet tested!");

  if (INPUT::IntegralValue<INPAR::CONTACT::AdhesionType>(contact, "ADHESION") !=
          INPAR::CONTACT::adhesion_none and
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
          INPAR::CONTACT::friction_none)
    dserror("Adhesion combined with friction not yet tested!");

  // *********************************************************************
  // generally invalid combinations (nts/mortar)
  // *********************************************************************
  if ((INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_penalty ||
          INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_nitsche) &&
      contact.get<double>("PENALTYPARAM") <= 0.0)
    dserror("Penalty parameter eps = 0, must be greater than 0");

  if ((INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_penalty ||
          INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_nitsche) &&
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
          INPAR::CONTACT::friction_none &&
      contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    dserror("Tangential penalty parameter eps = 0, must be greater than 0");

  if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          INPAR::CONTACT::solution_uzawa &&
      contact.get<double>("PENALTYPARAM") <= 0.0)
    dserror("Penalty parameter eps = 0, must be greater than 0");

  if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          INPAR::CONTACT::solution_uzawa &&
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
          INPAR::CONTACT::friction_none &&
      contact.get<double>("PENALTYPARAMTAN") <= 0.0)
    dserror("Tangential penalty parameter eps = 0, must be greater than 0");

  if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          INPAR::CONTACT::solution_uzawa &&
      contact.get<int>("UZAWAMAXSTEPS") < 2)
    dserror("Maximum number of Uzawa / Augmentation steps must be at least 2");

  if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          INPAR::CONTACT::solution_uzawa &&
      contact.get<double>("UZAWACONSTRTOL") <= 0.0)
    dserror("Constraint tolerance for Uzawa / Augmentation scheme must be greater than 0");

  if (INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
          INPAR::CONTACT::friction_none &&
      contact.get<double>("SEMI_SMOOTH_CT") == 0.0)
    dserror("Parameter ct = 0, must be greater than 0 for frictional contact");

  if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          INPAR::CONTACT::solution_augmented &&
      contact.get<double>("SEMI_SMOOTH_CN") <= 0.0)
    dserror("Regularization parameter cn, must be greater than 0 for contact problems");

  if (INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") ==
          INPAR::CONTACT::friction_tresca &&
      dim == 3 &&
      INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
          INPAR::CONTACT::solution_nitsche)
    dserror("3D frictional contact with Tresca's law not yet implemented");

  if (INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
          INPAR::CONTACT::friction_none &&
      INPUT::IntegralValue<int>(contact, "SEMI_SMOOTH_NEWTON") != 1 && dim == 3)
    dserror("3D frictional contact only implemented with Semi-smooth Newton");

  if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          INPAR::CONTACT::solution_augmented &&
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
          INPAR::CONTACT::friction_none)
    dserror("Frictional contact is for the augmented Lagrange formulation not yet implemented!");

  if (INPUT::IntegralValue<int>(mortar, "CROSSPOINTS") == true && dim == 3)
    dserror("Crosspoints / edge node modification not yet implemented for 3D");

  if (INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") ==
          INPAR::CONTACT::friction_tresca &&
      INPUT::IntegralValue<int>(contact, "FRLESS_FIRST") == true)
    // Hopefully coming soon, when Coulomb and Tresca are combined. Until then, throw error.
    dserror("Frictionless first contact step with Tresca's law not yet implemented");

  if (INPUT::IntegralValue<INPAR::CONTACT::Regularization>(contact, "CONTACT_REGULARIZATION") !=
          INPAR::CONTACT::reg_none &&
      INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
          INPAR::CONTACT::solution_lagmult)
    dserror(
        "Regularized Contact just available for Dual Mortar Contact with Lagrangean "
        "Multiplier!");

  if (INPUT::IntegralValue<INPAR::CONTACT::Regularization>(contact, "CONTACT_REGULARIZATION") !=
          INPAR::CONTACT::reg_none &&
      INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
          INPAR::CONTACT::friction_none)
    dserror("Regularized Contact for contact with friction not implemented yet!");

  // *********************************************************************
  // warnings
  // *********************************************************************
  if (mortar.get<double>("SEARCH_PARAM") == 0.0 && Comm().MyPID() == 0)
    std::cout << ("Warning: Contact search called without inflation of bounding volumes\n")
              << std::endl;

  if (INPUT::IntegralValue<INPAR::WEAR::WearSide>(wearlist, "WEAR_SIDE") != INPAR::WEAR::wear_slave)
    std::cout << ("\n \n Warning: Contact with both-sided wear is still experimental !")
              << std::endl;


  // *********************************************************************
  //                       MORTAR-SPECIFIC CHECKS
  // *********************************************************************
  if (INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(mortar, "ALGORITHM") ==
      INPAR::MORTAR::algorithm_mortar)
  {
    // *********************************************************************
    // invalid parameter combinations
    // *********************************************************************
    if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            INPAR::CONTACT::solution_lagmult &&
        INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            INPAR::MORTAR::shape_petrovgalerkin)
      dserror("Petrov-Galerkin approach for LM only with Lagrange multiplier strategy");

    if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
            INPAR::CONTACT::solution_lagmult &&
        (INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
                INPAR::MORTAR::shape_standard &&
            INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar, "LM_QUAD") !=
                INPAR::MORTAR::lagmult_const) &&
        INPUT::IntegralValue<INPAR::CONTACT::SystemType>(contact, "SYSTEM") ==
            INPAR::CONTACT::system_condensed)
      dserror("Condensation of linear system only possible for dual Lagrange multipliers");

    if (INPUT::IntegralValue<INPAR::MORTAR::ConsistentDualType>(mortar, "LM_DUAL_CONSISTENT") !=
            INPAR::MORTAR::consistent_none &&
        INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
            INPAR::CONTACT::solution_lagmult &&
        INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
            INPAR::MORTAR::shape_standard)
      dserror(
          "Consistent dual shape functions in boundary elements only for Lagrange "
          "multiplier strategy.");

    if (INPUT::IntegralValue<INPAR::MORTAR::ConsistentDualType>(mortar, "LM_DUAL_CONSISTENT") !=
            INPAR::MORTAR::consistent_none &&
        INPUT::IntegralValue<INPAR::MORTAR::IntType>(mortar, "INTTYPE") ==
            INPAR::MORTAR::inttype_elements &&
        (INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            INPAR::MORTAR::shape_dual))
      dserror(
          "Consistent dual shape functions in boundary elements not for purely "
          "element-based integration.");

    if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
            INPAR::CONTACT::solution_nitsche &&
        INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(mortar, "ALGORITHM") !=
            INPAR::MORTAR::algorithm_gpts)
      dserror("Nitsche contact only with GPTS algorithm.");


    // *********************************************************************
    // not (yet) implemented combinations
    // *********************************************************************

    if (INPUT::IntegralValue<int>(mortar, "CROSSPOINTS") == true &&
        INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar, "LM_QUAD") ==
            INPAR::MORTAR::lagmult_lin)
      dserror("Crosspoints and linear LM interpolation for quadratic FE not yet compatible");

    // check for self contact
    bool self = false;
    {
      std::vector<DRT::Condition*> contactCondition(0);
      Discret().GetCondition("Mortar", contactCondition);

      for (const auto& condition : contactCondition)
      {
        const std::string* side = condition->Get<std::string>("Side");
        if (*side == "Selfcontact") self = true;
      }
    }

    if (self == true &&
        Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(mortarParallelRedistParams,
            "PARALLEL_REDIST") != INPAR::MORTAR::ParallelRedist::redist_none)
      dserror("Self contact and parallel redistribution not yet compatible");

    if (INPUT::IntegralValue<int>(contact, "INITCONTACTBYGAP") == true &&
        contact.get<double>("INITCONTACTGAPVALUE") == 0.0)
      dserror("For initialization of init contact with gap, the INITCONTACTGAPVALUE is needed.");

    if (INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none &&
        INPUT::IntegralValue<int>(contact, "FRLESS_FIRST") == true)
      dserror("Frictionless first contact step with wear not yet implemented");

    if (problemtype != GLOBAL::ProblemType::ehl &&
        INPUT::IntegralValue<int>(contact, "REGULARIZED_NORMAL_CONTACT") == true)
      dserror("Regularized normal contact only implemented for EHL");

    // *********************************************************************
    // Augmented Lagrangian strategy
    // *********************************************************************
    if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
        INPAR::CONTACT::solution_augmented)
    {
      dserror("No longer supported!");
    }

    // *********************************************************************
    // thermal-structure-interaction contact
    // *********************************************************************
    if (problemtype == GLOBAL::ProblemType::tsi &&
        INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            INPAR::MORTAR::shape_standard)
      dserror("Thermal contact only for dual shape functions");

    if (problemtype == GLOBAL::ProblemType::tsi &&
        INPUT::IntegralValue<INPAR::CONTACT::SystemType>(contact, "SYSTEM") !=
            INPAR::CONTACT::system_condensed)
      dserror("Thermal contact only for dual shape functions with condensed system");

    // no nodal scaling in for thermal-structure-interaction
    if (problemtype == GLOBAL::ProblemType::tsi &&
        tsic.get<double>("TEMP_DAMAGE") <= tsic.get<double>("TEMP_REF"))
      dserror("damage temperature must be greater than reference temperature");

    // *********************************************************************
    // contact with wear
    // *********************************************************************
    if (INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") == INPAR::WEAR::wear_none &&
        wearlist.get<double>("WEARCOEFF") != 0.0)
      dserror("Wear coefficient only necessary in the context of wear.");

    if (problemtype == GLOBAL::ProblemType::structure and
        INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") !=
            INPAR::WEAR::wear_none and
        INPUT::IntegralValue<INPAR::WEAR::WearTimInt>(wearlist, "WEARTIMINT") !=
            INPAR::WEAR::wear_expl)
      dserror(
          "Wear calculation for pure structure problems only with explicit internal state "
          "variable approach reasonable!");

    if (INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") ==
            INPAR::CONTACT::friction_none &&
        INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none)
      dserror("Wear models only applicable to frictional contact.");

    if (INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none &&
        wearlist.get<double>("WEARCOEFF") <= 0.0)
      dserror("No valid wear coefficient provided, must be equal or greater 0.0");

    //    if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact,"STRATEGY") !=
    //    INPAR::CONTACT::solution_lagmult
    //        && INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW")     !=
    //        INPAR::WEAR::wear_none)
    //      dserror("Wear model only applicable in combination with Lagrange multiplier
    //      strategy.");

    if (INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") ==
            INPAR::CONTACT::friction_tresca &&
        INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none)
      dserror("Wear only for Coulomb friction!");

    // *********************************************************************
    // 3D quadratic mortar (choice of interpolation and testing fcts.)
    // *********************************************************************
    if (INPUT::IntegralValue<INPAR::MORTAR::LagMultQuad>(mortar, "LM_QUAD") ==
            INPAR::MORTAR::lagmult_pwlin &&
        INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") ==
            INPAR::MORTAR::shape_dual)
      dserror(
          "No piecewise linear approach (for LM) implemented for quadratic contact with "
          "DUAL shape fct.");

    // *********************************************************************
    // poroelastic contact
    // *********************************************************************
    if (problemtype == GLOBAL::ProblemType::poroelast ||
        problemtype == GLOBAL::ProblemType::poroscatra ||
        problemtype == GLOBAL::ProblemType::fpsi || problemtype == GLOBAL::ProblemType::fpsi_xfem)
    {
      const Teuchos::ParameterList& porodyn = GLOBAL::Problem::Instance()->PoroelastDynamicParams();
      if ((INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
                  INPAR::MORTAR::shape_dual &&
              INPUT::IntegralValue<INPAR::MORTAR::ShapeFcn>(mortar, "LM_SHAPEFCN") !=
                  INPAR::MORTAR::shape_petrovgalerkin) &&
          INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_lagmult)
        dserror("POROCONTACT: Only dual and petrovgalerkin shape functions implemented yet!");

      if (Teuchos::getIntegralValue<INPAR::MORTAR::ParallelRedist>(mortarParallelRedistParams,
              "PARALLEL_REDIST") != INPAR::MORTAR::ParallelRedist::redist_none &&
          INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_lagmult)
        dserror(
            "POROCONTACT: Parallel Redistribution not implemented yet!");  // Since we use Pointers
                                                                           // to Parent Elements,
                                                                           // which are not copied
                                                                           // to other procs!

      if (INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
              INPAR::CONTACT::solution_lagmult &&
          INPUT::IntegralValue<int>(porodyn, "CONTACTNOPEN"))
        dserror("POROCONTACT: Use Lagrangean Strategy for poro contact!");

      if (INPUT::IntegralValue<INPAR::CONTACT::FrictionType>(contact, "FRICTION") !=
              INPAR::CONTACT::friction_none &&
          INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_lagmult)
        dserror("POROCONTACT: Friction for poro contact not implemented!");

      if (INPUT::IntegralValue<INPAR::CONTACT::SystemType>(contact, "SYSTEM") !=
              INPAR::CONTACT::system_condensed &&
          INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
              INPAR::CONTACT::solution_lagmult)
        dserror("POROCONTACT: System has to be condensed for poro contact!");

      if ((dim != 3) && (dim != 2))
      {
        const Teuchos::ParameterList& porodyn =
            GLOBAL::Problem::Instance()->PoroelastDynamicParams();
        if (INPUT::IntegralValue<int>(porodyn, "CONTACTNOPEN"))
          dserror("POROCONTACT: PoroContact with no penetration just tested for 3d (and 2d)!");
      }
    }

    // *********************************************************************
    // element-based vs. segment-based mortar integration
    // *********************************************************************
    INPAR::MORTAR::IntType inttype =
        INPUT::IntegralValue<INPAR::MORTAR::IntType>(mortar, "INTTYPE");

    if (inttype == INPAR::MORTAR::inttype_elements && mortar.get<int>("NUMGP_PER_DIM") <= 0)
      dserror("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");

    if (inttype == INPAR::MORTAR::inttype_elements_BS && mortar.get<int>("NUMGP_PER_DIM") <= 0)
      dserror(
          "Invalid Gauss point number NUMGP_PER_DIM for element-based integration with "
          "boundary segmentation."
          "\nPlease note that the value you have to provide only applies to the element-based "
          "integration"
          "\ndomain, while pre-defined default values will be used in the segment-based boundary "
          "domain.");

    if ((inttype == INPAR::MORTAR::inttype_elements ||
            inttype == INPAR::MORTAR::inttype_elements_BS) &&
        mortar.get<int>("NUMGP_PER_DIM") <= 1)
      dserror("Invalid Gauss point number NUMGP_PER_DIM for element-based integration.");
  }  // END MORTAR CHECKS

  // *********************************************************************
  //                       NTS-SPECIFIC CHECKS
  // *********************************************************************
  else if (INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(mortar, "ALGORITHM") ==
           INPAR::MORTAR::algorithm_nts)
  {
    if (problemtype == GLOBAL::ProblemType::poroelast or problemtype == GLOBAL::ProblemType::fpsi or
        problemtype == GLOBAL::ProblemType::tsi)
      dserror("NTS only for problem type: structure");
  }  // END NTS CHECKS

  // *********************************************************************
  //                       GPTS-SPECIFIC CHECKS
  // *********************************************************************
  else if (INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(mortar, "ALGORITHM") ==
           INPAR::MORTAR::algorithm_gpts)
  {
    const_cast<Teuchos::ParameterList&>(GLOBAL::Problem::Instance()->ContactDynamicParams())
        .set("SYSTEM", "none");

    if (contact.get<double>("PENALTYPARAM") <= 0.0)
      dserror("Penalty parameter eps = 0, must be greater than 0");

    if (problemtype != GLOBAL::ProblemType::structure &&
        problemtype != GLOBAL::ProblemType::poroelast &&
        problemtype != GLOBAL::ProblemType::fsi_xfem &&
        problemtype != GLOBAL::ProblemType::fpsi_xfem)
      dserror(
          "GPTS algorithm only tested for structural, FSI-CutFEM, FPSI-CutFEM, and "
          "poroelastic problems");

    if (INPUT::IntegralValue<INPAR::WEAR::WearLaw>(wearlist, "WEARLAW") != INPAR::WEAR::wear_none)
      dserror("GPTS algorithm not implemented for wear");

  }  // END GPTS CHECKS

  // *********************************************************************
  // store contents of BOTH ParameterLists in local parameter list
  // *********************************************************************
  cparams.setParameters(mortar);
  cparams.setParameters(contact);
  cparams.setParameters(wearlist);
  cparams.setParameters(tsic);
  if (problemtype == GLOBAL::ProblemType::tsi)
    cparams.set<double>(
        "TIMESTEP", GLOBAL::Problem::Instance()->TSIDynamicParams().get<double>("TIMESTEP"));
  else if (problemtype != GLOBAL::ProblemType::structure)
  {
    // rauch 01/16
    if (Comm().MyPID() == 0)
      std::cout << "\n \n  Warning: CONTACT::Manager::ReadAndCheckInput() reads TIMESTEP = "
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
    case CORE::FE::ShapeFunctionType::nurbs:
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
  if (problemtype == GLOBAL::ProblemType::structure)
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::structure);
  }
  else if (problemtype == GLOBAL::ProblemType::tsi)
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::tsi);
  }
  else if (problemtype == GLOBAL::ProblemType::struct_ale)
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::structalewear);
  }
  else if (problemtype == GLOBAL::ProblemType::poroelast or
           problemtype == GLOBAL::ProblemType::fpsi or
           problemtype == GLOBAL::ProblemType::poroscatra)
  {
    const Teuchos::ParameterList& porodyn = GLOBAL::Problem::Instance()->PoroelastDynamicParams();
    if (problemtype == GLOBAL::ProblemType::poroelast or problemtype == GLOBAL::ProblemType::fpsi)
      cparams.set<int>("PROBTYPE", INPAR::CONTACT::poroelast);
    else if (problemtype == GLOBAL::ProblemType::poroscatra)
      cparams.set<int>("PROBTYPE", INPAR::CONTACT::poroscatra);
    // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
    double porotimefac =
        1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
    cparams.set<double>("porotimefac", porotimefac);
    cparams.set<bool>("CONTACTNOPEN",
        INPUT::IntegralValue<int>(porodyn, "CONTACTNOPEN"));  // used in the integrator
  }
  else if (problemtype == GLOBAL::ProblemType::fsi_xfem)
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::fsi);
  }
  else if (problemtype == GLOBAL::ProblemType::fpsi_xfem)
  {
    const Teuchos::ParameterList& porodyn = GLOBAL::Problem::Instance()->PoroelastDynamicParams();
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::fpi);
    // porotimefac = 1/(theta*dt) --- required for derivation of structural displacements!
    double porotimefac =
        1 / (stru.sublist("ONESTEPTHETA").get<double>("THETA") * stru.get<double>("TIMESTEP"));
    cparams.set<double>("porotimefac", porotimefac);
    cparams.set<bool>("CONTACTNOPEN",
        INPUT::IntegralValue<int>(porodyn, "CONTACTNOPEN"));  // used in the integrator
  }
  else
  {
    cparams.set<int>("PROBTYPE", INPAR::CONTACT::other);
  }

  // no parallel redistribution in the serial case
  if (Comm().NumProc() == 1)
    cparams.sublist("PARALLEL REDISTRIBUTION").set<std::string>("PARALLEL_REDIST", "None");

  // set dimension
  cparams.set<int>("DIMENSION", dim);
  return true;
}

/*----------------------------------------------------------------------*
 |  write restart information for contact (public)            popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::WriteRestart(IO::DiscretizationWriter& output, bool forcedrestart)
{
  // clear cache of maps due to varying vector size
  output.ClearMapCache();

  // quantities to be written for restart
  std::map<std::string, Teuchos::RCP<Epetra_Vector>> restart_vectors;

  // quantities to be written for restart
  GetStrategy().DoWriteRestart(restart_vectors, forcedrestart);

  if (GetStrategy().LagrMultOld() != Teuchos::null)
    output.WriteVector("lagrmultold", GetStrategy().LagrMultOld());

  // write all vectors specified by used strategy
  for (std::map<std::string, Teuchos::RCP<Epetra_Vector>>::const_iterator p =
           restart_vectors.begin();
       p != restart_vectors.end(); ++p)
    output.WriteVector(p->first, p->second);

  return;
}

/*----------------------------------------------------------------------*
 |  read restart information for contact (public)             popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::ReadRestart(IO::DiscretizationReader& reader,
    Teuchos::RCP<Epetra_Vector> dis, Teuchos::RCP<Epetra_Vector> zero)
{
  // If Parent Elements are required, we need to reconnect them before contact restart!
  INPAR::MORTAR::AlgorithmType atype =
      INPUT::IntegralValue<INPAR::MORTAR::AlgorithmType>(GetStrategy().Params(), "ALGORITHM");
  if (atype == INPAR::MORTAR::algorithm_gpts)
  {
    for (unsigned i = 0;
         i < dynamic_cast<CONTACT::AbstractStrategy&>(GetStrategy()).ContactInterfaces().size();
         ++i)
      dynamic_cast<CONTACT::AbstractStrategy&>(GetStrategy())
          .ContactInterfaces()[i]
          ->CreateVolumeGhosting();
  }

  // If Parent Elements are required, we need to reconnect them before contact restart!
  if ((GetStrategy().Params().get<int>("PROBTYPE") == INPAR::CONTACT::poroelast ||
          GetStrategy().Params().get<int>("PROBTYPE") == INPAR::CONTACT::poroscatra) ||
      GetStrategy().Params().get<int>("PROBTYPE") == INPAR::CONTACT::fpi)
    ReconnectParentElements();

  // this is contact, thus we need the displacement state for restart
  // let strategy object do all the work
  GetStrategy().DoReadRestart(reader, dis);

  return;
}

/*----------------------------------------------------------------------*
 |  write interface tractions for postprocessing (public)     popp 03/08|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::PostprocessQuantities(IO::DiscretizationWriter& output)
{
  if (GetStrategy().IsNitsche()) return;

  // *********************************************************************
  // active contact set and slip set
  // *********************************************************************

  // evaluate active set and slip set
  Teuchos::RCP<Epetra_Vector> activeset =
      Teuchos::rcp(new Epetra_Vector(*GetStrategy().ActiveRowNodes()));
  activeset->PutScalar(1.0);
  if (GetStrategy().Friction())
  {
    Teuchos::RCP<Epetra_Vector> slipset =
        Teuchos::rcp(new Epetra_Vector(*GetStrategy().SlipRowNodes()));
    slipset->PutScalar(1.0);
    Teuchos::RCP<Epetra_Vector> slipsetexp =
        Teuchos::rcp(new Epetra_Vector(*GetStrategy().ActiveRowNodes()));
    CORE::LINALG::Export(*slipset, *slipsetexp);
    activeset->Update(1.0, *slipsetexp, 1.0);
  }

  // export to problem node row map
  Teuchos::RCP<Epetra_Map> problemnodes = GetStrategy().ProblemNodes();
  Teuchos::RCP<Epetra_Vector> activesetexp = Teuchos::rcp(new Epetra_Vector(*problemnodes));
  CORE::LINALG::Export(*activeset, *activesetexp);

  if (GetStrategy().WearBothDiscrete())
  {
    Teuchos::RCP<Epetra_Vector> mactiveset =
        Teuchos::rcp(new Epetra_Vector(*GetStrategy().MasterActiveNodes()));
    mactiveset->PutScalar(1.0);
    Teuchos::RCP<Epetra_Vector> slipset =
        Teuchos::rcp(new Epetra_Vector(*GetStrategy().MasterSlipNodes()));
    slipset->PutScalar(1.0);
    Teuchos::RCP<Epetra_Vector> slipsetexp =
        Teuchos::rcp(new Epetra_Vector(*GetStrategy().MasterActiveNodes()));
    CORE::LINALG::Export(*slipset, *slipsetexp);
    mactiveset->Update(1.0, *slipsetexp, 1.0);

    Teuchos::RCP<Epetra_Vector> mactivesetexp = Teuchos::rcp(new Epetra_Vector(*problemnodes));
    CORE::LINALG::Export(*mactiveset, *mactivesetexp);
    activesetexp->Update(1.0, *mactivesetexp, 1.0);
  }

  output.WriteVector("activeset", activesetexp);

  // *********************************************************************
  //  weighted gap
  // *********************************************************************
  // export to problem dof row map
  Teuchos::RCP<Epetra_Map> gapnodes = GetStrategy().ProblemNodes();
  Teuchos::RCP<Epetra_Vector> gaps =
      Teuchos::rcp_dynamic_cast<CONTACT::AbstractStrategy>(strategy_)->ContactWGap();
  if (gaps != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> gapsexp = Teuchos::rcp(new Epetra_Vector(*gapnodes));
    CORE::LINALG::Export(*gaps, *gapsexp);

    output.WriteVector("gap", gapsexp);
  }

  // *********************************************************************
  // contact tractions
  // *********************************************************************

  // evaluate contact tractions
  GetStrategy().ComputeContactStresses();

  // export to problem dof row map
  Teuchos::RCP<Epetra_Map> problemdofs = GetStrategy().ProblemDofs();

  // normal direction
  Teuchos::RCP<Epetra_Vector> normalstresses = GetStrategy().ContactNorStress();
  Teuchos::RCP<Epetra_Vector> normalstressesexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  CORE::LINALG::Export(*normalstresses, *normalstressesexp);

  // tangential plane
  Teuchos::RCP<Epetra_Vector> tangentialstresses = GetStrategy().ContactTanStress();
  Teuchos::RCP<Epetra_Vector> tangentialstressesexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  CORE::LINALG::Export(*tangentialstresses, *tangentialstressesexp);

  // write to output
  // contact tractions in normal and tangential direction
  output.WriteVector("norcontactstress", normalstressesexp);
  output.WriteVector("tancontactstress", tangentialstressesexp);

  if (GetStrategy().ContactNorForce() != Teuchos::null)
  {
    // normal direction
    Teuchos::RCP<Epetra_Vector> normalforce = GetStrategy().ContactNorForce();
    Teuchos::RCP<Epetra_Vector> normalforceexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    CORE::LINALG::Export(*normalforce, *normalforceexp);

    // tangential plane
    Teuchos::RCP<Epetra_Vector> tangentialforce = GetStrategy().ContactTanForce();
    Teuchos::RCP<Epetra_Vector> tangentialforceexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    CORE::LINALG::Export(*tangentialforce, *tangentialforceexp);

    // write to output
    // contact tractions in normal and tangential direction
    output.WriteVector("norslaveforce", normalforceexp);
    output.WriteVector("tanslaveforce", tangentialforceexp);
  }


#ifdef CONTACTFORCEOUTPUT

  // *********************************************************************
  // contact forces on slave non master side,
  // in normal and tangential direction
  // *********************************************************************
  // vectors for contact forces
  Teuchos::RCP<Epetra_Vector> fcslavenor =
      Teuchos::rcp(new Epetra_Vector(GetStrategy().DMatrix()->RowMap()));
  Teuchos::RCP<Epetra_Vector> fcslavetan =
      Teuchos::rcp(new Epetra_Vector(GetStrategy().DMatrix()->RowMap()));
  Teuchos::RCP<Epetra_Vector> fcmasternor =
      Teuchos::rcp(new Epetra_Vector(GetStrategy().MMatrix()->DomainMap()));
  Teuchos::RCP<Epetra_Vector> fcmastertan =
      Teuchos::rcp(new Epetra_Vector(GetStrategy().MMatrix()->DomainMap()));

  // vectors with problem dof row map
  Teuchos::RCP<Epetra_Vector> fcslavenorexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  Teuchos::RCP<Epetra_Vector> fcslavetanexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  Teuchos::RCP<Epetra_Vector> fcmasternorexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
  Teuchos::RCP<Epetra_Vector> fcmastertanexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));

  // multiplication
  GetStrategy().DMatrix()->Multiply(true, *normalstresses, *fcslavenor);
  GetStrategy().DMatrix()->Multiply(true, *tangentialstresses, *fcslavetan);
  GetStrategy().MMatrix()->Multiply(true, *normalstresses, *fcmasternor);
  GetStrategy().MMatrix()->Multiply(true, *tangentialstresses, *fcmastertan);

#ifdef MASTERNODESINCONTACT
  // BEGIN: to output the global ID's of the master nodes in contact - devaal 02.2011

  int dim = GLOBAL::Problem::Instance()->NDim();

  if (dim == 2) dserror("Only working for 3D");

  std::vector<int> lnid, gnid;

  // std::cout << "MasterNor" << fcmasternor->MyLength() << std::endl;

  for (int i = 0; i < fcmasternor->MyLength(); i = i + 3)
  {
    // check if master node in contact
    if (sqrt(((*fcmasternor)[i]) * ((*fcmasternor)[i]) +
             ((*fcmasternor)[i + 1]) * ((*fcmasternor)[i + 1]) +
             ((*fcmasternor)[i + 2]) * ((*fcmasternor)[i] + 2)) > 0.00001)
    {
      lnid.push_back((fcmasternor->Map()).GID(i) / 3);
    }
  }

  // we want to gather data from on all procs
  std::vector<int> allproc(Comm().NumProc());
  for (int i = 0; i < Comm().NumProc(); ++i) allproc[i] = i;

  // communicate all data to proc 0
  CORE::LINALG::Gather<int>(lnid, gnid, static_cast<int>(llproc.size()), allproc.data(), Comm());

  // std::cout << " size of gnid:" << gnid.size() << std::endl;

  ////////////////
  ///// attempt at obtaining the nid and relative displacement u of master nodes in contact - devaal
  // define my own interface
  MORTAR::StrategyBase& myStrategy = GetStrategy();
  AbstractStrategy& myContactStrategy = dynamic_cast<AbstractStrategy&>(myStrategy);

  std::vector<Teuchos::RCP<CONTACT::Interface>> myInterface = myContactStrategy.ContactInterfaces();

  // check interface size - just doing this now for a single interface

  if (myInterface.size() != 1) dserror("Interface size should be 1");

  std::cout << "OUTPUT OF MASTER NODE IN CONTACT" << std::endl;
  for (const auto& globalNodeId : gnid) std::cout << globalNodeId << std::endl;

#endif  // MASTERNODESINCONTACT: to output the global ID's of the master nodes in contact

  //  // when we do a boundary modification we shift slave entries to the M matrix with
  //  // negative sign. Therefore, we have to extract the right force entries from the
  //  // master force which correcpond to the slave force!
  //  Teuchos::RCP<Epetra_Vector> slavedummy =
  //      Teuchos::rcp(new Epetra_Vector(GetStrategy().DMatrix()->RowMap(),true));
  //  CORE::LINALG::Export(*fcmasternor,*slavedummy);
  //  int err = fcslavenor->Update(-1.0,*slavedummy,1.0);
  //  if(err!=0)
  //    dserror("ERROR");
  //
  //  Teuchos::RCP<Epetra_Vector> masterdummy =
  //      Teuchos::rcp(new Epetra_Vector(GetStrategy().MMatrix()->DomainMap(),true));
  //  CORE::LINALG::Export(*slavedummy,*masterdummy);
  //  err = fcmasternor->Update(-1.0,*masterdummy,1.0);
  //  if(err!=0)
  //    dserror("ERROR");

  // export
  CORE::LINALG::Export(*fcslavenor, *fcslavenorexp);
  CORE::LINALG::Export(*fcslavetan, *fcslavetanexp);
  CORE::LINALG::Export(*fcmasternor, *fcmasternorexp);
  CORE::LINALG::Export(*fcmastertan, *fcmastertanexp);

  // contact forces on slave and master side
  output.WriteVector("norslaveforce", fcslavenorexp);
  output.WriteVector("tanslaveforce", fcslavetanexp);
  output.WriteVector("normasterforce", fcmasternorexp);
  output.WriteVector("tanmasterforce", fcmastertanexp);

#ifdef CONTACTEXPORT
  // export averaged node forces to xxx.force
  double resultnor[fcslavenor->NumVectors()];
  double resulttan[fcslavetan->NumVectors()];
  fcslavenor->Norm2(resultnor);
  fcslavetan->Norm2(resulttan);

  if (Comm().MyPID() == 0)
  {
    std::cout << "resultnor= " << resultnor[0] << std::endl;
    std::cout << "resulttan= " << resulttan[0] << std::endl;

    FILE* MyFile = nullptr;
    std::ostringstream filename;
    const std::string filebase =
        GLOBAL::Problem::Instance()->OutputControlFile()->FileNameOnlyPrefix();
    filename << filebase << ".force";
    MyFile = fopen(filename.str().c_str(), "at+");
    if (MyFile)
    {
      // fprintf(MyFile,valuename.c_str());
      fprintf(MyFile, "%g\t", resultnor[0]);
      fprintf(MyFile, "%g\n", resulttan[0]);
      fclose(MyFile);
    }
    else
      dserror("File for Output could not be opened.");
  }
#endif  // CONTACTEXPORT
#endif  // CONTACTFORCEOUTPUT

  // *********************************************************************
  // wear with internal state variable approach
  // *********************************************************************
  bool wwear = GetStrategy().WeightedWear();
  if (wwear)
  {
    // ***************************************************************************
    // we do not compute the non-weighted wear here. we just write    farah 06/13
    // the output. the non-weighted wear will be used as dirichlet-b.
    // for the ale problem. n.w.wear will be called in stru_ale_algorithm.cpp
    // and computed in GetStrategy().OutputWear();
    // ***************************************************************************

    // evaluate wear (not weighted)
    GetStrategy().OutputWear();

    // write output
    Teuchos::RCP<Epetra_Vector> wearoutput = GetStrategy().ContactWear();
    Teuchos::RCP<Epetra_Vector> wearoutputexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    CORE::LINALG::Export(*wearoutput, *wearoutputexp);
    output.WriteVector("wear", wearoutputexp);
    GetStrategy().ContactWear()->PutScalar(0.0);
  }

  // *********************************************************************
  // poro contact
  // *********************************************************************
  bool poro = GetStrategy().HasPoroNoPenetration();
  if (poro)
  {
    // output of poro no penetration lagrange multiplier!
    CONTACT::LagrangeStrategyPoro& costrategy =
        dynamic_cast<CONTACT::LagrangeStrategyPoro&>(GetStrategy());
    Teuchos::RCP<Epetra_Vector> lambdaout = costrategy.LambdaNoPen();
    Teuchos::RCP<Epetra_Vector> lambdaoutexp = Teuchos::rcp(new Epetra_Vector(*problemdofs));
    CORE::LINALG::Export(*lambdaout, *lambdaoutexp);
    output.WriteVector("poronopen_lambda", lambdaoutexp);
  }
  return;
}

/*-----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Manager::PostprocessQuantitiesPerInterface(
    Teuchos::RCP<Teuchos::ParameterList> outputParams)
{
  GetStrategy().PostprocessQuantitiesPerInterface(outputParams);
}

/*----------------------------------------------------------------------------------------------*
 |  Reconnect Contact Element -- Parent Element Pointers (required for restart)       ager 04/16|
 *---------------------------------------------------------------------------------------------*/
void CONTACT::Manager::ReconnectParentElements()
{
  {
    const Epetra_Map* elecolmap = discret_.ElementColMap();

    CONTACT::AbstractStrategy& strategy = dynamic_cast<CONTACT::AbstractStrategy&>(GetStrategy());

    for (auto& interface : strategy.ContactInterfaces())
    {
      const Epetra_Map* ielecolmap = interface->Discret().ElementColMap();

      for (int i = 0; i < ielecolmap->NumMyElements(); ++i)
      {
        int gid = ielecolmap->GID(i);

        DRT::Element* ele = interface->Discret().gElement(gid);
        if (!ele) dserror("Cannot find element with gid %", gid);
        DRT::FaceElement* faceele = dynamic_cast<DRT::FaceElement*>(ele);

        int volgid = faceele->ParentElementId();
        if (elecolmap->LID(volgid) == -1)  // Volume Discretization has not Element
          dserror(
              "Manager::ReconnectParentElements: Element %d does not exist on this Proc!", volgid);

        DRT::Element* vele = discret_.gElement(volgid);
        if (!vele) dserror("Cannot find element with gid %", volgid);

        faceele->SetParentMasterElement(vele, faceele->FaceParentNumber());
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  Set Parent Elements for Poro Face Elements                ager 11/15|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::SetPoroParentElement(int& slavetype, int& mastertype,
    Teuchos::RCP<CONTACT::Element>& cele, Teuchos::RCP<DRT::Element>& ele)
{
  // ints to communicate decision over poro bools between processors on every interface
  // safety check - because there may not be mixed interfaces and structural slave elements
  // slavetype ... 1 poro, 0 struct, -1 default
  // mastertype ... 1 poro, 0 struct, -1 default
  Teuchos::RCP<DRT::FaceElement> faceele = Teuchos::rcp_dynamic_cast<DRT::FaceElement>(ele, true);
  if (faceele == Teuchos::null) dserror("Cast to FaceElement failed!");
  cele->PhysType() = MORTAR::Element::other;
  std::vector<Teuchos::RCP<DRT::Condition>> porocondvec;
  discret_.GetCondition("PoroCoupling", porocondvec);
  if (!cele->IsSlave())  // treat an element as a master element if it is no slave element
  {
    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->Geometry().begin();
           eleitergeometry != porocondvec[i]->Geometry().end(); ++eleitergeometry)
      {
        if (faceele->ParentElement()->Id() == eleitergeometry->second->Id())
        {
          if (mastertype == 0)
            dserror(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          cele->PhysType() = MORTAR::Element::poro;
          mastertype = 1;
          break;
        }
      }
    }
    if (cele->PhysType() == MORTAR::Element::other)
    {
      if (mastertype == 1)
        dserror(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele->PhysType() = MORTAR::Element::structure;
      mastertype = 0;
    }
  }
  else if (cele->IsSlave())  // treat an element as slave element if it is one
  {
    for (unsigned int i = 0; i < porocondvec.size(); ++i)
    {
      std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator eleitergeometry;
      for (eleitergeometry = porocondvec[i]->Geometry().begin();
           eleitergeometry != porocondvec[i]->Geometry().end(); ++eleitergeometry)
      {
        if (faceele->ParentElement()->Id() == eleitergeometry->second->Id())
        {
          if (slavetype == 0)
            dserror(
                "struct and poro master elements on the same processor - no mixed interface "
                "supported");
          cele->PhysType() = MORTAR::Element::poro;
          slavetype = 1;
          break;
        }
      }
    }
    if (cele->PhysType() == MORTAR::Element::other)
    {
      if (slavetype == 1)
        dserror(
            "struct and poro master elements on the same processor - no mixed interface supported");
      cele->PhysType() = MORTAR::Element::structure;
      slavetype = 0;
    }
  }
  // store information about parent for porous contact (required for calculation of deformation
  // gradient!) in every contact element although only really needed for phystype poro
  cele->SetParentMasterElement(faceele->ParentElement(), faceele->FaceParentNumber());
  return;
}

/*----------------------------------------------------------------------*
 |  Find Physical Type (Poro or Structure) of Poro Interface  ager 11/15|
 *----------------------------------------------------------------------*/
void CONTACT::Manager::FindPoroInterfaceTypes(bool& poromaster, bool& poroslave, bool& structmaster,
    bool& structslave, int& slavetype, int& mastertype)
{
  // find poro and structure elements when a poro coupling condition is applied on an element
  // and restrict to pure poroelastic or pure structural interfaces' sides.
  //(only poro slave elements AND (only poro master elements or only structure master elements)
  // Tell the contact element which physical type it is to extract PhysType in contact integrator
  // bools to decide which side is structural and which side is poroelastic to manage all 4
  // constellations
  // s-s, p-s, s-p, p-p
  // wait for all processors to determine if they have poro or structural master or slave elements
  comm_->Barrier();
  std::vector<int> slaveTypeList(comm_->NumProc());
  std::vector<int> masterTypeList(comm_->NumProc());
  comm_->GatherAll(&slavetype, slaveTypeList.data(), 1);
  comm_->GatherAll(&mastertype, masterTypeList.data(), 1);
  comm_->Barrier();

  for (int i = 0; i < comm_->NumProc(); ++i)
  {
    switch (slaveTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structslave)
          dserror(
              "struct and poro slave elements in the same problem - no mixed interface "
              "constellations supported");
        // adjust dserror text, when more than one interface is supported
        poroslave = true;
        break;
      case 0:
        if (poroslave)
          dserror(
              "struct and poro slave elements in the same problem - no mixed interface "
              "constellations supported");
        structslave = true;
        break;
      default:
        dserror("this cannot happen");
        break;
    }
  }

  for (int i = 0; i < comm_->NumProc(); ++i)
  {
    switch (masterTypeList[i])
    {
      case -1:
        break;
      case 1:
        if (structmaster)
          dserror(
              "struct and poro master elements in the same problem - no mixed interface "
              "constellations supported");
        // adjust dserror text, when more than one interface is supported
        poromaster = true;
        break;
      case 0:
        if (poromaster)
          dserror(
              "struct and poro master elements in the same problem - no mixed interface "
              "constellations supported");
        structmaster = true;
        break;
      default:
        dserror("this cannot happen");
        break;
    }
  }
}

BACI_NAMESPACE_CLOSE
