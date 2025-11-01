// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beamcontact_beam3contact_manager.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beam3_kirchhoff.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beamcontact_beam3contact_octtree.hpp"
#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_rigidsphere.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  constructor (public)                                      popp 04/10|
 *----------------------------------------------------------------------*/
CONTACT::Beam3cmanager::Beam3cmanager(Core::FE::Discretization& discret, double alphaf)
    : numnodes_(0),
      numnodalvalues_(0),
      pdiscret_(discret),
      pdiscomm_(discret.get_comm()),
      searchradius_(0.0),
      sphericalsearchradius_(0.0),
      alphaf_(alphaf),
      constrnorm_(0.0),
      btsolconstrnorm_(0.0),
      maxtotalsimgap_(-1000.0),
      maxtotalsimgap_cp_(-1000.0),
      maxtotalsimgap_gp_(-1000.0),
      maxtotalsimgap_ep_(-1000.0),
      maxtotalsimrelgap_(-1000.0),
      mintotalsimgap_(1000.0),
      mintotalsimgap_cp_(1000.0),
      mintotalsimgap_gp_(1000.0),
      mintotalsimgap_ep_(1000.0),
      mintotalsimrelgap_(1000.0),
      mintotalsimunconvgap_(1000.0),
      totpenaltyenergy_(0.0),
      totpenaltywork_(0.0),
      maxdeltadisp_(0.0),
      totalmaxdeltadisp_(0.0),
      firststep_(true),
      elementtypeset_(false),
      outputcounter_(0),
      timen_(0.0),
      contactevaluationtime_(0.0),
      global_kappa_max_(0.0),
      step_(0)
{
  // initialize vectors of contact forces
  fc_ = Core::LinAlg::create_vector(*discret.dof_row_map(), false);
  fcold_ = Core::LinAlg::create_vector(*discret.dof_row_map(), false);
  fc_->put_scalar(0.0);
  fcold_->put_scalar(0.0);

  contactpairmap_.clear();
  oldcontactpairmap_.clear();
  btsolpairmap_.clear();
  oldbtsolpairmap_.clear();

  // read parameter lists from Global::Problem
  sbeamcontact_ = Global::Problem::instance()->beam_contact_params();
  scontact_ = Global::Problem::instance()->contact_dynamic_params();
  sstructdynamic_ = Global::Problem::instance()->structural_dynamic_params();

  // indicate if beam-to-solid contact is applied
  btsol_ = beam_contact_parameters().get<bool>("BEAMS_BTSOL");

  init_beam_contact_discret();

  // check input parameters
  if (sbeamcontact_.get<double>("BEAMS_BTBPENALTYPARAM") < 0.0 ||
      sbeamcontact_.get<double>("BEAMS_BTSPENALTYPARAM") < 0.0)
  {
    FOUR_C_THROW("ERROR: The penalty parameter has to be positive.");
  }

  // initialize beam-to-beam contact element pairs
  pairs_.resize(0);
  oldpairs_.resize(0);

  // initialize beam-to-solid contact element pairs
  btsolpairs_.resize(0);
  oldbtsolpairs_.resize(0);

  // initialize input parameters
  currentpp_ = sbeamcontact_.get<double>("BEAMS_BTBPENALTYPARAM");
  btspp_ = sbeamcontact_.get<double>("BEAMS_BTSPENALTYPARAM");

  if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
  {
    std::cout << "========================= Beam Contact =========================" << std::endl;
    std::cout << "Elements in discret.   = " << pdiscret_.num_global_elements() << std::endl;
  }

  // Set maximal and minimal beam/sphere radius occurring in discretization
  set_min_max_ele_radius();

  // Get search box increment from input file
  searchboxinc_ = BeamInteraction::determine_searchbox_inc(sbeamcontact_);

  if (searchboxinc_ < 0.0)
    FOUR_C_THROW("Choose a positive value for the searchbox extrusion factor BEAMS_EXTVAL!");

  // initialize octtree for contact search
  if (Teuchos::getIntegralValue<BeamContact::OctreeType>(sbeamcontact_, "BEAMS_OCTREE") !=
      BeamContact::boct_none)
  {
    if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
      std::cout << "BTB-CO penalty         = " << currentpp_ << std::endl;

    if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
      std::cout << "BTS-CO penalty         = " << btspp_ << std::endl;

    tree_ = std::make_shared<Beam3ContactOctTree>(sbeamcontact_, pdiscret_, *btsoldiscret_);
  }
  else
  {
    if (btsol_ || btsolmt_)
      FOUR_C_THROW(
          "Beam to solid contact/meshtying are only implemented for the octree contact search!");

    // compute the search radius for searching possible contact pairs
    compute_search_radius();
    tree_ = nullptr;
    if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
      std::cout << "\nBrute Force Search" << std::endl;
  }

  if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
  {
    if (Teuchos::getIntegralValue<BeamContact::Strategy>(sbeamcontact_, "BEAMS_STRATEGY") ==
        BeamContact::bstr_penalty)
      std::cout << "Strategy                 Penalty" << std::endl;
    else
      FOUR_C_THROW("Unknown strategy for beam contact!");

    switch (Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(sbeamcontact_, "BEAMS_PENALTYLAW"))
    {
      case BeamContact::pl_lp:
      {
        std::cout << "Regularization Type      Linear penalty law!" << std::endl;
        break;
      }
      case BeamContact::pl_qp:
      {
        std::cout << "Regularization Type      Quadratic penalty law!" << std::endl;
        break;
      }
      case BeamContact::pl_lnqp:
      {
        std::cout << "Regularization Type      Linear penalty law with quadratic regularization "
                     "for negative gaps!"
                  << std::endl;
        break;
      }
      case BeamContact::pl_lpqp:
      {
        std::cout << "Regularization Type      Linear penalty law with quadratic regularization "
                     "for positive gaps!"
                  << std::endl;
        break;
      }
      case BeamContact::pl_lpcp:
      {
        std::cout << "Regularization Type      Linear penalty law with cubic regularization for "
                     "positive gaps!"
                  << std::endl;
        break;
      }
      case BeamContact::pl_lpdqp:
      {
        std::cout << "Regularization Type      Linear penalty law with double quadratic "
                     "regularization for positive gaps!"
                  << std::endl;
        break;
      }
      case BeamContact::pl_lpep:
      {
        std::cout << "Regularization Type      Linear penalty law with exponential regularization "
                     "for positive gaps!"
                  << std::endl;
        break;
      }
    }

    if (Teuchos::getIntegralValue<BeamContact::PenaltyLaw>(sbeamcontact_, "BEAMS_PENALTYLAW") !=
        BeamContact::pl_lp)
    {
      std::cout << "Regularization Params    BEAMS_PENREGPARAM_G0 = "
                << sbeamcontact_.get<double>("BEAMS_PENREGPARAM_G0", -1.0)
                << ",  BEAMS_PENREGPARAM_F0 = "
                << sbeamcontact_.get<double>("BEAMS_PENREGPARAM_F0", -1.0)
                << ",  BEAMS_PENREGPARAM_C0 = "
                << sbeamcontact_.get<double>("BEAMS_PENREGPARAM_C0", -1.0)
                << ",  BEAMS_GAPSHIFTPARAM = "
                << sbeamcontact_.get<double>("BEAMS_GAPSHIFTPARAM", 0.0) << std::endl;
    }

    if (sbeamcontact_.get<bool>("BEAMS_DAMPING") == false)
      std::cout << "Damping                  No Contact Damping Force Applied!" << std::endl;
    else
    {
      std::cout << "Damping                  BEAMS_DAMPINGPARAM = "
                << sbeamcontact_.get<double>("BEAMS_DAMPINGPARAM", -1.0)
                << ",    BEAMS_DAMPREGPARAM1 = "
                << sbeamcontact_.get<double>("BEAMS_DAMPREGPARAM1", -1.0)
                << ",   BEAMS_DAMPREGPARAM2 = "
                << sbeamcontact_.get<double>("BEAMS_DAMPREGPARAM2", -1.0) << std::endl;
    }

    if (sbeamcontact_.get<double>("BEAMS_BASICSTIFFGAP", -1000.0) != -1000.0)
    {
      std::cout << "Linearization            For gaps < -"
                << sbeamcontact_.get<double>("BEAMS_BASICSTIFFGAP", -1000.0)
                << " only the basic part of the contact linearization is applied!" << std::endl;
    }

    std::cout << "================================================================\n" << std::endl;
  }

  dis_ = Core::LinAlg::create_vector(*problem_discret().dof_row_map(), true);
  dis_old_ = Core::LinAlg::create_vector(*problem_discret().dof_row_map(), true);

  return;
}

/*----------------------------------------------------------------------*
 |  print beam3 contact manager (public)                      popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::print(std::ostream& os) const
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    os << "Beam3 Contact discretization:" << std::endl;

  problem_discret().print(os);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate contact (public)                                 popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::evaluate(Core::LinAlg::SparseMatrix& stiffmatrix,
    Core::LinAlg::Vector<double>& fres, const Core::LinAlg::Vector<double>& disrow,
    Teuchos::ParameterList timeintparams, bool newsti, double time)
{
  // set time
  timen_ = time;

  // Set class variable
  dis_->update(1.0, disrow, 0.0);

  // map linking node numbers and current node positions
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions;
  currentpositions.clear();
  // extract fully overlapping displacement vector on contact discretization from
  // displacement vector in row map format on problem discretization
  Core::LinAlg::Vector<double> disccol(*bt_sol_discret().dof_col_map(), true);
  shift_dis_map(disrow, disccol);
  // update currentpositions
  set_current_positions(currentpositions, disccol);

  double t_start = 0.0;
  double t_end = 0.0;

  //**********************************************************************
  // SEARCH
  //**********************************************************************
  std::vector<std::vector<Core::Elements::Element*>> elementpairs, elementpairspot;
  elementpairs.clear();
  elementpairspot.clear();
  //**********************************************************************
  // Contact: Octree search
  //**********************************************************************
  if (tree_ != nullptr)
  {
    t_start = Teuchos::Time::wallTime();
    elementpairs = tree_->oct_tree_search(currentpositions);

    t_end = Teuchos::Time::wallTime() - t_start;
    Teuchos::ParameterList ioparams = Global::Problem::instance()->io_params();
    if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()) &&
        ioparams.get<int>("STDOUTEVERY", 0))
      Core::IO::cout(Core::IO::standard)
          << "      OctTree Search (Contact): " << t_end << " seconds" << Core::IO::endl;
  }
  //**********************************************************************
  // Contact: brute-force search
  //**********************************************************************
  else
  {
    t_start = Teuchos::Time::wallTime();

    elementpairs = brute_force_search(currentpositions, searchradius_, sphericalsearchradius_);
    t_end = Teuchos::Time::wallTime() - t_start;
    Teuchos::ParameterList ioparams = Global::Problem::instance()->io_params();
    if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()) &&
        ioparams.get<int>("STDOUTEVERY", 0))
      Core::IO::cout(Core::IO::standard)
          << "      Brute Force Search (Contact): " << t_end << " seconds" << Core::IO::endl;
  }

  t_start = Teuchos::Time::wallTime();

  // process the found element pairs and fill the BTB, BTSOL, BTSPH interaction pair vectors
  fill_contact_pairs_vectors(elementpairs);

  // update element state of all pairs with current positions (already calculated in
  // set_current_positions) and current tangents (will be calculated in set_state)
  set_state(currentpositions, disccol);


  // At this point we have all possible contact pairs_ with updated positions
  //**********************************************************************
  // evaluation of contact pairs
  //**********************************************************************
  // every proc that owns one of the nodes of a 'beam3contact' object has
  // to evaluate this object. Fc and Stiffc will be evaluated. Assembly
  // of the additional stiffness will be done by the objects themselves,
  // the assembly of Fc has to be done by the 'beam3cmanager', because the
  // additional force has to be known for the current and the last time
  // step due to generalized alpha time integration. The current contact
  // forces will be stored in 'fc_', the previous ones in 'fcold_'. An
  // update method at the end of each time step manages the data transfer
  // from 'fc_' to 'fcold_'. This update method is called by the time
  // integration class.
  //**********************************************************************
  // initialize global contact force vectors
  fc_->put_scalar(0.0);

  // initialize contact stiffness and uncomplete global stiffness
  stiffc_ = std::make_shared<Core::LinAlg::SparseMatrix>(stiffmatrix.range_map(), 100);
  stiffmatrix.un_complete();


  t_end = Teuchos::Time::wallTime() - t_start;
  if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
    Core::IO::cout(Core::IO::debug)
        << "      Pair management: " << t_end << " seconds. " << Core::IO::endl;
  t_start = Teuchos::Time::wallTime();

  // evaluate all element pairs (BTB, BTSOL, BTSPH; Contact)
  evaluate_all_pairs(timeintparams);

  t_end = Teuchos::Time::wallTime() - t_start;
  if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
    Core::IO::cout(Core::IO::debug)
        << "      Evaluate Contact Pairs: " << t_end << " seconds. " << Core::IO::endl;
  double sumproc_evaluationtime = 0.0;
  sumproc_evaluationtime = Core::Communication::sum_all(t_end, get_comm());
  contactevaluationtime_ += sumproc_evaluationtime;
  t_start = Teuchos::Time::wallTime();

  if (Teuchos::getIntegralValue<Inpar::Solid::MassLin>(sstructdynamic_, "MASSLIN") !=
      Inpar::Solid::MassLin::ml_rotations)
  {
    // assemble contact forces into global fres vector
    fres.update(1.0 - alphaf_, *fc_, 1.0);
    fres.update(alphaf_, *fcold_, 1.0);
    // determine contact stiffness matrix scaling factor (new STI)
    // (this is due to the fact that in the new STI, we hand in the
    // already appropriately scaled effective stiffness matrix. Thus,
    // the additional contact stiffness terms must be equally scaled
    // here, as well. In the old STI, the complete scaling operation
    // is done after contact evaluation within the time integrator,
    // therefore no special scaling needs to be applied here.)
    double scalemat = 1.0;
    if (newsti) scalemat = 1.0 - alphaf_;
    // assemble contact stiffness into global stiffness matrix
    stiffc_->complete();
    stiffmatrix.add(*stiffc_, false, scalemat, 1.0);
    stiffmatrix.complete();
  }
  else
  {
    // assemble contact forces into global fres vector
    fres.update(1.0, *fc_, 1.0);

    // assemble contact stiffness into global stiffness matrix
    stiffc_->complete();
    stiffmatrix.add(*stiffc_, false, 1.0, 1.0);
    stiffmatrix.complete();
  }

  t_end = Teuchos::Time::wallTime() - t_start;
  if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()))
    Core::IO::cout(Core::IO::debug)
        << "      Post-manage Pairs: " << t_end << " seconds. " << Core::IO::endl;

  return;
}

/*----------------------------------------------------------------------*
 |  Shift map of displacement vector                         meier 05/14|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::shift_dis_map(
    const Core::LinAlg::Vector<double>& disrow, Core::LinAlg::Vector<double>& disccol)
{
  // export displacements into fully overlapping column map format
  Core::LinAlg::Vector<double> discrow(*bt_sol_discret().dof_row_map(), true);
  int numbtsdofs = (*bt_sol_discret().dof_row_map()).num_my_elements();

  for (int i = 0; i < numbtsdofs; i++)
  {
    int btsolcontact_gid = (*bt_sol_discret().dof_row_map()).gid(i);
    int problem_gid = dofoffsetmap_[btsolcontact_gid];
    double disp =
        disrow.local_values_as_span()[(*problem_discret().dof_row_map()).lid(problem_gid)];
    discrow.replace_global_value(btsolcontact_gid, disp);
  }
  Core::LinAlg::export_to(discrow, disccol);

  return;
}

/*----------------------------------------------------------------------*
 | setup of contact discretization btsoldiscret_             grill 05/16|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::init_beam_contact_discret()
{
  // create new (basically copied) discretization for contact
  // (to ease our search algorithms we afford the luxury of
  // ghosting all nodes and all elements on all procs, i.e.
  // we export the discretization to full overlap. However,
  // we do not want to do this with the actual discretization
  // and thus create a stripped copy here that only contains
  // nodes and elements).
  // Then, within all beam contact specific routines we will
  // NEVER use the underlying problem discretization but always
  // the copied beam contact discretization.

  {
    MPI_Comm comm(pdiscret_.get_comm());
    btsoldiscret_ = std::make_shared<Core::FE::Discretization>(
        (std::string) "beam to solid contact", comm, Global::Problem::instance()->n_dim());
  }
  dofoffsetmap_.clear();
  std::map<int, std::vector<int>> nodedofs;
  nodedofs.clear();

  // loop over all column nodes of underlying problem discret and add
  for (auto node : problem_discret().my_col_node_range())
  {
    std::shared_ptr<Core::Nodes::Node> newnode =
        std::shared_ptr<Core::Nodes::Node>(node.user_node()->clone());

    if (BeamInteraction::Utils::is_beam_node(*newnode))
    {
      bt_sol_discret().add_node(node.x(), node.global_id(), newnode);
      nodedofs[node.global_id()] = problem_discret().dof(0, node);
    }
    else if (BeamInteraction::Utils::is_rigid_sphere_node(*newnode))
    {
      bt_sol_discret().add_node(node.x(), node.global_id(), newnode);
      nodedofs[node.global_id()] = problem_discret().dof(0, node);
    }
    else
    {
      if (btsol_ == false && btsolmt_ == false)
        FOUR_C_THROW(
            "Only beam elements are allowed as long as the flags btsol_ and btsolmt_ are set to "
            "false!");
    }
  }

  int maxproblemid = problem_discret().element_row_map()->max_all_gid();
  // loop over all column elements of underlying problem discret and add
  for (int i = 0; i < (problem_discret().element_col_map())->num_my_elements(); ++i)
  {
    Core::Elements::Element* ele = problem_discret().l_col_element(i);
    if (!ele) FOUR_C_THROW("Cannot find element with local id {}", i);
    std::shared_ptr<Core::Elements::Element> newele =
        std::shared_ptr<Core::Elements::Element>(ele->clone());
    if (BeamInteraction::Utils::is_beam_element(*newele) or
        BeamInteraction::Utils::is_rigid_sphere_element(*newele))
    {
      bt_sol_discret().add_element(newele);
    }
  }

  // begin: determine surface elements and their nodes

  // vector that contains solid-to-solid and beam-to-solid contact pairs
  std::vector<const Core::Conditions::Condition*> beamandsolidcontactconditions;
  problem_discret().get_condition("Contact", beamandsolidcontactconditions);

  // vector that solely contains beam-to-solid contact pairs
  std::vector<const Core::Conditions::Condition*> btscontactconditions;

  // vector that solely contains beam-to-solid meshtying pairs
  std::vector<const Core::Conditions::Condition*> btsmeshtyingconditions;

  // sort out solid-to-solid contact pairs, since these are treated in the contact framework
  for (int i = 0; i < (int)beamandsolidcontactconditions.size(); ++i)
  {
    if ((beamandsolidcontactconditions[i]->parameters().get<std::string>("Application")) ==
        "Beamtosolidcontact")
      btscontactconditions.push_back(beamandsolidcontactconditions[i]);
    if ((beamandsolidcontactconditions[i]->parameters().get<std::string>("Application")) ==
        "Beamtosolidmeshtying")
      btsmeshtyingconditions.push_back(beamandsolidcontactconditions[i]);
  }

  //******************************* BEAM-TO-SOLID CONTACT *******************************
  solcontacteles_.resize(0);
  solcontactnodes_.resize(0);
  int ggsize = 0;

  //-------------------------------------------------- process surface nodes
  for (int j = 0; j < (int)btscontactconditions.size(); ++j)
  {
    // get all nodes and add them
    const std::vector<int>* nodeids = btscontactconditions[j]->get_nodes();
    if (!nodeids) FOUR_C_THROW("Condition does not have Node Ids");
    for (int k = 0; k < (int)(*nodeids).size(); ++k)
    {
      int gid = (*nodeids)[k];
      // do only nodes that I have in my discretization
      if (!problem_discret().node_col_map()->my_gid(gid)) continue;
      Core::Nodes::Node* node = problem_discret().g_node(gid);

      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

      std::shared_ptr<CONTACT::Node> cnode = std::make_shared<CONTACT::Node>(node->id(), node->x(),
          node->owner(), problem_discret().dof(0, node),
          false,   // all solid elements are master elements
          false);  // no "initially active" decision necessary for beam to solid contact

      // note that we do not have to worry about double entries
      // as the add_node function can deal with this case!
      // the only problem would have occurred for the initial active nodes,
      // as their status could have been overwritten, but is prevented
      // by the "foundinitialactive" block above!
      solcontactnodes_.push_back(cnode);
      bt_sol_discret().add_node(cnode->x(), cnode->id(), cnode);
      nodedofs[node->id()] = problem_discret().dof(0, node);
    }
  }

  //----------------------------------------------- process surface elements
  for (int j = 0; j < (int)btscontactconditions.size(); ++j)
  {
    // get elements from condition j of current group
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& currele =
        btscontactconditions[j]->geometry();

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
      // The IDs of the surface elements of each conditions begin with zero. Therefore we have to
      // add ggsize in order to get unique element IDs in the end. Furthermore, only the solid
      // elements are added to the contact discretization btsoldiscret_ via the
      // btscontactconditions, whereas all beam elements with their original ID are simply cloned
      // from the problem discretization into the contact discretization. In order to avoid solid
      // element IDs being identical to these beam element IDs within the contact discretization we
      // have to add the additional offset maxproblemid, which is identical to the maximal element
      // ID in the problem discretization.
      std::shared_ptr<CONTACT::Element> cele =
          std::make_shared<CONTACT::Element>(ele->id() + ggsize + maxproblemid + 1, ele->owner(),
              ele->shape(), ele->num_node(), ele->node_ids(),
              false,   // all solid elements are master elements
              false);  // no nurbs allowed up to now

      solcontacteles_.push_back(cele);
      bt_sol_discret().add_element(cele);
    }  // for (fool=ele1.start(); fool != ele1.end(); ++fool)
    ggsize += gsize;  // update global element counter
  }
  // end: determine surface elements and their nodes

  //****************************** BEAM-TO-SOLID MESHTYING ******************************
  solmeshtyingeles_.resize(0);
  solmeshtyingnodes_.resize(0);

  //-------------------------------------------------- process surface nodes
  for (int j = 0; j < (int)btsmeshtyingconditions.size(); ++j)
  {
    // get all nodes and add them
    const std::vector<int>* nodeids = btsmeshtyingconditions[j]->get_nodes();
    if (!nodeids) FOUR_C_THROW("Condition does not have Node Ids");
    for (int k = 0; k < (int)(*nodeids).size(); ++k)
    {
      int gid = (*nodeids)[k];
      // do only nodes that I have in my discretization
      if (!problem_discret().node_col_map()->my_gid(gid)) continue;
      Core::Nodes::Node* node = problem_discret().g_node(gid);

      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);

      std::shared_ptr<Mortar::Node> mtnode = std::make_shared<Mortar::Node>(node->id(), node->x(),
          node->owner(), problem_discret().dof(0, node),
          false);  // all solid elements are master elements

      // note that we do not have to worry about double entries
      // as the add_node function can deal with this case!
      // the only problem would have occurred for the initial active nodes,
      // as their status could have been overwritten, but is prevented
      // by the "foundinitialactive" block above!
      solmeshtyingnodes_.push_back(mtnode);
      bt_sol_discret().add_node(mtnode->x(), mtnode->id(), mtnode);
      nodedofs[node->id()] = problem_discret().dof(0, node);
    }
  }

  //----------------------------------------------- process surface elements
  for (int j = 0; j < (int)btsmeshtyingconditions.size(); ++j)
  {
    // get elements from condition j of current group
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& currele =
        btsmeshtyingconditions[j]->geometry();

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
      // The IDs of the surface elements of each conditions begin with zero. Therefore we have to
      // add ggsize in order to get unique element IDs in the end. Furthermore, only the solid
      // elements are added to the contact discretization btsoldiscret_ via the
      // btscontactconditions, whereas all beam elements with their original ID are simply cloned
      // from the problem discretization into the contact discretization. In order to avoid solid
      // element IDs being identical to these beam element IDs within the contact discretization we
      // have to add the additional offset maxproblemid, which is identical to the maximal element
      // ID in the problem discretization.
      std::shared_ptr<Mortar::Element> mtele =
          std::make_shared<Mortar::Element>(ele->id() + ggsize + maxproblemid + 1, ele->owner(),
              ele->shape(), ele->num_node(), ele->node_ids(),
              false,   // all solid elements are master elements
              false);  // no nurbs allowed up to now

      solmeshtyingeles_.push_back(mtele);
      bt_sol_discret().add_element(mtele);
    }  // for (fool=ele1.start(); fool != ele1.end(); ++fool)
    ggsize += gsize;  // update global element counter
  }
  // end: determine surface elements and their nodes

  // build maps but do not assign dofs yet, we'll do this below
  // after shuffling around of nodes and elements (saves time)
  bt_sol_discret().fill_complete(Core::FE::OptionsFillComplete::none());

  // store the node and element row and column maps into this manager
  noderowmap_ = std::make_shared<Core::LinAlg::Map>(*(bt_sol_discret().node_row_map()));
  elerowmap_ = std::make_shared<Core::LinAlg::Map>(*(bt_sol_discret().element_row_map()));
  nodecolmap_ = std::make_shared<Core::LinAlg::Map>(*(bt_sol_discret().node_col_map()));
  elecolmap_ = std::make_shared<Core::LinAlg::Map>(*(bt_sol_discret().element_col_map()));

  // build fully overlapping node and element maps
  // fill my own row node ids into vector (e)sdata
  std::vector<int> sdata(noderowmap_->num_my_elements());
  std::vector<int> esdata(elerowmap_->num_my_elements());
  for (int i = 0; i < noderowmap_->num_my_elements(); ++i) sdata[i] = noderowmap_->gid(i);
  for (int i = 0; i < elerowmap_->num_my_elements(); ++i) esdata[i] = elerowmap_->gid(i);

  // if current proc is participating it writes row IDs into (e)stproc
  std::vector<int> stproc;
  std::vector<int> estproc;
  if (noderowmap_->num_my_elements())
    stproc.push_back(Core::Communication::my_mpi_rank(bt_sol_discret().get_comm()));
  if (elerowmap_->num_my_elements())
    estproc.push_back(Core::Communication::my_mpi_rank(bt_sol_discret().get_comm()));

  // information how many processors participate in total
  std::vector<int> allproc(Core::Communication::num_mpi_ranks(bt_sol_discret().get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(bt_sol_discret().get_comm()); ++i)
    allproc[i] = i;

  // declaring new variables into which the info of (e)stproc on all processors is gathered
  std::vector<int> rtproc;
  std::vector<int> ertproc;

  // gathers information of (e)stproc and writes it into (e)rtproc; in the end (e)rtproc
  // is a vector which contains the numbers of all processors which own nodes/elements.
  Core::LinAlg::gather<int>(stproc, rtproc,
      Core::Communication::num_mpi_ranks(bt_sol_discret().get_comm()), allproc.data(),
      bt_sol_discret().get_comm());
  Core::LinAlg::gather<int>(estproc, ertproc,
      Core::Communication::num_mpi_ranks(bt_sol_discret().get_comm()), allproc.data(),
      bt_sol_discret().get_comm());

  // in analogy to (e)stproc and (e)rtproc the variables (e)rdata gather all the row ID
  // numbers which are  stored on different processors in their own variables (e)sdata; thus,
  // each processor gets the information about all the row ID numbers existing in the problem
  std::vector<int> rdata;
  std::vector<int> erdata;

  // gather all gids of nodes redundantly from (e)sdata into (e)rdata
  Core::LinAlg::gather<int>(
      sdata, rdata, (int)rtproc.size(), rtproc.data(), bt_sol_discret().get_comm());
  Core::LinAlg::gather<int>(
      esdata, erdata, (int)ertproc.size(), ertproc.data(), bt_sol_discret().get_comm());

  // build completely overlapping node map (on participating processors)
  std::shared_ptr<Core::LinAlg::Map> newnodecolmap = std::make_shared<Core::LinAlg::Map>(
      -1, (int)rdata.size(), rdata.data(), 0, bt_sol_discret().get_comm());
  sdata.clear();
  stproc.clear();
  rdata.clear();
  allproc.clear();

  // build completely overlapping element map (on participating processors)
  std::shared_ptr<Core::LinAlg::Map> newelecolmap = std::make_shared<Core::LinAlg::Map>(
      -1, (int)erdata.size(), erdata.data(), 0, bt_sol_discret().get_comm());
  esdata.clear();
  estproc.clear();
  erdata.clear();

  // store the fully overlapping node and element maps
  nodefullmap_ = std::make_shared<Core::LinAlg::Map>(*newnodecolmap);
  elefullmap_ = std::make_shared<Core::LinAlg::Map>(*newelecolmap);

  // pass new fully overlapping node and element maps to beam contact discretization
  bt_sol_discret().export_column_nodes(*newnodecolmap);
  bt_sol_discret().export_column_elements(*newelecolmap);

  // complete beam contact discretization based on the new column maps
  // (this also assign new degrees of freedom what we actually do not
  // want, thus we have to introduce a dof mapping next)
  bt_sol_discret().fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  // communicate the map nodedofs to all proccs
  Core::Communication::Exporter ex(
      *(problem_discret().node_col_map()), *(bt_sol_discret().node_col_map()), get_comm());
  ex.do_export(nodedofs);

  // Determine offset between the IDs of problem discretization and BTSol discretization
  for (int i = 0; i < (bt_sol_discret().node_col_map())->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = bt_sol_discret().l_col_node(i);
    int nodeid = node->id();
    std::vector<int> btsolnodedofids = bt_sol_discret().dof(0, node);
    std::vector<int> originalnodedofids = nodedofs[nodeid];

    if (btsolnodedofids.size() != originalnodedofids.size())
      FOUR_C_THROW(
          "Number of nodal DoFs does not match! "
          "node (GID {}) originally had {} DoFs, now in BTSOLdiscret {} DoFs!",
          nodeid, originalnodedofids.size(), btsolnodedofids.size());

    for (int j = 0; j < (int)btsolnodedofids.size(); j++)
    {
      dofoffsetmap_[btsolnodedofids[j]] = originalnodedofids[j];
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Set current displacement state                            popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::set_current_positions(
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Vector<double>& disccol)
{
  //**********************************************************************
  // get positions of all nodes (fully overlapping map) and store
  // the current results into currentpositions
  //**********************************************************************

  // loop over all beam contact nodes
  for (int i = 0; i < full_nodes()->num_my_elements(); ++i)
  {
    // get node pointer
    Core::Nodes::Node* node = bt_sol_discret().l_col_node(i);

    // TODO maybe this can be done in a more elegant way in the future
    /* check whether node is a beam node which is NOT used for centerline interpolation
     * if so, we simply skip it because it does not have position (and tangent) DoFs */
    if (BeamInteraction::Utils::is_beam_node(*node) and
        !BeamInteraction::Utils::is_beam_centerline_node(*node))
      continue;

    // get GIDs of this node's degrees of freedom
    std::vector<int> dofnode = bt_sol_discret().dof(node);

    // nodal positions
    Core::LinAlg::Matrix<3, 1> currpos;
    currpos(0) = node->x()[0] +
                 disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[0])];
    currpos(1) = node->x()[1] +
                 disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[1])];
    currpos(2) = node->x()[2] +
                 disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[2])];

    // store into currentpositions
    currentpositions[node->id()] = currpos;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Set current displacement state                            popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::set_state(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Vector<double>& disccol)
{
  // map to store the nodal tangent vectors (necessary for Kirchhoff type beams) and address it with
  // the node ID
  std::map<int, Core::LinAlg::Matrix<3, 1>> currenttangents;
  currenttangents.clear();

  // Update of nodal tangents for Kirchhoff elements; nodal positions have already been set in
  // set_current_positions loop over all beam contact nodes
  for (int i = 0; i < full_nodes()->num_my_elements(); ++i)
  {
    // get node pointer
    Core::Nodes::Node* node = bt_sol_discret().l_col_node(i);

    // TODO maybe this can be done in a more elegant way in the future
    /* check whether node is a beam node which is NOT used for centerline interpolation
     * if so, we simply skip it because it does not have position (and tangent) DoFs */
    if (BeamInteraction::Utils::is_beam_node(*node) and
        !BeamInteraction::Utils::is_beam_centerline_node(*node))
      continue;


    // get nodal tangents for Kirchhoff elements
    if (numnodalvalues_ == 2 and BeamInteraction::Utils::is_beam_node(*node))
    {
      // get GIDs of this node's degrees of freedom
      std::vector<int> dofnode = bt_sol_discret().dof(node);

      Core::LinAlg::Matrix<3, 1> currtan(Core::LinAlg::Initialization::zero);
      for (int i = 0; i < numnodes_;
          i++)  // TODO for now, use number of centerline nodes numnodes_ (=2) (no matter how many
                // nodes the function call node->adjacent_elements()[0].user_element()->num_node()
                // would tell you)
      {
        if (node->adjacent_elements()[0].user_element()->nodes()[i]->id() == node->id() and
            node->adjacent_elements()[0].user_element()->element_type() ==
                Discret::Elements::Beam3ebType::instance())
        {
          const Discret::Elements::Beam3eb* ele = dynamic_cast<const Discret::Elements::Beam3eb*>(
              node->adjacent_elements()[0].user_element());
          currtan(0) =
              ((ele->tref())[i])(0) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[3])];
          currtan(1) =
              ((ele->tref())[i])(1) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[4])];
          currtan(2) =
              ((ele->tref())[i])(2) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[5])];
        }
        else if (node->adjacent_elements()[0].user_element()->nodes()[i]->id() == node->id() and
                 node->adjacent_elements()[0].user_element()->element_type() ==
                     Discret::Elements::Beam3kType::instance())
        {
          const Discret::Elements::Beam3k* ele = dynamic_cast<const Discret::Elements::Beam3k*>(
              node->adjacent_elements()[0].user_element());
          currtan(0) =
              ((ele->tref())[i])(0) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[3])];
          currtan(1) =
              ((ele->tref())[i])(1) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[4])];
          currtan(2) =
              ((ele->tref())[i])(2) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[5])];
        }
        else if (node->adjacent_elements()[0].user_element()->nodes()[i]->id() == node->id() and
                 node->adjacent_elements()[0].user_element()->element_type() ==
                     Discret::Elements::Beam3rType::instance())
        {
          const Discret::Elements::Beam3r* ele = dynamic_cast<const Discret::Elements::Beam3r*>(
              node->adjacent_elements()[0].user_element());
          currtan(0) =
              ((ele->tref())[i])(0) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[6])];
          currtan(1) =
              ((ele->tref())[i])(1) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[7])];
          currtan(2) =
              ((ele->tref())[i])(2) +
              disccol.local_values_as_span()[bt_sol_discret().dof_col_map()->lid(dofnode[8])];
        }
      }
      // store into currenttangents
      currenttangents[node->id()] = currtan;
    }
    // set currenttangents to zero for Lagrange centerline interpolation
    else
    {
      for (int i = 0; i < 3; i++) currenttangents[node->id()](i) = 0.0;
    }
  }

  //**********************************************************************
  // update nodal coordinates also in existing contact pairs objects
  //**********************************************************************
  // loop over all pairs

  for (int i = 0; i < (int)pairs_.size(); ++i)
  {
    // temporary matrices to store nodal coordinates of each element
    Core::LinAlg::SerialDenseMatrix ele1pos(3 * numnodalvalues_, numnodes_);
    Core::LinAlg::SerialDenseMatrix ele2pos(3 * numnodalvalues_, numnodes_);
    // Positions: Loop over all nodes of element 1
    /* be careful here: beam eles (such as beam3k, beam3r) may have intermediate
     * nodes which are not used for centerline interpolation and thus do not have
     * position or tangent DoFs. we therefore use numnodes_ here rather than the query
     * (btsphpotpairs_[i]->Element1())->num_node() */
    // TODO do the same for beam-to-solid contact pairs
    for (int m = 0; m < numnodes_; m++)
    {
      int tempGID = ((pairs_[i]->element1())->node_ids())[m];
      Core::LinAlg::Matrix<3, 1> temppos = currentpositions[tempGID];

      // store updated nodal coordinates
      for (int n = 0; n < 3; n++) ele1pos(n, m) = temppos(n);
      // store updated nodal tangents
    }
    if (numnodalvalues_ == 2)
    {
      // Tangents: Loop over all nodes of element 1
      for (int m = 0; m < numnodes_; m++)
      {
        int tempGID = ((pairs_[i]->element1())->node_ids())[m];
        Core::LinAlg::Matrix<3, 1> temptan = currenttangents[tempGID];

        // store updated nodal tangents
        for (int n = 0; n < 3; n++) ele1pos(n + 3, m) = temptan(n);
      }
    }
    // Positions: Loop over all nodes of element 2
    for (int m = 0; m < numnodes_; m++)
    {
      int tempGID = ((pairs_[i]->element2())->node_ids())[m];
      Core::LinAlg::Matrix<3, 1> temppos = currentpositions[tempGID];
      // store updated nodal coordinates
      for (int n = 0; n < 3; n++) ele2pos(n, m) = temppos(n);
    }
    if (numnodalvalues_ == 2)
    {
      // Tangents: Loop over all nodes of element 2
      for (int m = 0; m < numnodes_; m++)
      {
        int tempGID = ((pairs_[i]->element2())->node_ids())[m];
        Core::LinAlg::Matrix<3, 1> temptan = currenttangents[tempGID];

        // store updated nodal tangents
        for (int n = 0; n < 3; n++) ele2pos(n + 3, m) = temptan(n);
      }
    }
    // finally update nodal positions in contact pair objects
    pairs_[i]->update_ele_pos(ele1pos, ele2pos);
  }
  // Update also the interpolated tangents if the tangentsmoothing is activated for Reissner beams
  auto smoothing =
      Teuchos::getIntegralValue<BeamContact::Smoothing>(sbeamcontact_, "BEAMS_SMOOTHING");
  if (smoothing != BeamContact::bsm_none)
  {
    for (int i = 0; i < (int)pairs_.size(); ++i)
    {
      pairs_[i]->update_ele_smooth_tangents(currentpositions);
    }
  }

  // Do the same for the beam-to-solid contact pairs
  for (int i = 0; i < (int)btsolpairs_.size(); ++i)
  {
    int numnodessol = ((btsolpairs_[i])->element2())->num_node();
    // temporary matrices to store nodal coordinates of each element
    Core::LinAlg::SerialDenseMatrix ele1pos(3 * numnodalvalues_, numnodes_);
    Core::LinAlg::SerialDenseMatrix ele2pos(3, numnodessol);
    // Positions: Loop over all nodes of element 1 (beam element)
    for (int m = 0; m < numnodes_; m++)
    {
      int tempGID = ((btsolpairs_[i]->element1())->node_ids())[m];
      Core::LinAlg::Matrix<3, 1> temppos = currentpositions[tempGID];

      // store updated nodal coordinates
      for (int n = 0; n < 3; n++) ele1pos(n, m) = temppos(n);
      // store updated nodal tangents
    }
    if (numnodalvalues_ == 2)
    {
      // Tangents: Loop over all nodes of element 1
      for (int m = 0; m < numnodes_; m++)
      {
        int tempGID = ((btsolpairs_[i]->element1())->node_ids())[m];
        Core::LinAlg::Matrix<3, 1> temptan = currenttangents[tempGID];

        // store updated nodal tangents
        for (int n = 0; n < 3; n++) ele1pos(n + 3, m) = temptan(n);
      }
    }
    // Positions: Loop over all nodes of element 2 (solid element)
    for (int m = 0; m < (btsolpairs_[i]->element2())->num_node(); m++)
    {
      int tempGID = ((btsolpairs_[i]->element2())->node_ids())[m];
      Core::LinAlg::Matrix<3, 1> temppos = currentpositions[tempGID];
      // store updated nodal coordinates
      for (int n = 0; n < 3; n++) ele2pos(n, m) = temppos(n);
    }

    // finally update nodal positions in contact pair objects
    btsolpairs_[i]->update_ele_pos(ele1pos, ele2pos);
  }

  return;
}

/*---------------------------------------------------------------------*
 |  Evaluate all pairs stored in the pair vectors            grill 10/14|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::evaluate_all_pairs(Teuchos::ParameterList timeintparams)
{
  // Begin: Determine maximal curvature occurring in complete beam discretization
  double kappa_max = 0.0;
  global_kappa_max_ = 0.0;

  for (int i = 0; i < problem_discret().num_my_col_elements(); i++)
  {
    Core::Elements::Element* element = problem_discret().l_col_element(i);
    const Core::Elements::ElementType& eot = element->element_type();
    if (eot == Discret::Elements::Beam3ebType::instance())
    {
      const Discret::Elements::Beam3eb* beam3ebelement =
          dynamic_cast<const Discret::Elements::Beam3eb*>(element);

      if (fabs(beam3ebelement->get_kappa_max()) > kappa_max)
        kappa_max = fabs(beam3ebelement->get_kappa_max());
    }
    else if (eot == Discret::Elements::Beam3rType::instance())
    {
      const Discret::Elements::Beam3r* beam3relement =
          dynamic_cast<const Discret::Elements::Beam3r*>(element);

      if (fabs(beam3relement->get_kappa_max()) > kappa_max)
        kappa_max = fabs(beam3relement->get_kappa_max());
    }
    else
    {
      // std::cout << "Warning: Calculation of kappa_max only implemented for beam3eb elements so
      // far!" << std::endl;
      kappa_max = 0.0;
    }
  }

  global_kappa_max_ = Core::Communication::max_all(kappa_max, get_comm());
  //  std::cout << "global_kappa_max_: " << global_kappa_max_ << std::endl;
  timeintparams.set("kappa_max", global_kappa_max_);
  // End: Determine maximal curvature occurring in complete beam discretization

  // Loop over all BTB contact pairs
  for (int i = 0; i < (int)pairs_.size(); ++i)
  {
    // only evaluate pair for those procs owning or ghosting at
    // least one node of one of the two elements of the pair
    int firsteleid = (pairs_[i]->element1())->id();
    int secondeleid = (pairs_[i]->element2())->id();
    bool firstisincolmap = col_elements()->my_gid(firsteleid);
    bool secondisincolmap = col_elements()->my_gid(secondeleid);

    // evaluate additional contact forces and stiffness
    if (firstisincolmap || secondisincolmap)
    {
      pairs_[i]->evaluate(*stiffc_, *fc_, currentpp_, contactpairmap_, timeintparams);

      // if active, get minimal gap of contact element pair
      if (pairs_[i]->get_contact_flag() == true)
      {
        std::vector<double> pairgaps = pairs_[i]->get_gap();
        for (int i = 0; i < (int)pairgaps.size(); i++)
        {
          double gap = pairgaps[i];
          if (gap < mintotalsimunconvgap_) mintotalsimunconvgap_ = gap;
        }
      }
    }
  }

  // Loop over all BTSOL contact pairs
  for (int i = 0; i < (int)btsolpairs_.size(); ++i)
  {
    // only evaluate pair for those procs owning or ghosting at
    // least one node of one of the two elements of the pair
    int firsteleid = (btsolpairs_[i]->element1())->id();
    int secondeleid = (btsolpairs_[i]->element2())->id();
    bool firstisincolmap = col_elements()->my_gid(firsteleid);
    bool secondisincolmap = col_elements()->my_gid(secondeleid);
    // evaluate additional contact forces and stiffness
    if (firstisincolmap || secondisincolmap)
    {
      btsolpairs_[i]->evaluate(*stiffc_, *fc_, btspp_);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  process the found element pairs and fill the corresponding
 |  BTB, BTSOL and BTSPH contact pair vectors                grill 09/14|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::fill_contact_pairs_vectors(
    const std::vector<std::vector<Core::Elements::Element*>> elementpairs)
{
  std::vector<std::vector<Core::Elements::Element*>> formattedelementpairs;
  formattedelementpairs.clear();

  // Besides beam-to-beam contact we also can handle beam-to-solid contact and beam to sphere
  // contact(will be implemented in the future). In all cases element 1 has to be the beam element.

  // All other element pairs (solid-solid, sphere-solid etc.) will be sorted out later.
  for (int i = 0; i < (int)elementpairs.size(); i++)
  {
    // if ele1 is a beam element we take the pair directly
    if (BeamInteraction::Utils::is_beam_element(*(elementpairs[i])[0]))
    {
      formattedelementpairs.push_back(elementpairs[i]);
    }
    // if ele1 is no beam element, but ele2 is one, we have to change the order
    else if (BeamInteraction::Utils::is_beam_element(*(elementpairs[i])[1]))
    {
      std::vector<Core::Elements::Element*> elementpairaux;
      elementpairaux.clear();
      elementpairaux.push_back((elementpairs[i])[1]);
      elementpairaux.push_back((elementpairs[i])[0]);
      formattedelementpairs.push_back(elementpairaux);
    }
  }
  // Determine type of applied beam elements and set the corresponding values for the member
  // variables numnodes_ and numnodaldofs_. This has only to be done once in the beginning, since
  // beam contact simulations are only possible when using beam elements of one type!
  if (!elementtypeset_ and formattedelementpairs.size() > 0)
  {
    set_element_type_and_distype((formattedelementpairs[0])[0]);
    elementtypeset_ = true;
  }

  // So far, all beam elements occurring in the pairs_ vector have to be of the same type.
  // This will be checked in the following lines.
  if ((int)formattedelementpairs.size() > 0)
  {
    // search for the first beam element in vector pairs
    const Core::Elements::ElementType& pair1_ele1_type =
        ((formattedelementpairs[0])[0])->element_type();
    for (int k = 0; k < (int)formattedelementpairs.size(); ++k)
    {
      const Core::Elements::ElementType& ele1_type =
          ((formattedelementpairs[k])[0])->element_type();
      const Core::Elements::ElementType& ele2_type =
          ((formattedelementpairs[k])[1])->element_type();

      // ele1 and ele2 (in case this is a beam element) have to be of the same type as ele1 of the
      // first pair
      if (ele1_type != pair1_ele1_type or
          (BeamInteraction::Utils::is_beam_element(*(formattedelementpairs[k])[1]) and
              ele2_type != pair1_ele1_type))
      {
        FOUR_C_THROW(
            "All contacting beam elements have to be of the same type (beam3k, beam3eb or "
            "beam3r). Check your input file!");
      }
    }
  }
  // Only the element pairs of formattedelementpairs (found in the contact search) which have not
  // been found in the last time step (i.e. which are not in oldpairs_) will be generated as new
  // Beam3contact instances. Pairs which already exist in oldpairs_ will simply be copied to pairs_.
  // This procedure looks a bit circumstantial at first glance: However, it is not possible to
  // solely use formattedelementpairs and to simply delete oldpairs_ at the end of a time step if
  // the new gap function definition is used, since the latter needs history variables of the last
  // time step which are stored in the oldpairs_ vector. Only beam-to-beam contact pairs (not
  // beam-to-solid or beam-to-sphere pairs) need this history information.
  for (int k = 0; k < (int)formattedelementpairs.size(); k++)
  {
    Core::Elements::Element* ele1 = (formattedelementpairs[k])[0];
    Core::Elements::Element* ele2 = (formattedelementpairs[k])[1];
    int currid1 = ele1->id();
    int currid2 = ele2->id();

    // beam-to-beam pair
    if (BeamInteraction::Utils::is_beam_element(*(formattedelementpairs[k])[1]))
    {
      bool foundlasttimestep = false;
      bool isalreadyinpairs = false;

      if (contactpairmap_.find(std::make_pair(currid1, currid2)) != contactpairmap_.end())
        isalreadyinpairs = true;

      if (oldcontactpairmap_.find(std::make_pair(currid1, currid2)) != oldcontactpairmap_.end())
        foundlasttimestep = true;

      if (!isalreadyinpairs and foundlasttimestep)
      {
        pairs_.push_back(oldcontactpairmap_[std::make_pair(currid1, currid2)]);
        if (currid1 < currid2)
          contactpairmap_[std::make_pair(currid1, currid2)] = pairs_[pairs_.size() - 1];
        else
          FOUR_C_THROW("Element 1 has to have the smaller element-ID. Adapt your contact search!");

        isalreadyinpairs = true;
      }

      if (!isalreadyinpairs)
      {
        // Add new contact pair object: The auxiliary_instance of the abstract class
        // Beam3contactinterface is only needed here in order to call the function Impl() which
        // creates an instance of the templated class Beam3contactnew<numnodes, numnodalvalues> !
        pairs_.push_back(CONTACT::Beam3contactinterface::impl(numnodes_, numnodalvalues_,
            problem_discret(), bt_sol_discret(), dofoffsetmap_, ele1, ele2, sbeamcontact_));
        if (currid1 <= currid2)
          contactpairmap_[std::make_pair(currid1, currid2)] = pairs_[pairs_.size() - 1];
        else
          FOUR_C_THROW("Element 1 has to have the smaller element-ID. Adapt your contact search!");
      }
    }
    // beam-to-solid contact pair
    else if (BeamInteraction::solid_contact_element(*(formattedelementpairs[k])[1]))
    {
      bool foundlasttimestep = false;
      bool isalreadyinpairs = false;

      if (btsolpairmap_.find(std::make_pair(currid1, currid2)) != btsolpairmap_.end())
        isalreadyinpairs = true;

      if (oldbtsolpairmap_.find(std::make_pair(currid1, currid2)) != oldbtsolpairmap_.end())
        foundlasttimestep = true;

      if (!isalreadyinpairs and foundlasttimestep)
      {
        btsolpairs_.push_back(oldbtsolpairmap_[std::make_pair(currid1, currid2)]);
        if (currid1 < currid2)
          btsolpairmap_[std::make_pair(currid1, currid2)] = btsolpairs_[btsolpairs_.size() - 1];
        else
          FOUR_C_THROW("Element 1 has to have the smaller element-ID. Adapt your contact search!");

        isalreadyinpairs = true;
      }

      if (!isalreadyinpairs)
      {
        // Add new contact pair object: The auxiliary_instance of the abstract class
        // Beam3contactinterface is only needed here in order to call the function Impl() which
        // creates an instance of the templated class Beam3contactnew<numnodes, numnodalvalues> !
        btsolpairs_.push_back(CONTACT::Beam3tosolidcontactinterface::impl(
            (formattedelementpairs[k])[1]->num_node(), numnodes_, numnodalvalues_,
            problem_discret(), bt_sol_discret(), dofoffsetmap_, ele1, ele2, sbeamcontact_));
        if (currid1 <= currid2)
          btsolpairmap_[std::make_pair(currid1, currid2)] = btsolpairs_[btsolpairs_.size() - 1];
        else
          FOUR_C_THROW("Element 1 has to have the smaller element-ID. Adapt your contact search!");
      }
    }
    else
    {
      FOUR_C_THROW(
          "ERROR: Unknown element type in beam contact pairs (none of BTB, BTSco, BTSmt, BTSPH)");
    }
  }

  // screen output
  int numpairs = 0;
  int numpairsthisproc = pairs_.size();

  numpairs = Core::Communication::sum_all(numpairsthisproc, pdiscret_.get_comm());

  if (Core::Communication::my_mpi_rank(pdiscret_.get_comm()) == 0)
    Core::IO::cout(Core::IO::standard)
        << "\t Total number of BTB contact pairs:     " << numpairs << Core::IO::endl;

  if (btsol_)
  {
    numpairs = 0;
    numpairsthisproc = btsolpairs_.size();

    numpairs = Core::Communication::sum_all(numpairsthisproc, pdiscret_.get_comm());

    if (Core::Communication::my_mpi_rank(pdiscret_.get_comm()) == 0)
      Core::IO::cout(Core::IO::standard)
          << "\t Total number of BTSOL contact pairs:    " << numpairs << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------*
 |  search possible contact element pairs                     popp 04/10|
 *----------------------------------------------------------------------*/
std::vector<std::vector<Core::Elements::Element*>> CONTACT::Beam3cmanager::brute_force_search(
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions, const double searchradius,
    const double sphericalsearchradius)
{
  //**********************************************************************
  // Steps of search for element pairs that MIGHT get into contact:
  //
  // 1) Find non-neighboring node pairs
  // 2) Compute distance between node pairs and compare with search radius
  // 3) Find non-neighboring element pairs based on node pairs
  // 4) Check if new pair already exists. If not we've found a new entry!
  //
  // NOTE: This is a brute force search! The time to search across n nodes
  // goes with n^2, which is not efficient at all....
  //**********************************************************************
  std::vector<std::vector<Core::Elements::Element*>> newpairs;
  newpairs.clear();

  //**********************************************************************
  // LOOP 1: column nodes (overlap = 1)
  // each processor looks for close nodes (directly connect with the corresponding node) for each of
  // these nodes
  //**********************************************************************
  for (int i = 0; i < col_nodes()->num_my_elements(); ++i)
  {
    // get global id, node itself and current position
    int firstgid = col_nodes()->gid(i);
    Core::Nodes::Node* firstnode = bt_sol_discret().g_node(firstgid);

    // TODO see also LOOP 2 below
    /* check whether node position has been stored in currentpositions previously;
     * if not, it most likely is a beam node which is NOT used for centerline interpolation
     * if so, we simply skip it because it does not have position (and tangent) DoFs */
    if (currentpositions.find(firstgid) == currentpositions.end())
    {
      if (BeamInteraction::Utils::is_beam_node(*firstnode) and
          !BeamInteraction::Utils::is_beam_centerline_node(*firstnode))
      {
        continue;
      }
      else
      {
        FOUR_C_THROW("this should not happen!");
      }
    }

    Core::LinAlg::Matrix<3, 1> firstpos = currentpositions[firstgid];

    // create storage for neighbouring nodes to be excluded.
    std::vector<int> neighbournodeids;
    // create storage for near nodes to be identified
    std::vector<int> NearNodesGIDs;

    // get the elements 'firstnode' is linked to
    // loop over all adjacent elements and their nodes
    for (auto ele : firstnode->adjacent_elements())
    {
      Core::Elements::Element* thisele = ele.user_element();
      for (int k = 0; k < thisele->num_node(); ++k)
      {
        int nodeid = thisele->node_ids()[k];
        if (nodeid == firstgid) continue;

        // add to neighbouring node vector
        neighbournodeids.push_back(nodeid);
      }
    }
    //**********************************************************************
    // LOOP 2: all nodes (fully overlapping column map)
    // each processor looks for close nodes within these nodes
    //**********************************************************************
    for (int j = 0; j < full_nodes()->num_my_elements(); ++j)
    {
      // get global node id and current position
      int secondgid = full_nodes()->gid(j);

      // TODO see comment above
      if (currentpositions.find(secondgid) == currentpositions.end()) continue;

      Core::LinAlg::Matrix<3, 1> secondpos = currentpositions[secondgid];

      // nothing to do for identical pair
      if (firstgid == secondgid) continue;

      // check if second node is neighbouring node
      bool neighbouring = false;
      for (int k = 0; k < (int)neighbournodeids.size(); ++k)
      {
        if (secondgid == neighbournodeids[k])
        {
          neighbouring = true;
          break;
        }
      }
      // compute distance by comparing firstpos <-> secondpos
      if (neighbouring == false)
      {
        Core::LinAlg::Matrix<3, 1> distance;
        for (int k = 0; k < 3; k++) distance(k) = secondpos(k) - firstpos(k);

        // nodes are near if distance < search radius
        if (distance.norm2() < searchradius or searchradius == -1.0)
          NearNodesGIDs.push_back(secondgid);
      }
    }
    // AT THIS POINT WE HAVE FOUND AND STORED ALL NODES CLOSE TO FIRSTNODE BESIDES THE DIRECTLY
    // CONNECTED NEIGHBOR NODES!

    //*********************************************************************
    // Up to now we have only considered nodes, but in the end we will need
    // to find element pairs, that might possibly get into contact. To find
    // these element pairs, we combine the elements around 'firstnode' with
    // all elements around each 'NearNodesGIDs'-node. Repetitions of pairs
    // and neighboring pairs will be rejected. For the remaining GIDs,
    // pointers on these elements are be created. With these pointers, the
    // beam3contact objects are set up and stored into the vector pairs_.
    //*********************************************************************
    // vectors of element ids
    std::vector<int> FirstElesGIDs;
    std::vector<int> SecondElesGIDs;
    // loop over all elements adjacent to firstnode
    for (auto ele : firstnode->adjacent_elements())
    {
      // insert into element vector
      FirstElesGIDs.push_back(ele.global_id());
    }

    // loop over ALL nodes close to first node
    for (int j = 0; j < (int)NearNodesGIDs.size(); ++j)
    {
      // node pointer
      Core::Nodes::Node* tempnode = bt_sol_discret().g_node(NearNodesGIDs[j]);
      // loop over all elements adjacent to tempnode
      for (auto ele : tempnode->adjacent_elements())
      {
        SecondElesGIDs.push_back(ele.global_id());
      }
    }
    // AT THIS POINT WE HAVE FOUND AND STORED ALL ELEMENTS CLOSE TO FIRSTNODE!

    //*********************************************************************
    // The combination and creation of beam3contact objects can now begin.
    // First of all we reject all second element GIDs, that occur twice and
    // generate a new vector 'SecondElesGIDsRej' where each GID occurs only
    // once. This vector will then be used for generating the pair objects.
    //*********************************************************************

    // initialize reduced vector of close elements
    std::vector<int> SecondElesGIDsRej;
    SecondElesGIDsRej.clear();

    // loop over all close elements
    for (int j = 0; j < (int)SecondElesGIDs.size(); ++j)
    {
      // check if this element gid occurs twice
      int tempGID = SecondElesGIDs[j];
      bool twice = false;

      // loop over all close elements again
      for (int k = j + 1; k < (int)SecondElesGIDs.size(); ++k)
      {
        if (tempGID == SecondElesGIDs[k]) twice = true;
      }

      // only insert in reduced vector if not yet there
      if (twice == false) SecondElesGIDsRej.push_back(tempGID);
    }

    // now finally create element pairs via two nested loops
    for (int j = 0; j < (int)FirstElesGIDs.size(); ++j)
    {
      // beam element pointer
      Core::Elements::Element* ele1 = bt_sol_discret().g_element(FirstElesGIDs[j]);

      // node ids adjacent to this element
      const int* NodesEle1 = ele1->node_ids();

      // loop over all close elements
      for (int k = 0; k < (int)SecondElesGIDsRej.size(); ++k)
      {
        // get and cast a pointer on an element
        Core::Elements::Element* ele2 = bt_sol_discret().g_element(SecondElesGIDsRej[k]);

        // close element id
        const int* NodesEle2 = ele2->node_ids();

        // check if elements are neighbouring (share one common node)
        bool elements_neighbouring = false;
        for (int m = 0; m < ele1->num_node(); ++m)
        {
          for (int n = 0; n < ele2->num_node(); ++n)
          {
            // neighbouring if they share one common node
            if (NodesEle1[m] == NodesEle2[n]) elements_neighbouring = true;
          }
        }

        // Check if the pair 'jk' already exists in newpairs
        bool foundbefore = false;
        for (int m = 0; m < (int)newpairs.size(); ++m)
        {
          int id1 = (newpairs[m])[0]->id();
          int id2 = (newpairs[m])[1]->id();
          int currid1 = FirstElesGIDs[j];
          int currid2 = SecondElesGIDsRej[k];

          // already exists if can be found in newpairs
          if ((id1 == currid1 && id2 == currid2) || (id1 == currid2 && id2 == currid1))
            foundbefore = true;
        }

        // if NOT neighbouring and NOT found before
        // create new beam3contact object and store it into pairs_

        // Here we additionally apply the method close_midpoint_distance which sorts out all pairs
        // with a midpoint distance larger than sphericalsearchradius. Thus with this additional
        // method the search is based on spherical bounding boxes and node on node distances any
        // longer. The radius of these spheres is sphericalsearchradius/2.0, the center of such a
        // sphere is (r1+r2)/2, with r1 and r2 representing the nodal positions.
        if (!elements_neighbouring && !foundbefore &&
            close_midpoint_distance(ele1, ele2, currentpositions, sphericalsearchradius))
        {
          std::vector<Core::Elements::Element*> contactelementpair;
          contactelementpair.clear();
          if (ele1->id() < ele2->id())
          {
            contactelementpair.push_back(ele1);
            contactelementpair.push_back(ele2);
          }
          else
          {
            contactelementpair.push_back(ele2);
            contactelementpair.push_back(ele1);
          }
          newpairs.push_back(contactelementpair);
        }
      }
    }
  }  // LOOP 1
  return newpairs;
}

/*-------------------------------------------------------------------- -*
 |  Compute search radius from discretization data            popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::compute_search_radius()
{
  // some local variables
  double charactlength = 0.0;
  double globalcharactlength = 0.0;
  double maxelelength = 0.0;

  // look for maximum element length in the whole discretization
  get_max_ele_length(maxelelength);

  // select characeteristic length
  if (maxeleradius_ > maxelelength)
    charactlength = maxeleradius_;
  else
    charactlength = maxelelength;

  // communicate among all procs to find the global maximum
  globalcharactlength = Core::Communication::max_all(charactlength, get_comm());

  // Compute the search radius. This one is only applied to determine
  // close pairs considering the node-to-node distances.
  double nodalsearchfac = 3.0;
  searchradius_ = nodalsearchfac * (2.0 * searchboxinc_ + globalcharactlength);

  // In a second step spherical search boxes are applied which consider
  // the midpoint-to-midpoint distance. In the first (nodal-based) search step
  // it has to be ensured that all pairs relevant for this second search step will
  // be found. The most critical case (i.e. the case, where the midpoints are as close as
  // possible but the node distances are as large as possible) is the case where to (straight)
  // beams are perpendicular to each other and the beam midpoint coincide with the closest points
  // between these two beams. One can show, that in this case a value of nodalsearchfac=2.0 is
  // sufficient to find all relevant pairs in the first step. This factor should also be sufficient,
  // if the two beam elements are deformed (maximal assumed deformation of a beam element is a half
  // circle!). To be on the safe side (the number of elements found in the first search step is not
  // very relevant for the overall efficiency), we choose a factor of nodalsearchfac = 3.0.
  sphericalsearchradius_ = 2.0 * searchboxinc_ + globalcharactlength;

  // some information for the user
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Penalty parameter      = " << currentpp_ << std::endl;
    std::cout << "BTS-Penalty parameter  = " << btspp_ << std::endl;
    std::cout << "Maximum element radius = " << maxeleradius_ << std::endl;
    std::cout << "Maximum element length = " << maxelelength << std::endl;
    std::cout << "Search radius          = " << searchradius_ << std::endl << std::endl;
  }

  return;
}

/*-------------------------------------------------------------------- -*
 |  Find maximum element radius in discretization             popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::set_min_max_ele_radius()
{
  mineleradius_ = 0.0;
  maxeleradius_ = 0.0;

  bool minbeamradiusinitialized = false;

  // loop over all elements in row map
  for (int i = 0; i < row_elements()->num_my_elements(); ++i)
  {
    // get pointer onto element
    int gid = row_elements()->gid(i);
    Core::Elements::Element* thisele = bt_sol_discret().g_element(gid);

    double eleradius = 0.0;

    if (BeamInteraction::Utils::is_beam_element(*thisele) or
        BeamInteraction::Utils::is_rigid_sphere_element(*thisele))
    {  // compute eleradius from moment of inertia
      // (RESTRICTION: CIRCULAR CROSS SECTION !!!)
      eleradius = BeamInteraction::calc_ele_radius(thisele);

      // if current radius is larger than maximum radius -> update
      if (eleradius > maxeleradius_) maxeleradius_ = eleradius;

      // Initialize minbeamradius- with the first radius we get; otherwise its value will remain
      // 0.0!
      if (!minbeamradiusinitialized)
      {
        mineleradius_ = eleradius;
        minbeamradiusinitialized = true;
      }

      // if current radius is smaller than minimal radius -> update
      if (eleradius < mineleradius_) mineleradius_ = eleradius;
    }
  }

  return;
}

/*-------------------------------------------------------------------- -*
 |  Find maximum element length in discretization             popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::get_max_ele_length(double& maxelelength)
{
  // loop over all elements in row map
  for (int i = 0; i < row_elements()->num_my_elements(); i++)
  {
    // get pointer onto element
    int gid = row_elements()->gid(i);
    Core::Elements::Element* thisele = bt_sol_discret().g_element(gid);

    double elelength = 0.0;

    if (BeamInteraction::Utils::is_beam_element(*thisele))
    {
      // get global IDs of edge nodes and pointers
      int node0_gid = thisele->node_ids()[0];
      int node1_gid = thisele->node_ids()[1];
      Core::Nodes::Node* node0 = bt_sol_discret().g_node(node0_gid);
      Core::Nodes::Node* node1 = bt_sol_discret().g_node(node1_gid);

      // get coordinates of edge nodes
      std::vector<double> x_n0(3);
      std::vector<double> x_n1(3);
      for (int j = 0; j < 3; ++j)
      {
        x_n0[j] = node0->x()[j];
        x_n1[j] = node1->x()[j];
      }

      // compute distance vector and length
      // (APPROXIMATION FOR HIGHER-ORDER ELEMENTS !!!)
      std::vector<double> dist(3);
      for (int j = 0; j < 3; ++j) dist[j] = x_n0[j] - x_n1[j];
      elelength = sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]);
    }
    else if (BeamInteraction::Utils::is_rigid_sphere_element(*thisele))
      continue;  // elelength does not apply for rigid spheres, radius is already considered in
                 // MaxEleRadius(), so simply do nothing here
    else
      FOUR_C_THROW(
          "The function get_max_ele_length is only defined for beam elements and rigid sphere "
          "elements!");

    // if current length is larger than maximum length -> update
    if (elelength > maxelelength) maxelelength = elelength;
  }

  return;
}

/*-------------------------------------------------------------------- -*
 |  Update contact forces at the end of time step             popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::update(
    const Core::LinAlg::Vector<double>& disrow, const int& timestep, const int& newtonstep)
{
  // store values of fc_ into fcold_ (generalized alpha)
  fcold_->update(1.0, *fc_, 0.0);

  double disold_L2 = 0.0;
  dis_old_->norm_2(&disold_L2);

  // calculate (dis_old_-dis_):
  dis_old_->update(-1.0, *dis_, 1.0);
  // calculate inf-norm of (dis_old_-dis_)
  dis_old_->norm_inf(&maxdeltadisp_);
  // invert the last step and get again dis_old_
  dis_old_->update(1.0, *dis_, 1.0);
  // update dis_old_ -> dis_
  dis_old_->update(1.0, *dis_, 0.0);

  if (maxdeltadisp_ > totalmaxdeltadisp_ and
      disold_L2 > 1.0e-12)  // don't consider the first time step where disold_L2=0!
    totalmaxdeltadisp_ = maxdeltadisp_;

  // If the original gap function definition is applied, the displacement per time is not allowed
  // to be larger than the smallest beam cross section radius occurring in the discretization!
  bool newgapfunction = beam_contact_parameters().get<bool>("BEAMS_NEWGAP");
  if (!newgapfunction)
  {
    double maxdeltadisscalefac = sbeamcontact_.get<double>("BEAMS_MAXDELTADISSCALEFAC", 1.0);
    // TODO: shall we allow for larger displacements in the first time step in general?
    if (maxdeltadisp_ > maxdeltadisscalefac * mineleradius_ and timestep != 1)
    {
      // std::cout << "Minimal element radius: " << mineleradius_ << std::endl;
      // std::cout << "Maximal displacement per time step: " << maxdeltadisp_ << std::endl;
      // FOUR_C_THROW("Displacement increment per time step larger than smallest beam element
      // radius, "
      //        "but newgapfunction_ flag is not set. Choose smaller time step!");
    }
  }
  // std::cout << "mineleradius_: " << mineleradius_ << std::endl;
  // std::cout << "totalmaxdeltadisp_: " << totalmaxdeltadisp_ << std::endl

  // First, we check some restrictions concerning the new gap function definition
  for (int i = 0; i < (int)pairs_.size(); ++i)
  {
    // only relevant if current pair is active
    if (pairs_[i]->get_contact_flag() == true)
    {
      if (pairs_[i]->get_new_gap_status() == true)
      {
        // Necessary when using the new gap function definition (ngf_=true) for very slender beams
        // in order to avoid crossing: For very low penalty parameters and very slender beams it may
        // happen that in the converged configuration the remaining penetration is larger than the
        // sum of the beam radii (R1+R2), i.e. the beam centerlines remain crossed even in the
        // converged configuration. In this case the sign of the normal vector normal_ has to be
        // inverted at the end of the time step, since this quantity is stored in normal_old_
        // afterwards. Otherwise the contact force would be applied in the wrong direction and the
        // beams could cross!
        pairs_[i]->invert_normal();
        Teuchos::ParameterList ioparams = Global::Problem::instance()->io_params();
        if (!Core::Communication::my_mpi_rank(pdiscret_.get_comm()) &&
            ioparams.get<int>("STDOUTEVERY", 0))
          std::cout << "      Warning: Penetration to large, choose higher penalty parameter!"
                    << std::endl;
      }

      if (pairs_[i]->get_shift_status() == true)
      {
        // In case the contact points of two beams are identical (i.e. r1=r2), the nodal coordinates
        // of one beam are shifted by a small predefined value in order to enable evaluation of the
        // contact pair. This makes the Newton scheme more robust. However, in the converged
        // configuration we want to have the real nodal positions for all contact pairs!!!
        FOUR_C_THROW(
            "Contact pair with identical contact points (i.e. r1=r2) not possible in the converged "
            "configuration!");
      }
    }
  }

  // set normal_old_=normal_ for all contact pairs at the end of the time step
  update_all_pairs();

  // print some data to screen
  console_output();
  // store pairs_ in oldpairs_ to be available in next time step
  // this is needed for the new gapfunction definition and also for the output at the end of an time
  // step
  oldpairs_.clear();
  oldpairs_.resize(0);
  oldpairs_ = pairs_;

  oldcontactpairmap_.clear();
  oldcontactpairmap_ = contactpairmap_;

  oldbtsolpairs_.clear();
  oldbtsolpairs_.resize(0);
  oldbtsolpairs_ = btsolpairs_;

  oldbtsolpairmap_.clear();
  oldbtsolpairmap_ = btsolpairmap_;

  // clear potential contact pairs
  contactpairmap_.clear();
  btsolpairmap_.clear();

  pairs_.clear();
  pairs_.resize(0);

  btsolpairs_.clear();
  btsolpairs_.resize(0);

  return;
}

/*----------------------------------------------------------------------*
 |  Compute rotation matrix R                                cyron 01/09|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::transform_angle_to_triad(
    Core::LinAlg::SerialDenseVector& theta, Core::LinAlg::SerialDenseMatrix& R)
{
  // compute spin matrix according to Crisfield Vol. 2, equation (16.8)
  Core::LinAlg::SerialDenseMatrix spin(3, 3);
  compute_spin(spin, theta);

  // nompute norm of theta
  double theta_abs = Core::LinAlg::norm2(theta);

  // build an identity matrix
  Core::LinAlg::SerialDenseMatrix identity(3, 3);
  for (int i = 0; i < 3; i++) identity(i, i) = 1.0;

  // square of spin matrix
  Core::LinAlg::SerialDenseMatrix spin2(3, 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) spin2(i, k) += spin(i, j) * spin(j, k);


  // compute rotation matrix according to Crisfield Vol. 2, equation (16.22)
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      R(i, j) = identity(i, j) + spin(i, j) * (sin(theta_abs)) / theta_abs +
                (1 - (cos(theta_abs))) / (pow(theta_abs, 2)) * spin2(i, j);

  return;
}

/*----------------------------------------------------------------------*
 |  Compute spin (private)                                    cyron 01/09|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::compute_spin(
    Core::LinAlg::SerialDenseMatrix& spin, Core::LinAlg::SerialDenseVector& rotationangle)
{
  // initialization
  const double spinscale = 1.0;
  for (int i = 0; i < rotationangle.length(); ++i) rotationangle[i] *= spinscale;

  // initialize spin with zeros
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) spin(i, j) = 0;

  // fill spin with values (see Crisfield Vol. 2, equation (16.8))
  spin(0, 0) = 0;
  spin(0, 1) = -rotationangle[2];
  spin(0, 2) = rotationangle[1];
  spin(1, 0) = rotationangle[2];
  spin(1, 1) = 0;
  spin(1, 2) = -rotationangle[0];
  spin(2, 0) = -rotationangle[1];
  spin(2, 1) = rotationangle[0];
  spin(2, 2) = 0;

  return;
}

/*----------------------------------------------------------------------*
 |  Update contact constraint norm                            popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::update_constr_norm()
{
  // Next we want to find out the maximal and minimal gap
  // We have to distinguish between maximal and minimal gap, since our penalty
  // force law can already become active for positive gaps.
  double maxgap = -1000.0;
  double maxgap_cp = -1000.0;
  double maxgap_gp = -1000.0;
  double maxgap_ep = -1000.0;
  double maxallgap = -1000.0;
  double maxallgap_cp = -1000.0;
  double maxallgap_gp = -1000.0;
  double maxallgap_ep = -1000.0;
  double mingap = 1000.0;
  double mingap_cp = 1000.0;
  double mingap_gp = 1000.0;
  double mingap_ep = 1000.0;
  double minallgap = 1000.0;
  double minallgap_cp = 1000.0;
  double minallgap_gp = 1000.0;
  double minallgap_ep = 1000.0;
  double maxrelgap = -1000.0;
  double maxallrelgap = -1000.0;
  double minrelgap = 1000.0;
  double minallrelgap = 1000.0;

  // Clear class variable
  totpenaltyenergy_ = 0.0;
  double proclocal_penaltyenergy = 0.0;

  // Calculate contact work
  double deltapenaltywork = 0.0;
  Core::LinAlg::Vector<double> delta_disp(*dis_);
  delta_disp.update(-1.0, *dis_old_, 1.0);

  Core::LinAlg::Vector<double> fc_alpha(*fc_);
  fc_alpha.update(alphaf_, *fcold_, 1.0 - alphaf_);

  delta_disp.dot(fc_alpha, &deltapenaltywork);

  deltapenaltywork = -deltapenaltywork;

  totpenaltywork_ += deltapenaltywork;

  // loop over all pairs to find all gaps
  for (int i = 0; i < (int)pairs_.size(); ++i)
  {
    // make sure to evaluate each pair only once
    int firsteleid = (pairs_[i]->element1())->id();
    bool firstisinrowmap = row_elements()->my_gid(firsteleid);

    // only relevant if current pair is active
    if (pairs_[i]->get_contact_flag() and firstisinrowmap)
    {
      // Update penalty energy
      proclocal_penaltyenergy += pairs_[i]->get_energy();

      // get smaller radius of the two elements:
      double smallerradius = 0.0;
      double radius1 = BeamInteraction::calc_ele_radius(pairs_[i]->element1());
      double radius2 = BeamInteraction::calc_ele_radius(pairs_[i]->element2());
      if (radius1 < radius2)
        smallerradius = radius1;
      else
        smallerradius = radius2;

      std::vector<double> pairgaps = pairs_[i]->get_gap();
      int numcps = pairs_[i]->get_num_cps();
      int numgps = pairs_[i]->get_num_gps();
      int numeps = pairs_[i]->get_num_eps();

      if (pairgaps.size() != (unsigned)(numcps + numgps + numeps))
        FOUR_C_THROW(
            "size mismatch! total "
            "number of gaps unequal sum of individual contact type gaps");

      for (int i = 0; i < (int)pairgaps.size(); i++)
      {
        double gap = pairgaps[i];

        if (i < numcps)
        {
          if (gap > maxgap_cp) maxgap_cp = gap;

          if (gap < mingap_cp) mingap_cp = gap;
        }
        else if (i < numcps + numgps)
        {
          if (gap > maxgap_gp) maxgap_gp = gap;

          if (gap < mingap_gp) mingap_gp = gap;
        }
        else if (i < numcps + numgps + numeps)
        {
          if (gap > maxgap_ep) maxgap_ep = gap;

          if (gap < mingap_ep) mingap_ep = gap;
        }

        double relgap = gap / smallerradius;

        if (gap > maxgap) maxgap = gap;

        if (gap < mingap) mingap = gap;

        if (relgap > maxrelgap) maxrelgap = relgap;

        if (relgap < minrelgap) minrelgap = relgap;
      }
    }
  }

  // So far, we only have the processor local extrema, but we want the extrema of the whole problem
  // As long as the beam contact discretization is full overlapping, all pairs are stored in all
  // procs and don't need this procedure. However, for future applications (i.e. when abstain from a
  // fully overlapping discretization) it might be useful.
  maxallgap = Core::Communication::max_all(maxgap, get_comm());
  maxallgap_cp = Core::Communication::max_all(maxgap_cp, get_comm());
  maxallgap_gp = Core::Communication::max_all(maxgap_gp, get_comm());
  maxallgap_ep = Core::Communication::max_all(maxgap_ep, get_comm());
  minallgap = Core::Communication::min_all(mingap, get_comm());
  minallgap_cp = Core::Communication::min_all(mingap_cp, get_comm());
  minallgap_gp = Core::Communication::min_all(mingap_gp, get_comm());
  minallgap_ep = Core::Communication::min_all(mingap_ep, get_comm());
  maxallrelgap = Core::Communication::max_all(maxrelgap, get_comm());
  minallrelgap = Core::Communication::min_all(minrelgap, get_comm());

  totpenaltyenergy_ = Core::Communication::sum_all(proclocal_penaltyenergy, get_comm());

  // So far, we have determined the extrema of the current time step. Now, we want to determine the
  // extrema of the total simulation:
  if (maxallgap > maxtotalsimgap_) maxtotalsimgap_ = maxallgap;
  if (maxallgap_cp > maxtotalsimgap_cp_) maxtotalsimgap_cp_ = maxallgap_cp;
  if (maxallgap_gp > maxtotalsimgap_gp_) maxtotalsimgap_gp_ = maxallgap_gp;
  if (maxallgap_ep > maxtotalsimgap_ep_) maxtotalsimgap_ep_ = maxallgap_ep;

  if (minallgap < mintotalsimgap_) mintotalsimgap_ = minallgap;
  if (minallgap_cp < mintotalsimgap_cp_) mintotalsimgap_cp_ = minallgap_cp;
  if (minallgap_gp < mintotalsimgap_gp_) mintotalsimgap_gp_ = minallgap_gp;
  if (minallgap_ep < mintotalsimgap_ep_) mintotalsimgap_ep_ = minallgap_ep;

  if (maxallrelgap > maxtotalsimrelgap_) maxtotalsimrelgap_ = maxallrelgap;

  if (minallrelgap < mintotalsimrelgap_) mintotalsimrelgap_ = minallrelgap;

  // Set class variable
#ifdef RELCONSTRTOL
  constrnorm_ = fabs(minallrelgap);
#else
  constrnorm_ = fabs(minallgap);
#endif

  // TODO: Adapt this, as soon as we have a concrete implementation of beam-to-solid contact element
  // pairs
  btsolconstrnorm_ = 0.0;

  // print results to screen
  Teuchos::ParameterList ioparams = Global::Problem::instance()->io_params();
  if (Core::Communication::my_mpi_rank(get_comm()) == 0 && ioparams.get<int>("STDOUTEVERY", 0))
  {
    Core::IO::cout(Core::IO::debug)
        << Core::IO::endl
        << "      ***********************************BTB************************************"
        << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << "      Penalty parameter                = " << currentpp_ << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << "      Minimal current Gap              = " << minallgap << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << "      Minimal current rel. Gap         = " << minallrelgap << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << "      Current Constraint Norm          = " << constrnorm_ << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << "      Maximal current Gap              = " << maxallgap << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << "      Maximal current rel. Gap         = " << maxallrelgap << Core::IO::endl;

    if ((int)btsolpairs_.size())
    {
      Core::IO::cout(Core::IO::debug)
          << Core::IO::endl
          << "      ***********************************BTSOL**********************************"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "      BTSOL-Penalty parameter     = " << btspp_ << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "      Current Constraint Norm   = " << btsolconstrnorm_ << Core::IO::endl;
    }
    Core::IO::cout(Core::IO::debug)
        << "      **************************************************************************"
        << Core::IO::endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Shift normal vector to "normal_old_"                     meier 10/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::update_all_pairs()
{
  // loop over all potential contact pairs
  for (int i = 0; i < (int)pairs_.size(); ++i) pairs_[i]->update_class_variables_step();

  // loop over all potential beam to solid contact pairs
  for (int i = 0; i < (int)btsolpairs_.size(); ++i) btsolpairs_[i]->update_class_variables_step();
}

/*----------------------------------------------------------------------*
 |  Print active set                                          popp 04/10|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::console_output()
{
  Teuchos::ParameterList ioparams = Global::Problem::instance()->io_params();
  if (ioparams.get<int>("STDOUTEVERY", 0))
  {
    // begin output
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "\n    Active contact "
             "set------------------------------------------------------------\n";
      Core::IO::cout(Core::IO::verbose)
          << "    ID1            ID2              T xi       eta      angle    gap         force\n";
    }
    Core::Communication::barrier(get_comm());

    double maxangle = 0.0;
    double minangle = 90.0;
    double mincpgap = 1000.0;
    double mingpgap = 1000.0;
    double minepgap = 1000.0;
    double maxcpgap = -1000.0;
    double maxgpgap = -1000.0;
    double maxepgap = -1000.0;
    int numperpc = 0;
    int numparc = 0;
    int numepc = 0;
    int numperpc_transitions = 0;

#ifdef PRINTGAPFILE
    double error = 0.0;
    int gapsize = 0;
    double refgap = -0.002;
#endif

    double perpshiftangle1 = sbeamcontact_.get<double>("BEAMS_PERPSHIFTANGLE1");
    double perpshiftangle2 = sbeamcontact_.get<double>("BEAMS_PERPSHIFTANGLE2");

#ifdef PRINTGAPSOVERLENGTHFILE
    step_++;
    std::ostringstream filename;
    filename << "gp_gapsandforces_1x49filaments_consistentaxialtension_40elefil_" << step_
             << ".txt";

    // do output to file in c-style
    FILE* fp = nullptr;
    std::stringstream filecontent;

    // if(firststep_)
    if (true)
    {
      // open file to write output data into
      fp = fopen(filename.str().c_str(), "w");
      filecontent << "ID1 ID2 xi eta gap force angles\n";
      firststep_ = false;
    }
    else
    {
      // open file to write output data into
      fp = fopen(filename.str().c_str(), "a");
    }
#endif

    // loop over all pairs
    for (int i = 0; i < (int)pairs_.size(); ++i)
    {
      // check if this pair is active
      if (pairs_[i]->get_contact_flag())
      {
        // make sure to print each pair only once
        int firsteleid = (pairs_[i]->element1())->id();
        bool firstisinrowmap = row_elements()->my_gid(firsteleid);

        // abbreviations
        int id1 = (pairs_[i]->element1())->id();
        int id2 = (pairs_[i]->element2())->id();
        std::vector<double> gaps = pairs_[i]->get_gap();
        std::vector<int> types = pairs_[i]->get_contact_type();
        std::vector<double> forces = pairs_[i]->get_contact_force();
        std::vector<double> angles = pairs_[i]->get_contact_angle();
        std::vector<std::pair<double, double>> closestpoints = pairs_[i]->get_closest_point();
        std::pair<int, int> numsegments = pairs_[i]->get_num_segments();
        std::vector<std::pair<int, int>> segmentids = pairs_[i]->get_segment_ids();

        // print some output (use printf-method for formatted output)
        if (firstisinrowmap)
        {
          for (int j = 0; j < (int)gaps.size(); j++)
          {
#ifdef PRINTGAPFILE
            double gap = gaps[j];
            error += fabs((gap - refgap) / refgap);
            gapsize++;
#endif

            if (fabs(angles[j] / std::numbers::pi * 180.0) > maxangle)
              maxangle = fabs(angles[j] / std::numbers::pi * 180.0);

            if (fabs(angles[j] / std::numbers::pi * 180.0) < minangle)
              minangle = fabs(angles[j] / std::numbers::pi * 180.0);

            if (types[j] == 0)
            {
              if (gaps[j] < mincpgap) mincpgap = gaps[j];
              if (gaps[j] > maxcpgap) maxcpgap = gaps[j];
              numperpc++;

              if (fabs(angles[j] / std::numbers::pi * 180.0) < perpshiftangle2 and
                  fabs(angles[j] / std::numbers::pi * 180.0) > perpshiftangle1)
                numperpc_transitions++;
            }

            if (types[j] == 1)
            {
              if (gaps[j] < mingpgap) mingpgap = gaps[j];
              if (gaps[j] > maxgpgap) maxgpgap = gaps[j];
              numparc++;

#ifdef PRINTGAPSOVERLENGTHFILE
              // if(id1>=10 and id1 <=19 and id2>=20 and id2<=29)
              if (id1 >= 320 and id1 <= 359 and id2 >= 360 and id2 <= 399)
                filecontent << id1 << " " << id2 << " " << closestpoints[j].first << " "
                            << closestpoints[j].second << " " << gaps[j] << " " << forces[j] << " "
                            << angles[j] / std::numbers::pi * 180.0 << "\n";
#endif
            }

            if (types[j] == 2)
            {
              if (gaps[j] < minepgap) minepgap = gaps[j];
              if (gaps[j] > maxepgap) maxepgap = gaps[j];
              numepc++;
            }

            Core::IO::cout(Core::IO::verbose)
                << "    " << std::setw(5) << std::left << id1 << "(" << std::setw(3) << std::right
                << segmentids[j].first + 1 << "/" << std::setw(3) << numsegments.first << ")"
                << " " << std::setw(5) << std::left << id2 << "(" << std::setw(3) << std::right
                << segmentids[j].second + 1 << "/" << std::setw(3) << numsegments.second << ")"
                << "   " << types[j] << " " << std::setw(9) << std::left << std::setprecision(2)
                << closestpoints[j].first << std::setw(9) << std::left << std::setprecision(2)
                << closestpoints[j].second << std::setw(9) << std::left << std::setprecision(3)
                << angles[j] / std::numbers::pi * 180.0 << std::setw(12) << std::left
                << std::scientific << gaps[j] << std::setw(12) << std::left << std::scientific
                << forces[j] << std::setprecision(6) << std::resetiosflags(std::ios::scientific)
                << std::right << Core::IO::endl
                << Core::IO::flush;
          }
        }
      }
    }

    Core::Communication::barrier(get_comm());

#ifdef PRINTGAPSOVERLENGTHFILE
    // write content into file and close it
    fprintf(fp, filecontent.str().c_str());
    fclose(fp);
#endif

    // Calculate sum over all procs
    double sumpro_maxangle = 0.0;
    double sumpro_minangle = 0.0;
    double sumpro_mincpgap = 0.0;
    double sumpro_mingpgap = 0.0;
    double sumpro_minepgap = 0.0;
    double sumpro_maxcpgap = 0.0;
    double sumpro_maxgpgap = 0.0;
    double sumpro_maxepgap = 0.0;
    int sumpro_numperpc = 0;
    int sumpro_numparc = 0;
    int sumpro_numepc = 0;
    int sumpro_numperpc_transitions = 0;

    sumpro_maxangle = Core::Communication::max_all(maxangle, get_comm());
    sumpro_minangle = Core::Communication::min_all(minangle, get_comm());
    sumpro_mincpgap = Core::Communication::min_all(mincpgap, get_comm());
    sumpro_mingpgap = Core::Communication::min_all(mingpgap, get_comm());
    sumpro_minepgap = Core::Communication::min_all(minepgap, get_comm());
    sumpro_maxcpgap = Core::Communication::max_all(maxcpgap, get_comm());
    sumpro_maxgpgap = Core::Communication::max_all(maxgpgap, get_comm());
    sumpro_maxepgap = Core::Communication::max_all(maxepgap, get_comm());
    sumpro_numperpc = Core::Communication::sum_all(numperpc, get_comm());
    sumpro_numparc = Core::Communication::sum_all(numparc, get_comm());
    sumpro_numepc = Core::Communication::sum_all(numepc, get_comm());
    sumpro_numperpc_transitions = Core::Communication::sum_all(numperpc_transitions, get_comm());

#ifdef PRINTNUMCONTACTSFILE
    if (Core::Communication::my_mpi_rank(Comm()) == 0)
    {
      // TODO
      std::ostringstream filename;
      filename << "activecontacts_minmaxgapangle_statmech_37filaments_noendpoints.txt";

      // do output to file in c-style
      FILE* fp = nullptr;
      std::stringstream filecontent;

      if (firststep_)
      {
        // open file to write output data into
        fp = fopen(filename.str().c_str(), "w");
        filecontent << "perpcontacts parcontacts epcontacts pertransitions minangle maxangle "
                       "minperpgap minpargap minepgap global_kappa_max\n";

        firststep_ = false;
      }
      else
      {
        // open file to write output data into
        fp = fopen(filename.str().c_str(), "a");
      }

      filecontent << sumpro_numperpc << " " << sumpro_numparc << " " << sumpro_numepc << " "
                  << sumpro_numperpc_transitions << " ";
      filecontent << sumpro_minangle << " " << sumpro_maxangle << " " << sumpro_mincpgap << " "
                  << sumpro_mingpgap << " " << sumpro_minepgap << " " << global_kappa_max_ << "\n";

      // write content into file and close it
      fprintf(fp, filecontent.str().c_str());
      fclose(fp);
    }
#endif

#ifdef PRINTGAPFILE
    error = error / gapsize;
    std::cout << "error: " << error << std::endl;

    std::ostringstream filename;
    filename << "gaps.txt";

    // do output to file in c-style
    FILE* fp = nullptr;
    std::stringstream filecontent;

    if (firststep_)
    {
      // open file to write output data into
      fp = fopen(filename.str().c_str(), "w");
      filecontent << "gaperrors\n";
      firststep_ = false;
    }
    else
    {
      // open file to write output data into
      fp = fopen(filename.str().c_str(), "a");
    }

    filecontent << error << "\n";

    // write content into file and close it
    fprintf(fp, filecontent.str().c_str());
    fclose(fp);
#endif

    // print results to screen
    Teuchos::ParameterList ioparams = Global::Problem::instance()->io_params();
    if (Core::Communication::my_mpi_rank(get_comm()) == 0 && ioparams.get<int>("STDOUTEVERY", 0))
    {
      Core::IO::cout(Core::IO::standard)
          << "\n    Number of Point-to-Point Contact Pairs: " << sumpro_numperpc << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "    Number of Point Contacts in Transition Range: " << sumpro_numperpc_transitions
          << Core::IO::endl;
      Core::IO::cout(Core::IO::standard)
          << "    Number of Line-to-Line Contact Pairs: " << sumpro_numparc << Core::IO::endl;
      Core::IO::cout(Core::IO::standard)
          << "    Number of Endpoint Contact Pairs: " << sumpro_numepc << Core::IO::endl;

      Core::IO::cout(Core::IO::verbose)
          << "    Minimal contact angle: " << sumpro_minangle << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "    Maximal contact angle: " << sumpro_maxangle << Core::IO::endl;

      Core::IO::cout(Core::IO::verbose)
          << "    Minimal current Point-to-Point gap: " << sumpro_mincpgap << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "    Minimal current Line-to-Line gap: " << sumpro_mingpgap << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "    Minimal current Endpoint gap: " << sumpro_minepgap << Core::IO::endl;

      Core::IO::cout(Core::IO::standard)
          << "    Minimal total Point-to-Point Gap = " << mintotalsimgap_cp_ << Core::IO::endl;
      Core::IO::cout(Core::IO::standard)
          << "    Minimal total Line-to-Line Gap   = " << mintotalsimgap_gp_ << Core::IO::endl;
      Core::IO::cout(Core::IO::standard)
          << "    Minimal total Endpoint Gap       = " << mintotalsimgap_ep_ << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "    Minimal total rel. Gap           = " << mintotalsimrelgap_ << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "    Minimal total unconv. Gap        = " << mintotalsimunconvgap_ << Core::IO::endl;

      Core::IO::cout(Core::IO::debug)
          << "    Maximal current Point-to-Point gap: " << sumpro_maxcpgap << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "    Maximal current Line-to-Line gap: " << sumpro_maxgpgap << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "    Maximal current Endpoint gap: " << sumpro_maxepgap << Core::IO::endl;

      Core::IO::cout(Core::IO::debug)
          << "    Maximal total Point-to-Point Gap = " << maxtotalsimgap_cp_ << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "    Maximal total Line-to-Line Gap   = " << maxtotalsimgap_gp_ << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "    Maximal total Endpoint Gap       = " << maxtotalsimgap_ep_ << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "    Maximal total rel. Gap           = " << maxtotalsimrelgap_ << Core::IO::endl;

      Core::IO::cout(Core::IO::debug)
          << "    global_kappa_max_: " << global_kappa_max_ << Core::IO::endl;
      Core::IO::cout(Core::IO::debug)
          << "    contactevaluationtime_: " << contactevaluationtime_ << Core::IO::endl;
    }

    // end output
    Core::Communication::barrier(get_comm());
    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      Core::IO::cout(Core::IO::standard) << Core::IO::endl;
  }
  return;
}


/*----------------------------------------------------------------------*
 |  Determine number of nodes and nodal DoFs of element      meier 02/14|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::set_element_type_and_distype(Core::Elements::Element* ele1)
{
  const Discret::Elements::Beam3Base* ele = dynamic_cast<const Discret::Elements::Beam3Base*>(ele1);

  numnodes_ = ele->num_centerline_nodes();
  numnodalvalues_ = ele->hermite_centerline_interpolation() ? 2 : 1;

  return;
}

/*----------------------------------------------------------------------*
 | Is element midpoint distance smaller than search radius?  meier 02/14|
 *----------------------------------------------------------------------*/
bool CONTACT::Beam3cmanager::close_midpoint_distance(const Core::Elements::Element* ele1,
    const Core::Elements::Element* ele2,
    std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions, const double sphericalsearchradius)
{
  if (sphericalsearchradius == -1.0) return true;

  Core::LinAlg::Matrix<3, 1> midpos1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> midpos2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> diffvector(Core::LinAlg::Initialization::zero);

  // get midpoint position of element 1
  if (ele1->num_node() == 2)  // 2-noded beam element
  {
    const Core::Nodes::Node* node1ele1 = ele1->nodes()[0];
    const Core::Nodes::Node* node2ele1 = ele1->nodes()[1];

    for (int i = 0; i < 3; ++i)
      midpos1(i) =
          0.5 * ((currentpositions[node1ele1->id()])(i) + (currentpositions[node2ele1->id()])(i));
  }
  else if (ele1->num_node() == 1)  // rigidsphere element
  {
    const Core::Nodes::Node* node1ele1 = ele1->nodes()[0];

    for (int i = 0; i < 3; ++i) midpos1(i) = (currentpositions[node1ele1->id()])(i);
  }

  // get midpoint position of element 2
  if (ele2->num_node() == 2)  // 2-noded beam element
  {
    const Core::Nodes::Node* node1ele2 = ele2->nodes()[0];
    const Core::Nodes::Node* node2ele2 = ele2->nodes()[1];

    for (int i = 0; i < 3; ++i)
      midpos2(i) =
          0.5 * ((currentpositions[node1ele2->id()])(i) + (currentpositions[node2ele2->id()])(i));
  }
  else if (ele2->num_node() == 1)  // rigidsphere element
  {
    const Core::Nodes::Node* node1ele2 = ele2->nodes()[0];

    for (int i = 0; i < 3; ++i) midpos2(i) = (currentpositions[node1ele2->id()])(i);
  }

  // compute distance
  for (int i = 0; i < 3; i++) diffvector(i) = midpos1(i) - midpos2(i);

  if (diffvector.norm2() <= sphericalsearchradius)
    return true;
  else
    return false;
}

/*----------------------------------------------------------------------*
 | read contact force for restart  meier 02/15|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::read_restart(Core::IO::DiscretizationReader& reader)
{
  reader.read_vector(fcold_, "fcold");
  reader.read_vector(dis_old_, "dis_old");
  totpenaltywork_ = reader.read_double("totpenaltywork");

  maxtotalsimgap_ = reader.read_double("maxtotalsimgap");
  maxtotalsimgap_cp_ = reader.read_double("maxtotalsimgap_cp");
  maxtotalsimgap_gp_ = reader.read_double("maxtotalsimgap_gp");
  maxtotalsimgap_ep_ = reader.read_double("maxtotalsimgap_ep");
  maxtotalsimrelgap_ = reader.read_double("maxtotalsimrelgap");
  mintotalsimgap_ = reader.read_double("mintotalsimgap");
  mintotalsimgap_cp_ = reader.read_double("mintotalsimgap_cp");
  mintotalsimgap_gp_ = reader.read_double("mintotalsimgap_gp");
  mintotalsimgap_ep_ = reader.read_double("mintotalsimgap_ep");
  mintotalsimrelgap_ = reader.read_double("mintotalsimrelgap");
  mintotalsimunconvgap_ = reader.read_double("mintotalsimunconvgap");

  outputcounter_ = reader.read_int("outputcounter");

  return;
}

/*----------------------------------------------------------------------*
 | write contact force for restart  meier 02/15|
 *----------------------------------------------------------------------*/
void CONTACT::Beam3cmanager::write_restart(Core::IO::DiscretizationWriter& output)
{
  output.write_vector("fcold", fcold_);
  output.write_vector("dis_old", dis_old_);
  output.write_double("totpenaltywork", totpenaltywork_);

  output.write_double("maxtotalsimgap", maxtotalsimgap_);
  output.write_double("maxtotalsimgap_cp", maxtotalsimgap_cp_);
  output.write_double("maxtotalsimgap_gp", maxtotalsimgap_gp_);
  output.write_double("maxtotalsimgap_ep", maxtotalsimgap_ep_);
  output.write_double("maxtotalsimrelgap", maxtotalsimrelgap_);
  output.write_double("mintotalsimgap", mintotalsimgap_);
  output.write_double("mintotalsimgap_cp", mintotalsimgap_cp_);
  output.write_double("mintotalsimgap_gp", mintotalsimgap_gp_);
  output.write_double("mintotalsimgap_ep", mintotalsimgap_ep_);
  output.write_double("mintotalsimrelgap", mintotalsimrelgap_);
  output.write_double("mintotalsimunconvgap", mintotalsimunconvgap_);

  output.write_int("outputcounter", outputcounter_);  // ToDo needed?

  return;
}

FOUR_C_NAMESPACE_CLOSE
