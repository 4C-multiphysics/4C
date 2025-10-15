// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_periodic.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_levelset_algorithm.hpp"
#include "4C_levelset_intersection_utils.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_lsreinit.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_parameter_list.hpp"

#include <list>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | algebraic reinitialization via solution of equation  rasthofer 09/13 |
 | pde-based reinitialization according to Sussman 1994                 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::reinit_eq()
{
  if (myrank_ == 0)
    std::cout << "\n---------------------------------------  REINITIALIZATION SOLVER  "
                 "----------------------------\n";

  // -----------------------------------------------------------------
  //            prepare time loop for reinitialization
  // -----------------------------------------------------------------
  // set vectors and all further quantities such that the reinitialization
  // can be solved within the existing framework
  prepare_time_loop_reinit();

  // -----------------------------------------------------------------
  //            time loop for reinitialization equation
  // -----------------------------------------------------------------
  time_loop_reinit();

  // -----------------------------------------------------------------
  //            complete reinitialization via equation
  // -----------------------------------------------------------------

  // cleaning of all necessary modifications
  finish_time_loop_reinit();

  return;
}


/*-------------------------------------------------------------------*
 | set element parameters for reinitialization equation   fang 08/15 |
 *-------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::set_reinitialization_element_parameters(
    bool calcinitialtimederivative) const
{
  // create element parameter list
  Teuchos::ParameterList eleparams;

  // reinitialization equation is given in convective form
  eleparams.set<Inpar::ScaTra::ConvForm>("convform", Inpar::ScaTra::convform_convective);

  // no ALE intended
  eleparams.set("isale", false);

  // parameters for stabilization, which are the same as for the level-set equation (if turned on)
  eleparams.sublist("stabilization") = params_->sublist("STABILIZATION");

  // set flag for writing the flux vector fields
  eleparams.set<Inpar::ScaTra::FluxType>("calcflux_domain", calcflux_domain_);

  // set vector containing IDs of scalars for which flux vectors are calculated
  eleparams.set<std::shared_ptr<std::vector<int>>>("writefluxids", writefluxids_);

  // set level-set reinitialization specific parameters
  eleparams.sublist("REINITIALIZATION") = levelsetparams_->sublist("REINITIALIZATION");

  // turn off stabilization and artificial diffusivity when calculating initial time derivative
  if (calcinitialtimederivative)
  {
    eleparams.sublist("REINITIALIZATION")
        .set<Inpar::ScaTra::StabType>(
            "STABTYPEREINIT", Inpar::ScaTra::StabType::stabtype_no_stabilization);
    eleparams.sublist("REINITIALIZATION").set<bool>("ARTDIFFREINIT", false);
  }

  // parameters for finite difference check
  eleparams.set<Inpar::ScaTra::FdCheck>("fdcheck", fdcheck_);
  eleparams.set<double>("fdcheckeps", fdcheckeps_);
  eleparams.set<double>("fdchecktol", fdchecktol_);

  // overwrite some values in general stabilization parameter list by modified values in levelset
  // reinitialization parameter list
  eleparams.sublist("stabilization")
      .set<Inpar::ScaTra::TauType>(
          "DEFINITION_TAU", eleparams.sublist("REINITIALIZATION")
                                .get<Inpar::ScaTra::TauType>("DEFINITION_TAU_REINIT"));
  eleparams.sublist("stabilization")
      .set<Inpar::ScaTra::StabType>("STABTYPE",
          eleparams.sublist("REINITIALIZATION").get<Inpar::ScaTra::StabType>("STABTYPEREINIT"));
  eleparams.sublist("stabilization").set<bool>("SUGRVEL", false);
  eleparams.sublist("stabilization")
      .set<Inpar::ScaTra::AssgdType>(
          "DEFINITION_ASSGD", eleparams.sublist("REINITIALIZATION")
                                  .get<Inpar::ScaTra::AssgdType>("DEFINITION_ARTDIFFREINIT"));

  // set general parameters first
  Discret::Elements::ScaTraEleParameterStd::instance(discret_->name())->set_parameters(eleparams);

  // set additional, problem-dependent parameters
  Discret::Elements::ScaTraEleParameterLsReinit::instance(discret_->name())
      ->set_parameters(eleparams);
}



/*----------------------------------------------------------------------*
 | set time parameters for reinitialization equation    rasthofer 12/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::set_reinitialization_element_time_parameters()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", false);
  // reinitialization equation only implemented incrementally, since it is nonlinear
  eleparams.set<bool>("incremental solver", true);

  eleparams.set<double>("time-step length", dtau_);
  eleparams.set<double>("total time", dtau_ * pseudostep_);
  eleparams.set<double>("time factor", thetareinit_ * dtau_);
  eleparams.set<double>("alpha_F", 1.0);

  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
      ->set_parameters(eleparams);
}


/*----------------------------------------------------------------------*
 | prepare internal time loop for reinitialization equation             |
 |                                                      rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::prepare_time_loop_reinit()
{
  // set switch flag to true to active reinitialization specific parts
  switchreinit_ = true;

  // initial or start phi of reinitialization process
  initialphireinit_->update(1.0, *phinp_, 0.0);
  phin_->update(1.0, *phinp_, 0.0);

  // set internal step counter to zero
  pseudostep_ = 0;

  // set time-integration parameters for reinitialization equation
  set_reinitialization_element_time_parameters();
  // set element parameters for reinitialization equation
  set_reinitialization_element_parameters();

  return;
}


/*----------------------------------------------------------------------*
 | internal time loop for reinitialization equation     rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::time_loop_reinit()
{
  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:  + reinitialization time loop");

  //       e.g., steady state of interface nodal values
  //             integrated gradient norm
  bool converged = false;
  while (pseudostep_ < pseudostepmax_ and not converged)
  {
    // -------------------------------------------------------------------
    //                  prepare time step
    // -------------------------------------------------------------------
    prepare_time_step_reinit();

    // -------------------------------------------------------------------
    //                  solve nonlinear equation
    // -------------------------------------------------------------------
    solve_reinit();

    // -------------------------------------------------------------------
    //                  interface correction
    // -------------------------------------------------------------------
    if (reinitcorrector_) correction_reinit();

    // -------------------------------------------------------------------
    //                        check for convergence
    // -------------------------------------------------------------------
    converged = convergence_check_reinit();

    // -------------------------------------------------------------------
    //                        update solution
    // -------------------------------------------------------------------
    update_reinit();
  }

  return;
}


/*----------------------------------------------------------------------*
 | clean internal time loop for reinitialization equation               |
 |                                                      rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::finish_time_loop_reinit()
{
  // reset quantities that may have been overwritten
  // reset internal step counter
  pseudostep_ = 0;

  // reset time-integration parameters for element evaluation
  set_element_time_parameter();
  // reset general parameters for element evaluation
  set_element_general_parameters();
  set_element_turbulence_parameters();

  return;
}


/*----------------------------------------------------------------------*
 | convergence check for reinitialization equation      rasthofer 03/14 |
 *----------------------------------------------------------------------*/
bool ScaTra::LevelSetAlgorithm::convergence_check_reinit()
{
  bool abortreinitloop = false;

  if (reinit_tol_ > 0.0)
  {
    if (myrank_ == 0)
      std::cout << "## WARNING: convergence criterion for reinitialization equation not yet "
                   "carefully checked"
                << std::endl;

    // stop criterion according to Sussman et al 1994
    //         sum_(nodes A with abs(phi_n) < alpha) abs(phi_A_n+1 -phi_A_n)
    //  err = --------------------------------------------------------------- < dtau*h^2
    //                      sum_(nodes A with abs(phi_n) < alpha) 1

    double local_sum = 0.0;
    int local_num_nodes = 0;

    for (int inode = 0; inode < discret_->num_my_row_nodes(); inode++)
    {
      if (std::abs((*phin_)[inode]) < reinitbandwidth_)
      {
        local_sum += std::abs((*phinp_)[inode] - (*phin_)[inode]);
        local_num_nodes += 1;
      }
    }

    // communicate sums
    double global_sum = 0.0;
    int global_num_nodes = 0;
    global_sum = Core::Communication::sum_all(local_sum, discret_->get_comm());
    global_num_nodes = Core::Communication::sum_all(local_num_nodes, discret_->get_comm());

    // compute current error in band
    const double err = global_sum / ((double)global_num_nodes);

    if (myrank_ == 0)
      std::cout << "Convergence Check reinitialization: Err  " << err << "  Tol  " << reinit_tol_
                << " Number of nodes in band  " << global_num_nodes << std::endl;

    if (err <
        reinit_tol_)  //(dtau_*char_ele_length*char_ele_length) suggested by Sussman et al 1994
      abortreinitloop = true;
  }

  return abortreinitloop;
}


/*----------------------------------------------------------------------*
 | setup the variables to do a new reinitialization time step           |
 |                                                      rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::prepare_time_step_reinit()
{
  // prepare first time step
  if (pseudostep_ == 0)
  {
    // TODO: Restructure enforcement of Dirichlet boundary conditions on phin_
    apply_dirichlet_bc(time_, phin_, nullptr);
    calc_initial_time_derivative();
  }

  // increment time and step
  pseudostep_ += 1;

  // -------------------------------------------------------------------
  // set part of the rhs vector belonging to the old timestep
  // -------------------------------------------------------------------
  set_old_part_of_righthandside();
  set_reinitialization_element_time_parameters();

  // -------------------------------------------------------------------
  // compute node-based velocity field
  // -------------------------------------------------------------------
#ifdef USE_PHIN_FOR_VEL
  if (useprojectedreinitvel_ == Inpar::ScaTra::vel_reinit_node_based) calc_node_based_reinit_vel();
#endif

  return;
}


/*----------------------------------------------------------------------*
 | calculate node-based velocity field via L2-projection                |
 | for reinitialization                                 rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::calc_node_based_reinit_vel()
{
  // loop all space dimensions,
  // since assembler can only deal with one dof per node here
  for (int idim = 0; idim < 3; idim++)
  {
    // define vector for velocity component
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
    std::shared_ptr<Core::LinAlg::Vector<double>> velcomp =
        Core::LinAlg::create_vector(*dofrowmap, true);
    velcomp->put_scalar(0.0);

    if (lsdim_ == Inpar::ScaTra::ls_3D or (lsdim_ == Inpar::ScaTra::ls_2Dx and idim != 0) or
        (lsdim_ == Inpar::ScaTra::ls_2Dy and idim != 1) or
        (lsdim_ == Inpar::ScaTra::ls_2Dz and idim != 2))
    {
      // zero out matrix and rhs entries
      sysmat_->zero();
      residual_->put_scalar(0.0);

      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;

      // parameters for the elements
      // action
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
          "action", ScaTra::Action::calc_node_based_reinit_velocity, eleparams);
      // set current spatial direction
      // we have to loop the dimensions, since we merely have one dof per node here
      eleparams.set<int>("direction", idim);
      // activate reinitialization calculation routines
      eleparams.set<bool>("solve reinit eq", true);

      discret_->clear_state();  // TODO Caution if called from nonlinear_solve
      // set initial phi, i.e., solution of level-set equation
      discret_->set_state("phizero", *initialphireinit_);

      switch (reinitaction_)
      {
        case Inpar::ScaTra::reinitaction_sussman:
        {
          // set phin as phi used for velocity
          // note:read as phinp in sysmat_nodal_vel()
#ifdef USE_PHIN_FOR_VEL
          discret_->set_state("phinp", *phin_);
#else
          discret_->set_state("phinp", phinp_);
#endif
          break;
        }
        case Inpar::ScaTra::reinitaction_ellipticeq:
        {
          discret_->set_state("phinp", *phinp_);
          break;
        }
        default:
        {
          FOUR_C_THROW("Unknown reinitialization method for projection!");
        }
      }
      // call loop over elements
      discret_->evaluate(eleparams, sysmat_, residual_);
      discret_->clear_state();

      // finalize the complete matrix
      sysmat_->complete();

      // solve for velocity component
      Core::LinAlg::SolverParams solver_params;
      solver_params.refactor = true;
      solver_params.reset = true;
      solver_->solve(sysmat_, velcomp, residual_, solver_params);

      system_matrix()->reset();
      // reset the solver as well
      solver_->reset();

      // TODO: add simple ILU/SGS/... for projection case, such that the standard system can be
      // solved by efficient AMG methods
    }

    // loop over all local nodes of scatra discretization
    for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
    {
      // store velocity in reinitialization velocity
      const double val = (*velcomp)[lnodeid];
      (*nb_grad_val_)(idim).replace_local_value(lnodeid, val);
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 | contains call of nonlinear solver for reinitialization equation      |
 |                                                      rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::solve_reinit()
{
  // we simply call the nonlinear_solve (since the reinitialization equation is
  // indeed nonlinear), and all the rest concerning the correct action type and
  // parameters is handled via the switchreinit_-flag in the concrete time-integration
  // schemes for level-set problems
  nonlinear_solve();

  return;
}


/*----------------------------------------------------------------------*
 | correction step according to Sussman & Fatemi 1999   rasthofer 12/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::correction_reinit()
{
  if (myrank_ == 0) std::cout << "\n---------------  Correction projection\n";

  // this correction step should force the interface to stay fixed during reinitialization
  // according to Sussman & Fatemi 1999, it is given as
  //
  //    phinp_final = phinp_reinit + dtau * penaltyparameter * deriv H(phizero) ||nabla phizero||,
  //
  // which is used here in form of a projection

  // zero out matrix and rhs entries !
  sysmat_->zero();
  residual_->put_scalar(0.0);

  // generate a parameterlist for communication and control
  Teuchos::ParameterList eleparams;
  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_and_rhs_lsreinit_correction_step, eleparams);
  eleparams.set<bool>("solve reinit eq", true);

  // set state vectors
  discret_->clear_state();
  discret_->set_state("phizero", *initialphireinit_);
  discret_->set_state("phinp", *phinp_);


  // call loop over elements
  discret_->evaluate(eleparams, sysmat_, residual_);
  discret_->clear_state();

  // residual_->print(std::cout);

  // finalize the complete matrix
  sysmat_->complete();

  // solve for corrected phinp
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(sysmat_, phinp_, residual_, solver_params);

  // phinp_->print(std::cout);

  system_matrix()->reset();
  // reset the solver as well
  solver_->reset();
}


/*----------------------------------------------------------------------*
 | geometric reinitialization via distance to interface rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::reinit_geo(
    const std::map<int, Core::Geo::BoundaryIntCells>& interface)
{
  if (myrank_ == 0)
    std::cout << "---  reinitializing level-set field by computing distance to interface ..."
              << std::flush;

  // set switch flag to true to active reinitialization specific parts
  switchreinit_ = true;

  // map holding pbc nodes (masters or slaves) <pbc node id, distance to flame front>
  std::map<int, double> pbcnodes;

  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // determine the number of nodes per element
  int numnodesperele = 0;
  if (discret_->num_my_row_elements() <= 0)
    FOUR_C_THROW("This discretization does not have any row elements.");
  switch (discret_->l_row_element(0)->shape())
  {
    case Core::FE::CellType::hex8:
      numnodesperele = 8;
      break;
    case Core::FE::CellType::hex20:
      numnodesperele = 20;
      std::cout << "Warning, the fast signed distance reinitialization has not been tested with "
                   "hex20 elements!"
                << std::endl;
      break;
    case Core::FE::CellType::hex27:
      numnodesperele = 27;
      std::cout << "Warning, the fast signed distance reinitialization has not been tested with "
                   "hex27 elements!"
                << std::endl;
      break;
    default:
    {
      FOUR_C_THROW(
          "The fast signed distance reinitialization only supports hex8, hex20 and hex27 "
          "elements.");
      break;
    }
  }

  //========================================================================
  // get the following information about the pbc
  // - planenormaldirection e.g. (1,0,0)
  // - minimum in planenormaldirection
  // - maximum in planenormaldirection
  //========================================================================
  // std::vector<Core::Conditions::Condition*>* surfacepbcs = pbc_->ReturnSurfacePBCs();
  // get periodic surface boundary conditions
  std::vector<const Core::Conditions::Condition*> surfacepbcs;
  discret_->get_condition("SurfacePeriodic", surfacepbcs);
  if (surfacepbcs.empty()) discret_->get_condition("LinePeriodic", surfacepbcs);

  std::vector<int> planenormal;
  std::vector<double> globalmins;
  std::vector<double> globalmaxs;

  for (size_t i = 0; i < surfacepbcs.size(); ++i)
  {
    const auto ismaster = surfacepbcs[i]->parameters().get<std::string>("MASTER_OR_SLAVE");
    if (ismaster == "Master")
    {
      const int masterid = surfacepbcs[i]->parameters().get<int>("ID");
      std::vector<int> nodeids(*(surfacepbcs[i]->get_nodes()));
      for (auto& surfacepbc : surfacepbcs)
      {
        const int slaveid = surfacepbc->parameters().get<int>("ID");
        if (masterid == slaveid)
        {
          const auto isslave = surfacepbc->parameters().get<std::string>("MASTER_OR_SLAVE");
          if (isslave == "Slave")
          {
            const std::vector<int>* slavenodeids = surfacepbc->get_nodes();
            // append slave node Ids to node Ids for the complete condition
            for (int slavenodeid : *slavenodeids) nodeids.push_back(slavenodeid);
          }
        }
      }

      // Get normal direction of pbc plane
      const auto pbcplane = surfacepbcs[i]->parameters().get<std::string>("PLANE");
      if (pbcplane == "yz")
        planenormal.push_back(0);
      else if (pbcplane == "xz")
        planenormal.push_back(1);
      else if (pbcplane == "xy")
        planenormal.push_back(2);
      else
        FOUR_C_THROW("A PBC condition could not provide a plane normal.");

      double min = +10e19;
      double max = -10e19;
      for (int gid : nodeids)
      {
        const int lid = discret_->node_row_map()->lid(gid);
        if (lid < 0) continue;
        const Core::Nodes::Node* lnode = discret_->l_row_node(lid);
        const auto& coord = lnode->x();
        if (coord[planenormal.back()] < min) min = coord[planenormal.back()];
        if (coord[planenormal.back()] > max) max = coord[planenormal.back()];
      }
      globalmins.resize(planenormal.size());
      globalmaxs.resize(planenormal.size());
      (globalmins.back()) = Core::Communication::min_all(min, discret_->get_comm());
      (globalmaxs.back()) = Core::Communication::max_all(max, discret_->get_comm());
    }
  }  // end loop over all surfacepbcs


  //=======================================================================
  // Create a vector of eleGIDs and a vector of those eles' node coords and
  // redundantly store it on each proc
  //=======================================================================
  std::vector<int> allcuteleids;
  std::vector<double> allnodecoords;
  {
    // Here we simply take the eleids from the boundaryIntCells map, which leads to our list of cut
    // elements also there is no distribution necessary, as this map is already stored on every proc
    for (std::map<int, Core::Geo::BoundaryIntCells>::const_iterator elepatches = interface.begin();
        elepatches != interface.end(); ++elepatches)
      allcuteleids.push_back(elepatches->first);

    // our local nodecoords
    std::vector<double> nodecoords(3 * numnodesperele * (allcuteleids.size()), 0.0);
    allnodecoords.resize(nodecoords.size(), 0.0);

    // write the node coordinates of every cut rownode of this proc into nodecoords
    for (size_t ivec = 0; ivec < allcuteleids.size(); ++ivec)
    {
      int elegid = allcuteleids[ivec];
      int elelid = discret_->element_row_map()->lid(elegid);
      if (elelid >= 0)
      {
        const int coordbase = 3 * numnodesperele * ivec;
        const Core::Elements::Element* ele = discret_->l_row_element(elelid);
        const Core::Nodes::Node* const* nodes = ele->nodes();
        for (int inode = 0; inode < ele->num_node(); ++inode)
        {
          const int nodecoordbase = coordbase + 3 * inode;
          nodecoords[nodecoordbase + 0] = nodes[inode]->x()[0];
          nodecoords[nodecoordbase + 1] = nodes[inode]->x()[1];
          nodecoords[nodecoordbase + 2] = nodes[inode]->x()[2];
        }
      }
    }

    allnodecoords = Core::Communication::sum_all(nodecoords, discret_->get_comm());
  }

  //================================================================
  // loop all row nodes on the processor
  // those nodes will receive new phi values
  //================================================================
  for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); ++lnodeid)
  {
    // get the processor local node
    const Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);

    // get the dof associated with this node
    const int dofgid =
        discret_->dof(0, lnode, 0);  // since this is a scalar field the dof is always 0
    int doflid = dofrowmap->lid(dofgid);
    if (doflid < 0)
      FOUR_C_THROW(
          "Proc {}: Cannot find dof gid={} in Core::LinAlg::Vector<double>", myrank_, dofgid);

    // get physical coordinates of this node
    Core::LinAlg::Matrix<3, 1> nodecoord(Core::LinAlg::Initialization::uninitialized);
    nodecoord(0) = lnode->x()[0];
    nodecoord(1) = lnode->x()[1];
    nodecoord(2) = lnode->x()[2];

    //=======================================================================================
    // Build a list< pair< int eleGID, double distance > >
    // the distance is based on the distance between the current node and the closest node of
    // the cut element. This guarantees an estimated distance <= the real distance
    //=======================================================================================
    std::list<std::pair<int, double>> eledistance;

    {
      // loop all cut elements
      for (size_t ieleid = 0; ieleid < allcuteleids.size(); ++ieleid)
      {
        const size_t coordbase = 3 * numnodesperele * ieleid;
        double distance = 1.0e19;

        // loop all cut element's nodes
        for (int inode = 0; inode < numnodesperele; ++inode)
        {
          const int nodecoordbase = coordbase + 3 * inode;
          Core::LinAlg::Matrix<3, 1> delta(Core::LinAlg::Initialization::uninitialized);
          delta(0) = allnodecoords[nodecoordbase + 0];
          delta(1) = allnodecoords[nodecoordbase + 1];
          delta(2) = allnodecoords[nodecoordbase + 2];

          delta.update(1.0, nodecoord, -1.0);

          // take care of PBCs
          for (size_t ipbc = 0; ipbc < planenormal.size(); ++ipbc)
          {
            const double fulllength = (globalmaxs[ipbc] - globalmins[ipbc]);
            if (delta(planenormal[ipbc]) >= fulllength / 2.0)
              delta(planenormal[ipbc]) = fulllength - delta(planenormal[ipbc]);
            else if (delta(planenormal[ipbc]) <= -fulllength / 2.0)
              delta(planenormal[ipbc]) = delta(planenormal[ipbc]) + fulllength;
          }
          const double thisdistance =
              sqrt(delta(0) * delta(0) + delta(1) * delta(1) + delta(2) * delta(2));

          if (thisdistance < distance) distance = thisdistance;
        }

        std::pair<int, double> thispair;
        thispair.first = allcuteleids[ieleid];
        thispair.second = distance;
        eledistance.push_back(thispair);
      }
    }
    if (eledistance.empty()) FOUR_C_THROW("No intersected elements available! G-function correct?");


    //==================================================================
    // sort the the vector in ascending order by the estimated distance
    //==================================================================
    // this is the STL sorting, which is pretty fast
    eledistance.sort(my_compare_pairs);

    //--------------------------------------------------------------------------------
    // if a reinitbandwith is used the nodes not within the band will be set to the
    // estimated distance for all others the actual distance will be determined
    //--------------------------------------------------------------------------------
    if (!reinitband_ or (reinitband_ and fabs(eledistance.front().second) <= reinitbandwidth_))
    {
      //========================================================================
      // + update the eledistance vector with the real distance to the interface
      //   starting with the closest estimated element.
      // + Sort the vector by distance after every iteration.
      // + if the distance of the first element in the vector does not change
      //   any more, we have found the shortest distance
      //========================================================================
      int oldeleid = -1;
      while (oldeleid !=
             eledistance.front()
                 .first)  // this is just a safety check. usually loop should abort earlier.
      {
        oldeleid = eledistance.front().first;

        // the minimal distance, if all element patches and the PBCs are considered
        double pbcmindist = 1.0e19;

        // get patches belonging to first entry
        std::map<int, Core::Geo::BoundaryIntCells>::const_iterator elepatches =
            interface.find(eledistance.front().first);
        if (elepatches == interface.end())
          FOUR_C_THROW("Could not find the boundary integration cells belonging to Element {}.",
              eledistance.front().first);

        // number of flamefront patches for this element
        const std::vector<Core::Geo::BoundaryIntCell> patches = elepatches->second;
        const int numpatch = patches.size();

        //--------------------------------------------------------------------
        // due to the PBCs the node might actually be closer to the
        // interface then would be calculated if one only considered
        // the actual position of the node. In order to find the
        // smallest distance the node is copied along all PBC directions
        //
        //   +------------------+ - - - - - - - - - -+
        //   +             II   +
        //   +   x        I  I  +    y               +
        //   +             II   +
        //   +------------------+ - - - - - - - - - -+
        //         original           copy
        //
        //   x: current node
        //   y: copy of current node
        //   I: interface
        //   +: pbc
        //--------------------------------------------------------------------
        if (planenormal.size() > 3)
          FOUR_C_THROW(
              "Sorry, but currently a maximum of three periodic boundary conditions are supported "
              "by the combustion reinitializer.");

        // since there is no stl pow(INT, INT) function, we calculate it manually
        size_t looplimit = 1;
        for (size_t i = 0; i < planenormal.size(); ++i) looplimit *= 2;

        for (size_t ipbc = 0; ipbc < looplimit; ++ipbc)
        {
          double mindist = 1.0e19;
          Core::LinAlg::Matrix<3, 1> tmpcoord(nodecoord);

          // determine which pbcs have to be applied
          //
          // loopcounter | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
          // ------------+---+---+---+---+---+---+---+---+
          //  first PBC  |     x       x       x       x
          // second PBC  |         x   x           x   x
          //  third PBC  |                 x   x   x   x
          //
          // this is equivalent to the binary representation
          // of the size_t
          if (ipbc & 0x01)
          {
            const double pbclength = globalmaxs[0] - globalmins[0];
            if (nodecoord(0) < globalmins[0] or nodecoord(0) > globalmaxs[0]) continue;
            if (nodecoord(planenormal[0]) > globalmins[0] + pbclength / 2.0)
              tmpcoord(planenormal[0]) = nodecoord(planenormal[0]) - pbclength;
            else
              tmpcoord(planenormal[0]) = nodecoord(planenormal[0]) + pbclength;
          }
          if (ipbc & 0x02)
          {
            const double pbclength = globalmaxs[1] - globalmins[1];
            if (nodecoord(1) < globalmins[1] or nodecoord(1) > globalmaxs[1]) continue;
            if (nodecoord(planenormal[1]) > globalmins[1] + pbclength / 2.0)
              tmpcoord(planenormal[1]) = nodecoord(planenormal[1]) - pbclength;
            else
              tmpcoord(planenormal[1]) = nodecoord(planenormal[1]) + pbclength;
          }
          if (ipbc & 0x04)
          {
            const double pbclength = globalmaxs[2] - globalmins[2];
            if (nodecoord(2) < globalmins[2] or nodecoord(2) > globalmaxs[2]) continue;
            if (nodecoord(planenormal[2]) > globalmins[2] + pbclength / 2.0)
              tmpcoord(planenormal[2]) = nodecoord(planenormal[2]) - pbclength;
            else
              tmpcoord(planenormal[2]) = nodecoord(planenormal[2]) + pbclength;
          }

          //-----------------------------------------
          // loop flame front patches of this element
          //-----------------------------------------
          for (int ipatch = 0; ipatch < numpatch; ++ipatch)
          {
            // get a single patch from group of flamefront patches
            const Core::Geo::BoundaryIntCell patch = patches[ipatch];

            // only triangles and quadrangles are allowed as flame front patches (boundary cells)
            if (!(patch.shape() == Core::FE::CellType::tri3 or
                    patch.shape() == Core::FE::CellType::quad4))
            {
              FOUR_C_THROW("invalid type of boundary integration cell for reinitialization");
            }

            // get coordinates of vertices defining flame front patch
            const Core::LinAlg::SerialDenseMatrix& patchcoord = patch.cell_nodal_pos_xyz();

            // compute normal vector to flame front patch
            Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);
            compute_normal_vector_to_interface(patch, patchcoord, normal);

            //-----------------------------------------
            // find flame front patches facing the node
            //-----------------------------------------
            // boolean indicating if facing patch was found
            bool facenode = false;
            // distance to the facing patch
            double patchdist = 1.0e19;  // default value
            // check if this patch faces the node
            find_facing_patch_proj_cell_space(
                tmpcoord, patch, patchcoord, normal, facenode, patchdist);

            // a facing patch was found
            if (facenode == true)
            {
              // overwrite smallest distance if computed patch distance is smaller
              if (fabs(patchdist) < fabs(mindist))
              {
                // if G-value at the node is negative, the minimal distance has to be negative
                if ((*phinp_)[doflid] < 0.0)
                  mindist = -patchdist;
                else
                  mindist = patchdist;
              }
            }

            //-------------------------------------------------------------
            // compute smallest distance to edges of this flame front patch
            //-------------------------------------------------------------
            // distance to the patch edge
            double edgedist = 1.0e19;
            compute_distance_to_edge(tmpcoord, patch, patchcoord, edgedist);

            if (fabs(edgedist) < fabs(mindist))
            {
              // if G-value at the node is negative, the minimal distance has to be negative
              if ((*phinp_)[doflid] < 0.0)
                mindist = -edgedist;
              else
                mindist = edgedist;
            }

            //----------------------------------------------------------------
            // compute smallest distance to vertices of this flame front patch
            //----------------------------------------------------------------
            // distance to the patch vertex
            double vertexdist = 1.0e19;
            compute_distance_to_patch(tmpcoord, patch, patchcoord, vertexdist);

            if (fabs(vertexdist) < fabs(mindist))
            {
              // if G-value at the node is negative, the minimal distance has to be negative
              if ((*phinp_)[doflid] < 0.0)
                mindist = -vertexdist;
              else
                mindist = vertexdist;
            }
          }  // loop over flamefront patches

          if (fabs(mindist) < fabs(pbcmindist))
          {
            pbcmindist = mindist;
          }
        }  // loop over PBCs

        // store the new distance, which is >= the estimated distance
        eledistance.front().second = pbcmindist;


        //==============================================================
        // sort the the vector in ascending order by the distance
        //==============================================================
        // here we use the fact, that everything is already sorted but the first list item
        std::pair<int, double> tmppair = eledistance.front();
        std::list<std::pair<int, double>>::iterator insertiter = eledistance.begin();
        ++insertiter;

        int loopcount = 0;
        // find the place where the item must be inserted
        // while (fabs(tmppair.second) > fabs(insertiter->second) and insertiter !=
        // eledistance.end())
        while (insertiter != eledistance.end() and fabs(tmppair.second) > fabs(insertiter->second))
        {
          insertiter++;
          loopcount++;
        }

        // this removes the item from the front and inserts it where insertiter points to
        eledistance.splice(insertiter, eledistance, eledistance.begin());

        // if item was inserted at the beginning of the list, it must be the shortest distance
        // possible and we can stop checking the other elements' distances
        if (loopcount == 0) break;

      }  // loop over eledistance
    }
    // if outside the reinit band
    else
    {
      // correct the sign of estimated distance
      if ((*phinp_)[doflid] < 0.0) eledistance.front().second = -eledistance.front().second;
    }

    phinp_->replace_local_value(doflid, eledistance.front().second);
  }

  if (myrank_ == 0) std::cout << " done" << std::endl;

  return;
}


/*--------------------------------------------------------------------- -----------------*
 | find a facing flame front patch by projection of node into boundary cell space        |
 |                                                                           henke 12/09 |
 *----------------------------------------------------------------------  -------------- */
void ScaTra::LevelSetAlgorithm::find_facing_patch_proj_cell_space(
    const Core::LinAlg::Matrix<3, 1>& node, const Core::Geo::BoundaryIntCell& patch,
    const Core::LinAlg::SerialDenseMatrix& patchcoord, const Core::LinAlg::Matrix<3, 1>& normal,
    bool& facenode, double& patchdist)
{
  // indicator
  facenode = false;

  static Core::LinAlg::Matrix<2, 1> eta(Core::LinAlg::Initialization::zero);
  double alpha = 0.0;

  //-------------------------------------------------------
  // perform Newton-Raphson method to project node on patch
  //-------------------------------------------------------
  bool converged = false;
  switch (patch.shape())
  {
    case Core::FE::CellType::tri3:
    {
      converged = project_node_on_patch<Core::FE::CellType::tri3>(
          node, patch, patchcoord, normal, eta, alpha);
      break;
    }
    case Core::FE::CellType::quad4:
    {
      converged = project_node_on_patch<Core::FE::CellType::quad4>(
          node, patch, patchcoord, normal, eta, alpha);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown type of boundary integration cell");
      break;
    }
  }

  // Newton iteration converged
  //  std::cout << "Newton iteration converged in " << iter << " steps!" << std::endl;

  //----------------------------------------------------
  // check if projection lies within boundary cell space
  //----------------------------------------------------
  // remark: - tolerance has to be of same order as the tolerance that coordinates of projected
  // nodes
  //           differ from an exact position on edges of patches (e.g. 1.0E-7 ~ 1.0E-8 -> 1.0E-6)
  //         - if this is not the case, the level set function can become tilted, since valid
  //           patches are ignored
  double TOL = 1e-6;

  switch (patch.shape())
  {
    case Core::FE::CellType::tri3:
    {
      // criteria for tri3 patch
      if ((eta(0) > -TOL) and (eta(0) < 1.0 + TOL) and (eta(1) > -TOL) and (eta(1) < 1.0 + TOL) and
          (1.0 - eta(0) - eta(1) > -TOL) and (1.0 - eta(0) - eta(1) < 1.0 + TOL) and converged)
      {
        facenode = true;
        patchdist = fabs(alpha);
        //      std::cout << "facing patch found (tri3 patch)! coordinates eta(0): " << eta(0) << "
        //      eta(1) " << eta(1) << std::endl;
      }
      break;
    }
    case Core::FE::CellType::quad4:
    {
      // criteria for quad4 patch
      if ((eta(0) > -1.0 - TOL) and (eta(0) < 1.0 + TOL) and (eta(1) > -1.0 - TOL) and
          (eta(1) < 1.0 + TOL) and converged)
      {
        facenode = true;
        patchdist = fabs(alpha);
        //      std::cout << "facing patch found (quad4 patch)!" << std::endl;
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown type of boundary integration cell");
      break;
    }
  }
  //  if (!converged)
  //  {
  //    std::cout << "node x component " << node(0,0) << std::endl;
  //    std::cout << "node y component " << node(1,0) << std::endl;
  //    std::cout << "node z component " << node(2,0) << std::endl;
  //    std::cout << "eta1 " << eta(0) << std::endl;
  //    std::cout << "eta2 " << eta(1) << std::endl;
  //    std::cout << "alpha " << alpha << std::endl;
  //    std::cout << "patch vertices x component " << patchcoord(0,0) << " " << patchcoord(0,1) << "
  //    " << patchcoord(0,2) << std::endl; std::cout << "patch vertices y component " <<
  //    patchcoord(1,0) << " " << patchcoord(1,1) << " " << patchcoord(1,2) << std::endl; std::cout
  //    << "patch vertices z component " << patchcoord(2,0) << " " << patchcoord(2,1) << " " <<
  //    patchcoord(2,2) << std::endl;
  //  }

  return;
}


/*---------------------------------------------------------------------------------------*
 | compute distance to edge of patch                                         henke 08/09 |
 *-------------------------------------------------------------------------------------- */
void ScaTra::LevelSetAlgorithm::compute_distance_to_edge(const Core::LinAlg::Matrix<3, 1>& node,
    const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
    double& edgedist)
{
  // set temporary edgedist to large value
  double edgedisttmp = edgedist;

  // number of vertices of flame front patch (3 for tri3; 4 for quad4)
  const size_t numvertices = patchcoord.numCols();

  // current vertex of the patch (first vertex)
  static Core::LinAlg::Matrix<3, 1> vertex1(Core::LinAlg::Initialization::zero);
  // current next vertex of the patch (second vertex)
  static Core::LinAlg::Matrix<3, 1> vertex2(Core::LinAlg::Initialization::zero);
  // distance vector from first vertex to node
  static Core::LinAlg::Matrix<3, 1> vertex1tonode(Core::LinAlg::Initialization::zero);
  // distance vector from first vertex to second vertex
  static Core::LinAlg::Matrix<3, 1> vertex1tovertex2(Core::LinAlg::Initialization::zero);

  // compute distance to all vertices of patch
  for (size_t ivert = 0; ivert < numvertices; ++ivert)
  {
    // vertex1 of flame front patch
    vertex1(0) = patchcoord(0, ivert);
    vertex1(1) = patchcoord(1, ivert);
    vertex1(2) = patchcoord(2, ivert);

    if (ivert < (numvertices - 1))
    {
      vertex2(0) = patchcoord(0, ivert + 1);
      vertex2(1) = patchcoord(1, ivert + 1);
      vertex2(2) = patchcoord(2, ivert + 1);
    }
    else if (ivert == (numvertices - 1))
    {
      vertex2(0) = patchcoord(0, 0);
      vertex2(1) = patchcoord(1, 0);
      vertex2(2) = patchcoord(2, 0);
    }

    // compute distance vector from node to current first
    vertex1tonode.update(1.0, node, -1.0, vertex1);
    // compute distance vector from current second first vertex to current first vertex (edge)
    vertex1tovertex2.update(1.0, vertex2, -1.0, vertex1);
    double normvertex1tovertex2 = vertex1tovertex2.norm2();
    // normalize vector
    vertex1tovertex2.scale(1.0 / normvertex1tovertex2);

    // scalar product of vertex1tonode and the normed vertex1tovertex2
    double lotfusspointdist = vertex1tovertex2.dot(vertex1tonode);

    if ((lotfusspointdist >= 0.0) and
        (lotfusspointdist <= normvertex1tovertex2))  // lotfusspoint on edge
    {
      Core::LinAlg::Matrix<3, 1> lotfusspoint(Core::LinAlg::Initialization::zero);
      lotfusspoint.update(1.0, vertex1, lotfusspointdist, vertex1tovertex2);
      Core::LinAlg::Matrix<3, 1> nodetolotfusspoint(Core::LinAlg::Initialization::zero);
      nodetolotfusspoint.update(1.0, lotfusspoint, -1.0, node);

      // determine length of vector from node to lot fuss point
      edgedisttmp = nodetolotfusspoint.norm2();
      if (edgedisttmp < edgedist) edgedist = edgedisttmp;
    }
  }

  return;
}


/*---------------------------------------------- ----------------------------------------*
 | compute distance to vertex of patch                                       henke 08/09 |
 *-------------------------------------------------------------------------------------- */
void ScaTra::LevelSetAlgorithm::compute_distance_to_patch(const Core::LinAlg::Matrix<3, 1>& node,
    const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
    double& vertexdist)
{
  // set temporary vertexdist to large value
  double vertexdisttmp = vertexdist;

  // number of vertices of flame front patch (3 for tri3; 4 for quad4)
  const size_t numvertices = patchcoord.numCols();

  // current vertex of the patch
  static Core::LinAlg::Matrix<3, 1> vertex(Core::LinAlg::Initialization::zero);
  // distance vector from patch to node
  static Core::LinAlg::Matrix<3, 1> dist(Core::LinAlg::Initialization::zero);

  // compute distance to all vertices of patch
  for (size_t ivert = 0; ivert < numvertices; ++ivert)
  {
    // vertex of flame front patch
    vertex(0) = patchcoord(0, ivert);
    vertex(1) = patchcoord(1, ivert);
    vertex(2) = patchcoord(2, ivert);

    // compute distance vector from flame front to node
    dist.update(1.0, node, -1.0, vertex);

    // compute L2-norm of distance vector
    vertexdisttmp = dist.norm2();
    if (vertexdisttmp < vertexdist) vertexdist = vertexdisttmp;
  }

  return;
}


/*-------------------------------------------------------------------------------------*
 | compute normal vector to interface patch                                henke 08/09 |
 *------------------------------------------------- ---------------------------------- */
void ScaTra::LevelSetAlgorithm::compute_normal_vector_to_interface(
    const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
    Core::LinAlg::Matrix<3, 1>& normal)
{
  // first point of flame front patch
  Core::LinAlg::Matrix<3, 1> point1;
  point1(0) = patchcoord(0, 0);
  point1(1) = patchcoord(1, 0);
  point1(2) = patchcoord(2, 0);

  // second point of flame front patch
  Core::LinAlg::Matrix<3, 1> point2;
  point2(0) = patchcoord(0, 1);
  point2(1) = patchcoord(1, 1);
  point2(2) = patchcoord(2, 1);

  // first edge of flame front patch
  Core::LinAlg::Matrix<3, 1> edge1;
  edge1.update(1.0, point2, -1.0, point1);

  // third point of flame front patch
  point2(0) = patchcoord(0, 2);
  point2(1) = patchcoord(1, 2);
  point2(2) = patchcoord(2, 2);

  // second edge of flame front patch (if patch is triangle; if not: edge 2 is secant of polygon)
  Core::LinAlg::Matrix<3, 1> edge2;
  edge2.update(1.0, point2, -1.0, point1);

  // compute normal vector of patch (cross product: edge1 x edge2)
  // remark: normal vector points into unburnt domain (G<0)
  normal(0) = (edge1(1) * edge2(2) - edge1(2) * edge2(1));
  normal(1) = (edge1(2) * edge2(0) - edge1(0) * edge2(2));
  normal(2) = (edge1(0) * edge2(1) - edge1(1) * edge2(0));

  //  MPI_Comm comm = scatra_.discretization()->Comm();
  //  std::cout << "proc " << Core::Communication::my_mpi_rank(comm) << " normal " <<  normal <<
  //  std::endl; std::cout << "proc " << Core::Communication::my_mpi_rank(comm) << " patch " <<
  //  patchcoord << std::endl;

  // compute unit (normed) normal vector
  double norm = sqrt(normal(0) * normal(0) + normal(1) * normal(1) + normal(2) * normal(2));
  if (norm == 0.0) FOUR_C_THROW("norm of normal vector is zero!");
  normal.scale(1.0 / norm);

  return;
}


/*-------------------------------------------------------------------------------------*
 | project node into the boundary cell space                               henke 08/09 |
 *------------------------------------------------- ---------------------------------- */
template <Core::FE::CellType distype>
bool ScaTra::LevelSetAlgorithm::project_node_on_patch(const Core::LinAlg::Matrix<3, 1>& node,
    const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
    const Core::LinAlg::Matrix<3, 1>& normal, Core::LinAlg::Matrix<2, 1>& eta, double& alpha)
{
  // indicator for convergence of Newton-Raphson scheme
  bool converged = false;
  // number space dimensions for 3d combustion problems
  const size_t nsd = 3;
  // here, a triangular boundary integration cell is assumed (numvertices = 3)
  const size_t numvertices = Core::FE::num_nodes(distype);

  // get coordinates of vertices of flame front patch
  // remark: here we only get a view (bool true) on the SerialDenseMatrix returned by
  // CellNodalPosXYZ()
  Core::LinAlg::Matrix<nsd, numvertices> patchcoordfix(patchcoord.values(), true);

  static Core::LinAlg::Matrix<numvertices, 1> funct(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<2, numvertices> deriv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<nsd, 1> projX(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<nsd, 2> gradprojX(Core::LinAlg::Initialization::zero);

  //----------------------------------
  // start values for iterative scheme
  //----------------------------------
  // start position (barycenter of triangular boundary cell)
  eta(0) = 1.0 / 3.0;
  eta(1) = 1.0 / 3.0;
  // auxiliary variable
  // remark: third unknown to close system of equations; arbitrary value
  alpha = 0.0;

  // function F (system of equations)
  static Core::LinAlg::Matrix<nsd, 1> f(Core::LinAlg::Initialization::zero);
  // gradient of function F (dF/deta(0), dF/deta(1), dF/dalpha)
  static Core::LinAlg::Matrix<nsd, nsd> gradf(Core::LinAlg::Initialization::zero);
  // increment in Newton iteration (unknown to be solved for)
  static Core::LinAlg::Matrix<nsd, 1> incr(Core::LinAlg::Initialization::zero);

  // maximum number Newton iterations
  size_t maxiter = 3;
  // convergence tolerance
  double conv = 0.0;

  //------------------------------------------------------
  // Newton-Raphson loop for non-linear projection problem
  //------------------------------------------------------
  for (size_t iter = 0; iter < maxiter; ++iter)
  {
    // evaluate shape functions in boundary cell space at current position \eta_1,\eta_2 on the
    // patch
    funct.clear();
    Core::FE::shape_function_2d(funct, eta(0), eta(1), patch.shape());
    // evaluate derivatives of shape functions in boundary cell space at current position
    // \eta_1,\eta_2 on the patch
    deriv.clear();
    Core::FE::shape_function_2d_deriv1(deriv, eta(0), eta(1), patch.shape());

    // evaluate projection X of node P at current position \eta_1,\eta_2 on the patch
    // projX(i,j) = patchcoord(i,k)*funct(k,1)
    projX.clear();
    projX.multiply_nn(patchcoordfix, funct);

    // evaluate gradient of projection X of node P at current position \eta_1,\eta_2 on the patch
    // gradprojX(i,j) = patchcoord(i,k)*deriv(j,k)
    gradprojX.clear();
    gradprojX.multiply_nt(patchcoordfix, deriv);

    //---------------------------------------------------
    // build system of equations F and its gradient gradF
    //---------------------------------------------------
    // TODO documentation missing
    f.clear();
    gradf.clear();
    incr.clear();
    for (size_t icoord = 0; icoord < nsd; ++icoord)
    {
      // evaluate function f
      f(icoord) = projX(icoord) + alpha * normal(icoord) - node(icoord);
      // evaluate gradient of function at current position on patch in boundary cell space
      gradf(icoord, 0) = gradprojX(icoord, 0);
      gradf(icoord, 1) = gradprojX(icoord, 1);
      gradf(icoord, 2) = normal(icoord);
    }

    // check convergence
    conv = sqrt(f(0) * f(0) + f(1) * f(1) + f(2) * f(2));
    // std::cout << "iteration " << iter << ": -> |f|=" << conv << std::endl;
    if (conv <= 1.0E-12) break;

    //----------------------------------------------------
    // solve linear system of equations: gradF * incr = -F
    //----------------------------------------------------
    // F = F*-1.0
    f.scale(-1.0);
    // solve A.X=B
    Core::LinAlg::FixedSizeSerialDenseSolver<nsd, nsd, 1> solver;
    solver.set_matrix(gradf);                // set A=gradF
    solver.set_vectors(incr, f);             // set X=incr, B=F
    solver.factor_with_equilibration(true);  // "some easy type of preconditioning" (Michael)
    int err2 = solver.factor();              // ?
    int err = solver.solve();                // incr = gradF^-1.F
    if ((err != 0) || (err2 != 0))
      FOUR_C_THROW("solving linear system in Newton-Raphson method for projection failed");

    // update eta and alpha
    eta(0) += incr(0);
    eta(1) += incr(1);
    alpha += incr(2);
    // std::cout << "solution vector: component 1: " << eta(0) << " component 2: " << eta(1) << "
    // alpha: " << alpha << std::endl;
  }
  // change sign to preserve sign of G-function
  alpha = -alpha;

  // Newton iteration unconverged
  if (conv > 1.0E-12)
  {
    alpha = 7777.7;
    //        std::cout << "projection did not converge" << std::endl;
    // FOUR_C_THROW("projection did not converge!");
  }
  else
  {
    converged = true;
    // std::cout << "convergence criterion " << conv << std::endl;
    // std::cout << "solution vector: component 1: " << eta(0) << " component 2: " << eta(1) << "
    // alpha: " << alpha << std::endl;
  }

  return converged;
}


/*------------------------------------------------------------------------------------------------*
 | correct the volume of the minus domain after reinitialization                  rasthofer 07/11 |
 |                                                                                    DA wichmann |
 | Idea: shift level-set so that volume is conserved                                              |
 *------------------------------------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::correct_volume()
{
  double volminus = 0.0;
  double volplus = 0.0;
  double surface = 0.0;
  std::map<int, Core::Geo::BoundaryIntCells> interface;
  interface.clear();
  // reconstruct interface and calculate volumes, etc ...
  ScaTra::LevelSet::Intersection intersect;
  intersect.capture_zero_level_set(*phinp_, *discret_, volminus, volplus, surface, interface);

  const double voldelta = initvolminus_ - volminus;
  if (myrank_ == 0)
    Core::IO::cout << "Correcting volume of minus(-) domain by " << voldelta << " ... "
                   << Core::IO::endl;

  // This is a guess on how thick a layer needs to be added to the surface of the minus domain.
  // Due to $ \grad \phi \approx 1 $ this also happens to be the value that needs to be subtracted
  // of all phis. To make sure that \grad \phi really is close to 1 this function should only be
  // called after a reinitialization.
  const double thickness = -voldelta / surface;

  Core::LinAlg::Vector<double> one(phin_->get_map());
  one.put_scalar(1.0);

  // update phi
  phinp_->update(thickness, one, 1.0);

  if (myrank_ == 0) Core::IO::cout << "done" << Core::IO::endl;

  return;
}

/*----------------------------------------------------------------------*
 | elliptic reinitialization                            rasthofer 09/14 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::reinit_elliptic(
    std::map<int, Core::Geo::BoundaryIntCells>& interface)
{
  // store interface
  interface_eleq_ = std::make_shared<std::map<int, Core::Geo::BoundaryIntCells>>(interface);

  // call the executing method
  reinitialize_with_elliptic_equation();
}

/*----------------------------------------------------------------------*
 | elliptic reinitialization                            rasthofer 09/14 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetAlgorithm::reinitialize_with_elliptic_equation()
{
  //-------------------------------------------------
  // preparations
  //-------------------------------------------------

  // set switch flag to true to activate reinitialization specific parts
  switchreinit_ = true;

  // set element parameters for reinitialization equation
  set_reinitialization_element_parameters();

  // vector for initial phi (solution of level-set equation) of reinitialization process
  // this vector is only initialized: currently function calc_node_based_reinit_vel() is also
  // used to compute nodal level-set gradients, and this function expects that initialphireinit_ has
  // been set although it is not used for the present purposes
  initialphireinit_ = Core::LinAlg::create_vector(*(discret_->dof_row_map()), true);

  //-------------------------------------------------
  // solve
  //-------------------------------------------------

  // we simply call the linear_solve (since the elliptic reinitialization equation is
  // indeed nonlinear), and all the rest concerning the correct action type and
  // parameters is handled via the switchreinit_-flag in the concrete time-integration
  // schemes for level-set problems

  // some preparations
  Core::LinAlg::Vector<double> phinmloc(*phinp_);
  Core::LinAlg::Vector<double> inc(*(discret_->dof_row_map()), true);
  int step = 0;
  bool not_conv = true;

  while (not_conv)
  {
    step += 1;

    //-----------------------------
    // compute node-based gradient
    //-----------------------------
    if (projection_) calc_node_based_reinit_vel();

    //-----------------------------
    // setup and solve system
    //-----------------------------
    // caution: we can only use linear_solve together with linear-full strategy here
    linear_solve();

    //-----------------------------
    // check convergence
    //-----------------------------
    inc.update(1.0, *phinp_, -1.0, phinmloc, 0.0);
    double norm = 0.0;
    inc.norm_2(&norm);

    if (myrank_ == 0)
      std::cout << "STEP:  " << step << "/" << pseudostepmax_
                << "  -- inc norm L2:  " << std::setprecision(3) << std::scientific << norm
                << std::endl;

    if (reinit_tol_ > 0.0)
    {
      if (norm < reinit_tol_ or step >= pseudostepmax_) not_conv = false;
    }
    else
    {
      if (step >= pseudostepmax_) not_conv = false;
    }

    phinmloc.update(1.0, *phinp_, 0.0);
  }

  //-------------------------------------------------
  // finish
  //-------------------------------------------------

  // reset time-integration parameters for element evaluation
  // set_element_time_parameter(); -> have not been modified
  // reset general parameters for element evaluation
  set_element_general_parameters();
  set_element_turbulence_parameters();

  // clear variables
  interface_eleq_ = nullptr;
  initialphireinit_ = nullptr;
  if (projection_ == true) nb_grad_val_->put_scalar(0.0);

  return;
}

FOUR_C_NAMESPACE_CLOSE
