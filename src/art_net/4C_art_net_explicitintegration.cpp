// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_explicitintegration.hpp"

#include "4C_art_net_art_junction.hpp"
#include "4C_art_net_artery_ele_action.hpp"
#include "4C_art_net_artery_resulttest.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <stdio.h>

FOUR_C_NAMESPACE_OPEN



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 01/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//

Arteries::ArtNetExplicitTimeInt::ArtNetExplicitTimeInt(
    std::shared_ptr<Core::FE::Discretization> actdis, const int linsolvernumber,
    const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& artparams,
    Core::IO::DiscretizationWriter& output)
    : TimInt(actdis, linsolvernumber, probparams, artparams, output)
{
  const int restart_step = Global::Problem::instance()->restart();
  if (restart_step > 0)
  {
    Core::IO::DiscretizationReader reader(
        discret_, Global::Problem::instance()->input_control_file(), restart_step);

    time_ = reader.read_double("time");
  }

  visualization_writer_ = std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(
      actdis, Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), time_));
}  // ArtNetExplicitTimeInt::ArtNetExplicitTimeInt



/*----------------------------------------------------------------------*
 | Initialize the time integration.                                     |
 |                                                          ismail 12/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetExplicitTimeInt::init(const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& arteryparams, const std::string& scatra_disname)
{
  // time measurement: initialization
  if (!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR(" + initialization");
  }

  // call base class
  TimInt::init(globaltimeparams, arteryparams, scatra_disname);

  // ensure that degrees of freedom in the discretization have been set
  if (!discret_->filled() || !discret_->have_dofs()) discret_->fill_complete();

  // -------------------------------------------------------------------
  // Force the reducesd 1d arterial network discretization to run on
  // and only one cpu
  // -------------------------------------------------------------------
  // reduce the node row map into processor 0
  const Core::LinAlg::Map noderowmap_1_proc =
      *Core::LinAlg::allreduce_e_map(*discret_->node_row_map(), 0);
  // update the discetization by redistributing the new row map
  discret_->redistribute(noderowmap_1_proc, noderowmap_1_proc);

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  //  const Core::LinAlg::Map* dofcolmap  = discret_->DofColMap();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global node numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* noderowmap = discret_->node_row_map();


  // -------------------------------------------------------------------
  // get a vector layout from the discretization for a vector which only
  // contains the volumetric flow rate dofs and for one vector which only
  // contains cross-sectional area degrees of freedom.
  // -------------------------------------------------------------------


  // This is a first estimate for the number of non zeros in a row of
  // the matrix. Each node has 3 adjacent nodes (including itself), each
  // with 2 dofs. (3*2=6)
  // We do not need the exact number here, just for performance reasons
  // a 'good' estimate

  // initialize standard (stabilized) system matrix
  sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 6, false, true);

  // Vectors passed to the element
  // -----------------------------
  // Volumetric flow rate at time n+1, n and n-1
  qanp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  qan_ = Core::LinAlg::create_vector(*dofrowmap, true);
  qanm_ = Core::LinAlg::create_vector(*dofrowmap, true);

  qan_3D_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // Vectors associated to boundary conditions
  // -----------------------------------------
  Wfo_ = Core::LinAlg::create_vector(*noderowmap, true);
  Wbo_ = Core::LinAlg::create_vector(*noderowmap, true);
  Wfnp_ = Core::LinAlg::create_vector(*noderowmap, true);
  Wfn_ = Core::LinAlg::create_vector(*noderowmap, true);
  Wfnm_ = Core::LinAlg::create_vector(*noderowmap, true);
  Wbnp_ = Core::LinAlg::create_vector(*noderowmap, true);
  Wbn_ = Core::LinAlg::create_vector(*noderowmap, true);
  Wbnm_ = Core::LinAlg::create_vector(*noderowmap, true);

  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  // This part might be optimized later
  bcval_ = Core::LinAlg::create_vector(*dofrowmap, true);
  dbctog_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // Vectors used for postporcesing visualization
  // --------------------------------------------
  qn_ = Core::LinAlg::create_vector(*noderowmap, true);
  on_ = Core::LinAlg::create_vector(*noderowmap, true);
  an_ = Core::LinAlg::create_vector(*noderowmap, true);
  nodeIds_ = Core::LinAlg::create_vector(*noderowmap, true);

  // right hand side vector and right hand side corrector
  rhs_ = Core::LinAlg::create_vector(*dofrowmap, true);
  // create the junction boundary conditions
  Teuchos::ParameterList junparams;

  junc_nodal_vals_ =
      std::make_shared<std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>();

  junparams.set<
      std::shared_ptr<std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>>(
      "Junctions Parameters", junc_nodal_vals_);

  artjun_ = std::make_shared<Utils::ArtJunctionWrapper>(discret_, output_, junparams, dta_);

  // ---------------------------------------------------------------------------------------
  // Initialize all the arteries' cross-sectional areas to the initial crossectional area Ao
  // and the volumetric flow rate to 0
  // ---------------------------------------------------------------------------------------
  Teuchos::ParameterList eleparams;
  discret_->clear_state();
  discret_->set_state("qanp", *qanp_);

  // loop all elements on this proc (including ghosted ones)

  //  for (int nele=0;nele<discret_->NumMyColElements();++nele)
  {
    // get the element
    //    Core::Elements::Element* ele = discret_->lColElement(nele);

    // get element location vector, dirichlet flags and ownerships
    //    std::vector<int> lm;
    //    std::vector<int> lmstride;
    //    std::vector<int> lmowner;
    //        std::shared_ptr<std::vector<int> > lmowner = Teuchos::rcp(new std::vector<int>);
    //    ele->LocationVector(*discret_,lm,*lmowner,lmstride);

    // loop all nodes of this element, add values to the global vectors
    eleparams.set("qa0", qanp_);
    eleparams.set("wfo", Wfo_);
    eleparams.set("wbo", Wbo_);
    Wfn_->update(1.0, *Wfo_, 0.0);
    Wbn_->update(1.0, *Wbo_, 0.0);
    // eleparams.set("lmowner",lmowner);
    eleparams.set<Arteries::Action>("action", Arteries::get_initial_artery_state);
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
  // Fill the NodeId vector
  for (int nele = 0; nele < discret_->num_my_col_elements(); ++nele)
  {
    // get the element
    Core::Elements::Element* ele = discret_->l_col_element(nele);

    // get element location vector, dirichlet flags and ownerships
    std::vector<int> lm;
    std::vector<int> lmstride;
    // vector<int> lmowner;
    std::vector<int> lmowner;
    ele->location_vector(*discret_, lm, lmowner, lmstride);

    // loop all nodes of this element, add values to the global vectors

    if (myrank_ == (lmowner)[0])
    {
      int gid = lm[0];
      double val = gid;
      nodeIds_->replace_global_values(1, &val, &gid);
    }
    if (myrank_ == (lmowner)[1])
    {
      int gid = lm[1];
      double val = gid;
      nodeIds_->replace_global_values(1, &val, &gid);
    }
  }

  // -----------------------------------------------------------------------
  // initialize all scatra related stuff
  // -----------------------------------------------------------------------

  if (solvescatra_)
  {
    // initialize scatra system matrix
    scatra_sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 6, false, true);
    // right hand side vector and right hand side corrector
    scatra_rhs_ = Core::LinAlg::create_vector(*dofrowmap, true);

    // Scalar transport vector of O2 and CO2
    export_scatra_ = Core::LinAlg::create_vector(*noderowmap, true);
    scatraO2nm_ = Core::LinAlg::create_vector(*dofrowmap, true);
    scatraO2n_ = Core::LinAlg::create_vector(*dofrowmap, true);
    scatraO2np_ = Core::LinAlg::create_vector(*dofrowmap, true);
    scatraO2wfn_ = Core::LinAlg::create_vector(*noderowmap, true);
    scatraO2wfnp_ = Core::LinAlg::create_vector(*noderowmap, true);
    scatraO2wbn_ = Core::LinAlg::create_vector(*noderowmap, true);
    scatraO2wbnp_ = Core::LinAlg::create_vector(*noderowmap, true);

    scatraCO2n_ = Core::LinAlg::create_vector(*dofrowmap, true);
    scatraCO2np_ = Core::LinAlg::create_vector(*dofrowmap, true);
    scatraCO2wfn_ = Core::LinAlg::create_vector(*noderowmap, true);
    scatraCO2wfnp_ = Core::LinAlg::create_vector(*noderowmap, true);
    scatraCO2wbn_ = Core::LinAlg::create_vector(*noderowmap, true);
    scatraCO2wbnp_ = Core::LinAlg::create_vector(*noderowmap, true);

    // a vector of zeros to be used to enforce zero dirichlet boundary conditions
    // This part might be optimized later
    scatra_bcval_ = Core::LinAlg::create_vector(*dofrowmap, true);
    scatra_dbctog_ = Core::LinAlg::create_vector(*dofrowmap, true);
  }

}  // ArtNetExplicitTimeInt::Init

/*----------------------------------------------------------------------*
 | the solver for artery                                   ismail 06/09 |
 *----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::solve(
    std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  // time measurement: Artery
  if (!coupledTo3D_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("   + solving artery");
  }

  // -------------------------------------------------------------------
  // call elements to calculate system matrix
  // -------------------------------------------------------------------

  // get cpu time
  //  const double tcpuele = Teuchos::Time::wallTime();

  {
    // time measurement: element
    if (!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + element calls");
    }

    // set both system matrix and rhs vector to zero
    sysmat_->zero();
    rhs_->put_scalar(0.0);


    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<Arteries::Action>("action", Arteries::calc_sys_matrix_rhs);
    eleparams.set("time step size", dta_);

    // other parameters that might be needed by the elements
    eleparams.set("total time", time_);

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("qanp", *qanp_);


    // call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
    discret_->clear_state();

    // finalize the complete matrix
    sysmat_->complete();
  }
  // end time measurement for element

  // -------------------------------------------------------------------
  // call elements to calculate the Riemann problem
  // -------------------------------------------------------------------
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<Arteries::Action>("action", Arteries::solve_riemann_problem);

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("qanp", *qanp_);

    eleparams.set("time step size", dta_);
    eleparams.set("Wfnp", Wfnp_);
    eleparams.set("Wbnp", Wbnp_);

    eleparams.set("total time", time_);
    eleparams.set<
        std::shared_ptr<std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>>(
        "Junctions Parameters", junc_nodal_vals_);

    // call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
  }

  // Solve the boundary conditions
  bcval_->put_scalar(0.0);
  dbctog_->put_scalar(0.0);
  // Solve terminal BCs

  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<Arteries::Action>("action", Arteries::set_term_bc);

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("qanp", *qanp_);

    eleparams.set("time step size", dta_);
    eleparams.set("total time", time_);
    eleparams.set("bcval", bcval_);
    eleparams.set("dbctog", dbctog_);
    eleparams.set("Wfnp", Wfnp_);
    eleparams.set("Wbnp", Wbnp_);
    eleparams.set<
        std::shared_ptr<std::map<const int, std::shared_ptr<Arteries::Utils::JunctionNodeParams>>>>(
        "Junctions Parameters", junc_nodal_vals_);

    // Add the parameters to solve terminal BCs coupled to 3D fluid boundary
    eleparams.set("coupling with 3D fluid params", CouplingTo3DParams);

    // solve junction boundary conditions
    artjun_->solve(eleparams);

    // call standard loop over all elements
    discret_->evaluate(eleparams, sysmat_, rhs_);
  }


  // -------------------------------------------------------------------
  // Apply the BCs to the system matrix and rhs
  // -------------------------------------------------------------------
  {
    // time measurement: application of dbc
    if (!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + apply DBC");
    }
    Core::LinAlg::apply_dirichlet_to_system(*sysmat_, *qanp_, *rhs_, *bcval_, *dbctog_);
  }

  //-------solve for total new velocities and pressures
  // get cpu time
  const double tcpusolve = Teuchos::Time::wallTime();
  {
    // time measurement: solver
    if (!coupledTo3D_)
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + solver calls");
    }

    // call solver
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve(sysmat_, qanp_, rhs_, solver_params);
  }
  // end time measurement for solver
  dtsolve_ = Teuchos::Time::wallTime() - tcpusolve;

  if (myrank_ == 0) std::cout << "te=" << dtele_ << ", ts=" << dtsolve_ << "\n\n";

  // Update Wf and Wb
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<Arteries::Action>("action", Arteries::evaluate_wf_wb);

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("qanp", *qanp_);

    eleparams.set("time step size", dta_);
    eleparams.set("total time", time_);
    eleparams.set("Wfnp", Wfnp_);
    eleparams.set("Wbnp", Wbnp_);

    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
}  // ArtNetExplicitTimeInt:Solve


void Arteries::ArtNetExplicitTimeInt::solve_scatra()
{
  {
    scatraO2np_->put_scalar(0.0);
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<Arteries::Action>("action", Arteries::evaluate_scatra_analytically);

    // set vector values needed by elements
    discret_->clear_state();

    eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfn", Wfn_);
    eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbn", Wbn_);
    eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wfo", Wfo_);
    eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("Wbo", Wbo_);
    eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("scatran", scatraO2n_);
    eleparams.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("scatranp", scatraO2np_);

    eleparams.set("time step size", dta_);

    // call standard loop over all elements
    discret_->evaluate(eleparams, scatra_sysmat_, scatra_rhs_);
  }
  {
    scatra_bcval_->put_scalar(0.0);
    scatra_dbctog_->put_scalar(0.0);
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<Arteries::Action>("action", Arteries::set_scatra_term_bc);

    // set vector values needed by elements
    discret_->clear_state();
    discret_->set_state("qanp", *qanp_);

    eleparams.set("time step size", dta_);
    eleparams.set("time", time_);
    eleparams.set("bcval", scatra_bcval_);
    eleparams.set("dbctog", scatra_dbctog_);

    // call standard loop over all elements
    discret_->evaluate(eleparams, scatra_sysmat_, scatra_rhs_);
  }
  scatraO2np_->update(1.0, *scatra_bcval_, 1.0);
}

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                                      |
 |  qnm_   =  qn_                                                       |
 |  arean_ = areap_                                                     |
 |                                                                      |
 |                                                          ismail 06/09|
 *----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::time_update()
{
  // Volumetric Flow rate/Cross-sectional area of this step become most recent
  qanm_->update(1.0, *qan_, 0.0);
  qan_->update(1.0, *qanp_, 0.0);
  Wfn_->update(1.0, *Wfnp_, 0.0);
  Wbn_->update(1.0, *Wbnp_, 0.0);

  if (solvescatra_)
  {
    //    scatraO2wfn_->Update(1.0,*scatraO2wfnp_ ,0.0);
    //    scatraO2wbn_->Update(1.0,*scatraO2wbnp_ ,0.0);
    scatraO2nm_->update(1.0, *scatraO2n_, 0.0);
    scatraO2n_->update(1.0, *scatraO2np_, 0.0);
  }

  return;
}  // ArtNetExplicitTimeInt::TimeUpdate


/*----------------------------------------------------------------------*
 | Initializes state saving vectors                                     |
 |                                                                      |
 |  This is currently needed for strongly coupling 3D-1D fields         |
 |                                                                      |
 |                                                          ismail 04/14|
 *----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::init_save_state()
{
  // get the discretizations DOF row map
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // Volumetric Flow rate/Cross-sectional area of this step become most recent
  saved_qanp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  saved_qan_ = Core::LinAlg::create_vector(*dofrowmap, true);
  saved_qanm_ = Core::LinAlg::create_vector(*dofrowmap, true);

  saved_Wfnp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  saved_Wfn_ = Core::LinAlg::create_vector(*dofrowmap, true);
  saved_Wfnm_ = Core::LinAlg::create_vector(*dofrowmap, true);

  saved_Wbnp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  saved_Wbn_ = Core::LinAlg::create_vector(*dofrowmap, true);
  saved_Wbnm_ = Core::LinAlg::create_vector(*dofrowmap, true);

  if (solvescatra_)
  {
    saved_scatraO2np_ = Core::LinAlg::create_vector(*dofrowmap, true);
    saved_scatraO2n_ = Core::LinAlg::create_vector(*dofrowmap, true);
    saved_scatraO2nm_ = Core::LinAlg::create_vector(*dofrowmap, true);
  }

  return;
}  // ArtNetExplicitTimeInt::InitSaveState


/*----------------------------------------------------------------------*
 | Saves and backs up the current state.                                |
 |                                                                      |
 |  This is currently needed for strongly coupling 3D-0D fields          |
 |  example:                                                            |
 |  saved_qanp_ = qanp_                                                 |
 |  saved_Wfnp_ = Wfnp_                                                 |
 |                                                                      |
 |                                                          ismail 04/14|
 *----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::save_state()
{
  // Volumetric Flow rate/Cross-sectional area of this step become most recent
  saved_qanp_->update(1.0, *qanp_, 0.0);
  saved_qan_->update(1.0, *qan_, 0.0);
  saved_qanm_->update(1.0, *qanm_, 0.0);

  saved_Wfnp_->update(1.0, *Wfnp_, 0.0);
  saved_Wfn_->update(1.0, *Wfn_, 0.0);
  saved_Wfnm_->update(1.0, *Wfnm_, 0.0);

  saved_Wbnp_->update(1.0, *Wbnp_, 0.0);
  saved_Wbn_->update(1.0, *Wbn_, 0.0);
  saved_Wbnm_->update(1.0, *Wbnm_, 0.0);

  if (solvescatra_)
  {
    saved_scatraO2np_->update(1.0, *scatraO2np_, 0.0);
    saved_scatraO2n_->update(1.0, *scatraO2n_, 0.0);
    saved_scatraO2nm_->update(1.0, *scatraO2nm_, 0.0);
  }

  return;
}  // ArtNetExplicitTimeInt::SaveState


/*----------------------------------------------------------------------*
 | Loads backed up states.                                              |
 |                                                                      |
 |  This is currently needed for strongly coupling 3D-0D fields          |
 |  example:                                                            |
 |  qanp_   =  saved_qanp_                                              |
 |  Wfnp_   =  saved_Wfnp_                                              |
 |                                                                      |
 |                                                          ismail 04/14|
 *----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::load_state()
{
  // Volumetric Flow rate/Cross-sectional area of this step become most recent
  qanp_->update(1.0, *saved_qanp_, 0.0);
  qan_->update(1.0, *saved_qan_, 0.0);
  qanm_->update(1.0, *saved_qanm_, 0.0);

  Wfnp_->update(1.0, *saved_Wfnp_, 0.0);
  Wfn_->update(1.0, *saved_Wfn_, 0.0);
  Wfnm_->update(1.0, *saved_Wfnm_, 0.0);

  Wbnp_->update(1.0, *saved_Wbnp_, 0.0);
  Wbn_->update(1.0, *saved_Wbn_, 0.0);
  Wbnm_->update(1.0, *saved_Wbnm_, 0.0);

  if (solvescatra_)
  {
    scatraO2np_->update(1.0, *saved_scatraO2np_, 0.0);
    scatraO2n_->update(1.0, *saved_scatraO2n_, 0.0);
    scatraO2nm_->update(1.0, *saved_scatraO2nm_, 0.0);
  }

  return;
}  // ArtNetExplicitTimeInt::LoadState

void Arteries::ArtNetExplicitTimeInt::collect_runtime_output_data(bool CoupledTo3D, int step)
{
  visualization_writer_->append_result_data_vector_with_context(
      *qanp_, Core::IO::OutputEntity::dof, {"qanp"});

  // write domain decomposition for visualization (only once!)
  if (step_ == upres_) visualization_writer_->append_element_owner("Owner");

  this->calc_postprocessing_values();

  visualization_writer_->append_result_data_vector_with_context(
      *qn_, Core::IO::OutputEntity::node, {"one_d_artery_flow"});

  visualization_writer_->append_result_data_vector_with_context(
      *on_, Core::IO::OutputEntity::node, {"one_d_artery_pressure"});

  visualization_writer_->append_result_data_vector_with_context(
      *an_, Core::IO::OutputEntity::node, {"one_d_artery_area"});

  if (solvescatra_)
  {
    this->calc_scatra_from_scatra_fw(export_scatra_, scatraO2np_);
    visualization_writer_->append_result_data_vector_with_context(
        *export_scatra_, Core::IO::OutputEntity::node, {"one_d_o2_scatra"});
  }


  visualization_writer_->append_result_data_vector_with_context(
      *Wfnp_, Core::IO::OutputEntity::node, {"forward_speed"});

  visualization_writer_->append_result_data_vector_with_context(
      *Wfo_, Core::IO::OutputEntity::node, {"forward_speed0"});

  visualization_writer_->append_result_data_vector_with_context(
      *Wbnp_, Core::IO::OutputEntity::node, {"backward_speed"});

  visualization_writer_->append_result_data_vector_with_context(
      *Wbo_, Core::IO::OutputEntity::node, {"backward_speed0"});

  if (CoupledTo3D)
  {
    output_.write_int("Actual_RedD_step", step);
  }
}

void Arteries::ArtNetExplicitTimeInt::output_restart(bool CoupledTo3D, int step)
{
  // step number and time (only after that data output is possible)
  output_.new_step(step_, time_);

  // "volumetric flow rate/cross-sectional area" vector
  output_.write_vector("qanp", qanp_);

  // Export postpro results
  this->calc_postprocessing_values();
  output_.write_vector("one_d_artery_flow", qn_);
  output_.write_vector("one_d_artery_pressure", on_);
  output_.write_vector("one_d_artery_area", an_);

  if (solvescatra_)
  {
    this->calc_scatra_from_scatra_fw(export_scatra_, scatraO2np_);
    output_.write_vector("one_d_o2_scatra", export_scatra_);
  }


  output_.write_vector("forward_speed", Wfnp_);
  output_.write_vector("forward_speed0", Wfo_);
  output_.write_vector("backward_speed", Wbnp_);
  output_.write_vector("backward_speed0", Wbo_);

  if (CoupledTo3D)
  {
    output_.write_int("Actual_RedD_step", step);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::output(
    bool CoupledTo3D, std::shared_ptr<Teuchos::ParameterList> CouplingParams)
{
  int step = 0;
  int upres = 0;
  int uprestart = 0;
  double time_backup = 0.0;

  // -------------------------------------------------------------------
  // if coupled to 3D problem, then get the export information from
  // the 3D problem
  // -------------------------------------------------------------------

  if (CoupledTo3D)
  {
    step = step_;
    upres = upres_;
    uprestart = uprestart_;
    time_backup = time_;
    step_ = CouplingParams->get<int>("step");
    upres_ = CouplingParams->get<int>("upres");
    uprestart_ = CouplingParams->get<int>("uprestart");
    time_ = CouplingParams->get<double>("time");
  }

  if (step_ % upres_ == 0)
  {
    visualization_writer_->reset();

    collect_runtime_output_data(CoupledTo3D, step);

    visualization_writer_->write_to_disk(time_, step_);
  }
  if (uprestart_ != 0 && step_ % uprestart_ == 0)
  {
    output_restart(CoupledTo3D, step);
  }

  // -------------------------------------------------------------------
  // if coupled to 3D problem, then retrieve the old information of the
  // the reduced model problem
  // -------------------------------------------------------------------
  if (CoupledTo3D)
  {
    step_ = step;
    upres_ = upres;
    uprestart_ = uprestart;
    time_ = time_backup;
  }


}  // ArteryExplicitTimeInt::Output


/*----------------------------------------------------------------------*
 | read_restart (public)                                     ismail 07/09|
 -----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::read_restart(int step, bool coupledTo3D)
{
  coupledTo3D_ = coupledTo3D;
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);

  time_ = reader.read_double("time");

  if (coupledTo3D_)
  {
    step_ = reader.read_int("Actual_RedD_step");
  }
  else
  {
    step_ = reader.read_int("step");
  }

  reader.read_vector(qanp_, "qanp");
}



/*----------------------------------------------------------------------*
 | Calculate the post processing values (public)            ismail 04/10|
 *----------------------------------------------------------------------*/
void Arteries::ArtNetExplicitTimeInt::calc_postprocessing_values()
{
  //  std::cout<<"On proc("<<myrank_<<"): "<<"postpro values being calculated"<<std::endl;

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<Arteries::Action>("action", Arteries::calc_postpro_vals);

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state("qanp", *qanp_);

  eleparams.set("time step size", dta_);
  eleparams.set("total time", time_);
  eleparams.set("pressure", on_);
  eleparams.set("art_area", an_);
  eleparams.set("flow", qn_);
  //  std::cout<<"On proc("<<myrank_<<"): "<<"postpro evaluat disc"<<std::endl;
  // call standard loop over all elements
  discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  //  std::cout<<"On proc("<<myrank_<<"): "<<"postpro done "<<std::endl;
}  // Arteries::ArtNetExplicitTimeInt::calc_postprocessing_values


void Arteries::ArtNetExplicitTimeInt::calc_scatra_from_scatra_fw(
    std::shared_ptr<Core::LinAlg::Vector<double>> scatra,
    std::shared_ptr<Core::LinAlg::Vector<double>> scatra_fb)
{
  scatra->put_scalar(0.0);

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<Arteries::Action>("action", Arteries::calc_scatra_from_scatra_fb);

  // set vector values needed by elements
  discret_->clear_state();
  eleparams.set("scatra", scatra);
  eleparams.set("scatra_fb", scatra_fb);

  // call standard loop over all elements
  discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
}

void Arteries::ArtNetExplicitTimeInt::test_results()
{
  std::shared_ptr<Core::Utils::ResultTest> resulttest = create_field_test();
  Global::Problem::instance()->add_field_test(resulttest);
  Global::Problem::instance()->test_all(discret_->get_comm());
}

/*----------------------------------------------------------------------*
 | create result test for this field                   kremheller 03/18 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> Arteries::ArtNetExplicitTimeInt::create_field_test()
{
  return std::make_shared<Arteries::ArteryResultTest>(*(this));
}

FOUR_C_NAMESPACE_CLOSE
