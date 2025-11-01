// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_impl_stationary.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_art_net_artery_ele_action.hpp"
#include "4C_art_net_artery_resulttest.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_print.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_scatra_resulttest.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN



//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//

Arteries::ArtNetImplStationary::ArtNetImplStationary(
    std::shared_ptr<Core::FE::Discretization> actdis, const int linsolvernumber,
    const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& artparams,
    Core::IO::DiscretizationWriter& output)
    : TimInt(actdis, linsolvernumber, probparams, artparams, output)
{
  const int restart_step = Global::Problem::instance()->restart();
  if (restart_step > 0)
  {
    Core::IO::DiscretizationReader reader(
        *discret_, Global::Problem::instance()->input_control_file(), restart_step);

    time_ = reader.read_double("time");
  }

  visualization_writer_ = std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(
      actdis, Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), time_));


}  // ArtNetImplStationary::ArtNetImplStationary



/*----------------------------------------------------------------------*
 | Initialize the time integration.                                     |
 |                                                      kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::init(const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& arteryparams, const std::string& scatra_disname)
{
  // time measurement: initialization
  TEUCHOS_FUNC_TIME_MONITOR(" + initialization");

  if (coupledTo3D_)
    FOUR_C_THROW("this type of coupling is only available for explicit time integration");

  // call base class
  TimInt::init(globaltimeparams, arteryparams, scatra_disname);

  // ensure that degrees of freedom in the discretization have been set
  if ((not discret_->filled()) or (not discret_->have_dofs())) discret_->fill_complete();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices: local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // -------------------------------------------------------------------
  // create empty system matrix (6 adjacent nodes as 'good' guess)
  // -------------------------------------------------------------------
  sysmat_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*(discret_->dof_row_map()), 3, false, true);

  // right hand side vector
  rhs_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // -------------------------------------------------------------------
  // create vectors associated to boundary conditions
  // -------------------------------------------------------------------
  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  zeros_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>();
  {
    Teuchos::ParameterList eleparams;
    // other parameters needed by the elements
    eleparams.set("total time", time_);
    eleparams.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    discret_->evaluate_dirichlet(eleparams, zeros_, nullptr, nullptr, nullptr, dbcmaps_);
    zeros_->put_scalar(0.0);  // just in case of change
  }

  // the vector containing body and surface forces
  neumann_loads_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // -------------------------------------------------------------------
  // create vectors containing problem variables
  // -------------------------------------------------------------------
  // solutions at time n+1
  pressurenp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  pressureincnp_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // for output of volumetric flow
  ele_volflow_ = Core::LinAlg::create_vector(*discret_->element_row_map());

  // for output of element radius
  ele_radius_ = Core::LinAlg::create_vector(*discret_->element_row_map());

  // -------------------------------------------------------------------
  // set initial field
  // -------------------------------------------------------------------
  set_initial_field(Teuchos::getIntegralValue<ArtDyn::InitialField>(arteryparams, "INITIALFIELD"),
      arteryparams.get<int>("INITFUNCNO"));


  if (solvescatra_)
  {
    const Teuchos::ParameterList& myscatraparams =
        Global::Problem::instance()->scalar_transport_dynamic_params();
    if (Teuchos::getIntegralValue<Inpar::ScaTra::VelocityField>(myscatraparams, "VELOCITYFIELD") !=
        Inpar::ScaTra::velocity_zero)
      FOUR_C_THROW("set your velocity field to zero!");
    // construct the scatra problem
    scatra_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(globaltimeparams, myscatraparams,
        Global::Problem::instance()->solver_params(linsolvernumber_), scatra_disname, false);

    // initialize the base algo.
    // scatra time integrator is initialized inside.
    scatra_->init();

    // only now we must call setup() on the scatra time integrator.
    // all objects relying on the parallel distribution are
    // created and pointers are set.
    // calls setup() on the scatra time integrator inside.
    scatra_->scatra_field()->setup();
  }
}



/*----------------------------------------------------------------------*
 | (Linear) Solve.                                                      |
 |                                                      kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::solve(
    std::shared_ptr<Teuchos::ParameterList> CouplingTo3DParams)
{
  // time measurement: initialization
  TEUCHOS_FUNC_TIME_MONITOR(" + solve");

  if (coupledTo3D_)
    FOUR_C_THROW("this type of coupling is only available for implicit time integration");

  // call elements to calculate system matrix and rhs and assemble
  assemble_mat_and_rhs();

  // Prepare Linear Solve (Apply DBC)
  prepare_linear_solve();

  // solve linear system of equations
  linear_solve();
}


/*----------------------------------------------------------------------*
 | (Linear) Solve for ScaTra.                                           |
 |                                                      kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::solve_scatra()
{
  // print user info
  if (Core::Communication::my_mpi_rank(discretization()->get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "<      Scalar Transport in 1D Artery Network       >" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "\n";
  }

  // time measurement: initialization
  TEUCHOS_FUNC_TIME_MONITOR(" + solve scatra");

  if (coupledTo3D_)
    FOUR_C_THROW("this type of coupling is only available for explicit time integration");

  // provide scatra discretization with fluid primary variable field
  scatra_->scatra_field()->discretization()->set_state(1, "one_d_artery_pressure", *pressurenp_);
  scatra_->scatra_field()->prepare_time_step();

  // -------------------------------------------------------------------
  //                  solve nonlinear / linear equation
  // -------------------------------------------------------------------
  scatra_->scatra_field()->solve();
}

/*----------------------------------------------------------------------*
 | Prepare Linear Solve (Apply DBC).                                    |
 |                                                      kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::prepare_linear_solve()
{
  // apply map: rhs = pressurenp_
  Core::LinAlg::apply_dirichlet_to_system(
      *sysmat_, *pressureincnp_, *rhs_, *zeros_, *(dbcmaps_->cond_map()));
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | call elements to calculate system matrix/rhs and assemble            |
 |                                                     kremheller 03/18 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::assemble_mat_and_rhs()
{
  dtele_ = 0.0;

  TEUCHOS_FUNC_TIME_MONITOR("      + element calls");


  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // set both system matrix and rhs vector to zero
  sysmat_->zero();
  rhs_->put_scalar(0.0);

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<Arteries::Action>("action", Arteries::calc_sys_matrix_rhs);

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state(0, "pressurenp", *pressurenp_);

  // call standard loop over all elements
  discret_->evaluate(eleparams, sysmat_, rhs_);
  discret_->clear_state();

  // potential addition of Neumann terms
  add_neumann_to_residual();

  // finalize the complete matrix
  sysmat_->complete();

  // end time measurement for element
  double mydtele = Teuchos::Time::wallTime() - tcpuele;
  dtele_ = Core::Communication::max_all(mydtele, discret_->get_comm());

}  // ArtNetExplicitTimeInt::assemble_mat_and_rhs

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | call linear solver                                                   |
 |                                                     kremheller 03/18 |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::linear_solve()
{
  // time measurement: solver
  TEUCHOS_FUNC_TIME_MONITOR("      + solver");

  // get cpu time
  const double tcpusolve = Teuchos::Time::wallTime();

  // linear solve
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(sysmat_, pressureincnp_, rhs_, solver_params);
  // note: incremental form since rhs-coupling with poromultielastscatra-framework might be
  //       nonlinear
  pressurenp_->update(1.0, *pressureincnp_, 1.0);

  // end time measurement for solver
  double mydtsolve = Teuchos::Time::wallTime() - tcpusolve;
  dtsolve_ = Core::Communication::max_all(mydtsolve, discret_->get_comm());

}  // ArtNetImplStationary::linear_solve

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | Prepare time step (Apply DBC and Neumann)            kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::prepare_time_step()
{
  // call base class
  Arteries::TimInt::prepare_time_step();

  // Apply DBC
  apply_dirichlet_bc();

  // Apply Neumann
  apply_neumann_bc(*neumann_loads_);
}

/*----------------------------------------------------------------------*
 | evaluate Dirichlet boundary conditions at t_{n+1}   kremheller 03/18 |
 *----------------------------------------------------------------------*/
void Arteries::ArtNetImplStationary::apply_dirichlet_bc()
{
  // time measurement: apply Dirichlet conditions
  TEUCHOS_FUNC_TIME_MONITOR("      + apply dirich cond.");

  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time_);  // actual time t_{n+1}
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // Dirichlet values
  // \c  pressurenp_ then also holds prescribed new Dirichlet values
  discret_->clear_state();
  discret_->evaluate_dirichlet(p, pressurenp_, nullptr, nullptr, nullptr, dbcmaps_);
  discret_->clear_state();
}

/*----------------------------------------------------------------------*
 | reset artery diameter of previous time step         kremheller 11/20 |
 *----------------------------------------------------------------------*/
void Arteries::ArtNetImplStationary::reset_artery_diam_previous_time_step()
{
  // set the diameter in material
  for (int i = 0; i < discret_->num_my_col_elements(); ++i)
  {
    // pointer to current element
    Core::Elements::Element* actele = discret_->l_col_element(i);

    // get the artery-material
    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(actele->material());
    if (arterymat == nullptr) FOUR_C_THROW("cast to artery material failed");

    const double diam = arterymat->diam();
    arterymat->set_diam_previous_time_step(diam);
  }
}

/*----------------------------------------------------------------------*
 | evaluate Neumann boundary conditions                kremheller 03/18 |
 *----------------------------------------------------------------------*/
void Arteries::ArtNetImplStationary::apply_neumann_bc(Core::LinAlg::Vector<double>& neumann_loads)
{
  // prepare load vector
  neumann_loads.put_scalar(0.0);

  // create parameter list
  Teuchos::ParameterList condparams;
  condparams.set("total time", time_);
  condparams.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // evaluate Neumann boundary conditions
  discret_->evaluate_neumann(condparams, neumann_loads);
  discret_->clear_state();

  return;
}  // ArtNetImplStationary::apply_neumann_bc

/*----------------------------------------------------------------------*
 | add actual Neumann loads                            kremheller 03/18 |
 *----------------------------------------------------------------------*/
void Arteries::ArtNetImplStationary::add_neumann_to_residual()
{
  rhs_->update(1.0, *neumann_loads_, 1.0);
  return;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                      kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::time_update()
{
  // reset the artery diameter of the previous time step
  reset_artery_diam_previous_time_step();

  if (solvescatra_)
  {
    scatra_->scatra_field()->update();
    scatra_->scatra_field()->evaluate_error_compared_to_analytical_sol();
  }

  return;
}  // ArtNetExplicitTimeInt::TimeUpdate

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | prepare the time loop                                kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::prepare_time_loop()
{
  // call base class
  Arteries::TimInt::prepare_time_loop();

  // provide information about initial field (do not do for restarts!)
  if (step_ == 0)
  {
    // set artery diameter of previous time step to initial artery diameter
    reset_artery_diam_previous_time_step();
    // write out initial state
    output(false, nullptr);
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Arteries::ArtNetImplStationary::collect_runtime_output_data()
{
  // write domain decomposition for visualization (only once at step 0!)
  if (step_ == 0) visualization_writer_->append_element_owner("Owner");

  // radius
  get_radius();

  visualization_writer_->append_result_data_vector_with_context(
      *ele_radius_, Core::IO::OutputEntity::element, {"ele_radius"});

  // "pressure in the arteries" vector
  visualization_writer_->append_result_data_vector_with_context(
      *pressurenp_, Core::IO::OutputEntity::dof, {"pressure"});

  // flow
  reconstruct_flow();

  visualization_writer_->append_result_data_vector_with_context(
      *ele_volflow_, Core::IO::OutputEntity::element, {"ele_volflow"});
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Arteries::ArtNetImplStationary::output_restart()
{
  // step number and time (only after that data output is possible)
  output_.new_step(step_, time_);

  // solution
  output_.write_vector("one_d_artery_pressure", pressurenp_);
  output_.write_vector("ele_radius", ele_radius_, Core::IO::elementvector);
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | output of solution vector to binio                   kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::output(
    bool CoupledTo3D, std::shared_ptr<Teuchos::ParameterList> CouplingParams)
{
  // time measurement: output of solution
  TEUCHOS_FUNC_TIME_MONITOR("             + output of solution");

  // solution output and potentially restart data
  if (step_ % upres_ == 0)
  {
    visualization_writer_->reset();

    collect_runtime_output_data();

    visualization_writer_->write_to_disk(time_, step_);
  }

  if (solvescatra_) scatra_->scatra_field()->check_and_write_output_and_restart();

  if (step_ % uprestart_ == 0 and step_ != 0) output_restart();
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | output of element-based radius                       kremheller 07/19|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::get_radius()
{
  // loop over row elements
  const int numrowele = discret_->num_my_row_elements();
  for (int i = 0; i < numrowele; ++i)
  {
    Core::Elements::Element* actele = discret_->l_row_element(i);
    // cast the material to artery material material
    const std::shared_ptr<const Mat::Cnst1dArt>& arterymat =
        std::dynamic_pointer_cast<const Mat::Cnst1dArt>(actele->material());
    if (arterymat == nullptr)
      FOUR_C_THROW("cast to Mat::Cnst1dArt failed during output of radius!");
    const double radius = arterymat->diam() / 2.0;
    ele_radius_->replace_global_value(actele->id(), radius);
  }
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | output of element volumetric flow                    kremheller 09/19|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::reconstruct_flow()
{
  Core::LinAlg::SerialDenseMatrix dummyMat;
  Core::LinAlg::SerialDenseVector dummyVec;

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state(0, "pressurenp", *pressurenp_);

  // enough to loop over row nodes since element-based quantity
  for (int i = 0; i < discret_->num_my_row_elements(); ++i)
  {
    // pointer to current element
    Core::Elements::Element* actele = discret_->l_row_element(i);

    // list to define routines at elementlevel
    Teuchos::ParameterList p;
    p.set<Arteries::Action>("action", Arteries::calc_flow_pressurebased);

    Core::Elements::LocationArray la(discret_->num_dof_sets());
    actele->location_vector(*discret_, la);
    Core::LinAlg::SerialDenseVector flowVec(1);

    actele->evaluate(p, *discret_, la, dummyMat, dummyMat, flowVec, dummyVec, dummyVec);

    ele_volflow_->replace_local_value(i, flowVec(0));
  }
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | test results                                         kremheller 03/18|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::test_results()
{
  std::shared_ptr<Core::Utils::ResultTest> resulttest = create_field_test();
  Global::Problem::instance()->add_field_test(resulttest);
  if (solvescatra_)
  {
    Global::Problem::instance()->add_field_test(scatra_->create_scatra_field_test());
  }
  Global::Problem::instance()->test_all(discret_->get_comm());
}

/*----------------------------------------------------------------------*
 | create result test for this field                   kremheller 03/18 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> Arteries::ArtNetImplStationary::create_field_test()
{
  return std::make_shared<Arteries::ArteryResultTest>(*(this));
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | read_restart (public)                                 kremheller 03/18|
 -----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::ArtNetImplStationary::read_restart(int step, bool coupledTo3D)
{
  coupledTo3D_ = coupledTo3D;
  Core::IO::DiscretizationReader reader(
      *discret_, Global::Problem::instance()->input_control_file(), step);

  if (step != reader.read_int("step")) FOUR_C_THROW("Time step on file not equal to given step");

  time_ = reader.read_double("time");
  step_ = reader.read_int("step");

  reader.read_vector(pressurenp_, "one_d_artery_pressure");

  // read restart for diameter of previous time step
  reader.read_vector(ele_radius_, "ele_radius");
  std::shared_ptr<Core::LinAlg::Vector<double>> ele_radius_col =
      Core::LinAlg::create_vector(*discret_->element_col_map(), true);
  Core::LinAlg::export_to(*ele_radius_, *ele_radius_col);

  // set the diameter in material
  for (int i = 0; i < discret_->num_my_col_elements(); ++i)
  {
    // pointer to current element
    Core::Elements::Element* actele = discret_->l_col_element(i);

    // get the artery-material
    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(actele->material());
    if (arterymat == nullptr) FOUR_C_THROW("cast to artery material failed");

    const double diam = 2.0 * ele_radius_col->local_values_as_span()[i];

    // reset (if element is collapsed in previous step, set to zero)
    arterymat->set_diam_previous_time_step(diam);
    arterymat->set_diam(diam);
    if (diam < arterymat->collapse_threshold()) arterymat->set_diam(0.0);
  }

  if (solvescatra_)
    // read restart data for scatra field
    scatra_->scatra_field()->read_restart(step);
}

/*----------------------------------------------------------------------*
 |  set initial field for pressure                     kremheller 04/18 |
 *----------------------------------------------------------------------*/
void Arteries::ArtNetImplStationary::set_initial_field(
    const ArtDyn::InitialField init, const int startfuncno)
{
  switch (init)
  {
    case ArtDyn::initfield_zero_field:
    {
      pressurenp_->put_scalar(0.0);
      break;
    }
    case ArtDyn::initfield_field_by_function:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        int numdofs = nodedofset.size();
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);
          // evaluate component k of spatial function
          double initialval = Global::Problem::instance()
                                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                                  .evaluate(lnode.x(), time_, k);
          pressurenp_->replace_local_value(doflid, initialval);
        }
      }

      break;
    }
    case ArtDyn::initfield_field_by_condition:
    {
      // set initial field
      const std::string field = "Artery";
      std::vector<int> localdofs;
      localdofs.push_back(0);

      discret_->evaluate_initial_field(
          Global::Problem::instance()->function_manager(), field, *pressurenp_, localdofs);

      break;
    }
    default:
      FOUR_C_THROW("Unknown option for initial field: {}", init);
      break;
  }  // switch(init)

}  // ArtNetImplStationary::SetInitialField

FOUR_C_NAMESPACE_CLOSE
