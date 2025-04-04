// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_base.hpp"

#include "4C_adapter_porofluid_pressure_based_wrapper.hpp"
#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                              vuong 08/16  |
 *----------------------------------------------------------------------*/
POROMULTIPHASE::PoroMultiPhaseBase::PoroMultiPhaseBase(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : AlgorithmBase(comm, globaltimeparams),
      structure_(nullptr),
      fluid_(nullptr),
      struct_zeros_(nullptr),
      solve_structure_(true),
      artery_coupl_(globaltimeparams.get<bool>("ARTERY_COUPLING"))
{
}

/*----------------------------------------------------------------------*
 | initialize algorithm                                    vuong 08/16  |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::init(const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& structparams,
    const Teuchos::ParameterList& fluidparams, const std::string& struct_disname,
    const std::string& fluid_disname, bool isale, int nds_disp, int nds_vel, int nds_solidpressure,
    int ndsporofluid_scatra, const std::map<int, std::set<int>>* nearbyelepairs)
{
  // access the global problem
  Global::Problem* problem = Global::Problem::instance();

  // Create the two uncoupled subproblems.
  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis(struct_disname);

  // build structural time integrator
  std::shared_ptr<Adapter::StructureBaseAlgorithmNew> adapterbase =
      Adapter::build_structure_algorithm(structparams);
  adapterbase->init(globaltimeparams, const_cast<Teuchos::ParameterList&>(structparams), structdis);
  adapterbase->setup();
  structure_ = adapterbase->structure_field();

  // initialize zero vector for convenience
  struct_zeros_ = Core::LinAlg::create_vector(*structure_->dof_row_map(), true);
  // do we also solve the structure, this is helpful in case of fluid-scatra coupling without mesh
  // deformation
  solve_structure_ = algoparams.get<bool>("SOLVE_STRUCTURE");

  // get the solver number used for ScalarTransport solver
  const int linsolvernumber = fluidparams.get<int>("LINEAR_SOLVER");


  // -------------------------------------------------------------------
  // access the fluid discretization
  // -------------------------------------------------------------------
  std::shared_ptr<Core::FE::Discretization> fluiddis =
      Global::Problem::instance()->get_dis(fluid_disname);

  // -------------------------------------------------------------------
  // set degrees of freedom in the discretization
  // -------------------------------------------------------------------
  if (!fluiddis->filled()) fluiddis->fill_complete();

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  std::shared_ptr<Core::IO::DiscretizationWriter> output = fluiddis->writer();
  output->write_mesh(0, 0.0);

  // -------------------------------------------------------------------
  // algorithm construction depending on
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------
  auto timintscheme = Teuchos::getIntegralValue<POROFLUIDMULTIPHASE::TimeIntegrationScheme>(
      fluidparams, "TIMEINTEGR");

  // build poro fluid time integrator
  std::shared_ptr<Adapter::PoroFluidMultiphase> porofluid =
      POROFLUIDMULTIPHASE::Utils::create_algorithm(
          timintscheme, fluiddis, linsolvernumber, globaltimeparams, fluidparams, output);

  // wrap it
  fluid_ = std::make_shared<Adapter::PoroFluidMultiphaseWrapper>(porofluid);
  // initialize it
  fluid_->init(isale, nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra, nearbyelepairs);

  // done.
  return;
}

/*----------------------------------------------------------------------*
----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::post_init()
{
  // call post_setup routine of the structure field
  structure_->post_setup();
}


/*----------------------------------------------------------------------*
 | read restart information for given time step (public)   vuong 08/16  |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::read_restart(int restart)
{
  if (restart)
  {
    // read restart data for structure field (will set time and step internally)
    structure_->read_restart(restart);

    // read restart data for fluid field (will set time and step internally)
    fluid_->read_restart(restart);

    // reset time and step for the global algorithm
    set_time_step(structure_->time_old(), restart);
  }


  return;
}

/*----------------------------------------------------------------------*
 | time loop                                            kremheller 03/17 |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::timeloop()
{
  // prepare the loop
  prepare_time_loop();

  // time loop
  while (not_finished())
  {
    prepare_time_step();

    time_step();

    update_and_output();
  }

  return;
}

/*----------------------------------------------------------------------*
 | prepare the time loop                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::prepare_time_loop()
{
  // initial output
  if (solve_structure_)
  {
    constexpr bool force_prepare = true;
    structure_field()->prepare_output(force_prepare);
    structure_field()->output();
    set_struct_solution(structure_field()->dispnp(), structure_field()->velnp());
  }
  else
  {
    // Inform user that structure field has been disabled
    print_structure_disabled_info();
    // just set displacements and velocities to zero
    set_struct_solution(struct_zeros_, struct_zeros_);
  }
  fluid_field()->prepare_time_loop();

  return;
}

/*----------------------------------------------------------------------*
 | prepare one time step                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::prepare_time_step()
{
  increment_time_and_step();

  structure_field()->discretization()->set_state(1, "porofluid", *fluid_field()->phinp());

  if (solve_structure_)
  {
    // NOTE: the predictor of the structure is called in here
    structure_field()->prepare_time_step();
    set_struct_solution(structure_field()->dispnp(), structure_field()->velnp());
  }
  else
    set_struct_solution(struct_zeros_, struct_zeros_);

  fluid_field()->prepare_time_step();
}

/*----------------------------------------------------------------------*
 | Test the results of all subproblems                       vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::create_field_test()
{
  Global::Problem* problem = Global::Problem::instance();

  problem->add_field_test(structure_->create_field_test());
  problem->add_field_test(fluid_->create_field_test());
}

/*------------------------------------------------------------------------*
 | communicate the solution of the structure to the fluid    vuong 08/16  |
 *------------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::set_struct_solution(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  set_mesh_disp(disp);
  set_velocity_fields(vel);
}

/*------------------------------------------------------------------------*
 | communicate the structure velocity  to the fluid           vuong 08/16  |
 *------------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::set_velocity_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  fluid_->set_velocity_field(vel);
}

/*------------------------------------------------------------------------*
 | communicate the scatra solution to the fluid             vuong 08/16  |
 *------------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::set_scatra_solution(
    unsigned nds, std::shared_ptr<const Core::LinAlg::Vector<double>> scalars)
{
  fluid_->set_scatra_solution(nds, scalars);
}


/*------------------------------------------------------------------------*
 | communicate the structure displacement to the fluid        vuong 08/16  |
 *------------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::set_mesh_disp(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp)
{
  fluid_->apply_mesh_movement(disp);
}

/*----------------------------------------------------------------------*
 | update fields and output results                         vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::update_and_output()
{
  // prepare the output
  constexpr bool force_prepare = false;
  structure_field()->prepare_output(force_prepare);

  // update single fields
  structure_field()->update();
  fluid_field()->update();

  // evaluate error if desired
  fluid_field()->evaluate_error_compared_to_analytical_sol();

  // set structure on fluid (necessary for possible domain integrals)
  set_struct_solution(structure_field()->dispnp(), structure_field()->velnp());

  // output single fields
  structure_field()->output();
  fluid_field()->output();
}

/*------------------------------------------------------------------------*
 | dof map of vector of unknowns of structure field           vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> POROMULTIPHASE::PoroMultiPhaseBase::struct_dof_row_map()
    const
{
  return structure_->dof_row_map();
}

/*------------------------------------------------------------------------*
 | dof map of vector of unknowns of fluid field           vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> POROMULTIPHASE::PoroMultiPhaseBase::fluid_dof_row_map()
    const
{
  return fluid_->dof_row_map();
}

/*------------------------------------------------------------------------*
 | coupled artery-porofluid system matrix                kremheller 05/18 |
 *------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
POROMULTIPHASE::PoroMultiPhaseBase::artery_porofluid_sysmat() const
{
  return fluid_->artery_porofluid_sysmat();
}

/*------------------------------------------------------------------------*
 | dof map of vector of unknowns of artery field         kremheller 05/18 |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> POROMULTIPHASE::PoroMultiPhaseBase::artery_dof_row_map()
    const
{
  return fluid_->artery_dof_row_map();
}

/*------------------------------------------------------------------------*
 | return structure displacements                             vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
POROMULTIPHASE::PoroMultiPhaseBase::struct_dispnp() const
{
  return structure_->dispnp();
}

/*------------------------------------------------------------------------*
 | return structure velocities                               vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
POROMULTIPHASE::PoroMultiPhaseBase::struct_velnp() const
{
  return structure_->velnp();
}

/*------------------------------------------------------------------------*
 | return fluid Flux                                         vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiVector<double>>
POROMULTIPHASE::PoroMultiPhaseBase::fluid_flux() const
{
  return fluid_->flux();
}

/*------------------------------------------------------------------------*
 | return fluid Flux                                         vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
POROMULTIPHASE::PoroMultiPhaseBase::fluid_phinp() const
{
  return fluid_->phinp();
}

/*------------------------------------------------------------------------*
 | return fluid Flux                                         vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
POROMULTIPHASE::PoroMultiPhaseBase::fluid_saturation() const
{
  return fluid_->saturation();
}

/*------------------------------------------------------------------------*
 | return fluid Flux                                         vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
POROMULTIPHASE::PoroMultiPhaseBase::fluid_pressure() const
{
  return fluid_->pressure();
}

/*------------------------------------------------------------------------*
 | return fluid Flux                                         vuong 08/16  |
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
POROMULTIPHASE::PoroMultiPhaseBase::solid_pressure() const
{
  return fluid_->solid_pressure();
}

/*----------------------------------------------------------------------*
 | inform user that structure is not solved            kremheller 04/17 |
 *----------------------------------------------------------------------*/
void POROMULTIPHASE::PoroMultiPhaseBase::print_structure_disabled_info()
{
  // print out Info
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "\n";
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                 "++++++++++++++++++++++++++++++++\n";
    std::cout << " INFO:    STRUCTURE FIELD IS NOT SOLVED   \n";
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                 "++++++++++++++++++++++++++++++++\n";
  }
}

FOUR_C_NAMESPACE_CLOSE
