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
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastAlgorithm::PorofluidElastAlgorithm(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : AlgorithmBase(comm, globaltimeparams),
      structure_algo_(nullptr),
      porofluid_algo_(nullptr),
      solve_structure_(true)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::init(
    const Teuchos::ParameterList& global_time_params,
    const Teuchos::ParameterList& porofluid_elast_params,
    const Teuchos::ParameterList& structure_params, const Teuchos::ParameterList& porofluid_params,
    const std::string& structure_disname, const std::string& porofluid_disname, bool isale,
    int nds_disp, int nds_vel, int nds_solidpressure, int nds_porofluid_scatra,
    const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  const auto& algorithm_deps = this->algorithm_deps();
  FOUR_C_ASSERT_ALWAYS(
      algorithm_deps.discretization_by_name, "Discretization callback is not initialized.");

  // Create the two uncoupled subproblems.
  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structure_dis =
      algorithm_deps.discretization_by_name(structure_disname);

  // build underlying structure algorithm
  std::shared_ptr<Adapter::StructureBaseAlgorithmNew> structure_adapter_algo =
      Adapter::build_structure_algorithm(structure_params);

  // Translate updated porofluid input format to old structure format
  Teuchos::ParameterList structure_global_time_params;
  structure_global_time_params.set<double>(
      "TIMESTEP", global_time_params.sublist("time_integration").get<double>("time_step_size"));
  structure_global_time_params.set(
      "MAXTIME", global_time_params.get<double>("total_simulation_time"));
  structure_global_time_params.set(
      "NUMSTEP", global_time_params.sublist("time_integration").get<int>("number_of_time_steps"));
  structure_global_time_params.set(
      "RESULTSEVERY", global_time_params.sublist("output").get<int>("result_data_every"));
  structure_global_time_params.set(
      "RESTARTEVERY", global_time_params.sublist("output").get<int>("restart_data_every"));

  structure_adapter_algo->init(structure_global_time_params,
      const_cast<Teuchos::ParameterList&>(structure_params), structure_dis);
  structure_adapter_algo->setup();
  structure_algo_ = structure_adapter_algo->structure_field();

  // true if we solve the structure field
  // (false in case of porofluid-scatra coupling without mesh deformation)
  solve_structure_ = porofluid_elast_params.get<bool>("solve_structure");

  // get the solver number used for ScalarTransport solver
  const int linsolvernumber =
      porofluid_elast_params.sublist("nonlinear_solver").get<int>("linear_solver_id");

  // access the fluid discretization
  std::shared_ptr<Core::FE::Discretization> porofluid_dis =
      algorithm_deps.discretization_by_name(porofluid_disname);

  // set degrees of freedom in the discretization
  if (!porofluid_dis->filled()) porofluid_dis->fill_complete();

  // context for output and restart
  std::shared_ptr<Core::IO::DiscretizationWriter> output = porofluid_dis->writer();
  output->write_mesh(0, 0.0);

  // algorithm construction depending on time-integration scheme
  auto time_integration_scheme =
      Teuchos::getIntegralValue<PoroPressureBased::TimeIntegrationScheme>(
          porofluid_params.sublist("time_integration"), "scheme");

  // build porofluid algorithm
  std::shared_ptr<Adapter::PoroFluidMultiphase> porofluid_algo =
      PoroPressureBased::create_algorithm(time_integration_scheme, porofluid_dis, linsolvernumber,
          global_time_params, porofluid_params, output, algorithm_deps.porofluid_algorithm_deps);

  porofluid_algo_ = std::make_shared<Adapter::PoroFluidMultiphaseWrapper>(porofluid_algo);
  porofluid_algo_->init(
      isale, nds_disp, nds_vel, nds_solidpressure, nds_porofluid_scatra, nearby_ele_pairs);
}

/*----------------------------------------------------------------------*
----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::post_init()
{
  // call post_setup routine of the underlying structure algorithm
  structure_algo_->post_setup();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::read_restart(const int restart)
{
  if (restart)
  {
    // read restart data for structure field (will set time and step internally)
    structure_algo_->read_restart(restart);

    // read restart data for fluid field (will set time and step internally)
    porofluid_algo_->read_restart(restart);

    // reset time and step for the global algorithm
    set_time_step(structure_algo_->time_old(), restart);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::time_loop()
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
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::prepare_time_loop()
{
  // initial output
  if (solve_structure_)
  {
    constexpr bool force_prepare = true;
    structure_algo()->prepare_output(force_prepare);
    structure_algo()->output();
    set_structure_solution(structure_algo()->dispnp(), structure_algo()->velnp());
  }
  else
  {
    // inform the user that the structure field has been disabled
    print_structure_disabled_info();
    // directly set displacements and velocities to zero
    set_structure_solution(
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_algo_->dof_row_map(), true),
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_algo_->dof_row_map(), true));
  }
  porofluid_algo()->prepare_time_loop();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::prepare_time_step()
{
  increment_time_and_step();

  structure_algo()->discretization()->set_state(1, "porofluid", *porofluid_algo()->phinp());

  if (solve_structure_)
  {
    // NOTE: the predictor of the structure is called in here
    structure_algo()->prepare_time_step();
    set_structure_solution(structure_algo()->dispnp(), structure_algo()->velnp());
  }
  else
    set_structure_solution(
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_algo_->dof_row_map(), true),
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_algo_->dof_row_map(), true));

  porofluid_algo()->prepare_time_step();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::create_field_test()
{
  const auto& algorithm_deps = this->algorithm_deps();
  FOUR_C_ASSERT_ALWAYS(algorithm_deps.add_field_test, "Result test callback is not initialized.");

  algorithm_deps.add_field_test(structure_algo_->create_field_test());
  algorithm_deps.add_field_test(porofluid_algo_->create_field_test());
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::set_structure_solution(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel) const
{
  set_mesh_disp(std::move(disp));
  set_velocity_fields(std::move(vel));
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::set_velocity_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel) const
{
  porofluid_algo_->set_velocity_field(std::move(vel));
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::set_scatra_solution(
    unsigned nds, std::shared_ptr<const Core::LinAlg::Vector<double>> scalars) const
{
  porofluid_algo_->set_scatra_solution(nds, std::move(scalars));
}


/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::set_mesh_disp(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp) const
{
  porofluid_algo_->apply_mesh_movement(std::move(disp));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::update_and_output()
{
  // prepare the output
  constexpr bool force_prepare = false;
  structure_algo()->prepare_output(force_prepare);

  // update single fields
  structure_algo()->update();
  porofluid_algo()->update();

  // evaluate error if desired
  porofluid_algo()->evaluate_error_compared_to_analytical_sol();

  // set structure on fluid (necessary for possible domain integrals)
  set_structure_solution(structure_algo()->dispnp(), structure_algo()->velnp());

  // output single fields
  structure_algo()->output();
  porofluid_algo()->output();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastAlgorithm::structure_dof_row_map() const
{
  return structure_algo_->dof_row_map();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastAlgorithm::porofluid_dof_row_map() const
{
  return porofluid_algo_->dof_row_map();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
PoroPressureBased::PorofluidElastAlgorithm::artery_porofluid_sysmat() const
{
  return porofluid_algo_->artery_porofluid_sysmat();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastAlgorithm::artery_dof_row_map() const
{
  return porofluid_algo_->artery_dof_row_map();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::PorofluidElastAlgorithm::structure_dispnp() const
{
  return structure_algo_->dispnp();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MultiVector<double>>
PoroPressureBased::PorofluidElastAlgorithm::fluid_flux() const
{
  return porofluid_algo_->flux();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::PorofluidElastAlgorithm::fluid_phinp() const
{
  return porofluid_algo_->phinp();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::PorofluidElastAlgorithm::solid_pressure() const
{
  return porofluid_algo_->solid_pressure();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastAlgorithm::print_structure_disabled_info() const
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
