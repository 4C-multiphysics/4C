// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_main.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_input_field.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_rebalance.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_boundary_conditions.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_newton_linear_solver.hpp"
#include "4C_reduced_lung_newton_solver.hpp"
#include "4C_reduced_lung_resulttest.hpp"
#include "4C_reduced_lung_solver_profiles.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_reduced_lung_tree_linear_solver.hpp"
#include "4C_reduced_lung_tree_metadata.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    /**
     * Runtime context extracted from Global::Problem for the reduced-lung simulation.
     */
    struct ReducedLungContext
    {
      ReducedLungParameters parameters;
      MPI_Comm local_comm;
      Core::Rebalance::RebalanceParameters rebalance_parameters;
      const Teuchos::ParameterList& io_parameters;
      const Teuchos::ParameterList& linear_solver_parameters;
      std::function<const Teuchos::ParameterList&(int)> solver_params_callback;
      std::shared_ptr<Core::IO::OutputControl> output_control_file;
      const Core::Utils::FunctionManager& function_manager;
      std::function<void(std::shared_ptr<Core::Utils::ResultTest>)> add_field_test;
      std::function<void(MPI_Comm)> test_all;
    };

    /**
     * Collect reduced-lung input, communicator, IO, and solver configuration from the 4C problem.
     */
    ReducedLungContext make_reduced_lung_context_from_problem(Global::Problem& problem)
    {
      const ReducedLungParameters parameters =
          problem.parameters().get<ReducedLungParameters>("reduced_dimensional_lung");

      return ReducedLungContext{
          .parameters = parameters,
          .local_comm = problem.get_communicators().local_comm(),
          .rebalance_parameters =
              Core::Rebalance::RebalanceParameters{
                  .mesh_partitioning_parameters =
                      problem.parameters().get<Core::Rebalance::MeshPartitioningParameters>(
                          "MESH PARTITIONING"),
                  .geometric_search_parameters =
                      Core::GeometricSearch::geometric_search_params_factory(problem.parameters())},
          .io_parameters = problem.io_params(),
          .linear_solver_parameters = problem.solver_params(parameters.dynamics.linear_solver),
          .solver_params_callback = problem.solver_params_callback(),
          .output_control_file = problem.output_control_file(),
          .function_manager = problem.function_manager(),
          .add_field_test = [&problem](std::shared_ptr<Core::Utils::ResultTest> result_test)
          { problem.add_field_test(std::move(result_test)); },
          .test_all = [&problem](MPI_Comm comm) { problem.test_all(comm); },
      };
    }

    /**
     * Check whether the optional NewtonTree runtime profile should be printed.
     */
    bool tree_profile_enabled_from_environment()
    {
      const char* const value = std::getenv("FOUR_C_REDUCED_LUNG_TREE_PROFILE");
      if (value == nullptr) return false;

      const std::string setting(value);
      return !setting.empty() && setting != "0" && setting != "false" && setting != "FALSE" &&
             setting != "off" && setting != "OFF";
    }

    /**
     * Return a printable name for the selected reduced-lung nonlinear solver workflow.
     */
    const char* nonlinear_solver_name(ReducedLungParameters::NonlinearSolverType solver)
    {
      using SolverType = ReducedLungParameters::NonlinearSolverType;
      switch (solver)
      {
        case SolverType::Nox:
          return "NOX";
        case SolverType::NewtonSparse:
          return "NewtonSparse";
        case SolverType::NewtonTree:
          return "NewtonTree";
      }

      return "Unknown";
    }

    /**
     * Owns setup and time integration for one reduced-lung simulation run.
     */
    class ReducedLungSimulation
    {
     public:
      /**
       * Construct the simulation from an already extracted reduced-lung runtime context.
       */
      explicit ReducedLungSimulation(const ReducedLungContext& context)
          : context_(context),
            actdis_(
                std::make_shared<Core::FE::Discretization>("reduced_lung", context.local_comm, 3)),
            comm_(context.local_comm),
            dt_(context.parameters.dynamics.time_increment),
            n_timesteps_(context.parameters.dynamics.number_of_steps),
            tree_profile_enabled_(context.parameters.dynamics.nonlinear_solver ==
                                      ReducedLungParameters::NonlinearSolverType::NewtonTree &&
                                  tree_profile_enabled_from_environment())
      {
      }

      /**
       * Build the discretization, model blocks, maps, and selected nonlinear solver.
       */
      void initialize()
      {
        validate_parameters();
        build_discretization();
        build_element_models();
        build_node_entities();
        assign_equation_ids();
        build_maps_and_local_ids();
        build_linear_system_and_solver();
      }

      /**
       * Run the reduced-lung time loop with the configured nonlinear solver.
       */
      void run()
      {
        if (Core::Communication::my_mpi_rank(comm_) == 0)
        {
          std::cout << "-------- Start Time Integration --------\n"
                    << "Reduced lung nonlinear solver: "
                    << nonlinear_solver_name(context_.parameters.dynamics.nonlinear_solver) << "\n"
                    << "----------------------------------------\n"
                    << std::flush;
        }

        {
          TEUCHOS_FUNC_TIME_MONITOR("TimeLoop");
          for (int step = 1; step <= n_timesteps_; ++step)
          {
            solve_timestep(step);
            write_output_if_due(step);
          }
        }
        print_tree_profile_summary();
        run_result_tests();
      }

     private:
      /**
       * Validate reduced-lung time-integration and nonlinear solver parameters.
       */
      void validate_parameters() const
      {
        const auto& dynamics = context_.parameters.dynamics;

        if (dynamics.time_increment <= 0.0)
        {
          FOUR_C_THROW(
              "Reduced lung time_increment must be positive, got {}.", dynamics.time_increment);
        }
        if (dynamics.number_of_steps < 0)
        {
          FOUR_C_THROW("Reduced lung number_of_steps must be non-negative, got {}.",
              dynamics.number_of_steps);
        }
        if (dynamics.results_every <= 0)
        {
          FOUR_C_THROW(
              "Reduced lung results_every must be positive, got {}.", dynamics.results_every);
        }
        if (dynamics.max_nonlinear_iterations <= 0)
        {
          FOUR_C_THROW("Reduced lung max_nonlinear_iterations must be positive, got {}.",
              dynamics.max_nonlinear_iterations);
        }
      }

      /**
       * Build the local reduced-lung discretization and visualization writer.
       */
      void build_discretization()
      {
        lung_mesh_ = build_discretization_from_mesh(
            *actdis_, context_.parameters.geometry, context_.rebalance_parameters);
        element_types_ = create_element_types(lung_mesh_.blocks, context_.parameters.lung_tree);

        // Must happen before fill_complete(), which triggers the redistribution of the fields.
        resolve_mesh_data_fields(*actdis_, lung_mesh_.mesh);
        // The input fields have copied out what they need, so rank 0 can drop the mesh. Only the
        // small replicated parts of lung_mesh_ are used from here on.
        lung_mesh_.mesh = {};

        actdis_->fill_complete();

        visualization_writer_ = std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(
            actdis_, Core::IO::visualization_parameters_factory(
                         context_.io_parameters.sublist("RUNTIME VTK OUTPUT"),
                         *context_.output_control_file, 0));
        comm_ = actdis_->get_comm();
      }

      /**
       * Instantiate local airway and terminal-unit model blocks and their evaluators.
       */
      void build_element_models()
      {
        create_local_element_models(*actdis_, context_.parameters, element_types_, airways_,
            terminal_units_, dof_per_ele_, n_airways_, n_terminal_units_);

        create_global_dof_maps(dof_per_ele_, comm_, global_dof_per_ele_, first_global_dof_of_ele_);
        assign_global_dof_ids_to_models(first_global_dof_of_ele_, airways_, terminal_units_);

        TerminalUnits::create_evaluators(terminal_units_);
        Airways::create_evaluators(airways_);
      }

      /**
       * Create node-attached boundary conditions and junction equations.
       */
      void build_node_entities()
      {
        global_ele_ids_per_node_ = create_global_ele_ids_per_node(*actdis_, comm_);

        BoundaryConditions::create_boundary_conditions(*actdis_, context_.parameters,
            lung_mesh_.bc_nodes, global_ele_ids_per_node_, global_dof_per_ele_,
            first_global_dof_of_ele_, context_.function_manager, boundary_conditions_);
        BoundaryConditions::create_evaluators(boundary_conditions_);

        Junctions::create_junctions(*actdis_, global_ele_ids_per_node_, global_dof_per_ele_,
            first_global_dof_of_ele_, connections_, bifurcations_);

        const int n_connections = static_cast<int>(connections_.size());
        const int n_bifurcations = static_cast<int>(bifurcations_.size());
        const int n_boundary_conditions =
            BoundaryConditions::count_boundary_conditions(boundary_conditions_);
        print_instantiated_object_counts(comm_, n_airways_, n_terminal_units_, n_connections,
            n_bifurcations, n_boundary_conditions);
      }

      /**
       * Assign local equation ids to every residual-contributing reduced-lung object.
       */
      void assign_equation_ids()
      {
        int n_local_equations = 0;
        Airways::assign_local_equation_ids(airways_, n_local_equations);
        TerminalUnits::assign_local_equation_ids(terminal_units_, n_local_equations);
        Junctions::assign_junction_local_equation_ids(
            connections_, bifurcations_, n_local_equations);
        BoundaryConditions::assign_local_equation_ids(boundary_conditions_, n_local_equations);
      }

      /**
       * Build owned, row, and locally relevant maps and translate global ids to local ids.
       */
      void build_maps_and_local_ids()
      {
        locally_owned_dof_map_ = std::make_unique<Core::LinAlg::Map>(
            create_domain_map(comm_, airways_, terminal_units_));
        row_map_ = std::make_unique<Core::LinAlg::Map>(create_row_map(
            comm_, airways_, terminal_units_, connections_, bifurcations_, boundary_conditions_));
        locally_relevant_dof_map_ = std::make_unique<Core::LinAlg::Map>(
            create_column_map(comm_, airways_, terminal_units_, global_dof_per_ele_,
                first_global_dof_of_ele_, connections_, bifurcations_, boundary_conditions_));

        Junctions::assign_junction_global_equation_ids(*row_map_, connections_, bifurcations_);
        BoundaryConditions::assign_global_equation_ids(*row_map_, boundary_conditions_);

        Airways::assign_local_dof_ids(*locally_relevant_dof_map_, airways_);
        TerminalUnits::assign_local_dof_ids(*locally_relevant_dof_map_, terminal_units_);
        Junctions::assign_junction_local_dof_ids(
            *locally_relevant_dof_map_, connections_, bifurcations_);
        BoundaryConditions::assign_local_dof_ids(*locally_relevant_dof_map_, boundary_conditions_);
      }

      /**
       * Allocate shared solver vectors and instantiate the selected nonlinear solver path.
       */
      void build_linear_system_and_solver()
      {
        FOUR_C_ASSERT_ALWAYS(locally_owned_dof_map_ != nullptr && row_map_ != nullptr &&
                                 locally_relevant_dof_map_ != nullptr,
            "Reduced lung maps must be initialized before linear system setup.");

        dofs_ = std::make_unique<Core::LinAlg::Vector<double>>(*locally_owned_dof_map_, true);
        locally_relevant_dofs_ =
            std::make_unique<Core::LinAlg::Vector<double>>(*locally_relevant_dof_map_, true);
        x_ = std::make_unique<Core::LinAlg::Vector<double>>(*row_map_, true);

        assembly_pipeline_ = create_default_reduced_lung_assembly_pipeline(airways_,
            terminal_units_, connections_, bifurcations_, boundary_conditions_,
            context_.parameters.dynamics.nonlinear_solver !=
                ReducedLungParameters::NonlinearSolverType::NewtonTree);

        switch (context_.parameters.dynamics.nonlinear_solver)
        {
          case ReducedLungParameters::NonlinearSolverType::Nox:
            build_nox_solver();
            break;
          case ReducedLungParameters::NonlinearSolverType::NewtonSparse:
            build_newton_solver_with_sparse_linear_solver();
            break;
          case ReducedLungParameters::NonlinearSolverType::NewtonTree:
            build_newton_solver_with_tree_linear_solver();
            break;
          default:
            FOUR_C_THROW("Unknown reduced-lung nonlinear solver workflow.");
        }
      }

      /**
       * Build the legacy NOX-based nonlinear solver workflow.
       */
      void build_nox_solver()
      {
        FOUR_C_ASSERT_ALWAYS(dofs_ != nullptr && locally_relevant_dofs_ != nullptr &&
                                 x_ != nullptr && row_map_ != nullptr &&
                                 locally_relevant_dof_map_ != nullptr,
            "Reduced lung linear system must be initialized before NOX solver setup.");
        sysmat_ =
            std::make_unique<Core::LinAlg::SparseMatrix>(*row_map_, *locally_relevant_dof_map_, 4);

        const NoxSolverContext nox_solver_context{
            .comm = comm_,
            .dynamics = context_.parameters.dynamics,
            .linear_solver_parameters = context_.linear_solver_parameters,
            .solver_params_callback = context_.solver_params_callback,
            .assembly_pipeline = assembly_pipeline_,
            .dofs = *dofs_,
            .locally_relevant_dofs = *locally_relevant_dofs_,
            .x = *x_,
            .jacobian = *sysmat_,
        };

        nox_solver_ = std::make_unique<NoxSolver>(nox_solver_context, current_time_);
      }

      /**
       * Build the custom Newton workflow with the generic sparse linear solver backend.
       */
      void build_newton_solver_with_sparse_linear_solver()
      {
        FOUR_C_ASSERT_ALWAYS(row_map_ != nullptr && locally_relevant_dof_map_ != nullptr,
            "Reduced lung maps must be initialized before sparse Newton solver setup.");
        sysmat_ =
            std::make_unique<Core::LinAlg::SparseMatrix>(*row_map_, *locally_relevant_dof_map_, 4);
        newton_linear_solver_ =
            std::make_shared<SparseNewtonLinearSolver>(SparseNewtonLinearSolverContext{
                .comm = comm_,
                .linear_solver_parameters = context_.linear_solver_parameters,
                .solver_params_callback = context_.solver_params_callback,
                .correction_map = *row_map_,
                .jacobian = *sysmat_,
            });
        build_newton_solver();
      }

      /**
       * Build the custom Newton workflow with the serial structured-tree linear solver backend.
       */
      void build_newton_solver_with_tree_linear_solver()
      {
        int comm_size = 1;
        MPI_Comm_size(comm_, &comm_size);
        FOUR_C_ASSERT_ALWAYS(row_map_ != nullptr && locally_relevant_dof_map_ != nullptr,
            "Reduced lung maps must be initialized before tree Newton solver setup.");
        if (comm_size != 1)
        {
          FOUR_C_THROW(
              "Reduced lung NewtonTree is serial-only, but the run uses {} MPI ranks. Run "
              "NewtonTree with one MPI rank, or select nonlinear_solver: NewtonSparse or Nox "
              "for MPI runs.",
              comm_size);
        }
        FOUR_C_ASSERT_ALWAYS(sysmat_ == nullptr,
            "Reduced lung NewtonTree setup must not allocate a sparse Jacobian.");

        {
          TEUCHOS_FUNC_TIME_MONITOR("ReducedLung::NewtonTree:  0)   Setup");
          tree_metadata_ = build_reduced_lung_tree_metadata(ReducedLungTreeMetadataContext{
              .discretization = *actdis_,
              .element_types = element_types_,
              .first_global_dof_of_ele = first_global_dof_of_ele_,
              .global_dof_per_ele = global_dof_per_ele_,
              .airways = airways_,
              .terminal_units = terminal_units_,
              .connections = connections_,
              .bifurcations = bifurcations_,
              .boundary_conditions = boundary_conditions_,
              .row_map = *row_map_,
              .locally_relevant_dof_map = *locally_relevant_dof_map_,
          });
          newton_linear_solver_ = std::make_shared<TreeNewtonLinearSolver>(
              TreeNewtonLinearSolverContext{.tree_metadata = *tree_metadata_,
                  .coefficient_source =
                      TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks,
                  .profile = tree_profile_enabled_ ? &tree_profile_ : nullptr});
        }
        build_newton_solver();
      }

      /**
       * Build the common custom Newton driver around the selected Newton linear solver backend.
       */
      void build_newton_solver()
      {
        FOUR_C_ASSERT_ALWAYS(dofs_ != nullptr && locally_relevant_dofs_ != nullptr && x_ != nullptr,
            "Reduced lung linear system must be initialized before custom Newton solver setup.");
        FOUR_C_ASSERT_ALWAYS(newton_linear_solver_ != nullptr,
            "Reduced lung custom Newton solver requires a Newton linear solver.");

        const NewtonSolverContext newton_solver_context{
            .dynamics = context_.parameters.dynamics,
            .linear_solver = newton_linear_solver_,
            .assembly_pipeline = assembly_pipeline_,
            .dofs = *dofs_,
            .locally_relevant_dofs = *locally_relevant_dofs_,
            .x = *x_,
            .profile = tree_profile_enabled_ ? &newton_profile_ : nullptr,
        };

        newton_solver_ = std::make_unique<NewtonSolver>(newton_solver_context, current_time_);
      }

      /**
       * Advance one timestep and print compact convergence information for custom Newton runs.
       */
      void solve_timestep(int step)
      {
        FOUR_C_ASSERT_ALWAYS(nox_solver_ != nullptr || newton_solver_ != nullptr,
            "Reduced lung solver must be initialized before time integration.");
        FOUR_C_ASSERT_ALWAYS(locally_relevant_dofs_ != nullptr,
            "Reduced lung locally relevant dof vector must be initialized before time "
            "integration.");

        current_time_ += dt_;
        // Constant during the solve, so refresh once per timestep rather than per iteration.
        BoundaryConditions::refresh_total_terminal_unit_volume(
            boundary_conditions_, terminal_units_, comm_);
        if (nox_solver_ != nullptr)
        {
          if (Core::Communication::my_mpi_rank(comm_) == 0)
          {
            std::cout << "Timestep: " << step << "/" << n_timesteps_
                      << "\n----------------------------------------\n"
                      << std::flush;
          }
          nox_solver_->solve(current_time_);
        }
        else
        {
          const unsigned int iterations = newton_solver_->solve(current_time_);
          if (Core::Communication::my_mpi_rank(comm_) == 0)
          {
            std::ostringstream residual_norm;
            residual_norm << std::scientific << std::setprecision(2)
                          << newton_solver_->last_residual_norm();
            std::cout << "Timestep " << step << "/" << n_timesteps_
                      << " | Newton iters: " << iterations << " | ||F||: " << residual_norm.str()
                      << "\n"
                      << std::flush;
          }
        }

        TerminalUnits::end_of_timestep_routine(terminal_units_, *locally_relevant_dofs_, dt_);
        Airways::end_of_timestep_routine(airways_, *locally_relevant_dofs_, dt_);
      }

      /**
       * Write visualization output when the configured output interval is reached.
       */
      void write_output_if_due(int step)
      {
        if (step % context_.parameters.dynamics.results_every != 0)
        {
          return;
        }

        FOUR_C_ASSERT_ALWAYS(visualization_writer_ != nullptr,
            "Reduced lung visualization writer is not initialized.");
        FOUR_C_ASSERT_ALWAYS(locally_relevant_dofs_ != nullptr,
            "Reduced lung locally relevant dof vector must be initialized before output.");

        visualization_writer_->reset();
        collect_runtime_output_data(*visualization_writer_, airways_, terminal_units_,
            *locally_relevant_dofs_, actdis_->element_row_map(),
            context_.parameters.dynamics.output_verbosity);
        visualization_writer_->write_to_disk(current_time_, step);
      }

      /**
       * Print optional NewtonTree profiling counters collected during the run.
       */
      void print_tree_profile_summary() const
      {
        if (!tree_profile_enabled_ || Core::Communication::my_mpi_rank(comm_) != 0)
        {
          return;
        }

        const auto average = [](double total, unsigned int count)
        { return count > 0 ? total / static_cast<double>(count) : 0.0; };
        const auto average_uint64 = [](double total, std::uint64_t count)
        { return count > 0 ? total / static_cast<double>(count) : 0.0; };

        std::cout
            << "\n-------- Reduced Lung Tree Profile --------\n"
            << "newton_solves: " << newton_profile_.solve_count << '\n'
            << "tree_linear_solves: " << tree_profile_.solve_count << '\n'
            << "newton_total_s: " << newton_profile_.total_solve_time << '\n'
            << "newton_total_s_avg: "
            << average(newton_profile_.total_solve_time, newton_profile_.solve_count) << '\n'
            << "state_sync_s: " << newton_profile_.state_sync_time << '\n'
            << "residual_s: " << newton_profile_.residual_assembly_time << '\n'
            << "residual_clear_s: " << newton_profile_.residual_clear_time << '\n'
            << "residual_airways_s: " << newton_profile_.residual_airway_time << '\n'
            << "residual_terminal_units_s: " << newton_profile_.residual_terminal_unit_time << '\n'
            << "residual_junctions_s: " << newton_profile_.residual_junction_time << '\n'
            << "residual_boundary_conditions_s: "
            << newton_profile_.residual_boundary_condition_time << '\n'
            << "residual_other_s: " << newton_profile_.residual_other_time << '\n'
            << "residual_norm_s: " << newton_profile_.residual_norm_time << '\n'
            << "residual_evaluations: " << newton_profile_.residual_evaluation_count << '\n'
            << "sparse_assembly_s: " << newton_profile_.sparse_jacobian_assembly_time << '\n'
            << "sparse_complete_s: " << newton_profile_.sparse_jacobian_complete_time << '\n'
            << "tree_assembly_s: " << newton_profile_.structured_tree_linearization_assembly_time
            << '\n'
            // Keep this legacy output key so historical benchmark profiles remain comparable.
            << "tree_assembly_clear_s: " << newton_profile_.tree_linearization_preparation_time
            << '\n'
            << "tree_assembly_airways_s: " << newton_profile_.tree_linearization_airway_time << '\n'
            << "tree_assembly_terminal_units_s: "
            << newton_profile_.tree_linearization_terminal_unit_time << '\n'
            << "tree_assembly_junctions_s: " << newton_profile_.tree_linearization_junction_time
            << '\n'
            << "tree_assembly_boundary_conditions_s: "
            << newton_profile_.tree_linearization_boundary_condition_time << '\n'
            << "tree_assembly_other_s: " << newton_profile_.tree_linearization_other_time << '\n'
            << "tree_assembly_solver_update_s: "
            << newton_profile_.tree_linearization_solver_update_time << '\n'
            << "linear_solve_s: " << newton_profile_.linear_solve_time << '\n'
            << "tree_solve_s: " << tree_profile_.total_solve_time << '\n'
            << "tree_solve_s_avg: "
            << average(tree_profile_.total_solve_time, tree_profile_.solve_count) << '\n'
            << "tree_bottom_up_s: " << tree_profile_.bottom_up_time << '\n'
            << "tree_top_down_s: " << tree_profile_.top_down_time << '\n'
            << "tree_dense_s: " << tree_profile_.dense_solve_time << '\n'
            << "tree_dense_s_avg: "
            << average_uint64(tree_profile_.dense_solve_time, tree_profile_.dense_solve_count)
            << '\n'
            << "tree_lookup_s: " << tree_profile_.coefficient_lookup_time << '\n'
            << "tree_dense_solves: " << tree_profile_.dense_solve_count << '\n'
            << "tree_lookups: " << tree_profile_.coefficient_lookup_count << '\n'
            << "tree_simd_groups: " << tree_profile_.simd_group_count << '\n'
            << "tree_simd_lanes: " << tree_profile_.simd_lane_count << '\n'
            << "tree_scalar_groups: " << tree_profile_.scalar_group_count << '\n'
            << "tree_scalar_tail_lanes: " << tree_profile_.scalar_tail_lane_count << '\n'
            << "tree_dense_fallbacks: " << tree_profile_.dense_fallback_count << '\n'
            << "tree_unsupported_fallbacks: " << tree_profile_.unsupported_block_fallback_count
            << '\n'
            << "tree_elements: " << tree_profile_.element_count << '\n'
            << "tree_workspace_dofs: " << tree_profile_.total_local_block_dofs << '\n'
            << "tree_max_block: " << tree_profile_.max_local_block_size << '\n'
            << "-------------------------------------------\n"
            << std::flush;
      }

      /**
       * Evaluate input-file result assertions against the converged final state.
       */
      void run_result_tests() const
      {
        FOUR_C_ASSERT_ALWAYS(locally_relevant_dofs_ != nullptr,
            "Reduced lung result tests require the final locally relevant dof vector.");
        context_.add_field_test(
            std::make_shared<ResultTest>(airways_, terminal_units_, *locally_relevant_dofs_));
        context_.test_all(comm_);
      }

      const ReducedLungContext context_;
      std::shared_ptr<Core::FE::Discretization> actdis_;
      std::unique_ptr<Core::IO::DiscretizationVisualizationWriterMesh> visualization_writer_;
      MPI_Comm comm_;

      Airways::AirwayContainer airways_;
      LungMesh lung_mesh_;
      std::vector<ReducedLungParameters::LungTree::ElementType> element_types_;
      TerminalUnits::TerminalUnitContainer terminal_units_;
      std::map<int, int> dof_per_ele_;
      int n_airways_ = 0;
      int n_terminal_units_ = 0;
      std::map<int, int> first_global_dof_of_ele_;
      std::map<int, int> global_dof_per_ele_;
      std::map<int, std::vector<int>> global_ele_ids_per_node_;
      BoundaryConditions::BoundaryConditionContainer boundary_conditions_;
      Junctions::ConnectionData connections_;
      Junctions::BifurcationData bifurcations_;

      std::unique_ptr<Core::LinAlg::Map> locally_owned_dof_map_;
      std::unique_ptr<Core::LinAlg::Map> row_map_;
      std::unique_ptr<Core::LinAlg::Map> locally_relevant_dof_map_;
      std::unique_ptr<Core::LinAlg::Vector<double>> dofs_;
      std::unique_ptr<Core::LinAlg::Vector<double>> locally_relevant_dofs_;
      std::unique_ptr<Core::LinAlg::Vector<double>> x_;
      std::unique_ptr<Core::LinAlg::SparseMatrix> sysmat_;
      ReducedLungAssemblyPipeline assembly_pipeline_;

      std::unique_ptr<NoxSolver> nox_solver_;
      std::unique_ptr<NewtonSolver> newton_solver_;
      std::shared_ptr<NewtonLinearSolver> newton_linear_solver_;
      std::optional<ReducedLungTreeMetadata> tree_metadata_;
      const double dt_;
      const int n_timesteps_;
      const bool tree_profile_enabled_;
      NewtonSolverProfile newton_profile_;
      TreeNewtonLinearSolverProfile tree_profile_;
      double current_time_ = 0.0;
    };

  }  // namespace

  /**
   * Run the reduced-lung simulation for an explicit 4C problem instance.
   */
  void reduced_lung_main(Global::Problem& problem)
  {
    const ReducedLungContext context = make_reduced_lung_context_from_problem(problem);
    ReducedLungSimulation simulation(context);
    simulation.initialize();
    simulation.run();
  }

  /**
   * Run the reduced-lung simulation for the global 4C problem instance.
   */
  void reduced_lung_main() { reduced_lung_main(*Global::Problem::instance()); }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
