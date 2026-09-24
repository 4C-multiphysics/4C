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
#include "4C_reduced_lung_resulttest.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_reduced_lung_tree_linear_solver.hpp"
#include "4C_reduced_lung_tree_metadata.hpp"
#include "4C_reduced_lung_tree_newton_solver.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

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
     * Return a printable name for the selected reduced-lung nonlinear solver workflow.
     */
    const char* nonlinear_solver_name(ReducedLungParameters::NonlinearSolverType solver)
    {
      using SolverType = ReducedLungParameters::NonlinearSolverType;
      switch (solver)
      {
        case SolverType::Nox:
          return "NOX";
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
            n_timesteps_(context.parameters.dynamics.number_of_steps)
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

        assembly_pipeline_ = create_default_nonlinear_solver_assembly_pipeline(airways_,
            terminal_units_, connections_, bifurcations_, boundary_conditions_,
            context_.parameters.dynamics.nonlinear_solver ==
                ReducedLungParameters::NonlinearSolverType::Nox);

        switch (context_.parameters.dynamics.nonlinear_solver)
        {
          case ReducedLungParameters::NonlinearSolverType::Nox:
            build_nox_solver();
            break;
          case ReducedLungParameters::NonlinearSolverType::NewtonTree:
            build_tree_newton_solver_with_tree_linear_solver();
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
       * Build the NewtonTree workflow with the serial structured-tree linear solver backend.
       */
      void build_tree_newton_solver_with_tree_linear_solver()
      {
        const int comm_size = Core::Communication::num_mpi_ranks(comm_);
        FOUR_C_ASSERT_ALWAYS(row_map_ != nullptr && locally_relevant_dof_map_ != nullptr,
            "Reduced lung maps must be initialized before tree Newton solver setup.");
        if (comm_size != 1)
        {
          FOUR_C_THROW(
              "Reduced lung NewtonTree is serial-only, but the run uses {} MPI ranks. Run "
              "NewtonTree with one MPI rank, or select nonlinear_solver: Nox for MPI runs.",
              comm_size);
        }
        FOUR_C_ASSERT_ALWAYS(sysmat_ == nullptr,
            "Reduced lung NewtonTree setup must not allocate a sparse Jacobian.");

        tree_metadata_ = build_reduced_lung_tree_metadata(ReducedLungTreeMetadataContext{
            .discretization = *actdis_,
            .element_types = element_types_,
            .airways = airways_,
            .terminal_units = terminal_units_,
            .connections = connections_,
            .bifurcations = bifurcations_,
            .boundary_conditions = boundary_conditions_,
            .row_map = *row_map_,
            .locally_relevant_dof_map = *locally_relevant_dof_map_,
        });
        tree_newton_linear_solver_ = std::make_shared<TreeNewtonLinearSolver>(
            TreeNewtonLinearSolverContext{.tree_metadata = *tree_metadata_});
        build_tree_newton_solver();
      }

      /**
       * Build the NewtonTree driver around the tree Newton linear solver backend.
       */
      void build_tree_newton_solver()
      {
        FOUR_C_ASSERT_ALWAYS(dofs_ != nullptr && locally_relevant_dofs_ != nullptr && x_ != nullptr,
            "Reduced lung linear system must be initialized before NewtonTree solver setup.");
        FOUR_C_ASSERT_ALWAYS(tree_newton_linear_solver_ != nullptr,
            "Reduced lung NewtonTree solver requires a tree Newton linear solver.");

        const TreeNewtonSolverContext tree_newton_solver_context{
            .dynamics = context_.parameters.dynamics,
            .tree_linear_solver = tree_newton_linear_solver_,
            .assembly_pipeline = assembly_pipeline_,
            .dofs = *dofs_,
            .locally_relevant_dofs = *locally_relevant_dofs_,
            .x = *x_,
        };

        tree_newton_solver_ =
            std::make_unique<TreeNewtonSolver>(tree_newton_solver_context, current_time_);
      }

      /**
       * Advance one timestep and print compact convergence information for NewtonTree runs.
       */
      void solve_timestep(int step)
      {
        FOUR_C_ASSERT_ALWAYS(nox_solver_ != nullptr || tree_newton_solver_ != nullptr,
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
          const unsigned int iterations = tree_newton_solver_->solve(current_time_);
          if (Core::Communication::my_mpi_rank(comm_) == 0)
          {
            std::ostringstream residual_norm;
            residual_norm << std::scientific << std::setprecision(2)
                          << tree_newton_solver_->last_residual_norm();
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
      NonlinearSolverAssemblyPipeline assembly_pipeline_;

      std::unique_ptr<NoxSolver> nox_solver_;
      std::unique_ptr<TreeNewtonSolver> tree_newton_solver_;
      std::shared_ptr<TreeNewtonLinearSolver> tree_newton_linear_solver_;
      std::optional<ReducedLungTreeMetadata> tree_metadata_;
      const double dt_;
      const int n_timesteps_;
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
