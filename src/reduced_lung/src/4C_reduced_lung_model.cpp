// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_model.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_boundary_conditions.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_junctions.hpp"
#include "4C_reduced_lung_newton_sparse.hpp"
#include "4C_reduced_lung_newton_tree.hpp"
#include "4C_reduced_lung_nonlinear_solver.hpp"
#include "4C_reduced_lung_resulttest.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_reduced_lung_tree_linear_solver.hpp"
#include "4C_reduced_lung_tree_metadata.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>
#include <memory>
#include <optional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  /**
   * Owns setup, state, and solver of one reduced-lung model.
   */
  struct Model::Impl
  {
    explicit Impl(const ModelContext& context)
        : context_(context),
          actdis_(std::make_shared<Core::FE::Discretization>("reduced_lung", context.comm, 3)),
          comm_(context.comm)
    {
      if (context_.parameters.dynamics.max_nonlinear_iterations <= 0)
      {
        FOUR_C_THROW("Reduced lung max_nonlinear_iterations must be positive, got {}.",
            context_.parameters.dynamics.max_nonlinear_iterations);
      }

      build_discretization();
      build_element_models();
      build_node_entities();
      assign_equation_ids();
      build_maps_and_local_ids();
      build_linear_system_and_solver();
    }

    unsigned int solve_step(double dt)
    {
      FOUR_C_ASSERT_ALWAYS(dt > 0.0, "Reduced lung step size must be positive, got {}.", dt);
      FOUR_C_ASSERT_ALWAYS(nonlinear_solver_ != nullptr,
          "Reduced lung solver must be initialized before time integration.");

      // Evaluated from the accepted terminal-unit volumes, so repeated solves of the same step
      // see the same value.
      BoundaryConditions::refresh_total_terminal_unit_volume(
          boundary_conditions_, terminal_units_, comm_);

      const unsigned int iterations = nonlinear_solver_->solve(accepted_time_ + dt, dt);

      trial_dt_ = dt;
      return iterations;
    }

    /**
     * Advance the airway and terminal-unit history with the last trial solution.
     */
    void accept_step()
    {
      FOUR_C_ASSERT_ALWAYS(trial_dt_.has_value(),
          "Reduced lung accept_step() requires a solve_step() since the last accepted step.");

      TerminalUnits::end_of_timestep_routine(terminal_units_, *locally_relevant_dofs_, *trial_dt_);
      Airways::end_of_timestep_routine(airways_, *locally_relevant_dofs_, *trial_dt_);

      accepted_time_ += *trial_dt_;
      trial_dt_.reset();
    }

    void write_output(int step) const
    {
      FOUR_C_ASSERT_ALWAYS(visualization_writer_ != nullptr,
          "Reduced lung visualization writer is not initialized.");
      FOUR_C_ASSERT_ALWAYS(!trial_dt_.has_value(),
          "Reduced lung write_output() requires the trial solution to be accepted first.");

      visualization_writer_->reset();
      collect_runtime_output_data(*visualization_writer_, airways_, terminal_units_,
          *locally_relevant_dofs_, actdis_->element_row_map(),
          context_.parameters.dynamics.output_verbosity);
      visualization_writer_->write_to_disk(accepted_time_, step);
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
      Junctions::assign_junction_local_equation_ids(connections_, bifurcations_, n_local_equations);
      BoundaryConditions::assign_local_equation_ids(boundary_conditions_, n_local_equations);
    }

    /**
     * Build owned, row, and locally relevant maps and translate global ids to local ids.
     */
    void build_maps_and_local_ids()
    {
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
      FOUR_C_ASSERT_ALWAYS(row_map_ != nullptr && locally_relevant_dof_map_ != nullptr,
          "Reduced lung maps must be initialized before linear system setup.");

      locally_relevant_dofs_ =
          std::make_unique<Core::LinAlg::Vector<double>>(*locally_relevant_dof_map_, true);
      x_ = std::make_unique<Core::LinAlg::Vector<double>>(*row_map_, true);

      assembly_pipeline_ = create_default_nonlinear_solver_assembly_pipeline(airways_,
          terminal_units_, connections_, bifurcations_, boundary_conditions_,
          context_.parameters.dynamics.nonlinear_solver ==
              ReducedLungParameters::NonlinearSolverType::NewtonSparse);

      switch (context_.parameters.dynamics.nonlinear_solver)
      {
        case ReducedLungParameters::NonlinearSolverType::NewtonSparse:
          build_newton_sparse_solver();
          break;
        case ReducedLungParameters::NonlinearSolverType::NewtonTree:
          build_newton_tree_solver();
          break;
        default:
          FOUR_C_THROW("Unknown reduced-lung nonlinear solver workflow.");
      }
    }

    /**
     * Build the NewtonSparse driver with an assembled sparse Jacobian.
     */
    void build_newton_sparse_solver()
    {
      FOUR_C_ASSERT_ALWAYS(locally_relevant_dofs_ != nullptr && x_ != nullptr &&
                               row_map_ != nullptr && locally_relevant_dof_map_ != nullptr,
          "Reduced lung linear system must be initialized before NOX solver setup.");
      sysmat_ =
          std::make_unique<Core::LinAlg::SparseMatrix>(*row_map_, *locally_relevant_dof_map_, 4);

      const NewtonSparseContext newton_sparse_context{
          .comm = comm_,
          .dynamics = context_.parameters.dynamics,
          .linear_solver_parameters = context_.linear_solver_parameters,
          .solver_params_callback = context_.solver_params_callback,
          .assembly_pipeline = assembly_pipeline_,
          .locally_relevant_dofs = *locally_relevant_dofs_,
          .x = *x_,
          .jacobian = *sysmat_,
      };

      nonlinear_solver_ = std::make_unique<NewtonSparse>(newton_sparse_context);
    }

    /**
     * Build the NewtonTree driver with the serial structured-tree linear solver.
     */
    void build_newton_tree_solver()
    {
      const int comm_size = Core::Communication::num_mpi_ranks(comm_);
      FOUR_C_ASSERT_ALWAYS(row_map_ != nullptr && locally_relevant_dof_map_ != nullptr,
          "Reduced lung maps must be initialized before NewtonTree solver setup.");
      FOUR_C_ASSERT_ALWAYS(locally_relevant_dofs_ != nullptr && x_ != nullptr,
          "Reduced lung linear system must be initialized before NewtonTree solver setup.");
      if (comm_size != 1)
      {
        FOUR_C_THROW(
            "Reduced lung NewtonTree is serial-only, but the run uses {} MPI ranks. Run "
            "NewtonTree with one MPI rank, or select nonlinear_solver: NewtonSparse for MPI "
            "runs.",
            comm_size);
      }
      FOUR_C_ASSERT_ALWAYS(
          sysmat_ == nullptr, "Reduced lung NewtonTree setup must not allocate a sparse Jacobian.");

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
      tree_linear_solver_ = std::make_shared<TreeLinearSolver>(
          TreeLinearSolverContext{.tree_metadata = *tree_metadata_});

      const NewtonTreeContext newton_tree_context{
          .comm = comm_,
          .dynamics = context_.parameters.dynamics,
          .tree_linear_solver = tree_linear_solver_,
          .assembly_pipeline = assembly_pipeline_,
          .locally_relevant_dofs = *locally_relevant_dofs_,
          .x = *x_,
      };
      nonlinear_solver_ = std::make_unique<NewtonTree>(newton_tree_context);
    }

    const ModelContext context_;
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

    std::unique_ptr<Core::LinAlg::Map> row_map_;
    std::unique_ptr<Core::LinAlg::Map> locally_relevant_dof_map_;
    std::unique_ptr<Core::LinAlg::Vector<double>> locally_relevant_dofs_;
    std::unique_ptr<Core::LinAlg::Vector<double>> x_;
    std::unique_ptr<Core::LinAlg::SparseMatrix> sysmat_;
    NonlinearSolverAssemblyPipeline assembly_pipeline_;

    std::unique_ptr<NonlinearSolver> nonlinear_solver_;
    std::shared_ptr<TreeLinearSolver> tree_linear_solver_;
    std::optional<ReducedLungTreeMetadata> tree_metadata_;

    double accepted_time_ = 0.0;
    //! Step size of the last solve_step(), reset by accept_step().
    std::optional<double> trial_dt_;
  };

  Model::Model(const ModelContext& context) : pimpl_(std::make_unique<Impl>(context)) {}

  Model::~Model() = default;

  unsigned int Model::solve_step(double dt) { return pimpl_->solve_step(dt); }

  void Model::accept_step() { pimpl_->accept_step(); }

  double Model::time() const { return pimpl_->accepted_time_; }

  void Model::write_output(int step) const { pimpl_->write_output(step); }

  std::shared_ptr<Core::Utils::ResultTest> Model::create_result_test() const
  {
    return std::make_shared<ResultTest>(
        pimpl_->airways_, pimpl_->terminal_units_, *pimpl_->locally_relevant_dofs_);
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
