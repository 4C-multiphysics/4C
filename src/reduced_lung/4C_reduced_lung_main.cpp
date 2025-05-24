// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_main.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_maxwell_0d_acinus_NeoHookean.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  void reduced_lung_main()
  {
    // Access given 4C infrastructure.
    std::shared_ptr<Core::FE::Discretization> actdis =
        Global::Problem::instance()->get_dis("red_airway");
    if (!actdis->filled())
    {
      actdis->fill_complete();
    }
    const Teuchos::ParameterList& rawdyn =
        Global::Problem::instance()->reduced_d_airway_dynamic_params();
    const int linear_solver_number = rawdyn.get<int>("LINEAR_SOLVER");
    std::unique_ptr<Core::LinAlg::Solver> solver = std::make_unique<Core::LinAlg::Solver>(
        Global::Problem::instance()->solver_params(linear_solver_number), actdis->get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));
    actdis->compute_null_space_if_necessary(solver->params());
    // Create runtime output writer
    const auto visualization_writer =
        std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(
            actdis, Core::IO::visualization_parameters_factory(
                        Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                        *Global::Problem::instance()->output_control_file(), 0));
    // The existing mpi communicator is recycled for the new data layout.
    const auto& comm = actdis->get_comm();

    // Create vectors of local entities (equations acting on the dofs).
    // Physical "elements" of the lung tree introducing the dofs.
    std::vector<Airway> airways;
    std::vector<TerminalUnit> terminal_units;
    std::map<int, int> dof_per_ele;  // Map global element id -> dof.
    int n_airways = 0;
    int n_terminal_units = 0;
    // Loop over all elements in actdis and create the new element data layout (for airways and
    // terminal units). Adds all information directly given in the row element range.
    for (auto* ele : actdis->my_row_element_range())
    {
      int element_id = ele->id();
      int local_element_id = actdis->element_row_map()->lid(element_id);
      if (ele->element_type() == Discret::Elements::RedAirwayType::instance())
      {
        // Number of equations still missing! Needs to be changed when compliant airways are
        // implemented.
        airways.push_back(Airway{element_id, local_element_id, n_airways, AirwayType::resistive});
        dof_per_ele[element_id] = 3;
        n_airways++;
      }
      else if (ele->element_type() == Discret::Elements::RedAcinusType::instance())
      {
        double E;
        double eta;
        if (ele->material(0)->material_type() == Core::Materials::m_0d_maxwell_acinus_neohookean)
        {
          auto* param = ele->material(0)->parameter();
          auto* mat = static_cast<Mat::PAR::Maxwell0dAcinus*>(param);
          E = mat->stiffness1_;
          eta = mat->viscosity1_;
        }
        else
        {
          FOUR_C_THROW("Material not implemented.");
        }
        terminal_units.push_back(TerminalUnit{element_id, local_element_id, n_terminal_units,
            TerminalUnitType::kelvin_voigt, E, eta});
        dof_per_ele[element_id] = 3;
        n_terminal_units++;
      }
      else
      {
        FOUR_C_THROW("Unknown element type.");
      }
    }

    /* Create global dof numbering (done on every processor simultaneously)
       Logic: global dof ids are created from element global ids by expanding them with their
       associated dofs. Example: 3 resistive airway elements:
           ele ids         dof ids
                             {0       |\
             {0               1       |   dofs associated with element 0
                              2       |/
                              3       |\
              1       ->      4       |   dofs associated with element 1
                              5       |/
                              6       |\
              2}              7       |   dofs associated with element 2
                              8}      |/
    */
    auto global_dof_per_ele = Core::Communication::all_reduce(dof_per_ele, comm);
    std::map<int, int> first_global_dof_of_ele;
    int acc = 0;
    for (auto ele_dof : global_dof_per_ele)
    {
      first_global_dof_of_ele[ele_dof.first] = acc;
      acc += ele_dof.second;
    }
    // Assign every local element its associated global dof ids.
    for (auto& airway : airways)
    {
      int first_dof_gid = first_global_dof_of_ele[airway.global_element_id];
      int n_dof = dof_per_ele[airway.global_element_id];
      for (int i = 0; i < n_dof; i++)
      {
        airway.global_dof_ids.insert(airway.global_dof_ids.end(), first_dof_gid + i);
      }
    }
    for (auto& terminal_unit : terminal_units)
    {
      int first_dof_gid = first_global_dof_of_ele[terminal_unit.global_element_id];
      int n_dof = dof_per_ele[terminal_unit.global_element_id];
      for (int i = 0; i < n_dof; i++)
      {
        terminal_unit.global_dof_ids.insert(terminal_unit.global_dof_ids.end(), first_dof_gid + i);
      }
    }

    // Build local map node id -> adjacent element id and distribute to all processors.
    std::map<int, std::vector<int>> ele_ids_per_node;
    for (const auto* node : actdis->my_row_node_range())
    {
      for (int i = 0; i < node->num_element(); i++)
      {
        ele_ids_per_node[node->id()].push_back(node->elements()[i]->id());
      }
    }
    auto merge_maps =
        [](const std::map<int, std::vector<int>>& map1, const std::map<int, std::vector<int>>& map2)
    {
      std::map<int, std::vector<int>> result = map1;
      for (const auto& [key, values] : map2)
      {
        result[key].insert(result[key].end(), values.begin(), values.end());
      }
      return result;
    };
    auto global_ele_ids_per_node = Core::Communication::all_reduce<std::map<int, std::vector<int>>>(
        ele_ids_per_node, merge_maps, comm);

    // Create entities with equations connecting elements (acting on "nodes" of the lung tree).
    std::vector<BoundaryCondition> boundary_conditions;
    std::vector<Connection> connections;
    std::vector<Bifurcation> bifurcations;
    int n_boundary_conditions = 0;
    int n_connections = 0;
    int n_bifurcations = 0;
    // Loop over all local elements, get their nodes, create associated entity. This way, they are
    // created on the same ranks as at least one of their connected elements. This reduces the
    // amount of communication of dofs between ranks.

    std::vector<const Core::Conditions::Condition*> conditions;
    actdis->get_condition("RedAirwayPrescribedCond", conditions);
    const auto red_airway_prescribed_conditions =
        Core::Conditions::find_conditioned_node_ids_and_conditions(
            *actdis, conditions, Core::Conditions::LookFor::locally_owned_and_ghosted);

    for (const auto* ele : actdis->my_row_element_range())
    {
      const auto* nodes = ele->nodes();
      // WARNING: if node ordering is wrong (inlet in nodes[1]), the whole logic doesn't apply in
      // the same way as assumed throughout this implementation! A top-down ordering of nodes needs
      // to be enforced during tree creation. This is to a large amount asserted during creation of
      // the coupling entities. However, special cases might slip through.
      const auto& node_in = *nodes[0];
      const auto& node_out = *nodes[1];
      const auto node_in_n_eles = global_ele_ids_per_node[node_in.id()].size();
      const auto node_out_n_eles = global_ele_ids_per_node[node_out.id()].size();
      // Check whether element is starting point of tree (trachea in full lungs, lobe inlets for
      // single lobes, etc.)
      if (node_in_n_eles == 1)
      {
        FOUR_C_ASSERT_ALWAYS(red_airway_prescribed_conditions.count(node_in.id()) == 1,
            "Node {} is located at the boundary and needs to have exactly one boundary condition "
            "but it has {} conditions.",
            node_in.id(), red_airway_prescribed_conditions.count(node_in.id()));

        const auto* bc_condition = red_airway_prescribed_conditions.find(node_in.id())->second;
        const std::string bc_type = bc_condition->parameters().get<std::string>("boundarycond");
        const std::optional<int> funct_num =
            bc_condition->parameters().get<std::vector<std::optional<int>>>("curve")[0];
        if (bc_type == "pressure")
        {
          if (funct_num.has_value())
          {
            boundary_conditions.push_back(BoundaryCondition{ele->id(), 0, 0, n_boundary_conditions,
                BoundaryConditionType::pressure_in, first_global_dof_of_ele[ele->id()],
                funct_num.value()});
            n_boundary_conditions++;
          }
        }
        else
        {
          FOUR_C_THROW("Boundary condition not implemented!");
        }
      }
      // Create entities "top down"-like (a processor owning an element also owns its outlet node
      // entities).
      if (node_out_n_eles == 1)
      {
        FOUR_C_ASSERT_ALWAYS(red_airway_prescribed_conditions.count(node_out.id()) == 1,
            "Node {} is located at the boundary and needs to have exactly one boundary condition "
            "but it has {} conditions.",
            node_out.id(), red_airway_prescribed_conditions.count(node_out.id()));

        const auto* bc_condition = red_airway_prescribed_conditions.find(node_out.id())->second;
        const std::string bc_type = bc_condition->parameters().get<std::string>("boundarycond");
        const std::optional<int> funct_num =
            bc_condition->parameters().get<std::vector<std::optional<int>>>("curve")[0];
        if (bc_type == "pressure")
        {
          if (funct_num.has_value())
          {
            // Pressure bc at outlet node, so p2 (2nd dof of ele).
            boundary_conditions.push_back(BoundaryCondition{ele->id(), 0, 0, n_boundary_conditions,
                BoundaryConditionType::pressure_out, first_global_dof_of_ele[ele->id()] + 1,
                funct_num.value()});
            n_boundary_conditions++;
          }
        }
        else
        {
          FOUR_C_ASSERT(false, "Boundary condition not implemented!");
        }
      }
      else if (node_out_n_eles == 2)
      {
        std::array<int, 2> global_ele_ids{
            global_ele_ids_per_node[node_out.id()][0], global_ele_ids_per_node[node_out.id()][1]};
        // Check that each element only owns one connection at the outlet and that the same
        // connection isn't instantiated a second time from the other element.
        for (const auto& conn : connections)
        {
          FOUR_C_ASSERT_ALWAYS((conn.global_parent_element_id != global_ele_ids[1] ||
                                   conn.global_child_element_id != global_ele_ids[0]),
              "Connection instantiated twice! Check node ordering in input file.");
          FOUR_C_ASSERT_ALWAYS(conn.global_parent_element_id != global_ele_ids[0],
              "Second connection entity at parent element! Check input file.");
        }
        for (const auto& bif : bifurcations)
        {
          FOUR_C_ASSERT_ALWAYS((((bif.global_parent_element_id != global_ele_ids[1]) ||
                                    (bif.global_child_1_element_id != global_ele_ids[0] &&
                                        bif.global_child_2_element_id != global_ele_ids[0])) &&
                                   (bif.global_parent_element_id != global_ele_ids[0])),
              "Bifurcation and connection instantiated at the same node! Check input file.");
        }
        // dofs: {p2_parent, p1_child, q2_parent (q for non-compliant airways), q1_child}
        std::array<int, 4> global_dof_ids{first_global_dof_of_ele[global_ele_ids[0]] + 1,
            first_global_dof_of_ele[global_ele_ids[1]],
            first_global_dof_of_ele[global_ele_ids[0]] + global_dof_per_ele[global_ele_ids[0]] - 1,
            first_global_dof_of_ele[global_ele_ids[1]] + 2};
        Connection conn{0, 0, n_connections, global_ele_ids[0], global_ele_ids[1], global_dof_ids};
        connections.push_back(conn);
        n_connections++;
      }
      else if (node_out_n_eles == 3)
      {
        std::array<int, 3> global_ele_ids{global_ele_ids_per_node[node_out.id()][0],
            global_ele_ids_per_node[node_out.id()][1], global_ele_ids_per_node[node_out.id()][2]};
        // Check that each element only owns one bifurcation at the outlet and that the same
        // bifurcation isn't instantiated a second time from another element.
        for (const auto& bif : bifurcations)
        {
          FOUR_C_ASSERT_ALWAYS(((bif.global_parent_element_id != global_ele_ids[1] &&
                                    bif.global_parent_element_id != global_ele_ids[2]) ||
                                   (bif.global_child_1_element_id != global_ele_ids[0] &&
                                       bif.global_child_2_element_id != global_ele_ids[0])),
              "Bifurcation instantiated twice! Check node ordering in input file.");
          FOUR_C_ASSERT_ALWAYS(bif.global_parent_element_id != global_ele_ids[0],
              "Second connection entity at parent element! Check input file.");
        }
        for (auto conn : connections)
        {
          FOUR_C_ASSERT_ALWAYS(((conn.global_parent_element_id != global_ele_ids[1] ||
                                    conn.global_child_element_id != global_ele_ids[0]) &&
                                   conn.global_parent_element_id != global_ele_ids[0]),
              "Connection and bifurcation instantiated at the same node! Check input file.");
        }
        // dofs: {p2_parent, p1_child_1, p1_child_2, q2_parent (q for non-compliant airways),
        // q1_child_1, q1_child_2}
        std::array<int, 6> global_dof_ids{first_global_dof_of_ele[global_ele_ids[0]] + 1,
            first_global_dof_of_ele[global_ele_ids[1]], first_global_dof_of_ele[global_ele_ids[2]],
            first_global_dof_of_ele[global_ele_ids[0]] + global_dof_per_ele[global_ele_ids[0]] - 1,
            first_global_dof_of_ele[global_ele_ids[1]] + 2,
            first_global_dof_of_ele[global_ele_ids[2]] + 2};
        Bifurcation bif{0, 0, n_bifurcations, global_ele_ids[0], global_ele_ids[1],
            global_ele_ids[2], global_dof_ids};
        bifurcations.push_back(bif);
        n_bifurcations++;
      }
      else
      {
        FOUR_C_THROW("Too many elements at junction.");
      }
    }

    // Print info on instantiated objects.
    {
      int n_total_airways, n_total_terminal_units, n_total_connections, n_total_bifurcations,
          n_total_boundary_conditions;
      Core::Communication::sum_all(&n_airways, &n_total_airways, 1, comm);
      Core::Communication::sum_all(&n_terminal_units, &n_total_terminal_units, 1, comm);
      Core::Communication::sum_all(&n_connections, &n_total_connections, 1, comm);
      Core::Communication::sum_all(&n_bifurcations, &n_total_bifurcations, 1, comm);
      Core::Communication::sum_all(&n_boundary_conditions, &n_total_boundary_conditions, 1, comm);
      if (Core::Communication::my_mpi_rank(comm) == 0)
      {
        std::cout << "--------- Instantiated objects ---------"
                  << "\nAirways:              |  " << n_total_airways
                  << "\nTerminal Units:       |  " << n_total_terminal_units
                  << "\nConnections:          |  " << n_total_connections
                  << "\nBifurcations:         |  " << n_total_bifurcations
                  << "\nBoundary Conditions:  |  " << n_total_boundary_conditions << "\n\n"
                  << std::flush;
      }
    }

    // Calculate local and global number of "element" equations
    int n_local_equations = 0;
    for (const auto& airway : airways)
    {
      n_local_equations += airway.n_state_equations;
    }
    for (const auto& terminal_unit : terminal_units)
    {
      n_local_equations += terminal_unit.n_state_equations;
    }

    // Assign local equation ids to connections, bifurcations, and boundary conditions.
    for (Connection& conn : connections)
    {
      // Every connection adds 1 momentum  and 1 mass balance equation.
      conn.first_local_equation_id = n_local_equations;
      n_local_equations += 2;
    }
    for (Bifurcation& bif : bifurcations)
    {
      // Every bifurcation adds 2 momentum balance equations and 1 mass balance equation.
      bif.first_local_equation_id = n_local_equations;
      n_local_equations += 3;
    }
    for (BoundaryCondition& bc : boundary_conditions)
    {
      // Each boundaary condition adds 1 equation enforcing it at the respective dof.
      bc.local_equation_id = n_local_equations;
      n_local_equations++;
    }

    // Create all necessary maps for matrix, rhs, and dof-vector.
    // Map with all dof ids belonging to the local elements (airways and terminal units).
    const Core::LinAlg::Map locally_owned_dof_map =
        create_domain_map(comm, airways, terminal_units);
    // Map with row ids for the equations of local elements, connections, bifurcations, and boundary
    // conditions.
    const Core::LinAlg::Map row_map = create_row_map(
        comm, airways, terminal_units, connections, bifurcations, boundary_conditions);
    // Map with all relevant dof ids for the local equations.
    const Core::LinAlg::Map locally_relevant_dof_map =
        create_column_map(comm, airways, terminal_units, global_dof_per_ele,
            first_global_dof_of_ele, connections, bifurcations, boundary_conditions);

    // Assign global equation ids to connections, bifurcations, and boundary conditions based on the
    // row map. Maybe not necessary, but helps with debugging.
    for (Connection& conn : connections)
    {
      conn.first_global_equation_id = row_map.gid(conn.first_local_equation_id);
    }
    for (Bifurcation& bif : bifurcations)
    {
      bif.first_global_equation_id = row_map.gid(bif.first_local_equation_id);
    }
    for (BoundaryCondition& bc : boundary_conditions)
    {
      bc.global_equation_id = row_map.gid(bc.local_equation_id);
    }

    // Save locally relevant dof ids of every entity. Needed for local assembly.
    for (Airway& airway : airways)
    {
      for (const int& gid : airway.global_dof_ids)
      {
        airway.local_dof_ids.push_back(locally_relevant_dof_map.lid(gid));
      }
    }
    for (TerminalUnit& terminal_unit : terminal_units)
    {
      for (const int& gid : terminal_unit.global_dof_ids)
      {
        terminal_unit.local_dof_ids.push_back(locally_relevant_dof_map.lid(gid));
      }
    }
    for (Connection& conn : connections)
    {
      int i = 0;
      for (const int& gid : conn.global_dof_ids)
      {
        conn.local_dof_ids[i] = locally_relevant_dof_map.lid(gid);
        i++;
      }
    }
    for (Bifurcation& bif : bifurcations)
    {
      int i = 0;
      for (const int& gid : bif.global_dof_ids)
      {
        bif.local_dof_ids[i] = locally_relevant_dof_map.lid(gid);
        i++;
      }
    }
    for (BoundaryCondition& bc : boundary_conditions)
    {
      bc.local_dof_id = locally_relevant_dof_map.lid(bc.global_dof_id);
    }

    // Local airway vectors. Potentially add to airway objects in the future.
    std::vector<double> length(n_airways);
    std::vector<double> area(n_airways);
    std::vector<double> poiseuille_resistance(n_airways);

    // Airway constants.
    const double dynamic_viscosity_mu = 1.79195e-05;
    const double density_rho = 1.176e-06;

    // Fill airway data.
    for (const Airway& airway : airways)
    {
      const int airway_id = airway.local_airway_id;
      const auto* airway_ele =
          static_cast<Discret::Elements::RedAirway*>(actdis->g_element(airway.global_element_id));
      const Discret::ReducedLung::AirwayParams airway_params = airway_ele->get_airway_params();
      const std::vector<double>& coords_1 = airway_ele->nodes()[0]->x();
      const std::vector<double>& coords_2 = airway_ele->nodes()[1]->x();
      const double current_length =
          std::sqrt((coords_1[0] - coords_2[0]) * (coords_1[0] - coords_2[0]) +
                    (coords_1[1] - coords_2[1]) * (coords_1[1] - coords_2[1]) +
                    (coords_1[2] - coords_2[2]) * (coords_1[2] - coords_2[2]));
      length[airway_id] = current_length;
      area[airway_id] = airway_params.area;
      poiseuille_resistance[airway_id] = 8 * M_PI * dynamic_viscosity_mu * density_rho *
                                         current_length / (airway_params.area * airway_params.area);
    }

    // Local terminal unit vectors. Potentially add to terminal unit objects in the future.
    std::vector<double> volume(n_terminal_units);
    std::vector<double> volume_0(n_terminal_units);

    // Fill terminal unit data.
    for (const TerminalUnit& terminal_unit : terminal_units)
    {
      const int terminal_unit_id = terminal_unit.local_terminal_unit_id;
      const auto* terminal_unit_ele = static_cast<Discret::Elements::RedAcinus*>(
          actdis->g_element(terminal_unit.global_element_id));
      const std::vector<double>& coords_1 = terminal_unit_ele->nodes()[0]->x();
      const std::vector<double>& coords_2 = terminal_unit_ele->nodes()[1]->x();
      const double radius = std::sqrt((coords_1[0] - coords_2[0]) * (coords_1[0] - coords_2[0]) +
                                      (coords_1[1] - coords_2[1]) * (coords_1[1] - coords_2[1]) +
                                      (coords_1[2] - coords_2[2]) * (coords_1[2] - coords_2[2]));
      volume[terminal_unit_id] = volume_0[terminal_unit_id] = radius * radius * M_PI;
    }

    // Create system matrix and vectors:
    // Vector with all degrees of freedom (p1, p2, q, ...) associated to the elements.
    auto dofs = Core::LinAlg::Vector<double>(locally_owned_dof_map, true);
    // Vector with all degrees of freedom (p1, p2, q, ...) at the last timestep.
    auto dofs_n = Core::LinAlg::Vector<double>(locally_owned_dof_map, true);
    // Vector with locally relevant degrees of freedom, needs to import data from dofs vector.
    auto locally_relevant_dofs = Core::LinAlg::Vector<double>(locally_relevant_dof_map, true);
    // Solution vector of the system of equations with increments of all dofs calculated per
    // iteration.
    auto x = Core::LinAlg::Vector<double>(row_map, true);
    // Exported solution that can be directly added to dofs.
    auto x_mapped_to_dofs = Core::LinAlg::Vector<double>(locally_owned_dof_map, true);
    // Right hand side vector with residuals of the system equations.
    auto rhs = Core::LinAlg::Vector<double>(row_map, true);
    // Jacobian of the system equations.
    auto sysmat = Core::LinAlg::SparseMatrix(row_map, locally_relevant_dof_map, 3);

    const int results_every = rawdyn.get<int>("RESULTSEVERY");
    // Time integration parameters.
    const double dt = rawdyn.get<double>("TIMESTEP");
    const int n_timesteps = rawdyn.get<int>("NUMSTEP");
    // Time loop
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      std::cout << "-------- Start Time Integration --------\n"
                << "----------------------------------------\n"
                << std::flush;
    }
    for (int n = 1; n <= n_timesteps; n++)
    {
      if (Core::Communication::my_mpi_rank(comm) == 0)
      {
        std::cout << "Timestep: " << n << "/" << n_timesteps
                  << "\n----------------------------------------\n"
                  << std::flush;
      }
      dofs_n.update(1.0, dofs, 0.0);
      [[maybe_unused]] int err;  // Saves error code of trilinos functions.

      // Assemble system of equations.
      // Assemble airway equations in system matrix and rhs.
      const auto evaluate_aw_jacobian = [&poiseuille_resistance](
                                            std::array<double, 3>& values, const int& local_ele_id)
      { values = {1.0, -1.0, -poiseuille_resistance[local_ele_id]}; };
      // Momentum balance: p_in - p_out - R*q != 0.
      const auto evaluate_aw_rhs = [&poiseuille_resistance](double& res, const double& p_in,
                                       const double& p_out, const double& q,
                                       const int& local_ele_id)
      { res = -(p_in - p_out - poiseuille_resistance[local_ele_id] * q); };
      for (const Airway& airway : airways)
      {
        std::array<double, 3> vals;
        double res;
        evaluate_aw_jacobian(vals, airway.local_airway_id);
        evaluate_aw_rhs(res, locally_relevant_dofs[airway.local_dof_ids[p_in]],
            locally_relevant_dofs[airway.local_dof_ids[p_out]],
            locally_relevant_dofs[airway.local_dof_ids[q_in]], airway.local_airway_id);
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(
              airway.local_element_id, vals.size(), vals.data(), airway.local_dof_ids.data());
        }
        else
        {
          err = sysmat.replace_my_values(
              airway.local_element_id, vals.size(), vals.data(), airway.local_dof_ids.data());
        }
        FOUR_C_ASSERT(err == 0, "Internal error: Airway equation assembly did not work.");
        err = rhs.replace_local_value(airway.local_element_id, 0, res);
        FOUR_C_ASSERT(err == 0, "Internal error: Airway equation calculation did not work.");
      }

      // Assemble terminal unit equations.
      for (const TerminalUnit& terminal_unit : terminal_units)
      {
        // Momentum balance: p_in - p_pl - E*(V-V0)/V0 - nu*q/V0 = 0.
        const double& V_tu = volume[terminal_unit.local_terminal_unit_id];
        const double& V0_tu = volume_0[terminal_unit.local_terminal_unit_id];
        const double& E = terminal_unit.E;
        const double& eta = terminal_unit.eta;
        const std::array<double, 3> vals{1.0, -1.0, -(E * dt + eta) / V0_tu};
        const double res =
            -locally_relevant_dofs[terminal_unit.local_dof_ids[p_in]] +
            locally_relevant_dofs[terminal_unit.local_dof_ids[p_out]] +
            (E * dt + eta) / V0_tu * locally_relevant_dofs[terminal_unit.local_dof_ids[q_in]] +
            E * (V_tu - V0_tu) / V0_tu;
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(terminal_unit.local_element_id, vals.size(), vals.data(),
              terminal_unit.local_dof_ids.data());
        }
        else
        {
          err = sysmat.replace_my_values(terminal_unit.local_element_id, vals.size(), vals.data(),
              terminal_unit.local_dof_ids.data());
        }
        FOUR_C_ASSERT(err == 0, "Internal error: Terminal Unit equation assembly did not work.");
        err = rhs.replace_local_value(terminal_unit.local_element_id, 0, res);
        FOUR_C_ASSERT(err == 0, "Internal error: Terminal Unit equation calculation did not work.");
      }

      // Assemble connection equations.
      for (const Connection& conn : connections)
      {
        std::array<double, 2> vals;
        double res;
        // Momentum balance (p2_parent - p1_child = 0).
        vals = {1.0, -1.0};
        std::array<int, 2> local_dof_ids{conn.local_dof_ids[Connection::p_out_parent],
            conn.local_dof_ids[Connection::p_in_child]};
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(
              conn.first_local_equation_id, vals.size(), vals.data(), local_dof_ids.data());
        }
        else
        {
          err = sysmat.replace_my_values(
              conn.first_local_equation_id, vals.size(), vals.data(), local_dof_ids.data());
        }
        FOUR_C_ASSERT(
            err == 0, "Internal error: Connection momentum balance assembly did not work.");
        res = -locally_relevant_dofs[local_dof_ids[0]] + locally_relevant_dofs[local_dof_ids[1]];
        err = rhs.replace_local_value(conn.first_local_equation_id, 0, res);
        FOUR_C_ASSERT(
            err == 0, "Internal error: Connection momentum balance calculation did not work.");

        // Mass balance (q_out_parent - q_in_child = 0).
        vals = {1.0, -1.0};
        local_dof_ids = {conn.local_dof_ids[Connection::q_out_parent],
            conn.local_dof_ids[Connection::q_in_child]};
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(
              conn.first_local_equation_id + 1, vals.size(), vals.data(), local_dof_ids.data());
        }
        else
        {
          err = sysmat.replace_my_values(
              conn.first_local_equation_id + 1, vals.size(), vals.data(), local_dof_ids.data());
        }
        FOUR_C_ASSERT(err == 0, "Internal error: Connection mass balance assembly did not work.");
        res = -locally_relevant_dofs[local_dof_ids[0]] + locally_relevant_dofs[local_dof_ids[1]];
        err = rhs.replace_local_value(conn.first_local_equation_id + 1, 0, res);
        FOUR_C_ASSERT(
            err == 0, "Internal error: Connection mass balance calculation did not work.");
      }

      // Assemble bifurcation equations
      for (const Bifurcation& bif : bifurcations)
      {
        std::array<double, 2> vals_mom_balance;
        std::array<double, 3> vals_mass_balance;
        double res;
        // Momentum balance parent - child_1 (p2_parent - p1_child_1 = 0).
        vals_mom_balance = {1.0, -1.0};
        std::array<int, 2> local_dof_ids_mom_balance{bif.local_dof_ids[Bifurcation::p_out_parent],
            bif.local_dof_ids[Bifurcation::p_in_child_1]};
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(bif.first_local_equation_id, vals_mom_balance.size(),
              vals_mom_balance.data(), local_dof_ids_mom_balance.data());
        }
        else
        {
          err = sysmat.replace_my_values(bif.first_local_equation_id, vals_mom_balance.size(),
              vals_mom_balance.data(), local_dof_ids_mom_balance.data());
        }
        FOUR_C_ASSERT(
            err == 0, "Internal error: Bifurcation momentum balance assembly did not work.");
        res = -locally_relevant_dofs[local_dof_ids_mom_balance[0]] +
              locally_relevant_dofs[local_dof_ids_mom_balance[1]];
        err = rhs.replace_local_value(bif.first_local_equation_id, 0, res);
        FOUR_C_ASSERT(
            err == 0, "Internal error: Bifurcation momentum balance calculation did not work.");

        // Momentum balance parent - child_2 (p2_parent - p1_child_2 = 0).
        vals_mom_balance = {1.0, -1.0};
        local_dof_ids_mom_balance = {bif.local_dof_ids[Bifurcation::p_out_parent],
            bif.local_dof_ids[Bifurcation::p_in_child_2]};
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(bif.first_local_equation_id + 1, vals_mom_balance.size(),
              vals_mom_balance.data(), local_dof_ids_mom_balance.data());
        }
        else
        {
          err = sysmat.replace_my_values(bif.first_local_equation_id + 1, vals_mom_balance.size(),
              vals_mom_balance.data(), local_dof_ids_mom_balance.data());
        }
        FOUR_C_ASSERT(
            err == 0, "Internal error: Bifurcation momentum balance assembly did not work.");
        res = -locally_relevant_dofs[local_dof_ids_mom_balance[0]] +
              locally_relevant_dofs[local_dof_ids_mom_balance[1]];
        err = rhs.replace_local_value(bif.first_local_equation_id + 1, 0, res);
        FOUR_C_ASSERT(
            err == 0, "Internal error: Bifurcation momentum balance calculation did not work.");

        // Mass balance (q_out_parent - q_in_child_1 - q_in_child_1 = 0).
        vals_mass_balance = {1.0, -1.0, -1.0};
        std::array<int, 3> local_dof_ids_mass_balance = {
            bif.local_dof_ids[Bifurcation::q_out_parent],
            bif.local_dof_ids[Bifurcation::q_in_child_1],
            bif.local_dof_ids[Bifurcation::q_in_child_2]};
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(bif.first_local_equation_id + 2, vals_mass_balance.size(),
              vals_mass_balance.data(), local_dof_ids_mass_balance.data());
        }
        else
        {
          err = sysmat.replace_my_values(bif.first_local_equation_id + 2, vals_mass_balance.size(),
              vals_mass_balance.data(), local_dof_ids_mass_balance.data());
        }
        FOUR_C_ASSERT(err == 0, "Internal error: Bifurcation mass balance assembly did not work.");
        res = -locally_relevant_dofs[local_dof_ids_mass_balance[0]] +
              locally_relevant_dofs[local_dof_ids_mass_balance[1]] +
              locally_relevant_dofs[local_dof_ids_mass_balance[2]];
        err = rhs.replace_local_value(bif.first_local_equation_id + 2, 0, res);
        FOUR_C_ASSERT(
            err == 0, "Internal error: Bifurcation momentum balance calculation did not work.");
      }

      // Assemble boundary conditions (equation: dof_value - bc_value = 0).
      for (const BoundaryCondition& bc : boundary_conditions)
      {
        const double val = 1.0;
        double res;
        auto bc_value = Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfTime>(bc.funct_num)
                            .evaluate(n * dt);
        int local_dof_id = bc.local_dof_id;
        if (!sysmat.filled())
        {
          err = sysmat.insert_my_values(bc.local_equation_id, 1, &val, &local_dof_id);
        }
        else
        {
          err = sysmat.replace_my_values(bc.local_equation_id, 1, &val, &local_dof_id);
        }
        FOUR_C_ASSERT(err == 0, "Internal error: Boundary condition assembly did not work.");
        res = -locally_relevant_dofs[local_dof_id] + bc_value;
        err = rhs.replace_local_value(bc.local_equation_id, 0, res);
        FOUR_C_ASSERT(err == 0, "Internal error: Boundary condition evaluation did not work.");
      }

      // Fix sparsity pattern after the first assembly process.
      if (!sysmat.filled())
      {
        sysmat.complete();
      }

      // Solve.
      solver->solve(sysmat.epetra_operator(), Core::Utils::shared_ptr_from_ref(x),
          Core::Utils::shared_ptr_from_ref(rhs), {});

      // Update dofs with solution vector.
      export_to(x, x_mapped_to_dofs);
      dofs.update(1.0, x_mapped_to_dofs, 1.0);
      export_to(dofs, locally_relevant_dofs);

      // Update variable parameters depending on dofs.
      for (const auto& terminal_unit : terminal_units)
      {
        const int& q_id = terminal_unit.local_dof_ids[2];
        const int& tu_id = terminal_unit.local_terminal_unit_id;
        // Backwards Euler: V_n+1 = V_n + q_n+1 * dt.
        volume[tu_id] += locally_relevant_dofs[q_id] * dt;
      }

      // Runtime output
      if (n % results_every == 0)
      {
        visualization_writer->reset();
        collect_runtime_output_data(*visualization_writer, airways, terminal_units,
            locally_relevant_dofs, actdis->element_row_map());
        visualization_writer->write_to_disk(dt * n, n);
      }
    }
    // Print time monitor
    Teuchos::TimeMonitor::summarize(std::cout, false, true, false);
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
