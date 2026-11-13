// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_constraint_framework_submodelevaluator_mpc.hpp"

#include "4C_beam3_base.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_constraint_framework_equation.hpp"
#include "4C_constraint_framework_input.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometric_search_bounding_volume.hpp"
#include "4C_geometric_search_distributed_tree.hpp"
#include "4C_geometric_search_input.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_structure_new_timint_implicit.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  //! Print node ids in right-aligned columns, wrapped to about the width of the separator line
  void print_node_ids_debug(const std::vector<int>& node_ids)
  {
    const std::string prefix = "  Node IDs: ";
    constexpr std::size_t line_width = 70;

    std::size_t id_width = 1;
    for (const int node_id : node_ids)
      id_width = std::max(id_width, std::to_string(node_id).size());
    const std::size_t ids_per_line =
        std::max<std::size_t>(1, (line_width - prefix.size() + 2) / (id_width + 2));

    for (std::size_t i = 0; i < node_ids.size(); ++i)
    {
      if (i % ids_per_line == 0)
        Core::IO::cout(Core::IO::debug) << (i == 0 ? prefix : std::string(prefix.size(), ' '));

      const std::string id = std::to_string(node_ids[i]);
      Core::IO::cout(Core::IO::debug) << std::string(id_width - id.size(), ' ') << id;

      const bool is_last = (i + 1 == node_ids.size());
      const bool ends_line = ((i + 1) % ids_per_line == 0);
      if (!is_last) Core::IO::cout(Core::IO::debug) << (ends_line ? "," : ", ");
      if (is_last || ends_line) Core::IO::cout(Core::IO::debug) << Core::IO::endl;
    }
  }
}  // namespace

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::RveMultiPointConstraintManager(
    std::shared_ptr<const Core::FE::Discretization> disc_ptr, Core::LinAlg::SparseMatrix* st_ptr)
{
  discret_ptr_ = disc_ptr;
  stiff_ptr_ = st_ptr;

  check_input();

  // Maps of relevant boundary nodesets of the rve
  std::map<std::string, const std::vector<int>*> rveBoundaryNodeIdMap;

  // Map of the corner node ids
  std::map<std::string, int> rveCornerNodeIdMap;

  //  Map the Node IDs to the respective rve boundary --> rveBoundaryNodeIdMap
  build_periodic_rve_boundary_node_map(rveBoundaryNodeIdMap);

  // Map the Node ids to the respective corner of the rve --> rveCornerNodeIdMap
  switch (rve_ref_type_)
  {
    case Constraints::MultiPoint::RveReferenceDeformationDefinition::automatic:
    {
      build_periodic_rve_corner_node_map(rveBoundaryNodeIdMap, rveCornerNodeIdMap);
    }
    break;

    case Constraints::MultiPoint::RveReferenceDeformationDefinition::manual:
    {
      if (rve_dim_ != Constraints::MultiPoint::RveDimension::rve2d)
        FOUR_C_THROW("Manual Edge node definition is not implemented for 3D RVEs");

      // Read the reference points
      for (const auto& entry : point_periodic_rve_ref_conditions_)
      {
        const auto& str_id = entry->parameters().get<std::string>("POSITION");
        const auto* nodeInSet = entry->get_nodes();

        if (nodeInSet->size() > 1)
        {
          FOUR_C_THROW("There can only be a single node defined as a reference node");
        }
        rve_ref_node_map_[str_id] = discret_ptr_->g_node(nodeInSet->data()[0]);
      }

      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Reference geometry" << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl
          << "Reference nodes: N1L = " << rve_ref_node_map_["N1L"]->id()
          << ", N1B = " << rve_ref_node_map_["N1B"]->id()
          << ", N2 = " << rve_ref_node_map_["N2"]->id()
          << ", N4 = " << rve_ref_node_map_["N4"]->id() << Core::IO::endl;

      // calculate the Reference vectors between Ref. points
      r_xmxp_[0] = rve_ref_node_map_["N2"]->x()[0] - rve_ref_node_map_["N1L"]->x()[0];
      r_xmxp_[1] = rve_ref_node_map_["N2"]->x()[1] - rve_ref_node_map_["N1L"]->x()[1];
      Core::IO::cout(Core::IO::verbose)
          << "Reference vector x: [" << r_xmxp_[0] << "; " << r_xmxp_[1] << "]" << Core::IO::endl;

      r_ymyp_[0] = rve_ref_node_map_["N4"]->x()[0] - rve_ref_node_map_["N1B"]->x()[0];
      r_ymyp_[1] = rve_ref_node_map_["N4"]->x()[1] - rve_ref_node_map_["N1B"]->x()[1];
      Core::IO::cout(Core::IO::verbose)
          << "Reference vector y: [" << r_ymyp_[0] << "; " << r_ymyp_[1] << "]" << Core::IO::endl;
    }
    break;
  }

  // Create a vector with all MPCs describing the periodic BCs
  if (surface_periodic_rve_conditions_.size() != 0 || line_periodic_rve_conditions_.size() != 0)
    build_periodic_mp_cs(rveBoundaryNodeIdMap, rveCornerNodeIdMap);


  // Add Linear Coupled Equation MPCs
  if (point_linear_coupled_equation_conditions_.size() != 0)
  {
    int nLinCe = build_linear_mp_cs();
    Core::IO::cout(Core::IO::verbose)
        << "Total number of linear coupled equations: " << nLinCe << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::check_input()
{
  const auto geometric_search_params = Core::GeometricSearch::geometric_search_params_factory(
      Global::Problem::instance()->parameters());
  auto constraint_parameter_list = Global::Problem::instance()->constraint_params();

  strategy_ = Teuchos::getIntegralValue<Constraints::EnforcementStrategy>(
      constraint_parameter_list, "CONSTRAINT_ENFORCEMENT");

  auto mpc_parameter_list = constraint_parameter_list.sublist("MULTI POINT");

  rve_ref_type_ =
      Teuchos::getIntegralValue<Constraints::MultiPoint::RveReferenceDeformationDefinition>(
          mpc_parameter_list, "RVE_REFERENCE_POINTS");

  node_search_toler_ = geometric_search_params.point_tolerance;

  // Check the enforcement strategy
  switch (strategy_)
  {
    case Constraints::EnforcementStrategy::lagrange:
    {
      FOUR_C_THROW("Constraint Enforcement via Lagrange Multiplier Method is not impl.");
      break;
    }
    case Constraints::EnforcementStrategy::penalty:
    {
      get_penalty_parameter_ptr() = constraint_parameter_list.get<double>("PENALTY_PARAM");
      break;
    }
  }

  // Conditions definition
  discret_ptr_->get_condition("LinePeriodicRve", line_periodic_rve_conditions_);
  discret_ptr_->get_condition("SurfacePeriodicRve", surface_periodic_rve_conditions_);
  discret_ptr_->get_condition("PointPeriodicRveReferenceNode", point_periodic_rve_ref_conditions_);
  discret_ptr_->get_condition(
      "PointLinearCoupledEquation", point_linear_coupled_equation_conditions_);

  const bool is_parallel = Core::Communication::num_mpi_ranks(discret_ptr_->get_comm()) > 1;
  FOUR_C_ASSERT_ALWAYS(
      !is_parallel ||
          rve_ref_type_ != Constraints::MultiPoint::RveReferenceDeformationDefinition::manual,
      "Manual RVE reference points are not implemented in parallel.");
  FOUR_C_ASSERT_ALWAYS(!is_parallel || point_linear_coupled_equation_conditions_.empty(),
      "PointLinearCoupledEquation constraints are not implemented in parallel.");

  // Input Checks: Dimensions
  if (line_periodic_rve_conditions_.size() == 0 && surface_periodic_rve_conditions_.size() != 0)
  {
    rve_dim_ = Constraints::MultiPoint::RveDimension::rve3d;
  }
  else if (line_periodic_rve_conditions_.size() != 0 &&
           surface_periodic_rve_conditions_.size() == 0)
  {
    rve_dim_ = Constraints::MultiPoint::RveDimension::rve2d;
  }
  else
  {
    FOUR_C_THROW("Periodic rve edge condition cannot be combined with peridodic rve surf cond. ");
  }

  // Input Checks
  if (line_periodic_rve_conditions_.size() != 0)
  {
    if (line_periodic_rve_conditions_.size() != 4 && line_periodic_rve_conditions_.size() != 2)
    {
      FOUR_C_THROW("For a 2D RVE either all or two opposing edges must be used for PBCs");
    }
  }

  if (point_periodic_rve_ref_conditions_.size() == 0 &&
      rve_ref_type_ == Constraints::MultiPoint::RveReferenceDeformationDefinition::manual)
  {
    FOUR_C_THROW(
        "A DESIGN POINT PERIODIC RVE 2D BOUNDARY REFERENCE CONDITIONS is req. for manual ref. "
        "point "
        "definition");
  }

  if (point_periodic_rve_ref_conditions_.size() != 0 &&
      rve_ref_type_ == Constraints::MultiPoint::RveReferenceDeformationDefinition::automatic)
    FOUR_C_THROW("Set the RVE_REFERENCE_POINTS to manual");

  Core::IO::cout(Core::IO::minimal)
      << Core::IO::endl
      << "Periodic Boundary Conditions" << Core::IO::endl
      << "+--------------------------------------------------------------------+" << Core::IO::endl
      << "RVE dimension: "
      << (rve_dim_ == Constraints::MultiPoint::RveDimension::rve2d ? "2D" : "3D") << Core::IO::endl
      << "Constraint enforcement: penalty method" << Core::IO::endl;

  Core::IO::cout(Core::IO::standard)
      << "Penalty parameter: " << get_penalty_parameter_ptr() << Core::IO::endl
      << "Reference-point definition: "
      << (rve_ref_type_ == Constraints::MultiPoint::RveReferenceDeformationDefinition::automatic
                 ? "automatic"
                 : "manual")
      << Core::IO::endl;

  if (rve_dim_ == Constraints::MultiPoint::RveDimension::rve2d)
  {
    Core::IO::cout(Core::IO::standard)
        << "Periodic boundaries: " << line_periodic_rve_conditions_.size() << " edges"
        << Core::IO::endl;
  }
  else
  {
    Core::IO::cout(Core::IO::standard)
        << "Periodic boundaries: " << surface_periodic_rve_conditions_.size() << " surfaces"
        << Core::IO::endl;
  }

  Core::IO::cout(Core::IO::verbose)
      << "Geometric search tolerance: " << node_search_toler_ << Core::IO::endl;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::find_opposite_edge_node(
    const int nodeID, Constraints::MultiPoint::RveEdgeIdentifiers edge,
    std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap)
{
  std::string newPos;
  std::array<double, 2> R_ipim;

  Core::Nodes::Node* nodeA = discret_ptr_->g_node(nodeID);

  switch (edge)
  {
    case Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_xm:
    {
      R_ipim = r_xmxp_;
      newPos = "x+";
      break;
    }

    case Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_ym:
    {
      R_ipim = r_ymyp_;
      newPos = "y+";
      break;
    }
    default:
    {
      FOUR_C_THROW("Specify the negative edge, 3D not implemented");
    }
  }

  // Calculate the Position of the opposing edge node
  std::vector<double> newPosition = {0.0, 0.0};
  for (int i = 0; i < 2; i++) newPosition[i] = nodeA->x()[i] + R_ipim[i];

  // Loop all nodes of the relevant opposite edge line
  // ToDo: Switch to ArborX
  for (auto pairId : *rveBoundaryNodeIdMap[newPos])
  {
    Core::Nodes::Node* nodeB = discret_ptr_->g_node(pairId);

    if (std::abs(nodeB->x()[0] - newPosition[0]) < node_search_toler_)
    {
      if (std::abs(nodeB->x()[1] - newPosition[1]) < node_search_toler_)
      {
        return pairId;
      }
    }
  }
  FOUR_C_THROW("No matching periodic node found. Is the mesh periodic?");
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::build_periodic_mp_cs(
    std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap,
    std::map<std::string, int>& rveCornerNodeIdMap)
{
  const MPI_Comm comm = discret_ptr_->get_comm();
  const Core::LinAlg::Map& node_row_map = *discret_ptr_->node_row_map();
  const int num_dim = (rve_dim_ == Constraints::MultiPoint::RveDimension::rve2d) ? 2 : 3;

  // filled by both reference-point definitions (automatic and manual):
  // the four nodes of every periodic relation u(n0) - u(n1) = u(n2) - u(n3) created on this rank,
  // a label per relation for the output, and the dofs of nodes owned by other ranks
  std::vector<std::array<int, 4>> periodic_relation_nodes;
  std::vector<std::string> pbc_labels;
  std::map<int, std::array<int, 3>> gathered_node_dofs;
  std::map<std::string, std::size_t> boundary_node_pair_counts;

  switch (rve_ref_type_)
  {
    case Constraints::MultiPoint::RveReferenceDeformationDefinition::automatic:
    {
      std::map<std::string, std::string> ref_end_node_map = {{"x", "N2"}, {"y", "N4"}, {"z", "N5"}};
      if (rve_dim_ == Constraints::MultiPoint::rve2d) ref_end_node_map.erase("z");

      // gather the corner node coordinates needed for the reference vectors
      std::set<int> corner_gids = {rveCornerNodeIdMap["N1"]};
      for (const auto& [axis, end_node] : ref_end_node_map)
        corner_gids.insert(rveCornerNodeIdMap[end_node]);
      for (const int corner_gid : corner_gids)
        FOUR_C_ASSERT_ALWAYS(corner_gid >= 0, "A corner node of the RVE could not be found.");

      std::map<int, std::array<double, 3>> local_corner_coordinates;
      for (const int corner_gid : corner_gids)
        if (node_row_map.my_gid(corner_gid))
        {
          const auto x = discret_ptr_->g_node(corner_gid)->x();
          std::array<double, 3> coordinates = {};
          for (int i = 0; i < num_dim; ++i) coordinates[i] = x[i];
          local_corner_coordinates[corner_gid] = coordinates;
        }
      const auto corner_coordinates =
          Core::Communication::all_reduce(local_corner_coordinates, comm);

      std::map<std::string, std::array<double, 3>> reference_vector;
      for (const auto& [axis, end_node] : ref_end_node_map)
      {
        std::array<double, 3> shift = {};
        for (int i = 0; i < num_dim; ++i)
          shift[i] = corner_coordinates.at(rveCornerNodeIdMap[end_node])[i] -
                     corner_coordinates.at(rveCornerNodeIdMap["N1"])[i];
        reference_vector[axis] = shift;

        Core::IO::cout(Core::IO::verbose)
            << "Reference vector " << axis << ": [" << shift[0] << "; " << shift[1];
        if (num_dim == 3) Core::IO::cout(Core::IO::verbose) << "; " << shift[2];
        Core::IO::cout(Core::IO::verbose) << "]" << Core::IO::endl;
      }

      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Periodic relation search" << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl;

      // match "-" nodes (shifted by the reference vector) with "+" nodes across all ranks;
      // the owner of the "-" node creates the relation. Corners that already got their
      // relation on an earlier axis are excluded, so no duplicate relation is created across ranks
      std::set<int> excluded_minus_gids = {rveCornerNodeIdMap["N1"]};
      for (const auto& [axis, end_node] : ref_end_node_map)
      {
        const int ref_end_gid = rveCornerNodeIdMap[end_node];
        const int ref_base_gid = rveCornerNodeIdMap["N1"];

        std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> shifted_negative_nodes;
        for (const int minus_gid : *rveBoundaryNodeIdMap.at(axis + "-"))
        {
          if (excluded_minus_gids.contains(minus_gid) || !node_row_map.my_gid(minus_gid)) continue;
          const auto x = discret_ptr_->g_node(minus_gid)->x();
          Core::LinAlg::Matrix<3, 1, double> target_position(Core::LinAlg::Initialization::zero);
          for (int i = 0; i < num_dim; ++i) target_position(i) = x[i] + reference_vector[axis][i];
          Core::GeometricSearch::BoundingVolume bounding_volume;
          bounding_volume.add_point(target_position);
          bounding_volume.extend_boundaries(node_search_toler_);
          shifted_negative_nodes.emplace_back(minus_gid, bounding_volume);
        }

        std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> positive_nodes;
        for (const int plus_gid : *rveBoundaryNodeIdMap.at(axis + "+"))
        {
          if (!node_row_map.my_gid(plus_gid)) continue;
          const auto x = discret_ptr_->g_node(plus_gid)->x();
          Core::LinAlg::Matrix<3, 1, double> actual_position(Core::LinAlg::Initialization::zero);
          for (int i = 0; i < num_dim; ++i) actual_position(i) = x[i];
          Core::GeometricSearch::BoundingVolume bounding_volume;
          bounding_volume.add_point(actual_position);
          bounding_volume.extend_boundaries(node_search_toler_);
          positive_nodes.emplace_back(plus_gid, bounding_volume);
        }

        const auto matches = Core::GeometricSearch::global_collision_search(
            positive_nodes, shifted_negative_nodes, comm);

        std::map<int, std::vector<int>> partners_of_minus_node;
        for (const auto& match : matches)
          partners_of_minus_node[match.gid_predicate].push_back(match.gid_primitive);

        // every rank checks its own "-" nodes; the failures are gathered so that all ranks throw
        std::vector<int> my_unmatched_gids;
        std::vector<int> my_ambiguous_gids;
        for (const auto& [minus_gid, bounding_volume] : shifted_negative_nodes)
        {
          const auto partners = partners_of_minus_node.find(minus_gid);
          if (partners == partners_of_minus_node.end())
            my_unmatched_gids.push_back(minus_gid);
          else if (partners->second.size() != 1)
            my_ambiguous_gids.push_back(minus_gid);
        }
        const auto unmatched_gids = Core::Communication::all_reduce(my_unmatched_gids, comm);
        const auto ambiguous_gids = Core::Communication::all_reduce(my_ambiguous_gids, comm);
        auto join_gids = [](const std::vector<int>& gids)
        {
          std::string joined;
          for (const int gid : gids) joined += (joined.empty() ? "" : " ") + std::to_string(gid);
          return joined;
        };
        FOUR_C_ASSERT_ALWAYS(unmatched_gids.empty() && ambiguous_gids.empty(),
            "Periodic search on the '{}-' boundary failed: no partner for node(s) [{}], more than "
            "one partner for node(s) [{}]. Check mesh periodicity or POINT_TOLERANCE.",
            axis, join_gids(unmatched_gids), join_gids(ambiguous_gids));

        Core::IO::cout(Core::IO::debug) << axis << "-boundary pairs:" << Core::IO::endl;

        std::size_t pair_number = 0;
        for (const auto& [minus_gid, partners] : partners_of_minus_node)
        {
          // u(+) - u(-) = u(ref_end) - u(N1)
          const std::string pair_label = axis + "-pair " + std::to_string(++pair_number);
          periodic_relation_nodes.push_back({partners[0], minus_gid, ref_end_gid, ref_base_gid});
          pbc_labels.push_back(pair_label);

          Core::IO::cout(Core::IO::debug) << "  " << pair_label << ": u(" << partners[0] << ") - u("
                                          << minus_gid << ") = u(" << end_node << ":" << ref_end_gid
                                          << ") - u(N1:" << ref_base_gid << ")" << Core::IO::endl;
        }
        boundary_node_pair_counts[axis] =
            Core::Communication::sum_all(static_cast<int>(pair_number), comm);
        Core::IO::cout(Core::IO::debug) << Core::IO::endl;

        excluded_minus_gids.insert(ref_end_gid);
      }

      // the owner of a "-" node needs the dofs of its "+" partner and of the corner nodes, which
      // may live on other ranks: collect the dofs of all owned "+" and corner nodes on every rank
      std::map<int, std::array<int, 3>> my_node_dofs;
      auto add_owned_node_dofs = [&](const int node_gid)
      {
        if (!node_row_map.my_gid(node_gid)) return;
        const std::vector<int> dofs = discret_ptr_->dof(discret_ptr_->g_node(node_gid));
        std::array<int, 3> dofs_of_node = {-1, -1, -1};
        for (int i = 0; i < num_dim; ++i) dofs_of_node[i] = dofs[i];
        my_node_dofs[node_gid] = dofs_of_node;
      };
      for (const int corner_gid : corner_gids) add_owned_node_dofs(corner_gid);
      for (const auto& [axis, end_node] : ref_end_node_map)
        for (const int plus_gid : *rveBoundaryNodeIdMap.at(axis + "+"))
          add_owned_node_dofs(plus_gid);
      gathered_node_dofs = Core::Communication::all_reduce(my_node_dofs, comm);
      break;
    }
    case Constraints::MultiPoint::RveReferenceDeformationDefinition::manual:
    {
      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Periodic relation search" << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl;

      /* Loop over X- Edge */
      Core::IO::cout(Core::IO::debug) << "x-boundary pairs:" << Core::IO::endl;

      std::size_t x_pair_number = 0;
      for (auto nodeXm : *rveBoundaryNodeIdMap["x-"])
      {
        if (nodeXm != rve_ref_node_map_["N1L"]->id())  // exclude N1 - N2 = N1 - N2
        {
          const std::array<int, 4> node_set = {nodeXm,
              find_opposite_edge_node(nodeXm, Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_xm,
                  rveBoundaryNodeIdMap),
              rve_ref_node_map_["N1L"]->id(), rve_ref_node_map_["N2"]->id()};

          const std::string pair_label = "x-pair " + std::to_string(++x_pair_number);
          periodic_relation_nodes.push_back(node_set);
          pbc_labels.push_back(pair_label);

          Core::IO::cout(Core::IO::debug)
              << "  " << pair_label << ": u(" << node_set[0] << ") - u(" << node_set[1]
              << ") = u(N1L:" << node_set[2] << ") - u(N2:" << node_set[3] << ")" << Core::IO::endl;
        }
      }
      boundary_node_pair_counts["x"] = x_pair_number;
      Core::IO::cout(Core::IO::debug) << Core::IO::endl;

      /* Loop over Y- Edge*/
      Core::IO::cout(Core::IO::debug) << "y-boundary pairs:" << Core::IO::endl;

      std::size_t y_pair_number = 0;
      for (auto nodeYm : *rveBoundaryNodeIdMap["y-"])
      {
        if (nodeYm != rve_ref_node_map_["N1B"]->id() && nodeYm != rve_ref_node_map_["N2"]->id())
        {
          const std::array<int, 4> node_set = {
              find_opposite_edge_node(nodeYm, Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_ym,
                  rveBoundaryNodeIdMap),
              nodeYm, rve_ref_node_map_["N4"]->id(), rve_ref_node_map_["N1B"]->id()};

          const std::string pair_label = "y-pair " + std::to_string(++y_pair_number);
          periodic_relation_nodes.push_back(node_set);
          pbc_labels.push_back(pair_label);

          Core::IO::cout(Core::IO::debug)
              << "  " << pair_label << ": u(" << node_set[0] << ") - u(" << node_set[1]
              << ") = u(N4:" << node_set[2] << ") - u(N1B:" << node_set[3] << ")" << Core::IO::endl;
        }
      }
      boundary_node_pair_counts["y"] = y_pair_number;
      Core::IO::cout(Core::IO::debug) << Core::IO::endl;
      break;
    }
    default:
      FOUR_C_THROW("No ref def type defined");
  }
  if (!boundary_node_pair_counts.empty())
  {
    Core::IO::cout(Core::IO::standard) << "Candidate boundary-node pairs: ";
    bool is_first_entry = true;
    for (const auto& [direction, count] : boundary_node_pair_counts)
    {
      if (!is_first_entry)
      {
        Core::IO::cout(Core::IO::standard) << ", ";
      }
      Core::IO::cout(Core::IO::standard) << direction << " = " << count;
      is_first_entry = false;
    }
    Core::IO::cout(Core::IO::standard) << Core::IO::endl;
  }

  const int num_candidate_relations =
      Core::Communication::sum_all(static_cast<int>(periodic_relation_nodes.size()), comm);
  Core::IO::cout(Core::IO::verbose)
      << "Candidate periodic relations: " << num_candidate_relations << Core::IO::endl;

  // Ensure no constraint is enforced twice (the automatic search already skips duplicates, the
  // manual definition relies on this check):
  int indx = 0;
  std::map<int, std::vector<int>> idListSet;

  for (const auto& node_set : periodic_relation_nodes)
  {
    std::vector<int> ids(node_set.begin(), node_set.end());
    std::sort(ids.begin(), ids.end());
    idListSet[indx++] = ids;
  }
  std::vector<int> idsToRemove;
  std::map<int, int> duplicate_sources;
  for (const auto& entryA : idListSet)
  {
    if (std::find(idsToRemove.begin(), idsToRemove.end(), entryA.first) == idsToRemove.end())
    {
      for (const auto& entryB : idListSet)
      {
        if (entryA.second == entryB.second && entryA.first != entryB.first)
        {
          idsToRemove.push_back(entryB.first);
          duplicate_sources.emplace(entryB.first, entryA.first);
        }
      }
    }
  }

  for (const int duplicate_id : idsToRemove)
  {
    const int original_id = duplicate_sources.at(duplicate_id);
    Core::IO::cout(Core::IO::debug) << "Removed " << pbc_labels[duplicate_id]
                                    << ": same node set as " << pbc_labels[original_id] << " [";

    const auto& node_ids = idListSet.at(duplicate_id);
    const char* separator = "";
    for (const int node_id : node_ids)
    {
      Core::IO::cout(Core::IO::debug) << separator << node_id;
      separator = ", ";
    }
    Core::IO::cout(Core::IO::debug) << "]" << Core::IO::endl;
  }

  // Remove duplicate constraints
  std::sort(idsToRemove.rbegin(), idsToRemove.rend());
  for (int id : idsToRemove)
  {
    periodic_relation_nodes.erase(periodic_relation_nodes.begin() + id);
    pbc_labels.erase(pbc_labels.begin() + id);
  }

  const int num_removed_duplicates =
      Core::Communication::sum_all(static_cast<int>(idsToRemove.size()), comm);
  const int num_unique_relations =
      Core::Communication::sum_all(static_cast<int>(periodic_relation_nodes.size()), comm);
  Core::IO::cout(Core::IO::verbose)
      << "Removed duplicate relations: " << num_removed_duplicates << Core::IO::endl;
  Core::IO::cout(Core::IO::standard)
      << "Unique periodic relations: " << num_unique_relations << Core::IO::endl;

  // every equation needs a unique global id: start counting where the lower ranks stop
  int num_my_constraint_equations = static_cast<int>(periodic_relation_nodes.size()) * num_dim;
  std::vector<int> num_constraint_equations_per_rank(Core::Communication::num_mpi_ranks(comm), 0);
  Core::Communication::gather_all(
      &num_my_constraint_equations, num_constraint_equations_per_rank.data(), 1, comm);
  int equation_id = 0;
  for (int pid = 0; pid < Core::Communication::my_mpi_rank(comm); ++pid)
    equation_id += num_constraint_equations_per_rank[pid];

  // dof of a node: looked up directly if this rank owns the node, otherwise from the gathered table
  auto dof_of = [&](const int node_gid, const int dim)
  {
    if (node_row_map.my_gid(node_gid))
      return discret_ptr_->dof(discret_ptr_->g_node(node_gid))[dim];
    return gathered_node_dofs.at(node_gid)[dim];
  };

  Core::IO::cout(Core::IO::verbose)
      << Core::IO::endl
      << "Constraint equation generation" << Core::IO::endl
      << "+--------------------------------------------------------------------+" << Core::IO::endl;

  // u(n0) - u(n1) - u(n2) + u(n3) = 0
  const std::vector<double> pbc_coefficients = {1., -1., -1., 1.};
  const std::string displacement_components = "xyz";
  for (std::size_t relation_id = 0; relation_id < periodic_relation_nodes.size(); ++relation_id)
  {
    const auto& node_set = periodic_relation_nodes[relation_id];
    Core::IO::cout(Core::IO::debug)
        << "R" << relation_id << " (" << pbc_labels[relation_id] << "): u(" << node_set[0]
        << ") - u(" << node_set[1] << ") = u(" << node_set[2] << ") - u(" << node_set[3] << ")"
        << Core::IO::endl;

    for (int dim = 0; dim < num_dim; ++dim)
    {
      std::vector<int> pbc_dofs;
      for (const int node_gid : node_set) pbc_dofs.push_back(dof_of(node_gid, dim));

      Core::IO::cout(Core::IO::debug)
          << "  Equation " << equation_id << ", " << displacement_components[dim]
          << " displacement: d" << pbc_dofs[0] << " - d" << pbc_dofs[1] << " - d" << pbc_dofs[2]
          << " + d" << pbc_dofs[3] << " = 0" << Core::IO::endl;

      constraint_equations_.emplace_back(
          std::make_shared<LinearCoupledEquation>(equation_id++, pbc_dofs, pbc_coefficients));
    }
  }
  Core::IO::cout(Core::IO::debug) << Core::IO::endl;

  Core::IO::cout(Core::IO::verbose)
      << "Displacement components per relation: " << num_dim << Core::IO::endl;
  Core::IO::cout(Core::IO::standard) << "Periodic boundary conditions initialized: "
                                     << std::accumulate(num_constraint_equations_per_rank.begin(),
                                            num_constraint_equations_per_rank.end(), 0)
                                     << " scalar constraint equations" << Core::IO::endl
                                     << Core::IO::endl;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::build_linear_mp_cs()
{
  Core::IO::cout(Core::IO::verbose)
      << Core::IO::endl
      << "-------------------------------------------" << Core::IO::endl;
  Core::IO::cout(Core::IO::verbose)
      << "Reading linear coupled eq. from input file" << Core::IO::endl;
  Core::IO::cout(Core::IO::verbose)
      << "linear MPC condition count: " << point_linear_coupled_equation_conditions_.size()
      << Core::IO::endl;


  int nEq = 0;
  for (const auto& ceTerm : point_linear_coupled_equation_conditions_)
  {
    nEq = std::max(nEq, (ceTerm->parameters().get<int>("EQUATION")) - 1);
  }
  Core::IO::cout(Core::IO::verbose)
      << "There are " << nEq + 1 << " linear MPC Equations defined" << Core::IO::endl;

  int dofPos;
  int cond_num = 0;
  std::vector<std::vector<int>> constraintRowIds(nEq + 1);
  std::vector<std::vector<int>> constraintColIds(nEq + 1);
  std::vector<std::vector<double>> constraintCoeffs(nEq + 1);


  for (const auto& ceTerm : point_linear_coupled_equation_conditions_)
  {
    auto eq_id = (ceTerm->parameters().get<int>("EQUATION")) - 1;
    const auto* node_id = ceTerm->get_nodes();
    const auto& dofStr = ceTerm->parameters().get<std::string>("ADD");
    auto coef = ceTerm->parameters().get<double>("COEFFICIENT");
    auto* node = discret_ptr_->g_node(node_id->data()[0]);


    if (dofStr == "dispx")
    {
      dofPos = 0;
    }
    else if (dofStr == "dispy")
    {
      dofPos = 1;
    }
    else if (dofStr == "dispz")
    {
      dofPos = 2;
    }
    else
    {
      FOUR_C_THROW(
          "No dof specified for node {} in linear coupled equation (MPC) condition. Set ADD to "
          "dispx, dispy or dispz.",
          node->id());
    }
    auto dofID = discret_ptr_->dof(node)[dofPos];


    Core::IO::cout(Core::IO::debug) << Core::IO::endl;
    Core::IO::cout(Core::IO::debug) << "Condition Number " << cond_num++ << ": " << Core::IO::endl;
    Core::IO::cout(Core::IO::debug) << "Eq.Id: " << eq_id << Core::IO::endl;
    Core::IO::cout(Core::IO::debug) << "Node Id: " << node_id->data()[0] << Core::IO::endl;
    Core::IO::cout(Core::IO::debug) << "Disp String: " << dofStr.c_str() << Core::IO::endl;
    Core::IO::cout(Core::IO::debug) << "DOF ID: " << dofID << Core::IO::endl;
    Core::IO::cout(Core::IO::debug) << "COEF: " << coef << Core::IO::endl << Core::IO::endl;

    // Save the linear MPCs
    constraintRowIds[eq_id].push_back(eq_id);
    constraintColIds[eq_id].push_back(dofID);
    constraintCoeffs[eq_id].push_back(coef);
    Core::IO::cout(Core::IO::debug) << "Added Term Equation with ID: " << eq_id << Core::IO::endl;
    Core::IO::cout(Core::IO::debug)
        << "Current SIze constraintColIDs" << constraintColIds.size() << Core::IO::endl;
  }
  // Get number of MPC already in the MPCs List
  int nMPC = 0;
  for (const auto& mpc : constraint_equations_)
  {
    nMPC += mpc->get_number_of_constraint_equation_objects();
  }
  unsigned int i = 0;
  for (; i < constraintRowIds.size(); ++i)
  {
    constraint_equations_.emplace_back(
        std::make_shared<LinearCoupledEquation>(nMPC++, constraintColIds[i], constraintCoeffs[i]));

    Core::IO::cout(Core::IO::verbose) << "Linear MPC #" << i << "  Created: 0 = ";
    for (unsigned int o = 0; o < constraintColIds[i].size(); ++o)
    {
      Core::IO::cout(Core::IO::verbose)
          << " +" << constraintCoeffs[i][o] << "*d" << constraintColIds[i][o];
    }
    Core::IO::cout(Core::IO::verbose) << Core::IO::endl;
  }
  Core::IO::cout(Core::IO::verbose) << "Number of Linear MPCs Created: " << i + 1 << Core::IO::endl;
  Core::IO::cout(Core::IO::verbose)
      << "Number of Elements in the listMPC: " << nMPC << Core::IO::endl;

  return i + 1;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::find_periodic_rve_corner_nodes(
    const std::vector<int>* edge1, const std::vector<int>* edge2)
{
  for (int nodeId : *edge2)
  {
    if (std::find(edge1->begin(), edge1->end(), nodeId) != edge1->end()) return nodeId;
  }
  return -1;
}

int Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::find_periodic_rve_corner_nodes(
    const std::vector<int>* surf1, const std::vector<int>* surf2, const std::vector<int>* surf3)
{
  std::vector<int> commonNodeIds12, commonNodeIds123;
  std::set_intersection(surf1->begin(), surf1->end(), surf2->begin(), surf2->end(),
      std::back_inserter(commonNodeIds12));

  std::set_intersection(surf3->begin(), surf3->end(), commonNodeIds12.begin(),
      commonNodeIds12.end(), std::back_inserter(commonNodeIds123));

  if (commonNodeIds123.size() < 1)
  {
    FOUR_C_THROW("No common node found");
  }

  else if (commonNodeIds123.size() > 1)
  {
    FOUR_C_THROW("More than one common node found");
  }
  return commonNodeIds123[0];
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::
    build_periodic_rve_corner_node_map(
        std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap,
        std::map<std::string, int>& rveCornerNodeIdMap)
{
  switch (rve_dim_)
  {
    case Constraints::MultiPoint::RveDimension::rve2d:
    {
      //* Get the Corner Node Ids */
      /*              N4 -- N
       *              |     |
       *      N   --- |     |---   N
       *      |                    |
       *    x-|                    |x+
       *      N1L ---|       | ----N2
       *             |       |
       *             N1B --  N
       *
       *      N1 ------ y- ------ N2
       */

      //* Get the Corner Node Ids */

      /*
       *      N4 ------y+------ N3
       *      |                    |
       *      x-                   |x+
       *      |                   |
       *      N1 ------ y- ------ N2
       */

      rveCornerNodeIdMap["N1"] =
          find_periodic_rve_corner_nodes(rveBoundaryNodeIdMap["x-"], rveBoundaryNodeIdMap["y-"]);
      rveCornerNodeIdMap["N2"] =
          find_periodic_rve_corner_nodes(rveBoundaryNodeIdMap["x+"], rveBoundaryNodeIdMap["y-"]);
      rveCornerNodeIdMap["N3"] =
          find_periodic_rve_corner_nodes(rveBoundaryNodeIdMap["x+"], rveBoundaryNodeIdMap["y+"]);
      rveCornerNodeIdMap["N4"] =
          find_periodic_rve_corner_nodes(rveBoundaryNodeIdMap["x-"], rveBoundaryNodeIdMap["y+"]);


      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Reference geometry" << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl
          << "Reference nodes: N1 = " << rveCornerNodeIdMap["N1"]
          << ", N2 = " << rveCornerNodeIdMap["N2"] << ", N3 = " << rveCornerNodeIdMap["N3"]
          << ", N4 = " << rveCornerNodeIdMap["N4"] << Core::IO::endl;
    }
    break;
    case Constraints::MultiPoint::RveDimension::rve3d:
    {
      //  z ^        N8 +  +   +   N7
      //    |      + .            + +
      //    |    +   .          +   +
      //    |  +     .        +     +
      //    N5 +  +  +  +  N3       +
      //    +       .      +        +
      //    +       N4 .  .+   . .. N3
      //    +      .       +     +
      //    +    .         +  +
      //    +  .           +
      //   N1 +  +  +  +  N2 ------> x
      //

      std::vector<std::string> boundaryNames = {"x+", "x-", "y+", "y-", "z+", "z-"};
      rveCornerNodeIdMap["N1"] = find_periodic_rve_corner_nodes(
          rveBoundaryNodeIdMap["x-"], rveBoundaryNodeIdMap["z-"], rveBoundaryNodeIdMap["y-"]);

      rveCornerNodeIdMap["N2"] = find_periodic_rve_corner_nodes(
          rveBoundaryNodeIdMap["x+"], rveBoundaryNodeIdMap["z-"], rveBoundaryNodeIdMap["y-"]);

      rveCornerNodeIdMap["N4"] = find_periodic_rve_corner_nodes(
          rveBoundaryNodeIdMap["x-"], rveBoundaryNodeIdMap["z-"], rveBoundaryNodeIdMap["y+"]);

      rveCornerNodeIdMap["N5"] = find_periodic_rve_corner_nodes(
          rveBoundaryNodeIdMap["x-"], rveBoundaryNodeIdMap["z+"], rveBoundaryNodeIdMap["y-"]);

      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Reference geometry" << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl
          << "Reference nodes: N1 = " << rveCornerNodeIdMap["N1"]
          << ", N2 = " << rveCornerNodeIdMap["N2"] << ", N4 = " << rveCornerNodeIdMap["N4"]
          << ", N5 = " << rveCornerNodeIdMap["N5"] << Core::IO::endl;
      break;
    }
  }
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::
    build_periodic_rve_boundary_node_map(
        std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap)
{
  const std::map<std::string, int> boundary_rank = {
      {"x-", 0}, {"x+", 1}, {"y-", 2}, {"y+", 3}, {"z-", 4}, {"z+", 5}};

  switch (rve_dim_)
  {
    case Constraints::MultiPoint::RveDimension::rve2d:
    {
      discret_ptr_->get_condition("LinePeriodicRve", line_periodic_rve_conditions_);

      auto sorted_line_conditions = line_periodic_rve_conditions_;
      std::ranges::sort(sorted_line_conditions, {},
          [&boundary_rank](const Core::Conditions::Condition* condition_line)
          {
            const auto& boundary = condition_line->parameters().get<std::string>("EDGE");
            return boundary_rank.at(boundary);
          });

      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Periodic boundary nodes" << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl;

      for (const auto& conditionLine : sorted_line_conditions)
      {
        const auto& boundary = conditionLine->parameters().get<std::string>("EDGE");

        Core::IO::cout(Core::IO::verbose)
            << "EDGE " << boundary << ": " << conditionLine->get_nodes()->size() << " nodes"
            << Core::IO::endl;

        print_node_ids_debug(*conditionLine->get_nodes());

        // Create EdgeNodeMap
        rveBoundaryNodeIdMap[boundary] = conditionLine->get_nodes();
      }
    }
    break;

    case Constraints::MultiPoint::RveDimension::rve3d:
    {
      discret_ptr_->get_condition("SurfacePeriodicRve", surface_periodic_rve_conditions_);

      auto sorted_surface_conditions = surface_periodic_rve_conditions_;
      std::ranges::sort(sorted_surface_conditions, {},
          [&boundary_rank](const Core::Conditions::Condition* condition_surface)
          {
            const auto& boundary = condition_surface->parameters().get<std::string>("SURF");
            return boundary_rank.at(boundary);
          });

      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Periodic boundary nodes" << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl;

      for (const auto& conditionSurface : sorted_surface_conditions)
      {
        const auto& boundary = conditionSurface->parameters().get<std::string>("SURF");

        Core::IO::cout(Core::IO::verbose)
            << "SURFACE " << boundary << ": " << conditionSurface->get_nodes()->size() << " nodes"
            << Core::IO::endl;

        print_node_ids_debug(*conditionSurface->get_nodes());

        // Create SurfaceNodeMap
        rveBoundaryNodeIdMap[boundary] = conditionSurface->get_nodes();
      }
      break;
    }
  }
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::reset() {}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/

std::map<Solid::EnergyType, double>
Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::get_energy() const
{
  FOUR_C_THROW("This function is not implemented for the RveMultiPointConstraintManager.");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE
