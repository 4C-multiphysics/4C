// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_constraint_framework_submodelevaluator_mpc.hpp"

#include "4C_beam3_base.hpp"
#include "4C_constraint_framework_equation.hpp"
#include "4C_constraint_framework_input.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometric_search_access_traits.hpp"
#include "4C_geometric_search_bounding_volume.hpp"
#include "4C_geometric_search_bvh.hpp"
#include "4C_geometric_search_input.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_structure_new_timint_implicit.hpp"

#include <algorithm>
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
  if (Core::Communication::num_mpi_ranks(discret_ptr_->get_comm()) > 1)
    FOUR_C_THROW("periodic boundary conditions for RVEs are not implemented in parallel.");

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
  std::vector<std::vector<Core::Nodes::Node*>> PBCs;
  std::vector<Core::Nodes::Node*> PBC;
  std::vector<std::string> pbc_labels;
  std::map<std::string, std::size_t> boundary_node_pair_counts;

  switch (rve_ref_type_)
  {
    case Constraints::MultiPoint::RveReferenceDeformationDefinition::automatic:
    {
      switch (rve_dim_)
      {
        case Constraints::MultiPoint::RveDimension::rve3d:
        case Constraints::MultiPoint::RveDimension::rve2d:
        {
          int numDim = 3;
          std::map<std::string, std::string> refEndNodeMap = {
              {"x", "N2"}, {"y", "N4"}, {"z", "N5"}};

          if (rve_dim_ == Constraints::MultiPoint::rve2d)
          {
            refEndNodeMap.erase("z");
            numDim = 2;
          }

          std::map<std::string, std::vector<double>> rveRefVecMap;
          std::vector<double> rveRefVector;

          for (const auto& surf : refEndNodeMap)
          {
            {
              rveRefVector.clear();
              for (int i = 0; i < numDim; ++i)
              {
                rveRefVector.push_back(
                    discret_ptr_->g_node(rveCornerNodeIdMap[surf.second])->x()[i] -
                    discret_ptr_->g_node(rveCornerNodeIdMap["N1"])->x()[i]);
              }
              rveRefVecMap[surf.first] = rveRefVector;

              Core::IO::cout(Core::IO::verbose) << "Reference vector " << surf.first << ": ["
                                                << rveRefVector[0] << "; " << rveRefVector[1];
              if (rve_dim_ == Constraints::MultiPoint::rve3d)
                Core::IO::cout(Core::IO::verbose) << "; " << rveRefVector[2];
              Core::IO::cout(Core::IO::verbose) << "]" << Core::IO::endl;
            }
          }

          Core::IO::cout(Core::IO::verbose)
              << Core::IO::endl
              << "Periodic relation search" << Core::IO::endl
              << "+--------------------------------------------------------------------+"
              << Core::IO::endl;

          // Create PBC Node Pairs:
          for (const auto& surf : refEndNodeMap)
          {
            std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> bounding_volumes_neg;
            std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> bounding_volumes_pos;

            // Use nodes on the negative-normal surface and shift by the reference vector
            // to obtain the target position of the corresponding node on the positive side.
            for (auto node_gid : *rveBoundaryNodeIdMap[surf.first + "-"])
            {
              // Don't include the reference node pair
              if (node_gid == rveCornerNodeIdMap["N1"]) continue;

              bounding_volumes_neg.emplace_back(
                  std::make_pair(node_gid, Core::GeometricSearch::BoundingVolume()));

              // calculate the target location
              Core::LinAlg::Matrix<3, 1, double> target_node_position;
              for (int i = 0; i < numDim; ++i)
                target_node_position(i) =
                    discret_ptr_->g_node(node_gid)->x()[i] + rveRefVecMap[surf.first][i];

              bounding_volumes_neg.back().second.add_point(target_node_position);
              bounding_volumes_neg.back().second.extend_boundaries(node_search_toler_);
            }

            // Get the actual position of the nodes on the positive-normal surface (primitives)
            for (auto node_gid : *rveBoundaryNodeIdMap[surf.first + "+"])
            {
              bounding_volumes_pos.emplace_back(
                  std::make_pair(node_gid, Core::GeometricSearch::BoundingVolume()));

              // get the actual location
              Core::LinAlg::Matrix<3, 1, double> actual_node_position;
              for (int i = 0; i < numDim; ++i)
                actual_node_position(i) = discret_ptr_->g_node(node_gid)->x()[i];

              bounding_volumes_pos.back().second.add_point(actual_node_position);
              bounding_volumes_pos.back().second.extend_boundaries(node_search_toler_);
            }

            Core::GeometricSearch::BoundingVolumeHierarchy bvh_side_pos(
                Core::GeometricSearch::BoundingVolumeVectorPlaceholder<
                    Core::GeometricSearch::PrimitivesTag>{bounding_volumes_pos});

            // Search all points on negative side in bvh of positive side
            const auto [indices, offsets] = bvh_side_pos.query(bounding_volumes_neg);
            Core::IO::cout(Core::IO::debug) << surf.first << "-boundary pairs:" << Core::IO::endl;

            std::size_t pair_number = 0;
            for (std::size_t i = 0; i + 1 < offsets.extent(0); ++i)
            {
              const int nHits = offsets(i + 1) - offsets(i);
              const int xm_id = bounding_volumes_neg[i].first;

              if (nHits > 1)
              {
                FOUR_C_THROW(
                    "Periodic search failed on surface '%s': x- node %d has %d matches. "
                    "Check mesh periodicity or SPHERE_RADIUS_EXTENSION_FACTOR.",
                    surf.first.c_str(), xm_id, nHits);
              }

              if (nHits == 0)
              {
                FOUR_C_THROW(
                    "Periodic search failed on surface '%s': no match for x- node %d. "
                    "Check mesh periodicity or SPHERE_RADIUS_EXTENSION_FACTOR.",
                    surf.first.c_str(), xm_id);
              }

              // The order matters, because of sign (1) - (2) = (3) - (4)
              PBC.push_back(discret_ptr_->g_node(indices(i)));                     // + side
              PBC.push_back(discret_ptr_->g_node(bounding_volumes_neg[i].first));  // - side
              PBC.push_back(discret_ptr_->g_node(rveCornerNodeIdMap[surf.second]));
              PBC.push_back(discret_ptr_->g_node(rveCornerNodeIdMap["N1"]));

              const std::string pair_label = surf.first + "-pair " + std::to_string(++pair_number);
              PBCs.push_back(PBC);
              pbc_labels.push_back(pair_label);

              Core::IO::cout(Core::IO::debug)
                  << "  " << pair_label << ": u(" << indices(i) << ") - u(" << xm_id << ") = u("
                  << surf.second << ":" << rveCornerNodeIdMap[surf.second]
                  << ") - u(N1:" << rveCornerNodeIdMap["N1"] << ")" << Core::IO::endl;

              PBC.clear();
            }
            boundary_node_pair_counts[surf.first] = pair_number;
            Core::IO::cout(Core::IO::debug) << Core::IO::endl;
          }
        }
        break;
      }
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
          PBC.push_back(discret_ptr_->g_node(nodeXm));
          PBC.push_back(discret_ptr_->g_node(find_opposite_edge_node(nodeXm,
              Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_xm, rveBoundaryNodeIdMap)));
          PBC.push_back(rve_ref_node_map_["N1L"]);
          PBC.push_back(rve_ref_node_map_["N2"]);

          const std::string pair_label = "x-pair " + std::to_string(++x_pair_number);
          PBCs.push_back(PBC);
          pbc_labels.push_back(pair_label);

          Core::IO::cout(Core::IO::debug)
              << "  " << pair_label << ": u(" << PBC[0]->id() << ") - u(" << PBC[1]->id()
              << ") = u(N1L:" << PBC[2]->id() << ") - u(N2:" << PBC[3]->id() << ")"
              << Core::IO::endl;

          PBC.clear();
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
          PBC.push_back(discret_ptr_->g_node(find_opposite_edge_node(nodeYm,
              Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_ym, rveBoundaryNodeIdMap)));
          PBC.push_back(discret_ptr_->g_node(nodeYm));
          PBC.push_back(rve_ref_node_map_["N4"]);
          PBC.push_back(rve_ref_node_map_["N1B"]);

          const std::string pair_label = "y-pair " + std::to_string(++y_pair_number);
          PBCs.push_back(PBC);
          pbc_labels.push_back(pair_label);

          Core::IO::cout(Core::IO::debug) << "  " << pair_label << ": u(" << PBC[0]->id()
                                          << ") - u(" << PBC[1]->id() << ") = u(N4:" << PBC[2]->id()
                                          << ") - u(N1B:" << PBC[3]->id() << ")" << Core::IO::endl;

          PBC.clear();
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

  Core::IO::cout(Core::IO::verbose)
      << "Candidate periodic relations: " << PBCs.size() << Core::IO::endl;

  // Ensure no constraint is enforced twice:
  int indx = 0;
  std::vector<int> ids;
  std::map<int, std::vector<int>> idListSet;

  for (const auto& pbc : PBCs)
  {
    for (auto* node : pbc)
    {
      ids.push_back(node->id());
    }
    std::sort(ids.begin(), ids.end());
    idListSet[indx++] = ids;
    ids.clear();
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
    PBCs.erase(PBCs.begin() + id);
    pbc_labels.erase(pbc_labels.begin() + id);
  }

  Core::IO::cout(Core::IO::verbose)
      << "Removed duplicate relations: " << idsToRemove.size() << Core::IO::endl;
  Core::IO::cout(Core::IO::standard)
      << "Unique periodic relations: " << PBCs.size() << Core::IO::endl;

  // Create the vector of MPC "Elements
  int mpcId = 0;
  std::vector<int> pbcDofs;
  std::vector<double> pbcCoefs = {1., -1., -1., 1.};

  int nDofCoupled = 3;
  if (rve_dim_ == Constraints::MultiPoint::rve2d)
  {
    nDofCoupled = 2;
  }


  Core::IO::cout(Core::IO::verbose)
      << Core::IO::endl
      << "Constraint equation generation" << Core::IO::endl
      << "+--------------------------------------------------------------------+" << Core::IO::endl;

  const std::string displacement_components = "xyz";
  for (std::size_t relation_id = 0; relation_id < PBCs.size(); ++relation_id)
  {
    const auto& pbc = PBCs[relation_id];
    Core::IO::cout(Core::IO::debug)
        << "R" << relation_id << " (" << pbc_labels[relation_id] << "): u(" << pbc[0]->id()
        << ") - u(" << pbc[1]->id() << ") = u(" << pbc[2]->id() << ") - u(" << pbc[3]->id() << ")"
        << Core::IO::endl;

    for (int dim = 0; dim < nDofCoupled; ++dim)
    {
      pbcDofs.clear();
      for (auto* node : pbc)
      {
        // Create coupled equation dof list:
        pbcDofs.emplace_back(discret_ptr_->dof(node)[dim]);
      }

      // Signs follow pbcCoefs = {1, -1, -1, 1}
      Core::IO::cout(Core::IO::debug)
          << "  Equation " << mpcId << ", " << displacement_components[dim] << " displacement: d"
          << pbcDofs[0] << " - d" << pbcDofs[1] << " - d" << pbcDofs[2] << " + d" << pbcDofs[3]
          << " = 0" << Core::IO::endl;

      constraint_equations_.emplace_back(
          std::make_shared<LinearCoupledEquation>(mpcId++, pbcDofs, pbcCoefs));
    }
  }
  Core::IO::cout(Core::IO::debug) << Core::IO::endl;

  Core::IO::cout(Core::IO::verbose)
      << "Displacement components per relation: " << nDofCoupled << Core::IO::endl;
  Core::IO::cout(Core::IO::standard)
      << "Periodic boundary conditions initialized: " << constraint_equations_.size()
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
    else
    {
      FOUR_C_THROW(
          "Linear coupled equations (MPCs) are only implemented for 2D (dispx or dispy DOFs)");
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
