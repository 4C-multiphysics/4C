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
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometric_search_bounding_volume.hpp"
#include "4C_geometric_search_distributed_tree.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_structure_new_timint_implicit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <vector>

#ifdef FOUR_C_ENABLE_FE_TRAPPING
#include <cfenv>
#endif

FOUR_C_NAMESPACE_OPEN

namespace
{
  struct PeriodicRveMpcNodeSet
  {
    int plus_gid;
    int minus_gid;
    int ref_end_gid;
    int ref_base_gid;
  };

  //! Suspend floating point exception trapping while in scope. ArborX may raise benign fp
  //! exceptions on ranks whose local search input is empty.
  class SuspendFloatingPointTrapping
  {
   public:
    SuspendFloatingPointTrapping()
    {
#ifdef FOUR_C_ENABLE_FE_TRAPPING
      enabled_excepts_ = fedisableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#endif
    }
    ~SuspendFloatingPointTrapping()
    {
#ifdef FOUR_C_ENABLE_FE_TRAPPING
      feclearexcept(FE_ALL_EXCEPT);
      feenableexcept(enabled_excepts_);
#endif
    }
    SuspendFloatingPointTrapping(const SuspendFloatingPointTrapping&) = delete;
    SuspendFloatingPointTrapping& operator=(const SuspendFloatingPointTrapping&) = delete;

#ifdef FOUR_C_ENABLE_FE_TRAPPING
   private:
    int enabled_excepts_ = 0;
#endif
  };
}  // namespace

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::RveMultiPointConstraintManager(
    std::shared_ptr<Core::FE::Discretization> disc_ptr, Core::LinAlg::SparseMatrix* st_ptr)
{
  discret_ptr_ = disc_ptr;
  writable_discret_ = std::move(disc_ptr);
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
        rve_ref_node_map_[str_id.c_str()] = discret_ptr_->g_node(nodeInSet->data()[0]);

        Core::IO::cout(Core::IO::verbose)
            << "Map " << str_id.c_str() << " to node id " << (*nodeInSet)[0] << Core::IO::endl;
      }

      Core::IO::cout(Core::IO::verbose)
          << "Reference Points determined:"
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl;
      for (const auto& elem : rve_ref_node_map_)
      {
        Core::IO::cout(Core::IO::verbose) << elem.first << ": " << elem.second->id() << ", ";
      }
      Core::IO::cout(Core::IO::verbose) << Core::IO::endl;

      // calculate the Reference vectors between Ref. points
      r_xmxp_[0] = rve_ref_node_map_["N2"]->x()[0] - rve_ref_node_map_["N1L"]->x()[0];
      r_xmxp_[1] = rve_ref_node_map_["N2"]->x()[1] - rve_ref_node_map_["N1L"]->x()[1];
      Core::IO::cout(Core::IO::verbose) << "RVE reference vector (X- ---> X+ ) : [" << r_xmxp_[0]
                                        << ", " << r_xmxp_[1] << "]" << Core::IO::endl;

      r_ymyp_[0] = rve_ref_node_map_["N4"]->x()[0] - rve_ref_node_map_["N1B"]->x()[0];
      r_ymyp_[1] = rve_ref_node_map_["N4"]->x()[1] - rve_ref_node_map_["N1B"]->x()[1];
      Core::IO::cout(Core::IO::verbose) << "RVE reference vector (Y- ---> Y+ ) : [" << r_ymyp_[0]
                                        << ", " << r_ymyp_[1] << "]" << Core::IO::endl;
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
  auto geom_search_parameter_list = Global::Problem::instance()->geometric_search_params();
  auto constraint_parameter_list = Global::Problem::instance()->constraint_params();

  strategy_ = Teuchos::getIntegralValue<Constraints::EnforcementStrategy>(
      constraint_parameter_list, "CONSTRAINT_ENFORCEMENT");

  auto mpc_parameter_list = constraint_parameter_list.sublist("MULTI POINT");

  rve_ref_type_ =
      Teuchos::getIntegralValue<Constraints::MultiPoint::RveReferenceDeformationDefinition>(
          mpc_parameter_list, "RVE_REFERENCE_POINTS");

  node_search_toler_ = Teuchos::getDoubleParameter(geom_search_parameter_list, "POINT_TOLERANCE");

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
      Core::IO::cout(Core::IO::minimal)
          << "Constraint enforcement strategy: Penalty method" << Core::IO::endl;

      get_penalty_parameter_ptr() = constraint_parameter_list.get<double>("PENALTY_PARAM");
      Core::IO::cout(Core::IO::verbose)
          << "Penalty weight used: " << get_penalty_parameter_ptr() << Core::IO::endl;

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
  if (is_parallel &&
      rve_ref_type_ == Constraints::MultiPoint::RveReferenceDeformationDefinition::manual)
  {
    FOUR_C_THROW("Manual RVE reference points are not implemented in parallel.");
  }
  if (is_parallel && !point_linear_coupled_equation_conditions_.empty())
  {
    FOUR_C_THROW("PointLinearCoupledEquation constraints are not implemented in parallel.");
  }

  // Input Checks: Dimensions
  if (line_periodic_rve_conditions_.size() == 0 && surface_periodic_rve_conditions_.size() != 0)
  {
    Core::IO::cout(Core::IO::verbose) << "Rve dimension: 3d" << Core::IO::endl;
    rve_dim_ = Constraints::MultiPoint::RveDimension::rve3d;
  }
  else if (line_periodic_rve_conditions_.size() != 0 &&
           surface_periodic_rve_conditions_.size() == 0)
  {
    Core::IO::cout(Core::IO::verbose) << "Rve dimensions: 2d" << Core::IO::endl;
    rve_dim_ = Constraints::MultiPoint::RveDimension::rve2d;
  }
  else
  {
    FOUR_C_THROW("Periodic rve edge condition cannot be combined with peridodic rve surf cond. ");
  }

  // Input Checks
  Core::IO::cout(Core::IO::verbose)
      << "There are " << line_periodic_rve_conditions_.size()
      << " periodic rve edge conditions defined (2D)" << Core::IO::endl;
  if (line_periodic_rve_conditions_.size() != 0)
  {
    if (line_periodic_rve_conditions_.size() != 4 && line_periodic_rve_conditions_.size() != 2)
    {
      Core::IO::cout(Core::IO::verbose)
          << "Number of Conditions: " << line_periodic_rve_conditions_.size() << Core::IO::endl;
      FOUR_C_THROW("For a 2D RVE either all or two opposing edges must be used for PBCs");
    }
  }

  Core::IO::cout(Core::IO::verbose)
      << "There are " << surface_periodic_rve_conditions_.size()
      << " periodic rve surface conditions defined (3D)" << Core::IO::endl;

  Core::IO::cout(Core::IO::verbose)
      << "There are " << point_linear_coupled_equation_conditions_.size()
      << " linear coupled equations" << Core::IO::endl;

  Core::IO::cout(Core::IO::verbose)
      << "The geometric search tolerance is set to: " << node_search_toler_ << Core::IO::endl;

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
      Core::IO::cout(Core::IO::debug)
          << " Find partner node of Node " << nodeID << " on Edge:  " << newPos << Core::IO::endl;
      break;
    }

    case Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_ym:
    {
      R_ipim = r_ymyp_;
      newPos = "y+";
      Core::IO::cout(Core::IO::debug)
          << " Find partner node of Node " << nodeID << " on Edge:  " << newPos << Core::IO::endl;
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

  Core::IO::cout(Core::IO::debug) << "Calculated position of matching node: " << newPosition[0]
                                  << " / " << newPosition[1] << Core::IO::endl;

  // Loop all nodes of the relevant opposite edge line
  // ToDo: Switch to ArborX
  for (auto pairId : *rveBoundaryNodeIdMap[newPos])
  {
    Core::Nodes::Node* nodeB = discret_ptr_->g_node(pairId);

    if (std::abs(nodeB->x()[0] - newPosition[0]) < node_search_toler_)
    {
      if (std::abs(nodeB->x()[1] - newPosition[1]) < node_search_toler_)
      {
        Core::IO::cout(Core::IO::debug)
            << "Found Node Pair (IDs): A: " << nodeA->id() << " B: " << pairId << Core::IO::endl;
        return pairId;
      }
    }
  }
  FOUR_C_THROW("No matching node found! - Is the mesh perodic?");
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::build_periodic_mp_cs(
    std::map<std::string, const std::vector<int>*>& rveBoundaryNodeIdMap,
    std::map<std::string, int>& rveCornerNodeIdMap)
{
  std::vector<std::vector<Core::Nodes::Node*>> PBCs;
  std::vector<Core::Nodes::Node*> PBC;
  {
    Core::IO::cout(Core::IO::verbose)
        << "\nCreating Node Pairs for Periodic Boundary Conditions" << Core::IO::endl
        << "+--------------------------------------------------------------------+"
        << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose) << "RVE Type: ";
  }
  switch (rve_ref_type_)
  {
    case Constraints::MultiPoint::RveReferenceDeformationDefinition::automatic:
    {
      const MPI_Comm comm = writable_discret_->get_comm();
      const int my_rank = Core::Communication::my_mpi_rank(comm);
      const Core::LinAlg::Map& node_row_map = *writable_discret_->node_row_map();
      const int num_dim = (rve_dim_ == Constraints::MultiPoint::rve2d) ? 2 : 3;

      Core::IO::cout(Core::IO::verbose)
          << (num_dim == 2 ? "General 2D RVE" : "General 3D RVE") << Core::IO::endl;

      std::map<std::string, std::string> ref_end_node_map = {{"x", "N2"}, {"y", "N4"}, {"z", "N5"}};
      if (rve_dim_ == Constraints::MultiPoint::rve2d) ref_end_node_map.erase("z");

      // gather the corner node coordinates needed for the reference vectors
      std::set<int> corner_gids = {rveCornerNodeIdMap["N1"]};
      for (const auto& [axis, end_node] : ref_end_node_map)
        corner_gids.insert(rveCornerNodeIdMap[end_node]);

      std::map<int, std::array<double, 3>> local_corner_coordinates;
      for (const int corner_gid : corner_gids)
        if (node_row_map.my_gid(corner_gid))
        {
          const auto x = writable_discret_->g_node(corner_gid)->x();
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
      }

      // shift each negative-boundary node onto the positive boundary and match it there; the
      // match is returned to the owner of the negative node, which then owns the constraint
      std::vector<PeriodicRveMpcNodeSet> pbc_node_sets;
      for (const auto& [axis, end_node] : ref_end_node_map)
      {
        const int ref_end_gid = rveCornerNodeIdMap[end_node];
        const int ref_base_gid = rveCornerNodeIdMap["N1"];

        std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> shifted_negative_nodes;
        for (const int minus_gid : *rveBoundaryNodeIdMap.at(axis + "-"))
        {
          if (minus_gid == ref_base_gid || !node_row_map.my_gid(minus_gid)) continue;
          const auto x = writable_discret_->g_node(minus_gid)->x();
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
          const auto x = writable_discret_->g_node(plus_gid)->x();
          Core::LinAlg::Matrix<3, 1, double> actual_position(Core::LinAlg::Initialization::zero);
          for (int i = 0; i < num_dim; ++i) actual_position(i) = x[i];
          Core::GeometricSearch::BoundingVolume bounding_volume;
          bounding_volume.add_point(actual_position);
          bounding_volume.extend_boundaries(node_search_toler_);
          positive_nodes.emplace_back(plus_gid, bounding_volume);
        }

        std::vector<Core::GeometricSearch::GlobalCollisionSearchResult> matches;
        {
          SuspendFloatingPointTrapping suspend_fpe;
          matches = Core::GeometricSearch::global_collision_search(
              positive_nodes, shifted_negative_nodes, comm);
        }

        std::map<int, std::vector<int>> positive_partners;
        for (const auto& match : matches)
          positive_partners[match.gid_predicate].push_back(match.gid_primitive);

        for (const auto& [minus_gid, partners] : positive_partners)
        {
          if (partners.size() != 1)
            FOUR_C_THROW(
                "Periodic search on the '{}-' boundary found {} partners for node {} "
                "(expected exactly one). Check mesh periodicity or POINT_TOLERANCE.",
                axis, partners.size(), minus_gid);

          // Order matters because of the signs in (1) - (2) = (3) - (4)
          pbc_node_sets.push_back({.plus_gid = partners[0],
              .minus_gid = minus_gid,
              .ref_end_gid = ref_end_gid,
              .ref_base_gid = ref_base_gid});
        }
      }

      // ghost the partner and corner nodes so all four nodes of a constraint are local
      std::set<int> nodes_to_ghost;
      for (const auto& node_set : pbc_node_sets)
      {
        nodes_to_ghost.insert(node_set.plus_gid);
        nodes_to_ghost.insert(node_set.ref_end_gid);
        nodes_to_ghost.insert(node_set.ref_base_gid);
      }
      ghost_nodes(std::vector<int>(nodes_to_ghost.begin(), nodes_to_ghost.end()));

      // global row offset of this rank's constraint block
      int num_local_constraints = static_cast<int>(pbc_node_sets.size()) * num_dim;
      std::vector<int> num_constraints_per_rank(Core::Communication::num_mpi_ranks(comm), 0);
      Core::Communication::gather_all(
          &num_local_constraints, num_constraints_per_rank.data(), 1, comm);
      int mpc_id = 0;
      for (int pid = 0; pid < my_rank; ++pid) mpc_id += num_constraints_per_rank[pid];

      const std::vector<double> pbc_coefficients = {1., -1., -1., 1.};
      std::vector<int> owned_constraint_row_ids;
      for (const auto& node_set : pbc_node_sets)
      {
        const std::vector<int> plus_dofs =
            writable_discret_->dof(writable_discret_->g_node(node_set.plus_gid));
        const std::vector<int> minus_dofs =
            writable_discret_->dof(writable_discret_->g_node(node_set.minus_gid));
        const std::vector<int> ref_end_dofs =
            writable_discret_->dof(writable_discret_->g_node(node_set.ref_end_gid));
        const std::vector<int> ref_base_dofs =
            writable_discret_->dof(writable_discret_->g_node(node_set.ref_base_gid));
        for (int dim = 0; dim < num_dim; ++dim)
        {
          const std::vector<int> pbc_dofs = {
              plus_dofs[dim], minus_dofs[dim], ref_end_dofs[dim], ref_base_dofs[dim]};
          owned_constraint_row_ids.push_back(mpc_id);
          constraint_equations_.emplace_back(
              std::make_shared<LinearCoupledEquation>(mpc_id++, pbc_dofs, pbc_coefficients));
        }
      }
      set_owned_constraint_row_ids(std::move(owned_constraint_row_ids));

      Core::IO::cout(Core::IO::verbose)
          << "\nNumber of periodic constraint equations on this rank: "
          << constraint_equations_.size() << Core::IO::endl;
      return;
    }
    case Constraints::MultiPoint::RveReferenceDeformationDefinition::manual:
    {
      Core::IO::cout(Core::IO::verbose) << "General 2D RVE" << Core::IO::endl;
      for (const auto& elem : rve_ref_node_map_)
      {
        Core::IO::cout(Core::IO::verbose) << elem.first << " first " << elem.second->id() << " sec "
                                          << "\n";
      }

      /* Loop over X- Edge */
      for (auto nodeXm : *rveBoundaryNodeIdMap["x-"])
      {
        if (nodeXm != rve_ref_node_map_["N1L"]->id())  // exclude N1 - N2 = N1 - N2
        {
          PBC.push_back(discret_ptr_->g_node(nodeXm));
          PBC.push_back(discret_ptr_->g_node(find_opposite_edge_node(nodeXm,
              Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_xm, rveBoundaryNodeIdMap)));
          PBC.push_back(rve_ref_node_map_["N1L"]);
          PBC.push_back(rve_ref_node_map_["N2"]);
          PBCs.push_back(PBC);

          Core::IO::cout(Core::IO::debug)
              << "PBC MPC Set created X- ---> X+ Edge:  " << Core::IO::endl;
          for (auto* nnn : PBC)
          {
            Core::IO::cout(Core::IO::debug) << " ___ " << nnn->id();
          }
          Core::IO::cout(Core::IO::debug) << Core::IO::endl;
          PBC.clear();
        }
      }

      /* Loop over Y- Edge*/
      for (auto nodeYm : *rveBoundaryNodeIdMap["y-"])
      {
        if (nodeYm != rve_ref_node_map_["N1B"]->id() && nodeYm != rve_ref_node_map_["N2"]->id())
        {
          PBC.push_back(discret_ptr_->g_node(find_opposite_edge_node(nodeYm,
              Constraints::MultiPoint::RveEdgeIdentifiers::Gamma_ym, rveBoundaryNodeIdMap)));
          PBC.push_back(discret_ptr_->g_node(nodeYm));
          PBC.push_back(rve_ref_node_map_["N4"]);
          PBC.push_back(rve_ref_node_map_["N1B"]);
          PBCs.push_back(PBC);
          PBC.clear();
        }
      }
      break;
    }
    default:
      FOUR_C_THROW("No ref def type defined");
  }

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


  // Print
  int nr = 0;
  Core::IO::cout(Core::IO::debug) << "Sorted Pair Ids:" << Core::IO::endl;
  for (const auto& couple : idListSet)
  {
    Core::IO::cout(Core::IO::debug) << "Pair " << nr++ << ":";
    for (auto id : couple.second)
    {
      Core::IO::cout(Core::IO::debug) << id << ", ";
    }
    Core::IO::cout(Core::IO::debug) << Core::IO::endl;
  }

  std::vector<int> idsToRemove;
  for (const auto& entryA : idListSet)
  {
    if (std::find(idsToRemove.begin(), idsToRemove.end(), entryA.first) == idsToRemove.end())
    {
      for (const auto& entryB : idListSet)
      {
        if (entryA.second == entryB.second && entryA.first != entryB.first)
        {
          idsToRemove.push_back(entryB.first);
          Core::IO::cout(Core::IO::debug) << "remove pair: " << entryB.first << Core::IO::endl;
        }
      }
    }
  }

  // Remove duplicate constraints
  std::sort(idsToRemove.rbegin(), idsToRemove.rend());
  for (int id : idsToRemove)
  {
    PBCs.erase(PBCs.begin() + id);
  }
  Core::IO::cout(Core::IO::debug) << "All Node Pairs found. Following Nodes are coupled:"
                                  << Core::IO::endl;

  // Create the vector of MPC "Elements
  int mpcId = 0;
  std::vector<int> pbcDofs;
  std::vector<double> pbcCoefs = {1., -1., -1., 1.};

  int nDofCoupled = 3;
  if (rve_dim_ == Constraints::MultiPoint::rve2d)
  {
    nDofCoupled = 2;
  }


  for (const auto& pbc : PBCs)
  {
    Core::IO::cout(Core::IO::debug) << "Node Pair: ";
    for (int dim = 0; dim < nDofCoupled; ++dim)
    {
      pbcDofs.clear();
      for (auto* node : pbc)
      {
        Core::IO::cout(Core::IO::debug) << node->id() << ", ";

        // Create coupled equation dof list:
        pbcDofs.emplace_back(discret_ptr_->dof(node)[dim]);
      }
      Core::IO::cout(Core::IO::debug) << "\ndofs used in linear coupled equation: \n ";
      for (auto dof : pbcDofs)
      {
        Core::IO::cout(Core::IO::debug) << dof << ", ";
      }
      Core::IO::cout(Core::IO::debug) << Core::IO::endl;
      constraint_equations_.emplace_back(
          std::make_shared<LinearCoupledEquation>(mpcId++, pbcDofs, pbcCoefs));
    }
    Core::IO::cout(Core::IO::debug) << "\n";
  }
  Core::IO::cout(Core::IO::verbose)
      << Core::IO::endl
      << "Total number of node pairs created for periodic boundary conditions: "
      << constraint_equations_.size() << Core::IO::endl;
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
          << "\nAutomatically determined following reference Nodes: " << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "---------------------------------------------------" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "N1: " << rveCornerNodeIdMap["N1"] << ", ";
      Core::IO::cout(Core::IO::verbose) << "N2: " << rveCornerNodeIdMap["N2"] << ", ";
      Core::IO::cout(Core::IO::verbose) << "N3: " << rveCornerNodeIdMap["N3"] << ", ";
      Core::IO::cout(Core::IO::verbose) << "N4: " << rveCornerNodeIdMap["N4"] << Core::IO::endl;
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
          << "\nAutomatically determined following reference Nodes: " << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "---------------------------------------------------" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "N1: " << rveCornerNodeIdMap["N1"] << ", ";
      Core::IO::cout(Core::IO::verbose) << "N2: " << rveCornerNodeIdMap["N2"] << ", ";
      Core::IO::cout(Core::IO::verbose) << "N4: " << rveCornerNodeIdMap["N4"] << ", ";
      Core::IO::cout(Core::IO::verbose) << "N5: " << rveCornerNodeIdMap["N5"] << Core::IO::endl;
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
  switch (rve_dim_)
  {
    case Constraints::MultiPoint::RveDimension::rve2d:
    {
      discret_ptr_->get_condition("LinePeriodicRve", line_periodic_rve_conditions_);



      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Reading Line Conditions:  " << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl;
      for (const auto& conditionLine : line_periodic_rve_conditions_)
      {
        const auto& boundary = conditionLine->parameters().get<std::string>("EDGE");

        // Print the Edge Condition
        Core::IO::cout(Core::IO::verbose) << "EDGE: " << boundary.c_str() << " Node IDs: ";
        for (auto nodeId : *conditionLine->get_nodes())
        {
          Core::IO::cout(Core::IO::verbose) << nodeId << " ";
        }
        Core::IO::cout(Core::IO::verbose) << Core::IO::endl;

        // Create EdgeNodeMap
        rveBoundaryNodeIdMap[boundary.c_str()] = conditionLine->get_nodes();
      }
    }
    break;

    case Constraints::MultiPoint::RveDimension::rve3d:
    {
      discret_ptr_->get_condition("SurfacePeriodicRve", surface_periodic_rve_conditions_);

      Core::IO::cout(Core::IO::verbose)
          << Core::IO::endl
          << "Reading Surface Conditions:  " << Core::IO::endl
          << "+--------------------------------------------------------------------+"
          << Core::IO::endl;
      for (const auto& conditionLine : surface_periodic_rve_conditions_)
      {
        const auto& boundary = conditionLine->parameters().get<std::string>("SURF");

        Core::IO::cout(Core::IO::verbose) << "SURF: " << boundary.c_str() << " Node IDs: ";
        for (auto nodeId : *conditionLine->get_nodes())
        {
          Core::IO::cout(Core::IO::verbose) << nodeId << " ";
        }
        Core::IO::cout(Core::IO::verbose) << Core::IO::endl;
        rveBoundaryNodeIdMap[boundary.c_str()] = conditionLine->get_nodes();
      }
      break;
    }
  }
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::SubmodelEvaluator::RveMultiPointConstraintManager::ghost_nodes(
    const std::vector<int>& node_gids)
{
  const Core::LinAlg::Map& dof_row_map = *writable_discret_->dof_row_map();
  const std::vector<int> dof_gids_before(dof_row_map.my_global_elements(),
      dof_row_map.my_global_elements() + dof_row_map.num_my_elements());

  // add the requested nodes to the column map
  const Core::LinAlg::Map& node_col_map = *writable_discret_->node_col_map();
  std::vector<int> ghosted_node_gids(node_col_map.my_global_elements(),
      node_col_map.my_global_elements() + node_col_map.num_my_elements());
  ghosted_node_gids.insert(ghosted_node_gids.end(), node_gids.begin(), node_gids.end());
  std::sort(ghosted_node_gids.begin(), ghosted_node_gids.end());
  ghosted_node_gids.erase(
      std::unique(ghosted_node_gids.begin(), ghosted_node_gids.end()), ghosted_node_gids.end());

  // the refill below is collective, so decide collectively whether to skip it
  const int local_needs_ghosting =
      static_cast<int>(ghosted_node_gids.size()) > node_col_map.num_my_elements() ? 1 : 0;
  if (Core::Communication::max_all(local_needs_ghosting, writable_discret_->get_comm()) == 0)
    return;

  Core::LinAlg::Map ghosted_node_map(-1, static_cast<int>(ghosted_node_gids.size()),
      ghosted_node_gids.data(), 0, writable_discret_->get_comm());
  writable_discret_->export_column_nodes(ghosted_node_map);
  writable_discret_->fill_complete();

  // ghosting must not change the dof row map
  const Core::LinAlg::Map& dof_row_map_after = *writable_discret_->dof_row_map();
  const std::vector<int> dof_gids_after(dof_row_map_after.my_global_elements(),
      dof_row_map_after.my_global_elements() + dof_row_map_after.num_my_elements());
  if (dof_gids_before != dof_gids_after)
    FOUR_C_THROW("Ghosting the periodic partner nodes changed the dof row map.");
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
