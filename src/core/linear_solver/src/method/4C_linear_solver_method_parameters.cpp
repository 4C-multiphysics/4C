// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_method_parameters.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_utils_exceptions.hpp"

#include <Xpetra_EpetraIntMultiVector.hpp>

#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::Parameters::compute_solver_parameters(
    const Core::FE::Discretization& dis, Teuchos::ParameterList& solverlist)
{
  if (!dis.filled() or !dis.have_dofs())
  {
    FOUR_C_THROW(
        "Solver parameters can only be calculated on a filled discretization with assigned degrees "
        "of freedom.");
  }

  const auto nullspace_node_map =
      solverlist.get<std::shared_ptr<Core::LinAlg::Map>>("null space: node map", nullptr);
  auto nullspace_dof_map =
      solverlist.get<std::shared_ptr<Core::LinAlg::Map>>("null space: dof map", nullptr);

  int dimns = -1;

  // set parameter information for solver
  {
    int numdf = -1;

    if (nullspace_node_map == nullptr)
    {
      // no map given, just grab the block information on the first element that appears
      if (dis.num_my_row_elements() > 0)
      {
        auto* element = dis.l_row_element(0);
        element->element_type().nodal_block_information(element, numdf, dimns);
      }
    }
    else
    {
      // if a map is given, grab the block information of the first element in that map
      for (int i = 0; i < dis.num_my_row_nodes(); ++i)
      {
        auto* node = dis.l_row_node(i);

        if (nullspace_node_map->lid(node->id()) == -1) continue;
        if (node->adjacent_elements().empty()) continue;

        auto* element = node->adjacent_elements()[0].user_element();
        element->element_type().nodal_block_information(element, numdf, dimns);

        break;
      }
    }

    // communicate data to procs without row element
    std::array<int, 2> ldata{numdf, dimns};
    std::array<int, 2> gdata{0, 0};
    gdata = Core::Communication::max_all(ldata, dis.get_comm());
    numdf = gdata[0];
    dimns = gdata[1];

    FOUR_C_ASSERT_ALWAYS(numdf > 0,
        "Determination of 'numdf' did not work, it has still the unphysical default value!");
    FOUR_C_ASSERT_ALWAYS(dimns > 0,
        "Determination of 'dimns' did not work, it has still the unphysical default value!");

    // store dof information in solver list
    solverlist.set("PDE equations", numdf);
  }

  // set coordinate information
  {
    std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates;
    if (nullspace_node_map == nullptr)
      coordinates = extract_retained_node_coordinates(dis, *dis.node_row_map());
    else
      coordinates = extract_retained_node_coordinates(dis, *nullspace_node_map);

    solverlist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates", coordinates);
  }

  // set nullspace information
  {
    if (nullspace_dof_map == nullptr)
    {
      // if no map is given, we calculate the nullspace on the map describing the
      // whole discretization
      nullspace_dof_map = std::make_shared<Core::LinAlg::Map>(*dis.dof_row_map());
    }

    const auto nullspace = Core::FE::compute_null_space(dis, dimns, *nullspace_dof_map);

    solverlist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspace);
  }
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::Parameters::fix_null_space(const std::string& field,
    const Core::LinAlg::Map& oldmap, const Core::LinAlg::Map& newmap,
    Teuchos::ParameterList& solveparams)
{
  if (!Core::Communication::my_mpi_rank(oldmap.get_comm()))
    printf("Fixing %s Nullspace\n", field.c_str());

  // find the Teko or MueLu list
  Teuchos::ParameterList* params_ptr = nullptr;
  if (solveparams.isSublist("MueLu Parameters"))
    params_ptr = &(solveparams.sublist("MueLu Parameters"));
  else
    params_ptr = &(solveparams);
  Teuchos::ParameterList& params = *params_ptr;

  const auto nullspace =
      params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullptr);
  if (nullspace == nullptr) FOUR_C_THROW("List does not contain nullspace");

  const int ndim = nullspace->num_vectors();

  const int nullspaceLength = nullspace->local_length();
  const int newmapLength = newmap.num_my_elements();

  // Do nothing if the map of the nullspace and the new map match
  if (nullspace->get_map().same_as(newmap)) return;

  if (nullspaceLength != oldmap.num_my_elements())
    FOUR_C_THROW("Nullspace map of length {} does not match old map length of {}", nullspaceLength,
        oldmap.num_my_elements());
  if (newmapLength > nullspaceLength)
    FOUR_C_THROW("New problem size larger than old - full rebuild of nullspace necessary");

  const auto nullspaceNew = std::make_shared<Core::LinAlg::MultiVector<double>>(newmap, ndim, true);

  for (int i = 0; i < ndim; i++)
  {
    auto& nullspaceData = nullspace->get_vector(i);
    auto& nullspaceDataNew = nullspaceNew->get_vector(i);
    const int myLength = nullspaceDataNew.local_length();

    for (int j = 0; j < myLength; j++)
    {
      const int newmap_gid = newmap.gid(j);
      const int nullspace_lid = nullspace->get_map().lid(newmap_gid);
      if (nullspace_lid == -1) continue;
      nullspaceDataNew.get_values()[j] = nullspaceData.local_values_as_span()[nullspace_lid];
    }
  }

  params.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspaceNew);
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::Parameters::fix_coordinates(
    const std::string& field, const Core::LinAlg::Map& newmap, Teuchos::ParameterList& solveparams)
{
  if (Core::Communication::my_mpi_rank(newmap.get_comm()) == 0)
  {
    std::cout << "Fixing " << field << " Coordinates\n";
  }

  // find the Teko or MueLu list
  Teuchos::ParameterList* params_ptr = nullptr;
  if (solveparams.isSublist("MueLu Parameters"))
  {
    params_ptr = &(solveparams.sublist("MueLu Parameters"));
  }
  else
  {
    params_ptr = &(solveparams);
  }
  Teuchos::ParameterList& params = *params_ptr;

  const auto coordinates =
      params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates", nullptr);
  if (coordinates == nullptr) FOUR_C_THROW("List does not contain coordinates");

  const int number_of_equations = params.get<int>("PDE equations", -1);
  FOUR_C_ASSERT_ALWAYS(number_of_equations > 0,
      "Number of equations per node (\"PDE equations\") must be positive, but is {}",
      number_of_equations);

  const int num_local_dofs = newmap.num_my_elements();
  const bool divisible = (num_local_dofs % number_of_equations == 0);

  // If the local number of dofs is not a multiple of the number of equations on any rank, MueLu
  // itself will throw when deriving the nodal map from the block row map ("block size
  // incompatible with the number of local dofs"). Nothing sensible to do here, so make all
  // ranks return consistently.
  const int any_rank_not_divisible =
      Core::Communication::max_all(divisible ? 0 : 1, newmap.get_comm());
  if (any_rank_not_divisible != 0) return;

  // Derive the nodal map exactly as MueLu does in ReplaceCoordinateMap: take every number-of-
  // equations-th dof of the (local) block row map and collapse it onto a nodal gid. This only
  // matches the actual node numbering if the dofs of a node are numbered contiguously in the
  // global dof map, which is the same assumption MueLu relies on.
  const int num_local_nodes = num_local_dofs / number_of_equations;
  const int index_base = newmap.index_base();
  std::vector<int> node_gids(num_local_nodes);
  for (int k = 0; k < num_local_nodes; ++k)
    node_gids[k] =
        (newmap.gid(k * number_of_equations) - index_base) / number_of_equations + index_base;

  // The global number of elements is not known a priori, let the map derive it (-1).
  constexpr int invalid_global_size = -1;
  Core::LinAlg::Map node_map(
      invalid_global_size, num_local_nodes, node_gids.data(), index_base, newmap.get_comm());

  // Rebuild the coordinates if the maps do not match.
  if (node_map.same_as(coordinates->get_map())) return;

  const auto coordinates_new = std::make_shared<Core::LinAlg::MultiVector<double>>(
      node_map, coordinates->num_vectors(), true);

  // Import the nodal values (matched by global id) from the original coordinates.
  const Core::LinAlg::Import importer(node_map, coordinates->get_map());
  coordinates_new->import(*coordinates, importer, Core::LinAlg::CombineMode::insert);

  params.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates", coordinates_new);
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
Core::LinearSolver::Parameters::extract_nullspace_from_parameterlist(
    const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>& row_map,
    const Teuchos::ParameterList& list)
{
  auto nullspace_data = list.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace");
  if (!nullspace_data) FOUR_C_THROW("Nullspace data is null.");

  Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> nullspace =
      Teuchos::make_rcp<Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>>(
          Teuchos::rcpFromRef(nullspace_data->get_epetra_multi_vector()));

  nullspace->replaceMap(Teuchos::rcpFromRef(row_map));

  return nullspace;
}

FOUR_C_NAMESPACE_CLOSE
