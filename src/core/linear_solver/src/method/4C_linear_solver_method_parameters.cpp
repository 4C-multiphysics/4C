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
#include "4C_utils_exceptions.hpp"

#include <Xpetra_EpetraIntMultiVector.hpp>

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

    if (nullspace_node_map == nullptr and dis.num_my_row_nodes() > 0)
    {
      // no map given, just grab the block information on the first element that appears
      Core::Elements::Element* dwele = dis.l_row_element(0);
      dwele->element_type().nodal_block_information(dwele, numdf, dimns);
    }
    else
    {
      // if a map is given, grab the block information of the first element in that map
      for (int i = 0; i < dis.num_my_row_nodes(); ++i)
      {
        auto* actnode = dis.l_row_node(i);
        const auto node_gid = actnode->id();
        const int localIndex = nullspace_node_map->lid(node_gid);

        if (localIndex == -1) continue;

        Core::Elements::Element* dwele = dis.l_row_element(localIndex);
        actnode->adjacent_elements()[0].user_element()->element_type().nodal_block_information(
            dwele, numdf, dimns);
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
void Core::LinearSolver::Parameters::fix_null_space(std::string field,
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

  std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
      params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullptr);
  if (nullspace == nullptr) FOUR_C_THROW("List does not contain nullspace");

  const int ndim = nullspace->num_vectors();

  const int nullspaceLength = nullspace->local_length();
  const int newmapLength = newmap.num_my_elements();

  if (nullspaceLength == newmapLength) return;
  if (nullspaceLength != oldmap.num_my_elements())
    FOUR_C_THROW("Nullspace map of length {} does not match old map length of {}", nullspaceLength,
        oldmap.num_my_elements());
  if (newmapLength > nullspaceLength)
    FOUR_C_THROW("New problem size larger than old - full rebuild of nullspace necessary");

  std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspaceNew =
      std::make_shared<Core::LinAlg::MultiVector<double>>(newmap, ndim, true);

  for (int i = 0; i < ndim; i++)
  {
    auto& nullspaceData = nullspace->get_vector(i);
    auto& nullspaceDataNew = nullspaceNew->get_vector(i);
    const int myLength = nullspaceDataNew.local_length();

    for (int j = 0; j < myLength; j++)
    {
      int gid = newmap.gid(j);
      int olid = oldmap.lid(gid);
      if (olid == -1) continue;
      nullspaceDataNew.get_values()[j] = nullspaceData.local_values_as_span()[olid];
    }
  }

  params.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspaceNew);
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
