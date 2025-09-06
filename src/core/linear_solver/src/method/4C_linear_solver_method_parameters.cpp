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

  int numdf = 1;
  int dimns = 1;

  // set parameter information for solver
  {
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
        Core::Nodes::Node* actnode = dis.l_row_node(i);
        std::vector<int> dofs = dis.dof(0, actnode);

        const int localIndex = nullspace_node_map->lid(dofs[0]);

        if (localIndex == -1) continue;

        Core::Elements::Element* dwele = dis.l_row_element(localIndex);
        actnode->elements()[0]->element_type().nodal_block_information(dwele, numdf, dimns);
        break;
      }
    }

    // communicate data to procs without row element
    std::array<int, 2> ldata{numdf, dimns};
    std::array<int, 2> gdata{0, 0};
    Core::Communication::max_all(ldata.data(), gdata.data(), ldata.size(), dis.get_comm());
    numdf = gdata[0];
    dimns = gdata[1];

    // store nullspace information in solver list
    solverlist.set("PDE equations", numdf);
    solverlist.set("null space: dimension", dimns);
    solverlist.set("null space: type", "pre-computed");
    solverlist.set("null space: add default vectors", false);
  }

  // set coordinate information
  {
    std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates;
    if (nullspace_node_map == nullptr)
      coordinates = dis.build_node_coordinates();
    else
      coordinates = dis.build_node_coordinates(nullspace_node_map);

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

    const auto nullspace = Core::FE::compute_null_space(dis, numdf, dimns, *nullspace_dof_map);

    solverlist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspace);
    solverlist.set("null space: vectors", nullspace->Values());
    solverlist.set<bool>("ML validate parameter list", false);
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

  // find the ML or MueLu list
  Teuchos::ParameterList* params_ptr = nullptr;
  if (solveparams.isSublist("ML Parameters"))
    params_ptr = &(solveparams.sublist("ML Parameters"));
  else if (solveparams.isSublist("MueLu Parameters"))
    params_ptr = &(solveparams.sublist("MueLu Parameters"));
  else
    params_ptr = &(solveparams);
  Teuchos::ParameterList& params = *params_ptr;

  const int ndim = params.get("null space: dimension", -1);
  if (ndim == -1) FOUR_C_THROW("List does not contain nullspace dimension");

  std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
      params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullptr);
  if (nullspace == nullptr) FOUR_C_THROW("List does not contain nullspace");

  const int nullspaceLength = nullspace->MyLength();
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
    auto& nullspaceData = (*nullspace)(i);
    auto& nullspaceDataNew = (*nullspaceNew)(i);
    const int myLength = nullspaceDataNew.local_length();

    for (int j = 0; j < myLength; j++)
    {
      int gid = newmap.gid(j);
      int olid = oldmap.lid(gid);
      if (olid == -1) continue;
      nullspaceDataNew.get_values()[j] = nullspaceData[olid];
    }
  }

  params.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspaceNew);
  params.set("null space: vectors", nullspaceNew->Values());
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
Core::LinearSolver::Parameters::extract_nullspace_from_parameterlist(
    const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>& row_map,
    const Teuchos::ParameterList& list)
{
  if (!list.isParameter("null space: dimension"))
    FOUR_C_THROW("Multigrid parameter 'null space: dimension' missing  in solver parameter list.");

  const int nullspace_dimension = list.get<int>("null space: dimension");
  if (nullspace_dimension < 1)
    FOUR_C_THROW("Multigrid parameter 'null space: dimension' wrong. It has to be > 0.");

  auto nullspace_data = list.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace");
  if (!nullspace_data) FOUR_C_THROW("Nullspace data is null.");

  Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> nullspace =
      Teuchos::make_rcp<Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>>(
          Teuchos::rcpFromRef(nullspace_data->get_epetra_multi_vector()));

  nullspace->replaceMap(Teuchos::rcpFromRef(row_map));

  return nullspace;
}

FOUR_C_NAMESPACE_CLOSE
