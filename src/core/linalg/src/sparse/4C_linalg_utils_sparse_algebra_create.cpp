// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_utils_exceptions.hpp"

#include <MueLu_AmalgamationFactory.hpp>
#include <MueLu_CoalesceDropFactory.hpp>
#include <MueLu_CoarseMapFactory.hpp>
#include <MueLu_FactoryManager.hpp>
#include <MueLu_TentativePFactory.hpp>
#include <MueLu_UncoupledAggregationFactory.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <MueLu_Utilities.hpp>

FOUR_C_NAMESPACE_OPEN

using SC = Scalar;
using LO = LocalOrdinal;
using GO = GlobalOrdinal;
using NO = Node;

/*----------------------------------------------------------------------*
 |  create a Epetra_CrsMatrix                                mwgee 12/06|
 *----------------------------------------------------------------------*/
std::shared_ptr<Epetra_CrsMatrix> Core::LinAlg::create_matrix(
    const Epetra_Map& rowmap, const int npr)
{
  if (!rowmap.UniqueGIDs()) FOUR_C_THROW("Row map is not unique");
  return std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, npr, false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::create_identity_matrix(
    const Epetra_Map& map)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> eye = std::make_shared<SparseMatrix>(map, 1);
  fill_identity_matrix(*eye);
  return eye;
}

void Core::LinAlg::fill_identity_matrix(Core::LinAlg::SparseMatrix& mat)
{
  int numelements = mat.row_map().NumMyElements();
  int* gids = mat.row_map().MyGlobalElements();

  for (int i = 0; i < numelements; ++i)
  {
    int gid = gids[i];
    mat.assemble(1., gid, gid);
  }
  mat.complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix Core::LinAlg::create_interpolation_matrix(const SparseMatrix& matrix,
    const Core::LinAlg::MultiVector<double>& nullspace, Teuchos::ParameterList& params)
{
  using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GO, NO>;
  using EpetraMultiVector = Xpetra::EpetraMultiVectorT<GO, NO>;

  const Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> xpetra_nullspace =
      Teuchos::rcp(new EpetraMultiVector(
          Teuchos::rcp(new Epetra_MultiVector(*nullspace.get_ptr_of_Epetra_MultiVector()))));
  const int number_of_equations = params.get<int>("PDE equations");

  Teuchos::RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> mueluA =
      Teuchos::rcp(new EpetraCrsMatrix(Teuchos::rcpFromRef(*matrix.epetra_matrix())));
  Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> xpetra_matrix =
      Teuchos::rcp(new Xpetra::CrsMatrixWrap<SC, LO, GO, NO>(mueluA));
  xpetra_matrix->SetFixedBlockSize(number_of_equations);

  MueLu::Level fineLevel, coarseLevel;
  Teuchos::RCP<MueLu::FactoryManager<SC, LO, GO, NO>> factoryHandler =
      rcp(new MueLu::FactoryManager());
  factoryHandler->SetKokkosRefactor(false);
  fineLevel.SetFactoryManager(factoryHandler);
  coarseLevel.SetFactoryManager(factoryHandler);

  coarseLevel.SetPreviousLevel(rcpFromRef(fineLevel));

  fineLevel.SetLevelID(0);
  coarseLevel.SetLevelID(1);
  fineLevel.SetFactoryManager(Teuchos::null);
  coarseLevel.SetFactoryManager(Teuchos::null);

  fineLevel.Set("A", xpetra_matrix);
  fineLevel.Set("Nullspace", xpetra_nullspace);
  fineLevel.Set("DofsPerNode", number_of_equations);
  fineLevel.Set("Verbosity", Teuchos::VERB_EXTREME);

  Teuchos::RCP<MueLu::AmalgamationFactory<SC, LO, GO, NO>> amalgFact =
      rcp(new MueLu::AmalgamationFactory());
  Teuchos::RCP<MueLu::CoalesceDropFactory<SC, LO, GO, NO>> dropFact =
      rcp(new MueLu::CoalesceDropFactory());
  dropFact->SetFactory("UnAmalgamationInfo", amalgFact);
  dropFact->SetParameter("aggregation: use ml scaling of drop tol", Teuchos::ParameterEntry(true));

  Teuchos::RCP<MueLu::UncoupledAggregationFactory<LO, GO, NO>> UncoupledAggFact =
      rcp(new MueLu::UncoupledAggregationFactory());
  UncoupledAggFact->SetFactory("Graph", dropFact);
  UncoupledAggFact->SetParameter(
      "aggregation: preserve Dirichlet points", Teuchos::ParameterEntry(true));
  UncoupledAggFact->SetParameter("aggregation: match ML phase1", Teuchos::ParameterEntry(true));
  UncoupledAggFact->SetParameter("aggregation: match ML phase2a", Teuchos::ParameterEntry(true));
  UncoupledAggFact->SetParameter("aggregation: match ML phase2b", Teuchos::ParameterEntry(true));

  Teuchos::RCP<MueLu::CoarseMapFactory<SC, LO, GO, NO>> coarseMapFact =
      rcp(new MueLu::CoarseMapFactory());
  coarseMapFact->SetFactory("Aggregates", UncoupledAggFact);

  Teuchos::RCP<MueLu::TentativePFactory<SC, LO, GO, NO>> TentativePFact =
      rcp(new MueLu::TentativePFactory());
  TentativePFact->SetFactory("Aggregates", UncoupledAggFact);
  TentativePFact->SetFactory("UnAmalgamationInfo", amalgFact);
  TentativePFact->SetFactory("CoarseMap", coarseMapFact);

  coarseLevel.Request("P", TentativePFact.get());
  coarseLevel.Request(*TentativePFact);
  TentativePFact->Build(fineLevel, coarseLevel);

  Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> Ptent;
  coarseLevel.Get("P", Ptent, TentativePFact.get());

  auto prolongation_operator = MueLu::Utilities<SC, LO, GO, NO>::Op2NonConstEpetraCrs(Ptent);

  Core::LinAlg::SparseMatrix prolongator(
      std::make_shared<Epetra_CrsMatrix>(*prolongation_operator), Core::LinAlg::View);

  return prolongator;
}


/*----------------------------------------------------------------------*
 |  create a Core::LinAlg::Vector<double>                                   mwgee 12/06|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Core::LinAlg::create_vector(
    const Epetra_BlockMap& rowmap, const bool init)
{
  return std::make_shared<Core::LinAlg::Vector<double>>(rowmap, init);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::LinAlg::create_multi_vector(
    const Epetra_BlockMap& rowmap, const int numrows, const bool init)
{
  return std::make_shared<Core::LinAlg::MultiVector<double>>(rowmap, numrows, init);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Epetra_Map> Core::LinAlg::create_map(const std::set<int>& gids, MPI_Comm comm)
{
  std::vector<int> mapvec;
  mapvec.reserve(gids.size());
  mapvec.assign(gids.begin(), gids.end());
  std::shared_ptr<Epetra_Map> map = std::make_shared<Epetra_Map>(
      -1, mapvec.size(), mapvec.data(), 0, Core::Communication::as_epetra_comm(comm));
  mapvec.clear();
  return map;
}

/*----------------------------------------------------------------------*
 | create epetra_map with out-of-bound check                 farah 06/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<Epetra_Map> Core::LinAlg::create_map(const std::vector<int>& gids, MPI_Comm comm)
{
  std::shared_ptr<Epetra_Map> map;

  if ((int)gids.size() > 0)
    map = std::make_shared<Epetra_Map>(
        -1, gids.size(), gids.data(), 0, Core::Communication::as_epetra_comm(comm));
  else
    map = std::make_shared<Epetra_Map>(
        -1, gids.size(), nullptr, 0, Core::Communication::as_epetra_comm(comm));

  return map;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::create_map_extractor_from_discretization(
    const Core::FE::Discretization& dis, int ndim, Core::LinAlg::MultiMapExtractor& extractor)
{
  std::set<int> conddofset;
  std::set<int> otherdofset;

  int numrownodes = dis.num_my_row_nodes();
  for (int i = 0; i < numrownodes; ++i)
  {
    Core::Nodes::Node* node = dis.l_row_node(i);

    std::vector<int> dof = dis.dof(0, node);
    for (unsigned j = 0; j < dof.size(); ++j)
    {
      // test for dof position
      if (j != static_cast<unsigned>(ndim))
      {
        otherdofset.insert(dof[j]);
      }
      else
      {
        conddofset.insert(dof[j]);
      }
    }
  }

  std::vector<int> conddofmapvec;
  conddofmapvec.reserve(conddofset.size());
  conddofmapvec.assign(conddofset.begin(), conddofset.end());
  conddofset.clear();
  std::shared_ptr<Epetra_Map> conddofmap = std::make_shared<Epetra_Map>(-1, conddofmapvec.size(),
      conddofmapvec.data(), 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  conddofmapvec.clear();

  std::vector<int> otherdofmapvec;
  otherdofmapvec.reserve(otherdofset.size());
  otherdofmapvec.assign(otherdofset.begin(), otherdofset.end());
  otherdofset.clear();
  std::shared_ptr<Epetra_Map> otherdofmap = std::make_shared<Epetra_Map>(-1, otherdofmapvec.size(),
      otherdofmapvec.data(), 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  otherdofmapvec.clear();

  std::vector<std::shared_ptr<const Epetra_Map>> maps(2);
  maps[0] = otherdofmap;
  maps[1] = conddofmap;
  extractor.setup(*dis.dof_row_map(), maps);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::create_map_extractor_from_discretization(const Core::FE::Discretization& dis,
    const Core::DOFSets::DofSetInterface& dofset, int ndim, Core::LinAlg::MapExtractor& extractor)
{
  std::set<int> conddofset;
  std::set<int> otherdofset;

  int numrownodes = dis.num_my_row_nodes();
  for (int i = 0; i < numrownodes; ++i)
  {
    Core::Nodes::Node* node = dis.l_row_node(i);

    std::vector<int> dof = dofset.dof(node);
    for (unsigned j = 0; j < dof.size(); ++j)
    {
      // test for dof position
      if (j < static_cast<unsigned>(ndim))
      {
        otherdofset.insert(dof[j]);
      }
      else
      {
        conddofset.insert(dof[j]);
      }
    }
  }

  std::vector<int> conddofmapvec;
  conddofmapvec.reserve(conddofset.size());
  conddofmapvec.assign(conddofset.begin(), conddofset.end());
  conddofset.clear();
  std::shared_ptr<Epetra_Map> conddofmap = std::make_shared<Epetra_Map>(-1, conddofmapvec.size(),
      conddofmapvec.data(), 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  conddofmapvec.clear();

  std::vector<int> otherdofmapvec;
  otherdofmapvec.reserve(otherdofset.size());
  otherdofmapvec.assign(otherdofset.begin(), otherdofset.end());
  otherdofset.clear();
  std::shared_ptr<Epetra_Map> otherdofmap = std::make_shared<Epetra_Map>(-1, otherdofmapvec.size(),
      otherdofmapvec.data(), 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  otherdofmapvec.clear();

  extractor.setup(*dofset.dof_row_map(), conddofmap, otherdofmap);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::create_map_extractor_from_discretization(const Core::FE::Discretization& dis,
    int ndim_field1, int ndim_field2, Core::LinAlg::MultiMapExtractor& extractor)
{
  unsigned fp_dim = static_cast<unsigned>(ndim_field1 + ndim_field2);

  std::set<int> conddofset;
  std::set<int> otherdofset;

  int numrownodes = dis.num_my_row_nodes();
  for (int i = 0; i < numrownodes; ++i)
  {
    Core::Nodes::Node* node = dis.l_row_node(i);

    std::vector<int> dof = dis.dof(0, node);

    if ((dof.size() % fp_dim) != 0)
      FOUR_C_THROW(
          "Vector-Scalar-Split is not unique! Mismatch between number of dofs and vector/scalar "
          "dim");

    for (unsigned j = 0; j < dof.size(); ++j)
    {
      // test for dof position
      if (j % fp_dim < static_cast<unsigned>(ndim_field1))
      {
        otherdofset.insert(dof[j]);
      }
      else
      {
        conddofset.insert(dof[j]);
      }
    }
  }

  std::vector<int> conddofmapvec;
  conddofmapvec.reserve(conddofset.size());
  conddofmapvec.assign(conddofset.begin(), conddofset.end());
  conddofset.clear();
  std::shared_ptr<Epetra_Map> conddofmap = std::make_shared<Epetra_Map>(-1, conddofmapvec.size(),
      conddofmapvec.data(), 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  conddofmapvec.clear();

  std::vector<int> otherdofmapvec;
  otherdofmapvec.reserve(otherdofset.size());
  otherdofmapvec.assign(otherdofset.begin(), otherdofset.end());
  otherdofset.clear();
  std::shared_ptr<Epetra_Map> otherdofmap = std::make_shared<Epetra_Map>(-1, otherdofmapvec.size(),
      otherdofmapvec.data(), 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  otherdofmapvec.clear();

  std::vector<std::shared_ptr<const Epetra_Map>> maps(2);
  maps[0] = otherdofmap;
  maps[1] = conddofmap;
  extractor.setup(*dis.dof_row_map(), maps);
}

FOUR_C_NAMESPACE_CLOSE
