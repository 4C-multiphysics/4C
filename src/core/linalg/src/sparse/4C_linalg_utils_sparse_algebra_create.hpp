/*----------------------------------------------------------------------*/
/*! \file

\brief A collection of algebraic creation methods for namespace Core::LinAlg

\level 0
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_CREATE_HPP
#define FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_CREATE_HPP

#include "4C_config.hpp"

#include "4C_fem_dofset_interface.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_Comm.h>
#include <Epetra_CrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <Epetra_IntVector.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   \brief Create a new Epetra_CrsMatrix and return RefcountPtr to it

   \param rowmap (in): row map of matrix
   \param npr (in): estimated number of entries per row.
   (need not be exact, better should be too big rather then too small)
   */
  Teuchos::RCP<Epetra_CrsMatrix> create_matrix(const Epetra_Map& rowmap, const int npr);

  /*!
   \brief Create a new sparse identity matrix and return RefcountPtr to it

   \param rowmap (in): row map of matrix
   */
  Teuchos::RCP<Core::LinAlg::SparseMatrix> create_identity_matrix(const Epetra_Map& map);

  /*! \brief Create prolongation matrix using an external algebraic multigrid package
   *
   * \param matrix Input matrix (= fine level operator)
   * \param nullspace Set of nullspace vectors
   * \param params Additional configuration parameters for algebraic multigrid setup
   * \return Prolongation matrix
   */
  Core::LinAlg::SparseMatrix create_interpolation_matrix(
      const SparseMatrix& matrix, double* nullspace, Teuchos::ParameterList& params);

  /*!
   \brief Create a new Core::LinAlg::Vector and return RefcountPtr to it

   \param rowmap (in): row map of vector
   \param init (in): initialize vector to zero upon construction
   */
  Teuchos::RCP<Core::LinAlg::Vector> create_vector(
      const Epetra_BlockMap& rowmap, const bool init = true);

  /*!
   \brief Create a new Epetra_MultiVector and return RefcountPtr to it

   \param rowmap (in): row map of vector
   \param rowmap (in): number of vectors
   \param init (in): initialize vector to zero upon construction
   */
  Teuchos::RCP<Epetra_MultiVector> create_multi_vector(
      const Epetra_BlockMap& rowmap, const int numrows, const bool init = true);

  /*!
   \brief Create an Epetra_Map from a set of gids

   This is one of the basic operations that is needed every so often.

   \param gids The local gids of this map
   \param comm The map's communicator
   */
  Teuchos::RCP<Epetra_Map> create_map(const std::set<int>& gids, const Epetra_Comm& comm);

  /*!
   \brief Create an Epetra_Map from a vector of gids

   This is one of the basic operations that is needed every so often.

   \param gids The local gids of this map
   \param comm The map's communicator
   */
  Teuchos::RCP<Epetra_Map> create_map(const std::vector<int>& gids, const Epetra_Comm& comm);

  /*!
      \brief Creates MultiMapExtractor to split dofs at certain position

      We assume that each node possesses ndim dofs of one field and (optionally) remaining dofs of
      another field. The dof row map is thus split into two.

      The ndim dofs are assigned to map 0 (the other map) and the remaining dofs to map 1 (the
      condition map).

      \param dis : (in) discretization
      \param ndim : (in) dimensions of map 0
      \param extractor : (out) ready made map splitter

      \author u.kue
      \date 02/08
     */
  void create_map_extractor_from_discretization(
      const Core::FE::Discretization& dis, int ndim, Core::LinAlg::MultiMapExtractor& extractor);

  /*!
    \brief Creates MapExtractor to split dofs at certain position

    We assume that each node possesses ndim dofs of one field and (optionally) remaining dofs of
    another field. The dof row map is thus split into two.

    The ndim dofs are assigned to map 0 (the other map) and the remaining dofs to map 1 (the
    condition map).

    Nodal dofs and total map come from dofset.

    \param dis : (in) discretization
    \param dofset : (in) Degree of freedom set
    \param ndim : (in) dimensions of map 0
    \param extractor : (out) ready made map splitter

    \author u.kue
    \date 02/08
   */
  void create_map_extractor_from_discretization(const Core::FE::Discretization& dis,
      const Core::DOFSets::DofSetInterface& dofset, int ndim,
      Core::LinAlg::MapExtractor& extractor);


  /*!
    \brief Creates MultiMapExtractor to split dofs at certain position

    We assume that each node possesses ndim_field1 + ndim_field2 dofs. The dof row map is thus
    split into two.

    The ndim_field1 dofs are assigned to map 0 (the other map) and the ndim_field2 to map 1 (the
    condition map).

    \param dis : (in) discretization
    \param ndim_field1 : (in) dimensions of map 0
    \param ndim_field2 : (in) dimensions of map 1
    \param extractor : (out) ready made map splitter

    \author schott
    \date 12/11
   */
  void create_map_extractor_from_discretization(const Core::FE::Discretization& dis,
      int ndim_field1, int ndim_field2, Core::LinAlg::MultiMapExtractor& extractor);

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
