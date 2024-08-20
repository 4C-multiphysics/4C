/*-----------------------------------------------------------------------*/
/*! \file
\brief A set of utility functions for mortar methods

\level 1

*/
/*-----------------------------------------------------------------------*/
#ifndef FOUR_C_MORTAR_UTILS_HPP
#define FOUR_C_MORTAR_UTILS_HPP

#include "4C_config.hpp"

#include "4C_mortar_coupling3d_classes.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg

namespace Mortar
{

  /*!
  \brief Sort vector in ascending order

  This routine is taken from Trilinos MOERTEL package.

  \param dlist (in): vector to be sorted (unsorted on input, sorted on output)
  \param N (in):     length of vector to be sorted
  \param list2 (in): another vector which is sorted accordingly
  */
  void sort(double* dlist, int N, int* list2);

  /*!
  \brief Transform the row map of a matrix (only GIDs)

  This method changes the row map of an input matrix to a new row map
  with different GID numbering. However, the parallel distribution of
  this new row map is exactly the same as in the old row map. Thus, this
  is simply a processor-local 1:1 matching of old and new GIDs.

  @param inmat Matrix on which the row and column maps will be transformed
  @param newrowmap New row map used for the given input matrix

  \post Output matrix will be fill_complete()
  */
  Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix_row_transform_gids(
      Teuchos::RCP<const Core::LinAlg::SparseMatrix> inmat,
      Teuchos::RCP<const Epetra_Map> newrowmap);

  /*!
  \brief Transform the column map of a matrix (only GIDs)

  This method changes the column map of an input matrix to a new column
  map with different GID numbering (and the domain map, accordingly).
  However, the parallel distribution of the new domain map is exactly
  the same as in the old domain map. Thus, this is simply a processor-local
  1:1 matching of old and new GIDs.

  @param inmat Matrix on which the row and column maps will be transformed
  @param newdomainmap New domain map used for the given input matrix, which will indirectly
  transform the column map of the given input matrix

  \post Output matrix will be fill_complete()
  */
  Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix_col_transform_gids(
      Teuchos::RCP<const Core::LinAlg::SparseMatrix> inmat,
      Teuchos::RCP<const Epetra_Map> newdomainmap);

  /*! \brief Replace the column and domain map of a filled matrix
   *
   *  This method changes the domain map of an input matrix to a new domain
   *  map with different GID numbering (and the corresponding column map, accordingly).
   *  However, the parallel distribution of the new domain map is exactly
   *  the same as in the old domain map. Thus, this is simply a processor-local
   *  1:1 matching of old and new GIDs. But the creation of the column map can
   *  afford some effort. This effort can be avoided, if a suitable column map
   *  is provided by the user (e.g. if it doesn't change during one step to another).
   *
   *  \param mat           (in)     : Filled matrix, which has already a column and domain map
   *  \param newdomainmap  (in)     : new domain map which is supposed to replace the
   *                                  old domain map
   *  \param newcolmap_ptr (in/out) : This is optional and can be used to store the newly
   *                                  created column map for later use or to provide an already
   *                                  capable column map. In the latter case some communication
   *                                  effort can be avoided.
   *
   *  \author hiermeier \date 04/17  */
  void replace_column_and_domain_map(Core::LinAlg::SparseMatrix& mat,
      const Epetra_Map& newdomainmap, Teuchos::RCP<Epetra_Map>* const newcolmap_ptr = nullptr);

  /*! \brief Create a new column map
   *
   *  \param mat          (in) : Filled matrix, which has already a column and domain map
   *  \param newdomainmap (in) : new domain map which is going to replace the old column map of
   *                             the matrix %mat.
   *  \param newcolmap    (out): new column map
   *
   *  \author hiermeier \date 04/17 */
  void create_new_col_map(const Core::LinAlg::SparseMatrix& mat, const Epetra_Map& newdomainmap,
      Teuchos::RCP<Epetra_Map>& newcolmap);

  /*!
  \brief Transform the row and column maps of a matrix (only GIDs)

  This method changes the row and column maps of an input matrix to new
  row and column maps with different GID numbering (and the domain map,
  accordingly). However, the parallel distribution of the new row and
  domain maps is exactly the same as in the old ones. Thus, this is simply
  a processor-local 1:1 matching of old and new GIDs.

  @param inmat Matrix on which the row and column maps will be transformed
  @param newrowmap New row map used for the given input matrix
  @param newdomainmap New domain map used for the given input matrix, which will indirectly
  transform the column map of the given input matrix

  \post Output matrix will be fill_complete()
  */
  Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix_row_col_transform_gids(
      Teuchos::RCP<const Core::LinAlg::SparseMatrix> inmat,
      Teuchos::RCP<const Epetra_Map> newrowmap, Teuchos::RCP<const Epetra_Map> newdomainmap);

  /*!
  \brief Transform the row map of a matrix (parallel distribution)

  This method changes the row map of an input matrix to a new
  row map with identical GIDs but different parallel distribution.

  */
  Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix_row_transform(
      Teuchos::RCP<const Core::LinAlg::SparseMatrix> inmat,
      Teuchos::RCP<const Epetra_Map> newrowmap);

  /*!
  \brief Transform the column map of a matrix (parallel distribution)

  This method changes the column map of an input matrix to a new
  column map with identical GIDs but different parallel distribution
  (and the domain map, accordingly).

  */
  Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix_col_transform(
      Teuchos::RCP<const Core::LinAlg::SparseMatrix> inmat,
      Teuchos::RCP<const Epetra_Map> newdomainmap);

  /*!
  \brief Transform the row and column maps of a matrix (parallel distribution)

  This method changes the row and column maps of an input matrix
  to new row and column maps with identical GIDs but different
  parallel distribution (and the domain map, accordingly).

  */
  Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix_row_col_transform(
      Teuchos::RCP<const Core::LinAlg::SparseMatrix> inmat,
      Teuchos::RCP<const Epetra_Map> newrowmap, Teuchos::RCP<const Epetra_Map> newdomainmap);

  /*!
  \brief Parallel redistribution of a sparse matrix

  Helper method for the MatrixTransform() methods above.

  */
  Teuchos::RCP<Epetra_CrsMatrix> redistribute(const Core::LinAlg::SparseMatrix& src,
      const Epetra_Map& permrowmap, const Epetra_Map& permdomainmap);

  /*!
  \brief Convex hull points are sorted in order to obtain final clip polygon

  \param out (in): bool to switch output on/off
  \param transformed (in): coordinates of vertex objects transformed into auxiliary plane
  \param collconvexhull (in): vector of vertex objects to be sorted
  \param respoly (out): vector of vertex objects for result polygon
  \param tol (in): clipping tolerance for close vertices detection
  \return number of removed points from collconvexhull

  */
  int sort_convex_hull_points(bool out, Core::LinAlg::SerialDenseMatrix& transformed,
      std::vector<Vertex>& collconvexhull, std::vector<Vertex>& respoly, double& tol);

  namespace UTILS
  {
    /*!
    \brief copy the ghosting of dis_src to all discretizations with names in
           vector voldis. Material pointers can be added according to
           link_materials
    */
    void create_volume_ghosting(const Core::FE::Discretization& dis_src,
        const std::vector<Teuchos::RCP<Core::FE::Discretization>>& voldis,
        std::vector<std::pair<int, int>> material_links, bool check_on_in = true,
        bool check_on_exit = true);


    /*!
    \brief Prepare mortar element for nurbs case


    store knot vector, zerosized information and normal factor
    */
    void prepare_nurbs_element(Core::FE::Discretization& discret,
        Teuchos::RCP<Core::Elements::Element> ele, Teuchos::RCP<Mortar::Element> cele, int dim);

    /*!
    \brief Prepare mortar node for nurbs case

    store control point weight

    */
    void prepare_nurbs_node(Core::Nodes::Node* node, Teuchos::RCP<Mortar::Node> mnode);

    void mortar_matrix_condensation(Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase>& k,
        const std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>>& p);

    /*! \brief Perform static condensation of Jacobian with mortar matrix \f$D^{-1}M\f$
     *
     * @param[in/out] k Matrix to be condensed
     * @param[in] p_row Mortar projection operator for condensation of rows
     * @param[in] p_col Mortar projection operator for condenstaion of columns
     */
    void mortar_matrix_condensation(Teuchos::RCP<Core::LinAlg::SparseMatrix>& k,
        const Teuchos::RCP<const Core::LinAlg::SparseMatrix>& p_row,
        const Teuchos::RCP<const Core::LinAlg::SparseMatrix>& p_col);

    void mortar_rhs_condensation(
        Teuchos::RCP<Epetra_Vector>& rhs, const Teuchos::RCP<Core::LinAlg::SparseMatrix>& p);

    void mortar_rhs_condensation(Teuchos::RCP<Epetra_Vector>& rhs,
        const std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>>& p);

    void mortar_recover(
        Teuchos::RCP<Epetra_Vector>& inc, const Teuchos::RCP<Core::LinAlg::SparseMatrix>& p);

    void mortar_recover(Teuchos::RCP<Epetra_Vector>& inc,
        const std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>>& p);
  }  // namespace UTILS
}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
