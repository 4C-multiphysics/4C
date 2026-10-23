// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_UTILS_HPP
#define FOUR_C_MORTAR_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mortar_coupling3d_classes.hpp"

#include <memory>
#include <vector>

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
  \brief Compute the convex hull of a set of points in 2D

  Corners of the hull that form almost straight lines with their neighbors are removed.

  \param coordinates (in): coordinates of the points. First index is the coordinate direction,
  second index is the point number
  \param clipping_tolerance (in): tolerance used for removing almost straight corners
  \return indices of the points on the hull in clockwise order (less than three if the hull
  degenerates to a line or a point)
  */
  std::vector<int> sort_convex_hull_points(
      const Core::LinAlg::SerialDenseMatrix& coordinates, double clipping_tolerance);

  namespace Utils
  {
    /*!
    \brief copy the ghosting of dis_src to all discretizations with names in
           vector voldis. Material pointers can be added according to
           link_materials
    */
    void create_volume_ghosting(const Core::FE::Discretization& dis_src,
        const std::vector<std::shared_ptr<Core::FE::Discretization>>& voldis,
        std::vector<std::pair<int, int>> material_links, bool check_on_in = true,
        bool check_on_exit = true);


    /*!
    \brief Prepare mortar element for nurbs case


    store knot vector, zerosized information and normal factor
    */
    void prepare_nurbs_element(Core::FE::Discretization& discret,
        std::shared_ptr<Core::Elements::Element> ele, Mortar::Element& cele, int dim);

    /*!
    \brief Prepare mortar node for nurbs case

    store control point weight

    */
    void prepare_nurbs_node(Core::Nodes::Node* node, Mortar::Node& mnode);

    void mortar_matrix_condensation(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>& k,
        const std::vector<std::shared_ptr<Core::LinAlg::SparseMatrix>>& p);

    /*! \brief Perform static condensation of Jacobian with mortar matrix \f$D^{-1}M\f$
     *
     * @param[in/out] k Matrix to be condensed
     * @param[in] p_row Mortar projection operator for condensation of rows
     * @param[in] p_col Mortar projection operator for condenstaion of columns
     */
    void mortar_matrix_condensation(std::shared_ptr<Core::LinAlg::SparseMatrix>& k,
        const std::shared_ptr<const Core::LinAlg::SparseMatrix>& p_row,
        const std::shared_ptr<const Core::LinAlg::SparseMatrix>& p_col);

    void mortar_rhs_condensation(Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseMatrix& p);

    void mortar_rhs_condensation(Core::LinAlg::Vector<double>& rhs,
        const std::vector<std::shared_ptr<Core::LinAlg::SparseMatrix>>& p);

    void mortar_recover(Core::LinAlg::Vector<double>& inc, Core::LinAlg::SparseMatrix& p);

    void mortar_recover(Core::LinAlg::Vector<double>& inc,
        const std::vector<std::shared_ptr<Core::LinAlg::SparseMatrix>>& p);
  }  // namespace Utils
}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
