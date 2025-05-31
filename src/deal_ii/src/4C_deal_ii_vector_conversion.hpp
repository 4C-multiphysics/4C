// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_VECTOR_CONVERSION_HPP
#define FOUR_C_DEAL_II_VECTOR_CONVERSION_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"

#include <deal.II/dofs/dof_handler.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace DealiiWrappers
{

  /**
   * @brief Create a Map object that maps from deal.II dofs to 4C gids.
   *
   * Given a dealii::DoFHandler @p dof_handler and Core::FE::Discretization @p discretization,
   * create and return a Map that has the following properties:
   *
   *  - The *global numbering* of dofs in this map corresponds to the dof numbering in @p
   *    discretization.
   *  - The *local ordering* of dofs in this map corresponds to the order of dofs in @p dof_handler.
   *
   * This means, that the Map object can be used to import and export vectors and matrices
   * between 4C and deal.II layout via an intermediate vector, which uses the returned
   * map for its layout. To go from a deal.II to a 4C vector, one first needs to copy the local
   * elements of the deal.II vector to the intermediate vector. Next, one can use the
   * import mechanism to communicate the intermediate vector to the 4C vector. This mechanism
   * is conveniently available through the VectorConverter class.
   *
   * The DoFHandler should be constructed on a dealii::Triangulation obtained from
   * create_triangulation(). As further input, this function requires the @p context that is
   * returned from that same function.
   */
  template <int dim, int spacedim = dim>
  Core::LinAlg::Map create_dealii_to_four_c_map(
      const dealii::DoFHandler<dim, spacedim>& dof_handler,
      const Core::FE::Discretization& discretization, const Context<dim, spacedim>& context);


  /**
   * A helper class to convert between 4C vectors and deal.II vectors.
   *
   * This class only requires a dealii::DoFHandler and Core::FE::Discretization. The DoFHandler
   * should be constructed on a dealii::Triangulation obtained from CreateTriangulation(). This
   * class also requires the Context that is returned from that same function.
   *
   * Since this class needs to setup global maps and vectors, it is best to create an instance once
   * and reuse it for multiple conversions. The two functions to_four_c() and to_dealii() should
   * then be reasonably efficient.
   *
   * @tparam VectorType The type of deal.II vector this class should work with
   * @tparam dim Dimension of the problem
   * @tparam spacedim Spatial dimension of the problem
   */
  template <typename VectorType, int dim, int spacedim = dim>
  class VectorConverter
  {
    static_assert(dim == 3 && spacedim == 3);

   public:
    /**
     * Create a VectorConverter which works between the given @p dof_handler and @p discretization.
     * The @p context carries the usual metadata.
     */
    VectorConverter(const dealii::DoFHandler<dim, spacedim>& dof_handler,
        const Core::FE::Discretization& discretization, const Context<dim, spacedim>& context);

    /**
     * Convert @p dealii_vector to @p four_c_vector.
     *
     * @param four_c_vector A vector in dof_row_map layout. Its content will be overwritten.
     * @param dealii_vector An unghosted deal.II vector whose content will be copied.
     */
    void to_four_c(
        Core::LinAlg::Vector<double>& four_c_vector, const VectorType& dealii_vector) const;

    /**
     * Convert @p four_c_vector to @p dealii_vector.
     *
     * @param dealii_vector An unghosted deal.II vector. Its content will be overwritten.
     * @param four_c_vector A vector in dof_row_map layout whose content will be copied.
     */
    void to_dealii(
        VectorType& dealii_vector, const Core::LinAlg::Vector<double>& four_c_vector) const;

   private:
    /**
     * The Epetra_Map which describes the mapping and communication pattern between deal.II and
     * 4C.
     */
    Core::LinAlg::Map dealii_to_four_c_map_;

    /**
     * Core::LinAlg::Import object which can be reused across function calls.
     */
    Core::LinAlg::Import dealii_to_four_c_importer_;

    /**
     * A temporary vector which uses #dealii_to_four_c_map and receives/writes the local entries
     * from/to a deal.II vector. It can then be communicated to/from 4C layout via
     * #dealii_to_four_c_importer.
     */
    mutable Core::LinAlg::Vector<double> vector_in_dealii_layout_;
  };


}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif
