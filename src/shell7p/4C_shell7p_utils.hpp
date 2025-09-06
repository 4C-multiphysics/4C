// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL7P_UTILS_HPP
#define FOUR_C_SHELL7P_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Solid::Elements
{
  /*!
   * @brief A struct holding the number of EAS parameters for each locking type
   *
   *  To alleviate locking the following specific strain components are enhanced:
   *  membrane :  E_{11}, E_{12}, E_{22} constant
   *  bending : E_{11}, E_{12}, E_{22} linear
   *  thickness / curvature : E_{33} linear
   *  transverse shear strain :  E_{13}, E_{23} constant or linear
   */
  struct ShellLockingTypes
  {
    int membrane;
    int bending;
    int thickness;
    int transverse_shear_strain_const;
    int transverse_shear_strain_lin;
    int total;
  };

  /*!
   * @brief A struct holding the thickness, SDC scaling factor and number of collocation points
   * within the ANS method
   */
  struct ShellData
  {
    double sdc;
    double thickness;
    int num_ans;
  };
}  // namespace Solid::Elements

namespace Discret::Elements::Shell::Internal
{
  template <Core::FE::CellType distype>
  inline static constexpr int num_node = Core::FE::num_nodes(distype);
  inline static constexpr int num_dim = 3;
  inline static constexpr int node_dof = 6;
  inline static constexpr int num_internal_variables = 12;

  template <Core::FE::CellType distype>
  inline static constexpr int numdofperelement = num_node<distype> * node_dof;
}  // namespace Discret::Elements::Shell::Internal

namespace Solid::Utils::Shell
{
  /*!
   * \brief Helper function for the nodal nullspace of Shell elements in 3D
   *
   * This computes the contribution of the given node to the global nullspace vector and will be
   * used to fill one "row" of the global nullspace MultiVector.
   *
   * The rigid body modes for structures are:
   *
   *              | xtrans | ytrans | ztrans |  xrot |  yrot | zrot
   *    ----------------------------------------------------------
   *       x     |    1   |    0   |    0   |  0    |  z-z0 | -y+y0
   *       y     |    0   |    1   |    0   | -z+z0 |  0    |  x-x0
   *       z     |    0   |    0   |    1   |  y-y0 | -x+x0 |  0
   *       dx    |    0   |    0   |    0   |  0    |  a3   | -a2
   *       dy    |    0   |    0   |    0   | -a3   |  0    |  a1
   *       dz    |    0   |    0   |    0   |  a2   | -a1   |  0
   *
   *  @param node (in): node to calculate the nullspace on
   *  @param x0 (in): pre-computed geometric center of gravity of the discretization to be used
   * as center of rotation for the rotational modes of the nullspace
   *  @return Core::LinAlg::SerialDenseMatrix: Translational (x,y,z) and
   *  rotational (around x,y,z) nullspace contribution for given node
   */
  Core::LinAlg::SerialDenseMatrix compute_shell_null_space(
      Core::Nodes::Node& node, const double* x0, const Core::LinAlg::Matrix<3, 1>& dir);

  void nodal_block_information_shell(Core::Elements::Element* dwele, int& numdf, int& dimns);

  void lump_mass_matrix(Core::LinAlg::SerialDenseMatrix& mass_matrix);

  namespace Director
  {
    /*!
     * @brief A setup routine for the nodal director after the whole input is read and before
     * the evaluation.
     *
     * @param eletype (in) : Reference to the element type
     * @param dis (in) : Reference to the discretization
     */
    void setup_shell_element_directors(
        const Core::Elements::ElementType& eletype, Core::FE::Discretization& dis);

    /*!
     * @brief Sets the nodal directors for one element
     *
     * @param ele (in) : reference to the element
     * @param nodal_directors (in/out) : Nodal directors of one element
     */
    void setup_director_for_element(
        const Core::Elements::Element& ele, Core::LinAlg::SerialDenseMatrix& nodal_directors);

    /*!
     * @brief Evaluates the average director for one node depending on the neighboring nodes
     *
     * @param dir_list (in) : Nodal directors of neighboring nodes
     * @param num_directors (in) : Number of directors to be considered
     * @param nodal_director (in/out) : Nodal director
     */
    void average_director(const Core::LinAlg::Matrix<3, 8>& dir_list, const int num_directors,
        Core::LinAlg::Matrix<3, 1>& nodal_director);

    /*!
     * @brief Evaluates average directors at all nodes
     *
     * @param eletype (in) : Reference to the element type
     * @param dis (in) : Reference to the discretization
     * @param director_map (in/out) : Nodal director map
     */
    void average_directors_at_nodes(const Core::Elements::ElementType& eletype,
        const Core::FE::Discretization& dis, std::map<int, std::vector<double>>& director_map);

    /*!
     * @brief Export map of nodal directors from nodal row map to nodal column map
     *
     * @param eletype (in) : Reference to the element type
     * @param dis (in) : Reference to the discretization
     * @param director_map (in) : Nodal director map
     */
    void export_director_map_from_row_to_col_map(const Core::Elements::ElementType& eletype,
        Core::FE::Discretization& dis, std::map<int, std::vector<double>>& director_map);

  }  // namespace Director

  namespace ReadElement
  {
    int read_and_set_element_material(const Core::IO::InputParameterContainer& container);

    void read_and_set_locking_types(const Core::FE::CellType& distype,
        const Core::IO::InputParameterContainer& container,
        Solid::Elements::ShellLockingTypes& locking_types);

    int read_and_set_num_ans(const Core::FE::CellType& distype);

  }  // namespace ReadElement

}  // namespace Solid::Utils::Shell

FOUR_C_NAMESPACE_CLOSE

#endif
