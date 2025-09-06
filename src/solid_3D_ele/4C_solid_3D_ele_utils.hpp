// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_UTILS_HPP
#define FOUR_C_SOLID_3D_ELE_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_solid_3D_ele_properties.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid::Utils
{
  void nodal_block_information_solid(Core::Elements::Element* dwele, int& numdf, int& dimns);

  /*!
   * @brief Converts the 2nd Piola-Kirchhoff stress tensor to the
   * Cauchy stress tensor
   *
   * @param pk2 (in) : 2nd Piola-Kirchhoff stress tensor
   * @param defgrd (in) : Deformation gradient
   * @return Cauchy stress tensor
   */
  Core::LinAlg::SymmetricTensor<double, 3, 3> pk2_to_cauchy(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& pk2,
      const Core::LinAlg::Tensor<double, 3, 3>& defgrd);

  /*!
   * @brief Convert Green Lagrange strain tensor to Euler-Almansi
   * strain tensor.
   *
   * @param gl (in) : Green Lagrange strain tensor
   * @return Core::LinAlg::Matrix<6, 1> : Euler-Almansi strain tensor
   */
  Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_to_euler_almansi(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl,
      const Core::LinAlg::Tensor<double, 3, 3>& defgrd);

  /*!
   * @brief Convert Green Lagrange strain tensor in strain to Lograithmic strain
   * tensor.
   *
   * @param gl (in) : Green Lagrange strain tensor
   * @return Core::LinAlg::Matrix<6, 1> : Logarithmic strain tensor
   */
  Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_to_log_strain(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl);

  namespace ReadElement
  {
    int read_element_material(const Core::IO::InputParameterContainer& container);

    Discret::Elements::SolidElementProperties read_solid_element_properties(
        const Core::IO::InputParameterContainer& container);

  }  // namespace ReadElement

}  // namespace Solid::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
