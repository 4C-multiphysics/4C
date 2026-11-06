// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_TRAIT_PLASTIC_HPP
#define FOUR_C_MAT_TRAIT_PLASTIC_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat::Trait
{
  /*!
   * @brief Interface of plastic materials to their equivalent plastic strain, e.g., to drive damage
   */
  class Plastic
  {
   public:
    virtual ~Plastic() = default;

    //! Return the equivalent plastic strain of the last evaluation at Gauss point @p gp
    [[nodiscard]] virtual double equivalent_plastic_strain(int gp) const = 0;

    //! Return the derivative of the equivalent plastic strain of the last evaluation w.r.t. the
    //! Green-Lagrange strain at Gauss point @p gp
    [[nodiscard]] virtual Core::LinAlg::SymmetricTensor<double, 3, 3>
    d_equivalent_plastic_strain_d_glstrain(int gp) const = 0;
  };
}  // namespace Mat::Trait

FOUR_C_NAMESPACE_CLOSE

#endif
