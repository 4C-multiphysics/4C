// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_CYLINDER_COORDINATE_SYSTEM_PROVIDER_HPP
#define FOUR_C_MAT_ANISOTROPY_CYLINDER_COORDINATE_SYSTEM_PROVIDER_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  /*!
   * \brief Interface of a Cylinder Coordinate System Provider
   */
  class CylinderCoordinateSystemProvider
  {
   public:
    virtual ~CylinderCoordinateSystemProvider() = default;
    /*!
     * \brief Returns the unit vector pointing in radial direction
     *
     * \return unit vector pointing in radial direction
     */
    virtual const Core::LinAlg::Tensor<double, 3>& get_rad() const = 0;

    /*!
     * \brief Returns a reference to the unit vector pointing in axial direction
     *
     * \return const Core::LinAlg::Matrix<3, 1>& Reference to the unit vector pointing in axial
     * direction
     */
    virtual const Core::LinAlg::Tensor<double, 3>& get_axi() const = 0;

    /*!
     * \brief Returns a reference to the unit vector pointing in circumferential direction
     *
     * \return const Core::LinAlg::Matrix<3, 1>& Reference to the unit vector pointing in
     * circumferential direction
     */
    virtual const Core::LinAlg::Tensor<double, 3>& get_cir() const = 0;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
