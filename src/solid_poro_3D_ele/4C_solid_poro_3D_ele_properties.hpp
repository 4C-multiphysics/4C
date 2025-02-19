// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_PORO_3D_ELE_PROPERTIES_HPP
#define FOUR_C_SOLID_PORO_3D_ELE_PROPERTIES_HPP

#include "4C_config.hpp"

#include "4C_inpar_poro.hpp"
#include "4C_inpar_scatra.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN


namespace Discret::Elements
{
  /*!
   *  @brief struct for managing solidporo element properties
   */
  struct SolidPoroElementProperties
  {
    //! scalar transport implementation type (physics)
    Inpar::ScaTra::ImplType impltype = Inpar::ScaTra::ImplType::impltype_undefined;
  };


  /*!
   *  @brief struct for managing solidporo anisotropy properties
   */
  struct AnisotropyProperties
  {
    //  anisotropic permeability directions
    std::vector<std::vector<double>> directions_{};

    // scaling coefficients for anisotropic permeability
    std::vector<std::vector<double>> nodal_coeffs_{};
  };

}  // namespace Discret::Elements
FOUR_C_NAMESPACE_CLOSE
#endif