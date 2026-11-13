// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_THERMO_HPP
#define FOUR_C_MAT_THERMO_HPP

#include "4C_config.hpp"

#include "4C_mat_material_factory.hpp"
#include "4C_mat_trait_thermo.hpp"
#include "4C_material_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class ThermoMaterial : public Core::Mat::Material, public Trait::Thermo
  {
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
