// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_monolithic_solid_scalar_material.hpp"
#include "4C_mat_so3_material.hpp"

#ifndef FOUR_C_MAT_TRAIT_THERMO_SOLID_HPP
#define FOUR_C_MAT_TRAIT_THERMO_SOLID_HPP

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Trait
  {
    class ThermoSolid : public So3Material, public MonolithicSolidScalarMaterial
    {
     public:
      /*!
       * Set the current temperature for this material
       *
       * @param temperature
       * @param gp
       *
       */
      virtual void reinit(double temperature, unsigned gp) = 0;

      /*!
       * Return stress-temperature modulus and thermal derivative for coupled thermomechanics
       *
       * @param stm tensor to be filled with stress-temperature moduli
       * @param stm_deriv tensor to be filled with derivatives
       */
      virtual void stress_temperature_modulus_and_deriv(
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stm,
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stm_dT, int gp) = 0;
    };
  }  // namespace Trait
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif