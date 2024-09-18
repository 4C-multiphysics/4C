/*! \file
\brief Interface for every material that can evaluate coupled thermo-solid material laws

\level 3

*/

#include "4C_config.hpp"

#include "4C_mat_monolithic_solid_scalar_material.hpp"
#include "4C_mat_trait_solid.hpp"
#include "4C_mat_trait_thermo.hpp"

#ifndef FOUR_C_MAT_TRAIT_THERMO_SOLID_HPP
#define FOUR_C_MAT_TRAIT_THERMO_SOLID_HPP

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Trait
  {
    class ThermoSolid : public Thermo, public Solid, public MonolithicSolidScalarMaterial
    {
     public:
      /*!
       * Set current quantities for this material
       *
       * The quantities are used for evaluation and possibly in CommitCurrentState()
       * @param defgrd
       * @param glstrain
       * @param temperature
       * @param gp
       *
       */
      virtual void reinit(const Core::LinAlg::Matrix<3, 3>* defgrd,
          const Core::LinAlg::Matrix<6, 1>* glstrain, double temperature, unsigned gp) = 0;

      /*!
       * Return stress-temperature modulus and thermal derivative for coupled thermomechanics
       *
       * @param stm tensor to be filled with stress-temperature moduli
       * @param stm_deriv tensor to be filled with derivatives
       */
      virtual void stress_temperature_modulus_and_deriv(
          Core::LinAlg::Matrix<6, 1>& stm, Core::LinAlg::Matrix<6, 1>& stm_dT, int gp) = 0;
    };
  }  // namespace Trait
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif