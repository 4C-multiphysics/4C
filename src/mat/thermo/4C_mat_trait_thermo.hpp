// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_TRAIT_THERMO_HPP
#define FOUR_C_MAT_TRAIT_THERMO_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Trait
  {
    class Thermo
    {
     public:
      virtual ~Thermo() = default;

      //! Main material call to determine heat flux and constitutive tensor in 3D
      virtual void evaluate(
          const Core::LinAlg::Matrix<3, 1>& gradtemp,  ///< temperature gradient (strain tensor)
          Core::LinAlg::Matrix<3, 3>& cmat,            ///< constitutive matrix
          Core::LinAlg::Matrix<3, 1>& heatflux,        ///< heatflux
          const int eleGID) const = 0;

      //! Main material call to determine heat flux and constitutive tensor in 2D
      virtual void evaluate(
          const Core::LinAlg::Matrix<2, 1>& gradtemp,  ///< temperature gradient (strain tensor)
          Core::LinAlg::Matrix<2, 2>& cmat,            ///< constitutive matrix
          Core::LinAlg::Matrix<2, 1>& heatflux,        ///< heatflux
          const int eleGID) const = 0;

      //! Main material call to determine heat flux and constitutive tensor in 1D
      virtual void evaluate(
          const Core::LinAlg::Matrix<1, 1>& gradtemp,  ///< temperature gradient (strain tensor)
          Core::LinAlg::Matrix<1, 1>& cmat,            ///< constitutive matrix
          Core::LinAlg::Matrix<1, 1>& heatflux,        ///< heatflux
          const int eleGID) const = 0;


      //! @brief get conductivity
      virtual std::vector<double> conductivity(int eleGID = 0) const = 0;

      //! @name Derivatives of conductivity tensor
      //! @{

      //! @brief Derivative of conductivity tensor wrt to temperature
      virtual void conductivity_deriv_t(Core::LinAlg::Matrix<3, 3>& dCondDT) const = 0;

      virtual void conductivity_deriv_t(Core::LinAlg::Matrix<2, 2>& dCondDT) const = 0;

      virtual void conductivity_deriv_t(Core::LinAlg::Matrix<1, 1>& dCondDT) const = 0;

      //! @}

      //! @brief get volumetric heat capacity
      //!
      //! @pre the state must be set by Reinit() if necessary
      virtual double capacity() const = 0;

      //! @brief get derivative of volumetric heat capacity wrt temperature
      //!
      //! @pre state must be set by Reinit() if necessary.
      virtual double capacity_deriv_t() const = 0;

      //! Set necessary variables for Evaluation
      virtual void reinit(double temperature, unsigned gp) = 0;

      //! reset current state e.g. due to Newton failed
      virtual void reset_current_state() = 0;

      //! persist currently set state to history
      virtual void commit_current_state() = 0;
    };
  }  // namespace Trait
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
