// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_GROWTH_EVOLUTION_LINEAR_CAUCHY_POISSON_TURNOVER_HPP
#define FOUR_C_MIXTURE_GROWTH_EVOLUTION_LINEAR_CAUCHY_POISSON_TURNOVER_HPP


#include "4C_config.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  /*!
   * @brief Defines the evolution equation of net 1D mass production during growth and remodeling
   */
  template <typename Number>
  class LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution
  {
   public:
    /*!
     * \brief Default constructor
     *
     * \param k_sig Gain parameter of linear growth equation
     */
    LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution(
        double k_sig, double decay_time, bool enable_basal_mass_production)
        : k_sig_(k_sig),
          decay_time_(decay_time),
          enable_basal_mass_production_(enable_basal_mass_production)
    {
    }

    [[nodiscard]] Number evaluate_true_mass_production_rate(Number delta_sig) const
    {
      return k_sig_ * delta_sig + enable_basal_mass_production_ / decay_time_;
    }

    [[nodiscard]] Number evaluate_true_mass_removal_rate(Number delta_sig) const
    {
      return -1.0 / decay_time_;
    }

    [[nodiscard]] Number evaluate_d_true_mass_production_rate_d_sig(Number delta_sig) const
    {
      return k_sig_;
    }

    [[nodiscard]] Number evaluate_d_true_mass_removal_rate_d_sig(Number delta_sig) const
    {
      return 0.0;
    }

    [[nodiscard]] inline double evaluate_survival_function(double delta_time) const
    {
      return std::exp(-delta_time / decay_time_);
    }

    [[nodiscard]] inline double evaluate_survival_function_integration(
        const double current_time, const double t1, const double t2) const
    {
      return decay_time_ * (std::exp(-(current_time - t2) / decay_time_) -
                               std::exp(-(current_time - t1) / decay_time_));
    }

    /// Gain-Parameter controlling the speed of the growth process
    double k_sig_;

    /// decay time controlling the speed of turnover
    double decay_time_;

    /// a switch to enable the basal mass production rate
    bool enable_basal_mass_production_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
