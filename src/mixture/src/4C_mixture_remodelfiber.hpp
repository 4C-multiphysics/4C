// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_REMODELFIBER_HPP
#define FOUR_C_MIXTURE_REMODELFIBER_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication

namespace Mixture
{
  template <typename T>
  class RemodelFiberMaterial;


  namespace Implementation
  {
    template <int numstates, typename T>
    class RemodelFiberImplementation;
  }

  template <int numstates>
  class RemodelFiber
  {
    struct GRState
    {
      double growth_scalar = 1.0;
      double lambda_r = 1.0;
      double lambda_f = 1.0;
    };

   public:
    RemodelFiber(std::shared_ptr<const RemodelFiberMaterial<double>> material,
        LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution,
        double lambda_pre);

    /*!
     * @brief Pack all internal data into the #data
     *
     * @param data (out) : buffer to serialize data to.
     */
    void pack(Core::Communication::PackBuffer& data) const;

    /*!
     * @brief Unpack all internal data that was previously packed by
     * #pack(Core::Communication::PackBuffer&)
     *
     * @param position (in/out) : Position, where to start reading
     * @param data (in) : Vector of chars to extract data from
     */
    void unpack(Core::Communication::UnpackBuffer& buffer);

    /// @brief Updates previous history data
    void update();

    /*!
     * @brief Sets the deposition (homeostatic) stretch.
     *
     * @param lambda_pre
     */
    void update_deposition_stretch(double lambda_pre);

    /*!
     * @brief Set deformation state of the fiber
     *
     * @note This method has to be called before any Evaluation or local integration
     *
     * @param lambda_f (in) : total stretch in fiber direction
     * @param lambda_ext (in) : inelastic external stretch
     */
    void set_state(double lambda_f, double lambda_ext);

   public:
    /// @brief Evaluation methods
    ///
    /// @note It is important to call #set_state(double) first.
    ///
    /// @{

    /// @name Methods for doing explicit or implicit time integration
    /// @{
    /*!
     * @brief Integrate the local evolution equation with an implicit time integration scheme.
     *
     * @param dt (in) : timestep
     *
     * @return Derivative of the residuum of the time integration scheme w.r.t. growth scalar and
     * lambda_r
     */
    Core::LinAlg::Matrix<2, 2> integrate_local_evolution_equations_implicit(double dt);

    /*!
     * @brief Integrate the local evolution equation with an explicit time integration scheme.
     *
     * @param dt (in) : timestep
     */
    void integrate_local_evolution_equations_explicit(double dt);
    /// @}
    [[nodiscard]] double evaluate_current_homeostatic_fiber_cauchy_stress() const;
    [[nodiscard]] double evaluate_current_fiber_cauchy_stress() const;
    [[nodiscard]] double evaluate_current_fiber_pk2_stress() const;
    [[nodiscard]] double evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq() const;
    [[nodiscard]] double evaluate_d_current_fiber_pk2_stress_d_lambda_r() const;
    [[nodiscard]] double
    evaluate_d_current_growth_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
        double dt) const;
    [[nodiscard]] double
    evaluate_d_current_remodel_evolution_implicit_time_integration_residuum_d_lambda_f_sq(
        double dt) const;
    [[nodiscard]] double evaluate_current_growth_scalar() const;
    [[nodiscard]] double evaluate_current_lambda_r() const;
    [[nodiscard]] double evaluate_d_current_growth_scalar_d_lambda_f_sq() const;
    [[nodiscard]] double evaluate_d_current_lambda_r_d_lambda_f_sq() const;
    [[nodiscard]] double evaluate_d_current_cauchy_stress_d_lambda_f_sq() const;
    /// @}

   private:
    const std::shared_ptr<Implementation::RemodelFiberImplementation<2, double>> impl_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
