// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_IMPL_GENALPHA_HPP
#define FOUR_C_STRUCTURE_NEW_IMPL_GENALPHA_HPP

#include "4C_config.hpp"

#include "4C_structure_new_impl_generic.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class MultiVector;
}

namespace Solid
{
  namespace IMPLICIT
  {
    /*! Generalized-\f$\alpha\f$ time integration for 2nd-order ODEs
     *
     * <h3> References </h3>
     * - Chung J, Hulbert GM:
     *   A Time Integration Algorithm for Structural Dynamics With Improved Numerical Dissipation:
     *   The Generalized-\f$\alpha\f$ Method
     *   Journal of Applied Mechanics, 60(2):371--375 (1993)
     */
    class GenAlpha : public Generic
    {
     public:
      // forwards declaration
      struct Coefficients;

      //! constructor
      GenAlpha();

      //! Setup the class variables
      void setup() override;

      //! (derived)
      void post_setup() override;

      //! Reset state variables [derived]
      void set_state(const Core::LinAlg::Vector<double>& x) override;

      //! Apply the rhs only [derived]
      bool apply_force(
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f) override;

      //! Apply the stiffness only [derived]
      bool apply_stiff(
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) override;

      //! Apply force and stiff at once [derived]
      bool apply_force_stiff(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f,
          Core::LinAlg::SparseOperator& jac) override;

      //! [derived]
      bool assemble_force(Core::LinAlg::Vector<double>& f,
          const std::vector<Solid::ModelType>* without_these_models = nullptr) const override;

      bool assemble_jac(Core::LinAlg::SparseOperator& jac,
          const std::vector<Solid::ModelType>* without_these_models = nullptr) const override;

      //! [derived]
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! [derived]
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! [derived]
      double calc_ref_norm_force(const enum ::NOX::Abstract::Vector::NormType& type) const override;

      //! access the alphaf parameter of the Generalized-\f$\alpha\f$ Method
      double get_int_param() const override;

      /// access the parameter for the accelerations at \f$t_{n}\f$, i.e. alpham
      double get_acc_int_param() const override;

      //! @name Monolithic update routines
      //!@{

      //! Update configuration after time step [derived]
      void update_step_state() override;

      //! Update everything on element level after time step and after output [derived]
      void update_step_element() override;

      /*! \brief things that should be done after updating [derived]
       *
       *  We use in the GenAlpha case to update constant contributions (during one time step)
       *  of the set_state routine.*/
      void post_update() override;

      //!@}

      //! @name Predictor routines (dependent on the implicit integration scheme)
      //!@{

      //! predict constant displacements, consistent velocities and accelerations [derived]
      void predict_const_dis_consist_vel_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;

      //! predict displacements based on constant velocities and consistent accelerations [derived]
      bool predict_const_vel_consist_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;

      //! predict displacements based on constant accelerations and consistent velocities [derived]
      bool predict_const_acc(Core::LinAlg::Vector<double>& disnp,
          Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const override;

      /*! \brief Time integration coefficients container
       *
       * \note For bounds on the individual parameters, please consult the
       * original publication by Chung and Hulbert (1993). In practice however,
       * it is advised to choose the spectral radius as \f$\rho_\infty \in [0.5,1]\f$.
       */
      struct Coefficients
      {
        /// copy operator (currently default is enough)
        Coefficients& operator=(const Coefficients& source) = default;

        //! Parameter \f$\beta \in (0,1/2]\f$
        double beta_ = -1.0;

        //! Parameter \f$\gamma \in (0,1]\f$
        double gamma_ = -1.0;

        //! Parameter \f$\alpha_f \in [0,1)\f$
        double alphaf_ = -1.0;

        //! Parameter \f$\alpha_m \in [-1,1)\f$
        double alpham_ = -1.0;

        //! Spectral radius \f$\rho_\infty \in [0,1]\f$
        double rhoinf_ = -1.0;
      };

      //!@}

      //! @name Attribute access functions
      //@{

      //! Return name
      Solid::DynamicType method_name() const override { return Solid::DynamicType::GenAlpha; }

      //! Provide number of steps, e.g. a single-step method returns 1,
      //! a m-multistep method returns m
      int method_steps() const override { return 1; }

      //! Give linear order of accuracy of displacement part
      int method_order_of_accuracy_dis() const override
      {
        return (fabs(method_lin_err_coeff_dis2()) < 1e-6) ? 3 : 2;
      }

      //! Give linear order of accuracy of velocity part
      int method_order_of_accuracy_vel() const override
      {
        return (fabs(method_lin_err_coeff_vel1()) < 1e-6) ? 2 : 1;
      }

      //! Return linear error coefficient of displacements
      double method_lin_err_coeff_dis() const override
      {
        if (method_order_of_accuracy_dis() == 2)
          return method_lin_err_coeff_dis2();
        else
          return method_lin_err_coeff_dis3();
      }

      //! 2nd order linear error coefficient of displacements
      double method_lin_err_coeff_dis2() const
      {
        const double& alphaf = coeffs_.alphaf_;
        const double& alpham = coeffs_.alpham_;
        const double& beta = coeffs_.beta_;

        // at least true for am<1/2 and large enough n->infty
        return 1.0 / 6.0 - beta + alphaf / 2.0 - alpham / 2.0;
      }

      //! 3rd order linear error coefficient of displacements
      double method_lin_err_coeff_dis3() const
      {
        const double& alphaf = coeffs_.alphaf_;
        const double& alpham = coeffs_.alpham_;
        const double& beta = coeffs_.beta_;

        // at least true for am<1/2 and large enough n->infty
        return 1. / 24. - beta / 2. * (1. - 2 * alphaf + 2. * alpham) -
               1. / 4. * (alphaf - alpham) * (1. - 2. * alpham);
      }

      //! Return linear error coefficient of velocities
      double method_lin_err_coeff_vel() const override
      {
        if (method_order_of_accuracy_vel() == 1)
          return method_lin_err_coeff_vel1();
        else
          return method_lin_err_coeff_vel2();
      }

      //! 1st order linear error coefficient of velocities
      double method_lin_err_coeff_vel1() const
      {
        const double& alphaf = coeffs_.alphaf_;
        const double& alpham = coeffs_.alpham_;
        const double& gamma = coeffs_.gamma_;

        // at least true for am<1/2 and large enough n->infty
        return 1.0 / 2.0 - gamma + alphaf - alpham;
      }

      //! 2nd order linear error coefficient of velocities
      double method_lin_err_coeff_vel2() const
      {
        const double& alphaf = coeffs_.alphaf_;
        const double& alpham = coeffs_.alpham_;
        const double& gamma = coeffs_.gamma_;

        // at least true for am<1/2 and large enough n->infty
        return 1. / 6. - gamma / 2. * (1. - 2 * alphaf + 2. * alpham) -
               1. / 2. * (alphaf - alpham) * (1. - 2. * alpham);
      }

      //@}

     protected:
      //! reset the time step dependent parameters for the element evaluation [derived]
      void reset_eval_params() override;

      //! Refresh cached force pointers and map-dependent helper vectors after redistribution.
      void remap_integrator_state_after_redistribution() override;

     private:
      /*! \brief Add the viscous and mass contributions to the right hand side (TR-rule)
       *
       * \remark The remaining contributions have been considered in the corresponding model
       *         evaluators. This is due to the fact, that some models use a different
       *         time integration scheme for their terms (e.g. GenAlpha for the structure
       *         and OST for the remaining things).
       *
       *  \f[
       *    Res = M . [(1-\alpha_m) * A_{n+1} + \alpha_m * A_{n}]
       *        + C . [(1-\alpha_f) * V_{n+1} + \alpha_f * V_{n}]
       *        + (1-\alpha_f) * Res_{\mathrm{statics},n+1} + \alpha_f * Res_{\mathrm{statics},n}
       *  \f]
       *
       *  \param[in/out] f Right-hand side vector
       *
       */
      void add_visco_mass_contributions(Core::LinAlg::Vector<double>& f) const override;

      /*! \brief Add the viscous and mass contributions to the jacobian (TR-rule)
       *
       *  \remark The remaining blocks have been considered in the corresponding model
       *          evaluators. This is due to the fact, that some models use a different
       *          time integration scheme for their terms (e.g. GenAlpha for the structure
       *          and OST for the remaining things). Furthermore, constraint/Lagrange
       *          multiplier blocks need no scaling anyway.
       *
       *  \f[
       *    \boldsymbol{K}_{T,effdyn} = (1 - \frac{\alpha_m}{\beta (\Delta t)^{2}} \boldsymbol{M}
       *                + (1 - \frac{\alpha_f \gamma}{\beta \Delta t} \boldsymbol{C}
       *                + (1 - \alpha_f)  \boldsymbol{K}_{T}
       *  \f]
       *
       * \param[in/out] jac Jacobian matrix
       *
       */
      void add_visco_mass_contributions(Core::LinAlg::SparseOperator& jac) const override;

      /*! \brief Update constant contributions of the current state for the new time step
       * \f$ t_{n+1} \f$ based on the generalized alpha scheme:
       *
       * \f[
       *      V_{n+1} = (2.0 * \beta - \gamma)/(2.0 * \beta) * dt * A_{n} + (\beta - \gamma)/\beta
       *              * V_{n} - \gamma/(\beta * dt) * D_{n} + \gamma/(\beta * dt) * D_{n+1}
       *      A_{n+1} = (2.0 * \beta - 1.0)/(2.0 * \beta) * A_{n} - 1.0/(\beta * dt) * V_{n}
       *              - 1.0/(\beta * dt^2) * D_{n} + 1.0/(\beta * dt^2) * D_{n+1}
       * \f]
       *
       * Only the constant contributions, i.e. all components that depend on the state n are stored
       * in the const_vel_acc_update_ptr_ multi-vector pointer. The 1st entry represents the
       * velocity, and the 2nd the acceleration.
       *
       *  See the set_state() routine for the iterative update of the current state. */
      void update_constant_state_contributions() override;

      /// set the time integration coefficients
      void set_time_integration_coefficients(Coefficients& coeffs) const;

     private:
      Coefficients coeffs_;

     protected:
      /** @name Generalized alpha parameters
       *
       *  \note redirection to the content of the coefficient struct */
      //!@{

      //! Parameter \f$\beta \in (0,1/2]\f$
      double& beta_;

      //! Parameter \f$\gamma \in (0,1]\f$
      double& gamma_;

      //! Parameter \f$\alpha_f \in [0,1)\f$
      double& alphaf_;

      //! Parameter \f$\alpha_m \in [-1,1)\f$
      double& alpham_;

      //! Spectral radius \f$\rho_\infty \in [0,1]\f$
      double& rhoinf_;

      //!@}

      /*! @name New vectors for internal use only
       *
       *  If an external use seems necessary, move these vectors to the
       *  global state data container and just store a pointer to the global
       *  state variable. */
      //!@{

      /*! \brief Holds the during a time step constant contributions to
       *  the velocity and acceleration state update.
       *
       *  entry (0): constant velocity contribution \f$\tilde{V}_{n+1}\f$
       *  entry (1): constant acceleration contribution \f$\tilde{A}_{n+1}\f$ */
      std::shared_ptr<Core::LinAlg::MultiVector<double>> const_vel_acc_update_ptr_;

      //!@}

      //! @name pointers to the global state data container content
      //!@{

      //! viscous force vector F_viscous F_{viscous;n+1}
      std::shared_ptr<Core::LinAlg::Vector<double>> fvisconp_ptr_;

      //! viscous force vector F_viscous F_{viscous;n}
      std::shared_ptr<Core::LinAlg::Vector<double>> fviscon_ptr_;

      //! pointer to inertial force vector F_{inertial,n+1} at new time
      std::shared_ptr<Core::LinAlg::Vector<double>> finertianp_ptr_;

      //! pointer to inertial force vector F_{inertial,n} at last time
      std::shared_ptr<Core::LinAlg::Vector<double>> finertian_ptr_;

      //!@}
    };  // namespace IMPLICIT
  }  // namespace IMPLICIT

  /*! calculate GenAlpha parameters
   *
   * use the input to calculate GenAlpha parameters: alphaf, alpham, beta and gamma
   * if the user provides spectral radius RHO_INF with in [0,1], the optimal set of the four
   * parameters will be calculated and passed to following process
   * If the user provides the four parameters directly, he is asked to set RHO_INF as -1.0
   * this function also makes sure each parameter is in their correct range.
   \note just an internal function to reduce redundancy in structure and structure_new
   *
   * @param coeffs
   */
  void compute_generalized_alpha_parameters(Solid::IMPLICIT::GenAlpha::Coefficients& coeffs);

}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
