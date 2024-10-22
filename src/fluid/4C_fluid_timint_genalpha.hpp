// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_TIMINT_GENALPHA_HPP
#define FOUR_C_FLUID_TIMINT_GENALPHA_HPP


#include "4C_config.hpp"

#include "4C_fluid_implicit_integration.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  /*! \brief Generalized-Alpha time integration for fluid problems
   *
   *  <h3> References </h3>
   *  <ul>
   *  <li>[1] Jansen KE, Whiting CH, Hulbert GM: A generalized-\f$\alpha\f$ method
   *          for integrating the filered Navier--Stokes euqations with a
   *          stabilized finite element method, Comp. Meth. Appl. Mech. Engng.,
   *          190(3-4), pp. 305--319, 2000 </li>
   *  </ul>
   */
  class TimIntGenAlpha : public virtual FluidImplicitTimeInt
  {
   public:
    /// Standard Constructor
    TimIntGenAlpha(const Teuchos::RCP<Core::FE::Discretization>& actdis,
        const Teuchos::RCP<Core::LinAlg::Solver>& solver,
        const Teuchos::RCP<Teuchos::ParameterList>& params,
        const Teuchos::RCP<Core::IO::DiscretizationWriter>& output, bool alefluid = false);


    /*!
    \brief initialization

    */
    void init() override;

    /*!
    \brief Print information about current time step to screen (reimplementation)

    */
    void print_time_step_info() override;

    /*!
    \brief Set theta_ to its value, dependent on integration method for GenAlpha and BDF2

    */
    void set_theta() override;

    /*!
    \brief Set the part of the righthandside belonging to the last
           timestep for incompressible or low-Mach-number flow

       for low-Mach-number flow: distinguish momentum and continuity part
       (continuity part only meaningful for low-Mach-number flow)

       Stationary/af-generalized-alpha:

                     mom: hist_ = 0.0
                    (con: hist_ = 0.0)

       One-step-Theta:

                     mom: hist_ = veln_  + dt*(1-Theta)*accn_
                    (con: hist_ = densn_ + dt*(1-Theta)*densdtn_)

       BDF2: for constant time step:

                     mom: hist_ = 4/3 veln_  - 1/3 velnm_
                    (con: hist_ = 4/3 densn_ - 1/3 densnm_)


    */
    void set_old_part_of_righthandside() override;

    /*!
    \brief Set states in the time integration schemes: differs between GenAlpha and the others

    */
    void set_state_tim_int() override;

    /*!
    \brief Set time factor in GenAlpha

    */
    double set_time_fac() override;

    /*!
    \brief Calculate time derivatives for
           stationary/one-step-theta/BDF2/af-generalized-alpha time integration
           for incompressible and low-Mach-number flow
    */
    void calculate_acceleration(
        const Teuchos::RCP<const Core::LinAlg::Vector<double>> velnp,  ///< velocity at n+1
        const Teuchos::RCP<const Core::LinAlg::Vector<double>> veln,   ///< velocity at     n
        const Teuchos::RCP<const Core::LinAlg::Vector<double>> velnm,  ///< velocity at     n-1
        const Teuchos::RCP<const Core::LinAlg::Vector<double>> accn,   ///< acceleration at n-1
        const Teuchos::RCP<Core::LinAlg::Vector<double>> accnp         ///< acceleration at n+1
        ) override;

    /*!
    \brief compute values at intermediate time steps for gen.-alpha
           for given vectors and store result in given vectors
           Helper method which can be called from outside fluid (e.g. for coupled problems)

    */
    void gen_alpha_intermediate_values(Teuchos::RCP<Core::LinAlg::Vector<double>>& vecnp,
        Teuchos::RCP<Core::LinAlg::Vector<double>>& vecn) override;

    /*!
    \brief Set gamma to a value

    */
    void set_gamma(Teuchos::ParameterList& eleparams) override;

    /*!
    \brief Scale separation

    */
    void sep_multiply() override;

    /*!
    \brief Update velaf_ for GenAlpha

    */
    void update_velaf_gen_alpha() override;

    /*!
    \brief Output of filtered velocity

    */
    void outputof_filtered_vel(Teuchos::RCP<Core::LinAlg::Vector<double>> outvec,
        Teuchos::RCP<Core::LinAlg::Vector<double>> fsoutvec) override;

    /*!

    \brief parameter (fix over a time step) are set in this method.
           Therefore, these parameter are accessible in the fluid element
           and in the fluid boundary element

    */
    void set_element_time_parameter() override;

    /*!
    \brief return scheme-specific time integration parameter

    */
    double tim_int_param() const override { return (1.0 - alphaF_); }

    virtual double alpha_m() const { return alphaM_; }

    //! @name methods for fsi
    /// Extrapolation of vectors from mid-point to end-point t_{n+1}
    Teuchos::RCP<Core::LinAlg::Vector<double>> extrapolate_end_point(
        Teuchos::RCP<Core::LinAlg::Vector<double>> vecn,  ///< vector at time level t_n
        Teuchos::RCP<Core::LinAlg::Vector<double>> vecm   ///< vector at time level of equilibrium
        ) override;

    /*!
    \brief treat turbulence models in assemble_mat_and_rhs

    */
    void treat_turbulence_models(Teuchos::ParameterList& eleparams) override;

    //! @name Time Step Size Adaptivity
    //@{

    /*! \brief Give local order of accuracy of velocity part
     *
     *  Generalized-Alpha is second order accurate if
     *  \f$\gamma=\frac{1}{2}+\alpha_{m}-\alpha_{f}\f$
     *  (cf. eq. (17) in [Jansen, 2000]), otherwise only first order.
     *
     *  \author mayr.mt \date 04/2015
     */
    int method_order_of_accuracy_vel() const override;

    /*! \brief Give local order of accuracy of pressure part
     *
     *  Generalized-Alpha is second order accurate if
     *  \f$\gamma=\frac{1}{2}+\alpha_{m}-\alpha_{f}\f$
     *  (cf. eq. (17) in [Jansen, 2000]), otherwise only first order.
     *
     *  \author mayr.mt \date 04/2015
     */
    int method_order_of_accuracy_pres() const override;

    /*! \brief Return linear error coefficient of velocity
     *
     *  The linear discretization error reads
     *  \f[
     *  e \approx \Delta t_n^2\left(\frac{1}{2}-\gamma\right)\ddot{u}(t_n)
     *    + \Delta t_n^3\left(\frac{1}{6}-\frac{\gamma}{2}\right)\dddot{u}(t_n)
     *    + HOT\left(\Delta t_n^4\right)
     *  \f]
     *
     *  \author mayr.mt \date 04/2015
     */
    double method_lin_err_coeff_vel() const override;

    //@}

   protected:
    /// copy constructor
    TimIntGenAlpha(const TimIntGenAlpha& old);

    /*!
    \brief update acceleration for generalized-alpha time integration

    */
    void gen_alpha_update_acceleration() override;

    /*!
    \brief compute values at intermediate time steps for gen.-alpha

    */
    void gen_alpha_intermediate_values() override;

    /*!
    \brief return scaling of the residual

    */
    double residual_scaling() const override { return alphaM_ / (gamma_ * dta_); }

    /*!
    \brief velocity required for evaluation of related quantites required on element level

    */
    Teuchos::RCP<const Core::LinAlg::Vector<double>> evaluation_vel() override { return velaf_; };

    /// time factors for generalized-alpha time integration
    double alphaM_;
    double alphaF_;
    double gamma_;

    //! @name time stepping variables
    bool startalgo_;  ///< flag for starting algorithm


  };  // class TimIntGenAlpha

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
