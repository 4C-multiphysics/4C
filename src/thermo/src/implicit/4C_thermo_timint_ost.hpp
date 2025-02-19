// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_TIMINT_OST_HPP
#define FOUR_C_THERMO_TIMINT_OST_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 08/09 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_thermo_timint_impl.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | belongs to thermal dynamics namespace                     dano 08/09 |
 *----------------------------------------------------------------------*/
namespace Thermo
{
  /*====================================================================*/
  /*!
   * \brief One-step-theta time integration (or Crank-Nicholson scheme or generalised trapezoidal
   * rule)
   *
   * <h3> Background </h3>
   * One-step theta time integration is a finite difference method for 1st order ordinary
   * differential equations (ODE) of the type \f[ F(y,\dot{y},t) = \dot{y}(t) - f(y(t),t) = 0 \f]
   *
   * The one-step-theta time integration method discretises this equation into the following
   * reference formula \f[ \frac{y_{n+1} - y_n}{\Delta t}
   *   - \theta f(y_{n+1},t_{n+1}) + (1-\theta) f(y_n,t_n)
   *   = 0
   * \qquad
   * \f]
   * in which \f$\theta\in[0,1]\f$ is the key parameter. The method is implicit unless
   * \f$\theta=0\f$, which is the forward Euler scheme. The method recovers the backward Euler
   * method with \f$\theta=1\f$. The trapezoidal rule (TR, or average acceleration method) is
   * obtained with \f$\theta=1/2\f$. Only the trapezoidal rule is second order accurate, all other
   * schemes are only first order.
   *
   * This method is applied to the set of ODEs reflecting the first order degree of the governing
   * equations in thermal dynamics: \f[\left\{\begin{array}{rcl}
   *
   *   C \, R(t) + F_{int}(T,t) - F_{ext}(t) & = & 0
   * \end{array}\right.\f]
   * \f$C\f$ is a global capacity matrix, \f$T(t)\f$ the temperature (#temp_), \f$R(t)\f$ the
   * temperature rates (#rate_), \f$F_{int}\f$ the internal forces and \f$F_{ext}\f$ the external
   * forces. One obtains \f[\left\{\begin{array}{rcl} \frac{T_{n+1} - T_n}{\Delta t}
   *   - \theta R_{n+1} - (1-\theta) R_n
   *   & = & 0
   * \\
   *   C \frac{T_{n+1} - T_n}{\Delta t}
   *   + F_{int,n+\theta}
   *   - F_{ext,n+\theta}
   *   & = & 0
   * \end{array}\right.\f]
   * with
   * \f[
   *   F_{int,n+\theta}
   *   = \theta F_{int}(T_{n+1},t_{n+1}) + (1-\theta) F_{int}(T_{n+1},t_{n+1})
   *   \quad\mbox{and}\quad
   *   F_{ext,n+\theta}
   *   = \theta F_{ext}(t_{n+1}) + (1-\theta) F_{ext}(t_{n+1})
   * \f]
   * These vector equations can be rewritten such that the unknown temperature rates \f$r_{n+1}\f$
   * can be suppressed or rather expressed by the unknown temperatures \f$T_{n+1}\f$. The residual
   * is achieved \f[ R_{n+\theta}(T_{n+1}) = C R_{n+\theta}(T_{n+1})
   *   + F_{int,n+\theta} - F_{ext,n+\theta}
   *   = 0
   * \f]
   * in which
   * \f[\begin{array}{rclcl}
   *   R_{n+\theta}(T_{n+1})
   *   & = &
   *   \frac{1}{\Delta t} ( T_{n+1} - T_n )
   *   &&
   * \end{array}\f]
   *
   * <h3>Family members to be aware of</h3>
   * <table>
   *   <tr>
   *     <td>Name</td>
   *     <td>Abbreviation</td>
   *     <td>\f$\theta\f$</td>
   *     <td>Order of accuracy</td>
   *     <td>Stability</td>
   *   </tr>
   *   <tr>
   *     <td>Backward Euler</td>
   *     <td>BE</td>
   *     <td>\f$1\f$</td>
   *     <td>1</td>
   *     <td>A,L-stable</td>
   *   </tr>
   *   <tr>
   *     <td>Trapezoidal rule</td>
   *     <td>TR</td>
   *     <td>\f$\frac{1}{2}\f$</td>
   *     <td>2</td>
   *     <td>A-stable</td>
   *   </tr>
   * </table>
   *
   * <h3>References</h3>
   * - [1] HR Schwarz, Numerische Mathematik, Teubner, Stuttgart, 1997.
   * - [2] TJR Hughes, The finite element method, Dover, Mineola, 1987.
   * - [3] P Deuflhard and F Bornemann, Numerische Mathematik II: Integration gewohnlicher
   * Differentialgleichungen, Walter de Gryter, Berlin, 1994.
   * - [4] ...
   *
   *
   * \author bborn
   * \date 06/08
   */
  class TimIntOneStepTheta : public TimIntImpl
  {
   public:
    //! verify if given coefficients are in admissible range
    //! prints also info to STDOUT
    void verify_coeff();

    //! @name Construction
    //@{

    //! Constructor
    TimIntOneStepTheta(const Teuchos::ParameterList& ioparams,  //!< ioflags
        const Teuchos::ParameterList& tdynparams,               //!< input parameters
        const Teuchos::ParameterList& xparams,                  //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,       //!< current discretisation
        std::shared_ptr<Core::LinAlg::Solver> solver,           //!< the solver
        std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< the output
    );

    //@}

    //! @name Pure virtual methods which have to be implemented
    //@{

    //! Return name
    enum Thermo::DynamicType method_name() const override { return Thermo::dyna_onesteptheta; }

    //! Consistent predictor with constant temperatures
    //! and consistent temperature rates and temperatures
    void predict_const_temp_consist_rate() override;

    //! Evaluate ordinary internal force, its tangent at state
    void apply_force_tang_internal(const double time,               //!< evaluation time
        const double dt,                                            //!< step size
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< residual temperatures
        std::shared_ptr<Core::LinAlg::Vector<double>> fcap,         //!< capacity force
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,         //!< internal force
        std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
    );

    //! Evaluate ordinary internal force
    void apply_force_internal(const double time,                    //!< evaluation time
        const double dt,                                            //!< step size
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< incremental temperatures
        std::shared_ptr<Core::LinAlg::Vector<double>> fint          //!< internal force
    );

    //! Evaluate a convective boundary condition
    // (nonlinear --> add term to tangent)
    void apply_force_external_conv(const double time,               //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempn,  //!< temperature state T_n
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state T_n+1
        std::shared_ptr<Core::LinAlg::Vector<double>> fext,         //!< internal force
        std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
    );

    //! Create force residual #fres_ and its tangent #tang_
    void evaluate_rhs_tang_residual() override;

    //! Determine characteristic norm for temperatures
    //! \author lw (originally)
    double calc_ref_norm_temperature() override;

    //! Determine characteristic norm for force
    //! \author lw (originally)
    double calc_ref_norm_force() override;

    //! Update iteration incrementally
    //!
    //! This update is carried out by computing the new #raten_
    //! from scratch by using the newly updated #tempn_. The method
    //! respects the Dirichlet DOFs which are not touched.
    //! This method is necessary for certain predictors
    //! (like #predict_const_temp_consist_rate)
    void update_iter_incrementally() override;

    //! Update iteration iteratively
    //!
    //! This is the ordinary update of #tempn_ and #raten_ by
    //! incrementing these vector proportional to the residual
    //! temperatures #tempi_
    //! The Dirichlet BCs are automatically respected, because the
    //! residual temperatures #tempi_ are blanked at these DOFs.
    void update_iter_iteratively() override;

    //! Update step
    void update_step_state() override;

    //! Update element
    void update_step_element() override;

    //! Read and set restart for forces
    void read_restart_force() override;

    //! Write internal and external forces for restart
    void write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output) override;

    //@}

    //! @name Access methods
    //@{

    //! Return external force \f$F_{ext,n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext() override { return fext_; }

    //! Return external force \f$F_{ext,n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext_new() override { return fextn_; }

    //@}

    //! @name Generalised-alpha specific methods
    //@{

    //! Evaluate mid-state vectors by averaging end-point vectors
    void evaluate_mid_state();

    //@}

   protected:
    //! equal operator is NOT wanted
    TimIntOneStepTheta operator=(const TimIntOneStepTheta& old);

    //! copy constructor is NOT wanted
    TimIntOneStepTheta(const TimIntOneStepTheta& old);

    //! @name Key coefficients
    //@{
    double theta_;  //!< factor (0,1]
    //@}

    //! @name Global mid-state vectors
    //@{
    //! mid-temperatures \f$T_m = T_{n+\theta}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> tempt_;

    //@}

    //! @name Global force vectors
    //! Residual \c fres_ exists already in base class
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fint_;   //!< internal force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fintn_;  //!< internal force at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>>
        fcap_;  //!< capacity force \f$C\cdot\Theta_n\f$ at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        fcapn_;  //!< capacity force \f$C\cdot\Theta_{n+1}\f$ at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>> fext_;   //!< external force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fextn_;  //!< external force at \f$t_{n+1}\f$

    //@}

  };  // class TimIntOneStepTheta

}  // namespace Thermo

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
