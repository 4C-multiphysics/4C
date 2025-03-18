// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMINT_OST_HPP
#define FOUR_C_STRUCTURE_TIMINT_OST_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_structure_timint_impl.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* belongs to structural dynamics namespace */
namespace Solid
{
  /*====================================================================*/
  /*!
   * \brief One-step-theta time integration (or Crank-Nicolson scheme or generalised trapezoidal
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
   * This method is applied to the set of ODEs reflecting the second order degree of the governing
   * equations in structural dynamics: \f[\left\{\begin{array}{rcl} M \, \dot{D}(t) - M \, V(t) & =
   * & 0
   * \\
   *   M \, \dot{V}(t) + C \, V(t) + F_{int}(D,t) - F_{ext}(t) & = & 0
   * \end{array}\right.\f]
   * in which \f$M\f$ is the global mass matrix (#mass_), \f$C\f$ a global damping matrix (in our
   * algorithm the Rayleigh damping matrix, #damp_), \f$D(t)\f$ the displacements (#dis_),
   * \f$V(t)\f$ the velocities (#vel_), \f$F_{int}\f$ the internal forces and \f$F_{ext}\f$ the
   * external forces. One obtains \f[\left\{\begin{array}{rcl} \frac{D_{n+1} - D_n}{\Delta t}
   *   - \theta V_{n+1} - (1-\theta) V_n
   *   & = & 0
   * \\
   *   M \frac{V_{n+1} + V_n}{\Delta t}
   *   + C \big( \theta V_{n+1} - (1-\theta) V_n \big)
   *   + F_{int,n+\theta}
   *   - F_{ext,n+\theta}
   *   & = & 0
   * \end{array}\right.\f]
   * with
   * \f[
   *   F_{int,n+\theta}
   *   = \theta F_{int}(D_{n+1},t_{n+1}) + (1-\theta) F_{int}(D_{n+1},t_{n+1})
   *   \quad\mbox{and}\quad
   *   F_{ext,n+\theta}
   *   = \theta F_{ext}(t_{n+1}) + (1-\theta) F_{ext}(t_{n+1})
   * \f]
   * These vector equations can be rewritten such that the unknown velocities \f$V_{n+1}\f$ can be
   * suppressed or rather expressed by the unknown displacements \f$D_{n+1}\f$. The residual is
   * achieved \f[ R_{n+\theta}(D_{n+1}) = M A_{n+\theta}(D_{n+1}) + C V_{n+\theta}(D_{n+1})
   *   + F_{int,n+\theta} - F_{ext,n+\theta}
   *   = 0
   * \f]
   * in which
   * \f[\begin{array}{rclcl}
   *   A_{n+\theta}(D_{n+1})
   *   & = &
   *   \frac{1}{\theta \Delta t^2} ( D_{n+1} - D_n )
   *   - \frac{1}{\theta \Delta t} V_n
   *   & =: &
   *   \theta A_{n+1} + (1-\theta) A_n
   * \\
   *   V_{n+\theta}(D_{n+1})
   *   & = &
   *   \frac{1}{\Delta t} ( D_{n+1} - D_n )
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

   */
  class TimIntOneStepTheta : public TimIntImpl
  {
   public:
    //! verify if given coefficients are in admissible range;
    //! prints also info to STDOUT
    void verify_coeff();

    //! @name Construction
    //@{

    //! Constructor
    TimIntOneStepTheta(const Teuchos::ParameterList& timeparams,  //!< time parameters
        const Teuchos::ParameterList& ioparams,                   //!< ioflags
        const Teuchos::ParameterList& sdynparams,                 //!< input parameters
        const Teuchos::ParameterList& xparams,                    //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,         //!< current discretisation
        std::shared_ptr<Core::LinAlg::Solver> solver,             //!< the solver
        std::shared_ptr<Core::LinAlg::Solver> contactsolver,  //!< the solver for contact meshtying
        std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< the output
    );

    //! Destructor
    // ....

    /*! \brief Initialize this object

    Hand in all objects/parameters/etc. from outside.
    Construct and manipulate internal objects.

    \note Try to only perform actions in init(), which are still valid
          after parallel redistribution of discretizations.
          If you have to perform an action depending on the parallel
          distribution, make sure you adapt the affected objects after
          parallel redistribution.
          Example: cloning a discretization from another discretization is
          OK in init(...). However, after redistribution of the source
          discretization do not forget to also redistribute the cloned
          discretization.
          All objects relying on the parallel distribution are supposed to
          the constructed in \ref setup().

    \warning none
    \return bool

    */
    void init(const Teuchos::ParameterList& timeparams, const Teuchos::ParameterList& sdynparams,
        const Teuchos::ParameterList& xparams, std::shared_ptr<Core::FE::Discretization> actdis,
        std::shared_ptr<Core::LinAlg::Solver> solver) override;

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning none
    \return void

    */
    void setup() override;

    //! Resize #TimIntMStep<T> multi-step quantities
    //! Single-step method: nothing to do here
    void resize_m_step() override { ; }

    //@}

    //! @name Pure virtual methods which have to be implemented
    //@{

    //! Return name
    enum Inpar::Solid::DynamicType method_name() const override
    {
      return Inpar::Solid::DynamicType::OneStepTheta;
    }

    //! Provide number of steps, a single-step method returns 1
    int method_steps() const override { return 1; }

    //! Give local order of accuracy of displacement part
    int method_order_of_accuracy_dis() const override
    {
      return fabs(method_lin_err_coeff1()) < 1e-6 ? 2 : 1;
    }

    //! Give local order of accuracy of velocity part
    int method_order_of_accuracy_vel() const override { return method_order_of_accuracy_dis(); }

    //! Return linear error coefficient of displacements
    double method_lin_err_coeff_dis() const override
    {
      if (method_order_of_accuracy_dis() == 1)
        return method_lin_err_coeff1();
      else
        return method_lin_err_coeff2();
    }

    //! Return linear error coefficient of velocities
    double method_lin_err_coeff_vel() const override { return method_lin_err_coeff_dis(); }

    //! Linear error coefficient if 1st order accurate
    virtual double method_lin_err_coeff1() const { return 1. / 2. - theta_; }

    //! Linear error coefficient if 2nd order accurate
    virtual double method_lin_err_coeff2() const
    {
      return 1. / 6. - theta_ / 2.;  // this is -1/12
    }

    //! return time integration factor
    double tim_int_param() const override { return 1.0 - theta_; }

    //! Consistent predictor with constant displacements
    //! and consistent velocities and displacements
    void predict_const_dis_consist_vel_acc() override;

    //! Consistent predictor with constant velocities,
    //! extrapolated displacements and consistent accelerations
    void predict_const_vel_consist_acc() override;

    //! Consistent predictor with constant accelerations
    //! and extrapolated velocities and displacements
    void predict_const_acc() override;

    //! Create force residual #fres_ and its stiffness #stiff_
    void evaluate_force_stiff_residual(Teuchos::ParameterList& params) final;

    //! Evaluate/define the residual force vector #fres_ for
    //! relaxation solution with solve_relaxation_linear
    void evaluate_force_stiff_residual_relax(Teuchos::ParameterList& params) override;

    //! Evaluate residual #fres_
    void evaluate_force_residual() override;

    //! Determine characteristic norm for force
    double calc_ref_norm_force() override;

    //! Update iteration incrementally
    //!
    //! This update is carried out by computing the new #veln_ and #acc_ from scratch by using the
    //! newly updated #disn_.
    //! This method is necessary for certain predictors (like #predict_const_dis_consist_vel_acc)
    void update_iter_incrementally() override;

    //! Update iteration iteratively
    //!
    //! This is the ordinary update of #disn_, #veln_ and #accn_ by
    //! incrementing these vector proportional to the residual
    //! displacements #disi_
    //! The Dirichlet BCs are automatically respected, because the
    //! residual displacements #disi_ are blanked at these DOFs.
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

    //! @name One-Step-Theta specific methods
    //@{

    //! Evaluate mid-state vectors by averaging end-point vectors
    void evaluate_mid_state();

    //@}

   protected:
    //! equal operator is NOT wanted
    TimIntOneStepTheta operator=(const TimIntOneStepTheta& old);

    //! copy constructor is NOT wanted
    TimIntOneStepTheta(const TimIntOneStepTheta& old);

    //! Clear mass matrix and evaluate mass matrix again.
    //! \note not implemented in base class.
    void determine_mass() override;

    //! @name Key coefficients
    //@{
    double theta_;  //!< factor (0,1]
    //@}

    //! @name Global mid-state vectors
    //@{
    //! mid-displacements \f$D_m = D_{n+\theta}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> dist_;
    //! mid-velocities \f$V_m = V_{n+1-\theta}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> velt_;
    //! mid-accelerations \f$A_m = A_{n+1-\theta}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> acct_;
    //@}

    //! @name Global force vectors
    //! Residual \c fres_ exists already in base class
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fint_;   //!< internal force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fintn_;  //!< internal force at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>> fext_;   //!< external force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fextn_;  //!< external force at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>> finert_;   //!< inertia force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> finertt_;  //!< inertia mid-force
    std::shared_ptr<Core::LinAlg::Vector<double>> finertn_;  //!< inertia force at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>> fvisct_;  //!< viscous force

    //@}

  };  // class TimIntOneStepTheta

}  // namespace Solid

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
