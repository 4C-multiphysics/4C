// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_STR_TIMEADA_HPP
#define FOUR_C_ADAPTER_STR_TIMEADA_HPP

#include "4C_config.hpp"

#include "4C_adapter_str_wrapper.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Solid::TimeInt
{
  class Base;
}  // namespace Solid::TimeInt

namespace Global
{
  class Problem;
}  // namespace Global

namespace Adapter
{
  /*! \brief Adaptive time loop for structural simulations
   *
   *  This is a wrapper for the structural time integration which gives
   *  fine-grained access into the time loop by various pre- and post-operators.
   *
   *  To perform such pre- and post-operations, just derive from this class and
   *  overload the respective pre-/post-operator.
   *
   *  Implementations of pre-/post-operators in this class have to remain empty!
   */
  class StructureTimeAda : public StructureWrapper
  {
   public:
    //! List type of local error control
    enum CtrlEnum
    {
      ctrl_dis,         //!< check only displacements
      ctrl_vel,         //!< check only velocities
      ctrl_dis_and_vel  //!< check displacements and velocities
    };

    //! Type of adaptivity algorithm
    enum AdaEnum
    {
      ada_vague,     //!< algorithm is unknown
      ada_upward,    //!< of upward type, i.e. auxiliary scheme has \b higher order of accuracy than
                     //!< marching scheme
      ada_downward,  //!< of downward type, i.e. auxiliary scheme has \b lower order of accuracy
                     //!< than marching scheme
      ada_orderequal,  //!< of equal order type, i.e. auxiliary scheme has the \b same order of
                       //!< accuracy like the marching method
      ada_ident        //!< auxiliary scheme is \b identical to marching scheme
    };

    /// constructor
    StructureTimeAda(Global::Problem& problem, std::shared_ptr<Structure> structure);

    /*! \brief Utility function to create the adaptive time integration structure wrapper
     */
    static std::shared_ptr<Structure> create(Global::Problem& problem,
        const Teuchos::ParameterList& taflags,  //!< adaptive input flags
        std::shared_ptr<Solid::TimeInt::Base> ti_strategy);

    /// setup of the adaptive time integration
    void setup() override;

    /// read restart information for given time step
    void read_restart(int step) override;

    /// actual time loop
    int integrate() override;

    /// wrapper for things that should be done before prepare_time_step is called
    void pre_predict() override {};

    /// wrapper for things that should be done before solving the nonlinear iterations
    void pre_solve() override {};

    /// wrapper for things that should be done before updating
    void pre_update() override {};

    /// wrapper for things that should be done after solving the update
    void post_update() override {};

    /// output results
    void output(bool forced_writerestart = false) override;

    /// wrapper for things that should be done after the output
    void post_output() override
    {
      // first forward to base class which forwards to structure_->post_output();
      StructureWrapper::post_output();
    };

    //! Provide the name
    virtual Solid::TimAdaKind method_name() const = 0;

    //! Provide the name as std::string
    virtual std::string method_title() const = 0;

    //! Provide local order of accuracy based upon linear test equation
    //! for displacements
    virtual int method_order_of_accuracy_dis() const = 0;

    //! Provide local order of accuracy based upon linear test equation
    //! for velocities
    virtual int method_order_of_accuracy_vel() const = 0;

    //! Return linear error coefficient of displacements
    virtual double method_lin_err_coeff_dis() const = 0;

    //! Return linear error coefficient of velocities
    virtual double method_lin_err_coeff_vel() const = 0;

    //! Provide type of algorithm
    virtual AdaEnum method_adapt_dis() const = 0;

   protected:
    std::shared_ptr<Solid::TimeInt::Base> stm_;  //!< marching time integrator

    //! @name Plain time integration constants
    //@{
    double timeinitial_;      //!< initial time: t_0
    double timefinal_;        //!< final time
    int timedirect_;          //!< +1: in positive, -1: in negative time direction
    int timestepinitial_;     //!< initial time step index: 0 (often)
    int timestepfinal_;       //!< maximum time step: n_max
    double stepsizeinitial_;  //!< initial step size: dt_n
    //@}

    //! @name Adaptive time integration constants
    //@{
    double stepsizemax_;         //!< maximum time step size (upper limit)
    double stepsizemin_;         //!< minimum time step size (lower limit)
    double sizeratiomax_;        //!< maximally permitted increase of current step size
                                 //!< relative to last converged one
    double sizeratiomin_;        //!< minimally permitted increase
                                 //!< (or maximally permitted decrease)
                                 //!< of current step size relative to last converged one
    double sizeratioscale_;      //!< safety factor, should be lower than 1.0
    CtrlEnum errctrl_;           //!< type of control, see #CtrlEnum
    Solid::VectorNorm errnorm_;  //!< norm for local error vector
    double errtol_;              //!< target local error tolerance
    int errorder_;               //!< order of local error indication
    int adaptstepmax_;           //!< maximally permitted trials to find tolerable step size
    //@}

    //! @name plain time integration variables
    //@{
    double time_;   //!< current time \f$t_n\f$
    int timestep_;  //!< current time step \f$n\f$
    //@}

    //! @name Adaptive time integration variables
    //@{
    double stepsizepre_;  //!< previous time step size \f$\Delta t_{n-1}\f$
    double stepsize_;     //!< current time step size \f$\Delta t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> locerrdisn_;  //!< current local disp. error
                                                                //!< estimation \f$l_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> locerrveln_;  //!< current local vel. error
                                                                //!< estimation \f$\dot{l}_{n+1}\f$
    int adaptstep_;  //!< trial counter, cf. #adaptstepmax_
                     //@}

    //! @name Output settings
    //@{
    bool outsys_;                                 //!< do it this step: write system to file
    bool outstr_;                                 //!< do it this step: write stress/strain to file
    bool outene_;                                 //!< do it this step: write energy to file
    bool outrest_;                                //!< do it this step: write restart data to file
    double outsysperiod_;                         //!< print system (dis,vel,acc,...)
                                                  //!< every given period of time
    double outstrperiod_;                         //!< print stress/strain every given
                                                  //!< period of time
    double outeneperiod_;                         //!< print energies every given
                                                  //!< period of time
    double outrestperiod_;                        //!< print restart every given
                                                  //!< period of time
    int outsizeevery_;                            //!< print step size every given step
    double outsystime_;                           //!< next output time point for system
    double outstrtime_;                           //!< next output time point for stress/strain
    double outenetime_;                           //!< next output time point for energy
    double outresttime_;                          //!< next output time point for restart
    std::shared_ptr<std::ofstream> outsizefile_;  //!< outputfile for step sizes
                                                  //@}

   protected:
    /*! \brief Prepare repetition of current time step
     *
     *  Print to screen and reset certain quantities in case that the current time
     *  step has to be repeated.
     *
     */
    void reset_step() override;

    /// setup of the auxiliary time integrator
    virtual void setup_auxiliary() = 0;

    /// explicit global problem context
    Global::Problem& problem() { return problem_; }
    const Global::Problem& problem() const { return problem_; }

   private:
    /// explicit global problem context
    Global::Problem& problem_;

    //! Setup necessities for time adaptivity
    void setup_time_ada();

    /*! \brief Make one step with auxiliary scheme
     *
     *  Afterwards, the auxiliary solutions are stored in the local error
     *  vectors, ie:
     *  - \f$D_{n+1}^{AUX}\f$ in #locdiserrn_
     *  - \f$V_{n+1}^{AUX}\f$ in #locvelerrn_
     */
    virtual void integrate_step_auxiliary() = 0;

    /*! \brief Update the auxiliary integrator
     */
    virtual void update_auxiliary() = 0;

    //! Modify step size to hit precisely output period
    void size_for_output();

    //!  Update output periods
    void update_period();

    //! Indicate error and determine new step size
    void indicate(bool& accepted,  //!< true=accepted, false=not accepted
        double& stpsiznew          //!< step size prediction for next step or step repetition
    );

    /*! \brief Compute local discretisation error
     *
     *  Compute the local discretisation error vector of displacements/velocities
     *  specific to marching/auxiliary time integrator pair.
     *
     *  \note Solution of auxiliary step is already stored in ##locdiserrn_/#locvelerrn_,
     *  so we just need to subtract the solution of the marching TIS.
     */
    void evaluate_local_error_dis();

    /*! \brief Calculate time step size
     *
     *  Using the ratio of the desired tolerance \f$tol\f$ (#errtol_) to the
     *  estimated local discretization error, an optimal scaling
     *  factor \f$\kappa_{opt}\f$ is computed, such that the user given error
     *  tolerance is met 'exactly'.
     *  \f[
     *    \kappa_{opt} = \left(\frac{tol}{\vert error\vert}\right)^\frac{1}{p+1}
     *  \f]
     *  To reduce the number of time step repetitions, the scaling factor is
     *  reduced by a safety factor \f$\kappa_{safe} \in [0, 1]\f$ (#sizeratioscale_)
     *  to hopefully keep the achieved local discretization error a little bit
     *  below the tolerance.
     *
     *  Starting with the current time step size \f$\Delta t_{curr}\f$ (#stepsize_),
     *  the new time step size is computed as
     *  \f[
     *    \Delta t_{new} = \kappa_{opt} \cdot \kappa_{safe} \cdot \Delta t_{curr}
     *  \f]
     *
     *  Now, we update the actual scaling factor
     *  \f$\kappa_{eff} = \Delta t_{new} / \Delta t^{n-1}\f$,
     *  limit it by upper and lower bounds (#sizeratiomax_ and #sizeratiomin_)
     *  and recompute the new time step size, if necessary. Finally, we make sure
     *  that the new time step size also satisfies upper and lower bounds (#stepsizemax_
     *  and #stepsizemin_).
     *
     */
    virtual double calculate_dt(const double norm  ///< current norm of local discretization error
    );

    /// Determine norm of force residual
    double calculate_vector_norm(const Solid::VectorNorm norm,  ///< type of norm to use
        Core::LinAlg::Vector<double>& vect,                     ///< the vector of interest
        const int numneglect =
            0  ///< number of DOFs that have to be neglected for possible length scaling
    );

    /// Perform error action once the nonlinear iteration fails
    Solid::ConvergenceStatus perform_error_action(
        const Solid::DivContAct& action, double& stepsizenew);
  };

}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
