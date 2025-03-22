// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMADA_HPP
#define FOUR_C_STRUCTURE_TIMADA_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class Solver;
}

namespace Core::IO
{
  class DiscretizationWriter;
}


/*----------------------------------------------------------------------*/
/* belongs to structure dynamics namespace */
namespace Solid
{
  // forward declarations
  class TimInt;

  /*====================================================================*/
  /*!
   * \brief Front-end for time step size adaptivity in structural dynamics
   *
   * Time step size adaptivity is based on <e>a posteriori</e> error indicators/
   * estimators. Therefore, the solution obtained with the marching time integrator
   * is compared to one obtained with an auxiliary time integrator. Based on this
   * estimation of the local truncation error, a new time step size is suggested.
   *

   */
  class TimAda
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

    //! Constructor
    TimAda(const Teuchos::ParameterList& timeparams,  //!< TIS input parameters
        const Teuchos::ParameterList& tap,            //!< adaptive input flags
        std::shared_ptr<TimInt> tis                   //!< marching time integrator
    );

    //! Destructor
    virtual ~TimAda() = default;
    //! @name Actions
    //@{

    /*! \brief Integrate in time
     * This is the key method here, i.e. the time integration algorithm.
     */
    int integrate();

    /*! Finalize the class initialization
     * Merge() and ResizeMStep() need to be called after(!) both init()
     * and setup() have been called on both the marching time integrator
     * and the auxiliary time integrator if existing (popp 01/2017).
     */
    virtual void init(std::shared_ptr<TimInt>& sti) = 0;

    /*! \brief Make one step with auxiliary scheme
     *
     *  Afterwards, the auxiliary solutions are stored in the local error
     *  vectors, ie:
     *  - \f$D_{n+1}^{AUX}\f$ in #locdiserrn_
     *  - \f$V_{n+1}^{AUX}\f$ in #locvelerrn_
     */
    virtual void integrate_step_auxiliary() = 0;

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

    /*! \brief Prepare repetition of current time step
     *
     *  Print to screen and reset certain quantities in case that the current time
     *  step has to be repeated.
     *
     */
    virtual void reset_step();

    //@}

    //! @name Access routines
    //@{

    //! get the vector of the local discretization error
    std::shared_ptr<Core::LinAlg::Vector<double>>& loc_err_dis() { return locerrdisn_; }

    //@}

    //! @name Output
    //@{

    //! Print error norm string
    std::string print_err_norm() const;

    //! Print time adapting constants
    void print_constants(std::ostream& str  //!< output stream
    ) const;

    //! Print time adapting variables
    void print_variables(std::ostream& str  //!< output stream
    ) const;

    //! Print time adapting parameters:TimeIntegrator
    void print(std::ostream& str  //!< output stream
    ) const;

    //! Modify step size to hit precisely output period
    void size_for_output();

    //! Prepare output to file(s)
    void prepare_output_period(bool force_prepare);

    //! Output to file(s)
    void output_period();

    //!  Update output periods
    void update_period();

    //! Set new step size
    virtual void set_dt(const double dtnew);

    //! Update step size
    virtual void update_step_size();

    //! Update step size
    virtual void update_step_size(const double dtnew);

    //! Access to current time step size
    virtual double dt() const { return stepsize_; }

    //! Access to target time \f$t_{n+1}\f$ of current time step
    virtual double time() const { return time_ + stepsize_; }

    //! Check whether step size output file is attached
    bool attached_file_step_size()
    {
      if (outsizefile_)
        return true;
      else
        return false;
    }

    //! Attach file handle for step size file #outsizefile_
    void attach_file_step_size();


    //! Write step size
    void output_step_size();

    //@}

    //! @name Attributes
    //@{

    //! Provide the name
    virtual enum Inpar::Solid::TimAdaKind method_name() const = 0;

    //! Provide the name as std::string
    std::string method_title() const;

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
    virtual enum AdaEnum method_adapt_dis() const = 0;

    //@}

    //! use contact solver or not
    bool use_contact_solver() { return false; };

   protected:
    //! not wanted: copy constructor
    TimAda(const TimAda& old);

    //! A revolutionary routine to get --well-- the sign of a number.
    static int sign(const double number  //!< a real number
    )
    {
      return (number == 0.0) ? 0 : (number > 0.0) ? +1 : -1;
    }

    //! @name General purpose algorithm members
    //@{
    std::shared_ptr<TimInt> sti_;                             //!< marching time integrator
    std::shared_ptr<Core::FE::Discretization> discret_;       //!< attached discretisation
    int myrank_;                                              //!< processor ID
    std::shared_ptr<Core::LinAlg::Solver> solver_;            //!< linear algebraic solver
    std::shared_ptr<Core::IO::DiscretizationWriter> output_;  //!< binary output
    //@}

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
    double stepsizemax_;     //!< maximum time step size (upper limit)
    double stepsizemin_;     //!< minimum time step size (lower limit)
    double sizeratiomax_;    //!< maximally permitted increase of current step size
                             //!< relative to last converged one
    double sizeratiomin_;    //!< minimally permitted increase
                             //!< (or maximally permitted decrease)
                             //!< of current step size relative to last converged one
    double sizeratioscale_;  //!< safety factor, should be lower than 1.0
    enum CtrlEnum errctrl_;  //!< type of control, see #CtrlEnum
    enum Inpar::Solid::VectorNorm errnorm_;  //!< norm for local error vector
    double errtol_;                          //!< target local error tolerance
    int errorder_;                           //!< order of local error indication
    int adaptstepmax_;  //!< maximally permitted trials to find tolerable step size
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

  };  // class TimAda

}  // namespace Solid

/*======================================================================*/
/*!
 * \brief Out stream inserter for Solid::TimAda
 *
 */
std::ostream& operator<<(std::ostream& str, const Solid::TimAda& ta);


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
