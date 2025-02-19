// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_TIMINT_GENALPHA_HPP
#define FOUR_C_THERMO_TIMINT_GENALPHA_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 05/13 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_thermo_timint_impl.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | belongs to thermal dynamics namespace                     dano 05/13 |
 *----------------------------------------------------------------------*/
namespace Thermo
{
  /*====================================================================*/
  /*!
   * \brief Generalised-alpha time integration
   *
   * References
   * - [1] J Chung and GM Hulbert, A time integration algorithm for structural
   *   dynamics with improved numerical dissipation: the generalized-alpha method
   *   Journal of Applied Mechanics, 60:371-375, 1993.
   *
   * temporal discretisation of a first order ODE according to
   * - [2] KE Jansen, CH Whiting and GM Hulbert, A generalized-alpha
   *   method for integrating the filtered Navier-Stokes equations with a
   *   stabilized finite element method, Computer Methods in Applied Mechanics
   *   and Engineering, 190:305-319, 2000.
   *
   *
   * \author danowski
   * \date 06/13
   */
  class TimIntGenAlpha : public TimIntImpl
  {
   public:
    //! verify if given coefficients are in admissible range
    //! prints also info to STDOUT
    void verify_coeff();

    //! calculate coefficients for given spectral radius
    void calc_coeff();

    //! @name Construction
    //@{

    //! Constructor
    TimIntGenAlpha(const Teuchos::ParameterList& ioparams,      //!< ioflags
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
    enum Thermo::DynamicType method_name() const override { return Thermo::dyna_genalpha; }

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

    //! Update Element
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
    TimIntGenAlpha operator=(const TimIntGenAlpha& old);

    //! copy constructor is NOT wanted
    TimIntGenAlpha(const TimIntGenAlpha& old);

    //! @name set-up
    //@{
    //! mid-average type more at #MidAverageEnum
    enum Thermo::MidAverageEnum midavg_;
    //@}

    //! @name Key coefficients
    //! Please note, to obtain a second-order accurate scheme, you need
    //! to follow the following formulas in which \f$\rho_\infty\f$ is the
    //! spectral radius.
    //! \f[ \alpha_m = 1/2 (3 - \rho_\infty)/(\rho_\infty + 1) \f]
    //! \f[ \alpha_f = 1/(\rho_\infty + 1) \f]
    //! \f[ \gamma = 1/2 + \alpha_m - \alpha_f \f]
    //! The spectral radius is responsible for the magnitude of numerical
    //! dissipation introduced.
    //! For instance
    //!
    //! Without numerical dissipation at \f$\rho_\infty=1\f$: 2nd order mid-point rule
    //! \f[ \alpha_f=0.5, \alpha_m=0.5, \gamma=0.5 \f]
    //!
    //! Strong numerical dissipation at \f$\rho_\infty=0.5\f$: default
    //! \f[ \alpha_f=2/3, \alpha_m=5/6, \gamma=2/3 \f]
    //!
    //! Maximal numerical dissipation at \f$\rho_\infty=0.0\f$: BDF2
    //! \f[ \alpha_f=1, \alpha_m=3/2, \gamma=1 \f]
    //@{
    double gamma_;    //!< factor (0,1]
    double alphaf_;   //!< factor [0,1]
    double alpham_;   //!< factor [0,1.5]
    double rho_inf_;  //!< factor[0,1]
    //@}

    //! @name Global mid-state vectors
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> tempm_;  //!< mid-temperatures
                                                           //!< \f$T_m = T_{n+\alpha_f}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> ratem_;  //!< mid-temperature rates
                                                           //!< \f$R_m = R_{n+\alpha_m}\f$
    //@}

    //! @name Global force vectors
    //! Residual \c fres_ exists already in base class
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fint_;  //!< internal force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        fintm_;  //!< internal mid-force at \f$t_{n+\alpha_f}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fintn_;  //!< internal force at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>> fext_;  //!< external force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        fextm_;  //!< external mid-force \f$t_{n+\alpha_f}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fextn_;  //!< external force at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>>
        fcap_;  //!< capacity force \f$C\cdot\Theta_n\f$ at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        fcapm_;  //!< capacity force \f$C\cdot\Theta_{n+\alpha_m}\f$ at \f$t_{n+\alpha_m}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>>
        fcapn_;  //!< capacity force \f$C\cdot\Theta_{n+1}\f$ at \f$t_{n+1}\f$

    //@}

  };  // class TimIntGenAlpha

}  // namespace Thermo


FOUR_C_NAMESPACE_CLOSE

#endif
