// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_TIMINT_STATICS_HPP
#define FOUR_C_THERMO_TIMINT_STATICS_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 06/09 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_thermo_timint_impl.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | belongs to thermal dynamics namespace                    bborn 08/09 |
 *----------------------------------------------------------------------*/
namespace Thermo
{
  /*====================================================================*/
  /*!
   * \brief Static analysis
   *
   * This static analysis inside the thermal dynamics methods appears
   * slightly displaced, however, it comes in handy in case of
   * thermo-structure-interaction, which is built upon thermal
   * dynamics.
   *
   * Regarding this matter, please direct any complaints to Axel Gerstenberger.
   *
   * \author bborn
   * \date 06/08
   */
  class TimIntStatics : public TimIntImpl
  {
   public:
    //! @name Construction
    //@{

    //! Constructor
    TimIntStatics(const Teuchos::ParameterList& ioparams,       //!< ioflags
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
    enum Thermo::DynamicType method_name() const override { return Thermo::dyna_statics; }

    //! Consistent predictor with constant temperatures
    //! and consistent temperature rates and temperatures
    void predict_const_temp_consist_rate() override;

    //! Evaluate ordinary internal force, its tangent at state
    void apply_force_tang_internal(const double time,               //!< evaluation time
        const double dt,                                            //!< step size
        const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state
        const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< residual temperatures
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

    //! Evaluate/define the residual force vector #fres_ for
    //! relaxation solution with solve_relaxation_linear
    // void EvaluateForceTangResidualRelax();

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

   protected:
    //! equal operator is hidden
    TimIntStatics operator=(const TimIntStatics& old);

    //! copy constructor is hidden
    TimIntStatics(const TimIntStatics& old);

    //! @name Global force vectors
    //! Residual \c fres_ exists already in base class
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> fint_;   //!< internal force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fintn_;  //!< internal force at \f$t_{n+1}\f$

    std::shared_ptr<Core::LinAlg::Vector<double>> fext_;   //!< external force at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fextn_;  //!< external force at \f$t_{n+1}\f$
    //@}

  };  // class TimIntStatics

}  // namespace Thermo


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
