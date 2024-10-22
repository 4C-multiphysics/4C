// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_TIMINT_EXPL_HPP
#define FOUR_C_THERMO_TIMINT_EXPL_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_thermo_timint.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* belongs to thermal dynamics namespace */
namespace Thermo
{
  /*====================================================================*/
  /*!
   * \brief Front-end for structural dynamics
   *        with \b explicit time integrators
   *
   * <h3> About </h3>
   * This object bridges the gap between the base time integator Thermo::TimInt
   * and the specific implementation of explicit time integrators.
   *
   * \author bborn
   * \date 07/08
   */
  class TimIntExpl : public TimInt
  {
   public:
    //! @name Life
    //@{

    //! constructor
    TimIntExpl(const Teuchos::ParameterList& ioparams,       //!< ioflags
        const Teuchos::ParameterList& tdynparams,            //!< input parameters
        const Teuchos::ParameterList& xparams,               //!< extra flags
        Teuchos::RCP<Core::FE::Discretization> actdis,       //!< current discretisation
        Teuchos::RCP<Core::LinAlg::Solver> solver,           //!< the solver
        Teuchos::RCP<Core::IO::DiscretizationWriter> output  //!< the output
    );

    //! Empty constructor
    TimIntExpl() : TimInt() { ; }

    //! Copy constructor
    TimIntExpl(const TimIntExpl& old) : TimInt(old) { ; }

    //! Resize #TimIntMStep<T> multi-step quantities
    void resize_m_step() override = 0;

    //@}

    //! @name Actions
    //@{

    //! Do time integration of single step
    void integrate_step() override = 0;

    //! Solve dynamic equilibrium
    //! This is a general wrapper around the specific techniques.
    Inpar::Thermo::ConvergenceStatus solve() override
    {
      integrate_step();
      return Inpar::Thermo::conv_success;
    }

    //! build linear system tangent matrix, rhs/force residual
    //! Monolithic TSI accesses the linearised thermo problem
    void evaluate(Teuchos::RCP<const Core::LinAlg::Vector<double>> tempi) override
    {
      FOUR_C_THROW("not implemented for explicit time integration");
      return;
    }

    //! build linear system tangent matrix, rhs/force residual
    //! Monolithic TSI accesses the linearised thermo problem
    void evaluate() override
    {
      FOUR_C_THROW("not implemented for explicit time integration");
      return;
    }

    //! prepare time step
    void prepare_time_step() override
    {
      // do nothing
      return;
    }

    //! for implicit partitioned schemes
    void prepare_partition_step() override
    {
      // do nothing
      return;
    }

    //! Update configuration after time step
    //!
    //! Thus the 'last' converged is lost and a reset of the time step
    //! becomes impossible. We are ready and keen awating the next time step.
    void update_step_state() override = 0;

    //! Update Element
    void update_step_element() override = 0;

    //! update at time step end
    void update() override;

    //! update Newton step
    void update_newton(Teuchos::RCP<const Core::LinAlg::Vector<double>> tempi) override
    {
      FOUR_C_THROW("not needed for explicit time integration");
      return;
    }
    /*
        //! Update configuration and time after time step
        void UpdateStepAndTime()
        {
          // system state
          update_step_state();
          // update time and step
          time_->UpdateSteps(timen_);
          step_ = stepn_;
          //
          timen_ += (*dt_)[0];
          stepn_ += 1;
          // element update
          update_step_element();
        }
    */
    //@}

    //! @name Output
    //@{

    //! print summary after step
    void print_step() override;

    //! The text for summary print, see #print_step
    void print_step_text(FILE* ofile  //!< output file handle
    );

    //@}

    //! @name Attribute access functions
    //@{

    //! Return time integrator name
    enum Inpar::Thermo::DynamicType method_name() const override = 0;

    //! These time integrators are all explicit (mark their name)
    bool method_implicit() override { return false; }

    //! Provide number of steps, e.g. a single-step method returns 1,
    //! a m-multistep method returns m
    int method_steps() override = 0;

    //! Give local order of accuracy of displacement part
    int method_order_of_accuracy() override = 0;

    //! Return linear error coefficient of temperatures
    double method_lin_err_coeff() override = 0;

    //@}

    //! @name System vectors
    //@{

    //! Return external force \f$F_{ext,n}\f$
    Teuchos::RCP<Core::LinAlg::Vector<double>> fext() override = 0;

    //! Return reaction forces
    Teuchos::RCP<Core::LinAlg::Vector<double>> freact() override
    {
      FOUR_C_THROW("Not impl.");
      return Teuchos::null;
    };

    //! initial guess of Newton's method
    Teuchos::RCP<const Core::LinAlg::Vector<double>> initial_guess() override
    {
      FOUR_C_THROW("not needed for explicit time integration");
      return Teuchos::null;
    }

    //! right-hand side alias the dynamic force residual
    Teuchos::RCP<const Core::LinAlg::Vector<double>> rhs() override
    {
      FOUR_C_THROW("not needed for explicit time integration");
      return Teuchos::null;
    }

    //! Read and set external forces from file
    void read_restart_force() override = 0;

    //! Write internal and external forces for restart
    void write_restart_force(Teuchos::RCP<Core::IO::DiscretizationWriter> output) override = 0;

    //@}

   protected:
    // currently nothing
  };

}  // namespace Thermo

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
