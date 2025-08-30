// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMINT_EXPL_HPP
#define FOUR_C_STRUCTURE_TIMINT_EXPL_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_structure_timint.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* belongs to structural dynamics namespace */
namespace Solid
{
  namespace Aux
  {
    class MapExtractor;
  }

  /*====================================================================*/
  /*!
   * \brief Front-end for structural dynamics
   *        with \b explicit time integrators
   *
   * <h3> About </h3>
   * This object bridges the gap between the base time integator Solid::TimInt
   * and the specific implementation of explicit time integrators.
   *

   */
  class TimIntExpl : public TimInt
  {
   public:
    //! @name Life
    //@{

    //! constructor
    TimIntExpl(const Teuchos::ParameterList& timeparams,      //!< time parameters
        const Teuchos::ParameterList& ioparams,               //!< ioflags
        const Teuchos::ParameterList& sdynparams,             //!< input parameters
        const Teuchos::ParameterList& xparams,                //!< extra flags
        std::shared_ptr<Core::FE::Discretization> actdis,     //!< current discretisation
        std::shared_ptr<Core::LinAlg::Solver> solver,         //!< the solver
        std::shared_ptr<Core::LinAlg::Solver> contactsolver,  //!< the solver for contact meshtying
        std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< the output
    );


    //! Copy constructor
    TimIntExpl(const TimIntExpl& old) : TimInt(old) { ; }

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

    //@}

    //! @name Actions
    //@{

    //! Resize \p TimIntMStep<T> multi-step quantities
    void resize_m_step() override = 0;

    //! Do time integration of single step
    int integrate_step() override = 0;

    //! Update configuration after time step
    //!
    //! Thus the 'last' converged is lost and a reset of the time step
    //! becomes impossible. We are ready and keen awaiting the next time step.
    void update_step_state() override = 0;

    //! Update Element
    void update_step_element() override = 0;
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
    enum Inpar::Solid::DynamicType method_name() const override = 0;

    //! These time integrators are all explicit (mark their name)
    bool method_implicit() override { return false; }

    //! Provide number of steps, e.g. a single-step method returns 1,
    //! a m-multistep method returns m
    int method_steps() const override = 0;

    //! Give local order of accuracy of displacement part
    int method_order_of_accuracy_dis() const override = 0;

    //! Give local order of accuracy of velocity part
    int method_order_of_accuracy_vel() const override = 0;

    //! Return linear error coefficient of displacements
    double method_lin_err_coeff_dis() const override = 0;

    //! Return linear error coefficient of velocities
    double method_lin_err_coeff_vel() const override = 0;

    //! return time integration factor
    double tim_int_param() const override { return 0.0; }

    //@}

    //! @name System vectors
    //@{

    //! Return external force \f$F_{ext,n}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> fext() override = 0;

    //! Return reaction forces
    std::shared_ptr<Core::LinAlg::Vector<double>> freact() override
    {
      FOUR_C_THROW("Not impl.");
      return nullptr;
    };

    //! Read and set external forces from file
    void read_restart_force() override = 0;

    //! Write internal and external forces for restart
    void write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output) override = 0;

    //! initial_guess is not available for explicit time integrators
    std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() override
    {
      FOUR_C_THROW("initial_guess() is not available for explicit time integrators");
      return nullptr;
    }

    //! RHS() is not available for explicit time integrators
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override
    {
      FOUR_C_THROW("RHS() is not available for explicit time integrators");
      return nullptr;
    }

    //! Prepare time step
    void prepare_time_step() override
    {
      // safety checks
      check_is_init();
      check_is_setup();

      // update end time \f$t_{n+1}\f$ of this time step to cope with time step size adaptivity
      set_timen((*time_)[0] + (*dt_)[0]);

      // things that need to be done before Predict
      pre_predict();

      // prepare contact for new time step
      prepare_step_contact();

      return;
    }

    //!  Update displacement state in case of coupled problems
    void update_state_incrementally(
        std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc) override
    {
      FOUR_C_THROW(
          "All monolithically coupled problems work with implicit time "
          "integration schemes. Thus, calling update_state_incrementally() in an explicit scheme "
          "is not possible.");
    }

    //!  Evaluate routine for coupled problems with monolithic approach
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
            disiterinc  ///< iterative solution increment
        ) override
    {
      FOUR_C_THROW(
          "All monolithically coupled problems work with implicit time "
          "integration schemes. Thus, calling evaluate() in an explicit scheme "
          "is not possible.");
    }

    //! Apply external force
    void apply_force_external(const double time,                  //!< evaluation time
        const std::shared_ptr<Core::LinAlg::Vector<double>> dis,  //!< displacement state
        Core::LinAlg::Vector<double>& vel,                        // velocity state
        Core::LinAlg::Vector<double>& fext                        //!< external force
    );

    /// has to be renamed either here or print_step()
    void output(bool forced_writerestart) override
    {
      output_step(forced_writerestart);
      // write Gmsh output
      write_gmsh_struct_output_step();
      return;
    }

    /// has to be renamed either here or update_step_state() /UpdateStepStateElement()
    void update() override
    {
      pre_update();
      update_step_state();
      update_step_time();
      update_step_element();
      post_update();
      return;
    }

    //! Update routine for coupled problems with monolithic approach with time adaptivity
    void update(const double endtime) override
    {
      FOUR_C_THROW("Not implemented. No time adaptivity available for explicit time integration.");
    }


    /* Linear structure solve with just an interface load */
    std::shared_ptr<Core::LinAlg::Vector<double>> solve_relaxation_linear() override
    {
      FOUR_C_THROW("solve_relaxation_linear() not implemented");
      return nullptr;
    }

    /// are there any algebraic constraints?
    bool have_constraint() override
    {
      FOUR_C_THROW("HaveConstraint() has not been tested for explicit time integrators");
      return false;
    };

    /// are there any Cardiovascular0D bcs?
    virtual bool have_cardiovascular0_d()
    {
      FOUR_C_THROW("have_cardiovascular0_d() has not been tested for explicit time integrators");
      return false;
    };

    /// are there any spring dashpot BCs?
    bool have_spring_dashpot() override
    {
      FOUR_C_THROW("HaveSpringDashpot() has not been tested for explicit time integrators");
      return false;
    };

    //! Return Teuchos::rcp to ConstraintManager conman_
    std::shared_ptr<Constraints::ConstrManager> get_constraint_manager() override
    {
      FOUR_C_THROW("get_constraint_manager() has not been tested for explicit time integrators");
      return nullptr;
    };

    //! Return Teuchos::rcp to Cardiovascular0DManager windkman_
    virtual std::shared_ptr<Utils::Cardiovascular0DManager> get_cardiovascular0_d_manager()
    {
      FOUR_C_THROW(
          "get_cardiovascular0_d_manager() has not been tested for explicit time integrators");
      return nullptr;
    };

    //! Return Teuchos::rcp to SpringDashpotManager springman_
    std::shared_ptr<Constraints::SpringDashpotManager> get_spring_dashpot_manager() override
    {
      FOUR_C_THROW(
          "get_spring_dashpot_manager() has not been tested for explicit time integrators");
      return nullptr;
    };

    //! Get type of thickness scaling for thin shell structures
    Inpar::Solid::StcScale get_stc_algo() override
    {
      FOUR_C_THROW("STC is not supported in the old time integration framework!");
    };

    void update_iter_incr_constr(
        std::shared_ptr<Core::LinAlg::Vector<double>> lagrincr  ///< Lagrange multiplier increment
        ) override
    {
      FOUR_C_THROW("update_iter_incr_constr() has not been tested for explicit time integrators");
      return;
    }

    void update_iter_incr_cardiovascular0_d(
        std::shared_ptr<Core::LinAlg::Vector<double>> presincr  ///< pressure increment
        ) override
    {
      FOUR_C_THROW(
          "update_iter_incr_cardiovascular0_d() has not been tested for explicit time integrators");
      return;
    }

    /// Do the nonlinear solve, i.e. (multiple) corrector,
    /// for the time step. All boundary conditions have
    /// been set.
    Inpar::Solid::ConvergenceStatus solve() final
    {
      integrate_step();
      return Inpar::Solid::conv_success;
    }

    //! prepare partition step
    void prepare_partition_step() override
    {
      // do nothing for explicit time integrators
      return;
    }

    void use_block_matrix(std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
        std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps) override
    {
      FOUR_C_THROW("use_block_matrix() not implemented");
    }
    //@}
  };

}  // namespace Solid

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
