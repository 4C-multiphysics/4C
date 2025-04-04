// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_INTEGRATOR_HPP
#define FOUR_C_STRUCTURE_NEW_INTEGRATOR_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"               // enumerators
#include "4C_solver_nonlin_nox_enum_lists.hpp"  // enumerators

#include <NOX_Abstract_Vector.H>  // enumerators

#include <memory>

// forward declarations

#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
  class SparseOperator;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace Solid
{
  class ModelEvaluatorManager;
  class Dbc;
  class MonitorDbc;

  namespace ModelEvaluator
  {
    class Data;
    class Generic;
  }  // namespace ModelEvaluator

  namespace TimeInt
  {
    class Base;
    class BaseDataSDyn;
    class BaseDataGlobalState;
    class BaseDataIO;
  }  // namespace TimeInt

  /*! \brief Base class for all structural time integrators
   *
   */
  class Integrator
  {
   public:
    //! constructor
    Integrator();

    //! destructor
    virtual ~Integrator() = default;

    /*! \brief Initialization
     *
     * \param[in] sdyn_ptr Pointer to the structural dynamic data container
     * \param[in] gstate_ptr Pointer to the global state data container
     * \param[in] gio_ptr Pointer to the input/output data container
     * \param[in] dpc_ptr Pointer to the dirichlet boundary condition object
     * \param[in] timint_ptr Pointer to the underlying time integrator (read-only)
     */
    virtual void init(const std::shared_ptr<Solid::TimeInt::BaseDataSDyn>& sdyn_ptr,
        const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
        const std::shared_ptr<Solid::TimeInt::BaseDataIO>& gio_ptr,
        const std::shared_ptr<Solid::Dbc>& dbc_ptr,
        const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr);

    //! Setup (has to be implemented by the derived classes)
    virtual void setup();

    //! Post setup operation (compute initial equilibrium state), should be run directly after the
    //! setup routine has been finished
    virtual void post_setup() = 0;

    //! Set state variables
    virtual void set_state(const Core::LinAlg::Vector<double>& x) = 0;

    //! Set initial displacement field
    virtual void set_initial_displacement(
        const Inpar::Solid::InitialDisp init, const int startfuncno);

    /*! \brief Reset state variables of all models
     *  (incl. the structural dynamic state variables)
     *
     *  \param x (in) : current full state vector */
    void reset_model_states(const Core::LinAlg::Vector<double>& x);

    /*! \brief Add the viscous and mass contributions to the right hand side (TR-rule)
     *
     * \remark The remaining contributions have been considered in the corresponding model
     *         evaluators. This is due to the fact, that some models use a different
     *         time integration scheme for their terms (e.g. GenAlpha for the structure
     *         and OST for the remaining things). */
    virtual void add_visco_mass_contributions(Core::LinAlg::Vector<double>& f) const = 0;

    /*! \brief Add the viscous and mass contributions to the jacobian (TR-rule)
     *
     * \remark The remaining blocks have been considered in the corresponding model
     *         evaluators. This is due to the fact, that some models use a different
     *         time integration scheme for their terms (e.g. GenAlpha for the structure
     *         and OST for the remaining things). Furthermore, constraint/Lagrange
     *         multiplier blocks need no scaling anyway. */
    virtual void add_visco_mass_contributions(Core::LinAlg::SparseOperator& jac) const = 0;

    //! Apply the right hand side only
    virtual bool apply_force(
        const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f) = 0;

    /*! \brief Apply the stiffness only
     *
     * Normally this one is unnecessary, since it makes more sense
     * to evaluate the stiffness and right hand side at once, because of
     * the lower computational overhead. */
    virtual bool apply_stiff(
        const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) = 0;

    /*! \brief Apply force and stiff at once
     *
     *  Only one loop over all elements. Especially in the contact case,
     *  the difference between this call and first call apply_force and
     *  then apply_stiff is mentionable because of the projection operations. */
    virtual bool apply_force_stiff(const Core::LinAlg::Vector<double>& x,
        Core::LinAlg::Vector<double>& f, Core::LinAlg::SparseOperator& jac) = 0;

    /*! \brief Modify the right hand side and Jacobian corresponding to the requested correction
     * action of one (or several) second order constraint (SOC) model(s)
     */
    virtual bool apply_correction_system(const enum NOX::Nln::CorrectionType type,
        const std::vector<Inpar::Solid::ModelType>& constraint_models,
        const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f,
        Core::LinAlg::SparseOperator& jac) = 0;

    /*! \brief Remove contributions from the structural right-hand side stemming
     *  from any condensation operations (typical example is contact)
     */
    virtual void remove_condensed_contributions_from_rhs(
        Core::LinAlg::Vector<double>& rhs) const = 0;

    /*! \brief Calculate characteristic/reference norms for forces
     *
     *  The reference norms are used to scale the calculated iterative
     *  displacement norm and/or the residual force norm. For this
     *  purpose we only need the right order of magnitude, so we don't
     *  mind evaluating the corresponding norms at possibly different
     *  points within the time-step (end point, generalized midpoint). */
    virtual double calc_ref_norm_force(
        const enum ::NOX::Abstract::Vector::NormType& type) const = 0;

    //! compute the scaling operator for element based scaling using PTC
    virtual void compute_jacobian_contributions_from_element_level_for_ptc(
        std::shared_ptr<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr) = 0;

    //! Assemble the right hand side
    virtual bool assemble_force(Core::LinAlg::Vector<double>& f,
        const std::vector<Inpar::Solid::ModelType>* without_these_models = nullptr) const = 0;

    //! Assemble Jacobian
    virtual bool assemble_jac(Core::LinAlg::SparseOperator& jac,
        const std::vector<Inpar::Solid::ModelType>* without_these_models = nullptr) const
    {
      return false;
    };

    //! Create backup state
    void create_backup_state(const Core::LinAlg::Vector<double>& dir);

    //! recover state from the stored backup
    void recover_from_backup_state();

    //! return integration factor
    virtual double get_int_param() const = 0;
    virtual double get_acc_int_param() const { return get_int_param(); }

    /*!
     * \brief Allows to stop the simulation before the max. time or max timestep is reached
     *
     * \return true In case of an early stop is desired
     * \return false In case of no early stop (default)
     */
    virtual bool early_stopping() const { return false; }

    //! @name Restart and output related functions
    //!@{

    /*! write restart information of the different time integration schemes
     *  and model evaluators */
    void write_restart(Core::IO::DiscretizationWriter& iowriter) const
    {
      write_restart(iowriter, false);
    };
    virtual void write_restart(
        Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const = 0;

    /*! read restart information of the different time integration schemes
     *  and model evaluators */
    virtual void read_restart(Core::IO::DiscretizationReader& ioreader) = 0;

    //!@}

    //! @name Monolithic update routines
    //!@{

    //! things that should be done before updating
    virtual void pre_update() = 0;

    /*! \brief Update configuration after time step
     *
     *  Thus the 'last' converged is lost and a reset of the time step
     *  becomes impossible. We are ready and keen awaiting the next
     *  time step. */
    virtual void update_step_state() = 0;

    /*! \brief Update everything on element level after time step and after output
     *
     *  Thus the 'last' converged is lost and a reset of the time step
     *  becomes impossible. We are ready and keen awaiting the next
     *  time step. */
    virtual void update_step_element() = 0;

    //! calculate stresses and strains in the different model evaluators
    void determine_stress_strain();

    //! calculate the energy in the different model evaluators
    void determine_energy();

    //! get the model value in accordance with the currently active time integration
    virtual double get_model_value(const Core::LinAlg::Vector<double>& x);

    /*! return the total structural energy evaluated at the actual mid-time
     *  in accordance to the used time integration scheme */
    double get_total_mid_time_str_energy(const Core::LinAlg::Vector<double>& x);

    /// update the structural energy variable in the end of a successful time step
    void update_structural_energy();

    //! calculate an optional quantity in the different model evaluators
    void determine_optional_quantity();

    /*! \brief Output to file
     *
     *  This routine prints always the last converged state, i.e.
     *  \f$D_{n}, V_{n}, A_{n}\f$. So, #UpdateIncrement should be called
     *  upon object prior to writing stuff here.
     *
     *  */
    void output_step_state(Core::IO::DiscretizationWriter& iowriter) const;

    /**
     * \brief Do stuff that has to be done before runtime output is written.
     */
    void runtime_pre_output_step_state();

    /*! \brief Output to file during runtime (no filter necessary)
     *
     *  This routine prints always the last converged state, i.e.
     *  \f$D_{n}, V_{n}, A_{n}\f$. So, #UpdateIncrement should be called
     *  upon object prior to writing stuff here.
     *
     */
    void runtime_output_step_state() const;

    /** \brief reset step configuration after time step
     *
     *  This function is supposed to reset all variables which are directly related
     *  to the current new step n+1. To be more precise all variables ending with "Np"
     *  have to be reset. */
    virtual void reset_step_state();

    /// things that should be done after updating
    virtual void post_update() = 0;

    /// things that should be done after the timeloop
    virtual void post_time_loop();

    //! update constant contributions of the current state for the new time step \f$t_{n+1}\f$
    virtual void update_constant_state_contributions() = 0;

    //! things that should be done after output
    virtual void post_output();

    void monitor_dbc(Core::IO::DiscretizationWriter& writer) const;
    //!@}

    //! @name Accessors
    //!@{

    double get_condensed_update_norm(const enum NOX::Nln::StatusTest::QuantityType& qtype) const;

    double get_condensed_previous_sol_norm(
        const enum NOX::Nln::StatusTest::QuantityType& qtype) const;

    double get_condensed_solution_update_rms(
        const enum NOX::Nln::StatusTest::QuantityType& qtype) const;

    int get_condensed_dof_number(const enum NOX::Nln::StatusTest::QuantityType& qtype) const;

    //! Return the model evaluator control object (read and write)
    Solid::ModelEvaluatorManager& model_eval();

    //! Return the model evaluator control object (read-only)
    const Solid::ModelEvaluatorManager& model_eval() const;
    std::shared_ptr<const Solid::ModelEvaluatorManager> model_eval_ptr() const;

    //! Return the model evaluator object for the given model type
    Solid::ModelEvaluator::Generic& evaluator(const Inpar::Solid::ModelType& mt);

    //! Return the model evaluator object for the given model type
    const Solid::ModelEvaluator::Generic& evaluator(const Inpar::Solid::ModelType& mt) const;

    //! Return the model evaluator data object (read-only)
    const Solid::ModelEvaluator::Data& eval_data() const;

    //! Return the model evaluator data object (read and write access)
    Solid::ModelEvaluator::Data& eval_data();

    //! Return the Dirichlet boundary condition object (read-only)
    const Solid::Dbc& get_dbc() const;

    //!@}

   protected:
    //! returns init state
    inline const bool& is_init() const { return isinit_; };

    //! returns setup state
    inline const bool& is_setup() const { return issetup_; };

    //! Check the init state
    void check_init() const;

    //! Check the setup state
    void check_init_setup() const;

    /*! \brief Compute the mass matrix and identify consistent accelerations for the initial state.
     */
    void compute_mass_matrix_and_init_acc();

    /*! \brief Check if current state is equilibrium (with respect to
     *  a given tolerance of the inf-norm)
     *
     *  This is a sanity check in case of nonlinear mass
     *  and non-additive rotation pseudo-vector DoFs where
     *  determination of consistent accelerations is more intricate
     *  and not supported yet */
    bool current_state_is_equilibrium(const double& tol);

    //! Return the structural dynamic data container
    Solid::TimeInt::BaseDataSDyn& sdyn();

    //! Return the structural dynamic data container (read-only)
    const Solid::TimeInt::BaseDataSDyn& s_dyn() const;

    //! Return the global state data container
    Solid::TimeInt::BaseDataGlobalState& global_state();

    //! Return the global state data container (read-only)
    const Solid::TimeInt::BaseDataGlobalState& global_state() const;

    //! Return the Dirichlet boundary condition object
    Solid::Dbc& dbc();

    //! Return the time integration strategy object (read-only)
    const Solid::TimeInt::Base& tim_int() const;

    //! reset the time step dependent parameters for the element evaluation
    virtual void reset_eval_params() {};

    double get_condensed_global_norm(const enum NOX::Nln::StatusTest::QuantityType& qtype,
        const enum ::NOX::Abstract::Vector::NormType& normtype, double& mynorm) const;

   protected:
    //! indicates if the init() function has been called
    bool isinit_;

    //! indicates if the setup() function has been called
    bool issetup_;

    //! Mid-time energy container
    struct MidTimeEnergy
    {
      /// constructor
      MidTimeEnergy(const Integrator& integrator);

      /// setup
      void setup();

      /// can this container be used?
      bool is_correctly_configured() const;

      bool store_energy_n() const;

      /// print energy info to output stream
      void print(std::ostream& os) const;

      /// Get total energy measure in accordance to the surrounding time integrator
      double get_total() const;

      /// average quantities based on the used averaging type
      std::shared_ptr<const Core::LinAlg::Vector<double>> average(
          const Core::LinAlg::Vector<double>& state_np, const Core::LinAlg::Vector<double>& state_n,
          const double fac_n) const;

      /// copy current state to old state (during update)
      void copy_np_to_n();

      /// internal energy at \f$t_{n+1}\f$
      double int_energy_np_ = 0.0;

      /// external energy at \f$t_{n+1}\f$
      double ext_energy_np_ = 0.0;

      /// kinetic energy at \f$t_{n+1}\f$
      double kin_energy_np_ = 0.0;

      /// internal energy at \f$t_{n}\f$
      double int_energy_n_ = 0.0;

      /// external energy at \f$t_{n}\f$
      double ext_energy_n_ = 0.0;

      /// kinetic energy at \f$t_{n}\f$
      double kin_energy_n_ = 0.0;

     private:
      /// call-back
      const Integrator& integrator_;

      /// mid-time energy averaging type
      enum Inpar::Solid::MidAverageEnum avg_type_;

      /// setup flag
      bool issetup_ = false;
    };
    MidTimeEnergy mt_energy_ = MidTimeEnergy(*this);

   private:
    //! pointer to the model evaluator
    std::shared_ptr<Solid::ModelEvaluatorManager> modelevaluator_ptr_;

    //! pointer to model evaluator data
    std::shared_ptr<Solid::ModelEvaluator::Data> eval_data_ptr_;

    //! pointer to the structural dynamic data container
    std::shared_ptr<Solid::TimeInt::BaseDataSDyn> sdyn_ptr_;

    //! pointer to the global state data container
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;

    //! pointer to the input/output data container
    std::shared_ptr<Solid::TimeInt::BaseDataIO> io_ptr_;

    //! pointer to the dirichlet boundary condition object
    std::shared_ptr<Solid::Dbc> dbc_ptr_;

    //! pointer to the underlying time integrator (read-only)
    std::shared_ptr<const Solid::TimeInt::Base> timint_ptr_;

    //! pointer to the dirichlet boundary condition monitor
    std::shared_ptr<Solid::MonitorDbc> monitor_dbc_ptr_ = nullptr;
  };  // namespace Solid
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
