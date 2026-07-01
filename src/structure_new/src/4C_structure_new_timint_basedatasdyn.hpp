// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATASDYN_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATASDYN_HPP

#include "4C_config.hpp"

#include "4C_rebalance.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <NOX_Abstract_Vector.H>
#include <Teuchos_Time.hpp>

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
}  // namespace Core::LinAlg
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Geo::MeshFree
{
  class BoundingBox;
}  // namespace Core::Geo::MeshFree

namespace Solid
{
  namespace ModelEvaluator
  {
    class Generic;
  }  // namespace ModelEvaluator
  namespace TimeInt
  {
    struct DynamicRebalanceConfig
    {
      bool enabled = false;
      double imbalance_threshold = 1.1;
      int window_steps = 1;
      int cooldown_steps = 0;
      Core::Rebalance::MeshPartitioningParameters mesh_partitioning_parameters;
      double edge_weight_multiplier = 1.0;
    };

    DynamicRebalanceConfig parse_dynamic_rebalance_config(
        const Teuchos::ParameterList& rebalance_params);

    /** \brief Structural dynamics data container for the structural (time) integration
     *
     * This data container holds everything, which refers directly to the
     * control mechanisms of the structural dynamics algorithms. The input parameters
     * are transformed into enumerators. Furthermore, the maximum time and step number
     * can be found here.
     *
     * */
    class BaseDataSDyn
    {
     public:
      /// constructor
      BaseDataSDyn();

      /// destructor
      virtual ~BaseDataSDyn() = default;

      /// initialize class variables (already existing)
      virtual void init(const std::shared_ptr<Core::FE::Discretization> discret,
          const Teuchos::ParameterList& sDynParams, const Teuchos::ParameterList& xparams,
          const std::shared_ptr<std::set<Solid::ModelType>> modeltypes,
          const std::shared_ptr<std::set<Solid::EleTech>> eletechs,
          const std::shared_ptr<std::map<Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>>
              linsolvers);

      /// setup new class variables
      virtual void setup();

     protected:
      /// get the indicator state
      inline bool is_init() const { return isinit_; };

      /// get the indicator state
      inline bool is_setup() const { return issetup_; };

      /// Check if init() has been called, yet.
      inline void check_init() const
      {
        FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");
      }

      /// Check if init() and setup() have been called, yet.
      inline void check_init_setup() const
      {
        FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
      }

     public:
      /// @name Get general control parameters (read only access)
      ///@{

      /// Returns final time \f$t_\text{fin}\f$
      double get_time_max() const
      {
        check_init_setup();
        return timemax_;
      };

      /// Returns final time step \f$N\f$
      int get_step_max() const
      {
        check_init_setup();
        return stepmax_;
      };

      bool is_restarting_initial_state() const
      {
        check_init_setup();
        return isrestarting_initial_state_;
      }

      /// Returns timer for solution technique
      std::shared_ptr<const Teuchos::Time> get_timer() const
      {
        check_init_setup();
        return timer_;
      };

      /// Returns dynamic type
      Solid::DynamicType get_dynamic_type() const
      {
        check_init_setup();
        return dyntype_;
      };

      /// Get type of shell thickness scaling
      Solid::StcScale get_stc_algo_type() const
      {
        check_init_setup();
        return stcscale_;
      };

      /// Get number of layers for multilayered shell thickness scaling case
      int get_stc_layer() const
      {
        check_init_setup();
        return stclayer_;
      };

      /// Returns minimal non-linear iteration number
      int get_iter_min() const
      {
        check_init_setup();
        return itermin_;
      };

      /// Returns maximal non-linear iteration number
      int get_iter_max() const
      {
        check_init_setup();
        return itermax_;
      };

      /// Returns true if the external load should be linearized
      bool get_load_lin() const
      {
        check_init_setup();
        return loadlin_;
      }

      // Return time until the prestressing algorithm should be applied
      double get_pre_stress_time() const
      {
        check_init_setup();
        return prestresstime_;
      }

      /// Returns the tolerance for the displacements during prestressing
      double get_pre_stress_displacement_tolerance() const
      {
        check_init_setup();
        return prestress_displacement_tolerance_;
      }

      /// Returns the minimum number of load steps during prestressing
      [[nodiscard]] int get_pre_stress_minimum_number_of_load_steps() const
      {
        check_init_setup();
        return prestress_min_number_of_load_steps_;
      }

      /// Returns prestress type
      Solid::PreStress get_pre_stress_type() const
      {
        check_init_setup();
        return prestresstype_;
      };

      /// Returns predictor type
      Solid::PredEnum get_predictor_type() const
      {
        check_init_setup();
        return predtype_;
      };

      /// Returns nonlinear solver type
      Solid::NonlinSolTech get_nln_solver_type() const
      {
        check_init_setup();
        return nlnsolvertype_;
      };

      /// Returns the divergence action
      /// Short: What to do if the non-linear solver fails.
      Solid::DivContAct get_divergence_action() const
      {
        check_init_setup();
        return divergenceaction_;
      };

      /// Returns mid-time energy type
      Solid::MidAverageEnum get_mid_time_energy_type() const
      {
        check_init_setup();
        return mid_time_energy_type_;
      }

      /// Returns number of times you want to halve your timestep in case nonlinear solver
      /// diverges
      int get_max_div_con_refine_level() const
      {
        check_init_setup();
        return maxdivconrefinementlevel_;
      };

      /// Returns nox parameters
      const Teuchos::ParameterList& get_nox_params() const
      {
        check_init_setup();
        return *noxparams_;
      }

      /// Returns the initial pseudo time step for the PTC method
      double get_initial_ptc_pseudo_time_step() const { return ptc_delta_init_; }
      ///@}

      /// @name Get mutable linear solver variables (read only access)
      ///@{

      /// Returns linear solvers pointer
      const std::map<Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>& get_lin_solvers()
          const
      {
        check_init_setup();
        return *linsolvers_;
      }
      ///@}


      /// @name Get damping control parameters (read only access)
      ///@{
      /// Returns damping type
      Solid::DampKind get_damping_type() const
      {
        check_init_setup();
        return damptype_;
      };

      /// Returns damping factor for stiffness \f$c_\text{K}\f$
      double get_damping_stiffness_factor() const
      {
        check_init_setup();
        return dampk_;
      };

      /// Returns damping factor for mass \f$c_\text{M}\f$
      double get_damping_mass_factor() const
      {
        check_init_setup();
        return dampm_;
      };
      ///@}

      /// @name Get mass and inertia control parameters (read only access)
      ///@{
      /// Returns mass linearization type
      Solid::MassLin get_mass_lin_type() const
      {
        check_init_setup();
        return masslintype_;
      };

      bool is_mass_lumping() const
      {
        check_init_setup();
        return lumpmass_;
      }

      bool neglect_inertia() const
      {
        check_init_setup();
        return neglectinertia_;
      }
      ///@}

      /// @name Get model evaluator control parameters (read only access)
      ///@{
      /// Returns types of the current models
      const std::set<Solid::ModelType>& get_model_types() const
      {
        check_init_setup();
        return *modeltypes_;
      };

      /// Returns current active element technologies
      const std::set<Solid::EleTech>& get_element_technologies() const
      {
        check_init_setup();
        return *eletechs_;
      };

      /// check if the given model type is active.
      bool have_model_type(const Solid::ModelType& modeltype) const;

      /// check if the given element technology is active.
      bool have_ele_tech(const Solid::EleTech& eletech) const;
      ///@}

      /// @name Get model specific data container
      ///@{
      /// Returns periodic bounding box object (read access)
      std::shared_ptr<Core::Geo::MeshFree::BoundingBox> const& get_periodic_bounding_box() const
      {
        check_init_setup();
        return periodic_boundingbox_;
      }
      ///@}

      /// @name Get the different status test control parameters (read only)
      ///@{
      /// Returns the STR vector norm type
      Solid::VectorNorm get_norm_type() const
      {
        check_init_setup();
        return normtype_;
      }

      /// Returns the NOX normtype
      enum ::NOX::Abstract::Vector::NormType get_nox_norm_type() const
      {
        check_init_setup();
        return nox_normtype_;
      }

      /// Return random factor for timestep in case nonlinear solver diverges
      double get_random_time_step_factor() const
      {
        check_init_setup();
        return rand_tsfac_;
      }


      /// Return level of refinement in case of divercontype_ == adapt_step
      int get_div_con_refine_level() const
      {
        check_init_setup();
        return divconrefinementlevel_;
      }


      /// Return number of fine steps in case of in case of divercontype_ == adapt_step
      int get_div_con_num_fine_step() const
      {
        check_init_setup();
        return divconnumfinestep_;
      }

      /// @name Get residual and increment related parameters
      ///@{
      /// Returns the combination type of the two quantities
      Solid::BinaryOp get_res_incr_combo_type(const NOX::Nln::StatusTest::QuantityType& qtype_res,
          const NOX::Nln::StatusTest::QuantityType& qtype_incr) const;
      ///@}

      /// @name Get residual related parameters
      ///@{
      /// Returns the tolerance values for the different quantities
      double get_res_tolerance(const NOX::Nln::StatusTest::QuantityType& qtype) const;

      /// Returns the tolerance type of the different quantities
      Solid::ConvNorm get_res_tolerance_type(const NOX::Nln::StatusTest::QuantityType& qtype) const;

      /// Returns the combination type of the different quantities
      Solid::BinaryOp get_res_combo_type(const NOX::Nln::StatusTest::QuantityType& qtype) const;

      Solid::BinaryOp get_res_combo_type(const NOX::Nln::StatusTest::QuantityType& qtype_1,
          const NOX::Nln::StatusTest::QuantityType& qtype_2) const;
      ///@}

      /// @name Get increment related parameters
      ///@{
      /// Returns the tolerance values for the different quantities
      double get_incr_tolerance(const NOX::Nln::StatusTest::QuantityType& qtype) const;

      /// Returns the tolerance type of the different quantities
      Solid::ConvNorm get_incr_tolerance_type(
          const NOX::Nln::StatusTest::QuantityType& qtype) const;

      Solid::ConvNorm get_incr_tolerance_type(const NOX::Nln::StatusTest::QuantityType& qtype_1,
          const NOX::Nln::StatusTest::QuantityType& qtype_2) const;

      /// Returns the combination type of the different quantities
      Solid::BinaryOp get_incr_combo_type(const NOX::Nln::StatusTest::QuantityType& qtype) const;

      Solid::BinaryOp get_incr_combo_type(const NOX::Nln::StatusTest::QuantityType& qtype_1,
          const NOX::Nln::StatusTest::QuantityType& qtype_2) const;
      ///@}
      ///@}

      /// @name Set some general control parameters
      ///@{
      /// Set final time step \f$N\f$
      void set_step_max(int new_stepmax)
      {
        check_init_setup();
        stepmax_ = new_stepmax;
      };

      /// Set final time \f$t_\text{fin}\f$
      void set_time_max(double new_timemax)
      {
        check_init_setup();
        timemax_ = new_timemax;
      };

      /// Set random factor for timestep in case nonlinear solver diverges
      void set_random_time_step_factor(double new_rand_tsfac)
      {
        check_init_setup();
        rand_tsfac_ = new_rand_tsfac;
      }

      /// Set level of refinement in case of divercontype_ == adapt_step
      void set_div_con_refine_level(int new_divconrefinementlevel)
      {
        check_init_setup();
        divconrefinementlevel_ = new_divconrefinementlevel;
      }

      /// Return number of fine steps in case of in case of divercontype_ == adapt_step
      void set_div_con_num_fine_step(int new_divconnumfinestep)
      {
        check_init_setup();
        divconnumfinestep_ = new_divconnumfinestep;
      }
      ///@}

      /// @name Get mutable NOX parameters (read and write access)
      ///@{
      /// Returns nox parameters
      Teuchos::ParameterList& get_nox_params()
      {
        check_init_setup();
        return *noxparams_;
      }

      /// Returns nox parameters
      std::shared_ptr<Teuchos::ParameterList> get_nox_params_ptr()
      {
        check_init_setup();
        return noxparams_;
      }

      ///@}

      /// @name Get mutable linear solver variables (read and write access)
      ///@{
      /// Returns linear solvers pointer
      std::shared_ptr<std::map<Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>>
      get_lin_solvers_ptr()
      {
        check_init_setup();
        return linsolvers_;
      }

      /// Returns linear solvers pointer
      std::map<Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>& get_lin_solvers()
      {
        check_init_setup();
        return *linsolvers_;
      }
      ///@}

      /// @name Get model evaluator control parameters (read and write access)
      ///@{
      /// Return a pointer to the coupling model evaluator
      const std::shared_ptr<Solid::ModelEvaluator::Generic>& coupling_model_ptr()
      {
        return coupling_model_ptr_;
      }
      ///@}

      /// @name Get the initial displacement method
      ///@{
      /// Returns type of initial displacement
      Solid::InitialDisp get_initial_disp() const
      {
        check_init_setup();
        return initial_disp_;
      };

      /// Returns the function index to initialize the displacement
      int start_func_no() const
      {
        check_init_setup();
        return start_func_no_;
      }

      const DynamicRebalanceConfig& get_dynamic_rebalance_config() const
      {
        check_init_setup();
        return dynamic_rebalance_config_;
      }
      ///@}

     protected:
      /** \brief returns the structural dynamics parameter list (internal use only!)
       *
       * Do not touch this. It should be used only in derived Setup routines.
       * Do not call it from outside! */
      const Teuchos::ParameterList& get_sdyn_params() const
      {
        check_init();
        return *sdynparams_ptr_;
      }

     protected:
      /// @name variables for internal use only
      ///@{
      /// init flag
      bool isinit_;

      /// setup flag
      bool issetup_;
      ///@}

     private:
      /// @name General control parameters
      ///@{

      /// final time \f$t_\text{fin}\f$
      double timemax_;

      /// final time step \f$N\f$
      int stepmax_;

      /// flag indicating that if the initial state needs to be re-calculated
      bool isrestarting_initial_state_;

      ///@}

      /// @name Time measurement
      ///@{
      /// timer for solution technique
      std::shared_ptr<Teuchos::Time> timer_;

      ///@}

      /// @name Damping control parameters
      /// Rayleigh damping means \f${C} = c_\text{K} {K} + c_\text{M} {M}\f$
      ///@{
      /// damping type
      Solid::DampKind damptype_;

      /// damping factor for stiffness \f$c_\text{K}\f$
      double dampk_;

      /// damping factor for mass \f$c_\text{M}\f$
      double dampm_;
      ///@}

      /// @name Mass and inertia control parameters
      ///@{
      /// have inertia forces to be linearized?
      Solid::MassLin masslintype_;

      /// lump the mass matrix?
      bool lumpmass_;

      /// neglect inertia?
      bool neglectinertia_;
      ///@}

      /// @name Model evaluator control parameters
      ///@{

      /// current active model types for the model evaluator
      std::shared_ptr<std::set<Solid::ModelType>> modeltypes_;

      /// current active element technologies
      std::shared_ptr<std::set<Solid::EleTech>> eletechs_;

      /// pointer to the coupling model evaluator object
      std::shared_ptr<Solid::ModelEvaluator::Generic> coupling_model_ptr_;

      ///@}

      /// @name implicit and explicit time integrator parameters
      ///@{
      /// dynamic type
      Solid::DynamicType dyntype_;

      /// scaled thickness conditioning type (necessary for shells)
      Solid::StcScale stcscale_;

      /// number of layers for multilayered case
      int stclayer_;

      ///@}

      /// @name implicit time integrator parameters
      ///@{
      /// minimal non-linear iteration number
      int itermin_;

      /// maximal non-linear iteration number
      int itermax_;

      /// linearization of external follower load in Newton
      bool loadlin_;

      /// Time until the prestressing algorithm should be applied
      double prestresstime_;

      /// Tolerance for the residual displacements during prestressing
      double prestress_displacement_tolerance_;

      /// Minimum number of load steps during prestressing
      int prestress_min_number_of_load_steps_;

      /// type of pre-stressing
      Solid::PreStress prestresstype_;

      /// type of the predictor
      Solid::PredEnum predtype_;

      /// type of nonlinear solver
      Solid::NonlinSolTech nlnsolvertype_;

      /// action to be performed when the non-linear solver diverges
      Solid::DivContAct divergenceaction_;

      /// mid-time energy type
      Solid::MidAverageEnum mid_time_energy_type_;

      /// how often you want to half your timestep until you give up
      int maxdivconrefinementlevel_;

      /// nox parameters list
      std::shared_ptr<Teuchos::ParameterList> noxparams_;

      /// initial pseudo time step for the pseudo transient continuation (PTC) method
      double ptc_delta_init_;
      ///@}

      /// @name linear solver variables
      ///@{

      /// pointer to the linear solvers map
      std::shared_ptr<std::map<Solid::ModelType, std::shared_ptr<Core::LinAlg::Solver>>>
          linsolvers_;
      ///@}

      /// @name status test control parameters
      ///@{

      /// vector norm type for the status/convergence test
      Solid::VectorNorm normtype_;

      /// vector norm type for the status/convergence test
      enum ::NOX::Abstract::Vector::NormType nox_normtype_;

      /// @name primary variables
      ///@{
      /// tolerance residual displacements
      double tol_disp_incr_;

      /// tolerance type for the convergence check of the displacement vector
      Solid::ConvNorm toltype_disp_incr_;

      /// tolerance force residual
      double tol_fres_;

      /// tolerance type for the convergence check of the force residual
      Solid::ConvNorm toltype_fres_;

      /// tolerance pressure residual
      double tol_pres_;

      /// tolerance type for the convergence check of the pressure residual
      Solid::ConvNorm toltype_pres_;

      /// tolerance incompressible residual/ residual pressure forces
      double tol_inco_;

      /// tolerance type for the convergence check if the incompressible residual/ residual
      /// pressure forces
      Solid::ConvNorm toltype_inco_;

      /// tolerance EAS residual
      double tol_eas_res_;

      /// tolerance type for the convergence check of the EAS residual
      Solid::ConvNorm toltype_eas_res_;

      /// tolerance EAS increment
      double tol_eas_incr_;

      /// tolerance type for the convergence check of the EAS increment
      Solid::ConvNorm toltype_eas_incr_;

      /// type of combination of the displacement and the pressure test
      Solid::BinaryOp normcombo_disp_pres_;

      /// type of combination of the force and the pressure residual test
      Solid::BinaryOp normcombo_fres_inco_;

      /// type of combination of the force and the EAS residual
      Solid::BinaryOp normcombo_fres_eas_res_;

      /// type of combination of the displacement and the EAS increment
      Solid::BinaryOp normcombo_disp_eas_incr_;

      /// type of combination of the force and the displacement test
      Solid::BinaryOp normcombo_fres_disp_;
      ///@}

      /// @name constraint variables
      ///@{

      /// tolerance type for the convergence check of the 0D cardiovascular residual
      Solid::ConvNorm toltype_cardvasc0d_res_;

      /// tolerance 0D cardiovascular residual
      double tol_cardvasc0d_res_;

      /// tolerance type for the convergence check of the 0D cardiovascular dof increment
      Solid::ConvNorm toltype_cardvasc0d_incr_;

      /// tolerance 0D cardiovascular dof increment
      double tol_cardvasc0d_incr_;

      /// tolerance type for the convergence check of the constraint residual
      Solid::ConvNorm toltype_constr_res_;

      /// tolerance constraint residual
      double tol_constr_res_;

      /// tolerance type for the convergence check of the constraint LM increment
      Solid::ConvNorm toltype_constr_incr_;

      /// tolerance constraint LM increment
      double tol_constr_incr_;

      /// tolerance type for the convergence check of the contact residual
      Solid::ConvNorm toltype_contact_res_;

      /// tolerance contact constraint residual
      double tol_contact_res_;

      /// tolerance type  for the convergence check of the contact lagrange increment
      Solid::ConvNorm toltype_contact_lm_incr_;

      /// tolerance contact lagrange increment
      double tol_contact_lm_incr_;

      /// type of combination of the force and the contact residual
      Solid::BinaryOp normcombo_fres_contact_res_;

      /// type of combination of the displacement and the contact lagrange multiplier increment
      /// test
      Solid::BinaryOp normcombo_disp_contact_lm_incr_;

      /// type of combination of the force and the 0D cardiovascular model residual
      Solid::BinaryOp normcombo_fres_cardvasc0d_res_;

      /// type of combination of the displacement and the 0D cardiovascular model dof increment
      /// test
      Solid::BinaryOp normcombo_disp_cardvasc0d_incr_;

      /// type of combination of the force and the constraint residual
      Solid::BinaryOp normcombo_fres_constr_res_;

      /// type of combination of the displacement and the 0D cardiovascular model dof increment
      /// test
      Solid::BinaryOp normcombo_disp_constr_incr_;

      /// random factor for modifying time-step size in case this way of continuing non-linear
      /// iteration was chosen
      double rand_tsfac_;

      /// number of refinement level in case of divercontype_ == adapt_step
      int divconrefinementlevel_;

      /// number of converged time steps on current refinement level in case of divercontype_ ==
      /// adapt_step
      int divconnumfinestep_;

      ///@}

      /// @name initial displacement variables
      ///@{

      /// type of initial displacement initialization
      Solid::InitialDisp initial_disp_;
      int start_func_no_;

      ///@}

      /// @name pointer to model specific data container
      ///@{

      /// pointer to the linear solvers map
      std::shared_ptr<Core::Geo::MeshFree::BoundingBox> periodic_boundingbox_;
      ///@}

      ///@}
      /** \brief Ptr to the structural dynamics parameter list
       *
       * Do not touch this. It should be used only in derived Setup routines. Do not call it from
       * outside! */
      std::shared_ptr<const Teuchos::ParameterList> sdynparams_ptr_;

      /// Parsed dynamic rebalance settings used by the structural time integrator.
      DynamicRebalanceConfig dynamic_rebalance_config_;
    };  // class BaseDataSDyn

    /** \brief Generalized alpha structural dynamics data container
     *
     * This data container is derived from the standard structural dynamics data container
     * and contains some special GenAlpha input parameters.
     *
     * */
    class GenAlphaDataSDyn : public BaseDataSDyn
    {
     public:
      //! constructor
      GenAlphaDataSDyn();

      //! Setup function [derived]
      void setup() override;

     public:
      //! @name Read-only accessors
      //!@{

      //! returns the mid-average type (for more information see MidAverageEnum)
      const Solid::MidAverageEnum& get_mid_average_type() const
      {
        check_init_setup();
        return midavg_;
      };

      //! Return time integration parameter \f$\beta\f$
      const double& get_beta() const
      {
        check_init_setup();
        return beta_;
      };

      //! Return time integration parameter \f$\gamma\f$
      const double& get_gamma() const
      {
        check_init_setup();
        return gamma_;
      };

      //! Return time integration parameter \f$\alpha_f\f$
      const double& get_alpha_f() const
      {
        check_init_setup();
        return alphaf_;
      };

      //! Return time integration parameter \f$\alpha_m\f$
      const double& get_alpha_m() const
      {
        check_init_setup();
        return alpham_;
      };

      //! Return spectral radius \f$\rho_\infty\f$
      const double& get_rho_inf() const
      {
        check_init_setup();
        return rhoinf_;
      };

      //!@}

     private:
      //! mid-average type more at MidAverageEnum
      Solid::MidAverageEnum midavg_;

      //! Time integration parameter \f$\beta \in (0,1/2]\f$
      double beta_;

      //! Time integration parameter \f$\gamma \in (0,1]\f$
      double gamma_;

      //! Time integration parameter \f$\alpha_f \in [0,1)\f$
      double alphaf_;

      //! Time integration parameter \f$\alpha_m \in [-1,1)\f$
      double alpham_;

      //! Spectral radius \f$\rho_\infty \in [0,1]\f$
      double rhoinf_;
    };

    /** \brief One-step Theta structural dynamics data container
     *
     * This data container is derived from the standard structural dynamics data container
     * and contains some special OneStepTheta input parameters.
     *
     * */
    class OneStepThetaDataSDyn : public BaseDataSDyn
    {
     public:
      //! constructor
      OneStepThetaDataSDyn();

      //! Setup function [derived]
      void setup() override;

     public:
      //! @name Read-only accessors
      //! @{

      //! Return time integration parameter \f$\theta\f$
      const double& get_theta() const
      {
        check_init_setup();
        return theta_;
      };

      //! @}

     private:
      //! Time integration parameter \f$\theta \in [0,1]\f$
      double theta_;
    };

    /** \brief Forward Euler structural dynamics data container
     *
     * This data container is derived from the standard structural dynamics data container
     * and contains some special Forward Euler input parameters.
     *
     */
    class ExplEulerDataSDyn : public BaseDataSDyn
    {
     public:
      //! constructor
      ExplEulerDataSDyn();

      //! Setup function [derived]
      void setup() override;

     public:
      //! @name Read-only accessors
      //! @{

      //! Return time integration parameter \f$\theta\f$
      bool get_modified_forward_euler() const
      {
        check_init_setup();
        return modexpleuler_;
      };

      //! @}

     private:
      //! Modified forward Euler flag
      bool modexpleuler_;
    };

  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
