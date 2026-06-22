// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_exceptions.hpp"

#include <format>
#include <string>
#include <tuple>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// namespace: utilities for
  /// InelasticDefgradTransvIsotropElastViscoplast
  namespace InelasticDefgradTransvIsotropElastViscoplastUtils
  {
    /**
     * @brief Caches values evaluated at Gauss points.
     *
     * This utility stores values of type `T` together with a flag indicating
     * whether the value for a given Gauss point has already been evaluated.
     * It is intended to avoid repeated computations across material evaluations.
     *
     * @tparam T Type of the cached quantity.
     */
    template <typename T>
    class CachedQuantity
    {
     public:
      /**
       * @brief Returns the cached value for a given Gauss point.
       *
       * @param gp Gauss point index.
       * @return const reference to the cached value for the specified Gauss point.
       * @throws If the gauss point index is out of bounds or if the values is not evaluated yet.
       */
      [[nodiscard]] const T& value(std::size_t gp) const
      {
        FOUR_C_ASSERT_ALWAYS(gp < value_.size(), "Invalid gp {}!", gp);
        FOUR_C_ASSERT_ALWAYS(is_evaluated_[gp], "Value is not evaluated for gp {}!", gp);
        return value_[gp];
      }

      /**
       * @brief Checks if the value for a given Gauss point has been evaluated.
       *
       * @param gp Gauss point index.
       * @return true if the value is evaluated, false otherwise.
       * @throws If the gauss point index is out of bounds.
       */
      [[nodiscard]] bool is_evaluated(std::size_t gp) const
      {
        FOUR_C_ASSERT_ALWAYS(gp < is_evaluated_.size(), "Invalid gp {}!", gp);
        return is_evaluated_[gp];
      }

      /**
       * @brief Resizes the cached quantity to the given number of Gauss points.
       *
       * This method initializes the cache for the specified number of Gauss points and resets all
       * evaluation flags to false.
       *
       * @param numgp Number of Gauss points to resize the cache for.
       */
      void resize(std::size_t numgp)
      {
        value_.resize(numgp);
        is_evaluated_.assign(numgp, false);
      }

      /**
       * @brief Sets the value for a given Gauss point and marks it as evaluated.
       *
       * @param gp Gauss point index.
       * @param value The value to be cached for the specified Gauss point.
       * @throws If the gauss point index is out of bounds.
       */
      void set(std::size_t gp, T value)
      {
        FOUR_C_ASSERT_ALWAYS(gp < value_.size(), "Invalid gp {}!", gp);
        value_[gp] = std::move(value);
        is_evaluated_[gp] = true;
      }

      /**
       * @brief Marks the value for a given Gauss point as not evaluated.
       *
       * @param gp Gauss point index.
       * @throws If the gauss point index is out of bounds.
       */
      void reset(std::size_t gp)
      {
        FOUR_C_ASSERT_ALWAYS(gp < is_evaluated_.size(), "Invalid gp {}!", gp);
        is_evaluated_[gp] = false;
      }

     private:
      std::vector<T> value_;            ///< vector storing the cached values for each Gauss point
      std::vector<bool> is_evaluated_;  ///< vector of flags indicating whether the value for each
                                        ///< Gauss point has been evaluated
    };

    /// declare numerical tolerance to be used in the verification of (numerically) zero plastic
    /// strain increments
    constexpr double zero_plastic_strain_increment{1.0e-14};

    /// tolerance to determine if the incoming defgrad-temperature pair matches the last evaluated
    /// one
    constexpr double thermo_mechanical_state_equality_tolerance = 1.0e-12;

    /// enum class for error types in InelasticDefgradTransvIsotropElastViscoplast, used for
    /// triggering different procedures (e.g. Reinterpolation,
    /// substepping, line search) during the
    /// Local Newton Loop
    enum class ErrorType
    {
      no_errors,                ///< no errors
      negative_plastic_strain,  ///< negative plastic strain which does not allow for evaluations
                                ///< inside the viscoplasticity laws
      overflow_error,  ///< overflow error of the term \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$
                       ///< (and \f$ \boldsymbol{E}^{\text{p}}  = \exp(- \Delta t
                       ///< \dot{\varepsilon}^{\text{p}} \boldsymbol{N}^{\text{p}}) \f$)
      no_flow_resistance,  ///< the material has no flow resistance anymore, such that the
                           ///< evaluations model non-physical phenomena
      failed_solution_linear_system_lnl,  ///< solution of the linear system in the Local
                                          ///< Newton-Raphson Loop failed
      no_convergence_local_newton,  ///< the Local Newton Loop did not converge for the given loop
                                    ///< settings
      singular_jacobian,  ///< singular Jacobian after converged LNL, which does not enable our
                          ///< analytical evaluation of the linearization
      failed_solution_analytic_linearization,  ///< solution of the linear system in the analytical
                                               ///< linearization failed
      failed_computation_flow_resistance,  ///< failed in the computation of the flow resistance via
                                           ///< time integration of the hardening-rate equation
                                           ///(e.g., when using the Anand law)
      failed_computation_flow_resistance_derivs,  ///< failed in the computation of the flow
                                                  ///< resistance derivatives (e.g., when using the
                                                  ///< Anand law)
      failed_matrix_log_evaluation,   ///< failed evaluation of the matrix logarithm or its
                                      ///< derivative
      failed_matrix_exp_evaluation,   ///< failed evaluation of the matrix exponential or its
                                      ///< derivative
      failed_right_cg_interpolation,  ///< failed interpolation of the right Cauchy-Green tensor
      under_yield_surface  ///< mechanical state is "under" the yield surface, i.e., the evaluated
                           ///< stress is smaller than the yield stress
    };


    /// enum class for evaluation management actions in the iterations of the
    /// Local Newton loop
    enum class EvaluationAction
    {
      continue_current_iteration,    ///< continue current iteration
      continue_with_next_iteration,  ///< go to next iteration after performing certain reset steps
      exit_with_error,               ///< exit Local Newton Loop with the set error status
    };

    /// convert error type to detailed error message
    std::string get_detailed_error_message_for_error_type(ErrorType err_type);

    /// enum class for material behavior types
    enum class MatBehavior
    {
      isotropic,         ///< isotropic material behavior
      transv_isotropic,  ///< transversely isotropic material behavior
    };

    /// enum class for time integration types (Local Newton integration)
    enum class TimIntType
    {
      standard,     ///< standard time integration,
      logarithmic,  ///< time integration with logarithmically transformed residual equation for the
                    ///< evolution of the plastic deformation gradient
    };

    /// enum class for material linearization types
    enum class LinearizationType
    {
      analytic,  ///< analytical linearization involving the solution of a linear system of
                 ///< equations,
      perturbation_based,  ///< linearization based on perturbing the current state
    };

    //! matrix exponential and logarithm evaluation utilities
    struct MatrixExpLogUtils
    {
      //! Pade approximation order (to be used consistently: the
      //! derivative of the matrix functions should use the same Pade
      //! order as the evaluation of the matrix functions)
      unsigned int pade_order = 16;  // by default we set the highest order currently implemented
    };

    //! struct containing time step settings and time trackers
    struct TimeStepTracker
    {
      //! time step length
      double dt;
      //! currently computed time instant \f$ t_{n+1} \f$
      double tnp;
      //! minimum substep length
      double min_dt;
    };


    //! struct containing quantities at the last and current time points (i.e., at \f[ t_n \f] and
    //! \f[ t_{n+1} \f], respectively). The quantities are tracked at all Gauss points, in order to
    //! update them simultaneously during the update method call
    struct TimeStepQuantities
    {
      //! right Cauchy-Green deformation tensor at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_rightCG;

      //! inverse plastic deformation gradient at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrad_inverse;

      //! (equivalent) plastic strain at the last time step (for all Gauss points)
      std::vector<double> last_plastic_strain;

      //! equivalent stress at the previous time instant (for all Gauss points)
      std::vector<double> last_equiv_stress;


      //! last (reduced) deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_defgrad;

      //! absolute temperature at the last time instant (for all Gauss points)
      std::vector<double> last_temperature;

      //! temporary variable, for which we store the right Cauchy-Green deformation tensor at each
      //! evaluation (used in order to update last_rightCG_ once outer NR converges) (for all Gauss
      //! points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_rightCG;

      //! current (reduced) deformation gradient: used to check whether the inverse inelastic
      //! deformation gradient has already been evaluated (to improve the computation performance)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_defgrad;

      //! current inverse plastic deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_plastic_defgrad_inverse;

      //! current plastic strain (for all Gauss points)
      std::vector<double> current_plastic_strain;

      //! current equivalent stress (for all Gauss points)
      std::vector<double> current_equiv_stress;


      //! absolute temperature at the current time instant (for all Gauss points)
      std::vector<double> current_temperature;


      //! inverse plastic deformation gradient at the last computed time instant (after the last
      //! converged substep)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_substep_plastic_defgrad_inverse;
      //! plastic strain at the last computed time instant (after the last converged substep)
      std::vector<double> last_substep_plastic_strain;

      /**
       * @brief Set meaningful initial values. Done first for one single Gauss point (extended later
       * on using the resizing function).
       *
       * @param ref_temperature Reference temperature used to set initial values of last/current
       * temperature.
       */
      void init(const double ref_temperature);

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @note The value saved within the first item is taken for all items during resizing. We only
       * enable resizing if the current sizes of the involved vectors are 1
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Perform pre-evaluation tasks at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void pre_evaluate(const unsigned int gp);

      //!  Update values between time steps: last <- current
      void update();

      //! Pack values
      void pack(Core::Communication::PackBuffer& data) const;

      //! Unpack values
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called{false};
    };



    //! struct: constant non-material tensors, such as different
    //! identity tensors
    struct ConstNonMatTensors
    {
      static const ConstNonMatTensors& instance()
      {
        static ConstNonMatTensors instance;
        return instance;
      }

      //! constructor
      ConstNonMatTensors();
      //! second-order 3x3 identity tensor in matrix form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
      //! second-order 3x3 identity in Voigt stress form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 1> id6x1{Core::LinAlg::Initialization::zero};
      //! symmetric identity four tensor of dimension 3 \f$ \mathbb{I}_\text{S} \f$
      Core::LinAlg::Matrix<6, 6> id4_6x6{Core::LinAlg::Initialization::zero};
      //! deviatoric operator \f$ \mathbb{P}_{\text{dev}}  =  \mathbb{I}_\text{S} -
      //! \frac{1}{3} \boldsymbol{I} \otimes \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 6> dev_op{Core::LinAlg::Initialization::zero};
      //! identity fourth-order tensor in Voigt notation: delta_AC delta_BD in index notation
      Core::LinAlg::Matrix<9, 9> id4_9x9{Core::LinAlg::Initialization::zero};
      //! second-order 10x10 identity tensor in matrix form
      Core::LinAlg::Matrix<10, 10> id10x10{Core::LinAlg::Initialization::zero};
    };



    //! struct containing constant tensors which depend on the constant fiber direction \f$
    //! \boldsymbol{m} \f$
    struct ConstMatTensors
    {
      //! \f$ \boldsymbol{I} + \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> id_plus_mm;
      //! \f$ \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> mm{Core::LinAlg::Initialization::zero};
      //! deviatoric part \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //! \right)_\text{dev}\f$
      Core::LinAlg::Matrix<3, 3> mm_dev{Core::LinAlg::Initialization::zero};
      //! \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right) \otimes \left( \boldsymbol{m}
      //! \otimes \boldsymbol{m} \right) \f$ (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dyad_mm{Core::LinAlg::Initialization::zero};
      //!  \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right)_\text{dev} \otimes \left(
      //!  \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dev_dyad_mm{Core::LinAlg::Initialization::zero};
      //!  \f$ \boldsymbol{I} \otimes \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> id_dyad_mm;

      //! set tensors for a given fiber direction \f$ \boldsymbol{m} \f$
      void set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m);
    };

    //! class with local substepping utilities
    class LocalSubsteppingUtils
    {
     public:
      LocalSubsteppingUtils() = delete;
      //! Constructor (calling reset under the hood)
      explicit LocalSubsteppingUtils(double dt) { reset(dt); }

      //! reset routine: set a single substep of a given size dt
      void reset(const double dt);

      //! verify whether the substepping routine has reached its end
      [[nodiscard]] bool end_substepping() const
      {
        return substep_counter_ > total_num_of_substeps_;
      };

      //! increment substep
      void increment_substep();

      //! halve current substep and update relevant quantities
      void halve_substep();

      //! get substep size
      [[nodiscard]] double get_substep_size() const { return curr_dt_; }

      //! retrieve the normalized time parameter at the next time instant during substepping, i.e.,
      //! \f$ \frac{\left(t_{m} + \Delta t_{m}\right)}{\Delta t} \f$, where \f$t_{m}\f$ denotes the
      //! previously converged time instant, \f$\Delta t_{m}\f$ denotes the current substep size,
      //! and \f$\Delta t\f$ specifies the global timestep
      [[nodiscard]] double get_normalized_next_time_param(const double dt) const
      {
        return (t_ + curr_dt_) / dt;
      }

      //! get counter for the current number of time step halving procedures
      [[nodiscard]] unsigned int get_halving_counter() const { return time_step_halving_counter_; }

      //! get substepping info as string
      [[nodiscard]] std::string get_info() const
      {
        std::string out;
        out += "Substepping info: \n";
        out += std::format(
            "t: {}, substep_counter: {}, curr_dt: {}, time_step_halving_counter: {}, "
            "total_num_of_substeps: {} \n",
            t_, substep_counter_, curr_dt_, time_step_halving_counter_, total_num_of_substeps_);
        return out;
      };

     private:
      //! current time parameter ranging from 0 to the problem time step \f$ \Delta t \f$
      double t_;
      //! counter of evaluated substeps
      unsigned int substep_counter_;
      //! current substep size
      double curr_dt_;
      //! number of times the problem time step \f$ \Delta t \f$ has been halved
      unsigned int time_step_halving_counter_;
      //!  total number of substeps to be evaluated within the time step \f$ \Delta t
      //! \f$; this is not always directly proportional to time_step_halving_counter, since the
      //! halving does not have to be uniform (e.g. we could halve the time step twice and still
      //! have 3 substeps to evaluate instead of 4, i.e. if the first substep was evaluable
      //! numerically, but the second substep not, leading to another halving of the substep
      //! length)
      unsigned int total_num_of_substeps_;
    };

    /// enum class for state quantity evaluations in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// plastic strain rate,...)
    enum class StateQuantityEvalType
    {
      full_eval,  ///< full evaluation (full call of the evaluate_state_quantities method)
      plastic_strain_rate_only,  ///< return in evaluate_state_quantities once the plastic strain
                                 ///< rate has been evaluated
      equiv_stress_only,         ///< return in evaluate_state_quantities once the
                                 ///< equivalent stress has been evaluated
    };



    //! struct containing quantities computed from a given elasticity/plasticity state;
    //! given: current right Cauchy-Green deformation tensor, inelastic deformation gradient and
    //! plastic strain at the previous time instant
    struct StateQuantities
    {
      // ----- current state quantities (for the evaluated Gauss points) ----- //

      //! elastic right Cauchy-Green deformation tensor
      Core::LinAlg::Matrix<3, 3> curr_CeM{Core::LinAlg::Initialization::zero};

      //! isotropic stress factors
      Core::LinAlg::Matrix<3, 1> curr_gamma{Core::LinAlg::Initialization::zero};

      //! isotropic constitutive tensor factors
      Core::LinAlg::Matrix<8, 1> curr_delta{Core::LinAlg::Initialization::zero};

      //! elastic 2nd PK stress tensors (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<3, 3> curr_SeM{Core::LinAlg::Initialization::zero};

      //! elastic stiffness tensor (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<6, 6> curr_dSedCe{Core::LinAlg::Initialization::zero};

      //! deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      Core::LinAlg::Matrix<3, 3> curr_Mtheta_dev_sym_M{Core::LinAlg::Initialization::zero};

      //! equivalent tensile stress
      double curr_equiv_stress{0.0};

      //! equivalent plastic strain rate
      double curr_equiv_plastic_strain_rate{0.0};

      //! plastic flow direction tensor
      Core::LinAlg::Matrix<3, 3> curr_NpM{Core::LinAlg::Initialization::zero};

      //! plastic stretching tensor
      Core::LinAlg::Matrix<3, 3> curr_dpM{Core::LinAlg::Initialization::zero};

      //! plastic velocity gradient tensor
      Core::LinAlg::Matrix<3, 3> curr_lpM{Core::LinAlg::Initialization::zero};

      //! plastic update tensor
      Core::LinAlg::Matrix<3, 3> curr_EpM{Core::LinAlg::Initialization::zero};

      //! evaluation type
      StateQuantityEvalType eval_type;
    };

    /// enum class for evaluations of the state quantity derivatives in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// derivatives of the plastic strain rate,...)
    enum class StateQuantityDerivEvalType
    {
      full_eval,  ///< full evaluation (full call of the evaluate_state_quantity_derivatives
                  ///< method)
      plastic_strain_rate_derivs_only,  ///< return in evaluate_state_quantity_derivatives once the
                                        ///< derivatives of the plastic strain rate have been
                                        ///< evaluated
      equiv_stress_derivs_only,  ///< return in evaluate_state_quantities once the derivatives of
                                 ///< the equivalent stress has been evaluated
    };



    //! struct containing specific derivatives of quantities computed from a given
    //! elasticity/plasticity state; given: current right Cauchy-Green deformation tensor,
    //! inelastic deformation gradient and plastic strain at the previous time instant
    struct StateQuantityDerivatives
    {
      // ----- current state variable derivatives (for the evaluated Gauss points)----- //

      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the inverse
      //! inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dCediFin{Core::LinAlg::Initialization::zero};
      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the right
      //! Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dCedC{Core::LinAlg::Initialization::zero};

      //! derivatives of the equivalent tensile stress w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<1, 9> curr_dequiv_stress_diFin{Core::LinAlg::Initialization::zero};
      //! derivatives of the equivalent tensile stress w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress form)
      Core::LinAlg::Matrix<1, 6> curr_dequiv_stress_dC{Core::LinAlg::Initialization::zero};
      //! derivatives of the equivalent tensile stress w.r.t. the temperature
      double curr_dequiv_stress_dT{0.0};

      //! derivative of the deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      //! w.r.t. the inverse inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dMtheta_dev_sym_diFin{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      //! w.r.t. the right Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dMtheta_dev_sym_dC{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      //! w.r.t. the temperature (Voigt stress form)
      Core::LinAlg::Matrix<6, 1> curr_dMtheta_dev_sym_dT{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic strain rate w.r.t. the equivalent stress
      double curr_dpsr_dequiv_stress{0.0};
      //! derivative of the plastic strain rate w.r.t. the equivalent plastic strain
      double curr_dpsr_depsp{0.0};
      //! derivative of the plastic strain rate w.r.t. the temperature
      double curr_dpsr_dT{0.0};

      //! derivative of the plastic stretching tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_ddpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the equivalent plastic strain (Voigt
      //! stress form)
      Core::LinAlg::Matrix<6, 1> curr_ddpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_ddpdC{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the temperature (Voigt stress form)
      Core::LinAlg::Matrix<6, 1> curr_ddpdT{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic velocity gradient tensor w.r.t. the inverse inelastic
      //! deformation gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dlpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the equivalent plastic strain
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 1> curr_dlpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the right Cauchy-Green
      //! deformation tensor (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dlpdC{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the temperature
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 1> curr_dlpdT{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic update tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dEpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the equivalent plastic strain (Voigt
      //! notation)
      Core::LinAlg::Matrix<9, 1> curr_dEpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the right Cauchy-Green deformation tensor
      //! (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dEpdC{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the temperature (Voigt
      //! notation)
      Core::LinAlg::Matrix<9, 1> curr_dEpdT{Core::LinAlg::Initialization::zero};

      //! evaluation type
      StateQuantityDerivEvalType eval_type;
    };

    //! struct containing specific derivatives of the scalar plastic strain rate used in
    //! InelasticDefgradTransvIsotropElastViscoplast
    struct PlasticStrainRateDerivs
    {
      //! derivative with respect to the equivalent stress
      double deriv_equiv_stress = 0.0;

      //! derivative with respect to the plastic strain
      double deriv_plastic_strain = 0.0;

      //! derivative with respect to the temperature
      double deriv_temperature = 0.0;
    };

    /// Derivatives of the history variables wrt. the right Cauchy-Green deformation tensor
    struct HistoryVariablesDerivativesWrtCauchyGreen
    {
      //! derivative of the inverse plastic deformation gradient w.r.t. the right Cauchy-Green
      //! tensor \f$
      //! \frac{\mathrm{d}\boldsymbol{F}_{\mathrm{p},\,n+1}^{-1}}{\mathrm{d}\boldsymbol{C}_{n+1}}
      //! \f$ in Voigt notation (second dimension in stress-form)
      Core::LinAlg::Matrix<9, 6> inv_plastic_defgrad_wrt_cauchy_green{
          Core::LinAlg::Initialization::zero};
      //! derivative of the equivalent plastic strain w.r.t. the right Cauchy-Green tensor
      //! \f$ \frac{\mathrm{d}\varepsilon_{\mathrm{p},\,n+1}}{\mathrm{d}\boldsymbol{C}_{n+1}} \f$ in
      //! stress-form
      Core::LinAlg::Matrix<1, 6> plastic_strain_wrt_cauchy_green{
          Core::LinAlg::Initialization::zero};
    };

    /// Derivatives of the history variables wrt. temperature
    struct HistoryVariablesDerivativesWrtTemperature
    {
      //! derivative of the inverse plastic deformation gradient w.r.t. temperature
      //! \f$ \frac{\mathrm{d}\boldsymbol{F}_{\mathrm{p},\,n+1}^{-1}}{\mathrm{d}T_{n+1}} \f$ in
      //! Voigt notation
      Core::LinAlg::Matrix<9, 1> inv_plastic_defgrad_wrt_temperature{
          Core::LinAlg::Initialization::zero};
      //! derivative of the equivalent plastic strain w.r.t. temperature
      //! \f$ \frac{\mathrm{d}\varepsilon_{\mathrm{p},\,n+1}}{\mathrm{d}T_{n+1}} \f$
      double plastic_strain_wrt_temperature{0.0};
    };

    /**
     * @brief Subset of the `StateQuantities` struct relevant for thermo-mechanical coupling
     *
     * Can be default constructed with all values set to zero, or from a `StateQuantities` struct.
     */
    struct ThermoMechanicalCouplingState
    {
      /// equivalent stress \f$ \bar{\sigma} \f$
      double equiv_stress = 0.0;
      /// equivalent plastic strain rate \f$ \dot{\varepsilon}_\mathrm{p} \f$
      double plastic_strain_rate = 0.0;

      //! default constructor: Set all values to zero
      ThermoMechanicalCouplingState() = default;

      //! construct from state_quantities
      ThermoMechanicalCouplingState(const StateQuantities& state_quantities)
      {
        equiv_stress = state_quantities.curr_equiv_stress;
        plastic_strain_rate = state_quantities.curr_equiv_plastic_strain_rate;
      }
    };

    /**
     * @brief Subset of the `StateQuantityDerivatives` struct relevant for thermo-mechanical
     * coupling
     *
     * Can be default constructed with all values set to zero, or from a `StateQuantityDerivatives`
     * struct.
     */
    struct ThermoMechanicalCouplingStateDerivatives
    {
      /// partial derivative of the equivalent stress w.r.t. the inverse plastic deformation
      /// gradient \f$ \frac{\partial\bar{\sigma}}{\partial\mathbf{F}_\mathrm{p}^{-1}} \f$ in Voigt
      /// notation
      Core::LinAlg::Matrix<1, 9> equiv_stress_wrt_inverse_plastic_defgrad{
          Core::LinAlg::Initialization::zero};
      /// partial derivative of the equivalent stress w.r.t. the right Cauchy-Green deformation
      /// tensor \f$ \frac{\partial\bar{\sigma}}{\partial\mathbf{C}} \f$ in Voigt stress form
      Core::LinAlg::Matrix<1, 6> equiv_stress_wrt_cauchy_green{Core::LinAlg::Initialization::zero};
      /// partial derivative of the equivalent stress w.r.t. the temperature \f$
      /// \frac{\partial\bar{\sigma}}{\partial T} \f$
      double equiv_stress_wrt_temperature = 0.0;
      /// partial derivatives of the plastic strain rate w.r.t. the equivalent stress, plastic
      /// strain and temperature
      PlasticStrainRateDerivs plastic_strain_rate_derivs;

      //! default constructor: Set all values to zero
      ThermoMechanicalCouplingStateDerivatives() = default;

      //! construct from state_quantity_derivatives
      ThermoMechanicalCouplingStateDerivatives(
          const StateQuantityDerivatives& state_quantity_derivatives)
      {
        equiv_stress_wrt_inverse_plastic_defgrad =
            state_quantity_derivatives.curr_dequiv_stress_diFin;
        equiv_stress_wrt_cauchy_green = state_quantity_derivatives.curr_dequiv_stress_dC;
        equiv_stress_wrt_temperature = state_quantity_derivatives.curr_dequiv_stress_dT;
        plastic_strain_rate_derivs = {
            .deriv_equiv_stress = state_quantity_derivatives.curr_dpsr_dequiv_stress,
            .deriv_plastic_strain = state_quantity_derivatives.curr_dpsr_depsp,
            .deriv_temperature = state_quantity_derivatives.curr_dpsr_dT};
      }
    };


    /**
     * @brief This struct holds quantities to be cached across public evaluation calls in
     * thermo-mechanical coupling.
     *
     */
    struct ThermoMechanicalCouplingCache
    {
      CachedQuantity<ThermoMechanicalCouplingState> state;
      CachedQuantity<ThermoMechanicalCouplingStateDerivatives> state_derivatives;
      CachedQuantity<HistoryVariablesDerivativesWrtCauchyGreen> history_variables_wrt_cauchy_green;
      CachedQuantity<HistoryVariablesDerivativesWrtTemperature> history_variables_wrt_temperature;

      /// mark the whole cache as not evaluated at the specified gauss points. This should be done
      /// if the incoming state has changed.
      void reset(const int gp);

      /// resize all cached quantities to the given number of Gauss points
      void resize(const unsigned int numgp);

     private:
      auto quantities()
      {
        return std::tie(state, state_derivatives, history_variables_wrt_cauchy_green,
            history_variables_wrt_temperature);
      }
    };


    /**
     * Returns the derivative of the Taylor-Quinney term wrt. the right Cauchy-Green tensor at time
     * \f[\frac{\mathrm{d}}{\mathrm{d}\mathbf{C}}\left(
     * \xi_\mathrm{TQ}\,\bar{\sigma}\,\dot{\varepsilon}_\mathrm{p}\right)
     * =\xi_\mathrm{TQ}\left(
     * \frac{\mathrm{d}\bar{\sigma}}{\mathrm{d}\mathbf{C}}\,\dot{\varepsilon}_\mathrm{p}
     * +\bar{\sigma}\,\frac{\mathrm{d}\dot{\varepsilon}_\mathrm{p}}
     * {\mathrm{d}\mathbf{C}}\right)\f]
     *
     * with
     * \f[\frac{\mathrm{d}\bar{\sigma}}{\mathrm{d}\mathbf{C}}
     * =\frac{\partial \bar{\sigma}}{\partial \mathbf{C}}
     * +\frac{\partial \bar{\sigma}}{\partial \mathbf{F}_\mathrm{p}^{-1}}
     * :\frac{\mathrm{d} \mathbf{F}_{\mathrm{p},\,n+1}^{-1}}{\mathrm{d} \mathbf{C}_{n+1}}\f]
     *
     * and
     * \f[\frac{\mathrm{d}\dot{\varepsilon}_\mathrm{p}}{\mathrm{d}\mathbf{C}}
     * =\frac{\partial \dot{\varepsilon}_\mathrm{p}}{\partial \varepsilon_\mathrm{p}}
     * \frac{\mathrm{d} \varepsilon_{\mathrm{p},\,n+1}}{\mathrm{d} \mathbf{C}_{n+1}}
     * +\frac{\partial \dot{\varepsilon}_\mathrm{p}}{\partial \bar{\sigma}}
     * \frac{\mathrm{d}\bar{\sigma}}{\mathrm{d}\mathbf{C}}\f]
     */
    Core::LinAlg::Matrix<1, 6> compute_taylor_quinney_wrt_cauchygreen(
        const double taylor_quinney_coefficient, const ThermoMechanicalCouplingState& state,
        const ThermoMechanicalCouplingStateDerivatives& state_derivatives,
        const HistoryVariablesDerivativesWrtCauchyGreen& history_variables_derivatives);

    /**
     * Returns the derivative of the taylor-quinney term wrt. the temperature
     * \f[\frac{\mathrm{d}}{\mathrm{d}T}\left(
     * \xi_\mathrm{TQ}\,\bar{\sigma}\,\dot{\varepsilon}_\mathrm{p}\right)
     * =\xi_\mathrm{TQ}\left(
     * \frac{\mathrm{d}\bar{\sigma}}{\mathrm{d}T}\,\dot{\varepsilon}_\mathrm{p}
     * +\bar{\sigma}\,\frac{\mathrm{d}\dot{\varepsilon}_\mathrm{p}}{\mathrm{d}T}\right)\f]
     *
     * with
     * \f[\frac{\mathrm{d}\bar{\sigma}}{\mathrm{d}T}
     * =\frac{\partial \bar{\sigma}}{\partial T}
     * +\frac{\partial \bar{\sigma}}{\partial \mathbf{F}_\mathrm{p}^{-1}}
     * :\frac{\mathrm{d} \mathbf{F}_{\mathrm{p},\,n+1}^{-1}}{\mathrm{d} T_{n+1}}\f]
     *
     * and
     * \f[\frac{\mathrm{d}\dot{\varepsilon}_\mathrm{p}}{\mathrm{d}T}
     * =\frac{\partial \dot{\varepsilon}_\mathrm{p}}{\partial T}
     * +\frac{\partial \dot{\varepsilon}_\mathrm{p}}{\partial \varepsilon_\mathrm{p}}
     * \frac{\mathrm{d} \varepsilon_{\mathrm{p},\,n+1}}{\mathrm{d} T_{n+1}}
     * +\frac{\partial \dot{\varepsilon}_\mathrm{p}}{\partial \bar{\sigma}}
     * \frac{\mathrm{d}\bar{\sigma}}{\mathrm{d}T}\f]
     */
    double compute_taylor_quinney_wrt_temperature(const double taylor_quinney_coefficient,
        const ThermoMechanicalCouplingState& state,
        const ThermoMechanicalCouplingStateDerivatives& state_derivatives,
        const HistoryVariablesDerivativesWrtTemperature& history_variables_derivatives);


    /// enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonConvCheck
    {
      residual,         ///< verify convergence based on the absolute value of the Local Newton
                        ///< residual 2-norm
      increment_ratio,  ///< verify convergence based on the ratio of solution increment to current
                        ///< solution: \f$ \frac{\left| \Delta \boldsymbol{s}^{l+1} \right|}{\left|
                        ///< \boldsymbol{s}^{l} \right|}  \f$
      residual_and_increment_ratio,  ///< verify convergence based on both the absolute Local Newton
                                     ///< residual and the ratio of solution increment to current
                                     ///< solution
    };


    /// enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonDiverCont
    {
      stop,          ///< stop the simulation entirely
      continue_sim,  ///<  continue the simulation, and display warning in regards to the current
                     ///<  state within the Local Newton Loop
      continue_sim_with_safeguard  ///< continue the simulation only if the convergence tolerances
                                   ///< are not exceeded excessively, as specified with specific
                                   ///< exceedance factors for the tolerances
    };

    /// enum: quantities relevant for convergence checking within the Local Newton Loop
    struct LocalNewtonConvQuantities
    {
      //! residual 2-norm
      double residual_norm;

      //! ratio of solution increment to current solution: \f$ \frac{\left| \Delta
      //! \boldsymbol{s}^{l+1} \right|}{\left| \boldsymbol{s}^{l} \right|}  \f$
      double increment_norm;
    };



    //! struct containing parameter specifications for the Local Newton loop
    struct LocalNewtonParams
    {
      //! convergence tolerance: absolute residual value
      const double res_tol;

      //! convergence tolerance: ratio of solution increment to current solution
      const double incr_tol;

      //! convergence check strategy
      const LocalNewtonConvCheck conv_check;

      //! strategy for dealing with divergence
      const LocalNewtonDiverCont diver_cont;

      //! maximum number of local iterations
      const unsigned int max_iter;

      //! maximum exceedance factor for the residual tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      const double max_exceedance_fact_res_tol;

      //! maximum exceedance factor for the solution increment tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      const double max_exceedance_fact_incr_tol;
    };

    //! class for managing the Local Newton loop, containing the utilized parameters and iteration
    //! data
    class LocalNewtonManager
    {
     public:
      LocalNewtonManager() = delete;
      /*!
       * @brief Constructor
       *
       * @param[in] lnl_params Local Newton parameters
       *
       */
      explicit LocalNewtonManager(const LocalNewtonParams& lnl_params);

      /// getter for Local Newton parameters
      [[nodiscard]] LocalNewtonParams params() const { return params_; }

      /// getter for local iteration count
      [[nodiscard]] unsigned int iter() const { return iter_; }

      /// setter for local iteration count
      void set_iteration_count(const unsigned int iter) { iter_ = iter; }

      /// getter for total number of local iterations evaluated in this time step (vector over all
      /// Gauss points)
      [[nodiscard]] const std::vector<unsigned int>& curr_num_iters() const
      {
        return curr_num_iters_;
      }

      /// increment iteration count by 1
      void increment_iteration_count() { iter_++; }

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Routine to be run after the Local Newton-Raphson at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void update_after_local_newton(const unsigned int gp);

      //! reset method
      void reset();

      //! pack values
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack values
      void unpack(Core::Communication::UnpackBuffer& buffer);

     private:
      //! Local Newton parameters
      const LocalNewtonParams params_;

      //! current local iteration
      unsigned int iter_;

      //! total number of local iterations for the current timestep; vector of Gauss point values
      std::vector<unsigned int> curr_num_iters_;

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called_{false};
    };



  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
