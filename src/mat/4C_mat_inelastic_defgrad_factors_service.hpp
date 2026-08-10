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
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
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


    /// struct: settings for registering errors within the procedures used for the constitutive
    /// update
    struct ErrorRegistrationSettings
    {
      //! should overflow error be registered via ErrorType when the plastic strain increment
      //! exceeds the specified tolerance?
      bool register_plastic_strain_incr_overflow;

      //! maximum, numerically evaluable plastic strain increment before overflow error is
      //! registered?
      double max_plastic_strain_incr;

      //! should overflow error be registered via ErrorType when any of the plastic strain
      //! derivative increments exceeds the specified tolerance?
      bool register_plastic_strain_deriv_incr_overflow;

      //! maximum, numerically evaluable increment of plastic strain derivatives (time_step *
      //! derivative)
      double max_plastic_strain_deriv_incr;
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

      //!  Update values between time steps: last <- current, at a given Gauss points
      void update(const unsigned int gp);

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
        out += "\nSubstepping info: \n";
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

      //! thermal contribution due to thermal expansion to the thermo-elastic 2PK stress
      TensorAndTemperatureDerivative curr_ST{};

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
      //! \boldsymbol{s}^{l} \right|}{\left| \boldsymbol{s}^{l} \right|}  \f$
      double increment_norm;
    };



    //! struct containing parameter specifications for the Local Newton loop
    struct LocalNewtonParams
    {
      //! convergence tolerance: absolute residual value
      double res_tol;

      //! convergence tolerance: ratio of solution increment to current solution
      double incr_tol;

      //! convergence check strategy
      LocalNewtonConvCheck conv_check;

      //! strategy for dealing with divergence
      LocalNewtonDiverCont diver_cont;

      //! maximum number of local iterations
      int max_iter;

      //! maximum exceedance factor for the residual tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      double max_exceedance_fact_res_tol;

      //! maximum exceedance factor for the solution increment tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      double max_exceedance_fact_incr_tol;
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

      /// getter for total number of local iterations evaluated in this time step (vector over all
      /// Gauss points)
      [[nodiscard]] const std::vector<unsigned int>& curr_num_iters() const
      {
        return curr_num_iters_;
      }

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Initialize the solution vector and the convergence quantities, for the
       * subsequent Local Newton at the currently considered Gauss point
       *
       * @param[in] init_estimate initial estimate \f$ \boldsymbol{s}^{(0)} \f$
       */
      void save_init_estimate_and_reset_convergence_quantities(
          const Core::LinAlg::Matrix<10, 1>& init_estimate);

      /// reset iteration counter
      void reset_iter() { iter_ = 0; }

      /// sets the residual norm based on the given residual vector
      void set_residual_norm(const Core::LinAlg::Matrix<10, 1>& residual)
      {
        convergence_quantities_.residual_norm = residual.norm2();
      }

      /*!
       * @brief Determine whether the Local Newton Loop has converged, based on the saved
       * convergence quantities and the specified convergence checks.
       *
       * @return boolean: true = converged
       */
      [[nodiscard]] bool is_local_newton_converged() const;


      /*!
       * @brief   After an unsuccessful convergence check: determine whether the Local Newton is
       * stuck / stagnates, i.e., the relative solution increment is nearly 0, but there is no
       * convergence yet, based on the saved convergence quantities.
       *
       * @return boolean: true = stuck
       */
      [[nodiscard]] bool is_local_newton_stuck() const;


      /// is the maximum number of iterations reached?
      [[nodiscard]] bool is_max_iter_reached()
      {
        return iter_ >= static_cast<unsigned int>(params_.max_iter);
      }


      /// increment iteration counter
      void increment_iter() { ++iter_; }

      /*!
       * @brief Increments the solution vector \f$ \boldsymbol{s}^{(l+1)} = \boldsymbol{s}^{(l)}
       * +
       * \Delta \boldsymbol{s}^{(l+1)} \f$ after the current iteration \f$ l \f$
       *
       * @note Also updates the increment norm (ratio of increment to solution) internally based on
       * the provided increment
       *
       * @param[in] delta_sol increment vector for the next iteration \f$\Delta
       * \boldsymbol{s}^{(l+1)}\f$
       */
      void increment_solution_vector(const Core::LinAlg::Matrix<10, 1>& delta_sol);

      /// getter for the solution vector
      [[nodiscard]] Core::LinAlg::Matrix<10, 1> sol() const { return sol_; }


      /// getter for the convergence quantities
      [[nodiscard]] LocalNewtonConvQuantities convergence_quantities() const
      {
        return convergence_quantities_;
      }

      /*!
       * @brief Routine to be run after the Local Newton-Raphson at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void update_after_local_newton(const unsigned int gp);

      //! reset the saved number of iterations at a given Gauss point
      void reset_curr_num_iters(const unsigned int gp);

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

      //! solution vector in the current iteration \f$ \boldsymbol{s}^{(l)} \f$ (at the
      //! currently considered Gauss point)
      Core::LinAlg::Matrix<10, 1> sol_;

      //! quantities used for convergence checks
      LocalNewtonConvQuantities convergence_quantities_;

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called_{false};
    };

    //! configuration struct for LocalIntegrationInput
    struct LocalIntegrationInputConfig
    {
      //! deformation gradient \f$ \boldsymbol{F}_{n+1} \f$
      Core::LinAlg::Matrix<3, 3> defgrad;

      //! absolute temperature \f$ T_{n+1} \f$
      double temperature;

      //! previous inverse inelastic/plastic deformation gradient \f$ \mathbf{F}_{\text{p},n}^{-1}
      //! \f$
      Core::LinAlg::Matrix<3, 3> last_inv_inelastic_defgrad;

      //! previous plastic strain \f$ \varepsilon_{\text{p},n} \f$
      double last_plastic_strain;

      //! timestep / substep size \f$ \Delta t \f$ / \f$ \Delta \tilde{t} \f$
      double step;
    };

    //! helper struct containing relevant input for the local time integration via the Local
    //! Newton-Raphson scheme
    struct LocalIntegrationInput
    {
      //! constructor based on a given config
      explicit LocalIntegrationInput(const LocalIntegrationInputConfig& cfg);

      //! deformation gradient \f$ \mathbf{F}_{n+1} \f$
      Core::LinAlg::Matrix<3, 3> defgrad;

      //! inverse deformation gradient \f$ \mathbf{F}_{n+1}^{-1} \f$
      Core::LinAlg::Matrix<3, 3> inv_defgrad;

      //! right Cauchy-Green deformation tensor \f$ \mathbf{C}_{n+1} \f$
      Core::LinAlg::Matrix<3, 3> right_cg;

      //! inverse plastic deformation gradient within the elastic predictor \f$
      //! \left[ \mathbf{F}_{\mathrm{p},n+1}^{(\mathrm{E})} \right]^{-1} \f$
      Core::LinAlg::Matrix<3, 3> elastic_predictor_inverse_plastic_defgrad;

      //! elastic deformation gradient within the elastic predictor \f$
      //! \mathbf{F}_{\mathrm{e},n+1}^{(\mathrm{E})} \f$
      Core::LinAlg::Matrix<3, 3> elastic_predictor_elastic_defgrad;

      //! absolute temperature \f$ T_{n+1} \f$
      double temperature;

      //! accumulated plastic strain at the previous time instant \f$ \varepsilon_{\mathrm{P},n} \f$
      double last_plastic_strain;

      //! timestep / substep size \f$ \Delta t \f$ / \f$ \Delta \tilde{t} \f$
      double step;
    };

    //! namespace containing utilities dedicated to the Adaptive Estimate Interpolation algorithm,
    //! presented in Ana, Schmidt, Wall: Adaptive Estimate Interpolation: Accelerating Local
    //! Newton-Raphson Schemes in Computational Plasticity and Viscoplasticity, Preprint
    namespace AdaptiveEstimateInterpolation
    {

      //! namespace containing specifications for the preliminary plastic predictor used within the
      //! Adaptive Estimate Interpolation
      namespace PrelimPlasticPredictor
      {
        //! strategy for choosing the elastic stretch eigenvalues \f$ \boldsymbol{\Lambda} \f$
        enum class ElasticStretchEigenvalType : std::uint8_t
        {
          scale_unit,  ///< the unit tensor is scaled with the deformation gradient determinant to
                       ///< maintain plastic incompressibility: \f$ \boldsymbol{\Lambda} =
                       ///< \det(\boldsymbol{F}_{n+1})^{1/3} \boldsymbol{I} \f$
        };

        //! strategy for choosing the elastic stretch eigenvectors \f$ \boldsymbol{Q} \f$
        enum class ElasticStretchEigenvectType : std::uint8_t
        {
          from_elastic_predictor,  ///< the elastic stretch eigenvectors are taken directly from the
                                   ///< elastic predictor, which is a consistent assumption for
                                   ///< isotropic material behavior
        };

        //! strategy for choosing the elastic stretch rotations \f$ \boldsymbol{R} \f$
        enum class ElasticRotationType : std::uint8_t
        {
          from_elastic_predictor,  ///< the elastic rotation is taken directly from the
                                   ///< elastic predictor, which is a consistent assumption for
                                   ///< isotropic material behavior
        };
      }  // namespace PrelimPlasticPredictor

      //! starting point type to be used for the estimate interpolation between predictors
      enum class StartingPointType : std::uint8_t
      {
        user_set,             ///< user-set constant factor
        equiv_stress_history  ///< computes the interpolation factor based on the equivalent
                              ///< stress from the previous timestep with respect to its
                              ///< corresponding elastic and plastic predictors, see \emph{IH}
                              ///< strategy in the paper
      };

      //! enum class: method to be used for handling the hardening variables within the Adaptive
      //! Estimate Interpolation procedures
      enum class HardeningMethod : std::uint8_t
      {
        use_previous,  ///< use hardening variables from the previously converged time step
        integrate_via_evol_eqs,  ///< integrate the hardening variables via their dedicated
                                 ///< evolution equations, using the interpolated elastic
                                 ///< deformation gradient as input --> "smaller local Newton"
      };

      //! struct containing parameters for the iterative construction of the plastic predictor
      struct PlasticPredictorConstructionParams
      {
        //! elastic stretch eigenvalue specification for the preliminary plastic predictor
        PrelimPlasticPredictor::ElasticStretchEigenvalType elastic_stretch_eigenval_type;

        //! elastic stretch eigenvector specification for the preliminary plastic predictor
        PrelimPlasticPredictor::ElasticStretchEigenvectType elastic_stretch_eigenvect_type;

        //! elastic rotation specification for the preliminary plastic predictor
        PrelimPlasticPredictor::ElasticRotationType elastic_rotation_type;

        //! maximum number of construction iterations \f$ i_{\text{C,max}} \f$
        int max_iter;

        //! relative understress tolerance $\kappa_{\text{\sigma}_{\text{Y}}}$ for \f$
        //! \overline{\sigma}(\tau) / \sigma_{\text{Y},n} - 1
        //! \f$
        double relative_understress_tol;

        //! interval scanning parameter $s$ for updating the construction parameter \f$\tau \gets
        //! \tau_{\text{E}} + s \, \left( \tau_{\hat{\text{P}}} - \tau_{\text{E}} \right) \f$
        //! (bisection: = 1/2)
        double interval_scanning_param;
      };


      //! struct containing parameters for the estimate interpolation between the elastic and the
      //! constructed plastic predictor
      struct EstimateInterpolationParams
      {
        //! starting point type
        StartingPointType starting_point_type;

        //! specified starting point in case that the starting point is user_set
        double user_set_starting_point;

        //! maximum number of estimate interpolation iterations \f$ i_{\text{EI,max}} \f$
        int max_iter;

        //! interval scanning parameter $s$ for updating the interpolation parameter \f$\xi \gets
        //! \xi_{\text{E}} + s \, \left( \xi_{\text{P}} - \xi_{\text{E}} \right) \f$ (bisection: =
        //! 1/2)
        double interval_scanning_param;
      };


      //! struct containing parameters dedicated to handling the hardening variables
      struct HardeningParams
      {  //! method to use for handling / "interpolating" the hardening variables
        HardeningMethod method;

        //! maximum number of iterations for hardening integration via the evolution equations
        int max_iter_integration;

        //! tolerance for hardening integration via the evolution equations
        double tol_integration;
      };


      //! struct containing parameters for the re-estimation procedures
      struct ReestimationParams
      {
        //! maximum number of adaptive re-estimations allowed
        int max_num_reestimations;

        //! interval scanning parameter $s$ for determining the intermediate parameter \f$
        //! \xi_{\mathrm{I}} \gets
        //! \xi_{\mathrm{E}} + s \, \left( \xi - \xi_{\mathrm{E}}\right) \f$ (bisection: = 1 / 2)
        double interval_scanning_param;
      };


      //! struct containing preconditioning settings (for the elastic predictor)
      struct PreconditioningSettings
      {
        //! precondition the elastic deformation gradient within the elastic predictor to
        //! stabilize interpolation, i.e., components smaller than a set tolerance are set to 0.0 to
        //! avoid unnecessary, "numerical" rotations
        bool precondition_elastic_pred;

        //! tolerance for preconditioning the elastic deformation gradient within the elastic
        //! predictor to stabilize interpolation
        double tol_precondition_elastic_pred;
      };

      //! struct: parameters used for the AEI routines (main parameter set for the scheme)
      struct AEIParams
      {
        //! is the Adaptive Estimate Interpolation used?
        bool use_adaptive_estimate_interpolation;

        //! preconditioning settings
        PreconditioningSettings preconditioning;

        //! parameters for plastic predictor construction
        PlasticPredictorConstructionParams plastic_predictor_construction;

        //! parameters for estimate interpolation between predictors
        EstimateInterpolationParams estimate_interpolation;

        //! hardening parameters
        HardeningParams hardening;

        //! re-estimation parameters
        ReestimationParams reestimation;
      };

      //! struct: input for the interpolation point determination based on the equivalent
      //! stress of the previous solution between both predictors (I_HIST method from paper:
      //! $\hat{\xi}_{n+1} = \text{min}[1, \text{max}(\frac{\overline{\sigma}_n -
      //! \overline{\sigma}_n^{\text{E}}}{\overline{\sigma}_{n}^{\text{P}} -
      //! \overline{\sigma}_{n}^{\text{E}}},0)]$)
      struct InputEquivStressStartingPoint
      {
        //! equivalent stress of the solution: \f$ \overline{\sigma}_{n} \f$
        double equiv_stress_solution;

        //! equivalent stress of the elastic predictor: \f$ \overline{\sigma}_{n}^{\text{E}} \f$
        double equiv_stress_elast_pred;

        //! equivalent stress of the plastic predictor: \f$ \overline{\sigma}_{n}^{\text{P}} \f$
        double equiv_stress_plast_pred;
      };

      //! Interpolator of elastic deformation gradients between the elastic and the plastic
      //! predictors. This class stores the values of the involved quantities at
      //! all Gauss points.
      class PredictorInterpolator
      {
       public:
        //! constructor
        PredictorInterpolator();

        //! resizing based on a given number of Gauss points
        void resize(const unsigned int numgp);

        //! pack method
        void pack(Core::Communication::PackBuffer& data) const;

        //! unpack method
        void unpack(Core::Communication::UnpackBuffer& buffer);

        /*!
         * @brief Constructs a preliminary plastic predictor given the elastic predictor and the
         * specified AEI settings.
         *
         *
         * @param[in] gp Gauss point index
         * @param[in] elastic_defgrad_elastic_pred elastic deformation gradient within the elastic
         * predictor
         * @param[in] preconditioning_settings settings for preconditioning the elastic predictor
         * @param[in] plastic_predictor_construction_params parameters for plastic predictor
         * construction
         */
        void construct_prelim_plastic_pred(const unsigned int gp,
            const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_elastic_pred,
            const PreconditioningSettings& preconditioning_settings,
            const PlasticPredictorConstructionParams& plastic_predictor_construction_params);

        /*!
         * @brief Interpolates an elastic deformation gradient based on the stored predictor
         * quantities.
         *
         *
         * @param[in] gp Gauss point index
         * @param[in] interp_loc location used for interpolation; either \f$ \tau \f$ (plastic
         * predictor construction) or \f$ \xi \f$ (estimate interpolation)
         */
        Core::LinAlg::Matrix<3, 3> interpolate_elastic_defgrad(
            const unsigned int gp, const double interp_loc) const;

        /*!
         * @brief After the plastic predictor construction algorithm has succeeded in finding the
         * construction parameter \f$\tau \f$ associated with the plastic predictor, this function
         * sets the determined plastic predictor (more specifically: all class variables
         * associated with it).
         *
         *
         * @param[in] gp Gauss point index
         * @param[in] plastic_pred_loc location \f$ \tau \f$ determined in the plastic
         * predictor construction
         */
        void set_plastic_predictor_after_construction_algo(
            const unsigned int gp, const double plastic_pred_loc);

       private:
        /*!
         * @brief Interpolates eigenvalues and rotational contributions based on the stored
         * predictor quantities.
         *
         *
         * @param[in] gp Gauss point index
         * @param[in] interp_loc location used for interpolation; either \f$ \tau \f$ (plastic
         * predictor construction) or \f$ \xi \f$ (estimate interpolation)
         * @param[out] interp_rel_rot_quat interpolated relative elastic rotation quaternion
         * @param[out] interp_eigenval interpolated elastic eigenvalues (stored in descending order)
         * @param[out] interp_rel_eigenvect_rot_quat interpolated relative elastic eigenvector
         * quaternion
         */
        void interpolate_elastic_defgrad_contributions(const unsigned int gp,
            const double interp_loc, Core::LinAlg::Matrix<4, 1>& interp_rel_rot_quat,
            std::vector<double>& interp_eigenval,
            Core::LinAlg::Matrix<4, 1>& interp_rel_eigenvect_rot_quat) const;

        //! elastic predictor: elastic eigenvalue tensors \f$
        //! \boldsymbol{\Lambda}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{E})}}
        //! \f$ of all Gauss points
        std::vector<Core::LinAlg::Matrix<3, 3>> eigenval_elast_pred_;
        //! plastic predictor: elastic eigenvalue tensors \f$
        //! \boldsymbol{\Lambda}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{P})}} \f$ of all Gauss points
        std::vector<Core::LinAlg::Matrix<3, 3>> eigenval_plast_pred_;
        //! eigenvalue data storage to be directly used within the scalar interpolator; indexing:
        //! - first dimension: Gauss point.
        //! - second dimension: predictor type
        //!   (0 = elastic predictor, 1 = plastic predictor).
        //! - third dimension: eigenvalues corresponding to the selected predictor, stored in
        //! descending order.
        std::vector<std::vector<std::vector<double>>> scalar_interp_eigenval_;
        //! interpolation locations for the elastic and plastic predictor, to be directly used
        //! within the scalar interpolator; indexing:
        //! - 0: elastic predictor location.
        //! - 1: plastic predictor location.
        const std::vector<Core::LinAlg::Matrix<1, 1>> ref_predictor_locs_;
        //! elastic predictor: elastic stretch eigenvector tensors \f$
        //! \boldsymbol{Q}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{E})}} \f$ of all Gauss points
        std::vector<Core::LinAlg::Matrix<3, 3>> eigenvect_rot_elast_pred_;
        //! plastic predictor: relative elastic eigenvector quaternions \f$
        //! \boldsymbol{q}_{\boldsymbol{Q}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{P})}}, \mathrm{rel}}
        //! \f$ associated with \f$
        //! \boldsymbol{Q}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{P})}, \mathrm{rel}} =
        //! \boldsymbol{Q}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{E})}}^T
        //! \boldsymbol{Q}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{P})}}
        //! \f$ of all Gauss points
        std::vector<Core::LinAlg::Matrix<4, 1>> rel_eigenvect_rot_plast_pred_;
        //! elastic predictor: elastic rotation tensors \f$
        //! \boldsymbol{R}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{E})}} \f$ of all Gauss points
        std::vector<Core::LinAlg::Matrix<3, 3>> rot_elast_pred_;
        //! plastic predictor: relative elastic rotation quaternions \f$
        //! \boldsymbol{r}_{\boldsymbol{R}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{P})}, \mathrm{rel}}}
        //! \f$ associated with \f$ \boldsymbol{R}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{P})},
        //! \mathrm{rel}} =
        //! \boldsymbol{R}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{E})}}^T
        //! \boldsymbol{R}_{\boldsymbol{F}_{\text{e}}^{(\mathrm{P})}}\f$ of all Gauss points
        std::vector<Core::LinAlg::Matrix<4, 1>> rel_rot_plast_pred_;
        //! scalar eigenvalue interpolator
        Core::LinAlg::ScalarInterpolator<1> eigenval_interpolator_;
        //! tracks whether the resizing function has been called, to set the current number of Gauss
        //! points exactly once!
        bool resize_called_{false};
      };

      //! Container for interpolation points / bounds used in the
      //! Adaptive Estimate Interpolation. This class stores the values of the involved quantities
      //! at all Gauss points.
      struct InterpolationPointContainer
      {
       public:
        /*!
         * @brief Constructor
         *
         * @param[in] estimate_interpolation_params parameters for estimate interpolation between
         * predictors
         */
        explicit InterpolationPointContainer(
            const EstimateInterpolationParams& estimate_interpolation_params);

        //! reset interpolation interval and set the current interpolation point to its saved
        //! starting point at a given Gauss point
        void reset_bounds_and_current_interp_point(const unsigned int gp);

        //! resizing based on the given number of Gauss points
        void resize(const unsigned int numgp);

        //! pack method
        void pack(Core::Communication::PackBuffer& data) const;

        //! unpack method
        void unpack(Core::Communication::UnpackBuffer& buffer);

        //! current interpolation points \f$ \xi \f$ of all Gauss points
        std::vector<double> current_interp_points;

        //! lower interpolation bounds \f$ \xi_{\text{E}} \f$ of all Gauss points
        std::vector<double> lower_interp_bounds;

        //! upper interpolation bounds \f$ \xi_{\text{P}} \f$ of all Gauss points
        std::vector<double> upper_interp_bounds;

        //! starting points for interpolation \f$ \hat{\xi} \f$ of all Gauss points
        std::vector<double> starting_points;

        //! tracks whether the resizing function has been called, to set the current number of
        //! Gauss points exactly once!
        bool resize_called{false};
      };

      //! Shift direction for the interpolation intervals
      enum class InterpolationIntervalShift : std::uint8_t
      {
        towards_elastic_pred,  ///< shift towards the elastic predictor (e.g., in case of
                               ///< vanishing plastic strain increments)
        towards_plastic_pred,  ///< shift towards the plastic predictor (e.g., in case of
                               ///< overflow errors due to high overstresses)
      };

      /// Types of interpolation points / bounds which can be used to set the current interpolation
      /// point
      enum class CurrentInterpPointPreset : std::uint8_t
      {
        plastic_pred_construct_update,  ///< update current interpolation point between the lower
                                        ///< and upper bound for the plastic predictor
                                        ///< construction:
                                        ///< (\f$
                                        ///< \tau \gets \tau_{\text{E}} +  s \, \left(
                                        ///< \tau_{\text{P}} - \tau_{\text{E}}
                                        ///< \right) \f$)
        estimate_interpolation_update,  ///< update current interpolation point between the
                                        ///< lower and upper bound for the estimate interpolation
                                        ///< between predictors:
                                        ///< (\f$
                                        ///< \xi \gets \xi_{\text{E}} + s \, \left(
                                        ///< \xi_{\text{P}} - \xi_{\text{E}}
                                        ///< \right) \f$)
        lower_interp_bound,             ///< elastic predictor (\f$ \xi \gets \xi_{\text{E}} \f$)
        upper_interp_bound,             ///< elastic predictor (\f$ \xi \gets \xi_{\text{P}} \f$)
        elastic_predictor,              ///< elastic predictor (\f$ \xi \gets 0.0 \f$)
        plastic_predictor,              ///< plastic predictor (\f$ \xi \gets 1.0 \f$)
        starting_point,                 ///< starting point (\f$ \xi \gets \hat{\xi} \f$)
        intermediate_point  ///<  intermediate point between lower bound and current interpolation
                            ///<  point:
                            ///< (\f$
                            ///< \xi \gets \xi_{\text{E}} + s \, \left(
                            ///< \xi - \xi_{\text{E}}
                            ///< \right) \f$)
      };

      //! class: manager for the Adaptive Estimate Interpolation algorithm across all Gauss points
      class AEIManager
      {
       public:
        AEIManager() = delete;

        /*!
         * @brief Constructor
         *
         * @param[in] aei_params Adaptive Estimate Interpolation parameters
         */
        explicit AEIManager(const AEIParams& aei_params);

        //! resize method: set the correct number of Gauss Points
        void resize(const unsigned int num_gp);

        //! get info as string
        [[nodiscard]] std::string get_info(const unsigned int gp) const
        {
          std::string out;
          out += "\nAdaptive estimate interpolation info: \n";
          out += std::format(
              "number of plastic predictor construction iterations: {} / {}, number of estimate "
              "interpolation iterations: {} / {}, number of re-estimations: {} / {}, interpolation "
              "interval: [{}, {}], "
              "current interpolation point: {} \n",
              num_plastic_pred_construct_iters_, params_.plastic_predictor_construction.max_iter,
              num_estimate_interp_iters_, params_.estimate_interpolation.max_iter,
              num_reestimations_, params_.reestimation.max_num_reestimations,
              lower_interp_bound(gp), upper_interp_bound(gp), current_interp_point(gp));
          return out;
        };


        /*!
         * @brief Verify whether plastic predictor construction is still possible, based on the set
         * maximum number of iterations
         *
         *
         */
        bool is_plastic_pred_construct_possible()
        {
          return (
              num_plastic_pred_construct_iters_ < params_.plastic_predictor_construction.max_iter);
        };

        /*!
         * @brief Verify whether estimate interpolation is still possible, based on the set maximum
         * number of interpolation iterations
         *
         */
        bool is_estimate_interp_possible()
        {
          return (num_estimate_interp_iters_ < params_.estimate_interpolation.max_iter);
        };

        /*!
         * @brief Verify whether re-estimations are still possible, based on the
         * set maximum number of re-estimations
         *
         */
        bool is_reestimation_possible()
        {
          return (num_reestimations_ < params_.reestimation.max_num_reestimations);
        };

        /*!
         * @brief Reset tasks and construction of the preliminary plastic
         * predictor at a given Gauss point
         *
         * @param[in] gp Gauss point index
         * @param[in] local_integration_input input data used for local time integration
         */
        void reset_and_construct_prelim_plastic_pred(
            const unsigned int gp, const LocalIntegrationInput& local_integration_input);


        //! pack method
        void pack(Core::Communication::PackBuffer& data) const;

        //! unpack method
        void unpack(Core::Communication::UnpackBuffer& buffer);

        /*!
         * @brief Interpolate the inverse inelastic deformation gradient required by the
         * viscoplastic material.
         *
         * @note The interpolation takes place between the elastic deformation gradients
         * associated with the elastic and the plastic predictors at the current
         * interpolation point saved internally.
         *
         *
         * @param[in] gp Gauss point index
         * @param[in] inv_defgrad inverse deformation gradient
         */
        Core::LinAlg::Matrix<3, 3> interpolate_inverse_inelastic_defgrad(
            const unsigned gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad);

        /*!
         * Retrieves the inverse inelastic deformation gradient associated with the plastic
         * predictor, via interpolation using \f$ \xi = 1.0 \f$, at the specified Gauss point
         *
         * @param[in] gp Gauss point index
         * @param[in] inv_defgrad inverse deformation gradient
         */
        Core::LinAlg::Matrix<3, 3> get_inverse_inelastic_defgrad_plastic_pred(
            const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad);

        /*!
         * @brief Sets the plastic predictor quantities based on the current interpolation point;
         * then resets the interpolation point container consistently, at the specified Gauss point
         *
         * @param[in] gp Gauss point
         */
        void set_plastic_predictor_after_construction_algo(const unsigned int gp);

        /*!
         * @brief Adapt interpolation / construction intervals \f$ \left[\xi_{\mathrm{E}},
         * \xi_{\mathrm{P}} \right] \f$ based on evaluation error
         *
         * @param[in] gp Gauss point
         * @param[in] interval_shift interval shift direction
         */
        void adapt_interpolation_interval(
            const unsigned int gp, const InterpolationIntervalShift& interval_shift);

        //! get current interpolation point at a specified Gauss point
        [[nodiscard]] double current_interp_point(const unsigned int gp) const
        {
          FOUR_C_ASSERT(
              gp < interp_point_container_.current_interp_points.size(), "GP index out of range");
          return interp_point_container_.current_interp_points[gp];
        }

        //! set current interpolation point at the specified Gauss point using a given preset
        void set_current_interp_point(const unsigned int gp, const CurrentInterpPointPreset preset);

        //! get lower interpolation bound at the specified Gauss point
        [[nodiscard]] double lower_interp_bound(const unsigned int gp) const
        {
          FOUR_C_ASSERT(
              gp < interp_point_container_.lower_interp_bounds.size(), "GP index out of range");
          return interp_point_container_.lower_interp_bounds[gp];
        }

        //! get upper interpolation bound at the specified Gauss point
        [[nodiscard]] double upper_interp_bound(const unsigned int gp) const
        {
          FOUR_C_ASSERT(
              gp < interp_point_container_.upper_interp_bounds.size(), "GP index out of range");
          return interp_point_container_.upper_interp_bounds[gp];
        }

        //! get starting point at the specified Gauss point
        [[nodiscard]] double starting_point(const unsigned int gp) const
        {
          FOUR_C_ASSERT(
              gp < interp_point_container_.starting_points.size(), "GP index out of range");
          return interp_point_container_.starting_points[gp];
        }

        //! set starting point at a specified Gauss point to the user-set value
        void set_user_starting_point(const unsigned gp);

        //! set starting point at a specified Gauss point based on the equivalent stress from the
        //! previous solution (see I_HIST method from the paper)
        void set_stress_based_starting_point(
            const unsigned gp, InputEquivStressStartingPoint input_equiv_stress_starting_point);

        //! increment number of re-estimations
        void increment_num_reestimations() { ++num_reestimations_; }

        //! disable further re-estimations
        void disable_further_reestimations()
        {
          num_reestimations_ = params_.reestimation.max_num_reestimations + 1;
        }

        //! reset number of estimate interpolation iterations
        void reset_num_estimate_interp_iters() { num_estimate_interp_iters_ = 0; }

        //! increment number of estimate interpolation iterations
        void increment_num_estimate_interp_iters() { ++num_estimate_interp_iters_; }

        //! reset number of plastic predictor construction iterations
        void reset_num_plastic_pred_construct_iters() { num_plastic_pred_construct_iters_ = 0; }

        //! increment number of plastic predictor construction iterations
        void increment_num_plastic_pred_construct_iters() { ++num_plastic_pred_construct_iters_; }

       private:
        //! Adaptive Estimate Interpolation parameters
        const AEIParams params_;

        //! tracks whether the resizing function has been called, to set the current number of
        //! Gauss points exactly once!
        bool resize_called_{false};

        //! current number of plastic predictor construction iterations at the currently evaluated
        //! Gauss point
        int num_plastic_pred_construct_iters_;

        //! current number of estimation interpolation iterations at the currently evaluated Gauss
        //! point
        int num_estimate_interp_iters_;

        //! current number of re-estimations at the currently evaluated Gauss point
        int num_reestimations_;

        //! container of interpolation points / bounds for all Gauss points
        InterpolationPointContainer interp_point_container_;

        //! predictor interpolator containing data and interpolation logic for the elastic and
        //! plastic predictors for all Gauss points
        PredictorInterpolator predictor_interpolator_;
      };
    }  // namespace AdaptiveEstimateInterpolation
  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
