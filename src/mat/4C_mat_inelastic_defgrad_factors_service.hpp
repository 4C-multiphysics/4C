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
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>
#include <string>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// namespace: utilities for
  /// InelasticDefgradTransvIsotropElastViscoplast
  namespace InelasticDefgradTransvIsotropElastViscoplastUtils
  {
    /// declare numerical tolerance to be used in the verification of (numerically) zero plastic
    /// strain increments
    constexpr double zero_plastic_strain_increment{1.0e-14};

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
                       ///< (and \f$ \mathsymbol{E}^{\text{p}}  = \exp(- \Delta t
                       ///< \dot{\varepsilon}^{\text{p}} \mathsymbol{N}^{\text{p}}) \f$)
      no_flow_resistance,            ///< the material has no flow resistance anymore, such that the
                                     ///< evaluations model non-physical phenomena
      no_plastic_incompressibility,  ///< plastic incompressibility not enforced; the determinant of
                                     ///< the inelastic deformation gradient deviates from 1 beyond
                                     ///< a set tolerance
      failed_solution_linear_system_lnl,  ///< solution of the linear system in the Local
                                          ///< Newton-Raphson Loop failed
      no_convergence_local_newton,  ///< the Local Newton Loop did not converge for the given loop
                                    ///< settings
      singular_jacobian,  ///< singular Jacobian after converged LNL, which does not enable our
                          ///< analytical evaluation of the linearization
      failed_solution_analytic_linearization,  ///< solution of the linear system in the analytical
                                               ///< linearization failed
      failed_matrix_log_evaluation,            ///< failed evaluation of the matrix logarithm or its
                                               ///< derivative
      failed_matrix_exp_evaluation,   ///< failed evaluation of the matrix exponential or its
                                      ///< derivative
      failed_right_cg_interpolation,  ///< failed interpolation of the right Cauchy-Green tensor
      under_yield_surface  ///< mechanical state is "under" the yield surface, i.e., the evaluated
                           ///< stress is smaller than the yield stress
    };


    /// convert error type to detailed error message
    std::string get_detailed_error_message_for_error_type(ErrorType err_type);

    /// enum class for material behavior types
    enum class MatBehavior
    {
      isotrop,         ///< isotropic material behavior
      transv_isotrop,  ///< transversely isotropic material behavior
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

      //! inverse plastic deformation gradient at the last computed time instant (after the last
      //! converged substep)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_substep_plastic_defgrad_inverse;
      //! plastic strain at the last computed time instant (after the last converged substep)
      std::vector<double> last_substep_plastic_strain;
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

    //! struct with local substepping utilities
    struct LocalSubsteppingUtils
    {
      //! current time parameter ranging from 0 to the problem time step \f$ \Delta t \f$
      double t;
      //! counter of evaluated substeps
      unsigned int substep_counter;
      //! current substep size
      double curr_dt;
      //! number of times the problem time step \f$ \Delta t \f$ has been halved
      unsigned int time_step_halving_counter;
      //!  current total number of substeps to be evaluated within the time step \f$ \Delta t
      //! \f$; this is not always given by time_step_halving_counter, since the
      //! halving does not have to be uniform (e.g. we could halve the time step twice and still
      //! have 3 substeps to evaluate instead of 4, i.e. if the first substep was evaluable
      //! numerically, but the second substep not, leading to another halving of the substep
      //! length)
      unsigned int total_num_of_substeps;
      //! current Local Newton iteration index for the substep
      unsigned int iter;

      //! reset routine: basically, create a new empty object
      void reset();
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

      //! deviatoric, symmetric part of the Mandel stress tensor
      Core::LinAlg::Matrix<3, 3> curr_Me_dev_sym_M{Core::LinAlg::Initialization::zero};

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

      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the
      //! inverse inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dMe_dev_sym_diFin{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the
      //! right Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dMe_dev_sym_dC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic strain rate w.r.t. the equivalent stress
      double curr_dpsr_dequiv_stress{0.0};
      //! derivative of the plastic strain rate w.r.t. the equivalent plastic strain
      double curr_dpsr_depsp{0.0};

      //! derivative of the plastic stretching tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_ddpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the equivalent plastic strain (Voigt
      //! stress form)
      Core::LinAlg::Matrix<6, 1> curr_ddpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_ddpdC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic velocity gradient tensor w.r.t. the inverse inelastic
      //! deformation gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dlpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the equivalent plastic strain
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 1> curr_dlpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the right Cauchy-Green
      //! deformation tensor (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dlpdC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic update tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dEpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the equivalent plastic strain (Voigt
      //! notation)
      Core::LinAlg::Matrix<9, 1> curr_dEpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the right Cauchy-Green deformation tensor
      //! (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dEpdC{Core::LinAlg::Initialization::zero};
    };



    /// enum class for state quantity evaluations in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// plastic strain rate,...)
    enum class StateQuantityEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantities method)
      PlasticStrainRateOnly,  ///< return in evaluate_state_quantities once the plastic strain
                              ///< rate has been evaluated
      EquivStressOnly,        ///< return in evaluate_state_quantities once the
                              ///< equivalent stress has been evaluated
    };

    /// enum class for evaluations of the state quantity derivatives in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// derivatives of the plastic strain rate,...)
    enum class StateQuantityDerivEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantity_derivatives
                 ///< method)
      PlasticStrainRateDerivsOnly,  ///< return in evaluate_state_quantity_derivatives once the
                                    ///< derivatives of the plastic strain rate have been
                                    ///< evaluated
      EquivStressDerivsOnly,  ///< return in evaluate_state_quantities once the derivatives of the
                              ///< equivalent stress has been evaluated
    };

  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
