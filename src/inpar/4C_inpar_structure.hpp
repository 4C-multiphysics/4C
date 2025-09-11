// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_STRUCTURE_HPP
#define FOUR_C_INPAR_STRUCTURE_HPP


#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}
/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace Solid
  {
    /// Active element technologies
    enum class EleTech
    {
      eas,       ///< enhanced assumed strain
      fbar,      ///< F-bar
      pressure,  ///< additional pressure degree of freedom
      rotvec,    ///< non-additive rotation (pseudo-)vector DOFs
      ps_mulf    ///< prestressing: mulf
    };

    /// Type of the used models/conditions
    /// necessary for the model evaluator
    enum ModelType : int
    {
      model_vague = -1,            ///< undefined model type
      model_structure = 0,         ///< evaluate the structural model
      model_contact = 1,           ///< evaluate the contact model
      model_meshtying = 2,         ///< evaluate the meshtying model
      model_cardiovascular0d = 3,  ///< evaluate the cardiovascular0d model
      model_springdashpot = 4,     ///< evaluate the springdashpot model
      model_lag_pen_constraint =
          5,  ///< evaluate the Lagrange or/and penalty enforced constraint models
      model_monolithic_coupling = 6,   ///< evaluate any monolithic coupling contributions
      model_partitioned_coupling = 7,  ///< evaluate any partitioned coupling contributions
      model_beam_interaction_old =
          8,  ///< evaluate the model for contact/potential/... interactions of beams
      model_browniandyn = 9,       ///< evaluate the brownian dynamics model
      model_beaminteraction = 10,  ///< evaluate beaminteraction model
      model_basic_coupling = 11,   ///< evaluate coupling contributions that are independent of
                                   ///< monolithic or partitioned coupling
      model_constraints = 12,      ///< evaluate the contributions of the constraint framework
      model_multiscale = 13        ///< consider multi scale simulations
    };

    /// @name Time integration
    //!@{

    /*! \brief Time integration strategy
     *
     *  \warning Implement new features only in the \c int_standard framework
     *  as the old framework is supposed to go away.
     */
    enum IntegrationStrategy
    {
      int_old, /**< old structure (FixMe deprecated and should be deleted, as soon as the clean-up
                * is finished!) */
      int_standard,  ///< standard time integration (implicit or explicit)
      int_local      ///< path following with LOCAL
    };

    /// Type of time integrator including statics
    enum class DynamicType : int
    {
      Statics,           ///< static analysis
      GenAlpha,          ///< generalised-alpha time integrator (implicit)
      GenAlphaLieGroup,  ///< generalised-alpha time integrator for Lie groups (e.g. SO3 group
                         ///< of rotation matrices) (implicit)
      OneStepTheta,      ///< one-step-theta time integrator (implicit)
      ExplEuler,         ///< forward Euler (explicit)
      CentrDiff,         ///< central differences (explicit)
      AdamsBashforth2,   ///< Adams-Bashforth 2nd order (explicit)
      AdamsBashforth4    ///< Adams-Bashforth 4th order (explicit)
    };

    /// Type of (global) damping
    enum DampKind
    {
      damp_none = 0,  ///< damping off
      damp_rayleigh,  ///< globally applied Rayleigh damping
      damp_material,  ///< element-wise applied damping using element velocities
    };

    /*! \brief Mid-average type of internal forces for generalised-alpha-like time integration
     * schemes
     *

     */
    enum MidAverageEnum
    {
      midavg_vague = 0,  ///< undefined mid-averaging type
      midavg_imrlike,    ///< alphaf-mid-averaging is done IMR-like, i.e.
                         ///< \f$F_{int,m}\f$
                         ///< \f$= F_{int}(D_m)\f$
                         ///< \f$= F_{int}( (1-\alpha_f)*D_{n+1} + \alpha_f*D_n )\f$
                         ///< (IMR means implicit mid-point rule.)
      midavg_trlike      ///< alphaf-mid-averaging is done TR-like, i.e.
                         ///< \f$F_{int,m}\f$
                         ///< \f$= (1-\alpha_f)*F_{int,n+1} + \alpha_f*F_{int,n}\f$
                         ///<  \f$= (1-\alpha_f)*F_{int}(D_{n+1}) + \alpha_f*F_{int}(D_n)\f$
                         ///<  (TR means trapezoidal rule.)
    };


    /// Type of predictor
    enum PredEnum : int
    {
      pred_vague,     ///< undetermined
      pred_constdis,  ///< constant displacements, consistent velocities and accelerations
      pred_constvel,  ///< constant velocity, extrapolated displacements, consistent accelerations
      pred_constacc,  ///< constant acceleration, extrapolated displacements and velocities
      pred_constdisvelacc,  ///< constant displacements,
                            ///< velocities and accelerations
      pred_tangdis,         ///< linearised solution obeying DBC displacements via tangent
                     ///< D_{n+1}^{<0>} = D_{n} + Ktang_{n,eff}^{-1} . (- Ktang_{n} . (D_{n+1}^{DBC}
                     ///< - D_{n})) This looks hilarious, but remember Ktan_{n,eff}^{-1} is not the
                     ///< inverse of Ktan_{n} due to the application of the Dirichlet BCs (i.e. the
                     ///< reduction to the test space).
      pred_tangdis_constfext,  ///< tangential displacement predictor with respect
                               ///< to a constant Neumann force during the predictor
                               ///< step
      pred_constdispres,       ///< constant displacements and pressure
      pred_constdisvelaccpres  ///< constant displacements,
                               ///< velocities and accelerations
                               ///< and pressures
    };

    /// Map predictor enum term to std::string
    std::string pred_enum_string(const PredEnum name);

    //!@}

    /// @name Solution technique and related
    //!@{

    /// have inertia forces to be linearized?
    enum class MassLin
    {
      ml_none,      ///< constant mass matrix
      ml_rotations  ///< nonlinear inertia terms, rotational DoFs
    };

    /// type of solution techniques
    enum NonlinSolTech
    {
      soltech_vague,              ///< undefined
      soltech_newtonfull,         ///< full Newton-Raphson iteration
      soltech_newtonls,           ///< line search Newton-Raphson
      soltech_newtonmod,          ///< modified Newton-Raphson iteration
      soltech_newtonuzawalin,     ///< linear Uzawa iteration for
                                  ///< constraint system
      soltech_newtonuzawanonlin,  ///< non-linear Uzawa iteration
                                  ///< for constraint system
      soltech_ptc,                ///< pseudo transient continuation nonlinear iteration
      soltech_nox_nln,            ///< non-linear solution with NOX (new)
      soltech_singlestep          ///< single step for explicit dynamics
    };


    /// Handling of non-converged nonlinear solver
    enum DivContAct
    {
      divcont_stop,             ///< abort simulation
      divcont_continue,         ///< continue nevertheless
      divcont_repeat_step,      ///< repeat time step
      divcont_halve_step,       ///< halve time step and carry on with simulation
      divcont_adapt_step,       ///< adapt (halve or double) time step and carry on with simulation
      divcont_rand_adapt_step,  ///< adapt randomly time step and carry on with simulation
      divcont_rand_adapt_step_ele_err,  ///< adapt randomly time step and carry on with simulation,
                                        ///< including acceptance of element errors in form of
                                        ///< negative Jacobian determinant
      divcont_repeat_simulation,        ///< repeat the whole simulation
      divcont_adapt_penaltycontact,  ///< slightly adapt the penalty contact parameter if timestep
                                     ///< doesn't converge
      divcont_adapt_3D0Dptc_ele_err  ///< adaptive pseudo-transient continuation of structural part
                                     ///< of 3D0D-coupled problem
    };

    /// Handling of non-converged nonlinear solver
    enum ConvergenceStatus
    {
      conv_success = 0,      ///< converged successfully
      conv_nonlin_fail = 1,  ///< nonlinear solution procedure failed
      conv_lin_fail = 2,     ///< linear system failed
      conv_ele_fail = 3,     ///< failure in element in form of negative Jac. det.
      conv_fail_repeat = 4   ///< nonlinear solver failed, repeat step according to divercont action
                             ///< set in input file
    };

    /// type of norm to check for convergence
    enum ConvNorm
    {
      convnorm_abs,  ///< absolute norm
      convnorm_rel,  ///< relative norm
      convnorm_mix   ///< mixed absolute-relative norm
    };

    /// type of norm to check for convergence
    enum BinaryOp
    {
      bop_or,  ///<  or
      bop_and  ///<  and
    };

    //!@}

    /// type of prestressing algorithm
    enum class PreStress : char
    {
      none,  ///<  none
      mulf,  ///<  Modified Updated Lagrangian Formulation (cf. Gee et al. (2010)). Be careful, this
             ///<  prestress algorithm does not ensure equilibrium
      material_iterative  ///< Iterative prestress algorithm on the material level
    };

    /// STC scaling for thin shell structures
    enum StcScale
    {
      stc_inactive = 0,  ///< no scaling
      stc_curr,          ///< Non-symmetric STC
      stc_currsym        ///< Symmetric STC
    };

    /// @name Constraints (global, geometric)
    //!@{

    /// possible constraint solvers
    enum ConSolveAlgo
    {
      consolve_direct,  ///< build monolithic system for Lagrange multipliers
      consolve_uzawa,   ///< solve linear system iteratively (partitioned)
      consolve_simple   ///< use simple preconditioner for iterative solve
    };

    //!@}

    /// @name Output
    //!@{

    enum class ConditionNumber : char
    {
      none,             ///< no condition number
      gmres_estimate,   ///< use the GMRES method to compute an condition number estimate
      max_min_ev_ratio, /**< compute the condition number based on the maximal and minimal
                         *   eigenvalue */
      one_norm,         ///< Use the 1-norm to compute the condition number
      inf_norm          ///< Use the inf-norm to compute the condition number
    };

    /// Type of structural stress output
    /// (this enum represents the input file parameter STRUCT_STRESS)
    enum StressType
    {
      stress_none,    ///< no stress output
      stress_cauchy,  ///< output of Cauchy stresses
      stress_2pk      ///< output of 2nd Piola-Kirchhoff stresses
    };

    /// Type of structural strain output
    /// (this enum represents the input file parameter STRUCT_STRAIN and
    /// STRUCT_PLASTIC_STRAIN)
    enum StrainType
    {
      strain_none,  ///< no strain output
      strain_ea,    ///< output of Euler-Almansi strains
      strain_gl,    ///< output of Green-Lagrange strains
      strain_log    ///< output of Logarithmic (or Hencky) strains
    };

    /// Type of structural optional quantity output
    /// (this enum represents the input file parameter OPTIONAL_QUANTITY)
    enum OptQuantityType
    {
      optquantity_none,              ///< no optional output
      optquantity_membranethickness  ///< output of thickness of membrane finite elements
    };

    /*!
     * \brief Type of output of gauss point data
     */
    enum class GaussPointDataOutputType : char
    {
      none,            ///< No output of Gauss point data
      element_center,  ///< Output averaged within the element
      nodes,           ///< Output projected onto nodes
      gauss_points     ///< Raw output at the Gauss points
    };

    //!@}

    /// @name Time adaptivity
    //!@{

    /// type of adaptivity in time
    enum TimAdaKind
    {
      timada_kind_none,            ///< no time adaptivity
      timada_kind_zienxie,         ///< Zienkiewicz-Xie indicator
      timada_kind_joint_explicit,  ///< Auxiliary explitcit time integration indicator
      timada_kind_ab2,          ///< Adams-Bahsforth2 indicator // to be removed with the old time
                                ///< integration
      timada_kind_ab4,          ///< Adams-Bahsforth4 indicator // to be removed with the old time
                                ///< integration
      timada_kind_expleuler,    ///< Explicit Euler indicator // to be removed with the old time
                                ///< integration
      timada_kind_centraldiff,  ///< Central difference indicator // to be removed with the old time
                                ///< integration
    };

    //!@}

    /// @name General
    //!@{

    /// type of vector norm used for error/residual vectors
    enum VectorNorm
    {
      norm_vague = 0,  ///< undetermined norm
      norm_l1,         ///< L1/linear norm: \f$\vert x\vert_1 = \sum_{i=1}^N |x_i|\f$
      norm_l2,         ///< L2/Euclidean norm: \f$\vert x\vert_2 = \sum_{i=1}^N x_i^2\f$
      norm_rms,        ///< root mean square (RMS) norm: \f$\vert x\vert_{rms} = \left(\sum_{i=1}^N
                       ///< x_i^2\right) / \sqrt{N}\f$
      norm_inf         ///< Maximum/infinity norm: \f$\vert x\vert_\infty = \max{x_i}\f$
    };

    /// initial displacement field
    enum InitialDisp
    {
      initdisp_zero_disp,        ///< zero initial displacement
      initdisp_disp_by_function  ///< initial displacement from function
    };

    /// calculate error
    enum CalcError
    {
      no_error_calculation,  ///< no error calculation
      byfunct                ///< evaluate error from function
    };


    /// kinematic description
    enum class KinemType
    {
      vague,           ///< undetermined kinematics
      linear,          ///< linear kinematics
      nonlinearTotLag  ///< nonlinear kinematics Total Lagrange
    };

    std::string kinem_type_string(const KinemType kinem_type);
    //!@}

    /// structure parameters
    std::vector<typename Core::IO::InputSpec> valid_parameters();

    /// set structure-specific conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

  }  // namespace Solid

}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
