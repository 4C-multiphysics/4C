// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_INPUT_HPP
#define FOUR_C_THERMO_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  class ConditionDefinition;
}

namespace Thermo
{
  //! @name Time integration
  //@{

  //! Type of time integrator including statics
  enum class DynamicType
  {
    Undefined,     //!< undefined integrator (sth like a default)
    Statics,       //!< static analysis
    OneStepTheta,  //!< one-step-theta time integrator (implicit)
    GenAlpha,      //!< generalised-alpha time integrator (implicit)
  };  // DynamicType()

  //! initial field for scalar transport problem
  enum InitialField
  {
    initfield_zero_field,
    initfield_field_by_function,
    initfield_field_by_condition
  };

  //! Mid-average type of internal forces for generalised-alpha-like time integration schemes
  enum MidAverageEnum
  {
    midavg_vague = 0,  //!< undefined mid-averaging type
    midavg_imrlike,    //!< alphaf-mid-averaging is done IMR-like, i.e.
                       //!< \f$F_{int,m}\f$
                       //!< \f$= F_{int}(D_m)\f$
                       //!< \f$= F_{int}(\alpha_f . D_{n+1} + (1-\alpha_f) . D_n)\f$
                       //!< (IMR means implicit mid-point rule.)
    midavg_trlike      //!< alphaf-mid-averaging is done TR-like, i.e.
                       //!< \f$F_{int,m}\f$
                       //!< \f$ = \alpha_f . F_{int,n+1} + (1-\alpha_f) . F_{int,n}\f$
                       //!< \f$ = \alpha_f . F_{int}(\alpha_f . D_{n+1}) + (1-\alpha_f) .
                       //!< F_{int}(D_n)\f$
                       //!<  (TR means trapezoidal rule.)
  };  // MidAverageEnum()

  //@}

  //! @name Solution technique and related
  //@{

  //! type of solution techniques
  enum NonlinSolTech
  {
    soltech_vague,      //!< undefined
    soltech_newtonfull  //!< full Newton-Raphson iteration
  };

  /// type of solution techniques
  enum DivContAct
  {
    divcont_stop,               ///< abort simulation
    divcont_continue,           ///< continue nevertheless
    divcont_repeat_step,        ///< repeat time step
    divcont_halve_step,         ///< halve time step and carry on with simulation
    divcont_repeat_simulation,  ///< repeat the whole simulation
  };

  /// convergence of nonlinear solver
  enum ConvergenceStatus
  {
    conv_success = 0,      ///< converged successfully
    conv_nonlin_fail = 1,  ///< nonlinear solution procedure failed
    conv_lin_fail = 2,     ///< linear system failed
    conv_ele_fail = 3,     ///< failure in element in form of negative Jac. det.
    conv_fail_repeat = 4   ///< nonlinear solver failed, repeat step according to divercont action
                           ///< set in input file
  };


  //! Type of predictor
  enum PredEnum
  {
    pred_vague,          //!< undetermined
    pred_consttemp,      //!< constant temperatures
    pred_consttemprate,  //!< constant temperatures and rates
    pred_tangtemp        //!< linearised solution obeying DBC temperature via tangent
                   //!< T_{n+1}^{<0>} = T_{n} + Ktang_{n,eff}^{-1} . (- Ktang_{n} . (T_{n+1}^{DBC}
                   //!< - T_{n})) This looks hilarious, but remember Ktan_{n,eff}^{-1} is not the
                   //!< inverse of Ktan_{n} due to the application of the Dirichlet BCs (i.e. the
                   //!< reduction to the test space).
  };

  //! type of norm to check for convergence
  enum ConvNorm
  {
    convnorm_abs,  //!< absolute norm
    convnorm_rel,  //!< relative norm
    convnorm_mix   //!< mixed absolute-relative norm
  };

  //! type of norm to check for convergence
  enum BinaryOp
  {
    bop_or,  //!<  or
    bop_and  //!<  and
  };

  //@}

  //! @name Output
  //@{

  //! Type of thermal flux output
  //! (this enum represents the input file parameter HEATFLUX) CHECK IT!
  enum class HeatFluxType
  {
    None,     //!< no heatflux output
    Current,  //!< output of heatflux in current configuration
    Initial   //!< output of heat flux in initial configuration
  };

  //! Type of thermal gradient output
  //! (this enum represents the input file parameter TEMPGRAD) CHECK IT!
  enum class TempGradType
  {
    None,     //!< no thermal gradient output
    Current,  //!< output of thermal gradient in current configuration
    Initial   //!< output of thermal gradient in initial configuration
  };

  //@}

  //! @name General
  //@{

  //! type of vector norm used for error/residual vectors
  enum VectorNorm
  {
    norm_vague = 0,  //!< undetermined norm
    norm_l1,         //!< L1/linear norm
    norm_l2,         //!< L2/Euclidean norm
    norm_rms,        //!< root mean square (RMS) norm
    norm_inf         //!< Maximum/infinity norm
  };

  //@}

  //! error calculation
  enum CalcError
  {
    no_error_calculation,
    calcerror_byfunct
  };

  /// thermo parameters
  std::vector<Core::IO::InputSpec> valid_parameters();

  /// set thermo specific conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace Thermo

FOUR_C_NAMESPACE_CLOSE

#endif
