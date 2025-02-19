// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_INPUT_HPP
#define FOUR_C_THERMO_INPUT_HPP

#include "4C_config.hpp"

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
  enum DynamicType
  {
    dyna_undefined,     //!< undefined integrator (sth like a default)
    dyna_statics,       //!< static analysis
    dyna_onesteptheta,  //!< one-step-theta time integrator (implicit)
    dyna_genalpha,      //!< generalised-alpha time integrator (implicit)
  };  // DynamicType()

  //! Map time integrator to std::string
  static inline std::string dynamic_type_string(const enum DynamicType name)
  {
    switch (name)
    {
      case dyna_undefined:
        return "Undefined";
        break;
      case dyna_statics:
        return "Statics";
        break;
      case dyna_onesteptheta:
        return "OneStepTheta";
        break;
      case dyna_genalpha:
        return "GenAlpha";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string for time integrator %d", name);
        return "";
    }
  }  // DynamicTypeString()

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

  /// Map mid-averaging to std::string
  static inline std::string mid_average_string(const enum MidAverageEnum name)
  {
    switch (name)
    {
      case midavg_vague:
        return "Vague";
        break;
      case midavg_imrlike:
        return "IMR-like";
      case midavg_trlike:
        return "TR-like";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string for time integrator %d", name);
        return "";
    }
  }

  //@}

  //! @name Solution technique and related
  //@{

  //! type of solution techniques
  enum NonlinSolTech
  {
    soltech_vague,      //!< undefined
    soltech_newtonfull  //!< full Newton-Raphson iteration
  };

  //! Map solution technique enum to std::string
  static inline std::string nonlin_sol_tech_string(const enum NonlinSolTech name)
  {
    switch (name)
    {
      case soltech_vague:
        return "vague";
        break;
      case soltech_newtonfull:
        return "fullnewton";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string for solution technique %d", name);
        return "";
    }
  }

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


  /// Map  enum to string
  static inline std::string div_cont_act_string(const enum DivContAct name)
  {
    switch (name)
    {
      case divcont_stop:
        return "stop";
        break;
      case divcont_continue:
        return "continue";
        break;
      case divcont_repeat_step:
        return "repeat_step";
        break;
      case divcont_halve_step:
        return "halve_step";
        break;
      case divcont_repeat_simulation:
        return "repeat_simulation";
        break;
      default:
        FOUR_C_THROW("Cannot make string for solution div cont technique %d", name);
        return "";
    }
  }

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

  //! Map predictor enum term to std::string
  static inline std::string pred_enum_string(const PredEnum name)
  {
    switch (name)
    {
      case pred_vague:
        return "Vague";
        break;
      case pred_consttemp:
        return "ConstTemp";
        break;
      case pred_consttemprate:
        return "ConstTempRate";
        break;
      case pred_tangtemp:
        return "TangTemp";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string for predictor %d", name);
        return "";
    }
  }

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
  //! (this enum represents the input file parameter THERM_HEATFLUX) CHECK IT!
  enum HeatFluxType
  {
    heatflux_none,     //!< no heatflux output
    heatflux_current,  //!< output of heatflux in current configuration
    heatflux_initial   //!< output of heat flux in initial configuration
  };

  //! Map predictor enum term to std::string
  static inline std::string heat_flux_string(const HeatFluxType& name)
  {
    switch (name)
    {
      case heatflux_none:
        return "none";
        break;
      case heatflux_current:
        return "heatflux_current";
        break;
      case heatflux_initial:
        return "heatflux_initial";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string for predictor %d", name);
        return "";
    }
  }

  //! Type of thermal gradient output
  //! (this enum represents the input file parameter THERM_TEMPGRAD) CHECK IT!
  enum TempGradType
  {
    tempgrad_none,     //!< no thermal gradient output
    tempgrad_current,  //!< output of thermal gradient in current configuration
    tempgrad_initial   //!< output of thermal gradient in initial configuration
  };

  //! Map predictor enum term to std::string
  static inline std::string temp_grad_string(const TempGradType& name)
  {
    switch (name)
    {
      case tempgrad_none:
        return "none";
        break;
      case tempgrad_current:
        return "tempgrad_current";
        break;
      case tempgrad_initial:
        return "tempgrad_initial";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string for predictor %d", name);
        return "";
    }
  }

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

  //! map enum term to std::string
  static inline std::string vector_norm_string(const enum VectorNorm norm)
  {
    switch (norm)
    {
      case norm_vague:
        return "Vague";
        break;
      case norm_l1:
        return "L1";
        break;
      case norm_l2:
        return "L2";
        break;
      case norm_rms:
        return "Rms";
        break;
      case norm_inf:
        return "Inf";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string to vector norm %d", norm);
        return "";
    }
  }

  //@}

  //! error calculation
  enum CalcError
  {
    no_error_calculation,
    calcerror_byfunct
  };

  /// set the thermo parameters
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

  /// set thermo specific conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace Thermo

FOUR_C_NAMESPACE_CLOSE

#endif
