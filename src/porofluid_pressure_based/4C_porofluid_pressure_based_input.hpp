// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_INPUT_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

// forward declaration
/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/

namespace POROFLUIDMULTIPHASE
{
  /// time integration schemes
  enum TimeIntegrationScheme
  {
    timeint_one_step_theta
  };

  /// compute error compared to analytical solution
  enum CalcError
  {
    calcerror_no,
    calcerror_byfunction
  };

  //! type of norm to check for convergence
  enum ConvNorm
  {
    convnorm_abs,  //!< absolute norm
    convnorm_rel,  //!< relative norm
    convnorm_mix   //!< mixed absolute-relative norm
  };

  //! type of vector norm used for error/residual vectors
  enum VectorNorm
  {
    norm_undefined,
    norm_l1,         //!< L1/linear norm
    norm_l1_scaled,  //!< L1/linear norm scaled by length of vector
    norm_l2,         //!< L2/Euclidean norm
    norm_rms,        //!< root mean square (RMS) norm
    norm_inf         //!< Maximum/infinity norm
  };

  /// type of finite difference check
  enum FdCheck
  {
    fdcheck_none,
    fdcheck_global
  };

  /// initial field for scalar transport problem
  enum InitialField
  {
    initfield_zero_field,
    initfield_field_by_function,
    initfield_field_by_condition
  };

  /// Handling of non-converged nonlinear solver
  enum DivContAct
  {
    divcont_stop,     ///< abort simulation
    divcont_continue  ///< continue nevertheless
  };

  //! reconstruction type of gradients (e.g. velocity gradient)
  enum FluxReconstructionMethod
  {
    gradreco_none,
    // gradreco_spr, super-convergent patch recovery not activated yet
    gradreco_l2
  };

  /// set the lubrication parameters
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

}  // namespace POROFLUIDMULTIPHASE



FOUR_C_NAMESPACE_CLOSE

#endif
