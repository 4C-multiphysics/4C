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

namespace PoroPressureBased
{
  /// time integration schemes
  enum class TimeIntegrationScheme
  {
    one_step_theta
  };

  //! type of vector norm used for error/residual vectors
  enum class VectorNorm
  {
    undefined,
    l1,         //!< L1/linear norm
    l1_scaled,  //!< L1/linear norm scaled by length of vector
    l2,         //!< L2/Euclidean norm
    rms,        //!< root mean square (RMS) norm
    inf         //!< Maximum/infinity norm
  };

  /// initial field
  enum class InitialField
  {
    zero,
    by_function,
    by_condition
  };

  /// Handling of non-converged nonlinear solver
  enum class DivergenceAction
  {
    stop,            ///< abort simulation
    continue_anyway  ///< continue anyway
  };

  /// set the valid parameters
  void set_valid_parameters_porofluid(std::map<std::string, Core::IO::InputSpec>& list);

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
