// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_EHL_INPUT_HPP
#define FOUR_C_EHL_INPUT_HPP


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
/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
namespace EHL
{
  /// Type of coupling strategy for EHL problems
  enum SolutionSchemeOverFields
  {
    ehl_IterStagg,
    ehl_Monolithic
  };

  /// Type of coupling strategy between the two fields of the EHL problems
  enum FieldCoupling
  {
    coupling_none,
    coupling_matching
  };

  //! type of norm to check for convergence
  enum ConvNorm
  {
    convnorm_abs,  //!< absolute norm
    convnorm_rel,  //!< relative norm of EHL problem with initial EHL rhs
    convnorm_mix   //!< mixed absolute-relative norm
  };

  //! type of norm to check for convergence
  enum BinaryOp
  {
    bop_and,               //!< and
    bop_or,                //!< or
    bop_coupl_or_single,   //!< either EHL problem or single field problems converged
    bop_coupl_and_single,  //!< either EHL problem or single field problems converged
    bop_and_single,        //!< and in single field problems
    bop_or_single          //!< or in single field problems
  };

  //! type of solution techniques
  enum NlnSolTech
  {
    soltech_newtonfull  //!< full Newton-Raphson iteration
  };

  //! Map solution technique enum to std::string
  static inline std::string nln_sol_tech_string(const enum NlnSolTech name  //!< enum to convert
  )
  {
    switch (name)
    {
      case soltech_newtonfull:
        return "fullnewton";
        break;
      default:
        FOUR_C_THROW("Cannot make std::string for solution technique %d", name);
        return "";
    }
  }

  //! type of vector norm used for error/residual vectors
  enum VectorNorm
  {
    norm_vague = 0,  //!< undetermined norm
    norm_l1,         //!< L1/linear norm
    norm_l1_scaled,  //!< L1/linear norm scaled by length of vector
    norm_l2,         //!< L2/Euclidean norm
    norm_rms,        //!< root mean square (RMS) norm
    norm_inf         //!< Maximum/infinity norm
  };


  /// set the ehl parameters
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

  /// set specific ehl conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace EHL


FOUR_C_NAMESPACE_CLOSE

#endif
