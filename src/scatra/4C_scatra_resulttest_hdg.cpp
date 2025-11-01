// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_resulttest_hdg.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_scatra_timint_hdg.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                           hoermann 09/15 |
 *----------------------------------------------------------------------*/
ScaTra::HDGResultTest::HDGResultTest(const std::shared_ptr<ScaTraTimIntImpl> timint)
    : ScaTraResultTest::ScaTraResultTest(timint),
      scatratiminthdg_(std::dynamic_pointer_cast<const TimIntHDG>(timint))

{
  errors_ = scatratiminthdg_->compute_error();
  return;
}


/*----------------------------------------------------------------------*
 | get nodal result to be tested                         hoermann 09/15 |
 *----------------------------------------------------------------------*/
double ScaTra::HDGResultTest::result_node(
    const std::string quantity,  //! name of quantity to be tested
    Core::Nodes::Node* node      //! node carrying the result to be tested
) const
{
  // initialize variable for result
  double result(0.);

  // extract row map from solution vector
  const Core::LinAlg::Map& phinpmap = scatratiminthdg_->interpolated_phinp()->get_map();

  // test result value of single scalar field (averaged value on element node is tested)
  if (quantity == "phi")
    result =
        scatratiminthdg_->interpolated_phinp()->local_values_as_span()[phinpmap.lid(node->id())];
  else if (quantity == "abs_L2error_phi")
    result = std::sqrt((*errors_)[0]);
  else if (quantity == "rel_L2error_phi")
    result = std::sqrt((*errors_)[0] / (*errors_)[1]);
  else if (quantity == "abs_L2error_gradPhi")
    result = std::sqrt((*errors_)[2]);
  else if (quantity == "rel_L2error_gradPhi")
    result = std::sqrt((*errors_)[2] / (*errors_)[3]);
  // catch unknown quantity strings
  else
    FOUR_C_THROW("Quantity '{}' not supported in result test!", quantity);

  return result;
}  // ScaTra::HDGResultTest::ResultNode

FOUR_C_NAMESPACE_CLOSE
