/*----------------------------------------------------------------------*/
/*! \file

\brief result tests for HDG problems

\level 3


*/
/*----------------------------------------------------------------------*/
#include "4C_scatra_resulttest_hdg.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_scatra_timint_hdg.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                           hoermann 09/15 |
 *----------------------------------------------------------------------*/
ScaTra::HDGResultTest::HDGResultTest(const Teuchos::RCP<ScaTraTimIntImpl> timint)
    : ScaTraResultTest::ScaTraResultTest(timint),
      scatratiminthdg_(Teuchos::rcp_dynamic_cast<const TimIntHDG>(timint))

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
  const Epetra_BlockMap& phinpmap = scatratiminthdg_->interpolated_phinp()->Map();

  // test result value of single scalar field (averaged value on element node is tested)
  if (quantity == "phi")
    result = (*scatratiminthdg_->interpolated_phinp())[phinpmap.LID(node->id())];
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
    FOUR_C_THROW("Quantity '%s' not supported in result test!", quantity.c_str());

  return result;
}  // ScaTra::HDGResultTest::ResultNode

FOUR_C_NAMESPACE_CLOSE
