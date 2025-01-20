// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_sti_resulttest.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_sti_monolithic.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                               fang 01/17 |
 *----------------------------------------------------------------------*/
STI::STIResultTest::STIResultTest(const std::shared_ptr<STI::Algorithm>&
        sti_algorithm  //!< time integrator for scatra-thermo interaction
    )
    // call base class constructor
    : Core::Utils::ResultTest("STI"),

      // store pointer to time integrator for scatra-thermo interaction
      sti_algorithm_(sti_algorithm)
{
  return;
}


/*-------------------------------------------------------------------------------------*
 | test special quantity not associated with a particular element or node   fang 01/17 |
 *-------------------------------------------------------------------------------------*/
void STI::STIResultTest::test_special(
    const Core::IO::InputParameterContainer&
        container,   ///< container with expected results as specified in the input file
    int& nerr,       //!< number of failed result tests
    int& test_count  ///< number of result tests
)
{
  // make sure that quantity is tested only by one processor
  if (Core::Communication::my_mpi_rank(sti_algorithm_->get_comm()) == 0)
  {
    // extract name of quantity to be tested
    std::string quantity = container.get<std::string>("QUANTITY");

    // get result to be tested
    const double result = result_special(quantity);

    // compare values
    const int err = compare_values(result, "SPECIAL", container);
    nerr += err;
    ++test_count;
  }
}  // STI::STIResultTest::test_special


/*----------------------------------------------------------------------*
 | get special result to be tested                           fang 01/17 |
 *----------------------------------------------------------------------*/
double STI::STIResultTest::result_special(
    const std::string& quantity  //! name of quantity to be tested
) const
{
  // initialize variable for result
  double result(0.);

  // number of Newton-Raphson iterations in last time step
  if (quantity == "numiterlastnonlinearsolve") result = (double)sti_algorithm_->iter();

  // number of iterations performed by linear solver during last Newton-Raphson iteration
  else if (quantity == "numiterlastlinearsolve")
    result = (double)sti_monolithic().solver().get_num_iters();

  // catch unknown quantity strings
  else
    FOUR_C_THROW(
        "Quantity '%s' not supported by result testing functionality for scatra-thermo "
        "interaction!",
        quantity.c_str());

  return result;
}  // STI::STIResultTest::result_special


/*------------------------------------------------------------------------------*
 | return time integrator for monolithic scatra-thermo interaction   fang 09/17 |
 *------------------------------------------------------------------------------*/
const STI::Monolithic& STI::STIResultTest::sti_monolithic() const
{
  const STI::Monolithic* const sti_monolithic =
      dynamic_cast<const STI::Monolithic* const>(sti_algorithm_.get());
  if (sti_monolithic == nullptr)
    FOUR_C_THROW("Couldn't access time integrator for monolithic scatra-thermo interaction!");
  return *sti_monolithic;
}

FOUR_C_NAMESPACE_CLOSE
