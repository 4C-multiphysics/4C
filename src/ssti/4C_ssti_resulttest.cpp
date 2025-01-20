// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssti_resulttest.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_ssti_algorithm.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSTI::SSTIResultTest::SSTIResultTest(const SSTI::SSTIAlgorithm& ssti_algorithm)
    : Core::Utils::ResultTest("SSTI"), ssti_algorithm_(ssti_algorithm)
{
}

/*-------------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------------*/
void SSTI::SSTIResultTest::test_special(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // make sure that quantity is tested only by one processor
  if (Core::Communication::my_mpi_rank(ssti_algorithm_.get_comm()) == 0)
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
}

/*-------------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------------*/
double SSTI::SSTIResultTest::result_special(const std::string& quantity) const
{
  double result(0.0);

  // number of Newton-Raphson iterations (monolithic SSTI) in last time step
  if (quantity == "numiterlastnonlinearsolve")
  {
    result = static_cast<double>(ssti_algorithm_.iter());
  }

  // test total number of time steps
  else if (!quantity.compare(0, 7, "numstep"))
  {
    result = static_cast<double>(ssti_algorithm_.step());
  }
  // catch unknown quantity strings
  else
  {
    FOUR_C_THROW(
        "Quantity '%s' not supported by result testing functionality for scalar-structure-thermo "
        "interaction!",
        quantity.c_str());
  }

  return result;
}
FOUR_C_NAMESPACE_CLOSE
