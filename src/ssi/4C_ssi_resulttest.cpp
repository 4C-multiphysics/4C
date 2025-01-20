// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_resulttest.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_ssi_monolithic.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::SSIResultTest::SSIResultTest(const std::shared_ptr<const SSI::SSIBase> ssi_base)
    : Core::Utils::ResultTest("SSI"), ssi_base_(ssi_base)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double SSI::SSIResultTest::result_special(const std::string& quantity) const
{
  // initialize variable for result
  double result(0.0);

  // number of outer coupling iterations (partitioned SSI) or Newton-Raphson iterations (monolithic
  // SSI) in last time step
  if (quantity == "numiterlastnonlinearsolve")
  {
    result = static_cast<double>(ssi_base_->iteration_count());
  }

  // number of iterations performed by linear solver during last Newton-Raphson iteration
  // (monolithic SSI only)
  else if (quantity == "numiterlastlinearsolve")
  {
    result = static_cast<double>(ssi_mono().solver().get_num_iters());
  }

  // test total number of time steps
  else if (!quantity.compare(0, 7, "numstep"))
  {
    result = static_cast<double>(ssi_base_->step());
  }
  // catch unknown quantity strings
  else
  {
    FOUR_C_THROW(
        "Quantity '%s' not supported by result testing functionality for scalar-structure "
        "interaction!",
        quantity.c_str());
  }

  return result;
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
const SSI::SsiMono& SSI::SSIResultTest::ssi_mono() const
{
  const auto* const ssi_mono = dynamic_cast<const SSI::SsiMono* const>(ssi_base_.get());
  if (ssi_mono == nullptr)
    FOUR_C_THROW("Couldn't access time integrator for monolithic scalar-structure interaction!");
  return *ssi_mono;
}

/*-------------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------------*/
void SSI::SSIResultTest::test_special(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // make sure that quantity is tested only by one processor
  if (Core::Communication::my_mpi_rank(ssi_base_->get_comm()) == 0)
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

FOUR_C_NAMESPACE_CLOSE
