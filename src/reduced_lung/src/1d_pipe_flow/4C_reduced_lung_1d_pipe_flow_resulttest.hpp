// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_RESULTTEST_HPP
#define FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_RESULTTEST_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_result_test.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace ReducedLung1dPipeFlow
{
  // forward declaration
  class ResultTest : public Core::Utils::ResultTest
  {
   public:
    ResultTest(std::shared_ptr<Core::FE::Discretization> dis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> sol);

    void test_node(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

   private:
    std::shared_ptr<Core::FE::Discretization> dis_;
    std::shared_ptr<const Core::LinAlg::Vector<double>> sol_;
  };

}  // namespace ReducedLung1dPipeFlow

FOUR_C_NAMESPACE_CLOSE

#endif  // 4C_REDUCED_LUNG_1D_PIPE_FLOW_RESULTTEST_HPP
