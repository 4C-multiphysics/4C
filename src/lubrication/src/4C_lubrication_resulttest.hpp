// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LUBRICATION_RESULTTEST_HPP
#define FOUR_C_LUBRICATION_RESULTTEST_HPP


#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_result_test.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Lubrication
{
  // forward declaration
  class TimIntImpl;

  /*!
    \brief lubrication specific result test class

  */
  class ResultTest : public Core::Utils::ResultTest
  {
   public:
    /*!
    \brief constructor
    */
    ResultTest(std::shared_ptr<TimIntImpl> lubrication);


    /// our version of nodal value tests
    /*!
      Possible position flags is only "pre"
     */
    void test_node(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

    //! test special quantity not associated with a particular element or node
    void test_special(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

   protected:
    //! get nodal result to be tested
    double result_node(const std::string quantity,  //! name of quantity to be tested
        Core::Nodes::Node* node                     //! node carrying the result to be tested
    ) const;

    //! get special result to be tested
    virtual double result_special(const std::string quantity  //! name of quantity to be tested
    ) const;

   private:
    /// std::shared_ptr to lubrication discretization
    std::shared_ptr<Core::FE::Discretization> dis_;
    /// std::shared_ptr to solution vector
    std::shared_ptr<Core::LinAlg::Vector<double>> mysol_;
    /// number of iterations in last newton iteration
    int mynumiter_;
  };
}  // namespace Lubrication


FOUR_C_NAMESPACE_CLOSE

#endif
