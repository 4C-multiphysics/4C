// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_RESULTTEST_HPP
#define FOUR_C_SCATRA_RESULTTEST_HPP

#include "4C_config.hpp"

#include "4C_utils_result_test.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace ScaTra
{
  // forward declaration
  class ScaTraTimIntImpl;

  /*!
    \brief scalar-transport specific result test class

    \author gjb
    \date 07/08
  */
  class ScaTraResultTest : public Core::Utils::ResultTest
  {
   public:
    /*!
    \brief constructor
    */
    ScaTraResultTest(Teuchos::RCP<ScaTraTimIntImpl> scatratimint);


    /// our version of nodal value tests
    /*!
      Possible position flags is only "phi"
     */
    void test_node(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

    //! test special quantity not associated with a particular element or node
    void test_special(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

   protected:
    //! get nodal result to be tested
    virtual double result_node(const std::string quantity,  //! name of quantity to be tested
        Core::Nodes::Node* node  //! node carrying the result to be tested
    ) const;

    //! get special result to be tested
    virtual double result_special(const std::string quantity  //! name of quantity to be tested
    ) const;

    //! time integrator
    const Teuchos::RCP<const ScaTraTimIntImpl> scatratimint_;

   private:
  };
}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
