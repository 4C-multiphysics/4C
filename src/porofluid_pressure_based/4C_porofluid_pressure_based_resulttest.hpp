// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_RESULTTEST_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_RESULTTEST_HPP



#include "4C_config.hpp"

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

namespace Core::Elements
{
  class Element;
}

namespace PoroPressureBased
{
  // forward declaration
  class TimIntImpl;

  /*!
    \brief POROFLUIDMULTIPHASE specific result test class

  */
  class ResultTest : public Core::Utils::ResultTest
  {
   public:
    /*!
    \brief constructor
    */
    ResultTest(TimIntImpl& porotimint);


    /// our version of nodal value tests
    /*!
      Possible position flags is only "pre"
     */
    void test_node(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

    /// our version of element value tests
    void test_element(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

    //! test special quantity not associated with a particular element or node
    void test_special(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

   protected:
    //! get nodal result to be tested
    double result_node(const std::string quantity,  //! name of quantity to be tested
        Core::Nodes::Node* node                     //! node carrying the result to be tested
    ) const;

    //! get element result to be tested
    double result_element(const std::string quantity,  //! name of quantity to be tested
        const Core::Elements::Element* element         //! element carrying the result to be tested
    ) const;

    //! get special result to be tested
    virtual double result_special(const std::string quantity  //! name of quantity to be tested
    ) const;

   private:
    //! time integrator
    const TimIntImpl& porotimint_;
  };
}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif
