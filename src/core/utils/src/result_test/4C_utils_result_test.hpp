// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_RESULT_TEST_HPP
#define FOUR_C_UTILS_RESULT_TEST_HPP


#include "4C_config.hpp"

#include <mpi.h>

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputParameterContainer;
  class InputFile;
}  // namespace Core::IO

namespace Core::Utils
{
  /*!
    \brief Base class of all field test classes

    The idea is to have a subclass of this for every algorithm class
    that needs result testing. The Match method needs to be defined to
    state if a particular result value belongs to that field and needs
    to be checked here. And then there are testing methods for element
    tests, nodal tests and special cases (like beltrami fluid
    flow). These methods provide dummy (FOUR_C_THROW) implementations and
    have to be redefined in subclasses to actually do the testing.
  */
  class ResultTest
  {
   public:
    /// not yet documented
    explicit ResultTest(std::string name = "NONE");

    /**
     * Virtual destructor.
     */
    virtual ~ResultTest() = default;

    /// perform element value test
    virtual void test_element(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count);

    /*!
     * @brief  perform nodal value test
     *
     * @param[in] container   container containing result test specification
     * @param[in] nerr        number of failed result tests
     * @param[in] test_count  number of result tests
     */
    virtual void test_node(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count);

    /*!
     * @brief  perform nodal value test on a geometry. The operation can be e.g., sum, max and min
     *
     * @param[in] container   container containing result test specification
     * @param[in] nerr        number of failed result tests
     * @param[in] test_count  number of result tests
     */
    virtual void test_node_on_geometry(const Core::IO::InputParameterContainer& container,
        int& nerr, int& test_count, const std::vector<std::vector<std::vector<int>>>& nodeset);

    /// perform special case test
    virtual void test_special(const Core::IO::InputParameterContainer& container, int& nerr,
        int& test_count, int& unevaluated_test_count);

    /*!
     * @brief  perform special case test
     *
     * @param[in] container   container containing result test specification
     * @param[in] nerr        number of failed result tests
     * @param[in] test_count  number of result tests
     */
    virtual void test_special(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count);

    /// tell whether this field test matches to a given line
    virtual bool match(const Core::IO::InputParameterContainer& container);

    /**
     * The name of the field test. This name matches the top-level group of the input for a result
     * test.
     */
    [[nodiscard]] const std::string& name() const { return myname_; }

   protected:
    //! compare a calculated @param actresult with the expected one stored in the @param container
    //!
    //! There is a difference between node/element based results and special results.
    //! Node/element based results have to be compared at a specific node/element.
    //! Special results are not attached to a specific node/element, but to the
    //! overall algorithm.
    virtual int compare_values(
        double actresult, std::string type, const Core::IO::InputParameterContainer& container);

   private:
    /// specific name of a field test
    const std::string myname_;
  };

  /*!
    \brief Manager class of result test framework

    You have to create one object of this class to test the results of
    your calculation. For each field involved you will want to add a
    specific field test class (derived from ResultTest). Afterwards
    just start testing...
  */
  class ResultTestManager
  {
   public:
    /// add field specific result test object
    void add_field_test(std::shared_ptr<ResultTest> test);

    /// do all tests of all fields including appropriate output
    void test_all(MPI_Comm comm);

    /// Store the parsed @p results.
    void set_parsed_lines(std::vector<Core::IO::InputParameterContainer> results);

    /// Store the node set
    void set_node_set(const std::vector<std::vector<std::vector<int>>>& nodeset);

    /// Get the node set (design topology)
    [[nodiscard]] const std::vector<std::vector<std::vector<int>>>& get_node_set() const
    {
      return nodeset_;
    }

   private:
    /// set of field specific result test objects
    std::vector<std::shared_ptr<ResultTest>> fieldtest_;

    /// expected results
    std::vector<Core::IO::InputParameterContainer> results_;

    /// node set for values extraction
    std::vector<std::vector<std::vector<int>>> nodeset_;
  };

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
