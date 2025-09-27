// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_result_test.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_pstream.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN


Core::Utils::ResultTest::ResultTest(std::string name) : myname_(std::move(name)) {}

void Core::Utils::ResultTest::test_element(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  FOUR_C_THROW("no element test available");
}

void Core::Utils::ResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  FOUR_C_THROW("no node test available");
}

void Core::Utils::ResultTest::test_node_on_geometry(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count,
    const std::vector<std::vector<std::vector<int>>>& nodeset)
{
  FOUR_C_THROW("no geometry test available");
}

void Core::Utils::ResultTest::test_special(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  FOUR_C_THROW("no special case test available");
}

void Core::Utils::ResultTest::test_special(const Core::IO::InputParameterContainer& container,
    int& nerr, int& test_count, int& unevaluated_test_count)
{
  test_special(container, nerr, test_count);
}


int Core::Utils::ResultTest::compare_values(
    double actresult, std::string type, const Core::IO::InputParameterContainer& container)
{
  std::string quantity = container.get<std::string>("QUANTITY");
  double givenresult = container.get<double>("VALUE");
  double tolerance = container.get<double>("TOLERANCE");
  // safety check
  if (tolerance <= 0.) FOUR_C_THROW("Tolerance for result test must be strictly positive!");
  auto name = container.get<std::optional<std::string>>("NAME");

  // return value (0 if results are correct, 1 if results are not correct)
  int ret = 0;

  // prepare std::string stream 'msghead' containing general information on the current test
  std::stringstream msghead;
  msghead << std::left << std::setw(9) << myname_ << ": ";

  if (type != "SPECIAL")
  {
    msghead << std::left << std::setw(16) << container.get_or<std::string>("DIS", "");
  }

  msghead << std::left << std::setw(8) << quantity.c_str();

  if (name) msghead << "(" << *name << ")";

  if (type != "SPECIAL")
  {
    const int gid = container.get<int>(type);
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);
    msghead << " at " << type << " " << std::right << std::setw(3) << gid;
  }
  else
  {
    msghead << "\t";
  }

  // write something to screen depending if the result check was ok or not
  if (std::isnan(actresult))
  {
    // Result is 'not a number'
    std::cout << msghead.str() << "\t is NAN!\n";
    ret = 1;
  }
  else if (std::abs(actresult - givenresult) > tolerance)
  {
    // Result is wrong
    std::cout << msghead.str() << "\t is WRONG --> actresult=" << std::setw(24)
              << std::setprecision(17) << std::scientific << actresult
              << ", givenresult=" << std::setw(24) << givenresult << ", abs(diff)=" << std::setw(24)
              << std::abs(actresult - givenresult) << " >" << std::setw(24) << tolerance << "\n";

    ret = 1;
  }
  else
  {
    // Result is correct
    std::cout << msghead.str() << "\t is CORRECT"
              << ", abs(diff)=" << std::setw(24) << std::setprecision(17) << std::scientific
              << std::abs(actresult - givenresult) << " <" << std::setw(24) << tolerance << "\n";
  }

  return ret;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::Utils::ResultTest::match(const Core::IO::InputParameterContainer& container)
{
  return container.has_group(myname_);
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Utils::ResultTestManager::add_field_test(std::shared_ptr<ResultTest> test)
{
  fieldtest_.push_back(test);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Utils::ResultTestManager::test_all(MPI_Comm comm)
{
  int nerr = 0;                      // number of tests with errors
  int test_count = 0;                // number of tests performed
  int uneval_test_count = 0;         // number of unevaluated tests
  const int size = results_.size();  // total number of tests

  if (Core::Communication::my_mpi_rank(comm) == 0)
    Core::IO::cout << "\nChecking results of " << size << " tests:\n";

  for (const auto& result : results_)
  {
    for (const auto& fieldtest : fieldtest_)
    {
      if (fieldtest->match(result))
      {
        const auto& group = result.group(fieldtest->name());

        if (group.get_if<int>("ELEMENT") != nullptr)
          fieldtest->test_element(group, nerr, test_count);
        else if (group.get_if<int>("NODE") != nullptr)
          fieldtest->test_node(group, nerr, test_count);
        else if (group.get_if<int>("LINE") != nullptr || group.get_if<int>("SURFACE") != nullptr ||
                 group.get_if<int>("VOLUME") != nullptr)
          fieldtest->test_node_on_geometry(group, nerr, test_count, get_node_set());
        else
          fieldtest->test_special(group, nerr, test_count, uneval_test_count);
      }
    }
  }

  // print number of unevaluated tests to screen
  int guneval_test_count = 0;
  guneval_test_count = Core::Communication::sum_all(uneval_test_count, comm);
  if (guneval_test_count > 0 and Core::Communication::my_mpi_rank(comm) == 0)
    Core::IO::cout << guneval_test_count << " tests stay unevaluated" << Core::IO::endl;

  // determine the total number of errors
  int numerr;
  numerr = Core::Communication::sum_all(nerr, comm);

  if (numerr > 0)
  {
    FOUR_C_THROW("Result check failed with {} errors out of {} tests", numerr, size);
  }

  /* test_count == -1 means we had a special test routine. It's thus
   * illegal to use both a special routine and single tests. But who
   * wants that? */
  int count;
  if (test_count > -1)
  {
    int lcount = test_count + uneval_test_count;
    count = Core::Communication::sum_all(lcount, comm);

    /* It's indeed possible to count more tests than expected if you
     * happen to test values of a boundary element. We don't want this
     * FOUR_C_THROW to go off in that case. */
    if (count < size)
    {
      FOUR_C_THROW("expected {} tests but performed {}", size, count);
    }
  }

  if (Core::Communication::my_mpi_rank(comm) == 0) Core::IO::cout << "\nOK (" << count << ")\n";
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Utils::ResultTestManager::set_parsed_lines(
    std::vector<Core::IO::InputParameterContainer> results)
{
  results_ = std::move(results);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Utils::ResultTestManager::set_node_set(
    const std::vector<std::vector<std::vector<int>>>& nodeset)
{
  nodeset_ = std::move(nodeset);
}

FOUR_C_NAMESPACE_CLOSE
