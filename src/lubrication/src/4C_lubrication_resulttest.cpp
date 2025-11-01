// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_lubrication_resulttest.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_lubrication_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | ctor                                                     wirtz 11/15 |
 *----------------------------------------------------------------------*/
Lubrication::ResultTest::ResultTest(std::shared_ptr<TimIntImpl> lubrication)
    : Core::Utils::ResultTest("LUBRICATION"),
      dis_(lubrication->discretization()),
      mysol_(lubrication->prenp()),
      mynumiter_(lubrication->iter_num())
{
}


/*----------------------------------------------------------------------*
 | test node                                                wirtz 11/15 |
 *----------------------------------------------------------------------*/
void Lubrication::ResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != dis_->name()) return;

  int node = container.get<int>("NODE");
  node -= 1;

  int havenode(dis_->have_global_node(node));
  int isnodeofanybody(0);
  isnodeofanybody = Core::Communication::sum_all(havenode, dis_->get_comm());

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW("Node {} does not belong to discretization {}", node + 1, dis_->name());
  }
  else
  {
    if (dis_->have_global_node(node))
    {
      Core::Nodes::Node* actnode = dis_->g_node(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->owner() != Core::Communication::my_mpi_rank(dis_->get_comm())) return;

      // extract name of quantity to be tested
      std::string quantity = container.get<std::string>("QUANTITY");

      // get result to be tested
      const double result = result_node(quantity, actnode);

      nerr += compare_values(result, "NODE", container);
      test_count++;
    }
  }
}


/*----------------------------------------------------------------------*
 | get nodal result to be tested                            wirtz 11/15 |
 *----------------------------------------------------------------------*/
double Lubrication::ResultTest::result_node(
    const std::string quantity,  //! name of quantity to be tested
    Core::Nodes::Node* node      //! node carrying the result to be tested
) const
{
  // initialize variable for result
  double result(0.);

  // extract row map from solution vector
  const Core::LinAlg::Map& prenpmap = mysol_->get_map();

  // test result value of pressure field
  if (quantity == "pre")
    result = mysol_->local_values_as_span()[prenpmap.lid(dis_->dof(0, node, 0))];

  // catch unknown quantity strings
  else
    FOUR_C_THROW("Quantity '{}' not supported in result test!", quantity);

  return result;
}  // Lubrication::ResultTest::ResultNode


/*-------------------------------------------------------------------------------------*
 | test special quantity not associated with a particular element or node  wirtz 11/15 |
 *-------------------------------------------------------------------------------------*/
void Lubrication::ResultTest::test_special(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // make sure that quantity is tested only once
  if (Core::Communication::my_mpi_rank(dis_->get_comm()) == 0)
  {
    // extract name of quantity to be tested
    std::string quantity = container.get<std::string>("QUANTITY");

    // get result to be tested
    const double result = result_special(quantity);

    // compare values
    const int err = compare_values(result, "SPECIAL", container);
    nerr += err;
    test_count++;
  }
}


/*----------------------------------------------------------------------*
 | get special result to be tested                          wirtz 11/15 |
 *----------------------------------------------------------------------*/
double Lubrication::ResultTest::result_special(
    const std::string quantity  //! name of quantity to be tested
) const
{
  // initialize variable for result
  double result(0.);

  if (quantity == "numiterlastnewton") result = (double)mynumiter_;
  // catch unknown quantity strings
  else
    FOUR_C_THROW("Quantity '{}' not supported in result test!", quantity);

  return result;
}  // Lubrication::ResultTest::result_special

FOUR_C_NAMESPACE_CLOSE
