// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_resulttest.hpp"

#include "4C_ale.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
ALE::AleResultTest::AleResultTest(ALE::Ale& ale)
    : Core::Utils::ResultTest("ALE"), aledis_(ale.discretization()), dispnp_(ale.dispnp())
{
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::AleResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != aledis_->name()) return;

  int node = container.get<int>("NODE");
  node -= 1;

  int havenode(aledis_->have_global_node(node));
  int isnodeofanybody(0);
  isnodeofanybody = Core::Communication::sum_all(havenode, aledis_->get_comm());

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW("Node {} does not belong to discretization {}", node + 1, aledis_->name());
  }
  else
  {
    if (aledis_->have_global_node(node))
    {
      Core::Nodes::Node* actnode = aledis_->g_node(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->owner() != Core::Communication::my_mpi_rank(aledis_->get_comm())) return;

      double result = 0.;

      const Core::LinAlg::Map& dispnpmap = dispnp_->get_map();

      std::string position = container.get<std::string>("QUANTITY");
      if (position == "dispx")
      {
        result = dispnp_->local_values_as_span()[dispnpmap.lid(aledis_->dof(actnode, 0))];
      }
      else if (position == "dispy")
      {
        result = dispnp_->local_values_as_span()[dispnpmap.lid(aledis_->dof(actnode, 1))];
      }
      else if (position == "dispz")
      {
        result = dispnp_->local_values_as_span()[dispnpmap.lid(aledis_->dof(actnode, 2))];
      }
      else
      {
        FOUR_C_THROW("Quantity '{}' not supported in ALE testing", position);
      }

      nerr += compare_values(result, "NODE", container);
      test_count++;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
