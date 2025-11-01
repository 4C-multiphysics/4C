// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_resulttest.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                           dano 08/09 |
 *----------------------------------------------------------------------*/
Thermo::ResultTest::ResultTest(TimInt& tintegrator) : Core::Utils::ResultTest("THERMAL")
{
  temp_ = tintegrator.tempn();
  rate_ = tintegrator.raten();
  flux_ = tintegrator.freact();
  thrdisc_ = tintegrator.discretization();
}

/*----------------------------------------------------------------------*
 |                                                           dano 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::ResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != thrdisc_->name()) return;

  int node = container.get<int>("NODE");
  node -= 1;

  int havenode(thrdisc_->have_global_node(node));
  int isnodeofanybody(0);
  isnodeofanybody = Core::Communication::sum_all(havenode, thrdisc_->get_comm());

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW("Node {} does not belong to discretization {}", node + 1, thrdisc_->name());
  }
  else
  {
    // this implementation does not allow testing of heatfluxes
    if (thrdisc_->have_global_node(node))
    {
      const Core::Nodes::Node* actnode = thrdisc_->g_node(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->owner() != Core::Communication::my_mpi_rank(thrdisc_->get_comm())) return;

      std::string position = container.get<std::string>("QUANTITY");
      bool unknownpos = true;  // make sure the result value std::string can be handled
      double result = 0.0;     // will hold the actual result of run

      // test temperature
      if (temp_ != nullptr)
      {
        const Core::LinAlg::Map& tempmap = temp_->get_map();

        if (position == "temp")
        {
          unknownpos = false;
          result = temp_->local_values_as_span()[tempmap.lid(thrdisc_->dof(0, actnode, 0))];
        }
      }

      // test temperature rates
      if (rate_ != nullptr)
      {
        const Core::LinAlg::Map& ratemap = rate_->get_map();

        if (position == "rate")
        {
          unknownpos = false;
          result = rate_->local_values_as_span()[ratemap.lid(thrdisc_->dof(0, actnode, 0))];
        }
      }

      // test thermal flux
      if (flux_ != nullptr)
      {
        const Core::LinAlg::Map& fluxmap = flux_->get_map();

        if (position == "flux")
        {
          unknownpos = false;
          result = flux_->local_values_as_span()[fluxmap.lid(thrdisc_->dof(0, actnode, 0))];
        }
      }

      // catch position strings, which are not handled by thermo result test
      if (unknownpos) FOUR_C_THROW("Quantity '{}' not supported in thermo testing", position);

      // compare values
      const int err = compare_values(result, "NODE", container);
      nerr += err;
      test_count++;
    }
  }
}  // test_node

FOUR_C_NAMESPACE_CLOSE
