// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_artery_resulttest.hpp"

#include "4C_art_net_explicitintegration.hpp"
#include "4C_art_net_impl_stationary.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Arteries::ArteryResultTest::ArteryResultTest(ArtNetExplicitTimeInt& art_net)
    : Core::Utils::ResultTest("ARTNET")
{
  dis_ = art_net.discretization();
  mysol_ = art_net.q_anp();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Arteries::ArteryResultTest::ArteryResultTest(ArtNetImplStationary& art_net)
    : Core::Utils::ResultTest("ARTNET")
{
  dis_ = art_net.discretization();
  mysol_ = art_net.pressurenp();
  myelevolflow_ = art_net.ele_volflow();
  myeleradius_ = art_net.ele_radius();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Arteries::ArteryResultTest::test_node(
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

      // Strange! It seems we might actually have a global node around
      // even if it does not belong to us. But here we are just
      // interested in our nodes!
      if (actnode->owner() != Core::Communication::my_mpi_rank(dis_->get_comm())) return;

      double result = 0.;
      const Core::LinAlg::Map& pnpmap = mysol_->get_map();
      std::string position = container.get<std::string>("QUANTITY");

      // test result value of single scalar field
      if (position == "area")
        result = mysol_->local_values_as_span()[pnpmap.lid(dis_->dof(actnode, 0))];
      else if (position == "pressure")
        result = mysol_->local_values_as_span()[pnpmap.lid(dis_->dof(0, actnode, 0))];
      else if (position == "flowrate")
        result = mysol_->local_values_as_span()[pnpmap.lid(dis_->dof(actnode, 1))];
      else
      {
        FOUR_C_THROW(
            "Quantity '{}' not supported in result-test of artery transport problems", position);
      }

      nerr += compare_values(result, "NODE", container);
      test_count++;
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Arteries::ArteryResultTest::test_element(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");

  if (dis != dis_->name()) return;

  int element = container.get<int>("ELEMENT");
  element -= 1;

  int haveelement(dis_->have_global_element(element));
  int iselementofanybody(0);
  iselementofanybody = Core::Communication::sum_all(haveelement, dis_->get_comm());

  if (iselementofanybody == 0)
  {
    FOUR_C_THROW("Element {} does not belong to discretization {}", element + 1, dis_->name());
  }
  else
  {
    if (dis_->have_global_element(element))
    {
      const Core::Elements::Element* actelement = dis_->g_element(element);

      // Here we are just interested in the elements that we own (i.e. a row element)!
      if (actelement->owner() != Core::Communication::my_mpi_rank(dis_->get_comm())) return;

      // extract name of quantity to be tested
      std::string quantity = container.get<std::string>("QUANTITY");

      double result = 0.;
      // test result value of single scalar field
      if (quantity == "volflow")
      {
        if (myelevolflow_ == nullptr) FOUR_C_THROW("Element volume flow not available");
        result =
            myelevolflow_->local_values_as_span()[dis_->element_row_map()->lid(actelement->id())];
      }
      else if (quantity == "radius")
      {
        if (myeleradius_ == nullptr) FOUR_C_THROW("Element radius not available");
        result =
            (*myeleradius_).local_values_as_span()[dis_->element_row_map()->lid(actelement->id())];
      }
      else
      {
        FOUR_C_THROW(
            "Quantity '{}' not supported in result-test of artery transport problems", quantity);
      }

      nerr += compare_values(result, "ELEMENT", container);
      test_count++;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
