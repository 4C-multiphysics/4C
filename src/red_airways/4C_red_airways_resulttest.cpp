// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_red_airways_resulttest.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_red_airways_implicitintegration.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Airway::RedAirwayResultTest::RedAirwayResultTest(RedAirwayImplicitTimeInt& airways)
    : Core::Utils::ResultTest("RED_AIRWAY")
{
  dis_ = airways.discretization();
  mynodesol_pressure_ = airways.pnp();
  mynodesol_flow_in_ = airways.qin_np();
  mynodesol_flow_out_ = airways.qout_np();

  myelemsol_pressure_external_ = airways.pext_np();
  myelemsol_acinivol_ = airways.acini_volume();
  myelemsol_airwayvol_ = airways.airway_volume();
  myelemsol_open_ = airways.open();
  myelemsol_opening_trajectory_ = airways.opening_trajectory();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Airway::RedAirwayResultTest::test_node(
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
    FOUR_C_THROW("Node {} does not belong to discretisation {}", node + 1, dis_->name());
  }
  else
  {
    if (dis_->have_global_node(node))
    {
      Core::Nodes::Node* actnode = dis_->g_node(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->owner() != Core::Communication::my_mpi_rank(dis_->get_comm())) return;

      double result = 0.;
      const Core::LinAlg::Map& nodemap = mynodesol_pressure_->get_map();
      std::string position = container.get<std::string>("QUANTITY");

      // test result value of single scalar field
      if (position == "pressure")
      {
        result = mynodesol_pressure_->local_values_as_span()[nodemap.lid(dis_->dof(actnode, 0))];
      }
      else if (position == "flow_in")
      {
        result = mynodesol_flow_in_->local_values_as_span()[nodemap.lid(dis_->dof(actnode, 0))];
      }
      else if (position == "flow_out")
      {
        result = mynodesol_flow_out_->local_values_as_span()[nodemap.lid(dis_->dof(actnode, 0))];
      }
      // test result values for a system of scalars
      else
      {
        FOUR_C_THROW("Quantity '{}' not supported in result-test of red_airway problems", position);
      }

      nerr += compare_values(result, "NODE", container);
      test_count++;
    }
  }
}


/*----------------------------------------------------------------------*
 * Element based result test for red-airway problems. Tests the results *
 * of acini_volume.                                                     *
 *                                                         roth 11/2014 *
 *----------------------------------------------------------------------*/
void Airway::RedAirwayResultTest::test_element(
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
    FOUR_C_THROW("Node {} does not belong to discretisation {}", element + 1, dis_->name());
  }
  else
  {
    if (dis_->have_global_element(element))
    {
      const Core::Elements::Element* actelement = dis_->g_element(element);

      // Here we are just interested in the elements that we own (i.e. a row element)!
      if (actelement->owner() != Core::Communication::my_mpi_rank(dis_->get_comm())) return;

      double result = 0.;
      const Core::LinAlg::Map& elementmap = myelemsol_acinivol_->get_map();
      std::string position = container.get<std::string>("QUANTITY");
      if (position == "pressure_external")
      {
        result =
            myelemsol_pressure_external_->local_values_as_span()[elementmap.lid(actelement->id())];
      }
      else if (position == "acini_volume")
      {
        result = myelemsol_acinivol_->local_values_as_span()[elementmap.lid(actelement->id())];
      }
      else if (position == "airway_volume")
      {
        result = myelemsol_airwayvol_->local_values_as_span()[elementmap.lid(actelement->id())];
      }
      else if (position == "opening_status")
      {
        result = myelemsol_open_->local_values_as_span()[elementmap.lid(actelement->id())];
      }
      else if (position == "opening_trajectory")
      {
        result =
            myelemsol_opening_trajectory_->local_values_as_span()[elementmap.lid(actelement->id())];
      }
      else
      {
        FOUR_C_THROW(
            "Quantity '{}' not supported in result-test of red_airway problems.", position);
      }

      nerr += compare_values(result, "ELEMENT", container);
      test_count++;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
