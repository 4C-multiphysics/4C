// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_1d_pipe_flow_resulttest.hpp"

#include "4C_cardiovascular0d_arterialproxdist.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_reduced_lung_1d_pipe_flow_main.hpp"
#include "4C_utils_result_test.hpp"

FOUR_C_NAMESPACE_OPEN
void ReducedLung1D::ResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  std::string dis = container.get<std::string>("DIS");
  if (dis != dis_->name())
  {
    return;
  }

  int gid = container.get<int>("NODE") - 1;
  const int is_local_node = dis_->have_global_node(gid);

  if (const int is_node_of_any_rank = Core::Communication::sum_all(is_local_node, dis_->get_comm());
      is_node_of_any_rank == 0)
  {
    FOUR_C_THROW(" Node {} does not belong to discretization {}", gid + 1, dis_->name());
  }
  if (!dis_->have_global_node(gid)) return;

  Core::Nodes::Node* node_to_check = dis_->g_node(gid);
  if (node_to_check->owner() != Core::Communication::my_mpi_rank(dis_->get_comm())) return;

  const Core::LinAlg::Map& dofmap = sol_->get_map();

  std::string quantity = container.get<std::string>("QUANTITY");

  double result = 0.0;

  if (quantity == "area")
  {
    const int dof = dis_->dof(node_to_check, 0);  // A
    result = sol_->local_values_as_span()[dofmap.lid(dof)];
  }
  else if (quantity == "velocity")
  {
    const int dof = dis_->dof(node_to_check, 1);  // u
    result = sol_->local_values_as_span()[dofmap.lid(dof)];
  }
  else if (quantity == "flow")
  {
    const int dof_A = dis_->dof(node_to_check, 0);
    const int dof_u = dis_->dof(node_to_check, 1);  // u
    result = sol_->local_values_as_span()[dofmap.lid(dof_A)] *
             sol_->local_values_as_span()[dofmap.lid(dof_u)];
  }
  else
  {
    std::cout << "Unrecognized quantity \"" << quantity << "\"" << std::endl;
    FOUR_C_THROW("Unsupported QUANTITY '{}'", quantity);
  }

  nerr += compare_values(result, "NODE", container);
  test_count++;
}

FOUR_C_NAMESPACE_CLOSE