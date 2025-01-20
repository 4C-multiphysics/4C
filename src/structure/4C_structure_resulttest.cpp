// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_resulttest.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_timint.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
StruResultTest::StruResultTest(Solid::TimInt& tintegrator)
    : Core::Utils::ResultTest("STRUCTURE"),
      timeintegrator_(Core::Utils::shared_ptr_from_ref(tintegrator))
{
  dis_ = tintegrator.dis();
  vel_ = tintegrator.vel();
  acc_ = tintegrator.acc();
  strudisc_ = tintegrator.discretization();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StruResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // this implementation does not allow testing of stresses !

  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != strudisc_->name()) return;

  int node = container.get<int>("NODE");
  node -= 1;

  int havenode(strudisc_->have_global_node(node));
  int isnodeofanybody(0);
  Core::Communication::sum_all(&havenode, &isnodeofanybody, 1, strudisc_->get_comm());

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW(
        "Node %d does not belong to discretization %s", node + 1, strudisc_->name().c_str());
  }
  else
  {
    if (strudisc_->have_global_node(node))
    {
      const Core::Nodes::Node* actnode = strudisc_->g_node(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->owner() != Core::Communication::my_mpi_rank(strudisc_->get_comm())) return;

      std::string position = container.get<std::string>("QUANTITY");
      bool unknownpos = true;  // make sure the result value std::string can be handled
      double result = 0.0;     // will hold the actual result of run

      // test displacements or pressure
      if (dis_ != nullptr)
      {
        const Epetra_BlockMap& disnpmap = dis_->Map();
        int idx = -1;
        if (position == "dispx")
          idx = 0;
        else if (position == "dispy")
          idx = 1;
        else if (position == "dispz")
          idx = 2;
        else if (position == "press")
          idx = 3;

        if (idx >= 0)
        {
          unknownpos = false;
          int lid = disnpmap.LID(strudisc_->dof(0, actnode, idx));
          if (lid < 0)
            FOUR_C_THROW("You tried to test %s on nonexistent dof %d on node %d", position.c_str(),
                idx, actnode->id());
          result = (*dis_)[lid];
        }
      }

      // test velocities
      if (vel_ != nullptr)
      {
        const Epetra_BlockMap& velnpmap = vel_->Map();
        int idx = -1;
        if (position == "velx")
          idx = 0;
        else if (position == "vely")
          idx = 1;
        else if (position == "velz")
          idx = 2;

        if (idx >= 0)
        {
          unknownpos = false;
          int lid = velnpmap.LID(strudisc_->dof(0, actnode, idx));
          if (lid < 0)
            FOUR_C_THROW("You tried to test %s on nonexistent dof %d on node %d", position.c_str(),
                idx, actnode->id());
          result = (*vel_)[lid];
        }
      }

      // test accelerations
      if (acc_ != nullptr)
      {
        const Epetra_BlockMap& accnpmap = acc_->Map();
        int idx = -1;
        if (position == "accx")
          idx = 0;
        else if (position == "accy")
          idx = 1;
        else if (position == "accz")
          idx = 2;

        if (idx >= 0)
        {
          unknownpos = false;
          int lid = accnpmap.LID(strudisc_->dof(0, actnode, idx));
          if (lid < 0)
            FOUR_C_THROW("You tried to test %s on nonexistent dof %d on node %d", position.c_str(),
                idx, actnode->id());
          result = (*acc_)[lid];
        }
      }

      // catch position std::strings, which are not handled by structure result test
      if (unknownpos)
        FOUR_C_THROW("Quantity '%s' not supported in structure testing", position.c_str());

      // compare values
      const int err = compare_values(result, "NODE", container);
      nerr += err;
      test_count++;
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StruResultTest::test_special(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // extract name of quantity to be tested
  std::string quantity = container.get<std::string>("QUANTITY");

  // get result to be tested on all processors
  const double result = get_special_result_for_testing(quantity);

  // compare values on one processor only, as they are the same everywhere
  if (Core::Communication::my_mpi_rank(strudisc_->get_comm()) == 0)
  {
    const int err = compare_values(result, "SPECIAL", container);
    nerr += err;
    test_count++;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double StruResultTest::get_special_result_for_testing(const std::string& quantity)
{
  // initialize variable for result
  double result(0.);

  if (quantity == "lin_iters")  // number of iterations in solid linear solver
    result = static_cast<double>(timeintegrator_->solver()->get_num_iters());
  else if (quantity == "lin_iters_contact")  // number of iterations in contact linear solver
    result = static_cast<double>(timeintegrator_->contact_solver()->get_num_iters());
  else  // Catch unknown quantity strings
    FOUR_C_THROW("Quantity '%s' not supported in structure result test!", quantity.c_str());

  return result;
}

FOUR_C_NAMESPACE_CLOSE
