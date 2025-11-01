// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_result_test.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_global_data.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FLD::FluidResultTest::FluidResultTest(FluidImplicitTimeInt& fluid)
    : Core::Utils::ResultTest("FLUID")
{
  fluiddis_ = fluid.discret_;
  myerror_ = fluid.evaluate_error_compared_to_analytical_sol();
  mysol_ = fluid.velnp_;

  // quantities not implemented in the HDG formulation
  if (Global::Problem::instance()->spatial_approximation_type() != Core::FE::ShapeFunctionType::hdg)
  {
    mytraction_ = fluid.stressmanager_->get_pre_calc_stresses(*fluid.trueresidual_);
    mywss_ = fluid.stressmanager_->get_pre_calc_wall_shear_stresses(*fluid.trueresidual_);
    mydivu_ = fluid.evaluate_div_u();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::FluidResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != fluiddis_->name()) return;

  int node = container.get<int>("NODE");
  node -= 1;

  int havenode(fluiddis_->have_global_node(node));
  int isnodeofanybody(0);
  isnodeofanybody = Core::Communication::sum_all(havenode, fluiddis_->get_comm());

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW("Node {} does not belong to discretization {}", node + 1, fluiddis_->name());
  }
  else
  {
    if (fluiddis_->have_global_node(node))
    {
      const Core::Nodes::Node* actnode = fluiddis_->g_node(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->owner() != Core::Communication::my_mpi_rank(fluiddis_->get_comm())) return;

      double result = 0.;

      const Core::LinAlg::Map& velnpmap = mysol_->get_map();

      const int numdim = Global::Problem::instance()->n_dim();

      std::string position = container.get<std::string>("QUANTITY");
      if (position == "velx")
        result = mysol_->local_values_as_span()[velnpmap.lid(fluiddis_->dof(0, actnode, 0))];
      else if (position == "vely")
        result = mysol_->local_values_as_span()[velnpmap.lid(fluiddis_->dof(0, actnode, 1))];
      else if (position == "velz")
      {
        if (numdim == 2) FOUR_C_THROW("Cannot test result for velz in 2D case.");
        result = mysol_->local_values_as_span()[velnpmap.lid(fluiddis_->dof(0, actnode, 2))];
      }
      else if (position == "pressure")
      {
        if (fluiddis_->num_dof(0, actnode) < (numdim + 1))
          FOUR_C_THROW("too few dofs at node {} for pressure testing", actnode->id());
        result = mysol_->local_values_as_span()[velnpmap.lid(fluiddis_->dof(0, actnode, numdim))];
      }
      else if (position == "tractionx")
        result = mytraction_->local_values_as_span()[(mytraction_->get_map())
                .lid(fluiddis_->dof(0, actnode, 0))];
      else if (position == "tractiony")
        result = mytraction_->local_values_as_span()[(mytraction_->get_map())
                .lid(fluiddis_->dof(0, actnode, 1))];
      else if (position == "tractionz")
      {
        if (numdim == 2) FOUR_C_THROW("Cannot test result for tractionz in 2D case.");
        result = mytraction_->local_values_as_span()[(mytraction_->get_map())
                .lid(fluiddis_->dof(0, actnode, 2))];
      }
      else if (position == "wssx")
        result =
            mywss_->local_values_as_span()[(mywss_->get_map()).lid(fluiddis_->dof(0, actnode, 0))];
      else if (position == "wssy")
        result =
            mywss_->local_values_as_span()[(mywss_->get_map()).lid(fluiddis_->dof(0, actnode, 1))];
      else if (position == "wssz")
      {
        if (numdim == 2) FOUR_C_THROW("Cannot test result for wssz in 2D case.");
        result =
            mywss_->local_values_as_span()[(mywss_->get_map()).lid(fluiddis_->dof(0, actnode, 2))];
      }
      else if (position == "L2errvel")
        result = (*myerror_)[0];
      else if (position == "L2errpre")
        result = (*myerror_)[1];
      else if (position == "H1errvel")
        result = (*myerror_)[2];
      else if (position == "divu")
        result = (*mydivu_);
      else if (position == "L2errmix")
        result = (*myerror_)[0];
      else if (position == "L2errden")
        result = (*myerror_)[1];
      else if (position == "L2errmom")
        result = (*myerror_)[2];
      else
        FOUR_C_THROW("Quantity '{}' not supported in fluid testing", position);

      nerr += compare_values(result, "NODE", container);
      test_count++;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
