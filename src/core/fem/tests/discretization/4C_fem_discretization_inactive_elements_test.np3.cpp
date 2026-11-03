// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_map.hpp"
#include "4C_unittest_utils_create_discretization_helper_test.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <memory>
#include <set>
#include <vector>

namespace
{
  using namespace FourC;

  // A cube with 2x2x2 hex8 elements and 27 nodes distributed over all ranks. Element (i,j,k) has
  // the id 4*i + 2*j + k, node (i,j,k) has the id 9*i + 3*j + k.
  class InactiveElementsTest : public testing::Test
  {
   public:
    InactiveElementsTest() { TESTING::fill_discretization_hyper_cube(discretization_, 2, comm_); }

   protected:
    // Return the ids of the row elements of this rank which are part of the given ids
    std::vector<int> my_row_elements(const std::set<int>& element_ids) const
    {
      std::vector<int> my_element_ids;
      for (auto ele : discretization_.my_row_element_range())
        if (element_ids.contains(ele.global_id())) my_element_ids.push_back(ele.global_id());
      return my_element_ids;
    }

    // Deactivate the elements with the given ids
    void deactivate_elements(const std::set<int>& element_ids)
    {
      discretization_.deactivate_elements(my_row_elements(element_ids));
    }

    MPI_Comm comm_ = MPI_COMM_WORLD;
    Core::FE::Discretization discretization_{"dummy", comm_, 3};
    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard_;
  };

  TEST_F(InactiveElementsTest, ElementsAreActiveByDefault)
  {
    for (auto ele : discretization_.my_col_element_range())
      EXPECT_TRUE(ele.user_element()->is_active()) << "element " << ele.global_id();

    EXPECT_EQ(Core::FE::dofs_without_active_elements(discretization_).num_global_elements(), 0);
  }

  TEST_F(InactiveElementsTest, PackUnpackKeepsInactiveElements)
  {
    // the partitioning might leave ranks without elements
    if (discretization_.num_my_col_elements() == 0) return;

    Core::Elements::Element& element = *discretization_.l_col_element(0);
    element.set_active(false);

    Core::Communication::PackBuffer data;
    element.pack(data);

    Core::Communication::UnpackBuffer buffer(data());
    std::unique_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
    const auto* unpacked_element = dynamic_cast<const Core::Elements::Element*>(object.get());

    ASSERT_NE(unpacked_element, nullptr);
    EXPECT_EQ(unpacked_element->id(), element.id());
    EXPECT_FALSE(unpacked_element->is_active());
  }

  TEST_F(InactiveElementsTest, DeactivateElementsUpdatesGhostedElements)
  {
    const std::set<int> inactive_elements = {0, 5};
    deactivate_elements(inactive_elements);

    for (auto ele : discretization_.my_col_element_range())
    {
      EXPECT_EQ(ele.user_element()->is_active(), !inactive_elements.contains(ele.global_id()))
          << "element " << ele.global_id() << " on rank "
          << Core::Communication::my_mpi_rank(comm_);
    }
  }

  TEST_F(InactiveElementsTest, DeactivateElements)
  {
    const std::vector<int> elements = my_row_elements({4, 5, 6, 7});
    discretization_.deactivate_elements(elements);

    for (auto ele : discretization_.my_col_element_range())
      EXPECT_EQ(ele.user_element()->is_active(), ele.global_id() < 4)
          << "element " << ele.global_id();
  }

  TEST_F(InactiveElementsTest, DeactivateElementsIgnoresAlreadyInactiveElements)
  {
    deactivate_elements({0});

    // element 0 is inactive already and ignored
    discretization_.deactivate_elements(my_row_elements({0, 3, 6}));

    const std::set<int> inactive_elements = {0, 3, 6};
    for (auto ele : discretization_.my_col_element_range())
      EXPECT_EQ(ele.user_element()->is_active(), !inactive_elements.contains(ele.global_id()))
          << "element " << ele.global_id();
  }

  TEST_F(InactiveElementsTest, EvaluateSkipsInactiveElements)
  {
    deactivate_elements({1, 2, 7});

    std::set<int> my_evaluated_elements;
    discretization_.evaluate(
        [&](Core::Elements::Element& ele) { my_evaluated_elements.insert(ele.id()); });

    for (auto ele : discretization_.my_col_element_range())
    {
      EXPECT_EQ(my_evaluated_elements.contains(ele.global_id()), ele.user_element()->is_active())
          << "element " << ele.global_id();
    }

    int my_num_evaluated_row_elements = 0;
    for (auto ele : discretization_.my_row_element_range())
      if (my_evaluated_elements.contains(ele.global_id())) ++my_num_evaluated_row_elements;
    EXPECT_EQ(Core::Communication::sum_all(my_num_evaluated_row_elements, comm_), 5);
  }

  TEST_F(InactiveElementsTest, NodesOfSingleInactiveCornerElement)
  {
    // only the corner node 0 is exclusively connected to element 0
    deactivate_elements({0});

    const auto free_dofs = Core::FE::dofs_without_active_elements(discretization_);
    EXPECT_EQ(free_dofs.num_global_elements(), 3);

    if (discretization_.node_row_map()->my_gid(0))
    {
      for (const int dof : discretization_.dof(0, discretization_.g_node(0)))
        EXPECT_TRUE(free_dofs.my_gid(dof));
    }
  }

  TEST_F(InactiveElementsTest, NodesOfInactiveElementLayer)
  {
    // deactivate the layer of elements with i = 0, all nodes with i = 0 become free
    deactivate_elements({0, 1, 2, 3});

    const auto free_dofs = Core::FE::dofs_without_active_elements(discretization_);
    EXPECT_EQ(free_dofs.num_global_elements(), 9 * 3);

    for (auto node : discretization_.my_row_node_range())
    {
      const bool is_free_node = node.global_id() < 9;
      for (const int dof : discretization_.dof(0, node))
        EXPECT_EQ(free_dofs.my_gid(dof), is_free_node) << "node " << node.global_id();
    }
  }

  TEST_F(InactiveElementsTest, CopyElementActiveFlagsRejectsReactivation)
  {
    Core::FE::Discretization target("target", comm_, 3);
    TESTING::fill_discretization_hyper_cube(target, 2, comm_, false);

    std::vector<int> target_row_ids;
    for (auto ele : target.my_row_element_range())
      if (ele.global_id() == 0) target_row_ids.push_back(0);
    target.deactivate_elements(target_row_ids);

    EXPECT_ANY_THROW(Core::FE::copy_element_active_flags(discretization_, target));

    for (auto ele : target.my_col_element_range())
      EXPECT_EQ(ele.user_element()->is_active(), ele.global_id() != 0)
          << "element " << ele.global_id();
  }

  TEST_F(InactiveElementsTest, CopyElementActiveFlagsToClonedDiscretization)
  {
    Core::FE::Discretization target("target", comm_, 3);
    TESTING::fill_discretization_hyper_cube(target, 2, comm_, false);

    deactivate_elements({3, 4, 6});

    EXPECT_EQ(Core::FE::copy_element_active_flags(discretization_, target), 3);

    const std::set<int> inactive_elements = {3, 4, 6};
    for (auto ele : target.my_col_element_range())
    {
      EXPECT_EQ(ele.user_element()->is_active(), !inactive_elements.contains(ele.global_id()))
          << "element " << ele.global_id();
    }

    // the scalar valued target has one dof per node
    deactivate_elements({0, 1, 2, 5, 7});
    EXPECT_EQ(Core::FE::copy_element_active_flags(discretization_, target), 5);
    EXPECT_EQ(Core::FE::dofs_without_active_elements(target).num_global_elements(), 27);
  }
}  // namespace
