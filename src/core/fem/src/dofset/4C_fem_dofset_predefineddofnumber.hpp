// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_PREDEFINEDDOFNUMBER_HPP
#define FOUR_C_FEM_DOFSET_PREDEFINEDDOFNUMBER_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::DOFSets
{
  /// A DofSet that owns a predefined number of dofs
  /*!

     We need a DofSet that

    - owns auxiliary dofs that belong to the same nodes as the original dof set, but
    - are not necessarily build based on element information, but can be chosen arbitrarily

    This DofSet is meant to be used as secondary DofSet in a discretization
    if there are two volume coupled Discretizations with non-matching nodes. Think
    of Structure-Thermo coupling. In this case, the structure discretization gets a
    auxiliary dof set with one degree of freedom (temperature) per node and the thermo
    discretization gets an auxiliary dof set with three degrees of freedom (displacement)
    per node.

    Using the input 'uniqueGIDs' one can decide whether the dofs build by the auxiliary dof set
    should get unique global IDs.
   */

  class DofSetPredefinedDoFNumber : public DofSet
  {
   public:
    /// Constructor
    explicit DofSetPredefinedDoFNumber(
        int numdofpernode, int numdofperelement, int numdofperface, bool uniqueGIDs)
        : DofSet(),
          numdofpernode_(numdofpernode),
          numdofpernodenodewise_(nullptr),
          numdofperelement_(numdofperelement),
          numdofperelementelewise_(nullptr),
          numdofperface_(numdofperface),
          numdofperfacefacewise_(nullptr),
          unique_gids_(uniqueGIDs)
    {
      return;
    }

    /// Constructor
    DofSetPredefinedDoFNumber(int numdofpernode,
        const std::shared_ptr<Core::LinAlg::Vector<int>> numdofperelement, int numdofperface,
        bool uniqueGIDs)
        : DofSet(),
          numdofpernode_(numdofpernode),
          numdofpernodenodewise_(nullptr),
          numdofperelement_(0),
          numdofperelementelewise_(numdofperelement),
          numdofperface_(numdofperface),
          numdofperfacefacewise_(nullptr),
          unique_gids_(uniqueGIDs)
    {
      return;
    }

    /// Constructor
    explicit DofSetPredefinedDoFNumber(
        const std::shared_ptr<Core::LinAlg::Vector<int>> numdofpernode,
        const std::shared_ptr<Core::LinAlg::Vector<int>> numdofperelement,
        const std::shared_ptr<Core::LinAlg::Vector<int>> numdofperface, bool uniqueGIDs)
        : DofSet(),
          numdofpernode_(0),
          numdofpernodenodewise_(numdofpernode),
          numdofperelement_(0),
          numdofperelementelewise_(numdofperelement),
          numdofperface_(0),
          numdofperfacefacewise_(numdofperface),
          unique_gids_(uniqueGIDs)
    {
      return;
    }

    /// create a copy of this object
    std::shared_ptr<DofSet> clone() override
    {
      return std::make_shared<DofSetPredefinedDoFNumber>(*this);
    }

    /// Add Dof Set to list #static_dofsets_
    void add_dof_set_to_list() override
    {
      if (unique_gids_)
        // add to static list -> the auxiliary dofs will get unique gids
        DofSet::add_dof_set_to_list();
      else
        // do nothing -> probably gids assigned to auxiliary dofs will not be unique
        return;
    }

    /// Assign dof numbers using all elements and nodes of the discretization.
    int assign_degrees_of_freedom(
        const Core::FE::Discretization& dis, const unsigned dspos, const int start) override
    {
      // redistribute internal vectors if necessary
      if (numdofpernodenodewise_ != nullptr and
          not numdofpernodenodewise_->get_map().same_as(*dis.node_col_map()))
      {
        Core::LinAlg::Vector<int> numdofpernodenodewise_rowmap(*dis.node_row_map());
        Core::LinAlg::export_to(*numdofpernodenodewise_, numdofpernodenodewise_rowmap);
        numdofpernodenodewise_ = std::make_shared<Core::LinAlg::Vector<int>>(*dis.node_col_map());
        Core::LinAlg::export_to(numdofpernodenodewise_rowmap, *numdofpernodenodewise_);
      }
      if (numdofperelementelewise_ != nullptr and
          not numdofperelementelewise_->get_map().same_as(*dis.element_col_map()))
      {
        Core::LinAlg::Vector<int> numdofperelementelewise_rowmap(*dis.element_row_map());
        Core::LinAlg::export_to(*numdofperelementelewise_, numdofperelementelewise_rowmap);
        numdofperelementelewise_ =
            std::make_shared<Core::LinAlg::Vector<int>>(*dis.element_col_map());
        Core::LinAlg::export_to(numdofperelementelewise_rowmap, *numdofperelementelewise_);
      }
      if (numdofperfacefacewise_ != nullptr) FOUR_C_THROW("Redistribution not yet implemented!");

      // call base class routine
      return Core::DOFSets::DofSet::assign_degrees_of_freedom(dis, dspos, start);
    }

   protected:
    /// get number of nodal dofs
    int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
      if (numdofpernodenodewise_ == nullptr)
        return numdofpernode_;
      else
        return (numdofpernodenodewise_->get_local_values())[node.lid()];
    }

    /// get number of element dofs for this element
    int num_dof_per_element(const Core::Elements::Element& element) const override
    {
      if (numdofperelementelewise_ == nullptr)
        return numdofperelement_;
      else
        return (numdofperelementelewise_->get_local_values())[element.lid()];
    }

    /// get number of element dofs for this element
    int num_dof_per_face(const Core::Elements::Element& element, int face) const override
    {
      if (numdofperfacefacewise_ == nullptr)
        return numdofperface_;
      else
      {
        FOUR_C_THROW("Not yet implemented!");
        return -1;
      }
    }

   private:
    /// number of dofs per node of dofset
    const int numdofpernode_;

    /// another member
    std::shared_ptr<Core::LinAlg::Vector<int>> numdofpernodenodewise_;

    /// number of dofs per element of dofset
    const int numdofperelement_;

    /// another member
    std::shared_ptr<Core::LinAlg::Vector<int>> numdofperelementelewise_;

    /// number of dofs per element of dofset
    const int numdofperface_;

    /// another member
    std::shared_ptr<Core::LinAlg::Vector<int>> numdofperfacefacewise_;

    /// bool indicating if the dofs should get unique global IDs
    /// can be set to false, if the dofs never appear in a global map)
    const bool unique_gids_;

  };  // DofSetPredefinedDoFNumber

}  // namespace Core::DOFSets


FOUR_C_NAMESPACE_CLOSE

#endif
