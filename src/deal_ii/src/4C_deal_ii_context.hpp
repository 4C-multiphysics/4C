// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_CONTEXT_HPP
#define FOUR_C_DEAL_II_CONTEXT_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_element_conversion.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"

#include <deal.II/fe/mapping_fe_field.h>


FOUR_C_NAMESPACE_OPEN



namespace DealiiWrappers
{

  namespace Internal
  {
    /**
     * Convenience function to get the global dof indices of the local dofs of a given element.
     * The result is written back to the @p dof_indices vector.
     */
    void get_dof_indices(const Core::FE::Discretization& discretization,
        const Core::Elements::Element* element,
        std::vector<dealii::types::global_dof_index>& dof_indices);

    // Create a function alias for the above function
    constexpr auto get_dof_indices_four_c_ordering = get_dof_indices;

    /**
     * Same as above, but with a provided local reordering of the dofs. i.e. the GID of the i-th
     * local dof is written to dof_indices[local_reorder[i]].
     */
    void get_dof_indices_with_local_reorder(const Core::FE::Discretization& discretization,
        const Core::Elements::Element* element, const std::span<const int>& local_reorder,
        std::vector<dealii::types::global_dof_index>& dof_indices);

    constexpr auto get_dof_indices_deal_ii_ordering = get_dof_indices_with_local_reorder;
  }  // namespace Internal


  /**
   * @brief Context class that allows to translate between a 4C discretization and a deal.II
   * Triangulation together with some FE intformation.
   *
   * This class contains all the necessary information to translate discretizations from 4C to
   * deal.II.
   * It is built on a (non owned) pair of objects that describe the Finite Element discretization
   * in the two frameworks that describe the same mesh and the same mathematical finite element
   * setup.
   * In deal.II terms it is something like a DofHandler, insofar as it provides access to the
   * FE information on the cells of the triangulation and also to the local to global dof map of the
   * dofs via the get_dof_indices_four_c_ordering() method.
   * However, it is built to work with the 4C indexing of the shape functions.
   * Given that the context is only useful when it is used together with a specific discretization
   * and related triangulation, there is no public constructor for this class. Instead, the
   * class may only be created through the create_triangulation() function that builds the matching
   * triangulation from a given 4C discretization.
   */
  template <int dim, int spacedim = dim>
  class Context
  {
   public:
    /**
     * Const access to the underlying triangulation of this context.
     */
    const dealii::Triangulation<dim, spacedim>& get_triangulation() const;

    /**
     * Const access to the underlying discretization of this context.
     */
    const Core::FE::Discretization& get_discretization() const;

    /**
     * Convert a cell iterator to the corresponding 4C element.
     */
    const Core::Elements::Element* to_element(
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell) const;


    /**
     * Get the number of different finite elements that are used in this context.
     * This corresponds to the number of different shape functions that are used in the
     * Discretization (taking also into account different numbers of components).
     */
    unsigned int n_finite_elements() const;

    /**
     * Access to the finite elements that are used in this context.
     */
    const dealii::hp::FECollection<dim, spacedim>& get_finite_elements() const;

    /**
     * Get the finite element that is in use on a given cell of the triangulation.
     */
    const dealii::FiniteElement<dim, spacedim>& fe_on_cell(
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell) const;

    /**
     * Get the active fe index for the given cell. This index corresponds to the index of the
     * finite element in the finite_elements collection of this context.
     */
    unsigned int get_active_fe_index(
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell) const;

    /**
     * Extract the global dof indices for the local dofs of the given cell. Here we use the natural
     * 4C ordering of the dofs, i.e. as if we would use the shape functions defined within 4C.
     */
    void get_dof_indices_four_c_ordering(
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        std::vector<dealii::types::global_dof_index>& dof_indices) const;


   private:
    /**
     * Constructor for the Context class. Both the triangulation and the discretization must be
     * kept alive as long as this context is used, as this class does not take ownership of them.
     */
    Context(const dealii::Triangulation<dim, spacedim>& triangulation,
        const Core::FE::Discretization& discretization);

    // declare the create_triangulation function as a friend so that it can access the private
    // members as it acts as a factory function for this class.
    friend Context<dim, spacedim> create_triangulation<dim, spacedim>(
        dealii::Triangulation<dim, spacedim>& triangulation,
        const Core::FE::Discretization& discretization);


    const dealii::Triangulation<dim, spacedim>& triangulation_;
    const Core::FE::Discretization& discretization_;

    // Store the local mapping between deal.II cells and 4C elements
    std::vector<int> active_cell_index_to_element_lid_;
    // vector holding a list of active fe indices for each cell in the triangulation.
    // this is used to track which fe/mapping is active on a given cell/element.
    std::vector<unsigned int> active_fe_indices_;

    // These are in the same order as the finite elements within the finite_elements collection.
    dealii::hp::FECollection<dim, spacedim> finite_elements_;
  };  // class Context


  // ============================================================
  // Implementation of the Context class

  template <int dim, int spacedim>
  Context<dim, spacedim>::Context(const dealii::Triangulation<dim, spacedim>& triangulation,
      const Core::FE::Discretization& discretization)
      : triangulation_(triangulation), discretization_(discretization)
  {
  }


  template <int dim, int spacedim>
  const dealii::Triangulation<dim, spacedim>& Context<dim, spacedim>::get_triangulation() const
  {
    return triangulation_;
  }


  template <int dim, int spacedim>
  const Core::FE::Discretization& Context<dim, spacedim>::get_discretization() const
  {
    return discretization_;
  }


  template <int dim, int spacedim>
  unsigned int Context<dim, spacedim>::n_finite_elements() const
  {
    return finite_elements_.size();
  }


  template <int dim, int spacedim>
  const dealii::hp::FECollection<dim, spacedim>& Context<dim, spacedim>::get_finite_elements() const
  {
    return finite_elements_;
  }


  template <int dim, int spacedim>
  unsigned int Context<dim, spacedim>::get_active_fe_index(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell) const
  {
    FOUR_C_ASSERT(&cell->get_triangulation() == &triangulation_,
        "The cell iterator does not belong to the triangulation of this context.");
    FOUR_C_ASSERT(cell->active_cell_index() < active_fe_indices_.size(),
        "The active cell index {} is out of bounds for the active cell indices vector of size "
        "{}.",
        cell->active_cell_index(), active_fe_indices_.size());
    return active_fe_indices_[cell->active_cell_index()];
  }


  template <int dim, int spacedim>
  const Core::Elements::Element* Context<dim, spacedim>::to_element(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell) const
  {
    FOUR_C_ASSERT(&cell->get_triangulation() == &triangulation_,
        "The cell iterator does not belong to the triangulation of this context.");
    return discretization_.l_row_element(
        active_cell_index_to_element_lid_[cell->active_cell_index()]);
  }


  template <int dim, int spacedim>
  const dealii::FiniteElement<dim, spacedim>& Context<dim, spacedim>::fe_on_cell(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell) const
  {
    return finite_elements_[get_active_fe_index(cell)];
  }


  template <int dim, int spacedim>
  void Context<dim, spacedim>::get_dof_indices_four_c_ordering(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
      std::vector<dealii::types::global_dof_index>& dof_indices) const
  {
    Internal::get_dof_indices(get_discretization(), to_element(cell), dof_indices);
  }
}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif
