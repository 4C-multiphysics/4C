// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_FE_VALUES_CONTEXT_HPP
#define FOUR_C_DEAL_II_FE_VALUES_CONTEXT_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_element_conversion.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"

#include <deal.II/grid/grid_tools.h>
#include <deal.II/hp/fe_values.h>

FOUR_C_NAMESPACE_OPEN
namespace DealiiWrappers
{
  namespace Internal
  {
    /**
     * Class that implements the common functionality for FEValuesContext and
     * FEFaceValuesContext.
     */
    template <int dim, int spacedim>
    class FEValuesContextImplementation
    {
      const Context<dim, spacedim>& context_;

      struct CellData
      {
        const Core::Elements::Element* element;
        std::vector<int> four_c_shape_indices;  // reordering from deal.II to 4C
        unsigned int active_index;
      };
      CellData cell_data_;

     protected:
      FEValuesContextImplementation(const Context<dim, spacedim>& context);

      /**
       * Update and cache the necessary data for the given cell. Only the new 4C data is handled
       * here the actual reinit of the deal.II FEValues object is done in the derived classes as
       * they hold the deal.II FEValues object.
       */
      void reinit_cell_data(
          const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell);

      /**
       * Get the active index of the cell/element on which this object was last reinitialized.
       */
      unsigned int get_active_index() const;


     public:
      /**
       * On the local cell/element on which this object was last reinitialized, return the
       * transformation from the deal.II shape index to the four c shape index.
       * ,i.e., for the j-th shape function in the deal.II ordering we can access its
       * four c index via local_four_c_indexing()[j].
       */
      std::span<const int> local_four_c_indexing() const;
      /**
       * On the local cell/element on which this object was last reinitialized, return the
       * transformation from the 4C shape index to the deal.II shape index.
       * I.e. for the j-th shape function in the deal.II ordering we can access its
       * deal.II index via local_dealii_indexing()[j].
       * @note This is obviously the identity transformation, but is provided for convenience.
       */
      std::ranges::iota_view<int, int> local_dealii_indexing() const;



      /**
       * Fill the dof_indices vector with the global dof indices of dofs corresponding to the
       * local shape function indices on the cell/element on which this object was last
       * reinitialized. The dof_indices are ordered in the four_c ordering. I.e. in the ordering
       * in which the 4C shape-functions are defined. That means that for the j-th shape
       * function dof_indices[j] contains its corresponding global dof index.
       * @param dof_indices
       */
      void get_dof_indices_four_c_ordering(
          std::vector<dealii::types::global_dof_index>& dof_indices) const;

      /**
       * Fill the dof_indices vector with the global dof indices of dofs corresponding to the
       * local shape function indices on the cell/element on which this object was last
       * reinitialized. The dof_indices are ordered in the deal.II ordering. I.e. in the ordering
       * in which the deal.II shape-functions are defined. That means that for the j-th shape
       * function dof_indices[j] contains its corresponding global dof index.
       * @param dof_indices
       */
      void get_dof_indices_dealii_ordering(
          std::vector<dealii::types::global_dof_index>& dof_indices) const;

      /**
       * Access to the underlying context object that was used to create this object.
       */
      const Context<dim, spacedim>& get_context() const;
    };
  }  // namespace Internal



  /**
   * \brief A wrapper/context class around the deal.II FEValues class to evaluate finite elements.
   *
   * This class is a wrapper around the deal.II FEValues class that handles the setup and
   * reinitialization as well as some helper functions to access a dealii::FEValues object in
   * the context of a 4C discretization.
   *
   * The goal of this class is to allow for access of FEValues that is setup in a way that it
   * coincides with the 4C definitions of the equivalent mathematical objects (mostly the
   * actual finite element shape functions).
   *
   * The class follows a similar design as the deal.II FEValues class, and in fact the main
   * evaluation is dispatched to a deal.II FEValues  object that can (and should) be accessed
   * through the get_present_fe_values() method.
   * Before that is possible, this object must be initialized on a cell/element through the
   * reinit() methods. In that step a geometric entity (cell, element, face, etc.) is passed
   * and the FEValues object will internally precompute all the necessary data for the
   * operations that were requested in the constructor. The requests are specified through the
   * dealii::UpdateFlags that are passed to the constructor.
   * For more details on the FEValues class, see the deal.II documentation, as well as their
   * tutorials.
   *
   * <h3> REORDERING: </h3>
   *
   * The main difficulty and task of this class is that 4C and deal.II have different ways of
   * handling Finite Elements as well as different orderings of the shape functions.
   * The design of this class is focused on allowing the user to assemble matrices/vectors that
   * are compatible with the ones assembled using 4C directly. This means that we need to get from
   * a *local ordering* of shape functions defined on the current cell/element to the *global
   * ordering* defined by the global dof distribution of the 4C discretization.
   * The assembly typically happens in the following way:
   *
   * 1. loop over all shape functions of the current cell and for each compute some quantity to
   * write into local storage (e.g. a local matrix or vector)
   * 2. distribute the local storage to the global dof indices
   *
   * Since the natural ordering of the shape functions in deal.II might not match the 4C one
   * this requires reordering at some point. This can either be done
   *  - at the local step. In this case we need a local reordering of the shape functions
   *    so that we can loop over them in the 4C ordering.
   *  - at the global step. In this case we need to reorder the vector that contains the mapping
   *    from local shape function indices to global dof indices.
   *
   * This class provides both options, but you need to be aware of which one you are using and
   * that mixing them will lead to wrong results.
   * To use these methods you need to use the local_***_indexing() method together with the
   * corresponding get_dof_indices_***_ordering() method.
   * In this case *** == "four_c" corresponds to method 1 from above, while *** == "dealii"
   * corresponds to method 2.
   *
   * Below you can find a short example of how to use the class in a typical assembly loop.
   * Usage example:
   *  @code
   *  FEValuesContext phi(context, discretization, quadrature_collection)
   *  dealii::Vector<> global_vector;
   *  for (const auto& cell : tria.active_cell_iterators())
   *  {
   *    // cache the data for the current cell
   *    phi.reinit(cell);
   *
   *    // get the indexing
   *    std::vector<dealii::types::global_dof_index> dof_indices;
   *    auto local_indexing = phi.local_four_c_indexing();
   *    // or phi.local_dealii_indexing()
   *    phi.get_dof_indices_four_c_ordering(dof_indices);
   *    // or phi.get_dof_indices_dealii_ordering(dof_indices);
   *
   *    auto phi_present = phi.get_present_fe_values();
   *    dealii::Vector<double> local_vector(phi_present.dofs_per_cell);
   *
   *    for (auto q_index : phi_present.quadrature_point_indices()) // quadrature point loop
   *    {
   *      for (unsigned int i = 0; i < phi_present.dofs_per_cell; ++i) // local shape function loop
   *      {
   *        // compute some local value for the i-th shape function
   *        // make sure to use the local indexing during evaluation
   *        local_vector[i] = phi_present.shape_value(local_indexing[i], q_index);
   *      }
   *    }
   *    // distribute the local vector to the global dof indices
   *    global_vector.add(dof_indices, local_vector);
   *  }
   *  @endcode
   */
  template <int dim, int spacedim = dim>
  class FEValuesContext : public Internal::FEValuesContextImplementation<dim, spacedim>
  {
    dealii::hp::FEValues<dim, spacedim> fe_values_;

   public:
    FEValuesContext(const dealii::hp::MappingCollection<dim, spacedim>& mapping,
        const Context<dim, spacedim>& context,
        const dealii::hp::QCollection<dim>& quadrature_collection,
        dealii::UpdateFlags update_flags);

    /**
     * Reinitialize the FEValuesContext on the given cell/element. This will also reinitialize the
     * FEValues object with the correct finite element and quadrature rule.
     * This function does NOT reset the underlying setup of the FEValues object, instead it is used
     * to set up and cache all the data that is needed to do the required finite element
     * computations. See the FEValues class of deal.II for more details.
     * @param cell cell on the triangulation within the Context on which this object
     * was created.
     */
    void reinit(const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell);

    /**
     * get the fe_values that is initialized on the cell/element.
     * Note that this FEValues object is initialized on the dealii finite element
     * with its internal shape function ordering and Quadrature setup.
     */
    const dealii::FEValues<dim>& get_present_fe_values() const;
  };

  /**
   * Same as FEValuesContext, but for faces of cells/elements using the deal.II
   * FEFaceValues class.
   */
  template <int dim, int spacedim = dim>
  class FEFaceValuesContext : public Internal::FEValuesContextImplementation<dim, spacedim>
  {
    dealii::hp::FEFaceValues<dim, spacedim> fe_face_values_;

   public:
    FEFaceValuesContext(const dealii::hp::MappingCollection<dim, spacedim>& mapping,
        const Context<dim, spacedim>& context,
        const dealii::hp::QCollection<dim - 1>& quadrature_collection,
        dealii::UpdateFlags update_flags);



    void reinit(const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        unsigned int face_index);

    /**
     * get the FEFaceValues object that is initialized on the cell/element.
     * Note that this FEFaceValues object is initialized on the deal.II finite element
     * with its internal shape function ordering and Quadrature setup.
     */
    const dealii::FEFaceValues<dim>& get_present_fe_values() const;
  };

  // ===================================================================================
  // FEValuesContextImplementation implementation


  template <int dim, int spacedim>
  Internal::FEValuesContextImplementation<dim, spacedim>::FEValuesContextImplementation(
      const Context<dim, spacedim>& context)
      : context_(context)
  {
    cell_data_.element = nullptr;
    cell_data_.active_index = dealii::numbers::invalid_unsigned_int;
  }
  template <int dim, int spacedim>
  void Internal::FEValuesContextImplementation<dim, spacedim>::reinit_cell_data(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell)
  {
    const auto* element = get_context().to_element(cell);
    if (cell_data_.element == element) return;  // no need to reinitialize if the cell is the same
    cell_data_.element = element;
    cell_data_.active_index = get_context().get_active_fe_index(cell);
    const auto& fe = get_context().get_finite_elements()[cell_data_.active_index];
    FOUR_C_ASSERT(fe.dofs_per_cell == fe.n_components() * cell_data_.element->num_node(),
        "The number of dofs per cell in the deal.II finite element does not match the "
        "number of dofs per cell in the 4C finite element.");
    ConversionTools::FourCToDeal::reindex_shape_functions<dim, spacedim>(
        cell_data_.element->shape(), fe.n_components(), cell_data_.four_c_shape_indices);
  }
  template <int dim, int spacedim>
  unsigned int Internal::FEValuesContextImplementation<dim, spacedim>::get_active_index() const
  {
    return cell_data_.active_index;
  }
  template <int dim, int spacedim>
  std::span<const int>
  Internal::FEValuesContextImplementation<dim, spacedim>::local_four_c_indexing() const
  {
    return cell_data_.four_c_shape_indices;
  }
  template <int dim, int spacedim>
  std::ranges::iota_view<int, int>
  Internal::FEValuesContextImplementation<dim, spacedim>::local_dealii_indexing() const
  {
    return std::views::iota(0, cell_data_.four_c_shape_indices.size());
  }
  template <int dim, int spacedim>
  void Internal::FEValuesContextImplementation<dim, spacedim>::get_dof_indices_four_c_ordering(
      std::vector<dealii::types::global_dof_index>& dof_indices) const
  {
    Internal::get_dof_indices_four_c_ordering(
        get_context().get_discretization(), cell_data_.element, dof_indices);
  }
  template <int dim, int spacedim>
  void Internal::FEValuesContextImplementation<dim, spacedim>::get_dof_indices_dealii_ordering(
      std::vector<dealii::types::global_dof_index>& dof_indices) const
  {
    Internal::get_dof_indices_deal_ii_ordering(get_context().get_discretization(),
        cell_data_.element, cell_data_.four_c_shape_indices, dof_indices);
  }
  template <int dim, int spacedim>
  const Context<dim, spacedim>&
  Internal::FEValuesContextImplementation<dim, spacedim>::get_context() const
  {
    return context_;
  }


  // ===================================================================================
  // FEValuesContext implementation

  template <int dim, int spacedim>
  FEValuesContext<dim, spacedim>::FEValuesContext(
      const dealii::hp::MappingCollection<dim, spacedim>& mapping,
      const Context<dim, spacedim>& context,
      const dealii::hp::QCollection<dim>& quadrature_collection, dealii::UpdateFlags update_flags)
      : Internal::FEValuesContextImplementation<dim, spacedim>(context),
        fe_values_(mapping, context.get_finite_elements(), quadrature_collection, update_flags)
  {
  }

  template <int dim, int spacedim>
  void FEValuesContext<dim, spacedim>::reinit(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell)
  {
    this->reinit_cell_data(cell);
    fe_values_.reinit(
        cell, this->get_active_index(), this->get_active_index(), this->get_active_index());
  }
  template <int dim, int spacedim>
  const dealii::FEValues<dim>& FEValuesContext<dim, spacedim>::get_present_fe_values() const
  {
    return fe_values_.get_present_fe_values();
  }

  // ===================================================================================
  // FEFaceValuesContext implementation

  template <int dim, int spacedim>
  FEFaceValuesContext<dim, spacedim>::FEFaceValuesContext(
      const dealii::hp::MappingCollection<dim, spacedim>& mapping,
      const Context<dim, spacedim>& context,
      const dealii::hp::QCollection<dim - 1>& quadrature_collection,
      dealii::UpdateFlags update_flags)
      : Internal::FEValuesContextImplementation<dim, spacedim>(context),
        fe_face_values_(mapping, context.get_finite_elements(), quadrature_collection, update_flags)
  {
  }
  template <int dim, int spacedim>
  void FEFaceValuesContext<dim, spacedim>::reinit(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
      unsigned int face_index)
  {
    this->reinit_cell_data(cell);
    // reinit internal fe_values object
    fe_face_values_.reinit(cell, face_index, this->get_active_index(), this->get_active_index(),
        this->get_active_index());
  }
  template <int dim, int spacedim>
  const dealii::FEFaceValues<dim>& FEFaceValuesContext<dim, spacedim>::get_present_fe_values() const
  {
    return fe_face_values_.get_present_fe_values();
  }

}  // namespace DealiiWrappers
FOUR_C_NAMESPACE_CLOSE



#endif  // FOUR_C_DEAL_II_FE_VALUES_CONTEXT_HPP
