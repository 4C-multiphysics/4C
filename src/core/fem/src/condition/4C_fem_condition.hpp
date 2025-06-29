// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_CONDITION_HPP
#define FOUR_C_FEM_CONDITION_HPP


#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_legacy_enum_definitions_conditions.hpp"

#include <algorithm>
#include <memory>
#include <tuple>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::Conditions
{
  /**
   * Which type of entity is the ID of a condition referring to?
   */
  enum class EntityType
  {
    /**
     * Refers to a numbering of nodes specified in the input file directly. Note that an ID is only
     * unique within a GeometryType. The same ID may be used across different GeometryTypes.
     */
    legacy_id,

    /**
     * Refers to a node set ID.
     */
    node_set_id,

    /**
     * Refers to an element block ID.
     */
    element_block_id,
  };

  /*!
   * A condition is mainly used to realize boundary conditions. Parameters for the condition
   * are stored in a InputParameterContainer.
   * The condition can additionally store a discretization of the condition which is
   * driven by the discretization class that is evaluating this condition.
   * The discretization class is therefore a friend of the Condition and has access to
   * the protected methods dealing with the discretization of this condition.
   */
  class Condition
  {
   public:
    //! @name Enums and Friends

    /*!
    \brief discretization is a friend of the condition to have access
           to the private methods that would otherwise have to be public.

    */
    friend class Core::FE::Discretization;

    //@}

    //! @name Constructors and destructors

    /**
     * The way a condition is treated later on depends on the type of the
     * condition. E.g. Dirichlet conditions are treated differently from
     * Neumann conditions. How they are treated is not described here but in
     * Core::FE::Discretization.
     *
     * \param id (in): a unique id for this condition
     * \param type (in): type of the condition
     * \param buildgeometry (in): flag indicating whether explicit condition geometry
     *                            (elements) have to be build
     * \param gtype (in): type of geometric entity this condition lives on
     * \param entity_type (in): type of entity this condition is associated with
     */
    Condition(const int id, const Core::Conditions::ConditionType type, const bool buildgeometry,
        const Core::Conditions::GeometryType gtype, EntityType entity_type);

    /*!
    \brief Default constructor with type condition_none
    */
    Condition() = default;


    //@}

    //! @name Query methods

    /*!
    \brief Return condition id
    */
    [[nodiscard]] inline int id() const { return id_; }

    /*!
    \brief Return vector of my global node ids
    */
    [[nodiscard]] const std::vector<int>* get_nodes() const { return &nodes_; }

    /*!
    \brief Set vector of my global node ids
    */
    void set_nodes(const std::vector<int>& nodes) { nodes_ = nodes; }

    /*!
      \brief Return if a node gid is contained in this condition
     */
    [[nodiscard]] bool contains_node(int ngid) const
    {
      const std::vector<int>* n = get_nodes();
      // Core::Conditions::Condition nodes are ordered by design! So we can perform a binary
      // search here.
      return std::binary_search(n->begin(), n->end(), ngid);
    }

    /*!
    \brief Return flag indicating whether this condition needs to build a geometry
           description

    Some boundary conditions such as e.g. Neumann BCs need a geometry description
    to perform an integration on the boundary. Some BCs such as Dirichlet BCs
    don't need such a geometry description as it is sufficient to have access to
    the nodes only.<br>
    In case the condition needs to build elements describing the geometry of the
    condition the returned flag is true, otherwise its false;

    */
    [[nodiscard]] inline bool geometry_description() const { return buildgeometry_; }

    /*!
    \brief Return type of geometry this condition lives on

    The type of geometry this condition lives on determines what type of
    geometry description is build for this condition iff GeometryDescription()==true

    */
    [[nodiscard]] inline Core::Conditions::GeometryType g_type() const { return gtype_; }

    /**
     * Which entity does the condition refer to? The interpretation of the ID depends on this
     * entity type.
     */
    [[nodiscard]] inline EntityType entity_type() const { return entity_type_; }

    /*!
    \brief Print this Condition
    */
    void print(std::ostream& os) const;

    /*!
    \brief Return type of condition
    */
    [[nodiscard]] inline Core::Conditions::ConditionType type() const { return type_; }

    /*!
    \brief Get a reference to the geometry description of the condition

    */
    std::map<int, std::shared_ptr<Core::Elements::Element>>& geometry() { return geometry_; }

    [[nodiscard]] const std::map<int, std::shared_ptr<Core::Elements::Element>>& geometry() const
    {
      return geometry_;
    }

    //! Access the container that stores the input parameters.
    [[nodiscard]] const Core::IO::InputParameterContainer& parameters() const { return container_; }

    //! Access the container that stores the input parameters.
    Core::IO::InputParameterContainer& parameters() { return container_; }

    /*!
    \brief Adjust IDs of associated elements in order to obtain global
    unique IDs within one condition type
    */
    void adjust_id(const int shift);

    /**
     * Create a copy of this object but do not copy the geometry.
     */
    [[nodiscard]] std::shared_ptr<Core::Conditions::Condition> copy_without_geometry() const;

    //! Comparison operator.
    friend bool operator<(const Condition& lhs, const Condition& rhs);

    //@}

   private:
    //! @name Construction methods
    /*!
    \brief Add a geometry description to the condition

    A geometry description can be added to the condition.
    In case the condition refers to lines, surfaces or volumes, a
    geometry description might be needed to properly evaluate the condition
    (e.g. in the case of Neumann conditions).
    Such a geometry description is build in \ref
    Core::FE::Discretization::boundary_conditions_geometry and then added to this
    Condition. The geometry description consists of elements that are capable to perform the
    necessary operations on the condition (e.g. integrate a Neumann BC along a line). The matching
    nodes are taken from the underlying discretization itself. Also, it is actually the
    discretization class that drives this process, so do not add elements yourself to the condition,
    let the discretization do it for you.

    \param geom (in): Map of elements describing the geometry.
                      A deep copy of the map is made and stored.
                      Normally though, these elements are a line, surface or
                      volume elements produced by and shared with the discretization.
                      Do not mess with their std::shared_ptr!

    */
    void set_geometry(std::map<int, std::shared_ptr<Core::Elements::Element>>&& geom)
    {
      geometry_ = std::move(geom);
    }

    /*!
    \brief Delete a geometry description of the condition
    */
    void clear_geometry() { geometry_.clear(); }

    //@}

    Condition(const Condition& old) = default;

    Condition& operator=(const Condition& old) = default;

    //! Unique id of this condition, no second condition of the same type with same id may exist
    int id_{-1};

    //! global node ids
    std::vector<int> nodes_{};

    //! flag indicating whether this condition builds a geometry description or not
    bool buildgeometry_{};

    //! Type of this condition
    Core::Conditions::ConditionType type_{};

    //! Type of geometry the condition lives on
    Core::Conditions::GeometryType gtype_{};

    //! Type of entity this condition is associated with
    EntityType entity_type_{};

    //! Geometry description of this condition
    std::map<int, std::shared_ptr<Core::Elements::Element>> geometry_;

    Core::IO::InputParameterContainer container_;
  };  // class Condition

  inline bool operator<(const Condition& lhs, const Condition& rhs)
  {
    return std::tie(lhs.type_, lhs.id_) < std::tie(rhs.type_, rhs.id_);
  }

  //! Mapping of geometry type to its spatial dimensionality
  const std::map<GeometryType, unsigned int> geometry_type_to_dim = {
      std::make_pair(GeometryType::geometry_type_point, 0), std::make_pair(geometry_type_line, 1),
      std::make_pair(geometry_type_surface, 2), std::make_pair(geometry_type_volume, 3)};

}  // namespace Core::Conditions


//! << operator
std::ostream& operator<<(std::ostream& os, const Core::Conditions::Condition& cond);


FOUR_C_NAMESPACE_CLOSE

#endif
