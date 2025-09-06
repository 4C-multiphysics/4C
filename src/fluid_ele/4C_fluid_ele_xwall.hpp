// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_XWALL_HPP
#define FOUR_C_FLUID_ELE_XWALL_HPP

#include "4C_config.hpp"

#include "4C_fluid_ele.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Discret
{
  namespace Elements
  {
    class FluidXWallType : public FluidType
    {
     public:
      std::string name() const override { return "FluidXWallType"; }

      static FluidXWallType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static FluidXWallType instance_;
    };


    /*!
    \brief A C++ wrapper for the fluid element
    */
    class FluidXWall : public Fluid
    {
     public:
      //! @name constructors and destructors and related methods

      /*!
      \brief standard constructor
      */
      FluidXWall(int id,  ///< A unique global id
          int owner       ///< ???
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      FluidXWall(const FluidXWall& old);

      /*!
      \brief Deep copy this instance of fluid and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual Core::Elements::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        // number of Dof's is fluid-specific.
        const int nsd = Core::FE::get_dimension(distype_);
        if (nsd > 1)
          return 2 * nsd + 2;  // The enrichment dofs have to be a multiple of the usual dofs for
                               // the nullspace
        else
          FOUR_C_THROW("1D Fluid elements are not supported");

        return 0;
      }

      Core::Elements::ElementType& element_type() const override
      {
        return FluidXWallType::instance();
      }

      /*!
      \brief Return value how expensive it is to evaluate this element

      \param double (out): cost to evaluate this element
      */
      // the standard element is 10.0
      double evaluation_cost() override { return 50.0; }

      int unique_par_object_id() const override
      {
        return FluidXWallType::instance().unique_par_object_id();
      }

      /*!
      \brief Get vector of std::shared_ptrs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /*!
      \brief Get vector of std::shared_ptrs to the surfaces of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      //@}

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;



     protected:
      // don't want = operator
      FluidXWall& operator=(const FluidXWall& old);

    };  // class Fluid


    class FluidXWallBoundaryType : public FluidBoundaryType
    {
     public:
      std::string name() const override { return "FluidXWallBoundaryType"; }

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      static FluidXWallBoundaryType& instance();

     private:
      static FluidXWallBoundaryType instance_;
    };


    // class FluidXWallBoundary

    class FluidXWallBoundary : public FluidBoundary
    {
     public:
      //! @name Constructors and destructors and related methods

      //! number of space dimensions
      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner: Processor owning this surface
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param parent: The parent fluid element of this surface
      \param lsurface: the local surface number of this surface w.r.t. the parent element
      */
      FluidXWallBoundary(int id, int owner, int nnode, const int* nodeids,
          Core::Nodes::Node** nodes, Discret::Elements::Fluid* parent, const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      FluidXWallBoundary(const FluidXWallBoundary& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int unique_par_object_id() const override
      {
        return FluidXWallBoundaryType::instance().unique_par_object_id();
      }

      //@}

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return FluidXWallBoundaryType::instance();
      }

      //@}

      //! @name Evaluation

      //@}

      //! @name Evaluate methods

      //@}

      /*!
      \brief Return the location vector of this element

      The method computes degrees of freedom this element addresses.
      Degree of freedom ordering is as follows:<br>
      First all degrees of freedom of adjacent nodes are numbered in
      local nodal order, then the element internal degrees of freedom are
      given if present.<br>
      If a derived element has to use a different ordering scheme,
      it is welcome to overload this method as the assembly routines actually
      don't care as long as matrices and vectors evaluated by the element
      match the ordering, which is implicitly assumed.<br>
      Length of the output vector matches number of degrees of freedom
      exactly.<br>
      This version is intended to fill the LocationArray with the dofs
      the element will assemble into. In the standard case these dofs are
      the dofs of the element itself. For some special conditions (e.g.
      the weak dirichlet boundary condition) a surface element will assemble
      into the dofs of a volume element.<br>

      \note The degrees of freedom returned are not necessarily only nodal dofs.
            Depending on the element implementation, output might also include
            element dofs.

      \param dis (in)        : the discretization this element belongs to
      \param la (out)        : location data for all dofsets of the discretization
      \param condstring (in) : Name of condition to be evaluated
      \param params (in)     : List of parameters for use at element level
      */
      void location_vector(const Core::FE::Discretization& dis, Core::Elements::LocationArray& la,
          const std::string& condstring, Teuchos::ParameterList& params) const override;

     private:
      // don't want = operator
      FluidXWallBoundary& operator=(const FluidXWallBoundary& old);

    };  // class FluidXWallBoundary

  }  // namespace Elements
}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
