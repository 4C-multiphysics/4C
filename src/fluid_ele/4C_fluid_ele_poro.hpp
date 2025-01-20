// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_PORO_HPP
#define FOUR_C_FLUID_ELE_PORO_HPP

#include "4C_config.hpp"

#include "4C_fluid_ele.hpp"
#include "4C_inpar_structure.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class FluidPoroEleType : public FluidType
    {
     public:
      std::string name() const override { return "FluidPoroEleType"; }

      static FluidPoroEleType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

      //! pre-evaluation
      void pre_evaluate(Core::FE::Discretization& dis, Teuchos::ParameterList& p,
          std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
          std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
          std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
          std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
          std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3) override;

     private:
      static FluidPoroEleType instance_;
    };

    /*!
    \brief A C++ wrapper for the fluid element
    */
    class FluidPoro : public virtual Fluid
    {
     public:
      //!@}
      //! @name constructors and destructors and related methods

      /*!
      \brief standard constructor
      */
      FluidPoro(int id,  //!< A unique global id
          int owner      //!< number of owning processor
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      FluidPoro(const FluidPoro& old);

      /*!
      \brief Deep copy this instance of fluid and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return FluidPoroEleType::instance().unique_par_object_id();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref pack and \ref unpack are used to communicate this element

      */
      void pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref pack and \ref unpack are used to communicate this element
      */
      void unpack(Core::Communication::UnpackBuffer& buffer) override;

      /*!
      \brief Get vector of std::shared_ptrs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /*!
      \brief Get vector of std::shared_ptrs to the surfaces of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      //!@}

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return FluidPoroEleType::instance();
      }


      void set_kinematic_type(Inpar::Solid::KinemType kintype) { kintype_ = kintype; }

      Inpar::Solid::KinemType kinematic_type() const { return kintype_; }

      //! Set anisotropic permeability directions obtained from structure element during cloning
      void set_anisotropic_permeability_directions(
          const std::vector<std::vector<double>>& ref_anisotropic_permeability_directions)
      {
        anisotropic_permeability_directions_ = ref_anisotropic_permeability_directions;
      }

      //! Set anisotropic permeability coefficients obtained from structure element during cloning
      void set_anisotropic_permeability_nodal_coeffs(
          const std::vector<std::vector<double>>& ref_anisotropic_permeability_nodal_coeffs)
      {
        anisotropic_permeability_nodal_coeffs_ = ref_anisotropic_permeability_nodal_coeffs;
      }

      //! Provide the anisotropic permeability directions of the element
      const std::vector<std::vector<double>>& get_anisotropic_permeability_directions() const
      {
        return anisotropic_permeability_directions_;
      }

      //! Provide the nodal anisotropic permeability coefficients of the element
      const std::vector<std::vector<double>>& get_anisotropic_permeability_nodal_coeffs() const
      {
        return anisotropic_permeability_nodal_coeffs_;
      }

      //!@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element, that is, call the element routines to evaluate fluid
      element matrices and vectors or evaluate errors, statistics or updates etc. directly.

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      //!@}

     private:
      //! don't want = operator
      FluidPoro& operator=(const FluidPoro& old);

      //! kinematic type
      Inpar::Solid::KinemType kintype_;

      //! directions for anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_directions_;

      //! nodal coefficients for anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_nodal_coeffs_;
    };

    /*!
    \brief An element representing a boundary element of a fluid element

    */

    class FluidPoroBoundaryType : public FluidBoundaryType
    {
     public:
      std::string name() const override { return "FluidPoroBoundaryType"; }

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      static FluidPoroBoundaryType& instance();

     private:
      static FluidPoroBoundaryType instance_;
    };

    class FluidPoroBoundary : public FluidBoundary
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
      FluidPoroBoundary(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Discret::Elements::Fluid* parent, const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      FluidPoroBoundary(const FluidPoroBoundary& old);

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
        return FluidPoroBoundaryType::instance().unique_par_object_id();
      }

      //!@}

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return FluidPoroBoundaryType::instance();
      }

      //!@}

      //! @name Evaluation

      /*!
      \brief Evaluate element

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      //!@}

      //! @name Evaluate methods

      //!@}

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
      \param doDirichlet (in): whether to get the Dirichlet flags
      \param condstring (in) : Name of condition to be evaluated
      \param params (in)     : List of parameters for use at element level
      */
      void location_vector(const Core::FE::Discretization& dis, Core::Elements::LocationArray& la,
          bool doDirichlet, const std::string& condstring,
          Teuchos::ParameterList& params) const override;

     private:
      // don't want = operator
      FluidPoroBoundary& operator=(const FluidPoroBoundary& old);
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
