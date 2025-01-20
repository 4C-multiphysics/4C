// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ELEMAG_DIFF_ELE_HPP
#define FOUR_C_ELEMAG_DIFF_ELE_HPP

#include "4C_config.hpp"

#include "4C_elemag_ele.hpp"
#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class ElemagDiffType : public ElemagType
    {
     public:
      /// Type name
      std::string name() const override { return "ElemagDiffType"; }

      // Instance
      static ElemagDiffType& instance();
      /// Create
      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;
      /// Create
      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;
      /// Create
      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      /// Nodal block information
      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      /// Null space computation
      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      /// Element definition
      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      /// Instance
      static ElemagDiffType instance_;
    };


    /*!
    \brief electromagnetic diffusion element
    */
    class ElemagDiff : public Elemag
    {
     public:
      //@}
      //! @name constructors and destructors and related methods

      /*!
      \brief standard constructor
      */
      ElemagDiff(int id,  ///< A unique global id
          int owner       ///< Owner
      );
      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      ElemagDiff(const ElemagDiff& old);

      /*!
      \brief Deep copy this instance and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Get vector of std::shared_ptrs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /*!
      \brief Get vector of std::shared_ptrs to the surfaces of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      /*!
      \brief Get std::shared_ptr to the internal face adjacent to this element as master element and
      the parent_slave element
      */
      std::shared_ptr<Core::Elements::Element> create_face_element(
          Core::Elements::Element* parent_slave,  //!< parent slave element
          int nnode,                              //!< number of surface nodes
          const int* nodeids,                     //!< node ids of surface element
          Core::Nodes::Node** nodes,              //!< nodes of surface element
          const int lsurface_master,  //!< local surface number w.r.t master parent element
          const int lsurface_slave,   //!< local surface number w.r.t slave parent element
          const std::vector<int>& localtrafomap  //! local trafo map
          ) override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return ElemagDiffType::instance().unique_par_object_id();
      }

      //@}

      //! @name Geometry related methods

      //@}

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return ElemagDiffType::instance();
      }

      /// Element location data
      Core::Elements::LocationData lm_;

     private:
      // don't want = operator
      ElemagDiff& operator=(const ElemagDiff& old);

    };  // class ElemagDiff

    /// class ElemagDiffBoundaryType
    class ElemagDiffBoundaryType : public ElemagBoundaryType
    {
     public:
      /// Type name
      std::string name() const override { return "ElemagDiffBoundaryType"; }
      // Instance
      static ElemagDiffBoundaryType& instance();
      // Create
      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

     private:
      /// Instance
      static ElemagDiffBoundaryType instance_;
    };

    /// class ElemagDiffBoundary
    class ElemagDiffBoundary : public ElemagBoundary
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
      \param parent: The parent elemag element of this surface
      \param lsurface: the local surface number of this surface w.r.t. the parent element
      */
      ElemagDiffBoundary(int id, int owner, int nnode, const int* nodeids,
          Core::Nodes::Node** nodes, Discret::Elements::ElemagDiff* parent, const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      ElemagDiffBoundary(const ElemagDiffBoundary& old);

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
        return ElemagDiffBoundaryType::instance().unique_par_object_id();
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

      //@}

      //! @name Access methods


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
        return parent_element()->num_dof_per_node(node);
      }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      /// Return the instance of the element type
      Core::Elements::ElementType& element_type() const override
      {
        return ElemagDiffBoundaryType::instance();
      }

      //@}

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

      \param dis (in)      : the discretization this element belongs to
      \param la (out)      : location data for all dofsets of the discretization
      \param doDirichlet (in): whether to get the Dirichlet flags
      \param condstring (in): Name of condition to be evaluated
      \param condstring (in):  List of parameters for use at element level
      */
      void location_vector(const Core::FE::Discretization& dis, Core::Elements::LocationArray& la,
          bool doDirichlet, const std::string& condstring,
          Teuchos::ParameterList& params) const override;

     private:
      // don't want = operator
      ElemagDiffBoundary& operator=(const ElemagDiffBoundary& old);

    };  // class ElemagDiffBoundary

    /// class ElemagDiffIntFaceType
    class ElemagDiffIntFaceType : public Core::Elements::ElementType
    {
     public:
      /// Name of the element type
      std::string name() const override { return "ElemagDiffIntFaceType"; }

      /// Instance
      static ElemagDiffIntFaceType& instance();

      /// Create
      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      /// Nodal block information
      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override {};

      /// Null space
      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented");
        return nullspace;
      };

     private:
      /// instance of the class
      static ElemagDiffIntFaceType instance_;
    };

    /// class ElemagDiffIntFace
    class ElemagDiffIntFace : public ElemagIntFace
    {
     public:
      //! @name Constructors and destructors and related methods

      //! number of space dimensions
      /*!
      \brief Standard Constructor

      \param id: A unique global id
      \param owner: Processor owning this surface
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param master_parent: The master parent elemag element of this surface
      \param slave_parent: The slave parent elemag element of this surface
      \param lsurface_master: the local surface number of this surface w.r.t. the master parent
      element \param lsurface_slave: the local surface number of this surface w.r.t. the slave
      parent element \param localtrafomap: transformation map between the local coordinate systems
      of the face w.r.t the master parent element's face's coordinate system and the slave element's
      face's coordinate system
      */
      ElemagDiffIntFace(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Discret::Elements::ElemagDiff* parent_master, Discret::Elements::ElemagDiff* parent_slave,
          const int lsurface_master, const int lsurface_slave,
          const std::vector<int> localtrafomap);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      ElemagDiffIntFace(const ElemagDiffIntFace& old);

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
        return ElemagDiffIntFaceType::instance().unique_par_object_id();
      }

      //@}

      //! @name Access methods

      /*!
      \brief create the location vector for patch of master and slave element

      \note All dofs shared by master and slave element are contained only once. Dofs from interface
      nodes are also included.
      */
      void patch_location_vector(Core::FE::Discretization& discretization,  ///< discretization
          std::vector<int>& nds_master,        ///< nodal dofset w.r.t master parent element
          std::vector<int>& nds_slave,         ///< nodal dofset w.r.t slave parent element
          std::vector<int>& patchlm,           ///< local map for gdof ids for patch of elements
          std::vector<int>& master_lm,         ///< local map for gdof ids for master element
          std::vector<int>& slave_lm,          ///< local map for gdof ids for slave element
          std::vector<int>& face_lm,           ///< local map for gdof ids for face element
          std::vector<int>& lm_masterToPatch,  ///< local map between lm_master and lm_patch
          std::vector<int>& lm_slaveToPatch,   ///< local map between lm_slave and lm_patch
          std::vector<int>& lm_faceToPatch,    ///< local map between lm_face and lm_patch
          std::vector<int>&
              lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch  ///< local map between slave nodes and nodes in patch
      );
      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return ElemagDiffIntFaceType::instance();
      }

      //@}

      /*!
      \brief return the master parent elemag element
      */
      Discret::Elements::ElemagDiff* parent_master_element() const
      {
        Core::Elements::Element* parent =
            this->Core::Elements::FaceElement::parent_master_element();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<Discret::Elements::ElemagDiff*>(parent) != nullptr,
            "Master element is no elemag_diff element");
        return static_cast<Discret::Elements::ElemagDiff*>(parent);
      }

      /*!
      \brief return the slave parent elemag element
      */
      Discret::Elements::ElemagDiff* parent_slave_element() const
      {
        Core::Elements::Element* parent = this->Core::Elements::FaceElement::parent_slave_element();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<Discret::Elements::ElemagDiff*>(parent) != nullptr,
            "Slave element is no elemag_diff element");
        return static_cast<Discret::Elements::ElemagDiff*>(parent);
      }

      //@}

     private:
      // don't want = operator
      ElemagDiffIntFace& operator=(const ElemagDiffIntFace& old);

    };  // class ElemagDiffIntFace

  }  // namespace Elements
}  // namespace Discret



FOUR_C_NAMESPACE_CLOSE

#endif
