// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_HDG_HPP
#define FOUR_C_SCATRA_ELE_HDG_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_scatra_ele.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class ScaTraHDGType : public TransportType
    {
     public:
      std::string name() const override { return "ScaTraHDGType"; }

      static ScaTraHDGType& instance();

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
      static ScaTraHDGType instance_;
    };


    /*!
    \brief HDG transport element
    */
    class ScaTraHDG : public Transport
    {
     public:
      //! @name constructors and destructors and related methods

      /*!
      \brief standard constructor
      */
      ScaTraHDG(int id,  //!< A unique global id
          int owner      //!< ???
      );

      //! Makes a deep copy of a Element
      ScaTraHDG(const ScaTraHDG& old);

      /*!
      \brief Deep copy this instance of scatra and return pointer to the copy

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
        return ScaTraHDGType::instance().unique_par_object_id();
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
      \brief Pack Material

      \ref pack_material and \ref unpack_material are used to adapt material evaluation

      */
      virtual void pack_material(Core::Communication::PackBuffer& data) const;

      /*!
      \brief Unpack Material

      \ref pack_material and \ref unpack_material are used to adapt material evaluation

      */
      virtual void unpack_material(Core::Communication::UnpackBuffer& buffer) const;

      //! initialize the element
      int initialize() override;

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;

      //@}

      //! @name Access methods
      /*!
      \brief Get number of degrees of freedom per node

      HDG element: No dofs are associated with nodes
      */
      int num_dof_per_node(const Core::Nodes::Node&) const override { return 0; }

      //! Get number of degrees of freedom per face
      int num_dof_per_face(const unsigned face) const override
      {
        return num_dof_per_component(face);
      }

      //! Get number of dofs per component per face
      int num_dof_per_component(const unsigned face) const override
      {
        return Core::FE::get_basis_size(Core::FE::get_ele_face_shape_type(this->distype_),
            (this->faces()[face])->degree(), completepol_);
      }

      //! Get number of degrees of freedom per element, zero for the primary dof set
      //! and equal to the given number for the secondary dof set
      int num_dof_per_element() const override { return 0; }

      //! Returns the degree of the element
      int degree() const override { return degree_; }

      //! Returns the degree of the element
      int degree_old() const { return degree_old_; }

      //! Sets the degree of the element
      void set_degree(int degree)
      {
        degree_old_ = degree_;
        degree_ = degree;
        return;
      }

      //! Sets the number of all interior dofs of the element
      void set_dofs(int ndofs) { ndofs_ = ndofs; }

      //! Sets the number of all dofs on faces of the element
      void set_onf_dofs(int onfdofs)
      {
        onfdofs_old_ = onfdofs_;
        onfdofs_ = onfdofs;
      }

      //! Set completepol_ variable
      void set_complete_polynomial_space(bool completepol) { completepol_ = completepol; }

      //! Returns the degree of the element
      int uses_complete_polynomial_space() const { return completepol_; }

      //! Sets bool to false if degree of element changes after p-adaption
      void set_padapt_ele(bool adapt)
      {
        padpatele_ = adapt;
        return;
      }

      //! Returns bool if degree of element changes after p-adaption (true if degree changes)
      bool padapt_ele() const { return padpatele_; }

      //! Sets bool if element matrices are initialized
      void set_mat_init(bool matinit)
      {
        matinit_ = matinit;
        return;
      }

      //! Returns bool if element matrices are initialized
      bool mat_init() const { return matinit_; }

      //! Returns the degree of the element for the interior DG space
      int num_dof_per_element_auxiliary() const
      {
        return (Core::FE::get_dimension(distype_) + 1) *
               Core::FE::get_basis_size(distype_, degree_, completepol_);
      }

      //! Get vector of std::shared_ptrs to the lines of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      //! Get vector of std::shared_ptrs to the surfaces of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      //! Get std::shared_ptr to the internal face adjacent to this element as master element and
      //! the parent_slave element
      std::shared_ptr<Core::Elements::Element> create_face_element(
          Core::Elements::Element* parent_slave,  //!< parent slave fluid3 element
          int nnode,                              //!< number of surface nodes
          const int* nodeids,                     //!< node ids of surface element
          Core::Nodes::Node** nodes,              //!< nodes of surface element
          const int lsurface_master,  //!< local surface number w.r.t master parent element
          const int lsurface_slave,   //!< local surface number w.r.t slave parent element
          const std::vector<int>& localtrafomap  //! local trafo map
          ) override;
      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element, that is, call the element routines to evaluate scatra
      element matrices and vectors or evaluate errors, statistics or updates etc. directly.

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      //@}

      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1) override
      {
        FOUR_C_THROW("Volume Neumann not implemented");
        return 0;
      }

      //! Print this element
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return ScaTraHDGType::instance();
      }

      // element matrices are stored to save calculation time, since they stay the same in the pure
      // diffusion reaction problem If necessary or worthwhile this can be changed and thus the
      // element matrices are calculated on the fly
      //! @name element matrices
      //!@{
      Core::LinAlg::SerialDenseMatrix Amat_;  //!< concentrations - concentrations
      Core::LinAlg::SerialDenseMatrix Bmat_;  //!< concentrations - concentrations gradients
      Core::LinAlg::SerialDenseMatrix Cmat_;  //!< concentration - trace
      Core::LinAlg::SerialDenseMatrix
          Dmat_;  //!< concentrations gradients - concentrations gradients
      Core::LinAlg::SerialDenseMatrix Emat_;    //!< trace - concentrations gradients
      Core::LinAlg::SerialDenseMatrix Gmat_;    //!< concentrations gradients
      Core::LinAlg::SerialDenseMatrix Hmat_;    //!< trace -trace
      Core::LinAlg::SerialDenseMatrix Mmat_;    //!< mass matrix (concentrations - concentrations)
      Core::LinAlg::SerialDenseMatrix EmatT_;   //!< trace - concentrations gradients (E^T)
      Core::LinAlg::SerialDenseMatrix BmatMT_;  //!< concentrations gradients- concentrations (-B^T)

      Core::LinAlg::SerialDenseMatrix Kmat_;      //!< condensed matrix
      Core::LinAlg::SerialDenseMatrix invAMmat_;  //!< inverse of [A + (1/(dt*theta))*M]
      //!@}

      //!@{ auxiliary stuff to store
      Core::LinAlg::SerialDenseMatrix BTAMmat_;
      Core::LinAlg::SerialDenseMatrix invCondmat_;
      Core::LinAlg::SerialDenseMatrix Xmat_;
      Core::LinAlg::SerialDenseVector Ivecnp_;
      Core::LinAlg::SerialDenseVector Ivecn_;
      Core::LinAlg::SerialDenseMatrix Imatnpderiv_;
      //!@}

      //! diffusion tensor
      Core::LinAlg::SerialDenseMatrix diff_;
      //! inverse diffusion tensor
      std::vector<Core::LinAlg::SerialDenseMatrix> invdiff_;
      //! main diffusivity
      double diff1_;


      //! stores the number of all interior dofs of the element
      unsigned int ndofs_;

      //! stores the number of all dofs for all faces
      unsigned int onfdofs_;

      //! stores the number of all old dofs for all faces
      unsigned int onfdofs_old_;


     private:
      //! don't want = operator
      ScaTraHDG& operator=(const ScaTraHDG& old);

      //! stores the degree of the element
      int degree_;

      //! stores the degree of the element
      int degree_old_;

      //! stores the polynomial type (tensor product or complete polynomial)
      bool completepol_;

      //! stores if element degree changes after p-adaption
      bool padpatele_;

      //! stores if element matrices are initialized
      bool matinit_;

    };  // class ScaTraHDG


    /*!
    \brief An element representing a boundary element of an scatrahdg element

    \note This is a pure Neumann boundary condition element. It's only
          purpose is to evaluate surface Neumann boundary conditions that might be
          adjacent to a parent scatrahdg element. It therefore does not implement
          the Core::Elements::Element::Evaluate method and does not have its own ElementRegister
    class.

    */

    class ScaTraHDGBoundaryType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "ScaTraHDGBoundaryType"; }

      static ScaTraHDGBoundaryType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static ScaTraHDGBoundaryType instance_;
    };

    // class ScaTraHDGBoundary

    class ScaTraHDGBoundary : public Core::Elements::FaceElement
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
      \param parent: The parent acou element of this surface
      \param lsurface: the local surface number of this surface w.r.t. the parent element
      */
      ScaTraHDGBoundary(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Core::Elements::Element* parent, const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      ScaTraHDGBoundary(const ScaTraHDGBoundary& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override;

      /*!
      \brief Return number of lines of this element
      */
      int num_line() const override { return Core::FE::get_number_of_element_lines(shape()); }

      /*!
      \brief Return number of surfaces of this element
      */
      int num_surface() const override { return Core::FE::get_number_of_element_surfaces(shape()); }

      /*!
      \brief Get vector of std::shared_ptrs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /*!
      \brief Get vector of std::shared_ptrs to the surfaces of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int unique_par_object_id() const override
      {
        return ScaTraHDGBoundaryType::instance().unique_par_object_id();
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

      int num_dof_per_element() const override { return 0; }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return ScaTraHDGBoundaryType::instance();
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

      //! @name Evaluate methods

      /*!
      \brief Evaluate Neumann boundary condition

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): reference to the underlying discretization
      \param condition (in)     : condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

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
      \param condstring (in): Name of condition to be evaluated
      \param condstring (in):  List of parameters for use at element level
      */
      void location_vector(const Core::FE::Discretization& dis, Core::Elements::LocationArray& la,
          const std::string& condstring, Teuchos::ParameterList& params) const override;

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

      \note The degrees of freedom returned are not necessarily only nodal dofs.
            Depending on the element implementation, output might also include
            element dofs.

      \param dis (in)      : the discretization this element belongs to
      \param lm (out)      : vector of degrees of freedom addressed by this element
      \param lmowner (out) : vector of proc numbers indicating which dofs are owned
                             by which procs in a dof row map. Ordering
                             matches dofs in lm.

      */
      void location_vector(const Core::FE::Discretization& dis, std::vector<int>& lm,
          std::vector<int>& lmowner, std::vector<int>& lmstride) const override;

     private:
      // don't want = operator
      ScaTraHDGBoundary& operator=(const ScaTraHDGBoundary& old);

    };  // class ScaTraHDGBoundary


    /*!
    \brief An element representing an internal face element between two ScaTraHDG elements
    */
    class ScaTraHDGIntFaceType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "ScaTraHDGIntFaceType"; }

      static ScaTraHDGIntFaceType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static ScaTraHDGIntFaceType instance_;
    };


    // class ScaTraHDGIntFace

    class ScaTraHDGIntFace : public Core::Elements::FaceElement
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
      \param master_parent: The master parent ScaTraHDG element of this surface
      \param slave_parent: The slave parent ScaTraHDG element of this surface
      \param lsurface_master: the local surface number of this surface w.r.t. the master parent
      element \param lsurface_slave: the local surface number of this surface w.r.t. the slave
      parent element \param localtrafomap: transformation map between the local coordinate systems
      of the face w.r.t the master parent element's face's coordinate system and the slave element's
      face's coordinate system
      */
      ScaTraHDGIntFace(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Discret::Elements::ScaTraHDG* parent_master, Discret::Elements::ScaTraHDG* parent_slave,
          const int lsurface_master, const int lsurface_slave,
          const std::vector<int> localtrafomap);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      ScaTraHDGIntFace(const ScaTraHDGIntFace& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override;

      /*!
      \brief Return number of lines of this element
      */
      int num_line() const override { return Core::FE::get_number_of_element_lines(shape()); }

      /*!
      \brief Return number of surfaces of this element
      */
      int num_surface() const override { return Core::FE::get_number_of_element_surfaces(shape()); }

      /*!
      \brief Get vector of std::shared_ptrs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int unique_par_object_id() const override
      {
        return ScaTraHDGIntFaceType::instance().unique_par_object_id();
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
        return std::max(parent_master_element()->num_dof_per_node(node),
            parent_slave_element()->num_dof_per_node(node));
      }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual Core::Elements::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int num_dof_per_element() const override { return 0; }

      /*!
       \brief Returns the degree of the element
       */
      int degree() const override { return degree_; }

      /*!
       \brief Returns the old degree of the element
       */
      virtual int degree_old() const { return degree_old_; }

      void set_degree(int degree)
      {
        degree_old_ = degree_;
        degree_ = degree;
        return;
      }

      /*!
      \brief create the location vector for patch of master and slave element

      \note All dofs shared by master and slave element are contained only once. Dofs from interface
      nodes are also included.
      */
      void patch_location_vector(Core::FE::Discretization& discretization,  //!< discretization
          std::vector<int>& nds_master,        //!< nodal dofset w.r.t master parent element
          std::vector<int>& nds_slave,         //!< nodal dofset w.r.t slave parent element
          std::vector<int>& patchlm,           //!< local map for gdof ids for patch of elements
          std::vector<int>& master_lm,         //!< local map for gdof ids for master element
          std::vector<int>& slave_lm,          //!< local map for gdof ids for slave element
          std::vector<int>& face_lm,           //!< local map for gdof ids for face element
          std::vector<int>& lm_masterToPatch,  //!< local map between lm_master and lm_patch
          std::vector<int>& lm_slaveToPatch,   //!< local map between lm_slave and lm_patch
          std::vector<int>& lm_faceToPatch,    //!< local map between lm_face and lm_patch
          std::vector<int>&
              lm_masterNodeToPatch,  //!< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch  //!< local map between slave nodes and nodes in patch
      );

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return ScaTraHDGIntFaceType::instance();
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

      //! @name Evaluate methods

      /*!
      \brief Evaluate Neumann boundary condition

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): reference to the underlying discretization
      \param condition (in)     : condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      /*!
      \brief return the master parent ScaTraHDG element
      */
      Discret::Elements::ScaTraHDG* parent_master_element() const
      {
        Core::Elements::Element* parent =
            this->Core::Elements::FaceElement::parent_master_element();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<Discret::Elements::ScaTraHDG*>(parent) != nullptr,
            "Master element is no ScaTraHDG element");
        return static_cast<Discret::Elements::ScaTraHDG*>(parent);
      }

      /*!
      \brief return the slave parent ScaTraHDG element
      */
      Discret::Elements::ScaTraHDG* parent_slave_element() const
      {
        Core::Elements::Element* parent = this->Core::Elements::FaceElement::parent_slave_element();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<Discret::Elements::ScaTraHDG*>(parent) != nullptr,
            "Slave element is no ScaTraHDG element");
        return static_cast<Discret::Elements::ScaTraHDG*>(parent);
      }

      //@}

     private:
      // don't want = operator
      ScaTraHDGIntFace& operator=(const ScaTraHDGIntFace& old);

      // degree of this face element
      int degree_;

      // old degree of this face element
      int degree_old_;

    };  // class ScaTraHDGIntFace



  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
