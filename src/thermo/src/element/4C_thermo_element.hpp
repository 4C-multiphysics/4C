// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_ELEMENT_HPP
#define FOUR_C_THERMO_ELEMENT_HPP

/*----------------------------------------------------------------------*
 | headers                                                    gjb 01/08 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Thermo
{
  // forward declarations
  class FaceElement;

  class ElementType : public Core::Elements::ElementType
  {
   public:
    std::string name() const override { return "ElementType"; }

    static ElementType& instance();

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

    std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
        const std::string eledistype, const int id, const int owner) override;

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    void nodal_block_information(
        Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

    void setup_element_definition(
        std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
        override;

   private:
    static ElementType instance_;

  };  // class: ElementType

  //!
  //! \brief A C++ wrapper for the thermo element
  //!
  class Element : public Core::Elements::Element
  {
   public:
    //! @name Friends
    friend class FaceElement;

    //@}
    //! @name Constructors and destructors and related methods

    //! \brief Standard Constructor
    Element(int id,  ///< A unique global id
        int owner    ///< processor id who owns a certain instance of this class
    );

    //! \brief Copy Constructor
    //!
    //! Makes a deep copy of a Element
    Element(const Element& old);

    //! Deleted copy assignment.
    Element& operator=(const Element& old) = delete;

    //! \brief Deep copy this instance of Thermo and return pointer to the copy
    //!
    //! The clone() method is used from the virtual base class Element in cases
    //! where the type of the derived class is unknown and a copy-ctor is needed
    Core::Elements::Element* clone() const override;

    //! \brief Get shape type of element
    Core::FE::CellType shape() const override;

    //! \brief set discretization type of element
    virtual void set_dis_type(Core::FE::CellType shape)
    {
      distype_ = shape;
      return;
    };

    //! \brief Return number of lines of this element
    int num_line() const override { return Core::FE::get_number_of_element_lines(distype_); }

    //! \brief Return number of surfaces of this element
    int num_surface() const override
    {
      switch (distype_)
      {
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::hex20:
        case Core::FE::CellType::hex27:
        case Core::FE::CellType::nurbs27:
          return 6;
          break;
        case Core::FE::CellType::tet4:
        case Core::FE::CellType::tet10:
          return 4;
          break;
        case Core::FE::CellType::wedge6:
        case Core::FE::CellType::wedge15:
        case Core::FE::CellType::pyramid5:
          return 5;
          break;
        case Core::FE::CellType::quad4:
        case Core::FE::CellType::quad8:
        case Core::FE::CellType::quad9:
        case Core::FE::CellType::nurbs4:
        case Core::FE::CellType::nurbs9:
        case Core::FE::CellType::tri3:
        case Core::FE::CellType::tri6:
          return 1;
          break;
        case Core::FE::CellType::line2:
        case Core::FE::CellType::line3:
          return 0;
          break;
        default:
          FOUR_C_THROW("discretization type not yet implemented");
          break;
      }
      return 0;
    }

    //! \brief Return number of volumes of this element
    int num_volume() const override
    {
      switch (distype_)
      {
        case Core::FE::CellType::hex8:
        case Core::FE::CellType::hex20:
        case Core::FE::CellType::hex27:
        case Core::FE::CellType::tet4:
        case Core::FE::CellType::tet10:
        case Core::FE::CellType::wedge6:
        case Core::FE::CellType::wedge15:
        case Core::FE::CellType::pyramid5:
          return 1;
          break;
        case Core::FE::CellType::quad4:
        case Core::FE::CellType::quad8:
        case Core::FE::CellType::quad9:
        case Core::FE::CellType::nurbs4:
        case Core::FE::CellType::nurbs9:
        case Core::FE::CellType::tri3:
        case Core::FE::CellType::tri6:
        case Core::FE::CellType::line2:
        case Core::FE::CellType::line3:
          return 0;
          break;
        default:
          FOUR_C_THROW("discretization type not yet implemented");
          break;
      }
      return 0;
    }

    //! \brief Get vector of std::shared_ptrs to the lines of this element
    std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

    //! \brief Get vector of std::shared_ptrs to the surfaces of this element
    std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

    //! \brief Return unique ParObject id
    //!
    //! every class implementing ParObject needs a unique id defined at the
    //! top of this file.
    int unique_par_object_id() const override
    {
      return ElementType::instance().unique_par_object_id();
    }

    //! \brief Pack this class so it can be communicated
    //! \ref pack and \ref unpack are used to communicate this element
    void pack(Core::Communication::PackBuffer& data) const override;

    //! \brief Unpack data from a char vector into this class
    //!
    //! \ref pack and \ref unpack are used to communicate this element
    void unpack(Core::Communication::UnpackBuffer& buffer) override;


    //@}

    //! @name Access methods

    //! \brief Get number of degrees of freedom of a certain node
    //!        (implements pure virtual Core::Elements::Element)
    //!
    //! The element decides how many degrees of freedom its nodes must have.
    //! As this may vary along a simulation, the element can redecide the
    //! number of degrees of freedom per node along the way for each of it's nodes
    //! separately.
    int num_dof_per_node(const Core::Nodes::Node& node) const override { return numdofpernode_; }

    //!
    //! \brief Get number of degrees of freedom per element
    //!        (implements pure virtual Core::Elements::Element)
    //!
    //! The element decides how many element degrees of freedom it has.
    //! It can redecide along the way of a simulation.
    //!
    //! \note Element degrees of freedom mentioned here are dofs that are visible
    //!       at the level of the total system of equations. Purely internal
    //!       element dofs that are condensed internally should NOT be considered.
    int num_dof_per_element() const override { return 0; }

    //! \brief Print this element
    void print(std::ostream& os) const override;

    Core::Elements::ElementType& element_type() const override { return ElementType::instance(); }

    //! \brief Query names of element data to be visualized using BINIO
    //!
    //! The element fills the provided map with key names of
    //! visualization data the element wants to visualize AT THE CENTER
    //! of the element geometry. The value is supposed to be dimension of the
    //! data to be visualized. It can either be 1 (scalar), 3 (vector), 6 (sym.
    //! tensor) or 9 (nonsym. tensor)
    //!
    //! Example:
    //! \code
    //!  // Name of data is 'Owner', dimension is 1 (scalar value)
    //!  names.insert(std::pair<std::string,int>("Owner",1));
    //!  // Name of data is 'HeatfluxXYZ', dimension is 3 (vector value)
    //!  names.insert(std::pair<std::string,int>("HeatfluxXYZ",3));
    //! \endcode
    //!
    //! \param names (out): On return, the derived class has filled names with
    //!                     key names of data it wants to visualize and with int
    //!                     dimensions of that data.
    void vis_names(std::map<std::string, int>& names) override {};

    //! \brief Query data to be visualized using BINIO of a given name
    //!
    //! The method is supposed to call this base method to visualize the owner of
    //! the element.
    //! If the derived method recognizes a supported data name, it shall fill it
    //! with corresponding data.
    //! If it does NOT recognizes the name, it shall do nothing.
    //!
    //! \warning The method must not change size of data
    //!
    //! \param name (in):   Name of data that is currently processed for visualization
    //! \param data (out):  data to be filled by element if element recognizes the name
    bool vis_data(const std::string& name, std::vector<double>& data) override;

    //@}

    //! @name Input and Creation

    //! \brief Read input for this element
    bool read_element(const std::string& eletype, const std::string& distype,
        const Core::IO::InputParameterContainer& container) override;

    //@}

    //! @name Evaluation

    //! \brief Evaluate an element, i.e. call the implementation to evaluate element
    //! tangent, capacity, internal forces or evaluate errors, statistics or updates
    //! etc. directly.
    //!
    //! Following implementations of the element are allowed:
    //!       //!  o Evaluation of thermal system matrix and residual for the One-Step-Theta
    //!
    //!  o Evaluation of thermal system matrix and residual for the stationary thermal solver
    //!       //!
    //! \param params (in/out): ParameterList for communication between control routine
    //!                         and elements
    //! \param discretization (in): A reference to the underlying discretization
    //! \param la (in)        : location array of this element
    //! \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element to fill
    //!                         this matrix.
    //! \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element to fill
    //!                         this matrix.
    //! \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element
    //!                         to fill this vector
    //! \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element
    //!                         to fill this vector
    //! \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element
    //!                         to fill this vector
    //! \return 0 if successful, negative otherwise
    int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

    //! \brief Evaluate a Neumann boundary condition
    //!
    //! this method evaluates a surfaces Neumann condition on the shell element
    //!
    //! \param params (in/out)    : ParameterList for communication between control
    //!                             routine and elements
    //! \param discretization (in): A reference to the underlying discretization
    //! \param condition (in)     : The condition to be evaluated
    //! \param lm (in)            : location vector of this element
    //! \param elevec1 (out)      : vector to be filled by element. If nullptr on input,
    //!
    //! \return 0 if successful, negative otherwise
    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1) override;

    //@}

    //! kinematic type passed from structural element
    virtual void set_kinematic_type(Inpar::Solid::KinemType kintype)
    {
      kintype_ = kintype;
      return;
    };
    //! kinematic type
    Inpar::Solid::KinemType kintype_;

    Inpar::Solid::KinemType kin_type() const { return kintype_; }

   private:
    //! number of dofs per node (for systems of thermo equations)
    //! (storage necessary because we don't know the material in the post filters anymore)
    static constexpr int numdofpernode_ = 1;
    //! the element discretization type
    Core::FE::CellType distype_;

  };  // class Thermo


  ////=======================================================================
  ////=======================================================================
  ////=======================================================================
  ////=======================================================================
  class FaceElementType : public Core::Elements::ElementType
  {
   public:
    std::string name() const override { return "FaceElementType"; }

    static FaceElementType& instance();

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    void nodal_block_information(
        Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
    {
    }

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
    {
      Core::LinAlg::SerialDenseMatrix nullspace;
      FOUR_C_THROW("method ComputeNullSpace not implemented");
      return nullspace;
    }

   private:
    static FaceElementType instance_;
  };

  //! \brief An element representing a boundary element of a thermo element
  //!
  //! \note This is a pure boundary condition element. It's only
  //!       purpose is to evaluate certain boundary conditions that might be
  //!       adjacent to a parent Thermo element.
  class FaceElement : public Core::Elements::FaceElement
  {
   public:
    //! @name Constructors and destructors and related methods

    //! \brief Standard Constructor
    //!
    //! \param id : A unique global id
    //! \param owner: Processor owning this surface
    //! \param nnode: Number of nodes attached to this element
    //! \param nodeids: global ids of nodes attached to this element
    //! \param nodes: the discretization map of nodes to build ptrs to nodes from
    //! \param parent: The parent fluid element of this surface
    //! \param lsurface: the local surface number of this surface w.r.t. the parent element
    FaceElement(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
        Element* parent, const int lsurface);

    //! \brief Copy Constructor
    //!
    //! Makes a deep copy of a Element
    FaceElement(const FaceElement& old);

    //! \brief Deep copy this instance of an element and return pointer to the copy
    //!
    //! The clone() method is used from the virtual base class Element in cases
    //! where the type of the derived class is unknown and a copy-constructor is needed
    Core::Elements::Element* clone() const override;

    //! \brief Get shape type of element
    Core::FE::CellType shape() const override;

    //! \brief Return number of lines of boundary element
    int num_line() const override
    {
      // get spatial dimension of boundary
      const int nsd = Core::FE::get_dimension(parent_element()->shape()) - 1;

      if ((num_node() == 4) or (num_node() == 8) or (num_node() == 9))
        return 4;
      else if (num_node() == 6)
        return 3;
      else if ((num_node() == 3) and (nsd == 2))
        return 3;
      else if ((num_node() == 3) and (nsd == 1))
        return 1;
      else if (num_node() == 2)
        return 1;
      else
      {
        FOUR_C_THROW("Could not determine number of lines");
        return -1;
      }
    }

    //! \brief Return number of surfaces of boundary element
    int num_surface() const override
    {
      // get spatial dimension of parent element
      const int nsd = Core::FE::get_dimension(parent_element()->shape());

      if (nsd == 3)
        return 1;
      else
        return 0;
    }

    //! \brief Get vector of std::shared_ptrs to the lines of this element
    std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

    //! \brief Get vector of std::shared_ptrs to the surfaces of this element
    std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

    //! \brief Return unique ParObject id
    //!
    //! every class implementing ParObject needs a unique id defined at the
    //! top of the parobject.H file.
    int unique_par_object_id() const override
    {
      return FaceElementType::instance().unique_par_object_id();
    }

    //! \brief Pack this class so it can be communicated
    //!
    //! \ref pack and \ref unpack are used to communicate this element
    virtual void pack(std::vector<char>& data) const;

    //! \brief Unpack data from a char vector into this class
    //!
    //! \ref pack and \ref unpack are used to communicate this element
    void unpack(Core::Communication::UnpackBuffer& buffer) override;


    //@}

    //! @name Access methods

    //! \brief Get number of degrees of freedom of a certain node
    //!       (implements pure virtual Core::Elements::Element)
    //!
    //! The element decides how many degrees of freedom its nodes must have.
    //! As this may vary along a simulation, the element can redecide the
    //! number of degrees of freedom per node along the way for each of it's nodes
    //! separately.
    int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
      return parent_element()->num_dof_per_node(node);
    }

    /*
    //! Return a pointer to the parent element of this boundary element
    virtual Thermo::Element* parent_element()
    {
      return parent_;
    }
    */

    //! \brief Get number of degrees of freedom per element
    //!       (implements pure virtual Core::Elements::Element)
    //!
    //! The element decides how many element degrees of freedom it has.
    //! It can redecide along the way of a simulation.
    //!
    //! \note Element degrees of freedom mentioned here are dofs that are visible
    //!      at the level of the total system of equations. Purely internal
    //!      element dofs that are condensed internally should NOT be considered.
    int num_dof_per_element() const override { return 0; }

    //! \brief Print this element
    void print(std::ostream& os) const override;

    Core::Elements::ElementType& element_type() const override
    {
      return FaceElementType::instance();
    }

    //@}

    //! @name Evaluation

    //! \brief Evaluate an element
    //!
    //! Evaluate Thermo element tangent, capacity, internal forces etc
    //!
    //! \param params (in/out): ParameterList for communication between control routine
    //!                         and elements
    //! \param discretization (in): A reference to the underlying discretization
    //! \param la (in):         location array of this element, vector of
    //!                         degrees of freedom addressed by this element
    //! \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element to fill
    //!                         this matrix.
    //! \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element to fill
    //!                         this matrix.
    //! \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element
    //!                         to fill this vector
    //! \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element
    //!                         to fill this vector
    //! \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
    //!                         the controlling method does not expect the element
    //!                         to fill this vector
    //! \return 0 if successful, negative otherwise
    int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

    //@}

    //! @name Evaluate methods

    //! \brief Evaluate a Neumann boundary condition
    //!
    //! this method evaluates a surface Neumann condition on the thermo element
    //!
    //! \param params (in/out)    : ParameterList for communication between control routine
    //!                             and elements
    //! \param discretization (in): A reference to the underlying discretization
    //! \param condition (in)     : The condition to be evaluated
    //! \param lm (in)            : location vector of this element
    //! \param elevec1 (out)      : vector to be filled by element. If nullptr on input,
    //!
    //! \return 0 if successful, negative otherwise
    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1) override;

    //@}

   private:
    // don't want = operator
    FaceElement& operator=(const FaceElement& old);

  };  // class FaceElement

}  // namespace Thermo


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
