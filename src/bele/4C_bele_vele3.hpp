// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BELE_VELE3_HPP
#define FOUR_C_BELE_VELE3_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_vector.hpp"

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
    class Vele3Type : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Vele3Type"; }

      static Vele3Type& instance();

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
      static Vele3Type instance_;
    };

    /*!
     * A 3D volume element with no physics attached
     *
     * This element is meant to have no physics. It can be used to have a volume discretization
     * of volume elements. They can be of any 3d shape
     *
     * The number of dof per node is set to 3, so we can define displacement vectors by
     * using fill_complete on the discretization.
     *
     */


    /*!
    \brief A register for bele3 element

    */
    class Vele3 : public Core::Elements::Element
    {
     public:
      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor
      */
      explicit Vele3(int id,  ///< A unique global id
          int owner           ///< proc num that owns this element
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      explicit Vele3(const Vele3& old);


      /*!
      \brief Deep copy this instance of Fluid and return pointer to the copy

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
      int num_line() const override
      {
        if (num_node() == 8 || num_node() == 20 || num_node() == 27)
          return 12;
        else if (num_node() == 4 || num_node() == 10)
          return 6;
        else
        {
          FOUR_C_THROW("Could not determine number of lines");
          return -1;
        }
      }

      /*!
      \brief Return number of surfaces of this element
      */
      int num_surface() const override
      {
        if (num_node() == 8 || num_node() == 20 || num_node() == 27)
          return 6;
        else if (num_node() == 4 || num_node() == 10)
          return 4;
        else if (num_node() == 6 || num_node() == 15 || num_node() == 5)
          return 5;
        else
        {
          FOUR_C_THROW("Could not determine number of surfaces");
          return -1;
        }
      }

      /*!
      \brief Return number of volumes of this element (always 1)
      */
      int num_volume() const override { return 1; }

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
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return Vele3Type::instance().unique_par_object_id();
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
      int num_dof_per_node(const Core::Nodes::Node& node) const override { return 3; }

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

      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override { return Vele3Type::instance(); }


      //@}

      //@}

      //! @name Evaluation
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;


      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      /// Read input for this element
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;

     private:
      //! action parameters recognized by bele3
      enum ActionType
      {
        none
      };


      /*!
       * \brief check, whether higher order derivatives for shape functions (dxdx, dxdy, ...) are
       * necessary \return boolean indicating higher order status
       */
      bool is_higher_order_element(const Core::FE::CellType distype) const
      {
        bool hoel = true;
        switch (distype)
        {
          case Core::FE::CellType::hex8:
          case Core::FE::CellType::hex20:
          case Core::FE::CellType::hex27:
          case Core::FE::CellType::tet10:
          case Core::FE::CellType::wedge15:
          case Core::FE::CellType::nurbs8:
          case Core::FE::CellType::nurbs27:
            hoel = true;
            break;
          case Core::FE::CellType::tet4:
          case Core::FE::CellType::wedge6:
          case Core::FE::CellType::pyramid5:  //!!!TODO:  wedge und pyramid have 2nd
                                              //! derivatives!!!!!!!!!!!!!!!!!!!!!!!!
            hoel = false;
            break;
          default:
            FOUR_C_THROW("distype unknown!");
        }
        return hoel;
      };


      //! don't want = operator
      Vele3& operator=(const Vele3& old);

      //! set number of gauss points to element shape default
      Core::FE::GaussRule3D get_optimal_gaussrule(const Core::FE::CellType& distype) const;

    };  // class Bele3



    class Vele3SurfaceType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Vele3SurfaceType"; }

      static Vele3SurfaceType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented for element type vele3!");
        return nullspace;
      }

     private:
      static Vele3SurfaceType instance_;
    };

    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    class Vele3Surface : public Core::Elements::FaceElement
    {
     public:
      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor
      */
      explicit Vele3Surface(int id, int owner, int nnode, const int* nodeids,
          Core::Nodes::Node** nodes, Discret::Elements::Vele3* parent, const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      explicit Vele3Surface(const Vele3Surface& old);

      Core::Elements::Element* clone() const override;
      Core::FE::CellType shape() const override;
      int num_line() const override
      {
        if (num_node() == 9 || num_node() == 8 || num_node() == 4)
          return 4;
        else if (num_node() == 3 || num_node() == 6)
          return 3;
        else
        {
          FOUR_C_THROW("Could not determine number of lines");
          return -1;
        }
      }
      int num_surface() const override { return 1; }
      int num_volume() const override { return -1; }
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;
      int unique_par_object_id() const override
      {
        return Vele3SurfaceType::instance().unique_par_object_id();
      }
      void pack(Core::Communication::PackBuffer& data) const override;
      void unpack(Core::Communication::UnpackBuffer& buffer) override;


      //@}

      //! @name Access methods

      int num_dof_per_node(const Core::Nodes::Node&) const override { return 3; }
      int num_dof_per_element() const override { return 0; }
      void print(std::ostream& os) const override;
      Core::Elements::ElementType& element_type() const override
      {
        return Vele3SurfaceType::instance();
      }

      //@}

      //@}

      //! @name Evaluation
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;


      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;


     private:
      //! action parameters recognized by bele3
      enum ActionType
      {
        none
      };

      /*!
       * \brief check, whether higher order derivatives for shape functions (dxdx, dxdy, ...) are
       * necessary \return boolean indicating higher order status
       */
      bool is_higher_order_element(const Core::FE::CellType distype) const
      {
        bool hoel = true;
        switch (distype)
        {
          case Core::FE::CellType::quad4:
          case Core::FE::CellType::quad8:
          case Core::FE::CellType::quad9:
          case Core::FE::CellType::tri6:
            hoel = true;
            break;
          case Core::FE::CellType::tri3:
            hoel = false;
            break;
          default:
            FOUR_C_THROW("distype unknown!");
        }
        return hoel;
      };


      //! don't want = operator
      Vele3Surface& operator=(const Vele3Surface& old);

      //! set number of gauss points to element shape default
      Core::FE::GaussRule2D get_optimal_gaussrule(const Core::FE::CellType& distype) const;

    };  // class Vele3Surface


    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================


    class Vele3LineType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Vele3LineType"; }

      static Vele3LineType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented for element type vele3!");
        return nullspace;
      }

     private:
      static Vele3LineType instance_;
    };


    /*!
    \brief An element representing a line of a vele3 element

    */
    class Vele3Line : public Core::Elements::FaceElement
    {
     public:
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner: Processor owning this line
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param parent: The parent fluid element of this line
      \param lline: the local line number of this line w.r.t. the parent element
      */
      Vele3Line(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Core::Elements::Element* parent, const int lline);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Vele3Line(const Vele3Line& old);

      Core::Elements::Element* clone() const override;
      Core::FE::CellType shape() const override;
      int unique_par_object_id() const override
      {
        return Vele3LineType::instance().unique_par_object_id();
      }
      void pack(Core::Communication::PackBuffer& data) const override;
      void unpack(Core::Communication::UnpackBuffer& buffer) override;


      //@}

      //! @name Access methods


      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual Core::Elements::Element)

      For this 3D boundary element, we have 3 displacements, if needed
      */
      int num_dof_per_node(const Core::Nodes::Node&) const override { return 3; }

      int num_dof_per_element() const override { return 0; }

      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return Vele3LineType::instance();
      }

      //@}


      //! @name Evaluation
      int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;


      //! @name Evaluate methods
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

     private:
      //! action parameters recognized by Vele3Line
      enum ActionType
      {
        none
      };

      //! don't want = operator
      Vele3Line& operator=(const Vele3Line& old);

      //! Get Rule for Gaussintegration according to Discret::UTIL
      Core::FE::GaussRule1D get_optimal_gaussrule(const Core::FE::CellType& distype);

    };  // class Vele3Line


  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
