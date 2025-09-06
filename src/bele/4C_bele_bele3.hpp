// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BELE_BELE3_HPP
#define FOUR_C_BELE_BELE3_HPP


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
    class Bele3Type : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Bele3Type"; }

      static Bele3Type& instance();

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
      static Bele3Type instance_;
    };

    /*!
     * A 3D boundary element with no physics attached
     *
     * This element is meant to have no physics. It can be used to have a boundary discretization
     * of surface/boundary elements. They can be of any 2d shape (quad4,quad9,tri3,...)
     *
     * The number of dof per node is set to 3 per default, so we can define displacement vectors by
     * using fill_complete on the boundary discretization. Furthermore numdofpernode can be adapted
     * if necessary.
     *
     */
    class Bele3 : public Core::Elements::Element
    {
      // friend class to fill number of dofs per node exclusively during creation
      friend class Bele3Type;

     public:
      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor
      */
      explicit Bele3(int id,  ///< A unique global id
          int owner           ///< proc num that owns this element
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      explicit Bele3(const Bele3& old);

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
        return Bele3Type::instance().unique_par_object_id();
      }
      void pack(Core::Communication::PackBuffer& data) const override;
      void unpack(Core::Communication::UnpackBuffer& buffer) override;


      //@}

      //! @name Access methods

      int num_dof_per_node(const Core::Nodes::Node&) const override { return numdofpernode_; }
      int num_dof_per_element() const override { return 0; }
      void print(std::ostream& os) const override;
      Core::Elements::ElementType& element_type() const override { return Bele3Type::instance(); }

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
      //@}

      //! @name Other
      //! does this element have non-zero displacements or not
      //  bool IsMoving() const { return is_moving_; }

      //@}


     private:
      /*!
        \brief Set number of dofs

        \param numdofpernode: number of degrees of freedom for one node
       */
      virtual void set_num_dof_per_node(int numdofpernode) { numdofpernode_ = numdofpernode; }

      int numdofpernode_;  ///< number of degrees of freedom

      //! action parameters recognized by bele3
      enum ActionType
      {
        none,
        calc_struct_constrvol,
        calc_struct_volconstrstiff,
        calc_struct_stress
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
            break;
        }
        return hoel;
      };

      /*!
        \brief Create matrix with spatial configuration

        \param x     (out)  : nodal coords in spatial frame
        \param disp  (int)  : displacements
      */
      inline void spatial_configuration(
          Core::LinAlg::SerialDenseMatrix& x, const std::vector<double> disp) const
      {
        const int numnode = num_node();
        for (int i = 0; i < numnode; ++i)
        {
          x(i, 0) = nodes()[i]->x()[0] + disp[i * 3 + 0];
          x(i, 1) = nodes()[i]->x()[1] + disp[i * 3 + 1];
          x(i, 2) = nodes()[i]->x()[2] + disp[i * 3 + 2];
        }
        return;
      }

      //! Submethod to compute the enclosed volume for volume constraint boundary condition
      double compute_constr_vols(
          const Core::LinAlg::SerialDenseMatrix& xc,  ///< current configuration
          const int numnode                           ///< num nodes
      );

      //! Submethod to compute constraint volume and its first and second derivatives w.r.t. the
      //! displacements
      void compute_vol_deriv(const Core::LinAlg::SerialDenseMatrix& x,  ///< spatial configuration
          const int numnode,                                            ///< number of nodes
          const int ndof,                          ///< number of degrees of freedom
          double& V,                               ///< volume
          Core::LinAlg::SerialDenseVector& Vdiff,  ///< first derivative
          std::shared_ptr<Core::LinAlg::SerialDenseMatrix> Vdiff2,  ///< second derivative
          const int minind = 0,  ///< minimal index to compute enclosed volume with
          const int maxind = 2   ///< maximal index to compute enclosed volume with
      );

      //! vector with line elements
      //  std::vector<std::shared_ptr<Core::Elements::Element> >                      lines_;

      //! flag for fixed or moving boundary
      //  const bool                                      is_moving_;

      //! don't want = operator
      Bele3& operator=(const Bele3& old);

      //! set number of gauss points to element shape default
      Core::FE::GaussRule2D get_optimal_gaussrule() const;

    };  // class Bele3



    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================

    class Bele3LineType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Bele3LineType"; }

      static Bele3LineType& instance();

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override
      {
      }

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        Core::LinAlg::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented for element type bele3!");
        return nullspace;
      }

     private:
      static Bele3LineType instance_;
    };


    /*!
    \brief An element representing a line of a bele3 element

    */
    class Bele3Line : public Core::Elements::FaceElement
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
      Bele3Line(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
          Discret::Elements::Bele3* parent, const int lline);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Bele3Line(const Bele3Line& old);

      Core::Elements::Element* clone() const override;
      Core::FE::CellType shape() const override;
      int unique_par_object_id() const override
      {
        return Bele3LineType::instance().unique_par_object_id();
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
      int num_dof_per_node(const Core::Nodes::Node&) const override { return numdofpernode_; }

      int num_dof_per_element() const override { return 0; }

      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        return Bele3LineType::instance();
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
      /*!
        \brief Set number of dofs

        \param numdofpernode: number of degrees of freedom for one node
       */
      virtual void set_num_dof_per_node(int numdofpernode) { numdofpernode_ = numdofpernode; }

      int numdofpernode_;  ///< number of degrees of freedom

      //! action parameters recognized by Bele3Line
      enum ActionType
      {
        none,
        integrate_Shapefunction
      };

      //! don't want = operator
      Bele3Line& operator=(const Bele3Line& old);
    };  // class Bele3Line

  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
