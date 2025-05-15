// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_ELEMENT2_HPP
#define FOUR_C_CONSTRAINT_ELEMENT2_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
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
    // forward declarations
    // class ConstraintElementLine;

    class ConstraintElement2Type : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "ConstraintElement2Type"; }

      static ConstraintElement2Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

     private:
      static ConstraintElement2Type instance_;
    };

    /*!
     */
    class ConstraintElement2 : public Core::Elements::Element
    {
     public:
      //! @name Friends

      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      */
      ConstraintElement2(int id,  ///< A unique global id
          int owner               ///< element owner
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      ConstraintElement2(const ConstraintElement2& old);

      /*!
      \brief Deep copy this instance of ConstraintElement2 and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override
      {
        FOUR_C_THROW("ConstraintElement2 has no shape!");
        return Core::FE::CellType::dis_none;
      };

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return ConstraintElement2Type::instance().unique_par_object_id();
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
      int num_dof_per_node(const Core::Nodes::Node& node) const override { return 2; }

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

      Core::Elements::ElementType& element_type() const override
      {
        return ConstraintElement2Type::instance();
      }

      //@}

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      \return 0 if successful, negative otherwise
      */
      int evaluate(Teuchos::ParameterList& params,   ///< ParameterList for communication
          Core::FE::Discretization& discretization,  ///< discretization
          std::vector<int>& lm,                      ///< location vector
          Core::LinAlg::SerialDenseMatrix& elemat1,  ///< first matrix to be filled by element
          Core::LinAlg::SerialDenseMatrix& elemat2,  ///< second matrix to be filled by element
          Core::LinAlg::SerialDenseVector& elevec1,  ///< third matrix to be filled by element
          Core::LinAlg::SerialDenseVector& elevec2,  ///< first vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec3   ///< second vector to be filled by element
          ) override;


      /*!
      \brief Evaluate a Neumann boundary condition

      Since the element has no physics attached this method will give a FOUR_C_THROW

      \return 0 if successful, negative otherwise
      */
      int evaluate_neumann(Teuchos::ParameterList& params,  ///< ParameterList for communication
          Core::FE::Discretization& discretization,         ///< discretization
          const Core::Conditions::Condition& condition,     ///< Neumann condition to evaluate
          std::vector<int>& lm,                             ///< location vector
          Core::LinAlg::SerialDenseVector& elevec1,         ///< vector to be filled by element
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;


     private:
      //! action parameters recognized by constraint element
      enum ActionType
      {
        none,
        calc_MPC_dist_stiff,
        calc_MPC_angle_stiff
      };

      //! vector of surfaces of this element (length 1)
      std::vector<Core::Elements::Element*> surface_;

      // don't want = operator
      ConstraintElement2& operator=(const ConstraintElement2& old);


      /*!
      \brief Create matrix with material configuration for 2 dimensions and 3 nodes
      */
      inline void material_configuration(
          Core::LinAlg::Matrix<3, 2>& x  ///< nodal coords in material frame
      ) const
      {
        const int numnode = 3;
        const int numdim = 2;
        for (int i = 0; i < numnode; ++i)
        {
          for (int j = 0; j < numdim; ++j)
          {
            x(i, j) = nodes()[i]->x()[j];
          }
        }
        return;
      }

      /*!
      \brief Create matrix with spatial configuration for 2 dimensions and 3 nodes
      */
      inline void spatial_configuration(
          Core::LinAlg::Matrix<3, 2>& x,  ///< nodal coords in spatial frame
          const std::vector<double> disp  ///< displacements
      ) const
      {
        const int numnode = 3;
        const int numdim = 2;
        for (int i = 0; i < numnode; ++i)
        {
          for (int j = 0; j < numdim; ++j)
          {
            x(i, j) = nodes()[i]->x()[j] + disp[i * numdim + j];
          }
        }
        return;
      }


      /// compute normal for 2D case using the first two nodes to specify a line
      void compute_normal(const Core::LinAlg::Matrix<3, 2>& xc,  ///< nodal coords in spatial frame
          Core::LinAlg::Matrix<2, 1>& elenormal                  ///< resulting element normal
      );

      /// Compute normal distance between line and third node
      double compute_normal_dist(
          const Core::LinAlg::Matrix<3, 2>& xc,        ///< nodal coords in spatial frame
          const Core::LinAlg::Matrix<2, 1>& elenormal  ///< element normal
      );

      /// Compute first derivatives of normal distance with respect to the nodal displacements
      void compute_first_deriv_dist(
          const Core::LinAlg::Matrix<3, 2>& xc,        ///< nodal coords in spatial frame
          Core::LinAlg::SerialDenseVector& elevector,  ///< vector to store results into
          const Core::LinAlg::Matrix<2, 1>& elenormal  ///< element normal
      );

      /// Compute second derivatives of normal distance with respect to the nodal displacements
      void compute_second_deriv_dist(
          const Core::LinAlg::Matrix<3, 2>& xc,        ///< nodal coords in spatial frame
          Core::LinAlg::SerialDenseMatrix& elematrix,  ///< matrix to store results into
          const Core::LinAlg::Matrix<2, 1>& elenormal  ///< element normal
      );

      /// Compute angle at second node
      double compute_angle(const Core::LinAlg::Matrix<3, 2>& xc  ///< nodal coords in spatial frame
      );

      /// Compute first derivatives of angle at second node with respect to the nodal displacements
      void compute_first_deriv_angle(
          const Core::LinAlg::Matrix<3, 2>& xc,       ///< nodal coords in spatial frame
          Core::LinAlg::SerialDenseVector& elevector  ///< vector to store results into
      );

      /// Compute second derivatives of angle at second node with respect to the nodal displacements
      void compute_second_deriv_angle(
          const Core::LinAlg::Matrix<3, 2>& xc,       ///< nodal coords in spatial frame
          Core::LinAlg::SerialDenseMatrix& elematrix  ///< matrix to store results into
      );

    };  // class ConstraintElement2


    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================


  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
