// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_PORO_P1_HPP
#define FOUR_C_W1_PORO_P1_HPP

#include "4C_config.hpp"

#include "4C_w1_poro.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    /*!
    \brief A C++ version of a 2 dimensional solid element with modifications for porous media

    */
    template <Core::FE::CellType distype>
    class Wall1PoroP1 : public Wall1Poro<distype>
    {
      using Base = Discret::Elements::Wall1Poro<distype>;

     public:
      //!@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      Wall1PoroP1(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Wall1PoroP1(const Wall1PoroP1& old);

      //!@}

      //! number of dofs per node
      static constexpr int noddof_ = Base::noddof_ + 1;

      //! total dofs per element
      static constexpr int numdof_ = noddof_ * Base::numnod_;

      //! @name Access methods

      /*!
      \brief Deep copy this instance of Solid3 and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override;

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

      //! Get vector of std::shared_ptrs to the lines of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      //! Get vector of std::shared_ptrs to the surfaces of this element
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override;

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
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override;

      //!@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      Evaluate element stiffness, mass, internal forces, etc.

      If nullptr on input, the controlling method does not expect the element
      to fill these matrices or vectors.

      \return 0 if successful, negative otherwise
      */
      int evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          Core::FE::Discretization& discretization,  //!< pointer to discretization for de-assembly
          Core::Elements::LocationArray& la,         //!< location array for de-assembly
          Core::LinAlg::SerialDenseMatrix&
              elemat1,  //!< (stiffness-)matrix to be filled by element.
          Core::LinAlg::SerialDenseMatrix& elemat2,  //!< (mass-)matrix to be filled by element.
          Core::LinAlg::SerialDenseVector&
              elevec1,  //!< (internal force-)vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec2,  //!< vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec3   //!< vector to be filled by element
          ) override;

      //! Evaluate a Neumann boundary condition
      //!
      //! this method evaluates a surfaces Neumann condition on the wall element
      //!
      //! \return 0 if successful, negative otherwise
      int evaluate_neumann(
          Teuchos::ParameterList& params,  //!< (in/out) ParameterList for communication between
                                           //!< control routine and elements
          Core::FE::Discretization&
              discretization,  //!< A reference to the underlying discretization
          const Core::Conditions::Condition& condition,  //!<  The condition to be evaluated
          std::vector<int>& lm,                          //!< location vector of this element
          Core::LinAlg::SerialDenseVector& elevec1,      //!< vector to be filled by element.
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      //!@}

      //! initialize the inverse of the jacobian and its determinant in the material configuration
      void init_element() override;

      //! don't want = operator
      Wall1PoroP1& operator=(const Wall1PoroP1& old) = delete;

     protected:
      /*!
      \brief Evaluate an element

      Evaluate So3_poro element stiffness, mass, internal forces, etc.
      Templated evaluate routine of element matrixes

      If nullptr on input, the controlling method does not expect the element
      to fill these matrices or vectors.

      \return 0 if successful, negative otherwise
      */
      int my_evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          Core::FE::Discretization& discretization,  //!< pointer to discretization for de-assembly
          Core::Elements::LocationArray& la,         //!< location array for de-assembly
          Core::LinAlg::SerialDenseMatrix&
              elemat1,  //!< (stiffness-)matrix to be filled by element.
          Core::LinAlg::SerialDenseMatrix& elemat2,  //!< (mass-)matrix to be filled by element.
          Core::LinAlg::SerialDenseVector&
              elevec1,  //!< (internal force-)vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec2,  //!< vector to be filled by element
          Core::LinAlg::SerialDenseVector& elevec3   //!< vector to be filled by element
          ) override;

      //! Calculate nonlinear stiffness and internal force for poroelasticity problems
      void nonlinear_stiffness_poroelast(std::vector<int>& lm,       //!< location matrix
          Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& disp,  //!< current displacements
          Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& vel,   //!< current velocities
          Core::LinAlg::Matrix<Base::numnod_, 1>* porosity_dof,      //!< porosity at element nodes
          Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>&
              evelnp,                                           //!< fluid velocity of element
          Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,       //!< fluid pressure of element
          Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix,    //!< element reactive matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,              //!< element internal force vector
          Teuchos::ParameterList& params  //!< algorithmic parameters e.g. time
      );

      //! Calculate coupling terms in nonlinear stiffness and internal force for poroelasticity
      //! problems
      void coupling_poroelast(std::vector<int>& lm,                  //!< location matrix
          Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& disp,  //!< current displacements
          Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& vel,   //!< current velocities
          Core::LinAlg::Matrix<Base::numnod_, 1>* porosity,          //!< porosity value
          Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>&
              evelnp,                                      //!< fluid velocity of element
          Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,  //!< fluid pressure of element
          Core::LinAlg::Matrix<numdof_, (Base::numdim_ + 1) * Base::numnod_>*
              stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::Matrix<numdof_, (Base::numdim_ + 1) * Base::numnod_>*
              reamatrix,                            //!< element reactive matrix
          Core::LinAlg::Matrix<numdof_, 1>* force,  //!< element internal force vector
          Teuchos::ParameterList& params);          //!< algorithmic parameters e.g. time

      //! compute porosity at gausspoint and linearization of porosity w.r.t. structural
      //! displacements
      void compute_porosity_and_linearization(const Core::LinAlg::Matrix<Base::numnod_, 1>& shapfct,
          const Core::LinAlg::Matrix<Base::numnod_, 1>* myporosity, double& porosity,
          Core::LinAlg::Matrix<1, Base::numdof_>& dphi_dus);


      //! compute porosity at gausspoint and linearization of porosity w.r.t. fluid pressure
      void compute_porosity_and_linearization_od(Teuchos::ParameterList& params,
          const double& press, const double& J, const int& gp,
          const Core::LinAlg::Matrix<Base::numnod_, 1>& shapfct,
          const Core::LinAlg::Matrix<Base::numnod_, 1>* myporosity, double& porosity,
          double& dphi_dp) override;

      //! evaluate gauss points for diagonal terms
      void gauss_point_loop_p1(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xrefe,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xcurr,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodaldisp,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodalvel,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& evelnp,
          const Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,
          const Core::LinAlg::Matrix<Base::numnod_, 1>* porosity_dof,
          Core::LinAlg::Matrix<Base::numdof_, Base::numdof_>& erea_v,
          Core::LinAlg::Matrix<Base::numdof_, Base::numdof_>* sub_stiff,
          Core::LinAlg::Matrix<Base::numdof_, 1>* sub_force,
          Core::LinAlg::Matrix<Base::numdof_, Base::numnod_>& ecoupl_p1,
          Core::LinAlg::Matrix<Base::numnod_, numdof_>& estiff_p1,
          Core::LinAlg::Matrix<Base::numnod_, 1>& ecoupl_force_p1);

      //! evaluate gauss points for off-diagonal terms
      void gauss_point_loop_p1_od(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xrefe,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xcurr,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodaldisp,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodalvel,
          const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& evelnp,
          const Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,
          const Core::LinAlg::Matrix<Base::numnod_, 1>* porosity_dof,
          Core::LinAlg::Matrix<Base::numnod_, Base::numnod_>& ecoupl_p1,
          Core::LinAlg::Matrix<Base::numdof_, (Base::numdim_ + 1) * Base::numnod_>& sub_stiff);
    };
  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
