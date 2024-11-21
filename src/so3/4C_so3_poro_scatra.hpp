// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_PORO_SCATRA_HPP
#define FOUR_C_SO3_PORO_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_inpar_scatra.hpp"
#include "4C_so3_poro.hpp"

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
    \brief A C++ version of a 3 dimensional solid element with modifications for porous media
    including scatra functionality

    A structural 3 dimensional solid displacement element for large deformations
    and (near)-incompressibility.

    */
    template <class So3Ele, Core::FE::CellType distype>
    class So3PoroScatra : public So3Poro<So3Ele, distype>
    {
      using my = So3Poro<So3Ele, distype>;
      using my::numnod_;

     public:
      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      So3PoroScatra(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      So3PoroScatra(const So3PoroScatra& old);

      //@}

      //! @name Acess methods

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


      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override;

      //@}

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

      void pre_evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          Core::FE::Discretization& discretization,  //!< pointer to discretization for de-assembly
          Core::Elements::LocationArray& la          //!< location array for de-assembly
          ) override;

      //!@}

      /*!
       * @brief Evaluate Cauchy stress at given point in parameter space and calculate
       * linearizations
       *
       * @param[in] xi                   position in parameter space xi
       * @param[in] disp_nodal_values    vector containing nodal values of displacements
       * @param[in] scalar_nodal_values  vector containing nodal values of scalars
       * @param[in] n                    vector (\f[\mathbf{n}\f])
       * @param[in] dir                  direction vector (\f[\mathbf{v}\f]), can be either normal
       * or tangential vector
       * @param[out] cauchy_n_dir  cauchy stress tensor contracted using the vectors n and dir
       *                           (\f[ \mathbf{\sigma} \cdot \mathbf{n} \cdot \mathbf{v} \f])
       * @param[out] d_cauchyndir_dd  derivative of cauchy_n_dir w.r.t. displacements
       *                         (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n} \cdot
       * \mathbf{v}} { \mathrm{d} \mathbf{d}} \f])
       * @param d_cauchyndir_ds  derivative of cauchy_n_dir w.r.t. vector of scalars s
       *                        (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n} \cdot
       * \mathbf{v}} { \mathrm{d} \mathbf{s}} \f])
       * @param[out] d_cauchyndir_dn   derivative of cauchy_n_dir w.r.t. vector n
       *                        (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n} \cdot
       * \mathbf{v}} { \mathrm{d} \mathbf{n}} \f])
       * @param[out] d_cauchyndir_ddir  derivative of cauchy_n_dir w.r.t. direction vector v
       *                        (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n} \cdot
       * \mathbf{v}} { \mathrm{d} \mathbf{v}} \f])
       * @param[out] d_cauchyndir_dxi  derivative of cauchy_n_dir w.r.t. local parameter coord. xi
       *                        (\f[ \frac{ \mathrm{d} \mathbf{\sigma} \cdot \mathbf{n} \cdot
       * \mathbf{v}} { \mathrm{d} \mathbf{\xi}} \f])
       *
       * @note At the moment this method is only used for the nitsche contact formulation
       */
      void get_cauchy_n_dir_and_derivatives_at_xi(const Core::LinAlg::Matrix<3, 1>& xi,
          const std::vector<double>& disp_nodal_values,
          const std::vector<double>& pres_nodal_values,
          const std::vector<double>& scalar_nodal_values, const Core::LinAlg::Matrix<3, 1>& n,
          const Core::LinAlg::Matrix<3, 1>& dir, double& cauchy_n_dir,
          Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dd,
          Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dp,
          Core::LinAlg::SerialDenseMatrix* d_cauchyndir_ds,
          Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
          Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir,
          Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dxi);

      //! @name Input and Creation
      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& eledistype,
          const Core::IO::InputParameterContainer& container) override;

      /// @name params
      /// return ScaTra::ImplType
      const Inpar::ScaTra::ImplType& impl_type() const { return impltype_; };

     private:
      //! scalar transport implementation type (physics)
      Inpar::ScaTra::ImplType impltype_;
      //@}

     protected:
      //! don't want = operator
      So3PoroScatra& operator=(const So3PoroScatra& old);

    };  // class So3_Poro_Scatra
  }     // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
