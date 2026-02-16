// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MEMBRANE_SCATRA_HPP
#define FOUR_C_MEMBRANE_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_inpar_scatra.hpp"
#include "4C_membrane.hpp"
#include "4C_membrane_scatra_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{

  namespace Elements
  {
    template <Core::FE::CellType distype>
    class MembraneScatra : public Membrane<distype>
    {
     public:
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      MembraneScatra(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      MembraneScatra(const MembraneScatra<distype>& old);

      /*!
      \brief Deep copy this instance of Membrane and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-constructor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
          {
            return MembraneScatraTri3Type::instance().unique_par_object_id();
          }
          case Core::FE::CellType::tri6:
          {
            return MembraneScatraTri6Type::instance().unique_par_object_id();
          }
          case Core::FE::CellType::quad4:
          {
            return MembraneScatraQuad4Type::instance().unique_par_object_id();
          }
          case Core::FE::CellType::quad9:
          {
            return MembraneScatraQuad9Type::instance().unique_par_object_id();
          }
          default:
            FOUR_C_THROW("unknown element type!");
            break;
        }
        // Intel compiler needs a return so
        return -1;
      };

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
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      Core::Elements::ElementType& element_type() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
          {
            return MembraneScatraTri3Type::instance();
          }
          break;
          case Core::FE::CellType::tri6:
          {
            return MembraneScatraTri6Type::instance();
          }
          break;
          case Core::FE::CellType::quad4:
          {
            return MembraneScatraQuad4Type::instance();
          }
          break;
          case Core::FE::CellType::quad9:
          {
            return MembraneScatraQuad9Type::instance();
          }
          break;
          default:
            FOUR_C_THROW("unknown element type!");
            break;
        }
        // Intel compiler needs a return so
        return MembraneQuad4Type::instance();
      };

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& eledistype,
          const Core::IO::InputParameterContainer& container,
          const Core::IO::MeshInput::ElementDataFromCellData& element_data) override;

      //@}

      //! @name Evaluation

      /*!
      \brief Pre-evaluate an element

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param discretization : pointer to discretization for de-assembly
      \param la (in)        : location array for de-assembly
      */
      void pre_evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          Core::Elements::LocationArray& la);

      /*!
      \brief Evaluate an element

      Evaluate Membrane element stiffness, mass, internal forces etc

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param discretization : pointer to discretization for de-assembly
      \param la (in)        : location array for de-assembly
      \param elemat1 (out)  : (stiffness-)matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elemat2 (out)  : (mass-)matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elevec1 (out)  : (internal force-)vector to be filled by element. If nullptr on input,
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

      //! @name params
      /*!
      \brief return ScaTra::ImplType
      */
      const Inpar::ScaTra::ImplType& impl_type() const { return impltype_; };

      //@}

     private:
      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType shape() const override;

      //! @{
      //! scalar transport implementation type (physics)
      Inpar::ScaTra::ImplType impltype_;
      //@}

     protected:
      //! don't want = operator
      MembraneScatra& operator=(const MembraneScatra& old);
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
