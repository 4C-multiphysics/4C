/*----------------------------------------------------------------------*/
/*! \file

\level 3


\brief Nonlinear Membrane Finite Element with ScaTra coupling

*----------------------------------------------------------------------*/
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

  namespace ELEMENTS
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

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-constructor is needed

      */
      Core::Elements::Element* Clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
          {
            return MembraneScatraTri3Type::Instance().UniqueParObjectId();
          }
          case Core::FE::CellType::tri6:
          {
            return MembraneScatraTri6Type::Instance().UniqueParObjectId();
          }
          case Core::FE::CellType::quad4:
          {
            return MembraneScatraQuad4Type::Instance().UniqueParObjectId();
          }
          case Core::FE::CellType::quad9:
          {
            return MembraneScatraQuad9Type::Instance().UniqueParObjectId();
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
      void unpack(const std::vector<char>& data) override;


      //@}

      //! @name Acess methods

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      Core::Elements::ElementType& ElementType() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
          {
            return MembraneScatraTri3Type::Instance();
          }
          break;
          case Core::FE::CellType::tri6:
          {
            return MembraneScatraTri6Type::Instance();
          }
          break;
          case Core::FE::CellType::quad4:
          {
            return MembraneScatraQuad4Type::Instance();
          }
          break;
          case Core::FE::CellType::quad9:
          {
            return MembraneScatraQuad9Type::Instance();
          }
          break;
          default:
            FOUR_C_THROW("unknown element type!");
            break;
        }
        // Intel compiler needs a return so
        return MembraneQuad4Type::Instance();
      };

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& eledistype,
          Input::LineDefinition* linedef) override;

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
          Core::Elements::Element::LocationArray& la);

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
          Core::Elements::Element::LocationArray& la,
          Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
          Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
          Core::LinAlg::SerialDenseVector& elevec1_epetra,
          Core::LinAlg::SerialDenseVector& elevec2_epetra,
          Core::LinAlg::SerialDenseVector& elevec3_epetra) override;

      //@}

      //! @name params
      /*!
      \brief return ScaTra::ImplType
      */
      const Inpar::ScaTra::ImplType& ImplType() const { return impltype_; };

      //@}

     private:
      /*!
      \brief Get vector of ptrs to nodes
      */
      Core::Nodes::Node** Nodes() override;

      /*!
      \brief Get shape type of element
      */
      Core::FE::CellType Shape() const override;

      //! @{
      //! scalar transport implementation type (physics)
      Inpar::ScaTra::ImplType impltype_;
      //@}

     protected:
      //! don't want = operator
      MembraneScatra& operator=(const MembraneScatra& old);
    };

  }  // namespace ELEMENTS
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
