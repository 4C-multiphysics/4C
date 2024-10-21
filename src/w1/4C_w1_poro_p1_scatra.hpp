#ifndef FOUR_C_W1_PORO_P1_SCATRA_HPP
#define FOUR_C_W1_PORO_P1_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_inpar_scatra.hpp"
#include "4C_w1_poro_p1.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace ELEMENTS
  {
    /*!
    \brief A C++ version of a 2 dimensional solid element with modifications for porous media using
    p1 (mixed) approach including scatra functionality

    */
    template <Core::FE::CellType distype>
    class Wall1PoroP1Scatra : public Wall1PoroP1<distype>
    {
      typedef Discret::ELEMENTS::Wall1PoroP1<distype> my;

     public:
      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      Wall1PoroP1Scatra(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Wall1PoroP1Scatra(const Wall1PoroP1Scatra& old);


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

     protected:
      //! don't want = operator
      Wall1PoroP1Scatra& operator=(const Wall1PoroP1Scatra& old);
    };
  }  // namespace ELEMENTS
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
