// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_PORO_SCATRA_HPP
#define FOUR_C_W1_PORO_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_inpar_scatra.hpp"
#include "4C_w1_poro.hpp"
#include "4C_w1_poro_scatra_eletypes.hpp"

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
    class Wall1PoroScatra : public Wall1Poro<distype>
    {
      using my = Discret::Elements::Wall1Poro<distype>;

     public:
      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      Wall1PoroScatra(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      Wall1PoroScatra(const Wall1PoroScatra& old);

      //@}

      /*!
      \brief Deep copy this instance of Wall1_Poro_Scatra and return pointer to the copy

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
        int parobjectid(-1);
        switch (distype)
        {
          case Core::FE::CellType::tri3:
          {
            parobjectid =
                Discret::Elements::WallTri3PoroScatraType::instance().unique_par_object_id();
            break;
          }
          case Core::FE::CellType::quad4:
          {
            parobjectid =
                Discret::Elements::WallQuad4PoroScatraType::instance().unique_par_object_id();
            break;
          }
          case Core::FE::CellType::quad9:
          {
            parobjectid =
                Discret::Elements::WallQuad9PoroScatraType::instance().unique_par_object_id();
            break;
          }
          case Core::FE::CellType::nurbs4:
          {
            parobjectid =
                Discret::Elements::WallNurbs4PoroScatraType::instance().unique_par_object_id();
            break;
          }
          case Core::FE::CellType::nurbs9:
          {
            parobjectid =
                Discret::Elements::WallNurbs9PoroScatraType::instance().unique_par_object_id();
            break;
          }
          default:
          {
            FOUR_C_THROW("unknown element type");
            break;
          }
        }
        return parobjectid;
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

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      /*!
      \brief Return elementtype instance
      */
      Core::Elements::ElementType& element_type() const override
      {
        switch (distype)
        {
          case Core::FE::CellType::tri3:
            return Discret::Elements::WallTri3PoroScatraType::instance();
            break;
          case Core::FE::CellType::quad4:
            return Discret::Elements::WallQuad4PoroScatraType::instance();
            break;
          case Core::FE::CellType::quad9:
            return Discret::Elements::WallQuad9PoroScatraType::instance();
            break;
          case Core::FE::CellType::nurbs4:
            return Discret::Elements::WallNurbs4PoroScatraType::instance();
            break;
          case Core::FE::CellType::nurbs9:
            return Discret::Elements::WallNurbs9PoroScatraType::instance();
            break;
          default:
            FOUR_C_THROW("unknown element type");
            break;
        }
        return Discret::Elements::WallQuad4PoroScatraType::instance();
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

      /// @name params
      /*!
      \brief Return the SCATRA ImplType
      */
      const Inpar::ScaTra::ImplType& impl_type() const { return impltype_; };

     private:
      //! implementation type (physics)
      Inpar::ScaTra::ImplType impltype_;
      //@}

     protected:
      //! don't want = operator
      Wall1PoroScatra& operator=(const Wall1PoroScatra& old);

    };  // class Wall1_Poro_Scatra

  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
