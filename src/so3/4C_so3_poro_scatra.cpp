// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_poro_scatra.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_so3_poro_scatra_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
Discret::Elements::So3PoroScatra<So3Ele, distype>::So3PoroScatra(int id, int owner)
    : So3Poro<So3Ele, distype>(id, owner), impltype_(Inpar::ScaTra::impltype_undefined)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                    schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
Discret::Elements::So3PoroScatra<So3Ele, distype>::So3PoroScatra(
    const Discret::Elements::So3PoroScatra<So3Ele, distype>& old)
    : So3Poro<So3Ele, distype>(old), impltype_(old.impltype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance and return pointer to it (public)           |
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
Core::Elements::Element* Discret::Elements::So3PoroScatra<So3Ele, distype>::clone() const
{
  auto* newelement = new Discret::Elements::So3PoroScatra<So3Ele, distype>(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data (public)                                    schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3PoroScatra<So3Ele, distype>::pack(
    Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // pack scalar transport impltype
  add_to_pack(data, impltype_);

  // add base class Element
  my::pack(data);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data (public)                                  schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3PoroScatra<So3Ele, distype>::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract scalar transport impltype_
  extract_from_pack(buffer, impltype_);

  // extract base class Element
  my::unpack(buffer);



  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                           schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3PoroScatra<So3Ele, distype>::print(std::ostream& os) const
{
  os << "So3_Poro_Scatra ";
  os << Core::FE::cell_type_to_string(distype).c_str() << " ";
  Core::Elements::Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  read this element (public)                             schmidt 09/17|
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
bool Discret::Elements::So3PoroScatra<So3Ele, distype>::read_element(const std::string& eletype,
    const std::string& eledistype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  my::read_element(eletype, eledistype, container);

  // read scalar transport implementation type
  auto impltype = container.get<std::string>("TYPE");

  if (impltype == "Undefined")
    impltype_ = Inpar::ScaTra::impltype_undefined;
  else if (impltype == "AdvReac")
    impltype_ = Inpar::ScaTra::impltype_advreac;
  else if (impltype == "CardMono")
    impltype_ = Inpar::ScaTra::impltype_cardiac_monodomain;
  else if (impltype == "Chemo")
    impltype_ = Inpar::ScaTra::impltype_chemo;
  else if (impltype == "ChemoReac")
    impltype_ = Inpar::ScaTra::impltype_chemoreac;
  else if (impltype == "Loma")
    impltype_ = Inpar::ScaTra::impltype_loma;
  else if (impltype == "Poro")
    impltype_ = Inpar::ScaTra::impltype_poro;
  else if (impltype == "PoroReac")
    impltype_ = Inpar::ScaTra::impltype_pororeac;
  else if (impltype == "PoroReacECM")
    impltype_ = Inpar::ScaTra::impltype_pororeacECM;
  else if (impltype == "PoroMultiReac")
    impltype_ = Inpar::ScaTra::impltype_multipororeac;
  else if (impltype == "RefConcReac")
    impltype_ = Inpar::ScaTra::impltype_refconcreac;
  else if (impltype == "Std")
    impltype_ = Inpar::ScaTra::impltype_std;
  else
    FOUR_C_THROW("Invalid implementation type for So3_Poro_Scatra elements!");

  return true;
}

/*----------------------------------------------------------------------*
 | get the unique ParObject Id (public)                    schmidt 09/17|
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
int Discret::Elements::So3PoroScatra<So3Ele, distype>::unique_par_object_id() const
{
  int parobjectid(-1);
  switch (distype)
  {
    case Core::FE::CellType::tet4:
    {
      parobjectid = SoTet4PoroScatraType::instance().unique_par_object_id();
      break;
    }
    case Core::FE::CellType::tet10:
    {
      parobjectid = SoTet10PoroScatraType::instance().unique_par_object_id();
      break;
    }
    case Core::FE::CellType::hex8:
    {
      parobjectid = SoHex8PoroScatraType::instance().unique_par_object_id();
      break;
    }
    case Core::FE::CellType::hex27:
    {
      parobjectid = SoHex27PoroScatraType::instance().unique_par_object_id();
      break;
    }
    case Core::FE::CellType::nurbs27:
    {
      parobjectid = SoNurbs27PoroScatraType::instance().unique_par_object_id();
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown element type!");
      break;
    }
  }
  return parobjectid;
}

/*----------------------------------------------------------------------*
 | get the element type (public)                           schmidt 09/17|
 *----------------------------------------------------------------------*/
template <class So3Ele, Core::FE::CellType distype>
Core::Elements::ElementType& Discret::Elements::So3PoroScatra<So3Ele, distype>::element_type() const
{
  switch (distype)
  {
    case Core::FE::CellType::tet4:
      return SoTet4PoroScatraType::instance();
    case Core::FE::CellType::tet10:
      return SoTet10PoroScatraType::instance();
    case Core::FE::CellType::hex8:
      return SoHex8PoroScatraType::instance();
    case Core::FE::CellType::hex27:
      return SoHex27PoroScatraType::instance();
    case Core::FE::CellType::nurbs27:
      return SoNurbs27PoroScatraType::instance();
    default:
      FOUR_C_THROW("unknown element type!");
      break;
  }
  return SoHex8PoroScatraType::instance();
};

FOUR_C_NAMESPACE_CLOSE

#include "4C_so3_poro_scatra.inst.hpp"
