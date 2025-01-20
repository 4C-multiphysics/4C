// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1_poro_p1_scatra.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_w1_poro_p1_scatra_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::Wall1PoroP1Scatra<distype>::Wall1PoroP1Scatra(int id, int owner)
    : Discret::Elements::Wall1PoroP1<distype>(id, owner),
      impltype_(Inpar::ScaTra::impltype_undefined)
{
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                    schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::Wall1PoroP1Scatra<distype>::Wall1PoroP1Scatra(
    const Discret::Elements::Wall1PoroP1Scatra<distype>& old)
    : Discret::Elements::Wall1PoroP1<distype>(old), impltype_(old.impltype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance and return pointer to it (public)           |
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::Elements::Element* Discret::Elements::Wall1PoroP1Scatra<distype>::clone() const
{
  Discret::Elements::Wall1PoroP1Scatra<distype>* newelement =
      new Discret::Elements::Wall1PoroP1Scatra<distype>(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data (public)                                    schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1Scatra<distype>::pack(
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
template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1Scatra<distype>::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract scalar transport impltype
  extract_from_pack(buffer, impltype_);

  // extract base class Element
  my::unpack(buffer);



  return;
}

/*----------------------------------------------------------------------*
 |  read this element (public)                             schmidt 09/17|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Discret::Elements::Wall1PoroP1Scatra<distype>::read_element(const std::string& eletype,
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
    FOUR_C_THROW("Invalid implementation type for Wall1_PoroP1Scatra elements!");

  return true;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                           schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1Scatra<distype>::print(std::ostream& os) const
{
  os << "Wall1_PoroP1Scatra ";
  Core::Elements::Element::print(os);
  std::cout << std::endl;
  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                           schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::Wall1PoroP1Scatra<distype>::unique_par_object_id() const
{
  int parobjectid(-1);
  switch (distype)
  {
    case Core::FE::CellType::tri3:
    {
      parobjectid = Discret::Elements::WallTri3PoroP1ScatraType::instance().unique_par_object_id();
      break;
    }
    case Core::FE::CellType::quad4:
    {
      parobjectid = Discret::Elements::WallQuad4PoroP1ScatraType::instance().unique_par_object_id();
      break;
    }
    case Core::FE::CellType::quad9:
    {
      parobjectid = Discret::Elements::WallQuad9PoroP1ScatraType::instance().unique_par_object_id();
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown element type");
      break;
    }
  }
  return parobjectid;
}

/*----------------------------------------------------------------------*
 | get the element type (public)                           schmidt 09/17|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::Elements::ElementType& Discret::Elements::Wall1PoroP1Scatra<distype>::element_type() const
{
  switch (distype)
  {
    case Core::FE::CellType::tri3:
      return Discret::Elements::WallTri3PoroP1ScatraType::instance();
      break;
    case Core::FE::CellType::quad4:
      return Discret::Elements::WallQuad4PoroP1ScatraType::instance();
      break;
    case Core::FE::CellType::quad9:
      return Discret::Elements::WallQuad9PoroP1ScatraType::instance();
      break;
    default:
      FOUR_C_THROW("unknown element type");
      break;
  }
  return Discret::Elements::WallQuad4PoroP1ScatraType::instance();
}

/*----------------------------------------------------------------------*
 *                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
template class Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::tri3>;
template class Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad4>;
template class Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad9>;

FOUR_C_NAMESPACE_CLOSE
