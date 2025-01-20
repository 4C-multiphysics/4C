// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1_poro_scatra.hpp"

#include "4C_w1_poro_scatra_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::Wall1PoroScatra<distype>::Wall1PoroScatra(int id, int owner)
    : Discret::Elements::Wall1Poro<distype>(id, owner), impltype_(Inpar::ScaTra::impltype_undefined)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                    schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::Wall1PoroScatra<distype>::Wall1PoroScatra(
    const Discret::Elements::Wall1PoroScatra<distype>& old)
    : Discret::Elements::Wall1Poro<distype>(old), impltype_(old.impltype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance and return pointer to it (public)           |
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::Elements::Element* Discret::Elements::Wall1PoroScatra<distype>::clone() const
{
  Discret::Elements::Wall1PoroScatra<distype>* newelement =
      new Discret::Elements::Wall1PoroScatra<distype>(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data (public)                                    schmidt 09/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroScatra<distype>::pack(Core::Communication::PackBuffer& data) const
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
void Discret::Elements::Wall1PoroScatra<distype>::unpack(Core::Communication::UnpackBuffer& buffer)
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
template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroScatra<distype>::print(std::ostream& os) const
{
  os << "Wall1_Poro_Scatra ";
  Core::Elements::Element::print(os);
  std::cout << std::endl;
  return;
}

/*----------------------------------------------------------------------*
 |  read this element (public)                             schmidt 09/17|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Discret::Elements::Wall1PoroScatra<distype>::read_element(const std::string& eletype,
    const std::string& eledistype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  my::read_element(eletype, eledistype, container);

  // read implementation type
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
    FOUR_C_THROW("Invalid implementation type for Wall1_Poro_Scatra elements!");

  return true;
}


/*----------------------------------------------------------------------*
 |                                                         schmidt 09/17|
 *----------------------------------------------------------------------*/
template class Discret::Elements::Wall1PoroScatra<Core::FE::CellType::tri3>;
template class Discret::Elements::Wall1PoroScatra<Core::FE::CellType::quad4>;
template class Discret::Elements::Wall1PoroScatra<Core::FE::CellType::quad9>;
template class Discret::Elements::Wall1PoroScatra<Core::FE::CellType::nurbs4>;
template class Discret::Elements::Wall1PoroScatra<Core::FE::CellType::nurbs9>;

FOUR_C_NAMESPACE_CLOSE
