// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_membrane_scatra.hpp"

#include "4C_comm_pack_helpers.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  constructor (public)                                                |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::MembraneScatra<distype>::MembraneScatra(int id, int owner)
    : Membrane<distype>(id, owner), impltype_(Inpar::ScaTra::impltype_undefined)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-constructor (public)                                           |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::MembraneScatra<distype>::MembraneScatra(
    const Discret::Elements::MembraneScatra<distype>& old)
    : Membrane<distype>(old), impltype_(old.impltype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of MembraneScatra                           |
 |  and return pointer to it (public)                                   |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::Elements::Element* Discret::Elements::MembraneScatra<distype>::clone() const
{
  Discret::Elements::MembraneScatra<distype>* newelement =
      new Discret::Elements::MembraneScatra<distype>(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                                      |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::MembraneScatra<distype>::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // pack scalar transport impltype_
  add_to_pack(data, impltype_);

  // add base class Element
  Membrane<distype>::pack(data);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                                      |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::MembraneScatra<distype>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract scalar transport impltype
  extract_from_pack(buffer, impltype_);

  // extract base class Element
  Membrane<distype>::unpack(buffer);



  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                                         |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::MembraneScatra<distype>::print(std::ostream& os) const
{
  os << "MembraneScatra ";
  Membrane<distype>::print(os);

  return;
}

/*----------------------------------------------------------------------*
 |  read this element (public)                                          |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Discret::Elements::MembraneScatra<distype>::read_element(const std::string& eletype,
    const std::string& eledistype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  Membrane<distype>::read_element(eletype, eledistype, container);

  // read scalar transport implementation type
  std::string impltype = container.get<std::string>("TYPE");

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
  else if (impltype == "RefConcReac")
    impltype_ = Inpar::ScaTra::impltype_refconcreac;
  else if (impltype == "Std")
    impltype_ = Inpar::ScaTra::impltype_std;
  else
    FOUR_C_THROW("Invalid implementation type for Wall1_Scatra elements!");

  return true;
}

/*----------------------------------------------------------------------*
 |  Get vector of ptrs to nodes (private)                               |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
inline Core::Nodes::Node** Discret::Elements::MembraneScatra<distype>::nodes()
{
  return Membrane<distype>::nodes();
}

/*----------------------------------------------------------------------*
 |  Get shape type of element (private)                                 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::FE::CellType Discret::Elements::MembraneScatra<distype>::shape() const
{
  return Membrane<distype>::shape();
}

template class Discret::Elements::MembraneScatra<Core::FE::CellType::tri3>;
template class Discret::Elements::MembraneScatra<Core::FE::CellType::tri6>;
template class Discret::Elements::MembraneScatra<Core::FE::CellType::quad4>;
template class Discret::Elements::MembraneScatra<Core::FE::CellType::quad9>;

FOUR_C_NAMESPACE_CLOSE
