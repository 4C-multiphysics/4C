#include "4C_membrane_scatra.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_io_linedefinition.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  constructor (public)                                   sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::ELEMENTS::MembraneScatra<distype>::MembraneScatra(int id, int owner)
    : Membrane<distype>(id, owner), impltype_(Inpar::ScaTra::impltype_undefined)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-constructor (public)                              sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::ELEMENTS::MembraneScatra<distype>::MembraneScatra(
    const Discret::ELEMENTS::MembraneScatra<distype>& old)
    : Membrane<distype>(old), impltype_(old.impltype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of MembraneScatra              sfuchs 05/18 |
 |  and return pointer to it (public)                                   |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::Elements::Element* Discret::ELEMENTS::MembraneScatra<distype>::clone() const
{
  Discret::ELEMENTS::MembraneScatra<distype>* newelement =
      new Discret::ELEMENTS::MembraneScatra<distype>(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::ELEMENTS::MembraneScatra<distype>::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

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
 |                                                         sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::ELEMENTS::MembraneScatra<distype>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract scalar transport impltype
  impltype_ = static_cast<Inpar::ScaTra::ImplType>(extract_int(buffer));

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer basedata_buffer(basedata);
  Membrane<distype>::unpack(basedata_buffer);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");

  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                            sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::ELEMENTS::MembraneScatra<distype>::print(std::ostream& os) const
{
  os << "MembraneScatra ";
  Membrane<distype>::print(os);

  return;
}

/*----------------------------------------------------------------------*
 |  read this element (public)                             sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Discret::ELEMENTS::MembraneScatra<distype>::read_element(const std::string& eletype,
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
 |  Get vector of ptrs to nodes (private)                  sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
inline Core::Nodes::Node** Discret::ELEMENTS::MembraneScatra<distype>::nodes()
{
  return Membrane<distype>::nodes();
}

/*----------------------------------------------------------------------*
 |  Get shape type of element (private)                    sfuchs 05/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::FE::CellType Discret::ELEMENTS::MembraneScatra<distype>::shape() const
{
  return Membrane<distype>::shape();
}

template class Discret::ELEMENTS::MembraneScatra<Core::FE::CellType::tri3>;
template class Discret::ELEMENTS::MembraneScatra<Core::FE::CellType::tri6>;
template class Discret::ELEMENTS::MembraneScatra<Core::FE::CellType::quad4>;
template class Discret::ELEMENTS::MembraneScatra<Core::FE::CellType::quad9>;

FOUR_C_NAMESPACE_CLOSE
