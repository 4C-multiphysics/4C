// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1_scatra.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Wall1ScatraType Discret::Elements::Wall1ScatraType::instance_;

Discret::Elements::Wall1ScatraType& Discret::Elements::Wall1ScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::Wall1ScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Wall1Scatra* object = new Discret::Elements::Wall1Scatra(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Wall1ScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLSCATRA")
  {
    if (eledistype != "NURBS4" and eledistype != "NURBS9")
    {
      return std::make_shared<Discret::Elements::Wall1Scatra>(id, owner);
    }
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::Wall1ScatraType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Wall1Scatra>(id, owner);
}

void Discret::Elements::Wall1ScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_wall;
  Wall1Type::setup_element_definition(definitions_wall);

  auto& defs_wall = definitions_wall["WALL"];

  auto& defs = definitions["WALLSCATRA"];

  using namespace Core::IO::InputSpecBuilders;

  for (const auto& [key, wall_line_def] : defs_wall)
  {
    defs[key] = anonymous_group({
        wall_line_def,
        entry<std::string>("TYPE"),
    });
  }
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            vuong 01/14/|
 *----------------------------------------------------------------------*/
Discret::Elements::Wall1Scatra::Wall1Scatra(int id, int owner)
    : Wall1(id, owner), impltype_(Inpar::ScaTra::impltype_undefined)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       vuong 01/14|
 *----------------------------------------------------------------------*/
Discret::Elements::Wall1Scatra::Wall1Scatra(const Discret::Elements::Wall1Scatra& old)
    : Wall1(old), impltype_(old.impltype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Wall1 and return pointer to it (public) |
 |                                                            vuong 01/14 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Wall1Scatra::clone() const
{
  Discret::Elements::Wall1Scatra* newelement = new Discret::Elements::Wall1Scatra(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            vuong 01/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1Scatra::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // pack scalar transport impltype
  add_to_pack(data, impltype_);

  // add base class Element
  Wall1::pack(data);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            vuong 01/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1Scatra::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract scalar transport impltype
  extract_from_pack(buffer, impltype_);

  // extract base class Element
  Wall1::unpack(buffer);
}

/*----------------------------------------------------------------------*
 |  print this element (public)                              vuong 01/14|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1Scatra::print(std::ostream& os) const
{
  os << "Wall1_Scatra ";
  Wall1::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  read this element (public)                             schmidt 09/17|
 *----------------------------------------------------------------------*/
bool Discret::Elements::Wall1Scatra::read_element(const std::string& eletype,
    const std::string& eledistype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  Wall1::read_element(eletype, eledistype, container);

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
  else if (impltype == "RefConcReac")
    impltype_ = Inpar::ScaTra::impltype_refconcreac;
  else if (impltype == "Std")
    impltype_ = Inpar::ScaTra::impltype_std;
  else
    FOUR_C_THROW("Invalid implementation type for Wall1_Scatra elements!");

  return true;
}

FOUR_C_NAMESPACE_CLOSE
