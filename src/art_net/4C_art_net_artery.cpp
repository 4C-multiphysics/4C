// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_artery.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::Elements::ArteryType Discret::Elements::ArteryType::instance_;

Discret::Elements::ArteryType& Discret::Elements::ArteryType::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::ArteryType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Artery* object = new Discret::Elements::Artery(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::ArteryType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "ART")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Artery>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::ArteryType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Artery>(id, owner);
  return ele;
}


void Discret::Elements::ArteryType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["ART"];

  using namespace Core::IO::InputSpecBuilders;

  defs["LINE2"] = all_of({
      parameter<int>("MAT"),
      parameter<int>("GP"),
      parameter<std::string>("TYPE"),
      parameter<double>("DIAM"),
  });
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                           ismail 01/09|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::Artery::Artery(int id, int owner)
    : Core::Elements::Element(id, owner), impltype_(Inpar::ArtDyn::impltype_undefined)
{
  gaussrule_ = Core::FE::GaussRule1D::undefined;

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      ismail 01/09|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::Artery::Artery(const Discret::Elements::Artery& old)
    : Core::Elements::Element(old), impltype_(old.impltype_), gaussrule_(old.gaussrule_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Artery and return pointer to it (public) |
 |                                                         ismail 01/09 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Artery::clone() const
{
  Discret::Elements::Artery* newelement = new Discret::Elements::Artery(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         ismail 01/09 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Artery::shape() const
{
  switch (num_node())
  {
    case 2:
      return Core::FE::CellType::line2;
    default:
      FOUR_C_THROW("unexpected number of nodes {}", num_node());
  }
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         ismail 01/09 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Artery::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);
  // Gaussrule
  add_to_pack(data, gaussrule_);
  add_to_pack(data, impltype_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         ismail 01/09 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Artery::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);
  // Gaussrule
  extract_from_pack(buffer, gaussrule_);
  extract_from_pack(buffer, impltype_);


  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                       kremheller 10/18 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Artery::lines()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

int Discret::Elements::Artery::num_dof_per_node(const Core::Nodes::Node& node) const
{
  switch (impltype_)
  {
    case Inpar::ArtDyn::impltype_lin_exp:
    {
      return 2;
      break;
    }
    case Inpar::ArtDyn::impltype_pressure_based:
    {
      return 1;
      break;
    }
    default:
    {
      FOUR_C_THROW("Defined problem type {} does not exist!!", impltype_);
      break;
    }
  }

  return 0;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                             ismail 01/09|
 *----------------------------------------------------------------------*/
void Discret::Elements::Artery::print(std::ostream& os) const
{
  os << "Artery ";
  Element::print(os);

  return;
}

FOUR_C_NAMESPACE_CLOSE
