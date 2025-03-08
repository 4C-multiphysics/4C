// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_pack_helpers.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_pstream.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::FE;

Discret::Elements::RedAirwayType Discret::Elements::RedAirwayType::instance_;


Discret::Elements::RedAirwayType& Discret::Elements::RedAirwayType::instance() { return instance_; }


Core::Communication::ParObject* Discret::Elements::RedAirwayType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::RedAirway* object = new Discret::Elements::RedAirway(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::RedAirwayType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "RED_AIRWAY")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::RedAirway>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::RedAirwayType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::RedAirway>(id, owner);
  return ele;
}


/*--------------------------------------------------------------------  *
 | Read RED_AIRWAY element line and add element specific parameters     |
 |                                                             (public) |
 |                                                           roth 10/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAirwayType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["RED_AIRWAY"];

  using namespace Core::IO::InputSpecBuilders;

  defs["LINE2"] = all_of({
      parameter<std::vector<int>>("LINE2", {.size = 2}),
      parameter<int>("MAT"),
      parameter<std::string>("ElemSolvingType"),
      deprecated_selection<std::string>("TYPE",
          {"Resistive", "InductoResistive", "CompliantResistive", "RLC", "ViscoElasticRLC",
              "ConvectiveViscoElasticRLC"},
          {.description = "Reduced-dimensional model of this airway"}),
      parameter<std::string>("Resistance"),
      parameter<double>("PowerOfVelocityProfile"),
      parameter<double>("WallElasticity"),
      parameter<double>("PoissonsRatio"),
      parameter<double>("ViscousTs"),
      parameter<double>("ViscousPhaseShift"),
      parameter<double>("WallThickness"),
      parameter<double>("Area"),
      parameter<int>("Generation"),
      parameter<std::optional<double>>("AirwayColl"),
      parameter<std::optional<double>>("S_Close"),
      parameter<std::optional<double>>("S_Open"),
      parameter<std::optional<double>>("Pcrit_Open"),
      parameter<std::optional<double>>("Pcrit_Close"),
      parameter<std::optional<double>>("Open_Init"),
      parameter<std::optional<double>>("BranchLength"),
  });
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                           ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::RedAirway::RedAirway(int id, int owner) : Core::Elements::Element(id, owner) {}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::RedAirway::RedAirway(const Discret::Elements::RedAirway& old)
    : Core::Elements::Element(old),
      elem_type_(old.elem_type_),
      resistance_(old.resistance_),
      elemsolving_type_(old.elemsolving_type_),
      airway_params_(old.airway_params_)
{
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of RedAirway and return pointer             |
 |  to it                                                      (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::RedAirway::clone() const
{
  Discret::Elements::RedAirway* newelement = new Discret::Elements::RedAirway(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::RedAirway::shape() const
{
  switch (num_node())
  {
    case 2:
      return Core::FE::CellType::line2;
    case 3:
      return Core::FE::CellType::line3;
    default:
      FOUR_C_THROW("unexpected number of nodes %d", num_node());
  }
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAirway::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);

  add_to_pack(data, elem_type_);
  add_to_pack(data, resistance_);
  add_to_pack(data, elemsolving_type_);

  add_to_pack(data, airway_params_.power_velocity_profile);
  add_to_pack(data, airway_params_.wall_elasticity);
  add_to_pack(data, airway_params_.poisson_ratio);
  add_to_pack(data, airway_params_.wall_thickness);
  add_to_pack(data, airway_params_.area);
  add_to_pack(data, airway_params_.viscous_Ts);
  add_to_pack(data, airway_params_.viscous_phase_shift);
  add_to_pack(data, airway_params_.branch_length);
  add_to_pack(data, airway_params_.generation);

  add_to_pack(data, airway_params_.airway_coll);
  add_to_pack(data, airway_params_.s_close);
  add_to_pack(data, airway_params_.s_open);
  add_to_pack(data, airway_params_.p_crit_open);
  add_to_pack(data, airway_params_.p_crit_close);
  add_to_pack(data, airway_params_.open_init);
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAirway::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);

  extract_from_pack(buffer, elem_type_);
  extract_from_pack(buffer, resistance_);
  extract_from_pack(buffer, elemsolving_type_);

  extract_from_pack(buffer, airway_params_.power_velocity_profile);
  extract_from_pack(buffer, airway_params_.wall_elasticity);
  extract_from_pack(buffer, airway_params_.poisson_ratio);
  extract_from_pack(buffer, airway_params_.wall_thickness);
  extract_from_pack(buffer, airway_params_.area);
  extract_from_pack(buffer, airway_params_.viscous_Ts);
  extract_from_pack(buffer, airway_params_.viscous_phase_shift);
  extract_from_pack(buffer, airway_params_.branch_length);
  extract_from_pack(buffer, airway_params_.generation);

  extract_from_pack(buffer, airway_params_.airway_coll);
  extract_from_pack(buffer, airway_params_.s_close);
  extract_from_pack(buffer, airway_params_.s_open);
  extract_from_pack(buffer, airway_params_.p_crit_open);
  extract_from_pack(buffer, airway_params_.p_crit_close);
  extract_from_pack(buffer, airway_params_.open_init);
}



/*----------------------------------------------------------------------*
 |  print this element (public)                             ismail 01/10|
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAirway::print(std::ostream& os) const
{
  os << "RedAirway ";
  Element::print(os);
}


/*----------------------------------------------------------------------*
 |  Return visualization data (public)                     ismail 02/10 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::RedAirway::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return false;
}


const Discret::ReducedLung::AirwayParams& Discret::Elements::RedAirway::get_airway_params() const
{
  return airway_params_;
}

/*----------------------------------------------------------------------*
 |  get vector of lines              (public)              ismail  02/13|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::RedAirway::lines()
{
  FOUR_C_ASSERT(num_line() == 1, "RED_AIRWAY element must have one and only one line");

  return {Core::Utils::shared_ptr_from_ref(*this)};
}

FOUR_C_NAMESPACE_CLOSE
