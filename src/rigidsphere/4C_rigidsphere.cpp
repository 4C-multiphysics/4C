// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_rigidsphere.hpp"

#include "4C_beaminteraction_link_pinjointed.hpp"
#include "4C_browniandyn_input.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::Elements::RigidsphereType Discret::Elements::RigidsphereType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::RigidsphereType& Discret::Elements::RigidsphereType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::RigidsphereType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Rigidsphere* object = new Discret::Elements::Rigidsphere(-1, -1);
  object->unpack(buffer);
  return (object);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::RigidsphereType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "RIGIDSPHERE")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Rigidsphere>(id, owner);
    return (ele);
  }
  return (nullptr);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::RigidsphereType::create(
    const int id, const int owner)
{
  return (std::make_shared<Rigidsphere>(id, owner));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::RigidsphereType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 3;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::RigidsphereType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented!");
  return nullspace;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::RigidsphereType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["RIGIDSPHERE"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::point1] = all_of({
      parameter<double>("RADIUS"),
      parameter<double>("DENSITY"),
  });
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            meier 05/12|
 *----------------------------------------------------------------------*/
Discret::Elements::Rigidsphere::Rigidsphere(int id, int owner)
    : Core::Elements::Element(id, owner), radius_(0.0), rho_(0.0)
{
  mybondstobeams_.clear();
}
/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       meier 05/12|
 *----------------------------------------------------------------------*/
Discret::Elements::Rigidsphere::Rigidsphere(const Discret::Elements::Rigidsphere& old)
    : Core::Elements::Element(old), radius_(old.radius_), rho_(old.rho_)
{
  mybondstobeams_.clear();
  if (old.mybondstobeams_.size())
  {
    for (auto const& iter : old.mybondstobeams_)
    {
      if (iter.second != nullptr)
        mybondstobeams_[iter.first] =
            std::dynamic_pointer_cast<BeamInteraction::BeamLinkPinJointed>(iter.second->clone());
      else
        FOUR_C_THROW("something went wrong, I am sorry. Please go debugging.");
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Rigidsphere and return pointer to it (public) |
 |                                                            meier 05/12 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Rigidsphere::clone() const
{
  Discret::Elements::Rigidsphere* newelement = new Discret::Elements::Rigidsphere(*this);
  return (newelement);
}



/*----------------------------------------------------------------------*
 |  print this element (public)                              meier 05/12
 *----------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::print(std::ostream& os) const { return; }


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          meier 05/12 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Rigidsphere::shape() const
{
  return (Core::FE::CellType::point1);
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           meier 05/12/
 *----------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);

  // add all class variables
  add_to_pack(data, radius_);
  add_to_pack(data, rho_);

  add_to_pack(data, mybondstobeams_.size());
  for (auto const& iter : mybondstobeams_) iter.second->pack(data);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           meier 05/12|
 *----------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);


  // extract all class variables
  extract_from_pack(buffer, radius_);
  extract_from_pack(buffer, rho_);

  std::size_t numbonds;
  extract_from_pack(buffer, numbonds);
  for (int unsigned i = 0; i < numbonds; ++i)
  {
    std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
    std::shared_ptr<BeamInteraction::BeamLinkPinJointed> link =
        std::dynamic_pointer_cast<BeamInteraction::BeamLinkPinJointed>(object);
    if (link == nullptr) FOUR_C_THROW("Received object is not a beam to beam linkage");
    mybondstobeams_[link->id()] = link;
  }


  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             meier 02/14|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Rigidsphere::lines()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}


/*----------------------------------------------------------------------*
 |  Initialize (public)                                      meier 05/12|
 *----------------------------------------------------------------------*/
int Discret::Elements::RigidsphereType::initialize(Core::FE::Discretization& dis) { return 0; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
    interface_ptr_ = std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  else
    interface_ptr_ = nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::ParamsInterface>
Discret::Elements::Rigidsphere::params_interface_ptr()
{
  return interface_ptr_;
}

FOUR_C_NAMESPACE_CLOSE
