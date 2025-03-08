// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_maxwell_0d_acinus.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::FE;

Discret::Elements::RedAcinusType Discret::Elements::RedAcinusType::instance_;

Discret::Elements::RedAcinusType& Discret::Elements::RedAcinusType::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::RedAcinusType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::RedAcinus* object = new Discret::Elements::RedAcinus(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::RedAcinusType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "RED_ACINUS")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::RedAcinus>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::RedAcinusType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::RedAcinus>(id, owner);
  return ele;
}


/*--------------------------------------------------------------------  *
 | Read RED_ACINUS element line and add element specific parameters     |
 |                                                             (public) |
 |                                                           roth 10/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAcinusType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["RED_ACINUS"];

  using namespace Core::IO::InputSpecBuilders;

  defs["LINE2"] = all_of({
      parameter<std::vector<int>>("LINE2", {.size = 2}),
      parameter<int>("MAT"),
      deprecated_selection<std::string>("TYPE",
          {"NeoHookean", "Exponential", "DoubleExponential", "VolumetricOgden"},
          {.description = "Visco-elastic model of this acinus"}),
      parameter<double>("AcinusVolume"),
      parameter<double>("AlveolarDuctVolume"),
      // Maxwell exponential
      parameter<std::optional<double>>("E1_0"),
      parameter<std::optional<double>>("E1_LIN"),
      parameter<std::optional<double>>("E1_EXP"),
      parameter<std::optional<double>>("TAU"),
      // Maxwell double exponential
      parameter<std::optional<double>>("E1_01"),
      parameter<std::optional<double>>("E1_LIN1"),
      parameter<std::optional<double>>("E1_EXP1"),
      parameter<std::optional<double>>("TAU1"),
      parameter<std::optional<double>>("E1_02"),
      parameter<std::optional<double>>("E1_LIN2"),
      parameter<std::optional<double>>("E1_EXP2"),
      parameter<std::optional<double>>("TAU2"),
      // VolOgden
      parameter<std::optional<double>>("KAPPA"),
      parameter<std::optional<double>>("BETA"),
  });
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                           ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::RedAcinus::RedAcinus(int id, int owner) : Core::Elements::Element(id, owner) {}


/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::RedAcinus::RedAcinus(const Discret::Elements::RedAcinus& old)
    : Core::Elements::Element(old),
      elem_type_(old.elem_type_),
      resistance_(old.elem_type_),
      acinus_params_(old.acinus_params_)
{
}


/*----------------------------------------------------------------------*
 |  Deep copy this instance of RedAcinus and return pointer             |
 |  to it                                                      (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::RedAcinus::clone() const
{
  Discret::Elements::RedAcinus* newelement = new Discret::Elements::RedAcinus(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::RedAcinus::shape() const
{
  switch (num_node())
  {
    case 2:
      return Core::FE::CellType::line2;
    case 3:
      return Core::FE::CellType::line3;
    default:
      FOUR_C_THROW("unexpected number of nodes %d", num_node());
      break;
  }
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAcinus::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);

  add_to_pack(data, elem_type_);
  add_to_pack(data, resistance_);

  add_to_pack(data, acinus_params_.volume_relaxed);
  add_to_pack(data, acinus_params_.alveolar_duct_volume);
  add_to_pack(data, acinus_params_.volume_init);
  add_to_pack(data, acinus_params_.generation);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAcinus::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);

  extract_from_pack(buffer, elem_type_);
  extract_from_pack(buffer, resistance_);

  extract_from_pack(buffer, acinus_params_.volume_relaxed);
  extract_from_pack(buffer, acinus_params_.alveolar_duct_volume);
  extract_from_pack(buffer, acinus_params_.volume_init);
  extract_from_pack(buffer, acinus_params_.generation);



  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                             ismail 01/10|
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAcinus::print(std::ostream& os) const
{
  os << "RedAcinus ";
  Element::print(os);

  return;
}

/*-----------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
std::vector<double> Discret::Elements::RedAcinus::element_center_refe_coords()
{
  //  // update element geometry
  Core::Nodes::Node** nodes = RedAcinus::nodes();

  Core::LinAlg::SerialDenseMatrix mat(num_node(), 3, false);
  for (int i = 0; i < num_node(); ++i)
  {
    const auto& x = nodes[i]->x();
    mat(i, 0) = x[0];
    mat(i, 1) = x[1];
    mat(i, 2) = x[2];
  }

  std::vector<double> centercoords(3, 0);
  for (int i = 0; i < 3; ++i)
  {
    double var = 0;
    for (int j = 0; j < num_node(); ++j)
    {
      var = var + mat(j, i);
    }
    centercoords[i] = var / num_node();
  }

  return centercoords;
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data                     ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::Elements::RedAcinus::vis_names(std::map<std::string, int>& names)
{
  std::shared_ptr<Core::Mat::Material> mat = material();

  // cast to specific material, because general material does not have vis_names/vis_data
  std::shared_ptr<Mat::Maxwell0dAcinus> mxwll_0d_acin =
      std::dynamic_pointer_cast<Mat::Maxwell0dAcinus>(material());
  mxwll_0d_acin->vis_names(names);
}


/*----------------------------------------------------------------------*
 |  Return visualization data (public)                     ismail 02/10 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::RedAcinus::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  // cast to specific material, because general material does not have vis_names/vis_data
  std::shared_ptr<Mat::Maxwell0dAcinus> mxwll_0d_acin =
      std::dynamic_pointer_cast<Mat::Maxwell0dAcinus>(material());

  return mxwll_0d_acin->vis_data(name, data, this->id());
}


void Discret::Elements::RedAcinus::update_relaxed_volume(double newVol)
{
  acinus_params_.volume_relaxed = newVol;
}


const Discret::ReducedLung::AcinusParams& Discret::Elements::RedAcinus::get_acinus_params() const
{
  return acinus_params_;
}

/*----------------------------------------------------------------------*
 |  get vector of lines              (public)              ismail  02/13|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::RedAcinus::lines()
{
  FOUR_C_ASSERT(num_line() == 1, "RED_AIRWAY element must have one and only one line");

  return {Core::Utils::shared_ptr_from_ref(*this)};
}

FOUR_C_NAMESPACE_CLOSE
