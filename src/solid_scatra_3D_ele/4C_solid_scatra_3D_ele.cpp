// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_scatra_3D_ele.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_solid_3D_ele_interface_serializable.hpp"
#include "4C_solid_3D_ele_line.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"
#include "4C_solid_3D_ele_properties.hpp"
#include "4C_solid_3D_ele_surface.hpp"
#include "4C_solid_scatra_3D_ele_factory.hpp"
#include "4C_solid_scatra_3D_ele_lib.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

namespace
{
  template <Core::FE::CellType celltype>
  auto get_default_input_spec()
  {
    return all_of({
        parameter<int>("MAT"),
        deprecated_selection<Inpar::Solid::KinemType>("KINEM",
            {
                {kinem_type_string(Inpar::Solid::KinemType::linear),
                    Inpar::Solid::KinemType::linear},
                {kinem_type_string(Inpar::Solid::KinemType::nonlinearTotLag),
                    Inpar::Solid::KinemType::nonlinearTotLag},
            },
            {.description = "Whether to use linear kinematics (small displacements) or nonlinear "
                            "kinematics (large displacements)"}),
        parameter<std::string>("TYPE"),
        parameter<Discret::Elements::PrestressTechnology>(
            "PRESTRESS_TECH", {.description = "The technology used for prestressing",
                                  .default_value = Discret::Elements::PrestressTechnology::none}),
        parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
        parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
        parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
        parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
        parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
        parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),

    });
  }
}  // namespace

Discret::Elements::SolidScatraType Discret::Elements::SolidScatraType::instance_;

Discret::Elements::SolidScatraType& Discret::Elements::SolidScatraType::instance()
{
  return instance_;
}

void Discret::Elements::SolidScatraType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SOLIDSCATRA"];

  defsgeneral[Core::FE::CellType::hex8] = all_of({
      get_default_input_spec<Core::FE::CellType::hex8>(),
      parameter<ElementTechnology>("TECH", {.default_value = ElementTechnology::none}),
  });

  defsgeneral[Core::FE::CellType::hex27] = get_default_input_spec<Core::FE::CellType::hex27>();

  defsgeneral[Core::FE::CellType::tet4] = get_default_input_spec<Core::FE::CellType::tet4>();

  defsgeneral[Core::FE::CellType::tet10] = get_default_input_spec<Core::FE::CellType::tet10>();

  defsgeneral[Core::FE::CellType::nurbs27] = get_default_input_spec<Core::FE::CellType::nurbs27>();
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidScatraType::create(
    const std::string eletype, const std::string elecelltype, const int id, const int owner)
{
  if (eletype == "SOLIDSCATRA") return create(id, owner);
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidScatraType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::SolidScatra>(id, owner);
}

Core::Communication::ParObject* Discret::Elements::SolidScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::SolidScatra(-1, -1);
  object->unpack(buffer);
  return object;
}

void Discret::Elements::SolidScatraType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  Solid::Utils::nodal_block_information_solid(dwele, numdf, dimns);
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::SolidScatraType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<3>(node.x(), x0);
}

Discret::Elements::SolidScatra::SolidScatra(int id, int owner) : Core::Elements::Element(id, owner)
{
}

Core::Elements::Element* Discret::Elements::SolidScatra::clone() const
{
  return new Discret::Elements::SolidScatra(*this);
}

int Discret::Elements::SolidScatra::num_line() const
{
  return Core::FE::get_number_of_element_lines(celltype_);
}

int Discret::Elements::SolidScatra::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(celltype_);
}

int Discret::Elements::SolidScatra::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(celltype_);
}

std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::SolidScatra::lines()
{
  return Core::Communication::get_element_lines<SolidLine<3>, SolidScatra>(*this);
}

std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::SolidScatra::surfaces()
{
  return Core::Communication::get_element_surfaces<SolidSurface, SolidScatra>(*this);
}

void Discret::Elements::SolidScatra::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
  {
    interface_ptr_ = p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface");
    solid_interface_ptr_ =
        std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(interface_ptr_);
  }
  else
  {
    interface_ptr_ = nullptr;
    solid_interface_ptr_ = nullptr;
  }
}

bool Discret::Elements::SolidScatra::read_element(const std::string& eletype,
    const std::string& celltype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  // set cell type
  celltype_ = Core::FE::string_to_cell_type(celltype);

  // read number of material model
  set_material(0, Mat::factory(Solid::Utils::ReadElement::read_element_material(container)));

  // read scalar transport implementation type
  properties_.impltype = read_scatra_impl_type(container);

  properties_.solid = Solid::Utils::ReadElement::read_solid_element_properties(container);

  solid_scatra_calc_variant_ =
      create_solid_scatra_calculation_interface(celltype_, properties_.solid);

  // setup solid material
  std::visit([&](auto& solid_scatra) { solid_scatra->setup(solid_material(), container); },
      solid_scatra_calc_variant_);

  return true;
}

void Discret::Elements::SolidScatra::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  // add base class Element
  Core::Elements::Element::pack(data);

  add_to_pack(data, celltype_);
  Discret::Elements::add_to_pack(data, properties_);

  data.add_to_pack(material_post_setup_);

  // optional data, e.g., EAS data
  Discret::Elements::pack(solid_scatra_calc_variant_, data);
}

void Discret::Elements::SolidScatra::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);

  extract_from_pack(buffer, celltype_);

  Discret::Elements::extract_from_pack(buffer, properties_);

  extract_from_pack(buffer, material_post_setup_);

  // reset solid and scatra interfaces
  solid_scatra_calc_variant_ =
      create_solid_scatra_calculation_interface(celltype_, properties_.solid);

  Discret::Elements::unpack(solid_scatra_calc_variant_, buffer);
}

void Discret::Elements::SolidScatra::vis_names(std::map<std::string, int>& names)
{
  Core::Elements::Element::vis_names(names);
  solid_material().vis_names(names);
}

bool Discret::Elements::SolidScatra::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material().vis_data(name, data, id());
}

Mat::So3Material& Discret::Elements::SolidScatra::solid_material(int nummat) const
{
  return *std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

FOUR_C_NAMESPACE_CLOSE