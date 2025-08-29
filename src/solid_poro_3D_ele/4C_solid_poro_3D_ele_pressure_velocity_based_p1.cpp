// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_3D_ele_pressure_velocity_based_p1.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fluid_ele_nullspace.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_solid_3D_ele_interface_serializable.hpp"
#include "4C_solid_3D_ele_line.hpp"
#include "4C_solid_3D_ele_properties.hpp"
#include "4C_solid_3D_ele_surface.hpp"
#include "4C_solid_3D_ele_utils.hpp"
#include "4C_solid_poro_3D_ele_factory.hpp"

#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

namespace Discret::Elements::SolidPoroPressureVelocityBasedInternal
{
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
          parameter<std::optional<std::vector<double>>>("POROANISODIR1", {.size = 3}),
          parameter<std::optional<std::vector<double>>>("POROANISODIR2", {.size = 3}),
          parameter<std::optional<std::vector<double>>>("POROANISODIR3", {.size = 3}),
      });
    }
  }  // namespace
}  // namespace Discret::Elements::SolidPoroPressureVelocityBasedInternal

Discret::Elements::SolidPoroPressureVelocityBasedP1Type
    Discret::Elements::SolidPoroPressureVelocityBasedP1Type::instance_;

Discret::Elements::SolidPoroPressureVelocityBasedP1Type&
Discret::Elements::SolidPoroPressureVelocityBasedP1Type::instance()
{
  return instance_;
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SOLIDPORO_PRESSURE_VELOCITY_BASED_P1"];

  defsgeneral[Core::FE::CellType::hex8] = all_of({
      Discret::Elements::SolidPoroPressureVelocityBasedInternal::get_default_input_spec<
          Core::FE::CellType::hex8>(),
      parameter<std::optional<std::vector<double>>>(
          "POROANISONODALCOEFFS1", {.size = Core::FE::num_nodes(Core::FE::CellType::hex8)}),
      parameter<std::optional<std::vector<double>>>(
          "POROANISONODALCOEFFS2", {.size = Core::FE::num_nodes(Core::FE::CellType::hex8)}),
      parameter<std::optional<std::vector<double>>>(
          "POROANISONODALCOEFFS3", {.size = Core::FE::num_nodes(Core::FE::CellType::hex8)}),
  });

  defsgeneral[Core::FE::CellType::hex27] =
      Discret::Elements::SolidPoroPressureVelocityBasedInternal::get_default_input_spec<
          Core::FE::CellType::hex27>();


  defsgeneral[Core::FE::CellType::tet4] = all_of({
      Discret::Elements::SolidPoroPressureVelocityBasedInternal::get_default_input_spec<
          Core::FE::CellType::tet4>(),
      parameter<std::optional<std::vector<double>>>(
          "POROANISONODALCOEFFS1", {.size = Core::FE::num_nodes(Core::FE::CellType::tet4)}),
      parameter<std::optional<std::vector<double>>>(
          "POROANISONODALCOEFFS2", {.size = Core::FE::num_nodes(Core::FE::CellType::tet4)}),
      parameter<std::optional<std::vector<double>>>(
          "POROANISONODALCOEFFS3", {.size = Core::FE::num_nodes(Core::FE::CellType::tet4)}),
  });



  defsgeneral[Core::FE::CellType::tet10] =
      Discret::Elements::SolidPoroPressureVelocityBasedInternal::get_default_input_spec<
          Core::FE::CellType::tet10>();
}


std::shared_ptr<Core::Elements::Element>
Discret::Elements::SolidPoroPressureVelocityBasedP1Type::create(
    const std::string eletype, const std::string elecelltype, const int id, const int owner)
{
  if (eletype == "SOLIDPORO_PRESSURE_VELOCITY_BASED_P1") return create(id, owner);
  return nullptr;
}

std::shared_ptr<Core::Elements::Element>
Discret::Elements::SolidPoroPressureVelocityBasedP1Type::create(const int id, const int owner)
{
  return std::make_shared<Discret::Elements::SolidPoroPressureVelocityBasedP1>(id, owner);
}

Core::Communication::ParObject* Discret::Elements::SolidPoroPressureVelocityBasedP1Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::SolidPoroPressureVelocityBasedP1(-1, -1);
  object->unpack(buffer);
  return object;
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 4;
  dimns = 4;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix
Discret::Elements::SolidPoroPressureVelocityBasedP1Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

Discret::Elements::SolidPoroPressureVelocityBasedP1::SolidPoroPressureVelocityBasedP1(
    int id, int owner)
    : Core::Elements::Element(id, owner)
{
}

Core::Elements::Element* Discret::Elements::SolidPoroPressureVelocityBasedP1::clone() const
{
  return new Discret::Elements::SolidPoroPressureVelocityBasedP1(*this);
}

int Discret::Elements::SolidPoroPressureVelocityBasedP1::num_line() const
{
  return Core::FE::get_number_of_element_lines(celltype_);
}

int Discret::Elements::SolidPoroPressureVelocityBasedP1::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(celltype_);
}

int Discret::Elements::SolidPoroPressureVelocityBasedP1::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(celltype_);
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::SolidPoroPressureVelocityBasedP1::lines()
{
  return Core::Communication::get_element_lines<SolidLine<3>, SolidPoroPressureVelocityBasedP1>(
      *this);
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::SolidPoroPressureVelocityBasedP1::surfaces()
{
  return Core::Communication::get_element_surfaces<SolidSurface, SolidPoroPressureVelocityBasedP1>(
      *this);
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1::set_params_interface_ptr(
    const Teuchos::ParameterList& p)
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

bool Discret::Elements::SolidPoroPressureVelocityBasedP1::read_element(const std::string& eletype,
    const std::string& elecelltype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  // set cell type
  celltype_ = Core::FE::string_to_cell_type(elecelltype);

  // set anisotropic_properties
  anisotropic_permeability_property_.directions_.resize(3);
  anisotropic_permeability_property_.nodal_coeffs_.resize(3);

  // read number of material model
  set_material(0, Mat::factory(Solid::Utils::ReadElement::read_element_material(container)));

  // read kinematic type
  solid_ele_property_.kintype = container.get<Inpar::Solid::KinemType>("KINEM");


  read_anisotropic_permeability_directions_from_element_line_definition(container);
  read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(container);

  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_);
  solidporo_press_vel_based_calc_variant_ =
      create_solid_poro_pressure_velocity_based_p1_calculation_interface(celltype_);

  // setup solid material
  std::visit(
      [&](auto& solid) { solid->setup(struct_poro_material(), container); }, solid_calc_variant_);

  // setup poro material
  std::visit([&](auto& solidporopressurevelocitybased)
      { solidporopressurevelocitybased->poro_setup(struct_poro_material(), container); },
      solidporo_press_vel_based_calc_variant_);

  return true;
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1::
    read_anisotropic_permeability_directions_from_element_line_definition(
        const Core::IO::InputParameterContainer& container)
{
  for (int dim = 0; dim < 3; ++dim)
  {
    std::string definition_name = "POROANISODIR" + std::to_string(dim + 1);
    const auto& dir = container.get<std::optional<std::vector<double>>>(definition_name);
    anisotropic_permeability_property_.directions_[dim] = dir ? *dir : std::vector<double>(3, 0.0);
  }
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1::
    read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(
        const Core::IO::InputParameterContainer& container)
{
  for (int dim = 0; dim < 3; ++dim)
  {
    std::string definition_name = "POROANISONODALCOEFFS" + std::to_string(dim + 1);

    if (const auto& coeffs = container.get<std::optional<std::vector<double>>>(definition_name);
        coeffs && coeffs.has_value())
      anisotropic_permeability_property_.nodal_coeffs_[dim] = *coeffs;
    else
      anisotropic_permeability_property_.nodal_coeffs_[dim] = std::vector<double>(num_node(), 0.0);
  }
}


Mat::So3Material& Discret::Elements::SolidPoroPressureVelocityBasedP1::solid_poro_material(
    int nummat) const
{
  return *std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  // add base class Element
  Core::Elements::Element::pack(data);

  add_to_pack(data, celltype_);

  Discret::Elements::add_to_pack(data, solid_ele_property_);
  Core::Communication::add_to_pack(data, poro_ele_property_.impltype);

  data.add_to_pack(material_post_setup_);

  // anisotropic_permeability_directions_
  auto size = static_cast<int>(anisotropic_permeability_property_.directions_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i)
    add_to_pack(data, anisotropic_permeability_property_.directions_[i]);

  // anisotropic_permeability_nodal_coeffs_
  size = static_cast<int>(anisotropic_permeability_property_.nodal_coeffs_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i)
    add_to_pack(data, anisotropic_permeability_property_.nodal_coeffs_[i]);

  // optional data, e.g., EAS data
  Discret::Elements::pack(solid_calc_variant_, data);
  Discret::Elements::pack(solidporo_press_vel_based_calc_variant_, data);

  add_to_pack(data, initial_porosity_);
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);

  extract_from_pack(buffer, celltype_);

  Discret::Elements::extract_from_pack(buffer, solid_ele_property_);
  Core::Communication::extract_from_pack(buffer, poro_ele_property_.impltype);

  extract_from_pack(buffer, material_post_setup_);

  // anisotropic_permeability_directions_
  int size = 0;
  extract_from_pack(buffer, size);
  anisotropic_permeability_property_.directions_.resize(size, std::vector<double>(3, 0.0));
  for (int i = 0; i < size; ++i)
    extract_from_pack(buffer, anisotropic_permeability_property_.directions_[i]);

  // anisotropic_permeability_nodal_coeffs_
  size = 0;
  extract_from_pack(buffer, size);
  anisotropic_permeability_property_.nodal_coeffs_.resize(
      size, std::vector<double>(this->num_node(), 0.0));
  for (int i = 0; i < size; ++i)
    extract_from_pack(buffer, anisotropic_permeability_property_.nodal_coeffs_[i]);

  // reset solid and poro interfaces
  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_);
  solidporo_press_vel_based_calc_variant_ =
      create_solid_poro_pressure_velocity_based_p1_calculation_interface(celltype_);

  Discret::Elements::unpack(solid_calc_variant_, buffer);
  Discret::Elements::unpack(solidporo_press_vel_based_calc_variant_, buffer);


  extract_from_pack(buffer, initial_porosity_);
}

void Discret::Elements::SolidPoroPressureVelocityBasedP1::vis_names(
    std::map<std::string, int>& names)
{
  Core::Elements::Element::vis_names(names);
  solid_poro_material().vis_names(names);
}

bool Discret::Elements::SolidPoroPressureVelocityBasedP1::vis_data(
    const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_poro_material().vis_data(name, data, id());
}

Mat::StructPoro& Discret::Elements::SolidPoroPressureVelocityBasedP1::struct_poro_material(
    int nummat) const
{
  auto porostruct_mat =
      std::dynamic_pointer_cast<Mat::StructPoro>(Core::Elements::Element::material(nummat));

  if (porostruct_mat == nullptr) FOUR_C_THROW("cast to poro material failed");

  if (porostruct_mat->material_type() != Core::Materials::m_structporo and
      porostruct_mat->material_type() != Core::Materials::m_structpororeaction and
      porostruct_mat->material_type() != Core::Materials::m_structpororeactionECM)
    FOUR_C_THROW("invalid structure material for poroelasticity");

  return *porostruct_mat;
}


Mat::FluidPoro& Discret::Elements::SolidPoroPressureVelocityBasedP1::fluid_poro_material(
    int nummat) const
{
  if (this->num_material() <= 1)
  {
    FOUR_C_THROW("No second material defined for SolidPoroPressureVelocityBased element {}", id());
  }

  auto fluidmulti_mat =
      std::dynamic_pointer_cast<Mat::FluidPoro>(Core::Elements::Element::material(1));

  if (fluidmulti_mat == nullptr) FOUR_C_THROW("cast to multiphase fluid poro material failed");
  if (fluidmulti_mat->material_type() != Core::Materials::m_fluidporo)
    FOUR_C_THROW("invalid fluid material for poroelasticity");
  return *fluidmulti_mat;
}

FOUR_C_NAMESPACE_CLOSE
