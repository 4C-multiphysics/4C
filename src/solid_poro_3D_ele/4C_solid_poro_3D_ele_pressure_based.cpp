// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_3D_ele_pressure_based.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_solid_3D_ele_factory.hpp"
#include "4C_solid_3D_ele_interface_serializable.hpp"
#include "4C_solid_3D_ele_line.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"
#include "4C_solid_3D_ele_surface.hpp"
#include "4C_solid_3D_ele_utils.hpp"
#include "4C_solid_poro_3D_ele_factory.hpp"
#include "4C_solid_poro_3D_ele_utils.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

namespace Discret::Elements::SolidPoroPressureBasedInternal
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
          deprecated_selection<Inpar::ScaTra::ImplType>("TYPE",
              Discret::Elements::get_impltype_inpar_map(),
              {.description = "Scalar transport implementation type",
                  .default_value = Inpar::ScaTra::ImplType::impltype_undefined}),
      });
    }
  }  // namespace
}  // namespace Discret::Elements::SolidPoroPressureBasedInternal

Discret::Elements::SolidPoroPressureBasedType
    Discret::Elements::SolidPoroPressureBasedType::instance_;

Discret::Elements::SolidPoroPressureBasedType&
Discret::Elements::SolidPoroPressureBasedType::instance()
{
  return instance_;
}

void Discret::Elements::SolidPoroPressureBasedType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SOLIDPORO_PRESSURE_BASED"];

  defsgeneral[Core::FE::CellType::hex8] =
      all_of({Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
          Core::FE::CellType::hex8>()});

  defsgeneral[Core::FE::CellType::hex27] =
      Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
          Core::FE::CellType::hex27>();


  defsgeneral[Core::FE::CellType::tet4] =
      Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
          Core::FE::CellType::tet4>();

  defsgeneral[Core::FE::CellType::tet10] =
      Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
          Core::FE::CellType::tet10>();
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidPoroPressureBasedType::create(
    const std::string eletype, const std::string elecelltype, const int id, const int owner)
{
  if (eletype == "SOLIDPORO_PRESSURE_BASED") return create(id, owner);
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidPoroPressureBasedType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::SolidPoroPressureBased>(id, owner);
}

Core::Communication::ParObject* Discret::Elements::SolidPoroPressureBasedType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::SolidPoroPressureBased(-1, -1);
  object->unpack(buffer);
  return object;
}

void Discret::Elements::SolidPoroPressureBasedType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  Solid::Utils::nodal_block_information_solid(dwele, numdf, dimns);
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::SolidPoroPressureBasedType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<3>(node.x(), x0);
}

Discret::Elements::SolidPoroPressureBased::SolidPoroPressureBased(int id, int owner)
    : Core::Elements::Element(id, owner)
{
}

Core::Elements::Element* Discret::Elements::SolidPoroPressureBased::clone() const
{
  return new Discret::Elements::SolidPoroPressureBased(*this);
}

int Discret::Elements::SolidPoroPressureBased::num_line() const
{
  return Core::FE::get_number_of_element_lines(celltype_);
}

int Discret::Elements::SolidPoroPressureBased::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(celltype_);
}

int Discret::Elements::SolidPoroPressureBased::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(celltype_);
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::SolidPoroPressureBased::lines()
{
  return Core::Communication::get_element_lines<SolidLine<3>, SolidPoroPressureBased>(*this);
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::SolidPoroPressureBased::surfaces()
{
  return Core::Communication::get_element_surfaces<SolidSurface, SolidPoroPressureBased>(*this);
}

void Discret::Elements::SolidPoroPressureBased::set_params_interface_ptr(
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

bool Discret::Elements::SolidPoroPressureBased::read_element(const std::string& eletype,
    const std::string& elecelltype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  // set cell type
  celltype_ = Core::FE::string_to_cell_type(elecelltype);

  // read number of material model
  set_material(0, Mat::factory(Solid::Utils::ReadElement::read_element_material(container)));

  // read kinematic type
  solid_ele_property_.kintype = container.get<Inpar::Solid::KinemType>("KINEM");

  // read scalar transport implementation type
  poro_ele_property_.impltype = container.get<Inpar::ScaTra::ImplType>("TYPE");


  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_);
  solidporo_press_based_calc_variant_ =
      create_solid_poro_pressure_based_calculation_interface(celltype_);

  // setup solid material
  std::visit(
      [&](auto& solid) { solid->setup(struct_poro_material(), container); }, solid_calc_variant_);

  // setup poro material
  std::visit([&](auto& solidporopressurebased)
      { solidporopressurebased->poro_setup(struct_poro_material(), container); },
      solidporo_press_based_calc_variant_);

  return true;
}

Mat::So3Material& Discret::Elements::SolidPoroPressureBased::solid_poro_material(int nummat) const
{
  return *std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

void Discret::Elements::SolidPoroPressureBased::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  // add base class Element
  Core::Elements::Element::pack(data);

  add_to_pack(data, celltype_);

  Discret::Elements::add_to_pack(data, solid_ele_property_);

  add_to_pack(data, poro_ele_property_.impltype);

  data.add_to_pack(material_post_setup_);

  // optional data, e.g., EAS data
  Discret::Elements::pack(solid_calc_variant_, data);
  Discret::Elements::pack(solidporo_press_based_calc_variant_, data);
}

void Discret::Elements::SolidPoroPressureBased::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);

  extract_from_pack(buffer, celltype_);

  Discret::Elements::extract_from_pack(buffer, solid_ele_property_);

  extract_from_pack(buffer, poro_ele_property_.impltype);

  extract_from_pack(buffer, material_post_setup_);

  // reset solid and poro interfaces
  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_);
  solidporo_press_based_calc_variant_ =
      create_solid_poro_pressure_based_calculation_interface(celltype_);

  Discret::Elements::unpack(solid_calc_variant_, buffer);
  Discret::Elements::unpack(solidporo_press_based_calc_variant_, buffer);
}

void Discret::Elements::SolidPoroPressureBased::vis_names(std::map<std::string, int>& names)
{
  Core::Elements::Element::vis_names(names);
  solid_poro_material().vis_names(names);
}

bool Discret::Elements::SolidPoroPressureBased::vis_data(
    const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_poro_material().vis_data(name, data, id());
}

Mat::StructPoro& Discret::Elements::SolidPoroPressureBased::struct_poro_material(int nummat) const
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
Mat::FluidPoroMultiPhase& Discret::Elements::SolidPoroPressureBased::fluid_poro_material(
    int nummat) const
{
  if (this->num_material() <= 1)
  {
    FOUR_C_THROW("No second material defined for SolidPoroPressureBased element {}", id());
  }

  auto fluidmulti_mat =
      std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(Core::Elements::Element::material(1));

  if (fluidmulti_mat == nullptr) FOUR_C_THROW("cast to multiphase fluid poro material failed");
  if (fluidmulti_mat->material_type() != Core::Materials::m_fluidporo_multiphase and
      fluidmulti_mat->material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW("invalid fluid material for poro-multiphase-elasticity");
  if (fluidmulti_mat->num_fluid_phases() == 0)
  {
    FOUR_C_THROW(
        "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE = 0 currently not supported since this requires "
        "an adaption of the definition of the solid pressure");
  }
  return *fluidmulti_mat;
}

FOUR_C_NAMESPACE_CLOSE
