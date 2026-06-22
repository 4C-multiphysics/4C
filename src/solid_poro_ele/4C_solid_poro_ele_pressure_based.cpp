// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_poro_ele_pressure_based.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_scatra_input.hpp"
#include "4C_solid_ele_calc_lib_integration.hpp"
#include "4C_solid_ele_factory.hpp"
#include "4C_solid_ele_interface_serializable.hpp"
#include "4C_solid_ele_line.hpp"
#include "4C_solid_ele_nullspace.hpp"
#include "4C_solid_ele_surface.hpp"
#include "4C_solid_ele_utils.hpp"
#include "4C_solid_poro_ele_factory.hpp"
#include "4C_solid_poro_ele_utils.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

namespace Discret::Elements::SolidPoroPressureBasedInternal
{
  namespace
  {
    template <Core::FE::CellType celltype>
      requires(Core::FE::dim<celltype> == 3)
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
          deprecated_selection<ScaTra::ImplType>("TYPE",
              Discret::Elements::get_impltype_inpar_map(),
              {.description = "Scalar transport implementation type",
                  .default_value = ScaTra::ImplType::impltype_undefined}),
      });
    }


    template <Core::FE::CellType celltype>
      requires(Core::FE::dim<celltype> == 2)
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
          deprecated_selection<ScaTra::ImplType>("TYPE",
              Discret::Elements::get_impltype_inpar_map(),
              {.description = "Scalar transport implementation type",
                  .default_value = ScaTra::ImplType::impltype_undefined}),
          parameter<double>(
              "THICKNESS", {.description = "Reference thickness of the 2D solid element"}),
          parameter<Discret::Elements::PlaneAssumption>("PLANE_ASSUMPTION",
              {.description = "Plane assumption for the 2D solid element (Note: In solid-poro, "
                              "only plane strain makes physical sense)",
                  .validator = Validators::in_set(std::set<Discret::Elements::PlaneAssumption>{
                      Discret::Elements::PlaneAssumption::plane_strain})}),
      });
    }
  }  // namespace
}  // namespace Discret::Elements::SolidPoroPressureBasedInternal

template <unsigned dim>
Discret::Elements::SolidPoroPressureBasedType<dim>
    Discret::Elements::SolidPoroPressureBasedType<dim>::instance_;

template <unsigned dim>
Discret::Elements::SolidPoroPressureBasedType<dim>&
Discret::Elements::SolidPoroPressureBasedType<dim>::instance()
{
  return instance_;
}

template <unsigned dim>
void Discret::Elements::SolidPoroPressureBasedType<dim>::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SOLIDPORO_PRESSURE_BASED"];
  if constexpr (dim == 2)
  {
    defsgeneral[Core::FE::CellType::quad4] =
        all_of({Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
            Core::FE::CellType::quad4>()});

    defsgeneral[Core::FE::CellType::quad8] =
        Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
            Core::FE::CellType::quad8>();


    defsgeneral[Core::FE::CellType::quad9] =
        Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
            Core::FE::CellType::quad9>();

    defsgeneral[Core::FE::CellType::tri3] =
        Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
            Core::FE::CellType::tri3>();

    defsgeneral[Core::FE::CellType::tri6] =
        Discret::Elements::SolidPoroPressureBasedInternal::get_default_input_spec<
            Core::FE::CellType::tri6>();
  }
  else
  {
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
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidPoroPressureBasedType<dim>::create(
    const std::string& eletype, Core::FE::CellType celltype, const int id, const int owner)
{
  if (eletype == "SOLIDPORO_PRESSURE_BASED" && Core::FE::get_dimension(celltype) == dim)
    return create(id, owner);
  return nullptr;
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidPoroPressureBasedType<dim>::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::SolidPoroPressureBased<dim>>(id, owner);
}

template <unsigned dim>
Core::Communication::ParObject* Discret::Elements::SolidPoroPressureBasedType<dim>::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::SolidPoroPressureBased<dim>(-1, -1);
  object->unpack(buffer);
  return object;
}

template <unsigned dim>
void Discret::Elements::SolidPoroPressureBasedType<dim>::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  Solid::Utils::nodal_block_information_solid(dwele, numdf, dimns);
}

template <unsigned dim>
Core::LinAlg::SerialDenseMatrix
Discret::Elements::SolidPoroPressureBasedType<dim>::compute_null_space(
    Core::Nodes::Node& node, std::span<const double> x0, const int numdof)
{
  return compute_solid_null_space<dim>(node.x(), x0);
}

template <unsigned dim>
Discret::Elements::SolidPoroPressureBased<dim>::SolidPoroPressureBased(int id, int owner)
    : Core::Elements::Element(id, owner)
{
}

template <unsigned dim>
Core::Elements::Element* Discret::Elements::SolidPoroPressureBased<dim>::clone() const
{
  return new Discret::Elements::SolidPoroPressureBased<dim>(*this);
}

template <unsigned dim>
int Discret::Elements::SolidPoroPressureBased<dim>::num_line() const
{
  return Core::FE::get_number_of_element_lines(celltype_);
}

template <unsigned dim>
int Discret::Elements::SolidPoroPressureBased<dim>::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(celltype_);
}

template <unsigned dim>
int Discret::Elements::SolidPoroPressureBased<dim>::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(celltype_);
}

template <unsigned dim>
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::SolidPoroPressureBased<dim>::lines()
{
  return Core::Communication::get_element_lines<SolidLine<dim>, SolidPoroPressureBased<dim>>(*this);
}

template <unsigned dim>
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::SolidPoroPressureBased<dim>::surfaces()
{
  if constexpr (dim == 2)
  {
    // return the element itself if we are a surface
    return {Core::Utils::shared_ptr_from_ref(*this)};
  };
  return Core::Communication::get_element_surfaces<SolidSurface, SolidPoroPressureBased<dim>>(
      *this);
}

template <unsigned dim>
void Discret::Elements::SolidPoroPressureBased<dim>::set_params_interface_ptr(
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

template <unsigned dim>
bool Discret::Elements::SolidPoroPressureBased<dim>::read_element(const std::string& eletype,
    Core::FE::CellType celltype, const Core::IO::InputParameterContainer& container,
    const Core::IO::MeshInput::ElementDataFromCellData& element_data)
{
  // read base element
  // set cell type
  celltype_ = celltype;

  // read number of material model
  set_material(0, Mat::factory(Solid::Utils::ReadElement::read_element_material(container)));

  // read solid element properties
  solid_ele_property_ =
      FourC::Solid::Utils::ReadElement::read_solid_element_properties<dim>(container);

  if constexpr (dim == 2)
  {
    // For 2D solid-poro, only plane-strain makes physical sense
    FOUR_C_ASSERT_ALWAYS(
        solid_ele_property_.plane_assumption == Discret::Elements::PlaneAssumption::plane_strain,
        "For 2D solid-poro elements, only plane strain assumption makes physical sense.");
  }

  // read scalar transport implementation type
  poro_ele_property_.impltype = container.get<ScaTra::ImplType>("TYPE");

  // get possible bodyforce contribution
  bodyforce_function_ = get_bodyforce_function_from_input();

  SolidIntegrationRules<dim> rules =
      Core::FE::cell_type_switch<Discret::Elements::ImplementedSolidCellTypes<dim>>(celltype_,
          [](auto celltype_t) -> SolidIntegrationRules<dim>
          { return make_default_solid_integration_rules<celltype_t()>(); });
  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_, rules);
  solidporo_press_based_calc_variant_ =
      create_solid_poro_pressure_based_calculation_interface<dim>(solid_ele_property_, celltype_);

  // setup solid material
  std::visit(
      [&](auto& solid) { solid->setup(struct_poro_material(), container); }, *solid_calc_variant_);

  // setup poro material
  std::visit([&](auto& solidporopressurebased)
      { solidporopressurebased->poro_setup(struct_poro_material(), container); },
      *solidporo_press_based_calc_variant_);

  return true;
}

template <unsigned dim>
Mat::So3Material& Discret::Elements::SolidPoroPressureBased<dim>::solid_poro_material(
    int nummat) const
{
  return *std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

template <unsigned dim>
void Discret::Elements::SolidPoroPressureBased<dim>::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  // add base class Element
  Core::Elements::Element::pack(data);

  add_to_pack(data, celltype_);

  Core::Communication::add_to_pack(data, solid_ele_property_);

  add_to_pack(data, poro_ele_property_.impltype);

  data.add_to_pack(material_post_setup_);

  // optional data, e.g., EAS data
  FOUR_C_ASSERT(solid_calc_variant_.has_value(),
      "The solid calculation interface is not initialized for element id {}. The element needs to "
      "be fully setup before packing.",
      id());
  Discret::Elements::pack(*solid_calc_variant_, data);
  FOUR_C_ASSERT(solidporo_press_based_calc_variant_.has_value(),
      "The solidporo calculation interface is not initialized for element id {}. The element needs "
      "to be fully setup before packing.",
      id());
  Discret::Elements::pack(*solidporo_press_based_calc_variant_, data);
}

template <unsigned dim>
void Discret::Elements::SolidPoroPressureBased<dim>::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);

  extract_from_pack(buffer, celltype_);

  Core::Communication::extract_from_pack(buffer, solid_ele_property_);

  extract_from_pack(buffer, poro_ele_property_.impltype);

  extract_from_pack(buffer, material_post_setup_);

  bodyforce_function_ = get_bodyforce_function_from_input();

  // reset solid and poro interfaces
  SolidIntegrationRules<dim> rules =
      Core::FE::cell_type_switch<Discret::Elements::ImplementedSolidCellTypes<dim>>(celltype_,
          [](auto celltype_t) -> SolidIntegrationRules<dim>
          { return make_default_solid_integration_rules<celltype_t()>(); });
  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_, rules);
  solidporo_press_based_calc_variant_ =
      create_solid_poro_pressure_based_calculation_interface<dim>(solid_ele_property_, celltype_);

  Discret::Elements::unpack(*solid_calc_variant_, buffer);
  Discret::Elements::unpack(*solidporo_press_based_calc_variant_, buffer);
}

template <unsigned dim>
void Discret::Elements::SolidPoroPressureBased<dim>::vis_names(std::map<std::string, int>& names)
{
  Core::Elements::Element::vis_names(names);
  solid_poro_material().vis_names(names);
}

template <unsigned dim>
bool Discret::Elements::SolidPoroPressureBased<dim>::vis_data(
    const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  const unsigned int dummy_gp = 0;
  return solid_poro_material().vis_data(name, data, dummy_gp, id());
}

template <unsigned dim>
Mat::StructPoro& Discret::Elements::SolidPoroPressureBased<dim>::struct_poro_material(
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

template <unsigned dim>
Mat::FluidPoroMultiPhase& Discret::Elements::SolidPoroPressureBased<dim>::fluid_poro_material(
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

template <unsigned dim>
std::optional<const Core::Utils::FunctionOfSpaceTime*>
Discret::Elements::SolidPoroPressureBased<dim>::get_bodyforce_function_from_input() const
{
  // get poro-solid parameter
  const Teuchos::ParameterList& solid_poro_params =
      Global::Problem::instance()->poro_multi_phase_dynamic_params();


  if (solid_poro_params.get<std::optional<int>>("body_force_function").has_value())
  {
    const auto& funct =
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
            solid_poro_params.get<std::optional<int>>("body_force_function").value());

    // safety check
    FOUR_C_ASSERT_ALWAYS(static_cast<int>(funct.number_components()) == dim,
        "The dimension of your bodyforce vector and the dimension of the problem must be "
        "equal!");

    return &funct;
  }
  else
  {
    return std::nullopt;
  }
}

template class Discret::Elements::SolidPoroPressureBasedType<2>;
template class Discret::Elements::SolidPoroPressureBased<2>;
template class Discret::Elements::SolidPoroPressureBasedType<3>;
template class Discret::Elements::SolidPoroPressureBased<3>;

FOUR_C_NAMESPACE_CLOSE
