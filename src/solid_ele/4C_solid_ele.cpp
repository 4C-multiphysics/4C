// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_ele.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_solid_ele_calc_lib_integration.hpp"
#include "4C_solid_ele_factory.hpp"
#include "4C_solid_ele_interface_serializable.hpp"
#include "4C_solid_ele_line.hpp"
#include "4C_solid_ele_nullspace.hpp"
#include "4C_solid_ele_properties.hpp"
#include "4C_solid_ele_surface.hpp"
#include "4C_solid_ele_utils.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

namespace
{

  Core::IO::InputSpec get_kinem_type_input_spec()
  {
    return deprecated_selection<Solid::KinemType>("KINEM",
        {
            {kinem_type_string(Solid::KinemType::linear), Solid::KinemType::linear},
            {kinem_type_string(Solid::KinemType::nonlinearTotLag),
                Solid::KinemType::nonlinearTotLag},
        },
        {.description = "Whether to use linear kinematics (small displacements) or nonlinear "
                        "kinematics (large displacements)"});
  }

  template <Core::FE::CellType celltype>
  auto get_integration_rule_input_spec()
  {
    constexpr std::size_t dim = Core::FE::dim<celltype>;
    using GaussRule = std::conditional_t<dim == 3, Core::FE::GaussRule3D, Core::FE::GaussRule2D>;
    return group<Discret::Elements::SolidIntegrationRules<dim>>("INTEGRATION",
        {
            parameter<GaussRule>("RESIDUUM",
                {.description = "Gauss integration rule used to integrate the residuum and "
                                "its linearization",
                    .default_value = Discret::Elements::get_gauss_rule_stiffness_matrix<celltype>(),
                    .validator = Validators::in_set(std::set<GaussRule>{
                        begin(Discret::Elements::applicable_integration_rules<celltype>),
                        end(Discret::Elements::applicable_integration_rules<celltype>)}),
                    .store = in_struct(&Discret::Elements::SolidIntegrationRules<
                        Core::FE::dim<celltype>>::rule_residuum)}),
            parameter<GaussRule>("MASS",
                {.description = "Gauss integration rule used to integrate the mass matrix",
                    .default_value = Discret::Elements::get_gauss_rule_mass_matrix<celltype>(),

                    .validator = Validators::in_set(std::set<GaussRule>{
                        begin(Discret::Elements::applicable_integration_rules<celltype>),
                        end(Discret::Elements::applicable_integration_rules<celltype>)}),
                    .store = in_struct(&Discret::Elements::SolidIntegrationRules<
                        Core::FE::dim<celltype>>::rule_mass)}),
        },
        {.description = "Defines the integration rules for the solid element.", .required = false});
  }

  template <Core::FE::CellType celltype>
    requires(Core::FE::dim<celltype> == 2 || Core::FE::dim<celltype> == 3)
  auto get_default_input_spec()
  {
    constexpr auto dim = Core::FE::dim<celltype>;
    std::vector<Core::IO::InputSpec> specs{
        parameter<int>("MAT"),
        get_kinem_type_input_spec(),
        parameter<Discret::Elements::PrestressTechnology>(
            "PRESTRESS_TECH", {.description = "The technology used for prestressing",
                                  .default_value = Discret::Elements::PrestressTechnology::none}),
        parameter<std::optional<std::vector<double>>>("RAD", {.size = dim}),
        parameter<std::optional<std::vector<double>>>("AXI", {.size = dim}),
        parameter<std::optional<std::vector<double>>>("CIR", {.size = dim}),
        parameter<std::optional<std::vector<double>>>("FIBER1", {.size = dim}),
        parameter<std::optional<std::vector<double>>>("FIBER2", {.size = dim}),
        parameter<std::optional<std::vector<double>>>("FIBER3", {.size = dim}),
        get_integration_rule_input_spec<celltype>(),
    };
    if constexpr (dim == 2)
    {
      specs.emplace_back(parameter<double>(
          "THICKNESS", {.description = "Reference thickness of the 2D solid element"}));
      specs.emplace_back(parameter<Discret::Elements::PlaneAssumption>(
          "PLANE_ASSUMPTION", {.description = "Plane assumption for the 2D solid element"}));
    }
    return all_of(std::move(specs));
  }
}  // namespace

template <unsigned dim>
Discret::Elements::SolidType<dim> Discret::Elements::SolidType<dim>::instance_;

template <unsigned dim>
Discret::Elements::SolidType<dim>& Discret::Elements::SolidType<dim>::instance()
{
  return instance_;
}

template <unsigned dim>
void Discret::Elements::SolidType<dim>::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SOLID"];

  if constexpr (dim == 2)
  {
    defsgeneral[Core::FE::CellType::quad4] = all_of({
        get_default_input_spec<Core::FE::CellType::quad4>(),
        parameter<ElementTechnology>("TECH", {.default_value = ElementTechnology::none}),
    });
    defsgeneral[Core::FE::CellType::quad8] = get_default_input_spec<Core::FE::CellType::quad8>();
    defsgeneral[Core::FE::CellType::quad9] = get_default_input_spec<Core::FE::CellType::quad9>();
    defsgeneral[Core::FE::CellType::tri3] = get_default_input_spec<Core::FE::CellType::tri3>();
    defsgeneral[Core::FE::CellType::tri6] = get_default_input_spec<Core::FE::CellType::tri6>();

    defsgeneral[Core::FE::CellType::nurbs9] = get_default_input_spec<Core::FE::CellType::nurbs9>();
  }
  else if constexpr (dim == 3)
  {
    defsgeneral[Core::FE::CellType::hex8] = all_of({
        get_default_input_spec<Core::FE::CellType::hex8>(),
        parameter<ElementTechnology>("TECH", {.default_value = ElementTechnology::none}),
    });

    defsgeneral[Core::FE::CellType::hex18] = get_default_input_spec<Core::FE::CellType::hex18>();

    defsgeneral[Core::FE::CellType::hex20] = get_default_input_spec<Core::FE::CellType::hex20>();

    defsgeneral[Core::FE::CellType::hex27] = get_default_input_spec<Core::FE::CellType::hex27>();

    defsgeneral[Core::FE::CellType::tet4] = get_default_input_spec<Core::FE::CellType::tet4>();

    defsgeneral[Core::FE::CellType::tet10] = get_default_input_spec<Core::FE::CellType::tet10>();

    defsgeneral[Core::FE::CellType::wedge6] = all_of({
        get_default_input_spec<Core::FE::CellType::wedge6>(),
        deprecated_selection<ElementTechnology>("TECH",
            {
                {element_technology_string(ElementTechnology::none), ElementTechnology::none},
                {element_technology_string(ElementTechnology::shell_ans),
                    ElementTechnology::shell_ans},
                {element_technology_string(ElementTechnology::shell_eas_ans),
                    ElementTechnology::shell_eas_ans},
            },
            {.default_value = ElementTechnology::none}),
    });

    defsgeneral[Core::FE::CellType::pyramid5] = all_of({
        get_default_input_spec<Core::FE::CellType::pyramid5>(),
        deprecated_selection<ElementTechnology>("TECH",
            {
                {element_technology_string(ElementTechnology::none), ElementTechnology::none},
                {element_technology_string(ElementTechnology::fbar), ElementTechnology::fbar},
            },
            {.default_value = ElementTechnology::none}),
    });

    defsgeneral[Core::FE::CellType::nurbs27] =
        get_default_input_spec<Core::FE::CellType::nurbs27>();
  }
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidType<dim>::create(
    const std::string& eletype, Core::FE::CellType celltype, const int id, const int owner)
{
  if (eletype == "SOLID" && Core::FE::get_dimension(celltype) == dim) return create(id, owner);
  return nullptr;
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidType<dim>::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Solid<dim>>(id, owner);
}

template <unsigned dim>
Core::Communication::ParObject* Discret::Elements::SolidType<dim>::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::Solid<dim>(-1, -1);
  object->unpack(buffer);
  return object;
}

template <unsigned dim>
void Discret::Elements::SolidType<dim>::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  FourC::Solid::Utils::nodal_block_information_solid(dwele, numdf, dimns);
}

template <unsigned dim>
Core::LinAlg::SerialDenseMatrix Discret::Elements::SolidType<dim>::compute_null_space(
    Core::Nodes::Node& node, std::span<const double> x0, const int numdof)
{
  return compute_solid_null_space<dim>(node.x(), x0);
}

template <unsigned dim>
Discret::Elements::Solid<dim>::Solid(int id, int owner) : Core::Elements::Element(id, owner)
{
}


template <unsigned dim>
Core::Elements::Element* Discret::Elements::Solid<dim>::clone() const
{
  return new Solid(*this);
}

template <unsigned dim>
int Discret::Elements::Solid<dim>::num_line() const
{
  return Core::FE::get_number_of_element_lines(celltype_);
}

template <unsigned dim>
int Discret::Elements::Solid<dim>::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(celltype_);
}

template <unsigned dim>
int Discret::Elements::Solid<dim>::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(celltype_);
}

template <unsigned dim>
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Solid<dim>::lines()
{
  return Core::Communication::get_element_lines<SolidLine<dim>, Solid>(*this);
}

template <unsigned dim>
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Solid<dim>::surfaces()
{
  if constexpr (dim == 2)
  {
    // return the element itself if we are a surface
    return {Core::Utils::shared_ptr_from_ref(*this)};
  };
  return Core::Communication::get_element_surfaces<SolidSurface, Solid>(*this);
}

template <unsigned dim>
const Core::FE::GaussIntegration& Discret::Elements::Solid<dim>::get_gauss_rule() const
{
  FOUR_C_ASSERT(solid_calc_variant_.has_value(),
      "The solid calculation interface is not initialized for element id {}.", id());
  return std::visit([](auto& interface) -> const Core::FE::GaussIntegration&
      { return interface->get_gauss_rule_stiffness_integration(); }, *solid_calc_variant_);
}

template <unsigned dim>
void Discret::Elements::Solid<dim>::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  // add base class Element
  Core::Elements::Element::pack(data);

  add_to_pack(data, celltype_);
  add_to_pack(data, integration_rules_);

  Core::Communication::add_to_pack(data, solid_ele_property_);

  data.add_to_pack(material_post_setup_);

  FOUR_C_ASSERT(solid_calc_variant_.has_value(),
      "The solid calculation interface is not initialized for element id {}. The element needs "
      "to "
      "be fully setup before packing.",
      id());
  Discret::Elements::pack(*solid_calc_variant_, data);
}

template <unsigned dim>
void Discret::Elements::Solid<dim>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);

  extract_from_pack(buffer, celltype_);
  FOUR_C_ASSERT_ALWAYS(Core::FE::get_dimension(celltype_) == dim,
      "You try to create a solid element of dimension {} with a cell type {} of dimension {} "
      "that "
      "does not match.",
      dim, celltype_, Core::FE::get_dimension(celltype_));
  extract_from_pack(buffer, integration_rules_);

  Core::Communication::extract_from_pack(buffer, solid_ele_property_);

  extract_from_pack(buffer, material_post_setup_);

  // reset solid interface
  solid_calc_variant_ =
      create_solid_calculation_interface(celltype_, solid_ele_property_, integration_rules_);


  Discret::Elements::unpack(*solid_calc_variant_, buffer);
}

template <unsigned dim>
void Discret::Elements::Solid<dim>::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
  {
    interface_ptr_ = std::dynamic_pointer_cast<FourC::Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  }
  else
    interface_ptr_ = nullptr;
}

template <unsigned dim>
bool Discret::Elements::Solid<dim>::read_element(const std::string& eletype,
    Core::FE::CellType celltype, const Core::IO::InputParameterContainer& container,
    const Core::IO::MeshInput::ElementDataFromCellData& element_data)
{
  // set cell type
  celltype_ = celltype;
  FOUR_C_ASSERT_ALWAYS(Core::FE::get_dimension(celltype_) == dim,
      "You try to create a solid element of dimension {} with a cell type {} of dimension {} "
      "that "
      "does not match.",
      dim, celltype_, Core::FE::get_dimension(celltype_));

  // read number of material model
  set_material(0, Mat::factory(FourC::Solid::Utils::ReadElement::read_element_material(container)));

  solid_ele_property_ =
      FourC::Solid::Utils::ReadElement::read_solid_element_properties<dim>(container);
  integration_rules_ = container.get<SolidIntegrationRules<dim>>("INTEGRATION");

  solid_calc_variant_ =
      create_solid_calculation_interface(celltype_, solid_ele_property_, integration_rules_);

  std::visit([&](auto& interface) { interface->setup(*solid_material(), container); },
      *solid_calc_variant_);
  return true;
}

template <unsigned dim>
std::shared_ptr<Mat::So3Material> Discret::Elements::Solid<dim>::solid_material(int nummat) const
{
  return std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

template <unsigned dim>
void Discret::Elements::Solid<dim>::vis_names(std::map<std::string, int>& names) const
{
  Core::Elements::Element::vis_names(names);
  solid_material()->vis_names(names);
}

template <unsigned dim>
bool Discret::Elements::Solid<dim>::vis_data(
    const std::string& name, std::vector<double>& data) const
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, id());
}

template <unsigned dim>
void Discret::Elements::Solid<dim>::for_each_gauss_point(Core::FE::Discretization& discretization,
    std::vector<int>& lm,
    const std::function<void(Mat::So3Material&, double, int)>& integrator) const
{
  FOUR_C_ASSERT(solid_calc_variant_.has_value(),
      "The solid calculation interface is not initialized for element id {}.", id());
  std::visit(
      [&](auto& interface)
      {
        interface->for_each_gauss_point(*this, *solid_material(), discretization, lm, integrator);
      },
      *solid_calc_variant_);
}

template class Discret::Elements::SolidType<2>;
template class Discret::Elements::Solid<2>;
template class Discret::Elements::SolidType<3>;
template class Discret::Elements::Solid<3>;

FOUR_C_NAMESPACE_CLOSE
