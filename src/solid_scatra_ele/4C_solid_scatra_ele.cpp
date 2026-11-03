// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_scatra_ele.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_scatra_input.hpp"
#include "4C_solid_ele_interface_serializable.hpp"
#include "4C_solid_ele_line.hpp"
#include "4C_solid_ele_nullspace.hpp"
#include "4C_solid_ele_properties.hpp"
#include "4C_solid_ele_surface.hpp"
#include "4C_solid_scatra_ele_factory.hpp"
#include "4C_solid_scatra_ele_lib.hpp"
#include "4C_structure_new_input.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

namespace
{
  template <Core::FE::CellType celltype>
    requires(Core::FE::dim<celltype> == 2)
  auto get_default_input_spec()
  {
    return all_of({
        parameter<int>("MAT"),
        deprecated_selection<Solid::KinemType>("KINEM",
            {
                {kinem_type_string(Solid::KinemType::linear), Solid::KinemType::linear},
                {kinem_type_string(Solid::KinemType::nonlinearTotLag),
                    Solid::KinemType::nonlinearTotLag},
            },
            {.description = "Whether to use linear kinematics (small displacements) or nonlinear "
                            "kinematics (large displacements)"}),
        parameter<std::string>("TYPE"),
        parameter<Discret::Elements::PrestressTechnology>(
            "PRESTRESS_TECH", {.description = "The technology used for prestressing",
                                  .default_value = Discret::Elements::PrestressTechnology::none}),
        parameter<std::optional<std::vector<double>>>("RAD", {.size = 2}),
        parameter<std::optional<std::vector<double>>>("AXI", {.size = 2}),
        parameter<std::optional<std::vector<double>>>("CIR", {.size = 2}),
        parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 2}),
        parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 2}),
        parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 2}),
        parameter<double>(
            "THICKNESS", {.description = "Reference thickness of the 2D solid element"}),
        parameter<Discret::Elements::PlaneAssumption>(
            "PLANE_ASSUMPTION", {.description = "Plane assumption for the 2D solid element"}),

    });
  }

  template <Core::FE::CellType celltype>
    requires(Core::FE::dim<celltype> == 3)
  auto get_default_input_spec()
  {
    return all_of({
        parameter<int>("MAT"),
        deprecated_selection<Solid::KinemType>("KINEM",
            {
                {kinem_type_string(Solid::KinemType::linear), Solid::KinemType::linear},
                {kinem_type_string(Solid::KinemType::nonlinearTotLag),
                    Solid::KinemType::nonlinearTotLag},
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

template <unsigned dim>
Discret::Elements::SolidScatraType<dim> Discret::Elements::SolidScatraType<dim>::instance_;

template <unsigned dim>
Discret::Elements::SolidScatraType<dim>& Discret::Elements::SolidScatraType<dim>::instance()
{
  return instance_;
}

template <unsigned dim>
void Discret::Elements::SolidScatraType<dim>::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SOLIDSCATRA"];

  if constexpr (dim == 2)
  {
    defsgeneral[Core::FE::CellType::quad4] = all_of({
        get_default_input_spec<Core::FE::CellType::quad4>(),
        parameter<ElementTechnology>("TECH", {.default_value = ElementTechnology::none}),
    });
    defsgeneral[Core::FE::CellType::quad9] = get_default_input_spec<Core::FE::CellType::quad9>();
    defsgeneral[Core::FE::CellType::tri3] = get_default_input_spec<Core::FE::CellType::tri3>();
    defsgeneral[Core::FE::CellType::tri6] = get_default_input_spec<Core::FE::CellType::tri6>();
  }
  else
  {
    defsgeneral[Core::FE::CellType::hex8] = all_of({
        get_default_input_spec<Core::FE::CellType::hex8>(),
        parameter<ElementTechnology>("TECH", {.default_value = ElementTechnology::none}),
    });

    defsgeneral[Core::FE::CellType::hex27] = get_default_input_spec<Core::FE::CellType::hex27>();

    defsgeneral[Core::FE::CellType::tet4] = get_default_input_spec<Core::FE::CellType::tet4>();

    defsgeneral[Core::FE::CellType::tet10] = get_default_input_spec<Core::FE::CellType::tet10>();

    defsgeneral[Core::FE::CellType::nurbs27] =
        get_default_input_spec<Core::FE::CellType::nurbs27>();
  }
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidScatraType<dim>::create(
    const std::string& eletype, Core::FE::CellType celltype, const int id, const int owner)
{
  if (eletype == "SOLIDSCATRA" && Core::FE::get_dimension(celltype) == dim)
    return create(id, owner);
  return nullptr;
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidScatraType<dim>::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::SolidScatra<dim>>(id, owner);
}

template <unsigned dim>
Core::Communication::ParObject* Discret::Elements::SolidScatraType<dim>::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::SolidScatra<dim>(-1, -1);
  object->unpack(buffer);
  return object;
}

template <unsigned dim>
void Discret::Elements::SolidScatraType<dim>::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  Solid::Utils::nodal_block_information_solid(dwele, numdf, dimns);
}

template <unsigned dim>
Core::LinAlg::SerialDenseMatrix Discret::Elements::SolidScatraType<dim>::compute_null_space(
    Core::Nodes::Node& node, std::span<const double> x0, const int numdof)
{
  return compute_solid_null_space<dim>(node.x(), x0);
}

template <unsigned dim>
Discret::Elements::SolidScatra<dim>::SolidScatra(int id, int owner)
    : Core::Elements::Element(id, owner)
{
}

template <unsigned dim>
Core::Elements::Element* Discret::Elements::SolidScatra<dim>::clone() const
{
  return new Discret::Elements::SolidScatra<dim>(*this);
}

template <unsigned dim>
int Discret::Elements::SolidScatra<dim>::num_line() const
{
  return Core::FE::get_number_of_element_lines(celltype_);
}

template <unsigned dim>
int Discret::Elements::SolidScatra<dim>::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(celltype_);
}

template <unsigned dim>
int Discret::Elements::SolidScatra<dim>::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(celltype_);
}

template <unsigned dim>
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::SolidScatra<dim>::lines()
{
  return Core::Communication::get_element_lines<SolidLine<dim>, SolidScatra<dim>>(*this);
}

template <unsigned dim>
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::SolidScatra<dim>::surfaces()
{
  if constexpr (dim == 2)
  {
    // return the element itself if we are a surface
    return {Core::Utils::shared_ptr_from_ref(*this)};
  };
  return Core::Communication::get_element_surfaces<SolidSurface, SolidScatra<dim>>(*this);
}

template <unsigned dim>
void Discret::Elements::SolidScatra<dim>::set_params_interface_ptr(const Teuchos::ParameterList& p)
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
bool Discret::Elements::SolidScatra<dim>::read_element(const std::string& eletype,
    Core::FE::CellType celltype, const Core::IO::InputParameterContainer& container,
    const Core::IO::MeshInput::ElementDataFromCellData& element_data)
{
  // read base element
  // set cell type
  celltype_ = celltype;

  // read number of material model
  set_material(0, Mat::factory(Solid::Utils::ReadElement::read_element_material(container)));

  // read scalar transport implementation type
  properties_.impltype = read_scatra_impl_type(container);

  properties_.solid = Solid::Utils::ReadElement::read_solid_element_properties<dim>(container);

  solid_scatra_calc_variant_ =
      create_solid_scatra_calculation_interface(celltype_, properties_.solid);

  // setup solid material
  std::visit([&](auto& solid_scatra) { solid_scatra->setup(solid_material(), container); },
      *solid_scatra_calc_variant_);

  return true;
}

template <unsigned dim>
void Discret::Elements::SolidScatra<dim>::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  // add base class Element
  Core::Elements::Element::pack(data);

  add_to_pack(data, celltype_);
  Core::Communication::add_to_pack(data, properties_);

  data.add_to_pack(material_post_setup_);

  // optional data, e.g., EAS data
  FOUR_C_ASSERT(solid_scatra_calc_variant_.has_value(),
      "The solid-scatra calculation interface is not initialized for element id {}.", id());
  Discret::Elements::pack(*solid_scatra_calc_variant_, data);
}

template <unsigned dim>
void Discret::Elements::SolidScatra<dim>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);

  extract_from_pack(buffer, celltype_);

  Core::Communication::extract_from_pack(buffer, properties_);

  extract_from_pack(buffer, material_post_setup_);

  // reset solid and scatra interfaces
  solid_scatra_calc_variant_ =
      create_solid_scatra_calculation_interface(celltype_, properties_.solid);

  Discret::Elements::unpack(*solid_scatra_calc_variant_, buffer);
}

template <unsigned dim>
void Discret::Elements::SolidScatra<dim>::vis_names(std::map<std::string, int>& names) const
{
  Core::Elements::Element::vis_names(names);
  solid_material().vis_names(names);
}

template <unsigned dim>
bool Discret::Elements::SolidScatra<dim>::vis_data(
    const std::string& name, std::vector<double>& data) const
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material().vis_data(name, data, id());
}

template <unsigned dim>
Mat::So3Material& Discret::Elements::SolidScatra<dim>::solid_material(int nummat) const
{
  return *std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

template <unsigned dim>
bool Discret::Elements::SolidScatra<dim>::material_failed() const
{
  FOUR_C_ASSERT(solid_scatra_calc_variant_.has_value(),
      "The solid-scatra calculation interface is not initialized for element id {}.", id());
  return std::visit([&](auto& interface) { return interface->material_failed(solid_material()); },
      *solid_scatra_calc_variant_);
}

template class Discret::Elements::SolidScatraType<2>;
template class Discret::Elements::SolidScatra<2>;
template class Discret::Elements::SolidScatraType<3>;
template class Discret::Elements::SolidScatra<3>;

FOUR_C_NAMESPACE_CLOSE
