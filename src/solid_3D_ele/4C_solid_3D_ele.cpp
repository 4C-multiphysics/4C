// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_3D_ele.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_solid_3D_ele_factory.hpp"
#include "4C_solid_3D_ele_interface_serializable.hpp"
#include "4C_solid_3D_ele_line.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"
#include "4C_solid_3D_ele_properties.hpp"
#include "4C_solid_3D_ele_surface.hpp"
#include "4C_solid_3D_ele_utils.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

namespace
{

  Core::IO::InputSpec get_kinem_type_input_spec()
  {
    return deprecated_selection<Inpar::Solid::KinemType>("KINEM",
        {
            {kinem_type_string(Inpar::Solid::KinemType::linear), Inpar::Solid::KinemType::linear},
            {kinem_type_string(Inpar::Solid::KinemType::nonlinearTotLag),
                Inpar::Solid::KinemType::nonlinearTotLag},
        },
        {.description = "Whether to use linear kinematics (small displacements) or nonlinear "
                        "kinematics (large displacements)"});
  }

  template <Core::FE::CellType celltype>
  auto get_default_input_spec()
  {
    return all_of({
        parameter<int>("MAT"),
        get_kinem_type_input_spec(),
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

Discret::Elements::SolidType Discret::Elements::SolidType::instance_;

Discret::Elements::SolidType& Discret::Elements::SolidType::instance() { return instance_; }

void Discret::Elements::SolidType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SOLID"];

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

  defsgeneral[Core::FE::CellType::nurbs27] = all_of({
      parameter<int>("MAT"),
      get_kinem_type_input_spec(),
  });
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidType::create(
    const std::string eletype, const std::string elecelltype, const int id, const int owner)
{
  if (eletype == "SOLID") return create(id, owner);
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Solid>(id, owner);
}

Core::Communication::ParObject* Discret::Elements::SolidType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::Solid(-1, -1);
  object->unpack(buffer);
  return object;
}

void Discret::Elements::SolidType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  FourC::Solid::Utils::nodal_block_information_solid(dwele, numdf, dimns);
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::SolidType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  switch (numdof)
  {
    case 3:
      return compute_solid_null_space<3>(node.x(), x0);
    case 2:
      return compute_solid_null_space<2>(node.x(), x0);
    default:
      FOUR_C_THROW(
          "The null space computation of a solid element of dimension {} is not yet implemented",
          numdof);
  }
}

Discret::Elements::Solid::Solid(int id, int owner) : Core::Elements::Element(id, owner) {}


Core::Elements::Element* Discret::Elements::Solid::clone() const { return new Solid(*this); }

int Discret::Elements::Solid::num_line() const
{
  return Core::FE::get_number_of_element_lines(celltype_);
}

int Discret::Elements::Solid::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(celltype_);
}

int Discret::Elements::Solid::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(celltype_);
}

std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Solid::lines()
{
  return Core::Communication::get_element_lines<SolidLine<3>, Solid>(*this);
}

std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Solid::surfaces()
{
  return Core::Communication::get_element_surfaces<SolidSurface, Solid>(*this);
}

const Core::FE::GaussIntegration& Discret::Elements::Solid::get_gauss_rule() const
{
  return std::visit([](auto& interface) -> const Core::FE::GaussIntegration&
      { return interface->get_gauss_rule_stiffness_integration(); }, solid_calc_variant_);
}

void Discret::Elements::Solid::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, unique_par_object_id());

  // add base class Element
  Core::Elements::Element::pack(data);

  add_to_pack(data, celltype_);

  Discret::Elements::add_to_pack(data, solid_ele_property_);

  data.add_to_pack(material_post_setup_);

  Discret::Elements::pack(solid_calc_variant_, data);
}

void Discret::Elements::Solid::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);

  extract_from_pack(buffer, celltype_);

  Discret::Elements::extract_from_pack(buffer, solid_ele_property_);

  extract_from_pack(buffer, material_post_setup_);

  // reset solid interface
  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_);

  Discret::Elements::unpack(solid_calc_variant_, buffer);
}

void Discret::Elements::Solid::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
  {
    interface_ptr_ = std::dynamic_pointer_cast<FourC::Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  }
  else
    interface_ptr_ = nullptr;
}

bool Discret::Elements::Solid::read_element(const std::string& eletype, const std::string& celltype,
    const Core::IO::InputParameterContainer& container)
{
  // set cell type
  celltype_ = Core::FE::string_to_cell_type(celltype);

  // read number of material model
  set_material(0, Mat::factory(FourC::Solid::Utils::ReadElement::read_element_material(container)));

  solid_ele_property_ = FourC::Solid::Utils::ReadElement::read_solid_element_properties(container);


  solid_calc_variant_ = create_solid_calculation_interface(celltype_, solid_ele_property_);
  std::visit([&](auto& interface) { interface->setup(*solid_material(), container); },
      solid_calc_variant_);
  return true;
}

std::shared_ptr<Mat::So3Material> Discret::Elements::Solid::solid_material(int nummat) const
{
  return std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

void Discret::Elements::Solid::vis_names(std::map<std::string, int>& names)
{
  Core::Elements::Element::vis_names(names);
  solid_material()->vis_names(names);
}

bool Discret::Elements::Solid::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, id());
}

void Discret::Elements::Solid::for_each_gauss_point(Core::FE::Discretization& discretization,
    std::vector<int>& lm,
    const std::function<void(Mat::So3Material&, double, int)>& integrator) const
{
  std::visit(
      [&](auto& interface)
      {
        interface->for_each_gauss_point(*this, *solid_material(), discretization, lm, integrator);
      },
      solid_calc_variant_);
}

FOUR_C_NAMESPACE_CLOSE
