// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell7p_ele.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_shell7p_ele_factory.hpp"
#include "4C_shell7p_ele_interface_serializable.hpp"
#include "4C_shell7p_line.hpp"
#include "4C_shell7p_utils.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


Discret::Elements::Shell7pType Discret::Elements::Shell7pType::instance_;

Discret::Elements::Shell7pType& Discret::Elements::Shell7pType::instance() { return instance_; }


std::shared_ptr<Core::Elements::Element> Discret::Elements::Shell7pType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SHELL7P") return create(id, owner);
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::Shell7pType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Shell7p>(id, owner);
}

Core::Communication::ParObject* Discret::Elements::Shell7pType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::Shell7p(-1, -1);
  object->unpack(buffer);
  return object;
}

void Discret::Elements::Shell7pType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  Solid::Utils::Shell::nodal_block_information_shell(dwele, numdf, dimns);
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::Shell7pType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  auto* shell = dynamic_cast<Discret::Elements::Shell7p*>(node.elements()[0]);
  if (!shell) FOUR_C_THROW("Cannot cast to Shell7p");
  int j;
  for (j = 0; j < shell->num_node(); ++j)
    if (shell->nodes()[j]->id() == node.id()) break;
  if (j == shell->num_node()) FOUR_C_THROW("Can't find matching node..!");
  double half_thickness = shell->get_thickness() / 2.0;

  // set director
  const Core::LinAlg::SerialDenseMatrix nodal_directors = shell->get_directors();
  Core::LinAlg::Matrix<Shell::Internal::num_dim, 1> director(Core::LinAlg::Initialization::zero);
  for (int dim = 0; dim < Shell::Internal::num_dim; ++dim)
    director(dim, 0) = nodal_directors(j, dim) * half_thickness;

  return Solid::Utils::Shell::compute_shell_null_space(node, x0, director);
}


void Discret::Elements::Shell7pType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsgeneral = definitions["SHELL7P"];

  using namespace Core::IO::InputSpecBuilders;

  defsgeneral[Core::FE::CellType::quad4] = all_of({
      parameter<int>("MAT"),
      parameter<double>("THICK"),
      parameter<std::vector<std::string>>("EAS", {.size = 5}),
      parameter<double>("SDC"),
      parameter<bool>("USE_ANS", {.default_value = false}),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });

  defsgeneral[Core::FE::CellType::quad8] = all_of({
      parameter<int>("MAT"),
      parameter<double>("THICK"),
      parameter<std::vector<std::string>>("EAS", {.size = 5}),
      parameter<double>("SDC"),
      parameter<bool>("USE_ANS", {.default_value = false}),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });

  defsgeneral[Core::FE::CellType::quad9] = all_of({
      parameter<int>("MAT"),
      parameter<double>("THICK"),
      parameter<std::vector<std::string>>("EAS", {.size = 5}),
      parameter<double>("SDC"),
      parameter<bool>("USE_ANS", {.default_value = false}),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });

  defsgeneral[Core::FE::CellType::tri3] = all_of({
      parameter<int>("MAT"),
      parameter<double>("THICK"),
      parameter<double>("SDC"),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });

  defsgeneral[Core::FE::CellType::tri6] = all_of({
      parameter<int>("MAT"),
      parameter<double>("THICK"),
      parameter<double>("SDC"),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });
}

int Discret::Elements::Shell7pType::initialize(Core::FE::Discretization& dis)
{
  Solid::Utils::Shell::Director::setup_shell_element_directors(*this, dis);

  return 0;
}



Discret::Elements::Shell7p::Shell7p(const Discret::Elements::Shell7p& other)
    : Core::Elements::Element(other),
      distype_(other.distype_),
      interface_ptr_(other.interface_ptr_),
      eletech_(other.eletech_),
      thickness_(other.thickness_),
      nodal_directors_(other.nodal_directors_),
      material_post_setup_(other.material_post_setup_)
{
  // reset shell calculation interface
  shell_interface_ = Shell7pFactory::provide_shell7p_calculation_interface(other, other.eletech_);
}

Discret::Elements::Shell7p& Discret::Elements::Shell7p::operator=(
    const Discret::Elements::Shell7p& other)
{
  if (this == &other) return *this;
  Core::Elements::Element::operator=(other);
  distype_ = other.distype_;
  interface_ptr_ = other.interface_ptr_;
  eletech_ = other.eletech_;
  thickness_ = other.thickness_;
  nodal_directors_ = other.nodal_directors_;
  material_post_setup_ = other.material_post_setup_;

  shell_interface_ = Shell7pFactory::provide_shell7p_calculation_interface(other, other.eletech_);
  return *this;
}


Core::Elements::Element* Discret::Elements::Shell7p::clone() const { return new Shell7p(*this); }


int Discret::Elements::Shell7p::num_line() const
{
  return Core::FE::get_number_of_element_lines(distype_);
}


int Discret::Elements::Shell7p::num_surface() const { return 1; }


void Discret::Elements::Shell7p::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Core::Elements::Element::pack(data);
  // discretization type
  add_to_pack(data, distype_);
  // element technology
  add_to_pack(data, eletech_);
  // thickness in reference frame
  add_to_pack(data, thickness_);
  // nodal_directors
  add_to_pack(data, nodal_directors_);
  // Setup flag for material post setup
  data.add_to_pack(material_post_setup_);
  // optional data, e.g., EAS data, current thickness,..
  std::shared_ptr<Shell::Serializable> serializable_interface =
      std::dynamic_pointer_cast<Shell::Serializable>(shell_interface_);
  if (serializable_interface != nullptr) serializable_interface->pack(data);
}


void Discret::Elements::Shell7p::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);
  // discretization type
  extract_from_pack(buffer, distype_);
  // element technology
  extract_from_pack(buffer, eletech_);
  // thickness in reference frame
  extract_from_pack(buffer, thickness_);
  // nodal directors
  extract_from_pack(buffer, nodal_directors_);
  // Setup flag for material post setup
  extract_from_pack(buffer, material_post_setup_);
  // reset shell calculation interface
  shell_interface_ = Shell7pFactory::provide_shell7p_calculation_interface(*this, eletech_);
  std::shared_ptr<Shell::Serializable> serializable_interface =
      std::dynamic_pointer_cast<Shell::Serializable>(shell_interface_);
  if (serializable_interface != nullptr) serializable_interface->unpack(buffer);
}


std::shared_ptr<Mat::So3Material> Discret::Elements::Shell7p::solid_material(int nummat) const
{
  return std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}


void Discret::Elements::Shell7p::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
  {
    interface_ptr_ = std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  }
  else
  {
    interface_ptr_ = nullptr;
  }
}


void Discret::Elements::Shell7p::vis_names(std::map<std::string, int>& names)
{
  std::string result_thickness = "thickness";
  names[result_thickness] = 1;
  solid_material()->vis_names(names);
}  // vis_names()


bool Discret::Elements::Shell7p::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  shell_interface_->vis_data(name, data);

  return solid_material()->vis_data(name, data, id());

}  // vis_data()


void Discret::Elements::Shell7p::print(std::ostream& os) const
{
  os << "Shell7p ";
  os << " discretization type: " << Core::FE::cell_type_to_string(distype_).c_str();
  Element::print(os);
}


std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Shell7p::lines()
{
  return Core::Communication::element_boundary_factory<Shell7pLine, Shell7p>(
      Core::Communication::buildLines, *this);
}


std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Shell7p::surfaces()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

bool Discret::Elements::Shell7p::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  Solid::Elements::ShellData shell_data = {};

  // set discretization type
  distype_ = Core::FE::string_to_cell_type(distype);

  // set thickness in reference frame
  thickness_ = container.get<double>("THICK");
  if (thickness_ <= 0) FOUR_C_THROW("Shell element thickness needs to be > 0");
  shell_data.thickness = thickness_;

  // extract number of EAS parameters for different locking types
  Solid::Elements::ShellLockingTypes locking_types = {};
  if (container.get_if<std::vector<std::string>>("EAS") != nullptr)
  {
    eletech_.insert(Inpar::Solid::EleTech::eas);
    Solid::Utils::Shell::ReadElement::read_and_set_locking_types(
        distype_, container, locking_types);
  }

  // set calculation interface pointer
  shell_interface_ = Shell7pFactory::provide_shell7p_calculation_interface(*this, eletech_);

  // read and set ANS technology for element
  if (distype_ == Core::FE::CellType::quad4 or distype_ == Core::FE::CellType::quad6 or
      distype_ == Core::FE::CellType::quad9)
  {
    if (container.get<bool>("USE_ANS"))
    {
      shell_data.num_ans = Solid::Utils::Shell::ReadElement::read_and_set_num_ans(distype_);
    }
  }

  // read SDC
  shell_data.sdc = container.get<double>("SDC");

  // read and set number of material model
  set_material(
      0, Mat::factory(Solid::Utils::Shell::ReadElement::read_and_set_element_material(container)));

  // setup shell calculation interface
  shell_interface_->setup(*this, *solid_material(), container, locking_types, shell_data);
  if (!material_post_setup_)
  {
    shell_interface_->material_post_setup(*this, *solid_material());
    material_post_setup_ = true;
  }
  return true;
}
FOUR_C_NAMESPACE_CLOSE
