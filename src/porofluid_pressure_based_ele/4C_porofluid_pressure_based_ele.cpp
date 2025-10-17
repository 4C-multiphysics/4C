// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fluid_ele_nullspace.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*
 ******************  PoroFluidMultiPhase ElementType *********************
 *---------------------------------------------------------------------- *
 *-----------------------------------------------------------------------*/

Discret::Elements::PoroFluidMultiPhaseType Discret::Elements::PoroFluidMultiPhaseType::instance_;

Discret::Elements::PoroFluidMultiPhaseType& Discret::Elements::PoroFluidMultiPhaseType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::PoroFluidMultiPhaseType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::PoroFluidMultiPhase* object =
      new Discret::Elements::PoroFluidMultiPhase(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::PoroFluidMultiPhaseType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "POROFLUIDMULTIPHASE")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::PoroFluidMultiPhase>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::PoroFluidMultiPhaseType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::PoroFluidMultiPhase>(id, owner);
  return ele;
}

void Discret::Elements::PoroFluidMultiPhaseType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = dwele->num_dof_per_node(*(dwele->nodes()[0]));
  dimns = numdf;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::PoroFluidMultiPhaseType::compute_null_space(
    Core::Nodes::Node& node, std::span<const double> x0, const int numdof)
{
  switch (numdof)
  {
    case 2:
      return FLD::compute_fluid_null_space<2>();
    case 3:
      return FLD::compute_fluid_null_space<3>();
    case 4:
      return FLD::compute_fluid_null_space<4>();
    case 5:
      return FLD::compute_fluid_null_space<5>();
    default:
      FOUR_C_THROW(
          "The computation of a {}-dimensional null space is not yet implemented for the poro "
          "fluid multiphase element.",
          numdof);
  }
}

void Discret::Elements::PoroFluidMultiPhaseType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["POROFLUIDMULTIPHASE"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad4] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::quad8] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::quad9] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tri3] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tri6] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::line2] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::line3] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::hex8] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tet4] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tet10] = all_of({
      parameter<int>("MAT"),
  });
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*
 ******************  PoroFluidMultiPhase BoundaryType ********************
 *---------------------------------------------------------------------- *
 *-----------------------------------------------------------------------*/

Discret::Elements::PoroFluidMultiPhaseBoundaryType
    Discret::Elements::PoroFluidMultiPhaseBoundaryType::instance_;

Discret::Elements::PoroFluidMultiPhaseBoundaryType&
Discret::Elements::PoroFluidMultiPhaseBoundaryType::instance()
{
  return instance_;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::PoroFluidMultiPhaseBoundaryType::create(
    const int id, const int owner)
{
  // boundary elements are not created as stand-alone elements by the element type,
  // but they are build directly by the corresponding domain element instead.
  // See the element_boundary_factory.
  // Hence, boundary type classes actually are not used as factories, but
  // only for type identification.
  // To make this clear, a null pointer is returned.

  // return Teuchos::rcp( new PoroFluidMultiPhaseBoundary( id, owner ) );
  return nullptr;
}

int Discret::Elements::PoroFluidMultiPhaseType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    Discret::Elements::PoroFluidMultiPhase* actele =
        dynamic_cast<Discret::Elements::PoroFluidMultiPhase*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to PoroFluidMultiPhase* failed");
    actele->initialize();
  }
  return 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*
 *********************  PoroFluidMultiPhase Element **********************
 *---------------------------------------------------------------------- *
 *-----------------------------------------------------------------------*/

Discret::Elements::PoroFluidMultiPhase::PoroFluidMultiPhase(int id, int owner)
    : Core::Elements::Element(id, owner), distype_(Core::FE::CellType::dis_none), numdofpernode_(-1)
{
}

Discret::Elements::PoroFluidMultiPhase::PoroFluidMultiPhase(
    const Discret::Elements::PoroFluidMultiPhase& old)
    : Core::Elements::Element(old), distype_(old.distype_), numdofpernode_(old.numdofpernode_)
{
}

Core::Elements::Element* Discret::Elements::PoroFluidMultiPhase::clone() const
{
  Discret::Elements::PoroFluidMultiPhase* newelement =
      new Discret::Elements::PoroFluidMultiPhase(*this);
  return newelement;
}

Core::FE::CellType Discret::Elements::PoroFluidMultiPhase::shape() const { return distype_; }

void Discret::Elements::PoroFluidMultiPhase::initialize()
{
  std::shared_ptr<Mat::FluidPoroMultiPhase> actmat =
      std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(material());

  actmat->initialize();
}

void Discret::Elements::PoroFluidMultiPhase::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);

  // add internal data
  add_to_pack(data, distype_);
  add_to_pack(data, numdofpernode_);
}

void Discret::Elements::PoroFluidMultiPhase::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);

  // extract internal data
  extract_from_pack(buffer, distype_);
  extract_from_pack(buffer, numdofpernode_);
}

int Discret::Elements::PoroFluidMultiPhase::num_line() const
{
  return Core::FE::get_number_of_element_lines(distype_);
}

int Discret::Elements::PoroFluidMultiPhase::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(distype_);
}

int Discret::Elements::PoroFluidMultiPhase::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(distype_);
}

void Discret::Elements::PoroFluidMultiPhase::print(std::ostream& os) const
{
  os << "PoroFluidMultiPhase element";
  Element::print(os);
  std::cout << std::endl;
  std::cout << "DiscretizationType:  " << Core::FE::cell_type_to_string(distype_) << std::endl;
  std::cout << std::endl;
  std::cout << "Number DOF per Node: " << numdofpernode_ << std::endl;

  return;
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::PoroFluidMultiPhase::lines()
{
  return Core::Communication::get_element_lines<PoroFluidMultiPhaseBoundary, PoroFluidMultiPhase>(
      *this);
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::PoroFluidMultiPhase::surfaces()
{
  return Core::Communication::get_element_surfaces<PoroFluidMultiPhaseBoundary,
      PoroFluidMultiPhase>(*this);
}

bool Discret::Elements::PoroFluidMultiPhase::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  // set discretization type
  set_dis_type(Core::FE::string_to_cell_type(distype));

  return true;
}

void Discret::Elements::PoroFluidMultiPhase::set_material(
    const int index, std::shared_ptr<Core::Mat::Material> mat)
{
  // the standard part:
  Core::Elements::Element::set_material(index, mat);

  // the special part:
  // now the element knows its material, and we can use it to determine numdofpernode
  if (mat->material_type() == Core::Materials::m_fluidporo_multiphase or
      mat->material_type() == Core::Materials::m_fluidporo_multiphase_reactions)
  {
    const Mat::FluidPoroMultiPhase* actmat =
        dynamic_cast<const Mat::FluidPoroMultiPhase*>(mat.get());
    if (actmat == nullptr) FOUR_C_THROW("cast failed");
    numdofpernode_ = actmat->num_mat();
  }
  else
    FOUR_C_THROW(
        "PoroFluidMultiPhase element got unsupported material type {}", mat->material_type());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*
 ***************  PoroFluidMultiPhase Boundary Element *******************
 *---------------------------------------------------------------------- *
 *-----------------------------------------------------------------------*/

Discret::Elements::PoroFluidMultiPhaseBoundary::PoroFluidMultiPhaseBoundary(int id, int owner,
    int nnode, const int* nodeids, Core::Nodes::Node** nodes,
    Discret::Elements::PoroFluidMultiPhase* parent, const int lsurface)
    : Core::Elements::FaceElement(id, owner)
{
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  set_parent_master_element(parent, lsurface);
  return;
}

Discret::Elements::PoroFluidMultiPhaseBoundary::PoroFluidMultiPhaseBoundary(
    const Discret::Elements::PoroFluidMultiPhaseBoundary& old)
    : Core::Elements::FaceElement(old)
{
  return;
}

Core::Elements::Element* Discret::Elements::PoroFluidMultiPhaseBoundary::clone() const
{
  Discret::Elements::PoroFluidMultiPhaseBoundary* newelement =
      new Discret::Elements::PoroFluidMultiPhaseBoundary(*this);
  return newelement;
}

Core::FE::CellType Discret::Elements::PoroFluidMultiPhaseBoundary::shape() const
{
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_element()->shape());
}

void Discret::Elements::PoroFluidMultiPhaseBoundary::pack(
    Core::Communication::PackBuffer& data) const
{
  // boundary elements are rebuild by their parent element for each condition
  // after redistribution. This way we make sure, that the node ids always match.
  // -> no communication of boundary elements
  FOUR_C_THROW("This PoroFluidMultiPhaseBoundary element does not support communication");
  return;
}

void Discret::Elements::PoroFluidMultiPhaseBoundary::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // boundary elements are rebuild by their parent element for each condition
  // after redistribution. This way we make sure, that the node ids always match.
  // -> no communication of boundary elements
  FOUR_C_THROW("This PoroFluidMultiPhaseBoundary element does not support communication");
  return;
}


void Discret::Elements::PoroFluidMultiPhaseBoundary::print(std::ostream& os) const
{
  os << "PoroFluidMultiPhaseBoundary element";
  Element::print(os);
  std::cout << std::endl;
  std::cout << "DiscretizationType:  " << Core::FE::cell_type_to_string(shape()) << std::endl;
  std::cout << std::endl;
  return;
}

int Discret::Elements::PoroFluidMultiPhaseBoundary::num_line() const
{
  return Core::FE::get_number_of_element_lines(shape());
}

int Discret::Elements::PoroFluidMultiPhaseBoundary::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(shape());
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::PoroFluidMultiPhaseBoundary::lines()
{
  FOUR_C_THROW("Lines of PoroFluidMultiPhaseBoundary not implemented");
}

std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::PoroFluidMultiPhaseBoundary::surfaces()
{
  FOUR_C_THROW("Surfaces of PoroFluidMultiPhaseBoundary not implemented");
}

FOUR_C_NAMESPACE_CLOSE
