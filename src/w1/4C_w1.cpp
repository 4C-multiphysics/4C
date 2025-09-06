// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Wall1Type Discret::Elements::Wall1Type::instance_;

Discret::Elements::Wall1Type& Discret::Elements::Wall1Type::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::Wall1Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Wall1* object = new Discret::Elements::Wall1(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Wall1Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALL")
  {
    if (eledistype != "NURBS4" and eledistype != "NURBS9")
    {
      return std::make_shared<Discret::Elements::Wall1>(id, owner);
    }
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Wall1Type::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Wall1>(id, owner);
}


void Discret::Elements::Wall1Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 2;
  dimns = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::Wall1Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, int const numdof, int const dimnsp)
{
  return compute_solid_null_space<2>(node.x(), x0);
}

void Discret::Elements::Wall1Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["WALL"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });

  defs[Core::FE::CellType::quad8] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });

  defs[Core::FE::CellType::quad9] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });

  defs[Core::FE::CellType::tri3] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });

  defs[Core::FE::CellType::tri6] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });

  defs[Core::FE::CellType::nurbs4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });

  defs[Core::FE::CellType::nurbs9] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Wall1LineType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Wall1Line( id, owner ) );
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mgit 01/08/|
 *----------------------------------------------------------------------*/
Discret::Elements::Wall1::Wall1(int id, int owner)
    : Core::Elements::Element(id, owner),
      material_(0),
      thickness_(0.0),
      old_step_length_(0.0),
      gaussrule_(Core::FE::GaussRule2D::undefined),
      wtype_(plane_none),
      stresstype_(w1_none),
      iseas_(false),
      eastype_(eas_vague),
      easdata_(EASData()),
      distype_(Core::FE::CellType::dis_none)
{
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mgit 01/08|
 *----------------------------------------------------------------------*/
Discret::Elements::Wall1::Wall1(const Discret::Elements::Wall1& old)
    : Core::Elements::Element(old),
      material_(old.material_),
      thickness_(old.thickness_),
      old_step_length_(old.old_step_length_),
      gaussrule_(old.gaussrule_),
      wtype_(old.wtype_),
      stresstype_(old.stresstype_),
      iseas_(old.iseas_),
      eastype_(old.eas_vague),
      interface_ptr_(old.interface_ptr_),
      material_post_setup_(old.material_post_setup_),
      kintype_(old.kintype_),
      easdata_(old.easdata_),
      distype_(old.distype_)
{
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Wall1 and return pointer to it (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Wall1::clone() const
{
  Discret::Elements::Wall1* newelement = new Discret::Elements::Wall1(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          mgit 04/07 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Wall1::shape() const { return distype_; }


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Core::Elements::Element::pack(data);
  // material_
  add_to_pack(data, material_);
  // thickness
  add_to_pack(data, thickness_);
  // plane strain or plane stress information
  add_to_pack(data, wtype_);
  // gaussrule_
  add_to_pack(data, gaussrule_);
  // stresstype
  add_to_pack(data, stresstype_);
  // eas
  add_to_pack(data, iseas_);
  // eas type
  add_to_pack(data, eastype_);
  // eas data
  pack_eas_data(data);
  // distype
  add_to_pack(data, distype_);
  // line search
  add_to_pack(data, old_step_length_);
  Core::Communication::add_to_pack(data, material_post_setup_);
  Core::Communication::add_to_pack(data, kintype_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);
  // material_
  extract_from_pack(buffer, material_);
  // thickness_
  extract_from_pack(buffer, thickness_);
  // plane strain or plane stress information_
  extract_from_pack(buffer, wtype_);
  // gaussrule_
  extract_from_pack(buffer, gaussrule_);
  // stresstype_
  extract_from_pack(buffer, stresstype_);
  // iseas_
  extract_from_pack(buffer, iseas_);
  // eastype_
  extract_from_pack(buffer, eastype_);
  // easdata_
  unpack_eas_data(buffer);
  // distype_
  extract_from_pack(buffer, distype_);
  // line search
  extract_from_pack(buffer, old_step_length_);
  Core::Communication::extract_from_pack(buffer, material_post_setup_);
  Core::Communication::extract_from_pack(buffer, kintype_);

  return;
}



/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             mgit 07/07|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Wall1::lines()
{
  return Core::Communication::element_boundary_factory<Wall1Line, Wall1>(
      Core::Communication::buildLines, *this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                          mgit 03/07|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Wall1::surfaces()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

/*-----------------------------------------------------------------------------*
| Map plane Green-Lagrange strains to 3d                       mayr.mt 05/2014 |
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::green_lagrange_plane3d(
    const Core::LinAlg::SerialDenseVector& glplane,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& gl3d)
{
  gl3d(0, 0) = glplane(0);                       // E_{11}
  gl3d(1, 1) = glplane(1);                       // E_{22}
  gl3d(2, 2) = 0.0;                              // E_{33}
  gl3d(0, 1) = 0.5 * (glplane(2) + glplane(3));  // 2*E_{12}=E_{12}+E_{21}
  gl3d(1, 2) = 0.0;                              // 2*E_{23}
  gl3d(0, 2) = 0.0;                              // 2*E_{31}
}

void Discret::Elements::Wall1::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
    interface_ptr_ = p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface");
  else
    interface_ptr_ = nullptr;
}

std::shared_ptr<Core::Elements::ParamsInterface> Discret::Elements::Wall1::params_interface_ptr()
{
  return interface_ptr_;
}

// Check, whether the material post setup routine was
void Discret::Elements::Wall1::ensure_material_post_setup(Teuchos::ParameterList& params)
{
  if (!material_post_setup_)
  {
    material_post_setup(params);
  }
}

void Discret::Elements::Wall1::material_post_setup(Teuchos::ParameterList& params)
{
  // This is the minimal implementation. Advanced materials may need extra implementation here.
  solid_material()->post_setup(params, id());
  material_post_setup_ = true;
}

std::shared_ptr<Mat::So3Material> Discret::Elements::Wall1::solid_material(int nummat) const
{
  return std::dynamic_pointer_cast<Mat::So3Material>(Core::Elements::Element::material(nummat));
}

Solid::Elements::ParamsInterface& Discret::Elements::Wall1::str_params_interface()
{
  if (not is_params_interface()) FOUR_C_THROW("The interface ptr is not set!");
  return *(std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(interface_ptr_));
}

FOUR_C_NAMESPACE_CLOSE
