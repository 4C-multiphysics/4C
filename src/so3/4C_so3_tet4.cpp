// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_tet4.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_fiber_node_holder.hpp"
#include "4C_fem_general_fiber_node_utils.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_so3_line.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_so3_prestress.hpp"
#include "4C_so3_prestress_service.hpp"
#include "4C_so3_surface.hpp"
#include "4C_so3_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


Discret::Elements::SoTet4Type Discret::Elements::SoTet4Type::instance_;

Discret::Elements::SoTet4Type& Discret::Elements::SoTet4Type::instance() { return instance_; }

//------------------------------------------------------------------------
Core::Communication::ParObject* Discret::Elements::SoTet4Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::SoTet4(-1, -1);
  object->unpack(buffer);
  return object;
}


//------------------------------------------------------------------------
std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::SoTet4>(id, owner);
    return ele;
  }
  return nullptr;
}


//------------------------------------------------------------------------
std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::SoTet4>(id, owner);
  return ele;
}


//------------------------------------------------------------------------
void Discret::Elements::SoTet4Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

//------------------------------------------------------------------------
Core::LinAlg::SerialDenseMatrix Discret::Elements::SoTet4Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

//------------------------------------------------------------------------
void Discret::Elements::SoTet4Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions[get_element_type_string()];

  using namespace Core::IO::InputSpecBuilders;

  defs["TET4"] = anonymous_group({
      entry<std::vector<int>>("TET4", {.size = 4}),
      entry<int>("MAT"),
      entry<std::string>("KINEM"),
      entry<std::vector<double>>("RAD", {.required = false, .size = 3}),
      entry<std::vector<double>>("AXI", {.required = false, .size = 3}),
      entry<std::vector<double>>("CIR", {.required = false, .size = 3}),
      entry<std::vector<double>>("FIBER1", {.required = false, .size = 3}),
      entry<std::vector<double>>("FIBER2", {.required = false, .size = 3}),
      entry<std::vector<double>>("FIBER3", {.required = false, .size = 3}),
      entry<double>("GROWTHTRIG", {.required = false}),
  });
}

/*----------------------------------------------------------------------***
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoTet4::SoTet4(int id, int owner)
    : SoBase(id, owner),
      // material_(0),
      V_(-1.0),
      pstype_(Inpar::Solid::PreStress::none),
      pstime_(0.0),
      time_(0.0)
{
  std::shared_ptr<const Teuchos::ParameterList> params =
      Global::Problem::instance()->get_parameter_list();
  if (params != nullptr)
  {
    pstype_ = Prestress::get_type();
    pstime_ = Prestress::get_prestress_time();

    Discret::Elements::Utils::throw_error_fd_material_tangent(
        Global::Problem::instance()->structural_dynamic_params(), get_element_type_string());
  }
  if (Prestress::is_mulf(pstype_))
    prestress_ = std::make_shared<Discret::Elements::PreStress>(NUMNOD_SOTET4, NUMGPT_SOTET4, true);
}

/*----------------------------------------------------------------------***
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoTet4::SoTet4(const Discret::Elements::SoTet4& old)
    : SoBase(old),
      // material_(old.material_),
      V_(old.V_),
      pstype_(old.pstype_),
      pstime_(old.pstime_),
      time_(old.time_)
{
  if (Prestress::is_mulf(pstype_))
    prestress_ = std::make_shared<Discret::Elements::PreStress>(*(old.prestress_));
}

/*----------------------------------------------------------------------***
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::SoTet4::clone() const
{
  auto* newelement = new Discret::Elements::SoTet4(*this);
  return newelement;
}

/*----------------------------------------------------------------------***
 |                                                             (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::SoTet4::shape() const { return Core::FE::CellType::tet4; }

/*----------------------------------------------------------------------***
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoTet4::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  SoBase::pack(data);
  // ngp_
  // add_to_pack(data,ngp_,3*sizeof(int));
  // material_
  // add_to_pack(data,material_);

  // V_
  add_to_pack(data, V_);

  // Pack prestress
  add_to_pack(data, pstype_);
  add_to_pack(data, pstime_);
  add_to_pack(data, time_);
  if (Prestress::is_mulf(pstype_))
  {
    add_to_pack(data, *prestress_);
  }
}


/*----------------------------------------------------------------------***
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoTet4::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  SoBase::unpack(buffer);
  // ngp_
  // extract_from_pack(position,data,ngp_,3*sizeof(int));
  // material_
  // extract_from_pack(position,data,material_);
  // V_
  extract_from_pack(buffer, V_);

  // Extract prestress
  extract_from_pack(buffer, pstype_);
  extract_from_pack(buffer, pstime_);
  extract_from_pack(buffer, time_);
  if (Prestress::is_mulf(pstype_))
  {
    std::vector<char> tmpprestress(0);
    extract_from_pack(buffer, tmpprestress);
    if (prestress_ == nullptr)
      prestress_ =
          std::make_shared<Discret::Elements::PreStress>(NUMNOD_SOTET4, NUMGPT_SOTET4, true);
    Core::Communication::UnpackBuffer tmpprestress_buffer(tmpprestress);
    prestress_->unpack(tmpprestress_buffer);
  }


  return;
}


/*----------------------------------------------------------------------***
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoTet4::print(std::ostream& os) const
{
  os << "So_tet4 ";
  Element::print(os);
  std::cout << std::endl;
  return;
}

/*====================================================================*/
/* 4-node tetrahedra node topology*/
/*--------------------------------------------------------------------*/
/* parameter coordinates (ksi1, ksi2, ksi3) of nodes
 * of a common tetrahedron [0,1]x[0,1]x[0,1]
 *  4-node hexahedron: node 0,1,...,3
 *
 * -----------------------
 *- this is the numbering used in GiD & EXODUS!!
 *      3-
 *      |\ ---
 *      |  \    ---
 *      |    \      ---
 *      |      \        -2
 *      |        \       /\
 *      |          \   /   \
 *      |            X      \
 *      |          /   \     \
 *      |        /       \    \
 *      |      /           \   \
 *      |    /               \  \
 *      |  /                   \ \
 *      |/                       \\
 *      0--------------------------1
 */
/*====================================================================*/


/*----------------------------------------------------------------------*
|  get vector of surfaces (public)                             maf 04/07|
|  surface normals always point outward                                 |
*----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::SoTet4::surfaces()
{
  return Core::Communication::element_boundary_factory<StructuralSurface, Core::Elements::Element>(
      Core::Communication::buildSurfaces, *this);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
std::vector<double> Discret::Elements::SoTet4::element_center_refe_coords()
{
  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOTET4, NUMDIM_SOTET4> xrefe;  // material coord. of element
  for (int i = 0; i < NUMNOD_SOTET4; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];
  }
  Core::LinAlg::Matrix<NUMNOD_SOTET4, 1> funct;
  // Centroid of a tet with (0,1)(0,1)(0,1) is (0.25, 0.25, 0.25)
  Core::FE::shape_function_3d(funct, 0.25, 0.25, 0.25, Core::FE::CellType::tet4);
  Core::LinAlg::Matrix<1, NUMDIM_SOTET4> midpoint;
  // midpoint.multiply('T','N',1.0,funct,xrefe,0.0);
  midpoint.multiply_tn(funct, xrefe);
  std::vector<double> centercoords(3);
  centercoords[0] = midpoint(0, 0);
  centercoords[1] = midpoint(0, 1);
  centercoords[2] = midpoint(0, 2);
  return centercoords;
}

/*----------------------------------------------------------------------***++
 |  get vector of lines (public)                               maf 04/07|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::SoTet4::lines()
{
  return Core::Communication::element_boundary_factory<StructuralLine, Core::Elements::Element>(
      Core::Communication::buildLines, *this);
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                 st 01/10|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoTet4::vis_names(std::map<std::string, int>& names)
{
  solid_material()->vis_names(names);

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                          st 01/10|
 *----------------------------------------------------------------------*/
bool Discret::Elements::SoTet4::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, NUMGPT_SOTET4, this->id());
}

/*----------------------------------------------------------------------*
 |  Call post setup routine of the materials                            |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoTet4::material_post_setup(Teuchos::ParameterList& params)
{
  if (Core::Nodes::have_nodal_fibers<Core::FE::CellType::tet4>(nodes()))
  {
    // This element has fiber nodes.
    // Interpolate fibers to the Gauss points and pass them to the material

    // Get shape functions
    static const std::vector<Core::LinAlg::Matrix<NUMNOD_SOTET4, 1>> shapefcts =
        so_tet4_1gp_shapefcts();

    // add fibers to the ParameterList
    // ParameterList does not allow to store a std::vector, so we have to add every gp fiber
    // with a separate key. To keep it clean, It is added to a sublist.
    Core::Nodes::NodalFiberHolder fiberHolder;

    // Do the interpolation
    Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::tet4>(
        nodes(), shapefcts, fiberHolder);

    params.set("fiberholder", fiberHolder);
  }

  // Call super post setup
  SoBase::material_post_setup(params);

  // Cleanup ParameterList to not carry all fibers the whole simulation
  // do not throw an error if key does not exist.
  params.remove("fiberholder", false);
}

FOUR_C_NAMESPACE_CLOSE
