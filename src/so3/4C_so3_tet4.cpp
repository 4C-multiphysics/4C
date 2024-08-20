/*----------------------------------------------------------------------*/
/*! \file

\brief Solid Tet4 element

\level 1


*----------------------------------------------------------------------*/

#include "4C_so3_tet4.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_fiber_node_holder.hpp"
#include "4C_fem_general_fiber_node_utils.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"
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


Discret::ELEMENTS::SoTet4Type Discret::ELEMENTS::SoTet4Type::instance_;

Discret::ELEMENTS::SoTet4Type& Discret::ELEMENTS::SoTet4Type::instance() { return instance_; }

//------------------------------------------------------------------------
Core::Communication::ParObject* Discret::ELEMENTS::SoTet4Type::create(const std::vector<char>& data)
{
  auto* object = new Discret::ELEMENTS::SoTet4(-1, -1);
  object->unpack(data);
  return object;
}


//------------------------------------------------------------------------
Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoTet4Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::rcp(new Discret::ELEMENTS::SoTet4(id, owner));
    return ele;
  }
  return Teuchos::null;
}


//------------------------------------------------------------------------
Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoTet4Type::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::rcp(new Discret::ELEMENTS::SoTet4(id, owner));
  return ele;
}


//------------------------------------------------------------------------
void Discret::ELEMENTS::SoTet4Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

//------------------------------------------------------------------------
Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::SoTet4Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

//------------------------------------------------------------------------
void Discret::ELEMENTS::SoTet4Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["TET4"] = Input::LineDefinition::Builder()
                     .add_int_vector("TET4", 4)
                     .add_named_int("MAT")
                     .add_named_string("KINEM")
                     .add_optional_named_double_vector("RAD", 3)
                     .add_optional_named_double_vector("AXI", 3)
                     .add_optional_named_double_vector("CIR", 3)
                     .add_optional_named_double_vector("FIBER1", 3)
                     .add_optional_named_double_vector("FIBER2", 3)
                     .add_optional_named_double_vector("FIBER3", 3)
                     .add_optional_named_double("GROWTHTRIG")
                     .build();
}

/*----------------------------------------------------------------------***
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoTet4::SoTet4(int id, int owner)
    : SoBase(id, owner),
      // material_(0),
      V_(-1.0),
      pstype_(Inpar::Solid::PreStress::none),
      pstime_(0.0),
      time_(0.0)
{
  Teuchos::RCP<const Teuchos::ParameterList> params =
      Global::Problem::instance()->get_parameter_list();
  if (params != Teuchos::null)
  {
    pstype_ = Prestress::get_type();
    pstime_ = Prestress::get_prestress_time();

    Discret::ELEMENTS::UTILS::throw_error_fd_material_tangent(
        Global::Problem::instance()->structural_dynamic_params(), get_element_type_string());
  }
  if (Prestress::is_mulf(pstype_))
    prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(NUMNOD_SOTET4, NUMGPT_SOTET4, true));
}

/*----------------------------------------------------------------------***
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoTet4::SoTet4(const Discret::ELEMENTS::SoTet4& old)
    : SoBase(old),
      // material_(old.material_),
      V_(old.V_),
      pstype_(old.pstype_),
      pstime_(old.pstime_),
      time_(old.time_)
{
  if (Prestress::is_mulf(pstype_))
    prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(*(old.prestress_)));
}

/*----------------------------------------------------------------------***
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::SoTet4::clone() const
{
  auto* newelement = new Discret::ELEMENTS::SoTet4(*this);
  return newelement;
}

/*----------------------------------------------------------------------***
 |                                                             (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::SoTet4::shape() const { return Core::FE::CellType::tet4; }

/*----------------------------------------------------------------------***
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoTet4::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

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
  add_to_pack(data, static_cast<int>(pstype_));
  add_to_pack(data, pstime_);
  add_to_pack(data, time_);
  if (Prestress::is_mulf(pstype_))
  {
    Core::Communication::ParObject::add_to_pack(data, *prestress_);
  }
}


/*----------------------------------------------------------------------***
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoTet4::unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  Core::Communication::extract_and_assert_id(position, data, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(position, data, basedata);
  SoBase::unpack(basedata);
  // ngp_
  // extract_from_pack(position,data,ngp_,3*sizeof(int));
  // material_
  // extract_from_pack(position,data,material_);
  // V_
  extract_from_pack(position, data, V_);

  // Extract prestress
  pstype_ = static_cast<Inpar::Solid::PreStress>(extract_int(position, data));
  extract_from_pack(position, data, pstime_);
  extract_from_pack(position, data, time_);
  if (Prestress::is_mulf(pstype_))
  {
    std::vector<char> tmpprestress(0);
    extract_from_pack(position, data, tmpprestress);
    if (prestress_ == Teuchos::null)
      prestress_ =
          Teuchos::rcp(new Discret::ELEMENTS::PreStress(NUMNOD_SOTET4, NUMGPT_SOTET4, true));
    prestress_->unpack(tmpprestress);
  }

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}


/*----------------------------------------------------------------------***
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoTet4::print(std::ostream& os) const
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
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoTet4::surfaces()
{
  return Core::Communication::element_boundary_factory<StructuralSurface, Core::Elements::Element>(
      Core::Communication::buildSurfaces, *this);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
std::vector<double> Discret::ELEMENTS::SoTet4::element_center_refe_coords()
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
  const Core::FE::CellType distype = shape();
  Core::LinAlg::Matrix<NUMNOD_SOTET4, 1> funct;
  // Centroid of a tet with (0,1)(0,1)(0,1) is (0.25, 0.25, 0.25)
  Core::FE::shape_function_3d(funct, 0.25, 0.25, 0.25, distype);
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
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoTet4::lines()
{
  return Core::Communication::element_boundary_factory<StructuralLine, Core::Elements::Element>(
      Core::Communication::buildLines, *this);
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                 st 01/10|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoTet4::vis_names(std::map<std::string, int>& names)
{
  solid_material()->vis_names(names);

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                          st 01/10|
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::SoTet4::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, NUMGPT_SOTET4, this->id());
}

/*----------------------------------------------------------------------*
 |  Call post setup routine of the materials                            |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoTet4::material_post_setup(Teuchos::ParameterList& params)
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
