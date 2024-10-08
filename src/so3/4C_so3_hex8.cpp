/*----------------------------------------------------------------------*/
/*! \file

\brief Solid Hex8 element

\level 1


*----------------------------------------------------------------------*/

#include "4C_so3_hex8.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_fiber_node_holder.hpp"
#include "4C_fem_general_fiber_node_utils.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_so3_element_service.hpp"
#include "4C_so3_hex8fbar.hpp"
#include "4C_so3_line.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_so3_prestress.hpp"
#include "4C_so3_prestress_service.hpp"
#include "4C_so3_surface.hpp"
#include "4C_so3_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

Discret::ELEMENTS::SoHex8Type Discret::ELEMENTS::SoHex8Type::instance_;

Discret::ELEMENTS::SoHex8Type& Discret::ELEMENTS::SoHex8Type::instance() { return instance_; }

namespace
{
  const std::string name = Discret::ELEMENTS::SoHex8Type::instance().name();
}

Core::Communication::ParObject* Discret::ELEMENTS::SoHex8Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::SoHex8(-1, -1);
  object->unpack(buffer);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoHex8Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::make_rcp<Discret::ELEMENTS::SoHex8>(id, owner);
    return ele;
  }

  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoHex8Type::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::make_rcp<Discret::ELEMENTS::SoHex8>(id, owner);
  return ele;
}


void Discret::ELEMENTS::SoHex8Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::SoHex8Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::ELEMENTS::SoHex8Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX8"] = Input::LineDefinition::Builder()
                     .add_int_vector("HEX8", 8)
                     .add_named_int("MAT")
                     .add_named_string("KINEM")
                     .add_named_string("EAS")
                     .add_optional_named_double_vector("RAD", 3)
                     .add_optional_named_double_vector("AXI", 3)
                     .add_optional_named_double_vector("CIR", 3)
                     .add_optional_named_double_vector("FIBER1", 3)
                     .add_optional_named_double_vector("FIBER2", 3)
                     .add_optional_named_double_vector("FIBER3", 3)
                     .add_optional_named_double("STRENGTH")
                     .add_optional_named_double("GROWTHTRIG")
                     .build();
}

// initialization of static gauss point rule for the so_hex8 element
const Core::FE::IntPointsAndWeights<NUMDIM_SOH8> Discret::ELEMENTS::SoHex8::gp_rule_(
    Core::FE::IntPointsAndWeights<NUMDIM_SOH8>(
        static_cast<enum Core::FE::GaussRule3D>(GpRuleSoH8::rule)));

/*----------------------------------------------------------------------*
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoHex8::SoHex8(int id, int owner)
    : SoBase(id, owner),
      easdata_(EASData()),
      analyticalmaterialtangent_(true),
      pstype_(Inpar::Solid::PreStress::none),
      pstime_(0.0),
      time_(0.0),
      old_step_length_(0.0)
{
  eastype_ = soh8_easnone;
  neas_ = 0;
  invJ_.resize(NUMGPT_SOH8, Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>(true));
  detJ_.resize(NUMGPT_SOH8, 0.0);

  Teuchos::RCP<const Teuchos::ParameterList> params =
      Global::Problem::instance()->get_parameter_list();
  if (params != Teuchos::null)
  {
    const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();

    pstype_ = Prestress::get_type();
    pstime_ = Prestress::get_prestress_time();
    if (sdyn.get<std::string>("MATERIALTANGENT") != "analytical")
      analyticalmaterialtangent_ = false;
  }
  if (Prestress::is_mulf(pstype_))
    prestress_ = Teuchos::make_rcp<Discret::ELEMENTS::PreStress>(NUMNOD_SOH8, NUMGPT_SOH8);


  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoHex8::SoHex8(const Discret::ELEMENTS::SoHex8& old)
    : SoBase(old),
      eastype_(old.eastype_),
      neas_(old.neas_),
      easdata_(old.easdata_),
      detJ_(old.detJ_),
      analyticalmaterialtangent_(old.analyticalmaterialtangent_),
      pstype_(old.pstype_),
      pstime_(old.pstime_),
      time_(old.time_),
      old_step_length_(old.old_step_length_)
{
  invJ_.resize(old.invJ_.size());
  for (int i = 0; i < (int)invJ_.size(); ++i)
  {
    // can this size be anything but NUMDIM_SOH8 x NUMDIM_SOH8?
    // invJ_[i].Shape(old.invJ_[i].numRows(),old.invJ_[i].numCols());
    invJ_[i] = old.invJ_[i];
  }

  if (Prestress::is_mulf(pstype_))
    prestress_ = Teuchos::make_rcp<Discret::ELEMENTS::PreStress>(*(old.prestress_));

  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::SoHex8::clone() const
{
  auto* newelement = new Discret::ELEMENTS::SoHex8(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::SoHex8::shape() const { return Core::FE::CellType::hex8; }

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  SoBase::pack(data);
  // eastype_
  add_to_pack(data, eastype_);
  // neas_
  add_to_pack(data, neas_);
  // analyticalmaterialtangent_
  add_to_pack(data, analyticalmaterialtangent_);
  // eas data
  pack_eas_data(data);
  // line search
  add_to_pack(data, old_step_length_);
  // Pack prestress type
  add_to_pack(data, static_cast<int>(pstype_));
  add_to_pack(data, pstime_);
  add_to_pack(data, time_);
  if (Prestress::is_mulf(pstype_))
  {
    add_to_pack(data, *prestress_);
  }

  // detJ_
  add_to_pack(data, detJ_);

  // invJ_
  const auto size = (int)invJ_.size();
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, invJ_[i]);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer basedata_buffer(basedata);
  SoBase::unpack(basedata_buffer);
  // eastype_
  eastype_ = static_cast<EASType>(extract_int(buffer));
  // neas_
  extract_from_pack(buffer, neas_);
  // analyticalmaterialtangent_
  analyticalmaterialtangent_ = extract_int(buffer);
  // eas data
  unpack_eas_data(buffer);
  // line search
  extract_from_pack(buffer, old_step_length_);
  // Extract prestress
  pstype_ = static_cast<Inpar::Solid::PreStress>(extract_int(buffer));
  extract_from_pack(buffer, pstime_);
  extract_from_pack(buffer, time_);
  if (Prestress::is_mulf(pstype_))
  {
    std::vector<char> tmpprestress(0);
    extract_from_pack(buffer, tmpprestress);
    if (prestress_ == Teuchos::null)
    {
      int numgpt = NUMGPT_SOH8;
      // see whether I am actually a So_hex8fbar element
      auto* me = dynamic_cast<Discret::ELEMENTS::SoHex8fbar*>(this);
      if (me) numgpt += 1;  // one more history entry for centroid data in hex8fbar
      prestress_ = Teuchos::make_rcp<Discret::ELEMENTS::PreStress>(NUMNOD_SOH8, numgpt);
    }
    Core::Communication::UnpackBuffer tmpprestress_buffer(tmpprestress);
    prestress_->unpack(tmpprestress_buffer);
  }

  // detJ_
  extract_from_pack(buffer, detJ_);
  // invJ_
  int size = 0;
  extract_from_pack(buffer, size);
  invJ_.resize(size, Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>(true));
  for (int i = 0; i < size; ++i) extract_from_pack(buffer, invJ_[i]);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::print(std::ostream& os) const
{
  os << "So_hex8 ";
  Element::print(os);
  // std::cout << std::endl;
  return;
}

/*====================================================================*/
/* 8-node hexhedra node topology*/
/*--------------------------------------------------------------------*/
/* parameter coordinates (r,s,t) of nodes
 * of biunit cube [-1,1]x[-1,1]x[-1,1]
 *  8-node hexahedron: node 0,1,...,7
 *                      t
 *                      |
 *             4========|================7
 *           //|        |               /||
 *          // |        |              //||
 *         //  |        |             // ||
 *        //   |        |            //  ||
 *       //    |        |           //   ||
 *      //     |        |          //    ||
 *     //      |        |         //     ||
 *     5=========================6       ||
 *    ||       |        |        ||      ||
 *    ||       |        o--------||---------s
 *    ||       |       /         ||      ||
 *    ||       0------/----------||------3
 *    ||      /      /           ||     //
 *    ||     /      /            ||    //
 *    ||    /      /             ||   //
 *    ||   /      /              ||  //
 *    ||  /      /               || //
 *    || /      r                ||//
 *    ||/                        ||/
 *     1=========================2
 *
 */
/*====================================================================*/

/*----------------------------------------------------------------------*
|  get vector of surfaces (public)                             maf 04/07|
|  surface normals always point outward                                 |
*----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoHex8::surfaces()
{
  return Core::Communication::element_boundary_factory<StructuralSurface, Core::Elements::Element>(
      Core::Communication::buildSurfaces, *this);
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                               maf 04/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoHex8::lines()
{
  return Core::Communication::element_boundary_factory<StructuralLine, Core::Elements::Element>(
      Core::Communication::buildLines, *this);
}

/*----------------------------------------------------------------------*
 |  get location of element center                              jb 08/11|
 *----------------------------------------------------------------------*/
std::vector<double> Discret::ELEMENTS::SoHex8::element_center_refe_coords()
{
  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // material coord. of element
  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];
  }
  Core::LinAlg::Matrix<NUMNOD_SOH8, 1> funct;
  // Element midpoint at r=s=t=0.0
  Core::FE::shape_function_3d(funct, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);
  Core::LinAlg::Matrix<1, NUMDIM_SOH8> midpoint;
  // midpoint.multiply('T','N',1.0,funct,xrefe,0.0);
  midpoint.multiply_tn(funct, xrefe);
  std::vector<double> centercoords(3);
  centercoords[0] = midpoint(0, 0);
  centercoords[1] = midpoint(0, 1);
  centercoords[2] = midpoint(0, 2);
  return centercoords;
}
/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                maf 01/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8::vis_names(std::map<std::string, int>& names)
{
  solid_material()->vis_names(names);

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                         maf 01/08|
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::SoHex8::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, NUMGPT_SOH8, this->id());
}

// Compute nodal fibers and call post setup routine of the materials
void Discret::ELEMENTS::SoHex8::material_post_setup(Teuchos::ParameterList& params)
{
  if (Core::Nodes::have_nodal_fibers<Core::FE::CellType::hex8>(nodes()))
  {
    // This element has fiber nodes.
    // Interpolate fibers to the Gauss points and pass them to the material

    // Get shape functions
    const std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();

    // add fibers to the ParameterList
    // ParameterList does not allow to store a std::vector, so we have to add every gp fiber
    // with a separate key. To keep it clean, It is added to a sublist.
    Core::Nodes::NodalFiberHolder fiberHolder;

    // Do the interpolation
    Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::hex8>(
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
