/*----------------------------------------------------------------------*/
/*! \file

\brief Solid Wedge6 Element

\level 1


*----------------------------------------------------------------------*/

#include "4C_so3_weg6.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
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

Discret::ELEMENTS::SoWeg6Type Discret::ELEMENTS::SoWeg6Type::instance_;

Discret::ELEMENTS::SoWeg6Type& Discret::ELEMENTS::SoWeg6Type::instance() { return instance_; }

Core::Communication::ParObject* Discret::ELEMENTS::SoWeg6Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::SoWeg6(-1, -1);
  object->unpack(buffer);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoWeg6Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::make_rcp<Discret::ELEMENTS::SoWeg6>(id, owner);
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoWeg6Type::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::make_rcp<Discret::ELEMENTS::SoWeg6>(id, owner);
  return ele;
}


void Discret::ELEMENTS::SoWeg6Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::SoWeg6Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::ELEMENTS::SoWeg6Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["WEDGE6"] = Input::LineDefinition::Builder()
                       .add_int_vector("WEDGE6", 6)
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


/*----------------------------------------------------------------------*
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoWeg6::SoWeg6(int id, int owner)
    : SoBase(id, owner), pstype_(Inpar::Solid::PreStress::none), pstime_(0.0), time_(0.0)
{
  invJ_.resize(NUMGPT_WEG6);
  detJ_.resize(NUMGPT_WEG6);
  for (int i = 0; i < NUMGPT_WEG6; ++i)
  {
    detJ_[i] = 0.0;
    invJ_[i] = Core::LinAlg::Matrix<NUMDIM_WEG6, NUMDIM_WEG6>(true);
  }

  Teuchos::RCP<const Teuchos::ParameterList> params =
      Global::Problem::instance()->get_parameter_list();
  if (params != Teuchos::null)
  {
    pstype_ = Prestress::get_type();
    pstime_ = Prestress::get_prestress_time();

    Discret::ELEMENTS::Utils::throw_error_fd_material_tangent(
        Global::Problem::instance()->structural_dynamic_params(), get_element_type_string());
  }
  if (Prestress::is_mulf(pstype_))
    prestress_ = Teuchos::make_rcp<Discret::ELEMENTS::PreStress>(NUMNOD_WEG6, NUMGPT_WEG6);
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoWeg6::SoWeg6(const Discret::ELEMENTS::SoWeg6& old)
    : SoBase(old), detJ_(old.detJ_), pstype_(old.pstype_), pstime_(old.pstime_), time_(old.time_)
{
  invJ_.resize(old.invJ_.size());
  for (unsigned int i = 0; i < invJ_.size(); ++i)
  {
    invJ_[i] = old.invJ_[i];
  }

  if (Prestress::is_mulf(pstype_))
    prestress_ = Teuchos::make_rcp<Discret::ELEMENTS::PreStress>(*(old.prestress_));
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::SoWeg6::clone() const
{
  auto* newelement = new Discret::ELEMENTS::SoWeg6(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::SoWeg6::shape() const { return Core::FE::CellType::wedge6; }

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoWeg6::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  SoBase::pack(data);

  // Pack prestress
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
  const unsigned int size = invJ_.size();
  add_to_pack(data, size);
  for (unsigned int i = 0; i < size; ++i) add_to_pack(data, invJ_[i]);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoWeg6::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer basedata_buffer(basedata);
  SoBase::unpack(basedata_buffer);
  // prestress_
  pstype_ = static_cast<Inpar::Solid::PreStress>(extract_int(buffer));
  extract_from_pack(buffer, pstime_);
  extract_from_pack(buffer, time_);
  if (Prestress::is_mulf(pstype_))
  {
    std::vector<char> tmpprestress(0);
    extract_from_pack(buffer, tmpprestress);
    if (prestress_ == Teuchos::null)
      prestress_ = Teuchos::make_rcp<Discret::ELEMENTS::PreStress>(NUMNOD_WEG6, NUMGPT_WEG6);
    Core::Communication::UnpackBuffer tmpprestress_buffer(tmpprestress);
    prestress_->unpack(tmpprestress_buffer);
  }

  // detJ_
  extract_from_pack(buffer, detJ_);
  // invJ_
  int size = 0;
  extract_from_pack(buffer, size);
  invJ_.resize(size);
  for (int i = 0; i < size; ++i)
  {
    invJ_[i] = Core::LinAlg::Matrix<NUMDIM_WEG6, NUMDIM_WEG6>(true);
    extract_from_pack(buffer, invJ_[i]);
  }

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoWeg6::print(std::ostream& os) const
{
  os << "So_weg6 ";
  Element::print(os);
  std::cout << std::endl;
  return;
}


std::vector<double> Discret::ELEMENTS::SoWeg6::element_center_refe_coords()
{
  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_WEG6, NUMDIM_WEG6> xrefe;  // material coord. of element
  for (int i = 0; i < NUMNOD_WEG6; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];
  }
  Core::LinAlg::Matrix<NUMNOD_WEG6, 1> funct;
  // Element midpoint at r=s=1/3, t=0.0
  Core::FE::shape_function_3d(funct, 1.0 / 3.0, 1.0 / 3.0, 0.0, Core::FE::CellType::wedge6);
  Core::LinAlg::Matrix<1, NUMDIM_WEG6> midpoint;
  // midpoint.multiply('T','N',1.0,funct,xrefe,0.0);
  midpoint.multiply_tn(funct, xrefe);
  std::vector<double> centercoords(3);
  centercoords[0] = midpoint(0, 0);
  centercoords[1] = midpoint(0, 1);
  centercoords[2] = midpoint(0, 2);
  return centercoords;
}
/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                maf 07/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoWeg6::vis_names(std::map<std::string, int>& names)
{
  solid_material()->vis_names(names);

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                         maf 07/08|
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::SoWeg6::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, NUMGPT_WEG6, this->id());
}


/*----------------------------------------------------------------------*
|  get vector of surfaces (public)                             maf 04/07|
|  surface normals always point outward                                 |
*----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoWeg6::surfaces()
{
  return Core::Communication::element_boundary_factory<StructuralSurface, Core::Elements::Element>(
      Core::Communication::buildSurfaces, *this);
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                               maf 04/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoWeg6::lines()
{
  return Core::Communication::element_boundary_factory<StructuralLine, Core::Elements::Element>(
      Core::Communication::buildLines, *this);
}

FOUR_C_NAMESPACE_CLOSE
