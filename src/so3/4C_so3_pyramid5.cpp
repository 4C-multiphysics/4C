/*----------------------------------------------------------------------*/
/*! \file

\brief pyramid shaped solid element

\level 1


*----------------------------------------------------------------------*/

#include "4C_so3_pyramid5.hpp"

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
#include "4C_so3_pyramid5fbar.hpp"
#include "4C_so3_surface.hpp"
#include "4C_so3_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::ELEMENTS::SoPyramid5Type Discret::ELEMENTS::SoPyramid5Type::instance_;

Discret::ELEMENTS::SoPyramid5Type& Discret::ELEMENTS::SoPyramid5Type::instance()
{
  return instance_;
}


Core::Communication::ParObject* Discret::ELEMENTS::SoPyramid5Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::SoPyramid5(-1, -1);
  object->unpack(buffer);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoPyramid5Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::rcp(new Discret::ELEMENTS::SoPyramid5(id, owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoPyramid5Type::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::rcp(new Discret::ELEMENTS::SoPyramid5(id, owner));
  return ele;
}


void Discret::ELEMENTS::SoPyramid5Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::SoPyramid5Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::ELEMENTS::SoPyramid5Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["PYRAMID5"] = Input::LineDefinition::Builder()
                         .add_int_vector("PYRAMID5", 5)
                         .add_named_int("MAT")
                         .add_named_string("KINEM")
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


/*----------------------------------------------------------------------*
 |  ctor (public)                                                       |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoPyramid5::SoPyramid5(int id, int owner)
    : SoBase(id, owner), pstype_(Inpar::Solid::PreStress::none), pstime_(0.0), time_(0.0)
{
  kintype_ = Inpar::Solid::KinemType::nonlinearTotLag;
  invJ_.resize(NUMGPT_SOP5, Core::LinAlg::Matrix<NUMDIM_SOP5, NUMDIM_SOP5>(true));
  detJ_.resize(NUMGPT_SOP5, 0.0);

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
    prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(NUMNOD_SOP5, NUMGPT_SOP5));

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                                  |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoPyramid5::SoPyramid5(const Discret::ELEMENTS::SoPyramid5& old)
    : SoBase(old),
      kintype_(old.kintype_),
      detJ_(old.detJ_),
      pstype_(old.pstype_),
      pstime_(old.pstime_),
      time_(old.time_)
{
  invJ_.resize(old.invJ_.size());
  for (int i = 0; i < (int)invJ_.size(); ++i)
  {
    // can this size be anything but NUMDIM_SOP5 x NUMDIM_SOP5?
    invJ_[i] = old.invJ_[i];
  }

  if (Prestress::is_mulf(pstype_))
    prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(*(old.prestress_)));

  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::SoPyramid5::clone() const
{
  auto* newelement = new Discret::ELEMENTS::SoPyramid5(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::SoPyramid5::shape() const
{
  return Core::FE::CellType::pyramid5;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
  // kintype_
  add_to_pack(data, kintype_);

  // detJ_
  add_to_pack(data, detJ_);

  // invJ_
  const auto size = (int)invJ_.size();
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, invJ_[i]);

  // Pack prestress_
  add_to_pack(data, static_cast<int>(pstype_));
  add_to_pack(data, pstime_);
  add_to_pack(data, time_);
  if (Prestress::is_mulf(pstype_))
  {
    Core::Communication::ParObject::add_to_pack(data, *prestress_);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer base_buffer(basedata);
  Element::unpack(base_buffer);
  // kintype_
  kintype_ = static_cast<Inpar::Solid::KinemType>(extract_int(buffer));

  // detJ_
  extract_from_pack(buffer, detJ_);
  // invJ_
  int size = 0;
  extract_from_pack(buffer, size);
  invJ_.resize(size, Core::LinAlg::Matrix<NUMDIM_SOP5, NUMDIM_SOP5>(true));
  for (int i = 0; i < size; ++i) extract_from_pack(buffer, invJ_[i]);

  // Extract prestress_
  pstype_ = static_cast<Inpar::Solid::PreStress>(extract_int(buffer));
  extract_from_pack(buffer, pstime_);
  extract_from_pack(buffer, time_);
  if (Prestress::is_mulf(pstype_))
  {
    std::vector<char> tmpprestress(0);
    extract_from_pack(buffer, tmpprestress);
    if (prestress_ == Teuchos::null)
    {
      int numgpt = NUMGPT_SOP5;
      // see whether I am actually a So_pyramid5fbar element
      auto* me = dynamic_cast<Discret::ELEMENTS::SoPyramid5fbar*>(this);
      if (me) numgpt += 1;  // one more history entry for centroid data in pyramid5fbar
      prestress_ = Teuchos::rcp(new Discret::ELEMENTS::PreStress(NUMNOD_SOP5, numgpt));
    }
    Core::Communication::UnpackBuffer tmpprestress_buffer(tmpprestress);
    prestress_->unpack(tmpprestress_buffer);
    // end
  }

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                                         |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::print(std::ostream& os) const
{
  os << "So_pyramid5 ";
  Element::print(os);
  std::cout << std::endl;
  return;
}

/*====================================================================*/
/* 5-node pyramid node topology*/
/*--------------------------------------------------------------------*/
/* parameter coordinates (r,s,t) of nodes
 * of biunit pyramid [-1,1]x[-1,1]x[0,1]
 * 5-node pyramid: node 1,2,3,4,5


 *                /(5)\
 *              / //\\ \
 *            // //  \\ \\
 *          //  //    \\  \\
 *        //   //  t   \\   \\
 *      //    //   |    \\    \\
 *    //     //    |     \\     \\
 *  (4)-----//------------\\-----(3)
 *  ||     //      |       \\     ||
 *  ||    //       |        \\    ||
 *  ||   //        o---------\\---------s
 *  ||  //         |          \\  ||
 *  || //          |           \\ ||
 *  ||//           |            \\||
 *  (1)============|=============(2)
 *                 |
 *                 r
 *
 */
/*====================================================================*/

/*----------------------------------------------------------------------*
|  get vector of surfaces (public)                                      |
|  surface normals always point outward                                 |
*----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoPyramid5::surfaces()
{
  return Core::Communication::element_boundary_factory<StructuralSurface>(
      Core::Communication::buildSurfaces, *this);
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                                        |
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::SoPyramid5::lines()
{
  return Core::Communication::element_boundary_factory<StructuralLine, Core::Elements::Element>(
      Core::Communication::buildLines, *this);
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                         |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoPyramid5::vis_names(std::map<std::string, int>& names)
{
  solid_material()->vis_names(names);
  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                                  |
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::SoPyramid5::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, NUMGPT_SOP5, this->id());
}

FOUR_C_NAMESPACE_CLOSE
