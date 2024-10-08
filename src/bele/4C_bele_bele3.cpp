/*----------------------------------------------------------------------*/
/*! \file

\brief dummy 3D boundary element without any physics


\level 2
*/
/*----------------------------------------------------------------------*/

#include "4C_bele_bele3.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_utils_exceptions.hpp"

#include <sstream>

FOUR_C_NAMESPACE_OPEN


Discret::ELEMENTS::Bele3Type Discret::ELEMENTS::Bele3Type::instance_;


Discret::ELEMENTS::Bele3Type& Discret::ELEMENTS::Bele3Type::instance() { return instance_; }


Core::Communication::ParObject* Discret::ELEMENTS::Bele3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::ELEMENTS::Bele3* object = new Discret::ELEMENTS::Bele3(-1, -1);
  object->unpack(buffer);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::Bele3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  // Search for "BELE3". If found, search for "_"
  // the number after "_" is numdof: so BELE3_4 is a BELE3 element
  // with numdof=4
  std::size_t pos = eletype.rfind("BELE3");
  if (pos != std::string::npos)
  {
    if (eletype.substr(pos + 5, 1) == "_")
    {
      std::istringstream is(eletype.substr(pos + 6, 1));

      int numdof = -1;
      is >> numdof;
      Teuchos::RCP<Discret::ELEMENTS::Bele3> ele =
          Teuchos::RCP(new Discret::ELEMENTS::Bele3(id, owner));
      ele->set_num_dof_per_node(numdof);
      return ele;
    }
    else
    {
      FOUR_C_THROW("ERROR: Found BELE3 element without specified number of dofs!");
    }
  }

  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::Bele3Type::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::RCP(new Discret::ELEMENTS::Bele3(id, owner));
  return ele;
}


void Discret::ELEMENTS::Bele3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::Bele3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::ELEMENTS::Bele3Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs3 = definitions["BELE3_3"];

  defs3["TRI3"] = Input::LineDefinition::Builder()
                      .add_int_vector("TRI3", 3)
                      .add_optional_named_int("MAT")
                      .build();

  defs3["TRI6"] = Input::LineDefinition::Builder()
                      .add_int_vector("TRI6", 6)
                      .add_optional_named_int("MAT")
                      .build();

  defs3["QUAD4"] = Input::LineDefinition::Builder()
                       .add_int_vector("QUAD4", 4)
                       .add_optional_named_int("MAT")
                       .build();

  defs3["QUAD8"] = Input::LineDefinition::Builder()
                       .add_int_vector("QUAD8", 8)
                       .add_optional_named_int("MAT")
                       .build();

  defs3["QUAD9"] = Input::LineDefinition::Builder()
                       .add_int_vector("QUAD9", 9)
                       .add_optional_named_int("MAT")
                       .build();

  std::map<std::string, Input::LineDefinition>& defs4 = definitions["BELE3_4"];

  defs4["TRI3"] = Input::LineDefinition::Builder()
                      .add_int_vector("TRI3", 3)
                      .add_optional_named_int("MAT")
                      .build();

  defs4["TRI6"] = Input::LineDefinition::Builder()
                      .add_int_vector("TRI6", 6)
                      .add_optional_named_int("MAT")
                      .build();

  defs4["QUAD4"] = Input::LineDefinition::Builder()
                       .add_int_vector("QUAD4", 4)
                       .add_optional_named_int("MAT")
                       .build();

  defs4["QUAD8"] = Input::LineDefinition::Builder()
                       .add_int_vector("QUAD8", 8)
                       .add_optional_named_int("MAT")
                       .build();

  defs4["QUAD9"] = Input::LineDefinition::Builder()
                       .add_int_vector("QUAD9", 9)
                       .add_optional_named_int("MAT")
                       .build();
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::Bele3LineType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Bele3Line( id, owner ) );
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::Bele3::Bele3(int id, int owner)
    : Core::Elements::Element(id, owner), numdofpernode_(-1)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::Bele3::Bele3(const Discret::ELEMENTS::Bele3& old)
    : Core::Elements::Element(old), numdofpernode_(old.numdofpernode_)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::Bele3::clone() const
{
  Discret::ELEMENTS::Bele3* newelement = new Discret::ELEMENTS::Bele3(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::Bele3::shape() const
{
  switch (num_node())
  {
    case 3:
      return Core::FE::CellType::tri3;
    case 4:
      return Core::FE::CellType::quad4;
    case 6:
      return Core::FE::CellType::tri6;
    case 8:
      return Core::FE::CellType::quad8;
    case 9:
      return Core::FE::CellType::quad9;
    default:
      FOUR_C_THROW("unexpected number of nodes %d", num_node());
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::Bele3::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
  // numdofpernode_
  add_to_pack(data, numdofpernode_);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::Bele3::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer base_buffer(basedata);
  Element::unpack(base_buffer);
  // numdofpernode_
  numdofpernode_ = extract_int(buffer);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::Bele3::print(std::ostream& os) const
{
  os << "Bele3_" << numdofpernode_ << " " << Core::FE::cell_type_to_string(shape());
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                               gjb 05/08|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::Bele3::lines()
{
  return Core::Communication::element_boundary_factory<Bele3Line, Bele3>(
      Core::Communication::buildLines, *this);
}


/*----------------------------------------------------------------------*
 |  get vector of Surfaces (length 1) (public)               gammi 04/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::Bele3::surfaces()
{
  return {Teuchos::rcpFromRef(*this)};
}


Core::FE::GaussRule2D Discret::ELEMENTS::Bele3::get_optimal_gaussrule() const
{
  Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::undefined;
  switch (shape())
  {
    case Core::FE::CellType::quad4:
      rule = Core::FE::GaussRule2D::quad_4point;
      break;
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
      rule = Core::FE::GaussRule2D::quad_9point;
      break;
    case Core::FE::CellType::tri3:
      rule = Core::FE::GaussRule2D::tri_3point;
      break;
    case Core::FE::CellType::tri6:
      rule = Core::FE::GaussRule2D::tri_6point;
      break;
    default:
      FOUR_C_THROW("unknown number of nodes for gaussrule initialization");
      break;
  }
  return rule;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::Bele3::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // check if material is defined
  int material = container.get_or("MAT", -1);
  if (material != -1)
  {
    set_material(0, Mat::factory(material));
  }
  return true;
}

FOUR_C_NAMESPACE_CLOSE
