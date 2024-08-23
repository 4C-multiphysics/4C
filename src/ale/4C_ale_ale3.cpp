/*----------------------------------------------------------------------------*/
/*! \file

\brief 3D ALE element

\level 1

*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
#include "4C_ale_ale3.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::ELEMENTS::Ale3Type Discret::ELEMENTS::Ale3Type::instance_;

Discret::ELEMENTS::Ale3Type& Discret::ELEMENTS::Ale3Type::instance() { return instance_; }

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::ELEMENTS::Ale3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::ELEMENTS::Ale3* object = new Discret::ELEMENTS::Ale3(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::Ale3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele;

  if (eletype == "ALE3")
  {
    if (eledistype != "NURBS27")
    {
      ele = Teuchos::rcp(new Discret::ELEMENTS::Ale3(id, owner));
    }
  }

  return ele;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::Ale3Type::create(
    const int id, const int owner)
{
  return Teuchos::rcp(new Discret::ELEMENTS::Ale3(id, owner));
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::ELEMENTS::Ale3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::Ale3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::ELEMENTS::Ale3Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions["ALE3"];

  defs["HEX8"] =
      Input::LineDefinition::Builder().add_int_vector("HEX8", 8).add_named_int("MAT").build();

  defs["HEX20"] =
      Input::LineDefinition::Builder().add_int_vector("HEX20", 20).add_named_int("MAT").build();

  defs["HEX27"] =
      Input::LineDefinition::Builder().add_int_vector("HEX27", 27).add_named_int("MAT").build();

  defs["TET4"] =
      Input::LineDefinition::Builder().add_int_vector("TET4", 4).add_named_int("MAT").build();

  defs["TET10"] =
      Input::LineDefinition::Builder().add_int_vector("TET10", 10).add_named_int("MAT").build();

  defs["WEDGE6"] =
      Input::LineDefinition::Builder().add_int_vector("WEDGE6", 6).add_named_int("MAT").build();

  defs["WEDGE15"] =
      Input::LineDefinition::Builder().add_int_vector("WEDGE15", 15).add_named_int("MAT").build();

  defs["PYRAMID5"] =
      Input::LineDefinition::Builder().add_int_vector("PYRAMID5", 5).add_named_int("MAT").build();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::Ale3SurfaceType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Ale3Surface( id, owner ) );
  return Teuchos::null;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::ELEMENTS::Ale3::Ale3(int id, int owner) : Core::Elements::Element(id, owner) {}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::ELEMENTS::Ale3::Ale3(const Discret::ELEMENTS::Ale3& old) : Core::Elements::Element(old)
{
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::Ale3::clone() const
{
  Discret::ELEMENTS::Ale3* newelement = new Discret::ELEMENTS::Ale3(*this);
  return newelement;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::Ale3::shape() const
{
  switch (num_node())
  {
    case 4:
      return Core::FE::CellType::tet4;
    case 5:
      return Core::FE::CellType::pyramid5;
    case 6:
      return Core::FE::CellType::wedge6;
    case 8:
      return Core::FE::CellType::hex8;
    case 10:
      return Core::FE::CellType::tet10;
    case 20:
      return Core::FE::CellType::hex20;
    case 27:
      return Core::FE::CellType::hex27;
    default:
      FOUR_C_THROW("unexpected number of nodes %d", num_node());
      break;
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::ELEMENTS::Ale3::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::ELEMENTS::Ale3::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer base_buffer(basedata);
  Element::unpack(base_buffer);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::ELEMENTS::Ale3::print(std::ostream& os) const
{
  os << "Ale3 ";
  Element::print(os);
  std::cout << std::endl;
  // cout << data_;
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::Ale3::surfaces()
{
  return Core::Communication::element_boundary_factory<Ale3Surface, Ale3>(
      Core::Communication::buildSurfaces, *this);
}

FOUR_C_NAMESPACE_CLOSE
