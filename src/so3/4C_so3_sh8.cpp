/*----------------------------------------------------------------------*/
/*! \file

\brief solid shell8 element formulation

\level 1


*----------------------------------------------------------------------*/

#include "4C_so3_sh8.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_so3_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::ELEMENTS::SoSh8Type Discret::ELEMENTS::SoSh8Type::instance_;

Discret::ELEMENTS::SoSh8Type& Discret::ELEMENTS::SoSh8Type::instance() { return instance_; }

Core::Communication::ParObject* Discret::ELEMENTS::SoSh8Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::SoSh8(-1, -1);
  object->unpack(buffer);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoSh8Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::rcp(new Discret::ELEMENTS::SoSh8(id, owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoSh8Type::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::rcp(new Discret::ELEMENTS::SoSh8(id, owner));
  return ele;
}


void Discret::ELEMENTS::SoSh8Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::SoSh8Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::ELEMENTS::SoSh8Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX8"] = Input::LineDefinition::Builder()
                     .add_int_vector("HEX8", 8)
                     .add_named_int("MAT")
                     .add_named_string("KINEM")
                     .add_named_string("EAS")
                     .add_named_string("ANS")
                     .add_named_string("THICKDIR")
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
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoSh8::SoSh8(int id, int owner)
    : Discret::ELEMENTS::SoHex8(id, owner),
      thickdir_(undefined),
      anstype_(ansnone),
      nodes_rearranged_(false),
      thickvec_(3, 0.0)
{
  Teuchos::RCP<const Teuchos::ParameterList> params =
      Global::Problem::instance()->get_parameter_list();
  if (params != Teuchos::null)
  {
    Discret::ELEMENTS::UTILS::throw_error_fd_material_tangent(
        Global::Problem::instance()->structural_dynamic_params(), get_element_type_string());
  }

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoSh8::SoSh8(const Discret::ELEMENTS::SoSh8& old)
    : Discret::ELEMENTS::SoHex8(old),
      thickdir_(old.thickdir_),
      anstype_(old.anstype_),
      nodes_rearranged_(old.nodes_rearranged_),
      thickvec_(old.thickvec_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::SoSh8::clone() const
{
  auto* newelement = new Discret::ELEMENTS::SoSh8(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoSh8::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class So_hex8 Element
  Discret::ELEMENTS::SoHex8::pack(data);
  // thickdir
  add_to_pack(data, thickdir_);
  add_to_pack(data, thickvec_);
  add_to_pack(data, anstype_);
  add_to_pack(data, nodes_rearranged_);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoSh8::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class So_hex8 Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer basedata_buffer(basedata);
  Discret::ELEMENTS::SoHex8::unpack(basedata_buffer);
  // thickdir
  thickdir_ = static_cast<ThicknessDirection>(extract_int(buffer));
  extract_from_pack(buffer, thickvec_);
  anstype_ = static_cast<ANSType>(extract_int(buffer));
  nodes_rearranged_ = extract_int(buffer);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoSh8::print(std::ostream& os) const
{
  os << "So_sh8 ";
  Element::print(os);
  std::cout << std::endl;
  return;
}

FOUR_C_NAMESPACE_CLOSE
