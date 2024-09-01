/*---------------------------------------------------------------------*/
/*! \file

\brief Implements an acinus element


\level 3

*/
/*---------------------------------------------------------------------*/

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_maxwell_0d_acinus.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::FE;

Discret::ELEMENTS::RedAcinusType Discret::ELEMENTS::RedAcinusType::instance_;

Discret::ELEMENTS::RedAcinusType& Discret::ELEMENTS::RedAcinusType::instance() { return instance_; }

Core::Communication::ParObject* Discret::ELEMENTS::RedAcinusType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::ELEMENTS::RedAcinus* object = new Discret::ELEMENTS::RedAcinus(-1, -1);
  object->unpack(buffer);
  return object;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::RedAcinusType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "RED_ACINUS")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::rcp(new Discret::ELEMENTS::RedAcinus(id, owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::RedAcinusType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::rcp(new Discret::ELEMENTS::RedAcinus(id, owner));
  return ele;
}


/*--------------------------------------------------------------------  *
 | Read RED_ACINUS element line and add element specific parameters     |
 |                                                             (public) |
 |                                                           roth 10/14 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::RedAcinusType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defs = definitions["RED_ACINUS"];

  defs["LINE2"] = Input::LineDefinition::Builder()
                      .add_int_vector("LINE2", 2)
                      .add_named_int("MAT")
                      .add_named_string("TYPE")
                      .add_named_double("AcinusVolume")
                      .add_named_double("AlveolarDuctVolume")
                      .add_optional_named_double("E1_0")
                      .add_optional_named_double("E1_LIN")
                      .add_optional_named_double("E1_EXP")
                      .add_optional_named_double("TAU")
                      .add_optional_named_double("E1_01")
                      .add_optional_named_double("E1_LIN1")
                      .add_optional_named_double("E1_EXP1")
                      .add_optional_named_double("TAU1")
                      .add_optional_named_double("E1_02")
                      .add_optional_named_double("E1_LIN2")
                      .add_optional_named_double("E1_EXP2")
                      .add_optional_named_double("TAU2")
                      .add_optional_named_double("KAPPA")
                      .add_optional_named_double("BETA")
                      .add_optional_named_double("Area")
                      .build();
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                           ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::RedAcinus::RedAcinus(int id, int owner) : Core::Elements::Element(id, owner) {}


/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::RedAcinus::RedAcinus(const Discret::ELEMENTS::RedAcinus& old)
    : Core::Elements::Element(old),
      elem_type_(old.elem_type_),
      resistance_(old.elem_type_),
      acinus_params_(old.acinus_params_)
{
}


/*----------------------------------------------------------------------*
 |  Deep copy this instance of RedAcinus and return pointer             |
 |  to it                                                      (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::RedAcinus::clone() const
{
  Discret::ELEMENTS::RedAcinus* newelement = new Discret::ELEMENTS::RedAcinus(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::RedAcinus::shape() const
{
  switch (num_node())
  {
    case 2:
      return Core::FE::CellType::line2;
    case 3:
      return Core::FE::CellType::line3;
    default:
      FOUR_C_THROW("unexpected number of nodes %d", num_node());
      break;
  }
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::RedAcinus::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);

  add_to_pack(data, elem_type_);
  add_to_pack(data, resistance_);

  add_to_pack(data, acinus_params_.volume_relaxed);
  add_to_pack(data, acinus_params_.alveolar_duct_volume);
  add_to_pack(data, acinus_params_.area);
  add_to_pack(data, acinus_params_.volume_init);
  add_to_pack(data, acinus_params_.generation);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::RedAcinus::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer base_buffer(basedata);
  Element::unpack(base_buffer);

  extract_from_pack(buffer, elem_type_);
  extract_from_pack(buffer, resistance_);

  extract_from_pack(buffer, acinus_params_.volume_relaxed);
  extract_from_pack(buffer, acinus_params_.alveolar_duct_volume);
  extract_from_pack(buffer, acinus_params_.area);
  extract_from_pack(buffer, acinus_params_.volume_init);
  extract_from_pack(buffer, acinus_params_.generation);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");

  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                             ismail 01/10|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::RedAcinus::print(std::ostream& os) const
{
  os << "RedAcinus ";
  Element::print(os);

  return;
}

/*-----------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
std::vector<double> Discret::ELEMENTS::RedAcinus::element_center_refe_coords()
{
  //  // update element geometry
  Core::Nodes::Node** nodes = RedAcinus::nodes();

  Core::LinAlg::SerialDenseMatrix mat(num_node(), 3, false);
  for (int i = 0; i < num_node(); ++i)
  {
    const auto& x = nodes[i]->x();
    mat(i, 0) = x[0];
    mat(i, 1) = x[1];
    mat(i, 2) = x[2];
  }

  std::vector<double> centercoords(3, 0);
  for (int i = 0; i < 3; ++i)
  {
    double var = 0;
    for (int j = 0; j < num_node(); ++j)
    {
      var = var + mat(j, i);
    }
    centercoords[i] = var / num_node();
  }

  return centercoords;
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data                     ismail 01/10 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::RedAcinus::vis_names(std::map<std::string, int>& names)
{
  Teuchos::RCP<Core::Mat::Material> mat = material();

  // cast to specific material, because general material does not have vis_names/vis_data
  Teuchos::RCP<Mat::Maxwell0dAcinus> mxwll_0d_acin =
      Teuchos::rcp_dynamic_cast<Mat::Maxwell0dAcinus>(material());
  mxwll_0d_acin->vis_names(names);
}


/*----------------------------------------------------------------------*
 |  Return visualization data (public)                     ismail 02/10 |
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::RedAcinus::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  // cast to specific material, because general material does not have vis_names/vis_data
  Teuchos::RCP<Mat::Maxwell0dAcinus> mxwll_0d_acin =
      Teuchos::rcp_dynamic_cast<Mat::Maxwell0dAcinus>(material());

  return mxwll_0d_acin->vis_data(name, data, this->id());
}


void Discret::ELEMENTS::RedAcinus::update_relaxed_volume(double newVol)
{
  acinus_params_.volume_relaxed = newVol;
}


const Discret::ReducedLung::AcinusParams& Discret::ELEMENTS::RedAcinus::get_acinus_params() const
{
  return acinus_params_;
}

/*----------------------------------------------------------------------*
 |  get vector of lines              (public)              ismail  02/13|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::RedAcinus::lines()
{
  FOUR_C_ASSERT(num_line() == 1, "RED_AIRWAY element must have one and only one line");

  return {Teuchos::rcpFromRef(*this)};
}

FOUR_C_NAMESPACE_CLOSE
