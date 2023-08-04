/*----------------------------------------------------------------------*/
/*! \file

\brief Solid Hex8 element with F-bar modification

\level 1

*/
/*----------------------------------------------------------------------*/

#include "baci_so3_hex8fbar.H"

#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_lib_linedefinition.H"
#include "baci_lib_prestress_service.H"
#include "baci_so3_nullspace.H"
#include "baci_so3_prestress.H"
#include "baci_so3_utils.H"
#include "baci_utils_exceptions.H"

DRT::ELEMENTS::So_hex8fbarType DRT::ELEMENTS::So_hex8fbarType::instance_;

DRT::ELEMENTS::So_hex8fbarType& DRT::ELEMENTS::So_hex8fbarType::Instance() { return instance_; }

DRT::ParObject* DRT::ELEMENTS::So_hex8fbarType::Create(const std::vector<char>& data)
{
  auto* object = new DRT::ELEMENTS::So_hex8fbar(-1, -1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8fbarType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == GetElementTypeString())
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::So_hex8fbar(id, owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8fbarType::Create(const int id, const int owner)
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::So_hex8fbar(id, owner));
  return ele;
}


void DRT::ELEMENTS::So_hex8fbarType::NodalBlockInformation(
    DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
  np = 0;
}

CORE::LINALG::SerialDenseMatrix DRT::ELEMENTS::So_hex8fbarType::ComputeNullSpace(
    DRT::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return ComputeSolid3DNullSpace(node, x0);
}

void DRT::ELEMENTS::So_hex8fbarType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, DRT::INPUT::LineDefinition>& defs = definitions[GetElementTypeString()];

  defs["HEX8"]
      .AddIntVector("HEX8", 8)
      .AddNamedInt("MAT")
      .AddNamedString("KINEM")
      .AddOptionalNamedDoubleVector("RAD", 3)
      .AddOptionalNamedDoubleVector("AXI", 3)
      .AddOptionalNamedDoubleVector("CIR", 3)
      .AddOptionalNamedDoubleVector("FIBER1", 3)
      .AddOptionalNamedDoubleVector("FIBER2", 3)
      .AddOptionalNamedDoubleVector("FIBER3", 3)
      .AddOptionalNamedDouble("GROWTHTRIG");
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 07/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8fbar::So_hex8fbar(int id, int owner) : DRT::ELEMENTS::So_hex8(id, owner)
{
  if (::UTILS::PRESTRESS::IsMulf(pstype_))
    prestress_ = Teuchos::rcp(new DRT::ELEMENTS::PreStress(NUMNOD_SOH8, NUMGPT_SOH8 + 1));

  Teuchos::RCP<const Teuchos::ParameterList> params = DRT::Problem::Instance()->getParameterList();
  if (params != Teuchos::null)
  {
    DRT::ELEMENTS::UTILS::ThrowErrorFDMaterialTangent(
        DRT::Problem::Instance()->StructuralDynamicParams(), GetElementTypeString());
  }

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        popp 07/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8fbar::So_hex8fbar(const DRT::ELEMENTS::So_hex8fbar& old)
    : DRT::ELEMENTS::So_hex8(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            popp 07/10|
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::So_hex8fbar::Clone() const
{
  auto* newelement = new DRT::ELEMENTS::So_hex8fbar(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            popp 07/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8fbar::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);
  // add base class So_hex8 Element
  DRT::ELEMENTS::So_hex8::Pack(data);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            popp 07/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8fbar::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class So_hex8 Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  DRT::ELEMENTS::So_hex8::Unpack(basedata);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                             popp 07/10|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8fbar::~So_hex8fbar() { return; }

/*----------------------------------------------------------------------*
 |  print this element (public)                               popp 07/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8fbar::Print(std::ostream& os) const
{
  os << "So_hex8fbar ";
  Element::Print(os);
  std::cout << std::endl;
  std::cout << data_;
  return;
}
