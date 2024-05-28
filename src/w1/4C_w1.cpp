/*----------------------------------------------------------------------------*/
/*! \file
\brief wall1 element.

\level 1


*/
/*---------------------------------------------------------------------------*/

#include "4C_w1.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_lib_discret.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

DRT::ELEMENTS::Wall1Type DRT::ELEMENTS::Wall1Type::instance_;

DRT::ELEMENTS::Wall1Type& DRT::ELEMENTS::Wall1Type::Instance() { return instance_; }

CORE::COMM::ParObject* DRT::ELEMENTS::Wall1Type::Create(const std::vector<char>& data)
{
  DRT::ELEMENTS::Wall1* object = new DRT::ELEMENTS::Wall1(-1, -1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Wall1Type::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALL")
  {
    if (eledistype != "NURBS4" and eledistype != "NURBS9")
    {
      return Teuchos::rcp(new DRT::ELEMENTS::Wall1(id, owner));
    }
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Wall1Type::Create(const int id, const int owner)
{
  return Teuchos::rcp(new DRT::ELEMENTS::Wall1(id, owner));
}


void DRT::ELEMENTS::Wall1Type::nodal_block_information(
    DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 2;
  dimns = 3;
  nv = 2;
}

CORE::LINALG::SerialDenseMatrix DRT::ELEMENTS::Wall1Type::ComputeNullSpace(
    DRT::Node& node, const double* x0, int const numdof, int const dimnsp)
{
  return ComputeSolid2DNullSpace(node, x0);
}

void DRT::ELEMENTS::Wall1Type::setup_element_definition(
    std::map<std::string, std::map<std::string, INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, INPUT::LineDefinition>& defs = definitions["WALL"];

  defs["QUAD4"] = INPUT::LineDefinition::Builder()
                      .AddIntVector("QUAD4", 4)
                      .AddNamedInt("MAT")
                      .AddNamedString("KINEM")
                      .AddNamedString("EAS")
                      .AddNamedDouble("THICK")
                      .AddNamedString("STRESS_STRAIN")
                      .AddNamedIntVector("GP", 2)
                      .Build();

  defs["QUAD8"] = INPUT::LineDefinition::Builder()
                      .AddIntVector("QUAD8", 8)
                      .AddNamedInt("MAT")
                      .AddNamedString("KINEM")
                      .AddNamedString("EAS")
                      .AddNamedDouble("THICK")
                      .AddNamedString("STRESS_STRAIN")
                      .AddNamedIntVector("GP", 2)
                      .Build();

  defs["QUAD9"] = INPUT::LineDefinition::Builder()
                      .AddIntVector("QUAD9", 9)
                      .AddNamedInt("MAT")
                      .AddNamedString("KINEM")
                      .AddNamedString("EAS")
                      .AddNamedDouble("THICK")
                      .AddNamedString("STRESS_STRAIN")
                      .AddNamedIntVector("GP", 2)
                      .Build();

  defs["TRI3"] = INPUT::LineDefinition::Builder()
                     .AddIntVector("TRI3", 3)
                     .AddNamedInt("MAT")
                     .AddNamedString("KINEM")
                     .AddNamedString("EAS")
                     .AddNamedDouble("THICK")
                     .AddNamedString("STRESS_STRAIN")
                     .AddNamedIntVector("GP", 2)
                     .Build();

  defs["TRI6"] = INPUT::LineDefinition::Builder()
                     .AddIntVector("TRI6", 6)
                     .AddNamedInt("MAT")
                     .AddNamedString("KINEM")
                     .AddNamedString("EAS")
                     .AddNamedDouble("THICK")
                     .AddNamedString("STRESS_STRAIN")
                     .AddNamedIntVector("GP", 2)
                     .Build();

  defs["NURBS4"] = INPUT::LineDefinition::Builder()
                       .AddIntVector("NURBS4", 4)
                       .AddNamedInt("MAT")
                       .AddNamedString("KINEM")
                       .AddNamedString("EAS")
                       .AddNamedDouble("THICK")
                       .AddNamedString("STRESS_STRAIN")
                       .AddNamedIntVector("GP", 2)
                       .Build();

  defs["NURBS9"] = INPUT::LineDefinition::Builder()
                       .AddIntVector("NURBS9", 9)
                       .AddNamedInt("MAT")
                       .AddNamedString("KINEM")
                       .AddNamedString("EAS")
                       .AddNamedDouble("THICK")
                       .AddNamedString("STRESS_STRAIN")
                       .AddNamedIntVector("GP", 2)
                       .Build();
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Wall1LineType::Create(const int id, const int owner)
{
  // return Teuchos::rcp( new Wall1Line( id, owner ) );
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mgit 01/08/|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Wall1::Wall1(int id, int owner)
    : SoBase(id, owner),
      material_(0),
      thickness_(0.0),
      old_step_length_(0.0),
      gaussrule_(CORE::FE::GaussRule2D::undefined),
      wtype_(plane_none),
      stresstype_(w1_none),
      iseas_(false),
      eastype_(eas_vague),
      easdata_(EASData()),
      structale_(false),
      distype_(CORE::FE::CellType::dis_none)
{
  if (GLOBAL::Problem::Instance()->GetProblemType() == GLOBAL::ProblemType::struct_ale)
    structale_ = true;
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mgit 01/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Wall1::Wall1(const DRT::ELEMENTS::Wall1& old)
    : SoBase(old),
      material_(old.material_),
      thickness_(old.thickness_),
      old_step_length_(old.old_step_length_),
      gaussrule_(old.gaussrule_),
      wtype_(old.wtype_),
      stresstype_(old.stresstype_),
      iseas_(old.iseas_),
      eastype_(old.eas_vague),
      easdata_(old.easdata_),
      structale_(old.structale_),
      distype_(old.distype_)
// tsi_couptyp_(old.tsi_couptyp_)

{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Wall1 and return pointer to it (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::Wall1::Clone() const
{
  DRT::ELEMENTS::Wall1* newelement = new DRT::ELEMENTS::Wall1(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          mgit 04/07 |
 *----------------------------------------------------------------------*/
CORE::FE::CellType DRT::ELEMENTS::Wall1::Shape() const { return distype_; }


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Wall1::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);
  // add base class Element
  SoBase::Pack(data);
  // material_
  AddtoPack(data, material_);
  // thickness
  AddtoPack(data, thickness_);
  // plane strain or plane stress information
  AddtoPack(data, wtype_);
  // gaussrule_
  AddtoPack(data, gaussrule_);
  // stresstype
  AddtoPack(data, stresstype_);
  // eas
  AddtoPack(data, iseas_);
  // eas type
  AddtoPack(data, eastype_);
  // eas data
  pack_eas_data(data);
  // structale
  AddtoPack(data, structale_);
  // distype
  AddtoPack(data, distype_);
  // line search
  AddtoPack(data, old_step_length_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Wall1::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  SoBase::Unpack(basedata);
  // material_
  ExtractfromPack(position, data, material_);
  // thickness_
  ExtractfromPack(position, data, thickness_);
  // plane strain or plane stress information_
  wtype_ = static_cast<DimensionalReduction>(ExtractInt(position, data));
  // gaussrule_
  ExtractfromPack(position, data, gaussrule_);
  // stresstype_
  stresstype_ = static_cast<StressType>(ExtractInt(position, data));
  // iseas_
  iseas_ = ExtractInt(position, data);
  // eastype_
  eastype_ = static_cast<EasType>(ExtractInt(position, data));
  // easdata_
  unpack_eas_data(position, data);
  // structale_
  structale_ = ExtractInt(position, data);
  // distype_
  distype_ = static_cast<CORE::FE::CellType>(ExtractInt(position, data));
  // line search
  ExtractfromPack(position, data, old_step_length_);
  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}



/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             mgit 07/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::Wall1::Lines()
{
  return CORE::COMM::ElementBoundaryFactory<Wall1Line, Wall1>(CORE::COMM::buildLines, *this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                          mgit 03/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::Wall1::Surfaces()
{
  return {Teuchos::rcpFromRef(*this)};
}

/*-----------------------------------------------------------------------------*
| Map plane Green-Lagrange strains to 3d                       mayr.mt 05/2014 |
*-----------------------------------------------------------------------------*/
void DRT::ELEMENTS::Wall1::green_lagrange_plane3d(
    const CORE::LINALG::SerialDenseVector& glplane, CORE::LINALG::Matrix<6, 1>& gl3d)
{
  gl3d(0) = glplane(0);               // E_{11}
  gl3d(1) = glplane(1);               // E_{22}
  gl3d(2) = 0.0;                      // E_{33}
  gl3d(3) = glplane(2) + glplane(3);  // 2*E_{12}=E_{12}+E_{21}
  gl3d(4) = 0.0;                      // 2*E_{23}
  gl3d(5) = 0.0;                      // 2*E_{31}

  return;
}

FOUR_C_NAMESPACE_CLOSE