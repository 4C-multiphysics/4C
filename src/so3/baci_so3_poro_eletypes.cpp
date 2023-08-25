/*----------------------------------------------------------------------*/
/*! \file

\brief element types of the 3D solid-poro element


\level 2

*----------------------------------------------------------------------*/

#include "baci_so3_poro_eletypes.H"

#include "baci_lib_linedefinition.H"
#include "baci_so3_poro.H"

/*----------------------------------------------------------------------*
 |  HEX 8 Element                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8PoroType DRT::ELEMENTS::So_hex8PoroType::instance_;

DRT::ELEMENTS::So_hex8PoroType& DRT::ELEMENTS::So_hex8PoroType::Instance() { return instance_; }

DRT::ParObject* DRT::ELEMENTS::So_hex8PoroType::Create(const std::vector<char>& data)
{
  auto* object = new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex8, DRT::Element::hex8>(-1, -1);
  object->Unpack(data);
  return object;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8PoroType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == GetElementTypeString())
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
        new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex8, DRT::Element::hex8>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8PoroType::Create(const int id, const int owner)
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
      new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex8, DRT::Element::hex8>(id, owner));
  return ele;
}

void DRT::ELEMENTS::So_hex8PoroType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>> definitions_hex8;
  So_hex8Type::SetupElementDefinition(definitions_hex8);

  std::map<std::string, DRT::INPUT::LineDefinition>& defs_hex8 = definitions_hex8["SOLIDH8"];

  std::map<std::string, DRT::INPUT::LineDefinition>& defs = definitions[GetElementTypeString()];

  defs["HEX8"] = DRT::INPUT::LineDefinition::Builder(defs_hex8["HEX8"])
                     .AddOptionalNamedDoubleVector("POROANISODIR1", 3)
                     .AddOptionalNamedDoubleVector("POROANISODIR2", 3)
                     .AddOptionalNamedDoubleVector("POROANISODIR3", 3)
                     .AddOptionalNamedDoubleVector("POROANISONODALCOEFFS1", 8)
                     .AddOptionalNamedDoubleVector("POROANISONODALCOEFFS2", 8)
                     .AddOptionalNamedDoubleVector("POROANISONODALCOEFFS3", 8)
                     .Build();
}

int DRT::ELEMENTS::So_hex8PoroType::Initialize(DRT::Discretization& dis)
{
  So_hex8Type::Initialize(dis);
  for (int i = 0; i < dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    auto* actele =
        dynamic_cast<DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex8, DRT::Element::hex8>*>(
            dis.lColElement(i));
    if (!actele) dserror("cast to So_hex8_poro* failed");
    actele->InitElement();
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  TET 4 Element                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_tet4PoroType DRT::ELEMENTS::So_tet4PoroType::instance_;

DRT::ELEMENTS::So_tet4PoroType& DRT::ELEMENTS::So_tet4PoroType::Instance() { return instance_; }

DRT::ParObject* DRT::ELEMENTS::So_tet4PoroType::Create(const std::vector<char>& data)
{
  auto* object = new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet4, DRT::Element::tet4>(-1, -1);
  object->Unpack(data);
  return object;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet4PoroType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == GetElementTypeString())
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
        new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet4, DRT::Element::tet4>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet4PoroType::Create(const int id, const int owner)
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
      new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet4, DRT::Element::tet4>(id, owner));
  return ele;
}

void DRT::ELEMENTS::So_tet4PoroType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>> definitions_tet4;
  So_tet4Type::SetupElementDefinition(definitions_tet4);

  std::map<std::string, DRT::INPUT::LineDefinition>& defs_tet4 = definitions_tet4["SOLIDT4"];

  std::map<std::string, DRT::INPUT::LineDefinition>& defs = definitions[GetElementTypeString()];

  defs["TET4"] = DRT::INPUT::LineDefinition::Builder(defs_tet4["TET4"])
                     .AddOptionalNamedDoubleVector("POROANISODIR1", 3)
                     .AddOptionalNamedDoubleVector("POROANISODIR2", 3)
                     .AddOptionalNamedDoubleVector("POROANISODIR3", 3)
                     .AddOptionalNamedDoubleVector("POROANISONODALCOEFFS1", 4)
                     .AddOptionalNamedDoubleVector("POROANISONODALCOEFFS2", 4)
                     .AddOptionalNamedDoubleVector("POROANISONODALCOEFFS3", 4)
                     .Build();
}

int DRT::ELEMENTS::So_tet4PoroType::Initialize(DRT::Discretization& dis)
{
  So_tet4Type::Initialize(dis);
  for (int i = 0; i < dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    auto* actele =
        dynamic_cast<DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet4, DRT::Element::tet4>*>(
            dis.lColElement(i));
    if (!actele) dserror("cast to So_tet4_poro* failed");
    actele->So3_Poro<DRT::ELEMENTS::So_tet4, DRT::Element::tet4>::InitElement();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  HEX 27 Element                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex27PoroType DRT::ELEMENTS::So_hex27PoroType::instance_;

DRT::ELEMENTS::So_hex27PoroType& DRT::ELEMENTS::So_hex27PoroType::Instance() { return instance_; }

DRT::ParObject* DRT::ELEMENTS::So_hex27PoroType::Create(const std::vector<char>& data)
{
  auto* object = new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex27, DRT::Element::hex27>(-1, -1);
  object->Unpack(data);
  return object;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex27PoroType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == GetElementTypeString())
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
        new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex27, DRT::Element::hex27>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex27PoroType::Create(const int id, const int owner)
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
      new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex27, DRT::Element::hex27>(id, owner));
  return ele;
}

void DRT::ELEMENTS::So_hex27PoroType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>> definitions_hex27;
  So_hex27Type::SetupElementDefinition(definitions_hex27);

  std::map<std::string, DRT::INPUT::LineDefinition>& defs_hex27 = definitions_hex27["SOLIDH27"];

  std::map<std::string, DRT::INPUT::LineDefinition>& defs = definitions[GetElementTypeString()];

  defs["HEX27"] = DRT::INPUT::LineDefinition::Builder(defs_hex27["HEX27"])
                      .AddOptionalNamedDoubleVector("POROANISODIR1", 3)
                      .AddOptionalNamedDoubleVector("POROANISODIR2", 3)
                      .AddOptionalNamedDoubleVector("POROANISODIR3", 3)
                      .Build();
}

int DRT::ELEMENTS::So_hex27PoroType::Initialize(DRT::Discretization& dis)
{
  So_hex27Type::Initialize(dis);
  for (int i = 0; i < dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    auto* actele =
        dynamic_cast<DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_hex27, DRT::Element::hex27>*>(
            dis.lColElement(i));
    if (!actele) dserror("cast to So_hex27_poro* failed");
    actele->So3_Poro<DRT::ELEMENTS::So_hex27, DRT::Element::hex27>::InitElement();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  TET 10 Element                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_tet10PoroType DRT::ELEMENTS::So_tet10PoroType::instance_;

DRT::ELEMENTS::So_tet10PoroType& DRT::ELEMENTS::So_tet10PoroType::Instance() { return instance_; }

DRT::ParObject* DRT::ELEMENTS::So_tet10PoroType::Create(const std::vector<char>& data)
{
  auto* object = new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet10, DRT::Element::tet10>(-1, -1);
  object->Unpack(data);
  return object;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet10PoroType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == GetElementTypeString())
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
        new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet10, DRT::Element::tet10>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet10PoroType::Create(const int id, const int owner)
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
      new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet10, DRT::Element::tet10>(id, owner));
  return ele;
}

void DRT::ELEMENTS::So_tet10PoroType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>> definitions_tet10;
  So_tet10Type::SetupElementDefinition(definitions_tet10);

  std::map<std::string, DRT::INPUT::LineDefinition>& defs_tet10 = definitions_tet10["SOLIDT10"];

  std::map<std::string, DRT::INPUT::LineDefinition>& defs = definitions[GetElementTypeString()];

  defs["TET10"] = DRT::INPUT::LineDefinition::Builder(defs_tet10["TET10"])
                      .AddOptionalNamedDoubleVector("POROANISODIR1", 3)
                      .AddOptionalNamedDoubleVector("POROANISODIR2", 3)
                      .AddOptionalNamedDoubleVector("POROANISODIR3", 3)
                      .Build();
}

int DRT::ELEMENTS::So_tet10PoroType::Initialize(DRT::Discretization& dis)
{
  So_tet10Type::Initialize(dis);
  for (int i = 0; i < dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    auto* actele =
        dynamic_cast<DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::So_tet10, DRT::Element::tet10>*>(
            dis.lColElement(i));
    if (!actele) dserror("cast to So_tet10_poro* failed");
    actele->So3_Poro<DRT::ELEMENTS::So_tet10, DRT::Element::tet10>::InitElement();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  NURBS 27 Element                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_nurbs27PoroType DRT::ELEMENTS::So_nurbs27PoroType::instance_;

DRT::ELEMENTS::So_nurbs27PoroType& DRT::ELEMENTS::So_nurbs27PoroType::Instance()
{
  return instance_;
}

DRT::ParObject* DRT::ELEMENTS::So_nurbs27PoroType::Create(const std::vector<char>& data)
{
  auto* object =
      new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::NURBS::So_nurbs27, DRT::Element::nurbs27>(-1, -1);
  object->Unpack(data);
  return object;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_nurbs27PoroType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == GetElementTypeString())
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
        new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::NURBS::So_nurbs27, DRT::Element::nurbs27>(
            id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_nurbs27PoroType::Create(const int id, const int owner)
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(
      new DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::NURBS::So_nurbs27, DRT::Element::nurbs27>(
          id, owner));
  return ele;
}

void DRT::ELEMENTS::So_nurbs27PoroType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>> definitions_nurbs27;
  NURBS::So_nurbs27Type::SetupElementDefinition(definitions_nurbs27);

  std::map<std::string, DRT::INPUT::LineDefinition>& defs_nurbs27 =
      definitions_nurbs27["SONURBS27"];

  std::map<std::string, DRT::INPUT::LineDefinition>& defs = definitions[GetElementTypeString()];

  defs["NURBS27"] = DRT::INPUT::LineDefinition::Builder(defs_nurbs27["NURBS27"])
                        .AddOptionalNamedDoubleVector("POROANISODIR1", 3)
                        .AddOptionalNamedDoubleVector("POROANISODIR2", 3)
                        .AddOptionalNamedDoubleVector("POROANISODIR3", 3)
                        .Build();
}

int DRT::ELEMENTS::So_nurbs27PoroType::Initialize(DRT::Discretization& dis)
{
  NURBS::So_nurbs27Type::Initialize(dis);
  for (int i = 0; i < dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    auto* actele = dynamic_cast<
        DRT::ELEMENTS::So3_Poro<DRT::ELEMENTS::NURBS::So_nurbs27, DRT::Element::nurbs27>*>(
        dis.lColElement(i));
    if (!actele) dserror("cast to So_nurbs27_poro* failed");
    actele->So3_Poro<DRT::ELEMENTS::NURBS::So_nurbs27, DRT::Element::nurbs27>::InitElement();
  }
  return 0;
}
