/*!----------------------------------------------------------------------
\file So3_scatra_eletypes.cpp

<pre>
   Maintainer: Cristobal Bertoglio
               bertoglio@lnm.mw.tum.de
               http://www.lnm.mw.tum.de
               089 - 289-15264
</pre>

*----------------------------------------------------------------------*/

#include "so3_scatra_eletypes.H"
#include "so3_volcoupl.H"

#include "../drt_lib/drt_linedefinition.H"

/*----------------------------------------------------------------------*
 |  HEX 8 Element                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8ScatraType DRT::ELEMENTS::So_hex8ScatraType::instance_;


DRT::ParObject* DRT::ELEMENTS::So_hex8ScatraType::Create( const std::vector<char> & data )
{
   DRT::ELEMENTS::So3_volcoupl<So_hex8,So3_Scatra<DRT::Element::hex8> >* object =
         new DRT::ELEMENTS::So3_volcoupl<So_hex8, So3_Scatra<DRT::Element::hex8> >(-1,-1);
  object->Unpack(data);
  return object;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8ScatraType::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="SOLIDH8SCATRA" )
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So3_volcoupl<So_hex8,
                                                                    So3_Scatra<DRT::Element::hex8> >
                                                                    (id,owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8ScatraType::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So3_volcoupl<So_hex8,
                                                                        So3_Scatra<DRT::Element::hex8> >
                                                                        (id,owner));
  return ele;
}

void DRT::ELEMENTS::So_hex8ScatraType::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{

  std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> >  definitions_hex8;
  So_hex8Type::SetupElementDefinition(definitions_hex8);

  std::map<std::string, DRT::INPUT::LineDefinition>& defs_hex8 =
      definitions_hex8["SOLIDH8"];

  std::map<std::string, DRT::INPUT::LineDefinition>& defs =
      definitions["SOLIDH8SCATRA"];

  defs["HEX8"]=defs_hex8["HEX8"];
}

/*----------------------------------------------------------------------*
 |  init the element (public)                                           |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_hex8ScatraType::Initialize(DRT::Discretization& dis)
{
  for (int i=0; i<dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    DRT::ELEMENTS::So3_volcoupl<So_hex8, DRT::ELEMENTS::So3_Scatra<DRT::Element::hex8> >* actele =
        dynamic_cast<DRT::ELEMENTS::So3_volcoupl<So_hex8, DRT::ELEMENTS::So3_Scatra<DRT::Element::hex8> > * >(dis.lColElement(i));
    if (!actele) dserror("cast to So_hex8_scatra* failed");
    actele->So_hex8::InitJacobianMapping();
    actele->So3_Scatra<DRT::Element::hex8>::InitJacobianMapping();
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  TET 4 Element                                       |
 *----------------------------------------------------------------------*/


DRT::ELEMENTS::So_tet4ScatraType DRT::ELEMENTS::So_tet4ScatraType::instance_;


DRT::ParObject* DRT::ELEMENTS::So_tet4ScatraType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::So3_volcoupl<So_tet4,So3_Scatra<DRT::Element::tet4> >* object =
          new DRT::ELEMENTS::So3_volcoupl<So_tet4, So3_Scatra<DRT::Element::tet4> >(-1,-1);
  object->Unpack(data);
  return object;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet4ScatraType::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="SOLIDT4SCATRA" )
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So3_volcoupl<So_tet4,
                                                                    So3_Scatra<DRT::Element::tet4> >
                                                                    (id,owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet4ScatraType::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So3_volcoupl<So_tet4,
                                                                        So3_Scatra<DRT::Element::tet4> >
                                                                        (id,owner));
  return ele;
}

void DRT::ELEMENTS::So_tet4ScatraType::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{

  std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> >  definitions_tet4;
  So_tet4Type::SetupElementDefinition(definitions_tet4);

  std::map<std::string, DRT::INPUT::LineDefinition>& defs_tet4 =
      definitions_tet4["SOLIDT4"];

  std::map<std::string, DRT::INPUT::LineDefinition>& defs =
      definitions["SOLIDT4SCATRA"];

  defs["TET4"]=defs_tet4["TET4"];
}

/*----------------------------------------------------------------------*
 |  init the element (public)                                           |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_tet4ScatraType::Initialize(DRT::Discretization& dis)
{
  for (int i=0; i<dis.NumMyColElements(); ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    DRT::ELEMENTS::So3_volcoupl<So_tet4, DRT::ELEMENTS::So3_Scatra<DRT::Element::tet4> >* actele =
        dynamic_cast<DRT::ELEMENTS::So3_volcoupl<So_tet4,DRT::ELEMENTS::So3_Scatra<DRT::Element::tet4> > * >(dis.lColElement(i));
    if (!actele) dserror("cast to So_tet4_scatra* failed");
    actele->So_tet4::InitJacobianMapping();
    actele->So3_Scatra<DRT::Element::tet4>::InitJacobianMapping();
  }
  return 0;
}
