/*----------------------------------------------------------------------*/
/*!
\file drt_elementdefinition.cpp

\brief Central storage of element input line definitions

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/



#include "drt_elementdefinition.H"
#include "drt_parobjectfactory.H"


/*----------------------------------------------------------------------*/
//! Print function to be called from C
/*----------------------------------------------------------------------*/
extern "C"
void PrintElementDatHeader()
{
  DRT::INPUT::ElementDefinition ed;
  ed.PrintElementDatHeaderToStream(std::cout);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::ElementDefinition::PrintElementDatHeaderToStream(std::ostream& stream)
{
  SetupValidElementLines();

  PrintSectionHeader(stream,"STRUCTURE ELEMENTS");

  //  PrintElementLines(stream,"ART");
  PrintElementLines(stream,"BEAM2");
  PrintElementLines(stream,"BEAM2R");
  PrintElementLines(stream,"BEAM3");
  PrintElementLines(stream,"BEAM3II");
  PrintElementLines(stream,"BEAM3EB");
  //PrintElementLines(stream,"Smoothrod");
  //PrintElementLines(stream,"CONSTRELE2");
  //PrintElementLines(stream,"CONSTRELE3");
  //PrintElementLines(stream,"PTET4");
  PrintElementLines(stream,"NSTET4");
  PrintElementLines(stream,"NSTET5");
  PrintElementLines(stream,"SHELL8");
  PrintElementLines(stream,"SOLID3");
  PrintElementLines(stream,"SOLIDH20");
  PrintElementLines(stream,"SOLIDH27");
  PrintElementLines(stream,"SONURBS27");
  PrintElementLines(stream,"SOLIDH8");
  PrintElementLines(stream,"SOLIDH8P1J1");
  PrintElementLines(stream,"SOLIDH8FBAR");
  PrintElementLines(stream,"SOLIDH8PORO");
  PrintElementLines(stream,"SOLIDSH8");
  PrintElementLines(stream,"SOLIDSH8P8");
  PrintElementLines(stream,"SOLIDSHW6");
  PrintElementLines(stream,"SOLIDT10");
  PrintElementLines(stream,"SOLIDT4");
  PrintElementLines(stream,"SOLIDT4PORO");
  PrintElementLines(stream,"SOLIDT4SCATRA");
  PrintElementLines(stream,"SOLIDW6");
  PrintElementLines(stream,"TORSION2");
  PrintElementLines(stream,"TORSION3");
  PrintElementLines(stream,"TRUSS2");
  PrintElementLines(stream,"TRUSS3");
  PrintElementLines(stream,"WALL");


  PrintSectionHeader(stream,"FLUID ELEMENTS");
  PrintElementLines(stream,"COMBUST3");
  PrintElementLines(stream,"FLUID");
  PrintElementLines(stream,"FLUID2");
  PrintElementLines(stream,"FLUID3");
  PrintElementLines(stream,"XDIFF3");
  PrintElementLines(stream,"XFLUID3");

  PrintSectionHeader(stream,"TRANSPORT ELEMENTS");
  //PrintElementLines(stream,"CONDIF2");
  //PrintElementLines(stream,"CONDIF3");
  PrintElementLines(stream,"TRANSP");

  PrintSectionHeader(stream,"ALE ELEMENTS");
  PrintElementLines(stream,"ALE2");
  PrintElementLines(stream,"ALE3");

  //PrintElementLines(stream,"BELE3");
  //PrintElementLines(stream,"VELE3");

  PrintSectionHeader(stream,"THERMO ELEMENTS");
  PrintElementLines(stream,"THERMO");

  PrintSectionHeader(stream,"ARTERY ELEMENTS");
  PrintElementLines(stream,"ART");

  PrintSectionHeader(stream,"REDUCED D AIRWAYS ELEMENTS");
  PrintElementLines(stream,"RED_AIRWAY");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::ElementDefinition::PrintSectionHeader(std::ostream& stream, std::string name, bool color)
{
  std::string blue2light = "";
  std::string bluelight = "";
  std::string redlight = "";
  std::string yellowlight = "";
  std::string greenlight = "";
  std::string magentalight = "";
  std::string endcolor = "";

  if (color)
  {
  }

  unsigned l = name.length();
  stream << redlight << "--";
  for (int i=0; i<std::max<int>(65-l,0); ++i) stream << '-';
  stream << greenlight << name << endcolor << '\n';
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::ElementDefinition::PrintElementLines(std::ostream& stream, std::string name)
{
  if (definitions_.find(name)!=definitions_.end())
  {
    std::map<std::string,LineDefinition>& defs = definitions_[name];
    for (std::map<std::string,LineDefinition>::iterator i=defs.begin(); i!=defs.end(); ++i)
    {
      stream << "// 0 " << name << " " << i->first << " ";
      i->second.Print(stream);
      stream << '\n';
    }
  }
  else
    stream << "no element type '" << name << "' defined\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::INPUT::ElementDefinition::SetupValidElementLines()
{
  DRT::ParObjectFactory::Instance().SetupElementDefinition( definitions_ );
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DRT::INPUT::LineDefinition* DRT::INPUT::ElementDefinition::ElementLines(std::string name, std::string distype)
{
  // This is ugly. But we want to access both maps just once.
  std::map<std::string,std::map<std::string,LineDefinition> >::iterator j = definitions_.find(name);
  if (j!=definitions_.end())
  {
    std::map<std::string,LineDefinition>& defs = j->second;
    std::map<std::string,LineDefinition>::iterator i = defs.find(distype);
    if (i!=defs.end())
    {
      return &i->second;
    }
  }
  return NULL;
}


