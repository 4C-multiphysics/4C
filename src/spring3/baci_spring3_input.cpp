/*-----------------------------------------------------------*/
/*! \file
\brief three dimensional spring element


\level 3
*/
/*-----------------------------------------------------------*/

#include "baci_lib_linedefinition.H"
#include "baci_spring3.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Spring3::ReadElement(
    const std::string& eletype, const std::string& distype, DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  // Gruesse aus Barcelona, Martin und Dhruba
  linedef->ExtractDouble("CROSS", crosssec_);


  // set nodal tridas according to input file
  Qnew_.resize(NumNode());
  Qold_.resize(NumNode());

  Qold_ = Qnew_;

  return true;
}
/*------------------------------------------------------------------------*
 | Set cross section area                         (public) mukherjee 04/15|
 *------------------------------------------------------------------------*/
void DRT::ELEMENTS::Spring3::SetCrossSec(const double& crosssec)
{
  crosssec_ = crosssec;
  return;
}
