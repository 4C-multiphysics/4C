/*----------------------------------------------------------------------*/
/*! \file

\brief NStet5 element

\level 2


*----------------------------------------------------------------------*/

#include "baci_lib_linedefinition.H"
#include "baci_mat_elasthyper.H"
#include "baci_so3_nstet5.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::NStet5::ReadElement(
    const std::string& eletype, const std::string& distype, DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  if (Material()->MaterialType() == INPAR::MAT::m_elasthyper)
  {
    MAT::ElastHyper* elahy = dynamic_cast<MAT::ElastHyper*>(Material().get());
    elahy->Setup(0, linedef);
  }

  std::string buffer;
  linedef->ExtractString("KINEM", buffer);
  if (buffer == "linear")
  {
    // kintype_ not yet implemented for nstet5
    // kintype_ = sonstet5_linear;
    dserror("Reading of SO_NSTET5 element failed only nonlinear kinematics implemented");
  }
  else if (buffer == "nonlinear")
  {
    // kintype_ not yet implemented for nstet5
    // kintype_ = sonstet5_nonlinear;
  }
  else
    dserror("Reading SO_NSTET5 element failed KINEM unknown");

  return true;
}
