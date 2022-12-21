/*----------------------------------------------------------------------*/
/*! \file

\brief input-related methods of the quadratic NURBS 27 element

\level 2


*----------------------------------------------------------------------*/

#include "so_nurbs27.H"
#include "linedefinition.H"
#include "so3_material.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::NURBS::So_nurbs27::ReadElement(
    const std::string& eletype, const std::string& distype, DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  const int numgp = 27;
  SolidMaterial()->Setup(numgp, linedef);

  // read possible gaussian points, obsolete for computation
  std::vector<int> ngp;
  linedef->ExtractIntVector("GP", ngp);
  for (int i = 0; i < 3; ++i)
    if (ngp[i] != 3) dserror("Only version with 3 GP for So_N27 implemented");

  // we expect kintype to be total lagrangian
  kintype_ = INPAR::STR::kinem_nonlinearTotLag;

  // check if material kinematics is compatible to element kinematics
  SolidMaterial()->ValidKinematics(kintype_);

  // Validate that materials doesn't use extended update call.
  if (SolidMaterial()->UsesExtendedUpdate())
    dserror("This element currently does not support the extended update call.");

  return true;
}
