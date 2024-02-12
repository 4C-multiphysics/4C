/*----------------------------------------------------------------------------*/
/*! \file

\brief Input of 2D ALE elements

\level 1

*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
#include "baci_ale_ale2.hpp"
#include "baci_io_linedefinition.hpp"
#include "baci_mat_so3_material.hpp"

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool DRT::ELEMENTS::Ale2::ReadElement(
    const std::string& eletype, const std::string& distype, INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  // get gauss rule
  const CORE::FE::GaussRule2D gaussrule = getOptimalGaussrule(Shape());
  const CORE::FE::IntegrationPoints2D intpoints(gaussrule);
  const int numgp = intpoints.nquad;

  // get material
  Teuchos::RCP<MAT::Material> mat = Material();
  Teuchos::RCP<MAT::So3Material> so3mat = Teuchos::rcp_dynamic_cast<MAT::So3Material>(mat, true);

  // call material setup
  so3mat->Setup(numgp, linedef);
  return true;
}

BACI_NAMESPACE_CLOSE
