/*----------------------------------------------------------------------*/
/*! \file
\brief input for the artery element

\level 3


*----------------------------------------------------------------------*/

#include "baci_art_net_artery.H"
#include "baci_lib_linedefinition.H"
#include "baci_mat_cnst_1d_art.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Artery::ReadElement(
    const std::string& eletype, const std::string& distype, DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  int ngp;
  linedef->ExtractInt("GP", ngp);

  switch (ngp)
  {
    case 1:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_1point;
      break;
    case 2:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_2point;
      break;
    case 3:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_3point;
      break;
    case 4:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_4point;
      break;
    case 5:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_5point;
      break;
    case 6:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_6point;
      break;
    case 7:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_7point;
      break;
    case 8:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_8point;
      break;
    case 9:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_9point;
      break;
    case 10:
      gaussrule_ = CORE::DRT::UTILS::GaussRule1D::line_10point;
      break;
    default:
      dserror("Reading of ART element failed: Gaussrule for line not supported!\n");
  }

  // read artery implementation type
  std::string impltype;
  linedef->ExtractString("TYPE", impltype);

  if (impltype == "Undefined")
    impltype_ = INPAR::ARTDYN::impltype_undefined;
  else if (impltype == "LinExp")
    impltype_ = INPAR::ARTDYN::impltype_lin_exp;
  else if (impltype == "PressureBased")
    impltype_ = INPAR::ARTDYN::impltype_pressure_based;
  else
    dserror("Invalid implementation type for ARTERY elements!");

  // extract diameter
  double diam = 0.0;
  linedef->ExtractDouble("DIAM", diam);

  // set diameter in material
  SetDiamInMaterial(diam);

  return true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::ELEMENTS::Artery::SetDiamInMaterial(const double diam)
{
  // now the element knows its material, and we can use it to set the diameter
  Teuchos::RCP<MAT::Material> mat = Material();
  if (mat->MaterialType() == INPAR::MAT::m_cnst_art)
  {
    MAT::Cnst_1d_art* arterymat = dynamic_cast<MAT::Cnst_1d_art*>(mat.get());
    arterymat->SetDiam(diam);
    arterymat->SetDiamInitial(diam);
  }
  else
    dserror("Artery element got unsupported material type %d", mat->MaterialType());
  return;
}
