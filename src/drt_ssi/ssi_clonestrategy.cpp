/*----------------------------------------------------------------------*/
/*! \file
\brief mesh clone strategy for scatra-structure-interaction problems

\level 2


*/
/*----------------------------------------------------------------------*/

#include "ssi_clonestrategy.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_material.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_scatra_ele/scatra_ele.H"
#include "../drt_so3/so_nurbs27.H"
#include "../drt_so3/so3_scatra.H"
#include "../drt_w1/wall1_scatra.H"
#include "../drt_s8/shell8_scatra.H"
#include "../drt_membrane/membrane_scatra.H"



/*----------------------------------------------------------------------*
 | copy conditions to scatra discretization (public)      schmidt 09/17 |
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> SSI::ScatraStructureCloneStrategy::ConditionsToCopy()
{
  std::map<std::string, std::string> conditions_to_copy;

  conditions_to_copy.insert(std::pair<std::string, std::string>("TransportDirichlet", "Dirichlet"));
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("TransportPointNeumann", "PointNeumann"));
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("TransportLineNeumann", "LineNeumann"));
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("TransportSurfaceNeumann", "SurfaceNeumann"));
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("TransportVolumeNeumann", "VolumeNeumann"));
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("TransportNeumannInflow", "TransportNeumannInflow"));
  // for moving boundary problems
  conditions_to_copy.insert(std::pair<std::string, std::string>("FSICoupling", "FSICoupling"));
  // for coupled scalar transport fields
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("ScaTraCoupling", "ScaTraCoupling"));
  // boundary flux evaluation condition for scalar transport
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("ScaTraFluxCalc", "ScaTraFluxCalc"));
  // initial field conditions
  conditions_to_copy.insert(std::pair<std::string, std::string>("Initfield", "Initfield"));
  // scatra-scatra interface coupling conditions
  conditions_to_copy.insert(std::pair<std::string, std::string>("S2ICoupling", "S2ICoupling"));
  // copy partitioning of the scatra field
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("ScatraPartitioning", "ScatraPartitioning"));
  // electrode state of charge conditions
  conditions_to_copy.insert(std::pair<std::string, std::string>("ElectrodeSOC", "ElectrodeSOC"));
  // cell voltage conditions
  conditions_to_copy.insert(std::pair<std::string, std::string>("CellVoltage", "CellVoltage"));
  // calculate total and mean scalar
  conditions_to_copy.insert(
      std::pair<std::string, std::string>("TotalAndMeanScalar", "TotalAndMeanScalar"));
  // cell cycling conditions
  conditions_to_copy.insert(std::pair<std::string, std::string>("CCCVCycling", "CCCVCycling"));
  conditions_to_copy.insert(std::pair<std::string, std::string>("CCCVHalfCycle", "CCCVHalfCycle"));
  // convective heat transfer conditions (Newton's law of heat transfer)
  conditions_to_copy.insert(std::pair<std::string, std::string>(
      "TransportThermoConvections", "TransportThermoConvections"));

  return conditions_to_copy;
}


/*----------------------------------------------------------------------*
 | return SCATRA::ImplType of element (public)            schmidt 09/17 |
 *----------------------------------------------------------------------*/
INPAR::SCATRA::ImplType SSI::ScatraStructureCloneStrategy::GetImplType(
    DRT::Element* ele  //! element whose SCATRA::ImplType shall be determined
)
{
  INPAR::SCATRA::ImplType impltype(INPAR::SCATRA::impltype_undefined);

  // the element type name, needed to cast correctly in the following
  const std::string eletypename = ele->ElementType().Name();

  // tet 4 solid scatra
  if (eletypename == "So_tet4ScatraType")
  {
    impltype =
        (dynamic_cast<DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_tet4, DRT::Element::tet4>*>(ele))
            ->ImplType();
  }
  // tet10 solid scatra
  else if (eletypename == "So_tet10ScatraType")
  {
    impltype =
        (dynamic_cast<DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_tet10, DRT::Element::tet10>*>(
             ele))
            ->ImplType();
  }
  // HEX 8 Elements
  // hex8 solid scatra
  else if (eletypename == "So_hex8ScatraType")
  {
    impltype =
        (dynamic_cast<DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_hex8, DRT::Element::hex8>*>(ele))
            ->ImplType();
  }
  // hex8fbar solid scatra
  else if (eletypename == "So_hex8fbarScatraType")
  {
    impltype =
        (dynamic_cast<DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_hex8fbar, DRT::Element::hex8>*>(
             ele))
            ->ImplType();
  }
  // hex27 solid scatra
  else if (eletypename == "So_hex27ScatraType")
  {
    impltype =
        (dynamic_cast<DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_hex27, DRT::Element::hex27>*>(
             ele))
            ->ImplType();
  }
  // wedge6
  else if (eletypename == "So_weg6ScatraType")
  {
    impltype =
        (dynamic_cast<DRT::ELEMENTS::So3_Scatra<DRT::ELEMENTS::So_weg6, DRT::Element::wedge6>*>(
             ele))
            ->ImplType();
  }
  // wall scatra elements
  else if (eletypename == "Wall1ScatraType")
  {
    impltype = (dynamic_cast<DRT::ELEMENTS::Wall1_Scatra*>(ele))->ImplType();
  }
  // shell scatra elements
  else if (eletypename == "Shell8ScatraType")
  {
    impltype = (dynamic_cast<DRT::ELEMENTS::Shell8_Scatra*>(ele))->ImplType();
  }
  // membrane3 scatra element
  else if (eletypename == "MembraneScatra_tri3Type")
  {
    impltype = (dynamic_cast<DRT::ELEMENTS::MembraneScatra<DRT::Element::tri3>*>(ele))->ImplType();
  }
  // membrane6 scatra element
  else if (eletypename == "MembraneScatra_tri6Type")
  {
    impltype = (dynamic_cast<DRT::ELEMENTS::MembraneScatra<DRT::Element::tri6>*>(ele))->ImplType();
  }
  // membrane4 scatra element
  else if (eletypename == "MembraneScatra_quad4Type")
  {
    impltype = (dynamic_cast<DRT::ELEMENTS::MembraneScatra<DRT::Element::quad4>*>(ele))->ImplType();
  }
  // membrane9 scatra element
  else if (eletypename == "MembraneScatra_quad9Type")
    impltype = (dynamic_cast<DRT::ELEMENTS::MembraneScatra<DRT::Element::quad9>*>(ele))->ImplType();
  else
  {
    if (!(eletypename == "Bele3Type")) return impltype;
    impltype = INPAR::SCATRA::impltype_no_physics;
  }

  return impltype;
}


/*----------------------------------------------------------------------*
 | check if material type is admissible (protected)       schmidt 09/17 |
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategy::CheckMaterialType(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  INPAR::MAT::MaterialType mtype = DRT::Problem::Instance()->Materials()->ById(matid)->Type();
  if ((mtype != INPAR::MAT::m_scatra) && (mtype != INPAR::MAT::m_elchmat) &&
      (mtype != INPAR::MAT::m_electrode) && (mtype != INPAR::MAT::m_matlist) &&
      (mtype != INPAR::MAT::m_matlist_reactions) && (mtype != INPAR::MAT::m_myocard) &&
      (mtype != INPAR::MAT::m_thermostvenant))
    dserror("Material with ID %d is not admissible for scalar transport elements", matid);
}


/*----------------------------------------------------------------------*
 | set the element data (protected)                       schmidt 09/17 |
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategy::SetElementData(
    Teuchos::RCP<DRT::Element> newele, DRT::Element* oldele, const int matid, const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // note: SetMaterial() was reimplemented by the transport element!
  auto* trans = dynamic_cast<DRT::ELEMENTS::Transport*>(newele.get());
  if (trans != nullptr)
  {
    // set material
    trans->SetMaterial(matid, oldele);
    // set distype as well!
    trans->SetDisType(oldele->Shape());

    // now check whether ImplType is reasonable and if set the ImplType
    INPAR::SCATRA::ImplType impltype = SSI::ScatraStructureCloneStrategy::GetImplType(oldele);

    if (impltype == INPAR::SCATRA::impltype_undefined)
    {
      dserror(
          "ScatraStructureCloneStrategy copies scatra discretization from structure "
          "discretization, but the "
          "STRUCTURE elements that are defined in the .dat file are either not meant to be copied "
          "to scatra elements "
          "or the ImplType is set 'Undefined' which is not meaningful for the created scatra "
          "discretization! "
          "Use SOLIDSCATRA, WALLSCATRA or SHELLSCATRA elements with meaningful ImplType instead!");
    }
    else
      trans->SetImplType(impltype);
  }
  else
  {
    dserror("unsupported element type '%s'", typeid(*newele).name());
  }
}


/*----------------------------------------------------------------------*
 | determine the element type (protected)                 schmidt 09/17 |
 *----------------------------------------------------------------------*/
bool SSI::ScatraStructureCloneStrategy::DetermineEleType(
    DRT::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // note: ismyele, actele remain unused here! Used only for ALE creation

  // we only support transport elements here
  eletype.emplace_back("TRANSP");

  return true;  // yes, we copy EVERY element (no submeshes)
}
