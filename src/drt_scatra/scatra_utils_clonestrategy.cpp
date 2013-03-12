/*----------------------------------------------------------------------*/
/*!
\file scatra_utils_clonestrategy.cpp

\brief mesh clone strategy for scalar transport problems

<pre>
Maintainer: Georg Bauer
            bauer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/


#include "scatra_utils_clonestrategy.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_material.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_scatra/scatra_element.H"
#include "../drt_lib/drt_element.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::map<std::string,std::string> SCATRA::ScatraFluidCloneStrategy::ConditionsToCopy()
{
  std::map<std::string,std::string> conditions_to_copy;

  conditions_to_copy.insert(std::pair<std::string,std::string>("TransportDirichlet","Dirichlet"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("TransportPointNeumann","PointNeumann"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("TransportLineNeumann","LineNeumann"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("TransportSurfaceNeumann","SurfaceNeumann"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("TransportVolumeNeumann","VolumeNeumann"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("TransportNeumannInflow","TransportNeumannInflow"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("TaylorGalerkinOutflow","TaylorGalerkinOutflow")); // schott
  conditions_to_copy.insert(std::pair<std::string,std::string>("TaylorGalerkinNeumannInflow","TaylorGalerkinNeumannInflow")); // schott
  conditions_to_copy.insert(std::pair<std::string,std::string>("ReinitializationTaylorGalerkin","ReinitializationTaylorGalerkin")); // schott
  // when the fluid problem is periodic we also expect the mass transport to be so:
  conditions_to_copy.insert(std::pair<std::string,std::string>("LinePeriodic","LinePeriodic"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("SurfacePeriodic","SurfacePeriodic"));
  // when the fluid problem has a turbulent inflow section, we also expect this section for scatra:
  conditions_to_copy.insert(std::pair<std::string,std::string>("TurbulentInflowSection","TurbulentInflowSection"));

  conditions_to_copy.insert(std::pair<std::string,std::string>("LineNeumann","FluidLineNeumann"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("SurfaceNeumann","FluidSurfaceNeumann"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("VolumeNeumann","FluidVolumeNeumann"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("KrylovSpaceProjection","KrylovSpaceProjection"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("ElectrodeKinetics","ElectrodeKinetics"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("ScaTraFluxCalc","ScaTraFluxCalc"));
  conditions_to_copy.insert(std::pair<std::string,std::string>("Initfield","Initfield"));

  // for moving boundary problems
  conditions_to_copy.insert(std::pair<std::string,std::string>("FSICoupling","FSICoupling"));

  // mortar meshtying
  conditions_to_copy.insert(std::pair<std::string,std::string>("Mortar","Mortar"));

  // for coupled scalar transport fields
  conditions_to_copy.insert(std::pair<std::string,std::string>("ScaTraCoupling","ScaTraCoupling"));

  return conditions_to_copy;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SCATRA::ScatraFluidCloneStrategy::CheckMaterialType(const int matid)
{
// We take the material with the ID specified by the user
// Here we check first, whether this material is of admissible type
INPAR::MAT::MaterialType mtype = DRT::Problem::Instance()->Materials()->ById(matid)->Type();
if ((mtype != INPAR::MAT::m_scatra) &&
    (mtype != INPAR::MAT::m_mixfrac) &&
    (mtype != INPAR::MAT::m_sutherland) &&
    (mtype != INPAR::MAT::m_arrhenius_pv) &&
    (mtype != INPAR::MAT::m_ferech_pv) &&
    (mtype != INPAR::MAT::m_ion) &&
    (mtype != INPAR::MAT::m_biofilm) &&
    (mtype != INPAR::MAT::m_th_fourier_iso) &&
    (mtype != INPAR::MAT::m_thermostvenant) &&
    (mtype != INPAR::MAT::m_yoghurt) &&
    (mtype != INPAR::MAT::m_matlist) &&
    (mtype != INPAR::MAT::m_myocard))
  dserror("Material with ID %d is not admissible for scalar transport elements",matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SCATRA::ScatraFluidCloneStrategy::SetElementData(
    Teuchos::RCP<DRT::Element> newele,
    DRT::Element* oldele,
    const int matid,
    const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // note: SetMaterial() was reimplemented by the transport element!
#if defined(D_FLUID3)
      DRT::ELEMENTS::Transport* trans = dynamic_cast<DRT::ELEMENTS::Transport*>(newele.get());
      if (trans!=NULL)
      {
        trans->SetMaterial(matid,oldele);
        trans->SetDisType(oldele->Shape()); // set distype as well!
      }
      else
#endif
    {
      dserror("unsupported element type '%s'", typeid(*newele).name());
    }
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool SCATRA::ScatraFluidCloneStrategy::DetermineEleType(
    DRT::Element* actele,
    const bool ismyele,
    std::vector<std::string>& eletype)
{
  // note: ismyele, actele remain unused here! Used only for ALE creation

  // we only support transport elements here
  eletype.push_back("TRANSP");

  return true; // yes, we copy EVERY element (no submeshes)
}

