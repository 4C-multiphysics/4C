/*----------------------------------------------------------------------*/
/*! \file

\brief utility class supporting element evaluation for concentrated electrolytes

\level 2

 */
/*----------------------------------------------------------------------*/
#include "scatra_ele_utils_elch_diffcond.H"
#include "scatra_ele_calc_elch_diffcond.H"
#include "scatra_ele_calc_elch_diffcond_multiscale.H"

#include "../drt_mat/elchmat.H"
#include "../drt_mat/elchphase.H"
#include "../drt_mat/newman.H"
#include "../drt_mat/newman_multiscale.H"


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<distype>*
DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<distype>::Instance(const int numdofpernode,
    const int numscal, const std::string& disname, const ScaTraEleUtilsElchDiffCond* delete_me)
{
  // each discretization is associated with exactly one instance of this class according to a static
  // map
  static std::map<std::string, ScaTraEleUtilsElchDiffCond<distype>*> instances;

  // check whether instance already exists for current discretization, and perform instantiation if
  // not
  if (delete_me == nullptr)
  {
    if (instances.find(disname) == instances.end())
      instances[disname] = new ScaTraEleUtilsElchDiffCond<distype>(numdofpernode, numscal, disname);
  }

  // destruct instance
  else
  {
    for (auto i = instances.begin(); i != instances.end(); ++i)
    {
      if (i->second == delete_me)
      {
        delete i->second;
        instances.erase(i);
        return nullptr;
      }
    }
    dserror("Could not locate the desired instance. Internal error.");
  }

  // return existing or newly created instance
  return instances[disname];
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<distype>::ScaTraEleUtilsElchDiffCond(
    const int numdofpernode, const int numscal, const std::string& disname)
    : myelectrode::ScaTraEleUtilsElchElectrode(numdofpernode, numscal, disname)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<distype>::MatElchMat(
    const Teuchos::RCP<const MAT::Material>& material, const std::vector<double>& concentrations,
    const double temperature, const INPAR::ELCH::EquPot equpot, const double ffrt,
    const Teuchos::RCP<ScaTraEleDiffManagerElchDiffCond>& diffmanager,
    INPAR::ELCH::DiffCondMat& diffcondmat)
{
  // cast material to electrolyte material
  const Teuchos::RCP<const MAT::ElchMat> elchmat =
      Teuchos::rcp_static_cast<const MAT::ElchMat>(material);

  // safety check
  if (elchmat->NumPhase() != 1) dserror("Can only have a single electrolyte phase at the moment!");

  // extract electrolyte phase
  const Teuchos::RCP<const MAT::Material> elchphase = elchmat->PhaseById(elchmat->PhaseID(0));

  if (elchphase->MaterialType() == INPAR::MAT::m_elchphase)
  {
    // evaluate electrolyte phase
    MatElchPhase(elchphase, concentrations, temperature, equpot, ffrt, diffmanager, diffcondmat);
  }
  else
    dserror("Invalid material type!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<distype>::MatElchPhase(
    const Teuchos::RCP<const MAT::Material>& material, const std::vector<double>& concentrations,
    const double temperature, const INPAR::ELCH::EquPot& equpot, const double& ffrt,
    const Teuchos::RCP<ScaTraEleDiffManagerElchDiffCond>& diffmanager,
    INPAR::ELCH::DiffCondMat& diffcondmat)
{
  // cast material to electrolyte phase
  const Teuchos::RCP<const MAT::ElchPhase> matelchphase =
      Teuchos::rcp_static_cast<const MAT::ElchPhase>(material);

  // set porosity
  diffmanager->SetPhasePoro(matelchphase->Epsilon(), 0);

  // set tortuosity
  diffmanager->SetPhaseTort(matelchphase->Tortuosity(), 0);

  // loop over materials within electrolyte phase
  for (int imat = 0; imat < matelchphase->NumMat(); ++imat)
  {
    const Teuchos::RCP<const MAT::Material> elchPhaseMaterial =
        matelchphase->MatById(matelchphase->MatID(imat));

    switch (elchPhaseMaterial->MaterialType())
    {
      case INPAR::MAT::m_newman:
      case INPAR::MAT::m_newman_multiscale:
      {
        // safety check
        if (matelchphase->NumMat() != 1)
          dserror("Newman material must be the only transported species!");

        // set ion type
        diffcondmat = INPAR::ELCH::diffcondmat_newman;

        // evaluate standard Newman material
        if (elchPhaseMaterial->MaterialType() == INPAR::MAT::m_newman)
        {
          MatNewman(elchPhaseMaterial, concentrations[0], temperature, diffmanager);
        }
        // evaluate multi-scale Newman material
        else
          MatNewmanMultiScale(elchPhaseMaterial, concentrations[0], temperature, diffmanager);

        break;
      }

      case INPAR::MAT::m_ion:
      {
        // set ion type
        diffcondmat = INPAR::ELCH::diffcondmat_ion;

        myelch::MatIon(elchPhaseMaterial, imat, equpot, diffmanager);

        // calculation of conductivity and transference number based on diffusion coefficient and
        // valence
        if (imat == matelchphase->NumMat() - 1)
        {
          diffmanager->CalcConductivity(matelchphase->NumMat(), ffrt, concentrations);
          diffmanager->CalcTransNum(matelchphase->NumMat(), concentrations);
        }

        break;
      }

      default:
      {
        dserror("Invalid material type!");
        break;
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<distype>::MatNewman(
    const Teuchos::RCP<const MAT::Material>& material, const double concentration,
    const double temperature, const Teuchos::RCP<ScaTraEleDiffManagerElchDiffCond>& diffmanager)
{
  // cast material to Newman material
  const Teuchos::RCP<const MAT::Newman> matnewman =
      Teuchos::rcp_static_cast<const MAT::Newman>(material);

  // valence of ionic species
  diffmanager->SetValence(matnewman->Valence(), 0);

  // concentration depending diffusion coefficient
  diffmanager->SetIsotropicDiff(
      matnewman->ComputeDiffusionCoefficient(concentration, temperature), 0);
  // derivation of concentration depending diffusion coefficient wrt concentration
  diffmanager->SetConcDerivIsoDiffCoef(
      matnewman->ComputeConcentrationDerivativeOfDiffusionCoefficient(concentration, temperature),
      0, 0);

  // derivation of concentration depending diffusion coefficient wrt temperature
  diffmanager->SetTempDerivIsoDiffCoef(
      matnewman->ComputeTemperatureDerivativeOfDiffusionCoefficient(concentration, temperature), 0,
      0);

  // concentration depending transference number
  diffmanager->SetTransNum(matnewman->ComputeTransferenceNumber(concentration), 0);
  // derivation of concentration depending transference number wrt all ionic species
  diffmanager->SetDerivTransNum(matnewman->ComputeFirstDerivTrans(concentration), 0, 0);

  // thermodynamic factor of electrolyte solution
  diffmanager->SetThermFac(matnewman->ComputeThermFac(concentration));
  // derivative of conductivity with respect to concentrations
  diffmanager->SetDerivThermFac(matnewman->ComputeFirstDerivThermFac(concentration), 0);

  // conductivity and first derivative can maximally depend on one concentration
  // since time curve is used as input routine
  // conductivity of electrolyte solution
  diffmanager->SetCond(matnewman->ComputeConductivity(concentration, temperature));

  // derivative of electronic conductivity w.r.t. concentration
  diffmanager->SetConcDerivCond(
      matnewman->ComputeConcentrationDerivativeOfConductivity(concentration, temperature), 0);

  // derivative of electronic conductivity w.r.t. temperature
  diffmanager->SetTempDerivCond(
      matnewman->ComputeTemperatureDerivativeOfConductivity(concentration, temperature), 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<distype>::MatNewmanMultiScale(
    const Teuchos::RCP<const MAT::Material>& material, const double concentration,
    const double temperature, const Teuchos::RCP<ScaTraEleDiffManagerElchDiffCond>& diffmanager)
{
  // evaluate standard Newman material
  MatNewman(material, concentration, temperature, diffmanager);

  // cast material and diffusion manager
  const Teuchos::RCP<const MAT::NewmanMultiScale> newmanmultiscale =
      Teuchos::rcp_dynamic_cast<const MAT::NewmanMultiScale>(material);
  if (newmanmultiscale == Teuchos::null) dserror("Invalid material!");
  const Teuchos::RCP<ScaTraEleDiffManagerElchDiffCondMultiScale> diffmanagermultiscale =
      Teuchos::rcp_dynamic_cast<ScaTraEleDiffManagerElchDiffCondMultiScale>(diffmanager);
  if (diffmanagermultiscale == Teuchos::null) dserror("Invalid diffusion manager!");

  // set electronic conductivity
  diffmanagermultiscale->SetSigma(newmanmultiscale->Sigma());
}


// template classes
// 1D elements
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::line2>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::line3>;

// 2D elements
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::quad4>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::quad9>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::tri3>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::tri6>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::nurbs3>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::nurbs9>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::hex8>;
// template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::hex20>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::hex27>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::tet4>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::tet10>;
// template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::wedge6>;
template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::pyramid5>;
// template class DRT::ELEMENTS::ScaTraEleUtilsElchDiffCond<DRT::Element::nurbs27>;
