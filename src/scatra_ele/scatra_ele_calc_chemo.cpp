/*----------------------------------------------------------------------*/
/*! \file
 \brief main file containing routines for calculation of scatra element with chemotactic terms

\level 2

 *----------------------------------------------------------------------*/

#include "scatra_ele_calc_chemo.H"

#include "scatra_ele_parameter_std.H"
#include "scatra_ele_parameter_timint.H"

#include "lib_globalproblem.H"
#include "lib_discret.H"
#include "lib_element.H"

#include "mat_list_chemotaxis.H"
#include "mat_scatra_mat.H"
#include "mat_list.H"
#include "headers_singleton_owner.H"

//! note for chemotaxis in BACI:
//! assume the following situation: scalar A does follow the gradient of scalar B (i.e. B is the
//! attractant and scalar A the chemotractant) with chemotactic coefficient 3.0
//!
//! the corresponding equations are: \partial_t A = -(3.0*A \nabla B)  (negative since scalar is
//! chemotracted)
//!                              \partial_t B = 0
//!
//! this equation is in BACI achieved by the MAT_scatra_reaction material:
//! ----------------------------------------------------------MATERIALS
//! MAT 1 MAT_matlist_chemotaxis LOCAL No NUMMAT 2 MATIDS 101 102 NUMPAIR 1 PAIRIDS 111 END
//! MAT 101 MAT_scatra DIFFUSIVITY 0.0
//! MAT 102 MAT_scatra DIFFUSIVITY 0.0
//! MAT 111 MAT_scatra_chemotaxis NUMSCAL 2 PAIR -1 1 CHEMOCOEFF 3.0

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>*
DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::Instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = ::UTILS::MakeSingletonMap<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcChemo<distype, probdim>>(
            new ScaTraEleCalcChemo<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].Instance(
      ::UTILS::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 *  constructor---------------------------                              |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::ScaTraEleCalcChemo(
    const int numdofpernode, const int numscal, const std::string& disname)
    : my::ScaTraEleCalc(numdofpernode, numscal, disname),
      numcondchemo_(-1),
      pair_(0),
      chemocoeff_(0)
{
}

/*--------------------------------------------------------------------------- *
 |  calculation of chemotactic element matrix                     thon 06/ 15 |
 *----------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::CalcMatChemo(
    Epetra_SerialDenseMatrix& emat, const int k, const double timefacfac, const double timetaufac,
    const double densnp, const double scatrares, const LINALG::Matrix<nen_, 1>& sgconv,
    const LINALG::Matrix<nen_, 1>& diff)
{
  Teuchos::RCP<varmanager> varmanager = my::scatravarmanager_;

  for (int condnum = 0; condnum < numcondchemo_; condnum++)
  {
    const std::vector<int>& pair = pair_[condnum];    // get stoichometrie
    const double& chemocoeff = chemocoeff_[condnum];  // get chemotaxis coefficient

    if (pair[k] == -1)  // if chemotractant
    {
      // Standard Galerkin terms

      LINALG::Matrix<nen_, nen_> gradgradmatrix(true);
      LINALG::Matrix<nen_, 1> bigterm(true);

      const double chemofac = timefacfac * densnp;
      const int partner = GetPartner(pair);  // Get attracting partner ID

      const LINALG::Matrix<nsd_, 1> gradattractant =
          varmanager->GradPhi(partner);  // Gradient of attracting parnter

      bigterm.MultiplyTN(my::derxy_, gradattractant);
      gradgradmatrix.MultiplyTN(
          my::derxy_, my::derxy_);  // N1,x*N1,x+N1,y*N1,y+... ; N1,x*N2,x+N1,y*N2,y+...

      for (unsigned vi = 0; vi < nen_; vi++)
      {
        const int fvi = vi * my::numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ui++)
        {
          const int fui = ui * my::numdofpernode_ + k;

          emat(fvi, fui) -= chemofac * chemocoeff * bigterm(vi) * my::funct_(ui);
        }
      }

      for (unsigned vi = 0; vi < nen_; vi++)
      {
        const int fvi = vi * my::numdofpernode_ + k;

        for (unsigned ui = 0; ui < nen_; ui++)
        {
          const int fui = ui * my::numdofpernode_ + partner;
          emat(fvi, fui) -=
              chemofac * chemocoeff * my::scatravarmanager_->Phinp(k) * gradgradmatrix(vi, ui);
        }
      }

      if (my::scatrapara_->StabType() != INPAR::SCATRA::stabtype_no_stabilization)
      {
        dserror("stabilization for chemotactic problems is not jet implemented!");
      }  // end stabilization

    }  // pair[i] == -1
  }

}  // end CalcMatChemo


/*--------------------------------------------------------------------------- *
 |  calculation of chemotactic RHS                                thon 06/ 15 |
 *----------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::CalcRHSChemo(
    Epetra_SerialDenseVector& erhs, const int k, const double rhsfac, const double rhstaufac,
    const double scatrares, const double densnp)
{
  Teuchos::RCP<varmanager> varmanager = my::scatravarmanager_;

  for (int condnum = 0; condnum < numcondchemo_; condnum++)
  {
    const std::vector<int>& pair = pair_[condnum];    // get stoichometrie
    const double& chemocoeff = chemocoeff_[condnum];  // get reaction coefficient

    if (pair[k] == -1)  // if chemotractant
    {
      // Standard Galerkin terms

      const int partner = GetPartner(pair);

      LINALG::Matrix<nen_, 1> gradfunctattr(true);
      LINALG::Matrix<nsd_, 1> attractant = varmanager->GradPhi(partner);

      gradfunctattr.MultiplyTN(my::derxy_, attractant);

      const double decoyed = varmanager->Phinp(k);

      for (unsigned vi = 0; vi < nen_; vi++)
      {
        const int fvi = vi * my::numdofpernode_ + k;
        erhs[fvi] += rhsfac * chemocoeff * decoyed * gradfunctattr(vi);
      }


      if (my::scatrapara_->StabType() != INPAR::SCATRA::stabtype_no_stabilization)
      {
        dserror("stabilization for chemotactic problems is not jet implemented!");
      }  // end stabilization

    }  // pair[i] == -1
  }

}  // CalcRHSChemo


/*----------------------------------------------------------------------*
 |  get the attractant id                                   thon 06/ 15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
int DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::GetPartner(const std::vector<int> pair)
{
  int partner = 0;

  for (int numscalar = 0; numscalar < my::numscal_; numscalar++)
  {
    if (pair[numscalar] == 1)
    {
      partner = numscalar;
      break;
    }
  }

  return partner;
}


/*----------------------------------------------------------------------*
 |  get the material constants  (private)                   thon 06/ 15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::GetMaterialParams(
    const DRT::Element* ele,      //!< the element we are dealing with
    std::vector<double>& densn,   //!< density at t_(n)
    std::vector<double>& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,  //!< density at t_(n+alpha_M)
    double& visc,                 //!< fluid viscosity
    const int iquad               //!< id of current gauss point
)
{
  // get the material
  Teuchos::RCP<MAT::Material> material = ele->Material();

  // We may have some chemotactic and some non-chemotactic discretisation.
  // But since the calculation classes are singleton, we have to reset all chemotaxis stuff each
  // time
  ClearChemotaxisTerms();

  if (material->MaterialType() == INPAR::MAT::m_matlist)
  {
    const Teuchos::RCP<const MAT::MatList>& actmat =
        Teuchos::rcp_dynamic_cast<const MAT::MatList>(material);
    if (actmat->NumMat() != my::numscal_) dserror("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP<MAT::Material> singlemat = actmat->MaterialById(matid);

      my::Materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else if (material->MaterialType() == INPAR::MAT::m_matlist_chemotaxis)
  {
    const Teuchos::RCP<const MAT::MatListChemotaxis>& actmat =
        Teuchos::rcp_dynamic_cast<const MAT::MatListChemotaxis>(material);
    if (actmat->NumMat() != my::numscal_) dserror("Not enough materials in MatList.");

    GetChemotaxisCoefficients(
        actmat);  // read all chemotaxis input from material and copy it into local variables

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP<MAT::Material> singlemat = actmat->MaterialById(matid);

      my::Materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else
  {
    my::Materials(material, 0, densn[0], densnp[0], densam[0], visc, iquad);
  }
  return;
}  // ScaTraEleCalc::GetMaterialParams

/*-----------------------------------------------------------------------------------*
 |  Clear all chemotaxtis related class variable (i.e. set them to zero)  thon 06/15 |
 *-----------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::ClearChemotaxisTerms()
{
  numcondchemo_ = 0;

  // We always have to reinitialize these vectors since our elements are singleton
  pair_.resize(numcondchemo_);
  chemocoeff_.resize(numcondchemo_);
}

/*-----------------------------------------------------------------------------------------*
 |  get numcond, pairing list, chemotaxis coefficient from material            thon 06/ 15 |
 *-----------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::GetChemotaxisCoefficients(
    const Teuchos::RCP<const MAT::Material> material  //!< pointer to current material
)
{
  const Teuchos::RCP<const MAT::MatListChemotaxis>& actmat =
      Teuchos::rcp_dynamic_cast<const MAT::MatListChemotaxis>(material);

  if (actmat == Teuchos::null) dserror("cast to MatListChemotaxis failed");

  // We always have to reinitialize these vectors since our elements are singleton
  numcondchemo_ = actmat->NumPair();
  pair_.resize(numcondchemo_);
  chemocoeff_.resize(numcondchemo_);

  for (int i = 0; i < numcondchemo_; i++)
  {
    const int pairid = actmat->PairID(i);
    const Teuchos::RCP<const MAT::ScatraChemotaxisMat>& chemomat =
        Teuchos::rcp_dynamic_cast<const MAT::ScatraChemotaxisMat>(actmat->MaterialById(pairid));

    pair_[i] = *(chemomat->Pair());           // get pairing
    chemocoeff_[i] = chemomat->ChemoCoeff();  // get chemotaxis coefficient
  }
}

/*-------------------------------------------------------------------------------*
 |  calculation of strong residual for stabilization                             |
 | (depending on respective stationary or time-integration scheme)   thon 06/ 15 |
 *-------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcChemo<distype, probdim>::CalcStrongResidual(
    const int k,           //!< index of current scalar
    double& scatrares,     //!< residual of convection-diffusion-reaction eq
    const double densam,   //!< density at t_(n+am)
    const double densnp,   //!< density at t_(n+1)
    const double rea_phi,  //!< reactive contribution
    const double rhsint,   //!< rhs at integration point
    const double tau       //!< the stabilisation parameter
)
{
  // Note: order is important here
  // First the scatrares without chemotaxis..
  my::CalcStrongResidual(k, scatrares, densam, densnp, rea_phi, rhsint, tau);

  // Second chemotaxis to strong residual
  Teuchos::RCP<varmanager> varmanager = my::scatravarmanager_;

  double chemo_phi = 0;

  for (int condnum = 0; condnum < numcondchemo_; condnum++)
  {
    const std::vector<int>& pair = pair_[condnum];    // get stoichometrie
    const double& chemocoeff = chemocoeff_[condnum];  // get reaction coefficient

    if (pair[k] == -1)
    {
      const int partner = GetPartner(pair);
      double laplattractant = 0;

      if (my::use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        LINALG::Matrix<nen_, 1> laplace(true);
        ;
        my::GetLaplacianStrongForm(laplace);
        laplattractant = laplace.Dot(my::ephinp_[partner]);
      }

      LINALG::Matrix<1, 1> chemoderivattr(true);
      chemoderivattr.MultiplyTN(varmanager->GradPhi(partner), varmanager->GradPhi(k));

      chemo_phi += chemocoeff * (chemoderivattr(0, 0) + laplattractant * varmanager->Phinp(k));
    }
  }

  chemo_phi *= my::scatraparatimint_->TimeFac() / my::scatraparatimint_->Dt();

  // Add to residual
  scatrares += chemo_phi;

  return;
}

// template classes

// 1D elements
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::line2, 1>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::line2, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::line2, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::line3, 1>;

// 2D elements
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::tri3, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::tri3, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::tri6, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::quad4, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::quad4, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::quad9, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::nurbs9, 2>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::hex8, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::hex20>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::hex27, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::tet4, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::tet10, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::wedge6>;
template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::pyramid5, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcChemo<DRT::Element::nurbs27>;
