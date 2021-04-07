/*--------------------------------------------------------------------------*/
/*! \file

\brief evaluation of scatra elements for conservation of mass concentration and electronic charge
within isothermal electrodes

\level 2

*/
/*--------------------------------------------------------------------------*/
#include "scatra_ele_calc_elch_electrode.H"
#include "scatra_ele_parameter_elch_manifold.H"
#include "scatra_ele_parameter_std.H"
#include "scatra_ele_parameter_timint.H"
#include "scatra_ele_utils_elch_electrode.H"

#include "../drt_mat/material.H"


/*----------------------------------------------------------------------*
 | singleton access method                                   fang 02/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>*
DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::Instance(const int numdofpernode,
    const int numscal, const std::string& disname, const ScaTraEleCalcElchElectrode* delete_me)
{
  static std::map<std::string, ScaTraEleCalcElchElectrode<distype, probdim>*> instances;

  if (delete_me == nullptr)
  {
    if (instances.find(disname) == instances.end())
      instances[disname] =
          new ScaTraEleCalcElchElectrode<distype, probdim>(numdofpernode, numscal, disname);
  }

  else
  {
    for (auto instance = instances.begin(); instance != instances.end(); ++instance)
    {
      if (instance->second == delete_me)
      {
        delete instance->second;
        instances.erase(instance);
        return nullptr;
      }
    }
    dserror("Could not locate the desired instance. Internal error.");
  }

  return instances[disname];
}


/*----------------------------------------------------------------------*
 | singleton destruction                                     fang 02/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::Done()
{
  // delete singleton
  Instance(0, 0, "", this);
}


/*----------------------------------------------------------------------*
 | protected constructor for singletons                      fang 02/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::ScaTraEleCalcElchElectrode(
    const int numdofpernode, const int numscal, const std::string& disname)
    : myelch::ScaTraEleCalcElch(numdofpernode, numscal, disname),
      elchmanifoldparams_(DRT::ELEMENTS::ScaTraEleParameterElchManifold::Instance(disname))
{
  // replace elch diffusion manager by diffusion manager for electrodes
  my::diffmanager_ = Teuchos::rcp(new ScaTraEleDiffManagerElchElectrode(my::numscal_));

  // replace elch internal variable manager by internal variable manager for electrodes
  my::scatravarmanager_ =
      Teuchos::rcp(new ScaTraEleInternalVariableManagerElchElectrode<my::nsd_, my::nen_>(
          my::numscal_, myelch::elchparams_));

  // replace elch utility class by utility class for electrodes
  myelch::utils_ = DRT::ELEMENTS::ScaTraEleUtilsElchElectrode<distype>::Instance(
      numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------------------------------------*
 | calculate contributions to element matrix and residual (inside loop over all scalars)   fang
 02/15 |
 *----------------------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::CalcMatAndRhs(
    Epetra_SerialDenseMatrix& emat,  //!< element matrix to calculate
    Epetra_SerialDenseVector& erhs,  //!< element rhs to calculate+
    const int k,                     //!< index of current scalar
    const double fac,                //!< domain-integration factor
    const double timefacfac,         //!< domain-integration factor times time-integration factor
    const double rhsfac,      //!< time-integration factor for rhs times domain-integration factor
    const double taufac,      //!< tau times domain-integration factor
    const double timetaufac,  //!< domain-integration factor times tau times time-integration factor
    const double
        rhstaufac,  //!< time-integration factor for rhs times tau times domain-integration factor
    LINALG::Matrix<my::nen_, 1>&
        tauderpot,  //!< derivatives of stabilization parameter w.r.t. electric potential
    double& rhsint  //!< rhs at Gauss point
)
{
  //----------------------------------------------------------------------
  // 1) element matrix: instationary terms arising from transport equation
  //----------------------------------------------------------------------

  if (not my::scatraparatimint_->IsStationary())
    // 1a) element matrix: standard Galerkin mass term
    my::CalcMatMass(emat, k, fac, 1.);

  //--------------------------------------------------------------------
  // 2) element matrix: stationary terms arising from transport equation
  //--------------------------------------------------------------------

  // 2a) element matrix: standard Galerkin diffusive term
  my::CalcMatDiff(emat, k, timefacfac);

  // 2b) element matrix: additional term arising from concentration dependency of diffusion
  // coefficient
  CalcMatDiffCoeffLin(emat, k, timefacfac, VarManager()->GradPhi(k), 1.);

  // 2c) element matrix: conservative part of convective term, needed for deforming electrodes,
  //                     i.e., for scalar-structure interaction
  double vdiv(0.);
  if (my::scatrapara_->IsConservative())
  {
    my::GetDivergence(vdiv, my::evelnp_);
    my::CalcMatConvAddCons(emat, k, timefacfac, vdiv, 1.);
  }

  //----------------------------------------------------------------------------
  // 3) element right hand side vector (negative residual of nonlinear problem):
  //    terms arising from transport equation
  //----------------------------------------------------------------------------

  // 3a) element rhs: standard Galerkin contributions from non-history part of instationary term if
  // needed
  if (not my::scatraparatimint_->IsStationary()) my::CalcRHSLinMass(erhs, k, rhsfac, fac, 1., 1.);

  // 3b) element rhs: standard Galerkin contributions from rhsint vector (contains body force vector
  // and history vector) need to adapt rhsint vector to time integration scheme first
  my::ComputeRhsInt(rhsint, 1., 1., VarManager()->Hist(k));
  my::CalcRHSHistAndSource(erhs, k, fac, rhsint);

  // 3c) element rhs: standard Galerkin diffusion term
  my::CalcRHSDiff(erhs, k, rhsfac);

  // 3d) element rhs: conservative part of convective term, needed for deforming electrodes,
  //                  i.e., for scalar-structure interaction
  if (my::scatrapara_->IsConservative())
    CalcRhsConservativePartOfConvectiveTerm(erhs, k, rhsfac, vdiv);

  //----------------------------------------------------------------------------
  // 4) element matrix: stationary terms arising from potential equation
  // 5) element right hand side vector (negative residual of nonlinear problem):
  //    terms arising from potential equation
  //----------------------------------------------------------------------------
  // see function CalcMatAndRhsOutsideScalarLoop()
}


/*-----------------------------------------------------------------------------------------------------*
 | calculate contributions to element matrix and residual (outside loop over all scalars)   fang
 02/15 |
 *-----------------------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::CalcMatAndRhsOutsideScalarLoop(
    Epetra_SerialDenseMatrix& emat,  //!< element matrix to calculate
    Epetra_SerialDenseVector& erhs,  //!< element rhs to calculate
    const double fac,                //!< domain-integration factor
    const double timefacfac,         //!< domain-integration factor times time-integration factor
    const double rhsfac  //!< time-integration factor for rhs times domain-integration factor
)
{
  //--------------------------------------------------------------------
  // 4) element matrix: stationary terms arising from potential equation
  //--------------------------------------------------------------------

  // element matrix: standard Galerkin terms from potential equation
  CalcMatPotEquDiviOhm(emat, timefacfac, VarManager()->InvF(), VarManager()->GradPot(), 1.);

  //----------------------------------------------------------------------------
  // 5) element right hand side vector (negative residual of nonlinear problem):
  //    terms arising from potential equation
  //----------------------------------------------------------------------------

  // element rhs: standard Galerkin terms from potential equation
  CalcRhsPotEquDiviOhm(erhs, rhsfac, VarManager()->InvF(), VarManager()->GradPot(), 1.);
}


/*----------------------------------------------------------------------------------------------------------------*
 | CalcMat: linearizations of diffusion term and Ohmic overpotential w.r.t. structural displacements
 fang 11/17 |
 *----------------------------------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::CalcDiffODMesh(
    Epetra_SerialDenseMatrix& emat,  //!< element matrix
    const int k,                     //!< index of current scalar
    const int ndofpernodemesh,       //!< number of structural degrees of freedom per node
    const double diffcoeff,          //!< diffusion coefficient
    const double fac,                //!< domain-integration factor
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const double J,       //!< Jacobian determinant det(dx/ds)
    const LINALG::Matrix<my::nsd_, 1>& gradphi,    //!< gradient of current scalar
    const LINALG::Matrix<my::nsd_, 1>& convelint,  //!< convective velocity
    const LINALG::Matrix<1, my::nsd_ * my::nen_>&
        dJ_dmesh  //!< derivatives of Jacobian determinant det(dx/ds) w.r.t. structural
                  //!< displacements
)
{
  // safety check
  if (k != 0) dserror("Invalid species index!");

  // call base class routine to compute linearizations of diffusion term w.r.t. structural
  // displacements
  my::CalcDiffODMesh(
      emat, 0, ndofpernodemesh, diffcoeff, fac, rhsfac, J, gradphi, convelint, dJ_dmesh);

  // call base class routine again to compute linearizations of Ohmic overpotential w.r.t.
  // structural displacements
  my::CalcDiffODMesh(emat, 1, ndofpernodemesh, VarManager()->InvF() * DiffManager()->GetCond(), fac,
      rhsfac, J, VarManager()->GradPot(), convelint, dJ_dmesh);
}


/*--------------------------------------------------------------------------------*
 | CalcMat: linearization of diffusion coefficient in diffusion term   fang 02/15 |
 *--------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::CalcMatDiffCoeffLin(
    Epetra_SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                     //!< index of current scalar
    const double timefacfac,         //!< domain-integration factor times time-integration factor
    const LINALG::Matrix<my::nsd_, 1>& gradphi,  //!< gradient of concentration at GP
    const double scalar  //!< scaling factor for element matrix contributions
)
{
  // linearization of diffusion coefficient in ionic diffusion term (transport equation):
  //
  // (nabla w, D(D(c)) nabla c)
  //
  for (unsigned vi = 0; vi < my::nen_; ++vi)
  {
    for (unsigned ui = 0; ui < my::nen_; ++ui)
    {
      double laplawfrhs_gradphi(0.);
      my::GetLaplacianWeakFormRHS(laplawfrhs_gradphi, gradphi, vi);

      emat(vi * my::numdofpernode_ + k, ui * my::numdofpernode_ + k) +=
          scalar * timefacfac * DiffManager()->GetDerivIsoDiffCoef(k, k) * laplawfrhs_gradphi *
          my::funct_(ui);
    }
  }
}


/*--------------------------------------------------------------------------------------------*
 | CalcMat: potential equation div i with inserted current - ohmic overpotential   fang 02/15 |
 *--------------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::CalcMatPotEquDiviOhm(
    Epetra_SerialDenseMatrix& emat,  //!< element matrix to be filled
    const double timefacfac,         //!< domain-integration factor times time-integration factor
    const double invf,               //!< 1/F
    const LINALG::Matrix<my::nsd_, 1>& gradpot,  //!< gradient of potential at GP
    const double scalar  //!< scaling factor for element matrix contributions
)
{
  for (unsigned vi = 0; vi < my::nen_; ++vi)
  {
    for (unsigned ui = 0; ui < my::nen_; ++ui)
    {
      double laplawf(0.);
      my::GetLaplacianWeakForm(laplawf, ui, vi);

      // linearization of the ohmic term
      //
      // (grad w, 1/F kappa D(grad pot))
      //
      emat(vi * my::numdofpernode_ + my::numscal_, ui * my::numdofpernode_ + my::numscal_) +=
          scalar * timefacfac * invf * DiffManager()->GetCond() * laplawf;

      for (int iscal = 0; iscal < my::numscal_; ++iscal)
      {
        double laplawfrhs_gradpot(0.);
        my::GetLaplacianWeakFormRHS(laplawfrhs_gradpot, gradpot, vi);

        // linearization of the ohmic term with respect to conductivity
        //
        // (grad w, 1/F kappa D(grad pot))
        //
        emat(vi * my::numdofpernode_ + my::numscal_, ui * my::numdofpernode_ + iscal) +=
            scalar * timefacfac * invf * DiffManager()->GetDerivCond(iscal) * my::funct_(ui) *
            laplawfrhs_gradpot;
      }
    }
  }
}

/*--------------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype,
    probdim>::CalcRhsConservativePartOfConvectiveTerm(Epetra_SerialDenseVector& erhs, const int k,
    const double rhsfac, const double vdiv)
{
  double vrhs = rhsfac * my::scatravarmanager_->Phinp(k) * vdiv;
  for (unsigned vi = 0; vi < my::nen_; ++vi)
    erhs[vi * my::numdofpernode_ + k] -= vrhs * my::funct_(vi);
}

/*--------------------------------------------------------------------------------------------*
 | CalcRhs: potential equation div i with inserted current - ohmic overpotential   fang 02/15 |
 *--------------------------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::CalcRhsPotEquDiviOhm(
    Epetra_SerialDenseVector& erhs,  //!< element vector to be filled
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const double invf,    //!< 1./F
    const LINALG::Matrix<my::nsd_, 1>& gradpot,  //!< gradient of potential at GP
    const double scalar  //!< scaling factor for element residual contributions
)
{
  for (unsigned vi = 0; vi < my::nen_; ++vi)
  {
    double laplawfrhs_gradpot(0.);
    my::GetLaplacianWeakFormRHS(laplawfrhs_gradpot, gradpot, vi);

    erhs[vi * my::numdofpernode_ + my::numscal_] -=
        scalar * rhsfac * invf * DiffManager()->GetCond() * laplawfrhs_gradpot;
  }
}


/*----------------------------------------------------------------------*
 | get material parameters                                   fang 02/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype, int probdim>
void DRT::ELEMENTS::ScaTraEleCalcElchElectrode<distype, probdim>::GetMaterialParams(
    const DRT::Element* ele, std::vector<double>& densn, std::vector<double>& densnp,
    std::vector<double>& densam, double& visc, const int iquad)
{
  // get material
  Teuchos::RCP<const MAT::Material> material = ele->Material();

  // evaluate electrode material
  if (material->MaterialType() == INPAR::MAT::m_electrode)
  {
    Utils()->MatElectrode(
        material, VarManager()->Phinp(0), VarManager()->Temperature(), DiffManager());
  }
  else
    dserror("Material type not supported!");
}


// template classes
// 1D elements
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::line2, 1>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::line2, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::line2, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::line3, 1>;

// 2D elements
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::tri3, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::tri3, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::tri6, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::quad4, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::quad4, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::quad9, 2>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::nurbs9, 2>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::hex8, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::hex20>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::hex27, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::tet4, 3>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::tet10, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::wedge6>;
template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::pyramid5, 3>;
// template class DRT::ELEMENTS::ScaTraEleCalcElchElectrode<DRT::Element::nurbs27>;
