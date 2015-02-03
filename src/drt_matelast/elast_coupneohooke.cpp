/*----------------------------------------------------------------------*/
/*!
\file elast_coupneohooke.cpp
\brief


the input line should read
  MAT 1 ELAST_CoupNeoHooke YOUNG 1 NUE 1

<pre>
Maintainer: Sophie Rausch
            rausch@lnm.mw.tum.de
            089/289 15255
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */

/*----------------------------------------------------------------------*/
/* headers */
#include "elast_coupneohooke.H"
#include "../drt_mat/matpar_material.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::CoupNeoHooke::CoupNeoHooke(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  youngs_(matdata->GetDouble("YOUNG")),
  nue_(matdata->GetDouble("NUE"))
{
  // Material Constants c and beta
  c_ = youngs_/(4.0*(1.0+nue_));
  beta_ = nue_/(1.0-2.0*nue_);

}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::CoupNeoHooke::CoupNeoHooke(MAT::ELASTIC::PAR::CoupNeoHooke* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupNeoHooke::AddStrainEnergy(
    double& psi,
    const LINALG::Matrix<3,1>& prinv,
    const LINALG::Matrix<3,1>& modinv,
    const int eleGID)
{
  // material Constants c and beta
  const double c = params_->c_;
  const double beta = params_->beta_;

  // strain energy: psi = c / beta * (I3^{-beta} - 1) + c * (I1 - 3)
  double psiadd = c * (prinv(0) - 3.);
  if (beta != 0) // take care of possible division by zero in case or Poisson's ratio nu = 0.0
    psiadd += (c / beta) * (pow(prinv(2), -beta) - 1.);

  // add to overall strain energy
  psi += psiadd;
}


/*----------------------------------------------------------------------
 *                                                       birzle 12/2014 */
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupNeoHooke::AddDerivativesPrincipal(
    LINALG::Matrix<3,1>& dPI,
    LINALG::Matrix<6,1>& ddPII,
    const LINALG::Matrix<3,1>& prinv,
    const int eleGID
)
{
  const double beta  = params_->beta_;
  const double c     = params_->c_;

  dPI(0) += c;
  dPI(2) -= c * std::pow(prinv(2),-beta -1.);

  ddPII(2) += c*(beta+1.)*std::pow(prinv(2),-beta-2.);

  return;
}


/*----------------------------------------------------------------------*/
