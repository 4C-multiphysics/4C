/*----------------------------------------------------------------------*/
/*!
\file elast_vologden.cpp
\brief


the input line should read
  MAT 1 ELAST_VolOgden KAPPA 100 BETA -2

<pre>
Maintainer: Sophie Rausch
            rausch,kloeppel@lnm.mw.tum.de
            089/289 15255
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */

/*----------------------------------------------------------------------*/
/* headers */
#include "elast_vologden.H"
#include "../drt_mat/matpar_material.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::VolOgden::VolOgden(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  kappa_(matdata->GetDouble("KAPPA")),
  beta_(matdata->GetDouble("BETA"))
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::VolOgden::VolOgden(MAT::ELASTIC::PAR::VolOgden* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::VolOgden::AddStrainEnergy(
    double& psi,
    const LINALG::Matrix<3,1>& prinv,
    const LINALG::Matrix<3,1>& modinv,
    const int eleGID)
{
  const double kappa = params_ -> kappa_;
  const double beta = params_ -> beta_;

  // strain energy: Psi = \frac {\kappa}{\beta^2}(\beta lnJ + J^{-\beta}-1)
  // add to overall strain energy
  if (beta != 0)
    psi += kappa/(beta*beta)*(beta*log(modinv(2))+pow(modinv(2),-beta)-1.);

}

/*----------------------------------------------------------------------
 *                                                      birzle 11/2014  */
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::VolOgden::AddDerivativesModified(
    LINALG::Matrix<3,1>& dPmodI,
    LINALG::Matrix<6,1>& ddPmodII,
    const LINALG::Matrix<3,1>& modinv,
    const int eleGID
)
{
  const double kappa = params_ -> kappa_;
  const double beta = params_ -> beta_;

  dPmodI(2) += kappa/(beta*beta)*(beta/modinv(2)-beta*pow(modinv(2),-beta-1.));

  ddPmodII(2) += kappa/(beta*beta)*(-beta/(modinv(2)*modinv(2))+beta*(beta+1.)*pow(modinv(2),-beta-2.));

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::VolOgden::Add3rdVolDeriv(const LINALG::Matrix<3,1>& modinv, double& d3PsiVolDJ3)
{
  const double kappa = params_ -> kappa_;
  const double beta = params_ -> beta_;
  const double J=modinv(2);
  d3PsiVolDJ3 += kappa/(beta*J*J*J)*(2.-pow(J,-beta)*(beta*beta+3.*beta+2.));
  return;
}


/*----------------------------------------------------------------------*/
