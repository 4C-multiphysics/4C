/*----------------------------------------------------------------------*/
/*!
\file elast_volsussmanbathe.cpp
\brief


the input line should read
  MAT 1 ELAST_VolSussmanBathe KAPPA 100

<pre>
Maintainer: Sophie Rausch & Thomas Kloeppel
            rausch,kloeppel@lnm.mw.tum.de
            089/289 15255
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */

/*----------------------------------------------------------------------*/
/* headers */
#include "elast_volsussmanbathe.H"
#include "../drt_mat/matpar_material.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::VolSussmanBathe::VolSussmanBathe(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  kappa_(matdata->GetDouble("KAPPA"))
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::VolSussmanBathe::VolSussmanBathe(MAT::ELASTIC::PAR::VolSussmanBathe* params)
  : params_(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::VolSussmanBathe::AddCoefficientsModified(
  LINALG::Matrix<3,1>& gamma,
  LINALG::Matrix<5,1>& delta,
  const LINALG::Matrix<3,1>& modinv
  )
{

  const double kappa = params_ -> kappa_;

  gamma(2) += kappa*(modinv(2)-1);
  delta(4) += gamma(2) + modinv(2)*kappa;

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::VolSussmanBathe::Add3rdVolDeriv(const LINALG::Matrix<3,1>& modinv, double& d3PsiVolDJ3)
{
  d3PsiVolDJ3 += 0.;
  return;
}


/*----------------------------------------------------------------------*/
