/*----------------------------------------------------------------------*/
/*!
\file elast_iso2pow.cpp
\brief


the input line should read
  MAT 1 ELAST_Iso2Pow C 1 D 1

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
#include "elast_iso2pow.H"
#include "../drt_mat/matpar_material.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::Iso2Pow::Iso2Pow(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  c_(matdata->GetDouble("C")),
  d_(matdata->GetInt("D"))
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::Iso2Pow::Iso2Pow(MAT::ELASTIC::PAR::Iso2Pow* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::Iso2Pow::AddStrainEnergy(
    double& psi,
    const LINALG::Matrix<3,1>& prinv,
    const LINALG::Matrix<3,1>& modinv)
{
  // material Constants c and d
  const double c = params_->c_;
  const int d = params_->d_;

  // strain energy: Psi = C (\overline{II}_{\boldsymbol{C}}-3)^D
  // add to overall strain energy
  psi += c * pow((modinv(1)-3.),d);
}

/*----------------------------------------------------------------------
 *                                                      birzle 11/2014  */
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::Iso2Pow::AddDerivativesModified(
    LINALG::Matrix<3,1>& dPmodI,
    LINALG::Matrix<6,1>& ddPmodII,
    const LINALG::Matrix<3,1>& modinv
)
{
  const double c = params_ -> c_;
  const    int d = params_ -> d_;

  if (d<1)
    dserror("The Elast_Iso2Pow - material only works for positive integer exponents larger than one.");

  if (d==1)
    dPmodI(1) += c*d;
  else
    dPmodI(1) += c*d*pow(modinv(1)-3.,d-1.);

  if (d==1)
    ddPmodII(1) += 0.;
  else if (d==2)
    ddPmodII(1) += c*d*(d-1.);
  else
    ddPmodII(1) += c*d*(d-1.)*pow(modinv(1)-3.,d-2.);


  return;
}


/*----------------------------------------------------------------------*/
