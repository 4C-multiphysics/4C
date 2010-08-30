/*----------------------------------------------------------------------*/
/*!
\file elast_isoyeoh.cpp
\brief


the input line should read
  MAT 1 ELAST_IsoYeoh C1 100 C2 0 C3 200

<pre>
Maintainer: Sophie Rausch & Thomas Kloeppel
            rausch,kloeppel@lnm.mw.tum.de
            089/289 15257
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */
#include "elast_isoyeoh.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::IsoYeoh::IsoYeoh(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  c1_(matdata->GetDouble("C1")),
  c2_(matdata->GetDouble("C2")),
  c3_(matdata->GetDouble("C3"))
{
}


Teuchos::RCP<MAT::Material> MAT::ELASTIC::PAR::IsoYeoh::CreateMaterial()
{
  return Teuchos::null;
  //return Teuchos::rcp( new MAT::ELASTIC::IsoYeoh( this ) );
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::IsoYeoh::IsoYeoh()
  : Summand(),
    params_(NULL)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::IsoYeoh::IsoYeoh(MAT::ELASTIC::PAR::IsoYeoh* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::IsoYeoh::AddCoefficientsModified(
  bool& havecoefficients,
  LINALG::Matrix<3,1>& gamma,
  LINALG::Matrix<5,1>& delta,
  const LINALG::Matrix<3,1>& modinv
  )
{
  havecoefficients = havecoefficients or true;

  const double c1 = params_ -> c1_;
  const double c2 = params_ -> c2_;
  const double c3 = params_ -> c3_;

  //
  gamma(0) += 2*c1+4*c2*(modinv(0)-3)+6*c3*(modinv(0)-3)*(modinv(0)-3);

  delta(0) += 8*(c2+3*c3*(modinv(0)-3));

  return;
}


/*----------------------------------------------------------------------*/
#endif // CCADISCRET
