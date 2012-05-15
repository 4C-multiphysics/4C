/*----------------------------------------------------------------------*/
/*!
\file elast_isocub.cpp
\brief


the input line should read
  MAT 1 ELAST_IsoCub C 1

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
#include "elast_isocub.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::IsoCub::IsoCub(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  c_(matdata->GetDouble("C"))
{
}


Teuchos::RCP<MAT::Material> MAT::ELASTIC::PAR::IsoCub::CreateMaterial()
{
  return Teuchos::null;
  //return Teuchos::rcp( new MAT::ELASTIC::IsoCub( this ) );
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::IsoCub::IsoCub()
  : Summand(),
    params_(NULL)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::IsoCub::IsoCub(MAT::ELASTIC::PAR::IsoCub* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::IsoCub::AddCoefficientsModified(
  LINALG::Matrix<3,1>& gamma,
  LINALG::Matrix<5,1>& delta,
  const LINALG::Matrix<3,1>& modinv
  )
{

  const double c = params_ -> c_;

  //
  gamma(0) += 6*c*(modinv(0)-3)*(modinv(0)-3);

  delta(0) += 24*c*(modinv(0)-3);

  return;
}


/*----------------------------------------------------------------------*/
