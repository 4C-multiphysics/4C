/*----------------------------------------------------------------------*/
/*!
\file elast_couplogmixneohooke.cpp
\brief

the input line should read either
  MAT 1 ELAST_CoupLogMixNeoHooke MODE YN C1 1.044E7 C2 0.3
  ( C1 = Young, c2 = nue)
or
  MAT 1 ELAST_CoupLogMixNeoHooke MODE Lame C1 1. C2 1.

<pre>
Maintainer: Philipp Farah
            farah@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>
*/
/*----------------------------------------------------------------------*/
/* headers */
#include "elast_couplogmixneohooke.H"
#include "../drt_mat/matpar_material.H"

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::CoupLogMixNeoHooke::CoupLogMixNeoHooke(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata)
{
  std::string parmode = *(matdata->Get<std::string>("MODE"));
  double c1 = matdata->GetDouble("C1");
  double c2 = matdata->GetDouble("C2");

  if (parmode == "YN")
  {
    if (c2 <= 0.5 and c2 > -1.0)
    {
      lambda_ = (c2 == 0.5) ? 0.0 : c1*c2/((1.0+c2)*(1.0-2.0*c2));
      mue_ = c1/(2.0*(1.0+c2));  // shear modulus
    }
    else
      dserror("Poisson's ratio must be between -1.0 and 0.5!");
  }
  else if (parmode == "Lame")
  {
    mue_ = c1;
    lambda_ = c2;
  }
  else
    dserror("unknown parameter set for NeoHooke material!\n Must be either YN (Young's modulus and Poisson's ratio) or Lame");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::ELASTIC::CoupLogMixNeoHooke::CoupLogMixNeoHooke(MAT::ELASTIC::PAR::CoupLogMixNeoHooke* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupLogMixNeoHooke::AddShearMod(
  bool& haveshearmod,  ///< non-zero shear modulus was added
  double& shearmod     ///< variable to add upon
  ) const
{
  haveshearmod = true;

  shearmod += params_->mue_;

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupLogMixNeoHooke::AddStrainEnergy(
    double& psi,
    const LINALG::Matrix<3,1>& prinv,
    const LINALG::Matrix<3,1>& modinv)
{
  const double lambda=params_->lambda_;
  const double mue=params_->mue_;
  const double sq = std::sqrt(prinv(2));

  // strain energy: Psi = \frac{\mu}{2} (I_{\boldsymbol{C}} - 3)
  //                      - \mu \log(\sqrt{I\!I\!I_{\boldsymbol{C}}})
  //                      + \frac{\lambda}{2} \big((\sqrt{I\!I\!I_{\boldsymbol{C}}} - 1) \big)^2
  // add to overall strain energy
  psi += mue*0.5*(prinv(0)-3.) - mue*log(sq) + lambda*0.5*pow((sq-1.),2.);

}


/*----------------------------------------------------------------------
 *                                                       birzle 12/2014 */
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupLogMixNeoHooke::AddDerivativesPrincipal(
    LINALG::Matrix<3,1>& dPI,
    LINALG::Matrix<6,1>& ddPII,
    const LINALG::Matrix<3,1>& prinv
)
{
  const double lambda=params_->lambda_;
  const double mue=params_->mue_;
  const double sq = std::sqrt(prinv(2));

  dPI(0) += mue*0.5;
  dPI(2) += lambda*(sq-1.)/(2.*sq)-mue/(2.*prinv(2));

  ddPII(2) += lambda/(4.*prinv(2)) + mue/(2.*prinv(2)*prinv(2)) - lambda*(sq-1.)/(4.*sq*sq*sq);


  return;
}


/*----------------------------------------------------------------------*/
