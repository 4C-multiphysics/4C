/*----------------------------------------------------------------------*/
/*!
\file elast_coupansioexpotwocoup.cpp
\brief

the input line should read
    MAT 1 ELAST_CoupAnsioExpoTwoCoup A4 18472 B4 16.026 A6 2.481 B6 11.120 A8 216 B8 11.436 GAMMA 35.0 INIT_MODE 0

<pre>
Maintainer: Andreas Nagler
            nagler@lnm.mw.tum.de
            089/289 15255
</pre>
*/

/*----------------------------------------------------------------------*/
/* headers */
#include "elast_coupanisoexpotwocoup.H"
#include "../drt_lib/drt_linedefinition.H"

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::ELASTIC::PAR::CoupAnisoExpoTwoCoup::CoupAnisoExpoTwoCoup(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  A4_(matdata->GetDouble("A4")),
  B4_(matdata->GetDouble("B4")),
  A6_(matdata->GetDouble("A6")),
  B6_(matdata->GetDouble("B6")),
  A8_(matdata->GetDouble("A8")),
  B8_(matdata->GetDouble("B8")),
  gamma_(matdata->GetDouble("GAMMA")),
  init_mode_(matdata->GetInt("INIT_MODE"))
{
}


Teuchos::RCP<MAT::Material> MAT::ELASTIC::PAR::CoupAnisoExpoTwoCoup::CreateMaterial()
{
  return Teuchos::null;
  //return Teuchos::rcp( new MAT::ELASTIC::CoupAnisoExpoTwo( this ) );
}


/*----------------------------------------------------------------------*
 |  Constructor                                   (public)  bborn 04/09 |
 *----------------------------------------------------------------------*/
MAT::ELASTIC::CoupAnisoExpoTwoCoup::CoupAnisoExpoTwoCoup()
  : Summand(),
    params_(NULL)
{
}


/*----------------------------------------------------------------------*
 |  Constructor                             (public)   bborn 04/09 |
 *----------------------------------------------------------------------*/
MAT::ELASTIC::CoupAnisoExpoTwoCoup::CoupAnisoExpoTwoCoup(MAT::ELASTIC::PAR::CoupAnisoExpoTwoCoup* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupAnisoExpoTwoCoup::PackSummand(DRT::PackBuffer& data) const
{
  AddtoPack(data,a1_);
  AddtoPack(data,a2_);
  AddtoPack(data,A1_);
  AddtoPack(data,A2_);
  AddtoPack(data,A1A2_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupAnisoExpoTwoCoup::UnpackSummand(const std::vector<char>& data,
                                                  vector<char>::size_type& position)
{
  ExtractfromPack(position,data,a1_);
  ExtractfromPack(position,data,a2_);
  ExtractfromPack(position,data,A1_);
  ExtractfromPack(position,data,A2_);
  ExtractfromPack(position,data,A1A2_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupAnisoExpoTwoCoup::Setup(DRT::INPUT::LineDefinition* linedef)
{
  // fibers aligned in local element cosy with gamma around circumferential direction
  // -> check whether element supports local element cosy
  vector<double> rad;
  vector<double> axi;
  vector<double> cir;
  if (linedef->HaveNamed("RAD") and
      linedef->HaveNamed("AXI") and
      linedef->HaveNamed("CIR"))
  {
    // read local (cylindrical) cosy-directions at current element
    // basis is local cosy with third vec e3 = circumferential dir and e2 = axial dir
    LINALG::Matrix<3,3> locsys(true);
    linedef->ExtractDoubleVector("RAD",rad);
    linedef->ExtractDoubleVector("AXI",axi);
    linedef->ExtractDoubleVector("CIR",cir);
    double radnorm=0.; double axinorm=0.; double cirnorm=0.;

    for (int i = 0; i < 3; ++i)
    {
      radnorm += rad[i]*rad[i]; axinorm += axi[i]*axi[i]; cirnorm += cir[i]*cir[i];
    }
    radnorm = sqrt(radnorm); axinorm = sqrt(axinorm); cirnorm = sqrt(cirnorm);

    for (int i=0; i<3; ++i)
    {
      locsys(i,0) = rad[i]/radnorm;
      locsys(i,1) = axi[i]/axinorm;
      locsys(i,2) = cir[i]/cirnorm;
    }

    SetFiberVecs(locsys);
  }
  else
  {
    dserror("Reading of element local cosy for anisotropic materials failed");
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupAnisoExpoTwoCoup::AddStressAnisoPrincipal(
    const LINALG::Matrix<6,1> rcg,
    LINALG::Matrix<6,6>& cmat,
    LINALG::Matrix<6,1>& stress
)
{
  double I4 = 0.0;
  I4 =  A1_(0)*rcg(0) + A1_(1)*rcg(1) + A1_(2)*rcg(2)
      + A1_(3)*rcg(3) + A1_(4)*rcg(4) + A1_(5)*rcg(5);
  double I6 = 0.0;
  I6 =  A2_(0)*rcg(0) + A2_(1)*rcg(1) + A2_(2)*rcg(2)
      + A2_(3)*rcg(3) + A2_(4)*rcg(4) + A2_(5)*rcg(5);
  double I8 = 0.0;
  I8 =  A1A2_(0,0)*rcg(0) + A1A2_(1,1)*rcg(1) + A1A2_(2,2)*rcg(2)
      + 0.5*(A1A2_(0,1)*rcg(3) + A1A2_(1,2)*rcg(4) + A1A2_(0,2)*rcg(5))
      + 0.5*(A1A2_(1,0)*rcg(3) + A1A2_(2,1)*rcg(4) + A1A2_(2,0)*rcg(5));

  // build Voigt (stress-like) version of a1 \otimes a2 + a2 \otimes a1
  LINALG::Matrix<6,1> A1A2sym;
  A1A2sym(0)=2*A1A2_(0,0);
  A1A2sym(1)=2*A1A2_(1,1);
  A1A2sym(2)=2*A1A2_(2,2);
  A1A2sym(3)=A1A2_(0,1)+A1A2_(1,0);
  A1A2sym(4)=A1A2_(1,2)+A1A2_(2,1);
  A1A2sym(5)=A1A2_(0,2)+A1A2_(2,0);

  double A4=params_->A4_;
  double B4=params_->B4_;
  double A6=params_->A6_;
  double B6=params_->B6_;
  double A8=params_->A8_;
  double B8=params_->B8_;


  double gamma = 2.0 * A4 * (I4-1.)* exp(B4 * (I4-1.)*(I4-1.));
  stress.Update(gamma, A1_, 1.0);
  gamma = 2.0 * A6 * (I6-1.)* exp(B6 * (I6-1.)*(I6-1.));
  stress.Update(gamma, A2_, 1.0);
  gamma = A8 * I8 * exp( B8*I8*I8);
  stress.Update(gamma, A1A2sym, 1.0);

  double delta = 2.*(1. + 2.*B4*(I4-1.)*(I4-1.))*2.*A4* exp(B4*(I4-1.)*(I4-1.));
  cmat.MultiplyNT(delta, A1_, A1_, 1.0);
  delta = 2.*(1. + 2.*B6*(I6-1.)*(I6-1.))*2.*A6* exp(B6*(I6-1.)*(I6-1.));
  cmat.MultiplyNT(delta, A2_, A2_, 1.0);
  delta = A8 * exp( B8*I8*I8)*(1 + 2 * B8*I8*I8);
  cmat.MultiplyNT(delta, A1A2sym, A1A2sym, 1.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupAnisoExpoTwoCoup::SetFiberVecs(
    LINALG::Matrix<3,3> locsys
)
{

  // INIT_MODE = 0 : Fiber direction derived from local cosy
  if(params_->init_mode_ == 0)
  {
    // alignment angles gamma_i are read from first entry of then unnecessary vectors a1 and a2
    if ((params_->gamma_<0) || (params_->gamma_ >90)) dserror("Fiber angle not in [0,90]");
    //convert
    const double gamma = (params_->gamma_*PI)/180.;

    for (int i = 0; i < 3; ++i)
    {
     // a1 = cos gamma e3 + sin gamma e2
     a1_(i) = cos(gamma)*locsys(i,2) + sin(gamma)*locsys(i,1);
     // a2 = cos gamma e3 - sin gamma e2
     a2_(i) = cos(gamma)*locsys(i,2) - sin(gamma)*locsys(i,1);
    }
  }
  // INIT_MODE = 1 : Fiber direction aligned to local cosy
  else if (params_->init_mode_ == 1)
  {
    for (int i = 0; i < 3; ++i)
    {
     a1_(i) = locsys(i,0);
     a2_(i) = locsys(i,1);
    }
  }
  else
  {
    dserror("Problem with fiber initialization");
  }

  for (int i = 0; i < 3; ++i) {
    A1_(i) = a1_(i)*a1_(i);
    A2_(i) = a2_(i)*a2_(i);
  }
  A1_(3) = a1_(0)*a1_(1); A1_(4) = a1_(1)*a1_(2); A1_(5) = a1_(0)*a1_(2);
  A2_(3) = a2_(0)*a2_(1); A2_(4) = a2_(1)*a2_(2); A2_(5) = a2_(0)*a2_(2);

  A1A2_.MultiplyNT(a1_,a2_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::CoupAnisoExpoTwoCoup::GetFiberVecs(
    std::vector<LINALG::Matrix<3,1> >& fibervecs ///< vector of all fiber vectors
)
{
  fibervecs.push_back(a1_);
  fibervecs.push_back(a2_);
}
