/*!----------------------------------------------------------------------
\file lung_ogden.cpp
\brief
This file contains the routines required for isotropic nearly
incompressible lung tissue

The input line should read

MAT 1 MAT_Struct_LungOgden C 1000. K1 1500. K2 8.5 KAPPA 1000 BETA -2.0 DENS 0.001

<pre>
Maintainer: Lena Wiechert & Sophie Rausch
            {wiechert,rausch}@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
</pre>
*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "lung_ogden.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_bundle.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::LungOgden::LungOgden(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  kappa_(matdata->GetDouble("KAPPA")),
  beta_(matdata->GetDouble("BETA")),
  c_(matdata->GetDouble("C")),
  k1_(matdata->GetDouble("K1")),
  k2_(matdata->GetDouble("K2")),
  dens_(matdata->GetDouble("DENS"))
{
}


Teuchos::RCP<MAT::Material> MAT::PAR::LungOgden::CreateMaterial()
{
  return Teuchos::rcp(new MAT::LungOgden(this));
}


MAT::LungOgdenType MAT::LungOgdenType::instance_;


DRT::ParObject* MAT::LungOgdenType::Create( const std::vector<char> & data )
{
  MAT::LungOgden* lungog = new MAT::LungOgden();
  lungog->Unpack(data);
  return lungog;
}


/*----------------------------------------------------------------------*
 |  Constructor                                      (public)  lw 04/08 |
 *----------------------------------------------------------------------*/
MAT::LungOgden::LungOgden()
  : params_(NULL)
{
}


/*----------------------------------------------------------------------*
 |  Constructor                                (public)   lw 04/08 |
 *----------------------------------------------------------------------*/
MAT::LungOgden::LungOgden(MAT::PAR::LungOgden* params)
  : params_(params)
{
}


/*----------------------------------------------------------------------*
 |  Pack                                             (public)  lw 04/08 |
 *----------------------------------------------------------------------*/
void MAT::LungOgden::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);

  // matid
  int matid = -1;
  if (params_ != NULL) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data,matid);
}

/*----------------------------------------------------------------------*
 |  Unpack                                           (public)  lw 04/08 |
 *----------------------------------------------------------------------*/
void MAT::LungOgden::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid
  int matid;
  ExtractfromPack(position,data,matid);
  params_ = NULL;
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat = DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::LungOgden*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(), MaterialType());
    }

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",data.size(),position);
}

/*----------------------------------------------------------------------*
 |  Evaluate Material                                (public)  lw 04/08 |
 *----------------------------------------------------------------------*

 isochoric part of strain energy function:
 ----------------------------------------

 W_iso = c*((inv/iiinv^(1/3))-3) + k1/(2*k2)*(exp(k2*(1/3*inv/iiinv^(1/3)-1)^2)-1)

 with

 c, k1, k2             input parameters
 inv, iiinv            first and third invariant of right Cauchy-Green tensor

 (based on Holzapfel [1], Ogden [2] and Balzani, Schroeder, Neff [3])

 Note: the anisotropic invariant K found in [3] is replaced with
       1/3*inv*iiinv^(-1/3) here since a fiber dispersion parameter of 1/3 (isotropic
       case) is assumed.


 volumetric part of strain energy function:
 -----------------------------------------

 W_vol = komp*beta^(-2)*(beta*ln(J)+J^(-beta)-1)

 with

 komp                  input parameter (bulk modulus-like)
 beta                  input parameter ([4] proposed beta=9 for rubber-like materials,
                                        [5] proposed beta<-1, very popular choice: -2)
 J                     det(F) determinant of deformation  gradient


 (based on [4])


[1] G.A.Holzapfel, R.W.Ogden, A New Consitutive Framework for Arterial
    Wall Mechanics and a Comparative Study of Material Models, Journal
    of Elasticity 61, 1-48, 2000.
[2] R.W.Ogden, Anisotropy and Nonlinear Elasticity in Arterial Wall
    Mechanics, CISM Course on Biomechanical Modeling, Lectures 2,3, 2006.
[3] D.Balzani, P.Neff, J.Schroeder, G.A.Holzapfel, A Polyconvex
    Framework for Soft Biological Tissues - Adjustment to Experimental
    Data, Report-Preprint No. 22, 2005.
[4] G.A. Holzapfel, Nonlinear Solid Mechanics - A Continuum Approach
    for Engineering, Wiley, 244-245, 2001.
[5] S. Doll, K. Schweizerhof,  On the Development of Volumetric Strain
    Energy Functions, Journal of Applied Mechanics 67, 17-21, 2000.


*/

void MAT::LungOgden::Evaluate(LINALG::Matrix<6,1>* glstrain,
                              LINALG::Matrix<6,6>* cmat,
                              LINALG::Matrix<6,1>* stress)
{
  // material parameters for isochoric part
  double c  = C();             // parameter for ground substance
  double k1 = K1();            // parameter for fiber potential
  double k2 = K2();            // parameter for fiber potential

  // material parameters for volumetric part
  double komp = Kappa();       // bulk modulus-like parameter
  double beta = Beta();        // empirical coefficient

  //--------------------------------------------------------------------------------------
  // build identity tensor I
  LINALG::Matrix<6,1> identity(true);
  for (int i = 0; i < 3; i++)
    identity(i) = 1.;

  // right Cauchy-Green Tensor  C = 2 * E + I
  LINALG::Matrix<6,1> rcg;
  rcg.Update(2.0, *glstrain, 1.0, identity);

  // invariants
  double inv = rcg(0) + rcg(1) + rcg(2);  // 1st invariant, trace
  double iiinv = rcg(0)*rcg(1)*rcg(2)
        + 0.25 * rcg(3)*rcg(4)*rcg(5)
        - 0.25 * rcg(1)*rcg(5)*rcg(5)
        - 0.25 * rcg(2)*rcg(3)*rcg(3)
        - 0.25 * rcg(0)*rcg(4)*rcg(4);    // 3rd invariant, determinant

  double detf = 0.;
  if (iiinv < 0.0)
    dserror("fatal failure in LungOgden material");
  else
    detf = sqrt(iiinv);                   // determinant of deformation gradient

  //--- prepare some constants -----------------------------------------------------------
  const double third = 1./3.;
  const double twthi = 2./3.;
  const double K = third*inv*pow(iiinv,-third);

  //--------------------------------------------------------------------------------------
  // invert C
  LINALG::Matrix<6,1> invc;

  double invdet = 1./iiinv;

  invc(0) = rcg(1)*rcg(2) - 0.25*rcg(4)*rcg(4);
  invc(1) = rcg(0)*rcg(2) - 0.25*rcg(5)*rcg(5);
  invc(2) = rcg(0)*rcg(1) - 0.25*rcg(3)*rcg(3);
  invc(3) = 0.25*rcg(5)*rcg(4) - 0.5*rcg(3)*rcg(2);
  invc(4) = 0.25*rcg(3)*rcg(5) - 0.5*rcg(0)*rcg(4);
  invc(5) = 0.25*rcg(3)*rcg(4) - 0.5*rcg(5)*rcg(1);

  invc.Scale(invdet);

  //--- determine 2nd Piola Kirchhoff stresses pktwo -------------------------------------
  // 1st step: isochoric part
  //=========================

  // S_iso = isochor1 * I + isochor2 *C^(-1)

  double isochor1 = 2.*c*pow(iiinv,-third);                                    // ground substance/elastin fiber part

  if (K>1.)                                                                    // no need to include 1 since in that
  {                                                                            // case the contribution is 0 anyway
    isochor1 += twthi*k1*exp(k2*pow(K-1.,2.))*(K-1.)*pow(iiinv,-third);        // collagen fiber part
  }

  double isochor2 = - twthi*c*pow(iiinv,-third)*inv;                           // ground substance/elastin fiber  part

  if (K>1.)                                                                    // no need to include 1 since in that
  {                                                                            // case the contribution is 0 anyway
    isochor2 -= 2./9.*k1*exp(k2*pow((K-1.),2.))*(K-1.)*pow(iiinv,-third)*inv;  // collagen fiber part
  }

  // contribution: Cinv
  LINALG::Matrix<6,1> pktwoiso(invc, false);
  pktwoiso.Scale(isochor2);

  // contribution: I
  for (int i = 0; i < 3; i++)
    pktwoiso(i) += isochor1;

  // 2nd step: volumetric part
  //==========================
  double scalar = komp/beta*(1.-pow(detf,-beta));

  // initialise PKtwo with volumetric part
  LINALG::Matrix<6,1> pktwovol(invc, false);
  pktwovol.Scale(scalar);

  // 3rd step: add everything up
  //============================
  (*stress)  = pktwoiso;
  (*stress) += pktwovol;

  //--- do elasticity matrix -------------------------------------------------------------
  // ensure that cmat is zero when it enters the computation
  cmat->Scale(0.0);

  // 1st step: isochoric part
  //=========================

  // deltas (see also Holzapfel p.261)
  // note that these deltas serve for the isochoric part only
  double delta1 = 0.;
  if (K>1.)       delta1 += 4./9.*k1*exp(k2*(K-1.)*(K-1.))*pow(iiinv,-twthi)*(2*k2*(K-1.)*(K-1.)+1);
  double delta3 = -4./3.*c*pow(iiinv,-third);
  if (K>1.)       delta3 -= 4./27.*exp(k2*(K-1.)*(K-1.))*k1*pow(iiinv,-twthi)*(3*(K-1.)*pow(iiinv,third)+2*k2*(K-1.)*(K-1.)*inv+inv);
  double delta6 = 4./9.*c*inv*pow(iiinv,-third);
  if (K>1.)       delta6 += 4./81.*k1*exp(k2*(K-1.)*(K-1.))*inv*pow(iiinv,-twthi)*(3*(K-1.)*pow(iiinv,third)+2*k2*(K-1.)*(K-1.)*inv+inv);
  double delta7 = 4./3.*c*inv*pow(iiinv,-third);
  if (K>1.)       delta7 += 4./9.*k1*exp(k2*(K-1.)*(K-1.))*(K-1.)*inv*pow(iiinv,-third);

  // 2nd step: volumetric part
  //==========================
  delta6 += komp*pow(detf,-beta);
  delta7 -= 2.*scalar;

  // contribution: I \obtimes I
  if (K>1.)       // delta1 has only contributions of collagen fibers
  {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        (*cmat)(i,j) = delta1;
  }

  // contribution: Cinv \otimes Cinv
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
    {
      // contribution: Cinv \otimes I + I \otimes Cinv
      (*cmat)(i,j) += delta3 * ( identity(i)*invc(j) + invc(i)*identity(j) );
      // contribution: Cinv \otimes Cinv
      (*cmat)(i,j) += delta6 * invc(i)*invc(j);
    }

  // contribution: boeppel-product
  AddtoCmatHolzapfelProduct((*cmat),invc,delta7);

  return;
}


#endif

