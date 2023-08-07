/*----------------------------------------------------------------------*/
/*! \file
\brief
This file contains the routines required for aneurysmatic artery wall following
Raghavan and Vorp [2000]

The material is a special case of a generalised pover law neo-Hookean material

This material law accounts for spatial variation of the material parameters young and beta
by using an random field, such that every element gets its own set of mat parameters.

The sample of the random field is not calculated in the material routine but globally in MLMC
because it cannot be stored in each element. At the moment beta is the only stochastic parameter.

the input line should read
  MAT 1 MAT_Struct_AAANeoHooke_Stopro YOUNG 1.044E7 BETA 188.1E5 NUE 0.3 DENS 1.0 SIGMA 0.25
CORRLENGTH 5.0

\level 3


*----------------------------------------------------------------------*/

#include "baci_mat_aaaneohooke_stopro.H"

#include "baci_lib_globalproblem.H"
#include "baci_mat_par_bundle.H"
#include "baci_mat_service.H"


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::PAR::AAAneohooke_stopro::AAAneohooke_stopro(Teuchos::RCP<MAT::PAR::Material> matdata)
    : Parameter(matdata),
      youngs_mean_(matdata->GetDouble("YOUNG")),
      nue_(matdata->GetDouble("NUE")),
      beta_mean_(matdata->GetDouble("BETA")),
      density_(matdata->GetDouble("DENS"))
{
  init_ = 0;
}


Teuchos::RCP<MAT::Material> MAT::PAR::AAAneohooke_stopro::CreateMaterial()
{
  return Teuchos::rcp(new MAT::AAAneohooke_stopro(this));
}

MAT::AAAneohooke_stoproType MAT::AAAneohooke_stoproType::instance_;

DRT::ParObject* MAT::AAAneohooke_stoproType::Create(const std::vector<char>& data)
{
  MAT::AAAneohooke_stopro* aaa = new MAT::AAAneohooke_stopro();
  aaa->Unpack(data);
  return aaa;
}


/*----------------------------------------------------------------------*
 |  Constructor                                   (public)  chfoe 03/08 |
 *----------------------------------------------------------------------*/
MAT::AAAneohooke_stopro::AAAneohooke_stopro()
    : params_(nullptr), isinit_beta_(false), isinit_youngs_(false)
{
}


/*----------------------------------------------------------------------*
 |  Constructor                             (public)   chfoe 03/08 |
 *----------------------------------------------------------------------*/
MAT::AAAneohooke_stopro::AAAneohooke_stopro(MAT::PAR::AAAneohooke_stopro* params)
    : params_(params), beta_(0.0), isinit_beta_(false), youngs_(0.0), isinit_youngs_(false)
{
}

/*----------------------------------------------------------------------*
 |  Pack                                          (public)  chfoe 03/08 |
 *----------------------------------------------------------------------*/
void MAT::AAAneohooke_stopro::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data, matid);
}

/*----------------------------------------------------------------------*
 |  Unpack                                        (public)  chfoe 03/08 |
 *----------------------------------------------------------------------*/
void MAT::AAAneohooke_stopro::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid
  int matid;
  ExtractfromPack(position, data, matid);
  params_ = nullptr;
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat =
          DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::AAAneohooke_stopro*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }

  if (position != data.size()) dserror("Mismatch in size of data %d <-> %d", data.size(), position);
}



// initialize function to be called from MLMC to get beta from a random field
void MAT::AAAneohooke_stopro::Init(double value_stopro, std::string stochpar)
{
  if (!stochpar.compare("BETA"))
  {
    // calculation done in MLMC/elsewhere
    beta_ = value_stopro;
    isinit_beta_ = true;
  }
  else if (!stochpar.compare("YOUNG"))
  {
    youngs_ = value_stopro;
    isinit_youngs_ = true;
  }
  else
  {
    dserror("Unknown parameter in AAAneohooke_stopro::Init()");
  }

  return;
}


void MAT::AAAneohooke_stopro::Evaluate(const CORE::LINALG::Matrix<3, 3>* defgrd,
    const CORE::LINALG::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
    CORE::LINALG::Matrix<6, 1>* stress, CORE::LINALG::Matrix<6, 6>* cmat, const int gp,
    const int eleGID)
{
  double beta = params_->beta_mean_;
  double youngs = params_->youngs_mean_;

  // init check
  if (isinit_beta_ && !isinit_youngs_)
  {
    beta = beta_;
  }
  else if (!isinit_beta_ && isinit_youngs_)
  {
    youngs = youngs_;
  }
  else if (isinit_beta_ && isinit_youngs_)
  {
    beta = beta_;
    youngs = youngs_;
  }
  else
    dserror(
        "Stochastic Parameters of AAAneohooke_stopro have not been initialized! \n "
        "AAAneohooke_stopro for use with MLMC ONLY!!!");

  const double nue = params_->nue_;                     // Poisson's ratio
  const double alpha = youngs * 0.1666666666666666667;  // E = alpha * 6

  // material parameters for volumetric part
  const double beta2 = -2.0;                                           // parameter from Holzapfel
  double komp = (nue != 0.5) ? 2.0 * alpha / (1.0 - 2.0 * nue) : 0.0;  // bulk modulus

  //--------------------------------------------------------------------------------------
  // build identity tensor I
  CORE::LINALG::Matrix<6, 1> identity(true);
  for (int i = 0; i < 3; i++) identity(i) = 1.0;

  // right Cauchy-Green Tensor  C = 2 * E + I
  CORE::LINALG::Matrix<6, 1> rcg(*glstrain);
  rcg.Scale(2.0);
  rcg += identity;

  // invariants
  double inv = rcg(0) + rcg(1) + rcg(2);  // 1st invariant, trace
  double iiinv = rcg(0) * rcg(1) * rcg(2) + 0.25 * rcg(3) * rcg(4) * rcg(5) -
                 0.25 * rcg(1) * rcg(5) * rcg(5) - 0.25 * rcg(2) * rcg(3) * rcg(3) -
                 0.25 * rcg(0) * rcg(4) * rcg(4);  // 3rd invariant, determinante

  double detf = 0.0;
  if (iiinv < 0.0)
    dserror("fatal failure in aneurysmatic artery wall material");
  else
    detf = sqrt(iiinv);  // determinate of deformation gradient

  //--------------------------------------------------------------------------------------
  // invert C
  CORE::LINALG::Matrix<6, 1> invc(false);

  double invdet = 1. / iiinv;

  invc(0) = rcg(1) * rcg(2) - 0.25 * rcg(4) * rcg(4);
  invc(1) = rcg(0) * rcg(2) - 0.25 * rcg(5) * rcg(5);
  invc(2) = rcg(0) * rcg(1) - 0.25 * rcg(3) * rcg(3);
  invc(3) = 0.25 * rcg(5) * rcg(4) - 0.5 * rcg(3) * rcg(2);
  invc(4) = 0.25 * rcg(3) * rcg(5) - 0.5 * rcg(0) * rcg(4);
  invc(5) = 0.25 * rcg(3) * rcg(4) - 0.5 * rcg(5) * rcg(1);

  invc.Scale(invdet);

  //--- prepare some constants -----------------------------------------------------------
  const double third = 1.0 / 3.0;
  const double twthi = 2.0 / 3.0;

  double isochor1 = 0.0;
  double isochor2 = 0.0;

  int deriv = params.get<int>("matparderiv", -1);
  if (deriv == 0)
  {
    // std::cout << "DERIV YOUNGS" << std::endl;
    // deriv. w.r.t YOUNG!! -> factor 0.1666666666666666667 in here
    isochor1 = 2.0 * pow(iiinv, third) * pow(iiinv, -twthi) * 0.1666666666666666667;
    isochor2 = -twthi * inv * pow(iiinv, third) * pow(iiinv, -twthi) * 0.1666666666666666667;

    // do komp too:
    komp = 2.0 / (1.0 - 2.0 * nue) * 0.1666666666666666667;
  }
  else if (deriv == 2)
  {
    // std::cout << "DERIV BETA" << std::endl;
    // deriv. w.r.t beta
    isochor1 = 2.0 * (2.0 * inv - 6.0 * pow(iiinv, third)) * pow(iiinv, -twthi);
    isochor2 = -twthi * inv * (2.0 * inv - 6.0 * pow(iiinv, third)) * pow(iiinv, -twthi);

    // vol part is not a function of beta -> derivative has to be zero
    komp = 0.0;
  }
  else if (deriv == -1)
  {
    //--- determine 2nd Piola Kirchhoff stresses pktwo -------------------------------------
    // 1st step: isochoric part
    //=========================
    isochor1 = 2.0 *
               (alpha * pow(iiinv, third) + 2.0 * beta * inv - 6.0 * beta * pow(iiinv, third)) *
               pow(iiinv, -twthi);
    isochor2 = -twthi * inv *
               (alpha * pow(iiinv, third) + 2.0 * beta * inv - 6.0 * beta * pow(iiinv, third)) *
               pow(iiinv, -twthi);
  }
  else
    dserror("give valid parameter for differentiation");

  // contribution: Cinv
  CORE::LINALG::Matrix<6, 1> pktwoiso(invc);
  pktwoiso.Scale(isochor2);

  // contribution: I
  for (int i = 0; i < 3; i++) pktwoiso(i) += isochor1;


  // 2nd step: volumetric part
  //==========================
  double scalar = komp / beta2 * (1.0 - pow(detf, -beta2));

  // initialise PKtwo with volumetric part
  CORE::LINALG::Matrix<6, 1> pktwovol(invc);
  pktwovol.Scale(scalar);

  // 3rd step: add everything up
  //============================
  (*stress) = pktwoiso;
  (*stress) += pktwovol;


  //--- do elasticity matrix -------------------------------------------------------------
  // ensure that cmat is zero when it enters the computation
  // It is an implicit law that cmat is zero upon input
  // cmat.PutScalar(0.0);

  // 1st step: isochoric part
  //=========================

  // deltas (see also Holzapfel p.261)
  // note that these deltas serve for the isochoric part only
  double delta1 = 8.0 * beta * pow(iiinv, -twthi);
  double delta3 = -4. / 3 *
                  (alpha * pow(iiinv, third) + 4. * beta * inv - 6 * beta * pow(iiinv, third)) *
                  pow(iiinv, -twthi);
  double delta6 = 4. / 9 * inv *
                  (alpha * pow(iiinv, third) + 4. * beta * inv - 6 * beta * pow(iiinv, third)) *
                  pow(iiinv, -twthi);
  double delta7 = 4. / 3 * inv *
                  (alpha * pow(iiinv, third) + 2. * beta * inv - 6 * beta * pow(iiinv, third)) *
                  pow(iiinv, -twthi);

  // contribution: I \obtimes I
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) (*cmat)(i, j) = delta1;

  // contribution: Cinv \otimes Cinv
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
    {
      // contribution: Cinv \otimes I + I \otimes Cinv
      (*cmat)(i, j) += delta3 * (identity(i) * invc(j) + invc(i) * identity(j));
      // contribution: Cinv \otimes Cinv
      (*cmat)(i, j) += delta6 * invc(i) * invc(j);
    }

  // contribution: boeppel-product
  AddtoCmatHolzapfelProduct(*cmat, invc, delta7);

  // 2nd step: volumetric part
  //==========================
  delta6 = komp * pow(detf, -beta2);
  delta7 = -2.0 * scalar;

  // contribution: Cinv \otimes Cinv
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) (*cmat)(i, j) += delta6 * invc(i) * invc(j);

  // contribution: boeppel-product
  AddtoCmatHolzapfelProduct(*cmat, invc, delta7);

  return;
}
/*----------------------------------------------------------------------*
 |  Calculate strain energy                                biehler 12/13|
 *----------------------------------------------------------------------*/
void MAT::AAAneohooke_stopro::StrainEnergy(
    const CORE::LINALG::Matrix<6, 1>& glstrain, double& psi, const int gp, const int eleGID)
{
  /*
  plain strain energy function

  W    = alpha (Ic*IIIc^(-1/3) -3) + beta (Ic*IIIc^(-1/3)-3)^2

  Ic   .. first invariant of right Cauchy-Green tensor C
  IIIc .. third invariant of right Cauchy-Green tensor C

  The volumetric part is done by a volumetric strain energy function taken from
  Holzapfel

  W_vol = K beta2^(-2) ( beta2 ln (J) + J^(-beta2) -1 )

  where
  K    .. bulk modulus
  beta2 =  -2.0 a parameter according to Doll and Schweizerhof; 9.0 according to Holzapfel,
  alternatively; numerical stability parameter J    .. det(F) determinante of the Jacobian matrix


  Note: Young's modulus is in the input just for convenience. Actually we need the
        parameter alpha (see W above) which is related to E by

      E = 6.0 * alpha.

      Correspondingly the bulk modulus is given by
      K = E / (3-6*nu) = 2*alpha / (1-2*nu)

      with nu = 0.495 we have K = 200 alpha
      with nu = 0.45  we have K =  20 alpha

  */
  // get mat params
  double beta = params_->beta_mean_;
  double youngs = params_->youngs_mean_;

  // init check
  if (isinit_beta_ && !isinit_youngs_)
  {
    beta = beta_;
  }
  else if (!isinit_beta_ && isinit_youngs_)
  {
    youngs = youngs_;
  }
  else if (isinit_beta_ && isinit_youngs_)
  {
    beta = beta_;
    youngs = youngs_;
  }
  else
    dserror(
        "Stochastic Parameters of AAAneohooke_stopro have not been initialized! \n "
        "AAAneohooke_stopro for use with MLMC ONLY!!!");

  const double nue = params_->nue_;                     // Poisson's ratio
  const double alpha = youngs * 0.1666666666666666667;  // E = alpha * 6

  // material parameters for volumetric part
  const double beta2 = -2.0;                                           // parameter from Holzapfel
  double komp = (nue != 0.5) ? 2.0 * alpha / (1.0 - 2.0 * nue) : 0.0;  // bulk modulus

  // compute  invariants
  // build identity tensor I
  CORE::LINALG::Matrix<6, 1> identity(true);
  for (int i = 0; i < 3; i++) identity(i) = 1.0;

  // right Cauchy-Green Tensor  C = 2 * E + I
  CORE::LINALG::Matrix<6, 1> rcg(glstrain);
  rcg.Scale(2.0);
  rcg += identity;

  // invariants
  double inv = rcg(0) + rcg(1) + rcg(2);  // 1st invariant, trace
  double iiinv = rcg(0) * rcg(1) * rcg(2) + 0.25 * rcg(3) * rcg(4) * rcg(5) -
                 0.25 * rcg(1) * rcg(5) * rcg(5) - 0.25 * rcg(2) * rcg(3) * rcg(3) -
                 0.25 * rcg(0) * rcg(4) * rcg(4);  // 3rd invariant, determinante

  double detf = 0.0;
  if (iiinv < 0.0)
    dserror("fatal failure in aneurysmatic artery wall material");
  else
    detf = sqrt(iiinv);  // determinate of deformation gradient
  //--- prepare some constants -----------------------------------------------------------
  const double third = 1.0 / 3.0;
  // now compute
  // ISOCHORIC PART plain strain energy function
  // PSI_iso    = alpha (Ic*IIIc^(-1/3) -3) + beta (Ic*IIIc^(-1/3)-3)^2
  psi = alpha * (inv * pow(iiinv, -third) - 3) +
        beta * (inv * pow(iiinv, -third) - 3) * (inv * pow(iiinv, -third) - 3);
  // add volumetric part
  // W_vol = K beta2^(-2) ( beta2 ln (J) + J^(-beta2) -1 )
  psi = psi + komp * pow(beta2, -2.0) * (beta2 * log(detf) + pow(detf, -beta2) - 1);
  return;
}

void MAT::AAAneohooke_stopro::VisNames(std::map<std::string, int>& names)
{
  std::string fiber = "beta";
  names[fiber] = 1;  // scalar
  fiber = "youngs";
  names[fiber] = 1;  // scalar
  fiber = "BaciEleId";
  names[fiber] = 1;  // scalar
}


bool MAT::AAAneohooke_stopro::VisData(
    const std::string& name, std::vector<double>& data, int numgp, int eleID)
{
  if (name == "beta")
  {
    if ((int)data.size() != 1) dserror("size mismatch");
    data[0] = Beta();
  }
  else if (name == "youngs")
  {
    if ((int)data.size() != 1) dserror("size mismatch");
    data[0] = Youngs();
  }
  else if (name == "BaciEleId")
  {
    if ((int)data.size() != 1) dserror("size mismatch");
    data[0] = eleID;
  }
  else
  {
    return false;
  }
  return true;
}
