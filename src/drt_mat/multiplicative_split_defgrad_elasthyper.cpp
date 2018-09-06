/*----------------------------------------------------------------------*/
/*!
\file multiplicative_split_defgrad_elasthyper.cpp

\brief evaluation of a generic material whose deformation gradient is modeled to be split
multiplicatively into elastic and inelastic parts

\level 3

<pre>
\maintainer Christoph Schmidt
</pre>
*/
/*----------------------------------------------------------------------*/

/* headers */
#include "multiplicative_split_defgrad_elasthyper.H"

#include "material_service.H"
#include "matpar_bundle.H"
#include "inelastic_defgrad_factors.H"
#include "../drt_matelast/elast_summand.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/inpar_ssi.H"


/*--------------------------------------------------------------------*
 | constructor                                          schmidt 03/18 |
 *--------------------------------------------------------------------*/
MAT::PAR::MultiplicativeSplitDefgrad_ElastHyper::MultiplicativeSplitDefgrad_ElastHyper(
    Teuchos::RCP<MAT::PAR::Material> matdata)
    : Parameter(matdata),
      nummat_elast_(matdata->GetInt("NUMMATEL")),
      matids_elast_(matdata->Get<std::vector<int>>("MATIDSEL")),
      numfac_inel_(matdata->GetInt("NUMFACINEL")),
      inel_defgradfacids_(matdata->Get<std::vector<int>>("INELDEFGRADFACIDS")),
      density_(matdata->GetDouble("DENS"))
{
  // check if sizes fit
  if (nummat_elast_ != (int)matids_elast_->size())
    dserror("number of elastic materials %d does not fit to size of elastic material ID vector %d",
        nummat_elast_, matids_elast_->size());

  if (numfac_inel_ != (int)inel_defgradfacids_->size())
    dserror(
        "number of inelastic deformation gradient factors %d does not fit to size of inelastic "
        "deformation gradient ID vector %d",
        numfac_inel_, inel_defgradfacids_->size());
}


Teuchos::RCP<MAT::Material> MAT::PAR::MultiplicativeSplitDefgrad_ElastHyper::CreateMaterial()
{
  return Teuchos::rcp(new MAT::MultiplicativeSplitDefgrad_ElastHyper(this));
}


MAT::MultiplicativeSplitDefgrad_ElastHyperType
    MAT::MultiplicativeSplitDefgrad_ElastHyperType::instance_;


DRT::ParObject* MAT::MultiplicativeSplitDefgrad_ElastHyperType::Create(
    const std::vector<char>& data)
{
  MAT::MultiplicativeSplitDefgrad_ElastHyper* splitdefgrad_elhy =
      new MAT::MultiplicativeSplitDefgrad_ElastHyper();
  splitdefgrad_elhy->Unpack(data);

  return splitdefgrad_elhy;
}


/*--------------------------------------------------------------------*
 | construct empty material                             schmidt 03/18 |
 *--------------------------------------------------------------------*/
MAT::MultiplicativeSplitDefgrad_ElastHyper::MultiplicativeSplitDefgrad_ElastHyper()
    : params_(NULL), potsumel_(0), facdefgradin_(0)
{
}


/*--------------------------------------------------------------------*
 | construct material with specific material params     schmidt 03/18 |
 *--------------------------------------------------------------------*/
MAT::MultiplicativeSplitDefgrad_ElastHyper::MultiplicativeSplitDefgrad_ElastHyper(
    MAT::PAR::MultiplicativeSplitDefgrad_ElastHyper* params)
    : params_(params), potsumel_(0), facdefgradin_(0)
{
  std::vector<int>::const_iterator m;

  // elastic materials
  for (m = params_->matids_elast_->begin(); m != params_->matids_elast_->end(); ++m)
  {
    const int matid = *m;
    Teuchos::RCP<MAT::ELASTIC::Summand> sum = MAT::ELASTIC::Summand::Factory(matid);
    if (sum == Teuchos::null) dserror("Failed to allocate");
    potsumel_.push_back(sum);
  }

  // inelastic deformation gradient factors
  for (m = params_->inel_defgradfacids_->begin(); m != params_->inel_defgradfacids_->end(); ++m)
  {
    const int matid = *m;
    Teuchos::RCP<MAT::InelasticDefgradFactors> fac = MAT::InelasticDefgradFactors::Factory(matid);
    if (fac == Teuchos::null) dserror("Failed to allocate!");
    facdefgradin_.push_back(fac);
  }

  // safety checks
  // get the scatra structure control parameter list
  const Teuchos::ParameterList& ssicontrol = DRT::Problem::Instance()->SSIControlParams();
  // monolithic ssi coupling algorithm only implemented for MAT_InelasticDefgradLinScalarIso and
  // MAT_InelasticDefgradLinScalarAniso so far
  if (DRT::INPUT::IntegralValue<INPAR::SSI::SolutionSchemeOverFields>(ssicontrol, "COUPALGO") ==
      INPAR::SSI::ssi_Monolithic)
  {
    for (unsigned int p = 0; p < facdefgradin_.size(); ++p)
    {
      if ((facdefgradin_[p]->MaterialType() != INPAR::MAT::mfi_lin_scalar_aniso) and
          (facdefgradin_[p]->MaterialType() != INPAR::MAT::mfi_lin_scalar_iso))
        dserror(
            "When you use the 'COUPALGO' 'ssi_Monolithic' from the 'SSI CONTROL' section, you need "
            "to"
            " use the material 'MAT_InelasticDefgradLinScalarAniso' or "
            "'MAT_InelasticDefgradLinScalarIso'!"
            " If you want to use a different material, feel free to implement it! ;-)");
    }
  }
}


/*--------------------------------------------------------------------*
 | pack material for communication purposes             schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);
  // matid
  int matid = -1;
  if (params_ != NULL) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data, matid);

  if (params_ != NULL)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsumel_.size(); ++p) potsumel_[p]->PackSummand(data);
  }
}


/*--------------------------------------------------------------------*
 | unpack material data after communication             schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::Unpack(const std::vector<char>& data)
{
  // make sure we have a pristine material
  params_ = NULL;
  potsumel_.clear();
  facdefgradin_.clear();

  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid and recover params_
  int matid;
  ExtractfromPack(position, data, matid);
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
  {
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const unsigned int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat =
          DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::MultiplicativeSplitDefgrad_ElastHyper*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }
  }

  if (params_ != NULL)  // summands are not accessible in postprocessing mode
  {
    std::vector<int>::const_iterator m;

    // elastic materials
    for (m = params_->matids_elast_->begin(); m != params_->matids_elast_->end(); ++m)
    {
      const int matid = *m;
      Teuchos::RCP<MAT::ELASTIC::Summand> sum = MAT::ELASTIC::Summand::Factory(matid);
      if (sum == Teuchos::null) dserror("Failed to allocate");
      potsumel_.push_back(sum);
    }
    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsumel_.size(); ++p) potsumel_[p]->UnpackSummand(data, position);

    // inelastic deformation gradient factors
    for (m = params_->inel_defgradfacids_->begin(); m != params_->inel_defgradfacids_->end(); ++m)
    {
      const int matid = *m;
      Teuchos::RCP<MAT::InelasticDefgradFactors> fac = MAT::InelasticDefgradFactors::Factory(matid);
      if (fac == Teuchos::null) dserror("Failed to allocate");
      facdefgradin_.push_back(fac);
    }
  }
}


/*--------------------------------------------------------------------*
 | evaluate                                             schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::Evaluate(
    const LINALG::Matrix<3, 3>* defgrad,   ///< Deformation gradient
    const LINALG::Matrix<6, 1>* glstrain,  ///< Green-Lagrange strain
    Teuchos::ParameterList& params,        ///< Container for additional information
    LINALG::Matrix<6, 1>* stress,          ///< 2nd Piola-Kirchhoff stresses
    LINALG::Matrix<6, 6>* cmat,            ///< Constitutive matrix
    const int eleGID)                      ///< Element ID
{
  // do all stuff that only has to be done once per Evaluate() call
  PreEvaluate(params);

  // static variables
  static LINALG::Matrix<6, 6> cmatiso(true);
  static LINALG::Matrix<6, 9> dSdiFin(true);
  static LINALG::Matrix<6, 6> cmatadd(true);

  // build inverse inelastic deformation gradient
  static LINALG::Matrix<3, 3> iFinM(true);
  std::vector<LINALG::Matrix<3, 3>> iFinjM;
  EvaluateInverseInelasticDefGrad(defgrad, iFinjM, iFinM);

  // static variables of kinetic quantities
  static LINALG::Matrix<6, 1> iCV(true);
  static LINALG::Matrix<6, 1> iCinV(true);
  static LINALG::Matrix<6, 1> iCinCiCinV(true);
  static LINALG::Matrix<3, 3> iCinCM(true);
  static LINALG::Matrix<3, 3> iFinCeM(true);
  static LINALG::Matrix<9, 1> CiFin9x1(true);
  static LINALG::Matrix<9, 1> CiFinCe9x1(true);
  static LINALG::Matrix<9, 1> CiFiniCe9x1(true);
  static LINALG::Matrix<3, 1> prinv(true);
  EvaluateKinQuantElast(defgrad, iFinM, iCinV, iCinCiCinV, iCV, iCinCM, iFinCeM, CiFin9x1,
      CiFinCe9x1, CiFiniCe9x1, prinv);

  // derivatives of principle invariants
  static LINALG::Matrix<3, 1> dPIe(true);
  static LINALG::Matrix<6, 1> ddPIIe(true);
  EvaluateInvariantDerivatives(prinv, eleGID, dPIe, ddPIIe);

  // 2nd Piola Kirchhoff stresses factors (according to Holzapfel-Nonlinear Solid Mechanics p. 216)
  static LINALG::Matrix<3, 1> gamma(true);
  // constitutive tensor factors (according to Holzapfel-Nonlinear Solid Mechanics p. 261)
  static LINALG::Matrix<8, 1> delta(true);
  // compose coefficients
  CalculateGammaDelta(prinv, dPIe, ddPIIe, gamma, delta);

  // evaluate dSdiFin
  EvaluatedSdiFin(gamma, delta, iFinM, iCinCM, iCinV, CiFin9x1, CiFinCe9x1, iCinCiCinV, CiFiniCe9x1,
      iCV, iFinCeM, dSdiFin);

  // if cmat != NULL, we are evaluating the structural residual and linearizations, so we need to
  // calculate the stresses and the cmat if you like to evaluate the off-diagonal block of your
  // monolithic system (structural residual w.r.t. dofs of another field), you need to pass NULL as
  // the cmat when you call Evaluate() in the element
  if (cmat != NULL)
  {
    // cmat = 2 dS/dC = 2 \frac{\partial S}{\partial C} + 2 \frac{\partial S}{\partial F_{in}^{-1}}
    // : \frac{\partial F_{in}^{-1}}{\partial C} = cmatiso + cmatadd
    EvaluateStressCmatIso(iCV, iCinV, iCinCiCinV, gamma, delta, *stress, cmatiso);
    cmat->Update(1.0, cmatiso, 0.0);

    // evaluate additional terms for the elasticity tensor
    // cmatadd = 2 \frac{\partial S}{\partial F_{in}^{-1}} : \frac{\partial F_{in}^{-1}}{\partial
    // C}, where F_{in}^{-1} can be multiplicatively composed of several inelastic contributions
    EvaluateAdditionalCmat(defgrad, iFinjM, iCV, dSdiFin, cmatadd);
    cmat->Update(1.0, cmatadd, 1.0);
  }
  // evaluate OD Block
  else
    EvaluateODStiffMat(defgrad, iFinjM, dSdiFin, *stress);

  return;
}


/*--------------------------------------------------------------------*
 | evaluate stress and cmat                             schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::EvaluateStressCmatIso(
    const LINALG::Matrix<6, 1>& iCV,         ///< Inverse right Cauchy-Green tensor
    const LINALG::Matrix<6, 1>& iCinV,       ///< Inverse inelastic right Cauchy-Green tensor
    const LINALG::Matrix<6, 1>& iCinCiCinV,  ///< C_{in}^{-1} * C * C_{in}^{-1}
    const LINALG::Matrix<3, 1>& gamma,       ///< Factors for stress calculation
    const LINALG::Matrix<8, 1>& delta,       ///< Factors for elasticity tensor calculation
    LINALG::Matrix<6, 1>& stress,            ///< Isotropic stress tensor
    LINALG::Matrix<6, 6>& cmatiso) const     ///< Isotropic stiffness matrix
{
  // clear variables
  stress.Clear();
  cmatiso.Clear();

  // 2nd Piola Kirchhoff stresses
  stress.Update(gamma(0), iCinV, 1.0);
  stress.Update(gamma(1), iCinCiCinV, 1.0);
  stress.Update(gamma(2), iCV, 1.0);

  // constitutive tensor
  cmatiso.MultiplyNT(delta(0), iCinV, iCinV, 1.);
  cmatiso.MultiplyNT(delta(1), iCinCiCinV, iCinV, 1.);
  cmatiso.MultiplyNT(delta(1), iCinV, iCinCiCinV, 1.);
  cmatiso.MultiplyNT(delta(2), iCinV, iCV, 1.);
  cmatiso.MultiplyNT(delta(2), iCV, iCinV, 1.);
  cmatiso.MultiplyNT(delta(3), iCinCiCinV, iCinCiCinV, 1.);
  cmatiso.MultiplyNT(delta(4), iCinCiCinV, iCV, 1.);
  cmatiso.MultiplyNT(delta(4), iCV, iCinCiCinV, 1.);
  cmatiso.MultiplyNT(delta(5), iCV, iCV, 1.);
  AddtoCmatHolzapfelProduct(cmatiso, iCV, delta(6));
  AddtoCmatHolzapfelProduct(cmatiso, iCinV, delta(7));

  return;
}


/*--------------------------------------------------------------------*
 | evaluate kinetic quantities                          schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::EvaluateKinQuantElast(
    const LINALG::Matrix<3, 3>* const defgrad,  ///< Deformation gradient
    const LINALG::Matrix<3, 3>& iFinM,          ///< Inverse inelastic deformation gradient
    LINALG::Matrix<6, 1>& iCinV,                ///< Inverse inelastic right Cauchy-Green tensor
    LINALG::Matrix<6, 1>& iCinCiCinV,           ///< C_{in}^{-1} * C * C_{in}^{-1}
    LINALG::Matrix<6, 1>& iCV,                  ///< Inverse right Cauchy-Green tensor
    LINALG::Matrix<3, 3>& iCinCM,               ///< C_{in}^{-1} * C
    LINALG::Matrix<3, 3>& iFinCeM,              ///< F_{in}^{-1} * C_e
    LINALG::Matrix<9, 1>& CiFin9x1,             ///< C * F_{in}^{-1}
    LINALG::Matrix<9, 1>& CiFinCe9x1,           ///< C * F_{in}^{-1} * C_e
    LINALG::Matrix<9, 1>& CiFiniCe9x1,          ///< C * F_{in}^{-1} * C_e^{-1}
    LINALG::Matrix<3, 1>& prinv)
    const  ///< Principal invariants of elastic right Cauchy-Green tensor
{
  // inverse inelastic right Cauchy-Green
  static LINALG::Matrix<3, 3> iCinM(true);
  iCinM.MultiplyNT(1.0, iFinM, iFinM, 0.0);
  MatrixtoStressLikeVoigtNotation(iCinM, iCinV);

  // inverse right Cauchy-Green
  static LINALG::Matrix<3, 3> iCM(true);
  static LINALG::Matrix<3, 3> CM(true);
  CM.MultiplyTN(1.0, *defgrad, *defgrad, 0.0);
  iCM.Invert(CM);
  MatrixtoStressLikeVoigtNotation(iCM, iCV);

  // C_{in}^{-1} * C * C_{in}^{-1}
  static LINALG::Matrix<3, 3> tmp(true);
  static LINALG::Matrix<3, 3> iCinCiCinM;
  tmp.MultiplyNN(1.0, iCinM, CM, 0.0);
  iCinCiCinM.MultiplyNN(1.0, tmp, iCinM, 0.0);
  MatrixtoStressLikeVoigtNotation(iCinCiCinM, iCinCiCinV);

  // elastic right Cauchy-Green in strain-like Voigt notation.
  tmp.MultiplyNN(1.0, *defgrad, iFinM, 0.0);
  static LINALG::Matrix<3, 3> CeM(true);
  CeM.MultiplyTN(1.0, tmp, tmp, 0.0);
  static LINALG::Matrix<6, 1> CeV_strain(true);
  MatrixtoStrainLikeVoigtNotation(CeM, CeV_strain);

  // principal invariants of elastic right Cauchy-Green strain
  InvariantsPrincipal(CeV_strain, prinv);

  // C_{in}^{-1} * C
  iCinCM.MultiplyNN(1.0, iCinM, CM, 0.0);

  // F_{in}^{-1} * C_e
  iFinCeM.MultiplyNN(1.0, iFinM, CeM, 0.0);

  // C * F_{in}^{-1}
  static LINALG::Matrix<3, 3> CiFinM(true);
  CiFinM.MultiplyNN(1.0, CM, iFinM, 0.0);
  Matrix3x3to9x1(CiFinM, CiFin9x1);

  // C * F_{in}^{-1} * C_e
  static LINALG::Matrix<3, 3> CiFinCeM(true);
  tmp.MultiplyNN(1.0, CM, iFinM, 0.0);
  CiFinCeM.MultiplyNN(1.0, tmp, CeM, 0.0);
  Matrix3x3to9x1(CiFinCeM, CiFinCe9x1);

  // C * F_{in}^{-1} * C_e^{-1}
  static LINALG::Matrix<3, 3> CiFiniCeM(true);
  static LINALG::Matrix<3, 3> iCeM(true);
  iCeM.Invert(CeM);
  tmp.MultiplyNN(1.0, CM, iFinM, 0.0);
  CiFiniCeM.MultiplyNN(1.0, tmp, iCeM, 0.0);
  Matrix3x3to9x1(CiFiniCeM, CiFiniCe9x1);

  return;
}


/*--------------------------------------------------------------------*
 | evaluate principle invariants                        schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::InvariantsPrincipal(
    const LINALG::Matrix<6, 1>&
        C_strain,  ///< symmetric Cartesian 2-tensor in strain-like 6-Voigt notation
    LINALG::Matrix<3, 1>& prinv) const  ///< principal invariants
{
  // 1st invariant, trace
  prinv(0) = C_strain(0) + C_strain(1) + C_strain(2);
  // 2nd invariant
  prinv(1) = 0.5 * (prinv(0) * prinv(0) - C_strain(0) * C_strain(0) - C_strain(1) * C_strain(1) -
                       C_strain(2) * C_strain(2) - .5 * C_strain(3) * C_strain(3) -
                       .5 * C_strain(4) * C_strain(4) - .5 * C_strain(5) * C_strain(5));
  // 3rd invariant, determinant
  prinv(2) = C_strain(0) * C_strain(1) * C_strain(2) +
             0.25 * C_strain(3) * C_strain(4) * C_strain(5) -
             0.25 * C_strain(1) * C_strain(5) * C_strain(5) -
             0.25 * C_strain(2) * C_strain(3) * C_strain(3) -
             0.25 * C_strain(0) * C_strain(4) * C_strain(4);

  return;
}


/*--------------------------------------------------------------------*
 | evaluate derivatives of principle invariants         schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::EvaluateInvariantDerivatives(
    const LINALG::Matrix<3, 1>&
        prinv,                  ///< Principal invariants of the elastic right Cauchy-Green tensor
    const int eleGID,           ///< Element ID
    LINALG::Matrix<3, 1>& dPI,  ///< First derivative with respect to invariants
    LINALG::Matrix<6, 1>& ddPII) const  ///< Second derivative with respect to invariants
{
  // clear variables
  dPI.Clear();
  ddPII.Clear();

  // loop over map of associated potential summands
  // derivatives of strain energy function w.r.t. principal invariants
  for (unsigned p = 0; p < potsumel_.size(); ++p)
  {
    potsumel_[p]->AddDerivativesPrincipal(dPI, ddPII, prinv, eleGID);
  }

  return;
}


/*--------------------------------------------------------------------*
 | calculate factors for stress and cmat calculation    schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::CalculateGammaDelta(
    const LINALG::Matrix<3, 1>&
        prinv,  ///< principal invariants of the elastic right Cauchy-Green tensor
    const LINALG::Matrix<3, 1>& dPI,    ///< first derivative with respect to invariants
    const LINALG::Matrix<6, 1>& ddPII,  ///< second derivative with respect to invariants
    LINALG::Matrix<3, 1>& gamma,        ///< factors for stress calculation
    LINALG::Matrix<8, 1>& delta) const  ///< factors for elasticity tensor calculation
{
  // according to Holzapfel-Nonlinear Solid Mechanics p. 216 and p. 248
  gamma(0) = 2. * (dPI(0) + prinv(0) * dPI(1));
  gamma(1) = -2. * dPI(1);
  gamma(2) = 2. * prinv(2) * dPI(2);

  // according to Holzapfel-Nonlinear Solid Mechanics p. 261
  delta(0) = 4. * (ddPII(0) + 2.0 * prinv(0) * ddPII(5) + dPI(1) + prinv(0) * prinv(0) * ddPII(1));
  delta(1) = -4. * (ddPII(5) + prinv(0) * ddPII(1));
  delta(2) = 4. * (prinv(2) * ddPII(4) + prinv(0) * prinv(2) * ddPII(3));
  delta(3) = 4. * ddPII(1);
  delta(4) = -4. * prinv(2) * ddPII(3);
  delta(5) = 4. * (prinv(2) * dPI(2) + prinv(2) * prinv(2) * ddPII(2));
  delta(6) = -4. * prinv(2) * dPI(2);
  delta(7) = -4. * dPI(1);

  return;
}


/*--------------------------------------------------------------------*
 | evaluate derivative of stress w.r.t. inelastic                     |
 | deformation gradient                                 schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::EvaluatedSdiFin(
    const LINALG::Matrix<3, 1>& gamma,        ///< Factors for stress calculation
    const LINALG::Matrix<8, 1>& delta,        ///< Factors for elasticity tensor calculation
    const LINALG::Matrix<3, 3>& iFinM,        ///< Inverse inelastic deformation gradient
    const LINALG::Matrix<3, 3>& iCinCM,       ///< C_{in}^{-1} * C
    const LINALG::Matrix<6, 1>& iCinV,        ///< Inverse inelastic right Cauchy-Green tensor
    const LINALG::Matrix<9, 1>& CiFin9x1,     ///< C * F_{in}^{-1}
    const LINALG::Matrix<9, 1>& CiFinCe9x1,   ///< C * F_{in}^{-1} * C_e
    const LINALG::Matrix<6, 1>& iCinCiCinV,   ///< C_{in}^{-1} * C * C_{in}^{-1}
    const LINALG::Matrix<9, 1>& CiFiniCe9x1,  ///< C * F_{in}^{-1} * C_e^{-1}
    const LINALG::Matrix<6, 1>& iCV,          ///< Inverse right Cauchy-Green tensor
    const LINALG::Matrix<3, 3>& iFinCeM,      ///< F_{in}^{-1} * C_e
    LINALG::Matrix<6, 9>& dSdiFin) const      ///< derivative of 2nd Piola Kirchhoff stresses w.r.t.
                                              ///< inverse inelastic right Cauchy-Green tensor
{
  // clear variable
  dSdiFin.Clear();

  // calculate identity tensor
  static LINALG::Matrix<3, 3> id(true);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

  // derivative of second Piola Kirchhoff stresses w.r.t. inverse growth deformation gradient
  MAT::AddRightNonSymmetricHolzapfelProduct(dSdiFin, id, iFinM, gamma(0));
  MAT::AddRightNonSymmetricHolzapfelProduct(dSdiFin, iCinCM, iFinM, gamma(1));
  dSdiFin.MultiplyNT(delta(0), iCinV, CiFin9x1, 1.);
  dSdiFin.MultiplyNT(delta(1), iCinV, CiFinCe9x1, 1.);
  dSdiFin.MultiplyNT(delta(1), iCinCiCinV, CiFin9x1, 1.);
  dSdiFin.MultiplyNT(delta(2), iCinV, CiFiniCe9x1, 1.);
  dSdiFin.MultiplyNT(delta(2), iCV, CiFin9x1, 1.);
  dSdiFin.MultiplyNT(delta(3), iCinCiCinV, CiFinCe9x1, 1.);
  dSdiFin.MultiplyNT(delta(4), iCinCiCinV, CiFiniCe9x1, 1.);
  dSdiFin.MultiplyNT(delta(4), iCV, CiFinCe9x1, 1.);
  dSdiFin.MultiplyNT(delta(5), iCV, CiFiniCe9x1, 1.);
  MAT::AddRightNonSymmetricHolzapfelProduct(dSdiFin, id, iFinCeM, gamma(1));

  return;
}


/*--------------------------------------------------------------------*
 | evaluate additional contribution to cmat             schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::EvaluateAdditionalCmat(
    const LINALG::Matrix<3, 3>* const defgrad,  ///< Deformation gradient
    const std::vector<LINALG::Matrix<3, 3>>&
        iFinjM,  ///< Vector that holds all inverse inelastic deformation gradient factors as 3x3
                 ///< matrices
    const LINALG::Matrix<6, 1>& iCV,      ///< Inverse right Cauchy-Green tensor
    const LINALG::Matrix<6, 9>& dSdiFin,  ///< Derivative of 2nd Piola Kirchhoff stresses w.r.t. the
                                          ///< inverse inelastic deformation gradient
    LINALG::Matrix<6, 6>& cmatadd) const  ///< Additional elasticity tensor
{
  // clear variable
  cmatadd.Clear();

  // check amount of factors the inelastic deformation gradient consists of and choose
  // implementation accordingly
  switch (facdefgradin_.size())
  {
    case 1:
    {
      facdefgradin_[0]->EvaluateAdditionalCmat(defgrad, iFinjM[0], iCV, dSdiFin, cmatadd);

      break;
    }
    case 2:
    {
      // static variables
      // dSdiFinj = dSdiFin : diFindiFinj
      static LINALG::Matrix<6, 9> dSdiFinj(true);
      static LINALG::Matrix<9, 9> diFindiFinj(true);
      static LINALG::Matrix<3, 3> id(true);
      for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

      // evaluation of first inelastic deformation gradient factor contribution
      // clear static variable
      diFindiFinj.Clear();
      AddNonSymmetricProduct(1.0, iFinjM[1], id, diFindiFinj);
      dSdiFinj.Multiply(1.0, dSdiFin, diFindiFinj, 0.0);
      facdefgradin_[0]->EvaluateAdditionalCmat(defgrad, iFinjM[0], iCV, dSdiFinj, cmatadd);

      // evaluation of second inelastic deformation gradient factor contribution
      // clear static variable
      diFindiFinj.Clear();
      AddNonSymmetricProduct(1.0, id, iFinjM[0], diFindiFinj);
      dSdiFinj.Multiply(1.0, dSdiFin, diFindiFinj, 0.0);
      facdefgradin_[1]->EvaluateAdditionalCmat(defgrad, iFinjM[1], iCV, dSdiFinj, cmatadd);

      break;
    }
    default:
      dserror(
          "You defined %i inelastic deformation gradient factors. But framework is only "
          "implemented for a maximum of 2 inelastic contributions! "
          "If you really need more than 2 inelastic contributions, it's your turn to implement it! "
          ";-)",
          facdefgradin_.size());
      break;
  }

  return;
}


/*--------------------------------------------------------------------*
 | setup                                                schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::Setup(
    const int numgp, DRT::INPUT::LineDefinition* linedef)
{
  // elastic materials
  for (unsigned int p = 0; p < potsumel_.size(); ++p) potsumel_[p]->Setup(linedef);

  return;
}


/*--------------------------------------------------------------------*
 | update                                               schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::Update()
{
  // loop map of associated potential summands
  for (unsigned int p = 0; p < potsumel_.size(); ++p) potsumel_[p]->Update();

  return;
}


/*--------------------------------------------------------------------*
 | evaluate the inverse of the inelastic deformation                  |
 | gradient                                             schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::EvaluateInverseInelasticDefGrad(
    const LINALG::Matrix<3, 3>* const defgrad,  ///< Deformation gradient
    std::vector<LINALG::Matrix<3, 3>>& iFinjM,  ///< Vector that holds all inverse inelastic
                                                ///< deformation gradient factors as 3x3 matrices
    LINALG::Matrix<3, 3>& iFinM) const          ///< Inverse inelastic deformation gradient
{
  // temporary variables
  static LINALG::Matrix<3, 3> iFinp(true);
  static LINALG::Matrix<3, 3> iFin_init_store(true);

  // clear variables
  iFinM.Clear();
  iFin_init_store.Clear();

  // initialize them
  for (int i = 0; i < 3; ++i) iFin_init_store(i, i) = 1.0;

  // loop over all inelastic contributions
  for (unsigned int p = 0; p < facdefgradin_.size(); ++p)
  {
    // clear tmp variable
    iFinp.Clear();

    // calculate inelastic deformation gradient and its inverse
    facdefgradin_[p]->EvaluateInverseInelasticDefGrad(defgrad, iFinp);

    // store inelastic deformation gradient of p-th inelastic contribution
    iFinjM.push_back(iFinp);

    // update inverse inelastic deformation gradient
    iFinM.Multiply(iFin_init_store, iFinp);

    // store result for next evaluation
    iFin_init_store.Update(1.0, iFinM, 0.0);
  }

  return;
}


/*--------------------------------------------------------------------*
 | evaluate the off-diagonal contribution to the                      |
 | stiffness matrix (for monolithic calculation)        schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::EvaluateODStiffMat(
    const LINALG::Matrix<3, 3>* const defgrad,  ///< Deformation gradient
    const std::vector<LINALG::Matrix<3, 3>>&
        iFinMj,  ///< Vector that holds all inverse inelastic deformation gradient factors as 3x3
                 ///< matrices
    const LINALG::Matrix<6, 9>& dSdiFin,  ///< Derivative of 2nd Piola Kirchhoff stresses w.r.t. the
                                          ///< inverse inelastic deformation gradient
    LINALG::Matrix<6, 1>& dstressdx)  ///< Derivative of 2nd Piola Kirchhoff stresses w.r.t. primary
                                      ///< variable of different field
{
  // clear variable
  dstressdx.Clear();

  // check amount of factors the inelastic deformation gradient consists of and choose
  // implementation accordingly
  switch (facdefgradin_.size())
  {
    case 1:
    {
      facdefgradin_[0]->EvaluateODStiffMat(defgrad, iFinMj[0], dSdiFin, dstressdx);

      break;
    }
    case 2:
    {
      // static variables
      // dSdiFinj = dSdiFin : diFindiFinj
      static LINALG::Matrix<6, 9> dSdiFinj(true);
      static LINALG::Matrix<9, 9> diFindiFinj(true);
      static LINALG::Matrix<3, 3> id(true);
      for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

      // evaluation of first inelastic deformation gradient factor contribution
      // clear static variable
      diFindiFinj.Clear();
      AddNonSymmetricProduct(1.0, iFinMj[1], id, diFindiFinj);
      dSdiFinj.Multiply(1.0, dSdiFin, diFindiFinj, 0.0);
      facdefgradin_[0]->EvaluateODStiffMat(defgrad, iFinMj[0], dSdiFinj, dstressdx);

      // evaluation of second inelastic deformation gradient factor contribution
      // clear static variable
      diFindiFinj.Clear();
      AddNonSymmetricProduct(1.0, id, iFinMj[0], diFindiFinj);
      dSdiFinj.Multiply(1.0, dSdiFin, diFindiFinj, 0.0);
      facdefgradin_[1]->EvaluateODStiffMat(defgrad, iFinMj[1], dSdiFinj, dstressdx);

      break;
    }
    default:
      dserror(
          "You defined %i inelastic deformation gradient factors. But framework is only "
          "implemented for a maximum of 2 inelastic contributions! "
          "If you really need more than 2 inelastic contributions, it's your turn to implement it! "
          ";-)",
          facdefgradin_.size());
      break;
  }

  return;
}


/*--------------------------------------------------------------------*
 | pre evaluate                                         schmidt 03/18 |
 *--------------------------------------------------------------------*/
void MAT::MultiplicativeSplitDefgrad_ElastHyper::PreEvaluate(
    Teuchos::ParameterList& params) const  ///< parameter list as handed in from the element
{
  // loop over all inelastic contributions
  for (unsigned int p = 0; p < facdefgradin_.size(); ++p) facdefgradin_[p]->PreEvaluate(params);

  return;
}