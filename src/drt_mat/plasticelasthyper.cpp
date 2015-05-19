/*----------------------------------------------------------------------*/
/*!
\file plasticelasthyper.cpp
\brief
This file contains the hyperelastic toolbox with application to finite
strain plasticity using a semi-smooth Newton method. It allows summing up
several summands of isotropic non-splitted type to build
a hyperelastic strain energy function.

The input line should read
MAT 1 MAT_PlasticElastHyper NUMMAT 1 MATIDS 2 DENS 1.0 INITYIELD 0.45 ISOHARD 0.12924 EXPISOHARD 16.93 INFYIELD 0.715 KINHARD 0.0
                            CTE 1.0e-5 INITTEMP 293 YIELDSOFT 0.002 HARDSOFT 0.002 VISC 1e-4 VISC_TEMP 0.003
                            PL_SPIN_CHI -50 rY_11 1.0 rY_22 0.9 rY_33 0.9 rY_12 0.7 rY_23 0.57385 rY_13 0.7

<pre>
Maintainer: Alexander Seitz
            seitz@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15271
</pre>
*/

/*----------------------------------------------------------------------*/

#include "plasticelasthyper.H"
#include "../drt_lib/standardtypes_cpp.H"
#include "../drt_matelast/elast_summand.H"
#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_linedefinition.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_mat/material_service.H"
#include "Epetra_SerialDenseSolver.h"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::PlasticElastHyper::PlasticElastHyper(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: MAT::PAR::ElastHyper(matdata),
  inityield_(matdata->GetDouble("INITYIELD")),
  isohard_(matdata->GetDouble("ISOHARD")),
  expisohard_(matdata->GetDouble("EXPISOHARD")),
  infyield_(matdata->GetDouble("INFYIELD")),
  kinhard_(matdata->GetDouble("KINHARD")),
  visc_(matdata->GetDouble("VISC")),
  rate_dependency_(matdata->GetDouble("RATE_DEPENDENCY")),
  visc_soft_(matdata->GetDouble("VISC_SOFT")),
  cte_(matdata->GetDouble("CTE")),
  inittemp_(matdata->GetDouble("INITTEMP")),
  yieldsoft_(matdata->GetDouble("YIELDSOFT")),
  hardsoft_(matdata->GetDouble("HARDSOFT")),
  plspin_chi_(-1.*matdata->GetDouble("PL_SPIN_CHI")),
  rY_11_(matdata->GetDouble("rY_11")),
  rY_22_(matdata->GetDouble("rY_22")),
  rY_33_(matdata->GetDouble("rY_33")),
  rY_12_(matdata->GetDouble("rY_12")),
  rY_23_(matdata->GetDouble("rY_23")),
  rY_13_(matdata->GetDouble("rY_13")),
  cpl_(0.),
  stab_s_(0.),
  dis_mode_(INPAR::TSI::pl_multiplier)
{
  // check if sizes fit
  if (nummat_ != (int)matids_->size())
    dserror("number of materials %d does not fit to size of material vector %d", nummat_, matids_->size());

  // check plastic parameter validity
  if (inityield_<=0.)
    dserror("initial yield stress must be positive");
  // no infyield provided 0. is default
  if (infyield_==0.)
    if (expisohard_!=0.)
      dserror("hardening exponent provided without inf yield stress");
  if (expisohard_<0.)
    dserror("Nonlinear hardening exponent must be non-negative");

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<MAT::Material> MAT::PAR::PlasticElastHyper::CreateMaterial()
{
  return Teuchos::rcp(new MAT::PlasticElastHyper(this));
}


MAT::PlasticElastHyperType MAT::PlasticElastHyperType::instance_;


DRT::ParObject* MAT::PlasticElastHyperType::Create( const std::vector<char> & data )
{
  MAT::PlasticElastHyper* elhy = new MAT::PlasticElastHyper();
  elhy->Unpack(data);

  return elhy;
}


/*----------------------------------------------------------------------*
 |  initialise static arrays                                 seitz 05/14|
 *----------------------------------------------------------------------*/
const int MAT::PlasticElastHyper::VOIGT3X3_[3][3]       = {{0,3,5},{3,1,4},{5,4,2}};
const int MAT::PlasticElastHyper::VOIGT3X3NONSYM_[3][3] = {{0,3,5},{6,1,4},{8,7,2}};


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PlasticElastHyper::PlasticElastHyper()
  : params_(NULL),
    last_plastic_defgrd_inverse_(Teuchos::null),
    last_alpha_isotropic_(Teuchos::null),
    last_alpha_kinematic_(Teuchos::null),
    activity_state_(Teuchos::null)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PlasticElastHyper::PlasticElastHyper(MAT::PAR::PlasticElastHyper* params)
  : params_(params),
    HepDiss_(Teuchos::null),
    dHepDissdd_(Teuchos::null),
    dHepDissdT_(Teuchos::null),
    dHepDissdTeas_(Teuchos::null)
{
  // make sure the referenced materials in material list have quick access parameters
  std::vector<int>::const_iterator m;
  for (m=params_->matids_->begin(); m!=params_->matids_->end(); ++m)
  {
    const int matid = *m;
    Teuchos::RCP<MAT::ELASTIC::Summand> sum = MAT::ELASTIC::Summand::Factory(matid);
    if (sum == Teuchos::null) dserror("Failed to allocate");
    potsum_.push_back(sum);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::Pack(DRT::PackBuffer& data) const
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
  AddtoPack(data,(int)isoprinc_);
  AddtoPack(data,(int)isomod_);
  AddtoPack(data,(int)anisoprinc_);
  AddtoPack(data,(int)anisomod_);
  AddtoPack(data,(int)isovisco_);
  AddtoPack(data,(int)viscogenmax_);

  // plastic anisotropy
  AddtoPack(data,PlAniso_full_);
  AddtoPack(data,InvPlAniso_full_);

  if (params_ != NULL) // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (unsigned int p=0; p<potsum_.size(); ++p)
    {
     potsum_[p]->PackSummand(data);
    }
  }

  // plastic history data
  AddtoPack<3,3>(data,last_plastic_defgrd_inverse_);
  AddtoPack(data,last_alpha_isotropic_);
  AddtoPack<3,3>(data,last_alpha_kinematic_);

  // tsi data
  bool tsi = HepDiss_!=Teuchos::null;
  AddtoPack(data,(int)tsi);
  bool tsi_eas = dHepDissdTeas_!=Teuchos::null;
  AddtoPack(data,(int)tsi_eas);
  if (tsi)
    AddtoPack(data,(int)dHepDissdd_->at(0).M());

  // dissipation mode
  AddtoPack(data,(int)DisMode());

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::Unpack(const std::vector<char>& data)
{
  // make sure we have a pristine material
  params_ = NULL;
  potsum_.clear();

  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid and recover params_
  int matid;
  ExtractfromPack(position,data,matid);
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
  {
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const unsigned int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat = DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::PlasticElastHyper*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(), MaterialType());
    }
  }

  isoprinc_=(bool)ExtractInt(position,data);
  isomod_=(bool)ExtractInt(position,data);
  anisoprinc_=(bool)ExtractInt(position,data);
  anisomod_=(bool)ExtractInt(position,data);
  isovisco_=(bool)ExtractInt(position,data);
  viscogenmax_=(bool)ExtractInt(position,data);

  // plastic anisotropy
  ExtractfromPack(position,data,PlAniso_full_);
  ExtractfromPack(position,data,InvPlAniso_full_);

  if (params_ != NULL) // summands are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m=params_->matids_->begin(); m!=params_->matids_->end(); ++m)
    {
      const int matid = *m;
      Teuchos::RCP<MAT::ELASTIC::Summand> sum = MAT::ELASTIC::Summand::Factory(matid);
      if (sum == Teuchos::null) dserror("Failed to allocate");
      potsum_.push_back(sum);
    }

    // loop map of associated potential summands
    for (unsigned int p=0; p<potsum_.size(); ++p)
    {
     potsum_[p]->UnpackSummand(data,position);
    }
  }

  // plastic history data
  ExtractfromPack<3,3>(position,data,last_plastic_defgrd_inverse_);
  ExtractfromPack(position,data,last_alpha_isotropic_);
  ExtractfromPack<3,3>(position,data,last_alpha_kinematic_);

  // no need to pack this
  delta_alpha_i_ .resize(last_alpha_isotropic_.size(),0.);
  activity_state_.resize(last_alpha_isotropic_.size(),false);

  bool tsi = (bool)ExtractInt(position,data);
  bool tsi_eas = (bool)ExtractInt(position,data);
  if (!tsi)
  {
    HepDiss_=Teuchos::null;
    dHepDissdd_=Teuchos::null;
    dHepDissdT_=Teuchos::null;
    dHepDissdTeas_=Teuchos::null;
  }
  else
  {
    int ngp=last_alpha_isotropic_.size();
    HepDiss_=Teuchos::rcp(new std::vector<double>(ngp,0.0));
    int numdofperelement = ExtractInt(position,data);
    dHepDissdd_=Teuchos::rcp(new std::vector<Epetra_SerialDenseVector>
      (ngp,Epetra_SerialDenseVector(numdofperelement)));
    dHepDissdT_=Teuchos::rcp(new std::vector<double>(ngp,0.0));
    if (tsi_eas)
      dHepDissdTeas_=Teuchos::rcp(new std::vector<Epetra_SerialDenseVector>
        (ngp,Epetra_SerialDenseVector(numdofperelement/3)));
  }

  // dissipation mode
  INPAR::TSI::DissipationMode mode = (INPAR::TSI::DissipationMode)ExtractInt(position,data);
  SetDissipationMode(mode);

  // in the postprocessing mode, we do not unpack everything we have packed
  // -> position check cannot be done in this case
  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",data.size(),position);

    return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::Setup(int numgp, DRT::INPUT::LineDefinition* linedef)
{
  // Setup summands
  for (unsigned int p=0; p<potsum_.size(); ++p)
  {
    potsum_[p]->Setup(linedef);
  }

  // find out which formulations are used

  isoprinc_ = false ;
  isomod_ = false ;
  anisoprinc_ = false ;
  anisomod_ = false;
  isovisco_ = false;
  viscogenmax_ = false;

  for (unsigned int p=0; p<potsum_.size(); ++p)
  {
    potsum_[p]->SpecifyFormulation(isoprinc_,isomod_,anisoprinc_,anisomod_,isovisco_,viscogenmax_);
  }
  // in this case the mandel stress become non-symmetric and the
  // calculated derivatives have to be extended.
  if (anisomod_==true || anisoprinc_==true)
    dserror("PlasticElastHyper only for isotropic elastic material!");

  // check if either zero or three fiber directions are given
  if (linedef->HaveNamed("FIBER1") || linedef->HaveNamed("FIBER2") || linedef->HaveNamed("FIBER3"))
    if (!linedef->HaveNamed("FIBER1") || !linedef->HaveNamed("FIBER2") || !linedef->HaveNamed("FIBER3"))
      dserror("so3 expects no fibers or 3 fiber directions");

    // plastic anisotropy
    SetupHillPlasticity(linedef);

    // setup plastic history variables
    LINALG::Matrix<3,3> tmp(true);
    last_alpha_isotropic_        .resize(numgp,0.);
    last_alpha_kinematic_        .resize(numgp,tmp);
    for (int i=0; i<3; i++) tmp(i,i)=1.;
    last_plastic_defgrd_inverse_ .resize(numgp,tmp);
    activity_state_              .resize(numgp,false);
    delta_alpha_i_               .resize(numgp,0.);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::SetupTSI(const int numgp,
                                      const int numdofperelement,
                                      const bool eas,
                                      const INPAR::TSI::DissipationMode mode)
{
  // dissipation mode
  if (mode!=INPAR::TSI::pl_flow)
    if (params_->rY_11_!=0. || params_->rY_22_!=0. || params_->rY_33_!=0. ||
        params_->rY_12_!=0. || params_->rY_23_!=0. || params_->rY_13_!=0.)
      dserror("TSI with Hill plasticity only with DISSIPATION_MODE pl_flow");
  SetDissipationMode(mode);

  // allocate memory
  HepDiss_=Teuchos::rcp(new std::vector<double>(numgp,0.0));
  dHepDissdd_=Teuchos::rcp(new std::vector<Epetra_SerialDenseVector>
                               (numgp,Epetra_SerialDenseVector(numdofperelement)));
  dHepDissdT_=Teuchos::rcp(new std::vector<double>(numgp,0.0));
  if (eas) dHepDissdTeas_=Teuchos::rcp(new std::vector<Epetra_SerialDenseVector>
                               (numgp,Epetra_SerialDenseVector(numdofperelement/3)));

  // no TSI with kinematic hardening yet
  // be aware that in that case, another dependency of the NCP function on the
  // temperature arises, namely via H^k=H^k(T) in the computation of the effective
  // stress eta. Without this term, the only dependency of the NCP function is
  // via the effective yield stress Y^pl=Y^pl(T)
  if (Kinhard()!=0.)
    dserror("no kinematic hardening for TSI (yet)");
  // no TSI with plastic spin yet
  // besides the kinematic hardening term (see above) there is not much to do
  // just add the derivatives of theating and NCP function in the
  // EvaluateNCPandSpin(...) function
  if (PlSpinChi()!=0.)
    dserror("no thermo-plasticitiy with plastic spin");

  /// Hill TSI only with pl_flow dissipation
  if (params_->rY_11_!=0. && DisMode()!=INPAR::TSI::pl_flow)
    dserror("hill thermo plasticity only with dissipation mode pl_flow");

  /// viscoplastic TSI only with  pl_flow dissipation
  if (Visc()!=0. &&  DisMode()!=INPAR::TSI::pl_flow)
    dserror("thermo-visco-plasticity only with dissipation mode pl_flow");
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::SetupHillPlasticity(DRT::INPUT::LineDefinition* linedef)
{
  // check if parameters are valid
  if (params_->rY_11_!=0. || params_->rY_22_!=0. || params_->rY_33_!=0. ||
      params_->rY_12_!=0. || params_->rY_23_!=0. || params_->rY_13_!=0.)
    if (params_->rY_11_<=0. || params_->rY_22_<=0. || params_->rY_33_<=0. ||
        params_->rY_12_<=0. || params_->rY_23_<=0. || params_->rY_13_<=0.)
      dserror("Hill parameters all must be positive (incomplete set?)");

  // all (optional) Hill parameters are zero (default value)
  // --> we want to do von Mises plasticity
  if (params_->rY_11_==0. && params_->rY_22_==0. && params_->rY_33_==0. &&
      params_->rY_12_==0. && params_->rY_23_==0. && params_->rY_13_==0.)
  {
    PlAniso_full_.Clear();
    for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
        if (i==j) PlAniso_full_(i,j)=2./3.;
        else      PlAniso_full_(i,j)=-1./3.;
    for (int i=3; i<6; i++)
      PlAniso_full_(i,i)=1.;
    InvPlAniso_full_.Update(PlAniso_full_);
  }
  // we do Hill plasticity
  else
  {
    // anisotropy directions
    std::vector<LINALG::Matrix<3,1> > directions(3);

    // first anisotropy direction
    if (linedef->HaveNamed("FIBER1"))
    {
      std::vector<double> fiber;
       linedef->ExtractDoubleVector("FIBER1",fiber);
       double fnorm=0.;
       //normalization
       for (int i = 0; i < 3; ++i)
         fnorm += fiber[i]*fiber[i];
       fnorm = sqrt(fnorm);
       if (fnorm==0.)
         dserror("Fiber vector has norm zero");

       // fill final fiber vector
       for (int i = 0; i < 3; ++i)
         directions.at(0)(i) = fiber[i]/fnorm;
    }
    // second anisotropy direction
    if (linedef->HaveNamed("FIBER2"))
    {
      std::vector<double> fiber;
       linedef->ExtractDoubleVector("FIBER2",fiber);
       double fnorm=0.;
       //normalization
       for (int i = 0; i < 3; ++i)
         fnorm += fiber[i]*fiber[i];
       fnorm = sqrt(fnorm);
       if (fnorm==0.)
         dserror("Fiber vector has norm zero");

       // fill final fiber vector
       for (int i = 0; i < 3; ++i)
         directions.at(1)(i) = fiber[i]/fnorm;
    }
    // third anisotropy direction
    if (linedef->HaveNamed("FIBER3"))
    {
      std::vector<double> fiber;
       linedef->ExtractDoubleVector("FIBER3",fiber);
       double fnorm=0.;
       //normalization
       for (int i = 0; i < 3; ++i)
         fnorm += fiber[i]*fiber[i];
       fnorm = sqrt(fnorm);
       if (fnorm==0.)
         dserror("Fiber vector has norm zero");

       // fill final fiber vector
       for (int i = 0; i < 3; ++i)
         directions.at(2)(i) = fiber[i]/fnorm;
    }

    // check orthogonality
    LINALG::Matrix<1,1> matrix1;
    matrix1.MultiplyTN(directions.at(0),directions.at(1));
    if (std::abs(matrix1(0,0))>1.e-16)
      dserror("fiber directions not orthogonal");
    matrix1.MultiplyTN(directions.at(0),directions.at(2));
    if (std::abs(matrix1(0,0))>1.e-16)
      dserror("fiber directions not orthogonal");
    matrix1.MultiplyTN(directions.at(2),directions.at(1));
    if (std::abs(matrix1(0,0))>1.e-16)
      dserror("fiber directions not orthogonal");

    // check right-handed trihedron
    LINALG::Matrix<3,1> A0xA1;
    A0xA1(0) = (directions.at(0)(1)*directions.at(1)(2)-directions.at(0)(2)*directions.at(1)(1));
    A0xA1(1) = (directions.at(0)(2)*directions.at(1)(0)-directions.at(0)(0)*directions.at(1)(2));
    A0xA1(2) = (directions.at(0)(0)*directions.at(1)(1)-directions.at(0)(1)*directions.at(1)(0));
    A0xA1.Update(-1.,directions.at(2),1.);
    if (A0xA1.Norm2()>1.e-8)

      dserror("fibers don't form right-handed trihedron");

    // setup structural tensor for first and second direction
    // (as the directions are orthogonal, 2 structural tensors are sufficient)
    LINALG::Matrix<3,3> M0;
    M0.MultiplyNT(directions.at(0),directions.at(0));
    LINALG::Matrix<3,3> M1;
    M1.MultiplyNT(directions.at(1),directions.at(1));
    LINALG::Matrix<3,3> M2;
    M2.MultiplyNT(directions.at(2),directions.at(2));

    double alpha1 = 2./3./params_->rY_11_/params_->rY_11_;
    double alpha2 = 2./3./params_->rY_22_/params_->rY_22_;
    double alpha3 = 2./3./params_->rY_33_/params_->rY_33_;
    double alpha4 = 1./3./params_->rY_12_/params_->rY_12_;
    double alpha5 = 1./3./params_->rY_23_/params_->rY_23_;
    double alpha6 = 1./3./params_->rY_13_/params_->rY_13_;

    // calculate plastic anisotropy tensor
    PlAniso_full_.Clear();
    ElastSymTensorMultiply(PlAniso_full_,alpha1,M0,M0,1.);
    ElastSymTensorMultiply(PlAniso_full_,alpha2,M1,M1,1.);
    ElastSymTensorMultiply(PlAniso_full_,alpha3,M2,M2,1.);
    ElastSymTensorMultiplyAddSym(PlAniso_full_,0.5*(alpha3-alpha1-alpha2),M0,M1,1.);
    ElastSymTensorMultiplyAddSym(PlAniso_full_,0.5*(alpha1-alpha2-alpha3),M1,M2,1.);
    ElastSymTensorMultiplyAddSym(PlAniso_full_,0.5*(alpha2-alpha3-alpha1),M0,M2,1.);
    AddtodMdC_gamma2(PlAniso_full_,M0,M1,alpha4);
    AddtodMdC_gamma2(PlAniso_full_,M1,M2,alpha5);
    AddtodMdC_gamma2(PlAniso_full_,M0,M2,alpha6);

    // we need this matrix to get rid of the zero eigenvalue to be able to invert
    // the anisotropy tensor. After the inversion we expand the tensor again to 6x6
    // so that we have the correct pseudo-inverse.
    LINALG::Matrix<6,5> red(true);
    red(0,0) = 1.;
    red(1,1) = 1.;
    red(2,0) = -1.;
    red(2,1) = -1.;
    red(3,2) = 1.;
    red(4,3) = 1.;
    red(5,4) = 1.;

    // Invert plastic anisotropy tensor
    LINALG::Matrix<6,5> tmp;
    tmp.Multiply(PlAniso_full_,red);
    LINALG::Matrix<5,5> tmp55;
    tmp55.MultiplyTN(red,tmp);
    LINALG::FixedSizeSerialDenseSolver<5,5,1> solver;
    solver.SetMatrix(tmp55);
    int err2=solver.Factor();
    int err=solver.Invert();
    if ((err != 0) || (err2!=0)) dserror("Inversion of plastic anisotropy tensor failed");
    tmp.MultiplyNT(red,tmp55);
    InvPlAniso_full_.MultiplyNT(tmp,red);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate elastic stress and stiffness                   seitz 05/14 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateElast(
    const LINALG::Matrix<3,3>*defgrd,
    const LINALG::Matrix<3,3>* deltaLp,
    Teuchos::ParameterList& params,
    LINALG::Matrix<6,1>* pk2,
    LINALG::Matrix<6,6>* cmat,
    const int gp,
    const int eleGID)
{
  LINALG::Matrix<6,1> Cpi;
  LINALG::Matrix<6,1> CpiCCpi;
  LINALG::Matrix<6,1> ircg;

  LINALG::Matrix<3,1> prinv;

  LINALG::Matrix<3,1> dPI(true);
  LINALG::Matrix<6,1> ddPII(true);

  EvaluateKinQuantElast(defgrd,deltaLp,gp,Cpi,CpiCCpi,ircg,prinv);
  EvaluateInvariantDerivatives(prinv,dPI,ddPII,eleGID);

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  pk2->Clear();
  cmat->Clear();

  // isotropic elasticity in coupled strain energy format
  // isotropic elasticity in decoupled ("mod") format go here as well
  // as the modified gammas and deltas have been converted
  if (isoprinc_ || isomod_)
    EvaluateIsotropicPrincElast(*pk2,*cmat,Cpi,CpiCCpi,ircg,prinv,dPI,ddPII);
  else
    dserror("only isotropic hyperelastic materials");

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate thermal stress and stiffness                   seitz 06/14 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateThermalStress(
    const LINALG::Matrix<3,3>*defgrd,
    const double temp,
    Teuchos::ParameterList& params,
    LINALG::Matrix<6,1>* pk2,
    LINALG::Matrix<6,6>* cmat,
    const int gp,
    const int eleGID)
{
  // do TSI only for decoupled isotropic materials. By doing so, the stresses
  // due to thermal expansion can be easily calculated by the volumetric
  // part of the strain energy function
  if (isoprinc_ || anisomod_ || anisoprinc_ || isovisco_)
    dserror("TSI with semi-Smooth Newton type plasticity algorithm only "
        "with decoupled strain energy functions");

  // temperature difference
  double deltaT = temp-InitTemp();

  // we are only interested in the volumetric response
  // which is for decoupled strain energy functions defined by
  // modinv_3 = J only.
  LINALG::Matrix<3,1> modinv(true);
  modinv(2) = defgrd->Determinant();
  LINALG::Matrix<3,1> dPmodI;
  LINALG::Matrix<6,1> ddPmodII;
  double dddPmodIII =0.;

  // loop map of associated potential summands
  for (unsigned int p=0; p<potsum_.size(); ++p)
  {
    potsum_[p]->AddDerivativesModified(dPmodI,ddPmodII,modinv,eleGID);
    potsum_[p]->Add3rdVolDeriv(modinv,dddPmodIII);
  }

  // inverse RCG
  LINALG::Matrix<3,3> invRCG;
  invRCG.MultiplyTN(*defgrd,*defgrd);
  invRCG.Invert();
  LINALG::Matrix<6,1> icg;
  for (int i=0; i<3; ++i) icg(i) = invRCG(i,i);
  icg(3) = invRCG(0,1);
  icg(4) = invRCG(1,2);
  icg(5) = invRCG(0,2);

  pk2->Update(-3.*Cte()*deltaT*modinv(2)*ddPmodII(2),icg,1.);
  ElastSymTensorMultiply(*cmat,-3.*Cte()*deltaT*modinv(2)*modinv(2)*dddPmodIII,invRCG,invRCG,1.);
  ElastSymTensorMultiply(*cmat,-3.*Cte()*deltaT*modinv(2)*ddPmodII(2),invRCG,invRCG,1.);
  ElastSymTensor_o_Multiply(*cmat,+6.*Cte()*deltaT*modinv(2)*ddPmodII(2),invRCG,invRCG,1.);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate thermal stress and stiffness                   seitz 06/14 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateCTvol(
    const LINALG::Matrix<3,3>*defgrd,
    Teuchos::ParameterList& params,
    LINALG::Matrix<6,1>* cTvol,
    LINALG::Matrix<6,6>* dCTvoldE,
    const int gp,
    const int eleGID)
{
  // do TSI only for decoupled isotropic materials. By doing so, the stresses
  // due to thermal expansion can be easily calculated by the volumetric
  // part of the strain energy function
  if (isoprinc_ || anisomod_ || anisoprinc_ || isovisco_)
    dserror("TSI with semi-Smooth Newton type plasticity algorithm only "
        "with decoupled strain energy functions");

  // we are only interested in the volumetric response
  // which is for decoupled strain energy functions defined by
  // modinv_3 = J only.
  LINALG::Matrix<3,1> modinv(true);
  modinv(2) = defgrd->Determinant();
  LINALG::Matrix<3,1> dPmodI;
  LINALG::Matrix<6,1> ddPmodII;
  double dddPmodIII =0.;

  // loop map of associated potential summands
  for (unsigned int p=0; p<potsum_.size(); ++p)
  {
    potsum_[p]->AddDerivativesModified(dPmodI,ddPmodII,modinv,eleGID);
    potsum_[p]->Add3rdVolDeriv(modinv,dddPmodIII);
  }

  // clear
  cTvol->Clear();
  dCTvoldE->Clear();

  // inverse RCG
  LINALG::Matrix<3,3> invRCG;
  invRCG.MultiplyTN(*defgrd,*defgrd);
  invRCG.Invert();
  LINALG::Matrix<6,1> icg;
  for (int i=0; i<3; ++i) icg(i) = invRCG(i,i);
  icg(3) = invRCG(0,1);
  icg(4) = invRCG(1,2);
  icg(5) = invRCG(0,2);

  cTvol->Update(-3.*Cte()*modinv(2)*ddPmodII(2),icg,1.);
  ElastSymTensorMultiply(*dCTvoldE,-3.*Cte()*modinv(2)*modinv(2)*dddPmodIII,invRCG,invRCG,1.);
  ElastSymTensorMultiply(*dCTvoldE,-3.*Cte()*modinv(2)*ddPmodII(2),invRCG,invRCG,1.);
  ElastSymTensor_o_Multiply(*dCTvoldE,+6.*Cte()*modinv(2)*ddPmodII(2),invRCG,invRCG,1.);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate plastic stress and stiffness                   seitz 05/14 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluatePlast(
    const LINALG::Matrix<3,3>* defgrd,
    const LINALG::Matrix<3,3>* deltaDp,
    const double temp,
    Teuchos::ParameterList& params,
    LINALG::Matrix<6,6>* dPK2dDp,
    LINALG::Matrix<6,1>* NCP,
    LINALG::Matrix<6,6>* dNCPdC,
    LINALG::Matrix<6,6>* dNCPdDp,
    bool* active,
    bool* elast,
    bool* as_converged,
    const int gp,
    LINALG::Matrix<6,1>* dNCPdT,
    LINALG::Matrix<6,1>* dHdC,
    LINALG::Matrix<6,1>* dHdDp,
    const double dt,
    const int eleGID)
{
  LINALG::Matrix<6,1> Cpi;
  LINALG::Matrix<6,1> CpiCCpi;
  LINALG::Matrix<6,1> ircg;
  LINALG::Matrix<3,1> prinv;
  LINALG::Matrix<6,1> id2V;
  LINALG::Matrix<3,3> id2;
  LINALG::Matrix<3,3> CpiC;
  LINALG::Matrix<3,3> FpiCe;
  LINALG::Matrix<9,1> CFpiCei;
  LINALG::Matrix<9,1> CFpi;
  LINALG::Matrix<3,3> FpiTC;
  LINALG::Matrix<9,1> CFpiCe;
  LINALG::Matrix<3,3> CeFpiTC;
  LINALG::Matrix<6,1> Ce;
  LINALG::Matrix<3,3> CeM;
  LINALG::Matrix<6,1> Ce2;
  LINALG::Matrix<3,3> invpldefgrd;

  LINALG::Matrix<3,1> dPI;
  LINALG::Matrix<6,1> ddPII;
  LINALG::Matrix<3,1> gamma(true);
  LINALG::Matrix<8,1> delta(true);

  if (EvaluateKinQuantPlast(defgrd,deltaDp,gp,params,invpldefgrd,Cpi,CpiCCpi,ircg,Ce,CeM,Ce2,
                        id2V,id2,CpiC,FpiCe,CFpiCei,CFpi,FpiTC,CFpiCe,CeFpiTC,prinv))
    return;
  EvaluateInvariantDerivatives(prinv,dPI,ddPII,eleGID);
  CalculateGammaDelta(gamma,delta,prinv,dPI,ddPII);

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  dPK2dDp->Clear();
  NCP->Clear();
  dNCPdC->Clear();
  dNCPdDp->Clear();
  if (dNCPdT!=NULL) dNCPdT->Clear();

  // new temporary matrices
  LINALG::Matrix<3,3> mStr;     // Mandel stress tensor
  LINALG::Matrix<6,6> dMdC;     // derivative of Mandel stress w.r.t. RCG
  LINALG::Matrix<6,9> dMdFpinv; // derivative of Mandel stress w.r.t. inverse plastic deformation gradient
  LINALG::Matrix<6,9> dPK2dFpinv;

  // isotropic elasticity in coupled strain energy format
  // isotropic elasticity in decoupled ("mod") format go here as well
  // as the modified gammas and deltas have been converted
  if (isoprinc_ || isomod_)
  {
    EvaluateIsotropicPrincPlast(dPK2dFpinv,mStr,dMdC,dMdFpinv,
        Cpi,CpiCCpi,ircg,Ce,CeM,Ce2,id2V,id2,CpiC,FpiCe,
        invpldefgrd,CFpiCei,CFpi,FpiTC,CFpiCe,CeFpiTC,gamma,delta);
  }
  else
    dserror("only isotropic hypereleastic materials");

  EvaluateNCP(&mStr,&dMdC,&dMdFpinv,&dPK2dFpinv,deltaDp,gp,temp,NCP,dNCPdC,dNCPdDp,dNCPdT,dPK2dDp,active,elast,as_converged,dHdC,dHdDp,params,dt);

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate NCP function                                   seitz 05/14 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateNCP(
    const LINALG::Matrix<3,3>* mStr,
    const LINALG::Matrix<6,6>* dMdC,
    const LINALG::Matrix<6,9>* dMdFpinv,
    const LINALG::Matrix<6,9>* dPK2dFpinv,
    const LINALG::Matrix<3,3>* deltaDp,
    const int gp,
    const double temp,
    LINALG::Matrix<6,1>* NCP,
    LINALG::Matrix<6,6>* dNCPdC,
    LINALG::Matrix<6,6>* dNCPdDp,
    LINALG::Matrix<6,1>* dNCPdT,
    LINALG::Matrix<6,6>* dPK2dDp,
    bool* active,
    bool* elast,
    bool* as_converged,
    LINALG::Matrix<6,1>* dHdC,
    LINALG::Matrix<6,1>* dHdDp,
    Teuchos::ParameterList& params,
    const double dt
    )
{
  const double sq=sqrt(2./3.);
  LINALG::Matrix<6,1> tmp61;
  double dT=0.;
  if (dNCPdT) dT=temp-InitTemp();
  else        dT=0.;

  // deviatoric projection tensor
  LINALG::Matrix<6,6> pdev(true);
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      if (i==j) pdev(i,j) = 2./3.;
      else      pdev(i,j) = -1./3.;
  for (int i=3; i<6; i++) pdev(i,i) = 1.;

  // effective stress
  LINALG::Matrix<3,3> eta(*mStr);
  for (int i=0; i<3; i++)
    eta(i,i) -= 1./3.*((*mStr)(0,0) + (*mStr)(1,1) + (*mStr)(2,2));
  eta.Update(2./3.*Kinhard(),last_alpha_kinematic_[gp],1.);
  eta.Update(-2./3.*Kinhard(),*deltaDp,1.);

  // in stress-like voigt notation
  LINALG::Matrix<6,1> eta_v;   // in stress-like voigt notation
  LINALG::Matrix<6,1> etatr_v; // in stress-like voigt notation
  LINALG::Matrix<6,1> eta_s_v; // in stress-like voigt notation
  LINALG::Matrix<6,1> deltaDp_v;// in stress-like voigt notation
  for (int i=0; i<3; i++) { eta_v(i)=eta(i,i); deltaDp_v(i)=(*deltaDp)(i,i); }
  eta_v(3)=.5*(eta(1,0)+eta(0,1)); deltaDp_v(3)=.5*((*deltaDp)(0,1)+(*deltaDp)(1,0));
  eta_v(4)=.5*(eta(1,2)+eta(2,1)); deltaDp_v(4)=.5*((*deltaDp)(2,1)+(*deltaDp)(1,2));
  eta_v(5)=.5*(eta(2,0)+eta(0,2)); deltaDp_v(5)=.5*((*deltaDp)(0,2)+(*deltaDp)(2,0));

  // trial effective stress
  etatr_v.Update(eta_v);
  etatr_v.Multiply(cpl(),InvPlAniso_full_,deltaDp_v,1.);

  // in strain-like voigt notation
  LINALG::Matrix<6,1> eta_v_strainlike(eta_v);         // in strain-like voigt notation
  LINALG::Matrix<6,1> etatr_v_strainlike(etatr_v);     // in strain-like voigt notation
  LINALG::Matrix<6,1> deltaDp_v_strainlike(deltaDp_v); // in strain-like voigt notation
  for (int i=3; i<6; i++) { eta_v_strainlike(i)*=2.; etatr_v_strainlike(i)*=2.; deltaDp_v_strainlike(i)*=2.; }

  // different tensor norms
  tmp61.Multiply(PlAniso_full_,eta_v);
  double absHeta=NormStressLike(tmp61);
  double abseta_H=tmp61.Dot(eta_v_strainlike);;
  if (abseta_H<-1.e-16) dserror("this should not happen. eta : H : eta =%f < 0",abseta_H);
  else if (abseta_H>=0.) abseta_H=sqrt(abseta_H);
  else abseta_H=0.;
  double dDpHeta=tmp61.Dot(deltaDp_v_strainlike);
  tmp61.Multiply(PlAniso_full_,etatr_v);
  double absetatr_H=tmp61.Dot(etatr_v_strainlike);
  if (absetatr_H<-1.e-16) dserror("this should not happen. eta_tr : H : eta_tr =%f < 0",absetatr_H);
  else if (absetatr_H>=0.) absetatr_H=sqrt(absetatr_H);
  else absetatr_H=0.;
  LINALG::Matrix<6,1> HdDp;
  HdDp.Multiply(PlAniso_full_,deltaDp_v);
  LINALG::Matrix<6,1> HdDp_strainlike;
  HdDp_strainlike.Multiply(PlAniso_full_,deltaDp_v_strainlike);
  LINALG::Matrix<6,1> HetaH_strainlike;
  tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
  HetaH_strainlike.Multiply(PlAniso_full_,tmp61);

  // isotropic hardening increment
  delta_alpha_i_[gp]=0.;
  if (dDpHeta>0. && absHeta>0.)
    delta_alpha_i_[gp]=sq*dDpHeta*abseta_H/(absHeta*absHeta);
  // new isotropic hardening value
  const double aI = last_alpha_isotropic_[gp]+delta_alpha_i_[gp];

  // current yield stress equivalent (yield stress scaled by sqrt(2/3))
  double ypl = sq * (
                           (Infyield()*(1.-HardSoft()*dT) - Inityield()*(1.-YieldSoft()*dT))
                                 *( 1.-exp(-Expisohard()*aI) )
                            + Isohard() *(1.- HardSoft()*dT)*aI
                            +Inityield()*(1.-YieldSoft()*dT)
                     )
                      *pow(1.+Visc()*(1.-ViscSoft()*dT)*delta_alpha_i_[gp]/dt,ViscRate())
                     ;

  double dYpldT = sq*(
                        (Infyield()*(-HardSoft())-Inityield()*(-YieldSoft()))
                          *(1.-exp(-Expisohard()*aI))
                        -Isohard()*HardSoft()*aI
                        -Inityield()*YieldSoft()
                      )
                      *pow(1.+Visc()*(1.-ViscSoft()*dT)*delta_alpha_i_[gp]/dt,ViscRate())
                      ;

  dYpldT += sq * (
                  (Infyield()*(1.-HardSoft()*dT) - Inityield()*(1.-YieldSoft()*dT))
                        *( 1.-exp(-Expisohard()*aI) )
                   + Isohard() *(1.- HardSoft()*dT)*aI
                   +Inityield()*(1.-YieldSoft()*dT)
                 )
                   *pow(1.+Visc()*(1.-ViscSoft()*dT)*delta_alpha_i_[gp]/dt,ViscRate()-1.)
                     *ViscRate()*delta_alpha_i_[gp]/dt*Visc()*(-ViscSoft());

  // Factor of derivative of Y^pl w.r.t. delta alpha ^i
  // we have added the factor sqrt(2/3) from delta_alpha_i=sq*... here
  double dYplDai = 2./3.*(
                          +Isohard()*(1.-HardSoft()*dT)
                          +(Infyield()*(1.-HardSoft()*dT)-Inityield()*(1.-YieldSoft()*dT))
                              *Expisohard()*exp(-Expisohard()*aI)
                         )
                         *pow(1.+Visc()*(1.-ViscSoft()*dT)*delta_alpha_i_[gp]/dt,ViscRate())
                         ;
  dYplDai += 2./3.*(
                    (Infyield()*(1.-HardSoft()*dT) - Inityield()*(1.-YieldSoft()*dT))
                          *( 1.-exp(-Expisohard()*aI) )
                     + Isohard() *(1.- HardSoft()*dT)*aI
                     +Inityield()*(1.-YieldSoft()*dT)
                   )
                    *pow(1.+Visc()*(1.-ViscSoft()*dT)*delta_alpha_i_[gp]/dt,ViscRate()-1)
                      *ViscRate()*Visc()*(1.-ViscSoft()*dT)/dt;

  // activity state check
  if (ypl<absetatr_H)
  {
    if (activity_state_[gp]==false) // gp switches state
    {
      if (abs(ypl-absetatr_H)>AS_CONVERGENCE_TOL*Inityield()
          || deltaDp->NormInf()>AS_CONVERGENCE_TOL*Inityield()/cpl())
        *as_converged = false;
    }
    activity_state_[gp] = true;
    *active=true;
  }
  else
  {
    if (activity_state_[gp]==true) // gp switches state
    {
      if (abs(ypl-absetatr_H)>AS_CONVERGENCE_TOL*Inityield()
          || deltaDp->NormInf()>AS_CONVERGENCE_TOL*Inityield()/cpl())
        *as_converged = false;
    }
    activity_state_[gp] = false;
    *active=false;
  }

  // these cases have some terms in common
  if (*active || dDpHeta>0. || deltaDp->NormInf()>0.)
  {
    // damping parameter apl
    double apl=1.;
    if (ypl<abseta_H)
      apl=ypl/abseta_H;

    // eta_s to abbreviate calculation of the derivatives
    if (absetatr_H>0.)
      eta_s_v.Update((1.-s())*ypl/absetatr_H,etatr_v,1.);
    eta_s_v.Update(apl*s(),eta_v,1.);

    // matrix exponential derivative
    LINALG::Matrix<6,6> Dexp(false);
    LINALG::Matrix<3,3> tmp(*deltaDp);
    tmp.Scale(-1.);
    MatrixExponentialDerivativeSym3x3(tmp,Dexp);

    // Derivative of inverse plastic deformation gradient
    LINALG::Matrix<9,6> dFpiDdeltaDp(true);
    for (int A=0; A<3; A++)
      for (int a=0; a<3; a++)
        for (int b=0; b<3; b++)
          for (int i=0; i<6; i++)
            if (i<=2)
              dFpiDdeltaDp(VOIGT3X3NONSYM_[A][a],i) -= last_plastic_defgrd_inverse_[gp](A,b)*Dexp(VOIGT3X3_[b][a],i);
            else
              dFpiDdeltaDp(VOIGT3X3NONSYM_[A][a],i) -= 2.*last_plastic_defgrd_inverse_[gp](A,b)*Dexp(VOIGT3X3_[b][a],i);

    // derivative of mandel stress
    // we spare the deviatoric projection of the mandel stress derivative to get the effective stress derivative.
    // It is enforced implicitly as the "detaddp" is always contracted with deviatoric tensors.
    LINALG::Matrix<6,6> detaddp;
    detaddp.Multiply(*dMdFpinv,dFpiDdeltaDp);
    detaddp.Update(-2./3.*Kinhard(),pdev,1.);
    dPK2dDp->Multiply(*dPK2dFpinv,dFpiDdeltaDp);

    //TSI
    if (dNCPdT!=NULL)
    {
      // plastic heating
      double plHeating = (0.
          -Isohard()*HardSoft()*aI
          -(Infyield()*HardSoft()-Inityield()*YieldSoft())
          *(1.-exp(-Expisohard()*aI))
      )*temp*delta_alpha_i_[gp];
      switch (DisMode())
      {
      case INPAR::TSI::pl_multiplier:
        plHeating+=delta_alpha_i_[gp]*(0.
            +Inityield()*(1.-YieldSoft()*dT)
            +Isohard()*(1.-HardSoft()*dT)*aI
            +(Infyield()*(1.-HardSoft()*dT)-Inityield()*(1.-YieldSoft()*dT))
              *(1.-exp(-Expisohard()*aI))
                              );
        break;
      case INPAR::TSI::pl_flow:
        plHeating += eta_v_strainlike.Dot(deltaDp_v);
        break;
      default:
        dserror("unknown plastic dissipation mode: %d",DisMode());
        break;
      }

      // derivative w.r.t. temperature
      double dPlHeatingDT = (0.
          -Isohard()*HardSoft()*aI
          +(Infyield()*(-HardSoft())-Inityield()*(-YieldSoft()))
          *(1.-exp(-Expisohard()*aI))
      )*delta_alpha_i_[gp];
      switch (DisMode())
      {
      case INPAR::TSI::pl_multiplier:
        dPlHeatingDT+=-delta_alpha_i_[gp]*(0.
            +Inityield()*YieldSoft()
            +Isohard()*HardSoft()*aI
            +(Infyield()*HardSoft()-Inityield()*YieldSoft())
            *(1.-exp(-Expisohard()*aI))
        );
        break;
      case INPAR::TSI::pl_flow:
        // do nothing
        break;
      default:
        dserror("unknown plastic dissipation mode: %d",DisMode());
        break;
      }

      // derivative w.r.t. Delta alpha i
      double dPlHeatingDdai = temp*(0.
          -Isohard()*HardSoft()*aI
          +(Infyield()*(-HardSoft())-Inityield()*(-YieldSoft()))
          *(1.-exp(-Expisohard()*aI))
      )
      +temp*delta_alpha_i_[gp]*(0.
          -Isohard()*HardSoft()
          +(-Infyield()*HardSoft()+Inityield()*YieldSoft())
          *Expisohard()*exp(-Expisohard()*aI)
      );
      switch (DisMode())
      {
      case INPAR::TSI::pl_multiplier:
        dPlHeatingDdai+= +Inityield()*(1.-YieldSoft()*dT)
        +Isohard()*(1.-HardSoft()*dT)*(last_alpha_isotropic_[gp]+2.*delta_alpha_i_[gp])
        +(Infyield()*(1.-HardSoft()*dT)-Inityield()*(1.-YieldSoft()*dT))
        *( (1.-exp(-Expisohard()*aI)) + delta_alpha_i_[gp]*Expisohard()*exp(-Expisohard()*aI) );
        break;
      case INPAR::TSI::pl_flow:
        // do nothing
        break;
      default:
        dserror("unknown plastic dissipation mode: %d",DisMode());
        break;
      }

      // this factor is from the evolution equation for delta_alpha_i
      dPlHeatingDdai*=sq;

      // derivative w.r.t. eta
      LINALG::Matrix<6,1> dHpDeta(true);
      if (dDpHeta>0.)
      {
        tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
        dHpDeta.Update(dPlHeatingDdai*dDpHeta/(abseta_H*absHeta*absHeta),tmp61,1.);
        dHpDeta.Update(dPlHeatingDdai*abseta_H/(absHeta*absHeta),HdDp_strainlike,1.);
        dHpDeta.Update(-2.*dPlHeatingDdai*abseta_H*dDpHeta/(pow(absHeta,4.)),HetaH_strainlike,1.);
      }

      if (DisMode()==INPAR::TSI::pl_flow)
        dHpDeta.Update(1.,deltaDp_v_strainlike,1.);

      //derivative w.r.t. C
      dHdC->MultiplyTN(*dMdC,dHpDeta);

      // derivative w.r.t. Delta Dp
      dHdDp->MultiplyTN(detaddp,dHpDeta);
      if (dDpHeta>0.)
      {
        tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
        dHdDp->Update(dPlHeatingDdai*abseta_H/(absHeta*absHeta),tmp61,1.);
      }
      if (DisMode()==INPAR::TSI::pl_flow)
        dHdDp->Update(1.,eta_v_strainlike,1.);

      // scaling with time step
      plHeating          /=dt;
      dPlHeatingDT       /=dt;
      dHdC-> Scale(1./dt);
      dHdDp->Scale(1./dt);

      // communicate to the element via params (not nice)
      HepDiss(gp)+=plHeating;
      (*dHepDissdT_)[gp]+= dPlHeatingDT;
    } //TSI

    // plastic gp
    if (*active)
    {
      // this is a plastic gp
      *elast=false;

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      LINALG::Matrix<6,6> dNCPdeta;
      dNCPdeta.Update(1.-ypl/absetatr_H,pdev,1.);
      tmp61.Multiply(PlAniso_full_,etatr_v_strainlike);
      dNCPdeta.MultiplyNT(1./(absetatr_H*absetatr_H),eta_s_v,tmp61,1.);
      if (dDpHeta>0.)
      {
        tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
        dNCPdeta.MultiplyNT(-dYplDai*dDpHeta/(abseta_H*absHeta*absHeta*absetatr_H),etatr_v,tmp61,1.);
        dNCPdeta.MultiplyNT(-dYplDai*abseta_H/(absHeta*absHeta*absetatr_H),etatr_v,HdDp_strainlike,1.);
        dNCPdeta.MultiplyNT(2.*dYplDai*abseta_H*dDpHeta/(pow(absHeta,4.)*absetatr_H),etatr_v,HetaH_strainlike,1.);
      }

      // derivative w.r.t. C
      dNCPdC ->Multiply(dNCPdeta,*dMdC);

      // derivative w.r.t. deltaDp
      dNCPdDp->Multiply(dNCPdeta,detaddp);
      LINALG::Matrix<6,6> dNCPdetatr;
      tmp61.Multiply(PlAniso_full_,etatr_v_strainlike);
      dNCPdetatr.MultiplyNT(cpl()/(absetatr_H*absetatr_H),eta_s_v,tmp61,1.);
      dNCPdetatr.Update(-cpl()*ypl/absetatr_H,pdev,1.);
      dNCPdDp->Multiply(1.,dNCPdetatr,InvPlAniso_full_,1.);
      if (dDpHeta>0.)
      {
        tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
        dNCPdDp->MultiplyNT(-dYplDai*abseta_H/(absetatr_H*absHeta*absHeta),etatr_v,tmp61,1.);
      }

      // residual
      NCP->Update(eta_v);
      NCP->Update(-ypl/absetatr_H,etatr_v,1.);

      // derivative w.r.t. temperature
      if (dNCPdT!=NULL)
        dNCPdT->Update(-dYpldT/absetatr_H,etatr_v,0.);
    }

    // not active but needs condensation due to acitivity in last iteration
    else if (dDpHeta>0.)
    {
      // this is an elastic gp, but the flag "elast" is reserved for those
      // elastic GP that do not require a condensation
      *elast=false;

      // residual
      NCP->Multiply(-cpl(),InvPlAniso_full_,deltaDp_v,1.);

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      LINALG::Matrix<6,6> dNCPdeta;

      tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
      dNCPdeta.MultiplyNT(-s()*dYplDai*dDpHeta/(abseta_H*absHeta*absHeta*ypl),*NCP,tmp61,1.);
      dNCPdeta.MultiplyNT(-s()*dYplDai*abseta_H/(absHeta*absHeta*ypl),*NCP,HdDp_strainlike,1.);
      dNCPdeta.MultiplyNT(s()*2.*dYplDai*abseta_H*dDpHeta/(pow(absHeta,4.)*ypl),*NCP,HetaH_strainlike,1.);

        // derivative w.r.t. C
      dNCPdC->Multiply(dNCPdeta,*dMdC);

      // derivative w.r.t. deltaDp
      dNCPdDp->Update(-cpl(),InvPlAniso_full_,1.);
      dNCPdDp->Multiply(1.,dNCPdeta,detaddp,1.);
      tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
      dNCPdDp->MultiplyNT(dYplDai/(ypl*absHeta),*NCP,tmp61,1.);

      // derivative w.r.t. temperature
      if (dNCPdT!=NULL)
        dNCPdT->Multiply(-s()*cpl()/ypl*dYpldT,InvPlAniso_full_,deltaDp_v,0.);
    }
    else
    {
      // Cpl = cpl* Delta D^p = 0
      // we don't build the matrix blocks here.
      // The trivial identity is enforced at the element
      *elast=true;
      // todo: currently, the coupling term dypldT is neglected
      // in the elastic realm. this is a slightly inconsistent linearization for s!=0
      // however, that way we can ensure that deltaLp=0 (exactly) at any Newton iteration

      if (dNCPdT!=NULL)
        dNCPdT->Clear();
    }
  }
  // elastic gp
  else
  {
    *elast=true;
  }

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate plastic stress and stiffness (with pl. spin)   seitz 05/14 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluatePlast(
    const LINALG::Matrix<3,3>* defgrd,
    const LINALG::Matrix<3,3>* deltaLp,
    const double temp,
    Teuchos::ParameterList& params,
    LINALG::Matrix<6,9>* dPK2dLp,
    LINALG::Matrix<9,1>* NCP,
    LINALG::Matrix<9,6>* dNCPdC,
    LINALG::Matrix<9,9>* dNCPdLp,
    bool* active,
    bool* elast,
    bool* as_converged,
    const int gp,
    LINALG::Matrix<9,1>* dNCPdT,
    LINALG::Matrix<6,1>* dHdC,
    LINALG::Matrix<9,1>* dHdLp,
    const double dt,
    const int eleGID
    )
{
  LINALG::Matrix<6,1> Cpi;
  LINALG::Matrix<6,1> CpiCCpi;
  LINALG::Matrix<6,1> ircg;
  LINALG::Matrix<6,1> id2V;
  LINALG::Matrix<3,3> id2;
  LINALG::Matrix<3,3> CpiC;
  LINALG::Matrix<3,3> FpiCe;
  LINALG::Matrix<9,1> CFpiCei;
  LINALG::Matrix<9,1> CFpi;
  LINALG::Matrix<3,3> FpiTC;
  LINALG::Matrix<9,1> CFpiCe;
  LINALG::Matrix<3,3> CeFpiTC;
  LINALG::Matrix<6,1> Ce;
  LINALG::Matrix<3,3> CeM;
  LINALG::Matrix<6,1> Ce2;
  LINALG::Matrix<3,3> invpldefgrd;

  LINALG::Matrix<3,1> prinv;
  LINALG::Matrix<3,1> dPI(true);
  LINALG::Matrix<6,1> ddPII(true);
  LINALG::Matrix<3,1> gamma(true);
  LINALG::Matrix<8,1> delta(true);

  if (EvaluateKinQuantPlast(defgrd,deltaLp,gp,params,invpldefgrd,Cpi,CpiCCpi,ircg,Ce,CeM,Ce2,
                        id2V,id2,CpiC,FpiCe,CFpiCei,CFpi,FpiTC,CFpiCe,CeFpiTC,prinv))
    return;
  EvaluateInvariantDerivatives(prinv,dPI,ddPII,eleGID);
  CalculateGammaDelta(gamma,delta,prinv,dPI,ddPII);

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  dPK2dLp->Clear();
  NCP->Clear();
  dNCPdC->Clear();
  dNCPdLp->Clear();
  if (dNCPdT!=NULL) dNCPdT->Clear();

  // new temporary matrices
  LINALG::Matrix<3,3> mStr;     // Mandel stress tensor
  LINALG::Matrix<6,6> dMdC;     // derivative of Mandel stress w.r.t. RCG
  LINALG::Matrix<6,9> dMdFpinv; // derivative of Mandel stress w.r.t. inverse plastic deformation gradient
  LINALG::Matrix<6,9> dPK2dFpinv;

  // isotropic elasticity in coupled strain energy format
  // isotropic elasticity in decoupled ("mod") format go here as well
  // as the modified gammas and deltas have been converted
  if (isoprinc_ || isomod_)
  {
    EvaluateIsotropicPrincPlast(dPK2dFpinv,mStr,dMdC,dMdFpinv,
        Cpi,CpiCCpi,ircg,Ce,CeM,Ce2,id2V,id2,CpiC,FpiCe,
        invpldefgrd,CFpiCei,CFpi,FpiTC,CFpiCe,CeFpiTC,gamma,delta);
  }
  else
    dserror("only isotropic hypereleastic materials");

  EvaluateNCPandSpin(&mStr,&dMdC,&dMdFpinv,&dPK2dFpinv,deltaLp,gp,NCP,dNCPdC,dNCPdLp,dPK2dLp,active,elast,as_converged,dt);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate NCP function and plastic spin equation         seitz 05/14 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateNCPandSpin(
    const LINALG::Matrix<3,3>* mStr,
    const LINALG::Matrix<6,6>* dMdC,
    const LINALG::Matrix<6,9>* dMdFpinv,
    const LINALG::Matrix<6,9>* dPK2dFpinv,
    const LINALG::Matrix<3,3>* deltaLp,
    const int gp,
    LINALG::Matrix<9,1>* NCP,
    LINALG::Matrix<9,6>* dNCPdC,
    LINALG::Matrix<9,9>* dNCPdLp,
    LINALG::Matrix<6,9>* dPK2dLp,
    bool* active,
    bool* elast,
    bool* as_converged,
    const double dt)
{
  const double sq=sqrt(2./3.);
  LINALG::Matrix<6,1> tmp61;

  // deviatoric projection tensor
  LINALG::Matrix<6,6> pdev(true);
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      if (i==j) pdev(i,j) = +2./3.;
      else      pdev(i,j) = -1./3.;
  for (int i=3; i<6; i++) pdev(i,i) = 1.;

  // deviatoric symmetric projection tensor (A-->dev(sym(A))
  LINALG::Matrix<6,9> psymdev(true);
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      if (i==j) psymdev(i,j) = +2./3.;
      else      psymdev(i,j) = -1./3.;
  for (int i=3; i<6; i++) psymdev(i,i) = psymdev(i,i+3) = .5;

  // symmetric identity
  LINALG::Matrix<6,9> psym(true);
  for (int i=0; i<3; i++) psym(i,i)=1.;
  for (int i=3; i<6; i++) psym(i,i) = psym(i,i+3) = .5;


  // effective stress
  LINALG::Matrix<3,3> eta(*mStr);
  for (int i=0; i<3; i++)
    eta(i,i) -= 1./3.*((*mStr)(0,0) + (*mStr)(1,1) + (*mStr)(2,2));
  eta.Update(2./3.*Kinhard(),last_alpha_kinematic_[gp],1.);
  eta.Update(-1./3.*Kinhard(),*deltaLp,1.);
  eta.UpdateT(-1./3.*Kinhard(),*deltaLp,1.);

  // in stress-like voigt notation
  LINALG::Matrix<6,1> eta_v;   // in stress-like voigt notation
  LINALG::Matrix<6,1> etatr_v; // in stress-like voigt notation
  LINALG::Matrix<6,1> eta_s_v; // in stress-like voigt notation
  LINALG::Matrix<6,1> deltaDp_v;// in stress-like voigt notation
  for (int i=0; i<3; i++) { eta_v(i)=eta(i,i); deltaDp_v(i)=(*deltaLp)(i,i); }
  eta_v(3)=.5*(eta(1,0)+eta(0,1)); deltaDp_v(3)=.5*((*deltaLp)(0,1)+(*deltaLp)(1,0));
  eta_v(4)=.5*(eta(1,2)+eta(2,1)); deltaDp_v(4)=.5*((*deltaLp)(2,1)+(*deltaLp)(1,2));
  eta_v(5)=.5*(eta(2,0)+eta(0,2)); deltaDp_v(5)=.5*((*deltaLp)(0,2)+(*deltaLp)(2,0));

  // trial effective stress
  etatr_v.Update(eta_v);
  etatr_v.Multiply(cpl(),InvPlAniso_full_,deltaDp_v,1.);

  // in strain-like voigt notation
  LINALG::Matrix<6,1> eta_v_strainlike(eta_v);         // in strain-like voigt notation
  LINALG::Matrix<6,1> etatr_v_strainlike(etatr_v);     // in strain-like voigt notation
  LINALG::Matrix<6,1> deltaDp_v_strainlike(deltaDp_v); // in strain-like voigt notation
  for (int i=3; i<6; i++) { eta_v_strainlike(i)*=2.; etatr_v_strainlike(i)*=2.; deltaDp_v_strainlike(i)*=2.; }

  tmp61.Multiply(PlAniso_full_,eta_v);
  double absHeta=NormStressLike(tmp61);
  double abseta_H=tmp61.Dot(eta_v_strainlike);
  if (abseta_H<-1.e-16) dserror("this should not happen. tmp=%f",abseta_H);
  else if (abseta_H>=0.) abseta_H=sqrt(abseta_H);
  else dserror("this should not happen. tmp=%f",abseta_H);
  double dDpHeta=tmp61.Dot(deltaDp_v_strainlike);
  tmp61.Multiply(PlAniso_full_,etatr_v);
  double absetatr_H=tmp61.Dot(etatr_v_strainlike);
  if (absetatr_H<-1.e-16) dserror("this should not happen. tmp=%f",absetatr_H);
  else if (absetatr_H>=0.) absetatr_H=sqrt(absetatr_H);
  else dserror("this should not happen. tmp=%f",absetatr_H);
  LINALG::Matrix<6,1> HdDp;
  HdDp.Multiply(PlAniso_full_,deltaDp_v);
  LINALG::Matrix<6,1> HdDp_strainlike;
  HdDp_strainlike.Multiply(PlAniso_full_,deltaDp_v_strainlike);
  LINALG::Matrix<6,1> HetaH_strainlike;
  tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
  HetaH_strainlike.Multiply(PlAniso_full_,tmp61);

  // isotropic hardening increment
  delta_alpha_i_[gp]=0.;
  if (dDpHeta>0. && absHeta>0.)
    delta_alpha_i_[gp]=sq*dDpHeta*abseta_H/(absHeta*absHeta);

  // new isotropic hardening value
  const double aI = last_alpha_isotropic_[gp]+delta_alpha_i_[gp];

  // current yield stress equivalent (yield stress scaled by sqrt(2/3))
  double ypl = sq * (
                           (Infyield() - Inityield())
                                 *( 1.-exp(-Expisohard()*aI) )
                            + Isohard() *aI
                            +Inityield()
                     )
                      *pow(1.+Visc()*delta_alpha_i_[gp]/dt,ViscRate())
                     ;

  // check activity state
  if (ypl<absetatr_H)
  {
    if (activity_state_[gp]==false) // gp switches state
    {
      if (abs(ypl-absetatr_H)>AS_CONVERGENCE_TOL*Inityield()
          || deltaLp->NormInf()>AS_CONVERGENCE_TOL*Inityield()/cpl())
        *as_converged = false;
    }
    activity_state_[gp] = true;
    *active=true;
  }
  else
  {
    if (activity_state_[gp]==true) // gp switches state
    {
      if (abs(ypl-absetatr_H)>AS_CONVERGENCE_TOL*Inityield()
          || deltaLp->NormInf()>AS_CONVERGENCE_TOL*Inityield()/cpl())
        *as_converged = false;
    }
    activity_state_[gp] = false;
    *active=false;
  }

  // these cases have some terms in common
  if (*active || dDpHeta>0. || deltaLp->NormInf()>0.)
  {
    //derivative of the NCP function w.r.t. RCG / Delta Lp
    // without the lines corresponding to the plastic spin
    LINALG::Matrix<6,6> dNCPdC_red;
    LINALG::Matrix<6,9> dNCPdLp_red;
    LINALG::Matrix<6,1> NCP_red;

    // damping parameter apl
    double apl=1.;
    if (ypl/abseta_H<1.)
      apl=ypl/abseta_H;

    eta_s_v.Update((1.-s())*ypl/absetatr_H,etatr_v,1.);
    eta_s_v.Update(apl*s(),eta_v,1.);

    // matrix exponential derivative
    LINALG::Matrix<9,9> Dexp(false);
    LINALG::Matrix<3,3> tmp(*deltaLp);
    tmp.Scale(-1.);
    MatrixExponentialDerivative3x3(tmp,Dexp);

    // Derivative of inverse plastic deformation gradient
    LINALG::Matrix<9,9> dFpiDdeltaLp(true);
    for (int A=0; A<3; A++)
       for (int a=0; a<3; a++)
         for (int b=0; b<3; b++)
           for (int i=0; i<9; i++)
             dFpiDdeltaLp(VOIGT3X3NONSYM_[A][a],i) -= last_plastic_defgrd_inverse_[gp](A,b)*Dexp(VOIGT3X3NONSYM_[b][a],i);

    // derivative of mandel stress
    LINALG::Matrix<6,9> dMdLp;
    dMdLp.Multiply(*dMdFpinv,dFpiDdeltaLp);
    // we spare the deviatoric projection of the mandel stress derivative to get the effective stress derivative.
    // It is enforced implicitly as the "detadLp" is always contracted with deviatoric tensors.
    LINALG::Matrix<6,9> detadLp(dMdLp);
    detadLp.Update(-2./3.*Kinhard(),psymdev,1.);
    dPK2dLp->Multiply(*dPK2dFpinv,dFpiDdeltaLp);

    // Factor of derivative of Y^pl w.r.t. delta alpha ^i
    // we have added the factor sqrt(2/3) from delta_alpha_i=sq*... here
    double dYplDai = 2./3.*(
                            +Isohard()
                            +(Infyield()-Inityield())
                                *Expisohard()*exp(-Expisohard()*aI)
                           )
                           *pow(1.+Visc()*delta_alpha_i_[gp]/dt,ViscRate())
                           ;
    dYplDai += 2./3.*(
                      (Infyield() - Inityield())
                            *( 1.-exp(-Expisohard()*aI) )
                       + Isohard() *aI
                       +Inityield()
                     )
                      *pow(1.+Visc()*delta_alpha_i_[gp]/dt,ViscRate()-1)
                        *ViscRate()*Visc()/dt;

    // plastic gp
    if (*active)
    {
      // this is a plastic gp
      *elast=false;

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      LINALG::Matrix<6,6> dNCPdeta;
      dNCPdeta.Update(1.-ypl/absetatr_H,pdev,1.);
      tmp61.Multiply(PlAniso_full_,etatr_v_strainlike);
      dNCPdeta.MultiplyNT(1./(absetatr_H*absetatr_H),eta_s_v,tmp61,1.);
      if (dDpHeta>0.)
      {
        tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
        dNCPdeta.MultiplyNT(-dYplDai*dDpHeta/(abseta_H*absHeta*absHeta*absetatr_H),etatr_v,tmp61,1.);
        dNCPdeta.MultiplyNT(-dYplDai*abseta_H/(absHeta*absHeta*absetatr_H),etatr_v,HdDp_strainlike,1.);
        dNCPdeta.MultiplyNT(2.*dYplDai*abseta_H*dDpHeta/(pow(absHeta,4.)*absetatr_H),etatr_v,HetaH_strainlike,1.);
      }

      // derivative w.r.t. C
      dNCPdC_red.Multiply(dNCPdeta,*dMdC);

      // derivative w.r.t. deltaLp
      dNCPdLp_red.Multiply(dNCPdeta,detadLp);
      LINALG::Matrix<6,6> dNCPdetatr;
      tmp61.Multiply(PlAniso_full_,etatr_v_strainlike);
      dNCPdetatr.MultiplyNT(cpl()/(absetatr_H*absetatr_H),eta_s_v,tmp61,1.);
      dNCPdetatr.Update(-cpl()*ypl/absetatr_H,pdev,1.);
      LINALG::Matrix<6,6> dNCPdDp;
      dNCPdDp.Multiply(dNCPdetatr,InvPlAniso_full_);
      if (dDpHeta>0.)
      {
        tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
        dNCPdDp.MultiplyNT(-dYplDai*abseta_H/(absetatr_H*absHeta*absHeta),etatr_v,tmp61,1.);
      }
      dNCPdLp_red.Multiply(1.,dNCPdDp,psym,1.);

      // residual
      NCP_red.Update(eta_v);
      NCP_red.Update(-ypl/absetatr_H,etatr_v,1.);
    }

    // not active but needs condensation due to acitivity in last iteration
    else if (dDpHeta>0.)
    {
      // this is an elastic gp, but the flag "elast" is reserved for those
      // elastic GP that do not require a condensation
      *elast=false;

      // residual
      NCP_red.Multiply(-cpl(),InvPlAniso_full_,deltaDp_v,1.);

      // derivative of the complementarity function w.r.t. to the mandel stress tensor
      LINALG::Matrix<6,6> dNCPdeta;

      tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
      dNCPdeta.MultiplyNT(-s()*dYplDai*dDpHeta/(abseta_H*absHeta*absHeta*ypl),NCP_red,tmp61,1.);
      dNCPdeta.MultiplyNT(-s()*dYplDai*abseta_H/(absHeta*absHeta*ypl),NCP_red,HdDp_strainlike,1.);
      dNCPdeta.MultiplyNT(s()*2.*dYplDai*abseta_H*dDpHeta/(pow(absHeta,4.)*ypl),NCP_red,HetaH_strainlike,1.);

        // derivative w.r.t. C
      dNCPdC_red.Multiply(dNCPdeta,*dMdC);

      LINALG::Matrix<6,6> dNCPdDp;
      dNCPdDp.Update(-cpl(),InvPlAniso_full_,1.);
      tmp61.Multiply(PlAniso_full_,eta_v_strainlike);
      dNCPdDp.MultiplyNT(dYplDai/(ypl*absHeta),NCP_red,tmp61,1.);
      dNCPdLp_red.Multiply(1.,dNCPdDp,psym,1.);
      dNCPdLp_red.Multiply(1.,dNCPdeta,detadLp,1.);

    }
    else
    {
      // Cpl = cpl* Delta D^p = 0
      // we don't build the matrix blocks here.
      // The trivial identity is enforced at the element
      *elast=true;
    }

    // plastic spin equation
    // the tensor product Sigma.Dp is not made for voigt notation so we do it the hard way
    LINALG::Matrix<3,1> spEq;
    spEq(0)=.5*((*deltaLp)(0,1)-(*deltaLp)(1,0))-PlSpinChi()/Inityield()*(
         (*mStr)(0,0)*deltaDp_v(3)
        +(*mStr)(0,1)*deltaDp_v(1)
        +(*mStr)(0,2)*deltaDp_v(4)
        -(*mStr)(0,1)*deltaDp_v(0)
        -(*mStr)(1,1)*deltaDp_v(3)
        -(*mStr)(1,2)*deltaDp_v(5)
        );
    spEq(1)=.5*((*deltaLp)(1,2)-(*deltaLp)(2,1))-PlSpinChi()/Inityield()*(
        (*mStr)(0,1)*deltaDp_v(5)
       +(*mStr)(1,1)*deltaDp_v(4)
       +(*mStr)(1,2)*deltaDp_v(2)
       -(*mStr)(0,2)*deltaDp_v(3)
       -(*mStr)(1,2)*deltaDp_v(1)
       -(*mStr)(2,2)*deltaDp_v(4)
    );
    spEq(2)=.5*((*deltaLp)(0,2)-(*deltaLp)(2,0))-PlSpinChi()/Inityield()*(
        (*mStr)(0,0)*deltaDp_v(5)
       +(*mStr)(0,1)*deltaDp_v(4)
       +(*mStr)(0,2)*deltaDp_v(2)
       -(*mStr)(0,2)*deltaDp_v(0)
       -(*mStr)(1,2)*deltaDp_v(3)
       -(*mStr)(2,2)*deltaDp_v(5)
    );

    // Derivative of plastic spin equation w.r.t. mandel stress
    LINALG::Matrix<3,6> dSpdM;
    dSpdM(0,0)=+deltaDp_v(3);
    dSpdM(0,1)=-deltaDp_v(3);
    dSpdM(0,2)=0.;
    dSpdM(0,3)=+deltaDp_v(1)-deltaDp_v(0);
    dSpdM(0,4)=-deltaDp_v(5);
    dSpdM(0,5)=deltaDp_v(4);
    dSpdM(1,0)=0.;
    dSpdM(1,1)=deltaDp_v(4);
    dSpdM(1,2)=-deltaDp_v(4);
    dSpdM(1,3)=deltaDp_v(5);
    dSpdM(1,4)=deltaDp_v(2)-deltaDp_v(1);
    dSpdM(1,5)=-deltaDp_v(3);
    dSpdM(2,0)=deltaDp_v(5);
    dSpdM(2,1)=0.;
    dSpdM(2,2)=-deltaDp_v(5);
    dSpdM(2,3)=deltaDp_v(4);
    dSpdM(2,4)=-deltaDp_v(3);
    dSpdM(2,5)=deltaDp_v(2)-deltaDp_v(0);
    dSpdM.Scale(-PlSpinChi()/Inityield());

    // derivative of plastic spin equation w.r.t. deltaDp
    LINALG::Matrix<3,6> dSpdDp;
    dSpdDp(0,0)=-(*mStr)(0,1);
    dSpdDp(0,1)=(*mStr)(0,1);
    dSpdDp(0,2)=0.;
    dSpdDp(0,3)=(*mStr)(0,0)-(*mStr)(1,1);
    dSpdDp(0,4)=+(*mStr)(0,2);
    dSpdDp(0,5)=-(*mStr)(1,2);
    dSpdDp(1,0)=0.;
    dSpdDp(1,1)=-(*mStr)(1,2);
    dSpdDp(1,2)=+(*mStr)(1,2);
    dSpdDp(1,3)=-(*mStr)(0,2);
    dSpdDp(1,4)=+(*mStr)(1,1)-(*mStr)(2,2);
    dSpdDp(1,5)=(*mStr)(0,1);
    dSpdDp(2,0)=-(*mStr)(0,2);
    dSpdDp(2,1)=0.;
    dSpdDp(2,2)=+(*mStr)(0,2);
    dSpdDp(2,3)=-(*mStr)(1,2);
    dSpdDp(2,4)=+(*mStr)(0,1);
    dSpdDp(2,5)=(*mStr)(0,0)-(*mStr)(2,2);
    dSpdDp.Scale(-PlSpinChi()/Inityield());

    // derivative of plastic spin equation w.r.t. RCG
    LINALG::Matrix<3,6> dSpdC;
    dSpdC.Multiply(dSpdM,*dMdC);

    // derivative of plastic spin equation w.r.t. deltaLp
    LINALG::Matrix<3,9> dSpddLp;
    dSpddLp.Multiply(dSpdDp,psym);
    dSpddLp.Multiply(1.,dSpdM,dMdLp,1.);
    dSpddLp(0,3)+=.5; dSpddLp(0,6)-=.5;
    dSpddLp(1,4)+=.5; dSpddLp(1,7)-=.5;
    dSpddLp(2,5)+=.5; dSpddLp(2,8)-=.5;

    // combine NCP and plastic spin equations
    for (int i=0; i<6; i++) (*NCP)(i)=NCP_red(i);
    for (int i=6; i<9; i++) (*NCP)(i)=spEq(i-6);

    for (int i=0; i<6; i++)
      for (int j=0; j<6; j++)
        (*dNCPdC)(i,j)=dNCPdC_red(i,j);
    for (int i=6;i<9; i++)
      for (int j=0; j<6; j++)
        (*dNCPdC)(i,j)=dSpdC(i-6,j);

    for (int i=0; i<6; i++)
      for (int j=0; j<9; j++)
        (*dNCPdLp)(i,j)=dNCPdLp_red(i,j);
    for (int i=6; i<9; i++)
      for (int j=0; j<9; j++)
        (*dNCPdLp)(i,j)=dSpddLp(i-6,j);
  }
  // elastic gp
  else
  {
    *elast=true;
  }

  return;
}

void MAT::PlasticElastHyper::UpdateGP(const int gp, const LINALG::Matrix<3,3>* deltaDp)
{
  if (activity_state_[gp]==true)
  {
    // update plastic deformation gradient
    LINALG::Matrix<3,3> tmp;
    tmp.Update(-1.,*deltaDp);
    MatrixExponential3x3(tmp);
    LINALG::Matrix<3,3> fpi_last = last_plastic_defgrd_inverse_[gp];
    last_plastic_defgrd_inverse_[gp].Multiply(fpi_last,tmp);
    // update isotropic hardening
    last_alpha_isotropic_[gp] += delta_alpha_i_[gp];

    // update kinematic hardening
    last_alpha_kinematic_[gp].Update(-.5,*deltaDp,1.);
    last_alpha_kinematic_[gp].UpdateT(-.5,*deltaDp,1.);
  }

  return;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateKinQuantElast(
    const LINALG::Matrix<3,3>* defgrd,
    const LINALG::Matrix<3,3>* deltaLp,
    const int gp,
    LINALG::Matrix<6,1>& Cpi,
    LINALG::Matrix<6,1>& CpiCCpi,
    LINALG::Matrix<6,1>& ircg,
    LINALG::Matrix<3,1>& prinv)
{
  LINALG::Matrix<3,3> tmp;
  LINALG::Matrix<3,3> invpldefgrd;
  LINALG::Matrix<3,3>& InvPlasticDefgrdLast = last_plastic_defgrd_inverse_[gp];
  tmp.Update(-1.,*deltaLp);
  MatrixExponential3x3(tmp);
  invpldefgrd.Multiply(InvPlasticDefgrdLast,tmp);

  // inverse plastic right Cauchy-Green
  LINALG::Matrix<3,3> CpiM;
  CpiM.MultiplyNT(invpldefgrd,invpldefgrd);
  // stress-like Voigt notation
  for (int i=0; i<3; i++) Cpi(i) = CpiM(i,i);
  Cpi(3) = (CpiM(0,1)+CpiM(1,0))/2.;
  Cpi(4) = (CpiM(2,1)+CpiM(1,2))/2.;
  Cpi(5) = (CpiM(0,2)+CpiM(2,0))/2.;

  // inverse RCG
  LINALG::Matrix<3,3> iRCG;
  LINALG::Matrix<3,3> RCG;
  RCG.MultiplyTN(*defgrd,*defgrd);
  iRCG.Invert(RCG);
  // stress-like Voigt notation
  for (int i=0; i<3; i++) ircg(i) = iRCG(i,i);
  ircg(3) = (iRCG(0,1)+iRCG(1,0))/2.;
  ircg(4) = (iRCG(2,1)+iRCG(1,2))/2.;
  ircg(5) = (iRCG(0,2)+iRCG(2,0))/2.;

  // C_p^-1 * C * C_p^-1
  LINALG::Matrix<3,3> CpiCCpiM;
  tmp.Multiply(CpiM,RCG);
  CpiCCpiM.Multiply(tmp,CpiM);
  // stress-like Voigt notation
  for (int i=0; i<3; i++) CpiCCpi(i) = CpiCCpiM(i,i);
  CpiCCpi(3) = (CpiCCpiM(0,1)+CpiCCpiM(1,0))/2.;
  CpiCCpi(4) = (CpiCCpiM(2,1)+CpiCCpiM(1,2))/2.;
  CpiCCpi(5) = (CpiCCpiM(0,2)+CpiCCpiM(2,0))/2.;

  tmp.Multiply(*defgrd,invpldefgrd);
  LINALG::Matrix<3,3> CeM;
  CeM.MultiplyTN(tmp,tmp);
  // elastic right Cauchy-Green in strain-like Voigt notation.
  LINALG::Matrix<6,1> elasticRCGv;
  for (int i=0; i<3; i++)
    elasticRCGv(i)=CeM(i,i);
  elasticRCGv(3) = (CeM(0,1)+CeM(1,0));
  elasticRCGv(4) = (CeM(2,1)+CeM(1,2));
  elasticRCGv(5) = (CeM(0,2)+CeM(2,0));

  // principal invariants of elastic Cauchy-Green strain
  InvariantsPrincipal(prinv,elasticRCGv);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int MAT::PlasticElastHyper::EvaluateKinQuantPlast(
    const LINALG::Matrix<3,3>* defgrd,
    const LINALG::Matrix<3,3>* deltaLp,
    const int gp,
    Teuchos::ParameterList& params,
    LINALG::Matrix<3,3>& invpldefgrd,
    LINALG::Matrix<6,1>& Cpi,
    LINALG::Matrix<6,1>& CpiCCpi,
    LINALG::Matrix<6,1>& ircg,
    LINALG::Matrix<6,1>& Ce,
    LINALG::Matrix<3,3>& CeM,
    LINALG::Matrix<6,1>& Ce2,
    LINALG::Matrix<6,1>& id2V,
    LINALG::Matrix<3,3>& id2,
    LINALG::Matrix<3,3>& CpiC,
    LINALG::Matrix<3,3>& FpiCe,
    LINALG::Matrix<9,1>& CFpiCei,
    LINALG::Matrix<9,1>& CFpi,
    LINALG::Matrix<3,3>& FpiTC,
    LINALG::Matrix<9,1>& CFpiCe,
    LINALG::Matrix<3,3>& CeFpiTC,
    LINALG::Matrix<3,1>& prinv)
{
  id2.Clear();
  id2V.Clear();
  for (int i=0; i<3; i++)
  {
    id2V(i)  = 1.;
    id2(i,i) = 1.;
  }
  LINALG::Matrix<3,3> tmp;
  LINALG::Matrix<3,3> tmp33;
  LINALG::Matrix<3,3>& InvPlasticDefgrdLast = last_plastic_defgrd_inverse_[gp];
  tmp.Update(-1.,*deltaLp);
  MatrixExponential3x3(tmp);
  invpldefgrd.Multiply(InvPlasticDefgrdLast,tmp);

  tmp33.Multiply(*defgrd,invpldefgrd);
  CeM.MultiplyTN(tmp33,tmp33);
  // elastic right Cauchy-Green in strain-like Voigt notation.
  LINALG::Matrix<6,1> elasticRCGv;
  for (int i=0; i<3; i++)
    elasticRCGv(i)=CeM(i,i);
  elasticRCGv(3) = (CeM(0,1)+CeM(1,0));
  elasticRCGv(4) = (CeM(2,1)+CeM(1,2));
  elasticRCGv(5) = (CeM(0,2)+CeM(2,0));
  // elastic right Cauchy-Green in stress-like Voigt notation.
  for (int i=0; i<3; i++) Ce(i) = CeM(i,i);
  Ce(3) = (CeM(0,1)+CeM(1,0))/2.;
  Ce(4) = (CeM(2,1)+CeM(1,2))/2.;
  Ce(5) = (CeM(0,2)+CeM(2,0))/2.;

  // square of elastic right Cauchy-Green in stress-like Voigt notation.
  tmp.Multiply(CeM,CeM);
  for (int i=0; i<3; i++) Ce2(i) = tmp(i,i);
  Ce2(3) = (tmp(0,1)+tmp(1,0))/2.;
  Ce2(4) = (tmp(2,1)+tmp(1,2))/2.;
  Ce2(5) = (tmp(0,2)+tmp(2,0))/2.;

  // principal invariants of elastic Cauchy-Green strain
  InvariantsPrincipal(prinv,elasticRCGv);

  // inverse plastic right Cauchy-Green
  LINALG::Matrix<3,3> CpiM;
  CpiM.MultiplyNT(invpldefgrd,invpldefgrd);
  // stress-like Voigt notation
  for (int i=0; i<3; i++) Cpi(i) = CpiM(i,i);
  Cpi(3) = (CpiM(0,1)+CpiM(1,0))/2.;
  Cpi(4) = (CpiM(2,1)+CpiM(1,2))/2.;
  Cpi(5) = (CpiM(0,2)+CpiM(2,0))/2.;

  // inverse RCG
  LINALG::Matrix<3,3> iRCG;
  LINALG::Matrix<3,3> RCG;
  RCG.MultiplyTN(*defgrd,*defgrd);
  iRCG.Invert(RCG);
  // stress-like Voigt notation
  for (int i=0; i<3; i++) ircg(i) = iRCG(i,i);
  ircg(3) = (iRCG(0,1)+iRCG(1,0))/2.;
  ircg(4) = (iRCG(2,1)+iRCG(1,2))/2.;
  ircg(5) = (iRCG(0,2)+iRCG(2,0))/2.;

  // C_p^-1 * C * C_p^-1
  LINALG::Matrix<3,3> CpiCCpiM;
  tmp33.Multiply(CpiM,RCG);
  CpiCCpiM.Multiply(tmp33,CpiM);
  // stress-like Voigt notation
  for (int i=0; i<3; i++) CpiCCpi(i) = CpiCCpiM(i,i);
  CpiCCpi(3) = (CpiCCpiM(0,1)+CpiCCpiM(1,0))/2.;
  CpiCCpi(4) = (CpiCCpiM(2,1)+CpiCCpiM(1,2))/2.;
  CpiCCpi(5) = (CpiCCpiM(0,2)+CpiCCpiM(2,0))/2.;

  CpiC.Multiply(CpiM,RCG);
  FpiCe.Multiply(invpldefgrd,CeM);

  FpiTC.MultiplyTN(invpldefgrd,RCG);
  CeFpiTC.Multiply(CeM,FpiTC);

  tmp.Multiply(RCG,invpldefgrd);
  Matrix3x3to9x1(tmp,CFpi);
  tmp33.Multiply(tmp,CeM);
  Matrix3x3to9x1(tmp33,CFpiCe);

  double det=CeM.Determinant();
  if (det > -1e-30 and det < 1e-30)
    if (params.isParameter("tolerate_errors"))
      if (params.get<bool>("tolerate_errors")==true)
      {
        params.get<bool>("eval_error")=true;
        return 1;
      }

  tmp.Invert(CeM);
  tmp33.Multiply(invpldefgrd,tmp);
  tmp.Multiply(RCG,tmp33);
  Matrix3x3to9x1(tmp,CFpiCei);

  return 0;
}

/*----------------------------------------------------------------------*
 |  tensor norm from stress-like Voigt-notation             seitz 05/14 |
 *----------------------------------------------------------------------*/
double MAT::PlasticElastHyper::NormStressLike(const LINALG::Matrix<6,1>& stress)
{
  double norm=0;
  for (int i=0; i<3; ++i) norm += stress(i)*stress(i);
  for (int i=3; i<6; ++i) norm += 2.*stress(i)*stress(i);
  norm=sqrt(norm);
  return norm;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateIsotropicPrincElast(
    LINALG::Matrix<6,1>& stressisoprinc,
    LINALG::Matrix<6,6>& cmatisoprinc,
    LINALG::Matrix<6,1> Cpi,
    LINALG::Matrix<6,1> CpiCCpi,
    LINALG::Matrix<6,1> ircg,
    LINALG::Matrix<3,1> prinv,
    LINALG::Matrix<3,1> dPI,
    LINALG::Matrix<6,1> ddPII
    )
{
  // 2nd Piola Kirchhoff stress (according to Holzapfel-Nonlinear Solid Mechanics p. 216)
  // factors
  LINALG::Matrix<3,1> gamma(true);
  gamma(0) = 2.*(dPI(0)+prinv(0)*dPI(1));
  gamma(1) = -2.*dPI(1);
  gamma(2) = 2.*prinv(2)*dPI(2);

  //  // 2nd Piola Kirchhoff stresses
  stressisoprinc.Update(gamma(0), Cpi, 1.0);
  stressisoprinc.Update(gamma(1), CpiCCpi, 1.0);
  stressisoprinc.Update(gamma(2), ircg, 1.0);

  // constitutive tensor according to Holzapfel-Nonlinear Solid Mechanics p. 261)
  // factors
  LINALG::Matrix<8,1> delta(true);
  delta(0) = 4.*(ddPII(0) +2.*prinv(0)*ddPII(5) +dPI(1) +prinv(0)*prinv(0)*ddPII(1));
  delta(1) = -4.*(ddPII(5) +prinv(0)*ddPII(1));
  delta(2) = 4.*(prinv(2)*ddPII(4) +prinv(0)*prinv(2)*ddPII(3));
  delta(3) = 4.*ddPII(1);
  delta(4) = -4.*prinv(2)*ddPII(3);
  delta(5) = 4.*(prinv(2)*dPI(2) +prinv(2)*prinv(2)*ddPII(2));
  delta(6) = -4.*prinv(2)*dPI(2);
  delta(7) = -4.*dPI(1);

  // constitutive tensor
  cmatisoprinc.MultiplyNT(delta(0),Cpi,Cpi,1.);
  cmatisoprinc.MultiplyNT(delta(1),CpiCCpi,ircg,1.);
  cmatisoprinc.MultiplyNT(delta(1),ircg,CpiCCpi,1.);
  cmatisoprinc.MultiplyNT(delta(2),Cpi,ircg,1.);
  cmatisoprinc.MultiplyNT(delta(2),ircg,Cpi,1.);
  cmatisoprinc.MultiplyNT(delta(3),CpiCCpi,CpiCCpi,1.);
  cmatisoprinc.MultiplyNT(delta(4),CpiCCpi,ircg,1.);
  cmatisoprinc.MultiplyNT(delta(4),ircg,CpiCCpi,1.);
  cmatisoprinc.MultiplyNT(delta(5),ircg,ircg,1.);
  AddtoCmatHolzapfelProduct(cmatisoprinc,ircg,delta(6));
  AddtoCmatHolzapfelProduct(cmatisoprinc,Cpi,delta(7));

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::EvaluateIsotropicPrincPlast(
    LINALG::Matrix<6,9>& dPK2dFpinvIsoprinc,
    LINALG::Matrix<3,3>& MandelStressIsoprinc,
    LINALG::Matrix<6,6>& dMdCisoprinc,
    LINALG::Matrix<6,9>& dMdFpinvIsoprinc,
    const LINALG::Matrix<6,1>& Cpi,
    const LINALG::Matrix<6,1>& CpiCCpi,
    const LINALG::Matrix<6,1>& ircg,
    const LINALG::Matrix<6,1>& Ce,
    const LINALG::Matrix<3,3>& CeM,
    const LINALG::Matrix<6,1>& Ce2,
    const LINALG::Matrix<6,1>& id2V,
    const LINALG::Matrix<3,3>& id2,
    const LINALG::Matrix<3,3>& CpiC,
    const LINALG::Matrix<3,3>& FpiCe,
    const LINALG::Matrix<3,3>& Fpi,
    const LINALG::Matrix<9,1>& CFpiCei,
    const LINALG::Matrix<9,1>& CFpi,
    const LINALG::Matrix<3,3>& FpiTC,
    const LINALG::Matrix<9,1>& CFpiCe,
    const LINALG::Matrix<3,3>& CeFpiTC,
    const LINALG::Matrix<3,1>& gamma,
    const LINALG::Matrix<8,1>& delta
    )
{
    // derivative of PK2 w.r.t. inverse plastic deformation gradient
    AddtodPK2dFpinv(dPK2dFpinvIsoprinc,id2,Fpi,gamma(0));
    AddtodPK2dFpinv(dPK2dFpinvIsoprinc,CpiC,Fpi,gamma(1));
    dPK2dFpinvIsoprinc.MultiplyNT(delta(0),Cpi,CFpi,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(1),Cpi,CFpiCe,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(1),CpiCCpi,CFpi,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(2),Cpi,CFpiCei,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(2),ircg,CFpi,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(3),CpiCCpi,CFpiCe,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(4),CpiCCpi,CFpiCei,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(4),ircg,CFpiCe,1.);
    dPK2dFpinvIsoprinc.MultiplyNT(delta(5),ircg,CFpiCei,1.);
    AddtodPK2dFpinv(dPK2dFpinvIsoprinc,id2,FpiCe,0.5*delta(7));

    // Mandel stress
    LINALG::Matrix<6,1> Mv;
    Mv.Update(gamma(0),Ce);
    Mv.Update(gamma(1),Ce2,1.);
    Mv.Update(gamma(2),id2V,1.);
    for (int i=0; i<3; i++) MandelStressIsoprinc(i,i) += Mv(i);
    MandelStressIsoprinc(0,1) += Mv(3);
    MandelStressIsoprinc(1,0) += Mv(3);
    MandelStressIsoprinc(1,2) += Mv(4);
    MandelStressIsoprinc(2,1) += Mv(4);
    MandelStressIsoprinc(0,2) += Mv(5);
    MandelStressIsoprinc(2,0) += Mv(5);

    // derivative of Mandel stress w.r.t. GL
    AddtodMdC_gamma1(dMdCisoprinc,Fpi,gamma(0));
    AddtodMdC_gamma2(dMdCisoprinc,Fpi,FpiCe,gamma(1));
    dMdCisoprinc.MultiplyNT(delta(0),Ce,Cpi,1.);
    dMdCisoprinc.MultiplyNT(delta(1),Ce,CpiCCpi,1.);
    dMdCisoprinc.MultiplyNT(delta(1),Ce2,Cpi,1.);
    dMdCisoprinc.MultiplyNT(delta(2),Ce,ircg,1.);
    dMdCisoprinc.MultiplyNT(delta(2),id2V,Cpi,1.);
    dMdCisoprinc.MultiplyNT(delta(3),Ce2,CpiCCpi,1.);
    dMdCisoprinc.MultiplyNT(delta(4),Ce2,ircg,1.);
    dMdCisoprinc.MultiplyNT(delta(4),id2V,CpiCCpi,1.);
    dMdCisoprinc.MultiplyNT(delta(5),id2V,ircg,1.);

    // derivative of Mandel stress w.r.t. inverse plastic deformation gradient
    AddtodPK2dFpinv(dMdFpinvIsoprinc,FpiTC,id2,gamma(0));
    AddtodPK2dFpinv(dMdFpinvIsoprinc,FpiTC,CeM,gamma(1));
    AddtodPK2dFpinv(dMdFpinvIsoprinc,CeFpiTC,id2,gamma(1));
    dMdFpinvIsoprinc.MultiplyNT(delta(0),Ce,CFpi,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(1),Ce,CFpiCe,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(1),Ce2,CFpi,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(2),Ce,CFpiCei,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(2),id2V,CFpi,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(3),Ce2,CFpiCe,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(4),Ce2,CFpiCei,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(4),id2V,CFpiCe,1.);
    dMdFpinvIsoprinc.MultiplyNT(delta(5),id2V,CFpiCei,1.);

  return ;
}

/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void MAT::PlasticElastHyper::VisNames(std::map<std::string,int>& names)
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1; // scalar
  std::string plastic_zone = "plastic_zone";
  names[plastic_zone] = 1; // scalar

}  // VisNames()


/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool MAT::PlasticElastHyper::VisData(
  const std::string& name,
  std::vector<double>& data,
  int numgp,
  int eleID
  )
{
  if (name == "accumulatedstrain")
  {
    if ((int)data.size()!=1) dserror("size mismatch");
    double tmp=0.;
    for (unsigned gp=0; gp<last_alpha_isotropic_.size(); gp++)
      tmp+= AccumulatedStrain(gp);
    data[0] = tmp/last_alpha_isotropic_.size();
  }
  if (name == "plastic_zone")
  {
    bool plastic_history=false;
    bool curr_active=false;
    if ((int)data.size()!=1) dserror("size mismatch");
    for (unsigned gp=0; gp<last_alpha_isotropic_.size(); gp++)
    {
      if (AccumulatedStrain(gp)!=0.) plastic_history=true;
      if (Active(gp))                curr_active=true;
    }
    data[0] = plastic_history+curr_active;
  }
  return false;

}  // VisData()

/*----------------------------------------------------------------------*
 |                                                          seitz 09/13 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::AddtodPK2dFpinv(LINALG::Matrix<6,9>& dPK2dFpinv,
    LINALG::Matrix<3,3> A, LINALG::Matrix<3,3> B, double fac)
{
  dPK2dFpinv(0,0) += 2 * fac * A(0,0) * B(0,0);
  dPK2dFpinv(0,3) += 2 * fac * A(0,0) * B(0,1);
  dPK2dFpinv(0,5) += 2 * fac * A(0,0) * B(0,2);
  dPK2dFpinv(0,6) += 2 * fac * A(0,1) * B(0,0);
  dPK2dFpinv(0,1) += 2 * fac * A(0,1) * B(0,1);
  dPK2dFpinv(0,4) += 2 * fac * A(0,1) * B(0,2);
  dPK2dFpinv(0,8) += 2 * fac * A(0,2) * B(0,0);
  dPK2dFpinv(0,7) += 2 * fac * A(0,2) * B(0,1);
  dPK2dFpinv(0,2) += 2 * fac * A(0,2) * B(0,2);

  dPK2dFpinv(1,0) += 2 * fac * A(1,0) * B(1,0);
  dPK2dFpinv(1,3) += 2 * fac * A(1,0) * B(1,1);
  dPK2dFpinv(1,5) += 2 * fac * A(1,0) * B(1,2);
  dPK2dFpinv(1,6) += 2 * fac * A(1,1) * B(1,0);
  dPK2dFpinv(1,1) += 2 * fac * A(1,1) * B(1,1);
  dPK2dFpinv(1,4) += 2 * fac * A(1,1) * B(1,2);
  dPK2dFpinv(1,8) += 2 * fac * A(1,2) * B(1,0);
  dPK2dFpinv(1,7) += 2 * fac * A(1,2) * B(1,1);
  dPK2dFpinv(1,2) += 2 * fac * A(1,2) * B(1,2);

  dPK2dFpinv(2,0) += 2 * fac * A(2,0) * B(2,0);
  dPK2dFpinv(2,3) += 2 * fac * A(2,0) * B(2,1);
  dPK2dFpinv(2,5) += 2 * fac * A(2,0) * B(2,2);
  dPK2dFpinv(2,6) += 2 * fac * A(2,1) * B(2,0);
  dPK2dFpinv(2,1) += 2 * fac * A(2,1) * B(2,1);
  dPK2dFpinv(2,4) += 2 * fac * A(2,1) * B(2,2);
  dPK2dFpinv(2,8) += 2 * fac * A(2,2) * B(2,0);
  dPK2dFpinv(2,7) += 2 * fac * A(2,2) * B(2,1);
  dPK2dFpinv(2,2) += 2 * fac * A(2,2) * B(2,2);

  dPK2dFpinv(3,0) += fac * (A(0,0) * B(1,0) + A(1,0) * B(0,0));
  dPK2dFpinv(3,3) += fac * (A(0,0) * B(1,1) + A(1,0) * B(0,1));
  dPK2dFpinv(3,5) += fac * (A(0,0) * B(1,2) + A(1,0) * B(0,2));
  dPK2dFpinv(3,6) += fac * (A(0,1) * B(1,0) + A(1,1) * B(0,0));
  dPK2dFpinv(3,1) += fac * (A(0,1) * B(1,1) + A(1,1) * B(0,1));
  dPK2dFpinv(3,4) += fac * (A(0,1) * B(1,2) + A(1,1) * B(0,2));
  dPK2dFpinv(3,8) += fac * (A(0,2) * B(1,0) + A(1,2) * B(0,0));
  dPK2dFpinv(3,7) += fac * (A(0,2) * B(1,1) + A(1,2) * B(0,1));
  dPK2dFpinv(3,2) += fac * (A(0,2) * B(1,2) + A(1,2) * B(0,2));

  dPK2dFpinv(4,0) += fac * (A(1,0) * B(2,0) + A(2,0) * B(1,0));
  dPK2dFpinv(4,3) += fac * (A(1,0) * B(2,1) + A(2,0) * B(1,1));
  dPK2dFpinv(4,5) += fac * (A(1,0) * B(2,2) + A(2,0) * B(1,2));
  dPK2dFpinv(4,6) += fac * (A(1,1) * B(2,0) + A(2,1) * B(1,0));
  dPK2dFpinv(4,1) += fac * (A(1,1) * B(2,1) + A(2,1) * B(1,1));
  dPK2dFpinv(4,4) += fac * (A(1,1) * B(2,2) + A(2,1) * B(1,2));
  dPK2dFpinv(4,8) += fac * (A(1,2) * B(2,0) + A(2,2) * B(1,0));
  dPK2dFpinv(4,7) += fac * (A(1,2) * B(2,1) + A(2,2) * B(1,1));
  dPK2dFpinv(4,2) += fac * (A(1,2) * B(2,2) + A(2,2) * B(1,2));

  dPK2dFpinv(5,0) += fac * (A(0,0) * B(2,0) + A(2,0) * B(0,0));
  dPK2dFpinv(5,3) += fac * (A(0,0) * B(2,1) + A(2,0) * B(0,1));
  dPK2dFpinv(5,5) += fac * (A(0,0) * B(2,2) + A(2,0) * B(0,2));
  dPK2dFpinv(5,6) += fac * (A(0,1) * B(2,0) + A(2,1) * B(0,0));
  dPK2dFpinv(5,1) += fac * (A(0,1) * B(2,1) + A(2,1) * B(0,1));
  dPK2dFpinv(5,4) += fac * (A(0,1) * B(2,2) + A(2,1) * B(0,2));
  dPK2dFpinv(5,8) += fac * (A(0,2) * B(2,0) + A(2,2) * B(0,0));
  dPK2dFpinv(5,7) += fac * (A(0,2) * B(2,1) + A(2,2) * B(0,1));
  dPK2dFpinv(5,2) += fac * (A(0,2) * B(2,2) + A(2,2) * B(0,2));

  return;
}


/*----------------------------------------------------------------------*
 |                                                          seitz 09/13 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::AddtodMdC_gamma1(LINALG::Matrix<6,6>& dMdC,
    LINALG::Matrix<3,3> A, double fac)
{
  dMdC(0,0) += 2. * fac * A(0,0) * A(0,0);
  dMdC(0,3) += 2. * fac * A(0,0) * A(1,0);
  dMdC(0,5) += 2. * fac * A(0,0) * A(2,0);
  dMdC(0,1) += 2. * fac * A(1,0) * A(1,0);
  dMdC(0,4) += 2. * fac * A(1,0) * A(2,0);
  dMdC(0,2) += 2. * fac * A(2,0) * A(2,0);

  dMdC(1,0) += 2. * fac * A(0,1) * A(0,1);
  dMdC(1,3) += 2. * fac * A(0,1) * A(1,1);
  dMdC(1,5) += 2. * fac * A(0,1) * A(2,1);
  dMdC(1,1) += 2. * fac * A(1,1) * A(1,1);
  dMdC(1,4) += 2. * fac * A(1,1) * A(2,1);
  dMdC(1,2) += 2. * fac * A(2,1) * A(2,1);

  dMdC(2,0) += 2. * fac * A(0,2) * A(0,2);
  dMdC(2,3) += 2. * fac * A(0,2) * A(1,2);
  dMdC(2,5) += 2. * fac * A(0,2) * A(2,2);
  dMdC(2,1) += 2. * fac * A(1,2) * A(1,2);
  dMdC(2,4) += 2. * fac * A(1,2) * A(2,2);
  dMdC(2,2) += 2. * fac * A(2,2) * A(2,2);

  dMdC(3,0) += 2. * fac * A(0,0) * A(0,1);
  dMdC(3,3) += fac * (A(0,0) * A(1,1) + A(0,1) * A(1,0));
  dMdC(3,5) += fac * (A(0,0) * A(2,1) + A(0,1) * A(2,0));
  dMdC(3,1) += 2. * fac * A(1,0) * A(1,1);
  dMdC(3,4) += fac * (A(1,0) * A(2,1) + A(1,1) * A(2,0));
  dMdC(3,2) += 2. * fac * A(2,0) * A(2,1);

  dMdC(4,0) += 2. * fac * A(0,1) * A(0,2);
  dMdC(4,3) += fac * (A(0,1) * A(1,2) + A(0,2) * A(1,1));
  dMdC(4,5) += fac * (A(0,1) * A(2,2) + A(0,2) * A(2,1));
  dMdC(4,1) += 2. * fac * A(1,1) * A(1,2);
  dMdC(4,4) += fac * (A(1,1) * A(2,2) + A(1,2) * A(2,1));
  dMdC(4,2) += 2. * fac * A(2,1) * A(2,2);

  dMdC(5,0) += 2. * fac * A(0,0) * A(0,2);
  dMdC(5,3) += fac * (A(0,0) * A(1,2) + A(0,2) * A(1,0));
  dMdC(5,5) += fac * (A(0,0) * A(2,2) + A(0,2) * A(2,0));
  dMdC(5,1) += 2. * fac * A(1,0) * A(1,2);
  dMdC(5,4) += fac * (A(1,0) * A(2,2) + A(1,2) * A(2,0));
  dMdC(5,2) += 2. * fac * A(2,0) * A(2,2);

  return;
}


/*----------------------------------------------------------------------*
 |                                                          seitz 09/13 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::AddtodMdC_gamma2(LINALG::Matrix<6,6>& dMdC,
    LINALG::Matrix<3,3> A, LINALG::Matrix<3,3> B, double fac)
{
  dMdC(0,0) += 4 * fac * A(0,0) * B(0,0);
  dMdC(0,3) += fac * (2 * A(0,0) * B(1,0) + 2 * A(1,0) * B(0,0));
  dMdC(0,5) += fac * (2 * A(0,0) * B(2,0) + 2 * A(2,0) * B(0,0));
  dMdC(0,1) += 4 * fac * A(1,0) * B(1,0);
  dMdC(0,4) += fac * (2 * A(1,0) * B(2,0) + 2 * A(2,0) * B(1,0));
  dMdC(0,2) += 4 * fac * A(2,0) * B(2,0);

  dMdC(3,0) += fac * (2 * A(0,0) * B(0,1) + 2 * A(0,1) * B(0,0));
  dMdC(3,3) += fac * (A(0,0) * B(1,1) + A(1,0) * B(0,1) + A(1,1) * B(0,0) + A(0,1) * B(1,0));
  dMdC(3,5) += fac * (A(0,0) * B(2,1) + A(2,0) * B(0,1) + A(2,1) * B(0,0) + A(0,1) * B(2,0));
  dMdC(3,1) += fac * (2 * A(1,0) * B(1,1) + 2 * A(1,1) * B(1,0));
  dMdC(3,4) += fac * (A(1,0) * B(2,1) + A(2,0) * B(1,1) + A(2,1) * B(1,0) + A(1,1) * B(2,0));
  dMdC(3,2) += fac * (2 * A(2,0) * B(2,1) + 2 * A(2,1) * B(2,0));

  dMdC(5,0) += fac * (2 * A(0,0) * B(0,2) + 2 * A(0,2) * B(0,0));
  dMdC(5,3) += fac * (A(0,0) * B(1,2) + A(1,0) * B(0,2) + A(1,2) * B(0,0) + A(0,2) * B(1,0));
  dMdC(5,5) += fac * (A(0,0) * B(2,2) + A(2,0) * B(0,2) + A(2,2) * B(0,0) + A(0,2) * B(2,0));
  dMdC(5,1) += fac * (2 * A(1,0) * B(1,2) + 2 * A(1,2) * B(1,0));
  dMdC(5,4) += fac * (A(1,0) * B(2,2) + A(2,0) * B(1,2) + A(2,2) * B(1,0) + A(1,2) * B(2,0));
  dMdC(5,2) += fac * (2 * A(2,0) * B(2,2) + 2 * A(2,2) * B(2,0));

  dMdC(1,0) += 4 * fac * A(0,1) * B(0,1);
  dMdC(1,3) += fac * (2 * A(0,1) * B(1,1) + 2 * A(1,1) * B(0,1));
  dMdC(1,5) += fac * (2 * A(0,1) * B(2,1) + 2 * A(2,1) * B(0,1));
  dMdC(1,1) += 4 * fac * A(1,1) * B(1,1);
  dMdC(1,4) += fac * (2 * A(1,1) * B(2,1) + 2 * A(2,1) * B(1,1));
  dMdC(1,2) += 4 * fac * A(2,1) * B(2,1);

  dMdC(4,0) += fac * (2 * A(0,1) * B(0,2) + 2 * A(0,2) * B(0,1));
  dMdC(4,3) += fac * (A(0,1) * B(1,2) + A(1,1) * B(0,2) + A(1,2) * B(0,1) + A(0,2) * B(1,1));
  dMdC(4,5) += fac * (A(0,1) * B(2,2) + A(2,1) * B(0,2) + A(2,2) * B(0,1) + A(0,2) * B(2,1));
  dMdC(4,1) += fac * (2 * A(1,1) * B(1,2) + 2 * A(1,2) * B(1,1));
  dMdC(4,4) += fac * (A(1,1) * B(2,2) + A(2,1) * B(1,2) + A(2,2) * B(1,1) + A(1,2) * B(2,1));
  dMdC(4,2) += fac * (2 * A(2,1) * B(2,2) + 2 * A(2,2) * B(2,1));

  dMdC(2,0) += 4 * fac * A(0,2) * B(0,2);
  dMdC(2,3) += fac * (2 * A(0,2) * B(1,2) + 2 * A(1,2) * B(0,2));
  dMdC(2,5) += fac * (2 * A(0,2) * B(2,2) + 2 * A(2,2) * B(0,2));
  dMdC(2,1) += 4 * fac * A(1,2) * B(1,2);
  dMdC(2,4) += fac * (2 * A(1,2) * B(2,2) + 2 * A(2,2) * B(1,2));
  dMdC(2,2) += 4 * fac * A(2,2) * B(2,2);

  return;
}

/*----------------------------------------------------------------------*
 |                                                          seitz 09/13 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::Matrix3x3to9x1(LINALG::Matrix<3,3> A, LINALG::Matrix<9,1>& Out)
{
  Out(0) = A(0,0);
  Out(1) = A(1,1);
  Out(2) = A(2,2);
  Out(3) = A(0,1);
  Out(4) = A(1,2);
  Out(5) = A(0,2);
  Out(6) = A(1,0);
  Out(7) = A(2,1);
  Out(8) = A(2,0);
  return;
}

/*----------------------------------------------------------------------*
 |  matrix exponential                                      seitz 07/13 |
 *----------------------------------------------------------------------*/
void MAT::PlasticElastHyper::MatrixExponential3x3( LINALG::Matrix<3,3>& MatrixInOut )
{
  double Norm=MatrixInOut.Norm2();
  // direct calculation for zero-matrix
  if (Norm==0.)
  {
    MatrixInOut.Clear();
    for (int i=0; i<3; i++)
      MatrixInOut(i,i)=1.;
    return;
  }

  // Calculation of matrix exponential via power series. This is usually
  // faster than by polar decomposition for matrices are close to zero.
  // For small plastic increments this is the case
  LINALG::Matrix<3,3> In(MatrixInOut);
  int n=0;
  int facn=1;
  MatrixInOut.Clear();
  for (int i=0; i<3; i++)
    MatrixInOut(i,i)=1.;
  LINALG::Matrix<3,3> tmp(MatrixInOut);
  LINALG::Matrix<3,3> tmp2(MatrixInOut);
  while (n<50 && tmp.Norm2()/facn>1.e-32)
  {
    n++;
    facn*=n;
    tmp.Multiply(tmp2,In);
    tmp2=tmp;
    MatrixInOut.Update(1./facn,tmp,1.);
  }
  if (n==50) dserror("matrix exponential unconverged in %i steps",n);

  return;
}

/*---------------------------------------------------------------------------*
 |  matrix exponential derivative of a symmetric matrix          seitz 07/13 |
 *---------------------------------------------------------------------------*/
void MAT::PlasticElastHyper::MatrixExponentialDerivativeSym3x3(const LINALG::Matrix<3,3> MatrixIn, LINALG::Matrix<6,6>& MatrixExpDeriv)
{
  double norm=MatrixIn.Norm2();

  LINALG::Matrix<6,6> id4sharp(true);
  for (int i=0; i<3; i++) id4sharp(i,i) = 1.0;
  for (int i=3; i<6; i++) id4sharp(i,i) = 0.5;

  // direct calculation for zero-matrix
  if (norm==0.)
  {
    MatrixExpDeriv = id4sharp;
    return;
  }

  if(norm<0.3)
  {
    // see Souza-Neto: Computational Methods for plasticity, Box B.2.
    int nmax=0;
    int nIter=0;
    int nfac=1;
    LINALG::Matrix<3,3> tmp1;
    LINALG::Matrix<3,3> tmp2(true);
    for (int i=0; i<3; i++) tmp2(i,i)=1.;

    // all needed powers of X
    std::vector<LINALG::Matrix<3,3> > Xn;
    Xn.resize(0);
    Xn.push_back(tmp2);

    // all needed factorials
    std::vector<int> fac;
    fac.resize(0);
    fac.push_back(nfac);

    // compute nmax and Xn
    while (nIter<50 && tmp2.Norm2()/nfac>1.e-32)
    {
      nIter++;
      nfac *= nIter;
      fac.push_back(nfac);
      tmp1.Multiply(tmp2,MatrixIn);
      Xn.push_back(tmp1);
      tmp2=tmp1;
    }
    if (nIter==50) dserror("matrix exponential unconverged in %i steps",nIter);
    nmax=nIter;

    // compose derivative of matrix exponential (symmetric Voigt-notation)
    MatrixExpDeriv.Clear();
    for (int n=1; n<=nmax; n++)
    {
      for (int m=1; m<=n/2; m++)
        AddToSymMatrixExponentialDeriv(1./fac[n],Xn.at(m-1),Xn.at(n-m),MatrixExpDeriv);
      if (n%2==1)
        AddToSymMatrixExponentialDeriv(0.5/fac[n],Xn.at((n-1)/2),Xn.at((n-1)/2),MatrixExpDeriv);
    }
  }
  else
  {
    double EWtolerance=1.e-12;

    LINALG::Matrix<3,3> EV(MatrixIn);
    LINALG::Matrix<3,3> EW;
    LINALG::SYEV(EV,EW,EV);

    LINALG::Matrix<3,1> vec1;
    LINALG::Matrix<3,1> vec2;
    LINALG::Matrix<3,3> tmp1;
    LINALG::Matrix<3,3> tmp2;

    MatrixExpDeriv.Clear();
    // souza eq. (A.52)
    // note: EW stored in ascending order

    //  d X^2 / d X  =  1/2 * (  delta_jk X_lj + delta_il X_kj
    //                         + delta_jl X_ik + delta_kj X_il )
    //
    // y_i = log(x_i)
    // dy_i / dx_j = delta_ij 1/x_i

    LINALG::Matrix<3,3> id2(true);
    for (int i=0; i<3; i++)
      id2(i,i) =1.0 ;
    //  // --------------------------------- switch by number of equal eigenvalues

    if (abs(EW(0,0)-EW(1,1))<EWtolerance && abs(EW(1,1)-EW(2,2))<EWtolerance ) // ------------------ x_a == x_b == x_c
    {
      // calculate derivative
      MatrixExpDeriv = id4sharp;
      MatrixExpDeriv.Scale(exp(EW(0,0)));
    }

    else if ( ( abs(EW(0,0)-EW(1,1))<EWtolerance && abs(EW(1,1)-EW(2,2))>EWtolerance ) ||
        ( abs(EW(0,0)-EW(1,1))>EWtolerance && abs(EW(1,1)-EW(2,2))<EWtolerance )  ) // ---- x_a != x_b == x_c or x_a == x_b != x_c
    {
      // factors
      double s1=0.0;
      double s2=0.0;
      double s3=0.0;
      double s4=0.0;
      double s5=0.0;
      double s6=0.0;

      int a=0;
      int c=0;

      // switch which two EW are equal
      if ( abs(EW(0,0)-EW(1,1))<EWtolerance && abs(EW(1,1)-EW(2,2))>EWtolerance ) // ----------------------- x_a == x_b != x_c
      {
        a=2;
        c=0;
      }
      else if ( abs(EW(0,0)-EW(1,1))>EWtolerance && abs(EW(1,1)-EW(2,2))<EWtolerance) // ------------------ x_a != x_b == x_c
      {
        a=0;
        c=2;
      }
      else
        dserror("you should not be here");

      // in souza eq. (A.53):
      s1 = ( exp(EW(a,a)) - exp(EW(c,c)) ) / ( pow( EW(a,a) - EW(c,c),2.0 ) )  -  exp(EW(c,c)) / (EW(a,a)-EW(c,c));
      s2 = 2.0 * EW(c,c) * (exp(EW(a,a))-exp(EW(c,c)))/(pow(EW(a,a)-EW(c,c),2.0)) - (EW(a,a)+EW(c,c))/(EW(a,a)-EW(c,c)) * exp(EW(c,c));
      s3 = 2.0 * (exp(EW(a,a))-exp(EW(c,c)))/(pow(EW(a,a)-EW(c,c),3.0)) - (exp(EW(a,a)) + exp(EW(c,c)))/(pow(EW(a,a)-EW(c,c),2.0));
      s4 = EW(c,c)*s3;
      s5 = s4;
      s6 = EW(c,c)*EW(c,c) * s3;

      // calculate derivative
      MAT::AddToCmatDerivTensorSquare(MatrixExpDeriv,s1,MatrixIn,1.);
      MatrixExpDeriv.Update(-s2,id4sharp,1.);
      MAT::ElastSymTensorMultiply(MatrixExpDeriv,-1.*s3,MatrixIn,MatrixIn,1.);
      MAT::ElastSymTensorMultiply(MatrixExpDeriv,s4,MatrixIn,id2,1.);
      MAT::ElastSymTensorMultiply(MatrixExpDeriv,s5,id2,MatrixIn,1.);
      MAT::ElastSymTensorMultiply(MatrixExpDeriv,-s6,id2,id2,1.);
    }

    else if ( abs(EW(0,0)-EW(1,1))>EWtolerance && abs(EW(1,1)-EW(2,2))>EWtolerance ) // ----------------- x_a != x_b != x_c
    {
      for (int a=0; a<3; a++) // loop over all eigenvalues
      {
        int b = (a+1)%3;
        int c = (a+2)%3;

        LINALG::Matrix<3,1> ea;
        LINALG::Matrix<3,1> eb;
        LINALG::Matrix<3,1> ec;
        for (int i=0; i<3; i++)
        {
          ea(i) = EV(i,a);
          eb(i) = EV(i,b);
          ec(i) = EV(i,c);
        }
        LINALG::Matrix<3,3> Ea;
        Ea.MultiplyNT(ea,ea);
        LINALG::Matrix<3,3> Eb;
        Eb.MultiplyNT(eb,eb);
        LINALG::Matrix<3,3> Ec;
        Ec.MultiplyNT(ec,ec);

        double fac = exp(EW(a,a)) / ( (EW(a,a)-EW(b,b)) * (EW(a,a)-EW(c,c)) );

        // + d X^2 / d X
        MAT::AddToCmatDerivTensorSquare(MatrixExpDeriv,fac,MatrixIn,1.);

        // - (x_b + x_c) I_s
        MatrixExpDeriv.Update(-1.*(EW(b,b)+EW(c,c))*fac,id4sharp,1.);

        // - [(x_a - x_b) + (x_a - x_c)] E_a \dyad E_a
        MAT::ElastSymTensorMultiply(MatrixExpDeriv,-1.*fac * ( (EW(a,a)-EW(b,b)) + (EW(a,a)-EW(c,c)) ),Ea,Ea,1.);


        // - (x_b - x_c) (E_b \dyad E_b)
        MAT::ElastSymTensorMultiply(MatrixExpDeriv,-1.*fac * (EW(b,b) - EW(c,c)),Eb,Eb,1.);

        // + (x_b - x_c) (E_c \dyad E_c)
        MAT::ElastSymTensorMultiply(MatrixExpDeriv,fac * (EW(b,b) - EW(c,c)),Ec,Ec,1.);

        // dy / dx_a E_a \dyad E_a
        MAT::ElastSymTensorMultiply(MatrixExpDeriv,exp(EW(a,a)),Ea,Ea,1.);
      } // end loop over all eigenvalues

    }

    else dserror("you should not be here.");
  }
  return;
}

/*---------------------------------------------------------------------------*
 |  matrix exponential derivative of a symmetric matrix          seitz 09/13 |
 *---------------------------------------------------------------------------*/
void MAT::PlasticElastHyper::MatrixExponentialDerivative3x3(const LINALG::Matrix<3,3> MatrixIn, LINALG::Matrix<9,9>& MatrixExpDeriv)
{
  // see Souza-Neto: Computational Methods for plasticity, Box B.2.
  int nmax=0;
  int nIter=0;
  int nfac=1;
  LINALG::Matrix<3,3> tmp1;
  LINALG::Matrix<3,3> tmp2(true);
  for (int i=0; i<3; i++) tmp2(i,i)=1.;

  // all needed powers of X
  std::vector<LINALG::Matrix<3,3> > Xn;
  Xn.resize(0);
  Xn.push_back(tmp2);

  // all needed factorials
  std::vector<int> fac;
  fac.resize(0);
  fac.push_back(nfac);

  // compute nmax and Xn
  while (nIter<50 && tmp2.Norm2()/nfac>1.e-32)
  {
    nIter++;
    nfac *= nIter;
    fac.push_back(nfac);
    tmp1.Multiply(tmp2,MatrixIn);
    Xn.push_back(tmp1);
    tmp2=tmp1;
  }
  if (nIter==50) dserror("matrix exponential unconverged in %i steps",nIter);
  nmax=nIter;

  // compose derivative of matrix exponential (non-symmetric Voigt-notation)
  MatrixExpDeriv.Clear();
  for (int n=1; n<=nmax; n++)
    for (int m=1; m<=n; m++)
      AddToMatrixExponentialDeriv(1./fac[n],Xn.at(m-1),Xn.at(n-m),MatrixExpDeriv);

  return;
}

/*---------------------------------------------------------------------------*
 |  add terms for matrix exponential derivative of a symmetric matrix        |
 | via power series                                              seitz 08/13 |
 *---------------------------------------------------------------------------*/
void MAT::PlasticElastHyper::AddToSymMatrixExponentialDeriv(const double fac,
    const LINALG::Matrix<3,3> A,const LINALG::Matrix<3,3> B, LINALG::Matrix<6,6>& Dexp)
{
  Dexp(0,0) += 2. * fac * A(0,0) * B(0,0);
  Dexp(0,3) += fac * ( A(0,0) * B(0,1) + A(0,1) * B(0,0));
  Dexp(0,5) += fac * ( A(0,0) * B(0,2) + A(0,2) * B(0,0));
  Dexp(0,1) += 2. * fac * A(0,1) * B(0,1);
  Dexp(0,4) += fac * ( A(0,1) * B(0,2) + A(0,2) * B(0,1));
  Dexp(0,2) += 2. * fac * A(0,2) * B(0,2);

  Dexp(3,0) += fac * (A(0,0) * B(0,1) + A(0,1) * B(0,0));
  Dexp(3,3) += 0.5 * fac * (A(0,0) * B(1,1) + 2. * A(0,1) * B(0,1) + A(1,1) * B(0,0));
  Dexp(3,5) += 0.5 * fac * (A(0,0) * B(1,2) + A(0,2) * B(0,1) + A(1,2) * B(0,0) + A(0,1) * B(0,2));
  Dexp(3,1) +=  fac * ( A(0,1) * B(1,1) +  A(1,1) * B(0,1));
  Dexp(3,4) += 0.5 * fac * (A(0,1) * B(1,2) + A(0,2) * B(1,1) + A(1,2) * B(0,1) + A(1,1) * B(0,2));
  Dexp(3,2) += fac * ( A(0,2) * B(1,2) + A(1,2) * B(0,2));

  Dexp(5,0) += fac * (A(0,0) * B(0,2) + A(0,2) * B(0,0));
  Dexp(5,3) += 0.5 * fac * (A(0,0) * B(1,2) + A(0,2) * B(0,1) + A(1,2) * B(0,0) + A(0,1) * B(0,2));
  Dexp(5,5) += 0.5 * fac * (A(0,0) * B(2,2) + 2. * A(0,2) * B(0,2) + A(2,2) * B(0,0));
  Dexp(5,1) += fac * (A(0,1) * B(1,2) + A(1,2) * B(0,1));
  Dexp(5,4) += 0.5 * fac * (A(0,1) * B(2,2) + A(0,2) * B(1,2) + A(2,2) * B(0,1) + A(1,2) * B(0,2));
  Dexp(5,2) += fac * ( A(0,2) * B(2,2) +  A(2,2) * B(0,2));

  Dexp(1,0) += 2. * fac * A(0,1) * B(0,1);
  Dexp(1,3) +=  fac * ( A(0,1) * B(1,1) + A(1,1) * B(0,1));
  Dexp(1,5) +=  fac * ( A(0,1) * B(1,2) +  A(1,2) * B(0,1));
  Dexp(1,1) += 2. * fac * A(1,1) * B(1,1);
  Dexp(1,4) +=  fac * ( A(1,1) * B(1,2) + A(1,2) * B(1,1));
  Dexp(1,2) += 2. * fac * A(1,2) * B(1,2);

  Dexp(4,0) +=  fac * ( A(0,1) * B(0,2) +  A(0,2) * B(0,1));
  Dexp(4,3) += 0.5 * fac * (A(0,1) * B(1,2) + A(0,2) * B(1,1) + A(1,2) * B(0,1) + A(1,1) * B(0,2));
  Dexp(4,5) += 0.5 * fac * (A(0,1) * B(2,2) + A(0,2) * B(1,2) + A(2,2) * B(0,1) + A(1,2) * B(0,2));
  Dexp(4,1) +=  fac * ( A(1,1) * B(1,2) + A(1,2) * B(1,1));
  Dexp(4,4) += 0.5 * fac * (A(1,1) * B(2,2) + 2. * A(1,2) * B(1,2) + A(2,2) * B(1,1));
  Dexp(4,2) += fac * ( A(1,2) * B(2,2) + A(2,2) * B(1,2));

  Dexp(2,0) += 2. * fac * A(0,2) * B(0,2);
  Dexp(2,3) += fac * (A(0,2) * B(1,2) +  A(1,2) * B(0,2));
  Dexp(2,5) += fac * ( A(0,2) * B(2,2) +  A(2,2) * B(0,2));
  Dexp(2,1) += 2. * fac * A(1,2) * B(1,2);
  Dexp(2,4) += fac * ( A(1,2) * B(2,2) + A(2,2) * B(1,2));
  Dexp(2,2) += 2. * fac * A(2,2) * B(2,2);

  return;
}

/*---------------------------------------------------------------------------*
 |  add terms for matrix exponential derivative of a symmetric matrix        |
 | via power series                                              seitz 09/13 |
 *---------------------------------------------------------------------------*/
void MAT::PlasticElastHyper::AddToMatrixExponentialDeriv(const double fac,
    const LINALG::Matrix<3,3> A,const LINALG::Matrix<3,3> B, LINALG::Matrix<9,9>& Dexp)
{
  Dexp(0,0) += fac * A(0,0) * B(0,0);
  Dexp(0,3) += fac * A(0,0) * B(1,0);
  Dexp(0,5) += fac * A(0,0) * B(2,0);
  Dexp(0,6) += fac * A(0,1) * B(0,0);
  Dexp(0,1) += fac * A(0,1) * B(1,0);
  Dexp(0,4) += fac * A(0,1) * B(2,0);
  Dexp(0,8) += fac * A(0,2) * B(0,0);
  Dexp(0,7) += fac * A(0,2) * B(1,0);
  Dexp(0,2) += fac * A(0,2) * B(2,0);

  Dexp(3,0) += fac * A(0,0) * B(0,1);
  Dexp(3,3) += fac * A(0,0) * B(1,1);
  Dexp(3,5) += fac * A(0,0) * B(2,1);
  Dexp(3,6) += fac * A(0,1) * B(0,1);
  Dexp(3,1) += fac * A(0,1) * B(1,1);
  Dexp(3,4) += fac * A(0,1) * B(2,1);
  Dexp(3,8) += fac * A(0,2) * B(0,1);
  Dexp(3,7) += fac * A(0,2) * B(1,1);
  Dexp(3,2) += fac * A(0,2) * B(2,1);

  Dexp(5,0) += fac * A(0,0) * B(0,2);
  Dexp(5,3) += fac * A(0,0) * B(1,2);
  Dexp(5,5) += fac * A(0,0) * B(2,2);
  Dexp(5,6) += fac * A(0,1) * B(0,2);
  Dexp(5,1) += fac * A(0,1) * B(1,2);
  Dexp(5,4) += fac * A(0,1) * B(2,2);
  Dexp(5,8) += fac * A(0,2) * B(0,2);
  Dexp(5,7) += fac * A(0,2) * B(1,2);
  Dexp(5,2) += fac * A(0,2) * B(2,2);

  Dexp(6,0) += fac * A(1,0) * B(0,0);
  Dexp(6,3) += fac * A(1,0) * B(1,0);
  Dexp(6,5) += fac * A(1,0) * B(2,0);
  Dexp(6,6) += fac * A(1,1) * B(0,0);
  Dexp(6,1) += fac * A(1,1) * B(1,0);
  Dexp(6,4) += fac * A(1,1) * B(2,0);
  Dexp(6,8) += fac * A(1,2) * B(0,0);
  Dexp(6,7) += fac * A(1,2) * B(1,0);
  Dexp(6,2) += fac * A(1,2) * B(2,0);

  Dexp(1,0) += fac * A(1,0) * B(0,1);
  Dexp(1,3) += fac * A(1,0) * B(1,1);
  Dexp(1,5) += fac * A(1,0) * B(2,1);
  Dexp(1,6) += fac * A(1,1) * B(0,1);
  Dexp(1,1) += fac * A(1,1) * B(1,1);
  Dexp(1,4) += fac * A(1,1) * B(2,1);
  Dexp(1,8) += fac * A(1,2) * B(0,1);
  Dexp(1,7) += fac * A(1,2) * B(1,1);
  Dexp(1,2) += fac * A(1,2) * B(2,1);

  Dexp(4,0) += fac * A(1,0) * B(0,2);
  Dexp(4,3) += fac * A(1,0) * B(1,2);
  Dexp(4,5) += fac * A(1,0) * B(2,2);
  Dexp(4,6) += fac * A(1,1) * B(0,2);
  Dexp(4,1) += fac * A(1,1) * B(1,2);
  Dexp(4,4) += fac * A(1,1) * B(2,2);
  Dexp(4,8) += fac * A(1,2) * B(0,2);
  Dexp(4,7) += fac * A(1,2) * B(1,2);
  Dexp(4,2) += fac * A(1,2) * B(2,2);

  Dexp(8,0) += fac * A(2,0) * B(0,0);
  Dexp(8,3) += fac * A(2,0) * B(1,0);
  Dexp(8,5) += fac * A(2,0) * B(2,0);
  Dexp(8,6) += fac * A(2,1) * B(0,0);
  Dexp(8,1) += fac * A(2,1) * B(1,0);
  Dexp(8,4) += fac * A(2,1) * B(2,0);
  Dexp(8,8) += fac * A(2,2) * B(0,0);
  Dexp(8,7) += fac * A(2,2) * B(1,0);
  Dexp(8,2) += fac * A(2,2) * B(2,0);

  Dexp(7,0) += fac * A(2,0) * B(0,1);
  Dexp(7,3) += fac * A(2,0) * B(1,1);
  Dexp(7,5) += fac * A(2,0) * B(2,1);
  Dexp(7,6) += fac * A(2,1) * B(0,1);
  Dexp(7,1) += fac * A(2,1) * B(1,1);
  Dexp(7,4) += fac * A(2,1) * B(2,1);
  Dexp(7,8) += fac * A(2,2) * B(0,1);
  Dexp(7,7) += fac * A(2,2) * B(1,1);
  Dexp(7,2) += fac * A(2,2) * B(2,1);

  Dexp(2,0) += fac * A(2,0) * B(0,2);
  Dexp(2,3) += fac * A(2,0) * B(1,2);
  Dexp(2,5) += fac * A(2,0) * B(2,2);
  Dexp(2,6) += fac * A(2,1) * B(0,2);
  Dexp(2,1) += fac * A(2,1) * B(1,2);
  Dexp(2,4) += fac * A(2,1) * B(2,2);
  Dexp(2,8) += fac * A(2,2) * B(0,2);
  Dexp(2,7) += fac * A(2,2) * B(1,2);
  Dexp(2,2) += fac * A(2,2) * B(2,2);
}
