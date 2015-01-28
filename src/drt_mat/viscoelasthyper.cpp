/*----------------------------------------------------------------------*/
/*!
\file viscoelasthyper.cpp
\brief
This file contains the viscohyperelastic material.
This model can be applied to any hyperelastic law of the Elasthyper toolbox.
The viscos part is rate-dependent and summed up with the hyperelastic laws
to build a viscohyperelastic strain energy function.
(Description of hysteresis not added jet)

The input line should read
MAT 0   MAT_ViscoElastHyper   NUMMAT 0 MATIDS  DENS 0

<pre>
Maintainer: Anna Birzle
            birzle@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15255
</pre>
*/

/*----------------------------------------------------------------------*/
#include "viscoelasthyper.H"
#include "../drt_lib/standardtypes_cpp.H"
#include "../drt_matelast/elast_summand.H"
#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_linedefinition.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_mat/material_service.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::ViscoElastHyper::ViscoElastHyper(Teuchos::RCP<MAT::PAR::Material> matdata)
  : MAT::PAR::ElastHyper(matdata)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<MAT::Material> MAT::PAR::ViscoElastHyper::CreateMaterial()
{
  return Teuchos::rcp(new MAT::ViscoElastHyper(this));
}


MAT::ViscoElastHyperType MAT::ViscoElastHyperType::instance_;


DRT::ParObject* MAT::ViscoElastHyperType::Create( const std::vector<char> & data )
{
  MAT::ViscoElastHyper* elhy = new MAT::ViscoElastHyper();
  elhy->Unpack(data);

  return elhy;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::ViscoElastHyper::ViscoElastHyper()
  : MAT::ElastHyper()
{
  // history data 09/13
  isinitvis_=false;
  histmodrcgcurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
  histmodrcglast_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
  histstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histartstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histartstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::ViscoElastHyper::ViscoElastHyper(MAT::PAR::ViscoElastHyper* params)
  : MAT::ElastHyper(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::Pack(DRT::PackBuffer& data) const
{
//    MAT::ElastHyper::Pack(data);
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // matid
  int matid = -1;
  if (params_ != NULL) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data,matid);
  AddtoPack(data,isoprinc_);
  AddtoPack(data,isomod_);
  AddtoPack(data,anisoprinc_);
  AddtoPack(data,anisomod_);
  AddtoPack(data,isomodvisco_);
  AddtoPack(data,viscogenmax_);


  if (params_ != NULL) // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (unsigned int p=0; p<potsum_.size(); ++p)
    {
     potsum_[p]->PackSummand(data);
    }
  }

  //  pack history data 09/13
    int histsize;
    if (!Initialized())
    {
      histsize=0;
    }
    else
    {
      histsize = histmodrcglast_->size();
    }
    AddtoPack(data,histsize);  // Length of history vector(s)
    for (int var = 0; var < histsize; ++var)
    {
      AddtoPack(data,histmodrcglast_->at(var));
      AddtoPack(data,histstresslast_->at(var));
      AddtoPack(data,histartstresslast_->at(var));
    }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::Unpack(const std::vector<char>& data)
{
  // make sure we have a pristine material
  params_ = NULL;
  potsum_.clear();

  isoprinc_ = false;
  isomod_ = false;
  anisoprinc_ = false;
  anisomod_ = false;
  isomodvisco_ = false;
  viscogenmax_ = false;


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
        params_ = static_cast<MAT::PAR::ViscoElastHyper*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(), MaterialType());
    }
  }

  int isoprinc;
  int isomod;
  int anisoprinc;
  int anisomod;
  int isomodvisco;
  int viscogenmax;

  ExtractfromPack(position,data,isoprinc);
  ExtractfromPack(position,data,isomod);
  ExtractfromPack(position,data,anisoprinc);
  ExtractfromPack(position,data,anisomod);
  ExtractfromPack(position,data,isomodvisco);
  ExtractfromPack(position,data,viscogenmax);

  if (isoprinc != 0) isoprinc_ = true;
  if (isomod != 0) isomod_ = true;
  if (anisoprinc != 0) anisoprinc_ = true;
  if (anisomod != 0) anisomod_ = true;
  if (isomodvisco != 0) isomodvisco_ = true;
  if (viscogenmax != 0) viscogenmax_ =true;

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

    // history data 09/13
    isinitvis_ = true;
    int histsize;
    ExtractfromPack(position,data,histsize);

    if (histsize == 0) isinitvis_=false;

    histmodrcgcurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
    histmodrcglast_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
    histstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
    histstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
    histartstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
    histstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);

    for (int var=0; var<histsize; var+=1)
    {
      LINALG::Matrix<6,1> tmp(true);
      histmodrcgcurr_->push_back(tmp);
      histstresscurr_->push_back(tmp);
      histartstresscurr_->push_back(tmp);

      ExtractfromPack(position,data,tmp);
      histmodrcglast_->push_back(tmp);
      ExtractfromPack(position,data,tmp);
      histstresslast_->push_back(tmp);
      ExtractfromPack(position,data,tmp);
      histartstresslast_->push_back(tmp);
    }

    // in the postprocessing mode, we do not unpack everything we have packed
    // -> position check cannot be done in this case
    if (position != data.size())
      dserror("Mismatch in size of data %d <-> %d",data.size(),position);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::Setup(int numgp, DRT::INPUT::LineDefinition* linedef)
{
  MAT::ElastHyper::Setup(numgp, linedef);

  // Initialise/allocate history variables 09/13
  histmodrcgcurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
  histmodrcglast_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
  histstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histartstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histartstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);

  const LINALG::Matrix<6,1> emptyvec(true);
  histmodrcgcurr_->resize(numgp);
  histmodrcglast_->resize(numgp);
  histstresscurr_->resize(numgp);
  histstresslast_->resize(numgp);
  histartstresscurr_->resize(numgp);
  histartstresslast_->resize(numgp);

  for (int j=0; j<numgp; ++j)
  {
    histmodrcgcurr_->at(j) = emptyvec;
    histmodrcglast_->at(j) = emptyvec;
    histstresscurr_->at(j) = emptyvec;
    histstresslast_->at(j) = emptyvec;
    histartstresscurr_->at(j) = emptyvec;
    histartstresslast_->at(j) = emptyvec;

  }
  isinitvis_ = true;

  return;
}

/*------------------------------------------------------------------------------------------*
|  Setup internal stress variables - special for the inverse analysis (public)         09/13|
*-------------------------------------------------------------------------------------------*/
void MAT::ViscoElastHyper::ResetAll(const int numgp)
{
  // Initialise/allocate history variables 09/13

  histmodrcgcurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
  histmodrcglast_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
  histstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histartstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histartstresslast_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);

  const LINALG::Matrix<6,1> emptyvec(true);
  histmodrcgcurr_->resize(numgp);
  histmodrcglast_->resize(numgp);
  histstresscurr_->resize(numgp);
  histstresslast_->resize(numgp);
  histartstresscurr_->resize(numgp);
  histartstresslast_->resize(numgp);

  for (int j=0; j<numgp; ++j)
  {
    histmodrcgcurr_->at(j) = emptyvec;
    histmodrcglast_->at(j) = emptyvec;
    histstresscurr_->at(j) = emptyvec;
    histstresslast_->at(j) = emptyvec;
    histartstresscurr_->at(j) = emptyvec;
    histartstresslast_->at(j) = emptyvec;
  }
  isinitvis_ = true;

  return ;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::Update()
{
  MAT::ElastHyper::Update();

  // Update history values 09/13
  histmodrcglast_=histmodrcgcurr_;
  histstresslast_=histstresscurr_;
  histartstresslast_=histartstresscurr_;

  const LINALG::Matrix<6,1> emptyvec(true);
  histmodrcgcurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<6,1> >);
  histstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);
  histartstresscurr_=Teuchos::rcp(new std::vector<LINALG::Matrix<NUM_STRESS_3D,1> >);

  const int numgp=histmodrcglast_->size();
  histmodrcgcurr_->resize(numgp);
  histstresscurr_->resize(numgp);
  histartstresscurr_->resize(numgp);

  for (int j=0; j<numgp; ++j)
  {
    histmodrcgcurr_->at(j) = emptyvec;
    histstresscurr_->at(j) = emptyvec;
    histartstresscurr_->at(j) = emptyvec;
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::Evaluate(const LINALG::Matrix<3,3>* defgrd,
                               const LINALG::Matrix<6,1>* glstrain,
                               Teuchos::ParameterList& params,
                               LINALG::Matrix<6,1>* stress,
                               LINALG::Matrix<6,6>* cmat,
                               const int eleGID)
{

  LINALG::Matrix<6,1> id2(true) ;
  LINALG::Matrix<6,1> rcg(true) ;
  LINALG::Matrix<6,1> modrcg(true) ;
  LINALG::Matrix<6,1> scg(true) ;
  LINALG::Matrix<6,1> icg(true) ;
  LINALG::Matrix<6,6> id4(true) ;
  LINALG::Matrix<6,6> id4sharp(true) ;

  LINALG::Matrix<3,1> prinv(true);
  LINALG::Matrix<3,1> modinv(true);
  LINALG::Matrix<7,1> modrateinv (true);

  LINALG::Matrix<3,1> dPI(true);
  LINALG::Matrix<6,1> ddPII(true);

  LINALG::Matrix<6,1> modrcgrate(true);
  // for extension: LINALG::Matrix<6,1> modicgrate(true);
  LINALG::Matrix<8,1> modmy(true);
  LINALG::Matrix<33,1> modxi(true);

  const double tau(true);
  const double beta(true);

  EvaluateKinQuant(*glstrain,id2,scg,rcg,icg,id4,id4sharp,prinv);
  EvaluateInvariantDerivatives(prinv,dPI,ddPII);

  if (isomodvisco_)
  {
    // calculate modified invariants
    InvariantsModified(modinv,prinv);
    // calculate viscous quantities
    EvaluateKinQuantVis(rcg,modrcg,icg,params,prinv,modrcgrate,modrateinv);
    EvaluateMyXi(modinv,modmy,modxi,modrateinv,params);
  }

  // blank resulting quantities
  // ... even if it is an implicit law that cmat is zero upon input
  stress->Clear();
  cmat->Clear();

  // build stress response and elasticity tensor
  LINALG::Matrix<NUM_STRESS_3D,1> stressiso(true) ;
  LINALG::Matrix<NUM_STRESS_3D,NUM_STRESS_3D> cmatiso(true) ;

  EvaluateIsotropicStressCmat(stressiso,cmatiso,scg,id2,icg,id4sharp,prinv,dPI,ddPII);

  stress->Update(1.0, stressiso, 1.0);
  cmat->Update(1.0,cmatiso,1.0);

  // add viscous part (at the moment just exists for decoupled problems)
  if (isomod_)
  {
    if(isomodvisco_)
    {
      // add viscous part
      LINALG::Matrix<NUM_STRESS_3D,1> stressisomodisovisco(true);
      LINALG::Matrix<NUM_STRESS_3D,NUM_STRESS_3D> cmatisomodisovisco(true);
      LINALG::Matrix<NUM_STRESS_3D,1> stressisomodvolvisco(true) ;
      LINALG::Matrix<NUM_STRESS_3D,NUM_STRESS_3D> cmatisomodvolvisco(true) ;
      EvaluateIsoModVisco(stressisomodisovisco,stressisomodvolvisco,cmatisomodisovisco,cmatisomodvolvisco,prinv,modinv,modmy,modxi,rcg,id2,icg,id4,modrcgrate);
      stress->Update(1.0, stressisomodisovisco, 1.0);
      stress->Update(1.0, stressisomodvolvisco, 1.0);
      cmat->Update(1.0,cmatisomodisovisco,1.0);
      cmat->Update(1.0,cmatisomodvolvisco,1.0);
    }
  }

  // add contribution of viscogenmax-material
  if (viscogenmax_)
  {
    LINALG::Matrix<NUM_STRESS_3D,1> Q(true); // artificial viscous stress
    LINALG::Matrix<NUM_STRESS_3D,NUM_STRESS_3D> cmatq(true) ;
    EvaluateViscoGenMax(*stress,*cmat,Q,cmatq,tau,beta,params);
    stress->Update(1.0,Q,1.0);
    cmat->Update(1.0,cmatq,1.0);
  }


  /*----------------------------------------------------------------------*/
  // coefficients in principal stretches
  const bool havecoeffstrpr = HaveCoefficientsStretchesPrincipal();
  const bool havecoeffstrmod = HaveCoefficientsStretchesModified();
  if (havecoeffstrpr or havecoeffstrmod) {
    ResponseStretches(*cmat,*stress,rcg,havecoeffstrpr,havecoeffstrmod);
  }

  /*----------------------------------------------------------------------*/
  //Do all the anisotropic stuff!
  if (anisoprinc_)
  {
      LINALG::Matrix<NUM_STRESS_3D,1> stressanisoprinc(true) ;
      LINALG::Matrix<NUM_STRESS_3D,NUM_STRESS_3D> cmatanisoprinc(true) ;
      EvaluateAnisotropicPrinc(stressanisoprinc,cmatanisoprinc,rcg,params);
      stress->Update(1.0, stressanisoprinc, 1.0);
      cmat->Update(1.0, cmatanisoprinc, 1.0);
  }

  if (anisomod_)
  {
      LINALG::Matrix<NUM_STRESS_3D,1> stressanisomod(true) ;
      LINALG::Matrix<NUM_STRESS_3D,NUM_STRESS_3D> cmatanisomod(true) ;
      EvaluateAnisotropicMod(stressanisomod,cmatanisomod,rcg,icg,prinv);
      stress->Update(1.0, stressanisomod, 1.0);
      cmat->Update(1.0, cmatanisomod, 1.0);
  }

  return ;
}

/*----------------------------------------------------------------------*/
/* Evaluate Quantities for Viscos Part                            09/13 */
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::EvaluateKinQuantVis(
    LINALG::Matrix<6,1> rcg,
    LINALG::Matrix<6,1>& modrcg,
    LINALG::Matrix<6,1> icg,
    Teuchos::ParameterList& params,
    LINALG::Matrix<3,1> prinv,
    LINALG::Matrix<6,1>& modrcgrate,
    LINALG::Matrix<7,1>& modrateinv
    )

{
  // modrcg : \overline{C} = J^{-\frac{2}{3}} C
  const double modscale = std::pow(prinv(2),-1./3.);
  modrcg.Update(modscale,rcg);

  // get gauss point number of this element
  const int gp = params.get<int>("gp",-1);
  // get time algorithmic parameters
  double dt = params.get<double>("delta time");

  // read history
  LINALG::Matrix<6,1> modrcglast (histmodrcglast_->at(gp));

  // rate of Cauchy-Green Tensor \dot{C} = \frac {overline{C}^n - \overline{C}^{n-1}} {\Delta t}
  // REMARK: strain-like 6-Voigt vector
  modrcgrate.Update(1.0,modrcg,1.0);
  modrcgrate.Update(-1.0,modrcglast,1.0);
  modrcgrate.Scale(1/dt);

  // in the first time step, set modrcgrate to zero (--> first time step is just hyperelastic, not viscos)
  const LINALG::Matrix<6,1> emptyvec(true);
  if(modrcglast == emptyvec)
    {
    modrcgrate=emptyvec;
    }

  // Update history of Cauchy-Green Tensor
  histmodrcgcurr_->at(gp) = modrcg;

  // Second Invariant of modrcgrate \bar{J}_2 = \frac{1}{2} \tr (\dot{\overline{C^2}}
  modrateinv(1) = 0.5*( modrcgrate(0)*modrcgrate(0) + modrcgrate(1)*modrcgrate(1) + modrcgrate(2)*modrcgrate(2)
      + .5*modrcgrate(3)*modrcgrate(3) + .5*modrcgrate(4)*modrcgrate(4) + .5*modrcgrate(5)*modrcgrate(5) );


  // For further extension of material law (not necassary at the moment)
  /*
  // necassary transfer variable: LINALG::Matrix<6,1>& modicgrate
  // \overline{J}_3 = determinant of modified rate of right Cauchy-Green-Tensor
  modrateinv(2) = modrcgrate(0)*modrcgrate(1)*modrcgrate(2)
      + 0.25 * modrcgrate(3)*modrcgrate(4)*modrcgrate(5)
      - 0.25 * modrcgrate(1)*modrcgrate(5)*modrcgrate(5)
      - 0.25 * modrcgrate(2)*modrcgrate(3)*modrcgrate(3)
      - 0.25 * modrcgrate(0)*modrcgrate(4)*modrcgrate(4);

  // invert modified rate of right Cauchy-Green tensor
  // REMARK: stress-like 6-Voigt vector
  {
    modicgrate(0) = ( modrcgrate(1)*modrcgrate(2) - 0.25*modrcgrate(4)*modrcgrate(4) ) / modrateinv(2);
    modicgrate(1) = ( modrcgrate(0)*modrcgrate(2) - 0.25*modrcgrate(5)*modrcgrate(5) ) / modrateinv(2);
    modicgrate(2) = ( modrcgrate(0)*modrcgrate(1) - 0.25*modrcgrate(3)*modrcgrate(3) ) / modrateinv(2);
    modicgrate(3) = ( 0.25*modrcgrate(5)*modrcgrate(4) - 0.5*modrcgrate(3)*modrcgrate(2) ) / modrateinv(2);
    modicgrate(4) = ( 0.25*modrcgrate(3)*modrcgrate(5) - 0.5*modrcgrate(0)*modrcgrate(4) ) / modrateinv(2);
    modicgrate(5) = ( 0.25*modrcgrate(3)*modrcgrate(4) - 0.5*modrcgrate(5)*modrcgrate(1) ) / modrateinv(2);
  }
  */

}

/*----------------------------------------------------------------------*/
/* Evaluate Factors for Viscos Quantities                         09/13 */
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::EvaluateMyXi(
    LINALG::Matrix<3,1> modinv,
    LINALG::Matrix<8,1>& modmy,
    LINALG::Matrix<33,1>& modxi,
    LINALG::Matrix<7,1>& modrateinv,
    Teuchos::ParameterList& params
    )
{
  // modified coefficients
  // loop map of associated potential summands
  for (unsigned int p=0; p<potsum_.size(); ++p)
  {
    potsum_[p]->AddCoefficientsViscoModified(modinv,modmy,modxi,modrateinv,params);
  }
}

/*----------------------------------------------------------------------*/
/* Calculates the stress and constitutive tensor for viscos part  09/13 */
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::EvaluateIsoModVisco(
    LINALG::Matrix<6,1>& stressisomodisovisco,
    LINALG::Matrix<6,1>& stressisomodvolvisco,
    LINALG::Matrix<6,6>& cmatisomodisovisco,
    LINALG::Matrix<6,6>& cmatisomodvolvisco,
    LINALG::Matrix<3,1> prinv,
    LINALG::Matrix<3,1> modinv,
    LINALG::Matrix<8,1> modmy,
    LINALG::Matrix<33,1> modxi,
    LINALG::Matrix<6,1> rcg,
    LINALG::Matrix<6,1> id2,
    LINALG::Matrix<6,1> icg,
    LINALG::Matrix<6,6> id4,
    LINALG::Matrix<6,1> modrcgrate
    )
{
  // define necessary variables
  const double modscale = std::pow(prinv(2),-1./3.);

  // 2nd Piola Kirchhoff stresses

  // isochoric contribution
  LINALG::Matrix<6,1> modstress(true);
  modstress.Update(modmy(1), id2);
  modstress.Update(modmy(2), modrcgrate, 1.0);
  // build 4-tensor for projection as 6x6 tensor
  LINALG::Matrix<6,6> Projection;
  Projection.MultiplyNT(1./3., icg, rcg);
  Projection.Update(1.0, id4, -1.0);
  // isochoric stress
  stressisomodisovisco.MultiplyNN(modscale,Projection,modstress,1.0);

  // volumetric contribution:
  // with visco_isoratedep: no volumetric part added --> always 0


  // Constitutive Tensor

  //isochoric contribution
  // modified constitutive tensor
  LINALG::Matrix<6,6> modcmat(true);
  LINALG::Matrix<6,6> modcmat2(true);
  // contribution:  Id \otimes \overline{\dot{C}} + \overline{\dot{C}} \otimes Id
  modcmat.MultiplyNT(modxi(1), id2, modrcgrate);
  modcmat.MultiplyNT(modxi(1), modrcgrate, id2, 1.0);
  // contribution: Id4
  modcmat.Update(modxi(2), id4, 1.0);
  //scaling
  modcmat.Scale(std::pow(modinv(2),-4./3.));
  //contribution: P:\overline{C}:P
  modcmat2.MultiplyNN(Projection,modcmat);
  cmatisomodisovisco.MultiplyNT(1.0,modcmat2,Projection,1.0);
  // contribution: 2/3*Tr(J^(-2/3)modstress) (Cinv \odot Cinv - 1/3 Cinv \otimes Cinv)
  modcmat.Clear();
  modcmat.MultiplyNT(-1.0/3.0,icg,icg);
  AddtoCmatHolzapfelProduct(modcmat, icg, 1.0);
  LINALG::Matrix<1,1> tracemat;
  tracemat.MultiplyTN(2./3.*std::pow(modinv(2),-2./3.),modstress,rcg);
  cmatisomodisovisco.Update(tracemat(0,0),modcmat,1.0);
  //contribution: -2/3 (Cinv \otimes S_iso^v + S_iso^v \otimes Cinv)
  cmatisomodisovisco.MultiplyNT(-2./3.,icg,stressisomodisovisco,1.0);
  cmatisomodisovisco.MultiplyNT(-2./3.,stressisomodisovisco,icg,1.0);

  // volumetric contribution:
  // with visco_isoratedep: no volumetric part added --> always 0

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ViscoElastHyper::EvaluateViscoGenMax(
    LINALG::Matrix<6,1> stress,
    LINALG::Matrix<6,6> cmat,
    LINALG::Matrix<6,1>& Q,
    LINALG::Matrix<6,6>& cmatq,
    double tau,
    double beta,
    Teuchos::ParameterList& params
    )
{
  // read material parameters of viscogenmax-material
  for (unsigned int p=0; p<potsum_.size(); ++p)
  {
    potsum_[p]->ReadMaterialParameters(tau,beta);
  }

  //initialize scalars
  double lambdascalar1(true);
  double lambdascalar2(true);
  double deltascalar(true);
  double theta = 0.5;

  // get theta of global time integration scheme to use it here
  // if global time integration scheme is not ONESTEPTHETA, theta is by default = 0.5 (abirzle 09/14)
  std::string dyntype = DRT::Problem::Instance()->StructuralDynamicParams().get<std::string>("DYNAMICTYP");
  if(dyntype == "OneStepTheta")
    theta = DRT::Problem::Instance()->StructuralDynamicParams().sublist("ONESTEPTHETA").get<double>("THETA");

  // get time algorithmic parameters
  // NOTE: dt can be zero (in restart of STI) for Generalized Maxwell model
  // there is no special treatment required. Adaptation for Kelvin-Voigt were necessary.
  double dt = params.get<double>("delta time"); // TIMESTEP in the .dat file

  // evaluate scalars to compute
  // Q^(n+1) = tau/(tau+theta*dt) [(tau-dt+theta*dt)/tau Q + beta(S^(n+1) - S^n)]
  lambdascalar1=tau/(tau + theta*dt);
  lambdascalar2=(tau - dt + theta*dt)/tau;

  // factor to calculate visco stiffness matrix from elastic stiffness matrix
  // old Version: scalarvisco = 1+beta_isoprinc*exp(-dt/(2*tau_isoprinc));//+alpha1*tau/(tau+theta*dt);
  // Alines' version: scalarvisco = beta_isoprinc*exp(-dt/(2*tau_isoprinc));//+alpha1*tau/(tau+theta*dt);
  // Scalar consistent to derivation of Q with one-step-theta-schema (abirzle 09/14):
  deltascalar = beta*lambdascalar1;

  // read history
  const int gp = params.get<int>("gp",-1);
  if (gp == -1) dserror("no Gauss point number provided in material");
  LINALG::Matrix<NUM_STRESS_3D,1> S_n (histstresslast_->at(gp));
  LINALG::Matrix<NUM_STRESS_3D,1> Q_n (histartstresslast_->at(gp));

  // calculate artificial viscous stresses Q
  Q.Update(lambdascalar2,Q_n,1.0);
  Q.Update(beta,stress,1.0);
  Q.Update(-beta,S_n,1.0);
  Q.Scale(lambdascalar1);  // Q^(n+1) = lambdascalar1* [lambdascalar2* Q + beta*(S^(n+1) - S^n)]

  // update history
  histstresscurr_->at(gp) = stress;
  histartstresscurr_->at(gp) = Q;

  // viscous constitutive tensor
  // contribution : Cmat_vis = Cmat_inf*deltascalar
  cmatq.Update(deltascalar,cmat,1.0);

  return;
}
