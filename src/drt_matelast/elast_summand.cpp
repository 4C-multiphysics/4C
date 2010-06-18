/*----------------------------------------------------------------------*/
/*!
\file elast_summand.cpp

\brief Interface class for complex materials at Gauss points

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "../drt_lib/drt_globalproblem.H"

#include "../drt_mat/matpar_parameter.H"
#include "../drt_mat/material.H"
#include "../drt_mat/elasthyper.H"
#include "elast_couplogneohooke.H"
#include "elast_coupblatzko.H"
#include "elast_isoneohooke.H"
#include "elast_varisoneohooke.H"
#include "elast_isoyeoh.H"
#include "elast_isoquad.H"
#include "elast_varisoquad.H"
#include "elast_isocub.H"
#include "elast_varisocub.H"
#include "elast_isoexpo.H"
#include "elast_varisoexpo.H"
#include "elast_isomooneyrivlin.H"
#include "elast_volsussmanbathe.H"
#include "elast_volpenalty.H"
#include "elast_vologden.H"
#include "elast_coupanisoexpotwo.H"
#include "elast_coupanisoneohooketwo.H"
#include "elast_coupvarga.H"
#include "elast_isovarga.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_bundle.H"


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<MAT::ELASTIC::Summand> MAT::ELASTIC::Summand::Factory(int matnum)
{
  // for the sake of safety
  // in post-process mode we do not have any instance of DRT::Problem
  if (DRT::Problem::NumInstances() <= 0)
    dserror("Sorry dude, cannot work out problem instance.");

  // retrieve problem instance to read from
  const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
  // retrieve validated input line of material ID in question
  Teuchos::RCP<MAT::PAR::Material> curmat = DRT::Problem::Instance(probinst)->Materials()->ById(matnum);

  // check what was read
  //cout << *curmat << endl;

  switch (curmat->Type())
  {
  case INPAR::MAT::mes_couplogneohooke:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::CoupLogNeoHooke(curmat));
    MAT::ELASTIC::PAR::CoupLogNeoHooke* params = static_cast<MAT::ELASTIC::PAR::CoupLogNeoHooke*>(curmat->Parameter());
    return Teuchos::rcp(new CoupLogNeoHooke(params));
  }
  case INPAR::MAT::mes_coupblatzko:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::CoupBlatzKo(curmat));
    MAT::ELASTIC::PAR::CoupBlatzKo* params = static_cast<MAT::ELASTIC::PAR::CoupBlatzKo*>(curmat->Parameter());
    return Teuchos::rcp(new CoupBlatzKo(params));
  }
  case INPAR::MAT::mes_isoneohooke:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::IsoNeoHooke(curmat));
    MAT::ELASTIC::PAR::IsoNeoHooke* params = static_cast<MAT::ELASTIC::PAR::IsoNeoHooke*>(curmat->Parameter());
    return Teuchos::rcp(new IsoNeoHooke(params));
  }
  case INPAR::MAT::mes_varisoneohooke:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::VarIsoNeoHooke(curmat));
    MAT::ELASTIC::PAR::VarIsoNeoHooke* params = static_cast<MAT::ELASTIC::PAR::VarIsoNeoHooke*>(curmat->Parameter());
    return Teuchos::rcp(new VarIsoNeoHooke(params));
  }
  case INPAR::MAT::mes_isoyeoh:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::IsoYeoh(curmat));
    MAT::ELASTIC::PAR::IsoYeoh* params = static_cast<MAT::ELASTIC::PAR::IsoYeoh*>(curmat->Parameter());
    return Teuchos::rcp(new IsoYeoh(params));
  }
  case INPAR::MAT::mes_isoquad:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::IsoQuad(curmat));
    MAT::ELASTIC::PAR::IsoQuad* params = static_cast<MAT::ELASTIC::PAR::IsoQuad*>(curmat->Parameter());
    return Teuchos::rcp(new IsoQuad(params));
  }
  case INPAR::MAT::mes_varisoquad:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::VarIsoQuad(curmat));
    MAT::ELASTIC::PAR::VarIsoQuad* params = static_cast<MAT::ELASTIC::PAR::VarIsoQuad*>(curmat->Parameter());
    return Teuchos::rcp(new VarIsoQuad(params));
  }
  case INPAR::MAT::mes_isocub:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::IsoCub(curmat));
    MAT::ELASTIC::PAR::IsoCub* params = static_cast<MAT::ELASTIC::PAR::IsoCub*>(curmat->Parameter());
    return Teuchos::rcp(new IsoCub(params));
  }
  case INPAR::MAT::mes_varisocub:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::VarIsoCub(curmat));
    MAT::ELASTIC::PAR::VarIsoCub* params = static_cast<MAT::ELASTIC::PAR::VarIsoCub*>(curmat->Parameter());
    return Teuchos::rcp(new VarIsoCub(params));
  }
  case INPAR::MAT::mes_isoexpo:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::IsoExpo(curmat));
    MAT::ELASTIC::PAR::IsoExpo* params = static_cast<MAT::ELASTIC::PAR::IsoExpo*>(curmat->Parameter());
    return Teuchos::rcp(new IsoExpo(params));
  }
  case INPAR::MAT::mes_varisoexpo:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::VarIsoExpo(curmat));
    MAT::ELASTIC::PAR::VarIsoExpo* params = static_cast<MAT::ELASTIC::PAR::VarIsoExpo*>(curmat->Parameter());
    return Teuchos::rcp(new VarIsoExpo(params));
  }
  case INPAR::MAT::mes_isomooneyrivlin:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::IsoMooneyRivlin(curmat));
    MAT::ELASTIC::PAR::IsoMooneyRivlin* params = static_cast<MAT::ELASTIC::PAR::IsoMooneyRivlin*>(curmat->Parameter());
    return Teuchos::rcp(new IsoMooneyRivlin(params));
  }
  case INPAR::MAT::mes_volsussmanbathe:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::VolSussmanBathe(curmat));
    MAT::ELASTIC::PAR::VolSussmanBathe* params = static_cast<MAT::ELASTIC::PAR::VolSussmanBathe*>(curmat->Parameter());
    return Teuchos::rcp(new VolSussmanBathe(params));
  }
  case INPAR::MAT::mes_volpenalty:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::VolPenalty(curmat));
    MAT::ELASTIC::PAR::VolPenalty* params = static_cast<MAT::ELASTIC::PAR::VolPenalty*>(curmat->Parameter());
    return Teuchos::rcp(new VolPenalty(params));
  }
  case INPAR::MAT::mes_vologden:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::VolOgden(curmat));
    MAT::ELASTIC::PAR::VolOgden* params = static_cast<MAT::ELASTIC::PAR::VolOgden*>(curmat->Parameter());
    return Teuchos::rcp(new VolOgden(params));
  }
  case INPAR::MAT::mes_coupanisoexpotwo:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoExpoTwo(curmat));
    MAT::ELASTIC::PAR::CoupAnisoExpoTwo* params = static_cast<MAT::ELASTIC::PAR::CoupAnisoExpoTwo*>(curmat->Parameter());
    return Teuchos::rcp(new CoupAnisoExpoTwo(params));
  }
  case INPAR::MAT::mes_coupanisoneohooketwo:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::CoupAnisoNeoHookeTwo(curmat));
    MAT::ELASTIC::PAR::CoupAnisoNeoHookeTwo* params = static_cast<MAT::ELASTIC::PAR::CoupAnisoNeoHookeTwo*>(curmat->Parameter());
    return Teuchos::rcp(new CoupAnisoNeoHookeTwo(params));
  }
  case INPAR::MAT::mes_coupvarga:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::CoupVarga(curmat));
    MAT::ELASTIC::PAR::CoupVarga* params = static_cast<MAT::ELASTIC::PAR::CoupVarga*>(curmat->Parameter());
    return Teuchos::rcp(new CoupVarga(params));
  }
  case INPAR::MAT::mes_isovarga:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::ELASTIC::PAR::IsoVarga(curmat));
    MAT::ELASTIC::PAR::IsoVarga* params = static_cast<MAT::ELASTIC::PAR::IsoVarga*>(curmat->Parameter());
    return Teuchos::rcp(new IsoVarga(params));
  }
  default:
    dserror("cannot deal with type %d", curmat->Type());
  }

  return Teuchos::null;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ELASTIC::Summand::AddShearMod(
  bool& haveshearmod,
  double& shearmod
  ) const
{
  haveshearmod = haveshearmod or false;
  return;
}


#endif
