/*-----------------------------------------------------------*/
/*! \file

\brief factory for time integration base strategy and data container


\level 3

*/
/*-----------------------------------------------------------*/


#include "str_timint_factory.H"

#include "globalproblem.H"
#include "inpar_structure.H"
#include "prestress_service.H"

#include <Teuchos_ParameterList.hpp>

// supported time integrator
#include "str_timint_implicit.H"
#include "str_timint_explicit.H"
#include "str_timint_loca_continuation.H"

// supported data containers
#include "str_timint_basedatasdyn.H"
#include "strtimada.H"
#include "xstr_xstructure_structure_state.H"


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::Factory::Factory()
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TimAda> STR::TIMINT::Factory::BuildAdaptiveWrapper(
    const Teuchos::ParameterList& ioflags,       //!< input-output-flags
    const Teuchos::ParameterList& sdyn,          //!< structural dynamic flags
    const Teuchos::ParameterList& xparams,       //!< extra flags
    const Teuchos::ParameterList& taflags,       //!< adaptive input flags
    Teuchos::RCP<STR::TIMINT::Base> ti_strategy  //!< marching time integrator
) const
{
  Teuchos::RCP<STR::TimAda> adaintegrator = Teuchos::null;

  /*
  // auxiliary time integrator
  switch (DRT::INPUT::IntegralValue<INPAR::STR::TimAdaKind>(taflags,"KIND"))
  {

  case INPAR::STR::timada_kind_none :
    // No adaptivity in time
    adaintegrator = Teuchos::null;
    break;

  case INPAR::STR::timada_kind_zienxie :
    // Zienkiewicz-Xie error indicator for generalised-alpha
    adaintegrator = Teuchos::rcp(new STR::TimAdaZienXie(sdyn, taflags, timeintegrator));
    break;

  case INPAR::STR::timada_kind_ab2 :
    // Adams-Bashforth 2nd order
    adaintegrator = Teuchos::rcp(new STR::TimAdaJoint<STR::TimIntAB2>(
        ioflags, sdyn, xparams, taflags, timeintegrator));
    break;

  case INPAR::STR::timada_kind_expleuler :
    // Adams-Bashforth 2nd order
    adaintegrator = Teuchos::rcp(new STR::TimAdaJoint<STR::TimIntExplEuler>(
        ioflags, sdyn, xparams, taflags, timeintegrator));
    break;

  case INPAR::STR::timada_kind_centraldiff :
    // Adams-Bashforth 2nd order
    adaintegrator = Teuchos::rcp(new STR::TimAdaJoint<STR::TimIntCentrDiff>(
        ioflags, sdyn, xparams, taflags, timeintegrator));
    break;

  default :
    dserror("Auxiliary adaptive time integrator is not available.");
    break;

  }
  */
  // return the auxiliary integrator
  return adaintegrator;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::Factory::BuildStrategy(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::Base> ti_strategy = Teuchos::null;

  const enum INPAR::STR::IntegrationStrategy intstrat =
      DRT::INPUT::IntegralValue<INPAR::STR::IntegrationStrategy>(sdyn, "INT_STRATEGY");

  switch (intstrat)
  {
    case INPAR::STR::int_standard:
    {
      // Check first if a implicit integration strategy is desired
      ti_strategy = BuildImplicitStrategy(sdyn);
      // If there was no suitable implicit time integrator check for the
      // explicit case
      if (ti_strategy.is_null()) ti_strategy = BuildExplicitStrategy(sdyn);
      break;
    }
    case INPAR::STR::int_loca:
      ti_strategy = Teuchos::rcp(new STR::TIMINT::LOCAContinuation());
      break;
    default:
      dserror("Unknown integration strategy!");
      break;
  }

  return ti_strategy;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::Factory::BuildImplicitStrategy(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::Base> ti_strategy = Teuchos::null;

  // get the dynamic type
  const enum INPAR::STR::DynamicType dyntype =
      DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP");

  if (::UTILS::PRESTRESS::IsAny() or dyntype == INPAR::STR::dyna_statics or  // dynamic type
      dyntype == INPAR::STR::dyna_genalpha or dyntype == INPAR::STR::dyna_genalpha_liegroup or
      dyntype == INPAR::STR::dyna_onesteptheta or dyntype == INPAR::STR::dyna_gemm)
    ti_strategy = Teuchos::rcp(new STR::TIMINT::Implicit());

  return ti_strategy;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::Factory::BuildExplicitStrategy(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::Base> ti_strategy = Teuchos::null;

  // what's the current problem type?
  ProblemType probtype = DRT::Problem::Instance()->GetProblemType();

  if (probtype == ProblemType::fsi or probtype == ProblemType::fsi_redmodels or
      probtype == ProblemType::fsi_lung or probtype == ProblemType::gas_fsi or
      probtype == ProblemType::ac_fsi or probtype == ProblemType::biofilm_fsi or
      probtype == ProblemType::thermo_fsi)
    dserror("No explicit time integration with fsi");

  const enum INPAR::STR::DynamicType dyntype =
      DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP");

  if (dyntype == INPAR::STR::dyna_expleuler or dyntype == INPAR::STR::dyna_centrdiff or
      dyntype == INPAR::STR::dyna_ab2)
    //    ti_strategy = Teuchos::rcp(new STR::TIMINT::Explicit());
    dserror("Explicit time integration scheme is not yet implemented!");

  return ti_strategy;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataSDyn> STR::TIMINT::Factory::BuildDataSDyn(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::BaseDataSDyn> sdyndata_ptr = Teuchos::null;

  const enum INPAR::STR::DynamicType dyntype =
      DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP");

  switch (dyntype)
  {
    case INPAR::STR::dyna_genalpha:
    case INPAR::STR::dyna_genalpha_liegroup:
      sdyndata_ptr = Teuchos::rcp(new STR::TIMINT::GenAlphaDataSDyn());
      break;
    case INPAR::STR::dyna_onesteptheta:
      sdyndata_ptr = Teuchos::rcp(new STR::TIMINT::OneStepThetaDataSDyn());
      break;
    default:
      sdyndata_ptr = Teuchos::rcp(new STR::TIMINT::BaseDataSDyn());
      break;
  }

  return sdyndata_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> STR::TIMINT::Factory::BuildDataGlobalState() const
{
  return Teuchos::rcp(new STR::TIMINT::BaseDataGlobalState());
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::BuildStrategy(const Teuchos::ParameterList& sdyn)
{
  Factory factory;
  return factory.BuildStrategy(sdyn);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TimAda> STR::TIMINT::BuildAdaptiveWrapper(
    const Teuchos::ParameterList& ioflags,       //!< input-output-flags
    const Teuchos::ParameterList& sdyn,          //!< structural dynamic flags
    const Teuchos::ParameterList& xparams,       //!< extra flags
    const Teuchos::ParameterList& taflags,       //!< adaptive input flags
    Teuchos::RCP<STR::TIMINT::Base> ti_strategy  //!< marching time integrator
)
{
  Factory factory;
  return factory.BuildAdaptiveWrapper(ioflags, sdyn, xparams, taflags, ti_strategy);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataSDyn> STR::TIMINT::BuildDataSDyn(
    const Teuchos::ParameterList& sdyn)
{
  Factory factory;
  return factory.BuildDataSDyn(sdyn);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> STR::TIMINT::BuildDataGlobalState()
{
  Factory factory;
  return factory.BuildDataGlobalState();
}
