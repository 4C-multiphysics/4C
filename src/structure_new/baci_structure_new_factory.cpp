/*-----------------------------------------------------------*/
/*! \file

\brief factory for time integrator


\level 3

*/
/*-----------------------------------------------------------*/

#include "baci_structure_new_factory.H"

#include "baci_lib_prestress_service.H"
#include "baci_structure_new_dbc.H"
#include "baci_structure_new_impl_gemm.H"
#include "baci_structure_new_impl_genalpha.H"
#include "baci_structure_new_impl_genalpha_liegroup.H"
#include "baci_structure_new_impl_ost.H"        // derived from ost
#include "baci_structure_new_impl_prestress.H"  // derived from statics
#include "baci_structure_new_impl_statics.H"
#include "baci_structure_new_timint_base.H"
#include "baci_utils_exceptions.H"

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::Factory::Factory()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::Integrator> STR::Factory::BuildIntegrator(
    const STR::TIMINT::BaseDataSDyn& datasdyn) const
{
  Teuchos::RCP<STR::Integrator> int_ptr = Teuchos::null;
  int_ptr = BuildImplicitIntegrator(datasdyn);
  if (int_ptr.is_null()) int_ptr = BuildExplicitIntegrator(datasdyn);
  if (int_ptr.is_null()) dserror("We could not find a suitable dynamic integrator (Dynamic Type).");

  return int_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::Integrator> STR::Factory::BuildImplicitIntegrator(
    const STR::TIMINT::BaseDataSDyn& datasdyn) const
{
  Teuchos::RCP<STR::IMPLICIT::Generic> impl_int_ptr = Teuchos::null;

  const enum INPAR::STR::DynamicType& dyntype = datasdyn.GetDynamicType();
  const enum INPAR::STR::PreStress& prestresstype = datasdyn.GetPreStressType();

  // check if we have a problem that needs to be prestressed
  if (::UTILS::PRESTRESS::IsAny(prestresstype))
  {
    impl_int_ptr = Teuchos::rcp(new STR::IMPLICIT::PreStress());
    return impl_int_ptr;
  }

  switch (dyntype)
  {
    // Static analysis
    case INPAR::STR::dyna_statics:
    {
      impl_int_ptr = Teuchos::rcp(new STR::IMPLICIT::Statics());
      break;
    }

    // Generalised-alpha time integration
    case INPAR::STR::dyna_genalpha:
    {
      impl_int_ptr = Teuchos::rcp(new STR::IMPLICIT::GenAlpha());
      break;
    }

    // Generalised-alpha time integration for Lie groups (e.g. SO3 group of rotation matrices)
    case INPAR::STR::dyna_genalpha_liegroup:
    {
      impl_int_ptr = Teuchos::rcp(new STR::IMPLICIT::GenAlphaLieGroup());
      break;
    }

    // One-step-theta (OST) time integration
    case INPAR::STR::dyna_onesteptheta:
    {
      impl_int_ptr = Teuchos::rcp(new STR::IMPLICIT::OneStepTheta());
      break;
    }

    // Generalised energy-momentum method (GEMM)
    case INPAR::STR::dyna_gemm:
    {
      impl_int_ptr = Teuchos::rcp(new STR::IMPLICIT::Gemm());
      break;
    }

    // Everything else
    default:
    {
      /* Do nothing and return Techos::null. */
      break;
    }
  }  // end of switch(dynType)

  return impl_int_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::Integrator> STR::Factory::BuildExplicitIntegrator(
    const STR::TIMINT::BaseDataSDyn& datasdyn) const
{
  //  Teuchos::RCP<STR::EXPLICIT::Generic> expl_int_ptr = Teuchos::null;
  dserror("Not yet implemented!");

  return Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::Integrator> STR::BuildIntegrator(const STR::TIMINT::BaseDataSDyn& datasdyn)
{
  STR::Factory factory;

  return factory.BuildIntegrator(datasdyn);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::Dbc> STR::Factory::BuildDbc(const STR::TIMINT::BaseDataSDyn& datasdyn) const
{
  // if you want your model specific dbc object, check here if your model type is
  // active ( datasdyn.GetModelTypes() )and build your own dbc object
  Teuchos::RCP<STR::Dbc> dbc = Teuchos::null;
  dbc = Teuchos::rcp(new STR::Dbc());

  return dbc;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::Dbc> STR::BuildDbc(const STR::TIMINT::BaseDataSDyn& datasdyn)
{
  STR::Factory factory;

  return factory.BuildDbc(datasdyn);
}
