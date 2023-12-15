/*---------------------------------------------------------------------*/
/*! \file
\brief Concrete mplementation of all the %NOX::NLN::CONSTRAINT::Interface::Required
       (pure) virtual routines.

\level 3


*/
/*---------------------------------------------------------------------*/

#include "baci_contact_noxinterface.H"

#include "baci_contact_abstract_strategy.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_solver_nonlin_nox_aux.H"

#include <Epetra_Vector.h>
#include <NOX_Epetra_Vector.H>

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CONTACT::NoxInterface::NoxInterface()
    : isinit_(false),
      issetup_(false),
      strategy_ptr_(Teuchos::null),
      cycling_maps_(std::vector<Teuchos::RCP<Epetra_Map>>(0))
{
  // should stay empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::NoxInterface::Init(const Teuchos::RCP<CONTACT::CoAbstractStrategy>& strategy_ptr)
{
  issetup_ = false;

  strategy_ptr_ = strategy_ptr;

  // set flag at the end
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::NoxInterface::Setup()
{
  CheckInit();

  // set flag at the end
  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double CONTACT::NoxInterface::GetConstraintRHSNorms(const Epetra_Vector& F,
    NOX::NLN::StatusTest::QuantityType checkQuantity, ::NOX::Abstract::Vector::NormType type,
    bool isScaled) const
{
  if (checkQuantity != NOX::NLN::StatusTest::quantity_contact_normal and
      checkQuantity != NOX::NLN::StatusTest::quantity_contact_friction)
    return -1.0;

  Teuchos::RCP<const Epetra_Vector> constrRhs =
      Strategy().GetRhsBlockPtrForNormCheck(CONTACT::VecBlockType::constraint);

  // no contact contributions present
  if (constrRhs.is_null()) return 0.0;

  // export the vector to the current redistributed map
  Teuchos::RCP<Epetra_Vector> constrRhs_red = Teuchos::null;
  // Note: PointSameAs is faster than SameAs and should do the job right here,
  // since we replace the map afterwards anyway.               hiermeier 08/17
  if (not constrRhs->Map().PointSameAs(Strategy().LMDoFRowMap(true)))
  {
    constrRhs_red = Teuchos::rcp(new Epetra_Vector(Strategy().LMDoFRowMap(true)));
    CORE::LINALG::Export(*constrRhs, *constrRhs_red);
  }
  else
    constrRhs_red = Teuchos::rcp(new Epetra_Vector(*constrRhs));

  // replace the map
  constrRhs_red->ReplaceMap(Strategy().SlDoFRowMap(true));

  double constrNorm = -1.0;
  Teuchos::RCP<const ::NOX::Epetra::Vector> constrRhs_nox = Teuchos::null;
  switch (checkQuantity)
  {
    case NOX::NLN::StatusTest::quantity_contact_normal:
    {
      // create vector with redistributed slave dof row map in normal direction
      Teuchos::RCP<Epetra_Vector> nConstrRhs =
          CORE::LINALG::ExtractMyVector(*constrRhs_red, Strategy().SlNormalDoFRowMap(true));


      constrRhs_nox =
          Teuchos::rcp(new ::NOX::Epetra::Vector(nConstrRhs, ::NOX::Epetra::Vector::CreateView));
      break;
    }
    case NOX::NLN::StatusTest::quantity_contact_friction:
    {
      // create vector with redistributed slave dof row map in tangential directions
      Teuchos::RCP<Epetra_Vector> tConstrRhs =
          CORE::LINALG::ExtractMyVector(*constrRhs_red, Strategy().SlTangentialDoFRowMap(true));

      constrRhs_nox =
          Teuchos::rcp(new ::NOX::Epetra::Vector(tConstrRhs, ::NOX::Epetra::Vector::CreateView));
      break;
    }
    default:
    {
      dserror("Unsupported quantity type!");
      break;
    }
  }
  constrNorm = constrRhs_nox->norm(type);
  if (isScaled) constrNorm /= static_cast<double>(constrRhs_nox->length());

  return constrNorm;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double CONTACT::NoxInterface::GetLagrangeMultiplierUpdateRMS(const Epetra_Vector& xNew,
    const Epetra_Vector& xOld, double aTol, double rTol,
    NOX::NLN::StatusTest::QuantityType checkQuantity, bool disable_implicit_weighting) const
{
  if (checkQuantity != NOX::NLN::StatusTest::quantity_contact_normal and
      checkQuantity != NOX::NLN::StatusTest::quantity_contact_friction)
    return -1.0;

  double rms = -1.0;
  Teuchos::RCP<Epetra_Vector> z_ptr = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> zincr_ptr = Teuchos::null;
  switch (checkQuantity)
  {
    case NOX::NLN::StatusTest::quantity_contact_normal:
    {
      // extract vectors with redistributed slave dof row map in normal direction
      z_ptr = CORE::LINALG::ExtractMyVector(
          *Strategy().GetLagrMultNp(true), Strategy().SlNormalDoFRowMap(true));
      zincr_ptr = CORE::LINALG::ExtractMyVector(
          *Strategy().GetLagrMultSolveIncr(), Strategy().SlNormalDoFRowMap(true));

      break;
    }
    case NOX::NLN::StatusTest::quantity_contact_friction:
    {
      // extract vectors with redistributed slave dof row map in tangential directions
      z_ptr = CORE::LINALG::ExtractMyVector(
          *Strategy().GetLagrMultNp(true), Strategy().SlTangentialDoFRowMap(true));
      zincr_ptr = CORE::LINALG::ExtractMyVector(
          *Strategy().GetLagrMultSolveIncr(), Strategy().SlTangentialDoFRowMap(true));

      break;
    }
    default:
    {
      dserror("Unsupported quantity type!");
      break;
    }
  }

  rms = NOX::NLN::AUX::RootMeanSquareNorm(aTol, rTol, Strategy().GetLagrMultNp(true),
      Strategy().GetLagrMultSolveIncr(), disable_implicit_weighting);

  return rms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double CONTACT::NoxInterface::GetLagrangeMultiplierUpdateNorms(const Epetra_Vector& xNew,
    const Epetra_Vector& xOld, NOX::NLN::StatusTest::QuantityType checkQuantity,
    ::NOX::Abstract::Vector::NormType type, bool isScaled) const
{
  if (checkQuantity != NOX::NLN::StatusTest::quantity_contact_normal and
      checkQuantity != NOX::NLN::StatusTest::quantity_contact_friction)
    return -1.0;

  if (Strategy().GetLagrMultNp(true) == Teuchos::null) return 0.;

  double updatenorm = -1.0;
  Teuchos::RCP<Epetra_Vector> zincr_ptr = Teuchos::null;
  switch (checkQuantity)
  {
    case NOX::NLN::StatusTest::quantity_contact_normal:
    {
      // extract vector with redistributed slave dof row map in normal direction
      zincr_ptr = CORE::LINALG::ExtractMyVector(
          *Strategy().GetLagrMultSolveIncr(), Strategy().SlNormalDoFRowMap(true));
      break;
    }
    case NOX::NLN::StatusTest::quantity_contact_friction:
    {
      // extract vector with redistributed slave dof row map in tangential directions
      zincr_ptr = CORE::LINALG::ExtractMyVector(
          *Strategy().GetLagrMultSolveIncr(), Strategy().SlTangentialDoFRowMap(true));

      break;
    }
    default:
    {
      dserror("Unsupported quantity type!");
      break;
    }
  }

  Teuchos::RCP<const ::NOX::Epetra::Vector> zincr_nox_ptr =
      Teuchos::rcp(new ::NOX::Epetra::Vector(zincr_ptr, ::NOX::Epetra::Vector::CreateView));

  updatenorm = zincr_nox_ptr->norm(type);
  // do scaling if desired
  if (isScaled) updatenorm /= static_cast<double>(zincr_nox_ptr->length());

  return updatenorm;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double CONTACT::NoxInterface::GetPreviousLagrangeMultiplierNorms(const Epetra_Vector& xOld,
    NOX::NLN::StatusTest::QuantityType checkQuantity, ::NOX::Abstract::Vector::NormType type,
    bool isScaled) const
{
  if (checkQuantity != NOX::NLN::StatusTest::quantity_contact_normal and
      checkQuantity != NOX::NLN::StatusTest::quantity_contact_friction)
    return -1.0;

  double zoldnorm = -1.0;

  if (Strategy().GetLagrMultNp(true) == Teuchos::null) return 0;

  /* lagrange multiplier of the previous Newton step
   * (NOT equal to zOld_, which is stored in the Strategy object!!!) */
  Teuchos::RCP<Epetra_Vector> zold_ptr =
      Teuchos::rcp(new Epetra_Vector(*Strategy().GetLagrMultNp(true)));
  zold_ptr->Update(-1.0, *Strategy().GetLagrMultSolveIncr(), 1.0);
  Teuchos::RCP<::NOX::Epetra::Vector> zold_nox_ptr = Teuchos::null;
  switch (checkQuantity)
  {
    case NOX::NLN::StatusTest::quantity_contact_normal:
    {
      Teuchos::RCP<Epetra_Vector> znold_ptr =
          CORE::LINALG::ExtractMyVector(*zold_ptr, Strategy().SlNormalDoFRowMap(true));

      zold_nox_ptr =
          Teuchos::rcp(new ::NOX::Epetra::Vector(znold_ptr, ::NOX::Epetra::Vector::CreateView));
      break;
    }
    case NOX::NLN::StatusTest::quantity_contact_friction:
    {
      Teuchos::RCP<Epetra_Vector> ztold_ptr =
          CORE::LINALG::ExtractMyVector(*zold_ptr, Strategy().SlTangentialDoFRowMap(true));

      zold_nox_ptr =
          Teuchos::rcp(new ::NOX::Epetra::Vector(ztold_ptr, ::NOX::Epetra::Vector::CreateView));
      break;
    }
    default:
    {
      dserror("The given quantity type is unsupported!");
      break;
    }
  }

  zoldnorm = zold_nox_ptr->norm(type);
  // do scaling if desired
  if (isScaled) zoldnorm /= static_cast<double>(zold_nox_ptr->length());

  // avoid very small norm values for the pure inactive case
  if (not Strategy().IsInContact()) zoldnorm = std::max(zoldnorm, 1.0);

  return zoldnorm;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum ::NOX::StatusTest::StatusType CONTACT::NoxInterface::GetActiveSetInfo(
    NOX::NLN::StatusTest::QuantityType checkQuantity, int& activesetsize) const
{
  bool semismooth = DRT::INPUT::IntegralValue<int>(Strategy().Params(), "SEMI_SMOOTH_NEWTON");
  if (not semismooth) dserror("Currently we support only the semi-smooth Newton case!");
  // ---------------------------------------------------------------------------
  // get the number of active nodes for the given active set type
  // ---------------------------------------------------------------------------
  switch (checkQuantity)
  {
    case NOX::NLN::StatusTest::quantity_contact_normal:
    {
      activesetsize = Strategy().NumberOfActiveNodes();
      break;
    }
    case NOX::NLN::StatusTest::quantity_contact_friction:
    {
      activesetsize = Strategy().NumberOfSlipNodes();
      break;
    }
    default:
    {
      dserror("The given quantity type is unsupported!");
      break;
    }
  }
  // ---------------------------------------------------------------------------
  // translate the active set semi-smooth Newton convergence flag
  // ---------------------------------------------------------------------------
  if (Strategy().ActiveSetSemiSmoothConverged())
    return ::NOX::StatusTest::Converged;
  else
    return ::NOX::StatusTest::Unconverged;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> CONTACT::NoxInterface::GetCurrentActiveSetMap(
    enum NOX::NLN::StatusTest::QuantityType checkQuantity) const
{
  switch (checkQuantity)
  {
    case NOX::NLN::StatusTest::quantity_contact_normal:
    {
      return Strategy().ActiveRowNodes();
      break;
    }
    case NOX::NLN::StatusTest::quantity_contact_friction:
    {
      return Strategy().SlipRowNodes();
      break;
    }
    default:
    {
      dserror("The given active set type is unsupported!");
      break;
    }
  }  // switch (active_set_type)

  return Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> CONTACT::NoxInterface::GetOldActiveSetMap(
    enum NOX::NLN::StatusTest::QuantityType checkQuantity) const
{
  switch (checkQuantity)
  {
    case NOX::NLN::StatusTest::quantity_contact_normal:
    {
      return Strategy().GetOldActiveRowNodes();
      break;
    }
    case NOX::NLN::StatusTest::quantity_contact_friction:
    {
      return Strategy().GetOldSlipRowNodes();
      break;
    }
    default:
    {
      dserror("The given active set type is unsupported!");
      break;
    }
  }  // switch (active_set_type)

  return Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double CONTACT::NoxInterface::GetModelValue(NOX::NLN::MeritFunction::MeritFctName name) const
{
  switch (name)
  {
    case NOX::NLN::MeritFunction::mrtfct_lagrangian:
    case NOX::NLN::MeritFunction::mrtfct_lagrangian_active:
    {
      return Strategy().GetPotentialValue(name);
    }
    case NOX::NLN::MeritFunction::mrtfct_infeasibility_two_norm:
    case NOX::NLN::MeritFunction::mrtfct_infeasibility_two_norm_active:
    {
      double val = Strategy().GetPotentialValue(name);
      val = std::sqrt(val);

      return val;
    }
    case NOX::NLN::MeritFunction::mrtfct_energy:
    {
      // The energy of the primary field is considered, no contact contribution.
      return 0.0;
    }
    default:
      dserror("Unsupported Merit function name! (enum = %d)", name);
      exit(EXIT_FAILURE);
  }

  dserror("Impossible to reach this point.");
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double CONTACT::NoxInterface::GetLinearizedModelTerms(const Epetra_Vector& dir,
    const enum NOX::NLN::MeritFunction::MeritFctName name,
    const enum NOX::NLN::MeritFunction::LinOrder linorder,
    const enum NOX::NLN::MeritFunction::LinType lintype) const
{
  switch (name)
  {
    case NOX::NLN::MeritFunction::mrtfct_lagrangian:
    case NOX::NLN::MeritFunction::mrtfct_lagrangian_active:
    {
      return Strategy().GetLinearizedPotentialValueTerms(dir, name, linorder, lintype);
    }
    case NOX::NLN::MeritFunction::mrtfct_infeasibility_two_norm:
    case NOX::NLN::MeritFunction::mrtfct_infeasibility_two_norm_active:
    {
      double lin_val = Strategy().GetLinearizedPotentialValueTerms(dir, name, linorder, lintype);
      const double modelvalue = GetModelValue(name);
      if (modelvalue != 0.0) lin_val /= modelvalue;

      return lin_val;
    }
    default:
      dserror("Unsupported Merit function name! (enum = %d)", name);
      exit(EXIT_FAILURE);
  }

  dserror("Impossible to reach this point.");
  exit(EXIT_FAILURE);
}

BACI_NAMESPACE_CLOSE
