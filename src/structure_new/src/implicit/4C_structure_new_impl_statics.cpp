/*-----------------------------------------------------------*/
/*! \file

\brief Static (time) integrator.


\level 3

*/
/*-----------------------------------------------------------*/

#include "4C_structure_new_impl_statics.hpp"

#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_structure.hpp"
#include "4C_structure_new_predict_generic.hpp"
#include "4C_structure_new_timint_implicit.hpp"

#include <Epetra_Vector.h>
#include <NOX_Epetra_Vector.H>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::IMPLICIT::Statics::Statics()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::Setup()
{
  CheckInit();

  // Call the Setup() of the abstract base class first.
  Generic::Setup();

  // check for valid parameter combinations:
  if (EvalData().GetDampingType() != INPAR::STR::damp_none)
    FOUR_C_THROW("ERROR: Damping not provided for statics time integration!");

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::PostSetup()
{
  CheckInitSetup();
  // DO NOTHING
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::SetState(const Epetra_Vector& x)
{
  CheckInitSetup();
  if (IsPredictorState()) return;

  Teuchos::RCP<Epetra_Vector> disnp_ptr = GlobalState().ExtractDisplEntries(x);
  GlobalState().GetDisNp()->Scale(1.0, *disnp_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::ApplyForce(const Epetra_Vector& x, Epetra_Vector& f)
{
  CheckInitSetup();
  ResetEvalParams();
  return ModelEval().ApplyForce(x, f, 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::ApplyStiff(const Epetra_Vector& x, CORE::LINALG::SparseOperator& jac)
{
  CheckInitSetup();
  ResetEvalParams();
  bool ok = ModelEval().ApplyStiff(x, jac, 1.0);
  jac.Complete();
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::ApplyForceStiff(
    const Epetra_Vector& x, Epetra_Vector& f, CORE::LINALG::SparseOperator& jac)
{
  CheckInitSetup();
  ResetEvalParams();
  bool ok = ModelEval().ApplyForceStiff(x, f, jac, 1.0);
  jac.Complete();
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::AssembleForce(
    Epetra_Vector& f, const std::vector<INPAR::STR::ModelType>* without_these_models) const
{
  CheckInitSetup();
  return ModelEval().AssembleForce(1.0, f, without_these_models);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::WriteRestart(
    IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  CheckInitSetup();

  // create empty dynamic forces
  auto finertialn = CORE::LINALG::CreateVector(*GlobalState().DofRowMapView(), true);
  auto fviscon = CORE::LINALG::CreateVector(*GlobalState().DofRowMapView(), true);

  // write dynamic forces, so that it can be used later on for restart dynamics analysis
  iowriter.WriteVector("finert", finertialn);
  iowriter.WriteVector("fvisco", fviscon);

  ModelEval().WriteRestart(iowriter, forced_writerestart);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::ReadRestart(IO::DiscretizationReader& ioreader)
{
  CheckInitSetup();
  ModelEval().ReadRestart(ioreader);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::IMPLICIT::Statics::CalcRefNormForce(
    const enum ::NOX::Abstract::Vector::NormType& type) const
{
  CheckInitSetup();

  const Teuchos::RCP<Epetra_Vector> fintnp =
      Teuchos::rcp_const_cast<Epetra_Vector>(GlobalState().GetFintNp());
  const Teuchos::RCP<Epetra_Vector> fextnp =
      Teuchos::rcp_const_cast<Epetra_Vector>(GlobalState().GetFextNp());
  const Teuchos::RCP<Epetra_Vector> freactnp =
      Teuchos::rcp_const_cast<Epetra_Vector>(GlobalState().GetFreactNp());

  // switch from Epetra_Vector to ::NOX::Epetra::Vector (view but read-only)
  Teuchos::RCP<const ::NOX::Epetra::Vector> fintnp_nox_ptr =
      Teuchos::rcp(new ::NOX::Epetra::Vector(fintnp, ::NOX::Epetra::Vector::CreateView));
  Teuchos::RCP<const ::NOX::Epetra::Vector> fextnp_nox_ptr =
      Teuchos::rcp(new ::NOX::Epetra::Vector(fextnp, ::NOX::Epetra::Vector::CreateView));
  Teuchos::RCP<const ::NOX::Epetra::Vector> freactnp_nox_ptr =
      Teuchos::rcp(new ::NOX::Epetra::Vector(freactnp, ::NOX::Epetra::Vector::CreateView));

  // norm of the internal forces
  double fintnorm = fintnp_nox_ptr->norm(type);

  // norm of the external forces
  double fextnorm = fextnp_nox_ptr->norm(type);

  // norm of reaction forces
  double freactnorm = freactnp_nox_ptr->norm(type);

  // return characteristic norm
  return std::max(fintnorm, std::max(fextnorm, freactnorm));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::IMPLICIT::Statics::GetIntParam() const { return 0.0; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::PreUpdate()
{
  CheckInitSetup();
  const STR::TIMINT::Implicit* impl_ptr = dynamic_cast<const STR::TIMINT::Implicit*>(&TimInt());
  if (impl_ptr == nullptr) return;

  // get the time step size
  const double dt = (*GlobalState().GetDeltaTime())[0];

  const INPAR::STR::PredEnum& pred_type = impl_ptr->Predictor().GetType();
  Teuchos::RCP<Epetra_Vector>& accnp_ptr = GlobalState().GetAccNp();
  Teuchos::RCP<Epetra_Vector>& velnp_ptr = GlobalState().GetVelNp();

  switch (pred_type)
  {
    // case: constant acceleration
    case INPAR::STR::pred_constacc:
    {
      // read-only access
      Teuchos::RCP<const Epetra_Vector> veln_ptr = GlobalState().GetVelN();
      // update the pseudo acceleration (statics!)
      accnp_ptr->Update(1.0 / dt, *velnp_ptr, -1.0 / dt, *veln_ptr, 0.0);

      [[fallthrough]];
    }
    // case: constant acceleration OR constant velocity
    case INPAR::STR::pred_constvel:
    {
      // read-only access
      Teuchos::RCP<const Epetra_Vector> disn_ptr = GlobalState().GetDisN();
      Teuchos::RCP<const Epetra_Vector> disnp_ptr = GlobalState().GetDisNp();
      // update the pseudo velocity (statics!)
      velnp_ptr->Update(1.0 / dt, *disnp_ptr, -1.0 / dt, *disn_ptr, 0.0);
      // ATTENTION: Break for both cases!
      break;
    }
    default:
      /* do nothing */
      break;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::UpdateStepState()
{
  CheckInitSetup();
  // update model specific variables
  ModelEval().UpdateStepState(0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::UpdateStepElement()
{
  CheckInitSetup();
  ModelEval().UpdateStepElement();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::PredictConstDisConsistVelAcc(
    Epetra_Vector& disnp, Epetra_Vector& velnp, Epetra_Vector& accnp) const
{
  CheckInitSetup();
  // constant predictor : displacement in domain
  disnp.Update(1.0, *GlobalState().GetDisN(), 0.0);
  // new end-point velocities, these stay zero in static calculation
  velnp.PutScalar(0.0);
  // new end-point accelerations, these stay zero in static calculation
  accnp.PutScalar(0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::PredictConstVelConsistAcc(
    Epetra_Vector& disnp, Epetra_Vector& velnp, Epetra_Vector& accnp) const
{
  CheckInitSetup();
  // If there is not enough history information, return a fail status.
  if (GlobalState().GetStepN() == 0) return false;

  // Displacement increment over last time step
  Teuchos::RCP<Epetra_Vector> disp_inc =
      Teuchos::rcp(new Epetra_Vector(*GlobalState().DofRowMapView(), true));
  disp_inc->Update((*GlobalState().GetDeltaTime())[0], *GlobalState().GetVelN(), 0.);
  // apply the dbc on the auxiliary vector
  TimInt().GetDBC().ApplyDirichletToVector(disp_inc);
  // update the solution variables
  disnp.Update(1.0, *GlobalState().GetDisN(), 0.0);
  disnp.Update(1.0, *disp_inc, 1.0);
  velnp.Update(1.0, *GlobalState().GetVelN(), 0.0);
  accnp.Update(1.0, *GlobalState().GetAccN(), 0.0);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::PredictConstAcc(
    Epetra_Vector& disnp, Epetra_Vector& velnp, Epetra_Vector& accnp) const
{
  CheckInitSetup();
  // If there is not enough history information try a different predictor with
  // less requirements.
  if (GlobalState().GetStepN() < 2) return PredictConstVelConsistAcc(disnp, velnp, accnp);

  // Displacement increment over last time step
  Teuchos::RCP<Epetra_Vector> disp_inc =
      Teuchos::rcp(new Epetra_Vector(*GlobalState().DofRowMapView(), true));
  const double& dt = (*GlobalState().GetDeltaTime())[0];
  disp_inc->Update(dt, *GlobalState().GetVelN(), 0.);
  disp_inc->Update(0.5 * dt * dt, *GlobalState().GetAccN(), 1.0);
  // apply the dbc on the auxiliary vector
  TimInt().GetDBC().ApplyDirichletToVector(disp_inc);
  // update the solution variables
  disnp.Update(1.0, *GlobalState().GetDisN(), 0.0);
  disnp.Update(1., *disp_inc, 1.);
  velnp.Update(1.0, *GlobalState().GetVelN(), 0.0);
  accnp.Update(1.0, *GlobalState().GetAccN(), 0.0);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::ResetEvalParams()
{
  // call base class
  STR::IMPLICIT::Generic::ResetEvalParams();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::IMPLICIT::Statics::GetModelValue(const Epetra_Vector& x)
{
  Teuchos::RCP<const Epetra_Vector> disnp_ptr = GlobalState().ExtractDisplEntries(x);
  const Epetra_Vector& disnp = *disnp_ptr;

  SetState(disnp);

  EvalData().ClearValuesForAllEnergyTypes();
  STR::MODELEVALUATOR::Structure& str_model =
      dynamic_cast<STR::MODELEVALUATOR::Structure&>(Evaluator(INPAR::STR::model_structure));

  str_model.DetermineStrainEnergy(disnp, true);
  const double int_energy_np = EvalData().GetEnergyData(STR::internal_energy);
  double ext_energy_np = 0.0;
  GlobalState().GetFextNp()->Dot(disnp, &ext_energy_np);
  const double total = int_energy_np - ext_energy_np;

  std::ostream& os = IO::cout.os(IO::debug);
  os << __LINE__ << __PRETTY_FUNCTION__ << "\n";
  os << "internal/strain energy       = " << int_energy_np << "\n"
     << "external energy              = " << ext_energy_np << "\n";
  os << std::string(80, '-') << "\n";
  os << "Total                     = " << total << "\n";
  os << std::string(80, '-') << "\n";


  return total;
}

FOUR_C_NAMESPACE_CLOSE
