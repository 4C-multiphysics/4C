/*-----------------------------------------------------------*/
/*! \file

\brief Tangential displacement predictor.



\level 3

*/
/*-----------------------------------------------------------*/


#include "baci_structure_new_predict_tangdis.H"

#include "baci_linalg_sparsematrix.H"
#include "baci_linalg_utils_sparse_algebra_assemble.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_solver_nonlin_nox_group.H"
#include "baci_solver_nonlin_nox_group_prepostoperator.H"
#include "baci_structure_new_dbc.H"
#include "baci_structure_new_impl_generic.H"
#include "baci_structure_new_model_evaluator.H"
#include "baci_structure_new_model_evaluator_data.H"
#include "baci_structure_new_timint_base.H"
#include "baci_structure_new_utils.H"
#include "baci_utils_exceptions.H"

#include <NOX_Epetra_Vector.H>


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::PREDICT::TangDis::TangDis() : dbc_incr_ptr_(Teuchos::null), applyLinearReactionForces_(false)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::PREDICT::TangDis::Setup()
{
  CheckInit();
  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln group in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_grp_opt = NoxParams().sublist("Group Options");
  // Get the current map. If there is no map, return a new empty one. (reference)
  NOX::NLN::GROUP::PrePostOperator::Map& prepostgroup_map =
      NOX::NLN::GROUP::PrePostOp::GetMutableMap(p_grp_opt);
  // create the new tangdis pre/post operator
  Teuchos::RCP<NOX::NLN::Abstract::PrePostOperator> preposttangdis_ptr =
      Teuchos::rcp(new NOX::NLN::GROUP::PrePostOp::TangDis(Teuchos::rcp(this, false)));
  // insert/replace the old pointer in the map
  prepostgroup_map[NOX::NLN::GROUP::prepost_tangdis] = preposttangdis_ptr;

  issetup_ = true;

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::PREDICT::TangDis::Compute(NOX::Abstract::Group& grp)
{
  CheckInitSetup();
  NOX::NLN::Group* grp_ptr = dynamic_cast<NOX::NLN::Group*>(&grp);
  dsassert(grp_ptr != nullptr, "Dynamic cast failed!");
  grp_ptr->ResetPrePostOperator(NoxParams().sublist("Group Options"));

  ImplInt().EvalData().SetPredictorType(INPAR::STR::pred_tangdis);

  // ---------------------------------------------------------------------------
  // calculate the dbc increment on the dirichlet boundary
  // ---------------------------------------------------------------------------
  dbc_incr_ptr_ = Dbc().GetDirichletIncrement();

  // ---------------------------------------------------------------------------
  // We create at this point a new solution vector and initialize it
  // with the values of the last converged time step.
  // ---------------------------------------------------------------------------
  Teuchos::RCP<NOX::Epetra::Vector> x_ptr = GlobalState().CreateGlobalVector(
      DRT::UTILS::VecInitType::last_time_step, ImplInt().ModelEvalPtr());
  // Set the solution vector in the nox group. This will reset all isValid
  // flags.
  grp.setX(*x_ptr);

  // ---------------------------------------------------------------------------
  // Compute F and jacobian and apply the linear reaction forces due to changing
  // Dirichlet boundary conditions.
  // ---------------------------------------------------------------------------
  applyLinearReactionForces_ = true;
  grp_ptr->computeFandJacobian();
  applyLinearReactionForces_ = false;

  // ---------------------------------------------------------------------------
  // Check if we are using a Newton direction
  // ---------------------------------------------------------------------------
  std::string dir_str = NoxParams().sublist("Direction").get<std::string>("Method");
  if (dir_str == "User Defined")
    dir_str = NoxParams().sublist("Direction").get<std::string>("User Defined Method");
  if (dir_str != "Newton" and dir_str != "Modified Newton")
    dserror(
        "The TangDis predictor is currently only working for the direction-"
        "methods \"Newton\" and \"Modified Newton\".");

  // ---------------------------------------------------------------------------
  // (re)set the linear solver parameters
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p =
      NoxParams().sublist("Direction").sublist("Newton").sublist("Linear Solver");
  p.set<int>("Number of Nonlinear Iterations", 0);
  p.set<int>("Current Time Step", GlobalState().GetStepNp());
  // ToDo Get the actual tolerance value
  p.set<double>("Wanted Tolerance", 1.0e-6);

  // ---------------------------------------------------------------------------
  // solve the linear system of equations and update the current state
  // ---------------------------------------------------------------------------
  // compute the Newton direction
  grp_ptr->computeNewton(p);
  // reset isValid flags
  grp_ptr->computeX(*grp_ptr, grp_ptr->getNewton(), 1.0);
  // add the DBC values to the current state vector
  Teuchos::RCP<Epetra_Vector> dbc_incr_exp_ptr =
      Teuchos::rcp(new Epetra_Vector(GlobalState().GlobalProblemMap(), true));
  CORE::LINALG::Export(*dbc_incr_ptr_, *dbc_incr_exp_ptr);
  grp_ptr->computeX(*grp_ptr, *dbc_incr_exp_ptr, 1.0);
  // Reset the state variables
  const NOX::Epetra::Vector& x_eptra = dynamic_cast<const NOX::Epetra::Vector&>(grp_ptr->getX());
  // set the consistent state in the models (e.g. structure and contact models)
  ImplInt().ResetModelStates(x_eptra.getEpetraVector());

  // For safety purposes, we set the dbc_incr vector to zero
  dbc_incr_ptr_->PutScalar(0.0);

  ImplInt().ModelEval().Predict(GetType());

  ImplInt().EvalData().SetPredictorType(INPAR::STR::pred_vague);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Epetra_Vector& STR::PREDICT::TangDis::GetDbcIncr() const
{
  dsassert(!dbc_incr_ptr_.is_null(), "The dbc increment is not initialized!");
  return *dbc_incr_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const bool& STR::PREDICT::TangDis::IsApplyLinearReactionForces() const
{
  return applyLinearReactionForces_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::PREDICT::TangDis::PreApplyForceExternal(Epetra_Vector& fextnp) const
{
  CheckInitSetup();

  if (GetType() != INPAR::STR::pred_tangdis_constfext) return false;

  if (applyLinearReactionForces_)
  {
    fextnp.Scale(1.0, *GlobalState().GetFextN());
    return true;
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::GROUP::PrePostOp::TangDis::TangDis(
    const Teuchos::RCP<const ::STR::PREDICT::TangDis>& tang_predict_ptr)
    : tang_predict_ptr_(tang_predict_ptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::GROUP::PrePostOp::TangDis::runPostComputeF(
    Epetra_Vector& F, const NOX::NLN::Group& grp)
{
  // If we do not want to apply linear reaction forces due to changing Dirichlet
  // boundary conditions, we just return.
  if (not tang_predict_ptr_->IsApplyLinearReactionForces()) return;

  // get the new dirichlet boundary increment
  const Epetra_Vector& dbc_incr = tang_predict_ptr_->GetDbcIncr();

  double dbc_incr_nrm2 = 0.0;
  dbc_incr.Norm2(&dbc_incr_nrm2);

  // If there are only Neumann loads, do a direct return.
  if (dbc_incr_nrm2 == 0.0) return;

  /* Alternatively, it's also possible to get a const pointer on the jacobian
   * by calling grp.getLinearSystem()->getJacobianOperator()... */
  Teuchos::RCP<const CORE::LINALG::SparseMatrix> stiff_ptr =
      tang_predict_ptr_->GlobalState().GetJacobianDisplBlock();

  // check if the jacobian is filled
  if (not stiff_ptr->Filled()) dserror("The jacobian is not yet filled!");

  Teuchos::RCP<Epetra_Vector> freact_ptr =
      Teuchos::rcp(new Epetra_Vector(*tang_predict_ptr_->GlobalState().DofRowMapView()));
  if (stiff_ptr->Multiply(false, dbc_incr, *freact_ptr)) dserror("Multiply failed!");

  // finally add the linear reaction forces to the current rhs
  ::CORE::LINALG::AssembleMyVector(1.0, F, 1.0, *freact_ptr);

  return;
}
