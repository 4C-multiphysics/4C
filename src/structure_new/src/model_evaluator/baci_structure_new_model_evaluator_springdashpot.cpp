/*-----------------------------------------------------------*/
/*! \file

\brief Evaluation and assembly of all spring dashpot terms


\level 3

*/
/*-----------------------------------------------------------*/

#include "baci_structure_new_model_evaluator_springdashpot.hpp"

#include "baci_global_data.hpp"
#include "baci_inpar_structure.hpp"
#include "baci_io.hpp"
#include "baci_lib_discret.hpp"
#include "baci_linalg_sparsematrix.hpp"
#include "baci_linalg_sparseoperator.hpp"
#include "baci_linalg_utils_sparse_algebra_assemble.hpp"
#include "baci_structure_new_model_evaluator_data.hpp"
#include "baci_structure_new_timint_base.hpp"
#include "baci_structure_new_utils.hpp"
#include "baci_utils_exceptions.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
STR::MODELEVALUATOR::SpringDashpot::SpringDashpot()
    : disnp_ptr_(Teuchos::null),
      velnp_ptr_(Teuchos::null),
      stiff_spring_ptr_(Teuchos::null),
      fspring_np_ptr_(Teuchos::null)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::Setup()
{
  dsassert(IsInit(), "Init() has not been called, yet!");

  // get all spring dashpot conditions
  std::vector<Teuchos::RCP<DRT::Condition>> springdashpots;
  Discret().GetCondition("RobinSpringDashpot", springdashpots);

  // new instance of spring dashpot BC for each condition
  Teuchos::RCP<DRT::Discretization> discret_ptr = DiscretPtr();
  for (auto& springdashpot : springdashpots)
    springs_.emplace_back(Teuchos::rcp(new CONSTRAINTS::SpringDashpot(discret_ptr, springdashpot)));

  // setup the displacement pointer
  disnp_ptr_ = GState().GetDisNp();
  velnp_ptr_ = GState().GetVelNp();

  fspring_np_ptr_ = Teuchos::rcp(new Epetra_Vector(*GState().DofRowMapView()));
  stiff_spring_ptr_ =
      Teuchos::rcp(new CORE::LINALG::SparseMatrix(*GState().DofRowMapView(), 81, true, true));

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::Reset(const Epetra_Vector& x)
{
  CheckInitSetup();

  // loop over all spring dashpot conditions and reset them
  for (const auto& spring : springs_) spring->ResetNewton();

  // update the structural displacement vector
  disnp_ptr_ = GState().GetDisNp();

  // update the structural displacement vector
  velnp_ptr_ = GState().GetVelNp();

  fspring_np_ptr_->PutScalar(0.0);
  stiff_spring_ptr_->Zero();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::SpringDashpot::EvaluateForce()
{
  CheckInitSetup();

  Teuchos::ParameterList springdashpotparams;
  // loop over all spring dashpot conditions and evaluate them
  fspring_np_ptr_ = Teuchos::rcp(new Epetra_Vector(*GState().DofRowMapView()));
  for (const auto& spring : springs_)
  {
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->GetSpringType();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
    {
      springdashpotparams.set("total time", GState().GetTimeNp());
      spring->EvaluateRobin(
          Teuchos::null, fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
    {
      springdashpotparams.set("dt", (*GState().GetDeltaTime())[0]);
      spring->EvaluateForce(*fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::SpringDashpot::EvaluateStiff()
{
  CheckInitSetup();

  fspring_np_ptr_ = Teuchos::rcp(new Epetra_Vector(*GState().DofRowMapView(), true));

  // factors from time-integrator for derivative of d(v_{n+1}) / d(d_{n+1})
  // needed for stiffness contribution from dashpot
  const double fac_vel = EvalData().GetTimIntFactorVel();
  const double fac_disp = EvalData().GetTimIntFactorDisp();
  const double time_fac = fac_vel / fac_disp;
  Teuchos::ParameterList springdashpotparams;
  if (fac_vel > 0.0) springdashpotparams.set("time_fac", time_fac);

  // loop over all spring dashpot conditions and evaluate them
  for (const auto& spring : springs_)
  {
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->GetSpringType();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
    {
      springdashpotparams.set("total time", GState().GetTimeNp());
      spring->EvaluateRobin(
          stiff_spring_ptr_, Teuchos::null, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
    {
      springdashpotparams.set("dt", (*GState().GetDeltaTime())[0]);
      spring->EvaluateForceStiff(
          *stiff_spring_ptr_, *fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
  }

  if (not stiff_spring_ptr_->Filled()) stiff_spring_ptr_->Complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::SpringDashpot::EvaluateForceStiff()
{
  CheckInitSetup();

  // get displacement DOFs
  fspring_np_ptr_ = Teuchos::rcp(new Epetra_Vector(*GState().DofRowMap(), true));

  // factors from time-integrator for derivative of d(v_{n+1}) / d(d_{n+1})
  // needed for stiffness contribution from dashpot
  const double fac_vel = EvalData().GetTimIntFactorVel();
  const double fac_disp = EvalData().GetTimIntFactorDisp();
  const double time_fac = fac_vel / fac_disp;
  Teuchos::ParameterList springdashpotparams;

  if (fac_vel > 0.0) springdashpotparams.set("time_fac", time_fac);

  // loop over all spring dashpot conditions and evaluate them
  for (const auto& spring : springs_)
  {
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->GetSpringType();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
    {
      springdashpotparams.set("total time", GState().GetTimeNp());
      spring->EvaluateRobin(
          stiff_spring_ptr_, fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
    {
      springdashpotparams.set("dt", (*GState().GetDeltaTime())[0]);
      spring->EvaluateForceStiff(
          *stiff_spring_ptr_, *fspring_np_ptr_, disnp_ptr_, velnp_ptr_, springdashpotparams);
    }
  }

  if (not stiff_spring_ptr_->Filled()) stiff_spring_ptr_->Complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::SpringDashpot::AssembleForce(
    Epetra_Vector& f, const double& timefac_np) const
{
  CORE::LINALG::AssembleMyVector(1.0, f, timefac_np, *fspring_np_ptr_);
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::SpringDashpot::AssembleJacobian(
    CORE::LINALG::SparseOperator& jac, const double& timefac_np) const
{
  Teuchos::RCP<CORE::LINALG::SparseMatrix> jac_dd_ptr = GState().ExtractDisplBlock(jac);
  jac_dd_ptr->Add(*stiff_spring_ptr_, false, timefac_np, 1.0);
  // no need to keep it
  stiff_spring_ptr_->Zero();
  // nothing to do
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::WriteRestart(
    IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  // row maps for export
  Teuchos::RCP<Epetra_Vector> springoffsetprestr =
      Teuchos::rcp(new Epetra_Vector(*Discret().DofRowMap()));
  Teuchos::RCP<Epetra_MultiVector> springoffsetprestr_old =
      Teuchos::rcp(new Epetra_MultiVector(*(Discret().NodeRowMap()), 3, true));

  // collect outputs from all spring dashpot conditions
  for (const auto& spring : springs_)
  {
    // get spring type from current condition
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->GetSpringType();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
      spring->OutputPrestrOffset(springoffsetprestr);
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal)
      spring->OutputPrestrOffsetOld(springoffsetprestr_old);
  }

  // write vector to output for restart
  iowriter.WriteVector("springoffsetprestr", springoffsetprestr);
  // write vector to output for restart
  iowriter.WriteVector("springoffsetprestr_old", springoffsetprestr_old);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::ReadRestart(IO::DiscretizationReader& ioreader)
{
  Teuchos::RCP<Epetra_Vector> tempvec = Teuchos::rcp(new Epetra_Vector(*Discret().DofRowMap()));
  Teuchos::RCP<Epetra_MultiVector> tempvecold =
      Teuchos::rcp(new Epetra_MultiVector(*(Discret().NodeRowMap()), 3, true));

  ioreader.ReadVector(tempvec, "springoffsetprestr");
  ioreader.ReadMultiVector(tempvecold, "springoffsetprestr_old");

  // loop over all spring dashpot conditions and set restart
  for (const auto& spring : springs_)
  {
    // get spring type from current condition
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->GetSpringType();

    if (stype == CONSTRAINTS::SpringDashpot::xyz or
        stype == CONSTRAINTS::SpringDashpot::refsurfnormal)
      spring->SetRestart(tempvec);
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal) spring->SetRestartOld(tempvecold);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::UpdateStepState(const double& timefac_n)
{
  // add the old time factor scaled contributions to the residual
  Teuchos::RCP<Epetra_Vector>& fstructold_ptr = GState().GetFstructureOld();
  fstructold_ptr->Update(timefac_n, *fspring_np_ptr_, 1.0);

  // check for prestressing and reset if necessary
  const INPAR::STR::PreStress prestress_type = TimInt().GetDataSDyn().GetPreStressType();
  const double prestress_time = TimInt().GetDataSDyn().GetPreStressTime();

  if (prestress_type != INPAR::STR::PreStress::none &&
      GState().GetTimeNp() <= prestress_time + 1.0e-15)
  {
    switch (prestress_type)
    {
      case INPAR::STR::PreStress::mulf:
      case INPAR::STR::PreStress::material_iterative:
        for (const auto& spring : springs_) spring->ResetPrestress(GState().GetDisNp());
      default:
        break;
    }
  }
  for (const auto& spring : springs_) spring->Update();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::OutputStepState(IO::DiscretizationWriter& iowriter) const
{
  // row maps for export
  Teuchos::RCP<Epetra_Vector> gap =
      Teuchos::rcp(new Epetra_Vector(*(Discret().NodeRowMap()), true));
  Teuchos::RCP<Epetra_MultiVector> normals =
      Teuchos::rcp(new Epetra_MultiVector(*(Discret().NodeRowMap()), 3, true));
  Teuchos::RCP<Epetra_MultiVector> springstress =
      Teuchos::rcp(new Epetra_MultiVector(*(Discret().NodeRowMap()), 3, true));

  // collect outputs from all spring dashpot conditions
  bool found_cursurfnormal = false;
  for (const auto& spring : springs_)
  {
    spring->OutputGapNormal(gap, normals, springstress);

    // get spring type from current condition
    const CONSTRAINTS::SpringDashpot::SpringType stype = spring->GetSpringType();
    if (stype == CONSTRAINTS::SpringDashpot::cursurfnormal) found_cursurfnormal = true;
  }

  // write vectors to output
  if (found_cursurfnormal)
  {
    iowriter.WriteVector("gap", gap);
    iowriter.WriteVector("curnormals", normals);
  }

  // write spring stress if defined in io-flag
  if (CORE::UTILS::IntegralValue<bool>(GLOBAL::Problem::Instance()->IOParams(), "OUTPUT_SPRING"))
    iowriter.WriteVector("springstress", springstress);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::ResetStepState()
{
  CheckInitSetup();

  for (auto& spring : springs_)
  {
    spring->ResetStepState();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> STR::MODELEVALUATOR::SpringDashpot::GetBlockDofRowMapPtr() const
{
  CheckInitSetup();
  return GState().DofRowMap();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> STR::MODELEVALUATOR::SpringDashpot::GetCurrentSolutionPtr() const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> STR::MODELEVALUATOR::SpringDashpot::GetLastTimeStepSolutionPtr()
    const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STR::MODELEVALUATOR::SpringDashpot::PostOutput() { CheckInitSetup(); }

BACI_NAMESPACE_CLOSE
