/*-----------------------------------------------------------*/
/*! \file

\brief model evaluator for brownian (stochastic and damping)
       forces


\date May, 2016

\level 3

*/
/*-----------------------------------------------------------*/
#include "baci_browniandyn_str_model_evaluator.H"

#include "baci_beam3_base.H"
#include "baci_beaminteraction_calc_utils.H"
#include "baci_beaminteraction_periodic_boundingbox.H"
#include "baci_io.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_linalg_serialdensematrix.H"
#include "baci_linalg_utils_sparse_algebra_assemble.H"
#include "baci_rigidsphere.H"
#include "baci_structure_new_integrator.H"
#include "baci_structure_new_model_evaluator_data.H"
#include "baci_structure_new_timint_base.H"

#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::MODELEVALUATOR::BrownianDyn::BrownianDyn()
    : eval_browniandyn_ptr_(Teuchos::null),
      f_brown_np_ptr_(Teuchos::null),
      f_ext_np_ptr_(Teuchos::null),
      stiff_brownian_ptr_(Teuchos::null),
      maxrandnumelement_(0),
      discret_ptr_(Teuchos::null)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::Setup()
{
  CheckInit();

  // safety check, brownian dynamics simulation only for one step theta and
  // theta = 1.0 (see Cyron 2012)
  if (TimInt().GetDataSDynPtr()->GetDynamicType() != INPAR::STR::dyna_onesteptheta)
    dserror("Brownian dynamics simulation only consistent for one step theta schema.");

  discret_ptr_ = Teuchos::rcp_dynamic_cast<DRT::Discretization>(DiscretPtr(), true);

  // -------------------------------------------------------------------------
  // get pointer to biopolymer network data and init random number data
  // -------------------------------------------------------------------------
  eval_browniandyn_ptr_ = EvalData().BrownianDynPtr();
  brown_dyn_state_data_.browndyn_dt = eval_browniandyn_ptr_->TimeStepConstRandNumb();

  // todo: maybe make input of time step obligatory
  if (brown_dyn_state_data_.browndyn_dt < 0.0)
  {
    brown_dyn_state_data_.browndyn_dt = (*GState().GetDeltaTime())[0];
    if (GState().GetMyRank() == 0)
      std::cout << " Time step " << (*GState().GetDeltaTime())[0]
                << " form Structural Dynamic section used for stochastic forces.\n"
                << std::endl;
  }

  brown_dyn_state_data_.browndyn_step = -1;
  // -------------------------------------------------------------------------
  // setup the brownian forces and the external force pointers
  // -------------------------------------------------------------------------
  f_brown_np_ptr_ = Teuchos::rcp(new Epetra_Vector(*GState().DofRowMap(), true));
  f_ext_np_ptr_ = Teuchos::rcp(new Epetra_Vector(*GState().DofRowMap(), true));
  // -------------------------------------------------------------------------
  // setup the brownian forces and the external force pointers
  // -------------------------------------------------------------------------
  stiff_brownian_ptr_ =
      Teuchos::rcp(new CORE::LINALG::SparseMatrix(*GState().DofRowMapView(), 81, true, true));

  // -------------------------------------------------------------------------
  // get maximal number of random numbers required by any element in the
  // discretization and store them in randomnumbersperelement_
  // -------------------------------------------------------------------------
  RandomNumbersPerElement();
  // -------------------------------------------------------------------------
  // Generate random forces for first time step
  // -------------------------------------------------------------------------
  /* multivector for stochastic forces evaluated by each element; the numbers of
   * vectors in the multivector equals the maximal number of random numbers
   * required by any element in the discretization per time step; therefore this
   * multivector is suitable for synchronization of these random numbers in
   *  parallel computing*/
  eval_browniandyn_ptr_->ResizeRandomForceMVector(discret_ptr_, maxrandnumelement_);
  GenerateGaussianRandomNumbers();

  issetup_ = true;

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::Reset(const Epetra_Vector& x)
{
  CheckInitSetup();

  // todo: somewhat illegal considering of const correctness
  TimInt().GetDataSDynPtr()->GetPeriodicBoundingBox()->ApplyDirichlet(GState().GetTimeN());

  // -------------------------------------------------------------------------
  // reset brownian (stochastic and damping) forces
  // -------------------------------------------------------------------------
  f_brown_np_ptr_->PutScalar(0.0);
  // -------------------------------------------------------------------------
  // reset external forces
  // -------------------------------------------------------------------------
  f_ext_np_ptr_->PutScalar(0.0);
  // -------------------------------------------------------------------------
  // zero out brownian stiffness contributions
  // -------------------------------------------------------------------------
  stiff_brownian_ptr_->Zero();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::EvaluateForce()
{
  CheckInitSetup();
  bool ok = true;
  // ---------------------------------------
  // (1) EXTERNAL FORCES
  // ---------------------------------------
  ok = ApplyForceExternal();

  // ---------------------------------------
  // (2) INTERNAL FORCES
  // ---------------------------------------
  // ordinary internal force
  ok = (ok ? ApplyForceBrownian() : false);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::EvaluateStiff()
{
  CheckInitSetup();
  bool ok = true;

  /* We use the same routines as for the ApplyForceStiff case, but we
   * do not update the global force vector, which is used for the
   * solution process in the NOX library.
   * This is meaningful, since the computational overhead, which is
   * generated by evaluating the right hand side is negligible */

  // -------------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  // so far the Neumann loads implemented especially for brownian don't
  // have a contribution to the jacobian
  //   ApplyForceStiffExternal();

  // -------------------------------------------------------------------------
  // (2) BROWNIAN FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  ApplyForceStiffBrownian();

  if (not stiff_brownian_ptr_->Filled()) stiff_brownian_ptr_->Complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::EvaluateForceStiff()
{
  CheckInitSetup();
  bool ok = true;

  // -------------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  ApplyForceStiffExternal();
  // -------------------------------------------------------------------------
  // (2) BROWNIAN FORCES and STIFFNESS ENTRIES
  // -------------------------------------------------------------------------
  ApplyForceStiffBrownian();

  if (not stiff_brownian_ptr_->Filled()) stiff_brownian_ptr_->Complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::AssembleForce(
    Epetra_Vector& f, const double& timefac_np) const
{
  CheckInitSetup();

  // safety check, brownian dynamics simulation for with one step theta and
  // theta = 1.0 (see Cyron 2012)
  if (abs(timefac_np - 1.0) > 1.0e-8)
    dserror(
        "Brownian dynamics simulation only consistent for one step theta scheme"
        " and theta = 1.0 .");

  // -------------------------------------------------------------------------
  // *********** finally put everything together ***********
  // build residual  Res = F_{brw;n+1}
  //                     - F_{ext;n+1}
  // -------------------------------------------------------------------------
  CORE::LINALG::AssembleMyVector(1.0, f, -timefac_np, *f_ext_np_ptr_);
  CORE::LINALG::AssembleMyVector(1.0, f, timefac_np, *f_brown_np_ptr_);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::AssembleJacobian(
    CORE::LINALG::SparseOperator& jac, const double& timefac_np) const
{
  CheckInitSetup();

  Teuchos::RCP<CORE::LINALG::SparseMatrix> jac_dd_ptr = GState().ExtractDisplBlock(jac);
  jac_dd_ptr->Add(*stiff_brownian_ptr_, false, timefac_np, 1.0);
  // no need to keep it
  stiff_brownian_ptr_->Zero();

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceExternal()
{
  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // Set to default value  as it is unnecessary for the
  // EvaluateNeumann routine.
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::none);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0, "displacement", GState().GetDisN());
  // -------------------------------------------------------------------------
  // Evaluate brownian specific neumann conditions
  // -------------------------------------------------------------------------
  EvaluateNeumannBrownianDyn(f_ext_np_ptr_, Teuchos::null);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceBrownian()
{
  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // currently a fixed number of matrix and vector pointers are supported
  // set default matrices and vectors
  // -------------------------------------------------------------------------
  std::array<Teuchos::RCP<Epetra_Vector>, 3> eval_vec = {
      Teuchos::null, Teuchos::null, Teuchos::null};
  std::array<Teuchos::RCP<CORE::LINALG::SparseOperator>, 2> eval_mat = {
      Teuchos::null, Teuchos::null};
  // -------------------------------------------------------------------------
  // set brwonian force vector (gets filled on element level)
  // -------------------------------------------------------------------------
  eval_vec[0] = f_brown_np_ptr_;
  // -------------------------------------------------------------------------
  // set action for elements
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::struct_calc_brownianforce);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0, "displacement", GStatePtr()->GetDisNp());
  Discret().SetState(0, "velocity", GState().GetVelNp());
  // -------------------------------------------------------------------------
  // Evaluate Browian (stochastic and damping forces)
  // -------------------------------------------------------------------------
  EvaluateBrownian(eval_mat.data(), eval_vec.data());

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceStiffExternal()
{
  /* so far brownian specific neumann loads need no linearization,
   therefore ApplyForceStiffExternal is equal to ApplyForceExternal*/

  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // Set to default value, as it is unnecessary for the
  // EvaluateNeumann routine.
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::none);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0, "displacement", GState().GetDisN());
  // -------------------------------------------------------------------------
  // Evaluate brownian specific neumann conditions
  // -------------------------------------------------------------------------
  EvaluateNeumannBrownianDyn(f_ext_np_ptr_, Teuchos::null);

  return ok;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::ApplyForceStiffBrownian()
{
  CheckInitSetup();
  bool ok = true;
  // -------------------------------------------------------------------------
  // currently a fixed number of matrix and vector pointers are supported
  // set default matrices and vectors
  // -------------------------------------------------------------------------
  std::array<Teuchos::RCP<Epetra_Vector>, 3> eval_vec = {
      Teuchos::null, Teuchos::null, Teuchos::null};
  std::array<Teuchos::RCP<CORE::LINALG::SparseOperator>, 2> eval_mat = {
      Teuchos::null, Teuchos::null};
  // -------------------------------------------------------------------------
  // set jac matrix and brownian force vector (filled on element level)
  // -------------------------------------------------------------------------
  eval_mat[0] = stiff_brownian_ptr_;
  eval_vec[0] = f_brown_np_ptr_;
  // -------------------------------------------------------------------------
  // set action for elements
  // -------------------------------------------------------------------------
  EvalData().SetActionType(DRT::ELEMENTS::struct_calc_brownianstiff);
  // -------------------------------------------------------------------------
  // set vector values needed by elements
  // -------------------------------------------------------------------------
  Discret().ClearState();
  Discret().SetState(0, "displacement", GStatePtr()->GetDisNp());
  Discret().SetState(0, "velocity", GState().GetVelNp());
  // -------------------------------------------------------------------------
  // Evaluate brownian (stochastic and damping) forces
  EvaluateBrownian(eval_mat.data(), eval_vec.data());

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::EvaluateBrownian(
    Teuchos::RCP<CORE::LINALG::SparseOperator>* eval_mat, Teuchos::RCP<Epetra_Vector>* eval_vec)
{
  CheckInitSetup();

  // todo: just give ParamsInterface to elements (not a parameter list)
  Teuchos::ParameterList p;
  p.set<Teuchos::RCP<DRT::ELEMENTS::ParamsInterface>>("interface", EvalDataPtr());
  // -------------------------------------------------------------------------
  // Evaluate brownian (stochastic and damping) forces on element level
  // -------------------------------------------------------------------------
  EvaluateBrownian(p, eval_mat, eval_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::EvaluateBrownian(Teuchos::ParameterList& p,
    Teuchos::RCP<CORE::LINALG::SparseOperator>* eval_mat, Teuchos::RCP<Epetra_Vector>* eval_vec)
{
  CheckInitSetup();

  // todo: this needs to go, just pass ParamsInterface to elements
  if (p.numParams() > 1)
    dserror(
        "Please use the STR::ELEMENTS::Interface and its derived "
        "classes to set and get parameters.");
  // -------------------------------------------------------------------------
  // Evaluate brownian on element level
  // -------------------------------------------------------------------------
  Discret().Evaluate(p, eval_mat[0], eval_mat[1], eval_vec[0], eval_vec[1], eval_vec[2]);
  Discret().ClearState();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::EvaluateNeumannBrownianDyn(
    Teuchos::RCP<Epetra_Vector> eval_vec, Teuchos::RCP<CORE::LINALG::SparseOperator> eval_mat)
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // get interface pointer
  // -------------------------------------------------------------------------
  Teuchos::RCP<DRT::ELEMENTS::ParamsInterface> interface_ptr = EvalDataPtr();
  // -------------------------------------------------------------------------
  // evaluate brownian specific Neumann boundary conditions
  // -------------------------------------------------------------------------
  //  sm_manager_ptr_->EvaluateNeumannbrownian(interface_ptr,eval_vec,eval_mat);
  DiscretPtr()->ClearState();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::WriteRestart(
    IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  // nothing to do
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::ReadRestart(IO::DiscretizationReader& ioreader)
{
  // nothing to do
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::RunPostComputeX(
    const Epetra_Vector& xold, const Epetra_Vector& dir, const Epetra_Vector& xnew)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::UpdateStepState(const double& timefac_n)
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // add brownian force contributions to the old structural
  // residual state vector
  // -------------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector>& fstructold_ptr = GState().GetFstructureOld();
  fstructold_ptr->Update(timefac_n, *f_brown_np_ptr_, 1.0);
  fstructold_ptr->Update(-timefac_n, *f_ext_np_ptr_, 1.0);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::UpdateStepElement()
{
  // -------------------------------------------------------------------------
  // check if timestep changes according to action dt in input file
  // -------------------------------------------------------------------------
  // todo: this needs to go somewhere else, to a more global/general place
  // (console output at this point is also very unflattering)
  //  sm_manager_ptr_->UpdateTimeAndStepSize((*GStatePtr()->GetDeltaTime())[0],
  //                                           GStatePtr()->GetTimeN());

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::DetermineStressStrain()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::DetermineEnergy()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::DetermineOptionalQuantity()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::OutputStepState(IO::DiscretizationWriter& iowriter) const
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> STR::MODELEVALUATOR::BrownianDyn::GetBlockDofRowMapPtr() const
{
  CheckInitSetup();
  return GState().DofRowMap();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> STR::MODELEVALUATOR::BrownianDyn::GetCurrentSolutionPtr() const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> STR::MODELEVALUATOR::BrownianDyn::GetLastTimeStepSolutionPtr()
    const
{
  // there are no model specific solution entries
  return Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::PostOutput()
{
  CheckInitSetup();
  // -------------------------------------------------------------------------
  // Generate new random forces
  // -------------------------------------------------------------------------
  GenerateGaussianRandomNumbers();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::ResetStepState()
{
  CheckInitSetup();

  if (GState().GetMyRank() == 0)
    std::cout << " NOTE: stochastic forces stay unchanged in case of DIVERCONT" << std::endl;
  /*
    // -------------------------------------------------------------------------
    // Generate new random forces
    // -------------------------------------------------------------------------
    GenerateGaussianRandomNumbers();
    // -------------------------------------------------------------------------
    // Update number of unconverged steps
    // -------------------------------------------------------------------------
  */
  //  sm_manager_ptr_->UpdateNumberOfUnconvergedSteps();

  /* special part in brownian for predictor: initialize disn_ and veln_ with zero;
   * this is necessary only for the following case: Assume that an iteration
   * step did not converge and is repeated with new random numbers; if the
   * failure of convergence lead to disn_ = NaN and veln_ = NaN this would affect
   * also the next trial as e.g. disn_->Update(1.0,*((*dis_)(0)),0.0); would set
   * disn_ to NaN as even 0*NaN = NaN!; this would defeat the purpose of the
   * repeated iterations with new random numbers and has thus to be avoided;
   * therefore we initialized disn_ and veln_ with zero which has no effect
   * in any other case*/
  // todo: is this the right place for this (originally done in brownian predictor,
  // should work as prediction is the next thing that is done)
  GStatePtr()->GetDisNp()->PutScalar(0.0);
  GStatePtr()->GetVelNp()->PutScalar(0.0);
  // we only need this in case we use Lie Group gen alpha and calculate a consistent
  // mass matrix and acc vector (i.e. we are not neglecting inertia forces)
  GStatePtr()->GetAccNp()->PutScalar(0.0);

  GStatePtr()->GetDisNp()->Update(1.0, (*GStatePtr()->GetDisN()), 0.0);
  GStatePtr()->GetVelNp()->Update(1.0, (*GStatePtr()->GetVelN()), 0.0);
  GStatePtr()->GetAccNp()->Update(1.0, (*GStatePtr()->GetAccN()), 0.0);


  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::RandomNumbersPerElement()
{
  CheckInit();
  // -------------------------------------------------------------------------
  // maximal number of random numbers to be generated per time step for
  // any column map element of this processor
  // -------------------------------------------------------------------------
  int randomnumbersperlocalelement = 0;
  // -------------------------------------------------------------------------
  // check maximal number of nodes of an element with stochastic forces
  // on this processor
  // -------------------------------------------------------------------------
  // see whether current element needs more random numbers per time step
  // than any other before
  for (int i = 0; i < discret_ptr_->NumMyColElements(); ++i)
  {
    DRT::ELEMENTS::Beam3Base* beamele =
        dynamic_cast<DRT::ELEMENTS::Beam3Base*>(discret_ptr_->lColElement(i));
    if (beamele != nullptr)
    {
      randomnumbersperlocalelement =
          std::max(randomnumbersperlocalelement, beamele->HowManyRandomNumbersINeed());
    }
    else if (dynamic_cast<DRT::ELEMENTS::Rigidsphere*>(discret_ptr_->lColElement(i)) != nullptr)
    {
      randomnumbersperlocalelement = std::max(randomnumbersperlocalelement,
          dynamic_cast<DRT::ELEMENTS::Rigidsphere*>(discret_ptr_->lColElement(i))
              ->HowManyRandomNumbersINeed());
    }
    else
    {
      dserror("Brownian dynamics simulation not (yet) implemented for this element type.");
    }
  }
  // -------------------------------------------------------------------------
  // so far the maximal number of random numbers required per element
  // has been checked only locally on this processor; now we compare the
  // results of each processor and store the maximal one in
  // maxrandnumelement_
  // -------------------------------------------------------------------------
  DiscretPtr()->Comm().MaxAll(&randomnumbersperlocalelement, &maxrandnumelement_, 1);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MODELEVALUATOR::BrownianDyn::GenerateGaussianRandomNumbers()
{
  CheckInit();

  // only update random numbers and therefore stochastic forces each stochastic time step
  // note: in case of a restart, first stochastic time step can be smaller than
  // brown_dyn_state_data_.browndyn_dt, this is intended
  int browndyn_step = static_cast<int>((GState().GetTimeNp() - (*GStatePtr()->GetDeltaTime())[0]) /
                                           brown_dyn_state_data_.browndyn_dt +
                                       1.0e-8);

  if (browndyn_step == brown_dyn_state_data_.browndyn_step)
    return;
  else
    brown_dyn_state_data_.browndyn_step = browndyn_step;

  // initialize mean value 0 and and standard deviation (2KT / dt)^0.5
  double meanvalue = 0.0;

  // generate gaussian random numbers for parallel use with mean value 0 and
  // standard deviation (2KT / dt)^0.5
  double standarddeviation =
      pow(2.0 * eval_browniandyn_ptr_->KT() / brown_dyn_state_data_.browndyn_dt, 0.5);

  // Set mean value and standard deviation of normal distribution
  DRT::Problem::Instance()->Random()->SetMeanVariance(meanvalue, standarddeviation);
  DRT::Problem::Instance()->Random()->SetRandRange(0.0, 1.0);

  // multivector for stochastic forces evaluated by each element based on row map
  Teuchos::RCP<Epetra_MultiVector> randomnumbersrow = eval_browniandyn_ptr_->GetRandomForces();

  int numele = randomnumbersrow->MyLength();
  int numperele = randomnumbersrow->NumVectors();
  int count = numele * numperele;
  std::vector<double> randvec(count);
  DRT::Problem::Instance()->Random()->Normal(randvec, count);

  // MAXRANDFORCE is a multiple of the standard deviation
  double maxrandforcefac = eval_browniandyn_ptr_->MaxRandForce();
  if (maxrandforcefac == -1.0)
  {
    for (int i = 0; i < numele; ++i)
      for (int j = 0; j < numperele; ++j)
      {
        (*randomnumbersrow)[j][i] = randvec[i * numperele + j];
      }
  }
  else
  {
    for (int i = 0; i < numele; ++i)
      for (int j = 0; j < numperele; ++j)
      {
        (*randomnumbersrow)[j][i] = randvec[i * numperele + j];

        if ((*randomnumbersrow)[j][i] > maxrandforcefac * standarddeviation + meanvalue)
        {
          std::cout << "warning: stochastic force restricted according to MAXRANDFORCE"
                       " this should not happen to often"
                    << std::endl;
          (*randomnumbersrow)[j][i] = maxrandforcefac * standarddeviation + meanvalue;
        }
        else if ((*randomnumbersrow)[j][i] < -maxrandforcefac * standarddeviation + meanvalue)
        {
          std::cout << "warning: stochastic force restricted according to MAXRANDFORCE"
                       " this should not happen to often"
                    << std::endl;
          (*randomnumbersrow)[j][i] = -maxrandforcefac * standarddeviation + meanvalue;
        }
      }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::MODELEVALUATOR::BrownianDyn::IsAnyBeamElementLengthLargerThanMinHalfPBBEdgeLength() const
{
  const int numroweles = Discret().NumMyRowElements();
  const double halfofminimalperiodlength =
      0.5 * eval_browniandyn_ptr_->GetPeriodicBoundingBox()->EdgeLength(0);
  for (int i = 1; i < 3; ++i)
    std::min(halfofminimalperiodlength,
        0.5 * eval_browniandyn_ptr_->GetPeriodicBoundingBox()->EdgeLength(i));

  if (halfofminimalperiodlength != 0.0)
  {
    for (int elelid = 0; elelid < numroweles; ++elelid)
    {
      const DRT::ELEMENTS::Beam3Base* beamele =
          dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(Discret().lRowElement(elelid));

      if (beamele != nullptr and beamele->RefLength() >= halfofminimalperiodlength) return true;
    }
  }

  return false;
}

/*
----------------------------------------------------------------------------*
 | seed all random generators of this object with fixed seed if given and     |
 | with system time otherwise; seedparameter is used only in the first        |
 | case to calculate the actual seed variable based on some given fixed       |
 | seed value; note that seedparameter may be any integer, but has to be      |
 | been set in a deterministic way so that it for a certain call of this      |
 | method at a certain point in the program always the same number            |
 | whenever the program is used                                               |
 *----------------------------------------------------------------------------
void STR::MODELEVALUATOR::BrownianDyn::SeedRandomGenerator()
{
  CheckInit();

  const int    stepn  = GStatePtr()->GetStepN() + 1;
  const double timenp = GStatePtr()->GetTimeNp();
  const double dt     = (*GStatePtr()->GetDeltaTime())[0];
  const int myrank = GState().GetMyRank();

  // -----------------------------------------------------------------------
  // Decide if random numbers should change in every time step...
  // read time interval within the random numbers remain constant (-1.0 means no
  // prescribed time interval). This means new random numbers every
  // randnumtimeinc seconds.
  // -----------------------------------------------------------------------
  if( rand_data_.time_interv_with_const_rn == -1.0 )
  {
    // new random numbers every time step (same in each program start though)
    if ( rand_data_.randseed >= 0 )
      rand_data_.seedvariable = ( rand_data_.randseed + stepn ) * ( myrank + 1 );
    // else
    // set seed according to system time and different for each processor
    // once in the beginning (done in ReadParameter globalproblem.cpp)
    // in this case we have different random numbers in each program start
    // and time step
  }
  //...or only every time_interv_with_const_rn seconds
  else
  {
    // this variable changes every time_interv_with_const_rn seconds
    int seed_differs_every_time_int =
        static_cast<int>( (timenp - dt ) / rand_data_.time_interv_with_const_rn + 1.0e-8 );

    // same each program start
    if ( rand_data_.randseed >= 0 )
    {
      rand_data_.seedvariable =
          ( rand_data_.randseed + seed_differs_every_time_int ) * ( myrank + 1 );
    }
    // .. different each program start
    else if( seed_differs_every_time_int != rand_data_.seed_differs_every_time_int )
    {
      rand_data_.seedvariable = static_cast<int>( time(nullptr) ) + 27 * ( myrank + 1 );
      rand_data_.seed_differs_every_time_int = seed_differs_every_time_int;
    }
  }
  // -----------------------------------------------------------------------
  // seed random number generator and set uni range
  // -----------------------------------------------------------------------
  DRT::Problem::Instance()->Random()->SetRandSeed( static_cast<unsigned int>(
rand_data_.seedvariable ) ); DRT::Problem::Instance()->Random()->SetRandRange( 0.0, 1.0);

}*/
