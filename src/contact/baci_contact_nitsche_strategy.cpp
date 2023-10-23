/*---------------------------------------------------------------------*/
/*! \file
\brief Nitsche contact solving strategy

\level 3


*/
/*---------------------------------------------------------------------*/

#include "baci_contact_nitsche_strategy.H"

#include "baci_contact_interface.H"
#include "baci_contact_nitsche_utils.H"
#include "baci_contact_paramsinterface.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_so3_plast_ssn.H"

#include <Epetra_CrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Teuchos_Time.hpp>

/*----------------------------------------------------------------------*
 | global evaluation method called from time integrator     seitz 10/16 |
 *----------------------------------------------------------------------*/
void CONTACT::CoNitscheStrategy::ApplyForceStiffCmt(Teuchos::RCP<Epetra_Vector> dis,
    Teuchos::RCP<CORE::LINALG::SparseOperator>& kt, Teuchos::RCP<Epetra_Vector>& f, const int step,
    const int iter, bool predictor)
{
  // mortar initialization and evaluation
  SetState(MORTAR::state_new_displacement, *dis);

  // just a Nitsche-version
  Teuchos::RCP<Epetra_FEVector> fc = Teuchos::rcp(new Epetra_FEVector(f->Map()));
  Teuchos::RCP<CORE::LINALG::SparseMatrix> kc = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
      (dynamic_cast<Epetra_CrsMatrix*>(&(*kt->EpetraOperator())))->RowMap(), 100, true, false,
      CORE::LINALG::SparseMatrix::FE_MATRIX));

  // Evaluation for all interfaces
  for (const auto& interface : interface_)
  {
    interface->Initialize();
    interface->Evaluate(0, step_, iter_);
    for (int e = 0; e < interface->Discret().ElementColMap()->NumMyElements(); ++e)
    {
      auto* mele = dynamic_cast<MORTAR::MortarElement*>(
          interface->Discret().gElement(interface->Discret().ElementColMap()->GID(e)));
      mele->GetNitscheContainer().AssembleRHS(mele, DRT::UTILS::VecBlockType::displ, fc);
      mele->GetNitscheContainer().AssembleMatrix(mele, DRT::UTILS::MatBlockType::displ_displ, kc);
    }
  }

  // now we also did this state
  curr_state_eval_ = true;

  if (fc->GlobalAssemble(Add, false) != 0) dserror("GlobalAssemble failed");
  // add negative contact force here since the time integrator handed me a rhs!
  if (f->Update(-1., *fc, 1.)) dserror("update went wrong");
  dynamic_cast<Epetra_FECrsMatrix&>(*kc->EpetraMatrix()).GlobalAssemble(true, Add);
  kt->UnComplete();
  kt->Add(*kc, false, 1., 1.);
  kt->Complete();
}


/*----------------------------------------------------------------------*
 |  read restart information for contact                     seitz 10/16|
 *----------------------------------------------------------------------*/
void CONTACT::CoNitscheStrategy::DoReadRestart(IO::DiscretizationReader& reader,
    Teuchos::RCP<const Epetra_Vector> dis, Teuchos::RCP<CONTACT::ParamsInterface> cparams_ptr)
{
  // check whether this is a restart with contact of a previously
  // non-contact simulation run (if yes, we have to be careful not
  // to try to read certain, in this case non-existing, vectors
  // such as the activetoggle or sliptoggle vectors, but rather
  // initialize the restart active and slip sets as being empty)
  bool restartwithcontact = DRT::INPUT::IntegralValue<int>(Params(), "RESTART_WITH_CONTACT");
  if (restartwithcontact) dserror("not supported for nitsche contact");

  // set restart displacement state
  SetState(MORTAR::state_new_displacement, *dis);
  SetState(MORTAR::state_old_displacement, *dis);

  // Evaluation for all interfaces
  for (const auto& interface : interface_) interface->Initialize();

  if (friction_)
  {
    for (const auto& interface : interface_)
    {
      interface->EvaluateNodalNormals();
      interface->ExportNodalNormals();
    }
    StoreToOld(MORTAR::StrategyBase::n_old);
  }

  if (DRT::INPUT::IntegralValue<int>(Params(), "NITSCHE_PENALTY_ADAPTIVE"))
    UpdateTraceIneqEtimates();
}

void CONTACT::CoNitscheStrategy::SetState(
    const enum MORTAR::StateType& statename, const Epetra_Vector& vec)
{
  if (statename == MORTAR::state_new_displacement)
  {
    double inf_delta = 0.;
    if (curr_state_ == Teuchos::null)
    {
      curr_state_ = Teuchos::rcp(new Epetra_Vector(vec));
      inf_delta = 1.e12;
    }
    else
    {
      Epetra_Vector delta(vec);
      delta.Update(-1., *curr_state_, 1.);
      delta.NormInf(&inf_delta);
    }
    if (inf_delta < 1.e-12)
      return;
    else
    {
      curr_state_eval_ = false;
      (*curr_state_) = vec;
      CoAbstractStrategy::SetState(statename, vec);
      SetParentState(statename, vec);
    }
  }
  else
  {
    curr_state_eval_ = false;
    CoAbstractStrategy::SetState(statename, vec);
  }
}

/*------------------------------------------------------------------------*
 |                                                             seitz 10/16|
 *------------------------------------------------------------------------*/
void CONTACT::CoNitscheStrategy::SetParentState(
    const enum MORTAR::StateType& statename, const Epetra_Vector& vec)
{
  Teuchos::RCP<DRT::Discretization> dis = DRT::Problem::Instance()->GetDis("structure");
  if (dis == Teuchos::null) dserror("didn't get my discretization");
  if (statename == MORTAR::state_new_displacement || statename == MORTAR::state_svelocity)
  {
    Teuchos::RCP<Epetra_Vector> global = Teuchos::rcp(new Epetra_Vector(*dis->DofColMap(), true));
    CORE::LINALG::Export(vec, *global);

    // set state on interfaces
    for (const auto& interface : interface_)
    {
      DRT::Discretization& idiscret = interface->Discret();

      for (int j = 0; j < interface->Discret().ElementColMap()->NumMyElements(); ++j)
      {
        const int gid = interface->Discret().ElementColMap()->GID(j);

        auto* ele = dynamic_cast<MORTAR::MortarElement*>(idiscret.gElement(gid));

        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        // this gets values in local order
        ele->ParentElement()->LocationVector(*dis, lm, lmowner, lmstride);

        std::vector<double> myval;
        DRT::UTILS::ExtractMyValues(*global, myval, lm);

        switch (statename)
        {
          case MORTAR::state_new_displacement:
          {
            ele->MoData().ParentDisp() = myval;
            ele->MoData().ParentDof() = lm;
            break;
          }
          case MORTAR::state_svelocity:
          {
            ele->MoData().ParentVel() = myval;
            break;
          }
          default:
            dserror("unknown statename");
        }
      }
    }
  }
}

void CONTACT::CoNitscheStrategy::EvalForce(CONTACT::ParamsInterface& cparams)
{
  Integrate(cparams);
}

void CONTACT::CoNitscheStrategy::EvalForceStiff(CONTACT::ParamsInterface& cparams)
{
  Integrate(cparams);
}

void CONTACT::CoNitscheStrategy::Reset(
    const CONTACT::ParamsInterface& cparams, const Epetra_Vector& dispnp, const Epetra_Vector& xnew)
{
  SetState(MORTAR::state_new_displacement, dispnp);
}

void CONTACT::CoNitscheStrategy::RunPostComputeX(const CONTACT::ParamsInterface& cparams,
    const Epetra_Vector& xold, const Epetra_Vector& dir, const Epetra_Vector& xnew)
{
  // do nothing
}

void CONTACT::CoNitscheStrategy::Integrate(const CONTACT::ParamsInterface& cparams)
{
  // we already did this displacement state
  if (curr_state_eval_) return;

  // time measurement (on each processor)
  const double t_start = Teuchos::Time::wallTime();

  // Evaluation for all interfaces
  for (const auto& interface : interface_)
  {
    interface->InterfaceParams().set<double>("TIMESTEP", cparams.GetDeltaTime());
    interface->Initialize();
    interface->Evaluate(0, step_, iter_);

    // store required integration time
    inttime_ += interface->Inttime();
  }

  // check the parallel distribution
  CheckParallelDistribution(t_start);

  // now we also did this state
  curr_state_eval_ = true;

  // ... and we can assemble the matric and rhs
  fc_ = CreateRhsBlockPtr(DRT::UTILS::VecBlockType::displ);
  kc_ = CreateMatrixBlockPtr(DRT::UTILS::MatBlockType::displ_displ);
}

Teuchos::RCP<Epetra_FEVector> CONTACT::CoNitscheStrategy::SetupRhsBlockVec(
    const enum DRT::UTILS::VecBlockType& bt) const
{
  switch (bt)
  {
    case DRT::UTILS::VecBlockType::displ:
      return Teuchos::rcp(
          new Epetra_FEVector(*DRT::Problem::Instance()->GetDis("structure")->DofRowMap()));
    default:
      dserror("you should not be here");
      break;
  }
  return Teuchos::null;
}

Teuchos::RCP<Epetra_FEVector> CONTACT::CoNitscheStrategy::CreateRhsBlockPtr(
    const enum DRT::UTILS::VecBlockType& bt) const
{
  if (!curr_state_eval_) dserror("you didn't evaluate this contact state first");

  Teuchos::RCP<Epetra_FEVector> fc = SetupRhsBlockVec(bt);

  for (const auto& interface : interface_)
  {
    for (int e = 0; e < interface->Discret().ElementColMap()->NumMyElements(); ++e)
    {
      auto* mele = dynamic_cast<MORTAR::MortarElement*>(
          interface->Discret().gElement(interface->Discret().ElementColMap()->GID(e)));
      mele->GetNitscheContainer().AssembleRHS(mele, bt, fc);
    }
  }
  if (fc->GlobalAssemble(Add, false) != 0) dserror("GlobalAssemble failed");

  return fc;
}

Teuchos::RCP<const Epetra_Vector> CONTACT::CoNitscheStrategy::GetRhsBlockPtr(
    const enum DRT::UTILS::VecBlockType& bt) const
{
  if (!curr_state_eval_) dserror("you didn't evaluate this contact state first");
  switch (bt)
  {
    case DRT::UTILS::VecBlockType::displ:
      return Teuchos::rcp(new Epetra_Vector(Copy, *(fc_), 0));
    case DRT::UTILS::VecBlockType::constraint:
      return Teuchos::null;
    default:
      dserror("GetRhsBlockPtr: your type is no treated properly!");
      break;
  }

  return Teuchos::null;
}

Teuchos::RCP<CORE::LINALG::SparseMatrix> CONTACT::CoNitscheStrategy::SetupMatrixBlockPtr(
    const enum DRT::UTILS::MatBlockType& bt)
{
  switch (bt)
  {
    case DRT::UTILS::MatBlockType::displ_displ:
      return Teuchos::rcp(new CORE::LINALG::SparseMatrix(
          *Teuchos::rcpFromRef<const Epetra_Map>(
              *DRT::Problem::Instance()->GetDis("structure")->DofRowMap()),
          100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));
    default:
      dserror("you should not be here");
      break;
  }
  return Teuchos::null;
}

void CONTACT::CoNitscheStrategy::CompleteMatrixBlockPtr(
    const enum DRT::UTILS::MatBlockType& bt, Teuchos::RCP<CORE::LINALG::SparseMatrix> kc)
{
  switch (bt)
  {
    case DRT::UTILS::MatBlockType::displ_displ:
      kc->Complete();
      break;
    default:
      dserror("you should not be here");
      break;
  }
}

Teuchos::RCP<CORE::LINALG::SparseMatrix> CONTACT::CoNitscheStrategy::CreateMatrixBlockPtr(
    const enum DRT::UTILS::MatBlockType& bt)
{
  if (!curr_state_eval_) dserror("you didn't evaluate this contact state first");

  Teuchos::RCP<CORE::LINALG::SparseMatrix> kc = SetupMatrixBlockPtr(bt);

  for (const auto& interface : interface_)
  {
    for (int e = 0; e < interface->Discret().ElementColMap()->NumMyElements(); ++e)
    {
      auto* mele = dynamic_cast<MORTAR::MortarElement*>(
          interface->Discret().gElement(interface->Discret().ElementColMap()->GID(e)));
      mele->GetNitscheContainer().AssembleMatrix(mele, bt, kc);
    }
  }

  CompleteMatrixBlockPtr(bt, kc);

  return kc;
}

Teuchos::RCP<CORE::LINALG::SparseMatrix> CONTACT::CoNitscheStrategy::GetMatrixBlockPtr(
    const enum DRT::UTILS::MatBlockType& bt, const CONTACT::ParamsInterface* cparams) const
{
  if (!curr_state_eval_) dserror("you didn't evaluate this contact state first");

  if (bt == DRT::UTILS::MatBlockType::displ_displ)
    return kc_;
  else
    dserror("GetMatrixBlockPtr: your type is no treated properly!");

  return Teuchos::null;
}

void CONTACT::CoNitscheStrategy::Setup(bool redistributed, bool init)
{
  // we need to init the isselfcontact_ flag here, as we do not want to call the CoAbstractStrategy
  if (init)
  {
    // set potential global self contact status
    // (this is TRUE if at least one contact interface is a self contact interface)
    bool selfcontact = false;
    for (const auto& interface : Interfaces())
      if (interface->SelfContact()) selfcontact = true;

    if (selfcontact) isselfcontact_ = true;
  }
  ReconnectParentElements();
  curr_state_ = Teuchos::null;
  curr_state_eval_ = false;
}

void CONTACT::CoNitscheStrategy::UpdateTraceIneqEtimates()
{
  auto NitWgt =
      DRT::INPUT::IntegralValue<INPAR::CONTACT::NitscheWeighting>(Params(), "NITSCHE_WEIGHTING");
  for (const auto& interface : interface_)
  {
    for (int e = 0; e < interface->Discret().ElementColMap()->NumMyElements(); ++e)
    {
      auto* mele = dynamic_cast<MORTAR::MortarElement*>(
          interface->Discret().gElement(interface->Discret().ElementColMap()->GID(e)));
      if (NitWgt == INPAR::CONTACT::NitWgt_slave && !mele->IsSlave()) continue;
      if (NitWgt == INPAR::CONTACT::NitWgt_master && mele->IsSlave()) continue;
      mele->EstimateNitscheTraceMaxEigenvalueCombined();
    }
  }
}

void CONTACT::CoNitscheStrategy::Update(Teuchos::RCP<const Epetra_Vector> dis)
{
  if (DRT::INPUT::IntegralValue<int>(Params(), "NITSCHE_PENALTY_ADAPTIVE"))
    UpdateTraceIneqEtimates();
  if (friction_)
  {
    StoreToOld(MORTAR::StrategyBase::n_old);
    SetState(MORTAR::state_old_displacement, *dis);
  }
}

void CONTACT::CoNitscheStrategy::EvaluateReferenceState()
{
  if (friction_)
  {
    for (const auto& interface : interface_)
    {
      interface->EvaluateNodalNormals();
      interface->ExportNodalNormals();
    }
    StoreToOld(MORTAR::StrategyBase::n_old);
  }

  UpdateTraceIneqEtimates();
}


/*----------------------------------------------------------------------------------------------*
 |  Reconnect Contact Element -- Parent Element Pointers (required for restart)       ager 04/16|
 *---------------------------------------------------------------------------------------------*/
void CONTACT::CoNitscheStrategy::ReconnectParentElements()
{
  Teuchos::RCP<DRT::Discretization> voldis = DRT::Problem::Instance()->GetDis("structure");

  for (const auto& contact_interface : ContactInterfaces())
  {
    const Epetra_Map* elecolmap = voldis->ElementColMap();

    const Epetra_Map* ielecolmap = contact_interface->Discret().ElementColMap();

    for (int i = 0; i < ielecolmap->NumMyElements(); ++i)
    {
      const int gid = ielecolmap->GID(i);

      DRT::Element* ele = contact_interface->Discret().gElement(gid);
      if (!ele) dserror("Cannot find element with gid %", gid);
      auto* faceele = dynamic_cast<DRT::FaceElement*>(ele);

      const int volgid = faceele->ParentElementId();
      if (elecolmap->LID(volgid) == -1)  // Volume Discretization has not Element
        dserror(
            "CoManager::ReconnectParentElements: Element %d does not exist on this Proc!", volgid);

      DRT::Element* vele = voldis->gElement(volgid);
      if (!vele) dserror("Cannot find element with gid %", volgid);

      faceele->SetParentMasterElement(vele, faceele->FaceParentNumber());

      auto* vele_plast =
          dynamic_cast<DRT::ELEMENTS::So3_Plast<DRT::Element::DiscretizationType::hex8>*>(vele);
      if (vele_plast) vele_plast->SetIsNitscheContactEle(true);
    }
  }
}
