/*----------------------------------------------------------------------*/
/*! \file
 \brief a wrapper for porous multiphase flow algorithms

   \level 3

 *----------------------------------------------------------------------*/

#include "baci_adapter_porofluidmultiphase_wrapper.H"

#include "baci_inpar_validparameters.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_linear_solver_method_linalg.H"
#include "baci_porofluidmultiphase_timint_implicit.H"
#include "baci_porofluidmultiphase_timint_ost.H"

#include <Teuchos_StandardParameterEntryValidators.hpp>


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::PoroFluidMultiphaseWrapper::PoroFluidMultiphaseWrapper(
    Teuchos::RCP<PoroFluidMultiphase> porofluid)
    : porofluid_(porofluid)
{
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Init(const bool isale,  ///< ALE flag
    const int nds_disp,           ///< number of dofset associated with displacements
    const int nds_vel,            ///< number of dofset associated with fluid velocities
    const int nds_solidpressure,  ///< number of dofset associated with solid pressure
    const int
        ndsporofluid_scatra,  ///< number of dofset associated with scalar on fluid discretization
    const std::map<int, std::set<int>>* nearbyelepairs  ///< possible interaction partners between
                                                        ///< porofluid and artery discretization
)
{
  // initialize algorithm for specific time-integration scheme
  porofluid_->Init(
      isale, nds_disp, nds_vel, nds_solidpressure, ndsporofluid_scatra, nearbyelepairs);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<DRT::ResultTest> ADAPTER::PoroFluidMultiphaseWrapper::CreateFieldTest()
{
  return porofluid_->CreateFieldTest();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> ADAPTER::PoroFluidMultiphaseWrapper::DofRowMap(unsigned nds) const
{
  return porofluid_->DofRowMap(nds);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> ADAPTER::PoroFluidMultiphaseWrapper::ArteryDofRowMap() const
{
  return porofluid_->ArteryDofRowMap();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase>
ADAPTER::PoroFluidMultiphaseWrapper::ArteryPorofluidSysmat() const
{
  return porofluid_->ArteryPorofluidSysmat();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<DRT::Discretization> ADAPTER::PoroFluidMultiphaseWrapper::Discretization() const
{
  return porofluid_->Discretization();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::ReadRestart(int restart)
{
  return porofluid_->ReadRestart(restart);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::ApplyMeshMovement(
    Teuchos::RCP<const Epetra_Vector> dispnp  //!< displacement vector
)
{
  porofluid_->ApplyMeshMovement(dispnp);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::SetVelocityField(Teuchos::RCP<const Epetra_Vector> vel)
{
  porofluid_->SetVelocityField(vel);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::SetState(
    unsigned nds, const std::string& name, Teuchos::RCP<const Epetra_Vector> state)
{
  porofluid_->SetState(nds, name, state);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::SetScatraSolution(
    unsigned nds, Teuchos::RCP<const Epetra_Vector> scalars)
{
  SetState(nds, "scalars", scalars);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Phinp() const
{
  return porofluid_->Phinp();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Phin() const
{
  return porofluid_->Phin();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::SolidPressure() const
{
  return porofluid_->SolidPressure();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Pressure() const
{
  return porofluid_->Pressure();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::Saturation() const
{
  return porofluid_->Saturation();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::ValidVolFracSpecDofs() const
{
  return porofluid_->ValidVolFracSpecDofs();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_MultiVector> ADAPTER::PoroFluidMultiphaseWrapper::Flux() const
{
  return porofluid_->Flux();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int ADAPTER::PoroFluidMultiphaseWrapper::GetDofSetNumberOfSolidPressure() const
{
  return porofluid_->GetDofSetNumberOfSolidPressure();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::TimeLoop() { porofluid_->TimeLoop(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::PrepareTimeStep() { porofluid_->PrepareTimeStep(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Output() { porofluid_->Output(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Update() { porofluid_->Update(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::EvaluateErrorComparedToAnalyticalSol()
{
  porofluid_->EvaluateErrorComparedToAnalyticalSol();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Solve() { porofluid_->Solve(); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::PrepareTimeLoop() { porofluid_->PrepareTimeLoop(); }
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Teuchos::RCP<const CORE::LINALG::MapExtractor>
ADAPTER::PoroFluidMultiphaseWrapper::GetDBCMapExtractor() const
{
  return porofluid_->GetDBCMapExtractor();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::RHS() const
{
  return porofluid_->RHS();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> ADAPTER::PoroFluidMultiphaseWrapper::ArteryPorofluidRHS() const
{
  return porofluid_->ArteryPorofluidRHS();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::UpdateIter(const Teuchos::RCP<const Epetra_Vector> inc)
{
  porofluid_->UpdateIter(inc);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::ReconstructPressuresAndSaturations()
{
  porofluid_->ReconstructPressuresAndSaturations();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::ReconstructFlux() { porofluid_->ReconstructFlux(); }
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::Evaluate() { porofluid_->Evaluate(); }
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::SparseMatrix> ADAPTER::PoroFluidMultiphaseWrapper::SystemMatrix()
{
  return porofluid_->SystemMatrix();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::AssembleFluidStructCouplingMat(
    Teuchos::RCP<CORE::LINALG::SparseOperator> k_fs)
{
  return porofluid_->AssembleFluidStructCouplingMat(k_fs);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::PoroFluidMultiphaseWrapper::AssembleFluidScatraCouplingMat(
    Teuchos::RCP<CORE::LINALG::SparseOperator> k_pfs)
{
  return porofluid_->AssembleFluidScatraCouplingMat(k_pfs);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<ADAPTER::ArtNet> ADAPTER::PoroFluidMultiphaseWrapper::ArtNetTimInt()
{
  return porofluid_->ArtNetTimInt();
}
