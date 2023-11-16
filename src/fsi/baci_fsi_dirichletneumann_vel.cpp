/*----------------------------------------------------------------------*/
/*! \file
\file fsi_dirichletneumann_vel.cpp

\brief Solve FSI problems using a Dirichlet-Neumann partitioned approach


\level 3
*/
/*----------------------------------------------------------------------*/


#include "baci_fsi_dirichletneumann_vel.H"

#include "baci_adapter_fld_fbi_movingboundary.H"
#include "baci_adapter_str_fbiwrapper.H"
#include "baci_binstrategy.H"
#include "baci_fbi_adapter_constraintbridge.H"
#include "baci_fbi_beam_to_fluid_meshtying_output_writer.H"
#include "baci_fbi_beam_to_fluid_meshtying_params.H"
#include "baci_fbi_constraintenforcer.H"
#include "baci_fbi_constraintenforcer_factory.H"
#include "baci_inpar_fbi.H"
#include "baci_inpar_fsi.H"
#include "baci_io_control.H"
#include "baci_lib_globalproblem.H"
#include "baci_utils_exceptions.H"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <iostream>


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::DirichletNeumannVel::DirichletNeumannVel(const Epetra_Comm& comm)
    : DirichletNeumann(comm),
      constraint_manager_(ADAPTER::ConstraintEnforcerFactory::CreateEnforcer(
          DRT::Problem::Instance()->FSIDynamicParams(), DRT::Problem::Instance()->FBIParams()))
{
  // empty constructor
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::DirichletNeumannVel::Setup()
{
  // call setup of base class
  FSI::DirichletNeumann::Setup();
  const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
  const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");
  if (DRT::INPUT::IntegralValue<int>(fsipart, "COUPVARIABLE") == INPAR::FSI::CoupVarPart::disp)
    dserror("Please set the fsi coupling variable to Velocity or Force!\n");
  SetKinematicCoupling(
      DRT::INPUT::IntegralValue<int>(fsipart, "COUPVARIABLE") == INPAR::FSI::CoupVarPart::vel);
  if (Teuchos::rcp_dynamic_cast<ADAPTER::FBIStructureWrapper>(StructureField(), true) ==
      Teuchos::null)
  {
    dserror("Something went very wrong here! You should have a FBIStructureWrapper!\n");
  }
  if (Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(MBFluidField(), true) == Teuchos::null)
  {
    dserror("Something went very wrong here! You should have a FBIFluidMB adapter!\n");
  }
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::DirichletNeumannVel::FluidOp(
    Teuchos::RCP<Epetra_Vector> ivel, const FillType fillFlag)
{
  const Teuchos::ParameterList& fbi = DRT::Problem::Instance()->FBIParams();

  FSI::Partitioned::FluidOp(ivel, fillFlag);

  if (fillFlag == User)
  {
    dserror("Not implemented!\n");
    return FluidToStruct(MBFluidField()->RelaxationSolve(Teuchos::null, Dt()));
  }
  else
  {
    // A rather simple hack. We need something better!
    const int itemax = MBFluidField()->Itemax();

    if (fillFlag == MF_Res and mfresitemax_ > 0) MBFluidField()->SetItemax(mfresitemax_ + 1);

    MBFluidField()->NonlinearSolve(Teuchos::null, Teuchos::null);

    MBFluidField()->SetItemax(itemax);

    if (Teuchos::getIntegralValue<INPAR::FBI::BeamToFluidCoupling>(fbi, "COUPLING") !=
            INPAR::FBI::BeamToFluidCoupling::fluid &&
        fbi.get<int>("STARTSTEP") < Step())
    {
      constraint_manager_->RecomputeCouplingWithoutPairCreation();
    }

    return FluidToStruct(ivel);
  }
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::DirichletNeumannVel::StructOp(
    Teuchos::RCP<Epetra_Vector> iforce, const FillType fillFlag)
{
  FSI::Partitioned::StructOp(iforce, fillFlag);

  const Teuchos::ParameterList& fbi = DRT::Problem::Instance()->FBIParams();
  if (!(Teuchos::getIntegralValue<INPAR::FBI::BeamToFluidCoupling>(fbi, "COUPLING") ==
          INPAR::FBI::BeamToFluidCoupling::fluid) &&
      fbi.get<int>("STARTSTEP") < Step())
  {
    if (not use_old_structure_)
      StructureField()->ApplyInterfaceForces(iforce);
    else
      dserror(
          "Fluid beam interaction is not tested with the old structure time. You should not be "
          "here! Change the IntStrategy in your Input file to Standard.\n");
  }

  StructureField()->Solve();
  StructureField()->WriteGmshStrucOutputStep();

  if (fbi.get<int>("STARTSTEP") < Step())
  {
    constraint_manager_->PrepareFluidSolve();
    constraint_manager_->Evaluate();
    if (!(Teuchos::getIntegralValue<INPAR::FBI::BeamToFluidCoupling>(fbi, "COUPLING") ==
            INPAR::FBI::BeamToFluidCoupling::solid))
      Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(MBFluidField(), true)->ResetExternalForces();
  }

  return StructToFluid(iforce);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::DirichletNeumannVel::InitialGuess()
{
  if (GetKinematicCoupling())
  {
    // predict velocity
    // For now, we want to use no predictor. This function returns the current interface velocity.
    if (Teuchos::rcp_dynamic_cast<ADAPTER::FBIStructureWrapper>(StructureField(), true) !=
        Teuchos::null)
      return Teuchos::rcp_dynamic_cast<ADAPTER::FBIStructureWrapper>(StructureField(), true)
          ->PredictInterfaceVelnp();
  }
  else
  {
    const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
    const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");
    if (DRT::INPUT::IntegralValue<int>(fsipart, "PREDICTOR") != 1)
    {
      dserror(
          "unknown interface force predictor '%s'", fsipart.get<std::string>("PREDICTOR").c_str());
    }
    return InterfaceForce();
  }
  dserror("Something went very wrong here!\n");
  return Teuchos::null;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

Teuchos::RCP<Epetra_Vector> FSI::DirichletNeumannVel::FluidToStruct(Teuchos::RCP<Epetra_Vector> iv)
{
  return constraint_manager_->FluidToStructure();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

Teuchos::RCP<Epetra_Vector> FSI::DirichletNeumannVel::StructToFluid(Teuchos::RCP<Epetra_Vector> iv)
{
  return constraint_manager_->StructureToFluid(Step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::DirichletNeumannVel::Output()
{
  FSI::DirichletNeumann::Output();
  constraint_manager_->Output(Time(), Step());
  visualization_output_writer_->WriteOutputRuntime(constraint_manager_, Step(), Time());
  StructureField()->Discretization()->Writer()->ClearMapCache();
  MBFluidField()->Discretization()->Writer()->ClearMapCache();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::DirichletNeumannVel::Timeloop(
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& interface)
{
  constraint_manager_->Setup(StructureField(), MBFluidField());
  if (GetKinematicCoupling()) constraint_manager_->PrepareFluidSolve();
  visualization_output_writer_ =
      Teuchos::rcp(new BEAMINTERACTION::BeamToFluidMeshtyingVtkOutputWriter());
  visualization_output_writer_->Init();
  visualization_output_writer_->Setup(
      Teuchos::rcp_dynamic_cast<ADAPTER::FBIStructureWrapper>(StructureField(), true)->GetIOData(),
      constraint_manager_->GetBridge()->GetParams()->GetVisualizationOuputParamsPtr(), Time());
  constraint_manager_->Evaluate();
  if (GetKinematicCoupling()) StructToFluid(Teuchos::null);

  FSI::Partitioned::Timeloop(interface);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FSI::DirichletNeumannVel::SetBinning(Teuchos::RCP<BINSTRATEGY::BinningStrategy> binning)
{
  constraint_manager_->SetBinning(binning);
};
