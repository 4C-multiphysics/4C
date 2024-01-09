/*----------------------------------------------------------------------*/
/*! \file

\brief Structural adapter for FSI problems containing the interface
       and methods dependent on the interface


\level 1
*/

#include "baci_adapter_str_fsiwrapper.H"

#include "baci_fsi_str_model_evaluator_partitioned.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_linalg_utils_sparse_algebra_create.H"
#include "baci_structure_aux.H"

BACI_NAMESPACE_OPEN

namespace
{
  bool PrestressIsActive(const double currentTime)
  {
    INPAR::STR::PreStress pstype = Teuchos::getIntegralValue<INPAR::STR::PreStress>(
        DRT::Problem::Instance()->StructuralDynamicParams(), "PRESTRESS");
    const double pstime =
        DRT::Problem::Instance()->StructuralDynamicParams().get<double>("PRESTRESSTIME");
    return pstype != INPAR::STR::PreStress::none && currentTime <= pstime + 1.0e-15;
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::FSIStructureWrapper::FSIStructureWrapper(Teuchos::RCP<Structure> structure)
    : StructureWrapper(structure)
{
  // set-up FSI interface
  interface_ = Teuchos::rcp(new STR::MapExtractor);

  if (DRT::Problem::Instance()->GetProblemType() != ProblemType::fpsi)
    interface_->Setup(*Discretization(), *Discretization()->DofRowMap());
  else
    interface_->Setup(*Discretization(), *Discretization()->DofRowMap(),
        true);  // create overlapping maps for fpsi problem

  const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
  const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");
  predictor_ = INPUT::IntegralValue<int>(fsipart, "PREDICTOR");
}

/*------------------------------------------------------------------------------------*
 * Rebuild FSI interface on structure side                              sudhakar 09/13
 * This is necessary if elements are added/deleted from interface
 *------------------------------------------------------------------------------------*/
void ADAPTER::FSIStructureWrapper::RebuildInterface()
{
  interface_ = Teuchos::rcp(new STR::MapExtractor);
  interface_->Setup(*Discretization(), *Discretization()->DofRowMap());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::FSIStructureWrapper::UseBlockMatrix()
{
  StructureWrapper::UseBlockMatrix(interface_, interface_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::FSIStructureWrapper::RelaxationSolve(
    Teuchos::RCP<Epetra_Vector> iforce)
{
  ApplyInterfaceForces(iforce);
  FSIModelEvaluator()->SetIsRelaxationSolve(true);
  Teuchos::RCP<const Epetra_Vector> idisi = FSIModelEvaluator()->SolveRelaxationLinear(structure_);
  FSIModelEvaluator()->SetIsRelaxationSolve(false);

  // we are just interested in the incremental interface displacements
  return interface_->ExtractFSICondVector(idisi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::FSIStructureWrapper::PredictInterfaceDispnp()
{
  // prestressing business
  Teuchos::RCP<Epetra_Vector> idis;

  switch (predictor_)
  {
    case 1:
    {
      // d(n)
      // respect Dirichlet conditions at the interface (required for pseudo-rigid body)
      if (PrestressIsActive(Time()))
      {
        idis = Teuchos::rcp(new Epetra_Vector(*interface_->FSICondMap(), true));
      }
      else
      {
        idis = interface_->ExtractFSICondVector(Dispn());
      }
      break;
    }
    case 2:
      // d(n)+dt*(1.5*v(n)-0.5*v(n-1))
      dserror("interface velocity v(n-1) not available");
      break;
    case 3:
    {
      // d(n)+dt*v(n)
      if (PrestressIsActive(Time()))
        dserror("only constant interface predictor useful for prestressing");

      double dt = Dt();

      idis = interface_->ExtractFSICondVector(Dispn());
      Teuchos::RCP<Epetra_Vector> ivel = interface_->ExtractFSICondVector(Veln());

      idis->Update(dt, *ivel, 1.0);
      break;
    }
    case 4:
    {
      // d(n)+dt*v(n)+0.5*dt^2*a(n)
      if (PrestressIsActive(Time()))
        dserror("only constant interface predictor useful for prestressing");

      double dt = Dt();

      idis = interface_->ExtractFSICondVector(Dispn());
      Teuchos::RCP<Epetra_Vector> ivel = interface_->ExtractFSICondVector(Veln());
      Teuchos::RCP<Epetra_Vector> iacc = interface_->ExtractFSICondVector(Accn());

      idis->Update(dt, *ivel, 0.5 * dt * dt, *iacc, 1.0);
      break;
    }
    default:
      dserror("unknown interface displacement predictor '%s'", DRT::Problem::Instance()
                                                                   ->FSIDynamicParams()
                                                                   .sublist("PARTITIONED SOLVER")
                                                                   .get<std::string>("PREDICTOR")
                                                                   .c_str());
      break;
  }

  return idis;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::FSIStructureWrapper::ExtractInterfaceDispn()
{
  dsassert(interface_->FullMap()->PointSameAs(Dispn()->Map()),
      "Full map of map extractor and Dispn() do not match.");

  // prestressing business
  if (PrestressIsActive(Time()))
  {
    return Teuchos::rcp(new Epetra_Vector(*interface_->FSICondMap(), true));
  }
  else
  {
    return interface_->ExtractFSICondVector(Dispn());
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::FSIStructureWrapper::ExtractInterfaceDispnp()
{
  dsassert(interface_->FullMap()->PointSameAs(Dispnp()->Map()),
      "Full map of map extractor and Dispnp() do not match.");

  // prestressing business
  if (PrestressIsActive(Time()))
  {
    if (Discretization()->Comm().MyPID() == 0)
      std::cout << "Applying no displacements to the fluid since we do prestressing" << std::endl;

    return Teuchos::rcp(new Epetra_Vector(*interface_->FSICondMap(), true));
  }
  else
  {
    return interface_->ExtractFSICondVector(Dispnp());
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
// Apply interface forces
void ADAPTER::FSIStructureWrapper::ApplyInterfaceForces(Teuchos::RCP<Epetra_Vector> iforce)
{
  FSIModelEvaluator()->GetInterfaceForceNpPtr()->PutScalar(0.0);
  interface_->AddFSICondVector(iforce, FSIModelEvaluator()->GetInterfaceForceNpPtr());
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
// Apply interface forces deprecated version ! Remove as soon as possible!
void ADAPTER::FSIStructureWrapper::ApplyInterfaceForcesTemporaryDeprecated(
    Teuchos::RCP<Epetra_Vector> iforce)
{
  Teuchos::RCP<Epetra_Vector> fifc = CORE::LINALG::CreateVector(*DofRowMap(), true);

  interface_->AddFSICondVector(iforce, fifc);

  SetForceInterface(fifc);

  PreparePartitionStep();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<STR::MODELEVALUATOR::PartitionedFSI> ADAPTER::FSIStructureWrapper::FSIModelEvaluator()
{
  return fsi_model_evaluator_;
};

BACI_NAMESPACE_CLOSE
