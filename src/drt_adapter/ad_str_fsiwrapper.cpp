/*----------------------------------------------------------------------*/
/*!
\file FSIStructureWrapper.cpp

\brief Structural adapter for FSI problems containing the interface
       and methods dependent on the interface

<pre>
Maintainer: Georg Hammerl
            hammerl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/



#include "ad_str_fsiwrapper.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_utils.H"
#include "../drt_structure/stru_aux.H"



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::FSIStructureWrapper::FSIStructureWrapper(Teuchos::RCP<Structure> structure)
: StructureWrapper(structure)
{
  // set-up FSI interface
  interface_ = Teuchos::rcp(new STR::AUX::MapExtractor);

  if (DRT::Problem::Instance()->ProblemType() != prb_fpsi)
    interface_->Setup(*Discretization(), *Discretization()->DofRowMap());
  else
    interface_->Setup(*Discretization(), *Discretization()->DofRowMap(),true); //create overlapping maps for fpsi problem

  const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
  const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");
  predictor_ = DRT::INPUT::IntegralValue<int>(fsipart,"PREDICTOR");

}

/*------------------------------------------------------------------------------------*
 * Rebuild FSI interface on structure side                                      sudhakar 09/13
 * This is necessary if elements are added/deleted from interface
 *------------------------------------------------------------------------------------*/
void ADAPTER::FSIStructureWrapper::RebuildInterface()
{
  interface_ = Teuchos::rcp(new STR::AUX::MapExtractor);
  interface_->Setup(*Discretization(), *Discretization()->DofRowMap());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::FSIStructureWrapper::UseBlockMatrix()
{
  StructureWrapper::UseBlockMatrix(interface_,interface_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::FSIStructureWrapper::RelaxationSolve(Teuchos::RCP<Epetra_Vector> iforce)
{
  Teuchos::RCP<Epetra_Vector> relax = interface_->InsertFSICondVector(iforce);
  SetForceInterface(relax);
  Teuchos::RCP<Epetra_Vector> idisi = SolveRelaxationLinear();

  // we are just interested in the incremental interface displacements
  return interface_->ExtractFSICondVector(idisi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::FSIStructureWrapper::PredictInterfaceDispnp()
{
  // prestressing business
  double time = 0.0;
  double pstime = -1.0;
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  INPAR::STR::PreStress pstype = DRT::INPUT::IntegralValue<INPAR::STR::PreStress>(sdyn,"PRESTRESS");
  if (pstype != INPAR::STR::prestress_none)
  {
    time = Time();
    pstime = sdyn.get<double>("PRESTRESSTIME");
  }

  Teuchos::RCP<Epetra_Vector> idis;

  switch (predictor_)
  {
  case 1:
  {
    // d(n)
    // respect Dirichlet conditions at the interface (required for pseudo-rigid body)
    if (pstype != INPAR::STR::prestress_none && time <= pstime)
    {
      idis = Teuchos::rcp(new Epetra_Vector(*interface_->FSICondMap(),true));
    }
    else
    {
      idis  = interface_->ExtractFSICondVector(Dispn());
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
    if (pstype != INPAR::STR::prestress_none && time <= pstime)
      dserror("only constant interface predictor useful for prestressing");

    double dt = Dt();

    idis = interface_->ExtractFSICondVector(Dispn());
    Teuchos::RCP<Epetra_Vector> ivel
      = interface_->ExtractFSICondVector(Veln());

    idis->Update(dt,* ivel, 1.0);
    break;
  }
  case 4:
  {
    // d(n)+dt*v(n)+0.5*dt^2*a(n)
    if (pstype != INPAR::STR::prestress_none && time <= pstime)
      dserror("only constant interface predictor useful for prestressing");

    double dt = Dt();

    idis = interface_->ExtractFSICondVector(Dispn());
    Teuchos::RCP<Epetra_Vector> ivel
      = interface_->ExtractFSICondVector(Veln());
    Teuchos::RCP<Epetra_Vector> iacc
      = interface_->ExtractFSICondVector(Accn());

    idis->Update(dt, *ivel, 0.5*dt*dt, *iacc, 1.0);
    break;
  }
  default:
    dserror("unknown interface displacement predictor '%s'",
        DRT::Problem::Instance()->FSIDynamicParams().sublist("PARTITIONED SOLVER").get<std::string>("PREDICTOR").c_str());
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
  double time = 0.0;
  double pstime = -1.0;
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  INPAR::STR::PreStress pstype = DRT::INPUT::IntegralValue<INPAR::STR::PreStress>(sdyn,"PRESTRESS");
  if (pstype != INPAR::STR::prestress_none)
  {
    time = TimeOld();
    pstime = sdyn.get<double>("PRESTRESSTIME");
  }

  if (pstype != INPAR::STR::prestress_none && time <= pstime)
  {
    return Teuchos::rcp(new Epetra_Vector(*interface_->FSICondMap(),true));
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
  double time = 0.0;
  double pstime = -1.0;
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  INPAR::STR::PreStress pstype = DRT::INPUT::IntegralValue<INPAR::STR::PreStress>(sdyn,"PRESTRESS");
  if (pstype != INPAR::STR::prestress_none)
  {
    time = Time();
    pstime = sdyn.get<double>("PRESTRESSTIME");
  }

  if (pstype != INPAR::STR::prestress_none && time <= pstime)
  {
    return Teuchos::rcp(new Epetra_Vector(*interface_->FSICondMap(),true));
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
  Teuchos::RCP<Epetra_Vector> fifc = LINALG::CreateVector(*DofRowMap(), true);

  interface_->AddFSICondVector(iforce, fifc);

  SetForceInterface(fifc);

  PreparePartitionStep();

  return;
}



