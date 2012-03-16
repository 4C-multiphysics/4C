/*----------------------------------------------------------------------*/
/*!
\file fsi_algorithm.cpp

\brief Basis of all FSI algorithms

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "fsi_algorithm.H"
#include "../drt_adapter/adapter_coupling.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/drt_validparameters.H"
#include "../drt_adapter/FSIStructureWrapper.H"

#include <Teuchos_StandardParameterEntryValidators.hpp>


/*----------------------------------------------------------------------*/
// Note: The order of calling the three BaseAlgorithm-constructors is
// important here! In here control file entries are written. And these entries
// define the order in which the filters handle the Discretizations, which in
// turn defines the dof number ordering of the Discretizations.
/*----------------------------------------------------------------------*/
FSI::Algorithm::Algorithm(const Epetra_Comm& comm)
  : AlgorithmBase(comm,DRT::Problem::Instance()->FSIDynamicParams())
//    StructureBaseAlgorithm(DRT::Problem::Instance()->FSIDynamicParams()),
//    FluidMovingBoundaryBaseAlgorithm(DRT::Problem::Instance()->FSIDynamicParams(),"FSICoupling")
{
  Teuchos::RCP<ADAPTER::StructureBaseAlgorithm> structure =
      Teuchos::rcp(new ADAPTER::StructureBaseAlgorithm(DRT::Problem::Instance()->FSIDynamicParams()));
  fsistructure_ = Teuchos::rcp(new ADAPTER::FSIStructureWrapper(structure->StructureFieldrcp()));

  Teuchos::RCP< ::ADAPTER::FluidMovingBoundaryBaseAlgorithm> MBFluidbase =
      Teuchos::rcp(new ADAPTER::FluidMovingBoundaryBaseAlgorithm(DRT::Problem::Instance()->FSIDynamicParams(),"FSICoupling"));
  fluid_ = MBFluidbase->MBFluidFieldrcp();

  coupsf_ = Teuchos::rcp(new ADAPTER::Coupling());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::Algorithm::~Algorithm()
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::ReadRestart(int step)
{
  StructureField().ReadRestart(step);
  double time = MBFluidField().ReadRestart(step);
  SetTimeStep(time,step);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::PrepareTimeStep()
{
  IncrementTimeAndStep();

  PrintHeader();

  StructureField().PrepareTimeStep();
  MBFluidField().PrepareTimeStep();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::Update()
{
  StructureField().Update();
  MBFluidField().Update();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Algorithm::Output()
{
  // Note: The order is important here! In here control file entries are
  // written. And these entries define the order in which the filters handle
  // the Discretizations, which in turn defines the dof number ordering of the
  // Discretizations.
  StructureField().Output();
  MBFluidField().Output();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
// Teuchos::RCP<Epetra_Vector> FSI::Algorithm::StructToAle(Teuchos::RCP<Epetra_Vector> iv) const
// {
//   return coupsa_.MasterToSlave(iv);
// }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
// Teuchos::RCP<Epetra_Vector> FSI::Algorithm::AleToStruct(Teuchos::RCP<Epetra_Vector> iv) const
// {
//   return coupsa_.SlaveToMaster(iv);
// }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::Algorithm::StructToFluid(Teuchos::RCP<Epetra_Vector> iv) const
{
  return coupsf_->MasterToSlave(iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::Algorithm::FluidToStruct(Teuchos::RCP<Epetra_Vector> iv) const
{
  return coupsf_->SlaveToMaster(iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ADAPTER::Coupling& FSI::Algorithm::StructureFluidCoupling()
{
  return *coupsf_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const ADAPTER::Coupling& FSI::Algorithm::StructureFluidCoupling() const
{
  return *coupsf_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::Algorithm::StructToFluid(Teuchos::RCP<const Epetra_Vector> iv) const
{
  return coupsf_->MasterToSlave(iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::Algorithm::FluidToStruct(Teuchos::RCP<const Epetra_Vector> iv) const
{
  return coupsf_->SlaveToMaster(iv);
}


#endif
