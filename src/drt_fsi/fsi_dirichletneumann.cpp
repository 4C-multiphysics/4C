
#ifdef CCADISCRET

#include "fsi_dirichletneumann.H"

#include <Teuchos_StandardParameterEntryValidators.hpp>


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::DirichletNeumann::DirichletNeumann(Epetra_Comm& comm)
  : Partitioned(comm)
{
  const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
  displacementcoupling_ = fsidyn.get<std::string>("COUPVARIABLE") == "Displacement";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::DirichletNeumann::FSIOp(const Epetra_Vector &x, Epetra_Vector &F, const FillType fillFlag)
{
  if (displacementcoupling_)
  {
    const Teuchos::RCP<Epetra_Vector> idispn = rcp(new Epetra_Vector(x));

    const Teuchos::RCP<Epetra_Vector> iforce = FluidOp(idispn, fillFlag);
    const Teuchos::RCP<Epetra_Vector> idispnp = StructOp(iforce, fillFlag);

    F.Update(1.0, *idispnp, -1.0, *idispn, 0.0);
  }
  else
  {
    const Teuchos::RCP<Epetra_Vector> iforcen = rcp(new Epetra_Vector(x));

    const Teuchos::RCP<Epetra_Vector> idisp = StructOp(iforcen, fillFlag);
    const Teuchos::RCP<Epetra_Vector> iforcenp = FluidOp(idisp, fillFlag);

    F.Update(1.0, *iforcenp, -1.0, *iforcen, 0.0);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
FSI::DirichletNeumann::FluidOp(Teuchos::RCP<Epetra_Vector> idisp,
                               const FillType fillFlag)
{
  FSI::Partitioned::FluidOp(idisp,fillFlag);

  if (fillFlag==User)
  {
    // SD relaxation calculation
    return FluidToStruct(MBFluidField().RelaxationSolve(StructToFluid(idisp),Dt()));
  }
  else
  {
    // normal fluid solve

    // the displacement -> velocity conversion at the interface
    const Teuchos::RCP<Epetra_Vector> ivel = InterfaceVelocity(idisp);

    // A rather simple hack. We need something better!
    //const int itemax = MBFluidField().Itemax();
    //if (fillFlag==MF_Res and mfresitemax_ > 0)
    //  MBFluidField().SetItemax(mfresitemax_ + 1);

    MBFluidField().NonlinearSolve(StructToFluid(idisp),StructToFluid(ivel));

    //MBFluidField().SetItemax(itemax);

    return FluidToStruct(MBFluidField().ExtractInterfaceForces());
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
FSI::DirichletNeumann::StructOp(Teuchos::RCP<Epetra_Vector> iforce,
                                const FillType fillFlag)
{
  FSI::Partitioned::StructOp(iforce,fillFlag);

  if (fillFlag==User)
  {
    // SD relaxation calculation
    return StructureField().RelaxationSolve(iforce);
  }
  else
  {
    // normal structure solve
    StructureField().ApplyInterfaceForces(iforce);
    StructureField().Solve();
    return StructureField().ExtractInterfaceDispnp();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::DirichletNeumann::InitialGuess()
{
  if (displacementcoupling_)
  {
    // predict displacement
    return StructureField().PredictInterfaceDispnp();
  }
  else
  {
    const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
    if (Teuchos::getIntegralValue<int>(fsidyn,"PREDICTOR")!=1)
    {
      dserror("unknown interface force predictor '%s'",
              fsidyn.get<string>("PREDICTOR").c_str());
    }
    return InterfaceForce();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::DirichletNeumann::UpdateDisplacement(Teuchos::RCP<Epetra_Vector>& idispn,
                                               const Epetra_Vector& finalSolution)
{
  if (displacementcoupling_)
    idispn->Update(1.0, finalSolution, 0.0);
  else
    idispn = InterfaceDisp();
}


#endif
