  /*!----------------------------------------------------------------------
\file immersed_partitioned_protrusion_formation.cpp

\brief partitioned immersed ECM -Cell interaction algorithm

<pre>
Maintainers: Andreas Rauch
             rauch@lnm.mw.tum.de
             http://www.lnm.mw.tum.de
             089 - 289 -15240
</pre>
*----------------------------------------------------------------------*/
#include "immersed_partitioned_protrusion_formation.H"

#include "../drt_poroelast/poro_scatra_base.H"
//#include "../drt_poroelast/poroelast_utils_setup.H"

#include "../drt_adapter/ad_str_fsiwrapper_immersed.H"

//#include "../drt_inpar/inpar_immersed.H"


IMMERSED::ImmersedPartitionedProtrusionFormation::ImmersedPartitionedProtrusionFormation(const Teuchos::ParameterList& params, const Epetra_Comm& comm)
  : ImmersedPartitioned(comm)
{
  // get pointer to fluid search tree from ParameterList
  fluid_SearchTree_ = params.get<Teuchos::RCP<GEO::SearchTree> >("RCPToFluidSearchTree");

  // get pointer to cell search tree from ParameterList
  cell_SearchTree_ = params.get<Teuchos::RCP<GEO::SearchTree> >("RCPToCellSearchTree");

  // get pointer to the current position map of the cell
  currpositions_cell_ = params.get<std::map<int,LINALG::Matrix<3,1> >* >("PointerToCurrentPositionsCell");

  // get pointer to the current position map of the cell
  currpositions_ECM_ = params.get<std::map<int,LINALG::Matrix<3,1> >* >("PointerToCurrentPositionsECM");

  // get pointer to cell structure
  cellstructure_=params.get<Teuchos::RCP<ADAPTER::FSIStructureWrapperImmersed> >("RCPToCellStructure");

  // get pointer poroelast-scatra interaction subproblem
  poroscatra_subproblem_ = params.get<Teuchos::RCP<POROELAST::PoroScatraBase> >("RCPToPoroScatra");

  // get pointer structure-scatra interaction (ssi) subproblem
  cellscatra_subproblem_ = params.get<Teuchos::RCP<SSI::SSI_Base> >("RCPToCellScatra");

  // important variables for parallel simulations
  myrank_  = comm.MyPID();
  numproc_ = comm.NumProc();

  // get pointer to global problem
  globalproblem_ = DRT::Problem::Instance();

  // get pointer to discretizations
  backgroundfluiddis_     = globalproblem_->GetDis("porofluid");
  backgroundstructuredis_ = globalproblem_->GetDis("structure");
  immerseddis_            = globalproblem_->GetDis("cell");
  scatradis_              = globalproblem_->GetDis("scatra");

  // get coupling variable
  displacementcoupling_ = globalproblem_->ImmersedMethodParams().sublist("PARTITIONED SOLVER").get<std::string>("COUPVARIABLE_PROTRUSION") == "Displacement";
  if(displacementcoupling_ and myrank_==0)
    std::cout<<" Coupling variable for partitioned protrusion formation:  Displacements "<<std::endl;
  else if (!displacementcoupling_ and myrank_==0)
    std::cout<<" Coupling variable for partitioned protrusion formation :  Force "<<std::endl;

  // construct immersed exchange manager. singleton class that makes immersed variables comfortably accessible from everywhere in the code
  exchange_manager_ = DRT::ImmersedFieldExchangeManager::Instance();
  exchange_manager_->SetIsProtrusionFormation(true);

}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::ReadRestart(int step)
{
  cellstructure_->ReadRestart(step);
  poroscatra_subproblem_->ReadRestart(step);
  SetTimeStep(poroscatra_subproblem_->PoroField()->Time(),step);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
IMMERSED::ImmersedPartitionedProtrusionFormation::InitialGuess()
{
  if(displacementcoupling_)
    return cellstructure_->PredictImmersedInterfaceDispnp();
  else // FORCE COUPLING
  {
    dserror("Force Coupling for Protrusion Formation is not implemented, yet.");
    return Teuchos::null;
  }

}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::PrepareTimeStep()
{

  IncrementTimeAndStep();

  PrintHeader();

  if(myrank_==0)
    std::cout<<"Cell Predictor: "<<std::endl;
  cellstructure_->PrepareTimeStep();
  if(myrank_==0)
    std::cout<<"Poro Predictor: "<<std::endl;
  poroscatra_subproblem_->PrepareTimeStep();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::CouplingOp(const Epetra_Vector &x, Epetra_Vector &F, const FillType fillFlag)
{
  // DISPLACEMENT COUPLING
  if (displacementcoupling_)
  {
    const Teuchos::RCP<Epetra_Vector> idispn = Teuchos::rcp(new Epetra_Vector(x));

    ////////////////////
    // CALL ImmersedOp
    ////////////////////
    PrepareImmersedOp();
    Teuchos::RCP<Epetra_Vector> idispnp =
    ImmersedOp(Teuchos::null, fillFlag);

    ////////////////////
    // CALL BackgroundOp
    ////////////////////
    PrepareBackgroundOp();
    BackgroundOp(Teuchos::null, fillFlag);

    int err = F.Update(1.0, *idispnp, -1.0, *idispn, 0.0);
    if(err != 0)
      dserror("Vector update of Coupling-residual returned err=%d",err);

  }
  // FORCE COUPLING
  else if(!displacementcoupling_)
  {
   dserror("Force Coupling for Protrusion Formation is not implemented, yet.");

  } // displacement / force coupling

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::BackgroundOp(Teuchos::RCP<Epetra_Vector> backgrd_dirichlet_values,
                                                              const FillType fillFlag)
{
  IMMERSED::ImmersedPartitioned::BackgroundOp(backgrd_dirichlet_values,fillFlag);

  if (fillFlag==User)
  {
    dserror("fillFlag == User : not yet implemented");
  }
  else
  {
    if(myrank_ == 0)
      std::cout<<"BackgroundOp is empty. So far, this is only a one way coupled Problem.\n"<<std::endl;
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
IMMERSED::ImmersedPartitionedProtrusionFormation::ImmersedOp(Teuchos::RCP<Epetra_Vector> bdry_traction,
                                                       const FillType fillFlag)
{
  IMMERSED::ImmersedPartitioned::ImmersedOp(bdry_traction,fillFlag);

  if(fillFlag==User)
  {
    dserror("fillFlag == User : not yet implemented");
    return Teuchos::null;
  }
  else
  {

     //solve cell
    cellstructure_->Solve();

    return cellstructure_->ExtractImmersedInterfaceDispnp();
  }

}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::PrepareBackgroundOp()
{
  // do nothing so far

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::PrepareImmersedOp()
{
 // do nothing so far

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::PrepareOutput()
{
  poroscatra_subproblem_->PrepareOutput();
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::Output()
{
  cellstructure_->Output();
  poroscatra_subproblem_->Output();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedProtrusionFormation::Update()
{
  cellstructure_->Update();
  poroscatra_subproblem_->Update();

}
