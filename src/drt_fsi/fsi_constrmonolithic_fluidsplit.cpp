#ifdef CCADISCRET

#include "fsi_constrmonolithic_fluidsplit.H"

#include "../drt_fluid/fluid_utils_mapextractor.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/inpar_fsi.H"

#include "../drt_adapter/adapter_structure.H"
#include "../drt_adapter/adapter_fluid.H"

#include "../drt_constraint/constraint_manager.H"

#define FLUIDSPLITAMG

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::ConstrMonolithicFluidSplit::ConstrMonolithicFluidSplit(const Epetra_Comm& comm)
  : ConstrMonolithic(comm)
{
  sconT_ = rcp(new LINALG::SparseMatrix(*conman_->GetConstraintMap(),81,false,true));
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::SetupSystem()
{
  GeneralSetup();

  const Teuchos::ParameterList& fsidyn   = DRT::Problem::Instance()->FSIDynamicParams();

  //-----------------------------------------------------------------------------
  // create combined map
  //-----------------------------------------------------------------------------

  std::vector<Teuchos::RCP<const Epetra_Map> > vecSpaces;
  vecSpaces.push_back(StructureField().DofRowMap());
#ifdef FLUIDSPLITAMG
  vecSpaces.push_back(FluidField().DofRowMap());
#else
  vecSpaces.push_back(FluidField().Interface().OtherMap());
#endif
  vecSpaces.push_back(AleField().Interface().OtherMap());
  vecSpaces.push_back(conman_->GetConstraintMap());

  if (vecSpaces[0]->NumGlobalElements()==0)
    dserror("No inner structural equations. Splitting not possible. Panic.");

  SetDofRowMaps(vecSpaces);

  FluidField().UseBlockMatrix(true);

  // build ale system matrix in splitted system
  AleField().BuildSystemMatrix(false);

  // get the PCITER from inputfile
  vector<int> pciter;
  vector<double> pcomega;
  vector<int> spciter;
  vector<double> spcomega;
  vector<int> fpciter;
  vector<double> fpcomega;
  vector<int> apciter;
  vector<double> apcomega;
  {
    int    word1;
    double word2;
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsidyn,"PCITER"));
      std::istringstream pcomegastream(Teuchos::getNumericStringParameter(fsidyn,"PCOMEGA"));
      while (pciterstream >> word1)
        pciter.push_back(word1);
      while (pcomegastream >> word2)
        pcomega.push_back(word2);
    }
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsidyn,"STRUCTPCITER"));
      std::istringstream pcomegastream(Teuchos::getNumericStringParameter(fsidyn,"STRUCTPCOMEGA"));
      while (pciterstream >> word1)
        spciter.push_back(word1);
      while (pcomegastream >> word2)
        spcomega.push_back(word2);
    }
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsidyn,"FLUIDPCITER"));
      std::istringstream pcomegastream(Teuchos::getNumericStringParameter(fsidyn,"FLUIDPCOMEGA"));
      while (pciterstream >> word1)
        fpciter.push_back(word1);
      while (pcomegastream >> word2)
        fpcomega.push_back(word2);
    }
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsidyn,"ALEPCITER"));
      std::istringstream pcomegastream(Teuchos::getNumericStringParameter(fsidyn,"ALEPCOMEGA"));
      while (pciterstream >> word1)
        apciter.push_back(word1);
      while (pcomegastream >> word2)
        apcomega.push_back(word2);
    }
  }

  //-----------------------------------------------------------------------------
  // create block system matrix
  //-----------------------------------------------------------------------------

  switch(linearsolverstrategy_)
  {
  case INPAR::FSI::PreconditionedKrylov:
    systemmatrix_ = Teuchos::rcp(new ConstrOverlappingBlockMatrix(Extractor(),
                                                                StructureField(),
                                                                FluidField(),
                                                                AleField(),
                                                                false,
                                                                DRT::INPUT::IntegralValue<int>(fsidyn,"SYMMETRICPRECOND"),
                                                                pcomega[0],
                                                                pciter[0],
                                                                spcomega[0],
                                                                spciter[0],
                                                                fpcomega[0],
                                                                fpciter[0],
                                                                apcomega[0],
                                                                apciter[0],
                                                                DRT::Problem::Instance()->ErrorFile()->Handle()));

  break;
  case INPAR::FSI::FSIAMG:
  default:
    dserror("Unsupported type of monolithic solver! Only Preconditioned Krylov supported!");
  break;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::SetupRHS(Epetra_Vector& f, bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::SetupRHS");

  double scale = FluidField().ResidualScaling();

  SetupVector(f,
    StructureField().RHS(),
    FluidField().RHS(),
    AleField().RHS(),
    conman_->GetError(),
    scale);

  // add additional ale residual
  Extractor().AddVector(*aleresidual_,2,f);

  if (firstcall)
  {
    Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockf = FluidField().BlockSystemMatrix();

    LINALG::SparseMatrix& fig = blockf->Matrix(0,1);
    LINALG::SparseMatrix& fgg = blockf->Matrix(1,1);

    Teuchos::RCP<Epetra_Vector> fveln = FluidField().ExtractInterfaceVeln();
    double timescale = FluidField().TimeScaling();
    double scale     = FluidField().ResidualScaling();

    Teuchos::RCP<Epetra_Vector> rhs = Teuchos::rcp(new Epetra_Vector(fig.RowMap()));

    fig.Apply(*fveln,*rhs);
    rhs->Scale(timescale*Dt());

#ifdef FLUIDSPLITAMG
    rhs = FluidField().Interface().InsertOtherVector(rhs);
#endif
    Extractor().AddVector(*rhs,1,f);

    rhs = Teuchos::rcp(new Epetra_Vector(fgg.RowMap()));

    fgg.Apply(*fveln,*rhs);
    rhs->Scale(scale*timescale*Dt());

    rhs = FluidToStruct(rhs);
    rhs = StructureField().Interface().InsertFSICondVector(rhs);
    Extractor().AddVector(*rhs,0,f);
  }

  // NOX expects a different sign here.
  f.Scale(-1.);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::SetupSystemMatrix(LINALG::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::SetupSystemMatrix");

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here. Extract Jacobian matrices and put them into composite system
  // matrix W

  const ADAPTER::Coupling& coupsf = StructureFluidCoupling();
  const ADAPTER::Coupling& coupsa = StructureAleCoupling();
  const ADAPTER::Coupling& coupfa = FluidAleCoupling();

  /*----------------------------------------------------------------------*/

  double scale     = FluidField().ResidualScaling();
  double timescale = FluidField().TimeScaling();

  Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockf = FluidField().BlockSystemMatrix();


  LINALG::SparseMatrix& fii = blockf->Matrix(0,0);
  LINALG::SparseMatrix& fig = blockf->Matrix(0,1);
  LINALG::SparseMatrix& fgi = blockf->Matrix(1,0);
  LINALG::SparseMatrix& fgg = blockf->Matrix(1,1);
  /*----------------------------------------------------------------------*/

  Teuchos::RCP<LINALG::BlockSparseMatrixBase> a = AleField().BlockSystemMatrix();

  if (a==Teuchos::null)
    dserror("expect ale block matrix");

  LINALG::SparseMatrix& aii = a->Matrix(0,0);
  LINALG::SparseMatrix& aig = a->Matrix(0,1);
  /*----------------------------------------------------------------------*/
  // structure part

  Teuchos::RCP<LINALG::SparseMatrix> s = StructureField().SystemMatrix();
  //const std::string fname = "cfsstructmatrix.mtl";
  //LINALG::PrintMatrixInMatlabFormat(fname,*(s->EpetraMatrix()));


  /*----------------------------------------------------------------------*/
  // structure constraint part
  RCP<LINALG::SparseOperator> tmp = conman_->GetConstrMatrix();
  LINALG::SparseMatrix scon = *(Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(tmp));

  scon.Complete(*conman_->GetConstraintMap(),s->RangeMap());
  scon.ApplyDirichlet( *(StructureField().GetDBCMapExtractor()->CondMap()),false);

  mat.Assign(0,3,View,scon);

  /*----------------------------------------------------------------------*/
  // fluid part

//    // uncomplete because the fluid interface can have more connections than the
  // structural one. (Tet elements in fluid can cause this.) We should do
//   this just once...
  s->UnComplete();


  fggtransform_(fgg,
                scale*timescale,
                ADAPTER::Coupling::SlaveConverter(coupsf),
                ADAPTER::Coupling::SlaveConverter(coupsf),
                *s,
                true,
                true);

  mat.Assign(0,0,View,*s);

  fgitransform_(fgi,
                scale,
                ADAPTER::Coupling::SlaveConverter(coupsf),
                mat.Matrix(0,1));

  figtransform_(blockf->FullRowMap(),
                blockf->FullColMap(),
                fig,
                timescale,
                ADAPTER::Coupling::SlaveConverter(coupsf),
                mat.Matrix(1,0));

#ifdef FLUIDSPLITAMG
  mat.Matrix(1,1).Add(fii,false,1.,0.0);
  Teuchos::RCP<LINALG::SparseMatrix> eye = LINALG::Eye(*FluidField().Interface().FSICondMap());
  mat.Matrix(1,1).Add(*eye,false,1.,1.0);
#else
  mat.Assign(1,1,View,fii);
#endif

  aigtransform_(a->FullRowMap(),
                a->FullColMap(),
                aig,
                1.,
                ADAPTER::Coupling::SlaveConverter(coupsa),
                mat.Matrix(2,0));
  mat.Assign(2,2,View,aii);

  /*----------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block

  Teuchos::RCP<LINALG::BlockSparseMatrixBase> mmm = FluidField().ShapeDerivatives();
  if (mmm!=Teuchos::null)
  {
    LINALG::SparseMatrix& fmii = mmm->Matrix(0,0);
    LINALG::SparseMatrix& fmgi = mmm->Matrix(1,0);

#if 1
    LINALG::SparseMatrix& fmig = mmm->Matrix(0,1);
    LINALG::SparseMatrix& fmgg = mmm->Matrix(1,1);

    // reuse transform objects to add shape derivative matrices to structural blocks

    figtransform_(blockf->FullRowMap(),
                  blockf->FullColMap(),
                  fmig,
                  1.,
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  mat.Matrix(1,0),
                  false,
                  true);

    fggtransform_(fmgg,
                  scale,
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  mat.Matrix(0,0),
                  false,
                  true);
#endif

    // We cannot copy the pressure value. It is not used anyway. So no exact
    // match here.
    fmiitransform_(mmm->FullRowMap(),
                   mmm->FullColMap(),
                   fmii,
                   1.,
                   ADAPTER::Coupling::MasterConverter(coupfa),
                   mat.Matrix(1,2),
                   false);

    fmgitransform_(fmgi,
                   scale,
                   ADAPTER::Coupling::SlaveConverter(coupsf),
                   ADAPTER::Coupling::MasterConverter(coupfa),
                   mat.Matrix(0,2),
                   false,
                   false);
  }
  /*----------------------------------------------------------------------*/
  // constraint part -> structure

  scon = *(Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(tmp));

  sconT_->Add(scon,true,1.0,0.0);
  sconT_->Complete(scon.RangeMap(),scon.DomainMap());
  mat.Assign(3,0,View,*sconT_);

  /*----------------------------------------------------------------------*/
  // done. make sure all blocks are filled.
  mat.Complete();


}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::InitialGuess(Teuchos::RCP<Epetra_Vector> ig)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::InitialGuess");

  Teuchos::RCP<Epetra_Vector> ConstraintInitialGuess = rcp(new Epetra_Vector(*(conman_->GetConstraintMap()),true));

  SetupVector(*ig,
              StructureField().InitialGuess(),
              FluidField().InitialGuess(),
              AleField().InitialGuess(),
              ConstraintInitialGuess,
              0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::SetupVector(Epetra_Vector &f,
                                                    Teuchos::RCP<const Epetra_Vector> sv,
                                                    Teuchos::RCP<const Epetra_Vector> fv,
                                                    Teuchos::RCP<const Epetra_Vector> av,
                                                    Teuchos::RCP<const Epetra_Vector> cv,
                                                    double fluidscale)
{
  // extract the inner and boundary dofs of all three fields

  Teuchos::RCP<Epetra_Vector> fov = FluidField().Interface().ExtractOtherVector(fv);
#ifdef FLUIDSPLITAMG
  fov = FluidField().Interface().InsertOtherVector(fov);
#endif
  Teuchos::RCP<Epetra_Vector> aov = AleField().Interface().ExtractOtherVector(av);

  if (fabs(fluidscale)>=1.0E-10)
  {
    // add fluid interface values to structure vector
    Teuchos::RCP<Epetra_Vector> fcv = FluidField().Interface().ExtractFSICondVector(fv);
    Teuchos::RCP<Epetra_Vector> modsv = StructureField().Interface().InsertFSICondVector(FluidToStruct(fcv));
    modsv->Update(1.0, *sv, fluidscale);

    Extractor().InsertVector(*modsv,0,f);
  }
  else
  {
    Extractor().InsertVector(*sv,0,f);
  }

  Extractor().InsertVector(*fov,1,f);
  Extractor().InsertVector(*aov,2,f);
  Epetra_Vector modcv = *cv;
  modcv.Scale(-1.0);
  Extractor().InsertVector(modcv,3,f);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithicFluidSplit::ExtractFieldVectors(Teuchos::RCP<const Epetra_Vector> x,
                                                        Teuchos::RCP<const Epetra_Vector>& sx,
                                                        Teuchos::RCP<const Epetra_Vector>& fx,
                                                        Teuchos::RCP<const Epetra_Vector>& ax)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::ConstrMonolithicFluidSplit::ExtractFieldVectors");

  // We have overlap at the interface. Thus we need the interface part of the
  // structure vector and append it to the fluid and ale vector. (With the
  // right translation.)

  sx = Extractor().ExtractVector(x,0);
  Teuchos::RCP<const Epetra_Vector> scx = StructureField().Interface().ExtractFSICondVector(sx);

  // process fluid unknowns

  Teuchos::RCP<const Epetra_Vector> fox = Extractor().ExtractVector(x,1);
#ifdef FLUIDSPLITAMG
  fox = FluidField().Interface().ExtractOtherVector(fox);
#endif
  Teuchos::RCP<Epetra_Vector> fcx = StructToFluid(scx);

  FluidField().DisplacementToVelocity(fcx);

  Teuchos::RCP<Epetra_Vector> f = FluidField().Interface().InsertOtherVector(fox);
  FluidField().Interface().InsertFSICondVector(fcx, f);
  fx = f;

  // process ale unknowns

  Teuchos::RCP<const Epetra_Vector> aox = Extractor().ExtractVector(x,2);
  Teuchos::RCP<Epetra_Vector> acx = StructToAle(scx);
  Teuchos::RCP<Epetra_Vector> a = AleField().Interface().InsertOtherVector(aox);
  AleField().Interface().InsertVector(acx, 1, a);

  ax = a;
}


#endif
