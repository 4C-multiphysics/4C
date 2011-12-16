#ifdef CCADISCRET

#include "../drt_lib/standardtypes_cpp.H"

#include "fsi_monolithicfluidsplit.H"
#include "fsi_debugwriter.H"
#include "fsi_statustest.H"
#include "fsi_nox_linearsystem_bgs.H"
#include "fsi_overlapprec_fsiamg.H"
#include "fsi_monolithic_linearsystem.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/drt_validparameters.H"
#include "../drt_fluid/fluid_utils_mapextractor.H"

#include "../drt_io/io_control.H"

#define FLUIDSPLITAMG

/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::MonolithicFluidSplit::MonolithicFluidSplit(const Epetra_Comm& comm)
  : BlockMonolithic(comm)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::SetupSystem()
{

  const Teuchos::ParameterList& fsidyn   = DRT::Problem::Instance()->FSIDynamicParams();
  linearsolverstrategy_ = DRT::INPUT::IntegralValue<INPAR::FSI::LinearBlockSolver>(fsidyn,"LINEARBLOCKSOLVER");

  SetDefaultParameters(fsidyn,NOXParameterList());

  // right now we use matching meshes at the interface

  ADAPTER::Coupling& coupsf = StructureFluidCoupling();
  ADAPTER::Coupling& coupsa = StructureAleCoupling();
  ADAPTER::Coupling& coupfa = FluidAleCoupling();

  // structure to fluid

  coupsf.SetupConditionCoupling(*StructureField().Discretization(),
                                 StructureField().Interface().FSICondMap(),
                                *FluidField().Discretization(),
                                 FluidField().Interface().FSICondMap(),
                                "FSICoupling",
                                genprob.ndim);

  // structure to ale

  coupsa.SetupConditionCoupling(*StructureField().Discretization(),
                                 StructureField().Interface().FSICondMap(),
                                *AleField().Discretization(),
                                 AleField().Interface().FSICondMap(),
                                 "FSICoupling",
                                genprob.ndim);

  // In the following we assume that both couplings find the same dof
  // map at the structural side. This enables us to use just one
  // interface dof map for all fields and have just one transfer
  // operator from the interface map to the full field map.
  if (not coupsf.MasterDofMap()->SameAs(*coupsa.MasterDofMap()))
    dserror("structure interface dof maps do not match");

  if (coupsf.MasterDofMap()->NumGlobalElements()==0)
    dserror("No nodes in matching FSI interface. Empty FSI coupling condition?");

  // the fluid-ale coupling always matches
  const Epetra_Map* fluidnodemap = FluidField().Discretization()->NodeRowMap();
  const Epetra_Map* alenodemap   = AleField().Discretization()->NodeRowMap();

  coupfa.SetupCoupling(*FluidField().Discretization(),
                       *AleField().Discretization(),
                       *fluidnodemap,
                       *alenodemap,
                       genprob.ndim);

  FluidField().SetMeshMap(coupfa.MasterDofMap());

  // create combined map

  std::vector<Teuchos::RCP<const Epetra_Map> > vecSpaces;
  vecSpaces.push_back(StructureField().DofRowMap());
#ifdef FLUIDSPLITAMG
  vecSpaces.push_back(FluidField()    .DofRowMap());
#else
  vecSpaces.push_back(FluidField()    .Interface().OtherMap());
#endif
  vecSpaces.push_back(AleField()      .Interface().OtherMap());

  if (vecSpaces[1]->NumGlobalElements()==0)
    dserror("No inner fluid equations. Splitting not possible. Panic.");

  SetDofRowMaps(vecSpaces);

  /*----------------------------------------------------------------------*/
  // Switch fluid to interface split block matrix
  FluidField().UseBlockMatrix(true);

  // build ale system matrix in splitted system
  AleField().BuildSystemMatrix(false);

  aleresidual_ = Teuchos::rcp(new Epetra_Vector(*AleField().Interface().OtherMap()));

  vector<int> pciter;
  vector<double> pcomega;
  vector<int> spciter;
  vector<double> spcomega;
  vector<int> fpciter;
  vector<double> fpcomega;
  vector<int> apciter;
  vector<double> apcomega;
  vector<string> blocksmoother;
  vector<double> schuromega;
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
    {
      string word;
      std::istringstream blocksmootherstream(Teuchos::getNumericStringParameter(fsidyn,"BLOCKSMOOTHER"));
      while (blocksmootherstream >> word)
        blocksmoother.push_back(word);
    }
    {
      std::istringstream blocksmootherstream(Teuchos::getNumericStringParameter(fsidyn,"SCHUROMEGA"));
      while (blocksmootherstream >> word2)
        schuromega.push_back(word2);
    }
  }

  // enable debugging
  if (DRT::INPUT::IntegralValue<int>(fsidyn,"DEBUGOUTPUT") & 2)
  {
    pcdbg_ = Teuchos::rcp(new UTILS::MonolithicDebugWriter(*this));
  }

  // create block system matrix
  switch(linearsolverstrategy_)
  {
  case INPAR::FSI::PreconditionedKrylov:
  case INPAR::FSI::FSIAMG:
    systemmatrix_ = Teuchos::rcp(new OverlappingBlockMatrixFSIAMG(
                                   Extractor(),
                                   StructureField(),
                                   FluidField(),
                                   AleField(),
                                   false,
                                   DRT::INPUT::IntegralValue<int>(fsidyn,"SYMMETRICPRECOND"),
                                   blocksmoother,
                                   schuromega,
                                   pcomega,
                                   pciter,
                                   spcomega,
                                   spciter,
                                   fpcomega,
                                   fpciter,
                                   apcomega,
                                   apciter,
                                   DRT::INPUT::IntegralValue<int>(fsidyn,"FSIAMGANALYZE"),
                                   linearsolverstrategy_,
                                   DRT::Problem::Instance()->ErrorFile()->Handle()));
    break;
#if 0 // no longer in use
  case INPAR::FSI::PreconditionedKrylov:
    systemmatrix_ = Teuchos::rcp(new OverlappingBlockMatrix(
                                   pcdbg_,
                                   Extractor(),
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
#endif    
  default:
    dserror("Unsupported type of monolithic solver");
    break;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::SetupRHS(Epetra_Vector& f, bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::SetupRHS");

  SetupVector(f,
              StructureField().RHS(),
              FluidField().RHS(),
              AleField().RHS(),
              FluidField().ResidualScaling());

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
    Teuchos::RCP<const Epetra_Vector> zeros = Teuchos::rcp(new const Epetra_Vector(rhs->Map(),true));
    LINALG::ApplyDirichlettoSystem(rhs,zeros,*(StructureField().GetDBCMapExtractor()->CondMap()));


    Extractor().AddVector(*rhs,1,f);

    rhs = Teuchos::rcp(new Epetra_Vector(fgg.RowMap()));

    fgg.Apply(*fveln,*rhs);
    rhs->Scale(scale*timescale*Dt());

    rhs = FluidToStruct(rhs);
    rhs = StructureField().Interface().InsertFSICondVector(rhs);

    zeros = Teuchos::rcp(new const Epetra_Vector(rhs->Map(),true));
    LINALG::ApplyDirichlettoSystem(rhs,zeros,*(StructureField().GetDBCMapExtractor()->CondMap()));

    if (StructureField().GetSTCAlgo() == INPAR::STR::stc_currsym)
    {
      Teuchos::RCP<LINALG::SparseMatrix> stcmat = StructureField().GetSTCMat();
      stcmat->Multiply(true,*rhs,*rhs);
    }
    Extractor().AddVector(*rhs,0,f);
  }

  // NOX expects a different sign here.
  f.Scale(-1.);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::SetupSystemMatrix(LINALG::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::SetupSystemMatrix");

  // extract Jacobian matrices and put them into composite system
  // matrix W

  const ADAPTER::Coupling& coupsf = StructureFluidCoupling();
  const ADAPTER::Coupling& coupsa = StructureAleCoupling();
  const ADAPTER::Coupling& coupfa = FluidAleCoupling();

  // get info about STC feature
  INPAR::STR::STC_Scale stcalgo = StructureField().GetSTCAlgo();
  Teuchos::RCP<LINALG::SparseMatrix> stcmat = Teuchos::null;
  // if STC is to be used, get STC matrix from strucutre field
  if (stcalgo != INPAR::STR::stc_none)
    stcmat = StructureField().GetSTCMat();

  Teuchos::RCP<LINALG::SparseMatrix> s = StructureField().SystemMatrix();

  // split fluid matrix

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

  double scale     = FluidField().ResidualScaling();
  double timescale = FluidField().TimeScaling();

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  // uncomplete because the fluid interface can have more connections than the
  // structural one. (Tet elements in fluid can cause this.) We should do
  // this just once...
  s->UnComplete();

  fggtransform_(fgg,
                scale*timescale,
                ADAPTER::Coupling::SlaveConverter(coupsf),
                ADAPTER::Coupling::SlaveConverter(coupsf),
                *s,
                true,
                true);

  RCP<LINALG::SparseMatrix> lfgi = rcp(new LINALG::SparseMatrix(s->RowMap(),81,false));
  fgitransform_(fgi,
                scale,
                ADAPTER::Coupling::SlaveConverter(coupsf),
                *lfgi);

  lfgi->Complete(fgi.DomainMap(),s->RangeMap());
  lfgi->ApplyDirichlet( *(StructureField().GetDBCMapExtractor()->CondMap()),false);

  if (stcalgo == INPAR::STR::stc_currsym)
    lfgi = LINALG::MLMultiply(*stcmat, true, *lfgi, false, true, true, true);

  #ifdef FLUIDSPLITAMG
    mat.Matrix(0,1).UnComplete();
    mat.Matrix(0,1).Add(*lfgi,false,1.,0.0);
  #else
    mat.Assign(0,1,View,*lfgi);
  #endif

  if (stcalgo == INPAR::STR::stc_none)
  {
    figtransform_(blockf->FullRowMap(),
                  blockf->FullColMap(),
                  fig,
                  timescale,
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  mat.Matrix(1,0));
  }
  else
  {
    RCP<LINALG::SparseMatrix> lfig = rcp(new LINALG::SparseMatrix(fig.RowMap(),81,false));
    figtransform_(blockf->FullRowMap(),
                  blockf->FullColMap(),
                  fig,
                  timescale,
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  *lfig);

    lfig->Complete(s->DomainMap(),fig.RangeMap());

    lfig = LINALG::MLMultiply(*lfig,false,*stcmat, false, false, false,true);

    #ifdef FLUIDSPLITAMG
      mat.Matrix(1,0).UnComplete();
      mat.Matrix(1,0).Add(*lfig,false,1.,0.0);
    #else
      mat.Assign(1,0,View,*lfig);
    #endif
  }

#ifdef FLUIDSPLITAMG
  mat.Matrix(1,1).Add(fii,false,1.,0.0);
  Teuchos::RCP<LINALG::SparseMatrix> eye = LINALG::Eye(*FluidField().Interface().FSICondMap());
  mat.Matrix(1,1).Add(*eye,false,1.,1.0);
#else
  mat.Assign(1,1,View,fii);
#endif

  if (stcalgo == INPAR::STR::stc_none)
  {
    aigtransform_(a->FullRowMap(),
                  a->FullColMap(),
                  aig,
                  1.,
                  ADAPTER::Coupling::SlaveConverter(coupsa),
                  mat.Matrix(2,0));
  }
  else
  {
    RCP<LINALG::SparseMatrix> laig = rcp(new LINALG::SparseMatrix(aii.RowMap(),81,false));
    aigtransform_(a->FullRowMap(),
                  a->FullColMap(),
                  aig,
                  1.,
                  ADAPTER::Coupling::SlaveConverter(coupsa),
                  *laig);

    laig->Complete(s->DomainMap(),laig->RangeMap());
    laig->ApplyDirichlet( *(AleField().GetDBCMapExtractor()->CondMap()),false);

    if (stcalgo != INPAR::STR::stc_none)
    {
      laig = LINALG::MLMultiply(*laig,false,*stcmat, false, false, false,true);
    }

    mat.Assign(2,0,View,*laig);
  }

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

    if (stcalgo == INPAR::STR::stc_none)
    {
      figtransform_(blockf->FullRowMap(),
                  blockf->FullColMap(),
                  fmig,
                  1.,
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  mat.Matrix(1,0),
                  false,
                  true);
    }
    else
    {
      RCP<LINALG::SparseMatrix> lfmig = rcp(new LINALG::SparseMatrix(fmig.RowMap(),81,false));
      figtransform_(blockf->FullRowMap(),
                  blockf->FullColMap(),
                  fmig,
                  1.,
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  *lfmig,
                  false,
                  true);


      lfmig->Complete(s->DomainMap(),fmig.RangeMap());

      lfmig->ApplyDirichlet( *(FluidField().GetDBCMapExtractor()->CondMap()),false);

      if (stcalgo != INPAR::STR::stc_none)
      {
        lfmig = LINALG::MLMultiply(*lfmig,false,*stcmat, false, false, false,true);
      }

      mat.Matrix(1,0).Add(*lfmig,false,1.0,1.0);
    }

    fggtransform_(fmgg,
                  scale,
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  ADAPTER::Coupling::SlaveConverter(coupsf),
                  *s,
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

    {
      RCP<LINALG::SparseMatrix> lfmgi = rcp(new LINALG::SparseMatrix(s->RowMap(),81,false));
      fmgitransform_(fmgi,
                     scale,
                     ADAPTER::Coupling::SlaveConverter(coupsf),
                     ADAPTER::Coupling::MasterConverter(coupfa),
                     *lfmgi,//mat.Matrix(0,2),
                     false,
                     false);

      lfmgi->Complete(aii.DomainMap(),s->RangeMap());
      lfmgi->ApplyDirichlet( *(StructureField().GetDBCMapExtractor()->CondMap()),false);

      if (stcalgo == INPAR::STR::stc_currsym)
        lfmgi = LINALG::MLMultiply(*stcmat, true, *lfmgi, false, true, true, true);

      #ifdef FLUIDSPLITAMG
        mat.Matrix(0,2).UnComplete();
        mat.Matrix(0,2).Add(*lfmgi,false,1.,0.0);
      #else
        mat.Assign(0,2,View,*lfmgi);
      #endif

    }

  }

  s->Complete();

  s->ApplyDirichlet( *(StructureField().GetDBCMapExtractor()->CondMap()),true);

  //apply STC matrix on block (0,0)
  if (stcalgo != INPAR::STR::stc_none)
  {
    s = LINALG::MLMultiply(*s, false, *stcmat, false, true, true, true);
    if (stcalgo == INPAR::STR::stc_currsym)
      s = LINALG::MLMultiply(*stcmat, true, *s, false, true, true, false);
  }
  else
  {
    s->UnComplete();
  }

  //finally assign structure block
  mat.Matrix(0,0).Assign(View,*s);

  // done. make sure all blocks are filled.
  mat.Complete();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::InitialGuess(Teuchos::RCP<Epetra_Vector> ig)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::InitialGuess");

  SetupVector(*ig,
              StructureField().InitialGuess(),
              FluidField().InitialGuess(),
              AleField().InitialGuess(),
              0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::ScaleSystem(LINALG::BlockSparseMatrixBase& mat, Epetra_Vector& b)
{
  const Teuchos::ParameterList& fsidyn   = DRT::Problem::Instance()->FSIDynamicParams();
  const bool scaling_infnorm = (bool)DRT::INPUT::IntegralValue<int>(fsidyn,"INFNORMSCALING");

  if (scaling_infnorm)
  {
    // The matrices are modified here. Do we have to change them back later on?

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.Matrix(0,0).EpetraMatrix();
    srowsum_ = rcp(new Epetra_Vector(A->RowMap(),false));
    scolsum_ = rcp(new Epetra_Vector(A->RowMap(),false));
    A->InvRowSums(*srowsum_);
    A->InvColSums(*scolsum_);
    if (A->LeftScale(*srowsum_) or
        A->RightScale(*scolsum_) or
        mat.Matrix(0,1).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(0,2).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(1,0).EpetraMatrix()->RightScale(*scolsum_) or
        mat.Matrix(2,0).EpetraMatrix()->RightScale(*scolsum_))
      dserror("structure scaling failed");

    A = mat.Matrix(2,2).EpetraMatrix();
    arowsum_ = rcp(new Epetra_Vector(A->RowMap(),false));
    acolsum_ = rcp(new Epetra_Vector(A->RowMap(),false));
    A->InvRowSums(*arowsum_);
    A->InvColSums(*acolsum_);
    if (A->LeftScale(*arowsum_) or
        A->RightScale(*acolsum_) or
        mat.Matrix(2,0).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(2,1).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(0,2).EpetraMatrix()->RightScale(*acolsum_) or
        mat.Matrix(1,2).EpetraMatrix()->RightScale(*acolsum_))
      dserror("ale scaling failed");

    Teuchos::RCP<Epetra_Vector> sx = Extractor().ExtractVector(b,0);
    Teuchos::RCP<Epetra_Vector> ax = Extractor().ExtractVector(b,2);

    if (sx->Multiply(1.0, *srowsum_, *sx, 0.0))
      dserror("structure scaling failed");
    if (ax->Multiply(1.0, *arowsum_, *ax, 0.0))
      dserror("ale scaling failed");

    Extractor().InsertVector(*sx,0,b);
    Extractor().InsertVector(*ax,2,b);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::UnscaleSolution(LINALG::BlockSparseMatrixBase& mat, Epetra_Vector& x, Epetra_Vector& b)
{
  const Teuchos::ParameterList& fsidyn   = DRT::Problem::Instance()->FSIDynamicParams();
  const bool scaling_infnorm = (bool)DRT::INPUT::IntegralValue<int>(fsidyn,"INFNORMSCALING");

  if (scaling_infnorm)
  {
    Teuchos::RCP<Epetra_Vector> sy = Extractor().ExtractVector(x,0);
    Teuchos::RCP<Epetra_Vector> ay = Extractor().ExtractVector(x,2);

    if (sy->Multiply(1.0, *scolsum_, *sy, 0.0))
      dserror("structure scaling failed");
    if (ay->Multiply(1.0, *acolsum_, *ay, 0.0))
      dserror("ale scaling failed");

    // get info about STC feature and unscale solution if necessary
    INPAR::STR::STC_Scale stcalgo = StructureField().GetSTCAlgo();
    if (stcalgo != INPAR::STR::stc_none)
    {
      StructureField().GetSTCMat()->Multiply(false,*sy,*sy);
    }

    Extractor().InsertVector(*sy,0,x);
    Extractor().InsertVector(*ay,2,x);

    Teuchos::RCP<Epetra_Vector> sx = Extractor().ExtractVector(b,0);
    Teuchos::RCP<Epetra_Vector> ax = Extractor().ExtractVector(b,2);

    if (sx->ReciprocalMultiply(1.0, *srowsum_, *sx, 0.0))
      dserror("structure scaling failed");
    if (ax->ReciprocalMultiply(1.0, *arowsum_, *ax, 0.0))
      dserror("ale scaling failed");

    if (stcalgo != INPAR::STR::stc_none)
    {
      StructureField().GetSTCMat()->Multiply(false,*sx,*sx);
    }

    Extractor().InsertVector(*sx,0,b);
    Extractor().InsertVector(*ax,2,b);

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.Matrix(0,0).EpetraMatrix();
    srowsum_->Reciprocal(*srowsum_);
    scolsum_->Reciprocal(*scolsum_);
    if (A->LeftScale(*srowsum_) or
        A->RightScale(*scolsum_) or
        mat.Matrix(0,1).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(0,2).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(1,0).EpetraMatrix()->RightScale(*scolsum_) or
        mat.Matrix(2,0).EpetraMatrix()->RightScale(*scolsum_))
      dserror("structure scaling failed");

    A = mat.Matrix(2,2).EpetraMatrix();
    arowsum_->Reciprocal(*arowsum_);
    acolsum_->Reciprocal(*acolsum_);
    if (A->LeftScale(*arowsum_) or
        A->RightScale(*acolsum_) or
        mat.Matrix(2,0).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(2,1).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(0,2).EpetraMatrix()->RightScale(*acolsum_) or
        mat.Matrix(1,2).EpetraMatrix()->RightScale(*acolsum_))
      dserror("ale scaling failed");
  }

#if 1
  // very simple hack just to see the linear solution

  Epetra_Vector r(b.Map());
  mat.Apply(x,r);
  r.Update(1.,b,1.);

  Teuchos::RCP<Epetra_Vector> sr = Extractor().ExtractVector(r,0);
  Teuchos::RCP<Epetra_Vector> fr = Extractor().ExtractVector(r,1);
  Teuchos::RCP<Epetra_Vector> ar = Extractor().ExtractVector(r,2);

  // increment additional ale residual
  aleresidual_->Update(-1.,*ar,0.);

  ios_base::fmtflags flags = Utils()->out().flags();

  double n,ns,nf,na;
  r.Norm2(&n);
  sr->Norm2(&ns);
  fr->Norm2(&nf);
  ar->Norm2(&na);
  Utils()->out() << std::scientific
                 << "\nlinear solver quality:\n"
                 << "L_2-norms:\n"
                 << END_COLOR "   |r|=" YELLOW << n
                 << END_COLOR "   |rs|=" YELLOW << ns
                 << END_COLOR "   |rf|=" YELLOW << nf
                 << END_COLOR "   |ra|=" YELLOW << na
                 << END_COLOR "\n";
  r.NormInf(&n);
  sr->NormInf(&ns);
  fr->NormInf(&nf);
  ar->NormInf(&na);
  Utils()->out() << "L_inf-norms:\n"
                 << END_COLOR "   |r|=" YELLOW << n
                 << END_COLOR "   |rs|=" YELLOW << ns
                 << END_COLOR "   |rf|=" YELLOW << nf
                 << END_COLOR "   |ra|=" YELLOW << na
                 << END_COLOR "\n";

  Utils()->out().flags(flags);
#endif

  if (StructureField().GetSTCAlgo() != INPAR::STR::stc_none)
    StructureField().SystemMatrix()->Reset();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::SetupVector(Epetra_Vector &f,
                                         Teuchos::RCP<const Epetra_Vector> sv,
                                         Teuchos::RCP<const Epetra_Vector> fv,
                                         Teuchos::RCP<const Epetra_Vector> av,
                                         double fluidscale)
{

  // extract the inner and boundary dofs of all three fields

  Teuchos::RCP<Epetra_Vector> fov = FluidField()    .Interface().ExtractOtherVector(fv);
#ifdef FLUIDSPLITAMG
  fov = FluidField().Interface().InsertOtherVector(fov);
#endif
  Teuchos::RCP<Epetra_Vector> aov = AleField()      .Interface().ExtractOtherVector(av);

  if (fluidscale!=0)
  {
    // add fluid interface values to structure vector
    Teuchos::RCP<Epetra_Vector> fcv = FluidField().Interface().ExtractFSICondVector(fv);
    Teuchos::RCP<Epetra_Vector> modsv = StructureField().Interface().InsertFSICondVector(FluidToStruct(fcv));
    modsv->Update(1.0, *sv, fluidscale);

    Teuchos::RCP<const Epetra_Vector> zeros = Teuchos::rcp(new const Epetra_Vector(modsv->Map(),true));
    LINALG::ApplyDirichlettoSystem(modsv,zeros,*(StructureField().GetDBCMapExtractor()->CondMap()));


    if (StructureField().GetSTCAlgo() == INPAR::STR::stc_currsym)
    {
      Teuchos::RCP<LINALG::SparseMatrix> stcmat = StructureField().GetSTCMat();
      stcmat->Multiply(true,*modsv,*modsv);
    }

    Extractor().InsertVector(*modsv,0,f);
  }
  else
  {
    Teuchos::RCP<Epetra_Vector> modsv =  rcp(new Epetra_Vector(*sv));
    if (StructureField().GetSTCAlgo() == INPAR::STR::stc_currsym)
    {
      Teuchos::RCP<LINALG::SparseMatrix> stcmat = StructureField().GetSTCMat();
      stcmat->Multiply(true,*sv,*modsv);
    }

    Extractor().InsertVector(*modsv,0,f);
  }

  Extractor().InsertVector(*fov,1,f);
  Extractor().InsertVector(*aov,2,f);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<NOX::Epetra::LinearSystem>
FSI::MonolithicFluidSplit::CreateLinearSystem(ParameterList& nlParams,
                                           NOX::Epetra::Vector& noxSoln,
                                           Teuchos::RCP<NOX::Utils> utils)
{
  Teuchos::RCP<NOX::Epetra::LinearSystem> linSys;

  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");
  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  Teuchos::ParameterList& newtonParams = dirParams.sublist("Newton");
  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");

  NOX::Epetra::Interface::Jacobian* iJac = this;
  NOX::Epetra::Interface::Preconditioner* iPrec = this;
  const Teuchos::RCP< Epetra_Operator > J = systemmatrix_;
  const Teuchos::RCP< Epetra_Operator > M = systemmatrix_;

  switch (linearsolverstrategy_)
  {
  case INPAR::FSI::PreconditionedKrylov:
  case INPAR::FSI::FSIAMG:
#if 0
    linSys = Teuchos::rcp(new FSI::MonolithicLinearSystem::MonolithicLinearSystem(
#else
    linSys = Teuchos::rcp(new NOX::Epetra::LinearSystemAztecOO(
#endif
                                                               printParams,
                                                               lsParams,
                                                               Teuchos::rcp(iJac,false),
                                                               J,
                                                               Teuchos::rcp(iPrec,false),
                                                               M,
                                                               noxSoln));
    break;
  case INPAR::FSI::BGSAitken:
  case INPAR::FSI::BGSVectorExtrapolation:
  case INPAR::FSI::BGSJacobianFreeNewtonKrylov:
    linSys = Teuchos::rcp(new NOX::FSI::LinearBGSSolver(printParams,
                                                        lsParams,
                                                        Teuchos::rcp(iJac,false),
                                                        J,
                                                        noxSoln,
                                                        StructureField().LinearSolver(),
                                                        FluidField().LinearSolver(),
                                                        AleField().LinearSolver(),
                                                        linearsolverstrategy_));
    break;
  default:
    dserror("unsupported linear block solver strategy: %d", linearsolverstrategy_);
  }

  return linSys;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<NOX::StatusTest::Combo>
FSI::MonolithicFluidSplit::CreateStatusTest(Teuchos::ParameterList& nlParams,
                                         Teuchos::RCP<NOX::Epetra::Group> grp)
{
  // Create the convergence tests
  Teuchos::RCP<NOX::StatusTest::Combo> combo       = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  Teuchos::RCP<NOX::StatusTest::Combo> converged   = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));

  Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters = Teuchos::rcp(new NOX::StatusTest::MaxIters(nlParams.get("Max Iterations", 100)));
  Teuchos::RCP<NOX::StatusTest::FiniteValue> fv    = Teuchos::rcp(new NOX::StatusTest::FiniteValue);

  combo->addStatusTest(fv);
  combo->addStatusTest(converged);
  combo->addStatusTest(maxiters);

  // require one solve
  converged->addStatusTest(Teuchos::rcp(new NOX::FSI::MinIters(1)));

  // setup tests for structural displacements

  Teuchos::RCP<NOX::StatusTest::Combo> structcombo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> structureDisp =
    Teuchos::rcp(new NOX::FSI::PartialNormF("displacement",
                                            Extractor(),0,
                                            nlParams.get("Norm abs disp", 1.0e-6),
                                            NOX::Abstract::Vector::TwoNorm,
                                            NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> structureDispUpdate =
    Teuchos::rcp(new NOX::FSI::PartialNormUpdate("displacement update",
                                                 Extractor(),0,
                                                 nlParams.get("Norm abs disp", 1.0e-6),
                                                 NOX::FSI::PartialNormUpdate::Scaled));

  AddStatusTest(structureDisp);
  structcombo->addStatusTest(structureDisp);
  //structcombo->addStatusTest(structureDispUpdate);

  converged->addStatusTest(structcombo);

  // setup tests for fluid velocities

  std::vector<Teuchos::RCP<const Epetra_Map> > fluidvel;
  fluidvel.push_back(FluidField().InnerVelocityRowMap());
  fluidvel.push_back(Teuchos::null);
  LINALG::MultiMapExtractor fluidvelextract(*DofRowMap(),fluidvel);

  Teuchos::RCP<NOX::StatusTest::Combo> fluidvelcombo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> innerFluidVel =
    Teuchos::rcp(new NOX::FSI::PartialNormF("velocity",
                                            fluidvelextract,0,
                                            nlParams.get("Norm abs vel", 1.0e-6),
                                            NOX::Abstract::Vector::TwoNorm,
                                            NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> innerFluidVelUpdate =
    Teuchos::rcp(new NOX::FSI::PartialNormUpdate("velocity update",
                                                 fluidvelextract,0,
                                                 nlParams.get("Norm abs vel", 1.0e-6),
                                                 NOX::FSI::PartialNormUpdate::Scaled));

  AddStatusTest(innerFluidVel);
  fluidvelcombo->addStatusTest(innerFluidVel);
  //fluidvelcombo->addStatusTest(innerFluidVelUpdate);

  converged->addStatusTest(fluidvelcombo);

  // setup tests for fluid pressure

  std::vector<Teuchos::RCP<const Epetra_Map> > fluidpress;
  fluidpress.push_back(FluidField().PressureRowMap());
  fluidpress.push_back(Teuchos::null);
  LINALG::MultiMapExtractor fluidpressextract(*DofRowMap(),fluidpress);

  Teuchos::RCP<NOX::StatusTest::Combo> fluidpresscombo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> fluidPress =
    Teuchos::rcp(new NOX::FSI::PartialNormF("pressure",
                                            fluidpressextract,0,
                                            nlParams.get("Norm abs pres", 1.0e-6),
                                            NOX::Abstract::Vector::TwoNorm,
                                            NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> fluidPressUpdate =
    Teuchos::rcp(new NOX::FSI::PartialNormUpdate("pressure update",
                                                 fluidpressextract,0,
                                                 nlParams.get("Norm abs pres", 1.0e-6),
                                                 NOX::FSI::PartialNormUpdate::Scaled));

  AddStatusTest(fluidPress);
  fluidpresscombo->addStatusTest(fluidPress);
  //fluidpresscombo->addStatusTest(fluidPressUpdate);

  converged->addStatusTest(fluidpresscombo);

  return combo;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::ExtractFieldVectors(Teuchos::RCP<const Epetra_Vector> x,
                                                 Teuchos::RCP<const Epetra_Vector>& sx,
                                                 Teuchos::RCP<const Epetra_Vector>& fx,
                                                 Teuchos::RCP<const Epetra_Vector>& ax)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicFluidSplit::ExtractFieldVectors");

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
  AleField().Interface().InsertFSICondVector(acx, a);
  ax = a;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicFluidSplit::PrepareTimeStep()
{
  IncrementTimeAndStep();

  PrintHeader();

  if (StructureField().GetSTCAlgo() != INPAR::STR::stc_none)
    StructureField().SystemMatrix()->Reset();
  StructureField().PrepareTimeStep();
  FluidField().    PrepareTimeStep();
  AleField().      PrepareTimeStep();
}

#endif
