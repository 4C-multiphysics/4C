/*----------------------------------------------------------------------*/
/*! \file

 \brief  monolithic fluid split poroelasticity algorithms

\level 2

 *------------------------------------------------------------------------------------------------*/

#include "poroelast_monolithicfluidsplit.H"

#include <Teuchos_TimeMonitor.hpp>

#include "adapter_str_fpsiwrapper.H"
#include "adapter_fld_poro.H"
#include "coupling_adapter.H"

#include "fsi_overlapprec_fsiamg.H"
#include "fluid_utils_mapextractor.H"

#include "structure_aux.H"

#include "linalg_utils_sparse_algebra_assemble.H"
#include "linalg_utils_sparse_algebra_create.H"
#include "linalg_matrixtransform.H"

#define FLUIDSPLITAMG


POROELAST::MonolithicFluidSplit::MonolithicFluidSplit(const Epetra_Comm& comm,
    const Teuchos::ParameterList& timeparams, Teuchos::RCP<LINALG::MapExtractor> porosity_splitter)
    : MonolithicSplit(comm, timeparams, porosity_splitter)
{
  fggtransform_ = Teuchos::rcp(new LINALG::MatrixRowColTransform);
  fgitransform_ = Teuchos::rcp(new LINALG::MatrixRowTransform);
  figtransform_ = Teuchos::rcp(new LINALG::MatrixColTransform);
  cfggtransform_ = Teuchos::rcp(new LINALG::MatrixRowTransform);
  csggtransform_ = Teuchos::rcp(new LINALG::MatrixColTransform);
  cfgitransform_ = Teuchos::rcp(new LINALG::MatrixRowTransform);
  csigtransform_ = Teuchos::rcp(new LINALG::MatrixColTransform);

  // Recovering of Lagrange multiplier happens on structure field
  lambda_ = Teuchos::rcp(new Epetra_Vector(*FluidField()->Interface()->FSICondMap()));
}

void POROELAST::MonolithicFluidSplit::SetupSystem()
{
  {
    // create combined map
    std::vector<Teuchos::RCP<const Epetra_Map>> vecSpaces;

    vecSpaces.push_back(StructureField()->DofRowMap());
#ifdef FLUIDSPLITAMG
    vecSpaces.push_back(FluidField()->DofRowMap());
#else
    vecSpaces.push_back(FluidField()->Interface()->OtherMap());
#endif

    if (vecSpaces[0]->NumGlobalElements() == 0) dserror("No structure equation. Panic.");
    if (vecSpaces[1]->NumGlobalElements() == 0) dserror("No fluid equation. Panic.");

    // full Poroelasticity-map
    fullmap_ = LINALG::MultiMapExtractor::MergeMaps(vecSpaces);
    // full Poroelasticity-blockmap
    blockrowdofmap_->Setup(*fullmap_, vecSpaces);
  }

  // Switch fluid to interface split block matrix
  FluidField()->UseBlockMatrix(true);

  SetupCouplingAndMatrices();

  BuildCombinedDBCMap();

  SetupEquilibration();
}

void POROELAST::MonolithicFluidSplit::SetupRHS(bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("POROELAST::MonolithicFluidSplit::SetupRHS");

  // create full monolithic rhs vector
  rhs_ = Teuchos::rcp(new Epetra_Vector(*DofRowMap(), true));

  SetupVector(*rhs_, StructureField()->RHS(), FluidField()->RHS(), FluidField()->ResidualScaling());

  if (firstcall and evaluateinterface_)
  {
    // add additional rhs-terms depending on the solution on the interface
    // of the previous time step

    // get time integration parameters of structure an fluid time integrators
    // to enable consistent time integration among the fields
    double stiparam = StructureField()->TimIntParam();
    double ftiparam = FluidField()->TimIntParam();

    Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockf = FluidField()->BlockSystemMatrix();
    Teuchos::RCP<LINALG::BlockSparseMatrixBase> k_sf = StructFluidCouplingBlockMatrix();
    if (k_sf == Teuchos::null) dserror("expect coupling block matrix");

    LINALG::SparseMatrix& fig = blockf->Matrix(0, 1);
    LINALG::SparseMatrix& fgg = blockf->Matrix(1, 1);
    LINALG::SparseMatrix& kig = k_sf->Matrix(0, 1);
    LINALG::SparseMatrix& kgg = k_sf->Matrix(1, 1);

    Teuchos::RCP<Epetra_Vector> fveln = FluidField()->ExtractInterfaceVeln();
    double timescale = FluidField()->TimeScaling();
    double scale = FluidField()->ResidualScaling();

    Teuchos::RCP<Epetra_Vector> rhs = Teuchos::rcp(new Epetra_Vector(fig.RowMap()));

    fig.Apply(*fveln, *rhs);
    rhs->Scale(timescale * Dt());

#ifdef FLUIDSPLITAMG
    rhs = FluidField()->Interface()->InsertOtherVector(rhs);
#endif

    Extractor()->AddVector(*rhs, 1, *rhs_);  // add fluid contributions to 'f'

    rhs = Teuchos::rcp(new Epetra_Vector(fgg.RowMap()));

    fgg.Apply(*fveln, *rhs);
    rhs->Scale(scale * timescale * Dt());
    rhs->Scale(
        (1.0 - stiparam) / (1.0 - ftiparam));  // scale 'rhs' due to consistent time integration

    rhs = FluidToStructureAtInterface(rhs);
    rhs = StructureField()->Interface()->InsertFSICondVector(rhs);

    Extractor()->AddVector(*rhs, 0, *rhs_);  // add structure contributions to 'f'

    rhs = Teuchos::rcp(new Epetra_Vector(kig.RowMap()));

    kig.Apply(*fveln, *rhs);
    rhs->Scale(timescale * Dt());

    rhs = StructureField()->Interface()->InsertOtherVector(rhs);

    Extractor()->AddVector(*rhs, 0, *rhs_);  // add structure contributions to 'f'

    rhs = Teuchos::rcp(new Epetra_Vector(kgg.RowMap()));

    kgg.Apply(*fveln, *rhs);
    rhs->Scale(timescale * Dt());

    rhs = StructureField()->Interface()->InsertFSICondVector(rhs);

    Extractor()->AddVector(*rhs, 0, *rhs_);  // add structure contributions to 'f'
  }

  // store interface force onto the structure to know it in the next time step as previous force
  // in order to recover the Lagrange multiplier
  fgcur_ = FluidField()->Interface()->ExtractFSICondVector(FluidField()->RHS());
}

void POROELAST::MonolithicFluidSplit::SetupSystemMatrix(LINALG::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("POROELAST::MonolithicFluidSplit::SetupSystemMatrix");

  Teuchos::RCP<LINALG::SparseMatrix> s = StructureField()->SystemMatrix();
  if (s == Teuchos::null) dserror("expect structure matrix");
  Teuchos::RCP<LINALG::BlockSparseMatrixBase> f = FluidField()->BlockSystemMatrix();
  if (f == Teuchos::null) dserror("expect fluid block matrix");

  mat.Matrix(0, 1).Zero();
  mat.Matrix(1, 0).Zero();
#ifdef FLUIDSPLITAMG
  mat.Matrix(1, 1).Zero();
#endif

  /*----------------------------------------------------------------------*/

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  /*----------------------------------------------------------------------*/
  // structural part k_sf (3nxn)
  // build mechanical-fluid block

  // create empty matrix
  Teuchos::RCP<LINALG::BlockSparseMatrixBase> k_sf = StructFluidCouplingBlockMatrix();
  if (k_sf == Teuchos::null) dserror("expect coupling block matrix");

  // call the element and calculate the matrix block
  ApplyStrCouplMatrix(k_sf);

  /*----------------------------------------------------------------------*/
  // fluid part k_fs ( (3n+1)x3n )
  // build fluid-mechanical block

  // create empty matrix
  Teuchos::RCP<LINALG::BlockSparseMatrixBase> k_fs = FluidStructCouplingBlockMatrix();
  if (k_fs == Teuchos::null) dserror("expect coupling block matrix");

  // call the element and calculate the matrix block
  ApplyFluidCouplMatrix(k_fs);

  /*----------------------------------------------------------------------*/

  k_fs->Complete();
  k_sf->Complete();

  s->UnComplete();

  /*----------------------------------------------------------------------*/

  if (evaluateinterface_)
  {
    double timescale = FluidField()->TimeScaling();

    (*figtransform_)(f->FullRowMap(), f->FullColMap(), f->Matrix(0, 1), timescale,
        ADAPTER::CouplingSlaveConverter(*icoupfs_), k_fs->Matrix(0, 1), true, true);

    (*csggtransform_)(f->FullRowMap(), f->FullColMap(), k_sf->Matrix(1, 1), timescale,
        ADAPTER::CouplingSlaveConverter(*icoupfs_), *s, true, true);

    (*csigtransform_)(f->FullRowMap(), f->FullColMap(), k_sf->Matrix(0, 1), timescale,
        ADAPTER::CouplingSlaveConverter(*icoupfs_), *s, true, true);
  }

  /*----------------------------------------------------------------------*/
  // pure fluid part
  // uncomplete because the fluid interface can have more connections than the
  // structural one. (Tet elements in fluid can cause this.) We should do
  // this just once...
#ifdef FLUIDSPLITAMG
  mat.Matrix(1, 1).Add(f->Matrix(0, 0), false, 1., 0.0);
  Teuchos::RCP<LINALG::SparseMatrix> eye = LINALG::Eye(*FluidField()->Interface()->FSICondMap());
  mat.Matrix(1, 1).Add(*eye, false, 1., 1.0);
#else
  f->Matrix(0, 0).UnComplete();
  mat.Assign(1, 1, View, f->Matrix(0, 0));
#endif

  // fluid coupling part
  mat.Matrix(1, 0).Add(k_fs->Matrix(0, 0), false, 1.0, 0.0);
  mat.Matrix(1, 0).Add(k_fs->Matrix(0, 1), false, 1.0, 1.0);

  // pure structure part
  mat.Assign(0, 0, LINALG::View, *s);

  // structure coupling part
  mat.Matrix(0, 1).Add(k_sf->Matrix(0, 0), false, 1.0, 0.0);
  mat.Matrix(0, 1).Add(k_sf->Matrix(1, 0), false, 1.0, 1.0);
  /*----------------------------------------------------------------------*/
  // done. make sure all blocks are filled.
  mat.Complete();

  fgicur_ = Teuchos::rcp(new LINALG::SparseMatrix(f->Matrix(1, 0)));
  fggcur_ = Teuchos::rcp(new LINALG::SparseMatrix(f->Matrix(1, 1)));
  cgicur_ = Teuchos::rcp(new LINALG::SparseMatrix(k_fs->Matrix(1, 0)));
  cggcur_ = Teuchos::rcp(new LINALG::SparseMatrix(k_fs->Matrix(1, 1)));
}

void POROELAST::MonolithicFluidSplit::SetupVector(Epetra_Vector& f,
    Teuchos::RCP<const Epetra_Vector> sv, Teuchos::RCP<const Epetra_Vector> fv, double fluidscale)
{
  // extract the inner and boundary dofs of all fields

  Teuchos::RCP<Epetra_Vector> fov = FluidField()->Interface()->ExtractOtherVector(fv);
#ifdef FLUIDSPLITAMG
  fov = FluidField()->Interface()->InsertOtherVector(fov);
#endif

  Extractor()->InsertVector(*sv, 0, f);

  Extractor()->InsertVector(*fov, 1, f);  // add fluid contributions to 'f'
}

void POROELAST::MonolithicFluidSplit::ExtractFieldVectors(Teuchos::RCP<const Epetra_Vector> x,
    Teuchos::RCP<const Epetra_Vector>& sx, Teuchos::RCP<const Epetra_Vector>& fx, bool firstcall)
{
  TEUCHOS_FUNC_TIME_MONITOR("POROELAST::MonolithicFluidSplit::ExtractFieldVectors");

  // process structure unknowns
  sx = Extractor()->ExtractVector(x, 0);

  // process fluid unknowns
  if (evaluateinterface_)
  {
    Teuchos::RCP<const Epetra_Vector> scx = StructureField()->Interface()->ExtractFSICondVector(sx);

    Teuchos::RCP<Epetra_Vector> fcx = StructureToFluidAtInterface(scx);
    Teuchos::RCP<const Epetra_Vector> fox = Extractor()->ExtractVector(x, 1);
#ifdef FLUIDSPLITAMG
    fox = FluidField()->Interface()->ExtractOtherVector(fox);
#endif

    {
      double timescale = FluidField()->TimeScaling();
      fcx->Scale(timescale);
    }

    Teuchos::RCP<Epetra_Vector> f = FluidField()->Interface()->InsertOtherVector(fox);
    FluidField()->Interface()->InsertFSICondVector(fcx, f);

    Teuchos::RCP<const Epetra_Vector> zeros = Teuchos::rcp(new const Epetra_Vector(f->Map(), true));
    LINALG::ApplyDirichlettoSystem(f, zeros, *(FluidField()->GetDBCMapExtractor()->CondMap()));

    fx = f;

    // Store field vectors to know them later on as previous quantities
    Teuchos::RCP<Epetra_Vector> sox = StructureField()->Interface()->ExtractOtherVector(sx);
    if (solipre_ != Teuchos::null)
      ddiinc_->Update(1.0, *sox, -1.0, *solipre_, 0.0);  // compute current iteration increment
    else
      ddiinc_ = Teuchos::rcp(new Epetra_Vector(*sox));  // first iteration increment

    solipre_ = sox;  // store current step increment

    if (solgvelpre_ != Teuchos::null)
      duginc_->Update(1.0, *fcx, -1.0, *solgvelpre_, 0.0);  // compute current iteration increment
    else
      duginc_ = Teuchos::rcp(new Epetra_Vector(*fcx));  // first iteration increment

    solgvelpre_ = fcx;  // store current step increment

    if (solivelpre_ != Teuchos::null)
      duiinc_->Update(1.0, *fox, -1.0, *solivelpre_, 0.0);  // compute current iteration increment
    else
      duiinc_ = Teuchos::rcp(new Epetra_Vector(*fox));  // first iteration increment

    solivelpre_ = fox;  // store current step increment
  }
  else
    fx = Extractor()->ExtractVector(x, 1);
}

void POROELAST::MonolithicFluidSplit::RecoverLagrangeMultiplierAfterTimeStep()
{
  TEUCHOS_FUNC_TIME_MONITOR("POROELAST::MonolithicFluidSplit::RecoverLagrangeMultiplier");

  if (evaluateinterface_)
  {
    // get time integration parameter of structural time integrator
    // to enable consistent time integration among the fields
    double ftiparam = FluidField()->TimIntParam();
    double timescale = FluidField()->TimeScaling();

    // store the product F_{\GammaI} \Delta u_I^{n+1} in here
    Teuchos::RCP<Epetra_Vector> fgiddi =
        LINALG::CreateVector(*FluidField()->Interface()->FSICondMap(), true);
    // compute the above mentioned product
    fgicur_->Multiply(false, *duiinc_, *fgiddi);

    // store the product C_{\GammaI} \Delta d_I^{n+1} in here
    Teuchos::RCP<Epetra_Vector> sgiddi =
        LINALG::CreateVector(*FluidField()->Interface()->FSICondMap(), true);
    // compute the above mentioned product
    cgicur_->Multiply(false, *ddiinc_, *sgiddi);

    // store the product F_{\Gamma\Gamma} \Delta u_\Gamma^{n+1} in here
    Teuchos::RCP<Epetra_Vector> sggddg =
        LINALG::CreateVector(*FluidField()->Interface()->FSICondMap(), true);
    // compute the above mentioned product
    fggcur_->Multiply(false, *duginc_, *sggddg);

    // store the prodcut C_{\Gamma\Gamma} \Delta d_\Gamma^{n+1} in here
    Teuchos::RCP<Epetra_Vector> cggddg =
        LINALG::CreateVector(*FluidField()->Interface()->FSICondMap(), true);
    // compute the above mentioned product
    cggcur_->Multiply(false, *duginc_, *cggddg);
    cggddg->Scale(1.0 / timescale);

    // Update the Lagrange multiplier:
    /* \lambda^{n+1} =  1/b * [ - a*\lambda^n - f_\Gamma^S
     *                          - F_{\Gamma I} \Delta u_I
     *                          - C_{\Gamma I} \Delta d_I
     *                          - F_{\Gamma\Gamma} \Delta u_\Gamma]
     *                          - C_{\Gamma\Gamma} * Delta t / 2 * \Delta u_\Gamma]
     */
    lambda_->Update(1.0, *fgcur_, -ftiparam);
    lambda_->Update(-1.0, *fgiddi, -1.0, *sggddg, 1.0);
    lambda_->Update(-1.0, *fgiddi, -1.0, *cggddg, 1.0);
    lambda_->Scale(1 / (1.0 - ftiparam));  // entire Lagrange multiplier is divided by (1.-ftiparam)
  }
}
