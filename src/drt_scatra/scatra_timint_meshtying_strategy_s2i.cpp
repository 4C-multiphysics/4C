/*!----------------------------------------------------------------------
\file scatra_timint_meshtying_strategy_s2i.cpp

\brief Scatra-scatra interface coupling strategy for standard scalar transport problems

<pre>
Maintainer: Rui Fang
            fang@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/
            089 - 289-15251
</pre>

*----------------------------------------------------------------------*/
#include "scatra_timint_meshtying_strategy_s2i.H"
#include "scatra_timint_implicit.H"
#include "scatra_timint_meshtying_strategy_s2i_elch.H"

#include "../drt_adapter/adapter_coupling.H"
#include "../drt_adapter/adapter_coupling_mortar.H"

#include "../drt_fluid/fluid_utils.H"

#include "../drt_fsi/fsi_matrixtransform.H"

#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../drt_mortar/mortar_coupling3d_classes.H"
#include "../drt_mortar/mortar_interface.H"
#include "../drt_mortar/mortar_projector.H"
#include "../drt_mortar/mortar_utils.H"

#include "../drt_scatra_ele/scatra_ele_action.H"
#include "../drt_scatra_ele/scatra_ele_calc_utils.H"
#include "../drt_scatra_ele/scatra_ele_parameter_timint.H"

#include "../drt_volmortar/volmortar_shape.H"

#include "../linalg/linalg_solver.H"

/*----------------------------------------------------------------------*
 | constructor                                               fang 12/14 |
 *----------------------------------------------------------------------*/
SCATRA::MeshtyingStrategyS2I::MeshtyingStrategyS2I(
    SCATRA::ScaTraTimIntImpl*       scatratimint,   //! scalar transport time integrator
    const Teuchos::ParameterList&   parameters      //! input parameters for scatra-scatra interface coupling
    ) :
MeshtyingStrategyBase(scatratimint),
interfacemaps_(Teuchos::null),
blockmaps_(Teuchos::null),
blockmaps_slave_(Teuchos::null),
blockmaps_master_(Teuchos::null),
icoup_(Teuchos::null),
icoupmortar_(),
imortarcells_(),
islavematrix_(Teuchos::null),
imastermatrix_(Teuchos::null),
mortartype_(DRT::INPUT::IntegralValue<INPAR::S2I::MortarType>(parameters,"MORTARTYPE")),
D_(Teuchos::null),
M_(Teuchos::null),
E_(Teuchos::null),
P_(Teuchos::null),
Q_(Teuchos::null),
lm_(Teuchos::null),
extendedmaps_(Teuchos::null),
lmresidual_(Teuchos::null),
lmincrement_(Teuchos::null),
imastertoslaverowtransform_(Teuchos::null),
islavetomastercoltransform_(Teuchos::null),
islavetomasterrowtransform_(Teuchos::null),
islavetomasterrowcoltransform_(Teuchos::null),
islaveresidual_(Teuchos::null),
imasterresidual_(Teuchos::null),
imasterphinp_(Teuchos::null),
invrowsums_(Teuchos::null),
invcolsums_(Teuchos::null),
lmside_(DRT::INPUT::IntegralValue<INPAR::S2I::InterfaceSides>(parameters,"LMSIDE")),
matrixtype_(DRT::INPUT::IntegralValue<INPAR::S2I::MatrixType>(parameters,"MATRIXTYPE")),
slaveconditions_(),
rowequilibration_(
    DRT::INPUT::IntegralValue<INPAR::S2I::EquilibrationMethods>(parameters,"EQUILIBRATION") == INPAR::S2I::equilibration_rows
    or
    DRT::INPUT::IntegralValue<INPAR::S2I::EquilibrationMethods>(parameters,"EQUILIBRATION") == INPAR::S2I::equilibration_full
    ),
colequilibration_(
    DRT::INPUT::IntegralValue<INPAR::S2I::EquilibrationMethods>(parameters,"EQUILIBRATION") == INPAR::S2I::equilibration_columns
    or
    DRT::INPUT::IntegralValue<INPAR::S2I::EquilibrationMethods>(parameters,"EQUILIBRATION") == INPAR::S2I::equilibration_full
    ),
slaveonly_(DRT::INPUT::IntegralValue<bool>(parameters,"SLAVEONLY"))
{
  return;
} // SCATRA::MeshtyingStrategyS2I::MeshtyingStrategyS2I


/*-----------------------------------------------------------------------*
 | condense global system of equations                        fang 01/16 |
 *-----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::CondenseMatAndRHS(
    const Teuchos::RCP<LINALG::SparseOperator>&   systemmatrix,       //!< system matrix
    const Teuchos::RCP<Epetra_Vector>&            residual,           //!< residual vector
    const bool                                    calcinittimederiv   //!< flag for calculation of initial time derivative
    ) const
{
  switch(mortartype_)
  {
    case INPAR::S2I::mortar_condensed_bubnov:
    {
      // extract global system matrix
      Teuchos::RCP<LINALG::SparseMatrix> sparsematrix = scatratimint_->SystemMatrix();
      if(sparsematrix == Teuchos::null)
        dserror("System matrix is not a sparse matrix!");

      if(lmside_ == INPAR::S2I::side_slave)
      {
        // initialize temporary matrix for slave-side rows of global system matrix
        LINALG::SparseMatrix sparsematrixrowsslave(*interfacemaps_->Map(1),81);

        // loop over all slave-side rows of global system matrix
        for(int slavedoflid=0; slavedoflid<interfacemaps_->Map(1)->NumMyElements(); ++slavedoflid)
        {
          // determine global ID of current matrix row
          const int slavedofgid = interfacemaps_->Map(1)->GID(slavedoflid);
          if(slavedofgid < 0)
            dserror("Couldn't find local ID %d in map!",slavedoflid);

          // extract current matrix row from global system matrix
          const int length = sparsematrix->EpetraMatrix()->NumGlobalEntries(slavedofgid);
          int numentries(0);
          std::vector<double> values(length,0.);
          std::vector<int> indices(length,0);
          if(sparsematrix->EpetraMatrix()->ExtractGlobalRowCopy(slavedofgid,length,numentries,&values[0],&indices[0]))
            dserror("Cannot extract matrix row with global ID %d from global system matrix!",slavedofgid);

          // copy current matrix row of global system matrix into temporary matrix
          if(sparsematrixrowsslave.EpetraMatrix()->InsertGlobalValues(slavedofgid,numentries,&values[0],&indices[0]) < 0)
            dserror("Cannot insert matrix row with global ID %d into temporary matrix!",slavedofgid);
        }

        // finalize temporary matrix with slave-side rows of global system matrix
        sparsematrixrowsslave.Complete(*interfacemaps_->FullMap(),*interfacemaps_->Map(1));

        // zero out slave-side rows of global system matrix after having extracted them into temporary matrix
        // and replace them by projected slave-side rows including interface contributions
        sparsematrix->Complete();
        sparsematrix->ApplyDirichlet(*interfacemaps_->Map(1),false);
        sparsematrix->Add(*LINALG::MLMultiply(*Q_,true,sparsematrixrowsslave,false,false,false,true),false,1.,1.);
        // during calculation of initial time derivative, standard global system matrix is replaced by global mass matrix,
        // and hence interface contributions must not be included
        if(!calcinittimederiv)
          sparsematrix->Add(*islavematrix_,false,1.,1.);

        // add projected slave-side rows to master-side rows of global system matrix
        sparsematrix->Add(*LINALG::MLMultiply(*P_,true,sparsematrixrowsslave,false,false,false,true),false,1.,1.);

        // extract slave-side entries of global residual vector
        Teuchos::RCP<Epetra_Vector> residualslave = interfacemaps_->ExtractVector(scatratimint_->Residual(),1);

        // replace slave-side entries of global residual vector by projected slave-side entries including interface contributions
        Epetra_Vector Q_residualslave(*interfacemaps_->Map(1));
        if(Q_->Multiply(true,*residualslave,Q_residualslave))
          dserror("Matrix-vector multiplication failed!");
        interfacemaps_->InsertVector(Q_residualslave,1,*scatratimint_->Residual());
        interfacemaps_->AddVector(islaveresidual_,1,scatratimint_->Residual());

        // add projected slave-side entries to master-side entries of global residual vector
        Epetra_Vector P_residualslave(*interfacemaps_->Map(2));
        if(P_->Multiply(true,*residualslave,P_residualslave))
          dserror("Matrix-vector multiplication failed!");
        interfacemaps_->AddVector(P_residualslave,2,*scatratimint_->Residual());
      }

      else
      {
        // initialize temporary matrix for master-side rows of global system matrix
        LINALG::SparseMatrix sparsematrixrowsmaster(*interfacemaps_->Map(2),81);

        // loop over all master-side rows of global system matrix
        for(int masterdoflid=0; masterdoflid<interfacemaps_->Map(2)->NumMyElements(); ++masterdoflid)
        {
          // determine global ID of current matrix row
          const int masterdofgid = interfacemaps_->Map(2)->GID(masterdoflid);
          if(masterdofgid < 0)
            dserror("Couldn't find local ID %d in map!",masterdoflid);

          // extract current matrix row from global system matrix
          const int length = sparsematrix->EpetraMatrix()->NumGlobalEntries(masterdofgid);
          int numentries(0);
          std::vector<double> values(length,0.);
          std::vector<int> indices(length,0);
          if(sparsematrix->EpetraMatrix()->ExtractGlobalRowCopy(masterdofgid,length,numentries,&values[0],&indices[0]))
            dserror("Cannot extract matrix row with global ID %d from global system matrix!",masterdofgid);

          // copy current matrix row of global system matrix into temporary matrix
          if(sparsematrixrowsmaster.EpetraMatrix()->InsertGlobalValues(masterdofgid,numentries,&values[0],&indices[0]) < 0)
            dserror("Cannot insert matrix row with global ID %d into temporary matrix!",masterdofgid);
        }

        // finalize temporary matrix with master-side rows of global system matrix
        sparsematrixrowsmaster.Complete(*interfacemaps_->FullMap(),*interfacemaps_->Map(2));

        // zero out master-side rows of global system matrix after having extracted them into temporary matrix
        // and replace them by projected master-side rows including interface contributions
        sparsematrix->Complete();
        sparsematrix->ApplyDirichlet(*interfacemaps_->Map(2),false);
        sparsematrix->Add(*LINALG::MLMultiply(*Q_,true,sparsematrixrowsmaster,false,false,false,true),false,1.,1.);
        // during calculation of initial time derivative, standard global system matrix is replaced by global mass matrix,
        // and hence interface contributions must not be included
        if(!calcinittimederiv)
          sparsematrix->Add(*imastermatrix_,false,1.,1.);

        // add projected master-side rows to slave-side rows of global system matrix
        sparsematrix->Add(*LINALG::MLMultiply(*P_,true,sparsematrixrowsmaster,false,false,false,true),false,1.,1.);

        // extract master-side entries of global residual vector
        Teuchos::RCP<Epetra_Vector> residualmaster = interfacemaps_->ExtractVector(scatratimint_->Residual(),2);

        // replace master-side entries of global residual vector by projected master-side entries including interface contributions
        Epetra_Vector Q_residualmaster(*interfacemaps_->Map(2));
        if(Q_->Multiply(true,*residualmaster,Q_residualmaster))
          dserror("Matrix-vector multiplication failed!");
        interfacemaps_->InsertVector(Q_residualmaster,2,*scatratimint_->Residual());
        interfacemaps_->AddVector(*imasterresidual_,2,*scatratimint_->Residual());

        // add projected master-side entries to slave-side entries of global residual vector
        Epetra_Vector P_residualmaster(*interfacemaps_->Map(1));
        if(P_->Multiply(true,*residualmaster,P_residualmaster))
          dserror("Matrix-vector multiplication failed!");
        interfacemaps_->AddVector(P_residualmaster,1,*scatratimint_->Residual());
      }

      break;
    }

    case INPAR::S2I::mortar_none:
    case INPAR::S2I::mortar_standard:
    case INPAR::S2I::mortar_saddlepoint_petrov:
    case INPAR::S2I::mortar_saddlepoint_bubnov:
    case INPAR::S2I::mortar_condensed_petrov:
    {
      // do nothing in these cases
      break;
    }

    default:
    {
      dserror("Type of mortar meshtying for scatra-scatra interface coupling not recognized!");
      break;
    }
  }

  return;
}


/*-----------------------------------------------------------------------*
 | evaluate scatra-scatra interface coupling conditions       fang 10/14 |
 *-----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::EvaluateMeshtying()
{
  // time measurement: evaluate condition 'S2ICoupling'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'S2ICoupling'");

  // extract scatra-scatra coupling conditions from discretization
  std::vector<DRT::Condition*> conditions;
  scatratimint_->Discretization()->GetCondition("S2ICoupling",conditions);

  switch(mortartype_)
  {
  case INPAR::S2I::mortar_none:
  {
    // create parameter list
    Teuchos::ParameterList condparams;

    // action for elements
    condparams.set<int>("action",SCATRA::bd_calc_s2icoupling);

    // set global state vectors according to time-integration scheme
    scatratimint_->Discretization()->ClearState();
    scatratimint_->AddTimeIntegrationSpecificVectors();

    // fill interface state vector imasterphinp_ with transformed master dof values and add to discretization
    interfacemaps_->InsertVector(icoup_->MasterToSlave(interfacemaps_->ExtractVector(*(scatratimint_->Phiafnp()),2)),1,imasterphinp_);
    scatratimint_->Discretization()->SetState("imasterphinp",imasterphinp_);

    // evaluate scatra-scatra interface coupling at time t_{n+1} or t_{n+alpha_F}
    islavematrix_->Zero();
    if(not slaveonly_)
      imastermatrix_->Zero();
    islaveresidual_->PutScalar(0.);
    for(unsigned icondition=0; icondition<conditions.size(); ++icondition)
    {
      if(conditions[icondition]->GetInt("interface side") == INPAR::S2I::side_slave)
      {
        if(not slaveonly_)
          scatratimint_->Discretization()->EvaluateCondition(condparams,islavematrix_,imastermatrix_,islaveresidual_,Teuchos::null,Teuchos::null,"S2ICoupling",conditions[icondition]->GetInt("ConditionID"));
        else
          scatratimint_->Discretization()->EvaluateCondition(condparams,islavematrix_,Teuchos::null,islaveresidual_,Teuchos::null,Teuchos::null,"S2ICoupling",conditions[icondition]->GetInt("ConditionID"));
      }
    }
    scatratimint_->Discretization()->ClearState();

    // finalize interface matrices
    islavematrix_->Complete();
    if(not slaveonly_)
      imastermatrix_->Complete();

    // assemble global system matrix depending on matrix type
    switch(matrixtype_)
    {
      case INPAR::S2I::matrix_sparse:
      {
        // check matrix
        Teuchos::RCP<LINALG::SparseMatrix> systemmatrix = scatratimint_->SystemMatrix();
        if(systemmatrix == Teuchos::null)
          dserror("System matrix is not a sparse matrix!");

        // assemble linearizations of slave fluxes w.r.t. slave dofs into global system matrix
        systemmatrix->Add(*islavematrix_,false,1.,1.);

        if(not slaveonly_)
        {
          // transform linearizations of slave fluxes w.r.t. master dofs and assemble into global system matrix
          (*islavetomastercoltransform_)(imastermatrix_->RowMap(),imastermatrix_->ColMap(),*imastermatrix_,1.,
              ADAPTER::CouplingSlaveConverter(*icoup_),*systemmatrix,true,true);

          // derive linearizations of master fluxes w.r.t. slave dofs and assemble into global system matrix
          (*islavetomasterrowtransform_)(*islavematrix_,-1.,ADAPTER::CouplingSlaveConverter(*icoup_),*systemmatrix,true);

          // derive linearizations of master fluxes w.r.t. master dofs and assemble into global system matrix
          (*islavetomasterrowcoltransform_)(*imastermatrix_,-1.,ADAPTER::CouplingSlaveConverter(*icoup_),ADAPTER::CouplingSlaveConverter(*icoup_),*systemmatrix,true,true);
        }

        // In case the interface linearizations and residuals are evaluated on slave side only,
        // we now apply a standard meshtying algorithm to condense out the master-side degrees of freedom.
        else
        {
          // initialize temporary matrix for master-side rows of system matrix
          LINALG::SparseMatrix systemmatrixrowsmaster(*icoup_->MasterDofMap(),81);

          // loop over all master-side rows of system matrix
          for(int masterdoflid=0; masterdoflid<icoup_->MasterDofMap()->NumMyElements(); ++masterdoflid)
          {
            // determine global ID of current matrix row
            const int masterdofgid = icoup_->MasterDofMap()->GID(masterdoflid);
            if(masterdofgid < 0)
              dserror("Couldn't find local ID %d in map!",masterdoflid);

            // extract current matrix row from system matrix
            const int length = systemmatrix->EpetraMatrix()->NumGlobalEntries(masterdofgid);
            int numentries(0);
            std::vector<double> values(length,0.);
            std::vector<int> indices(length,0);
            if(systemmatrix->EpetraMatrix()->ExtractGlobalRowCopy(masterdofgid,length,numentries,&values[0],&indices[0]) != 0)
              dserror("Cannot extract matrix row with global ID %d from system matrix!",masterdofgid);

            // copy current matrix row of system matrix into temporary matrix
            if(systemmatrixrowsmaster.EpetraMatrix()->InsertGlobalValues(masterdofgid,numentries,&values[0],&indices[0]) < 0)
              dserror("Cannot insert matrix row with global ID %d into temporary matrix!",masterdofgid);
          }

          // zero out master-side rows of system matrix and put a one on the main diagonal
          systemmatrix->Complete();
          systemmatrix->ApplyDirichlet(*icoup_->MasterDofMap(),true);
          systemmatrix->UnComplete();

          // loop over all master-side rows of system matrix
          for(int masterdoflid=0; masterdoflid<icoup_->MasterDofMap()->NumMyElements(); ++masterdoflid)
          {
            // determine global ID of current matrix row
            const int masterdofgid = icoup_->MasterDofMap()->GID(masterdoflid);
            if(masterdofgid < 0)
              dserror("Couldn't find local ID %d in map!",masterdoflid);

            // determine global ID of associated slave-side matrix column
            const int slavedofgid = icoup_->PermSlaveDofMap()->GID(masterdoflid);
            if(slavedofgid < 0)
              dserror("Couldn't find local ID %d in permuted map!",masterdoflid);

            // insert value -1. into intersection of master-side row and slave-side column in system matrix
            // this effectively forces the master-side degree of freedom to assume the same value as the slave-side degree of freedom
            const double value(-1.);
            if(systemmatrix->EpetraMatrix()->InsertGlobalValues(masterdofgid,1,&value,&slavedofgid) < 0)
              dserror("Cannot insert value -1. into matrix row with global ID %d and matrix column with global ID %d!",masterdofgid,slavedofgid);

            // insert zero into intersection of master-side row and slave-side column in temporary matrix
            // this prevents the system matrix from changing its graph when calling this function again during the next Newton iteration
            const double zero(0.);
            if(systemmatrixrowsmaster.EpetraMatrix()->InsertGlobalValues(masterdofgid,1,&zero,&slavedofgid) < 0)
              dserror("Cannot insert zero into matrix row with global ID %d and matrix column with global ID %d!",masterdofgid,slavedofgid);
          }

          // finalize temporary matrix with master-side rows of system matrix
          systemmatrixrowsmaster.Complete(*scatratimint_->DofRowMap(),*icoup_->MasterDofMap());

          // add master-side rows of system matrix to corresponding slave-side rows to finalize matrix condensation of master-side degrees of freedom
          (*imastertoslaverowtransform_)(
              systemmatrixrowsmaster,
              1.,
              ADAPTER::CouplingMasterConverter(*icoup_),
              *systemmatrix,
              true
              );
        }

        break;
      }

      case INPAR::S2I::matrix_block_geometry:
      {
        // check matrix
        Teuchos::RCP<LINALG::BlockSparseMatrixBase> blocksystemmatrix = scatratimint_->BlockSystemMatrix();
        if(blocksystemmatrix == Teuchos::null)
          dserror("System matrix is not a block matrix!");

        // assemble linearizations of slave fluxes w.r.t. slave dofs into global system matrix
        blocksystemmatrix->Matrix(1,1).Add(*islavematrix_,false,1.,1.);

        if(not slaveonly_)
        {
          // transform linearizations of slave fluxes w.r.t. master dofs and assemble into global system matrix
          (*islavetomastercoltransform_)(imastermatrix_->RowMap(),imastermatrix_->ColMap(),*imastermatrix_,1.,
              ADAPTER::CouplingSlaveConverter(*icoup_),blocksystemmatrix->Matrix(1,2));

          // derive linearizations of master fluxes w.r.t. slave dofs and assemble into global system matrix
          (*islavetomasterrowtransform_)(*islavematrix_,-1.,ADAPTER::CouplingSlaveConverter(*icoup_),blocksystemmatrix->Matrix(2,1));

          // derive linearizations of master fluxes w.r.t. master dofs and assemble into global system matrix
          (*islavetomasterrowcoltransform_)(*imastermatrix_,-1.,ADAPTER::CouplingSlaveConverter(*icoup_),ADAPTER::CouplingSlaveConverter(*icoup_),blocksystemmatrix->Matrix(2,2),true,true);
        }

        // safety check
        else
          dserror("Scatra-scatra interface coupling with evaluation of interface linearizations and residuals on slave side only is not yet available for block system matrices!");

        break;
      }

      case INPAR::S2I::matrix_block_condition:
      case INPAR::S2I::matrix_block_condition_dof:
      {
        // check matrix
        Teuchos::RCP<LINALG::BlockSparseMatrixBase> blocksystemmatrix = scatratimint_->BlockSystemMatrix();
        if(blocksystemmatrix == Teuchos::null)
          dserror("System matrix is not a block matrix!");

        Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockkss(islavematrix_->Split<LINALG::DefaultBlockMatrixStrategy>(*blockmaps_slave_,*blockmaps_slave_));
        blockkss->Complete();

        // assemble interface block matrix into global block system matrix
        blocksystemmatrix->Add(*blockkss,false,1.,1.);

        if(not slaveonly_)
        {
          Teuchos::RCP<LINALG::SparseMatrix> ksm(Teuchos::rcp(new LINALG::SparseMatrix(*icoup_->SlaveDofMap(),81,false)));
          Teuchos::RCP<LINALG::SparseMatrix> kms(Teuchos::rcp(new LINALG::SparseMatrix(*icoup_->MasterDofMap(),81,false)));
          Teuchos::RCP<LINALG::SparseMatrix> kmm(Teuchos::rcp(new LINALG::SparseMatrix(*icoup_->MasterDofMap(),81,false)));

          // transform linearizations of slave fluxes w.r.t. master dofs
          (*islavetomastercoltransform_)(imastermatrix_->RowMap(),imastermatrix_->ColMap(),*imastermatrix_,1.,ADAPTER::CouplingSlaveConverter(*icoup_),*ksm);
          ksm->Complete(*icoup_->MasterDofMap(),*icoup_->SlaveDofMap());

          // derive linearizations of master fluxes w.r.t. slave dofs
          (*islavetomasterrowtransform_)(*islavematrix_,-1.,ADAPTER::CouplingSlaveConverter(*icoup_),*kms);
          kms->Complete(*icoup_->SlaveDofMap(),*icoup_->MasterDofMap());

          // derive linearizations of master fluxes w.r.t. master dofs
          (*islavetomasterrowcoltransform_)(*imastermatrix_,-1.,ADAPTER::CouplingSlaveConverter(*icoup_),ADAPTER::CouplingSlaveConverter(*icoup_),*kmm);
          kmm->Complete();

          Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockksm(ksm->Split<LINALG::DefaultBlockMatrixStrategy>(*blockmaps_master_,*blockmaps_slave_));
          blockksm->Complete();
          Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockkms(kms->Split<LINALG::DefaultBlockMatrixStrategy>(*blockmaps_slave_,*blockmaps_master_));
          blockkms->Complete();
          Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockkmm(kmm->Split<LINALG::DefaultBlockMatrixStrategy>(*blockmaps_master_,*blockmaps_master_));
          blockkmm->Complete();

          // assemble interface block matrices into global block system matrix
          blocksystemmatrix->Add(*blockksm,false,1.,1.);
          blocksystemmatrix->Add(*blockkms,false,1.,1.);
          blocksystemmatrix->Add(*blockkmm,false,1.,1.);
        }

        // safety check
        else
          dserror("Scatra-scatra interface coupling with evaluation of interface linearizations and residuals on slave side only is not yet available for block system matrices!");

        break;
      }

      default:
      {
        dserror("Type of global system matrix for scatra-scatra interface coupling not recognized!");
        break;
      }
    }

    // assemble slave residuals into global residual vector
    interfacemaps_->AddVector(islaveresidual_,1,scatratimint_->Residual());

    if(not slaveonly_)
      // transform master residuals and assemble into global residual vector
      interfacemaps_->AddVector(icoup_->SlaveToMaster(islaveresidual_),2,scatratimint_->Residual(),-1.);

    // In case the interface linearizations and residuals are evaluated on slave side only,
    // we now apply a standard meshtying algorithm to condense out the master-side degrees of freedom.
    else
    {
      // initialize temporary vector for master-side entries of residual vector
      Teuchos::RCP<Epetra_Vector> residualmaster = Teuchos::rcp(new Epetra_Vector(*icoup_->MasterDofMap()));

      // loop over all master-side entries of residual vector
      for(int masterdoflid=0; masterdoflid<icoup_->MasterDofMap()->NumMyElements(); ++masterdoflid)
      {
        // determine global ID of current vector entry
        const int masterdofgid = icoup_->MasterDofMap()->GID(masterdoflid);
        if(masterdofgid < 0)
          dserror("Couldn't find local ID %d in map!",masterdoflid);

        // copy current vector entry into temporary vector
        if(residualmaster->ReplaceGlobalValue(masterdofgid,0,(*scatratimint_->Residual())[scatratimint_->DofRowMap()->LID(masterdofgid)]))
          dserror("Cannot insert residual vector entry with global ID %d into temporary vector!",masterdofgid);

        // zero out current vector entry
        if(scatratimint_->Residual()->ReplaceGlobalValue(masterdofgid,0,0.))
          dserror("Cannot insert zero into residual vector entry with global ID %d!",masterdofgid);
      }

      // add master-side entries of residual vector to corresponding slave-side entries to finalize vector condensation of master-side degrees of freedom
      interfacemaps_->AddVector(icoup_->MasterToSlave(residualmaster),1,scatratimint_->Residual());
    }

    break;
  }

  case INPAR::S2I::mortar_standard:
  case INPAR::S2I::mortar_saddlepoint_petrov:
  case INPAR::S2I::mortar_saddlepoint_bubnov:
  case INPAR::S2I::mortar_condensed_petrov:
  case INPAR::S2I::mortar_condensed_bubnov:
  {
    // initialize auxiliary system matrix and vector for slave side
    if(mortartype_ == INPAR::S2I::mortar_standard or lmside_ == INPAR::S2I::side_slave)
    {
      islavematrix_->Zero();
      islaveresidual_->PutScalar(0.);
    }

    // initialize auxiliary system matrix and vector for master side
    if(mortartype_ == INPAR::S2I::mortar_standard or lmside_ == INPAR::S2I::side_master)
    {
      imastermatrix_->Zero();
      imasterresidual_->PutScalar(0.);
    }

    // loop over all scatra-scatra coupling interfaces
    for(std::map<const int,DRT::Condition* const>::iterator islavecondition=slaveconditions_.begin(); islavecondition!=slaveconditions_.end(); ++islavecondition)
    {
      // extract mortar interface discretization
      DRT::Discretization& idiscret = icoupmortar_[islavecondition->first]->Interface()->Discret();

      // export global state vector to mortar interface
      Teuchos::RCP<Epetra_Vector> iphinp = Teuchos::rcp(new Epetra_Vector(*idiscret.DofColMap(),false));
      LINALG::Export(*scatratimint_->Phiafnp(),*iphinp);
      idiscret.SetState("iphinp",iphinp);

      // create parameter list for mortar integration cells
      Teuchos::ParameterList params;

      // set action
      params.set<int>("action",INPAR::S2I::evaluate_condition);

      // add current condition to parameter list
      params.set<DRT::Condition*>("condition",islavecondition->second);

      // loop over all mortar integration cells at current interface
      for(unsigned icell=0; icell<imortarcells_[islavecondition->first].size(); ++icell)
      {
        // safety check
        if(imortarcells_[islavecondition->first][icell] == Teuchos::null)
          dserror("Invalid mortar integration cell!");

        // evaluate current cell
        EvaluateMortarCell(islavematrix_,imastermatrix_,Teuchos::null,islaveresidual_,imasterresidual_,*imortarcells_[islavecondition->first][icell],params,idiscret);
      }
    }

    // finalize auxiliary system matrix and residual vector for slave side
    if(mortartype_ == INPAR::S2I::mortar_standard or lmside_ == INPAR::S2I::side_slave)
      islavematrix_->Complete(*interfacemaps_->FullMap(),*interfacemaps_->Map(1));

    // finalize auxiliary system matrix and residual vector for master side
    if(mortartype_ == INPAR::S2I::mortar_standard or lmside_ == INPAR::S2I::side_master)
    {
      imastermatrix_->Complete(*interfacemaps_->FullMap(),*interfacemaps_->Map(2));
      if(imasterresidual_->GlobalAssemble(Add,true))
        dserror("Assembly of auxiliary residual vector for master residuals not successful!");
    }

    // extract global system matrix from time integrator
    const Teuchos::RCP<LINALG::SparseMatrix> systemmatrix = scatratimint_->SystemMatrix();
    if(systemmatrix == Teuchos::null)
      dserror("System matrix is not a sparse matrix!");

    // assemble interface contributions into global system of equations
    switch(mortartype_)
    {
      case INPAR::S2I::mortar_standard:
      {
        systemmatrix->Add(*islavematrix_,false,1.,1.);
        systemmatrix->Add(*imastermatrix_,false,1.,1.);
        interfacemaps_->AddVector(islaveresidual_,1,scatratimint_->Residual());
        interfacemaps_->AddVector(*imasterresidual_,2,*scatratimint_->Residual());

        break;
      }

      case INPAR::S2I::mortar_saddlepoint_petrov:
      case INPAR::S2I::mortar_saddlepoint_bubnov:
      {
        if(lmside_ == INPAR::S2I::side_slave)
        {
          // assemble slave-side interface contributions into global residual vector
          Epetra_Vector islaveresidual(*interfacemaps_->Map(1));
          if(D_->Multiply(true,*lm_,islaveresidual))
            dserror("Matrix-vector multiplication failed!");
          interfacemaps_->AddVector(islaveresidual,1,*scatratimint_->Residual(),-1.);

          // assemble master-side interface contributions into global residual vector
          Epetra_Vector imasterresidual(*interfacemaps_->Map(2));
          if(M_->Multiply(true,*lm_,imasterresidual))
            dserror("Matrix-vector multiplication failed!");
          interfacemaps_->AddVector(imasterresidual,2,*scatratimint_->Residual());

          // build constraint residual vector associated with Lagrange multiplier dofs
          Epetra_Vector ilmresidual(*islaveresidual_);
          if(ilmresidual.ReplaceMap(*extendedmaps_->Map(1)))
            dserror("Couldn't replace map!");
          if(lmresidual_->Update(1.,ilmresidual,0.))
            dserror("Vector update failed!");
          if(E_->Multiply(true,*lm_,ilmresidual))
            dserror("Matrix-vector multiplication failed!");
          if(lmresidual_->Update(1.,ilmresidual,1.))
            dserror("Vector update failed!");
        }
        else
        {
          // assemble slave-side interface contributions into global residual vector
          Epetra_Vector islaveresidual(*interfacemaps_->Map(1));
          if(M_->Multiply(true,*lm_,islaveresidual))
            dserror("Matrix-vector multiplication failed!");
          interfacemaps_->AddVector(islaveresidual,1,*scatratimint_->Residual());

          // assemble master-side interface contributions into global residual vector
          Epetra_Vector imasterresidual(*interfacemaps_->Map(2));
          if(D_->Multiply(true,*lm_,imasterresidual))
            dserror("Matrix-vector multiplication failed!");
          interfacemaps_->AddVector(imasterresidual,2,*scatratimint_->Residual(),-1.);

          // build constraint residual vector associated with Lagrange multiplier dofs
          Epetra_Vector ilmresidual(Copy,*imasterresidual_,0);
          if(ilmresidual.ReplaceMap(*extendedmaps_->Map(1)))
            dserror("Couldn't replace map!");
          if(lmresidual_->Update(1.,ilmresidual,0.))
            dserror("Vector update failed!");
          if(E_->Multiply(true,*lm_,ilmresidual))
            dserror("Matrix-vector multiplication failed!");
          if(lmresidual_->Update(1.,ilmresidual,1.))
            dserror("Vector update failed!");
        }

        break;
      }

      case INPAR::S2I::mortar_condensed_petrov:
      {
        if(lmside_ == INPAR::S2I::side_slave)
        {
          systemmatrix->Add(*islavematrix_,false,1.,1.);
          systemmatrix->Add(*LINALG::MLMultiply(*P_,true,*islavematrix_,false,false,false,true),false,-1.,1.);
          interfacemaps_->AddVector(islaveresidual_,1,scatratimint_->Residual());
          Epetra_Vector imasterresidual(*interfacemaps_->Map(2));
          if(P_->Multiply(true,*islaveresidual_,imasterresidual))
            dserror("Matrix-vector multiplication failed!");
          interfacemaps_->AddVector(imasterresidual,2,*scatratimint_->Residual(),-1.);
        }
        else
        {
          systemmatrix->Add(*LINALG::MLMultiply(*P_,true,*imastermatrix_,false,false,false,true),false,-1.,1.);
          systemmatrix->Add(*imastermatrix_,false,1.,1.);
          Epetra_Vector islaveresidual(*interfacemaps_->Map(1));
          if(P_->Multiply(true,*imasterresidual_,islaveresidual))
            dserror("Matrix-vector multiplication failed!");
          interfacemaps_->AddVector(islaveresidual,1,*scatratimint_->Residual(),-1.);
          interfacemaps_->AddVector(*imasterresidual_,2,*scatratimint_->Residual());
        }

        break;
      }

      case INPAR::S2I::mortar_condensed_bubnov:
      {
        // during calculation of initial time derivative, condensation must not be performed here, but after assembly of the modified global system of equations
        if(scatratimint_->Step() > 0)
          CondenseMatAndRHS(systemmatrix,scatratimint_->Residual());

        break;
      }

      default:
      {
        dserror("Not yet implemented!");
        break;
      }
    }

    break;
  }

  default:
  {
    dserror("Not yet implemented!");
    break;
  }
  }

  return;
} // SCATRA::MeshtyingStrategyS2I::EvaluateMeshtying


/*--------------------------------------------------------------------------------------*
 | evaluate single mortar integration cell                                   fang 01/16 |
 *--------------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::EvaluateMortarCell(
    Teuchos::RCP<LINALG::SparseMatrix>   imatrix1,          //!< interface matrix 1
    Teuchos::RCP<LINALG::SparseMatrix>   imatrix2,          //!< interface matrix 2
    Teuchos::RCP<LINALG::SparseMatrix>   imatrix3,          //!< interface matrix 3
    Teuchos::RCP<Epetra_Vector>          islaveresidual,    //!< slave-side residual vector
    Teuchos::RCP<Epetra_FEVector>        imasterresidual,   //!< master-side residual vector
    MORTAR::IntCell&                     cell,              //!< mortar integration cell
    const Teuchos::ParameterList&        params,            //!< parameter list
    const DRT::Discretization&           idiscret           //!< interface discretization
    ) const
{
  // extract slave-side element associated with current cell
  MORTAR::MortarElement* slaveelement = dynamic_cast<MORTAR::MortarElement*>(idiscret.gElement(cell.GetSlaveId()));
  if(!slaveelement)
    dserror("Couldn't extract slave element from mortar interface discretization!");

  // extract master-side element associated with current cell
  MORTAR::MortarElement* masterelement = dynamic_cast<MORTAR::MortarElement*>(idiscret.gElement(cell.GetMasterId()));
  if(!masterelement)
    dserror("Couldn't extract master element from mortar interface discretization!");

  // safety check
  if(!slaveelement->IsSlave() or masterelement->IsSlave())
    dserror("Something is wrong with the slave-master element pairing!");

  // construct slave-side and master-side location arrays
  DRT::Element::LocationArray la_slave(idiscret.NumDofSets());
  slaveelement->LocationVector(idiscret,la_slave,false);
  DRT::Element::LocationArray la_master(idiscret.NumDofSets());
  masterelement->LocationVector(idiscret,la_master,false);

  // evaluate single mortar integration cell
  SCATRA::MortarCellFactory::MortarCellCalc(*slaveelement,*masterelement,this,mortartype_,lmside_)->Evaluate(
      imatrix1,
      imatrix2,
      imatrix3,
      islaveresidual,
      imasterresidual,
      idiscret,
      params,
      *slaveelement,
      *masterelement,
      cell,
      la_slave,
      la_master
      );

  return;
}


/*------------------------------------------------------------------------------------------------------------*
 | provide instance of mortar cell evaluation class of particular slave-side discretization type   fang 01/16 |
 *------------------------------------------------------------------------------------------------------------*/
SCATRA::MortarCellInterface* SCATRA::MortarCellFactory::MortarCellCalc(
    const MORTAR::MortarElement&        slaveelement,    //!< slave-side mortar element
    const MORTAR::MortarElement&        masterelement,   //!< master-side mortar element
    const MeshtyingStrategyS2I* const   strategy,        //!< scatra-scatra interface coupling strategy
    const INPAR::S2I::MortarType&       mortartype,      //!< flag for meshtying method
    const INPAR::S2I::InterfaceSides&   lmside           //!< flag for interface side underlying Lagrange multiplier definition
    )
{
  // extract number of slave-side degrees of freedom per node
  const int numdofpernode_slave = slaveelement.NumDofPerNode(*slaveelement.Nodes()[0]);

  switch(slaveelement.Shape())
  {
    case DRT::Element::tri3:
    {
      return MortarCellCalc<DRT::Element::tri3>(masterelement,strategy,mortartype,lmside,numdofpernode_slave);
      break;
    }

    default:
    {
      dserror("Invalid slave-side discretization type!");
      break;
    }
  }

  return NULL;
}


/*-----------------------------------------------------------------------------------------------------------------------------*
 | provide instance of mortar cell evaluation class of particular slave-side and master-side discretization types   fang 01/16 |
 *-----------------------------------------------------------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS>
SCATRA::MortarCellInterface* SCATRA::MortarCellFactory::MortarCellCalc(
    const MORTAR::MortarElement&        masterelement,        //!< master-side mortar element
    const MeshtyingStrategyS2I* const   strategy,             //!< scatra-scatra interface coupling strategy
    const INPAR::S2I::MortarType&       mortartype,           //!< flag for meshtying method
    const INPAR::S2I::InterfaceSides&   lmside,               //!< flag for interface side underlying Lagrange multiplier definition
    const int&                          numdofpernode_slave   //!< number of slave-side degrees of freedom per node
    )
{
  // extract number of master-side degrees of freedom per node
  const int numdofpernode_master = masterelement.NumDofPerNode(*masterelement.Nodes()[0]);

  switch(masterelement.Shape())
  {
    case DRT::Element::tri3:
    {
      return MortarCellCalc<distypeS,DRT::Element::tri3>(strategy,mortartype,lmside,numdofpernode_slave,numdofpernode_master);
      break;
    }

    case DRT::Element::quad4:
    {
      return MortarCellCalc<distypeS,DRT::Element::quad4>(strategy,mortartype,lmside,numdofpernode_slave,numdofpernode_master);
      break;
    }

    default:
    {
      dserror("Invalid master-side discretization type!");
      break;
    }
  }

  return NULL;
}


/*--------------------------------------------------------------------------------------*
 | provide specific instance of mortar cell evaluation class                 fang 01/16 |
 *--------------------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
SCATRA::MortarCellInterface* SCATRA::MortarCellFactory::MortarCellCalc(
    const MeshtyingStrategyS2I* const   strategy,              //!< scatra-scatra interface coupling strategy
    const INPAR::S2I::MortarType&       mortartype,            //!< flag for meshtying method
    const INPAR::S2I::InterfaceSides&   lmside,                //!< flag for interface side underlying Lagrange multiplier definition
    const int&                          numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
    const int&                          numdofpernode_master   //!< number of master-side degrees of freedom per node
    )
{
  if(dynamic_cast<const MeshtyingStrategyS2IElch*>(strategy))
    return SCATRA::MortarCellCalcElch<distypeS,distypeM>::Instance(mortartype,lmside,numdofpernode_slave,numdofpernode_master);
  else if(dynamic_cast<const MeshtyingStrategyS2I*>(strategy))
    return SCATRA::MortarCellCalc<distypeS,distypeM>::Instance(mortartype,lmside,numdofpernode_slave,numdofpernode_master);
  else
    dserror("Unknown type of scatra-scatra interface coupling strategy!");

  return NULL;
}


/*------------------------------------------------------------------------*
 | instantiate strategy for Newton-Raphson convergence check   fang 02/16 |
 *------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::InitConvCheckStrategy()
{
  if(mortartype_ == INPAR::S2I::mortar_saddlepoint_petrov or mortartype_ == INPAR::S2I::mortar_saddlepoint_bubnov)
    convcheckstrategy_ = Teuchos::rcp(new SCATRA::ConvCheckStrategyS2ILM(scatratimint_->ScatraParameterList()->sublist("NONLINEAR")));
  else
    convcheckstrategy_ = Teuchos::rcp(new SCATRA::ConvCheckStrategyStd(scatratimint_->ScatraParameterList()->sublist("NONLINEAR")));

  return;
} // SCATRA::MeshtyingStrategyS2I::InitConvCheckStrategy


/*----------------------------------------------------------------------*
 | perform setup of scatra-scatra interface coupling         fang 10/14 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::InitMeshtying()
{
  // instantiate strategy for Newton-Raphson convergence check
  InitConvCheckStrategy();

  // extract scatra-scatra coupling conditions from discretization
  std::vector<DRT::Condition*> conditions(0,NULL);
  scatratimint_->Discretization()->GetCondition("S2ICoupling",conditions);
  slaveconditions_.clear();
  std::map<const int,DRT::Condition* const> masterconditions;
  for(unsigned icondition=0; icondition<conditions.size(); ++icondition)
  {
    DRT::Condition* const condition = conditions[icondition];
    const int condid = condition->GetInt("ConditionID");

    switch(condition->GetInt("interface side"))
    {
      case INPAR::S2I::side_slave:
      {
        if(slaveconditions_.find(condid) == slaveconditions_.end())
          slaveconditions_.insert(std::pair<const int,DRT::Condition* const>(condid,condition));
        else
          dserror("Cannot have multiple slave-side scatra-scatra interface coupling conditions with the same ID!");
        break;
      }

      case INPAR::S2I::side_master:
      {
        if(masterconditions.find(condid) == masterconditions.end())
          masterconditions.insert(std::pair<const int,DRT::Condition* const>(condid,condition));
        else
          dserror("Cannot have multiple master-side scatra-scatra interface coupling conditions with the same ID!");
        break;
      }

      default:
      {
        dserror("Invalid scatra-scatra interface coupling condition!");
        break;
      }
    }
  }

  // determine type of mortar meshtying
  switch(mortartype_)
  {
  // setup scatra-scatra interface coupling for interfaces with pairwise overlapping interface nodes
  case INPAR::S2I::mortar_none:
  {
    // overwrite IDs of master-side scatra-scatra interface coupling conditions with the value -1
    // to prevent them from being evaluated when calling EvaluateCondition on the discretization
    for(std::map<const int,DRT::Condition* const>::iterator imastercondition=masterconditions.begin(); imastercondition!=masterconditions.end(); ++imastercondition)
      imastercondition->second->Add("ConditionID",-1);

    // initialize int vectors for global ids of slave and master interface nodes
    std::vector<int> islavenodegidvec;
    std::vector<int> imasternodegidvec;

    // fill vectors
    for (std::map<const int,DRT::Condition* const>::iterator islavecondition=slaveconditions_.begin(); islavecondition!=slaveconditions_.end(); ++islavecondition)
    {
      const std::vector<int>* islavenodegids = islavecondition->second->Nodes();

      for (unsigned islavenode=0; islavenode<islavenodegids->size(); ++islavenode)
      {
        const int islavenodegid = (*islavenodegids)[islavenode];

        // insert global id of current node into associated vector only if node is owned by current processor
        // need to make sure that node is stored on current processor, otherwise cannot resolve "->Owner()"
        if(scatratimint_->Discretization()->HaveGlobalNode(islavenodegid) and scatratimint_->Discretization()->gNode(islavenodegid)->Owner() == scatratimint_->Discretization()->Comm().MyPID())
          islavenodegidvec.push_back(islavenodegid);
      }
    }
    for (std::map<const int,DRT::Condition* const>::iterator imastercondition=masterconditions.begin(); imastercondition!=masterconditions.end(); ++imastercondition)
    {
      const std::vector<int>* imasternodegids = imastercondition->second->Nodes();

      for (unsigned imasternode=0; imasternode<imasternodegids->size(); ++imasternode)
      {
        const int imasternodegid = (*imasternodegids)[imasternode];

        // insert global id of current node into associated vector only if node is owned by current processor
        // need to make sure that node is stored on current processor, otherwise cannot resolve "->Owner()"
        if(scatratimint_->Discretization()->HaveGlobalNode(imasternodegid) and scatratimint_->Discretization()->gNode(imasternodegid)->Owner() == scatratimint_->Discretization()->Comm().MyPID())
          imasternodegidvec.push_back(imasternodegid);
      }
    }

    // remove potential duplicates from vectors
    std::sort(islavenodegidvec.begin(),islavenodegidvec.end());
    islavenodegidvec.erase(unique(islavenodegidvec.begin(),islavenodegidvec.end()),islavenodegidvec.end());
    std::sort(imasternodegidvec.begin(),imasternodegidvec.end());
    imasternodegidvec.erase(unique(imasternodegidvec.begin(),imasternodegidvec.end()),imasternodegidvec.end());

    // initialize non-mortar coupling adapter
    if(scatratimint_->NumScal() < 1)
      dserror("Number of transported scalars not correctly set!");
    icoup_ = Teuchos::rcp(new ADAPTER::Coupling());
    icoup_->SetupCoupling(*(scatratimint_->Discretization()),*(scatratimint_->Discretization()),imasternodegidvec,islavenodegidvec,scatratimint_->NumDofPerNode(),true,1.e-8);

    // generate interior and interface maps
    Teuchos::RCP<Epetra_Map> ifullmap = LINALG::MergeMap(icoup_->SlaveDofMap(),icoup_->MasterDofMap(),false);
    std::vector<Teuchos::RCP<const Epetra_Map> > imaps;
    imaps.push_back(LINALG::SplitMap(*(scatratimint_->Discretization()->DofRowMap()),*ifullmap));
    imaps.push_back(icoup_->SlaveDofMap());
    imaps.push_back(icoup_->MasterDofMap());

    // initialize global map extractor
    interfacemaps_ = Teuchos::rcp(new LINALG::MultiMapExtractor(*(scatratimint_->Discretization()->DofRowMap()),imaps));
    interfacemaps_->CheckForValidMapExtractor();

    // initialize interface vector
    // Although the interface vector only contains the transformed master interface dofs, we still initialize it with
    // the full DofRowMap of the discretization to make it work for parallel computations.
    imasterphinp_ = LINALG::CreateVector(*(scatratimint_->Discretization()->DofRowMap()),false);

    // initialize auxiliary system matrices and associated transformation operators
    islavematrix_ = Teuchos::rcp(new LINALG::SparseMatrix(*(icoup_->SlaveDofMap()),81));
    imastermatrix_ = Teuchos::rcp(new LINALG::SparseMatrix(*(icoup_->SlaveDofMap()),81));
    if(not slaveonly_)
    {
      islavetomastercoltransform_ = Teuchos::rcp(new FSI::UTILS::MatrixColTransform);
      islavetomasterrowtransform_ = Teuchos::rcp(new FSI::UTILS::MatrixRowTransform);
      islavetomasterrowcoltransform_ = Teuchos::rcp(new FSI::UTILS::MatrixRowColTransform);
    }
    else
      imastertoslaverowtransform_ = Teuchos::rcp(new FSI::UTILS::MatrixRowTransform);

    // initialize auxiliary residual vector
    islaveresidual_ = Teuchos::rcp(new Epetra_Vector(*(icoup_->SlaveDofMap())));

    // further initializations depending on type of global system matrix
    switch(matrixtype_)
    {
    case INPAR::S2I::matrix_sparse:
      // nothing needs to be done in this case
      break;

    case INPAR::S2I::matrix_block_condition:
    case INPAR::S2I::matrix_block_condition_dof:
    case INPAR::S2I::matrix_block_geometry:
    {
      // safety check
      if(!scatratimint_->Solver()->Params().isSublist("AMGnxn Parameters"))
        dserror("Global system matrix with block structure requires AMGnxn block preconditioner!");

      // initialize map extractors associated with blocks of global system matrix
      BuildBlockMapExtractors();

      // feed AMGnxn block preconditioner with null space information for each block of global block system matrix
      BuildBlockNullSpaces();

      break;
    }
    }

    break;
  }

  // setup scatra-scatra interface coupling for interfaces with non-overlapping interface nodes
  case INPAR::S2I::mortar_standard:
  case INPAR::S2I::mortar_saddlepoint_petrov:
  case INPAR::S2I::mortar_saddlepoint_bubnov:
  case INPAR::S2I::mortar_condensed_petrov:
  case INPAR::S2I::mortar_condensed_bubnov:
  {
    // extract parameter list for mortar coupling from problem instance
    const Teuchos::ParameterList& mortarparams = DRT::Problem::Instance()->MortarCouplingParams();

    // safety checks
    if(DRT::INPUT::IntegralValue<INPAR::MORTAR::ParRedist>(mortarparams,"PARALLEL_REDIST") != INPAR::MORTAR::parredist_none)
      dserror("Parallel redistribution not yet implemented for scatra-scatra interface coupling!");
    if(DRT::INPUT::IntegralValue<INPAR::MORTAR::MeshRelocation>(mortarparams,"MESH_RELOCATION") != INPAR::MORTAR::relocation_none)
      dserror("Mesh relocation not yet implemented for scatra-scatra interface coupling!");

    // initialize empty interface maps
    Teuchos::RCP<Epetra_Map> imastermap = Teuchos::rcp(new Epetra_Map(0,0,scatratimint_->Discretization()->Comm()));
    Teuchos::RCP<Epetra_Map> islavemap = Teuchos::rcp(new Epetra_Map(0,0,scatratimint_->Discretization()->Comm()));
    Teuchos::RCP<Epetra_Map> ifullmap = Teuchos::rcp(new Epetra_Map(0,0,scatratimint_->Discretization()->Comm()));

    // loop over all slave-side scatra-scatra interface coupling conditions
    for(std::map<const int,DRT::Condition* const>::iterator islavecondition=slaveconditions_.begin(); islavecondition!=slaveconditions_.end(); ++islavecondition)
    {
      // initialize maps for row nodes associated with current condition
      std::map<int,DRT::Node*> masternodes;
      std::map<int,DRT::Node*> slavenodes;

      // initialize maps for column nodes associated with current condition
      std::map<int,DRT::Node*> mastergnodes;
      std::map<int,DRT::Node*> slavegnodes;

      // initialize maps for elements associated with current condition
      std::map<int,Teuchos::RCP<DRT::Element> > masterelements;
      std::map<int,Teuchos::RCP<DRT::Element> > slaveelements;

      // extract current slave-side and associated master-side scatra-scatra interface coupling conditions
      std::vector<DRT::Condition*> mastercondition(1,masterconditions[islavecondition->first]);
      std::vector<DRT::Condition*> slavecondition(1,islavecondition->second);

      // fill maps
      DRT::UTILS::FindConditionObjects(*scatratimint_->Discretization(),masternodes,mastergnodes,masterelements,mastercondition);
      DRT::UTILS::FindConditionObjects(*scatratimint_->Discretization(),slavenodes,slavegnodes,slaveelements,slavecondition);

      // initialize mortar coupling adapter
      icoupmortar_[islavecondition->first] = Teuchos::rcp(new ADAPTER::CouplingMortar());
      std::vector<int> coupleddof(scatratimint_->NumDofPerNode(),1);
      icoupmortar_[islavecondition->first]->SetupInterface(
          scatratimint_->Discretization(),
          scatratimint_->Discretization(),
          coupleddof,
          mastergnodes,
          slavegnodes,
          masterelements,
          slaveelements,
          scatratimint_->Discretization()->Comm()
          );

      // generate mortar integration cells
      imortarcells_[islavecondition->first] = icoupmortar_[islavecondition->first]->EvaluateGeometry();

      // build interface maps
      imastermap = LINALG::MergeMap(imastermap,icoupmortar_[islavecondition->first]->MasterDofMap(),false);
      islavemap = LINALG::MergeMap(islavemap,icoupmortar_[islavecondition->first]->SlaveDofMap(),false);
      ifullmap = LINALG::MergeMap(ifullmap,LINALG::MergeMap(icoupmortar_[islavecondition->first]->MasterDofMap(),icoupmortar_[islavecondition->first]->SlaveDofMap(),false),false);
    }

    // generate interior and interface maps
    std::vector<Teuchos::RCP<const Epetra_Map> > imaps;
    imaps.push_back(LINALG::SplitMap(*(scatratimint_->Discretization()->DofRowMap()),*ifullmap));
    imaps.push_back(islavemap);
    imaps.push_back(imastermap);

    // initialize global map extractor
    interfacemaps_ = Teuchos::rcp(new LINALG::MultiMapExtractor(*(scatratimint_->Discretization()->DofRowMap()),imaps));
    interfacemaps_->CheckForValidMapExtractor();

    if(mortartype_ == INPAR::S2I::mortar_standard or lmside_ == INPAR::S2I::side_slave)
    {
      // initialize auxiliary system matrix for slave side
      islavematrix_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(1),81));

      // initialize auxiliary residual vector for slave side
      islaveresidual_ = Teuchos::rcp(new Epetra_Vector(*interfacemaps_->Map(1)));
    }

    if(mortartype_ == INPAR::S2I::mortar_standard or lmside_ == INPAR::S2I::side_master)
    {
      // initialize auxiliary system matrix for master side
      imastermatrix_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(2),81,true,false,LINALG::SparseMatrix::FE_MATRIX));

      // initialize auxiliary residual vector for master side
      imasterresidual_ = Teuchos::rcp(new Epetra_FEVector(*interfacemaps_->Map(2)));
    }

    switch(mortartype_)
    {
      case INPAR::S2I::mortar_saddlepoint_petrov:
      case INPAR::S2I::mortar_saddlepoint_bubnov:
      case INPAR::S2I::mortar_condensed_petrov:
      case INPAR::S2I::mortar_condensed_bubnov:
      {
        if(lmside_ == INPAR::S2I::side_slave)
        {
          D_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(1),81));
          M_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(1),81));
          if(mortartype_ == INPAR::S2I::mortar_saddlepoint_bubnov or mortartype_ == INPAR::S2I::mortar_condensed_bubnov)
            E_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(1),81));
        }
        else
        {
          D_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(2),81,true,false,LINALG::SparseMatrix::FE_MATRIX));
          M_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(2),81,true,false,LINALG::SparseMatrix::FE_MATRIX));
          if(mortartype_ == INPAR::S2I::mortar_saddlepoint_bubnov or mortartype_ == INPAR::S2I::mortar_condensed_bubnov)
            E_ = Teuchos::rcp(new LINALG::SparseMatrix(*interfacemaps_->Map(2),81,true,false,LINALG::SparseMatrix::FE_MATRIX));
        }

        // loop over all scatra-scatra coupling interfaces
        for(std::map<const int,DRT::Condition* const>::iterator islavecondition=slaveconditions_.begin(); islavecondition!=slaveconditions_.end(); ++islavecondition)
        {
          // extract mortar interface discretization
          DRT::Discretization& idiscret = icoupmortar_[islavecondition->first]->Interface()->Discret();

          // create parameter list for mortar integration cells
          Teuchos::ParameterList params;

          // set action
          params.set<int>("action",INPAR::S2I::evaluate_mortar_matrices);

          // loop over all mortar integration cells at current interface
          for(unsigned icell=0; icell<imortarcells_[islavecondition->first].size(); ++icell)
          {
            // safety check
            if(imortarcells_[islavecondition->first][icell] == Teuchos::null)
              dserror("Invalid mortar integration cell!");

            // evaluate current cell
            EvaluateMortarCell(
                D_,
                M_,
                E_,
                Teuchos::null,
                Teuchos::null,
                *imortarcells_[islavecondition->first][icell],
                params,
                idiscret
                );
          }
        }

        // finalize mortar matrices D, M, and E
        D_->Complete();
        if(lmside_ == INPAR::S2I::side_slave)
          M_->Complete(*interfacemaps_->Map(2),*interfacemaps_->Map(1));
        else
          M_->Complete(*interfacemaps_->Map(1),*interfacemaps_->Map(2));
        if(mortartype_ == INPAR::S2I::mortar_saddlepoint_bubnov or mortartype_ == INPAR::S2I::mortar_condensed_bubnov)
          E_->Complete();

        switch(mortartype_)
        {
          case INPAR::S2I::mortar_condensed_petrov:
          case INPAR::S2I::mortar_condensed_bubnov:
          {
            // set up mortar projector P
            Teuchos::RCP<Epetra_Vector> D_diag(Teuchos::null);
            if(lmside_ == INPAR::S2I::side_slave)
              D_diag = LINALG::CreateVector(*interfacemaps_->Map(1));
            else
              D_diag = LINALG::CreateVector(*interfacemaps_->Map(2));
            if(D_->ExtractDiagonalCopy(*D_diag))
              dserror("Couldn't extract main diagonal from mortar matrix D!");
            if(D_diag->Reciprocal(*D_diag))
              dserror("Couldn't invert main diagonal entries of mortar matrix D!");;
            P_ = Teuchos::rcp(new LINALG::SparseMatrix(*M_));
            if(P_->LeftScale(*D_diag))
              dserror("Setup of mortar projector P failed!");

            // free memory
            D_ = Teuchos::null;
            M_ = Teuchos::null;

            if(mortartype_ == INPAR::S2I::mortar_condensed_bubnov)
            {
              // set up mortar projector Q
              Q_ = Teuchos::rcp(new LINALG::SparseMatrix(*E_));
              if(Q_->LeftScale(*D_diag))
                dserror("Setup of mortar projector Q failed!");

              // free memory
              E_ = Teuchos::null;
            }

            break;
          }

          case INPAR::S2I::mortar_saddlepoint_petrov:
          case INPAR::S2I::mortar_saddlepoint_bubnov:
          {
            // determine number of Lagrange multiplier dofs owned by each processor
            const Epetra_Comm& comm(scatratimint_->Discretization()->Comm());
            const int numproc(comm.NumProc());
            const int mypid(comm.MyPID());
            std::vector<int> localnumlmdof(numproc,0);
            std::vector<int> globalnumlmdof(numproc,0);
            if(lmside_ == INPAR::S2I::side_slave)
              localnumlmdof[mypid] = interfacemaps_->Map(1)->NumMyElements();
            else
              localnumlmdof[mypid] = interfacemaps_->Map(2)->NumMyElements();
            comm.SumAll(&localnumlmdof[0],&globalnumlmdof[0],numproc);

            // for each processor, determine offset of minimum Lagrange multiplier dof GID w.r.t. maximum standard dof GID
            int offset(0);
            for(int ipreviousproc=0; ipreviousproc<mypid; ++ipreviousproc)
              offset += globalnumlmdof[ipreviousproc];

            // for each processor, determine Lagrange multiplier dof GIDs
            std::vector<int> lmdofgids(globalnumlmdof[mypid],0);
            for(int lmdoflid=0; lmdoflid<globalnumlmdof[mypid]; ++lmdoflid)
              lmdofgids[lmdoflid] = scatratimint_->DofRowMap()->MaxAllGID()+1+offset+lmdoflid;

            // build Lagrange multiplier dofrowmap
            const Teuchos::RCP<Epetra_Map> lmdofrowmap = Teuchos::rcp(new Epetra_Map(-1,(int) lmdofgids.size(),&lmdofgids[0],0,comm));

            // initialize vectors associated with Lagrange multiplier dofs
            lm_ = Teuchos::rcp(new Epetra_Vector(*lmdofrowmap));
            lmresidual_ = Teuchos::rcp(new Epetra_Vector(*lmdofrowmap));
            lmincrement_ = Teuchos::rcp(new Epetra_Vector(*lmdofrowmap));

            // initialize extended map extractor
            Teuchos::RCP<Epetra_Map> extendedmap = LINALG::MergeMap(*(scatratimint_->Discretization()->DofRowMap()),*lmdofrowmap,false);
            extendedmaps_ = Teuchos::rcp(new LINALG::MapExtractor(*extendedmap,lmdofrowmap,scatratimint_->Discretization()->DofRowMap()));
            extendedmaps_->CheckForValidMapExtractor();

            // transform range map of mortar matrices D and M from slave-side dofrowmap to Lagrange multiplier dofrowmap
            D_ = MORTAR::MatrixRowTransformGIDs(D_,lmdofrowmap);
            M_ = MORTAR::MatrixRowTransformGIDs(M_,lmdofrowmap);

            if(mortartype_ == INPAR::S2I::mortar_saddlepoint_petrov)
            {
              // transform domain map of mortar matrix D from slave-side dofrowmap to Lagrange multiplier dofrowmap and store transformed matrix as mortar matrix E
              E_ = MORTAR::MatrixColTransformGIDs(D_,lmdofrowmap);
            }
            else
            {
              // transform domain and range maps of mortar matrix E from slave-side dofrowmap to Lagrange multiplier dofrowmap
              E_ = MORTAR::MatrixRowColTransformGIDs(E_,lmdofrowmap,lmdofrowmap);
            }

            break;
          }

          default:
          {
            dserror("Invalid type of mortar meshtying!");
            break;
          }
        }

        break;
      }

      default:
      {
        // do nothing
        break;
      }
    }

    break;
  }

  default:
  {
    dserror("Type of mortar meshtying for scatra-scatra interface coupling not recognized!");
    break;
  }
  }

  // initialize vectors for row and column sums of global system matrix if necessary
  if(rowequilibration_)
    invrowsums_ = Teuchos::rcp(new Epetra_Vector(*scatratimint_->Discretization()->DofRowMap(),false));
  if(colequilibration_)
    invcolsums_ = Teuchos::rcp(new Epetra_Vector(*scatratimint_->Discretization()->DofRowMap(),false));

  return;
} // SCATRA::MeshtyingStrategyS2I::InitMeshtying


/*----------------------------------------------------------------------------------*
 | build map extractors associated with blocks of global system matrix   fang 07/15 |
 *----------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::BuildBlockMapExtractors()
{
  if(matrixtype_ == INPAR::S2I::matrix_block_condition or matrixtype_ == INPAR::S2I::matrix_block_condition_dof)
  {
    // extract domain partitioning conditions from discretization
    std::vector<Teuchos::RCP<DRT::Condition> > partitioningconditions;
    scatratimint_->Discretization()->GetCondition("S2ICouplingPartitioning",partitioningconditions);

    // safety check
    if(!partitioningconditions.size())
      dserror("For block preconditioning based on domain partitioning, at least one associated condition needs to be specified in the input file!");

    // build maps associated with blocks of global system matrix
    std::vector<Teuchos::RCP<const Epetra_Map> > blockmaps;
    BuildBlockMaps(partitioningconditions,blockmaps);

    // initialize full map extractor associated with blocks of global system matrix
    blockmaps_ = Teuchos::rcp(new LINALG::MultiMapExtractor(*(scatratimint_->Discretization()->DofRowMap()),blockmaps));
    blockmaps_->CheckForValidMapExtractor();

    // initialize reduced interface map extractors associated with blocks of global system matrix
    const unsigned nblocks = blockmaps.size();
    std::vector<Teuchos::RCP<const Epetra_Map> > blockmaps_slave(nblocks);
    std::vector<Teuchos::RCP<const Epetra_Map> > blockmaps_master(nblocks);
    for(unsigned iblock=0; iblock<nblocks; ++iblock)
    {
      std::vector<Teuchos::RCP<const Epetra_Map> > maps(2);
      maps[0] = blockmaps[iblock];
      maps[1] = icoup_->SlaveDofMap();
      blockmaps_slave[iblock] = LINALG::MultiMapExtractor::IntersectMaps(maps);
      maps[1] = icoup_->MasterDofMap();
      blockmaps_master[iblock] = LINALG::MultiMapExtractor::IntersectMaps(maps);
    }
    blockmaps_slave_ = Teuchos::rcp(new LINALG::MultiMapExtractor(*icoup_->SlaveDofMap(),blockmaps_slave));
    blockmaps_slave_->CheckForValidMapExtractor();
    blockmaps_master_ = Teuchos::rcp(new LINALG::MultiMapExtractor(*icoup_->MasterDofMap(),blockmaps_master));
    blockmaps_master_->CheckForValidMapExtractor();
  }

  else if(matrixtype_ == INPAR::S2I::matrix_block_geometry)
    // matrix block map extractor equals interface map extractor in this case
    blockmaps_ = interfacemaps_;

  return;
} // SCATRA::MeshtyingStrategyS2I::BuildBlockMapExtractors


/*----------------------------------------------------------------------------*
 | build maps associated with blocks of global system matrix       fang 06/15 |
 *----------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::BuildBlockMaps(
    const std::vector<Teuchos::RCP<DRT::Condition> >&   partitioningconditions,   //! domain partitioning conditions
    std::vector<Teuchos::RCP<const Epetra_Map> >&       blockmaps                 //! empty vector for maps to be built
    ) const
{
  if(matrixtype_ == INPAR::S2I::matrix_block_condition)
  {
    // extract number of domain partitioning conditions
    const unsigned ncond = partitioningconditions.size();

    // prepare vector for maps to be built
    blockmaps.resize(ncond,Teuchos::null);

    // loop over all domain partitioning conditions
    for(unsigned icond=0; icond<ncond; ++icond)
    {
      // initialize set for dof IDs associated with current partitioning condition
      std::set<int> dofids;

      // extract nodes associated with current domain partitioning condition
      const std::vector<int>* nodegids = partitioningconditions[icond]->Nodes();

      // loop over all nodes associated with current domain partitioning condition
      for (unsigned inode=0; inode<nodegids->size(); ++inode)
      {
        // extract global ID of current node
        const int nodegid = (*nodegids)[inode];

        // consider current node only if node is owned by current processor
        // need to make sure that node is stored on current processor, otherwise cannot resolve "->Owner()"
        if(scatratimint_->Discretization()->HaveGlobalNode(nodegid) and scatratimint_->Discretization()->gNode(nodegid)->Owner() == scatratimint_->Discretization()->Comm().MyPID())
        {
          // add dof IDs associated with current node to corresponding set
          const std::vector<int> nodedofs = scatratimint_->Discretization()->Dof(scatratimint_->Discretization()->gNode(nodegid));
          std::copy(nodedofs.begin(),nodedofs.end(),std::inserter(dofids,dofids.end()));
        }
      }

      // transform set for dof IDs into vector and then into Epetra map
      int nummyelements(0);
      int* myglobalelements(NULL);
      std::vector<int> dofidvec;
      if(dofids.size() > 0)
      {
        dofidvec.reserve(dofids.size());
        dofidvec.assign(dofids.begin(),dofids.end());
        nummyelements = dofidvec.size();
        myglobalelements = &(dofidvec[0]);
      }
      blockmaps[icond] = Teuchos::rcp(new Epetra_Map(-1,nummyelements,myglobalelements,scatratimint_->DofRowMap()->IndexBase(),scatratimint_->DofRowMap()->Comm()));
    }
  }

  // safety check
  else
    dserror("Invalid type of global system matrix!");

  return;
} // SCATRA::MeshtyingStrategyS2I::BuildBlockMaps


/*-------------------------------------------------------------------------------*
 | build null spaces associated with blocks of global system matrix   fang 07/15 |
 *-------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::BuildBlockNullSpaces() const
{
  // loop over blocks of global system matrix
  for(int iblock=0; iblock<blockmaps_->NumMaps(); ++iblock)
  {
    // store number of current block as string, starting from 1
    std::ostringstream iblockstr;
    iblockstr << iblock+1;

    // equip smoother for current matrix block with empty parameter sublists to trigger null space computation
    Teuchos::ParameterList& blocksmootherparams = scatratimint_->Solver()->Params().sublist("Inverse"+iblockstr.str());
    blocksmootherparams.sublist("Aztec Parameters");
    blocksmootherparams.sublist("MueLu Parameters");

    // equip smoother for current matrix block with null space associated with all degrees of freedom on discretization
    scatratimint_->Discretization()->ComputeNullSpaceIfNecessary(blocksmootherparams);

    // reduce full null space to match degrees of freedom associated with current matrix block
    LINALG::Solver::FixMLNullspace("Block "+iblockstr.str(),*scatratimint_->Discretization()->DofRowMap(),*blockmaps_->Map(iblock),blocksmootherparams);
  }

  return;
} // SCATRA::MeshtyingStrategyS2I::BuildBlockNullSpaces


/*----------------------------------------------------------------------------*
 | initialize system matrix for scatra-scatra interface coupling   fang 10/14 |
 *----------------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseOperator> SCATRA::MeshtyingStrategyS2I::InitSystemMatrix() const
{
  Teuchos::RCP<LINALG::SparseOperator> systemmatrix(Teuchos::null);

  switch(matrixtype_)
  {
  case INPAR::S2I::matrix_sparse:
  {
    // initialize system matrix
    systemmatrix = Teuchos::rcp(new LINALG::SparseMatrix(*(scatratimint_->Discretization()->DofRowMap()),27,false,true));
    break;
  }

  case INPAR::S2I::matrix_block_geometry:
  {
    // initialize system matrix and associated strategy
    systemmatrix = Teuchos::rcp(new LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>(*interfacemaps_,*interfacemaps_));
    break;
  }

  case INPAR::S2I::matrix_block_condition:
  case INPAR::S2I::matrix_block_condition_dof:
  {
    // initialize system matrix and associated strategy
    systemmatrix = Teuchos::rcp(new LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>(*blockmaps_,*blockmaps_));

    break;
  }

  default:
  {
    dserror("Type of global system matrix for scatra-scatra interface coupling not recognized!");
    break;
  }
  }

  return systemmatrix;
} // SCATRA::MeshtyingStrategyS2I::InitSystemMatrix


/*------------------------------------------------------------------------------------*
 | solve linear system of equations for scatra-scatra interface coupling   fang 12/14 |
 *------------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::Solve(
    const Teuchos::RCP<LINALG::Solver>&            solver,         //! solver
    const Teuchos::RCP<LINALG::SparseOperator>&    systemmatrix,   //! system matrix
    const Teuchos::RCP<Epetra_Vector>&             increment,      //! increment vector
    const Teuchos::RCP<Epetra_Vector>&             residual,       //! residual vector
    const Teuchos::RCP<Epetra_Vector>&             phinp,          //! state vector at time n+1
    const int&                                     iteration,      //! number of current Newton-Raphson iteration
    const Teuchos::RCP<LINALG::KrylovProjector>&   projector       //! Krylov projector
    ) const
{
  switch(mortartype_)
  {
    case INPAR::S2I::mortar_none:
    case INPAR::S2I::mortar_standard:
    case INPAR::S2I::mortar_condensed_petrov:
    case INPAR::S2I::mortar_condensed_bubnov:
    {
      // equilibrate global system of equations if necessary
      EquilibrateSystem(systemmatrix,residual);

      // solve global system of equations
      solver->Solve(systemmatrix->EpetraOperator(),increment,residual,true,iteration==1,projector);

      // unequilibrate global increment vector if necessary
      UnequilibrateIncrement(increment);

      break;
    }

    case INPAR::S2I::mortar_saddlepoint_petrov:
    case INPAR::S2I::mortar_saddlepoint_bubnov:
    {
      // check scalar transport system matrix
      Teuchos::RCP<LINALG::SparseMatrix> sparsematrix = Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(systemmatrix);
      if(sparsematrix == Teuchos::null)
        dserror("System matrix is not a sparse matrix!");

      // assemble extended system matrix including rows and columns associated with Lagrange multipliers
      LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy> extendedsystemmatrix(*extendedmaps_,*extendedmaps_);
      extendedsystemmatrix.Assign(0,0,LINALG::View,*sparsematrix);
      if(lmside_ == INPAR::S2I::side_slave)
      {
        extendedsystemmatrix.Matrix(0,1).Add(*D_,true,1.,0.);
        extendedsystemmatrix.Matrix(0,1).Add(*M_,true,-1.,1.);
        extendedsystemmatrix.Matrix(1,0).Add(*MORTAR::MatrixRowTransformGIDs(islavematrix_,extendedmaps_->Map(1)),false,1.,0.);
      }
      else
      {
        extendedsystemmatrix.Matrix(0,1).Add(*M_,true,-1.,0.);
        extendedsystemmatrix.Matrix(0,1).Add(*D_,true,1.,1.);
        extendedsystemmatrix.Matrix(1,0).Add(*MORTAR::MatrixRowTransformGIDs(imastermatrix_,extendedmaps_->Map(1)),false,1.,0.);
      }
      extendedsystemmatrix.Matrix(1,1).Add(*E_,true,-1.,0.);
      extendedsystemmatrix.Complete();
      extendedsystemmatrix.Matrix(0,1).ApplyDirichlet(*scatratimint_->DirichMaps()->CondMap(),false);

      Teuchos::RCP<Epetra_Vector> extendedresidual = LINALG::CreateVector(*extendedmaps_->FullMap());
      extendedmaps_->InsertVector(scatratimint_->Residual(),0,extendedresidual);
      extendedmaps_->InsertVector(lmresidual_,1,extendedresidual);

      Teuchos::RCP<Epetra_Vector> extendedincrement = LINALG::CreateVector(*extendedmaps_->FullMap());
      extendedmaps_->InsertVector(scatratimint_->Increment(),0,extendedincrement);
      extendedmaps_->InsertVector(lmincrement_,1,extendedincrement);

      // solve extended system of equations
      solver->Solve(extendedsystemmatrix.EpetraOperator(),extendedincrement,extendedresidual,true,iteration==1,projector);

      // store solution
      extendedmaps_->ExtractVector(extendedincrement,0,scatratimint_->Increment());
      extendedmaps_->ExtractVector(extendedincrement,1,lmincrement_);

      // update Lagrange multipliers
      lm_->Update(1.,*lmincrement_,1.);

      break;
    }

    default:
    {
      dserror("Type of mortar meshtying for scatra-scatra interface coupling not recognized!");
      break;
    }
  }

  return;
} // SCATRA::MeshtyingStrategyS2I::Solve


/*----------------------------------------------------------------------*
 | equilibrate global system of equations if necessary       fang 05/15 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::EquilibrateSystem(
    const Teuchos::RCP<LINALG::SparseOperator>&   systemmatrix,   //! system matrix
    const Teuchos::RCP<Epetra_Vector>&            residual        //! residual vector
    ) const
{
  if(rowequilibration_ or colequilibration_)
  {
    // perform equilibration depending on type of global system matrix
    switch(matrixtype_)
    {
    case INPAR::S2I::matrix_sparse:
    {
      // check matrix
      Teuchos::RCP<LINALG::SparseMatrix> sparsematrix = Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(systemmatrix);
      if(sparsematrix == Teuchos::null)
        dserror("System matrix is not a sparse matrix!");

      // perform row equilibration
      if(rowequilibration_)
      {
        // compute inverse row sums of global system matrix
        ComputeInvRowSums(*sparsematrix,invrowsums_);

        // perform row equilibration of global system matrix
        EquilibrateMatrixRows(*sparsematrix,invrowsums_);
      }

      // perform column equilibration
      if(colequilibration_)
      {
        // compute inverse column sums of global system matrix
        ComputeInvColSums(*sparsematrix,invcolsums_);

        // perform column equilibration of global system matrix
        EquilibrateMatrixColumns(*sparsematrix,invcolsums_);
      }

      break;
    }

    case INPAR::S2I::matrix_block_condition:
    case INPAR::S2I::matrix_block_condition_dof:
    case INPAR::S2I::matrix_block_geometry:
    {
      // check matrix
      Teuchos::RCP<LINALG::BlockSparseMatrixBase> blocksparsematrix = Teuchos::rcp_dynamic_cast<LINALG::BlockSparseMatrixBase>(systemmatrix);
      if(blocksparsematrix == Teuchos::null)
        dserror("System matrix is not a block sparse matrix!");

      // perform row equilibration
      if(rowequilibration_)
      {
        for(int i=0; i<blocksparsematrix->Rows(); ++i)
        {
          // compute inverse row sums of current main diagonal matrix block
          Teuchos::RCP<Epetra_Vector> invrowsums(Teuchos::rcp(new Epetra_Vector(blocksparsematrix->Matrix(i,i).RangeMap())));
          ComputeInvRowSums(blocksparsematrix->Matrix(i,i),invrowsums);

          // perform row equilibration of matrix blocks in current row block of global system matrix
          for(int j=0; j<blocksparsematrix->Cols(); ++j)
            EquilibrateMatrixRows(blocksparsematrix->Matrix(i,j),invrowsums);

          // insert inverse row sums of current main diagonal matrix block into global vector
          blockmaps_->InsertVector(invrowsums,i,invrowsums_);
        }
      }

      // perform column equilibration
      if(colequilibration_)
      {
        for(int j=0; j<blocksparsematrix->Cols(); ++j)
        {
          // compute inverse column sums of current main diagonal matrix block
          Teuchos::RCP<Epetra_Vector> invcolsums(Teuchos::rcp(new Epetra_Vector(blocksparsematrix->Matrix(j,j).DomainMap())));
          ComputeInvColSums(blocksparsematrix->Matrix(j,j),invcolsums);

          // perform column equilibration of matrix blocks in current column block of global system matrix
          for(int i=0; i<blocksparsematrix->Rows(); ++i)
            EquilibrateMatrixColumns(blocksparsematrix->Matrix(i,j),invcolsums);

          // insert inverse column sums of current main diagonal matrix block into global vector
          blockmaps_->InsertVector(invcolsums,j,invcolsums_);
        }
      }

      break;
    }

    default:
    {
      dserror("Equilibration of global system of equations for scatra-scatra interface coupling is not implemented for chosen type of global system matrix!");
      break;
    }
    }

    // perform equilibration of global residual vector
    if(rowequilibration_)
      if(residual->Multiply(1.,*invrowsums_,*residual,0.))
        dserror("Equilibration of global residual vector failed!");
  }

  return;
} // SCATRA::MeshtyingStrategyS2I::EquilibrateSystem


/*----------------------------------------------------------------------------*
 | compute inverse sums of absolute values of matrix row entries   fang 06/15 |
 *----------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::ComputeInvRowSums(
    const LINALG::SparseMatrix&          matrix,      //! matrix
    const Teuchos::RCP<Epetra_Vector>&   invrowsums   //! inverse sums of absolute values of row entries in matrix
    ) const
{
  // compute inverse row sums of matrix
  if(matrix.EpetraMatrix()->InvRowSums(*invrowsums))
    dserror("Inverse row sums of matrix could not be successfully computed!");

  // take square root of inverse row sums if matrix is scaled from left and right
  if(colequilibration_)
    for(int i=0; i<invrowsums->MyLength(); ++i)
      (*invrowsums)[i] = sqrt((*invrowsums)[i]);

  return;
} // SCATRA::MeshtyingStrategyS2I::ComputeInvRowSums


/*-------------------------------------------------------------------------------*
 | compute inverse sums of absolute values of matrix column entries   fang 06/15 |
 *-------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::ComputeInvColSums(
    const LINALG::SparseMatrix&          matrix,      //! matrix
    const Teuchos::RCP<Epetra_Vector>&   invcolsums   //! inverse sums of absolute values of column entries in matrix
    ) const
{
  // compute inverse column sums of matrix
  if(matrix.EpetraMatrix()->InvColSums(*invcolsums))
    dserror("Inverse column sums of matrix could not be successfully computed!");

  // take square root of inverse column sums if matrix is scaled from left and right
  if(rowequilibration_)
    for(int i=0; i<invcolsums->MyLength(); ++i)
      (*invcolsums)[i] = sqrt((*invcolsums)[i]);

  return;
} // SCATRA::MeshtyingStrategyS2I::ComputeInvColSums


/*----------------------------------------------------------------------*
 | equilibrate matrix rows                                   fang 06/15 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::EquilibrateMatrixRows(
    LINALG::SparseMatrix&                matrix,      //! matrix
    const Teuchos::RCP<Epetra_Vector>&   invrowsums   //! sums of absolute values of row entries in matrix
    ) const
{
  if(matrix.LeftScale(*invrowsums))
    dserror("Row equilibration of matrix failed!");

  return;
} // SCATRA::MeshtyingStrategyS2I::EquilibrateMatrixRows


/*----------------------------------------------------------------------*
 | equilibrate matrix columns                                fang 06/15 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::EquilibrateMatrixColumns(
    LINALG::SparseMatrix&                matrix,      //! matrix
    const Teuchos::RCP<Epetra_Vector>&   invcolsums   //! sums of absolute values of column entries in matrix
    ) const
{
  if(matrix.RightScale(*invcolsums))
    dserror("Column equilibration of matrix failed!");

  return;
} // SCATRA::MeshtyingStrategyS2I::EquilibrateMatrixColumns


/*----------------------------------------------------------------------*
 | unequilibrate global increment vector if necessary        fang 05/15 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyS2I::UnequilibrateIncrement(
    const Teuchos::RCP<Epetra_Vector>&   increment   //! increment vector
    ) const
{
  // unequilibrate global increment vector if necessary
  if(colequilibration_)
  {
    if(increment->Multiply(1.,*invcolsums_,*increment,0.))
      dserror("Unequilibration of global increment vector failed!");
  }

  return;
} // SCATRA::MeshtyingStrategyS2I::UnequilibrateIncrement


/*--------------------------------------------------------------------------------------*
 | protected constructor for singletons                                      fang 01/16 |
 *--------------------------------------------------------------------------------------*/
SCATRA::MortarCellInterface::MortarCellInterface(
    const INPAR::S2I::MortarType&       mortartype,            //!< flag for meshtying method
    const INPAR::S2I::InterfaceSides&   lmside,                //!< flag for interface side underlying Lagrange multiplier definition
    const int&                          numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
    const int&                          numdofpernode_master   //!< number of master-side degrees of freedom per node
    ) :
    lmside_(lmside),
    mortartype_(mortartype),
    numdofpernode_slave_(numdofpernode_slave),
    numdofpernode_master_(numdofpernode_master)
{
  return;
}


/*----------------------------------------------------------------------*
 | singleton access method                                   fang 01/16 |
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
SCATRA::MortarCellCalc<distypeS,distypeM>* SCATRA::MortarCellCalc<distypeS,distypeM>::Instance(
    const INPAR::S2I::MortarType&       mortartype,             //!< flag for meshtying method
    const INPAR::S2I::InterfaceSides&   lmside,                 //!< flag for interface side underlying Lagrange multiplier definition
    const int&                          numdofpernode_slave,    //!< number of slave-side degrees of freedom per node
    const int&                          numdofpernode_master,   //!< number of master-side degrees of freedom per node
    bool                                create                  //!< creation flag
    )
{
  static MortarCellCalc<distypeS,distypeM>* instance;

  if(create)
  {
    if(instance == NULL)
      instance = new MortarCellCalc<distypeS,distypeM>(mortartype,lmside,numdofpernode_slave,numdofpernode_master);
  }

  else if(instance != NULL)
  {
    delete instance;
    instance = NULL;
  }

  return instance;
}


/*----------------------------------------------------------------------*
 | singleton destruction                                     fang 01/16 |
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
void SCATRA::MortarCellCalc<distypeS,distypeM>::Done()
{
  // delete singleton
  Instance(INPAR::S2I::mortar_undefined,INPAR::S2I::side_undefined,0,0,false);

  return;
}


/*--------------------------------------------------------------------------------------------------------------------*
 | evaluate single mortar integration cell of particular slave-side and master-side discretization types   fang 01/16 |
 *--------------------------------------------------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
void SCATRA::MortarCellCalc<distypeS,distypeM>::Evaluate(
    Teuchos::RCP<LINALG::SparseMatrix>   imatrix1,          //!< interface matrix 1
    Teuchos::RCP<LINALG::SparseMatrix>   imatrix2,          //!< interface matrix 2
    Teuchos::RCP<LINALG::SparseMatrix>   imatrix3,          //!< interface matrix 3
    Teuchos::RCP<Epetra_Vector>          islaveresidual,    //!< slave-side residual vector
    Teuchos::RCP<Epetra_FEVector>        imasterresidual,   //!< master-side residual vector
    const DRT::Discretization&           idiscret,          //!< interface discretization
    const Teuchos::ParameterList&        params,            //!< parameter list
    MORTAR::MortarElement&               slaveelement,      //!< slave-side mortar element
    MORTAR::MortarElement&               masterelement,     //!< master-side mortar element
    MORTAR::IntCell&                     cell,              //!< mortar integration cell
    DRT::Element::LocationArray&         la_slave,          //!< slave-side location array
    DRT::Element::LocationArray&         la_master          //!< master-side location array
    )
{
  // extract and evaluate action
  switch(DRT::INPUT::get<INPAR::S2I::EvaluationActions>(params,"action"))
  {
    case INPAR::S2I::evaluate_mortar_matrices:
    {
      // evaluate mortar matrices
      EvaluateMortarMatrices(imatrix1,imatrix2,imatrix3,idiscret,slaveelement,masterelement,cell,la_slave,la_master);

      break;
    }

    case INPAR::S2I::evaluate_condition:
    {
      // extract condition from parameter list
      DRT::Condition* condition = params.get<DRT::Condition*>("condition");
      if(condition == NULL)
        dserror("Cannot access scatra-scatra interface coupling condition!");

      // extract nodal state variables associated with slave and master elements
      std::vector<LINALG::Matrix<nen_slave_,1> > ephinp_slave(numdofpernode_slave_,LINALG::Matrix<nen_slave_,1>(true));
      std::vector<LINALG::Matrix<nen_master_,1> > ephinp_master(numdofpernode_master_,LINALG::Matrix<nen_master_,1>(true));
      ExtractNodeValues(ephinp_slave,ephinp_master,idiscret,la_slave,la_master);

      // evaluate and assemble interface linearizations and residuals
      EvaluateCondition(
          imatrix1,
          imatrix2,
          islaveresidual,
          imasterresidual,
          idiscret,
          *condition,
          slaveelement,
          masterelement,
          cell,
          ephinp_slave,
          ephinp_master,
          la_slave,
          la_master
          );

      break;
    }

    default:
    {
      dserror("Unknown action for mortar cell evaluation!");
      break;
    }
  }

  return;
}


/*--------------------------------------------------------------------------------------*
 | protected constructor for singletons                                      fang 01/16 |
 *--------------------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
SCATRA::MortarCellCalc<distypeS,distypeM>::MortarCellCalc(
    const INPAR::S2I::MortarType&       mortartype,            //!< flag for meshtying method
    const INPAR::S2I::InterfaceSides&   lmside,                //!< flag for interface side underlying Lagrange multiplier definition
    const int&                          numdofpernode_slave,   //!< number of slave-side degrees of freedom per node
    const int&                          numdofpernode_master   //!< number of master-side degrees of freedom per node
    ) :
    MortarCellInterface(mortartype,lmside,numdofpernode_slave,numdofpernode_master),
    funct_slave_(true),
    funct_master_(true),
    shape_lm_slave_(true),
    shape_lm_master_(true),
    test_lm_slave_(true),
    test_lm_master_(true)
{
  // safety check
  if(nsd_slave_ != 2 or nsd_master_ != 2)
    dserror("Scatra-scatra interface coupling with non-matching interface discretization currently only implemented for two-dimensional interface manifolds!");

  return;
}


/*--------------------------------------------------------------------------------------*
 | extract nodal state variables associated with slave and master elements   fang 01/16 |
 *--------------------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
void SCATRA::MortarCellCalc<distypeS,distypeM>::ExtractNodeValues(
    std::vector<LINALG::Matrix<nen_slave_,1> >&    ephinp_slave,    //!< state variables at slave-side nodes
    std::vector<LINALG::Matrix<nen_master_,1> >&   ephinp_master,   //!< state variables at master-side nodes
    const DRT::Discretization&                    idiscret,        //!< interface discretization
    DRT::Element::LocationArray&                  la_slave,        //!< slave-side location array
    DRT::Element::LocationArray&                  la_master        //!< master-side location array
    ) const
{
  // extract interface state vector from interface discretization
  Teuchos::RCP<const Epetra_Vector> iphinp = idiscret.GetState("iphinp");
  if(iphinp == Teuchos::null)
    dserror("Cannot extract state vector \"iphinp\" from interface discretization!");

  // extract nodal state variables associated with slave and master elements
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nen_slave_,1> >(*iphinp,ephinp_slave,la_slave[0].lm_);
  DRT::UTILS::ExtractMyValues<LINALG::Matrix<nen_master_,1> >(*iphinp,ephinp_master,la_master[0].lm_);

  return;
}


/*------------------------------------------------------------------------------------------*
 | evaluate shape functions and domain integration factor at integration point   fang 01/16 |
 *------------------------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
double SCATRA::MortarCellCalc<distypeS,distypeM>::EvalShapeFuncAndDomIntFacAtIntPoint(
    MORTAR::MortarElement&                               slaveelement,    //!< slave-side mortar element
    MORTAR::MortarElement&                               masterelement,   //!< master-side mortar element
    MORTAR::IntCell&                                     cell,            //!< mortar integration cell
    const DRT::UTILS::IntPointsAndWeights<nsd_slave_>&   intpoints,       //!< quadrature rule
    const int                                            iquad            //!< ID of integration point
    )
{
  // reference coordinates of integration point
  double coordinates_ref[nsd_slave_] = {};
  for(int idim=0; idim<nsd_slave_; ++idim)
    coordinates_ref[idim] = intpoints.IP().qxg[iquad][idim];

  // global coordinates of integration point
  double coordinates_global[nsd_slave_+1] = {};
  cell.LocalToGlobal(coordinates_ref,coordinates_global,0);

  // project integration point onto slave and master elements
  double coordinates_slave[nsd_slave_] = {};
  double coordinates_master[nsd_master_] = {};
  double dummy(0.);
  MORTAR::MortarProjector::Impl(slaveelement)->ProjectGaussPointAuxn3D(coordinates_global,cell.Auxn(),slaveelement,coordinates_slave,dummy);
  MORTAR::MortarProjector::Impl(masterelement)->ProjectGaussPointAuxn3D(coordinates_global,cell.Auxn(),masterelement,coordinates_master,dummy);

  // evaluate shape functions at current integration point on slave and master elements
  VOLMORTAR::UTILS::shape_function<distypeS>(funct_slave_,coordinates_slave);
  VOLMORTAR::UTILS::shape_function<distypeM>(funct_master_,coordinates_master);
  switch(mortartype_)
  {
    case INPAR::S2I::mortar_standard:
    {
      // there actually aren't any Lagrange multipliers, but we still need to set pseudo Lagrange multiplier test functions
      // equal to the standard shape and test functions for correct evaluation of the scatra-scatra interface coupling conditions
      test_lm_slave_ = funct_slave_;
      test_lm_master_ = funct_master_;

      break;
    }

    case INPAR::S2I::mortar_saddlepoint_petrov:
    case INPAR::S2I::mortar_condensed_petrov:
    {
      // dual Lagrange multiplier shape functions combined with standard Lagrange multiplier test functions
      if(lmside_ == INPAR::S2I::side_slave)
      {
        VOLMORTAR::UTILS::dual_shape_function<distypeS>(shape_lm_slave_,coordinates_slave,slaveelement);
        test_lm_slave_ = funct_slave_;
      }
      else
      {
        VOLMORTAR::UTILS::dual_shape_function<distypeM>(shape_lm_master_,coordinates_master,masterelement);
        test_lm_master_ = funct_master_;
      }

      break;
    }

    case INPAR::S2I::mortar_saddlepoint_bubnov:
    case INPAR::S2I::mortar_condensed_bubnov:
    {
      // dual Lagrange multiplier shape functions combined with dual Lagrange multiplier test functions
      if(lmside_ == INPAR::S2I::side_slave)
      {
        VOLMORTAR::UTILS::dual_shape_function<distypeS>(shape_lm_slave_,coordinates_slave,slaveelement);
        test_lm_slave_ = shape_lm_slave_;
      }
      else
      {
        VOLMORTAR::UTILS::dual_shape_function<distypeM>(shape_lm_master_,coordinates_master,masterelement);
        test_lm_master_ = shape_lm_master_;
      }

      break;
    }

    default:
    {
      dserror("Not yet implemented!");
      break;
    }
  }

  // integration weight
  const double weight = intpoints.IP().qwgt[iquad];

  // Jacobian determinant
  const double jacobian = cell.Jacobian(coordinates_ref);

  // domain integration factor
  return jacobian*weight;
}


/*---------------------------------------------------------------------------*
 | evaluate mortar matrices                                       fang 01/16 |
 *---------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
void SCATRA::MortarCellCalc<distypeS,distypeM>::EvaluateMortarMatrices(
    Teuchos::RCP<LINALG::SparseMatrix>   D,               //!< mortar matrix D
    Teuchos::RCP<LINALG::SparseMatrix>   M,               //!< mortar matrix M
    Teuchos::RCP<LINALG::SparseMatrix>   E,               //!< mortar matrix E
    const DRT::Discretization&           idiscret,        //!< interface discretization
    MORTAR::MortarElement&               slaveelement,    //!< slave-side mortar element
    MORTAR::MortarElement&               masterelement,   //!< master-side mortar element
    MORTAR::IntCell&                     cell,            //!< mortar integration cell
    DRT::Element::LocationArray&         la_slave,        //!< slave-side location array
    DRT::Element::LocationArray&         la_master        //!< master-side location array
    )
{
  // safety check
  if(numdofpernode_slave_ != numdofpernode_master_)
    dserror("Must have same number of degrees of freedom per node on slave and master sides!");

  // determine quadrature rule
  const DRT::UTILS::IntPointsAndWeights<2> intpoints(DRT::UTILS::intrule_tri_7point);

  // loop over all integration points
  for(int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    // evaluate shape functions and domain integration factor at current integration point
    const double fac = EvalShapeFuncAndDomIntFacAtIntPoint(slaveelement,masterelement,cell,intpoints,iquad);

    if(lmside_ == INPAR::S2I::side_slave)
    {
      // loop over all degrees of freedom per node
      for(int k=0; k<numdofpernode_slave_; ++k)
      {
        for (int vi=0; vi<nen_slave_; ++vi)
        {
          const int row_index = vi*numdofpernode_slave_+k;
          if(la_slave[0].lmowner_[row_index] == idiscret.Comm().MyPID())
          {
            const int row_slave = la_slave[0].lm_[row_index];

            switch(mortartype_)
            {
              case INPAR::S2I::mortar_saddlepoint_petrov:
              case INPAR::S2I::mortar_saddlepoint_bubnov:
              case INPAR::S2I::mortar_condensed_petrov:
              case INPAR::S2I::mortar_condensed_bubnov:
              {
                D->Assemble(shape_lm_slave_(vi)*fac,row_slave,row_slave);

                if(mortartype_ == INPAR::S2I::mortar_saddlepoint_bubnov or mortartype_ == INPAR::S2I::mortar_condensed_bubnov)
                  for(int ui=0; ui<nen_slave_; ++ui)
                  {
                    const int col_slave = la_slave[0].lm_[ui*numdofpernode_slave_+k];

                    E->Assemble(shape_lm_slave_(vi)*test_lm_slave_(ui)*fac,row_slave,col_slave);
                  }

                break;
              }

              default:
              {
                for (int ui=0; ui<nen_slave_; ++ui)
                {
                  const int col_slave = la_slave[0].lm_[ui*numdofpernode_slave_+k];

                  D->Assemble(shape_lm_slave_(vi)*funct_slave_(ui)*fac,row_slave,col_slave);
                }

                break;
              }
            }

            for(int ui=0; ui<nen_master_; ++ui)
            {
              const int col_master = la_master[0].lm_[ui*numdofpernode_master_+k];

              M->Assemble(shape_lm_slave_(vi)*funct_master_(ui)*fac,row_slave,col_master);
            }
          }
        }
      }
    }

    else
    {
      if(slaveelement.Owner() == idiscret.Comm().MyPID())
      {
        // loop over all degrees of freedom per node
        for(int k=0; k<numdofpernode_master_; ++k)
        {
          for (int vi=0; vi<nen_master_; ++vi)
          {
            const int row_master = la_master[0].lm_[vi*numdofpernode_master_+k];

            switch(mortartype_)
            {
              case INPAR::S2I::mortar_saddlepoint_petrov:
              case INPAR::S2I::mortar_saddlepoint_bubnov:
              case INPAR::S2I::mortar_condensed_petrov:
              case INPAR::S2I::mortar_condensed_bubnov:
              {
                D->FEAssemble(shape_lm_master_(vi)*fac,row_master,row_master);

                if(mortartype_ == INPAR::S2I::mortar_saddlepoint_bubnov or mortartype_ == INPAR::S2I::mortar_condensed_bubnov)
                  for(int ui=0; ui<nen_master_; ++ui)
                  {
                    const int col_master = la_master[0].lm_[ui*numdofpernode_master_+k];

                    E->FEAssemble(shape_lm_master_(vi)*test_lm_master_(ui)*fac,row_master,col_master);
                  }

                break;
              }

              default:
              {
                for (int ui=0; ui<nen_master_; ++ui)
                {
                  const int col_master = la_master[0].lm_[ui*numdofpernode_master_+k];

                  D->FEAssemble(shape_lm_master_(vi)*funct_master_(ui)*fac,row_master,col_master);
                }

                break;
              }
            }

            for(int ui=0; ui<nen_slave_; ++ui)
            {
              const int col_slave = la_slave[0].lm_[ui*numdofpernode_slave_+k];

              M->FEAssemble(shape_lm_master_(vi)*funct_slave_(ui)*fac,row_master,col_slave);
            }
          }
        }
      }
    }
  }

  return;
}


/*---------------------------------------------------------------------------*
 | evaluate and assemble interface linearizations and residuals   fang 01/16 |
 *---------------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distypeS,DRT::Element::DiscretizationType distypeM>
void SCATRA::MortarCellCalc<distypeS,distypeM>::EvaluateCondition(
    Teuchos::RCP<LINALG::SparseMatrix>             islavematrix,      //!< linearizations of slave-side residuals
    Teuchos::RCP<LINALG::SparseMatrix>             imastermatrix,     //!< linearizations of master-side residuals
    Teuchos::RCP<Epetra_Vector>                    islaveresidual,    //!< slave-side residual vector
    Teuchos::RCP<Epetra_FEVector>                  imasterresidual,   //!< master-side residual vector
    const DRT::Discretization&                     idiscret,          //!< interface discretization
    DRT::Condition&                                condition,         //!< scatra-scatra interface coupling condition
    MORTAR::MortarElement&                         slaveelement,      //!< slave-side mortar element
    MORTAR::MortarElement&                         masterelement,     //!< master-side mortar element
    MORTAR::IntCell&                               cell,              //!< mortar integration cell
    std::vector<LINALG::Matrix<nen_slave_,1> >&    ephinp_slave,      //!< state variables at slave-side nodes
    std::vector<LINALG::Matrix<nen_master_,1> >&   ephinp_master,     //!< state variables at master-side nodes
    DRT::Element::LocationArray&                   la_slave,          //!< slave-side location array
    DRT::Element::LocationArray&                   la_master          //!< master-side location array
    )
{
  // safety check
  if(numdofpernode_slave_ != 1 or numdofpernode_master_ != 1)
    dserror("Invalid number of degrees of freedom per node!");

  // access input parameters associated with current condition
  const int kineticmodel = condition.GetInt("kinetic model");
  if(kineticmodel != INPAR::S2I::kinetics_constperm)
    dserror("Invalid kinetic model for scatra-scatra interface coupling!");
  const std::vector<double>* permeabilities = condition.GetMutable<std::vector<double> >("permeabilities");
  if(permeabilities == NULL)
    dserror("Cannot access vector of permeabilities for scatra-scatra interface coupling!");
  if(permeabilities->size() != 1)
    dserror("Number of permeabilities does not match number of scalars!");

  // determine quadrature rule
  const DRT::UTILS::IntPointsAndWeights<2> intpoints(DRT::UTILS::intrule_tri_7point);

  // loop over all integration points
  for(int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    // evaluate shape functions and domain integration factor at current integration point
    const double fac = EvalShapeFuncAndDomIntFacAtIntPoint(slaveelement,masterelement,cell,intpoints,iquad);

    // overall integration factors
    const double timefacfac = DRT::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->TimeFac()*fac;
    const double timefacrhsfac = DRT::ELEMENTS::ScaTraEleParameterTimInt::Instance("scatra")->TimeFacRhs()*fac;
    if(timefacfac < 0. or timefacrhsfac < 0.)
      dserror("Integration factor is negative!");

    // evaluate state variables at current integration point
    const double eslavephiint = funct_slave_.Dot(ephinp_slave[0]);
    const double emasterphiint = funct_master_.Dot(ephinp_master[0]);

    // core residual
    const double N = timefacrhsfac*(*permeabilities)[0]*(eslavephiint-emasterphiint);

    // core linearizations
    const double dN_dc_slave = timefacfac*(*permeabilities)[0];
    const double dN_dc_master = -dN_dc_slave;

    if(islavematrix != Teuchos::null and islaveresidual != Teuchos::null)
    {
      for (int vi=0; vi<nen_slave_; ++vi)
      {
        if(la_slave[0].lmowner_[vi] == idiscret.Comm().MyPID())
        {
          const int row_slave = la_slave[0].lm_[vi];

          for (int ui=0; ui<nen_slave_; ++ui)
          {
            const int col_slave = la_slave[0].lm_[ui];

            islavematrix->Assemble(test_lm_slave_(vi)*dN_dc_slave*funct_slave_(ui),row_slave,col_slave);
          }

          for(int ui=0; ui<nen_master_; ++ui)
          {
            const int col_master = la_master[0].lm_[ui];

            islavematrix->Assemble(test_lm_slave_(vi)*dN_dc_master*funct_master_(ui),row_slave,col_master);
          }

          if(islaveresidual->SumIntoGlobalValue(row_slave,0,-test_lm_slave_(vi)*N))
            dserror("Assembly into slave-side residual vector not successful!");
        }
      }
    }
    else if(islavematrix != Teuchos::null or islaveresidual != Teuchos::null)
      dserror("Must provide both slave-side matrix and slave-side vector or none of them!");

    if(imastermatrix != Teuchos::null and imasterresidual != Teuchos::null)
    {
      if(slaveelement.Owner() == idiscret.Comm().MyPID())
      {
        for(int vi=0; vi<nen_master_; ++vi)
        {
          const int row_master = la_master[0].lm_[vi];

          for (int ui=0; ui<nen_slave_; ++ui)
          {
            const int col_slave = la_slave[0].lm_[ui];

            imastermatrix->FEAssemble(-test_lm_master_(vi)*dN_dc_slave*funct_slave_(ui),row_master,col_slave);
          }

          for(int ui=0; ui<nen_master_; ++ui)
          {
            const int col_master = la_master[0].lm_[ui];

            imastermatrix->FEAssemble(-test_lm_master_(vi)*dN_dc_master*funct_master_(ui),row_master,col_master);
          }

          const double residual_master = test_lm_master_(vi)*N;
          if(imasterresidual->SumIntoGlobalValues(1,&row_master,&residual_master))
            dserror("Assembly into master-side residual vector not successful!");
        }
      }
    }
    else if(imastermatrix != Teuchos::null or imasterresidual != Teuchos::null)
      dserror("Must provide both master-side matrix and master-side vector or none of them!");
  }

  return;
}


// forward declarations
template class SCATRA::MortarCellCalc<DRT::Element::tri3,DRT::Element::tri3>;
template class SCATRA::MortarCellCalc<DRT::Element::tri3,DRT::Element::quad4>;
