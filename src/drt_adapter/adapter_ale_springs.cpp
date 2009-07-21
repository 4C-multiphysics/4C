/*----------------------------------------------------------------------*/
/*!
\file adapter_ale_springs.cpp

\brief

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "adapter_ale_springs.H"


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ADAPTER::AleSprings::AleSprings(RCP<DRT::Discretization> actdis,
                                Teuchos::RCP<LINALG::Solver> solver,
                                Teuchos::RCP<ParameterList> params,
                                Teuchos::RCP<IO::DiscretizationWriter> output,
                                bool dirichletcond)
  : discret_(actdis),
    solver_ (solver),
    params_ (params),
    output_ (output),
    step_(0),
    time_(0.0),
    sysmat_(null),
    uprestart_(params->get("write restart every", -1))
{
  numstep_ = params_->get<int>("numstep");
  maxtime_ = params_->get<double>("maxtime");
  dt_      = params_->get<double>("dt");

  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  dispn_          = LINALG::CreateVector(*dofrowmap,true);
  dispnp_         = LINALG::CreateVector(*dofrowmap,true);
  residual_       = LINALG::CreateVector(*dofrowmap,true);
  incr_           = LINALG::CreateVector(*dofrowmap,true);

  DRT::UTILS::SetupNDimExtractor(*actdis,"FSICoupling",interface_);
  DRT::UTILS::SetupNDimExtractor(*actdis,"FREESURFCoupling",freesurface_);

  // set fixed nodes (conditions != 0 are not supported right now)
  ParameterList eleparams;
  eleparams.set("total time", time_);
  eleparams.set("delta time", dt_);
  dbcmaps_ = Teuchos::rcp(new LINALG::MapExtractor());
  discret_->EvaluateDirichlet(eleparams,dispnp_,null,null,null,dbcmaps_);

  if (dirichletcond)
  {
    // for partitioned FSI the interface becomes a Dirichlet boundary
    std::vector<Teuchos::RCP<const Epetra_Map> > condmaps;
    condmaps.push_back(interface_.CondMap());
    condmaps.push_back(dbcmaps_->CondMap());
    Teuchos::RCP<Epetra_Map> condmerged = LINALG::MultiMapExtractor::MergeMaps(condmaps);
    *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), condmerged);
  }

  if (dirichletcond and freesurface_.Relevant())
  {
    // for partitioned solves the free surface becomes a Dirichlet boundary
    std::vector<Teuchos::RCP<const Epetra_Map> > condmaps;
    condmaps.push_back(freesurface_.CondMap());
    condmaps.push_back(dbcmaps_->CondMap());
    Teuchos::RCP<Epetra_Map> condmerged = LINALG::MultiMapExtractor::MergeMaps(condmaps);
    *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), condmerged);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::BuildSystemMatrix(bool full)
{
  if (full)
  {
    const Epetra_Map* dofrowmap = discret_->DofRowMap();
    sysmat_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap,81,false,true));
  }
  else
  {
    if (freesurface_.Relevant())
    {
      sysmat_ = Teuchos::rcp(new LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>(freesurface_,freesurface_,81,false,true));
    }
    else
    {
      sysmat_ = Teuchos::rcp(new LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>(interface_,interface_,81,false,true));
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::PrepareTimeStep()
{
  step_ += 1;
  time_ += dt_;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::Evaluate(Teuchos::RCP<const Epetra_Vector> ddisp)
{
  // We save the current solution here. This will not change the
  // result of our element call, but the next time somebody asks us we
  // know the displacements.
  //
  // Note: What we get here is the sum of all increments in this time
  // step, not just the latest increment. Be careful.

  if (ddisp!=Teuchos::null)
  {
    // Dirichlet boundaries != 0 are not supported.

    incr_->Update(1.0,*ddisp,1.0,*dispn_,0.0);
  }

  EvaluateElements();
  LINALG::ApplyDirichlettoSystem(sysmat_,incr_,residual_,dispnp_,*(dbcmaps_->CondMap()));
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::Solve()
{
  EvaluateElements();

  // set fixed nodes
  ParameterList eleparams;
  eleparams.set("total time", time_);
  eleparams.set("delta time", dt_);
  // the DOFs with Dirchlet BCs are not rebuild, they are assumed to be correct
  discret_->EvaluateDirichlet(eleparams,dispnp_,null,null,Teuchos::null,Teuchos::null);

  incr_->Update(1.0,*dispnp_,-1.0,*dispn_,0.0);

  LINALG::ApplyDirichlettoSystem(sysmat_,incr_,residual_,incr_,*(dbcmaps_->CondMap()));

  solver_->Solve(sysmat_->EpetraOperator(),incr_,residual_,true);

  incr_->Update(1.0,*dispn_,1.0);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::Update()
{
  dispn_-> Update(1.0,*incr_,0.0);
  dispnp_->Update(1.0,*incr_,0.0);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::Output()
{
  // We do not need any output -- the fluid writes its
  // displacements itself. But we need restart.

  output_->NewStep    (step_,time_);
  output_->WriteVector("dispnp", dispnp_);

  if (uprestart_ != 0 and step_ % uprestart_ == 0)
  {
    // add restart data
    output_->WriteVector("dispn", dispn_);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::Integrate()
{
  while (step_ < numstep_-1 and time_ <= maxtime_)
  {
    PrepareTimeStep();
    Solve();
    Update();
    Output();
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::EvaluateElements()
{

  //find out if we have free surface nodes with heightfunction coupling
  std::vector<DRT::Condition*> hfconds;
  if (freesurface_.Relevant())
  {
    // select free surface nodes
    std::string condname = "FREESURFCoupling";

    std::vector<DRT::Condition*> conds;
    discret_->GetCondition(condname, conds);

    // select only heightfunction conditions here
    for (unsigned i=0; i<conds.size(); ++i)
    {
      if (*conds[i]->Get<std::string>("coupling")=="heightfunction")
        hfconds.push_back(conds[i]);
    }
    conds.clear();
  }

  // Are there free surface conditions with heightfunction coupling?
  if (freesurface_.Relevant() and hfconds.size()>0)
  {
    // ====================================================================================
    // ====================================================================================
    // Heightfunction for the free surface. We manipulate the sum of all
    // Newton-increments and the ALE-systemmatrix here.
    //
    //
    // NOTE: This is still uncomplete or even incorrect. The correct
    // implementation is sketched in Lorenz G�rcke's bachelor thesis.
    //
    // The height function like in Genkinger (21) must not be implemented by a
    // straightforward manipulation of nodal values (as is done here). Due to
    // discretization this is not mass-consistent!  Furthermore, partitioned
    // algorithms demand different treatment than monolithic algorithms. But
    // both call this EvaluateElements.
    //
    // ====================================================================================



    // ============= get ALE node normals =============================================

    // these normals are calculated acc. to the volume integral in Wall
    // (7.13). Since there is no Evaluate call to Ale3Surface elements so far,
    // this is the only possible way to get node normals. The important
    // drawback is that the volume integral considers each surface of the Ale3
    // element to be an free surface. Consequently also the walls of our
    // channel and the inflow and outflow cross sections are considered and
    // the node normals of these nodes point outwards.
    //
    // The heightfunction is strongly related to the node normals - this is
    // why we need to manipulate those normals/ heightfunction (only on
    // nodes in the inflow and outflow cross section theoretically - at these
    // nodes we have a fluid velocity in the direction of the erroneous
    // heightfunction component. At the rim this does not matter since the
    // velocities pendicular to the wall are zero anyway.)
    //
    // We delete the x- and y- components of the heighfunction on all boundary
    // nodes here. This is far from perfect.
    // Note: ndnorm0 stores the heightfunction later on.

    // create the parameters for the discretization
    ParameterList eleparams;

    // set action for elements
    eleparams.set("action","calc_ale_node_normal");

    // create nonoverlapping vector as output for calc_ale_node_normal
    const Epetra_Map* dofrowmap = discret_->DofRowMap();
    Teuchos::RCP<Epetra_Vector> ndnorm0 = LINALG::CreateVector(*dofrowmap,true);

    //call loop over elements, note: normal vectors do not yet have length = 1.0
    //Although this evaluates the whole ALE-domain, only normals on surface
    //nodes are needed.
    discret_->ClearState();
    discret_->SetState("dispnp", dispnp_);
    discret_->Evaluate(eleparams,Teuchos::null,ndnorm0);
    discret_->ClearState();

    //======= get GIDs of free surface nodes w. heightfunction==

    std::vector< int > myGIDnodes;  // GIDs of fs nodes w. heightfunction owned by this processor
    std::set<int> mynodeset;
    std::set<int> nodeset;  // GIDs of fs nodes w. hf. owned by proc or adjacent
    const int myrank = discret_->Comm().MyPID();
    for (unsigned i=0; i<hfconds.size(); ++i)
    {
      const std::vector<int>* n = hfconds[i]->Nodes();
      for (unsigned j=0; j<n->size(); ++j)
      {
        const int gid = (*n)[j];
        if (discret_->HaveGlobalNode(gid))
        {
          nodeset.insert(gid);
          if (discret_->gNode(gid)->Owner()==myrank)
          {
            mynodeset.insert(gid);
          }
        }
      }
    }
    myGIDnodes.reserve(mynodeset.size());
    myGIDnodes.assign(mynodeset.begin(),mynodeset.end());

    // ========== manipulate ndnorm0(->heightfunc.) and displacements ===========================

    // Get the sum of all newton-increments on the free surface in this timestep.
    // This is what is manipulated by this part of heightfunction.
    incr_->Update(-1.0,*dispn_,1.0); // = ddisp; see AleSprings::Evaluate

    for (unsigned int node=0; node<(myGIDnodes.size()); node++)
    {
      //get LID for this node
      int rndLID = (discret_->NodeRowMap())->LID(myGIDnodes[node]);
      if (rndLID == -1) dserror("No LID for free surface node");

      //get vector of this rownode's dof GIDs
      std::vector< int > rdofGID;
      DRT::Node* fsnode = discret_->lRowNode(rndLID);
      discret_->Dof(fsnode, rdofGID);

      //get local indices for dofs in ndnorm0 and incr_
      int numdof = rdofGID.size();
      std::vector< int > rdofLID;
      rdofLID.resize(numdof);

      for (int i=0; i<numdof; i++)
      {
        int rgid = rdofGID[i];

#if DEBUG
        if (!ndnorm0->Map().MyGID(rgid))
          dserror("Sparse vector does not have global row  %d",rgid);

        if (!incr_->Map().MyGID(rgid))
          dserror("Sparse vector does not have global row  %d",rgid);

        if (ndnorm0->Map().LID(rgid) != incr_->Map().LID(rgid))
          dserror("Vectors don't match");
#endif

        rdofLID[i] = ndnorm0->Map().LID(rgid);
      }

#if DEBUG
      std::vector<double> gdbnorm(4);
#endif

      //check if this node is on the boundary of the free surface (e.g. on inflow
      //cross section):
      int NumElement = fsnode->NumElement();

      if (NumElement == 1 or NumElement == 2) // this node is on the boundary
      {
        DRT::Element** ElementsPtr = fsnode->Elements();

        for (int jEle = 0; jEle < NumElement; jEle++)
        {
          DRT::Element* Element = ElementsPtr[jEle];
          vector< RCP< DRT::Element > > Surfaces = Element->Surfaces();

          //which surfaces are adjacent to this node? These surfaces influenced
          //this node's node normal.
          for (unsigned int surfele = 0; surfele < Surfaces.size(); surfele++)
          {
            bool withthisnode = false;
            int numfsnodes = 0;

            const int* searchnodes = Surfaces[surfele]->NodeIds();
#ifdef DEBUG
            if (searchnodes==NULL) dserror("No Nodes in Surface Element");
#endif
            for (int num = 0; num < Surfaces[surfele]->NumNode(); num++)
            {
              if (searchnodes[num] == myGIDnodes[node])
              {
                withthisnode = true;
              }
              std::set<int>::iterator it = nodeset.find(searchnodes[num]);
              if (it != nodeset.end()) // another free surface node w. hf.
              {
                numfsnodes++;
              }
            }

            if (withthisnode and numfsnodes<Surfaces[surfele]->NumNode())
            {
              // this surface is adjacent to this node but no free surface
              // so it falsified or heighfunction on the node. Fix it.
              // this is not a mass-consistent solution!
              (*ndnorm0)[rdofLID[0]] = 0.0;
              (*ndnorm0)[rdofLID[1]] = 0.0;
              (*ndnorm0)[rdofLID[2]] = 1.0;
            }
          }
        }
    }

      // compute mass-consistent z-displacements for free surface nodes.
      // dx_G = dy_G = 0.0
      double pointproduct = 0.0;
      for (int i=0; i<numdof; i++)
      {
#if DEBUG
        gdbnorm[i] = (*ndnorm0)[rdofLID[i]];
#endif
        //mass-consistent heightfuntion acc. to Wall (7.13)
        //See also Wall and Genkinger 2.4.2. eq (21)
        (*ndnorm0)[rdofLID[i]] = (1.0/(*ndnorm0)[rdofLID[2]]) * (*ndnorm0)[rdofLID[i]];
        pointproduct += (*ndnorm0)[rdofLID[i]] * (*incr_)[rdofLID[i]];
      }

      for (int i=0; i<numdof; i++)
      {
        //The last entry of d_G is delta_phi/delta_t,
        //the other entries are zero
        if (i == 2)
          (*incr_)[rdofLID[i]]  = pointproduct;
        else
          (*incr_)[rdofLID[i]] = 0.0;
      }
    }

    // Get back to the sum of all increments and dispn_
    incr_->Update(1.0,*dispn_,1.0);

    // ======= END: manipulate ndnorm0 and displacements ==============================


    // =========== compute ale springs stiffness matrix ===============================
    // The ALE-springs-algorithm computes its stiffness matrix for the actual
    // diplacement! adapter_ale_springs_fixed_ref only uses the initial mesh geometry

    sysmat_->Zero();

    // zero out residual
    residual_->PutScalar(0.0);

    // set vector values needed by elements
    discret_->ClearState();

    // action for elements
    eleparams.set("action", "calc_ale_spring");

    // use newton-increments in displacements in ALE-Evaluate
    discret_->SetState("dispnp", incr_);
    discret_->Evaluate(eleparams,sysmat_,residual_);
    discret_->ClearState();

    sysmat_->Complete();
    // ========== END:  compute ale springs stiffness matrix ==========================



    // ========================= Loop over conditioned rownodes =======================
    // ======================= Manipulate ALE-systemmatrix ============================


    // THIS IS FOR AN ALE BLOCK MATRIX ONLY (as in the monolithic freesurface algorithm)
    // our sysmat_ is Teuchos::rcp(new LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>(freesurface_,freesurface_,81,false,true));
    // see ADAPTER::AleSprings::BuildSystemMatrix
    Teuchos::RCP<LINALG::BlockSparseMatrixBase> blockmat =
      Teuchos::rcp_dynamic_cast<LINALG::BlockSparseMatrixBase>(sysmat_);

    // Get the interface block of the ALE-matrix. This one is manipulated so
    // that the internal ALE-nodes are not dragged away by du_G_Fluid
    LINALG::SparseMatrix& aig = blockmat->Matrix(0,1);

    // This is the multiplier that projects the delta_u_gamma of the fluid on
    // the free direction (here: z) in a mass-consistent way acc. to heightfunction.
    Teuchos::RCP<LINALG::SparseMatrix> H  =
      Teuchos::rcp(new LINALG::SparseMatrix(*(freesurface_.CondMap()),3,false,false,LINALG::SparseMatrix::CRS_MATRIX));
    Teuchos::RCP<Epetra_CrsMatrix> ep_H = H->EpetraMatrix();

    for (unsigned int node=0; node<(myGIDnodes.size()); node++)
    {
      //get LID for this node
      int rndLID = (discret_->NodeRowMap())->LID(myGIDnodes[node]);
      if (rndLID == -1) dserror("No LID for free surface node");

      //get vector of this node's dof GIDs
      std::vector< int > dofGID;
      discret_->Dof(discret_->lRowNode(rndLID), dofGID);

      //get local indices for dofs in ndnorm0
      const int numdof = dofGID.size();
      std::vector< int > dofLID;
      dofLID.resize(numdof);

      for (int i=0; i<numdof; i++)
      {
        int rgid = dofGID[i];

#if DEBUG
        if (!ndnorm0->Map().MyGID(rgid))
          dserror("Vector does not have global row %d",rgid);
#endif

        dofLID[i] = ndnorm0->Map().LID(rgid);
      }


     //get values and indices of entries in this row
      std::vector<double> values(numdof);
      std::vector<int> cgids(numdof);

      for (int i=0; i<numdof; i++)
      {
        values[i] = (*ndnorm0)[dofLID[i]];
        cgids[i] = dofGID[i];
      }

      // write in 3rd row only. Internal ALE nodes cannot see Fluidvelocities in x and y direction.
      int rgid = dofGID[2];
      const int err = ep_H->InsertGlobalValues(rgid,3,&values[0],&cgids[0]);
      if (err!=0) dserror("Epetra_CrsMatrix::InsertGlobalValues returned error code %d",err);

    }

    H->Complete();

    Teuchos::RCP<LINALG::SparseMatrix> Mult = LINALG::Multiply(aig,false,*H,false,true);

    aig = *Mult;
  }
  // ====================================================================================
  //
  // End of Heightfunction
  // ====================================================================================
  // ====================================================================================


  else
  {
    // The ALE-springs-algorithm computes its stiffness matrix for the actual
    // diplacement! adapter_ale_springs_fixed_ref only uses the initial mesh geometry

    sysmat_->Zero();

    // zero out residual
    residual_->PutScalar(0.0);

    // create the parameters for the discretization
    ParameterList eleparams;

    // set vector values needed by elements
    discret_->ClearState();

    // action for elements
    eleparams.set("action", "calc_ale_spring");

    // newton-increments in displacements are of no importance for ALE-Evaluate
    discret_->SetState("dispnp", dispn_);
    discret_->Evaluate(eleparams,sysmat_,residual_);
    discret_->ClearState();

    sysmat_->Complete();
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::ApplyInterfaceDisplacements(Teuchos::RCP<Epetra_Vector> idisp)
{
  interface_.InsertCondVector(idisp,dispnp_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::ApplyFreeSurfaceDisplacements(Teuchos::RCP<Epetra_Vector> fsdisp)
{
  freesurface_.InsertCondVector(fsdisp,dispnp_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::AleSprings::ExtractDisplacement() const
{
  return incr_;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ADAPTER::AleSprings::ReadRestart(int step)
{
  IO::DiscretizationReader reader(discret_,step);
  time_ = reader.ReadDouble("time");
  step_ = reader.ReadInt("step");

  reader.ReadVector(dispnp_, "dispnp");
  reader.ReadVector(dispn_,  "dispn");
}


#endif
