/*!----------------------------------------------------------------------
\file immersed_partitioned_adhesion_traction.cpp
\level 2

\brief partitioned immersed cell-ecm interaction via adhesion traction

<pre>
Maintainers: Andreas Rauch
             rauch@lnm.mw.tum.de
             http://www.lnm.mw.tum.de
             089 - 289 -15240
</pre>
*----------------------------------------------------------------------*/
#include "immersed_partitioned_adhesion_traction.H"

#include "../drt_structure/stru_aux.H"
#include "../drt_poroelast/poro_scatra_base.H"
#include "../drt_adapter/ad_str_fsiwrapper_immersed.H"
#include "../drt_adapter/ad_fld_poro.H"
#include "../drt_fluid_ele/fluid_ele_action.H"
#include "../linalg/linalg_utils.H"


IMMERSED::ImmersedPartitionedAdhesionTraction::ImmersedPartitionedAdhesionTraction(const Teuchos::ParameterList& params, const Epetra_Comm& comm)
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

  // create instance of poroelast subproblem
  poroscatra_subproblem_ = params.get<Teuchos::RCP<POROELAST::PoroScatraBase> >("RCPToPoroScatra");

  // set pointer to poro fpsi structure
  porostructure_ = poroscatra_subproblem_->PoroField()->StructureField();

  // important variables for parallel simulations
  myrank_  = comm.MyPID();
  numproc_ = comm.NumProc();

  // get pointer to global problem
  globalproblem_ = DRT::Problem::Instance();

  backgroundfluiddis_     = globalproblem_->GetDis("porofluid");
  backgroundstructuredis_ = globalproblem_->GetDis("structure");
  immerseddis_            = globalproblem_->GetDis("cell");

  // construct immersed exchange manager. singleton class that makes immersed variables comfortably accessible from everywhere in the code
  exchange_manager_ = DRT::ImmersedFieldExchangeManager::Instance();
  exchange_manager_->SetIsPureAdhesionSimulation(params.get<bool>("IsPureAdhesionSimulation")==true);

  // get coupling variable
  displacementcoupling_ = globalproblem_->ImmersedMethodParams().sublist("PARTITIONED SOLVER").get<std::string>("COUPVARIABLE_ADHESION") == "Displacement";
  if(displacementcoupling_ and myrank_==0)
    std::cout<<" Coupling variable for partitioned Cell-ECM Adhesion Dynamics scheme :  Displacements "<<std::endl;
  else if (!displacementcoupling_ and myrank_==0)
    std::cout<<" Coupling variable for partitioned Cell-ECM Adhesion Dynamics scheme :  Force "<<std::endl;

  // vector of adhesion forces in ecm
  ecm_adhesion_forces_ = Teuchos::rcp(new Epetra_Vector(*(poroscatra_subproblem_->StructureField()->DofRowMap()),true));
  cell_adhesion_disp_ = Teuchos::rcp(new Epetra_Vector(*(cellstructure_->DofRowMap()),true));
  Freact_cell_ = cellstructure_->Freact();

  // set pointer to adhesion force vector in immersed exchange manager
  exchange_manager_->SetPointerECMAdhesionForce(ecm_adhesion_forces_);
  //curr_subset_of_backgrounddis_=exchange_manager_->GetPointerToCurrentSubsetOfBackgrdDis(); todo build in global control algo and hand pointer into ParameterList

  // PSEUDO2D switch
  isPseudo2D_ = DRT::INPUT::IntegralValue<int>(globalproblem_->CellMigrationParams(),"PSEUDO2D");

  // initial invalid immersed information
  immersed_information_invalid_=true;

  // output after every fixed-point iteration?
  output_evry_nlniter_=false;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::CouplingOp(const Epetra_Vector &x, Epetra_Vector &F, const FillType fillFlag)
{
  ReinitTransferVectors();

  // DISPLACEMENT COUPLING
  if (displacementcoupling_)
  { dserror("COUPVARIABLE 'Displacement' is not tested, yet!!");
    const Teuchos::RCP<Epetra_Vector> idispn = Teuchos::rcp(new Epetra_Vector(x));

    ////////////////////
    // CALL BackgroundOp
    ////////////////////
    PrepareBackgroundOp();
    BackgroundOp(Teuchos::null, fillFlag);

    ////////////////////
    // CALL ImmersedOp
    ////////////////////
    //PrepareImmersedOp();
    Teuchos::RCP<Epetra_Vector> idispnp =
        ImmersedOp(Teuchos::null, fillFlag);

    int err = F.Update(1.0, *idispnp, -1.0, *idispn, 0.0);
    if(err != 0)
      dserror("Vector update of Coupling-residual returned err=%d",err);

  }
  // FORCE COUPLING
  else if(!displacementcoupling_)
  {
    const Teuchos::RCP<Epetra_Vector> cell_reactforcen = Teuchos::rcp(new Epetra_Vector(x));

    ////////////////////
    // CALL BackgroundOp
    ////////////////////
    PrepareBackgroundOp();
    BackgroundOp(cell_reactforcen, fillFlag);

    ////////////////////
    // CALL ImmersedOp
    ////////////////////
    //PrepareImmersedOp();
    ImmersedOp(Teuchos::null, fillFlag);

    int err = F.Update(1.0, *Freact_cell_, -1.0, *cell_reactforcen, 0.0);

    if(err != 0)
      dserror("Vector update of FSI-residual returned err=%d",err);

  } // displacement / force coupling

  // write output after every solve of ECM and Cell
  // current limitations:
  // max 100 partitioned iterations and max 100 timesteps in total
  if(output_evry_nlniter_)
  {
    int iter = ((FSI::Partitioned::IterationCounter())[0]);
    Teuchos::rcp_dynamic_cast<ADAPTER::FluidAleImmersed>(MBFluidField())->Output((Step()*100)+(iter-1),Time()-Dt()*((100-iter)/100.0));
    StructureField()->PrepareOutput();
    Teuchos::rcp_dynamic_cast<ADAPTER::FSIStructureWrapperImmersed>(StructureField())->Output(false,(Step()*100)+(iter-1),Time()-Dt()*((100-iter)/100.0));
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::BackgroundOp(Teuchos::RCP<Epetra_Vector> backgrd_dirichlet_values,
                                                              const FillType fillFlag)
{
  IMMERSED::ImmersedPartitioned::BackgroundOp(backgrd_dirichlet_values,fillFlag);

  if (fillFlag==User)
  {
    dserror("fillFlag == User : not yet implemented");
  }
  else
  {
    Teuchos::ParameterList params;

    if(immersed_information_invalid_)
    {
      if(curr_subset_of_backgrounddis_.empty()==false)
        EvaluateImmersedNoAssembly(params,
            backgroundfluiddis_,
            &curr_subset_of_backgrounddis_,
            cell_SearchTree_, currpositions_cell_,
            (int)FLD::update_immersed_information);
      else
      {
        // do nothing : a proc without subset of backgrd dis. does not need to enter evaluation,
        //              since cell is ghosted on all procs and no communication has to be performed.
      }

      immersed_information_invalid_=false;
    }

    DistributeAdhesionForce(backgrd_dirichlet_values);

    // solve poro
    poroscatra_subproblem_->Solve();

  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
IMMERSED::ImmersedPartitionedAdhesionTraction::ImmersedOp(Teuchos::RCP<Epetra_Vector> bdry_traction,
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
    //prescribe dirichlet values at cell adhesion nodes
    CalcAdhesionDisplacements();
    DoImmersedDirichletCond(cellstructure_->WriteAccessDispnp(),cell_adhesion_disp_, cellstructure_->GetDBCMapExtractor()->CondMap());

    if(IterationCounter()[0]>1)
      cellstructure_->PreparePartitionStep();

     //solve cell
    cellstructure_->Solve();

    return cellstructure_->ExtractImmersedInterfaceDispnp();
  }

}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::PrepareBackgroundOp()
{

  immerseddis_->SetState(0,"displacement",cellstructure_->Dispnp());
  backgroundfluiddis_->SetState(0,"dispnp",poroscatra_subproblem_->FluidField()->Dispnp());

  double structsearchradiusfac = DRT::Problem::Instance()->ImmersedMethodParams().get<double>("STRCT_SRCHRADIUS_FAC");

  // determine subset of fluid discretization which is potentially underlying the immersed discretization
  //
  // get state
  Teuchos::RCP<const Epetra_Vector> celldisplacements = cellstructure_->Dispnp();

  // find current positions for immersed structural discretization
  std::map<int,LINALG::Matrix<3,1> > my_currpositions_cell;
  for (int lid = 0; lid < immerseddis_->NumMyRowNodes(); ++lid)
  {
    const DRT::Node* node = immerseddis_->lRowNode(lid);
    LINALG::Matrix<3,1> currpos;
    std::vector<int> dofstoextract(3);
    std::vector<double> mydisp(3);

    // get the current displacement
    immerseddis_->Dof(node,0,dofstoextract);
    DRT::UTILS::ExtractMyValues(*celldisplacements,mydisp,dofstoextract);

    currpos(0) = node->X()[0]+mydisp.at(0);
    currpos(1) = node->X()[1]+mydisp.at(1);
    currpos(2) = node->X()[2]+mydisp.at(2);

    my_currpositions_cell[node->Id()] = currpos;
  }

  // communicate local currpositions:
  // map with current cell positions should be same on all procs
  // to make use of the advantages of ghosting the cell redundantly
  // on all procs.
  int procs[Comm().NumProc()];
  for(int i=0;i<Comm().NumProc();i++)
    procs[i]=i;
  LINALG::Gather<int,LINALG::Matrix<3,1> >(my_currpositions_cell,*currpositions_cell_,Comm().NumProc(),&procs[0],Comm());

  // get bounding box of current configuration of structural dis
  const LINALG::Matrix<3,2> structBox = GEO::getXAABBofDis(*immerseddis_,*currpositions_cell_);
  double max_radius = sqrt(pow(structBox(0,0)-structBox(0,1),2)+pow(structBox(1,0)-structBox(1,1),2)+pow(structBox(2,0)-structBox(2,1),2));
  // search for background elements within a certain radius around the center of the immersed bounding box
  LINALG::Matrix<3,1> boundingboxcenter;
  boundingboxcenter(0) = structBox(0,0)+(structBox(0,1)-structBox(0,0))*0.5;
  boundingboxcenter(1) = structBox(1,0)+(structBox(1,1)-structBox(1,0))*0.5;
  boundingboxcenter(2) = structBox(2,0)+(structBox(2,1)-structBox(2,0))*0.5;

  // search background elements covered by bounding box of cell
  curr_subset_of_backgrounddis_ = fluid_SearchTree_->searchElementsInRadius(*backgroundfluiddis_,*currpositions_ECM_,boundingboxcenter,structsearchradiusfac*max_radius,0);

  if(curr_subset_of_backgrounddis_.empty() == false)
    std::cout<<"\nPrepareBackgroundOp returns "<<curr_subset_of_backgrounddis_.begin()->second.size()<<" background elements on Proc "<<Comm().MyPID()<<std::endl;

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::PrepareTimeStep()
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
Teuchos::RCP<Epetra_Vector>
IMMERSED::ImmersedPartitionedAdhesionTraction::InitialGuess()
{
  if(displacementcoupling_)
    return cellstructure_->PredictImmersedInterfaceDispnp();
  else // FORCE COUPLING
  {
    return cellstructure_->Freact();
  }

}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::CalcAdhesionForce()
{

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::DistributeAdhesionForce(Teuchos::RCP<Epetra_Vector> freact)
{
  double adhesionforcenorm = -1234.0;

  immerseddis_->SetState(0,"displacement",cellstructure_->Dispnp());
  backgroundstructuredis_->SetState(0,"displacement",porostructure_->Dispnp());

  // reinitialize map to insert new adhesion node -> background element pairs
  adh_nod_param_coords_in_backgrd_ele_.clear();

  if(myrank_ == 0)
  {
    std::cout<<"################################################################################################"<<std::endl;
    std::cout<<"###   Spread adhesion forces onto ecm ...                  "<<std::endl;
  }

  double tol = 1e-13;

  bool match = false;

  DRT::Element* ele;  //!< pointer to background structure ecm element
  DRT::Element* iele; //!< pointer to background fluid element (carries immersed information)

  LINALG::Matrix<3,1> xi(true);        //!< parameter space coordinate of anode in background element
  LINALG::Matrix<8,1> shapefcts;       //!< shapefunctions must be evaluated at xi
  std::vector<double> anode_coord(3);  //!< current coordinates of anode
  std::vector<double> myvalues;        //!< processor local ecm element displacements

  // get reaction forces at cell surface
  double* freact_values = freact->Values();
  double freactnorm=-1234.0;
  freact->Norm2(&freactnorm);
  if(myrank_ == 0)
  {
    std::cout<<"###   L2-Norm of Cell Reaction Force Vector: "<<std::setprecision(11)<<freactnorm<<std::endl;
  }

  // get condition which marks adhesion nodes
  DRT::Condition* condition = immerseddis_->GetCondition("CellFocalAdhesion");
  const std::vector<int>* adhesion_nodes = condition->Nodes();
  int adhesion_nodesize = adhesion_nodes->size();
  DRT::Node* adhesion_node;

  Teuchos::RCP<const Epetra_Vector> ecmstate = backgroundstructuredis_->GetState("displacement");
  if(ecmstate == Teuchos::null)
    dserror("Could not get state displacement from background structure");

  Teuchos::RCP<const Epetra_Vector> cellstate = immerseddis_->GetState("displacement");
  if(cellstate == Teuchos::null)
    dserror("Could not get state displacement from cell structure");


  // loop over all adhesion nodes
  for(int anode=0;anode<adhesion_nodesize;anode++)
  {
    xi(0)=2.0; xi(1)=2.0; xi(2)=2.0;
    match=false;

    int anodeid = adhesion_nodes->at(anode);
    adhesion_node = immerseddis_->gNode(anodeid);

    // get coordinates of adhesion node
    const double* X = adhesion_node->X();

    // get displacements
    DRT::Element** adjacent_elements = adhesion_node->Elements();
    DRT::Element::LocationArray la(1);
    adjacent_elements[0]->LocationVector(*immerseddis_,la,false);
    // extract local values of the global vectors
    myvalues.resize(la[0].lm_.size());
    DRT::UTILS::ExtractMyValues(*cellstate,myvalues,la[0].lm_);
    double adhesioneledisp[24];
    for(int node=0;node<8;++node)
      for(int dof=0; dof<3;++dof)
        adhesioneledisp[node*3+dof]=myvalues[node*3+dof];

    // determine which node of element is anode
    int locid = -1;
    DRT::Node** nodes = adjacent_elements[0]->Nodes();
    for(int i=0;i<adjacent_elements[0]->NumNode();++i)
    {
      if(nodes[i]->Id() == anodeid)
      {
        locid = i;
        break;
      }
    }

    if(locid == -1)
      dserror("could not get local index of adhesion node in element");

    // fill vector with current coordinate
    anode_coord[0] = X[0] + adhesioneledisp[locid*3+0];
    anode_coord[1] = X[1] + adhesioneledisp[locid*3+1];
    anode_coord[2] = X[2] + adhesioneledisp[locid*3+2];

    // find ecm element in which adhesion node is immersed
    // every proc that has searchboxgeom elements
    for(std::map<int, std::set<int> >::const_iterator closele = curr_subset_of_backgrounddis_.begin(); closele != curr_subset_of_backgrounddis_.end(); closele++)
    {
      if(match)
        break;

      for(std::set<int>::const_iterator eleIter = (closele->second).begin(); eleIter != (closele->second).end(); eleIter++)
      {
        if(match)
          break;

        ele=backgroundstructuredis_->gElement(*eleIter);
        iele=backgroundfluiddis_->gElement(*eleIter);

        DRT::ELEMENTS::FluidImmersedBase* immersedele = dynamic_cast<DRT::ELEMENTS::FluidImmersedBase*>(iele);
        if(immersedele == NULL)
          dserror("dynamic cast from DRT::Element* to DRT::ELEMENTS::FluidImmersedBase* failed");

        if(immersedele->IsBoundaryImmersed())
        {
          bool converged = false;
          double residual = -1234.0;

          DRT::Element::LocationArray la(1);
          ele->LocationVector(*backgroundstructuredis_,la,false);
          // extract local values of the global vectors
          myvalues.resize(la[0].lm_.size());
          DRT::UTILS::ExtractMyValues(*ecmstate,myvalues,la[0].lm_);
          double sourceeledisp[24];
          for(int node=0;node<8;++node)
            for(int dof=0; dof<3;++dof)
              sourceeledisp[node*3+dof]=myvalues[node*4+dof];

          // node 1  and node 7 coords of current source element (diagonal points)
          const double* X1 = ele->Nodes()[1]->X();
          double x1[3];
          x1[0]=X1[0]+sourceeledisp[1*3+0];
          x1[1]=X1[1]+sourceeledisp[1*3+1];
          x1[2]=X1[2]+sourceeledisp[1*3+2];
          const double* X7 = ele->Nodes()[7]->X();
          double diagonal = sqrt(pow(X1[0]-X7[0],2)+pow(X1[1]-X7[1],2)+pow(X1[2]-X7[2],2));

          // calc distance of current anode to arbitrary node (e.g. node 1) of curr source element
          double distance = sqrt(pow(x1[0]-anode_coord[0],2)+pow(x1[1]-anode_coord[1],2)+pow(x1[2]-anode_coord[2],2));

          // get parameter space coords xi in source element of global point anode
          // NOTE: if the anode is very far away from the source element ele
          //       it is unnecessary to jump into this method and invoke a newton iteration.
          // Therefore: only call GlobalToCurrentLocal if distance is smaller than factor*characteristic element length
          if(distance < 2.5*diagonal)
          {
            MORTAR::UTILS::GlobalToCurrentLocal<DRT::Element::hex8>(*ele,&sourceeledisp[0],&anode_coord[0],&xi(0),converged,residual);
            //MORTAR::UTILS::GlobalToLocal<DRT::Element::hex8>(*ele,&anode_coord[0],&xi(0),converged);
            if(converged == false)
            {
              std::cout<<"Warning! GlobalToCurrentLocal did not converge for adhesion node "<<anodeid<<". Res="<<residual<<std::endl;
              xi(0)=2.0; xi(1)=2.0; xi(2)=2.0;
            }
          }
          else
          {
            xi(0)=2.0; xi(1)=2.0; xi(2)=2.0;
          }
        } // if cut by boundary

        // anode lies in element ele
        if (abs(xi(0))<(1.0+tol) and abs(xi(1))<(1.0+tol) and abs(xi(2))<(1.0+tol))
        {
          match = true;

          // write pair cell adhesion node id -> xi in backgroundele in map
          adh_nod_param_coords_in_backgrd_ele_.insert(std::pair<int,LINALG::Matrix<3,1> >(anodeid,xi));
          // write pair adhesion node id -> backgrdele id in map
          adh_nod_backgrd_ele_mapping_.insert(std::pair<int,int>(anodeid,ele->Id()));

          // spread force to nodes of ecm ele
          std::vector<int> dofs = immerseddis_->Dof(0,adhesion_node);
          if(dofs.size()!=3)
            dserror("dofs=3 expected. dofs=%d instead",dofs.size());

          for(int dof=0;dof<(3-isPseudo2D_);dof++)
          {
            int doflid = freact->Map().LID(dofs[dof]);
            double dofval = freact_values[doflid];

            // evaluate shapefcts of ele at point xi
            DRT::UTILS::shape_function<DRT::Element::hex8>(xi,shapefcts);

            // write entry into ecm_adhesion_forces_
            DRT::Node** spreadnodes = ele->Nodes();
            for(int snode=0;snode<8;snode++)
            {
              std::vector<int> sdofs = backgroundstructuredis_->Dof(0,spreadnodes[snode]);
              if(sdofs.size()!=4)
                dserror("dofs=4 expected. dofs=%d instead",sdofs.size()); // 4 dofs per node in porostructure

              int error = ecm_adhesion_forces_->SumIntoGlobalValue(sdofs[dof],0,0,shapefcts(snode)*dofval);
              if(error != 0)
                dserror("SumIntoGlobalValue returned err=%d",error);
            } // node loop
          }// dof loop

        } // adhesion node is matched

      } // loop over element ids
    } // loop over curr_subset_of_backgrounddis_

  } // loop over all adhesion nodes

  ecm_adhesion_forces_->Norm2(&adhesionforcenorm);

  if(myrank_ == 0)
  {
    std::cout<<"###   L2-Norm of  ECM Adhesion Force Vector: "<<std::setprecision(11)<<adhesionforcenorm<<std::endl;
    std::cout<<"################################################################################################"<<std::endl;
  }
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::CalcAdhesionDisplacements()
{
  double adhesiondispnorm = -1234.0;

    if(myrank_ == 0)
    {
      std::cout<<"################################################################################################"<<std::endl;
      std::cout<<"###   Apply Dirichlet to Adhering Nodes                  "<<std::endl;
    }

    LINALG::Matrix<3,1> xi(true);        //!< parameter space coordinate of anode in background element
    LINALG::Matrix<8,1> shapefcts;       //!< shapefunctions must be evaluated at xi
    LINALG::Matrix<3,1> adhesiondisp;    //!< displacement of ECM element interpolated to adhesion node

    DRT::Element* backgrdele; //! pointer to ECM element

    // loop over all cell adhesion nodes
    for (std::map<int,LINALG::Matrix<3,1> >::iterator it=adh_nod_param_coords_in_backgrd_ele_.begin(); it!=adh_nod_param_coords_in_backgrd_ele_.end(); ++it)
    {
      backgrdele = backgroundstructuredis_->gElement(adh_nod_backgrd_ele_mapping_.at(it->first));
      adhesiondisp.PutScalar(0.0);

      // 1) extract background element nodal displacements
      DRT::Element::LocationArray la(1);
      backgrdele->LocationVector(*backgroundstructuredis_,la,false);
      std::vector<double> myvalues(la[0].lm_.size());
      DRT::UTILS::ExtractMyValues(*(poroscatra_subproblem_->StructureField()->Dispnp()),myvalues,la[0].lm_);


      // 2) interpolate ECM displacement to parameter space coordinate occupied by the cell adhesion node prior to deformation.
      //    the adhesion node is supposed to be fixed to the same material point.

      // get parameter space coordinate of cell adhesion node
      xi=it->second;

      // evaluate shapefcts of ele at point xi
      DRT::UTILS::shape_function<DRT::Element::hex8>(xi,shapefcts);

      // interpolate
      for(int node=0;node<backgrdele->NumNode();node++)
      {
        adhesiondisp(0)+=shapefcts(node)*myvalues[node*4+0];
        adhesiondisp(1)+=shapefcts(node)*myvalues[node*4+1];
        adhesiondisp(2)+=shapefcts(node)*myvalues[node*4+2];
      }


      // 3) apply previously calculated adhesion node displacement to cell

      // get dofs of cell adhesion node
      std::vector<int> dofs = immerseddis_->Dof(0,immerseddis_->gNode(it->first));
      if(dofs.size()!=3)
        dserror("dofs=3 expected. dofs=%d instead",dofs.size());

      // write displacmement into global vector
      for(int dof=0;dof<(3-isPseudo2D_);dof++)
      {
        // get lid of current gid in 'dofs'
        int lid = immerseddis_->DofRowMap()->LID(dofs[dof]);
        int error = cell_adhesion_disp_->SumIntoMyValue(lid,0,adhesiondisp(dof));
        if(error != 0)
          dserror("SumIntoMyValue returned err=%d",error);
      }

    } // do for all elements in map adh_nod_param_coords_in_backgrd_ele_

    cell_adhesion_disp_->Norm2(&adhesiondispnorm);

    if(myrank_ == 0)
    {
      std::cout<<"###   L2-Norm of Cell Adhesion Disp Vector: "<<std::setprecision(11)<<adhesiondispnorm<<std::endl;
      std::cout<<"################################################################################################"<<std::endl;
    }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::ReadRestart(int step)
{
  dserror("RESTART in ImmersedPartitionedAdhesionTraction not supported, yet.");
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::PrepareOutput()
{
  cellstructure_->PrepareOutput();
  poroscatra_subproblem_->PrepareOutput();
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::Output()
{
  cellstructure_->Output();
  poroscatra_subproblem_->Output();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::Update()
{
  cellstructure_->Update();
  poroscatra_subproblem_->Update();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::BuildImmersedDirichMap(Teuchos::RCP<DRT::Discretization> dis,
                                                                        Teuchos::RCP<Epetra_Map>& dirichmap,
                                                                        const Teuchos::RCP<const Epetra_Map>& dirichmap_original,
                                                                        int dofsetnum)
{
  const Epetra_Map* elerowmap = dis->ElementRowMap();
  std::vector<int> mydirichdofs(0);

  for(int i=0; i<elerowmap->NumMyElements(); ++i)
  {
    // dynamic_cast necessary because virtual inheritance needs runtime information
    DRT::ELEMENTS::FluidImmersedBase* immersedele = dynamic_cast<DRT::ELEMENTS::FluidImmersedBase*>(dis->gElement(elerowmap->GID(i)));
    if(immersedele->HasProjectedDirichlet())
    {
      DRT::Node** nodes = immersedele->Nodes();
      for (int inode=0; inode<(immersedele->NumNode()); inode++)
      {
        if(static_cast<IMMERSED::ImmersedNode* >(nodes[inode])->IsMatched() and nodes[inode]->Owner()==myrank_)
        {
          std::vector<int> dofs = dis->Dof(dofsetnum,nodes[inode]);

          for (int dim=0;dim<3;++dim)
          {
            if(dirichmap_original->LID(dofs[dim]) == -1) // if not already in original dirich map
              mydirichdofs.push_back(dofs[dim]);
          }

        }
      }
    }
  }

  int nummydirichvals = mydirichdofs.size();
  dirichmap = Teuchos::rcp( new Epetra_Map(-1,nummydirichvals,&(mydirichdofs[0]),0,dis->Comm()) );

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IMMERSED::ImmersedPartitionedAdhesionTraction::DoImmersedDirichletCond(Teuchos::RCP<Epetra_Vector> statevector, Teuchos::RCP<Epetra_Vector> dirichvals, Teuchos::RCP<const Epetra_Map> dbcmap)
{
  int mynumvals = dbcmap->NumMyElements();
  double* myvals = dirichvals->Values();

  for(int i=0;i<mynumvals;++i)
  {
    int gid = dbcmap->GID(i);

#ifdef DEBUG
    int err = -2;
    int lid = dirichvals->Map().LID(gid);
    err = statevector -> ReplaceGlobalValue(gid,0,myvals[lid]);
    if(err==-1)
      dserror("VectorIndex >= NumVectors()");
    else if (err==1)
        dserror("GlobalRow not associated with calling processor");
    else if (err != -1 and err != 1 and err != 0)
      dserror("Trouble using ReplaceGlobalValue on fluid state vector. ErrorCode = %d",err);
#else
    int lid = dirichvals->Map().LID(gid);
    statevector -> ReplaceGlobalValue(gid,0,myvals[lid]);
#endif

  }

  return;
}

