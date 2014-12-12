/*----------------------------------------------------------------------*/
/*!
\file mc_var_thickness_manager.H
\brief class to modify wall thickness of cardiovascular structures on the fly

<pre>
Maintainer: Jonas Biehler
            biehler@lnm.mw.tum.de
            089 - 28915276
</pre>

!*/

/*----------------------------------------------------------------------------*/
/* headers */

#ifdef HAVE_FFTW

#include "mc_var_thickness_manager.H"
#include "../drt_inpar/inpar_mlmc.H"
#include "../drt_inpar/inpar_material.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_utils_createdis.H"
#include "../drt_lib/drt_utils_materials.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_element.H"
#include "../drt_io/io.H"
#include "../drt_io/io_pstream.H"
#include "../drt_mat/material.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_comm/comm_utils.H"
#include "../linalg/linalg_utils.H"
#include "randomfield.H"
#include "randomfield_fourier.H"
#include "randomfield_spectral.H"
#include "../drt_ale/ale_utils_clonestrategy.H"
#include "../drt_adapter/ad_ale_wrapper.H"
#include "../drt_adapter/adapter_coupling_mortar.H"
#include "../drt_adapter/adapter_coupling.H"
#include "../drt_mortar/mortar_interface.H"
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
UQ::MCVarThicknessManager::MCVarThicknessManager(
    Teuchos::RCP<DRT::Discretization> discret,
    const int my_thickness_field_id):
    discret_(discret),
    aledis_(Teuchos::null),
    randomfield_(Teuchos::null),
    my_uncert_nodeids_(),
    my_uncert_nnodes_(-1),
    my_uncertain_nodes_org_pos_(),
    my_uncertain_nodes_dbc_onoff_(),
    my_uncertain_nodes_dbc_val_(),
    my_uncertain_nodes_dbc_curve_(),
    my_uncertain_nodes_dbc_funct_(),
    my_uncertain_nodes_delta_pos_(),
    org_geom_(Teuchos::null),
    initial_wall_thickness_(-1.0),
    use_mean_field_from_cond_(false),
    start_rand_geo_above_bif_(false),
    z_pos_start_rf_(-10e12),
    transition_width_(0.0),
    my_uncertain_nodes_normals_()
{
  if (not discret_->Filled() || not discret_->HaveDofs())
    dserror("Discretisation is not complete or has no dofs!");

  // input parameters structural dynamics
  const Teuchos::ParameterList& sdyn =
      DRT::Problem::Instance()->StructuralDynamicParams();

  //check if quasistatic analysis
  if (sdyn.get<std::string>("DYNAMICTYP") != "Statics")
    dserror(
        "Structure with ale only for quasistatic analysis so in new sti so far.");


  // input parameters
  const Teuchos::ParameterList& mlmcp =
     DRT::Problem::Instance()->MultiLevelMonteCarloParams();

  initial_wall_thickness_ = mlmcp.get<double>("INITIALTHICKNESS");
  use_mean_field_from_cond_ = DRT::INPUT::IntegralValue<int>(mlmcp,
     "MEAN_GEO_FROM_ALE_POINTDBC");

  start_rand_geo_above_bif_ = DRT::INPUT::IntegralValue<int>(mlmcp,
     "START_RF_ABOVE_BIFURCATION");

  z_pos_start_rf_ = mlmcp.get<double>("Z_POS_AAA_START_RF");
  transition_width_ = mlmcp.get<double>("TRANSITION_WIDTH");


  // create empty discretization and add it to the problem
  aledis_ = Teuchos::rcp(
      new DRT::Discretization("ale",
          DRT::Problem::Instance()->GetNPGroup()->LocalComm()));

  aledis_->SetWriter(Teuchos::rcp(new IO::DiscretizationWriter(aledis_)));

  DRT::Problem::Instance()->AddDis("ale", aledis_);

  if (!aledis_->Filled())
     aledis_->FillComplete();

  // we use the structure discretization as layout for the ale discretization
  if (discret_->NumGlobalNodes() == 0)
    dserror("Structure discretization is empty!");

  // clone ale mesh from structure discretization
  if (aledis_->NumGlobalNodes() == 0)
  {
    DRT::UTILS::CloneDiscretization<ALE::UTILS::AleCloneStrategy>(discret_,
        aledis_);
    // setup material in every ALE element
    Teuchos::ParameterList params;
    params.set<std::string>("action", "setup_material");
    aledis_->Evaluate(params);
  }
  else
    dserror(
        "Reading an ALE mesh from the input file is not supported for this problem type.");

  // initialize random field
  int my_dummy_seed = 1;
  randomfield_ = CreateRandomField(my_thickness_field_id, my_dummy_seed);

  // get the conditions for the current evaluation
  std::vector<DRT::Condition*> uncert_surface;
  aledis_->GetCondition("UncertainSurface", uncert_surface);

  // check wether length of condition is one
  if (uncert_surface.size() != 1)
    dserror("Uncertain Surface currently only implemented for 1 condition only");

  // get all point dirich conditions
  std::vector<DRT::Condition*> temp;
  std::vector<DRT::Condition*> uncertain_nodes_condition;

  aledis_->GetCondition("Dirichlet", temp);
  if(temp.size())
  {
    // obtain just point conditions
    for(int i=0;i<(int)temp.size();++i)
    {
      if( temp.at(i)->Type() == DRT::Condition::PointDirichlet )
      {
        uncertain_nodes_condition.push_back(temp.at(i));
      }

    }
  }

  // store point dbc in maps
  if (uncertain_nodes_condition.size() && use_mean_field_from_cond_)
  {
    // loop all point conditions
    for (int i = 0; i < (int) uncertain_nodes_condition.size(); ++i)
    {
      //
      if (uncertain_nodes_condition[i]->Nodes()->size() == 1)
      {
        my_uncert_nodeids_.push_back(
            uncertain_nodes_condition[i]->Nodes()->at(0));

        int temp_node_id = uncertain_nodes_condition[i]->Nodes()->at(0);
        const std::vector<int>* temp_onoff = uncertain_nodes_condition[i]->Get<
            std::vector<int> >("onoff");
        const std::vector<double>* temp_val = uncertain_nodes_condition[i]->Get<
            std::vector<double> >("val");
        const std::vector<int>* temp_curve = uncertain_nodes_condition[i]->Get<
            std::vector<int> >("curve");
        const std::vector<int>* temp_funct = uncertain_nodes_condition[i]->Get<
            std::vector<int> >("funct");
        my_uncertain_nodes_dbc_onoff_.insert(
            std::pair<int, std::vector<int> >(temp_node_id, *temp_onoff));
        my_uncertain_nodes_dbc_val_.insert(
            std::pair<int, std::vector<double> >(temp_node_id, *temp_val));
        my_uncertain_nodes_dbc_curve_.insert(
            std::pair<int, std::vector<int> >(temp_node_id, *temp_curve));
        my_uncertain_nodes_dbc_funct_.insert(
            std::pair<int, std::vector<int> >(temp_node_id, *temp_funct));
      }
      else
        dserror("PointDirichlet condition contains more than one node!");
    }
    LINALG::GatherAll(my_uncertain_nodes_dbc_val_, aledis_->Comm());
    LINALG::GatherAll(my_uncertain_nodes_dbc_onoff_, aledis_->Comm());
    LINALG::GatherAll(my_uncertain_nodes_dbc_curve_, aledis_->Comm());
    LINALG::GatherAll(my_uncertain_nodes_dbc_funct_, aledis_->Comm());


    my_uncert_nnodes_ = (int) my_uncert_nodeids_.size();

    std::vector<double> nodepos(3, 0.0);
    // we need the original node positions
    // loop nodes in condition
    for (int i = 0; i < my_uncert_nnodes_; ++i)
    {
      // do only nodes in my row map
      int nlid = aledis_->NodeRowMap()->LID((my_uncert_nodeids_)[i]);
      if (nlid < 0)
        continue;
      DRT::Node* actnode = aledis_->lRowNode(nlid);

      for (int dim = 0; dim < 3; ++dim)
      {
        nodepos[dim] = actnode->X()[dim];
      }

      my_uncertain_nodes_org_pos_.insert(
          std::pair<int, std::vector<double> >((my_uncert_nodeids_)[i],
              nodepos));


    }
    LINALG::GatherAll(my_uncertain_nodes_org_pos_, aledis_->Comm());

    // check whether we have an ale point db condition on all nodes in uncertain Surface
    if((int)uncert_surface.at(0)->Nodes()->size()!=my_uncert_nnodes_)
      dserror("Number of point Dirichlet conditions does not match number of nodes in Uncertain Surface condition" );

  }
  // we do not use ALE DBC to get a spatially dependend mean for the rf
  else if (!uncertain_nodes_condition.size() && !use_mean_field_from_cond_)
  {
    my_uncert_nnodes_=uncert_surface.at(0)->Nodes()->size();
         //
    std::vector<double> nodepos(3, 0.0);
    for(int i = 0;i< my_uncert_nnodes_; ++i)
    {
      my_uncert_nodeids_.push_back(uncert_surface.at(0)->Nodes()->at(i));

      // do only nodes in my row map
      int nlid = aledis_->NodeRowMap()->LID((my_uncert_nodeids_)[i]);
      if (nlid < 0)
        continue;
      DRT::Node* actnode = aledis_->lRowNode(nlid);

      for (int dim = 0; dim < 3; ++dim)
      {
        nodepos[dim] = actnode->X()[dim];
      }
      my_uncertain_nodes_org_pos_.insert(
        std::pair<int, std::vector<double> >((my_uncert_nodeids_)[i],
            nodepos));

      // do not allow nodes to move orthogonal to normal
      // if this behaviour is desired  set only first component to one
      std::vector<int> temp_onoff (3,1);

      std::vector<double> temp_val(3, 0.0);
      std::vector<int> temp_curve(3, 0);
      temp_curve[0] = 0;
      std::vector<int> temp_funct(3, 0);
      temp_funct[0] = 0;

      my_uncertain_nodes_dbc_onoff_.insert(
          std::pair<int, std::vector<int> >((my_uncert_nodeids_)[i],
              temp_onoff));
      my_uncertain_nodes_dbc_val_.insert(
          std::pair<int, std::vector<double> >((my_uncert_nodeids_)[i],
              temp_val));
      my_uncertain_nodes_dbc_curve_.insert(
          std::pair<int, std::vector<int> >((my_uncert_nodeids_)[i],
              temp_curve));
      my_uncertain_nodes_dbc_funct_.insert(
          std::pair<int, std::vector<int> >((my_uncert_nodeids_)[i],
              temp_funct));
    }
    LINALG::GatherAll(my_uncertain_nodes_org_pos_, aledis_->Comm());
    LINALG::GatherAll(my_uncertain_nodes_dbc_val_, aledis_->Comm());
    LINALG::GatherAll(my_uncertain_nodes_dbc_onoff_, aledis_->Comm());
    LINALG::GatherAll(my_uncertain_nodes_dbc_curve_, aledis_->Comm());
    LINALG::GatherAll(my_uncertain_nodes_dbc_funct_, aledis_->Comm());
  }
  else
  {
    dserror("Found ale dbc's while use_mean_field_from_cond_ is off! Fix your input file ");
  }

  // check whether we have an ale point db condition on all nodes in uncertain Surface

  // setup adapter in order to be able to apply ale displacements the structure
  coupsa_ = Teuchos::rcp(new ADAPTER::Coupling());
  coupsa_->SetupCoupling(*discret_,
                         *aledis_,
                         *(discret_->NodeRowMap()),
                         *(aledis_->NodeRowMap()),
                         3);

  // store initial geometry of structure in dofrowmap layout
  org_geom_ = Teuchos::rcp(new Epetra_Vector(*(discret_->DofRowMap()), true));
  const int nummyrownode = (discret_->NodeRowMap())->NumMyElements();

   // loop over all nodes
  for (int i = 0; i < nummyrownode; i++)
  {
    // get current node
    DRT::Node* mynode = discret_->lRowNode(i);

    std::vector<int> globaldofs = discret_->Dof(0, mynode);

    // determine number of space dimensions
    const int numdim = DRT::Problem::Instance()->NDim();

    for (int j = 0; j < numdim; j++)
    {
      const int lid = org_geom_->Map().LID(globaldofs[j]);
      if (lid < 0)
        dserror("Proc %d: Cannot find gid=%d in Epetra_Vector",
            org_geom_->Comm().MyPID(), globaldofs[j]);
      (*org_geom_)[lid] = (mynode->X())[j];
    }
  }
  ComputeNormals();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::SetUpThickness(const unsigned int myseed,
    double para_cont_parameter, bool reuse_rf)
{
  if (!reuse_rf)
    CreateNewRealizationOfRandomField(myseed);

  ModifyGeometryBasedOnRF(myseed);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<UQ::RandomField> UQ::MCVarThicknessManager::CreateRandomField(
    int random_field_id, unsigned int myseed)
{
  const Teuchos::ParameterList& rfp =
      DRT::Problem::Instance()->RandomFieldParams(random_field_id);

  // before calling the constructor make a quick safety check whether this
  // random field was activated in
  // the input file or if the section contains only the default parameters
  bool active = DRT::INPUT::IntegralValue<int>(rfp, "ACTIVE");
  if (!active)
    dserror("Trying to setup random field that is not active");

  // call constructor based on type of random field
  INPAR::MLMC::CalcMethod calcm = DRT::INPUT::IntegralValue<
      INPAR::MLMC::CalcMethod>(rfp, "CALC_METHOD");
  Teuchos::RCP<RandomField> rf = Teuchos::null;

  switch (calcm)
  {
  case INPAR::MLMC::calc_m_fft:
    rf = Teuchos::rcp(new RandomFieldSpectral(myseed, discret_, rfp));
    break;
  case INPAR::MLMC::calc_m_cos:
    rf = Teuchos::rcp(new RandomFieldSpectral(myseed, discret_, rfp));
    break;
  case INPAR::MLMC::calc_m_fourier:
    rf = Teuchos::rcp(new RandomFieldFourier(myseed, discret_, rfp));
    break;
  default:
    dserror("Unknown simulation method for RF choose fft or cos or fourier");
    break;
  }
  return rf;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::AddPointDBCToAleDiscretization(
    const std::map<int, std::vector<double> >& ale_bc_nodes_val,
    const std::map<int, std::vector<int> >& ale_bc_nodes_onoff,
    const std::map<int, std::vector<int> >&ale_bc_nodes_curve,
    const std::map<int, std::vector<int> >& ale_bc_nodes_funct)
{
  std::map<int, std::vector<double> >::const_iterator iter;
  int id = 1;
  for (iter = ale_bc_nodes_val.begin(); iter != ale_bc_nodes_val.end(); iter++)
  {
    Teuchos::RCP<DRT::Condition> cond = Teuchos::rcp(
        new DRT::Condition(id++, DRT::Condition::PointDirichlet, false,
            DRT::Condition::Point));

    std::vector<int> onoff(3, 1);
    std::vector<int> t_curve(3, 0);

    cond->Add("Node Ids", iter->first);
    cond->Add("onoff", ale_bc_nodes_onoff.at(iter->first));
    cond->Add("val", iter->second);
    cond->Add("curve",ale_bc_nodes_curve.at(iter->first));
    cond->Add("funct",ale_bc_nodes_funct.at(iter->first));

    aledis_->SetCondition("Dirichlet", cond);
  }

  aledis_->FillComplete();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::DeletePointDBCConditionsFromAleDiscretization()
{
  std::vector<
      std::multimap<std::string, Teuchos::RCP<DRT::Condition> >::iterator> del;

  std::multimap<std::string, Teuchos::RCP<DRT::Condition> >::iterator conit;
  std::multimap<std::string, Teuchos::RCP<DRT::Condition> >& allcondn =
      aledis_->GetAllConditions();
  for (conit = allcondn.begin(); conit != allcondn.end(); conit++)
  {
    Teuchos::RCP<DRT::Condition> cond = conit->second;
    if (cond->Nodes()->size() == 1)
      del.push_back(conit);
  }

  for (unsigned i = 0; i < del.size(); i++)
  {
    conit = del[i];
    allcondn.erase(conit);
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::CreateNewRealizationOfRandomField(
    unsigned int myseed)
{
  randomfield_->CreateNewSample(myseed + 174368);
}



/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::ResetGeometry()
{
  const int numnode = (discret_->NodeColMap())->NumMyElements();

  //Create Vector which holds all col-displacements of processor
  Teuchos::RCP<Epetra_Vector> coldisp = Teuchos::rcp(
      new Epetra_Vector(*(discret_->DofColMap())));

  //Export row-displacments to col-displacements
  LINALG::Export(*org_geom_, *coldisp);

  const Epetra_Vector& gvector = *coldisp;

  // loop over all nodes
  for (int index = 0; index < numnode; ++index)
  {
    // get current node
    DRT::Node* mynode = discret_->lColNode(index);

    std::vector<int> globaldofs = discret_->Dof(0, mynode);
    std::vector<double> nvector(globaldofs.size());

    // determine number of space dimensions
    const int numdim = DRT::Problem::Instance()->NDim();

    for (int i = 0; i < numdim; ++i) {
      const int lid = gvector.Map().LID(globaldofs[i]);

      if (lid < 0)
        dserror("Proc %d: Cannot find gid=%d in Epetra_Vector",
            gvector.Comm().MyPID(), globaldofs[i]);
      nvector[i] += gvector[lid];
    }

    mynode->SetPos(nvector);
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::ComputeNewAleDBCFromRF()
{
  my_uncertain_nodes_delta_pos_.clear();
  // compute delta t
  for (int i = 0; i < my_uncert_nnodes_; ++i)
  {
    // do only nodes in my row map
    int nlid = discret_->NodeRowMap()->LID((my_uncert_nodeids_)[i]);
    if (nlid < 0)
      continue;
    // get org node position
    std::vector<double> nodepos(
        my_uncertain_nodes_org_pos_.at((my_uncert_nodeids_)[i]));

    double rf_thick = 0.0;
    if (use_mean_field_from_cond_)
    {
      // evaluate random field
      rf_thick = randomfield_->EvalFieldAtLocation(nodepos,
          my_uncertain_nodes_dbc_val_.at((my_uncert_nodeids_)[i])[0], 1.0,
          false, false);
    }
    else
    {
      rf_thick = randomfield_->EvalFieldAtLocation(nodepos, 1.0, false, false);
    }
    double delta_t;

    //
    if(start_rand_geo_above_bif_)
    {
      double z_rel= nodepos[2]-(z_pos_start_rf_);
      if(z_rel > transition_width_)
      {
        delta_t= rf_thick - initial_wall_thickness_;
      }
      else if (z_rel > 0.0 && z_rel< transition_width_)
      {
        delta_t=(rf_thick - initial_wall_thickness_)*z_rel/transition_width_;
      }
      else // (z_rel< 0.0):
      {
        delta_t=0;
      }
    }
    else
    {
      delta_t = rf_thick - initial_wall_thickness_;
    }

    std::vector<double> temp_delta_t(3, 0.0);
    // use normals instead
    // note that my_uncertain_nodes_normals_ were computed on structural dis
    // hence it is assumed that node IDs of both discretizations are matching
    temp_delta_t[0] = delta_t
        * my_uncertain_nodes_normals_.at((my_uncert_nodeids_)[i])[0];
    temp_delta_t[1] = delta_t
        * my_uncertain_nodes_normals_.at((my_uncert_nodeids_)[i])[1];
    temp_delta_t[2] = delta_t
        * my_uncertain_nodes_normals_.at((my_uncert_nodeids_)[i])[2];

    my_uncertain_nodes_delta_pos_.insert(
        std::pair<int, std::vector<double> >((my_uncert_nodeids_)[i],
            temp_delta_t));

  }

  // communicate delta_pos
  LINALG::GatherAll(my_uncertain_nodes_delta_pos_, aledis_->Comm());

}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::ModifyGeometryBasedOnRF(const unsigned int myseed)
{
  // stop ALE from writing output
  const Teuchos::ParameterList& alep =
      DRT::Problem::Instance()->AleDynamicParams();
  Teuchos::ParameterList ale_new_p(alep);

  ale_new_p.set("RESTARTEVRY", 0);
  ale_new_p.set("RESULTSEVRY", 0);

  // setup ale time integration
  Teuchos::RCP< ::ADAPTER::AleBaseAlgorithm> ale =
      Teuchos::rcp(new ::ADAPTER::AleBaseAlgorithm(ale_new_p, aledis_));
  Teuchos::RCP< ::ADAPTER::Ale> my_ale_timint = ale->AleField();
  //my_ale_timint->WriteAccessDiscretization()->Writer()->NewResultFile("ladida",100);
  //my_ale_timint->WriteAccessDiscretization()->Writer()->WriteMesh(0, 0.01);

  ComputeNewAleDBCFromRF();

  // add thickness to dbc vector substract initial thickness
  DeletePointDBCConditionsFromAleDiscretization();

  AddPointDBCToAleDiscretization(my_uncertain_nodes_delta_pos_,
       my_uncertain_nodes_dbc_onoff_, my_uncertain_nodes_dbc_curve_,
       my_uncertain_nodes_dbc_funct_);

  // since we potentially changed the DB conditions we have to rebuild
  my_ale_timint->SetupDBCMapEx(ALE::UTILS::MapExtractor::dbc_set_std,
      Teuchos::null, Teuchos::null);

  // check whether locsys condition is in place to move nodes in normal dir
//  std::vector<DRT::Condition*> locsysconditions(0);
//  my_ale_timint->Discretization()->GetCondition("Locsys", locsysconditions);
//  if (!locsysconditions.size())
//    dserror("no locsysy cond found on ale dis");

  // integrate ale problem
  const double t0 = Teuchos::Time::wallTime();
  my_ale_timint->Integrate();
  const double t1 = Teuchos::Time::wallTime();
  // plot some time measurements
  if(!discret_->Comm().MyPID())
  {
    IO::cout<<"\n=================Time  Measurement================"<<IO::endl;
    IO::cout<<"Solve ALE:\t"<<std::setprecision(4)<<t1-t0<<"\ts"<<IO::endl;
  }

  // transfer ale displacements to structure dis
  Teuchos::RCP<Epetra_Vector> structdisp = coupsa_->SlaveToMaster(
      my_ale_timint->WriteAccessDispnp());
  DRT::UTILS::UpdateMaterialConfigWithDispVector(discret_, structdisp);
  discret_->FillComplete(false, true, true);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double UQ::MCVarThicknessManager::EvalThicknessAtLocation(
    std::vector<double> myloc, double para_cont_parameter)
{
  double rf_thick = 0.0;

  // evaluate random field
  rf_thick = randomfield_->EvalFieldAtLocation(myloc, para_cont_parameter,
      false, false);

  if(use_mean_field_from_cond_)
  {
    dserror("Currently not implemented");
  }
  return rf_thick;

}
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void UQ::MCVarThicknessManager::ComputeNormals()
{
  ADAPTER::CouplingMortar mytemp;
  mytemp.SetupForUQAbuseNormalCalculation(discret_, discret_->Comm());
  mytemp.Interface()->EvaluateNodalNormals(my_uncertain_nodes_normals_);
  LINALG::GatherAll(my_uncertain_nodes_normals_, discret_->Comm());
}

#endif /* #ifdef HAVE_FFTW */
