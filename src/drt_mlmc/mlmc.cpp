/*!----------------------------------------------------------------------
\file  mlmc.cpp
\brief Class for performing Multi Level Monte Carlo (MLMC)analysis of structure


 <pre>
Maintainer: Jonas Biehler
            biehler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15276
</pre>
 *!----------------------------------------------------------------------*/
#ifdef HAVE_FFTW
#ifdef CCADISCRET
#include "../drt_adapter/adapter_structure.H"
#include <Teuchos_TimeMonitor.hpp>
#include "mlmc.H"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include "Epetra_SerialDenseMatrix.h"
#include "../global_full/global_inp_control.H"
#include "../drt_io/io_hdf.H"
#include "../drt_mat/material.H"
#include "../drt_mat/aaaneohooke_stopro.H"
#include "../drt_mat/matpar_bundle.H"
#include "randomfield.H"
#include "gen_randomfield.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_comm/comm_utils.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io.H"
#include "../linalg/linalg_utils.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"

#include "../drt_comm/comm_utils.H"
//for file output
#include <fstream>


/*----------------------------------------------------------------------*/
/* standard constructor */
STR::MLMC::MLMC(Teuchos::RCP<DRT::Discretization> dis,
                Teuchos::RCP<LINALG::Solver> solver,
                Teuchos::RCP<IO::DiscretizationWriter> output)
  : discret_(dis),
    solver_(solver),
    output_(output)
    //sti_(Teuchos::null)
{

  // get coarse and fine discretizations

    actdis_coarse_ = DRT::Problem::Instance(0)->Dis(genprob.numsf, 0);
    // set degrees of freedom in the discretization
    if (not actdis_coarse_->Filled()) actdis_coarse_->FillComplete();
  //int myrank = dis->Comm().MyPID();

  reset_out_count_=0;

    filename_ = DRT::Problem::Instance()->OutputControlFile()->FileName();

  // input parameters structural dynamics
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  // get number of timesteps
  tsteps_ = sdyn.get<int>("NUMSTEP");
  // input parameters multi level monte carlo
  const Teuchos::ParameterList& mlmcp = DRT::Problem::Instance()->MultiLevelMonteCarloParams();

  // Get number of Newton iterations
  num_newton_it_ = mlmcp.get<int>("ITENODEINELE");
  // Get convergence tolerance
  convtol_    = mlmcp.get<double>("CONVTOL");

  // calculate difference to lower level yes/no
  calc_diff_ = DRT::INPUT::IntegralValue<int>(mlmcp ,"DIFF_TO_LOWER_LEVEL");

  // prolongate results yes/no
  prolongate_res_ = DRT::INPUT::IntegralValue<int>(mlmcp ,"PROLONGATERES");

  // get starting random seed
  start_random_seed_ = mlmcp.get<int>("INITRANDOMSEED");

  // get name of lower level outputfiles
  filename_lower_level_ =  mlmcp.get<std::string>("OUTPUT_FILE_OF_LOWER_LEVEL");

  // get numerb of current level
  num_level_ = mlmcp.get<int>("LEVELNUMBER");

  //write statistics every write_stat_ steps
  write_stats_ = mlmcp.get<int>("WRITESTATS");

  // In element critirion xsi_i < 1 + eps  eps = MLMCINELETOL
  InEleRange_ = 1.0 + 10e-3;
  //ReadInParameters();


  // controlling parameter
  start_run_ = mlmcp.get<int>("START_RUN");
  int numruns = mlmcp.get<int>("NUMRUNS");
  Teuchos::RCP<DRT::Problem> problem = DRT::Problem::Instance();
  Teuchos::RCP<Epetra_Comm> lcomm = problem->GetNPGroup()->LocalComm();
  int NNestedGroups = problem->GetNPGroup()->NumGroups();
  int i = problem->GetNPGroup()->GroupId();

  numruns_pergroup_= int(ceil(numruns/NNestedGroups));
  start_run_  += (i)*numruns_pergroup_;

  numb_run_ =  start_run_;//+numruns_pergroup_;     // counter of how many runs were made monte carlo

  //int mygroup =i;

  //local_numruns_=numruns_pergroup;



  // meshfiel name to be written to controlfile in prolongated results
  std::stringstream meshfilename_helper1;
  string meshfilename_helper2;
  meshfilename_helper1 << filename_ << "_prolongated_run_" << start_run_ ;
  meshfilename_helper2 = meshfilename_helper1.str();
  // strip path from name
  string::size_type pos = meshfilename_helper2.find_last_of('/');
      if (pos==string::npos)
              meshfilename_ = meshfilename_helper2;
       else
              meshfilename_ = meshfilename_helper2.substr(pos+1);

  //init stuff that is only needed when we want to prolongate the results to a finer mesh,
  // and hence have a fine discretization
  if(prolongate_res_ )
  {
    // Get finest Grid problem instance

    actdis_fine_ = DRT::Problem::Instance(1)->Dis(genprob.numsf, 0);
    // set degrees of freedom in the discretization
    if (not actdis_fine_->Filled()) actdis_fine_->FillComplete();
    // Get coarse Grid problem instance
    cout << "before def actidis fine " << endl;
    output_control_fine_ = rcp(new IO::OutputControl(actdis_fine_->Comm(), "structure", "Polynomial", filename_, filename_, 3, 0, 20));
    output_fine_ = Teuchos::rcp(new IO::DiscretizationWriter(actdis_fine_,output_control_fine_));

    // init vectors to store mean stresses and displacements
    mean_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    mean_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    mean_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    // init vectors to store standard dev of  stresses and displacements
    var_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    var_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    var_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    // init vectors to calc standard dev of  stresses and displacements
    delta_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    delta_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    delta_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    m2_var_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    m2_var_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    m2_var_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    m2_helper_var_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    m2_helper_var_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    m2_helper_var_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));


    // same vectors for difference between two levels
    // init vectors to store mean stresses and displacements
    diff_mean_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    diff_mean_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    diff_mean_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    // init vectors to store standard dev of  stresses and displacements
    diff_var_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    diff_var_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    diff_var_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    // init vectors to calc standard dev of  stresses and displacements
    diff_delta_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    diff_delta_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    diff_delta_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    diff_m2_var_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    diff_m2_var_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    diff_m2_var_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    diff_m2_helper_var_disp_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    diff_m2_helper_var_stress_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    diff_m2_helper_var_strain_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));


    disp_lower_level_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));
    stress_lower_level_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
    strain_lower_level_ = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
  }
}


/*----------------------------------------------------------------------*/
/* analyse */
void STR::MLMC::Integrate()
{
  // init vector to store displacemnet

  //Teuchos::RCP<Epetra_Vector> dis_coarse = rcp(new Epetra_Vector(*actdis_coarse_->DofRowMap(),true));
  Teuchos::RCP<const Epetra_Vector> dis_coarse = Teuchos::null;


  const int myrank = discret_->Comm().MyPID();

  //measure time
  Epetra_Time timer(discret_->Comm());
  //double t1 = timer.ElapsedTime();
  const Teuchos::ParameterList& mlmcp = DRT::Problem::Instance()->MultiLevelMonteCarloParams();
  //int numruns = mlmcp.get<int>("NUMRUNS")+start_run_;
  // nested par hack
  int numruns =numruns_pergroup_+start_run_;


  // get initial random seed from inputfile
  unsigned int random_seed= mlmcp.get<int>("INITRANDOMSEED");
  do
  {
    //cout << "numbrun_ " << numb_run_ << endl;
    if (myrank == 0)
    {
      cout << GREEN_LIGHT "================================================================================" << endl;
      cout << "                            MULTILEVEL MONTE CARLO                              " << endl;
      cout << "                              RUN: " << numb_run_ << "  of  " << numruns  << endl;
      cout << "================================================================================" END_COLOR << endl;
    }
    //double t1 = timer.ElapsedTime();
    if (myrank == 0)
      cout << RED_LIGHT " PRESTRESS NOT RESET " END_COLOR << endl;
    // ResetPrestress();
    SetupStochMat((random_seed+(unsigned int)numb_run_));
    discret_->Comm().Barrier();

    //double t2 = timer.ElapsedTime();
    output_->NewResultFile(filename_,(numb_run_));
    //cout << "ATTENTION NO NEW RESULTFILE CREATED" << endl;

    // get input lists
    const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();

    // major switch to different time integrators
    switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn,"DYNAMICTYP"))
    {
      case INPAR::STR::dyna_gen_alfa:
      case INPAR::STR::dyna_gen_alfa_statics:
      case INPAR::STR::dyna_statics:
      case INPAR::STR::dyna_genalpha:
      case INPAR::STR::dyna_onesteptheta:
      case INPAR::STR::dyna_gemm:
      case INPAR::STR::dyna_expleuler:
      case INPAR::STR::dyna_centrdiff:
      case INPAR::STR::dyna_ab2:
      case INPAR::STR::dyna_euma:
      case INPAR::STR::dyna_euimsto:
      {
        // instead of calling dyn_nlnstructural_drt(); here build the adapter here so that we have acces to the results
        // What follows is basicaly a copy of whats usually in dyn_nlnstructural_drt();
        // create an adapterbase and adapter
        ADAPTER::StructureBaseAlgorithm adapterbase(DRT::Problem::Instance()->StructuralDynamicParams());
        ADAPTER::Structure& structadaptor = const_cast<ADAPTER::Structure&>(adapterbase.StructureField());

        // do restart
        if (genprob.restart)
        {
          structadaptor.ReadRestart(genprob.restart);
        }
        structadaptor.Integrate();
        //const Epetra_Vector* disp = structadaptor.Dispn().get();
        //cout << "test displaement " << *disp << endl;
        dis_coarse= rcp(new const Epetra_Vector(*(structadaptor.Dispn())));
        // test results
        DRT::Problem::Instance()->AddFieldTest(structadaptor.CreateFieldTest());
        DRT::Problem::Instance()->TestAll(structadaptor.DofRowMap()->Comm());

        // print monitoring of time consumption
        //Teuchos::TimeMonitor::summarize();

      #ifdef TRILINOS_DEV
           Teuchos::RCP<const Teuchos::Comm<int> > TeuchosComm = COMM_UTILS::toTeuchosComm<int>(structadaptor.DofRowMap()->Comm());
           Teuchos::TimeMonitor::summarize(TeuchosComm.ptr(), std::cout, false, true, false);
      #else
           Teuchos::TimeMonitor::summarize(std::cout, false, true, false);
      #endif

        // time to go home...
      }
        break;
      default:
        dserror("unknown time integration scheme '%s'", sdyn.get<std::string>("DYNAMICTYP").c_str());
    }
    EvalDisAtNodes(dis_coarse);
    //discret_->Comm().Barrier();
    if (numb_run_-start_run_== 0 &&  prolongate_res_)
    {
     // not parallel
      SetupProlongatorParallel();
    }
    if (calc_diff_)
    {
      ReadResultsFromLowerLevel();
    }
    if ( prolongate_res_)
    {
      ProlongateResults();
      // write statoutput evey now and then
      if(numb_run_% write_stats_ == 0)
        WriteStatOutput();
    }

    numb_run_++;
    } while (numb_run_< numruns);
  if( prolongate_res_ )
  {
    WriteStatOutput();
  }

  //ReadResultsFromLowerLevel(12);
  return;
}
//---------------------------------------------------------------------------------------------
void STR::MLMC::SetupProlongatorParallel()
{

  // number of outliers
  int num_outliers = 0;
  const Epetra_Map* rmap_disp = NULL;
  const Epetra_Map* dmap_disp = NULL;
  const Epetra_Map* rmap_stress = NULL;
  const Epetra_Map* dmap_stress = NULL;
  rmap_disp= (actdis_fine_->DofRowMap());
  dmap_disp=(actdis_coarse_->DofRowMap());
  //maps for stress prolongator
  rmap_stress = (actdis_fine_->NodeRowMap());
  dmap_stress = (actdis_coarse_->NodeRowMap());
  int bg_ele_id;
  // store location of node
  double xsi[3] = {0.0, 0.0,0.0};
  prolongator_disp_crs_ = rcp(new Epetra_FECrsMatrix(::Copy,*rmap_disp,*dmap_disp,8,false));
  prolongator_stress_crs_ = rcp(new Epetra_FECrsMatrix(::Copy,*rmap_stress,*dmap_stress,8,false));

  //loop over nodes of fine discretization on this proc
  Teuchos::RCP<Epetra_Vector> node_vector = Teuchos::rcp(new Epetra_Vector(*(actdis_fine_->NodeRowMap()),true));
  // loop over dofs
  for (int i=0; i< actdis_fine_->NumMyRowNodes(); i++  )
  //  for (int i=0; i<2; i++  )
  {
    //cout << "in ele looop " << endl;
    // Get node
    DRT::Node* node = actdis_fine_->lRowNode(i);
    //cout << "node " << node->Id() << endl;
    // Get background element and local coordinates
    num_outliers += FindBackgroundElement(*node, actdis_coarse_, &bg_ele_id, xsi);
    //cout << "in ele looop 2" << endl;
    // Get element
    DRT::Element* bg_ele = actdis_coarse_->gElement(bg_ele_id);

    Epetra_SerialDenseVector shape_fcts(bg_ele->NumNode());
    DRT::UTILS::shape_function_3D(shape_fcts,xsi[0],xsi[1],xsi[2],bg_ele->Shape());


    int* rows = NULL;
    int* cols = NULL;
    double* values = NULL;
    int numColumns = bg_ele->NumNode();
    // insert rows dof wise
    int numRows   = 1;


    cols = new int[numColumns];
    rows = new int[numRows];
    values = new double[numColumns];


    // fill prolongators
    //
    // DIM = 3
    for (int j = 0; j<3 ; j++)
    {
      int index = 0;
      for (int k=0; k< bg_ele->NumNode(); k ++)
      {
        // store global indices in cols
       // (*rcp_Array)[index]=dmap_disp->GID((bg_ele->Nodes()[k]->Id()*3)+j);
        cols[index]=dmap_disp->GID((bg_ele->Nodes()[k]->Id()*3)+j);
        rows[0]=i*3+j;
        values[index]=shape_fcts[k];
        index++;
      }
      int err=  prolongator_disp_crs_->InsertGlobalValues(1,rows,8,cols,values,Epetra_FECrsMatrix::COLUMN_MAJOR);
      if (err != 0)
      {
        dserror("Could not insert global values");
      }

    } // loop j
    //cout << "Setup Prolongator LINE  "<< __LINE__ << " myrank  " << myrank << endl;
    // stress prolongator
    for (int k=0; k< bg_ele->NumNode(); k ++)
    {
      // store global indices in cols
      cols[k]=dmap_stress->GID(bg_ele->Nodes()[k]->Id());
      values[k]=shape_fcts[k];
    }
    rows[0]=i;
    //cout << "row i  " << rows[0] << endl;
    int err=  prolongator_stress_crs_->InsertGlobalValues(1,rows,8,cols,values,Epetra_FECrsMatrix::COLUMN_MAJOR);
    //cout << "errror   " << err << endl;
    if (err != 0)
    {
      dserror("Could not insert global values");
    }
    //cout << "Setup Prolongator LINE  "<< __LINE__ << " myrank  " << myrank << endl;

    delete [] cols;
    delete [] rows;
    delete [] values;

    rows = NULL;
    cols = NULL;
    values = NULL;

  } // End of loop over nodes of fine discretzation

  // Assembly
  prolongator_disp_crs_->GlobalAssemble(*dmap_disp,*rmap_disp,true);
  prolongator_stress_crs_->GlobalAssemble(*dmap_stress,*rmap_stress,true);
  cout << "################################################### " << endl;
  cout << "   SUCCESSFULLY INITIALIZED  PROLONGATOR" << endl;
  cout <<  num_outliers << " Nodes do not lie within a background element " << endl;
  cout << "################################################### " << endl;
}
//---------------------------------------------------------------------------------------------
void STR::MLMC::SetupProlongator()
{
  // This functions calculates the prolongtators for the displacement and the nodal stresses
  cout << "debugging in  LINE "<<  __LINE__<< endl;
  // 3D Problem
  int num_columns_prolongator_disp = actdis_fine_->NumGlobalNodes()*3;
  int num_columns_prolongator_stress = actdis_fine_->NumGlobalNodes();

  double xsi[3] = {0.0, 0.0,0.0};
  cout << "debugging in  LINE "<<  __LINE__<< endl;
  // loop over nodes of fine dis
  int num_nodes;
  int bg_ele_id;
  num_nodes = actdis_fine_->NumGlobalNodes();
  cout << "debugging in  LINE "<<  __LINE__<< endl;
  // init prolongators
  cout << "num_columns proongator "  << num_columns_prolongator_disp << endl;
  cout << "num_columns proongator stress  "  << num_columns_prolongator_stress << endl;
  prolongator_disp_ = rcp(new Epetra_MultiVector(*actdis_coarse_->DofRowMap(),num_columns_prolongator_disp,true));
  prolongator_stress_ = rcp(new Epetra_MultiVector(*actdis_coarse_->NodeRowMap(),num_columns_prolongator_stress,true));
  cout << "debugging in  LINE "<<  __LINE__<< endl;
  for (int i = 0; i < num_nodes ; i++)
  {

    // Get node
    DRT::Node* node = actdis_fine_->gNode(i);

    // Get background element and local coordinates
    FindBackgroundElement(*node, actdis_coarse_, &bg_ele_id, xsi);
    // Get element
    DRT::Element* bg_ele = actdis_coarse_->gElement(bg_ele_id);

    Epetra_SerialDenseVector shape_fcts(bg_ele->NumNode());
    DRT::UTILS::shape_function_3D(shape_fcts,xsi[0],xsi[1],xsi[2],bg_ele->Shape());

    // fill prolongators add values to prolongator
    for (int j = 0; j<3 ; j++)
    {

      for (int k=0; k< bg_ele->NumNode(); k ++)
      {
        //prolongator_disp_ is dof based
        (*prolongator_disp_)[(i*3)+j][(bg_ele->Nodes()[k]->Id()*3)+j]= shape_fcts[k];
        // prolongator stress_ is node based
        (*prolongator_stress_)[i][(bg_ele->Nodes()[k]->Id())]= shape_fcts[k];
      }
    }
  } // End of loop over nodes of fine discretzation

}
//---------------------------------------------------------------------------------------------
void STR::MLMC::ProlongateResults()
{
  cout << "Prolongating Resuts " << endl;
  // To avoid messing with the timeintegration we read in the results of the coarse discretization here
  // Get coarse Grid problem instance
  std::stringstream name;
  string filename_helper;
  name << filename_ << "_run_"<< numb_run_;
  filename_helper = name.str();

  RCP<IO::InputControl> inputcontrol = rcp(new IO::InputControl(filename_helper, false));

  IO::DiscretizationReader input_coarse(actdis_coarse_, inputcontrol,tsteps_);
  // Vector for displacements
  Teuchos::RCP<Epetra_Vector> dis_coarse = rcp(new Epetra_Vector(*actdis_coarse_->DofRowMap(),true));

  // read in displacements
  input_coarse.ReadVector(dis_coarse, "displacement");

  Teuchos::RCP<Epetra_MultiVector> dis_fine = Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->DofRowMap()),1,true));

  // Try new and shiny prolongator based on crs Matrix
  int error = prolongator_disp_crs_->Multiply(false,*dis_coarse,*dis_fine);
  if(error!=0)
  {
    dserror("stuff went wrong");
  }


  // create new resultfile for prolongated results
  std::stringstream name_prolong;
  string filename_helper_prolong;
  name_prolong << filename_ << "_prolongated";
  filename_helper_prolong = name_prolong.str();

  output_fine_->NewResultFile(filename_helper_prolong ,(numb_run_));


  output_fine_->WriteMesh(1, 0.01, meshfilename_);
  // exception for first run
  if(numb_run_==start_run_)
    output_fine_->WriteMesh(1, 0.01);

  output_fine_->NewStep( 1, 0.01);


  // Write interpolated displacement to file
  cout << " before displacement  " << endl;
  output_fine_->WriteVector("displacement", dis_fine, output_fine_->dofvector);
  Teuchos::RCP<Epetra_Vector> dis_fine_single = rcp(new Epetra_Vector(*actdis_fine_->DofRowMap(),true));

  // transfer to Epetra_Vector
  for( int i = 0;i< dis_fine->MyLength(); i++)
  {
   (*dis_fine_single)[i]= (*dis_fine)[0][i];
  }

  //#####################################################################################
  //
  //                  prolongate stresses based on interpolated displacement field
  //
  //#####################################################################################

  // set up parameters for Evaluation
  double timen         = 0.9;  // params_.get<double>("total time"             ,0.0);
  double dt            = 0.1; //params_.get<double>("delta time"             ,0.01);
  double alphaf        = 0.459; // params_.get<double>("alpha f"                ,0.459);
  INPAR::STR::StressType iostress =INPAR::STR::stress_2pk; //stress_none;
  INPAR::STR::StrainType iostrain= INPAR::STR::strain_gl; // strain_none;
  RCP<Epetra_Vector>    zeros_ = rcp(new Epetra_Vector(*actdis_fine_->DofRowMap(),true));
  RCP<Epetra_Vector>    dis_ = dis_fine_single;
  RCP<Epetra_Vector>    vel_ = rcp(new Epetra_Vector(*actdis_fine_->DofRowMap(),true));
  // create the parameters for the discretization
  ParameterList p;
  // action for elements

  p.set("action","calc_struct_stress");
  // other parameters that might be needed by the elements
  p.set("total time",timen);
  p.set("delta time",dt);
  p.set("alpha f",alphaf);

  Teuchos::RCP<std::vector<char> > stress = Teuchos::rcp(new std::vector<char>());
  Teuchos::RCP<std::vector<char> > strain = Teuchos::rcp(new std::vector<char>());
  // plastic strains need to be init as well
  Teuchos::RCP<std::vector<char> > plstrain = Teuchos::rcp(new std::vector<char>());
  p.set("stress", stress);
  p.set("plstrain",plstrain);
  //

  p.set<int>("iostress", iostress);
  p.set("strain", strain);
  p.set<int>("iostrain", iostrain);
  // set vector values needed by elements
  p.set<double>("random test",5.0);
  actdis_fine_->ClearState();
  actdis_fine_->SetState("residual displacement",zeros_);
  actdis_fine_->SetState("displacement",dis_);
  actdis_fine_->SetState("velocity",vel_);

  // Evaluate Stresses based on interpolated displacements
  //actdis_fine_->Evaluate(p,null,null,null,null,null);
  //actdis_fine_->ClearState();
  // Write to file
  // interpolated stresses from disp field
  //output_fine_->WriteVector("gauss_2PK_stresses_xyz",*stress,*(actdis_fine_->ElementRowMap()));




  //#####################################################################################
  //
  //                  prolongate stresses based on interpolated nodal stress field
  //
  //#####################################################################################

  // use same parameter list as above
  RCP<Epetra_Vector>    zeros_coarse = rcp(new Epetra_Vector(*actdis_coarse_->DofRowMap(),true));
  //RCP<Epetra_Vector>    dis_ = dis_fine_single;
  RCP<Epetra_Vector>    vel_coarse = rcp(new Epetra_Vector(*actdis_coarse_->DofRowMap(),true));
  actdis_coarse_->ClearState();
  actdis_coarse_->SetState("residual displacement",zeros_coarse);
  actdis_coarse_->SetState("displacement",dis_coarse);
  actdis_coarse_->SetState("velocity",vel_coarse);
  // Alrigth lets get the nodal stresses
  p.set("action","calc_global_gpstresses_map");

  const RCP<map<int,RCP<Epetra_SerialDenseMatrix> > > gpstressmap = rcp(new std::map<int, RCP<Epetra_SerialDenseMatrix> >);
  p.set("gpstressmap", gpstressmap);

  const RCP<map<int,RCP<Epetra_SerialDenseMatrix> > > gpstrainmap = rcp(new std::map<int, RCP<Epetra_SerialDenseMatrix> >);
  p.set("gpstrainmap", gpstrainmap);

  actdis_coarse_->Evaluate(p,null,null,null,null,null);

  // st action to calc poststresse
  p.set("action","postprocess_stress");
  // Multivector to store poststresses
  RCP<Epetra_MultiVector> poststress =  Teuchos::rcp(new Epetra_MultiVector(*(actdis_coarse_->NodeRowMap()),6,true));
  // for fine diskretization as well
  RCP<Epetra_MultiVector> poststress_fine =  Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));

  p.set("poststress", poststress);
  p.set("stresstype","ndxyz");
  // cout << " Debugging   LINE   " << __LINE__ << endl;
  actdis_coarse_->ClearState();
  actdis_coarse_->Evaluate(p,null,null,null,null,null);
  actdis_coarse_->ClearState();

  // again for strains
  p.set("action","postprocess_stress");
  // Multivector to store poststrains
  RCP<Epetra_MultiVector> poststrain =  Teuchos::rcp(new Epetra_MultiVector(*(actdis_coarse_->NodeRowMap()),6,true));
  // for fine diskretization as well
  RCP<Epetra_MultiVector> poststrain_fine =  Teuchos::rcp(new Epetra_MultiVector(*(actdis_fine_->NodeRowMap()),6,true));
  p.set("poststress", poststrain);
  p.set("gpstressmap", gpstrainmap);
  p.set("stresstype","ndxyz");
  // cout << " Debugging   LINE   " << __LINE__ << endl;
  actdis_coarse_->ClearState();
  actdis_coarse_->Evaluate(p,null,null,null,null,null);
  actdis_coarse_->ClearState();


  // try new and shiny crs prolongator
  int error2 = prolongator_stress_crs_->Multiply(false,*poststress,*poststress_fine);
  if(error2!=0)
  {
    dserror("stuff went wrong");
  }
  // strains as well
  int error3 = prolongator_stress_crs_->Multiply(false,*poststrain,*poststrain_fine);
  if(error3!=0)
  {
    dserror("stuff went wrong");
  }

  cout << " before prolongated gauss 2pk stresses " << endl;
  output_fine_->WriteVector("prolongated_gauss_2PK_stresses_xyz", poststress_fine, output_fine_->nodevector);
  output_fine_->WriteVector("prolongated_gauss_GL_strains_xyz", poststrain_fine, output_fine_->nodevector);
  // do some statistics
  if(calc_diff_)
  {
    CalcDifferenceToLowerLevel(poststress_fine,poststrain_fine, dis_fine);
    // Write Difference between two Discretizations to File
    output_fine_->WriteVector("diff_to_ll_displacement", disp_lower_level_, output_fine_->dofvector);
    output_fine_->WriteVector("diff_to_ll_prolongated_gauss_2PK_stresses_xyz", stress_lower_level_, output_fine_->nodevector);
    output_fine_->WriteVector("diff_to_ll_prolongated_gauss_GL_strains_xyz", strain_lower_level_, output_fine_->nodevector);
  }
  CalcStatStressDisp(poststress_fine,poststrain_fine, dis_fine);
  // write some output to another file
  HelperFunctionOutputTube(poststress_fine,poststrain_fine,dis_fine);
}
//-----------------------------------------------------------------------------------
int STR::MLMC::FindBackgroundElement(DRT::Node node, Teuchos::RCP<DRT::Discretization> background_dis, int* bg_ele_id, double* xsi)
{
  bool outlier = false;
  Teuchos::RCP<Epetra_Vector> element_vector = Teuchos::rcp(new Epetra_Vector(*(background_dis->ElementRowMap()),true));
  // loop over discretization

  double pseudo_distance = 0.0;
  double min_pseudo_distance = 2.0;
  int back_ground_ele_id = 0;
  double background_xsi[3] = {0,0,0};
  //for (int i=0; i< 2 && min_pseudo_distance > InEleRange_ ; i++  )
  for (int i=0; i< element_vector->MyLength() && min_pseudo_distance > InEleRange_ ; i++  )
  {
    int globel_ele_id= background_dis->ElementRowMap()->GID(i);
    DRT::Element* ele = background_dis->gElement(globel_ele_id);
    //inEle = CheckIfNodeInElement(node, *ele);
    pseudo_distance = CheckIfNodeInElement(node, *ele, xsi);
    //cout << " pseudo_distance  " << pseudo_distance  << endl;
    // check if node is in Element
    if (pseudo_distance < min_pseudo_distance)
    {
      min_pseudo_distance = pseudo_distance;
      back_ground_ele_id = globel_ele_id;
      background_xsi[0]=xsi[0];
      background_xsi[1]=xsi[1];
      background_xsi[2]=xsi[2];
    }
  } // end of loop over elements
  // Debug
  if(min_pseudo_distance < InEleRange_)
  {
    //cout << "found background element Ele ID is " <<  back_ground_ele_id << endl;
    //cout << "Local Coordinates are "<< "xsi_0 " << xsi[0] << " xsi_1 " << xsi[1] << " xsi_2 " << xsi[2] << endl;

  }
  else
  {
   //cout << "did not find background element, closest element is: Ele ID: " << back_ground_ele_id << endl;
   //cout << "Local Coordinates are "<< "xsi_0 " << background_xsi[0] << " xsi_1 " << background_xsi[1] << " xsi_2 " << background_xsi[2] << endl;
   // dserror("stop right here");
   // write closest element xsi* into  pointer
    xsi[0]=background_xsi[0];
    xsi[1]=background_xsi[1];
    xsi[2]=background_xsi[2];
    outlier =true;
  }
  *bg_ele_id = back_ground_ele_id;
  return outlier;


}

//-----------------------------------------------------------------------------------
double STR::MLMC::CheckIfNodeInElement(DRT::Node& node, DRT::Element& ele, double* xsi)
{
  // init speudo distance, which is essentially largest values of xsi[i]
  double pseudo_distance = 0.0;
  //local/element coordinates
  xsi[0] = xsi[1] = xsi[2] = 0.0 ;
  // Replace later with problem dimension or element type
  //if (Dim()==3)

  if (ele.Shape()==DRT::Element::hex8)
    {
      // function f (vector-valued)
      double f[3] = {0.0, 0.0, 0.0};
      LINALG::Matrix<3,1> b;
      // gradient of f (df/dxsi[0], df/dxsi[1], df/dxsi[2])
      LINALG::Matrix<3,3> df;
      //convergeence check
      double conv = 0.0;

      for (int k=0;k<num_newton_it_;++k)
        {
          EvaluateF(f,node,ele,xsi);
          conv = sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
          //cout << "Iteration " << k << ": -> |f|=" << conv << endl;
          if (conv <= convtol_) break;

          EvaluateGradF(df,node,ele,xsi);

          // solve dxsi = - inv(df) * f
          //cout << "herer "<< endl;
          df.Invert();
          //cout << "after " << endl;
          //cout << "xsi_0 " << xsi[0] << "xsi_1 " << xsi[1] << "xsi_2 " << xsi[2] << endl;

          // update xsi
          xsi[0] += -df(0,0)*f[0] - df(1,0)*f[1] - df(2,0)*f[2];
          xsi[1] += -df(0,1)*f[0] - df(1,1)*f[1] - df(2,1)*f[2];
          xsi[2] += -df(0,2)*f[0] - df(1,2)*f[1] - df(2,2)*f[2];
          //cout << "iteration " << k<< "xsi: " << xsi[0] <<" " << xsi[1] << " "<<  xsi[2]<<endl ;
        }

      // Newton iteration unconverged
      if (conv > convtol_)
      {
        dserror("ERROR: CheckIfNodeInElement: Newton unconverged for NodeID %i "
                   "and ElementID %i", node.Id(), ele.Id());
      }
      // Newton iteration converged
      // find largest value of xsi[i] and return as pseudo_distance
      for (int i = 0; i < 3; i++)
      {
        if (fabs(xsi[i]) > pseudo_distance)
        {
          pseudo_distance = fabs(xsi[i]);
        }
      }
      return pseudo_distance;

    }
    else
    {
      dserror("CheckIfNodeInElement only implememted for hex8 Elements");
      return pseudo_distance;
    }
}
//-----------------------------------------------------------------------------------
bool STR::MLMC::EvaluateF(double* f,DRT::Node& node, DRT::Element& ele,const double* xsi)
{

  LINALG::Matrix<8,1> funct ;
  //DRT::Element::DiscretizationType distype = ele.DiscretizationType;
  const DRT::Element::DiscretizationType distype = DRT::Element::hex8;
  DRT::UTILS::shape_function_3D(funct, xsi[0], xsi[1], xsi[2],distype);

  //LINALG::Matrix<ele.NumNode(),ele->numdim> xrefe;  // material coord. of element
  LINALG::Matrix<8,3> xrefe;  // material coord. of element
    for (int i=0; i<ele.NumNode(); ++i){
      const double* x =ele.Nodes()[i]->X();
      xrefe(i,0) = x[0];
      xrefe(i,1) = x[1];
      xrefe(i,2) = x[2];
    }
    // Calc Difference in Location
    LINALG::Matrix<1,3> point;
    point.MultiplyTN(funct, xrefe);
    f[0]=point(0,0)-node.X()[0];
    f[1]=point(0,1)-node.X()[1];
    f[2]=point(0,2)-node.X()[2];
    return true;
}


//-----------------------------------------------------------------------------------
bool STR::MLMC::EvaluateGradF(LINALG::Matrix<3,3>& fgrad,DRT::Node& node, DRT::Element& ele,const double* xsi)
{
  //static const int iel = DRT::UTILS::DisTypeToNumNodePerEle<distype>::numNodePerElement;
  LINALG::Matrix<3,8> deriv1;
  const DRT::Element::DiscretizationType distype = DRT::Element::hex8;

  DRT::UTILS::shape_function_3D_deriv1(deriv1 ,xsi[0], xsi[1], xsi[2],distype);

  LINALG::Matrix<8,3> xrefe;  // material coord. of element
  int NUMNOD_SOH8 = 8;
  for (int i=0; i<NUMNOD_SOH8; ++i)
  {
    const double* x =ele.Nodes()[i]->X();
    xrefe(i,0) = x[0];
    xrefe(i,1) = x[1];
    xrefe(i,2) = x[2];
  }

  fgrad.MultiplyNN(deriv1,xrefe);

  return true;
}
// Read in Stresses and Displacements from corresponding Run on lower Level
void STR::MLMC::ReadResultsFromLowerLevel()
{
  std::stringstream name_helper;
  // assamble filename and pathfor cluster jobs
  //name_helper << "../../level"<<num_level_-1<< "/START_RUN_"<< start_run_ <<"/"<<filename_lower_level_<<"_prolongated"<< "_run_" << numb_run_ ;

  name_helper << filename_lower_level_ <<"_prolongated"<< "_run_" << numb_run_ ;
  string name_ll = name_helper.str();
  // check if bool should be true or false
  RCP<IO::InputControl> inputcontrol = rcp(new IO::InputControl(name_ll, false));
  IO::DiscretizationReader input_coarse(actdis_fine_, inputcontrol,1);

  // read in displacements
  input_coarse.ReadMultiVector(disp_lower_level_, "displacement");
  // read in prolongated GP Stresses
  input_coarse.ReadMultiVector(stress_lower_level_, "prolongated_gauss_2PK_stresses_xyz");
  // read in prolongated GP Strain
   input_coarse.ReadMultiVector(strain_lower_level_, "prolongated_gauss_GL_strains_xyz");
}

void STR::MLMC::CalcDifferenceToLowerLevel(RCP< Epetra_MultiVector> stress, RCP< Epetra_MultiVector> strain, RCP<Epetra_MultiVector> disp)
{
  // store difference not in new vectors to save memory
  disp_lower_level_->Update(1.0,*disp,-1.0);
  stress_lower_level_->Update(1.0,*stress,-1.0);
  strain_lower_level_->Update(1.0,*strain,-1.0);

}


// Setup Material Parameters in each element based on Random Field
void STR::MLMC::SetupStochMat(unsigned int random_seed)
{
  // Variables for Random field
  double sigma =0.0 , corrlength = 0.0,beta_mean = 0.0;
  double youngs = 0.0, youngs_mean = 0.0;
  // element center
  vector<double> ele_c_location;

  // flag have init stochmat??
  int stochmat_flag=0;
  // Get parameters from stochastic matlaw
  const int myrank = discret_->Comm().MyPID();

  // loop all materials in problem
  const map<int,RCP<MAT::PAR::Material> >& mats = *DRT::Problem::Instance()->Materials()->Map();
  if (myrank == 0) printf("No. material laws considered : %d\n",(int) mats.size());
  map<int,RCP<MAT::PAR::Material> >::const_iterator curr;
  for (curr=mats.begin(); curr != mats.end(); curr++)
  {
    const RCP<MAT::PAR::Material> actmat = curr->second;
    switch(actmat->Type())
    {
       case INPAR::MAT::m_aaaneohooke_stopro:
       {
         stochmat_flag=1;
         MAT::PAR::AAAneohooke_stopro* params = dynamic_cast<MAT::PAR::AAAneohooke_stopro*>(actmat->Parameter());
         if (!params) dserror("Cannot cast material parameters");
         // these matparams are not needed anymore
         sigma = params->sigma_0_;
         corrlength = params->corrlength_;
         beta_mean =params->beta_mean_;
         youngs_mean =params->youngs_mean_;
       }
       break;
      default:
      {
       cout << "MAT CURR " << actmat->Type() << "not stochastic" << endl;
      }

    }
  } // EOF loop over mats
  if (!stochmat_flag)// ignore unknown materials ?
          {
            //cout << "Mat Type   " << actmat->Type() << endl;
            dserror("No stochastic material supplied");
          }
  // get elements on proc use col map to init ghost elements as well
  Teuchos::RCP<Epetra_Vector> my_ele = rcp(new Epetra_Vector(*discret_->ElementColMap(),true));
  cout << "numb_run_" << numb_run_ << "start_run_ " << start_run_ << endl;
  if (numb_run_-start_run_== 0 )
  {
    random_field_ = Teuchos::rcp(new GenRandomField(random_seed,discret_));
  }
  else
 {
   random_field_->CreateNewSample(random_seed);
 }
    //random_field_->WriteRandomFieldToFile();
    //dserror("stop right here");
  // loop over all elements
  for (int i=0; i< (discret_->NumMyColElements()); i++)
  {

    if(discret_->lColElement(i)->Material()->MaterialType()==INPAR::MAT::m_aaaneohooke_stopro)
    {
      MAT::AAAneohooke_stopro* aaa_stopro = static_cast <MAT::AAAneohooke_stopro*>(discret_->lColElement(i)->Material().get());
      vector<double> ele_center;
      ele_center = discret_->lColElement(i)->ElementCenterRefeCoords();

      //special hack here assuming circular geometry with r=25 mm
      double phi= acos(ele_center[0]/25);
      //compute x coord
      ele_center[0]=phi*25;
      ele_center[1]=ele_center[2];
      if (i==0)
        youngs = random_field_->EvalFieldAtLocation(ele_center,false,true);
      else
        youngs = random_field_->EvalFieldAtLocation(ele_center,false,false);
      aaa_stopro->Init(youngs,"beta");
      }
    } // EOF loop elements

}

// calculate some statistics
void STR::MLMC::CalcStatStressDisp(RCP< Epetra_MultiVector> curr_stress,RCP< Epetra_MultiVector> curr_strain,RCP<Epetra_MultiVector> curr_disp)
{
  // in order to avoid saving the stresses and displacements for each step an online
  // algorithm to compute the std deviation is needed .

  // Such an algorithm can be found here:

  // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

  // This it how it goes
  //def online_variance(data):
   //   n = 0
   //   mean = 0
   //   M2 = 0
   //
   //   for x in data:
    //      n = n + 1
    //      delta = x - mean
     //     mean = mean + delta/n
      //    M2 = M2 + delta*(x - mean)  # This expression uses the new value of mean
   //
     // variance_n = M2/n
     /// variance = M2/(n - 1)
    //  return variance*/
  // since numb_run_ does not start from zero anymore we need
  int n = numb_run_-start_run_+1;
  // calc mean and variance for displacement
  delta_disp_->Update(1.0,*curr_disp,-1.0,*mean_disp_,0.0);
  mean_disp_->Update(1.0/n,*delta_disp_,1.0);
  m2_helper_var_disp_->Update(1.0,*curr_disp,-1.0,*mean_disp_,0.0);
  m2_var_disp_->Multiply(1.0,*delta_disp_,*m2_helper_var_disp_ ,1.0);
  var_disp_->Update(1.0/(numb_run_),*m2_var_disp_,0.0);
  // do the same for stresses
  delta_stress_->Update(1.0,*curr_stress,-1.0,*mean_stress_,0.0);
  mean_stress_->Update(1.0/n,*delta_stress_,1.0);
  m2_helper_var_stress_->Update(1.0,*curr_stress,-1.0,*mean_stress_,0.0);
  m2_var_stress_->Multiply(1.0,*delta_stress_,*m2_helper_var_stress_ ,1.0);
  var_stress_->Update(1.0/(numb_run_),*m2_var_stress_,0.0);
  // and for strains
  delta_strain_->Update(1.0,*curr_strain,-1.0,*mean_strain_,0.0);
  mean_strain_->Update(1.0/n,*delta_strain_,1.0);
  m2_helper_var_strain_->Update(1.0,*curr_strain,-1.0,*mean_strain_,0.0);
  m2_var_strain_->Multiply(1.0,*delta_strain_,*m2_helper_var_strain_ ,1.0);
  var_strain_->Update(1.0/(numb_run_),*m2_var_strain_,0.0);

  // quick check if we need difference stats
  if (calc_diff_)
  {
  // calc mean and variance for difference between levels
  diff_delta_disp_->Update(1.0,*disp_lower_level_,-1.0,*diff_mean_disp_,0.0);
  diff_mean_disp_->Update(1.0/n,*diff_delta_disp_,1.0);
  diff_m2_helper_var_disp_->Update(1.0,*disp_lower_level_,-1.0,*diff_mean_disp_,0.0);
  diff_m2_var_disp_->Multiply(1.0,*diff_delta_disp_,*diff_m2_helper_var_disp_ ,1.0);
  diff_var_disp_->Update(1.0/(numb_run_),*diff_m2_var_disp_,0.0);
  // do the same for stresses
  diff_delta_stress_->Update(1.0,*stress_lower_level_,-1.0,*diff_mean_stress_,0.0);
  diff_mean_stress_->Update(1.0/n,*diff_delta_stress_,1.0);
  diff_m2_helper_var_stress_->Update(1.0,*stress_lower_level_,-1.0,*diff_mean_stress_,0.0);
  diff_m2_var_stress_->Multiply(1.0,*diff_delta_stress_,*diff_m2_helper_var_stress_ ,1.0);
  diff_var_stress_->Update(1.0/(numb_run_),*diff_m2_var_stress_,0.0);
  // do the same for strains
  diff_delta_strain_->Update(1.0,*strain_lower_level_,-1.0,*diff_mean_strain_,0.0);
  diff_mean_strain_->Update(1.0/n,*diff_delta_strain_,1.0);
  diff_m2_helper_var_strain_->Update(1.0,*strain_lower_level_,-1.0,*diff_mean_strain_,0.0);
  diff_m2_var_strain_->Multiply(1.0,*diff_delta_strain_,*diff_m2_helper_var_strain_ ,1.0);
  diff_var_strain_->Update(1.0/(numb_run_),*diff_m2_var_strain_,0.0);
  }



}
void STR::MLMC::WriteStatOutput()
{  //
  std::stringstream name_helper;
  name_helper << filename_ << "_statistics";
  output_fine_->NewResultFile(name_helper.str(),numb_run_);
  output_fine_->WriteMesh(1, 0.01);
  output_fine_->NewStep( 1, 0.01);
  output_fine_->WriteVector("mean_displacements", mean_disp_, output_fine_->dofvector);
  output_fine_->WriteVector("variance_displacements", var_disp_, output_fine_->dofvector);
  output_fine_->WriteVector("mean_gauss_2PK_stresses_xyz", mean_stress_, output_fine_->nodevector);
  output_fine_->WriteVector("variance_gauss_2PK_stresses_xyz", var_stress_, output_fine_->nodevector);
  output_fine_->WriteVector("mean_gauss_GL_strain_xyz", mean_strain_, output_fine_->nodevector);
  output_fine_->WriteVector("variance_gauss_GL_strain_xyz", var_strain_, output_fine_->nodevector);
  // write stats with respect to lower level
  if (calc_diff_)
  {
    output_fine_->WriteVector("diff_mean_displacements", diff_mean_disp_, output_fine_->dofvector);
    output_fine_->WriteVector("diff_variance_displacements", diff_var_disp_, output_fine_->dofvector);
    output_fine_->WriteVector("diff_mean_gauss_2PK_stresses_xyz", diff_mean_stress_, output_fine_->nodevector);
    output_fine_->WriteVector("diff_variance_gauss_2PK_stresses_xyz", diff_var_stress_, output_fine_->nodevector);
    output_fine_->WriteVector("diff_mean_gauss_GL_strain_xyz", diff_mean_strain_, output_fine_->nodevector);
    output_fine_->WriteVector("diff_variance_gauss_GL_strain_xyz", diff_var_strain_, output_fine_->nodevector);
   }
}
void STR::MLMC::ResetPrestress()
{
  // Reset Presstress possibly still present in Discretization

  // Get prestress parameter
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  // get prestress type
  INPAR::STR::PreStress pstype = DRT::INPUT::IntegralValue<INPAR::STR::PreStress>(sdyn,"PRESTRESS");
    switch(pstype)
    {
    case INPAR::STR::prestress_none:
    {
      cout << "nothing to do";
    }
    break;
    case INPAR::STR::prestress_mulf:
    {
      ParameterList p;
      // action for elements
      p.set("action","calc_struct_reset_discretization");
      discret_->Evaluate(p,null,null,null,null,null);
     }
    break;
    case INPAR::STR::prestress_id:
    {
      dserror("MLMC and pressstressing with ID do not go great together");
    }
    break;
    default:
      dserror("Unknown type of prestressing");
    }
}

void STR::MLMC::HelperForDebuggin()
{
  double sigma = 1.0;
  double corrlength = 30.0;
  //double beta_mean =2.9;
  /// file to write beta output
  ofstream File("OutputStopro2.txt");
  cout << "Debugging LINE "<< __LINE__ <<endl;
  // get element 12 centercoords
  DRT::Node** nodes = discret_->gElement(2)->Nodes();
  vector<double> ele_center;
  // init to zero
  ele_center.push_back(0.0);
  ele_center.push_back(0.0);
  ele_center.push_back(0.0);
  cout << "Debugging LINE "<< __LINE__ <<endl;
  for (int i = 0; i < 8; i++ )
  {
    ele_center[0] += nodes[i]->X()[0]/8.;
    ele_center[1] += nodes[i]->X()[1]/8.;
    ele_center[2] += nodes[i]->X()[2]/8.;
  }

  // Eval Random Field at same location and write results to file
  RandomField field(1232132,sigma,corrlength);
  cout << "sigma  " << sigma << "corrlenght " << corrlength << endl;
  // generate grid
  int num_grid_points= 1000 ;
  double spacing = 5.0;
  vector<double> x;
  vector<double> y;
  vector<double> z;
  vector<double> result;
  x.reserve( num_grid_points);
  y.reserve( num_grid_points);
  z.reserve( num_grid_points);
  for(int i = 0; i < num_grid_points ; i++)
  {
    x[i]= spacing*i;
    y[i]= spacing*i;
    z[i]= spacing*i;
  }
  //result.reserve( num_grid_points*num_grid_points*num_grid_points);
  for (int i=0;i <num_grid_points; i++)
  {
    for (int j=0;j <num_grid_points; j++)
    {
      //for (int k=0;k <num_grid_points; k++)
      //{
        //result[i + j*num_grid_points + k*num_grid_points*num_grid_points]=field.EvalRandomField3D(x[i],y[j],z[k]);
        File << field.EvalRandomField2D(x[i],y[j],z[j])<< endl;
      //}
    }
  }

}

void STR::MLMC::HelperFunctionOutput(RCP< Epetra_MultiVector> stress,RCP< Epetra_MultiVector> strain, RCP<Epetra_MultiVector> disp)
{
  // assamble name for outputfil
  std::stringstream outputfile;
  outputfile << filename_ << "_statistics_output_" << start_run_ << ".txt";
  string name = outputfile.str();;
  /// file to write output
  ofstream File;
  if (numb_run_ == 0 || numb_run_ == start_run_)
  {
    File.open(name.c_str(),ios::out);
    File << "run id   "<< "xdisp node 24 " << "S_xx node 436 "<< endl;
    File.close();
  }
  // reopen in append mode
  File.open(name.c_str(),ios::app);
  File << numb_run_ << "    "<< (*disp)[0][72]<< "    " << (*stress)[0][435] << endl;
  File.close();
}
void STR::MLMC::HelperFunctionOutputTube(RCP< Epetra_MultiVector> stress,RCP< Epetra_MultiVector> strain, RCP<Epetra_MultiVector> disp)
{

  // assamble name for outputfil
  std::stringstream outputfile;
  outputfile << filename_ << "_statistics_output_" << start_run_ << ".txt";
  string name = outputfile.str();;
  /// file to write output
  // paraview ids of nodes
  int node[5] = {566, 1764, 3402,5194,6510};
  double disp_mag[5];
  double stress_mag[5];
  //double strain_mag[5];
  ofstream File;
  if (numb_run_ == 0 || numb_run_ == start_run_)
  {
    File.open(name.c_str(),ios::out);
    if (File.is_open())
    {
      //if(error == -1)
      //dserror("Unable to open Statistics output Filename");
      File << "run id   "<< "disp mag node   " << node[0] << "S_mag node   " << node[0] << "disp mag node   " << node[1] << "S_mag node   " << node[1]
       << "disp mag node   " << node[2] << "S_mag node   " << node[2] << "disp mag node   " << node[3] << "S_mag node   "
       << node[3] << "disp mag node   " << node[4] << "S_mag node   " << node[4]<< endl;
      File.close();
    }
    //
    else
    {
    dserror("Unable to open statistics output file");
    }
  }
  for(int i =0; i<5 ; i++)
  {
    //calc mag disp
    disp_mag[i]=sqrt(pow((*disp)[0][node[i]*3],2)+pow((*disp)[0][node[i]*3+1],2)+pow((*disp)[0][node[i]*3+2],2));
    stress_mag[i]=sqrt(pow((*stress)[0][node[i]],2)+pow((*stress)[1][node[i]],2)+ pow((*stress)[2][node[i]],2)+ pow((*stress)[3][node[i]],2)+
        pow((*stress)[4][node[i]],2)+pow((*stress)[5][node[i]],2));

  }
  // reopen in append mode
  File.open(name.c_str(),ios::app);
  File << numb_run_ << "    "<< disp_mag[0]<< "    " << stress_mag[0] << "    "<< disp_mag[1]<< "    " << stress_mag[1] << "    "<< disp_mag[2]<< "    " << stress_mag[2]
    << "    "<< disp_mag[3]<< "    " << stress_mag[3] << "    "<< disp_mag[4]<< "    " << stress_mag[4] <<  endl;
  File.close();
}

void STR::MLMC::EvalDisAtNodes(Teuchos::RCP<const Epetra_Vector> disp )
{

  const int myrank = actdis_coarse_->Comm().MyPID();
  // build map that lives only on proc 0
  // nodes for fine discretization
   int node[5] = {1528, 3905, 7864, 10832, 13720};
   int dofs[15]={4584, 4585, 4586,11715, 11716, 11717,23592,23593, 23594,32496, 32497, 32498,41160, 41161, 41162};
   // nodes for coarse sicretization
   //int node[5] = {112, 257, 526, 728, 910};
  //const int* targetgids;
  //targetgids =
  int numglobalelements =5;
   int numglobalelements_dof =15;
  int nummyelements;
  int nummyelements_dof;

  if (actdis_coarse_->Comm().MyPID()==0)
  {
   nummyelements = 5;
   nummyelements_dof = 15;
  }
  else
  {
   nummyelements = 0;
   nummyelements_dof = 0;
  }

  Epetra_Map output_node_map(numglobalelements,nummyelements,&node[0],0,actdis_coarse_->Comm());
  Epetra_Map output_dof_map(numglobalelements_dof,nummyelements_dof,&dofs[0],0,actdis_coarse_->Comm());

  INPAR::STR::StressType iostress =INPAR::STR::stress_2pk; //stress_none;
  INPAR::STR::StrainType iostrain= INPAR::STR::strain_gl; // strain_none;
  Teuchos::RCP<std::vector<char> > stress = Teuchos::rcp(new std::vector<char>());
  Teuchos::RCP<std::vector<char> > strain = Teuchos::rcp(new std::vector<char>());
  Teuchos::RCP<std::vector<char> > plstrain = Teuchos::rcp(new std::vector<char>());

  // create the parameters for the discretization
  ParameterList p;
  p.set("action","calc_struct_stress");
  p.set("stress", stress);
  p.set("plstrain",plstrain);
  p.set("strain", strain);

  p.set<int>("iostress", iostress);
  p.set<int>("iostrain", iostrain);

  RCP<Epetra_Vector>    zeros_coarse = rcp(new Epetra_Vector(*actdis_coarse_->DofRowMap(),true));
  RCP<Epetra_Vector>    vel_coarse = rcp(new Epetra_Vector(*actdis_coarse_->DofRowMap(),true));
  actdis_coarse_->ClearState();
  actdis_coarse_->SetState("residual displacement",zeros_coarse);
  // disp is passed to the function no need for reading in results anymore
  actdis_coarse_->SetState("displacement",disp);
  actdis_coarse_->SetState("velocity",vel_coarse);
  // Alrigth lets get the nodal stresses
  p.set("action","calc_global_gpstresses_map");
  const RCP<map<int,RCP<Epetra_SerialDenseMatrix> > > gpstressmap = rcp(new std::map<int, RCP<Epetra_SerialDenseMatrix> >);
  p.set("gpstressmap", gpstressmap);

  const RCP<map<int,RCP<Epetra_SerialDenseMatrix> > > gpstrainmap = rcp(new std::map<int, RCP<Epetra_SerialDenseMatrix> >);
 p.set("gpstrainmap", gpstrainmap);


  actdis_coarse_->Evaluate(p,null,null,null,null,null);
  actdis_coarse_->ClearState();


  // st action to calc poststresse
  p.set("action","postprocess_stress");
  // Multivector to store poststresses
  // WE need the collumn Map here because we need all the ghosted nodes to calculate stresses at the nodes
  RCP<Epetra_MultiVector> poststress =  Teuchos::rcp(new Epetra_MultiVector(*(actdis_coarse_->NodeColMap()),6,true));

  p.set("gpstressmap", gpstressmap);
  p.set("poststress", poststress);
  p.set("stresstype","ndxyz");

  //actdis_coarse_->ClearState();
  actdis_coarse_->Evaluate(p,null,null,null,null,null);
  actdis_coarse_->ClearState();

  // again for strains
  p.set("action","postprocess_stress");
  // Multivector to store poststrains
  RCP<Epetra_MultiVector> poststrain =  Teuchos::rcp(new Epetra_MultiVector(*(actdis_coarse_->NodeColMap()),6,true));

  p.set("poststress", poststrain);
  p.set("gpstressmap", gpstrainmap);
  p.set("stresstype","ndxyz");

  actdis_coarse_->ClearState();
  actdis_coarse_->Evaluate(p,null,null,null,null,null);
  actdis_coarse_->ClearState();


  // assamble name for outputfile
  std::stringstream outputfile2;
  outputfile2 << filename_ << "_statistics_output_" << start_run_ << ".txt";
  string name2 = outputfile2.str();;
  // file to write output

  RCP<Epetra_Vector> output_disp = LINALG::CreateVector(output_dof_map,false);
  RCP<Epetra_MultiVector> output_stress =  Teuchos::rcp(new Epetra_MultiVector(output_node_map,6,true));
  RCP<Epetra_MultiVector> output_strain =  Teuchos::rcp(new Epetra_MultiVector(output_node_map,6,true));
  LINALG::Export(*poststress,*output_stress);
  LINALG::Export(*poststrain,*output_strain);
  LINALG::Export(*disp,*output_disp);


  if(myrank==0)
  {
    ofstream File;
    if (numb_run_ == 0 || numb_run_ == start_run_)
    {
      File.open(name2.c_str(),ios::out);
      if (File.is_open())
      {
        //if(error == -1)
        //dserror("Unable to open Statistics output Filename");
        File << "run id   "<< "disp x  disp y disp z  node   " << node[0] << " "
               << "disp x  disp y disp z  node   " << node[1]<< " "
               << "disp x  disp y disp z  node   " << node[2]<< " "
               << "disp x  disp y disp z  node   " << node[3]<< " "
               << "disp x  disp y disp z  node   " << node[4]<< " "
               // stresses
               << "stress xx  stress yy stress zz stress xy stress yz stress xz node   " << node[0] << " "
               << "stress xx  stress yy stress zz stress xy stress yz stress xz node   " << node[1] << " "
               << "stress xx  stress yy stress zz stress xy stress yz stress xz node   " << node[2] << " "
               << "stress xx  stress yy stress zz stress xy stress yz stress xz node   " << node[3] << " "
               << "stress xx  stress yy stress zz stress xy stress yz stress xz node   " << node[4] << " "
              // << "stress xx  stress yy stress zz stress xy stress yz stress xz node   " << node[5] << " "
               // strains
               << "strain xx  strain yy strain zz strain xy strain yz strain xz node   " << node[0] << " "
               << "strain xx  strain yy strain zz strain xy strain yz strain xz node   " << node[1] << " "
               << "strain xx  strain yy strain zz strain xy strain yz strain xz node   " << node[2] << " "
               << "strain xx  strain yy strain zz strain xy strain yz strain xz node   " << node[3] << " "
               << "strain xx  strain yy strain zz strain xy strain yz strain xz node   " << node[4] << " "
              // << "strain xx  strain yy strain zz strain xy strain yz strain xz node   " << node[5] << " "
               << endl;
        File.close();
      }
      //
      else
      {
      dserror("Unable to open statistics output file");
      }
    }
    // reopen in append mode
    File.open(name2.c_str(),ios::app);
    File << numb_run_ ;
    for(int i=0;i<15;i++)
    {
      File <<  "  " << (*output_disp)[i];
    }
    for (int i=0;i<5;i++)
    {
      for(int j=0;j<6;j++)
      {
        File << " " << (*output_stress)[j][i];
      }
    }
    for (int i=0;i<5;i++)
    {
      for(int j=0;j<6;j++)
      {
        File << " " << (*output_strain)[j][i];
      }
    }
    File << endl;
    File.close();
  }
  actdis_coarse_->Comm().Barrier();
}

#endif /*CCARAT*/
#endif // FFTW
