/*----------------------------------------------------------------------*/
/*!
 * \file gen_inv_analysis.cpp

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/gee
            089 - 289-15239
</pre>
*/
/*----------------------------------------------------------------------*/


#include "gen_inv_analysis.H"
#include "../drt_inpar/inpar_material.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io_hdf.H"
#include "../drt_io/io.H"
#include "../drt_lib/drt_condition.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_element.H"
#include "../drt_lib/drt_function.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_mat/aaaneohooke.H"
#include "../drt_mat/constraintmixture.H"
#include "../drt_mat/growth_law.H"
#include "../drt_mat/elasthyper.H"
#include "../drt_mat/material.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_mat/matpar_parameter.H"
#include "../drt_mat/micromaterial.H"
#include "../drt_mat/neohooke.H"
#include "../drt_matelast/elast_coup1pow.H"
#include "../drt_matelast/elast_coup2pow.H"
#include "../drt_mat/viscogenmax.H"
#include "../drt_mat/viscoelasthyper.H"
#include "../drt_matelast/elast_coupblatzko.H"
#include "../drt_matelast/elast_couplogneohooke.H"
#include "../drt_matelast/elast_coupmooneyrivlin.H"
#include "../drt_matelast/elast_coupneohooke.H"
#include "../drt_matelast/elast_iso1pow.H"
#include "../drt_matelast/elast_iso2pow.H"
#include "../drt_matelast/elast_isocub.H"
#include "../drt_matelast/elast_isoexpopow.H"
#include "../drt_matelast/elast_isomooneyrivlin.H"
#include "../drt_matelast/elast_isoneohooke.H"
#include "../drt_matelast/elast_isoquad.H"
#include "../drt_matelast/elast_isoyeoh.H"
#include "../drt_matelast/elast_vologden.H"
#include "../drt_matelast/elast_volpenalty.H"
#include "../drt_matelast/elast_volsussmanbathe.H"
#include "../drt_matelast/visco_isoratedep.H"
#include "../drt_structure/strtimint_create.H"
#include "../drt_structure/strtimint.H"
#include "../drt_stru_multi/microstatic.H"
#include "../linalg/linalg_utils.H"
#include "Epetra_SerialDenseMatrix.h"
#include "inv_analysis.H"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "../drt_lib/drt_discret.H"
#include "../drt_comm/comm_utils.H"
#include "../drt_inpar/inpar_invanalysis.H"



#include "../drt_structure/stru_resulttest.H"



/*----------------------------------------------------------------------*/
/* standard constructor */
STR::GenInvAnalysis::GenInvAnalysis(Teuchos::RCP<DRT::Discretization> dis,
                                    Teuchos::RCP<LINALG::Solver> solver,
                                    Teuchos::RCP<IO::DiscretizationWriter> output)
  : discret_(dis),
    solver_(solver),
    output_(output),
    sti_(Teuchos::null)
{
  int myrank = dis->Comm().MyPID();

  reset_out_count_=0;

  // input parameters inverse analysis
  const Teuchos::ParameterList& iap = DRT::Problem::Instance()->InverseAnalysisParams();
  //  tolerance for the curve fitting
  tol_ = iap.get<double>("INV_ANA_TOL");

  /* this is how a monitor file should look like:

steps 25 nnodes 5
11 2 0 1
19 2 0 1
27 2 0 1
35 2 0 1
42 2 0 1
# any number of comment lines, but only at this position
# x and y disps at 5 nodes
# time node 11 x y 19 x y 27 x y 35 x y 42 x y
#
2.000000e-02   -9.914936e-02    7.907108e-02   -2.421293e-01    3.475674e-01   -2.018457e-01    6.164171e-01   2.622612e-02    5.351817e-01  2.623198e-03    6.524309e-02
4.000000e-02   -2.060081e-01    1.561979e-01   -4.495438e-01    6.425663e-01   -3.561049e-01    1.099916e+00   7.625965e-02    9.675498e-01  4.770331e-02    1.287581e-01
6.000000e-02   -3.063373e-01    2.286715e-01   -6.147396e-01    8.904266e-01   -4.628068e-01    1.481550e+00   1.405767e-01    1.313955e+00  1.144058e-01    1.871826e-01
8.000000e-02   -3.995311e-01    2.965706e-01   -7.500371e-01    1.103765e+00   -5.399646e-01    1.795528e+00   2.080302e-01    1.600827e+00  1.902863e-01    2.404236e-01
1.000000e-01   -4.871174e-01    3.603335e-01   -8.656673e-01    1.291507e+00   -6.003143e-01    2.063556e+00   2.732486e-01    1.846735e+00  2.684362e-01    2.890503e-01
1.200000e-01   -5.705636e-01    4.204134e-01   -9.684391e-01    1.459720e+00   -6.514304e-01    2.298985e+00   3.338117e-01    2.063424e+00  3.449541e-01    3.336880e-01
1.400000e-01   -6.508979e-01    4.772118e-01   -1.062618e+00    1.612613e+00   -6.975891e-01    2.510286e+00   3.888783e-01    2.258437e+00  4.178564e-01    3.749320e-01
1.600000e-01   -7.287384e-01    5.310691e-01   -1.150807e+00    1.753191e+00   -7.411234e-01    2.703069e+00   4.384501e-01    2.436805e+00  4.863525e-01    4.133178e-01
1.800000e-01   -8.044079e-01    5.822715e-01   -1.234572e+00    1.883657e+00   -7.832539e-01    2.881204e+00   4.829452e-01    2.602019e+00  5.503314e-01    4.493058e-01
2.000000e-01   -8.780469e-01    6.310615e-01   -1.314853e+00    2.005671e+00   -8.245805e-01    3.047458e+00   5.229514e-01    2.756584e+00  6.100265e-01    4.832787e-01
2.200000e-01   -9.496984e-01    6.776474e-01   -1.392223e+00    2.120513e+00   -8.653673e-01    3.203878e+00   5.590896e-01    2.902354e+00  6.658185e-01    5.155483e-01
2.400000e-01   -1.019365e+00    7.222113e-01   -1.467040e+00    2.229192e+00   -9.057055e-01    3.352017e+00   5.919446e-01    3.040735e+00  7.181279e-01    5.463664e-01
2.600000e-01   -1.087042e+00    7.649147e-01   -1.539546e+00    2.332515e+00   -9.456044e-01    3.493080e+00   6.220354e-01    3.172817e+00  7.673616e-01    5.759360e-01
2.800000e-01   -1.152735e+00    8.059027e-01   -1.609916e+00    2.431140e+00   -9.850398e-01    3.628021e+00   6.498057e-01    3.299461e+00  8.138890e-01    6.044213e-01
3.000000e-01   -1.216466e+00    8.453065e-01   -1.678289e+00    2.525611e+00   -1.023980e+00    3.757606e+00   6.756268e-01    3.421355e+00  8.580353e-01    6.319557e-01
3.200000e-01   -1.278278e+00    8.832457e-01   -1.744784e+00    2.616378e+00   -1.062399e+00    3.882462e+00   6.998053e-01    3.539062e+00  9.000811e-01    6.586487e-01
3.400000e-01   -1.338225e+00    9.198289e-01   -1.809512e+00    2.703823e+00   -1.100276e+00    4.003104e+00   7.225922e-01    3.653044e+00  9.402670e-01    6.845912e-01
3.600000e-01   -1.396378e+00    9.551554e-01   -1.872574e+00    2.788270e+00   -1.137603e+00    4.119965e+00   7.441921e-01    3.763688e+00  9.787981e-01    7.098591e-01
3.800000e-01   -1.452811e+00    9.893160e-01   -1.934068e+00    2.869995e+00   -1.174379e+00    4.233408e+00   7.647719e-01    3.871317e+00  1.015850e+00    7.345163e-01
4.000000e-01   -1.507606e+00    1.022393e+00   -1.994085e+00    2.949237e+00   -1.210611e+00    4.343745e+00   7.844680e-01    3.976208e+00  1.051572e+00    7.586177e-01
4.200000e-01   -1.560847e+00    1.054463e+00   -2.052713e+00    3.026205e+00   -1.246311e+00    4.451242e+00   8.033921e-01    4.078601e+00  1.086093e+00    7.822100e-01
4.400000e-01   -1.612617e+00    1.085593e+00   -2.110034e+00    3.101081e+00   -1.281495e+00    4.556132e+00   8.216362e-01    4.178700e+00  1.119525e+00    8.053342e-01
4.600000e-01   -1.662996e+00    1.115847e+00   -2.166128e+00    3.174023e+00   -1.316181e+00    4.658615e+00   8.392767e-01    4.276685e+00  1.151964e+00    8.280258e-01
4.800000e-01   -1.712064e+00    1.145282e+00   -2.221065e+00    3.245174e+00   -1.350390e+00    4.758870e+00   8.563772e-01    4.372714e+00  1.183494e+00    8.503160e-01
5.000000e-01   -1.759895e+00    1.173949e+00   -2.274916e+00    3.314659e+00   -1.384141e+00    4.857055e+00   8.729911e-01    4.466927e+00  1.214188e+00    8.722325e-01


  */

  // open monitor file and read it
  ndofs_ = 0;
  {
    char* foundit = NULL;
    std::string monitorfilename = iap.get<std::string>("MONITORFILE");
    if (monitorfilename=="none.monitor") dserror("No monitor file provided");
    // insert path to monitor file if necessary
    if (monitorfilename[0]!='/')
    {
      std::string filename = DRT::Problem::Instance()->OutputControlFile()->InputFileName();
      std::string::size_type pos = filename.rfind('/');
      if (pos!=std::string::npos)
      {
        std::string path = filename.substr(0,pos+1);
        monitorfilename.insert(monitorfilename.begin(), path.begin(), path.end());
      }
    }

    FILE* file = fopen(monitorfilename.c_str(),"rb");
    if (file==NULL) dserror("Could not open monitor file %s",monitorfilename.c_str());

    char buffer[150000];
    fgets(buffer,150000,file);
    // read steps
    foundit = strstr(buffer,"steps"); foundit += strlen("steps");
    nsteps_ = strtol(foundit,&foundit,10);
    timesteps_.resize(nsteps_);
    // read nnodes
    foundit = strstr(buffer,"nnodes"); foundit += strlen("nnodes");
    nnodes_ = strtol(foundit,&foundit,10);
    // read nodes
    nodes_.resize(nnodes_);
    dofs_.resize(nnodes_);
    for (int i=0; i<nnodes_; ++i)
    {
      fgets(buffer,150000,file);
      foundit = buffer;
      nodes_[i] = strtol(foundit,&foundit,10);
      int ndofs = strtol(foundit,&foundit,10);
      ndofs_ += ndofs;
      dofs_[i].resize(ndofs,-1);
      if (!myrank) printf("Monitored node %d ndofs %d dofs ",nodes_[i],(int)dofs_[i].size());
      for (int j=0; j<ndofs; ++j)
      {
        dofs_[i][j] = strtol(foundit,&foundit,10);
        if (!myrank) printf("%d ",dofs_[i][j]);
      }
      if (!myrank) printf("\n");
    }

    // total number of measured dofs
    nmp_    = ndofs_*nsteps_;


    // read in measured curve

    {
      mcurve_ = Epetra_SerialDenseVector(nmp_);

      if (!myrank) printf("nsteps %d ndofs %d\n",nsteps_,ndofs_);

      // read comment lines
      foundit = buffer;
      fgets(buffer,150000,file);
      while(strstr(buffer,"#"))
        fgets(buffer,150000,file);

      // read in the values for each node in dirs directions
      int count=0;
      for (int i=0; i<nsteps_; ++i)
      {
        // read the time step
        timesteps_[i] = strtod(foundit,&foundit);
        for (int j=0; j<ndofs_; ++j)
          mcurve_[count++] = strtod(foundit,&foundit);
        fgets(buffer,150000,file);
        foundit = buffer;
      }
      if (count != nmp_) dserror("Number of measured disps wrong on input");
    }
  }

  // error: diference of the measured to the calculated curve
  error_  = 1.0E6;
  error_o_= 1.0E6;
  //
  error_grad_  = 1.0E6;
  error_grad_o_= 1.0E6;

  // trainings parameter
  mu_ = iap.get<double>("INV_INITREG");
  mu_o_ = mu_;
  kappa_multi_=1.0;

  // update strategy for mu
  switch(DRT::INPUT::IntegralValue<INPAR::STR::RegStratUpdate>(iap,"UPDATE_REG"))
   {
     case INPAR::STR::reg_update_grad:
     {
      reg_update_ = grad_based;
     }
     break;
     case INPAR::STR::reg_update_res:
     {
       reg_update_ = res_based;
     }
     break;
     default:
       dserror("Unknown update strategy for regularization parameter! Fix your input file");
		  break;
   }

  check_neg_params_ = DRT::INPUT::IntegralValue<int>(iap,"PARAM_BOUNDS");

  // list of materials for each problem instance that should be fitted
  for (unsigned prob=0; prob<DRT::Problem::NumInstances(); ++prob)
  {
    const Teuchos::ParameterList& myiap = DRT::Problem::Instance(prob)->InverseAnalysisParams();
    std::set<int> myset;

    int word1;
    std::istringstream matliststream(Teuchos::getNumericStringParameter(myiap,"INV_LIST"));
    while (matliststream >> word1)
    {
      if (word1!=-1) // this means there was no matlist specified in the input file
        myset.insert(word1);
    }
    matset_.push_back(myset);
  }

  // read material parameters from input file
  ReadInParameters();
  p_o_ = p_;

  // Number of material parameters
  np_ = p_.Length();

  // controlling parameter
  numb_run_ =  0;     //
                      // counter of how many runs were made in the inverse analysis

}


/*----------------------------------------------------------------------*/
/* analyse */
void STR::GenInvAnalysis::Integrate()
{
  const int myrank  = discret_->Comm().MyPID();
  const Teuchos::ParameterList& iap = DRT::Problem::Instance()->InverseAnalysisParams();
  int max_itter = iap.get<int>("INV_ANA_MAX_RUN");
  int newfiles = DRT::INPUT::IntegralValue<int>(iap,"NEW_FILES");

  // fitting loop
  do
  {
    if (!myrank)
    {
      std::cout << "#################################################################" << std::endl;
      std::cout << "########################### making Jacobian matrix ##############" <<std::endl;
      std::cout << "#################################################################" << std::endl;
      printf("Measured parameters nmp_ %d # parameters to fit np_ %d\n",nmp_,np_);
    }

    // pertubation of material parameter (should be relativ to the value that is perturbed)
    std::vector<double> perturb(np_,0.0);
    double alpha = iap.get<double>("INV_ALPHA");
    double beta  = iap.get<double>("INV_BETA");
    for (int i=0; i<np_; ++i)
    {
      perturb[i] = alpha + beta * p_[i];
      if (!myrank) printf("perturbation[%d] %15.10e\n",i,perturb[i]);
    }

    // as the actual inv analysis is on proc 0 only, do only provide storage on proc 0
    Epetra_SerialDenseMatrix cmatrix;
    if (!myrank) cmatrix.Shape(nmp_,np_+1);

    // loop over parameters to fit and build cmatrix
    for (int i=0; i<np_+1;i++)
    {
      bool outputtofile = false;
      // output only for last run
      if (i==np_) outputtofile = true;

      if (outputtofile)
      {
        // no parameters for newfile yet
        if (newfiles)
          output_->NewResultFile((numb_run_));
        else
          output_->OverwriteResultFile();

        output_->WriteMesh(0,0.0);
      }

      // Multi-scale: if an inverse analysis is performed on the micro-level,
      // the time and step need to be reset now. Furthermore, the result file
      // needs to be opened.
      MultiInvAnaInit();

      if (!myrank)
        std::cout << "--------------------------- run "<< i+1 << " of: " << np_+1 <<" -------------------------" <<std::endl;
      // make current set of material parameters
      Epetra_SerialDenseVector p_cur = p_;
      // perturb parameter i
      if (i!= np_) p_cur[i] = p_[i] + perturb[i];

      // put perturbed material parameters to material laws
      discret_->Comm().Broadcast(&p_cur[0],p_cur.Length(),0);
      SetParameters(p_cur);

      // compute nonlinear problem and obtain computed displacements
      // output at the last step
      Epetra_SerialDenseVector cvector;
      cvector = CalcCvector(outputtofile);

      // copy displacements to sensitivity matrix
      if (!myrank)
        for (int j=0; j<nmp_;j++)
          cmatrix(j,i) = cvector[j];

      Teuchos::ParameterList p;
      p.set("action","calc_struct_reset_all");
      discret_->Evaluate(p,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);
    }

    discret_->Comm().Barrier();
    if (!myrank) CalcNewParameters(cmatrix,perturb);

    // set new material parameters
    discret_->Comm().Broadcast(&p_[0],p_.Length(),0);
    SetParameters(p_);
    numb_run_++;
    discret_->Comm().Broadcast(&error_o_,1,0);
    discret_->Comm().Broadcast(&error_,1,0);
    discret_->Comm().Broadcast(&error_grad_o_,1,0);
    discret_->Comm().Broadcast(&error_grad_,1,0);
    discret_->Comm().Broadcast(&numb_run_,1,0);

    if(reg_update_ == grad_based)
      error_i_ = error_grad_;
    else if(reg_update_ == res_based)
      error_i_ = error_;


  } while (error_i_>tol_ && numb_run_<max_itter);


  // print results to file
  if (!myrank) PrintFile();

  // stop supporting processors in multi scale simulations
  Teuchos::RCP<Epetra_Comm> subcomm = DRT::Problem::Instance(0)->GetNPGroup()->SubComm();
  if(subcomm != Teuchos::null)
  {
    STRUMULTI::stop_np_multiscale();
  }

  return;
}

/*----------------------------------------------------------------------*/
/* analyse */
void STR::GenInvAnalysis::NPIntegrate()
{
  Teuchos::RCP<COMM_UTILS::NestedParGroup> group = DRT::Problem::Instance()->GetNPGroup();
  Teuchos::RCP<Epetra_Comm> lcomm = group->LocalComm();
  Teuchos::RCP<Epetra_Comm> gcomm = group->GlobalComm();
  const int lmyrank  = lcomm->MyPID();
  const int gmyrank  = gcomm->MyPID();
  const int groupid  = group->GroupId();
  const int ngroup   = group->NumGroups();

  // make some plausability checks:
  //  np_+1 must divide by the number of groups (np_+1) % ngroup = 0
  if ( (np_+1)%ngroup != 0)
    dserror("# material parameters + 1 (%d) must divide by # groups (%d)",np_+1,ngroup);

  // group 0 does the unpermuted solution
  // groups 1 to np_ do the permuted versions
  // proc 0 of group 0 (which is also gproc 0) does the optimization step

  if (!discret_->Filled() || !discret_->HaveDofs()) discret_->FillComplete(true,true,true);

  const Teuchos::ParameterList& iap = DRT::Problem::Instance()->InverseAnalysisParams();
  int max_itter = iap.get<int>("INV_ANA_MAX_RUN");
  int newfiles = DRT::INPUT::IntegralValue<int>(iap,"NEW_FILES");

  // fitting loop
  do
  {
    gcomm->Barrier();

    if (!gmyrank)
    {
      std::cout << "#################################################################" << std::endl;
      std::cout << "######################## NP making Jacobian matrix ##############" <<std::endl;
      std::cout << "#################################################################" << std::endl;
      printf("Measured parameters nmp_ %d # parameters to fit np_ %d NP groups %d\n",nmp_,np_,ngroup);
      fflush(stdout);
    }
    gcomm->Barrier();

    // pertubation of material parameter (should be relativ to the value that is perturbed)
    std::vector<double> perturb(np_,0.0);
    double alpha = iap.get<double>("INV_ALPHA");
    double beta  = iap.get<double>("INV_BETA");
    for (int i=0; i<np_; ++i)
    {
      perturb[i] = alpha + beta * p_[i];
      if (!gmyrank) printf("perturbation[%d] %15.10e\n",i,perturb[i]);
    }

    gcomm->Barrier();

    // as the actual inv analysis is on proc 0 only, do only provide storage on proc 0
    Epetra_SerialDenseMatrix cmatrix;
    if (!gmyrank) cmatrix.Shape(nmp_,np_+1);

    //---------------------------- loop over parameters to fit and build cmatrix
    int i=groupid; // every group start with different material parameter perturbed
    for (; i<np_+1; i+=ngroup) // loop with increments of ngroup
    {
      bool outputtofile = false;
      // output only for last run
      if (i==np_) outputtofile = true;

      if (outputtofile)
      {
        // no parameters for newfile yet
        if (newfiles)
          output_->NewResultFile((numb_run_));
        else
          output_->OverwriteResultFile();

        output_->WriteMesh(0,0.0);
      }

      // Multi-scale: if an inverse analysis is performed on the micro-level,
      // the time and step need to be reset now. Furthermore, the result file
      // needs to be opened.
      MultiInvAnaInit(); // this might not work in here, I'm not sure

      if (!lmyrank)
        std::cout << "--------------------------- run "<< i+1 << " of: " << np_+1 <<" -------------------------" <<std::endl;
      // make current set of material parameters
      Epetra_SerialDenseVector p_cur = p_;
      // perturb parameter i
      if (i!= np_) p_cur[i] = p_[i] + perturb[i];

      // put perturbed material parameters to material laws
      // these are different for every group
      lcomm->Broadcast(&p_cur[0],p_cur.Length(),0);
      SetParameters(p_cur);

      // compute nonlinear problem and obtain computed displacements
      // output at the last step
      Epetra_SerialDenseVector cvector;
      cvector = CalcCvector(outputtofile,group);
      // output cvector is on lmyrank=0 of every group
      // all other procs have zero vectors of the same size here
      // communicate cvectors to gmyrank 0 and put into cmatrix
      for (int j=0; j<ngroup; ++j)
      {
        int ssender=0;
        int sender=0;
        if (lmyrank==0 && groupid==j) ssender = gmyrank;
        gcomm->SumAll(&ssender,&sender,1);
        int n[2];
        n[0] = cvector.Length();
        n[1] = i;
        gcomm->Broadcast(n,2,sender);
        Epetra_SerialDenseVector sendbuff(n[0]);
        if (sender==gmyrank) sendbuff = cvector;
        gcomm->Broadcast(sendbuff.Values(),n[0],sender);
        if (gmyrank==0) // gproc 0 puts received cvector in cmatrix
        {
          if (nmp_ != n[0]) dserror("Size mismatch nmp_ %d != n %d",nmp_,n[0]);
          int ii = n[1];
          for (int k=0; k<nmp_; ++k)
            cmatrix(k,ii) = sendbuff[k];
        }
      }

      // reset discretization to blank
      Teuchos::ParameterList p;
      p.set("action","calc_struct_reset_all");
      discret_->Evaluate(p,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null,Teuchos::null);
    } //--------------------------------------------------------------------------------

    gcomm->Barrier();
    if (!gmyrank) CalcNewParameters(cmatrix,perturb);

    // set new material parameters
    // these are the same for all groups
    gcomm->Broadcast(&p_[0],p_.Length(),0);
    SetParameters(p_);

    numb_run_++;
    gcomm->Broadcast(&error_o_,1,0);
    gcomm->Broadcast(&error_,1,0);
    gcomm->Broadcast(&error_grad_o_,1,0);
    gcomm->Broadcast(&error_grad_,1,0);
    gcomm->Broadcast(&numb_run_,1,0);

    if(reg_update_ == grad_based)
      error_i_ = error_grad_;
    else if(reg_update_ == res_based)
      error_i_ = error_;

  } while (error_i_>tol_ && numb_run_<max_itter);
//printf("gmyrank %d reached this point\n",gmyrank); fflush(stdout); exit(0);


  // print results to file
  if (!gmyrank) PrintFile();

  return;
}

//---------------------------------------------------------------------------------------------
void STR::GenInvAnalysis::CalcNewParameters(Epetra_SerialDenseMatrix& cmatrix, std::vector<double>& perturb)
{
  // initalization of the Jacobian and other storage
  Epetra_SerialDenseMatrix sto(np_,  np_);
  Epetra_SerialDenseVector delta_p(np_);
  Epetra_SerialDenseVector tmp(np_);
  Epetra_SerialDenseVector rcurve(nmp_);
  Epetra_SerialDenseVector ccurve(nmp_);

  // copy column with unperturbed values to extra vector
  for (int i=0; i<nmp_; i++) ccurve[i] = cmatrix(i,np_);

  // remove the extra column np_+1 with unperturbed values
  cmatrix.Reshape(nmp_,np_);

  // reuse the cmatrix array as J to save storage
  Epetra_SerialDenseMatrix& J = cmatrix;

  //calculating J(p)
  for (int i=0; i<nmp_; i++)
    for (int j=0; j<np_; j++)
    {
      J(i,j) -= ccurve[i];
      J(i,j) /= perturb[j];
    }

  //calculating J.T*J)
  sto.Multiply('T','N',1.0,J,J,0.0);

  // calculating  J.T*J + mu*diag(J.T*J)
  // do regularization by adding artifical mass on the main diagonal
  for (int i=0; i<np_; i++)
    sto(i,i) += mu_*sto(i,i);

  //calculating R
  // compute residual displacement (measured vs. computed)
  for (int i=0; i<nmp_; i++)
    rcurve[i] = mcurve_[i] - ccurve[i];

  // delta_p = (J.T*J+mu*diag(J.T*J))^{-1} * J.T*R
  tmp.Multiply('T','N',1.0,J,rcurve,0.0);
  Epetra_SerialDenseSolver solver;
  solver.SetMatrix(sto);
  solver.SetVectors(delta_p,tmp);
  solver.SolveToRefinedSolution(true);
  solver.Solve();

  // dependent on the # of steps
  error_grad_o_ = error_grad_;
  error_o_ = error_;
  mu_o_ = mu_;
  p_o_ = p_;

  // update res based error
  error_   = rcurve.Norm2()/sqrt(nmp_);
  // Gradient based update of mu based on
  //Kelley, C. T., Liao, L. Z., Qi, L., Chu, M. T., Reese, J. P., & Winton, C. (2009)
  //Projected pseudotransient continuation. SIAM Journal on Numerical Analysis, 46(6), 3071.
  if(reg_update_ == grad_based)
  {
    //get jacobian:
    Epetra_SerialDenseVector Ji(np_);
    for (int i=0; i<np_; i++)
      for (int j=0; j<nmp_; j++)
        Ji[i]+= rcurve[j]*J(j,i);

    error_grad_ = Ji.Norm2();

    if (!numb_run_) error_grad_o_ = error_grad_;
    //Adjust training parameter based on gradient
    // check for a descent step in general
    if (error_ < error_o_)
    {
      //update mu_ only if error in df/dp decreases
      if (error_grad_ < error_grad_o_)
        mu_ *= (error_grad_/error_grad_o_);
      // update parameters
      for (int i=0;i<np_;i++)
          p_[i] += delta_p[i];
    }
    else
      std::cout << "WARNING: MAT Params not updated! No descent direction" << std::endl;
  }
  else
  // res_based update
  {
    // update params no matter what
    for (int i=0;i<np_;i++)
      p_[i] += delta_p[i];
    if (!numb_run_) error_o_ = error_;

    //Adjust training parameter based on residuum of objective function ()
    mu_ *= (error_/error_o_);
  }

  // return cmatrix to previous size and zero out
  cmatrix.Shape(nmp_,np_+1);

  PrintStorage(cmatrix, delta_p);

  if (check_neg_params_) CheckOptStep();
  return;
}

/*----------------------------------------------------------------------*/
/* */
Epetra_SerialDenseVector STR::GenInvAnalysis::CalcCvector(bool outputtofile)
{
  int myrank = discret_->Comm().MyPID();
  // get input parameter lists
  const Teuchos::ParameterList& ioflags
    = DRT::Problem::Instance()->IOParams();
  const Teuchos::ParameterList& sdyn
    = DRT::Problem::Instance()->StructuralDynamicParams();
  Teuchos::ParameterList xparams;
  xparams.set<FILE*>("err file", DRT::Problem::Instance()->ErrorFile()->Handle());
  xparams.set<int>("REDUCED_OUTPUT",0);
  // create time integrator
  sti_ = TimIntCreate(ioflags, sdyn, xparams, discret_, solver_, solver_, output_);
  if (sti_ == Teuchos::null) dserror("Failed in creating integrator.");
  // initialize time loop / Attention the Functions give back the
  // time and the step not timen and stepn value that is why we have
  // to use < instead of <= for the while loop
  double time = sti_->GetTime();
  const double timemax = sti_->GetTimeEnd();
  int step = sti_->GetStep();
  int writestep=0;
  const int stepmax = sti_->GetTimeNumStep();
  Epetra_SerialDenseVector cvector(nmp_);

  // time loop
  while ( (time < timemax) && (step < stepmax) )
  {
    // integrate time step
    // after this step we hold disn_, etc
    sti_->IntegrateStep();

    // calculate stresses, strains, energies
    sti_->PrepareOutput();

    // update displacements, velocities, accelerations
    // after this call we will have disn_==dis_, etc
    sti_->UpdateStepState();

    // update time and step
    sti_->UpdateStepTime();

    // Update Element
    sti_->UpdateStepElement();

    // print info about finished time step
    sti_->PrintStep();

    // write output
    if (outputtofile) sti_->OutputStep();

    // get current time ...
    time = sti_->GetTime();
    // ... and step
    step = sti_->GetStep();

    // get the displacements of the monitored timesteps
    {
//      std::cout << "time = " << time << std::endl ;
//      std::cout << "timestep = " << timesteps_[writestep] << std::endl ;
      if (abs(time-timesteps_[writestep]) < 1.0e-5)
      {
        Epetra_SerialDenseVector cvector_arg = GetCalculatedCurve(*(sti_->DisNew()));
        if (!myrank)
          for (int j=0; j<ndofs_; ++j)
            cvector[writestep*ndofs_+j] = cvector_arg[j];
        writestep+=1;
      }

      // check if timestepsize is smaller than the tolerance above
      const double deltat = sti_->Dt();
      if (deltat < 1.0e-5)
        dserror("your time step size is too small, you will have problems with the monitored steps, thus adapt the tolerance");
    }
  }

  if (!(writestep*ndofs_==nmp_)) dserror("# of monitored timesteps does not match # of timesteps extracted from the simulation ");

  return cvector;
}


/*----------------------------------------------------------------------*/
/* nested parallelity version of the same method            mwgee 05/12 */
Epetra_SerialDenseVector STR::GenInvAnalysis::CalcCvector(
                              bool outputtofile,
                              Teuchos::RCP<COMM_UTILS::NestedParGroup> group)
{
  Teuchos::RCP<Epetra_Comm> lcomm = group->LocalComm();
  Teuchos::RCP<Epetra_Comm> gcomm = group->GlobalComm();
  const int lmyrank  = lcomm->MyPID();
  //const int gmyrank  = gcomm->MyPID();
  const int groupid  = group->GroupId();
  //const int ngroup   = group->NumGroups();

  // get input parameter lists (ioflags as deep copy to modify it)
  Teuchos::ParameterList ioflags = DRT::Problem::Instance()->IOParams();
  if (groupid != 0) ioflags.set("STDOUTEVRY",0);

  const Teuchos::ParameterList& sdyn
    = DRT::Problem::Instance()->StructuralDynamicParams();

  Teuchos::ParameterList xparams;
  xparams.set<FILE*>("err file", DRT::Problem::Instance()->ErrorFile()->Handle());
  xparams.set<int>("REDUCED_OUTPUT",0);

  // create time integrator
  sti_ = TimIntCreate(ioflags, sdyn, xparams, discret_, solver_, solver_, output_);
  if (sti_ == Teuchos::null) dserror("Failed in creating integrator.");
  fflush(stdout);
  gcomm->Barrier();

  // initialize time loop / Attention the Functions give back the
  // time and the step not timen and stepn value that is why we have
  // to use < instead of <= for the while loop
  double time = sti_->GetTime();
  const double timemax = sti_->GetTimeEnd();
  int step = sti_->GetStep();
  int writestep=0;
  const int stepmax = sti_->GetTimeNumStep();
  Epetra_SerialDenseVector cvector(nmp_);

  // time loop
  while ( (time < timemax) && (step < stepmax) )
  {
    // integrate time step
    // after this step we hold disn_, etc
    sti_->IntegrateStep();

    // calculate stresses, strains, energies
    sti_->PrepareOutput();

    // update displacements, velocities, accelerations
    // after this call we will have disn_==dis_, etc
    sti_->UpdateStepState();

    // update time and step
    sti_->UpdateStepTime();

    // Update Element
    sti_->UpdateStepElement();

    // print info about finished time step
    sti_->PrintStep();

    // write output
    if (outputtofile) sti_->OutputStep();

    // get current time ...
    time = sti_->GetTime();
    // ... and step
    step = sti_->GetStep();

    // get the displacements of the monitored timesteps
    {
//      std::cout << "time = " << time << std::endl ;
//      std::cout << "timestep = " << timesteps_[writestep] << std::endl ;

      if (abs(time-timesteps_[writestep]) < 1.0e-5)
      {
        Epetra_SerialDenseVector cvector_arg = GetCalculatedCurve(*(sti_->DisNew()));
        if (!lmyrank)
          for (int j=0; j<ndofs_; ++j)
            cvector[writestep*ndofs_+j] = cvector_arg[j];
        writestep+=1;
      }

      // check if timestepsize is smaller than the tolerance above
      const double deltat = sti_->Dt();
      if (deltat < 1.0e-5)
        dserror("your time step size is too small, you will have problems with the monitored steps, thus adapt the tolerance");
    }
  }

  if (!(writestep*ndofs_==nmp_)) dserror("# of monitored timesteps does not match # of timesteps extracted from the simulation ");

  return cvector;
}

/*----------------------------------------------------------------------*/
/* */
Epetra_SerialDenseVector STR::GenInvAnalysis::GetCalculatedCurve(Epetra_Vector& disp)
{
  int myrank = discret_->Comm().MyPID();

  // values observed at current time step
  Epetra_SerialDenseVector lcvector_arg(ndofs_);
  Epetra_SerialDenseVector gcvector_arg(ndofs_);
  int count=0;
  for (int i=0; i<nnodes_; ++i)
  {
    int gnode = nodes_[i];
    if (!discret_->NodeRowMap()->MyGID(gnode))
    {
      count += (int)dofs_[i].size();
      continue;
    }
    DRT::Node* node = discret_->gNode(gnode);
    if (!node) dserror("Cannot find node on this proc");
    if (myrank != node->Owner())
    {
      count += (int)dofs_[i].size();
      continue;
    }
    for (int j=0; j<(int)dofs_[i].size(); ++j)
    {
      int ldof = dofs_[i][j];
      int gdof = discret_->Dof(node,ldof);
      if (!disp.Map().MyGID(gdof)) dserror("Cannot find dof on this proc");
      lcvector_arg[count+j] = disp[disp.Map().LID(gdof)];
    }
    count += (int)dofs_[i].size();
  }
  discret_->Comm().SumAll(&lcvector_arg[0],&gcvector_arg[0],ndofs_);

  return gcvector_arg;
}

/*----------------------------------------------------------------------*/
/* only global proc 0 comes in here! mwgee 5/2012 */
void STR::GenInvAnalysis::PrintStorage(Epetra_SerialDenseMatrix cmatrix, Epetra_SerialDenseVector delta_p)
{
  Teuchos::RCP<COMM_UTILS::NestedParGroup> group = DRT::Problem::Instance()->GetNPGroup();
  Teuchos::RCP<Epetra_Comm> lcomm = group->LocalComm();
  Teuchos::RCP<Epetra_Comm> gcomm = group->GlobalComm();
  const int gmyrank  = group->GlobalComm()->MyPID();
  if (gmyrank != 0) dserror("Only gmyrank=0 is supposed to be in here, but is not, gmyrank=%d",gmyrank);

  // storing current material parameters
  // storing current material parameter increments
  p_s_.Reshape(numb_run_+1,  np_);
  delta_p_s_.Reshape(numb_run_+1,  np_);
  for (int i=0; i<np_; i++)
  {
    p_s_(numb_run_, i)=p_(i);
    delta_p_s_(numb_run_, i)=delta_p(i);
  }

  // this memory is going to explode, do we really need this? mwgee
  //ccurve_s_.Reshape(nmp_,  numb_run_+1);
  //for (int i=0; i<nmp_; i++)
  //  ccurve_s_(i, numb_run_)= cmatrix(i, cmatrix.ColDim()-1);

  mu_s_.Resize(numb_run_+1);
  mu_s_(numb_run_)=mu_;

  error_s_.Resize(numb_run_+1);
  error_s_(numb_run_) = error_;
  error_grad_s_.Resize(numb_run_+1);
  error_grad_s_(numb_run_) = error_grad_;
  // print error and parameter

  if (gmyrank==0) // this if should actually not be necessary since there is only gproc 0 in here
  {
      std::cout << std::endl;
      printf("################################################");
      printf("##############################################\n");
      printf("############################ Inverse Analysis ##");
      printf("##############################################\n");
      printf("################################### run ########");
      printf("##############################################\n");
      printf("################################### %3i ########",  numb_run_);
      printf("##############################################\n");
      printf("################################################");
      printf("##############################################\n");

      for (int i=0; i < numb_run_+1; i++)
      {
        printf("Error: ");
        printf("%10.3e", error_s_(i));
        printf("\tGrad_error: ");
        printf("%10.3e", error_grad_s_(i));
        printf("\tParameter: ");
        for (int j=0; j < delta_p.Length(); j++)
          printf("%10.3e", p_s_(i, j));
        //printf("\tDelta_p: ");
        //for (int j=0; j < delta_p.Length(); j++)
        //  printf("%10.3e", delta_p_s_(i, j));
        printf("\tmu: ");
        printf("%10.3e", mu_s_(i));
        printf("\n");
      }

// I don't like this printout, there is no legend. Also, its sorted in x and y
// which is absolutely not guaranteed to be true upon input. So its potentially mixed!
// I just leave it here because somebody might need it?
// I commented this out becauses it yields columns of zeros only anyway
// mwgee
#if 0
      printf("\n");
      for (int i=0; i < nmp_/2.; i++)
      {
        printf(" %10.5f ",  mcurve_(i*2));
        if (numb_run_<15)
        {
          for (int j=0; j<numb_run_+1; j++)
            printf(" %10.5f ",  ccurve_s_((i)*2, j));
        }
        else
        {
          for (int j=numb_run_-14; j<numb_run_+1; j++)
            printf(" %10.5f ",  ccurve_s_((i)*2, j));
        }
        printf("\n");
      }

      printf("\n");

      for (int i=0; i < nmp_/2.; i++)
      {
        printf(" %10.5f ",  mcurve_((i)*2+1));
        if (numb_run_<15)
        {
          for (int j=0; j<numb_run_+1; j++)
            printf(" %10.5f ",  ccurve_s_((i)*2+1, j));
        }
        else
        {
          for (int j=numb_run_-14; j<numb_run_+1; j++)
            printf(" %10.5f ",  ccurve_s_((i)*2+1, j));
        }
        printf("\n");
      }

      printf("################################################");
      printf("##############################################\n");
      std::cout << std::endl;
#endif
      printf("\n"); fflush(stdout);
  }

  return;
}


/*----------------------------------------------------------------------*/
/* only gmyrank=0 comes in here */
void STR::GenInvAnalysis::PrintFile()
{
  //FILE * cxFile;
  //FILE * cyFile;
  FILE * pFile;

  std::string name = DRT::Problem::Instance()->OutputControlFile()->FileName();
  name.append(filename_);

  if (name.rfind("_run_")!=std::string::npos)
  {
    size_t pos = name.rfind("_run_");
    if (pos==std::string::npos)
      dserror("inconsistent file name");
    name = name.substr(0, pos);
  }

  std::string gp     = name+"_plot.gp";
  std::string xcurve = name+"_Curve_x.txt";
  std::string ycurve = name+"_Curve_y.txt";
  std::string para   = name+"_Para.txt";

#if 0 // this only produces columns of zeros anyway
// it will also burst memory with ccurve_s_ pretty quickly for larger problems
  cxFile = fopen((xcurve).c_str(), "w");
  for (int i=0; i < nmp_/2.; i++)
  {
    fprintf(cxFile, " %10.5f ,",  mcurve_(i*2));
    for (int j=0; j<numb_run_; j++)
      fprintf(cxFile, " %10.5f ,",  ccurve_s_(i*2, j));
    fprintf(cxFile, "\n");
  }
  fclose(cxFile);

  cyFile = fopen((ycurve).c_str(), "w");
  for (int i=0; i < nmp_/2.; i++)
  {
    fprintf(cyFile, " %10.5f ,",  mcurve_((i)*2+1));
    for (int j=0; j<numb_run_; j++)
      fprintf(cyFile, " %10.5f ,",  ccurve_s_((i)*2+1, j));
    fprintf(cyFile, "\n");
  }
  fclose(cyFile);
#endif

  pFile  = fopen((para).c_str(), "w");
  fprintf(pFile, "#Error    #Grad_error       Parameter    Delta_p      mu \n");
  for (int i=0; i < numb_run_; i++)
  {
    fprintf(pFile, "%10.6f, ", error_s_(i));
    fprintf(pFile, "%10.6f, ", error_grad_s_(i));
    for (int j=0; j < np_; j++)
      fprintf(pFile, "%10.6f, ", p_s_(i, j));
    for (int j=0; j < np_; j++)
      fprintf(pFile, "%10.6f, ", delta_p_s_(i, j));
    fprintf(pFile, "%10.6f", mu_s_(i));
    fprintf(pFile, "\n");
  }
  fclose(pFile);

  numb_run_=numb_run_-1;
}


//-----------------------------------------------------------------------------------
void STR::GenInvAnalysis::ReadInParameters()
{
  const int myrank = discret_->Comm().MyPID();

  for (unsigned prob=0; prob<DRT::Problem::NumInstances(); ++prob)
  {
    const std::map<int,Teuchos::RCP<MAT::PAR::Material> >& mats = *DRT::Problem::Instance(prob)->Materials()->Map();
    std::set<int> mymatset = matset_[prob];

    unsigned int overallnummat = mats.size();
    if (mymatset.size() > 0 && overallnummat > mymatset.size()) overallnummat = mymatset.size();
    if (myrank == 0) printf("No. material laws/summands considered : %d\n", overallnummat);

    std::map<int,Teuchos::RCP<MAT::PAR::Material> >::const_iterator curr;

    for (curr=mats.begin(); curr != mats.end(); ++curr)
    {
      const Teuchos::RCP<MAT::PAR::Material> actmat = curr->second;

      switch(actmat->Type())
      {
      case INPAR::MAT::m_aaaneohooke:
      {
        if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
        {
          MAT::PAR::AAAneohooke* params = dynamic_cast<MAT::PAR::AAAneohooke*>(actmat->Parameter());
          if (!params) dserror("Cannot cast material parameters");
          const int j = p_.Length();
          p_.Resize(j+2);
          p_(j)   = params->GetParameter(params->young,-1);
          p_(j+1)     = params->GetParameter(params->beta,-1);
          //p_(j+2) = params->nue_; // need also change resize above to invoke
          //nue
        }
      }
      break;
      case INPAR::MAT::m_neohooke:
      {
        if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
        {
          MAT::PAR::NeoHooke* params = dynamic_cast<MAT::PAR::NeoHooke*>(actmat->Parameter());
          if (!params) dserror("Cannot cast material parameters");
          const int j = p_.Length();
          p_.Resize(j+2);
          p_(j)   = params->youngs_;
          p_(j+1) = params->poissonratio_;
        }
      }
      break;
      case INPAR::MAT::m_constraintmixture:
      {
        if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
        {
          MAT::PAR::ConstraintMixture* params = dynamic_cast<MAT::PAR::ConstraintMixture*>(actmat->Parameter());
          if (!params) dserror("Cannot cast material parameters");
          const int j = p_.Length();
          p_.Resize(j+1);
          //p_(j)   = params->mue_;
          //p_(j+1) = params->k1_;
          //p_(j+2) = params->k2_;
          //p_(j) = params->prestretchcollagen_;
          p_(j) = params->GetParameter(params->growthfactor,-1);
        }
      }
      break;
      case INPAR::MAT::m_growth_linear:
      {
        if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
        {
          MAT::PAR::GrowthLawLinear* params = dynamic_cast<MAT::PAR::GrowthLawLinear*>(actmat->Parameter());
          if (!params) dserror("Cannot cast material parameters");
          const int j = p_.Length();
          p_.Resize(j+1);
          p_(j) = params->kthetaplus_;
        }
      }
      break;
      case INPAR::MAT::m_elasthyper:
      {
        MAT::PAR::ElastHyper* params = dynamic_cast<MAT::PAR::ElastHyper*>(actmat->Parameter());
        if (!params) dserror("Cannot cast material parameters");
        const int nummat               = params->nummat_;
        const std::vector<int>* matids = params->matids_;
        for (int i=0; i<nummat; ++i)
        {
          const int id = (*matids)[i];

          if (mymatset.size()==0 or mymatset.find(id)!=mymatset.end())
          {
            const Teuchos::RCP<MAT::PAR::Material> actelastmat = mats.find(id)->second;
            switch (actelastmat->Type())
            {
            case INPAR::MAT::mes_couplogneohooke:
            {
              filename_=filename_+"_couplogneohooke";
              const MAT::ELASTIC::PAR::CoupLogNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupLogNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->mue_;
              p_[j+1] = params2->lambda_;
              break;
            }
            case INPAR::MAT::mes_coupneohooke:
            {
              filename_=filename_+"_coupneohooke";
              const MAT::ELASTIC::PAR::CoupNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j] = params2->c_;
              p_[j+1] = params2->beta_;
              break;
            }
            case INPAR::MAT::mes_coupblatzko:
            {
              filename_=filename_+"_coupblatzko";
              const MAT::ELASTIC::PAR::CoupBlatzKo* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupBlatzKo*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->mue_;
              p_[j+1] = (1./(1.-2.*params2->nue_))-1.;
              //p_[j+1] = params2->f_;
              break;
            }
            case INPAR::MAT::mes_isoneohooke:
            {
              filename_=filename_+"_isoneohooke";
              const MAT::ELASTIC::PAR::IsoNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->mue_;
              break;
            }
            case INPAR::MAT::mes_isoyeoh:
            {
              filename_=filename_+"_isoyeoh";
              const MAT::ELASTIC::PAR::IsoYeoh* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoYeoh*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+3);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              p_[j+2] = params2->c3_;
              break;
            }
            case INPAR::MAT::mes_isoquad:
            {
              filename_=filename_+"_isoquad";
              const MAT::ELASTIC::PAR::IsoQuad* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoQuad*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_isocub:
            {
              filename_=filename_+"_isocub";
              const MAT::ELASTIC::PAR::IsoCub* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoCub*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_iso1pow:
            {
              filename_=filename_+"_iso1pow";
              const MAT::ELASTIC::PAR::Iso1Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Iso1Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_iso2pow:
            {
              filename_=filename_+"_iso2pow";
              const MAT::ELASTIC::PAR::Iso2Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Iso2Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coup1pow:
            {
              filename_=filename_+"_coup1pow";
              const MAT::ELASTIC::PAR::Coup1Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Coup1Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coup2pow:
            {
              filename_=filename_+"_coup2pow";
              const MAT::ELASTIC::PAR::Coup2Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Coup2Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coupmooneyrivlin:
            {
              filename_=filename_+"_coupmooneyrivlin";
              const MAT::ELASTIC::PAR::CoupMooneyRivlin* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupMooneyRivlin*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+3);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              p_[j+2] = params2->c3_;
              break;
            }
            case INPAR::MAT::mes_isoexpopow:
            {
              filename_=filename_+"_isoexpopow";
              const MAT::ELASTIC::PAR::IsoExpoPow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoExpoPow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->k1_;
              p_[j+1] = params2->k2_;
              break;
            }
            case INPAR::MAT::mes_isomooneyrivlin:
            {
              filename_=filename_+"_isomooneyrivlin";
              const MAT::ELASTIC::PAR::IsoMooneyRivlin* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoMooneyRivlin*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              break;
            }
            case INPAR::MAT::mes_volsussmanbathe:
            {
              filename_=filename_+"_volsussmanbathe";
              const MAT::ELASTIC::PAR::VolSussmanBathe* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolSussmanBathe*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->kappa_;
              break;
            }
            case INPAR::MAT::mes_volpenalty:
            {
              filename_=filename_+"_volpenalty";
              const MAT::ELASTIC::PAR::VolPenalty* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolPenalty*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->eps_;
              p_[j+1]   = params2->gam_;
              break;
            }
            case INPAR::MAT::mes_vologden:
            {
              filename_=filename_+"_vologden";
              const MAT::ELASTIC::PAR::VolOgden* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolOgden*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->kappa_;
              //p_[j+1] = params2->beta_;
              break;
            }
            default:
              dserror("cannot deal with this material");
              break;
            }
          }
        }
        break;
      }
      case INPAR::MAT::m_viscoelasthyper:
      {
        MAT::PAR::ViscoElastHyper* params = dynamic_cast<MAT::PAR::ViscoElastHyper*>(actmat->Parameter());
        if (!params) dserror("Cannot cast material parameters");
        const int nummat               = params->nummat_;
        const std::vector<int>* matids = params->matids_;
        for (int i=0; i<nummat; ++i)
        {
          const int id = (*matids)[i];

          if (mymatset.size()==0 or mymatset.find(id)!=mymatset.end())
          {
            const Teuchos::RCP<MAT::PAR::Material> actelastmat = mats.find(id)->second;
            switch (actelastmat->Type())
            {
            case INPAR::MAT::mes_couplogneohooke:
            {
              filename_=filename_+"_couplogneohooke";
              const MAT::ELASTIC::PAR::CoupLogNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupLogNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->mue_;
              p_[j+1] = params2->lambda_;
              break;
            }
            case INPAR::MAT::mes_coupneohooke:
            {
              filename_=filename_+"_coupneohooke";
              const MAT::ELASTIC::PAR::CoupNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j] = params2->c_;
              p_[j+1] = params2->beta_;
              break;
            }
            case INPAR::MAT::mes_coupblatzko:
            {
              filename_=filename_+"_coupblatzko";
              const MAT::ELASTIC::PAR::CoupBlatzKo* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupBlatzKo*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->mue_;
              p_[j+1] = (1./(1.-2.*params2->nue_))-1.;
              //p_[j+1] = params2->f_;
              break;
            }
            case INPAR::MAT::mes_isoneohooke:
            {
              filename_=filename_+"_isoneohooke";
              const MAT::ELASTIC::PAR::IsoNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->mue_;
              break;
            }
            case INPAR::MAT::mes_isoyeoh:
            {
              filename_=filename_+"_isoyeoh";
              const MAT::ELASTIC::PAR::IsoYeoh* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoYeoh*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+3);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              p_[j+2] = params2->c3_;
              break;
            }
            case INPAR::MAT::mes_isoquad:
            {
              filename_=filename_+"_isoquad";
              const MAT::ELASTIC::PAR::IsoQuad* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoQuad*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_isocub:
            {
              filename_=filename_+"_isocub";
              const MAT::ELASTIC::PAR::IsoCub* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoCub*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_iso1pow:
            {
              filename_=filename_+"_iso1pow";
              const MAT::ELASTIC::PAR::Iso1Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Iso1Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_iso2pow:
            {
              filename_=filename_+"_iso2pow";
              const MAT::ELASTIC::PAR::Iso2Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Iso2Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coup1pow:
            {
              filename_=filename_+"_coup1pow";
              const MAT::ELASTIC::PAR::Coup1Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Coup1Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coup2pow:
            {
              filename_=filename_+"_coup2pow";
              const MAT::ELASTIC::PAR::Coup2Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Coup2Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coupmooneyrivlin:
            {
              filename_=filename_+"_coupmooneyrivlin";
              const MAT::ELASTIC::PAR::CoupMooneyRivlin* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupMooneyRivlin*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+3);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              p_[j+2] = params2->c3_;
              break;
            }
            case INPAR::MAT::mes_isoexpopow:
            {
              filename_=filename_+"_isoexpopow";
              const MAT::ELASTIC::PAR::IsoExpoPow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoExpoPow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->k1_;
              p_[j+1] = params2->k2_;
              break;
            }
            case INPAR::MAT::mes_isomooneyrivlin:
            {
              filename_=filename_+"_isomooneyrivlin";
              const MAT::ELASTIC::PAR::IsoMooneyRivlin* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoMooneyRivlin*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              break;
            }
            case INPAR::MAT::mes_volsussmanbathe:
            {
              filename_=filename_+"_volsussmanbathe";
              const MAT::ELASTIC::PAR::VolSussmanBathe* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolSussmanBathe*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->kappa_;
              break;
            }
            case INPAR::MAT::mes_volpenalty:
            {
              filename_=filename_+"_volpenalty";
              const MAT::ELASTIC::PAR::VolPenalty* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolPenalty*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->eps_;
              p_[j+1]   = params2->gam_;
              break;
            }
            case INPAR::MAT::mes_vologden:
            {
              filename_=filename_+"_vologden";
              const MAT::ELASTIC::PAR::VolOgden* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolOgden*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->kappa_;
              //p_[j+1] = params2->beta_;
              break;
            }
            case INPAR::MAT::mes_isoratedep:
            {
              filename_=filename_+"_isoratedep";
              const MAT::ELASTIC::PAR::IsoRateDep* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoRateDep*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->n_;
              break;
            }
            default:
              dserror("cannot deal with this material");
              break;
            }
          }
        }
        break;
      }

      // at this level do nothing, its inside the INPAR::MAT::m_elasthyper block or an interface to a micro material
      case INPAR::MAT::mes_couplogneohooke:
      case INPAR::MAT::mes_coupneohooke:
      case INPAR::MAT::mes_coupblatzko:
      case INPAR::MAT::mes_isoneohooke:
      case INPAR::MAT::mes_isoyeoh:
      case INPAR::MAT::mes_isoquad:
      case INPAR::MAT::mes_isocub:
      case INPAR::MAT::mes_iso1pow:
      case INPAR::MAT::mes_iso2pow:
      case INPAR::MAT::mes_coup1pow:
      case INPAR::MAT::mes_coup2pow:
      case INPAR::MAT::mes_isoexpopow:
      case INPAR::MAT::mes_isomooneyrivlin:
      case INPAR::MAT::mes_coupmooneyrivlin:
      case INPAR::MAT::mes_volsussmanbathe:
      case INPAR::MAT::mes_volpenalty:
      case INPAR::MAT::mes_vologden:
      case INPAR::MAT::mes_isoratedep:
      case INPAR::MAT::m_struct_multiscale:
        break;
      case INPAR::MAT::m_viscogenmax:
      {
        MAT::PAR::ViscoGenMax* viscoparams = dynamic_cast<MAT::PAR::ViscoGenMax*>(actmat->Parameter());
        if (!viscoparams) dserror("Cannot cast material parameters");
        const int nummat               = viscoparams->nummat_;
        const std::vector<int>* matids = viscoparams->matids_;

        if (viscoparams->relax_isot_princ_ != 0)
        {
          int j = p_.Length();
          p_.Resize(j+1);
          p_[j] = viscoparams->relax_isot_princ_;
//            p_[j+1] = viscoparams->beta_isot_princ_;
        }
        if (viscoparams->relax_isot_mod_vol_ != 0)
        {
          int j = p_.Length();
          p_.Resize(j+1);
          p_[j] = viscoparams->relax_isot_mod_vol_;
//            p_[j+1] = viscoparams->beta_isot_mod_vol_;
        }
//          if (viscoparams->relax_isot_mod_isoc_ != 0)
//          {
//            int j = p_.Length();
//            p_.Resize(j+1);
//            p_[j] = viscoparams->relax_isot_mod_isoc_;
////            p_[j+1] = viscoparams->beta_isot_mod_isoc_;
//          }
        if (viscoparams->relax_anisot_princ_ != 0)
        {
          int j = p_.Length();
          p_.Resize(j+1);
          p_[j] = viscoparams->relax_anisot_princ_;
//            p_[j+1] = viscoparams->beta_anisot_princ_;
        }

        for (int i=0; i<nummat; ++i)
        {
          const int id = (*matids)[i];

          if (mymatset.size()==0 or mymatset.find(id)!=mymatset.end())
          {
            const Teuchos::RCP<MAT::PAR::Material> actelastmat = mats.find(id)->second;
            switch (actelastmat->Type())
            {
            case INPAR::MAT::mes_couplogneohooke:
            {
              filename_=filename_+"_couplogneohooke";
              const MAT::ELASTIC::PAR::CoupLogNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupLogNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->mue_;
              p_[j+1] = params2->lambda_;
              break;
            }
            case INPAR::MAT::mes_coupneohooke:
            {
              filename_=filename_+"_coupneohooke";
              const MAT::ELASTIC::PAR::CoupNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j] = params2->c_;
              p_[j+1] = params2->beta_;
              break;
            }
            case INPAR::MAT::mes_coupblatzko:
            {
              filename_=filename_+"_coupblatzko";
              const MAT::ELASTIC::PAR::CoupBlatzKo* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupBlatzKo*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->mue_;
              p_[j+1] = (1./(1.-2.*params2->nue_))-1.;
              //p_[j+1] = params2->f_;
              break;
            }
            case INPAR::MAT::mes_isoneohooke:
            {
              filename_=filename_+"_isoneohooke";
              const MAT::ELASTIC::PAR::IsoNeoHooke* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoNeoHooke*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->mue_;
              break;
            }
            case INPAR::MAT::mes_isoyeoh:
            {
              filename_=filename_+"_isoyeoh";
              const MAT::ELASTIC::PAR::IsoYeoh* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoYeoh*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+3);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              p_[j+2] = params2->c3_;
              break;
            }
            case INPAR::MAT::mes_isoquad:
            {
              filename_=filename_+"_isoquad";
              const MAT::ELASTIC::PAR::IsoQuad* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoQuad*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_isocub:
            {
              filename_=filename_+"_isocub";
              const MAT::ELASTIC::PAR::IsoCub* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoCub*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_iso1pow:
            {
              filename_=filename_+"_iso1pow";
              const MAT::ELASTIC::PAR::Iso1Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Iso1Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_iso2pow:
            {
              filename_=filename_+"_iso2pow";
              const MAT::ELASTIC::PAR::Iso2Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Iso2Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coup1pow:
            {
              filename_=filename_+"_coup1pow";
              const MAT::ELASTIC::PAR::Coup1Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Coup1Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coup2pow:
            {
              filename_=filename_+"_coup2pow";
              const MAT::ELASTIC::PAR::Coup2Pow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::Coup2Pow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->c_;
              break;
            }
            case INPAR::MAT::mes_coupmooneyrivlin:
            {
              filename_=filename_+"_coupmooneyrivlin";
              const MAT::ELASTIC::PAR::CoupMooneyRivlin* params2 = dynamic_cast<const MAT::ELASTIC::PAR::CoupMooneyRivlin*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+3);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              p_[j+2] = params2->c3_;
              break;
            }
            case INPAR::MAT::mes_isoexpopow:
            {
              filename_=filename_+"_isoexpopow";
              const MAT::ELASTIC::PAR::IsoExpoPow* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoExpoPow*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->k1_;
              p_[j+1] = params2->k2_;
              break;
            }
            case INPAR::MAT::mes_isomooneyrivlin:
            {
              filename_=filename_+"_isomooneyrivlin";
              const MAT::ELASTIC::PAR::IsoMooneyRivlin* params2 = dynamic_cast<const MAT::ELASTIC::PAR::IsoMooneyRivlin*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->c1_;
              p_[j+1] = params2->c2_;
              break;
            }
            case INPAR::MAT::mes_volsussmanbathe:
            {
              filename_=filename_+"_volsussmanbathe";
              const MAT::ELASTIC::PAR::VolSussmanBathe* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolSussmanBathe*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->kappa_;
              break;
            }
            case INPAR::MAT::mes_volpenalty:
            {
              filename_=filename_+"_volpenalty";
              const MAT::ELASTIC::PAR::VolPenalty* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolPenalty*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+2);
              p_[j]   = params2->eps_;
              p_[j+1]   = params2->gam_;
              break;
            }
            case INPAR::MAT::mes_vologden:
            {
              filename_=filename_+"_vologden";
              const MAT::ELASTIC::PAR::VolOgden* params2 = dynamic_cast<const MAT::ELASTIC::PAR::VolOgden*>(actelastmat->Parameter());
              int j = p_.Length();
              p_.Resize(j+1);
              p_[j]   = params2->kappa_;
              //p_[j+1] = params2->beta_;
              break;
            }
            default:
              dserror("cannot deal with this material");
              break;
            }
          }
        }
        break;
      }
      default:
        dserror("Unknown type of material");
        break;
      }
    }
  }
  return;
}
//--------------------------------------------------------------------------------------
void STR::GenInvAnalysis::SetParameters(Epetra_SerialDenseVector p_cur)
{
  // check whether there is a micro scale
  Teuchos::RCP<Epetra_Comm> subcomm = DRT::Problem::Instance(0)->GetNPGroup()->SubComm();
  if(subcomm != Teuchos::null)
  {
    // tell supporting procs that material parameters have to be set
    int task[2] = {7, np_};
    subcomm->Broadcast(task, 2, 0);
    // broadcast p_cur to the micro scale
    subcomm->Broadcast(&p_cur[0], np_, 0);
  }

  // write new material parameter
  for (unsigned prob=0; prob<DRT::Problem::NumInstances(); ++prob)
  {
    std::set<int> mymatset = matset_[prob];

    // in case of micro scale: broadcast mymatset once per problem instance
    if(subcomm != Teuchos::null)
    {
      LINALG::GatherAll<int>(mymatset, *subcomm);
    }

    // material parameters are set for the current problem instance
    STR::SetMaterialParameters(prob, p_cur, mymatset);
  }
  return;
}

void STR::SetMaterialParameters(int prob, Epetra_SerialDenseVector& p_cur, std::set<int>& mymatset)
{
  Teuchos::RCP<COMM_UTILS::NestedParGroup> group = DRT::Problem::Instance()->GetNPGroup();
  Teuchos::RCP<Epetra_Comm> lcomm = group->LocalComm();
  Teuchos::RCP<Epetra_Comm> gcomm = group->GlobalComm();
  const int lmyrank  = lcomm->MyPID();
  //const int gmyrank  = gcomm->MyPID();
  const int groupid  = group->GroupId();
  //const int ngroup   = group->NumGroups();

  // loop all materials in problem
  const std::map<int,Teuchos::RCP<MAT::PAR::Material> >& mats = *DRT::Problem::Instance(prob)->Materials()->Map();
  int j=0;
  std::map<int,Teuchos::RCP<MAT::PAR::Material> >::const_iterator curr;
  for (curr=mats.begin(); curr != mats.end(); ++curr)
  {
    const Teuchos::RCP<MAT::PAR::Material> actmat = curr->second;

    switch(actmat->Type())
    {
    case INPAR::MAT::m_aaaneohooke:
    {
      if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
      {
        MAT::PAR::AAAneohooke* params = dynamic_cast<MAT::PAR::AAAneohooke*>(actmat->Parameter());
        if (!params) dserror("Cannot cast material parameters");
        //Epetra_Map dummy_map(1,1,0,*(DRT::Problem::Instance()->GetNPGroup()->LocalComm()));
        Epetra_Map dummy_map(1,1,0,*gcomm);

        Teuchos::RCP <Epetra_Vector> parameter= Teuchos::rcp(new Epetra_Vector(dummy_map,true));
        parameter->PutScalar(double (p_cur[j]));
        params->SetParameter(params->young ,parameter);
        parameter->PutScalar(p_cur[j+1]);
        params->SetParameter(params->beta,parameter);
        if (lmyrank==0)   printf("NPGroup %3d: ",groupid);
        if (lmyrank == 0) printf("MAT::PAR::AAAneohooke %20.15e %20.15e\n",p_cur[j],p_cur[j+1]);
        j += 2;
      }
    }
    break;
    case INPAR::MAT::m_neohooke:
    {
      if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
      {
        MAT::PAR::NeoHooke* params = dynamic_cast<MAT::PAR::NeoHooke*>(actmat->Parameter());
        if (!params) dserror("Cannot cast material parameters");
        // This is a tiny little bit brutal!!!
        const_cast<double&>(params->youngs_)       = p_cur[j];
        const_cast<double&>(params->poissonratio_) = p_cur[j+1];
        if (lmyrank==0)   printf("NPGroup %3d: ",groupid);
        if (lmyrank == 0) printf("MAT::PAR::NeoHooke %20.15e %20.15e\n",params->youngs_,params->poissonratio_);
        j += 2;
      }
    }
    break;
    case INPAR::MAT::m_elasthyper:
    {
      MAT::PAR::ElastHyper* params = dynamic_cast<MAT::PAR::ElastHyper*>(actmat->Parameter());
      if (!params) dserror("Cannot cast material parameters");
      const int nummat               = params->nummat_;
      const std::vector<int>* matids = params->matids_;
      for (int i=0; i<nummat; ++i)
      {
        const int id = (*matids)[i];

        if (mymatset.size()==0 or mymatset.find(id)!=mymatset.end())
        {
          const Teuchos::RCP<MAT::PAR::Material> actelastmat = mats.find(id)->second;
          switch (actelastmat->Type())
          {
          case INPAR::MAT::mes_couplogneohooke:
          {
            MAT::ELASTIC::PAR::CoupLogNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupLogNeoHooke*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            params2->SetLambda(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_coupneohooke:
          {
            MAT::ELASTIC::PAR::CoupNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupNeoHooke*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            params2->SetBeta(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_coupblatzko:
          {
            MAT::ELASTIC::PAR::CoupBlatzKo* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupBlatzKo*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            //params2->SetF(abs(p_cur(j+1)));
            params2->SetNue((abs(p_cur(j+1)))/(2.*(abs(p_cur(j+1))+1.)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isoneohooke:
          {
            MAT::ELASTIC::PAR::IsoNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoNeoHooke*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_isoyeoh:
          {
            MAT::ELASTIC::PAR::IsoYeoh* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoYeoh*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            params2->SetC3(abs(p_cur(j+2)));
            j = j+3;
            break;
          }
          case INPAR::MAT::mes_isoquad:
          {
            MAT::ELASTIC::PAR::IsoQuad* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoQuad*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_isocub:
          {
            MAT::ELASTIC::PAR::IsoCub* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoCub*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_iso1pow:
          {
            MAT::ELASTIC::PAR::Iso1Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Iso1Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_iso2pow:
          {
            MAT::ELASTIC::PAR::Iso2Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Iso2Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coup1pow:
          {
            MAT::ELASTIC::PAR::Coup1Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Coup1Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coup2pow:
          {
            MAT::ELASTIC::PAR::Coup2Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Coup2Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coupmooneyrivlin:
          {
            MAT::ELASTIC::PAR::CoupMooneyRivlin* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupMooneyRivlin*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            params2->SetC3(abs(p_cur(j+2)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isoexpopow:
          {
            MAT::ELASTIC::PAR::IsoExpoPow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoExpoPow*>(actelastmat->Parameter());
            params2->SetK1(abs(p_cur(j)));
            params2->SetK2(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isomooneyrivlin:
          {
            MAT::ELASTIC::PAR::IsoMooneyRivlin* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoMooneyRivlin*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_volsussmanbathe:
          {
            MAT::ELASTIC::PAR::VolSussmanBathe* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolSussmanBathe*>(actelastmat->Parameter());
            params2->SetKappa(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_volpenalty:
          {
            MAT::ELASTIC::PAR::VolPenalty* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolPenalty*>(actelastmat->Parameter());
            params2->SetEpsilon(abs(p_cur(j)));
            params2->SetGamma(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_vologden:
          {
            MAT::ELASTIC::PAR::VolOgden* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolOgden*>(actelastmat->Parameter());
            params2->SetKappa(abs(p_cur(j)));
            //params2->SetBeta(abs(p_cur(j+1)));
            j = j+1;
            break;
          }
          default:
            dserror("cannot deal with this material");
          }
        }
      }
    }
    break;
    case INPAR::MAT::mes_couplogneohooke:
    case INPAR::MAT::mes_coupneohooke:
    case INPAR::MAT::mes_coupblatzko:
    case INPAR::MAT::mes_isoneohooke:
    case INPAR::MAT::mes_isoyeoh:
    case INPAR::MAT::mes_isoquad:
    case INPAR::MAT::mes_isocub:
    case INPAR::MAT::mes_iso1pow:
    case INPAR::MAT::mes_iso2pow:
    case INPAR::MAT::mes_coup1pow:
    case INPAR::MAT::mes_coup2pow:
    case INPAR::MAT::mes_isoexpopow:
    case INPAR::MAT::mes_isomooneyrivlin:
    case INPAR::MAT::mes_coupmooneyrivlin:
    case INPAR::MAT::mes_volsussmanbathe:
    case INPAR::MAT::mes_volpenalty:
    case INPAR::MAT::mes_vologden:
    case INPAR::MAT::mes_isoratedep:
    case INPAR::MAT::m_struct_multiscale:
      break;

    case INPAR::MAT::m_viscogenmax:
    {
      MAT::PAR::ViscoGenMax* viscoparams = dynamic_cast<MAT::PAR::ViscoGenMax*>(actmat->Parameter());
      if (!viscoparams) dserror("Cannot cast material parameters");
      const int nummat               = viscoparams->nummat_;
      const std::vector<int>* matids = viscoparams->matids_;

      if (viscoparams->relax_isot_princ_ != 0)
      {
        const_cast<double&> (viscoparams->relax_isot_princ_) = p_cur[j];
        if (lmyrank == 0)
        {
          printf("MAT::PAR::ViscoGenMax - tau_isot_princ = %20.15e \n",p_cur[j]);
        }
//            const_cast<double&> (viscoparams->beta_isot_princ_)  = p_cur[j+1];
//            j += 2;
        j += 1;
      }
      if (viscoparams->relax_isot_mod_vol_ != 0)
      {
        const_cast<double&> (viscoparams->relax_isot_mod_vol_) = p_cur[j];
//            if (viscoparams->relax_isot_mod_isoc_ != 0)
//            {
//              const_cast<double&> (viscoparams->relax_isot_mod_isoc_) = p_cur[j];
//              if (lmyrank == 0) printf("MAT::PAR::ViscoGenMax - tau_isot_mod_isoc = %20.15e \n",p_cur[j]);
//            }
//            viscoparams->SetTauIsoModVol(abs(p_cur(j)));
        if (lmyrank == 0) printf("MAT::PAR::ViscoGenMax - tau_isot_mod_vol = %20.15e \n",p_cur[j]);
//            const_cast<double&> (viscoparams->beta_isot_mod_vol_)  = p_cur[j+1];
//            j += 2;
        j += 1;
      }
//          if (viscoparams->relax_isot_mod_isoc_ != 0)
//          {
//            const_cast<double&> (viscoparams->relax_isot_mod_isoc_) = p_cur[j];
////            viscoparams->SetTauIsoModIsoc(abs(p_cur(j)));
//            if (lmyrank == 0) printf("MAT::PAR::ViscoGenMax - tau_isot_mod_iso = %20.15e \n",p_cur[j]);
////            const_cast<double&> (viscoparams->beta_isot_mod_isoc_)  = p_cur[j+1];
////            j += 2;
//            j += 1;
//          }
      if (viscoparams->relax_anisot_princ_ != 0)
      {
        const_cast<double&> (viscoparams->relax_anisot_princ_) = p_cur[j];
//            viscoparams->SetTauAnisoPri(abs(p_cur(j)));
        if (lmyrank == 0) printf("MAT::PAR::ViscoGenMax - tau_anisot_princ = %20.15e \n",p_cur[j]);
//            const_cast<double&> (viscoparams->beta_anisot_princ_)  = p_cur[j+1];
//            j += 2;
        j += 1;
      }
//      if (viscoparams->relax_anisot_mod_vol_ != 0)
//      {
//        const_cast<double&> (viscoparams->relax_anisot_mod_vol_) = p_cur[j];
////            viscoparams->SetTauAnisoModVol(abs(p_cur(j)));
//        if (lmyrank == 0) printf("MAT::PAR::ViscoGenMax - tau_anisot_mod_vol = %20.15e \n",p_cur[j]);
////            const_cast<double&> (viscoparams->beta_anisot_mod_vol_)  = p_cur[j+1];
////            j += 2;
//        j += 1;
//      }
//      if (viscoparams->relax_anisot_mod_isoc_ != 0)
//      {
//        const_cast<double&> (viscoparams->relax_anisot_mod_isoc_) = p_cur[j];
////            viscoparams->SetTauAnisoModIsoc(abs(p_cur(j)));
//        if (lmyrank == 0) printf("MAT::PAR::ViscoGenMax - tau_anisot_mod_isoc = %20.15e \n",p_cur[j]);
////            const_cast<double&> (viscoparams->beta_anisot_mod_isoc_)  = p_cur[j+1];
////            j += 2;
//        j += 1;
//      }

      for (int i=0; i<nummat; ++i)
      {
        const int id = (*matids)[i];
        if (mymatset.size()==0 or mymatset.find(id)!=mymatset.end())
        {
          const Teuchos::RCP<MAT::PAR::Material> actelastmat = mats.find(id)->second;
          switch (actelastmat->Type())
          {
          case INPAR::MAT::mes_couplogneohooke:
          {
            MAT::ELASTIC::PAR::CoupLogNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupLogNeoHooke*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            params2->SetLambda(abs(p_cur(j+1)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::CoupLogNeoHooke - Mue = %20.15e \n",p_cur[j]);
              printf("MAT::PAR::CoupLogNeoHooke - Lambda = %20.15e \n",p_cur[j+1]);
            }
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_coupneohooke:
          {
            MAT::ELASTIC::PAR::CoupNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupNeoHooke*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            params2->SetBeta(abs(p_cur(j+1)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::CoupNeoHooke - C = %20.15e \n",p_cur[j]);
              printf("MAT::PAR::CoupNeoHooke - Beta = %20.15e \n",p_cur[j+1]);
            }
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_coupblatzko:
          {
            MAT::ELASTIC::PAR::CoupBlatzKo* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupBlatzKo*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            //params2->SetF(abs(p_cur(j+1)));
            params2->SetNue((abs(p_cur(j+1)))/(2.*(abs(p_cur(j+1))+1.)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::CoupBlatzKo - Mue = %20.15e \n",p_cur[j]);
              //printf("MAT::PAR::CoupBlatzKo - Nue = %20.15e \n",p_cur[j+1]);
            }
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isoneohooke:
          {
            MAT::ELASTIC::PAR::IsoNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoNeoHooke*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::IsoNeoHooke - mue = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_isoyeoh:
          {
            MAT::ELASTIC::PAR::IsoYeoh* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoYeoh*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            params2->SetC3(abs(p_cur(j+2)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::IsoYeoh - C1 = %20.15e \n",p_cur[j]);
              printf("MAT::PAR::IsoYeoh - C2 = %20.15e \n",p_cur[j+1]);
              printf("MAT::PAR::IsoYeoh - C3 = %20.15e \n",p_cur[j+2]);
            }
            j = j+3;
            break;
          }
          case INPAR::MAT::mes_isoquad:
          {
            MAT::ELASTIC::PAR::IsoQuad* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoQuad*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            if (lmyrank == 0)
            {
            printf("MAT::PAR::IsoQuad - C2 = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_isocub:
          {
            MAT::ELASTIC::PAR::IsoCub* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoCub*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::IsoCub - C3 = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_iso1pow:
          {
            MAT::ELASTIC::PAR::Iso1Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Iso1Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::Iso1Pow - C = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_iso2pow:
          {
            MAT::ELASTIC::PAR::Iso2Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Iso2Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::Iso2Pow - C = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coup1pow:
          {
            MAT::ELASTIC::PAR::Coup1Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Coup1Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::Coup1Pow - C = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coup2pow:
          {
            MAT::ELASTIC::PAR::Coup2Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Coup2Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::Coup2Pow - C = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coupmooneyrivlin:
          {
            MAT::ELASTIC::PAR::CoupMooneyRivlin* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupMooneyRivlin*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            params2->SetC3(abs(p_cur(j+2)));
            if (lmyrank == 0)
            {
            printf("MAT::PAR::CoupMooneyRivlin - C1 = %20.15e \n",p_cur[j]);
            printf("MAT::PAR::CoupMooneyRivlin - C2 = %20.15e \n",p_cur[j+1]);
            printf("MAT::PAR::CoupMooneyRivlin - C3 = %20.15e \n",p_cur[j+2]);
            }
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isoexpopow:
          {
            MAT::ELASTIC::PAR::IsoExpoPow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoExpoPow*>(actelastmat->Parameter());
            params2->SetK1(abs(p_cur(j)));
            params2->SetK2(abs(p_cur(j+1)));
            if (lmyrank == 0)
             {
               printf("MAT::PAR::IsoExpoPow - K1 = %20.15e \n",p_cur[j]);
               printf("MAT::PAR::IsoExpoPow - K2 = %20.15e \n",p_cur[j+1]);
             }
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isomooneyrivlin:
          {
            MAT::ELASTIC::PAR::IsoMooneyRivlin* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoMooneyRivlin*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::IsoMooneyRivlin - C1 = %20.15e \n",p_cur[j]);
              printf("MAT::PAR::IsoMooneyRivlin - C2 = %20.15e \n",p_cur[j+1]);
            }
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_volsussmanbathe:
          {
            MAT::ELASTIC::PAR::VolSussmanBathe* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolSussmanBathe*>(actelastmat->Parameter());
            params2->SetKappa(abs(p_cur(j)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::VolSussmanBathe - Kappa = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_volpenalty:
          {
            MAT::ELASTIC::PAR::VolPenalty* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolPenalty*>(actelastmat->Parameter());
            params2->SetEpsilon(abs(p_cur(j)));
            params2->SetGamma(abs(p_cur(j+1)));
            if (lmyrank == 0)
            {
              printf("MAT::PAR::VolPenalty - Epsilon = %20.15e \n",p_cur[j]);
              printf("MAT::PAR::VolPenalty - Gamma = %20.15e \n",p_cur[j+1]);
            }
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_vologden:
          {
            MAT::ELASTIC::PAR::VolOgden* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolOgden*>(actelastmat->Parameter());
            params2->SetKappa(abs(p_cur(j)));
            //params2->SetBeta(abs(p_cur(j+1)));
            if (lmyrank == 0)
            {
            printf("MAT::PAR::VolOgden - Kappa = %20.15e \n",p_cur[j]);
            }
            j = j+1;
            break;
          }
          default:
            dserror("cannot deal with this material");
          }
        }
      }
    }
    break;
    case INPAR::MAT::m_constraintmixture:
    {
      if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
      {
        MAT::PAR::ConstraintMixture* params = dynamic_cast<MAT::PAR::ConstraintMixture*>(actmat->Parameter());
        if (!params) dserror("Cannot cast material parameters");
        // This is a tiny little bit brutal!!!
        //const_cast<double&>(params->mue_)  = p_cur[j];
        //const_cast<double&>(params->k1_)  = p_cur[j+1];
        //const_cast<double&>(params->k2_)  = p_cur[j+2];
        //const_cast<double&>(params->prestretchcollagen_) = p_cur[j];

        Epetra_Map dummy_map(1,1,0,*gcomm);
        Teuchos::RCP <Epetra_Vector> parameter= Teuchos::rcp(new Epetra_Vector(dummy_map,true));
        parameter->PutScalar(double (p_cur[j]));
        params->SetParameter(params->growthfactor, parameter);
        if (lmyrank==0)   printf("NPGroup %3d: ",groupid);
        if (lmyrank == 0) printf("MAT::PAR::ConstraintMixture %20.15e\n",p_cur[j]);
        j += 1;
      }
    }
    break;
    case INPAR::MAT::m_growth_linear:
    {
      if (mymatset.size()==0 or mymatset.find(actmat->Id())!=mymatset.end())
      {
        MAT::PAR::GrowthLawLinear* params = dynamic_cast<MAT::PAR::GrowthLawLinear*>(actmat->Parameter());
        if (!params) dserror("Cannot cast material parameters");
        // This is a tiny little bit brutal!!!
        const_cast<double&>(params->kthetaplus_) = p_cur[j];
        if (lmyrank==0)   printf("NPGroup %3d: ",groupid);
        if (lmyrank == 0) printf("MAT::PAR::Growth %20.15e\n",params->kthetaplus_);
        j += 1;
      }
    }
    break;
    case INPAR::MAT::m_viscoelasthyper:
    {
      MAT::PAR::ViscoElastHyper* params = dynamic_cast<MAT::PAR::ViscoElastHyper*>(actmat->Parameter());
      if (!params) dserror("Cannot cast material parameters");
      const int nummat               = params->nummat_;
      const std::vector<int>* matids = params->matids_;
      for (int i=0; i<nummat; ++i)
      {
        const int id = (*matids)[i];

        if (mymatset.size()==0 or mymatset.find(id)!=mymatset.end())
        {
          const Teuchos::RCP<MAT::PAR::Material> actelastmat = mats.find(id)->second;
          switch (actelastmat->Type())
          {
          case INPAR::MAT::mes_couplogneohooke:
          {
            MAT::ELASTIC::PAR::CoupLogNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupLogNeoHooke*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            params2->SetLambda(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_coupneohooke:
          {
            MAT::ELASTIC::PAR::CoupNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupNeoHooke*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            params2->SetBeta(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_coupblatzko:
          {
            MAT::ELASTIC::PAR::CoupBlatzKo* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupBlatzKo*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            //params2->SetF(abs(p_cur(j+1)));
            params2->SetNue((abs(p_cur(j+1)))/(2.*(abs(p_cur(j+1))+1.)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isoneohooke:
          {
            MAT::ELASTIC::PAR::IsoNeoHooke* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoNeoHooke*>(actelastmat->Parameter());
            params2->SetMue(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_isoyeoh:
          {
            MAT::ELASTIC::PAR::IsoYeoh* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoYeoh*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            params2->SetC3(abs(p_cur(j+2)));
            j = j+3;
            break;
          }
          case INPAR::MAT::mes_isoquad:
          {
            MAT::ELASTIC::PAR::IsoQuad* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoQuad*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_isocub:
          {
            MAT::ELASTIC::PAR::IsoCub* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoCub*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_iso1pow:
          {
            MAT::ELASTIC::PAR::Iso1Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Iso1Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_iso2pow:
          {
            MAT::ELASTIC::PAR::Iso2Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Iso2Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coup1pow:
          {
            MAT::ELASTIC::PAR::Coup1Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Coup1Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coup2pow:
          {
            MAT::ELASTIC::PAR::Coup2Pow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::Coup2Pow*>(actelastmat->Parameter());
            params2->SetC(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_coupmooneyrivlin:
          {
            MAT::ELASTIC::PAR::CoupMooneyRivlin* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::CoupMooneyRivlin*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            params2->SetC3(abs(p_cur(j+2)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isoexpopow:
          {
            MAT::ELASTIC::PAR::IsoExpoPow* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoExpoPow*>(actelastmat->Parameter());
            params2->SetK1(abs(p_cur(j)));
            params2->SetK2(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_isomooneyrivlin:
          {
            MAT::ELASTIC::PAR::IsoMooneyRivlin* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoMooneyRivlin*>(actelastmat->Parameter());
            params2->SetC1(abs(p_cur(j)));
            params2->SetC2(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_volsussmanbathe:
          {
            MAT::ELASTIC::PAR::VolSussmanBathe* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolSussmanBathe*>(actelastmat->Parameter());
            params2->SetKappa(abs(p_cur(j)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_volpenalty:
          {
            MAT::ELASTIC::PAR::VolPenalty* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolPenalty*>(actelastmat->Parameter());
            params2->SetEpsilon(abs(p_cur(j)));
            params2->SetGamma(abs(p_cur(j+1)));
            j = j+2;
            break;
          }
          case INPAR::MAT::mes_vologden:
          {
            MAT::ELASTIC::PAR::VolOgden* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::VolOgden*>(actelastmat->Parameter());
            params2->SetKappa(abs(p_cur(j)));
            //params2->SetBeta(abs(p_cur(j+1)));
            j = j+1;
            break;
          }
          case INPAR::MAT::mes_isoratedep:
          {
            MAT::ELASTIC::PAR::IsoRateDep* params2 =
              dynamic_cast<MAT::ELASTIC::PAR::IsoRateDep*>(actelastmat->Parameter());
            params2->SetN(abs(p_cur(j)));
            j = j+1;
            break;
          }
          default:
            dserror("cannot deal with this material");
          }
        }
      }
    }
    break;
    default:
      // ignore unknown materials ?
      dserror("Unknown type of material");
      break;
    }
  }
  return;
}


void STR::GenInvAnalysis::CheckOptStep()
{
  for (int i=0; i<np_; i++)
  {
    if (p_(i) < 0.0)
    {
      std::cout << "at least one negative material parameter detected: reverse update" << std::endl;
      // reset the update step in case it was too large and increase regularization;
      // storage is not touched, so screen print out does not reflect what really happens
      p_ = p_o_;
      error_ = error_o_;
      error_grad_ = error_grad_o_;
      mu_= mu_o_*2.0;
      return;
    }
  }
}


//===========================================================================================
void STR::GenInvAnalysis::MultiInvAnaInit()
{
  for (int i=0; i<discret_->NumMyColElements(); i++)
  {
    DRT::Element* actele = discret_->lColElement(i);
    Teuchos::RCP<MAT::Material> mat = actele->Material();
    if (mat->MaterialType() == INPAR::MAT::m_struct_multiscale)
    {
      MAT::MicroMaterial* micro = static_cast <MAT::MicroMaterial*>(mat.get());
      bool eleowner = false;
      if (discret_->Comm().MyPID()==actele->Owner()) eleowner = true;
      micro->InvAnaInit(eleowner,actele->Id());
    }
  }
}

