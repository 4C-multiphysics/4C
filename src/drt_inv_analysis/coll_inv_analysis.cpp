/*----------------------------------------------------------------------*/
/*!
 * \file coll_inv_analysis.cpp

<pre>
Maintainer: Sophie Rausch
            rausch@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/rausch
            089 - 289-15255
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "coll_inv_analysis.H"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include "Epetra_SerialDenseMatrix.h"
#include "../drt_lib/global_inp_control2.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/drt_function.H"
#include "../drt_io/io_hdf.H"
#include "../drt_lib/linalg_ana.H"
#include "../drt_mat/material.H"
#include "../drt_mat/lung_ogden.H"
#include "../drt_mat/lung_penalty.H"
#include "../drt_structure/strtimint_create.H"
#include "../drt_mat/elasthyper.H"
#include "../drt_matelast/elast_coupanisoexpotwo.H"
#include "../drt_matelast/elast_coupanisoneohooketwo.H"
#include "../drt_matelast/elast_coupblatzko.H"
#include "../drt_matelast/elast_couplogneohooke.H"
#include "../drt_matelast/elast_isoexpo.H"
#include "../drt_matelast/elast_isomooneyrivlin.H"
#include "../drt_matelast/elast_isoneohooke.H"
#include "../drt_matelast/elast_isoyeoh.H"
#include "../drt_matelast/elast_isoquad.H"
#include "../drt_matelast/elast_isocub.H"
#include "../drt_matelast/elast_volpenalty.H"
#include "../drt_matelast/elast_vologden.H"
#include "../drt_matelast/elast_volsussmanbathe.H"

//using namespace LINALG::ANA;
using namespace std;
using namespace DRT;
using namespace MAT;


#include "../drt_structure/stru_resulttest.H"


/*----------------------------------------------------------------------*/
/* standard constructor */
STR::CollInvAnalysis::CollInvAnalysis(Teuchos::RCP<DRT::Discretization> dis,
                                Teuchos::RCP<LINALG::Solver> solver,
                                Teuchos::RCP<IO::DiscretizationWriter> output)
  : discret_(dis),
    solver_(solver),
    output_(output),
    sti_(Teuchos::null)
{

  // Getting boundary conditions
  discret_->GetCondition("SurfaceNeumann",surfneum_ );
  discret_->GetCondition("Dirichlet",surfdir_ );
  reset_out_count_=0;
  cout << "----------------------------------------------------------------------------------------------" << endl;
  cout << "----------------------------- Inverse Analyse based on the -----------------------------------" << endl;
  cout << "------------------------- Collagenase and Elastase Experiments -------------------------------" << endl;
  cout << "----------------------------------------------------------------------------------------------" << endl;
  if (surfneum_.size()>1)
    dserror("The inverse analysis only works for 1 NBC with 2 DBC or for 2 DBC!");

  // input parameters structural dynamics
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();

  // measured points, gives the number how many displacment steps are
  // measured
  nmp_   = 2*sdyn.get<int>("NUMSTEP");
  tstep_ = sdyn.get<double>("TIMESTEP");
  // get total timespan of simulation 0.5 due to factor 2 in nmp_
  double ttime_ = nmp_*tstep_*0.5;
  // input parameters inverse analysis
  const Teuchos::ParameterList& iap = DRT::Problem::Instance()->InverseAnalysisParams();

  //  tolerance for the curve fitting
  tol_ = iap.get<double>("INV_ANA_TOL");

  // experimentally measured curve
  {
    mcurve_   = Epetra_SerialDenseVector(nmp_);  //
    double cpx0 = iap.get<double>("MC_X_0");
    double cpx1 = iap.get<double>("MC_X_1");
    double cpx2 = iap.get<double>("MC_X_2");
    double cpy0 = iap.get<double>("MC_Y_0");
    double cpy1 = iap.get<double>("MC_Y_1");
    double cpy2 = iap.get<double>("MC_Y_2");
    for (int i=0; i<nmp_; i++)
    {
      mcurve_[i] = cpx0*(1-exp(-pow(((1000./ttime_)*cpx1*(i)*ttime_/nmp_), cpx2)));
      i=i+1;
      mcurve_[i] = cpy0*(1-exp(-pow(((1000./ttime_)*cpy1*(i-1)*ttime_/nmp_), cpy2)));
     }
  }
  //dserror("Halt");

  // error: diference of the measured to the calculated curve
  error_  = 1.0E6;
  error_o_= 1.5E6;

  // trainings parameter
  mu_  = 1.;
  tol_mu_ = tol_;

  // read material parameters from input file
  ReadInParameters();

  // Number of material parameters
  np_ = p_.Length();

  // controlling parameter
  numb_run_ =  0;         // counter of how many runs were made in the inverse analysis
}


/*----------------------------------------------------------------------*/
/* analyse */
void STR::CollInvAnalysis::Integrate()
{
  const Teuchos::ParameterList& iap = DRT::Problem::Instance()->InverseAnalysisParams();
  double alpha  = iap.get<double>("INV_ALPHA");
  double beta   = iap.get<double>("INV_BETA");
  int max_itter = iap.get<int>("INV_ANA_MAX_RUN");
  output_->NewResultFile((numb_run_-1));
  output_->WriteMesh(0,0.0);
  // fitting loop
  do
  {
    vector<double> inc(np_, 0.0);
    for (int i=0; i<np_;i++)
      inc[i] = alpha + beta * p_[i];

    Epetra_SerialDenseMatrix cmatrix(nmp_, np_+1);

    if (discret_->Comm().MyPID()==0)
      cout << "-----------------------------making Jaccobien matrix-------------------------" <<endl;
    for (int i=0; i<np_+1;i++)
    {
      if (discret_->Comm().MyPID()==0)
        cout << "------------------------------- run "<< i+1 << " of: " << np_+1 <<" ---------------------------------" <<endl;
      Epetra_SerialDenseVector p_cur = p_;
      if (i!= np_)
        p_cur[i]=p_[i] + inc[i];
      SetParameters(p_cur);
      Epetra_SerialDenseVector cvector = CalcCvector();
      for (int j=0; j<nmp_;j++)
        cmatrix(j, i)=cvector[j];
      output_->NewResultFile((numb_run_));
    }

    discret_->Comm().Barrier();
    for (int proc=0; proc<discret_->Comm().NumProc(); ++proc)
    {
      if (proc==discret_->Comm().MyPID())
      {
        if (proc == 0)
        {
          CalcNewParameters(cmatrix,  inc);
        }
      }
    }
    discret_->Comm().Barrier();
    // set new material parameters
    SetParameters(p_);
    numb_run_++;
    discret_->Comm().Broadcast(&error_o_,1,0);
    discret_->Comm().Broadcast(&error_,1,0);
    discret_->Comm().Broadcast(&numb_run_,1,0);
  }while(numb_run_<max_itter && error_>tol_) ;      // while (abs(error_o_-error_)>0.001 && error_>tol_ && numb_run_<max_itter);

  discret_->Comm().Barrier();
  for (int proc=0; proc<discret_->Comm().NumProc(); ++proc)
    if (proc==discret_->Comm().MyPID())
      if (proc == 0)

        PrintFile();
  discret_->Comm().Barrier();
  return;
}

void STR::CollInvAnalysis::CalcNewParameters(Epetra_SerialDenseMatrix cmatrix,  vector<double> inc)
{
  // initalization of the Jacobi and storage matrix
  Epetra_SerialDenseMatrix J(nmp_, np_);
  Epetra_SerialDenseMatrix sto(np_,  np_);
  Epetra_SerialDenseMatrix sto2(np_, nmp_);
  Epetra_SerialDenseVector delta_p(np_);
  Epetra_SerialDenseVector rcurve(nmp_);

  for (int i=0; i<nmp_; i++)
    for (int j=0; j<np_; j++)
      J(i, j) = (cmatrix(i, j)-cmatrix(i, np_)) / inc[j];     //calculating J(p)

  sto.Multiply('T',  'N',  1,  J, J,  0);     //calculating J.T*J

  for (int i=0; i<np_; i++)
    sto(i, i) += (mu_*sto(i, i));

  LINALG::NonSymmetricInverse(sto,  np_);     //calculating (J.T*J+mu*I).I
  sto2.Multiply('N', 'T', 1,  sto, J, 0);     //calculating (J.T*J+mu*I).I*J.T

  for (int i = 0; i<nmp_; i++)
    rcurve[i]=mcurve_[i]-cmatrix(i, np_);
  delta_p.Multiply('N', 'N', 1., sto2, rcurve, 0.);

  for (int i=0;i<np_;i++)
    p_[i] += delta_p[i];//*p_[i];

  // dependent on the # of steps
  error_o_   = error_;
  error_   = rcurve.Norm2()/sqrt(nmp_);

  //Adjust training parameter
  mu_ *= (error_/error_o_);
  if (numb_run_==0)
    mu_=1.;

  PrintStorage(cmatrix,  delta_p);
  return;
}

Epetra_SerialDenseVector STR::CollInvAnalysis::CalcCvector()
{
  // get input parameter lists
  const Teuchos::ParameterList& ioflags
    = DRT::Problem::Instance()->IOParams();
  const Teuchos::ParameterList& sdyn
    = DRT::Problem::Instance()->StructuralDynamicParams();
  Teuchos::ParameterList xparams;
  xparams.set<FILE*>("err file", DRT::Problem::Instance()->ErrorFile()->Handle());


  // create time integrator
  sti_ = TimIntCreate(ioflags, sdyn, xparams, discret_, solver_, output_);
  if (sti_ == Teuchos::null) dserror("Failed in creating integrator.");

  // initialize time loop / Attention the Functions give back the
  // time and the step not timen and stepn value that is why we have
  // to use < instead of <= for the while loop
  double time = sti_->GetTime();
  const double timemax = sti_->GetTimeEnd();
  int step = sti_->GetStep();
  const int stepmax = sti_->GetTimeNumStep();

  Epetra_SerialDenseVector cvector(2*stepmax);
  // time loop
  while ( (time < timemax) && (step < stepmax) )
  {
    // integrate time step
    // after this step we hold disn_, etc
    sti_->IntegrateStep();

    // update displacements, velocities, accelerations
    // after this call we will have disn_==dis_, etc
    sti_->UpdateStepState();

    // gets the displacments per timestep
    {
      Epetra_SerialDenseVector cvector_arg = GetCalculatedCurve();
      cvector[2*step]   = cvector_arg[0];
      cvector[2*step+1] = cvector_arg[1];
    }

    //dserror("Halt");
    // update time and step
    sti_->UpdateStepTime();

    // print info about finished time step
    sti_->PrintStep();

    // write output
    sti_->OutputStep();

    // Update Element
    sti_->UpdateStepElement();

    // get current time ...
    time = sti_->GetTime();
    // ... and step
    step = sti_->GetStep();

  }
  return cvector;
}


/*----------------------------------------------------------------------*/
/* */
Epetra_SerialDenseVector STR::CollInvAnalysis::GetCalculatedCurve()
{
  Epetra_SerialDenseVector cvector_arg(2);

  // current displacement vector
  Teuchos::RCP<Epetra_Vector> disp = sti_->DisNew();

  vector<DRT::Condition*> invanacond;
  discret_->GetCondition("SurfInvAna",  invanacond);

  //nodes of the pulling direction
  const vector<int>* ia_nd_ps  = invanacond[0]->Nodes();

  for (vector<int>::const_iterator inodegid = ia_nd_ps->begin();
       inodegid !=ia_nd_ps->end();
       ++inodegid)
  {
    if (discret_->HaveGlobalNode(*inodegid))
    {
      if (discret_->gNode(*inodegid)->Owner() == discret_->Comm().MyPID())
      {
        const DRT::Node* node = discret_->gNode(*inodegid);
        vector<int> lm = discret_->Dof(node);
        const double disp_x = (*disp)[disp->Map().LID(lm[0])];
        cvector_arg[0] += disp_x;
      }
    }
  }

  {
    double test;
    discret_->Comm().SumAll(&cvector_arg[0],&test,1);
    cvector_arg[0] = test/(*ia_nd_ps).size();
  }


  //nodes to determine the compression
  const vector<int>* ia_nd_fs_p = invanacond[1]->Nodes();
  const vector<int>* ia_nd_fs_n = invanacond[2]->Nodes();

  for (vector<int>::const_iterator inodegid = ia_nd_fs_p->begin();
       inodegid !=ia_nd_fs_p->end();
       ++inodegid)
  {
    if (discret_->HaveGlobalNode(*inodegid))
    {
      if (discret_->gNode(*inodegid)->Owner() == discret_->Comm().MyPID())
      {
        const DRT::Node* node = discret_->gNode(*inodegid);
        vector<int> lm = discret_->Dof(node);
        const double disp_y = (*disp)[disp->Map().LID(lm[1])];
        cvector_arg[1] += disp_y;
      }
    }
  }

  for (vector<int>::const_iterator inodegid = ia_nd_fs_n->begin();
       inodegid !=ia_nd_fs_n->end();
       ++inodegid)
  {
    if (discret_->HaveGlobalNode(*inodegid))
    {
      if (discret_->gNode(*inodegid)->Owner() == discret_->Comm().MyPID())
      {
        const DRT::Node* node = discret_->gNode(*inodegid);
        vector<int> lm = discret_->Dof(node);
        const double disp_y = (*disp)[disp->Map().LID(lm[1])];
        cvector_arg[1] -= disp_y;
      }
    }
  }

  {
    double test;
    discret_->Comm().SumAll(&cvector_arg[1],&test,1);
    cvector_arg[1] = test/(2.*(*ia_nd_fs_p).size());
  }
  return cvector_arg;
}

/*----------------------------------------------------------------------*/
/* */
void STR::CollInvAnalysis::PrintStorage(Epetra_SerialDenseMatrix cmatrix,  Epetra_SerialDenseVector delta_p)
{

  // store the error and mu_

  p_s_.Reshape(numb_run_+1,  np_);
  for (int i=0; i<np_; i++)
    p_s_(numb_run_, i)=p_(i);

  delta_p_s_.Reshape(numb_run_+1,  np_);
  for (int i=0; i<np_; i++)
    delta_p_s_(numb_run_, i)=delta_p(i);

  ccurve_s_.Reshape(nmp_,  numb_run_+1);
  for (int i=0; i<nmp_; i++)
    ccurve_s_(i, numb_run_)= cmatrix(i, cmatrix.ColDim()-1);

  mu_s_.Resize(numb_run_+1);
  mu_s_(numb_run_)=mu_;

  error_s_.Resize(numb_run_+1);
  error_s_(numb_run_) = error_;
  // print error and parameter

  cout << endl;
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
    printf("%10.3f", error_s_(i));
    printf("\tParameter: ");
    for (int j=0; j < delta_p.Length(); j++)
      printf("%10.3f", p_s_(i, j));
    //printf("\tDelta_p: ");
    //for (int j=0; j < delta_p.Length(); j++)
    //  printf("%10.3f", delta_p_s_(i, j));
    printf("\tmu: ");
    printf("%10.3f", mu_s_(i));
    printf("\n");
  }

  printf("\n");
  for (int i=0; i < nmp_/2.; i++)
  {
    printf(" %10.2f ",  mcurve_(i*2));
    if (numb_run_<15)
    {
      for (int j=0; j<numb_run_+1; j++)
        printf(" %10.2f ",  ccurve_s_((i)*2, j));
    }
    else
    {
      for (int j=numb_run_-14; j<numb_run_+1; j++)
        printf(" %10.2f ",  ccurve_s_((i)*2, j));
    }
    printf("\n");
  }

  printf("\n");

  for (int i=0; i < nmp_/2.; i++)
  {
    printf(" %10.2f ",  mcurve_((i)*2+1));
    if (numb_run_<15)
    {
      for (int j=0; j<numb_run_+1; j++)
        printf(" %10.2f ",  ccurve_s_((i)*2+1, j));
    }
    else
    {
      for (int j=numb_run_-14; j<numb_run_+1; j++)
        printf(" %10.2f ",  ccurve_s_((i)*2+1, j));
    }
    printf("\n");
  }

  printf("################################################");
  printf("##############################################\n");
  cout << endl;
}


void STR::CollInvAnalysis::PrintFile()
{

  FILE * gplot;
  FILE * cxFile;
  FILE * cyFile;
  FILE * pFile;

  string name = DRT::Problem::Instance()->OutputControlFile()->FileName();
  name.append(filename_);

  if (name.rfind("_run_")!=string::npos)
  {
    size_t pos = name.rfind("_run_");
    if (pos==string::npos)
      dserror("inconsistent file name");
    name = name.substr(0, pos);
  }

  string gp     = name+"_plot.gp";
  string xcurve = name+"_Curve_x.txt";
  string ycurve = name+"_Curve_y.txt";
  string para   = name+"_Para.txt";

  cxFile = fopen((xcurve).c_str(), "w");
  for (int i=0; i < nmp_/2.; i++)
  {
    fprintf(cxFile, " %10.5f ,",  mcurve_(i*2));
    for (int j=0; j<numb_run_+1; j++)
      fprintf(cxFile, " %10.5f ,",  ccurve_s_(i*2, j));
    fprintf(cxFile, "\n");
  }
  fclose(cxFile);

  cyFile = fopen((ycurve).c_str(), "w");
  for (int i=0; i < nmp_/2.; i++)
  {
    fprintf(cyFile, " %10.5f ,",  mcurve_((i)*2+1));
    for (int j=0; j<numb_run_+1; j++)
      fprintf(cyFile, " %10.5f ,",  ccurve_s_((i)*2+1, j));
    fprintf(cyFile, "\n");
  }
  fclose(cyFile);

  pFile  = fopen((para).c_str(), "w");
  fprintf(pFile, "#Error       Parameter    Delta_p      mu \n");
  for (int i=0; i < numb_run_+1; i++)
  {
    fprintf(pFile, "%10.3f,", error_s_(i));
    for (int j=0; j < np_; j++)
      fprintf(pFile, "%10.3f,", p_s_(i, j));
    for (int j=0; j < np_; j++)
      fprintf(pFile, "%10.3f,", delta_p_s_(i, j));
    fprintf(pFile, "%10.3f", mu_s_(i));
    fprintf(pFile, "\n");
  }
  fclose(pFile);

  numb_run_=numb_run_-1;
}


void STR::CollInvAnalysis::ReadInParameters()
{

  Teuchos::RCP<const MAT::Material> material = discret_->lRowElement(0)->Material();
  if (material->MaterialType() == INPAR::MAT::m_lung_penalty)
  {
    const MAT::LungPenalty* actmat = static_cast<const MAT::LungPenalty*>(material.get());
    int j = p_.Length();
    p_.Resize(j+3);
    p_(j)   = actmat->C();
    p_(j+1) = actmat->K1();
    p_(j+2) = actmat->K2();
  }
  else if (material->MaterialType() == INPAR::MAT::m_lung_ogden)
  {
    const MAT::LungOgden* actmat = static_cast<const MAT::LungOgden*>(material.get());
    int j = p_.Length();
    p_.Resize(j+3);
    p_(j)   = actmat->C();
    p_(j+1) = actmat->K1();
    p_(j+2) = actmat->K2();
  }
  else if (material->MaterialType() == INPAR::MAT::m_elasthyper)
  {
    // Create a pointer on the Material
    const MAT::ElastHyper* actmat = static_cast<const MAT::ElastHyper*>(material.get());

    // For each of the summands of the hyperelastic material we need to add the
    // parameters to the inverse analysis

    // Problems with beta, is it the only negative parameter? Maybe
    // we should exclude it

    for (int i=0; i< actmat->NumMat(); i++)
    {
      //get the material of the summand
      Teuchos::RCP< MAT::ELASTIC::Summand > summat = MAT::ELASTIC::Summand::Factory(actmat->MatID(i));
      filename_=filename_+"_";
      switch (summat->MaterialType())
      {
      case INPAR::MAT::mes_couplogneohooke:
      {
        filename_=filename_+"_couplogneohooke";
        const MAT::ELASTIC::CoupLogNeoHooke* actmat2 = static_cast<const MAT::ELASTIC::CoupLogNeoHooke*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+2);
        //p_[j]   = actmat2->Mue();
        //p_[j+1] = actmat2->Lambda();
        //p_[j+2] = actmat2->Parmode();
        p_[j] = actmat2->Youngs();
        p_[j+1] = (1./(1.-2.*actmat2->Nue()))-1.;
        cout << "Get the parameter: " << p_[j+1] << " for the Simulation " << actmat2->Nue() << " was used!" << endl;
        break;
      }
      case INPAR::MAT::mes_coupblatzko:
      {

        filename_=filename_+"_coupblatzko";
        const MAT::ELASTIC::CoupBlatzKo* actmat2 = static_cast<const MAT::ELASTIC::CoupBlatzKo*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+2);
        p_[j]   = (actmat2->Mue());
        p_[j+1] = (1./(1.-2.*actmat2->Nue()))-1.;
        //p_[j+1] = actmat2->F();
        break;
      }
      case INPAR::MAT::mes_isoneohooke:
      {
        filename_=filename_+"_isoneohooke";
        const MAT::ELASTIC::IsoNeoHooke* actmat2 = static_cast<const MAT::ELASTIC::IsoNeoHooke*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+1);
        p_[j]   = actmat2->Mue();
        break;
      }
      case INPAR::MAT::mes_isoyeoh:
      {
        filename_=filename_+"_isoyeoh";
        const MAT::ELASTIC::IsoYeoh* actmat2 = static_cast<const MAT::ELASTIC::IsoYeoh*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+3);
        p_[j]   = actmat2->C1();
        p_[j+1] = actmat2->C2();
        p_[j+2] = actmat2->C3();
        break;
      }
      case INPAR::MAT::mes_isoquad:
      {
        filename_=filename_+"_isoquad";
        const MAT::ELASTIC::IsoQuad* actmat2 = static_cast<const MAT::ELASTIC::IsoQuad*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+1);
        p_[j]   = actmat2->C();
        break;
      }
      case INPAR::MAT::mes_isocub:
      {
        filename_=filename_+"_isocub";
        const MAT::ELASTIC::IsoCub* actmat2 = static_cast<const MAT::ELASTIC::IsoCub*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+1);
        p_[j]   = actmat2->C();
        break;
      }
      case INPAR::MAT::mes_isoexpo:
      {
        filename_=filename_+"_isoexpo";
        const MAT::ELASTIC::IsoExpo* actmat2 = static_cast<const MAT::ELASTIC::IsoExpo*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+2);
        p_[j]   = actmat2->K1();
        p_[j+1] = actmat2->K2();
        break;
      }
      case INPAR::MAT::mes_isomooneyrivlin:
      {
        filename_=filename_+"_isomooneyrivlin";
        const MAT::ELASTIC::IsoMooneyRivlin* actmat2 = static_cast<const MAT::ELASTIC::IsoMooneyRivlin*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+2);
        p_[j]   = actmat2->C1();
        p_[j+1] = actmat2->C2();
        break;
      }
      case INPAR::MAT::mes_volsussmanbathe:
      {
        filename_=filename_+"_volsussmanbathe";
        const MAT::ELASTIC::VolSussmanBathe* actmat2 = static_cast<const MAT::ELASTIC::VolSussmanBathe*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+1);
        p_[j]   = actmat2->Kappa();
        break;
      }
      case INPAR::MAT::mes_volpenalty:
      {
        filename_=filename_+"_volpenalty";
        const MAT::ELASTIC::VolPenalty* actmat2 = static_cast<const MAT::ELASTIC::VolPenalty*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+2);
        p_[j]   = actmat2->Epsilon();
        p_[j+1]   = actmat2->Gamma();
        break;
      }
      case INPAR::MAT::mes_vologden:
      {
        filename_=filename_+"_vologden";
        const MAT::ELASTIC::VolOgden* actmat2 = static_cast<const MAT::ELASTIC::VolOgden*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+1);
        p_[j]   = actmat2->Kappa();
        //p_[j+1] = actmat2->Beta();
        break;
      }
      case INPAR::MAT::mes_coupanisoexpotwo:
      {
        filename_=filename_+"_coupanisoexpotwo";
        const MAT::ELASTIC::CoupAnisoExpoTwo* actmat2 = static_cast<const MAT::ELASTIC::CoupAnisoExpoTwo*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+4);
        p_[j]   = actmat2->K1();
        p_[j+1] = actmat2->K2();
        p_[j+2] = actmat2->K3();
        p_[j+3] = actmat2->K4();
        break;
      }
      case INPAR::MAT::mes_coupanisoneohooketwo:
      {
        filename_=filename_+"_coupanisoneohooketwo";
        const MAT::ELASTIC::CoupAnisoNeoHookeTwo* actmat2 = static_cast<const MAT::ELASTIC::CoupAnisoNeoHookeTwo*>(summat.get());
        int j = p_.Length();
        p_.Resize(j+2);
        p_[j]   = actmat2->C1();
        p_[j+1] = actmat2->C2();
        break;
      }
      default:
      {
        dserror("cannot deal with this material");
      }
      }
    }
  }
  else dserror("The inverse analysis is not implemented for this material");

  return;
}

void STR::CollInvAnalysis::SetParameters(Epetra_SerialDenseVector p_cur)
{

  discret_->Comm().Broadcast(&p_cur[0],np_,0);
  {

  // write new material parameter

  Teuchos::RCP<MAT::Material> material = discret_->lRowElement(0)->Material();
  if (material->MaterialType() == INPAR::MAT::m_lung_penalty)
  {
    MAT::LungPenalty* actmat = static_cast<MAT::LungPenalty*>(material.get());
    actmat->SetC(abs(p_cur(0)));
    actmat->SetK1(abs(p_cur(1)));
    actmat->SetK2(abs(p_cur(2)));
  }
  else if (material->MaterialType() == INPAR::MAT::m_lung_ogden)
  {
    MAT::LungOgden* actmat = static_cast<MAT::LungOgden*>(material.get());
    actmat->SetC(abs(p_cur(0)));
    actmat->SetK1(abs(p_cur(1)));
    actmat->SetK2(abs(p_cur(2)));
  }
  else if (material->MaterialType() == INPAR::MAT::m_elasthyper)
  {
    // Create a pointer on the Material
    const MAT::ElastHyper* actmat = static_cast<const MAT::ElastHyper*>(material.get());

    // For each of the summands of the hyperelastic material we need to add the
    // parameters to the inverse analysis

    // Problems with beta, is it the only negative parameter? Maybe
    // we should exclude it

    //itterator to go through the parameters
    int j = 0;

    for (int i=0; i< actmat->NumMat(); i++)
    {
      //get the material of the summand
      Teuchos::RCP< MAT::ELASTIC::Summand > summat =
        MAT::ELASTIC::Summand::Factory(actmat->MatID(i));
      switch (summat->MaterialType())
      {
      case INPAR::MAT::mes_couplogneohooke:
      {
        MAT::ELASTIC::CoupLogNeoHooke* actmat2 =
          static_cast<MAT::ELASTIC::CoupLogNeoHooke*>(summat.get());
        //actmat2->SetMue(abs(p_cur(j)));
        //actmat2->SetLambda(abs(p_cur(j+1)));
        //actmat2->SetParmode(abs(p_cur(j+2)));
        actmat2->SetYoungs(abs(p_cur(j)));
        actmat2->SetNue((abs(p_cur(j+1)))/(2.*(abs(p_cur(j+1))+1.)));
        j = j+2;
        break;
      }
      case INPAR::MAT::mes_coupblatzko:
      {
        MAT::ELASTIC::CoupBlatzKo* actmat2 =
          static_cast<MAT::ELASTIC::CoupBlatzKo*>(summat.get());
        actmat2->SetMue(abs(p_cur(j)));
        //actmat2->SetF(abs(p_cur(j+1)));
        actmat2->SetNue((abs(p_cur(j+1)))/(2.*(abs(p_cur(j+1))+1.)));
        j = j+2;
        break;
      }
      case INPAR::MAT::mes_isoneohooke:
      {
        MAT::ELASTIC::IsoNeoHooke* actmat2 =
          static_cast<MAT::ELASTIC::IsoNeoHooke*>(summat.get());
        actmat2->SetMue(abs(p_cur(j)));
        j = j+1;
        break;
      }
      case INPAR::MAT::mes_isoyeoh:
      {
        MAT::ELASTIC::IsoYeoh* actmat2 =
          static_cast<MAT::ELASTIC::IsoYeoh*>(summat.get());
        actmat2->SetC1(abs(p_cur(j)));
        actmat2->SetC2(abs(p_cur(j+1)));
        actmat2->SetC3(abs(p_cur(j+2)));
        j = j+3;
        break;
      }
      case INPAR::MAT::mes_isoquad:
      {
        MAT::ELASTIC::IsoQuad* actmat2 =
          static_cast<MAT::ELASTIC::IsoQuad*>(summat.get());
        actmat2->SetC(abs(p_cur(j)));
        j = j+1;
        break;
      }
      case INPAR::MAT::mes_isocub:
      {
        MAT::ELASTIC::IsoCub* actmat2 =
          static_cast<MAT::ELASTIC::IsoCub*>(summat.get());
        actmat2->SetC(abs(p_cur(j)));
        j = j+1;
        break;
      }
      case INPAR::MAT::mes_isoexpo:
      {
        MAT::ELASTIC::IsoExpo* actmat2 =
          static_cast<MAT::ELASTIC::IsoExpo*>(summat.get());
        actmat2->SetK1(abs(p_cur(j)));
        actmat2->SetK2(abs(p_cur(j+1)));
        j = j+2;
        break;
      }
      case INPAR::MAT::mes_isomooneyrivlin:
      {
        MAT::ELASTIC::IsoMooneyRivlin* actmat2 =
          static_cast<MAT::ELASTIC::IsoMooneyRivlin*>(summat.get());
        actmat2->SetC1(abs(p_cur(j)));
        actmat2->SetC2(abs(p_cur(j+1)));
        j = j+2;
        break;
      }
      case INPAR::MAT::mes_volsussmanbathe:
      {
        MAT::ELASTIC::VolSussmanBathe* actmat2 =
          static_cast<MAT::ELASTIC::VolSussmanBathe*>(summat.get());
        actmat2->SetKappa(abs(p_cur(j)));
        j = j+1;
        break;
      }
      case INPAR::MAT::mes_volpenalty:
      {
        MAT::ELASTIC::VolPenalty* actmat2 =
          static_cast<MAT::ELASTIC::VolPenalty*>(summat.get());
        actmat2->SetEpsilon(abs(p_cur(j)));
        actmat2->SetGamma(abs(p_cur(j+1)));
        j = j+2;
        break;
      }
      case INPAR::MAT::mes_vologden:
      {
        MAT::ELASTIC::VolOgden* actmat2 =
          static_cast<MAT::ELASTIC::VolOgden*>(summat.get());
        actmat2->SetKappa(abs(p_cur(j)));
        //actmat2->SetBeta(abs(p_cur(j+1)));
        j = j+1;
        break;
      }
      case INPAR::MAT::mes_coupanisoexpotwo:
      {
        MAT::ELASTIC::CoupAnisoExpoTwo* actmat2 =
          static_cast<MAT::ELASTIC::CoupAnisoExpoTwo*>(summat.get());
        actmat2->SetK1(abs(p_cur(j)));
        actmat2->SetK2(abs(p_cur(j+1)));
        actmat2->SetK3(abs(p_cur(j+2)));
        actmat2->SetK4(abs(p_cur(j+3)));
        j = j+4;
        break;
      }
      case INPAR::MAT::mes_coupanisoneohooketwo:
      {
        MAT::ELASTIC::CoupAnisoNeoHookeTwo* actmat2 =
          static_cast<MAT::ELASTIC::CoupAnisoNeoHookeTwo*>(summat.get());
        actmat2->SetC1(abs(p_cur(j)));
        actmat2->SetC2(abs(p_cur(j+1)));
        j = j+2;
        break;
      }
      default:
        dserror("cannot deal with this material");
      }
    }
  }
  }
}



#endif
