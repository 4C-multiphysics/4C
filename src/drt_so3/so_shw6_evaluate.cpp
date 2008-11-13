/*!----------------------------------------------------------------------
\file so_shw6_evaluate.cpp
\brief

<pre>
Maintainer: Moritz Frenzel
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15240
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET

// This is just here to get the c++ mpi header, otherwise it would
// use the c version included inside standardtypes.h
#ifdef PARALLEL
#include "mpi.h"
#endif
#include "so_shw6.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_exporter.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/linalg_serialdensematrix.H"
#include "../drt_lib/linalg_serialdensevector.H"
#include "../drt_fem_general/drt_utils_integration.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
#include "Epetra_SerialDenseSolver.h"

using namespace std; // cout etc.
using namespace LINALG; // our linear algebra

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              maf 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_shw6::Evaluate(ParameterList& params,
                                    DRT::Discretization&      discretization,
                                    vector<int>&              lm,
                                    Epetra_SerialDenseMatrix& elemat1_epetra,
                                    Epetra_SerialDenseMatrix& elemat2_epetra,
                                    Epetra_SerialDenseVector& elevec1_epetra,
                                    Epetra_SerialDenseVector& elevec2_epetra,
                                    Epetra_SerialDenseVector& elevec3_epetra)
{
  LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,NUMDOF_WEG6> elemat1(elemat1_epetra.A(),true);
  LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,NUMDOF_WEG6> elemat2(elemat2_epetra.A(),true);
  LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,1> elevec1(elevec1_epetra.A(),true);
  LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,1> elevec2(elevec2_epetra.A(),true);
  // elevec3 is not used anyway

  // start with "none"
  DRT::ELEMENTS::So_weg6::ActionType act = So_weg6::none;

  // get the required action
  string action = params.get<string>("action","none");
  if (action == "none") dserror("No action supplied");
  else if (action=="calc_struct_linstiff")      act = So_weg6::calc_struct_linstiff;
  else if (action=="calc_struct_nlnstiff")      act = So_weg6::calc_struct_nlnstiff;
  else if (action=="calc_struct_internalforce") act = So_weg6::calc_struct_internalforce;
  else if (action=="calc_struct_linstiffmass")  act = So_weg6::calc_struct_linstiffmass;
  else if (action=="calc_struct_nlnstiffmass")  act = So_weg6::calc_struct_nlnstiffmass;
  else if (action=="calc_struct_nlnstifflmass") act = So_weg6::calc_struct_nlnstifflmass;
  else if (action=="calc_struct_stress")        act = So_weg6::calc_struct_stress;
  else if (action=="calc_struct_eleload")       act = So_weg6::calc_struct_eleload;
  else if (action=="calc_struct_fsiload")       act = So_weg6::calc_struct_fsiload;
  else if (action=="calc_struct_update_istep")  act = So_weg6::calc_struct_update_istep;
  else if (action=="calc_struct_update_imrlike") act = So_weg6::calc_struct_update_imrlike;
  else if (action=="calc_struct_reset_istep")   act = So_weg6::calc_struct_reset_istep;
  else if (action=="postprocess_stress")        act = So_weg6::postprocess_stress;
  else dserror("Unknown type of action for So_weg6");

  // what should the element do
  switch(act) {
    // linear stiffness
    case calc_struct_linstiff: {
      // need current displacement and residual forces
      vector<double> mydisp(lm.size());
      for (int i=0; i<(int)mydisp.size(); ++i) mydisp[i] = 0.0;
      vector<double> myres(lm.size());
      for (int i=0; i<(int)myres.size(); ++i) myres[i] = 0.0;
      soshw6_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,params);
    }
    break;

    // nonlinear stiffness and internal force vector
    case calc_struct_nlnstiff: {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      soshw6_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,params);
    }
    break;

    // internal force vector only
    case calc_struct_internalforce:
    {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      // create a dummy element matrix to apply linearised EAS-stuff onto
      LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,NUMDOF_WEG6> myemat(true);
      soshw6_nlnstiffmass(lm,mydisp,myres,&myemat,NULL,&elevec1,NULL,NULL,params);
    }
    break;

    // linear stiffness and consistent mass matrix
    case calc_struct_linstiffmass:
      dserror("Case 'calc_struct_linstiffmass' not yet implemented");
    break;

    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case calc_struct_nlnstiffmass:
    case calc_struct_nlnstifflmass:
    {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      soshw6_nlnstiffmass(lm,mydisp,myres,&elemat1,&elemat2,&elevec1,NULL,NULL,params);
      if (act==calc_struct_nlnstifflmass) sow6_lumpmass(&elemat2);
    }
    break;

    // evaluate stresses at gauss points
    case calc_struct_stress:{
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      RCP<vector<char> > stressdata = params.get<RCP<vector<char> > >("stress", null);
      RCP<vector<char> > straindata = params.get<RCP<vector<char> > >("strain", null);
      if (disp==null) dserror("Cannot get state vectors 'displacement'");
      if (stressdata==null) dserror("Cannot get stress 'data'");
      if (straindata==null) dserror("Cannot get strain 'data'");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      LINALG::FixedSizeSerialDenseMatrix<NUMGPT_WEG6,NUMSTR_WEG6> stress;
      LINALG::FixedSizeSerialDenseMatrix<NUMGPT_WEG6,NUMSTR_WEG6> strain;
      bool cauchy = params.get<bool>("cauchy", false);
      string iostrain = params.get<string>("iostrain", "none");
      if (iostrain!="euler_almansi") soshw6_nlnstiffmass(lm,mydisp,myres,NULL,NULL,NULL,&stress,&strain,params,cauchy);
      else dserror("requested option not yet implemented for solidshw6");
      AddtoPack(*stressdata, stress);
      AddtoPack(*straindata, strain);
    }
    break;

    // postprocess stresses and strains at gauss points

    // note that in the following, quantities are always referred to as
    // "stresses" etc. although they might also apply to strains
    // (depending on what this routine is called for from the post filter)
    case postprocess_stress:{

      const RCP<std::map<int,RCP<Epetra_SerialDenseMatrix> > > gpstressmap=
        params.get<RCP<std::map<int,RCP<Epetra_SerialDenseMatrix> > > >("gpstressmap",null);
      if (gpstressmap==null)
        dserror("no gp stress/strain map available for postprocessing");
      string stresstype = params.get<string>("stresstype","ndxyz");
      int gid = Id();
      LINALG::FixedSizeSerialDenseMatrix<NUMGPT_WEG6,NUMSTR_WEG6> gpstress(((*gpstressmap)[gid])->A(),true);

      if (stresstype=="ndxyz") {
        // extrapolate stresses/strains at Gauss points to nodes
        LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,NUMSTR_WEG6> nodalstresses;
        soweg6_expol(gpstress,nodalstresses);

        // average nodal stresses/strains between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_WEG6);

        DRT::Node** nodes = Nodes();
        for (int i=0;i<NUMNOD_WEG6;++i)
        {
          DRT::Node* node = nodes[i];
          numadjele[i]=node->NumElement();
        }

        for (int i=0;i<NUMNOD_WEG6;++i){
          elevec1(3*i)=nodalstresses(i,0)/numadjele[i];
          elevec1(3*i+1)=nodalstresses(i,1)/numadjele[i];
          elevec1(3*i+2)=nodalstresses(i,2)/numadjele[i];
        }
        for (int i=0;i<NUMNOD_WEG6;++i){
          elevec2(3*i)=nodalstresses(i,3)/numadjele[i];
          elevec2(3*i+1)=nodalstresses(i,4)/numadjele[i];
          elevec2(3*i+2)=nodalstresses(i,5)/numadjele[i];
        }
      }
      else if (stresstype=="cxyz") {
        RCP<Epetra_MultiVector> elestress=params.get<RCP<Epetra_MultiVector> >("elestress",null);
        if (elestress==null)
          dserror("No element stress/strain vector available");
        const Epetra_BlockMap elemap = elestress->Map();
        int lid = elemap.LID(Id());
        if (lid!=-1)
        {
          for (int i = 0; i < NUMSTR_WEG6; ++i)
          {
            double& s = (*((*elestress)(i)))[lid]; // resolve pointer for faster access
            s = 0.;
            for (int j = 0; j < NUMGPT_WEG6; ++j)
            {
              s += gpstress(j,i);
            }
            s *= 1.0/NUMGPT_WEG6;
          }
        }
      }
      else if (stresstype=="cxyz_ndxyz") {
        // extrapolate stresses/strains at Gauss points to nodes
        LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,NUMSTR_WEG6> nodalstresses;
        soweg6_expol(gpstress,nodalstresses);

        // average nodal stresses/strains between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_WEG6);

        DRT::Node** nodes = Nodes();
        for (int i=0;i<NUMNOD_WEG6;++i){
          DRT::Node* node=nodes[i];
          numadjele[i]=node->NumElement();
        }

        for (int i=0;i<NUMNOD_WEG6;++i){
          elevec1(3*i)=nodalstresses(i,0)/numadjele[i];
          elevec1(3*i+1)=nodalstresses(i,1)/numadjele[i];
          elevec1(3*i+2)=nodalstresses(i,2)/numadjele[i];
        }
        for (int i=0;i<NUMNOD_WEG6;++i){
          elevec2(3*i)=nodalstresses(i,3)/numadjele[i];
          elevec2(3*i+1)=nodalstresses(i,4)/numadjele[i];
          elevec2(3*i+2)=nodalstresses(i,5)/numadjele[i];
        }
        RCP<Epetra_MultiVector> elestress=params.get<RCP<Epetra_MultiVector> >("elestress",null);
        if (elestress==null)
          dserror("No element stress/strain vector available");
        const Epetra_BlockMap elemap = elestress->Map();
        int lid = elemap.LID(Id());
        if (lid!=-1) {
          for (int i = 0; i < NUMSTR_WEG6; ++i)
          {
            double& s = (*((*elestress)(i)))[lid]; // resolve pointer for faster access
            s = 0.;
            for (int j = 0; j < NUMGPT_WEG6; ++j)
            {
              s += gpstress(j,i);
            }
            s *= 1.0/NUMGPT_WEG6;
          }
        }
      }
      else{
        dserror("unknown type of stress/strain output on element level");
      }
    }
    break;

    case calc_struct_eleload:
      dserror("this method is not supposed to evaluate a load, use EvaluateNeumann(...)");
    break;

    case calc_struct_fsiload:
      dserror("Case not yet implemented");
    break;

    case calc_struct_update_istep: {
      // do something with internal EAS, etc parameters
      if (eastype_ == soshw6_easpoisthick) {
        Epetra_SerialDenseMatrix* alpha = data_.GetMutable<Epetra_SerialDenseMatrix>("alpha");  // Alpha_{n+1}
        Epetra_SerialDenseMatrix* alphao = data_.GetMutable<Epetra_SerialDenseMatrix>("alphao");  // Alpha_n
        // alphao := alpha
        LINALG::DENSEFUNCTIONS::update<soshw6_easpoisthick,1>(*alphao,*alpha);
      }
    }
    break;

    case calc_struct_update_imrlike: {
      // do something with internal EAS, etc parameters
      // this depends on the applied solution technique (static, generalised-alpha,
      // or other time integrators)
      if (eastype_ == soshw6_easpoisthick) {
        double alphaf = params.get<double>("alpha f", 0.0);  // generalised-alpha TIS parameter alpha_f
        Epetra_SerialDenseMatrix* alpha = data_.GetMutable<Epetra_SerialDenseMatrix>("alpha");  // Alpha_{n+1-alphaf}
        Epetra_SerialDenseMatrix* alphao = data_.GetMutable<Epetra_SerialDenseMatrix>("alphao");  // Alpha_n
        // alphao = (-alphaf/(1.0-alphaf))*alphao  + 1.0/(1.0-alphaf) * alpha
        LINALG::DENSEFUNCTIONS::update<soshw6_easpoisthick,1>(-alphaf/(1.0-alphaf),*alphao,1.0/(1.0-alphaf),*alpha);
        LINALG::DENSEFUNCTIONS::update<soshw6_easpoisthick,1>(*alpha,*alphao); // alpha := alphao
      }
    }
    break;

    case calc_struct_reset_istep: {
      // do something with internal EAS, etc parameters
      if (eastype_ == soshw6_easpoisthick) {
        Epetra_SerialDenseMatrix* alpha = data_.GetMutable<Epetra_SerialDenseMatrix>("alpha");  // Alpha_{n+1}
        Epetra_SerialDenseMatrix* alphao = data_.GetMutable<Epetra_SerialDenseMatrix>("alphao");  // Alpha_n
        // alpha := alphao
        LINALG::DENSEFUNCTIONS::update<soshw6_easpoisthick,1>(*alpha, *alphao);
      }
    }
    break;

    default:
      dserror("Unknown type of action for Solid3");
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (private)                             maf 04/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_shw6::soshw6_nlnstiffmass(
      vector<int>&              lm,             // location matrix
      vector<double>&           disp,           // current displacements
      vector<double>&           residual,       // current residual displ
      LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,NUMDOF_WEG6>* stiffmatrix, // element stiffness matrix
      LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,NUMDOF_WEG6>* massmatrix,  // element mass matrix
      LINALG::FixedSizeSerialDenseMatrix<NUMDOF_WEG6,1>* force,                 // element internal force vector
      LINALG::FixedSizeSerialDenseMatrix<NUMGPT_WEG6,NUMSTR_WEG6>* elestress,   // stresses at GP
      LINALG::FixedSizeSerialDenseMatrix<NUMGPT_WEG6,NUMSTR_WEG6>* elestrain,   // strains at GP
      ParameterList&            params,         // algorithmic parameters e.g. time
      const bool                cauchy)         // stress output option
{

/* ============================================================================*
** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for Wedge_6 with 6 GAUSS POINTS*
** ============================================================================*/
  const static vector<LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,1> > shapefcts = sow6_shapefcts();
  const static vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMNOD_WEG6> > derivs = sow6_derivs();
  const static vector<double> gpweights = sow6_weights();
/* ============================================================================*/

  // update element geometry
  LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,NUMDIM_WEG6> xrefe;  // material coord. of element
  LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,NUMDIM_WEG6> xcurr;  // current  coord. of element
  DRT::Node** nodes = Nodes();
  for (int i=0; i<NUMNOD_WEG6; ++i)
  {
    const double* x = nodes[i]->X();
    xrefe(i,0) = x[0];
    xrefe(i,1) = x[1];
    xrefe(i,2) = x[2];

    xcurr(i,0) = xrefe(i,0) + disp[i*NODDOF_WEG6+0];
    xcurr(i,1) = xrefe(i,1) + disp[i*NODDOF_WEG6+1];
    xcurr(i,2) = xrefe(i,2) + disp[i*NODDOF_WEG6+2];
  }

  /*
  ** EAS Technology: declare, intialize, set up, and alpha history -------- EAS
  */
  // in any case declare variables, sizes etc. only in eascase
  Epetra_SerialDenseMatrix* alpha = NULL;  // EAS alphas
  vector<Epetra_SerialDenseMatrix>* M_GP = NULL;  // EAS matrix M at all GPs
  LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,soshw6_easpoisthick> M; // EAS matrix M at current GP, fixed for sosh8
  Epetra_SerialDenseVector feas;    // EAS portion of internal forces
  Epetra_SerialDenseMatrix Kaa;     // EAS matrix Kaa
  Epetra_SerialDenseMatrix Kda;     // EAS matrix Kda
  double detJ0;                     // detJ(origin)
  Epetra_SerialDenseMatrix* oldfeas = NULL;   // EAS history
  Epetra_SerialDenseMatrix* oldKaainv = NULL; // EAS history
  Epetra_SerialDenseMatrix* oldKda = NULL;    // EAS history
  // transformation matrix T0, maps M-matrix evaluated at origin
  // between local element coords and global coords
  // here we already get the inverse transposed T0
  LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,NUMSTR_WEG6> T0invT;  // trafo matrix
  if (eastype_ == soshw6_easpoisthick) {
    /*
    ** EAS Update of alphas:
    ** the current alphas are (re-)evaluated out of
    ** Kaa and Kda of previous step to avoid additional element call.
    ** This corresponds to the (innermost) element update loop
    ** in the nonlinear FE-Skript page 120 (load-control alg. with EAS)
    */
    alpha = data_.GetMutable<Epetra_SerialDenseMatrix>("alpha");   // get old alpha
    // evaluate current (updated) EAS alphas (from history variables)
    // get stored EAS history
    oldfeas = data_.GetMutable<Epetra_SerialDenseMatrix>("feas");
    oldKaainv = data_.GetMutable<Epetra_SerialDenseMatrix>("invKaa");
    oldKda = data_.GetMutable<Epetra_SerialDenseMatrix>("Kda");
    if (!alpha || !oldKaainv || !oldKda || !oldfeas) dserror("Missing EAS history-data");

    // we need the (residual) displacement at the previous step
    Epetra_SerialDenseVector res_d(NUMDOF_WEG6);
    for (int i = 0; i < NUMDOF_WEG6; ++i) {
      res_d(i) = residual[i];
    }
    // add Kda . res_d to feas
    LINALG::DENSEFUNCTIONS::multiply<soshw6_easpoisthick, NUMDOF_WEG6,1>(1.0, *oldfeas, 1.0, *oldKda, res_d);
    // "new" alpha is: - Kaa^-1 . (feas + Kda . old_d), here: - Kaa^-1 . feas
    LINALG::DENSEFUNCTIONS::multiply<soshw6_easpoisthick,soshw6_easpoisthick,1>(1.0,*alpha,-1.0,*oldKaainv,*oldfeas);
    /* end of EAS Update ******************/

    // EAS portion of internal forces, also called enhacement vector s or Rtilde
    feas.Size(neas_);

    // EAS matrix K_{alpha alpha}, also called Dtilde
    Kaa.Shape(neas_,neas_);

    // EAS matrix K_{d alpha}
    Kda.Shape(neas_,NUMDOF_WEG6);

    /* evaluation of EAS variables (which are constant for the following):
    ** -> M defining interpolation of enhanced strains alpha, evaluated at GPs
    ** -> determinant of Jacobi matrix at element origin (r=s=t=0.0)
    ** -> T0^{-T}
    */
    soshw6_eassetup(&M_GP,detJ0,T0invT,xrefe);
  } else if (eastype_ == soshw6_easnone){
  //cout << "Warning: Solid-Shell Wegde6 without EAS" << endl;
  } else dserror("Unknown EAS-type for solid wedge6");// ------------------- EAS

  /*
  ** ANS Element technology to remedy
  *  - transverse-shear locking E_rt and E_st
  *  - trapezoidal (curvature-thickness) locking E_tt
  */
  // modified B-operator in local(parameter) element space
  // ANS modified rows of bop in local(parameter) coords
  LINALG::FixedSizeSerialDenseMatrix<num_ans*num_sp,NUMDOF_WEG6> B_ans_loc;
  // Jacobian evaluated at all ANS sampling points
  vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> > jac_sps(num_sp);
  // CURRENT Jacobian evaluated at all ANS sampling points
  vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> > jac_cur_sps(num_sp);
  // pointer to derivs evaluated at all sampling points
  vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMNOD_WEG6> >* deriv_sp = NULL;   //derivs eval. at all sampling points
  // evaluate all necessary variables for ANS
  soshw6_anssetup(xrefe,xcurr,&deriv_sp,jac_sps,jac_cur_sps,B_ans_loc);
  // (r,s) gp-locations of fully integrated linear 6-node wedge
  // necessary for ANS interpolation
  const DRT::UTILS::GaussRule3D gaussrule_ = DRT::UTILS::intrule_wedge_6point;
  const DRT::UTILS::IntegrationPoints3D intpoints = getIntegrationPoints3D(gaussrule_);

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp=0; gp<NUMGPT_WEG6; ++gp) {

    /* compute the Jacobian matrix which looks like:
    **         [ x_,r  y_,r  z_,r ]
    **     J = [ x_,s  y_,s  z_,s ]
    **         [ x_,t  y_,t  z_,t ]
    */
    LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> jac;
    jac.Multiply(derivs[gp],xrefe);

    // compute determinant of Jacobian by Sarrus' rule
    double detJ = jac.Determinant();
    if (abs(detJ) < 1E-16) dserror("ZERO JACOBIAN DETERMINANT");
    else if (detJ < 0.0) dserror("NEGATIVE JACOBIAN DETERMINANT");

    /* compute the CURRENT Jacobian matrix which looks like:
    **         [ xcurr_,r  ycurr_,r  zcurr_,r ]
    **  Jcur = [ xcurr_,s  ycurr_,s  zcurr_,s ]
    **         [ xcurr_,t  ycurr_,t  zcurr_,t ]
    ** Used to transform the global displacements into parametric space
    */
    LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> jac_cur;
    jac_cur.Multiply(derivs[gp],xcurr);

    // need gp-locations for ANS interpolation
    const double r = intpoints.qxg[gp][0];
    const double s = intpoints.qxg[gp][1];
    //const double t = intpoints.qxg[gp][2]; // not needed

    // set up B-Operator in local(parameter) element space including ANS
    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,NUMDOF_WEG6> bop_loc;
    for (int inode = 0; inode < NUMNOD_WEG6; ++inode) {
      for (int dim = 0; dim < NUMDIM_WEG6; ++dim) {
        // B_loc_rr = N_r.X_r
        bop_loc(0,inode*3+dim) = derivs[gp](0,inode) * jac_cur(0,dim);
        // B_loc_ss = N_s.X_s
        bop_loc(1,inode*3+dim) = derivs[gp](1,inode) * jac_cur(1,dim);
        // B_loc_tt = interpolation along (r x s) of ANS B_loc_tt
        //          = (1-r-s) * B_ans(SP C) + r * B_ans(SP D) + s * B_ans(SP E)
        bop_loc(2,inode*3+dim) = (1-r-s) * B_ans_loc(0+2*num_ans,inode*3+dim)
                                + r      * B_ans_loc(0+3*num_ans,inode*3+dim)
                                + s      * B_ans_loc(0+4*num_ans,inode*3+dim);
        // B_loc_rs = N_r.X_s + N_s.X_r
        bop_loc(3,inode*3+dim) = derivs[gp](0,inode) * jac_cur(1,dim)
                                +derivs[gp](1,inode) * jac_cur(0,dim);
        // B_loc_st = interpolation along r of ANS B_loc_st
        //          = r * B_ans(SP B)
        bop_loc(4,inode*3+dim) = r * B_ans_loc(1+1*num_ans,inode*3+dim);
        // B_loc_rt = interpolation along s of ANS B_loc_rt
        //          = s * B_ans(SP A)
        bop_loc(5,inode*3+dim) = s * B_ans_loc(2+0*num_ans,inode*3+dim);

//        // testing without ans:
//        bop_loc(2,inode*3+dim) = deriv_gp(2,inode) * jac_cur(2,dim);
//        bop_loc(4,inode*3+dim) = deriv_gp(1,inode) * jac_cur(2,dim)
//                                +deriv_gp(2,inode) * jac_cur(1,dim);
//        bop_loc(5,inode*3+dim) = deriv_gp(0,inode) * jac_cur(2,dim)
//                                +deriv_gp(2,inode) * jac_cur(0,dim);
      }
    }

    // transformation from local (parameter) element space to global(material) space
    // with famous 'T'-matrix already used for EAS but now evaluated at each gp
    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,NUMSTR_WEG6> TinvT;
    soshw6_evaluateT(jac,TinvT);
    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,NUMDOF_WEG6> bop;
    bop.Multiply(TinvT,bop_loc);

    // local GL strain vector lstrain={Err,Ess,Ett,2*Ers,2*Est,2*Ert}
    // but with modified ANS strains Ett, Est and Ert
    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,1> lstrain;
    // evaluate glstrains in local(parameter) coords
    // Err = 0.5 * (dx/dr * dx/dr^T - dX/dr * dX/dr^T)
    lstrain(0)= 0.5 * (
       +(jac_cur(0,0)*jac_cur(0,0) + jac_cur(0,1)*jac_cur(0,1) + jac_cur(0,2)*jac_cur(0,2))
       -(jac(0,0)*jac(0,0)         + jac(0,1)*jac(0,1)         + jac(0,2)*jac(0,2)));
    // Ess = 0.5 * (dy/ds * dy/ds^T - dY/ds * dY/ds^T)
    lstrain(1)= 0.5 * (
       +(jac_cur(1,0)*jac_cur(1,0) + jac_cur(1,1)*jac_cur(1,1) + jac_cur(1,2)*jac_cur(1,2))
       -(jac(1,0)*jac(1,0)         + jac(1,1)*jac(1,1)         + jac(1,2)*jac(1,2)));
    // Ers = (dx/ds * dy/dr^T - dX/ds * dY/dr^T)
    lstrain(3)= (
       +(jac_cur(0,0)*jac_cur(1,0) + jac_cur(0,1)*jac_cur(1,1) + jac_cur(0,2)*jac_cur(1,2))
       -(jac(0,0)*jac(1,0)         + jac(0,1)*jac(1,1)         + jac(0,2)*jac(1,2)));

//    // testing without ans:
//    lstrain(2)= 0.5 * (
//       +(jac_cur(2,0)*jac_cur(2,0) + jac_cur(2,1)*jac_cur(2,1) + jac_cur(2,2)*jac_cur(2,2))
//       -(jac(2,0)*jac(2,0)         + jac(2,1)*jac(2,1)         + jac(2,2)*jac(2,2)));
//    lstrain(4)= (
//       +(jac_cur(1,0)*jac_cur(2,0) + jac_cur(1,1)*jac_cur(2,1) + jac_cur(1,2)*jac_cur(2,2))
//       -(jac(1,0)*jac(2,0)         + jac(1,1)*jac(2,1)         + jac(1,2)*jac(2,2)));
//    lstrain(5)= (
//       +(jac_cur(0,0)*jac_cur(2,0) + jac_cur(0,1)*jac_cur(2,1) + jac_cur(0,2)*jac_cur(2,2))
//       -(jac(0,0)*jac(2,0)         + jac(0,1)*jac(2,1)         + jac(0,2)*jac(2,2)));

    // ANS modification of strains ************************************** ANS
    double dydt_A = 0.0; double dYdt_A = 0.0; const int spA = 0;
    double dxdt_B = 0.0; double dXdt_B = 0.0; const int spB = 1;
    double dzdt_C = 0.0; double dZdt_C = 0.0; const int spC = 2;
    double dzdt_D = 0.0; double dZdt_D = 0.0; const int spD = 3;
    double dzdt_E = 0.0; double dZdt_E = 0.0; const int spE = 4;

    const int xdir = 0; // index to matrix x-row, r-row respectively
    const int ydir = 1; // index to matrix y-row, s-row respectively
    const int zdir = 2; // index to matrix z-row, t-row respectively

    // vector product of rows of jacobians at corresponding sampling point
    for (int dim = 0; dim < NUMDIM_WEG6; ++dim) {
      dydt_A += jac_cur_sps[spA](xdir,dim) * jac_cur_sps[spA](zdir,dim);
      dYdt_A += jac_sps[spA](xdir,dim)     * jac_sps[spA](zdir,dim);
      dxdt_B += jac_cur_sps[spB](ydir,dim) * jac_cur_sps[spB](zdir,dim);
      dXdt_B += jac_sps[spB](ydir,dim)     * jac_sps[spB](zdir,dim);

      dzdt_C += jac_cur_sps[spC](zdir,dim) * jac_cur_sps[spC](zdir,dim);
      dZdt_C += jac_sps[spC](zdir,dim)     * jac_sps[spC](zdir,dim);
      dzdt_D += jac_cur_sps[spD](zdir,dim) * jac_cur_sps[spD](zdir,dim);
      dZdt_D += jac_sps[spD](zdir,dim)     * jac_sps[spD](zdir,dim);
      dzdt_E += jac_cur_sps[spE](zdir,dim) * jac_cur_sps[spE](zdir,dim);
      dZdt_E += jac_sps[spE](zdir,dim)     * jac_sps[spE](zdir,dim);
}
    // E33: remedy of curvature thickness locking
    // Ett = 0.5* ( (1-r-s) * Ett(SP C) + r * Ett(SP D) + s * Ett(SP E) )
    lstrain(2) = 0.5 * ( (1-r-s) * (dzdt_C - dZdt_C)
                        + r * (dzdt_D - dZdt_D)
                        + s * (dzdt_E - dZdt_E));
    // E23: remedy of transverse shear locking
    // Est = r * Est(SP B)
    lstrain(4) = r * (dxdt_B - dXdt_B);
    // E13: remedy of transverse shear locking
    // Ert = s * Est(SP A)
    lstrain(5) = s * (dydt_A - dYdt_A);
    // ANS modification of strains ************************************** ANS

    // transformation of local glstrains 'back' to global(material) space
    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,1> glstrain(true);
    glstrain.Multiply(TinvT,lstrain);

    // EAS technology: "enhance the strains"  ----------------------------- EAS
    if (eastype_ == soshw6_easpoisthick) {
      // map local M to global, also enhancement is refered to element origin
      // M = detJ0/detJ T0^{-T} . M
      LINALG::DENSEFUNCTIONS::multiply<NUMSTR_WEG6,NUMSTR_WEG6,soshw6_easpoisthick>(M.A(),detJ0/detJ,T0invT.A(),M_GP->at(gp).A());
      // add enhanced strains = M . alpha to GL strains to "unlock" element
      LINALG::DENSEFUNCTIONS::multiply<NUMSTR_WEG6,soshw6_easpoisthick,1>(1.0,glstrain.A(),1.0,M.A(),(*alpha).A());
    } // ------------------------------------------------------------------ EAS

    // return gp strains (only in case of stress/strain output)
    if (elestress != NULL){
      for (int i = 0; i < 3; ++i) {
        (*elestrain)(gp,i) = glstrain(i);
      }
      for (int i = 3; i < 6; ++i) {
        (*elestrain)(gp,i) = 0.5 * glstrain(i);
      }
    }

    /* call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ** Here all possible material laws need to be incorporated,
    ** the stress vector, a C-matrix, and a density must be retrieved,
    ** every necessary data must be passed.
    */
    double density = 0.0;
    // Caution!! the defgrd can not be modified with ANS to remedy locking
    // therefore it is empty and passed only for compatibility reasons
    LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> defgrd; // Caution!! empty!!
//#define disp_based_F
#ifdef disp_based_F
    Epetra_SerialDenseMatrix defgrd_epetra(View,defgrd->A(),defgrd->Rows(),defgrd->Rows(),defgrd->Columns());
    LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> invJ;
    invJ.Multiply(derivs[gp],xrefe);
    invJ.Invert();
    LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMNOD_WEG6> N_XYZ;
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.Multiply(invJ,derivs[gp]);
    // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
    LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> defgrd;
    defgrd.MultiplyTT(xcurr,N_XYZ);
    for (int i = 0; i < NUMDIM_WEG6; ++i) {
      for (int j = 0; j < NUMDIM_WEG6; ++j) {
        defgrd_epetra(i,j) = defgrd(i,j);
      }
    }
#endif
    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,NUMSTR_WEG6> cmat(true);
    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,1> stress(true);
    sow6_mat_sel(&stress,&cmat,&density,&glstrain, &defgrd, gp, params);
    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    // return gp stresses
    if (elestress != NULL){
      if (!cauchy) {                       // return 2nd Piola-Kirchhoff stresses
        for (int i = 0; i < NUMSTR_WEG6; ++i) {
          (*elestress)(gp,i) = stress(i);
        }
      }
      else {                               // return Cauchy stresses
        dserror("output of Cauchy stresses not supported for solid shell wedge6");
      }
    }

    double detJ_w = detJ*gpweights[gp];
    if (force != NULL && stiffmatrix != NULL) {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      force->MultiplyTN(detJ_w, bop, stress, 1.0);

      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6, NUMDOF_WEG6> cb;
      cb.Multiply(cmat,bop); // temporary C . B
      stiffmatrix->MultiplyTN(detJ_w,bop,cb,1.0);

      // integrate `geometric' stiffness matrix and add to keu *****************
      // here also the ANS interpolation comes into play
      for (int inod=0; inod<NUMNOD_WEG6; ++inod) {
        for (int jnod=0; jnod<NUMNOD_WEG6; ++jnod) {
          LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,1> G_ij;
          G_ij(0) = derivs[gp](0, inod) * derivs[gp](0, jnod); // rr-dir
          G_ij(1) = derivs[gp](1, inod) * derivs[gp](1, jnod); // ss-dir
          G_ij(3) = derivs[gp](0, inod) * derivs[gp](1, jnod)
                  + derivs[gp](1, inod) * derivs[gp](0, jnod); // rs-dir

//          // testing without ANS:
//          G_ij(2) = derivs[gp](2, inod) * derivs[gp](2, jnod); // tt-dir
//          G_ij(4) = derivs[gp](1, inod) * derivs[gp](2, jnod)
//                  + derivs[gp](2, inod) * derivs[gp](1, jnod); // st-dir
//          G_ij(5) = derivs[gp](0, inod) * derivs[gp](2, jnod)
//                  + derivs[gp](2, inod) * derivs[gp](0, jnod); // rt-dir

          // ANS modification in tt-dir
          G_ij(2) = (1-r-s) * (*deriv_sp)[spC](zdir, inod)
                            * (*deriv_sp)[spC](zdir, jnod)
                   + r * (*deriv_sp)[spD](zdir, inod)
                       * (*deriv_sp)[spD](zdir, jnod)
                   + s * (*deriv_sp)[spE](zdir, inod)
                       * (*deriv_sp)[spE](zdir, jnod);
          // ANS modification in st-dir
          G_ij(4) = r * ((*deriv_sp)[spB](ydir, inod)
                          *(*deriv_sp)[spB](zdir, jnod)
                          +(*deriv_sp)[spB](zdir, inod)
                          *(*deriv_sp)[spB](ydir, jnod));
          // ANS modification in rt-dir
          G_ij(5) = s * ((*deriv_sp)[spA](xdir, inod)
                          *(*deriv_sp)[spA](zdir, jnod)
                          +(*deriv_sp)[spA](zdir, inod)
                          *(*deriv_sp)[spA](xdir, jnod));
          // transformation of local(parameter) space 'back' to global(material) space
          LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,1> G_ij_glob;
          G_ij_glob.Multiply(TinvT, G_ij);

          // Scalar Gij results from product of G_ij with stress, scaled with detJ*weights
          double Gij = detJ_w * stress.Dot(G_ij_glob);

          // add "geometric part" Gij times detJ*weights to stiffness matrix
          (*stiffmatrix)(NUMDIM_WEG6*inod+0, NUMDIM_WEG6*jnod+0) += Gij;
          (*stiffmatrix)(NUMDIM_WEG6*inod+1, NUMDIM_WEG6*jnod+1) += Gij;
          (*stiffmatrix)(NUMDIM_WEG6*inod+2, NUMDIM_WEG6*jnod+2) += Gij;
        }
      } // end of intergrate `geometric' stiffness ******************************

      // EAS technology: integrate matrices --------------------------------- EAS
      if (eastype_ == soshw6_easpoisthick) {
        // integrate Kaa: Kaa += (M^T . cmat . M) * detJ * w(gp)
        LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,soshw6_easpoisthick> cM; // temporary c . M
        cM.Multiply(cmat, M);
        LINALG::DENSEFUNCTIONS::multiplyTN<soshw6_easpoisthick,NUMSTR_WEG6,soshw6_easpoisthick>(1.0, Kaa.A(), detJ_w, M.A(), cM.A());

        // integrate Kda: Kda += (M^T . cmat . B) * detJ * w(gp)
        LINALG::DENSEFUNCTIONS::multiplyTN<soshw6_easpoisthick,NUMSTR_WEG6,NUMDOF_WEG6>(1.0, Kda.A(), detJ_w, M.A(), cb.A());

        // integrate feas: feas += (M^T . sigma) * detJ *wp(gp)
        LINALG::DENSEFUNCTIONS::multiplyTN<soshw6_easpoisthick,NUMSTR_WEG6,1>(1.0, feas.A(), detJ_w, M.A(), stress.A());
      } // ------------------------------------------------------------------ EAS
    }

    if (massmatrix != NULL){ // evaluate mass matrix +++++++++++++++++++++++++
      // integrate consistent mass matrix
      const double factor = detJ_w * density;
      double ifactor, massfactor;
      for (int inod=0; inod<NUMNOD_WEG6; ++inod)
      {
        ifactor = shapefcts[gp](inod) * factor;
        for (int jnod=0; jnod<NUMNOD_WEG6; ++jnod)
        {
          massfactor = shapefcts[gp](jnod) * ifactor;     // intermediate factor
          (*massmatrix)(NUMDIM_WEG6*inod+0,NUMDIM_WEG6*jnod+0) += massfactor;
          (*massmatrix)(NUMDIM_WEG6*inod+1,NUMDIM_WEG6*jnod+1) += massfactor;
          (*massmatrix)(NUMDIM_WEG6*inod+2,NUMDIM_WEG6*jnod+2) += massfactor;
        }
      }
    } // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++
   /* =========================================================================*/
  }/* ==================================================== end of Loop over GP */
   /* =========================================================================*/

  if (force != NULL && stiffmatrix != NULL) {
    // EAS technology: ------------------------------------------------------ EAS
    // subtract EAS matrices from disp-based Kdd to "soften" element
    if (eastype_ == soshw6_easpoisthick) {
      // we need the inverse of Kaa
      Epetra_SerialDenseSolver solve_for_inverseKaa;
      solve_for_inverseKaa.SetMatrix(Kaa);
      solve_for_inverseKaa.Invert();

      LINALG::SerialDenseMatrix KdaTKaa(NUMDOF_WEG6, soshw6_easpoisthick); // temporary Kda^T.Kaa^{-1}
      LINALG::DENSEFUNCTIONS::multiplyTN<NUMDOF_WEG6,soshw6_easpoisthick,soshw6_easpoisthick>(KdaTKaa, Kda, Kaa);

      // EAS-stiffness matrix is: Kdd - Kda^T . Kaa^-1 . Kda
      LINALG::DENSEFUNCTIONS::multiply<NUMDOF_WEG6,soshw6_easpoisthick,NUMDOF_WEG6>(1.0, stiffmatrix->A(), -1.0, KdaTKaa.A(), Kda.A());

      // EAS-internal force is: fint - Kda^T . Kaa^-1 . feas
      LINALG::DENSEFUNCTIONS::multiply<NUMDOF_WEG6,soshw6_easpoisthick,1>(1.0, force->A(), -1.0, KdaTKaa.A(), feas.A());

      // store current EAS data in history
      for (int i=0; i<soshw6_easpoisthick; ++i) {
        for (int j=0; j<soshw6_easpoisthick; ++j) (*oldKaainv)(i,j) = Kaa(i,j);
        for (int j=0; j<NUMDOF_WEG6; ++j) (*oldKda)(i, j) = Kda(i,j);
        (*oldfeas)(i, 0) = feas(i);
      }
    } // -------------------------------------------------------------------- EAS
  }

  return;
}


/*----------------------------------------------------------------------*
 |  setup of constant ANS data (private)                       maf 05/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_shw6::soshw6_anssetup(
        const LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,NUMDIM_WEG6>& xrefe, // material element coords
        const LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,NUMDIM_WEG6>& xcurr, // current element coords
        vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMNOD_WEG6> >** deriv_sp,   // derivs eval. at all sampling points
        vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> >& jac_sps,     // jac at all sampling points
        vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> >& jac_cur_sps, // current jac at all sampling points
        LINALG::FixedSizeSerialDenseMatrix<num_ans*num_sp,NUMDOF_WEG6>& B_ans_loc) // modified B
{
  // static matrix object of derivs at sampling points, kept in memory
  static vector<LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMNOD_WEG6> > df_sp(num_sp);
  static bool dfsp_eval;                      // flag for re-evaluate everything

  if (dfsp_eval!=0){             // if true f,df already evaluated
    *deriv_sp = &df_sp;         // return adress of static object to target of pointer
  } else {
  /*====================================================================*/
  /* 6-node wedge Solid-Shell node topology
   * and location of sampling points A to E                             */
  /*--------------------------------------------------------------------*/
  /*
   *                             s
   *                   6        /
   *                 //||\\   /
   *      t        //  ||   \\
   *      ^      //    || /    \\
   *      |    //      E          \\
   *      |  //        ||            \\
   *      |//       /  ||               \\
   *      5===============================6
   *     ||      B      3                 ||
   *     ||    /      // \\               ||
   *     || /       //      \\            ||
   *   - C -  -  -// -  A  -  -\\ -  -   -D  ----> r
   *     ||     //                \\      ||
   *  /  ||   //                     \\   ||
   *     || //                          \\||
   *      1================================2
   *
   */
  /*====================================================================*/
    // (r,s,t) gp-locations of sampling points A,B,C,D,E
    // numsp = 5 here set explicitly to allow direct initializing
    double r[5] = { 0.5, 0.0, 0.0, 1.0, 0.0};
    double s[5] = { 0.0, 0.5, 0.0, 0.0, 1.0};
    double t[5] = { 0.0, 0.0, 0.0, 0.0, 0.0};

    // fill up df_sp w.r.t. rst directions (NUMDIM) at each sp
    for (int i=0; i<num_sp; ++i) {
      DRT::UTILS::shape_function_3D_deriv1(df_sp[i], r[i], s[i], t[i], wedge6);
    }

    // return adresses of just evaluated matrices
    *deriv_sp = &df_sp;         // return adress of static object to target of pointer
    dfsp_eval = 1;               // now all arrays are filled statically
  }

  for (int sp=0; sp<num_sp; ++sp){
    // compute Jacobian matrix at all sampling points
    jac_sps[sp].Multiply(df_sp[sp],xrefe);
    // compute CURRENT Jacobian matrix at all sampling points
    jac_cur_sps[sp].Multiply(df_sp[sp],xcurr);
  }

  /*
  ** Compute modified B-operator in local(parametric) space,
  ** evaluated at all sampling points
  */
  // loop over each sampling point
  LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> jac_cur;
  for (int sp = 0; sp < num_sp; ++sp) {
    /* compute the CURRENT Jacobian matrix at the sampling point:
    **         [ xcurr_,r  ycurr_,r  zcurr_,r ]
    **  Jcur = [ xcurr_,s  ycurr_,s  zcurr_,s ]
    **         [ xcurr_,t  ycurr_,t  zcurr_,t ]
    ** Used to transform the global displacements into parametric space
    */
    jac_cur.Multiply(df_sp[sp],xcurr);

    // fill up B-operator
    for (int inode = 0; inode < NUMNOD_WEG6; ++inode) {
      for (int dim = 0; dim < NUMDIM_WEG6; ++dim) {
        // modify B_loc_tt = N_t.X_t
        B_ans_loc(sp*num_ans+0,inode*3+dim) = df_sp[sp](2,inode)*jac_cur(2,dim);
        // modify B_loc_st = N_s.X_t + N_t.X_s
        B_ans_loc(sp*num_ans+1,inode*3+dim) = df_sp[sp](1,inode)*jac_cur(2,dim)
                                            +df_sp[sp](2,inode)*jac_cur(1,dim);
        // modify B_loc_rt = N_r.X_t + N_t.X_r
        B_ans_loc(sp*num_ans+2,inode*3+dim) = df_sp[sp](0,inode)*jac_cur(2,dim)
                                            +df_sp[sp](2,inode)*jac_cur(0,dim);
      }
    }
  }


  return;
}

/*----------------------------------------------------------------------*
 |  evaluate 'T'-transformation matrix )                       maf 05/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_shw6::soshw6_evaluateT(const LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6>& jac,
                                                    LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,NUMSTR_WEG6>& TinvT)
{
  // build T^T transformation matrix which maps
  // between global (r,s,t)-coordinates and local (x,y,z)-coords
  // later, invert the transposed to map from local to global
  // see literature for details (e.g. Andelfinger)
  // it is based on the voigt notation for strains: xx,yy,zz,xy,yz,xz
  TinvT(0,0) = jac(0,0) * jac(0,0);
  TinvT(1,0) = jac(1,0) * jac(1,0);
  TinvT(2,0) = jac(2,0) * jac(2,0);
  TinvT(3,0) = 2 * jac(0,0) * jac(1,0);
  TinvT(4,0) = 2 * jac(1,0) * jac(2,0);
  TinvT(5,0) = 2 * jac(0,0) * jac(2,0);

  TinvT(0,1) = jac(0,1) * jac(0,1);
  TinvT(1,1) = jac(1,1) * jac(1,1);
  TinvT(2,1) = jac(2,1) * jac(2,1);
  TinvT(3,1) = 2 * jac(0,1) * jac(1,1);
  TinvT(4,1) = 2 * jac(1,1) * jac(2,1);
  TinvT(5,1) = 2 * jac(0,1) * jac(2,1);

  TinvT(0,2) = jac(0,2) * jac(0,2);
  TinvT(1,2) = jac(1,2) * jac(1,2);
  TinvT(2,2) = jac(2,2) * jac(2,2);
  TinvT(3,2) = 2 * jac(0,2) * jac(1,2);
  TinvT(4,2) = 2 * jac(1,2) * jac(2,2);
  TinvT(5,2) = 2 * jac(0,2) * jac(2,2);

  TinvT(0,3) = jac(0,0) * jac(0,1);
  TinvT(1,3) = jac(1,0) * jac(1,1);
  TinvT(2,3) = jac(2,0) * jac(2,1);
  TinvT(3,3) = jac(0,0) * jac(1,1) + jac(1,0) * jac(0,1);
  TinvT(4,3) = jac(1,0) * jac(2,1) + jac(2,0) * jac(1,1);
  TinvT(5,3) = jac(0,0) * jac(2,1) + jac(2,0) * jac(0,1);


  TinvT(0,4) = jac(0,1) * jac(0,2);
  TinvT(1,4) = jac(1,1) * jac(1,2);
  TinvT(2,4) = jac(2,1) * jac(2,2);
  TinvT(3,4) = jac(0,1) * jac(1,2) + jac(1,1) * jac(0,2);
  TinvT(4,4) = jac(1,1) * jac(2,2) + jac(2,1) * jac(1,2);
  TinvT(5,4) = jac(0,1) * jac(2,2) + jac(2,1) * jac(0,2);

  TinvT(0,5) = jac(0,0) * jac(0,2);
  TinvT(1,5) = jac(1,0) * jac(1,2);
  TinvT(2,5) = jac(2,0) * jac(2,2);
  TinvT(3,5) = jac(0,0) * jac(1,2) + jac(1,0) * jac(0,2);
  TinvT(4,5) = jac(1,0) * jac(2,2) + jac(2,0) * jac(1,2);
  TinvT(5,5) = jac(0,0) * jac(2,2) + jac(2,0) * jac(0,2);

  // now evaluate T^{-T} with solver
  LINALG::FixedSizeSerialDenseSolver<NUMSTR_WEG6,NUMSTR_WEG6,1> solve_for_inverseT;
  solve_for_inverseT.SetMatrix(TinvT);
  int err2 = solve_for_inverseT.Factor();
  int err = solve_for_inverseT.Invert();
  if ((err != 0) && (err2!=0)) dserror("Inversion of Tinv (Jacobian) failed");
  return;
}

/*----------------------------------------------------------------------*
 |  initialize EAS data (private)                              maf 05/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_shw6::soshw6_easinit()
{
  // all parameters are stored in Epetra_SerialDenseMatrix as only
  // those can be added to DRT::Container

  // EAS enhanced strain parameters at currently investigated load/time step
  Epetra_SerialDenseMatrix alpha(neas_,1);
  // EAS enhanced strain parameters of last converged load/time step
  Epetra_SerialDenseMatrix alphao(neas_,1);
  // EAS portion of internal forces, also called enhacement vector s or Rtilde
  Epetra_SerialDenseMatrix feas(neas_, 1);
  // EAS matrix K_{alpha alpha}, also called Dtilde
  Epetra_SerialDenseMatrix invKaa(neas_,neas_);
  // EAS matrix K_{d alpha}
  Epetra_SerialDenseMatrix Kda(neas_,NUMDOF_WEG6);

  // save EAS data into element container
  data_.Add("alpha",alpha);
  data_.Add("alphao",alphao);
  data_.Add("feas",feas);
  data_.Add("invKaa",invKaa);
  data_.Add("Kda",Kda);

  return;
}

/*----------------------------------------------------------------------*
 |  setup of constant EAS data (private)                       maf 05/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_shw6::soshw6_eassetup(
          vector<Epetra_SerialDenseMatrix>** M_GP,    // M-matrix evaluated at GPs
          double& detJ0,                      // det of Jacobian at origin
          LINALG::FixedSizeSerialDenseMatrix<NUMSTR_WEG6,NUMSTR_WEG6>& T0invT,   // maps M(origin) local to global
          const LINALG::FixedSizeSerialDenseMatrix<NUMNOD_WEG6,NUMDIM_WEG6>& xrefe)    // material element coords
{
  // shape function derivatives, evaluated at origin (r=s=t=0.0)
  const DRT::UTILS::IntegrationPoints3D  intpoints = getIntegrationPoints3D(DRT::UTILS::intrule_wedge_1point);
  LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMNOD_WEG6> df0;
  DRT::UTILS::shape_function_3D_deriv1(df0,intpoints.qxg[0][0],intpoints.qxg[0][1],intpoints.qxg[0][2],DRT::Element::wedge6);

  // compute Jacobian, evaluated at element origin (r=s=t=0.0)
  LINALG::FixedSizeSerialDenseMatrix<NUMDIM_WEG6,NUMDIM_WEG6> jac0;
  jac0.Multiply(df0,xrefe);

  // compute determinant of Jacobian at origin by Sarrus' rule
  detJ0 = jac0.Determinant();

  // get T-matrix at element origin
  soshw6_evaluateT(jac0,T0invT);

  // build EAS interpolation matrix M, evaluated at the GPs of soshw6
  static vector<Epetra_SerialDenseMatrix> M(NUMGPT_WEG6);
  static bool M_eval;

  if (M_eval==true){          // if true M already evaluated
      *M_GP = &M;             // return adress of static object to target of pointer
    return;
  } else {
    // (r,s,t) gp-locations of fully integrated linear 6-node Wedge
    const DRT::UTILS::IntegrationPoints3D intpoints = getIntegrationPoints3D(DRT::UTILS::intrule_wedge_6point);

    // fill up M at each gp
    if (eastype_ == soshw6_easpoisthick) {
      /* EAS interpolation of 1 mode corr. to linear thickness strain
      **            0
      **            0
      **    M =     t
      **            0
      **            0
      **            0
      */
      for (int i=0; i<intpoints.nquad; ++i) {
        M[i].Shape(NUMSTR_WEG6,soshw6_easpoisthick);
        M[i](2,0) = intpoints.qxg[i][2];  // t at gp
        //M[i](2,1) = intpoints.qxg[i][0]*intpoints.qxg[i][2];  // r*t at gp
        //M[i](2,2) = intpoints.qxg[i][1]*intpoints.qxg[i][2];  // s*t at gp
      }

      // return adress of just evaluated matrix
      *M_GP = &M;            // return adress of static object to target of pointer
      M_eval = true;         // now the array is filled statically
    } else {
    dserror("eastype not implemented");
    }
  }
}


#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_WEG6
