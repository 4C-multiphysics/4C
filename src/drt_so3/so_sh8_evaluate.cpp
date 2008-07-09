/*!----------------------------------------------------------------------
\file so_sh8_evaluate.cpp
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
#include "so_sh8.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_exporter.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/linalg_serialdensematrix.H"
#include "../drt_lib/linalg_serialdensevector.H"
#include "Epetra_SerialDenseSolver.h"
#include "../drt_io/io_gmsh.H"
#include "Epetra_Time.h"
#include "Teuchos_TimeMonitor.hpp"
#include "../drt_mat/visconeohooke.H"

using namespace std; // cout etc.
using namespace LINALG; // our linear algebra

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              maf 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_sh8::Evaluate(ParameterList&            params,
                                    DRT::Discretization&      discretization,
                                    vector<int>&              lm,
                                    Epetra_SerialDenseMatrix& elemat1,
                                    Epetra_SerialDenseMatrix& elemat2,
                                    Epetra_SerialDenseVector& elevec1,
                                    Epetra_SerialDenseVector& elevec2,
                                    Epetra_SerialDenseVector& elevec3)
{
  // start with "none"
  DRT::ELEMENTS::So_hex8::ActionType act = So_hex8::none;

  // get the required action
  string action = params.get<string>("action","none");
  if (action == "none") dserror("No action supplied");
  else if (action=="calc_struct_linstiff")        act = So_hex8::calc_struct_linstiff;
  else if (action=="calc_struct_nlnstiff")        act = So_hex8::calc_struct_nlnstiff;
  else if (action=="calc_struct_internalforce")   act = So_hex8::calc_struct_internalforce;
  else if (action=="calc_struct_linstiffmass")    act = So_hex8::calc_struct_linstiffmass;
  else if (action=="calc_struct_nlnstiffmass")    act = So_hex8::calc_struct_nlnstiffmass;
  else if (action=="calc_struct_nlnstifflmass")   act = So_hex8::calc_struct_nlnstifflmass;
  else if (action=="calc_struct_stress")          act = So_hex8::calc_struct_stress;
  else if (action=="calc_struct_eleload")         act = So_hex8::calc_struct_eleload;
  else if (action=="calc_struct_fsiload")         act = So_hex8::calc_struct_fsiload;
  else if (action=="calc_struct_update_istep")    act = So_hex8::calc_struct_update_istep;
  else if (action=="calc_struct_update_imrlike")  act = So_hex8::calc_struct_update_imrlike;
  else if (action=="calc_homog_stressdens")       act = So_hex8::calc_homog_stressdens;
  else if (action=="postprocess_stress")          act = So_hex8::postprocess_stress;
  else dserror("Unknown type of action for So_hex8");

  // what should the element do
  switch(act) {
    // linear stiffness
    case calc_struct_linstiff: {
      // need current displacement and residual forces
      vector<double> mydisp(lm.size());
      for (int i=0; i<(int)mydisp.size(); ++i) mydisp[i] = 0.0;
      vector<double> myres(lm.size());
      for (int i=0; i<(int)myres.size(); ++i) myres[i] = 0.0;
      // decide whether evaluate 'thin' sosh stiff or 'thick' so_hex8 stiff
      if (Type() == DRT::Element::element_sosh8){
        sosh8_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,params);
      } else if (Type() == DRT::Element::element_so_hex8){
        soh8_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,params);
      }
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
      // decide whether evaluate 'thin' sosh stiff or 'thick' so_hex8 stiff
      if (Type() == DRT::Element::element_sosh8){
        sosh8_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,params);
      } else if (Type() == DRT::Element::element_so_hex8){
        soh8_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,params);
      }
    }
    break;

    // internal force vector only
    case calc_struct_internalforce: {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      // create a dummy element matrix to apply linearised EAS-stuff onto
      Epetra_SerialDenseMatrix myemat(lm.size(),lm.size());
      // decide whether evaluate 'thin' sosh stiff or 'thick' so_hex8 stiff
      if (Type() == DRT::Element::element_sosh8) {
        sosh8_nlnstiffmass(lm,mydisp,myres,&myemat,NULL,&elevec1,NULL,NULL,params);
      } else if (Type() == DRT::Element::element_so_hex8) {
        soh8_nlnstiffmass(lm,mydisp,myres,&myemat,NULL,&elevec1,NULL,NULL,params);
      }
    }
    break;

    // linear stiffness and consistent mass matrix
    case calc_struct_linstiffmass:
      dserror("Case 'calc_struct_linstiffmass' not yet implemented");
    break;

    // nonlinear stiffness, internal force vector, and consistent/lumped mass matrix
    case calc_struct_nlnstiffmass:
    case calc_struct_nlnstifflmass: {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      // decide whether evaluate 'thin' sosh stiff or 'thick' so_hex8 stiff
      if (Type() == DRT::Element::element_sosh8){
        sosh8_nlnstiffmass(lm,mydisp,myres,&elemat1,&elemat2,&elevec1,NULL,NULL,params);
      } else if (Type() == DRT::Element::element_so_hex8){
        soh8_nlnstiffmass(lm,mydisp,myres,&elemat1,&elemat2,&elevec1,NULL,NULL,params);
      }
      // lump mass
      if (act==calc_struct_nlnstifflmass) soh8_lumpmass(&elemat2);
    }
    break;

    // evaluate stresses and strains at gauss points
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
      Epetra_SerialDenseMatrix stress(NUMGPT_SOH8,NUMSTR_SOH8);
      Epetra_SerialDenseMatrix strain(NUMGPT_SOH8,NUMSTR_SOH8);
      bool cauchy = params.get<bool>("cauchy", false);
      string iostrain = params.get<string>("iostrain", "none");
      // decide whether evaluate 'thin' sosh stiff or 'thick' so_hex8 stiff
      if (Type() == DRT::Element::element_sosh8){
        if (iostrain != "euler_almansi") sosh8_nlnstiffmass(lm,mydisp,myres,NULL,NULL,NULL,&stress,&strain,params,cauchy);
        else    dserror("requested option not yet implemented for solidsh8");
      } else if (Type() == DRT::Element::element_so_hex8){
        if (iostrain == "euler_almansi") soh8_nlnstiffmass(lm,mydisp,myres,NULL,NULL,NULL,&stress,&strain,params,cauchy,true);
        else soh8_nlnstiffmass(lm,mydisp,myres,NULL,NULL,NULL,&stress,&strain,params,cauchy,false);
      }
      AddtoPack(*stressdata, stress);
      AddtoPack(*straindata, strain);
    }
    break;

    // postprocess stresses/strains at gauss points

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
      RCP<Epetra_SerialDenseMatrix> gpstress = (*gpstressmap)[gid];

      if (stresstype=="ndxyz") {
        // extrapolate stresses/strains at Gauss points to nodes
        Epetra_SerialDenseMatrix nodalstresses(NUMNOD_SOH8,NUMSTR_SOH8);
        soh8_expol(*gpstress,nodalstresses);

        // average nodal stresses/strains between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_SOH8);

        for (int i=0;i<NUMNOD_SOH8;++i){
          DRT::Node* node=Nodes()[i];
          numadjele[i]=node->NumElement();
        }

        for (int i=0;i<NUMNOD_SOH8;++i){
          elevec1(3*i)=nodalstresses(i,0)/numadjele[i];
          elevec1(3*i+1)=nodalstresses(i,1)/numadjele[i];
          elevec1(3*i+2)=nodalstresses(i,2)/numadjele[i];
        }
        for (int i=0;i<NUMNOD_SOH8;++i){
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
        if (lid!=-1) {
          for (int i = 0; i < NUMSTR_SOH8; ++i) {
            (*((*elestress)(i)))[lid] = 0.;
            for (int j = 0; j < NUMGPT_SOH8; ++j) {
              //(*((*elestress)(i)))[lid] += 0.125 * (*gpstress)(j,i);
              (*((*elestress)(i)))[lid] += 1.0/NUMGPT_SOH8 * (*gpstress)(j,i);
            }
          }
        }
      }
      else if (stresstype=="cxyz_ndxyz") {
        // extrapolate stresses/strains at Gauss points to nodes
        Epetra_SerialDenseMatrix nodalstresses(NUMNOD_SOH8,NUMSTR_SOH8);
        soh8_expol(*gpstress,nodalstresses);

        // average nodal stresses/strains between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_SOH8);

        for (int i=0;i<NUMNOD_SOH8;++i){
          DRT::Node* node=Nodes()[i];
          numadjele[i]=node->NumElement();
        }

        for (int i=0;i<NUMNOD_SOH8;++i){
          elevec1(3*i)=nodalstresses(i,0)/numadjele[i];
          elevec1(3*i+1)=nodalstresses(i,1)/numadjele[i];
          elevec1(3*i+2)=nodalstresses(i,2)/numadjele[i];
        }
        for (int i=0;i<NUMNOD_SOH8;++i){
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
          for (int i = 0; i < NUMSTR_SOH8; ++i) {
            (*((*elestress)(i)))[lid] = 0.;
            for (int j = 0; j < NUMGPT_SOH8; ++j) {
              //(*((*elestress)(i)))[lid] += 0.125 * (*gpstress)(j,i);
              (*((*elestress)(i)))[lid] += 1.0/NUMGPT_SOH8 * (*gpstress)(j,i);
            }
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
      if (eastype_ != soh8_easnone) {
        Epetra_SerialDenseMatrix* alpha = data_.GetMutable<Epetra_SerialDenseMatrix>("alpha");  // Alpha_{n+1}
        Epetra_SerialDenseMatrix* alphao = data_.GetMutable<Epetra_SerialDenseMatrix>("alphao");  // Alpha_n
        Epetra_BLAS::Epetra_BLAS blas;
        blas.COPY((*alphao).M()*(*alphao).N(), (*alpha).A(), (*alphao).A());  // alphao := alpha
      }
      // Update of history for visco material
      RefCountPtr<MAT::Material> mat = Material();
      if (mat->MaterialType() == m_visconeohooke)
      {
        MAT::ViscoNeoHooke* visco = static_cast <MAT::ViscoNeoHooke*>(mat.get());
        visco->Update();
      }
    }
    break;

    case calc_struct_update_imrlike: {
      // do something with internal EAS, etc parameters
      // this depends on the applied solution technique (static, generalised-alpha,
      // or other time integrators)
      if (eastype_ != soh8_easnone) {
        double alphaf = params.get<double>("alpha f", 0.0);  // generalised-alpha TIS parameter alpha_f
        Epetra_SerialDenseMatrix* alpha = data_.GetMutable<Epetra_SerialDenseMatrix>("alpha");  // Alpha_{n+1-alphaf}
        Epetra_SerialDenseMatrix* alphao = data_.GetMutable<Epetra_SerialDenseMatrix>("alphao");  // Alpha_n
        Epetra_BLAS::Epetra_BLAS blas;
        blas.SCAL((*alphao).M()*(*alphao).N(), -alphaf/(1.0-alphaf), (*alphao).A());  // alphao *= -alphaf/(1.0-alphaf)
        blas.AXPY((*alphao).M()*(*alphao).N(), 1.0/(1.0-alphaf), (*alpha).A(), (*alphao).A());  // alphao += 1.0/(1.0-alphaf) * alpha
        blas.COPY((*alpha).M()*(*alpha).N(), (*alphao).A(), (*alpha).A());  // alpha := alphao
      }
    }
    break;

    case calc_homog_stressdens: {
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      soh8_homog(params, mydisp, myres);
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
void DRT::ELEMENTS::So_sh8::sosh8_nlnstiffmass(
      vector<int>&              lm,             // location matrix
      vector<double>&           disp,           // current displacements
      vector<double>&           residual,       // current residuum
      Epetra_SerialDenseMatrix* stiffmatrix,    // element stiffness matrix
      Epetra_SerialDenseMatrix* massmatrix,     // element mass matrix
      Epetra_SerialDenseVector* force,          // element internal force vector
      Epetra_SerialDenseMatrix* elestress,      // element stresses
      Epetra_SerialDenseMatrix* elestrain,      // strains at GP
      ParameterList&            params,         // algorithmic parameters e.g. time
      const bool                cauchy)         // stress output option
{
/* ============================================================================*
** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_8 with 8 GAUSS POINTS*
** ============================================================================*/
  const static vector<Epetra_SerialDenseVector> shapefcts = sosh8_shapefcts();
  const static vector<Epetra_SerialDenseMatrix> derivs = sosh8_derivs();
  const static vector<double> gpweights = sosh8_weights();
/* ============================================================================*/

  // update element geometry
  LINALG::SerialDenseMatrix xrefe(NUMNOD_SOH8,NUMDIM_SOH8);  // material coord. of element
  LINALG::SerialDenseMatrix xcurr(NUMNOD_SOH8,NUMDIM_SOH8);  // current  coord. of element
  for (int i=0; i<NUMNOD_SOH8; ++i){
    xrefe(i,0) = Nodes()[i]->X()[0];
    xrefe(i,1) = Nodes()[i]->X()[1];
    xrefe(i,2) = Nodes()[i]->X()[2];

    xcurr(i,0) = xrefe(i,0) + disp[i*NODDOF_SOH8+0];
    xcurr(i,1) = xrefe(i,1) + disp[i*NODDOF_SOH8+1];
    xcurr(i,2) = xrefe(i,2) + disp[i*NODDOF_SOH8+2];
  }

  /*
  ** EAS Technology: declare, intialize, set up, and alpha history -------- EAS
  */
  // in any case declare variables, sizes etc. only in eascase
  Epetra_SerialDenseMatrix* alpha = NULL;         // EAS alphas
  vector<Epetra_SerialDenseMatrix>* M_GP = NULL;  // EAS matrix M at all GPs
  LINALG::SerialDenseMatrix M;                    // EAS matrix M at current GP
  Epetra_SerialDenseVector feas;                  // EAS portion of internal forces
  Epetra_SerialDenseMatrix Kaa;                   // EAS matrix Kaa
  Epetra_SerialDenseMatrix Kda;                   // EAS matrix Kda
  double detJ0;                                   // detJ(origin)
  Epetra_SerialDenseMatrix T0invT;                // trafo matrix
  Epetra_SerialDenseMatrix* oldfeas = NULL;       // EAS history
  Epetra_SerialDenseMatrix* oldKaainv = NULL;     // EAS history
  Epetra_SerialDenseMatrix* oldKda = NULL;        // EAS history
  if (eastype_ == soh8_eassosh8) {
    /*
    ** EAS Update of alphas:
    ** the current alphas are (re-)evaluated out of
    ** Kaa and Kda of previous step to avoid additional element call.
    ** This corresponds to the (innermost) element update loop
    ** in the nonlinear FE-Skript page 120 (load-control alg. with EAS)
    */
    //(*alpha).Shape(neas_,1);
    alpha = data_.GetMutable<Epetra_SerialDenseMatrix>("alpha");   // get old alpha
    // evaluate current (updated) EAS alphas (from history variables)
    // get stored EAS history
    oldfeas = data_.GetMutable<Epetra_SerialDenseMatrix>("feas");
    oldKaainv = data_.GetMutable<Epetra_SerialDenseMatrix>("invKaa");
    oldKda = data_.GetMutable<Epetra_SerialDenseMatrix>("Kda");
    if (!alpha || !oldKaainv || !oldKda || !oldfeas) dserror("Missing EAS history-data");

    // we need the (residual) displacement at the previous step
    LINALG::SerialDenseVector res_d(NUMDOF_SOH8);
    for (int i = 0; i < NUMDOF_SOH8; ++i) {
      res_d(i) = residual[i];
    }
    // add Kda . res_d to feas
    (*oldfeas).Multiply('N','N',1.0,(*oldKda),res_d,1.0);
    // "new" alpha is: - Kaa^-1 . (feas + Kda . old_d), here: - Kaa^-1 . feas
    (*alpha).Multiply('N','N',-1.0,(*oldKaainv),(*oldfeas),1.0);
    /* end of EAS Update ******************/

    // EAS portion of internal forces, also called enhacement vector s or Rtilde
    feas.Size(neas_);

    // EAS matrix K_{alpha alpha}, also called Dtilde
    Kaa.Shape(neas_,neas_);

    // EAS matrix K_{d alpha}
    Kda.Shape(neas_,NUMDOF_SOH8);

    // transformation matrix T0, maps M-matrix evaluated at origin
    // between local element coords and global coords
    // here we already get the inverse transposed T0
    T0invT.Shape(NUMSTR_SOH8,NUMSTR_SOH8);

    /* evaluation of EAS variables (which are constant for the following):
    ** -> M defining interpolation of enhanced strains alpha, evaluated at GPs
    ** -> determinant of Jacobi matrix at element origin (r=s=t=0.0)
    ** -> T0^{-T}
    */
    soh8_eassetup(&M_GP,detJ0,T0invT,xrefe);
  } else if (eastype_ == soh8_easnone){
  //cout << "Warning: Solid-Shell8 without EAS" << endl;
  } else dserror("Solid-Shell8 only with eas_sosh8");// ------------------- EAS

  /*
  ** ANS Element technology to remedy
  *  - transverse-shear locking E_rt and E_st
  *  - trapezoidal (curvature-thickness) locking E_tt
  */
  // modified B-operator in local(parameter) element space
  const int num_sp = 8;       // number of ANS sampling points
  const int num_ans = 3;      // number of modified ANS strains (E_rt,E_st,E_tt)
  // ANS modified rows of bop in local(parameter) coords
  Epetra_SerialDenseMatrix B_ans_loc(num_ans*num_sp,NUMDOF_SOH8);
  // Jacobian evaluated at all ANS sampling points
  LINALG::SerialDenseMatrix jac_sps(NUMDIM_SOH8*num_sp,NUMDIM_SOH8);
  // CURRENT Jacobian evaluated at all ANS sampling points
  LINALG::SerialDenseMatrix jac_cur_sps(NUMDIM_SOH8*num_sp,NUMDIM_SOH8);
  // pointer to derivs evaluated at all sampling points
  Epetra_SerialDenseMatrix* deriv_sp; //[NUMDIM_SOH8*numsp][NUMNOD_SOH8]
  // evaluate all necessary variables for ANS
  sosh8_anssetup(num_sp,num_ans,xrefe,xcurr,&deriv_sp,jac_sps,jac_cur_sps,B_ans_loc);
  // (r,s) gp-locations of fully integrated linear 8-node Hex
  // necessary for ANS interpolation
  const double gploc    = 1.0/sqrt(3.0);    // gp sampling point value for linear fct
  const double r[NUMGPT_SOH8] = {-gploc, gploc, gploc,-gploc,-gploc, gploc, gploc,-gploc};
  const double s[NUMGPT_SOH8] = {-gploc,-gploc, gploc, gploc,-gploc,-gploc, gploc, gploc};

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp=0; gp<NUMGPT_SOH8; ++gp) {

    /* compute the Jacobian matrix which looks like:
    **         [ x_,r  y_,r  z_,r ]
    **     J = [ x_,s  y_,s  z_,s ]
    **         [ x_,t  y_,t  z_,t ]
    */
    LINALG::SerialDenseMatrix jac(NUMDIM_SOH8,NUMDIM_SOH8);
    jac.Multiply('N','N',1.0,derivs[gp],xrefe,0.0);

    // compute determinant of Jacobian by Sarrus' rule
    double detJ= jac(0,0) * jac(1,1) * jac(2,2)
               + jac(0,1) * jac(1,2) * jac(2,0)
               + jac(0,2) * jac(1,0) * jac(2,1)
               - jac(0,0) * jac(1,2) * jac(2,1)
               - jac(0,1) * jac(1,0) * jac(2,2)
               - jac(0,2) * jac(1,1) * jac(2,0);
    if (detJ == 0.0) dserror("ZERO JACOBIAN DETERMINANT");
    else if (detJ < 0.0) dserror("NEGATIVE JACOBIAN DETERMINANT");

    /* compute the CURRENT Jacobian matrix which looks like:
    **         [ xcurr_,r  ycurr_,r  zcurr_,r ]
    **  Jcur = [ xcurr_,s  ycurr_,s  zcurr_,s ]
    **         [ xcurr_,t  ycurr_,t  zcurr_,t ]
    ** Used to transform the global displacements into parametric space
    */
    LINALG::SerialDenseMatrix jac_cur(NUMDIM_SOH8,NUMDIM_SOH8);
    jac_cur.Multiply('N','N',1.0,derivs[gp],xcurr,0.0);

    // set up B-Operator in local(parameter) element space including ANS
    LINALG::SerialDenseMatrix bop_loc(NUMSTR_SOH8,NUMDOF_SOH8);
    for (int inode = 0; inode < NUMNOD_SOH8; ++inode) {
      for (int dim = 0; dim < NUMDIM_SOH8; ++dim) {
        // B_loc_rr = N_r.X_r
        bop_loc(0,inode*3+dim) = derivs[gp](0,inode) * jac_cur(0,dim);
        // B_loc_ss = N_s.X_s
        bop_loc(1,inode*3+dim) = derivs[gp](1,inode) * jac_cur(1,dim);
        // B_loc_tt = interpolation along (r x s) of ANS B_loc_tt
        //          = (1-r)(1-s)/4 * B_ans(SP E) + (1+r)(1-s)/4 * B_ans(SP F)
        //           +(1+r)(1+s)/4 * B_ans(SP G) + (1-r)(1+s)/4 * B_ans(SP H)
        bop_loc(2,inode*3+dim) = 0.25*(1-r[gp])*(1-s[gp]) * B_ans_loc(0+4*num_ans,inode*3+dim)
                                +0.25*(1+r[gp])*(1-s[gp]) * B_ans_loc(0+5*num_ans,inode*3+dim)
                                +0.25*(1+r[gp])*(1+s[gp]) * B_ans_loc(0+6*num_ans,inode*3+dim)
                                +0.25*(1-r[gp])*(1+s[gp]) * B_ans_loc(0+7*num_ans,inode*3+dim);
        // B_loc_rs = N_r.X_s + N_s.X_r
        bop_loc(3,inode*3+dim) = derivs[gp](0,inode) * jac_cur(1,dim)
                                +derivs[gp](1,inode) * jac_cur(0,dim);
        // B_loc_st = interpolation along r of ANS B_loc_st
        //          = (1+r)/2 * B_ans(SP B) + (1-r)/2 * B_ans(SP D)
        bop_loc(4,inode*3+dim) = 0.5*(1.0+r[gp]) * B_ans_loc(1+1*num_ans,inode*3+dim)
                                +0.5*(1.0-r[gp]) * B_ans_loc(1+3*num_ans,inode*3+dim);
        // B_loc_rt = interpolation along s of ANS B_loc_rt
        //          = (1-s)/2 * B_ans(SP A) + (1+s)/2 * B_ans(SP C)
        bop_loc(5,inode*3+dim) = 0.5*(1.0-s[gp]) * B_ans_loc(2+0*num_ans,inode*3+dim)
                                +0.5*(1.0+s[gp]) * B_ans_loc(2+2*num_ans,inode*3+dim);
      }
    }

    // transformation from local (parameter) element space to global(material) space
    // with famous 'T'-matrix already used for EAS but now evaluated at each gp
    Epetra_SerialDenseMatrix TinvT(NUMSTR_SOH8,NUMSTR_SOH8);
    sosh8_evaluateT(jac,TinvT);
    LINALG::SerialDenseMatrix bop(NUMSTR_SOH8,NUMDOF_SOH8);
    bop.Multiply('N','N',1.0,TinvT,bop_loc,0.0);

    // local GL strain vector lstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    // but with modified ANS strains E33, E23 and E13
    LINALG::SerialDenseVector lstrain(NUMSTR_SOH8);
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

    // ANS modification of strains ************************************** ANS
    double dydt_A = 0.0; double dYdt_A = 0.0;
    double dxdt_B = 0.0; double dXdt_B = 0.0;
    double dydt_C = 0.0; double dYdt_C = 0.0;
    double dxdt_D = 0.0; double dXdt_D = 0.0;
    double dzdt_E = 0.0; double dZdt_E = 0.0;
    double dzdt_F = 0.0; double dZdt_F = 0.0;
    double dzdt_G = 0.0; double dZdt_G = 0.0;
    double dzdt_H = 0.0; double dZdt_H = 0.0;

    // vector product of rows of jacobians at corresponding sampling point    cout << jac_cur_sps;
    for (int dim = 0; dim < NUMDIM_SOH8; ++dim) {
      dydt_A += jac_cur_sps(0+0*NUMDIM_SOH8,dim) * jac_cur_sps(2+0*NUMDIM_SOH8,dim);
      dYdt_A += jac_sps(0+0*NUMDIM_SOH8,dim)     * jac_sps(2+0*NUMDIM_SOH8,dim);
      dxdt_B += jac_cur_sps(1+1*NUMDIM_SOH8,dim) * jac_cur_sps(2+1*NUMDIM_SOH8,dim);
      dXdt_B += jac_sps(1+1*NUMDIM_SOH8,dim)     * jac_sps(2+1*NUMDIM_SOH8,dim);
      dydt_C += jac_cur_sps(0+2*NUMDIM_SOH8,dim) * jac_cur_sps(2+2*NUMDIM_SOH8,dim);
      dYdt_C += jac_sps(0+2*NUMDIM_SOH8,dim)     * jac_sps(2+2*NUMDIM_SOH8,dim);
      dxdt_D += jac_cur_sps(1+3*NUMDIM_SOH8,dim) * jac_cur_sps(2+3*NUMDIM_SOH8,dim);
      dXdt_D += jac_sps(1+3*NUMDIM_SOH8,dim)     * jac_sps(2+3*NUMDIM_SOH8,dim);

      dzdt_E += jac_cur_sps(2+4*NUMDIM_SOH8,dim) * jac_cur_sps(2+4*NUMDIM_SOH8,dim);
      dZdt_E += jac_sps(2+4*NUMDIM_SOH8,dim)     * jac_sps(2+4*NUMDIM_SOH8,dim);
      dzdt_F += jac_cur_sps(2+5*NUMDIM_SOH8,dim) * jac_cur_sps(2+5*NUMDIM_SOH8,dim);
      dZdt_F += jac_sps(2+5*NUMDIM_SOH8,dim)     * jac_sps(2+5*NUMDIM_SOH8,dim);
      dzdt_G += jac_cur_sps(2+6*NUMDIM_SOH8,dim) * jac_cur_sps(2+6*NUMDIM_SOH8,dim);
      dZdt_G += jac_sps(2+6*NUMDIM_SOH8,dim)     * jac_sps(2+6*NUMDIM_SOH8,dim);
      dzdt_H += jac_cur_sps(2+7*NUMDIM_SOH8,dim) * jac_cur_sps(2+7*NUMDIM_SOH8,dim);
      dZdt_H += jac_sps(2+7*NUMDIM_SOH8,dim)     * jac_sps(2+7*NUMDIM_SOH8,dim);
    }
    // E33: remedy of curvature thickness locking
    // Ett = 0.5* ( (1-r)(1-s)/4 * Ett(SP E) + ... + (1-r)(1+s)/4 * Ett(SP H) )
    lstrain(2) = 0.5 * (
       0.25*(1-r[gp])*(1-s[gp]) * (dzdt_E - dZdt_E)
      +0.25*(1+r[gp])*(1-s[gp]) * (dzdt_F - dZdt_F)
      +0.25*(1+r[gp])*(1+s[gp]) * (dzdt_G - dZdt_G)
      +0.25*(1-r[gp])*(1+s[gp]) * (dzdt_H - dZdt_H));
    // E23: remedy of transverse shear locking
    // Est = (1+r)/2 * Est(SP B) + (1-r)/2 * Est(SP D)
    lstrain(4) = 0.5*(1+r[gp]) * (dxdt_B - dXdt_B) + 0.5*(1-r[gp]) * (dxdt_D - dXdt_D);
    // E13: remedy of transverse shear locking
    // Ert = (1-s)/2 * Ert(SP A) + (1+s)/2 * Ert(SP C)
    lstrain(5) = 0.5*(1-s[gp]) * (dydt_A - dYdt_A) + 0.5*(1+s[gp]) * (dydt_C - dYdt_C);
    // ANS modification of strains ************************************** ANS

    // transformation of local glstrains 'back' to global(material) space
    LINALG::SerialDenseVector glstrain(NUMSTR_SOH8);
    glstrain.Multiply('N','N',1.0,TinvT,lstrain,0.0);

    // EAS technology: "enhance the strains"  ----------------------------- EAS
    if (eastype_ != soh8_easnone) {
      M.LightShape(NUMSTR_SOH8,neas_);
      // map local M to global, also enhancement is refered to element origin
      // M = detJ0/detJ T0^{-T} . M
      //Epetra_SerialDenseMatrix Mtemp(M); // temp M for Matrix-Matrix-Product
      M.Multiply('N','N',detJ0/detJ,T0invT,M_GP->at(gp),0.0);
      // add enhanced strains = M . alpha to GL strains to "unlock" element
      glstrain.Multiply('N','N',1.0,M,(*alpha),1.0);
    } // ------------------------------------------------------------------ EAS

    // return gp strains (only in case of stress/strain output)
    if (elestrain != NULL){
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
    Epetra_SerialDenseMatrix cmat(NUMSTR_SOH8,NUMSTR_SOH8);
    Epetra_SerialDenseVector stress(NUMSTR_SOH8);
    double density;
    // Caution!! the defgrd can not be modified with ANS to remedy locking
    // therefore it is empty and passed only for compatibility reasons
    Epetra_SerialDenseMatrix defgrd; // Caution!! empty!!
    soh8_mat_sel(&stress,&cmat,&density,&glstrain,&defgrd,gp,params);
    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    // return gp stresses
    if (elestress != NULL){
      if (!cauchy) {                 // return 2nd Piola-Kirchhoff stresses
        for (int i = 0; i < NUMSTR_SOH8; ++i) {
          (*elestress)(gp,i) = stress(i);
        }
      }
      else {                         // return Cauchy stresses
        sosh8_Cauchy(elestress,gp,derivs[gp],xrefe,xcurr,glstrain,stress);
      }
    }

    if (force != NULL && stiffmatrix != NULL) {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      (*force).Multiply('T', 'N', detJ * gpweights[gp], bop, stress, 1.0);
      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      LINALG::SerialDenseMatrix cb(NUMSTR_SOH8, NUMDOF_SOH8);
      cb.Multiply('N', 'N', 1.0, cmat, bop, 0.0); // temporary C . B
      (*stiffmatrix).Multiply('T', 'N', detJ * gpweights[gp], bop, cb, 1.0);

      // intergrate `geometric' stiffness matrix and add to keu *****************
      // here also the ANS interpolation comes into play
      for (int inod=0; inod<NUMNOD_SOH8; ++inod) {
        for (int jnod=0; jnod<NUMNOD_SOH8; ++jnod) {
          Epetra_SerialDenseVector G_ij(NUMSTR_SOH8);
          G_ij(0) = derivs[gp](0, inod) * derivs[gp](0, jnod); // rr-dir
          G_ij(1) = derivs[gp](1, inod) * derivs[gp](1, jnod); // ss-dir
          G_ij(3) = derivs[gp](0, inod) * derivs[gp](1, jnod)
                  + derivs[gp](1, inod) * derivs[gp](0, jnod); // rs-dir
          // ANS modification in tt-dir
          G_ij(2) = 0.25*(1-r[gp])*(1-s[gp]) * (*deriv_sp)(2+4*NUMDIM_SOH8,inod) * (*deriv_sp)(2+4*NUMDIM_SOH8,jnod)
                   +0.25*(1+r[gp])*(1-s[gp]) * (*deriv_sp)(2+5*NUMDIM_SOH8,inod) * (*deriv_sp)(2+5*NUMDIM_SOH8,jnod)
                   +0.25*(1+r[gp])*(1+s[gp]) * (*deriv_sp)(2+6*NUMDIM_SOH8,inod) * (*deriv_sp)(2+6*NUMDIM_SOH8,jnod)
                   +0.25*(1-r[gp])*(1+s[gp]) * (*deriv_sp)(2+7*NUMDIM_SOH8,inod) * (*deriv_sp)(2+7*NUMDIM_SOH8,jnod);
          // ANS modification in st-dir
          G_ij(4) = 0.5*((1+r[gp]) * ((*deriv_sp)(1+1*NUMDIM_SOH8,inod) * (*deriv_sp)(2+1*NUMDIM_SOH8,jnod)
                                     +(*deriv_sp)(2+1*NUMDIM_SOH8,inod) * (*deriv_sp)(1+1*NUMDIM_SOH8,jnod))
                        +(1-r[gp]) * ((*deriv_sp)(1+3*NUMDIM_SOH8,inod) * (*deriv_sp)(2+3*NUMDIM_SOH8,jnod)
                                     +(*deriv_sp)(2+3*NUMDIM_SOH8,inod) * (*deriv_sp)(1+3*NUMDIM_SOH8,jnod)));
          // ANS modification in rt-dir
          G_ij(5) = 0.5*((1-s[gp]) * ((*deriv_sp)(0+0*NUMDIM_SOH8,inod) * (*deriv_sp)(2+0*NUMDIM_SOH8,jnod)
                                     +(*deriv_sp)(2+0*NUMDIM_SOH8,inod) * (*deriv_sp)(0+0*NUMDIM_SOH8,jnod))
                        +(1+s[gp]) * ((*deriv_sp)(0+2*NUMDIM_SOH8,inod) * (*deriv_sp)(2+2*NUMDIM_SOH8,jnod)
                                     +(*deriv_sp)(2+2*NUMDIM_SOH8,inod) * (*deriv_sp)(0+2*NUMDIM_SOH8,jnod)));
          // transformation of local(parameter) space 'back' to global(material) space
          Epetra_SerialDenseVector G_ij_glob(NUMSTR_SOH8);
          G_ij_glob.Multiply('N', 'N', 1.0, TinvT, G_ij, 0.0);

          // Scalar Gij results from product of G_ij with stress, scaled with detJ*weights
          Epetra_SerialDenseVector Gij(1); // this is a scalar
          Gij.Multiply('T', 'N', detJ * gpweights[gp], stress, G_ij_glob, 0.0);

          // add "geometric part" Gij times detJ*weights to stiffness matrix
          (*stiffmatrix)(NUMDIM_SOH8*inod+0, NUMDIM_SOH8*jnod+0) += Gij(0);
          (*stiffmatrix)(NUMDIM_SOH8*inod+1, NUMDIM_SOH8*jnod+1) += Gij(0);
          (*stiffmatrix)(NUMDIM_SOH8*inod+2, NUMDIM_SOH8*jnod+2) += Gij(0);
        }
      } // end of intergrate `geometric' stiffness ******************************

      // EAS technology: integrate matrices --------------------------------- EAS
      if (eastype_ != soh8_easnone) {
        double integrationfactor = detJ * gpweights[gp];
        // integrate Kaa: Kaa += (M^T . cmat . M) * detJ * w(gp)
        LINALG::SerialDenseMatrix cM(NUMSTR_SOH8, neas_); // temporary c . M
        cM.Multiply('N', 'N', 1.0, cmat, M, 0.0);
        Kaa.Multiply('T', 'N', integrationfactor, M, cM, 1.0);

        // integrate Kda: Kda += (M^T . cmat . B) * detJ * w(gp)
        Kda.Multiply('T', 'N', integrationfactor, M, cb, 1.0);

        // integrate feas: feas += (M^T . sigma) * detJ *wp(gp)
        feas.Multiply('T', 'N', integrationfactor, M, stress, 1.0);
      } // ------------------------------------------------------------------ EAS
    }

    if (massmatrix != NULL){ // evaluate mass matrix +++++++++++++++++++++++++
      // integrate concistent mass matrix
      for (int inod=0; inod<NUMNOD_SOH8; ++inod) {
        for (int jnod=0; jnod<NUMNOD_SOH8; ++jnod) {
          double massfactor = shapefcts[gp](inod) * shapefcts[gp](jnod)
                              * density * detJ * gpweights[gp];     // intermediate factor
          (*massmatrix)(NUMDIM_SOH8*inod+0,NUMDIM_SOH8*jnod+0) += massfactor;
          (*massmatrix)(NUMDIM_SOH8*inod+1,NUMDIM_SOH8*jnod+1) += massfactor;
          (*massmatrix)(NUMDIM_SOH8*inod+2,NUMDIM_SOH8*jnod+2) += massfactor;
        }
      }
    } // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++
   /* =========================================================================*/
  }/* ==================================================== end of Loop over GP */
   /* =========================================================================*/

  if (force != NULL && stiffmatrix != NULL) {
    // EAS technology: ------------------------------------------------------ EAS
    // subtract EAS matrices from disp-based Kdd to "soften" element
    if (eastype_ != soh8_easnone) {
      // we need the inverse of Kaa
      Epetra_SerialDenseSolver solve_for_inverseKaa;
      solve_for_inverseKaa.SetMatrix(Kaa);
      solve_for_inverseKaa.Invert();

      LINALG::SerialDenseMatrix KdaTKaa(NUMDOF_SOH8, neas_); // temporary Kda^T.Kaa^{-1}
      KdaTKaa.Multiply('T', 'N', 1.0, Kda, Kaa, 0.0);

      // EAS-stiffness matrix is: Kdd - Kda^T . Kaa^-1 . Kda
      (*stiffmatrix).Multiply('N', 'N', -1.0, KdaTKaa, Kda, 1.0);

      // EAS-internal force is: fint - Kda^T . Kaa^-1 . feas
      (*force).Multiply('N', 'N', -1.0, KdaTKaa, feas, 1.0);

      // store current EAS data in history
      for (int i=0; i<neas_; ++i) {
        for (int j=0; j<neas_; ++j) (*oldKaainv)(i,j) = Kaa(i,j);
        for (int j=0; j<NUMDOF_SOH8; ++j) (*oldKda)(i, j) = Kda(i,j);
        (*oldfeas)(i, 0) = feas(i);
      }
    } // -------------------------------------------------------------------- EAS
  }

  return;
} // DRT::ELEMENTS::So_sh8::sosh8_nlnstiffmass


/*----------------------------------------------------------------------*
 |  Evaluate Hex8 Shape fcts at all 8 Gauss Points             maf 05/08|
 *----------------------------------------------------------------------*/
const vector<Epetra_SerialDenseVector> DRT::ELEMENTS::So_sh8::sosh8_shapefcts()
{
  vector<Epetra_SerialDenseVector> shapefcts(NUMGPT_SOH8);
  // (r,s,t) gp-locations of fully integrated linear 8-node Hex
  const double gploc    = 1.0/sqrt(3.0);    // gp sampling point value for linear fct
  const double r[NUMGPT_SOH8] = {-gploc, gploc, gploc,-gploc,-gploc, gploc, gploc,-gploc};
  const double s[NUMGPT_SOH8] = {-gploc,-gploc, gploc, gploc,-gploc,-gploc, gploc, gploc};
  const double t[NUMGPT_SOH8] = {-gploc,-gploc,-gploc,-gploc, gploc, gploc, gploc, gploc};
  // fill up nodal f at each gp
  for (int i=0; i<NUMGPT_SOH8; ++i) {
    shapefcts[i].Size(NUMNOD_SOH8);
    (shapefcts[i])(0) = (1.0-r[i])*(1.0-s[i])*(1.0-t[i])*0.125;
    (shapefcts[i])(1) = (1.0+r[i])*(1.0-s[i])*(1.0-t[i])*0.125;
    (shapefcts[i])(2) = (1.0+r[i])*(1.0+s[i])*(1.0-t[i])*0.125;
    (shapefcts[i])(3) = (1.0-r[i])*(1.0+s[i])*(1.0-t[i])*0.125;
    (shapefcts[i])(4) = (1.0-r[i])*(1.0-s[i])*(1.0+t[i])*0.125;
    (shapefcts[i])(5) = (1.0+r[i])*(1.0-s[i])*(1.0+t[i])*0.125;
    (shapefcts[i])(6) = (1.0+r[i])*(1.0+s[i])*(1.0+t[i])*0.125;
    (shapefcts[i])(7) = (1.0-r[i])*(1.0+s[i])*(1.0+t[i])*0.125;
  }
  return shapefcts;
}


/*----------------------------------------------------------------------*
 |  Evaluate Hex8 Shape fct derivs at all 8 Gauss Points       maf 05/08|
 *----------------------------------------------------------------------*/
const vector<Epetra_SerialDenseMatrix> DRT::ELEMENTS::So_sh8::sosh8_derivs()
{
  vector<Epetra_SerialDenseMatrix> derivs(NUMGPT_SOH8);
  // (r,s,t) gp-locations of fully integrated linear 8-node Hex
  const double gploc    = 1.0/sqrt(3.0);    // gp sampling point value for linear fct
  const double r[NUMGPT_SOH8] = {-gploc, gploc, gploc,-gploc,-gploc, gploc, gploc,-gploc};
  const double s[NUMGPT_SOH8] = {-gploc,-gploc, gploc, gploc,-gploc,-gploc, gploc, gploc};
  const double t[NUMGPT_SOH8] = {-gploc,-gploc,-gploc,-gploc, gploc, gploc, gploc, gploc};
  // fill up df w.r.t. rst directions (NUMDIM) at each gp
  for (int i=0; i<NUMGPT_SOH8; ++i) {
    (derivs[i]).Shape(NUMDIM_SOH8,NUMNOD_SOH8);
    // df wrt to r for each node(0..7) at each gp [i]
    (derivs[i])(0,0) = -(1.0-s[i])*(1.0-t[i])*0.125;
    (derivs[i])(0,1) =  (1.0-s[i])*(1.0-t[i])*0.125;
    (derivs[i])(0,2) =  (1.0+s[i])*(1.0-t[i])*0.125;
    (derivs[i])(0,3) = -(1.0+s[i])*(1.0-t[i])*0.125;
    (derivs[i])(0,4) = -(1.0-s[i])*(1.0+t[i])*0.125;
    (derivs[i])(0,5) =  (1.0-s[i])*(1.0+t[i])*0.125;
    (derivs[i])(0,6) =  (1.0+s[i])*(1.0+t[i])*0.125;
    (derivs[i])(0,7) = -(1.0+s[i])*(1.0+t[i])*0.125;

    // df wrt to s for each node(0..7) at each gp [i]
    (derivs[i])(1,0) = -(1.0-r[i])*(1.0-t[i])*0.125;
    (derivs[i])(1,1) = -(1.0+r[i])*(1.0-t[i])*0.125;
    (derivs[i])(1,2) =  (1.0+r[i])*(1.0-t[i])*0.125;
    (derivs[i])(1,3) =  (1.0-r[i])*(1.0-t[i])*0.125;
    (derivs[i])(1,4) = -(1.0-r[i])*(1.0+t[i])*0.125;
    (derivs[i])(1,5) = -(1.0+r[i])*(1.0+t[i])*0.125;
    (derivs[i])(1,6) =  (1.0+r[i])*(1.0+t[i])*0.125;
    (derivs[i])(1,7) =  (1.0-r[i])*(1.0+t[i])*0.125;

    // df wrt to t for each node(0..7) at each gp [i]
    (derivs[i])(2,0) = -(1.0-r[i])*(1.0-s[i])*0.125;
    (derivs[i])(2,1) = -(1.0+r[i])*(1.0-s[i])*0.125;
    (derivs[i])(2,2) = -(1.0+r[i])*(1.0+s[i])*0.125;
    (derivs[i])(2,3) = -(1.0-r[i])*(1.0+s[i])*0.125;
    (derivs[i])(2,4) =  (1.0-r[i])*(1.0-s[i])*0.125;
    (derivs[i])(2,5) =  (1.0+r[i])*(1.0-s[i])*0.125;
    (derivs[i])(2,6) =  (1.0+r[i])*(1.0+s[i])*0.125;
    (derivs[i])(2,7) =  (1.0-r[i])*(1.0+s[i])*0.125;
  }
  return derivs;
}

/*----------------------------------------------------------------------*
 |  Evaluate Hex8 Weights at all 8 Gauss Points                maf 05/08|
 *----------------------------------------------------------------------*/
const vector<double> DRT::ELEMENTS::So_sh8::sosh8_weights()
{
  vector<double> weights(NUMGPT_SOH8);
  for (int i = 0; i < NUMGPT_SOH8; ++i) {
    weights[i] = 1.0;
  }
  return weights;
}


/*----------------------------------------------------------------------*
 |  setup of constant ANS data (private)                       maf 05/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_sh8::sosh8_anssetup(
          const int numsp,              // number of sampling points
          const int numans,             // number of ans strains
          const Epetra_SerialDenseMatrix& xrefe, // material element coords
          const Epetra_SerialDenseMatrix& xcurr, // current element coords
          Epetra_SerialDenseMatrix** deriv_sp,   // derivs eval. at all sampling points
          Epetra_SerialDenseMatrix& jac_sps,     // jac at all sampling points
          Epetra_SerialDenseMatrix& jac_cur_sps, // current jac at all sampling points
          Epetra_SerialDenseMatrix& B_ans_loc) // modified B
{
  // static matrix object of derivs at sampling points, kept in memory
  static Epetra_SerialDenseMatrix df_sp(NUMDIM_SOH8*numsp,NUMNOD_SOH8);
  static bool dfsp_eval;                      // flag for re-evaluate everything

  if (dfsp_eval!=0){             // if true f,df already evaluated
    *deriv_sp = &df_sp;         // return adress of static object to target of pointer
  } else {
  /*====================================================================*/
  /* 8-node hexhedra Solid-Shell node topology
   * and location of sampling points A to H                             */
  /*--------------------------------------------------------------------*/
  /*                      t
   *                      |
   *             4========|================7
   *          // |        |              //||
   *        //   |        |            //  ||
   *      //     |        |   D      //    ||
   *     5=======E=================6       H
   *    ||       |        |        ||      ||
   *    ||   A   |        o--------||-- C -------s
   *    ||       |       /         ||      ||
   *    F        0----- B ---------G ------3
   *    ||     //     /            ||    //
   *    ||   //     /              ||  //
   *    || //     r                ||//
   *     1=========================2
   *
   */
  /*====================================================================*/
    // (r,s,t) gp-locations of sampling points A,B,C,D,E,F,G,H
    // numsp = 8 here set explicitly to allow direct initializing
    double r[8] = { 0.0, 1.0, 0.0,-1.0,-1.0, 1.0, 1.0,-1.0};
    double s[8] = {-1.0, 0.0, 1.0, 0.0,-1.0,-1.0, 1.0, 1.0};
    double t[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // fill up df_sp w.r.t. rst directions (NUMDIM) at each sp
    for (int i=0; i<numsp; ++i) {
        // df wrt to r "+0" for each node(0..7) at each sp [i]
        df_sp(NUMDIM_SOH8*i+0,0) = -(1.0-s[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+0,1) =  (1.0-s[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+0,2) =  (1.0+s[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+0,3) = -(1.0+s[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+0,4) = -(1.0-s[i])*(1.0+t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+0,5) =  (1.0-s[i])*(1.0+t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+0,6) =  (1.0+s[i])*(1.0+t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+0,7) = -(1.0+s[i])*(1.0+t[i])*0.125;

        // df wrt to s "+1" for each node(0..7) at each sp [i]
        df_sp(NUMDIM_SOH8*i+1,0) = -(1.0-r[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+1,1) = -(1.0+r[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+1,2) =  (1.0+r[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+1,3) =  (1.0-r[i])*(1.0-t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+1,4) = -(1.0-r[i])*(1.0+t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+1,5) = -(1.0+r[i])*(1.0+t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+1,6) =  (1.0+r[i])*(1.0+t[i])*0.125;
        df_sp(NUMDIM_SOH8*i+1,7) =  (1.0-r[i])*(1.0+t[i])*0.125;

        // df wrt to t "+2" for each node(0..7) at each sp [i]
        df_sp(NUMDIM_SOH8*i+2,0) = -(1.0-r[i])*(1.0-s[i])*0.125;
        df_sp(NUMDIM_SOH8*i+2,1) = -(1.0+r[i])*(1.0-s[i])*0.125;
        df_sp(NUMDIM_SOH8*i+2,2) = -(1.0+r[i])*(1.0+s[i])*0.125;
        df_sp(NUMDIM_SOH8*i+2,3) = -(1.0-r[i])*(1.0+s[i])*0.125;
        df_sp(NUMDIM_SOH8*i+2,4) =  (1.0-r[i])*(1.0-s[i])*0.125;
        df_sp(NUMDIM_SOH8*i+2,5) =  (1.0+r[i])*(1.0-s[i])*0.125;
        df_sp(NUMDIM_SOH8*i+2,6) =  (1.0+r[i])*(1.0+s[i])*0.125;
        df_sp(NUMDIM_SOH8*i+2,7) =  (1.0-r[i])*(1.0+s[i])*0.125;
    }

    // return adresses of just evaluated matrices
    *deriv_sp = &df_sp;         // return adress of static object to target of pointer
    dfsp_eval = 1;               // now all arrays are filled statically
  }

  // compute Jacobian matrix at all sampling points
  jac_sps.Multiply('N','N',1.0,df_sp,xrefe,0.0);

  // compute CURRENT Jacobian matrix at all sampling points
  jac_cur_sps.Multiply('N','N',1.0,df_sp,xcurr,0.0);

  /*
  ** Compute modified B-operator in local(parametric) space,
  ** evaluated at all sampling points
  */
  // loop over each sampling point
  for (int sp = 0; sp < numsp; ++sp) {
    // get submatrix of deriv_sp at actual sp
    Epetra_SerialDenseMatrix deriv_asp(NUMDIM_SOH8,numsp);
    for (int m=0; m<NUMDIM_SOH8; ++m) {
      for (int n=0; n<numsp; ++n) {
        deriv_asp(m,n)=df_sp(NUMDIM_SOH8*sp+m,n);
      }
    }
    /* compute the CURRENT Jacobian matrix at the sampling point:
    **         [ xcurr_,r  ycurr_,r  zcurr_,r ]
    **  Jcur = [ xcurr_,s  ycurr_,s  zcurr_,s ]
    **         [ xcurr_,t  ycurr_,t  zcurr_,t ]
    ** Used to transform the global displacements into parametric space
    */
    Epetra_SerialDenseMatrix jac_cur(NUMDIM_SOH8,NUMDIM_SOH8);
    jac_cur.Multiply('N','N',1.0,deriv_asp,xcurr,1.0);

    // fill up B-operator
    for (int inode = 0; inode < NUMNOD_SOH8; ++inode) {
      for (int dim = 0; dim < NUMDIM_SOH8; ++dim) {
        // modify B_loc_tt = N_t.X_t
        B_ans_loc(sp*numans+0,inode*3+dim) = deriv_asp(2,inode)*jac_cur(2,dim);
        // modify B_loc_st = N_s.X_t + N_t.X_s
        B_ans_loc(sp*numans+1,inode*3+dim) = deriv_asp(1,inode)*jac_cur(2,dim)
                                            +deriv_asp(2,inode)*jac_cur(1,dim);
        // modify B_loc_rt = N_r.X_t + N_t.X_r
        B_ans_loc(sp*numans+2,inode*3+dim) = deriv_asp(0,inode)*jac_cur(2,dim)
                                            +deriv_asp(2,inode)*jac_cur(0,dim);
      }
    }
  }


  return;
}

/*----------------------------------------------------------------------*
 |  evaluate 'T'-transformation matrix )                       maf 05/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_sh8::sosh8_evaluateT(const Epetra_SerialDenseMatrix& jac,
                                            Epetra_SerialDenseMatrix& TinvT)
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
  Epetra_SerialDenseSolver solve_for_inverseT;
  solve_for_inverseT.SetMatrix(TinvT);
  int err2 = solve_for_inverseT.Factor();
  int err = solve_for_inverseT.Invert();
  if ((err != 0) && (err2!=0)) dserror("Inversion of Tinv (Jacobian) failed");
  return;
}

/*----------------------------------------------------------------------*
 |  return Cauchy stress at gp                                 maf 06/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_sh8::sosh8_Cauchy(Epetra_SerialDenseMatrix* elestress,
                                         const int gp,
                                         const Epetra_SerialDenseMatrix& deriv,
                                         const Epetra_SerialDenseMatrix& xrefe,
                                         const Epetra_SerialDenseMatrix& xcurr,
                                         const Epetra_SerialDenseVector& glstrain,
                                         const Epetra_SerialDenseVector& stress)
{
  // with ANS you do NOT have the correct (locking-free) F, so we
  // compute it here JUST for mapping of correct (locking-free) stresses
  LINALG::SerialDenseMatrix invJ(NUMDIM_SOH8,NUMDIM_SOH8);
  invJ.Multiply('N','N',1.0,deriv,xrefe,0.0);
  LINALG::NonsymInverse3x3(invJ);
  LINALG::SerialDenseMatrix N_XYZ(NUMDIM_SOH8,NUMNOD_SOH8);
  LINALG::SerialDenseMatrix temp(NUMDIM_SOH8,NUMDIM_SOH8);
  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  // by N_XYZ = J^-1 * N_rst
  N_XYZ.Multiply('N','N',1.0,invJ,deriv,0.0);
  // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
  LINALG::SerialDenseMatrix defgrd(NUMDIM_SOH8,NUMDIM_SOH8);
  defgrd.Multiply('T','T',1.0,xcurr,N_XYZ,0.0);

# if consistent_F
  //double disp1 = defgrd.NormOne();
  //double dispinf = defgrd.NormInf();

  /* to get the consistent (locking-free) F^mod, we need two spectral
   * compositions. First, find R (rotation tensor) from F=RU,
   * then from E^mod = 1/2((U^mod)^2 - 1) find U^mod,
   * and finally F^mod = RU^mod */

  // polar decomposition of displacement based F
  LINALG::SerialDenseMatrix u(NUMDIM_SOH8,NUMDIM_SOH8);
  LINALG::SerialDenseMatrix s(NUMDIM_SOH8,NUMDIM_SOH8);
  LINALG::SerialDenseMatrix v(NUMDIM_SOH8,NUMDIM_SOH8);
  SVD(defgrd,u,s,v); // Singular Value Decomposition
  LINALG::SerialDenseMatrix rot(NUMDIM_SOH8,NUMDIM_SOH8);
  rot.Multiply('N','N',1.0,u,v,0.0);
  //temp.Multiply('N','N',1.0,v,s,0.0);
  //LINALG::SerialDenseMatrix stretch_disp(NUMDIM_SOH8,NUMDIM_SOH8);
  //stretch_disp.Multiply('N','T',1.0,temp,v,0.0);
  //defgrd.Multiply('N','N',1.0,rot,stretch_disp,0.0);
  //cout << defgrd;

  // get modified squared stretch (U^mod)^2 from glstrain
  LINALG::SerialDenseMatrix Usq_mod(NUMDIM_SOH8,NUMDIM_SOH8);
  for (int i = 0; i < NUMDIM_SOH8; ++i) Usq_mod(i,i) = 2.0 * glstrain(i) + 1.0;
  // off-diagonal terms are already twice in the Voigt-GLstrain-vector
  Usq_mod(0,1) =  glstrain(3);  Usq_mod(1,0) =  glstrain(3);
  Usq_mod(1,2) =  glstrain(4);  Usq_mod(2,1) =  glstrain(4);
  Usq_mod(0,2) =  glstrain(5);  Usq_mod(2,0) =  glstrain(5);
  // polar decomposition of (U^mod)^2
  SVD(Usq_mod,u,s,v); // Singular Value Decomposition
  LINALG::SerialDenseMatrix U_mod(NUMDIM_SOH8,NUMDIM_SOH8);
  for (int i = 0; i < NUMDIM_SOH8; ++i) s(i,i) = sqrt(s(i,i));
  temp.Multiply('N','N',1.0,u,s,0.0);
  U_mod.Multiply('N','N',1.0,temp,v,0.0);

  // F^mod = RU^mod
  defgrd.Multiply('N','N',1.0,rot,U_mod,0.0);

  /*
  double mod1 = defgrd.NormOne();
  double modinf = defgrd.NormInf();
  if(((mod1-disp1)/mod1 > 0.03) || ((modinf-dispinf)/modinf > 0.03)){
    cout << "difference in F! mod1= " << mod1 << " disp1= " << disp1 << " modinf= " << modinf << " dispinf= " << dispinf << endl;
    cout << "Fmod" << endl << defgrd;
  }
  */
#endif

  double detF = defgrd(0,0)*defgrd(1,1)*defgrd(2,2) +
                defgrd(0,1)*defgrd(1,2)*defgrd(2,0) +
                defgrd(0,2)*defgrd(1,0)*defgrd(2,1) -
                defgrd(0,2)*defgrd(1,1)*defgrd(2,0) -
                defgrd(0,0)*defgrd(1,2)*defgrd(2,1) -
                defgrd(0,1)*defgrd(1,0)*defgrd(2,2);

  LINALG::SerialDenseMatrix pkstress(NUMDIM_SOH8,NUMDIM_SOH8);
  pkstress(0,0) = stress(0);
  pkstress(0,1) = stress(3);
  pkstress(0,2) = stress(5);
  pkstress(1,0) = pkstress(0,1);
  pkstress(1,1) = stress(1);
  pkstress(1,2) = stress(4);
  pkstress(2,0) = pkstress(0,2);
  pkstress(2,1) = pkstress(1,2);
  pkstress(2,2) = stress(2);

  LINALG::SerialDenseMatrix cauchystress(NUMDIM_SOH8,NUMDIM_SOH8);
  temp.Multiply('N','N',1.0/detF,defgrd,pkstress,0.);
  cauchystress.Multiply('N','T',1.0,temp,defgrd,0.);

  (*elestress)(gp,0) = cauchystress(0,0);
  (*elestress)(gp,1) = cauchystress(1,1);
  (*elestress)(gp,2) = cauchystress(2,2);
  (*elestress)(gp,3) = cauchystress(0,1);
  (*elestress)(gp,4) = cauchystress(1,2);
  (*elestress)(gp,5) = cauchystress(0,2);
}




/*----------------------------------------------------------------------*
 |  init the element (public)                                  maf 07/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Sosh8Register::Initialize(DRT::Discretization& dis)
{
  //sosh8_gmshplotdis(dis);

  int num_morphed_so_hex8 = 0;

  // Loop through all elements
  for (int i=0; i<dis.NumMyColElements(); ++i)
  {
    // get the actual element
    if (dis.lColElement(i)->Type() != DRT::Element::element_sosh8) continue;
    DRT::ELEMENTS::So_sh8* actele = dynamic_cast<DRT::ELEMENTS::So_sh8*>(dis.lColElement(i));
    if (!actele) dserror("cast to So_sh8* failed");

    if (!actele->nodes_rearranged_) {
      // check for automatic definition of thickness direction
      if (actele->thickdir_ == DRT::ELEMENTS::So_sh8::autoj) {
        actele->thickdir_ = actele->sosh8_findthickdir();
      }

      int new_nodeids[NUMNOD_SOH8];

      switch (actele->thickdir_) {
      case DRT::ELEMENTS::So_sh8::autor:
      case DRT::ELEMENTS::So_sh8::globx: {
        // resorting of nodes to arrive at local t-dir for global x-dir
        new_nodeids[0] = actele->NodeIds()[7];
        new_nodeids[1] = actele->NodeIds()[4];
        new_nodeids[2] = actele->NodeIds()[0];
        new_nodeids[3] = actele->NodeIds()[3];
        new_nodeids[4] = actele->NodeIds()[6];
        new_nodeids[5] = actele->NodeIds()[5];
        new_nodeids[6] = actele->NodeIds()[1];
        new_nodeids[7] = actele->NodeIds()[2];
//        actele->sosh8_gmshplotlabeledelement(actele->NodeIds());
//        actele->sosh8_gmshplotlabeledelement(new_nodeids);
        actele->SetNodeIds(NUMNOD_SOH8, new_nodeids);
        actele->nodes_rearranged_ = true;
        break;
      }
      case DRT::ELEMENTS::So_sh8::autos:
      case DRT::ELEMENTS::So_sh8::globy: {
        // resorting of nodes to arrive at local t-dir for global y-dir
        new_nodeids[0] = actele->NodeIds()[4];
        new_nodeids[1] = actele->NodeIds()[5];
        new_nodeids[2] = actele->NodeIds()[1];
        new_nodeids[3] = actele->NodeIds()[0];
        new_nodeids[4] = actele->NodeIds()[7];
        new_nodeids[5] = actele->NodeIds()[6];
        new_nodeids[6] = actele->NodeIds()[2];
        new_nodeids[7] = actele->NodeIds()[3];
        actele->SetNodeIds(NUMNOD_SOH8, new_nodeids);
        actele->nodes_rearranged_ = true;
        break;
      }
      case DRT::ELEMENTS::So_sh8::autot:
      case DRT::ELEMENTS::So_sh8::globz: {
        // no resorting necessary
        for (int node = 0; node < 8; ++node) {
          new_nodeids[node] = actele->NodeIds()[node];
        }
        actele->SetNodeIds(NUMNOD_SOH8, new_nodeids);
        actele->nodes_rearranged_ = true;
        break;
      }
      case DRT::ELEMENTS::So_sh8::undefined: {
        // here comes plan B: morph So_sh8 to So_hex8
        actele->SetType(DRT::Element::element_so_hex8);
        actele->soh8_reiniteas(DRT::ELEMENTS::So_hex8::soh8_easmild);
        actele->InitJacobianMapping();
        num_morphed_so_hex8++;
        break;
      }
      case DRT::ELEMENTS::So_sh8::none: break;
      default:
        dserror("no thickness direction for So_sh8");
      }
      //actele->sosh8_gmshplotlabeledelement(actele->NodeIds());
    }
  }

  if (num_morphed_so_hex8>0){
    cout << endl << num_morphed_so_hex8
    << " Sosh8-Elements have no clear 'thin' direction and have morphed to So_hex8 with eas_mild" << endl;
  }


  // fill complete again to reconstruct element-node pointers,
  // but without element init, etc.
  dis.FillComplete(false,false,false);

  // **************** debug printout ot gmesh **********************************
  //sosh8_gmshplotdis(dis);

  return 0;
}


#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_SOLID3
