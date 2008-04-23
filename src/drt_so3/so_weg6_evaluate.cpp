/*!----------------------------------------------------------------------
\file so_weg6_evaluate.cpp
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
#include "so_weg6.H"
#include "so_hex8.H"
#include "so_tet10.H"
#include "so_tet4.H"
#include "so_disp.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_exporter.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/linalg_serialdensematrix.H"
#include "../drt_lib/linalg_serialdensevector.H"
#include "../drt_lib/drt_utils_integration.H"
#include "../drt_lib/drt_utils_fem_shapefunctions.H"
#include "Epetra_SerialDenseSolver.h"

using namespace std; // cout etc.
using namespace LINALG; // our linear algebra

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              maf 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_weg6::Evaluate(ParameterList& params,
                                    DRT::Discretization&      discretization,
                                    vector<int>&              lm,
                                    Epetra_SerialDenseMatrix& elemat1,
                                    Epetra_SerialDenseMatrix& elemat2,
                                    Epetra_SerialDenseVector& elevec1,
                                    Epetra_SerialDenseVector& elevec2,
                                    Epetra_SerialDenseVector& elevec3)
{
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
  else if (action=="calc_struct_stress")        act = So_weg6::calc_struct_stress;
  else if (action=="calc_struct_eleload")       act = So_weg6::calc_struct_eleload;
  else if (action=="calc_struct_fsiload")       act = So_weg6::calc_struct_fsiload;
  else if (action=="calc_struct_update_istep")  act = So_weg6::calc_struct_update_istep;
  else if (action=="calc_struct_update_genalpha_imrlike")  act = So_weg6::calc_struct_update_genalpha_imrlike;
  else if (action=="postprocess_stress")        act = So_weg6::postprocess_stress;
  else dserror("Unknown type of action for So_weg6");

  const double time = params.get("total time",-1.0);

  // what should the element do
  switch(act) {
    // linear stiffness
    case calc_struct_linstiff: {
      // need current displacement and residual forces
      vector<double> mydisp(lm.size());
      for (int i=0; i<(int)mydisp.size(); ++i) mydisp[i] = 0.0;
      vector<double> myres(lm.size());
      for (int i=0; i<(int)myres.size(); ++i) myres[i] = 0.0;
      sow6_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,time);
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
      sow6_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,time);
    }
    break;

    // internal force vector only
    case calc_struct_internalforce:
      dserror("Case 'calc_struct_internalforce' not yet implemented");
    break;

    // linear stiffness and consistent mass matrix
    case calc_struct_linstiffmass:
      dserror("Case 'calc_struct_linstiffmass' not yet implemented");
    break;

    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case calc_struct_nlnstiffmass: {
      // need current displacement and residual forces
      RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
      RefCountPtr<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (disp==null || res==null) dserror("Cannot get state vectors 'displacement' and/or residual");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      sow6_nlnstiffmass(lm,mydisp,myres,&elemat1,&elemat2,&elevec1,NULL,NULL,time);
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
      Epetra_SerialDenseMatrix stress(NUMGPT_WEG6,NUMSTR_WEG6);
      Epetra_SerialDenseMatrix strain(NUMGPT_WEG6,NUMSTR_WEG6);
      bool cauchy = params.get<bool>("cauchy", false);
      sow6_nlnstiffmass(lm,mydisp,myres,NULL,NULL,NULL,&stress,&strain,time,cauchy);
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
        Epetra_SerialDenseMatrix nodalstresses(NUMNOD_WEG6,NUMSTR_WEG6);
        soweg6_expol(*gpstress,nodalstresses);

        // average nodal stresses/strains between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_WEG6);

        for (int i=0;i<NUMNOD_WEG6;++i){
          DRT::Node* node=Nodes()[i];
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
        if (lid!=-1) {
          for (int i = 0; i < NUMSTR_WEG6; ++i) {
            (*((*elestress)(i)))[lid] = 0.;
            for (int j = 0; j < NUMGPT_WEG6; ++j) {
              //(*((*elestress)(i)))[lid] += 0.125 * (*gpstress)(j,i);
              (*((*elestress)(i)))[lid] += 1.0/NUMGPT_WEG6 * (*gpstress)(j,i);
            }
          }
        }
      }
      else if (stresstype=="cxyz_ndxyz") {
        // extrapolate stresses/strains at Gauss points to nodes
        Epetra_SerialDenseMatrix nodalstresses(NUMNOD_WEG6,NUMSTR_WEG6);
        soweg6_expol(*gpstress,nodalstresses);

        // average nodal stresses/strains between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_WEG6);

        for (int i=0;i<NUMNOD_WEG6;++i){
          DRT::Node* node=Nodes()[i];
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
          for (int i = 0; i < NUMSTR_WEG6; ++i) {
            (*((*elestress)(i)))[lid] = 0.;
            for (int j = 0; j < NUMGPT_WEG6; ++j) {
              //(*((*elestress)(i)))[lid] += 0.125 * (*gpstress)(j,i);
              (*((*elestress)(i)))[lid] += 1.0/NUMGPT_WEG6 * (*gpstress)(j,i);
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
      ;// there is nothing to do here at the moment
    }
    break;

    case calc_struct_update_genalpha_imrlike: {
      ;// there is nothing to do here at the moment
    }
    break;

    default:
      dserror("Unknown type of action for Solid3");
  }
  return 0;
}



/*----------------------------------------------------------------------*
 |  Integrate a Volume Neumann boundary condition (public)     maf 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_weg6::EvaluateNeumann(ParameterList& params,
                                           DRT::Discretization&      discretization,
                                           DRT::Condition&           condition,
                                           vector<int>&              lm,
                                           Epetra_SerialDenseVector& elevec1)
{
  dserror("Body force of wedge6 not implemented");
  return 0;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (private)                             maf 04/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::sow6_nlnstiffmass(
      vector<int>&              lm,             // location matrix
      vector<double>&           disp,           // current displacements
      vector<double>&           residual,       // current residuum
      Epetra_SerialDenseMatrix* stiffmatrix,    // element stiffness matrix
      Epetra_SerialDenseMatrix* massmatrix,     // element mass matrix
      Epetra_SerialDenseVector* force,          // element internal force vector
      Epetra_SerialDenseMatrix* elestress,      // element stresses
      Epetra_SerialDenseMatrix* elestrain,      // strains at GP
      const double              time,           // current absolute time
      const bool                cauchy)         // stress output option
{

/* ============================================================================*
** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for Wedge_6 with 6 GAUSS POINTS*
** ============================================================================*/
/* pointer to (static) shape function array
 * for each node, evaluated at each gp*/
  Epetra_SerialDenseMatrix* shapefct; //[NUMNOD_WEG6][NUMGPT_WEG6]
/* pointer to (static) shape function derivatives array
 * for each node wrt to each direction, evaluated at each gp*/
  Epetra_SerialDenseMatrix* deriv;    //[NUMGPT_WEG6*NUMDIM][NUMNOD_WEG6]
/* pointer to (static) weight factors at each gp */
  Epetra_SerialDenseVector* weights;  //[NUMGPT_WEG6]
  sow6_shapederiv(&shapefct,&deriv,&weights);   // call to evaluate
/* ============================================================================*/

  // update element geometry
  Epetra_SerialDenseMatrix xrefe(NUMNOD_WEG6,NUMDIM_WEG6);  // material coord. of element
  Epetra_SerialDenseMatrix xcurr(NUMNOD_WEG6,NUMDIM_WEG6);  // current  coord. of element
  for (int i=0; i<NUMNOD_WEG6; ++i){
    xrefe(i,0) = Nodes()[i]->X()[0];
    xrefe(i,1) = Nodes()[i]->X()[1];
    xrefe(i,2) = Nodes()[i]->X()[2];

    xcurr(i,0) = xrefe(i,0) + disp[i*NODDOF_WEG6+0];
    xcurr(i,1) = xrefe(i,1) + disp[i*NODDOF_WEG6+1];
    xcurr(i,2) = xrefe(i,2) + disp[i*NODDOF_WEG6+2];
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  for (int gp=0; gp<NUMGPT_WEG6; ++gp) {

    // get submatrix of deriv at actual gp
    Epetra_SerialDenseMatrix deriv_gp(NUMDIM_WEG6,NUMGPT_WEG6);
    for (int m=0; m<NUMDIM_WEG6; ++m) {
      for (int n=0; n<NUMGPT_WEG6; ++n) {
        deriv_gp(m,n)=(*deriv)(NUMDIM_WEG6*gp+m,n);
      }
    }

    /* compute the Jacobian matrix which looks like:
    **         [ x_,r  y_,r  z_,r ]
    **     J = [ x_,s  y_,s  z_,s ]
    **         [ x_,t  y_,t  z_,t ]
    */
    Epetra_SerialDenseMatrix jac(NUMDIM_WEG6,NUMDIM_WEG6);
    jac.Multiply('N','N',1.0,deriv_gp,xrefe,1.0);

    // compute determinant of Jacobian by Sarrus' rule
    double detJ= jac(0,0) * jac(1,1) * jac(2,2)
               + jac(0,1) * jac(1,2) * jac(2,0)
               + jac(0,2) * jac(1,0) * jac(2,1)
               - jac(0,0) * jac(1,2) * jac(2,1)
               - jac(0,1) * jac(1,0) * jac(2,2)
               - jac(0,2) * jac(1,1) * jac(2,0);
    if (abs(detJ) < 1E-16) dserror("ZERO JACOBIAN DETERMINANT");
    else if (detJ < 0.0) dserror("NEGATIVE JACOBIAN DETERMINANT");

    /* compute derivatives N_XYZ at gp w.r.t. material coordinates
    ** by solving   Jac . N_XYZ = N_rst   for N_XYZ
    ** Inverse of Jacobian is therefore not explicitly computed
    */
    Epetra_SerialDenseMatrix N_XYZ(NUMDIM_WEG6,NUMNOD_WEG6);
    Epetra_SerialDenseSolver solve_for_inverseJac;  // solve A.X=B
    solve_for_inverseJac.SetMatrix(jac);            // set A=jac
    solve_for_inverseJac.SetVectors(N_XYZ,deriv_gp);// set X=N_XYZ, B=deriv_gp
    solve_for_inverseJac.FactorWithEquilibration(true);
    int err2 = solve_for_inverseJac.Factor();
    int err = solve_for_inverseJac.Solve();         // N_XYZ = J^-1.N_rst
    if ((err != 0) && (err2!=0)) dserror("Inversion of Jacobian failed");

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
    Epetra_SerialDenseMatrix defgrd(NUMDIM_WEG6,NUMDIM_WEG6);
    defgrd.Multiply('T','T',1.0,xcurr,N_XYZ,1.0);

    // Right Cauchy-Green tensor = F^T * F
    Epetra_SerialDenseMatrix cauchygreen(NUMDIM_WEG6,NUMDIM_WEG6);
    cauchygreen.Multiply('T','N',1.0,defgrd,defgrd,1.0);

    // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    Epetra_SerialDenseVector glstrain(NUMSTR_WEG6);
    glstrain(0) = 0.5 * (cauchygreen(0,0) - 1.0);
    glstrain(1) = 0.5 * (cauchygreen(1,1) - 1.0);
    glstrain(2) = 0.5 * (cauchygreen(2,2) - 1.0);
    glstrain(3) = cauchygreen(0,1);
    glstrain(4) = cauchygreen(1,2);
    glstrain(5) = cauchygreen(2,0);

    // return gp strains (only in case of stress/strain output)
    if (elestress != NULL){
      for (int i = 0; i < 3; ++i) {
        (*elestrain)(gp,i) = glstrain(i);
      }
      for (int i = 3; i < 6; ++i) {
        (*elestrain)(gp,i) = 0.5 * glstrain(i);
      }
    }

    /* non-linear B-operator (may so be called, meaning
    ** of B-operator is not so sharp in the non-linear realm) *
    ** B = F . Bl *
    **
    **      [ ... | F_11*N_{,1}^k  F_21*N_{,1}^k  F_31*N_{,1}^k | ... ]
    **      [ ... | F_12*N_{,2}^k  F_22*N_{,2}^k  F_32*N_{,2}^k | ... ]
    **      [ ... | F_13*N_{,3}^k  F_23*N_{,3}^k  F_33*N_{,3}^k | ... ]
    ** B =  [ ~~~   ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~   ~~~ ]
    **      [       F_11*N_{,2}^k+F_12*N_{,1}^k                       ]
    **      [ ... |          F_21*N_{,2}^k+F_22*N_{,1}^k        | ... ]
    **      [                       F_31*N_{,2}^k+F_32*N_{,1}^k       ]
    **      [                                                         ]
    **      [       F_12*N_{,3}^k+F_13*N_{,2}^k                       ]
    **      [ ... |          F_22*N_{,3}^k+F_23*N_{,2}^k        | ... ]
    **      [                       F_32*N_{,3}^k+F_33*N_{,2}^k       ]
    **      [                                                         ]
    **      [       F_13*N_{,1}^k+F_11*N_{,3}^k                       ]
    **      [ ... |          F_23*N_{,1}^k+F_21*N_{,3}^k        | ... ]
    **      [                       F_33*N_{,1}^k+F_31*N_{,3}^k       ]
    */
    Epetra_SerialDenseMatrix bop(NUMSTR_WEG6,NUMDOF_WEG6);
    for (int i=0; i<NUMNOD_WEG6; ++i) {
      bop(0,NODDOF_WEG6*i+0) = defgrd(0,0)*N_XYZ(0,i);
      bop(0,NODDOF_WEG6*i+1) = defgrd(1,0)*N_XYZ(0,i);
      bop(0,NODDOF_WEG6*i+2) = defgrd(2,0)*N_XYZ(0,i);
      bop(1,NODDOF_WEG6*i+0) = defgrd(0,1)*N_XYZ(1,i);
      bop(1,NODDOF_WEG6*i+1) = defgrd(1,1)*N_XYZ(1,i);
      bop(1,NODDOF_WEG6*i+2) = defgrd(2,1)*N_XYZ(1,i);
      bop(2,NODDOF_WEG6*i+0) = defgrd(0,2)*N_XYZ(2,i);
      bop(2,NODDOF_WEG6*i+1) = defgrd(1,2)*N_XYZ(2,i);
      bop(2,NODDOF_WEG6*i+2) = defgrd(2,2)*N_XYZ(2,i);
      /* ~~~ */
      bop(3,NODDOF_WEG6*i+0) = defgrd(0,0)*N_XYZ(1,i) + defgrd(0,1)*N_XYZ(0,i);
      bop(3,NODDOF_WEG6*i+1) = defgrd(1,0)*N_XYZ(1,i) + defgrd(1,1)*N_XYZ(0,i);
      bop(3,NODDOF_WEG6*i+2) = defgrd(2,0)*N_XYZ(1,i) + defgrd(2,1)*N_XYZ(0,i);
      bop(4,NODDOF_WEG6*i+0) = defgrd(0,1)*N_XYZ(2,i) + defgrd(0,2)*N_XYZ(1,i);
      bop(4,NODDOF_WEG6*i+1) = defgrd(1,1)*N_XYZ(2,i) + defgrd(1,2)*N_XYZ(1,i);
      bop(4,NODDOF_WEG6*i+2) = defgrd(2,1)*N_XYZ(2,i) + defgrd(2,2)*N_XYZ(1,i);
      bop(5,NODDOF_WEG6*i+0) = defgrd(0,2)*N_XYZ(0,i) + defgrd(0,0)*N_XYZ(2,i);
      bop(5,NODDOF_WEG6*i+1) = defgrd(1,2)*N_XYZ(0,i) + defgrd(1,0)*N_XYZ(2,i);
      bop(5,NODDOF_WEG6*i+2) = defgrd(2,2)*N_XYZ(0,i) + defgrd(2,0)*N_XYZ(2,i);
    }

    /* call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ** Here all possible material laws need to be incorporated,
    ** the stress vector, a C-matrix, and a density must be retrieved,
    ** every necessary data must be passed.
    */
    Epetra_SerialDenseMatrix cmat(NUMSTR_WEG6,NUMSTR_WEG6);
    Epetra_SerialDenseVector stress(NUMSTR_WEG6);
    double density;
    sow6_mat_sel(&stress,&cmat,&density,&glstrain, time);
    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    // return gp stresses
    if (elestress != NULL){
      if (!cauchy) {
        for (int i = 0; i < NUMSTR_WEG6; ++i) {
          (*elestress)(gp,i) = stress(i);
        }
      }
      else {                               // return Cauchy stresses
        double detF = defgrd(0,0)*defgrd(1,1)*defgrd(2,2) +
                      defgrd(0,1)*defgrd(1,2)*defgrd(2,0) +
                      defgrd(0,2)*defgrd(1,0)*defgrd(2,1) -
                      defgrd(0,2)*defgrd(1,1)*defgrd(2,0) -
                      defgrd(0,0)*defgrd(1,2)*defgrd(2,1) -
                      defgrd(0,1)*defgrd(1,0)*defgrd(2,2);

        LINALG::SerialDenseMatrix pkstress(NUMDIM_WEG6,NUMDIM_WEG6);
        pkstress(0,0) = stress(0);
        pkstress(0,1) = stress(3);
        pkstress(0,2) = stress(5);
        pkstress(1,0) = pkstress(0,1);
        pkstress(1,1) = stress(1);
        pkstress(1,2) = stress(4);
        pkstress(2,0) = pkstress(0,2);
        pkstress(2,1) = pkstress(1,2);
        pkstress(2,2) = stress(2);

        LINALG::SerialDenseMatrix temp(NUMDIM_WEG6,NUMDIM_WEG6);
        LINALG::SerialDenseMatrix cauchystress(NUMDIM_WEG6,NUMDIM_WEG6);
        temp.Multiply('N','N',1.0/detF,defgrd,pkstress,0.);
        cauchystress.Multiply('N','T',1.0,temp,defgrd,0.);

        (*elestress)(gp,0) = cauchystress(0,0);
        (*elestress)(gp,1) = cauchystress(1,1);
        (*elestress)(gp,2) = cauchystress(2,2);
        (*elestress)(gp,3) = cauchystress(0,1);
        (*elestress)(gp,4) = cauchystress(1,2);
        (*elestress)(gp,5) = cauchystress(0,2);
      }
    }

    if (force != NULL && stiffmatrix != NULL) {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      (*force).Multiply('T','N',detJ * (*weights)(gp),bop,stress,1.0);

      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      Epetra_SerialDenseMatrix cb(NUMSTR_WEG6,NUMDOF_WEG6);
      cb.Multiply('N','N',1.0,cmat,bop,1.0);          // temporary C . B
      (*stiffmatrix).Multiply('T','N',detJ * (*weights)(gp),bop,cb,1.0);

      // integrate `geometric' stiffness matrix and add to keu *****************
      Epetra_SerialDenseVector sfac(stress); // auxiliary integrated stress
      sfac.Scale(detJ * (*weights)(gp));     // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]
      vector<double> SmB_L(NUMDIM_WEG6);     // intermediate Sm.B_L
      // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
      for (int inod=0; inod<NUMNOD_WEG6; ++inod){
        SmB_L[0] = sfac(0) * N_XYZ(0,inod) + sfac(3) * N_XYZ(1,inod) + sfac(5) * N_XYZ(2,inod);
        SmB_L[1] = sfac(3) * N_XYZ(0,inod) + sfac(1) * N_XYZ(1,inod) + sfac(4) * N_XYZ(2,inod);
        SmB_L[2] = sfac(5) * N_XYZ(0,inod) + sfac(4) * N_XYZ(1,inod) + sfac(2) * N_XYZ(2,inod);
        for (int jnod=0; jnod<NUMNOD_WEG6; ++jnod){
          double bopstrbop = 0.0;            // intermediate value
          for (int idim=0; idim<NUMDIM_WEG6; ++idim) bopstrbop += N_XYZ(idim,jnod) * SmB_L[idim];
          (*stiffmatrix)(NUMDIM_WEG6*inod+0,NUMDIM_WEG6*jnod+0) += bopstrbop;
          (*stiffmatrix)(NUMDIM_WEG6*inod+1,NUMDIM_WEG6*jnod+1) += bopstrbop;
          (*stiffmatrix)(NUMDIM_WEG6*inod+2,NUMDIM_WEG6*jnod+2) += bopstrbop;
        }
      } // end of integrate `geometric' stiffness ******************************
    }

    if (massmatrix != NULL){ // evaluate mass matrix +++++++++++++++++++++++++
      // integrate concistent mass matrix
      for (int inod=0; inod<NUMNOD_WEG6; ++inod) {
        for (int jnod=0; jnod<NUMNOD_WEG6; ++jnod) {
          double massfactor = (*shapefct)(inod,gp) * density * (*shapefct)(jnod,gp)
                            * detJ * (*weights)(gp);     // intermediate factor
          (*massmatrix)(NUMDIM_WEG6*inod+0,NUMDIM_WEG6*jnod+0) += massfactor;
          (*massmatrix)(NUMDIM_WEG6*inod+1,NUMDIM_WEG6*jnod+1) += massfactor;
          (*massmatrix)(NUMDIM_WEG6*inod+2,NUMDIM_WEG6*jnod+2) += massfactor;
        }
      }
    } // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++
   /* =========================================================================*/
  }/* ==================================================== end of Loop over GP */
   /* =========================================================================*/

  return;
} // DRT::ELEMENTS::Shell8::s8_nlnstiffmass



/*----------------------------------------------------------------------*
 |  shape functions and derivatives for So_hex8                maf 04/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_weg6::sow6_shapederiv(
      Epetra_SerialDenseMatrix** shapefct,  // pointer to pointer of shapefct
      Epetra_SerialDenseMatrix** deriv,     // pointer to pointer of derivs
      Epetra_SerialDenseVector** weights)   // pointer to pointer of weights
{
  // static matrix objects, kept in memory
  static Epetra_SerialDenseMatrix  f(NUMNOD_WEG6,NUMGPT_WEG6);  // shape functions
  static Epetra_SerialDenseMatrix df(NUMDOF_WEG6,NUMNOD_WEG6);  // derivatives
  static Epetra_SerialDenseVector weightfactors(NUMGPT_WEG6);   // weights for each gp
  static bool fdf_eval;                      // flag for re-evaluate everything


  if (fdf_eval==true) { // if true f,df already evaluated
    *shapefct = &f; // return adress of static object to target of pointer
    *deriv = &df; // return adress of static object to target of pointer
    *weights = &weightfactors; // return adress of static object to target of pointer
    return;
  }
  else {
    // (r,s,t) gp-locations of fully integrated linear 6-node Wedge
    // fill up nodal f at each gp
    // fill up df w.r.t. rst directions (NUMDIM) at each gp
    const DRT::UTILS::GaussRule3D gaussrule_ = DRT::UTILS::intrule_wedge_6point;
    const DRT::UTILS::IntegrationPoints3D intpoints = getIntegrationPoints3D(gaussrule_);
    for (int igp = 0; igp < intpoints.nquad; ++igp) {
      const double r = intpoints.qxg[igp][0];
      const double s = intpoints.qxg[igp][1];
      const double t = intpoints.qxg[igp][2];

      Epetra_SerialDenseVector funct(NUMNOD_WEG6);
      Epetra_SerialDenseMatrix deriv(NUMDIM_WEG6, NUMNOD_WEG6);
      DRT::UTILS::shape_function_3D(funct, r, s, t, wedge6);
      DRT::UTILS::shape_function_3D_deriv1(deriv, r, s, t, wedge6);
      for (int inode = 0; inode < NUMNOD_WEG6; ++inode) {
        f(inode, igp) = funct(inode);
        df(igp*NUMDIM_WEG6+0, inode) = deriv(0, inode);
        df(igp*NUMDIM_WEG6+1, inode) = deriv(1, inode);
        df(igp*NUMDIM_WEG6+2, inode) = deriv(2, inode);
        weightfactors[igp] = intpoints.qwgt[igp];
      }
    }
    // return adresses of just evaluated matrices
    *shapefct = &f; // return adress of static object to target of pointer
    *deriv = &df; // return adress of static object to target of pointer
    *weights = &weightfactors; // return adress of static object to target of pointer
    fdf_eval = true; // now all arrays are filled statically
  }
  return;
}  // of sow6_shapederiv

/*----------------------------------------------------------------------*
 |  init the element (public)                                  maf 07/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Sow6Register::Initialize(DRT::Discretization& dis)
{
  int j =0;
  while (!dynamic_cast<DRT::ELEMENTS::So_weg6*>(dis.lColElement(j)))
  {
	j++;
  }
  DRT::ELEMENTS::So_weg6* actele = dynamic_cast<DRT::ELEMENTS::So_weg6*>(dis.lColElement(j));
  if (!actele->donerewinding_)
    {
      DRT::UTILS::Rewinding3D(dis);
      dis.FillComplete(false,false,false);

	  for (int i=0; i<dis.NumMyColElements(); ++i)
	    {
		  // get the actual element
		  if (dynamic_cast<DRT::ELEMENTS::So_weg6*>(dis.lColElement(i)))
		  {
			  DRT::ELEMENTS::So_weg6* actele = dynamic_cast<DRT::ELEMENTS::So_weg6*>(dis.lColElement(i));
			  actele->donerewinding_ = true;
		  }
		  else if (dynamic_cast<DRT::ELEMENTS::So_hex8*>(dis.lColElement(i)))
		  {
			  DRT::ELEMENTS::So_hex8* actele = dynamic_cast<DRT::ELEMENTS::So_hex8*>(dis.lColElement(i));
			  actele->donerewinding_ = true;
		  }
		  else if (dynamic_cast<DRT::ELEMENTS::So_tet4*>(dis.lColElement(i)))
		  {
			  DRT::ELEMENTS::So_tet4* actele = dynamic_cast<DRT::ELEMENTS::So_tet4*>(dis.lColElement(i));
		  	  actele->donerewinding_ = true;
		  }
		  else if (dynamic_cast<DRT::ELEMENTS::So_tet10*>(dis.lColElement(i)))
		  {
			  DRT::ELEMENTS::So_tet10* actele = dynamic_cast<DRT::ELEMENTS::So_tet10*>(dis.lColElement(i));
		  	  actele->donerewinding_ = true;
		  }
		  else if (dynamic_cast<DRT::ELEMENTS::SoDisp*>(dis.lColElement(i)))
		  {
			  DRT::ELEMENTS::SoDisp* actele = dynamic_cast<DRT::ELEMENTS::SoDisp*>(dis.lColElement(i));
		  	  actele->donerewinding_ = true;
		  }
		  else
		  {
			  cout << "???" << endl;
		  }

	    }
	// fill complete again to reconstruct element-node pointers,
	// but without element init, etc.
	dis.FillComplete(false,false,false);
    }


  return 0;
}


#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_WEG6
