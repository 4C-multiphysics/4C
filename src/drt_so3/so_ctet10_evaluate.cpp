/*!----------------------------------------------------------------------*
\file so_ctet10_evaluate.cpp
\brief quadratic nonlinear tetrahedron 

<pre>
Maintainer: Moritz Frenzel
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15240
written by: Alexander Volf
			alexander.volf@mytum.de 
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET

// This is just here to get the c++ mpi header, otherwise it would
// use the c version included inside standardtypes.h
#ifdef PARALLEL
#include "mpi.h"
#endif
#include "so_integrator.H"
#include "so_ctet10.H"
#include "so_hex8.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_exporter.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/linalg_serialdensematrix.H"
#include "../drt_lib/linalg_serialdensevector.H"
#include "Epetra_SerialDenseSolver.h"


//#define VERBOSE_OUTPUT
using namespace std; // cout etc.
using namespace LINALG; // our linear algebra

/*----------------------------------------------------------------------*
 |                                                         maf 04/07    |
 | vector of material laws                                              |
 | defined in global_control.c											|
 *----------------------------------------------------------------------*/
extern struct _MATERIAL  *mat;


/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              vlf 06/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_ctet10::Evaluate(ParameterList& params,
                                    DRT::Discretization&      discretization,
                                    vector<int>&              lm,
                                    Epetra_SerialDenseMatrix& elemat1,
                                    Epetra_SerialDenseMatrix& elemat2,
                                    Epetra_SerialDenseVector& elevec1,
                                    Epetra_SerialDenseVector& elevec2,
                                    Epetra_SerialDenseVector& elevec3)
{
  // start with "none"
  DRT::ELEMENTS::So_ctet10::ActionType act = So_ctet10::none;

  // get the required action
  string action = params.get<string>("action","none");
  if (action == "none") dserror("No action supplied");
  else if (action=="calc_struct_linstiff")      act = So_ctet10::calc_struct_linstiff;
  else if (action=="calc_struct_nlnstiff")      act = So_ctet10::calc_struct_nlnstiff;
  else if (action=="calc_struct_internalforce") act = So_ctet10::calc_struct_internalforce;
  else if (action=="calc_struct_linstiffmass")  act = So_ctet10::calc_struct_linstiffmass;
  else if (action=="calc_struct_nlnstiffmass")  act = So_ctet10::calc_struct_nlnstiffmass;
  else if (action=="calc_struct_nlnstifflmass") act = So_ctet10::calc_struct_nlnstifflmass;
  else if (action=="calc_struct_stress")        act = So_ctet10::calc_struct_stress;
  else if (action=="calc_struct_eleload")       act = So_ctet10::calc_struct_eleload;
  else if (action=="calc_struct_fsiload")       act = So_ctet10::calc_struct_fsiload;
  else if (action=="calc_struct_update_istep")  act = So_ctet10::calc_struct_update_istep;
  else if (action=="calc_struct_update_genalpha_imrlike")  act = So_ctet10::calc_struct_update_genalpha_imrlike;
  else if (action=="postprocess_stress")        act = So_ctet10::postprocess_stress;
  else dserror("Unknown type of action for So_ctet10");

  // get the material law
  MATERIAL* actmat = &(mat[material_-1]);

  // what should the element do
  switch(act) {
    // linear stiffness
    case calc_struct_linstiff: {
      // need current displacement and residual forces
      vector<double> mydisp(lm.size());
      for (int i=0; i<(int)mydisp.size(); ++i) mydisp[i] = 0.0;
      vector<double> myres(lm.size());
      for (int i=0; i<(int)myres.size(); ++i) myres[i] = 0.0;
      so_ctet10_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,actmat);

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

      so_ctet10_nlnstiffmass(lm,mydisp,myres,&elemat1,NULL,&elevec1,NULL,NULL,actmat);
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

      so_ctet10_nlnstiffmass(lm,mydisp,myres,&elemat1,&elemat2,&elevec1,NULL,NULL,actmat);

      if (act==calc_struct_nlnstifflmass) so_ctet10_lumpmass(&elemat2);
    }
    break;

     // evaluate stresses
    case calc_struct_stress: {
      RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      RCP<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      RCP<vector<char> > stressdata = params.get<RCP<vector<char> > >("stress", null);
      RCP<vector<char> > straindata = params.get<RCP<vector<char> > >("strain", null);
      if (disp==null) dserror("Cannot get state vectors 'displacement'");
      if (stressdata==null) dserror("Cannot get stress 'data'");
      if (straindata==null) dserror("Cannot get strain 'data'");
      vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
      vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);
      Epetra_SerialDenseMatrix stress(NUMNOD_SOCTET10,NUMSTR_SOCTET10);
      Epetra_SerialDenseMatrix strain(NUMNOD_SOCTET10,NUMSTR_SOCTET10);
      bool cauchy = params.get<bool>("cauchy", false);
      so_ctet10_nlnstiffstress(lm,mydisp,myres,NULL,NULL,NULL,&stress,&strain,actmat,cauchy);
      AddtoPack(*stressdata, stress);
      AddtoPack(*straindata, strain);
    }
    break;


	// postprocess stresses at gauss points
    case postprocess_stress:{
      const RCP<std::map<int,RCP<Epetra_SerialDenseMatrix> > > gpstressmap=
        params.get<RCP<std::map<int,RCP<Epetra_SerialDenseMatrix> > > >("gpstressmap",null);
      if (gpstressmap==null)
        dserror("no gp stress map available for postprocessing");
      string stresstype = params.get<string>("stresstype","ndxyz");
      int gid = Id();
      RCP<Epetra_SerialDenseMatrix> gpstress = (*gpstressmap)[gid];

      if (stresstype=="ndxyz") {
        // extrapolate stresses at Gauss points to nodes
        Epetra_SerialDenseMatrix nodalstresses(NUMNOD_SOCTET10,NUMSTR_SOCTET10);
        so_ctet10_expol(*gpstress,nodalstresses);

        // average nodal stresses between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_SOCTET10);

        for (int i=0;i<NUMNOD_SOCTET10;++i){
          DRT::Node* node=Nodes()[i];
          numadjele[i]=node->NumElement();
        }

        for (int i=0;i<NUMNOD_SOCTET10;++i){
          elevec1(3*i)=nodalstresses(i,0)/numadjele[i];
          elevec1(3*i+1)=nodalstresses(i,1)/numadjele[i];
          elevec1(3*i+2)=nodalstresses(i,2)/numadjele[i];
        }
        for (int i=0;i<NUMNOD_SOCTET10;++i){
          elevec2(3*i)=nodalstresses(i,3)/numadjele[i];
          elevec2(3*i+1)=nodalstresses(i,4)/numadjele[i];
          elevec2(3*i+2)=nodalstresses(i,5)/numadjele[i];
        }
      }
      else if (stresstype=="cxyz") {
        RCP<Epetra_MultiVector> elestress=params.get<RCP<Epetra_MultiVector> >("elestress",null);
        if (elestress==null)
          dserror("No element stress vector available");
        const Epetra_BlockMap elemap = elestress->Map();
        int lid = elemap.LID(Id());
        if (lid!=-1) {
          for (int i = 0; i < NUMSTR_SOCTET10; ++i) {
            (*((*elestress)(i)))[lid] = 0.;
            for (int j = 0; j < NUMGPT_SOCTET10; ++j) {
              (*((*elestress)(i)))[lid] += 1.0/NUMGPT_SOCTET10 * (*gpstress)(j,i);
            }
          }
        }
      }
      else if (stresstype=="cxyz_ndxyz") {
        // extrapolate stresses at Gauss points to nodes
        Epetra_SerialDenseMatrix nodalstresses(NUMNOD_SOCTET10,NUMSTR_SOCTET10);
        so_ctet10_expol(*gpstress,nodalstresses);

        // average nodal stresses between elements
        // -> divide by number of adjacent elements
        vector<int> numadjele(NUMNOD_SOCTET10);

        for (int i=0;i<NUMNOD_SOCTET10;++i){
          DRT::Node* node=Nodes()[i];
          numadjele[i]=node->NumElement();
        }

        for (int i=0;i<NUMNOD_SOCTET10;++i){
          elevec1(3*i)=nodalstresses(i,0)/numadjele[i];
          elevec1(3*i+1)=nodalstresses(i,1)/numadjele[i];
          elevec1(3*i+2)=nodalstresses(i,2)/numadjele[i];
        }
        for (int i=0;i<NUMNOD_SOCTET10;++i){
          elevec2(3*i)=nodalstresses(i,3)/numadjele[i];
          elevec2(3*i+1)=nodalstresses(i,4)/numadjele[i];
          elevec2(3*i+2)=nodalstresses(i,5)/numadjele[i];
        }
        RCP<Epetra_MultiVector> elestress=params.get<RCP<Epetra_MultiVector> >("elestress",null);
        if (elestress==null)
          dserror("No element stress vector available");
        const Epetra_BlockMap elemap = elestress->Map();
        int lid = elemap.LID(Id());
        if (lid!=-1) {
          for (int i = 0; i < NUMSTR_SOCTET10; ++i) {
            (*((*elestress)(i)))[lid] = 0.;
            for (int j = 0; j < NUMGPT_SOCTET10; ++j) {
              (*((*elestress)(i)))[lid] += 1.0/NUMGPT_SOCTET10 * (*gpstress)(j,i);
            }
          }
        }
      }
      else{
        dserror("unknown type of stress output on element level");
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
 |  Integrate a Volume Neumann boundary condition (public)     vlf 01/08|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::So_ctet10::EvaluateNeumann(ParameterList& params,
                                           DRT::Discretization&      discretization,
                                           DRT::Condition&           condition,
                                           vector<int>&              lm,
                                           Epetra_SerialDenseVector& elevec1)
{
  // get values and switches from the condition
  const vector<int>*    onoff = condition.Get<vector<int> >   ("onoff");
  const vector<double>* val   = condition.Get<vector<double> >("val"  );

  /*
  **    TIME CURVE BUSINESS
  */
  // find out whether we will use a time curve
  bool usetime = true;
  const double time = params.get("total time",-1.0);
  if (time<0.0) usetime = false;

  // find out whether we will use a time curve and get the factor
  const vector<int>* curve  = condition.Get<vector<int> >("curve");
  int curvenum = -1;
  if (curve) curvenum = (*curve)[0];
  double curvefac = 1.0;
  if (curvenum>=0 && usetime)
    curvefac = DRT::UTILS::TimeCurveManager::Instance().Curve(curvenum).f(time);
  // **
  
/* =========================*
 * SUB STRUCTURE for CTET_10 
 * =========================*/
  // update element geometry
  Epetra_SerialDenseMatrix xrefe(NUMNOD_SOCTET10,NUMDIM_SOCTET10);  // material coord. of element
  /* structure of xrefe:
    **             [  X_1   Y_1   Z_1  ]
    **     xrefe = [  X_2   Y_2   Z_2  ]
    **             [   |     |     |   ]
    **             [  X_10  Y_10  Z_10 ]
    */
  // the ussual translation of the whole element to the zero is done
  // in order to improve numerical accuracy  
  for (int i=0; i<NUMNOD_SOCTET10; ++i){
    xrefe(i,0) = Nodes()[i]->X()[0] - Nodes()[0]->X()[0];
    xrefe(i,1) = Nodes()[i]->X()[1] - Nodes()[0]->X()[1];
    xrefe(i,2) = Nodes()[i]->X()[2] - Nodes()[0]->X()[2];
  }
 
  Epetra_SerialDenseMatrix jac_coord(NUMCOORD_SUBTET4,NUMCOORD_SUBTET4);
  for (int i=0; i<4; i++)  jac_coord(0,i)=1;
  for (int row=0;row<3;row++)
  {
   	for (int col=0;col<4;col++)
       jac_coord(row+1,col) = xrefe(row,col);	
  }
   
  double volume=det_volf(jac_coord)/(double) 6;    //volume of a tetrahedra

  // integration factor 
  // - is quite straight forward here as all sub elems are linear 
  // their shape function values on all nodes are 1
  // a sumation makes 
  double intfactor = curvefac * volume / 32.0 * curvefac;
  for (int i=0 ; i< 4; i ++)
  {
  		for (int dim = 0; dim < NUMDIM_SOCTET10; dim ++)
  		{
  			elevec1(NUMDIM_SOCTET10*i+dim)= intfactor * (*onoff)[dim] * (*val)[dim];
  		}
  }
  
  intfactor = curvefac * volume * 7.0 /48.0;
  for (int i=4 ; i< NUMNOD_SOCTET10; i ++)
  {
   		for (int dim = 0; dim < NUMDIM_SOCTET10; dim ++)
  		{
  			elevec1(NUMDIM_SOCTET10*i+dim)= intfactor * (*onoff)[dim] * (*val)[dim];
  		}
  }

  return 0;
} // DRT::ELEMENTS::So_ctet10::EvaluateNeumann



/*----------------------------------------------------------------------*
 |  evaluate the element (private)                            vlf 06/07 |
 *----------------------------------------------------------------------*/

void DRT::ELEMENTS::So_ctet10::so_ctet10_nlnstiffmass(
      vector<int>&              lm,             // location matrix
      vector<double>&           disp,           // current displacements
      vector<double>&           residual,       // current residuum
      Epetra_SerialDenseMatrix* stiffmatrix,    // element stiffness matrix
      Epetra_SerialDenseMatrix* massmatrix,     // element mass matrix
      Epetra_SerialDenseVector* force,          // element internal force vector
      Epetra_SerialDenseMatrix* elestress,      // stresses at GP
      Epetra_SerialDenseMatrix* elestrain,      // strains at GP
      struct _MATERIAL*         material,       // element material data
      const bool                cauchy)         // stress output options
{
/* =========================*
 * SUB STRUCTURE for CTET_10 
 * =========================*/
  // update element geometry
  Epetra_SerialDenseMatrix xrefe(NUMNOD_SOCTET10+1,NUMDIM_SOCTET10);  // material coord. of element
  /* structure of xrefe:
    **             [  X_1   Y_1   Z_1  ]
    **     xrefe = [  X_2   Y_2   Z_2  ]
    **             [   |     |     |   ]
    **             [  X_10  Y_10  Z_10 ]
    */
  Epetra_SerialDenseMatrix xcurr(NUMNOD_SOCTET10,NUMDIM_SOCTET10);  // current  coord. of element
  /* structure of xcurr:
    **             [  x_1   y_1   z_1  ]
    **     xcurr = [  x_2   y_2   z_2  ]
    **             [   |     |     |   ]
    **             [  x_10  y_10  z_10 ]
    */   
    
  Epetra_SerialDenseMatrix xdisp(NUMNOD_SOCTET10,NUMDIM_SOCTET10);  // current  coord. of element
  // translation of the whole element to the zero is done
  // in order to improve numerical accuracy  
  for (int i=0; i<NUMNOD_SOCTET10; ++i){
    xrefe(i,0) = Nodes()[i]->X()[0] - Nodes()[0]->X()[0];
    xrefe(i,1) = Nodes()[i]->X()[1] - Nodes()[0]->X()[1];
    xrefe(i,2) = Nodes()[i]->X()[2] - Nodes()[0]->X()[2];

    xcurr(i,0) = xrefe(i,0) + disp[i*NODDOF_SOCTET10+0];
    xcurr(i,1) = xrefe(i,1) + disp[i*NODDOF_SOCTET10+1];
    xcurr(i,2) = xrefe(i,2) + disp[i*NODDOF_SOCTET10+2];
    
    xdisp(i,0) = disp[i*NODDOF_SOCTET10+0];
    xdisp(i,1) = disp[i*NODDOF_SOCTET10+1];
    xdisp(i,2) = disp[i*NODDOF_SOCTET10+2];
  }
 
  //create the midpoint of the tetrahedron as the 11th node of the element 
  xrefe(NUMNOD_SOCTET10,0)=ctet10_midpoint(0);
  xrefe(NUMNOD_SOCTET10,1)=ctet10_midpoint(1);
  xrefe(NUMNOD_SOCTET10,2)=ctet10_midpoint(2);
  
  //build the sub-elements for the whole tetrahedron
  SUB_STRUCTURE sub(xrefe);
 
  //remove the extra added 11th node
  xrefe.Reshape(NUMNOD_SOCTET10,NUMDIM_SOCTET10);

  //create the integrator from the sub-elements  
  L_AJ_integrator L_aj_int(sub);  
 	
  /* =========================================================================*/
  /* ============================================== Loop over Gauss Points ===*/
  /* =========================================================================*/
  for (int gp=0; gp<NUMGPT_SOCTET10; gp++) {
    
    /* structure of L_aj_int.deriv_gp, which replaces N_XYZ:
    **                               [   dN_1     dN_1     dN_1   ]
    **                               [  ------   ------   ------  ]
    **                               [    dX       dY       dZ    ]
    **    L_XYZ (is equivalent to)=  [     |        |        |    ]
    **                               [                            ]
    **                               [   dN_10    dN_10    dN_10  ]
    **                               [  -------  -------  ------- ]
    **                               [    dX       dY       dZ    ]
    */
    
    //                                      d xcurr 
    // (material) deformation gradient F = --------- = L_XYZ^T * xcurr
    //                                      d xrefe  
    
    /*structure of F
    **             [    dx       dy       dz    ]
    **             [  ------   ------   ------  ]
    **             [    dX       dX       dX    ]
    **             [                            ]
    **      F   =  [    dx       dy       dz    ]
    **             [  ------   ------   ------  ]
    **             [    dY       dY       dY    ]
    **             [                            ]
    **             [    dx       dy       dz    ]
    **             [  ------   ------   ------  ]
    **             [    dZ       dZ       dZ    ]
    */
    
    /* Here additionally a decompostion of F = L_XYZ^T * (xrefe + (x)disp)
     * Knowing that L_XYZ^T * xrefe = diag(1,1,1) ,
     * we can ensure F = diag(1,1,1) if disp = 0 
     */

    Epetra_SerialDenseMatrix defgrd(NUMDIM_SOCTET10,NUMDIM_SOCTET10);
    
    defgrd.Multiply('T','N',1.0,xdisp,L_aj_int.deriv_gp[gp],0.0);
    defgrd(0,0)+=1;
    defgrd(1,1)+=1;
    defgrd(2,2)+=1;
    
    
    #ifdef VERBOSE_OUTPUT
	cout << "defgr\n " << defgrd;
	#endif //VERBOSE_OUTPUT

    // Right Cauchy-Green tensor = F^T * F 
    Epetra_SerialDenseMatrix cauchygreen(NUMDIM_SOCTET10,NUMDIM_SOCTET10);
   
    cauchygreen.Multiply('T','N',1.0,defgrd,defgrd,0.0);
    
	#ifdef VERBOSE_OUTPUT
	cout << "cauchygreen\n" << cauchygreen;
	getchar();
	#endif //VERBOSE_OUTPUT
	
    // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    Epetra_SerialDenseVector glstrain(NUMSTR_SOCTET10);
    glstrain(0) = 0.5 * (cauchygreen(0,0) - 1.0);
    glstrain(1) = 0.5 * (cauchygreen(1,1) - 1.0);
    glstrain(2) = 0.5 * (cauchygreen(2,2) - 1.0);
    glstrain(3) = cauchygreen(0,1);
    glstrain(4) = cauchygreen(1,2);
    glstrain(5) = cauchygreen(2,0);
    
    #ifdef VERBOSE_OUTPUT
	cout << "glstrain\n" << glstrain;
	#endif //VERBOSE_OUTPUT
	
	/*----------------------------------------------------------------------*
      the B-operator used is equivalent to the one used in hex8, this needs
      to be checked if it is ok, but from the mathematics point of view, the only
      thing that needed to be changed is tho NUMDOF       vlf 07/07
      ----------------------------------------------------------------------*/
    /* non-linear B-operator (may be so called, meaning 
    ** of B-operator is not so sharp in the non-linear realm) *
    ** B = F . Bl *
    **
    **      [ ... | F_11*L_aj_{,1}^k  F_21*L_aj_{,1}^k  F_31*L_aj_{,1}^k | ... ]
    **      [ ... | F_12*L_aj_{,2}^k  F_22*L_aj_{,2}^k  F_32*L_aj_{,2}^k | ... ]
    **      [ ... | F_13*L_aj_{,3}^k  F_23*L_aj_{,3}^k  F_33*L_aj_{,3}^k | ... ]
    ** B =  [ ~~~    ~~~~~~~~~~~~~     ~~~~~~~~~~~~~     ~~~~~~~~~~~~~     ~~~ ]
    **      [       F_11*L_aj_{,2}^k+F_12*L_aj_{,1}^k                          ]
    **      [ ... |           F_21*L_aj_{,2}^k+F_22*L_aj_{,1}^k          | ... ]
    **      [                          F_31*L_aj_{,2}^k+F_32*L_aj_{,1}^k       ]
    **      [                                                                  ]
    **      [       F_12*L_aj_{,3}^k+F_13*L_aj_{,2}^k                          ]
    **      [ ... |           F_22*L_aj_{,3}^k+F_23*L_aj_{,2}^k          | ... ]
    **      [                          F_32*L_aj_{,3}^k+F_33*L_aj_{,2}^k       ]
    **      [                                                                  ]
    **      [       F_13*L_aj_{,1}^k+F_11*L_aj_{,3}^k                          ]
    **      [ ... |           F_23*L_aj_{,1}^k+F_21*L_aj_{,3}^k          | ... ]
    **      [                          F_33*L_aj_{,1}^k+F_31*L_aj_{,3}^k       ]
    */
    
    Epetra_SerialDenseMatrix bop(NUMSTR_SOCTET10,NUMDOF_SOCTET10);
    #ifdef VERBOSE_OUTPUT
    cout << bop;
    cout << defgrd;
    cout << N_XYZ;
    #endif //VERBOSE_OUTPUT
    
    for (int numnode=0; numnode<NUMNOD_SOTET10; numnode++) {
    	for (int numdof=0; numdof<NODDOF_SOTET10; numdof++) {
      	bop(0,NODDOF_SOTET10*numnode+numdof) = defgrd(numdof,0)*L_aj_int.deriv_gp[gp](numnode,0);
      	bop(1,NODDOF_SOTET10*numnode+numdof) = defgrd(numdof,1)*L_aj_int.deriv_gp[gp](numnode,1);
      	bop(2,NODDOF_SOTET10*numnode+numdof) = defgrd(numdof,2)*L_aj_int.deriv_gp[gp](numnode,2);
      	bop(3,NODDOF_SOTET10*numnode+numdof) = defgrd(numdof,0)*L_aj_int.deriv_gp[gp](numnode,1) + \
      			    						   defgrd(numdof,1)*L_aj_int.deriv_gp[gp](numnode,0);
      	bop(4,NODDOF_SOTET10*numnode+numdof) = defgrd(numdof,1)*L_aj_int.deriv_gp[gp](numnode,2) + \
      										   defgrd(numdof,2)*L_aj_int.deriv_gp[gp](numnode,1);
      	bop(5,NODDOF_SOCTET10*numnode+numdof) = defgrd(numdof,2)*L_aj_int.deriv_gp[gp](numnode,0) + \
      										   defgrd(numdof,0)*L_aj_int.deriv_gp[gp](numnode,2);
    	}
    }
    
  	#ifdef VERBOSE_OUTPUT
	cout << "bop\n" << bop;
	#endif //VERBOSE_OUTPUT
	
    /* call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ** Here all possible material laws need to be incorporated,
    ** the stress vector, a C-matrix, and a density must be retrieved,
    ** every necessary data must be passed.
    */
	
    Epetra_SerialDenseMatrix cmat(NUMSTR_SOCTET10,NUMSTR_SOCTET10);
    Epetra_SerialDenseVector stress(NUMSTR_SOCTET10);
    double density;
    so_ctet10_mat_sel(&stress,&cmat,&density,&glstrain, &defgrd, gp);
    
    if (elestress != NULL){
      for (int i = 0; i < NUMSTR_SOCTET10; ++i) {
        (*elestress)(gp,i) = stress(i);
      }
    }
    
 	#ifdef VERBOSE_OUTPUT    
    cout << "material input\n";
   	#endif //VERBOSE_OUTPUT
    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc
    
	if (force != NULL && stiffmatrix != NULL){	
	#ifdef VERBOSE_OUTPUT    
    cout << "material input\n";
   	#endif //VERBOSE_OUTPUT
 
    // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)   
    (*force).Multiply('T','N', L_aj_int.weights(gp) ,bop,stress,1.0);
    
    // integrate `elastic' and `initial-displacement' stiffness matrix
    // keu = keu + (B^T . C . B) * detJ * w(gp)
    Epetra_SerialDenseMatrix cb(NUMSTR_SOCTET10,NUMDOF_SOCTET10);
    cb.Multiply('N','N',1.0,cmat,bop,0.0);          // temporary C . B
    stiffmatrix->Multiply('T','N', L_aj_int.weights(gp),bop,cb,1.0);

    // intergrate `geometric' stiffness matrix and add to keu *****************
    
    Epetra_SerialDenseVector sfac(stress); // auxiliary integrated stress
    sfac.Scale(L_aj_int.weights(gp));     // w(gp) * [S11,S22,S33,S12=S21,S23=S32,S13=S31]
    vector<double> SmB_L(NUMDIM_SOCTET10);     // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
    for (int inod=0; inod<NUMNOD_SOCTET10; ++inod){
      SmB_L[0] = sfac(0) * L_aj_int.deriv_gp[gp](inod,0) +\
      		sfac(3) * L_aj_int.deriv_gp[gp](inod,1) + sfac(5) * L_aj_int.deriv_gp[gp](inod,2);
      SmB_L[1] = sfac(3) * L_aj_int.deriv_gp[gp](inod,0) +\
      		sfac(1) * L_aj_int.deriv_gp[gp](inod,1) + sfac(4) * L_aj_int.deriv_gp[gp](inod,2);
      SmB_L[2] = sfac(5) * L_aj_int.deriv_gp[gp](inod,0) +\
      		sfac(4) * L_aj_int.deriv_gp[gp](inod,1) + sfac(2) * L_aj_int.deriv_gp[gp](inod,2);
      for (int jnod=0; jnod<NUMNOD_SOCTET10; ++jnod){
        double bopstrbop = 0.0;            // intermediate value
        for (int idim=0; idim<NUMDIM_SOCTET10; ++idim) bopstrbop += L_aj_int.deriv_gp[gp](jnod,idim)*\
        															SmB_L[idim];
        (*stiffmatrix)(NUMDIM_SOCTET10*inod+0,NUMDIM_SOCTET10*jnod+0) += bopstrbop;
        (*stiffmatrix)(NUMDIM_SOCTET10*inod+1,NUMDIM_SOCTET10*jnod+1) += bopstrbop;
        (*stiffmatrix)(NUMDIM_SOCTET10*inod+2,NUMDIM_SOCTET10*jnod+2) += bopstrbop;
      }
    } // end of intergrate `geometric' stiffness ******************************
	}
 
   /* =========================================================================*/
  }/* ==================================================== end of Loop over GP */
   /* =========================================================================*/
   
  /*
   * Lumped Mass Matrix, due to Composite formulation
   */
#if 0 // find density from material
  Epetra_SerialDenseMatrix jac_coord(NUMCOORD_SUBTET4,NUMCOORD_SUBTET4);
  for (int i=0; i<4; i++)  jac_coord(0,i)=1;
  for (int row=0;row<3;row++)
  {
   	for (int col=0;col<4;col++)
       jac_coord(row+1,col) = xrefe(row,col);	
  }
   
  double volume=det_volf(jac_coord)/(double) 6;    //volume of a tetrahedra
  //not needed (redundant check)
  
  for (i=0 ; i< 4; i ++)
  {
  		double massfactor= density * volume / 32.0;
  		(*massmatrix)(NODDOF_SOCTET10*i+0,NUMDIM_SOCTET10*i+0)= massfactor;
  		(*massmatrix)(NODDOF_SOCTET10*i+1,NUMDIM_SOCTET10*i+1)= massfactor;
  		(*massmatrix)(NODDOF_SOCTET10*i+2,NUMDIM_SOCTET10*i+2)= massfactor;
  }
  
  for (i=4 ; i< NUMNOD_SOCTET10; i ++)
  {
  		double massfactor= density * volume * 7.0 /48.0;
  		(*massmatrix)(NODDOF_SOCTET10*i+0,NUMDIM_SOCTET10*i+0)= massfactor;
  		(*massmatrix)(NODDOF_SOCTET10*i+1,NUMDIM_SOCTET10*i+1)= massfactor;
  		(*massmatrix)(NODDOF_SOCTET10*i+2,NUMDIM_SOCTET10*i+2)= massfactor;
  }
#endif
 
  #ifdef VERBOSE_OUTPUT    
  cout << (*stiffmatrix);
  #endif //VERBOSE_OUTPUT
  return;
} // DRT::ELEMENTS::So_ctet10::so_tet10_nlnstiffmass


/*----------------------------------------------------------------------*
 |  lump mass matrix (private)                               bborn 07/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_ctet10::so_ctet10_lumpmass(Epetra_SerialDenseMatrix* emass)
{
  // lump mass matrix
  if (emass != NULL)
  {
    // we assume #elemat2 is a square matrix
    for (int c=0; c<(*emass).N(); ++c)  // parse columns
    {
      double d = 0.0;  
      for (int r=0; r<(*emass).M(); ++r)  // parse rows
      {
        d += (*emass)(r,c);  // accumulate row entries
        (*emass)(r,c) = 0.0;
      }
      (*emass)(c,c) = d;  // apply sum of row entries on diagonal
    }
  }
}


//should be changed if distorted refernce configurations are used
double DRT::ELEMENTS::So_ctet10::ctet10_midpoint(int coord)
{
	long double midpoint=0;
	for (int nd_num=0;nd_num<4;nd_num++)
		midpoint+= (long double) (Nodes()[nd_num]->X()[coord]- Nodes()[0]->X()[coord]);
	return midpoint/(long double) (4.0);		
}

DRT::ELEMENTS::So_ctet10::SUB_NODE::SUB_NODE()
{
	//nothing to do
}

void DRT::ELEMENTS::So_ctet10::SUB_NODE::init(
	const int in_local_id,
	const int in_global_id,
	const Epetra_SerialDenseMatrix& xrefe)
{	
	local_id =in_local_id;
  	global_id=in_global_id;
  	
  	my_x[0]=xrefe(global_id,0);
	my_x[1]=xrefe(global_id,1);
	my_x[2]=xrefe(global_id,2);
}

DRT::ELEMENTS::So_ctet10::SUB_NODE::~SUB_NODE()
{
	//nothing to do
}


DRT::ELEMENTS::So_ctet10::TET4_SUB::TET4_SUB()
{
	//nothing to do
}

void DRT::ELEMENTS::So_ctet10::TET4_SUB::init(
	const int& node1,
	const int& node2,
	const int& node3,
	const int& node4,
	const Epetra_SerialDenseMatrix& xrefe,
  	const double& xi_babicentre1, const double& xi_babicentre2,
  	const double& xi_babicentre3, const double& xi_babicentre4)
{
	my_nodes[0].init(0,node1,xrefe);
	my_nodes[1].init(1,node2,xrefe);
	my_nodes[2].init(2,node3,xrefe);
	my_nodes[3].init(3,node4,xrefe);
	global_baricentre_xi[0]=xi_babicentre1;
	global_baricentre_xi[1]=xi_babicentre2;
	global_baricentre_xi[2]=xi_babicentre3;
	global_baricentre_xi[3]=xi_babicentre4;
}



void DRT::ELEMENTS::So_ctet10::TET4_SUB::integrate(
    Epetra_SerialDenseVector& xi_c,
    Epetra_SerialDenseMatrix& in_Naj)

{
	const DRT::ELEMENTS::Integrator_tet4_1point tet4_lin_int;
	//set gp = 0 because Tet4_integrator_4point has only one gp, so gp loop is obsolete
	
	const int gp = 0;
	
	Epetra_SerialDenseMatrix xrefe(4,3);
	for (int i=0; i<NUMNOD_SUBTET4; ++i){
    	xrefe(i,0) = (my_nodes[i]).my_x[0];
    	xrefe(i,1) = (my_nodes[i]).my_x[1];
    	xrefe(i,2) = (my_nodes[i]).my_x[2];
	}
	
	/* compute the Jacobian matrix which looks like this:
    **         [  1        1        1  	     1      ]
    **   jac = [ X_,xsi1  X_,xsi2  X_,xsi3  X_,xsi4 ]
    **         [ Y_,xsi1  Y_,xsi2  Y_,xsi3  Y_,xsi4 ]
    **		   [ Z_,xsi1  Z_,xsi2  Z_,xsi3  Z_,xsi4 ]
    */
    
    Epetra_SerialDenseMatrix jac_temp(NUMCOORD_SUBTET4-1,NUMCOORD_SUBTET4);
    Epetra_SerialDenseMatrix jac(NUMCOORD_SUBTET4,NUMCOORD_SUBTET4);
    jac_temp.Multiply('T','N',1.0,xrefe,tet4_lin_int.deriv_gp[gp],0.0);
   
    for (int i=0; i<4; i++) jac(0,i)=1;
    for (int row=0;row<3;row++)
    {
    	for (int col=0;col<4;col++)
    	jac(row+1,col)=jac_temp(row,col);	
    }

    /* compute partial derivatives at gp xsi_1, xsi_2, xsi_3, xsi_4 material coordinates
    ** by solving   Jac . partials = I_aug   for partials
    ** Inverse of Jacobian is therefore not explicitly computed
    */ 
    /* structure of partials:
    **             [  dxsi_1   dxsi_1   dxsi_1  ]
    **             [  ------   ------   ------  ]
    **             [    dX       dY       dZ    ]
    **             [     |        |        |    ]
    ** partials =  [                            ]
    **             [  dxsi_4   dxsi_4   dxsi_4  ]
    **             [  ------   ------   ------  ]
    **             [    dX       dY       dZ    ]
    */   
    
    Epetra_SerialDenseMatrix I_aug(NUMCOORD_SUBTET4,NUMDIM_SUBTET4);
    Epetra_SerialDenseMatrix partials(NUMCOORD_SUBTET4,NUMDIM_SUBTET4);
    Epetra_SerialDenseMatrix N_XYZ(NUMNOD_SUBTET4,NUMDIM_SUBTET4);
    I_aug(1,0)=1;
	I_aug(2,1)=1;
	I_aug(3,2)=1;
	
    Epetra_SerialDenseSolver solve_for_inverseJac;  // solve A.X=B
    solve_for_inverseJac.SetMatrix(jac);            // set A=jac
    solve_for_inverseJac.SetVectors(partials,I_aug);// set X=partials, B=I_aug
    solve_for_inverseJac.FactorWithEquilibration(true);
    int err2 = solve_for_inverseJac.Factor();        
    int err = solve_for_inverseJac.Solve();         // partials = jac^-1.I_aug
    if ((err != 0) && (err2!=0)){
    	dserror("Inversion of Jacobian failed");
    }

    N_XYZ.Multiply('N','N',my_volume(),tet4_lin_int.deriv_gp[gp],partials,0.0); //N_XYZ = N_xsi_k*partials
    //static Epetra_SerialDenseMatrix N_aJ(11,NUMDIM_SUBTET4);
    //N_aJ.Shape(11,NUMDIM_SUBTET4);
	in_Naj.Shape(11,NUMDIM_SUBTET4);
    for (int node = 0; node < NUMNOD_SUBTET4; node++)
    {
    	for (int J = 0; J < NUMDIM_SUBTET4; J++)
    	{
    		in_Naj((my_nodes[node]).global_id , J) = N_XYZ(node,J);
    	}
    }
   
    for (int node = 4; node < NUMNOD_SOCTET10; node++)
    {
    	for (int J = 0; J < NUMDIM_SUBTET4; J++)
    	{
    		in_Naj(node, J) += in_Naj(10, J)/6.0;
    	}
    }
    
    for (int i=0; i<4 ;i++)
    {
    	xi_c[i]= global_baricentre_xi[i];
    }
}

double DRT::ELEMENTS::So_ctet10::TET4_SUB::my_volume()
{
  Epetra_SerialDenseMatrix jac_coord(NUMCOORD_SUBTET4,NUMCOORD_SUBTET4);
  for (int i=0; i<4; i++)  jac_coord(0,i)=1;
  for (int row=0;row<3;row++)
  {
   	for (int col=0;col<4;col++)
       jac_coord(row+1,col) = (my_nodes[col]).my_x[row];	
  }
   
  double detJ=(double)det_volf(jac_coord)/(double) 6;    //volume of a tetrahedra
  if (detJ == 0.0) dserror("ZERO JACOBIAN DETERMINANT");
  else if (detJ < 0.0) dserror("NEGATIVE JACOBIAN DETERMINANT");
  return detJ;  
} 


DRT::ELEMENTS::So_ctet10::TET4_SUB::~TET4_SUB()
{
	//nothing to do
}

DRT::ELEMENTS::So_ctet10::SUB_STRUCTURE::SUB_STRUCTURE(const Epetra_SerialDenseMatrix& xrefe)
{
	// sub elements conectivity
	my_elements[ 0].init( 0, 4, 6, 7,xrefe, double(5)/8.0,  double(1)/8.0,  double(1)/8.0,  double(1)/8.0);
	my_elements[ 1].init( 1, 5, 4, 8,xrefe, double(1)/8.0,  double(5)/8.0,  double(1)/8.0,  double(1)/8.0);
	my_elements[ 2].init( 2, 6, 5, 9,xrefe, double(1)/8.0,  double(1)/8.0,  double(5)/8.0,  double(1)/8.0);
	my_elements[ 3].init( 3, 8, 7, 9,xrefe, double(1)/8.0,  double(1)/8.0,  double(1)/8.0,  double(5)/8.0);
	my_elements[ 4].init( 4, 8, 5,10,xrefe, double(3)/16.0, double(7)/16.0, double(3)/16, double(3)/16.0);
	my_elements[ 5].init( 5, 8, 9,10,xrefe, double(1)/16.0, double(5)/16.0, double(5)/16, double(5)/16.0);
	my_elements[ 6].init( 9, 8, 7,10,xrefe, double(3)/16.0, double(3)/16.0, double(3)/16, double(7)/16.0);
	my_elements[ 7].init( 7, 8, 4,10,xrefe, double(5)/16.0, double(5)/16.0, double(1)/16, double(5)/16.0);
	my_elements[ 8].init( 4, 5, 6,10,xrefe, double(5)/16.0, double(5)/16.0, double(5)/16, double(1)/16.0);
	my_elements[ 9].init( 5, 9, 6,10,xrefe, double(3)/16.0, double(3)/16.0, double(7)/16, double(3)/16.0);
	my_elements[10].init( 9, 7, 6,10,xrefe, double(5)/16.0, double(1)/16.0, double(5)/16, double(5)/16.0);
	my_elements[11].init( 7, 4, 6,10,xrefe, double(7)/16.0, double(3)/16.0, double(3)/16, double(3)/16.0);
}

/*----------------------------------------------------------------------*
 | constructor for a integrator class              			  volf 09/07|
 | uses shape functions of a quadratic tetrahedra using so-called       |
 | "natural coordinates" as described by Carlos A. Felippa in Adv. FEM  |
 | Aerospace Engineering Sciences - University of Colorado at Boulder   |
 *----------------------------------------------------------------------*/
 
DRT::ELEMENTS::So_ctet10::Tet10c_integrator_5point::Tet10c_integrator_5point(SUB_STRUCTURE& sub_struct)
{
  num_gp = 5;
  num_nodes = 10;
  num_coords = NUMCOORD_SOCTET10;
  shapefct_gp.resize(NUMGPT_SOCTET10);
  //deriv_gp    = NULL;
  deriv_gp.resize(NUMGPT_SOCTET10);
  weights.Size(num_gp);
  
  //guadrature rule from M. Ortiz Tetrahedral composite finite elements
  
  // (xsi1, xsi2, xsi3 ,xsi4) gp-locations
  const double xsi1[NUMGPT_SOCTET10] = {double(1)/4.0, double(1)/2.0, double(1)/6.0, double(1)/6.0, double(1)/6.0};
  const double xsi2[NUMGPT_SOCTET10] = {double(1)/4.0, double(1)/6.0, double(1)/2.0, double(1)/6.0, double(1)/6.0};
  const double xsi3[NUMGPT_SOCTET10] = {double(1)/4.0, double(1)/6.0, double(1)/6.0, double(1)/2.0, double(1)/6.0};
  const double xsi4[NUMGPT_SOCTET10] = {double(1)/4.0, double(1)/6.0, double(1)/6.0, double(1)/6.0, double(1)/2.0};
  
  weights[0]= ( (sub_struct.my_elements[ 4]).my_volume()    \
  	          + (sub_struct.my_elements[ 6]).my_volume()    \
  	          + (sub_struct.my_elements[ 9]).my_volume()    \
  	          + (sub_struct.my_elements[11]).my_volume() )/2.0\
  	          + (sub_struct.my_elements[ 5]).my_volume()    \
  	          + (sub_struct.my_elements[ 7]).my_volume()    \
  	          + (sub_struct.my_elements[ 8]).my_volume()    \
              + (sub_struct.my_elements[10]).my_volume();
  weights[1]=   sub_struct.my_elements[ 0].my_volume()    \
              + sub_struct.my_elements[11].my_volume()/2.0;
  weights[2]=   sub_struct.my_elements[ 1].my_volume()    \
              + sub_struct.my_elements[ 4].my_volume()/2.0;
  weights[3]=   sub_struct.my_elements[ 2].my_volume()    \
              + sub_struct.my_elements[ 9].my_volume()/2.0;
  weights[4]=   sub_struct.my_elements[ 3].my_volume()    \
              + sub_struct.my_elements[ 6].my_volume()/2.0;
              
   for (int gp=0; gp<num_gp; gp++) {
   	  //different than tet10
      (shapefct_gp[gp]).Size(num_coords);
      (shapefct_gp[gp])[0] = xsi1[gp];
      (shapefct_gp[gp])[1] = xsi2[gp];
      (shapefct_gp[gp])[2] = xsi3[gp];
      (shapefct_gp[gp])[3] = xsi4[gp];
  }  
}


/*----------------------------------------------------------------------*
 | constructor for a integrator class              			   vlf 05/08|
 | Evaluates the L_aj matrix that replaces the N_,X matrices known from |
 | other formualtions. Additionally agregates the information for       | 
 | integrating the stiffness matrix.                                    |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_ctet10::L_AJ_integrator::L_AJ_integrator(SUB_STRUCTURE& sub_struct)
{
	num_gp      = NUMGPT_SOCTET10;
    num_nodes   = 10;
    num_coords  =  3; 
    shapefct_gp.resize(NUMGPT_SOCTET10);//here derivatives N_,X instead of N_,xi are used
    deriv_gp.resize(NUMGPT_SOCTET10);
    weights.Size(NUMGPT_SOCTET10);
	
	Tet10c_integrator_5point Mbc_integrator(sub_struct);
	Epetra_SerialDenseMatrix Mbc(4,4);
	
	// compute M_bc^-1
	for(int b = 0; b < 4; b ++)
	  for(int c = 0; c < 4; c ++)
	    for(int gp = 0; gp < Mbc_integrator.num_gp ; gp ++)
	    {
	    	Mbc(b,c)+= (Mbc_integrator.shapefct_gp[gp])[b] * (Mbc_integrator.shapefct_gp[gp])[c]\
	    	           * (Mbc_integrator.weights)[gp];
	    }	 

	Mbc_inv.Reshape(4,4);
	Epetra_SerialDenseMatrix I_matrix(4,4);
	for (int i = 0;i < 4; i++) I_matrix(i,i)=1;

	Epetra_SerialDenseSolver solve_for_Mbc_inv;  // solve A.X=B
    solve_for_Mbc_inv.SetMatrix(Mbc);            // set A=Mbc
    solve_for_Mbc_inv.SetVectors(Mbc_inv,I_matrix);// set X=Mbc_inv, B=I_matrix
    solve_for_Mbc_inv.FactorWithEquilibration(true);
    int err2 = solve_for_Mbc_inv.Factor();        
    int err = solve_for_Mbc_inv.Solve();         // Mbc_inv = Mbc^-1.I_matrix
    if ((err != 0) && (err2!=0)){
    	dserror("Inversion of Mbc");
    }

	
	//compute L_aj on each GP
	for (int gp = 0;gp < NUMGPT_SOCTET10; gp++)
	{
		deriv_gp[gp].Shape(num_nodes,num_coords);
		
		// integrate over sub elements
		for (int sub_num = 0 ;sub_num < 12 ; sub_num++)
		{
			Epetra_SerialDenseVector my_xi_b(4);
            Epetra_SerialDenseMatrix my_N_aJ(num_nodes+1,num_coords);
            
            // call  integration routine for the linear sub element
            // & map the integrated functions back to the global element
            (sub_struct.my_elements[sub_num]).integrate(my_xi_b,my_N_aJ);
		    my_N_aJ.Reshape(num_nodes,num_coords);
		    Epetra_SerialDenseVector temp_vector(4);
			temp_vector.Multiply('N','N', 1, Mbc_inv, my_xi_b, 0);

			double temp_scale = temp_vector.Dot(Mbc_integrator.shapefct_gp[gp]);
			my_N_aJ.Scale(temp_scale);
			deriv_gp[gp]+= my_N_aJ;
		}		
	}  
//------------
	for (int gp= 0; gp< num_gp ; gp++)
	{
		weights(gp)= (Mbc_integrator.weights)[gp];
	}
}


/*----------------------------------------------------------------------*
 | constructor for a integrator class              			   vlf 05/08|
 | Evaluates the L_aj matrix that replaces the N_,X matrices known from |
 | other formualtions. These are evaluated at nodes in order to obtain  | 
 | the stresses at the nodes of the element. This is a consistent       |
 | evaluation of these stresses                                         |
 *----------------------------------------------------------------------*/

DRT::ELEMENTS::So_ctet10::\
L_AJ_stress_integrator::L_AJ_stress_integrator(SUB_STRUCTURE& sub_struct)
{
	num_gp      = 10;
    num_nodes   = 10;
    num_coords  =  3; //here dervatives N_,X instead of N_,xi are used
    shapefct_gp.resize(NUMNOD_SOCTET10);
    deriv_gp.resize(NUMNOD_SOCTET10);
    weights.Size(NUMNOD_SOCTET10);
	
	// compute M_bc^-1
	Tet10c_integrator_5point Mbc_integrator(sub_struct);
	Epetra_SerialDenseMatrix Mbc(4,4);
	
	for(int b = 0; b < 4; b ++)
	  for(int c = 0; c < 4; c ++)
	    for(int gp = 0; gp < Mbc_integrator.num_gp ; gp ++)
	    {
	    	Mbc(b,c)+= (Mbc_integrator.shapefct_gp[gp])[b] * (Mbc_integrator.shapefct_gp[gp])[c]\
	    	           * (Mbc_integrator.weights)[gp];
	    }	 

	Epetra_SerialDenseMatrix Mbc_inv(4,4);
	Epetra_SerialDenseMatrix I_matrix(4,4);
	for (int i = 0;i < 4; i++)I_matrix(i,i)=1;

	Epetra_SerialDenseSolver solve_for_Mbc_inv;  // solve A.X=B
    solve_for_Mbc_inv.SetMatrix(Mbc);            // set A=Mbc)
    solve_for_Mbc_inv.SetVectors(Mbc_inv,I_matrix);// set X=Mbc_inv, B=I_matrix
    solve_for_Mbc_inv.FactorWithEquilibration(true);
    int err2 = solve_for_Mbc_inv.Factor();        
    int err = solve_for_Mbc_inv.Solve();         // Mbc_inv = Mbc^-1.I_matrix
    if ((err != 0) && (err2!=0)){
    	dserror("Inversion of Mbc");
    }

	//use the tet10 node-based integrator, it contains
	//nodal coordinates at nodes for a 10-node tetrahedral
	Integrator_tet10_10node _10node_integrator;
	// integrate for each node
	for (int gp = 0;gp < _10node_integrator.num_gp; gp++)
	{
		deriv_gp[gp].Shape(num_nodes,num_coords);
		
		// integrate over sub elements
		for (int sub_num = 0 ;sub_num < 12 ; sub_num++)
		{
			Epetra_SerialDenseVector my_xi_b(4);
            Epetra_SerialDenseMatrix my_N_aJ(num_nodes+1,num_coords);
            
            // call  integration routine for the linear sub element
            // & map the integrated functions back to the global element
            (sub_struct.my_elements[sub_num]).integrate(my_xi_b,my_N_aJ);
		    my_N_aJ.Reshape(num_nodes,num_coords);
		    Epetra_SerialDenseVector temp_vector(4);
			temp_vector.Multiply('N','N', 1 , Mbc_inv, my_xi_b,0);
			
			double temp_scale = temp_vector.Dot(_10node_integrator.shapefct_gp_lin[gp]);
			my_N_aJ.Scale(temp_scale);
			deriv_gp[gp]+= my_N_aJ;
		}		
	}  
//------------
	for (int gp= 0; gp< num_gp ; gp++)
	{
		weights(gp)= (_10node_integrator.weights)[gp];
	}
}

DRT::ELEMENTS::So_ctet10::SUB_STRUCTURE::~SUB_STRUCTURE()
{
	//nothing to do
}


int DRT::ELEMENTS::Soctet10Register::Initialize(DRT::Discretization& dis)
{
  return 0;
}

#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_SOLID3
