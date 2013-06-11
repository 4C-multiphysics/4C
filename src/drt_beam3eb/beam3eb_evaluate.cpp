/*!----------------------------------------------------------------------
\file beam3eb.H

\brief three dimensional nonlinear torsionless rod based on a C1 curve

<pre>
Maintainer: Christoph Meier
            meier@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15301
</pre>

 *-----------------------------------------------------------------------------------------------------------*/

#include "beam3eb.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_exporter.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_utils.H"
#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_timecurve.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
#include "../drt_mat/stvenantkirchhoff.H"
#include "../linalg/linalg_fixedsizematrix.H"
#include "../drt_fem_general/largerotations.H"
#include "../drt_fem_general/drt_utils_integration.H"
#include "../drt_inpar/inpar_structure.H"
#include <Epetra_CrsMatrix.h>

#include "Sacado.hpp"
typedef Sacado::Fad::DFad<double> FAD;

/*-----------------------------------------------------------------------------------------------------------*
 |  evaluate the element (public)                                                                 meier 05/12|
 *----------------------------------------------------------------------------------------------------------*/
int DRT::ELEMENTS::Beam3eb::Evaluate(Teuchos::ParameterList& params,
                                     DRT::Discretization& discretization,
                                     std::vector<int>& lm,
                                     Epetra_SerialDenseMatrix& elemat1,
                                     Epetra_SerialDenseMatrix& elemat2,
                                     Epetra_SerialDenseVector& elevec1,
                                     Epetra_SerialDenseVector& elevec2,
                                     Epetra_SerialDenseVector& elevec3)
{

  DRT::ELEMENTS::Beam3eb::ActionType act = Beam3eb::calc_none;
  // get the action required
  std::string action = params.get<std::string>("action","calc_none");

  if 	  (action == "calc_none") 				dserror("No action supplied");
  else if (action=="calc_struct_linstiff") 		act = Beam3eb::calc_struct_linstiff;
  else if (action=="calc_struct_nlnstiff") 		act = Beam3eb::calc_struct_nlnstiff;
  else if (action=="calc_struct_internalforce") act = Beam3eb::calc_struct_internalforce;
  else if (action=="calc_struct_linstiffmass") 	act = Beam3eb::calc_struct_linstiffmass;
  else if (action=="calc_struct_nlnstiffmass") 	act = Beam3eb::calc_struct_nlnstiffmass;
  else if (action=="calc_struct_nlnstifflmass") act = Beam3eb::calc_struct_nlnstifflmass; //with lumped mass matrix
  else if (action=="calc_struct_stress") 		act = Beam3eb::calc_struct_stress;
  else if (action=="calc_struct_eleload") 		act = Beam3eb::calc_struct_eleload;
  else if (action=="calc_struct_fsiload") 		act = Beam3eb::calc_struct_fsiload;
  else if (action=="calc_struct_update_istep")  act = Beam3eb::calc_struct_update_istep;
  else if (action=="calc_struct_reset_istep")   act = Beam3eb::calc_struct_reset_istep;
  else if (action=="calc_struct_ptcstiff")		act = Beam3eb::calc_struct_ptcstiff;
  else 	  dserror("Unknown type of action for Beam3eb");

  std::string test = params.get<std::string>("action","calc_none");

  switch(act)
  {

    case Beam3eb::calc_struct_ptcstiff:
    {
      dserror("no ptc implemented for Beam3eb element");
    }
    break;

    case Beam3eb::calc_struct_linstiff:
    {
      //only nonlinear case implemented!
      dserror("linear stiffness matrix called, but not implemented");
    }
    break;

    //nonlinear stiffness and mass matrix are calculated even if only nonlinear stiffness matrix is required
    case Beam3eb::calc_struct_nlnstiffmass:
    case Beam3eb::calc_struct_nlnstifflmass:
    case Beam3eb::calc_struct_nlnstiff:
    case Beam3eb::calc_struct_internalforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacement and residual values for each degree of freedom

      //Uncomment the following line to calculate the entire problem with arbitrary precision
      #ifdef PRECISION
      {
      HighPrecissionCalc();
      }
      #endif

      // get element displacements
      RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp==Teuchos::null) dserror("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);

      // get residual displacements
      RCP<const Epetra_Vector> res  = discretization.GetState("residual displacement");
      if (res==Teuchos::null) dserror("Cannot get state vectors 'residual displacement'");
      std::vector<double> myres(lm.size());
      DRT::UTILS::ExtractMyValues(*res,myres,lm);


      //TODO: Only in the dynamic case the velocities are needed.
      // get element velocities
      std::vector<double> myvel(lm.size());

      const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();

      if(DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP")!=INPAR::STR::dyna_statics)
      {
        RCP<const Epetra_Vector> vel  = discretization.GetState("velocity");
        if (vel==Teuchos::null) dserror("Cannot get state vectors 'velocity'");
        DRT::UTILS::ExtractMyValues(*vel,myvel,lm);
      }
      if (act == Beam3eb::calc_struct_nlnstiffmass)
      {
			eb_nlnstiffmass(params,myvel,mydisp,&elemat1,&elemat2,&elevec1);
      }
      else if (act == Beam3eb::calc_struct_nlnstifflmass)
      {
  	  		eb_nlnstiffmass(params,myvel,mydisp,&elemat1,&elemat2,&elevec1);
  	  		lumpmass(&elemat2);
      }
      else if (act == Beam3eb::calc_struct_nlnstiff)
      {
  	  		eb_nlnstiffmass(params,myvel,mydisp,&elemat1,NULL,&elevec1);
      }
      else if (act == Beam3eb::calc_struct_internalforce)
      {
  	  		eb_nlnstiffmass(params,myvel,mydisp,NULL,NULL,&elevec1);
      }
    }
    break;

    case calc_struct_stress:
    	dserror("No stress output implemented for beam3 elements");
    break;
    case calc_struct_update_istep:
    	//not necessary since no class variables are modified in predicting steps
    break;
    case calc_struct_reset_istep:
    	//not necessary since no class variables are modified in predicting steps
    break;

    default:
      dserror("Unknown type of action for Beam3eb %d", act);
     break;
  }//switch(act)

  return 0;

}	//DRT::ELEMENTS::Beam3eb::Evaluate

/*-----------------------------------------------------------------------------------------------------------*
 |  Integrate a Surface/Line Neumann boundary condition (public)                                  meier 05/12|
 *-----------------------------------------------------------------------------------------------------------*/

int DRT::ELEMENTS::Beam3eb::EvaluateNeumann(Teuchos::ParameterList& params,
                                            DRT::Discretization& discretization,
                                            DRT::Condition& condition,
                                            std::vector<int>& lm,
                                            Epetra_SerialDenseVector& elevec1,
                                            Epetra_SerialDenseMatrix* elemat1)
{
  // get element displacements
  RCP<const Epetra_Vector> disp = discretization.GetState("displacement new");
  if (disp==Teuchos::null) dserror("Cannot get state vector 'displacement new'");
  std::vector<double> mydisp(lm.size());
  DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);

  const int dofpn = 3*NODALDOFS;
  const int nnodes = 2;

  for (int i=0; i<nnodes*dofpn;i++)
  {
    mydisp[i] = mydisp[i]*ScaleFactorColumn;
  }

  /*
  // get element velocities (UNCOMMENT IF NEEDED)
  RCP<const Epetra_Vector> vel  = discretization.GetState("velocity");
  if (vel==Teuchos::null) dserror("Cannot get state vectors 'velocity'");
  std::vector<double> myvel(lm.size());
  DRT::UTILS::ExtractMyValues(*vel,myvel,lm);
  */

  // find out whether we will use a time curve
  bool usetime = true;
  const double time = params.get("total time",-1.0);
  if (time<0.0) usetime = false;

  // find out whether we will use a time curve and get the factor
  const std::vector<int>* curve = condition.Get<std::vector<int> >("curve");

  int curvenum = -1;

  // number of the load curve related with a specific Neumann condition called
  if (curve) curvenum = (*curve)[0];

  // amplitude of load curve at current time called
  double curvefac = 1.0;

  if (curvenum>=0 && usetime)
    curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);

  // get values and switches from the condition
  // onoff is related to the first 6 flags of a line Neumann condition in the input file;
  // value 1 for flag i says that condition is active for i-th degree of freedom
  const std::vector<int>* onoff = condition.Get<std::vector<int> >("onoff");
  // val is related to the 6 "val" fields after the onoff flags of the Neumann condition

  // in the input file; val gives the values of the force as a multiple of the prescribed load curve
  const std::vector<double>* val = condition.Get<std::vector<double> >("val");

  // funct is related to the 6 "funct" fields after the val field of the Neumann condition
  // in the input file; funct gives the number of the function defined in the section FUNCT
  const std::vector<int>* functions = condition.Get<std::vector<int> >("funct");

  //find out which node is correct
  const std::vector< int > * nodeids = condition.Nodes();

  //if a point neumann condition needs to be linearized
  if(condition.Type() == DRT::Condition::PointNeumannEB)
  {
    //find out local element number --> this is done since the first element of a neumann point condition is used for this function
    //in this case we do not know whether it is the left or the right node.
    int insert = -1;

    if((*nodeids)[0] == Nodes()[0]->Id())
      insert = 0;
    else if((*nodeids)[0] == Nodes()[1]->Id())
      insert = 1;

    if (insert == -1)
      dserror("\nNode could not be found on nodemap!\n");

    //add forces to Res_external according to (5.56). There is a factor (-1) needed, as fext is multiplied by (-1) in BACI
    for(int i = 0; i < 3 ; i++)
    {
      elevec1(insert*dofpn + i) += (*onoff)[i]*(*val)[i]*curvefac*ScaleFactorLine;
    }

    //matrix for current tangent, moment at node and crossproduct
    LINALG::Matrix<3,1> tangent;
    LINALG::Matrix<3,1> crossproduct;
    LINALG::Matrix<3,1> moment;
    LINALG::Matrix<3,3> spinmatrix;

    //clear all matrices
    tangent.Clear();
    crossproduct.Clear();
    moment.Clear();
    spinmatrix.Clear();

    //assemble current tangent and moment at node
    for (int dof = 3 ; dof < 6 ; dof++)
    {
      //get current tangent at nodes
      tangent(dof-3) = Tref_[insert](dof-3) + mydisp[insert*dofpn + dof];
      moment(dof-3) = (*onoff)[dof]*(*val)[dof]*curvefac;
    }

    double abs_tangent = 0.0;

    //Res will be normalized with the length of the current tangent
    abs_tangent = tangent.Norm2();

    //computespin = S ( tangent ) using the spinmatrix in namespace largerotations
    LARGEROTATIONS::computespin(spinmatrix,tangent);

    //matrixoperation crossproduct = t x m
    for(int i=0; i<3; i++)
    {
      for(int j=0; j<3; j++)
      {
        crossproduct(i) += spinmatrix(i,j) * moment(j);
      }
    }

    //add moments to Res_external according to (5.56). There is a factor (-1) needed, as fext is multiplied by (-1) in BACI
    for(int i = 3; i < 6 ; i++)
    {
      elevec1(insert*dofpn + i) -= crossproduct(i-3) / pow(abs_tangent,2.0)*ScaleFactorLine;
    }

    //assembly for stiffnessmatrix
    LINALG::Matrix<3,3> crossxtangent;

    crossxtangent.Clear();

    //perform matrix operation
    for(int i=0; i<3; i++)
    {
      for(int j=0; j<3; j++)
      {
        crossxtangent(i,j) = crossproduct(i) * tangent(j);
      }
    }

    spinmatrix.Clear();

    //spinmatrix = S ( m )
    LARGEROTATIONS::computespin(spinmatrix,moment);

    //add R_external to stiffness matrix
    //all parts have been evaluated at the boundaries which helps simplifying the matrices
    //In contrast to the Neumann part of the residual force here is NOT a factor of (-1) needed, as elemat1 is directly added to the stiffness matrix
    //without sign change.
    double Factor = ScaleFactorLine;
    Factor = Factor * ScaleFactorColumn;
    for(int i = 3; i < 6 ; i++)
    {
      for(int j = 3; j < 6 ; j++)
      {
        (*elemat1)(insert*dofpn + i, insert*dofpn + j) -= 2.0 * crossxtangent(i-3,j-3) / pow(abs_tangent,4.0)*Factor;
        (*elemat1)(insert*dofpn + i, insert*dofpn + j) -= spinmatrix(i-3,j-3) / pow(abs_tangent,2.0)*Factor;
      }
    }

    bool precond = PreConditioning;
    if (precond)
    {
      double length = jacobi_*2.0;
      double radius = std::pow(crosssec_/M_PI,0.5);
      for (int zeile=0; zeile <2; zeile++)
      {
        for (int spalte=0; spalte<12; spalte++)
        {
          (*elemat1)(6*zeile,spalte)=(*elemat1)(6*zeile,spalte)*length;
          (*elemat1)(6*zeile+1,spalte)=(*elemat1)(6*zeile+1,spalte)*pow(length,3.0)/pow(radius,2.0);
          (*elemat1)(6*zeile+2,spalte)=(*elemat1)(6*zeile+2,spalte)*pow(length,3.0)/pow(radius,2.0);
          (*elemat1)(6*zeile+4,spalte)=(*elemat1)(6*zeile+4,spalte)*pow(length,2.0)/pow(radius,2.0);
          (*elemat1)(6*zeile+5,spalte)=(*elemat1)(6*zeile+5,spalte)*pow(length,2.0)/pow(radius,2.0);
        }
        elevec1(6*zeile)=elevec1(6*zeile)*length;
        elevec1(6*zeile+1)=elevec1(6*zeile+1)*pow(length,3.0)/pow(radius,2.0);
        elevec1(6*zeile+2)=elevec1(6*zeile+2)*pow(length,3.0)/pow(radius,2.0);
        elevec1(6*zeile+4)=elevec1(6*zeile+4)*pow(length,2.0)/pow(radius,2.0);
        elevec1(6*zeile+5)=elevec1(6*zeile+5)*pow(length,2.0)/pow(radius,2.0);
      }
    }

  }

  //if a line neumann condition needs to be linearized
  else if(condition.Type() == DRT::Condition::LineNeumann)
  {

    // gaussian points
    DRT::UTILS::IntegrationPoints1D gausspoints = DRT::UTILS::IntegrationPoints1D(DRT::UTILS::mygaussruleeb);
    LINALG::Matrix<1,NODALDOFS*nnodes> N_i;

    //integration loops
    for (int numgp=0; numgp<gausspoints.nquad; ++numgp)
    {

      //cout << "numgp: " << numgp + 1 << endl;
      //integration points in parameter space and weights
      const double xi = gausspoints.qxg[numgp][0];
      const double wgt = gausspoints.qwgt[numgp];

      //Get DiscretizationType of beam element
      const DRT::Element::DiscretizationType distype = Shape();

      //Clear matrix for shape functions
      N_i.Clear();

      //evaluation of shape funcitons at Gauss points
      #if (NODALDOFS == 2)
      //Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      DRT::UTILS::shape_function_hermite_1D(N_i,xi,jacobi_*2.0,distype);
      //end--------------------------------------------------------
      #elif (NODALDOFS == 3)
      //specific-for----------------------------------Frenet Serret
      //Get hermite derivatives N'xi, N''xi and N'''xi
      DRT::UTILS::shape_function_hermite_1D_order5(N_i,xi,jacobi_*2.0,distype);
      //end--------------------------------------------------------
      #else
      dserror("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
      #endif

      //position vector at the gauss point at reference configuration needed for function evaluation
      std::vector<double> X_ref(3,0.0);
      //calculate coordinates of corresponding Guass point in reference configuration
      for (int node=0;node<2;node++)
      {
        #if (NODALDOFS == 2)
        for (int dof=0;dof<3;dof++)
        {
          X_ref[dof]+=Nodes()[node]->X()[dof]*N_i(2*node)+Tref_[node](dof)*N_i(2*node + 1);
        }
        #elif (NODALDOFS ==3)
        for (int dof=0;dof<3;dof++)
        {
          X_ref[dof]+=Nodes()[node]->X()[dof]*N_i(3*node)+Tref_[node](dof)*N_i(3*node + 1) + Kref_[node](dof)*N_i(3*node + 2);
        }
        #else
        dserror("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
        #endif
      }

      double fac=0.0;
      fac = wgt * jacobi_;

      // load vector ar
      double ar[6];

      // loop the dofs of a node
      for (int dof=0; dof<6; ++dof)
        ar[dof] = fac * (*onoff)[dof]*(*val)[dof]*curvefac;
      double functionfac = 1.0;
      int functnum = -1;

      //Check if also moment line Neumann conditions are implemented accidentally and throw error
      for (int dof=3; dof<6; ++dof)
      {
        if (functions) functnum = (*functions)[dof];
        else functnum = -1;

        if (functnum>0)
        {
          dserror("Line Neumann conditions for distributed moments are not implemented for beam3eb so far! Only the function flag 1, 2 and 3 can be set!");
        }
      }

      //sum up load components
      for (int dof=0; dof<3; ++dof)
      {
        if (functions) functnum = (*functions)[dof];
        else functnum = -1;

        if (functnum>0)
        {
          // evaluate function at the position of the current node       --> dof here correct?
          functionfac = DRT::Problem::Instance()->Funct(functnum-1).Evaluate(dof, &X_ref[0], time, NULL);
        }
        else functionfac = 1.0;

        for (int node=0; node<2*NODALDOFS; ++node)
        {
          elevec1[node*3 + dof] += N_i(node) *ar[dof] *functionfac;
        }
      }
    } // for (int numgp=0; numgp<intpoints.nquad; ++numgp)

  }

  //Uncomment the next line if the implementation of the Neumann part of the analytical stiffness matrix should be checked by Forward Automatic Differentiation (FAD)
  //FADCheckNeumann(params, discretization, condition, lm, elevec1, elemat1);

  return 0;

}	//DRT::ELEMENTS::Beam3eb::EvaluateNeumann



/*------------------------------------------------------------------------------------------------------------*
 | nonlinear stiffness and mass matrix (private)                                                   meier 05/12|
 *-----------------------------------------------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3eb::eb_nlnstiffmass(Teuchos::ParameterList& params,
                                              std::vector<double>& vel,
                                              std::vector<double>& disp,
                                              Epetra_SerialDenseMatrix* stiffmatrix,
                                              Epetra_SerialDenseMatrix* massmatrix,
                                              Epetra_SerialDenseVector* force)
{


#ifdef SIMPLECALC
{
  //dimensions of freedom per node
  const int dofpn = 3*NODALDOFS;

  //number of nodes fixed for these element
  const int nnode = 2;

  //matrix for current positions and tangents
  std::vector<double> disp_totlag(nnode*dofpn,0.0);

  LINALG::Matrix<3,1> r_;
  LINALG::Matrix<3,1> r_x;
  LINALG::Matrix<3,1> r_xx;

  LINALG::Matrix<3,1> f1;
  LINALG::Matrix<3,1> f2;
  LINALG::Matrix<3,1> n1;

  double rxxrxx;
  double rxrx;
  double tension;

  LINALG::Matrix<dofpn*nnode,dofpn*nnode> NTildex;
  LINALG::Matrix<dofpn*nnode,dofpn*nnode> NTildexx;

  LINALG::Matrix<dofpn*nnode,1> NxTrx;
  LINALG::Matrix<dofpn*nnode,1> NxxTrxx;

  LINALG::Matrix<dofpn*nnode,dofpn*nnode> M2;
  LINALG::Matrix<dofpn*nnode,dofpn*nnode> NxTrxrxTNx;

  //Matrices for N_i,xi and N_i,xixi. 2*nnode due to hermite shapefunctions
  LINALG::Matrix<1,NODALDOFS*nnode> N_i;
  LINALG::Matrix<1,NODALDOFS*nnode> N_i_x;
  LINALG::Matrix<1,NODALDOFS*nnode> N_i_xx;

  LINALG::Matrix<3,nnode*dofpn> N;
  LINALG::Matrix<3,nnode*dofpn> N_x;
  LINALG::Matrix<3,nnode*dofpn> N_xx;

  //stiffness due to tension and bending
  LINALG::Matrix<nnode*dofpn,nnode*dofpn> R_tension;
  LINALG::Matrix<nnode*dofpn,nnode*dofpn> R_bending;

  //internal force due to tension and bending
  LINALG::Matrix<nnode*dofpn,1> Res_tension;
  LINALG::Matrix<nnode*dofpn,1> Res_bending;

  //some matrices necessary for ANS approach
  #ifdef ANS
    #if (NODALDOFS ==3)
    dserror("ANS approach so far only defined for third order Hermitian shape functions, set NODALDOFS=2!!!");
    #endif
  LINALG::Matrix<1,3> L_i;
  L_i.Clear();
  LINALG::Matrix<nnode*dofpn,1> Res_tension_ANS;
  Res_tension_ANS.Clear();
  LINALG::Matrix<nnode*dofpn,nnode*dofpn> R_tension_ANS;
  R_tension_ANS.Clear();
  double epsilon_ANS = 0.0;
  LINALG::Matrix<1,nnode*dofpn> lin_epsilon_ANS;
  lin_epsilon_ANS.Clear();
  #endif

  //first of all we get the material law
  Teuchos::RCP<const MAT::Material> currmat = Material();
  double ym = 0;
  //Uncomment the next line for the dynamic case: so far only the static case is implemented
  //double density = 0;

  //assignment of material parameters; only St.Venant material is accepted for this beam
  switch(currmat->MaterialType())
  {
    case INPAR::MAT::m_stvenant:// only linear elastic material supported
    {
      const MAT::StVenantKirchhoff* actmat = static_cast<const MAT::StVenantKirchhoff*>(currmat.get());
      ym = actmat->Youngs();
      //Uncomment the next line for the dynamic case: so far only the static case is implemented
      //density = actmat->Density();
    }
    break;
    default:
    dserror("unknown or improper type of material law");
    break;
  }

  //TODO: The integration rule should be set via input parameter and not hard coded as here
  //Get integrationpoints for exact integration
  DRT::UTILS::IntegrationPoints1D gausspoints = DRT::UTILS::IntegrationPoints1D(DRT::UTILS::mygaussruleeb);

  //Get DiscretizationType of beam element
  const DRT::Element::DiscretizationType distype = Shape();

  //update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
  for (int node = 0 ; node < nnode ; node++)
  {
    for (int dof = 0 ; dof < dofpn ; dof++)
    {
      if(dof < 3)
      {
        //position of nodes
        disp_totlag[node*dofpn + dof] = (Nodes()[node]->X()[dof] + disp[node*dofpn + dof])*ScaleFactorColumn;
      }
      else if(dof<6)
      {
        //tangent at nodes
        disp_totlag[node*dofpn + dof] = (Tref_[node](dof-3) + disp[node*dofpn + dof])*ScaleFactorColumn;
      }
      else if(dof>=6)
      {
        #if NODALDOFS ==3
        //curvatures at nodes
        disp_totlag[node*dofpn + dof] = (Kref_[node](dof-6) + disp[node*dofpn + dof])*ScaleFactorColumn;
        #endif
      }
    }
  } //for (int node = 0 ; node < nnode ; node++)

  //Calculate epsilon at collocation points
  #ifdef ANS
  LINALG::Matrix<3,1> epsilon_cp;
  epsilon_cp.Clear();
  LINALG::Matrix<3,3> tangent_cp;
  tangent_cp.Clear();
  LINALG::Matrix<3,NODALDOFS*6> lin_epsilon_cp;
  lin_epsilon_cp.Clear();

  N_i_x.Clear();
  DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,0.0,jacobi_*2.0,distype);
  for (int i=0;i<2*NODALDOFS;i++)
  {
    N_i_x(i)=N_i_x(i)/jacobi_;
  }

  for (int i=0;i<3;i++)
  {
    tangent_cp(i,0)=disp_totlag[i+3];
    tangent_cp(i,1)=disp_totlag[i+9];

    for (int j=0;j<2*NODALDOFS;j++)
    {
      tangent_cp(i,2)+=N_i_x(j)*disp_totlag[3*j+i];
    }
  }
  for (int i=0;i<3;i++)
  {
    for (int j=0;j<3;j++)
    {
      epsilon_cp(i)+=tangent_cp(j,i)*tangent_cp(j,i);
    }
    epsilon_cp(i)=pow(epsilon_cp(i),0.5)-1.0;
  }

  for (int k=0;k<3;k++)
  {
    N_i_x.Clear();

    switch (k)
    {
    case 0:
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,-1.0,jacobi_*2.0,distype);
      break;
    case 1:
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,1.0,jacobi_*2.0,distype);
      break;
    case 2:
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,0.0,jacobi_*2.0,distype);
      break;
    default:
      dserror("Index k should only run from 1 to 3 (three collocation points)!");
     break;
    }

    for (int i=0;i<2*NODALDOFS;i++)
    {
      N_i_x(i)=N_i_x(i)/jacobi_;
    }
    //loop over space dimensions
    for (int i=0;i<3;i++)
    { //loop over all shape functions
      for (int j=0;j<2*NODALDOFS;j++)
      { //loop over CPs
          lin_epsilon_cp(k,3*j + i)+=tangent_cp(i,k)*N_i_x(j)/(epsilon_cp(k)+1);
      }
    }
  }

  #endif

  //Loop through all GP and calculate their contribution to the internal forcevector and stiffnessmatrix
  for(int numgp=0; numgp < gausspoints.nquad; numgp++)
  {
    //all matrices and scalars are set to zero again!!!
    //factors for stiffness assembly

    r_.Clear();
    r_x.Clear();
    r_xx.Clear();

    f1.Clear();
    f2.Clear();
    n1.Clear();

    rxxrxx=0;
    rxrx=0;
    tension=0;

    NTildex.Clear();
    NTildexx.Clear();

    NxTrx.Clear();
    NxxTrxx.Clear();

    M2.Clear();
    NxTrxrxTNx.Clear();

    N_i.Clear();
    N_i_x.Clear();
    N_i_xx.Clear();

    N.Clear();
    N_x.Clear();
    N_xx.Clear();

    R_tension.Clear();
    R_bending.Clear();

    Res_tension.Clear();
    Res_bending.Clear();

    //Get location and weight of GP in parameter space
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

#if (NODALDOFS == 2)
    //Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
    DRT::UTILS::shape_function_hermite_1D(N_i,xi,jacobi_*2.0,distype);
    DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,xi,jacobi_*2.0,distype);
    DRT::UTILS::shape_function_hermite_1D_deriv2(N_i_xx,xi,jacobi_*2.0,distype);
    //end--------------------------------------------------------
#elif (NODALDOFS == 3)
    //specific-for----------------------------------Frenet Serret
    //Get hermite derivatives N'xi, N''xi and N'''xi
    DRT::UTILS::shape_function_hermite_1D_order5(N_i,xi,jacobi_*2.0,distype);
    DRT::UTILS::shape_function_hermite_1D_order5_deriv1(N_i_x,xi,jacobi_*2.0,distype);
    DRT::UTILS::shape_function_hermite_1D_order5_deriv2(N_i_xx,xi,jacobi_*2.0,distype);
    //end--------------------------------------------------------
#else
    dserror("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif


    //calculate r' and r''
    for (int i=0 ; i < 3 ; i++)
    {
      for (int j=0; j<2*NODALDOFS; j++)
      {
        r_(i,0)+= N_i(j)*disp_totlag[3*j + i];
        r_x(i,0)+= N_i_x(j) * disp_totlag[3*j + i];
        r_xx(i,0)+= N_i_xx(j) * disp_totlag[3*j + i];
      }
    }

    for (int i=0; i<3; i++)
    {
      rxxrxx+=r_xx(i)*r_xx(i);
      rxrx+=r_x(i)*r_x(i);
    }

    tension = 1/jacobi_ - 1/pow(rxrx,0.5);

    for (int i=0; i<3; ++i)
    {
      for (int j=0; j<2*NODALDOFS; ++j)
      {
        N(i,i+3*j) += N_i(j);
        N_x(i,i+3*j) += N_i_x(j);
        N_xx(i,i+3*j) += N_i_xx(j);
        NxTrx(i+3*j)+=N_i_x(j)*r_x(i);
        NxxTrxx(i+3*j)+=N_i_xx(j)*r_xx(i);
      }
    }

    NTildex.MultiplyTN(N_x,N_x);
    NTildexx.MultiplyTN(N_xx,N_xx);

    for (int i= 0; i<nnode*dofpn; i++)
    {
      for (int j= 0; j<nnode*dofpn; j++)
      {
        M2(i,j)+= NxxTrxx(i)*NxTrx(j);
        NxTrxrxTNx(i,j)+= NxTrx(i)*NxTrx(j);
      }
    }

#ifdef ANS
    DRT::UTILS::shape_function_1D(L_i,xi,line3);
    epsilon_ANS = 0.0;
    lin_epsilon_ANS.Clear();
    for (int i=0;i<3;i++)
    {
      epsilon_ANS+=L_i(i)*epsilon_cp(i);
      for (int j=0;j<nnode*dofpn;j++)
      {
        lin_epsilon_ANS(j)+=L_i(i)*lin_epsilon_cp(i,j);
      }
    }

    Res_tension_ANS.Clear();
    R_tension_ANS.Clear();

    for (int i=0;i<nnode*dofpn;i++)
    {
      for (int j=0;j<nnode*dofpn;j++)
      {
        R_tension_ANS(i,j)+=NxTrx(i)*lin_epsilon_ANS(j)/jacobi_;
      }
    }
#endif

    //assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
    if (stiffmatrix != NULL)
    {

      //assemble parts from tension
      #ifndef ANS
      R_tension = NTildex;
      R_tension.Scale(tension);
      R_tension.Update(1.0 / pow(rxrx,1.5),NxTrxrxTNx,1.0);
      R_tension.Scale(ym * crosssec_ * wgt);
      #else
      //attention: in epsilon_ANS and lin_epsilon_ANS the corresponding jacobi factors are allready considered,
      //all the other jacobi factors due to differentiation and integration cancel out!!!
      R_tension_ANS.Update(epsilon_ANS/jacobi_,NTildex,1.0);
      R_tension_ANS.Scale(ym * crosssec_ * wgt);
      #endif

      //assemble parts from bending
      R_bending.Update(-rxxrxx/pow(jacobi_,2.0) ,NTildex,1.0);
      R_bending.Update(1.0,NTildexx,1.0);
      R_bending.UpdateT(- 2.0 / pow(jacobi_,2.0) , M2 , 1.0);

      R_bending.Scale(ym * Izz_ * wgt / pow(jacobi_,3));

      //shifting values from fixed size matrix to epetra matrix *stiffmatrix
      for(int i = 0; i < dofpn*nnode; i++)
      {
        for(int j = 0; j < dofpn*nnode; j++)
        {
          #ifndef ANS
          (*stiffmatrix)(i,j) += R_tension(i,j);
          #else
          (*stiffmatrix)(i,j) += R_tension_ANS(i,j);
          #endif
          (*stiffmatrix)(i,j) += R_bending(i,j);
        }
      } //for(int i = 0; i < dofpn*nnode; i++)
    }//if (stiffmatrix != NULL)

    for (int i= 0; i<3; i++)
    {
      f1(i)=-r_x(i)*rxxrxx;
      f2(i)=r_xx(i);
      n1(i)=r_x(i)*tension;
    }

    //assemble internal force vector f_internal / Res in thesis Meier
    if (force != NULL)
    {
      for (int i=0;i<3;i++)
      {
        for (int j=0;j<2*NODALDOFS;j++)
        {
          Res_bending(j*3 + i)+= N_i_x(j)*f1(i)/pow(jacobi_,5) + N_i_xx(j)*f2(i)/pow(jacobi_,3);
          #ifndef ANS
          Res_tension(j*3 + i)+= N_i_x(j)*n1(i);
          #endif
        }
      }
      #ifdef ANS
      Res_tension_ANS.Update(ym * crosssec_ * wgt*epsilon_ANS / jacobi_,NxTrx,1.0);
      #endif
      Res_bending.Scale(ym * Izz_ * wgt);
      Res_tension.Scale(ym * crosssec_ * wgt);

      //shifting values from fixed size vector to epetra vector *force
      for(int i = 0; i < dofpn*nnode; i++)
      {
        #ifndef ANS
        (*force)(i) += Res_tension(i);
        #else
        (*force)(i) += Res_tension_ANS(i);
        #endif
        (*force)(i) += Res_bending(i) ;
      }
    } //if (force != NULL)

    //assemble massmatrix if requested
    if (massmatrix != NULL)
    {
      cout << "\n\nWarning: Massmatrix not implemented yet!";
    }//if (massmatrix != NULL)
  } //for(int numgp=0; numgp < gausspoints.nquad; numgp++)

  //Uncomment the following line to print the elment stiffness matrix to matlab format
  /*
  const std::string fname = "stiffmatrixele.mtl";
  cout<<"Printing stiffmatrixele to file"<<endl;
  LINALG::PrintSerialDenseMatrixInMatlabFormat(fname,*stiffmatrix);
  */

  //with the following lines the conditioning of the stiffness matrix should be improved: its not fully tested up to now!!!
  bool precond = PreConditioning;
  if (precond)
  {
    #if NODALDOFS ==3
      dserror("Preconditioning only implemented for NODALDOFS ==2!!!");
    #endif
    double length = jacobi_*2.0;
    double radius = std::pow(crosssec_/M_PI,0.5);
    for (int zeile=0; zeile <2; zeile++)
    {
      for (int spalte=0; spalte<12; spalte++)
      {
        (*stiffmatrix)(6*zeile,spalte)=(*stiffmatrix)(6*zeile,spalte)*length;
        (*stiffmatrix)(6*zeile+1,spalte)=(*stiffmatrix)(6*zeile+1,spalte)*pow(length,3.0)/pow(radius,2.0);
        (*stiffmatrix)(6*zeile+2,spalte)=(*stiffmatrix)(6*zeile+2,spalte)*pow(length,3.0)/pow(radius,2.0);
        (*stiffmatrix)(6*zeile+4,spalte)=(*stiffmatrix)(6*zeile+4,spalte)*pow(length,2.0)/pow(radius,2.0);
        (*stiffmatrix)(6*zeile+5,spalte)=(*stiffmatrix)(6*zeile+5,spalte)*pow(length,2.0)/pow(radius,2.0);
      }
        (*force)(6*zeile)=(*force)(6*zeile)*length;
        (*force)(6*zeile+1)=(*force)(6*zeile+1)*pow(length,3.0)/pow(radius,2.0);
        (*force)(6*zeile+2)=(*force)(6*zeile+2)*pow(length,3.0)/pow(radius,2.0);
        (*force)(6*zeile+4)=(*force)(6*zeile+4)*pow(length,2.0)/pow(radius,2.0);
        (*force)(6*zeile+5)=(*force)(6*zeile+5)*pow(length,2.0)/pow(radius,2.0);
    }
  }

  //with the following lines the conditioning of the stiffness matrix can be improved by multiplying the lines and columns with the factors
  //ScaleFactorLine and ScaleFactorColumn
  double Factor = ScaleFactorLine;
  Factor = Factor * ScaleFactorColumn;

  for (int zeile=0; zeile <nnode*dofpn; zeile++)
  {
    for (int spalte=0; spalte<nnode*dofpn; spalte++)
    {
      (*stiffmatrix)(zeile,spalte)=(*stiffmatrix)(zeile,spalte)*Factor;
    }
    (*force)(zeile)=(*force)(zeile)*ScaleFactorLine;
  }
}
#else
{
   //dimensions of freedom per node
  const int dofpn = 3*NODALDOFS;

  //number of nodes fixed for these element
  const int nnode = 2;

  //matrix for current positions and tangents
  std::vector<double> disp_totlag(nnode*dofpn, 0.0);
  std::vector<FAD> disp_totlag_fad(nnode*dofpn, 0.0);

  LINALG::Matrix<3,1> r_;
  LINALG::Matrix<3,1> r_x;
  LINALG::Matrix<3,1> r_xx;

  LINALG::Matrix<3,1> f1;
  LINALG::Matrix<3,1> f2;
  LINALG::Matrix<3,1> n1;

  double rxrxx;
  double rxxrxx;
  double rxrx;
  double tension;

  LINALG::TMatrix<FAD,3,1> rx_fad;
  FAD rxrx_fad;

  LINALG::Matrix<dofpn*nnode,dofpn*nnode> NTilde;
  LINALG::Matrix<dofpn*nnode,dofpn*nnode> NTildex;
  LINALG::Matrix<dofpn*nnode,dofpn*nnode> NTildexx;

  LINALG::Matrix<dofpn*nnode,1> NxTrx;
  LINALG::Matrix<dofpn*nnode,1> NxTrxx;
  LINALG::Matrix<dofpn*nnode,1> NxxTrx;
  LINALG::Matrix<dofpn*nnode,1> NxxTrxx;

  LINALG::Matrix<dofpn*nnode,dofpn*nnode> M1;
  LINALG::Matrix<dofpn*nnode,dofpn*nnode> M2;
  LINALG::Matrix<dofpn*nnode,dofpn*nnode> M3;
  LINALG::Matrix<dofpn*nnode,dofpn*nnode> NxTrxrxTNx;

  //Matrices for N_i,xi and N_i,xixi. 2*nnode due to hermite shapefunctions
  LINALG::Matrix<1,NODALDOFS*nnode> N_i;
  LINALG::Matrix<1,NODALDOFS*nnode> N_i_x;
  LINALG::Matrix<1,NODALDOFS*nnode> N_i_xx;

  LINALG::Matrix<3,nnode*dofpn> N_x;
  LINALG::Matrix<3,nnode*dofpn> N_xx;

  //stiffness due to tension and bending
  LINALG::Matrix<nnode*dofpn,nnode*dofpn> R_tension;
  LINALG::Matrix<nnode*dofpn,nnode*dofpn> R_bending;

  //internal force due to tension and bending
  LINALG::Matrix<nnode*dofpn,1> Res_tension;
  LINALG::Matrix<nnode*dofpn,1> Res_bending;

  //some matrices necessary for ANS approach
  #ifdef ANS
    #if (NODALDOFS ==3)
    dserror("ANS approach so far only defined for third order Hermitian shape functions, set NODALDOFS=2!!!");
    #endif
  LINALG::Matrix<1,3> L_i;
  L_i.Clear();
  LINALG::Matrix<nnode*dofpn,1> Res_tension_ANS;
  Res_tension_ANS.Clear();
  LINALG::Matrix<nnode*dofpn,nnode*dofpn> R_tension_ANS;
  R_tension_ANS.Clear();
  double epsilon_ANS = 0.0;
  LINALG::Matrix<1,nnode*dofpn> lin_epsilon_ANS;
  lin_epsilon_ANS.Clear();

  LINALG::TMatrix<FAD,nnode*dofpn,1> Res_tension_ANS_fad;
  Res_tension_ANS_fad.Clear();
  LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> R_tension_ANS_fad;
  R_tension_ANS_fad.Clear();
  FAD epsilon_ANS_fad = 0.0;
  #endif

  //first of all we get the material law
  Teuchos::RCP<const MAT::Material> currmat = Material();
  double ym = 0;
  //Uncomment the next line for the dynamic case: so far only the static case is implemented
  //double density = 0;

  //assignment of material parameters; only St.Venant material is accepted for this beam
  switch(currmat->MaterialType())
  {
    case INPAR::MAT::m_stvenant:// only linear elastic material supported
    {
      const MAT::StVenantKirchhoff* actmat = static_cast<const MAT::StVenantKirchhoff*>(currmat.get());
      ym = actmat->Youngs();
      //Uncomment the next line for the dynamic case: so far only the static case is implemented
      //density = actmat->Density();
    }
    break;
    default:
    dserror("unknown or improper type of material law");
    break;
  }

  //TODO: The integration rule should be set via input parameter and not hard coded as here (standard: 3)
  //Get integrationpoints for exact integration
  DRT::UTILS::IntegrationPoints1D gausspoints = DRT::UTILS::IntegrationPoints1D(DRT::UTILS::mygaussruleeb);

  //Get DiscretizationType of beam element
  const DRT::Element::DiscretizationType distype = Shape();

  //update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
  for (int node = 0 ; node < nnode ; node++)
  {
    for (int dof = 0 ; dof < dofpn ; dof++)
    {
      if(dof < 3)
      {
        //position of nodes
        disp_totlag[node*dofpn + dof] = (Nodes()[node]->X()[dof] + disp[node*dofpn + dof])*ScaleFactorColumn;
      }
      else if(dof<6)
      {
        //tangent at nodes
        disp_totlag[node*dofpn + dof] = (Tref_[node](dof-3) + disp[node*dofpn + dof])*ScaleFactorColumn;
      }
      else if(dof>=6)
      {
        #if NODALDOFS ==3
        //curvatures at nodes
        disp_totlag[node*dofpn + dof] = (Kref_[node](dof-6) + disp[node*dofpn + dof])*ScaleFactorColumn;
        #endif
      }
    }
  } //for (int node = 0 ; node < nnode ; node++)
  for (int dof=0;dof<nnode*dofpn;dof++)
  {
    disp_totlag_fad[dof]=disp_totlag[dof];
    disp_totlag_fad[dof].diff(dof,nnode*dofpn);
  }

  //Calculate epsilon at collocation points
  #ifdef ANS
  LINALG::Matrix<3,1> epsilon_cp;
  epsilon_cp.Clear();
  LINALG::Matrix<3,3> tangent_cp;
  tangent_cp.Clear();
  LINALG::Matrix<3,NODALDOFS*6> lin_epsilon_cp;
  lin_epsilon_cp.Clear();

  LINALG::TMatrix<FAD,3,1> epsilon_cp_fad;
  epsilon_cp_fad.Clear();
  LINALG::TMatrix<FAD,3,3> tangent_cp_fad;
  tangent_cp_fad.Clear();

  N_i_x.Clear();
  DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,0.0,jacobi_*2.0,distype);
  for (int i=0;i<2*NODALDOFS;i++)
  {
    N_i_x(i)=N_i_x(i)/jacobi_;
  }

  for (int i=0;i<3;i++)
  {
    tangent_cp(i,0)=disp_totlag[i+3];
    tangent_cp(i,1)=disp_totlag[i+9];

    tangent_cp_fad(i,0)=disp_totlag_fad[i+3];
    tangent_cp_fad(i,1)=disp_totlag_fad[i+9];
    for (int j=0;j<2*NODALDOFS;j++)
    {
      tangent_cp(i,2)+=N_i_x(j)*disp_totlag[3*j+i];
      tangent_cp_fad(i,2)+=N_i_x(j)*disp_totlag_fad[3*j+i];
    }
  }
  for (int i=0;i<3;i++)
  {
    for (int j=0;j<3;j++)
    {
      epsilon_cp(i)+=tangent_cp(j,i)*tangent_cp(j,i);
      epsilon_cp_fad(i)+=tangent_cp_fad(j,i)*tangent_cp_fad(j,i);
    }
    epsilon_cp(i)=pow(epsilon_cp(i),0.5)-1.0;
    epsilon_cp_fad(i)=pow(epsilon_cp_fad(i),0.5)-1.0;
  }

  for (int k=0;k<3;k++)
  {
    N_i_x.Clear();

    switch (k)
    {
    case 0:
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,-1.0,jacobi_*2.0,distype);
      break;
    case 1:
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,1.0,jacobi_*2.0,distype);
      break;
    case 2:
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,0.0,jacobi_*2.0,distype);
      break;
    default:
      dserror("Index k should only run from 1 to 3 (three collocation points)!");
     break;
    }

    for (int i=0;i<2*NODALDOFS;i++)
    {
      N_i_x(i)=N_i_x(i)/jacobi_;
    }
    //loop over space dimensions
    for (int i=0;i<3;i++)
    { //loop over all shape functions
      for (int j=0;j<2*NODALDOFS;j++)
      { //loop over CPs
          lin_epsilon_cp(k,3*j + i)+=tangent_cp(i,k)*N_i_x(j)/(epsilon_cp(k)+1);
      }
    }
  }

  #endif

  //Loop through all GP and calculate their contribution to the internal forcevector and stiffnessmatrix
  for(int numgp=0; numgp < gausspoints.nquad; numgp++)
  {
    //all matrices and scalars are set to zero again!!!
    //factors for stiffness assembly

    r_.Clear();
    r_x.Clear();
    r_xx.Clear();

    f1.Clear();
    f2.Clear();
    n1.Clear();

    rxrxx=0;
    rxxrxx=0;
    rxrx=0;
    tension=0;

    rx_fad.Clear();
    rxrx_fad=0.0;

    NTilde.Clear();
    NTildex.Clear();
    NTildexx.Clear();

    NxTrx.Clear();
    NxTrxx.Clear();
    NxxTrx.Clear();
    NxxTrxx.Clear();

    M1.Clear();
    M2.Clear();
    M3.Clear();
    NxTrxrxTNx.Clear();

    N_i.Clear();
    N_i_x.Clear();
    N_i_xx.Clear();

    N_x.Clear();
    N_xx.Clear();

    R_tension.Clear();
    R_bending.Clear();

    Res_tension.Clear();
    Res_bending.Clear();

    //Get location and weight of GP in parameter space
    const double xi = gausspoints.qxg[numgp][0];
    const double wgt = gausspoints.qwgt[numgp];

#if (NODALDOFS == 2)
    //Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
    DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,xi,jacobi_*2.0,distype);
    DRT::UTILS::shape_function_hermite_1D_deriv2(N_i_xx,xi,jacobi_*2.0,distype);
    //end--------------------------------------------------------
#elif (NODALDOFS == 3)
    //specific-for----------------------------------Frenet Serret
    //Get hermite derivatives N'xi, N''xi and N'''xi
    DRT::UTILS::shape_function_hermite_1D_order5_deriv1(N_i_x,xi,jacobi_*2.0,distype);
    DRT::UTILS::shape_function_hermite_1D_order5_deriv2(N_i_xx,xi,jacobi_*2.0,distype);
    //end--------------------------------------------------------
#else
    dserror("Only the values NODALDOFS = 2 and NODALDOFS = 3 are valid!");
#endif

    //calculate r' and r''
    for (int i=0 ; i < 3 ; i++)
    {
      for (int j=0; j<nnode*NODALDOFS; j++)
      {
        //r_(i,0)+= N_i(j)*disp_totlag[3*j + i];
        r_x(i,0)+= N_i_x(j) * disp_totlag[3*j + i];
        rx_fad(i,0)+= N_i_x(j) * disp_totlag_fad[3*j + i];
        r_xx(i,0)+= N_i_xx(j) * disp_totlag[3*j + i];
      }
    }

    for (int i= 0; i<3; i++)
    {
      rxrxx+=r_x(i)*r_xx(i);
      rxxrxx+=r_xx(i)*r_xx(i);
      rxrx+=r_x(i)*r_x(i);
      rxrx_fad+=rx_fad(i)*rx_fad(i);
    }

    tension = 1/jacobi_ - 1/pow(rxrx,0.5);

    for (int i=0; i<3; ++i)
    {
      for (int j=0; j<nnode*NODALDOFS; ++j)
      {
        N_x(i,i+3*j) += N_i_x(j);
        N_xx(i,i+3*j) += N_i_xx(j);
        NxTrx(i+3*j)+=N_i_x(j)*r_x(i);
        NxTrxx(i+3*j)+=N_i_x(j)*r_xx(i);
        NxxTrx(i+3*j)+=N_i_xx(j)*r_x(i);
        NxxTrxx(i+3*j)+=N_i_xx(j)*r_xx(i);
      }
    }

    NTilde.MultiplyTN(N_x,N_xx);
    NTildex.MultiplyTN(N_x,N_x);
    NTildexx.MultiplyTN(N_xx,N_xx);

    for (int i= 0; i<nnode*dofpn; i++)
    {
      for (int j= 0; j<nnode*dofpn; j++)
      {
        M1(i,j)+= NxTrx(i)*(NxxTrx(j)+NxTrxx(j));
        M2(i,j)+= NxxTrxx(i)*NxTrx(j);
        M3(i,j)+= (NxTrxx(i)+NxxTrx(i))*(NxTrxx(j)+NxxTrx(j));
        NxTrxrxTNx(i,j)+= NxTrx(i)*NxTrx(j);
      }
    }

    //calculate quantities necessary for ANS approach
    #ifdef ANS
    DRT::UTILS::shape_function_1D(L_i,xi,line3);
    epsilon_ANS = 0.0;
    epsilon_ANS_fad = 0.0;
    lin_epsilon_ANS.Clear();
    for (int i=0;i<3;i++)
    {
      epsilon_ANS+=L_i(i)*epsilon_cp(i);
      epsilon_ANS_fad+=L_i(i)*epsilon_cp_fad(i);
      for (int j=0;j<nnode*dofpn;j++)
      {
        lin_epsilon_ANS(j)+=L_i(i)*lin_epsilon_cp(i,j);
      }
    }

    Res_tension_ANS.Clear();
    R_tension_ANS.Clear();

    Res_tension_ANS_fad.Clear();
    R_tension_ANS_fad.Clear();
    for (int i=0;i<nnode*dofpn;i++)
    {
      for (int j=0;j<nnode*dofpn;j++)
      {
        R_tension_ANS(i,j)+=NxTrx(i)*lin_epsilon_ANS(j)/pow(rxrx,0.5);
      }
      for (int k=0;k<3;k++)
      {
        Res_tension_ANS_fad(i)+=N_x(k,i)*rx_fad(k)/pow(rxrx_fad,0.5)*ym * crosssec_ * wgt*epsilon_ANS_fad;
      }
    }
    for (int i=0;i<nnode*dofpn;i++)
    {
      for (int j=0;j<nnode*dofpn;j++)
      {
        R_tension_ANS_fad(i,j)=Res_tension_ANS_fad(i).dx(j);
      }
    }

    LINALG::Matrix<nnode*dofpn,nnode*dofpn> R_tension_ANS_test;
    R_tension_ANS_test.Clear();
    #endif

    //assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
    if (stiffmatrix != NULL)
    {
      //assemble parts from tension
      #ifndef ANS
      R_tension = NTildex;
      R_tension.Scale(tension);
      R_tension.Update(1.0 / pow(rxrx,1.5),NxTrxrxTNx,1.0);
      R_tension.Scale(ym * crosssec_ * wgt);
      #else
      //attention: in epsilon_ANS and lin_epsilon_ANS the corresponding jacobi factors are allready considered,
      //all the other jacobi factors due to differentiation and integration cancel out!!!

      R_tension_ANS.Update(-epsilon_ANS / pow(rxrx,1.5),NxTrxrxTNx,1.0);
      R_tension_ANS.Update(epsilon_ANS / pow(rxrx,0.5),NTildex,1.0);
      R_tension_ANS.Scale(ym * crosssec_ * wgt);
      #endif

      //assemble parts from bending
      R_bending = NTildex;
      R_bending.Scale(2.0 * pow(rxrxx,2.0) / pow(rxrx,3.0));
      R_bending.Update(-rxxrxx/pow(rxrx,2.0),NTildex,1.0);
      R_bending.Update(-rxrxx/pow(rxrx,2.0),NTilde,1.0);
      R_bending.UpdateT(-rxrxx/pow(rxrx,2.0),NTilde,1.0);
      R_bending.Update(1.0/rxrx,NTildexx,1.0);
      R_bending.Update(-12.0 * pow(rxrxx,2.0)/pow(rxrx,4.0),NxTrxrxTNx,1.0);
      R_bending.Update(4.0 * rxrxx / pow(rxrx,3.0) , M1 , 1.0);
      R_bending.UpdateT(4.0 * rxrxx / pow(rxrx,3.0) , M1 , 1.0);
      R_bending.Update(4.0 * rxxrxx / pow(rxrx,3.0) , NxTrxrxTNx, 1.0);
      R_bending.Update(- 2.0 / pow(rxrx,2.0) , M2 , 1.0);
      R_bending.UpdateT(- 2.0 / pow(rxrx,2.0) , M2 , 1.0);
      R_bending.Update(- 1.0 / pow(rxrx,2.0) , M3 , 1.0);

      R_bending.Scale(ym * Izz_ * wgt / jacobi_);

      //shifting values from fixed size matrix to epetra matrix *stiffmatrix
      for(int i = 0; i < dofpn*nnode; i++)
      {
        for(int j = 0; j < dofpn*nnode; j++)
        {
          #ifndef ANS
          (*stiffmatrix)(i,j) += R_tension(i,j);
          #else
          (*stiffmatrix)(i,j) += R_tension_ANS(i,j);
          #endif
          (*stiffmatrix)(i,j) += R_bending(i,j);
        }
      } //for(int i = 0; i < dofpn*nnode; i++)
    }//if (stiffmatrix != NULL)

    for (int i= 0; i<3; i++)
    {
      f1(i)=2*r_x(i)*pow(rxrxx,2.0)/pow(rxrx,3.0)-(r_x(i)*rxxrxx+r_xx(i)*rxrxx)/pow(rxrx,2.0);
      f2(i)=r_xx(i)/rxrx-r_x(i)*rxrxx/pow(rxrx,2.0);
      n1(i)=r_x(i)*tension;
    }
    //assemble internal force vector f_internal / Res in thesis Meier
    if (force != NULL)
    {
      for (int i=0;i<3;i++)
      {
        for (int j=0;j<nnode*NODALDOFS;j++)
        {
          Res_bending(j*3 + i)+= N_i_x(j)*f1(i) + N_i_xx(j)*f2(i);
          #ifndef ANS
          Res_tension(j*3 + i)+= N_i_x(j)*n1(i);
          #endif
        }
      }
      #ifdef ANS
      Res_tension_ANS.Update(ym * crosssec_ * wgt*epsilon_ANS / pow(rxrx,0.5),NxTrx,1.0);
      #endif
      Res_bending.Scale(ym * Izz_ * wgt / jacobi_);
      Res_tension.Scale(ym * crosssec_ * wgt);

      //shifting values from fixed size vector to epetra vector *force
      for(int i = 0; i < dofpn*nnode; i++)
      {
          #ifndef ANS
          (*force)(i) += Res_tension(i);
          #else
          (*force)(i) += Res_tension_ANS(i);
          #endif
          (*force)(i) += Res_bending(i) ;
      }
    } //if (force != NULL)

    //assemble massmatrix if requested
    if (massmatrix != NULL)
    {
      cout << "\n\nWarning: Massmatrix not implemented yet!";
    }//if (massmatrix != NULL)
  } //for(int numgp=0; numgp < gausspoints.nquad; numgp++)

  //Uncomment the following line to print the elment stiffness matrix to matlab format
  /*
  const std::string fname = "stiffmatrixele.mtl";
  cout<<"Printing stiffmatrixele to file"<<endl;
  LINALG::PrintSerialDenseMatrixInMatlabFormat(fname,*stiffmatrix);
  */

  //with the following lines the conditioning of the stiffness matrix should be improved: its not fully tested up to now!!!
  bool precond = PreConditioning;
  if (precond)
  {
#if NODALDOFS == 3
    dserror("Preconditioning is not implemented for the case NODALDOFS = 3!");
#endif
    double length = jacobi_*2.0;
    double radius = std::pow(crosssec_/M_PI,0.5);
    for (int zeile=0; zeile <2; zeile++)
    {
      for (int spalte=0; spalte<12; spalte++)
      {
        (*stiffmatrix)(6*zeile,spalte)=(*stiffmatrix)(6*zeile,spalte)*length;
        (*stiffmatrix)(6*zeile+1,spalte)=(*stiffmatrix)(6*zeile+1,spalte)*pow(length,3.0)/pow(radius,2.0);
        (*stiffmatrix)(6*zeile+2,spalte)=(*stiffmatrix)(6*zeile+2,spalte)*pow(length,3.0)/pow(radius,2.0);
        (*stiffmatrix)(6*zeile+4,spalte)=(*stiffmatrix)(6*zeile+4,spalte)*pow(length,2.0)/pow(radius,2.0);
        (*stiffmatrix)(6*zeile+5,spalte)=(*stiffmatrix)(6*zeile+5,spalte)*pow(length,2.0)/pow(radius,2.0);
      }
      (*force)(6*zeile)=(*force)(6*zeile)*length;
      (*force)(6*zeile+1)=(*force)(6*zeile+1)*pow(length,3.0)/pow(radius,2.0);
      (*force)(6*zeile+2)=(*force)(6*zeile+2)*pow(length,3.0)/pow(radius,2.0);
      (*force)(6*zeile+4)=(*force)(6*zeile+4)*pow(length,2.0)/pow(radius,2.0);
      (*force)(6*zeile+5)=(*force)(6*zeile+5)*pow(length,2.0)/pow(radius,2.0);
    }
  }

  //with the following lines the conditioning of the stiffness matrix can be improved by multiplying the lines and columns with the factors
  //ScaleFactorLine and ScaleFactorColumn
  double Factor = ScaleFactorLine;
  Factor = Factor * ScaleFactorColumn;

  for (int zeile=0; zeile <nnode*dofpn; zeile++)
  {
    for (int spalte=0; spalte<nnode*dofpn; spalte++)
    {
      (*stiffmatrix)(zeile,spalte)=(*stiffmatrix)(zeile,spalte)*Factor;
    }
    (*force)(zeile)=(*force)(zeile)*ScaleFactorLine;
  }
}
#endif
  //Uncomment the next line if the implementation of the analytical stiffness matrix should be checked by Forward Automatic Differentiation (FAD)
  //FADCheckStiffMatrix(disp, stiffmatrix, force);

  return;

} // DRT::ELEMENTS::Beam3eb::eb_nlnstiffmass

/*------------------------------------------------------------------------------------------------------------*
 | lump mass matrix					   (private)                                                   meier 05/12|
 *------------------------------------------------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3eb::lumpmass(Epetra_SerialDenseMatrix* emass)
{
  cout << "\n\nWarning: Massmatrix not implemented yet!";
}

//***************************************************************************************
//Methods for FAD Check
//***************************************************************************************

void DRT::ELEMENTS::Beam3eb::FADCheckStiffMatrix(std::vector<double>& disp,
                                                 Epetra_SerialDenseMatrix* stiffmatrix,
                                                 Epetra_SerialDenseVector* force)
{
#if NODALDOFS == 3
    dserror("FADCheck are not implemented for the case NODALDOFS = 3!!!");
#endif
  #ifdef SIMPLECALC
  {
    //see also so_nstet_nodalstrain.cpp, so_nstet.H, autodiff.cpp and autodiff.H
    //FAD calculated stiff matrix for validation purposes
    Epetra_SerialDenseMatrix stiffmatrix_check;
    LINALG::TMatrix<FAD,12,1> force_check;

    //reshape stiffmatrix_check
    stiffmatrix_check.Shape(12,12);

    //dimensions of freedom per node
    const int dofpn = 6;

    //number of nodes fixed for these element
    const int nnode = 2;

    //matrix for current positions and tangents
    std::vector<FAD> disp_totlag(nnode*dofpn,0.0);

    //abbreviated matrices for clearness
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde;
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde_x;
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde_xx;
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde_aux;

    //matrices helping to assemble above
    LINALG::TMatrix<FAD,3,nnode*dofpn> N_x;
    LINALG::TMatrix<FAD,3,nnode*dofpn> N_xx;

    //Matrices for N_i,xi and N_i,xixi. 2*nnode due to hermite shapefunctions
    LINALG::TMatrix<FAD,1,2*nnode> N_i_x;
    LINALG::TMatrix<FAD,1,2*nnode> N_i_xx;

    //stiffness due to tension and bending
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> R_tension;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> R_bending;

    //internal force due to tension and bending
    LINALG::TMatrix<FAD,nnode*dofpn,1> Res_tension;
    LINALG::TMatrix<FAD,nnode*dofpn,1> Res_bending;

    //algebraic operations
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilded;
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilde_xd;
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilde_xxd;
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilde_auxd;

    LINALG::TMatrix<FAD,1,nnode*dofpn> dTNTilde_x;
    LINALG::TMatrix<FAD,1,nnode*dofpn> dTNTilde_xx;
    LINALG::TMatrix<FAD,1,nnode*dofpn> dTNTilde_aux;

    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xddTNTilde_x;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xddTNTilde_aux;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_auxddTNTilde_x;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xxddTNTilde_x;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xddTNTilde_xx;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_auxddTNTilde_aux;

    //first of all we get the material law
    Teuchos::RCP<const MAT::Material> currmat = Material();
    double ym = 0;
    //Uncomment the next line for the dynamic case: so far only the static case is implemented
    //double density = 0;

    //assignment of material parameters; only St.Venant material is accepted for this beam
    switch(currmat->MaterialType())
    {
      case INPAR::MAT::m_stvenant:// only linear elastic material supported
      {
        const MAT::StVenantKirchhoff* actmat = static_cast<const MAT::StVenantKirchhoff*>(currmat.get());
        ym = actmat->Youngs();
        //Uncomment the next line for the dynamic case: so far only the static case is implemented
        //density = actmat->Density();
      }
      break;
      default:
      dserror("unknown or improper type of material law");
      break;
    }

    //TODO: The integration rule should be set via input parameter and not hard coded as here
    //Get integrationpoints for exact integration
    DRT::UTILS::IntegrationPoints1D gausspoints = DRT::UTILS::IntegrationPoints1D(DRT::UTILS::mygaussruleeb);

    //Get DiscretizationType of beam element
    const DRT::Element::DiscretizationType distype = Shape();

    //update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
    for (int node = 0 ; node < nnode ; node++)
    {
      for (int dof = 0 ; dof < dofpn ; dof++)
      {
        if(dof < 3)
        {
          //position of nodes
          disp_totlag[node*dofpn + dof] = Nodes()[node]->X()[dof] + disp[node*dofpn + dof];
          disp_totlag[node*dofpn + dof].diff(node*dofpn + dof,nnode*dofpn);
        }
        else if(dof>=3)
        {
          //tangent at nodes
          disp_totlag[node*dofpn + dof] = Tref_[node](dof-3) + disp[node*dofpn + dof];
          disp_totlag[node*dofpn + dof].diff(node*dofpn + dof,nnode*dofpn);
        }
      }
    } //for (int node = 0 ; node < nnode ; node++)

    //Loop through all GP and calculate their contribution to the internal forcevector and stiffnessmatrix
    for(int numgp=0; numgp < gausspoints.nquad; numgp++)
    {
      //all matrices and scalars are set to zero again!!!
      //factors for stiffness assembly
      FAD dTNTilded  = 0.0;
      FAD dTNTilde_xd = 0.0;
      FAD dTNTilde_xxd = 0.0;

      //initialize all matrices
      NTilde.Clear();
      NTilde_x.Clear();
      NTilde_xx.Clear();
      NTilde_aux.Clear();

      N_x.Clear();
      N_xx.Clear();

      R_tension.Clear();
      R_bending.Clear();

      Res_tension.Clear();
      Res_bending.Clear();

      N_i_x.Clear();
      N_i_xx.Clear();

      NTilded.Clear();
      NTilde_xd.Clear();
      NTilde_xxd.Clear();
      NTilde_auxd.Clear();

      dTNTilde_x.Clear();
      dTNTilde_xx.Clear();
      dTNTilde_aux.Clear();

      NTilde_xddTNTilde_x.Clear();
      NTilde_xddTNTilde_aux.Clear();
      NTilde_auxddTNTilde_x.Clear();
      NTilde_xxddTNTilde_x.Clear();
      NTilde_xddTNTilde_xx.Clear();
      NTilde_auxddTNTilde_aux.Clear();

      //Get location and weight of GP in parameter space
      const double xi = gausspoints.qxg[numgp][0];
      const double wgt = gausspoints.qwgt[numgp];

      //Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,xi,jacobi_*2.0,distype);
      DRT::UTILS::shape_function_hermite_1D_deriv2(N_i_xx,xi,jacobi_*2.0,distype);

      //assemble test and trial functions
      for (int r=0; r<3; ++r)
      {
        for (int d=0; d<4; ++d)
        {
          //include jacobi factor due to coordinate transformation from local in global system
          N_x(r,r+3*d) = N_i_x(d)/jacobi_;
          N_xx(r,r+3*d) = N_i_xx(d)/pow(jacobi_,2.0);
        } //for (int d=0; d<4; ++d)
      } //for (int r=0; r<3; ++r)

      //create matrices to help assemble the stiffness matrix and internal force vector:: NTilde_x = N'^T * N'; NTilde_xx = N''^T * N''; NTilde = N'^T * N''
      NTilde_x.MultiplyTN(N_x,N_x);
      NTilde_xx.MultiplyTN(N_xx,N_xx);
      NTilde.MultiplyTN(N_x,N_xx);

      //NTilde_aux = N + N^T
      NTilde_aux = NTilde;
      NTilde_aux.UpdateT(1.0, NTilde,1.0);

      //calculate factors
      //row
      for (int i=0 ; i < dofpn*nnode ; i++)
      {
        //column
        for (int j=0 ; j < dofpn*nnode ; j++)
        {
          NTilded(i)     += NTilde(i,j)*disp_totlag[j];
          NTilde_xd(i)    += NTilde_x(i,j)*disp_totlag[j];
          NTilde_xxd(i)    += NTilde_xx(i,j)*disp_totlag[j];
          NTilde_auxd(i) += NTilde_aux(i,j)*disp_totlag[j];

          dTNTilde_x(i)    += disp_totlag[j]*NTilde_x(j,i);
          dTNTilde_xx(i)    += disp_totlag[j]*NTilde_xx(j,i);
          dTNTilde_aux(i) += disp_totlag[j]*NTilde_aux(j,i);
        } //for (int j=0 ; j < dofpn*nnode ; j++)

        dTNTilded  += disp_totlag[i] * NTilded(i);
        dTNTilde_xd += disp_totlag[i] * NTilde_xd(i);
        dTNTilde_xxd += disp_totlag[i] * NTilde_xxd(i);
      } //for (int i=0 ; i < dofpn*nnode ; i++)

      //calculate factors
      //row
      for (int i=0 ; i < dofpn*nnode ; i++)
      {
        //column
        for (int j=0 ; j < dofpn*nnode ; j++)
        {
          NTilde_xddTNTilde_x(j,i)       = NTilde_xd(j)*dTNTilde_x(i);
          NTilde_xddTNTilde_aux(j,i)    = NTilde_xd(j)*dTNTilde_aux(i);
          NTilde_auxddTNTilde_x(j,i)    = NTilde_auxd(j)*dTNTilde_x(i);
          NTilde_xxddTNTilde_x(j,i)       = NTilde_xxd(j)*dTNTilde_x(i);
          NTilde_xddTNTilde_xx(j,i)       = NTilde_xd(j)*dTNTilde_xx(i);
          NTilde_auxddTNTilde_aux(j,i) = NTilde_auxd(j)*dTNTilde_aux(i);
        } //for (int j=0 ; j < dofpn*nnode ; j++)
      } //for (int i=0 ; i < dofpn*nnode ; i++)

      //assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
      //assemble parts from tension
      R_tension = NTilde_x;
      R_tension.Scale(1.0 - 1.0/pow(dTNTilde_xd,0.5));
      R_tension.Update(1.0 / pow(dTNTilde_xd,1.5),NTilde_xddTNTilde_x,1.0);

      R_tension.Scale(ym * crosssec_ * jacobi_ * wgt);

      //assemble parts from bending
      R_bending.Update(-dTNTilde_xxd,NTilde_x,1.0);
      R_bending.Update(1.0,NTilde_xx,1.0);
      R_bending.Update(- 2.0 , NTilde_xddTNTilde_xx , 1.0);

      R_bending.Scale(ym * Izz_ * wgt * jacobi_);

      //assemble internal force vector f_internal / Res in thesis Meier
      //assemble parts from tension
      Res_tension = NTilde_xd;
      Res_tension.Scale(1.0 - 1.0 /pow(dTNTilde_xd,0.5));

      Res_tension.Scale(ym * crosssec_ * jacobi_ * wgt);

      //assemble parts from bending
      Res_bending.Update(-dTNTilde_xxd,NTilde_xd,1.0);
      Res_bending.Update(1.0 ,NTilde_xxd,1.0);
      Res_bending.Scale(ym * Izz_ * jacobi_ * wgt);

      //shifting values from fixed size vector to epetra vector *force
      for(int i = 0; i < dofpn*nnode; i++)
      {
          force_check(i,0) += Res_tension(i) ;
          force_check(i,0) += Res_bending(i) ;
      }

      //shifting values from fixed size matrix to epetra matrix *stiffmatrix
      for(int i = 0; i < dofpn*nnode; i++)
      {
        for(int j = 0; j < dofpn*nnode; j++)
        {
          stiffmatrix_check(i,j) = force_check(i,0).dx(j) ;
        }
      } //for(int i = 0; i < dofpn*nnode; i++)
    } //for(int numgp=0; numgp < gausspoints.nquad; numgp++)

    Epetra_SerialDenseMatrix stiff_relerr;
    stiff_relerr.Shape(12,12);

    for(int line=0; line<12; line++)
    {
      for(int col=0; col<12; col++)
      {
        stiff_relerr(line,col)= fabs(  (    pow(stiffmatrix_check(line,col),2) - pow( (*stiffmatrix)(line,col),2 )    )/(  (  (*stiffmatrix)(line,col) + stiffmatrix_check(line,col)  ) * (*stiffmatrix)(line,col)  )  );

        //suppressing small entries whose effect is only confusing and NaN entires (which arise due to zero entries)
        //if ( fabs( stiff_relerr(line,col) ) < h_rel*50 || isnan( stiff_relerr(line,col)) || elemat1(line,col) == 0) //isnan = is not a number
        if ( fabs( stiff_relerr(line,col) ) < 1.0e-15 || isnan( stiff_relerr(line,col)) || (*stiffmatrix)(line,col) == 0) //isnan = is not a number
          stiff_relerr(line,col) = 0;
      } //for(int col=0; col<3*nnode; col++)
    } //for(int line=0; line<3*nnode; line++)


    std::cout<<"\n\n original stiffness matrix: "<< endl;
    for(int i = 0; i< 12; i++)
    {
      for(int j = 0; j< 12; j++)
      {
        cout << std::setw(9) << std::setprecision(4) << std::scientific << (*stiffmatrix)(i,j);
      }
      cout<<endl;
    }

    std::cout<<"\n\n analytical stiffness matrix: "<< endl;
    for(int i = 0; i< 12; i++)
    {
      for(int j = 0; j< 12; j++)
      {
        cout << std::setw(9) << std::setprecision(4) << std::scientific << (stiffmatrix_check)(i,j);
      }
      cout<<endl;
    }

    //std::cout<<"\n\n FAD stiffness matrix"<< stiffmatrix_check;
    std::cout<<"\n\n rel error of stiffness matrix"<< stiff_relerr;
    std::cout<<"Force_FAD: "<< force_check << endl;
    std::cout<<"Force_original: "<< *force << endl;
  }
  #else
  {
    //see also so_nstet_nodalstrain.cpp, so_nstet.H, autodiff.cpp and autodiff.H
    //FAD calculated stiff matrix for validation purposes
    Epetra_SerialDenseMatrix stiffmatrix_check;
    LINALG::TMatrix<FAD,12,1> force_check;

    //reshape stiffmatrix_check
    stiffmatrix_check.Shape(12,12);

    //dimensions of freedom per node
    const int dofpn = 6;

    //number of nodes fixed for these element
    const int nnode = 2;

    //matrix for current positions and tangents
    std::vector<FAD> disp_totlag(nnode*dofpn,0.0);

    //abbreviated matrices for clearness
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde;
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde_x;
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde_xx;
    LINALG::TMatrix<FAD,dofpn*nnode,dofpn*nnode> NTilde_aux;

    //matrices helping to assemble above
    LINALG::TMatrix<FAD,3,nnode*dofpn> N_x;
    LINALG::TMatrix<FAD,3,nnode*dofpn> N_xx;

    //Matrices for N_i,xi and N_i,xixi. 2*nnode due to hermite shapefunctions
    LINALG::TMatrix<FAD,1,2*nnode> N_i_x;
    LINALG::TMatrix<FAD,1,2*nnode> N_i_xx;

    //stiffness due to tension and bending
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> R_tension;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> R_bending;

    //internal force due to tension and bending
    LINALG::TMatrix<FAD,nnode*dofpn,1> Res_tension;
    LINALG::TMatrix<FAD,nnode*dofpn,1> Res_bending;

    //algebraic operations
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilded;
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilde_xd;
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilde_xxd;
    LINALG::TMatrix<FAD,nnode*dofpn,1> NTilde_auxd;

    LINALG::TMatrix<FAD,1,nnode*dofpn> dTNTilde_x;
    LINALG::TMatrix<FAD,1,nnode*dofpn> dTNTilde_xx;
    LINALG::TMatrix<FAD,1,nnode*dofpn> dTNTilde_aux;

    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xddTNTilde_x;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xddTNTilde_aux;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_auxddTNTilde_x;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xxddTNTilde_x;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_xddTNTilde_xx;
    LINALG::TMatrix<FAD,nnode*dofpn,nnode*dofpn> NTilde_auxddTNTilde_aux;

    //first of all we get the material law
    Teuchos::RCP<const MAT::Material> currmat = Material();
    double ym = 0;
    //Uncomment the next line for the dynamic case: so far only the static case is implemented
    //double density = 0;

    //assignment of material parameters; only St.Venant material is accepted for this beam
    switch(currmat->MaterialType())
    {
      case INPAR::MAT::m_stvenant:// only linear elastic material supported
      {
        const MAT::StVenantKirchhoff* actmat = static_cast<const MAT::StVenantKirchhoff*>(currmat.get());
        ym = actmat->Youngs();
        //Uncomment the next line for the dynamic case: so far only the static case is implemented
        //density = actmat->Density();
      }
      break;
      default:
      dserror("unknown or improper type of material law");
      break;
    }

    //TODO: The integration rule should be set via input parameter and not hard coded as here
    //Get integrationpoints for exact integration
    DRT::UTILS::IntegrationPoints1D gausspoints = DRT::UTILS::IntegrationPoints1D(DRT::UTILS::mygaussruleeb);

    //Get DiscretizationType of beam element
    const DRT::Element::DiscretizationType distype = Shape();

    //update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
    for (int node = 0 ; node < nnode ; node++)
    {
      for (int dof = 0 ; dof < dofpn ; dof++)
      {
        if(dof < 3)
        {
          //position of nodes
          disp_totlag[node*dofpn + dof] = Nodes()[node]->X()[dof] + disp[node*dofpn + dof];
          disp_totlag[node*dofpn + dof].diff(node*dofpn + dof,nnode*dofpn);
        }
        else if(dof>=3)
        {
          //tangent at nodes
          disp_totlag[node*dofpn + dof] = Tref_[node](dof-3) + disp[node*dofpn + dof];
          disp_totlag[node*dofpn + dof].diff(node*dofpn + dof,nnode*dofpn);
        }
      }
    } //for (int node = 0 ; node < nnode ; node++)

    //Loop through all GP and calculate their contribution to the internal forcevector and stiffnessmatrix
    for(int numgp=0; numgp < gausspoints.nquad; numgp++)
    {
      //all matrices and scalars are set to zero again!!!
      //factors for stiffness assembly
      FAD dTNTilded  = 0.0;
      FAD dTNTilde_xd = 0.0;
      FAD dTNTilde_xxd = 0.0;

      //initialize all matrices
      NTilde.Clear();
      NTilde_x.Clear();
      NTilde_xx.Clear();
      NTilde_aux.Clear();

      N_x.Clear();
      N_xx.Clear();

      R_tension.Clear();
      R_bending.Clear();

      Res_tension.Clear();
      Res_bending.Clear();

      N_i_x.Clear();
      N_i_xx.Clear();

      NTilded.Clear();
      NTilde_xd.Clear();
      NTilde_xxd.Clear();
      NTilde_auxd.Clear();

      dTNTilde_x.Clear();
      dTNTilde_xx.Clear();
      dTNTilde_aux.Clear();

      NTilde_xddTNTilde_x.Clear();
      NTilde_xddTNTilde_aux.Clear();
      NTilde_auxddTNTilde_x.Clear();
      NTilde_xxddTNTilde_x.Clear();
      NTilde_xddTNTilde_xx.Clear();
      NTilde_auxddTNTilde_aux.Clear();

      //Get location and weight of GP in parameter space
      const double xi = gausspoints.qxg[numgp][0];
      const double wgt = gausspoints.qwgt[numgp];

      //Get hermite derivatives N'xi and N''xi (jacobi_*2.0 is length of the element)
      DRT::UTILS::shape_function_hermite_1D_deriv1(N_i_x,xi,jacobi_*2.0,distype);
      DRT::UTILS::shape_function_hermite_1D_deriv2(N_i_xx,xi,jacobi_*2.0,distype);

      //assemble test and trial functions
      for (int r=0; r<3; ++r)
      {
        for (int d=0; d<4; ++d)
        {
          //include jacobi factor due to coordinate transformation from local in global system
          N_x(r,r+3*d) = N_i_x(d)/jacobi_;
          N_xx(r,r+3*d) = N_i_xx(d)/pow(jacobi_,2.0);
        } //for (int d=0; d<4; ++d)
      } //for (int r=0; r<3; ++r)

      //create matrices to help assemble the stiffness matrix and internal force vector:: NTilde_x = N'^T * N'; NTilde_xx = N''^T * N''; NTilde = N'^T * N''
      NTilde_x.MultiplyTN(N_x,N_x);
      NTilde_xx.MultiplyTN(N_xx,N_xx);
      NTilde.MultiplyTN(N_x,N_xx);

      //NTilde_aux = N + N^T
      NTilde_aux = NTilde;
      NTilde_aux.UpdateT(1.0, NTilde,1.0);

      //calculate factors
      //row
      for (int i=0 ; i < dofpn*nnode ; i++)
      {
        //column
        for (int j=0 ; j < dofpn*nnode ; j++)
        {
          NTilded(i)     += NTilde(i,j)*disp_totlag[j];
          NTilde_xd(i)    += NTilde_x(i,j)*disp_totlag[j];
          NTilde_xxd(i)    += NTilde_xx(i,j)*disp_totlag[j];
          NTilde_auxd(i) += NTilde_aux(i,j)*disp_totlag[j];

          dTNTilde_x(i)    += disp_totlag[j]*NTilde_x(j,i);
          dTNTilde_xx(i)    += disp_totlag[j]*NTilde_xx(j,i);
          dTNTilde_aux(i) += disp_totlag[j]*NTilde_aux(j,i);
        } //for (int j=0 ; j < dofpn*nnode ; j++)

        dTNTilded  += disp_totlag[i] * NTilded(i);
        dTNTilde_xd += disp_totlag[i] * NTilde_xd(i);
        dTNTilde_xxd += disp_totlag[i] * NTilde_xxd(i);
      } //for (int i=0 ; i < dofpn*nnode ; i++)

      //calculate factors
      //row
      for (int i=0 ; i < dofpn*nnode ; i++)
      {
        //column
        for (int j=0 ; j < dofpn*nnode ; j++)
        {
          NTilde_xddTNTilde_x(j,i)       = NTilde_xd(j)*dTNTilde_x(i);
          NTilde_xddTNTilde_aux(j,i)    = NTilde_xd(j)*dTNTilde_aux(i);
          NTilde_auxddTNTilde_x(j,i)    = NTilde_auxd(j)*dTNTilde_x(i);
          NTilde_xxddTNTilde_x(j,i)       = NTilde_xxd(j)*dTNTilde_x(i);
          NTilde_xddTNTilde_xx(j,i)       = NTilde_xd(j)*dTNTilde_xx(i);
          NTilde_auxddTNTilde_aux(j,i) = NTilde_auxd(j)*dTNTilde_aux(i);
        } //for (int j=0 ; j < dofpn*nnode ; j++)
      } //for (int i=0 ; i < dofpn*nnode ; i++)

      //assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
      //assemble parts from tension
      R_tension = NTilde_x;
      R_tension.Scale(1.0 - 1.0/pow(dTNTilde_xd,0.5));
      R_tension.Update(1.0 / pow(dTNTilde_xd,1.5),NTilde_xddTNTilde_x,1.0);

      R_tension.Scale(ym * crosssec_ * jacobi_ * wgt);

      //assemble parts from bending
      R_bending = NTilde_x;
      R_bending.Scale(2.0 * pow(dTNTilded,2.0) / pow(dTNTilde_xd,3.0));
      R_bending.Update(-dTNTilde_xxd/pow(dTNTilde_xd,2.0),NTilde_x,1.0);
      R_bending.Update(-dTNTilded/pow(dTNTilde_xd,2.0),NTilde_aux,1.0);
      R_bending.Update(1.0/dTNTilde_xd,NTilde_xx,1.0);
      R_bending.Update(-12.0 * pow(dTNTilded,2.0)/pow(dTNTilde_xd,4.0),NTilde_xddTNTilde_x,1.0);
      R_bending.Update(4.0 * dTNTilded / pow(dTNTilde_xd,3.0) , NTilde_xddTNTilde_aux , 1.0);
      R_bending.Update(4.0 * dTNTilded / pow(dTNTilde_xd,3.0) , NTilde_auxddTNTilde_x , 1.0);
      R_bending.Update(4.0 * dTNTilde_xxd / pow(dTNTilde_xd,3.0) , NTilde_xddTNTilde_x , 1.0);
      R_bending.Update(- 2.0 / pow(dTNTilde_xd,2.0) , NTilde_xxddTNTilde_x , 1.0);
      R_bending.Update(- 2.0 / pow(dTNTilde_xd,2.0) , NTilde_xddTNTilde_xx , 1.0);
      R_bending.Update(- 1.0 / pow(dTNTilde_xd,2.0) , NTilde_auxddTNTilde_aux , 1.0);

      R_bending.Scale(ym * Izz_ * jacobi_ * wgt);

      //assemble internal force vector f_internal / Res in thesis Meier
      //assemble parts from tension
      Res_tension = NTilde_xd;
      Res_tension.Scale(1.0 - 1.0 /pow(dTNTilde_xd,0.5));

      Res_tension.Scale(ym * crosssec_ * jacobi_ * wgt);

      //assemble parts from bending
      Res_bending = NTilde_xd;
      Res_bending.Scale(2.0 * pow(dTNTilded,2.0)/pow(dTNTilde_xd,3.0));
      Res_bending.Update(-dTNTilde_xxd / pow(dTNTilde_xd,2.0),NTilde_xd,1.0);
      Res_bending.Update(-dTNTilded / pow(dTNTilde_xd,2.0),NTilde_auxd,1.0);
      Res_bending.Update(1.0 / dTNTilde_xd,NTilde_xxd,1.0);

      Res_bending.Scale(ym * Izz_ * jacobi_ * wgt);

      cout << "Resbending: " << Res_bending << endl;
      cout << "Restension: " << Res_tension << endl;

      //shifting values from fixed size vector to epetra vector *force
      for(int i = 0; i < dofpn*nnode; i++)
      {
          force_check(i,0) += Res_tension(i) ;
          force_check(i,0) += Res_bending(i) ;
      }

      //shifting values from fixed size matrix to epetra matrix *stiffmatrix
      for(int i = 0; i < dofpn*nnode; i++)
      {
        for(int j = 0; j < dofpn*nnode; j++)
        {
          stiffmatrix_check(i,j) = force_check(i,0).dx(j) ;
        }
      } //for(int i = 0; i < dofpn*nnode; i++)
    } //for(int numgp=0; numgp < gausspoints.nquad; numgp++)

    Epetra_SerialDenseMatrix stiff_relerr;
    stiff_relerr.Shape(12,12);

    for(int line=0; line<12; line++)
    {
      for(int col=0; col<12; col++)
      {
        stiff_relerr(line,col)= fabs(  (    pow(stiffmatrix_check(line,col),2) - pow( (*stiffmatrix)(line,col),2 )    )/(  (  (*stiffmatrix)(line,col) + stiffmatrix_check(line,col)  ) * (*stiffmatrix)(line,col)  )  );

        //suppressing small entries whose effect is only confusing and NaN entires (which arise due to zero entries)
        //if ( fabs( stiff_relerr(line,col) ) < h_rel*50 || isnan( stiff_relerr(line,col)) || elemat1(line,col) == 0) //isnan = is not a number
        if ( fabs( stiff_relerr(line,col) ) < 1.0e-15 || isnan( stiff_relerr(line,col)) || (*stiffmatrix)(line,col) == 0) //isnan = is not a number
          stiff_relerr(line,col) = 0;
      } //for(int col=0; col<3*nnode; col++)
    } //for(int line=0; line<3*nnode; line++)


    std::cout<<"\n\n original stiffness matrix: "<< endl;
    for(int i = 0; i< 12; i++)
    {
      for(int j = 0; j< 12; j++)
      {
        cout << std::setw(9) << std::setprecision(4) << std::scientific << (*stiffmatrix)(i,j);
      }
      cout<<endl;
    }

    std::cout<<"\n\n analytical stiffness matrix: "<< endl;
    for(int i = 0; i< 12; i++)
    {
      for(int j = 0; j< 12; j++)
      {
        cout << std::setw(9) << std::setprecision(4) << std::scientific << (stiffmatrix_check)(i,j);
      }
      cout<<endl;
    }

    std::cout<<"\n\n FAD stiffness matrix"<< stiffmatrix_check;
    std::cout<<"\n\n rel error of stiffness matrix"<< stiff_relerr;
    std::cout<<"Force: "<< force_check << endl;
  }
  #endif
}

void DRT::ELEMENTS::Beam3eb::FADCheckNeumann(Teuchos::ParameterList& params,
                                             DRT::Discretization& discretization,
                                             DRT::Condition& condition,
                                             std::vector<int>& lm,
                                             Epetra_SerialDenseVector& elevec1,
                                             Epetra_SerialDenseMatrix* elemat1)
{
#if NODALDOFS == 3
    dserror("FADChecks are not implemented for the case NODALDOFS = 3!!!");
#endif
  //FAD calculated stiff matrix for validation purposes
  Epetra_SerialDenseMatrix stiffmatrix_check;

  const int nnode=2;
  const int dofpn=6;

  LINALG::TMatrix<FAD,dofpn*nnode,1> force_check;

  //reshape stiffmatrix_check
  stiffmatrix_check.Shape((dofpn)*nnode,(dofpn)*nnode);

  for (int i=0; i<(dofpn)*nnode; i++)
  {
    for (int j=0; j<(dofpn)*nnode; j++)
    {
      stiffmatrix_check(i,j)=0;
    }
    force_check(i,0)=0;
  }

  //get element displacements
  RCP<const Epetra_Vector> disp = discretization.GetState("displacement new");
  if (disp==Teuchos::null) dserror("Cannot get state vector 'displacement new'");
  std::vector<double> mydisp(lm.size());
  DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);

  //matrix for current positions and tangents
  std::vector<FAD> disp_totlag((dofpn)*nnode,0.0);

  for (int i=0; i<(dofpn)*nnode; i++)
  {
    disp_totlag[i]=mydisp[i];
    disp_totlag[i].diff(i,(dofpn)*nnode);
  }

  // find out whether we will use a time curve
  bool usetime = true;
  const double time = params.get("total time",-1.0);
  if (time<0.0) usetime = false;

  // find out whether we will use a time curve and get the factor
  const std::vector<int>* curve = condition.Get<std::vector<int> >("curve");

  int curvenum = -1;

  // number of the load curve related with a specific line Neumann condition called
  if (curve) curvenum = (*curve)[0];

  // amplitude of load curve at current time called
  double curvefac = 1.0;

  if (curvenum>=0 && usetime)
    curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);

  // get values and switches from the condition

  // onoff is related to the first 6 flags of a line Neumann condition in the input file;
  // value 1 for flag i says that condition is active for i-th degree of freedom
  const std::vector<int>* onoff = condition.Get<std::vector<int> >("onoff");
  // val is related to the 6 "val" fields after the onoff flags of the Neumann condition

  // in the input file; val gives the values of the force as a multiple of the prescribed load curve
  const std::vector<double>* val = condition.Get<std::vector<double> >("val");

  //find out which node is correct
  const std::vector< int > * nodeids = condition.Nodes();

  //if a point neumann condition needs to be linearized
  if(condition.Type() == DRT::Condition::PointNeumannEB)
  {
    //find out local node number --> this is done since the first element of a neumann point condition is used for this function
    //in this case we do not know whether it is the left or the right node.
    int insert = -1;

    if((*nodeids)[0] == Nodes()[0]->Id())
      insert = 0;
    else if((*nodeids)[0] == Nodes()[1]->Id())
      insert = 1;

    if (insert == -1)
      dserror("\nNode could not be found on nodemap!\n");

    //add forces to Res_external according to (5.56)
    for(int i = 0; i < 3 ; i++)
    {
      force_check(insert*(dofpn) + i) += (*onoff)[i]*(*val)[i]*curvefac;
    }

    //matrix for current tangent, moment at node and crossproduct
    LINALG::TMatrix<FAD,3,1> tangent;
    LINALG::TMatrix<FAD,3,1> crossproduct;
    LINALG::TMatrix<FAD,3,1> moment;
    LINALG::TMatrix<FAD,3,3> spinmatrix;

    //clear all matrices
    tangent.Clear();
    crossproduct.Clear();
    moment.Clear();
    spinmatrix.Clear();

    //assemble current tangent and moment at node
    for (int dof = 3 ; dof < 6 ; dof++)
    {
      //get current tangent at nodes
      tangent(dof-3) = Tref_[insert](dof-3) + disp_totlag[insert*(dofpn) + dof];
      moment(dof-3) = (*onoff)[dof]*(*val)[dof]*curvefac;
    }

    FAD abs_tangent = 0.0;

    for (int i=0; i<3;  i++)
     {
      abs_tangent += std::pow(tangent(i,0),2);
     }
    abs_tangent=pow(abs_tangent,0.5);

    //computespin = S ( tangent ) using the spinmatrix in namespace largerotations
    LARGEROTATIONS::computespin<FAD>(spinmatrix,tangent);

    //matrixoperation crossproduct = r' x m and rxrxTNx = I/|r'|-r'r'T/|r'|^3
    for(int i=0; i<3; i++)
    {
      for(int j=0; j<3; j++)
      {
        crossproduct(i,0) += spinmatrix(i,j) * moment(j);
      }
    }

    //add moments to Res_external according to (5.56)
    for(int i = 3; i < 6 ; i++)
    {
      force_check(insert*(dofpn) + i) -= crossproduct(i-3,0) / pow(abs_tangent,2.0);
    }

    //assembly for stiffnessmatrix
    LINALG::TMatrix<FAD,3,3> crossxtangent;

    crossxtangent.Clear();

    //perform matrix operation
    for(int i=0; i<3; i++)
    {
      for(int j=0; j<3; j++)
      {
        crossxtangent(i,j) = crossproduct(i,0) * tangent(j);
      }
    }

    spinmatrix.Clear();

    //spinmatrix = S ( m )
    LARGEROTATIONS::computespin<FAD>(spinmatrix,moment);

    for(int i = 0; i < (dofpn)*nnode; i++)
    {
      for(int j = 0; j < (dofpn)*nnode; j++)
      {
        stiffmatrix_check(i,j) = -force_check(i).dx(j) ;
      }
    }
  }

  //if a line neumann condition needs to be linearized
  else if(condition.Type() == DRT::Condition::LineNeumann)
  {
  }

  Epetra_SerialDenseMatrix stiff_relerr;
  stiff_relerr.Shape((dofpn+1)*nnode,(dofpn+1)*nnode);

  for(int line=0; line<(dofpn)*nnode; line++)
  {
    for(int col=0; col<(dofpn)*nnode; col++)
    {
      stiff_relerr(line,col)= fabs((pow(stiffmatrix_check(line,col),2) - pow((*elemat1)(line,col),2))/(((*elemat1)(line,col) + stiffmatrix_check(line,col)) * (*elemat1)(line,col)));

      //suppressing small entries whose effect is only confusing and NaN entires (which arise due to zero entries)
      if ( fabs( stiff_relerr(line,col) ) < 1.0e-10 || isnan( stiff_relerr(line,col)) || (*elemat1)(line,col) == 0) //isnan = is not a number
        stiff_relerr(line,col) = 0;
    } //for(int col=0; col<3*nnode; col++)
  } //for(int line=0; line<3*nnode; line++)

  Epetra_SerialDenseMatrix force_relerr;
  force_relerr.Shape((dofpn)*nnode,1);
  for (int line=0; line<(dofpn)*nnode; line++)
  {
    force_relerr(line,0)= fabs(pow(force_check(line,0).val(),2.0) - pow((elevec1)(line),2.0 ));
  }
  std::cout<<"\n\n Rel error stiffness matrix Neumann: "<< stiff_relerr << endl;

}

//***************************************************************************************
//End: Methods for FAD Check
//***************************************************************************************


//***************************************************************************************
//Methods for arbitrary precision calculation
//***************************************************************************************
#ifdef PRECISION

  void DRT::ELEMENTS::Beam3eb::eb_nlnstiffmassprec( LINALG::TMatrix<cl_F,12,1>* displocal,
                                                    LINALG::TMatrix<cl_F,12,12>* stifflocal,
                                                    LINALG::TMatrix<cl_F,12,1>* reslocal,
                                                    LINALG::TMatrix<cl_F,6,1>* xreflocal)
  {



#if NODALDOFS == 3
    dserror("High precision calculation is not implemented for the case NODALDOFS = 3!!!");
#endif

        std::vector<double> forcetest2(12);
        for (int i=0; i<12;i++)
        {
          forcetest2[i]=0.0;
        }

        //dimensions of freedom per node
        const int dofpn = 6;

        //number of nodes fixed for these element
        const int nnode = 2;

        //matrix for current positions and tangents
        std::vector<cl_F> disp_totlag(nnode*dofpn);

        LINALG::TMatrix<cl_F,3,1> r_;
        LINALG::TMatrix<cl_F,3,1> r_x;
        LINALG::TMatrix<cl_F,3,1> r_xx;

        LINALG::TMatrix<cl_F,3,1> f1;
        LINALG::TMatrix<cl_F,3,1> f2;
        LINALG::TMatrix<cl_F,3,1> n1;

        cl_F rxrxx;
        cl_F rxxrxx;
        cl_F rxrx;
        cl_F tension;

        LINALG::TMatrix<cl_F,dofpn*nnode,dofpn*nnode> NTilde;
        LINALG::TMatrix<cl_F,dofpn*nnode,dofpn*nnode> NTildex;
        LINALG::TMatrix<cl_F,dofpn*nnode,dofpn*nnode> NTildexx;

        LINALG::TMatrix<cl_F,dofpn*nnode,1> NxTrx;
        LINALG::TMatrix<cl_F,dofpn*nnode,1> NxTrxx;
        LINALG::TMatrix<cl_F,dofpn*nnode,1> NxxTrx;
        LINALG::TMatrix<cl_F,dofpn*nnode,1> NxxTrxx;

        LINALG::TMatrix<cl_F,dofpn*nnode,dofpn*nnode> M1;
        LINALG::TMatrix<cl_F,dofpn*nnode,dofpn*nnode> M2;
        LINALG::TMatrix<cl_F,dofpn*nnode,dofpn*nnode> M3;
        LINALG::TMatrix<cl_F,dofpn*nnode,dofpn*nnode> NxTrxrxTNx;

        //Matrices for N_i,xi and N_i,xixi. 2*nnode due to hermite shapefunctions
        LINALG::TMatrix<cl_F,1,2*nnode> N_i;
        LINALG::TMatrix<cl_F,1,2*nnode> N_i_x;
        LINALG::TMatrix<cl_F,1,2*nnode> N_i_xx;

        LINALG::TMatrix<cl_F,3,nnode*dofpn> N_x;
        LINALG::TMatrix<cl_F,3,nnode*dofpn> N_xx;

        //stiffness due to tension and bending
        LINALG::TMatrix<cl_F,nnode*dofpn,nnode*dofpn> R_tension;
        LINALG::TMatrix<cl_F,nnode*dofpn,nnode*dofpn> R_bending;

        //internal force due to tension and bending
        LINALG::TMatrix<cl_F,nnode*dofpn,1> Res_tension;
        LINALG::TMatrix<cl_F,nnode*dofpn,1> Res_bending;

        //clear disp_totlag vector before assembly
        disp_totlag.clear();

        //update displacement vector /d in thesis Meier d = [ r1 t1 r2 t2]
        for (int node = 0 ; node < nnode ; node++)
        {
          for (int dof = 0 ; dof < dofpn ; dof++)
          {
            if(dof < 3)
            {
              disp_totlag[node*dofpn + dof] = cl_float(0,float_format(40));
              disp_totlag[node*dofpn + dof] = (*xreflocal)(3*node + dof,0) + (*displocal)(node*dofpn + dof,0);
            }
            else if(dof>=3)
            {
              //tangent at nodes
              disp_totlag[node*dofpn + dof] = cl_float(0,float_format(40));
              disp_totlag[node*dofpn + dof] = Trefprec_(dof-3) + (*displocal)(node*dofpn + dof,0);
            }
          }
        } //for (int node = 0 ; node < nnode ; node++)

        //begin: gausspoints exact
        std::vector<cl_F> xivec(6);
        std::vector<cl_F> wgtvec(6);
        xivec[0]="-0.9324695142031520278123016_40";
        xivec[5]="0.9324695142031520278123016_40";
        xivec[1]="-0.6612093864662645136613996_40";
        xivec[4]="0.6612093864662645136613996_40";
        xivec[2]="-0.2386191860831969086305017_40";
        xivec[3]="0.2386191860831969086305017_40";
        wgtvec[0]="0.171324492379170345040296_40";
        wgtvec[5]="0.171324492379170345040296_40";
        wgtvec[1]="0.360761573048138607569834_40";
        wgtvec[4]="0.360761573048138607569834_40";
        wgtvec[2]="0.467913934572691047389870_40";
        wgtvec[3]="0.467913934572691047389870_40";
        //end: gausspoints exact

        //Loop through all GP and calculate their contribution to the internal forcevector and stiffnessmatrix
        for(int numgp=0; numgp < 6; numgp++)
        {
          //all matrices and scalars are set to zero again!!!
          //factors for stiffness assembly

          for (int i=0;i<3;i++)
          {
          r_(i)=cl_float(0,float_format(40));
          r_x(i)=cl_float(0,float_format(40));
          r_xx(i)=cl_float(0,float_format(40));
          f1(i)=cl_float(0,float_format(40));
          f2(i)=cl_float(0,float_format(40));
          n1(i)=cl_float(0,float_format(40));
            for (int j=0;j<12;j++)
            {
              N_x(i,j)=cl_float(0,float_format(40));
              N_xx(i,j)=cl_float(0,float_format(40));
            }
          }

          rxrxx=cl_float(0,float_format(40));
          rxxrxx=cl_float(0,float_format(40));
          rxrx=cl_float(0,float_format(40));
          tension=cl_float(0,float_format(40));

          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              NTilde(i,j)=cl_float(0,float_format(40));
              NTildex(i,j)=cl_float(0,float_format(40));
              NTildexx(i,j)=cl_float(0,float_format(40));
              M1(i,j)=cl_float(0,float_format(40));
              M2(i,j)=cl_float(0,float_format(40));
              M3(i,j)=cl_float(0,float_format(40));
              NxTrxrxTNx(i,j)=cl_float(0,float_format(40));
              R_tension(i,j)=cl_float(0,float_format(40));
              R_bending(i,j)=cl_float(0,float_format(40));
            }
            NxTrx(i)=cl_float(0,float_format(40));
            NxTrxx(i)=cl_float(0,float_format(40));
            NxxTrx(i)=cl_float(0,float_format(40));
            NxxTrxx(i)=cl_float(0,float_format(40));
            Res_tension(i)=cl_float(0,float_format(40));
            Res_bending(i)=cl_float(0,float_format(40));
          }

          for (int i=0;i<4;i++)
          {
            N_i(i)=cl_float(0,float_format(40));
            N_i_x(i)=cl_float(0,float_format(40));
            N_i_xx(i)=cl_float(0,float_format(40));
          }

          //Get location and weight of GP in parameter space
          cl_F xi=xivec[numgp];
          cl_F wgt=wgtvec[numgp];

          //Begin: shape fuction variante für arbitrary precision

           cl_F l = cl_float(2.0,float_format(40))*jacobiprec_;
           N_i_x(0)= cl_float(0.25,float_format(40))*(- cl_float(3.0,float_format(40)) + cl_float(3.0,float_format(40)) * cl_float(expt(xi,2.0),float_format(40)));
           N_i_x(1)= l / cl_float(8.0,float_format(40)) * (- cl_float(1.0,float_format(40)) - cl_float(2.0,float_format(40)) * xi + cl_float(3.0,float_format(40)) * cl_float(expt(xi,2.0),float_format(40)));
           N_i_x(2)= cl_float(0.25,float_format(40))*(cl_float(3.0,float_format(40)) - cl_float(3.0,float_format(40)) * cl_float(expt(xi,2.0),float_format(40)));
           N_i_x(3)= l / cl_float(8.0,float_format(40)) * (- cl_float(1.0,float_format(40)) + cl_float(2.0,float_format(40)) * xi + cl_float(3.0,float_format(40)) * cl_float(expt(xi,2.0),float_format(40)));

           N_i_xx(0)= cl_float(1.5,float_format(40)) * xi;
           N_i_xx(1)= l / cl_float(8.0,float_format(40)) * (- cl_float(2.0,float_format(40)) + cl_float(6.0,float_format(40)) * xi);
           N_i_xx(2)= - cl_float(1.5,float_format(40)) * xi;
           N_i_xx(3)= l / cl_float(8.0,float_format(40)) * (cl_float(2.0,float_format(40)) + cl_float(6.0,float_format(40)) * xi);

           //end: shape fuction variante für arbitrary precision


          //calculate r' and r''
          for (int i=0 ; i < 3 ; i++)
          {
            for (int j=0; j<4; j++)
            {
              r_x(i,0)+= N_i_x(j) * disp_totlag[3*j + i];
              r_xx(i,0)+= N_i_xx(j) * disp_totlag[3*j + i];
            }
          }

          for (int i= 0; i<3; i++)
          {
            rxrxx+=r_x(i)*r_xx(i);
            rxxrxx+=r_xx(i)*r_xx(i);
            rxrx+=r_x(i)*r_x(i);
          }

          tension = cl_float(1.0,float_format(40))/jacobiprec_ - cl_float(1.0,float_format(40))/sqrt(rxrx);

          for (int i=0; i<3; ++i)
          {
            for (int j=0; j<4; ++j)
            {
              N_x(i,i+3*j) += N_i_x(j);
              N_xx(i,i+3*j) += N_i_xx(j);
              NxTrx(i+3*j)+=N_i_x(j)*r_x(i);
              NxTrxx(i+3*j)+=N_i_x(j)*r_xx(i);
              NxxTrx(i+3*j)+=N_i_xx(j)*r_x(i);
              NxxTrxx(i+3*j)+=N_i_xx(j)*r_xx(i);
            }
          }

          NTilde.MultiplyTN(N_x,N_xx);
          NTildex.MultiplyTN(N_x,N_x);
          NTildexx.MultiplyTN(N_xx,N_xx);


          for (int i= 0; i<12; i++)
          {
            for (int j= 0; j<12; j++)
            {
              M1(i,j)+= NxTrx(i)*(NxxTrx(j)+NxTrxx(j));
              M2(i,j)+= NxxTrxx(i)*NxTrx(j);
              M3(i,j)+= (NxTrxx(i)+NxxTrx(i))*(NxTrxx(j)+NxxTrx(j));
              NxTrxrxTNx(i,j)+= NxTrx(i)*NxTrx(j);
            }
          }

          //assemble internal stiffness matrix / R = d/(dd) Res in thesis Meier
          if (stifflocal != NULL)
          {
            //assemble parts from tension
            R_tension = NTildex;
            R_tension.Scale(tension);
            R_tension.Update(cl_float(1.0,float_format(40)) / sqrt(expt(rxrx,3.0)),NxTrxrxTNx,cl_float(1.0,float_format(40)));

            R_tension.Scale(Eprec_ * crosssecprec_ * wgt);

            //assemble parts from bending
            R_bending = NTildex;
            R_bending.Scale(cl_float(cl_float(2.0,float_format(40)) * expt(rxrxx,2.0) / expt(rxrx,3.0),float_format(40)));
            R_bending.Update(-rxxrxx/expt(rxrx,2.0),NTildex,cl_float(1.0,float_format(40)));
            R_bending.Update(-rxrxx/expt(rxrx,2.0),NTilde,cl_float(1.0,float_format(40)));
            R_bending.UpdateT(-rxrxx/expt(rxrx,2.0),NTilde,cl_float(1.0,float_format(40)));
            R_bending.Update(cl_float(1.0,float_format(40))/rxrx,NTildexx,cl_float(1.0,float_format(40)));
            R_bending.Update(cl_float(-cl_float(12.0,float_format(40)) * expt(rxrxx,2.0)/expt(rxrx,4.0),float_format(40)),NxTrxrxTNx,cl_float(1.0,float_format(40)));
            R_bending.Update(cl_float(4.0,float_format(40)) * rxrxx / expt(rxrx,3.0) , M1 , cl_float(1.0,float_format(40)));
            R_bending.UpdateT(cl_float(4.0,float_format(40)) * rxrxx / expt(rxrx,3.0) , M1 , cl_float(1.0,float_format(40)));
            R_bending.Update(cl_float(4.0,float_format(40)) * rxxrxx / expt(rxrx,3.0) , NxTrxrxTNx, cl_float(1.0,float_format(40)));
            R_bending.Update(-cl_float(2.0,float_format(40)) / expt(rxrx,2.0) , M2 , cl_float(1.0,float_format(40)));
            R_bending.UpdateT(-cl_float(2.0,float_format(40)) / expt(rxrx,2.0) , M2 , cl_float(1.0,float_format(40)));
            R_bending.Update(-cl_float(1.0,float_format(40)) / expt(rxrx,2.0) , M3 , cl_float(1.0,float_format(40)));

            R_bending.Scale(Eprec_ * Izzprec_ * wgt / jacobiprec_);

            //shifting values from fixed size matrix to epetra matrix *stiffmatrix
            for(int i = 0; i < dofpn*nnode; i++)
            {
              for(int j = 0; j < dofpn*nnode; j++)
              {
                //cout << "Rbending: " << R_bending(i,j) << endl;
                (*stifflocal)(i,j) += R_tension(i,j);
                (*stifflocal)(i,j) += R_bending(i,j);
                //stifftest_(i,j)+= R_bending(i,j);
              }

            } //for(int i = 0; i < dofpn*nnode; i++)

          }//if (stiffmatrix != NULL)

          for (int i= 0; i<3; i++)
          {
            f1(i)=cl_float(2.0,float_format(40))*r_x(i)*expt(rxrxx,2.0)/expt(rxrx,3.0)-(r_x(i)*rxxrxx+r_xx(i)*rxrxx)/expt(rxrx,2.0);
            f2(i)=r_xx(i)/rxrx-r_x(i)*rxrxx/expt(rxrx,2.0);
            n1(i)=r_x(i)*tension;
          }


          //assemble internal force vector f_internal / Res in thesis Meier
          if (reslocal != NULL)
          {
            for (int i=0;i<3;i++)
            {
              for (int j=0;j<4;j++)
              {
                Res_bending(j*3 + i)+= N_i_x(j)*f1(i) + N_i_xx(j)*f2(i);
                Res_tension(j*3 + i)+= N_i_x(j)*n1(i);
              }
            }
            Res_bending.Scale(Eprec_ * Izzprec_ * wgt / jacobiprec_);
            Res_tension.Scale(Eprec_ * crosssecprec_ * wgt);

            //shifting values from fixed size vector to epetra vector *force
            for(int i = 0; i < dofpn*nnode; i++)
            {
                (*reslocal)(i) += Res_tension(i) ;
                (*reslocal)(i) += Res_bending(i) ;
                //restest_(i) += Res_bending(i);
            }
          } //if (force != NULL)

        } //for(int numgp=0; numgp < gausspoints.nquad; numgp++)

  /*    const std::string fname = "stiffmatrixele.mtl";
        cout<<"Printing stiffmatrixele to file"<<endl;
        LINALG::PrintSerialDenseMatrixInMatlabFormat(fname,*stiffmatrix);*/

    return;
  } // DRT::ELEMENTS::Beam3eb::eb_nlnstiffmass

  void DRT::ELEMENTS::Beam3eb::HighPrecissionCalc()
  {
#if NODALDOFS == 3
    dserror("High precision calculation is not implemented for the case NODALDOFS = 3!!!");
#endif
    //Uncomment the following line to avoid floating point underflow --> small numbers are then rounded to zero
    //cl_inhibit_floating_point_underflow = true;

    //Input Parameters
    const cl_F RESTOL = "1.0e-35_40";
    const cl_F DISPTOL = "1.0e-35_40";
    const cl_F TOLLINSOLV = "1.0e-50_40";
    const int numele = 32;
    const int NUMLOADSTEP = 250;
    cl_F balkenlaenge = "10.0_40";
    balkenradiusprec_ = "1.0_40";
    cl_F fext = "0.0_40";
    LINALG::TMatrix<cl_F,3,1> mextvec;
    mextvec(0)="0.0_40";
    mextvec(1)="0.0_40";
    mextvec(2)="0.0_40";
    //End: Input Parameters


    //Referenzgeometrieberechnung
    const int numnode = numele+1;
    cl_F elementlaenge = "0.0_40";
    elementlaenge = balkenlaenge / numele;
    jacobiprec_ = elementlaenge / cl_F("2.0_40");
    const cl_F PIPREC = "3.1415926535897932384626433832795028841971_40";
    crosssecprec_ = cl_float(expt(balkenradiusprec_,2.0) * PIPREC, float_format(40));
    Izzprec_ = cl_float(expt(balkenradiusprec_,4.0) * PIPREC / cl_F("4.0_40"),float_format(40));
    Eprec_= "1.0_40";
    cl_F mext = Izzprec_ * Eprec_ * cl_F("2.0_40") * PIPREC / balkenlaenge;
    mextvec(2)= mext;

    LINALG::TMatrix<cl_F,numnode*3,1> xrefglobal;

    for (int i=0;i<numnode;i++)
    {
      for (int j=0;j<3;j++)
      {
        xrefglobal(3*i+j,0)=cl_F("0.0_40");
        xrefglobal(3*i,0)=-balkenlaenge/2 + i*elementlaenge;
        Trefprec_(j,0)="0.0_40";
      }
    }
    Trefprec_(0,0)="1.0_40";
    //End: Referenzgeometrieberechnung


    //Globale Groeßen definieren
    cl_F resnorm="10.0_40";
    cl_F dispnorm="10.0_40";
    cl_F linsolverrornorm="10.0_40";
    LINALG::TMatrix<cl_F,numnode*6,numnode*6> stiffglobal;
    LINALG::TMatrix<cl_F,numnode*6,1> resglobal;
    LINALG::TMatrix<cl_F,numnode*6,1> dispglobal;
    LINALG::TMatrix<cl_F,numnode*6,1> deltadispglobal;
    LINALG::TMatrix<cl_F,numnode*6,1> fextglobal;

    for (int i=0;i<6*numnode;i++)
    {
      for (int j=0;j<6*numnode;j++)
      {
        stiffglobal(i,j)=cl_F("0.0_40");
      }
      resglobal(i,0)=cl_F("0.0_40");
      dispglobal(i,0)=cl_F("0.0_40");
      fextglobal(i,0)=cl_F("0.0_40");
    }
    //Ende: Globale Groeßen definieren


    //Loadsteps
    LINALG::TMatrix<cl_F,3,1>mextvecstep;
    cl_F fextstep="0.0_40";
    for (int i=0;i<3;i++)
    {
      mextvecstep(i)="0.0_40";
    }
    for (int lastschritt=0; lastschritt<NUMLOADSTEP; lastschritt++)
    {
      cout << "Lastschritt: " << lastschritt + 1 << endl;
      fextstep=fext*cl_float((lastschritt+1),float_format(40))/cl_float(NUMLOADSTEP,float_format(40));
      for (int j=0;j<3;j++)
      {
        mextvecstep(j)=mextvec(j)*cl_float((lastschritt+1),float_format(40))/cl_float(NUMLOADSTEP,float_format(40));
      }

      cout << "begin of Newton Iteration" << endl;
      int iter=0;
      resnorm="1.0_40";

      //Newton
      while (resnorm>"1.0e-50_40")
      {
        iter++;
        LINALG::TMatrix<cl_F,12,12> stifflocal;
        LINALG::TMatrix<cl_F,12,1> reslocal;
        LINALG::TMatrix<cl_F,12,1> displocal;
        LINALG::TMatrix<cl_F,6,1> xreflocal;

        //Normen nullen
        resnorm="0.0_40";
        dispnorm="0.0_40";
        linsolverrornorm="0.0_40";
        //end: Normen nullen

        for (int i=0;i<6*numnode;i++)
        {
          for (int j=0;j<6*numnode;j++)
          {
            stiffglobal(i,j)=cl_F("0.0_40");
          }
          resglobal(i,0)=cl_F("0.0_40");
        }

        //Evaluate all elements and assemble
        for (int ele=0;ele<numele;ele++)
        {

          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              stifflocal(i,j)=cl_F("0.0_40");
            }
            reslocal(i,0)=cl_F("0.0_40");
            displocal(i,0)=cl_F("0.0_40");
            xreflocal=cl_F("0.0_40");
          }

          for (int k=0;k<12;k++)
          {
            displocal(k,0)=dispglobal(ele*6 + k ,0);
          }

          for (int k=0;k<6;k++)
          {
            xreflocal(k,0)=xrefglobal(ele*3 + k ,0);
          }


          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              stifftest_(i,j) ="0.0_40";
            }
            restest_(i)="0.0_40";
          }

          eb_nlnstiffmassprec(&displocal, &stifflocal, &reslocal, &xreflocal);

          //Uncomment the following Code block to compare the high precision stiffness matrix with the standard precision stiffness matrix
          /*
          //Begin: Compare with old stiffness
          //Uncomment the following block to compare with the original stiffness calculation
          std::vector<double> testdisp(12);
          for (int i=0;i<12;i++)
          {
            testdisp[i]=double_approx(displocal(i,0));
          }
          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              elemat1(i,j) =0;
            }
            elevec1[i]=0;
          }
          eb_nlnstiffmass(params,testdisp,testdisp,&elemat1,NULL,&elevec1);
          //End: Compare with old stiffness
          cout << "resnew: " << endl;
          for (int i=0;i<12;i++)
          {
            cout << std::setprecision(15) << double_approx(restest_(i)) << endl;
          }

          cout << "resold: " << endl;
          for (int i=0;i<12;i++)
          {
            cout << std::setprecision(15) << elevec1[i] << endl;
          }

          cout << "stiffnew: " << endl;
          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              cout << std::setprecision(15) << double_approx(stifftest_(i,j)) << "  ";
            }
            cout << endl;
          }

          cout << "stiffold: " << endl;
          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              cout << std::setprecision(15) << elemat1(i,j) << "  ";
            }
            cout << endl;
          }

          LINALG::Matrix<12,12> stiff_error;
          LINALG::Matrix<12,1> res_error;
          for(int line=0; line<12; line++)
          {
            for(int col=0; col<12; col++)
            {
              if (stifftest_(line,col)<cl_F("1.0e-15_40"))
              {stiff_error(line,col)=fabs( double_approx(stifftest_(line,col)) - elemat1(line,col) );}
              else
              {stiff_error(line,col)= fabs( double_approx(stifftest_(line,col)) - elemat1(line,col) ) /  double_approx(stifftest_(line,col));}
              //{stiff_error(line,col)= cl_float(abs( ( expt(stifflocal(line,col),2.0) - expt(stiff_approx(line,col),2.0) )/ ( (stifflocal(line,col) + stiff_approx(line,col)) * stifflocal(line,col) )),float_format(40));}

              //suppressing small entries whose effect is only confusing and NaN entires (which arise due to zero entries)
              //if ( fabs( stiff_error(line,col) ) < 1.0e-15 ) //isnan = is not a number
              //stiff_error(line,col) = 0.0;
            } //for(int col=0; col<3*nnode; col++)
          } //for(int line=0; line<3*nnode; line++)

          for(int line=0; line<12; line++)
          {
              if (restest_(line)<cl_F("1.0e-15_40"))
              {res_error(line)=fabs( double_approx(restest_(line)) - elevec1(line) );}
              else
              {res_error(line)= fabs( double_approx(restest_(line)) - elevec1(line) ) /  double_approx(restest_(line));}
              //{stiff_error(line,col)= cl_float(abs( ( expt(stifflocal(line,col),2.0) - expt(stiff_approx(line,col),2.0) )/ ( (stifflocal(line,col) + stiff_approx(line,col)) * stifflocal(line,col) )),float_format(40));}

              //suppressing small entries whose effect is only confusing and NaN entires (which arise due to zero entries)
              if ( fabs( res_error(line) ) < 1.0e-15 ) //isnan = is not a number
              res_error(line) = 0.0;
          } //for(int line=0; line<3*nnode; line++)

          cout << "stifferror: " << endl;
          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              cout << std::setprecision(5) << std::setw(8) << stiff_error(i,j) << "  ";
            }
            cout << endl;
          }

          cout << "reserror: " << endl;
          for (int i=0;i<12;i++)
          {
            cout << std::setprecision(15) << res_error(i) << endl;
          }End: Compare with old stiffness
          */


          for (int i=0;i<12;i++)
          {
            for (int j=0;j<12;j++)
            {
              stiffglobal(ele*6 + i ,ele*6 + j)+=stifflocal(i,j);
            }
            resglobal(ele*6 + i ,0)+=reslocal(i,0);
          }

        }//End: Evaluate all elements and assemble


        //add fext
        for (int i=0;i<6*numnode;i++)
        {
          //forces:
          fextglobal(i)="0.0_40";
        }
        fextglobal(numnode*6 -1 -4,0)=fextstep;

        LINALG::TMatrix<cl_F,3,1> fextm;
        LINALG::TMatrix<cl_F,3,3> stiffextm;
        LINALG::TMatrix<cl_F,3,1> tangentdisp;
        for (int i=0;i<3;i++)
        {
          for (int j=0;j<3;j++)
          {
            stiffextm(i,j)="0.0_40";
          }
          fextm(i)="0.0_40";
          tangentdisp(i)=dispglobal(numnode*6-3+i);
        }

        EvaluateNeumannPrec(tangentdisp, mextvecstep, &fextm, &stiffextm);

        for (int i=0;i<3;i++)
        {
          //moments:
          fextglobal(numnode*6-3+i,0)+=fextm(i);
        }

        for (int i=0;i<6*numnode;i++)
        {
          //add fext to residual:
          resglobal(i,0)-=fextglobal(i,0);
          resglobal(i,0)= -resglobal(i,0);
        }

        for (int i=0;i<3;i++)
        {
          for (int j=0;j<3;j++)
          {
            stiffglobal(numnode*6 - 3 + i,numnode*6 - 3 + j)+=stiffextm(i,j);
          }
        }

        //end: add fext

        //apply dirichlet
        for (int j=0;j<3;j++)
        {
          for (int i=0;i<6*numnode;i++)
          {
            stiffglobal(j,i)=cl_F("0.0_40");
            stiffglobal(i,j)=cl_F("0.0_40");
          }
          resglobal(j,0)=cl_F("0.0_40");
          stiffglobal(j,j)=cl_F("1.0_40");
        }

        for (int j=4;j<6;j++)
        {
          for (int i=0;i<6*numnode;i++)
          {
            stiffglobal(j,i)=cl_F("0.0_40");
            stiffglobal(i,j)=cl_F("0.0_40");
          }
          resglobal(j,0)=cl_F("0.0_40");
          stiffglobal(j,j)=cl_F("1.0_40");
        }
        //end: apply dirichlet

        //linear solver
        LINALG::TMatrix<cl_F,numnode*6,numnode*6> stiffglobalsolv;
        LINALG::TMatrix<cl_F,numnode*6,1> resglobalsolv;

        for (int i=0;i<6*numnode;i++)
        {
          for (int j=0;j<6*numnode;j++)
          {
            stiffglobalsolv(i,j)=stiffglobal(i,j);
          }
          resglobalsolv(i,0)=resglobal(i,0);
        }

        //Obere Dreiecksmatrix erzeugen
        for (int k=1; k<numnode*6; k++)
        {
          for (int zeile=k; zeile<numnode*6; zeile++)
          {
            if (abs(stiffglobalsolv(zeile,k-1))<TOLLINSOLV)
            {
              stiffglobalsolv(zeile,k-1)=cl_F("0.0_40");
            }
            else
            {
              cl_F faktor= stiffglobalsolv(zeile,k-1);
              for (int spalte=k-1; spalte<numnode*6; spalte++)
              { //cout << "vorher k, zeile, spalte" << k << " , " << zeile << " , " << spalte << ": " << stiffglobalsolv(zeile, spalte) << endl;
                stiffglobalsolv(zeile,spalte)= -stiffglobalsolv(k-1,spalte)*faktor/stiffglobalsolv(k-1,k-1)+stiffglobalsolv(zeile, spalte);
                //cout << "nachher k, zeile, spalte" << k << " , " << zeile << " , " << spalte << ": " << stiffglobalsolv(zeile, spalte) << endl;
              }
              resglobalsolv(zeile,0)= -resglobalsolv(k-1,0)*faktor/stiffglobalsolv(k-1,k-1)+resglobalsolv(zeile, 0);
            }
          }
        }
        //End:Obere Dreiecksmatrix erzeugen


        //globales deltaDisplacement nullen
        for (int i=0;i<6*numnode;i++)
        {
          deltadispglobal(i,0)=cl_F("0.0_40");
        }
        //globales deltaDisplacement nullen


        //Rückwärtseliminierung
        for (int zeile=numnode*6-1;zeile>-1; zeile--)
        {
          deltadispglobal(zeile,0) = resglobalsolv(zeile,0);
          for (int spalte=zeile+1;spalte<numnode*6; spalte++)
          {
            deltadispglobal(zeile,0) -= deltadispglobal(spalte,0)*stiffglobalsolv(zeile,spalte);
          }
          deltadispglobal(zeile,0) = deltadispglobal(zeile,0)/stiffglobalsolv(zeile,zeile);
        }
        //End: Rückwärtseliminierung

        //Ermittlung des Fehlers
        LINALG::TMatrix<cl_F,numnode*6,1> disperror;
        for (int i=0; i<6*numnode;i++)
        {
          disperror(i,0)= cl_F("0.0_40");
          for (int j=0; j<6*numnode; j++)
          {
            disperror(i,0)+=stiffglobal(i,j)*deltadispglobal(j,0);
          }
          disperror(i,0)-=resglobal(i,0);
          //cout << "disperror " << i << ": " << disperror(i,0) << endl;
        }
        //End: Ermittlung des Fehlers
        //end: linear solver

        //Update Verschiebungszustand
        for (int i=0;i<numnode*6;i++)
        {
          dispglobal(i,0)+=deltadispglobal(i,0);
        }
        //End: Update Verschiebungszustand


        //Berechnung und Ausgabe der Normen
        for (int i=0; i<numnode*6;i++)
        {
          resnorm += resglobal(i,0)*resglobal(i,0);
          dispnorm += deltadispglobal(i,0)*deltadispglobal(i,0);
          linsolverrornorm += disperror(i,0)*disperror(i,0);
        }
        resnorm = sqrt(resnorm)/sqrt(cl_float(numnode*6,float_format(40)));
        dispnorm = sqrt(dispnorm)/sqrt(cl_float(numnode*6,float_format(40)));
        linsolverrornorm = sqrt(linsolverrornorm)/sqrt(cl_float(numnode*6,float_format(40)));
        cout << "iter: " << iter << "   resnorm: " << double_approx(resnorm) << "   dispnorm: " << double_approx(dispnorm) << "   linsolverrornorm: " << double_approx(linsolverrornorm) << endl;
        //End: Berechnung und Ausgabe der Normen

      }//End: Newton

      cout << "end of Newton Iteration" << endl;
      cout << "dispglobalx: " << dispglobal(6*numnode-6) << endl;
      cout << "dispglobaly: " << dispglobal(6*numnode-5) << endl;
      cout << "dispglobalz: " << dispglobal(6*numnode-4) << endl;

    }//End Load steps

    exit(0);

    return;
  }

  void DRT::ELEMENTS::Beam3eb::EvaluateNeumannPrec( LINALG::TMatrix<cl_F,3,1> tangentdisp,
                                                    LINALG::TMatrix<cl_F,3,1> mextvec,
                                                    LINALG::TMatrix<cl_F,3,1>* fextm,
                                                    LINALG::TMatrix<cl_F,3,3>* stiffextm)
  {

#if NODALDOFS == 3
    dserror("High precision calculation is not implemented for the case NODALDOFS = 3!!!");
#endif

    LINALG::TMatrix<cl_F,3,1> tangent;
    LINALG::TMatrix<cl_F,3,1> crossproduct;
    cl_F abs_tangent_quadr = "0.0_40";
    //assemble current tangent and moment at node
    for (int i = 0 ; i < 3 ; i++)
    {
      //get current tangent at nodes
      tangent(i) = Trefprec_(i) + tangentdisp(i);
      abs_tangent_quadr += expt(tangent(i),2.0);
    }

    //calculate crossproduct
    (*fextm)(0)=-(tangent(1)*mextvec(2)-tangent(2)*mextvec(1))/abs_tangent_quadr;
    (*fextm)(1)=-(tangent(2)*mextvec(0)-tangent(0)*mextvec(2))/abs_tangent_quadr;
    (*fextm)(2)=-(tangent(0)*mextvec(1)-tangent(1)*mextvec(0))/abs_tangent_quadr;

    //assembly for stiffnessmatrix
     LINALG::TMatrix<cl_F,3,3> crossxtangent;
     LINALG::TMatrix<cl_F,3,3> spinmatrix;

     //perform matrix operation
     for(int i=0; i<3; i++)
     {
       for(int j=0; j<3; j++)
       {
         crossxtangent(i,j) = -(*fextm)(i) * tangent(j);
         spinmatrix(i,j)="0.0_40";
       }
     }

     //Compute Spinmatrix
     spinmatrix(0,1) = -mextvec(2);
     spinmatrix(0,2) = mextvec(1);
     spinmatrix(1,0) = mextvec(2);
     spinmatrix(1,2) = -mextvec(0);
     spinmatrix(2,0) = -mextvec(1);
     spinmatrix(2,1) = mextvec(0);

     //add R_external to stiffness matrix
     //all parts have been evaluated at the boundaries which helps simplifying the matrices
     //In contrast to the Neumann part of the residual force here is NOT a factor of (-1) needed, as elemat1 is directly added to the stiffness matrix
     //without sign change.
     for(int i = 0; i < 3 ; i++)
     {
       for(int j = 0; j < 3 ; j++)
       {
         (*stiffextm)(i,j) -= cl_F("2.0_40") * crossxtangent(i,j) / abs_tangent_quadr;
         (*stiffextm)(i,j) -= spinmatrix(i,j) / abs_tangent_quadr;
       }
     }

    return;
  }


#endif
//***************************************************************************************
//End: Methods for arbitrary precision calculation
//***************************************************************************************

