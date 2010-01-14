/*!----------------------------------------------------------------------
\file so_surface_evaluate.cpp
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET

#include <blitz/array.h>
#include "so_surface.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/linalg_serialdensematrix.H"
#include "../drt_lib/linalg_serialdensevector.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"
#include "../drt_surfstress/drt_surfstress_manager.H"
#include "../drt_potential/drt_potential_manager.H"
#include "../drt_statmech/bromotion_manager.H"

#include "Sacado.hpp"

using UTILS::SurfStressManager;
using POTENTIAL::PotentialManager;

/*----------------------------------------------------------------------*
 * Integrate a Surface Neumann boundary condition (public)     gee 04/08|
 * ---------------------------------------------------------------------*/
int DRT::ELEMENTS::StructuralSurface::EvaluateNeumann(ParameterList&           params,
                                                      DRT::Discretization&     discretization,
                                                      DRT::Condition&          condition,
                                                      vector<int>&             lm,
                                                      Epetra_SerialDenseVector& elevec1,
                                                      Epetra_SerialDenseMatrix* elemat1)
{
  // get type of condition
  enum LoadType
  {
    neum_none,
    neum_live,
    neum_orthopressure
  };
  // spatial or material configuration depends on the type of load
  enum Configuration
  {
    config_none,
    config_material,
    config_spatial,
    config_both
  };

  bool loadlin = (elemat1!=NULL);
  if (NumNode()!=4) loadlin = false; // implemented for 4 node quads only
  if (loadlin && elemat1->M() != 12) dserror("Mismatch in matrix size");

  Configuration config = config_none;
  LoadType ltype       = neum_none;
  const string* type = condition.Get<string>("type");
  if      (*type == "neum_live")
  {
    ltype = neum_live;
    config = config_material;
    loadlin = false; // no linearization as load applies to reference config
  }
  else if (*type == "neum_orthopressure")
  {
    ltype = neum_orthopressure;
    config = config_spatial;
  }
  else
  {
    dserror("Unknown type of SurfaceNeumann condition");
  }

  // get values and switches from the condition
  const vector<int>*    onoff = condition.Get<vector<int> >   ("onoff");
  const vector<double>* val   = condition.Get<vector<double> >("val"  );
  const vector<int>*    spa_func  = condition.Get<vector<int> > ("funct");

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
    curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);

  // element geometry update
  const int numnode = NumNode();
  const int numdf=3;
  LINALG::SerialDenseMatrix x(numnode,numdf);
  LINALG::SerialDenseMatrix xc;
  switch (config)
  {
  case config_material:
    MaterialConfiguration(x);
  break;
  case config_spatial:
  {
    xc.LightShape(numnode,numdf);
#ifndef INVERSEDESIGNCREATE
    RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
    if (disp==null) dserror("Cannot get state vector 'displacement'");
    vector<double> mydisp(lm.size());
    DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
    SpatialConfiguration(xc,mydisp);
//    MaterialConfiguration(xc);
#else
    // in inverse design analysis, the current configuration is the reference
    MaterialConfiguration(xc);
    loadlin = false; // follower load is with respect to known current config in inverse design
#endif
  }
  break;
  case config_both:
  {
    xc.LightShape(numnode,3);
    RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
    if (disp==null) dserror("Cannot get state vector 'displacement'");
    vector<double> mydisp(lm.size());
    DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
    MaterialConfiguration(x);
    SpatialConfiguration(xc,mydisp);
  }
  break;
  default: dserror("Unknown case of frame");
  break;
  }

  // allocate vector for shape functions and matrix for derivatives
  LINALG::SerialDenseVector  funct(numnode);
  LINALG::SerialDenseMatrix  deriv(2,numnode);

  /*----------------------------------------------------------------------*
  |               start loop over integration points                     |
  *----------------------------------------------------------------------*/
  const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);
  for (int gp=0; gp<intpoints.nquad; gp++)
  {
    const double e0 = intpoints.qxg[gp][0];
    const double e1 = intpoints.qxg[gp][1];

    // get shape functions and derivatives in the plane of the element
    DRT::UTILS::shape_function_2D(funct,e0,e1,Shape());
    DRT::UTILS::shape_function_2D_deriv1(deriv,e0,e1,Shape());
    //Stuff to get spatial Neumann
    const int numdim = 3;
    LINALG::SerialDenseMatrix gp_coord(1,numdim);
     
    switch(ltype)
    {
    case neum_live:
    {
      LINALG::SerialDenseMatrix dxyzdrs(2,3);
      dxyzdrs.Multiply('N','N',1.0,deriv,x,0.0);
      LINALG::SerialDenseMatrix  metrictensor(2,2);
      metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      const double detA = sqrt( metrictensor(0,0)*metrictensor(1,1)
                                -metrictensor(0,1)*metrictensor(1,0));
      
      double functfac = 1.0;
      int functnum = -1;
      double val_curvefac_functfac;

      for(int dof=0;dof<numdim;dof++)
      {
        if ((*onoff)[dof]) // is this dof activated?
        {
          
          //factor given by spatial function
          if (spa_func) 
            functnum = (*spa_func)[dof];
                   
          if (functnum>0)
          {
            //Calculate reference position of GP 
            gp_coord.Multiply('T','N',1.0,funct,x,0.0);;
            // write coordinates in another datatype
            double gp_coord2[numdim];
            for(int i=0;i<numdim;i++)
            {
              gp_coord2[i]=gp_coord(0,i);
            }
            const double* coordgpref = &gp_coord2[0]; // needed for function evaluation
    
            //evaluate function at current gauss point
            functfac = DRT::Problem::Instance()->Funct(functnum-1).Evaluate(dof,coordgpref,0.0,NULL);
          }
          else
            functfac = 1.0;
                   
          val_curvefac_functfac = functfac*curvefac;
          const double fac = intpoints.qwgt[gp] * detA * (*val)[dof] * val_curvefac_functfac;
          for (int node=0; node < numnode; ++node)
          {
            elevec1[node*numdf+dof]+= funct[node] * fac;
          }
        }
        
      }
    }
    break;
    case neum_orthopressure:
    {
     if ((*onoff)[0] != 1) dserror("orthopressure on 1st dof only!");
      for (int checkdof = 1; checkdof < 3; ++checkdof)
        if ((*onoff)[checkdof] != 0) dserror("orthopressure on 1st dof only!");
      double ortho_value = (*val)[0];
      if (!ortho_value) dserror("no orthopressure value given!");
      vector<double> normal(3);
      SurfaceIntegration(normal, xc,deriv);
      //Calculate spatial position of GP 
      double functfac = 1.0;
      int functnum = -1;
      double val_curvefac_functfac; 
      
      // factor given by spatial function
      if (spa_func) functnum = (*spa_func)[0];
      {
        if (functnum>0)
        {
          gp_coord.Multiply('T','N',1.0,funct,xc,0.0);
          // write coordinates in another datatype
          double gp_coord2[numdim];
          for(int i=0;i<numdim;i++)
          {
            gp_coord2[i]=gp_coord(0,i);
          }
          const double* coordgpref = &gp_coord2[0]; // needed for function evaluation

          // evaluate function at current gauss point
          functfac = DRT::Problem::Instance()->Funct(functnum-1).Evaluate(0,coordgpref,0.0,NULL);
        }
        else
          functfac = 1.0;
      }

       val_curvefac_functfac = curvefac*functfac;    
           
      
      const double fac = intpoints.qwgt[gp] * val_curvefac_functfac* ortho_value;
      for (int node=0; node < numnode; ++node)
        for(int dim=0 ; dim<3; dim++)
          elevec1[node*numdf+dim] += funct[node] * normal[dim] * fac;

      if (loadlin)
      {
        Epetra_SerialDenseMatrix a_Dnormal(3,12);

        //analytical_DSurfaceIntegration(a_Dnormal, xc, deriv); // this one quad4 only!
        FAD_DFAD_DSurfaceIntegration(a_Dnormal, xc, deriv);     // this one for arbitrary surface elements
        //FAD_SFAD_DSurfaceIntegration(a_Dnormal, xc, deriv);   // this one quad4 only!
        //FiniteDiff_DSurfaceIntegration(a_Dnormal, xc, deriv); // this one for arbitrary surface elements

        for (int node=0; node < numnode; ++node)
          for (int dim=0 ; dim<3; dim++)
            for (int dof=0; dof<elevec1.M(); dof++)
              (*elemat1)(node*numdf+dim,dof) += funct[node] * a_Dnormal(dim, dof) * fac;
      }


    }
    break;
    default:
      dserror("Unknown type of SurfaceNeumann load");
    break;
    }

  } /* end of loop over integration points gp */

  return 0;
}

/*----------------------------------------------------------------------*
 * Evaluate normal at gp (private)                             gee 08/08|
 * ---------------------------------------------------------------------*/
void DRT::ELEMENTS::StructuralSurface::SurfaceIntegration(vector<double>& normal,
                                                          const Epetra_SerialDenseMatrix& x,
                                                          const Epetra_SerialDenseMatrix& deriv)
{
  // note that the length of this normal is the area dA

  // compute dXYZ / drs
  LINALG::SerialDenseMatrix dxyzdrs(2,3);
  dxyzdrs.Multiply('N','N',1.0,deriv,x,0.0);

  normal[0] = dxyzdrs(0,1) * dxyzdrs(1,2) - dxyzdrs(0,2) * dxyzdrs(1,1);
  normal[1] = dxyzdrs(0,2) * dxyzdrs(1,0) - dxyzdrs(0,0) * dxyzdrs(1,2);
  normal[2] = dxyzdrs(0,0) * dxyzdrs(1,1) - dxyzdrs(0,1) * dxyzdrs(1,0);

  return;
}


/*----------------------------------------------------------------------*
 * Calculates dnormal/dx_j with Saccado  DFAD            holfelder 04/09|
 * ---------------------------------------------------------------------*/

void DRT::ELEMENTS::StructuralSurface::FAD_DFAD_DSurfaceIntegration(Epetra_SerialDenseMatrix& d_normal,
                                                          const Epetra_SerialDenseMatrix& x,
                                                          const Epetra_SerialDenseMatrix& deriv)
{
  // this routine holds for all element shapes

  //Erstellen eines Vektor des Typs Saccado fuer x und deriv
  vector<Sacado::Fad::DFad<double> > saccado_x(x.N()*x.M());
  vector<Sacado::Fad::DFad<double> > saccado_deriv(deriv.N()*deriv.M());
  vector<Sacado::Fad::DFad<double> > saccado_g1(3);
  vector<Sacado::Fad::DFad<double> > saccado_g2(3);

  vector<Sacado::Fad::DFad<double> > saccado_normal(3);

  //Kopieren der Daten der x_Matrix
  for(int row=0; row < x.M(); row++){
  	for(int column = 0; column < x.N(); column++){
  		saccado_x[x.N()*row+column] = x(row,column);
		saccado_x[x.N()*row+column].diff(x.N()*row+column,x.N()*x.M());
	}
  }
  //Kopieren der Daten der deriv Matrix
  for(int row=0; row < deriv.M(); row++){
  	for(int column = 0; column < deriv.N(); column++){
  		saccado_deriv[deriv.N()*row+column]= deriv(row,column);
	}
  }

  //Berechung von deriv*x
  for(int dim = 0; dim < 3; dim++){
	for(int column = 0; column < deriv.N(); column++){
		saccado_g1[dim] += saccado_deriv[column]* saccado_x[column*x.N()+dim];
		saccado_g2[dim] += saccado_deriv[column+deriv.N()]* saccado_x[column*x.N()+dim];
	}
  }


  //Berechnen der Normalen
  saccado_normal[0] = saccado_g1[1]*saccado_g2[2]-saccado_g1[2]*saccado_g2[1];
  saccado_normal[1] = saccado_g1[2]*saccado_g2[0]-saccado_g1[0]*saccado_g2[2];
  saccado_normal[2] = saccado_g1[0]*saccado_g2[1]-saccado_g1[1]*saccado_g2[0];

  //Direktzugriff auf die Ableitungen
  for(int dim = 0; dim <3; dim++){
	for(int dxyz = 0; dxyz<12; dxyz++){
		d_normal(dim, dxyz) = saccado_normal[dim].fastAccessDx(dxyz);
	}
  }

  return;
}


/*----------------------------------------------------------------------*
 * Calculates dnormal/dx_j with Saccado  SFAD            holfelder 04/09|
 * ---------------------------------------------------------------------*/

void DRT::ELEMENTS::StructuralSurface::FAD_SFAD_DSurfaceIntegration(Epetra_SerialDenseMatrix& d_normal,
                                                          const Epetra_SerialDenseMatrix& x,
                                                          const Epetra_SerialDenseMatrix& deriv)
{
  // this routine holds for quad4 element only

  //Erstellen eines Vektor des Typs Saccado fuer x und deriv
  vector<Sacado::Fad::SFad<double, 12> > saccado_x(x.N()*x.M());
  vector<Sacado::Fad::SFad<double, 12> > saccado_deriv(deriv.N()*deriv.M());
  vector<Sacado::Fad::SFad<double, 12> > saccado_g1(3);
  vector<Sacado::Fad::SFad<double, 12> > saccado_g2(3);

  vector<Sacado::Fad::SFad<double, 12> > saccado_normal(3);

  //Kopieren der Daten der x_Matrix
  for(int row=0; row < x.M(); row++){
  	for(int column = 0; column < x.N(); column++){
  		saccado_x[x.N()*row+column] = x(row,column);
		saccado_x[x.N()*row+column].diff(x.N()*row+column,x.N()*x.M());
	}
  }
  //Kopieren der Daten der deriv Matrix
  for(int row=0; row < deriv.M(); row++){
  	for(int column = 0; column < deriv.N(); column++){
  		saccado_deriv[deriv.N()*row+column]= deriv(row,column);
	}
  }

  //Berechung von deriv*x
  for(int dim = 0; dim < 3; dim++){
	for(int column = 0; column < deriv.N(); column++){
		saccado_g1[dim] += saccado_deriv[column]* saccado_x[column*x.N()+dim];
		saccado_g2[dim] += saccado_deriv[column+deriv.N()]* saccado_x[column*x.N()+dim];
	}
  }


  //Berechnen der Normalen
  saccado_normal[0] = saccado_g1[1]*saccado_g2[2]-saccado_g1[2]*saccado_g2[1];
  saccado_normal[1] = saccado_g1[2]*saccado_g2[0]-saccado_g1[0]*saccado_g2[2];
  saccado_normal[2] = saccado_g1[0]*saccado_g2[1]-saccado_g1[1]*saccado_g2[0];

  //Direktzugriff auf die Ableitungen
  for(int dim = 0; dim <3; dim++){
	for(int dxyz = 0; dxyz<12; dxyz++){
			d_normal(dim, dxyz) = saccado_normal[dim].fastAccessDx(dxyz);
	}
  }

  return;
}


/*----------------------------------------------------------------------*
 * Calculates dnormal/dx_j analytically                  holfelder 04/09|
 * ---------------------------------------------------------------------*/

void DRT::ELEMENTS::StructuralSurface::	analytical_DSurfaceIntegration(
                                                          Epetra_SerialDenseMatrix& d_normal,
                                                          const Epetra_SerialDenseMatrix& x,
                                                          const Epetra_SerialDenseMatrix& deriv)
{
  // this routine holds for quad4 element only

  // compute dXYZ / drs
  LINALG::SerialDenseMatrix dxyzdrs(2,3);
  dxyzdrs.Multiply('N','N',1.0,deriv, x, 0.0);

  Epetra_SerialDenseMatrix g_dg1(3,3);
  Epetra_SerialDenseMatrix g_dg2(3,3);
  Epetra_SerialDenseMatrix g1_dx(3,12);
  Epetra_SerialDenseMatrix g2_dx(3,12);
  Epetra_SerialDenseMatrix g_dx(3,12);

  g1_dx.Scale(0);
  g2_dx.Scale(0);
  g_dg1.Scale(0);
  g_dg2.Scale(0);

  for(int dim = 0; dim < 3; dim++){
	g1_dx(dim, dim) = deriv(0,0);
	g1_dx(dim, dim+3) = deriv(0,1);
	g1_dx(dim, dim+6) = deriv(0,2);
	g1_dx(dim, dim+9) = deriv(0,3);

	g2_dx(dim, dim)   = deriv(1,0);
	g2_dx(dim, dim+3) = deriv(1,1);
	g2_dx(dim, dim+6) = deriv(1,2);
	g2_dx(dim, dim+9) = deriv(1,3);

  }

  g_dg1(0,1) =  dxyzdrs(1,2);
  g_dg1(0,2) = -dxyzdrs(1,1);
  g_dg1(1,0) = -dxyzdrs(1,2);
  g_dg1(1,2) =  dxyzdrs(1,0);
  g_dg1(2,0) =  dxyzdrs(1,1);
  g_dg1(2,1) = -dxyzdrs(1,0);

  g_dg2(0,1) = -dxyzdrs(0,2);
  g_dg2(0,2) =  dxyzdrs(0,1);
  g_dg2(1,0) =  dxyzdrs(0,2);
  g_dg2(1,2) = -dxyzdrs(0,0);
  g_dg2(2,0) = -dxyzdrs(0,1);
  g_dg2(2,1) =  dxyzdrs(0,0);

  g_dx.Multiply('N','N',1.0, g_dg1, g1_dx, 0.0);
  d_normal=g_dx;
  g_dx.Scale(0);
  g_dx.Multiply('N','N',1.0, g_dg2, g2_dx, 0.0);
  d_normal+=g_dx;
  return;
}

/*----------------------------------------------------------------------*
 * Calculates dnormal/dx_j with Finite Differences       holfelder 04/09|
 * ---------------------------------------------------------------------*/

void DRT::ELEMENTS::StructuralSurface::	FiniteDiff_DSurfaceIntegration(Epetra_SerialDenseMatrix& d_normal,
                                                          const Epetra_SerialDenseMatrix& x,
                                                          const Epetra_SerialDenseMatrix& deriv)
{
  // this routine holds for all element shapes

  vector<double> Dnormal(3);
  vector<double> normal(3);
  Epetra_SerialDenseMatrix dxc(x);

  double delta = 1e-6;

  //Zeile der Jacobimatrix
  for(int dx=0;dx<3;dx++){
    //Lauf ueber die Freiheitsgrade der Funktion
    for (int node=0; node < x.M(); ++node){
      for(int dim=0 ; dim<3; dim++){
        //Gleichsetzen der Matrizen
        dxc=x;
        //Variation um den Wert delta
        dxc(node, dim) += delta;
        SurfaceIntegration(Dnormal, dxc, deriv);
        SurfaceIntegration(normal, x, deriv),
        d_normal(dx,node*3+dim) = (Dnormal[dx] - normal[dx])/delta;
      }
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 * Evaluate sqrt of determinant of metric at gp (private)      gee 04/08|
 * ---------------------------------------------------------------------*/
void DRT::ELEMENTS::StructuralSurface::SurfaceIntegration(double& detA,
                                                          vector<double>& normal,
                                                          const Epetra_SerialDenseMatrix& x,
                                                          const Epetra_SerialDenseMatrix& deriv)
{

  // compute dXYZ / drs
  LINALG::SerialDenseMatrix dxyzdrs(2,3);
  dxyzdrs.Multiply('N','N',1.0,deriv,x,0.0);

  /* compute covariant metric tensor G for surface element
  **                        | g11   g12 |
  **                    G = |           |
  **                        | g12   g22 |
  ** where (o denotes the inner product, xyz a vector)
  **
  **       dXYZ   dXYZ          dXYZ   dXYZ          dXYZ   dXYZ
  ** g11 = ---- o ----    g12 = ---- o ----    g22 = ---- o ----
  **        dr     dr            dr     ds            ds     ds
  */
  LINALG::SerialDenseMatrix metrictensor(2,2);
  metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
  detA = sqrt( metrictensor(0,0)*metrictensor(1,1)-metrictensor(0,1)*metrictensor(1,0) );
  normal[0] = dxyzdrs(0,1) * dxyzdrs(1,2) - dxyzdrs(0,2) * dxyzdrs(1,1);
  normal[1] = dxyzdrs(0,2) * dxyzdrs(1,0) - dxyzdrs(0,0) * dxyzdrs(1,2);
  normal[2] = dxyzdrs(0,0) * dxyzdrs(1,1) - dxyzdrs(0,1) * dxyzdrs(1,0);

  return;
}

/*----------------------------------------------------------------------*
 * Evaluate method for StructuralSurface-Elements               tk 10/07*
 * ---------------------------------------------------------------------*/
int DRT::ELEMENTS::StructuralSurface::Evaluate(ParameterList&            params,
                                               DRT::Discretization&      discretization,
                                               vector<int>&              lm,
                                               Epetra_SerialDenseMatrix& elematrix1,
                                               Epetra_SerialDenseMatrix& elematrix2,
                                               Epetra_SerialDenseVector& elevector1,
                                               Epetra_SerialDenseVector& elevector2,
                                               Epetra_SerialDenseVector& elevector3)
{
  // start with "none"
  DRT::ELEMENTS::StructuralSurface::ActionType act = StructuralSurface::none;

  // get the required action
  string action = params.get<string>("action","none");
  if (action == "none") dserror("No action supplied");
  else if (action=="calc_struct_constrvol")        act = StructuralSurface::calc_struct_constrvol;
  else if (action=="calc_struct_volconstrstiff")   act = StructuralSurface::calc_struct_volconstrstiff;
  else if (action=="calc_struct_monitarea")        act = StructuralSurface::calc_struct_monitarea;
  else if (action=="calc_struct_constrarea")       act = StructuralSurface::calc_struct_constrarea;
  else if (action=="calc_struct_areaconstrstiff")  act = StructuralSurface::calc_struct_areaconstrstiff;
  else if (action=="calc_init_vol")                act = StructuralSurface::calc_init_vol;
  else if (action=="calc_surfstress_stiff")        act = StructuralSurface::calc_surfstress_stiff;
  else if (action=="calc_potential_stiff")         act = StructuralSurface::calc_potential_stiff;
  else if (action=="calc_brownian_motion")         act = StructuralSurface::calc_brownian_motion;
  else if (action=="calc_brownian_motion_damping") act = StructuralSurface::calc_brownian_motion_damping;
  else if (action=="calc_struct_centerdisp") 	   act = StructuralSurface::calc_struct_centerdisp;
  else
  {
    cout << action << endl;
    dserror("Unknown type of action for StructuralSurface");
  }

  //create communicator
  const Epetra_Comm& Comm = discretization.Comm();
  // what the element has to do
  switch(act)
  {
      //gives the center displacement for SlideALE
	
  case calc_struct_centerdisp:
  {
	  //We are not interested in ghosted elements
	  if(Comm.MyPID()==Owner())
	  {
		  // element geometry update
		  RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacementtotal");
		  if (disp==null) dserror("Cannot get state vector 'displacementtotal'");
		  vector<double> mydisp(lm.size());
		  DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
		  const int numnode = NumNode();
		  const int numdf=3;
		  LINALG::SerialDenseMatrix x(numnode,numdf);
		  LINALG::SerialDenseMatrix xc(numnode,numdf);
		  SpatialConfiguration(xc,mydisp);

		  //integration of the displacements over the surface
		  // allocate vector for shape functions and matrix for derivatives
		  LINALG::SerialDenseVector  funct(numnode);
		  LINALG::SerialDenseMatrix  deriv(2,numnode);

		  /*----------------------------------------------------------------------*
		    |               start loop over integration points                     |
		   *----------------------------------------------------------------------*/
		  const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);

		  RefCountPtr<const Epetra_Vector> dispincr = discretization.GetState("displacementincr");
		  vector<double> edispincr(lm.size());
		  DRT::UTILS::ExtractMyValues(*dispincr,edispincr,lm);
		  elevector2[0] = 0;

		  for (int gp=0; gp<intpoints.nquad; gp++)
		  {
			  const double e0 = intpoints.qxg[gp][0];
			  const double e1 = intpoints.qxg[gp][1];

			  // get shape functions and derivatives in the plane of the element
			  DRT::UTILS::shape_function_2D(funct,e0,e1,Shape());
			  DRT::UTILS::shape_function_2D_deriv1(deriv,e0,e1,Shape());

			  vector<double> normal(3);
			  double detA;
			  SurfaceIntegration(detA,normal,xc,deriv);

			  elevector2[0] +=  sqrt( normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2] );  		

			  for (int dim=0; dim<3; dim++)
			  {
				  if (gp == 0) 
					  elevector3[dim] = 0;

				  for (int j=0; j<numnode; ++j)
				  {
					  elevector3[dim] +=  funct[j] * intpoints.qwgt[gp] 
					                          * edispincr[j*numdf + dim] * detA;
				  }
			  }

		  }

	  } 
  }
  break;   
  
  
  	case calc_struct_constrvol:
      {
        //We are not interested in volume of ghosted elements
        if(Comm.MyPID()==Owner())
        {
          // element geometry update
          RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
          if (disp==null) dserror("Cannot get state vector 'displacement'");
          vector<double> mydisp(lm.size());
          DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
          const int numdim = 3;
          LINALG::SerialDenseMatrix xscurr(NumNode(),numdim);  // material coord. of element
          SpatialConfiguration(xscurr,mydisp);
          //call submethod for volume evaluation and store rseult in third systemvector
          double volumeele = ComputeConstrVols(xscurr,NumNode());
          elevector3[0]= volumeele;
        }
      }
      break;
      case calc_struct_volconstrstiff:
      {
        // element geometry update
        RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
        if (disp==null) dserror("Cannot get state vector 'displacement'");
        vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
        const int numdim =3;
        LINALG::SerialDenseMatrix xscurr(NumNode(),numdim);  // material coord. of element
        SpatialConfiguration(xscurr,mydisp);
        double volumeele;
        // first partial derivatives
        RCP<Epetra_SerialDenseVector> Vdiff1 = rcp(new Epetra_SerialDenseVector);
        // second partial derivatives
        RCP<Epetra_SerialDenseMatrix> Vdiff2 = rcp(new Epetra_SerialDenseMatrix);

        //get projection method
        RCP<DRT::Condition> condition = params.get<RefCountPtr<DRT::Condition> >("condition");
        const string* projtype = condition->Get<string>("projection");

        if (projtype != NULL)
        {
          //call submethod to compute volume and its derivatives w.r.t. to current displ.
          if (*projtype == "yz")
          {
            ComputeVolDeriv(xscurr, NumNode(),numdim*NumNode(), volumeele, Vdiff1, Vdiff2, 0, 0);
          }
          else if (*projtype == "xz")
          {
            ComputeVolDeriv(xscurr, NumNode(),numdim*NumNode(), volumeele, Vdiff1, Vdiff2, 1, 1);
          }
          else if (*projtype == "xy")
          {
            ComputeVolDeriv(xscurr, NumNode(),numdim*NumNode(), volumeele, Vdiff1, Vdiff2, 2, 2);
          }
          else
          {
            ComputeVolDeriv(xscurr, NumNode(),numdim*NumNode(), volumeele, Vdiff1, Vdiff2);
          }
        }
        else
          ComputeVolDeriv(xscurr, NumNode(),numdim*NumNode(), volumeele, Vdiff1, Vdiff2);

        //update rhs vector and corresponding column in "constraint" matrix
        elevector1 = *Vdiff1;
        elevector2 = *Vdiff1;
        elematrix1 = *Vdiff2;
        //call submethod for volume evaluation and store result in third systemvector
        elevector3[0]=volumeele;
      }
      break;
      case calc_init_vol:
      {
        // the reference volume of the RVE (including inner
        // holes) is calculated by evaluating the following
        // surface integral:
        // V = 1/3*int(div(X))dV = 1/3*int(N*X)dA
        // with X being the reference coordinates and N the
        // normal vector of the surface element (exploiting the
        // fact that div(X)=1.0)

        // this is intended to be used in the serial case (microstructure)

        // NOTE: there must not be any holes penetrating the boundary!

        double V = params.get<double>("V0", 0.0);
        double dV = 0.0;
        const int numnode = NumNode();
        LINALG::SerialDenseMatrix x(numnode,3);
        MaterialConfiguration(x);

        // allocate vector for shape functions and matrix for derivatives
        LINALG::SerialDenseVector  funct(numnode);
        LINALG::SerialDenseMatrix  deriv(2,numnode);

        /*----------------------------------------------------------------------*
         |               start loop over integration points                     |
         *----------------------------------------------------------------------*/
        const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);

        for (int gp=0; gp<intpoints.nquad; gp++)
        {
          const double e0 = intpoints.qxg[gp][0];
          const double e1 = intpoints.qxg[gp][1];

          // get shape functions and derivatives in the plane of the element
          DRT::UTILS::shape_function_2D(funct,e0,e1,Shape());
          DRT::UTILS::shape_function_2D_deriv1(deriv,e0,e1,Shape());

          vector<double> normal(3);
          double detA;
          SurfaceIntegration(detA,normal,x,deriv);
          const double fac = intpoints.qwgt[gp] * detA;

          double temp = 0.0;
          vector<double> X(3,0.);

          for (int i=0; i<numnode; i++)
          {
            X[0] += funct[i]*x(i,0);
            X[1] += funct[i]*x(i,1);
            X[2] += funct[i]*x(i,2);
          }

          for (int i=0;i<3;++i)
          {
            temp += normal[i]*normal[i];
          }

          if (temp<0.)
            dserror("calculation of initial volume failed in surface element");
          double absnorm = sqrt(temp);

          for (int i=0;i<3;++i)
          {
            normal[i] /= absnorm;
          }
          for (int i=0;i<3;++i)
          {
            dV += 1/3.0*fac*normal[i]*X[i];
          }
        }
        params.set("V0", V+dV);
      }
      break;

      case calc_surfstress_stiff:
      {
        RefCountPtr<SurfStressManager> surfstressman =
          params.get<RefCountPtr<SurfStressManager> >("surfstr_man", null);

        if (surfstressman==null)
          dserror("No SurfStressManager in Solid3 Surface available");

        RefCountPtr<DRT::Condition> cond = params.get<RefCountPtr<DRT::Condition> >("condition",null);
        if (cond==null)
          dserror("Condition not available in Solid3 Surface");

        double time = params.get<double>("total time",-1.0);
        double dt = params.get<double>("delta time",0.0);
        bool newstep = params.get<bool>("newstep", false);

        // element geometry update

        const int numnode = NumNode();
        LINALG::SerialDenseMatrix x(numnode,3);

        RefCountPtr<const Epetra_Vector> dism = discretization.GetState("displacement");
        if (dism==null) dserror("Cannot get state vector 'displacement'");
        vector<double> mydism(lm.size());
        DRT::UTILS::ExtractMyValues(*dism,mydism,lm);
        SpatialConfiguration(x,mydism);

        const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);

        // set up matrices and parameters needed for the evaluation of current
        // interfacial area and its derivatives w.r.t. the displacements

        int ndof = 3*numnode;                                     // overall number of surface dofs
        double A;                                                 // interfacial area
        // first partial derivatives
        RCP<Epetra_SerialDenseVector> Adiff = rcp(new Epetra_SerialDenseVector);
        // second partial derivatives
        RCP<Epetra_SerialDenseMatrix> Adiff2 = rcp(new Epetra_SerialDenseMatrix);

        ComputeAreaDeriv(x, numnode, ndof, A, Adiff, Adiff2);

        if (cond->Type()==DRT::Condition::Surfactant)     // dynamic surfactant model
        {
          int curvenum = cond->GetInt("curve");
          double k1xC = cond->GetDouble("k1xCbulk");
          double k2 = cond->GetDouble("k2");
          double m1 = cond->GetDouble("m1");
          double m2 = cond->GetDouble("m2");
          double gamma_0 = cond->GetDouble("gamma_0");
          double gamma_min = cond->GetDouble("gamma_min");
          double gamma_min_eq = cond->GetDouble("gamma_min_eq");
          double con_quot_max = (gamma_min_eq-gamma_min)/m2+1.;
          double con_quot_eq = (k1xC)/(k1xC+k2);

          // element geometry update (n+1)
          RefCountPtr<const Epetra_Vector> disn = discretization.GetState("new displacement");
          if (disn==null) dserror("Cannot get state vector 'new displacement'");
          vector<double> mydisn(lm.size());
          DRT::UTILS::ExtractMyValues(*disn,mydisn,lm);
          SpatialConfiguration(x,mydisn);

          // set up matrices and parameters needed for the evaluation of
          // interfacial area and its first derivative w.r.t. the displacements at (n+1)
          double Anew = 0.;                                            // interfacial area
          RCP<Epetra_SerialDenseVector> Adiffnew = rcp(new Epetra_SerialDenseVector);

          ComputeAreaDeriv(x, numnode, ndof, Anew, Adiffnew, null);

          surfstressman->StiffnessAndInternalForces(curvenum, A, Adiff, Adiff2, Anew, Adiffnew, elevector1, elematrix1, this->Id(),
                                                    time, dt, 0, 0.0, k1xC, k2, m1, m2, gamma_0,
                                                    gamma_min, gamma_min_eq, con_quot_max,
                                                    con_quot_eq, newstep);
        }
        else if (cond->Type()==DRT::Condition::SurfaceTension) // ideal liquid
        {
          int curvenum = cond->GetInt("curve");
          double const_gamma = cond->GetDouble("gamma");
          surfstressman->StiffnessAndInternalForces(curvenum, A, Adiff, Adiff2, 0., Adiff, elevector1, elematrix1, this->Id(),
                                                    time, dt, 1, const_gamma, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                    0.0, 0.0, 0.0, 0.0, newstep);
        }
        else
          dserror("Unknown condition type %d",cond->Type());
      }
      break;

      // compute additional stresses due to intermolecular potential forces
      case calc_potential_stiff:
      {
        RefCountPtr<PotentialManager> potentialmanager =
          params.get<RefCountPtr<PotentialManager> >("pot_man", null);
        if (potentialmanager==null)
          dserror("No PotentialManager in Solid3 Surface available");

        RefCountPtr<DRT::Condition> cond = params.get<RefCountPtr<DRT::Condition> >("condition",null);
        if (cond==null)
          dserror("Condition not available in Solid3 Surface");

        if (cond->Type()==DRT::Condition::LJ_Potential_Surface) // Lennard-Jones potential
        {
          potentialmanager->StiffnessAndInternalForcesPotential(this, gaussrule_, params, lm, elematrix1, elevector1);
        }
        if (cond->Type()==DRT::Condition::ElectroRepulsion_Potential_Surface) // Electrostatic potential
        {
        	potentialmanager->StiffnessAndInternalForcesPotential(this, gaussrule_, params,lm, elematrix1, elevector1);
        }
        if (cond->Type()==DRT::Condition::VanDerWaals_Potential_Surface) // Electrostatic potential
        {
          potentialmanager->StiffnessAndInternalForcesPotential(this, gaussrule_, params,lm, elematrix1, elevector1);
        }
        if( cond->Type()!=DRT::Condition::LJ_Potential_Surface &&
            cond->Type()!=DRT::Condition::ElectroRepulsion_Potential_Surface &&
            cond->Type()!=DRT::Condition::VanDerWaals_Potential_Surface)
                    dserror("Unknown condition type %d",cond->Type());
      }
      break;

      // compute stochastical forces due to Brownian Motion
      case calc_brownian_motion:
      {
        dserror("not commited");
      }
      break;

      // compute damping matrix due to Brownian Motion
      case calc_brownian_motion_damping:
      {
          dserror("not yet comitted");
      }
      break;

      //compute the area (e.g. for initialization)
      case calc_struct_monitarea:
      {
        //We are not interested in volume of ghosted elements
        if(Comm.MyPID()==Owner())
        {
          // element geometry update
          RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
          if (disp==null) dserror("Cannot get state vector 'displacement'");
          vector<double> mydisp(lm.size());
          DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
          const int numdim = 3;
          LINALG::SerialDenseMatrix xscurr(NumNode(),numdim);  // material coord. of element
          SpatialConfiguration(xscurr,mydisp);

          RCP<DRT::Condition> condition = params.get<RefCountPtr<DRT::Condition> >("condition");
          const string* projtype = condition->Get<string>("projection");

          // To compute monitored area consider required projection method
          // and set according coordinates to zero
          if (*projtype == "yz")
          {
            xscurr(0,0)=0;
            xscurr(1,0)=0;
            xscurr(2,0)=0;
            xscurr(3,0)=0;
          }
          else if (*projtype == "xz")
          {
            xscurr(0,1)=0;
            xscurr(1,1)=0;
            xscurr(2,1)=0;
            xscurr(3,1)=0;
          }
          else if (*projtype == "xy")
          {
            xscurr(0,2)=0;
            xscurr(1,2)=0;
            xscurr(2,2)=0;
            xscurr(3,2)=0;
          }

          double areaele=0.0;
          const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);
          // allocate matrix for derivatives of shape functions
          LINALG::SerialDenseMatrix  deriv(2,NumNode());

          //Compute area
          for (int gp=0; gp<intpoints.nquad; gp++)
          {
            const double e0 = intpoints.qxg[gp][0];
            const double e1 = intpoints.qxg[gp][1];

            // get shape functions and derivatives in the plane of the element
            DRT::UTILS::shape_function_2D_deriv1(deriv,e0,e1,Shape());

            vector<double> normal(3);
            double detA;
            SurfaceIntegration(detA,normal,xscurr,deriv);
            const double fac = intpoints.qwgt[gp] * detA;
            areaele += fac;

          }

          //store result in third systemvector
          elevector3[0]=areaele;
        }

      }
      break;
      case calc_struct_constrarea:
      {
        // element geometry update
        RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
        if (disp==null) dserror("Cannot get state vector 'displacement'");
        vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
        const int numdim =3;
        LINALG::SerialDenseMatrix xscurr(NumNode(),numdim);  // material coord. of element
        SpatialConfiguration(xscurr,mydisp);
        // initialize variables
        double elearea;
        // first partial derivatives
        RCP<Epetra_SerialDenseVector> Adiff = rcp(new Epetra_SerialDenseVector);
        // second partial derivatives
        RCP<Epetra_SerialDenseMatrix> Adiff2 = rcp(new Epetra_SerialDenseMatrix);

        //call submethod
        ComputeAreaDeriv(xscurr, NumNode(),numdim*NumNode(), elearea, Adiff, Adiff2);
        // store result
        elevector3[0] = elearea;

      }
      break;
      case calc_struct_areaconstrstiff:
      {
        // element geometry update
        RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
        if (disp==null) dserror("Cannot get state vector 'displacement'");
        vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp,mydisp,lm);
        const int numdim =3;
        LINALG::SerialDenseMatrix xscurr(NumNode(),numdim);  // material coord. of element
        SpatialConfiguration(xscurr,mydisp);
        // initialize variables
        double elearea;
        // first partial derivatives
        RCP<Epetra_SerialDenseVector> Adiff = rcp(new Epetra_SerialDenseVector);
        // second partial derivatives
        RCP<Epetra_SerialDenseMatrix> Adiff2 = rcp(new Epetra_SerialDenseMatrix);

        //call submethod
        ComputeAreaDeriv(xscurr, NumNode(),numdim*NumNode(), elearea, Adiff, Adiff2);
        //update elematrices and elevectors
        elevector1 = *Adiff;
        elevector1.Scale(-1.0);
        elevector2 = elevector1;
        elematrix1 = *Adiff2;
        elematrix1.Scale(-1.0);
        elevector3[0] = elearea;
      }
      break;
      default:
        dserror("Unimplemented type of action for StructuralSurface");

  }
  return 0;
  }

/*----------------------------------------------------------------------*
 * Compute Volume enclosed by surface.                          tk 10/07*
 * ---------------------------------------------------------------------*/
double DRT::ELEMENTS::StructuralSurface::ComputeConstrVols
(
    const LINALG::SerialDenseMatrix& xc,
    const int numnode
)
{
  double V = 0.0;

  //Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = 0; indc < 3; indc++)
  {
    //split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    LINALG::SerialDenseMatrix ab= xc;
    LINALG::SerialDenseVector c (numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i,indc) = 0.0; // project by z_i = 0.0
      c(i) = xc(i,indc); // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc+1)%3;
    int indb = (indc+2)%3;

    // get gaussrule
    const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    LINALG::SerialDenseVector  funct(numnode);
    LINALG::SerialDenseMatrix  deriv(2,numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of the element
      DRT::UTILS::shape_function_2D(funct,e0,e1,Shape());
      DRT::UTILS::shape_function_2D_deriv1(deriv,e0,e1,Shape());

      double detA;
      // compute "metric tensor" deriv*ab, which is a 2x3 matrix with zero indc'th column
      LINALG::SerialDenseMatrix metrictensor(2,3);
      metrictensor.Multiply('N','N',1.0,deriv,ab,0.0);
      //LINALG::SerialDenseMatrix metrictensor(2,2);
      //metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      detA =  metrictensor(0,inda)*metrictensor(1,indb)-metrictensor(0,indb)*metrictensor(1,inda);
      const double dotprodc = funct.Dot(c);
      // add weighted volume at gausspoint
      V -= dotprodc*detA*intpoints.qwgt[gpid];

    }
  }
  return V/3.0;
}

/*----------------------------------------------------------------------*
 * Compute volume and its first and second derivatives          tk 02/09*
 * with respect to the displacements                                    *
 * ---------------------------------------------------------------------*/
void DRT::ELEMENTS::StructuralSurface::ComputeVolDeriv
(
  const LINALG::SerialDenseMatrix& xc,
  const int numnode,
  const int ndof,
  double& V,
  RCP<Epetra_SerialDenseVector> Vdiff1,
  RCP<Epetra_SerialDenseMatrix> Vdiff2,
  const int minindex,
  const int maxindex
)
{
  // necessary constants
  const int numdim = 3;
  const double invnumind = 1.0/(maxindex-minindex+1.0);

  // initialize
  V = 0.0;
  Vdiff1->Size(ndof);
  if (Vdiff2!=null) Vdiff2->Shape(ndof, ndof);

  //Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = minindex; indc < maxindex+1; indc++)
  {
    //split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    LINALG::SerialDenseMatrix ab= xc;
    LINALG::SerialDenseVector c (numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i,indc) = 0.0; // project by z_i = 0.0
      c(i) = xc(i,indc); // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc+1)%3;
    int indb = (indc+2)%3;

    // get gaussrule
    const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    LINALG::SerialDenseVector  funct(numnode);
    LINALG::SerialDenseMatrix  deriv(2,numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of the element
      DRT::UTILS::shape_function_2D(funct,e0,e1,Shape());
      DRT::UTILS::shape_function_2D_deriv1(deriv,e0,e1,Shape());

      // evaluate Jacobi determinant, for projected dA*
      vector<double> normal(numdim);
      double detA;
      // compute "metric tensor" deriv*xy, which is a 2x3 matrix with zero 3rd column
      LINALG::SerialDenseMatrix metrictensor(2,numdim);
      metrictensor.Multiply('N','N',1.0,deriv,ab,0.0);
      //metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      detA =  metrictensor(0,inda)*metrictensor(1,indb)-metrictensor(0,indb)*metrictensor(1,inda);
      const double dotprodc = funct.Dot(c);
      // add weighted volume at gausspoint
      V -= dotprodc*detA*intpoints.qwgt[gpid];

      //-------- compute first derivative
      for (int i = 0; i < numnode ; i++)
      {
        (*Vdiff1)[3*i+inda] += invnumind*intpoints.qwgt[gpid]*dotprodc*(deriv(0,i)*metrictensor(1,indb)-metrictensor(0,indb)*deriv(1,i));
        (*Vdiff1)[3*i+indb] += invnumind*intpoints.qwgt[gpid]*dotprodc*(deriv(1,i)*metrictensor(0,inda)-metrictensor(1,inda)*deriv(0,i));
        (*Vdiff1)[3*i+indc] += invnumind*intpoints.qwgt[gpid]*funct[i]*detA;
      }

      //-------- compute second derivative
      if (Vdiff2!=null)
      {
        for (int i = 0; i < numnode ; i++)
        {
          for (int j = 0; j < numnode ; j++)
          {
            //"diagonal" (dV)^2/(dx_i dx_j) = 0, therefore only six entries have to be specified
            (*Vdiff2)(3*i+inda,3*j+indb) += invnumind*intpoints.qwgt[gpid]*dotprodc*(deriv(0,i)*deriv(1,j)-deriv(1,i)*deriv(0,j));
            (*Vdiff2)(3*i+indb,3*j+inda) += invnumind*intpoints.qwgt[gpid]*dotprodc*(deriv(0,j)*deriv(1,i)-deriv(1,j)*deriv(0,i));
            (*Vdiff2)(3*i+inda,3*j+indc) += invnumind*intpoints.qwgt[gpid]*funct[j]*(deriv(0,i)*metrictensor(1,indb)-metrictensor(0,indb)*deriv(1,i));
            (*Vdiff2)(3*i+indc,3*j+inda) += invnumind*intpoints.qwgt[gpid]*funct[i]*(deriv(0,j)*metrictensor(1,indb)-metrictensor(0,indb)*deriv(1,j));
            (*Vdiff2)(3*i+indb,3*j+indc) += invnumind*intpoints.qwgt[gpid]*funct[j]*(deriv(1,i)*metrictensor(0,inda)-metrictensor(1,inda)*deriv(0,i));
            (*Vdiff2)(3*i+indc,3*j+indb) += invnumind*intpoints.qwgt[gpid]*funct[i]*(deriv(1,j)*metrictensor(0,inda)-metrictensor(1,inda)*deriv(0,j));
          }
        }
      }

    }
  }
  V*=invnumind;
  return;
}


/*----------------------------------------------------------------------*
 * Compute surface area and its first and second derivatives    lw 05/08*
 * with respect to the displacements                                    *
 * ---------------------------------------------------------------------*/
void DRT::ELEMENTS::StructuralSurface::ComputeAreaDeriv(const LINALG::SerialDenseMatrix& x,
                                                        const int numnode,
                                                        const int ndof,
                                                        double& A,
                                                        RCP<Epetra_SerialDenseVector> Adiff,
                                                        RCP<Epetra_SerialDenseMatrix> Adiff2)
{
  // initialization
  A = 0.;
  Adiff->Size(ndof);

  if (Adiff2!=null) Adiff2->Shape(ndof, ndof);

  const DRT::UTILS::IntegrationPoints2D  intpoints(gaussrule_);

  int ngp = intpoints.nquad;

  // allocate vector for shape functions and matrix for derivatives
  LINALG::SerialDenseMatrix  deriv(2,numnode);
  LINALG::SerialDenseMatrix  dxyzdrs(2,3);

  /*----------------------------------------------------------------------*
   |               start loop over integration points                     |
   *----------------------------------------------------------------------*/

  for (int gpid = 0; gpid < ngp; ++gpid)
  {
    const double e0 = intpoints.qxg[gpid][0];
    const double e1 = intpoints.qxg[gpid][1];

    // get derivatives of shape functions in the plane of the element
    DRT::UTILS::shape_function_2D_deriv1(deriv,e0,e1,Shape());

    vector<double> normal(3);
    double detA;
    SurfaceIntegration(detA,normal,x,deriv);
    A += detA*intpoints.qwgt[gpid];

    blitz::Array<double,2> ddet(3,ndof);
    blitz::Array<double,3> ddet2(3,ndof,ndof);
    ddet2 = 0.;
    blitz::Array<double,1> jacobi_deriv(ndof);

    dxyzdrs.Multiply('N','N',1.0,deriv,x,0.0);

    /*--------------- derivation of minor determiants of the Jacobian
     *----------------------------- with respect to the displacements */
    for (int i=0;i<numnode;++i)
    {
      ddet(0,3*i)   = 0.;
      ddet(0,3*i+1) = deriv(0,i)*dxyzdrs(1,2)-deriv(1,i)*dxyzdrs(0,2);
      ddet(0,3*i+2) = deriv(1,i)*dxyzdrs(0,1)-deriv(0,i)*dxyzdrs(1,1);

      ddet(1,3*i)   = deriv(1,i)*dxyzdrs(0,2)-deriv(0,i)*dxyzdrs(1,2);
      ddet(1,3*i+1) = 0.;
      ddet(1,3*i+2) = deriv(0,i)*dxyzdrs(1,0)-deriv(1,i)*dxyzdrs(0,0);

      ddet(2,3*i)   = deriv(0,i)*dxyzdrs(1,1)-deriv(1,i)*dxyzdrs(0,1);
      ddet(2,3*i+1) = deriv(1,i)*dxyzdrs(0,0)-deriv(0,i)*dxyzdrs(1,0);
      ddet(2,3*i+2) = 0.;

      jacobi_deriv(i*3)   = 1/detA*(normal[2]*ddet(2,3*i  )+normal[1]*ddet(1,3*i  ));
      jacobi_deriv(i*3+1) = 1/detA*(normal[2]*ddet(2,3*i+1)+normal[0]*ddet(0,3*i+1));
      jacobi_deriv(i*3+2) = 1/detA*(normal[0]*ddet(0,3*i+2)+normal[1]*ddet(1,3*i+2));
    }

    /*--- calculation of first derivatives of current interfacial area
     *----------------------------- with respect to the displacements */
    for (int i=0;i<ndof;++i)
    {
      (*Adiff)[i] += jacobi_deriv(i)*intpoints.qwgt[gpid];
    }

    if (Adiff2!=null)
    {
      /*--------- second derivates of minor determiants of the Jacobian
       *----------------------------- with respect to the displacements */
      for (int n=0;n<numnode;++n)
      {
        for (int o=0;o<numnode;++o)
        {
          ddet2(0,n*3+1,o*3+2) = deriv(0,n)*deriv(1,o)-deriv(1,n)*deriv(0,o);
          ddet2(0,n*3+2,o*3+1) = - ddet2(0,n*3+1,o*3+2);

          ddet2(1,n*3  ,o*3+2) = deriv(1,n)*deriv(0,o)-deriv(0,n)*deriv(1,o);
          ddet2(1,n*3+2,o*3  ) = - ddet2(1,n*3,o*3+2);

          ddet2(2,n*3  ,o*3+1) = ddet2(0,n*3+1,o*3+2);
          ddet2(2,n*3+1,o*3  ) = - ddet2(2,n*3,o*3+1);
        }
      }

      /*- calculation of second derivatives of current interfacial areas
       *----------------------------- with respect to the displacements */
      for (int i=0;i<ndof;++i)
      {
        int var1, var2;

        if (i%3==0)           // displacement in x-direction
        {
          var1 = 1;
          var2 = 2;
        }
        else if ((i-1)%3==0)  // displacement in y-direction
        {
          var1 = 0;
          var2 = 2;
        }
        else if ((i-2)%3==0)  // displacement in z-direction
        {
          var1 = 0;
          var2 = 1;
        }
        else
        {
          dserror("calculation of second derivatives of interfacial area failed");
          exit(1);
        }

        for (int j=0;j<ndof;++j)
        {
          (*Adiff2)(i,j) += (-1/detA*jacobi_deriv(j)*jacobi_deriv(i)+1/detA*
                             (ddet(var1,i)*ddet(var1,j)+normal[var1]*ddet2(var1,i,j)+
                              ddet(var2,i)*ddet(var2,j)+normal[var2]*ddet2(var2,i,j)))*intpoints.qwgt[gpid];
        }
      }
    }
  }

  return;
}

#endif  // #ifdef CCADISCRET
#endif // #ifdef D_SOLID3
