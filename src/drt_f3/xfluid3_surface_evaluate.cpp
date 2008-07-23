/*!----------------------------------------------------------------------
\file xfluid3_surface_evaluate.cpp
\brief

Integrate a Surface Neumann boundary condition on a given boundary
element (tri or quad)

<pre>
Maintainer: Axel Gerstenberger
            gerstenberger@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>

*----------------------------------------------------------------------*/
#ifdef D_FLUID3
#ifdef CCADISCRET

#include "xfluid3.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_timecurve.H"


/*----------------------------------------------------------------------*
 |  evaluate the element (public)                            g.bau 03/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::XFluid3Surface::Evaluate(
    ParameterList&            params,
    DRT::Discretization&      discretization,
    std::vector<int>&         lm,
    Epetra_SerialDenseMatrix& elemat1,
    Epetra_SerialDenseMatrix& elemat2,
    Epetra_SerialDenseVector& elevec1,
    Epetra_SerialDenseVector& elevec2,
    Epetra_SerialDenseVector& elevec3)
{
    DRT::ELEMENTS::XFluid3Surface::ActionType act = XFluid3Surface::none;
    string action = params.get<string>("action","none");
    if (action == "none") dserror("No action supplied");
    else if (action == "integrate_Shapefunction")
        act = XFluid3Surface::integrate_Shapefunction;
    else dserror("Unknown type of action for Fluid3_Surface");

    switch(act)
    {
    case integrate_Shapefunction:
    {
      RCP<const Epetra_Vector> dispnp;
      std::vector<double> mydispnp;

      dispnp = discretization.GetState("dispnp");
      if (dispnp!=null)
      {
        mydispnp.resize(lm.size());
        DRT::UTILS::ExtractMyValues(*dispnp,mydispnp,lm);
      }
      IntegrateShapeFunction(params,discretization,lm,elevec1,mydispnp);
      break;
    }
    default:
        dserror("Unknown type of action for Fluid3_Surface");
    } // end of switch(act)

    return 0;
}

/*----------------------------------------------------------------------*
 |  Integrate a Surface Neumann boundary condition (public)  gammi 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::XFluid3Surface::EvaluateNeumann(
    ParameterList&            params,
    DRT::Discretization&      discretization,
    DRT::Condition&           condition,
    std::vector<int>&         lm,
    Epetra_SerialDenseVector& elevec1)
{
  // there are 3 velocities and 1 pressure
  return 0;
#if 0
  const int numdf = 4;

  const double thsl = params.get("thsl",0.0);

  const DiscretizationType distype = this->Shape();

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

  // get values and switches from the condition
  const vector<int>*    onoff = condition.Get<vector<int> >   ("onoff");
  const vector<double>* val   = condition.Get<vector<double> >("val"  );

  // set number of nodes
  const int iel   = this->NumNode();

  GaussRule2D  gaussrule = intrule2D_undefined;
  switch(distype)
  {
  case quad4:
      gaussrule = intrule_quad_4point;
      break;
  case quad8: case quad9:
      gaussrule = intrule_quad_9point;
      break;
  case tri3 :
      gaussrule = intrule_tri_3point;
      break;
  case tri6:
      gaussrule = intrule_tri_6point;
      break;
  default:
      dserror("shape type unknown!\n");
  }

  // allocate vector for shape functions and matrix for derivatives
  Epetra_SerialDenseVector  funct       (iel);
  Epetra_SerialDenseMatrix  deriv       (2,iel);

  // node coordinates
  Epetra_SerialDenseMatrix      xyze        (3,iel);

  // the metric tensor and the area of an infintesimal surface element
  Epetra_SerialDenseMatrix  metrictensor(2,2);
  double                        drs;

  // get node coordinates
  for(int i=0;i<iel;i++)
  {
    xyze(0,i)=this->Nodes()[i]->X()[0];
    xyze(1,i)=this->Nodes()[i]->X()[1];
    xyze(2,i)=this->Nodes()[i]->X()[2];
  }

  /*----------------------------------------------------------------------*
  |               start loop over integration points                     |
  *----------------------------------------------------------------------*/
  const IntegrationPoints2D  intpoints = getIntegrationPoints2D(gaussrule);
  for (int gpid=0; gpid<intpoints.nquad; gpid++)
  {
    const double e0 = intpoints.qxg[gpid][0];
    const double e1 = intpoints.qxg[gpid][1];

    // get shape functions and derivatives in the plane of the element
    shape_function_2D(funct, e0, e1, distype);
    shape_function_2D_deriv1(deriv, e0, e1, distype);

    // compute measure tensor for surface element and the infinitesimal
    // area element drs for the integration
    f3_metric_tensor_for_surface(xyze,deriv,metrictensor,&drs);

    // values are multiplied by the product from inf. area element,
    // the gauss weight, the timecurve factor and the constant
    // belonging to the time integration algorithm (theta*dt for
    // one step theta, 2/3 for bdf with dt const.)
    const double fac = intpoints.qwgt[gpid] * drs * curvefac * thsl;

    for (int node=0;node<iel;++node)
    {
      for(int dim=0;dim<3;dim++)
      {
        elevec1[node*numdf+dim]+=
          funct[node] * (*onoff)[dim] * (*val)[dim] * fac;
      }
    }

  } /* end of loop over integration points gpid */

  return 0;
#endif
}


/* compute kovariant metric tensor G for fluid element        gammi 04/07

                        +-       -+
                        | g11 g12 |
                    G = |         |
                        | g12 g22 |
                        +-       -+

 where (o denotes the inner product, xyz a vector)


                            dxyz   dxyz
                    g11 =   ---- o ----
                             dr     dr

                            dxyz   dxyz
                    g12 =   ---- o ----
                             dr     ds

                            dxyz   dxyz
                    g22 =   ---- o ----
                             ds     ds


 and the square root of the first fundamental form


                          +--------------+
                         /               |
           sqrtdetg =   /  g11*g22-g12^2
                      \/

 they are needed for the integration over the surface element

*/
void  DRT::ELEMENTS::XFluid3Surface::f3_metric_tensor_for_surface(
  const Epetra_SerialDenseMatrix  xyze,
  const Epetra_SerialDenseMatrix  deriv,
  Epetra_SerialDenseMatrix&       metrictensor,
  double                         *sqrtdetg)
{
  /*
  |                                              0 1 2
  |                                             +-+-+-+
  |       0 1 2              0...iel-1          | | | | 0
  |      +-+-+-+             +-+-+-+-+          +-+-+-+
  |      | | | | 1           | | | | | 0        | | | | .
  |      +-+-+-+       =     +-+-+-+-+       *  +-+-+-+ .
  |      | | | | 2           | | | | | 1        | | | | .
  |      +-+-+-+             +-+-+-+-+          +-+-+-+
  |                                             | | | | iel-1
  |                                             +-+-+-+
  |
  |       dxyzdrs             deriv              xyze^T
  |
  |
  |                                     +-            -+
  |  	   	    	    	        | dx   dy   dz |
  |  	   	    	    	        | --   --   -- |
  | 	   	   	   	        | dr   dr   dr |
  | 	yields               dxyzdrs =  |              |
  |  	   	    	    	        | dx   dy   dz |
  |  	   	    	    	        | --   --   -- |
  | 	   	   	   	        | ds   ds   ds |
  |                                     +-            -+
  |
  */
  Epetra_SerialDenseMatrix dxyzdrs (2,3);

  dxyzdrs.Multiply('N','T',1.0,deriv,xyze,0.0);

  /*
  |
  |      +-           -+    +-            -+   +-            -+ T
  |      |             |    | dx   dy   dz |   | dx   dy   dz |
  |      |  g11   g12  |    | --   --   -- |   | --   --   -- |
  |      |             |    | dr   dr   dr |   | dr   dr   dr |
  |      |             |  = |              | * |              |
  |      |             |    | dx   dy   dz |   | dx   dy   dz |
  |      |  g21   g22  |    | --   --   -- |   | --   --   -- |
  |      |             |    | ds   ds   ds |   | ds   ds   ds |
  |      +-           -+    +-            -+   +-            -+
  |
  | the calculation of g21 is redundant since g21=g12
  */
  metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);

/*
                          +--------------+
                         /               |
           sqrtdetg =   /  g11*g22-g12^2
                      \/
*/

  sqrtdetg[0]= sqrt(metrictensor(0,0)*metrictensor(1,1)
                    -
                    metrictensor(0,1)*metrictensor(1,0));

  return;
}

/*----------------------------------------------------------------------*
 |  Integrate shapefunctions over surface (private)          g.bau 07/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XFluid3Surface::IntegrateShapeFunction(
    ParameterList&                   params,
    DRT::Discretization&             discretization,
    const std::vector<int>&          lm,
    Epetra_SerialDenseVector&        elevec1,
    const std::vector<double>&       edispnp)
{
  // there are 3 velocities and 1 pressure
  return;
  const int numdf = 4;

//  const double thsl = params.get("thsl",1.0);

  const DiscretizationType distype = this->Shape();

/*  // find out whether we will use a time curve
  bool usetime = true;
  const double time = params.get("total time",-1.0);
  if (time<0.0) usetime = false;
*/

  // set number of nodes
  const int iel   = this->NumNode();

  DRT::UTILS::GaussRule2D  gaussrule = DRT::UTILS::intrule2D_undefined;
  switch(distype)
  {
  case quad4:
      gaussrule = DRT::UTILS::intrule_quad_4point;
      break;
  case quad8: case quad9:
      gaussrule = DRT::UTILS::intrule_quad_9point;
      break;
  case tri3 :
      gaussrule = DRT::UTILS::intrule_tri_3point;
      break;
  case tri6:
      gaussrule = DRT::UTILS::intrule_tri_6point;
      break;
  default:
      dserror("shape type unknown!\n");
  }

    // allocate vector for shape functions and matrix for derivatives
  Epetra_SerialDenseVector      funct       (iel);
  Epetra_SerialDenseMatrix 	deriv       (2,iel);

  // node coordinates
  Epetra_SerialDenseMatrix      xyze        (3,iel);

  // the metric tensor and the area of an infintesimal surface element
  Epetra_SerialDenseMatrix 	metrictensor(2,2);
  double                        drs;

  // get node coordinates
  for(int i=0;i<iel;i++)
  {
    xyze(0,i)=this->Nodes()[i]->X()[0];
    xyze(1,i)=this->Nodes()[i]->X()[1];
    xyze(2,i)=this->Nodes()[i]->X()[2];
  }

  if (edispnp.size()!=0)
  {
    for (int i=0;i<iel;i++)
    {
      xyze(0,i) += edispnp[4*i];
      xyze(1,i) += edispnp[4*i+1];
      xyze(2,i) += edispnp[4*i+2];
    }
  }

  /*----------------------------------------------------------------------*
  |               start loop over integration points                     |
  *----------------------------------------------------------------------*/
  const DRT::UTILS::IntegrationPoints2D  intpoints = DRT::UTILS::getIntegrationPoints2D(gaussrule);

  for (int gpid=0; gpid<intpoints.nquad; gpid++)
  {
    const double e0 = intpoints.qxg[gpid][0];
    const double e1 = intpoints.qxg[gpid][1];

    // get shape functions and derivatives in the plane of the element
    DRT::UTILS::shape_function_2D(funct, e0, e1, distype);
    DRT::UTILS::shape_function_2D_deriv1(deriv, e0, e1, distype);

    // compute measure tensor for surface element and the infinitesimal
    // area element drs for the integration

    f3_metric_tensor_for_surface(xyze,deriv,metrictensor,&drs);

    // values are multiplied by the product from inf. area element,
    // the gauss weight and the constant
    // belonging to the time integration algorithm (theta*dt for
    // one step theta, 2/3 for bdf with dt const.)

    //const double fac = intpoints.qwgt[gpid] * drs * thsl;
    const double fac = intpoints.qwgt[gpid] * drs;

    for (int node=0;node<iel;++node)
    {
      for(int dim=0;dim<3;dim++)
      {
        elevec1[node*numdf+dim]+=
          funct[node] * fac;
      }
    }

  } /* end of loop over integration points gpid */


return;
} // DRT::ELEMENTS::XFluid3Surface::IntegrateShapeFunction

#endif  // #ifdef CCADISCRET
#endif // #ifdef D_FLUID3
