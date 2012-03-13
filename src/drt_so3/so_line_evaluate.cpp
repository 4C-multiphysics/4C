/*!----------------------------------------------------------------------
\file so_line_evaluate.cpp
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "so_line.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_serialdensevector.H"
#include "../linalg/linalg_serialdensematrix.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"


/*-----------------------------------------------------------------------*
 * Integrate a Line Neumann boundary condition (public)         gee 04/08|
 * ----------------------------------------------------------------------*/
int DRT::ELEMENTS::StructuralLine::EvaluateNeumann(ParameterList&            params,
                                                   DRT::Discretization&      discretization,
                                                   DRT::Condition&           condition,
                                                   vector<int>&              lm,
                                                   Epetra_SerialDenseVector& elevec1,
                                                   Epetra_SerialDenseMatrix* elemat1)
{
  // get type of condition
  enum LoadType
  {
    neum_none,
    neum_live
  };
  LoadType ltype;
  // spatial or material configuration depends on the type of load
  // currently only material frame used
  enum Configuration
  {
    config_none,
    config_material,
    config_spatial,
    config_both
  };
  Configuration config = config_none;
  const string* type = condition.Get<string>("type");
  if (*type == "neum_live")
  {
    ltype  = neum_live;
    config = config_material;
  }
  else dserror("Unknown type of LineNeumann condition");

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
    curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);

  // element geometry update - currently only material configuration
  const int numnode = NumNode();
  LINALG::SerialDenseMatrix x(numnode,3);
  MaterialConfiguration(x);

  // integration parameters
  const DRT::UTILS::IntegrationPoints1D  intpoints(gaussrule_);
  const int ngp = intpoints.nquad;
  LINALG::SerialDenseVector funct(numnode);
  LINALG::SerialDenseMatrix deriv(1,numnode);
  const DRT::Element::DiscretizationType shape = Shape();

  // integration
  for (int gp = 0; gp < ngp; ++gp)
  {
    // get shape functions and derivatives of element surface
    const double e   = intpoints.qxg[gp][0];
    const double wgt = intpoints.qwgt[gp];
    DRT::UTILS::shape_function_1D(funct,e,shape);
    DRT::UTILS::shape_function_1D_deriv1(deriv,e,shape);
    switch(ltype)
    {
    case neum_live:
    {            // uniform load on reference configuration
      double dL;
      LineIntegration(dL,x,deriv);
      double fac = wgt * dL * curvefac;   // integration factor
      // distribute over element load vector
      for (int nodid=0; nodid < numnode; ++nodid)
        for(int dim=0; dim < 3; ++dim)
          elevec1[nodid*3 + dim] += funct[nodid] * (*onoff)[dim] * (*val)[dim] * fac;
    }
    break;
    default:
      dserror("Unknown type of LineNeumann load");
    break;
    }

  }

  return 0;
}

/*----------------------------------------------------------------------*
 *  (private)                                                  gee 04/08|
 * ---------------------------------------------------------------------*/
void DRT::ELEMENTS::StructuralLine::LineIntegration(double&                          dL,
                                                    const LINALG::SerialDenseMatrix& x,
                                                    const LINALG::SerialDenseMatrix& deriv)
{
  // compute dXYZ / drs
  LINALG::SerialDenseMatrix dxyzdrs(1,3);
  dxyzdrs.Multiply('N','N',1.0,deriv,x,0.0);
  dL=0.0;
  for (int i=0; i<3; ++i) dL += dxyzdrs(0,i)*dxyzdrs(0,i);
  dL = sqrt(dL);
  return;
}



#endif  // #ifdef CCADISCRET
