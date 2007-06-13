/*!----------------------------------------------------------------------
\file shell8_line_evaluate.cpp
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SHELL8
#ifdef CCADISCRET
#ifdef TRILINOS_PACKAGE

#include "shell8.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_timecurve.H"

extern "C"
{
#include "../headers/standardtypes.h"
#include "../shell8/shell8.h"
}
#include "../drt_lib/dstrc.H"


/*----------------------------------------------------------------------*
 |  Integrate a Line Neumann boundary condition (public)     mwgee 01/07|
 *----------------------------------------------------------------------*/
int DRT::Elements::Shell8Line::EvaluateNeumann(
                                           ParameterList& params,
                                           DRT::Discretization&      discretization,
                                           DRT::Condition&           condition,
                                           vector<int>&              lm,
                                           Epetra_SerialDenseVector& elevec1)
{
  DSTraceHelper dst("Shell8Line::EvaluateNeumann");
  RefCountPtr<const Epetra_Vector> disp = discretization.GetState("displacement");
  if (disp==null) dserror("Cannot get state vector 'displacement'");
  vector<double> mydisp(lm.size());
  DRT::Utils::ExtractMyValues(*disp,mydisp,lm);

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
    curvefac = DRT::TimeCurveManager::Instance().Curve(curvenum).f(time);

  // init gaussian points of parent element
  S8_DATA s8data;
  parent_->s8_integration_points(s8data);

  // number of parent element nodes
  const int iel = parent_->NumNode();

  const int nir = parent_->ngp_[0];
  const int nis = parent_->ngp_[1];
  const int numdf = 6;
  const vector<double>* thick = parent_->data_.Get<vector<double> >("thick");
  if (!thick) dserror("Cannot find vector of nodal thicknesses");

  const Epetra_SerialDenseMatrix* a3ref = parent_->data_.Get<Epetra_SerialDenseMatrix>("a3ref");
  if (!a3ref) dserror("Cannot find array of directors");

  vector<double> funct(iel);
  Epetra_SerialDenseMatrix deriv(2,iel);

  double a3r[3][MAXNOD_SHELL8];
  double xrefe[3][MAXNOD_SHELL8];
  double xjm[3][3];

  // get geometry
  for (int k=0; k<iel; ++k)
  {
    const double h2 = (*thick)[k];

    a3r[0][k] = (*a3ref)(0,k)*h2;
    a3r[1][k] = (*a3ref)(1,k)*h2;
    a3r[2][k] = (*a3ref)(2,k)*h2;

    xrefe[0][k] = parent_->Nodes()[k]->X()[0];
    xrefe[1][k] = parent_->Nodes()[k]->X()[1];
    xrefe[2][k] = parent_->Nodes()[k]->X()[2];
  }

  // check which line this is to the parent and get no. of gaussian points
  const int line = lline_;
  int ngp = 0;
  if (line==0 || line==2) ngp = nir;
  else                    ngp = nis;

  // make coords of integration points
  double xgp[3];   // coord of integration point in line direction
  double wgp[3];   // weigth of this point
  double xgp_n[3]; // coord of intgration point orthogonal to line direction
  int    dir;      // direction of integration, either 0 or 1
  int lnode[3];    // local node numbers of this line w.r.t to parent element
  switch (line)
  {
  case 0:
    for (int i=0; i<ngp; ++i)
    {
      xgp[i]   = s8data.xgpr[i];
      wgp[i]   = s8data.wgtr[i];
      xgp_n[i] = 1.0;
    }
    dir = 0; // direction of integration is r
    lnode[0] = 0;
    lnode[1] = 1;
    lnode[2] = 4;
  break;
  case 2:
    for (int i=0; i<ngp; ++i)
    {
      xgp[i]   = s8data.xgpr[i];
      wgp[i]   = s8data.wgtr[i];
      xgp_n[i] = -1.0;
    }
    dir = 0; // direction of integration is r
    lnode[0] = 2;
    lnode[1] = 3;
    lnode[2] = 6;
  break;
  case 1:
    for (int i=0; i<ngp; ++i)
    {
      xgp[i]   = s8data.xgps[i];
      wgp[i]   = s8data.wgts[i];
      xgp_n[i] = -1.0;
    }
    dir = 1; // direction of integration is s
    lnode[0] = 1;
    lnode[1] = 2;
    lnode[2] = 5;
  break;
  case 3:
    for (int i=0; i<ngp; ++i)
    {
      xgp[i]   = s8data.xgps[i];
      wgp[i]   = s8data.wgts[i];
      xgp_n[i] = 1.0;
    }
    dir = 1; // direction of integration is s
    lnode[0] = 3;
    lnode[1] = 0;
    lnode[2] = 7;
  break;
  default:
    dserror("Unknown local line number %d",line);
  break;
  }

  // get values and switches from the condition
  const vector<int>*    onoff = condition.Get<vector<int> >("onoff");
  const vector<double>* val   = condition.Get<vector<double> >("val");


  // do integration
  for (int gp=0; gp<ngp; ++gp)
  {
    // gaussian point and weight
    double e1   = xgp[gp];
    double e2   = xgp_n[gp];
    double facr = wgp[gp];

    // shape function and derivatives at this point
    if (dir==0) // integration in r
      parent_->s8_shapefunctions(funct,deriv,e1,e2,parent_->NumNode(),1);
    else
      parent_->s8_shapefunctions(funct,deriv,e2,e1,parent_->NumNode(),1);
    // covariant metrics
    // g1,g2,g3 stored in xjm
    // Jacobian matrix J = (g1,g2,g3)
    for (int i=0; i<2; ++i)
      for (int j=0; j<3; ++j)
      {
        xjm[i][j] = 0.0;
        for (int k=0; k<iel; ++k)
          xjm[i][j] += deriv(i,k)*xrefe[j][k];
      }
    for (int j=0; j<3; ++j)
    {
      xjm[2][j] = 0.0;
      for (int k=0; k<iel; ++k)
        xjm[2][j] += funct[k] * (*thick)[k] * (*a3ref)(j,k) / 2.0;
    }
    // ds = |g1| in dir=0 and ds = |g2| in dir=1
    double ds = sqrt(xjm[dir][0]*xjm[dir][0]+xjm[dir][1]*xjm[dir][1]+xjm[dir][2]*xjm[dir][2]);
    // load vector ar
    double ar[3];
    // loop the dofs of a node
    // ar[i] = ar[i] * facr * ds * onoff[i] * val[i] * curvefac
    for (int i=0; i<3; ++i)
      ar[i] = facr * ds * (*onoff)[i] * (*val)[i] * curvefac;
    // add load components
    for (int node=0; node<NumNode(); ++node)
      for (int j=0; j<3; ++j)
      {
        elevec1[node*numdf+j] += funct[lnode[node]] *ar[j];
      }
  } // for (int gp=0; gp<ngp; ++gp)
  return 0;
}


#endif  // #ifdef TRILINOS_PACKAGE
#endif  // #ifdef CCADISCRET
#endif // #ifdef D_SHELL8
