/*-----------------------------------------------------------------------*/
/*! \file
\brief Shape function repository for mortar coupling element

\level 1

*/
/*-----------------------------------------------------------------------*/

#include "baci_discretization_fem_general_utils_nurbs_shapefunctions.H"
#include "baci_linalg_serialdensematrix.H"
#include "baci_linalg_serialdensevector.H"
#include "baci_linalg_utils_densematrix_inverse.H"
#include "baci_linalg_utils_densematrix_multiply.H"
#include "baci_mortar_defines.H"
#include "baci_mortar_element.H"
#include "baci_mortar_node.H"
#include "baci_mortar_shape_utils.H"

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  1D/2D shape function repository                           popp 04/08|
 *----------------------------------------------------------------------*/
void MORTAR::Element::ShapeFunctions(MORTAR::Element::ShapeType shape, const double* xi,
    CORE::LINALG::SerialDenseVector& val, CORE::LINALG::SerialDenseMatrix& deriv)
{
  switch (shape)
  {
    // *********************************************************************
    // 1D standard linear shape functions (line2)
    // (used for interpolation of displacement field)
    // *********************************************************************
    case MORTAR::Element::lin1D:
    {
      val[0] = 0.5 * (1 - xi[0]);
      val[1] = 0.5 * (1 + xi[0]);
      deriv(0, 0) = -0.5;
      deriv(1, 0) = 0.5;
      break;
    }
    // *********************************************************************
    // 1D modified standard shape functions (const replacing linear, line2)
    // (used for interpolation of Lagrange mult. field near boundaries)
    // *********************************************************************
    case MORTAR::Element::lin1D_edge0:
    {
      dserror(
          "ERROR: explicit edge modification is outdated! We apply a genreal transformaiton "
          "instead");
      val[0] = 0.0;
      val[1] = 1.0;
      deriv(0, 0) = 0.0;
      deriv(1, 0) = 0.0;
      break;
    }
    // *********************************************************************
    // 1D modified standard shape functions (const replacing linear, line2)
    // (used for interpolation of Lagrange mult. field near boundaries)
    // *********************************************************************
    case MORTAR::Element::lin1D_edge1:
    {
      dserror(
          "ERROR: explicit edge modification is outdated! We apply a genreal transformaiton "
          "instead");
      val[0] = 1.0;
      val[1] = 0.0;
      deriv(0, 0) = 0.0;
      deriv(1, 0) = 0.0;
      break;
    }
    // *********************************************************************
    // 2D standard linear shape functions (tri3)
    // (used for interpolation of displacement field)
    // *********************************************************************
    case MORTAR::Element::lin2D:
    {
      val[0] = 1.0 - xi[0] - xi[1];
      val[1] = xi[0];
      val[2] = xi[1];
      deriv(0, 0) = -1.0;
      deriv(0, 1) = -1.0;
      deriv(1, 0) = 1.0;
      deriv(1, 1) = 0.0;
      deriv(2, 0) = 0.0;
      deriv(2, 1) = 1.0;
      break;
    }
      // *********************************************************************
      // 2D standard bilinear shape functions (quad4)
      // (used for interpolation of displacement field)
      // *********************************************************************
    case MORTAR::Element::bilin2D:
    {
      val[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
      val[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
      val[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
      val[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);
      deriv(0, 0) = -0.25 * (1.0 - xi[1]);
      deriv(0, 1) = -0.25 * (1.0 - xi[0]);
      deriv(1, 0) = 0.25 * (1.0 - xi[1]);
      deriv(1, 1) = -0.25 * (1.0 + xi[0]);
      deriv(2, 0) = 0.25 * (1.0 + xi[1]);
      deriv(2, 1) = 0.25 * (1.0 + xi[0]);
      deriv(3, 0) = -0.25 * (1.0 + xi[1]);
      deriv(3, 1) = 0.25 * (1.0 - xi[0]);
      break;
    }
      // *********************************************************************
      // 1D standard quadratic shape functions (line3)
      // (used for interpolation of displacement field)
      // *********************************************************************
    case MORTAR::Element::quad1D:
    {
      val[0] = 0.5 * xi[0] * (xi[0] - 1.0);
      val[1] = 0.5 * xi[0] * (xi[0] + 1.0);
      val[2] = (1.0 - xi[0]) * (1.0 + xi[0]);
      deriv(0, 0) = xi[0] - 0.5;
      deriv(1, 0) = xi[0] + 0.5;
      deriv(2, 0) = -2.0 * xi[0];
      break;
    }
      // *********************************************************************
      // 1D modified (hierarchical) quadratic shape functions (line3)
      // (used in combination with linear dual LM field in 2D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::quad1D_hierarchical:
    {
      val[0] = 0.5 * (1 - xi[0]);
      val[1] = 0.5 * (1 + xi[0]);
      val[2] = (1 - xi[0]) * (1 + xi[0]);

      deriv(0, 0) = -0.5;
      deriv(1, 0) = 0.5;
      deriv(2, 0) = -2.0 * xi[0];
      break;
    }
      // *********************************************************************
      // 1D modified quadratic shape functions (line3)
      // (used in combination with quadr dual LM field in 2D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::quad1D_modified:
    {
      dserror("Quadratic LM for quadratic interpolation in 2D not available!");
      break;
    }
      // *********************************************************************
      // 1D modified standard shape functions (linear replacing quad, line3)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // *********************************************************************
    case MORTAR::Element::quad1D_edge0:
    {
      dserror(
          "ERROR: explicit edge modification is outdated! We apply a genreal transformaiton "
          "instead");
      val[0] = 0.0;
      val[1] = xi[0];
      val[2] = 1.0 - xi[0];
      deriv(0, 0) = 0.0;
      deriv(1, 0) = 1.0;
      deriv(2, 0) = -1.0;
      break;
    }
      // *********************************************************************
      // 1D modified standard shape functions (linear replacing quad, line3)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // *********************************************************************
    case MORTAR::Element::quad1D_edge1:
    {
      dserror(
          "ERROR: explicit edge modification is outdated! We apply a genreal transformaiton "
          "instead");
      val[0] = -xi[0];
      val[1] = 0.0;
      val[2] = 1.0 + xi[0];
      deriv(0, 0) = -1.0;
      deriv(1, 0) = 0.0;
      deriv(2, 0) = 1.0;
      break;
    }
      // *********************************************************************
      // 1D linear part of standard quadratic shape functions (line3)
      // (used for linear interpolation of std LM field in 2D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::quad1D_only_lin:
    {
      val[0] = 0.5 * (1.0 - xi[0]);
      val[1] = 0.5 * (1.0 + xi[0]);
      val[2] = 0.0;
      deriv(0, 0) = -0.5;
      deriv(1, 0) = 0.5;
      deriv(2, 0) = 0.0;
      break;
    }
      // *********************************************************************
      // 2D standard quadratic shape functions (tri6)
      // (used for interpolation of displacement field)
      // *********************************************************************
    case MORTAR::Element::quad2D:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double t1 = 1.0 - r - s;
      const double t2 = r;
      const double t3 = s;

      val[0] = t1 * (2.0 * t1 - 1.0);
      val[1] = t2 * (2.0 * t2 - 1.0);
      val[2] = t3 * (2.0 * t3 - 1.0);
      val[3] = 4.0 * t2 * t1;
      val[4] = 4.0 * t2 * t3;
      val[5] = 4.0 * t3 * t1;

      deriv(0, 0) = -3.0 + 4.0 * (r + s);
      deriv(0, 1) = -3.0 + 4.0 * (r + s);
      deriv(1, 0) = 4.0 * r - 1.0;
      deriv(1, 1) = 0.0;
      deriv(2, 0) = 0.0;
      deriv(2, 1) = 4.0 * s - 1.0;
      deriv(3, 0) = 4.0 * (1.0 - 2.0 * r - s);
      deriv(3, 1) = -4.0 * r;
      deriv(4, 0) = 4.0 * s;
      deriv(4, 1) = 4.0 * r;
      deriv(5, 0) = -4.0 * s;
      deriv(5, 1) = 4.0 * (1.0 - r - 2.0 * s);

      break;
    }
      // *********************************************************************
      // 2D modified quadratic shape functions (tri6)
      // (used in combination with quadr dual LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::quad2D_modified:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double t1 = 1.0 - r - s;
      const double t2 = r;
      const double t3 = s;

      CORE::LINALG::SerialDenseVector valtmp(NumNode(), 1);
      CORE::LINALG::SerialDenseMatrix derivtmp(NumNode(), 2);

      valtmp[0] = t1 * (2.0 * t1 - 1.0);
      valtmp[1] = t2 * (2.0 * t2 - 1.0);
      valtmp[2] = t3 * (2.0 * t3 - 1.0);
      valtmp[3] = 4.0 * t2 * t1;
      valtmp[4] = 4.0 * t2 * t3;
      valtmp[5] = 4.0 * t3 * t1;

      derivtmp(0, 0) = -3.0 + 4.0 * (r + s);
      derivtmp(0, 1) = -3.0 + 4.0 * (r + s);
      derivtmp(1, 0) = 4.0 * r - 1.0;
      derivtmp(1, 1) = 0.0;
      derivtmp(2, 0) = 0.0;
      derivtmp(2, 1) = 4.0 * s - 1.0;
      derivtmp(3, 0) = 4.0 * (1.0 - 2.0 * r - s);
      derivtmp(3, 1) = -4.0 * r;
      derivtmp(4, 0) = 4.0 * s;
      derivtmp(4, 1) = 4.0 * r;
      derivtmp(5, 0) = -4.0 * s;
      derivtmp(5, 1) = 4.0 * (1.0 - r - 2.0 * s);

      // define constant modification factor 1/5
      // (NOTE: lower factors, e.g. 1/12 would be sufficient here
      // as well, but in order to be globally continuous for mixed
      // meshes with tet10/hex20 elements, we always choose 1/5.)
      const double fac = 1.0 / 5.0;

      // apply constant modification at vertex nodes and PoU
      val[0] = valtmp[0] + (valtmp[3] + valtmp[5]) * fac;
      val[1] = valtmp[1] + (valtmp[3] + valtmp[4]) * fac;
      val[2] = valtmp[2] + (valtmp[4] + valtmp[5]) * fac;
      val[3] = valtmp[3] * (1.0 - 2.0 * fac);
      val[4] = valtmp[4] * (1.0 - 2.0 * fac);
      val[5] = valtmp[5] * (1.0 - 2.0 * fac);

      deriv(0, 0) = derivtmp(0, 0) + (derivtmp(3, 0) + derivtmp(5, 0)) * fac;
      deriv(0, 1) = derivtmp(0, 1) + (derivtmp(3, 1) + derivtmp(5, 1)) * fac;
      deriv(1, 0) = derivtmp(1, 0) + (derivtmp(3, 0) + derivtmp(4, 0)) * fac;
      deriv(1, 1) = derivtmp(1, 1) + (derivtmp(3, 1) + derivtmp(4, 1)) * fac;
      deriv(2, 0) = derivtmp(2, 0) + (derivtmp(4, 0) + derivtmp(5, 0)) * fac;
      deriv(2, 1) = derivtmp(2, 1) + (derivtmp(4, 1) + derivtmp(5, 1)) * fac;
      deriv(3, 0) = derivtmp(3, 0) * (1.0 - 2.0 * fac);
      deriv(3, 1) = derivtmp(3, 1) * (1.0 - 2.0 * fac);
      deriv(4, 0) = derivtmp(4, 0) * (1.0 - 2.0 * fac);
      deriv(4, 1) = derivtmp(4, 1) * (1.0 - 2.0 * fac);
      deriv(5, 0) = derivtmp(5, 0) * (1.0 - 2.0 * fac);
      deriv(5, 1) = derivtmp(5, 1) * (1.0 - 2.0 * fac);

      break;
    }
      // *********************************************************************
      // 2D modified (hierarchical) quadratic shape functions (tri6)
      // (used in combination with linear dual LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::quad2D_hierarchical:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double t1 = 1.0 - r - s;
      const double t2 = r;
      const double t3 = s;

      val[0] = t1;
      val[1] = t2;
      val[2] = t3;
      val[3] = 4.0 * t2 * t1;
      val[4] = 4.0 * t2 * t3;
      val[5] = 4.0 * t3 * t1;

      deriv(0, 0) = -1.0;
      deriv(0, 1) = -1.0;
      deriv(1, 0) = 1.0;
      deriv(1, 1) = 0.0;
      deriv(2, 0) = 0.0;
      deriv(2, 1) = 1.0;
      deriv(3, 0) = 4.0 * (1.0 - 2.0 * r - s);
      deriv(3, 1) = -4.0 * r;
      deriv(4, 0) = 4.0 * s;
      deriv(4, 1) = 4.0 * r;
      deriv(5, 0) = -4.0 * s;
      deriv(5, 1) = 4.0 * (1.0 - r - 2.0 * s);

      break;
    }
      // *********************************************************************
      // 2D linear part of standard quadratic shape functions (tri6)
      // (used for linear interpolation of std LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::quad2D_only_lin:
    {
      val[0] = 1.0 - xi[0] - xi[1];
      val[1] = xi[0];
      val[2] = xi[1];
      val[3] = 0.0;
      val[4] = 0.0;
      val[5] = 0.0;

      deriv(0, 0) = -1.0;
      deriv(0, 1) = -1.0;
      deriv(1, 0) = 1.0;
      deriv(1, 1) = 0.0;
      deriv(2, 0) = 0.0;
      deriv(2, 1) = 1.0;
      deriv(3, 0) = 0.0;
      deriv(3, 1) = 0.0;
      deriv(4, 0) = 0.0;
      deriv(4, 1) = 0.0;
      deriv(5, 0) = 0.0;
      deriv(5, 1) = 0.0;

      break;
    }
      // *********************************************************************
      // 2D serendipity shape functions (quad8)
      // (used for interpolation of displacement field)
      // *********************************************************************
    case MORTAR::Element::serendipity2D:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;
      const double r2 = 1.0 - r * r;
      const double s2 = 1.0 - s * s;

      // values for centernodes are straight forward
      //      0.5*(1-xi*xi)*(1-eta) (0 for xi=+/-1 and eta=+/-1/0
      //                             0 for xi=0    and eta= 1
      //                             1 for xi=0    and eta=-1    )
      // use shape functions on centernodes to zero out the corner node
      // shape functions on the centernodes
      // (0.5 is the value of the linear shape function in the centernode)
      //
      //  0.25*(1-xi)*(1-eta)-0.5*funct[neighbor1]-0.5*funct[neighbor2]

      val[0] = 0.25 * (rm * sm - (r2 * sm + s2 * rm));
      val[1] = 0.25 * (rp * sm - (r2 * sm + s2 * rp));
      val[2] = 0.25 * (rp * sp - (s2 * rp + r2 * sp));
      val[3] = 0.25 * (rm * sp - (r2 * sp + s2 * rm));
      val[4] = 0.5 * r2 * sm;
      val[5] = 0.5 * s2 * rp;
      val[6] = 0.5 * r2 * sp;
      val[7] = 0.5 * s2 * rm;

      deriv(0, 0) = 0.25 * sm * (2 * r + s);
      deriv(0, 1) = 0.25 * rm * (r + 2 * s);
      deriv(1, 0) = 0.25 * sm * (2 * r - s);
      deriv(1, 1) = 0.25 * rp * (2 * s - r);
      deriv(2, 0) = 0.25 * sp * (2 * r + s);
      deriv(2, 1) = 0.25 * rp * (r + 2 * s);
      deriv(3, 0) = 0.25 * sp * (2 * r - s);
      deriv(3, 1) = 0.25 * rm * (2 * s - r);
      deriv(4, 0) = -sm * r;
      deriv(4, 1) = -0.5 * rm * rp;
      deriv(5, 0) = 0.5 * sm * sp;
      deriv(5, 1) = -rp * s;
      deriv(6, 0) = -sp * r;
      deriv(6, 1) = 0.5 * rm * rp;
      deriv(7, 0) = -0.5 * sm * sp;
      deriv(7, 1) = -rm * s;

      break;
    }
      // *********************************************************************
      // 2D modified serendipity shape functions (quad8)
      // (used in combination with quadr dual LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::serendipity2D_modified:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;
      const double r2 = 1.0 - r * r;
      const double s2 = 1.0 - s * s;

      // values for centernodes are straight forward
      //      0.5*(1-xi*xi)*(1-eta) (0 for xi=+/-1 and eta=+/-1/0
      //                             0 for xi=0    and eta= 1
      //                             1 for xi=0    and eta=-1    )
      // use shape functions on centernodes to zero out the corner node
      // shape functions on the centernodes
      // (0.5 is the value of the linear shape function in the centernode)
      //
      //  0.25*(1-xi)*(1-eta)-0.5*funct[neighbor1]-0.5*funct[neighbor2]

      CORE::LINALG::SerialDenseVector valtmp(NumNode(), 1);
      CORE::LINALG::SerialDenseMatrix derivtmp(NumNode(), 2);

      valtmp[0] = 0.25 * (rm * sm - (r2 * sm + s2 * rm));
      valtmp[1] = 0.25 * (rp * sm - (r2 * sm + s2 * rp));
      valtmp[2] = 0.25 * (rp * sp - (s2 * rp + r2 * sp));
      valtmp[3] = 0.25 * (rm * sp - (r2 * sp + s2 * rm));
      valtmp[4] = 0.5 * r2 * sm;
      valtmp[5] = 0.5 * s2 * rp;
      valtmp[6] = 0.5 * r2 * sp;
      valtmp[7] = 0.5 * s2 * rm;

      derivtmp(0, 0) = 0.25 * sm * (2 * r + s);
      derivtmp(0, 1) = 0.25 * rm * (r + 2 * s);
      derivtmp(1, 0) = 0.25 * sm * (2 * r - s);
      derivtmp(1, 1) = 0.25 * rp * (2 * s - r);
      derivtmp(2, 0) = 0.25 * sp * (2 * r + s);
      derivtmp(2, 1) = 0.25 * rp * (r + 2 * s);
      derivtmp(3, 0) = 0.25 * sp * (2 * r - s);
      derivtmp(3, 1) = 0.25 * rm * (2 * s - r);
      derivtmp(4, 0) = -sm * r;
      derivtmp(4, 1) = -0.5 * rm * rp;
      derivtmp(5, 0) = 0.5 * sm * sp;
      derivtmp(5, 1) = -rp * s;
      derivtmp(6, 0) = -sp * r;
      derivtmp(6, 1) = 0.5 * rm * rp;
      derivtmp(7, 0) = -0.5 * sm * sp;
      derivtmp(7, 1) = -rm * s;

      // define constant modification factor 1/5
      const double fac = 1.0 / 5.0;

      // apply constant modification at vertex nodes and PoU
      val[0] = valtmp[0] + (valtmp[4] + valtmp[7]) * fac;
      val[1] = valtmp[1] + (valtmp[4] + valtmp[5]) * fac;
      val[2] = valtmp[2] + (valtmp[5] + valtmp[6]) * fac;
      val[3] = valtmp[3] + (valtmp[6] + valtmp[7]) * fac;
      val[4] = valtmp[4] * (1.0 - 2.0 * fac);
      val[5] = valtmp[5] * (1.0 - 2.0 * fac);
      val[6] = valtmp[6] * (1.0 - 2.0 * fac);
      val[7] = valtmp[7] * (1.0 - 2.0 * fac);

      deriv(0, 0) = derivtmp(0, 0) + (derivtmp(4, 0) + derivtmp(7, 0)) * fac;
      deriv(0, 1) = derivtmp(0, 1) + (derivtmp(4, 1) + derivtmp(7, 1)) * fac;
      deriv(1, 0) = derivtmp(1, 0) + (derivtmp(4, 0) + derivtmp(5, 0)) * fac;
      deriv(1, 1) = derivtmp(1, 1) + (derivtmp(4, 1) + derivtmp(5, 1)) * fac;
      deriv(2, 0) = derivtmp(2, 0) + (derivtmp(5, 0) + derivtmp(6, 0)) * fac;
      deriv(2, 1) = derivtmp(2, 1) + (derivtmp(5, 1) + derivtmp(6, 1)) * fac;
      deriv(3, 0) = derivtmp(3, 0) + (derivtmp(6, 0) + derivtmp(7, 0)) * fac;
      deriv(3, 1) = derivtmp(3, 1) + (derivtmp(6, 1) + derivtmp(7, 1)) * fac;
      deriv(4, 0) = derivtmp(4, 0) * (1.0 - 2.0 * fac);
      deriv(4, 1) = derivtmp(4, 1) * (1.0 - 2.0 * fac);
      deriv(5, 0) = derivtmp(5, 0) * (1.0 - 2.0 * fac);
      deriv(5, 1) = derivtmp(5, 1) * (1.0 - 2.0 * fac);
      deriv(6, 0) = derivtmp(6, 0) * (1.0 - 2.0 * fac);
      deriv(6, 1) = derivtmp(6, 1) * (1.0 - 2.0 * fac);
      deriv(7, 0) = derivtmp(7, 0) * (1.0 - 2.0 * fac);
      deriv(7, 1) = derivtmp(7, 1) * (1.0 - 2.0 * fac);

      break;
    }
      // *********************************************************************
      // 2D modified (hierarchical) serendipity shape functions (quad8)
      // (used in combination with linear dual LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::serendipity2D_hierarchical:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;
      const double r2 = 1.0 - r * r;
      const double s2 = 1.0 - s * s;

      val[0] = 0.25 * rm * sm;
      val[1] = 0.25 * rp * sm;
      val[2] = 0.25 * rp * sp;
      val[3] = 0.25 * rm * sp;
      val[4] = 0.5 * r2 * sm;
      val[5] = 0.5 * s2 * rp;
      val[6] = 0.5 * r2 * sp;
      val[7] = 0.5 * s2 * rm;

      deriv(0, 0) = -0.25 * sm;
      deriv(0, 1) = -0.25 * rm;
      deriv(1, 0) = 0.25 * sm;
      deriv(1, 1) = -0.25 * rp;
      deriv(2, 0) = 0.25 * sp;
      deriv(2, 1) = 0.25 * rp;
      deriv(3, 0) = -0.25 * sp;
      deriv(3, 1) = 0.25 * rm;
      deriv(4, 0) = -sm * r;
      deriv(4, 1) = -0.5 * rm * rp;
      deriv(5, 0) = 0.5 * sm * sp;
      deriv(5, 1) = -rp * s;
      deriv(6, 0) = -sp * r;
      deriv(6, 1) = 0.5 * rm * rp;
      deriv(7, 0) = -0.5 * sm * sp;
      deriv(7, 1) = -rm * s;

      break;
    }
      // *********************************************************************
      // 2D bilinear part of serendipity quadratic shape functions (quad8)
      // (used for linear interpolation of std LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::serendipity2D_only_lin:
    {
      val[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
      val[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
      val[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
      val[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);
      val[4] = 0.0;
      val[5] = 0.0;
      val[6] = 0.0;
      val[7] = 0.0;

      deriv(0, 0) = -0.25 * (1.0 - xi[1]);
      deriv(0, 1) = -0.25 * (1.0 - xi[0]);
      deriv(1, 0) = 0.25 * (1.0 - xi[1]);
      deriv(1, 1) = -0.25 * (1.0 + xi[0]);
      deriv(2, 0) = 0.25 * (1.0 + xi[1]);
      deriv(2, 1) = 0.25 * (1.0 + xi[0]);
      deriv(3, 0) = -0.25 * (1.0 + xi[1]);
      deriv(3, 1) = 0.25 * (1.0 - xi[0]);
      deriv(4, 0) = 0.0;
      deriv(4, 1) = 0.0;
      deriv(5, 0) = 0.0;
      deriv(5, 1) = 0.0;
      deriv(6, 0) = 0.0;
      deriv(6, 1) = 0.0;
      deriv(7, 0) = 0.0;
      deriv(7, 1) = 0.0;

      break;
    }
      // *********************************************************************
      // 2D standard biquadratic shape functions (quad9)
      // (used for interpolation of displacement field)
      // *********************************************************************
    case MORTAR::Element::biquad2D:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;
      const double r2 = 1.0 - r * r;
      const double s2 = 1.0 - s * s;
      const double rh = 0.5 * r;
      const double sh = 0.5 * s;
      const double rs = rh * sh;
      const double rhp = r + 0.5;
      const double rhm = r - 0.5;
      const double shp = s + 0.5;
      const double shm = s - 0.5;

      val[0] = rs * rm * sm;
      val[1] = -rs * rp * sm;
      val[2] = rs * rp * sp;
      val[3] = -rs * rm * sp;
      val[4] = -sh * sm * r2;
      val[5] = rh * rp * s2;
      val[6] = sh * sp * r2;
      val[7] = -rh * rm * s2;
      val[8] = r2 * s2;

      deriv(0, 0) = -rhm * sh * sm;
      deriv(0, 1) = -shm * rh * rm;
      deriv(1, 0) = -rhp * sh * sm;
      deriv(1, 1) = shm * rh * rp;
      deriv(2, 0) = rhp * sh * sp;
      deriv(2, 1) = shp * rh * rp;
      deriv(3, 0) = rhm * sh * sp;
      deriv(3, 1) = -shp * rh * rm;
      deriv(4, 0) = 2.0 * r * sh * sm;
      deriv(4, 1) = shm * r2;
      deriv(5, 0) = rhp * s2;
      deriv(5, 1) = -2.0 * s * rh * rp;
      deriv(6, 0) = -2.0 * r * sh * sp;
      deriv(6, 1) = shp * r2;
      deriv(7, 0) = rhm * s2;
      deriv(7, 1) = 2.0 * s * rh * rm;
      deriv(8, 0) = -2.0 * r * s2;
      deriv(8, 1) = -2.0 * s * r2;

      break;
    }
      // *********************************************************************
      // 2D standard biquadratic shape functions (quad9)
      // (used in combination with quadr dual LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::biquad2D_modified:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;
      const double r2 = 1.0 - r * r;
      const double s2 = 1.0 - s * s;
      const double rh = 0.5 * r;
      const double sh = 0.5 * s;
      const double rs = rh * sh;
      const double rhp = r + 0.5;
      const double rhm = r - 0.5;
      const double shp = s + 0.5;
      const double shm = s - 0.5;

      CORE::LINALG::Matrix<9, 1> valtmp;
      CORE::LINALG::Matrix<9, 2> derivtmp;

      valtmp(0) = rs * rm * sm;
      valtmp(1) = -rs * rp * sm;
      valtmp(2) = rs * rp * sp;
      valtmp(3) = -rs * rm * sp;
      valtmp(4) = -sh * sm * r2;
      valtmp(5) = rh * rp * s2;
      valtmp(6) = sh * sp * r2;
      valtmp(7) = -rh * rm * s2;
      valtmp(8) = r2 * s2;

      derivtmp(0, 0) = -rhm * sh * sm;
      derivtmp(0, 1) = -shm * rh * rm;
      derivtmp(1, 0) = -rhp * sh * sm;
      derivtmp(1, 1) = shm * rh * rp;
      derivtmp(2, 0) = rhp * sh * sp;
      derivtmp(2, 1) = shp * rh * rp;
      derivtmp(3, 0) = rhm * sh * sp;
      derivtmp(3, 1) = -shp * rh * rm;
      derivtmp(4, 0) = 2.0 * r * sh * sm;
      derivtmp(4, 1) = shm * r2;
      derivtmp(5, 0) = rhp * s2;
      derivtmp(5, 1) = -2.0 * s * rh * rp;
      derivtmp(6, 0) = -2.0 * r * sh * sp;
      derivtmp(6, 1) = shp * r2;
      derivtmp(7, 0) = rhm * s2;
      derivtmp(7, 1) = 2.0 * s * rh * rm;
      derivtmp(8, 0) = -2.0 * r * s2;
      derivtmp(8, 1) = -2.0 * s * r2;

      // define constant modification factor
      // (CURRENTLY NOT USED -> ZERO)
      const double fac = 0.0;

      // apply constant modification at vertex nodes and PoU
      val[0] = valtmp(0) + (valtmp(4) + valtmp(7)) * fac + 0.5 * valtmp(8) * fac;
      val[1] = valtmp(1) + (valtmp(4) + valtmp(5)) * fac + 0.5 * valtmp(8) * fac;
      val[2] = valtmp(2) + (valtmp(5) + valtmp(6)) * fac + 0.5 * valtmp(8) * fac;
      val[3] = valtmp(3) + (valtmp(6) + valtmp(7)) * fac + 0.5 * valtmp(8) * fac;
      val[4] = valtmp(4) * (1.0 - 2.0 * fac);
      val[5] = valtmp(5) * (1.0 - 2.0 * fac);
      val[6] = valtmp(6) * (1.0 - 2.0 * fac);
      val[7] = valtmp(7) * (1.0 - 2.0 * fac);
      val[8] = valtmp(8) * (1.0 - 4.0 * 0.5 * fac);

      deriv(0, 0) =
          derivtmp(0, 0) + (derivtmp(4, 0) + derivtmp(7, 0)) * fac + 0.5 * derivtmp(8, 0) * fac;
      deriv(0, 1) =
          derivtmp(0, 1) + (derivtmp(4, 1) + derivtmp(7, 1)) * fac + 0.5 * derivtmp(8, 1) * fac;
      deriv(1, 0) =
          derivtmp(1, 0) + (derivtmp(4, 0) + derivtmp(5, 0)) * fac + 0.5 * derivtmp(8, 0) * fac;
      deriv(1, 1) =
          derivtmp(1, 1) + (derivtmp(4, 1) + derivtmp(5, 1)) * fac + 0.5 * derivtmp(8, 1) * fac;
      deriv(2, 0) =
          derivtmp(2, 0) + (derivtmp(5, 0) + derivtmp(6, 0)) * fac + 0.5 * derivtmp(8, 0) * fac;
      deriv(2, 1) =
          derivtmp(2, 1) + (derivtmp(5, 1) + derivtmp(6, 1)) * fac + 0.5 * derivtmp(8, 1) * fac;
      deriv(3, 0) =
          derivtmp(3, 0) + (derivtmp(6, 0) + derivtmp(7, 0)) * fac + 0.5 * derivtmp(8, 0) * fac;
      deriv(3, 1) =
          derivtmp(3, 1) + (derivtmp(6, 1) + derivtmp(7, 1)) * fac + 0.5 * derivtmp(8, 1) * fac;
      deriv(4, 0) = derivtmp(4, 0) * (1.0 - 2.0 * fac);
      deriv(4, 1) = derivtmp(4, 1) * (1.0 - 2.0 * fac);
      deriv(5, 0) = derivtmp(5, 0) * (1.0 - 2.0 * fac);
      deriv(5, 1) = derivtmp(5, 1) * (1.0 - 2.0 * fac);
      deriv(6, 0) = derivtmp(6, 0) * (1.0 - 2.0 * fac);
      deriv(6, 1) = derivtmp(6, 1) * (1.0 - 2.0 * fac);
      deriv(7, 0) = derivtmp(7, 0) * (1.0 - 2.0 * fac);
      deriv(7, 1) = derivtmp(7, 1) * (1.0 - 2.0 * fac);
      deriv(8, 0) = derivtmp(8, 0) * (1.0 - 4.0 * 0.5 * fac);
      deriv(8, 1) = derivtmp(8, 1) * (1.0 - 4.0 * 0.5 * fac);

      break;
    }
      // *********************************************************************
      // 2D standard biquadratic shape functions (quad9)
      // (used in combination with linear dual LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::biquad2D_hierarchical:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;
      const double r2 = 1.0 - r * r;
      const double s2 = 1.0 - s * s;
      const double rh = 0.5 * r;
      const double sh = 0.5 * s;
      const double rhp = r + 0.5;
      const double rhm = r - 0.5;
      const double shp = s + 0.5;
      const double shm = s - 0.5;

      val[0] = 0.25 * rm * sm;
      val[1] = 0.25 * rp * sm;
      val[2] = 0.25 * rp * sp;
      val[3] = 0.25 * rm * sp;
      val[4] = -sh * sm * r2;
      val[5] = rh * rp * s2;
      val[6] = sh * sp * r2;
      val[7] = -rh * rm * s2;
      val[8] = r2 * s2;

      deriv(0, 0) = -0.25 * sm;
      deriv(0, 1) = -0.25 * rm;
      deriv(1, 0) = 0.25 * sm;
      deriv(1, 1) = -0.25 * rp;
      deriv(2, 0) = 0.25 * sp;
      deriv(2, 1) = 0.25 * rp;
      deriv(3, 0) = -0.25 * sp;
      deriv(3, 1) = 0.25 * rm;
      deriv(4, 0) = 2.0 * r * sh * sm;
      deriv(4, 1) = shm * r2;
      deriv(5, 0) = rhp * s2;
      deriv(5, 1) = -2.0 * s * rh * rp;
      deriv(6, 0) = -2.0 * r * sh * sp;
      deriv(6, 1) = shp * r2;
      deriv(7, 0) = rhm * s2;
      deriv(7, 1) = 2.0 * s * rh * rm;
      deriv(8, 0) = -2.0 * r * s2;
      deriv(8, 1) = -2.0 * s * r2;

      break;
    }
      // *********************************************************************
      // 2D bilinear part of biquadratic quadratic shape functions (quad9)
      // (used for linear interpolation of std LM field in 3D quadratic mortar)
      // *********************************************************************
    case MORTAR::Element::biquad2D_only_lin:
    {
      val[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
      val[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
      val[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
      val[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);
      val[4] = 0.0;
      val[5] = 0.0;
      val[6] = 0.0;
      val[7] = 0.0;
      val[8] = 0.0;

      deriv(0, 0) = -0.25 * (1.0 - xi[1]);
      deriv(0, 1) = -0.25 * (1.0 - xi[0]);
      deriv(1, 0) = 0.25 * (1.0 - xi[1]);
      deriv(1, 1) = -0.25 * (1.0 + xi[0]);
      deriv(2, 0) = 0.25 * (1.0 + xi[1]);
      deriv(2, 1) = 0.25 * (1.0 + xi[0]);
      deriv(3, 0) = -0.25 * (1.0 + xi[1]);
      deriv(3, 1) = 0.25 * (1.0 - xi[0]);
      deriv(4, 0) = 0.0;
      deriv(4, 1) = 0.0;
      deriv(5, 0) = 0.0;
      deriv(5, 1) = 0.0;
      deriv(6, 0) = 0.0;
      deriv(6, 1) = 0.0;
      deriv(7, 0) = 0.0;
      deriv(7, 1) = 0.0;
      deriv(8, 0) = 0.0;
      deriv(8, 1) = 0.0;

      break;
    }
      // *********************************************************************
      // 1D dual linear shape functions (line2)
      // (used for interpolation of Lagrange mutliplier field)
      // *********************************************************************
    case MORTAR::Element::lindual1D:
    {
      int dim = 1;

      // use element-based dual shape functions if no coefficient matrix is stored
      if (MoData().DualShape() == Teuchos::null)
      {
        val[0] = 0.5 * (1.0 - 3.0 * xi[0]);
        val[1] = 0.5 * (1.0 + 3.0 * xi[0]);
        deriv(0, 0) = -1.5;
        deriv(1, 0) = 1.5;
      }

      // pre-calculated consistent dual shape functions
      else
      {
#ifdef BACI_DEBUG
        if (MoData().DualShape()->numCols() != 2 && MoData().DualShape()->numRows() != 2)
          dserror("Dual shape functions coefficient matrix calculated in the wrong size");
#endif
        const int nnodes = NumNode();
        CORE::LINALG::SerialDenseVector stdval(nnodes, true);
        CORE::LINALG::SerialDenseMatrix stdderiv(nnodes, dim, true);
        CORE::LINALG::SerialDenseVector checkval(nnodes, true);
        EvaluateShape(xi, stdval, stdderiv, nnodes);
        CORE::LINALG::SerialDenseMatrix& ae = *(MoData().DualShape());

        for (int i = 0; i < NumNode(); ++i)
        {
          val[i] = 0.0;
          deriv(i, 0) = 0.0;
          for (int j = 0; j < NumNode(); ++j)
          {
            val[i] += stdval[j] * ae(i, j);
            deriv(i, 0) += ae(i, j) * stdderiv(j, 0);
          }
        }
      }
      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (const replacing linear, line2)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // *********************************************************************
    case MORTAR::Element::lindual1D_edge0:
    {
      dserror(
          "ERROR: explicit edge modification is outdated! We apply a genreal transformaiton "
          "instead");
      val[0] = 0.0;
      val[1] = 1.0;
      deriv(0, 0) = 0.0;
      deriv(1, 0) = 0.0;
      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (const replacing linear, line2)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // *********************************************************************
    case MORTAR::Element::lindual1D_edge1:
    {
      dserror(
          "ERROR: explicit edge modification is outdated! We apply a genreal transformaiton "
          "instead");
      val[0] = 1.0;
      val[1] = 0.0;
      deriv(0, 0) = 0.0;
      deriv(1, 0) = 0.0;
      break;
    }
      // *********************************************************************
      // 2D dual linear shape functions (tri3)
      // (used for interpolation of Lagrange mutliplier field)
      // *********************************************************************
    case MORTAR::Element::lindual2D:
    {
      if (MoData().DualShape() == Teuchos::null)
      {
        val[0] = 3.0 - 4.0 * xi[0] - 4.0 * xi[1];
        val[1] = 4.0 * xi[0] - 1.0;
        val[2] = 4.0 * xi[1] - 1.0;
        deriv(0, 0) = -4.0;
        deriv(0, 1) = -4.0;
        deriv(1, 0) = 4.0;
        deriv(1, 1) = 0.0;
        deriv(2, 0) = 0.0;
        deriv(2, 1) = 4.0;
      }
      else
      {
        const int nnodes = NumNode();
        // get solution matrix with dual parameters
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);
        // get dual shape functions coefficient matrix from data container
        ae = *(MoData().DualShape());

        // evaluate dual shape functions at loc. coord. xi
        // need standard shape functions at xi first
        EvaluateShape(xi, val, deriv, nnodes);

        // dimension
        int dim = 2;

        // evaluate dual shape functions
        CORE::LINALG::SerialDenseVector valtemp(nnodes, true);
        CORE::LINALG::SerialDenseMatrix derivtemp(nnodes, dim, true);
        for (int i = 0; i < nnodes; ++i)
          for (int j = 0; j < nnodes; ++j)
          {
            valtemp[i] += ae(i, j) * val[j];
            derivtemp(i, 0) += ae(i, j) * deriv(j, 0);
            derivtemp(i, 1) += ae(i, j) * deriv(j, 1);
          }

        val = valtemp;
        deriv = derivtemp;
      }
      break;
    }
      // *********************************************************************
      // 2D dual bilinear shape functions (quad4)
      // (used for interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case MORTAR::Element::bilindual2D:
    {
      const int nnodes = 4;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif
      // get solution matrix with dual parameters
      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // no pre-computed dual shape functions
      if (MoData().DualShape() == Teuchos::null)
      {
        // establish fundamental data
        double detg = 0.0;

        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());

        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * val[j] * val[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * val[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        CORE::LINALG::InvertAndMultiplyByCholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }

      // pre-computed dual shape functions
      else
      {
        // get dual shape functions coefficient matrix from data container
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      // need standard shape functions at xi first
      EvaluateShape(xi, val, deriv, nnodes);

      // dimension
      const int dim = 2;

      // evaluate dual shape functions
      CORE::LINALG::SerialDenseVector valtemp(nnodes, true);
      CORE::LINALG::SerialDenseMatrix derivtemp(nnodes, dim, true);
      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          valtemp[i] += ae(i, j) * val[j];
          derivtemp(i, 0) += ae(i, j) * deriv(j, 0);
          derivtemp(i, 1) += ae(i, j) * deriv(j, 1);
        }
      }

      val = valtemp;
      deriv = derivtemp;
      break;
    }
      // *********************************************************************
      // 1D dual quadratic shape functions (line3)
      // (used for interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case MORTAR::Element::quaddual1D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 3;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());

        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * val[j] * val[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * val[j] * detg;
            }
        }

        // calcute coefficient matrix
        CORE::LINALG::InvertAndMultiplyByCholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      // need standard shape functions at xi first
      EvaluateShape(xi, val, deriv, nnodes);

      // check whether this is a 1D or 2D mortar element
      int dim = 1;

      // evaluate dual shape functions
      CORE::LINALG::SerialDenseVector valtemp(nnodes, true);
      CORE::LINALG::SerialDenseMatrix derivtemp(nnodes, dim, true);
      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          valtemp[i] += ae(i, j) * val[j];
          derivtemp(i, 0) += ae(i, j) * deriv(j, 0);
          if (dim == 2) derivtemp(i, 1) += ae(i, j) * deriv(j, 1);
        }
      }

      val = valtemp;
      deriv = derivtemp;

      break;
    }
      // *********************************************************************
      // 1D linear part of dual quadratic shape functions (line3)
      // (used for LINEAR interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case MORTAR::Element::quaddual1D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 3;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 2);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          ShapeFunctions(MORTAR::Element::quad1D_only_lin, gpc, valquad, derivquad);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 2;

        // reduce me to non-zero nodes before inverting
        CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin;
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        CORE::LINALG::Inverse(melin);

        // re-inflate inverse of melin to full size
        CORE::LINALG::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        CORE::LINALG::multiply(ae, de, invme);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      ShapeFunctions(MORTAR::Element::quad1D_only_lin, xi, valquad, derivquad);
      val.putScalar(0.0);
      deriv.putScalar(0.0);

      // check whether this is a 1D or 2D mortar element
      int dim = 1;

      // evaluate dual shape functions
      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          val[i] += ae(i, j) * valquad[j];
          deriv(i, 0) += ae(i, j) * derivquad(j, 0);
          if (dim == 2) deriv(i, 1) += ae(i, j) * derivquad(j, 1);
        }
      }

      break;
    }
      // *********************************************************************
      // 2D dual quadratic shape functions (tri6)
      // (used for interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::quaddual2D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 6;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 2);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, valquad, derivquad, nnodes, true);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * valquad[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        CORE::LINALG::InvertAndMultiplyByCholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      EvaluateShape(xi, valquad, derivquad, nnodes, true);
      val.putScalar(0.0);
      deriv.putScalar(0.0);

      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          val[i] += ae(i, j) * valquad[j];
          deriv(i, 0) += ae(i, j) * derivquad(j, 0);
          deriv(i, 1) += ae(i, j) * derivquad(j, 1);
        }
      }

      break;
    }

    // *********************************************************************
    // 2D dual serendipity shape functions (quad8)
    // (used for interpolation of Lagrange mutliplier field)
    // (including adaption process for distorted elements)
    // (including modification of displacement shape functions)
    // *********************************************************************
    case MORTAR::Element::serendipitydual2D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 8;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 2);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, valquad, derivquad, nnodes, true);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * valquad[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        CORE::LINALG::InvertAndMultiplyByCholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      EvaluateShape(xi, valquad, derivquad, nnodes, true);
      val.putScalar(0.0);
      deriv.putScalar(0.0);

      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          val[i] += ae(i, j) * valquad[j];
          deriv(i, 0) += ae(i, j) * derivquad(j, 0);
          deriv(i, 1) += ae(i, j) * derivquad(j, 1);
        }
      }

      break;
    }
      // *********************************************************************
      // 2D dual biquadratic shape functions (quad9)
      // (used for interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::biquaddual2D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 9;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 2);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, valquad, derivquad, nnodes, true);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * valquad[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        CORE::LINALG::InvertAndMultiplyByCholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      EvaluateShape(xi, valquad, derivquad, nnodes, true);
      val.putScalar(0.0);
      deriv.putScalar(0.0);

      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          val[i] += ae(i, j) * valquad[j];
          deriv(i, 0) += ae(i, j) * derivquad(j, 0);
          deriv(i, 1) += ae(i, j) * derivquad(j, 1);
        }
      }

      break;
    }
      // *********************************************************************
      // 2D dual quadratic shape functions (tri6)
      // (used for LINEAR interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::quaddual2D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 6;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 2);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          ShapeFunctions(MORTAR::Element::quad2D_only_lin, gpc, valquad, derivquad);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 3;

        // reduce me to non-zero nodes before inverting
        CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin;
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        CORE::LINALG::Inverse(melin);

        // re-inflate inverse of melin to full size
        CORE::LINALG::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        CORE::LINALG::multiply(ae, de, invme);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      ShapeFunctions(MORTAR::Element::quad2D_only_lin, xi, valquad, derivquad);
      val.putScalar(0.0);
      deriv.putScalar(0.0);

      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          val[i] += ae(i, j) * valquad[j];
          deriv(i, 0) += ae(i, j) * derivquad(j, 0);
          deriv(i, 1) += ae(i, j) * derivquad(j, 1);
        }
      }

      break;
    }
      // *********************************************************************
      // 2D dual serendipity shape functions (quad8)
      // (used for LINEAR interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::serendipitydual2D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 8;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 2);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          ShapeFunctions(MORTAR::Element::serendipity2D_only_lin, gpc, valquad, derivquad);

          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 4;

        // reduce me to non-zero nodes before inverting
        CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin(true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        CORE::LINALG::Inverse(melin);

        // re-inflate inverse of melin to full size
        CORE::LINALG::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        CORE::LINALG::multiply(ae, de, invme);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      ShapeFunctions(MORTAR::Element::serendipity2D_only_lin, xi, valquad, derivquad);
      val.putScalar(0.0);
      deriv.putScalar(0.0);

      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          val[i] += ae(i, j) * valquad[j];
          deriv(i, 0) += ae(i, j) * derivquad(j, 0);
          deriv(i, 1) += ae(i, j) * derivquad(j, 1);
        }
      }

      break;
    }
      // *********************************************************************
      // 2D dual biquadratic shape functions (quad9)
      // (used for LINEAR interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::biquaddual2D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 9;

#ifdef BACI_DEBUG
      if (nnodes != NumNode())
        dserror("MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

      CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 2);

      if (MoData().DualShape() == Teuchos::null)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          ShapeFunctions(MORTAR::Element::biquad2D_only_lin, gpc, valquad, derivquad);
          detg = Jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.Weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.Weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 4;

        // reduce me to non-zero nodes before inverting
        CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin(true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        CORE::LINALG::Inverse(melin);

        // re-inflate inverse of melin to full size
        CORE::LINALG::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        CORE::LINALG::multiply(ae, de, invme);

        // store coefficient matrix
        MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
      }
      else
      {
        ae = *(MoData().DualShape());
      }

      // evaluate dual shape functions at loc. coord. xi
      ShapeFunctions(MORTAR::Element::biquad2D_only_lin, xi, valquad, derivquad);
      val.putScalar(0.0);
      deriv.putScalar(0.0);

      for (int i = 0; i < nnodes; ++i)
      {
        for (int j = 0; j < nnodes; ++j)
        {
          val[i] += ae(i, j) * valquad[j];
          deriv(i, 0) += ae(i, j) * derivquad(j, 0);
          deriv(i, 1) += ae(i, j) * derivquad(j, 1);
        }
      }

      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (linear replacing quad, line3)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // (only form a basis and have to be adapted for distorted elements)
      // *********************************************************************
    case MORTAR::Element::dual1D_base_for_edge0:
    {
      val[0] = xi[0];
      val[1] = 1.0 - xi[0];
      deriv(0, 0) = 1.0;
      deriv(1, 0) = -1.0;
      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (linear replacing quad, line3)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // (only form a basis and have to be adapted for distorted elements)
      // *********************************************************************
    case MORTAR::Element::dual1D_base_for_edge1:
    {
      val[0] = -xi[0];
      val[1] = 1.0 + xi[0];
      deriv(0, 0) = -1.0;
      deriv(1, 0) = 1.0;
      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (linear replacing quad, line3)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case MORTAR::Element::quaddual1D_edge0:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = NumNode();

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 1);
      CORE::LINALG::SerialDenseVector vallin(nnodes - 1);
      CORE::LINALG::SerialDenseMatrix derivlin(nnodes - 1, 1);
      CORE::LINALG::SerialDenseVector valtemp(nnodes, true);
      CORE::LINALG::SerialDenseMatrix derivtemp(nnodes, 1, true);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      MORTAR::ElementIntegrator integrator(Shape());

      CORE::LINALG::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      CORE::LINALG::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      for (int i = 0; i < integrator.nGP(); ++i)
      {
        double gpc[2] = {integrator.Coordinate(i, 0), 0.0};
        ShapeFunctions(MORTAR::Element::quad1D, gpc, valquad, derivquad);
        ShapeFunctions(MORTAR::Element::dual1D_base_for_edge0, gpc, vallin, derivlin);
        detg = Jacobian(gpc);

        for (int j = 1; j < nnodes; ++j)
          for (int k = 1; k < nnodes; ++k)
          {
            me(j - 1, k - 1) += integrator.Weight(i) * vallin[j - 1] * valquad[k] * detg;
            de(j - 1, k - 1) += (j == k) * integrator.Weight(i) * valquad[k] * detg;
          }
      }

      // invert bi-ortho matrix me
      // CAUTION: This is a non-symmetric inverse operation!
      const double detmeinv = 1.0 / (me(0, 0) * me(1, 1) - me(0, 1) * me(1, 0));
      CORE::LINALG::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      CORE::LINALG::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      CORE::LINALG::multiply(ae, de, me);

      // evaluate dual shape functions at loc. coord. xi
      ShapeFunctions(MORTAR::Element::dual1D_base_for_edge0, xi, vallin, derivlin);
      for (int i = 1; i < nnodes; ++i)
        for (int j = 1; j < nnodes; ++j)
        {
          valtemp[i] += ae(i - 1, j - 1) * vallin[j - 1];
          derivtemp(i, 0) += ae(i - 1, j - 1) * derivlin(j - 1, 0);
        }

      val[0] = 0.0;
      val[1] = valtemp[1];
      val[2] = valtemp[2];
      deriv(0, 0) = 0.0;
      deriv(1, 0) = derivtemp(1, 0);
      deriv(2, 0) = derivtemp(2, 0);

      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (linear replacing quad, line3)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case MORTAR::Element::quaddual1D_edge1:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = NumNode();

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 1);
      CORE::LINALG::SerialDenseVector vallin(nnodes - 1);
      CORE::LINALG::SerialDenseMatrix derivlin(nnodes - 1, 1);
      CORE::LINALG::SerialDenseVector valtemp(nnodes, true);
      CORE::LINALG::SerialDenseMatrix derivtemp(nnodes, 1, true);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      MORTAR::ElementIntegrator integrator(Shape());

      CORE::LINALG::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      CORE::LINALG::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      for (int i = 0; i < integrator.nGP(); ++i)
      {
        double gpc[2] = {integrator.Coordinate(i, 0), 0.0};
        ShapeFunctions(MORTAR::Element::quad1D, gpc, valquad, derivquad);
        ShapeFunctions(MORTAR::Element::dual1D_base_for_edge1, gpc, vallin, derivlin);
        detg = Jacobian(gpc);

        for (int j = 0; j < nnodes - 1; ++j)
          for (int k = 0; k < nnodes - 1; ++k)
          {
            me(j, k) += integrator.Weight(i) * vallin[j] * valquad[2 * k] * detg;
            de(j, k) += (j == k) * integrator.Weight(i) * valquad[2 * k] * detg;
          }
      }

      // invert bi-ortho matrix me
      // CAUTION: This is a non-symmetric inverse operation!
      double detmeinv = 1.0 / (me(0, 0) * me(1, 1) - me(0, 1) * me(1, 0));
      CORE::LINALG::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      CORE::LINALG::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      CORE::LINALG::multiply(ae, de, me);

      // evaluate dual shape functions at loc. coord. xi
      ShapeFunctions(MORTAR::Element::dual1D_base_for_edge1, xi, vallin, derivlin);
      for (int i = 0; i < nnodes - 1; ++i)
        for (int j = 0; j < nnodes - 1; ++j)
        {
          valtemp[2 * i] += ae(i, j) * vallin[j];
          derivtemp(2 * i, 0) += ae(i, j) * derivlin(j, 0);
        }

      val[0] = valtemp[0];
      val[1] = 0.0;
      val[2] = valtemp[2];
      deriv(0, 0) = derivtemp(0, 0);
      deriv(1, 0) = 0.0;
      deriv(2, 0) = derivtemp(2, 0);

      break;
    }
      // *********************************************************************
      // Unkown shape function type
      // *********************************************************************
    default:
    {
      dserror("Unknown shape function type identifier");
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate displacement shape functions                     popp 01/08|
 *----------------------------------------------------------------------*/
bool MORTAR::Element::EvaluateShape(const double* xi, CORE::LINALG::SerialDenseVector& val,
    CORE::LINALG::SerialDenseMatrix& deriv, const int valdim, bool dualquad)
{
  if (!xi) dserror("EvaluateShape called with xi=nullptr");

  // get node number and node pointers
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("EvaluateShapeLagMult: Null pointer!");

  // check for boundary nodes
  bool bound = false;
  for (int i = 0; i < NumNode(); ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
    if (!mymrtrnode) dserror("EvaluateShapeLagMult: Null pointer!");
    bound += mymrtrnode->IsOnBound();
  }

  switch (Shape())
  {
    // 2D linear case (2noded line element)
    case CORE::FE::CellType::line2:
    {
      if (valdim != 2) dserror("Inconsistency in EvaluateShape");
      ShapeFunctions(MORTAR::Element::lin1D, xi, val, deriv);
      break;
    }
      // 2D quadratic case (3noded line element)
    case CORE::FE::CellType::line3:
    {
      if (valdim != 3) dserror("Inconsistency in EvaluateShape");

      if (dualquad && !bound)
        dserror(
            "There is no quadratic interpolation for dual shape functions for 2-D problems with "
            "quadratic elements available!");
      else if (dualquad && bound)
        ShapeFunctions(MORTAR::Element::quad1D_hierarchical, xi, val, deriv);
      else
        ShapeFunctions(MORTAR::Element::quad1D, xi, val, deriv);
      break;
    }
      // 3D linear case (3noded triangular element)
    case CORE::FE::CellType::tri3:
    {
      if (valdim != 3) dserror("Inconsistency in EvaluateShape");
      ShapeFunctions(MORTAR::Element::lin2D, xi, val, deriv);
      break;
    }
      // 3D bilinear case (4noded quadrilateral element)
    case CORE::FE::CellType::quad4:
    {
      if (valdim != 4) dserror("Inconsistency in EvaluateShape");
      ShapeFunctions(MORTAR::Element::bilin2D, xi, val, deriv);
      break;
    }
      // 3D quadratic case (6noded triangular element)
    case CORE::FE::CellType::tri6:
    {
      if (valdim != 6) dserror("Inconsistency in EvaluateShape");
      if (dualquad && !bound)
        ShapeFunctions(MORTAR::Element::quad2D_modified, xi, val, deriv);
      else if (dualquad && bound)
        ShapeFunctions(MORTAR::Element::quad2D_hierarchical, xi, val, deriv);
      else
        ShapeFunctions(MORTAR::Element::quad2D, xi, val, deriv);
      break;
    }
      // 3D serendipity case (8noded quadrilateral element)
    case CORE::FE::CellType::quad8:
    {
      if (valdim != 8) dserror("Inconsistency in EvaluateShape");
      if (dualquad && !bound)
        ShapeFunctions(MORTAR::Element::serendipity2D_modified, xi, val, deriv);
      else if (dualquad && bound)
        ShapeFunctions(MORTAR::Element::serendipity2D_hierarchical, xi, val, deriv);
      else
        ShapeFunctions(MORTAR::Element::serendipity2D, xi, val, deriv);
      break;
    }
      // 3D biquadratic case (9noded quadrilateral element)
    case CORE::FE::CellType::quad9:
    {
      if (valdim != 9) dserror("Inconsistency in EvaluateShape");
      if (dualquad && !bound)
        ShapeFunctions(MORTAR::Element::biquad2D_modified, xi, val, deriv);
      else if (dualquad && bound)
        ShapeFunctions(MORTAR::Element::biquad2D_hierarchical, xi, val, deriv);
      else
        ShapeFunctions(MORTAR::Element::biquad2D, xi, val, deriv);
      break;
    }

      //==================================================
      //                     NURBS
      //==================================================

      // 1D -- nurbs2
    case CORE::FE::CellType::nurbs2:
    {
      if (valdim != 2) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseMatrix auxderiv(1, NumNode());
      CORE::FE::NURBS::nurbs_get_1D_funct_deriv(
          val, auxderiv, xi[0], Knots()[0], weights, CORE::FE::CellType::nurbs2);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < NumNode(); ++i) deriv(i, 0) = auxderiv(0, i);

      break;
    }

      // 1D -- nurbs3
    case CORE::FE::CellType::nurbs3:
    {
      if (valdim != 3) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseMatrix auxderiv(1, NumNode());
      CORE::FE::NURBS::nurbs_get_1D_funct_deriv(
          val, auxderiv, xi[0], Knots()[0], weights, CORE::FE::CellType::nurbs3);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < NumNode(); ++i) deriv(i, 0) = auxderiv(0, i);

      break;
    }

      // ===========================================================
      // 2D -- nurbs4
    case CORE::FE::CellType::nurbs4:
    {
      if (valdim != 4) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      CORE::LINALG::SerialDenseMatrix auxderiv(2, NumNode());
      CORE::FE::NURBS::nurbs_get_2D_funct_deriv(
          val, auxderiv, uv, Knots(), weights, CORE::FE::CellType::nurbs4);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 2; ++d)
        for (int i = 0; i < NumNode(); ++i) deriv(i, d) = auxderiv(d, i);

      break;
    }

      // 2D -- nurbs9
    case CORE::FE::CellType::nurbs9:
    {
      if (valdim != 9) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];


      CORE::LINALG::SerialDenseMatrix auxderiv(2, NumNode());
      CORE::FE::NURBS::nurbs_get_2D_funct_deriv(
          val, auxderiv, uv, Knots(), weights, CORE::FE::CellType::nurbs9);

#ifdef BACI_DEBUG
      if (deriv.numCols() != 2 || deriv.numRows() != NumNode())
        dserror("Inconsistency in EvaluateShape");
#endif

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 2; ++d)
        for (int i = 0; i < NumNode(); ++i) deriv(i, d) = auxderiv(d, i);


      break;
    }
      // unknown case
    default:
    {
      dserror("EvaluateShape called for unknown MORTAR::Element type");
      break;
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate Lagrange multiplier shape functions              popp 12/07|
 *----------------------------------------------------------------------*/
bool MORTAR::Element::EvaluateShapeLagMult(const INPAR::MORTAR::ShapeFcn& lmtype, const double* xi,
    CORE::LINALG::SerialDenseVector& val, CORE::LINALG::SerialDenseMatrix& deriv, const int valdim,
    bool boundtrafo)
{
  // some methods don't need a Lagrange multiplier interpolation
  if (lmtype == INPAR::MORTAR::shape_none) return true;

  if (!xi) dserror("EvaluateShapeLagMult called with xi=nullptr");

  // dual LM shape functions or not
  bool dual = false;
  if (lmtype == INPAR::MORTAR::shape_dual or lmtype == INPAR::MORTAR::shape_petrovgalerkin)
    dual = true;

  // get node number and node pointers
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("EvaluateShapeLagMult: Null pointer!");

  switch (Shape())
  {
    // 2D linear case (2noded line element)
    case CORE::FE::CellType::line2:
    {
      if (valdim != 2) dserror("Inconsistency in EvaluateShape");

      if (dual)
        ShapeFunctions(MORTAR::Element::lindual1D, xi, val, deriv);
      else
        ShapeFunctions(MORTAR::Element::lin1D, xi, val, deriv);
      break;
    }

      // 2D quadratic case (3noded line element)
    case CORE::FE::CellType::line3:
    {
      if (valdim != 3) dserror("Inconsistency in EvaluateShape");

      if (dual)
        ShapeFunctions(MORTAR::Element::quaddual1D, xi, val, deriv);
      else
        ShapeFunctions(MORTAR::Element::quad1D, xi, val, deriv);

      break;
    }

      // 3D cases
    case CORE::FE::CellType::tri3:
    case CORE::FE::CellType::quad4:
    case CORE::FE::CellType::tri6:
    case CORE::FE::CellType::quad8:
    case CORE::FE::CellType::quad9:
    {
      // dual Lagrange multipliers
      if (dual)
      {
        if (Shape() == CORE::FE::CellType::tri3)
          ShapeFunctions(MORTAR::Element::lindual2D, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::quad4)
          ShapeFunctions(MORTAR::Element::bilindual2D, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::tri6)
          ShapeFunctions(MORTAR::Element::quaddual2D, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::quad8)
          ShapeFunctions(MORTAR::Element::serendipitydual2D, xi, val, deriv);
        else
          /*Shape()==quad9*/ ShapeFunctions(MORTAR::Element::biquaddual2D, xi, val, deriv);
      }

      // standard Lagrange multipliers
      else
      {
        if (Shape() == CORE::FE::CellType::tri3)
          ShapeFunctions(MORTAR::Element::lin2D, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::quad4)
          ShapeFunctions(MORTAR::Element::bilin2D, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::tri6)
          ShapeFunctions(MORTAR::Element::quad2D, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::quad8)
          ShapeFunctions(MORTAR::Element::serendipity2D, xi, val, deriv);
        else
          /*Shape()==quad9*/ ShapeFunctions(MORTAR::Element::biquad2D, xi, val, deriv);
      }

      break;
    }
      //==================================================
      //                     NURBS
      //==================================================

      // 1D -- nurbs2
    case CORE::FE::CellType::nurbs2:
    {
      if (dual)
        dserror("no dual shape functions provided for nurbs!");
      else
        EvaluateShape(xi, val, deriv, valdim);

      break;
    }

      // 1D -- nurbs3
    case CORE::FE::CellType::nurbs3:
    {
      if (dual)
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 3;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);
        if (MoData().DualShape() == Teuchos::null)
        {
          // compute entries to bi-ortho matrices me/de with Gauss quadrature
          MORTAR::ElementIntegrator integrator(Shape());

          CORE::LINALG::Matrix<nnodes, nnodes> me(true);
          CORE::LINALG::Matrix<nnodes, nnodes> de(true);

          for (int i = 0; i < integrator.nGP(); ++i)
          {
            double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
            EvaluateShape(gpc, val, deriv, nnodes);
            detg = Jacobian(gpc);

            for (int j = 0; j < nnodes; ++j)
            {
              for (int k = 0; k < nnodes; ++k)
              {
                me(j, k) += integrator.Weight(i) * val[j] * val[k] * detg;
                de(j, k) += (j == k) * integrator.Weight(i) * val[j] * detg;
              }
            }
          }

          // get solution matrix with dual parameters
          CORE::LINALG::InvertAndMultiplyByCholesky<nnodes>(me, de, ae);

          // store coefficient matrix
          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        else
          ae = *(MoData().DualShape());

        // evaluate dual shape functions at loc. coord. xi
        // need standard shape functions at xi first
        EvaluateShape(xi, val, deriv, nnodes);

        // check whether this is a 1D or 2D mortar element
        const int dim = 1;
        // evaluate dual shape functions
        CORE::LINALG::SerialDenseVector valtemp(nnodes, true);
        CORE::LINALG::SerialDenseMatrix derivtemp(nnodes, dim, true);
        for (int i = 0; i < nnodes; ++i)
        {
          for (int j = 0; j < nnodes; ++j)
          {
            valtemp[i] += ae(i, j) * val[j];
            derivtemp(i, 0) += ae(i, j) * deriv(j, 0);
          }
        }

        val = valtemp;
        deriv = derivtemp;
      }
      else
        EvaluateShape(xi, val, deriv, valdim);

      break;
    }

      // ===========================================================
      // 2D -- nurbs4
    case CORE::FE::CellType::nurbs4:
    {
      if (dual)
        dserror("no dual shape functions provided for nurbs!");
      else
        EvaluateShape(xi, val, deriv, valdim);

      break;
    }

      // 2D -- nurbs8
    case CORE::FE::CellType::nurbs8:
    {
      if (dual)
        dserror("no dual shape functions provided for nurbs!");
      else
        EvaluateShape(xi, val, deriv, valdim);

      break;
    }

      // 2D -- nurbs9
    case CORE::FE::CellType::nurbs9:
    {
      if (dual)
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 9;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes);

        if (MoData().DualShape() == Teuchos::null)
        {
          // compute entries to bi-ortho matrices me/de with Gauss quadrature
          MORTAR::ElementIntegrator integrator(Shape());

          CORE::LINALG::Matrix<nnodes, nnodes> me(true);
          CORE::LINALG::Matrix<nnodes, nnodes> de(true);

          for (int i = 0; i < integrator.nGP(); ++i)
          {
            double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
            EvaluateShape(gpc, val, deriv, nnodes);
            detg = Jacobian(gpc);

            for (int j = 0; j < nnodes; ++j)
            {
              for (int k = 0; k < nnodes; ++k)
              {
                me(j, k) += integrator.Weight(i) * val[j] * val[k] * detg;
                de(j, k) += (j == k) * integrator.Weight(i) * val[j] * detg;
              }
            }
          }

          // get solution matrix with dual parameters
          CORE::LINALG::InvertAndMultiplyByCholesky<nnodes>(me, de, ae);

          // store coefficient matrix
          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        else
          ae = *(MoData().DualShape());

        // evaluate dual shape functions at loc. coord. xi
        // need standard shape functions at xi first
        EvaluateShape(xi, val, deriv, nnodes);

        // check whether this is a 1D or 2D mortar element
        const int dim = 2;
        // evaluate dual shape functions
        CORE::LINALG::SerialDenseVector valtemp(nnodes, true);
        CORE::LINALG::SerialDenseMatrix derivtemp(nnodes, dim, true);
        for (int i = 0; i < nnodes; ++i)
        {
          for (int j = 0; j < nnodes; ++j)
          {
            valtemp[i] += ae(i, j) * val[j];
            derivtemp(i, 0) += ae(i, j) * deriv(j, 0);
            derivtemp(i, 1) += ae(i, j) * deriv(j, 1);
          }
        }

        val = valtemp;
        deriv = derivtemp;
      }
      else
        EvaluateShape(xi, val, deriv, valdim);

      break;
    }

      // unknown case
    default:
    {
      dserror("EvaluateShapeLagMult called for unknown element type");
      break;
    }
  }

  // if no trafo is required return!
  if (!boundtrafo) return true;

  // check if we need trafo
  const int nnodes = NumNode();
  bool bound = false;
  for (int i = 0; i < nnodes; ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);

    if (Shape() == CORE::FE::CellType::line2 or Shape() == CORE::FE::CellType::line3 or
        Shape() == CORE::FE::CellType::nurbs2 or Shape() == CORE::FE::CellType::nurbs3)
    {
      // is on corner or bound?
      if (mymrtrnode->IsOnCornerorBound())
      {
        bound = true;
        break;
      }
    }
    else
    {  // is on corner or edge or bound ?
      if (mymrtrnode->IsOnBoundorCE())
      {
        bound = true;
        break;
      }
    }
  }

  if (!bound) return true;

  //---------------------------------
  // do trafo for bound elements
  CORE::LINALG::SerialDenseMatrix trafo(nnodes, nnodes, true);

  if (MoData().Trafo() == Teuchos::null)
  {
    // 2D case!
    if (Shape() == CORE::FE::CellType::line2 or Shape() == CORE::FE::CellType::line3 or
        Shape() == CORE::FE::CellType::nurbs2 or Shape() == CORE::FE::CellType::nurbs3)
    {
      // get number of bound nodes
      std::vector<int> ids;
      for (int i = 0; i < nnodes; ++i)
      {
        Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
        if (mymrtrnode->IsOnCornerorBound())
        {
          // get local bound id
          ids.push_back(i);
        }
      }

      int numbound = (int)ids.size();

      // if all bound: error
      if ((nnodes - numbound) < 1e-12) dserror("all nodes are bound");

      const double factor = 1.0 / (nnodes - numbound);
      // row loop
      for (int i = 0; i < nnodes; ++i)
      {
        Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
        if (!mymrtrnode->IsOnCornerorBound())
        {
          trafo(i, i) = 1.0;
          for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
        }
      }
    }

    // 3D case!
    else if (Shape() == CORE::FE::CellType::tri6 or Shape() == CORE::FE::CellType::tri3 or
             Shape() == CORE::FE::CellType::quad4 or Shape() == CORE::FE::CellType::quad8 or
             Shape() == CORE::FE::CellType::quad9 or Shape() == CORE::FE::CellType::quad4 or
             Shape() == CORE::FE::CellType::nurbs9)
    {
      // get number of bound nodes
      std::vector<int> ids;
      for (int i = 0; i < nnodes; ++i)
      {
        Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
        if (mymrtrnode->IsOnBoundorCE())
        {
          // get local bound id
          ids.push_back(i);
        }
      }

      int numbound = (int)ids.size();

      // if all bound: error
      if ((nnodes - numbound) < 1e-12)
      {
        std::cout << "numnode= " << nnodes << "shape= " << CORE::FE::CellTypeToString(Shape())
                  << std::endl;
        dserror("all nodes are bound");
      }

      const double factor = 1.0 / (nnodes - numbound);
      // row loop
      for (int i = 0; i < nnodes; ++i)
      {
        Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
        if (!mymrtrnode->IsOnBoundorCE())
        {
          trafo(i, i) = 1.0;
          for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
        }
      }
    }
    else
      dserror("unknown element type!");

    MoData().Trafo() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(trafo));
  }
  else
  {
    trafo = *(MoData().Trafo());
  }

  int eledim = -1;
  if (Shape() == CORE::FE::CellType::tri6 or Shape() == CORE::FE::CellType::tri3 or
      Shape() == CORE::FE::CellType::quad4 or Shape() == CORE::FE::CellType::quad8 or
      Shape() == CORE::FE::CellType::quad9 or Shape() == CORE::FE::CellType::nurbs4 or
      Shape() == CORE::FE::CellType::nurbs9)
  {
    eledim = 2;
  }
  else if (Shape() == CORE::FE::CellType::line2 or Shape() == CORE::FE::CellType::line3 or
           Shape() == CORE::FE::CellType::nurbs2 or Shape() == CORE::FE::CellType::nurbs3)
  {
    eledim = 1;
  }
  else
  {
    dserror("unknown shape");
  }

  CORE::LINALG::SerialDenseVector tempval(nnodes, true);
  CORE::LINALG::SerialDenseMatrix tempderiv(nnodes, eledim, true);

  for (int i = 0; i < nnodes; ++i)
    for (int j = 0; j < nnodes; ++j) tempval(i) += trafo(i, j) * val(j);

  for (int k = 0; k < eledim; ++k)
    for (int i = 0; i < nnodes; ++i)
      for (int j = 0; j < nnodes; ++j) tempderiv(i, k) += trafo(i, j) * deriv(j, k);

  for (int i = 0; i < nnodes; ++i) val(i) = tempval(i);

  for (int k = 0; k < eledim; ++k)
    for (int i = 0; i < nnodes; ++i) deriv(i, k) = tempderiv(i, k);

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate Lagrange multiplier shape functions             seitz 09/17|
 |  THIS IS A SPECIAL VERSION FOR 3D QUADRATIC MORTAR WITH CONST LM!    |
 *----------------------------------------------------------------------*/
bool MORTAR::Element::EvaluateShapeLagMultConst(const INPAR::MORTAR::ShapeFcn& lmtype,
    const double* xi, CORE::LINALG::SerialDenseVector& val, CORE::LINALG::SerialDenseMatrix& deriv,
    const int valdim)
{
  MORTAR::UTILS::EvaluateShape_LM_Const(lmtype, xi, val, *this, valdim);
  deriv.putScalar(0.0);

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate Lagrange multiplier shape functions              popp 12/07|
 |  THIS IS A SPECIAL VERSION FOR 3D QUADRATIC MORTAR WITH LIN LM!      |
 *----------------------------------------------------------------------*/
bool MORTAR::Element::EvaluateShapeLagMultLin(const INPAR::MORTAR::ShapeFcn& lmtype,
    const double* xi, CORE::LINALG::SerialDenseVector& val, CORE::LINALG::SerialDenseMatrix& deriv,
    const int valdim)
{
  // some methods don't need a Lagrange multiplier interpolation
  if (lmtype == INPAR::MORTAR::shape_none) return true;

  if (!xi) dserror("EvaluateShapeLagMultLin called with xi=nullptr");
  if (!IsSlave()) dserror("EvaluateShapeLagMultLin called for master element");

  // check for feasible element types (line3,tri6, quad8 or quad9)
  if (Shape() != CORE::FE::CellType::line3 && Shape() != CORE::FE::CellType::tri6 &&
      Shape() != CORE::FE::CellType::quad8 && Shape() != CORE::FE::CellType::quad9)
    dserror("Linear LM interpolation only for quadratic finite elements");

  // dual shape functions or not
  bool dual = false;
  if (lmtype == INPAR::MORTAR::shape_dual || lmtype == INPAR::MORTAR::shape_petrovgalerkin)
    dual = true;

  // get node number and node pointers
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("EvaluateShapeLagMult: Null pointer!");

  // check for boundary nodes
  bool bound = false;
  for (int i = 0; i < NumNode(); ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
    if (!mymrtrnode) dserror("EvaluateShapeLagMult: Null pointer!");
    bound += mymrtrnode->IsOnBound();
  }

  // all nodes are interior: use unmodified shape functions
  if (!bound)
  {
    dserror("You should not be here...");
  }

  switch (Shape())
  {
    // 2D quadratic case (quadratic line)
    case CORE::FE::CellType::line3:
    {
      // the middle node is defined as slave boundary (=master)
      // dual Lagrange multipliers
      if (dual) ShapeFunctions(MORTAR::Element::quaddual1D_only_lin, xi, val, deriv);
      // standard Lagrange multipliers
      else
        ShapeFunctions(MORTAR::Element::quad1D_only_lin, xi, val, deriv);

      break;
    }

      // 3D quadratic cases (quadratic triangle, biquadratic and serendipity quad)
    case CORE::FE::CellType::tri6:
    case CORE::FE::CellType::quad8:
    case CORE::FE::CellType::quad9:
    {
      // the edge nodes are defined as slave boundary (=master)
      // dual Lagrange multipliers
      if (dual)
      {
        // dserror("Quad->Lin modification of dual LM shape functions not yet implemented");
        if (Shape() == CORE::FE::CellType::tri6)
          ShapeFunctions(MORTAR::Element::quaddual2D_only_lin, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::quad8)
          ShapeFunctions(MORTAR::Element::serendipitydual2D_only_lin, xi, val, deriv);
        else
          /*Shape()==quad9*/ ShapeFunctions(MORTAR::Element::biquaddual2D_only_lin, xi, val, deriv);
      }

      // standard Lagrange multipliers
      else
      {
        if (Shape() == CORE::FE::CellType::tri6)
          ShapeFunctions(MORTAR::Element::quad2D_only_lin, xi, val, deriv);
        else if (Shape() == CORE::FE::CellType::quad8)
          ShapeFunctions(MORTAR::Element::serendipity2D_only_lin, xi, val, deriv);
        else
          /*Shape()==quad9*/ ShapeFunctions(MORTAR::Element::biquad2D_only_lin, xi, val, deriv);
      }

      break;
    }

      // unknown case
    default:
    {
      dserror("EvaluateShapeLagMult called for unknown element type");
      break;
    }
  }

  return true;
}
/*----------------------------------------------------------------------*
 |  1D/2D shape function linearizations repository            popp 05/08|
 *----------------------------------------------------------------------*/
void MORTAR::Element::ShapeFunctionLinearizations(MORTAR::Element::ShapeType shape,
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivdual)
{
  switch (shape)
  {
    // in case of consistent dual shape functions we have an entry here
    case MORTAR::Element::lindual1D:
    case MORTAR::Element::lindual2D:
    {
      if (MoData().DerivDualShape() != Teuchos::null) derivdual = *(MoData().DerivDualShape());
      break;
    }

      // *********************************************************************
      // 2D dual bilinear shape functions (quad4)
      // (used for interpolation of Lagrange multiplier field)
      // (linearization necessary due to adaption for distorted elements !!!)
      // *********************************************************************
    case MORTAR::Element::bilindual2D:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        static const int nnodes = 4;
#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif
        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);
        CORE::LINALG::Matrix<nnodes, 1> val;

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          UTILS::mortar_shape_function_2D(val, gpc[0], gpc[1], MORTAR::Element::bilin2D);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 3);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (MoData().DualShape() == Teuchos::null)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        else
          ae = *(MoData().DualShape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        derivdual = *(MoData().DerivDualShape());
      }
      break;
    }
    // *********************************************************************
    // 1D dual quadratic shape functions (line3/nurbs3)
    // (used for interpolation of Lagrange multiplier field)
    // (linearization necessary due to adaption for distorted elements !!!)
    // *********************************************************************
    case MORTAR::Element::quaddual1D:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 3;
#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif
        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 2, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::SerialDenseVector val(nnodes);
        CORE::LINALG::SerialDenseMatrix deriv(nnodes, 2, true);
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 2, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));


        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 2);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (MoData().DualShape() == Teuchos::null)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        else
          ae = *(MoData().DualShape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
      }
      derivdual = *(MoData().DerivDualShape());

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=0;i<nnodes;++i)
      //  for (int j=0;j<nnodes;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;

      /*
       #ifdef BACI_DEBUG
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       CORE::LINALG::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;
       int thedim=3;
       if (shape==MORTAR::Element::quaddual1D) thedim=2;

       for (int dim=0;dim<thedim;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       DRT::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       mycnode->xspatial()[dim] += delta;

       CORE::LINALG::SerialDenseVector val1(nnodes);
       CORE::LINALG::SerialDenseMatrix deriv1(nnodes,2,true);
       CORE::LINALG::SerialDenseMatrix me1(nnodes,nnodes,true);
       CORE::LINALG::SerialDenseMatrix de1(nnodes,nnodes,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
       EvaluateShape(gpc1, val1, deriv1, nnodes);
       detg = Jacobian(gpc1);

       for (int j=0;j<nnodes;++j)
       for (int k=0;k<nnodes;++k)
       {
       double facme1 = integrator.Weight(i)*val1[j]*val1[k];
       double facde1 = (j==k)*integrator.Weight(i)*val1[j];

       me1(j,k)+=facme1*detg;
       de1(j,k)+=facde1*detg;
       }
       }

       // invert bi-ortho matrix me
       CORE::LINALG::SymmetricInverse(me1,nnodes);

       // get solution matrix ae with dual parameters
       CORE::LINALG::SerialDenseMatrix ae1(nnodes,nnodes);
       ae1.Multiply('N','N',1.0,de1,me1,0.0);
       int col= mycnode->Dofs()[dim];

       std::cout << "A-Derivative: " << col << std::endl;

       // FD solution
       for (int i=0;i<nnodes;++i)
       for (int j=0;j<nnodes;++j)
       {
       double val = (ae1(i,j)-aeref(i,j))/delta;
       std::cout << "A" << i << j << " " << val << std::endl;
       }

       // undo FD
       mycnode->xspatial()[dim] -= delta;
       }
       }
       // *******************************************************************
       #endif // #ifdef BACI_DEBUG
       */

      break;
    }
    // *********************************************************************
    // 1D dual quadratic shape functions (line3)
    // (used for linear interpolation of Lagrange multiplier field)
    // (linearization necessary due to adaption for distorted elements !!!)
    // *********************************************************************
    case MORTAR::Element::quaddual1D_only_lin:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());
      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 3;

#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 2, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::SerialDenseVector val(nnodes);
        CORE::LINALG::SerialDenseMatrix deriv(nnodes, 2, true);
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 2, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes, true);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 2);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);
              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }

          double fac = 0.0;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // compute inverse of matrix M_e and matrix A_e
        if (MoData().DualShape() == Teuchos::null)
        {
          // how many non-zero nodes
          const int nnodeslin = 2;

          // reduce me to non-zero nodes before inverting
          CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin;
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

          // invert bi-ortho matrix melin
          CORE::LINALG::Inverse(melin);

          // ensure zero coefficients for nodes without Lagrange multiplier
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k) me(j, k) = 0.0;

          // re-inflate inverse of melin to full size
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) me(j, k) = melin(j, k);

          // get solution matrix with dual parameters
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          // store coefficient matrix
          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        // compute inverse of matrix M_e and get matrix A_e
        else
        {
          // invert matrix M_e
          CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

          // get coefficient matrix A_e
          ae = *(MoData().DualShape());
        }

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Ae * Me = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);
              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        derivdual = *(MoData().DerivDualShape());
      }

      break;
    }
    // *********************************************************************
    // 2D dual biquadratic shape functions (quad9)
    // (used for interpolation of Lagrange multiplier field)
    // (linearization necessary due to adaption for distorted elements !!!)
    // *********************************************************************
    case MORTAR::Element::biquaddual2D:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 9;
#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::SerialDenseVector val(nnodes);
        CORE::LINALG::SerialDenseMatrix deriv(nnodes, 2, true);
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 3);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (MoData().DualShape() == Teuchos::null)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        else
          ae = *(MoData().DualShape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        derivdual = *(MoData().DerivDualShape());
      }

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=0;i<nnodes;++i)
      //  for (int j=0;j<nnodes;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;

      /*
       #ifdef BACI_DEBUG
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       CORE::LINALG::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;
       int thedim=3;
       if (shape==MORTAR::Element::quaddual1D) thedim=2;

       for (int dim=0;dim<thedim;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       DRT::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       mycnode->xspatial()[dim] += delta;

       CORE::LINALG::SerialDenseVector val1(nnodes);
       CORE::LINALG::SerialDenseMatrix deriv1(nnodes,2,true);
       CORE::LINALG::SerialDenseMatrix me1(nnodes,nnodes,true);
       CORE::LINALG::SerialDenseMatrix de1(nnodes,nnodes,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
       EvaluateShape(gpc1, val1, deriv1, nnodes);
       detg = Jacobian(gpc1);

       for (int j=0;j<nnodes;++j)
       for (int k=0;k<nnodes;++k)
       {
       double facme1 = integrator.Weight(i)*val1[j]*val1[k];
       double facde1 = (j==k)*integrator.Weight(i)*val1[j];

       me1(j,k)+=facme1*detg;
       de1(j,k)+=facde1*detg;
       }
       }

       // invert bi-ortho matrix me
       CORE::LINALG::SymmetricInverse(me1,nnodes);

       // get solution matrix ae with dual parameters
       CORE::LINALG::SerialDenseMatrix ae1(nnodes,nnodes);
       ae1.Multiply('N','N',1.0,de1,me1,0.0);
       int col= mycnode->Dofs()[dim];

       std::cout << "A-Derivative: " << col << std::endl;

       // FD solution
       for (int i=0;i<nnodes;++i)
       for (int j=0;j<nnodes;++j)
       {
       double val = (ae1(i,j)-aeref(i,j))/delta;
       std::cout << "A" << i << j << " " << val << std::endl;
       }

       // undo FD
       mycnode->xspatial()[dim] -= delta;
       }
       }
       // *******************************************************************
       #endif // #ifdef BACI_DEBUG
       */

      break;
    }
      // *********************************************************************
      // 2D dual quadratic shape functions (tri6)
      // (used for interpolation of Lagrange multiplier field)
      // (linearization necessary due to adaption for distorted elements !!!)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::quaddual2D:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 6;
#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::SerialDenseVector val(nnodes);
        CORE::LINALG::SerialDenseMatrix deriv(nnodes, 2, true);
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes, true);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 3);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (MoData().DualShape() == Teuchos::null)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        else
          ae = *(MoData().DualShape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        derivdual = *(MoData().DerivDualShape());
      }

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=0;i<nnodes;++i)
      //  for (int j=0;j<nnodes;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;
      /*
       #ifdef BACI_DEBUG
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       CORE::LINALG::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;
       int thedim=3;

       for (int dim=0;dim<thedim;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       DRT::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       mycnode->xspatial()[dim] += delta;

       CORE::LINALG::SerialDenseVector val1(nnodes);
       CORE::LINALG::SerialDenseMatrix deriv1(nnodes,2,true);
       CORE::LINALG::SerialDenseMatrix me1(nnodes,nnodes,true);
       CORE::LINALG::SerialDenseMatrix de1(nnodes,nnodes,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
       EvaluateShape(gpc1, val1, deriv1, nnodes, true);
       detg = Jacobian(gpc1);

       for (int j=0;j<nnodes;++j)
       for (int k=0;k<nnodes;++k)
       {
       double facme1 = integrator.Weight(i)*val1[j]*val1[k];
       double facde1 = (j==k)*integrator.Weight(i)*val1[j];

       me1(j,k)+=facme1*detg;
       de1(j,k)+=facde1*detg;
       }
       }

       // invert bi-ortho matrix me
       CORE::LINALG::SymmetricInverse(me1,nnodes);

       // get solution matrix ae with dual parameters
       CORE::LINALG::SerialDenseMatrix ae1(nnodes,nnodes);
       ae1.Multiply('N','N',1.0,de1,me1,0.0);
       int col= mycnode->Dofs()[dim];

       std::cout << "A-Derivative: " << col << std::endl;

       // FD solution
       for (int i=0;i<nnodes;++i)
       for (int j=0;j<nnodes;++j)
       {
       double val = (ae1(i,j)-aeref(i,j))/delta;
       std::cout << "A" << i << j << " " << val << std::endl;
       }

       // undo FD
       mycnode->xspatial()[dim] -= delta;
       }
       }
       // *******************************************************************
       #endif // #ifdef BACI_DEBUG
       */
      break;
    }
      // *********************************************************************
      // 2D dual serendipity shape functions (quad8)
      // (used for interpolation of Lagrange multiplier field)
      // (linearization necessary due to adaption for distorted elements !!!)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::serendipitydual2D:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 8;
#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::SerialDenseVector val(nnodes);
        CORE::LINALG::SerialDenseMatrix deriv(nnodes, 2, true);
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes, true);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 3);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (MoData().DualShape() == Teuchos::null)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        else
          ae = *(MoData().DualShape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        derivdual = *(MoData().DerivDualShape());

        //      std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
        //      for (int i=0;i<nnodes;++i)
        //        for (int j=0;j<nnodes;++j)
        //          for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
        //            std::cout << "A" << i << j << " " << p->first << " " << p->second <<
        //            std::endl;
        //
        //      // *******************************************************************
        //      // FINITE DIFFERENCE check of Lin(Ae)
        //      // *******************************************************************
        //
        //      std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
        //      CORE::LINALG::SerialDenseMatrix aeref(ae);
        //      double delta = 1e-8;
        //      int thedim=3;
        //
        //      for (int dim=0;dim<thedim;++dim)
        //      {
        //        for (int node=0;node<nnodes;++node)
        //        {
        //          // apply FD
        //          DRT::Node** mynodes = Nodes();
        //          Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
        //          mycnode->xspatial()[dim] += delta;
        //
        //          CORE::LINALG::SerialDenseVector val1(nnodes);
        //          CORE::LINALG::SerialDenseMatrix deriv1(nnodes,2,true);
        //          CORE::LINALG::SerialDenseMatrix me1(nnodes,nnodes,true);
        //          CORE::LINALG::SerialDenseMatrix de1(nnodes,nnodes,true);
        //
        //          // build me, de
        //          for (int i=0;i<integrator.nGP();++i)
        //          {
        //            double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
        //            EvaluateShape(gpc1, val1, deriv1, nnodes, true);
        //            detg = Jacobian(gpc1);
        //
        //            for (int j=0;j<nnodes;++j)
        //              for (int k=0;k<nnodes;++k)
        //              {
        //                double facme1 = integrator.Weight(i)*val1[j]*val1[k];
        //                double facde1 = (j==k)*integrator.Weight(i)*val1[j];
        //
        //                me1(j,k)+=facme1*detg;
        //                de1(j,k)+=facde1*detg;
        //              }
        //          }
        //
        //          // invert bi-ortho matrix me
        //          CORE::LINALG::SymmetricInverse(me1,nnodes);
        //
        //          // get solution matrix ae with dual parameters
        //          CORE::LINALG::SerialDenseMatrix ae1(nnodes,nnodes);
        //          ae1.Multiply('N','N',1.0,de1,me1,0.0);
        //          int col= mycnode->Dofs()[dim];
        //
        //          std::cout << "A-Derivative: " << col << std::endl;
        //
        //          // FD solution
        //          for (int i=0;i<nnodes;++i)
        //            for (int j=0;j<nnodes;++j)
        //            {
        //              double val = (ae1(i,j)-aeref(i,j))/delta;
        //              std::cout << "A" << i << j << " " << val << std::endl;
        //            }
        //
        //          // undo FD
        //          mycnode->xspatial()[dim] -= delta;
        //        }
        //      }
        // *******************************************************************
      }

      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (linear)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // (linearization necessary due to adaption for distorted elements !!!)
      // *********************************************************************
    case MORTAR::Element::quaddual1D_edge0:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = NumNode();
      typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 1);
      CORE::LINALG::SerialDenseVector vallin(nnodes - 1);
      CORE::LINALG::SerialDenseMatrix derivlin(nnodes - 1, 1);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      MORTAR::ElementIntegrator integrator(Shape());

      CORE::LINALG::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      CORE::LINALG::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      // two-dim arrays of maps for linearization of me/de
      std::vector<std::vector<CORE::GEN::pairedvector<int, double>>> derivme(
          nnodes, std::vector<CORE::GEN::pairedvector<int, double>>(nnodes, 3 * nnodes));
      std::vector<std::vector<CORE::GEN::pairedvector<int, double>>> derivde(
          nnodes, std::vector<CORE::GEN::pairedvector<int, double>>(nnodes, 3 * nnodes));

      for (int i = 0; i < integrator.nGP(); ++i)
      {
        double gpc[2] = {integrator.Coordinate(i, 0), 0.0};
        ShapeFunctions(MORTAR::Element::quad1D, gpc, valquad, derivquad);
        ShapeFunctions(MORTAR::Element::dual1D_base_for_edge0, gpc, vallin, derivlin);
        detg = Jacobian(gpc);

        // directional derivative of Jacobian
        CORE::GEN::pairedvector<int, double> testmap(nnodes * 2);
        DerivJacobian(gpc, testmap);

        // loop over all entries of me/de
        for (int j = 1; j < nnodes; ++j)
          for (int k = 1; k < nnodes; ++k)
          {
            double facme = integrator.Weight(i) * vallin[j - 1] * valquad[k];
            double facde = (j == k) * integrator.Weight(i) * valquad[k];

            me(j - 1, k - 1) += facme * detg;
            de(j - 1, k - 1) += facde * detg;

            // loop over all directional derivatives
            for (CI p = testmap.begin(); p != testmap.end(); ++p)
            {
              derivme[j - 1][k - 1][p->first] += facme * (p->second);
              derivde[j - 1][k - 1][p->first] += facde * (p->second);
            }
          }
      }

      // invert bi-ortho matrix me
      // CAUTION: This is a non-symmetric inverse operation!
      const double detmeinv = 1.0 / (me(0, 0) * me(1, 1) - me(0, 1) * me(1, 0));
      CORE::LINALG::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      CORE::LINALG::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      CORE::LINALG::multiply(ae, de, me);

      // build linearization of ae and store in derivdual
      // (this is done according to a quite complex formula, which
      // we get from the linearization of the biorthogonality condition:
      // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )

      // loop over all entries of ae (index i,j)
      for (int i = 1; i < nnodes; ++i)
      {
        for (int j = 1; j < nnodes; ++j)
        {
          // compute Lin(Ae) according to formula above
          for (int l = 1; l < nnodes; ++l)  // loop over sum l
          {
            // part1: Lin(De)*Inv(Me)
            for (CI p = derivde[i - 1][l - 1].begin(); p != derivde[i - 1][l - 1].end(); ++p)
              derivdual[i][j][p->first] += me(l - 1, j - 1) * (p->second);

            // part2: Ae*Lin(Me)*Inv(Me)
            for (int k = 1; k < nnodes; ++k)  // loop over sum k
            {
              for (CI p = derivme[k - 1][l - 1].begin(); p != derivme[k - 1][l - 1].end(); ++p)
                derivdual[i][j][p->first] -= ae(i - 1, k - 1) * me(l - 1, j - 1) * (p->second);
            }
          }
        }
      }

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=1;i<nnodes;++i)
      //  for (int j=1;j<nnodes;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;
      /*
       #ifdef BACI_DEBUG
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       CORE::LINALG::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;

       for (int dim=0;dim<2;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       coord(dim,node)+=delta;

       // empty shape function vals + derivs
       CORE::LINALG::SerialDenseVector valquad1(nnodes);
       CORE::LINALG::SerialDenseMatrix derivquad1(nnodes,1);
       CORE::LINALG::SerialDenseVector vallin1(nnodes-1);
       CORE::LINALG::SerialDenseMatrix derivlin1(nnodes-1,1);
       //CORE::LINALG::SerialDenseVector valtemp1(nnodes);
       //CORE::LINALG::SerialDenseMatrix derivtemp1(nnodes,1);
       CORE::LINALG::SerialDenseMatrix me1(nnodes-1,nnodes-1,true);
       CORE::LINALG::SerialDenseMatrix de1(nnodes-1,nnodes-1,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i), 0.0};
       ShapeFunctions(MORTAR::Element::quad1D,gpc1,valquad1,derivquad1);
       ShapeFunctions(MORTAR::Element::dual1D_base_for_edge0,gpc1,vallin1,derivlin1);
       detg = Jacobian(valquad1,derivquad1,coord);

       for (int j=1;j<nnodes;++j)
       for (int k=1;k<nnodes;++k)
       {
       double facme1 = integrator.Weight(i)*vallin1[j-1]*valquad1[k];
       double facde1 = (j==k)*integrator.Weight(i)*valquad1[k];

       me1(j-1,k-1)+=facme1*detg;
       de1(j-1,k-1)+=facde1*detg;
       }
       }

       // invert bi-ortho matrix me1
       double detme1 = me1(0,0)*me1(1,1)-me1(0,1)*me1(1,0);
       CORE::LINALG::SerialDenseMatrix meold(nnodes-1,nnodes-1);
       meold=me1;
       me1(0,0) =  1/detme1*meold(1,1);
       me1(0,1) = -1/detme1*meold(0,1);
       me1(1,0) = -1/detme1*meold(1,0);
       me1(1,1) =  1/detme1*meold(0,0);

       // get solution matrix ae with dual parameters
       CORE::LINALG::SerialDenseMatrix ae1(nnodes-1,nnodes-1);
       ae1.Multiply('N','N',1.0,de1,me1,0.0);

       DRT::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       int col= mycnode->Dofs()[dim];

       std::cout << "A-Derivative: " << col << std::endl;

       // FD solution
       for (int i=1;i<nnodes;++i)
       for (int j=1;j<nnodes;++j)
       {
       double val = (ae1(i-1,j-1)-aeref(i-1,j-1))/delta;
       std::cout << "A" << i << j << " " << val << std::endl;
       }

       // undo FD
       coord(dim,node)-=delta;
       }
       }
       // *******************************************************************
       #endif // #ifdef BACI_DEBUG
       */
      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (linear)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // (linearization necessary due to adaption for distorted elements !!!)
      // *********************************************************************
    case MORTAR::Element::quaddual1D_edge1:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = NumNode();
      typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;

      // empty shape function vals + derivs
      CORE::LINALG::SerialDenseVector valquad(nnodes);
      CORE::LINALG::SerialDenseMatrix derivquad(nnodes, 1);
      CORE::LINALG::SerialDenseVector vallin(nnodes - 1);
      CORE::LINALG::SerialDenseMatrix derivlin(nnodes - 1, 1);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      MORTAR::ElementIntegrator integrator(Shape());

      CORE::LINALG::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      CORE::LINALG::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      // two-dim arrays of maps for linearization of me/de
      std::vector<std::vector<CORE::GEN::pairedvector<int, double>>> derivme(
          nnodes, std::vector<CORE::GEN::pairedvector<int, double>>(nnodes, 2 * nnodes));
      std::vector<std::vector<CORE::GEN::pairedvector<int, double>>> derivde(
          nnodes, std::vector<CORE::GEN::pairedvector<int, double>>(nnodes, 2 * nnodes));

      for (int i = 0; i < integrator.nGP(); ++i)
      {
        double gpc[2] = {integrator.Coordinate(i, 0), 0.0};
        ShapeFunctions(MORTAR::Element::quad1D, gpc, valquad, derivquad);
        ShapeFunctions(MORTAR::Element::dual1D_base_for_edge1, gpc, vallin, derivlin);
        detg = Jacobian(gpc);

        // directional derivative of Jacobian
        CORE::GEN::pairedvector<int, double> testmap(nnodes * 2);
        DerivJacobian(gpc, testmap);

        // loop over all entries of me/de
        for (int j = 0; j < nnodes - 1; ++j)
          for (int k = 0; k < nnodes - 1; ++k)
          {
            double facme = integrator.Weight(i) * vallin[j] * valquad[2 * k];
            double facde = (j == k) * integrator.Weight(i) * valquad[2 * k];

            me(j, k) += facme * detg;
            de(j, k) += facde * detg;

            // loop over all directional derivatives
            for (CI p = testmap.begin(); p != testmap.end(); ++p)
            {
              derivme[j][k][p->first] += facme * (p->second);
              derivde[j][k][p->first] += facde * (p->second);
            }
          }
      }

      // invert bi-ortho matrix me
      // CAUTION: This is a non-symmetric inverse operation!
      const double detmeinv = 1.0 / (me(0, 0) * me(1, 1) - me(0, 1) * me(1, 0));
      CORE::LINALG::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      CORE::LINALG::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      CORE::LINALG::multiply(ae, de, me);

      // build linearization of ae and store in derivdual
      // (this is done according to a quite complex formula, which
      // we get from the linearization of the biorthogonality condition:
      // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )

      // loop over all entries of ae (index i,j)
      for (int i = 0; i < nnodes - 1; ++i)
      {
        for (int j = 0; j < nnodes - 1; ++j)
        {
          // compute Lin(Ae) according to formula above
          for (int l = 0; l < nnodes - 1; ++l)  // loop over sum l
          {
            // part1: Lin(De)*Inv(Me)
            for (CI p = derivde[i][l].begin(); p != derivde[i][l].end(); ++p)
              derivdual[i][j][p->first] += me(l, j) * (p->second);

            // part2: Ae*Lin(Me)*Inv(Me)
            for (int k = 0; k < nnodes - 1; ++k)  // loop over sum k
            {
              for (CI p = derivme[k][l].begin(); p != derivme[k][l].end(); ++p)
                derivdual[i][j][p->first] -= ae(i, k) * me(l, j) * (p->second);
            }
          }
        }
      }

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=0;i<nnodes-1;++i)
      //  for (int j=0;j<nnodes-1;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;
      /*
       #ifdef BACI_DEBUG
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       CORE::LINALG::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;

       for (int dim=0;dim<2;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       coord(dim,node)+=delta;

       // empty shape function vals + derivs
       CORE::LINALG::SerialDenseVector valquad1(nnodes);
       CORE::LINALG::SerialDenseMatrix derivquad1(nnodes,1);
       CORE::LINALG::SerialDenseVector vallin1(nnodes-1);
       CORE::LINALG::SerialDenseMatrix derivlin1(nnodes-1,1);
       //CORE::LINALG::SerialDenseVector valtemp1(nnodes);
       //CORE::LINALG::SerialDenseMatrix derivtemp1(nnodes,1);
       CORE::LINALG::SerialDenseMatrix me1(nnodes-1,nnodes-1,true);
       CORE::LINALG::SerialDenseMatrix de1(nnodes-1,nnodes-1,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i), 0.0};
       ShapeFunctions(MORTAR::Element::quad1D,gpc1,valquad1,derivquad1);
       ShapeFunctions(MORTAR::Element::dual1D_base_for_edge1,gpc1,vallin1,derivlin1);
       detg = Jacobian(valquad1,derivquad1,coord);

       for (int j=0;j<nnodes-1;++j)
       for (int k=0;k<nnodes-1;++k)
       {
       double facme1 = integrator.Weight(i)*vallin1[j]*valquad1[2*k];
       double facde1 = (j==k)*integrator.Weight(i)*valquad1[2*k];

       me1(j,k)+=facme1*detg;
       de1(j,k)+=facde1*detg;
       }
       }

       // invert bi-ortho matrix me1
       double detme1 = me1(0,0)*me1(1,1)-me1(0,1)*me1(1,0);
       CORE::LINALG::SerialDenseMatrix meold(nnodes-1,nnodes-1);
       meold=me1;
       me1(0,0) =  1/detme1*meold(1,1);
       me1(0,1) = -1/detme1*meold(0,1);
       me1(1,0) = -1/detme1*meold(1,0);
       me1(1,1) =  1/detme1*meold(0,0);

       // get solution matrix ae with dual parameters
       CORE::LINALG::SerialDenseMatrix ae1(nnodes-1,nnodes-1);
       ae1.Multiply('N','N',1.0,de1,me1,0.0);

       DRT::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       int col= mycnode->Dofs()[dim];

       std::cout << "A-Derivative: " << col << std::endl;

       // FD solution
       for (int i=0;i<nnodes-1;++i)
       for (int j=0;j<nnodes-1;++j)
       {
       double val = (ae1(i,j)-aeref(i,j))/delta;
       std::cout << "A" << i << j << " " << val << std::endl;
       }

       // undo FD
       coord(dim,node)-=delta;
       }
       }
       // *******************************************************************
       #endif // #ifdef BACI_DEBUG
       */
      break;
    }
      //***********************************************************************
      // 2D dual quadratic shape functions (tri6)
      // (used for LINEAR interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      //***********************************************************************
    case MORTAR::Element::quaddual2D_only_lin:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());
      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 6;

#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::SerialDenseVector val(nnodes);
        CORE::LINALG::SerialDenseMatrix deriv(nnodes, 2, true);
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes, true);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 3);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);
              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }

          double fac = 0.0;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // compute inverse of matrix M_e and matrix A_e
        if (MoData().DualShape() == Teuchos::null)
        {
          // how many non-zero nodes
          const int nnodeslin = 3;

          // reduce me to non-zero nodes before inverting
          CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin;
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

          // invert bi-ortho matrix melin
          CORE::LINALG::Inverse(melin);

          // ensure zero coefficients for nodes without Lagrange multiplier
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k) me(j, k) = 0.0;

          // re-inflate inverse of melin to full size
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) me(j, k) = melin(j, k);

          // get solution matrix with dual parameters
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          // store coefficient matrix
          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        // compute inverse of matrix M_e and get matrix A_e
        else
        {
          // invert matrix M_e
          CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

          // get coefficient matrix A_e
          ae = *(MoData().DualShape());
        }

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Ae * Me = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);
              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        derivdual = *(MoData().DerivDualShape());
      }
      break;
    }

      // *********************************************************************
      // 2D dual serendipity shape functions (quad8)
      // (used for LINEAR interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::serendipitydual2D_only_lin:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());
      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 8;

#ifdef BACI_DEBUG
        if (nnodes != NumNode())
          dserror(
              "MORTAR::Element shape function for LM incompatible with number of element nodes!");
#endif

        MoData().DerivDualShape() =
            Teuchos::rcp(new CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>(
                nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes)));
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivae =
            *(MoData().DerivDualShape());

        typedef CORE::GEN::pairedvector<int, double>::const_iterator CI;
        CORE::LINALG::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        MORTAR::ElementIntegrator integrator(Shape());
        CORE::LINALG::SerialDenseVector val(nnodes);
        CORE::LINALG::SerialDenseMatrix deriv(nnodes, 2, true);
        CORE::LINALG::Matrix<nnodes, nnodes> me(true);
        CORE::LINALG::Matrix<nnodes, nnodes> de(true);

        // two-dim arrays of maps for linearization of me/de
        CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, CORE::LINALG::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.nGP(); ++i)
        {
          double gpc[2] = {integrator.Coordinate(i, 0), integrator.Coordinate(i, 1)};
          EvaluateShape(gpc, val, deriv, nnodes, true);
          detg = Jacobian(gpc);

          // directional derivative of Jacobian
          CORE::GEN::pairedvector<int, double> testmap(nnodes * 3);
          DerivJacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.Weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.Weight(i) * val(j);
              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }

          double fac = 0.0;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.Weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // compute inverse of matrix M_e and matrix A_e
        if (MoData().DualShape() == Teuchos::null)
        {
          // how many non-zero nodes
          const int nnodeslin = 4;

          // reduce me to non-zero nodes before inverting
          CORE::LINALG::Matrix<nnodeslin, nnodeslin> melin;
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

          // invert bi-ortho matrix melin
          CORE::LINALG::Inverse(melin);

          // ensure zero coefficients for nodes without Lagrange multiplier
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k) me(j, k) = 0.0;

          // re-inflate inverse of melin to full size
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) me(j, k) = melin(j, k);

          // get solution matrix with dual parameters
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          // store coefficient matrix
          MoData().DualShape() = Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix(ae));
        }
        // compute inverse of matrix M_e and get matrix A_e
        else
        {
          // invert matrix M_e
          CORE::LINALG::SymmetricPositiveDefiniteInverse<nnodes>(me);

          // get coefficient matrix A_e
          ae = *(MoData().DualShape());
        }

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Ae * Me = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          CORE::LINALG::SerialDenseMatrix& dtmp = derivde_me[p->first];
          CORE::LINALG::SerialDenseMatrix& pt = derivae[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);
              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        derivdual = *(MoData().DerivDualShape());
      }
      break;
    }

      // *********************************************************************
      // 2D dual biquadratic shape functions (quad9)
      // (used for LINEAR interpolation of Lagrange mutliplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case MORTAR::Element::biquaddual2D_only_lin:
    {
      dserror("biquaddual2D_only_lin not available!");
      break;
    }
      // *********************************************************************
      // Unknown shape function type
      // *********************************************************************
    default:
    {
      dserror("Unknown shape function type identifier");
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate 2nd derivative of shape functions                popp 05/08|
 *----------------------------------------------------------------------*/
bool MORTAR::Element::Evaluate2ndDerivShape(
    const double* xi, CORE::LINALG::SerialDenseMatrix& secderiv, const int& valdim)
{
  if (!xi) dserror("Evaluate2ndDerivShape called with xi=nullptr");

  //**********************************************************************
  // IMPORTANT NOTE: In 3D the ordering of the 2nd derivatives is:
  // 1) dxi,dxi 2) deta,deta 3) dxi,deta
  //**********************************************************************

  switch (Shape())
  {
    // 2D linear case (2noded line element)
    case CORE::FE::CellType::line2:
    {
      secderiv(0, 0) = 0.0;
      secderiv(1, 0) = 0.0;
      break;
    }
      // 2D quadratic case (3noded line element)
    case CORE::FE::CellType::line3:
    {
      secderiv(0, 0) = 1.0;
      secderiv(1, 0) = 1.0;
      secderiv(2, 0) = -2.0;
      break;
    }
      // 3D linear case (3noded triangular element)
    case CORE::FE::CellType::tri3:
    {
      secderiv(0, 0) = 0.0;
      secderiv(0, 1) = 0.0;
      secderiv(0, 2) = 0.0;
      secderiv(1, 0) = 0.0;
      secderiv(1, 1) = 0.0;
      secderiv(1, 2) = 0.0;
      secderiv(2, 0) = 0.0;
      secderiv(2, 1) = 0.0;
      secderiv(2, 2) = 0.0;
      break;
    }
      // 3D bilinear case (4noded quadrilateral element)
    case CORE::FE::CellType::quad4:
    {
      secderiv(0, 0) = 0.0;
      secderiv(0, 1) = 0.0;
      secderiv(0, 2) = 0.25;
      secderiv(1, 0) = 0.0;
      secderiv(1, 1) = 0.0;
      secderiv(1, 2) = -0.25;
      secderiv(2, 0) = 0.0;
      secderiv(2, 1) = 0.0;
      secderiv(2, 2) = 0.25;
      secderiv(3, 0) = 0.0;
      secderiv(3, 1) = 0.0;
      secderiv(3, 2) = -0.25;
      break;
    }
      // 3D quadratic case (6noded triangular element)
    case CORE::FE::CellType::tri6:
    {
      secderiv(0, 0) = 4.0;
      secderiv(0, 1) = 4.0;
      secderiv(0, 2) = 4.0;
      secderiv(1, 0) = 4.0;
      secderiv(1, 1) = 0.0;
      secderiv(1, 2) = 0.0;
      secderiv(2, 0) = 0.0;
      secderiv(2, 1) = 4.0;
      secderiv(2, 2) = 0.0;
      secderiv(3, 0) = -8.0;
      secderiv(3, 1) = 0.0;
      secderiv(3, 2) = -4.0;
      secderiv(4, 0) = 0.0;
      secderiv(4, 1) = 0.0;
      secderiv(4, 2) = 4.0;
      secderiv(5, 0) = 0.0;
      secderiv(5, 1) = -8.0;
      secderiv(5, 2) = -4.0;
      break;
    }
      // 3D serendipity case (8noded quadrilateral element)
    case CORE::FE::CellType::quad8:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;

      secderiv(0, 0) = 0.5 * sm;
      secderiv(0, 1) = 0.5 * rm;
      secderiv(0, 2) = -0.25 * (2 * r + 2 * s - 1.0);
      secderiv(1, 0) = 0.5 * sm;
      secderiv(1, 1) = 0.5 * rp;
      secderiv(1, 2) = 0.25 * (-2 * r + 2 * s - 1.0);
      secderiv(2, 0) = 0.5 * sp;
      secderiv(2, 1) = 0.5 * rp;
      secderiv(2, 2) = 0.25 * (2 * r + 2 * s + 1.0);
      secderiv(3, 0) = 0.5 * sp;
      secderiv(3, 1) = 0.5 * rm;
      secderiv(3, 2) = -0.25 * (-2 * r + 2 * s + 1.0);
      secderiv(4, 0) = -sm;
      secderiv(4, 1) = 0.0;
      secderiv(4, 2) = r;
      secderiv(5, 0) = 0.0;
      secderiv(5, 1) = -rp;
      secderiv(5, 2) = -s;
      secderiv(6, 0) = -sp;
      secderiv(6, 1) = 0.0;
      secderiv(6, 2) = -r;
      secderiv(7, 0) = 0.0;
      secderiv(7, 1) = -rm;
      secderiv(7, 2) = s;
      break;
    }
      // 3D biquadratic case (9noded quadrilateral element)
    case CORE::FE::CellType::quad9:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double rp = 1.0 + r;
      const double rm = 1.0 - r;
      const double sp = 1.0 + s;
      const double sm = 1.0 - s;
      const double r2 = 1.0 - r * r;
      const double s2 = 1.0 - s * s;
      const double rh = 0.5 * r;
      const double sh = 0.5 * s;
      const double rhp = r + 0.5;
      const double rhm = r - 0.5;
      const double shp = s + 0.5;
      const double shm = s - 0.5;

      secderiv(0, 0) = -sh * sm;
      secderiv(0, 1) = -rh * rm;
      secderiv(0, 2) = shm * rhm;
      secderiv(1, 0) = -sh * sm;
      secderiv(1, 1) = rh * rp;
      secderiv(1, 2) = shm * rhp;
      secderiv(2, 0) = sh * sp;
      secderiv(2, 1) = rh * rp;
      secderiv(2, 2) = shp * rhp;
      secderiv(3, 0) = sh * sp;
      secderiv(3, 1) = -rh * rm;
      secderiv(3, 2) = shp * rhm;
      secderiv(4, 0) = 2.0 * sh * sm;
      secderiv(4, 1) = r2;
      secderiv(4, 2) = -2.0 * r * shm;
      secderiv(5, 0) = s2;
      secderiv(5, 1) = -2.0 * rh * rp;
      secderiv(5, 2) = -2.0 * s * rhp;
      secderiv(6, 0) = -2.0 * sh * sp;
      secderiv(6, 1) = r2;
      secderiv(6, 2) = -2.0 * r * shp;
      secderiv(7, 0) = s2;
      secderiv(7, 1) = 2.0 * rh * rm;
      secderiv(7, 2) = -2.0 * s * rhm;
      secderiv(8, 0) = -2.0 * s2;
      secderiv(8, 1) = -2.0 * r2;
      secderiv(8, 2) = 2.0 * s * 2.0 * r;
      break;
    }

      //==================================================
      //                     NURBS
      //==================================================
      // 1D -- nurbs2
    case CORE::FE::CellType::nurbs2:
    {
      if (valdim != 2) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseVector auxval(NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv(1, NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv2(1, NumNode());

      CORE::FE::NURBS::nurbs_get_1D_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, xi[0], Knots()[0], weights, CORE::FE::CellType::nurbs2);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < NumNode(); ++i) secderiv(i, 0) = auxderiv2(0, i);

      break;
    }

      // 1D -- nurbs3
    case CORE::FE::CellType::nurbs3:
    {
      if (valdim != 3) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(3);
      for (int inode = 0; inode < 3; ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseVector auxval(3);
      CORE::LINALG::SerialDenseMatrix auxderiv(1, 3);
      CORE::LINALG::SerialDenseMatrix auxderiv2(1, 3);

      CORE::FE::NURBS::nurbs_get_1D_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, xi[0], Knots()[0], weights, CORE::FE::CellType::nurbs3);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < NumNode(); ++i) secderiv(i, 0) = auxderiv2(0, i);

      break;
    }

      // ===========================================================
      // 2D -- nurbs4
    case CORE::FE::CellType::nurbs4:
    {
      if (valdim != 4) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      CORE::LINALG::SerialDenseVector auxval(NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv(2, NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv2(3, NumNode());

      CORE::FE::NURBS::nurbs_get_2D_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, uv, Knots(), weights, CORE::FE::CellType::nurbs4);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 3; ++d)
        for (int i = 0; i < NumNode(); ++i) secderiv(i, d) = auxderiv2(d, i);

      break;
    }

      // 2D -- nurbs8
    case CORE::FE::CellType::nurbs8:
    {
      if (valdim != 8) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      CORE::LINALG::SerialDenseVector auxval(NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv(2, NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv2(3, NumNode());

      CORE::FE::NURBS::nurbs_get_2D_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, uv, Knots(), weights, CORE::FE::CellType::nurbs8);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 3; ++d)
        for (int i = 0; i < NumNode(); ++i) secderiv(i, d) = auxderiv2(d, i);

      break;
    }

      // 2D -- nurbs9
    case CORE::FE::CellType::nurbs9:
    {
      if (valdim != 9) dserror("Inconsistency in EvaluateShape");

      CORE::LINALG::SerialDenseVector weights(NumNode());
      for (int inode = 0; inode < NumNode(); ++inode)
        weights(inode) = dynamic_cast<MORTAR::Node*>(Nodes()[inode])->NurbsW();

      CORE::LINALG::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      CORE::LINALG::SerialDenseVector auxval(NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv(2, NumNode());
      CORE::LINALG::SerialDenseMatrix auxderiv2(3, NumNode());

      CORE::FE::NURBS::nurbs_get_2D_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, uv, Knots(), weights, CORE::FE::CellType::nurbs9);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 3; ++d)
        for (int i = 0; i < NumNode(); ++i) secderiv(i, d) = auxderiv2(d, i);

      break;
    }
      // unknown case
    default:
      dserror("Evaluate2ndDerivShape called for unknown element type");
      break;
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Compute directional derivative of dual shape functions    popp 05/08|
 *----------------------------------------------------------------------*/
bool MORTAR::Element::DerivShapeDual(
    CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>& derivdual)
{
  // get node number and node pointers
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("DerivShapeDual: Null pointer!");

  switch (Shape())
  {
    // 2D linear case (2noded line element)
    case CORE::FE::CellType::line2:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());
      else
        derivdual.resize(0);

      break;
    }
      // 3D linear case (3noded triangular element)
    case CORE::FE::CellType::tri3:
    {
      if (MoData().DerivDualShape() != Teuchos::null)
        derivdual = *(MoData().DerivDualShape());
      else
        derivdual.resize(0);
      break;
    }

      // 2D quadratic case (3noded line element)
    case CORE::FE::CellType::line3:
    {
      // check for middle "bound" node
      Node* mycnode2 = dynamic_cast<Node*>(mynodes[2]);
      if (!mycnode2) dserror("DerivShapeDual: Null pointer!");
      bool isonbound2 = mycnode2->IsOnBound();

      // locally linear Lagrange multipliers
      if (isonbound2) ShapeFunctionLinearizations(MORTAR::Element::quaddual1D_only_lin, derivdual);
      // use unmodified dual shape functions
      else
        ShapeFunctionLinearizations(MORTAR::Element::quaddual1D, derivdual);

      break;
    }

      // all other 3D cases
    case CORE::FE::CellType::quad4:
    case CORE::FE::CellType::tri6:
    case CORE::FE::CellType::quad8:
    case CORE::FE::CellType::quad9:
    {
      if (Shape() == CORE::FE::CellType::quad4)
        ShapeFunctionLinearizations(MORTAR::Element::bilindual2D, derivdual);
      else if (Shape() == CORE::FE::CellType::tri6)
        ShapeFunctionLinearizations(MORTAR::Element::quaddual2D, derivdual);
      else if (Shape() == CORE::FE::CellType::quad8)
        ShapeFunctionLinearizations(MORTAR::Element::serendipitydual2D, derivdual);
      else
        /*Shape()==quad9*/ ShapeFunctionLinearizations(MORTAR::Element::biquaddual2D, derivdual);

      break;
    }

    //==================================================
    //                     NURBS
    //==================================================
    case CORE::FE::CellType::nurbs3:
    {
      ShapeFunctionLinearizations(MORTAR::Element::quaddual1D, derivdual);
      break;
    }
    case CORE::FE::CellType::nurbs9:
    {
      ShapeFunctionLinearizations(MORTAR::Element::biquaddual2D, derivdual);
      break;
    }
      // unknown case
    default:
    {
      dserror("DerivShapeDual called for unknown element type");
      break;
    }
  }

  // check if we need trafo
  const int nnodes = NumNode();
  bool bound = false;
  for (int i = 0; i < nnodes; ++i)
  {
    Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
    if (mymrtrnode->IsOnBoundorCE())
    {
      bound = true;
      break;
    }
  }

  if (!bound) return true;

  //---------------------------------
  // do trafo for bound elements
  CORE::LINALG::SerialDenseMatrix trafo(nnodes, nnodes, true);

  // 2D case!
  if (Shape() == CORE::FE::CellType::line2 or Shape() == CORE::FE::CellType::line3 or
      Shape() == CORE::FE::CellType::nurbs2 or Shape() == CORE::FE::CellType::nurbs3)
  {
    // get number of bound nodes
    std::vector<int> ids;
    for (int i = 0; i < nnodes; ++i)
    {
      Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
      if (mymrtrnode->IsOnCorner())
      {
        // get local bound id
        ids.push_back(i);
      }
    }

    int numbound = (int)ids.size();

    // if all bound: error
    if ((nnodes - numbound) < 1e-12) dserror("all nodes are bound");

    const double factor = 1.0 / (nnodes - numbound);
    // row loop
    for (int i = 0; i < nnodes; ++i)
    {
      Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
      if (!mymrtrnode->IsOnCorner())
      {
        trafo(i, i) = 1.0;
        for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
      }
    }
  }

  // 3D case!
  else if (Shape() == CORE::FE::CellType::tri6 or Shape() == CORE::FE::CellType::tri3 or
           Shape() == CORE::FE::CellType::quad4 or Shape() == CORE::FE::CellType::quad8 or
           Shape() == CORE::FE::CellType::quad9 or Shape() == CORE::FE::CellType::nurbs4 or
           Shape() == CORE::FE::CellType::nurbs9)
  {
    // get number of bound nodes
    std::vector<int> ids;
    for (int i = 0; i < nnodes; ++i)
    {
      Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
      if (mymrtrnode->IsOnBoundorCE())
      {
        // get local bound id
        ids.push_back(i);
      }
    }

    int numbound = (int)ids.size();

    // if all bound: error
    if ((nnodes - numbound) < 1e-12) dserror("all nodes are bound");

    const double factor = 1.0 / (nnodes - numbound);
    // row loop
    for (int i = 0; i < nnodes; ++i)
    {
      Node* mymrtrnode = dynamic_cast<Node*>(mynodes[i]);
      if (!mymrtrnode->IsOnBoundorCE())
      {
        trafo(i, i) = 1.0;
        for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
      }
    }
  }
  else
    dserror("unknown element type!");



  // do trafo
  CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix> dummy(
      nnodes * nnodes * 3 * 10, 0, CORE::LINALG::SerialDenseMatrix(nnodes, nnodes, true));

  typedef CORE::GEN::pairedvector<int, CORE::LINALG::SerialDenseMatrix>::const_iterator _CIM;
  for (_CIM p = derivdual.begin(); p != derivdual.end(); ++p)
  {
    for (int i = 0; i < nnodes; ++i)
    {
      for (int j = 0; j < nnodes; ++j)
      {
        dummy[p->first](i, j) += trafo(i, j) * (p->second)(j, i);
      }
    }
  }

  for (_CIM p = dummy.begin(); p != dummy.end(); ++p) derivdual[p->first] = p->second;


  return true;
}

BACI_NAMESPACE_CLOSE
