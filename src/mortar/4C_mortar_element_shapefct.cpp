// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"
#include "4C_mortar_shape_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  1D/2D shape function repository                           popp 04/08|
 *----------------------------------------------------------------------*/
void Mortar::Element::shape_functions(Mortar::Element::ShapeType shape, const double* xi,
    Core::LinAlg::SerialDenseVector& val, Core::LinAlg::SerialDenseMatrix& deriv) const
{
  switch (shape)
  {
    // *********************************************************************
    // 1D standard linear shape functions (line2)
    // (used for interpolation of displacement field)
    // *********************************************************************
    case Mortar::Element::lin1D:
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
    case Mortar::Element::lin1D_edge0:
    {
      FOUR_C_THROW(
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
    case Mortar::Element::lin1D_edge1:
    {
      FOUR_C_THROW(
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
    case Mortar::Element::lin2D:
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
    case Mortar::Element::bilin2D:
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
    case Mortar::Element::quad1D:
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
    case Mortar::Element::quad1D_hierarchical:
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
    case Mortar::Element::quad1D_modified:
    {
      FOUR_C_THROW("Quadratic LM for quadratic interpolation in 2D not available!");
      break;
    }
      // *********************************************************************
      // 1D modified standard shape functions (linear replacing quad, line3)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // *********************************************************************
    case Mortar::Element::quad1D_edge0:
    {
      FOUR_C_THROW(
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
    case Mortar::Element::quad1D_edge1:
    {
      FOUR_C_THROW(
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
    case Mortar::Element::quad1D_only_lin:
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
    case Mortar::Element::quad2D:
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
    case Mortar::Element::quad2D_modified:
    {
      const double r = xi[0];
      const double s = xi[1];
      const double t1 = 1.0 - r - s;
      const double t2 = r;
      const double t3 = s;

      Core::LinAlg::SerialDenseVector valtmp(num_node(), 1);
      Core::LinAlg::SerialDenseMatrix derivtmp(num_node(), 2);

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
    case Mortar::Element::quad2D_hierarchical:
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
    case Mortar::Element::quad2D_only_lin:
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
    case Mortar::Element::serendipity2D:
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
    case Mortar::Element::serendipity2D_modified:
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

      Core::LinAlg::SerialDenseVector valtmp(num_node(), 1);
      Core::LinAlg::SerialDenseMatrix derivtmp(num_node(), 2);

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
    case Mortar::Element::serendipity2D_hierarchical:
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
    case Mortar::Element::serendipity2D_only_lin:
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
    case Mortar::Element::biquad2D:
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
    case Mortar::Element::biquad2D_modified:
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

      Core::LinAlg::Matrix<9, 1> valtmp;
      Core::LinAlg::Matrix<9, 2> derivtmp;

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
    case Mortar::Element::biquad2D_hierarchical:
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
    case Mortar::Element::biquad2D_only_lin:
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
      // (used for interpolation of Lagrange multiplier field)
      // *********************************************************************
    case Mortar::Element::lindual1D:
    {
      int dim = 1;

      // use element-based dual shape functions if no coefficient matrix is stored
      if (mo_data().dual_shape() == nullptr)
      {
        val[0] = 0.5 * (1.0 - 3.0 * xi[0]);
        val[1] = 0.5 * (1.0 + 3.0 * xi[0]);
        deriv(0, 0) = -1.5;
        deriv(1, 0) = 1.5;
      }

      // pre-calculated consistent dual shape functions
      else
      {
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (mo_data().dual_shape()->numCols() != 2 && mo_data().dual_shape()->numRows() != 2)
          FOUR_C_THROW("Dual shape functions coefficient matrix calculated in the wrong size");
#endif
        const int nnodes = num_node();
        Core::LinAlg::SerialDenseVector stdval(nnodes, true);
        Core::LinAlg::SerialDenseMatrix stdderiv(nnodes, dim, true);
        Core::LinAlg::SerialDenseVector checkval(nnodes, true);
        evaluate_shape(xi, stdval, stdderiv, nnodes);
        const Core::LinAlg::SerialDenseMatrix& ae = *(mo_data().dual_shape());

        for (int i = 0; i < num_node(); ++i)
        {
          val[i] = 0.0;
          deriv(i, 0) = 0.0;
          for (int j = 0; j < num_node(); ++j)
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
    case Mortar::Element::lindual1D_edge0:
    {
      FOUR_C_THROW(
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
    case Mortar::Element::lindual1D_edge1:
    {
      FOUR_C_THROW(
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
      // (used for interpolation of Lagrange multiplier field)
      // *********************************************************************
    case Mortar::Element::lindual2D:
    {
      if (mo_data().dual_shape() == nullptr)
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
        const int nnodes = num_node();
        // get solution matrix with dual parameters
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);
        // get dual shape functions coefficient matrix from data container
        ae = *(mo_data().dual_shape());

        // evaluate dual shape functions at loc. coord. xi
        // need standard shape functions at xi first
        evaluate_shape(xi, val, deriv, nnodes);

        // dimension
        int dim = 2;

        // evaluate dual shape functions
        Core::LinAlg::SerialDenseVector valtemp(nnodes, true);
        Core::LinAlg::SerialDenseMatrix derivtemp(nnodes, dim, true);
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
      // (used for interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case Mortar::Element::bilindual2D:
    {
      const int nnodes = 4;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif
      // get solution matrix with dual parameters
      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // no pre-computed dual shape functions
      if (mo_data().dual_shape() == nullptr)
      {
        // establish fundamental data
        double detg = 0.0;

        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());

        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * val[j] * val[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * val[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        Core::LinAlg::invert_and_multiply_by_cholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }

      // pre-computed dual shape functions
      else
      {
        // get dual shape functions coefficient matrix from data container
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      // need standard shape functions at xi first
      evaluate_shape(xi, val, deriv, nnodes);

      // dimension
      const int dim = 2;

      // evaluate dual shape functions
      Core::LinAlg::SerialDenseVector valtemp(nnodes, true);
      Core::LinAlg::SerialDenseMatrix derivtemp(nnodes, dim, true);
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
      // (used for interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case Mortar::Element::quaddual1D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 3;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());

        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * val[j] * val[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * val[j] * detg;
            }
        }

        // calcute coefficient matrix
        Core::LinAlg::invert_and_multiply_by_cholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      // need standard shape functions at xi first
      evaluate_shape(xi, val, deriv, nnodes);

      // check whether this is a 1D or 2D mortar element
      int dim = 1;

      // evaluate dual shape functions
      Core::LinAlg::SerialDenseVector valtemp(nnodes, true);
      Core::LinAlg::SerialDenseMatrix derivtemp(nnodes, dim, true);
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
      // (used for LINEAR interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // *********************************************************************
    case Mortar::Element::quaddual1D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 3;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 2);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          shape_functions(Mortar::Element::quad1D_only_lin, gpc, valquad, derivquad);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 2;

        // reduce me to non-zero nodes before inverting
        Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin;
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        Core::LinAlg::inverse(melin);

        // re-inflate inverse of melin to full size
        Core::LinAlg::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        Core::LinAlg::multiply(ae, de, invme);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      shape_functions(Mortar::Element::quad1D_only_lin, xi, valquad, derivquad);
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
      // (used for interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::quaddual2D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 6;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 2);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, valquad, derivquad, nnodes, true);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * valquad[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        Core::LinAlg::invert_and_multiply_by_cholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      evaluate_shape(xi, valquad, derivquad, nnodes, true);
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
    // (used for interpolation of Lagrange multiplier field)
    // (including adaption process for distorted elements)
    // (including modification of displacement shape functions)
    // *********************************************************************
    case Mortar::Element::serendipitydual2D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 8;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 2);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, valquad, derivquad, nnodes, true);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * valquad[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        Core::LinAlg::invert_and_multiply_by_cholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      evaluate_shape(xi, valquad, derivquad, nnodes, true);
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
      // (used for interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::biquaddual2D:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 9;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 2);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, valquad, derivquad, nnodes, true);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * valquad[j] * detg;
            }
          }
        }

        // calcute coefficient matrix
        Core::LinAlg::invert_and_multiply_by_cholesky<nnodes>(me, de, ae);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      evaluate_shape(xi, valquad, derivquad, nnodes, true);
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
      // (used for LINEAR interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::quaddual2D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 6;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 2);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          shape_functions(Mortar::Element::quad2D_only_lin, gpc, valquad, derivquad);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 3;

        // reduce me to non-zero nodes before inverting
        Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin;
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        Core::LinAlg::inverse(melin);

        // re-inflate inverse of melin to full size
        Core::LinAlg::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        Core::LinAlg::multiply(ae, de, invme);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      shape_functions(Mortar::Element::quad2D_only_lin, xi, valquad, derivquad);
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
      // (used for LINEAR interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::serendipitydual2D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 8;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 2);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          shape_functions(Mortar::Element::serendipity2D_only_lin, gpc, valquad, derivquad);

          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 4;

        // reduce me to non-zero nodes before inverting
        Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin(Core::LinAlg::Initialization::zero);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        Core::LinAlg::inverse(melin);

        // re-inflate inverse of melin to full size
        Core::LinAlg::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        Core::LinAlg::multiply(ae, de, invme);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      shape_functions(Mortar::Element::serendipity2D_only_lin, xi, valquad, derivquad);
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
      // (used for LINEAR interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::biquaddual2D_only_lin:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = 9;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (nnodes != num_node())
        FOUR_C_THROW(
            "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

      Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 2);

      if (mo_data().dual_shape() == nullptr)
      {
        // compute entries to bi-ortho matrices me/de with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::SerialDenseMatrix de(nnodes, nnodes, true);

        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          shape_functions(Mortar::Element::biquad2D_only_lin, gpc, valquad, derivquad);
          detg = jacobian(gpc);

          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              me(j, k) += integrator.weight(i) * valquad[j] * valquad[k] * detg;
              de(j, k) += (j == k) * integrator.weight(i) * valquad[j] * detg;
            }
          }
        }

        // how many non-zero nodes
        const int nnodeslin = 4;

        // reduce me to non-zero nodes before inverting
        Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin(Core::LinAlg::Initialization::zero);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

        // invert bi-ortho matrix melin
        Core::LinAlg::inverse(melin);

        // re-inflate inverse of melin to full size
        Core::LinAlg::SerialDenseMatrix invme(nnodes, nnodes, true);
        for (int j = 0; j < nnodeslin; ++j)
          for (int k = 0; k < nnodeslin; ++k) invme(j, k) = melin(j, k);

        // get solution matrix with dual parameters
        Core::LinAlg::multiply(ae, de, invme);

        // store coefficient matrix
        mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
      }
      else
      {
        ae = *(mo_data().dual_shape());
      }

      // evaluate dual shape functions at loc. coord. xi
      shape_functions(Mortar::Element::biquad2D_only_lin, xi, valquad, derivquad);
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
    case Mortar::Element::dual1D_base_for_edge0:
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
    case Mortar::Element::dual1D_base_for_edge1:
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
    case Mortar::Element::quaddual1D_edge0:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = num_node();

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 1);
      Core::LinAlg::SerialDenseVector vallin(nnodes - 1);
      Core::LinAlg::SerialDenseMatrix derivlin(nnodes - 1, 1);
      Core::LinAlg::SerialDenseVector valtemp(nnodes, true);
      Core::LinAlg::SerialDenseMatrix derivtemp(nnodes, 1, true);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      Mortar::ElementIntegrator integrator(Element::shape());

      Core::LinAlg::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      Core::LinAlg::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      for (int i = 0; i < integrator.n_gp(); ++i)
      {
        double gpc[2] = {integrator.coordinate(i, 0), 0.0};
        shape_functions(Mortar::Element::quad1D, gpc, valquad, derivquad);
        shape_functions(Mortar::Element::dual1D_base_for_edge0, gpc, vallin, derivlin);
        detg = jacobian(gpc);

        for (int j = 1; j < nnodes; ++j)
          for (int k = 1; k < nnodes; ++k)
          {
            me(j - 1, k - 1) += integrator.weight(i) * vallin[j - 1] * valquad[k] * detg;
            de(j - 1, k - 1) += (j == k) * integrator.weight(i) * valquad[k] * detg;
          }
      }

      // invert bi-ortho matrix me
      // CAUTION: This is a non-symmetric inverse operation!
      const double detmeinv = 1.0 / (me(0, 0) * me(1, 1) - me(0, 1) * me(1, 0));
      Core::LinAlg::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      Core::LinAlg::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      Core::LinAlg::multiply(ae, de, me);

      // evaluate dual shape functions at loc. coord. xi
      shape_functions(Mortar::Element::dual1D_base_for_edge0, xi, vallin, derivlin);
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
    case Mortar::Element::quaddual1D_edge1:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = num_node();

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 1);
      Core::LinAlg::SerialDenseVector vallin(nnodes - 1);
      Core::LinAlg::SerialDenseMatrix derivlin(nnodes - 1, 1);
      Core::LinAlg::SerialDenseVector valtemp(nnodes, true);
      Core::LinAlg::SerialDenseMatrix derivtemp(nnodes, 1, true);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      Mortar::ElementIntegrator integrator(Element::shape());

      Core::LinAlg::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      Core::LinAlg::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      for (int i = 0; i < integrator.n_gp(); ++i)
      {
        double gpc[2] = {integrator.coordinate(i, 0), 0.0};
        shape_functions(Mortar::Element::quad1D, gpc, valquad, derivquad);
        shape_functions(Mortar::Element::dual1D_base_for_edge1, gpc, vallin, derivlin);
        detg = jacobian(gpc);

        for (int j = 0; j < nnodes - 1; ++j)
          for (int k = 0; k < nnodes - 1; ++k)
          {
            me(j, k) += integrator.weight(i) * vallin[j] * valquad[2 * k] * detg;
            de(j, k) += (j == k) * integrator.weight(i) * valquad[2 * k] * detg;
          }
      }

      // invert bi-ortho matrix me
      // CAUTION: This is a non-symmetric inverse operation!
      double detmeinv = 1.0 / (me(0, 0) * me(1, 1) - me(0, 1) * me(1, 0));
      Core::LinAlg::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      Core::LinAlg::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      Core::LinAlg::multiply(ae, de, me);

      // evaluate dual shape functions at loc. coord. xi
      shape_functions(Mortar::Element::dual1D_base_for_edge1, xi, vallin, derivlin);
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
      // Unknown shape function type
      // *********************************************************************
    default:
    {
      FOUR_C_THROW("Unknown shape function type identifier");
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate displacement shape functions                     popp 01/08|
 *----------------------------------------------------------------------*/
bool Mortar::Element::evaluate_shape(const double* xi, Core::LinAlg::SerialDenseVector& val,
    Core::LinAlg::SerialDenseMatrix& deriv, const int valdim, bool dualquad) const
{
  if (!xi) FOUR_C_THROW("evaluate_shape called with xi=nullptr");

  // get node number and node pointers
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("evaluate_shape_lag_mult: Null pointer!");

  // check for boundary nodes
  bool bound = false;
  for (int i = 0; i < num_node(); ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
    if (!mymrtrnode) FOUR_C_THROW("evaluate_shape_lag_mult: Null pointer!");
    bound += mymrtrnode->is_on_bound();
  }

  switch (Element::shape())
  {
    // 2D linear case (2noded line element)
    case Core::FE::CellType::line2:
    {
      if (valdim != 2) FOUR_C_THROW("Inconsistency in evaluate_shape");
      shape_functions(Mortar::Element::lin1D, xi, val, deriv);
      break;
    }
      // 2D quadratic case (3noded line element)
    case Core::FE::CellType::line3:
    {
      if (valdim != 3) FOUR_C_THROW("Inconsistency in evaluate_shape");

      if (dualquad && !bound)
        FOUR_C_THROW(
            "There is no quadratic interpolation for dual shape functions for 2-D problems with "
            "quadratic elements available!");
      else if (dualquad && bound)
        shape_functions(Mortar::Element::quad1D_hierarchical, xi, val, deriv);
      else
        shape_functions(Mortar::Element::quad1D, xi, val, deriv);
      break;
    }
      // 3D linear case (3noded triangular element)
    case Core::FE::CellType::tri3:
    {
      if (valdim != 3) FOUR_C_THROW("Inconsistency in evaluate_shape");
      shape_functions(Mortar::Element::lin2D, xi, val, deriv);
      break;
    }
      // 3D bilinear case (4noded quadrilateral element)
    case Core::FE::CellType::quad4:
    {
      if (valdim != 4) FOUR_C_THROW("Inconsistency in evaluate_shape");
      shape_functions(Mortar::Element::bilin2D, xi, val, deriv);
      break;
    }
      // 3D quadratic case (6noded triangular element)
    case Core::FE::CellType::tri6:
    {
      if (valdim != 6) FOUR_C_THROW("Inconsistency in evaluate_shape");
      if (dualquad && !bound)
        shape_functions(Mortar::Element::quad2D_modified, xi, val, deriv);
      else if (dualquad && bound)
        shape_functions(Mortar::Element::quad2D_hierarchical, xi, val, deriv);
      else
        shape_functions(Mortar::Element::quad2D, xi, val, deriv);
      break;
    }
      // 3D serendipity case (8noded quadrilateral element)
    case Core::FE::CellType::quad8:
    {
      if (valdim != 8) FOUR_C_THROW("Inconsistency in evaluate_shape");
      if (dualquad && !bound)
        shape_functions(Mortar::Element::serendipity2D_modified, xi, val, deriv);
      else if (dualquad && bound)
        shape_functions(Mortar::Element::serendipity2D_hierarchical, xi, val, deriv);
      else
        shape_functions(Mortar::Element::serendipity2D, xi, val, deriv);
      break;
    }
      // 3D biquadratic case (9noded quadrilateral element)
    case Core::FE::CellType::quad9:
    {
      if (valdim != 9) FOUR_C_THROW("Inconsistency in evaluate_shape");
      if (dualquad && !bound)
        shape_functions(Mortar::Element::biquad2D_modified, xi, val, deriv);
      else if (dualquad && bound)
        shape_functions(Mortar::Element::biquad2D_hierarchical, xi, val, deriv);
      else
        shape_functions(Mortar::Element::biquad2D, xi, val, deriv);
      break;
    }

      //==================================================
      //                     NURBS
      //==================================================

      // 1D -- nurbs2
    case Core::FE::CellType::nurbs2:
    {
      if (valdim != 2) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseMatrix auxderiv(1, num_node());
      Core::FE::Nurbs::nurbs_get_1d_funct_deriv(
          val, auxderiv, xi[0], knots()[0], weights, Core::FE::CellType::nurbs2);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < num_node(); ++i) deriv(i, 0) = auxderiv(0, i);

      break;
    }

      // 1D -- nurbs3
    case Core::FE::CellType::nurbs3:
    {
      if (valdim != 3) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseMatrix auxderiv(1, num_node());
      Core::FE::Nurbs::nurbs_get_1d_funct_deriv(
          val, auxderiv, xi[0], knots()[0], weights, Core::FE::CellType::nurbs3);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < num_node(); ++i) deriv(i, 0) = auxderiv(0, i);

      break;
    }

      // ===========================================================
      // 2D -- nurbs4
    case Core::FE::CellType::nurbs4:
    {
      if (valdim != 4) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      Core::LinAlg::SerialDenseMatrix auxderiv(2, num_node());
      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(
          val, auxderiv, uv, knots(), weights, Core::FE::CellType::nurbs4);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 2; ++d)
        for (int i = 0; i < num_node(); ++i) deriv(i, d) = auxderiv(d, i);

      break;
    }

      // 2D -- nurbs9
    case Core::FE::CellType::nurbs9:
    {
      if (valdim != 9) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      Core::LinAlg::SerialDenseMatrix auxderiv(2, num_node());
      Core::FE::Nurbs::nurbs_get_2d_funct_deriv(
          val, auxderiv, uv, knots(), weights, Core::FE::CellType::nurbs9);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (deriv.numCols() != 2 || deriv.numRows() != num_node())
        FOUR_C_THROW("Inconsistency in evaluate_shape");
#endif

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 2; ++d)
        for (int i = 0; i < num_node(); ++i) deriv(i, d) = auxderiv(d, i);


      break;
    }
      // unknown case
    default:
    {
      FOUR_C_THROW("evaluate_shape called for unknown Mortar::Element type");
      break;
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate Lagrange multiplier shape functions              popp 12/07|
 *----------------------------------------------------------------------*/
bool Mortar::Element::evaluate_shape_lag_mult(const Inpar::Mortar::ShapeFcn& lmtype,
    const double* xi, Core::LinAlg::SerialDenseVector& val, Core::LinAlg::SerialDenseMatrix& deriv,
    const int valdim, bool boundtrafo) const
{
  // some methods don't need a Lagrange multiplier interpolation
  if (lmtype == Inpar::Mortar::shape_none) return true;

  if (!xi) FOUR_C_THROW("evaluate_shape_lag_mult called with xi=nullptr");

  // dual LM shape functions or not
  bool dual = false;
  if (lmtype == Inpar::Mortar::shape_dual or lmtype == Inpar::Mortar::shape_petrovgalerkin)
    dual = true;

  // get node number and node pointers
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("evaluate_shape_lag_mult: Null pointer!");

  switch (Element::shape())
  {
    // 2D linear case (2noded line element)
    case Core::FE::CellType::line2:
    {
      if (valdim != 2) FOUR_C_THROW("Inconsistency in evaluate_shape");

      if (dual)
        shape_functions(Mortar::Element::lindual1D, xi, val, deriv);
      else
        shape_functions(Mortar::Element::lin1D, xi, val, deriv);
      break;
    }

      // 2D quadratic case (3noded line element)
    case Core::FE::CellType::line3:
    {
      if (valdim != 3) FOUR_C_THROW("Inconsistency in evaluate_shape");

      if (dual)
        shape_functions(Mortar::Element::quaddual1D, xi, val, deriv);
      else
        shape_functions(Mortar::Element::quad1D, xi, val, deriv);

      break;
    }

      // 3D cases
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::tri6:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      // dual Lagrange multipliers
      if (dual)
      {
        if (shape() == Core::FE::CellType::tri3)
          shape_functions(Mortar::Element::lindual2D, xi, val, deriv);
        else if (shape() == Core::FE::CellType::quad4)
          shape_functions(Mortar::Element::bilindual2D, xi, val, deriv);
        else if (shape() == Core::FE::CellType::tri6)
          shape_functions(Mortar::Element::quaddual2D, xi, val, deriv);
        else if (shape() == Core::FE::CellType::quad8)
          shape_functions(Mortar::Element::serendipitydual2D, xi, val, deriv);
        else
          /*Shape()==quad9*/ shape_functions(Mortar::Element::biquaddual2D, xi, val, deriv);
      }

      // standard Lagrange multipliers
      else
      {
        if (shape() == Core::FE::CellType::tri3)
          shape_functions(Mortar::Element::lin2D, xi, val, deriv);
        else if (shape() == Core::FE::CellType::quad4)
          shape_functions(Mortar::Element::bilin2D, xi, val, deriv);
        else if (shape() == Core::FE::CellType::tri6)
          shape_functions(Mortar::Element::quad2D, xi, val, deriv);
        else if (shape() == Core::FE::CellType::quad8)
          shape_functions(Mortar::Element::serendipity2D, xi, val, deriv);
        else
          /*Shape()==quad9*/ shape_functions(Mortar::Element::biquad2D, xi, val, deriv);
      }

      break;
    }
      //==================================================
      //                     NURBS
      //==================================================

      // 1D -- nurbs2
    case Core::FE::CellType::nurbs2:
    {
      if (dual)
        FOUR_C_THROW("no dual shape functions provided for nurbs!");
      else
        evaluate_shape(xi, val, deriv, valdim);

      break;
    }

      // 1D -- nurbs3
    case Core::FE::CellType::nurbs3:
    {
      if (dual)
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 3;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);
        if (mo_data().dual_shape() == nullptr)
        {
          // compute entries to bi-ortho matrices me/de with Gauss quadrature
          Mortar::ElementIntegrator integrator(Element::shape());

          Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
          Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

          for (int i = 0; i < integrator.n_gp(); ++i)
          {
            double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
            evaluate_shape(gpc, val, deriv, nnodes);
            detg = jacobian(gpc);

            for (int j = 0; j < nnodes; ++j)
            {
              for (int k = 0; k < nnodes; ++k)
              {
                me(j, k) += integrator.weight(i) * val[j] * val[k] * detg;
                de(j, k) += (j == k) * integrator.weight(i) * val[j] * detg;
              }
            }
          }

          // get solution matrix with dual parameters
          Core::LinAlg::invert_and_multiply_by_cholesky<nnodes>(me, de, ae);

          // store coefficient matrix
          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        else
          ae = *(mo_data().dual_shape());

        // evaluate dual shape functions at loc. coord. xi
        // need standard shape functions at xi first
        evaluate_shape(xi, val, deriv, nnodes);

        // check whether this is a 1D or 2D mortar element
        const int dim = 1;
        // evaluate dual shape functions
        Core::LinAlg::SerialDenseVector valtemp(nnodes, true);
        Core::LinAlg::SerialDenseMatrix derivtemp(nnodes, dim, true);
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
        evaluate_shape(xi, val, deriv, valdim);

      break;
    }

      // ===========================================================
      // 2D -- nurbs4
    case Core::FE::CellType::nurbs4:
    {
      if (dual)
        FOUR_C_THROW("no dual shape functions provided for nurbs!");
      else
        evaluate_shape(xi, val, deriv, valdim);

      break;
    }

      // 2D -- nurbs8
    case Core::FE::CellType::nurbs8:
    {
      if (dual)
        FOUR_C_THROW("no dual shape functions provided for nurbs!");
      else
        evaluate_shape(xi, val, deriv, valdim);

      break;
    }

      // 2D -- nurbs9
    case Core::FE::CellType::nurbs9:
    {
      if (dual)
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 9;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes);

        if (mo_data().dual_shape() == nullptr)
        {
          // compute entries to bi-ortho matrices me/de with Gauss quadrature
          Mortar::ElementIntegrator integrator(Element::shape());

          Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
          Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

          for (int i = 0; i < integrator.n_gp(); ++i)
          {
            double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
            evaluate_shape(gpc, val, deriv, nnodes);
            detg = jacobian(gpc);

            for (int j = 0; j < nnodes; ++j)
            {
              for (int k = 0; k < nnodes; ++k)
              {
                me(j, k) += integrator.weight(i) * val[j] * val[k] * detg;
                de(j, k) += (j == k) * integrator.weight(i) * val[j] * detg;
              }
            }
          }

          // get solution matrix with dual parameters
          Core::LinAlg::invert_and_multiply_by_cholesky<nnodes>(me, de, ae);

          // store coefficient matrix
          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        else
          ae = *(mo_data().dual_shape());

        // evaluate dual shape functions at loc. coord. xi
        // need standard shape functions at xi first
        evaluate_shape(xi, val, deriv, nnodes);

        // check whether this is a 1D or 2D mortar element
        const int dim = 2;
        // evaluate dual shape functions
        Core::LinAlg::SerialDenseVector valtemp(nnodes, true);
        Core::LinAlg::SerialDenseMatrix derivtemp(nnodes, dim, true);
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
        evaluate_shape(xi, val, deriv, valdim);

      break;
    }

      // unknown case
    default:
    {
      FOUR_C_THROW("evaluate_shape_lag_mult called for unknown element type");
      break;
    }
  }

  // if no trafo is required return!
  if (!boundtrafo) return true;

  // check if we need trafo
  const int nnodes = num_node();
  bool bound = false;
  for (int i = 0; i < nnodes; ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);

    if (shape() == Core::FE::CellType::line2 or shape() == Core::FE::CellType::line3 or
        shape() == Core::FE::CellType::nurbs2 or shape() == Core::FE::CellType::nurbs3)
    {
      // is on corner or bound?
      if (mymrtrnode->is_on_corneror_bound())
      {
        bound = true;
        break;
      }
    }
    else
    {  // is on corner or edge or bound ?
      if (mymrtrnode->is_on_boundor_ce())
      {
        bound = true;
        break;
      }
    }
  }

  if (!bound) return true;

  //---------------------------------
  // do trafo for bound elements
  Core::LinAlg::SerialDenseMatrix trafo(nnodes, nnodes, true);

  if (mo_data().trafo() == nullptr)
  {
    // 2D case!
    if (shape() == Core::FE::CellType::line2 or shape() == Core::FE::CellType::line3 or
        shape() == Core::FE::CellType::nurbs2 or shape() == Core::FE::CellType::nurbs3)
    {
      // get number of bound nodes
      std::vector<int> ids;
      for (int i = 0; i < nnodes; ++i)
      {
        const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
        if (mymrtrnode->is_on_corneror_bound())
        {
          // get local bound id
          ids.push_back(i);
        }
      }

      int numbound = (int)ids.size();

      // if all bound: error
      if ((nnodes - numbound) < 1e-12) FOUR_C_THROW("all nodes are bound");

      const double factor = 1.0 / (nnodes - numbound);
      // row loop
      for (int i = 0; i < nnodes; ++i)
      {
        const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
        if (!mymrtrnode->is_on_corneror_bound())
        {
          trafo(i, i) = 1.0;
          for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
        }
      }
    }

    // 3D case!
    else if (shape() == Core::FE::CellType::tri6 or shape() == Core::FE::CellType::tri3 or
             shape() == Core::FE::CellType::quad4 or shape() == Core::FE::CellType::quad8 or
             shape() == Core::FE::CellType::quad9 or shape() == Core::FE::CellType::quad4 or
             shape() == Core::FE::CellType::nurbs9)
    {
      // get number of bound nodes
      std::vector<int> ids;
      for (int i = 0; i < nnodes; ++i)
      {
        const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
        if (mymrtrnode->is_on_boundor_ce())
        {
          // get local bound id
          ids.push_back(i);
        }
      }

      int numbound = (int)ids.size();

      // if all bound: error
      if ((nnodes - numbound) < 1e-12)
      {
        std::cout << "numnode= " << nnodes
                  << "shape= " << Core::FE::cell_type_to_string(Element::shape()) << std::endl;
        FOUR_C_THROW("all nodes are bound");
      }

      const double factor = 1.0 / (nnodes - numbound);
      // row loop
      for (int i = 0; i < nnodes; ++i)
      {
        const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
        if (!mymrtrnode->is_on_boundor_ce())
        {
          trafo(i, i) = 1.0;
          for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
        }
      }
    }
    else
      FOUR_C_THROW("unknown element type!");

    mo_data().trafo() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(trafo);
  }
  else
  {
    trafo = *(mo_data().trafo());
  }

  int eledim = -1;
  if (shape() == Core::FE::CellType::tri6 or shape() == Core::FE::CellType::tri3 or
      shape() == Core::FE::CellType::quad4 or shape() == Core::FE::CellType::quad8 or
      shape() == Core::FE::CellType::quad9 or shape() == Core::FE::CellType::nurbs4 or
      shape() == Core::FE::CellType::nurbs9)
  {
    eledim = 2;
  }
  else if (shape() == Core::FE::CellType::line2 or shape() == Core::FE::CellType::line3 or
           shape() == Core::FE::CellType::nurbs2 or shape() == Core::FE::CellType::nurbs3)
  {
    eledim = 1;
  }
  else
  {
    FOUR_C_THROW("unknown shape");
  }

  Core::LinAlg::SerialDenseVector tempval(nnodes, true);
  Core::LinAlg::SerialDenseMatrix tempderiv(nnodes, eledim, true);

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
bool Mortar::Element::evaluate_shape_lag_mult_const(const Inpar::Mortar::ShapeFcn& lmtype,
    const double* xi, Core::LinAlg::SerialDenseVector& val, Core::LinAlg::SerialDenseMatrix& deriv,
    const int valdim) const
{
  Mortar::Utils::evaluate_shape_lm_const(lmtype, xi, val, *this, valdim);
  deriv.putScalar(0.0);

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate Lagrange multiplier shape functions              popp 12/07|
 |  THIS IS A SPECIAL VERSION FOR 3D QUADRATIC MORTAR WITH LIN LM!      |
 *----------------------------------------------------------------------*/
bool Mortar::Element::evaluate_shape_lag_mult_lin(const Inpar::Mortar::ShapeFcn& lmtype,
    const double* xi, Core::LinAlg::SerialDenseVector& val, Core::LinAlg::SerialDenseMatrix& deriv,
    const int valdim) const
{
  // some methods don't need a Lagrange multiplier interpolation
  if (lmtype == Inpar::Mortar::shape_none) return true;

  if (!xi) FOUR_C_THROW("evaluate_shape_lag_mult_lin called with xi=nullptr");
  if (!is_slave()) FOUR_C_THROW("evaluate_shape_lag_mult_lin called for master element");

  // check for feasible element types (line3,tri6, quad8 or quad9)
  if (shape() != Core::FE::CellType::line3 && shape() != Core::FE::CellType::tri6 &&
      shape() != Core::FE::CellType::quad8 && shape() != Core::FE::CellType::quad9)
    FOUR_C_THROW("Linear LM interpolation only for quadratic finite elements");

  // dual shape functions or not
  bool dual = false;
  if (lmtype == Inpar::Mortar::shape_dual || lmtype == Inpar::Mortar::shape_petrovgalerkin)
    dual = true;

  // get node number and node pointers
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("evaluate_shape_lag_mult: Null pointer!");

  // check for boundary nodes
  bool bound = false;
  for (int i = 0; i < num_node(); ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
    if (!mymrtrnode) FOUR_C_THROW("evaluate_shape_lag_mult: Null pointer!");
    bound += mymrtrnode->is_on_bound();
  }

  // all nodes are interior: use unmodified shape functions
  if (!bound)
  {
    FOUR_C_THROW("You should not be here...");
  }

  switch (Element::shape())
  {
    // 2D quadratic case (quadratic line)
    case Core::FE::CellType::line3:
    {
      // the middle node is defined as slave boundary (=master)
      // dual Lagrange multipliers
      if (dual) shape_functions(Mortar::Element::quaddual1D_only_lin, xi, val, deriv);
      // standard Lagrange multipliers
      else
        shape_functions(Mortar::Element::quad1D_only_lin, xi, val, deriv);

      break;
    }

      // 3D quadratic cases (quadratic triangle, biquadratic and serendipity quad)
    case Core::FE::CellType::tri6:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      // the edge nodes are defined as slave boundary (=master)
      // dual Lagrange multipliers
      if (dual)
      {
        // FOUR_C_THROW("Quad->Lin modification of dual LM shape functions not yet implemented");
        if (shape() == Core::FE::CellType::tri6)
          shape_functions(Mortar::Element::quaddual2D_only_lin, xi, val, deriv);
        else if (shape() == Core::FE::CellType::quad8)
          shape_functions(Mortar::Element::serendipitydual2D_only_lin, xi, val, deriv);
        else
          /*Shape()==quad9*/ shape_functions(
              Mortar::Element::biquaddual2D_only_lin, xi, val, deriv);
      }

      // standard Lagrange multipliers
      else
      {
        if (shape() == Core::FE::CellType::tri6)
          shape_functions(Mortar::Element::quad2D_only_lin, xi, val, deriv);
        else if (shape() == Core::FE::CellType::quad8)
          shape_functions(Mortar::Element::serendipity2D_only_lin, xi, val, deriv);
        else
          /*Shape()==quad9*/ shape_functions(Mortar::Element::biquad2D_only_lin, xi, val, deriv);
      }

      break;
    }

      // unknown case
    default:
    {
      FOUR_C_THROW("evaluate_shape_lag_mult called for unknown element type");
      break;
    }
  }

  return true;
}
/*----------------------------------------------------------------------*
 |  1D/2D shape function linearizations repository            popp 05/08|
 *----------------------------------------------------------------------*/
void Mortar::Element::shape_function_linearizations(Mortar::Element::ShapeType shape,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivdual)
{
  switch (shape)
  {
    // in case of consistent dual shape functions we have an entry here
    case Mortar::Element::lindual1D:
    case Mortar::Element::lindual2D:
    {
      if (mo_data().deriv_dual_shape() != nullptr) derivdual = *(mo_data().deriv_dual_shape());
      break;
    }

      // *********************************************************************
      // 2D dual bilinear shape functions (quad4)
      // (used for interpolation of Lagrange multiplier field)
      // (linearization necessary due to adaption for distorted elements !!!)
      // *********************************************************************
    case Mortar::Element::bilindual2D:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        static const int nnodes = 4;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif
        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, 1> val;

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          Utils::mortar_shape_function_2d(val, gpc[0], gpc[1], Mortar::Element::bilin2D);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 3);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (mo_data().dual_shape() == nullptr)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        else
          ae = *(mo_data().dual_shape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
        derivdual = *(mo_data().deriv_dual_shape());
      }
      break;
    }
    // *********************************************************************
    // 1D dual quadratic shape functions (line3/nurbs3)
    // (used for interpolation of Lagrange multiplier field)
    // (linearization necessary due to adaption for distorted elements !!!)
    // *********************************************************************
    case Mortar::Element::quaddual1D:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 3;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif
        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 2, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::SerialDenseVector val(nnodes);
        Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 2, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 2);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (mo_data().dual_shape() == nullptr)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        else
          ae = *(mo_data().dual_shape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
      }
      derivdual = *(mo_data().deriv_dual_shape());

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=0;i<nnodes;++i)
      //  for (int j=0;j<nnodes;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;

      /*
       #ifdef FOUR_C_ENABLE_ASSERTIONS
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       Core::LinAlg::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;
       int thedim=3;
       if (shape==Mortar::Element::quaddual1D) thedim=2;

       for (int dim=0;dim<thedim;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       Core::Nodes::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       mycnode->xspatial()[dim] += delta;

       Core::LinAlg::SerialDenseVector val1(nnodes);
       Core::LinAlg::SerialDenseMatrix deriv1(nnodes,2,true);
       Core::LinAlg::SerialDenseMatrix me1(nnodes,nnodes,true);
       Core::LinAlg::SerialDenseMatrix de1(nnodes,nnodes,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
       evaluate_shape(gpc1, val1, deriv1, nnodes);
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
       Core::LinAlg::SymmetricInverse(me1,nnodes);

       // get solution matrix ae with dual parameters
       Core::LinAlg::SerialDenseMatrix ae1(nnodes,nnodes);
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
       #endif
       */

      break;
    }
    // *********************************************************************
    // 1D dual quadratic shape functions (line3)
    // (used for linear interpolation of Lagrange multiplier field)
    // (linearization necessary due to adaption for distorted elements !!!)
    // *********************************************************************
    case Mortar::Element::quaddual1D_only_lin:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());
      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 3;

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 2, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::SerialDenseVector val(nnodes);
        Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 2, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes, true);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 2);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);
              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }

          double fac = 0.0;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // compute inverse of matrix M_e and matrix A_e
        if (mo_data().dual_shape() == nullptr)
        {
          // how many non-zero nodes
          const int nnodeslin = 2;

          // reduce me to non-zero nodes before inverting
          Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin;
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

          // invert bi-ortho matrix melin
          Core::LinAlg::inverse(melin);

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
          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        // compute inverse of matrix M_e and get matrix A_e
        else
        {
          // invert matrix M_e
          Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

          // get coefficient matrix A_e
          ae = *(mo_data().dual_shape());
        }

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Ae * Me = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);
              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
        derivdual = *(mo_data().deriv_dual_shape());
      }

      break;
    }
    // *********************************************************************
    // 2D dual biquadratic shape functions (quad9)
    // (used for interpolation of Lagrange multiplier field)
    // (linearization necessary due to adaption for distorted elements !!!)
    // *********************************************************************
    case Mortar::Element::biquaddual2D:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 9;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::SerialDenseVector val(nnodes);
        Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 3);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (mo_data().dual_shape() == nullptr)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        else
          ae = *(mo_data().dual_shape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
        derivdual = *(mo_data().deriv_dual_shape());
      }

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=0;i<nnodes;++i)
      //  for (int j=0;j<nnodes;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;

      /*
       #ifdef FOUR_C_ENABLE_ASSERTIONS
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       Core::LinAlg::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;
       int thedim=3;
       if (shape==Mortar::Element::quaddual1D) thedim=2;

       for (int dim=0;dim<thedim;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       Core::Nodes::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       mycnode->xspatial()[dim] += delta;

       Core::LinAlg::SerialDenseVector val1(nnodes);
       Core::LinAlg::SerialDenseMatrix deriv1(nnodes,2,true);
       Core::LinAlg::SerialDenseMatrix me1(nnodes,nnodes,true);
       Core::LinAlg::SerialDenseMatrix de1(nnodes,nnodes,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
       evaluate_shape(gpc1, val1, deriv1, nnodes);
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
       Core::LinAlg::SymmetricInverse(me1,nnodes);

       // get solution matrix ae with dual parameters
       Core::LinAlg::SerialDenseMatrix ae1(nnodes,nnodes);
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
       #endif // #ifdef FOUR_C_ENABLE_ASSERTIONS
       */

      break;
    }
      // *********************************************************************
      // 2D dual quadratic shape functions (tri6)
      // (used for interpolation of Lagrange multiplier field)
      // (linearization necessary due to adaption for distorted elements !!!)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::quaddual2D:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 6;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::SerialDenseVector val(nnodes);
        Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes, true);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 3);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (mo_data().dual_shape() == nullptr)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        else
          ae = *(mo_data().dual_shape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
        derivdual = *(mo_data().deriv_dual_shape());
      }

      // std::cout linearization of Ae
      // std::cout << "Analytical A-derivative of Element: " << Id() << std::endl;
      // for (int i=0;i<nnodes;++i)
      //  for (int j=0;j<nnodes;++j)
      //    for (CI p=derivdual[i][j].begin();p!=derivdual[i][j].end();++p)
      //      std::cout << "A" << i << j << " " << p->first << " " << p->second << std::endl;
      /*
       #ifdef FOUR_C_ENABLE_ASSERTIONS
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       Core::LinAlg::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;
       int thedim=3;

       for (int dim=0;dim<thedim;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       Core::Nodes::Node** mynodes = Nodes();
       Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
       mycnode->xspatial()[dim] += delta;

       Core::LinAlg::SerialDenseVector val1(nnodes);
       Core::LinAlg::SerialDenseMatrix deriv1(nnodes,2,true);
       Core::LinAlg::SerialDenseMatrix me1(nnodes,nnodes,true);
       Core::LinAlg::SerialDenseMatrix de1(nnodes,nnodes,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
       evaluate_shape(gpc1, val1, deriv1, nnodes, true);
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
       Core::LinAlg::SymmetricInverse(me1,nnodes);

       // get solution matrix ae with dual parameters
       Core::LinAlg::SerialDenseMatrix ae1(nnodes,nnodes);
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
       #endif // #ifdef FOUR_C_ENABLE_ASSERTIONS
       */
      break;
    }
      // *********************************************************************
      // 2D dual serendipity shape functions (quad8)
      // (used for interpolation of Lagrange multiplier field)
      // (linearization necessary due to adaption for distorted elements !!!)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::serendipitydual2D:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());

      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 8;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::SerialDenseVector val(nnodes);
        Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes, true);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 3);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);

              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }
          double fac = 0.;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // invert me
        Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

        // get solution matrix ae with dual parameters
        if (mo_data().dual_shape() == nullptr)
        {
          // matrix marix multiplication
          for (int j = 0; j < nnodes; ++j)
            for (int k = 0; k < nnodes; ++k)
              for (int u = 0; u < nnodes; ++u) ae(j, k) += de(j, u) * me(u, k);

          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        else
          ae = *(mo_data().dual_shape());

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);

              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
        derivdual = *(mo_data().deriv_dual_shape());

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
        //      Core::LinAlg::SerialDenseMatrix aeref(ae);
        //      double delta = 1e-8;
        //      int thedim=3;
        //
        //      for (int dim=0;dim<thedim;++dim)
        //      {
        //        for (int node=0;node<nnodes;++node)
        //        {
        //          // apply FD
        //          Core::Nodes::Node** mynodes = Nodes();
        //          Node* mycnode = dynamic_cast<Node*> (mynodes[node]);
        //          mycnode->xspatial()[dim] += delta;
        //
        //          Core::LinAlg::SerialDenseVector val1(nnodes);
        //          Core::LinAlg::SerialDenseMatrix deriv1(nnodes,2,true);
        //          Core::LinAlg::SerialDenseMatrix me1(nnodes,nnodes,true);
        //          Core::LinAlg::SerialDenseMatrix de1(nnodes,nnodes,true);
        //
        //          // build me, de
        //          for (int i=0;i<integrator.nGP();++i)
        //          {
        //            double gpc1[2] = {integrator.Coordinate(i,0), integrator.Coordinate(i,1)};
        //            evaluate_shape(gpc1, val1, deriv1, nnodes, true);
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
        //          Core::LinAlg::SymmetricInverse(me1,nnodes);
        //
        //          // get solution matrix ae with dual parameters
        //          Core::LinAlg::SerialDenseMatrix ae1(nnodes,nnodes);
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
    case Mortar::Element::quaddual1D_edge0:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = num_node();
      typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 1);
      Core::LinAlg::SerialDenseVector vallin(nnodes - 1);
      Core::LinAlg::SerialDenseMatrix derivlin(nnodes - 1, 1);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      Mortar::ElementIntegrator integrator(Element::shape());

      Core::LinAlg::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      Core::LinAlg::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      // two-dim arrays of maps for linearization of me/de
      std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivme(
          nnodes, std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, 3 * nnodes));
      std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivde(
          nnodes, std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, 3 * nnodes));

      for (int i = 0; i < integrator.n_gp(); ++i)
      {
        double gpc[2] = {integrator.coordinate(i, 0), 0.0};
        shape_functions(Mortar::Element::quad1D, gpc, valquad, derivquad);
        shape_functions(Mortar::Element::dual1D_base_for_edge0, gpc, vallin, derivlin);
        detg = jacobian(gpc);

        // directional derivative of Jacobian
        Core::Gen::Pairedvector<int, double> testmap(nnodes * 2);
        deriv_jacobian(gpc, testmap);

        // loop over all entries of me/de
        for (int j = 1; j < nnodes; ++j)
          for (int k = 1; k < nnodes; ++k)
          {
            double facme = integrator.weight(i) * vallin[j - 1] * valquad[k];
            double facde = (j == k) * integrator.weight(i) * valquad[k];

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
      Core::LinAlg::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      Core::LinAlg::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      Core::LinAlg::multiply(ae, de, me);

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
       #ifdef FOUR_C_ENABLE_ASSERTIONS
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       Core::LinAlg::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;

       for (int dim=0;dim<2;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       coord(dim,node)+=delta;

       // empty shape function vals + derivs
       Core::LinAlg::SerialDenseVector valquad1(nnodes);
       Core::LinAlg::SerialDenseMatrix derivquad1(nnodes,1);
       Core::LinAlg::SerialDenseVector vallin1(nnodes-1);
       Core::LinAlg::SerialDenseMatrix derivlin1(nnodes-1,1);
       //Core::LinAlg::SerialDenseVector valtemp1(nnodes);
       //Core::LinAlg::SerialDenseMatrix derivtemp1(nnodes,1);
       Core::LinAlg::SerialDenseMatrix me1(nnodes-1,nnodes-1,true);
       Core::LinAlg::SerialDenseMatrix de1(nnodes-1,nnodes-1,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i), 0.0};
       ShapeFunctions(Mortar::Element::quad1D,gpc1,valquad1,derivquad1);
       ShapeFunctions(Mortar::Element::dual1D_base_for_edge0,gpc1,vallin1,derivlin1);
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
       Core::LinAlg::SerialDenseMatrix meold(nnodes-1,nnodes-1);
       meold=me1;
       me1(0,0) =  1/detme1*meold(1,1);
       me1(0,1) = -1/detme1*meold(0,1);
       me1(1,0) = -1/detme1*meold(1,0);
       me1(1,1) =  1/detme1*meold(0,0);

       // get solution matrix ae with dual parameters
       Core::LinAlg::SerialDenseMatrix ae1(nnodes-1,nnodes-1);
       ae1.Multiply('N','N',1.0,de1,me1,0.0);

       Core::Nodes::Node** mynodes = Nodes();
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
       #endif // #ifdef FOUR_C_ENABLE_ASSERTIONS
       */
      break;
    }
      // *********************************************************************
      // 1D modified dual shape functions (linear)
      // (used for interpolation of Lagrange mult. field near boundaries)
      // (linearization necessary due to adaption for distorted elements !!!)
      // *********************************************************************
    case Mortar::Element::quaddual1D_edge1:
    {
      // establish fundamental data
      double detg = 0.0;
      const int nnodes = num_node();
      typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;

      // empty shape function vals + derivs
      Core::LinAlg::SerialDenseVector valquad(nnodes);
      Core::LinAlg::SerialDenseMatrix derivquad(nnodes, 1);
      Core::LinAlg::SerialDenseVector vallin(nnodes - 1);
      Core::LinAlg::SerialDenseMatrix derivlin(nnodes - 1, 1);

      // compute entries to bi-ortho matrices me/de with Gauss quadrature
      Mortar::ElementIntegrator integrator(Element::shape());

      Core::LinAlg::SerialDenseMatrix me(nnodes - 1, nnodes - 1, true);
      Core::LinAlg::SerialDenseMatrix de(nnodes - 1, nnodes - 1, true);

      // two-dim arrays of maps for linearization of me/de
      std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivme(
          nnodes, std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, 2 * nnodes));
      std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivde(
          nnodes, std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, 2 * nnodes));

      for (int i = 0; i < integrator.n_gp(); ++i)
      {
        double gpc[2] = {integrator.coordinate(i, 0), 0.0};
        shape_functions(Mortar::Element::quad1D, gpc, valquad, derivquad);
        shape_functions(Mortar::Element::dual1D_base_for_edge1, gpc, vallin, derivlin);
        detg = jacobian(gpc);

        // directional derivative of Jacobian
        Core::Gen::Pairedvector<int, double> testmap(nnodes * 2);
        deriv_jacobian(gpc, testmap);

        // loop over all entries of me/de
        for (int j = 0; j < nnodes - 1; ++j)
          for (int k = 0; k < nnodes - 1; ++k)
          {
            double facme = integrator.weight(i) * vallin[j] * valquad[2 * k];
            double facde = (j == k) * integrator.weight(i) * valquad[2 * k];

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
      Core::LinAlg::SerialDenseMatrix meold(nnodes - 1, nnodes - 1);
      meold = me;
      me(0, 0) = detmeinv * meold(1, 1);
      me(0, 1) = -detmeinv * meold(0, 1);
      me(1, 0) = -detmeinv * meold(1, 0);
      me(1, 1) = detmeinv * meold(0, 0);

      // get solution matrix with dual parameters
      Core::LinAlg::SerialDenseMatrix ae(nnodes - 1, nnodes - 1);
      Core::LinAlg::multiply(ae, de, me);

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
       #ifdef FOUR_C_ENABLE_ASSERTIONS
       // *******************************************************************
       // FINITE DIFFERENCE check of Lin(Ae)
       // *******************************************************************

       std::cout << "FD Check for A-derivative of Element: " << Id() << std::endl;
       Core::LinAlg::SerialDenseMatrix aeref(ae);
       double delta = 1e-8;

       for (int dim=0;dim<2;++dim)
       {
       for (int node=0;node<nnodes;++node)
       {
       // apply FD
       coord(dim,node)+=delta;

       // empty shape function vals + derivs
       Core::LinAlg::SerialDenseVector valquad1(nnodes);
       Core::LinAlg::SerialDenseMatrix derivquad1(nnodes,1);
       Core::LinAlg::SerialDenseVector vallin1(nnodes-1);
       Core::LinAlg::SerialDenseMatrix derivlin1(nnodes-1,1);
       //Core::LinAlg::SerialDenseVector valtemp1(nnodes);
       //Core::LinAlg::SerialDenseMatrix derivtemp1(nnodes,1);
       Core::LinAlg::SerialDenseMatrix me1(nnodes-1,nnodes-1,true);
       Core::LinAlg::SerialDenseMatrix de1(nnodes-1,nnodes-1,true);

       // build me, de
       for (int i=0;i<integrator.nGP();++i)
       {
       double gpc1[2] = {integrator.Coordinate(i), 0.0};
       ShapeFunctions(Mortar::Element::quad1D,gpc1,valquad1,derivquad1);
       ShapeFunctions(Mortar::Element::dual1D_base_for_edge1,gpc1,vallin1,derivlin1);
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
       Core::LinAlg::SerialDenseMatrix meold(nnodes-1,nnodes-1);
       meold=me1;
       me1(0,0) =  1/detme1*meold(1,1);
       me1(0,1) = -1/detme1*meold(0,1);
       me1(1,0) = -1/detme1*meold(1,0);
       me1(1,1) =  1/detme1*meold(0,0);

       // get solution matrix ae with dual parameters
       Core::LinAlg::SerialDenseMatrix ae1(nnodes-1,nnodes-1);
       ae1.Multiply('N','N',1.0,de1,me1,0.0);

       Core::Nodes::Node** mynodes = Nodes();
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
       #endif // #ifdef FOUR_C_ENABLE_ASSERTIONS
       */
      break;
    }
      //***********************************************************************
      // 2D dual quadratic shape functions (tri6)
      // (used for LINEAR interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      //***********************************************************************
    case Mortar::Element::quaddual2D_only_lin:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());
      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 6;

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::SerialDenseVector val(nnodes);
        Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes, true);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 3);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);
              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }

          double fac = 0.0;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // compute inverse of matrix M_e and matrix A_e
        if (mo_data().dual_shape() == nullptr)
        {
          // how many non-zero nodes
          const int nnodeslin = 3;

          // reduce me to non-zero nodes before inverting
          Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin;
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

          // invert bi-ortho matrix melin
          Core::LinAlg::inverse(melin);

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
          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        // compute inverse of matrix M_e and get matrix A_e
        else
        {
          // invert matrix M_e
          Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

          // get coefficient matrix A_e
          ae = *(mo_data().dual_shape());
        }

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Ae * Me = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);
              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
        derivdual = *(mo_data().deriv_dual_shape());
      }
      break;
    }

      // *********************************************************************
      // 2D dual serendipity shape functions (quad8)
      // (used for LINEAR interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::serendipitydual2D_only_lin:
    {
      if (mo_data().deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().deriv_dual_shape());
      else
      {
        // establish fundamental data
        double detg = 0.0;
        const int nnodes = 8;

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (nnodes != num_node())
          FOUR_C_THROW(
              "Mortar::Element shape function for LM incompatible with number of element nodes!");
#endif

        std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>> derivae =
            std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
                nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));

        typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;
        Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);

        // prepare computation with Gauss quadrature
        Mortar::ElementIntegrator integrator(Element::shape());
        Core::LinAlg::SerialDenseVector val(nnodes);
        Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
        Core::LinAlg::Matrix<nnodes, nnodes> me(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nnodes, nnodes> de(Core::LinAlg::Initialization::zero);

        // two-dim arrays of maps for linearization of me/de
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> derivde_me(
            nnodes * 3, 0, Core::LinAlg::SerialDenseMatrix(nnodes + 1, nnodes));

        // build me, de, derivme, derivde
        for (int i = 0; i < integrator.n_gp(); ++i)
        {
          double gpc[2] = {integrator.coordinate(i, 0), integrator.coordinate(i, 1)};
          evaluate_shape(gpc, val, deriv, nnodes, true);
          detg = jacobian(gpc);

          // directional derivative of Jacobian
          Core::Gen::Pairedvector<int, double> testmap(nnodes * 3);
          deriv_jacobian(gpc, testmap);

          // loop over all entries of me/de
          for (int j = 0; j < nnodes; ++j)
          {
            for (int k = 0; k < nnodes; ++k)
            {
              double facme = integrator.weight(i) * val(j) * val(k);
              double facde = (j == k) * integrator.weight(i) * val(j);
              me(j, k) += facme * detg;
              de(j, k) += facde * detg;
            }
          }

          double fac = 0.0;
          // loop over all directional derivatives
          for (CI p = testmap.begin(); p != testmap.end(); ++p)
          {
            Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
            const double& ps = p->second;
            for (int j = 0; j < nnodes; ++j)
            {
              fac = integrator.weight(i) * val(j) * ps;
              dtmp(nnodes, j) += fac;
              for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * val(k);
            }
          }
        }

        // compute inverse of matrix M_e and matrix A_e
        if (mo_data().dual_shape() == nullptr)
        {
          // how many non-zero nodes
          const int nnodeslin = 4;

          // reduce me to non-zero nodes before inverting
          Core::LinAlg::Matrix<nnodeslin, nnodeslin> melin;
          for (int j = 0; j < nnodeslin; ++j)
            for (int k = 0; k < nnodeslin; ++k) melin(j, k) = me(j, k);

          // invert bi-ortho matrix melin
          Core::LinAlg::inverse(melin);

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
          mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);
        }
        // compute inverse of matrix M_e and get matrix A_e
        else
        {
          // invert matrix M_e
          Core::LinAlg::symmetric_positive_definite_inverse<nnodes>(me);

          // get coefficient matrix A_e
          ae = *(mo_data().dual_shape());
        }

        // build linearization of ae and store in derivdual
        // (this is done according to a quite complex formula, which
        // we get from the linearization of the biorthogonality condition:
        // Lin (Ae * Me = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
        typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
        for (_CIM p = derivde_me.begin(); p != derivde_me.end(); ++p)
        {
          Core::LinAlg::SerialDenseMatrix& dtmp = derivde_me[p->first];
          Core::LinAlg::SerialDenseMatrix& pt = (*derivae)[p->first];
          for (int i = 0; i < nnodes; ++i)
            for (int j = 0; j < nnodes; ++j)
            {
              pt(i, j) += me(i, j) * dtmp(nnodes, i);
              for (int k = 0; k < nnodes; ++k)
                for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * me(l, j) * dtmp(l, k);
            }
        }
        mo_data().deriv_dual_shape() = derivae;
        derivdual = *(mo_data().deriv_dual_shape());
      }
      break;
    }

      // *********************************************************************
      // 2D dual biquadratic shape functions (quad9)
      // (used for LINEAR interpolation of Lagrange multiplier field)
      // (including adaption process for distorted elements)
      // (including modification of displacement shape functions)
      // *********************************************************************
    case Mortar::Element::biquaddual2D_only_lin:
    {
      FOUR_C_THROW("biquaddual2D_only_lin not available!");
      break;
    }
      // *********************************************************************
      // Unknown shape function type
      // *********************************************************************
    default:
    {
      FOUR_C_THROW("Unknown shape function type identifier");
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate 2nd derivative of shape functions                popp 05/08|
 *----------------------------------------------------------------------*/
bool Mortar::Element::evaluate2nd_deriv_shape(
    const double* xi, Core::LinAlg::SerialDenseMatrix& secderiv, const int& valdim) const
{
  if (!xi) FOUR_C_THROW("evaluate2nd_deriv_shape called with xi=nullptr");

  //**********************************************************************
  // IMPORTANT NOTE: In 3D the ordering of the 2nd derivatives is:
  // 1) dxi,dxi 2) deta,deta 3) dxi,deta
  //**********************************************************************

  switch (Element::shape())
  {
    // 2D linear case (2noded line element)
    case Core::FE::CellType::line2:
    {
      secderiv(0, 0) = 0.0;
      secderiv(1, 0) = 0.0;
      break;
    }
      // 2D quadratic case (3noded line element)
    case Core::FE::CellType::line3:
    {
      secderiv(0, 0) = 1.0;
      secderiv(1, 0) = 1.0;
      secderiv(2, 0) = -2.0;
      break;
    }
      // 3D linear case (3noded triangular element)
    case Core::FE::CellType::tri3:
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
    case Core::FE::CellType::quad4:
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
    case Core::FE::CellType::tri6:
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
    case Core::FE::CellType::quad8:
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
    case Core::FE::CellType::quad9:
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
    case Core::FE::CellType::nurbs2:
    {
      if (valdim != 2) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseVector auxval(num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv(1, num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv2(1, num_node());

      Core::FE::Nurbs::nurbs_get_1d_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, xi[0], knots()[0], weights, Core::FE::CellType::nurbs2);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < num_node(); ++i) secderiv(i, 0) = auxderiv2(0, i);

      break;
    }

      // 1D -- nurbs3
    case Core::FE::CellType::nurbs3:
    {
      if (valdim != 3) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(3);
      for (int inode = 0; inode < 3; ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseVector auxval(3);
      Core::LinAlg::SerialDenseMatrix auxderiv(1, 3);
      Core::LinAlg::SerialDenseMatrix auxderiv2(1, 3);

      Core::FE::Nurbs::nurbs_get_1d_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, xi[0], knots()[0], weights, Core::FE::CellType::nurbs3);

      // copy entries for to be conform with the mortar code!
      for (int i = 0; i < num_node(); ++i) secderiv(i, 0) = auxderiv2(0, i);

      break;
    }

      // ===========================================================
      // 2D -- nurbs4
    case Core::FE::CellType::nurbs4:
    {
      if (valdim != 4) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      Core::LinAlg::SerialDenseVector auxval(num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv(2, num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv2(3, num_node());

      Core::FE::Nurbs::nurbs_get_2d_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, uv, knots(), weights, Core::FE::CellType::nurbs4);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 3; ++d)
        for (int i = 0; i < num_node(); ++i) secderiv(i, d) = auxderiv2(d, i);

      break;
    }

      // 2D -- nurbs8
    case Core::FE::CellType::nurbs8:
    {
      if (valdim != 8) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      Core::LinAlg::SerialDenseVector auxval(num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv(2, num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv2(3, num_node());

      Core::FE::Nurbs::nurbs_get_2d_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, uv, knots(), weights, Core::FE::CellType::nurbs8);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 3; ++d)
        for (int i = 0; i < num_node(); ++i) secderiv(i, d) = auxderiv2(d, i);

      break;
    }

      // 2D -- nurbs9
    case Core::FE::CellType::nurbs9:
    {
      if (valdim != 9) FOUR_C_THROW("Inconsistency in evaluate_shape");

      Core::LinAlg::SerialDenseVector weights(num_node());
      for (int inode = 0; inode < num_node(); ++inode)
        weights(inode) = dynamic_cast<const Mortar::Node*>(nodes()[inode])->nurbs_w();

      Core::LinAlg::SerialDenseVector uv(2);
      uv(0) = xi[0];
      uv(1) = xi[1];

      Core::LinAlg::SerialDenseVector auxval(num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv(2, num_node());
      Core::LinAlg::SerialDenseMatrix auxderiv2(3, num_node());

      Core::FE::Nurbs::nurbs_get_2d_funct_deriv_deriv2(
          auxval, auxderiv, auxderiv2, uv, knots(), weights, Core::FE::CellType::nurbs9);

      // copy entries for to be conform with the mortar code!
      for (int d = 0; d < 3; ++d)
        for (int i = 0; i < num_node(); ++i) secderiv(i, d) = auxderiv2(d, i);

      break;
    }
      // unknown case
    default:
      FOUR_C_THROW("evaluate2nd_deriv_shape called for unknown element type");
      break;
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Compute directional derivative of dual shape functions    popp 05/08|
 *----------------------------------------------------------------------*/
bool Mortar::Element::deriv_shape_dual(
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivdual)
{
  // get node number and node pointers
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("DerivShapeDual: Null pointer!");

  switch (Element::shape())
  {
    // 2D linear case (2noded line element)
    case Core::FE::CellType::line2:
    {
      if (mo_data().get_deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().get_deriv_dual_shape());
      else
        derivdual.resize(0);

      break;
    }
      // 3D linear case (3noded triangular element)
    case Core::FE::CellType::tri3:
    {
      if (mo_data().get_deriv_dual_shape() != nullptr)
        derivdual = *(mo_data().get_deriv_dual_shape());
      else
        derivdual.resize(0);
      break;
    }

      // 2D quadratic case (3noded line element)
    case Core::FE::CellType::line3:
    {
      // check for middle "bound" node
      const Node* mycnode2 = dynamic_cast<const Node*>(mynodes[2]);
      if (!mycnode2) FOUR_C_THROW("DerivShapeDual: Null pointer!");
      bool isonbound2 = mycnode2->is_on_bound();

      // locally linear Lagrange multipliers
      if (isonbound2)
        shape_function_linearizations(Mortar::Element::quaddual1D_only_lin, derivdual);
      // use unmodified dual shape functions
      else
        shape_function_linearizations(Mortar::Element::quaddual1D, derivdual);

      break;
    }

      // all other 3D cases
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::tri6:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      if (shape() == Core::FE::CellType::quad4)
        shape_function_linearizations(Mortar::Element::bilindual2D, derivdual);
      else if (shape() == Core::FE::CellType::tri6)
        shape_function_linearizations(Mortar::Element::quaddual2D, derivdual);
      else if (shape() == Core::FE::CellType::quad8)
        shape_function_linearizations(Mortar::Element::serendipitydual2D, derivdual);
      else
        /*Shape()==quad9*/ shape_function_linearizations(Mortar::Element::biquaddual2D, derivdual);

      break;
    }

    //==================================================
    //                     NURBS
    //==================================================
    case Core::FE::CellType::nurbs3:
    {
      shape_function_linearizations(Mortar::Element::quaddual1D, derivdual);
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      shape_function_linearizations(Mortar::Element::biquaddual2D, derivdual);
      break;
    }
      // unknown case
    default:
    {
      FOUR_C_THROW("DerivShapeDual called for unknown element type");
      break;
    }
  }

  // check if we need trafo
  const int nnodes = num_node();
  bool bound = false;
  for (int i = 0; i < nnodes; ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
    if (mymrtrnode->is_on_boundor_ce())
    {
      bound = true;
      break;
    }
  }

  if (!bound) return true;

  //---------------------------------
  // do trafo for bound elements
  Core::LinAlg::SerialDenseMatrix trafo(nnodes, nnodes, true);

  // 2D case!
  if (shape() == Core::FE::CellType::line2 or shape() == Core::FE::CellType::line3 or
      shape() == Core::FE::CellType::nurbs2 or shape() == Core::FE::CellType::nurbs3)
  {
    // get number of bound nodes
    std::vector<int> ids;
    for (int i = 0; i < nnodes; ++i)
    {
      const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
      if (mymrtrnode->is_on_corner())
      {
        // get local bound id
        ids.push_back(i);
      }
    }

    int numbound = (int)ids.size();

    // if all bound: error
    if ((nnodes - numbound) < 1e-12) FOUR_C_THROW("all nodes are bound");

    const double factor = 1.0 / (nnodes - numbound);
    // row loop
    for (int i = 0; i < nnodes; ++i)
    {
      const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
      if (!mymrtrnode->is_on_corner())
      {
        trafo(i, i) = 1.0;
        for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
      }
    }
  }

  // 3D case!
  else if (shape() == Core::FE::CellType::tri6 or shape() == Core::FE::CellType::tri3 or
           shape() == Core::FE::CellType::quad4 or shape() == Core::FE::CellType::quad8 or
           shape() == Core::FE::CellType::quad9 or shape() == Core::FE::CellType::nurbs4 or
           shape() == Core::FE::CellType::nurbs9)
  {
    // get number of bound nodes
    std::vector<int> ids;
    for (int i = 0; i < nnodes; ++i)
    {
      const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
      if (mymrtrnode->is_on_boundor_ce())
      {
        // get local bound id
        ids.push_back(i);
      }
    }

    int numbound = (int)ids.size();

    // if all bound: error
    if ((nnodes - numbound) < 1e-12) FOUR_C_THROW("all nodes are bound");

    const double factor = 1.0 / (nnodes - numbound);
    // row loop
    for (int i = 0; i < nnodes; ++i)
    {
      const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
      if (!mymrtrnode->is_on_boundor_ce())
      {
        trafo(i, i) = 1.0;
        for (int j = 0; j < (int)ids.size(); ++j) trafo(i, ids[j]) = factor;
      }
    }
  }
  else
    FOUR_C_THROW("unknown element type!");



  // do trafo
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix> dummy(
      nnodes * nnodes * 3 * 10, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes, true));

  typedef Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator _CIM;
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

FOUR_C_NAMESPACE_CLOSE
