/*----------------------------------------------------------------------*/
/*! \file

\brief A set of integration routines for faces on elements

\level 0
*----------------------------------------------------------------------*/

#include "baci_discretization_fem_general_utils_boundary_integration.H"

#include "baci_discretization_fem_general_utils_integration.H"


/* compute kovariant metric tensor G for surface element     gammi 04/07

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
void CORE::DRT::UTILS::ComputeMetricTensorForSurface(const CORE::LINALG::SerialDenseMatrix& xyze,
    const CORE::LINALG::SerialDenseMatrix& deriv, CORE::LINALG::SerialDenseMatrix& metrictensor,
    double* sqrtdetg)
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
  |                                 +-            -+
  |                                 | dx   dy   dz |
  |                                 | --   --   -- |
  |                                 | dr   dr   dr |
  |     yields           dxyzdrs =  |              |
  |                                 | dx   dy   dz |
  |                                 | --   --   -- |
  |                                 | ds   ds   ds |
  |                                 +-            -+
  |
  */
  CORE::LINALG::SerialDenseMatrix dxyzdrs(2, 3);

  dxyzdrs.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1.0, deriv, xyze, 0.0);

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
  metrictensor.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1.0, dxyzdrs, dxyzdrs, 0.0);

  /*
                            +--------------+
                           /               |
             sqrtdetg =   /  g11*g22-g12^2
                        \/
  */

  sqrtdetg[0] =
      sqrt(metrictensor(0, 0) * metrictensor(1, 1) - metrictensor(0, 1) * metrictensor(1, 0));

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int parent_ele_dim>
CORE::LINALG::Matrix<parent_ele_dim, 1> CORE::DRT::UTILS::CalculateParentGPFromFaceElementData(
    const double* faceele_xi, const ::DRT::FaceElement* faceele)
{
  static CORE::LINALG::Matrix<parent_ele_dim - 1, 1> xi;
  for (int i = 0; i < parent_ele_dim - 1; ++i)
  {
    xi(i) = faceele_xi[i];
  }
  const double dummy_gp_wgt(0.0);
  CORE::DRT::UTILS::CollectedGaussPoints intpoints;
  intpoints.Append(xi, dummy_gp_wgt);

  // get coordinates of gauss point w.r.t. local parent coordinate system
  CORE::LINALG::SerialDenseMatrix pqxg(1, parent_ele_dim);
  CORE::LINALG::Matrix<parent_ele_dim, parent_ele_dim> derivtrafo(true);
  CORE::DRT::UTILS::BoundaryGPToParentGP<parent_ele_dim>(pqxg, derivtrafo, intpoints,
      faceele->ParentElement()->Shape(), faceele->Shape(), faceele->FaceParentNumber());

  CORE::LINALG::Matrix<parent_ele_dim, 1> xi_parent(true);
  for (auto i = 0; i < parent_ele_dim; ++i)
  {
    xi_parent(i) = pqxg(0, i);
  }

  return xi_parent;
}

/*-----------------------------------------------------------------

\brief Transform Gausspoints on line element to 2d space of
       parent element (required for integrations of parent-element
       shape functions over boundary elements, for example
       in weak dirichlet boundary conditions).

  -----------------------------------------------------------------*/
template <class V, class W, typename IntegrationPoints>
void CORE::DRT::UTILS::LineGPToParentGP(V& pqxg, W& derivtrafo, const IntegrationPoints& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int lineid)
{
  // resize output array
  // pqxg.Shape(pqxg.numRows(),2);
  // derivtrafo.Shape(2,2);


  if ((distype == ::DRT::Element::line2 && pdistype == ::DRT::Element::quad4) or
      (distype == ::DRT::Element::line3 && pdistype == ::DRT::Element::quad8) or
      (distype == ::DRT::Element::line3 && pdistype == ::DRT::Element::quad9))
  {
    switch (lineid)
    {
      case 0:
      {
        /*                s|
                           |

                 3                   2
                  +-----------------+
                  |                 |
                  |                 |
                  |                 |
                  |        |        |             r
                  |        +--      |         -----
                  |                 |
                  |                 |
                  |                 |
                  |                 |
                  +-----------*-----+
                 0                   1
                        -->|gp|<--               */


        // s=-1
        /*

                  parent                line

                               r                     r
                +---+---+  -----      +---+---+ ------
               0   1   2             0   1   2

        */
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = intpoints.Point(iquad)[0];
          pqxg(iquad, 1) = -1.0;
        }
        derivtrafo(0, 0) = 1.0;
        derivtrafo(1, 1) = -1.0;
        break;
      }
      case 1:
      {
        /*                s|
                           |

                 3                   2
                  +-----------------+
                  |                 | |
                  |                 | v
                  |                 *---
                  |        |        | gp          r
                  |        +--      |---      -----
                  |                 | ^
                  |                 | |
                  |                 |
                  |                 |
                  +-----------------+
                 0                   1
                                                 */

        // r=+1
        /*
                  parent               surface

                   s|                        r|
                    |                         |
                        +                     +
                       8|                    2|
                        +                     +
                       5|                    1|
                        +                     +
                       2                     0
        */
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = 1.0;
          pqxg(iquad, 1) = intpoints.Point(iquad)[0];
        }
        derivtrafo(0, 1) = 1.0;
        derivtrafo(1, 0) = 1.0;
        break;
      }
      case 2:
      {
        /*                s|
                           |

                 3   -->|gp|<--
                  +-----*-----------+
                  |                 |
                  |                 |
                  |                 |
                  |        |        |             r
                  |        +--      |         -----
                  |                 |
                  |                 |
                  |                 |
                  |                 |
                  +-----------------+
                 0                   1
                                                 */

        // s=+1
        /*

                  parent                line

                               r                           r
                +---+---+  -----             +---+---+ -----
               6   7   8                    0   1   2

        */

        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = -intpoints.Point(iquad)[0];
          pqxg(iquad, 1) = 1.0;
        }
        derivtrafo(0, 0) = -1.0;
        derivtrafo(1, 1) = 1.0;
        break;
      }
      case 3:
      {
        /*                s|
                           |

                 3
                  +-----*-----------+
                  |                 |
                  |                 |
                | |                 |
                v |        |        |             r
               ---|        +--      |         -----
                gp|                 |
               ---*                 |
                ^ |                 |
                | |                 |
                  +-----------------+
                 0                   1
                                                 */

        // r=-1
        /*
                  parent               surface

                   s|                        r|
                    |                         |
                    +                         +
                   6|                        2|
                    +                         +
                   3|                        1|
                    +                         +
                    0                         0
        */
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = -1.0;
          pqxg(iquad, 1) = -intpoints.Point(iquad)[0];
        }
        derivtrafo(0, 1) = -1.0;
        derivtrafo(1, 0) = -1.0;
        break;
      }
      default:
        dserror("invalid number of lines, unable to determine intpoint in parent");
        break;
    }
  }
  else if (distype == ::DRT::Element::nurbs3 && pdistype == ::DRT::Element::nurbs9)
  {
    switch (lineid)
    {
      case 0:
      {
        // s=-1
        /*

                  parent                line

                               r                     r
                +---+---+  -----      +---+---+ ------
               0   1   2             0   1   2

        */
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = intpoints.Point(iquad)[0];
          pqxg(iquad, 1) = -1.0;
        }
        derivtrafo(0, 0) = 1.0;
        derivtrafo(1, 1) = -1.0;
        break;
      }
      case 1:
      {
        // r=+1
        /*
                  parent               surface

                   s|                    r|
                    |                     |
                    +                     +
                   8|                    2|
                    +                     +
                   5|                    1|
                    +                     +
                   2                     0
        */
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = 1.0;
          pqxg(iquad, 1) = intpoints.Point(iquad)[0];
        }
        derivtrafo(0, 1) = 1.0;
        derivtrafo(1, 0) = 1.0;
        break;
      }
      case 2:
      {
        // s=+1
        /*

                  parent                line

                               r                           r
                +---+---+  -----             +---+---+ -----
               6   7   8                    0   1   2

        */

        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = intpoints.Point(iquad)[0];
          pqxg(iquad, 1) = 1.0;
        }
        derivtrafo(0, 0) = 1.0;
        derivtrafo(1, 1) = 1.0;
        break;
      }
      case 3:
      {
        // r=-1
        /*
                  parent               surface

                s|                           r|
                 |                            |
                 +                            +
                6|                           2|
                 +                            +
                3|                           1|
                 +                            +
                0                            0
        */
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = -1.0;
          pqxg(iquad, 1) = intpoints.Point(iquad)[0];
        }
        derivtrafo(1, 0) = -1.0;
        derivtrafo(0, 1) = 1.0;
        break;
      }
      default:
        dserror("invalid number of lines, unable to determine intpoint in parent");
        break;
    }
  }
  else if ((distype == ::DRT::Element::line2 && pdistype == ::DRT::Element::tri3) or
           (distype == ::DRT::Element::line3 && pdistype == ::DRT::Element::tri6))
  {
    switch (lineid)
    {
      case 0:
      {
        // s=0
        /*
                    parent               line

                 s|
                  |

                2 +
                  |
                  |
                  |             r
                  +-------+  -----      +-------+   -----r
                 0        1             0       1
         */

        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = 0.5 + 0.5 * intpoints.Point(iquad)[0];
          pqxg(iquad, 1) = 0.0;
        }
        break;
      }
      case 1:
      {
        // 1-r-s=0
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = 0.5 - 0.5 * intpoints.Point(iquad)[0];
          pqxg(iquad, 1) = 0.5 + 0.5 * intpoints.Point(iquad)[0];
        }
        break;
      }
      case 2:
      {
        // r=0
        for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
        {
          pqxg(iquad, 0) = 0.0;
          pqxg(iquad, 1) = 0.5 - 0.5 * intpoints.Point(iquad)[0];
        }
        break;
      }
      default:
        dserror("invalid number of surfaces, unable to determine intpoint in parent");
        break;
    }
  }
  else
  {
    dserror(
        "only line2/quad4, line3/quad8, line3/quad9, nurbs3/nurbs9, line2/tri3 and line3/tri6 "
        "mappings of surface gausspoint to parent element implemented up to now\n");
  }

  return;
}

/*-----------------------------------------------------------------
\brief Template version of transformation of Gausspoints on boundary element to space of
       parent element
 ----------------------------------------------------------------------------------*/

//! specialization for 3D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<3>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::SerialDenseMatrix& derivtrafo,
    const ::CORE::DRT::UTILS::IntPointsAndWeights<2>& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.IP().nquad, 3);
  if (derivtrafo.numRows() != 3 || derivtrafo.numCols() != 3) derivtrafo.shape(3, 3);

  DRT::UTILS::SurfaceGPToParentGP(pqxg, derivtrafo, intpoints.IP(), pdistype, distype, surfaceid);
  return;
}

//! specialization for 2D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<2>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::SerialDenseMatrix& derivtrafo,
    const ::CORE::DRT::UTILS::IntPointsAndWeights<1>& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.IP().nquad, 2);
  if (derivtrafo.numRows() != 2 || derivtrafo.numCols() != 2) derivtrafo.shape(2, 2);

  DRT::UTILS::LineGPToParentGP(pqxg, derivtrafo, intpoints.IP(), pdistype, distype, surfaceid);
  return;
}

//! specialization for 3D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<3>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::Matrix<3, 3>& derivtrafo,
    const ::CORE::DRT::UTILS::IntPointsAndWeights<2>& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.IP().nquad, 3);
  derivtrafo.Clear();

  DRT::UTILS::SurfaceGPToParentGP(pqxg, derivtrafo, intpoints.IP(), pdistype, distype, surfaceid);
  return;
}

//! specialization for 2D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<2>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::Matrix<2, 2>& derivtrafo,
    const ::CORE::DRT::UTILS::IntPointsAndWeights<1>& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.IP().nquad, 2);
  derivtrafo.Clear();

  DRT::UTILS::LineGPToParentGP(pqxg, derivtrafo, intpoints.IP(), pdistype, distype, surfaceid);
  return;
}

//! specializations for GaussPoint quadrature rules
//! specialization for 3D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<3>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::SerialDenseMatrix& derivtrafo, const CORE::DRT::UTILS::GaussPoints& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.NumPoints(), 3);
  if (derivtrafo.numRows() != 3 || derivtrafo.numCols() != 3) derivtrafo.shape(3, 3);

  DRT::UTILS::SurfaceGPToParentGP(pqxg, derivtrafo, intpoints, pdistype, distype, surfaceid);
  return;
}

//! specialization for 2D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<2>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::SerialDenseMatrix& derivtrafo, const CORE::DRT::UTILS::GaussPoints& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.NumPoints(), 2);
  if (derivtrafo.numRows() != 2 || derivtrafo.numCols() != 2) derivtrafo.shape(2, 2);

  DRT::UTILS::LineGPToParentGP(pqxg, derivtrafo, intpoints, pdistype, distype, surfaceid);
  return;
}

//! specialization for 3D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<3>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::Matrix<3, 3>& derivtrafo, const CORE::DRT::UTILS::GaussPoints& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.NumPoints(), 3);
  derivtrafo.Clear();

  DRT::UTILS::SurfaceGPToParentGP(pqxg, derivtrafo, intpoints, pdistype, distype, surfaceid);
  return;
}

//! specialization for 2D
template <>
void CORE::DRT::UTILS::BoundaryGPToParentGP<2>(CORE::LINALG::SerialDenseMatrix& pqxg,
    CORE::LINALG::Matrix<2, 2>& derivtrafo, const CORE::DRT::UTILS::GaussPoints& intpoints,
    const ::DRT::Element::DiscretizationType pdistype,
    const ::DRT::Element::DiscretizationType distype, const int surfaceid)
{
  // resize output array
  pqxg.shape(intpoints.NumPoints(), 2);
  derivtrafo.Clear();

  DRT::UTILS::LineGPToParentGP(pqxg, derivtrafo, intpoints, pdistype, distype, surfaceid);
  return;
}


template CORE::LINALG::Matrix<3, 1> CORE::DRT::UTILS::CalculateParentGPFromFaceElementData<3>(
    const double* faceele_xi, const ::DRT::FaceElement* faceele);
