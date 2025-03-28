// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_BOUNDARY_INTEGRATION_HPP
#define FOUR_C_FEM_GENERAL_UTILS_BOUNDARY_INTEGRATION_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  struct IntegrationPoints1D;
  struct IntegrationPoints2D;

  /*-----------------------------------------------------------------

    \brief Transform Gausspoints on surface element to 3d space of
           parent element (required for integrations of parent-element
           shape functions over boundary elements, required for example
           in weak dirichlet boundary conditions).

  \param pqxg       (o) transformed integration points of surface
                        elements in parent elements reference
                        coordinates
  \param derivtrafo (o) transformation matrix between parent and boundary local coordinates
                        for derivatives w.r.t. local coordinates
  \param intpoints  (i) integration points of surface element in its
                        reference coordinates
  \param pdistype   (i) discretisation type of parent element
  \param distype    (i) discretisation type of boundary element
  \param lineid     (i) local id of boundary element


    -----------------------------------------------------------------*/
  template <class V, class W, typename IntegrationPoints>
  void surface_gp_to_parent_gp(V& pqxg, W& derivtrafo, const IntegrationPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid)
  {
    if ((distype == Core::FE::CellType::quad4 && pdistype == Core::FE::CellType::hex8) or
        (distype == Core::FE::CellType::quad9 && pdistype == Core::FE::CellType::hex27))
    {
      switch (surfaceid)
      {
        case 0:
        {
          // t=-1
          /*
                    parent               surface

                     r|                    s|
                      |                     |
                 1         2           3         2
                  +-------+             +-------+
                  |   |   |      s      |   |   |      r
                  |   +-- |  -----      |   +-- |  -----
                  |       |             |       |
                  +-------+             +-------+
                 0         3           0         1
          */

          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[1];
            pqxg(iquad, 1) = intpoints.point(iquad)[0];
            pqxg(iquad, 2) = -1.0;
          }
          derivtrafo(0, 1) = 1.0;
          derivtrafo(1, 0) = 1.0;
          derivtrafo(2, 2) = -1.0;
          break;
        }
        case 1:
        {
          // s=-1
          /*
                    parent               surface

                     t|                    s|
                      |                     |
                 4         5           3         2
                  +-------+             +-------+
                  |   |   |      r      |   |   |      r
                  |   +-- |  -----      |   +-- |  -----
                  |       |             |       |
                  +-------+             +-------+
                 0         1           0         1
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[0];
            pqxg(iquad, 1) = -1.0;
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 0) = 1.0;
          derivtrafo(1, 2) = -1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        case 2:
        {
          // r= 1
          /*
                    parent               surface

                     t|                    s|
                      |                     |
                 5         6           3         2
                  +-------+             +-------+
                  |   |   |      s      |   |   |      r
                  |   +-- |  -----      |   +-- |  -----
                  |       |             |       |
                  +-------+             +-------+
                 1         2           0         1
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0;
            pqxg(iquad, 1) = intpoints.point(iquad)[0];
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 2) = 1.0;
          derivtrafo(1, 0) = 1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        case 3:
        {
          // s= 1
          /*
                    parent               surface

                     t|                    s|
                      |                     |
                 6         7           3         2
                  +-------+             +-------+
            r     |   |   |             |   |   |      r
            ----  |   +-- |             |   +-- |  -----
                  |       |             |       |
                  +-------+             +-------+
                 2         3           0         1
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -intpoints.point(iquad)[0];
            pqxg(iquad, 1) = 1.0;
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 0) = -1.0;
          derivtrafo(1, 2) = 1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        case 4:
        {
          // r=-1
          /*
                    parent               surface

                     s|                    s|
                      |                     |
                 3         7           3         2
                  +-------+             +-------+
                  |   |   |      t      |   |   |      r
                  |   +-- |  -----      |   +-- |  -----
                  |       |             |       |
                  +-------+             +-------+
                 0         4           0         1
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -1.0;
            pqxg(iquad, 1) = intpoints.point(iquad)[1];
            pqxg(iquad, 2) = intpoints.point(iquad)[0];
          }
          derivtrafo(0, 2) = -1.0;
          derivtrafo(1, 1) = 1.0;
          derivtrafo(2, 0) = 1.0;
          break;
        }
        case 5:
        {
          // t=1
          /*
                    parent               surface

                     s|                    s|
                      |                     |
                 7         6           3         2
                  +-------+             +-------+
                  |   |   |      r      |   |   |      r
                  |   +-- |  -----      |   +-- |  -----
                  |       |             |       |
                  +-------+             +-------+
                 4         5           0         1
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[0];
            pqxg(iquad, 1) = intpoints.point(iquad)[1];
            pqxg(iquad, 2) = 1.0;
          }
          derivtrafo(0, 0) = 1.0;
          derivtrafo(1, 1) = 1.0;
          derivtrafo(2, 2) = 1.0;
          break;
        }
        default:
          FOUR_C_THROW("invalid number of surfaces, unable to determine intpoint in parent");
      }
    }
    else if (distype == Core::FE::CellType::nurbs9 && pdistype == Core::FE::CellType::nurbs27)
    {
      switch (surfaceid)
      {
        case 0:
        {
          // t=-1
          /*
                    parent               surface

                     s|                    s|
                      |                     |
                  +---+---+             +---+---+
                 6|  7|  8|      r     6|  7|  8|      r
                  +   +-- +  -----      +   +-- +  -----
                 3|  4   5|            3|  4   5|
                  +---+---+             +---+---+
                 0   1   2             0   1   2
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[0];
            pqxg(iquad, 1) = intpoints.point(iquad)[1];
            pqxg(iquad, 2) = -1.0;
          }
          derivtrafo(0, 0) = 1.0;
          derivtrafo(1, 1) = 1.0;
          derivtrafo(2, 2) = -1.0;
          break;
        }
        case 1:
        {
          // t=+1
          /*
                    parent               surface

                     s|                    s|
                      |                     |
                  +---+---+             +---+---+
                24| 25| 26|      r     6|  7|  8|      r
                  +   +-- +  -----      +   +-- +  -----
                21| 22  23|            3|  4   5|
                  +---+---+             +---+---+
                18  19  20             0   1   2
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[0];
            pqxg(iquad, 1) = intpoints.point(iquad)[1];
            pqxg(iquad, 2) = 1.0;
          }
          derivtrafo(0, 0) = 1.0;
          derivtrafo(1, 1) = 1.0;
          derivtrafo(2, 2) = 1.0;
          break;
        }
        case 2:
        {
          // s=-1
          /*
                    parent               surface

                     t|                    s|
                      |                     |
                  +---+---+             +---+---+
                18| 19| 20|      r     6|  7|  8|      r
                  +   +-- +  -----      +   +-- +  -----
                 9| 10  11|            3|  4   5|
                  +---+---+             +---+---+
                 0   1   2             0   1   2
          */

          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[0];
            pqxg(iquad, 1) = -1.0;
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 0) = 1.0;
          derivtrafo(1, 2) = -1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        case 3:
        {
          // s=+1
          /*
                    parent               surface

                     t|                    s|
                      |                     |
                  +---+---+             +---+---+
                24| 25| 26|    r       6|  7|  8|      r
                  +   +-- + ----        +   +-- +  -----
                15| 16  17|            3|  4   5|
                  +---+---+             +---+---+
                 6   7   8             0   1   2
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[0];
            pqxg(iquad, 1) = 1.0;
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 0) = 1.0;
          derivtrafo(1, 2) = 1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        case 4:
        {
          // r=+1
          /*
                    parent               surface

                     t|                    s|
                      |                     |
                  +---+---+             +---+---+
                20| 23| 26|      s     6|  7|  8|      r
                  +   +-- +  -----      +   +-- +  -----
                11| 14  17|            3|  4   5|
                  +---+---+             +---+---+
                 2   5   8             0   1   2
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0;
            pqxg(iquad, 1) = intpoints.point(iquad)[0];
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 2) = 1.0;
          derivtrafo(1, 0) = 1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        case 5:
        {
          // r=-1
          /*
                    parent               surface

                     t|                    s|
                      |                     |
                  +---+---+             +---+---+
                18| 21| 24|      s     6|  7|  8|      r
                  +   +-- +  -----      +   +-- +  -----
                 9| 12  15|            3|  4   5|
                  +---+---+             +---+---+
                 0   3   6             0   1   2
          */
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -1.0;
            pqxg(iquad, 1) = intpoints.point(iquad)[0];
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 2) = 1.0;
          derivtrafo(1, 0) = 1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        default:
          FOUR_C_THROW("invalid number of surfaces, unable to determine intpoint in parent");
      }
    }
    else if ((distype == Core::FE::CellType::tri3 && pdistype == Core::FE::CellType::tet4) or
             (distype == Core::FE::CellType::tri6 && pdistype == Core::FE::CellType::tet10))
    {
      switch (surfaceid)
      {
        case 0:
        {
          // s=0
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[0];
            pqxg(iquad, 1) = 0.0;
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 0) = 1.0;
          derivtrafo(1, 2) = -1.0;
          derivtrafo(2, 1) = 1.0;
          break;
        }
        case 1:
        {
          // r+s+t=1
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0 - intpoints.point(iquad)[0] - intpoints.point(iquad)[1];
            pqxg(iquad, 1) = intpoints.point(iquad)[0];
            pqxg(iquad, 2) = intpoints.point(iquad)[1];
          }
          derivtrafo(0, 0) = -1.0;
          derivtrafo(0, 1) = -1.0;
          derivtrafo(0, 2) = 1.0;
          derivtrafo(1, 0) = 1.0;
          derivtrafo(1, 2) = 1.0;
          derivtrafo(2, 1) = 1.0;
          derivtrafo(2, 2) = 1.0;
          break;
        }
        case 2:
        {
          // r=0
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 0.0;
            pqxg(iquad, 1) = intpoints.point(iquad)[1];
            pqxg(iquad, 2) = intpoints.point(iquad)[0];
          }
          derivtrafo(0, 2) = -1.0;
          derivtrafo(1, 1) = 1.0;
          derivtrafo(2, 0) = 1.0;
          break;
        }
        case 3:
        {
          // t=0
          for (int iquad = 0; iquad < pqxg.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints.point(iquad)[1];
            pqxg(iquad, 1) = intpoints.point(iquad)[0];
            pqxg(iquad, 2) = 0.0;
          }
          derivtrafo(0, 1) = 1.0;
          derivtrafo(1, 0) = 1.0;
          derivtrafo(2, 2) = -1.0;
          break;
        }
        default:
          FOUR_C_THROW("invalid number of surfaces, unable to determine intpoint in parent");
      }
    }
    else
    {
      FOUR_C_THROW(
          "only quad4/hex8, quad9/hex27, tri3/tet4 and nurbs9/nurbs27 mappings of surface "
          "gausspoint to parent element implemented up to now\n");
    }
  }

  /*-----------------------------------------------------------------

  \brief Transform Gausspoints on line element to 2d space of
         parent element (required for integrations of parent-element
         shape functions over boundary elements, for example
         in weak dirichlet boundary conditions).


  \param pqxg       (o) transformed integration points of surface
                        elements in parent elements reference
                        coordinates
  \param derivtrafo (o) transformation matrix between parent and boundary local coordinates
                        for derivatives w.r.t. local coordinates
  \param intpoints  (i) integration points of surface element in its
                        reference coordinates
  \param pdistype   (i) discretisation type of parent element
  \param distype    (i) discretisation type of boundary element
  \param lineid     (i) local id of boundary element


    -----------------------------------------------------------------*/
  template <class V, class W, typename IntegrationPoints>
  void line_gp_to_parent_gp(V& pqxg, W& derivtrafo, const IntegrationPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int lineid);

  /*-----------------------------------------------------------------

  \brief Template version of transformation of Gausspoints on boundary element to space of
         parent element

  \param pqxg       (o) transformed integration points of surface
                        elements in parent elements reference
                        coordinates
  \param derivtrafo (o) transformation matrix between parent and boundary local coordinates
                        for derivatives w.r.t. local coordinates
  \param intpoints  (i) integration points of surface element in its
                        reference coordinates
  \param pdistype   (i) discretisation type of parent element
  \param distype    (i) discretisation type of boundary element
  \param lineid     (i) local id of boundary element

    ----------------------------------------------------------------------------------*/
  template <const int nsd>
  void boundary_gp_to_parent_gp(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::SerialDenseMatrix& derivtrafo,
      const Core::FE::IntPointsAndWeights<nsd - 1>& intpoints, const Core::FE::CellType pdistype,
      const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 3D
  template <>
  void boundary_gp_to_parent_gp<3>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::SerialDenseMatrix& derivtrafo,
      const Core::FE::IntPointsAndWeights<2>& intpoints, const Core::FE::CellType pdistype,
      const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 2D
  template <>
  void boundary_gp_to_parent_gp<2>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::SerialDenseMatrix& derivtrafo,
      const Core::FE::IntPointsAndWeights<1>& intpoints, const Core::FE::CellType pdistype,
      const Core::FE::CellType distype, const int surfaceid);



  template <const int nsd>
  void boundary_gp_to_parent_gp(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::Matrix<nsd, nsd>& derivtrafo,
      const Core::FE::IntPointsAndWeights<nsd - 1>& intpoints, const Core::FE::CellType pdistype,
      const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 3D
  template <>
  void boundary_gp_to_parent_gp<3>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::Matrix<3, 3>& derivtrafo, const Core::FE::IntPointsAndWeights<2>& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 2D
  template <>
  void boundary_gp_to_parent_gp<2>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::Matrix<2, 2>& derivtrafo, const Core::FE::IntPointsAndWeights<1>& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);



  template <const int nsd>
  void boundary_gp_to_parent_gp(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::SerialDenseMatrix& derivtrafo, const GaussPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 3D
  template <>
  void boundary_gp_to_parent_gp<3>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::SerialDenseMatrix& derivtrafo, const GaussPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 2D
  template <>
  void boundary_gp_to_parent_gp<2>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::SerialDenseMatrix& derivtrafo, const GaussPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);

  template <const int nsd>
  void boundary_gp_to_parent_gp(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::Matrix<nsd, nsd>& derivtrafo, const GaussPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 3D
  template <>
  void boundary_gp_to_parent_gp<3>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::Matrix<3, 3>& derivtrafo, const GaussPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);

  //! specialization for 2D
  template <>
  void boundary_gp_to_parent_gp<2>(Core::LinAlg::SerialDenseMatrix& pqxg,
      Core::LinAlg::Matrix<2, 2>& derivtrafo, const GaussPoints& intpoints,
      const Core::FE::CellType pdistype, const Core::FE::CellType distype, const int surfaceid);

  /*----------------------------------------------------------------------------------------*/

  /*!
   * @brief Calculates the parent element gauss point coordinates from a given face element gauss
   * point coordinate
   *
   * @tparam parent_ele_dim  dimension of the parent element
   * @param faceele_xi       coordinates of the face element gauss point in parameter space
   * @param faceele          face element
   * @return  coordinates of the parent element gauss point corresponding to the face element gauss
   * point in parameter space
   */
  template <int parent_ele_dim>
  Core::LinAlg::Matrix<parent_ele_dim, 1> calculate_parent_gp_from_face_element_data(
      const double* faceele_xi, const Core::Elements::FaceElement* faceele);

  //! compute covariant metric tensor for surface element
  void compute_metric_tensor_for_surface(const Core::LinAlg::SerialDenseMatrix& xyze,
      const Core::LinAlg::SerialDenseMatrix& deriv, Core::LinAlg::SerialDenseMatrix& metrictensor,
      double* sqrtdetg);

  //! compute covariant metric tensor for surface/line element and optionally, the normalized
  //! normal vector at the Gauss-point (template)
  template <Core::FE::CellType distype, int probdim, typename ValueType>
  void compute_metric_tensor_for_boundary_ele(
      const Core::LinAlg::Matrix<probdim, Core::FE::num_nodes<distype>, ValueType>& xyze,
      const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>, ValueType>&
          deriv,
      Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>, ValueType>& metrictensor,
      ValueType& sqrtdetg, const bool throw_error,
      Core::LinAlg::Matrix<probdim, 1, ValueType>* normalvec = nullptr,
      const bool unit_normal = true)
  {
    /* 2D boundary Element
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

    /* 1D boundary Element
    |
    |   dxyzdrs(1,2) = deriv(1,iel) * xyze(2, iel)^T
     */
    Core::LinAlg::Matrix<Core::FE::dim<distype>, probdim, ValueType> dxyzdrs;
    dxyzdrs.multiply_nt(deriv, xyze);

    /* 2D boundary Element
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

    /* 1D boundary Element
    |
    |   metrictensor(1,1) = dxyzdrs(1,2) * dxyzdrs(1,2)^T
     */

    metrictensor.clear();
    metrictensor.multiply_nt(dxyzdrs, dxyzdrs);

    /*
                              +--------------+
                             /               |
               sqrtdetg =   /  g11*g22-g12^2
                          \/
    */
    sqrtdetg = metrictensor.determinant();
    if (sqrtdetg > 0.0)
      sqrtdetg = Core::MathOperations<ValueType>::sqrt(sqrtdetg);
    else if (throw_error)
    {
      FOUR_C_THROW(
          "--- ERROR DETECTED ---\n The determinant of the matrix is equal zero or negative!");
    }

    // Calculate outward pointing normal vector
    if (normalvec)
    {
      if (probdim == 3 and Core::FE::dim<distype> == 2)
      {
        (*normalvec)(0) = dxyzdrs(0, 1) * dxyzdrs(1, 2) - dxyzdrs(1, 1) * dxyzdrs(0, 2);
        (*normalvec)(1) = dxyzdrs(0, 2) * dxyzdrs(1, 0) - dxyzdrs(1, 2) * dxyzdrs(0, 0);
        (*normalvec)(2) = dxyzdrs(0, 0) * dxyzdrs(1, 1) - dxyzdrs(1, 0) * dxyzdrs(0, 1);
      }
      else if (probdim == 2 and Core::FE::dim<distype> == 1)
      {
        (*normalvec)(0) = dxyzdrs(0, 1);
        (*normalvec)(1) = -dxyzdrs(0, 0);
      }
      else if (probdim == 3 and Core::FE::dim<distype> == 1)
      {
        // handle small normal vectors with dxyzdrs(0,0), dxyzdrs(0,1) < 1e-6
        // and other components = 0.0
        if (Core::MathOperations<ValueType>::abs(dxyzdrs(0, 0)) < 1.0e-6 and
            Core::MathOperations<ValueType>::abs(dxyzdrs(0, 1)) < 1.0e-6 and
            (dxyzdrs(0, 2) != 0.0 or dxyzdrs(0, 1) != 0.0))
        {
          (*normalvec)(0) = 0.0;
          (*normalvec)(1) = -dxyzdrs(0, 2);
          (*normalvec)(2) = dxyzdrs(0, 1);
        }
        else
        {
          (*normalvec)(0) = dxyzdrs(0, 1);
          (*normalvec)(1) = -dxyzdrs(0, 0);
          (*normalvec)(2) = 0.0;
        }
      }
      else
        FOUR_C_THROW("There are only 2D and 1D boundary elements");

      // compute unit normal (outward pointing)
      if (unit_normal)
      {
        const ValueType norm2 = normalvec->norm2();
        if (norm2 < 0.0)
        {
          FOUR_C_THROW("The L2-norm of the normal vector is smaller than 0.0!");
        }
        normalvec->scale(1.0 / norm2);
      }
    }
  }

  //! compute kovariant metric tensor for surface/line element and optionally, the normalized
  //! normal vector at the Gausspoint (template)
  template <Core::FE::CellType distype>
  void compute_metric_tensor_for_boundary_ele(
      const Core::LinAlg::Matrix<(1 + Core::FE::dim<distype>), Core::FE::num_nodes<distype>>& xyze,
      const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& deriv,
      Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::dim<distype>>& metrictensor,
      double& sqrtdetg, Core::LinAlg::Matrix<(1 + Core::FE::dim<distype>), 1>* normalvec = nullptr)
  {
    const bool throw_error_if_negative_determinant(true);
    compute_metric_tensor_for_boundary_ele<distype, (1 + Core::FE::dim<distype>)>(
        xyze, deriv, metrictensor, sqrtdetg, throw_error_if_negative_determinant, normalvec);
  }

  /*-----------------------------------------------------------------

  \brief Transform Gausspoints on boundary element to space of
         parent element (required for integrations of parent-element
         shape functions over boundary elements, for example
         in weak dirichlet boundary conditions).
         2D-version
    -----------------------------------------------------------------*/
  inline void boundary_gp_to_parent_gp2(Core::LinAlg::SerialDenseMatrix& pqxg,
      const Core::LinAlg::SerialDenseMatrix& intpoints, const Core::FE::CellType pdistype,
      const Core::FE::CellType distype, const int beleid)
  {
    // resize output array
    pqxg.shape(intpoints.numRows(), 2);

    if ((distype == Core::FE::CellType::line2 && pdistype == Core::FE::CellType::quad4) or
        (distype == Core::FE::CellType::line3 && pdistype == Core::FE::CellType::quad8) or
        (distype == Core::FE::CellType::line3 && pdistype == Core::FE::CellType::quad9))
    {
      switch (beleid)
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
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = -1.0;
          }
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
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0;
            pqxg(iquad, 1) = intpoints(iquad, 0);
          }
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

          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -intpoints(iquad, 0);
            pqxg(iquad, 1) = 1.0;
          }
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

           s|                           r|
            |                            |
            +                            +
           6|                           2|
            +                            +
           3|                           1|
            +                            +
           0                            0
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -1.0;
            pqxg(iquad, 1) = -intpoints(iquad, 0);
          }
          break;
        }
        default:
          FOUR_C_THROW("invalid number of lines, unable to determine intpoint in parent");
      }
    }
    else if ((distype == Core::FE::CellType::line2 && pdistype == Core::FE::CellType::tri3) or
             (distype == Core::FE::CellType::line3 && pdistype == Core::FE::CellType::tri6))
    {
      switch (beleid)
      {
        case 0:
        {
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 0.5 + 0.5 * intpoints(iquad, 0);
            pqxg(iquad, 1) = 0.0;
          }
          break;
        }
        case 1:
        {
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 0.5 - 0.5 * intpoints(iquad, 0);
            pqxg(iquad, 1) = 0.5 + 0.5 * intpoints(iquad, 0);  // r+s=1.0 -> s= 1.0-r
          }
          break;
        }
        case 2:
        {
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 0.0;
            pqxg(iquad, 1) = 0.5 - 0.5 * intpoints(iquad, 0);
          }
          break;
        }
        default:
          FOUR_C_THROW("invalid number of lines, unable to determine intpoint in parent");
      }
    }
    else if (distype == Core::FE::CellType::nurbs3 && pdistype == Core::FE::CellType::nurbs9)
    {
      switch (beleid)
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
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = -1.0;
          }
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
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0;
            pqxg(iquad, 1) = intpoints(iquad, 0);
          }
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

          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = 1.0;
          }
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
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -1.0;
            pqxg(iquad, 1) = intpoints(iquad, 0);
          }
          break;
        }
        default:
          FOUR_C_THROW("invalid number of lines, unable to determine intpoint in parent");
      }
    }
    else
    {
      FOUR_C_THROW(
          "only line2/quad4 and nurbs3/nurbs9 mappings of surface gausspoint to parent element "
          "implemented up to now\n");
    }
  };
  /*-----------------------------------------------------------------
  \brief Transform Gausspoints on boundary element to space of
         parent element (required for integrations of parent-element
         shape functions over boundary elements, for example
         in weak dirichlet boundary conditions).
         3D-version
    -----------------------------------------------------------------*/
  inline void boundary_gp_to_parent_gp3(Core::LinAlg::SerialDenseMatrix& pqxg,
      const Core::LinAlg::SerialDenseMatrix& intpoints, const Core::FE::CellType pdistype,
      const Core::FE::CellType distype, const int beleid)
  {
    // resize output array
    pqxg.shape(intpoints.numRows(), 3);

    if ((distype == Core::FE::CellType::quad4 && pdistype == Core::FE::CellType::hex8) or
        (distype == Core::FE::CellType::quad8 && pdistype == Core::FE::CellType::hex20) or
        (distype == Core::FE::CellType::quad9 && pdistype == Core::FE::CellType::hex27))
    {
      switch (beleid)
      {
        case 0:
        {
          // t=-1
          /*
                  parent               surface

                   r|                    s|
                    |                     |
               1         2           3         2
                +-------+             +-------+
                |   |   |      s      |   |   |      r
                |   +-- |  -----      |   +-- |  -----
                |       |             |       |
                +-------+             +-------+
               0         3           0         1
          */

          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 1);
            pqxg(iquad, 1) = intpoints(iquad, 0);
            pqxg(iquad, 2) = -1.0;
          }
          break;
        }
        case 1:
        {
          // s=-1
          /*
                  parent               surface
                   t|                    s|
                    |                     |
               4         5           3         2
                +-------+             +-------+
                |   |   |      r      |   |   |      r
                |   +-- |  -----      |   +-- |  -----
                |       |             |       |
                +-------+             +-------+
               0         1           0         1
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = -1.0;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 2:
        {
          // r= 1
          /*
                  parent               surface

                   t|                    s|
                    |                     |
               5         6           3         2
                +-------+             +-------+
                |   |   |      s      |   |   |      r
                |   +-- |  -----      |   +-- |  -----
                |       |             |       |
                +-------+             +-------+
               1         2           0         1
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0;
            pqxg(iquad, 1) = intpoints(iquad, 0);
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 3:
        {
          // s= 1
          /*
                  parent               surface

                   t|                    s|
                    |                     |
               6         7           3         2
                +-------+             +-------+
          r     |   |   |             |   |   |      r
          ----  | --+   |             |   +-- |  -----
                |       |             |       |
                +-------+             +-------+
               2         3           0         1
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -intpoints(iquad, 0);
            pqxg(iquad, 1) = 1.0;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 4:
        {
          // r=-1
          /*
                  parent               surface

                   s|                    s|
                    |                     |
               3         7           3         2
                +-------+             +-------+
                |   |   |      t      |   |   |      r
                |   +-- |  -----      |   +-- |  -----
                |       |             |       |
                +-------+             +-------+
               0         4           0         1
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -1.0;
            pqxg(iquad, 1) = intpoints(iquad, 1);
            pqxg(iquad, 2) = intpoints(iquad, 0);
          }
          break;
        }
        case 5:
        {
          // t=1
          /*
                  parent               surface

                   s|                    s|
                    |                     |
               7         6           3         2
                +-------+             +-------+
                |   |   |      r      |   |   |      r
                |   +-- |  -----      |   +-- |  -----
                |       |             |       |
                +-------+             +-------+
               4         5           0         1
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = intpoints(iquad, 1);
            pqxg(iquad, 2) = 1.0;
          }
          break;
        }
        default:
          FOUR_C_THROW("invalid number of surfaces, unable to determine intpoint in parent");
      }
    }
    else if ((distype == Core::FE::CellType::tri3 && pdistype == Core::FE::CellType::tet4) or
             (distype == Core::FE::CellType::tri6 && pdistype == Core::FE::CellType::tet10))
    {
      switch (beleid)
      {
        case 0:
        {
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = 0.0;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 1:
        {
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0 - intpoints(iquad, 0) - intpoints(iquad, 1);
            pqxg(iquad, 1) = intpoints(iquad, 0);
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 2:
        {
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 0.0;
            pqxg(iquad, 1) = intpoints(iquad, 1);
            pqxg(iquad, 2) = intpoints(iquad, 0);
          }
          break;
        }
        case 3:
        {
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 1);
            pqxg(iquad, 1) = intpoints(iquad, 0);
            pqxg(iquad, 2) = 0.0;
          }
          break;
        }
        default:
          FOUR_C_THROW("invalid number of surfaces, unable to determine intpoint in parent");
      }
    }
    else if ((distype == Core::FE::CellType::quad4 && pdistype == Core::FE::CellType::wedge6) or
             (distype == Core::FE::CellType::quad8 && pdistype == Core::FE::CellType::wedge15))
    {
      switch (beleid)
      {
        case 0:
        {
          // r+s=1
          /*
                        parent               surface

                     t|                        s|
                      |                         |
                     3|        4           3         2
                      +-------+             +-------+
                      |       |      s      |   |   |      r
                      +-------|  -----      |   +-- |  -----
                      |       |             |       |
                      +-------+             +-------+
                     0         1           0         1
           */

          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -0.5 * intpoints(iquad, 0) + 0.5;
            pqxg(iquad, 1) = 0.5 * intpoints(iquad, 0) + 0.5;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 1:
        {
          // r=0
          /*
                        parent               surface
                             t|                s|
                              |                 |
                     4        |5           3         2
                      +-------+             +-------+
                 s    |       |             |   |   |      r
                 -----|-------+             |   +-- |  -----
                      |       |             |       |
                      +-------+             +-------+
                     1         2           0         1
           */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 0.0;
            pqxg(iquad, 1) = -0.5 * intpoints(iquad, 0) + 0.5;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 2:
        {
          // s=0
          /*
                        parent               surface

                     t|                        s|
                      |                         |
                     5|        3           3    |    2
                      +-------+             +-------+
                      |       |      r      |   |   |      r
                      +       |  -----      |   +-- |  -----
                      |       |             |       |
                      +-------+             +-------+
                     2         0           0         1
           */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 0.5 * intpoints(iquad, 0) + 0.5;
            pqxg(iquad, 1) = 0.0;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        default:
          FOUR_C_THROW(
              "invalid number of quad4 surface for wedge element, unable to determine intpoint "
              "in parent");
      }
    }
    else if ((distype == Core::FE::CellType::tri3 && pdistype == Core::FE::CellType::wedge6) or
             (distype == Core::FE::CellType::tri6 && pdistype == Core::FE::CellType::wedge15))
    {
      switch (beleid)
      {
        case 3:
        {
          // t=-1.0
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 1);
            pqxg(iquad, 1) = intpoints(iquad, 0);
            pqxg(iquad, 2) = -1.0;
          }
          break;
        }
        case 4:
        {
          // t=1.0
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = intpoints(iquad, 1);
            pqxg(iquad, 2) = 1.0;
          }
          break;
        }
        default:
          FOUR_C_THROW(
              "invalid number of tri3 surface for wedge element, unable to determine intpoint in "
              "parent");
      }
    }
    else if (distype == Core::FE::CellType::nurbs9 && pdistype == Core::FE::CellType::nurbs27)
    {
      switch (beleid)
      {
        case 0:
        {
          // t=-1
          /*
                  parent               surface

                   s|                    s|
                    |                     |
                +---+---+             +---+---+
               6|  7|  8|      r     6|  7|  8|      r
                +   +-- +  -----      +   +-- +  -----
               3|  4   5|            3|  4   5|
                +---+---+             +---+---+
               0   1   2             0   1   2
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = intpoints(iquad, 1);
            pqxg(iquad, 2) = -1.0;
          }
          break;
        }
        case 1:
        {
          // t=+1
          /*
                  parent               surface

                   s|                    s|
                    |                     |
                +---+---+             +---+---+
              24| 25| 26|      r     6|  7|  8|      r
                +   +-- +  -----      +   +-- +  -----
              21| 22  23|            3|  4   5|
                +---+---+             +---+---+
              18  19  20             0   1   2
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = intpoints(iquad, 1);
            pqxg(iquad, 2) = 1.0;
          }
          break;
        }
        case 2:
        {
          // s=-1
          /*
                  parent               surface

                   t|                    s|
                    |                     |
                +---+---+             +---+---+
              18| 19| 20|      r     6|  7|  8|      r
                +   +-- +  -----      +   +-- +  -----
               9| 10  11|            3|  4   5|
                +---+---+             +---+---+
               0   1   2             0   1   2
          */

          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = -1.0;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 3:
        {
          // s=+1
          /*
                  parent               surface

                   t|                    s|
                    |                     |
                +---+---+             +---+---+
              24| 25| 26|    r       6|  7|  8|      r
                +   +-- + ----        +   +-- +  -----
              15| 16  17|            3|  4   5|
                +---+---+             +---+---+
               6   7   8             0   1   2
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = intpoints(iquad, 0);
            pqxg(iquad, 1) = 1.0;
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 4:
        {
          // r=+1
          /*
                  parent               surface

                   t|                    s|
                    |                     |
                +---+---+             +---+---+
              20| 23| 26|      s     6|  7|  8|      r
                +   +-- +  -----      +   +-- +  -----
              11| 14  17|            3|  4   5|
                +---+---+             +---+---+
               2   5   8             0   1   2
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = 1.0;
            pqxg(iquad, 1) = intpoints(iquad, 0);
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        case 5:
        {
          // r=-1
          /*
                  parent               surface

                   t|                    s|
                    |                     |
                +---+---+             +---+---+
              18| 21| 24|      s     6|  7|  8|      r
                +   +-- +  -----      +   +-- +  -----
               9| 12  15|            3|  4   5|
                +---+---+             +---+---+
               0   3   6             0   1   2
          */
          for (int iquad = 0; iquad < intpoints.numRows(); ++iquad)
          {
            pqxg(iquad, 0) = -1.0;
            pqxg(iquad, 1) = intpoints(iquad, 0);
            pqxg(iquad, 2) = intpoints(iquad, 1);
          }
          break;
        }
        default:
          FOUR_C_THROW("invalid number of surfaces, unable to determine intpoint in parent");
      }
    }
    else
    {
      FOUR_C_THROW(
          "only quad4/hex8 and nurbs9/nurbs27 mappings of surface gausspoint to parent element "
          "implemented up to now\n");
    }
  };

  // Calculation of the following things for the boundary element (all outputs):
  // - Shape function 'funct' and its derivative 'deriv' at the Gauss point
  //   as well as the integration factor 'fac'
  // - Unit normal vector 'unitnormal' (pointing out of the domain) and the
  //   infinitesimal area element 'drs' at the Gauss points
  // - Coordinates of the current integration point in reference coordinates
  //   'xsi' and the node coordinates for the boundary element 'xyze'
  // Inputs:
  // - intpoints: integration points
  // - gpid: actual Gauss point
  // - myknots: nurbs specific knotes
  // - isnurbs: if true, some NURBS specific stuff is done
  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  template <Core::FE::CellType distype>
  void eval_shape_func_at_bou_int_point(
      Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, 1>& funct,
      Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& deriv,
      double& fac, Core::LinAlg::Matrix<Core::FE::dim<distype> + 1, 1>& unitnormal, double& drs,
      Core::LinAlg::Matrix<Core::FE::dim<distype>, 1>& xsi,
      Core::LinAlg::Matrix<Core::FE::dim<distype> + 1, Core::FE::num_nodes<distype>>& xyze,
      const Core::FE::IntPointsAndWeights<Core::FE::dim<distype>>& intpoints, const int gpid,
      const std::vector<Core::LinAlg::SerialDenseVector>* myknots,
      const Core::LinAlg::SerialDenseVector* weights, const bool isnurbs)
  {
    // Obtain number of spatial dimensions of boundary element
    const int bdrynsd = Core::FE::dim<distype>;

    // local coordinates of the current integration point
    const double* gpcoord = (intpoints.ip().qxg)[gpid];
    for (int idim = 0; idim < bdrynsd; ++idim)
    {
      xsi(idim) = gpcoord[idim];
    }

    // get shape functions and derivatives in the plane of the element
    if (not isnurbs)
    {
      // shape functions and their first derivatives of boundary element
      Core::FE::shape_function<distype>(xsi, funct);
      Core::FE::shape_function_deriv1<distype>(xsi, deriv);
    }
    // only for NURBS!!!
    else
    {
      Core::FE::Nurbs::nurbs_get_funct_deriv(funct, deriv, xsi, *myknots, *weights, distype);
    }

    // compute measure tensor for surface element, infinitesimal area element drs
    // and (outward-pointing) unit normal vector
    Core::LinAlg::Matrix<bdrynsd, bdrynsd> metrictensor(Core::LinAlg::Initialization::zero);
    Core::FE::compute_metric_tensor_for_boundary_ele<distype>(
        xyze, deriv, metrictensor, drs, &unitnormal);

    // compute integration factor
    fac = intpoints.ip().qwgt[gpid] * drs;
  };  // eval_shape_func_at_bou_int_point

  // Calculate mass-consistent node normals for the boundary element
  // according to dissertation of Prof. Wall, equation (7.13)
  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  template <Core::FE::CellType distype>
  void element_node_normal(Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, 1>& funct,
      Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& deriv,
      double& fac, Core::LinAlg::Matrix<Core::FE::dim<distype> + 1, 1>& unitnormal, double& drs,
      Core::LinAlg::Matrix<Core::FE::dim<distype>, 1>& xsi,
      Core::LinAlg::Matrix<Core::FE::dim<distype> + 1, Core::FE::num_nodes<distype>>& xyze,
      Core::Elements::Element* ele, Core::FE::Discretization& discretization,
      Core::LinAlg::SerialDenseVector& elevec1, const std::vector<double>& edispnp,
      const bool isnurbs, const bool isale)
  {
    /*----------------------------------------------------------------------*
     |                          Initialization                              |
     *----------------------------------------------------------------------*/
    // Number of spatial dimensions of boundary element
    const int bdrynsd = Core::FE::dim<distype>;

    // Number of spatial dimensions of parent element
    static constexpr int nsd = bdrynsd + 1;

    // Number of nodes of boundary element
    static constexpr int bdrynen = Core::FE::num_nodes<distype>;

    // Number of degrees of freedom per node
    int numdofpernode = nsd + 1;  // standard case is 'fluid'
    if ((discretization.name() == "ale") or (discretization.name() == "structure"))
      numdofpernode = nsd;

    // This functionality has only been tested for types fluid, ale and structure
    if (not((discretization.name() == "fluid") or (discretization.name() == "inflow") or
            (discretization.name() == "ale") or (discretization.name() == "structure")))
    {
      FOUR_C_THROW(
          "element_node_normal: The mass-consistent-node-normal calculation can currently only be "
          "performed for discretization types 'fluid', 'inflow', 'ale' and 'structure'.");
    }

    // Get gaussrule
    const Core::FE::IntPointsAndWeights<bdrynsd> intpoints(
        Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

    // Get node coordinates
    Core::Geo::fill_initial_position_array<distype, nsd, Core::LinAlg::Matrix<nsd, bdrynen>>(
        ele, xyze);

    // Add displacements to reference coordinates, if an ALE description is used
    if (isale)
    {
      FOUR_C_ASSERT(edispnp.size() != 0, "paranoid");

      for (int inode = 0; inode < bdrynen; ++inode)
      {
        for (int idim = 0; idim < (nsd); ++idim)
        {
          xyze(idim, inode) += edispnp[numdofpernode * inode + idim];
        }
      }
    }

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
    {
      // Computation of the integration factor & shape function at the Gauss point & derivative of
      // the shape function at the Gauss point Computation of the unit normal vector at the Gauss
      // points Computation of nurb specific stuff is not activated here
      Core::FE::eval_shape_func_at_bou_int_point<distype>(funct, deriv, fac, unitnormal, drs, xsi,
          xyze, intpoints, gpid, nullptr, nullptr, isnurbs);

      for (int inode = 0; inode < bdrynen; ++inode)
      {
        for (int idim = 0; idim < nsd; ++idim)
        {
          elevec1(inode * numdofpernode + idim) += unitnormal(idim) * funct(inode) * fac;
        }
        if (numdofpernode > nsd)
        {  // this is the case for the fluid
          // pressure dof is set to zero
          elevec1(inode * numdofpernode + (nsd)) = 0.0;
        }
      }
    } /* end of loop over integration points gpid */
  }  // element_node_normal

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
