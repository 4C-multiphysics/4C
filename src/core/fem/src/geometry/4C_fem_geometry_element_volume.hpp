// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_ELEMENT_VOLUME_HPP
#define FOUR_C_FEM_GEOMETRY_ELEMENT_VOLUME_HPP


#include "4C_config.hpp"

#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Geo
{
  /**
   * @brief Compute the length of a 1D finite element or NURBS element.
   *
   * The element length is evaluated by Gauss integration of the Jacobian
   * \( \| \partial \mathbf{x} / \partial \xi \| \) over the parametric domain.
   * Standard finite elements use polynomial shape functions, while NURBS
   * elements use rational basis functions defined by the supplied knot vector
   * and weights.
   *
   * @tparam distype Element cell type (e.g. line2, line3, nurbs2, nurbs3).
   * @tparam Matrixtype Matrix-like type containing the element coordinates.
   *
   * @param xyze Nodal or control-point coordinates of the element.
   * @param knots Knot vectors for NURBS elements (unused for standard elements).
   * @param weights Control-point weights for NURBS elements (unused otherwise).
   *
   * @return The physical length of the element.
   */
  template <Core::FE::CellType distype, class Matrixtype>
  double element_length(const Matrixtype& xyze,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots = {},
      const Core::LinAlg::SerialDenseVector& weights = Core::LinAlg::SerialDenseVector{})
  {
    static constexpr int numnode = Core::FE::num_nodes(distype);
    const Core::FE::GaussIntegration intpoints(distype);

    Core::LinAlg::SerialDenseVector eleCoord(1);
    Core::LinAlg::Matrix<numnode, 1> funct;
    Core::LinAlg::Matrix<1, numnode> deriv;
    Core::LinAlg::Matrix<1, 3> xjm;

    double length = 0.0;

    for (int iquad = 0; iquad < intpoints.num_points(); ++iquad)
    {
      // coordinates of the current integration point in element coordinates \xi
      eleCoord(0) = intpoints.point(iquad)[0];

      // shape functions and their first derivatives
      if (distype != FE::CellType::nurbs2 and distype != FE::CellType::nurbs3)
      {
        Core::FE::shape_function_1d_deriv1(deriv, eleCoord(0), distype);
      }
      else
      {
        Core::FE::Nurbs::nurbs_get_1d_funct_deriv(
            funct, deriv, eleCoord(0), knots[0], weights, distype);
      }

      // get transposed of the jacobian matrix d x / d \xi
      xjm = 0;

      for (int inode = 0; inode < numnode; ++inode)
        for (int i = 0; i < 1; ++i)
          for (int j = 0; j < 3; ++j) xjm(i, j) += deriv(i, inode) * xyze(j, inode);

      const double fac = intpoints.weight(iquad) * xjm.norm2();

      length += fac;
    }

    return length;
  }

  /**
   * @brief Compute the area of a 2D finite element or NURBS element.
   *
   * The element area is evaluated by Gauss integration of the surface Jacobian
   * \( \sqrt{\det(\mathbf{J}\mathbf{J}^\mathsf{T})} \) over the parametric domain.
   * Standard finite elements use polynomial shape functions, while NURBS
   * elements use rational basis functions defined by the supplied knot vectors
   * and weights.
   *
   * @tparam distype Element cell type (e.g. tri3, tri6, quad4, quad8, quad9, nurbs4, nurbs9).
   * @tparam Matrixtype Matrix-like type containing the element coordinates.
   *
   * @param xyze Nodal or control-point coordinates of the element.
   * @param knots Knot vectors for NURBS elements (unused for standard elements).
   * @param weights Control-point weights for NURBS elements (unused otherwise).
   *
   * @return The physical area of the element.
   */
  template <Core::FE::CellType distype, class Matrixtype>
  double element_area(const Matrixtype& xyze,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots = {},
      const Core::LinAlg::SerialDenseVector& weights = Core::LinAlg::SerialDenseVector{})
  {
    static constexpr int numnode = Core::FE::num_nodes(distype);
    const Core::FE::GaussIntegration intpoints(distype);

    Core::LinAlg::SerialDenseVector eleCoord(2);
    Core::LinAlg::Matrix<numnode, 1> funct;
    Core::LinAlg::Matrix<2, numnode> deriv;
    Core::LinAlg::Matrix<2, 3> xjm;

    double area = 0.0;

    for (int iquad = 0; iquad < intpoints.num_points(); ++iquad)
    {
      // coordinates of the current integration point in element coordinates \xi
      eleCoord(0) = intpoints.point(iquad)[0];
      eleCoord(1) = intpoints.point(iquad)[1];

      // shape functions and their first derivatives
      if (distype != FE::CellType::nurbs4 and distype != FE::CellType::nurbs9)
      {
        Core::FE::shape_function_2d_deriv1(deriv, eleCoord(0), eleCoord(1), distype);
      }
      else
      {
        Core::FE::Nurbs::nurbs_get_2d_funct_deriv(funct, deriv, eleCoord, knots, weights, distype);
      }

      // get transposed of the jacobian matrix d x / d \xi
      xjm = 0;

      for (int inode = 0; inode < numnode; ++inode)
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 3; ++j) xjm(i, j) += deriv(i, inode) * xyze(j, inode);

      Core::LinAlg::Matrix<3, 1> tangent_xi;
      Core::LinAlg::Matrix<3, 1> tangent_eta;

      for (int dim = 0; dim < 3; ++dim)
      {
        tangent_xi(dim) = xjm(0, dim);
        tangent_eta(dim) = xjm(1, dim);
      }

      Core::LinAlg::Matrix<3, 1> surface_normal;
      surface_normal.cross_product(tangent_xi, tangent_eta);

      const double fac = intpoints.weight(iquad) * surface_normal.norm2();

      area += fac;
    }

    return area;
  }

  /**
   * @brief Compute the volume of a 3D finite element or NURBS element.
   *
   * The element volume is evaluated by Gauss integration of the Jacobian
   * determinant \( \det(\partial \mathbf{x} / \partial \boldsymbol{\xi}) \)
   * over the parametric domain. Standard finite elements use polynomial
   * shape functions, while NURBS elements use rational basis functions
   * defined by the supplied knot vectors and weights.
   *
   * @tparam distype Element cell type (e.g. tet4, tet10, hex8, hex20, hex27,
   *                 wedge6, wedge15, pyramid5, nurbs8, nurbs27).
   * @tparam Matrixtype Matrix-like type containing the element coordinates.
   *
   * @param xyze Nodal or control-point coordinates of the element.
   * @param knots Knot vectors for NURBS elements (unused for standard elements).
   * @param weights Control-point weights for NURBS elements (unused otherwise).
   *
   * @return The physical volume of the element.
   *
   * @throws FOUR_C_THROW If the Jacobian determinant is non-positive.
   */
  template <Core::FE::CellType distype, class Matrixtype>
  double element_volume(const Matrixtype& xyze,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots = {},
      const Core::LinAlg::SerialDenseVector& weights = Core::LinAlg::SerialDenseVector{})
  {
    const int numnode = Core::FE::num_nodes(distype);
    const Core::FE::GaussIntegration intpoints(distype);

    Core::LinAlg::SerialDenseVector eleCoord(3);
    Core::LinAlg::Matrix<numnode, 1> funct;
    Core::LinAlg::Matrix<3, numnode> deriv;
    Core::LinAlg::Matrix<3, 3> xjm;

    double vol = 0.0;

    for (int iquad = 0; iquad < intpoints.num_points(); ++iquad)
    {
      // coordinates of the current integration point in element coordinates \xi
      eleCoord(0) = intpoints.point(iquad)[0];
      eleCoord(1) = intpoints.point(iquad)[1];
      eleCoord(2) = intpoints.point(iquad)[2];

      // shape functions and their first derivatives
      if (distype != FE::CellType::nurbs8 and distype != FE::CellType::nurbs27)
      {
        Core::FE::shape_function_3d_deriv1(deriv, eleCoord(0), eleCoord(1), eleCoord(2), distype);
      }
      else
      {
        Core::FE::Nurbs::nurbs_get_3d_funct_deriv(funct, deriv, eleCoord, knots, weights, distype);
      }

      // get transposed of the jacobian matrix d x / d \xi
      xjm = 0;

      for (int inode = 0; inode < numnode; ++inode)
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) xjm(i, j) += deriv(i, inode) * xyze(j, inode);

      const double det = xjm.determinant();
      const double fac = intpoints.weight(iquad) * det;

      if (det <= 0.0) FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT: {}", det);

      vol += fac;
    }

    return vol;
  }

  /**
   * @brief Compute the geometric measure of an element.
   *
   * Returns the element length, area, or volume depending on the given
   * @p distype. The function dispatches to the corresponding element-specific
   * implementation for standard finite elements and NURBS elements.
   *
   * @tparam Matrixtype Matrix-like type containing nodal or control-point coordinates.
   *
   * @param distype Element cell type.
   * @param xyze Element coordinates.
   * @param knots Knot vectors for NURBS elements only.
   * @param weights Control-point weights for NURBS elements only.
   *
   * @return Length for 1D elements, area for 2D elements, or volume for 3D elements.
   *
   * @note Despite its name, this function also returns lengths and areas for
   * 1D and 2D element types.
   */
  template <class Matrixtype>
  double element_volume(const Core::FE::CellType distype, const Matrixtype& xyze,
      const std::vector<Core::LinAlg::SerialDenseVector>& knots = {},
      const Core::LinAlg::SerialDenseVector& weights = Core::LinAlg::SerialDenseVector{})
  {
    switch (distype)
    {
      case Core::FE::CellType::line2:
        return element_length<Core::FE::CellType::line2>(xyze);
      case Core::FE::CellType::line3:
        return element_length<Core::FE::CellType::line3>(xyze);
      case Core::FE::CellType::nurbs2:
        return element_length<Core::FE::CellType::nurbs2>(xyze, knots, weights);
      case Core::FE::CellType::nurbs3:
        return element_length<Core::FE::CellType::nurbs3>(xyze, knots, weights);
      case Core::FE::CellType::tri3:
        return element_area<Core::FE::CellType::tri3>(xyze);
      case Core::FE::CellType::tri6:
        return element_area<Core::FE::CellType::tri6>(xyze);
      case Core::FE::CellType::quad4:
        return element_area<Core::FE::CellType::quad4>(xyze);
      case Core::FE::CellType::quad8:
        return element_area<Core::FE::CellType::quad8>(xyze);
      case Core::FE::CellType::quad9:
        return element_area<Core::FE::CellType::quad9>(xyze);
      case Core::FE::CellType::nurbs4:
        return element_area<Core::FE::CellType::nurbs4>(xyze, knots, weights);
      case Core::FE::CellType::nurbs9:
        return element_area<Core::FE::CellType::nurbs9>(xyze, knots, weights);
      case Core::FE::CellType::hex8:
        return element_volume<Core::FE::CellType::hex8>(xyze);
      case Core::FE::CellType::hex20:
        return element_volume<Core::FE::CellType::hex20>(xyze);
      case Core::FE::CellType::hex27:
        return element_volume<Core::FE::CellType::hex27>(xyze);
      case Core::FE::CellType::tet4:
        return element_volume<Core::FE::CellType::tet4>(xyze);
      case Core::FE::CellType::tet10:
        return element_volume<Core::FE::CellType::tet10>(xyze);
      case Core::FE::CellType::wedge6:
        return element_volume<Core::FE::CellType::wedge6>(xyze);
      case Core::FE::CellType::wedge15:
        return element_volume<Core::FE::CellType::wedge15>(xyze);
      case Core::FE::CellType::pyramid5:
        return element_volume<Core::FE::CellType::pyramid5>(xyze);
      case Core::FE::CellType::nurbs8:
        return element_volume<Core::FE::CellType::nurbs8>(xyze, knots, weights);
      case Core::FE::CellType::nurbs27:
        return element_volume<Core::FE::CellType::nurbs27>(xyze, knots, weights);
      default:
        FOUR_C_THROW(
            "Element volume calculation is currently not implemented for the given cell type.");
    }
  }

}  // namespace Core::Geo


FOUR_C_NAMESPACE_CLOSE

#endif
