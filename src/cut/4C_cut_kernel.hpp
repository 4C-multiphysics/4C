// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_KERNEL_HPP
#define FOUR_C_CUT_KERNEL_HPP

#include "4C_config.hpp"

#include "4C_cut_enum.hpp"
#include "4C_cut_tolerance.hpp"
#include "4C_cut_utils.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_gauss.templates.hpp"
#include "4C_utils_cln_matrix_conversion.hpp"
#include "4C_utils_clnwrapper.hpp"
#include "4C_utils_mathoperations_cln.hpp"

#include <unordered_map>


// #define DEBUG_CUTKERNEL_OUTPUT
#define CUT_CLN_CALC


#ifdef DEBUG_CUTKERNEL_OUTPUT
#include "4C_cut_output.hpp"
#endif

FOUR_C_NAMESPACE_OPEN

namespace Cut::Kernel
{
  // functions to compare determinant to zero, e.g when computing the condition_number
  bool close_to_zero(const double a);

  bool close_to_zero(const Core::CLN::ClnWrapper& a);

  // Class to collects statistics about runs on double and cln in the cut intersection
  class CutKernelStatistics
  {
   public:
    static CutKernelStatistics& get_cut_kernel_statistics()
    {
      static CutKernelStatistics intersection_counter_;
      return intersection_counter_;
    }
    void double_intersection_counter() { double_int_++; };
    void double_distance_counter() { double_dist_++; };
    void cln_intersection_counter() { cln_int_++; };
    void cln_distance_counter() { cln_dist_++; };
    ~CutKernelStatistics()
    {
      std::cout << "\n\n =====================INTERSECTION "
                   "STATISTICS====================================\n\n";
      std::cout << "During compute intersection " << double_int_ << "/" << double_int_ + cln_int_
                << " was done on double only " << std::endl;
      std::cout << "During compute distance     " << double_dist_ << "/" << double_dist_ + cln_dist_
                << " was done on double only " << std::endl;
    }

   private:
    CutKernelStatistics() { /* blank */ };
    unsigned long long int double_int_;
    unsigned long long int double_dist_;
    unsigned long long int cln_int_;
    unsigned long long int cln_dist_;
  };

  /// Information about the location of the point on the surface
  class PointOnSurfaceLoc
  {
   public:
    PointOnSurfaceLoc() : within_side_(false), on_side_(false) {}

    PointOnSurfaceLoc(bool within_side, bool on_side) : within_side_(within_side), on_side_(on_side)
    {
    }

    bool within_side() { return within_side_; }

    bool on_side() { return on_side_; }

   private:
    bool within_side_;
    bool on_side_;
  };

  unsigned find_next_corner_point(const std::vector<Point*>& points, Core::LinAlg::Matrix<3, 1>& x1,
      Core::LinAlg::Matrix<3, 1>& x2, Core::LinAlg::Matrix<3, 1>& x3,
      Core::LinAlg::Matrix<3, 1>& b1, Core::LinAlg::Matrix<3, 1>& b2,
      Core::LinAlg::Matrix<3, 1>& b3, unsigned i);

  /// check if the given corner points can belong to a point1 "element"
  bool is_valid_point1(const std::vector<Point*>& corner_points);

  /// check if the given corner points can belong to a line2 element
  bool is_valid_line2(const std::vector<Point*>& corner_points);

  /// check if the given corner points can belong to a tri3 element
  inline bool is_valid_tri3(const std::vector<Point*>& points) { return points.size() == 3; }

  /// check if the given corner points can belong to a quad4 element
  bool is_valid_quad4(const std::vector<Point*>& points);

  template <int points>
  double find_l2_scaling(Core::LinAlg::Matrix<3, points>& xyze)
  {
    double scale = 0;
    Core::LinAlg::Matrix<3, 1> d;
    for (unsigned int i = 0; i < points; ++i)
    {
      Core::LinAlg::Matrix<3, 1> x1(&xyze(0, i), true);
      Core::LinAlg::Matrix<3, 1> x2(&xyze(0, (i + 1) % points), true);
      d.update(1, x2, -1, x1, 0);
      scale += d.norm2();
    }
    scale /= points;
    return scale;
  }

  /// get all edges adjacent to given local coordinates
  template <Core::FE::CellType side_type, class T, class FloatType>
  void get_edges_at(const T& xsi, std::vector<int>& edges_id, const FloatType& tol)
  {
    switch (side_type)
    {
      case Core::FE::CellType::tri3:
      {
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(1)) <=
            Core::MathOperations<typename T::scalar_type>::abs(tol(1)))
          edges_id.push_back(0);
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + xsi(1) - 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::sqrt(
                tol(0) * tol(0) * 0.25 + tol(1) * tol(1) * 0.25))
          edges_id.push_back(1);
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0)) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0)))
          edges_id.push_back(2);
        break;
      }
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad9:
      case Core::FE::CellType::nurbs9:
      {
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(1) + 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1)))
          edges_id.push_back(0);
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0)))
          edges_id.push_back(1);
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(1) - 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1)))
          edges_id.push_back(2);
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0)))
          edges_id.push_back(3);
        break;
      }
      case Core::FE::CellType::line2:
      {
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0)))
          edges_id.push_back(0);
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0)))
          edges_id.push_back(1);
        break;
      }
      default:
      {
        throw "Unknown/unsupported side type! \n";
        break;
      }
    }  // switch (sidetype)
  }

  /// get all nodes adjacent to given local coordinates
  template <Core::FE::CellType side_type, class T, class FloatType>
  void get_nodes_at(const T& xsi, std::vector<int>& nodes_id, const FloatType& tol)
  {
    switch (side_type)
    {
      case Core::FE::CellType::tri3:
      {
        if ((Core::MathOperations<typename T::scalar_type>::abs(xsi(1)) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1))) and
            (Core::MathOperations<typename T::scalar_type>::abs(xsi(0)) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0))))
          nodes_id.push_back(0);
        if ((Core::MathOperations<typename T::scalar_type>::abs(xsi(1)) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1))) and
            (Core::MathOperations<typename T::scalar_type>::abs(1.0 - xsi(0)) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0))))
          nodes_id.push_back(1);
        if (Core::MathOperations<typename T::scalar_type>::abs(1.0 - xsi(1)) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1)))
          nodes_id.push_back(2);
        break;
      }
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad9:
      case Core::FE::CellType::nurbs9:
      {
        if ((Core::MathOperations<typename T::scalar_type>::abs(xsi(1) + 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1))) and
            (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0))))
          nodes_id.push_back(0);
        if ((Core::MathOperations<typename T::scalar_type>::abs(xsi(1) + 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1))) and
            (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0))))
          nodes_id.push_back(1);
        if ((Core::MathOperations<typename T::scalar_type>::abs(xsi(1) - 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1))) and
            (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0))))
          nodes_id.push_back(2);
        if ((Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(1))) and
            (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + 1.0) <=
                Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0))))
          nodes_id.push_back(3);
        break;
      }
      case Core::FE::CellType::line2:
      {
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0)))
          nodes_id.push_back(0);
        if (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1.0) <=
            Core::MathOperations<typename FloatType::scalar_type>::abs(tol(0)))
          nodes_id.push_back(1);
        break;
      }
      default:
      {
        throw "Unknown/unsupported side type! \n";
        break;
      }
    }  // switch (sidetype)
  }

  /** \brief check if the given local point \c xsi of the side is on an edge
   *
   *  The function returns \TRUE if the given point lies on one of the edges
   *  of the side.
   *
   *  \param xsi (in) : local parameter space coordinates
   *  \param tol (in) : tolerance for the check (default: \c tol=REFERENCETOL) */
  template <Core::FE::CellType element_type, class T>
  bool at_edge(const T& xsi, const double& tol)
  {
    // sanity check
    if (xsi.Rows() < Core::FE::dim<element_type>)
      FOUR_C_THROW(
          "The given local coordinate has the wrong dimension!\n"
          "xsi.Rows() < Dim <===> {}<{}",
          xsi.Rows(), Core::FE::dim<element_type>);

    switch (element_type)
    {
      case Core::FE::CellType::line2:
      {
        return (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + 1) < tol or
                Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1) < tol);
        break;
      }
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad8:
      case Core::FE::CellType::quad9:
      case Core::FE::CellType::nurbs9:
      {
        return (Core::MathOperations<typename T::scalar_type>::abs(xsi(0) + 1) < tol or
                Core::MathOperations<typename T::scalar_type>::abs(xsi(1) + 1) < tol or
                Core::MathOperations<typename T::scalar_type>::abs(xsi(0) - 1) < tol or
                Core::MathOperations<typename T::scalar_type>::abs(xsi(1) - 1) < tol);
        break;
      }
      case Core::FE::CellType::tri3:
      case Core::FE::CellType::tri6:
      {
        return (Core::MathOperations<double>::abs(xsi(0) + 0) < tol or
                Core::MathOperations<double>::abs(xsi(1) + 0) < tol or
                Core::MathOperations<typename T::scalar_type>::abs(xsi(1) + xsi(0) - 1) < tol);
        break;
      }
      default:
        FOUR_C_THROW("unsupported element type: {}", Core::FE::cell_type_to_string(element_type));
        break;
    }
  }
  template <Core::FE::CellType element_type, class T>
  bool at_edge(const T& xsi)
  {
    return at_edge<element_type>(xsi, REFERENCETOL);
  }


  /// withinlimits with 3 manually specified (independent from each other) tolerances
  template <Core::FE::CellType element_type, class T, class FloatType, unsigned int dim>
  bool within_limits_split_quad(const T& xsi, const Core::LinAlg::Matrix<dim, 1, FloatType>& tol)
  {
    if (element_type == Core::FE::CellType::tri3)
    {
      return xsi(0) >= 0.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
             xsi(1) >= 0.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
             xsi(0) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
             xsi(1) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
             xsi(1) <= (1.0 - xsi(0)) + tol(2);
    }
    else
      FOUR_C_THROW("This function is only called for triangulated elements");
  }


  template <Core::FE::CellType element_type, class T, class FloatType, unsigned int dim>
  bool within_limits(const T& xsi, const Core::LinAlg::Matrix<dim, 1, FloatType>& tol)
  {
    if (xsi.m() < Core::FE::dim<element_type>)
      FOUR_C_THROW(
          "The given local coordinate has the wrong dimension!\n"
          "xsi.Rows() < eleDim <===> {}<{}",
          xsi.m(), Core::FE::dim<element_type>);

    switch (element_type)
    {
      // ---------------------------------------------------------------
      // 1-D elements
      // ---------------------------------------------------------------
      case Core::FE::CellType::line2:
      {
        return (
            xsi(0) >= -1.0 - Core::MathOperations<typename std::decay<decltype(tol(0))>::type>::abs(
                                 tol(0)) and
            xsi(0) <= 1.0 + Core::MathOperations<typename std::decay<decltype(tol(0))>::type>::abs(
                                tol(0)));
        break;
      }
      // ---------------------------------------------------------------
      // 2-D elements
      // ---------------------------------------------------------------
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad8:
      case Core::FE::CellType::quad9:
      case Core::FE::CellType::nurbs9:
      {
        return xsi(0) >=
                   -1.0 - Core::MathOperations<typename std::decay<decltype(tol(0))>::type>::abs(
                              tol(0)) and
               xsi(1) >=
                   -1.0 - Core::MathOperations<typename std::decay<decltype(tol(1))>::type>::abs(
                              tol(1)) and
               xsi(0) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
               xsi(1) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(1));
        break;
      }
      case Core::FE::CellType::tri3:
      case Core::FE::CellType::tri6:
      {
        return xsi(0) >= 0.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
               xsi(1) >= 0.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
               xsi(0) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
               xsi(1) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
               xsi(1) <=
                   (1.0 - xsi(0)) +
                       Core::MathOperations<typename T::scalar_type>::sqrt(
                           Core::MathOperations<typename T::scalar_type>::abs(tol(0)) *
                               Core::MathOperations<typename T::scalar_type>::abs(tol(0)) * 0.25 +
                           Core::MathOperations<typename T::scalar_type>::abs(tol(1)) *
                               Core::MathOperations<typename T::scalar_type>::abs(tol(1)) * 0.25);
        break;
      }
      // NOTE: Not sure if all the tolerances are set up correctly, especially points close to
      // the "\" diagonal line
      // ---------------------------------------------------------------
      // 3-D elements
      // ---------------------------------------------------------------
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::hex16:
      case Core::FE::CellType::hex20:
      case Core::FE::CellType::hex27:
      {
        return (xsi(0) >= -1.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
                xsi(1) >= -1.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
                xsi(2) >= -1.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(2)) and
                xsi(0) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
                xsi(1) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
                xsi(2) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(2)));
        break;
      }
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::tet10:
      {
        return (xsi(0) >= -Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
                xsi(1) >= -Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
                xsi(2) >= -Core::MathOperations<typename T::scalar_type>::abs(tol(2)) and
                xsi(0) + xsi(1) + xsi(2) <=
                    1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)));
        break;
      }
      case Core::FE::CellType::pyramid5:
      {
        return (xsi(0) >= -1.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
                xsi(1) >= -1.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
                xsi(2) >= -Core::MathOperations<typename T::scalar_type>::abs(tol(2)) and
                xsi(0) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
                xsi(1) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
                (Core::MathOperations<typename T::scalar_type>::abs(xsi(0)) >
                            Core::MathOperations<typename T::scalar_type>::abs(xsi(1))
                        ? (xsi(2) + Core::MathOperations<typename T::scalar_type>::abs(xsi(0)) <=
                              1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)))
                        : (xsi(2) + Core::MathOperations<typename T::scalar_type>::abs(xsi(1)) <=
                              1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)))));
        break;
      }
      case Core::FE::CellType::wedge6:
      case Core::FE::CellType::wedge15:
      {
        return (
            xsi(0) >= -Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
            xsi(1) >= -Core::MathOperations<typename T::scalar_type>::abs(tol(1)) and
            xsi(2) >= -1.0 - Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
            xsi(2) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)) and
            xsi(0) + xsi(1) <= 1.0 + Core::MathOperations<typename T::scalar_type>::abs(tol(0)));
        break;
      }
      default:
        FOUR_C_THROW("unsupported element type: {}", Core::FE::cell_type_to_string(element_type));
        break;
    }  // switch ( elementType )
    return false;
  }


  // template< Core::FE::CellType elementType, class T>

  /** \brief check if the local coordinate is within the specified limits
   *
   *  Returns \TRUE if the given point lies inside the element.
   *
   *  \param xsi (in) : local parameter space coordinates
   *  \param tol (in) : tolerance for the check (default: \c tol=REFERENCETOL) */
  template <Core::FE::CellType element_type, class T, class FloatType>
  bool within_limits(const T& xsi, const FloatType& tol)
  {
    if (xsi.m() < Core::FE::dim<element_type>)
      FOUR_C_THROW(
          "The given local coordinate has the wrong dimension!\n"
          "xsi.Rows() < eleDim <===> {}<{}",
          xsi.m(), Core::FE::dim<element_type>);

    switch (element_type)
    {
      // ---------------------------------------------------------------
      // 1-D elements
      // ---------------------------------------------------------------
      case Core::FE::CellType::line2:
      {
        return (xsi(0) >= -1.0 - tol and xsi(0) <= 1.0 + tol);
        break;
      }
      // ---------------------------------------------------------------
      // 2-D elements
      // ---------------------------------------------------------------
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad8:
      case Core::FE::CellType::quad9:
      case Core::FE::CellType::nurbs9:
      {
        return xsi(0) >= -1.0 - tol and xsi(1) >= -1.0 - tol and xsi(0) <= 1.0 + tol and
               xsi(1) <= 1.0 + tol;
        break;
      }
      case Core::FE::CellType::tri3:
      case Core::FE::CellType::tri6:
      {
        return xsi(0) >= 0.0 - tol and xsi(1) >= 0.0 - tol and xsi(0) <= 1.0 + tol and
               xsi(1) <= 1.0 + tol and xsi(1) <= (1.0 - xsi(0)) + tol;
        break;
      }
      // ---------------------------------------------------------------
      // 3-D elements
      // ---------------------------------------------------------------
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::hex16:
      case Core::FE::CellType::hex20:
      case Core::FE::CellType::hex27:
      {
        return (xsi(0) >= -1.0 - tol and xsi(1) >= -1.0 - tol and xsi(2) >= -1.0 - tol and
                xsi(0) <= 1.0 + tol and xsi(1) <= 1.0 + tol and xsi(2) <= 1.0 + tol);
        break;
      }
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::tet10:
      {
        return (xsi(0) >= -tol and xsi(1) >= -tol and xsi(2) >= -tol and
                xsi(0) + xsi(1) + xsi(2) <= 1.0 + tol);
        break;
      }
      case Core::FE::CellType::pyramid5:
      {
        return (xsi(0) >= -1.0 - tol and xsi(1) >= -1.0 - tol and xsi(2) >= -tol and
                xsi(0) <= 1.0 + tol and xsi(1) <= 1.0 + tol and
                (Core::MathOperations<typename T::scalar_type>::abs(xsi(0)) >
                            Core::MathOperations<typename T::scalar_type>::abs(xsi(1))
                        ? (xsi(2) + Core::MathOperations<typename T::scalar_type>::abs(xsi(0)) <=
                              1.0 + tol)
                        : (xsi(2) + Core::MathOperations<typename T::scalar_type>::abs(xsi(1)) <=
                              1.0 + tol)));
        break;
      }
      case Core::FE::CellType::wedge6:
      case Core::FE::CellType::wedge15:
      {
        return (xsi(0) >= -tol and xsi(1) >= -tol and xsi(2) >= -1.0 - tol and
                xsi(2) <= 1.0 + tol and xsi(0) + xsi(1) <= 1.0 + tol);
        break;
      }
      default:
        FOUR_C_THROW("unsupported element type: {}", Core::FE::cell_type_to_string(element_type));
        break;
    }  // switch ( elementType )
    return false;
  }


  template <Core::FE::CellType element_type, class T>
  bool within_limits(const T& xsi)
  {
    return within_limits<element_type>(xsi, REFERENCETOL);
  }

  /** \brief check if the local augmented coordinates are within the specified limits
   *
   *  */
  template <unsigned prob_dim, Core::FE::CellType side_type,
      unsigned dim_side = Core::FE::dim<side_type>, class T>
  bool within_limits_embedded_manifold(const T& xsi_aug, double tol, bool allow_dist, double tol2)
  {
    if (prob_dim == 1) FOUR_C_THROW("probDim is not allowed to be equal 1!");
    if (xsi_aug.m() != prob_dim)
      FOUR_C_THROW(
          "Wrong dimension of the augmented xsi input variable. \n"
          "The dimension is supposed to be equal to the problem dimension. \n"
          "The last coordinates are holding the distances from the global \n"
          "point to the side element. ");

    bool check = true;
    // check the distances if desired
    if (not allow_dist)
      for (unsigned r = dim_side; r < prob_dim; ++r)
        check = (check and (xsi_aug(r, 0) >= -tol2 and xsi_aug(r, 0) <= tol2) ? true : false);

    /* Since the first dimSide vector entries are the important ones,
     * we do not have to perform any extraction. */
    return (check and within_limits<side_type>(xsi_aug, tol));
  }
  template <unsigned prob_dim, Core::FE::CellType side_type, class T>
  bool within_limits_embedded_manifold(const T& xsi_aug, const bool& allow_dist, const double& tol2)
  {
    return Kernel::within_limits_embedded_manifold<prob_dim, side_type>(
        xsi_aug, REFERENCETOL, allow_dist, tol2);
  }

  /*! \brief Check whether three points are on the same line
   *  by checking whether the cross product is zero
   *
   */
  bool is_on_line(Point*& pt1, Point*& pt2, Point*& pt3, bool DeleteInlinePts = false);

  /*! \brief Check whether the polygon defined by the set of points is convex */
  std::vector<int> check_convexity(const std::vector<Point*>& ptlist, Cut::FacetShape& geomType,
      bool InSplit = true, bool DeleteInlinePts = false);

  // std::vector<double> EqnPlanePolygon( const std::vector<Point*>& ptlist, bool
  // DeleteInlinePts = false );


  /*! \brief Compute the equation of plane of this polygon using Newell's method */
  std::vector<double> eqn_plane_of_polygon(const std::vector<Point*>& ptlist);

  /*! \brief Compute the equation of plane of this polygon using Newell's method */
  std::vector<double> eqn_plane_of_polygon(const std::vector<std::vector<double>>& vertices);

  /*! \brief Find the equation of plane that contains these non-collinear points
   *
   *  It must be noted while using this function to find equation of facets,
   *  none of these 3 points must be a reflex (concave) point */
  std::vector<double> eqn_plane(Point*& pt1, Point*& pt2, Point*& pt3);

  /*! \brief Check whether the point named \c check is inside the triangle
   *  formed by tri */
  bool pt_inside_triangle(std::vector<Point*> tri, Point* check, bool DeleteInlinePts = false);

  /*! \brief Check whether the point \c check is inside the Quad */
  bool pt_inside_quad(std::vector<Point*> quad, Point* check);

  /*! \brief Return \TRUE if the points of the polygon are ordered clockwise
   *
   *  Polygon in 3D space is first projected into 2D plane, and the plane
   *  of projection is returned in projType.
   *
   */
  bool is_clockwise_ordered_polygon(std::vector<Point*> polyPoints, std::string& projPlane);

  /*! \brief Delete unnecessary in-line points.
   *
   *  If more than two points are on a line, all points except the end points
   *  are deleted. This is checked for all the lines for a facet. So once this is
   *  called the facet is free of more than 2 inline points
   *
   */
  void delete_inline_pts(std::vector<Point*>& poly);

  /*! \brief Returns true if at least 3 points are collinear
   *
   */
  bool have_inline_pts(std::vector<Point*>& poly);

  /*! \brief Finds tree points of the polygon which are not collinear
   *
   */
  std::vector<Point*> get3_noncollinear_pts(std::vector<Point*>& polyPoints);

  /*! \brief Find appropriate projection plane
   *
   *  In several cases, it is appropriate to project the surface in 3D space into appropriate
   *  coordinate plane. It is better to project over the plane which is max normal component
   *  because this will reduce the round-off error in further calculations.
   *
   */
  void find_projection_plane(std::string& projPlane, const std::vector<double>& eqn);

  /*! \brief Split a QUAD4 element into two TRI3 elements
   *
   *  Here it is important that the triangle is created in the same rotation
   *  as the QUAD4 is, to get normal in the same direction and therefore the
   *  same signed distance:
   *
   *  tri3_id = 0 ---> QUAD4 nodes = {0 1 2}
   *  tri3_id = 1 ---> QUAD4 nodes = {2 3 0}
   *
   *  \param xyze_quad4 (in) : nodal coordinates of a QUAD4 element
   *  \param tri3_id    (in) : id of the desired TRI3 element \f& id \in \{0,\; 1\}\f$
   *  \param xyze_tri3  (out): created/filled TRI3 nodal coordinate matrix
   *
   *  */
  template <class T1, class T2>
  void split_quad4_into_tri3(const T1& xyze_quad4, const unsigned& tri3_id, T2& xyze_tri3)
  {
    if (tri3_id > 1)
      FOUR_C_THROW(
          "A QUAD4 is supposed to be split into 2 TRI3 elements, \n"
          "therefore you have the choice between the tri3_id's 0 and 1. \n"
          "Nevertheless, you tried to access the id {}.",
          tri3_id);

    unsigned n0 = 2 * tri3_id;
    for (unsigned r = 0; r < static_cast<unsigned>(xyze_quad4.m()); ++r)
      for (unsigned c = 0; c < 3; ++c) xyze_tri3(r, c) = xyze_quad4(r, ((c + n0) % 4));
  };

  /*! \brief Get area of triangle in 3D space
   *
   */
  double get_area_tri(
      const std::vector<Point*>& poly, Core::LinAlg::Matrix<3, 1>* normalvec = nullptr);
  double get_area_tri(const double* p0_ptr, const double* p1_ptr, const double* p2_ptr,
      Core::LinAlg::Matrix<3, 1>* normalvec = nullptr);
  template <class T>
  double get_area_tri(const T& xyze, Core::LinAlg::Matrix<3, 1>* normalvec = nullptr)
  {
    if (xyze.m() != 3) FOUR_C_THROW("Currently unsupported element dimension!");
    if (xyze.n() != 3) FOUR_C_THROW("Wrong node number!");

    return getAreaTri(&xyze(0, 0), &xyze(0, 1), &xyze(0, 2), normalvec);
  }

  /*! \brief Get area of convex Quad in 3D space
   *
   *  Quad is split into two triangles and area of tri are summed up
   *
   *  \param poly (in): vector containing four pointers to the quad points
   *
   */
  double get_area_convex_quad(std::vector<Point*>& poly);

  enum NewtonStatus
  {
    converged = 1,
    unconverged = 0,
    failed = -1
  };

  /** \brief Build an adaptive combined Newton tolerance
   *
   *  First a scaled absolute tolerance is calculated, based on the
   *  inf-norm of the absolute coordinate values of the considered side
   *  element and point.
   *
   *  In a second step a relative tolerance is added, which accounts for the
   *  fact, that the point px_ is maybe far outside of the considered side.
   *  If the point is inside or near the side, the additional term will stay
   *  very low and will not distort the used tolerance too much.
   *
   *  */
  template <class T1, class T2, class T3>
  double adaptive_combined_newton_tolerance(

      const T1& xyze, const T2& px, const T3& initial_rhs)
  {
    /* --- Build the absolute tolerance */
    double tol = xyze.norm_inf();
    double linescale = px.norm_inf();
    if (linescale > tol) tol = linescale;
    tol *= LINSOLVETOL;

    /* --- Add the relative tolerance */
    tol += LINSOLVETOL * initial_rhs.norm_inf();

    return tol;
  }

  template <class T1, class T2, class T3>
  double adaptive_combined_newton_tolerance(

      const T1& xyze, const T2& px, const T3& initial_rhs, double)
  {
    return adaptive_combined_newton_tolerance(
        xyze, px, initial_rhs);  // forwarding to normal function
  }

#ifdef CUT_CLN_CALC

  /// computes adaptive precision for AdaptivePrecision strategies
  template <class T1, class T2, class T3>
  Core::CLN::ClnWrapper adaptive_combined_newton_tolerance(
      const T1& xyze, const T2& px, const T3& initial_rhs, Core::CLN::ClnWrapper&)
  {
    /* --- Build the absolute tolerance */
    // cln does not have built it pow function and using double will cause floating point
    // overflow, so we using manual cln conversion
    std::stringstream string_buffer;
    int nsize = Core::CLN::ClnWrapper::get_precision();
    string_buffer << nsize;
    // construct string that represent 1e^-(current_cln_precision - 2) explicitly
    // otherwise cln might think it too close to zero, and convert to short float zero and mess
    // the whole computation up
    std::string clnumstr = "0." + std::string(nsize - 2, '0') + "1" + "e+1_" + string_buffer.str();
    cln::cl_F cln_tol = clnumstr.c_str();
    // in order to make sure that we maintain proper precision
    Core::CLN::ClnWrapper cln_tol_cln =
        cln::cl_float(cln_tol, cln::float_format(Core::CLN::ClnWrapper::get_precision()));
    Core::CLN::ClnWrapper cln_linsolvetol =
        cln_tol_cln /
        Core::MathOperations<Core::CLN::ClnWrapper>::sqrt(Core::CLN::ClnWrapper(3.0)) *
        Core::CLN::ClnWrapper(10.0);  //  similarly as for double

    Core::CLN::ClnWrapper tol = xyze.norm_inf();
    Core::CLN::ClnWrapper linescale = px.norm_inf();

    if (linescale > tol) tol = linescale;

    tol *= cln_linsolvetol;
    /* --- Add the relative tolerance */
    tol += cln_linsolvetol * initial_rhs.norm_inf();

    if (tol == 0.0)
    {
      FOUR_C_THROW(
          "Newton tolerance is cut_kernel for CLN is equal to zero! This should not happen!");
    }

    return tol;
  }

#endif

  /// Convert Newton Status enumerator to string
  static inline std::string newton_status_to_string(const enum NewtonStatus& status)
  {
    switch (status)
    {
      case converged:
        return "CONVERGED";
      case unconverged:
        return "UNCONVERGED";
      case failed:
        return "FAILED";
      default:
        FOUR_C_THROW("Unknown Newton status!");
        exit(EXIT_FAILURE);
    }
    exit(EXIT_FAILURE);
  }

  /*--------------------------------------------------------------------------*/
  /// generic Newton algorithm
  template <class Strategy, unsigned dim = 3, unsigned maxiter = 50>
  class NewtonSolve : public Strategy
  {
   public:
    /// constructor
    NewtonSolve(Core::LinAlg::Matrix<dim, 1>& xsi, bool checklimits) : Strategy(xsi, checklimits) {}
#ifdef CUT_CLN_CALC
    /// required constructor for CLN
    NewtonSolve(Core::LinAlg::Matrix<dim, 1, Core::CLN::ClnWrapper>& xsi, bool checklimits)
        : Strategy(xsi, checklimits)
    {
    }
#endif

    bool solve()
    {
      this->setup_solve();

      for (unsigned iter = 0; iter < maxiter; ++iter)
      {
        this->setup_step(iter);

        switch (this->test_converged(iter))
        {
          // Newton iteration was successful
          case converged:
          {
            return true;
          }
          // Newton did not yet converge
          case unconverged:
          {
            break;
          }
          // Newton failed, thus we can stop here
          case failed:
          {
            return this->newton_failed();
          }
        }

        if (not this->linear_solve(iter))
        {
          return false;
        }

        if (not this->update(iter))
        {
          return false;
        }
      }

      return this->newton_failed();
    }

  };  // class class NewtonSolve

  /*--------------------------------------------------------------------------*/
  /// empty strategy for generic Newton algorithm
  class EmptyNewtonStrategy
  {
   public:
    void setup_solve() {}

    void setup_step(int iter) {}

    enum NewtonStatus test_converged(int iter) { return unconverged; }

    bool linear_solve(int iter) { return true; }

    bool update(int iter) { return true; }

    bool newton_failed() { return false; }

    double get_tolerance() { FOUR_C_THROW("Try to get tolerance from EmptyNewtonStrategy!"); }

    bool zero_area() { FOUR_C_THROW("Try to get zero_area Information from EmptyNewtonStrategy!"); }

    void write_to_gmsh(std::ofstream& file)
    {
      FOUR_C_THROW("Try write_to_gmsh() from EmptyNewtonStrategy!");
    }
  };  // class EmptyNewtonStrategy

  /*--------------------------------------------------------------------------*/
  /// Debug helper strategy for generic Newton algorithm
  template <class Strategy, unsigned dim = 3>
  class DebugNewtonStrategy : public Strategy
  {
   public:
    /// constructor
    DebugNewtonStrategy(Core::LinAlg::Matrix<dim, 1>& xsi, bool checklimits)
        : Strategy(xsi, checklimits)
    {
    }

    void setup_solve()
    {
      std::cout << "SetupSolve()\n";
      Strategy::setup_solve();
    }

    void setup_step(int iter)
    {
      std::cout << "SetupStep( iter = " << std::setw(2) << iter << " )\n";
      Strategy::setup_step(iter);
    }

    enum NewtonStatus test_converged(int iter)
    {
      enum NewtonStatus status = Strategy::test_converged(iter);
      std::cout << "TestConverged( iter = " << std::setw(2) << iter
                << " ) = " << newton_status_to_string(status) << "\n"
                << std::flush;
      return status;
    }

    bool linear_solve(int iter)
    {
      bool res = Strategy::linear_solve(iter);
      std::cout << "linear_solve( iter = " << std::setw(2) << iter
                << " )   = " << (res ? "SUCCESS" : "FAILED") << "\n"
                << std::flush;
      return res;
    }

    bool update(int iter)
    {
      bool res = Strategy::update(iter);
      std::cout << "update( iter = " << std::setw(2) << iter
                << " )        = " << (res ? "SUCCESS" : "FAILED") << "\n"
                << std::flush;
      return res;
    }

    bool newton_failed()
    {
      bool res = Strategy::newton_failed();
      std::cout << "NewtonFailed() = " << (res ? "SUCCESS" : "FAILED") << "\n" << std::flush;
      return res;
    }
  };  // class DebugNewtonStrategy

  // Static data storage of ComputePosition data members to inherit from
  template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
      unsigned num_nodes_element = Core::FE::num_nodes<element_type>,
      unsigned dim = Core::FE::dim<element_type>, typename FloatType = double>
  struct ComputePositionStaticMembers
  {
    /// nodal shape function values at the position xsi_
    static Core::LinAlg::Matrix<num_nodes_element, 1, FloatType> funct_;
    /** nodal first derivative shape function values at the position xsi_ */
    static Core::LinAlg::Matrix<prob_dim, num_nodes_element, FloatType> deriv1_;

    /** \brief (extended) jacobian matrix
     *
     *  Keep in mind, that the jacobian has to be extended if dim < probDim! */
    static Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> A_;
    /// right hand side vector b_ = -(px_ - x_(xsi_))
    static Core::LinAlg::Matrix<prob_dim, 1, FloatType> b_;
    /// newton increment (extended) parameter space coordinates
    static Core::LinAlg::Matrix<prob_dim, 1, FloatType> dx_;
  };

  // Data storage of ComputePosition data members to inherit from
  template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
      unsigned num_nodes_element = Core::FE::num_nodes<element_type>,
      unsigned dim = Core::FE::dim<element_type>, typename FloatType = double>
  struct ComputePositionNoStaticMembers
  {
    Core::LinAlg::Matrix<num_nodes_element, 1, FloatType> funct_;

    Core::LinAlg::Matrix<prob_dim, num_nodes_element, FloatType> deriv1_;

    Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> A_;

    Core::LinAlg::Matrix<prob_dim, 1, FloatType> b_;

    Core::LinAlg::Matrix<prob_dim, 1, FloatType> dx_;
  };



  /*--------------------------------------------------------------------------*/
  /** \brief strategy for position of point within element
   *
   *  inheritance diagram:
   *
   *  ComputePosition --> GenericComputePosition --> NewtonSolve
   *  --> ComputePositionStrategy --> EmptyNewtonStrategy
   *      ^^^^^^^^^^^^^^^^^^^^^^^
   */
  template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
      unsigned num_nodes_element = Core::FE::num_nodes<element_type>,
      unsigned dim = Core::FE::dim<element_type>, typename FloatType = double,
      template <bool, unsigned, Core::FE::CellType, unsigned, unsigned,
          typename> class MemberStoragePolicy = ComputePositionStaticMembers>
  class ComputePositionStrategy : public EmptyNewtonStrategy,
                                  private MemberStoragePolicy<debug, prob_dim, element_type,
                                      num_nodes_element, dim, FloatType>
  {
   public:
    /// constructor
    ComputePositionStrategy(Core::LinAlg::Matrix<dim, 1, FloatType>& xsi, bool checklimits)
        : xsi_(xsi), xyze_(nullptr), px_(nullptr), tol_(0.0)
    {
    }

    /** setup the computation
     *
     *  \param xyze (in) : global nodal coordinates
     *  \param px   (in) : global position vector of the searched parameter space point */
    void setup(const Core::LinAlg::Matrix<prob_dim, num_nodes_element, FloatType>& xyze,
        const Core::LinAlg::Matrix<prob_dim, 1, FloatType>& px)
    {
      // std::cout << "Entering setup for the position strategy" << std::endl;
      xyze_ = &xyze;
      px_ = &px;

      /* use tol_ first just to store the scaling for the tolerance ..
       * set it to the maximal absolute coordinate value of the involved element
       * or point!
       *
       * --> this defines accuracy of the intersection (for good conditioned
       *     linear systems) */

      tol_ = xyze_->norm_inf();
      FloatType linescale = px_->norm_inf();

      if (linescale > tol_) tol_ = linescale;

      tol_ *= LINSOLVETOL;

      if (debug)
      {
        std::cout << "\n=== ComputePosition ===\n";
        std::cout << "Element    = " << Core::FE::cell_type_to_string(element_type) << "\n";
        std::cout << "xyze_      = " << *xyze_;
        std::cout << "px_        = " << *px_;
        std::cout << "\n";
      }
    }

    /// get the local solution coordinates
    const Core::LinAlg::Matrix<dim, 1, FloatType>& local_coordinates() { return xsi_; }
    /// get the solution coordaintes in the CLN format
    /// initialize the solution variable to zero
    void setup_solve()
    {
      xsi_ = 0.0;
      b_ = 0.0;
      position_rhs(*xyze_, *px_, b_);
      tol_ = adaptive_combined_newton_tolerance(*xyze_, *px_, b_, xsi_(0, 0));
    }

    /** \brief evaluate the current right hand side
     *
     *  \f[
     *      b_ = - (px_ - x(xsi_))
     *  \f] */
    void setup_step(int iter)
    {
      if (iter > 0)
      {
        position_rhs(*xyze_, *px_, b_);
      }
    }

    /// calculate the position rhs
    void position_rhs(const Core::LinAlg::Matrix<prob_dim, num_nodes_element, FloatType>& xyze,
        const Core::LinAlg::Matrix<prob_dim, 1, FloatType>& px,
        Core::LinAlg::Matrix<prob_dim, 1, FloatType>& b)
    {
      Core::FE::shape_function<element_type>(xsi_, funct_);
      b = *px_;
      b.multiply(-1.0, xyze, funct_, 1.0);
    }

    /// check for convergence
    enum NewtonStatus test_converged(int iter)
    {
      FloatType residual = b_.norm2();

      if (debug)
      {
        std::cout << "within_limits             = "
                  << (within_limits<element_type>(xsi_) ? "TRUE" : "FALSE") << "\n"
                  << "rsd_.norm2()             = " << xsi_.norm2() << "\n"
                  << "dx_.norm2() / rsd.norm2()= "
                  << (iter != 0 ? dx_.norm2() / xsi_.norm2() : -1.0) << "\n"
                  << "b_.norm2() ( RHS )       = " << b_.norm2() << "\n"
                  << "b_.norm2() / rsd.norm2() = " << (iter != 0 ? b_.norm2() / xsi_.norm2() : -1.0)
                  << "\n"
                  << "tol_                     = " << tol_ << "\n"
                  << "rsd_ ( aka xsi_ )        = " << xsi_ << "b_ (RHS)                 = " << b_
                  << "\n"
                  << std::flush;
      }

      return (Core::MathOperations<FloatType>::abs(residual) < tol_ ? converged : unconverged);
    }

    /** \brief Build and solve the linear system
     *
     *  \param iter (in): Currently unused. Maximal number of linear iterations,
     *                    but we do a direct solve here.
     *
     * \f[
     * \underline{\underline{J}} \Delta \underline{\xi} = - (\underline{x}_{p} -
     * \underline{x(\underline{\xi})},\\ \text{ where } \underline{\underline{J}} =
     * \begin{pmatrix}
     * \frac{\partial x(\underline{\xi})}{\partial \xi} & \frac{\partial
     * x(\underline{\xi})}{\partial \eta} & \frac{\partial x(\underline{\xi})}{\partial \zeta}
     * \\
     * \frac{\partial x(\underline{\xi})}{\partial \xi} & \frac{\partial
     * y(\underline{\xi})}{\partial \eta} & \frac{\partial y(\underline{\xi})}{\partial \zeta}
     * \\ \frac{\partial x(\underline{\xi})}{\partial \xi} & \frac{\partial
     * z(\underline{\xi})}{\partial \eta} & \frac{\partial z(\underline{\xi})}{\partial \zeta}
     * \end{pmatrix}.
     * \f]
     *
     * For the special case that the element dimension is smaller than the
     * problem dimension (i.e. manifold), see the compute_distance classes. */
    bool linear_solve(int iter)
    {
      FloatType det = 0.0;
      // ToDo check if it is possible to switch completely to the manifold case,
      // without too much loss in performance.
      Core::FE::shape_function_deriv1<element_type>(xsi_, deriv1_);
      A_.multiply_nt(*xyze_, deriv1_);
      dx_ = 0.0;
      det = Core::LinAlg::gauss_elimination<true, prob_dim, FloatType>(A_, b_, dx_);

      if (debug)
      {
        std::cout << "det                        = " << det << "\n";
        std::cout << "dx_ ( solution increment ) = " << dx_;
        std::cout << "dx_.norm2()                = " << dx_.norm2() << "\n";
      }

      return Core::MathOperations<FloatType>::abs(det) >= LINSOLVETOL;
    }

    /// Update the solution parameter space coordinates
    bool update(int iter)
    {
      // update the solution variables (element dimension)
      for (unsigned i = 0; i < dim; ++i) xsi_(i, 0) += dx_(i, 0);

      return true;
    }

    bool newton_failed() { return false; }

    std::pair<bool, FloatType> condition_number()
    {
      Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> A_inv;
      FloatType det = A_.determinant();
      if (Kernel::close_to_zero(det))
      {
#ifdef DEBUG_CUTKERNEL_OUTPUT
        std::cout << " WARNING: Condition number is equal to infinity in the problem. Stopping "
                     "the increase in precision"
                  << std::endl;
#endif
        return std::make_pair(false, -1.0);
      }
      A_inv.invert(A_);
      FloatType cond = A_inv.norm2() * A_.norm2();
      return std::make_pair(true, cond);
    }
    /// get the Newton tolerance
    FloatType get_tolerance() const { return tol_; }

    FloatType get_local_tolerance(const FloatType& real_tolerance)
    {
      FOUR_C_THROW(
          "Not implemented! Have a look at  GetLocalTolerance for compute_distance strategy to "
          "find out the idea of implementation!");
      return real_tolerance;
    }

   private:
    /** parameter space coordinates (final result of the calculation)
     * corresponding to the global position px_. */
    Core::LinAlg::Matrix<dim, 1, FloatType>& xsi_;

    /// pointer to the global nodal positions
    const Core::LinAlg::Matrix<prob_dim, num_nodes_element, FloatType>* xyze_;
    /// global position vector (we are looking for the corresponding parameter coordinates)
    const Core::LinAlg::Matrix<prob_dim, 1, FloatType>* px_;

    using MemberStoragePolicy<debug, prob_dim, element_type, num_nodes_element, dim,
        FloatType>::funct_;

    using MemberStoragePolicy<debug, prob_dim, element_type, num_nodes_element, dim,
        FloatType>::deriv1_;

    using MemberStoragePolicy<debug, prob_dim, element_type, num_nodes_element, dim, FloatType>::A_;

    using MemberStoragePolicy<debug, prob_dim, element_type, num_nodes_element, dim, FloatType>::b_;

    using MemberStoragePolicy<debug, prob_dim, element_type, num_nodes_element, dim,
        FloatType>::dx_;

    /// adapted Newton tolerance
    FloatType tol_;

    // precision for cln
  };  // class ComputePositionStrategy


#ifdef DEBUG_MEMORY_ALLOCATION
  // shared data between all the template instantiations, for debugging memory allocations
  class AdaptiveKernelSharedData
  {
   public:
    static bool custom_allocator_run_;

   protected:
    static size_t cln_byte_size_[CLN_LIMIT_ITER];

    static std::unordered_map<size_t, int> memory_allocations_;

    static std::map<size_t, int> memory_allocations_intersection_;

    static std::map<size_t, int> memory_allocations_distance_;

    static std::map<size_t, int> memory_allocations_position_;

    static bool all_intersections_done_once_;

    static bool all_distance_done_once_;

    static bool all_position_done_once_;
    // specifies whether to run on the custor allocator now

    void update_memory_allocations(const std::unordered_map<size_t, int>& allocations)
    {
      for (auto const& [size, times] : allocations)
      {
        if (memory_allocations_[size] < times)
        {
          std::cout << "Size " << size << " was allocated " << times << " times " << std::endl;
          memory_allocations_[size] = times;
        }
      }
    }

    static int& get_position_count()
    {
      static int int_count = 0;
      return int_count;
    }
    static int& get_distance_count()
    {
      static int int_count = 0;
      return int_count;
    }
    static int& get_intersection_count()
    {
      static int int_count = 0;
      return int_count;
    }

    static int& position_counter()
    {
      static int counter = 0;
      return counter;
    }
    static int increase_position_counter() { return (++position_counter()); }

    static int& distance_counter()
    {
      static int counter = 0;
      return counter;
    }
    static int increase_distance_counter() { return (++distance_counter()); }

    static int& intersection_counter()
    {
      static int counter = 0;
      return counter;
    }
    static int increase_intersection_counter() { return (++intersection_counter()); }

    static void report_intersection_allocated()
    {
      size_t total = 0;
      std::cout << "REPORTING MAXIMUM MEMORY ALLOCATION IN INTERSECTION" << std::endl;
      for (std::map<size_t, int>::iterator it = memory_allocations_intersection_.begin();
          it != memory_allocations_intersection_.end(); ++it)
      {
        std::cout << "Size " << (*it).first << " was allocated " << (*it).second << " times "
                  << std::endl;
        if (memory_allocations_[it->first] < it->second)
        {
          memory_allocations_[it->first] = it->second;
        }
        total += (*it).second * (*it).first;
      }
      std::cout << "Totally maximum required allocationd " << total << " bytes" << std::endl;
    }

    static void report_position_allocated()
    {
      size_t total = 0;
      std::cout << "REPORTING MAXIMUM MEMORY ALLOCATION IN THE POSITION" << std::endl;
      for (std::map<size_t, int>::iterator it = memory_allocations_position_.begin();
          it != memory_allocations_position_.end(); ++it)
      {
        std::cout << "Size " << (*it).first << " was allocated " << (*it).second << " times "
                  << std::endl;
        if (memory_allocations_[it->first] < it->second)
        {
          memory_allocations_[it->first] = it->second;
        }
        total += (*it).second * (*it).first;
      }
      std::cout << "Totally maximum required allocationd " << total << " bytes" << std::endl;
    }

    static void report_distance_allocated()
    {
      size_t total = 0;
      std::cout << "REPORTING MAXIMUM MEMORY ALLOCATION IN THE DISTANCE" << std::endl;
      for (std::map<size_t, int>::iterator it = memory_allocations_distance_.begin();
          it != memory_allocations_distance_.end(); ++it)
      {
        std::cout << "Size " << (*it).first << " was allocated " << (*it).second << " times "
                  << std::endl;
        if (memory_allocations_[it->first] < it->second)
        {
          memory_allocations_[it->first] = it->second;
        }
        total += (*it).second * (*it).first;
      }
      std::cout << "Totally maximum required allocationd " << total << " bytes" << std::endl;
    }

    static void report_total_allocated()
    {
      size_t total = 0;
      std::cout << "REPORTING MAXIMUM MEMORY ALLOCATION" << std::endl;
      for (auto const& [size, times] : memory_allocations_)
      {
        std::cout << "Size " << size << " was allocated " << times << " times " << std::endl;
        total += times * size;
      }
      std::cout << "Totally maximum required allocationd " << total << " bytes" << std::endl;
    }
  };

#endif

#ifdef CUT_CLN_CALC

  /*--------------------------------------------------------------------------*/
  /** \brief Strategy for position of point within element for arbitrary precision
   *
   *  inheritance diagram:
   *
   *  ComputePosition --> GenericComputePosition --> NewtonSolve
   *  --> ComputePositionAdaptivePrecision --> EmptyNewtonStrategy
   *      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   */
  template <class Strategy, unsigned prob_dim, Core::FE::CellType element_type,
      unsigned num_nodes_element = Core::FE::num_nodes<element_type>,
      unsigned dim = Core::FE::dim<element_type>>
  class ComputePositionAdaptivePrecision : Strategy
#ifdef DEBUG_MEMORY_ALLOCATION
      ,
                                           AdaptiveKernelSharedData
#endif
  {
   public:
    ComputePositionAdaptivePrecision(Core::LinAlg::Matrix<dim, 1>& xsi)
        : Strategy(clnxsi_, true),
          xsi_(xsi),
          clnxsi_(Core::LinAlg::Initialization::set_zero),
          cond_infinity_(false)
    {
    }

    bool operator()(const Core::LinAlg::Matrix<prob_dim, num_nodes_element>& xyze,
        const Core::LinAlg::Matrix<prob_dim, 1>& px)
    {
      if (!((element_type == Core::FE::CellType::hex8) ||
              (element_type == Core::FE::CellType::tet4) ||
              (element_type == Core::FE::CellType::pyramid5) ||
              (element_type == Core::FE::CellType::quad4) ||
              (element_type == Core::FE::CellType::hex20) ||
              (element_type == Core::FE::CellType::tri3) ||
              (element_type == Core::FE::CellType::line2) ||
              (element_type == Core::FE::CellType::wedge6)))
      {
        FOUR_C_THROW(
            "This type of element {}  is not yet implemented for the CLN calculation. You are "
            "welcome to edit fem_shapefunctions.H file to fix it",
            Core::FE::cell_type_to_string(element_type).c_str());
      }

      bool conv;
      int iter = 0;
      int prec = CLN_START_PRECISION;  // standard initial precision
      Core::CLN::ClnWrapper err;
      Core::CLN::ClnWrapper cond_number;
      cond_infinity_ = false;  // test check if condition number is equal to infinity

#if DEBUG_MEMORY_ALLOCATION
      // Report total memory allocation done, for all possible template instantiations
      if (all_distance_done_once_ and all_intersections_done_once_ and all_position_done_once_ and
          (!custom_allocator_run_))
      {
        Cut::MemorySingleton::getinstance().ReportAllocated();
        report_intersection_allocated();
        report_position_allocated();
        report_distance_allocated();
        report_total_allocated();
        Cut::MemorySingleton::getinstance().set_state(1, memory_allocations_);
        custom_allocator_run_ = true;
      }
      if (first_run_) std::cout << "In cut position statistics is" << std::endl;
#endif
      do
      {
        Core::CLN::ClnWrapper::set_precision(prec);
#ifdef CUSTOM_MEMORY_ALLOCATOR
#if DEBUG_MEMORY_ALLOCATION
        // set current constainer for most frequenty byte size allocation for this iteration
        if (custom_allocator_run_)
#endif
          Cut::MemorySingleton::getinstance().get_memory_pool_allocator().SetCurrent(
              cln_byte_size_[iter]);
#endif

#if DEBUG_MEMORY_ALLOCATION
        // Report memory allocation of the class if doing this for the first time
        if (first_run_)
        {
          if (Cut::MemorySingleton::getinstance().IsRecording())
          {
            update_memory_allocations(Cut::MemorySingleton::getinstance().GetMemoryPattern());
            Cut::MemorySingleton::getinstance().StopRecord();
          }
          Cut::MemorySingleton::getinstance().StartRecord();
        }
#endif
        // conversion to correspondent precision
        Core::CLN::conv_double_cln(xyze, clnxyze_, prec);
        Core::CLN::conv_double_cln(px, clnpx_, prec);

        this->setup(clnxyze_, clnpx_);
        conv = this->solve();

        // safety check of precision loss
        cln::float_format_t prec_beg = cln::float_format(clnxyze_(0, 0).Value());
        for (unsigned int i = 0; i < dim; ++i)
        {
          cln::float_format_t prec_end = cln::float_format(clnxsi_(i, 0).Value());
          if (prec_beg != prec_end)
          {
            FOUR_C_THROW(
                "There is a loss of cln-precision during computation of intersection. "
                "Something is wrong with the conversion");
          }
        }

        std::pair<bool, Core::CLN::ClnWrapper> cond_pair = this->condition_number();
        cond_number = cond_pair.second;
        if (not cond_pair.first)
        {
#ifdef DEBUG_CUTKERNEL_OUTPUT
          std::cout << "Condition number of this problem is equal to infinity" << std::endl;
#endif
          cond_infinity_ = true;
        }

        err = compute_error(xyze, px, this->local_coordinates(), CLN_REFERENCE_PREC) * cond_number;
        // get location taking error into account
        //  NOTE: Here one might need to get tolerance in the local coordinates, similarly as
        //  done
        // in the compute_distance and ComputeIntersection
        //  location_ = GetPreciseLocation<elementType>( clnxsi_, err );

#if DEBUG_MEMORY_ALLOCATION
        update_memory_usage(prec, iter);
#endif

        prec += CLN_INCREMENT_STEP;
        iter++;

      } while (((!conv) || (err > CLN_LIMIT_ERROR) || (cond_infinity_)) &&
               (iter < CLN_LIMIT_ITER) && (prec < CLN_LIMIT_PREC));


      // safely convert the values back
      Core::CLN::conv_cln_double(clnxsi_, xsi_);
      if ((not conv) && (err < CLN_LIMIT_ERROR) && (err > 0.0)) conv = true;

#ifdef CUSTOM_MEMORY_ALLOCATOR
      if (first_run_)
      {
        const int num_inter = 3;
        if ((++get_position_count()) == num_inter)
        {
          all_position_done_once_ = true;
          std::cout << "All possible template position are done!" << std::endl;
        }
        first_run_ = false;
      }
#endif
      return conv;
    }

    const Core::LinAlg::Matrix<dim, 1, Core::CLN::ClnWrapper>& local_coordinates()
    {
      return Strategy::local_coordinates();
    }

    std::pair<bool, Core::CLN::ClnWrapper> condition_number()
    {
      return Strategy::condition_number();
    }

    Core::CLN::ClnWrapper get_tolerance() const { return Strategy::get_tolerance(); }

    PointOnSurfaceLoc get_side_location() { return location_; }

   private:
    // Evaluate difference between initial global coordinates passed to ComputePosition and
    // and global coordinates based on loc coordinates calculated in the ComputePosition,
    // using conversion to very high reference precision
    Core::CLN::ClnWrapper compute_error(
        const Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<element_type>>& ref_shape_xyz,
        const Core::LinAlg::Matrix<prob_dim, 1>& glob_init,
        const Core::LinAlg::Matrix<dim, 1, Core::CLN::ClnWrapper>& loc_calc, int prec)
    {
      unsigned int prev_prec = Core::CLN::ClnWrapper::get_precision();
      Core::CLN::ClnWrapper::set_precision(prec);
      // Converting input data into higher precision floating point format
      Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<element_type>, Core::CLN::ClnWrapper>
          cln_ref_shape_xyz;
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> cln_glob_init;
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> cln_loc_calc;

      Core::CLN::update_precision(loc_calc, cln_loc_calc, prec);
      Core::CLN::conv_double_cln(ref_shape_xyz, cln_ref_shape_xyz, prec);
      Core::CLN::conv_double_cln(glob_init, cln_glob_init, prec);

      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> cln_glob_calc;
      Core::LinAlg::Matrix<Core::FE::num_nodes<element_type>, 1, Core::CLN::ClnWrapper> shapeFunct;
      Core::FE::shape_function<element_type>(cln_loc_calc, shapeFunct);
      for (unsigned int inode = 0; inode < Core::FE::num_nodes<element_type>; ++inode)
      {
        for (unsigned int isd = 0; isd < prob_dim; ++isd)
        {
          cln_glob_calc(isd) += cln_ref_shape_xyz(isd, inode) * shapeFunct(inode);
        }
      }
      // evaluating error by comparison with global position, that was set up initially and the
      // one that we obtained from the ComputePosition evaluation and the interpolation
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> diff_vec;
      for (unsigned int i = 0; i < prob_dim; ++i) diff_vec(i) = cln_glob_calc(i) - cln_glob_init(i);
      // resetting precision to previous
      Core::CLN::ClnWrapper::set_precision(prev_prec);
      return diff_vec.norm2();
    }


    // Update global maximum number of allocations ( number of containers of particular byte
    // size) if allocations on this precisions were more. Also set what was the most frequent
    // size of allocation for the current iteration
    void update_memory_usage(int prec, int iter)
    {
#if DEBUG_MEMORY_ALLOCATION
      if (first_run_)
      {
        Cut::MemorySingleton::getinstance().StopRecord();

        std::unordered_map<size_t, int>& allocation_map =
            Cut::MemorySingleton::getinstance().GetMemoryPattern();

        // update the global number of allocations to the maximum
        int max_num = 0;
        int size_max_num = 0;
        for (std::unordered_map<size_t, int>::iterator it = allocation_map.begin();
            it != allocation_map.end(); ++it)
        {
          size_t size = it->first;
          if (memory_allocations_position_[size] < allocation_map[size])
          {
            memory_allocations_position_[size] = allocation_map[size];
          }
          if (allocation_map[size] > max_num)
          {
            size_max_num = size;
            max_num = allocation_map[size];
          }
        }

        if (size_max_num != 0)
        {
          cln_sizes_[iter] = size_max_num;
          std::cout << "Size for the precision " << prec << " is " << size_max_num << std::endl;
          std::cout << "It is allocated " << max_num << " times " << std::endl;
        }
        else
          FOUR_C_THROW("This should not be possible!");

        Cut::MemorySingleton::getinstance().ResetAllocated();
      }
#endif
    }

   private:
    // variable for cln computation
    Core::LinAlg::Matrix<prob_dim, num_nodes_element, Core::CLN::ClnWrapper> clnxyze_;

    Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> clnpx_;

    // reference to the double TMATRIX
    Core::LinAlg::Matrix<dim, 1>& xsi_;

    Core::LinAlg::Matrix<dim, 1, Core::CLN::ClnWrapper> clnxsi_;

    // if condition number if infinity or not
    bool cond_infinity_;

    // determines location of this point with respect to the edge and side that cut it
    PointOnSurfaceLoc location_;

    static bool first_run_;

    // sizes of cln variables with different preciion
    static size_t cln_sizes_[CLN_LIMIT_ITER];
  };

#endif

  /*--------------------------------------------------------------------------*/
  /** \brief generic strategy for position of point within element
   *
   *  inheritance diagram:
   *
   *  ComputePosition --> GenericComputePosition --> NewtonSolve
   *                      ^^^^^^^^^^^^^^^^^^^^^^
   *  --> ComputePositionStrategy --> EmptyNewtonStrategy
   */
  template <class Strategy, unsigned prob_dim, Core::FE::CellType element_type,
      bool compute_cln = false, unsigned num_nodes_element = Core::FE::num_nodes<element_type>,
      unsigned dim = Core::FE::dim<element_type>>
  class GenericComputePosition : Strategy
  {
   public:
    GenericComputePosition(Core::LinAlg::Matrix<dim, 1>& xsi)
        : Strategy(xsi, true)
#ifdef CUT_CLN_CALC
          ,
          xsi_(xsi)
#endif
    {
    }

    bool operator()(const Core::LinAlg::Matrix<prob_dim, num_nodes_element>& xyze,
        const Core::LinAlg::Matrix<prob_dim, 1>& px)
    {
      bool conv;
      this->setup(xyze, px);
      conv = this->solve();
#ifdef CUT_CLN_CALC
      if (compute_cln)  // this is not used up to now!
      {
        {
          ComputePositionAdaptivePrecision<
              NewtonSolve<ComputePositionStrategy<false, prob_dim, element_type, num_nodes_element,
                              dim, Core::CLN::ClnWrapper, ComputePositionNoStaticMembers>,
                  dim>,
              prob_dim, element_type>
              clncalc(xsi_);

          bool clnsolver = clncalc(xyze, px);
          conv = clnsolver;
          location_ = clncalc.get_side_location();
          Core::CLN::ClnWrapper::reset_precision();
        }
#ifdef CUSTOM_MEMORY_ALLOCATOR
        Cut::MemorySingleton::getinstance().Finalize();
#endif
      }
#endif
      return conv;
    }

    const Core::LinAlg::Matrix<dim, 1>& local_coordinates()
    {
      return Strategy::local_coordinates();
    }

    std::pair<bool, double> condition_number() { return Strategy::condition_number(); }

    double get_tolerance() const { return Strategy::get_tolerance(); }
#ifdef CUT_CLN_CALC

    PointOnSurfaceLoc get_side_location() { return location_; }

   private:
    PointOnSurfaceLoc location_;

    // touched edges ids
    static std::vector<int> touched_edges_ids_;

    // to hold the reference to cln calculation object
    Core::LinAlg::Matrix<dim, 1>& xsi_;
#endif
  };  // class GenericComputePosition

  /*--------------------------------------------------------------------------*/
  /** \brief most derived strategy class for position of point within element
   *
   *  inheritance diagram:
   *
   *  ComputePosition --> GenericComputePosition --> NewtonSolve
   *  ^^^^^^^^^^^^^^^
   *  --> ComputePositionStrategy --> EmptyNewtonStrategy
   */
  template <unsigned prob_dim, Core::FE::CellType element_type, bool compute_cln = false,
      unsigned num_nodes_element = Core::FE::num_nodes<element_type>,
      unsigned dim = Core::FE::dim<element_type>>
  class ComputePosition
      : public GenericComputePosition<
            NewtonSolve<ComputePositionStrategy<false, prob_dim, element_type>, dim>, prob_dim,
            element_type, compute_cln>
  {
   public:
    ComputePosition(Core::LinAlg::Matrix<dim, 1>& xsi)
        : GenericComputePosition<
              NewtonSolve<ComputePositionStrategy<false, prob_dim, element_type, num_nodes_element>,
                  dim>,
              prob_dim, element_type, compute_cln>(xsi)
    {
      if (dim != prob_dim)
        FOUR_C_THROW(
            "You called the wrong object! Use the compute_distance class "
            "for the embedded case.");
    }
  };  // class ComputePosition

  /*--------------------------------------------------------------------------*/
  template <unsigned prob_dim, Core::FE::CellType element_type,
      unsigned num_nodes_element = Core::FE::num_nodes<element_type>,
      unsigned dim = Core::FE::dim<element_type>>
  class DebugComputePosition
      : public GenericComputePosition<
            NewtonSolve<
                DebugNewtonStrategy<
                    ComputePositionStrategy<true, prob_dim, element_type, num_nodes_element>, dim>,
                dim>,
            prob_dim, element_type>
  {
   public:
    DebugComputePosition(Core::LinAlg::Matrix<dim, 1>& xsi)
        : GenericComputePosition<
              NewtonSolve<DebugNewtonStrategy<ComputePositionStrategy<true, prob_dim, element_type,
                                                  num_nodes_element>,
                              dim>,
                  dim>,
              prob_dim, element_type>(xsi)
    {
    }
  };  // class DebugComputePosition

  // Static storage class of compute_distance data members to inherit from
  template <bool debug, unsigned prob_dim, Core::FE::CellType side_type,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>, typename FloatType = double>
  struct ComputeDistanceStaticMembers
  {
    /// nodal shape function values at \c xsi
    static Core::LinAlg::Matrix<num_nodes_side, 1, FloatType> sideFunct_;

    /** nodal 1-st derivative values at \c xsi
     *
     *  \remark This is an augmented derivatives matrix. The actual
     *  dimension is \f$ dim \times numNodesSide \f$. The remaining
     *  entries are filled with zeros. This is due to the use of an
     *  UTILS function, which expects this input. */
    static Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType> sideDeriv1_;

    /** \brief nodal 2-nd derivatives at \c xsi
     *
     *  1-D case: \f$ x_{,\xi \xi}
     *                \;\in \mathbb{R}^{1 \times N}\f$,
     *
     *  2-D case: \f$ \left( x_{,\xi \xi}, \;x_{,\eta \eta}, \;
     *                       x_{,\xi\eta} \right)
     *                \;\in \mathbb{R}^{3 \times N} \f$ */
    static Core::LinAlg::Matrix<2 * dim_side - 1, num_nodes_side, FloatType> sideDeriv2_;

    /// complete linearization matrix
    static Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> A_;
    /// auxiliary matrix
    static Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> B_;
    /** auxiliary matrix holding 2-nd derivatives at \c xsi_
     *  w.r.t. the parameter space coordinates */
    static Core::LinAlg::Matrix<prob_dim, 2 * dim_side - 1, FloatType> C_;
    /// right hand side vector
    static Core::LinAlg::Matrix<prob_dim, 1, FloatType> b_;
    /** \brief solution increment
     *
     * (parameter space coordinates of the side + distance increment) */
    static Core::LinAlg::Matrix<prob_dim, 1, FloatType> dx_;
    /// (unscaled) normal vectors
    static Core::LinAlg::Matrix<prob_dim, 2, FloatType> N_;

    static Core::LinAlg::Matrix<prob_dim, 2, FloatType> nvec_;  // proper normal vector
  };

  // Static storage class of compute_distance data members to inherit from
  template <bool debug, unsigned prob_dim, Core::FE::CellType side_type,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>, typename FloatType = double>
  struct ComputeDistanceNoStaticMembers
  {
    Core::LinAlg::Matrix<num_nodes_side, 1, FloatType> sideFunct_;

    Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType> sideDeriv1_;

    Core::LinAlg::Matrix<2 * dim_side - 1, num_nodes_side, FloatType> sideDeriv2_;

    Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> A_;

    Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> B_;

    Core::LinAlg::Matrix<prob_dim, 2 * dim_side - 1, FloatType> C_;

    Core::LinAlg::Matrix<prob_dim, 1, FloatType> b_;

    Core::LinAlg::Matrix<prob_dim, 1, FloatType> dx_;

    Core::LinAlg::Matrix<prob_dim, 2, FloatType> N_;

    Core::LinAlg::Matrix<prob_dim, 2, FloatType> nvec_;
  };

  template <bool debug, unsigned prob_dim, Core::FE::CellType side_type,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>, typename FloatType = double,
      template <bool, unsigned, Core::FE::CellType, unsigned, unsigned,
          typename> class MemberStoragePolicy = ComputeDistanceStaticMembers>
  class ComputeDistanceStrategy
      : public EmptyNewtonStrategy,
        private MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>
  {
   public:
    ComputeDistanceStrategy(Core::LinAlg::Matrix<prob_dim, 1, FloatType>& xsi, bool checklimits)
        : xsi_(xsi),
          // we do initializing of distance in SetupSolve, because xsi could be from derived
          // class and not initialize at this point yet
          distance_(nullptr),
          xyze_side_(nullptr),
          px_(nullptr),
          tol_(0.0),
          zeroarea_(false)
    {
    }

    /** \brief Setup the distance calculation problem
     *
     *  \param xyze_side (in) : global coordinates of the considered side
     *  \param px        (in) : global coordinates of the considered point */
    void setup(const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, 1, FloatType>& px, bool zeroarea)
    {
      zeroarea_ = zeroarea;
      xyze_side_ = &xyze_side;
      px_ = &px;

      if (debug)
      {
        std::cout << "\n=== compute_distance ===\n";
        std::cout << "Side       = " << Core::FE::cell_type_to_string(side_type) << "\n";
        std::cout << "xyze_side_ = " << *xyze_side_;
        std::cout << "px_        = " << *px_;
        std::cout << "\n";
      }
    }
    /// return the absolute value of the distance of the point \c px_ to the given side
    FloatType distance() const
    {
      switch (prob_dim - dim_side)
      {
        case 1:
          return Core::MathOperations<FloatType>::abs(distance_[0]);
        case 2:
          return Core::MathOperations<FloatType>::sqrt(
              distance_[0] * distance_[0] + distance_[1] * distance_[1]);
        default:
          FOUR_C_THROW("Unsupported probDim and dimSide combination!");
          exit(EXIT_FAILURE);
      }
      exit(EXIT_FAILURE);
    }

    /// return the signed distance of the point px_ to the given side
    const FloatType* signed_distance() const { return distance_; }

    /// return the local solution vector (parameter space coordinates + distance)
    const Core::LinAlg::Matrix<prob_dim, 1, FloatType>& local_coordinates() const { return xsi_; }

    /// transform tolerance from global to local coordinates
    bool get_local_tolerance(const FloatType& global_tolerance,
        Core::LinAlg::Matrix<dim_side, 1, FloatType>& scaled_tolerance)

    {
      Core::LinAlg::Matrix<prob_dim, 1, FloatType> real_tolerance;
      // set all values of the vector to be equal to the global tolerance
      real_tolerance = global_tolerance;
      // extract part of the side from the Jacobian
      Core::LinAlg::Matrix<prob_dim, dim_side, FloatType> A;
      for (unsigned int row = 0; row < prob_dim; ++row)
      {
        for (unsigned int col = 0; col < dim_side; ++col) A(row, col) = A_(row, col);
      }

      Core::LinAlg::Matrix<dim_side, dim_side, FloatType> TN_inv;
      Core::LinAlg::Matrix<dim_side, prob_dim, FloatType> aux;
      TN_inv.multiply_tn(A, A);


      // we will not be able to find inverse
      FloatType det = TN_inv.determinant();
      if (Kernel::close_to_zero(det)) return false;


      TN_inv.invert();
      aux.multiply_nt(TN_inv, A);
      scaled_tolerance.multiply(aux, real_tolerance);

      return true;
    }

    /// return isInfinite and possibly floatType
    std::pair<bool, FloatType> condition_number()
    {
      Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> A_inv;
      FloatType det = A_.determinant();
      if (Kernel::close_to_zero(det))
      {
#ifdef DEBUG_CUTKERNEL_OUTPUT
        std::cout << " WARNING: Condition number is equal to infinity in the problem. Stopping "
                     "the increase in precision"
                  << std::endl;
#endif
        return std::make_pair(false, -1.0);
      }
      A_inv.invert(A_);
      FloatType cond = A_inv.norm2() * A_.norm2();
      return std::make_pair(true, cond);
    }

    /// set the initial solution vector to zero
    void setup_solve()
    {
      xsi_ = 0.0;
      distance_ = xsi_.data() + dim_side;
      // evaluate initial rhs value (w/o distance contribution)
      distance_rhs(*xyze_side_, *px_, b_);
      tol_ = adaptive_combined_newton_tolerance(*xyze_side_, *px_, b_, xsi_(0, 0));
    }

    /** \brief Setup routine for a new Newton step
     *
     *  Setup the system of equations. */
    void setup_step(int iter)
    {
      if (iter > 0)
      {
        distance_rhs(*xyze_side_, *px_, b_);
      }

      // build the linear system of equations

      distance_system(*xyze_side_, *px_, distance_, A_, B_, C_, N_, b_);
    }
    /// compute the rhs value ( w/o distance contributions )
    void distance_rhs(const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, 1, FloatType>& px,
        Core::LinAlg::Matrix<prob_dim, 1, FloatType>& b)
    {
      Core::LinAlg::Matrix<dim_side, 1, FloatType> xsi_side(xsi_.data(), true);
      Core::FE::shape_function<side_type>(xsi_side, sideFunct_);
      b = px;
      b.multiply(-1.0, xyze_side, sideFunct_, 1.0);
    }

    /// Test the stop / convergence criterion of the Newton scheme
    enum NewtonStatus test_converged(int iter)
    {
      if (debug)
      {
        std::cout << "within_limits             = "
                  << (within_limits<side_type>(xsi_) ? "TRUE" : "FALSE") << "\n"
                  << "rsd_.norm2()             = " << xsi_.norm2() << "\n"
                  << "dx_.norm2() / rsd.norm2()= "
                  << (iter != 0 ? dx_.norm2() / xsi_.norm2() : -1.0) << "\n"
                  << "b_.norm2() ( RHS )       = " << b_.norm2() << "\n"
                  << "b_.norm2() / rsd.norm2() = " << (iter != 0 ? b_.norm2() / xsi_.norm2() : -1.0)
                  << "\n"
                  << "tol_                     = " << tol_ << "\n"
                  << "rsd_ ( aka xsi_ )        = " << xsi_ << "b_ (RHS)                 = " << b_
                  << "\n"
                  << std::flush;
      }

      if (zeroarea_)
      {
        return failed;
      }

      FloatType residual = b_.norm2();

      if (residual < tol_)
      {
        // actually I'm not sure why we are doing this here... ( hiermeier 08/16 )
        tol_ *= std::sqrt(static_cast<double>(prob_dim));
        return converged;
      }
      else
        return unconverged;
    }

    /** solve the linear system:
     *  \f[
     *      A \; \Delta \xi = b
     *  \f] */
    bool linear_solve(int iter)
    {
      dx_ = 0.0;
      if (zeroarea_) return false;
      if (debug)
      {
        std::cout << "Matrix A  is " << A_ << std::endl;
        std::cout << "Matrix dx is  " << dx_ << std::endl;
        std::cout << "Matrix b  is  " << b_ << std::endl;
      }

      FloatType det = Core::LinAlg::gauss_elimination<true, prob_dim, FloatType>(A_, b_, dx_);

      if (debug)
      {
        std::cout << "det                        = " << det << "\n";
        std::cout << "dx_ ( solution increment ) = " << dx_;
        std::cout << "dx_.norm2()                = " << dx_.norm2() << "\n";
      }

      return (det != 0.0);  // here det = 0 will just happen if the surface element is
                            // distorted!
    }

    /// update the local coordinates solution vector
    bool update(int iter)
    {
      xsi_ += dx_;
      return true;
    }

    /// fall back routine, if the Newton failed
    bool newton_failed()
    {
      tol_ = b_.norm2();
#ifdef DEBUG_CUTKERNEL_OUTPUT
      std::stringstream str;
      str << "compute_distance: Newton scheme did not converge:\n"
          << std::setprecision(16) << (*xyze_side_) << (*px_) << xsi_;

      std::string filename = Output::GenerateGmshOutputFilename(".NewtonFailed_distance.pos");
      std::ofstream file(filename.c_str());
      write_to_gmsh(file);
      file.close();
#endif
      return false;
    }

    /// Gmsh debug output
    void write_to_gmsh(std::ofstream& file)
    {
      file.precision(32);  // higher precision!
      char elementType;
      switch (num_nodes_side)
      {
        case 3:
          elementType = 'T';
          break;
        case 4:
          elementType = 'Q';
          break;
        default:
          FOUR_C_THROW(
              "unsupported element type in gmsh_side_dump."
              " Please feel free to extend the functionality if necessary.");
      }

      file << "View \""
           << "Side"
           << "\" {\n";
      {
        file << "S" << elementType << "(";
        for (unsigned i = 0; i < num_nodes_side; ++i)
        {
          if (i != 0) file << ",";
          file << (*xyze_side_)(0, i) << "," << (*xyze_side_)(1, i) << "," << (*xyze_side_)(2, i);
        }
        file << "){";
        for (unsigned i = 0; i < num_nodes_side; ++i)
        {
          if (i != 0) file << ",";
          file << "1";
        }
        file << "};\n";
      }
      file << "};\n";

      // calculate the points
      Core::LinAlg::Matrix<num_nodes_side, 1, FloatType> surface;

      Core::FE::shape_function_2d(surface, xsi_(0), xsi_(1), side_type);

      Core::LinAlg::Matrix<prob_dim, 1, FloatType> x1;

      x1.multiply(*xyze_side_, surface);

      file << "View \""
           << "Point"
           << "\" {\n";
      {
        file << "SP (";
        file << (*px_)(0, 0) << "," << (*px_)(1, 0) << "," << (*px_)(2, 0);
        file << "){";
        file << 1;
        file << "};\n";

        file << "SP (";
        file << x1(0, 0) << "," << x1(1, 0) << "," << x1(2, 0);
        file << "){";
        file << 2;
        file << "};\n";
      }
      file << "};";
    }

    /// get normal vectors
    const Core::LinAlg::Matrix<prob_dim, 2, FloatType>& get_normal_vector() { return nvec_; }

    /// get the adapted Newton tolerance
    FloatType get_tolerance() const { return tol_; }

    /// get actual reached residual
    FloatType get_residual_l2_norm() const { return b_.norm2(); }

    /// return \TRUE if the side metric vectors is zero
    bool zero_area() const { return zeroarea_; }

   private:
    enum NormalPlane
    {
      normal_in_xy_plane,
      normal_in_yz_plane,
      normal_plane_undefined
    };

    enum NormalPlane detect_normal_plane(
        const Core::LinAlg::Matrix<prob_dim, 1, FloatType>& n) const
    {
      if (prob_dim < 3) FOUR_C_THROW("This function makes only sense for the 3-D case!");

      if (n(2) == 0.0)
        return normal_in_xy_plane;
      else if (n(0) == 0.0)
        return normal_in_yz_plane;

      FOUR_C_THROW("Couldn't detect a feasible plane for the given normal vector!");
      exit(EXIT_FAILURE);
    }

    /** \build Build the linear system of equations for the Newton scheme
     *
     *  Right-hand-side:
     *  \f[
     *    b = \hat{x} - x(\xi) - \frac{\tilde{n}}{\| \tilde{n} \|} d,
     *  \f]
     *  where \f$ \hat{x} \in \mathbb{R}^{d}\f$ is the given point,
     *        \f$ \xi \in \mathbb{R}^{d-1}\f$ is the projected in-plane
     *        parameter space coordinate vector,
     *        \f$ \tilde{n} \f$ is the non-unit normal vector on the side
     *        in the point \f$ \xi \f$,
     *        \f$ d \f$ is the distance in normal direction between the point
     *        \f$\hat{x}\f$ and the side.
     *
     *  The matrix \f$ A \f$ holds the consistent linearization.
     *
     */
    bool distance_system(const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, 1, FloatType>& px, FloatType* distance,
        Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType>& A,
        Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType>& B,
        Core::LinAlg::Matrix<prob_dim, 2 * dim_side - 1, FloatType>& C,
        Core::LinAlg::Matrix<prob_dim, 2, FloatType>& N,
        Core::LinAlg::Matrix<prob_dim, 1, FloatType>& b)
    {
      Core::LinAlg::Matrix<prob_dim, 1, FloatType> n1(&N(0, 0), true);
      Core::LinAlg::Matrix<prob_dim, 1, FloatType> n2(&N(0, 1), true);
      /* get only the local side coordinates and use them to evaluate
       * the 1-st and 2-nd derivatives */

      Core::LinAlg::Matrix<dim_side, 1, FloatType> xsi_side(xsi_.data(), true);

      // pre-evaluate some important variables
      B = 0.0;

      FloatType det = eval_derivs_in_parameter_space<prob_dim, side_type, FloatType>(
          xyze_side, xsi_side, sideDeriv1_, B, nullptr, &n1, &n2, false);

      A.update_t(B);


      if (debug)
      {
        std::cout << "det-metric               = " << det << std::endl;
        if (det < 1.0e-16) std::cout << "!!! determinant of jacobian is smaller than 1.0e-16 !!!\n";
      }

      if (det < 1.0e-16 && det > 1.0e-16)
#if EXTENDED_CUT_DEBUG_OUTPUT
        std::cout << "Determinant in  compute position is very close to zero" << std::endl;
#endif
      if (Core::MathOperations<FloatType>::abs(det) == 0.0)  // here might lie problem for the cln
      {
        /* then calculation of a normal to a line like this makes no sense at
         * all! */
#if EXTENDED_CUT_DEBUG_OUTPUT
        std::cout << "Absolute value of the determinant is zero " << std::endl;
#endif
        zeroarea_ = true;
        return false;
      }

      // inverse of the normal direction norm
      FloatType n1norm_inv = 1.0 / n1.norm2();
      // --- add the distance part to the right-hand-side
      b.update(-distance[0] * n1norm_inv, n1, 1.0);

      for (unsigned r = 0; r < prob_dim; ++r)  // store normal vector
        nvec_(r, 0) = n1(r);
      // --- evaluate the linearization
      // scale the 1-st normal in A to unit length
      for (unsigned r = 0; r < prob_dim; ++r) A(r, dim_side) *= n1norm_inv;
      // --- evaluate 2-nd derivatives at xsi w.r.t. the parameter space coordinates
      C = 0.0;
      Core::FE::shape_function_deriv2<side_type>(xsi_side, sideDeriv2_);
      C.multiply_nt(xyze_side, sideDeriv2_);

      // reset B-matrix
      B = 0.0;
      // 1-dimensional side ( a.k.a. edge or line ) embedded in 3-dimensional space
      if (prob_dim == 3 and dim_side == 1)
      {
        FloatType n2norm_inv = 1.0 / n2.norm2();

        // scale the 2-nd normal in the last column of A to unit length
        for (unsigned r = 0; r < prob_dim; ++r) A(r, 2) *= n2norm_inv;

        for (unsigned r = 0; r < prob_dim; ++r) nvec_(r, 1) = n2(r);

        // the normal linearization can be skipped for this case
        if (ComputeDistanceStrategy::distance() == 0.0) return true;

        // add extra term to the right hand side
        b.update(-distance[1] * n2norm_inv, n2, 1.0);
        // update stored normal vector

        // linearization of the 1-st unscaled normal vector
        switch (detect_normal_plane(n1))
        {
          case normal_in_xy_plane:
          {
            // d(n1_0) / dr =   d^2(y) / dr^2
            B(0, 0) = C(1, 0);
            // d(n1_1) / dr = - d^2(x) / dr^2
            B(1, 0) = -C(0, 0);
            // d(n1_2) / dr =   0.0
            break;
          }
          case normal_in_yz_plane:
          {
            // d(n1_0) / dr =    0.0
            // d(n1_1) / dr =  - d^2(z) / dr^2
            B(1, 0) = -C(2, 0);
            // d(n1_2) / dr =    d^2(y) / dr^2
            // u
            B(2, 0) = C(1, 0);
            break;
          }
          default:
          {
            FOUR_C_THROW("Shouldn't happen!");
            exit(EXIT_FAILURE);
          }
        }

        // linearization of the 2-nd unscaled normal vector
        // d(n2_0) / dr
        B(0, 1) = (C(1, 0) * n1(2) + A(1, 0) * B(2, 0) - (C(2, 0) * n1(1) + A(2, 0) * B(1, 0)));
        // d(n2_1) / dr
        B(1, 1) = (C(2, 0) * n1(0) + A(2, 0) * B(0, 0) - (C(0, 0) * n1(2) + A(0, 0) * B(2, 0)));
        // d(n2_2) / dr
        B(2, 1) = (C(0, 0) * n1(1) + A(0, 0) * B(1, 0) - (C(1, 0) * n1(0) + A(1, 0) * B(0, 0)));

        // linearization of the scaling factor ( 1-st normal )
        FloatType lin_nnorm = 0.0;

        lin_nnorm = (n1(0) * B(0, 0) + n1(1) * B(1, 0) + n1(2) * B(2, 0)) * n1norm_inv * n1norm_inv;
        for (unsigned r = 0; r < prob_dim; ++r) B(r, 0) -= lin_nnorm * A(r, 1);

        // linearization of the scaling factor ( 2-nd normal )
        lin_nnorm = (n2(0) * B(0, 1) + n2(1) * B(1, 1) + n2(2) * B(2, 1)) * n2norm_inv * n2norm_inv;
        for (unsigned r = 0; r < prob_dim; ++r) B(r, 1) -= lin_nnorm * A(r, 2);

        // final scaling and add
        for (unsigned r = 0; r < prob_dim; ++r)
        {
          A(r, 0) += distance[0] * B(r, 0);
          A(r, 0) += distance[1] * B(r, 1);
        }
      }
      // 2-dimensional side embedded in 3-dimensional space
      else if (prob_dim == 3 and dim_side == 2)
      {
        // the normal linearization can be skipped for this case
        if (ComputeDistanceStrategy::distance() == 0.0) return true;

        // linearization of the unscaled normal vector
        // d(n_0) / dr
        B(0, 0) = (C(1, 0) * A(2, 1) + A(1, 0) * C(2, 2) - (C(2, 0) * A(1, 1) + A(2, 0) * C(1, 2)));
        // d(n_0) / ds
        B(0, 1) = (C(1, 2) * A(2, 1) + A(1, 0) * C(2, 1) - (C(2, 2) * A(1, 1) + A(2, 0) * C(1, 1)));

        // d(n_1) / dr
        B(1, 0) = (C(2, 0) * A(0, 1) + A(2, 0) * C(0, 2) - (C(0, 0) * A(2, 1) + A(0, 0) * C(2, 2)));
        // d(n_1) / ds
        B(1, 1) = (C(2, 2) * A(0, 1) + A(2, 0) * C(0, 1) - (C(0, 2) * A(2, 1) + A(0, 0) * C(2, 1)));

        // d(n_2) / dr
        B(2, 0) = (C(0, 0) * A(1, 1) + A(0, 0) * C(1, 2) - (C(1, 0) * A(0, 1) + A(1, 0) * C(0, 2)));
        // d(n_2) / ds
        B(2, 1) = (C(0, 2) * A(1, 1) + A(0, 0) * C(1, 1) - (C(1, 2) * A(0, 1) + A(1, 0) * C(0, 1)));

        // linearization of the scaling factor fact
        FloatType lin_nnorm = 0.0;

        for (unsigned c = 0; c < dim_side; ++c)
        {
          lin_nnorm =
              (n1(0) * B(0, c) + n1(1) * B(1, c) + n1(2) * B(2, c)) * n1norm_inv * n1norm_inv;
          for (unsigned r = 0; r < prob_dim; ++r) B(r, c) -= lin_nnorm * A(r, 2);
        }

        // final scaling and add
        A.update(distance[0], B, 1.0);
      }
      // 1-dimensional side/line element embedded in 2-dimensional space
      else if (prob_dim == 2 and dim_side == 1)
      {
        // the normal linearization can be skipped for this case
        if (ComputeDistanceStrategy::distance() == 0.0) return true;

        // linearization of the unscaled normal vector
        B(0, 0) = C(1, 0);
        B(1, 0) = -C(0, 0);

        // linearization of the scaling factor fact
        FloatType lin_nnorm = (C(1, 0) + C(0, 0)) * n1norm_inv * n1norm_inv;
        for (unsigned r = 0; r < prob_dim; ++r) B(r, 0) -= lin_nnorm * A(r, 1);

        // final scaling and add
        A.update(distance[0], B, 1.0);
      }
      else
        FOUR_C_THROW("Unsupported dim <--> probDim relation!");
      return true;
    }

    /// solution vector (parameter space coordinates + distance values)
    Core::LinAlg::Matrix<prob_dim, 1, FloatType>& xsi_;

    /// distance values
    FloatType* distance_;

    const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>* xyze_side_;
    /// given global point coordinates
    const Core::LinAlg::Matrix<prob_dim, 1, FloatType>* px_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side,
        FloatType>::sideFunct_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side,
        FloatType>::sideDeriv1_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side,
        FloatType>::sideDeriv2_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>::A_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>::B_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>::C_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>::b_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>::dx_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>::N_;

    /// adapted Newton tolerance
    FloatType tol_;

    /** \brief  indicates that the surface element has a area value of zero
     *
     *  --> cross product for normal is zero!!! */
    bool zeroarea_;

    using MemberStoragePolicy<debug, prob_dim, side_type, dim_side, num_nodes_side,
        FloatType>::nvec_;
  };  // class ComputeDistanceStrategy


#ifdef CUT_CLN_CALC
  /*--------------------------------------------------------------------------*/
  /** \brief Strategy class for  distance between side and point for arbitrary precision
   *
   *  inheritance diagram:
   *
   *  compute_distance --> ComputeDistanceAdaptivePrecision --> NewtonSolve
   *                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   *  --> ComputeDistanceStrategy --> EmptyNewtonStrategy  */
  template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class ComputeDistanceAdaptivePrecision : Strategy
#ifdef DEBUG_MEMORY_ALLOCATION
      ,
                                           AdaptiveKernelSharedData
#endif
  {
   public:
    /// constructor
    ComputeDistanceAdaptivePrecision(Core::LinAlg::Matrix<prob_dim, 1>& xsi, bool checklimits)
        : Strategy(clnxsi_, checklimits),
          xsi_(xsi),
          clnxsi_(Core::LinAlg::Initialization::set_zero),
          cond_infinity_(false)
    {
    }

    /// start and solve the distance calculation
    bool operator()(const Core::LinAlg::Matrix<prob_dim, num_nodes_side>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, 1>& px, double& distance, bool signeddistance = false)
    {
      if (!((side_type == Core::FE::CellType::hex8) || (side_type == Core::FE::CellType::quad4) ||
              (side_type == Core::FE::CellType::hex8) || (side_type == Core::FE::CellType::tri3) ||
              (side_type == Core::FE::CellType::line2)) ||
          (side_type == Core::FE::CellType::hex20) || (side_type == Core::FE::CellType::tet4) ||
          (side_type == Core::FE::CellType::pyramid5))
      {
        FOUR_C_THROW(
            "This type of element ({})  is not tested for the CLN calculation. You are welcome "
            "to edit ../fem_general/utils_fem_shapefunctions.H file to fix it. Just "
            "change all the integers occurring there double for CLN to work.",
            Core::FE::cell_type_to_string(side_type).c_str());
      }

      int prec = CLN_START_PRECISION;  // standard initial precision
      bool conv;
      Core::CLN::ClnWrapper err;
      Core::CLN::ClnWrapper cond_number;
      int iter = 0;
      cond_infinity_ = false;
      bool zero_area = false;

      // converting everything to the cln precision
#ifdef CUSTOM_MEMORY_ALLOCATOR
      if (all_distance_done_once_ and all_intersections_done_once_ and all_position_done_once_ and
          (!custom_allocator_run_))
      {
#if DEBUG_MEMORY_ALLOCATION
        Cut::MemorySingleton::getinstance().ReportAllocated();
        report_intersection_allocated();
        report_position_allocated();
        report_distance_allocated();
        report_total_allocated();
        // start setting up the memory container
        Cut::MemorySingleton::getinstance().set_state(1, memory_allocations_);
#else
        Cut::MemorySingleton::getinstance().SwitchState();
#endif
        custom_allocator_run_ = true;
      }
#if DEBUG_MEMORY_ALLOCATION
      if (first_run_) std::cout << "In cut distance statistics is" << std::endl;
#endif
#endif
      do
      {
        Core::CLN::ClnWrapper::set_precision(prec);
#ifdef CUSTOM_MEMORY_ALLOCATOR
#if DEBUG_MEMORY_ALLOCATION
        if (custom_allocator_run_)
#endif
          Cut::MemorySingleton::getinstance().get_memory_pool_allocator().SetCurrent(
              cln_byte_size_[iter]);
#endif
#if DEBUG_MEMORY_ALLOCATION

        if (first_run_)
        {
          if (Cut::MemorySingleton::getinstance().IsRecording())
          {
            update_memory_allocations(Cut::MemorySingleton::getinstance().GetMemoryPattern());
            Cut::MemorySingleton::getinstance().StopRecord();
          }
          Cut::MemorySingleton::getinstance().StartRecord();
        }

#endif
        Core::CLN::conv_double_cln(xyze_side, clnxyze_side_, prec);
        Core::CLN::conv_double_cln(px, clnpx_, prec);
        //  no need to convert, we don't actually compute anything
        // clndistance_ = cln::cl_float(distance, cln::float_format(prec));
        this->setup(clnxyze_side_, clnpx_, false);

        conv = this->solve();

#if CUT_DEVELOP
        // safety check for precision loss across the array
        cln::float_format_t prec_beg = cln::float_format(clnxyze_side_(0, 0).Value());
        for (unsigned int i = 0; i < probDim; ++i)
        {
          cln::float_format_t prec_end = cln::float_format(clnxsi_(i, 0).Value());
          if (prec_beg != prec_end)
          {
            FOUR_C_THROW(
                "There is a loss of cln-precision during computation of intersection. "
                "Something is wrong with the conversion");
          }
        }
#endif
        if (not signeddistance)
          clndistance_ = this->distance();
        else
        {
          switch (prob_dim - dim_side)
          {
            case 1:
              clndistance_ = this->signed_distance()[0];
              break;
            default:
              FOUR_C_THROW("A scalar signed distance value is not available!");
              exit(EXIT_FAILURE);
          }
        }
        std::pair<bool, Core::CLN::ClnWrapper> cond_pair = this->condition_number();
        cond_number = cond_pair.second;

        if (not cond_pair.first)
        {
          cond_infinity_ = true;
        }
        // if side is zerorea-ed, we should' not divide by vector norm in the error calculation,
        // but just continue increase in the precision
        zero_area = this->zero_area();
        if (not zero_area)
          err = compute_error(xyze_side, px, this->local_coordinates(), this->get_normal_vector(),
                    this->signed_distance(), CLN_REFERENCE_PREC) *
                cond_number;
#if DEBUG_MEMORY_ALLOCATION
        update_memory_usage(prec, iter);
#endif
        prec += CLN_INCREMENT_STEP;
        iter++;

      } while (((!conv) || (err > CLN_LIMIT_ERROR) || (cond_infinity_) || zero_area) &&
               (prec < CLN_LIMIT_PREC) && (iter < CLN_LIMIT_ITER));

      // now since we converged or went over maximum allowed number of loops  we convert the
      // values back if we did not converge, by newton tolerance, but the error is small enough
      // treat as converged
      if ((not conv) && (err < CLN_LIMIT_ERROR) && (err > 0.0)) conv = true;

      if (cond_infinity_)
      {
#ifdef DEBUG_CUTKERNEL_OUTPUT
        std::cout << "Condition number of this problem is equal to infinity" << std::endl;
#endif
      }

      get_topology_information();
      // NOTE: Might be not needed later
      fix_corner_case();
      // converting all the values back
      Core::CLN::conv_cln_double(clnxsi_, xsi_);

      distance = cln::double_approx(clndistance_.Value());

#ifdef CUSTOM_MEMORY_ALLOCATOR
      if (first_run_)
      {
        const int num_inter = 3;
        if ((++get_distance_count()) == num_inter)
        {
          all_distance_done_once_ = true;
          std::cout << "All possible template distance are done!" << std::endl;
        }
        first_run_ = false;
      }
#endif
      return conv;
    }
    // get the local coordinates
    const Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper>& local_coordinates() const
    {
      return Strategy::local_coordinates();
    }

    const Core::CLN::ClnWrapper* signed_distance() const { return Strategy::signed_distance(); }

    Core::CLN::ClnWrapper distance() const { return Strategy::distance(); }

    std::pair<bool, Core::CLN::ClnWrapper> condition_number()
    {
      return Strategy::condition_number();
    }

    Core::LinAlg::Matrix<prob_dim, 2, Core::CLN::ClnWrapper> get_normal_vector()
    {
      return Strategy::get_normal_vector();
    }

    /// access the Newton tolerance
    Core::CLN::ClnWrapper get_tolerance() const { return Strategy::get_tolerance(); }
    PointOnSurfaceLoc get_side_location() { return location_; }
    const std::vector<int>& get_touched_side_edges() { return touched_edges_ids_; }

    const std::vector<int>& get_touched_nodes() { return touched_nodes_ids_; }

    void write_to_gmsh(std::ofstream& file) { Strategy::write_to_gmsh(file); }

    bool is_condition_infinity() { return cond_infinity_; }

   private:
    // Evaluate difference between initial global coordinates passed to compute_distance and
    // and global coordinates based on loc coordinates and distance  calculated in the
    // compute_distance, using conversion to very high reference precision
    Core::CLN::ClnWrapper compute_error(
        const Core::LinAlg::Matrix<prob_dim,
            Core::FE::num_nodes<side_type>>& refshape_xyze,  // referenape
        const Core::LinAlg::Matrix<prob_dim, 1>& p,          // global position of the point
        const Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper>&
            loc_calc,  // calculated wrt reference shape local position
        const Core::LinAlg::Matrix<prob_dim, 2, Core::CLN::ClnWrapper>& nvec,  // normal vector
        const Core::CLN::ClnWrapper*
            distance,  // distance from the projection on the reference shape  to the point
        int prec       // precision for calculation
    )
    {
      unsigned int prev_prec = Core::CLN::ClnWrapper::get_precision();
      Core::CLN::ClnWrapper::set_precision(prec);
      // Converting input arrays  to higher precision floating points
      Core::CLN::ClnWrapper clndistance;
      Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<side_type>, Core::CLN::ClnWrapper>
          xyze_side;
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> clnxi;
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> clnpx;
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> n1;

      clndistance = cln::cl_float(distance[0].Value(), cln::float_format(prec));
      Core::CLN::conv_double_cln(refshape_xyze, xyze_side, prec);
      Core::CLN::conv_double_cln(p, clnpx, prec);
      Core::CLN::update_precision(loc_calc, clnxi, prec);

      for (unsigned int i = 0; i < prob_dim; ++i)
      {
        n1(i) = cln::cl_float(nvec(i, 0).Value(), cln::float_format(prec));
      }

      Core::LinAlg::Matrix<Core::FE::num_nodes<side_type>, 1, Core::CLN::ClnWrapper> surfaceFunct;
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> b;

      Core::LinAlg::Matrix<dim_side, 1, Core::CLN::ClnWrapper> clnxiside(clnxi.data(), true);
      Core::FE::shape_function<side_type>(clnxiside, surfaceFunct);
      b = clnpx;
      b.multiply(-1.0, xyze_side, surfaceFunct, 1.0);

      // inverse of the normal direction norm
      Core::CLN::ClnWrapper n1norm_inv = 1.0 / n1.norm2();
      // --- add the distance part to the right-hand-side
      b.update(-clndistance * n1norm_inv, n1, 1.0);
      // in a special case add more of the distance
      if (prob_dim == 3 and dim_side == 1)
      {
        Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> n2;
        for (unsigned int i = 0; i < prob_dim; ++i)
          n2(i) = cln::cl_float(nvec(i, 1).Value(), cln::float_format(prec));

        Core::CLN::ClnWrapper n2norm_inv = 1.0 / n2.norm2();
        b.update(-distance[1] * n2norm_inv, n2, 1.0);
      }
      Core::CLN::ClnWrapper res = b.norm2();
      // resetting precision to previous
      Core::CLN::ClnWrapper::set_precision(prev_prec);
      return res;
    }

    enum PointOnSurfacePlane
    {
      above = 0,
      below = 1,
      on = 2
    };

    // Transform tolerances into local coordinate system and get and
    // get location of the point on the surface  as well as touched edges
    void get_topology_information()
    {
      Core::LinAlg::Matrix<dim_side, 1, Core::CLN::ClnWrapper> scaled_tolerance_side_touched_edges;
      this->get_local_tolerance(SIDE_DETECTION_TOLERANCE, scaled_tolerance_side_touched_edges);
      Core::LinAlg::Matrix<dim_side, 1, Core::CLN::ClnWrapper> zero_tolerance_side;
      bool is_inside = within_limits<side_type>(clnxsi_, zero_tolerance_side);
      PointOnSurfacePlane point_on_surface = get_location(SIDE_DETECTION_TOLERANCE);

      location_ = PointOnSurfaceLoc(is_inside, point_on_surface == on);
      touched_edges_ids_.clear();
      touched_nodes_ids_.clear();
      // remove edges, that are too far away
      if (location_.on_side())
      {
        get_edges_at<side_type>(clnxsi_, touched_edges_ids_, scaled_tolerance_side_touched_edges);
        get_nodes_at<side_type>(clnxsi_, touched_nodes_ids_, scaled_tolerance_side_touched_edges);
      }
    }

    /// try to fix the case that is close to the corner and outside - in that case point
    /// is close to corner and outside might be touching two edges but difference between it
    /// and corner point > tolerance - hence this must be corrected
    void fix_corner_case()
    {
      if (touched_edges_ids_.size() >= 2)
      {
        // if we are inside the "tolerance wrapper" around the side
        if (location_.within_side())
        {
          Core::LinAlg::Matrix<dim_side, 1, Core::CLN::ClnWrapper> zero_tolerance_side;
          // this means we are on the outer boundary of the tolerance
          if (not within_limits<side_type>(clnxsi_, zero_tolerance_side))
          {
            // we transform our point to global tolerance and check if it is close enough to the
            // nodal point of the side, that is common point of the touching edges
            const Core::LinAlg::Matrix<prob_dim, num_nodes_side, Core::CLN::ClnWrapper>&
                clnxyze_side_latest = clnxyze_side_;
            const Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper>& xsi_global = clnpx_;

            // find minimum distance and check if it is closer than 1e-14
            Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> dist;
            Core::CLN::ClnWrapper min_dist;
            for (unsigned int inode = 0; inode < num_nodes_side; ++inode)
            {
              // access raw responsible for that node
              Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> coord(
                  clnxyze_side_latest.data() + inode * prob_dim, true);
              dist.update(1.0, coord, -1.0, xsi_global);
              Core::CLN::ClnWrapper tmp_dist = dist.norm2();
              if ((inode == 0) or (tmp_dist < min_dist))
              {
                min_dist = tmp_dist;
              }
            }

            if (min_dist > NON_TOPOLOGICAL_TOLERANCE)
            {
              // then it will not be merged to this point in the pointpool. Hence we remove the
              // topological connection and change location of the point to outside
              touched_edges_ids_.clear();
              // toggle the bit off,
              location_ = PointOnSurfaceLoc(false, location_.on_side());
            }
          }
        }
      }
    }

    // Update global maximum number of allocations ( number of containers of particular byte
    // size) if allocations on this precisions were more. Also set what was the most frequent
    // size of allocation for the current iteration
    void update_memory_usage(int prec, int iter)
    {
#if DEBUG_MEMORY_ALLOCATION

      if (first_run_)
      {
        Cut::MemorySingleton::getinstance().StopRecord();

        std::unordered_map<size_t, int>& allocation_map =
            Cut::MemorySingleton::getinstance().GetMemoryPattern();
        int max_num = 0;
        int size_max_num = 0;
        // update the global number of allocations to the maximum
        for (std::unordered_map<size_t, int>::iterator it = allocation_map.begin();
            it != allocation_map.end(); ++it)
        {
          size_t size = it->first;
          if (memory_allocations_distance_[size] < allocation_map[size])
          {
            memory_allocations_distance_[size] = allocation_map[size];
          }
          if (allocation_map[size] > max_num)
          {
            size_max_num = size;
            max_num = allocation_map[size];
          }
        }

        if (size_max_num != 0)
        {
          std::cout << "Size for the precision " << prec << " is " << size_max_num << std::endl;
          std::cout << "It is allocated " << max_num << " times " << std::endl;
          cln_sizes_[iter] = size_max_num;
        }
        else
          FOUR_C_THROW("This should not be possible");

        Cut::MemorySingleton::getinstance().ResetAllocated();
      }
#endif
    }

    // Get location of the point with respect to the plain ( above/on/below )
    PointOnSurfacePlane get_location(const Core::CLN::ClnWrapper& err)
    {
      switch (prob_dim - dim_side)
      {
        case 1:
        {
          PointOnSurfacePlane result;
          const Core::CLN::ClnWrapper* signed_distance = Strategy::signed_distance();

          if ((signed_distance[0] + err < 0.0) && (signed_distance[0] - err < 0.0))
          {
            // it is below the side (distance is negative and bigger than tolerance)
            result = below;
          }
          else if ((signed_distance[0] + err > 0.0) && (signed_distance[0] - err > 0.0))
          {
            // it is above the side (distance is negative and higher than tolerance)
            result = above;
          }
          else
          {
            // else it is on surface
            result = on;
          }
          return result;
          break;
        }
        default:
        {
          // we cat only get signed distance for edge edge intersection
          const Core::CLN::ClnWrapper distance = Strategy::distance();
          // some safety checks
          if (distance < 0.0) FOUR_C_THROW("Not possible");
          if (err <= 0.0) FOUR_C_THROW("Error should be equal to basic tolerance!");
          if ((distance - err <= 0.0) and (distance + err > 0.0))
            return on;
          else
            return below;
        }
      }
    }

    // touched edges ids
    static std::vector<int> touched_edges_ids_;

    // touched edges ids
    static std::vector<int> touched_nodes_ids_;

    // variable for cln computation
    Core::LinAlg::Matrix<prob_dim, num_nodes_side, Core::CLN::ClnWrapper> clnxyze_side_;

    // global position of the point converted to cln precision
    Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> clnpx_;

    Core::CLN::ClnWrapper clndistance_;

    // reference to the double Matrix
    Core::LinAlg::Matrix<prob_dim, 1>& xsi_;

    Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> clnxsi_;

    // determines position of this point with respect to the side limits (values:
    // inside/outsaide)
    PointOnSurfaceLoc location_;

    // is condition number infinity or not
    bool cond_infinity_;

    static bool first_run_;

    // sizes of cln variables with different preciion
    static size_t cln_sizes_[CLN_LIMIT_ITER];
  };  // class ComputeDistanceAdaptivePrecision

#endif

  /*--------------------------------------------------------------------------*/
  /** \brief generic distance between side and point
   *
   *  inheritance diagram:
   *
   *  compute_distance --> GenericComputeDistance --> NewtonSolve
   *                      ^^^^^^^^^^^^^^^^^^^^^^
   *  --> ComputeDistanceStrategy --> EmptyNewtonStrategy  */
  template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type,
      bool compute_cln = false, unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class GenericComputeDistance : Strategy
  {
   public:
    /// constructor
    GenericComputeDistance(Core::LinAlg::Matrix<prob_dim, 1>& xsi, bool checklimits)
        : Strategy(xsi, checklimits),
          xsi_ref_(xsi),
          cond_infinity_(false)
#ifdef CUT_CLN_CALC
          ,
          checklimits_ref_(checklimits)
#endif
    {
    }

    /// start and solve the distance calculation
    bool operator()(const Core::LinAlg::Matrix<prob_dim, num_nodes_side>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, 1>& px, double& distance, bool signeddistance = false)
    {
      // now we are doing basic iteration on double precision
      bool conv;
      this->setup(xyze_side, px, false);
      conv = this->solve();

      if (not signeddistance)
        distance = this->distance();
      else
      {
        switch (prob_dim - dim_side)
        {
          case 1:
            distance = this->signed_distance()[0];
            break;
          default:
            FOUR_C_THROW("A scalar signed distance value is not available!");
            exit(EXIT_FAILURE);
        }
      }
      bool got_topology_info = get_topology_information();
#ifdef CUT_CLN_CALC
      if (compute_cln or !got_topology_info)
      {
#if DOUBLE_PLUS_CLN_COMPUTE
        bool result_fail = false;
        bool major_fail =
            this->zero_area() or (not conv) or (cond_infinity_) or (!got_topology_info);
        // otherwise computation will run into error for the
        // case of zero area
        if (!major_fail) result_fail = compute_error(xyze_side, px) > DOUBLE_LIMIT_ERROR;

        if (major_fail or result_fail)
        {
          // CutKernelStatistics::get_cut_kernel_statistics().ClnDistanceCounter();

#endif
          {
            ComputeDistanceAdaptivePrecision<
                NewtonSolve<
                    ComputeDistanceStrategy<false, prob_dim, side_type, dim_side, num_nodes_side,
                        Core::CLN::ClnWrapper, ComputeDistanceNoStaticMembers>,
                    prob_dim>,
                prob_dim, side_type>
                cln_calc(xsi_ref_, checklimits_ref_);

            bool distanceworked = cln_calc(xyze_side, px, distance, signeddistance);
            conv = distanceworked;
            location_ = cln_calc.get_side_location();
            // assign other location
            touched_edges_ids_.clear();
            touched_nodes_ids_.clear();
            const std::vector<int>& touched_edges_ids_cln = cln_calc.get_touched_side_edges();
            const std::vector<int>& touched_nodes_ids_cln = cln_calc.get_touched_nodes();
            touched_edges_ids_ = touched_edges_ids_cln;
            touched_nodes_ids_ = touched_nodes_ids_cln;
            cond_infinity_ = cln_calc.is_condition_infinity();
            Core::CLN::ClnWrapper::reset_precision();
          }
          // finalize memory allocator
#ifdef CUSTOM_MEMORY_ALLOCATOR
          Cut::MemorySingleton::getinstance().Finalize();
#endif
#if DOUBLE_PLUS_CLN_COMPUTE
        }
        else
        {
          // CutKernelStatistics::get_cut_kernel_statistics().double_distance_counter();
        }
#endif
      }
#endif
      return conv;
    }
    // get the local coordinates
    const Core::LinAlg::Matrix<prob_dim, 1>& local_coordinates() const
    {
      return Strategy::local_coordinates();
    }

    const double* signed_distance() const { return Strategy::signed_distance(); }

    double distance() const { return Strategy::distance(); }

    std::pair<bool, double> condition_number() { return Strategy::condition_number(); }

    Core::LinAlg::Matrix<prob_dim, 2> get_normal_vector() { return Strategy::get_normal_vector(); }

    bool surface_within_limits(double tolerance = REFERENCETOL) const
    {
      return Cut::Kernel::within_limits<side_type>(local_coordinates(), tolerance);
    }

    /// access the Newton tolerance
    double get_tolerance() const { return Strategy::get_tolerance(); }

    /// access the l2 norm of the reached residual
    double get_residual_l2_norm() const { return Strategy::get_residual_l2_norm(); }

    bool zero_area() const { return Strategy::zero_area(); }
    const std::vector<int>& get_touched_side_edges() { return touched_edges_ids_; }

    const std::vector<int>& get_touched_nodes() { return touched_nodes_ids_; }

    void get_touched_side_edges(std::vector<int>& inout)
    {
      inout.clear();
      inout = touched_edges_ids_;
    }

    bool is_condition_infinity() { return cond_infinity_; }


    // Get location of the point, based on the extended tolerance of 1e-14 for the diagonal
    // edge of the split triangle, and tolerance 0.0 for the normal edges, used
    // to avoid possibility of middle point of the quad4 split to be missed because of
    // numerical errors
    PointOnSurfaceLoc get_side_location_triangle_split()
    {
      if (side_type != Core::FE::CellType::tri3)
        FOUR_C_THROW("This method only works for tri3 side. Current side is {}",
            Core::FE::cell_type_to_string(side_type).c_str());

      Core::LinAlg::Matrix<dim_side, 1> scaled_tolerance;
      double distance_tolerance = TOPOLOGICAL_TOLERANCE;
      // get tolerance with 1e-14 around the triangle
      this->get_local_tolerance(distance_tolerance, scaled_tolerance);
      // Diagonal one, corresponds to the middle one, tolerance there should be original
      Core::LinAlg::Matrix<3, 1> real_tolerance;
      real_tolerance(0) = 0.0;
      real_tolerance(1) = scaled_tolerance(1);
      real_tolerance(2) = 0.0;
      if (within_limits_split_quad<side_type>(xsi_ref_, real_tolerance))
        side_location_triangle_split_ = PointOnSurfaceLoc(true, true);
      else
        side_location_triangle_split_ = PointOnSurfaceLoc(false, true);
      return side_location_triangle_split_;
    }

    PointOnSurfaceLoc get_side_location() { return location_; }

    //  Converts tolerance in the local coordinates and computes topological information, such
    //  as touched_edges and  location on the surface. In the case we cannot produce a valid
    //  conversion from global to local tolerance return false
    bool get_topology_information()
    {
      Core::LinAlg::Matrix<prob_dim, 1> xsi = this->local_coordinates();
      // set up required tolerances
      Core::LinAlg::Matrix<dim_side, 1> scaled_tolerance;
      double distance_tolerance = SIDE_DETECTION_TOLERANCE;
      bool success = this->get_local_tolerance(distance_tolerance, scaled_tolerance);
      if (not success) return false;
      Core::LinAlg::Matrix<dim_side, 1> zero_tolerance_side;
      zero_tolerance_side = 0.0;
      // find location
      bool is_inside = within_limits<side_type>(xsi, zero_tolerance_side);
      PointOnSurfacePlane point_on_surface = get_location(distance_tolerance);
      location_ = PointOnSurfaceLoc(is_inside, point_on_surface == on);
      // clear previous and get edges nearby
      touched_edges_ids_.clear();
      touched_nodes_ids_.clear();
      // if on side, get neighboring edges
      if (location_.on_side())
      {
        get_edges_at<side_type>(xsi, touched_edges_ids_, scaled_tolerance);
        get_nodes_at<side_type>(xsi, touched_nodes_ids_, scaled_tolerance);
      }
      return true;
    }

    void write_to_gmsh(std::ofstream& file) { Strategy::write_to_gmsh(file); }

    // Evaluate difference between initial global coordinates passed to compute_distance and
    // and global coordinates based on loc coordinates and distance  calculated in the
    // compute_distance
    double compute_error(
        const Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<side_type>>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, 1>& px)
    {
      const Core::LinAlg::Matrix<prob_dim, 1>& xsi = this->local_coordinates();
      const Core::LinAlg::Matrix<prob_dim, 2>& n_vec = this->get_normal_vector();
      const double* distance = this->signed_distance();
      Core::LinAlg::Matrix<Core::FE::num_nodes<side_type>, 1> surfaceFunct;
      Core::LinAlg::Matrix<prob_dim, 1> b;
      Core::LinAlg::Matrix<prob_dim, 1> n1(n_vec.data(), true);

      Core::LinAlg::Matrix<dim_side, 1> xsi_side(xsi.data(), true);
      Core::FE::shape_function<side_type>(xsi_side, surfaceFunct);
      b = px;

      b.multiply(-1.0, xyze_side, surfaceFunct, 1.0);

      double n1norm_inv = 1.0 / n1.norm2();
      b.update(-distance[0] * n1norm_inv, n1, 1.0);
      if (prob_dim == 3 and dim_side == 1)
      {
        Core::LinAlg::Matrix<prob_dim, 1> n2(n_vec.data() + prob_dim, true);
        double n2norm_inv = 1.0 / n2.norm2();
        b.update(-distance[1] * n2norm_inv, n2, 1.0);
      }
      double res = b.norm2();
      if (res == 0) res = MINIMUM_DOUBLE_ERROR;
      std::pair<bool, double> cond_pair = this->condition_number();
      if (not cond_pair.first)
      {
        cond_infinity_ = true;
      }
      return (res * cond_pair.second);
    }

   private:
    enum PointOnSurfacePlane
    {
      above = 0,
      below = 1,
      on = 2
    };

    // Get location of the point with respect to the plain ( above/on/below )
    PointOnSurfacePlane get_location(const double& err)
    {
      switch (prob_dim - dim_side)
      {
        case 1:
        {
          PointOnSurfacePlane result;
          // use it anywhere for now.
          const double* signed_distance = Strategy::signed_distance();

          if ((signed_distance[0] + err < 0.0) && (signed_distance[0] - err < 0.0))
          {
            /// it is below the side (distance is negative and bigger than tolerance)
            result = below;
          }
          else if ((signed_distance[0] + err > 0.0) && (signed_distance[0] - err > 0.0))
          {
            /// it is above the side (distance is negative and higher than tolerance)
            result = above;
          }
          else
          {
            // else it is on surface
            result = on;
          }
          return result;
          break;
        }
        default:
        {
          // we cat only get signed distance for edge edge intersection
          const double distance = Strategy::distance();
          // some safety checks
          if (distance < 0.0) FOUR_C_THROW("Not possible");
          if (err <= 0.0) FOUR_C_THROW("Error should be equal to basic tolerance!");
          if ((distance - err <= 0.0) and (distance + err > 0.0))
            return on;
          else
            return below;
        }
      }
    }
    // That encodes the location, last two bits represent whether point is on the surface and
    // within its limits
    PointOnSurfaceLoc location_;

    // touched edges ids
    static std::vector<int> touched_edges_ids_;

    // touched nodes id
    static std::vector<int> touched_nodes_ids_;

    // determines location of this point, with extended tolerance over side, that is shared for
    // another triangle (only for tri3)
    PointOnSurfaceLoc side_location_triangle_split_;
    // we need to do detection of location and touching edge
    Core::LinAlg::Matrix<prob_dim, 1>& xsi_ref_;

    // if condition number is equal to infinity
    bool cond_infinity_;
#ifdef CUT_CLN_CALC
    // Need to remember it in order to call to ComputeDistanceAdaptivePrecision
    bool checklimits_ref_;
#endif
  };  // class GenericComputeDistance

  /*--------------------------------------------------------------------------*/
  /** \brief most derived strategy class for distance between side and point
   *
   *  inheritance diagram:
   *
   *  compute_distance --> GenericComputeDistance --> NewtonSolve
   *  ^^^^^^^^^^^^^^^
   *  --> ComputeDistanceStrategy --> EmptyNewtonStrategy  */
  template <unsigned prob_dim, Core::FE::CellType side_type, bool compute_cln = false,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class ComputeDistance
      : public GenericComputeDistance<
            NewtonSolve<ComputeDistanceStrategy<false, prob_dim, side_type>, prob_dim>, prob_dim,
            side_type, compute_cln>
  {
   public:
    /// constructor
    ComputeDistance(Core::LinAlg::Matrix<prob_dim, 1>& xsi, bool checklimits = true)
        : GenericComputeDistance<
              NewtonSolve<ComputeDistanceStrategy<false, prob_dim, side_type>, prob_dim>, prob_dim,
              side_type, compute_cln>(xsi, checklimits)
    {
    }
  };  // class compute_distance

  /*--------------------------------------------------------------------------*/
  template <unsigned prob_dim, Core::FE::CellType side_type,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class DebugComputeDistance
      : public GenericComputeDistance<
            NewtonSolve<
                DebugNewtonStrategy<ComputeDistanceStrategy<true, prob_dim, side_type>, prob_dim>,
                prob_dim>,
            prob_dim, side_type>
  {
   public:
    /// constructor
    DebugComputeDistance(Core::LinAlg::Matrix<prob_dim, 1>& xsi, bool checklimits = true)
        : GenericComputeDistance<
              NewtonSolve<
                  DebugNewtonStrategy<ComputeDistanceStrategy<true, prob_dim, side_type>, prob_dim>,
                  prob_dim>,
              prob_dim, side_type>(xsi, checklimits)
    {
    }
  };  // class DebugComputeDistance

  // Static storage class of ComputeIntersection data members to inherit from
  template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type,
      Core::FE::CellType side_type, unsigned dim_edge = Core::FE::dim<edge_type>,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_edge = Core::FE::num_nodes<edge_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>, typename FloatType = double>
  struct ComputeIntersectionStaticMembers
  {
    static Core::LinAlg::Matrix<num_nodes_side, 1, FloatType> sideFunct_;
    static Core::LinAlg::Matrix<num_nodes_edge, 1, FloatType> edgeFunct_;
    static Core::LinAlg::Matrix<dim_side, num_nodes_side, FloatType> sideDeriv1_;
    static Core::LinAlg::Matrix<dim_edge, num_nodes_edge, FloatType> edgeDeriv1_;

    static Core::LinAlg::Matrix<dim_edge + dim_side, dim_edge + dim_side, FloatType> A_;
    static Core::LinAlg::Matrix<prob_dim, dim_edge + dim_side, FloatType> B_;
    static Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType> b_;
    static Core::LinAlg::Matrix<prob_dim, 1, FloatType> c_;

    /// increment in local coordinates d(xi1_side, xi2_side, xi_line)
    static Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType> dx_;
  };


  // Storage class of ComputeIntersection data members to inherit from
  template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type,
      Core::FE::CellType side_type, unsigned dim_edge = Core::FE::dim<edge_type>,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_edge = Core::FE::num_nodes<edge_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>, typename FloatType = double>
  struct ComputeIntersectionNoStaticMembers
  {
    Core::LinAlg::Matrix<num_nodes_side, 1, FloatType> sideFunct_;
    Core::LinAlg::Matrix<num_nodes_edge, 1, FloatType> edgeFunct_;
    Core::LinAlg::Matrix<dim_side, num_nodes_side, FloatType> sideDeriv1_;
    Core::LinAlg::Matrix<dim_edge, num_nodes_edge, FloatType> edgeDeriv1_;

    Core::LinAlg::Matrix<dim_edge + dim_side, dim_edge + dim_side, FloatType> A_;
    Core::LinAlg::Matrix<prob_dim, dim_edge + dim_side, FloatType> B_;
    Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType> b_;
    Core::LinAlg::Matrix<prob_dim, 1, FloatType> c_;

    Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType> dx_;
  };

  /*--------------------------------------------------------------------------*/
  /** \brief strategy class for intersection of side and edge/line
   *
   *  inheritance diagram:
   *
   *  ComputeIntersection --> GenericComputeIntersection --> NewtonSolve
   *  --> ComputeIntersectionStrategy --> EmptyNewtonStrategy
   *      ^^^^^^^^^^^^^^^^^^^^^^^^^^^  */
  template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type,
      Core::FE::CellType side_type, unsigned dim_edge = Core::FE::dim<edge_type>,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_edge = Core::FE::num_nodes<edge_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>, typename FloatType = double,
      template <bool, unsigned, Core::FE::CellType, Core::FE::CellType, unsigned, unsigned,
          unsigned, unsigned, typename> class MemberStoragePolicy =
          ComputeIntersectionStaticMembers>
  class ComputeIntersectionStrategy
      : public EmptyNewtonStrategy,
        MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
            num_nodes_edge, num_nodes_side, FloatType>
  {
   public:
    /// constructor
    ComputeIntersectionStrategy(
        Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType>& xsi, bool checklimits)
        : xsi_(xsi),
          xyze_side_(nullptr),
          xyze_edge_(nullptr),
          residual_(0.0),
          off_count_(0),
          check_limits_(checklimits),
          tol_(0.0)
    {
    }

    void setup(const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_edge, FloatType>& xyze_edge)
    {
      xyze_side_ = &xyze_side;
      xyze_edge_ = &xyze_edge;

      if (debug)
      {
        std::cout << "\n\n === ComputeIntersection ===\n";
        std::cout << "--- setup()\n";
        std::cout << "  Edge = " << Core::FE::cell_type_to_string(edge_type) << "\n";
        std::cout << "  Side = " << Core::FE::cell_type_to_string(side_type) << "\n";
        std::cout << "  xyze_side = " << std::setprecision(15) << xyze_side;
        std::cout << "  xyze_edge = " << std::setprecision(15) << xyze_edge;
      }
    }

    std::pair<bool, FloatType> condition_number()
    {
      Core::LinAlg::Matrix<dim_edge + dim_side, dim_edge + dim_side, FloatType> A_inv;
      FloatType det = A_.determinant();
      if (Kernel::close_to_zero(det))
      {
#ifdef DEBUG_CUTKERNEL_OUTPUT
        std::cout << " WARNING: Condition number is equal to infinity in the problem. "
                     "Continuing increase in precision "
                  << std::endl;
#endif
        return std::make_pair(false, -1.0);
      }
      A_inv.invert(A_);
      FloatType cond = A_inv.norm2() * A_.norm2();
      return std::make_pair(true, cond);
    }

    bool get_local_tolerance_edge(const FloatType& global_tolerance,
        Core::LinAlg::Matrix<dim_edge, 1, FloatType>& scaled_tolerance)
    {
      Core::LinAlg::Matrix<prob_dim, 1, FloatType> real_tolerance;
      real_tolerance =
          global_tolerance;  // set all values of the vector to be equal to the global tolerance
      // extract part of the side from the Jacobian
      Core::LinAlg::Matrix<prob_dim, dim_edge, FloatType> A;
      if (dim_edge + dim_side == prob_dim)
      {
        for (unsigned int row = 0; row < prob_dim; ++row)
        {
          for (unsigned int col = 0; col < dim_edge; ++col) A(row, col) = A_(row, dim_side + col);
        }
      }
      else
      {
        for (unsigned int row = 0; row < prob_dim; ++row)
        {
          for (unsigned int col = 0; col < dim_edge; ++col) A(row, col) = B_(row, dim_side + col);
        }
      }

      Core::LinAlg::Matrix<dim_edge, dim_edge, FloatType> TN_inv;
      Core::LinAlg::Matrix<dim_edge, prob_dim, FloatType> aux;
      TN_inv.multiply_tn(A, A);

      FloatType det = TN_inv.determinant();
      // we will not be able to find inverse
      if (Kernel::close_to_zero(det)) return false;

      TN_inv.invert();
      aux.multiply_nt(TN_inv, A);
      scaled_tolerance.multiply(aux, real_tolerance);

      return true;
    }

    bool get_local_tolerance_side(const FloatType& global_tolerance,
        Core::LinAlg::Matrix<dim_side, 1, FloatType>& scaled_tolerance)
    {
      // extract part of the side from the Jacobian
      Core::LinAlg::Matrix<prob_dim, dim_side, FloatType> A;
      if (dim_edge + dim_side == prob_dim)
      {
        for (unsigned int row = 0; row < prob_dim; ++row)
        {
          for (unsigned int col = 0; col < dim_side; ++col) A(row, col) = A_(row, col);
        }
      }
      else
      {
        for (unsigned int row = 0; row < prob_dim; ++row)
        {
          for (unsigned int col = 0; col < dim_side; ++col) A(row, col) = B_(row, col);
        }
      }

      for (unsigned idim = 0; idim < dim_side; ++idim)
      {
        FloatType A2_tmp = A(0, idim) * A(0, idim);
        for (unsigned intersect_dim = 1; intersect_dim < prob_dim; ++intersect_dim)
          A2_tmp += A(intersect_dim, idim) * A(intersect_dim, idim);
        scaled_tolerance(idim) =
            global_tolerance * 1.0 / Core::MathOperations<FloatType>::sqrt(A2_tmp);
      }
      return true;
    }

    const Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType>& local_coordinates() const
    {
      return xsi_;
    }

    FloatType distance_between() const { return c_.norm2(); }

    void setup_solve()
    {
      xsi_ = 0.0;
      dx_ = 0.0;
      c_ = 0.0;


      off_count_ = 0;
      residual_ = 0.0;

      // evaluate initial rhs value
      const Core::LinAlg::Matrix<dim_side, 1, FloatType> xsi_side(xsi_.data(), true);
      const Core::LinAlg::Matrix<dim_edge, 1, FloatType> xsi_edge(xsi_.data() + dim_side, true);
      intersection_rhs(xsi_edge, xsi_side, *xyze_edge_, *xyze_side_, c_);
      tol_ = adaptive_combined_newton_tolerance(*xyze_side_, *xyze_edge_, c_, xsi_(0, 0));
    }

    void setup_step(int iter)
    {
      // build linear system of equations
      const Core::LinAlg::Matrix<dim_side, 1, FloatType> xsi_side(xsi_.data(), true);
      const Core::LinAlg::Matrix<dim_edge, 1, FloatType> xsi_edge(xsi_.data() + dim_side, true);
      if (iter > 0)
      {
        intersection_rhs(xsi_edge, xsi_side, *xyze_edge_, *xyze_side_, c_);
      }
      intersection_system(xsi_edge, xsi_side, *xyze_edge_, *xyze_side_, c_, A_, B_, b_);
    }

    /// compute the right-hand-side
    void intersection_rhs(const Core::LinAlg::Matrix<dim_edge, 1, FloatType>& xsi_edge,
        const Core::LinAlg::Matrix<dim_side, 1, FloatType>& xsi_side,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_edge, FloatType>& xyze_edge,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>& xyze_side,
        Core::LinAlg::Matrix<prob_dim, 1, FloatType>& c)
    {
      Core::FE::shape_function<side_type>(xsi_side, sideFunct_);
      Core::FE::shape_function<edge_type>(xsi_edge, edgeFunct_);

      c.multiply(xyze_edge, edgeFunct_);
      c.multiply(-1.0, xyze_side, sideFunct_, 1.0);
    }


    enum NewtonStatus test_converged(int iter)
    {
      residual_ = b_.norm2();


      if (debug)
      {
        std::cout << "--- TestConverged\n";
        FloatType incr = dx_.norm2();
        std::cout << "  residual = " << std::setprecision(15) << residual_
                  << "  incr = " << std::setprecision(15) << incr << "\n";
      }

      if (c_.norm2() < tol_ && iter > 0)
      {
        tol_ *= std::sqrt(static_cast<double>(prob_dim));
        return converged;
      }

      /* If the norm of c_ does not but the norm of b_ does fulfill the criterion,
       * the least square scheme failed! */
      if (residual_ < tol_ && iter > 0)
      {
        return failed;
      }

      return unconverged;
    }

    bool linear_solve(int iter)
    {
      dx_ = 0.0;
      FloatType det =
          Core::LinAlg::gauss_elimination<true, dim_side + dim_edge, FloatType>(A_, b_, dx_);

      if (debug)
      {
        std::cout << "--- linear_solve\n";
        std::cout << "  A_ = " << A_;
        std::cout << "  b_ = " << b_;
        std::cout << "  det = " << std::setprecision(15) << det << "\n";
        std::cout << "  dxi_ = " << dx_;
      }

      return (det != 0.0);
    }

    bool update(int iter)
    {
      xsi_.update(1.0, dx_, 1.0);


#ifndef CUT_CLN_CALC
      if (debug)
      {
        std::cout << "--- Update\n";
        const Core::LinAlg::Matrix<dimSide, 1> xsi_side(xsi_.data(), true);
        const Core::LinAlg::Matrix<dimEdge, 1> xsi_edge(xsi_.data() + dimSide, true);
        std::cout << "  off_count=" << off_count_ << "\n  Side within limits = "
                  << (within_limits<sideType>(xsi_side) ? "TRUE" : "FALSE")
                  << "\n  Edge within limits = "
                  << (within_limits<edgeType>(xsi_edge) ? "TRUE" : "FALSE") << "\n";
        std::cout << "  xsi = " << xsi_;
        std::cout << "  xsi_side = " << xsi_side;
        std::cout << "  xsi_edge = " << xsi_edge;

        Core::LinAlg::Matrix<numNodesSide, 1> sideFunct;
        Core::LinAlg::Matrix<numNodesEdge, 1> edgeFunct;

        Core::FE::shape_function<sideType>(xsi_side, sideFunct);
        Core::FE::shape_function<edgeType>(xsi_edge, edgeFunct);

        Core::LinAlg::Matrix<probDim, 1> x1;
        Core::LinAlg::Matrix<probDim, 1> x2;

        x1.multiply(*xyze_side_, sideFunct);
        x2.multiply(*xyze_edge_, edgeFunct);

        std::cout << "  x_side =" << x1;
        std::cout << "  x_edge =" << x2;

        x2.update(1, x1, -1);
        std::cout << "  (x_side - x_edge) =" << x2;
      }
#endif

      return true;
    }

    bool newton_failed()
    {
#ifdef DEBUG_CUTKERNEL_OUTPUT
      // If it is not edge-edge intersections we explicitly notify, that Newton's method failed
      if (probDim == dimEdge + dimSide)
      {
        std::cout << "Current type is " << typeid(floatType).name() << std::endl;
        std::cout << "Newton method failed" << std::endl;
        std::cout << "Residual is" << residual_ << std::endl;
        std::cout << "Tolerance is" << tol_ << std::endl;
        std::cout << "Precision is" << Core::CLN::ClnWrapper::precision_ << "decimal points"
                  << std::endl;
        std::cout << "Local coordinates are " << xsi_ << std::endl;
      }
#endif

      tol_ = c_.norm2() * std::sqrt(3.0);

#ifdef DEBUG_CUTKERNEL_OUTPUT
      if (probDim == dimEdge + dimSide)
      {
        std::stringstream str;
        str << "Newton scheme did not converge:\n  "
#ifndef CLN_CALC
            << (*xyze_side_)
            << (*xyze_edge_)
#endif
#ifndef CLN_CALC
                   DumpDoubles(str, xyze_side_->data(), xyze_side_->m() * xyze_side_->n());
        str << "\n";
        DumpDoubles(str, xyze_edge_->data(), xyze_edge_->m() * xyze_edge_->n());
#endif


        std::stringstream f_str;
        f_str << ".NewtonFailed_intersection.pos";
        std::string filename(Cut::Output::GenerateGmshOutputFilename(f_str.str()));
        std::ofstream file(filename.c_str());
        write_to_gmsh(file);
      }
#endif
      return false;
    }

    FloatType get_tolerance() const { return tol_; }

    void write_to_gmsh(std::ofstream& file)
    {
      file.precision(32);  // higher precision!
      char elementType = '\0';
      switch (num_nodes_side)
      {
        case 3:
          elementType = 'T';
          break;
        case 4:
          elementType = 'Q';
          break;
        default:
          FOUR_C_THROW(
              "unsupported element type ({}) in write_to_gmsh."
              " Please feel free to extend the functionality if necessary.",
              Core::FE::cell_type_to_string(side_type));
      }

      file << "View \""
           << "Side"
           << "\" {\n";
      {
        file << "S" << elementType << "(";
        for (unsigned i = 0; i < num_nodes_side; ++i)
        {
          if (i != 0) file << ",";
          file << (*xyze_side_)(0, i) << "," << (*xyze_side_)(1, i) << "," << (*xyze_side_)(2, i);
        }
        file << "){";
        for (unsigned i = 0; i < num_nodes_side; ++i)
        {
          if (i != 0) file << ",";
          file << "1";
        }
        file << "};\n";
      }
      file << "};\n";

      file << "View \""
           << "Line"
           << "\" {\n";
      for (unsigned iline = 0; iline < num_nodes_edge - 1; ++iline)
      {
        file << "SL (";
        file << (*xyze_edge_)(0, iline) << "," << (*xyze_edge_)(1, iline) << ","
             << (*xyze_edge_)(2, iline) << ",";
        file << (*xyze_edge_)(0, iline + 1) << "," << (*xyze_edge_)(1, iline + 1) << ","
             << (*xyze_edge_)(2, iline + 1);
        file << "){";
        file << 1 << ",";
        file << 1;
        file << "};\n";
      }
      file << "};\n";


      // calculate the points
      Core::LinAlg::Matrix<num_nodes_side, 1, FloatType> surface;
      Core::LinAlg::Matrix<num_nodes_edge, 1, FloatType> line;

      Core::FE::shape_function_2d(surface, xsi_(0), xsi_(1), side_type);
      Core::FE::shape_function_1d(line, xsi_(2), edge_type);

      Core::LinAlg::Matrix<prob_dim, 1, FloatType> x1;
      Core::LinAlg::Matrix<prob_dim, 1, FloatType> x2;

      x1.multiply(*xyze_side_, surface);
      x2.multiply(*xyze_edge_, line);


      file << "View \""
           << "SidePoint"
           << "\" {\n";
      {
        file << "SP (";
        file << x1(0, 0) << "," << x1(1, 0) << "," << x1(2, 0);
        file << "){";
        file << 1;
        file << "};\n";
        file << "};\n";
      }
      file << "View \""
           << "EdgePoint"
           << "\" {\n";
      {
        file << "SP (";
        file << x2(0, 0) << "," << x2(1, 0) << "," << x2(2, 0);
        file << "){";
        file << 2;
        file << "};\n";
        file << "};\n";
      }
    }

   private:
    /** \brief Build the linear system of equations
     *
     *  Right-hand-side:
     *  \f[
     *    b = x(\xi) - x(\tilde{\xi}),
     *  \f]
     *  where \f$\xi \in \mathbb{R}^{dimEdge}\f$ is the parameter space
     *  coordinate of the edge-element and \f$ \tilde{\xi} \in \mathbb{R}^{dimSide} \f$
     *  are the parameter space coordinates of the side element.
     *
     *  The matrix \f$ A \f$ holds the consistent linearization. If the problem
     *  dimension is larger than the accumulated dimensions of the side and edge
     *  elements, a least square approach will be used.
     *
     */
    void intersection_system(const Core::LinAlg::Matrix<dim_edge, 1, FloatType>& xsi_edge,
        const Core::LinAlg::Matrix<dim_side, 1, FloatType>& xsi_side,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_edge, FloatType>& xyze_edge,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>& xyze_side,
        Core::LinAlg::Matrix<prob_dim, 1, FloatType>& c,
        Core::LinAlg::Matrix<dim_edge + dim_side, dim_edge + dim_side, FloatType>& A,
        Core::LinAlg::Matrix<prob_dim, dim_edge + dim_side, FloatType>& B,
        Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType>& b)
    {
      // --- compute the linearization
      Core::FE::shape_function_deriv1<side_type>(xsi_side, sideDeriv1_);
      Core::FE::shape_function_deriv1<edge_type>(xsi_edge, edgeDeriv1_);

      A = 0.0;
      B = 0.0;

      // linearization w.r.t. the parameter space coordinate(s) of the side
      for (unsigned inode = 0; inode < num_nodes_side; ++inode)
        for (unsigned row = 0; row < prob_dim; ++row)
          for (unsigned col = 0; col < dim_side; ++col)
            B(row, col) += xyze_side(row, inode) * sideDeriv1_(col, inode);

      // linearization w.r.t. the parameter space coordinate(s) of the edge
      for (unsigned inode = 0; inode < num_nodes_edge; ++inode)
        for (unsigned row = 0; row < prob_dim; ++row)
          for (unsigned col = 0; col < dim_edge; ++col)
            B(row, dim_side + col) -= xyze_edge(row, inode) * edgeDeriv1_(col, inode);

      // default case
      if (prob_dim == dim_side + dim_edge)
      {
        A.set_view(B.data());
        b.set_view(c.data());
      }
      /* Actually there is only one relevant case, which comes into my mind
       * and that is the intersection of two lines in 3 dimensions. Anyway
       * we use a more general if-statement here.
       *
       * If we have to handle such a configuration we use a Least-Squares
       * approach. */
      else if (prob_dim > dim_side + dim_edge)
      {
        A.multiply_tn(B, B);
        b.multiply_tn(B, c);
      }
      else
        FOUR_C_THROW(
            "The problem dimension is smaller than the combination of"
            "the side and edge element dimensions. This case is currently "
            "unsupported!");
    }

   private:
    /// local coordinates (xi1_side, xi2_side, xi_line)
    Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType>& xsi_;

    const Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType>* xyze_side_;

    const Core::LinAlg::Matrix<prob_dim, num_nodes_edge, FloatType>* xyze_edge_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::sideFunct_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::edgeFunct_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::sideDeriv1_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::edgeDeriv1_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::A_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::B_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::b_;

    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::c_;
    /// increment in local coordinates d(xi1_side, xi2_side, xi_line)
    using MemberStoragePolicy<debug, prob_dim, edge_type, side_type, dim_edge, dim_side,
        num_nodes_edge, num_nodes_side, FloatType>::dx_;

    /// residual norm
    FloatType residual_;

    /** count how many iterations during the Newton the local coordinates
     *  are not within limits! Stop the newton scheme if the iteration is
     *  too often outside the limits */
    int off_count_;

    /** shall we check the limits? ( seems out-dated, since we donot check
     *  limits by default ) */
    bool check_limits_;
    // basic tolerance!!!
    FloatType tol_;
  };  // class ComputeIntersectionStrategy


#ifdef CUT_CLN_CALC

  /*--------------------------------------------------------------------------*/
  /** \brief Strategy class for intersection of side and line/edge for arbitrary precision
   *
   *  inheritance diagram:
   *
   *  ComputeIntersection --> ComputeIntersectionAdaptivePrecision --> NewtonSolve
   *                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   *  --> ComputeIntersectionStrategy --> EmptyNewtonStrategy  */
  template <class Strategy, unsigned prob_dim, Core::FE::CellType edge_type,
      Core::FE::CellType side_type, unsigned dim_edge = Core::FE::dim<edge_type>,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_edge = Core::FE::num_nodes<edge_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class ComputeIntersectionAdaptivePrecision : Strategy
#ifdef DEBUG_MEMORY_ALLOCATION
      ,
                                               AdaptiveKernelSharedData
#endif
  {
   public:
    //! constructor
    ComputeIntersectionAdaptivePrecision(
        Core::LinAlg::Matrix<dim_edge + dim_side, 1>& xsi, bool checklimits)
        : Strategy(clnxsi_, checklimits),
          xsi_(xsi),
          clnxsi_(Core::LinAlg::Initialization::set_zero),
          cond_infinity_(false)
    {
    }

    bool operator()(const Core::LinAlg::Matrix<prob_dim, num_nodes_side>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_edge>& xyze_edge)
    {
      if (!(((edge_type == Core::FE::CellType::tet4) ||
                (edge_type == Core::FE::CellType::pyramid5) ||
                (edge_type == Core::FE::CellType::hex8) ||
                (edge_type == Core::FE::CellType::quad4) ||
                (edge_type == Core::FE::CellType::hex20) ||
                (edge_type == Core::FE::CellType::tri3) ||
                (edge_type == Core::FE::CellType::hex8) ||
                (edge_type == Core::FE::CellType::line2)) &&
              ((side_type == Core::FE::CellType::hex8) ||
                  (side_type == Core::FE::CellType::quad4) ||
                  (side_type == Core::FE::CellType::tri3) ||
                  (side_type == Core::FE::CellType::line2) ||
                  (side_type == Core::FE::CellType::hex20) ||
                  (side_type == Core::FE::CellType::tet4) ||
                  (side_type == Core::FE::CellType::pyramid5) ||
                  (side_type == Core::FE::CellType::hex8))))
      {
        FOUR_C_THROW(
            "This type of element ({})  is not tested for the CLN calculation. You are welcome "
            "to edit ../fem_general/utils_fem_shapefunctions.H file to fix it. Just "
            "change all the integers occurring there double for CLN to work.",
            Core::FE::cell_type_to_string(side_type).c_str());
      }

      // Converting values for the calculation on the CLN
      int prec = CLN_START_PRECISION;
      int iter = 0;
      bool conv;
      Core::CLN::ClnWrapper err;
      Core::CLN::ClnWrapper cond_number;
      cond_infinity_ = false;

#ifdef CUSTOM_MEMORY_ALLOCATOR
      if (all_distance_done_once_ and all_intersections_done_once_ and all_position_done_once_ and
          (!custom_allocator_run_))
      {
#if DEBUG_MEMORY_ALLOCATION
        Cut::MemorySingleton::getinstance().ReportAllocated();
        report_intersection_allocated();
        report_position_allocated();
        report_distance_allocated();
        report_total_allocated();
        Cut::MemorySingleton::getinstance().set_state(1, memory_allocations_);
#else
        Cut::MemorySingleton::getinstance().SwitchState();
#endif
        custom_allocator_run_ = true;
      }
#if DEBUG_MEMORY_ALLOCATION
      if (first_run_) std::cout << "In cut intersection statistics is" << std::endl;
#endif
#endif

      do
      {
        Core::CLN::ClnWrapper::set_precision(prec);
#ifdef CUSTOM_MEMORY_ALLOCATOR
#if DEBUG_MEMORY_ALLOCATION
        if (custom_allocator_run_)
#endif
          Cut::MemorySingleton::getinstance().get_memory_pool_allocator().SetCurrent(
              cln_byte_size_[iter]);
#endif
#if DEBUG_MEMORY_ALLOCATION

        if (first_run_)
        {
          if (Cut::MemorySingleton::getinstance().IsRecording())
          {
            update_memory_allocations(Cut::MemorySingleton::getinstance().GetMemoryPattern());
            Cut::MemorySingleton::getinstance().StopRecord();
          }
          Cut::MemorySingleton::getinstance().StartRecord();
        }

#endif
        Core::CLN::conv_double_cln(xyze_side, clnxyze_side_, prec);
        Core::CLN::conv_double_cln(xyze_edge, clnxyze_edge_, prec);
        this->setup(clnxyze_side_, clnxyze_edge_);

        conv = this->solve();

#if CUT_DEVELOP
        // safety check for precision loss
        cln::float_format_t prec_beg = cln::float_format(clnxyze_side_(0, 0).Value());
        for (unsigned int i = 0; i < dimEdge + dimSide; ++i)
        {
          cln::float_format_t prec_end = cln::float_format(clnxsi_(i, 0).Value());
          if (prec_beg != prec_end)
          {
            FOUR_C_THROW(
                "There is a loss of cln-precision during computation of intersection. "
                "Something is wrong with the conversion");
          }
        }
#endif

        std::pair<bool, Core::CLN::ClnWrapper> cond_pair = this->condition_number();
        cond_number = cond_pair.second;
        if (not cond_pair.first)
        {
          cond_infinity_ = true;
        }

        err = cond_number *
              compute_error(xyze_side, xyze_edge, this->local_coordinates(), CLN_REFERENCE_PREC);

        // compute scaled value of the error with respect to global coordinate
#if DEBUG_MEMORY_ALLOCATION
        update_memory_usage(prec, iter);
#endif
        prec += CLN_INCREMENT_STEP;
        iter++;
        // increase the precision
      } while (((!conv) || (err > CLN_LIMIT_ERROR) || (cond_infinity_)) &&
               (prec < CLN_LIMIT_PREC) && (iter < CLN_LIMIT_ITER));


      if ((not conv) && (err < CLN_LIMIT_ERROR) && (err > 0.0) &&
          ((dim_edge + dim_side) == prob_dim))
        conv = true;
      else if (not conv)
      {
#ifdef DEBUG_CUTKERNEL_OUTPUT
        if (probDim == dimSide + dimEdge)
          std::cout << "Newton method failed and the residual is " << err << " / "
                    << CLN_LIMIT_ERROR << std::endl;
#endif
      }

      if (cond_infinity_)
      {
#ifdef DEBUG_CUTKERNEL_OUTPUT
        std::cout << "Condition number of this problem is equal to infinity" << std::endl;
#endif
      }

      get_topology_information();
      // NOTE: Might be not needed later
      fix_corner_case();
#ifdef CUSTOM_MEMORY_ALLOCATOR
      // We need it because on the first run, we try to do the try run with allocation of the
      // required non-reusable doubles, e.g. double->cln compile time constant and other global
      // variables. For compile time constants it does not really matter, but for the global
      // variables we need somehow to allocate them in the not reusable container
      if (first_run_)
      {
        const int num_inter = 3;
        if ((++get_intersection_count()) == num_inter)
        {
          all_intersections_done_once_ = true;
          std::cout << "All possible template intersection are done!" << std::endl;
        }
        first_run_ = false;
      }
#endif
      Core::CLN::conv_cln_double(clnxsi_, xsi_);
      clnxsi_ = 0.0;  // resetting clnxsi_
      return conv;
    }

    const Core::LinAlg::Matrix<dim_edge + dim_side, 1, Core::CLN::ClnWrapper>& local_coordinates()
        const
    {
      return Strategy::local_coordinates();
    }

    std::pair<bool, Core::CLN::ClnWrapper> condition_number()
    {
      return Strategy::condition_number();
    }

    bool surface_within_limits(double tolerance = REFERENCETOL) const
    {
      return within_limits<side_type>(local_coordinates(), tolerance);
    }

    bool line_within_limits(double tolerance = REFERENCETOL) const
    {
      const Core::LinAlg::Matrix<dim_edge, 1> xsi_line(local_coordinates().data() + dim_side, true);
      return within_limits<edge_type>(xsi_line, tolerance);
    }

    Core::CLN::ClnWrapper get_tolerance() const { return Strategy::get_tolerance(); }

    void write_to_gmsh(std::ofstream& file) { Strategy::write_to_gmsh(file); }

    bool is_condition_infinity() { return cond_infinity_; }

    PointOnSurfaceLoc get_side_location() { return side_location_; }

    PointOnSurfaceLoc get_edge_location() { return edge_location_; }

    const std::vector<int>& get_touched_side_edges() { return touched_edges_ids_; }

    Core::CLN::ClnWrapper distance_between() const { return Strategy::distance_between(); }

   private:
    //  Converts tolerance in the local coordinates and computes topological information, such
    //  as touched_edges and  location on the surface
    void get_topology_information()
    {
      // tolerance of getting nearby edges
      Core::LinAlg::Matrix<dim_side, 1, Core::CLN::ClnWrapper> scaled_tolerance_side_touched_edges;
      // tolerances of determining inside/outside surface limits
      Core::LinAlg::Matrix<dim_side, 1, Core::CLN::ClnWrapper> scaled_tolerance_side;
      Core::LinAlg::Matrix<dim_edge, 1, Core::CLN::ClnWrapper> scaled_tolerance_edge;
      // compute location
      this->get_local_tolerance_side(INSIDE_OUTSIDE_TOLERANCE, scaled_tolerance_side);
      this->get_local_tolerance_side(SIDE_DETECTION_TOLERANCE, scaled_tolerance_side_touched_edges);
      this->get_local_tolerance_edge(INSIDE_OUTSIDE_TOLERANCE, scaled_tolerance_edge);
      // compute inside-outside relation
      if (within_limits<side_type>(clnxsi_, scaled_tolerance_side))
        side_location_ = PointOnSurfaceLoc(true, true);
      else
        side_location_ = PointOnSurfaceLoc(false, true);
      const Core::LinAlg::Matrix<dim_edge, 1, Core::CLN::ClnWrapper> clnxsi_line(
          clnxsi_.data() + dim_side, true);
      if (within_limits<edge_type>(clnxsi_line, scaled_tolerance_edge))
        edge_location_ = PointOnSurfaceLoc(true, true);
      else
        edge_location_ = PointOnSurfaceLoc(false, true);
      // NOTE: In compute intersection we don't actually use touched_edges now. left for the
      // future
      touched_edges_ids_.clear();
      if (side_location_.within_side())
      {
        get_edges_at<side_type>(clnxsi_, touched_edges_ids_, scaled_tolerance_side_touched_edges);
      }
    }

    // Evaluate difference between global coordinates in the intersection when computed based on
    // side and based on edge using conversion to very high reference precision
    Core::CLN::ClnWrapper compute_error(
        const Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<side_type>>& refside_xyz,
        const Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<edge_type>>& refedge_xyz,
        const Core::LinAlg::Matrix<dim_edge + dim_side, 1, Core::CLN::ClnWrapper>& loc_calc,
        int prec)
    {
      unsigned int prev_prec = Core::CLN::ClnWrapper::get_precision();
      Core::CLN::ClnWrapper::set_precision(prec);

      // Converting input arrays  to higher precision floating points
      Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<side_type>, Core::CLN::ClnWrapper>
          cln_refside_xyz;
      Core::LinAlg::Matrix<prob_dim, Core::FE::num_nodes<edge_type>, Core::CLN::ClnWrapper>
          cln_refedge_xyz;
      Core::LinAlg::Matrix<dim_edge + dim_side, 1, Core::CLN::ClnWrapper> cln_loc_calc;

      Core::CLN::conv_double_cln(refside_xyz, cln_refside_xyz, prec);
      Core::CLN::conv_double_cln(refedge_xyz, cln_refedge_xyz, prec);
      Core::CLN::update_precision(loc_calc, cln_loc_calc, prec);

      // Calculating interpolation from the shapefunction of the side
      Core::LinAlg::Matrix<Core::FE::num_nodes<side_type>, 1, Core::CLN::ClnWrapper> sideFunct;
      Core::LinAlg::Matrix<dim_edge + dim_side, 1, Core::CLN::ClnWrapper> cln_glob_calc_side;
      Core::LinAlg::Matrix<dim_edge + dim_side, 1, Core::CLN::ClnWrapper> cln_loc_calc_side(
          cln_loc_calc.data(), true);
      Core::FE::shape_function<side_type>(cln_loc_calc_side, sideFunct);

      for (unsigned int inode = 0; inode < Core::FE::num_nodes<side_type>; ++inode)
      {
        for (unsigned int isd = 0; isd < (dim_edge + dim_side); ++isd)
        {
          cln_glob_calc_side(isd) += cln_refside_xyz(isd, inode) * sideFunct(inode);
        }
      }

      // Calculating interpolation from the shapefunction of the edge
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> cln_glob_calc_edge;
      Core::LinAlg::Matrix<Core::FE::num_nodes<side_type>, 1, Core::CLN::ClnWrapper> edgeFunct;
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> cln_loc_calc_edge(
          cln_loc_calc.data() + Core::FE::dim<side_type>, true);

      Core::FE::shape_function<edge_type>(cln_loc_calc_edge, edgeFunct);
      for (unsigned int inode = 0; inode < Core::FE::num_nodes<edge_type>; ++inode)
      {
        for (unsigned int isd = 0; isd < (dim_edge + dim_side); ++isd)
        {
          cln_glob_calc_edge(isd) += cln_refedge_xyz(isd, inode) * edgeFunct(inode);
        }
      }

      // Evaluating error
      Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper>
          diff_vec;  // vector of difference between interpolation
      for (unsigned int i = 0; i < (dim_edge + dim_side); ++i)
        diff_vec(i) = cln_glob_calc_edge(i) - cln_glob_calc_side(i);

      // resetting precision to previous
      Core::CLN::ClnWrapper::set_precision(prev_prec);
      return diff_vec.norm2();
    }

    /// Try to fix the case that is close to the corner and outside - in that case point
    /// is close to corner and outside might be touching two edges but difference between it
    /// and corner point > tolerance - hence this must be corrected
    void fix_corner_case()
    {
      // if touches more than one edge
      if (touched_edges_ids_.size() >= 2)
      {
        if (side_location_.within_side())
        {
          Core::LinAlg::Matrix<dim_side, 1, Core::CLN::ClnWrapper> zero_tolerance_side;
          // this means we are on the outer boundary of the tolerance
          if (not within_limits<side_type>(clnxsi_, zero_tolerance_side))
          {
            // we transform our point to global tolerance and check if it is close enough to the
            // nodal point of the side, that is common point of the touching edges
            Core::LinAlg::Matrix<num_nodes_side, 1, Core::CLN::ClnWrapper> sideFunct;
            Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> xsi_global;
            xsi_global = 0.0;
            const Core::LinAlg::Matrix<dim_side, 1> xsi_side(xsi_.data(), true);
            Core::FE::shape_function<side_type>(xsi_side, sideFunct);
            // taking latest cln based coordinate
            const Core::LinAlg::Matrix<prob_dim, num_nodes_side, Core::CLN::ClnWrapper>&
                clnxyze_side_latest = clnxyze_side_;
            for (unsigned int inode = 0; inode < num_nodes_side; ++inode)
              for (unsigned int isd = 0; isd < prob_dim; ++isd)
                xsi_global(isd) += clnxyze_side_latest(isd, inode) * sideFunct(inode);

            // lame method for now. find minimum distance and check if it is closer than 1e-14
            Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> dist;
            Core::CLN::ClnWrapper min_dist;
            for (unsigned int inode = 0; inode < num_nodes_side; ++inode)
            {
              // access raw responsible for that node
              Core::LinAlg::Matrix<prob_dim, 1, Core::CLN::ClnWrapper> coord(
                  clnxyze_side_latest.data() + inode * prob_dim, true);
              dist.update(1.0, coord, -1.0, xsi_global);
              Core::CLN::ClnWrapper tmp_dist = dist.norm2();
              if ((inode == 0) or (tmp_dist < min_dist)) min_dist = tmp_dist;
            }

            // this should not be possible now
            if (min_dist > NON_TOPOLOGICAL_TOLERANCE)
            {
              touched_edges_ids_.clear();
              side_location_ = PointOnSurfaceLoc(false, true);
              std::stringstream msg;
              msg << "NOTICE: intersection point is close to 2 side edges, but too far from "
                     "the corner point"
                  << "\n Distance is " << min_dist;
              FOUR_C_THROW("{}", msg.str());
            }
          }
        }
        else
          FOUR_C_THROW("This should not be possible!");
      }
    }

    // Update global maximum number of allocations ( number of containers of particular byte
    // size) if allocations on this precisions were more. Also set what was the most frequent
    // size of allocation for the current iteration
    void update_memory_usage(int prec, int iter)
    {
#if DEBUG_MEMORY_ALLOCATION
      // if first run need also to run until the end
      if (first_run_)
      {
        Cut::MemorySingleton::getinstance().StopRecord();

        std::unordered_map<size_t, int>& allocation_map =
            Cut::MemorySingleton::getinstance().GetMemoryPattern();

        int max_num = 0;
        int size_max_num = 0;
        // update the global number of allocations to the maximum
        for (std::unordered_map<size_t, int>::iterator it = allocation_map.begin();
            it != allocation_map.end(); ++it)
        {
          size_t size = it->first;
          if (memory_allocations_intersection_[size] < allocation_map[size])
          {
            memory_allocations_intersection_[size] = allocation_map[size];
          }
          if (allocation_map[size] > max_num)
          {
            size_max_num = size;
            max_num = allocation_map[size];
          }
        }

        if (size_max_num != 0)
        {
          std::cout << "Size for the precision " << prec << " is " << size_max_num << std::endl;
          std::cout << "It is allocated " << max_num << " times " << std::endl;
          cln_sizes_[iter] = size_max_num;
        }
        else
          FOUR_C_THROW("This should not be possible!");

        Cut::MemorySingleton::getinstance().ResetAllocated();
      }
#endif
    }

    static std::vector<int> touched_edges_ids_;

    Core::LinAlg::Matrix<prob_dim, num_nodes_side, Core::CLN::ClnWrapper> clnxyze_side_;

    Core::LinAlg::Matrix<prob_dim, num_nodes_edge, Core::CLN::ClnWrapper> clnxyze_edge_;

    // holds reference to original value
    Core::LinAlg::Matrix<dim_edge + dim_side, 1>& xsi_;

    Core::LinAlg::Matrix<dim_edge + dim_side, 1, Core::CLN::ClnWrapper> clnxsi_;

    static bool first_run_;

    // if condition number is infinity or not
    bool cond_infinity_;

    // determines location of this point with respect to the edge that cuts it
    PointOnSurfaceLoc side_location_;

    // determines location of this point with respect to the side that cuts it
    PointOnSurfaceLoc edge_location_;

    // sizes of cln variables with different preciion
    static size_t cln_sizes_[CLN_LIMIT_ITER];
  };  // class ComputeIntersectionAdaptivePrecision

#endif

  /*--------------------------------------------------------------------------*/
  /** \brief generic strategy class for intersection of side and line/edge
   *
   *  inheritance diagram:
   *
   *  ComputeIntersection --> GenericComputeIntersection --> NewtonSolve
   *                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
   *  --> ComputeIntersectionStrategy --> EmptyNewtonStrategy  */
  template <class Strategy, unsigned prob_dim, Core::FE::CellType edge_type,
      Core::FE::CellType side_type, bool compute_cln = false,
      unsigned dim_edge = Core::FE::dim<edge_type>, unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_edge = Core::FE::num_nodes<edge_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class GenericComputeIntersection : Strategy
  {
   public:
    //! constructor
    GenericComputeIntersection(Core::LinAlg::Matrix<dim_edge + dim_side, 1>& xsi, bool checklimits)
        : Strategy(xsi, checklimits),
          cond_infinity_(false),
          distance_between_(0.0),
          xsi_(xsi)
#ifdef CUT_CLN_CALC
          ,
          checklimits_(checklimits)
#endif
    {
    }

    bool operator()(const Core::LinAlg::Matrix<prob_dim, num_nodes_side>& xyze_side,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_edge>& xyze_edge)
    {
      bool conv;
      this->setup(xyze_side, xyze_edge);
      conv = this->solve();
      bool got_topology_info = get_topology_information();

#ifdef CUT_CLN_CALC
      if (compute_cln or (!got_topology_info))
      {
#if DOUBLE_PLUS_CLN_COMPUTE
        bool major_fail = (not conv) or (cond_infinity_) or (!got_topology_info);
        bool result_fail = compute_error(xyze_side, xyze_edge, xsi_) > DOUBLE_LIMIT_ERROR;

        if (major_fail or result_fail)
        {
//             CutKernelStatistics::get_cut_kernel_statistics().cln_intersection_counter();
#endif
          {
            ComputeIntersectionAdaptivePrecision<
                NewtonSolve<ComputeIntersectionStrategy<false, prob_dim, edge_type, side_type,
                                dim_edge, dim_side, num_nodes_edge, num_nodes_side,
                                Core::CLN::ClnWrapper, ComputeIntersectionNoStaticMembers>,
                    dim_edge + dim_side>,
                prob_dim, edge_type, side_type>
                clncalc(xsi_, checklimits_);
            bool clnsolved = clncalc(xyze_side, xyze_edge);
            conv = clnsolved;
            edge_location_ = clncalc.get_edge_location();
            side_location_ = clncalc.get_side_location();
            cond_infinity_ = clncalc.is_condition_infinity();
            touched_edges_ids_.clear();
            const std::vector<int>& touched_edges_ids_cln = clncalc.get_touched_side_edges();
            touched_edges_ids_ = touched_edges_ids_cln;

            if (prob_dim > dim_edge + dim_side)
              distance_between_ = cln::double_approx(clncalc.distance_between().Value());

            Core::CLN::ClnWrapper::reset_precision();
          }
#ifdef CUSTOM_MEMORY_ALLOCATOR
          Cut::MemorySingleton::getinstance().Finalize();
#endif
#if DOUBLE_PLUS_CLN_COMPUTE
        }
        else
        {
          //             CutKernelStatistics::get_cut_kernel_statistics().double_intersection_counter();
        }
#endif
      }
#endif
      return conv;
    }

    const Core::LinAlg::Matrix<dim_edge + dim_side, 1>& local_coordinates() const
    {
      return Strategy::local_coordinates();
    }

    std::pair<bool, double> condition_number() { return Strategy::condition_number(); }

    bool surface_within_limits(double tolerance = REFERENCETOL) const
    {
      return within_limits<side_type>(local_coordinates(), tolerance);
    }

    bool line_within_limits(double tolerance = REFERENCETOL) const
    {
      const Core::LinAlg::Matrix<dim_edge, 1> xsi_line(local_coordinates().data() + dim_side, true);
      return within_limits<edge_type>(xsi_line, tolerance);
    }

    double get_tolerance() const { return Strategy::get_tolerance(); }

    void write_to_gmsh(std::ofstream& file) { Strategy::write_to_gmsh(file); }

    PointOnSurfaceLoc get_side_location() { return side_location_; }

    PointOnSurfaceLoc get_edge_location() { return edge_location_; }

    bool is_condition_infinity() { return cond_infinity_; }

    const std::vector<int>& get_touched_side_edges() { return touched_edges_ids_; }

    void get_touched_side_edges(std::vector<int>& inout)
    {
      inout.clear();
      inout = touched_edges_ids_;
    }

    // Get location of the point, based on the extended tolerance of 1e-14 for the diagonal
    // edge of the split triangle, and tolerance 0.0 for the normal edges.
    // Used to avoid possibility of middle point of the quad4 split to be missed because of
    // numerical errors
    PointOnSurfaceLoc get_side_location_triangle_split()
    {
      if (side_type != Core::FE::CellType::tri3)
        FOUR_C_THROW("This method only works for tri3 side. Current side is {}",
            Core::FE::cell_type_to_string(side_type).c_str());

      double distance_tolerance = TOPOLOGICAL_TOLERANCE;
      // get tolerance with 1e-14 around the triangle, but in local coordinates
      Core::LinAlg::Matrix<dim_side, 1> scaled_tolerance;
      this->get_local_tolerance_side(distance_tolerance, scaled_tolerance);
      // Diagonal one, corresponds to the middle one for both triangles, tolerance there should
      // be 1e-14
      Core::LinAlg::Matrix<3, 1> real_tolerance;
      real_tolerance(0) = scaled_tolerance(0);
      real_tolerance(1) = 0.0;
      real_tolerance(2) = 0.0;
      if (within_limits_split_quad<side_type>(xsi_, real_tolerance))
        side_location_triangle_split_ = PointOnSurfaceLoc(true, true);
      else
        side_location_triangle_split_ = PointOnSurfaceLoc(false, true);
      return side_location_triangle_split_;
    }

    // Distance between edges in edge-edge intersection
    double distance_between() const
    {
      if (prob_dim > dim_edge + dim_side)
        return distance_between_;
      else
        FOUR_C_THROW("This method should only be used for edge-edge intersection");
    }

   private:
    // Get local tolerance and based on it get location of the point on the surface
    // as well as touched edges
    bool get_topology_information()
    {
      Core::LinAlg::Matrix<dim_edge + dim_side, 1> xsi = Strategy::local_coordinates();
      // tolerance of getting nearby edges
      Core::LinAlg::Matrix<dim_side, 1> scaled_tolerance_side_touched_edges;
      // tolerances of determining inside/outside surface limits
      Core::LinAlg::Matrix<dim_edge, 1> scaled_tolerance_edge;
      Core::LinAlg::Matrix<dim_side, 1> scaled_tolerance_side;
      // compute location finally
      if ((not this->get_local_tolerance_edge(INSIDE_OUTSIDE_TOLERANCE, scaled_tolerance_edge)) or
          (not this->get_local_tolerance_side(INSIDE_OUTSIDE_TOLERANCE, scaled_tolerance_side)) or
          (not this->get_local_tolerance_side(
              SIDE_DETECTION_TOLERANCE, scaled_tolerance_side_touched_edges)))
      {
        return false;
      }
      // compute inside-outside relation
      if (within_limits<side_type>(xsi, scaled_tolerance_side))
        side_location_ = PointOnSurfaceLoc(true, true);
      else
        side_location_ = PointOnSurfaceLoc(false, true);
      const Core::LinAlg::Matrix<dim_edge, 1> xsi_line(xsi.data() + dim_side, true);
      if (within_limits<edge_type>(xsi_line, scaled_tolerance_edge))
        edge_location_ = PointOnSurfaceLoc(true, true);
      else
        edge_location_ = PointOnSurfaceLoc(false, true);
      if (prob_dim > dim_edge + dim_side) distance_between_ = Strategy::distance_between();

      touched_edges_ids_.clear();
      if (side_location_.within_side())
      {
        get_edges_at<side_type>(xsi, touched_edges_ids_, scaled_tolerance_side_touched_edges);
      }
      return true;
    }


    // Evaluate difference between global coordinates in the intersection when computed based on
    // side and based on edge
    double compute_error(const Core::LinAlg::Matrix<prob_dim, num_nodes_side>& side_xyz,
        const Core::LinAlg::Matrix<prob_dim, num_nodes_edge>& edge_xyz,
        const Core::LinAlg::Matrix<dim_edge + dim_side, 1>& locxyz)
    {
      // Calculating interpolation from the shapefunction of the side
      Core::LinAlg::Matrix<num_nodes_side, 1> sideFunct;
      Core::LinAlg::Matrix<dim_edge + dim_side, 1> globxyz_side;
      Core::LinAlg::Matrix<dim_edge + dim_side, 1> locxyz_side(locxyz.data(), true);
      Core::FE::shape_function<side_type>(locxyz_side, sideFunct);
      for (unsigned int inode = 0; inode < num_nodes_side; ++inode)
      {
        for (unsigned int isd = 0; isd < dim_edge + dim_side; ++isd)
        {
          globxyz_side(isd) += side_xyz(isd, inode) * sideFunct(inode);
        }
      }

      // Calculating interpolation from the shapefunction of the edge

      Core::LinAlg::Matrix<prob_dim, 1> globxyz_edge;
      Core::LinAlg::Matrix<num_nodes_edge, 1> edgeFunct;
      Core::LinAlg::Matrix<prob_dim, 1> locxyz_edge(locxyz.data() + dim_side, true);

      Core::FE::shape_function<edge_type>(locxyz_edge, edgeFunct);
      for (unsigned int inode = 0; inode < num_nodes_edge; ++inode)
      {
        for (unsigned int isd = 0; isd < dim_edge + dim_side; ++isd)
        {
          globxyz_edge(isd) += edge_xyz(isd, inode) * edgeFunct(inode);
        }
      }

      // Evaluating error
      Core::LinAlg::Matrix<prob_dim, 1> diffVec;
      for (unsigned int i = 0; i < (dim_edge + dim_side); ++i)
        diffVec(i) = globxyz_edge(i) - globxyz_side(i);
      double error = diffVec.norm2();
      if (error == 0) error = MINIMUM_DOUBLE_ERROR;
      std::pair<bool, double> cond_pair = this->condition_number();
      if (not cond_pair.first)
      {
        cond_infinity_ = true;
      }
      return (error * cond_pair.second);
    }

    // touched edges ids
    static std::vector<int> touched_edges_ids_;

    // determines location of this point with respect to the edge that cuts it
    PointOnSurfaceLoc side_location_;

    // determines location of this point with respect to the side that cuts it
    PointOnSurfaceLoc edge_location_;
    // determines location of this point, with extended tolerance over side, that is shared for
    // another triangle (only for tri3)
    PointOnSurfaceLoc side_location_triangle_split_;

    // if condition number is infinity or not
    bool cond_infinity_;

    // holds distance between lines in case of edge-edge intersection
    double distance_between_;

    // holds reference to original value passed
    Core::LinAlg::Matrix<dim_edge + dim_side, 1>& xsi_;

#ifdef CUT_CLN_CALC
    bool checklimits_;
#endif
  };  // class GenericComputeIntersection


  /*--------------------------------------------------------------------------*/
  /** \brief most derived strategy class for intersection of side and line
   *
   *  inheritance diagram:
   *
   *  ComputeIntersection --> GenericComputeIntersection --> NewtonSolve
   *  ^^^^^^^^^^^^^^^^^^^
   *  --> ComputeIntersectionStrategy --> EmptyNewtonStrategy  */
  template <unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
      bool compute_cln = false, unsigned dim_edge = Core::FE::dim<edge_type>,
      unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_edge = Core::FE::num_nodes<edge_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class ComputeIntersection
      : public GenericComputeIntersection<
            NewtonSolve<ComputeIntersectionStrategy<false, prob_dim, edge_type, side_type>,
                dim_edge + dim_side>,
            prob_dim, edge_type, side_type, compute_cln>
  {
   public:
    //! constructor
    ComputeIntersection(Core::LinAlg::Matrix<dim_edge + dim_side, 1>& xsi, bool checklimits = true)
        : GenericComputeIntersection<
              NewtonSolve<ComputeIntersectionStrategy<false, prob_dim, edge_type, side_type>,
                  dim_edge + dim_side>,
              prob_dim, edge_type, side_type, compute_cln>(xsi, checklimits)
    {
    }
  };  // class ComputeIntersection

  /*--------------------------------------------------------------------------*/
  /** \brief most derived strategy class for debugging intersection of side
   *  and line */
  template <unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
      unsigned dim_edge = Core::FE::dim<edge_type>, unsigned dim_side = Core::FE::dim<side_type>,
      unsigned num_nodes_edge = Core::FE::num_nodes<edge_type>,
      unsigned num_nodes_side = Core::FE::num_nodes<side_type>>
  class DebugComputeIntersection
      : public GenericComputeIntersection<
            NewtonSolve<DebugNewtonStrategy<
                            ComputeIntersectionStrategy<true, prob_dim, edge_type, side_type>,
                            dim_edge + dim_side>,
                dim_edge + dim_side>,
            prob_dim, edge_type, side_type>
  {
   public:
    //! constructor
    DebugComputeIntersection(
        Core::LinAlg::Matrix<dim_edge + dim_side, 1>& xsi, bool checklimits = true)
        : GenericComputeIntersection<
              NewtonSolve<DebugNewtonStrategy<
                              ComputeIntersectionStrategy<true, prob_dim, edge_type, side_type>,
                              dim_edge + dim_side>,
                  dim_edge + dim_side>,
              prob_dim, edge_type, side_type>(xsi, checklimits)
    {
    }

  };  // class DebugComputeIntersection

}  // namespace Cut::Kernel


// static members in KERNEL
// in compute position strategy
template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
    unsigned num_nodes_element, unsigned dim, typename FloatType>
Core::LinAlg::Matrix<num_nodes_element, 1, FloatType> Cut::Kernel::ComputePositionStaticMembers<
    debug, prob_dim, element_type, num_nodes_element, dim, FloatType>::funct_;
template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
    unsigned num_nodes_element, unsigned dim, typename FloatType>
Core::LinAlg::Matrix<prob_dim, num_nodes_element, FloatType>
    Cut::Kernel::ComputePositionStaticMembers<debug, prob_dim, element_type, num_nodes_element, dim,
        FloatType>::deriv1_;
template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
    unsigned num_nodes_element, unsigned dim, typename FloatType>
Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> Cut::Kernel::ComputePositionStaticMembers<debug,
    prob_dim, element_type, num_nodes_element, dim, FloatType>::A_;
template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
    unsigned num_nodes_element, unsigned dim, typename FloatType>
Core::LinAlg::Matrix<prob_dim, 1, FloatType> Cut::Kernel::ComputePositionStaticMembers<debug,
    prob_dim, element_type, num_nodes_element, dim, FloatType>::b_;
template <bool debug, unsigned prob_dim, Core::FE::CellType element_type,
    unsigned num_nodes_element, unsigned dim, typename FloatType>
Core::LinAlg::Matrix<prob_dim, 1, FloatType> Cut::Kernel::ComputePositionStaticMembers<debug,
    prob_dim, element_type, num_nodes_element, dim, FloatType>::dx_;

// in compute distance strategy
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<num_nodes_side, 1, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<debug,
    prob_dim, side_type, dim_side, num_nodes_side, FloatType>::sideFunct_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, num_nodes_side, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<
    debug, prob_dim, side_type, dim_side, num_nodes_side, FloatType>::sideDeriv1_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<2 * dim_side - 1, num_nodes_side, FloatType>
    Cut::Kernel::ComputeDistanceStaticMembers<debug, prob_dim, side_type, dim_side, num_nodes_side,
        FloatType>::sideDeriv2_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<debug,
    prob_dim, side_type, dim_side, num_nodes_side, FloatType>::A_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, prob_dim, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<debug,
    prob_dim, side_type, dim_side, num_nodes_side, FloatType>::B_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, 2 * dim_side - 1, FloatType>
    Cut::Kernel::ComputeDistanceStaticMembers<debug, prob_dim, side_type, dim_side, num_nodes_side,
        FloatType>::C_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, 1, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<debug,
    prob_dim, side_type, dim_side, num_nodes_side, FloatType>::b_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, 1, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<debug,
    prob_dim, side_type, dim_side, num_nodes_side, FloatType>::dx_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, 2, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<debug,
    prob_dim, side_type, dim_side, num_nodes_side, FloatType>::N_;
template <bool debug, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side, typename FloatType>
Core::LinAlg::Matrix<prob_dim, 2, FloatType> Cut::Kernel::ComputeDistanceStaticMembers<debug,
    prob_dim, side_type, dim_side, num_nodes_side, FloatType>::nvec_;

// in compute intersection strategy
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<num_nodes_side, 1, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::sideFunct_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<num_nodes_edge, 1, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::edgeFunct_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<dim_side, num_nodes_side, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::sideDeriv1_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<dim_edge, num_nodes_edge, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::edgeDeriv1_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<dim_edge + dim_side, dim_edge + dim_side, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::A_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<prob_dim, dim_edge + dim_side, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::B_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::b_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<prob_dim, 1, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::c_;
template <bool debug, unsigned prob_dim, Core::FE::CellType edge_type, Core::FE::CellType side_type,
    unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge, unsigned num_nodes_side,
    typename FloatType>
Core::LinAlg::Matrix<dim_edge + dim_side, 1, FloatType>
    Cut::Kernel::ComputeIntersectionStaticMembers<debug, prob_dim, edge_type, side_type, dim_edge,
        dim_side, num_nodes_edge, num_nodes_side, FloatType>::dx_;


// for generic compute intersection
template <class Strategy, unsigned prob_dim, Core::FE::CellType edge_type,
    Core::FE::CellType side_type, bool compute_cln, unsigned dim_edge, unsigned dim_side,
    unsigned num_nodes_edge, unsigned num_nodes_side>
std::vector<int> Cut::Kernel::GenericComputeIntersection<Strategy, prob_dim, edge_type, side_type,
    compute_cln, dim_edge, dim_side, num_nodes_edge, num_nodes_side>::touched_edges_ids_;

// for generic compute distance
template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type, bool compute_cln,
    unsigned dim_side, unsigned num_nodes_side>
std::vector<int> Cut::Kernel::GenericComputeDistance<Strategy, prob_dim, side_type, compute_cln,
    dim_side, num_nodes_side>::touched_edges_ids_;

template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type, bool compute_cln,
    unsigned dim_side, unsigned num_nodes_side>
std::vector<int> Cut::Kernel::GenericComputeDistance<Strategy, prob_dim, side_type, compute_cln,
    dim_side, num_nodes_side>::touched_nodes_ids_;

#ifdef CUT_CLN_CALC
// for generic compute position
template <class Strategy, unsigned prob_dim, Core::FE::CellType element_type, bool compute_cln,
    unsigned num_nodes_element, unsigned dim>
std::vector<int> Cut::Kernel::GenericComputePosition<Strategy, prob_dim, element_type, compute_cln,
    num_nodes_element, dim>::touched_edges_ids_;
#endif



#ifdef CUT_CLN_CALC

// Initializing for static arrays for AdaptivePrecision strategies


// ComputePosition
template <class Strategy, unsigned prob_dim, Core::FE::CellType element_type,
    unsigned num_nodes_element, unsigned dim>
bool Cut::Kernel::ComputePositionAdaptivePrecision<Strategy, prob_dim, element_type,
    num_nodes_element, dim>::first_run_ = true;
template <class Strategy, unsigned prob_dim, Core::FE::CellType element_type,
    unsigned num_nodes_element, unsigned dim>
size_t Cut::Kernel::ComputePositionAdaptivePrecision<Strategy, prob_dim, element_type,
    num_nodes_element, dim>::cln_sizes_[];

// Cut::Kernel::compute_distance

template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side>
std::vector<int> Cut::Kernel::ComputeDistanceAdaptivePrecision<Strategy, prob_dim, side_type,
    dim_side, num_nodes_side>::touched_edges_ids_;
template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side>
std::vector<int> Cut::Kernel::ComputeDistanceAdaptivePrecision<Strategy, prob_dim, side_type,
    dim_side, num_nodes_side>::touched_nodes_ids_;
template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side>
bool Cut::Kernel::ComputeDistanceAdaptivePrecision<Strategy, prob_dim, side_type, dim_side,
    num_nodes_side>::first_run_ = true;
template <class Strategy, unsigned prob_dim, Core::FE::CellType side_type, unsigned dim_side,
    unsigned num_nodes_side>
size_t Cut::Kernel::ComputeDistanceAdaptivePrecision<Strategy, prob_dim, side_type, dim_side,
    num_nodes_side>::cln_sizes_[];

// Cut::Kernel::ComputeIntersection

template <class Strategy, unsigned prob_dim, Core::FE::CellType edge_type,
    Core::FE::CellType side_type, unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge,
    unsigned num_nodes_side>
std::vector<int> Cut::Kernel::ComputeIntersectionAdaptivePrecision<Strategy, prob_dim, edge_type,
    side_type, dim_edge, dim_side, num_nodes_edge, num_nodes_side>::touched_edges_ids_;

template <class Strategy, unsigned prob_dim, Core::FE::CellType edge_type,
    Core::FE::CellType side_type, unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge,
    unsigned num_nodes_side>
bool Cut::Kernel::ComputeIntersectionAdaptivePrecision<Strategy, prob_dim, edge_type, side_type,
    dim_edge, dim_side, num_nodes_edge, num_nodes_side>::first_run_ = true;

template <class Strategy, unsigned prob_dim, Core::FE::CellType edge_type,
    Core::FE::CellType side_type, unsigned dim_edge, unsigned dim_side, unsigned num_nodes_edge,
    unsigned num_nodes_side>
size_t Cut::Kernel::ComputeIntersectionAdaptivePrecision<Strategy, prob_dim, edge_type, side_type,
    dim_edge, dim_side, num_nodes_edge, num_nodes_side>::cln_sizes_[];

#endif

FOUR_C_NAMESPACE_CLOSE

#endif
