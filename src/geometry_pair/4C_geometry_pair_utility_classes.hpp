// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_UTILITY_CLASSES_HPP
#define FOUR_C_GEOMETRY_PAIR_UTILITY_CLASSES_HPP


#include "4C_config.hpp"

#include "4C_geometry_pair_constants.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace GEOMETRYPAIR
{
  /**
   * \brief Result of a projection with the geometry pairs.
   */
  enum class ProjectionResult
  {
    //! Default value
    none,
    //! System of equations could not be solved.
    projection_not_found,
    //! Projection found, but the parameter coordinates are not all valid.
    projection_found_not_valid,
    //! Projection found and the parameter coordinates are valid.
    projection_found_valid
  };

  /**
   * \brief Class that represents a projection from a 1D structure (usually a line) to a 3D
   * structure (can be volume, as well as surface including normal direction).
   * @tparam scalar_type Scalar type of the parameter coordinate values.
   */
  template <typename ScalarType>
  class ProjectionPoint1DTo3D
  {
   public:
    /**
     * \brief Constructor.
     * @param eta Parameter coordinate on line.
     * @param xi Parameter coordinates in volume.
     * @param gauss_weight Gauss weight for this point.
     */
    ProjectionPoint1DTo3D(
        ScalarType eta, Core::LinAlg::Matrix<3, 1, ScalarType> xi, double gauss_weight)
        : eta_(eta),
          xi_(xi),
          projection_result_(ProjectionResult::none),
          gauss_weight_(gauss_weight),
          intersection_face_(-1),
          eta_cross_section_(Core::LinAlg::Initialization::zero),
          is_cross_section_point_(false) {};

    /**
     * \brief Constructor.
     * @param eta Parameter coordinate on line.
     * @param xi Parameter coordinates in volume.
     */
    ProjectionPoint1DTo3D(ScalarType eta, Core::LinAlg::Matrix<3, 1, ScalarType> xi)
        : ProjectionPoint1DTo3D(eta, xi, -1.) {};

    /**
     * \brief Constructor.
     * @param eta Parameter coordinate on the line.
     */
    ProjectionPoint1DTo3D(ScalarType eta)
        : ProjectionPoint1DTo3D(eta,
              Core::LinAlg::Matrix<3, 1, ScalarType>(Core::LinAlg::Initialization::zero), -1.) {};

    /**
     * \brief Empty constructor.
     */
    ProjectionPoint1DTo3D() : ProjectionPoint1DTo3D(0.0) {};

    /**
     * \brief Destructor.
     */
    virtual ~ProjectionPoint1DTo3D() = default;

    /**
     * \brief Construct the point from another point where all scalar values are cast to double.
     * @param point_double Projection point with scalar type double.
     */
    template <typename ScalarTypeOther>
    inline void set_from_other_point_double(
        const ProjectionPoint1DTo3D<ScalarTypeOther>& point_other)
    {
      eta_ = Core::FADUtils::cast_to_double(point_other.get_eta());
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        xi_(i_dim) = Core::FADUtils::cast_to_double(point_other.get_xi()(i_dim));
      projection_result_ = point_other.get_projection_result();
      gauss_weight_ = point_other.get_gauss_weight_no_check();
      intersection_face_ = point_other.get_intersection_face();
    }

    /**
     * \brief Set the parameter coordinate on the line.
     */
    inline void set_eta(const ScalarType& eta) { eta_ = eta; };

    /**
     * \brief Get the parameter coordinate on the line.
     */
    inline const ScalarType& get_eta() const { return eta_; };

    /**
     * \brief Get a mutable reference to the parameter coordinate on the line.
     */
    inline ScalarType& get_eta() { return eta_; };

    /**
     * \brief Set the parameter coordinates in the volume.
     */
    inline void set_xi(const Core::LinAlg::Matrix<3, 1, ScalarType>& xi) { xi_ = xi; };

    /**
     * \brief Get the parameter coordinates in the volume.
     */
    inline const Core::LinAlg::Matrix<3, 1, ScalarType>& get_xi() const { return xi_; };

    /**
     * \brief Get the parameter coordinates in the volume as a reference.
     */
    inline Core::LinAlg::Matrix<3, 1, ScalarType>& get_xi() { return xi_; };

    /**
     * \brief Set the parameter coordinates in the cross section.
     */
    inline void set_eta_cross_section(Core::LinAlg::Matrix<2, 1, ScalarType> eta_cross_section)
    {
      eta_cross_section_ = eta_cross_section;
      is_cross_section_point_ = true;
    };

    /**
     * \brief Get the parameter coordinates in the cross section.
     */
    inline Core::LinAlg::Matrix<2, 1, ScalarType> get_eta_cross_section() const
    {
      if (!is_cross_section_point_) FOUR_C_THROW("The cross section coordinate has not been set!");
      return eta_cross_section_;
    };

    /**
     * Set the projection result for this projection point.
     * @param projection_result
     */
    inline void set_projection_result(ProjectionResult projection_result)
    {
      projection_result_ = projection_result;
    }

    /**
     * Get the projection result for this projection point.
     */
    inline ProjectionResult get_projection_result() const { return projection_result_; }

    /**
     * Get the projection result for this projection point as a reference.
     */
    inline ProjectionResult& get_projection_result() { return projection_result_; }

    /**
     * \brief Set the Gauss weight for this point.
     * @param gauss_weight
     */
    inline void set_gauss_weight(double gauss_weight) { gauss_weight_ = gauss_weight; };

    /**
     * \brief Get the Gauss weight for this point, if none is defined, an error is thrown.
     */
    inline double get_gauss_weight() const
    {
      if (gauss_weight_ < 0.)
        FOUR_C_THROW(
            "Negative Gauss weight not possible. Probably the default value was not overwritten!");
      return gauss_weight_;
    }

    /**
     * \brief Get the Gauss weight for this point.
     */
    inline double get_gauss_weight_no_check() const { return gauss_weight_; }

    /**
     * \brief Set the index of the intersection face.
     */
    inline void set_intersection_face(const int intersection_face)
    {
      intersection_face_ = intersection_face;
    }

    /**
     * \brief Get the index of the intersection face.
     */
    inline int get_intersection_face() const { return intersection_face_; }

    /**
     * \brief Overloaded $<$ operator.
     * @param lhs
     * @param rhs
     * @return True if smaller, false if larger.
     */
    friend bool operator<(
        const ProjectionPoint1DTo3D<ScalarType>& lhs, const ProjectionPoint1DTo3D<ScalarType>& rhs)
    {
      if (lhs.get_eta() < rhs.get_eta() - Constants::projection_xi_eta_tol)
        return true;
      else
        return false;
    };

    /**
     * \brief Overloaded $>$ operator.
     * @param lhs
     * @param rhs
     * @return False if smaller, true if larger.
     */
    friend bool operator>(
        const ProjectionPoint1DTo3D<ScalarType>& lhs, const ProjectionPoint1DTo3D<ScalarType>& rhs)
    {
      if (lhs.get_eta() > rhs.get_eta() + Constants::projection_xi_eta_tol)
        return true;
      else
        return false;
    };

   private:
    //! Parameter coordinate on line.
    ScalarType eta_;

    //! Parameter coordinates in volume.
    Core::LinAlg::Matrix<3, 1, ScalarType> xi_;

    //! Projection result.
    ProjectionResult projection_result_;

    //! Gauss weight for this point.
    double gauss_weight_;

    //! If this point is an intersection point, this is the index of the local face that the
    //! intersection occurs on.
    int intersection_face_;

    //! Parameter coordinates in the cross section.
    Core::LinAlg::Matrix<2, 1, ScalarType> eta_cross_section_;

    //! Flag if this is a point on a cross section.
    bool is_cross_section_point_;
  };

  /**
   * \brief Class to manage a segment on a line.
   * @tparam scalar_type Scalar type of the parameter coordinate values.
   */
  template <typename ScalarType>
  class LineSegment
  {
   public:
    /**
     * \brief Default constructor, the segment is from -1 to 1.
     */
    LineSegment() : LineSegment(ScalarType(-1.0), ScalarType(1.0)) {};

    /**
     * \brief Constructor. Set the range of the segment.
     * @param start_point
     * @param end_point
     */
    LineSegment(
        ProjectionPoint1DTo3D<ScalarType> start_point, ProjectionPoint1DTo3D<ScalarType> end_point)
        : start_point_(start_point), end_point_(end_point), segment_projection_points_()
    {
      // Sanity check that eta_a is larger than eta_b.
      if (!(get_eta_a() < get_eta_b()))
        FOUR_C_THROW(
            "The segment is created with eta_a={} and eta_b={}, this is not possible, as eta_a "
            "has "
            "to be smaller than eta_b!",
            Core::FADUtils::cast_to_double(get_eta_a()),
            Core::FADUtils::cast_to_double(get_eta_b()));
    }

    /**
     * \brief Get the length of the segment in parameter coordinates.
     * @return Segment length.
     */
    inline ScalarType get_segment_length() const { return get_eta_b() - get_eta_a(); }

    /**
     * \brief Return a const reference to eta start.
     */
    inline const ScalarType& get_eta_a() const { return start_point_.get_eta(); };

    /**
     * \brief Return a const reference to eta end.
     */
    inline const ScalarType& get_eta_b() const { return end_point_.get_eta(); };

    /**
     * \brief Return a const reference to the start point.
     */
    inline const ProjectionPoint1DTo3D<ScalarType>& get_start_point() const
    {
      return start_point_;
    };

    /**
     * \brief Return a mutable reference to the start point.
     */
    inline ProjectionPoint1DTo3D<ScalarType>& get_start_point() { return start_point_; };

    /**
     * \brief Return a const reference to the end point.
     */
    inline const ProjectionPoint1DTo3D<ScalarType>& get_end_point() const { return end_point_; };

    /**
     * \brief Return a mutable reference to the end point.
     */
    inline ProjectionPoint1DTo3D<ScalarType>& get_end_point() { return end_point_; };

    /**
     * \brief Add a projection point to the projection point vector.
     * @param projection_point projection point to add to the end of the vector.
     */
    inline void add_projection_point(ProjectionPoint1DTo3D<ScalarType> projection_point)
    {
      segment_projection_points_.push_back(projection_point);
    }

    /**
     * \brief Return the number of projection points in this segment.
     * @return Number of projection points.
     */
    inline unsigned int get_number_of_projection_points() const
    {
      return segment_projection_points_.size();
    }

    /**
     * \brief Return a const reference to the projection points in this segment.
     * @return Reference to projection point vector.
     */
    inline const std::vector<ProjectionPoint1DTo3D<ScalarType>>& get_projection_points() const
    {
      return segment_projection_points_;
    }

    /**
     * \brief Return a mutable reference to the projection points in this segment.
     * @return Reference to projection point vector.
     */
    inline std::vector<ProjectionPoint1DTo3D<ScalarType>>& get_projection_points()
    {
      return segment_projection_points_;
    }

    /**
     * \brief Return a vector with all points of this segment:
     * {start_point, projection_point_1, projection_point_2, ..., end_point}
     */
    std::vector<ProjectionPoint1DTo3D<ScalarType>> get_all_segment_points() const
    {
      std::vector<ProjectionPoint1DTo3D<ScalarType>> all_points;
      all_points.reserve(segment_projection_points_.size() + 2);
      all_points.push_back(start_point_);
      all_points.insert(
          all_points.end(), segment_projection_points_.begin(), segment_projection_points_.end());
      all_points.push_back(end_point_);
      return all_points;
    }

    /**
     * \brief Overloaded $<$ operator.
     * @param lhs
     * @param rhs
     * @return True if smaller, false if larger. Throw error if the segments are overlapping.
     */
    friend bool operator<(const LineSegment<ScalarType>& lhs, const LineSegment<ScalarType>& rhs)
    {
      if (lhs.get_eta_b() < rhs.get_eta_a() + Constants::projection_xi_eta_tol)
        return true;
      else if (lhs.get_eta_a() > rhs.get_eta_b() - Constants::projection_xi_eta_tol)
      {
        // The segments do not overlap.
      }
      else if (abs(lhs.get_eta_a() - rhs.get_eta_a()) < Constants::projection_xi_eta_tol &&
               abs(lhs.get_eta_b() - rhs.get_eta_b()) < Constants::projection_xi_eta_tol)
      {
        // The segments are equal.
      }
      else
        FOUR_C_THROW("The two segments are overlapping. This is fatal!");

      return false;
    };

    /**
     * \brief Overloaded $>$ operator.
     * @param lhs
     * @param rhs
     * @return False if smaller, true if larger. Throw error if the segments are overlapping.
     */
    friend bool operator>(const LineSegment<ScalarType>& lhs, const LineSegment<ScalarType>& rhs)
    {
      if (lhs.get_eta_a() > rhs.get_eta_b() - Constants::projection_xi_eta_tol)
        return true;
      else if (lhs.get_eta_b() < rhs.get_eta_a() + Constants::projection_xi_eta_tol)
      {
        // The segments do not overlap.
      }
      else if (abs(lhs.get_eta_a() - rhs.get_eta_a()) < Constants::projection_xi_eta_tol &&
               abs(lhs.get_eta_b() - rhs.get_eta_b()) < Constants::projection_xi_eta_tol)
      {
        // The segments are equal.
      }
      else
        FOUR_C_THROW("The two segments are overlapping. This is fatal!");

      return false;
    };

   private:
    //! Start point of the segment.
    ProjectionPoint1DTo3D<ScalarType> start_point_;

    //! Endpoint of the segment.
    ProjectionPoint1DTo3D<ScalarType> end_point_;

    //! Vector to store projection points for this segment.
    std::vector<ProjectionPoint1DTo3D<ScalarType>> segment_projection_points_;
  };
}  // namespace GEOMETRYPAIR


FOUR_C_NAMESPACE_CLOSE

#endif
