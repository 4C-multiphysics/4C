// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_LINE_PROJECTION_HPP
#define FOUR_C_GEOMETRY_PAIR_LINE_PROJECTION_HPP


#include "4C_config.hpp"

#include "4C_geometry_pair_element.hpp"

#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace Core::LinAlg
{
  template <unsigned int rows, unsigned int cols, class ValueType>
  class Matrix;
}
namespace Core::FE
{
  struct IntegrationPoints1D;
}  // namespace Core::FE
namespace GEOMETRYPAIR
{
  enum class ProjectionResult;

  template <typename ScalarType>
  class ProjectionPoint1DTo3D;

  template <typename ScalarType>
  class LineSegment;
}  // namespace GEOMETRYPAIR


namespace GEOMETRYPAIR
{
  /**
   * \brief This class contains static methods for common line-to-xxx algorithms.
   *
   * This class contains common methods for line-to-3D (volume or surface with normal direction)
   * interactions.
   *
   * The class methods have a variable number of input arguments (using the c++11 feature
   * typename...) by doing so we can handle a variable amount of arguments for the methods, e.g.
   * surface elements can have the optional argument for the averaged nodal normals.
   *
   * @tparam pair_type Class of the line-to-xxx pair that the segmentation / Gauss-point-projection
   * will be performed on.
   */
  template <typename PairType>
  class LineTo3DBase
  {
   private:
    //! Alias for the scalar type.
    using scalar_type = typename PairType::t_scalar_type;

    //! Alias for the line type.
    using line = typename PairType::t_line;

    //! Alias for the other geometry type.
    using other = typename PairType::t_other;

   public:
    /**
     * \brief Project a point on the line to the other geometry element.
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_other (in) Degrees of freedom for the other geometry.
     * @param eta (in) Parameter coordinate on the line.
     * @param xi (in/out) Parameter coordinates in the other geometry. The given values are the
     * start values for the Newton iteration.
     * @param projection_result (out) Flag for the result of the projection.
     */
    static void project_point_on_line_to_other(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other, const scalar_type& eta,
        Core::LinAlg::Matrix<3, 1, scalar_type>& xi, ProjectionResult& projection_result);

    /**
     * \brief Project multiple points on the line to the other geometry. The value of eta and xi in
     * the projection points is the start value for the iteration.
     *
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line geometry.
     * @param element_data_line (in) Degrees of freedom for the other geometry.
     * @param projection_points (in/out) Vector with projection points. The given values for eta and
     * xi are the start values for the iteration.
     * @param n_projections_valid (out) Number of valid projections.
     * @param n_projections (out) Number of points, where the nonlinear system could be solved.
     */
    static void project_points_on_line_to_other(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other,
        std::vector<ProjectionPoint1DTo3D<scalar_type>>& projection_points,
        unsigned int& n_projections_valid, unsigned int& n_projections);

    /**
     * \brief Project multiple points on the line to the other geometry. The value of eta and xi in
     * the projection points is the start value for the iteration.
     *
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_other (in) Degrees of freedom for the other geometry.
     * @param projection_points (in/out) Vector with projection points. The given values for eta and
     * xi are the start values for the iteration.
     * @param n_projections_valid (out) Number of valid projections.
     */
    static void project_points_on_line_to_other(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other,
        std::vector<ProjectionPoint1DTo3D<scalar_type>>& projection_points,
        unsigned int& n_projections_valid);

    /**
     * \brief Project Gauss points on the line segment to the other geometry. If not all points can
     * be projected, an error is thrown. Only projections inside the two elements are considered
     * valid.
     *
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_other (in) Degrees of freedom for the other geometry.
     * @param segment (in/out) Vector with found projection points.
     */
    static void project_gauss_points_on_segment_to_other(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other,
        LineSegment<scalar_type>& segment);

    /**
     * \brief Intersect a line with all surfaces of an other geometry. Use default starting values
     * for eta and xi.
     *
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_other (in) Degrees of freedom for the other geometry.
     * @param intersection_points (out) vector with the found surface intersections.
     */
    static void intersect_line_with_other(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other,
        std::vector<ProjectionPoint1DTo3D<scalar_type>>& intersection_points);
  };


  /**
   * \brief This class contains static methods for Gauss point projection algorithms.
   * @tparam pair_type Class of the line-to-xxx pair that the segmentation / Gauss-point-projection
   * will be performed on.
   */
  template <typename PairType>
  class LineTo3DGaussPointProjection : public LineTo3DBase<PairType>
  {
   private:
    //! Alias for the scalar type.
    using scalar_type = typename PairType::t_scalar_type;

    //! Alias for the line type.
    using line = typename PairType::t_line;

    //! Alias for the other geometry type.
    using other = typename PairType::t_other;

   public:
    /**
     * \brief Try to project all Gauss points to the geometry.
     *
     * Only points are checked that do not
     * already have a valid projection in the projection tracker of the evaluation data container.
     * Eventually needed segmentation at lines poking out of the other geometry is done in the
     * Evaluate method.
     *
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_other (in) Degrees of freedom for the geometry.
     * @param segments (out) Vector with the segments of this line to xxx pair.
     */
    static void pre_evaluate(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other,
        std::vector<LineSegment<scalar_type>>& segments);

    /**
     * \brief Check if a Gauss point projected valid for this pair in pre_evaluate.
     *
     * If so, all Gauss points have to project valid (in the tracker, since some can be valid on
     * other pairs). If not all project, the beam pokes out of the other geometry and in this method
     * segmentation is performed.
     *
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_other (in) Degrees of freedom for the other geometry.
     * @param segments (out) Vector with the segments of this line-to-xxx pair.
     */
    static void evaluate(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other,
        std::vector<LineSegment<scalar_type>>& segments);

   private:
    /**
     * \brief Get the line projection vector for the line element in the pair.
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @return  reference to line projection vector.
     */
    static std::vector<bool>& get_line_projection_vector(const PairType* pair);
  };


  /**
   * \brief This class contains static methods for segmentation algorithms.
   * @tparam pair_type Class of the line-to-xxx pair that the segmentation / Gauss-point-projection
   * will be performed on.
   */
  template <typename PairType>
  class LineTo3DSegmentation : public LineTo3DBase<PairType>
  {
   private:
    //! Alias for the scalar type.
    using scalar_type = typename PairType::t_scalar_type;

    //! Alias for the line type.
    using line = typename PairType::t_line;

    //! Alias for the other geometry type.
    using other = typename PairType::t_other;

   public:
    /**
     * \brief This method performs the segmentation of the line with the other geometry.
     *
     * First every search point on the beam is projected to the other geometry. For every search
     * point that has a projection (Newton converged, point does not have to be inside the other
     * geometry), surface intersections are checked for all surfaces of the other geometry (with the
     * search point as start point for the Newton iteration). The resulting surface intersections
     * points are sorted and only unique points are kept (since some intersections are found from
     * multiple starting points). Between two successive intersection points there is a line segment
     * (either inside or outside of the beam). To check whether the segment is part of this pair, a
     * point between two successive intersection points is projected, and if it projects valid, the
     * segment is assumed to be in this pair.
     *
     * There can be the special case when two consecutive segments are inside the pair, this happens
     * when the solid is exactly along the surface of a other geometry and a search point lies on
     * that surface.
     *
     * This approach should work for almost all possible cases. One case that can occur, is when
     * none of the search points project, but the line intersects the other geometry. In this case
     * more search points are the (expensive!) solution.
     *
     * In the case where a line lies on the interface between two geometries (e.g. the surface
     * between two volumes), segments on both other geometries are valid. To avoid this case a
     * global segment tracker is kept in the evaluation data, that allows a segment only once on all
     * line-to-xxx pairs. This does not work in the very special case, that the other geometry
     * have non-conforming discretization.
     *
     * @param pair (in) Pointer to the pair object that is being evaluated.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_other (in) Degrees of freedom for the other geometry.
     * @param segments (out) Vector with the segments of this line to other geometry pair.
     */
    static void evaluate(const PairType* pair,
        const ElementData<line, scalar_type>& element_data_line,
        const ElementData<other, scalar_type>& element_data_other,
        std::vector<LineSegment<scalar_type>>& segments);

   private:
    /**
     * \brief Get the segment tracking vector for the line element the pair. The type is double
     * for the segments there, as this is only needed for comparison purposes. The segment values of
     * the tracker should NOT be used for further calculation as AD will not work correctly in this
     * case.
     * @return Reference to segment tracking vector.
     */
    static std::set<LineSegment<double>>& get_segment_tracking_set(const PairType* pair);
  };

}  // namespace GEOMETRYPAIR

FOUR_C_NAMESPACE_CLOSE

#endif
