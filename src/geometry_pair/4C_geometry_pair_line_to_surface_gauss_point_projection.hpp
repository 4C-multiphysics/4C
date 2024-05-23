/*----------------------------------------------------------------------*/
/*! \file

\brief Line to surface interaction with simple Gauss point projection and boundary segmentation.

\level 1
*/
// End doxygen header.


#ifndef FOUR_C_GEOMETRY_PAIR_LINE_TO_SURFACE_GAUSS_POINT_PROJECTION_HPP
#define FOUR_C_GEOMETRY_PAIR_LINE_TO_SURFACE_GAUSS_POINT_PROJECTION_HPP


#include "4C_config.hpp"

#include "4C_geometry_pair_line_to_surface.hpp"

FOUR_C_NAMESPACE_OPEN

namespace GEOMETRYPAIR
{
  /**
   * \brief Class that handles the geometrical interactions of a line and a surface by projecting
   * Gauss points from the line to the volume. In case a line pokes out of the volumes it is
   * segmented.
   * @tparam scalar_type Type that will be used for scalar values.
   * @tparam line Type of line element.
   * @tparam volume Type of volume element.
   */
  template <typename scalar_type, typename line, typename surface>
  class GeometryPairLineToSurfaceGaussPointProjection
      : public GeometryPairLineToSurface<scalar_type, line, surface>
  {
   public:
    //! Public alias for scalar type so that other classes can use this type.
    using t_scalar_type = scalar_type;

    //! Public alias for line type so that other classes can use this type.
    using t_line = line;

    //! Public alias for surface type so that other classes can use this type.
    using t_other = surface;

   public:
    /**
     * \brief Constructor.
     */
    GeometryPairLineToSurfaceGaussPointProjection(const DRT::Element* element1,
        const DRT::Element* element2,
        const Teuchos::RCP<GEOMETRYPAIR::LineToSurfaceEvaluationData>&
            line_to_surface_evaluation_data);


    /**
     * \brief Try to project all Gauss points to the surface.
     *
     * Only points are checked that do not
     * already have a valid projection in the projection tracker of the evaluation data container.
     * Eventually needed segmentation at lines poking out of the volume is done in the Evaluate
     * method.
     *
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_surface (in) Degrees of freedom for the volume.
     * @param segments (out) Vector with the segments of this line to volume pair.
     * @param nodal_normals (in) Optional - Normals on the nodes.
     */
    void PreEvaluate(const ElementData<line, scalar_type>& element_data_line,
        const ElementData<surface, scalar_type>& element_data_surface,
        std::vector<LineSegment<scalar_type>>& segments) const override;

    /**
     * \brief Check if a Gauss point projected valid for this pair in PreEvaluate.
     *
     * If so, all Gauss points have to project valid (in the tracker, since some can be valid on
     * other pairs). If not all project, the beam pokes out of the volumes and in this method
     * segmentation is performed.
     *
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_surface (in) Degrees of freedom for the volume.
     * @param segments (out) Vector with the segments of this line to volume pair.
     * @param nodal_normals (in) Optional - Normals on the nodes.
     */
    void Evaluate(const ElementData<line, scalar_type>& element_data_line,
        const ElementData<surface, scalar_type>& element_data_surface,
        std::vector<LineSegment<scalar_type>>& segments) const override;

   private:
    /**
     * \brief Get the line projection vector for the line element in this pair.
     * @return  reference to line projection vector.
     */
    std::vector<bool>& get_line_projection_vector() const;
  };
}  // namespace GEOMETRYPAIR

FOUR_C_NAMESPACE_CLOSE

#endif
