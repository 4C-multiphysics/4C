// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_LINE_TO_VOLUME_SEGMENTATION_HPP
#define FOUR_C_GEOMETRY_PAIR_LINE_TO_VOLUME_SEGMENTATION_HPP


#include "4C_config.hpp"

#include "4C_geometry_pair_line_to_volume.hpp"

FOUR_C_NAMESPACE_OPEN

namespace GeometryPair
{
  /**
   * \brief Class that handles the geometrical interactions of a line and a volume by segmenting the
   * line at all points where it intersects the volume.
   * @param scalar_type Type that will be used for scalar values.
   * @param line Type of line element.
   * @param volume Type of volume element.
   */
  template <typename ScalarType, typename Line, typename Volume>
  class GeometryPairLineToVolumeSegmentation
      : public GeometryPairLineToVolume<ScalarType, Line, Volume>
  {
   public:
    //! Public alias for scalar type so that other classes can use this type.
    using t_scalar_type = ScalarType;

    //! Public alias for line type so that other classes can use this type.
    using t_line = Line;

    //! Public alias for volume type so that other classes can use this type.
    using t_other = Volume;

   public:
    /**
     * \brief Constructor.
     */
    GeometryPairLineToVolumeSegmentation(const Core::Elements::Element* element1,
        const Core::Elements::Element* element2,
        const std::shared_ptr<GeometryPair::LineTo3DEvaluationData>& evaluation_data);


    /**
     * \brief This method performs the segmentation of the line with the volume.
     * @param element_data_line (in) Degrees of freedom for the line.
     * @param element_data_volume (in) Degrees of freedom for the volume.
     * @param segments (out) Vector with the segments of this line to volume pair.
     */
    void evaluate(const ElementData<Line, ScalarType>& element_data_line,
        const ElementData<Volume, ScalarType>& element_data_volume,
        std::vector<LineSegment<ScalarType>>& segments) const override;
  };
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif
