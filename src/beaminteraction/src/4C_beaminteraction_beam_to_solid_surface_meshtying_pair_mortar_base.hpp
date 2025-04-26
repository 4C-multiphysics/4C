// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_MORTAR_BASE_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_MORTAR_BASE_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_base.hpp"
#include "4C_geometry_pair_scalar_types.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /**
   * \brief Base class for Mortar beam to surface surface mesh tying.
   * @tparam scalar_type Type for scalar variables.
   * @tparam beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @tparam surface Type from GeometryPair::ElementDiscretization... representing the surface.
   * @tparam mortar Type from BeamInteraction::ElementDiscretization... representing the mortar
   * shape functions.
   */
  template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
  class BeamToSolidSurfaceMeshtyingPairMortarBase
      : public BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam, Surface>
  {
   private:
    //! Shortcut to the base class.
    using base_class = BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam, Surface>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceMeshtyingPairMortarBase();


    /**
     * \brief This pair enforces constraints via a mortar-type method, which requires an own
     * assembly method (provided by the mortar manager).
     */
    inline bool is_assembly_direct() const override { return false; };

    /**
     * \brief Add the visualization of this pair to the beam to solid visualization output writer.
     * This will add mortar specific data to the output.
     * @param visualization_writer (out) Object that manages all visualization related data for beam
     * to solid pairs.
     * @param visualization_params (in) Parameter list (not used in this class).
     */
    void get_pair_visualization(
        std::shared_ptr<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
        Teuchos::ParameterList& visualization_params) const override;

    /**
     * \brief The mortar energy contribution will be calculated globally in the mortar manager.
     */
    double get_energy() const override { return 0.0; }

   protected:
    //! Number of rotational Lagrange multipliers.
    unsigned int n_mortar_rot_;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
