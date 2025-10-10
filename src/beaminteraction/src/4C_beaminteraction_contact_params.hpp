// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_CONTACT_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_CONTACT_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_beamcontact_input.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  class BeamToBeamContactParams;
  class BeamToSphereContactParams;
  class BeamToSolidVolumeMeshtyingParams;
  class BeamToSolidSurfaceMeshtyingParams;
  class BeamToSolidSurfaceContactParams;
  class BeamContactRuntimeVisualizationOutputParams;
  struct BeamToSolidEdgeContactParameters;

  /*!
   *  */
  class BeamContactParams
  {
   public:
    //! constructor
    BeamContactParams();

    //! destructor
    virtual ~BeamContactParams() = default;

    //! builds a new beam_to_beam_contact_params object
    void build_beam_to_beam_contact_params();

    //! builds a new beam_to_sphere_contact_params object
    void build_beam_to_sphere_contact_params();

    //! builds a new beam_to_solid_volume_meshtying_params object
    void build_beam_to_solid_volume_meshtying_params();

    //! builds a new beam_to_solid_surface_meshtying_params object
    void build_beam_to_solid_surface_meshtying_params();

    //! builds a new beam_to_solid_surface_contact_params object
    void build_beam_to_solid_surface_contact_params();

    //! builds a new BeamContactRuntimeOutputParams object
    void build_beam_contact_runtime_output_params(double restart_time);

    //! builds the beam-to-solid edge contact parameters
    void build_beam_to_solid_edge_contact_params();


    inline std::shared_ptr<BeamInteraction::BeamToBeamContactParams> beam_to_beam_contact_params()
        const
    {
      return beam_to_beam_contact_params_;
    }

    inline std::shared_ptr<BeamInteraction::BeamToSphereContactParams>
    beam_to_sphere_contact_params() const
    {
      return beam_to_sphere_contact_params_;
    }

    inline std::shared_ptr<BeamInteraction::BeamToSolidVolumeMeshtyingParams>
    beam_to_solid_volume_meshtying_params() const
    {
      return beam_to_solid_volume_meshtying_params_;
    }

    inline std::shared_ptr<BeamInteraction::BeamToSolidSurfaceMeshtyingParams>
    beam_to_solid_surface_meshtying_params() const
    {
      return beam_to_solid_surface_meshtying_params_;
    }

    inline std::shared_ptr<BeamInteraction::BeamToSolidSurfaceContactParams>
    beam_to_solid_surface_contact_params() const
    {
      return beam_to_solid_surface_contact_params_;
    }

    inline std::shared_ptr<BeamInteraction::BeamContactRuntimeVisualizationOutputParams>
    beam_contact_runtime_visualization_output_params() const
    {
      return beam_contact_runtime_output_params_;
    }

    inline std::shared_ptr<BeamInteraction::BeamToSolidEdgeContactParameters>
    beam_to_solid_edge_contact_params() const
    {
      return beam_to_solid_edge_contact_params_;
    }

   private:
    //! pointer to the parameter class of beam-to-beam contact
    std::shared_ptr<BeamInteraction::BeamToBeamContactParams> beam_to_beam_contact_params_;

    //! pointer to the parameter class of beam-to-sphere contact
    std::shared_ptr<BeamInteraction::BeamToSphereContactParams> beam_to_sphere_contact_params_;

    //! pointer to the parameter class of beam-to-solid-volume contact
    std::shared_ptr<BeamInteraction::BeamToSolidVolumeMeshtyingParams>
        beam_to_solid_volume_meshtying_params_;

    //! pointer to the parameter class of beam-to-solid-surface mesh tying
    std::shared_ptr<BeamInteraction::BeamToSolidSurfaceMeshtyingParams>
        beam_to_solid_surface_meshtying_params_;

    //! pointer to the parameter class of beam-to-solid-surface contact
    std::shared_ptr<BeamInteraction::BeamToSolidSurfaceContactParams>
        beam_to_solid_surface_contact_params_;

    //! pointer to the parameter class of beam contact visualization output
    std::shared_ptr<BeamInteraction::BeamContactRuntimeVisualizationOutputParams>
        beam_contact_runtime_output_params_;

    //! pointer to the beam-to-solid edge contact parameters
    std::shared_ptr<BeamInteraction::BeamToSolidEdgeContactParameters>
        beam_to_solid_edge_contact_params_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
