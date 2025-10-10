// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PARAMS_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_params_base.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declaration.
namespace BeamInteraction
{
  class BeamToSolidSurfaceVisualizationOutputParams;
}

namespace BeamInteraction
{
  /**
   * \brief Class for beam to solid contact parameters.
   */
  class BeamToSolidSurfaceContactParams : public BeamToSolidParamsBase
  {
   public:
    /**
     * \brief Constructor.
     */
    BeamToSolidSurfaceContactParams();


    /**
     * \brief Initialize with the stuff coming from input file.
     */
    void init() override;

    /**
     * \brief Returns true if the coupling should be evaluated with FAD.
     */
    inline bool get_is_fad() const override { return true; }

    /**
     * \brief Returns the order of the FAD type.
     */
    int get_fad_order() const override;

    /**
     * \brief Returns the contact type.
     */
    inline Inpar::BeamToSolid::BeamToSolidSurfaceContact get_contact_type() const
    {
      return contact_type_;
    }

    /**
     * \brief Returns the information for the penalty law.
     */
    inline const PenaltyLawParameters& get_penalty_law() const { return penalty_law_data_; }

    /**
     * \brief Returns the configuration where the mortar contact is defined in.
     */
    inline Inpar::BeamToSolid::BeamToSolidSurfaceContactMortarDefinedIn
    get_beam_to_solid_surface_contact_mortar_defined_in() const
    {
      return mortar_contact_configuration_;
    }

    /**
     * \brief Returns a pointer to the visualization output parameters.
     * @return Pointer to visualization output parameters.
     */
    std::shared_ptr<BeamToSolidSurfaceVisualizationOutputParams>
    get_visualization_output_params_ptr()
    {
      return output_params_ptr_;
    }

   private:
    //! Type of contact constraints.
    Inpar::BeamToSolid::BeamToSolidSurfaceContact contact_type_;

    //! Penalty law parameters
    PenaltyLawParameters penalty_law_data_;

    //! Configuration where the mortar contact is defined
    Inpar::BeamToSolid::BeamToSolidSurfaceContactMortarDefinedIn mortar_contact_configuration_;

    //! Pointer to the visualization output parameters for beam to solid surface problems.
    std::shared_ptr<BeamToSolidSurfaceVisualizationOutputParams> output_params_ptr_;
  };

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
