// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PAIR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PAIR_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_pair_base.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /**
   * \brief Class for beam to surface surface contact based on manual variation of the gap function.
   * @tparam scalar_type Type for scalar DOF values.
   * @tparam beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @tparam surface Type from GeometryPair::ElementDiscretization... representing the surface.
   */
  template <typename ScalarType, typename Beam, typename Surface>
  class BeamToSolidSurfaceContactPairGapVariation
      : public BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>
  {
   protected:
    //! Shortcut to the base class.
    using base_class = BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceContactPairGapVariation();


    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     */
    void evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;
  };

  /**
   * \brief Class for beam to surface surface contact based on variation of the penalty potential.
   * @tparam scalar_type Type for scalar DOF values.
   * @tparam beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @tparam surface Type from GeometryPair::ElementDiscretization... representing the surface.
   */
  template <typename ScalarType, typename Beam, typename Surface>
  class BeamToSolidSurfaceContactPairPotential
      : public BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>
  {
   protected:
    //! Shortcut to the base class.
    using base_class = BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceContactPairPotential();


    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     */
    void evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
