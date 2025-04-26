// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_2D_3D_PLANE_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_2D_3D_PLANE_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /**
   * \brief Class for 2D-3D beam-to-solid volume mesh tying based on a plane beam element. This
   * simplifies the triad construction and torsion free beam elements can be used.
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @param solid Type from GeometryPair::ElementDiscretization... representing the solid.
   */
  template <typename Beam, typename Solid>
  class BeamToSolidVolumeMeshtyingPair2D3DPlane
      : public BeamToSolidVolumeMeshtyingPair2D3DBase<
            GeometryPair::line_to_volume_scalar_type<Beam, Solid>, Beam, Solid>
  {
   private:
    //! Shortcut to the base class.
    using base_class = BeamToSolidVolumeMeshtyingPair2D3DBase<
        GeometryPair::line_to_volume_scalar_type<Beam, Solid>, Beam, Solid>;

    //! Type to be used for scalar AD variables. This can not be inherited from the base class.
    using scalar_type = typename base_class::scalar_type;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidVolumeMeshtyingPair2D3DPlane() = default;


    /*!
     *\brief things that need to be done in a separate loop before the actual evaluation loop
     *      over all contact pairs
     */
    void pre_evaluate() override;

    /**
     * \brief Evaluate this contact element pair.
     * @param forcevec1 (out) Force vector on element 1.
     * @param forcevec2 (out) Force vector on element 2.
     * @param stiffmat11 (out) Stiffness contributions on element 1 - element 1.
     * @param stiffmat12 (out) Stiffness contributions on element 1 - element 2.
     * @param stiffmat21 (out) Stiffness contributions on element 2 - element 1.
     * @param stiffmat22 (out) Stiffness contributions on element 2 - element 2.
     * @return True if pair is in contact.
     */
    bool evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
        Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
        Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
        Core::LinAlg::SerialDenseMatrix* stiffmat22) override;

   protected:
    /**
     * \brief Get the triad of the beam at the parameter coordinate xi (derived)
     */
    void get_triad_at_xi_double(const double xi, Core::LinAlg::Matrix<3, 3, double>& triad,
        const bool reference) const override;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
