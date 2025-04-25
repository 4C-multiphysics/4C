// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_GAUSS_POINT_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_GAUSS_POINT_HPP


#include "4C_config.hpp"

#include "4C_fbi_beam_to_fluid_meshtying_pair_base.hpp"

namespace
{
  class BeamToFluidMeshtyingPairGPTSTest;
}

FOUR_C_NAMESPACE_OPEN

namespace FBI
{
  class PairFactory;
}
namespace BeamInteraction
{
  /**
   * \brief Class for beam to fluid meshtying using Gauss point projection.
   *
   * \param[in] beam Type from GeometryPair::ElementDiscretization representing the beam.
   * \param[in] volume Type from GeometryPair::ElementDiscretization... representing the fluid.
   */
  template <typename Beam, typename Fluid>
  class BeamToFluidMeshtyingPairGaussPoint : public BeamToFluidMeshtyingPairBase<Beam, Fluid>
  {
    friend FBI::PairFactory;
    friend BeamToFluidMeshtyingPairGPTSTest;

   public:
    /**
     * \brief Evaluate this contact element pair.
     *
     * \param[inout] forcevec1 (out) Force vector on element 1.
     * \param[inout] forcevec2 (out) Force vector on element 2.
     * \param[inout] stiffmat11 (out) Stiffness contributions on element 1 - element 1.
     * \param[inout] stiffmat12 (out) Stiffness contributions on element 1 - element 2.
     * \param[inout] stiffmat21 (out) Stiffness contributions on element 2 - element 1.
     * \param[inout] stiffmat22 (out) Stiffness contributions on element 2 - element 2.
     *
     * \returns True if pair is in contact.
     */
    bool evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
        Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
        Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
        Core::LinAlg::SerialDenseMatrix* stiffmat22) override;

   protected:
    /** \brief You will have to use the FBI::PairFactory
     *
     */

    BeamToFluidMeshtyingPairGaussPoint();

    //! Shortcut to base class.
    using base_class = BeamToFluidMeshtyingPairBase<Beam, Fluid>;

    //! Scalar type for FAD variables.
    using scalar_type = typename base_class::scalar_type;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
