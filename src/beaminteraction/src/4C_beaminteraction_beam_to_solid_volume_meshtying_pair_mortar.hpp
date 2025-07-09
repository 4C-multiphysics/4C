// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_MORTAR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_MORTAR_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_base.hpp"

// Forward declarations.
namespace
{
  class BeamToSolidVolumeMeshtyingPairMortarTest;
}

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  /**
   * \brief Class for beam to solid meshtying using mortar shape functions for the contact
   * tractions.
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @param solid Type from GeometryPair::ElementDiscretization... representing the solid.
   * @param mortar Type from BeamInteraction::ElementDiscretization... representing the mortar shape
   * functions.
   */
  template <typename Beam, typename Solid, typename Mortar>
  class BeamToSolidVolumeMeshtyingPairMortar
      : public BeamToSolidVolumeMeshtyingPairBase<double, Beam, Solid>
  {
    //! Define the unit test class as friend so it can set up a valid pair state for the test cases.
    friend BeamToSolidVolumeMeshtyingPairMortarTest;

   protected:
    //! Shortcut to the base class.
    using base_class = BeamToSolidVolumeMeshtyingPairBase<double, Beam, Solid>;

    //! Type to be used for scalar AD variables.
    using scalar_type = typename base_class::scalar_type;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidVolumeMeshtyingPairMortar();

    /**
     * \brief Evaluate the global matrices and vectors resulting from mortar coupling. (derived)
     */
    void evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
        Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
        Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
        Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
        Core::LinAlg::FEVector<double>& global_constraint,
        Core::LinAlg::FEVector<double>& global_kappa,
        Core::LinAlg::SparseMatrix& global_kappa_lin_beam,
        Core::LinAlg::SparseMatrix& global_kappa_lin_solid,
        Core::LinAlg::FEVector<double>& global_lambda_active,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;

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

   protected:
    /**
     * \brief Evaluate the local mortar matrices for this contact element pair.
     */
    void evaluate_dm(Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double>& local_D,
        Core::LinAlg::Matrix<Mortar::n_dof_, Solid::n_dof_, double>& local_M,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint) const;

    /**
     * \brief For the mortar pairs it does not make sense to calculate forces at the integration
     * points.
     * @param r_beam (in) Position on the beam.
     * @param r_solid (in) Position on the solid.
     * @param force (out) Return 0 by default.
     */
    void evaluate_penalty_force_double(const Core::LinAlg::Matrix<3, 1, double>& r_beam,
        const Core::LinAlg::Matrix<3, 1, double>& r_solid,
        Core::LinAlg::Matrix<3, 1, double>& force) const override;

   protected:
    //! Number of rotational Lagrange multiplier DOFS.
    unsigned int n_mortar_rot_;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
