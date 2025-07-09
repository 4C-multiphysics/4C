// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PAIR_MORTAR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PAIR_MORTAR_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_contact_pair_base.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_linalg_sparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  //! Forward declarations
  class BeamToSolidSurfaceContactParams;

  /**
   * \brief Class for mortar beam to surface surface contact based on a scalar Lagrange multiplier
   * interpolation
   * @tparam scalar_type Type for scalar DOF values.
   * @tparam beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @tparam surface Type from GeometryPair::ElementDiscretization... representing the surface.
   * @tparam mortar Type from GeometryPair::ElementDiscretization... representing the interpolation
   * of the Lagrange multiplier.
   */
  template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
  class BeamToSolidSurfaceContactPairMortar
      : public BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>
  {
   protected:
    //! Shortcut to the base class.
    using base_class = BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>;

    // Type from GeometryPair::ElementDiscretization... representing the interpolation of the
    // Lagrange multiplier variations. For now this is always equal to the primal interpolation.
    using mortar_trial = Mortar;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceContactPairMortar();

    /**
     * \brief Destructor.
     */
    ~BeamToSolidSurfaceContactPairMortar() override {};

    /**
     * \brief This pair has no direct stiffness contribution.
     */
    bool is_assembly_direct() const override { return false; }

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
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     */
    void evaluate_and_assemble(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& global_lambda,
        const Core::LinAlg::Vector<double>& displacement_vector) override;

    /**
     * \brief Add the visualization of this pair to the beam to solid visualization output writer.
     *
     * Overwritten for specific mortar output.
     */
    void get_pair_visualization(
        std::shared_ptr<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
        Teuchos::ParameterList& visualization_params) const override;

   private:
    /**
     * @brief Get the Jacobian for the configuration the Lagrange multipliers are defined in
     */
    ScalarType get_jacobian_for_configuration(const ScalarType& eta,
        const Inpar::BeamToSolid::BeamToSolidSurfaceContactMortarDefinedIn mortar_configuration)
        const;

   private:
    //! Integral of the shape function matrix of the beam (transposed) multiplied with the normal
    //! vector multiplied with the shape function matrix of the Lagrange multipliers
    Core::LinAlg::Matrix<Beam::n_dof_, Mortar::n_dof_, ScalarType>
        beam_shape_times_normal_times_lambda_shape_;
    //! Integral of the shape function matrix of the surface (transposed) multiplied with the normal
    //! vector multiplied with the shape function matrix of the Lagrange multipliers
    Core::LinAlg::Matrix<Surface::n_dof_, Mortar::n_dof_, ScalarType>
        surface_shape_times_normal_times_lambda_shape_;
  };

  /**
   * \brief Factory function for beam-to-solid contact mortar pairs.
   */
  std::shared_ptr<BeamInteraction::BeamContactPair>
  beam_to_solid_surface_contact_pair_mortar_factory(
      const BeamToSolidSurfaceContactParams& beam_to_surface_contact_params,
      const Core::FE::CellType& surface_type, const bool beam_is_hermite);

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
