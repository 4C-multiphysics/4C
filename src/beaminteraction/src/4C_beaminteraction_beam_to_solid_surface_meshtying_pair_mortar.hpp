// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_MORTAR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_MORTAR_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_mortar_base.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometry_pair_scalar_types.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declaration.
namespace Inpar
{
  namespace BeamToSolid
  {
    enum class BeamToSolidMortarShapefunctions;
  }
}  // namespace Inpar


namespace BeamInteraction
{
  /**
   * \brief Class for Mortar beam to surface surface mesh tying.
   * @tparam beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @tparam surface Type from GeometryPair::ElementDiscretization... representing the surface.
   * @tparam mortar Type from BeamInteraction::ElementDiscretization... representing the mortar
   * shape functions.
   */
  template <typename Beam, typename Surface, typename Mortar>
  class BeamToSolidSurfaceMeshtyingPairMortar
      : public BeamToSolidSurfaceMeshtyingPairMortarBase<
            GeometryPair::line_to_surface_scalar_type<Beam, Surface>, Beam, Surface, Mortar>
  {
   private:
    //! Type to be used for scalar AD variables.
    using scalar_type = GeometryPair::line_to_surface_scalar_type<Beam, Surface>;

    //! Shortcut to the base class.
    using base_class =
        BeamToSolidSurfaceMeshtyingPairMortarBase<scalar_type, Beam, Surface, Mortar>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceMeshtyingPairMortar();


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

   protected:
    /**
     * \brief Evaluate the local mortar matrices for this contact element pair.
     */
    void evaluate_dm(Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double>& local_D,
        Core::LinAlg::Matrix<Mortar::n_dof_, Surface::n_dof_, double>& local_M,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint) const;
  };

  /**
   * \brief Factory function for beam-to-solid mortar pairs.
   * @param surface_shape (in) Type of surface element.
   * @param mortar_shapefunction (in) Type of mortar shape function.
   * @return Pointer to the created pair.
   */
  std::shared_ptr<BeamInteraction::BeamContactPair>
  beam_to_solid_surface_meshtying_pair_mortar_factory(const Core::FE::CellType surface_shape,
      const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shapefunction);
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
