/*----------------------------------------------------------------------*/
/*! \file

\brief Gauss point to segment mesh tying element for between a 3D beam and a surface element.

\level 3
*/
// End doxygen header.


#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_GAUSS_POINT_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_MESHTYING_PAIR_GAUSS_POINT_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_gauss_point_base.hpp"
#include "4C_geometry_pair_scalar_types.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BEAMINTERACTION
{
  /**
   * \brief Class for Gauss-point-to-segment beam to surface surface mesh tying.
   * @tparam beam Type from GEOMETRYPAIR::ElementDiscretization... representing the beam.
   * @tparam surface Type from GEOMETRYPAIR::ElementDiscretization... representing the surface.
   */
  template <typename Beam, typename Surface>
  class BeamToSolidSurfaceMeshtyingPairGaussPoint
      : public BeamToSolidSurfaceMeshtyingPairGaussPointBase<
            GEOMETRYPAIR::line_to_surface_scalar_type<Beam, Surface>, Beam, Surface>
  {
   private:
    //! Type to be used for scalar AD variables.
    using scalar_type = GEOMETRYPAIR::line_to_surface_scalar_type<Beam, Surface>;

    //! Shortcut to the base class.
    using base_class = BeamToSolidSurfaceMeshtyingPairGaussPointBase<scalar_type, Beam, Surface>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceMeshtyingPairGaussPoint();


    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     */
    void evaluate_and_assemble(const Teuchos::RCP<const Core::FE::Discretization>& discret,
        const Teuchos::RCP<Epetra_FEVector>& force_vector,
        const Teuchos::RCP<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Teuchos::RCP<const Core::LinAlg::Vector>& displacement_vector) override;
  };
}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
