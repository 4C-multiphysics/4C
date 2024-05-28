/*----------------------------------------------------------------------*/
/*! \file

\brief Gauss point to segment mesh tying element for between a 3D beam and a surface element. The
coupling terms are evaluated using FAD.

\level 3
*/


#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_gauss_point_FAD.hpp"

#include "4C_geometry_pair_element_faces.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_geometry_pair_scalar_types.hpp"

#include <Epetra_FEVector.h>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename scalar_type, typename beam, typename surface>
BEAMINTERACTION::BeamToSolidSurfaceMeshtyingPairGaussPointFAD<scalar_type, beam,
    surface>::BeamToSolidSurfaceMeshtyingPairGaussPointFAD()
    : base_class()
{
  // Empty constructor.
}

/**
 *
 */
template <typename scalar_type, typename beam, typename surface>
void BEAMINTERACTION::BeamToSolidSurfaceMeshtyingPairGaussPointFAD<scalar_type, beam,
    surface>::EvaluateAndAssemble(const Teuchos::RCP<const DRT::Discretization>& discret,
    const Teuchos::RCP<Epetra_FEVector>& force_vector,
    const Teuchos::RCP<CORE::LINALG::SparseMatrix>& stiffness_matrix,
    const Teuchos::RCP<const Epetra_Vector>& displacement_vector)
{
  // Call Evaluate on the geometry Pair. Only do this once for mesh tying.
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->Evaluate(this->ele1posref_,
        this->face_element_->get_face_reference_element_data(), this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, no coupling terms will be assembled.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Get the penalty potential.
  scalar_type potential = this->get_penalty_potential();

  // Get the pair GIDs.
  std::vector<int> pair_gid = this->get_pair_gid(*discret);

  // If given, assemble force terms into the global vector.
  if (force_vector != Teuchos::null)
  {
    std::vector<double> force_pair_double(pair_gid.size());
    for (unsigned int j_dof = 0; j_dof < pair_gid.size(); j_dof++)
      force_pair_double[j_dof] = CORE::FADUTILS::CastToDouble(potential.dx(j_dof));
    force_vector->SumIntoGlobalValues(pair_gid.size(), pair_gid.data(), force_pair_double.data());
  }

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != Teuchos::null)
    for (unsigned int i_dof = 0; i_dof < pair_gid.size(); i_dof++)
      for (unsigned int j_dof = 0; j_dof < pair_gid.size(); j_dof++)
        stiffness_matrix->FEAssemble(CORE::FADUTILS::CastToDouble(potential.dx(i_dof).dx(j_dof)),
            pair_gid[i_dof], pair_gid[j_dof]);
}


/**
 * Explicit template initialization of template class.
 */
namespace BEAMINTERACTION
{
  using namespace GEOMETRYPAIR;

  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<line_to_surface_patch_scalar_type,
      t_hermite, t_quad4>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<line_to_surface_patch_scalar_type,
      t_hermite, t_quad8>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<line_to_surface_patch_scalar_type,
      t_hermite, t_quad9>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<line_to_surface_patch_scalar_type,
      t_hermite, t_tri3>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<line_to_surface_patch_scalar_type,
      t_hermite, t_tri6>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hermite, t_quad4>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hermite, t_quad8>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hermite, t_quad9>;
}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE