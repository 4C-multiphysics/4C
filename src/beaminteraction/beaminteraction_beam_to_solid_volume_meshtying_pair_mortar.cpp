/*----------------------------------------------------------------------*/
/*! \file

\brief Meshtying element for meshtying between a 3D beam and a 3D solid element using mortar shape
functions for the traction.

\level 3
*/


#include "beaminteraction_beam_to_solid_volume_meshtying_pair_mortar.H"

#include "beaminteraction_beam_to_solid_vtu_output_writer_base.H"
#include "beaminteraction_beam_to_solid_vtu_output_writer_visualization.H"
#include "beaminteraction_beam_to_solid_volume_meshtying_vtk_output_params.H"
#include "beaminteraction_beam_to_solid_mortar_manager.H"
#include "beaminteraction_beam_to_solid_utils.H"

#include "lib_utils.H"
#include "linalg_utils_densematrix_inverse.H"
#include "linalg_serialdensematrix.H"
#include "linalg_serialdensevector.H"

#include "geometry_pair_element_functions.H"
#include "geometry_pair_line_to_volume.H"

#include <unordered_set>


/**
 *
 */
template <typename beam, typename solid, typename mortar>
BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairMortar<beam, solid,
    mortar>::BeamToSolidVolumeMeshtyingPairMortar()
    : BeamToSolidVolumeMeshtyingPairBase<beam, solid>(), n_mortar_rot_(0)
{
  // Empty constructor.
}

/**
 *
 */
template <typename beam, typename solid, typename mortar>
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairMortar<beam, solid,
    mortar>::EvaluateAndAssembleMortarContributions(const ::DRT::Discretization& discret,
    const BeamToSolidMortarManager* mortar_manager, LINALG::SparseMatrix& global_G_B,
    LINALG::SparseMatrix& global_G_S, LINALG::SparseMatrix& global_FB_L,
    LINALG::SparseMatrix& global_FS_L, Epetra_FEVector& global_constraint,
    Epetra_FEVector& global_kappa, Epetra_FEVector& global_lambda_active,
    const Teuchos::RCP<const Epetra_Vector>& displacement_vector)
{
  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    LINALG::Matrix<beam::n_dof_, 1, double> beam_coupling_ref;
    LINALG::Matrix<solid::n_dof_, 1, double> solid_coupling_ref;
    this->GetCouplingReferencePosition(beam_coupling_ref, solid_coupling_ref);
    this->CastGeometryPair()->Evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Initialize variables for local mortar matrices.
  LINALG::Matrix<mortar::n_dof_, beam::n_dof_, double> local_D(false);
  LINALG::Matrix<mortar::n_dof_, solid::n_dof_, double> local_M(false);
  LINALG::Matrix<mortar::n_dof_, 1, double> local_kappa(false);
  LINALG::Matrix<mortar::n_dof_, 1, double> local_constraint(false);

  // Evaluate the local mortar contributions.
  EvaluateDM(local_D, local_M, local_kappa, local_constraint);

  // Assemble into global matrices.
  AssembleLocalMortarContributions<beam, solid, mortar>(this, discret, mortar_manager, global_G_B,
      global_G_S, global_FB_L, global_FS_L, global_constraint, global_kappa, global_lambda_active,
      local_D, local_M, local_kappa, local_constraint, n_mortar_rot_);
}

/**
 *
 */
template <typename beam, typename solid, typename mortar>
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairMortar<beam, solid,
    mortar>::GetPairVisualization(Teuchos::RCP<BeamToSolidVtuOutputWriterBase> visualization_writer,
    Teuchos::ParameterList& visualization_params) const
{
  // Get visualization of base method.
  BeamToSolidVolumeMeshtyingPairBase<beam, solid>::GetPairVisualization(
      visualization_writer, visualization_params);

  Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization_discret =
      visualization_writer->GetVisualizationWriter("btsvc-mortar");
  Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization_continuous =
      visualization_writer->GetVisualizationWriter("btsvc-mortar-continuous");
  if (visualization_discret.is_null() and visualization_continuous.is_null()) return;

  const Teuchos::RCP<const BeamToSolidVolumeMeshtyingVtkOutputParams>& output_params_ptr =
      visualization_params.get<Teuchos::RCP<const BeamToSolidVolumeMeshtyingVtkOutputParams>>(
          "btsvc-output_params_ptr");
  const bool write_unique_ids = output_params_ptr->GetWriteUniqueIDsFlag();

  if (visualization_discret != Teuchos::null || visualization_continuous != Teuchos::null)
  {
    // Setup variables.
    LINALG::Matrix<mortar::n_dof_, 1, double> q_lambda;
    LINALG::Matrix<3, 1, scalar_type> X;
    LINALG::Matrix<3, 1, scalar_type> r;
    LINALG::Matrix<3, 1, scalar_type> u;
    LINALG::Matrix<3, 1, double> lambda_discret;
    LINALG::Matrix<3, 1, double> xi_mortar_node;

    // Get the mortar manager and the global lambda vector, those objects will be used to get the
    // discrete Lagrange multiplier values for this pair.
    Teuchos::RCP<const BEAMINTERACTION::BeamToSolidMortarManager> mortar_manager =
        visualization_params.get<Teuchos::RCP<const BEAMINTERACTION::BeamToSolidMortarManager>>(
            "mortar_manager");
    Teuchos::RCP<Epetra_Vector> lambda =
        visualization_params.get<Teuchos::RCP<Epetra_Vector>>("lambda");

    // Get the lambda GIDs of this pair.
    std::vector<int> lambda_row;
    GetMortarGID(mortar_manager.get(), this, mortar::n_dof_, n_mortar_rot_, &lambda_row, nullptr);
    std::vector<double> lambda_pair;
    DRT::UTILS::ExtractMyValues(*lambda, lambda_pair, lambda_row);
    for (unsigned int i_dof = 0; i_dof < mortar::n_dof_; i_dof++)
      q_lambda(i_dof) = lambda_pair[i_dof];

    // Add the discrete values of the Lagrange multipliers.
    if (visualization_discret != Teuchos::null)
    {
      // Check if data for this beam was already written.
      Teuchos::RCP<std::unordered_set<int>> beam_tracker =
          visualization_params.get<Teuchos::RCP<std::unordered_set<int>>>("beam_tracker");

      auto it = beam_tracker->find(this->Element1()->Id());
      if (it == beam_tracker->end())
      {
        // Only do something if this beam element did not write any output yet.

        // Add this element Id to the tracker.
        beam_tracker->insert(this->Element1()->Id());

        // Get the visualization vectors.
        std::vector<double>& point_coordinates =
            visualization_discret->GetMutablePointCoordinateVector();
        std::vector<double>& displacement =
            visualization_discret->GetMutablePointDataVector("displacement");
        std::vector<double>& lambda_vis =
            visualization_discret->GetMutablePointDataVector("lambda");

        std::vector<double>* pair_beam_id = nullptr;
        std::vector<double>* pair_solid_id = nullptr;
        if (write_unique_ids)
        {
          pair_beam_id = &(visualization_discret->GetMutablePointDataVector("uid_0_pair_beam_id"));
          pair_solid_id =
              &(visualization_discret->GetMutablePointDataVector("uid_1_pair_solid_id"));
        }

        for (unsigned int i_node = 0; i_node < mortar::n_nodes_; i_node++)
        {
          // Get the local coordinate of this node.
          xi_mortar_node = CORE::DRT::UTILS::getNodeCoordinates(i_node, mortar::discretization_);

          // Get position and displacement of the mortar node.
          GEOMETRYPAIR::EvaluatePosition<beam>(
              xi_mortar_node(0), this->ele1pos_, r, this->Element1());
          GEOMETRYPAIR::EvaluatePosition<beam>(
              xi_mortar_node(0), this->ele1posref_, X, this->Element1());
          u = r;
          u -= X;

          // Get the discrete Lagrangian multiplier.
          GEOMETRYPAIR::EvaluatePosition<mortar>(xi_mortar_node(0), q_lambda, lambda_discret);

          // Add to output data.
          for (unsigned int dim = 0; dim < 3; dim++)
          {
            point_coordinates.push_back(CORE::FADUTILS::CastToDouble(X(dim)));
            displacement.push_back(CORE::FADUTILS::CastToDouble(u(dim)));
            lambda_vis.push_back(CORE::FADUTILS::CastToDouble(lambda_discret(dim)));
          }

          if (write_unique_ids)
          {
            pair_beam_id->push_back(this->Element1()->Id());
            pair_solid_id->push_back(this->Element2()->Id());
          }
        }
      }
    }


    // Add the continuous values for the Lagrange multipliers.
    if (visualization_continuous != Teuchos::null and this->line_to_3D_segments_.size() > 0)
    {
      const unsigned int mortar_segments =
          visualization_params
              .get<Teuchos::RCP<const BeamToSolidVolumeMeshtyingVtkOutputParams>>(
                  "btsvc-output_params_ptr")
              ->GetMortarLambdaContinuousSegments();
      double xi;
      std::vector<double>& point_coordinates =
          visualization_continuous->GetMutablePointCoordinateVector(
              (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<double>& displacement = visualization_continuous->GetMutablePointDataVector(
          "displacement", (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<double>& lambda_vis = visualization_continuous->GetMutablePointDataVector(
          "lambda", (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<uint8_t>& cell_type = visualization_continuous->GetMutableCellTypeVector();
      std::vector<int32_t>& cell_offset = visualization_continuous->GetMutableCellOffsetVector();

      std::vector<double>* pair_point_beam_id = nullptr;
      std::vector<double>* pair_point_solid_id = nullptr;
      std::vector<double>* pair_cell_beam_id = nullptr;
      std::vector<double>* pair_cell_solid_id = nullptr;
      if (write_unique_ids)
      {
        pair_point_beam_id =
            &(visualization_continuous->GetMutablePointDataVector("uid_0_pair_beam_id"));
        pair_point_solid_id =
            &(visualization_continuous->GetMutablePointDataVector("uid_1_pair_solid_id"));
        pair_cell_beam_id =
            &(visualization_continuous->GetMutableCellDataVector("uid_0_pair_beam_id"));
        pair_cell_solid_id =
            &(visualization_continuous->GetMutableCellDataVector("uid_1_pair_solid_id"));
      }

      for (const auto& segment : this->line_to_3D_segments_)
      {
        for (unsigned int i_curve_segment = 0; i_curve_segment <= mortar_segments;
             i_curve_segment++)
        {
          // Get the position, displacement and lambda value at the current point.
          xi = segment.GetEtaA() +
               i_curve_segment * (segment.GetEtaB() - segment.GetEtaA()) / (double)mortar_segments;
          GEOMETRYPAIR::EvaluatePosition<beam>(xi, this->ele1pos_, r, this->Element1());
          GEOMETRYPAIR::EvaluatePosition<beam>(xi, this->ele1posref_, X, this->Element1());
          u = r;
          u -= X;
          GEOMETRYPAIR::EvaluatePosition<mortar>(xi, q_lambda, lambda_discret);

          // Add to output data.
          for (unsigned int dim = 0; dim < 3; dim++)
          {
            point_coordinates.push_back(CORE::FADUTILS::CastToDouble(X(dim)));
            displacement.push_back(CORE::FADUTILS::CastToDouble(u(dim)));
            lambda_vis.push_back(CORE::FADUTILS::CastToDouble(lambda_discret(dim)));
          }
        }

        // Add the cell for this segment (poly line).
        cell_type.push_back(4);
        cell_offset.push_back(point_coordinates.size() / 3);

        if (write_unique_ids)
        {
          pair_cell_beam_id->push_back(this->Element1()->Id());
          pair_cell_solid_id->push_back(this->Element2()->Id());
          for (unsigned int i_curve_segment = 0; i_curve_segment <= mortar_segments;
               i_curve_segment++)
          {
            pair_point_beam_id->push_back(this->Element1()->Id());
            pair_point_solid_id->push_back(this->Element2()->Id());
          }
        }
      }
    }
  }
}

/**
 *
 */
template <typename beam, typename solid, typename mortar>
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairMortar<beam, solid, mortar>::EvaluateDM(
    LINALG::Matrix<mortar::n_dof_, beam::n_dof_, double>& local_D,
    LINALG::Matrix<mortar::n_dof_, solid::n_dof_, double>& local_M,
    LINALG::Matrix<mortar::n_dof_, 1, double>& local_kappa,
    LINALG::Matrix<mortar::n_dof_, 1, double>& local_constraint) const
{
  // Initialize the local mortar matrices.
  local_D.PutScalar(0.0);
  local_M.PutScalar(0.0);
  local_kappa.PutScalar(0.0);
  local_constraint.PutScalar(0.0);

  // Initialize variables for shape function values.
  LINALG::Matrix<1, mortar::n_nodes_ * mortar::n_val_, double> N_mortar(true);
  LINALG::Matrix<1, beam::n_nodes_ * beam::n_val_, double> N_beam(true);
  LINALG::Matrix<1, solid::n_nodes_ * solid::n_val_, double> N_solid(true);

  // Initialize variable for beam position derivative.
  LINALG::Matrix<3, 1, double> dr_beam_ref(true);

  // Initialize scalar variables.Clear
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;

  // Calculate the mortar matrices.
  // Loop over segments.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].GetSegmentLength();

    // Gauss point loop.
    const unsigned int n_gp = this->line_to_3D_segments_[i_segment].GetProjectionPoints().size();
    for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
    {
      // Get the current Gauss point.
      const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].GetProjectionPoints()[i_gp];

      // Get the jacobian in the reference configuration.
      GEOMETRYPAIR::EvaluatePositionDerivative1<beam>(
          projected_gauss_point.GetEta(), this->ele1posref_, dr_beam_ref, this->Element1());

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.Norm2() * beam_segmentation_factor;

      // Get the shape function matrices.
      N_mortar.Clear();
      N_beam.Clear();
      N_solid.Clear();
      mortar::EvaluateShapeFunction(
          N_mortar, projected_gauss_point.GetEta(), std::integral_constant<unsigned int, 1>{});
      beam::EvaluateShapeFunction(N_beam, projected_gauss_point.GetEta(),
          std::integral_constant<unsigned int, 1>{}, this->Element1());
      solid::EvaluateShapeFunction(N_solid, projected_gauss_point.GetXi(),
          std::integral_constant<unsigned int, 3>{}, this->Element2());

      // Fill in the local templated mortar matrix D.
      for (unsigned int i_mortar_node = 0; i_mortar_node < mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < mortar::n_val_; i_mortar_val++)
          for (unsigned int i_beam_node = 0; i_beam_node < beam::n_nodes_; i_beam_node++)
            for (unsigned int i_beam_val = 0; i_beam_val < beam::n_val_; i_beam_val++)
              for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
                local_D(i_mortar_node * mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                    i_beam_node * beam::n_val_ * 3 + i_beam_val * 3 + i_dim) +=
                    N_mortar(i_mortar_node * mortar::n_val_ + i_mortar_val) *
                    N_beam(i_beam_node * beam::n_val_ + i_beam_val) *
                    projected_gauss_point.GetGaussWeight() * segment_jacobian;

      // Fill in the local templated mortar matrix M.
      for (unsigned int i_mortar_node = 0; i_mortar_node < mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < mortar::n_val_; i_mortar_val++)
          for (unsigned int i_solid_node = 0; i_solid_node < solid::n_nodes_; i_solid_node++)
            for (unsigned int i_solid_val = 0; i_solid_val < solid::n_val_; i_solid_val++)
              for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
                local_M(i_mortar_node * mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                    i_solid_node * solid::n_val_ * 3 + i_solid_val * 3 + i_dim) +=
                    N_mortar(i_mortar_node * mortar::n_val_ + i_mortar_val) *
                    N_solid(i_solid_node * solid::n_val_ + i_solid_val) *
                    projected_gauss_point.GetGaussWeight() * segment_jacobian;

      // Fill in the local templated mortar scaling vector kappa.
      for (unsigned int i_mortar_node = 0; i_mortar_node < mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < mortar::n_val_; i_mortar_val++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            local_kappa(i_mortar_node * mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim) +=
                N_mortar(i_mortar_node * mortar::n_val_ + i_mortar_val) *
                projected_gauss_point.GetGaussWeight() * segment_jacobian;
    }
  }

  // Add the local constraint contributions.
  for (unsigned int i_lambda = 0; i_lambda < mortar::n_dof_; i_lambda++)
  {
    for (unsigned int i_beam = 0; i_beam < beam::n_dof_; i_beam++)
      local_constraint(i_lambda) +=
          local_D(i_lambda, i_beam) * CORE::FADUTILS::CastToDouble(this->ele1pos_(i_beam));
    for (unsigned int i_solid = 0; i_solid < solid::n_dof_; i_solid++)
      local_constraint(i_lambda) -=
          local_M(i_lambda, i_solid) * CORE::FADUTILS::CastToDouble(this->ele2pos_(i_solid));
  }
}

/**
 *
 */
template <typename beam, typename solid, typename mortar>
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingPairMortar<beam, solid,
    mortar>::EvaluatePenaltyForceDouble(const LINALG::Matrix<3, 1, double>& r_beam,
    const LINALG::Matrix<3, 1, double>& r_solid, LINALG::Matrix<3, 1, double>& force) const
{
  force.PutScalar(0.0);
}


/**
 * Explicit template initialization of template class.
 */
namespace BEAMINTERACTION
{
  using namespace GEOMETRYPAIR;

  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex8, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex20, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex27, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet4, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet10, t_line2>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_nurbs27, t_line2>;

  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex8, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex20, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex27, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet4, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet10, t_line3>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_nurbs27, t_line3>;

  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex8, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex20, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_hex27, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet4, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_tet10, t_line4>;
  template class BeamToSolidVolumeMeshtyingPairMortar<t_hermite, t_nurbs27, t_line4>;
}  // namespace BEAMINTERACTION
