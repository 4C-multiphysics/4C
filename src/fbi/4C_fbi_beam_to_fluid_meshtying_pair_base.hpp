#ifndef FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_BASE_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_BASE_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_base.hpp"

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace GEOMETRYPAIR
{
  template <typename ScalarType, typename Line, typename Volume>
  class GeometryPairLineToVolume;
  class LineTo3DEvaluationData;
  class GeometryEvaluationDataBase;
}  // namespace GEOMETRYPAIR


namespace BEAMINTERACTION
{
  /**
   * \brief Class representing a pair of elements for beam to fluid meshtying
   *
   * \param[in] beam Type from GEOMETRYPAIR::ElementDiscretization representing the beam.
   * \param[in] fluid Type from GEOMETRYPAIR::ElementDiscretization representing the fluid.
   */
  template <typename Beam, typename Fluid>
  class BeamToFluidMeshtyingPairBase
      : public BeamToSolidVolumeMeshtyingPairBase<double, Beam, Fluid>
  {
   protected:
    //! Shortcut to base class.
    using base_class = BeamToSolidVolumeMeshtyingPairBase<double, Beam, Fluid>;

    //! Scalar type for FAD variables.
    using scalar_type = typename base_class::scalar_type;

   public:
    /**
     * \brief Setup the contact pair and set information on the current position of the elements in
     * the pair
     */
    void setup() override;

    /**
     * \brief Things that need to be done in a separate loop before the actual evaluation loop over
     * all contact pairs.
     */
    void pre_evaluate() override;

    /**
     * \brief Update state of translational nodal DoFs (absolute positions, tangents and velocities)
     * of both elements.
     * @param beam_centerline_dofvec current nodal beam positions extracted from the element and
     * nodal velocities computed by the time integrator
     * @param fluid_nodal_dofvec current nodal fluid positions (only for ALE different from the
     * reference nodal values) and nodal velocities
     */
    void reset_state(const std::vector<double>& beam_centerline_dofvec,
        const std::vector<double>& fluid_nodal_dofvec) override;

    /**
     * \brief Print information about this beam contact element pair to screen.
     */
    void print(std::ostream& out) const override;

    /**
     * \brief Print this beam contact element pair to screen.
     */
    void print_summary_one_line_per_active_segment_pair(std::ostream& out) const override;

    /**
     * \brief Add the visualization of this pair to the beam to solid visualization output writer.
     *
     * This base class creates output of (if selected in the input file) the segmentation, the
     * integration points - and if implemented in the derived classes - the forces at the
     * integration points.
     *
     * @param visualization_writer (out) Object that manages all visualization related data for beam
     * to solid pairs.
     * @param visualization_params (in) Parameter list (not used in this class).
     */
    void get_pair_visualization(
        Teuchos::RCP<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
        Teuchos::ParameterList& visualization_params) const override;

    /**
     * \brief Create the geometry pair for this contact pair.
     * @param element1 Pointer to the first element
     * @param element2 Pointer to the second element
     * @param geometry_evaluation_data_ptr Evaluation data that will be linked to the pair.
     */
    void create_geometry_pair(const Core::Elements::Element* element1,
        const Core::Elements::Element* element2,
        const Teuchos::RCP<GEOMETRYPAIR::GeometryEvaluationDataBase>& geometry_evaluation_data_ptr)
        override;

   protected:
    /** \brief You will have to use the FBI::PairFactory
     *
     */
    BeamToFluidMeshtyingPairBase();

    void evaluate_beam_position(
        const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& integration_point,
        Core::LinAlg::Matrix<3, 1, scalar_type>& r_beam, bool reference) const;

    //! Current nodal velocities of the two elements.
    GEOMETRYPAIR::ElementData<Beam, scalar_type> ele1vel_;
    GEOMETRYPAIR::ElementData<Fluid, scalar_type> ele2vel_;

    //! Current nodal positions (and tangents) of the two elements.
    GEOMETRYPAIR::ElementData<Beam, double> ele1poscur_;
    GEOMETRYPAIR::ElementData<Fluid, double> ele2poscur_;
  };
}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
