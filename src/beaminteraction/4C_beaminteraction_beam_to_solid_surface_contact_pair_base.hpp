/*----------------------------------------------------------------------*/
/*! \file

\brief Base contact element for contact between a 3D beam and a surface element.

\level 3
*/
// End doxygen header.


#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PAIR_BASE_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_SURFACE_CONTACT_PAIR_BASE_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_solid_pair_base.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"

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
  template <typename ScalarType, typename Line, typename Surface>
  class GeometryPairLineToSurface;

  class FaceElement;

  template <typename Surface, typename ScalarType>
  class FaceElementTemplate;
}  // namespace GEOMETRYPAIR
namespace BEAMINTERACTION
{
  class BeamToSolidOutputWriterVisualization;
}  // namespace BEAMINTERACTION


namespace BEAMINTERACTION
{
  /**
   * \brief Base class for beam to surface surface contact.
   * @tparam scalar_type Type for scalar DOF values.
   * @tparam beam Type from GEOMETRYPAIR::ElementDiscretization... representing the beam.
   * @tparam surface Type from GEOMETRYPAIR::ElementDiscretization... representing the surface.
   */
  template <typename ScalarType, typename Beam, typename Surface>
  class BeamToSolidSurfaceContactPairBase
      : public BeamToSolidPairBase<ScalarType, ScalarType, Beam, Surface>
  {
   protected:
    //! Shortcut to the base class.
    using base_class = BeamToSolidPairBase<ScalarType, ScalarType, Beam, Surface>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidSurfaceContactPairBase();


    /**
     * \brief Update state of translational nodal DoFs (absolute positions and tangents) of the beam
     * element. (derived)
     *
     * This function has to be overwritten here, since the size of FAD variables for surface
     * elements is not known at compile time and has to be set depending on the surface patch that
     * the surface element is part of.
     *
     * @param beam_centerline_dofvec
     * @param solid_nodal_dofvec
     */
    void reset_state(const std::vector<double>& beam_centerline_dofvec,
        const std::vector<double>& solid_nodal_dofvec) override;

    /**
     * \brief Things that need to be done in a separate loop before the actual evaluation loop over
     * the contact pairs.
     */
    void pre_evaluate() override;

    /**
     * \brief Add the visualization of this pair to the beam to solid visualization output writer.
     *
     * Create segmentation and integration points output.
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

    /**
     * \brief Link the contact pair with the face element storing information on the averaged nodal
     * normals (derived).
     */
    void set_face_element(Teuchos::RCP<GEOMETRYPAIR::FaceElement>& face_element) override;

   protected:
    /**
     * \brief Return a cast of the geometry pair to the type for this contact pair.
     * @return RPC with the type of geometry pair for this beam contact pair.
     */
    Teuchos::RCP<GEOMETRYPAIR::GeometryPairLineToSurface<ScalarType, Beam, Surface>>
    cast_geometry_pair() const;

    /**
     * @brief Evaluate the contact kinematics at a projection point
     */
    std::tuple<Core::LinAlg::Matrix<3, 1, ScalarType>, Core::LinAlg::Matrix<3, 1, ScalarType>,
        Core::LinAlg::Matrix<3, 1, ScalarType>, ScalarType>
    evaluate_contact_kinematics_at_projection_point(
        const GEOMETRYPAIR::ProjectionPoint1DTo3D<ScalarType>& projection_point,
        const double beam_cross_section_radius) const;

   private:
    /**
     * \brief Add points on the beam element to an output writer.
     * @param visualization_writer (in/out) Output writer the points are appended to.
     * @param points (in) Vector with the projection points.
     * @param visualization_params (in) Parameter list with visualization parameters.
     */
    void add_visualization_integration_points(
        BeamToSolidOutputWriterVisualization& visualization_writer,
        const std::vector<GEOMETRYPAIR::ProjectionPoint1DTo3D<ScalarType>>& points,
        const Teuchos::ParameterList& visualization_params) const;

   protected:
    //! Pointer to the face element object which manages the positions on the surface, including the
    //! averaged nodal normals.
    Teuchos::RCP<GEOMETRYPAIR::FaceElementTemplate<Surface, ScalarType>> face_element_;
  };
}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
