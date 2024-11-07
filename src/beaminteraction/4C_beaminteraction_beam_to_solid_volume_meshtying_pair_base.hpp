// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_BASE_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_BASE_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_pair_base.hpp"
#include "4C_geometry_pair_scalar_types.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declarations.
namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg
namespace GEOMETRYPAIR
{
  template <typename ScalarType, typename Line, typename Volume>
  class GeometryPairLineToVolume;
}  // namespace GEOMETRYPAIR


namespace BEAMINTERACTION
{
  /**
   * \brief Class for beam to solid meshtying.
   * @tparam ScalarType Scalar FAD type to be used in this pair.
   * @tparam Beam Type from GEOMETRYPAIR::ElementDiscretization... representing the beam.
   * @tparam Solid Type from GEOMETRYPAIR::ElementDiscretization... representing the solid.
   */
  template <typename ScalarType, typename Beam, typename Solid>
  class BeamToSolidVolumeMeshtyingPairBase
      : public BeamToSolidPairBase<ScalarType, double, Beam, Solid>
  {
   protected:
    //! Type to be used for scalar AD variables.
    using scalar_type = ScalarType;

    //! Shortcut to the base class.
    using base_class = BeamToSolidPairBase<ScalarType, double, Beam, Solid>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidVolumeMeshtyingPairBase();

    /**
     * \brief Setup the contact pair (derived).
     *
     * This method sets the solid reference positions for this pair. This can not be done in the
     * base class, since the beam-to-surface (which derive from the base class) need a different
     * handling of the solid DOF.
     */
    void setup() override;

    /**
     * \brief Things that need to be done in a separate loop before the actual evaluation loop over
     * all contact pairs.
     */
    void pre_evaluate() override;

    /**
     * \brief Update state of translational nodal DoFs (absolute positions and tangents) of both
     * elements.
     *
     * Update of the solid positions is performed in this class method, the beam positions are set
     * in the parent class method, which is called here.
     *
     * @param beam_centerline_dofvec
     * @param solid_nodal_dofvec
     */
    void reset_state(const std::vector<double>& beam_centerline_dofvec,
        const std::vector<double>& solid_nodal_dofvec) override;

    /**
     * \brief Set the restart displacement in this pair.
     *
     * If coupling interactions should be evaluated w.r.t the restart state, this method will set
     * them in the pair accordingly.
     *
     * @param centerline_restart_vec_ (in) Vector with the centerline displacements at the restart
     * step, for all contained elements (Vector of vector).
     */
    void set_restart_displacement(
        const std::vector<std::vector<double>>& centerline_restart_vec_) override;

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
        std::shared_ptr<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
        Teuchos::ParameterList& visualization_params) const override;

    /**
     * \brief Create the geometry pair for this contact pair.
     * @param element1 Pointer to the first element
     * @param element2 Pointer to the second element
     * @param geometry_evaluation_data_ptr Evaluation data that will be linked to the pair.
     */
    void create_geometry_pair(const Core::Elements::Element* element1,
        const Core::Elements::Element* element2,
        const std::shared_ptr<GEOMETRYPAIR::GeometryEvaluationDataBase>&
            geometry_evaluation_data_ptr) override;

   protected:
    /**
     * \brief Return a cast of the geometry pair to the type for this contact pair.
     * @return RPC with the type of geometry pair for this beam contact pair.
     */
    inline std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToVolume<double, Beam, Solid>>
    cast_geometry_pair() const
    {
      return std::dynamic_pointer_cast<GEOMETRYPAIR::GeometryPairLineToVolume<double, Beam, Solid>>(
          this->geometry_pair_);
    };

    /**
     * \brief This function evaluates the penalty force from a given beam position and a given solid
     * position.
     *
     * This method is mainly used for visualization.
     *
     * @param r_beam (in) Position on the beam.
     * @param r_solid (in) Position on the solid.
     * @param force (out) Force acting on the beam (the negative force acts on the solid).
     */
    virtual void evaluate_penalty_force_double(const Core::LinAlg::Matrix<3, 1, double>& r_beam,
        const Core::LinAlg::Matrix<3, 1, double>& r_solid,
        Core::LinAlg::Matrix<3, 1, double>& force) const;

    /**
     * \brief Get the reference position to be used for the evaluation of the coupling terms.
     *
     * @param beam_coupling_ref (out) shifted reference position of the beam.
     * @param solid_coupling_ref (out) shifted reference position of the solid.
     */
    void get_coupling_reference_position(GEOMETRYPAIR::ElementData<Beam, double>& beam_coupling_ref,
        GEOMETRYPAIR::ElementData<Solid, double>& solid_coupling_ref) const;

   protected:
    //! Flag if the meshtying has been evaluated already.
    bool meshtying_is_evaluated_;

    //! Current nodal positions (and tangents) of the solid.
    GEOMETRYPAIR::ElementData<Solid, ScalarType> ele2pos_;

    //! Reference nodal positions (and tangents) of the solid.
    GEOMETRYPAIR::ElementData<Solid, double> ele2posref_;

    //! Offset of solid DOFs for coupling. This will be used when the state that should be coupled
    //! is not the undeformed reference position, i.e. in restart simulations where the restart
    //! state is coupled. This only makes sense for volume mesh tying, which is why we also define
    //! the beam restart DOFs here.
    Core::LinAlg::Matrix<Beam::n_dof_, 1, double> ele1posref_offset_;
    Core::LinAlg::Matrix<Solid::n_dof_, 1, double> ele2posref_offset_;
  };
}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
