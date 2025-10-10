// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_EDGE_CONTACT_PAIR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_EDGE_CONTACT_PAIR_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_edge_contact_params.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_utils_exceptions.hpp"

#include <Sacado.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /**
   * \brief Class for beam-to-solid edge contact.
   */
  template <typename Beam, typename Edge>
  class BeamToSolidEdgeContactPair : public BeamContactPair
  {
   public:
    /**
     * \brief Standard Constructor.
     */
    BeamToSolidEdgeContactPair(
        std::shared_ptr<BeamToSolidEdgeContactParameters> beam_to_solid_edge_parameters);

    /**
     * \brief Setup the contact pair.
     */
    void setup() override;

    /**
     * \brief We don't need to store a geometry pair for this pair.
     */
    void create_geometry_pair(const Core::Elements::Element* element1,
        const Core::Elements::Element* element2,
        const std::shared_ptr<GeometryPair::GeometryEvaluationDataBase>&
            geometry_evaluation_data_ptr) override {};

    /**
     * \brief Things that need to be done in a separate loop before the actual evaluation loop over
     * all contact pairs. (derived)
     */
    void pre_evaluate() override {};

    /**
     * \brief Evaluate this contact element pair.
     */
    bool evaluate(Core::LinAlg::SerialDenseVector* forcevec1,
        Core::LinAlg::SerialDenseVector* forcevec2, Core::LinAlg::SerialDenseMatrix* stiffmat11,
        Core::LinAlg::SerialDenseMatrix* stiffmat12, Core::LinAlg::SerialDenseMatrix* stiffmat21,
        Core::LinAlg::SerialDenseMatrix* stiffmat22) override
    {
      return false;
    }

    /**
     * \brief Evaluate the pair and directly assemble it into the global force vector and stiffness
     * matrix (derived).
     *
     * @param discret (in) Pointer to the discretization.
     * @param force_vector (in / out) Global force vector.
     * @param stiffness_matrix (in / out) Global stiffness matrix.
     * @param displacement_vector (in) Global displacement vector.
     */
    void evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;

    /**
     * \brief No need to update pair state vectors, as everything is done in the
     * evaluate_and_assemble call.
     */
    void reset_state(const std::vector<double>& beam_centerline_dofvec,
        const std::vector<double>& solid_nodal_dofvec) override {};

    /**
     * \brief This pair is always active.
     */
    inline bool get_contact_flag() const override
    {
      // TODO add a flag here!
      return true;
    }

    /**
     * \brief Get number of active contact point pairs on this element pair. Not yet implemented.
     */
    unsigned int get_num_all_active_contact_point_pairs() const override
    {
      FOUR_C_THROW("get_num_all_active_contact_point_pairs not yet implemented!");
      return 0;
    };

    /**
     * \brief Get coordinates of all active contact points on element1. Not yet implemented.
     */
    void get_all_active_contact_point_coords_element1(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const override
    {
      FOUR_C_THROW("get_all_active_contact_point_coords_element1 not yet implemented!");
    }

    /**
     * \brief Get coordinates of all active contact points on element2. Not yet implemented.
     */
    void get_all_active_contact_point_coords_element2(
        std::vector<Core::LinAlg::Matrix<3, 1, double>>& coords) const override
    {
      FOUR_C_THROW("get_all_active_contact_point_coords_element2 not yet implemented!");
    }

    /**
     * \brief Get all (scalar) contact forces of this contact pair. Not yet implemented.
     */
    void get_all_active_beam_to_beam_visualization_values(std::vector<double>& forces,
        std::vector<double>& gaps, std::vector<double>& angles,
        std::vector<int>& types) const override
    {
      FOUR_C_THROW("get_all_active_contact_forces not yet implemented!");
    }

    /**
     * \brief Get energy of penalty contact. Not yet implemented.
     */
    double get_energy() const override
    {
      FOUR_C_THROW("get_energy not implemented yet!");
      return 0.0;
    }

    /**
     * \brief Print information about this beam contact element pair to screen.
     */
    void print(std::ostream& out) const override;

    /**
     * \brief Print this beam contact element pair to screen.
     */
    void print_summary_one_line_per_active_segment_pair(std::ostream& out) const override;

    /**
     * \brief Returns the type of this beam point coupling pair.
     */
    ContactPairType get_type() const override { return ContactPairType::beam_to_solid_edge; }

   private:
    //! Parameters for this pair.
    std::shared_ptr<BeamToSolidEdgeContactParameters> beam_to_solid_edge_parameters_;
  };  // namespace BeamInteraction
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
