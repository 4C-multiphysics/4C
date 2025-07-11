// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_POINT_COUPLING_PAIR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_POINT_COUPLING_PAIR_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_contact_pair.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN


// Forward declarations.
namespace Core::LargeRotations
{
  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors;
}  // namespace Core::LargeRotations


namespace BeamInteraction
{
  /**
   * \brief Class for point-wise beam to beam mesh tying.
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   */
  template <typename Beam>
  class BeamToBeamPointCouplingPair : public BeamContactPair
  {
   protected:
    //! FAD type for rotational coupling. The 6 dependent DOFs are the 3 rotational DOFs of each
    //! beam element.
    using scalar_type_rot = typename Sacado::Fad::SLFad<double, 6>;

    //! FAD type for positional coupling.
    using scalar_type_pos = typename Sacado::Fad::SLFad<double, 2 * Beam::n_dof_>;

   public:
    /**
     * \brief Standard Constructor.
     *
     * @param penalty_parameter_rot (in) Penalty parameter for rotational coupling.
     * @param penalty_parameter_pos (in) Penalty parameter for positional coupling.
     * @param pos_in_parameterspace (in) Coupling positions in the beam parameter spaces.
     */
    BeamToBeamPointCouplingPair(double penalty_parameter_rot, double penalty_parameter_pos,
        std::array<double, 2> pos_in_parameterspace);


    /**
     * \brief Setup the beam coupling pair.
     */
    void setup() override;

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
    inline bool get_contact_flag() const override { return true; }

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
    void get_all_active_contact_forces(std::vector<double>& forces) const override
    {
      FOUR_C_THROW("get_all_active_contact_forces not yet implemented!");
    }

    /**
     * \brief Get all (scalar) gap values of this contact pair. Not yet implemented.
     */
    void get_all_active_contact_gaps(std::vector<double>& gaps) const override
    {
      FOUR_C_THROW("get_all_active_contact_gaps not yet implemented!");
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

   private:
    /**
     * \brief Evaluate the positional coupling terms and directly assemble them into the global
     * force vector and stiffness matrix.
     *
     * @param discret (in) Pointer to the discretization.
     * @param force_vector (in / out) Global force vector.
     * @param stiffness_matrix (in / out) Global stiffness matrix.
     * @param displacement_vector (in) Global displacement vector.
     */
    void evaluate_and_assemble_positional_coupling(const Core::FE::Discretization& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& displacement_vector) const;

    /**
     * \brief Evaluate the rotational coupling terms and directly assemble them into the global
     * force vector and stiffness matrix.
     *
     * @param discret (in) Pointer to the discretization.
     * @param force_vector (in / out) Global force vector.
     * @param stiffness_matrix (in / out) Global stiffness matrix.
     * @param displacement_vector (in) Global displacement vector.
     */
    void evaluate_and_assemble_rotational_coupling(const Core::FE::Discretization& discret,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& displacement_vector) const;

   private:
    //! Number of rotational DOF for the SR beams;
    static constexpr unsigned int n_dof_rot_ = 9;

    //! Number of dimensions for each rotation.
    const unsigned int rot_dim_ = 3;

    //! Penalty parameter for positional coupling.
    double penalty_parameter_pos_;

    //! Penalty parameter for rotational coupling.
    double penalty_parameter_rot_;

    //! Coupling point positions in the element parameter spaces.
    std::array<double, 2> position_in_parameterspace_;
  };  // namespace BeamInteraction
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
