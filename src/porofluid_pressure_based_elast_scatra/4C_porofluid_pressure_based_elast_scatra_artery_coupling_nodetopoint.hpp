// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NODETOPOINT_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NODETOPOINT_HPP


#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nonconforming.hpp"

FOUR_C_NAMESPACE_OPEN

namespace PoroPressureBased
{
  //! Node-to-point coupling between artery network and porofluid-elasticity-scatra algorithm
  class PorofluidElastScatraArteryCouplingNodeToPointAlgorithm final
      : public PorofluidElastScatraArteryCouplingNonConformingAlgorithm
  {
   public:
    //! constructor
    PorofluidElastScatraArteryCouplingNodeToPointAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params, const std::string& condition_name);

    //! set up of the global system of equations of the coupled problem
    void setup_system(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_artery,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_homogenized,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_artery) override;

    //! set up the strategy
    void setup() override;

    //! apply mesh movement (on artery elements)
    void apply_mesh_movement() override;

    //! access to the blood vessel volume fraction
    std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() override;

    //! Evaluate the 1D-3D coupling
    void evaluate(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) override;

    /*!
     * @brief set the artery diameter in material to be able to use it on 1D discretization
     * \note not possible for node-to-point formulation since variable diameter not yet possible
     */
    void set_artery_diameter_in_material() override
    {
      FOUR_C_THROW(
          "Function 'set_artery_diam_in_material()' not possible for node-to-point coupling");
    };

    /*!
     * @brief reset the integrated diameter vector to zero
     * \note not possible for node-to-point formulation since variable diameter not yet possible
     */
    void reset_integrated_diameter_to_zero() override
    {
      FOUR_C_THROW(
          "Function 'reset_integrated_diam_to_zero()' not possible for node-to-point coupling");
    };

    /*!
     * @brief evaluate additional linearization of (integrated) element diameter dependent terms
     * (Hagen-Poiseuille)
     * \note not possible for node-to-point formulation since variable diameter not yet possible
     */
    void evaluate_additional_linearization_of_integrated_diameter() override
    {
      FOUR_C_THROW(
          "Function 'evaluate_additional_linearization_of_integrated_diameter()' not possible for "
          "node-to-point coupling");
    };

    /*!
     * @brief get the segment lengths of element @p artery_ele_gid
     * \note segment length is set to zero since we have no segments in node-to-point coupling
     */
    std::vector<double> get_ele_segment_length(const int artery_ele_gid) override { return {0.0}; };

   private:
    //! print out the coupling method
    void print_coupling_method() const override;

    //! pre-evaluate the coupling pairs
    void pre_evaluate_coupling_pairs();

    //! Output coupling pairs
    void output_coupling_pairs() const;
  };
}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif