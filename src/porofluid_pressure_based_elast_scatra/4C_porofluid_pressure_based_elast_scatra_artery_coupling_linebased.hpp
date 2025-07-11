// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_LINEBASED_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_LINEBASED_HPP

#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nonconforming.hpp"


FOUR_C_NAMESPACE_OPEN

namespace PoroPressureBased
{
  class PorofluidElastScatraArteryCouplingPairBase;

  //! Line-based coupling between artery network and porofluid-elasticity-scatra algorithm
  class PorofluidElastScatraArteryCouplingLineBasedAlgorithm final
      : public PorofluidElastScatraArteryCouplingNonConformingAlgorithm
  {
   public:
    PorofluidElastScatraArteryCouplingLineBasedAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params, const std::string& condition_name);

    //! set up the global system of equations of the coupled problem
    void setup_system(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_artery,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_homogenized,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_artery) override;

    //! set up the algorithm
    void setup() override;

    //! apply mesh movement (on artery elements)
    void apply_mesh_movement() override;

    //! access to the blood vessel volume fraction
    std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() override;

   private:
    //! pre-evaluate the coupling pairs and delete duplicates
    void pre_evaluate_coupling_pairs();

    //! fill the length not changed by deformation and initialize the current length
    void fill_unaffected_artery_length();

    //! fill the integrated diameter not changed by varying blood vessel diameter
    void fill_unaffected_integrated_diameter() const;

    //! calculate the volume fraction occupied by blood vessels
    void calculate_blood_vessel_volume_fraction();

    //! create the GID to segment vector
    void create_gid_to_segment_vector();

    //! fill the GID to segment vector
    void fill_gid_to_segment_vector(
        const std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>>&
            coupled_ele_pairs,
        std::map<int, std::vector<double>>& gid_to_segment_length) const;

    //! set the artery diameter in column-based vector
    void fill_artery_ele_diam_col();

    //! (re-)set the artery diameter in material to be able to use it on 1D discretization
    void set_artery_diameter_in_material() override;

    //! reset the integrated diameter vector to zero
    void reset_integrated_diameter_to_zero() override;

    /*!
     * @brief Depth-first search for the connected components of the 1D artery discretization
     *
     * @param current_node : currently checked node
     * @param checked_nodes : vector that marks already checked nodes
     * @param artery_dis_fully_overlapping : artery-discretization in fully overlapping format
     * @param artery_ele_diameters_fully_overlapping : vector of element diameters in fully
     * overlapping format
     * @param current_connected_component : current connected component
     */
    void depth_first_search(Core::Nodes::Node* current_node,
        std::shared_ptr<Core::LinAlg::Vector<int>> checked_nodes,
        std::shared_ptr<Core::FE::Discretization> artery_dis_fully_overlapping,
        std::shared_ptr<const Core::LinAlg::Vector<double>> artery_ele_diameters_fully_overlapping,
        std::vector<int>& current_connected_component);

    /*!
     * @brief Find free-hanging 1D elements
     *
     * Find the free hanging 1D elements which have to be taken out of simulation during blood
     * vessel collapse. This is realized by getting the connected components of the 1D graph. If no
     * node of a connected component has a DBC or if it is smaller than a user-specified
     * threshold, all its elements are deleted.
     * @param elements_to_be_deleted : vector of free-hanging elements which should be deleted
     */
    void find_free_hanging_1d_elements(std::vector<int>& elements_to_be_deleted);

    //! evaluate additional linearization of (integrated) element diameter dependent terms
    //! (Hagen-Poiseuille)
    void evaluate_additional_linearization_of_integrated_diameter() override;

    /*!
     * @brief Apply additional Dirichlet boundary condition for collapsed 1D elements to avoid
     * singular stiffness matrix.
     *
     * Apply additional zero-pressure or zero-mass-fraction Dirichlet boundary conditions to nodes
     * that are adjacent to collapsed 1D elements (i.e., free-hanging nodes with zero rows in the
     * global stiffness matrix) to avoid singularity of the matrix.
     * \note This procedure is equivalent to deleting collapsed elements from the simulation.
     *
     * @param[in]        dbcmap_artery map of nodes with DBC of 1D discretization
     * @param[in,out]   rhs_artery_with_collapsed right-hand side of artery subpart
     * @returns dbcmap, also containing additional boundary condition for collapsed elements
     */
    std::shared_ptr<Core::LinAlg::Map> get_additional_dbc_for_collapsed_elements(
        const Core::LinAlg::MapExtractor& dbcmap_artery,
        Core::LinAlg::Vector<double>& rhs_artery_with_collapsed) const;

    //! FE-assemble into global force and stiffness
    void assemble(const int& ele1_gid, const int& ele2_gid, const double& integrated_diameter,
        std::vector<Core::LinAlg::SerialDenseVector> const& ele_rhs,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& ele_matrix,
        std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) override;

    //! get the segment length
    std::vector<double> get_ele_segment_length(int artery_ele_gid) override;

    //! check for duplicate segments
    bool is_duplicate_segment(
        const std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>>&
            coupled_ele_pairs,
        const PorofluidElastScatraArteryCouplingPairBase& possible_duplicate);

    //! check if segments are identical
    bool is_identical_segment(
        const std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>>&
            coupled_ele_pairs,
        const int& ele1_gid, const double& etaA, const double& etaB, int& ele_pair_id);

    //! set flag if variable diameter has to be calculated
    void set_flag_variable_diameter() override;

    //! print output of meshtying pairs
    void output_summary() const;

    //! print out the coupling method
    void print_coupling_method() const override;

    //! maximum number of segments per artery element
    int max_num_segments_per_artery_element_;

    //! lengths of artery elements unaffected by deformation
    std::shared_ptr<Core::LinAlg::FEVector<double>> unaffected_artery_segment_lengths_;

    //! lengths of artery elements in current configuration
    std::shared_ptr<Core::LinAlg::FEVector<double>> current_artery_segment_lengths_;

    //! diameters of artery elements integrated over the length of the elements (row format
    //! and FE vector due to non-local assembly)
    std::shared_ptr<Core::LinAlg::FEVector<double>> integrated_artery_diameters_row_;

    //! diameters of artery elements integrated over the length of the elements (col format)
    std::shared_ptr<Core::LinAlg::Vector<double>> integrated_artery_diameters_col_;

    //! diameters of artery elements (col format)
    std::shared_ptr<Core::LinAlg::Vector<double>> artery_elements_diameters_col_;

    //! unaffected diameter integrated over the length of the artery element
    //! (protruding elements for which diameter does not change)
    std::shared_ptr<Core::LinAlg::Vector<double>> unaffected_integrated_artery_diameters_col_;

    //! volume fraction of blood vessels (for output)
    std::shared_ptr<Core::LinAlg::Vector<double>> blood_vessel_volfrac_;

    //! gid to segment: stores [GID; [eta_a eta_b]_1, [eta_a eta_b]_2, ..., [eta_a eta_b]_n]
    //!  of artery elements in column format, i.e. fully overlapping
    std::map<int, std::vector<double>> gid_to_segment_;

    //! gid to segment length: stores [GID; seglength_1, seglength_2, ..., seglength_n]
    //!  of artery elements in column format, i.e., fully overlapping (only used for
    //!  porofluid-problems)
    std::map<int, std::vector<double>> gid_to_segment_length_;
  };
}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
