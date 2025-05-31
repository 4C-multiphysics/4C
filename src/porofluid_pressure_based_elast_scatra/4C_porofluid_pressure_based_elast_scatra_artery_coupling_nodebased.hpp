// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NODEBASED_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_NODEBASED_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FSI
{
  class Monolithic;
}  // namespace FSI


namespace PoroPressureBased
{
  //! Node-based coupling between artery network and porofluid-elasticity-scatra algorithm
  class PorofluidElastScatraArteryCouplingNodeBasedAlgorithm final
      : public PorofluidElastScatraArteryCouplingBaseAlgorithm
  {
   public:
    PorofluidElastScatraArteryCouplingNodeBasedAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& meshtying_params, const std::string& condition_name,
        const std::string& artery_coupled_dof_name,
        const std::string& homogenized_coupled_dof_name);

    //! Evaluate the 1D-3D coupling
    void evaluate(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) override
    {
      // nothing to do here, is done in SetupSystem for this type of coupling
    }

    //! set up the linear system of equations of the coupled problem
    void setup_system(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_cont,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_art,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_cont,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_art) override;

    /*!
     * @brief setup global vector
     *
     * @param[out]  global_vector combined vector containing both artery and homogenized field
     * quantities
     * @param[in]   homogenized_vector vector containing quantities from homogenized field
     * @param[in]   artery_vector vector containing quantities from artery field
     */
    void setup_global_vector(std::shared_ptr<Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector) override;

    /*!
     * @brief extract single field vectors
     *
     * @param[out]  global_vector combined vector containing both artery and homogenized field
     * quantities
     * @param[in]   homogenized_vector vector containing quantities from homogenized field
     * @param[in]   artery_vector vector containing quantities from artery field
     */
    void extract_single_field_vectors(
        std::shared_ptr<const Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& artery_vector) override;

    //! check if initial fields on coupled DOFs are equal
    void check_initial_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector) override;

    //! access artery (1D) dof row map
    std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const override;

    //! access full dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const override;

    //! init the strategy
    void init() override;

    //! set up the strategy
    void setup() override
    {
      // do nothing
    }

    //! apply mesh movement (on artery elements)
    void apply_mesh_movement() override;

    //! access to the blood vessel volume fraction
    std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() override;

    //! print out the coupling method
    void print_coupling_method() const override;

   private:
    //! set-up of global rhs vector of the coupled problem
    void setup_rhs(std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery);

    //! set-up of global matrix of the coupled problem
    void setup_matrix(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
        Core::LinAlg::SparseMatrix& sysmat_artery) const;

    /*!
     * @brief setup map extractor for artery mesh tying
     *
     * full map -> all DOFs
     * maps(0)  -> coupled DOFs
     * maps(1)  -> uncoupled DOFs
     *
     * @param[in]   map_extractor the map extractor to set up
     * @param[in]   dis discretization
     * @param[in]   coupled_dofs vector with DOFs to couple
     */
    void setup_map_extractor(Core::LinAlg::MultiMapExtractor& map_extractor,
        Core::FE::Discretization& dis, const std::vector<int>& coupled_dofs) const;

    /*!
     * @brief check if Dirichlet BC is defined on coupled dofs, which is not possible
     *
     * @param[in]   dis discretization
     * @param[in]   coupled_dof_map map with coupled DOFs
     */
    void check_dbc_on_coupled_dofs(const Core::FE::Discretization& dis,
        const std::shared_ptr<const Core::LinAlg::Map>& coupled_dof_map) const;

    //! name of the condition
    const std::string condition_name_;

    //! dof row map split in (field) blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> block_row_dof_map_;

    //! extractors for continuous and artery field, maps(0) coupled Dofs, maps(1) uncoupled Dofs
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> homogenized_field_extractor_;
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> artery_field_extractor_;

    //! coupling adapter
    std::shared_ptr<Coupling::Adapter::Coupling> coupling_artery_homogenized_;

    //! needed for matrix transforms
    std::shared_ptr<Coupling::Adapter::MatrixRowColTransform> row_col_transform_;
    std::shared_ptr<Coupling::Adapter::MatrixRowTransform> row_transform_;
    std::shared_ptr<Coupling::Adapter::MatrixColTransform> col_transform_;
  };

}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif
