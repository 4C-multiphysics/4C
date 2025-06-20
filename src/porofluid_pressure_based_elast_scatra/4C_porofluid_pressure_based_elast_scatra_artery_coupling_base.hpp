// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_BASE_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_BASE_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace PoroPressureBased
{
  //! base class for coupling between artery network and porofluid-elasticity-scatra algorithm
  class PorofluidElastScatraArteryCouplingBaseAlgorithm
  {
   public:
    //! constructor
    PorofluidElastScatraArteryCouplingBaseAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params);

    //! virtual destructor
    virtual ~PorofluidElastScatraArteryCouplingBaseAlgorithm() = default;

    //! access to the full DOF map
    const std::shared_ptr<const Core::LinAlg::Map>& full_map() const;

    //! Recompute the CouplingDOFs for each CouplingNode if ntp-coupling active
    void recompute_coupled_dofs_for_node_to_point_coupling(
        std::vector<const Core::Conditions::Condition*> coupling_condition,
        unsigned int coupling_node_idx);

    //! get global extractor
    const std::shared_ptr<Core::LinAlg::MultiMapExtractor>& global_extractor() const;

    //! check if initial fields on coupled DOFs are equal
    virtual void check_initial_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector) = 0;

    //! access artery (1D) dof row map
    virtual std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const = 0;

    //! access full dof row map
    virtual std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const = 0;

    //! print the coupling method
    virtual void print_coupling_method() const = 0;

    //! Evaluate the 1D-3D coupling
    virtual void evaluate(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs) = 0;

    //! set up the global system of equations of the coupled problem
    virtual void setup_system(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
        std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_artery,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_homogenized,
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_artery) = 0;

    //! set solution vectors of single fields
    virtual void set_solution_vectors(
        std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> phin_homogenized,
        std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_artery)
    {
    }

    //! set the element pairs that are close as found by search algorithm
    virtual void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearby_ele_pairs) {}

    /*!
     * @brief setup global vector
     *
     * @param[out]  global_vector combined vector containing both artery and continuous field
     * quantities
     * @param[in]   homogenized_vector vector containing quantities from homogenized field
     * @param[in]   artery_vector vector containing quantities from artery field
     */
    virtual void setup_global_vector(std::shared_ptr<Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector) = 0;

    /*!
     * @brief extract single field vectors
     *
     * @param[out]  global_vector combined vector containing both artery and continuous field
     * quantities
     * @param[in]   homogenized_vector vector containing quantities from homogenized field
     * @param[in]   artery_vector vector containing quantities from artery field
     */
    virtual void extract_single_field_vectors(
        std::shared_ptr<const Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& artery_vector) = 0;

    //! init the strategy
    virtual void init() = 0;

    //! setup the strategy
    virtual void setup() = 0;

    //! apply mesh movement (on artery elements)
    virtual void apply_mesh_movement() = 0;

    //! return blood vessel volume fraction inside each 2D/3D element
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() = 0;

   protected:
    //! communicator
    MPI_Comm get_comm() const { return comm_; }

    //! artery (1D) discretization
    std::shared_ptr<Core::FE::Discretization> artery_dis_;

    //! homogenized field (2D, 3D) discretization
    std::shared_ptr<Core::FE::Discretization> homogenized_dis_;

    //! coupled dofs of artery field
    std::vector<int> coupled_dofs_artery_;

    //! coupled dofs of homogenized field
    std::vector<int> coupled_dofs_homogenized_;

    //! number of coupled dofs
    int num_coupled_dofs_;

    //! dof row map (not split)
    std::shared_ptr<Core::LinAlg::Map> fullmap_;

    //! global extractor
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> global_extractor_;

    //! mpi rank of the calling process in the communicator
    const int my_mpi_rank_;

    /*!
     * @brief decide if artery elements are evaluated in reference configuration
     *
     * so far, it is assumed that artery elements always follow the deformation of the underlying
     * porous medium. Hence, we actually have to evaluate them in current configuration. If this
     * flag is set to true, artery elements will not move and are evaluated in reference
     * configuration
     */
    bool evaluate_in_ref_config_;

   private:
    //! mpi communicator (mainly for screen output)
    MPI_Comm comm_;
  };

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
