// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_MESHTYING_STRATEGY_ARTERY_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_MESHTYING_STRATEGY_ARTERY_HPP

#include "4C_config.hpp"

#include "4C_linear_solver_method_linalg.hpp"
#include "4C_porofluid_pressure_based_algorithm.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace PoroPressureBased
{
  class PorofluidElastScatraArteryCouplingBaseAlgorithm;
}

namespace PoroPressureBased
{
  class MeshtyingArtery
  {
   public:
    //! constructor
    explicit MeshtyingArtery(PorofluidAlgorithm* porofluid_algorithm,
        const Teuchos::ParameterList& problem_params,
        const Teuchos::ParameterList& porofluid_params);


    //! prepare time loop
    void prepare_time_loop() const;

    //! prepare the time step
    void prepare_time_step() const;

    //! update
    void update() const;

    //! output
    void output() const;

    //! Initialize the linear solver
    void initialize_linear_solver(Core::LinAlg::Solver& solver) const;

    //! solve the linear system of equations
    void linear_solve(std::shared_ptr<Core::LinAlg::Solver> solver,
        Core::LinAlg::SolverParams& solver_params) const;

    //! calculate norms for convergence checks
    void calculate_norms(std::vector<double>& residual_pressure_norm,
        std::vector<double>& increment_pressure_norm, std::vector<double>& pressure_norm) const;

    //! create the field test
    void create_result_test() const;

    //! restart
    void read_restart(int step) const;

    //! evaluate meshtying
    void evaluate() const;

    //! extract increments and update mesh tying
    std::shared_ptr<const Core::LinAlg::Vector<double>> extract_and_update_iter(
        std::shared_ptr<const Core::LinAlg::Vector<double>> increment) const;

    //! return arterial network algorithm
    std::shared_ptr<Adapter::ArtNet> artery_algorithm() { return artery_algorithm_; }

    //! access dof row map
    std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const;

    //! right-hand side for the coupled system
    std::shared_ptr<const Core::LinAlg::Vector<double>> artery_porofluid_rhs() const;

    //! access to block system matrix of the artery-porofluid problem
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> artery_porofluid_sysmat() const;

    //! get global (combined) increment of the artery-porofluid problem
    std::shared_ptr<const Core::LinAlg::Vector<double>> combined_increment() const;

    //! check if initial fields on coupled DOFs are equal
    void check_initial_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> vector_homogenized) const;

    //! set the element pairs that are close as found by search algorithm
    void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearby_ele_pairs) const;

    //! set up the strategy
    void setup() const;

    //! apply the mesh movement
    void apply_mesh_movement() const;

    //! return blood vessel volume fraction
    std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() const;

   private:
    //! porofluid algorithm
    PorofluidAlgorithm* porofluid_algorithm_;

    //! global parameter list of the algorithm
    const Teuchos::ParameterList& params_;

    //! parameter list of porofluid algorithm
    const Teuchos::ParameterList& porofluid_params_;

    //! vector norm for residuals
    VectorNorm vector_norm_res_;

    //! vector norm for increments
    VectorNorm vector_norm_inc_;

    //! artery algorithm
    std::shared_ptr<Adapter::ArtNet> artery_algorithm_;

    //! artery discretization
    std::shared_ptr<Core::FE::Discretization> artery_dis_;

    //! the mesh tying object
    std::shared_ptr<PorofluidElastScatraArteryCouplingBaseAlgorithm>
        artery_porofluid_coupling_algorithm_;

    //! block system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> global_sysmat_;

    //! global rhs
    std::shared_ptr<Core::LinAlg::Vector<double>> global_rhs_;

    //! global increment
    std::shared_ptr<Core::LinAlg::Vector<double>> global_increment_;

    //! global solution at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> global_phinp_;
  };

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
