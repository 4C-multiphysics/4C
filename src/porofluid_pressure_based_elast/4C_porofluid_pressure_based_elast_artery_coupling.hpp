// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_ARTERY_COUPLING_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_ARTERY_COUPLING_HPP

#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_porofluid_pressure_based_elast_monolithic.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Global
{
  class Problem;
}  // namespace Global

namespace PoroPressureBased
{
  ///! Monolithic solution scheme for porofluid-elasticity problems with artery coupling
  class PorofluidElastArteryCouplingAlgorithm final : public PorofluidElastMonolithicAlgorithm
  {
   public:
    PorofluidElastArteryCouplingAlgorithm(
        Global::Problem& problem, MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    //! extract the field vectors from a given composed vector.
    /*!
     x is the sum of all increments up to this point.
     \param x  (i) composed vector that contains all field vectors
     \param sx (o) structural vector (e.g. displacements)
     \param fx (o) fluid vector (primary variables of fluid field, i.e., pressures or saturations,
     and 1D artery pressure)
     */
    void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx) override;

    //! build norms for convergence check
    void build_convergence_norms() override;

   protected:
    /// setup composed system matrix from field solvers
    void setup_maps() override;

    /// setup composed system matrix from field solvers
    void setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat) override;

    /// setup global rhs
    void setup_rhs() override;

    //! build the combined dirichletbcmap
    void build_combined_dbc_map() override;

    //! Create the linear solver
    void create_linear_solver(const Teuchos::ParameterList& solverparams,
        const Core::LinearSolver::SolverType solvertype) override;

    //! build the block null spaces
    void build_artery_block_null_space(
        std::shared_ptr<Core::LinAlg::Solver>& solver, const int& arteryblocknum) override;

    //! dof row map (not split)
    std::shared_ptr<Core::LinAlg::Map> fullmap_artporo_;

    //! dof row map split in (field) blocks
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockrowdofmap_artporo_;
  };

}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
