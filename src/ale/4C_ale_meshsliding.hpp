// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ALE_MESHSLIDING_HPP
#define FOUR_C_ALE_MESHSLIDING_HPP

#include "4C_config.hpp"

#include "4C_ale_meshtying.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class CouplingNonLinMortar;
}

namespace ALE
{
  class Meshsliding : public Meshtying
  {
   public:
    //! Constructor
    Meshsliding(std::shared_ptr<Core::FE::Discretization> dis,  ///> actual discretisation
        Core::LinAlg::Solver& solver,                           ///> solver
        int msht,                                               ///> meshting parameter list
        int nsd,                                                ///> number space dimensions
        const Utils::MapExtractor* surfacesplitter = nullptr);  ///> surface splitter

    //! Set up mesh sliding framework
    std::shared_ptr<Core::LinAlg::SparseOperator> setup(std::vector<int> coupleddof,
        std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp) override;

   private:
    //! Call the constructor and the setup of the mortar coupling adapter
    void adapter_mortar(std::vector<int> coupleddof) override;

    //! Compare the size of the slave and master dof row map
    void compare_num_dof() override;

    //! Get function for the slave and master dof row map
    void dof_row_maps() override;

    //! Get function for the P matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> get_mortar_matrix_p() override;

    //! Condensation operation for a block matrix
    void condensation_operation_block_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator>&
            sysmat,  ///> sysmat established by the element routine
        std::shared_ptr<Core::LinAlg::Vector<double>>&
            residual,  ///> residual established by the element routine
        std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
        override;  ///> current displacement vector

    //! Get functions for the mortar matrices
    void get_mortar_matrices(std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_mm,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_ms,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_sm,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& Aco_ss,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& N_m,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& N_s);

    //! Split the mortar matrix into its slave and its master part
    void split_mortar_matrix(std::shared_ptr<Core::LinAlg::SparseMatrix>& MortarMatrix,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& MasterMatrix,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& SlaveMatrix,
        std::shared_ptr<const Core::LinAlg::Map>& dofrowmap);

    //! Compute and update the increments of the slave node (do nothing in the mesh sliding case)
    void update_source_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& inc,
        std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp) override {};

    //! Recover method for Lagrange multipliers
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>>& inc) override;

    //! Solve ALE mesh sliding problem
    int solve_meshtying(Core::LinAlg::Solver& solver,
        std::shared_ptr<Core::LinAlg::SparseOperator> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& disi,
        std::shared_ptr<Core::LinAlg::Vector<double>> residual,
        std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp) override;

    //! adapter to nonlinear mortar coupling framework
    std::shared_ptr<Adapter::CouplingNonLinMortar> adaptermeshsliding_;

    std::shared_ptr<Core::LinAlg::Vector<double>>
        lm_;  // current vector of Lagrange multipliers at t_n+1

    std::shared_ptr<Core::LinAlg::SparseMatrix> a_ss_;  // stiffness block A_ss (needed for LM)
    std::shared_ptr<Core::LinAlg::SparseMatrix> a_sm_;  // stiffness block A_sm (needed for LM)
    std::shared_ptr<Core::LinAlg::SparseMatrix> a_sn_;  // stiffness block A_sn (needed for LM)
    std::shared_ptr<Core::LinAlg::SparseMatrix>
        d_inv_;  // inverse of Mortar matrix D (needed for LM)
    std::shared_ptr<Core::LinAlg::Vector<double>>
        rs_;  // slave side effective forces (needed for LM)
  };

}  // namespace ALE

FOUR_C_NAMESPACE_CLOSE

#endif
