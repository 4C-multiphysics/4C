// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_MESHTYING_PORO_LAGRANGE_STRATEGY_HPP
#define FOUR_C_CONTACT_MESHTYING_PORO_LAGRANGE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_meshtying_lagrange_strategy.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  class PoroMtLagrangeStrategy : public MtLagrangeStrategy
  {
   public:
    /*!
    \brief Standard Constructor

    */
    PoroMtLagrangeStrategy(const Core::LinAlg::Map* dof_row_map,
        const Core::LinAlg::Map* NodeRowMap, Teuchos::ParameterList params,
        std::vector<std::shared_ptr<Mortar::Interface>> interface, int dim, MPI_Comm comm,
        double alphaf, int maxdof);


    /*!
    /brief initial Poro Meshtying calculations
    */
    void initialize_poro_mt(std::shared_ptr<Core::LinAlg::SparseMatrix>& kteffoffdiag);

    /*!
    /brief modify off diagonal system matrix for structural displacement meshtying
     */
    void evaluate_meshtying_poro_off_diag(
        std::shared_ptr<Core::LinAlg::SparseMatrix>& kteffoffdiag);

    /*!
    \brief Recovery method
    This method recovers the Lagrange multiplier correctly for the fluid-structure coupling
    matrix block. Complete Recovery only correct if the structural part is recovered elsewhere
    additionally. This is not needed for solving the problem in pure elast-poroelast meshtying
    cases. It is needed for postprocessing though.*/

    void recover_coupling_matrix_partof_lmp(Core::LinAlg::Vector<double>& veli);

    std::shared_ptr<Core::LinAlg::SparseMatrix> cs_;  // source matrix block row (needed for LM)

    std::shared_ptr<Core::LinAlg::Map> fvelrow_;  // fluid row map (needed for splitting)

  };  // class POROLagrangeStrategy
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
