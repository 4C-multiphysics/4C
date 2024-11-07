// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_METHOD_PARAMETERS_HPP
#define FOUR_C_LINEAR_SOLVER_METHOD_PARAMETERS_HPP

#include "4C_config.hpp"

#include <Epetra_Map.h>
#include <MueLu_UseDefaultTypes.hpp>
#include <Teuchos_ParameterListAcceptor.hpp>
#include <Xpetra_MultiVector.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinearSolver
{
  class Parameters
  {
   public:
    /*!
      \brief Setting parameters related to specific solvers

      This method sets specific solver parameters such as the nullspace vectors,
      coordinates as well as block information and number of degrees of freedom
      into the solver parameter list.
    */
    static void compute_solver_parameters(
        Core::FE::Discretization& dis, Teuchos::ParameterList& solverlist);

    /*!
     * \brief Fix the nullspace to match a new given map
     *
     * The nullspace is looked for in the parameter list. If found, it is assumed that
     * it matches the oldmap. Then it is fixed to match the new map.
     *
     * \param field (in): field name (just used for output)
     * \param oldmap (in): row map of nullspace
     * \param newmap (in): row map of nullspace upon exit
     * \param solveparams (in): parameterlist including nullspace vector
     */
    static void fix_null_space(std::string field, const Epetra_Map& oldmap,
        const Epetra_Map& newmap, Teuchos::ParameterList& solveparams);

    /*!
     * \brief Extract nullspace from parameter list and convert to Xpetra::MultiVector
     *
     * \pre The input parameter list needs to contain these entries:
     *   - "null space: dimension" (type: \c int )
     *   - "nullspace" (type: \c RCP<Core::LinAlg::MultiVector<double>> )
     *
     * @param[in] row_map Xpetra-style map to be used to create the nullspace vector
     * @param[in] list Parameter list, where 4C has stored the nullspace data as
     * Core::LinAlg::MultiVector<double>
     * @return Xpetra-style multi vector with nullspace data
     */
    static Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
    extract_nullspace_from_parameterlist(
        const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>& row_map,
        Teuchos::ParameterList& list);
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
