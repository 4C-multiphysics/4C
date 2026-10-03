// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_METHOD_PARAMETERS_HPP
#define FOUR_C_LINEAR_SOLVER_METHOD_PARAMETERS_HPP

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"

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
        const Core::FE::Discretization& dis, Teuchos::ParameterList& solverlist);

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
    static void fix_null_space(const std::string& field, const Core::LinAlg::Map& oldmap,
        const Core::LinAlg::Map& newmap, Teuchos::ParameterList& solveparams);

    /*!
     * \brief Fix the coordinates to match the map of a preconditioner block
     *
     * The coordinates are looked for in the parameter list together with the number of
     * equations per node ("PDE equations"). If found, the nodal map consistent with the given
     * block (dof) map is derived as it is done by MueLu's coordinate handling and the
     * coordinates are rebuilt on this map. This is required whenever the block matrix row map
     * handed to a Teko/MueLu preconditioner is a subset of (or differently owned than) the map
     * the coordinates were originally computed on.
     *
     * \param field (in): field name (just used for output)
     * \param newmap (in): row map of the matrix block the coordinates should match
     * \param solveparams (in): parameter list including coordinates vector and "PDE equations"
     */
    static void fix_coordinates(const std::string& field, const Core::LinAlg::Map& newmap,
        Teuchos::ParameterList& solveparams);

    /*!
     * \brief Extract nullspace from parameter list and convert to Xpetra::MultiVector
     *
     * \pre The input parameter list needs to contain these entries:
     *   - "nullspace" (type: \c std::shared_ptr<Core::LinAlg::MultiVector<double>> )
     *
     * @param[in] row_map Xpetra-style map to be used to create the nullspace vector
     * @param[in] list Parameter list, where 4C has stored the nullspace data as
     * Core::LinAlg::MultiVector<double>
     * @return Xpetra-style multi vector with nullspace data
     */
    static Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
    extract_nullspace_from_parameterlist(
        const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>& row_map,
        const Teuchos::ParameterList& list);
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
