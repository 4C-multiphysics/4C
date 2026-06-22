// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_L2_PROJECTION_HPP
#define FOUR_C_FEM_GENERAL_L2_PROJECTION_HPP

#include "4C_config.hpp"

#include "4C_linalg_multi_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <functional>
#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::FE
{
  /*!
    \brief compute L2 projection of a dof based field onto a node based field in a least
    squares sense.
    WARNING: Make sure to pass down a dofrowmap appropriate for your discretization.

    \return an Core::LinAlg::MultiVector<double> based on the discret's node row map containing
    numvec vectors with the projected state

   */
  std::shared_ptr<Core::LinAlg::MultiVector<double>> compute_nodal_l2_projection(
      Core::FE::Discretization& dis,   ///< underlying discretization
      const std::string& statename,    ///< name of state which will be set
      const int& numvec,               ///< number of entries per node to project
      Teuchos::ParameterList& params,  ///< parameter list that contains the element action
      const Teuchos::ParameterList&
          solverparams,  ///< solver parameters for solving the resulting global system;
      const std::function<const Teuchos::ParameterList&(int)>
          get_solver_params  ///< function that returns the solver parameters for the i-th solver
  );

  std::shared_ptr<Core::LinAlg::MultiVector<double>> evaluate_and_solve_nodal_l2_projection(
      Core::FE::Discretization& dis, const Core::LinAlg::Map& noderowmap,
      const std::string& statename, const int& numvec, Teuchos::ParameterList& params,
      const Teuchos::ParameterList& solverparams,
      const std::function<const Teuchos::ParameterList&(int)> get_solver_params,
      const Core::LinAlg::Map& fullnoderowmap,
      const std::map<int, int>& source_to_target_colnodesmap);

  std::shared_ptr<Core::LinAlg::MultiVector<double>> solve_nodal_l2_projection(
      Core::LinAlg::SparseMatrix& massmatrix, Core::LinAlg::MultiVector<double>& rhs, MPI_Comm comm,
      const int& numvec, const Teuchos::ParameterList& solverparams,
      const std::function<const Teuchos::ParameterList&(int)> get_solver_params,
      const Core::LinAlg::Map& noderowmap, const Core::LinAlg::Map& fullnoderowmap,
      const std::map<int, int>& source_to_target_colnodesmap);

}  // namespace Core::FE


FOUR_C_NAMESPACE_CLOSE

#endif
