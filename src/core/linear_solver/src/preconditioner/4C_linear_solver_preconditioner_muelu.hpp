// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_MUELU_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_MUELU_HPP

#include "4C_config.hpp"

#include "4C_linear_solver_preconditioner_type.hpp"

#include <MueLu_Hierarchy.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <Xpetra_MultiVector.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /*! \brief General layout of the MueLu preconditioner working with sparse matrices
   * in CRS format.
   *
   * For more specialized versions (e.g. block systems) derive from this
   * class as it already owns all core components of a MueLu preconditioner.
   *
   * MueLu User's Guide: L. Berger-Vergiat, C. A. Glusa, J. J. Hu, M. Mayr,
   * A. Prokopenko, C. M. Siefert, R. S. Tuminaro, T. A. Wiesner:
   * MueLu User's Guide, Technical Report, Sandia National Laboratories, SAND2019-0537, 2019
   *
   * Also works as preconditioner for blocked linear systems of equations for contact problems
   * in saddlepoint formulation.
   *
   *  T. A. Wiesner, M. Mayr, A. Popp, M. W. Gee, W. A. Wall: Algebraic multigrid methods for
   * saddle point systems arising from mortar contact formulations, International Journal for
   * Numerical Methods in Engineering, 122(15):3749-3779, 2021, https://doi.org/10.1002/nme.6680
   */
  class MueLuPreconditioner : public PreconditionerTypeBase
  {
   public:
    explicit MueLuPreconditioner(Teuchos::ParameterList& muelulist);

    /*! \brief Create and compute the preconditioner
     *
     * The MueLu preconditioner works for matrices of the type SparseMatrix and BlockSparseMatrix.
     * We check whether the input matrix is of proper type and throw an error if not!
     *
     * This routine either re-creates the entire preconditioner from scratch or
     * it re-uses the existing preconditioner and only updates the fine level matrix
     * for the Krylov solver.
     *
     * @param matrix Sparse operator to be used as input for the preconditioner
     * @param x Solution of the linear system
     * @param b Right-hand side of the linear system
     */
    void setup(Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b) override;

    //! linear operator used for preconditioning
    Teuchos::RCP<const Thyra::LinearOpBase<double>> prec_operator() const override { return p_; }

   private:
    //! system of equations used for preconditioning used by P_ only
    Teuchos::RCP<const Thyra::LinearOpBase<double>> pmatrix_;

   protected:
    //! MueLu parameter list
    Teuchos::ParameterList& muelulist_;

    //! preconditioner
    Teuchos::RCP<const Thyra::LinearOpBase<double>> p_;

    //! MueLu hierarchy
    Teuchos::RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node>> H_;
  };


  template <typename LO, typename GO, typename Node>
  Teuchos::RCP<const Xpetra::Map<LO, GO, Node>> unwrap_xpetra_map(
      Teuchos::RCP<const Xpetra::Map<LO, GO, Node>> map)
  {
    FOUR_C_ASSERT_ALWAYS(map != Teuchos::null, "Cannot unwrap a null Xpetra map.");

    while (true)
    {
      if (const auto blocked_map =
              Teuchos::rcp_dynamic_cast<const Xpetra::BlockedMap<LO, GO, Node>>(map, false);
          blocked_map != Teuchos::null)
      {
        map = blocked_map->getMap();
        continue;
      }

      if (const auto strided_map =
              Teuchos::rcp_dynamic_cast<const Xpetra::StridedMap<LO, GO, Node>>(map, false);
          strided_map != Teuchos::null)
      {
        map = strided_map->getMap();
        continue;
      }

      return map;
    }
  }

  /*!
   * @brief Check whether the chosen coarse solver type is actually available
   *
   * @param[in] params MueLu parameter list to validate the coarse solver settings if set
   */
  void validate_coarse_solver(const Teuchos::ParameterList& params);
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
