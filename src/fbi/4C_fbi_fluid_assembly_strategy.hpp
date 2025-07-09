// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_FLUID_ASSEMBLY_STRATEGY_HPP
#define FOUR_C_FBI_FLUID_ASSEMBLY_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_linalg_fevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace BeamInteraction
{
  class BeamContactPair;
}

namespace FBI
{
  namespace Utils
  {
    /**
     * \brief This class assembles the contributions of fluid beam mesh tying pairs into the global
     * matrices in the standard case of a fluid without internal mesh tying.
     *
     * The form of the fluid matrix and in an extension the required assembly method
     * depend on the fluid problem, particularly if mesh tying is used.
     */
    class FBIAssemblyStrategy
    {
     public:
      /**
       * \brief Destructor.
       */
      virtual ~FBIAssemblyStrategy() = default;

      /**
       * \brief Calls the correct assembly method for the used global fluid matrix depending on the
       * fluid problem
       *
       * \param[in, out] cff fluid coupling matrix
       * \param[in] eid element gid
       * \param[in] Aele dense matrix to be assembled
       * \param[in] lmrow vector with row gids
       * \param[in] lmrowowner vector with owner procs of row gids
       * \param[in] lmcol vector with column gids
       */
      virtual void assemble_fluid_matrix(std::shared_ptr<Core::LinAlg::SparseOperator> cff,
          int elegid, const std::vector<int>& lmstride,
          const Core::LinAlg::SerialDenseMatrix& elemat, const std::vector<int>& lmrow,
          const std::vector<int>& lmrowowner, const std::vector<int>& lmcol);

      /**
       * \brief Assembles element coupling contributions into global coupling matrices and force
       * vectors needed for partitioned algorithms
       *
       * \param[in] discretization1 discretization to the first field
       * \param[in] discretization2 discretization to the second field
       * \param[in] elegids vector of length 2 containing the global IDs of the interacting elements
       * \param[in] elevec vector of length 2 containing the discrete element residual vectors of
       * the interacting elements
       * \param[in, out] c22 coupling matrix relating DOFs in the second
       * discretization to each other
       * \param[in, out] c11 coupling matrix relating DOFs in the first
       * discretization to each other
       * \param[in, out] c12 coupling matrix relating DOFs in the
       * second discretization to DOFs in the first discretization
       * \param[in, out] c21 coupling
       * matrix relating DOFs in the first discretization to DOFs in the second discretization
       *
       */
      virtual void assemble(const Core::FE::Discretization& discretization1,
          const Core::FE::Discretization& discretization2, std::vector<int> const& elegid,
          std::vector<Core::LinAlg::SerialDenseVector> const& elevec,
          std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& elemat,
          std::shared_ptr<Core::LinAlg::FEVector<double>>& f1,
          std::shared_ptr<Core::LinAlg::FEVector<double>>& f2,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& c11,
          std::shared_ptr<Core::LinAlg::SparseOperator> c22,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& c12,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& c21);
    };
  }  // namespace Utils
}  // namespace FBI

FOUR_C_NAMESPACE_CLOSE

#endif
