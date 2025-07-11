// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_fluid_assembly_strategy.hpp"

#include "4C_beam3_base.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/

void FBI::Utils::FBIAssemblyStrategy::assemble(const Core::FE::Discretization& discretization1,
    const Core::FE::Discretization& discretization2, std::vector<int> const& elegid,
    std::vector<Core::LinAlg::SerialDenseVector> const& elevec,
    std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& elemat,
    std::shared_ptr<Core::LinAlg::FEVector<double>>& f1,
    std::shared_ptr<Core::LinAlg::FEVector<double>>& f2,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& c11,
    std::shared_ptr<Core::LinAlg::SparseOperator> c22,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& c12,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& c21)
{
  // the entries of elevecX  belong to the Dofs of the element with GID elegidX
  // the rows    of elematXY belong to the Dofs of the element with GID elegidX
  // the columns of elematXY belong to the Dofs of the element with GID elegidY
  const Core::Elements::Element* ele1 = discretization1.g_element(elegid[0]);
  const Core::Elements::Element* ele2 = discretization2.g_element(elegid[1]);

  // get element location vector and ownerships
  std::vector<int> lmrow1;
  std::vector<int> lmrow2;
  std::vector<int> lmrowowner1;
  std::vector<int> lmrowowner2;
  std::vector<int> lmstride;

  ele1->location_vector(discretization1, lmrow1, lmrowowner1, lmstride);
  ele2->location_vector(discretization2, lmrow2, lmrowowner2, lmstride);

  // assemble both element vectors into global system vector
  if (f1 != nullptr)
  {
    f1->sum_into_global_values(elevec[0].length(), lmrow1.data(), elevec[0].values());
  }
  if (f2 != nullptr)
  {
    f2->sum_into_global_values(elevec[1].length(), lmrow2.data(), elevec[1].values());
  }

  // and finally also assemble stiffness contributions
  if (c11 != nullptr)
  {
    c11->fe_assemble(elemat[0][0], lmrow1, lmrow1);
  }
  if (c12 != nullptr)
  {
    c12->fe_assemble(elemat[0][1], lmrow1, lmrow2);
  }
  if (c21 != nullptr)
  {
    c21->fe_assemble(elemat[1][0], lmrow2, lmrow1);
  }
  if (c22 != nullptr)
  {
    assemble_fluid_matrix(c22, elegid[1], lmstride, elemat[1][1], lmrow2, lmrowowner2, lmrow2);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/

void FBI::Utils::FBIAssemblyStrategy::assemble_fluid_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> cff, int elegid, const std::vector<int>& lmstride,
    const Core::LinAlg::SerialDenseMatrix& elemat, const std::vector<int>& lmrow,
    const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
{
  std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(cff)->fe_assemble(elemat, lmrow, lmcol);
}

FOUR_C_NAMESPACE_CLOSE
