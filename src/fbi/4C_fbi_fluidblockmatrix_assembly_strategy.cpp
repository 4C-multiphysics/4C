// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_fluidblockmatrix_assembly_strategy.hpp"

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

void FBI::Utils::FBIBlockAssemblyStrategy::assemble_fluid_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator> cff, int elegid, const std::vector<int>& lmstride,
    const Core::LinAlg::SerialDenseMatrix& elemat, const std::vector<int>& lmrow,
    const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
{
  cff->assemble(lmstride, elemat, lmrow, lmrowowner, lmcol);
}

FOUR_C_NAMESPACE_CLOSE
