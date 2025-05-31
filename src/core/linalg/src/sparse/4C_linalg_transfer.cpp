// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later



#include "4C_config.hpp"

#include "4C_linalg_transfer.hpp"

#include "4C_comm_utils.hpp"

FOUR_C_NAMESPACE_OPEN

//! Standard constructor using source and target maps
Core::LinAlg::Import::Import::Import(const Map& targetMap, const Map& sourceMap)
    : import_(
          Utils::make_owner<Epetra_Import>(targetMap.get_epetra_map(), sourceMap.get_epetra_map()))
{
}
//! Getter for raw Epetra_Import reference
const Epetra_Import& Core::LinAlg::Import::Import::get_epetra_import() const { return *import_; }


FOUR_C_NAMESPACE_CLOSE
