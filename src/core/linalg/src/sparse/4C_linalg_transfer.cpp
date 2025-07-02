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
    : import_(Utils::make_owner<Epetra_Import>(
          targetMap.get_epetra_block_map(), sourceMap.get_epetra_block_map())),
      sourceMap_(sourceMap),
      targetMap_(targetMap)
{
}

//! Getter for raw Epetra_Import reference
const Epetra_Import& Core::LinAlg::Import::Import::get_epetra_import() const { return *import_; }

//! Standard constructor
Core::LinAlg::Export::Export::Export(const Map& target_map, const Map& source_map)
    : export_(Utils::make_owner<Epetra_Export>(
          target_map.get_epetra_block_map(), source_map.get_epetra_block_map()))
{
}

//! Copy constructor
Core::LinAlg::Export::Export(const Export& exporter)
    : export_(Utils::make_owner<Epetra_Export>(exporter.get_epetra_export()))
{
}
const Core::LinAlg::Map& Core::LinAlg::Import::Import::sourcemap() const { return sourceMap_; }

const Core::LinAlg::Map& Core::LinAlg::Import::Import::targetmap() const { return targetMap_; }



//! Getter for raw Epetra_Import reference
const Epetra_Export& Core::LinAlg::Export::Export::get_epetra_export() const { return *export_; }

FOUR_C_NAMESPACE_CLOSE
