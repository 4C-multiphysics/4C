// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TRANSFER_HPP
#define FOUR_C_LINALG_TRANSFER_HPP

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_utils_owner_or_view.hpp"

#include <Epetra_Export.h>
#include <Epetra_Import.h>
FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Import
  {
   public:
    Import(const Map& target_map, const Map& source_map);

    const Epetra_Import& get_epetra_import() const;
    const Map& source_map() const;
    const Map& target_map() const;


   private:
    Utils::OwnerOrView<Epetra_Import> import_;
    Map source_map_;
    Map target_map_;
  };
  class Export
  {
   public:
    Export(const Map& source_map, const Map& target_map);

    Export(const Export& exporter);

    const Epetra_Export& get_epetra_export() const;

   private:
    Utils::OwnerOrView<Epetra_Export> export_;
  };
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE
#endif
