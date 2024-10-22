// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_DISCRET_EXTRACTOR_HPP
#define FOUR_C_FLUID_DISCRET_EXTRACTOR_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FLD
{
  class FluidDiscretExtractor
  {
   public:
    /*!
   \brief Constructor

   */
    FluidDiscretExtractor(Teuchos::RCP<Core::FE::Discretization> actdis,  //! parent discretization
        const std::string& condition,  //! condition for separation of domain
        bool yescondition);  //! (unused) bool to distinguish between all nodes having the condition
                             //! and all nodes not having it

    /*!
   \brief Destructor

   */
    virtual ~FluidDiscretExtractor() = default;

    //! get child discretization
    Teuchos::RCP<Core::FE::Discretization> get_child_discretization() { return childdiscret_; }
    //! get node to node coupling in case of periodic boundary conditions (column and row version)
    Teuchos::RCP<std::map<int, std::vector<int>>> get_coupled_col_nodes_child_discretization()
    {
      return col_pbcmapmastertoslave_;
    }
    Teuchos::RCP<std::map<int, std::vector<int>>> get_coupled_row_nodes_child_discretization()
    {
      return row_pbcmapmastertoslave_;
    }

   private:
    //! the parent discretization
    Teuchos::RCP<Core::FE::Discretization> parentdiscret_;
    //! the child discretization
    Teuchos::RCP<Core::FE::Discretization> childdiscret_;
    //! periodic boundary condition: node to node coupling (column and row version)
    Teuchos::RCP<std::map<int, std::vector<int>>> col_pbcmapmastertoslave_;
    Teuchos::RCP<std::map<int, std::vector<int>>> row_pbcmapmastertoslave_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
