// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_CONTEXT_HPP
#define FOUR_C_DEAL_II_CONTEXT_HPP

#include "4C_config.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{

  // forward declaration
  namespace Internal
  {
    template <int dim, int spacedim>
    struct ContextImplementation;
  }

  /**
   * This class holds data which helps other classes and functions in this namespace to understand
   * the link between 4C and deal.II data structures. There is no documented and supported
   * functionality for this class. In fact, you should not try to do anything with the internals of
   * this class.
   */
  template <int dim, int spacedim = dim>
  struct Context
  {
    std::shared_ptr<Internal::ContextImplementation<dim, spacedim>> pimpl_;
  };
}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif
