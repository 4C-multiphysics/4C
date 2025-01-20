// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_SCATRA_3D_ELE_LIB_HPP
#define FOUR_C_SOLID_SCATRA_3D_ELE_LIB_HPP

#include "4C_config.hpp"

#include "4C_inpar_scatra.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputParameterContainer;
}

namespace Discret::Elements
{
  /*!
   * @brief Read the scatra implementation type from the a container of data to create the element
   *
   * @param container
   * @return Inpar::ScaTra::ImplType
   */
  Inpar::ScaTra::ImplType read_scatra_impl_type(const Core::IO::InputParameterContainer& container);
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif
