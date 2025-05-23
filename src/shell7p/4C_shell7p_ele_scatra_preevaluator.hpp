// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL7P_ELE_SCATRA_PREEVALUATOR_HPP
#define FOUR_C_SHELL7P_ELE_SCATRA_PREEVALUATOR_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_inpar_scatra.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE


namespace Discret::Elements::Shell
{
  /*!
   * \brief Preevaluate anything that needs to be done before the standard Evaluate
   *
   * For the scatra coupling we need for example the concentrations. Empty function in the
   * base class that may be overloaded in derived elements.
   *
   * @param ele  (in) : Reference to the element
   * @param discretization (in) : Reference to the discretization
   * @param dof_index_array (in) : The location array of the owned dofs
   */
  void pre_evaluate_scatra_by_element(Core::Elements::Element& ele, Teuchos::ParameterList& params,
      Core::FE::Discretization& discretization, Core::Elements::LocationArray& dof_index_array);

  /*!
   * @brief Preevaluate anything that needs to be done before the standard Evaluate of the element
   * @p element with the discretization type known at compile time.
   *
   *
   * @tparam distype : discretization type known at compile time
   *
   * @param ele  (in) : Reference to the element
   * @param discretization (in) : Reference to the discretization
   * @param dof_index_array (in) : The location array of the owned dofs
   */
  template <Core::FE::CellType distype>
  void pre_evaluate_scatra(Core::Elements::Element& ele, Teuchos::ParameterList& params,
      Core::FE::Discretization& discretization, Core::Elements::LocationArray& dof_index_array);

}  // namespace Discret::Elements::Shell

FOUR_C_NAMESPACE_CLOSE

#endif
