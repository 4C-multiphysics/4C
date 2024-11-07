// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_ELEMENTS_PARAMSINTERFACE_HPP
#define FOUR_C_FEM_GENERAL_ELEMENTS_PARAMSINTERFACE_HPP

#include "4C_config.hpp"

#include "4C_legacy_enum_definitions_element_actions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  class FunctionManager;
}

namespace Core::Elements
{

  /*! \brief Parameter interface for the element <--> time integrator data exchange
   *
   *  Pure virtual interface class. This class is supposed to replace the current
   *  tasks of the Teuchos::ParameterList.
   *  Please consider to derive a special interface class, if you need special parameters inside
   *  of your element. Keep the Evaluate call untouched and cast the interface object to the
   *  desired specification when and where you need it.
   *
   *  ToDo Currently we set the interface in the elements via the Teuchos::ParameterList.
   *  Theoretically, the Teuchos::ParameterList can be replaced by the interface itself!
   *
   *  \date 03/2016
   *  \author hiermeier */
  class ParamsInterface
  {
   public:
    //! destructor
    virtual ~ParamsInterface() = default;

    //! @name Access general control parameters
    //! @{
    //! get the desired action type
    virtual enum ActionType get_action_type() const = 0;

    //! get the current total time for the evaluate call
    virtual double get_total_time() const = 0;

    //! get the current time step
    virtual double get_delta_time() const = 0;

    //! get function manager
    virtual const Core::Utils::FunctionManager* get_function_manager() const = 0;
    //! @}
  };  // class ParamsInterface
}  // namespace Core::Elements


FOUR_C_NAMESPACE_CLOSE

#endif
