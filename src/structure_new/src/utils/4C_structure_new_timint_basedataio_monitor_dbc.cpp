// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_basedataio_monitor_dbc.hpp"

#include "4C_structure_new_monitor_dbc.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Solid::TimeInt::ParamsMonitorDBC::ParamsMonitorDBC(
    const Teuchos::ParameterList& IO_monitor_dbc_structure_paramslist)
    : output_interval_steps_(IO_monitor_dbc_structure_paramslist.get<int>("INTERVAL_STEPS")),
      of_precision_(IO_monitor_dbc_structure_paramslist.get<int>("PRECISION_FILE")),
      os_precision_(IO_monitor_dbc_structure_paramslist.get<int>("PRECISION_SCREEN"))
{
  // empty constructor
}

FOUR_C_NAMESPACE_CLOSE
