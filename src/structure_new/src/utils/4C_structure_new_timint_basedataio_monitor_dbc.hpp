// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_MONITOR_DBC_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_MONITOR_DBC_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Solid::TimeInt
{
  /** \brief Input data container for monitoring reaction forces for structural (time) integration
   *
   * */
  class ParamsMonitorDBC
  {
   public:
    /// constructor
    explicit ParamsMonitorDBC(const Teuchos::ParameterList& IO_monitor_dbc_structure_paramslist);

    /// destructor
    virtual ~ParamsMonitorDBC() = default;

    /// output interval regarding steps: write output every INTERVAL_STEPS steps
    [[nodiscard]] int output_interval_in_steps() const { return output_interval_steps_; }

    /// precision for screen output
    [[nodiscard]] unsigned screen_precision() const { return os_precision_; }

   private:
    /// @name variables controlling output
    /// @{
    /// output interval regarding steps: write output every INTERVAL_STEPS steps
    int output_interval_steps_;

    /// precision for screen output
    unsigned os_precision_;
    /// @}
  };

}  // namespace Solid::TimeInt


FOUR_C_NAMESPACE_CLOSE

#endif
