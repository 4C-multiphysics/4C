// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_basedataio_runtime_vtp_output.hpp"

#include "4C_beam3_discretization_runtime_output_params.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Solid::TimeInt::ParamsRuntimeVtpOutput::init(
    const Teuchos::ParameterList& IO_vtp_structure_paramslist)
{
  // We have to call setup() after init()
  issetup_ = false;

  output_interval_steps_ = IO_vtp_structure_paramslist.get<int>("INTERVAL_STEPS");

  output_every_iteration_ = IO_vtp_structure_paramslist.get<bool>("EVERY_ITERATION");

  if (output_every_iteration_)
    FOUR_C_THROW("Every iteration output not implemented for structure vtp output!");

  output_owner_ = IO_vtp_structure_paramslist.get<bool>("OWNER");

  output_orientationandlength_ = IO_vtp_structure_paramslist.get<bool>("ORIENTATIONANDLENGTH");

  output_numberofbonds_ = IO_vtp_structure_paramslist.get<bool>("NUMBEROFBONDS");

  output_linkingforce_ = IO_vtp_structure_paramslist.get<bool>("LINKINGFORCE");


  isinit_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Solid::TimeInt::ParamsRuntimeVtpOutput::setup()
{
  FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");

  // Nothing to do here at the moment

  issetup_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Solid::TimeInt::ParamsRuntimeVtpOutput::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

FOUR_C_NAMESPACE_CLOSE
