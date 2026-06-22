// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_basedataio_runtime_vtk_output.hpp"

#include "4C_beam3_discretization_runtime_output_params.hpp"
#include "4C_global_data.hpp"
#include "4C_structure_new_discretization_runtime_output_params.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Solid::TimeInt::ParamsRuntimeOutput::init(
    const Teuchos::ParameterList& IO_vtk_structure_paramslist)
{
  // We have to call setup() after init()
  issetup_ = false;

  // initialize the parameter values
  output_interval_steps_ = IO_vtk_structure_paramslist.get<int>("INTERVAL_STEPS");
  output_step_offset_ = IO_vtk_structure_paramslist.get<int>("STEP_OFFSET");
  output_every_iteration_ = IO_vtk_structure_paramslist.get<bool>("EVERY_ITERATION");
  output_element_eval_times_ = IO_vtk_structure_paramslist.get<bool>("ELEMENT_EVAL_TIME");

  // check for output of structure discretization which is to be handled by an own writer object
  output_structure_ =
      IO_vtk_structure_paramslist.sublist("STRUCTURE").get<bool>("OUTPUT_STRUCTURE");

  // create and initialize parameter container object for structure specific runtime output
  if (output_structure_)
  {
    params_runtime_output_structure_ =
        std::make_shared<Discret::Elements::StructureRuntimeOutputParams>();

    params_runtime_output_structure_->init(IO_vtk_structure_paramslist.sublist("STRUCTURE"));
    params_runtime_output_structure_->setup();
  }


  // check for special beam output which is to be handled by an own writer object
  output_beams_ = IO_vtk_structure_paramslist.sublist("BEAMS").get<bool>("OUTPUT_BEAMS");

  // create and initialize parameter container object for beam specific runtime output
  if (output_beams_)
  {
    params_runtime_output_beams_ = std::make_shared<Discret::Elements::BeamRuntimeOutputParams>();

    params_runtime_output_beams_->init(IO_vtk_structure_paramslist.sublist("BEAMS"));
    params_runtime_output_beams_->setup();
  }


  isinit_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Solid::TimeInt::ParamsRuntimeOutput::setup()
{
  FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");

  // Nothing to do here at the moment

  issetup_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void Solid::TimeInt::ParamsRuntimeOutput::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

FOUR_C_NAMESPACE_CLOSE
