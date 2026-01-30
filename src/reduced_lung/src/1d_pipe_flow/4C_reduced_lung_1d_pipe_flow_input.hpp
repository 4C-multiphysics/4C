// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_INPUT_HPP
#define FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
#include "4C_utils_function_of_time.hpp"

#include <map>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung1dPipeFlow
{
  struct Parameters
  {
    double final_time;
    int n_steps;
    int result_every;
    struct Fluid
    {
      double density_rho;
      double viscosity_mu;
      double K_R;
    } fluid;
    struct Material
    {
      Core::IO::InputField<double> youngs_modulus_E;
      double poisson_ratio_nu;
    } material;
    struct Geometry
    {
      double reference_radius;
      Core::IO::InputField<double> reference_area_A0;
      Core::IO::InputField<double> thickness_th;
    } geometry;
    struct BoundaryConditions
    {
      std::string input;
      std::string output;
      int function_id_inflow;
      double condition_outflow;
      std::optional<double> cycle_period;
      std::optional<double> pulse_width;
      const Core::Utils::FunctionOfTime* bc_fct;
    } boundary_conditions;
  };
  Core::IO::InputSpec valid_parameters();
}  // namespace ReducedLung1dPipeFlow

FOUR_C_NAMESPACE_CLOSE


#endif
