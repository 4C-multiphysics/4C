// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_error_evaluator.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_dofset_interface.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_RCPDecl.hpp>

FOUR_C_NAMESPACE_OPEN

Solid::ErrorEvaluator::Parameters Solid::ErrorEvaluator::error_evaluator_parameter_factory(
    const Teuchos::ParameterList& parameter_list)
{
  Solid::ErrorEvaluator::Parameters parameters;

  parameters.evaluate_error_analytical =
      parameter_list.get<bool>("EVALUATE_ERROR_ANALYTICAL_REFERENCE");
  parameters.evaluate_error_analytical_displacement_function_id =
      parameter_list.get<int>("ANALYTICAL_DISPLACEMENT_FUNCTION");
  parameters.evaluate_error_analytical_lagrange_multiplier_function_id =
      parameter_list.get<int>("ANALYTICAL_LAGRANGE_MULTIPLIER_FUNCTION");

  return parameters;
}


void Solid::ErrorEvaluator::write_error(
    const Results& results, Core::FE::Discretization& discretization)
{
  const int csv_precision = 16;
  Core::IO::RuntimeCsvWriter csv_writer(Core::Communication::my_mpi_rank(discretization.get_comm()),
      *Global::Problem::instance()->output_control_file(), "error_evaluation_analytical_reference");

  if (results.integrated_volume)
    csv_writer.register_data_vector("reference_volume", 1, csv_precision);
  if (results.integrated_squared_displacements)
    csv_writer.register_data_vector("displacement_integral", 1, csv_precision);
  if (results.integrated_squared_error_displacements)
    csv_writer.register_data_vector("displacement_error_l2_norm", 1, csv_precision);
  if (results.integrated_squared_gradient_error_displacements)
    csv_writer.register_data_vector("displacement_error_h1_semi_norm", 1, csv_precision);
  if (results.integrated_squared_error_lagrange_multipliers)
    csv_writer.register_data_vector("lagrange_multiplier_error_l2_norm", 1, csv_precision);

  csv_writer.reset_time_and_time_step(0.0, 0);

  if (results.integrated_volume)
    csv_writer.append_data_vector("reference_volume", {*results.integrated_volume});
  if (results.integrated_squared_displacements)
    csv_writer.append_data_vector(
        "displacement_integral", {*results.integrated_squared_displacements});
  if (results.integrated_squared_error_displacements)
    csv_writer.append_data_vector(
        "displacement_error_l2_norm", {std::sqrt(*results.integrated_squared_error_displacements)});
  if (results.integrated_squared_gradient_error_displacements)
    csv_writer.append_data_vector("displacement_error_h1_semi_norm",
        {std::sqrt(*results.integrated_squared_gradient_error_displacements)});
  if (results.integrated_squared_error_lagrange_multipliers)
    csv_writer.append_data_vector("lagrange_multiplier_error_l2_norm",
        {std::sqrt(*results.integrated_squared_error_lagrange_multipliers)});

  csv_writer.write_collected_data_to_file();
}

FOUR_C_NAMESPACE_CLOSE
