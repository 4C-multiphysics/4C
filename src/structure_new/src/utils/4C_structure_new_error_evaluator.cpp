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

  return parameters;
}

void Solid::ErrorEvaluator::evaluate_error(const Parameters& error_evaluator_parameters,
    Teuchos::ParameterList& discretization_evaluation_parameters,
    Core::FE::Discretization& discretization)
{
  // Set action type
  auto evaluation_data = std::dynamic_pointer_cast<Solid::ModelEvaluator::Data>(
      discretization_evaluation_parameters.get<std::shared_ptr<Core::Elements::ParamsInterface>>(
          "interface"));
  evaluation_data->set_action_type(Core::Elements::struct_calc_analytical_error);

  // Create a vector that will be filled with the sum of the square of the error at each Gauss-Point
  auto error_squared = std::make_shared<Core::LinAlg::SerialDenseVector>();
  error_squared->size(3);

  // Set the function ID
  discretization_evaluation_parameters.set("analytical_function_id",
      error_evaluator_parameters.evaluate_error_analytical_displacement_function_id);

  // Evaluate error on all elements and broadcast (and sum up) results to all processors
  discretization.evaluate_scalars(discretization_evaluation_parameters, *error_squared);

  // Write result to csv file
  const int csv_precision = 16;
  Core::IO::RuntimeCsvWriter csv_writer(Core::Communication::my_mpi_rank(discretization.get_comm()),
      *Global::Problem::instance()->output_control_file(), "error_evaluation_analytical_reference");
  csv_writer.register_data_vector("reference_volume", 1, csv_precision);
  csv_writer.register_data_vector("displacement_integral", 1, csv_precision);
  csv_writer.register_data_vector("displacement_error_l2_norm", 1, csv_precision);

  csv_writer.reset_time_and_time_step(0.0, 0);

  csv_writer.append_data_vector("reference_volume", {(*error_squared)(2)});
  csv_writer.append_data_vector("displacement_integral", {(*error_squared)(1)});
  csv_writer.append_data_vector("displacement_error_l2_norm", {sqrt((*error_squared)(0))});

  csv_writer.write_collected_data_to_file();
}

FOUR_C_NAMESPACE_CLOSE
