// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_function_of_time.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_symbolic_expression.hpp"

#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN


Core::Utils::SymbolicFunctionOfTime::SymbolicFunctionOfTime(
    const std::vector<std::string>& expressions,
    std::vector<std::shared_ptr<FunctionVariable>> variables)
    : variables_(std::move(variables))
{
  for (const auto& expression : expressions)
  {
    {
      auto symbolicexpression =
          std::make_shared<Core::Utils::SymbolicExpression<double>>(expression);
      expr_.push_back(symbolicexpression);
    }
  }
}

double Core::Utils::SymbolicFunctionOfTime::evaluate(
    const double time, const std::size_t component) const
{
  std::map<std::string, double> variable_values;

  variable_values.emplace("t", time);

  for (const auto& variable : variables_)
  {
    variable_values.emplace(variable->name(), variable->value(time));
  }

  return expr_[component]->value(variable_values);
}

double Core::Utils::SymbolicFunctionOfTime::evaluate_derivative(
    const double time, const std::size_t component) const
{
  using FirstDerivativeType = SymbolicExpression<double>::FirstDerivativeType;
  std::map<std::string, FirstDerivativeType> variable_values;

  // define FAD variables
  // argument is only time
  const int number_of_arguments = 1;
  // we consider a function of the type F = F ( t, v1(t), ..., vn(t) )
  const int fad_size = number_of_arguments + static_cast<int>(variables_.size());
  FirstDerivativeType tfad(fad_size, 0, time);

  std::vector<FirstDerivativeType> fadvectvars(variables_.size());
  for (int i = 0; i < static_cast<int>(variables_.size()); ++i)
  {
    fadvectvars[i] =
        FirstDerivativeType(fad_size, number_of_arguments + i, variables_[i]->value(time));
    fadvectvars[i].val() = variables_[i]->value(time);
  }

  // set temporal variable
  variable_values.emplace("t", tfad);

  // set the values of the variables at time t
  for (unsigned int i = 0; i < variables_.size(); ++i)
  {
    variable_values.emplace(variables_[i]->name(), fadvectvars[i]);
  }

  auto f_dfad = expr_[component]->first_derivative(variable_values, {});

  double f_dt = f_dfad.dx(0);
  for (int i = 0; i < static_cast<int>(variables_.size()); ++i)
  {
    f_dt += f_dfad.dx(number_of_arguments + i) * variables_[i]->time_derivative_value(time);
  }

  return f_dt;
}

std::shared_ptr<Core::Utils::FunctionOfTime> Core::Utils::try_create_function_of_time(
    const std::vector<Core::IO::InputParameterContainer>& parameters)
{
  // evaluate the maximum component and the number of variables
  int maxcomp = 0;
  bool found_function_of_time(false);
  for (const auto& ith_function_lin_def : parameters)
  {
    maxcomp = ith_function_lin_def.get_or<int>("COMPONENT", 0);
    if (ith_function_lin_def.get_if<std::string>("SYMBOLIC_FUNCTION_OF_TIME") != nullptr)
      found_function_of_time = true;
  }

  if (!found_function_of_time) return nullptr;

  // evaluate the number of rows used for the definition of the variables
  std::size_t numrowsvar = parameters.size() - maxcomp - 1;

  // define a vector of strings
  std::vector<std::string> functstring(maxcomp + 1);

  // read each row where the components of the i-th function are defined
  for (int n = 0; n <= maxcomp; ++n)
  {
    // update the current row
    const auto& functcomp = parameters[n];

    // check the validity of the n-th component
    int compid = functcomp.get_or<int>("COMPONENT", 0);
    if (compid != n) FOUR_C_THROW("expected COMPONENT %d but got COMPONENT %d", n, compid);

    // read the expression of the n-th component of the i-th function
    functstring[n] = functcomp.get<std::string>("SYMBOLIC_FUNCTION_OF_TIME");
  }

  std::map<int, std::vector<std::shared_ptr<FunctionVariable>>> variable_pieces;

  // read each row where the variables of the i-th function are defined
  for (std::size_t j = 1; j <= numrowsvar; ++j)
  {
    // update the current row
    const auto& line = parameters[maxcomp + j];

    // read the number of the variable
    int varid = line.get<int>("VARIABLE");

    const auto variable = std::invoke(
        [&line]() -> std::shared_ptr<Core::Utils::FunctionVariable>
        {
          // read the name of the variable
          std::string varname = line.get<std::string>("NAME");

          // read the type of the variable
          std::string vartype = line.get<std::string>("TYPE");

          // read periodicity data
          Periodicstruct periodicdata{};

          periodicdata.periodic = line.get<bool>("PERIODIC");
          if (periodicdata.periodic)
          {
            periodicdata.t1 = line.get<double>("T1");
            periodicdata.t2 = line.get<double>("T2");
          }
          else
          {
            periodicdata.t1 = 0;
            periodicdata.t2 = 0;
          }

          // distinguish the type of the variable
          if (vartype == "expression")
          {
            auto description_vec = line.get<std::vector<std::string>>("DESCRIPTION");

            if (description_vec.size() != 1)
            {
              FOUR_C_THROW(
                  "Only expect one DESCRIPTION for variable of type 'expression' but %d were "
                  "given.",
                  description_vec.size());
            }

            return std::make_shared<ParsedFunctionVariable>(varname, description_vec.front());
          }
          else if (vartype == "linearinterpolation")
          {
            // read times
            std::vector<double> times = Internal::extract_time_vector(line);

            // read values
            auto values = line.get<std::vector<double>>("VALUES");

            return std::make_shared<LinearInterpolationVariable>(
                varname, times, values, periodicdata);
          }
          else if (vartype == "multifunction")
          {
            // read times
            std::vector<double> times = Internal::extract_time_vector(line);

            // read descriptions (strings separated with spaces)
            auto description_vec = line.get<std::vector<std::string>>("DESCRIPTION");

            // check if the number of times = number of descriptions + 1
            std::size_t numtimes = times.size();
            std::size_t numdescriptions = description_vec.size();
            if (numtimes != numdescriptions + 1)
              FOUR_C_THROW(
                  "the number of TIMES and the number of DESCRIPTIONNs must be consistent");

            return std::make_shared<MultiFunctionVariable>(
                varname, times, description_vec, periodicdata);
          }
          else if (vartype == "fourierinterpolation")
          {
            // read times
            std::vector<double> times = Internal::extract_time_vector(line);

            // read values
            auto values = line.get<std::vector<double>>("VALUES");

            return std::make_shared<FourierInterpolationVariable>(
                varname, times, values, periodicdata);
          }
          else
          {
            FOUR_C_THROW("unknown variable type");
          }
        });

    variable_pieces[varid].emplace_back(variable);
  }

  std::vector<std::shared_ptr<FunctionVariable>> functvarvector;

  for (const auto& [id, pieces] : variable_pieces)
  {
    // exactly one variable piece -> can be added directly
    if (pieces.size() == 1) functvarvector.emplace_back(pieces[0]);
    // multiple pieces make up this variable -> join them in a PiecewiseVariable
    else
    {
      const auto& name = pieces.front()->name();

      const bool names_of_all_pieces_equal = std::all_of(
          pieces.begin(), pieces.end(), [&name](auto& var) { return var->name() == name; });
      if (not names_of_all_pieces_equal)
        FOUR_C_THROW("Variable %d has a piece-wise definition with inconsistent names.", id);

      functvarvector.emplace_back(std::make_shared<PiecewiseVariable>(name, pieces));
    }
  }

  return std::make_shared<SymbolicFunctionOfTime>(functstring, functvarvector);
}

FOUR_C_NAMESPACE_CLOSE
