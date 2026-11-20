// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_function.hpp"

#include "4C_io.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_manager.hpp"
#include "4C_utils_symbolic_expression.hpp"

#include <Sacado.hpp>

#include <string>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using SecondDerivativeType = Sacado::Fad::DFad<Sacado::Fad::DFad<double>>;

  /// converts the values of variables from type double to FAD double and returns the modified
  /// vector of name-value-pairs
  std::vector<std::pair<std::string, Sacado::Fad::DFad<double>>>
  convert_variable_values_to_fad_objects(
      const std::vector<std::pair<std::string, double>>& variables)
  {
    // prepare return vector
    std::vector<std::pair<std::string, Sacado::Fad::DFad<double>>> variables_FAD;

    // number of variables
    auto numvariables = static_cast<int>(variables.size());

    // counter for variable numbering
    int counter = 0;

    // set the values of the variables
    for (const auto& [name, value] : variables)
    {
      // FAD object for 1st order derivatives
      Sacado::Fad::DFad<double> varfad(numvariables, counter, value);

      // create name-value-pairs with values now of type FAD double and add to vector
      variables_FAD.emplace_back(name, varfad);

      // update counter
      counter++;
    }
    return variables_FAD;
  }



  /// sets the values of the variables in second derivative of expression
  void set_values_in_expression_second_deriv(
      const std::vector<std::shared_ptr<Core::Utils::FunctionVariable>>& variables,
      std::span<const double> x, const double t,
      std::map<std::string, Sacado::Fad::DFad<Sacado::Fad::DFad<double>>>& variable_values)
  {
    // define Fad object for evaluation
    using FAD = Sacado::Fad::DFad<Sacado::Fad::DFad<double>>;

    const auto safe_coordinate = [&x](size_t i) -> double
    {
      if (i < x.size())
        return x[i];
      else
        return 0.0;
    };

    // define FAD variables
    // arguments are: x, y, z, and t
    const int number_of_arguments = 4;
    // we consider a function of the type F = F ( x, y, z, t, v1(t), ..., vn(t) )
    const int fad_size = number_of_arguments + static_cast<int>(variables.size());
    FAD xfad(fad_size, 0, safe_coordinate(0));
    FAD yfad(fad_size, 1, safe_coordinate(1));
    FAD zfad(fad_size, 2, safe_coordinate(2));
    FAD tfad(fad_size, 3, t);

    xfad.val() = Sacado::Fad::DFad<double>(fad_size, 0, safe_coordinate(0));
    yfad.val() = Sacado::Fad::DFad<double>(fad_size, 1, safe_coordinate(1));
    zfad.val() = Sacado::Fad::DFad<double>(fad_size, 2, safe_coordinate(2));
    tfad.val() = Sacado::Fad::DFad<double>(fad_size, 3, t);

    std::vector<FAD> fadvectvars(variables.size());
    for (int i = 0; i < static_cast<int>(variables.size()); ++i)
    {
      fadvectvars[i] = FAD(fad_size, number_of_arguments + i, variables[i]->value(t));
      fadvectvars[i].val() =
          Sacado::Fad::DFad<double>(fad_size, number_of_arguments + i, variables[i]->value(t));
    }

    variable_values.emplace("x", xfad);
    variable_values.emplace("y", yfad);
    variable_values.emplace("z", zfad);

    // set temporal variable
    variable_values.emplace("t", tfad);

    // set the values of the variables at time t
    for (unsigned int i = 0; i < variables.size(); ++i)
    {
      variable_values.emplace(variables[i]->name(), fadvectvars[i]);
    }
  }


  /// evaluate an expression and assemble to the result vector
  std::vector<double> evaluate_and_assemble_expression_to_result_vector(
      const std::map<std::string, Sacado::Fad::DFad<double>>& variables,
      const std::size_t component,
      std::vector<std::shared_ptr<Core::Utils::SymbolicExpression<double>>> expr,
      const std::map<std::string, double>& constant_values)
  {
    // number of variables
    auto numvariables = static_cast<int>(variables.size());

    std::map<std::string, Sacado::Fad::DFad<double>> variable_values = variables;
    std::copy(constant_values.begin(), constant_values.end(),
        std::inserter(variable_values, variable_values.end()));

    // evaluate the expression
    Sacado::Fad::DFad<double> fdfad = expr[component]->first_derivative(variable_values);

    // resulting vector
    std::vector<double> res(numvariables);

    // fill the result vector
    for (int i = 0; i < numvariables; i++) res[i] = fdfad.dx(i);

    return res;
  }

  /// modifies the component to zero in case the expression is of size one
  std::size_t find_modified_component(const std::size_t component,
      const std::vector<std::shared_ptr<Core::Utils::SymbolicExpression<double>>>& expr)
  {
    return (expr.size() == 1) ? 0 : component;
  }

  //! throw an error if a constant given in the input file is a primary variables
  template <typename T>
  void assert_valid_input(const std::map<std::string, T>& variable_values,
      const std::vector<std::pair<std::string, Core::IO::InputField<double>>>& parameter_fields)
  {
    const bool all_parameter_fields_valid =
        std::all_of(parameter_fields.begin(), parameter_fields.end(),
            [&](const auto& var_name) { return variable_values.count(var_name.first) == 0; });

    if (!all_parameter_fields_valid)
    {
      const auto join_keys = [](const auto& map)
      {
        return std::accumulate(map.begin(), map.end(), std::string(),
            [](const std::string& acc, const auto& v)
            { return acc.empty() ? v.first : acc + ", " + v.first; });
      };

      FOUR_C_THROW(
          "It is not allowed to set primary variables of your problem as constants in the "
          "VARFUNCTION.\n\n"
          "Variables passed to Evaluate: {} \n"
          "Constants from Input: {}",
          join_keys(variable_values).c_str(), join_keys(parameter_fields).c_str());
    }
  }
}  // namespace



std::shared_ptr<Core::Utils::FunctionOfAnything>
Core::Utils::try_create_symbolic_function_of_anything(
    const std::vector<Core::IO::InputParameterContainer>& parameters)
{
  if (parameters.size() != 1) return nullptr;

  const auto& function_lin_def = parameters.front();

  if (function_lin_def.get_if<std::string>("VARFUNCTION") != nullptr)

  {
    std::string component = function_lin_def.get<std::string>("VARFUNCTION");
    
    std::vector<std::pair<std::string, Core::IO::InputField<double>>> constants; //"Constant" parameters from input

    if (auto* constants_list = function_lin_def.get_if<Core::IO::InputParameterContainer::List>("PARAMETERS"))
    {
      for (const auto& entry : *constants_list)
        {
          std::string name = entry.get<std::string>("NAME");
          auto field = entry.get<Core::IO::InputField<double>>("value");
          constants.emplace_back(name, field);
        }
    }

    return std::make_shared<Core::Utils::SymbolicFunctionOfAnything>(component, constants);
  }
  else
  {
    return nullptr;
  }
}



std::shared_ptr<Core::Utils::FunctionOfSpaceTime>
Core::Utils::try_create_symbolic_function_of_space_time(
    const std::vector<Core::IO::InputParameterContainer>& parameters)
{
  // evaluate the maximum component and the number of variables
  int maxcomp = 0;
  int maxvar = -1;
  bool found_function_of_space_time(false);
  for (const auto& ith_function_lin_def : parameters)
  {
    // We need to use get_if since we call this function for lines that are completely wrong.
    // This will go away when the functions are restructured.
    auto* comp = ith_function_lin_def.get_if<std::optional<int>>("COMPONENT");
    if (comp) maxcomp = comp->value_or(maxcomp);

    maxvar = ith_function_lin_def.get_or<int>("VARIABLE", maxvar);
    if (ith_function_lin_def.get_if<std::string>("SYMBOLIC_FUNCTION_OF_SPACE_TIME") != nullptr)
      found_function_of_space_time = true;
  }

  if (!found_function_of_space_time) return nullptr;

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
    const int compid = functcomp.get<std::optional<int>>("COMPONENT").value_or(0);
    if (compid != n) FOUR_C_THROW("expected COMPONENT {} but got COMPONENT {}", n, compid);


    // read the expression of the n-th component of the i-th function
    functstring[n] = functcomp.get<std::string>("SYMBOLIC_FUNCTION_OF_SPACE_TIME");
  }

  std::map<int, std::vector<std::shared_ptr<FunctionVariable>>> variable_pieces;

  // read each row where the variables of the i-th function are defined
  for (std::size_t j = 1; j <= numrowsvar; ++j)
  {
    // update the current row
    const auto& line = parameters[maxcomp + j];

    // read the number of the variable
    const int varid = line.get<int>("VARIABLE");

    const auto variable = std::invoke(
        [&]() -> std::shared_ptr<Core::Utils::FunctionVariable>
        {
          // read the name of the variable
          std::string varname = line.get<std::string>("NAME");

          // read the type of the variable
          std::string vartype = line.get<std::string>("TYPE");

          // read periodicity data
          Periodicstruct periodicdata{};

          periodicdata.periodic = line.has_group("PERIODIC");
          if (periodicdata.periodic)
          {
            periodicdata.t1 = line.group("PERIODIC").get<double>("T1");
            periodicdata.t2 = line.group("PERIODIC").get<double>("T2");
          }
          else
          {
            periodicdata.t1 = 0;
            periodicdata.t2 = 0;
          }

          // distinguish the type of the variable
          if (vartype == "expression")
          {
            auto description = line.get<std::string>("DESCRIPTION");

            return std::make_shared<ParsedFunctionVariable>(varname, description);
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
            return nullptr;
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
        FOUR_C_THROW("Variable {} has a piece-wise definition with inconsistent names.", id);

      functvarvector.emplace_back(std::make_shared<PiecewiseVariable>(name, pieces));
    }
  }

  return std::make_shared<SymbolicFunctionOfSpaceTime>(functstring, functvarvector);
}

void Core::Utils::FunctionOfSpaceTime::evaluate_vector(
    std::span<const double> x, double t, std::span<double> values) const
{
  for (std::size_t i = 0; i < number_components(); ++i) values[i] = evaluate(x, t, i);
}

std::vector<double> Core::Utils::FunctionOfSpaceTime::evaluate_spatial_derivative(
    std::span<const double> x, double t, std::size_t component) const
{
  FOUR_C_THROW("The evaluation of the derivative is not implemented for this function");
}

std::vector<double> Core::Utils::FunctionOfSpaceTime::evaluate_time_derivative(
    std::span<const double> x, double t, unsigned deg, std::size_t component) const
{
  FOUR_C_THROW("The evaluation of the time derivative is not implemented for this function");
}

// Case with spatially varying material properties (inputfield)
double Core::Utils::FunctionOfSpaceTime::evaluate(
    std::span<const double> x, double t, std::size_t component, int eleGID) const
{
  FOUR_C_THROW("The evaluation with 4 arguments is not implemented for this function");
}

void Core::Utils::FunctionOfSpaceTime::evaluate_vector(
    std::span<const double> x, double t, std::span<double> values, int eleGID) const
{
  for (std::size_t i = 0; i < number_components(); ++i) values[i] = evaluate(x, t, i, eleGID);
}

std::vector<double> Core::Utils::FunctionOfSpaceTime::evaluate_spatial_derivative(
    std::span<const double> x, double t, std::size_t component, int eleGID) const
{
  FOUR_C_THROW("The evaluation of the derivative (with 4 arguments) is not implemented for this function");
}

std::vector<double> Core::Utils::FunctionOfSpaceTime::evaluate_time_derivative(
    std::span<const double> x, double t, unsigned deg, std::size_t component, int eleGID) const
{
  FOUR_C_THROW("The evaluation of the time derivative (with 4 arguments) is not implemented for this function");
}

Core::Utils::SymbolicFunctionOfSpaceTime::SymbolicFunctionOfSpaceTime(
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

double Core::Utils::SymbolicFunctionOfSpaceTime::evaluate(
    std::span<const double> x, const double t, const std::size_t component) const
{
  std::size_t component_mod = find_modified_component(component, expr_);

  if (component_mod >= expr_.size())
    FOUR_C_THROW(
        "There are {} expressions but tried to access component {}", expr_.size(), component);

  // create map for variables
  std::map<std::string, double> variable_values;

  // set spatial variables
  variable_values.emplace("x", x[0]);
  if (x.size() > 1) variable_values.emplace("y", x[1]);
  if (x.size() > 2) variable_values.emplace("z", x[2]);
  if (x.size() > 3)
    FOUR_C_THROW("The function can only be evaluated in 1D, 2D or 3D space, but got {}D", x.size());

  // set temporal variable
  variable_values.emplace("t", t);

  // set the values of the variables at time t
  for (const auto& variable : variables_)
  {
    variable_values.emplace(variable->name(), variable->value(t));
  }

  // evaluate F = F ( x, y, z, t, v1, ..., vn )
  return expr_[component_mod]->value(variable_values);
}

std::vector<double> Core::Utils::SymbolicFunctionOfSpaceTime::evaluate_spatial_derivative(
    std::span<const double> x, const double t, const std::size_t component) const
{
  std::size_t component_mod = find_modified_component(component, expr_);

  if (component_mod >= expr_.size())
    FOUR_C_THROW(
        "There are {} expressions but tried to access component {}", expr_.size(), component);


  // variables
  std::map<std::string, Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> variable_values;

  set_values_in_expression_second_deriv(variables_, x, t, variable_values);

  // The expression evaluates to an FAD object for up to second derivatives
  SecondDerivativeType fdfad = expr_[component_mod]->second_derivative(variable_values);

  // Here we return the first spatial derivatives given by FAD component 0, 1 and 2
  return {fdfad.dx(0).val(), fdfad.dx(1).val(), fdfad.dx(2).val()};
}

std::vector<double> Core::Utils::SymbolicFunctionOfSpaceTime::evaluate_time_derivative(
    std::span<const double> x, const double t, const unsigned deg,
    const std::size_t component) const
{
  // result vector
  std::vector<double> res(deg + 1);

  std::size_t component_mod = find_modified_component(component, expr_);

  // variables
  std::map<std::string, Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> variable_values;

  set_values_in_expression_second_deriv(variables_, x, t, variable_values);

  // FAD object for evaluation of derivatives
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> fdfad;

  const int number_of_arguments = 4;

  // add the value at time t
  res[0] = evaluate(x, t, component);

  // add the 1st time derivative at time t
  if (deg >= 1)
  {
    // evaluation of derivatives

    fdfad = expr_[component_mod]->second_derivative(variable_values);

    // evaluation of dF/dt applying the chain rule:
    // dF/dt = dF*/dt + sum_i(dF/dvi*dvi/dt)
    double fdfad_dt = fdfad.dx(3).val();                           // 1) dF*/dt
    for (int i = 0; i < static_cast<int>(variables_.size()); ++i)  // 2) sum_i{...}
    {
      fdfad_dt += fdfad.dx(number_of_arguments + i).val() * variables_[i]->time_derivative_value(t);
    }

    res[1] = fdfad_dt;
  }

  // add the 2nd time derivative at time t
  if (deg >= 2)
  {
    // evaluation of d^2F/dt^2 applying the chain rule:
    // d^2F/dt^2 = d(dF*/dt)/dt + sum_i{
    //                                [d(dF*/dt)/dvi + d(dF/dvi)/dt + sum_j[d(dF/dvi)/dvj * dvj/dt]]
    //                                * dvi/dt +
    //                                 + dF/dvi * d^2vi/dt^2
    //                                }
    double fdfad_dt2 = fdfad.dx(3).dx(3);                   // 1) add d(dF*/dt)/dt to d^2F/dt^2
    std::vector<double> fdfad_dt2_term(variables_.size());  // prepare sum_i{...}

    for (int i = 0; i < static_cast<int>(variables_.size()); ++i)
    {
      fdfad_dt2_term[i] = 0;

      fdfad_dt2_term[i] += fdfad.dx(3).dx(number_of_arguments + i);  // ... + d(dF*/dt)/dvi
      fdfad_dt2_term[i] += fdfad.dx(number_of_arguments + i).dx(3);  // ... + d(dF/dvi)/dt

      for (int j = 0; j < static_cast<int>(variables_.size()); ++j)  // prepare + sum_j{...}
      {
        fdfad_dt2_term[i] +=
            fdfad.dx(number_of_arguments + i).dx(number_of_arguments + j) *  // d(dF/dvi)/dvj ...
            variables_[j]->time_derivative_value(t);                         // ... * dvj/dt
      }

      fdfad_dt2_term[i] *= variables_[i]->time_derivative_value(t);  // ... * dvi/dt

      fdfad_dt2_term[i] += fdfad.dx(number_of_arguments + i).val() *    /// ... + dF/dvi ...
                           variables_[i]->time_derivative_value(t, 2);  /// ... * d^2vi/dt^2

      fdfad_dt2 += fdfad_dt2_term[i];  // 2) add sum_i{...} to d^2F/dt^2
    }

    res[2] = fdfad_dt2;
  }

  // error for higher derivatives
  if (deg >= 3)
  {
    FOUR_C_THROW("Higher time derivatives than second not supported!");
  }

  // return derivatives
  return res;
}


Core::Utils::SymbolicFunctionOfAnything::SymbolicFunctionOfAnything(
    const std::string& component, std::vector<std::pair<std::string, Core::IO::InputField<double>>> constants)
    : parameter_fields_(std::move(constants))
{
  // build the parser for the function evaluation
  auto symbolicexpression = std::make_shared<Core::Utils::SymbolicExpression<double>>(component);

  // save the parsers
  expr_.push_back(symbolicexpression);
}



double Core::Utils::SymbolicFunctionOfAnything::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const std::size_t component) const
{
  return evaluate(variables, constants, component, 0); // We call for the new evaluate method and pass 0 as element_id
}

// Case with spatially varying material properties (inputfield)   
double Core::Utils::SymbolicFunctionOfAnything::evaluate(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const std::size_t component,
    int element_id) const
{
  // create map for variables
  std::map<std::string, double> variable_values;

  // convert vector of pairs to map variables_values
  std::copy(
      variables.begin(), variables.end(), std::inserter(variable_values, variable_values.begin()));

  // add global constants
  std::copy(
      constants.begin(), constants.end(), std::inserter(variable_values, variable_values.end()));

  // add "constants" from inputfield 
  if (parameter_fields_.size() != 0)
  {
    // check if constants from input are valid
    assert_valid_input<double>(variable_values, parameter_fields_);
    for (const auto& [name, field] : parameter_fields_)
    {
      variable_values[name] = field.at(element_id);
    }
  }
  // evaluate the function and return the result
  return expr_[component]->value(variable_values);
}


std::vector<double> Core::Utils::SymbolicFunctionOfAnything::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const std::size_t component) const
{
    return evaluate_derivative(variables, constants, component, 0); // We call for the new evaluate_derivative method and pass 0 as element_id
}

// Case with spatially varying material properties (inputfield)   
std::vector<double> Core::Utils::SymbolicFunctionOfAnything::evaluate_derivative(
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const std::size_t component,
    int element_id) const
{
  auto variables_FAD = convert_variable_values_to_fad_objects(variables);

  std::map<std::string, Sacado::Fad::DFad<double>> variable_values;

  std::map<std::string, double> constant_values;

  // convert vector of pairs to map variables_values
  std::copy(variables_FAD.begin(), variables_FAD.end(),
      std::inserter(variable_values, variable_values.begin()));

  // add constants
  std::copy(
      constants.begin(), constants.end(), std::inserter(constant_values, constant_values.begin()));


  // add element-wise parameters from inputfield (defined as inputfields in VARFUNCTION)
  if (parameter_fields_.size() != 0)
  {
    // check if parameters from input file are valid
    assert_valid_input<Sacado::Fad::DFad<double>>(variable_values, parameter_fields_);
    
    // add parameters from input file
    for (const auto& [name, field] : parameter_fields_)
    {
      constant_values[name]=field.at(element_id);
    }
  }
  return evaluate_and_assemble_expression_to_result_vector(
      variable_values, component, expr_, constant_values);
}

FOUR_C_NAMESPACE_CLOSE
