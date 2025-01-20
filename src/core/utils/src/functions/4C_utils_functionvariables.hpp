// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_FUNCTIONVARIABLES_HPP
#define FOUR_C_UTILS_FUNCTIONVARIABLES_HPP

#include "4C_config.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputParameterContainer;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
namespace Core::Utils
{
  template <class T>
  class SymbolicExpression;

  struct Periodicstruct
  {
    bool periodic;
    double t1;
    double t2;
  };

  /// class that defines the main properties of a time dependent variable
  class FunctionVariable
  {
   public:
    /// Base constructor taking the @p name of the FunctionVariable.
    FunctionVariable(std::string name);

    /// Virtual destructor.
    virtual ~FunctionVariable() = default;

    /// evaluate the function at given time
    virtual double value(const double t) = 0;

    /// evaluate the time derivative at given time
    virtual double time_derivative_value(const double t, const unsigned deg = 1) = 0;

    /// check the inclusion of the considered time
    virtual bool contain_time(const double t) = 0;

    /// return the name of the variable
    std::string name() { return name_; }

   private:
    /// name of the variable
    const std::string name_;
  };


  /// variable based on a parsed expression
  class ParsedFunctionVariable : public FunctionVariable
  {
   public:
    ParsedFunctionVariable(std::string name, const std::string& buf);

    /// evaluate the function at given time
    double value(const double t) override;

    /// evaluate the time derivative at given time
    double time_derivative_value(const double t, const unsigned deg = 1) override;

    /// check the inclusion of the considered time
    bool contain_time(const double t) override;

   private:
    /// parsed function
    std::shared_ptr<Core::Utils::SymbolicExpression<double>> timefunction_;
  };


  /// variable based on a linear interpolation
  class LinearInterpolationVariable : public FunctionVariable
  {
   public:
    LinearInterpolationVariable(std::string name, std::vector<double> times,
        std::vector<double> values, struct Periodicstruct periodicdata);

    /// evaluate the function at given time
    double value(const double t) override;

    /// templated function to evaluate and to derive from using sacado
    template <typename ScalarT>
    ScalarT value(const ScalarT& t);

    /// evaluate the time derivative at given time
    double time_derivative_value(const double t, const unsigned deg = 1) override;

    /// check the inclusion of the considered time
    bool contain_time(const double t) override;

   private:
    /// times for the interpolation
    const std::vector<double> times_;

    /// values for the interpolation
    const std::vector<double> values_;

    /// flag for periodic repetition
    const bool periodic_;

    /// initial time of the periodic repetition
    const double t1_;

    /// final time of the periodic repetition
    const double t2_;
  };


  /// variable based on a set of parsed expressions
  class MultiFunctionVariable : public FunctionVariable
  {
   public:
    MultiFunctionVariable(std::string name, std::vector<double> times,
        std::vector<std::string> description_vec, struct Periodicstruct periodicdata);

    /// evaluate the function at given time
    double value(const double t) override;

    /// evaluate the time derivative at given time
    double time_derivative_value(const double t, const unsigned deg = 1) override;

    /// check the inclusion of the considered time
    bool contain_time(const double t) override;

   private:
    /// times defining each interval
    const std::vector<double> times_;

    /// vector of parsed functions
    std::vector<std::shared_ptr<Core::Utils::SymbolicExpression<double>>> timefunction_;


    /// flag for periodic repetition
    const bool periodic_;

    /// initial time of the periodic repetition
    const double t1_;

    /// final time of the periodic repetition
    const double t2_;
  };


  /// variable based on a Fourier interpolation
  class FourierInterpolationVariable : public FunctionVariable
  {
   public:
    FourierInterpolationVariable(std::string name, std::vector<double> times,
        std::vector<double> values, struct Periodicstruct periodicdata);

    /// evaluate the function at given time
    double value(const double t) override;

    /// templated function to evaluate and to derive from using sacado
    template <typename ScalarT>
    ScalarT value(const ScalarT& t);

    /// evaluate the time derivative at given time
    double time_derivative_value(const double t, const unsigned deg = 1) override;

    /// check the inclusion of the considered time
    bool contain_time(const double t) override;

   private:
    /// times for the interpolation
    const std::vector<double> times_;

    /// values for the interpolation
    const std::vector<double> values_;

    /// flag for periodic repetition
    const bool periodic_;

    /// initial time of the periodic repetition
    const double t1_;

    /// final time of the periodic repetition
    const double t2_;
  };


  /**
   * @brief A FunctionVariable constructed piece-wise from other FunctionVariables.
   *
   * When the function is evaluated with either a call to Value() or TimeDerivativeValue(), the
   * first piece that contains the given time will be used for evaluation, even if multiple pieces
   * would be able to evaluate the time.
   */
  class PiecewiseVariable : public FunctionVariable
  {
   public:
    //! Create a PiecewiseVariable from the given @p pieces.
    PiecewiseVariable(
        const std::string& name, std::vector<std::shared_ptr<FunctionVariable>> pieces);

    double value(double t) override;

    double time_derivative_value(double t, unsigned int deg) override;

    bool contain_time(double t) override;

   private:
    //! Helper function to access the piece that contains time @p time. Returns the first
    //! FunctionVariable that contains the time @p t and does not check if other FunctionVariables
    //! also contain it.
    FunctionVariable& find_piece_for_time(double t);

    //! Store the pieces that make up the variable in different time intervals.
    std::vector<std::shared_ptr<FunctionVariable>> pieces_;
  };

  namespace Internal
  {
    //! Internal helper to figure out the correct time points from input.
    std::vector<double> extract_time_vector(const Core::IO::InputParameterContainer& timevar);
  }  // namespace Internal
}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
