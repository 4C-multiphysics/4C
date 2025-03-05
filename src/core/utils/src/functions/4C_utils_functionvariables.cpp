// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_functionvariables.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_utils_symbolic_expression.hpp"

#include <Sacado.hpp>

#include <utility>

FOUR_C_NAMESPACE_OPEN


Core::Utils::FunctionVariable::FunctionVariable(std::string name) : name_(std::move(name)) {}


Core::Utils::ParsedFunctionVariable::ParsedFunctionVariable(
    std::string name, const std::string& buf)
    : FunctionVariable(std::move(name)),
      timefunction_(std::make_shared<Core::Utils::SymbolicExpression<double>>(buf))
{
}


double Core::Utils::ParsedFunctionVariable::value(const double t)
{
  // evaluate the value of the function
  double value = timefunction_->value({{"t", t}});

  return value;
}


double Core::Utils::ParsedFunctionVariable::time_derivative_value(
    const double t, const unsigned deg)
{
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> tfad(1, 0, t);
  tfad.val() = Sacado::Fad::DFad<double>(1, 0, t);
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> vfad;

  vfad = timefunction_->second_derivative({{"t", tfad}}, {});

  if (deg == 0)
  {
    const double v = value(t);
    return v;
  }
  if (deg == 1)
  {
    const double dv_dt = vfad.dx(0).val();
    return dv_dt;
  }
  else if (deg == 2)
  {
    const double d2v_dt2 = vfad.dx(0).dx(0);
    return d2v_dt2;
  }
  else
  {
    FOUR_C_THROW("Higher than second derivative is not implemented!");
    return 0.0;
  }
}


bool Core::Utils::ParsedFunctionVariable::contain_time(const double t) { return true; }


Core::Utils::LinearInterpolationVariable::LinearInterpolationVariable(std::string name,
    std::vector<double> times, std::vector<double> values, struct Periodicstruct periodicdata)
    : FunctionVariable(std::move(name)),
      times_(std::move(times)),
      values_(std::move(values)),
      periodic_(periodicdata.periodic),
      t1_(periodicdata.t1),
      t2_(periodicdata.t2)
{
}


double Core::Utils::LinearInterpolationVariable::value(const double t)
{
  // evaluate an equivalent time for a periodic variable
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  return LinearInterpolationVariable::value<double>(t_equivalent);
}

template <typename ScalarT>
ScalarT Core::Utils::LinearInterpolationVariable::value(const ScalarT& t)
{
  ScalarT value = 0.0;

  // find the right time slice
  double t_temp = times_[0];
  unsigned int index = 0;
  if (t < times_[0] - 1.0e-14)
  {
    FOUR_C_THROW("t_equivalent is smaller than the starting time");
    return 0.0;
  }
  else if (t <= times_[0] + 1.0e-14)
  {
    index = 1;
  }
  else
  {
    while (t_temp <= t - 1.0e-14)
    {
      index++;
      t_temp = times_[index];

      if (index == times_.size())
      {
        FOUR_C_THROW("t_equivalent is bigger than the ending time");
        return 0.0;
      }
    }
  }

  // evaluate the value of the function
  value = values_[index - 1] + (values_[index] - values_[index - 1]) /
                                   (times_[index] - times_[index - 1]) * (t - times_[index - 1]);

  return value;
}


double Core::Utils::LinearInterpolationVariable::time_derivative_value(
    const double t, const unsigned deg)
{
  // evaluate an equivalent time for a periodic variable
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> tfad(1, 0, t_equivalent);
  tfad.val() = Sacado::Fad::DFad<double>(1, 0, t_equivalent);
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> vfad =
      value<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>>(tfad);

  if (deg == 0)
  {
    const double v = value(t);
    return v;
  }
  if (deg == 1)
  {
    const double dv_dt = vfad.dx(0).val();
    return dv_dt;
  }
  else if (deg == 2)
  {
    //    const double d2v_dt2 = vfad.dx(0).dx(0);
    return 0.0;
  }
  else
  {
    FOUR_C_THROW("Higher than second derivative is not implemented!");
    return 0.0;
  }
}


bool Core::Utils::LinearInterpolationVariable::contain_time(const double t)
{
  /// check the inclusion of the considered time
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  if (t_equivalent >= times_[0] - 1.0e-14 && t_equivalent <= times_[times_.size() - 1] + 1.0e-14)
  {
    return true;
  }
  else
  {
    return false;
  }
}


Core::Utils::MultiFunctionVariable::MultiFunctionVariable(std::string name,
    std::vector<double> times, std::vector<std::string> description_vec,
    struct Periodicstruct periodicdata)
    : FunctionVariable(std::move(name)),
      times_(std::move(times)),
      periodic_(periodicdata.periodic),
      t1_(periodicdata.t1),
      t2_(periodicdata.t2)
{
  // create vectors of timefunction and timederivative
  timefunction_.resize(times_.size() - 1);
  for (unsigned int n = 0; n < times_.size() - 1; ++n)
  {
    timefunction_[n] =
        std::make_shared<Core::Utils::SymbolicExpression<double>>(description_vec[n]);
  }
}


double Core::Utils::MultiFunctionVariable::value(const double t)
{
  // evaluate an equivalent time for a periodic variable
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  // find the right time slice
  double t_temp = times_[0];
  unsigned int index = 0;
  if (t_equivalent < times_[0] - 1.0e-14)
  {
    FOUR_C_THROW("t_equivalent is smaller than the starting time");
    return 0.0;
  }
  else
  {
    while (t_temp < t_equivalent - 1.0e-14)
    {
      index++;
      t_temp = times_[index];
      if (index == times_.size())
      {
        FOUR_C_THROW("t_equivalent is bigger than the ending time");
        return 0.0;
      }
    }
  }

  // evaluate the value of the function considering the different possibilities
  double value;
  if (index == 0)
  {
    value = timefunction_[0]->value({{"t", t_equivalent}});
  }
  else
  {
    value = timefunction_[index - 1]->value({{"t", t_equivalent}});
  }

  return value;
}


double Core::Utils::MultiFunctionVariable::time_derivative_value(const double t, const unsigned deg)
{
  // evaluate an equivalent time for a periodic variable
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  // find the right time slice
  double t_temp = times_[0];
  unsigned int index = 0;
  if (t_equivalent < times_[0])
  {
    FOUR_C_THROW("t_equivalent is smaller than the starting time");
    return 0.0;
  }
  else
  {
    while (t_temp < t_equivalent - 1.0e-14)
    {
      index++;
      t_temp = times_[index];
      if (index == times_.size())
      {
        FOUR_C_THROW("t_equivalent is bigger than the ending time");
        return 0.0;
      }
    }
  }

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> tfad(1, 0, t_equivalent);
  tfad.val() = Sacado::Fad::DFad<double>(1, 0, t_equivalent);
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> vfad;

  // evaluate the derivative of the function considering the different possibilities
  if (index == 0)
  {
    vfad = timefunction_[0]->second_derivative({{"t", tfad}}, {});
  }
  else
  {
    vfad = timefunction_[index - 1]->second_derivative({{"t", tfad}}, {});
  }

  if (deg == 0)
  {
    const double v = value(t);
    return v;
  }
  if (deg == 1)
  {
    const double dv_dt = vfad.dx(0).val();
    return dv_dt;
  }
  else if (deg == 2)
  {
    const double d2v_dt2 = vfad.dx(0).dx(0);
    return d2v_dt2;
  }
  else
  {
    FOUR_C_THROW("Higher than second derivative is not implemented!");
    return 0.0;
  }
}


bool Core::Utils::MultiFunctionVariable::contain_time(const double t)
{
  /// check the inclusion of the considered time
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  if (t_equivalent >= times_[0] - 1.0e-14 && t_equivalent <= times_[times_.size() - 1] + 1.0e-14)
  {
    return true;
  }
  else
  {
    return false;
  }
}


Core::Utils::FourierInterpolationVariable::FourierInterpolationVariable(std::string name,
    std::vector<double> times, std::vector<double> values, struct Periodicstruct periodicdata)
    : FunctionVariable(std::move(name)),
      times_(std::move(times)),
      values_(std::move(values)),
      periodic_(periodicdata.periodic),
      t1_(periodicdata.t1),
      t2_(periodicdata.t2)
{
}

double Core::Utils::FourierInterpolationVariable::value(const double t)
{
  // evaluate an equivalent time for a periodic variable
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  return value<double>(t_equivalent);
}

template <typename ScalarT>
ScalarT Core::Utils::FourierInterpolationVariable::value(const ScalarT& t)
{
  // source: https://en.wikipedia.org/wiki/Trigonometric_interpolation
  ScalarT value = 0.0;

  // number of interpolation nodes
  auto N = (const double)times_.size();

  // adjusting the spacing of the given independent variable
  const double scale = (times_[1] - times_[0]) * N / 2;

  // evaluate interpolant
  for (unsigned int k = 1; k <= times_.size(); ++k)
  {
    ScalarT tau;
    ScalarT xt = (t - times_[k - 1]) / scale;
    if (xt >= -1.0e-14 and xt <= 1.0e-14)
    {
      tau = 1;
    }
    else
    {
      const int mod = (int)N % 2;
      if (mod == 1)  // odd
      {
        tau = sin(N * M_PI * xt / 2) / (N * sin(M_PI * xt / 2));
      }
      else  // even
      {
        tau = sin(N * M_PI * xt / 2) / (N * tan(M_PI * xt / 2));
      }
    }

    value += values_[k - 1] * tau;
  }

  return value;
}


double Core::Utils::FourierInterpolationVariable::time_derivative_value(
    const double t, const unsigned deg)
{
  // evaluate an equivalent time for a periodic variable
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> tfad(1, 0, t_equivalent);
  tfad.val() = Sacado::Fad::DFad<double>(1, 0, t_equivalent);
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> vfad =
      value<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>>(tfad);

  if (deg == 0)
  {
    const double v = value(t);
    return v;
  }
  if (deg == 1)
  {
    const double dv_dt = vfad.dx(0).val();
    return dv_dt;
  }
  else if (deg == 2)
  {
    const double d2v_dt2 = vfad.dx(0).dx(0);
    return d2v_dt2;
  }
  else
  {
    FOUR_C_THROW("Higher than second derivative is not implemented!");
    return 0.0;
  }
}


bool Core::Utils::FourierInterpolationVariable::contain_time(const double t)
{
  /// check the inclusion of the considered time
  double t_equivalent = t;

  // handle periodicity
  if (periodic_ and t >= t1_ - 1.0e-14 and t <= t2_ + 1.0e-14)
  {
    t_equivalent = fmod(t + 1.0e-14, times_[times_.size() - 1] - times_[0]) - 1.0e-14;
  }

  if (t_equivalent >= times_[0] - 1.0e-14 && t_equivalent <= times_[times_.size() - 1] + 1.0e-14)
  {
    return true;
  }
  else
  {
    return false;
  }
}


Core::Utils::PiecewiseVariable::PiecewiseVariable(
    const std::string& name, std::vector<std::shared_ptr<FunctionVariable>> pieces)
    : FunctionVariable(name), pieces_(std::move(pieces))
{
  if (pieces_.empty())
    FOUR_C_THROW("A PiecewiseVariable must have at least one FunctionVariable piece.");
}


double Core::Utils::PiecewiseVariable::value(const double t)
{
  return find_piece_for_time(t).value(t);
}


double Core::Utils::PiecewiseVariable::time_derivative_value(const double t, const unsigned int deg)
{
  return find_piece_for_time(t).time_derivative_value(t, deg);
}


bool Core::Utils::PiecewiseVariable::contain_time(const double t)
{
  const auto active_piece =
      std::find_if(pieces_.begin(), pieces_.end(), [t](auto& var) { return var->contain_time(t); });

  return active_piece != pieces_.end();
}


Core::Utils::FunctionVariable& Core::Utils::PiecewiseVariable::find_piece_for_time(const double t)
{
  auto active_piece =
      std::find_if(pieces_.begin(), pieces_.end(), [t](auto& var) { return var->contain_time(t); });

  if (active_piece == pieces_.end())
    FOUR_C_THROW("Piece-wise variable <%s> is not defined at time %f.", name().c_str(), t);

  return **active_piece;
}

std::vector<double> Core::Utils::Internal::extract_time_vector(
    const Core::IO::InputParameterContainer& timevar)
{
  // read the number of points
  int numpoints = timevar.get<int>("NUMPOINTS");

  // read whether times are defined by number of points or by vector
  bool bynum = timevar.has_group("BYNUM");

  // read respectively create times vector
  std::vector<double> times = std::invoke(
      [&]()
      {
        if (bynum)  // times defined by number of points
        {
          // read the time range
          auto timerange = timevar.group("BYNUM").get<std::vector<double>>("TIMERANGE");

          std::vector<double> times;

          // get initial and final time
          double t_initial = timerange[0];
          double t_final = timerange[1];

          // build the vector of times
          times.push_back(t_initial);
          int n = 0;
          double dt = (t_final - t_initial) / (numpoints - 1);
          while (times[n] + dt <= t_final + 1.0e-14)
          {
            if (times[n] + 2 * dt <= t_final + 1.0e-14)
            {
              times.push_back(times[n] + dt);
            }
            else
            {
              times.push_back(t_final);
            }
            ++n;
          }
          return times;
        }
        else  // times defined by vector
        {
          auto times = timevar.get<std::vector<double>>("TIMES");

          return times;
        }
      });

  // check if the times are in ascending order
  if (!std::is_sorted(times.begin(), times.end()))
    FOUR_C_THROW("the TIMES must be in ascending order");

  return times;
}

FOUR_C_NAMESPACE_CLOSE
