// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_muscle_utils.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace
{
  double lineraly_interpolate_between_times(
      const std::vector<std::pair<double, double>>& time_value_pairs, double time)
  {
    // iterator pointing to the first element greater than time
    auto upper_bound_time = std::upper_bound(time_value_pairs.begin(), time_value_pairs.end(), time,
        [](double t, const std::pair<double, double>& pair) { return t < pair.first; });

    // catch errors if time is smaller or larger than the time range
    if (upper_bound_time == time_value_pairs.begin() || upper_bound_time == time_value_pairs.end())
    {
      FOUR_C_THROW(
          "Linear interpolation failed. The given time {} is outside the range of provided "
          "time-value pairs.",
          time);
    }
    // check if exact time value is present (within tolerance)
    else if (std::abs(upper_bound_time->first - time) < 1.0e-12)
    {
      return upper_bound_time->second;
    }
    // linear interpolation for time between times
    else
    {
      auto lower_bound_time = std::prev(upper_bound_time);

      double lower_value = lower_bound_time->second;
      double upper_value = upper_bound_time->second;
      double t1 = lower_bound_time->first;
      double t2 = upper_bound_time->first;

      // do linear interpolation
      return lower_value + (upper_value - lower_value) * (time - t1) / (t2 - t1);
    }
  }
}  // namespace

void Mat::Utils::Muscle::evaluate_lambert(
    const double xi, double& W0, const double tol, const int maxiter)
{
  double W0_old =
      std::numeric_limits<double>::infinity();  // s.t. error is infinite in the beginning
  int numiter = 0;                              // number of iterations

  // Halley's method
  while ((std::abs(W0 - W0_old) / std::abs(W0) > tol) && (numiter <= maxiter))
  {
    numiter++;
    W0_old = W0;
    W0 = W0 - (W0 * std::exp(W0) - xi) /
                  ((std::exp(W0) * (W0 + 1) - (W0 + 2) * (W0 * std::exp(W0) - xi) / (2 * W0 + 2)));
  }

  // error handling
  if (numiter >= maxiter)
  {
    FOUR_C_THROW(
        "Maximal number of iterations for evaluation of Lambert W function with Halley's method "
        "exceeded for tolerance {}.",
        tol);
  }
}

double Mat::Utils::Muscle::evaluate_force_stretch_dependency_ehret(
    const double lambdaM, const double lambdaMin, const double lambdaOpt)
{
  double fxi = 0.0;
  double explambda = std::exp(((2 * lambdaMin - lambdaM - lambdaOpt) * (lambdaM - lambdaOpt)) /
                              (2 * std::pow(lambdaMin - lambdaOpt, 2)));  // prefactor
  if (lambdaM > lambdaMin)
  {
    fxi = ((lambdaM - lambdaMin) / (lambdaOpt - lambdaMin)) * explambda;
  }

  return fxi;
}

double Mat::Utils::Muscle::evaluate_derivative_force_stretch_dependency_ehret(
    const double lambdaM, const double lambdaMin, const double lambdaOpt)
{
  double dFxidLambdaM = 0.0;
  double explambda = std::exp(((2 * lambdaMin - lambdaM - lambdaOpt) * (lambdaM - lambdaOpt)) /
                              (2 * std::pow(lambdaMin - lambdaOpt, 2)));  // prefactor

  if (lambdaM > lambdaMin)
  {
    dFxidLambdaM = ((std::pow(lambdaMin - lambdaM, 2) - std::pow(lambdaMin - lambdaOpt, 2)) /
                       std::pow(lambdaMin - lambdaOpt, 3)) *
                   explambda;
  }

  return dFxidLambdaM;
}

double Mat::Utils::Muscle::evaluate_integral_force_stretch_dependency_ehret(
    const double lambdaM, const double lambdaMin, const double lambdaOpt)
{
  double intFxi = 0.0;
  double explambda = std::exp(((2 * lambdaMin - lambdaM - lambdaOpt) * (lambdaM - lambdaOpt)) /
                              (2 * std::pow(lambdaMin - lambdaOpt, 2)));  // prefactor

  if (lambdaM > lambdaMin)
  {
    intFxi = (lambdaMin - lambdaOpt) * (explambda - std::exp(0.5));
  }

  return intFxi;
}

double Mat::Utils::Muscle::evaluate_force_velocity_dependency_boel(const double dotLambdaM,
    const double dotLambdaMMin, const double de, const double dc, const double ke, const double kc)
{
  // helper variable
  double ratioDotLambdaM = dotLambdaM / dotLambdaMMin;

  // velocity dependency fv
  double fv = 1.0;
  if (dotLambdaM > 0)
  {
    fv = (1 + de) - de * (1 + ratioDotLambdaM) / (1 - ke * kc * ratioDotLambdaM);
  }
  else
  {
    fv = (1 - dc) + dc * (1 - ratioDotLambdaM) / (1 + kc * ratioDotLambdaM);
  }

  return fv;
}

double Mat::Utils::Muscle::evaluate_derivative_force_velocity_dependency_boel(
    const double dotLambdaM, const double dDotLambdaMdLambdaM, const double dotLambdaMMin,
    const double de, const double dc, const double ke, const double kc)
{
  // helper variable
  double ratioDotLambdaM = dotLambdaM / dotLambdaMMin;

  // derivative of fv w.r.t. dotLambdaM
  double dFvdDotLambdaM = 1.0;
  if (dotLambdaM > 0)
  {
    dFvdDotLambdaM =
        -de * ((1 + ke * kc) / (dotLambdaMMin * std::pow((1 - ke * kc * ratioDotLambdaM), 2)));
  }
  else
  {
    dFvdDotLambdaM = -dc * ((1 + kc) / (dotLambdaMMin * std::pow((1 + kc * ratioDotLambdaM), 2)));
  }

  // derivative of fv w.r.t. lambdaM
  double dFvdLambdaM = dFvdDotLambdaM * dDotLambdaMdLambdaM;

  return dFvdLambdaM;
}

double Mat::Utils::Muscle::evaluate_time_dependent_active_stress_ehret(const double Na,
    const int muTypesNum, const std::vector<double>& rho, const std::vector<double>& I,
    const std::vector<double>& F, const std::vector<double>& T, const int actIntervalsNum,
    const std::vector<double>& actTimes, const std::vector<double>& actValues,
    const double currentTime)
{
  // compute twitch force of motor unit (MU) type iMU
  double t_iMU_jstim = 0;                // time of arriving stimulus jstim for MU type iMU
  double t_end = 0;                      // helper variable
  double ratiotime = 1.0;                // helper variable
  std::vector<double> sumg(muTypesNum);  // superposition of single twitches until current time
  std::vector<double> G(muTypesNum);     // gain function for MU type iMU
  std::vector<double> Ft(muTypesNum);    // twitch force for MU type iMU

  // for all motor unit types i
  for (int iMU = 0; iMU < muTypesNum; ++iMU)
  {
    // for all activation intervals
    for (int actinterval = 0; actinterval < actIntervalsNum; ++actinterval)
    {
      // set time of first stimulus jstim=1 to start time of the current activation interval
      t_iMU_jstim = actTimes[actinterval];

      // determine end time of activation interval
      // if inside of actinterval
      if (currentTime < actTimes[actinterval + 1])
      {
        t_end = currentTime;  // set end time to current simulation time
      }
      // if outside of actinterval
      else
      {
        t_end = actTimes[actinterval + 1];  // set end time to end time of actinterval
      }

      // superposition of single twitches
      // for all impulses from start of actinterval until determined end time of actinterval
      while (t_iMU_jstim < t_end)
      {
        ratiotime = (currentTime - t_iMU_jstim) / T[iMU];

        // add single twitch force response for stimulus jstim and motor unit iMU scaled by
        // percentage activation prescribed in actinterval
        sumg[iMU] += actValues[actinterval] * ratiotime * F[iMU] * std::exp(1 - ratiotime);

        // next impulse jstim at time t_iMU_jstim after stimulation interval I
        t_iMU_jstim += I[iMU];
      }
    }  // end actinterval

    // gain function for MU type iMU
    G[iMU] = (1 - std::exp(-2 * std::pow(T[iMU] / I[iMU], 3))) / (T[iMU] / I[iMU]);

    // twitch force for MU type iMU
    Ft[iMU] = G[iMU] * sumg[iMU];
  }  // end iMU

  // compute force-time/stimulation frequency dependency Poptft
  double Poptft = 0.0;
  for (int iMU = 0; iMU < muTypesNum; ++iMU)
  {
    // sum up twitch forces for all MU types weighted by the respective MU type fraction
    Poptft += Na * Ft[iMU] * rho[iMU];
  }

  return Poptft;
}

double Mat::Utils::Muscle::evaluate_active_force_stretch_dependency_blemker(
    const double lambdaM, const double lambdaOpt)
{
  // helper variable
  double ratio_lambda = lambdaM / lambdaOpt;

  // active stretch dependency fxi
  double fxi = 1.0;
  if (lambdaM <= 0.6 * lambdaOpt)
  {
    fxi = 9 * std::pow(ratio_lambda - 0.4, 2.0);
  }
  else if (lambdaM < 1.4 * lambdaOpt)
  {
    fxi = 1 - 4 * std::pow(1 - ratio_lambda, 2.0);
  }
  else
  {
    fxi = 9 * std::pow(ratio_lambda - 1.6, 2.0);
  }

  return fxi;
}

double Mat::Utils::Muscle::evaluate_derivative_active_force_stretch_dependency_blemker(
    const double lambdaM, const double lambdaOpt)
{
  // helper variable
  double ratio_lambda = lambdaM / lambdaOpt;

  // derivative of active stretch dependency fxi w.r.t. fiber stretch
  double dFxidLambdaM = 1.0;
  if (lambdaM <= 0.6 * lambdaOpt)
  {
    dFxidLambdaM = 18 / lambdaOpt * (ratio_lambda - 0.4);
  }
  else if (lambdaM < 1.4 * lambdaOpt)
  {
    dFxidLambdaM = 8 / lambdaOpt * (1 - ratio_lambda);
  }
  else
  {
    dFxidLambdaM = 18 / lambdaOpt * (ratio_lambda - 1.6);
  }

  return dFxidLambdaM;
}

double Mat::Utils::Muscle::evaluate_passive_force_stretch_dependency_blemker(const double lambdaM,
    const double lambdaOpt, const double lambdaStar, const double P1, const double P2)
{
  // helper variable
  double ratio_lambda = lambdaM / lambdaOpt;

  // calculate constants
  double P3 = P1 * P2 * std::exp(P2 * (lambdaStar / lambdaOpt - 1.0)) / lambdaOpt;
  double P4 =
      P1 * (std::exp(P2 * ((lambdaStar / lambdaOpt) - 1.0)) - 1.0) - P3 * lambdaStar / lambdaOpt;

  // passive stretch dependency fxi
  double fxi = 1.0;
  if (lambdaM <= lambdaOpt)
  {
    fxi = 0.0;
  }
  else if (lambdaM < lambdaStar)
  {
    fxi = P1 * (std::exp(P2 * (ratio_lambda - 1.0)) - 1.0);
  }
  else
  {
    fxi = P3 * ratio_lambda + P4;
  }

  return fxi;
}

double Mat::Utils::Muscle::evaluate_derivative_passive_force_stretch_dependency_blemker(
    const double lambdaM, const double lambdaOpt, const double lambdaStar, const double P1,
    const double P2)
{
  // helper variable
  double ratio_lambda = lambdaM / lambdaOpt;

  // calculate constant
  double P3 = P1 * P2 * std::exp(P2 * (lambdaStar / lambdaOpt - 1.0)) / lambdaOpt;

  // derivative of passive stretch dependency fxi w.r.t. fibre stretch
  double dFxidLambdaM = 1.0;
  if (lambdaM <= lambdaOpt)
  {
    dFxidLambdaM = 0.0;
  }
  else if (lambdaM < lambdaStar)
  {
    dFxidLambdaM = P1 * std::exp(P2 * (ratio_lambda - 1.0)) * P2 / lambdaOpt;
  }
  else
  {
    dFxidLambdaM = P3 / lambdaOpt;
  }

  return dFxidLambdaM;
}

double Mat::Utils::Muscle::evaluate_time_dependent_active_stress_tanh(const double sigma_max,
    const double alpha, const double beta, const double t_act_start, const double t_current)
{
  // compute time-dependency ft
  double ft = 0;
  if (t_current >= t_act_start)
  {
    ft = alpha * std::tanh(beta * (t_current - t_act_start));
  }

  // compute active optimal stress at current time
  double sigma_max_ft = sigma_max * ft;

  return sigma_max_ft;
}

double Mat::Utils::Muscle::evaluate_time_space_dependent_active_stress_by_funct(
    const double sigma_max, const Core::Utils::FunctionOfSpaceTime& activation_function,
    const double t_current, const Core::LinAlg::Matrix<3, 1>& x)
{
  const std::vector<double> x_vec{x(0), x(1), x(2)};

  // compute time-dependency ft
  const double ft = activation_function.evaluate(&x_vec.front(), t_current, 0);

  // ft needs to be in interval [0, 1]
  if (ft < 0.00 || ft > 1.00)
    FOUR_C_THROW(
        "Function value not physical, please prescribe a function with values in interval [0,1].");

  const double sigma_max_ft = sigma_max * ft;

  return sigma_max_ft;
}

double Mat::Utils::Muscle::evaluate_time_space_dependent_active_stress_by_map(
    const double sigma_max,
    const std::unordered_map<int, std::vector<std::pair<double, double>>>& activation_map,
    const double t_current, const int activation_map_key)
{
  // compute time-dependency ft
  auto it = activation_map.find(activation_map_key);

  if (it == activation_map.end())
  {
    FOUR_C_THROW("Key (element id) {} not found in activation map.", activation_map_key);
  }
  const double ft = lineraly_interpolate_between_times(it->second, t_current);

  // ft needs to be in interval [0, 1]
  if (ft < 0.00 || ft > 1.00)
    FOUR_C_THROW(
        "Function value not physical, please prescribe activation values in interval [0,1].");

  const double sigma_max_ft = sigma_max * ft;

  return sigma_max_ft;
}

double Mat::Utils::Muscle::fiber_stretch(
    const Core::LinAlg::Matrix<3, 3>& C, const Core::LinAlg::Matrix<3, 3>& M)
{
  // product C^T*M
  Core::LinAlg::Matrix<3, 3> transpCM(Core::LinAlg::Initialization::uninitialized);
  transpCM.multiply_tn(C, M);  // C^TM = C^T*M

  // stretch in fibre direction lambdaM
  // lambdaM = sqrt(C:M) = sqrt(tr(C^T M)), see Holzapfel2000, p.14
  double lambdaM = std::sqrt(transpCM(0, 0) + transpCM(1, 1) + transpCM(2, 2));

  return lambdaM;
}

Core::LinAlg::Matrix<3, 3> Mat::Utils::Muscle::d_fiber_stretch_dc(
    const double lambdaM, const Core::LinAlg::Matrix<3, 3>& C, const Core::LinAlg::Matrix<3, 3>& M)
{
  // derivative of lambdaM w.r.t. C
  Core::LinAlg::Matrix<3, 3> dlambdaMdC(M);
  dlambdaMdC.scale(0.5 / lambdaM);

  return dlambdaMdC;
}

double Mat::Utils::Muscle::contraction_velocity_bw_euler(
    const double lambdaM, const double lambdaMOld, const double timeStepSize)
{
  double dotLambdaM = (lambdaM - lambdaMOld) / timeStepSize;
  return dotLambdaM;
}
FOUR_C_NAMESPACE_CLOSE
