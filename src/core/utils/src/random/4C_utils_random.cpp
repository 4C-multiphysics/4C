// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_random.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/// get a random number
double Core::Utils::Random::uni() { return uni_dist_(rand_engine_); }

/// get a vector of random numbers of size count
void Core::Utils::Random::uni(std::vector<double>& randvec, int count)
{
  // resize vector
  randvec.resize(count);

  for (int i = 0; i < count; ++i)
  {
    randvec[i] = uni_dist_(rand_engine_);
  }
}

/// get a random number
double Core::Utils::Random::normal() { return norm_dist_(rand_engine_); }

/// get a vector of random numbers of size count
void Core::Utils::Random::normal(std::vector<double>& randvec, int count)
{
  // resize vector
  randvec.resize(count);

  for (int i = 0; i < count; ++i)
  {
    randvec[i] = norm_dist_(rand_engine_);
  }
}

/// set the random seed
void Core::Utils::Random::set_rand_seed(const unsigned int seed) { rand_engine_.seed(seed); }

/// set the range for the uniform rng
void Core::Utils::Random::set_rand_range(const double lower, const double upper)
{
  std::uniform_real_distribution<double>::param_type parameters(lower, upper);
  uni_dist_.param(parameters);
}

void Core::Utils::Random::set_mean_stddev(const double mean, const double stddev)
{
  FOUR_C_ASSERT_ALWAYS(stddev > 0.0,
      "Standard deviation of normal distribution must be positive, but is {}.", stddev);
  std::normal_distribution<double>::param_type parameters(mean, stddev);
  norm_dist_.param(parameters);
}


FOUR_C_NAMESPACE_CLOSE
