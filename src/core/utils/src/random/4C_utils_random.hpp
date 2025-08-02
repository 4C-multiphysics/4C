// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_RANDOM_HPP
#define FOUR_C_UTILS_RANDOM_HPP

#include "4C_config.hpp"

#include <random>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  /*!
  \brief handles random numbers
  */
  class Random
  {
   public:
    /// get a random number
    double uni();

    /// get a vector of random numbers of size count
    void uni(std::vector<double>& randvec, int count);

    /// get a random number
    double normal();

    /// get a vector of random numbers of size count
    void normal(std::vector<double>& randvec, int count);

    /// set the random seed
    void set_rand_seed(const unsigned int seed);

    /// set the range for the uniform rng
    void set_rand_range(const double lower, const double upper);

    /// Set the mean and standard deviation for the normal rng
    void set_mean_stddev(const double mean, const double stddev);

   private:
    /// @name Random number generation
    /// @{
    /// random number engine
    std::mt19937 rand_engine_{0};

    /// uniform random number distribution between -1.0 and 1.0
    std::uniform_real_distribution<double> uni_dist_{-1.0, 1.0};

    /// unit normal random number distribution
    std::normal_distribution<double> norm_dist_{};
    //@}
  };
}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
