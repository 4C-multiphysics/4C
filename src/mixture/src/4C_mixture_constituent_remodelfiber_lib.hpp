// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_LIB_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_LIB_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <Sacado.hpp>

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace Mixture::PAR
{
  template <typename T>
  class RemodelFiberMaterial;

  /*!
   * @brief Create a remodel fiber material from the material id given in the input file
   *
   * @param matid material id given in the input file
   * @return const Mixture::PAR::RemodelFiberMaterial<double>*
   */
  [[nodiscard]] const Mixture::PAR::RemodelFiberMaterial<double>* fiber_material_factory(int matid);

  struct ExponentialFiberParameters
  {
    double k1_;
    double k2_;
    bool supports_compression_;
  };
}  // namespace Mixture::PAR

namespace Mixture
{
  template <typename T>
  [[nodiscard]] inline T get_exponential_fiber_strain_energy(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 1 && !params.supports_compression_)
    {
      return 0.0;  // Following Holzapfel & Ogden (2009): Fibers under compression do not contribute
    }

    return (params.k1_ / (2.0 * params.k2_)) *
           (std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0)) - 1.0);
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_strain_energy_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 1 && !params.supports_compression_)
    {
      return 0.0;  // Following Holzapfel & Ogden (2009): Fibers under compression do not contribute
    }

    return params.k1_ * (I4 - 1.0) * std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0));
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_strain_energy_d_i4_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 1 && !params.supports_compression_)
    {
      return 0.0;  // Following Holzapfel & Ogden (2009): Fibers under compression do not contribute
    }

    return (1.0 + 2.0 * params.k2_ * std::pow((I4 - 1.0), 2)) * params.k1_ *
           std::exp(params.k2_ * std::pow((I4 - 1.0), 2));
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_strain_energy_d_i4_d_i4_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 1 && !params.supports_compression_)
    {
      return 0.0;  // Following Holzapfel & Ogden (2009): Fibers under compression do not contribute
    }

    return 4 * params.k2_ * (I4 - 1.0) * params.k1_ *
               std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0)) +
           (1 + 2 * params.k2_ * (I4 - 1.0) * (I4 - 1.0)) * params.k1_ * 2 * params.k2_ *
               (I4 - 1.0) * std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0));
  }

  template <typename T>
  [[nodiscard]] inline T get_exponential_fiber_cauchy_stress(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    const T dPsi = Mixture::get_d_exponential_fiber_strain_energy_d_i4<T>(params, I4);

    return 2.0 * dPsi * I4;
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_cauchy_stress_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    const T dPsi = Mixture::get_d_exponential_fiber_strain_energy_d_i4<T>(params, I4);
    const T ddPsi = Mixture::get_d_exponential_fiber_strain_energy_d_i4_d_i4<T>(params, I4);

    return 2.0 * (dPsi + I4 * ddPsi);
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_cauchy_stress_d_i4_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    const T ddPsi = Mixture::get_d_exponential_fiber_strain_energy_d_i4_d_i4<T>(params, I4);
    const T dddPsi = Mixture::get_d_exponential_fiber_strain_energy_d_i4_d_i4_d_i4<T>(params, I4);

    return 2.0 * (2 * ddPsi + I4 * dddPsi);
  }
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
