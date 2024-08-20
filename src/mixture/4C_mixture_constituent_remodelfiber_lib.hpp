/*----------------------------------------------------------------------*/
/*! \file
\brief Declaration of helper functions for the remodel fiber constituent
\level 3
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_LIB_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_LIB_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <Sacado.hpp>

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace MIXTURE::PAR
{
  template <typename T>
  class RemodelFiberMaterial;

  /*!
   * @brief Create a remodel fiber material from the material id given in the input file
   *
   * @param matid material id given in the input file
   * @return const MIXTURE::PAR::RemodelFiberMaterial<double>*
   */
  [[nodiscard]] const MIXTURE::PAR::RemodelFiberMaterial<double>* fiber_material_factory(int matid);

  struct ExponentialFiberParameters
  {
    double k1_;
    double k2_;
    bool supports_compression_;
  };
}  // namespace MIXTURE::PAR

namespace MIXTURE
{
  template <typename T>
  [[nodiscard]] inline T get_exponential_fiber_strain_energy(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 0 && !params.supports_compression_)
      FOUR_C_THROW("The fiber is under compression, but does not support that.");

    return (params.k1_ / (2.0 * params.k2_)) *
           (std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0)) - 1.0);
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_strain_energy_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 0 && !params.supports_compression_)
      FOUR_C_THROW("The fiber is under compression, but does not support that.");

    return params.k1_ * (I4 - 1.0) * std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0));
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_strain_energy_d_i4_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 0 && !params.supports_compression_)
      FOUR_C_THROW("The fiber is under compression, but does not support that.");

    return (1.0 + 2.0 * params.k2_ * std::pow((I4 - 1.0), 2)) * params.k1_ *
           std::exp(params.k2_ * std::pow((I4 - 1.0), 2));
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_strain_energy_d_i4_d_i4_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    if (I4 < 0 && !params.supports_compression_)
      FOUR_C_THROW("The fiber is under compression, but does not support that.");

    return 4 * params.k2_ * (I4 - 1.0) * params.k1_ *
               std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0)) +
           (1 + 2 * params.k2_ * (I4 - 1.0) * (I4 - 1.0)) * params.k1_ * 2 * params.k2_ *
               (I4 - 1.0) * std::exp(params.k2_ * (I4 - 1.0) * (I4 - 1.0));
  }

  template <typename T>
  [[nodiscard]] inline T get_exponential_fiber_cauchy_stress(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    const T dPsi = MIXTURE::get_d_exponential_fiber_strain_energy_d_i4<T>(params, I4);

    return 2.0 * dPsi * I4;
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_cauchy_stress_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    const T dPsi = MIXTURE::get_d_exponential_fiber_strain_energy_d_i4<T>(params, I4);
    const T ddPsi = MIXTURE::get_d_exponential_fiber_strain_energy_d_i4_d_i4<T>(params, I4);

    return 2.0 * (dPsi + I4 * ddPsi);
  }

  template <typename T>
  [[nodiscard]] inline T get_d_exponential_fiber_cauchy_stress_d_i4_d_i4(
      const PAR::ExponentialFiberParameters& params, const T I4)
  {
    const T ddPsi = MIXTURE::get_d_exponential_fiber_strain_energy_d_i4_d_i4<T>(params, I4);
    const T dddPsi = MIXTURE::get_d_exponential_fiber_strain_energy_d_i4_d_i4_d_i4<T>(params, I4);

    return 2.0 * (2 * ddPsi + I4 * dddPsi);
  }
}  // namespace MIXTURE

FOUR_C_NAMESPACE_CLOSE

#endif
