// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_MATERIAL_EXPONENTIAL_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_MATERIAL_EXPONENTIAL_HPP

#include "4C_config.hpp"

#include "4C_mixture_constituent_remodelfiber_lib.hpp"
#include "4C_mixture_constituent_remodelfiber_material.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  // forward declarations
  template <typename T>
  class RemodelFiberMaterialExponential;

  namespace PAR
  {
    template <typename T>
    class RemodelFiberMaterialExponential : public RemodelFiberMaterial<T>
    {
      friend class Mixture::RemodelFiberMaterialExponential<T>;

     public:
      explicit RemodelFiberMaterialExponential(const Core::Mat::PAR::Parameter::Data& matdata);

      [[nodiscard]] std::unique_ptr<Mixture::RemodelFiberMaterial<T>>
      create_remodel_fiber_material() const override;

      /// @name parameters of the exponential strain energy function
      /// @{
      Mixture::PAR::ExponentialFiberParameters params_;
      /// @}
    };
  }  // namespace PAR

  template <typename T>
  class RemodelFiberMaterialExponential : public RemodelFiberMaterial<T>
  {
   public:
    RemodelFiberMaterialExponential(const PAR::RemodelFiberMaterialExponential<T>* matdata);

    [[nodiscard]] T get_cauchy_stress(T I4) const final;

    [[nodiscard]] T get_d_cauchy_stress_d_i4(T I4) const final;

    [[nodiscard]] T get_d_cauchy_stress_d_i4_d_i4(T I4) const final;

   private:
    const PAR::RemodelFiberMaterialExponential<T>* params_;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
