// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_MATERIAL_EXPONENTIAL_ACTIVE_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_MATERIAL_EXPONENTIAL_ACTIVE_HPP

#include "4C_config.hpp"

#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  // forward declarations
  template <typename T>
  class RemodelFiberMaterialExponentialActive;

  namespace PAR
  {
    template <typename T>
    class RemodelFiberMaterialExponentialActive : public RemodelFiberMaterial<T>
    {
      friend class Mixture::RemodelFiberMaterialExponentialActive<T>;

     public:
      explicit RemodelFiberMaterialExponentialActive(
          const Core::Mat::PAR::Parameter::Data& matdata);

      [[nodiscard]] std::unique_ptr<Mixture::RemodelFiberMaterial<T>>
      create_remodel_fiber_material() const override;

      Mixture::PAR::ExponentialFiberParameters passive_params_;

      /// @name parameters of the exponential strain energy function
      /// @{
      const double initial_reference_density_;
      const double sigma_act_max_;
      const double lambda_act_max_;
      const double lambda_act_0_;
      const double lambda_act_;
      const double dPsiAct_;
      /// @}
    };
  }  // namespace PAR

  template <typename T>
  class RemodelFiberMaterialExponentialActive : public RemodelFiberMaterial<T>
  {
   public:
    RemodelFiberMaterialExponentialActive(
        const PAR::RemodelFiberMaterialExponentialActive<T>* matdata);

    [[nodiscard]] T get_cauchy_stress(T I4) const final;

    [[nodiscard]] T get_d_cauchy_stress_d_i4(T I4) const final;

    [[nodiscard]] T get_d_cauchy_stress_d_i4_d_i4(T I4) const final;

   private:
    const PAR::RemodelFiberMaterialExponentialActive<T>* params_;
  };

}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
