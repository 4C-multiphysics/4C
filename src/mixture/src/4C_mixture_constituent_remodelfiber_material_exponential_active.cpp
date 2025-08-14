// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_remodelfiber_material_exponential_active.hpp"

#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_constituent_remodelfiber_lib.hpp"

#include <Sacado.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

template <typename T>
Mixture::PAR::RemodelFiberMaterialExponentialActive<T>::RemodelFiberMaterialExponentialActive(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : RemodelFiberMaterial<T>(matdata),
      passive_params_{matdata.parameters.get<double>("K1"), matdata.parameters.get<double>("K2"),
          matdata.parameters.get<bool>("COMPRESSION")},
      initial_reference_density_(matdata.parameters.get<double>("DENS")),
      sigma_act_max_(matdata.parameters.get<double>("SIGMA_MAX")),
      lambda_act_max_(matdata.parameters.get<double>("LAMBDAMAX")),
      lambda_act_0_(matdata.parameters.get<double>("LAMBDA0")),
      lambda_act_(matdata.parameters.get<double>("LAMBDAACT")),
      dPsiAct_(sigma_act_max_ / initial_reference_density_ *
               (1.0 - std::pow(lambda_act_max_ - lambda_act_, 2) /
                          std::pow(lambda_act_max_ - lambda_act_0_, 2)))
{
}

template <typename T>
std::unique_ptr<Mixture::RemodelFiberMaterial<T>>
Mixture::PAR::RemodelFiberMaterialExponentialActive<T>::create_remodel_fiber_material() const
{
  return std::make_unique<Mixture::RemodelFiberMaterialExponentialActive<T>>(this);
}


template <typename T>
Mixture::RemodelFiberMaterialExponentialActive<T>::RemodelFiberMaterialExponentialActive(
    const PAR::RemodelFiberMaterialExponentialActive<T>* matdata)
    : params_(matdata)
{
}

template <typename T>
T Mixture::RemodelFiberMaterialExponentialActive<T>::get_cauchy_stress(T I4) const
{
  const T dPIact = params_->dPsiAct_;

  return get_exponential_fiber_cauchy_stress<T>(params_->passive_params_, I4) + dPIact;
}

template <typename T>
T Mixture::RemodelFiberMaterialExponentialActive<T>::get_d_cauchy_stress_d_i4(T I4) const
{
  return get_d_exponential_fiber_cauchy_stress_d_i4<T>(params_->passive_params_, I4);
}

template <typename T>
T Mixture::RemodelFiberMaterialExponentialActive<T>::get_d_cauchy_stress_d_i4_d_i4(T I4) const
{
  return get_d_exponential_fiber_cauchy_stress_d_i4_d_i4<T>(params_->passive_params_, I4);
}

template class Mixture::PAR::RemodelFiberMaterialExponentialActive<double>;
template class Mixture::PAR::RemodelFiberMaterialExponentialActive<Sacado::Fad::DFad<double>>;
template class Mixture::RemodelFiberMaterialExponentialActive<double>;
template class Mixture::RemodelFiberMaterialExponentialActive<Sacado::Fad::DFad<double>>;
FOUR_C_NAMESPACE_CLOSE
