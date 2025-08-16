// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"

#include "4C_mat_par_bundle.hpp"

#include <Sacado.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

template <typename T>
Mixture::PAR::RemodelFiberMaterialExponential<T>::RemodelFiberMaterialExponential(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : RemodelFiberMaterial<T>(matdata),
      params_{matdata.parameters.get<double>("K1"), matdata.parameters.get<double>("K2"),
          matdata.parameters.get<bool>("COMPRESSION")}
{
}

template <typename T>
std::unique_ptr<Mixture::RemodelFiberMaterial<T>>
Mixture::PAR::RemodelFiberMaterialExponential<T>::create_remodel_fiber_material() const
{
  return std::make_unique<Mixture::RemodelFiberMaterialExponential<T>>(this);
}

template <typename T>
Mixture::RemodelFiberMaterialExponential<T>::RemodelFiberMaterialExponential(
    const PAR::RemodelFiberMaterialExponential<T>* matdata)
    : params_(matdata)
{
}

template <typename T>
T Mixture::RemodelFiberMaterialExponential<T>::get_cauchy_stress(T I4) const
{
  return Mixture::get_exponential_fiber_cauchy_stress<T>(params_->params_, I4);
}

template <typename T>
T Mixture::RemodelFiberMaterialExponential<T>::get_d_cauchy_stress_d_i4(T I4) const
{
  return Mixture::get_d_exponential_fiber_cauchy_stress_d_i4<T>(params_->params_, I4);
}

template <typename T>
T Mixture::RemodelFiberMaterialExponential<T>::get_d_cauchy_stress_d_i4_d_i4(T I4) const
{
  return Mixture::get_d_exponential_fiber_cauchy_stress_d_i4_d_i4<T>(params_->params_, I4);
}

template class Mixture::PAR::RemodelFiberMaterialExponential<double>;
template class Mixture::PAR::RemodelFiberMaterialExponential<Sacado::Fad::DFad<double>>;
template class Mixture::RemodelFiberMaterialExponential<double>;
template class Mixture::RemodelFiberMaterialExponential<Sacado::Fad::DFad<double>>;
FOUR_C_NAMESPACE_CLOSE
