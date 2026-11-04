// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupanisoexpo.hpp"

#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::CoupAnisoExpoAnisotropyExtension::CoupAnisoExpoAnisotropyExtension(
    const int init_mode, const double gamma, const bool adapt_angle,
    const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& structuralTensorStrategy,
    const int fiber_id)
    : DefaultAnisotropyExtension<1>(
          init_mode, gamma, adapt_angle, structuralTensorStrategy, {fiber_id - 1})
{
}

double Mat::Elastic::CoupAnisoExpoAnisotropyExtension::get_scalar_product(int gp) const
{
  return 1.0;
}

const Core::LinAlg::Tensor<double, 3>& Mat::Elastic::CoupAnisoExpoAnisotropyExtension::get_fiber(
    int gp) const
{
  return DefaultAnisotropyExtension<1>::get_fiber(gp, 0);
}

const Core::LinAlg::SymmetricTensor<double, 3, 3>&
Mat::Elastic::CoupAnisoExpoAnisotropyExtension::get_structural_tensor(int gp) const
{
  return DefaultAnisotropyExtension<1>::get_structural_tensor(gp, 0);
}

Mat::Elastic::PAR::CoupAnisoExpo::CoupAnisoExpo(const Core::Mat::PAR::Parameter::Data& matdata)
    : Mat::PAR::ParameterAniso(matdata),
      Mat::Elastic::PAR::CoupAnisoExpoBase(matdata),
      adapt_angle_(matdata.parameters.get<bool>("ADAPT_ANGLE")),
      fiber_id_(matdata.parameters.get<int>("FIBER_ID"))
{
}

Mat::Elastic::CoupAnisoExpo::CoupAnisoExpo(Mat::Elastic::PAR::CoupAnisoExpo* params)
    : Mat::Elastic::CoupAnisoExpoBase(params),
      params_(params),
      anisotropy_extension_(params_->init_, params->gamma_, params_->adapt_angle_ != 0,
          params_->structural_tensor_strategy(), params->fiber_id_)
{
  anisotropy_extension_.register_needed_tensors(
      FiberAnisotropyExtension<1>::FIBER_VECTORS | FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

void Mat::Elastic::CoupAnisoExpo::register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mat::Elastic::CoupAnisoExpo::pack_summand(Core::Communication::PackBuffer& data) const
{
  anisotropy_extension_.pack_anisotropy(data);
}

void Mat::Elastic::CoupAnisoExpo::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mat::Elastic::CoupAnisoExpo::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
) const
{
  if (anisotropy_extension_.fibers_initialized())
    fibervecs.push_back(anisotropy_extension_.get_fiber(BaseAnisotropyExtension::GPDEFAULT));
}

void Mat::Elastic::CoupAnisoExpo::set_fiber_vecs(const double newgamma,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  anisotropy_extension_.set_fiber_vecs(newgamma, locsys, defgrd);
}

void Mat::Elastic::CoupAnisoExpo::set_fiber_vecs(const Core::LinAlg::Tensor<double, 3>& fibervec)
{
  anisotropy_extension_.set_fiber_vecs(fibervec);
}

FOUR_C_NAMESPACE_CLOSE
