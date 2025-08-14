// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_growth_strategy_anisotropic.hpp"

#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_growth_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::AnisotropicGrowthStrategy::AnisotropicGrowthStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Mixture::PAR::MixtureGrowthStrategy(matdata),
      init_mode_(matdata.parameters.get<int>("INIT")),
      fiber_id_(matdata.parameters.get<int>("FIBER_ID"))
{
}

std::unique_ptr<Mixture::MixtureGrowthStrategy>
Mixture::PAR::AnisotropicGrowthStrategy::create_growth_strategy()
{
  return std::make_unique<Mixture::AnisotropicGrowthStrategy>(this);
}

Mixture::AnisotropicGrowthStrategy::AnisotropicGrowthStrategy(
    Mixture::PAR::AnisotropicGrowthStrategy* params)
    : params_(params),
      anisotropy_extension_(params_->init_mode_, 0.0, false,
          std::make_shared<Mat::Elastic::StructuralTensorStrategyStandard>(nullptr),
          {params->fiber_id_ - 1})
{
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

void Mixture::AnisotropicGrowthStrategy::pack_mixture_growth_strategy(
    Core::Communication::PackBuffer& data) const
{
  MixtureGrowthStrategy::pack_mixture_growth_strategy(data);

  anisotropy_extension_.pack_anisotropy(data);
}

void Mixture::AnisotropicGrowthStrategy::unpack_mixture_growth_strategy(
    Core::Communication::UnpackBuffer& buffer)
{
  MixtureGrowthStrategy::unpack_mixture_growth_strategy(buffer);

  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mixture::AnisotropicGrowthStrategy::register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mixture::AnisotropicGrowthStrategy::evaluate_inverse_growth_deformation_gradient(
    Core::LinAlg::Tensor<double, 3, 3>& iFgM, const Mixture::MixtureRule& mixtureRule,
    double currentReferenceGrowthScalar, int gp) const
{
  const Core::LinAlg::Matrix<3, 3> Id = Core::LinAlg::identity_matrix<3>();

  iFgM = Core::LinAlg::get_full((1.0 / currentReferenceGrowthScalar - 1.0) *
                                    anisotropy_extension_.get_structural_tensor(gp, 0) +
                                Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

void Mixture::AnisotropicGrowthStrategy::evaluate_growth_stress_cmat(
    const Mixture::MixtureRule& mixtureRule, double currentReferenceGrowthScalar,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& dCurrentReferenceGrowthScalarDC,
    const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID) const
{
  S_stress = {};
  cmat = {};
}
FOUR_C_NAMESPACE_CLOSE
