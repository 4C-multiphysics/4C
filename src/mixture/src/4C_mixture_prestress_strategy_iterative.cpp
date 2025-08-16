// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_prestress_strategy_iterative.hpp"

#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_tensor_svd.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elast_volsussmanbathe.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mixture_constituent_elasthyper.hpp"
#include "4C_mixture_rule.hpp"

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::IterativePrestressStrategy::IterativePrestressStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : PrestressStrategy(matdata),
      isochoric_(matdata.parameters.get<bool>("ISOCHORIC")),
      is_active_(matdata.parameters.get<bool>("ACTIVE"))
{
}

std::unique_ptr<Mixture::PrestressStrategy>
Mixture::PAR::IterativePrestressStrategy::create_prestress_strategy()
{
  std::unique_ptr<Mixture::PrestressStrategy> prestressStrategy(
      new Mixture::IterativePrestressStrategy(this));
  return prestressStrategy;
}

Mixture::IterativePrestressStrategy::IterativePrestressStrategy(
    Mixture::PAR::IterativePrestressStrategy* params)
    : PrestressStrategy(params), params_(params)
{
}

void Mixture::IterativePrestressStrategy::setup(Mixture::MixtureConstituent& constituent,
    const Teuchos::ParameterList& params, int numgp, int eleGID)
{
  // nothing to do
}

void Mixture::IterativePrestressStrategy::evaluate_prestress(const MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
    Mixture::MixtureConstituent& constituent, Core::LinAlg::SymmetricTensor<double, 3, 3>& G,
    const Teuchos::ParameterList& params, int gp, int eleGID)
{
  // Start with zero prestretch
  G = Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
}

void Mixture::IterativePrestressStrategy::update(
    const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
    Mixture::MixtureConstituent& constituent, const Core::LinAlg::Tensor<double, 3, 3>& F,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& G, const Teuchos::ParameterList& params, int gp,
    int eleGID)
{
  // only update prestress if it is active
  if (!params_->is_active_) return;

  // Compute isochoric part of the deformation
  Core::LinAlg::Tensor<double, 3, 3> F_bar;
  if (params_->isochoric_)
  {
    F_bar = std::pow(Core::LinAlg::det(F), -1.0 / 3.0) * F;
  }
  else
  {
    F_bar = F;
  }
  // Singular value decomposition of F = RU
  const auto [Q, s, VT] = Core::LinAlg::svd(F_bar * G);

  // Compute stretch tensor G = U = V * S * VT
  G = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(VT) * Core::LinAlg::make_rectangular_diagonal_matrix<3, 3>(s) * VT);
}
FOUR_C_NAMESPACE_CLOSE
