// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_prestress_strategy_constant.hpp"

#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_matrix_conversion.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_coordinate_system_provider.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_elast_volsussmanbathe.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mixture_constituent_elasthyper.hpp"
#include "4C_mixture_rule.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::ConstantPrestressStrategy::ConstantPrestressStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : PrestressStrategy(matdata), prestretch_()
{
  std::ranges::copy(matdata.parameters.get<std::vector<double>>("PRESTRETCH"), prestretch_.begin());
}

std::unique_ptr<Mixture::PrestressStrategy>
Mixture::PAR::ConstantPrestressStrategy::create_prestress_strategy()
{
  std::unique_ptr<Mixture::PrestressStrategy> prestressStrategy(
      new Mixture::ConstantPrestressStrategy(this));
  return prestressStrategy;
}

Mixture::ConstantPrestressStrategy::ConstantPrestressStrategy(
    Mixture::PAR::ConstantPrestressStrategy* params)
    : PrestressStrategy(params), params_(params)
{
}

void Mixture::ConstantPrestressStrategy::setup(Mixture::MixtureConstituent& constituent,
    const Teuchos::ParameterList& params, int numgp, int eleGID)
{
  // nothing to do
}

void Mixture::ConstantPrestressStrategy::evaluate_prestress(const MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
    Mixture::MixtureConstituent& constituent, Core::LinAlg::SymmetricTensor<double, 3, 3>& G,
    const Teuchos::ParameterList& params, int gp, int eleGID)
{
  // setup prestretch
  const Core::LinAlg::Matrix<6, 1> prestretch_vector(params_->prestretch_.data(), true);

  G = Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(prestretch_vector);
}

void Mixture::ConstantPrestressStrategy::update(
    const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
    Mixture::MixtureConstituent& constituent, const Core::LinAlg::Tensor<double, 3, 3>& F,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& G, const Teuchos::ParameterList& params, int gp,
    int eleGID)
{
}
FOUR_C_NAMESPACE_CLOSE
