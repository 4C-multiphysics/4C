// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_prestress_strategy_prescribed.hpp"

#include "4C_io_input_field.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_mat_anisotropy_coordinate_system_provider.hpp"
#include "4C_mat_service.hpp"
#include "4C_mixture_rule.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::PrescribedPrestressStrategy::PrescribedPrestressStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : PrestressStrategy(matdata),
      prestretch_(matdata.parameters
              .get<Core::IO::InterpolatedInputField<Core::LinAlg::SymmetricTensor<double, 3, 3>,
                  Core::LinAlg::SymmetricPositiveDefiniteInterpolation<double>>>("PRESTRETCH"))
{
}

std::unique_ptr<Mixture::PrestressStrategy>
Mixture::PAR::PrescribedPrestressStrategy::create_prestress_strategy()
{
  std::unique_ptr<Mixture::PrestressStrategy> prestressStrategy(
      new Mixture::PrescribedPrestressStrategy(this));
  return prestressStrategy;
}

Mixture::PrescribedPrestressStrategy::PrescribedPrestressStrategy(
    Mixture::PAR::PrescribedPrestressStrategy* params)
    : PrestressStrategy(params), params_(params)
{
}

void Mixture::PrescribedPrestressStrategy::setup(Mixture::MixtureConstituent& constituent,
    const Teuchos::ParameterList& params, int numgp, int eleGID)
{
  // nothing to do
}

void Mixture::PrescribedPrestressStrategy::evaluate_prestress(const MixtureRule& mixtureRule,
    const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
    Mixture::MixtureConstituent& constituent, Core::LinAlg::SymmetricTensor<double, 3, 3>& G,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, int gp, int eleGID)
{
  G = params_->prestretch_.interpolate(eleGID, context.xi->as_span());
}

void Mixture::PrescribedPrestressStrategy::update(
    const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
    Mixture::MixtureConstituent& constituent, const Core::LinAlg::Tensor<double, 3, 3>& F,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& G, const Teuchos::ParameterList& params,
    const Mat::EvaluationContext& context, int gp, int eleGID)
{
}
FOUR_C_NAMESPACE_CLOSE
