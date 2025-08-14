// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_rule_simple.hpp"

#include "4C_io_input_field.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <numeric>
#include <vector>

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::SimpleMixtureRule::SimpleMixtureRule(const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureRule(matdata),
      initial_reference_density_(matdata.parameters.get<double>("DENS")),
      mass_fractions_(matdata.parameters.get<Core::IO::InputField<std::vector<double>>>("MASSFRAC"))
{
  // check, whether the mass fractions sum up to 1 in the setup call
}

std::unique_ptr<Mixture::MixtureRule> Mixture::PAR::SimpleMixtureRule::create_rule()
{
  return std::unique_ptr<Mixture::SimpleMixtureRule>(new Mixture::SimpleMixtureRule(this));
}

Mixture::SimpleMixtureRule::SimpleMixtureRule(Mixture::PAR::SimpleMixtureRule* params)
    : MixtureRule(params), params_(params)
{
}

void Mixture::SimpleMixtureRule::setup(const Teuchos::ParameterList& params, int eleGID)
{
  // Call the setup of the base class
  Mixture::MixtureRule::setup(params, eleGID);

  // Check whether the mass fractions sum up to 1
  const auto& fractions = params_->mass_fractions_.at(eleGID);
  const double sum = std::accumulate(fractions.begin(), fractions.end(), 0.0);
  if (std::abs(1.0 - sum) > 1e-8)
    FOUR_C_THROW(
        "Mass fractions at element {} sum to {} instead of 1.0, which is unphysical.", eleGID, sum);
}

void Mixture::SimpleMixtureRule::evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  // define temporary matrices
  Core::LinAlg::SymmetricTensor<double, 3, 3> cstress;
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> ccmat;

  // This is the simplest mixture rule
  // Just iterate over all constituents and add all stress/cmat contributions
  for (std::size_t i = 0; i < constituents().size(); ++i)
  {
    MixtureConstituent& constituent = *constituents()[i];
    cstress = {};
    ccmat = {};
    constituent.evaluate(F, E_strain, params, cstress, ccmat, gp, eleGID);

    // Add stress contribution to global stress
    // In this basic mixture rule, the mass fractions do not change
    double constituent_density =
        params_->initial_reference_density_ * params_->mass_fractions_.at(eleGID)[i];
    S_stress += constituent_density * cstress;
    cmat += constituent_density * ccmat;
  }
}
FOUR_C_NAMESPACE_CLOSE
