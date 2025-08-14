// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_rule_function.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <string>


FOUR_C_NAMESPACE_OPEN

namespace
{
  std::vector<const Core::Utils::FunctionOfSpaceTime*> create_functions_from_function_ids(
      const std::vector<int>& funct_ids)
  {
    std::vector<const Core::Utils::FunctionOfSpaceTime*> functions;
    // get function handles from function ids
    for (int id : funct_ids)
    {
      const auto* function =
          &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(id);

      FOUR_C_ASSERT(
          function != nullptr, "pointer to mass fraction function with id {} is nullptr!", id);

      functions.emplace_back(function);
    }
    return functions;
  }
}  // namespace

Mixture::PAR::FunctionMixtureRule::FunctionMixtureRule(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureRule(matdata),
      initial_reference_density_(matdata.parameters.get<double>("DENS")),
      mass_fractions_funct_ids_(matdata.parameters.get<std::vector<int>>("MASSFRACFUNCT")) {};

std::unique_ptr<Mixture::MixtureRule> Mixture::PAR::FunctionMixtureRule::create_rule()
{
  return std::make_unique<Mixture::FunctionMixtureRule>(this);
}

Mixture::FunctionMixtureRule::FunctionMixtureRule(Mixture::PAR::FunctionMixtureRule* params)
    : MixtureRule(params), params_(params), mass_fractions_functions_()
{
  // cannot setup mass_fractions_functions_ here because at this state, functions are not yet read
  // from input
}

void Mixture::FunctionMixtureRule::setup(const Teuchos::ParameterList& params, const int eleGID)
{
  MixtureRule::setup(params, eleGID);

  mass_fractions_functions_ =
      create_functions_from_function_ids(params_->mass_fractions_funct_ids_);
}

void Mixture::FunctionMixtureRule::unpack_mixture_rule(Core::Communication::UnpackBuffer& buffer)
{
  Mixture::MixtureRule::unpack_mixture_rule(buffer);

  mass_fractions_functions_ =
      create_functions_from_function_ids(params_->mass_fractions_funct_ids_);
}

void Mixture::FunctionMixtureRule::evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  // define temporary matrices
  Core::LinAlg::SymmetricTensor<double, 3, 3> cstress;
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> ccmat;

  // initialize sum of mass fractions for validity check
  double sum = 0.0;

  // Iterate over all constituents and add all stress/cmat contributions
  for (std::size_t i = 0; i < constituents().size(); ++i)
  {
    // mass fractions are defined by evaluating the specified function at the gauss point reference
    // coordinates (and the current time)

    // get gauss point reference coordinates and current time
    const auto& reference_coordinates =
        params.get<Core::LinAlg::Tensor<double, 3>>("gp_coords_ref");
    const double time = params.get<double>("total time");

    // evaluate the mass fraction function at the gauss point reference coordinates and current time
    const double massfrac =
        mass_fractions_functions_[i]->evaluate(reference_coordinates.data(), time, 0);
    sum += massfrac;
    double constituent_density = params_->initial_reference_density_ * massfrac;

    // add stress contribution to global stress
    MixtureConstituent& constituent = *constituents()[i];
    cstress = {};
    ccmat = {};
    constituent.evaluate(F, E_strain, params, cstress, ccmat, gp, eleGID);

    S_stress += constituent_density * cstress;
    cmat += constituent_density * ccmat;
  }

  // validity check whether mass fractions summed up to 1
  if (std::abs(1.0 - sum) > 1e-8)
    FOUR_C_THROW("Evaluated mass fractions don't sum up to 1, which is unphysical.");
}


FOUR_C_NAMESPACE_CLOSE
