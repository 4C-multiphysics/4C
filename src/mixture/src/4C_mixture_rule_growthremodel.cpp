// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_rule_growthremodel.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_mixture_growth_strategy.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

#include <algorithm>
#include <cmath>
#include <iosfwd>
#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::Communication
{
  class PackBuffer;
}

Mixture::PAR::GrowthRemodelMixtureRule::GrowthRemodelMixtureRule(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureRule(matdata),
      growth_strategy_matid_(matdata.parameters.get<int>("GROWTH_STRATEGY")),
      initial_reference_density_(matdata.parameters.get<double>("DENS")),
      mass_fractions_(matdata.parameters.get<std::vector<double>>("MASSFRAC"))
{
}

std::unique_ptr<Mixture::MixtureRule> Mixture::PAR::GrowthRemodelMixtureRule::create_rule()
{
  return std::unique_ptr<Mixture::GrowthRemodelMixtureRule>(
      new Mixture::GrowthRemodelMixtureRule(this));
}

Mixture::GrowthRemodelMixtureRule::GrowthRemodelMixtureRule(
    Mixture::PAR::GrowthRemodelMixtureRule* params)
    : MixtureRule(params), params_(params)
{
  if (params->growth_strategy_matid_ <= 0)
  {
    FOUR_C_THROW(
        "You have not specified a growth strategy material id. Reference to the material with the "
        "growth strategy.");
  }
  growth_strategy_ = Mixture::PAR::MixtureGrowthStrategy::factory(params->growth_strategy_matid_)
                         ->create_growth_strategy();
}

void Mixture::GrowthRemodelMixtureRule::pack_mixture_rule(
    Core::Communication::PackBuffer& data) const
{
  MixtureRule::pack_mixture_rule(data);

  growth_strategy_->pack_mixture_growth_strategy(data);
}

void Mixture::GrowthRemodelMixtureRule::unpack_mixture_rule(
    Core::Communication::UnpackBuffer& buffer)
{
  MixtureRule::unpack_mixture_rule(buffer);

  growth_strategy_->unpack_mixture_growth_strategy(buffer);
}

void Mixture::GrowthRemodelMixtureRule::register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
{
  growth_strategy_->register_anisotropy_extensions(anisotropy);
}

void Mixture::GrowthRemodelMixtureRule::setup(
    const Teuchos::ParameterList& params, const int eleGID)
{
  MixtureRule::setup(params, 0);
}

void Mixture::GrowthRemodelMixtureRule::update(Core::LinAlg::Tensor<double, 3, 3> const& F,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, const int gp,
    const int eleGID)
{
  // Update base mixture rule, which also updates the constituents.
  MixtureRule::update(F, params, context, gp, eleGID);


  // Evaluate inverse growth deformation gradient
  if (growth_strategy_->has_inelastic_growth_deformation_gradient())
  {
    FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
    const double dt = *context.time_step_size;

    // Evaluate inverse growth deformation gradient
    Core::LinAlg::Tensor<double, 3, 3> iFg;
    growth_strategy_->evaluate_inverse_growth_deformation_gradient(
        iFg, *this, compute_current_reference_growth_scalar(gp), context, gp, eleGID);

    for (const auto& constituent : constituents())
    {
      constituent->update_elastic_part(F, iFg, params, context, dt, gp, eleGID);
    }
  }
}

void Mixture::GrowthRemodelMixtureRule::evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  Core::LinAlg::Tensor<double, 3, 3> iF_gM;  // growth deformation gradient

  if (growth_strategy_->has_inelastic_growth_deformation_gradient())
  {
    // Evaluate growth kinematics
    const double currentReferenceGrowthScalar = compute_current_reference_growth_scalar(gp);

    growth_strategy_->evaluate_inverse_growth_deformation_gradient(
        iF_gM, *this, currentReferenceGrowthScalar, context, gp, eleGID);
  }

  // define temporary matrices
  Core::LinAlg::SymmetricTensor<double, 3, 3> cstress;
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> ccmat;

  // Iterate over all constituents and apply their contributions to the stress and linearization
  for (std::size_t i = 0; i < constituents().size(); ++i)
  {
    MixtureConstituent& constituent = *constituents()[i];
    cstress = {};
    ccmat = {};
    if (growth_strategy_->has_inelastic_growth_deformation_gradient())
      constituent.evaluate_elastic_part(F, iF_gM, params, context, cstress, ccmat, gp, eleGID);
    else
      constituent.evaluate(F, E_strain, params, context, cstress, ccmat, gp, eleGID);


    // Add stress contribution to global stress
    const double current_ref_constituent_density = params_->initial_reference_density_ *
                                                   params_->mass_fractions_[i] *
                                                   constituent.get_growth_scalar(gp);

    const Core::LinAlg::SymmetricTensor<double, 3, 3> dGrowthScalarDC =
        constituent.get_d_growth_scalar_d_cg(gp, eleGID);

    S_stress += current_ref_constituent_density * cstress;
    cmat += current_ref_constituent_density * ccmat;

    cmat += 2.0 * params_->initial_reference_density_ * params_->mass_fractions_[i] *
            Core::LinAlg::dyadic(cstress, dGrowthScalarDC);
  }

  cstress = {};
  ccmat = {};


  const auto [currentReferenceGrowthScalar, dCurrentReferenceGrowthScalarDC] = std::invoke(
      [&]()
      {
        double growthScalar = 0.0;
        Core::LinAlg::SymmetricTensor<double, 3, 3> dGrowthScalarDC{};

        for (std::size_t i = 0; i < constituents().size(); ++i)
        {
          MixtureConstituent& constituent = *constituents()[i];

          growthScalar += params_->mass_fractions_[i] * constituent.get_growth_scalar(gp);
          dGrowthScalarDC +=
              params_->mass_fractions_[i] * constituent.get_d_growth_scalar_d_cg(gp, eleGID);
        }

        return std::make_tuple(growthScalar, dGrowthScalarDC);
      });

  growth_strategy_->evaluate_growth_stress_cmat(*this, currentReferenceGrowthScalar,
      dCurrentReferenceGrowthScalarDC, F, E_strain, params, cstress, ccmat, gp, eleGID);

  S_stress += cstress;
  cmat += ccmat;
}

double Mixture::GrowthRemodelMixtureRule::compute_current_reference_growth_scalar(int gp) const
{
  double current_reference_growth_scalar = 0.0;
  for (std::size_t i = 0; i < constituents().size(); ++i)
  {
    MixtureConstituent& constituent = *constituents()[i];
    current_reference_growth_scalar +=
        params_->mass_fractions_[i] * constituent.get_growth_scalar(gp);
  }
  return current_reference_growth_scalar;
}

double Mixture::GrowthRemodelMixtureRule::get_constituent_initial_reference_mass_density(
    const MixtureConstituent& constituent) const
{
  for (std::size_t i = 0; i < constituents().size(); ++i)
  {
    MixtureConstituent& cur_constituent = *constituents()[i];
    if (cur_constituent.id() == constituent.id())
    {
      return params_->initial_reference_density_ * params_->mass_fractions_[i];
    }
  }
  FOUR_C_THROW("The constituent could not be found!");
  return 0.0;
}

void Mixture::GrowthRemodelMixtureRule::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size[OUTPUT_CURRENT_REFERENCE_DENSITY] = 1;
}

bool Mixture::GrowthRemodelMixtureRule::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == OUTPUT_CURRENT_REFERENCE_DENSITY)
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) =
          compute_current_reference_growth_scalar(gp) * params_->initial_reference_density_;
    }
    return true;
  }
  return false;
}
FOUR_C_NAMESPACE_CLOSE
