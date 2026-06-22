// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_remodelfiber_ssi.hpp"

#include "4C_global_data.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_mixture_constituent_remodelfiber_lib.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential_active.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_utils_function_of_time.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

FOUR_C_NAMESPACE_OPEN

// anonymous namespace for helper classes and functions
namespace
{
  [[nodiscard]] Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_cauchy_green(
      const Core::LinAlg::Tensor<double, 3, 3>& F)
  {
    return Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(F) * F);
  }

  [[nodiscard]] Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_inelastic_inv_cauchy_green(
      const Core::LinAlg::Tensor<double, 3, 3>& iFext)
  {
    return Core::LinAlg::assume_symmetry(iFext * Core::LinAlg::transpose(iFext));
  }
}  // namespace

Mixture::PAR::MixtureConstituentRemodelFiberSsi::MixtureConstituentRemodelFiberSsi(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituent(matdata),
      fiber_orientation(
          matdata.parameters.get<Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
              Mat::FiberInterpolation>>("ORIENTATION")),
      fiber_material_id_(matdata.parameters.get<int>("FIBER_MATERIAL_ID")),
      fiber_material_(fiber_material_factory(fiber_material_id_)),
      enable_growth_(matdata.parameters.get<bool>("ENABLE_GROWTH")),
      enable_basal_mass_production_(matdata.parameters.get<bool>("ENABLE_BASAL_MASS_PRODUCTION")),
      poisson_decay_time_(matdata.parameters.get<double>("DECAY_TIME")),
      growth_constant_(matdata.parameters.get<double>("GROWTH_CONSTANT")),
      deposition_stretch_(matdata.parameters.get<double>("DEPOSITION_STRETCH")),
      deposition_stretch_timefunc_num_(matdata.parameters.get<int>("DEPOSITION_STRETCH_TIMEFUNCT")),
      inelastic_external_deformation_(matdata.parameters.get<bool>("INELASTIC_GROWTH")),
      growth_scalar_id_(matdata.parameters.get<std::optional<int>>("GROWTH_SCALAR_ID")),
      remodeling_scalar_id_(matdata.parameters.get<std::optional<int>>("REMODELING_SCALAR_ID")),
      nonlocal_stimulus_scalar_id_(
          matdata.parameters.get<std::optional<int>>("NONLOCAL_STIMULUS_SCALAR_ID")),
      implicit_integration_(matdata.parameters.get<bool>("IMPLICIT_INTEGRATION"))
{
}

std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentRemodelFiberSsi::create_constituent(int id)
{
  return std::make_unique<Mixture::MixtureConstituentRemodelFiberSsi>(this, id);
}

Mixture::MixtureConstituentRemodelFiberSsi::MixtureConstituentRemodelFiberSsi(
    Mixture::PAR::MixtureConstituentRemodelFiberSsi* params, int id)
    : MixtureConstituent(params, id), params_(params), remodel_fiber_()
{
}

Core::Materials::MaterialType Mixture::MixtureConstituentRemodelFiberSsi::material_type() const
{
  return Core::Materials::mix_remodelfiber_ssi;
}

void Mixture::MixtureConstituentRemodelFiberSsi::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  Mixture::MixtureConstituent::pack_constituent(data);

  for (const RemodelFiber<2>& fiber : remodel_fiber_) fiber.pack(data);

  Core::Communication::add_to_pack(data, structural_tensors_);
}

void Mixture::MixtureConstituentRemodelFiberSsi::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  Mixture::MixtureConstituent::unpack_constituent(buffer);
  initialize();

  for (RemodelFiber<2>& fiber : remodel_fiber_) fiber.unpack(buffer);

  Core::Communication::extract_from_pack(buffer, structural_tensors_);
}

void Mixture::MixtureConstituentRemodelFiberSsi::initialize()
{
  std::shared_ptr<const RemodelFiberMaterial<double>> material =
      params_->fiber_material_->create_remodel_fiber_material();

  remodel_fiber_.clear();
  for (int gp = 0; gp < num_gp(); ++gp)
  {
    LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        params_->growth_constant_, params_->poisson_decay_time_,
        params_->enable_basal_mass_production_);
    remodel_fiber_.emplace_back(material, growth_evolution, evaluate_deposition_stretch(0.0));
  }
}

void Mixture::MixtureConstituentRemodelFiberSsi::read_element(int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  Mixture::MixtureConstituent::read_element(numgp, fibers, coord_system);
  initialize();
}

void Mixture::MixtureConstituentRemodelFiberSsi::setup(
    const Teuchos::ParameterList& params, int eleGID)
{
  Mixture::MixtureConstituent::setup(params, eleGID);
  update_homeostatic_values(params, 0.0, eleGID);

  // Validate scalar-ID configuration: exactly one of the two coupling modes must be active.
  if (is_nonlocal_stimulus_mode())
  {
    FOUR_C_ASSERT_ALWAYS(
        !params_->growth_scalar_id_.has_value() and !params_->remodeling_scalar_id_.has_value(),
        "MIX_Constituent_SsiRemodelFiber NONLOCAL_STIMULUS_SCALAR_ID is set, but "
        "GROWTH_SCALAR_ID and/or REMODELING_SCALAR_ID are also set. Specify exactly "
        "one coupling mode: either NONLOCAL_STIMULUS_SCALAR_ID or both GROWTH_SCALAR_ID and "
        "REMODELING_SCALAR_ID.");
  }
  else
  {
    FOUR_C_ASSERT_ALWAYS(
        params_->growth_scalar_id_.has_value() and params_->remodeling_scalar_id_.has_value(),
        "MIX_Constituent_SsiRemodelFiber requires either NONLOCAL_STIMULUS_SCALAR_ID or both "
        "GROWTH_SCALAR_ID and REMODELING_SCALAR_ID to be set.");
  }
}

void Mixture::MixtureConstituentRemodelFiberSsi::update_elastic_part(
    const Core::LinAlg::Tensor<double, 3, 3>& F, const Core::LinAlg::Tensor<double, 3, 3>& iFext,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext<3>& context, const double dt,
    const int gp, const int eleGID)
{
  MixtureConstituent::update_elastic_part(F, iFext, params, context, dt, gp, eleGID);

  if (!params_->inelastic_external_deformation_)
  {
    FOUR_C_THROW(
        "You specified that there is no inelastic external deformation in the input file, but this "
        "method is only called if there is one. Probably, you are using a mixture rule with "
        "inelastic growth. You have to set INELASTIC_GROWTH to true or use a different growth "
        "rule.");
  }
  const double lambda_f = evaluate_lambdaf(evaluate_cauchy_green(F), gp, eleGID);
  const double lambda_ext = evaluate_lambda_ext(iFext, gp, eleGID);
  remodel_fiber_[gp].set_state(lambda_f, lambda_ext);
  remodel_fiber_[gp].update();

  if (params_->enable_growth_)
  {
    if (is_nonlocal_stimulus_mode())
    {
      const auto& scalars = *params.get<std::shared_ptr<std::vector<double>>>("scalars");
      const double psi =
          scalars[static_cast<std::size_t>(params_->nonlocal_stimulus_scalar_id_.value())];
      if (!params_->implicit_integration_)
      {
        remodel_fiber_[gp].integrate_local_evolution_equations_explicit_with_nonlocal_stimulus(
            psi, dt);
      }
    }
    else
    {
      set_current_growth_scalar(params, gp);
      set_current_lambda_r(params, gp);
    }
  }
}

void Mixture::MixtureConstituentRemodelFiberSsi::update(const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext<3>& context, const int gp,
    const int eleGID)
{
  MixtureConstituent::update(F, params, context, gp, eleGID);

  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  update_homeostatic_values(params, *context.total_time, eleGID);

  if (!params_->inelastic_external_deformation_)
  {
    // Update state
    const double lambda_f = evaluate_lambdaf(evaluate_cauchy_green(F), gp, eleGID);
    remodel_fiber_[gp].set_state(lambda_f, 1.0);
    remodel_fiber_[gp].update();

    if (params_->enable_growth_)
    {
      if (is_nonlocal_stimulus_mode())
      {
        const double dt = *context.time_step_size;
        const auto& scalars = *params.get<std::shared_ptr<std::vector<double>>>("scalars");
        const double psi =
            scalars[static_cast<std::size_t>(params_->nonlocal_stimulus_scalar_id_.value())];
        if (!params_->implicit_integration_)
        {
          remodel_fiber_[gp].integrate_local_evolution_equations_explicit_with_nonlocal_stimulus(
              psi, dt);
        }
      }
      else
      {
        set_current_growth_scalar(params, gp);
        set_current_lambda_r(params, gp);
      }
    }
  }
}
void Mixture::MixtureConstituentRemodelFiberSsi::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  MixtureConstituent::register_output_data_names(names_and_size);
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig_h"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_growth_scalar"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_lambda_r"] = 1;
}

bool Mixture::MixtureConstituentRemodelFiberSsi::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "mixture_constituent_" + std::to_string(id()) + "_sig_h")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_homeostatic_fiber_cauchy_stress();
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_sig")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_fiber_cauchy_stress();
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_growth_scalar")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_growth_scalar();
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_lambda_r")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = remodel_fiber_[gp].evaluate_current_lambda_r();
    }
    return true;
  }
  return MixtureConstituent::evaluate_output_data(name, data);
}

Core::LinAlg::SymmetricTensor<double, 3, 3>
Mixture::MixtureConstituentRemodelFiberSsi::evaluate_current_pk2(int gp, int eleGID) const
{
  const double fiber_pk2 = remodel_fiber_[gp].evaluate_current_fiber_pk2_stress();

  return fiber_pk2 * structural_tensors_[gp];
}

Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>
Mixture::MixtureConstituentRemodelFiberSsi::evaluate_current_cmat(
    const int gp, const int eleGID) const
{
  const double dPK2dlambdafsq =
      remodel_fiber_[gp].evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq();

  return 2.0 * dPK2dlambdafsq *
         Core::LinAlg::dyadic(structural_tensors_[gp], structural_tensors_[gp]);
}

void Mixture::MixtureConstituentRemodelFiberSsi::evaluate(
    const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  if (params_->inelastic_external_deformation_)
  {
    FOUR_C_THROW(
        "You specified that there is inelastic external deformation in the input file, but this "
        "method is only called if there is none. Probably, you are using a mixture rule without "
        "inelastic growth. You have to set INELASTIC_GROWTH to false or use a different growth "
        "rule.");
  }

  if (static_cast<int>(structural_tensors_.size()) < gp + 1)
  {
    // once possible: Setup structural tensors in the setup phase
    FOUR_C_ASSERT(std::cmp_equal(structural_tensors_.size(), gp),
        "Expecting the Gauss points to be called in order!");

    Core::LinAlg::Tensor<double, 3> orientation =
        params_->fiber_orientation.interpolate(eleGID, context.xi->as_span());
    structural_tensors_.emplace_back(Core::LinAlg::self_dyadic(orientation));
  }

  Core::LinAlg::SymmetricTensor<double, 3, 3> C = evaluate_cauchy_green(F);
  const double lambda_f = evaluate_lambdaf(C, gp, eleGID);
  remodel_fiber_[gp].set_state(lambda_f, 1.0);

  if (params_->enable_growth_ && is_nonlocal_stimulus_mode() && params_->implicit_integration_)
  {
    FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
    const double dt = *context.time_step_size;
    const auto& scalars = *params.get<std::shared_ptr<std::vector<double>>>("scalars");
    const double psi =
        scalars[static_cast<std::size_t>(params_->nonlocal_stimulus_scalar_id_.value())];
    remodel_fiber_[gp].integrate_local_evolution_equations_implicit_with_nonlocal_stimulus(psi, dt);
  }

  S_stress = evaluate_current_pk2(gp, eleGID);
  cmat = evaluate_current_cmat(gp, eleGID);
}

void Mixture::MixtureConstituentRemodelFiberSsi::evaluate_elastic_part(
    const Core::LinAlg::Tensor<double, 3, 3>& FM, const Core::LinAlg::Tensor<double, 3, 3>& iFextin,
    const Teuchos::ParameterList& params, const Mat::EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  if (!params_->inelastic_external_deformation_)
  {
    FOUR_C_THROW(
        "You specified that there is no inelastic external deformation in the input file, but this "
        "method is only called if there is one. Probably, you are using a mixture rule with "
        "inelastic growth. You have to set INELASTIC_GROWTH to true or use a different growth "
        "rule.");
  }

  if (static_cast<int>(structural_tensors_.size()) < gp + 1)
  {
    // once possible: Setup structural tensors in the setup phase
    FOUR_C_ASSERT(std::cmp_equal(structural_tensors_.size(), gp),
        "Expecting the Gauss points to be called in order!");

    Core::LinAlg::Tensor<double, 3> orientation =
        params_->fiber_orientation.interpolate(eleGID, context.xi->as_span());
    structural_tensors_.emplace_back(Core::LinAlg::self_dyadic(orientation));
  }

  Core::LinAlg::SymmetricTensor<double, 3, 3> C = evaluate_cauchy_green(FM);

  const double lambda_f = evaluate_lambdaf(C, gp, eleGID);
  const double lambda_ext = evaluate_lambda_ext(iFextin, gp, eleGID);
  remodel_fiber_[gp].set_state(lambda_f, lambda_ext);

  if (params_->enable_growth_ && is_nonlocal_stimulus_mode() && params_->implicit_integration_)
  {
    FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
    const double dt = *context.time_step_size;
    const auto& scalars = *params.get<std::shared_ptr<std::vector<double>>>("scalars");
    const double psi =
        scalars[static_cast<std::size_t>(params_->nonlocal_stimulus_scalar_id_.value())];
    remodel_fiber_[gp].integrate_local_evolution_equations_implicit_with_nonlocal_stimulus(psi, dt);
  }

  S_stress = evaluate_current_pk2(gp, eleGID);
  cmat = evaluate_current_cmat(gp, eleGID);
}

double Mixture::MixtureConstituentRemodelFiberSsi::get_growth_scalar(int gp) const
{
  return remodel_fiber_[gp].evaluate_current_growth_scalar();
}

void Mixture::MixtureConstituentRemodelFiberSsi::set_current_growth_scalar(
    const Teuchos::ParameterList& params, int gp)
{
  const auto& scalars = *params.get<std::shared_ptr<std::vector<double>>>("scalars");
  double scalar = scalars[static_cast<std::size_t>(params_->growth_scalar_id_.value())];
  remodel_fiber_[gp].set_growth_scalar(scalar);
}
void Mixture::MixtureConstituentRemodelFiberSsi::set_current_lambda_r(
    const Teuchos::ParameterList& params, int gp)
{
  const auto& scalars = *params.get<std::shared_ptr<std::vector<double>>>("scalars");
  double scalar = scalars[static_cast<std::size_t>(params_->remodeling_scalar_id_.value())];
  remodel_fiber_[gp].set_lambda_r(scalar);
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_remodeling_reaction_coefficient(
    int gp) const
{
  if (params_->enable_growth_) return remodel_fiber_[gp].evaluate_remodeling_reaction_coefficient();
  return 0.0;
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_growth_reaction_coefficient(
    int gp) const
{
  if (params_->enable_growth_) return remodel_fiber_[gp].evaluate_growth_reaction_coefficient();
  return 0.0;
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_current_fiber_pk2_stress(int gp) const
{
  return remodel_fiber_[gp].evaluate_current_fiber_pk2_stress();
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_d_growth_scalar_d_nonlocal_stimulus(
    int gp) const
{
  return remodel_fiber_[gp].evaluate_d_growth_scalar_d_nonlocal_stimulus();
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_d_cauchy_stress_d_lambda_f_sq(
    int gp) const
{
  return remodel_fiber_[gp].evaluate_d_current_cauchy_stress_d_lambda_f_sq();
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_local_stimulus(int gp) const
{
  const double sigma = remodel_fiber_[gp].evaluate_current_fiber_cauchy_stress();
  const double sigma_h = remodel_fiber_[gp].evaluate_current_homeostatic_fiber_cauchy_stress();
  return sigma - sigma_h;
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_deposition_stretch(
    const double time) const
{
  if (params_->deposition_stretch_timefunc_num_ == 0)
  {
    return params_->deposition_stretch_;
  }

  return Global::Problem::instance()
      ->function_by_id<Core::Utils::FunctionOfTime>(params_->deposition_stretch_timefunc_num_)
      .evaluate(time);
}
void Mixture::MixtureConstituentRemodelFiberSsi::update_homeostatic_values(
    const Teuchos::ParameterList& params, const double total_time, const int eleGID)
{
  double new_lambda_pre = evaluate_deposition_stretch(total_time);

  for (auto& fiber : remodel_fiber_)
  {
    fiber.update_deposition_stretch(new_lambda_pre);
  }
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_lambdaf(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& C, const int gp, const int eleGID) const
{
  return std::sqrt(Core::LinAlg::ddot(C, structural_tensors_[gp]));
}

double Mixture::MixtureConstituentRemodelFiberSsi::evaluate_lambda_ext(
    const Core::LinAlg::Tensor<double, 3, 3>& iFext, const int gp, const int eleGID) const
{
  return 1.0 / std::sqrt(Core::LinAlg::ddot(
                   evaluate_inelastic_inv_cauchy_green(iFext), structural_tensors_[gp]));
}
FOUR_C_NAMESPACE_CLOSE
