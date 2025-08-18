// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_full_constrained_mixture_fiber.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_constituent_remodelfiber_lib.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_utils_function_of_time.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>

FOUR_C_NAMESPACE_OPEN

// anonymous namespace for helper classes and functions
namespace
{
  [[nodiscard]] static inline Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_c(
      const Core::LinAlg::Tensor<double, 3, 3>& F)
  {
    return Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(F) * F);
  }

  [[nodiscard]] static inline double get_total_time(const Teuchos::ParameterList& params)
  {
    double time = params.get<double>("total time");
    if (time < 0) return 0.0;  // Time has not been set by the time integrator during setup
    return time;
  }

  [[nodiscard]] static inline double get_delta_time(const Teuchos::ParameterList& params)
  {
    return params.get<double>("delta time");
  }

  Mixture::HistoryAdaptionStrategy get_history_adaption_strategy_from_input(
      const std::string& input)
  {
    if (input == "none")
    {
      return Mixture::HistoryAdaptionStrategy::none;
    }
    else if (input == "window")
    {
      return Mixture::HistoryAdaptionStrategy::window;
    }
    else if (input == "model_equation")
    {
      return Mixture::HistoryAdaptionStrategy::model_equation;
    }
    else if (input == "higher_order")
    {
      return Mixture::HistoryAdaptionStrategy::higher_order_integration;
    }
    else
    {
      FOUR_C_THROW("Unknown history adaption strategy {}!", input);
    }
  }
}  // namespace

Mixture::PAR::MixtureConstituentFullConstrainedMixtureFiber::
    MixtureConstituentFullConstrainedMixtureFiber(const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituent(matdata),
      fiber_id_(matdata.parameters.get<int>("FIBER_ID") - 1),
      init_(matdata.parameters.get<int>("INIT")),
      fiber_material_id_(matdata.parameters.get<int>("FIBER_MATERIAL_ID")),
      fiber_material_(fiber_material_factory(fiber_material_id_)),
      enable_growth_(matdata.parameters.get<bool>("ENABLE_GROWTH")),
      enable_basal_mass_production_(matdata.parameters.get<bool>("ENABLE_BASAL_MASS_PRODUCTION")),
      poisson_decay_time_(matdata.parameters.get<double>("DECAY_TIME")),
      growth_constant_(matdata.parameters.get<double>("GROWTH_CONSTANT")),
      deposition_stretch_(matdata.parameters.get<double>("DEPOSITION_STRETCH")),
      initial_deposition_stretch_timefunc_num_(
          matdata.parameters.get<int>("INITIAL_DEPOSITION_STRETCH_TIMEFUNCT")),
      adaptive_history_strategy_(get_history_adaption_strategy_from_input(
          matdata.parameters.get<std::string>("ADAPTIVE_HISTORY_STRATEGY"))),
      adaptive_history_tolerance_(matdata.parameters.get<double>("ADAPTIVE_HISTORY_TOLERANCE"))
{
}

std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentFullConstrainedMixtureFiber::create_constituent(int id)
{
  return std::make_unique<Mixture::MixtureConstituentFullConstrainedMixtureFiber>(this, id);
}

Mixture::MixtureConstituentFullConstrainedMixtureFiber::
    MixtureConstituentFullConstrainedMixtureFiber(
        Mixture::PAR::MixtureConstituentFullConstrainedMixtureFiber* params, int id)
    : MixtureConstituent(params, id),
      params_(params),
      full_constrained_mixture_fiber_(),
      anisotropy_extension_(params_->init_, 0.0, false,
          std::make_shared<Mat::Elastic::StructuralTensorStrategyStandard>(nullptr),
          {params_->fiber_id_})
{
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

Core::Materials::MaterialType
Mixture::MixtureConstituentFullConstrainedMixtureFiber::material_type() const
{
  return Core::Materials::mix_full_constrained_mixture_fiber;
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  Mixture::MixtureConstituent::pack_constituent(data);
  anisotropy_extension_.pack_anisotropy(data);

  for (const FullConstrainedMixtureFiber<double>& fiber : full_constrained_mixture_fiber_)
    fiber.pack(data);

  add_to_pack(data, last_lambda_f_);
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  Mixture::MixtureConstituent::unpack_constituent(buffer);
  initialize();

  anisotropy_extension_.unpack_anisotropy(buffer);

  for (FullConstrainedMixtureFiber<double>& fiber : full_constrained_mixture_fiber_)
    fiber.unpack(buffer);

  extract_from_pack(buffer, last_lambda_f_);

  if (params_->enable_growth_)
  {
    for (auto gp = 0; gp < num_gp(); ++gp)
    {
      full_constrained_mixture_fiber_[gp].reinitialize_history(
          last_lambda_f_[gp], full_constrained_mixture_fiber_[gp].get_last_time_in_history());
    }
  }
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::register_anisotropy_extensions(
    Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::initialize()
{
  full_constrained_mixture_fiber_.clear();
  std::shared_ptr<const RemodelFiberMaterial<double>> material =
      params_->fiber_material_->create_remodel_fiber_material();

  last_lambda_f_.resize(num_gp(), 1.0);

  for (int gp = 0; gp < num_gp(); ++gp)
  {
    LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        params_->growth_constant_, params_->poisson_decay_time_,
        params_->enable_basal_mass_production_);
    full_constrained_mixture_fiber_.emplace_back(material, growth_evolution,
        evaluate_initial_deposition_stretch(0.0), params_->adaptive_history_strategy_,
        params_->enable_growth_);
    full_constrained_mixture_fiber_[gp].adaptive_tolerance_ = params_->adaptive_history_tolerance_;
  }
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  Mixture::MixtureConstituent::read_element(numgp, container);
  initialize();
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::setup(
    const Teuchos::ParameterList& params, int eleGID)
{
  Mixture::MixtureConstituent::setup(params, eleGID);

  if (params_->enable_growth_)
  {
    for (auto& fiber : full_constrained_mixture_fiber_)
    {
      // No deformation at t=0
      fiber.reinitialize_history(1.0, 0.0);
    }
  }
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::update(
    const Core::LinAlg::Tensor<double, 3, 3>& F, const Teuchos::ParameterList& params, const int gp,
    const int eleGID)
{
  MixtureConstituent::update(F, params, gp, eleGID);

  const double time = get_total_time(params);
  full_constrained_mixture_fiber_[gp].set_deposition_stretch(
      evaluate_initial_deposition_stretch(time));
  last_lambda_f_[gp] = evaluate_lambdaf(evaluate_c(F), gp, eleGID);

  // Update state
  full_constrained_mixture_fiber_[gp].update();
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  MixtureConstituent::register_output_data_names(names_and_size);
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig_h"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_growth_scalar"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_history_size"] = 1;
}

bool Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "mixture_constituent_" + std::to_string(id()) + "_sig_h")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = full_constrained_mixture_fiber_[gp].sig_h_;
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_sig")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = full_constrained_mixture_fiber_[gp].computed_sigma_;
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_growth_scalar")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = full_constrained_mixture_fiber_[gp].computed_growth_scalar_;
    }
    return true;
  }
  else if (name == "mixture_constituent_" + std::to_string(id()) + "_history_size")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      data(gp, 0) = std::accumulate(full_constrained_mixture_fiber_[gp].history_.begin(),
          full_constrained_mixture_fiber_[gp].history_.end(), 0,
          [](std::size_t number, const DepositionHistoryInterval<double>& item)
          { return number + item.timesteps.size(); });
    }
    return true;
  }
  return MixtureConstituent::evaluate_output_data(name, data);
}

Core::LinAlg::SymmetricTensor<double, 3, 3>
Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate_d_lambdafsq_dc(
    int gp, int eleGID) const
{
  return anisotropy_extension_.get_structural_tensor(gp, 0);
}

Core::LinAlg::SymmetricTensor<double, 3, 3>
Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate_current_p_k2(
    int gp, int eleGID) const
{
  const double fiber_pk2 = full_constrained_mixture_fiber_[gp].evaluate_current_second_pk_stress();

  return fiber_pk2 * anisotropy_extension_.get_structural_tensor(gp, 0);
}

Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>
Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate_current_cmat(
    const int gp, const int eleGID) const
{
  const double dPK2dlambdafsq =
      full_constrained_mixture_fiber_[gp].evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq();

  return 2.0 * dPK2dlambdafsq *
         Core::LinAlg::dyadic(anisotropy_extension_.get_structural_tensor(gp, 0),
             evaluate_d_lambdafsq_dc(gp, eleGID));
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate(
    const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const double time = get_total_time(params);
  const double delta_time = get_delta_time(params);

  Core::LinAlg::SymmetricTensor<double, 3, 3> C = evaluate_c(F);

  const double lambda_f = evaluate_lambdaf(C, gp, eleGID);
  full_constrained_mixture_fiber_[gp].recompute_state(lambda_f, time, delta_time);

  S_stress = evaluate_current_p_k2(gp, eleGID);
  cmat = evaluate_current_cmat(gp, eleGID);
}

void Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate_elastic_part(
    const Core::LinAlg::Tensor<double, 3, 3>& FM, const Core::LinAlg::Tensor<double, 3, 3>& iFextin,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  FOUR_C_THROW(
      "The full constrained mixture fiber cannot be evaluated with an additional inelastic "
      "deformation.");
}

double Mixture::MixtureConstituentFullConstrainedMixtureFiber::get_growth_scalar(int gp) const
{
  return full_constrained_mixture_fiber_[gp].computed_growth_scalar_;
}

Core::LinAlg::SymmetricTensor<double, 3, 3>
Mixture::MixtureConstituentFullConstrainedMixtureFiber::get_d_growth_scalar_d_cg(
    int gp, int eleGID) const
{
  if (!params_->enable_growth_) return {};

  return 2.0 * full_constrained_mixture_fiber_[gp].computed_dgrowth_scalar_dlambda_f_sq_ *
         evaluate_d_lambdafsq_dc(gp, eleGID);
}

double Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate_initial_deposition_stretch(
    const double time) const
{
  if (params_->initial_deposition_stretch_timefunc_num_ == 0)
  {
    return params_->deposition_stretch_;
  }

  return Global::Problem::instance()
      ->function_by_id<Core::Utils::FunctionOfTime>(
          params_->initial_deposition_stretch_timefunc_num_)
      .evaluate(time);
}

double Mixture::MixtureConstituentFullConstrainedMixtureFiber::evaluate_lambdaf(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& C, const int gp, const int eleGID) const
{
  return std::sqrt(Core::LinAlg::ddot(C, anisotropy_extension_.get_structural_tensor(gp, 0)));
}
FOUR_C_NAMESPACE_CLOSE
