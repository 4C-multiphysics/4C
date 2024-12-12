// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_remodelfiber_impl.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mixture_constituent_remodelfiber_lib.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential_active.hpp"
#include "4C_mixture_growth_evolution_linear_cauchy_poisson_turnover.hpp"
#include "4C_utils_function_of_time.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>

FOUR_C_NAMESPACE_OPEN

// anonymous namespace for helper classes and functions
namespace
{
  [[nodiscard]] Core::LinAlg::Matrix<3, 3> evaluate_c(const Core::LinAlg::Matrix<3, 3>& F)
  {
    Core::LinAlg::Matrix<3, 3> C(false);
    C.multiply_tn(F, F);
    return C;
  }
}  // namespace

Mixture::PAR::MixtureConstituentRemodelFiberImpl::MixtureConstituentRemodelFiberImpl(
    const Core::Mat::PAR::Parameter::Data& matdata)
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
      deposition_stretch_timefunc_num_(matdata.parameters.get<int>("DEPOSITION_STRETCH_TIMEFUNCT"))
{
}

std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentRemodelFiberImpl::create_constituent(int id)
{
  return std::make_unique<Mixture::MixtureConstituentRemodelFiberImpl>(this, id);
}

Mixture::MixtureConstituentRemodelFiberImpl::MixtureConstituentRemodelFiberImpl(
    Mixture::PAR::MixtureConstituentRemodelFiberImpl* params, int id)
    : MixtureConstituent(params, id),
      params_(params),
      remodel_fiber_(),
      anisotropy_extension_(params_->init_, 0.0, false,
          std::make_shared<Mat::Elastic::StructuralTensorStrategyStandard>(nullptr),
          {params_->fiber_id_})
{
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR_STRESS |
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
}

Core::Materials::MaterialType Mixture::MixtureConstituentRemodelFiberImpl::material_type() const
{
  return Core::Materials::mix_remodelfiber_impl;
}

void Mixture::MixtureConstituentRemodelFiberImpl::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  Mixture::MixtureConstituent::pack_constituent(data);
  anisotropy_extension_.pack_anisotropy(data);

  for (const RemodelFiber<2>& fiber : remodel_fiber_) fiber.pack(data);
}

void Mixture::MixtureConstituentRemodelFiberImpl::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  Mixture::MixtureConstituent::unpack_constituent(buffer);
  initialize();

  anisotropy_extension_.unpack_anisotropy(buffer);
  for (RemodelFiber<2>& fiber : remodel_fiber_) fiber.unpack(buffer);
}

void Mixture::MixtureConstituentRemodelFiberImpl::register_anisotropy_extensions(
    Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mixture::MixtureConstituentRemodelFiberImpl::initialize()
{
  remodel_fiber_.clear();
  std::shared_ptr<const RemodelFiberMaterial<double>> material =
      params_->fiber_material_->create_remodel_fiber_material();

  for (int gp = 0; gp < num_gp(); ++gp)
  {
    LinearCauchyGrowthWithPoissonTurnoverGrowthEvolution<double> growth_evolution(
        params_->growth_constant_, params_->poisson_decay_time_,
        params_->enable_basal_mass_production_);
    remodel_fiber_.emplace_back(material, growth_evolution, evaluate_deposition_stretch(0.0));
  }
}

void Mixture::MixtureConstituentRemodelFiberImpl::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  Mixture::MixtureConstituent::read_element(numgp, container);
  initialize();
}

void Mixture::MixtureConstituentRemodelFiberImpl::setup(Teuchos::ParameterList& params, int eleGID)
{
  Mixture::MixtureConstituent::setup(params, eleGID);
  update_homeostatic_values(params, eleGID);
}

void Mixture::MixtureConstituentRemodelFiberImpl::update(const Core::LinAlg::Matrix<3, 3>& F,
    Teuchos::ParameterList& params, const int gp, const int eleGID)
{
  MixtureConstituent::update(F, params, gp, eleGID);

  // Update state
  remodel_fiber_[gp].update();

  update_homeostatic_values(params, eleGID);
}

void Mixture::MixtureConstituentRemodelFiberImpl::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  MixtureConstituent::register_output_data_names(names_and_size);
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig_h"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_sig"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_growth_scalar"] = 1;
  names_and_size["mixture_constituent_" + std::to_string(id()) + "_lambda_r"] = 1;
}

bool Mixture::MixtureConstituentRemodelFiberImpl::evaluate_output_data(
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

Core::LinAlg::Matrix<1, 6> Mixture::MixtureConstituentRemodelFiberImpl::evaluate_d_lambdafsq_dc(
    int gp, int eleGID) const
{
  Core::LinAlg::Matrix<1, 6> dLambdafDC(false);
  dLambdafDC.update_t(anisotropy_extension_.get_structural_tensor_stress(gp, 0));
  return dLambdafDC;
}

Core::LinAlg::Matrix<6, 1> Mixture::MixtureConstituentRemodelFiberImpl::evaluate_current_p_k2(
    int gp, int eleGID) const
{
  Core::LinAlg::Matrix<6, 1> S_stress(false);
  const double fiber_pk2 = remodel_fiber_[gp].evaluate_current_fiber_pk2_stress();

  S_stress.update(fiber_pk2, anisotropy_extension_.get_structural_tensor_stress(gp, 0));

  return S_stress;
}

Core::LinAlg::Matrix<6, 6> Mixture::MixtureConstituentRemodelFiberImpl::evaluate_current_cmat(
    const int gp, const int eleGID) const
{
  const double dPK2dlambdafsq =
      remodel_fiber_[gp].evaluate_d_current_fiber_pk2_stress_d_lambda_f_sq();

  Core::LinAlg::Matrix<6, 6> cmat(false);
  cmat.multiply_nt(2.0 * dPK2dlambdafsq, anisotropy_extension_.get_structural_tensor_stress(gp, 0),
      anisotropy_extension_.get_structural_tensor_stress(gp, 0));

  // additional linearization from implicit integration
  if (params_->enable_growth_)
  {
    Core::LinAlg::Matrix<1, 6> d_lambda_r_d_cauchy_green = evaluate_d_lambdafsq_dc(gp, eleGID);
    d_lambda_r_d_cauchy_green.scale(remodel_fiber_[gp].evaluate_d_current_lambda_r_d_lambda_f_sq());

    const double dpk2dlambdar = remodel_fiber_[gp].evaluate_d_current_fiber_pk2_stress_d_lambda_r();
    cmat.multiply_nn(2.0 * dpk2dlambdar, anisotropy_extension_.get_structural_tensor_stress(gp, 0),
        d_lambda_r_d_cauchy_green, 1.0);
  }

  return cmat;
}

void Mixture::MixtureConstituentRemodelFiberImpl::integrate_local_evolution_equations(
    const double dt, int gp, int eleGID)
{
  FOUR_C_ASSERT(params_->enable_growth_,
      "The integration of the local evolution equation should only be called if growth is "
      "enabled!");

  // Integrate local evolution equations
  Core::LinAlg::Matrix<2, 2> K =
      remodel_fiber_[gp].integrate_local_evolution_equations_implicit(dt);
}

void Mixture::MixtureConstituentRemodelFiberImpl::evaluate(const Core::LinAlg::Matrix<3, 3>& F,
    const Core::LinAlg::Matrix<6, 1>& E_strain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>& S_stress, Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  const double dt = params.get<double>("delta time");

  Core::LinAlg::Matrix<3, 3> C = evaluate_c(F);

  const double lambda_f = evaluate_lambdaf(C, gp, eleGID);
  remodel_fiber_[gp].set_state(lambda_f, 1.0);

  if (params_->enable_growth_) integrate_local_evolution_equations(dt, gp, eleGID);

  S_stress.update(evaluate_current_p_k2(gp, eleGID));
  cmat.update(evaluate_current_cmat(gp, eleGID));
}

void Mixture::MixtureConstituentRemodelFiberImpl::evaluate_elastic_part(
    const Core::LinAlg::Matrix<3, 3>& FM, const Core::LinAlg::Matrix<3, 3>& iFextin,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  FOUR_C_THROW(
      "The implicit remodel fiber cannot be evaluated with an additional inelastic deformation. "
      "You can either use the explicit remodel fiber or use a growth strategy without an inelastic "
      "external deformation.");
}

double Mixture::MixtureConstituentRemodelFiberImpl::get_growth_scalar(int gp) const
{
  return remodel_fiber_[gp].evaluate_current_growth_scalar();
}

Core::LinAlg::Matrix<1, 6> Mixture::MixtureConstituentRemodelFiberImpl::get_d_growth_scalar_d_cg(
    int gp, int eleGID) const
{
  if (!params_->enable_growth_) return Core::LinAlg::Matrix<1, 6>(true);
  Core::LinAlg::Matrix<1, 6> d_growth_scalar_d_cauchy_green = evaluate_d_lambdafsq_dc(gp, eleGID);
  d_growth_scalar_d_cauchy_green.scale(
      remodel_fiber_[gp].evaluate_d_current_growth_scalar_d_lambda_f_sq());
  return d_growth_scalar_d_cauchy_green;
}

double Mixture::MixtureConstituentRemodelFiberImpl::evaluate_deposition_stretch(
    const double time) const
{
  if (params_->deposition_stretch_timefunc_num_ == 0)
  {
    return params_->deposition_stretch_;
  }

  return Global::Problem::instance()
      ->function_by_id<Core::Utils::FunctionOfTime>(params_->deposition_stretch_timefunc_num_ - 1)
      .evaluate(time);
}
void Mixture::MixtureConstituentRemodelFiberImpl::update_homeostatic_values(
    const Teuchos::ParameterList& params, const int eleGID)
{
  // Update deposition stretch / prestretch of fiber depending on time function
  const double time = std::invoke(
      [&]()
      {
        constexpr auto total_time_key = "total time";
        if (!params.isParameter(total_time_key)) return 0.0;

        const double total_time = params.get<double>(total_time_key);
        if (total_time < 0.0) return 0.0;

        return total_time;
      });

  const double new_lambda_pre = evaluate_deposition_stretch(time);

  for (auto& fiber : remodel_fiber_)
  {
    fiber.update_deposition_stretch(new_lambda_pre);
  }
}

double Mixture::MixtureConstituentRemodelFiberImpl::evaluate_lambdaf(
    const Core::LinAlg::Matrix<3, 3>& C, const int gp, const int eleGID) const
{
  return std::sqrt(C.dot(anisotropy_extension_.get_structural_tensor(gp, 0)));
}
FOUR_C_NAMESPACE_CLOSE
