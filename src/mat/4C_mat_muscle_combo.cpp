// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_muscle_combo.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_field.hpp"
#include "4C_linalg_fixedsizematrix_tensor_derivatives.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mat_muscle_utils.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using ActivationFieldType = Core::IO::InputField<std::vector<std::pair<double, double>>>;

  std::optional<Mat::PAR::MuscleCombo::ActiveParams> parse_active(
      const Core::Mat::PAR::Parameter::Data& matdata)
  {
    if (!matdata.parameters.has_group("ACTIVE")) return std::nullopt;

    const auto& g = matdata.parameters.group("ACTIVE");

    Mat::PAR::MuscleCombo::ActiveParams a = {.Popt = g.get<double>("POPT"),
        .lambdaMin = g.get<double>("LAMBDAMIN"),
        .lambdaOpt = g.get<double>("LAMBDAOPT")};

    // resolve the activation prescription
    if (g.get_if<int>("ACTIVATION_FUNCTION_ID"))
    {
      a.source = Mat::PAR::MuscleCombo::FunctionActivation{
          .function_id = g.get<int>("ACTIVATION_FUNCTION_ID")};
    }
    else if (g.get_if<ActivationFieldType>("ACTIVATION_VALUES"))
    {
      a.source = Mat::PAR::MuscleCombo::FieldActivation{
          .field = g.get<ActivationFieldType>("ACTIVATION_VALUES")};
    }
    else
    {
      a.source = std::monostate{};
    }

    return a;
  }

  Mat::PAR::MuscleCombo::PassiveParams parse_passive(const Core::Mat::PAR::Parameter::Data& matdata)
  {
    const auto& g = matdata.parameters.group("PASSIVE");

    return Mat::PAR::MuscleCombo::PassiveParams{.alpha = g.get<double>("ALPHA"),
        .beta = g.get<double>("BETA"),
        .gamma = g.get<double>("GAMMA"),
        .kappa = g.get<double>("KAPPA"),
        .omega0 = g.get<double>("OMEGA0")};
  }

  struct ActivationParamsVisitor
  {
    Mat::MuscleCombo::ActivationEvaluator operator()(
        const Mat::PAR::MuscleCombo::FunctionActivation& fa) const
    {
      return &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
          fa.function_id);
    }

    Mat::MuscleCombo::ActivationEvaluator operator()(
        const Mat::PAR::MuscleCombo::FieldActivation& fa) const
    {
      return &fa.field;
    }

    Mat::MuscleCombo::ActivationEvaluator operator()(const std::monostate& /*unused*/) const
    {
      FOUR_C_THROW(
          "Error in ActivationParamsVisitor. You're calling it with the default-constructed "
          "state.");
    }
  };

  struct ActivationEvalVisitor
  {
    double operator()(const ActivationFieldType* field) const
    {
      // use one-based element ids in the json input file (corresponding to the ones in the input
      // file)
      return Mat::Utils::Muscle::evaluate_time_space_dependent_activation_by_field(
          *field, t_tot_, eleGID_);
    }

    double operator()(const Core::Utils::FunctionOfSpaceTime*& function) const
    {
      return Mat::Utils::Muscle::evaluate_time_space_dependent_activation_by_funct(
          *function, t_tot_, element_center_reference_coordinates_);
    }

    double operator()(const std::monostate& /*unused*/) const
    {
      FOUR_C_THROW(
          "Error in ActivationEvalVisitor. You're calling it with the default-constructed state.");
    }

    const double& t_tot_;
    const Core::LinAlg::Tensor<double, 3>& element_center_reference_coordinates_;
    const int& eleGID_;
  };
}  // namespace


Mat::PAR::MuscleCombo::MuscleCombo(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      passive_(parse_passive(matdata)),
      active_(parse_active(matdata)),
      density_(matdata.parameters.get<double>("DENS")),
      fiber_orientation_(
          matdata.parameters.get<Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
              Mat::FiberInterpolation>>("FIBER_ORIENTATION"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::MuscleCombo::create_material()
{
  return std::make_shared<Mat::MuscleCombo>(this);
}

Mat::MuscleComboType Mat::MuscleComboType::instance_;

Core::Communication::ParObject* Mat::MuscleComboType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* muscle_combo = new Mat::MuscleCombo();
  muscle_combo->unpack(buffer);
  return muscle_combo;
}

Mat::MuscleCombo::MuscleCombo() : params_(nullptr), activation_evaluator_(std::monostate{}) {}

Mat::MuscleCombo::MuscleCombo(Mat::PAR::MuscleCombo* params)
    : params_(params), activation_evaluator_(std::monostate{})
{
}

void Mat::MuscleCombo::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

void Mat::MuscleCombo::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // make sure we have a pristine material
  params_ = nullptr;

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);

  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = static_cast<Mat::PAR::MuscleCombo*>(mat);
        if (params_->active_)
          activation_evaluator_ = std::visit(ActivationParamsVisitor(), params_->active_->source);
        else
          activation_evaluator_ = std::monostate{};
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }
}

void Mat::MuscleCombo::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  if (params_->active_)
    activation_evaluator_ = std::visit(ActivationParamsVisitor(), params_->active_->source);
}

void Mat::MuscleCombo::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, int const gp,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int const eleGID)
{
}

void Mat::MuscleCombo::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // get passive material parameters
  const auto passive_params = params_->passive_;
  const double alpha = passive_params.alpha;
  const double beta = passive_params.beta;
  const double gamma = passive_params.gamma;
  const double kappa = passive_params.kappa;
  const double omega0 = passive_params.omega0;

  // compute matrices
  // right Cauchy Green tensor C = F^{T} F
  const Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(*defgrad) * *defgrad);

  // inverse right Cauchy Green tensor C^-1
  const Core::LinAlg::SymmetricTensor<double, 3, 3> invC = Core::LinAlg::inv(C);

  // interpolate fiber orientation at current integration point
  Core::LinAlg::Tensor<double, 3> orientation =
      params_->fiber_orientation_.interpolate(eleGID, context.xi->as_span());

  // structural tensor M, i.e. dyadic product of fibre directions
  const Core::LinAlg::SymmetricTensor<double, 3, 3> M = Core::LinAlg::self_dyadic(orientation);

  // structural tensor L = omega0/3*Identity + omegap*M
  const Core::LinAlg::SymmetricTensor<double, 3, 3> L =
      omega0 / 3 * Core::LinAlg::TensorGenerators::identity<double, 3, 3> + (1 - omega0) * M;

  // product invC*L*invC
  const Core::LinAlg::SymmetricTensor<double, 3, 3> invCLinvC =
      Core::LinAlg::assume_symmetry(invC * L * invC);

  // stretch in fibre direction lambdaM
  // lambdaM = sqrt(C:M) = sqrt(tr(C^T M)), see Holzapfel2000, 14
  const double lambdaM = Mat::Utils::Muscle::fiber_stretch(C, M);

  // compute the activation terms
  const auto active_terms = evaluate_activation_level(params, context, eleGID, lambdaM);

  // compute derivative \frac{\partial omegaa}{\partial C}
  const Core::LinAlg::SymmetricTensor<double, 3, 3> domegaadC =
      active_terms.derivOmegaa / (2 * lambdaM) * M;

  // compute helper quantities for further calculation
  const Core::LinAlg::SymmetricTensor<double, 3, 3> LomegaaM = L + active_terms.omegaa * M;
  const double fac =
      (1.0 + active_terms.omegaa * alpha * std::pow(lambdaM, 2)) / (alpha * std::pow(lambdaM, 2));
  const Core::LinAlg::SymmetricTensor<double, 3, 3> LfacomegaaM = L + fac * M;
  const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dyadic_M = Core::LinAlg::dyadic(M, M);
  const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dyadic_invC =
      Core::LinAlg::dyadic(invC, invC);

  // generalized invariants including active material properties
  const double detC = Core::LinAlg::det(C);  // detC = det(C)
  // I = C:(L+omegaa*M) = tr(C^T (L+omegaa*M)) since A:B = tr(A^T B) for real matrices
  const double I = Core::LinAlg::ddot(C, LomegaaM);
  // J = cof(C):L = tr(cof(C)^T L) = tr(adj(C) L) = tr(det(C) C^-1 L) = det(C)*tr(C^-1 L)
  const double J = detC * Core::LinAlg::trace(invC * L);

  // exponential prefactors
  const double expalpha = std::exp(alpha * (I - 1.0));
  const double expbeta = std::exp(beta * (J - 1.0));

  // compute second Piola-Kirchhoff stress
  stress = 0.5 * gamma *
           (expalpha * LomegaaM + expbeta * (J * invC - detC * invCLinvC) -
               std::pow(detC, -kappa) * invC + 0.5 * active_terms.eta * M);

  // compute material tangent
  cmat = alpha * expalpha *
         (Core::LinAlg::dyadic(LomegaaM, LomegaaM) +
             std::pow(lambdaM, 2) * Core::LinAlg::dyadic(LfacomegaaM, domegaadC));
  cmat += beta * expbeta * std::pow(detC, 2.) * Core::LinAlg::dyadic(invCLinvC, invCLinvC);
  cmat -= (beta * J + 1.) * expbeta * detC *
          (Core::LinAlg::dyadic(invC, invCLinvC) + Core::LinAlg::dyadic(invCLinvC, invC));
  cmat += ((beta * J + 1.) * J * expbeta) * dyadic_invC;
  cmat += kappa * std::pow(detC, -kappa) * dyadic_invC;
  cmat -= (J * expbeta - std::pow(detC, -kappa)) *
          Core::LinAlg::FourTensorOperations::holzapfel_product(invC);
  cmat -= expbeta * detC *
          Core::LinAlg::FourTensorOperations::derivative_of_inva_b_inva_product(invC, invCLinvC);
  cmat += active_terms.dEta / (8 * lambdaM) * dyadic_M;
  cmat *= gamma;
}

Mat::MuscleCombo::ActivationTerms Mat::MuscleCombo::evaluate_activation_level(
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, const int eleGID,
    const double lambdaM)
{
  // initialize activation level omegaa, prefactor eta and derivatives w.r.t. lambdaM
  ActivationTerms terms{
      .omegaa = 0.0, .derivOmegaa = 0.0, .derivDerivOmegaa = 0.0, .eta = 0.0, .dEta = 0.0};

  // computation of active nominal stress Pa, and derivative derivPa
  if (params_->active_.has_value())
  {
    const auto& active_params = params_->active_.value();

    // save current simulation time
    FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
    double t_tot = *context.total_time;

    // get element center coordinates in reference configuration
    const auto& element_center_reference_coordinates =
        params.get<Core::LinAlg::Tensor<double, 3>>("elecenter_coords_ref");

    // compute force-time-space dependency
    const double ft = std::visit(
        ActivationEvalVisitor{.t_tot_ = t_tot,
            .element_center_reference_coordinates_ = element_center_reference_coordinates,
            .eleGID_ = eleGID},
        activation_evaluator_);

    // compute the force-stretch dependency fxi, its integral in the boundaries lambdaMin to
    // lambdaM, and its derivative w.r.t. lambdaM
    const double intFxi = Mat::Utils::Muscle::evaluate_integral_force_stretch_dependency_ehret(
        lambdaM, active_params.lambdaMin, active_params.lambdaOpt);
    const double fxi = Mat::Utils::Muscle::evaluate_force_stretch_dependency_ehret(
        lambdaM, active_params.lambdaMin, active_params.lambdaOpt);
    const double dFxidLambdaM =
        Mat::Utils::Muscle::evaluate_derivative_force_stretch_dependency_ehret(
            lambdaM, active_params.lambdaMin, active_params.lambdaOpt);

    // compute active nominal stress Pa, its integral in the boundaries lambdaMin to lambdaM,
    // and its derivative w.r.t. lambdaM
    const double Poptft = active_params.Popt * ft;
    const double intPa = Poptft * intFxi;
    const double Pa = Poptft * fxi;
    const double derivPa = Poptft * dFxidLambdaM;

    // get passive material parameters
    const auto& passive_params = params_->passive_;
    const double alpha = passive_params.alpha;
    const double gamma = passive_params.gamma;
    const double omega0 = passive_params.omega0;

    // passive part of invariant I and its first and second derivatives w.r.t. lambdaM
    const double Ip = (omega0 / 3.0) * (std::pow(lambdaM, 2) + 2.0 / lambdaM) +
                      (1.0 - omega0) * std::pow(lambdaM, 2);
    const double dIp = (omega0 / 3.0) * (2.0 * lambdaM - 2.0 / std::pow(lambdaM, 2)) +
                       2.0 * (1.0 - omega0) * lambdaM;
    const double ddIp = (omega0 / 3.0) * (2.0 + 4.0 / std::pow(lambdaM, 3)) + 2.0 * (1.0 - omega0);

    // helper tau and its first and second derivatives w.r.t. lambdaM
    const double tau = alpha * (1 - Ip);
    const double dTau = -alpha * dIp;
    const double ddTau = -alpha * ddIp;

    // helper phi and its first and second derivatives w.r.t. lambdaM
    const double phi = 1 + ((4 * alpha) / gamma) * intPa * std::exp(tau);
    const double dPhi = (4 * alpha / gamma) * (intPa * std::exp(tau) * dTau + std::exp(tau) * Pa);
    const double ddPhi = (4 * alpha / gamma) * std::exp(tau) *
                         (2 * Pa * dTau + intPa * std::pow(dTau, 2) + intPa * ddTau + derivPa);

    // computation of activation level omegaa
    terms.omegaa = std::log(phi) / (alpha * std::pow(lambdaM, 2));

    // computation of partial derivatives of omegaa w.r.t. lambdaM
    terms.derivOmegaa = -2 * std::log(phi) / (alpha * std::pow(lambdaM, 3)) +
                        dPhi / (phi * alpha * std::pow(lambdaM, 2));
    terms.derivDerivOmegaa = 6 * std::log(phi) / (alpha * std::pow(lambdaM, 4)) -
                             4 * dPhi / (phi * alpha * std::pow(lambdaM, 3)) -
                             std::pow(dPhi, 2) / (phi * phi * alpha * std::pow(lambdaM, 2)) +
                             ddPhi / (phi * alpha * std::pow(lambdaM, 2));

    // invariant I and its first derivative w.r.t. lambdaM
    const double I = Ip + terms.omegaa * std::pow(lambdaM, 2);
    const double dI =
        dIp + 2 * terms.omegaa * lambdaM + 2 * std::pow(lambdaM, 2) * terms.derivOmegaa;

    // prefactor eta and its first derivative w.r.t. lambdaM
    terms.eta = exp(alpha * (I - 1)) * lambdaM * terms.derivOmegaa;
    terms.dEta = exp(alpha * (I - 1)) * lambdaM *
                 (terms.derivOmegaa * (1 / lambdaM + alpha * dI) + terms.derivDerivOmegaa);
  }

  return terms;
}

FOUR_C_NAMESPACE_CLOSE
