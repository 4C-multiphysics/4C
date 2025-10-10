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
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
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

  Mat::PAR::MuscleCombo::ActivationParameterVariant get_activation_params(
      const Core::Mat::PAR::Parameter::Data& matdata)
  {
    if (matdata.parameters.get_if<int>("ACTIVATION_FUNCTION_ID"))
    {
      auto actFunctId = matdata.parameters.get<int>("ACTIVATION_FUNCTION_ID");
      if (actFunctId <= 0) FOUR_C_THROW("Function id must be positive");
      return actFunctId;
    }
    else if (matdata.parameters.get_if<ActivationFieldType>("ACTIVATION_VALUES"))
    {
      return matdata.parameters.get<ActivationFieldType>("ACTIVATION_VALUES");
    }
    else
      return std::monostate{};
  }

  struct ActivationParamsVisitor
  {
    Mat::MuscleCombo::ActivationEvaluatorVariant operator()(const int function_id) const
    {
      return &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
          function_id);
    }

    Mat::MuscleCombo::ActivationEvaluatorVariant operator()(const ActivationFieldType& map) const
    {
      return &map;
    }

    Mat::MuscleCombo::ActivationEvaluatorVariant operator()(const std::monostate& /*unused*/) const
    {
      FOUR_C_THROW(
          "Error in ActivationParamsVisitor. You're calling it with the default-constructed "
          "state.");
    }
  };

  struct ActivationEvalVisitor
  {
    double operator()(const ActivationFieldType* map) const
    {
      // use one-based element ids in the json input file (corresponding to the ones in the input
      // file)
      return Mat::Utils::Muscle::evaluate_time_space_dependent_active_stress_by_map(
          Popt_, *map, t_tot_, eleGID_);
    }

    double operator()(const Core::Utils::FunctionOfSpaceTime*& function) const
    {
      return Mat::Utils::Muscle::evaluate_time_space_dependent_active_stress_by_funct(
          Popt_, *function, t_tot_, element_center_reference_coordinates_);
    }

    double operator()(const std::monostate& /*unused*/) const
    {
      FOUR_C_THROW(
          "Error in ActivationEvalVisitor. You're calling it with the default-constructed state.");
    }

    const double& Popt_;
    const double& t_tot_;
    const Core::LinAlg::Tensor<double, 3>& element_center_reference_coordinates_;
    const int& eleGID_;
  };
}  // namespace


Mat::PAR::MuscleCombo::MuscleCombo(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      alpha_(matdata.parameters.get<double>("ALPHA")),
      beta_(matdata.parameters.get<double>("BETA")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      kappa_(matdata.parameters.get<double>("KAPPA")),
      omega0_(matdata.parameters.get<double>("OMEGA0")),
      Popt_(matdata.parameters.get<double>("POPT")),
      lambdaMin_(matdata.parameters.get<double>("LAMBDAMIN")),
      lambdaOpt_(matdata.parameters.get<double>("LAMBDAOPT")),
      activationParams_(get_activation_params(matdata)),
      density_(matdata.parameters.get<double>("DENS"))
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

Mat::MuscleCombo::MuscleCombo()
    : params_(nullptr),
      anisotropy_(),
      anisotropy_extension_(true, 0.0, 0,
          std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
              new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {0}),
      activation_evaluator_(std::monostate{})
{
}

Mat::MuscleCombo::MuscleCombo(Mat::PAR::MuscleCombo* params)
    : params_(params),
      anisotropy_(),
      anisotropy_extension_(true, 0.0, 0,
          std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
              new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {0}),
      activation_evaluator_(std::monostate{})
{
  // register anisotropy extension to global anisotropy
  anisotropy_.register_anisotropy_extension(anisotropy_extension_);

  // initialize fiber directions and structural tensor
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::FIBER_VECTORS |
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);

  // cannot set activation_function here, because function manager did not yet read functions
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

  anisotropy_extension_.pack_anisotropy(data);
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
        activation_evaluator_ = std::visit(ActivationParamsVisitor(), params_->activationParams_);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mat::MuscleCombo::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Read anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(fibers, coord_system);

  activation_evaluator_ = std::visit(ActivationParamsVisitor(), params_->activationParams_);
}

void Mat::MuscleCombo::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, int const gp,
    const Teuchos::ParameterList& params, const EvaluationContext& context, int const eleGID)
{
}

void Mat::MuscleCombo::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);

  Core::LinAlg::Matrix<6, 1> Sc_stress(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> ccmat(Core::LinAlg::Initialization::zero);

  // get passive material parameters
  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double gamma = params_->gamma_;
  const double kappa = params_->kappa_;
  const double omega0 = params_->omega0_;

  // compute matrices
  // right Cauchy Green tensor C
  Core::LinAlg::SymmetricTensor<double, 3, 3> C = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(*defgrad) * *defgrad);  // matrix notation
  const Core::LinAlg::Matrix<3, 3> C_mat =
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(C));                    // matrix notation
  Core::LinAlg::Matrix<6, 1> Cv(Core::LinAlg::Initialization::uninitialized);  // Voigt notation
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(C_mat, Cv);                  // Cv

  // inverse right Cauchy Green tensor C^-1
  Core::LinAlg::Matrix<3, 3> invC(Core::LinAlg::Initialization::uninitialized);   // matrix notation
  invC.invert(C_mat);                                                             // invC = C^-1
  Core::LinAlg::Matrix<6, 1> invCv(Core::LinAlg::Initialization::uninitialized);  // Voigt notation
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(invC, invCv);                   // invCv

  // structural tensor M, i.e. dyadic product of fibre directions
  Core::LinAlg::SymmetricTensor<double, 3, 3> M =
      anisotropy_extension_.get_structural_tensor(gp, 0);
  const Core::LinAlg::Matrix<3, 3> M_mat = Core::LinAlg::make_matrix(Core::LinAlg::get_full(M));
  Core::LinAlg::Matrix<6, 1> Mv(Core::LinAlg::Initialization::uninitialized);  // Voigt notation
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(M_mat, Mv);                  // Mv
  // structural tensor L = omega0/3*Identity + omegap*M
  Core::LinAlg::Matrix<3, 3> L(M_mat);
  L.scale(1.0 - omega0);  // omegap*M
  for (unsigned i = 0; i < 3; ++i) L(i, i) += omega0 / 3.0;

  // product invC*L
  Core::LinAlg::Matrix<3, 3> invCL(Core::LinAlg::Initialization::uninitialized);
  invCL.multiply_nn(invC, L);

  // product invC*L*invC
  Core::LinAlg::Matrix<3, 3> invCLinvC(
      Core::LinAlg::Initialization::uninitialized);  // matrix notation
  invCLinvC.multiply_nn(invCL, invC);
  Core::LinAlg::Matrix<6, 1> invCLinvCv(
      Core::LinAlg::Initialization::uninitialized);  // Voigt notation
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(invCLinvC, invCLinvCv);

  // stretch in fibre direction lambdaM
  // lambdaM = sqrt(C:M) = sqrt(tr(C^T M)), see Holzapfel2000, p.14
  double lambdaM = Mat::Utils::Muscle::fiber_stretch(C, M);

  // computation of active nominal stress Pa, and derivative derivPa
  double intPa = 0.0;
  double Pa = 0.0;
  double derivPa = 0.0;
  if (params_->Popt_ != 0)
  {  // if active material
    evaluate_active_nominal_stress(params, context, eleGID, lambdaM, intPa, Pa, derivPa);
  }  // else: intPa, Pa and derivPa remain 0.0

  // computation of activation level omegaa and derivative \frac{\partial omegaa}{\partial C}
  double omegaa = 0.0;
  double derivOmegaa = 0.0;
  double derivDerivOmegaa = 0.0;

  // prefactor eta and its first derivative w.r.t. lambdaM
  double eta = 0;
  double dEta = 0;

  // compute activation level and derivative only if active nominal stress is not zero
  if (Pa >= 1E-12)
  {
    evaluate_activation_level(lambdaM, intPa, Pa, derivPa, omegaa, derivOmegaa, derivDerivOmegaa);

    // passive part of invariant I and its first and second derivatives w.r.t. lambdaM
    double Ip =
        (omega0 / 3) * (std::pow(lambdaM, 2) + 2 / lambdaM) + (1 - omega0) * std::pow(lambdaM, 2);
    double dIp =
        -2 * lambdaM * (omega0 - 1) + (omega0 / 3) * (2 * lambdaM - 2 / std::pow(lambdaM, 2));

    // invariant I and its first derivative w.r.t. lambdaM
    double I = Ip + omegaa * std::pow(lambdaM, 2);
    double dI = dIp + 2 * omegaa * lambdaM + 2 * std::pow(lambdaM, 2) * derivOmegaa;

    // prefactor eta and its first derivative w.r.t. lambdaM
    eta = exp(alpha * (I - 1)) * lambdaM * derivOmegaa;
    dEta = exp(alpha * (I - 1)) * lambdaM *
           (derivOmegaa * (1 / lambdaM + alpha * dI) + derivDerivOmegaa);
  }

  // compute derivative \frac{\partial omegaa}{\partial C} in Voigt notation
  Core::LinAlg::Matrix<6, 1> domegaadCv(Mv);
  domegaadCv.scale(derivOmegaa * 0.5 / lambdaM);

  // compute helper matrices for further calculation
  Core::LinAlg::Matrix<3, 3> LomegaaM(L);
  LomegaaM.update(omegaa, M_mat, 1.0);  // LomegaaM = L + omegaa*M
  Core::LinAlg::Matrix<6, 1> LomegaaMv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(LomegaaM, LomegaaMv);

  Core::LinAlg::Matrix<3, 3> LfacomegaaM(L);  // LfacomegaaM = L + fac*M
  LfacomegaaM.update((1.0 + omegaa * alpha * std::pow(lambdaM, 2)) / (alpha * std::pow(lambdaM, 2)),
      M_mat, 1.0);  // + fac*M
  Core::LinAlg::Matrix<6, 1> LfacomegaaMv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(LfacomegaaM, LfacomegaaMv);

  Core::LinAlg::Matrix<3, 3> transpCLomegaaM(Core::LinAlg::Initialization::uninitialized);
  transpCLomegaaM.multiply_tn(1.0, C_mat, LomegaaM);  // C^T*(L+omegaa*M)
  Core::LinAlg::Matrix<6, 1> transpCLomegaaMv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(transpCLomegaaM, transpCLomegaaMv);

  // generalized invariants including active material properties
  double detC = C_mat.determinant();  // detC = det(C)
  // I = C:(L+omegaa*M) = tr(C^T (L+omegaa*M)) since A:B = tr(A^T B) for real matrices
  double I = transpCLomegaaM(0, 0) + transpCLomegaaM(1, 1) + transpCLomegaaM(2, 2);
  // J = cof(C):L = tr(cof(C)^T L) = tr(adj(C) L) = tr(det(C) C^-1 L) = det(C)*tr(C^-1 L)
  double J = detC * (invCL(0, 0) + invCL(1, 1) + invCL(2, 2));
  // exponential prefactors
  double expalpha = std::exp(alpha * (I - 1.0));
  double expbeta = std::exp(beta * (J - 1.0));

  // compute second Piola-Kirchhoff stress
  Core::LinAlg::Matrix<3, 3> stressM(Core::LinAlg::Initialization::uninitialized);
  stressM.update(expalpha, LomegaaM, 0.0);  // add contributions
  stressM.update(-expbeta * detC, invCLinvC, 1.0);
  stressM.update(J * expbeta - std::pow(detC, -kappa), invC, 1.0);
  stressM.update(0.5 * eta, M_mat, 1.0);
  stressM.scale(0.5 * gamma);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(
      stressM, Sc_stress);  // convert to Voigt notation and update stress

  // compute cmat
  ccmat.multiply_nt(alpha * expalpha, LomegaaMv, LomegaaMv, 1.0);  // add contributions
  ccmat.multiply_nt(alpha * std::pow(lambdaM, 2) * expalpha, LfacomegaaMv, domegaadCv, 1.0);
  ccmat.multiply_nt(beta * expbeta * std::pow(detC, 2.), invCLinvCv, invCLinvCv, 1.0);
  ccmat.multiply_nt(-(beta * J + 1.) * expbeta * detC, invCv, invCLinvCv, 1.0);
  ccmat.multiply_nt(-(beta * J + 1.) * expbeta * detC, invCLinvCv, invCv, 1.0);
  ccmat.multiply_nt(
      (beta * J + 1.) * J * expbeta + kappa * std::pow(detC, -kappa), invCv, invCv, 1.0);
  // adds scalar * (invC boeppel invC) to cmat, see Holzapfel2000, p. 254
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(
      ccmat, invCv, -(J * expbeta - std::pow(detC, -kappa)));
  // adds -expbeta*detC * dinvCLinvCdCv to cmats
  Core::LinAlg::FourTensorOperations::add_derivative_of_inva_b_inva_product(
      -expbeta * detC, invCv, invCLinvCv, ccmat);
  // additional term for corrected derivative of strain energy function
  ccmat.multiply_nt(dEta / (8 * lambdaM), Mv, Mv, 1.0);
  ccmat.scale(gamma);

  // update stress and material tangent with the computed stress and cmat values
  Core::LinAlg::make_stress_like_voigt_view(stress).update(1.0, Sc_stress, 1.0);
  Core::LinAlg::make_stress_like_voigt_view(cmat).update(1.0, ccmat, 1.0);
}

void Mat::MuscleCombo::evaluate_active_nominal_stress(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int eleGID, const double lambdaM, double& intPa,
    double& Pa, double& derivPa)
{
  // save current simulation time
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  double t_tot = *context.total_time;

  // get active parameters from params_
  const double lambdaMin = params_->lambdaMin_;
  const double lambdaOpt = params_->lambdaOpt_;

  const double Popt = params_->Popt_;

  // get element center coordinates in reference configuration
  const auto& element_center_reference_coordinates =
      params.get<Core::LinAlg::Tensor<double, 3>>("elecenter_coords_ref");

  // compute force-time-space dependency Poptft
  const double Poptft =
      std::visit(ActivationEvalVisitor{Popt, t_tot, element_center_reference_coordinates, eleGID},
          activation_evaluator_);

  // compute the force-stretch dependency fxi, its integral in the boundaries lambdaMin to lambdaM,
  // and its derivative w.r.t. lambdaM
  double intFxi = Mat::Utils::Muscle::evaluate_integral_force_stretch_dependency_ehret(
      lambdaM, lambdaMin, lambdaOpt);
  double fxi =
      Mat::Utils::Muscle::evaluate_force_stretch_dependency_ehret(lambdaM, lambdaMin, lambdaOpt);
  double dFxidLambdaM = Mat::Utils::Muscle::evaluate_derivative_force_stretch_dependency_ehret(
      lambdaM, lambdaMin, lambdaOpt);

  // compute active nominal stress Pa, its integral in the boundaries lambdaMin to lambdaM,
  // and its derivative w.r.t. lambdaM
  intPa = Poptft * intFxi;
  Pa = Poptft * fxi;
  derivPa = Poptft * dFxidLambdaM;
}

void Mat::MuscleCombo::evaluate_activation_level(const double lambdaM, const double intPa,
    const double Pa, const double derivPa, double& omegaa, double& derivOmegaa,
    double& derivDerivOmegaa)
{
  // get passive material parameters
  const double alpha = params_->alpha_;
  const double gamma = params_->gamma_;
  const double omega0 = params_->omega0_;

  // passive part of invariant I and its first and second derivatives w.r.t. lambdaM
  double Ip = (omega0 / 3.0) * (std::pow(lambdaM, 2) + 2.0 / lambdaM) +
              (1.0 - omega0) * std::pow(lambdaM, 2);
  double dIp = (omega0 / 3.0) * (2.0 * lambdaM - 2.0 / std::pow(lambdaM, 2)) +
               2.0 * (1.0 - omega0) * lambdaM;
  double ddIp = (omega0 / 3.0) * (2.0 + 4.0 / std::pow(lambdaM, 3)) + 2.0 * (1.0 - omega0);

  // helper tau and its first and second derivatives w.r.t. lambdaM
  double tau = alpha * (1 - Ip);
  double dTau = -alpha * dIp;
  double ddTau = -alpha * ddIp;

  // helper phi and its first and second derivatives w.r.t. lambdaM
  double phi = 1 + ((4 * alpha) / gamma) * intPa * std::exp(tau);
  double dPhi = (4 * alpha / gamma) * (intPa * std::exp(tau) * dTau + std::exp(tau) * Pa);
  double ddPhi = (4 * alpha / gamma) * std::exp(tau) *
                 (2 * Pa * dTau + intPa * std::pow(dTau, 2) + intPa * ddTau + derivPa);

  // computation of activation level omegaa
  omegaa = std::log(phi) / (alpha * std::pow(lambdaM, 2));

  // computation of partial derivative of omegaa w.r.t. lambdaM
  derivOmegaa = -2 * std::log(phi) / (alpha * std::pow(lambdaM, 3)) +
                dPhi / (phi * alpha * std::pow(lambdaM, 2));
  derivDerivOmegaa = 6 * std::log(phi) / (alpha * std::pow(lambdaM, 4)) -
                     4 * dPhi / (phi * alpha * std::pow(lambdaM, 3)) -
                     std::pow(dPhi, 2) / (phi * phi * alpha * std::pow(lambdaM, 2)) +
                     ddPhi / (phi * alpha * std::pow(lambdaM, 2));
}
FOUR_C_NAMESPACE_CLOSE
