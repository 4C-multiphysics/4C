// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_muscle_weickenmeier.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_derivatives.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_muscle_utils.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::PAR::MuscleWeickenmeier::MuscleWeickenmeier(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      alpha_(matdata.parameters.get<double>("ALPHA")),
      beta_(matdata.parameters.get<double>("BETA")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      kappa_(matdata.parameters.get<double>("KAPPA")),
      omega0_(matdata.parameters.get<double>("OMEGA0")),
      Na_(matdata.parameters.get<double>("ACTMUNUM")),
      muTypesNum_(matdata.parameters.get<int>("MUTYPESNUM")),
      I_((matdata.parameters.get<std::vector<double>>("INTERSTIM"))),
      rho_((matdata.parameters.get<std::vector<double>>("FRACACTMU"))),
      F_((matdata.parameters.get<std::vector<double>>("FTWITCH"))),
      T_((matdata.parameters.get<std::vector<double>>("TTWITCH"))),
      lambdaMin_(matdata.parameters.get<double>("LAMBDAMIN")),
      lambdaOpt_(matdata.parameters.get<double>("LAMBDAOPT")),
      dotLambdaMMin_(matdata.parameters.get<double>("DOTLAMBDAMIN")),
      ke_(matdata.parameters.get<double>("KE")),
      kc_(matdata.parameters.get<double>("KC")),
      de_(matdata.parameters.get<double>("DE")),
      dc_(matdata.parameters.get<double>("DC")),
      actTimesNum_(matdata.parameters.get<int>("ACTTIMESNUM")),
      actTimes_((matdata.parameters.get<std::vector<double>>("ACTTIMES"))),
      actIntervalsNum_(matdata.parameters.get<int>("ACTINTERVALSNUM")),
      actValues_((matdata.parameters.get<std::vector<double>>("ACTVALUES"))),
      density_(matdata.parameters.get<double>("DENS")),
      fiber_orientation_(
          matdata.parameters.get<Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
              Mat::FiberInterpolation>>("FIBER_ORIENTATION"))

{
  // active material parameters
  // stimulation frequency dependent parameters
  double sumrho = 0.0;
  for (int iMU = 0; iMU < muTypesNum_; ++iMU)
  {
    sumrho += rho_[iMU];
  }

  if (muTypesNum_ > 1 && sumrho != 1.0) FOUR_C_THROW("Sum of fractions of MU types must equal one");

  // prescribed activation in time intervals
  if (actTimesNum_ != int(actTimes_.size()))
    FOUR_C_THROW("Number of activation times ACTTIMES must equal ACTTIMESNUM");
  if (actIntervalsNum_ != int(actValues_.size()))
    FOUR_C_THROW("Number of activation values ACTVALUES must equal ACTINTERVALSNUM");
  if (actTimesNum_ != actIntervalsNum_ + 1)
    FOUR_C_THROW("ACTTIMESNUM must be one smaller than ACTINTERVALSNUM");
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::MuscleWeickenmeier::create_material()
{
  return std::make_shared<Mat::MuscleWeickenmeier>(this);
}

Mat::MuscleWeickenmeierType Mat::MuscleWeickenmeierType::instance_;

Core::Communication::ParObject* Mat::MuscleWeickenmeierType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* muscle_weickenmeier = new Mat::MuscleWeickenmeier();
  muscle_weickenmeier->unpack(buffer);
  return muscle_weickenmeier;
}

Mat::MuscleWeickenmeier::MuscleWeickenmeier() : params_(nullptr), lambda_m_old_(1.0) {}

Mat::MuscleWeickenmeier::MuscleWeickenmeier(Mat::PAR::MuscleWeickenmeier* params)
    : params_(params), lambda_m_old_(1.0)
{
  // initialize lambdaMOld_
  lambda_m_old_ = 1.0;
}

void Mat::MuscleWeickenmeier::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  add_to_pack(data, lambda_m_old_);
}

void Mat::MuscleWeickenmeier::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::MuscleWeickenmeier*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  extract_from_pack(buffer, lambda_m_old_);
}

void Mat::MuscleWeickenmeier::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
}

void Mat::MuscleWeickenmeier::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, int const gp,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, int const eleGID)
{
  // compute the current fibre stretch using the deformation gradient and the structural tensor
  // right Cauchy Green tensor C= F^T F
  Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrd) * defgrd);

  // interpolate fiber orientation at current integration point
  Core::LinAlg::Tensor<double, 3> orientation =
      params_->fiber_orientation_.interpolate(eleGID, context.xi->as_span());

  // structural tensor M, i.e. dyadic product of fibre directions
  const Core::LinAlg::SymmetricTensor<double, 3, 3> M = Core::LinAlg::self_dyadic(orientation);

  // save the current fibre stretch in lambdaMOld_
  lambda_m_old_ = Mat::Utils::Muscle::fiber_stretch(C, M);
}

void Mat::MuscleWeickenmeier::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // get passive material parameters
  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double gamma = params_->gamma_;
  const double kappa = params_->kappa_;
  const double omega0 = params_->omega0_;

  // compute matrices
  // right Cauchy Green tensor C
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
  // lambdaM = sqrt(C:M) = sqrt(tr(C^T M)), see Holzapfel2000, p.14
  const double lambdaM = Mat::Utils::Muscle::fiber_stretch(C, M);

  // computation of active nominal stress Pa, and derivative derivPa
  double Pa = 0.0;
  double derivPa = 0.0;
  if (params_->muTypesNum_ != 0)
  {  // if active material
    evaluate_active_nominal_stress(params, context, lambdaM, Pa, derivPa);
  }  // else: Pa and derivPa remain 0.0

  // computation of activation level omegaa and derivative w.r.t. fiber stretch
  double omegaa = 0.0;
  double derivOmegaa = 0.0;
  // compute activation level and derivative only if active nominal stress is not zero
  // if active nominal stress is zero, material is purely passive, thus activation level and
  // derivative are zero
  if (Pa != 0.0)
  {
    evaluate_activation_level(lambdaM, Pa, derivPa, omegaa, derivOmegaa);
  }
  // compute derivative \frac{\partial omegaa}{\partial C} in Voigt notation
  const Core::LinAlg::SymmetricTensor<double, 3, 3> domegaadC = derivOmegaa * 0.5 / lambdaM * M;

  // compute helper matrices for further calculation
  const Core::LinAlg::SymmetricTensor<double, 3, 3> LomegaaM = L + omegaa * M;
  const double fac = (1.0 + omegaa * alpha * std::pow(lambdaM, 2)) / (alpha * std::pow(lambdaM, 2));
  const Core::LinAlg::SymmetricTensor<double, 3, 3> LfacomegaaM = L + fac * M;
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
           (expalpha * LomegaaM - expbeta * detC * invCLinvC +
               (J * expbeta - std::pow(detC, -kappa)) * invC);

  // compute cmat
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
  cmat *= gamma;
}

void Mat::MuscleWeickenmeier::evaluate_active_nominal_stress(const Teuchos::ParameterList& params,
    const Mat::EvaluationContext<3>& context, const double lambdaM, double& Pa, double& derivPa)
{
  // save current simulation time
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  double t_tot = *context.total_time;
  // save (time) step size
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double timestep = *context.time_step_size;

  // approximate first time derivative of lambdaM through BW Euler
  // dotLambdaM = (lambdaM_n - lambdaM_{n-1})/dt
  double dotLambdaM = (lambdaM - lambda_m_old_) / timestep;

  // approximate second time derivative of lambdaM through BW Euler
  // dDotLambdaMdLambdaM = 1/dt approximated through BW Euler
  double dDotLambdaMdLambdaM = 1 / timestep;

  // get active microstructural parameters from params_
  const double Na = params_->Na_;
  const int muTypesNum = params_->muTypesNum_;
  const auto& I = params_->I_;
  const auto& rho = params_->rho_;
  const auto& F = params_->F_;
  const auto& T = params_->T_;

  const double lambdaMin = params_->lambdaMin_;
  const double lambdaOpt = params_->lambdaOpt_;

  const double dotLambdaMMin = params_->dotLambdaMMin_;
  const double ke = params_->ke_;
  const double kc = params_->kc_;
  const double de = params_->de_;
  const double dc = params_->dc_;

  const int actIntervalsNum = params_->actIntervalsNum_;
  const auto& actTimes = params_->actTimes_;
  const auto& actValues = params_->actValues_;

  // compute force-time/stimulation frequency dependency Poptft
  double Poptft = Mat::Utils::Muscle::evaluate_time_dependent_active_stress_ehret(
      Na, muTypesNum, rho, I, F, T, actIntervalsNum, actTimes, actValues, t_tot);

  // compute force-stretch dependency fxi
  double fxi =
      Mat::Utils::Muscle::evaluate_force_stretch_dependency_ehret(lambdaM, lambdaMin, lambdaOpt);

  // compute force-velocity dependency fv
  double fv = Mat::Utils::Muscle::evaluate_force_velocity_dependency_boel(
      dotLambdaM, dotLambdaMMin, de, dc, ke, kc);

  // compute active nominal stress Pa
  Pa = Poptft * fxi * fv;

  // compute derivative of force-stretch dependency fxi and of force-velocity dependency fv
  // w.r.t. lambdaM
  double dFxidLambdaM = 0.0;
  double dFvdLambdaM = 0.0;
  if (Pa != 0)
  {
    dFxidLambdaM = Mat::Utils::Muscle::evaluate_derivative_force_stretch_dependency_ehret(
        lambdaM, lambdaMin, lambdaOpt);
    dFvdLambdaM = Mat::Utils::Muscle::evaluate_derivative_force_velocity_dependency_boel(
        dotLambdaM, dDotLambdaMdLambdaM, dotLambdaMMin, de, dc, ke, kc);
  }

  // compute derivative of active nominal stress Pa w.r.t. lambdaM
  derivPa = Poptft * (fv * dFxidLambdaM + fxi * dFvdLambdaM);
}

void Mat::MuscleWeickenmeier::evaluate_activation_level(const double lambdaM, const double Pa,
    const double derivPa, double& omegaa, double& derivOmegaa)
{
  // get passive material parameters
  const double alpha = params_->alpha_;
  const double gamma = params_->gamma_;
  const double omega0 = params_->omega0_;

  // passive part of invariant I and its first and second derivatives w.r.t. lambdaM
  double Ip = (omega0 / 3.0) * (std::pow(lambdaM, 2.) + 2.0 / lambdaM) +
              (1.0 - omega0) * std::pow(lambdaM, 2.);
  double derivIp = (omega0 / 3.0) * (2.0 * lambdaM - 2.0 / std::pow(lambdaM, 2.)) +
                   2.0 * (1.0 - omega0) * lambdaM;
  double derivderivIp = (omega0 / 3.0) * (2.0 + 4.0 / std::pow(lambdaM, 3.)) + 2.0 * (1.0 - omega0);

  // argument for Lambert W function
  const double xi =
      Pa * ((2.0 * alpha * lambdaM) / gamma) *
          std::exp(0.5 * alpha * (2.0 - 2.0 * Ip + lambdaM * derivIp)) +
      0.5 * alpha * lambdaM * derivIp * std::exp(0.5 * alpha * lambdaM * derivIp);  // argument xi

  // solution W0 of principal branch of Lambert W function approximated with Halley's method
  double W0 = 1.0;           // starting guess for solution
  const double tol = 1e-15;  // tolerance for numeric approximation 10^-15
  const int maxiter = 100;   // maximal number of iterations
  Mat::Utils::Muscle::evaluate_lambert(xi, W0, tol, maxiter);

  // derivatives of xi and W0 w.r.t. lambdaM used for activation level computation
  double derivXi =
      (2.0 * alpha / gamma * std::exp(0.5 * alpha * (2.0 - 2.0 * Ip + lambdaM * derivIp))) *
          (Pa + lambdaM * derivPa +
              0.5 * alpha * Pa * lambdaM * (lambdaM * derivderivIp - derivIp)) +
      0.5 * alpha * (1.0 + 0.5 * alpha) * std::exp(0.5 * alpha * lambdaM * derivIp) *
          (derivIp + lambdaM * derivderivIp);
  double derivLambert = derivXi / ((1.0 + W0) * std::exp(W0));

  // computation of activation level omegaa
  omegaa = W0 / (alpha * std::pow(lambdaM, 2.)) - derivIp / (2.0 * lambdaM);

  // computation of partial derivative of omegaa w.r.t. lambdaM
  derivOmegaa = derivLambert / (alpha * lambdaM * lambdaM) -
                2.0 * W0 / (alpha * lambdaM * lambdaM * lambdaM) - derivderivIp / (2.0 * lambdaM) +
                derivIp / (2.0 * lambdaM * lambdaM);
}
FOUR_C_NAMESPACE_CLOSE
