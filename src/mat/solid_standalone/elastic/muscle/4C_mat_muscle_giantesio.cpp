// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_muscle_giantesio.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_derivatives.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_einstein.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_muscle_utils.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{

  struct StressAndDeriv
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> Se;
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dSedC;
  };

  /*!
   * @brief Compute the elastic second Piola Kirchhoff stress tensor Se and its derivative dSedC
   * w.r.t. the right Cauchy Green tensor C
   */
  StressAndDeriv compute_se_and_d_sed_c(const double& alpha, const double& beta,
      const double& gamma, const double& omega0,
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& M,
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& invFa,
      const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& dinvFadC,
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& C,
      const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& dCdC)
  {
    // structural tensor L = omega0/3*Identity + omegap*M
    const auto L =
        omega0 / 3. * Core::LinAlg::TensorGenerators::identity<double, 3, 3> + (1 - omega0) * M;

    // elastic right Cauchy Green tensor Ce = Fe^T Fe
    // = Fa^-T C Fa^-1 = Fa^-1 C Fa^-1 (with Fa^-1 sym.)
    const auto Ce = Core::LinAlg::assume_symmetry(invFa * C * invFa);

    // derivative of Ce w.r.t C
    // dCedC_ijkl = dinvFadC_iakl (C invFa)_aj + invFa_ia dCdC_abkl invFa_bj + (invFa C)_ab
    // dinvFadC_bjkl
    const auto dCedC = Core::LinAlg::einsum<"iakl", "aj">(dinvFadC, C * invFa) +
                       Core::LinAlg::einsum<"ia", "abkl", "bj">(invFa, dCdC, invFa) +
                       Core::LinAlg::einsum<"ab", "bjkl">(invFa * C, dinvFadC);

    // inverse of the elastic right Cauchy Green tensor Ce
    const auto invCe = Core::LinAlg::inv(Ce);

    // product invCe*L*invCe
    const auto invCeLinvCe = Core::LinAlg::assume_symmetry(invCe * L * invCe);

    // generalized elastic invariants including only passive properties
    const double detCe = Core::LinAlg::det(Ce);
    const double Ie = Core::LinAlg::ddot(Ce, L);
    // J = cof(Ce):L = tr(cof(Ce)^T L) = tr(adj(Ce) L) = tr(det(Ce) Ce^-1 L) = det(Ce)*tr(Ce^-1 L)
    const double Je = detCe * Core::LinAlg::trace(invCe * L);

    // exponential prefactors
    const double expalpha = std::exp(alpha * (Ie - 1.0));
    const double expbeta = std::exp(beta * (Je - 1.0));

    // elastic second Piola Kirchhoff stress tensor Se
    const auto Se = 0.5 * gamma * (expalpha * L + expbeta * (Je * invCe - detCe * invCeLinvCe));

    // derivative of Se w.r.t Ce
    // taken from Weickenmeier et al.
    auto dSedCe = alpha * expalpha * Core::LinAlg::dyadic(L, L);
    dSedCe += beta * expbeta * detCe * detCe * Core::LinAlg::dyadic(invCeLinvCe, invCeLinvCe);
    dSedCe -= (beta * Je + 1.) * expbeta * detCe *
              (Core::LinAlg::dyadic(invCe, invCeLinvCe) + Core::LinAlg::dyadic(invCeLinvCe, invCe));
    dSedCe += (beta * Je + 1.) * Je * expbeta * Core::LinAlg::dyadic(invCe, invCe);
    dSedCe -= Je * expbeta * Core::LinAlg::FourTensorOperations::holzapfel_product(invCe);
    dSedCe -=
        expbeta * detCe *
        Core::LinAlg::FourTensorOperations::derivative_of_inva_b_inva_product(invCe, invCeLinvCe);
    dSedCe *= gamma / 2;

    // derivative of Se w.r.t C
    // dSedC_ijkl = dSedCe_ijab dCedC_abkl
    const auto dSedC = Core::LinAlg::assume_symmetry(Core::LinAlg::ddot(dSedCe, dCedC));

    const StressAndDeriv Se_dSedC = {.Se = Se, .dSedC = dSedC};

    return Se_dSedC;
  }
}  // namespace

Mat::PAR::MuscleGiantesio::MuscleGiantesio(const Core::Mat::PAR::Parameter::Data& matdata)
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

std::shared_ptr<Core::Mat::Material> Mat::PAR::MuscleGiantesio::create_material()
{
  return std::make_shared<Mat::MuscleGiantesio>(this);
}

Mat::MuscleGiantesioType Mat::MuscleGiantesioType::instance_;

Core::Communication::ParObject* Mat::MuscleGiantesioType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* muscle_giantesio = new Mat::MuscleGiantesio();
  muscle_giantesio->unpack(buffer);
  return muscle_giantesio;
}

Mat::MuscleGiantesio::MuscleGiantesio() : params_(nullptr), lambda_m_old_(1.0), omegaa_old_(-1.0) {}

Mat::MuscleGiantesio::MuscleGiantesio(Mat::PAR::MuscleGiantesio* params)
    : params_(params), lambda_m_old_(1.0), omegaa_old_(-1.0)
{
  // initialize lambdaMOld_ and omegaOld_
  lambda_m_old_ = 1.0;
  omegaa_old_ = -1.0;
}

void Mat::MuscleGiantesio::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  add_to_pack(data, lambda_m_old_);
  add_to_pack(data, omegaa_old_);
}

void Mat::MuscleGiantesio::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::MuscleGiantesio*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  extract_from_pack(buffer, lambda_m_old_);
  extract_from_pack(buffer, omegaa_old_);
}

void Mat::MuscleGiantesio::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
}

void Mat::MuscleGiantesio::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, int const gp,
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

void Mat::MuscleGiantesio::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrd,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  // get passive material parameters
  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double gamma = params_->gamma_;
  const double kappa = params_->kappa_;
  const double omega0 = params_->omega0_;

  // save current simulation time
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  double currentTime = *context.total_time;

  // save (time) step size
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  double timeStepSize = *context.time_step_size;

  // right Cauchy Green tensor C
  const Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(*defgrd) * *defgrd);

  // determinant of C
  const double detC = Core::LinAlg::det(C);

  // derivative of C w.r.t C, i.e., forth order identity tensor
  const auto dCdC =
      Core::LinAlg::assume_symmetry(Core::LinAlg::TensorGenerators::identity<double, 3, 3, 3, 3>);

  // inverse right Cauchy Green tensor C^-1
  const auto invC = Core::LinAlg::inv(C);

  // interpolate fiber orientation at current integration point
  Core::LinAlg::Tensor<double, 3> orientation =
      params_->fiber_orientation_.interpolate(eleGID, context.xi->as_span());

  // structural tensor M, i.e. dyadic product of fibre directions
  const Core::LinAlg::SymmetricTensor<double, 3, 3> M = Core::LinAlg::self_dyadic(orientation);

  // fiber stretch lambdaM
  const double lambdaM = Mat::Utils::Muscle::fiber_stretch(C, M);

  // derivative of lambdaM w.r.t. C
  const auto dlambdaMdC = Mat::Utils::Muscle::d_fiber_stretch_dc(lambdaM, C, M);

  // contraction velocity dotLambdaM
  const double dotLambdaM =
      Mat::Utils::Muscle::contraction_velocity_bw_euler(lambdaM, lambda_m_old_, timeStepSize);

  // compute activation level omegaa and derivative w.r.t. fiber stretch if material is active
  Core::Utils::ValuesFunctAndFunctDerivs omegaaAndDerivs = {
      .val_funct = 0.0, .val_deriv_funct = 0.0, .val_deriv_deriv_funct = 0.0};

  if (is_active(currentTime))
  {
    omegaaAndDerivs = evaluate_activation_level_and_derivatives(lambdaM, dotLambdaM, currentTime);
    omegaa_old_ = omegaaAndDerivs.val_funct;  // update omegaaOld
  }
  else
  {
    omegaa_old_ = -1;  // reset omegaaOld to -1
  }

  // --------------------------------------------------------------
  // first derivative of omegaa w.r.t. C
  // domegaadC = domegaadlambdaM dlambdaMdC
  const auto domegaadC = omegaaAndDerivs.val_deriv_funct * dlambdaMdC;

  // second derivative of omegaa w.r.t. C
  // ddomegaaddC = (ddomegaaddlambdaM - 1/lambdaM domegaadlambdaM) dlambdaM_ij dlambdaM_kl
  const auto ddomegaaddC =
      (omegaaAndDerivs.val_deriv_deriv_funct - omegaaAndDerivs.val_deriv_funct / lambdaM) *
      Core::LinAlg::dyadic(dlambdaMdC, dlambdaMdC);

  // --------------------------------------------------------------
  // active deformation gradient Fa
  const auto Fa = act_def_grad(omegaaAndDerivs.val_funct, M);

  // first derivative of Fa w.r.t. omegaa
  const auto dFadomegaa = d_act_def_grad_d_act_level(omegaaAndDerivs.val_funct, M);

  // second derivative of Fa w.r.t. omegaa
  const auto ddFaddomegaa = dd_act_def_grad_dd_act_level(omegaaAndDerivs.val_funct, M);

  // determinant of Fa
  const double detFa = Core::LinAlg::det(Fa);

  // --------------------------------------------------------------
  // inverse of active deformation gradient Fa^{-1}
  const auto invFa = Core::LinAlg::inv(Fa);

  // first derivative of Fa^{-1} w.r.t. omegaa
  const auto dinvFadomegaa = d_inv_act_def_grad_d_act_level(invFa, dFadomegaa);

  // first derivative of Fa^{-1} w.r.t. C
  // dinvFadC_ijkl = dinvFadomegaa_ij domegaadC_kl
  const auto dinvFadC = Core::LinAlg::dyadic(dinvFadomegaa, domegaadC);

  // --------------------------------------------------------------
  // elastic second Piola Kirchhoff stress tensor Se and its derivative w.r.t C
  StressAndDeriv Se_dSedC =
      compute_se_and_d_sed_c(alpha, beta, gamma, omega0, M, invFa, dinvFadC, C, dCdC);

  // --------------------------------------------------------------
  // compute second Piola Kirchhoff stress components S1, S2, Svol
  // note: S3 = invF * ddetFadF * Psie = 0 since ddetFawa = 0 and ddetFadF = ddetFadwa dwadF

  // S1 = detFa * invFa * Se * invFa^T
  // in index notation: S1_sl = detFa * invFa_sr * Se_rq * invFa_rl (with invFa sym.)
  const auto S1 = detFa * Core::LinAlg::assume_symmetry(invFa * Se_dSedC.Se * invFa);

  // S2 = detFa * invF * dPsiedFe * (F * dinvFadF)  = 2 detFa (C Fa^-1 Se) dFa^-1dC
  // or: S2 = -2 S1 : (C Fa^-1 dFadomegaa) domegaadC
  // in index notation: S2_sl = -2 S1_ij (C Fa^-1 dFadomegaa)_ij domegaadC_sl
  const auto CinvFadFadomegaa = Core::LinAlg::assume_symmetry(C * invFa * dFadomegaa);
  const double scalar_contraction = -2.0 * Core::LinAlg::ddot(S1, CinvFadFadomegaa);
  const auto S2 = scalar_contraction * domegaadC;

  // Svol = - gamma/2 * detC^-kappa
  const auto Svol = -0.5 * gamma * std::pow(detC, -kappa) * invC;

  // --------------------------------------------------------------
  // compute material tangent components cmat1, cmat2, cmatvol

  // compute cmat1 = 2 dS1dC
  // derivative of S1 = detFa * invFa * Se * invFa^T  w.r.t. C
  // = detFa * (dinvFadC * (Se * dinvFa^T) + invFa * dSedC * invFa^T + (invFa * Se) * dinvFadC)
  auto dS1dC = Core::LinAlg::einsum<"ijkm", "mn", "nl">(
      dinvFadC, Se_dSedC.Se, Core::LinAlg::transpose(invFa));  // dinvFadC * (Se * dinvFa^T)
  dS1dC += (Core::LinAlg::einsum<"im", "mjkn", "nl">(
      invFa, Se_dSedC.dSedC, Core::LinAlg::transpose(invFa)));  // invFa * dSedC * invFa^T
  dS1dC += (Core::LinAlg::einsum<"im", "mn", "njkl">(
      invFa, Se_dSedC.Se, dinvFadC));  //  (invFa * Se) * dinvFadC
  dS1dC *= detFa;

  const auto cmat1 = 2 * Core::LinAlg::assume_symmetry(dS1dC);

  // compute cmat2 = 2 dS2dC
  // derivative of S2 w.r.t. C
  // in index notation: S2_sl = -2 S1_ij (C Fa^-1 dFadomegaa)_ij domegaadC_sl
  // dS2dC_slpq = -4 [ S1_ij d(C Fa^-1 dFadomegaa)dC_ijpq domegaadC_sl +
  //                 [ S1_ij (C Fa^-1 dFadomegaa)_ij ddomegaaddC_slpq ]
  // dS2dC_slpq =  2 [ dscalardC_pq domegaadC_sl + [ scalar_contraction ddomegaaddC_slpq ]

  // derivative of the scalar (-2 S1_ij (C Fa^-1 dFadomegaa)_ij) w.r.t. C
  // dscalardC_pq = -2 (dS1dC_ijpq (C invFa dFadomegaa)_ij + S1_ij d(C invFa dFadomegaa)dC_ijpq)
  const auto H = dinvFadomegaa * dFadomegaa + invFa * ddFaddomegaa;
  const auto CH = Core::LinAlg::assume_symmetry(C * H);
  const double helper = Core::LinAlg::ddot(S1, CH);

  const auto dscalardC =
      helper * domegaadC + Core::LinAlg::assume_symmetry(S1 * dFadomegaa * invFa) +
      Core::LinAlg::assume_symmetry(Core::LinAlg::einsum<"ij", "ijpq">(CinvFadFadomegaa, dS1dC));

  // compute cmat2 = 2 dS2dC = 2 [domegaadC_sl dscalardC_pq + scalar * ddomegaaddC_slpq]
  const auto cmat2 =
      -4.0 * Core::LinAlg::dyadic(domegaadC, dscalardC)  // 2 * domegaadC_sl dscalardC_pq
      + 2.0 * scalar_contraction * ddomegaaddC;          // 2 * scalar * ddomegaaddC_slpq

  // compute cmatvol = 2 dSvoldC
  const auto cmatvol = gamma * std::pow(detC, -kappa) *
                       (kappa * Core::LinAlg::dyadic(invC, invC) +
                           Core::LinAlg::FourTensorOperations::holzapfel_product(invC));

  // --------------------------------------------------------------
  // update constituent stress and material tangent
  stress = S1 + S2 + Svol;
  cmat = cmat1 + cmat2 + cmatvol;
}

Core::Utils::ValuesFunctAndFunctDerivs
Mat::MuscleGiantesio::evaluate_activation_level_and_derivatives(
    const double& lambdaM, const double& dotLambdaM, const double& currentTime)
{
  // setup function
  std::function<double(double)> FunctionSolveActivationLevelEquation =
      [this, &dotLambdaM, &currentTime](double lambda_i)
  { return this->solve_activation_level_equation(lambda_i, dotLambdaM, currentTime); };

  // step size for finite differences (error will be of O(h^2))
  // rule of thumb: h = x0*sqrt(ulp); but omegaa is only precise to 1e-12
  const double h = lambdaM * std::sqrt(1e-12);

  // compute activation level omegaa and its first and second derivative w.r.t. lambdaM using
  // central differences
  Core::Utils::ValuesFunctAndFunctDerivs omegaaAndDerivs =
      Core::Utils::evaluate_function_and_derivatives_central_differences(
          FunctionSolveActivationLevelEquation, lambdaM, h);

  return omegaaAndDerivs;
}

double Mat::MuscleGiantesio::solve_activation_level_equation(
    const double& lambdaM, const double& dotLambdaM, const double& currentTime)
{
  // compute right hand side of equation
  double rhs = evaluate_rhs_activation_level_equation(lambdaM, dotLambdaM, currentTime);

  // setup the activation level equation as f(omegaa) = lhs-rhs
  // and its derivative as df/domegaa(omegaa) = dlhs/domegaa
  auto actLevelEquationAndDeriv = [this, &lambdaM, &rhs](double omegaa_init)
  { return this->evaluate_activation_level_equation_and_deriv(omegaa_init, lambdaM, rhs); };

  // determine starting guess for newton solver
  double omegaa_init;
  if (omegaa_old_ < 0.0)
  {
    // setup parameters for bisection method for the approximation of the starting guess for the
    // newton solver in case omegaa is not available from a previous timestep
    const double tol_bisec = 1e-2;  // abs(f(x)) <= 0.01
    const int maxiter_bisec = 100;
    const double omegaa_a_init = 0.0;  // omegaa >= 0
    const double omegaa_b_init = 1.0;  // omegaa <= 1

    // approximate the starting guess for the newton solver via bisection method
    omegaa_init = Core::Utils::bisection([&](double omegaa_init)
        { return std::get<0>(actLevelEquationAndDeriv(omegaa_init)); }, omegaa_a_init,
        omegaa_b_init, tol_bisec, maxiter_bisec);
  }
  else
    // use omegaa from the previous timestep as a starting guess
    omegaa_init = omegaa_old_;

  // setup parameters for newton method
  const double tol_newton = 1e-12;  // reasonably small
  const int maxiter_newton = 200;

  // compute activation level as solution of activation level equation
  double omegaa = Core::Utils::solve_local_newton(
      actLevelEquationAndDeriv, omegaa_init, tol_newton, maxiter_newton);

  return omegaa;
}

std::tuple<double, double> Mat::MuscleGiantesio::evaluate_activation_level_equation_and_deriv(
    double omegaa, const double& lambdaM, const double& rhs)
{
  // get active microstructural parameters from params_
  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double omega0 = params_->omega0_;

  // elastic invariants Ie and Je
  double Ie = (-omega0 * (std::pow(omegaa - 1., 3) / std::pow(lambdaM, 3) + 1.) + 1.5) *
              std::pow(lambdaM, 2) / (1.5 * std::pow(omegaa - 1., 2));
  double Je = (-omega0 * (std::pow(lambdaM, 3) / std::pow(omegaa - 1., 3) + 1.) + 1.5) *
              std::pow(omegaa - 1., 2) / (1.5 * std::pow(lambdaM, 2));

  // derivatives of elastic invariants Ie and Je w.r.t. lambdaM
  double derivIe = (std::pow(lambdaM, 3) * (2 * omega0 - 3) - omega0 * std::pow(omegaa - 1., 3)) /
                   (1.5 * lambdaM * std::pow(omegaa - 1., 3));
  double derivJe = (std::pow(lambdaM, 3) * omega0 - (2 * omega0 - 3) * std::pow(omegaa - 1., 3)) /
                   (1.5 * std::pow(lambdaM, 2) * std::pow(omegaa - 1., 2));

  // compute left hand side of the activation level equation
  double lhs = std::exp(alpha * (Ie - 1)) / alpha + std::exp(beta * (Je - 1)) / beta;

  // compute derivative of left hand side of the activation level equation
  double derivlhs = derivIe * std::exp(alpha * (Ie - 1)) + derivJe * std::exp(beta * (Je - 1));

  // compute activation level equation and its derivative
  return {lhs - rhs, derivlhs};
}

double Mat::MuscleGiantesio::evaluate_rhs_activation_level_equation(
    const double& lambdaM, const double& dotLambdaM, const double& currentTime)
{
  // get active microstructural parameters from params_
  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double gamma = params_->gamma_;
  const double omega0 = params_->omega0_;

  // compute active nominaal stress and its integral w.r.t. lambdaM
  double intPa = evaluate_active_nominal_stress_integral(lambdaM, dotLambdaM, currentTime);

  // passive part of invariants Ip and Jp w.r.t. lambdaM
  const double Ip = (omega0 / std::pow(lambdaM, 3) - omega0 + 1.5) * std::pow(lambdaM, 2) / 1.5;
  const double Jp = (omega0 * std::pow(lambdaM, 3) - omega0 + 1.5) / std::pow(lambdaM, 2) / 1.5;

  // compute right hand side of equation
  double rhs = 4.0 * intPa / gamma + std::exp(alpha * (Ip - 1.0)) / alpha +
               std::exp(beta * (Jp - 1.0)) / beta;

  return rhs;
}

double Mat::MuscleGiantesio::evaluate_active_nominal_stress_integral(
    const double& lambdaM, const double& dotLambdaM, const double& currentTime)
{
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
      Na, muTypesNum, rho, I, F, T, actIntervalsNum, actTimes, actValues, currentTime);

  // compute integral of the force-stretch dependency fxi in the boundaries lambdaMin to lambdaM
  double intFxi = Mat::Utils::Muscle::evaluate_integral_force_stretch_dependency_ehret(
      lambdaM, lambdaMin, lambdaOpt);

  // compute force-velocity dependency fv
  double fv = Mat::Utils::Muscle::evaluate_force_velocity_dependency_boel(
      dotLambdaM, dotLambdaMMin, de, dc, ke, kc);

  // compute integral of the active nominal stress Pa
  double intPa = Poptft * intFxi * fv;

  return intPa;
}

Core::LinAlg::SymmetricTensor<double, 3, 3> Mat::MuscleGiantesio::act_def_grad(
    const double omegaa, const Core::LinAlg::SymmetricTensor<double, 3, 3>& M)
{
  // active deformation gradient Fa
  auto Fa = (1 - omegaa - std::pow(1. - omegaa, -0.5)) * M;
  Fa += std::pow(1. - omegaa, -0.5) * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  return Fa;
}

Core::LinAlg::SymmetricTensor<double, 3, 3> Mat::MuscleGiantesio::d_act_def_grad_d_act_level(
    const double omegaa, const Core::LinAlg::SymmetricTensor<double, 3, 3>& M)
{
  // first derivative of Fa w.r.t. omegaa
  Core::LinAlg::SymmetricTensor<double, 3, 3> dFadomegaa{};
  if (omegaa != 0)
  {
    dFadomegaa = (-1.0 - 0.5 * std::pow(1. - omegaa, -1.5)) * M;
    dFadomegaa +=
        0.5 * std::pow(1. - omegaa, -1.5) * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  }
  return dFadomegaa;
}

Core::LinAlg::SymmetricTensor<double, 3, 3> Mat::MuscleGiantesio::dd_act_def_grad_dd_act_level(
    const double omegaa, const Core::LinAlg::SymmetricTensor<double, 3, 3>& M)
{
  // second derivative of Fa w.r.t. omegaa
  Core::LinAlg::SymmetricTensor<double, 3, 3> ddFaddomegaa{};
  if (omegaa != 0)
  {
    ddFaddomegaa = (-0.75 * std::pow(1. - omegaa, -2.5)) * M;
    ddFaddomegaa +=
        0.75 * std::pow(1. - omegaa, -2.5) * Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
  }
  return ddFaddomegaa;
}

Core::LinAlg::SymmetricTensor<double, 3, 3> Mat::MuscleGiantesio::d_inv_act_def_grad_d_act_level(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& invFa,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& dFadomegaa)
{
  // first derivative of Fa^{-1} w.r.t. omegaa
  // dinvFadomegaa_ij = - invFa_ik dFadomegaa_kl invFa_lj
  const auto dinvFadomegaa = -1.0 * Core::LinAlg::assume_symmetry(invFa * dFadomegaa * invFa);
  return dinvFadomegaa;
}

bool Mat::MuscleGiantesio::is_active(const double& currentTime)
{
  bool isActive = false;

  int times_index = 0;
  while (currentTime >= params_->actTimes_[times_index]) times_index++;
  int values_index = times_index - 1;

  // values_index = -1 -> currentTime = 0 -> choose first value
  if (values_index == -1) values_index = 0;

  if (values_index >= 0)
  {
    if (params_->actValues_[values_index] > 1e-14) isActive = true;
  }

  return isActive;
};
FOUR_C_NAMESPACE_CLOSE
