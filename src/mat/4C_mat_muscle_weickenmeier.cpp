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
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
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
      density_(matdata.parameters.get<double>("DENS"))
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

Mat::MuscleWeickenmeier::MuscleWeickenmeier()
    : params_(nullptr),
      lambda_m_old_(1.0),
      anisotropy_(),
      anisotropy_extension_(true, 0.0, 0,
          std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
              new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {0})
{
}

Mat::MuscleWeickenmeier::MuscleWeickenmeier(Mat::PAR::MuscleWeickenmeier* params)
    : params_(params),
      lambda_m_old_(1.0),
      anisotropy_(),
      anisotropy_extension_(true, 0.0, 0,
          std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
              new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {0})
{
  // initialize lambdaMOld_
  lambda_m_old_ = 1.0;

  // register anisotropy extension to global anisotropy
  anisotropy_.register_anisotropy_extension(anisotropy_extension_);

  // initialize fiber directions and structural tensor
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::FIBER_VECTORS |
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
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

  anisotropy_extension_.pack_anisotropy(data);
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

  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mat::MuscleWeickenmeier::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Read anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(fibers, coord_system);
}

void Mat::MuscleWeickenmeier::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, int const gp,
    const Teuchos::ParameterList& params, const EvaluationContext& context, int const eleGID)
{
  // compute the current fibre stretch using the deformation gradient and the structural tensor
  // right Cauchy Green tensor C= F^T F
  Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrd) * defgrd);

  // structural tensor M, i.e. dyadic product of fibre directions
  const Core::LinAlg::SymmetricTensor<double, 3, 3>& M =
      anisotropy_extension_.get_structural_tensor(gp, 0);

  // save the current fibre stretch in lambdaMOld_
  lambda_m_old_ = Mat::Utils::Muscle::fiber_stretch(C, M);
}

void Mat::MuscleWeickenmeier::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
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
  Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(*defgrad) * *defgrad);
  const Core::LinAlg::Matrix<3, 3> C_mat = Core::LinAlg::make_matrix(Core::LinAlg::get_full(C));
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
  Core::LinAlg::Matrix<6, 1> domegaadCv(Mv);
  domegaadCv.scale(derivOmegaa * 0.5 / lambdaM);

  // compute helper matrices for further calculation
  Core::LinAlg::Matrix<3, 3> LomegaaM(L);
  LomegaaM.update(omegaa, M_mat, 1.0);  // LomegaaM = L + omegaa*M
  Core::LinAlg::Matrix<6, 1> LomegaaMv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(LomegaaM, LomegaaMv);

  Core::LinAlg::Matrix<3, 3> LfacomegaaM(L);  // LfacomegaaM = L + fac*M
  LfacomegaaM.update(
      (1.0 + omegaa * alpha * std::pow(lambdaM, 2.)) / (alpha * std::pow(lambdaM, 2.)), M_mat,
      1.0);  // + fac*M
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
  stressM.scale(0.5 * gamma);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(
      stressM, Sc_stress);  // convert to Voigt notation and update stress

  // compute cmat
  ccmat.multiply_nt(alpha * expalpha, LomegaaMv, LomegaaMv, 1.0);  // add contributions
  ccmat.multiply_nt(alpha * std::pow(lambdaM, 2.) * expalpha, LfacomegaaMv, domegaadCv, 1.0);
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
  ccmat.scale(gamma);

  // update stress and material tangent with the computed stress and cmat values
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  stress_view.update(1.0, Sc_stress, 1.0);
  cmat_view.update(1.0, ccmat, 1.0);
}

void Mat::MuscleWeickenmeier::evaluate_active_nominal_stress(const Teuchos::ParameterList& params,
    const Mat::EvaluationContext& context, const double lambdaM, double& Pa, double& derivPa)
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
