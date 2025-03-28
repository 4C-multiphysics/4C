// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_isomuscle_blemker.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_muscle_utils.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::IsoMuscleBlemker::IsoMuscleBlemker(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      G1_(matdata.parameters.get<double>("G1")),
      G2_(matdata.parameters.get<double>("G2")),
      P1_(matdata.parameters.get<double>("P1")),
      P2_(matdata.parameters.get<double>("P2")),
      sigma_max_(matdata.parameters.get<double>("SIGMAMAX")),
      lambda_ofl_(matdata.parameters.get<double>("LAMBDAOFL")),
      lambda_star_(matdata.parameters.get<double>("LAMBDASTAR")),
      alpha_(matdata.parameters.get<double>("ALPHA")),
      beta_(matdata.parameters.get<double>("BETA")),
      t_act_start_(matdata.parameters.get<double>("ACTSTARTTIME"))
{
  // error handling for parameter ranges
  if (G1_ < 0.0) FOUR_C_THROW("Material parameter G1 must be positive or zero");
  if (G2_ < 0.0) FOUR_C_THROW("Material parameter G2 must be positive or zero");
  if (P1_ <= 0.0) FOUR_C_THROW("Material parameter P1 must be greater zero");
  if (P2_ <= 0.0) FOUR_C_THROW("Material parameter P2 must be greater zero");
  if (sigma_max_ < 0.0) FOUR_C_THROW("Material parameter SIGMA_MAX must be positive or zero");
  if (lambda_ofl_ <= 0.0) FOUR_C_THROW("Material parameter LAMBDA_OFL must be greater zero");
  if (lambda_star_ <= 0.0) FOUR_C_THROW("Material parameter LAMBDA_STAR must be greater zero");
  if (alpha_ < 0.0) FOUR_C_THROW("Material parameter ALPHA must be positive or zero");
  if (beta_ < 0.0) FOUR_C_THROW("Material parameter BETA must be positive or zero");
}


Mat::Elastic::IsoMuscleBlemker::IsoMuscleBlemker(Mat::Elastic::PAR::IsoMuscleBlemker* params)
    : params_(params),
      anisotropy_extension_(true, 0.0, false,
          std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
              new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {0})

{
  // initialize fiber directions and structural tensor
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::FIBER_VECTORS |
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR |
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR_STRESS);
}

void Mat::Elastic::IsoMuscleBlemker::pack_summand(Core::Communication::PackBuffer& data) const
{
  anisotropy_extension_.pack_anisotropy(data);
}

void Mat::Elastic::IsoMuscleBlemker::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mat::Elastic::IsoMuscleBlemker::register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
{
  anisotropy.register_anisotropy_extension(anisotropy_extension_);
}

void Mat::Elastic::IsoMuscleBlemker::add_stress_aniso_modified(
    const Core::LinAlg::Matrix<6, 1>& rcg, const Core::LinAlg::Matrix<6, 1>& icg,
    Core::LinAlg::Matrix<6, 6>& cmat, Core::LinAlg::Matrix<6, 1>& stress, double I3, const int gp,
    const int eleGID, Teuchos::ParameterList& params)
{
  // right Cauchy Green tensor C in matrix notation
  Core::LinAlg::Matrix<3, 3> C(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::vector_to_matrix(rcg, C);

  // compute volume ratio J
  double J = std::sqrt(I3);
  double incJ = std::pow(J, -2.0 / 3.0);

  // compute modified right Cauchy-Green Tensor
  Core::LinAlg::Matrix<3, 3> modC(Core::LinAlg::Initialization::uninitialized);
  modC.update(incJ, C, 0.0);

  // structural tensor M, i.e. dyadic product of fibre directions
  Core::LinAlg::Matrix<3, 3> M = anisotropy_extension_.get_structural_tensor(gp, 0);
  Core::LinAlg::Matrix<6, 1> Mv = anisotropy_extension_.get_structural_tensor_stress(gp, 0);

  // compute modified invariants modI1, modI4 and modI5
  double modI1 = modC(0, 0) + modC(1, 1) + modC(2, 2);

  Core::LinAlg::Matrix<3, 3> modCM(Core::LinAlg::Initialization::zero);
  modCM.multiply_nn(1.0, modC, M);                           // modC*M
  double modI4 = (modCM(0, 0) + modCM(1, 1) + modCM(2, 2));  // modI4 = tr(modC*M)

  Core::LinAlg::Matrix<3, 3> modC2(Core::LinAlg::Initialization::zero);
  modC2.multiply_nn(1.0, modC, modC);  // modC*modC
  Core::LinAlg::Matrix<3, 3> modC2M(Core::LinAlg::Initialization::zero);
  modC2M.multiply_nn(1.0, modC2, M);                            // (modC*modC)*M
  double modI5 = (modC2M(0, 0) + modC2M(1, 1) + modC2M(2, 2));  // modI5 = tr((modC*modC)*M)

  // compute time-dependent activation level
  const double& alpha = params_->alpha_;
  const double& beta = params_->beta_;
  const double& sigma_max = params_->sigma_max_;
  const double& t_act_start = params_->t_act_start_;
  const double& t_tot = params.get<double>("total time");  // current simulation time
  double sigma_max_ft = Mat::Utils::Muscle::evaluate_time_dependent_active_stress_tanh(
      sigma_max, alpha, beta, t_act_start, t_tot);

  // compute total fiber cauchy stress and derivative w.r.t. fibre stretch
  double sigma_fiber_total = 0.0;
  double deriv_sigma_fiber_total = 0.0;
  double lambdaM = std::sqrt(modI4);  // fiber stretch
  evaluate_total_fiber_cauchy_stress_and_derivative(
      lambdaM, sigma_max_ft, sigma_fiber_total, deriv_sigma_fiber_total);

  // helper variables for computation of 2nd Piola Kirchhoff stress and elasticity tensor
  double H1 = (modI1 * modI4 - modI5) / (2.0 * lambdaM);
  // prevents singularitys in cross-fiber-shear-free states
  if ((H1 - 1.0) < 1e-15) H1 = 1.0 + 1e-15;
  double H2 = std::sqrt(H1 * H1 - 1.0);
  double H3 = modI1 / (2.0 * lambdaM) - H1 / (2.0 * modI4);
  double B2 = std::acosh(H1);

  // get material parameters
  const double& G1 = params_->G1_;
  const double& G2 = params_->G2_;

  // scalar prefactors, gamma_i = derivative of strain-energy function w.r.t. modified
  // invariant i; Psi_iso = W1(modI4, modI5) + W2(mod_I1, modI4, modI5) + W3(modI4)
  double gamma1 = 2.0 * G2 * (B2 / H2) * lambdaM;
  double gamma4_1 = -4.0 * G1 * modI5 / std::pow(modI4, 3.0);  // dW1/dI4
  double gamma4_2 = 4.0 * G2 * (B2 / H2) * H3;                 // dW2/dI4
  double gamma4_3 = sigma_fiber_total / modI4;                 // dW3/dI4
  double gamma4 = gamma4_1 + gamma4_2 + gamma4_3;
  double gamma5 = 2.0 * G1 / std::pow(modI4, 2.0) - 2.0 * G2 * (B2 / H2) / lambdaM;

  // matrix terms for stress evaluation
  // unitary 3x3 matrix
  const Core::LinAlg::Matrix<3, 3> Id3 = Core::LinAlg::identity_matrix<3>();
  Core::LinAlg::Matrix<6, 1> Id3v(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::identity_matrix(Id3v);

  // sum modC*M + M*modC = dI5/dC
  Core::LinAlg::Matrix<3, 3> modCMsumMmodC(modCM);
  modCMsumMmodC.multiply_nn(1.0, M, modC, 1.0);
  Core::LinAlg::Matrix<6, 1> modCMsumMmodCv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(modCMsumMmodC, modCMsumMmodCv);

  // fictitious 2nd Piola-Kirchhoff stress tensor modS
  Core::LinAlg::Matrix<3, 3> modS(Core::LinAlg::Initialization::zero);
  modS.update(gamma1, Id3, 1.0);            // + gamma1*I
  modS.update(gamma4, M, 1.0);              // + gamma4*M
  modS.update(gamma5, modCMsumMmodC, 1.0);  // dyad(a0,modC*a0)+dyad(a0*C,a0) = (modC*M)' +modC'*M
  Core::LinAlg::Matrix<6, 1> modSv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(modS, modSv);

  // isometirc 2nd Piola-Kirchhoff tensor S_iso from fictitious 2nd PK stress
  double traceCmodS = modSv(0) * rcg(0) + modSv(1) * rcg(1) + modSv(2) * rcg(2) +
                      (modSv(3) * rcg(3) + modSv(4) * rcg(4) + modSv(5) * rcg(5));
  Core::LinAlg::Matrix<6, 1> S_isov(modSv);
  S_isov.update((-incJ / 3.0) * traceCmodS, icg, incJ);  // icg comes in stress-like notation

  // update 2nd Piola-Kirchhoff  tensor
  stress.update(1.0, S_isov, 1.0);

  // scalar prefactors for computation of elasticity tensor
  double delta1 =
      (2. * G2 * modI4) / (std::pow(H2, 2.)) - (2. * B2 * G2 * H1 * modI4) / (std::pow(H2, 3.));
  double delta5 = (2. * B2 * G2) / (H2 * lambdaM) + (4. * G2 * H3 * lambdaM) / (std::pow(H2, 2.)) -
                  (4. * B2 * G2 * H1 * H3 * lambdaM) / (std::pow(H2, 3.));
  double delta7 = (24. * G1 * modI5) / (std::pow(modI4, 4.)) +
                  (8. * G2 * std::pow(H3, 2.)) / (std::pow(H2, 2.)) +
                  (2. * B2 * G2 * (3. * H1 - 2. * modI1 * lambdaM)) / (H2 * std::pow(modI4, 2.)) -
                  (8. * B2 * G2 * std::pow(H3, 2.) * H1) / (std::pow(H2, 3.));
  delta7 += 2. * (-sigma_fiber_total / std::pow(modI4, 2.) + deriv_sigma_fiber_total / modI4);
  double delta8 = (2. * B2 * G2 * H1) / (std::pow(H2, 3.)) - (2. * G2) / (std::pow(H2, 2.));
  double delta10 =
      (2. * G2) / (std::pow(H2, 2) * modI4) - (2. * B2 * G2 * H1) / (modI4 * std::pow(H2, 3.));
  double delta11 = (2. * B2 * G2) / (H2 * std::pow(modI4, 3. / 2.)) -
                   (8. * G1) / (std::pow(modI4, 3.)) -
                   (4. * G2 * H3) / (std::pow(H2, 2.) * lambdaM) +
                   (4. * B2 * G2 * H3 * H1) / (lambdaM * std::pow(H2, 3.));
  double delta12 = (4. * G1) / (std::pow(modI4, 2.)) - (4. * B2 * G2) / (H2 * lambdaM);

  // summand tensors for computation of elasticity tensor
  Core::LinAlg::Matrix<6, 6> IdId(Core::LinAlg::Initialization::uninitialized);
  IdId.multiply_nt(Id3v, Id3v);  // summand1 = dyad(Id,Id) in Voigt-Notation

  Core::LinAlg::Matrix<6, 6> IdMsumMId(Core::LinAlg::Initialization::uninitialized);
  IdMsumMId.multiply_nt(Id3v, Mv);
  IdMsumMId.multiply_nt(
      1.0, Mv, Id3v, 1.0);  // summand5 = dyad(Id,Md) + dyad(M,Id) in Voigt-Notation

  Core::LinAlg::Matrix<6, 6> MM(Core::LinAlg::Initialization::uninitialized);
  MM.multiply_nt(Mv, Mv);  // summand7 = dyad(M,M) in Voigt-Notation

  Core::LinAlg::Matrix<6, 6> IddI5sumdI5Id(Core::LinAlg::Initialization::uninitialized);
  IddI5sumdI5Id.multiply_nt(Id3v, modCMsumMmodCv);
  IddI5sumdI5Id.multiply_nt(
      1.0, modCMsumMmodCv, Id3v, 1.0);  // summand8 = dyad(Id,dI5) + dyad(dI5,Id)

  Core::LinAlg::Matrix<6, 6> dI5dI5(Core::LinAlg::Initialization::uninitialized);
  dI5dI5.multiply_nt(modCMsumMmodCv, modCMsumMmodCv);  // summand10 = dyad(dI5,dI5)

  Core::LinAlg::Matrix<6, 6> MdI5sumdI5M(Core::LinAlg::Initialization::uninitialized);
  MdI5sumdI5M.multiply_nt(Mv, modCMsumMmodCv);
  MdI5sumdI5M.multiply_nt(1.0, modCMsumMmodCv, Mv, 1.0);  // summand11 = dyad(M,dI5) + dyad(dI5,M)

  // fictitious elasticiy tensor
  Core::LinAlg::Matrix<6, 6> modcmat(Core::LinAlg::Initialization::uninitialized);
  modcmat.update(delta1, IdId);
  modcmat.update(delta5, IdMsumMId, 1.0);
  modcmat.update(delta7, MM, 1.0);
  modcmat.update(delta8, IddI5sumdI5Id, 1.0);
  modcmat.update(delta10, dI5dI5, 1.0);
  modcmat.update(delta11, MdI5sumdI5M, 1.0);
  Core::LinAlg::Tensor::add_elasticity_tensor_product(
      modcmat, delta12, Id3, M, 1.0);  // summand 12 = ddI5/dC^2 = Id_ik*M_jl ...
  Core::LinAlg::Tensor::add_elasticity_tensor_product(
      modcmat, delta12, M, Id3, 1.0);  // ... + M_ik*Id_jl
  modcmat.scale(std::pow(J, -4.0 / 3.0));

  // modified projection tensor Psl = Cinv o Cinv - 1/3 Cinv x Cinv
  Core::LinAlg::Matrix<6, 6> Psl(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Tensor::add_holzapfel_product(Psl, icg, 1.0);
  Psl.multiply_nt(-1.0 / 3.0, icg, icg, 1.0);

  // Right Cauchy-Green tensor in stress-like Voigt notation
  static Core::LinAlg::Matrix<6, 1> rcg_stress(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Strains::to_stress_like(rcg, rcg_stress);

  // compute the projection tensor P = II - 1/3 Cinv x C
  Core::LinAlg::Matrix<6, 6> P(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::fourth_order_identity_matrix<Core::LinAlg::Voigt::NotationType::stress,
      Core::LinAlg::Voigt::NotationType::stress>(P);
  P.multiply_nt(-1.0 / 3.0, icg, rcg_stress, 1.0);

  // compute the transpose of the projection tensor PT = II - 1/3 C x Cinv
  Core::LinAlg::Matrix<6, 6> PT(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::fourth_order_identity_matrix<Core::LinAlg::Voigt::NotationType::stress,
      Core::LinAlg::Voigt::NotationType::stress>(PT);
  PT.multiply_nt(-1.0 / 3.0, rcg_stress, icg, 1.0);

  // compute isochoric cmat from fictitious elasticiy tensor
  Core::LinAlg::Matrix<6, 6> cmatiso(Core::LinAlg::Initialization::uninitialized);
  cmatiso.multiply_nn(P, modcmat);
  cmatiso.multiply_nn(1.0, modcmat, PT, 1.0);
  cmatiso.update(2.0 / 3.0 * incJ * traceCmodS, Psl, 1.0);
  cmatiso.multiply_nt(-4.0 / 3.0, icg, S_isov, 1.0);
  cmatiso.multiply_nt(-2.0 / 3.0, S_isov, icg, 1.0);

  // update cmat
  cmat.update(1.0, cmatiso, 1.0);
}

void Mat::Elastic::IsoMuscleBlemker::evaluate_total_fiber_cauchy_stress_and_derivative(
    double lambdaM, double sigma_max_ft, double& sigma_fiber_total, double& deriv_sigma_fiber_total)
{
  // get active material parameters
  const double& sigma_max = params_->sigma_max_;
  const double& lambda_star = params_->lambda_star_;
  const double& lambda_ofl = params_->lambda_ofl_;
  const double& P1 = params_->P1_;
  const double& P2 = params_->P2_;

  // compute normalized passive fiber force and derivative w.r.t. fibre stretch
  double f_passive = Mat::Utils::Muscle::evaluate_passive_force_stretch_dependency_blemker(
      lambdaM, 1.0, lambda_star, P1, P2);
  double deriv_f_passive =
      Mat::Utils::Muscle::evaluate_derivative_passive_force_stretch_dependency_blemker(
          lambdaM, 1.0, lambda_star, P1, P2);

  // compute normalized normalized active fiber force and derivative w.r.t. fibre stretch
  double f_active =
      Mat::Utils::Muscle::evaluate_active_force_stretch_dependency_blemker(lambdaM, lambda_ofl);
  double deriv_f_active =
      Mat::Utils::Muscle::evaluate_derivative_active_force_stretch_dependency_blemker(
          lambdaM, lambda_ofl);

  sigma_fiber_total = (sigma_max * f_passive + sigma_max_ft * f_active) * lambdaM / lambda_ofl;
  deriv_sigma_fiber_total =
      (sigma_max * deriv_f_passive + sigma_max_ft * deriv_f_active) * lambdaM / lambda_ofl +
      sigma_fiber_total / lambda_ofl;
}
FOUR_C_NAMESPACE_CLOSE
