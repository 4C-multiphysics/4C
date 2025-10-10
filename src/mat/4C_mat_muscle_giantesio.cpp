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
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_muscle_utils.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*!
   * @brief Returns the fourth order tensor
   *        result_ijkl = scalar * (tensor_A_ijkm * matrix_B_mn * matrix_C_nl
   *               + matrix_A_im * tensor_B_mjkn * matrix_C_nl
   *               + matrix_A_im * matrix_B_mn * tensor_C_njkl)
   */
  Core::LinAlg::FourTensor<3> sum_multiply_tmm_mtm_mmt(const Core::LinAlg::Matrix<3, 3>& matrix_A,
      const Core::LinAlg::Matrix<3, 3>& matrix_B, const Core::LinAlg::Matrix<3, 3>& matrix_C,
      const Core::LinAlg::FourTensor<3>& tensor_A, const Core::LinAlg::FourTensor<3>& tensor_B,
      const Core::LinAlg::FourTensor<3>& tensor_C, const double& scalar)
  {
    Core::LinAlg::Matrix<3, 3> mBmC(Core::LinAlg::Initialization::uninitialized);
    mBmC.multiply_nn(matrix_B, matrix_C);

    Core::LinAlg::Matrix<3, 3> mAmB(Core::LinAlg::Initialization::uninitialized);
    mAmB.multiply_nn(matrix_A, matrix_B);

    // part1 = tensor_A * matrix_B * matrix_C
    Core::LinAlg::FourTensor<3> part1(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor_by_second_index<3>(
        part1, mBmC, tensor_A, false);

    // part2 = matrix_A * tensor_B * matrix_C
    Core::LinAlg::FourTensor<3> mAtB(true);  // matrix_A * tensor_B
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        mAtB, matrix_A, tensor_B, false);
    Core::LinAlg::FourTensor<3> part2(true);  // (matrix_A * tensor_B) * matrix_C
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor_by_second_index<3>(
        part2, matrix_C, mAtB, false);

    // part3 = matrix_A * matrix_B * tensor_C
    Core::LinAlg::FourTensor<3> part3(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        part3, mAmB, tensor_C, false);

    // result = part1 + part2 + part3
    Core::LinAlg::FourTensor<3> tAmBmCmAtBmCmAmBtC(true);
    for (unsigned i = 0; i < 3; ++i)
      for (unsigned j = 0; j < 3; ++j)
        for (unsigned k = 0; k < 3; ++k)
          for (unsigned l = 0; l < 3; ++l)
            tAmBmCmAtBmCmAmBtC(i, j, k, l) +=
                scalar * (part1(i, j, k, l) + part2(i, j, k, l) + part3(i, j, k, l));

    return tAmBmCmAtBmCmAmBtC;
  }

  /*!
   * @brief Adds the triple multiplication to the result matrix
   *        NNN_ij += scalar * left_ij middle_jk right_kl
   */
  void multiply_nnn(Core::LinAlg::Matrix<3, 3>& NNN, const double& scalar,
      const Core::LinAlg::Matrix<3, 3>& left, const Core::LinAlg::Matrix<3, 3>& middle,
      const Core::LinAlg::Matrix<3, 3>& right)
  {
    Core::LinAlg::Matrix<3, 3> NN(Core::LinAlg::Initialization::zero);
    NN.multiply_nn(left, middle);
    NNN.multiply_nn(scalar, NN, right, 1.0);
  }

  /*!
   * @brief Compute the derivative of the Cauchy Green tensor C w.r.t. C
   */
  Core::LinAlg::FourTensor<3> compute_dcdc()
  {
    Core::LinAlg::Matrix<6, 6> dCdCv(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) dCdCv(i, i) = 1.0;
    for (int i = 3; i < 6; i++) dCdCv(i, i) = 0.5;

    Core::LinAlg::FourTensor<3> dCdC(true);
    Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(dCdC, dCdCv);

    return dCdC;
  }

  /*!
   * @brief Compute the structural tensor L
   */
  Core::LinAlg::Matrix<3, 3> compute_structural_tensor_l(
      const Core::LinAlg::Matrix<3, 3>& M, const double& omega0)
  {
    Core::LinAlg::Matrix<3, 3> L(M);
    L.scale(1.0 - omega0);  // omegap*M
    for (unsigned i = 0; i < 3; ++i) L(i, i) += omega0 / 3.0;

    return L;
  }

  struct StressAndDeriv
  {
    Core::LinAlg::Matrix<3, 3> Se;
    Core::LinAlg::FourTensor<3> dSedC;
  };

  /*!
   * @brief Compute the elastic second Piola Kirchhoff stress tensor Se and its derivative dSedC
   * w.r.t. the right Cauchy Green tensor C
   */
  StressAndDeriv compute_se_and_d_sed_c(const double& alpha, const double& beta,
      const double& gamma, const double& omega0, const Core::LinAlg::Matrix<3, 3>& M,
      const Core::LinAlg::Matrix<3, 3>& invFa, const Core::LinAlg::FourTensor<3>& dinvFadC,
      const Core::LinAlg::Matrix<3, 3>& C, const Core::LinAlg::FourTensor<3>& dCdC)
  {
    // structural tensor L = omega0/3*Identity + omegap*M
    Core::LinAlg::Matrix<3, 3> L = compute_structural_tensor_l(M, omega0);
    Core::LinAlg::Matrix<6, 1> Lv(Core::LinAlg::Initialization::uninitialized);  // Voigt notation
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(L, Lv);

    // elastic right Cauchy Green tensor Ce = Fe^T Fe
    // = Fa^-T C Fa^-1 = Fa^-1 C Fa^-1 (with Fa^-1 sym.)
    Core::LinAlg::Matrix<3, 3> Ce(Core::LinAlg::Initialization::zero);
    multiply_nnn(Ce, 1.0, invFa, C, invFa);

    // derivative of Ce w.r.t C
    // dCedC_ijkl = dinvFadC_iakl (C invFa)_aj + invFa_ia dCdC_abkl invFa_bj + (invFa C)_ab
    // dinvFadC_bjkl
    Core::LinAlg::FourTensor<3> dCedC(true);
    dCedC = sum_multiply_tmm_mtm_mmt(invFa, C, invFa, dinvFadC, dCdC, dinvFadC, 1.0);
    Core::LinAlg::Matrix<6, 6> dCedCv(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(dCedCv, dCedC);

    // inverse of the elastic right Cauchy Green tensor Ce
    Core::LinAlg::Matrix<3, 3> invCe(Core::LinAlg::Initialization::zero);
    invCe.invert(Ce);
    Core::LinAlg::Matrix<6, 1> invCev(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(invCe, invCev);

    // product invCe*L
    Core::LinAlg::Matrix<3, 3> invCeL(Core::LinAlg::Initialization::zero);
    invCeL.multiply_nn(invCe, L);

    // product invCe*L*invCe
    Core::LinAlg::Matrix<3, 3> invCeLinvCe(Core::LinAlg::Initialization::zero);
    invCeLinvCe.multiply_nn(invCeL, invCe);
    Core::LinAlg::Matrix<6, 1> invCeLinvCev(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(invCeLinvCe, invCeLinvCev);

    // product Ce^T*L
    Core::LinAlg::Matrix<3, 3> transpCeL(Core::LinAlg::Initialization::zero);
    transpCeL.multiply_tn(Ce, L);

    // generalized elastic invariants including only passive properties
    double detCe = Ce.determinant();  // determinant of Ce
    // Ie = C:L = tr(C^T L) since A:B = tr(A^T B) for real matrices
    double Ie = transpCeL(0, 0) + transpCeL(1, 1) + transpCeL(2, 2);
    // Je = cof(Ce):L = tr(cof(Ce)^T L) = tr(adj(Ce) L) = tr(det(Ce) Ce^-1 L) = det(Ce)*tr(Ce^-1 L)
    double Je = detCe * (invCeL(0, 0) + invCeL(1, 1) + invCeL(2, 2));
    // exponential prefactors
    double expalpha = std::exp(alpha * (Ie - 1.0));
    double expbeta = std::exp(beta * (Je - 1.0));

    // elastic second Piola Kirchhoff stress tensor Se
    // Se = (0.5 * gamma * (expalpha * L + expbeta * (Je * invCe - invCeLinvCe)))
    Core::LinAlg::Matrix<3, 3> Se(Core::LinAlg::Initialization::zero);
    Se.update(expalpha, L);
    Se.update(-expbeta * detCe, invCeLinvCe, 1.0);
    Se.update(Je * expbeta, invCe, 1.0);
    Se.scale(0.5 * gamma);

    // derivative of Se w.r.t Ce
    // taken from Weickenmeier et al.
    Core::LinAlg::Matrix<6, 6> dSedCev(Core::LinAlg::Initialization::zero);
    dSedCev.multiply_nt(alpha * expalpha, Lv, Lv, 1.0);  // add contributions
    dSedCev.multiply_nt(beta * expbeta * std::pow(detCe, 2), invCeLinvCev, invCeLinvCev, 1.0);
    dSedCev.multiply_nt(-(beta * Je + 1.) * expbeta * detCe, invCev, invCeLinvCev, 1.0);
    dSedCev.multiply_nt(-(beta * Je + 1.) * expbeta * detCe, invCeLinvCev, invCev, 1.0);
    dSedCev.multiply_nt((beta * Je + 1.) * Je * expbeta, invCev, invCev, 1.0);
    // adds scalar * (invC boeppel invC) to cmat, see Holzapfel2000, p. 254
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(dSedCev, invCev, -Je * expbeta);
    // adds -expbeta * detCe * dinvCLinvCdCv to cmat
    Core::LinAlg::FourTensorOperations::add_derivative_of_inva_b_inva_product(
        -expbeta * detCe, invCev, invCeLinvCev, dSedCev);
    dSedCev.scale(gamma / 2);

    Core::LinAlg::FourTensor<3> dSedCe(true);
    Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(dSedCe, dSedCev);

    // derivative of Se w.r.t C
    // dSedC_ijkl = dSedCe_ijab dCedC_abkl
    Core::LinAlg::FourTensor<3> dSedC(true);
    Core::LinAlg::FourTensorOperations::multiply_four_tensor_four_tensor<3>(
        dSedC, dSedCe, dCedC, true);

    StressAndDeriv Se_dSedC = {.Se = Se, .dSedC = dSedC};

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

Mat::MuscleGiantesio::MuscleGiantesio()
    : params_(nullptr),
      lambda_m_old_(1.0),
      omegaa_old_(-1.0),
      anisotropy_(),
      anisotropy_extension_(true, 0.0, 0,
          std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
              new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {0})
{
}

Mat::MuscleGiantesio::MuscleGiantesio(Mat::PAR::MuscleGiantesio* params)
    : params_(params),
      lambda_m_old_(1.0),
      omegaa_old_(-1.0),
      anisotropy_(),
      anisotropy_extension_(true, 0.0, 0,
          std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
              new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)),
          {0})
{
  // initialize lambdaMOld_ and omegaOld_
  lambda_m_old_ = 1.0;
  omegaa_old_ = -1.0;

  // register anisotropy extension to global anisotropy
  anisotropy_.register_anisotropy_extension(anisotropy_extension_);

  // initialize fiber directions and structural tensor
  anisotropy_extension_.register_needed_tensors(
      Mat::FiberAnisotropyExtension<1>::FIBER_VECTORS |
      Mat::FiberAnisotropyExtension<1>::STRUCTURAL_TENSOR);
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

  anisotropy_extension_.pack_anisotropy(data);
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

  anisotropy_extension_.unpack_anisotropy(buffer);
}

void Mat::MuscleGiantesio::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // Read anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(fibers, coord_system);
}

void Mat::MuscleGiantesio::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, int const gp,
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

void Mat::MuscleGiantesio::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrd,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID)
{
  Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrd);
  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  stress_view.clear();
  cmat_view.clear();

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

  // compute matrices
  // right Cauchy Green tensor C
  Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(*defgrd) * *defgrd);
  const Core::LinAlg::Matrix<3, 3> C_mat = Core::LinAlg::make_matrix(Core::LinAlg::get_full(C));

  // determinant of C
  double detC = Core::LinAlg::det(C);

  // derivative of C w.r.t C
  Core::LinAlg::FourTensor<3> dCdC = compute_dcdc();

  // inverse right Cauchy Green tensor C^-1
  Core::LinAlg::Matrix<3, 3> invC(Core::LinAlg::Initialization::zero);   // matrix notation
  invC.invert(C_mat);                                                    // invC = C^-1
  Core::LinAlg::Matrix<6, 1> invCv(Core::LinAlg::Initialization::zero);  // Voigt notation
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(invC, invCv);          // invCv

  // structural tensor M, i.e. dyadic product of fibre directions
  Core::LinAlg::SymmetricTensor<double, 3, 3> M =
      anisotropy_extension_.get_structural_tensor(gp, 0);
  const Core::LinAlg::Matrix<3, 3> M_mat = Core::LinAlg::make_matrix(Core::LinAlg::get_full(M));
  double lambdaM = Mat::Utils::Muscle::fiber_stretch(C, M);
  // derivative of lambdaM w.r.t. C
  Core::LinAlg::Matrix<3, 3> dlambdaMdC =
      Mat::Utils::Muscle::d_fiber_stretch_dc(lambdaM, C_mat, M_mat);
  Core::LinAlg::Matrix<6, 1> dlambdaMdCv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(dlambdaMdC, dlambdaMdCv);

  // contraction velocity dotLambdaM
  double dotLambdaM =
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
  Core::LinAlg::Matrix<3, 3> domegaadC(dlambdaMdC);
  domegaadC.scale(omegaaAndDerivs.val_deriv_funct);
  Core::LinAlg::Matrix<6, 1> domegaadCv(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(domegaadC, domegaadCv);

  // second derivative of omegaa w.r.t. C
  // ddomegaaddC = (ddomegaaddlambdaM - 1/lambdaM domegaadlambdaM) dlambdaM_ij dlambdaM_kl
  Core::LinAlg::Matrix<6, 6> ddomegaaddCv(Core::LinAlg::Initialization::uninitialized);
  ddomegaaddCv.multiply_nt(
      omegaaAndDerivs.val_deriv_deriv_funct - omegaaAndDerivs.val_deriv_funct / lambdaM,
      dlambdaMdCv, dlambdaMdCv);

  // --------------------------------------------------------------
  // active deformation gradient Fa
  Core::LinAlg::Matrix<3, 3> Fa = act_def_grad(omegaaAndDerivs.val_funct, M_mat);

  // first derivative of Fa w.r.t. omegaa
  Core::LinAlg::Matrix<3, 3> dFadomegaa =
      d_act_def_grad_d_act_level(omegaaAndDerivs.val_funct, M_mat);

  // second derivative of Fa w.r.t. omegaa
  Core::LinAlg::Matrix<3, 3> ddFaddomegaa =
      dd_act_def_grad_dd_act_level(omegaaAndDerivs.val_funct, M_mat);

  // determinant of Fa
  double detFa = Fa.determinant();

  // --------------------------------------------------------------
  // inverse of active deformation gradient Fa^{-1}
  Core::LinAlg::Matrix<3, 3> invFa = inv_act_def_grad(Fa);

  // first derivative of Fa^{-1} w.r.t. omegaa
  Core::LinAlg::Matrix<3, 3> dinvFadomegaa = d_inv_act_def_grad_d_act_level(Fa, dFadomegaa);

  // first derivative of Fa^{-1} w.r.t. C
  // dinvFadC_ijkl = dinvFadomegaa_ij domegaadC_kl
  Core::LinAlg::FourTensor<3> dinvFadC(true);
  Core::LinAlg::FourTensorOperations::add_dyadic_product_matrix_matrix(
      dinvFadC, dinvFadomegaa, domegaadC);

  // --------------------------------------------------------------
  // elastic second Piola Kirchhoff stress tensor Se and its derivative w.r.t C
  StressAndDeriv Se_dSedC =
      compute_se_and_d_sed_c(alpha, beta, gamma, omega0, M_mat, invFa, dinvFadC, C_mat, dCdC);

  // --------------------------------------------------------------
  // compute second Piola Kirchhoff stress components S1, S2, Svol
  // note: S3 = invF * ddetFadF * Psie = 0 since ddetFawa = 0 and ddetFadF = ddetFadwa dwadF

  // S1 = detFa * invFa * Se * invFa^T
  // in index notation: S1_sl = detFa * invFa_sr * Se_rq * invFa_rl (with invFa sym.)
  Core::LinAlg::Matrix<3, 3> S1(Core::LinAlg::Initialization::zero);
  multiply_nnn(S1, detFa, invFa, Se_dSedC.Se, invFa);
  Core::LinAlg::Matrix<6, 1> S1v(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(S1, S1v);

  // S2 = detFa * invF * dPsiedFe * (F * dinvFadF)  = 2 detFa (C Fa^-1 Se) dFa^-1dC
  // or: S2 = -2 S1 : (C Fa^-1 dFadomegaa) domegaadC
  // in index notation: S2_sl = -2 S1_ij (C Fa^-1 dFadomegaa)_ij domegaadC_sl
  Core::LinAlg::Matrix<3, 3> CinvFadFadomegaa(Core::LinAlg::Initialization::zero);
  multiply_nnn(CinvFadFadomegaa, 1.0, C_mat, invFa, dFadomegaa);

  Core::LinAlg::Matrix<6, 1> S2v(domegaadCv);
  double scalar_contraction =
      -2.0 * Core::LinAlg::FourTensorOperations::contract_matrix_matrix(CinvFadFadomegaa, S1);
  S2v.scale(scalar_contraction);

  // Svol = - gamma/2 * detC^-kappa
  Core::LinAlg::Matrix<6, 1> Svolv(invCv);
  Svolv.scale(-0.5 * gamma * std ::pow(detC, -kappa));

  // --------------------------------------------------------------
  // compute material tangent components cmat1, cmat2, cmatvol

  // compute cmat1 = 2 dS1dC
  Core::LinAlg::Matrix<6, 6> cmat1v(Core::LinAlg::Initialization::zero);

  // derivative of S1 = detFa * invFa * Se * invFa^T  w.r.t. C
  // = detFa * (dinvFadC * (Se * dinvFa^T) + invFa * dSedC * invFa^T + (invFa * Se) * dinvFadC)
  Core::LinAlg::FourTensor<3> dS1dC(true);
  dS1dC = sum_multiply_tmm_mtm_mmt(
      invFa, Se_dSedC.Se, invFa, dinvFadC, Se_dSedC.dSedC, dinvFadC, detFa);
  Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(cmat1v, dS1dC);
  cmat1v.scale(2.0);

  // compute cmat2 = 2 dS2dC
  Core::LinAlg::Matrix<6, 6> cmat2v(Core::LinAlg::Initialization::zero);

  // derivative of S2 w.r.t. C
  // in index notation: S2_sl = -2 S1_ij (C Fa^-1 dFadomegaa)_ij domegaadC_sl
  // dS2dC_slpq = -4 [ S1_ij d(C Fa^-1 dFadomegaa)dC_ijpq domegaadC_sl +
  //                 [ S1_ij (C Fa^-1 dFadomegaa)_ij ddomegaaddC_slpq ]
  // dS2dC_slpq =  2 [ dscalardC_pq domegaadC_sl + [ scalar_contraction ddomegaaddC_slpq ]

  // derivative of the scalar (-2 S1_ij (C Fa^-1 dFadomegaa)_ij) w.r.t. C
  // dscalardC_pq = -2 (dS1dC_ijpq (C invFa dFadomegaa)_ij + S1_ij d(C invFa dFadomegaa)dC_ijpq )
  Core::LinAlg::Matrix<3, 3> H(Core::LinAlg::Initialization::zero);
  H.multiply_nn(dinvFadomegaa, dFadomegaa);
  H.multiply_nn(1.0, invFa, ddFaddomegaa, 1.0);
  Core::LinAlg::Matrix<3, 3> CH(Core::LinAlg::Initialization::zero);
  CH.multiply_nn(C_mat, H);
  double helper = Core::LinAlg::FourTensorOperations::contract_matrix_matrix(S1, CH);

  Core::LinAlg::Matrix<3, 3> dscalardC(domegaadC);
  dscalardC.scale(helper);
  multiply_nnn(dscalardC, 1.0, S1, dFadomegaa, invFa);
  Core::LinAlg::FourTensorOperations::add_contraction_matrix_four_tensor(
      dscalardC, CinvFadFadomegaa, dS1dC);
  Core::LinAlg::Matrix<6, 1> dscalardCv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(
      dscalardC, dscalardCv);  // not yet scaled with -2.0

  // compute cmat2 = 2 dS2dC = 2 [domegaadC_sl dscalardC_pq + scalar * ddomegaaddC_slpq]
  cmat2v.multiply_nt(-4.0, domegaadCv, dscalardCv);            // 2 * dscalardC_pq domegaadC_sl
  cmat2v.update(2.0 * scalar_contraction, ddomegaaddCv, 1.0);  // 2 * scalar * ddomegaaddC_slpq

  // compute cmatvol = 2 dS2dC
  Core::LinAlg::Matrix<6, 6> cmatvolv(Core::LinAlg::Initialization::zero);
  cmatvolv.multiply_nt(gamma * kappa * std::pow(detC, -kappa), invCv, invCv, 1.0);
  Core::LinAlg::FourTensorOperations::add_holzapfel_product(
      cmatvolv, invCv, gamma * std::pow(detC, -kappa));

  // --------------------------------------------------------------
  // update constituent stress and material tangent
  stress_view.update(1.0, S1v, 1.0);
  stress_view.update(1.0, S2v, 1.0);
  stress_view.update(1.0, Svolv, 1.0);

  cmat_view.update(1.0, cmat1v, 1.0);
  cmat_view.update(1.0, cmat2v, 1.0);
  cmat_view.update(1.0, cmatvolv, 1.0);
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

Core::LinAlg::Matrix<3, 3> Mat::MuscleGiantesio::act_def_grad(
    const double omegaa, const Core::LinAlg::Matrix<3, 3>& M)
{
  // active deformation gradient Fa
  Core::LinAlg::Matrix<3, 3> Fa(M);
  Fa.scale(1 - omegaa - std::pow(1. - omegaa, -0.5));
  for (unsigned j = 0; j < 3; ++j) Fa(j, j) += std::pow(1. - omegaa, -0.5);
  return Fa;
}

Core::LinAlg::Matrix<3, 3> Mat::MuscleGiantesio::d_act_def_grad_d_act_level(
    const double omegaa, const Core::LinAlg::Matrix<3, 3>& M)
{
  // first derivative of Fa w.r.t. omegaa
  Core::LinAlg::Matrix<3, 3> dFadomegaa(M);
  if (omegaa != 0)
  {
    dFadomegaa.scale(-1.0 - 0.5 * std::pow(1. - omegaa, -1.5));
    for (unsigned j = 0; j < 3; ++j) dFadomegaa(j, j) += 0.5 * std::pow(1. - omegaa, -1.5);
  }
  else
  {
    dFadomegaa.clear();
  }
  return dFadomegaa;
}

Core::LinAlg::Matrix<3, 3> Mat::MuscleGiantesio::dd_act_def_grad_dd_act_level(
    const double omegaa, const Core::LinAlg::Matrix<3, 3>& M)
{
  // second derivative of Fa w.r.t. omegaa
  Core::LinAlg::Matrix<3, 3> ddFaddomegaa(M);
  if (omegaa != 0)
  {
    ddFaddomegaa.scale(-0.75 * std::pow(1. - omegaa, -2.5));
    for (unsigned j = 0; j < 3; ++j) ddFaddomegaa(j, j) += 0.75 * std::pow(1. - omegaa, -2.5);
  }
  else
  {
    ddFaddomegaa.clear();
  }
  return ddFaddomegaa;
}

Core::LinAlg::Matrix<3, 3> Mat::MuscleGiantesio::inv_act_def_grad(
    const Core::LinAlg::Matrix<3, 3>& Fa)
{
  // inverse of active deformation gradient Fa^{-1}
  Core::LinAlg::Matrix<3, 3> invFa(Core::LinAlg::Initialization::zero);
  invFa.invert(Fa);
  return invFa;
}

Core::LinAlg::Matrix<3, 3> Mat::MuscleGiantesio::d_inv_act_def_grad_d_act_level(
    const Core::LinAlg::Matrix<3, 3>& Fa, const Core::LinAlg::Matrix<3, 3>& dFadomegaa)
{
  Core::LinAlg::Matrix<3, 3> invFa = inv_act_def_grad(Fa);

  // first derivative of Fa^{-1} w.r.t. omegaa
  // dinvFadomegaa_ij = - invFa_ik dFadomegaa_kl invFa_lj
  Core::LinAlg::Matrix<3, 3> dinvFadomegaa(Core::LinAlg::Initialization::zero);
  multiply_nnn(dinvFadomegaa, -1.0, invFa, dFadomegaa, invFa);

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
