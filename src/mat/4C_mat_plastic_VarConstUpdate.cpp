// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_plastic_VarConstUpdate.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

using vmap = Core::LinAlg::Voigt::IndexMappings;

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::PlasticElastHyperVCU::PlasticElastHyperVCU(const Core::Mat::PAR::Parameter::Data& matdata)
    : Mat::PAR::PlasticElastHyper(matdata)
{
  // polyconvexity check is just implemented for isotropic hyperlastic materials
  if (polyconvex_)
    FOUR_C_THROW(
        "This polyconvexity-check is just implemented for isotropic "
        "hyperelastic-materials (do not use for plastic materials).");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::PlasticElastHyperVCU::create_material()
{
  return std::make_shared<Mat::PlasticElastHyperVCU>(this);
}


Mat::PlasticElastHyperVCUType Mat::PlasticElastHyperVCUType::instance_;


Core::Communication::ParObject* Mat::PlasticElastHyperVCUType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::PlasticElastHyperVCU* elhy = new Mat::PlasticElastHyperVCU();
  elhy->unpack(buffer);

  return elhy;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PlasticElastHyperVCU::PlasticElastHyperVCU() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PlasticElastHyperVCU::PlasticElastHyperVCU(Mat::PAR::PlasticElastHyperVCU* params)
    : params_(params)
{
  // make sure the referenced materials in material list have quick access parameters
  std::vector<int>::const_iterator m;
  for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
  {
    const int matid = *m;
    std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
    if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
    potsum_.push_back(sum);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyperVCU::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (mat_params() != nullptr) matid = mat_params()->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
  summandProperties_.pack(data);

  if (mat_params() != nullptr)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsum_.size(); ++p)
    {
      potsum_[p]->pack_summand(data);
    }
  }

  // plastic history data
  add_to_pack(data, last_plastic_defgrd_inverse_);
  add_to_pack(data, last_alpha_isotropic_);

  // insert current iteration states
  add_to_pack(data, plastic_defgrd_inverse_);
  add_to_pack(data, delta_alpha_i_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyperVCU::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  params_ = nullptr;
  potsum_.clear();

  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover MatParams()
  int matid;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const unsigned int probinst =
          Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::PlasticElastHyperVCU*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  summandProperties_.unpack(buffer);

  if (mat_params() != nullptr)  // summands are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = mat_params()->matids_.begin(); m != mat_params()->matids_.end(); ++m)
    {
      const int matid = *m;
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      potsum_.push_back(sum);
    }

    // loop map of associated potential summands
    for (unsigned int p = 0; p < potsum_.size(); ++p)
    {
      potsum_[p]->unpack_summand(buffer);
    }
  }

  // plastic history data
  extract_from_pack(buffer, last_plastic_defgrd_inverse_);
  extract_from_pack(buffer, last_alpha_isotropic_);

  // current iteration states
  extract_from_pack(buffer, plastic_defgrd_inverse_);
  extract_from_pack(buffer, delta_alpha_i_);

  // in the postprocessing mode, we do not unpack everything we have packed
  // -> position check cannot be done in this case
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::PlasticElastHyperVCU::setup(int numgp, const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // setup the plasticelasthyper data
  PlasticElastHyper::setup(numgp, fibers, coord_system);

  // setup history
  plastic_defgrd_inverse_.resize(numgp);

  return;
}

// MAIN
void Mat::PlasticElastHyperVCU::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, const EvaluationContext& context,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)  ///< Element GID
{
  const Core::LinAlg::Matrix<3, 3> defgrd_mat = Core::LinAlg::make_matrix_view(*defgrad);

  Core::LinAlg::Matrix<6, 1> stress_view = Core::LinAlg::make_stress_like_voigt_view(stress);
  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  double last_ai = last_alpha_isotropic_[gp];
  Core::LinAlg::Matrix<3, 3> empty;

  // Get cetrial
  Core::LinAlg::Matrix<3, 3> id2;
  for (int i = 0; i < 3; i++) id2(i, i) = 1.0;

  Core::LinAlg::Matrix<3, 3> cetrial;
  Core::LinAlg::SymmetricTensor<double, 3, 3> ee_test;
  comp_elast_quant(&defgrd_mat, last_plastic_defgrd_inverse_[gp], id2, &cetrial, ee_test);

  // get 2pk stresses
  Core::LinAlg::SymmetricTensor<double, 3, 3> etstr;
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> etcmat;
  ElastHyper::evaluate(nullptr, ee_test, params, context, etstr, etcmat, gp, eleGID);

  double yf;
  double normZero = 0.0;

  Core::LinAlg::Matrix<3, 3> mandelStr;
  Core::LinAlg::Matrix<3, 3> devMandelStr;
  yield_function(last_ai, normZero, id2, cetrial, Core::LinAlg::make_stress_like_voigt_view(etstr),
      &yf, devMandelStr, mandelStr);

  if (yf <= 0)
  {
    // step is elastic
    stress_view.clear();
    cmat_view.clear();

    Core::LinAlg::SymmetricTensor<double, 3, 3> checkStr;
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> checkCmat;
    Core::LinAlg::Matrix<3, 3> emptymat;
    PlasticElastHyper::evaluate_elast(&defgrd_mat, &emptymat, &stress_view, &cmat_view, gp, eleGID);
    ElastHyper::evaluate(defgrad, ee_test, params, context, checkStr, checkCmat, gp, eleGID);

    // push back
    const Core::LinAlg::Matrix<3, 3> checkStrMat =
        Core::LinAlg::make_matrix(Core::LinAlg::get_full(checkStr));

    Core::LinAlg::Matrix<3, 3> tmp33;
    Core::LinAlg::Matrix<3, 3> strWithPlast;
    tmp33.multiply(last_plastic_defgrd_inverse_[gp], checkStrMat);
    strWithPlast.multiply_nt(tmp33, last_plastic_defgrd_inverse_[gp]);

    plastic_defgrd_inverse_[gp] = last_plastic_defgrd_inverse_[gp];
    delta_alpha_i_[gp] = 0.;
  }

  else
  {
    int iterator = 0;
    const int maxiter = 200;

    Core::LinAlg::Matrix<3, 3> devMandelStr_direction(devMandelStr);
    devMandelStr_direction.scale(1. / devMandelStr.norm2());

    Core::LinAlg::Matrix<5, 5> iH;

    double plastMulti = 1.e-8;
    Core::LinAlg::Matrix<3, 3> dLpStart;
    dLpStart.update(plastMulti, devMandelStr_direction);

    Core::LinAlg::Matrix<5, 1> beta;
    beta(0) = dLpStart(0, 0);
    beta(1) = dLpStart(1, 1);
    beta(2) = dLpStart(0, 1);
    beta(3) = dLpStart(1, 2);
    beta(4) = dLpStart(0, 2);

    Core::LinAlg::Matrix<3, 3> dLp;
    bool converged = false;

    Core::LinAlg::Matrix<5, 5> hess;
    Core::LinAlg::Matrix<5, 1> rhs;
    Core::LinAlg::Matrix<6, 6> dcedlp;
    Core::LinAlg::Matrix<9, 6> dFpiDdeltaDp;
    Core::LinAlg::Matrix<6, 5> P;
    P(0, 0) = 1.;
    P(1, 1) = 1.;
    P(2, 0) = -1.;
    P(2, 1) = -1.;
    P(3, 2) = 1.;
    P(4, 3) = 1.;
    P(5, 4) = 1.;

    do
    {
      iterator++;

      dLp(0, 0) = beta(0);
      dLp(1, 1) = beta(1);
      dLp(2, 2) = -beta(0) - beta(1);
      dLp(0, 1) = dLp(1, 0) = beta(2);
      dLp(2, 1) = dLp(1, 2) = beta(3);
      dLp(0, 2) = dLp(2, 0) = beta(4);

      Core::LinAlg::Matrix<5, 1> rhsElast;
      Core::LinAlg::SymmetricTensor<double, 3, 3> eeOut;
      evaluate_rhs(
          gp, dLp, defgrd_mat, eeOut, rhs, rhsElast, dcedlp, dFpiDdeltaDp, params, context, eleGID);

      // Hessian matrix elastic component
      Core::LinAlg::SymmetricTensor<double, 3, 3> elastStress;
      Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> elastCmat;
      Core::LinAlg::Matrix<6, 1> elastStress_view =
          Core::LinAlg::make_stress_like_voigt_view(elastStress);
      Core::LinAlg::Matrix<6, 6> elastCmat_view =
          Core::LinAlg::make_stress_like_voigt_view(elastCmat);
      Core::LinAlg::Matrix<6, 1> elastStressDummy;
      Core::LinAlg::Matrix<6, 6> elastCmatDummy;
      ElastHyper::evaluate(nullptr, eeOut, params, context, elastStress, elastCmat, gp, eleGID);

      Core::LinAlg::Matrix<6, 6> d2ced2lpVoigt[6];
      ce2nd_deriv(&defgrd_mat, last_plastic_defgrd_inverse_[gp], dLp, d2ced2lpVoigt);

      Core::LinAlg::Matrix<6, 6> cpart_tmp;
      cpart_tmp.multiply(elastCmat_view, dcedlp);
      Core::LinAlg::Matrix<6, 6> cpart;
      cpart.multiply_tn(.25, dcedlp, cpart_tmp);

      Core::LinAlg::Matrix<6, 6> spart6x6;
      for (int A = 0; A < 6; ++A)
        for (int B = 0; B < 6; ++B)
          for (int C = 0; C < 6; ++C)
            if (A < 3)
              spart6x6(B, C) += 0.5 * elastStress_view(A) * d2ced2lpVoigt[A](B, C);
            else
              spart6x6(B, C) += 1. * elastStress_view(A) * d2ced2lpVoigt[A](B, C);

      Core::LinAlg::Matrix<6, 6> hessElast6x6(spart6x6);
      hessElast6x6.update(1., cpart, 1.);

      Core::LinAlg::Matrix<6, 5> tmp65;
      tmp65.multiply(hessElast6x6, P);
      Core::LinAlg::Matrix<5, 5> hessElast;
      hessElast.multiply_tn(P, tmp65);

      Core::LinAlg::Matrix<5, 1> dlp_vec;
      dlp_vec(0) = 2. * beta(0) + beta(1);
      dlp_vec(1) = 2. * beta(1) + beta(0);
      dlp_vec(2) = 2. * beta(2);
      dlp_vec(3) = 2. * beta(3);
      dlp_vec(4) = 2. * beta(4);

      // dissipative component
      Core::LinAlg::Matrix<5, 5> hess_a;
      for (int i = 0; i < 5; ++i) hess_a(i, i) = 2.;
      hess_a(0, 1) = 1.;
      hess_a(1, 0) = 1.;
      Core::LinAlg::Matrix<5, 5> hess_aiso(hess_a);
      hess_a.scale(1. / dLp.norm2());
      Core::LinAlg::Matrix<5, 5> tmpSummandIdentity(hess_a);
      double hess_aisoScalar = isohard();
      hess_aisoScalar *= last_alpha_isotropic_[gp] / dLp.norm2();
      hess_aisoScalar += isohard();
      hess_aiso.scale(hess_aisoScalar);

      hess_a.multiply_nt((-1.) / (dLp.norm2() * dLp.norm2() * dLp.norm2()), dlp_vec, dlp_vec, 1.);
      Core::LinAlg::Matrix<5, 5> tmpSummandDlpVec;
      tmpSummandDlpVec.multiply_nt(
          (-1.) / (dLp.norm2() * dLp.norm2() * dLp.norm2()), dlp_vec, dlp_vec);
      tmpSummandDlpVec.scale(sqrt(2. / 3.) * inityield());
      tmpSummandIdentity.scale(sqrt(2. / 3.) * inityield());
      hess_a.scale(sqrt(2. / 3.) * inityield());

      Core::LinAlg::Matrix<5, 5> tmp55;
      tmp55.multiply_nt(dlp_vec, dlp_vec);
      tmp55.scale((sqrt(2. / 3.) * last_alpha_isotropic_[gp] * isohard()) /
                  (dLp.norm2() * dLp.norm2() * dLp.norm2()));

      hess_aiso.update(-1., tmp55, 1.);
      hess_aiso.scale(sqrt(2. / 3.));

      // Hessian matrix for nonlinear iso hardening
      Core::LinAlg::Matrix<5, 5> hessIsoNL;
      for (int i = 0; i < 5; ++i) hessIsoNL(i, i) = 2.;
      hessIsoNL(0, 1) = 1.;
      hessIsoNL(1, 0) = 1.;
      double hessIsoNLscalar = isohard();
      double new_ai = last_alpha_isotropic_[gp] + sqrt(2. / 3.) * dLp.norm2();
      double k = infyield() - inityield();
      hessIsoNLscalar *= new_ai;
      hessIsoNLscalar += k;
      hessIsoNLscalar -= k * std::exp(-1. * expisohard() * new_ai);
      hessIsoNL.scale(hessIsoNLscalar / dLp.norm2());
      Core::LinAlg::Matrix<5, 5> tmpNL55;
      tmpNL55.multiply_nt(dlp_vec, dlp_vec);
      Core::LinAlg::Matrix<5, 5> tmp2(tmpNL55);
      tmpNL55.scale(hessIsoNLscalar / (dLp.norm2() * dLp.norm2() * dLp.norm2()));
      hessIsoNL.update(-1., tmpNL55, 1.);
      double tmp2scalar = isohard();
      tmp2scalar += expisohard() * k * std::exp(-1. * expisohard() * new_ai);
      tmp2.scale((sqrt(2. / 3.) * tmp2scalar) / (dLp.norm2() * dLp.norm2()));
      hessIsoNL.update(1., tmp2, 1.);
      hessIsoNL.scale(sqrt(2. / 3.));

      // Compose the hessian matrix
      Core::LinAlg::Matrix<5, 5> hess_analyt(hessElast);
      hess_analyt.update(1., hess_a, 1.);
      hess_analyt.update(1., hessIsoNL, 1.);  // For nonlinear isotropic hardening

      if (rhs.norm2() < 1.0e-12) converged = true;

      iH = hess_analyt;
      Core::LinAlg::FixedSizeSerialDenseSolver<5, 5, 1> solver;
      solver.set_matrix(iH);
      if (solver.invert() != 0) FOUR_C_THROW("Inversion failed");

      Core::LinAlg::Matrix<5, 1> beta_incr;
      beta_incr.multiply(-1., iH, rhs, 0.);

      beta.update(1., beta_incr, 1.);

    } while (!converged && (iterator < maxiter));

    if (!converged)
    {
      std::cout << "eleGID: " << eleGID << "gp: " << gp << std::endl;
      FOUR_C_THROW("unconverged");
    }

    dLp(0, 0) = beta(0);
    dLp(1, 1) = beta(1);
    dLp(2, 2) = -beta(0) - beta(1);
    dLp(0, 1) = dLp(1, 0) = beta(2);
    dLp(2, 1) = dLp(1, 2) = beta(3);
    dLp(0, 2) = dLp(2, 0) = beta(4);

    // Get exp, Dexp and DDexp
    Core::LinAlg::Matrix<3, 3> input_dLp(dLp);
    input_dLp.scale(-1.);
    Core::LinAlg::MatrixFunctErrorType exp_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    Core::LinAlg::Matrix<3, 3> expOut = Core::LinAlg::matrix_exp(input_dLp, exp_err_status);
    FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix exponential evaluation failed!");
    Core::LinAlg::Matrix<6, 6> dexpOut_mat =
        Core::LinAlg::sym_matrix_3x3_exp_1st_deriv(input_dLp, exp_err_status);
    FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix exponential derivative evaluation failed!");

    plastic_defgrd_inverse_[gp].multiply(last_plastic_defgrd_inverse_[gp], expOut);
    delta_alpha_i_[gp] = sqrt(2. / 3.) * dLp.norm2();

    // Compute the total stresses
    Core::LinAlg::Matrix<6, 6> tangent_elast;
    PlasticElastHyper::evaluate_elast(&defgrd_mat, &dLp, &stress_view, &tangent_elast, gp, eleGID);

    Core::LinAlg::Matrix<6, 9> dPK2dFpinvIsoprinc;
    dpk2d_fpi(gp, eleGID, &defgrd_mat, &plastic_defgrd_inverse_[gp], dPK2dFpinvIsoprinc);

    Core::LinAlg::Matrix<6, 6> mixedDeriv;
    mixedDeriv.multiply(dPK2dFpinvIsoprinc, dFpiDdeltaDp);


    Core::LinAlg::Matrix<6, 5> tmp2;
    tmp2.multiply(mixedDeriv, P);

    Core::LinAlg::Matrix<6, 5> mixDerivInvHess;
    mixDerivInvHess.multiply(tmp2, iH);

    Core::LinAlg::Matrix<6, 6> cmat_summand1;
    cmat_summand1.update(1., tangent_elast);
    Core::LinAlg::Matrix<6, 6> cmat_summand2;
    cmat_summand2.multiply_nt(-1., mixDerivInvHess, tmp2);

    cmat_view.update(cmat_summand1, cmat_summand2);
  }
}


/// update after converged time step
void Mat::PlasticElastHyperVCU::update()
{
  // update local history data F_n <-- F_{n+1}
  for (unsigned gp = 0; gp < last_plastic_defgrd_inverse_.size(); ++gp)
  {
    // for a real plastic update, update the plastic history here
    last_plastic_defgrd_inverse_[gp] = plastic_defgrd_inverse_[gp];
    last_alpha_isotropic_[gp] += delta_alpha_i_[gp];
  }

  return;
};



// Evaluate dCedlp
void Mat::PlasticElastHyperVCU::eval_dce_dlp(const Core::LinAlg::Matrix<3, 3> fpi,
    const Core::LinAlg::Matrix<3, 3>* defgrd, const Core::LinAlg::Matrix<6, 6> Dexp,
    const Core::LinAlg::Matrix<3, 3> cetrial, const Core::LinAlg::Matrix<3, 3> explp,
    Core::LinAlg::Matrix<6, 6>& dceDdeltalp, Core::LinAlg::Matrix<9, 6>& dFpiDdeltaDp)
{
  // compute dcedlp this way: dcedlp = dcedfpi : dfpidlp
  // first compute dcedfpi
  Core::LinAlg::Matrix<3, 3> id2;
  for (int i = 0; i < 3; i++) id2(i, i) += 1.0;
  Core::LinAlg::Matrix<3, 3> next_fpi;
  next_fpi.multiply(fpi, explp);
  Core::LinAlg::Matrix<3, 3> rcg;
  rcg.multiply_tn(*defgrd, *defgrd);
  Core::LinAlg::Matrix<3, 3> inputB;
  inputB.multiply(next_fpi, rcg);

  Core::LinAlg::Matrix<6, 9> dcedfpi;

  Core::LinAlg::Matrix<3, 3> tmp;
  tmp.multiply_tn(next_fpi, rcg);
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dcedfpi, tmp, id2, 1.0);

  // Derivative of inverse plastic deformation gradient
  dFpiDdeltaDp.clear();
  for (int A = 0; A < 3; A++)
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        for (int i = 0; i < 6; i++)
          if (i <= 2)
            dFpiDdeltaDp(vmap::non_symmetric_tensor_to_voigt9_index(A, a), i) -=
                fpi(A, b) * Dexp(vmap::symmetric_tensor_to_voigt6_index(b, a), i);
          else
            dFpiDdeltaDp(vmap::non_symmetric_tensor_to_voigt9_index(A, a), i) -=
                2. * fpi(A, b) *
                Dexp(vmap::symmetric_tensor_to_voigt6_index(b, a), i);  // fixme factor 2

  dceDdeltalp.multiply(dcedfpi, dFpiDdeltaDp);

  for (int i = 3; i < 6; ++i)
    for (int j = 0; j < 6; ++j) dceDdeltalp(i, j) *= 2.;
}


// MAP 5x1 on 9x1
void Mat::PlasticElastHyperVCU::yield_function(const double last_ai,
    const double norm_dLp,                     // this is zero when yf<0 is checked
    const Core::LinAlg::Matrix<3, 3> ExpEqui,  // this is identity when yf < 0 is checked
    const Core::LinAlg::Matrix<3, 3> cetr, const Core::LinAlg::Matrix<6, 1> str, double* yieldFunc,
    Core::LinAlg::Matrix<3, 3>& devMandelStr, Core::LinAlg::Matrix<3, 3>& MandelStr)
{
  double Qi = 0.0;
  double new_ai = last_ai + norm_dLp * sqrt(2. / 3.);  // fixme sqrt
  double k = infyield() - inityield();
  Qi -= isohard() * new_ai;
  Qi -= k;
  Qi += k * std::exp(-1. * expisohard() * new_ai);
  Qi *= sqrt(2. / 3.);

  double Qeq = sqrt(2. / 3.) * inityield();

  Core::LinAlg::Matrix<3, 3> ce;
  Core::LinAlg::Matrix<3, 3> tmp;
  tmp.multiply(cetr, ExpEqui);
  ce.multiply(ExpEqui, tmp);

  // Compute the deviator of mandelStr, then its norm
  Core::LinAlg::Matrix<3, 3> devMandelStrSumm2;

  // se strainlike
  Core::LinAlg::Matrix<6, 1> se_strain(str);
  for (int i = 3; i < 6; i++) se_strain(i) *= 2.;

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        MandelStr(i, k) += ce(i, j) * str(vmap::symmetric_tensor_to_voigt6_index(j, k));

  // Compute the trace of the mandel stresses
  Core::LinAlg::Matrix<3, 3> id2;
  for (int i = 0; i < 3; i++) id2(i, i) = 1.0;
  double trMandelStr = 0.0;
  for (int i = 0; i < 3; i++) trMandelStr += MandelStr(i, i);
  devMandelStrSumm2.update(-trMandelStr / 3.0, id2);
  devMandelStr.update(MandelStr, devMandelStrSumm2);

  double NormDevMandelStr = devMandelStr.norm2();

  // Compose the yield function
  double yf = 0.0;
  yf = NormDevMandelStr + Qi - Qeq;
  *yieldFunc = yf;

  return;
}


// Get elastic quantities Cetrial and Ee_n+1
void Mat::PlasticElastHyperVCU::comp_elast_quant(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<3, 3> fpi, const Core::LinAlg::Matrix<3, 3> MatExp,
    Core::LinAlg::Matrix<3, 3>* cetrial, Core::LinAlg::SymmetricTensor<double, 3, 3>& Ee)

{
  Core::LinAlg::Matrix<3, 3> fetrial;
  fetrial.multiply(*defgrd, fpi);
  Core::LinAlg::Matrix<3, 3> cetr;
  cetr.multiply_tn(fetrial, fetrial);
  *cetrial = cetr;

  Core::LinAlg::Matrix<3, 3> tmp;
  tmp.multiply(MatExp, cetr);
  Core::LinAlg::Matrix<3, 3> next_ce;
  next_ce.multiply(tmp, MatExp);

  // Compute Ee_n+1 (first in 3x3 then map to 6x1 VOIGT)
  Core::LinAlg::Matrix<3, 3> id2int;
  for (int i = 0; i < 3; i++) id2int(i, i) = 1.0;
  Core::LinAlg::Matrix<3, 3> next_ee3x3;
  next_ee3x3.update(0.5, next_ce, -0.5, id2int);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) Ee(i, j) = next_ee3x3(i, j);

  return;
}

/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void Mat::PlasticElastHyperVCU::vis_names(std::map<std::string, int>& names) const
{
  std::string accumulatedstrain = "accumulatedstrain";
  names[accumulatedstrain] = 1;  // scalar

}  // vis_names()


/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool Mat::PlasticElastHyperVCU::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID) const
{
  if (name == "accumulatedstrain")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");
    double tmp = 0.;
    for (unsigned gp = 0; gp < last_alpha_isotropic_.size(); gp++) tmp += accumulated_strain(gp);
    data[0] = tmp / last_alpha_isotropic_.size();
  }
  return false;

}  // vis_data()


void Mat::PlasticElastHyperVCU::evaluate_rhs(const int gp, const Core::LinAlg::Matrix<3, 3> dLp,
    const Core::LinAlg::Matrix<3, 3> defgrd, Core::LinAlg::SymmetricTensor<double, 3, 3>& eeOut,
    Core::LinAlg::Matrix<5, 1>& rhs, Core::LinAlg::Matrix<5, 1>& rhsElast,
    Core::LinAlg::Matrix<6, 6>& dcedlp, Core::LinAlg::Matrix<9, 6>& dFpiDdeltaDp,
    const Teuchos::ParameterList& params, const EvaluationContext& context, const int eleGID)
{
  Core::LinAlg::Matrix<3, 3> zeros;
  Core::LinAlg::Matrix<6, 6> zeros66;
  // set zero
  rhs.clear();

  // Get exp, Dexp and DDexp
  Core::LinAlg::Matrix<3, 3> dLpIn(dLp);
  dLpIn.scale(-1.);
  Core::LinAlg::MatrixFunctErrorType exp_err_status{Core::LinAlg::MatrixFunctErrorType::no_errors};
  Core::LinAlg::Matrix<3, 3> expOut = Core::LinAlg::matrix_exp(dLpIn, exp_err_status);
  FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
      "Matrix exponential evaluation failed!");
  Core::LinAlg::Matrix<6, 6> dexpOut_mat =
      Core::LinAlg::sym_matrix_3x3_exp_1st_deriv(dLpIn, exp_err_status);
  FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
      "Matrix exponential derivative evaluation failed!");

  Core::LinAlg::Matrix<3, 3> fpi_incr(dLp);
  fpi_incr.scale(-1.);
  Core::LinAlg::MatrixFunctErrorType exp_deriv_err_status{
      Core::LinAlg::MatrixFunctErrorType::no_errors};
  Core::LinAlg::Matrix<6, 6> derivExpMinusLP =
      Core::LinAlg::sym_matrix_3x3_exp_1st_deriv(fpi_incr, exp_deriv_err_status);
  FOUR_C_ASSERT_ALWAYS(exp_deriv_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
      "Matrix exponential derivative evaluation failed!");

  Core::LinAlg::Matrix<3, 3> fetrial;
  fetrial.multiply(defgrd, last_plastic_defgrd_inverse_[gp]);
  Core::LinAlg::Matrix<3, 3> cetrial;
  cetrial.multiply_tn(fetrial, fetrial);

  Core::LinAlg::Matrix<3, 3> fpi;
  fpi.multiply(last_plastic_defgrd_inverse_[gp], expOut);
  Core::LinAlg::Matrix<3, 3> fe;
  fe.multiply(defgrd, fpi);
  Core::LinAlg::Matrix<3, 3> ce;
  ce.multiply_tn(fe, fe);

  for (int i = 0; i < 3; ++i) eeOut(i, i) = 0.5 * (ce(i, i) - 1.);
  eeOut(0, 1) = 0.5 * ce(0, 1);
  eeOut(1, 2) = 0.5 * ce(1, 2);
  eeOut(0, 2) = 0.5 * ce(0, 2);

  Core::LinAlg::SymmetricTensor<double, 3, 3> se;
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dummy;
  ElastHyper::evaluate(nullptr, eeOut, params, context, se, dummy, gp, eleGID);

  eval_dce_dlp(last_plastic_defgrd_inverse_[gp], &defgrd, dexpOut_mat, cetrial, expOut, dcedlp,
      dFpiDdeltaDp);

  Core::LinAlg::Matrix<6, 1> rhs6;

  Core::LinAlg::Matrix<6, 1> dLp_vec;
  for (int i = 0; i < 3; ++i) dLp_vec(i) = dLp(i, i);
  dLp_vec(3) = 2. * dLp(0, 1);
  dLp_vec(4) = 2. * dLp(1, 2);
  dLp_vec(5) = 2. * dLp(0, 2);


  double new_ai = last_alpha_isotropic_[gp] + sqrt(2. / 3.) * dLp.norm2();
  double k = infyield() - inityield();
  double rhsPlastScalar = isohard();
  rhsPlastScalar *= new_ai;
  rhsPlastScalar += k;
  rhsPlastScalar -= k * std::exp(-1. * expisohard() * new_ai);
  rhs6.update((rhsPlastScalar * sqrt(2. / 3.)) / dLp.norm2(), dLp_vec,
      1.);  // plastic component, nonlinear iso hardening


  rhs6.update((inityield() * sqrt(2. / 3.)) / dLp.norm2(), dLp_vec, 1.);  // dissipative component

  rhs6.multiply_tn(
      0.5, dcedlp, Core::LinAlg::make_stress_like_voigt_view(se), 1.);  // elastic component
  Core::LinAlg::Matrix<6, 1> rhs6Elast(
      rhs6);  // rhs6Elast is the elastic part of the right hand side. We
              // need this to build the hessian numerically


  Core::LinAlg::Matrix<6, 5> dAlphadBeta;
  dAlphadBeta(0, 0) = 1.;
  dAlphadBeta(1, 1) = 1.;
  dAlphadBeta(2, 0) = -1.;
  dAlphadBeta(2, 1) = -1.;
  dAlphadBeta(3, 2) = 1.;
  dAlphadBeta(4, 3) = 1.;
  dAlphadBeta(5, 4) = 1.;

  // Outputs
  rhs.multiply_tn(1., dAlphadBeta, rhs6, 0.);
  rhsElast.multiply_tn(1., dAlphadBeta, rhs6Elast, 0.);

  return;
}
void Mat::PlasticElastHyperVCU::evaluate_plast(Core::LinAlg::Matrix<6, 9>& dPK2dFpinvIsoprinc,
    const Core::LinAlg::Matrix<3, 1>& gamma, const Core::LinAlg::Matrix<8, 1>& delta,
    const Core::LinAlg::Matrix<3, 3>& id2, const Core::LinAlg::Matrix<6, 1>& Cpi,
    const Core::LinAlg::Matrix<3, 3>& Fpi, const Core::LinAlg::Matrix<3, 3>& CpiC,
    const Core::LinAlg::Matrix<9, 1>& CFpi, const Core::LinAlg::Matrix<9, 1>& CFpiCei,
    const Core::LinAlg::Matrix<6, 1>& ircg, const Core::LinAlg::Matrix<3, 3>& FpiCe,
    const Core::LinAlg::Matrix<9, 1>& CFpiCe, const Core::LinAlg::Matrix<6, 1>& CpiCCpi)
{
  // derivative of PK2 w.r.t. inverse plastic deformation gradient
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dPK2dFpinvIsoprinc, id2, Fpi, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dPK2dFpinvIsoprinc, CpiC, Fpi, gamma(1));
  dPK2dFpinvIsoprinc.multiply_nt(delta(0), Cpi, CFpi, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(1), Cpi, CFpiCe, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(1), CpiCCpi, CFpi, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(2), Cpi, CFpiCei, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(2), ircg, CFpi, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(3), CpiCCpi, CFpiCe, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(4), CpiCCpi, CFpiCei, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(4), ircg, CFpiCe, 1.);
  dPK2dFpinvIsoprinc.multiply_nt(delta(5), ircg, CFpiCei, 1.);
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dPK2dFpinvIsoprinc, id2, FpiCe, 0.5 * delta(7));
}

void Mat::PlasticElastHyperVCU::evaluate_kin_quant_plast(const int gp, const int eleGID,
    const Core::LinAlg::Matrix<3, 3>* defgrd, const Core::LinAlg::Matrix<3, 3>* fpi,
    Core::LinAlg::Matrix<3, 1>& gamma, Core::LinAlg::Matrix<8, 1>& delta,
    Core::LinAlg::Matrix<3, 3>& id2, Core::LinAlg::Matrix<6, 1>& Cpi,
    Core::LinAlg::Matrix<3, 3>& CpiC, Core::LinAlg::Matrix<9, 1>& CFpi,
    Core::LinAlg::Matrix<9, 1>& CFpiCei, Core::LinAlg::Matrix<6, 1>& ircg,
    Core::LinAlg::Matrix<3, 3>& FpiCe, Core::LinAlg::Matrix<9, 1>& CFpiCe,
    Core::LinAlg::Matrix<6, 1>& CpiCCpi)
{
  id2.clear();
  for (int i = 0; i < 3; i++) id2(i, i) = 1.;

  Core::LinAlg::Matrix<3, 3> fe;
  fe.multiply(*defgrd, *fpi);
  Core::LinAlg::Matrix<3, 3> ce3x3;
  ce3x3.multiply_tn(fe, fe);

  // ce here strainlike
  Core::LinAlg::Matrix<6, 1> ce;
  for (int i = 0; i < 3; i++) ce(i) = ce3x3(i, i);
  ce(3) = 2. * ce3x3(0, 1);
  ce(4) = 2. * ce3x3(1, 2);
  ce(5) = 2. * ce3x3(0, 2);

  Core::LinAlg::Matrix<6, 1> ce_stresslike;
  for (int i = 0; i < 6; i++)
    if (i < 3)
      ce_stresslike(i) = ce(i);
    else
      ce_stresslike(i) = 0.5 * ce(i);

  Core::LinAlg::Matrix<3, 1> prinv;
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, ce);

  Core::LinAlg::Matrix<3, 1> dPI;
  Core::LinAlg::Matrix<6, 1> ddPII;
  elast_hyper_evaluate_invariant_derivatives(
      prinv, dPI, ddPII, potsum_, summandProperties_, gp, eleGID);
  calculate_gamma_delta(gamma, delta, prinv, dPI, ddPII);

  // inverse plastic right Cauchy-Green
  Core::LinAlg::Matrix<3, 3> CpiM;
  CpiM.multiply_nt(*fpi, *fpi);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) Cpi(i) = CpiM(i, i);
  Cpi(3) = (CpiM(0, 1) + CpiM(1, 0)) / 2.;
  Cpi(4) = (CpiM(2, 1) + CpiM(1, 2)) / 2.;
  Cpi(5) = (CpiM(0, 2) + CpiM(2, 0)) / 2.;

  // inverse RCG
  Core::LinAlg::Matrix<3, 3> iRCG;
  Core::LinAlg::Matrix<3, 3> RCG;
  RCG.multiply_tn(*defgrd, *defgrd);
  iRCG.invert(RCG);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) ircg(i) = iRCG(i, i);
  ircg(3) = (iRCG(0, 1) + iRCG(1, 0)) / 2.;
  ircg(4) = (iRCG(2, 1) + iRCG(1, 2)) / 2.;
  ircg(5) = (iRCG(0, 2) + iRCG(2, 0)) / 2.;

  // C_p^-1 * C * C_p^-1
  Core::LinAlg::Matrix<3, 3> tmp33;
  Core::LinAlg::Matrix<3, 3> CpiCCpiM;
  tmp33.multiply(CpiM, RCG);
  CpiCCpiM.multiply(tmp33, CpiM);
  // stress-like Voigt notation
  for (int i = 0; i < 3; i++) CpiCCpi(i) = CpiCCpiM(i, i);
  CpiCCpi(3) = (CpiCCpiM(0, 1) + CpiCCpiM(1, 0)) / 2.;
  CpiCCpi(4) = (CpiCCpiM(2, 1) + CpiCCpiM(1, 2)) / 2.;
  CpiCCpi(5) = (CpiCCpiM(0, 2) + CpiCCpiM(2, 0)) / 2.;

  CpiC.multiply(CpiM, RCG);
  FpiCe.multiply(*fpi, ce3x3);

  //    FpiTC.multiply_tn(invpldefgrd,RCG);
  //    CeFpiTC.multiply(CeM,FpiTC);
  Core::LinAlg::Matrix<3, 3> tmp;
  tmp.multiply(RCG, *fpi);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(tmp, CFpi);
  tmp33.multiply(tmp, ce3x3);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(tmp33, CFpiCe);



  tmp.invert(ce3x3);
  tmp33.multiply(*fpi, tmp);
  tmp.multiply(RCG, tmp33);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(tmp, CFpiCei);
}
void Mat::PlasticElastHyperVCU::dpk2d_fpi(const int gp, const int eleGID,
    const Core::LinAlg::Matrix<3, 3>* defgrd, const Core::LinAlg::Matrix<3, 3>* fpi,
    Core::LinAlg::Matrix<6, 9>& dPK2dFpinvIsoprinc)
{
  Core::LinAlg::Matrix<3, 1> gamma;
  Core::LinAlg::Matrix<8, 1> delta;
  Core::LinAlg::Matrix<3, 3> id2;
  Core::LinAlg::Matrix<6, 1> Cpi;
  Core::LinAlg::Matrix<3, 3> CpiC;
  Core::LinAlg::Matrix<9, 1> CFpi;
  Core::LinAlg::Matrix<9, 1> CFpiCei;
  Core::LinAlg::Matrix<6, 1> ircg;
  Core::LinAlg::Matrix<3, 3> FpiCe;
  Core::LinAlg::Matrix<9, 1> CFpiCe;
  Core::LinAlg::Matrix<6, 1> CpiCCpi;
  evaluate_kin_quant_plast(gp, eleGID, defgrd, fpi, gamma, delta, id2, Cpi, CpiC, CFpi, CFpiCei,
      ircg, FpiCe, CFpiCe, CpiCCpi);

  evaluate_plast(dPK2dFpinvIsoprinc, gamma, delta, id2, Cpi, *fpi, CpiC, CFpi, CFpiCei, ircg, FpiCe,
      CFpiCe, CpiCCpi);
}

// Compute dpsiplast_dalphaiso
void Mat::PlasticElastHyperVCU::dpsiplast_dalphaiso(const double norm_dLp,
    const double last_alphaiso, const double isoHardMod, const double initYield,
    const double infYield, const double expIsoHard, double* dpsiplastdalphaiso)
{
  double new_alphaiso = 0.0;
  new_alphaiso = last_alphaiso + norm_dLp;
  double summand1 = 0.0;
  summand1 = isoHardMod * new_alphaiso;
  double summand2 = 0.0;
  double k = infYield - initYield;
  summand2 = k;
  double summand3 = 0.0;
  summand3 = k * std::exp(-expIsoHard * new_alphaiso);


  double out = 0.0;
  out = summand1 + summand2 - summand3;
  *dpsiplastdalphaiso = out;
}



// Compute DDceDdLpDdLp
void Mat::PlasticElastHyperVCU::ce2nd_deriv(const Core::LinAlg::Matrix<3, 3>* defgrd,
    const Core::LinAlg::Matrix<3, 3> fpi, const Core::LinAlg::Matrix<3, 3> dLp,
    Core::LinAlg::Matrix<6, 6>* DDceDdLpDdLpVoigt)
{
  // Compute ce trial
  Core::LinAlg::Matrix<3, 3> fetrial;
  fetrial.multiply(*defgrd, fpi);
  Core::LinAlg::Matrix<3, 3> cetrial;
  cetrial.multiply_tn(fetrial, fetrial);

  // Get matrix exponential derivatives
  Core::LinAlg::Matrix<3, 3> minus_dLp(dLp);
  minus_dLp.scale(-1.);
  Core::LinAlg::Matrix<3, 3> zeros;
  Core::LinAlg::Matrix<3, 3> exp_dLp;
  Core::LinAlg::Matrix<6, 6> Dexp_dLp_mat;
  Core::LinAlg::Matrix<6, 6> D2exp_VOIGT[6];
  Core::LinAlg::MatrixFunctErrorType exp_2nd_deriv_err_status{
      Core::LinAlg::MatrixFunctErrorType::no_errors};
  Core::LinAlg::sym_matrix_3x3_exp_2nd_deriv_voigt(
      minus_dLp, exp_dLp, Dexp_dLp_mat, D2exp_VOIGT, exp_2nd_deriv_err_status);
  FOUR_C_ASSERT_ALWAYS(exp_2nd_deriv_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
      "The evaluation of the second derivative of the matrix exponential failed!");

  Core::LinAlg::Matrix<3, 3> exp_dLp_cetrial;
  exp_dLp_cetrial.multiply(exp_dLp, cetrial);

  for (int a = 0; a < 3; a++)
    for (int d = a; d < 3; d++)
      for (int b = 0; b < 3; b++)
        for (int C = 0; C < 6; C++)
          for (int D = 0; D < 6; D++)
          {
            DDceDdLpDdLpVoigt[vmap::non_symmetric_tensor_to_voigt9_index(a, d)](C, D) +=
                (1. + (D > 2)) * (1. + (C > 2)) *
                (exp_dLp_cetrial(a, b) *
                        D2exp_VOIGT[vmap::symmetric_tensor_to_voigt6_index(b, d)](C, D) +
                    D2exp_VOIGT[vmap::symmetric_tensor_to_voigt6_index(a, b)](C, D) *
                        exp_dLp_cetrial(d, b));
            for (int c = 0; c < 3; c++)
              DDceDdLpDdLpVoigt[vmap::non_symmetric_tensor_to_voigt9_index(a, d)](C, D) +=
                  (1. + (C > 2)) * (1. + (D > 2)) *
                  (Dexp_dLp_mat(vmap::symmetric_tensor_to_voigt6_index(a, b), C) * cetrial(b, c) *
                          Dexp_dLp_mat(vmap::symmetric_tensor_to_voigt6_index(c, d), D) +
                      Dexp_dLp_mat(vmap::symmetric_tensor_to_voigt6_index(a, b), D) *
                          cetrial(b, c) *
                          Dexp_dLp_mat(vmap::symmetric_tensor_to_voigt6_index(c, d), C));
          }
}

FOUR_C_NAMESPACE_CLOSE
