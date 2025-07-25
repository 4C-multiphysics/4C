// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_service.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential.hpp"
#include "4C_mixture_constituent_remodelfiber_material_exponential_active.hpp"
#include "4C_mixture_growth_strategy_anisotropic.hpp"
#include "4C_mixture_growth_strategy_isotropic.hpp"
#include "4C_mixture_growth_strategy_stiffness.hpp"
#include "4C_mixture_prestress_strategy_constant.hpp"
#include "4C_mixture_prestress_strategy_isocyl.hpp"
#include "4C_mixture_prestress_strategy_iterative.hpp"
#include "4C_mixture_rule_function.hpp"
#include "4C_mixture_rule_growthremodel.hpp"
#include "4C_mixture_rule_simple.hpp"
#include "4C_utils_enum.hpp"

#include <Sacado.hpp>

using FAD = Sacado::Fad::DFad<double>;

FOUR_C_NAMESPACE_OPEN



double Mat::second_invariant_of_deviatoric_stress(const Core::LinAlg::Matrix<3, 3>& stress)
{
  const double p = 1.0 / 3 * (stress(0, 0) + stress(1, 1) + stress(2, 2));
  const double s11 = stress(0, 0) - p;
  const double s22 = stress(1, 1) - p;
  const double s33 = stress(2, 2) - p;
  const double s12 = stress(0, 1);
  const double s23 = stress(1, 2);
  const double s13 = stress(0, 2);
  const double J2 =
      0.5 * (s11 * s11 + s22 * s22 + s33 * s33 + 2 * (s12 * s12 + s23 * s23 + s13 * s13));
  return J2;
}

void Mat::volumetrify_and_isochorify(Core::LinAlg::Matrix<6, 1>* pk2vol,
    Core::LinAlg::Matrix<6, 6>* cvol, Core::LinAlg::Matrix<6, 1>* pk2iso,
    Core::LinAlg::Matrix<6, 6>* ciso, const Core::LinAlg::Matrix<6, 1>& gl,
    const Core::LinAlg::Matrix<6, 1>& pk2, const Core::LinAlg::Matrix<6, 6>& cmat)
{
  // useful call?
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if ((pk2vol == nullptr) and (cvol == nullptr) and (pk2iso == nullptr) and (ciso == nullptr))
    FOUR_C_THROW("Useful call? Apparently you do not want to compute anything");
#endif

  // right Cauchy--Green tensor
  // REMARK: stored in _strain_-like 6-Voigt vector
  Core::LinAlg::Matrix<6, 1> rcg(gl);
  rcg.scale(2.0);
  for (int i = 0; i < 3; i++) rcg(i) += 1.0;

  // third invariant (determinant) of right Cauchy--Green strains
  const double rcg3rd = rcg(0) * rcg(1) * rcg(2) + 0.25 * rcg(3) * rcg(4) * rcg(5) -
                        0.25 * rcg(1) * rcg(5) * rcg(5) - 0.25 * rcg(2) * rcg(3) * rcg(3) -
                        0.25 * rcg(0) * rcg(4) * rcg(4);

  // inverse right Cauchy--Green tensor C^{-1}
  // REMARK: stored in as _stress_ 6-Voigt vector
  Core::LinAlg::Matrix<6, 1> icg(Core::LinAlg::Initialization::uninitialized);
  icg(0) = (rcg(1) * rcg(2) - 0.25 * rcg(4) * rcg(4)) / rcg3rd;        // (C^{-1})^{11}
  icg(1) = (rcg(0) * rcg(2) - 0.25 * rcg(5) * rcg(5)) / rcg3rd;        // (C^{-1})^{22}
  icg(2) = (rcg(0) * rcg(1) - 0.25 * rcg(3) * rcg(3)) / rcg3rd;        // (C^{-1})^{33}
  icg(3) = (0.25 * rcg(5) * rcg(4) - 0.5 * rcg(3) * rcg(2)) / rcg3rd;  // (C^{-1})^{12}
  icg(4) = (0.25 * rcg(3) * rcg(5) - 0.5 * rcg(0) * rcg(4)) / rcg3rd;  // (C^{-1})^{23}
  icg(5) = (0.25 * rcg(3) * rcg(4) - 0.5 * rcg(5) * rcg(1)) / rcg3rd;  // (C^{-1})^{31}

  // double contraction of 2nd Piola--Kirchhoff stress and right Cauchy--Green strain,
  // i.e. in index notation S^{AB} C_{AB}
  // REMARK: equal to S^T C, because S is stress-like and C is strain-like 6-Voigt vector
  const double pk2rcg = pk2.dot(rcg);

  // stress splitting
  {
    // volumetric 2nd Piola--Kirchhoff stress
    Core::LinAlg::Matrix<6, 1> pk2vol_tmp(Core::LinAlg::Initialization::uninitialized);
    if (pk2vol != nullptr) pk2vol_tmp.set_view(*pk2vol);
    pk2vol_tmp.update(pk2rcg / 3.0, icg);

    // isochoric 2nd Piola--Kirchhoff stress
    // S^{AB}_iso = S^{AB} - S^{AB}_{vol}
    if (pk2iso != nullptr) pk2iso->update(1.0, pk2, -1.0, pk2vol_tmp);
  }

  // elasticity tensor splitting
  {
    // 'linearised' 2nd Piola--Kirchhoff stress
    // S^{CD}_lin = S^{CD} + 1/2 C_{AB} C^{ABCD}
    Core::LinAlg::Matrix<6, 1> pk2lin(pk2);
    pk2lin.multiply_tn(0.5, cmat, rcg, 1.0);  // transpose on purpose

    // volumetric part of constitutive tensor
    // C^{ABCD}_vol = 2/3 (C^{-1})^{AB} S^{CD}_lin
    //              - 2/3 (S^{EF} C_{EF}) ( 1/2 (
    //                (C^{-1})^{AC} (C^{-1})^{BD} + (C^{-1})^{AD} (C^{-1})^{BC}
    //              ) )
    Core::LinAlg::Matrix<6, 6> cvol_tmp(Core::LinAlg::Initialization::uninitialized);
    if (cvol != nullptr) cvol_tmp.set_view(*cvol);
    cvol_tmp.multiply_nt(2.0 / 3.0, icg, pk2lin);
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cvol_tmp, icg, -2.0 / 3.0 * pk2rcg);

    // isochoric part of constitutive tensor
    // C^{ABCD}_iso = C^{ABCD} - C^{ABCD}_vol
    if (ciso != nullptr) ciso->update(1.0, cmat, -1.0, cvol_tmp);
  }
}



void Mat::invariants_principal(
    Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 3>& tens)
{
  // 1st invariant, trace tens
  prinv(0) = tens(0, 0) + tens(1, 1) + tens(2, 2);

  // 2nd invariant, 0.5( (trace(tens))^2 - trace(tens^2))
  prinv(1) = tens(0, 0) * tens(1, 1) + tens(1, 1) * tens(2, 2) + tens(0, 0) * tens(2, 2) -
             tens(0, 1) * tens(1, 0) - tens(1, 2) * tens(2, 1) - tens(0, 2) * tens(2, 0);

  // 3rd invariant, determinant tens
  prinv(2) = tens.determinant();
}

void Mat::invariants_modified(
    Core::LinAlg::Matrix<3, 1>& modinv, const Core::LinAlg::Matrix<3, 1>& prinv)
{
  // 1st invariant, trace
  modinv(0) = prinv(0) * std::pow(prinv(2), -1. / 3.);
  // 2nd invariant
  modinv(1) = prinv(1) * std::pow(prinv(2), -2. / 3.);
  // J
  modinv(2) = std::pow(prinv(2), 1. / 2.);
}

void Mat::stretches_principal(Core::LinAlg::Matrix<3, 1>& prstr, Core::LinAlg::Matrix<3, 3>& prdir,
    const Core::LinAlg::Matrix<6, 1>& rcg)
{
  // create right Cauchy-Green 2-tensor
  Core::LinAlg::Matrix<3, 3> rcgt(Core::LinAlg::Initialization::uninitialized);
  rcgt(0, 0) = rcg(0);
  rcgt(1, 1) = rcg(1);
  rcgt(2, 2) = rcg(2);
  rcgt(0, 1) = rcgt(1, 0) = 0.5 * rcg(3);
  rcgt(1, 2) = rcgt(2, 1) = 0.5 * rcg(4);
  rcgt(2, 0) = rcgt(0, 2) = 0.5 * rcg(5);

  // eigenvalue decomposition
  Core::LinAlg::Matrix<3, 3> prstr2;  // squared principal stretches
  Core::LinAlg::syev(rcgt, prstr2, prdir);

  // THE principal stretches
  for (int al = 0; al < 3; ++al) prstr(al) = std::sqrt(prstr2(al, al));
}

void Mat::stretches_modified(
    Core::LinAlg::Matrix<3, 1>& modstr, const Core::LinAlg::Matrix<3, 1>& prstr)
{
  // determinant of deformation gradient
  const double detdefgrad = prstr(0) * prstr(1) * prstr(2);

  // determine modified principal stretches
  modstr.update(std::pow(detdefgrad, -1.0 / 3.0), prstr);
}

Core::LinAlg::Matrix<6, 6> Mat::pull_back_four_tensor(const double det_F,
    const Core::LinAlg::Matrix<3, 3>& F_inv, const Core::LinAlg::Matrix<6, 6>& cmat_voigt)
{
  Core::LinAlg::FourTensor<3> cmat_tensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(cmat_tensor, cmat_voigt);

  // We can use the fact that cmat_result_voigt(i,j,k,l)=cmat_result_voigt(k,l,i,j) if we have a
  // hyper-elastic material
  Core::LinAlg::Matrix<6, 6> cmat_result_voigt(Core::LinAlg::Initialization::zero);

  cmat_result_voigt(0, 0) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 0, 0, 0);
  cmat_result_voigt(0, 1) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 0, 1, 1);
  cmat_result_voigt(0, 2) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 0, 2, 2);
  cmat_result_voigt(0, 3) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 0, 0, 1);
  cmat_result_voigt(0, 4) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 0, 1, 2);
  cmat_result_voigt(0, 5) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 0, 0, 2);
  cmat_result_voigt(1, 0) = cmat_result_voigt(0, 1);
  cmat_result_voigt(1, 1) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 1, 1, 1, 1);
  cmat_result_voigt(1, 2) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 1, 1, 2, 2);
  cmat_result_voigt(1, 3) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 1, 1, 0, 1);
  cmat_result_voigt(1, 4) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 1, 1, 1, 2);
  cmat_result_voigt(1, 5) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 1, 1, 0, 2);
  cmat_result_voigt(2, 0) = cmat_result_voigt(0, 2);
  cmat_result_voigt(2, 1) = cmat_result_voigt(1, 2);
  cmat_result_voigt(2, 2) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 2, 2, 2, 2);
  cmat_result_voigt(2, 3) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 2, 2, 0, 1);
  cmat_result_voigt(2, 4) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 2, 2, 1, 2);
  cmat_result_voigt(2, 5) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 2, 2, 0, 2);
  cmat_result_voigt(3, 0) = cmat_result_voigt(0, 3);
  cmat_result_voigt(3, 1) = cmat_result_voigt(1, 3);
  cmat_result_voigt(3, 2) = cmat_result_voigt(2, 3);
  cmat_result_voigt(3, 3) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 1, 0, 1);
  cmat_result_voigt(3, 4) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 1, 1, 2);
  cmat_result_voigt(3, 5) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 1, 0, 2);
  cmat_result_voigt(4, 0) = cmat_result_voigt(0, 4);
  cmat_result_voigt(4, 1) = cmat_result_voigt(1, 4);
  cmat_result_voigt(4, 2) = cmat_result_voigt(2, 4);
  cmat_result_voigt(4, 3) = cmat_result_voigt(3, 4);
  cmat_result_voigt(4, 4) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 1, 2, 1, 2);
  cmat_result_voigt(4, 5) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 1, 2, 0, 2);
  cmat_result_voigt(5, 0) = cmat_result_voigt(0, 5);
  cmat_result_voigt(5, 1) = cmat_result_voigt(1, 5);
  cmat_result_voigt(5, 2) = cmat_result_voigt(2, 5);
  cmat_result_voigt(5, 3) = cmat_result_voigt(3, 5);
  cmat_result_voigt(5, 4) = cmat_result_voigt(4, 5);
  cmat_result_voigt(5, 5) = get_pull_back_four_tensor_entry(det_F, F_inv, cmat_tensor, 0, 2, 0, 2);

  return cmat_result_voigt;
}

double Mat::get_pull_back_four_tensor_entry(const double det_F,
    const Core::LinAlg::Matrix<3, 3>& F_inv, const Core::LinAlg::FourTensor<3>& four_tensor,
    const int i, const int j, const int k, const int l)
{
  double cMatResult_ijkl(0.0);

  for (int A = 0; A < 3; ++A)
  {
    for (int B = 0; B < 3; ++B)
    {
      for (int C = 0; C < 3; ++C)
      {
        for (int D = 0; D < 3; ++D)
        {
          cMatResult_ijkl +=
              F_inv(i, A) * F_inv(j, B) * F_inv(k, C) * F_inv(l, D) * four_tensor(A, B, C, D);
        }
      }
    }
  }

  return cMatResult_ijkl * det_F;
}

Core::LinAlg::Matrix<6, 6> Mat::push_forward_four_tensor(const double det_F,
    const Core::LinAlg::Matrix<3, 3>& defgrd, const Core::LinAlg::Matrix<6, 6>& cmat_lagr_voigt)
{
  Core::LinAlg::Matrix<6, 6> cmatEul(Core::LinAlg::Initialization::zero);
  for (int p = 0; p < 6; p++)
  {
    for (int q = 0; q < 6; q++)
    {
      int i = Core::LinAlg::Voigt::IndexMappings::voigt6_to_matrix_row_index(p);
      int j = Core::LinAlg::Voigt::IndexMappings::voigt6_to_matrix_column_index(p);
      int k = Core::LinAlg::Voigt::IndexMappings::voigt6_to_matrix_row_index(q);
      int l = Core::LinAlg::Voigt::IndexMappings::voigt6_to_matrix_column_index(q);

      for (int A = 0; A < 3; A++)
      {
        for (int B = 0; B < 3; B++)
        {
          for (int C = 0; C < 3; C++)
          {
            for (int D = 0; D < 3; D++)
            {
              int M = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(A, B);
              int N = Core::LinAlg::Voigt::IndexMappings::symmetric_tensor_to_voigt6_index(C, D);

              cmatEul(p, q) +=
                  defgrd(i, A) * defgrd(j, B) * defgrd(k, C) * defgrd(l, D) * cmat_lagr_voigt(M, N);
            }
          }
        }
      }
    }
  }
  cmatEul.scale(1.0 / det_F);

  return cmatEul;
}

Core::LinAlg::Matrix<6, 1> Mat::push_forward_stress_tensor_voigt(
    const Core::LinAlg::Matrix<6, 1>& stress_elastic,
    const Core::LinAlg::Matrix<3, 3>& deformation_gradient)
{
  Core::LinAlg::Matrix<3, 3> GS_tilde(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> GS_tildeGT(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> S_elastic(Core::LinAlg::Initialization::zero);

  // convert original stress to stress like Voigt notation
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(stress_elastic, S_elastic);

  // compute G*S_tilde*GT
  GS_tilde.multiply_nn(deformation_gradient, S_elastic);
  GS_tildeGT.multiply_nt(GS_tilde, deformation_gradient);

  // compute pk2 in reference configuration
  Core::LinAlg::Matrix<6, 1> stress(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(GS_tildeGT, stress);
  stress.scale(1.0 / deformation_gradient.determinant());

  return stress;
}



void Mat::setup_linear_isotropic_elastic_tensor(Core::LinAlg::FourTensor<3>& elasticity_tensor,
    const double youngs_modulus, const double poisson_ratio)
{
  const double lambda =
      poisson_ratio * youngs_modulus / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio));
  const double mu = youngs_modulus / (2 * (1 + poisson_ratio));

  const auto eye = [](int i, int j) { return i == j ? 1.0 : 0.0; };

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      for (unsigned int k = 0; k < 3; ++k)
        for (unsigned int l = 0; l < 3; ++l)
          elasticity_tensor(i, j, k, l) =
              lambda * eye(i, j) * eye(k, l) + mu * (eye(i, k) * eye(j, l) + eye(i, l) * eye(j, k));
}


FOUR_C_NAMESPACE_CLOSE
