// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elasthyper_service.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mat_service.hpp"

FOUR_C_NAMESPACE_OPEN

void Mat::elast_hyper_evaluate(const Core::LinAlg::Matrix<3, 3>& defgrd,
    const Core::LinAlg::Matrix<6, 1>& glstrain, Teuchos::ParameterList& params,
    Core::LinAlg::Matrix<6, 1>& stress, Core::LinAlg::Matrix<6, 6>& cmat, const int gp, int eleGID,
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
    const SummandProperties& properties, bool checkpolyconvexity)
{
  static Core::LinAlg::Matrix<6, 1> id2(Core::LinAlg::Initialization::uninitialized);
  id2.clear();
  static Core::LinAlg::Matrix<6, 1> C_strain(Core::LinAlg::Initialization::uninitialized);
  C_strain.clear();
  static Core::LinAlg::Matrix<6, 1> iC_strain(Core::LinAlg::Initialization::uninitialized);
  iC_strain.clear();
  static Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::uninitialized);
  prinv.clear();
  static Core::LinAlg::Matrix<3, 1> dPI(Core::LinAlg::Initialization::uninitialized);
  dPI.clear();
  static Core::LinAlg::Matrix<6, 1> ddPII(Core::LinAlg::Initialization::uninitialized);
  ddPII.clear();

  // Evaluate identity tensor
  Core::LinAlg::Voigt::identity_matrix(id2);

  // Evaluate Right Cauchy-Green strain tensor in strain-like Voigt notation
  evaluate_right_cauchy_green_strain_like_voigt(glstrain, C_strain);

  // invert Right Cauchy Green Strain tensor
  Core::LinAlg::Voigt::Strains::inverse_tensor(C_strain, iC_strain);

  // Evaluate principle invariants
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, C_strain);

  // Evaluate derivatives of potsum w.r.t the principal invariants
  elast_hyper_evaluate_invariant_derivatives(prinv, dPI, ddPII, potsum, properties, gp, eleGID);

  // check if system is polyconvex (set "POLYCONVEX 1" in material input-line)
  if (checkpolyconvexity)
    elast_hyper_check_polyconvexity(defgrd, prinv, dPI, ddPII, params, gp, eleGID, properties);


  // clear stress and cmat (for safety reasons)
  stress.clear();
  cmat.clear();

  // Evaluate isotropic stress response
  elast_hyper_add_isotropic_stress_cmat(stress, cmat, C_strain, iC_strain, prinv, dPI, ddPII);

  if (properties.coeffStretchesPrinc || properties.coeffStretchesMod)
  {
    elast_hyper_add_response_stretches(cmat, stress, C_strain, potsum, properties, gp, eleGID);
  }

  // Evaluate anisotropic stress response from summands with principle invariants formulation
  if (properties.anisoprinc)
    elast_hyper_add_anisotropic_princ(stress, cmat, C_strain, params, gp, eleGID, potsum);

  // Evaluate anisotropic stress response from summands with modified invariants formulation
  if (properties.anisomod)
    elast_hyper_add_anisotropic_mod(
        stress, cmat, C_strain, iC_strain, prinv, gp, eleGID, params, potsum);
}

void Mat::evaluate_right_cauchy_green_strain_like_voigt(
    const Core::LinAlg::Matrix<6, 1>& E_strain, Core::LinAlg::Matrix<6, 1>& C_strain)
{
  // C = 2*E+I
  C_strain.update(2.0, E_strain, 0.0);

  // Add Identity
  for (unsigned i = 0; i < 3; ++i) C_strain(i) += 1.0;
}

void Mat::elast_hyper_evaluate_invariant_derivatives(const Core::LinAlg::Matrix<3, 1>& prinv,
    Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII,
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
    const SummandProperties& properties, const int gp, int eleGID)
{
  // derivatives of principla materials
  if (properties.isoprinc)
  {
    // loop map of associated potential summands
    for (auto& p : potsum)
    {
      p->add_derivatives_principal(dPI, ddPII, prinv, gp, eleGID);
    }
  }

  // derivatives of decoupled (volumetric or isochoric) materials
  if (properties.isomod)
  {
    static Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
    modinv.clear();
    static Core::LinAlg::Matrix<3, 1> dPmodI(Core::LinAlg::Initialization::zero);
    dPmodI.clear();
    static Core::LinAlg::Matrix<6, 1> ddPmodII(Core::LinAlg::Initialization::zero);
    ddPmodII.clear();

    // Evaluate modified invariants
    Mat::invariants_modified(modinv, prinv);

    for (auto& p : potsum)
    {
      p->add_derivatives_modified(dPmodI, ddPmodII, modinv, gp, eleGID);
    }

    // convert decoupled derivatives to principal derivatives
    Mat::convert_mod_to_princ(prinv, dPmodI, ddPmodII, dPI, ddPII);
  }
}

void Mat::convert_mod_to_princ(const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& dPmodI, const Core::LinAlg::Matrix<6, 1>& ddPmodII,
    Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII)
{
  // Conversions to dPI
  dPI(0) += std::pow(prinv(2), -1. / 3.) * dPmodI(0);
  dPI(1) += std::pow(prinv(2), -2. / 3.) * dPmodI(1);
  dPI(2) += 0.5 * std::pow(prinv(2), -0.5) * dPmodI(2) -
            1. / 3. * prinv(0) * std::pow(prinv(2), -4. / 3.) * dPmodI(0) -
            2. / 3. * prinv(1) * std::pow(prinv(2), -5. / 3.) * dPmodI(1);

  // Conversions to ddPII
  ddPII(0) += std::pow(prinv(2), -2. / 3.) * ddPmodII(0);
  ddPII(1) += std::pow(prinv(2), -4. / 3.) * ddPmodII(1);
  ddPII(2) += (1. / 9.) * std::pow(prinv(2), -8. / 3.) * prinv(0) * prinv(0) * ddPmodII(0) +
              (4. / 9.) * prinv(0) * prinv(1) * std::pow(prinv(2), -3.) * ddPmodII(5) -
              (1. / 3.) * std::pow(prinv(2), -11. / 6.) * prinv(0) * ddPmodII(4) +
              (4. / 9.) * std::pow(prinv(2), -7. / 3.) * prinv(0) * dPmodI(0) +
              (4. / 9.) * std::pow(prinv(2), -10. / 3.) * prinv(1) * prinv(1) * ddPmodII(1) -
              (2. / 3.) * std::pow(prinv(2), -13. / 6.) * prinv(1) * ddPmodII(3) +
              (10. / 9.) * std::pow(prinv(2), -8. / 3.) * prinv(1) * dPmodI(1) +
              0.25 * std::pow(prinv(2), -1.) * ddPmodII(2) -
              0.25 * std::pow(prinv(2), -1.5) * dPmodI(2);
  ddPII(3) += -(1. / 3.) * std::pow(prinv(2), -2.) * prinv(0) * ddPmodII(5) -
              (2. / 3.) * std::pow(prinv(2), -7. / 3.) * prinv(1) * ddPmodII(1) +
              0.5 * std::pow(prinv(2), -7. / 6.) * ddPmodII(3) -
              (2. / 3.) * std::pow(prinv(2), -5. / 3.) * dPmodI(1);
  ddPII(4) += -(1. / 3.) * std::pow(prinv(2), -5. / 3.) * prinv(0) * ddPmodII(0) -
              (2. / 3.) * std::pow(prinv(2), -2.) * prinv(1) * ddPmodII(5) +
              0.5 * std::pow(prinv(2), -5. / 6.) * ddPmodII(4) -
              (1. / 3.) * std::pow(prinv(2), -4. / 3.) * dPmodI(0);
  ddPII(5) += std::pow(prinv(2), -1.) * ddPmodII(5);
}

void Mat::elast_hyper_add_isotropic_stress_cmat(Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, const Core::LinAlg::Matrix<6, 1>& C_strain,
    const Core::LinAlg::Matrix<6, 1>& iC_strain, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& dPI, const Core::LinAlg::Matrix<6, 1>& ddPII)
{
  // 2nd Piola Kirchhoff stress factors (according to Holzapfel-Nonlinear Solid Mechanics p. 216)
  static Core::LinAlg::Matrix<3, 1> gamma(Core::LinAlg::Initialization::zero);
  // constitutive tensor factors (according to Holzapfel-Nonlinear Solid Mechanics p. 261)
  static Core::LinAlg::Matrix<8, 1> delta(Core::LinAlg::Initialization::zero);
  // 2nd order identity tensor
  static Core::LinAlg::Matrix<6, 1> id2(Core::LinAlg::Initialization::uninitialized);
  // Right Cauchy-Green tensor in stress-like Voigt notation
  static Core::LinAlg::Matrix<6, 1> C_stress(Core::LinAlg::Initialization::uninitialized);
  // Inverse Right Cauchy-Green tensor in stress-like Voigt notation
  static Core::LinAlg::Matrix<6, 1> iC_stress(Core::LinAlg::Initialization::uninitialized);
  // 4th order identity tensor (rows and columns are stress-like)
  static Core::LinAlg::Matrix<6, 6> id4sharp(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::fourth_order_identity_matrix<Core::LinAlg::Voigt::NotationType::stress,
      Core::LinAlg::Voigt::NotationType::stress>(id4sharp);

  // initialize matrices
  Core::LinAlg::Voigt::identity_matrix(id2);
  Core::LinAlg::Voigt::Strains::to_stress_like(C_strain, C_stress);
  Core::LinAlg::Voigt::Strains::to_stress_like(iC_strain, iC_stress);

  // compose coefficients
  calculate_gamma_delta(gamma, delta, prinv, dPI, ddPII);

  // 2nd Piola Kirchhoff stress
  S_stress.update(gamma(0), id2, 1.0);
  S_stress.update(gamma(1), C_stress, 1.0);
  S_stress.update(gamma(2), iC_stress, 1.0);

  // constitutive tensor
  // contribution: Id \otimes Id
  cmat.multiply_nt(delta(0), id2, id2, 1.0);
  // contribution: Id \otimes C + C \otimes Id
  cmat.multiply_nt(delta(1), id2, C_stress, 1.0);
  cmat.multiply_nt(delta(1), C_stress, id2, 1.0);
  // contribution: Id \otimes Cinv + Cinv \otimes Id
  cmat.multiply_nt(delta(2), id2, iC_stress, 1.0);
  cmat.multiply_nt(delta(2), iC_stress, id2, 1.0);
  // contribution: C \otimes C
  cmat.multiply_nt(delta(3), C_stress, C_stress, 1.0);
  // contribution: C \otimes Cinv + Cinv \otimes C
  cmat.multiply_nt(delta(4), C_stress, iC_stress, 1.0);
  cmat.multiply_nt(delta(4), iC_stress, C_stress, 1.0);
  // contribution: Cinv \otimes Cinv
  cmat.multiply_nt(delta(5), iC_stress, iC_stress, 1.0);
  // contribution: Cinv \odot Cinv
  Core::LinAlg::Tensor::add_holzapfel_product(cmat, iC_stress, delta(6));
  // contribution: Id4^#
  cmat.update(delta(7), id4sharp, 1.0);
}

void Mat::elast_hyper_add_response_stretches(Core::LinAlg::Matrix<6, 6>& cmat,
    Core::LinAlg::Matrix<6, 1>& S_stress, const Core::LinAlg::Matrix<6, 1>& C_strain,
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
    const SummandProperties& properties, const int gp, int eleGID)
{
  // get principal stretches and directions
  Core::LinAlg::Matrix<3, 1> prstr;
  Core::LinAlg::Matrix<3, 3> prdir;
  stretches_principal(prstr, prdir, C_strain);
  // modified stretches
  Core::LinAlg::Matrix<3, 1> modstr;
  stretches_modified(modstr, prstr);
  // determinant of deformation gradient
  const double detdefgrad = prstr(0) * prstr(1) * prstr(2);

  // get coefficients
  Core::LinAlg::Matrix<3, 1> gamma_(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> delta_(Core::LinAlg::Initialization::zero);
  if (properties.coeffStretchesPrinc)
  {
    // loop map of associated potential summands
    for (const auto& p : potsum)
    {
      p->add_coefficients_stretches_principal(gamma_, delta_, prstr);
    }
  }
  if (properties.coeffStretchesMod)
  {
    // reciprocal of cubic root of determinant of deformation gradient (convenience)
    const double detdefgrad13 = std::pow(detdefgrad, -1.0 / 3.0);
    // retrieve coefficients with respect to modified principal stretches
    static Core::LinAlg::Matrix<3, 1> modgamma(Core::LinAlg::Initialization::zero);
    modgamma.clear();
    static Core::LinAlg::Matrix<6, 1> moddelta(Core::LinAlg::Initialization::zero);
    moddelta.clear();
    {
      // loop map of associated potential summands
      for (const auto& p : potsum)
      {
        p->add_coefficients_stretches_modified(modgamma, moddelta, modstr);
      }
    }
    // convert modified coefficients to ordinary counterparts
    //
    // derivatives of modified pr. stretches WRT pr. stretches
    static Core::LinAlg::Matrix<3, 3> modbypr(Core::LinAlg::Initialization::uninitialized);
    for (int al = 0; al < 3; ++al)
    {
      for (int be = 0; be < 3; ++be)
      {
        modbypr(al, be) = -modstr(al) / modstr(be);
      }
      modbypr(al, al) += 3.0;
    }
    modbypr.scale(detdefgrad13 / 3.0);
    // determine unmodified coefficients gamma and add them
    gamma_.multiply_tn(1.0, modbypr, modgamma, 1.0);
    // determine unmodified coefficients delta and add them
    //
    // rewrite mod.coeff. as 2-tensor
    static Core::LinAlg::Matrix<3, 3> moddeltat(Core::LinAlg::Initialization::uninitialized);
    moddeltat(0, 0) = moddelta(0);
    moddeltat(1, 1) = moddelta(1);
    moddeltat(2, 2) = moddelta(2);
    moddeltat(0, 1) = moddeltat(1, 0) = moddelta(3);
    moddeltat(1, 2) = moddeltat(2, 1) = moddelta(4);
    moddeltat(2, 0) = moddeltat(0, 2) = moddelta(5);
    // Psi_{,barlam barlam} barlam_{,lam} barlam_{,lam}
    static Core::LinAlg::Matrix<3, 3> aux(Core::LinAlg::Initialization::uninitialized);
    aux.multiply_tn(modbypr, moddeltat);
    static Core::LinAlg::Matrix<3, 3> deltat(Core::LinAlg::Initialization::uninitialized);
    deltat.multiply_nn(aux, modbypr);
    // Psi_{,barlam} barlam_{,lam lam}
    for (int be = 0; be < 3; ++be)
    {
      for (int ga = 0; ga < 3; ++ga)
      {
        double deltat_bega = 0.0;
        for (int al = 0; al < 3; ++al)
        {
          deltat_bega += -modgamma(al) * modbypr(al, be) / (3.0 * prstr(ga));
          if (ga == al) deltat_bega += -modgamma(al) * detdefgrad13 / (3.0 * prstr(be));
          if (be == ga)
            deltat_bega += modgamma(al) * detdefgrad13 * prstr(al) / (3.0 * prstr(be) * prstr(be));
        }
        deltat(be, ga) += deltat_bega;
      }
    }
    // add to delta
    // Psi_{lam lam} = Psi_{,barlam barlam} barlam_{,lam} barlam_{,lam}
    //               + Psi_{,barlam} barlam_{,lam lam}
    delta_(0) += deltat(0, 0);
    delta_(1) += deltat(1, 1);
    delta_(2) += deltat(2, 2);
    delta_(3) += deltat(0, 1);
    delta_(4) += deltat(1, 2);
    delta_(5) += deltat(2, 0);
  }

  // principal 2nd Piola--Kirchhoff stress tensor, cf [1] Eq (6.47)
  static Core::LinAlg::Matrix<3, 1> prsts(Core::LinAlg::Initialization::zero);
  prsts.clear();
  for (int al = 0; al < 3; ++al)
  {
    // PK2 principal stresses
    prsts(al) = gamma_(al) / prstr(al);
    // PK2 tensor in Voigt notation
    S_stress(0) += prsts(al) * prdir(0, al) * prdir(0, al);  // S^11
    S_stress(1) += prsts(al) * prdir(1, al) * prdir(1, al);  // S^22
    S_stress(2) += prsts(al) * prdir(2, al) * prdir(2, al);  // S^33
    S_stress(3) += prsts(al) * prdir(0, al) * prdir(1, al);  // S^12
    S_stress(4) += prsts(al) * prdir(1, al) * prdir(2, al);  // S^23
    S_stress(5) += prsts(al) * prdir(2, al) * prdir(0, al);  // S^31
  }

  using map = Core::LinAlg::Voigt::IndexMappings;

  // integration factor prfact_{al be}
  static Core::LinAlg::Matrix<6, 1> prfact1(Core::LinAlg::Initialization::zero);
  prfact1.clear();
  static Core::LinAlg::Matrix<6, 1> prfact2(Core::LinAlg::Initialization::zero);
  prfact2.clear();
  for (int albe = 0; albe < 6; ++albe)
  {
    const int al = map::voigt6_to_matrix_row_index(albe);
    const int be = map::voigt6_to_matrix_column_index(albe);
    double prfact1_albe = delta_(albe) / (prstr(al) * prstr(be));
    if (albe < 3) prfact1_albe -= gamma_(al) / (prstr(be) * prstr(al) * prstr(al));
    prfact1(albe) = prfact1_albe;
    if (al != be)
    {
      if (fabs(prstr(al) - prstr(be)) < 1e-6)
        prfact2(albe) = (prfact1(be) - prfact1(albe)) / 2.0;
      else
        prfact2(albe) = (prsts(be) - prsts(al)) / (prstr(be) * prstr(be) - prstr(al) * prstr(al));
    }
  }

  // add elasticity 4-tensor, cf Holzapfel [1] Eq (6.180),(6.196)
  for (int kl = 0; kl < 6; ++kl)
  {
    const int k = map::voigt6_to_matrix_row_index(kl);
    const int l = map::voigt6_to_matrix_column_index(kl);
    for (int ij = 0; ij < 6; ++ij)
    {
      const int i = map::voigt6_to_matrix_row_index(ij);
      const int j = map::voigt6_to_matrix_column_index(ij);
      double c_ijkl = 0.0;
      for (int albe = 0; albe < 6; ++albe)
      {
        const int al = map::voigt6_to_matrix_row_index(albe);
        const int be = map::voigt6_to_matrix_column_index(albe);
        const double fact1 = prfact1(albe);
        c_ijkl += fact1 * prdir(i, al) * prdir(j, al) * prdir(k, be) * prdir(l, be);
        if (albe >= 3)
        {  // al!=be
          c_ijkl += fact1 * prdir(i, be) * prdir(j, be) * prdir(k, al) * prdir(l, al);
          const double fact2 = prfact2(albe);
          c_ijkl += fact2 * prdir(i, al) * prdir(j, be) * prdir(k, al) * prdir(l, be) +
                    fact2 * prdir(i, al) * prdir(j, be) * prdir(k, be) * prdir(l, al) +
                    fact2 * prdir(i, be) * prdir(j, al) * prdir(k, be) * prdir(l, al) +
                    fact2 * prdir(i, be) * prdir(j, al) * prdir(k, al) * prdir(l, be);
        }
      }
      cmat(ij, kl) += c_ijkl;
    }
  }
}

void Mat::elast_hyper_add_anisotropic_princ(Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, const Core::LinAlg::Matrix<6, 1>& C_strain,
    Teuchos::ParameterList& params, const int gp, int eleGID,
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum)
{
  // Loop over all summands and add aniso stress
  // ToDo: This should be solved in analogy to the solution in elast_remodelfiber.cpp
  // ToDo: i.e. by evaluating the derivatives of the potsum w.r.t. the anisotropic invariants
  for (auto& p : potsum)
    p->add_stress_aniso_principal(C_strain, cmat, S_stress, params, gp, eleGID);
}

void Mat::elast_hyper_add_anisotropic_mod(Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, const Core::LinAlg::Matrix<6, 1>& C_strain,
    const Core::LinAlg::Matrix<6, 1>& iC_strain, const Core::LinAlg::Matrix<3, 1>& prinv,
    const int gp, int eleGID, Teuchos::ParameterList& params,
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum)
{
  static Core::LinAlg::Matrix<6, 1> iC_stress(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Strains::to_stress_like(iC_strain, iC_stress);
  // Loop over all summands and add aniso stress
  // ToDo: This should be solved in analogy to the solution in elast_remodelfiber.cpp
  // ToDo: i.e. by evaluating the derivatives of the potsum w.r.t. the anisotropic invariants
  for (auto& p : potsum)
    p->add_stress_aniso_modified(C_strain, iC_stress, cmat, S_stress, prinv(2), gp, eleGID, params);
}

void Mat::calculate_gamma_delta(Core::LinAlg::Matrix<3, 1>& gamma,
    Core::LinAlg::Matrix<8, 1>& delta, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& dPI, const Core::LinAlg::Matrix<6, 1>& ddPII)
{
  // according to Holzapfel-Nonlinear Solid Mechanics p. 216
  gamma(0) = 2. * (dPI(0) + prinv(0) * dPI(1));
  gamma(1) = -2. * dPI(1);
  gamma(2) = 2. * prinv(2) * dPI(2);

  // according to Holzapfel-Nonlinear Solid Mechanics p. 261
  delta(0) = 4. * (ddPII(0) + 2. * prinv(0) * ddPII(5) + dPI(1) + prinv(0) * prinv(0) * ddPII(1));
  delta(1) = -4. * (ddPII(5) + prinv(0) * ddPII(1));
  delta(2) = 4. * (prinv(2) * ddPII(4) + prinv(0) * prinv(2) * ddPII(3));
  delta(3) = 4. * ddPII(1);
  delta(4) = -4. * prinv(2) * ddPII(3);
  delta(5) = 4. * (prinv(2) * dPI(2) + prinv(2) * prinv(2) * ddPII(2));
  delta(6) = -4. * prinv(2) * dPI(2);
  delta(7) = -4. * dPI(1);
}

void Mat::elast_hyper_properties(const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
    SummandProperties& properties)
{
  for (auto& p : potsum)
  {
    p->specify_formulation(properties.isoprinc, properties.isomod, properties.anisoprinc,
        properties.anisomod, properties.viscoGeneral);

    properties.coeffStretchesPrinc |= p->have_coefficients_stretches_principal();
    properties.coeffStretchesMod |= p->have_coefficients_stretches_modified();
  }
}

void Mat::elast_hyper_check_polyconvexity(const Core::LinAlg::Matrix<3, 3>& defgrd,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& dPI,
    const Core::LinAlg::Matrix<6, 1>& ddPII, Teuchos::ParameterList& params, const int gp,
    const int eleGID, const SummandProperties& properties)
{
  // This polyconvexity-test is just implemented for isotropic hyperelastic-materials
  // --> error if anisotropic material is tested (plastic and viscoelastic materials should not get
  // in here)
  if (properties.anisoprinc || properties.anisomod)
    FOUR_C_THROW(
        "This polyconvexity-check is just implemented for isotropic "
        "hyperelastic-materials (do not use for anistropic materials).");

  // principal invariants (i)
  // first strain energy derivative dPI (i)
  // second strain energy derivative ddPII (i)

  // J = sqrt(I_3) = modinv(2)
  double J = std::pow(prinv(2), 1. / 2.);

  // defgrd = F (i)
  // dfgrd = F in Voigt - Notation
  static Core::LinAlg::Matrix<9, 1> dfgrd(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(defgrd, dfgrd);

  // Cof(F) = J*F^(-T)
  static Core::LinAlg::Matrix<3, 3> CoFacF(
      Core::LinAlg::Initialization::zero);  // Cof(F) in Matrix-Notation
  static Core::LinAlg::Matrix<9, 1> CofF(
      Core::LinAlg::Initialization::zero);  // Cof(F) in Voigt-Notation
  CoFacF.invert(defgrd);
  CoFacF.scale(J);
  // sort in Voigt-Notation and invert!
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CoFacF, CofF);

  // id4 (9x9)
  static Core::LinAlg::Matrix<9, 9> ID4(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 9; i++)
    for (int j = 0; j < 9; j++)
      if (i == j) ID4(i, j) = 1.0;

  // Frechet Derivative according to Ebbing, PhD-thesis page 79, Eq: (5.31)
  static Core::LinAlg::Matrix<19, 19> FreeD(Core::LinAlg::Initialization::zero);
  FreeD.clear();

  // single matrices of Frechet Derivative:

  // d^2P/dFdF
  // = 4 d^2\Psi/dI_1dI_1 F \otimes F + 2 \d\Psi/dI_1 *II
  static Core::LinAlg::Matrix<9, 9> FreeDFF(Core::LinAlg::Initialization::zero);
  FreeDFF.clear();
  FreeDFF.multiply_nt(4 * ddPII(0), dfgrd, dfgrd, 1.0);
  FreeDFF.update(2 * dPI(0), ID4, 1.0);

  // d^2P/d(cofF)d(cofF)
  // = = 4 d^2\Psi/dI_2dI_2 cof(F) \otimes cof(F) + 2 \d\Psi/dI_2 *II
  static Core::LinAlg::Matrix<9, 9> FreeDcFcF(Core::LinAlg::Initialization::zero);
  FreeDcFcF.clear();
  FreeDcFcF.multiply_nt(4 * ddPII(1), CofF, CofF, 1.0);
  FreeDcFcF.update(2 * dPI(1), ID4, 1.0);

  // d^2P/d(detF)d(detF)
  // = 2*d \Psi/dI_3 + 4*I_3*d^2\Psi/dI_3dI_3
  double FreeDJJ(true);
  FreeDJJ += 2 * dPI(2) + 4 * prinv(2) * ddPII(2);

  // d^2P/d(cofF)dF
  // = 4*d\Psi/dI_1dI_2 F /otimes CofF
  static Core::LinAlg::Matrix<9, 9> FreeDcFF(Core::LinAlg::Initialization::zero);
  FreeDcFF.clear();
  FreeDcFF.multiply_nt(4 * ddPII(5), dfgrd, CofF, 1.0);

  // d^2P/d(detF)d(cofF)
  // = 4*J*d^2 \Psi /dI_2 dI_3 \mat{CofF}
  static Core::LinAlg::Matrix<9, 1> FreeDcFJ(Core::LinAlg::Initialization::zero);
  FreeDcFF.clear();
  FreeDcFJ.update(4 * J * ddPII(3), CofF, 1.0);

  // d^2P/d(detF) dF = d^2P/dF d(detF)
  // = 4*J*d^2 \Psi /dI_1 dI_3 \mat{F}
  static Core::LinAlg::Matrix<9, 1> FreeDFJ(Core::LinAlg::Initialization::zero);
  FreeDcFF.clear();
  FreeDFJ.update(4 * J * ddPII(4), dfgrd, 1.0);

  // Sort values in Frechet Derivative

  // FreeD = [FreeDFF   FreeDcFF    FreeDFJ
  //         FreeDcFF  FreeDcFcF   FreeDcFJ
  //         FreeDFJ   FreeDcFJ    FreeDJJ]
  for (int i = 0; i < 9; i++)
    for (int j = 0; j < 9; j++)
    {
      FreeD(i, j) = FreeDFF(i, j);
      FreeD(i, j + 9) = FreeDcFF(i, j);
      FreeD(i + 9, j) = FreeDcFF(i, j);
      FreeD(i + 9, j + 9) = FreeDcFcF(i, j);
    }

  for (int i = 0; i < 9; i++)
  {
    FreeD(i + 9, 18) = FreeDcFJ(i);
    FreeD(18, i + 9) = FreeDcFJ(i);
    FreeD(i, 18) = FreeDFJ(i);
    FreeD(18, i) = FreeDFJ(i);
  }

  FreeD(18, 18) = FreeDJJ;

  // EigenValues of Frechet Derivative
  static Core::LinAlg::Matrix<19, 19> EWFreeD(
      Core::LinAlg::Initialization::zero);  // EW on diagonal
  static Core::LinAlg::Matrix<19, 19> EVFreeD(Core::LinAlg::Initialization::zero);
  Core::LinAlg::syev(FreeD, EWFreeD, EVFreeD);

  // Just positive EigenValues --> System is polyconvex
  for (int i = 0; i < 19; i++)
    for (int j = 0; j < 19; j++)
      if (i == j)  // values on diagonal = EigenValues
        if (EWFreeD(i, i) <
            (-1.0e-10 * EWFreeD.norm_inf()))  // do not test < 0, but reasonable small value
        {
          std::cout << "\nWARNING: Your system is not polyconvex!" << std::endl;
          std::cout << "Polyconvexity fails at: Element-Id: " << eleGID
                    << " and Gauss-Point: " << gp << std::endl;
          std::cout << "Eigenvalues of the Frechet Derivative are: " << EWFreeD << std::endl;
        }
}

void Mat::elast_hyper_get_derivs_of_elastic_right_cg_tensor(const Core::LinAlg::Matrix<3, 3>& iFinM,
    const Core::LinAlg::Matrix<3, 3>& CM, Core::LinAlg::Matrix<6, 6>& dCedC,
    Core::LinAlg::Matrix<6, 9>& dCediFin)
{  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::FourTensor<3> tempFourTensor(true);

  // C * F_{in}^{-1}
  static Core::LinAlg::Matrix<3, 3> CiFinM(Core::LinAlg::Initialization::zero);
  CiFinM.multiply_nn(1.0, CM, iFinM, 0.0);

  // \frac{\partial C^e}{\partial C}
  dCedC.clear();
  Core::LinAlg::Matrix<3, 3> iFinTM(Core::LinAlg::Initialization::zero);
  iFinTM.multiply_nt(1.0, id3x3, iFinM, 0.0);
  Core::LinAlg::Tensor::add_kronecker_tensor_product(dCedC, 1.0, iFinTM, iFinTM, 0.0);

  // \frac{\partial C^e}{\partial F_{in}^{-1}}
  dCediFin.clear();
  Core::LinAlg::Matrix<3, 3> iFinTCM(Core::LinAlg::Initialization::zero);
  iFinTCM.multiply_tn(1.0, iFinM, CM, 0.0);
  temp9x9.clear();
  Core::LinAlg::Tensor::add_adbc_tensor_product(1.0, id3x3, iFinTCM, temp9x9);
  Core::LinAlg::Tensor::add_non_symmetric_product(1.0, iFinTCM, id3x3, temp9x9);
  Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(tempFourTensor, temp9x9);
  Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(dCediFin, tempFourTensor);
}

FOUR_C_NAMESPACE_CLOSE
