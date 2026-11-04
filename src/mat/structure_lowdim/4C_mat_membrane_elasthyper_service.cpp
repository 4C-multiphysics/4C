// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_membrane_elasthyper_service.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_mat_elasthyper_service.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  void membrane_elast_hyper_evaluate_invariant_derivatives(const Core::LinAlg::Matrix<3, 1>& prinv,
      Core::LinAlg::Matrix<2, 1>& dPI, Core::LinAlg::Matrix<3, 1>& ddPII, int gp, int eleGID,
      const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
      const SummandProperties& properties)
  {
    Core::LinAlg::Matrix<3, 1> dPI_full(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ddPII_full(Core::LinAlg::Initialization::zero);

    // REMARK: since incompressibility (J=1) is assumed principal and modified invariants are equal
    // no transformation between principal and modified invariants needed below
    if (prinv(2) != 1.0)
      FOUR_C_THROW("Incompressibility assumption not fulfilled in membrane hyperelastic material!");

    // derivatives of principal materials
    if (properties.isoprinc)
    {
      // loop map of associated potential summands
      for (const auto& p : potsum)
      {
        p->add_derivatives_principal(dPI_full, ddPII_full, prinv, gp, eleGID);
      }
    }

    // derivatives of decoupled (volumetric or isochoric) materials
    if (properties.isomod)
    {
      // loop map of associated potential summands
      for (const auto& p : potsum)
      {
        p->add_derivatives_modified(dPI_full, ddPII_full, prinv, gp, eleGID);
      }
    }

    // the derivatives dPI_full(2), ddPII_full(2), ddPII_full(3) and ddPII_full(4) are w.r.t.
    // prinv(2) = 1.0 and thus not needed in this formulation
    dPI(0) = dPI_full(0);
    dPI(1) = dPI_full(1);

    ddPII(0) = ddPII_full(0);
    ddPII(1) = ddPII_full(1);
    ddPII(2) = ddPII_full(5);
  }

  void membrane_elast_hyper_calculate_gamma_delta(Core::LinAlg::Matrix<3, 1>& gamma,
      Core::LinAlg::Matrix<8, 1>& delta, const Core::LinAlg::Matrix<3, 1>& prinv,
      const Core::LinAlg::Matrix<2, 1>& dPI, const Core::LinAlg::Matrix<3, 1>& ddPII,
      const double rcg33)
  {
    // according to Fakhreddine2011 equation (11)
    gamma(0) = 2.0 * (dPI(0) + prinv(0) * dPI(1));
    gamma(1) = -2.0 * dPI(1);
    gamma(2) = -rcg33 * gamma(0) - rcg33 * rcg33 * gamma(1);

    // according to Fakhreddine2011 equation (15)
    delta(0) =
        4.0 * (ddPII(0) + 2.0 * prinv(0) * ddPII(2) + dPI(1) + prinv(0) * prinv(0) * ddPII(1));
    delta(1) = -4.0 * (ddPII(2) + prinv(0) * ddPII(1));
    delta(2) = -4.0 * rcg33 *
               (ddPII(0) + prinv(0) * ddPII(2) + dPI(1) +
                   (prinv(0) - rcg33) * (ddPII(2) + prinv(0) * ddPII(1)));
    delta(3) = 4.0 * ddPII(1);
    delta(4) = 4.0 * rcg33 * (ddPII(2) + (prinv(0) - rcg33) * ddPII(1));
    delta(5) = -2.0 * gamma(2) + 4.0 * rcg33 * rcg33 *
                                     (ddPII(0) + 2.0 * (prinv(0) - rcg33) * ddPII(2) +
                                         std::pow((prinv(0) - rcg33), 2.0) * ddPII(1));
    delta(6) = -2.0 * gamma(2);
    delta(7) = 2.0 * gamma(1);
  }
}  // namespace Mat

void Mat::membrane_elast_hyper_evaluate_isotropic_stress_cmat(
    const Core::LinAlg::Matrix<3, 3>& cauchygreen, const Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<3, 3>& Q_trafo, Core::LinAlg::Matrix<3, 1>& stress,
    Core::LinAlg::Matrix<3, 3>& cmat, int gp, int eleGID,
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
    const SummandProperties& properties)
{
  // blank resulting quantities
  stress.clear();
  cmat.clear();

  // kinematic quantities and identity tensors
  Core::LinAlg::Matrix<3, 1> id2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> id4sharp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> rcg(Core::LinAlg::Initialization::zero);
  double rcg33;
  Core::LinAlg::Matrix<3, 1> icg(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_evaluate_kin_quant(cauchygreen, id2, id4sharp, rcg, rcg33, icg);

  // evaluate isotropic 2nd Piola-Kirchhoff stress and constitutive tensor
  membrane_elast_hyper_evaluate_isotropic_stress_cmat(
      stress, cmat, id2, id4sharp, rcg, rcg33, icg, gp, eleGID, potsum, properties);
}


void Mat::membrane_elast_hyper_evaluate_kin_quant(const Core::LinAlg::Matrix<3, 3>& cauchygreen,
    Core::LinAlg::Matrix<3, 1>& id2, Core::LinAlg::Matrix<3, 3>& id4sharp,
    Core::LinAlg::Matrix<3, 1>& rcg, double& rcg33, Core::LinAlg::Matrix<3, 1>& icg)
{
  // build Cartesian identity 2-tensor I_{AB}
  for (int i = 0; i < 2; i++) id2(i) = 1.0;

  // set Cartesian identity 4-tensor in 3-Voigt matrix notation
  // this is a fully 'contra-variant' identity tensor, ie I^{ABCD}
  // REMARK: rows are stress-like 3-Voigt
  //         columns are stress-like 3-Voigt
  for (int i = 0; i < 2; i++) id4sharp(i, i) = 1.0;
  for (int i = 2; i < 3; i++) id4sharp(i, i) = 0.5;

  // right Cauchy-Green
  // REMARK: stress-like 3-Voigt vector
  rcg(0) = cauchygreen(0, 0);
  rcg(1) = cauchygreen(1, 1);
  rcg(2) = cauchygreen(0, 1);

  // component in thickness direction of membrane
  // assuming incompressibility (J=detF=1)
  rcg33 = 1.0 / (rcg(0) * rcg(1) - std::pow(rcg(2), 2.0));

  // inverse right Cauchy-Green
  // REMARK: stress-like 3-Voigt vector
  icg(0) = rcg(1) * rcg33;
  icg(1) = rcg(0) * rcg33;
  icg(2) = -rcg(2) * rcg33;
}

void Mat::membrane_elast_hyper_evaluate_isotropic_stress_cmat(
    Core::LinAlg::Matrix<3, 1>& stress_iso, Core::LinAlg::Matrix<3, 3>& cmat_iso,
    const Core::LinAlg::Matrix<3, 1>& id2, const Core::LinAlg::Matrix<3, 3>& id4sharp,
    const Core::LinAlg::Matrix<3, 1>& rcg, const double& rcg33,
    const Core::LinAlg::Matrix<3, 1>& icg, int gp, int eleGID,
    const std::vector<std::shared_ptr<Mat::Elastic::Summand>>& potsum,
    const SummandProperties& properties)
{
  // principal isotropic invariants
  Core::LinAlg::Matrix<3, 1> prinv_iso(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_invariants_principal(prinv_iso, rcg, rcg33);

  // 1st and 2nd derivative of the isotropic strain energy function
  Core::LinAlg::Matrix<2, 1> dPI_iso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ddPII_iso(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_evaluate_invariant_derivatives(
      prinv_iso, dPI_iso, ddPII_iso, gp, eleGID, potsum, properties);

  // stress and constitutive tensor factors according to Fakhreddine2011 equation (11,15)
  Core::LinAlg::Matrix<3, 1> gamma_iso(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<8, 1> delta_iso(Core::LinAlg::Initialization::zero);
  membrane_elast_hyper_calculate_gamma_delta(
      gamma_iso, delta_iso, prinv_iso, dPI_iso, ddPII_iso, rcg33);

  // isotropic 2nd Piola Kirchhoff stress
  stress_iso.update(gamma_iso(0), id2, 1.0);
  stress_iso.update(gamma_iso(1), rcg, 1.0);
  stress_iso.update(gamma_iso(2), icg, 1.0);

  // isotropic constitutive tensor
  // contribution: Id \otimes Id
  cmat_iso.multiply_nt(delta_iso(0), id2, id2, 0.0);
  // contribution: Id \otimes C + C \otimes Id
  cmat_iso.multiply_nt(delta_iso(1), id2, rcg, 1.0);
  cmat_iso.multiply_nt(delta_iso(1), rcg, id2, 1.0);
  // contribution: Id \otimes Cinv + Cinv \otimes Id
  cmat_iso.multiply_nt(delta_iso(2), id2, icg, 1.0);
  cmat_iso.multiply_nt(delta_iso(2), icg, id2, 1.0);
  // contribution: C \otimes C
  cmat_iso.multiply_nt(delta_iso(3), rcg, rcg, 1.0);
  // contribution: C \otimes Cinv + Cinv \otimes C
  cmat_iso.multiply_nt(delta_iso(4), rcg, icg, 1.0);
  cmat_iso.multiply_nt(delta_iso(4), icg, rcg, 1.0);
  // contribution: Cinv \otimes Cinv
  cmat_iso.multiply_nt(delta_iso(5), icg, icg, 1.0);
  // contribution: Cinv \odot Cinv
  cmat_iso(0, 0) += delta_iso(6) * icg(0) * icg(0);
  cmat_iso(0, 1) += delta_iso(6) * icg(2) * icg(2);
  cmat_iso(0, 2) += delta_iso(6) * icg(0) * icg(2);
  cmat_iso(1, 0) += delta_iso(6) * icg(2) * icg(2);
  cmat_iso(1, 1) += delta_iso(6) * icg(1) * icg(1);
  cmat_iso(1, 2) += delta_iso(6) * icg(1) * icg(2);
  cmat_iso(2, 0) += delta_iso(6) * icg(0) * icg(2);
  cmat_iso(2, 1) += delta_iso(6) * icg(1) * icg(2);
  cmat_iso(2, 2) += delta_iso(6) * 0.5 * (icg(0) * icg(1) + icg(2) * icg(2));
  // contribution: Id4^#
  cmat_iso.update(delta_iso(7), id4sharp, 1.0);
}

void Mat::membrane_elast_hyper_invariants_principal(
    Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& rcg, const double& rcg33)
{
  prinv(0) = rcg(0) + rcg(1) + rcg33;
  prinv(1) =
      0.5 * (std::pow(prinv(0), 2.0) - (std::pow(rcg(0), 2.0) + std::pow(rcg(1), 2.0) +
                                           std::pow(rcg33, 2.0) + 2.0 * std::pow(rcg(2), 2.0)));
  prinv(2) = 1.0;  // incompressibility condition
}
FOUR_C_NAMESPACE_CLOSE
