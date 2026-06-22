// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_viscoelast_isoratedep.hpp"

#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"

#include <Teuchos_ParameterList.hpp>

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace Mat::ViscoElast
{

  PAR::IsoRateDep::IsoRateDep(const Core::Mat::PAR::Parameter::Data& matdata)
      : Parameter(matdata), n_(matdata.parameters.get<double>("N"))
  {
  }

  IsoRateDep::IsoRateDep(PAR::IsoRateDep* params) : params_(params) {}

  void IsoRateDep::add_coefficients_visco_modified(const Core::LinAlg::Matrix<3, 1>& modinv,
      Core::LinAlg::Matrix<8, 1>& modmu, Core::LinAlg::Matrix<33, 1>& modxi,
      Core::LinAlg::Matrix<7, 1>& modrateinv, const Teuchos::ParameterList& /*params*/,
      const double dt, const int gp, const int eleGID)
  {
    const double n = params_->n_;

    modmu(1) += 2. * n * modrateinv(1);
    modmu(2) += (2. * n * (modinv(0) - 3.)) / dt;

    modxi(1) += (4. * n) / dt;
    modxi(2) += (4. * n * (modinv(0) - 3.)) / (dt * dt);
  }


  void IsoRateContribution::setup(const ContributionSetupContext& context) {}


  void IsoRateContribution::evaluate(const IsoRateEvaluateContext& context)
  {
    evaluate_kinematics(context);
    evaluate_mu_xi(context);
    add_stress_tangent(context);
  }


  void IsoRateContribution::update(const ContributionUpdateContext& context) {}


  void IsoRateContribution::evaluate_kinematics(const IsoRateEvaluateContext& context)
  {
    const auto& point = context.base.point;

    if (context.effective_properties.isomod)
    {
      invariants_modified(context.modinv, context.prinv);
    }

    const auto iso_rate_previous = context.base.state.iso_rate_prev_point(point.gp);
    Kernels::evaluate_kin_quant_vis_kernel(context.c_strain, context.c_stress, context.prinv,
        iso_rate_previous.scg, iso_rate_previous.modrcg, context.base.dt, context.mod_c_strain,
        context.scgrate, context.modrcgrate, context.modrateinv, point.visco_mat_id, point.gp);

    context.base.state.set_iso_rate_current_point(point.gp, context.c_stress, context.mod_c_strain);
  }


  void IsoRateContribution::evaluate_mu_xi(const IsoRateEvaluateContext& context)
  {
    const auto& point = context.base.point;

    Kernels::evaluate_mu_xi_kernel(context.visco_summands, context.effective_properties.isoprinc,
        context.effective_properties.isomod, context.prinv, context.modinv, context.mu,
        context.modmu, context.xi, context.modxi, context.rateinv, context.modrateinv,
        context.params, context.base.dt, point.gp, point.ele_gid);
  }


  void IsoRateContribution::add_stress_tangent(const IsoRateEvaluateContext& context)
  {
    auto& stress = context.base.stress;
    auto& cmat = context.base.cmat;

    if (context.effective_properties.isomod)
    {
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressisomodisovisco(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatisomodisovisco(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressisomodvolvisco(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatisomodvolvisco(
          Core::LinAlg::Initialization::zero);

      Kernels::evaluate_iso_visco_modified_kernel(stressisomodisovisco, stressisomodvolvisco,
          cmatisomodisovisco, cmatisomodvolvisco, context.prinv, context.modinv, context.modmu,
          context.modxi, context.c_strain, context.id2, context.i_c_stress, context.id4,
          context.modrcgrate);

      stress.update(1.0, stressisomodisovisco, 1.0);
      stress.update(1.0, stressisomodvolvisco, 1.0);
      cmat.update(1.0, cmatisomodisovisco, 1.0);
      cmat.update(1.0, cmatisomodvolvisco, 1.0);
    }

    if (context.effective_properties.isoprinc)
    {
      Core::LinAlg::Matrix<NUM_STRESS_3D, 1> stressisovisco(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D> cmatisovisco(
          Core::LinAlg::Initialization::zero);

      Kernels::evaluate_iso_visco_principal_kernel(
          stressisovisco, cmatisovisco, context.mu, context.xi, context.id4sharp, context.scgrate);

      stress.update(1.0, stressisovisco, 1.0);
      cmat.update(1.0, cmatisovisco, 1.0);
    }
  }


  void Kernels::evaluate_mu_xi_kernel(const std::vector<std::shared_ptr<Summand>>& summands,
      const bool isoprinc_active, const bool isomod_active, Matrix31& prinv, Matrix31& modinv,
      Matrix81& mu, Matrix81& modmu, Matrix331& xi, Matrix331& modxi, Matrix71& rateinv,
      Matrix71& modrateinv, const Teuchos::ParameterList& params, const double dt, const int gp,
      const int ele_gid)
  {
    if (isoprinc_active)
    {
      for (const auto& summand : summands)
        summand->add_coefficients_visco_principal(prinv, mu, xi, rateinv, params, dt, gp, ele_gid);
    }

    if (isomod_active)
    {
      for (const auto& summand : summands)
        summand->add_coefficients_visco_modified(
            modinv, modmu, modxi, modrateinv, params, dt, gp, ele_gid);
    }
  }


  void Kernels::evaluate_kin_quant_vis_kernel(const Matrix61& rcg, const Matrix61& scg,
      const Matrix31& prinv, const Matrix61& scg_previous, const Matrix61& modrcg_previous,
      const double dt, Matrix61& modrcg, Matrix61& scgrate, Matrix61& modrcgrate,
      Matrix71& modrateinv, const int visco_mat_id, const int gp)
  {
    FOUR_C_ASSERT_ALWAYS(dt > 0.0,
        "Invalid time step size dt={} in MAT_ViscoElastHyper (MAT {}, GP {}) for "
        "rate-dependent viscous update. Expected dt > 0.",
        dt, visco_mat_id, gp);

    const double modscale = std::pow(prinv(2), -1. / 3.);
    modrcg.update(modscale, rcg, 0.0);

    scgrate.update(1.0, scg, 0.0);
    scgrate.update(-1.0, scg_previous, 1.0);
    scgrate.scale(1.0 / dt);

    modrcgrate.update(1.0, modrcg, 0.0);
    modrcgrate.update(-1.0, modrcg_previous, 1.0);
    modrcgrate.scale(1.0 / dt);

    modrateinv(1) =
        0.5 * (modrcgrate(0) * modrcgrate(0) + modrcgrate(1) * modrcgrate(1) +
                  modrcgrate(2) * modrcgrate(2) + .5 * modrcgrate(3) * modrcgrate(3) +
                  .5 * modrcgrate(4) * modrcgrate(4) + .5 * modrcgrate(5) * modrcgrate(5));
  }


  void Kernels::evaluate_iso_visco_principal_kernel(Matrix61& stress, Matrix66& cmat,
      const Matrix81& mu, const Matrix331& xi, const Matrix66& id4sharp, const Matrix61& scgrate)
  {
    stress.update(mu(2), scgrate, 1.0);
    cmat.update(xi(2), id4sharp, 1.0);
  }


  void Kernels::evaluate_iso_visco_modified_kernel(Matrix61& stressisomodisovisco,
      Matrix61& stressisomodvolvisco, Matrix66& cmatisomodisovisco, Matrix66& cmatisomodvolvisco,
      const Matrix31& prinv, const Matrix31& modinv, const Matrix81& modmu, const Matrix331& modxi,
      const Matrix61& rcg, const Matrix61& id2, const Matrix61& icg, const Matrix66& id4,
      const Matrix61& modrcgrate)
  {
    const double modscale = std::pow(prinv(2), -1. / 3.);

    Matrix61 modstress(Core::LinAlg::Initialization::zero);
    modstress.update(modmu(1), id2);
    modstress.update(modmu(2), modrcgrate, 1.0);

    Matrix66 projection;
    projection.multiply_nt(1. / 3., icg, rcg);
    projection.update(1.0, id4, -1.0);

    stressisomodisovisco.multiply_nn(modscale, projection, modstress, 1.0);

    Matrix66 modcmat(Core::LinAlg::Initialization::zero);
    Matrix66 modcmat2(Core::LinAlg::Initialization::zero);
    modcmat.multiply_nt(modxi(1), id2, modrcgrate);
    modcmat.multiply_nt(modxi(1), modrcgrate, id2, 1.0);
    modcmat.update(modxi(2), id4, 1.0);
    modcmat.scale(std::pow(modinv(2), -4. / 3.));

    modcmat2.multiply_nn(projection, modcmat);
    cmatisomodisovisco.multiply_nt(1.0, modcmat2, projection, 1.0);

    modcmat.clear();
    modcmat.multiply_nt(-1.0 / 3.0, icg, icg);
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(modcmat, icg, 1.0);
    Core::LinAlg::Matrix<1, 1> tracemat;
    tracemat.multiply_tn(2. / 3. * std::pow(modinv(2), -2. / 3.), modstress, rcg);
    cmatisomodisovisco.update(tracemat(0, 0), modcmat, 1.0);
    cmatisomodisovisco.multiply_nt(-2. / 3., icg, stressisomodisovisco, 1.0);
    cmatisomodisovisco.multiply_nt(-2. / 3., stressisomodisovisco, icg, 1.0);

    // no volumetric visco contribution for visco_isoratedep
    stressisomodvolvisco.clear();
    cmatisomodvolvisco.clear();
  }

}  // namespace Mat::ViscoElast

FOUR_C_NAMESPACE_CLOSE
