// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_maxwell_0d_acinus_Exponential.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_red_airways_elem_params.hpp"
#include "4C_red_airways_elementbase.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Maxwell0dAcinusExponential::Maxwell0dAcinusExponential(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Maxwell0dAcinus(matdata)
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::Maxwell0dAcinusExponential::create_material()
{
  return std::make_shared<Mat::Maxwell0dAcinusExponential>(this);
}


Mat::Maxwell0dAcinusExponentialType Mat::Maxwell0dAcinusExponentialType::instance_;


Core::Communication::ParObject* Mat::Maxwell0dAcinusExponentialType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::Maxwell0dAcinusExponential* mxwll_0d_acin = new Mat::Maxwell0dAcinusExponential();
  mxwll_0d_acin->unpack(buffer);
  return mxwll_0d_acin;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinusExponential::Maxwell0dAcinusExponential() : Maxwell0dAcinus() {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinusExponential::Maxwell0dAcinusExponential(Mat::PAR::Maxwell0dAcinus* params)
    : Maxwell0dAcinus(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusExponential::pack(Core::Communication::PackBuffer& data) const
{
  // Pack type of this instance of ParObject
  int type = unique_par_object_id();

  add_to_pack(data, type);
  add_to_pack(data, e1_0_);
  add_to_pack(data, e1_lin_);
  add_to_pack(data, e1_exp_);
  add_to_pack(data, tau_);

  // Pack matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusExponential::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // Extract e1_0_, e1_lin_, e1_exp_, tau_
  extract_from_pack(buffer, e1_0_);
  extract_from_pack(buffer, e1_lin_);
  extract_from_pack(buffer, e1_exp_);
  extract_from_pack(buffer, tau_);

  // Extract matid
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::Maxwell0dAcinusExponential*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }
}


/*----------------------------------------------------------------------*
 | Setup routine to add Exponential material specific parameters E1_0   |
 | E1_LIN, E1_EXP, TAU to material                          roth 10/2014|
 *----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusExponential::setup(const Core::IO::InputParameterContainer& container)
{
  e1_0_ = container.get<double>("E1_0");
  e1_lin_ = container.get<double>("E1_LIN");
  e1_exp_ = container.get<double>("E1_EXP");
  tau_ = container.get<double>("TAU");
  // TODO bool -variable init, in Evaluate abfragen ob init=true
}


/*----------------------------------------------------------------------*
 | Evaluate Exponential material and build system matrix and rhs.       |
 |                                                          roth 10/2014|
 *----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusExponential::evaluate(Core::LinAlg::SerialDenseVector& epnp,
    Core::LinAlg::SerialDenseVector& epn, Core::LinAlg::SerialDenseVector& epnm,
    Core::LinAlg::SerialDenseMatrix& sysmat, Core::LinAlg::SerialDenseVector& rhs,
    const Discret::ReducedLung::ElemParams& params, const double NumOfAcini, const double Vo,
    double time, double dt)
{
  // Set sysmat and rhs to zero
  sysmat.putScalar(0.0);
  rhs.putScalar(0.0);

  // Get acinar volume in next and current timestep
  double acin_vnp = params.acin_vnp;
  double acin_vn = params.acin_vn;

  // Get flow in next and current timestep
  double qnp = params.qin_np;
  double qn = params.qin_n;

  // Get acini pressure and beginning and end of acinus element
  double p1n = epn(0);
  double p2n = epn(1);

  // Safety check for NumOfAcini
  if (NumOfAcini < 1.0)
  {
    FOUR_C_THROW("Acinus condition at node (%d) has zero acini");
  }

  // Calculate volume and flow difference per acinuar duct
  double dvnp = (acin_vnp / NumOfAcini) - Vo;
  double dvn = (acin_vn / NumOfAcini) - Vo;

  //------------------------------------------------------------
  // V  = A + B*exp(-K*P)
  //
  // The P-V curve is fitted to create the following
  // P1 = E1.(V-Vo)
  //
  // E1 = e1_0_ + e1_lin_.(V-Vo) + e1_exp_.exp(tau_.(V-Vo))
  //------------------------------------------------------------
  double kp_np = viscosity1() / (stiffness2() * dt) + 1;
  double kp_n = -viscosity1() / (stiffness2() * dt);
  double kq_np = viscosity1() * viscosity2() / (stiffness2() * dt) + (viscosity2() + viscosity1());
  double kq_n = -viscosity1() * viscosity2() / (stiffness2() * dt);

  // Get the terms associated with the nonlinear behavior of E1
  double term_nonlin = 0.0;
  double pnpi = 0.0;
  double pnpi2 = 0.0;
  double dpnpi_dt = 0.0;
  double dpnpi2_dt = 0.0;

  // Components of linearized E1
  pnpi = (e1_0_ + e1_lin_ * dvnp + e1_exp_ * exp(tau_ * dvnp)) * dvnp;
  pnpi2 = (e1_0_ + 2 * e1_lin_ * dvnp + e1_exp_ * exp(tau_ * dvnp) * (tau_ * dvnp + 1));

  // Components of linearized d(E1)/dt
  dpnpi_dt = (e1_0_ + 2 * e1_lin_ * dvnp + e1_exp_ * exp(tau_ * dvnp) * (1 + tau_ * dvnp)) *
             (dvnp - dvn) / dt;
  dpnpi2_dt = (2 * e1_lin_ + tau_ * e1_exp_ * exp(tau_ * dvnp) * (1 + tau_ * dvnp) +
                  e1_exp_ * tau_ * exp(tau_ * dvnp)) *
                  (dvnp - dvn) / dt +
              (e1_0_ + 2 * e1_lin_ * dvnp + e1_exp_ * exp(tau_ * dvnp) * (1 + tau_ * dvnp)) / dt;

  term_nonlin = pnpi + pnpi2 * (-(dvnp) + (qn / NumOfAcini) * dt / 2 + dvn);
  kq_np = kq_np + pnpi2 / 2 * dt;
  term_nonlin =
      term_nonlin + dpnpi_dt * viscosity1() / stiffness2() +
      dpnpi2_dt * viscosity1() / stiffness2() * (-(dvnp) + (qnp / NumOfAcini) * dt / 2 + dvn);
  kq_np = kq_np + dpnpi2_dt * viscosity1() / stiffness2() / 2 * dt;

  // Build the system matrix for \boldsymbol{K} * \boldsymbol{P} = \boldsymbol{Q}
  sysmat(0, 0) = -1.0 * (kp_np / kq_np) * NumOfAcini;
  sysmat(0, 1) = 1.0 * (kp_np / kq_np) * NumOfAcini;
  sysmat(1, 0) = 1.0 * (kp_np / kq_np) * NumOfAcini;
  sysmat(1, 1) = -1.0 * (kp_np / kq_np) * NumOfAcini;

  // Build the corresponding right hand side
  rhs(0) = -1.0 * (-(kp_n * (p1n - p2n) - term_nonlin) * NumOfAcini / kq_np + (kq_n * qn) / kq_np);
  rhs(1) = 1.0 * (-(kp_n * (p1n - p2n) - term_nonlin) * NumOfAcini / kq_np + (kq_n * qn) / kq_np);
}

FOUR_C_NAMESPACE_CLOSE
