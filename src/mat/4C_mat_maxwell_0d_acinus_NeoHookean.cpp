// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_maxwell_0d_acinus_NeoHookean.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_red_airways_elem_params.hpp"
#include "4C_red_airways_elementbase.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Maxwell0dAcinusNeoHookean::Maxwell0dAcinusNeoHookean(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Maxwell0dAcinus(matdata)
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::Maxwell0dAcinusNeoHookean::create_material()
{
  return std::make_shared<Mat::Maxwell0dAcinusNeoHookean>(this);
}


Mat::Maxwell0dAcinusNeoHookeanType Mat::Maxwell0dAcinusNeoHookeanType::instance_;


Core::Communication::ParObject* Mat::Maxwell0dAcinusNeoHookeanType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::Maxwell0dAcinusNeoHookean* mxwll_0d_acin = new Mat::Maxwell0dAcinusNeoHookean();
  mxwll_0d_acin->unpack(buffer);
  return mxwll_0d_acin;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinusNeoHookean::Maxwell0dAcinusNeoHookean() : Maxwell0dAcinus() {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinusNeoHookean::Maxwell0dAcinusNeoHookean(Mat::PAR::Maxwell0dAcinus* params)
    : Maxwell0dAcinus(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusNeoHookean::pack(Core::Communication::PackBuffer& data) const
{
  // Pack type of this instance of ParObject
  int type = unique_par_object_id();

  add_to_pack(data, type);

  // Pack matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusNeoHookean::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

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
        params_ = static_cast<Mat::PAR::Maxwell0dAcinusNeoHookean*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->type(),
            material_type());
    }
}


/*----------------------------------------------------------------------*
 | Setup routine for NeoHookean material                                |
 |                                                          roth 10/2014|
 *----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusNeoHookean::setup(const Core::IO::InputParameterContainer& container)
{
  // do nothing, all parameters are read by base class already
}


/*----------------------------------------------------------------------*
 | Evaluate NeoHookean material and build system matrix and rhs         |
 |                                                          roth 10/2014|
 *----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusNeoHookean::evaluate(Core::LinAlg::SerialDenseVector& epnp,
    Core::LinAlg::SerialDenseVector& epn, Core::LinAlg::SerialDenseVector& epnm,
    Core::LinAlg::SerialDenseMatrix& sysmat, Core::LinAlg::SerialDenseVector& rhs,
    const Discret::ReducedLung::ElemParams& params, const double NumOfAcini, const double Vo,
    double time, double dt)
{
  // Set sysmat and rhs to zero
  sysmat.putScalar(0.0);
  rhs.putScalar(0.0);

  // Get acini pressure and beginning and end of acinus element
  double p1n = epn(0);
  double p2n = epn(1);

  // Safety check for NumOfAcini
  if (NumOfAcini < 1.0)
  {
    FOUR_C_THROW("Acinus condition at node (%d) has zero acini");
  }

  // Linear branches of the Maxwell model (Stiffness2(), B=R_t, B_a=R_a), notation according to
  // interacinar dependency paper
  const double Kp_np = 1.0 / (stiffness1() * dt);
  const double Kp_n = 1.0 / (stiffness1() * dt);

  // Build the system matrix for \boldsymbol{K} * \boldsymbol{P} = \boldsymbol{Q}
  sysmat(0, 0) = -1.0 * (Kp_np)*NumOfAcini;
  sysmat(0, 1) = 1.0 * (Kp_np)*NumOfAcini;
  sysmat(1, 0) = 1.0 * (Kp_np)*NumOfAcini;
  sysmat(1, 1) = -1.0 * (Kp_np)*NumOfAcini;

  // Build the corresponding right hand side
  rhs(0) = -1.0 * (Kp_n * (p1n - p2n)) * NumOfAcini;
  rhs(1) = 1.0 * (Kp_n * (p1n - p2n)) * NumOfAcini;
}

FOUR_C_NAMESPACE_CLOSE
