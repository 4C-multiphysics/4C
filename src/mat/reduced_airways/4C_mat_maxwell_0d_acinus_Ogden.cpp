// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_maxwell_0d_acinus_Ogden.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_red_airways_elem_params.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Maxwell0dAcinusOgden::Maxwell0dAcinusOgden(const Core::Mat::PAR::Parameter::Data& matdata)
    : Maxwell0dAcinus(matdata)
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::Maxwell0dAcinusOgden::create_material()
{
  return std::make_shared<Mat::Maxwell0dAcinusOgden>(this);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

Mat::Maxwell0dAcinusOgdenType Mat::Maxwell0dAcinusOgdenType::instance_;


Core::Communication::ParObject* Mat::Maxwell0dAcinusOgdenType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::Maxwell0dAcinusOgden* mxwll_0d_acin = new Mat::Maxwell0dAcinusOgden();
  mxwll_0d_acin->unpack(buffer);
  return mxwll_0d_acin;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinusOgden::Maxwell0dAcinusOgden() : Maxwell0dAcinus() {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinusOgden::Maxwell0dAcinusOgden(Mat::PAR::Maxwell0dAcinus* params)
    : Maxwell0dAcinus(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusOgden::pack(Core::Communication::PackBuffer& data) const
{
  // Pack type of this instance of ParObject
  int type = unique_par_object_id();

  add_to_pack(data, type);
  add_to_pack(data, kappa_);
  add_to_pack(data, beta_);

  // Pack matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusOgden::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // Extract kappa and beta
  extract_from_pack(buffer, kappa_);
  extract_from_pack(buffer, beta_);

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
        params_ = static_cast<Mat::PAR::Maxwell0dAcinusOgden*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}


/*----------------------------------------------------------------------*
 | Setup routine to add Ogden material specific parameters kappa and    |
 | beta to material                                         roth 10/2014|
 *----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusOgden::setup(const Core::IO::InputParameterContainer& container)
{
  kappa_ = *container.get<std::optional<double>>("KAPPA");
  beta_ = *container.get<std::optional<double>>("BETA");
}


/*----------------------------------------------------------------------*
 | Evaluate Ogden material and build system matrix and rhs. Acinus Type |
 | "VolumetricOgden": continuum mechanics derivation of cauchy stress   |
 | (=hydrostatic pressure) for Ogden material for purely volumetric     |
 | deformation                                                          |
 |                                                          roth 10/2014|
 *----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusOgden::evaluate(Core::LinAlg::SerialDenseVector& epnp,
    Core::LinAlg::SerialDenseVector& epn, Core::LinAlg::SerialDenseVector& epnm,
    Core::LinAlg::SerialDenseMatrix& sysmat, Core::LinAlg::SerialDenseVector& rhs,
    const Discret::ReducedLung::ElemParams& params, const double NumOfAcini, const double Vo,
    double time, double dt)
{
  // Set sysmat and rhs to zero
  sysmat.putScalar(0.0);
  rhs.putScalar(0.0);

  // Get acinar volume in current timestep
  double acin_vn = params.acin_vn;

  // Get flow in current and next timestep
  double qnp = params.qin_np;
  double qn = params.qin_n;

  // Get acini pressure and beginning and end of acinus element
  double p1n = epn(0);
  double p2n = epn(1);

  // Safety check for NumOfAcini
  if (NumOfAcini < 1.0)
  {
    FOUR_C_THROW("Acinus condition has zero acini");
  }
  // Calculate volume and flow per acinuar duct
  double vi_n = (acin_vn / NumOfAcini);
  double qi_n = (qn / NumOfAcini);
  double qi_np = (qnp / NumOfAcini);

  // Linear branches of the Maxwell model (Stiffness2(), B=R_t, B_a=R_a), notation according to
  // interacinar dependency paper
  double Kp_np = viscosity1() / (stiffness2() * dt) + 1.0;
  double Kp_n = -viscosity1() / (stiffness2() * dt);
  double Kq_np = viscosity1() * viscosity2() / (stiffness2() * dt) + viscosity1() + viscosity2();
  double Kq_n = -viscosity1() * viscosity2() / (stiffness2() * dt);
  double rhsLin = -Kp_n * (p1n - p2n) + Kq_n * qi_n;

  // Branch E_1 of the Maxwell model: Hydrostatic pressure (=Cauchy stress) for Ogden material
  // P_1  = P_c + P_d
  // where P_c = (kappa/beta_)*(lambda^(-3))
  //       P_d =-(kappa/beta_)*(lambda^(-3-3*beta_))
  //       \lambda is the volumetric strain ratio, \lambda = (V/Vo)^(1/3)
  double vi_np = qi_np * dt + vi_n;
  double Kq_npNL = (viscosity1() / stiffness2()) *
                   (-kappa_ * Vo / (pow(vi_np, 2.0) * beta_) +
                       (beta_ + 1.0) * kappa_ * (pow(Vo / (vi_np), beta_ + 1.0)) / ((vi_np)*beta_));
  double rhsNL = (Vo / vi_n) * (kappa_ / beta_) * (1 - pow((Vo / vi_n), beta_));

  // add linearisation part to system matrix
  Kq_np += Kq_npNL;

  // Build the system matrix for \boldsymbol{K} * \boldsymbol{P} = \boldsymbol{Q}
  sysmat(0, 0) = -1.0 * (Kp_np / Kq_np) * NumOfAcini;
  sysmat(0, 1) = 1.0 * (Kp_np / Kq_np) * NumOfAcini;
  sysmat(1, 0) = 1.0 * (Kp_np / Kq_np) * NumOfAcini;
  sysmat(1, 1) = -1.0 * (Kp_np / Kq_np) * NumOfAcini;

  // Build the corresponding right hand side
  rhs(0) = -1.0 * ((rhsLin + rhsNL) * NumOfAcini / Kq_np);
  rhs(1) = 1.0 * ((rhsLin + rhsNL) * NumOfAcini / Kq_np);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Maxwell0dAcinusOgden::get_params(std::string parametername)
{
  if (parametername == "kappa")
    return kappa_;
  else if (parametername == "beta")
    return beta_;
  else
  {
    FOUR_C_THROW("Chosen Parameter can not be returned with this function!");
    return 0;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusOgden::set_params(std::string parametername, double new_value)
{
  if (parametername == "kappa")
    kappa_ = new_value;
  else
    FOUR_C_THROW("Chosen Parameter can not be set with this function yet!");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinusOgden::vis_names(std::map<std::string, int>& names)
{
  std::string fiber = "kappa";
  names[fiber] = 1;  // scalar
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Mat::Maxwell0dAcinusOgden::vis_data(
    const std::string& name, std::vector<double>& data, int eleGID)
{
  if (name == "kappa")
  {
    if ((int)data.size() != 1) FOUR_C_THROW("size mismatch");

    data[0] = get_params("kappa");
  }
  else
  {
    return false;
  }
  return true;
}

FOUR_C_NAMESPACE_CLOSE
