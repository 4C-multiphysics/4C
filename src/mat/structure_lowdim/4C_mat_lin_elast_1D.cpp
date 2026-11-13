// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_lin_elast_1D.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function_library.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::LinElast1D::LinElast1D(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      density_(matdata.parameters.get<double>("DENS"))
{
  if (youngs_ <= 0.) FOUR_C_THROW("Young's modulus must be greater zero");
  if (density_ <= 0.) FOUR_C_THROW("Density must be greater zero");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::LinElast1D::create_material()
{
  return std::make_shared<Mat::LinElast1D>(this);
}

Mat::LinElast1DType Mat::LinElast1DType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::LinElast1DType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* stvenantk = new Mat::LinElast1D(nullptr);
  stvenantk->unpack(buffer);
  return stvenantk;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::LinElast1D::LinElast1D(Mat::PAR::LinElast1D* params) : params_(params) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::LinElast1D::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::LinElast1D::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::LinElast1D*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::LinElast1DGrowth::LinElast1DGrowth(const Core::Mat::PAR::Parameter::Data& matdata)
    : LinElast1D(matdata),
      c0_(matdata.parameters.get<double>("C0")),
      poly_num_(matdata.parameters.get<int>("POLY_PARA_NUM")),
      poly_params_(matdata.parameters.get<std::vector<double>>("POLY_PARAMS")),
      amount_prop_growth_(matdata.parameters.get<bool>("AOS_PROP_GROWTH"))
{
  if (c0_ <= 0.0) FOUR_C_THROW("Reference concentration must be greater than zero");
  if (poly_num_ <= 0) FOUR_C_THROW("Polynomial order must be greater than zero");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::LinElast1DGrowth::create_material()
{
  return std::make_shared<Mat::LinElast1DGrowth>(this);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::LinElast1DGrowthType Mat::LinElast1DGrowthType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::LinElast1DGrowthType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* stvk_growth = new Mat::LinElast1DGrowth(nullptr);
  stvk_growth->unpack(buffer);
  return stvk_growth;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::LinElast1DGrowth::LinElast1DGrowth(Mat::PAR::LinElast1DGrowth* params)
    : LinElast1D(static_cast<Mat::PAR::LinElast1D*>(params)), growth_params_(params)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::LinElast1DGrowth::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  Mat::LinElast1D::pack(data);

  // matid
  int matid = -1;
  if (growth_params_ != nullptr)
    matid = growth_params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::LinElast1DGrowth::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class
  Mat::LinElast1D::unpack(buffer);

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  growth_params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        growth_params_ = static_cast<Mat::PAR::LinElast1DGrowth*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::LinElast1DGrowth::evaluate_p_k2(const double def_grad, const double conc) const
{
  const double def_grad_inel = amount_prop_growth() ? get_growth_factor_ao_s_prop(conc, def_grad)
                                                    : get_growth_factor_conc_prop(conc);

  const double def_grad_el = def_grad / def_grad_inel;
  const double epsilon_el = 0.5 * (def_grad_el * def_grad_el - 1.0);

  return growth_params_->youngs_ * epsilon_el / def_grad_inel;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::LinElast1DGrowth::evaluate_stiffness(const double def_grad, const double conc) const
{
  // F_in
  const double def_grad_inel = amount_prop_growth() ? get_growth_factor_ao_s_prop(conc, def_grad)
                                                    : get_growth_factor_conc_prop(conc);

  // F_el
  const double def_grad_el = def_grad / def_grad_inel;

  // E_el
  const double epsilon_el = 0.5 * (def_grad_el * def_grad_el - 1.0);

  // dF_in/dF
  const double d_def_grad_inel_d_def_grad =
      amount_prop_growth() ? get_growth_factor_ao_s_prop_deriv(conc, def_grad) : 0.0;

  // dF_el_dF
  const double d_def_grad_el_d_def_grad =
      (def_grad_inel - def_grad * d_def_grad_inel_d_def_grad) / (def_grad_inel * def_grad_inel);

  // dE_el_dFel
  const double d_epsilon_el_d_def_grad_el = def_grad_el;

  // dE_el_dF
  const double d_epsilon_el_d_def_grad = d_epsilon_el_d_def_grad_el * d_def_grad_el_d_def_grad;

  return growth_params_->youngs_ *
         (d_epsilon_el_d_def_grad * def_grad_inel - epsilon_el * d_def_grad_inel_d_def_grad) /
         (def_grad_inel * def_grad_inel);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::LinElast1DGrowth::evaluate_elastic_energy(double def_grad, double conc) const
{
  const double def_grad_inel = amount_prop_growth() ? get_growth_factor_ao_s_prop(conc, def_grad)
                                                    : get_growth_factor_conc_prop(conc);

  const double def_grad_el = def_grad / def_grad_inel;
  const double epsilon_el = 0.5 * (def_grad_el * def_grad_el - 1.0);

  return 0.5 * (2.0 * growth_params_->youngs_ * epsilon_el / def_grad_inel) * epsilon_el;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::LinElast1DGrowth::get_growth_factor_conc_prop(const double conc) const
{
  return Core::FE::Polynomial(growth_params_->poly_params_).evaluate(conc - growth_params_->c0_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::LinElast1DGrowth::get_growth_factor_ao_s_prop(
    const double conc, const double def_grad) const
{
  return Core::FE::Polynomial(growth_params_->poly_params_)
      .evaluate(conc * def_grad - growth_params_->c0_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::LinElast1DGrowth::get_growth_factor_ao_s_prop_deriv(
    const double conc, const double def_grad) const
{
  const double first_deriv = Core::FE::Polynomial(growth_params_->poly_params_)
                                 .evaluate_derivative(conc * def_grad - growth_params_->c0_, 1);

  return first_deriv * conc;
}

FOUR_C_NAMESPACE_CLOSE
