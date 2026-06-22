// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_newman.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function_of_time.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Newman::Newman(const Core::Mat::PAR::Parameter::Data& matdata)
    : ElchSingleMat(matdata),
      valence_(matdata.parameters.get<double>("VALENCE")),
      transference_number_(matdata.parameters.get<double>("TRANSFERENCE_NR")),
      transference_number_concentration_scaling_funct_num_(
          matdata.parameters.get<std::optional<int>>("TRANSFERENCE_NR_CONC_SCALE_FUNCT")),
      thermodynamic_factor_(matdata.parameters.get<double>("THERM_FAC")),
      thermodynamic_factor_concentration_scaling_funct_num_(
          matdata.parameters.get<std::optional<int>>("THERM_FAC_CONC_SCALE_FUNCT"))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfTime&
Mat::PAR::Newman::transference_number_concentration_scaling_funct()
{
  FOUR_C_ASSERT(has_transference_number_concentration_scaling(),
      "You try to access the function defining the optional concentration scaling of the "
      "transference number, but no function number has been set! Check your implementation that "
      "the input file parameter 'TRANSFERENCE_NR_CONC_SCALE_FUNCT' is properly set and that you "
      "only call this function if this optional quantity is set.");

  if (!transference_number_concentration_scaling_funct_.has_value())
  {
    transference_number_concentration_scaling_funct_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
            transference_number_concentration_scaling_funct_num_.value()));
  }
  return transference_number_concentration_scaling_funct_->get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfTime&
Mat::PAR::Newman::thermodynamic_factor_concentration_scaling_funct()
{
  FOUR_C_ASSERT(has_thermodynamic_factor_concentration_scaling(),
      "You try to access the function defining the optional concentration scaling of the "
      "thermodynamic factor, but no function number has been set! Check your implementation that "
      "the input file parameter 'THERM_FAC_CONC_SCALE_FUNCT' is properly set and that you "
      "only call this function if this optional quantity is set.");

  if (!thermodynamic_factor_concentration_scaling_funct_.has_value())
  {
    thermodynamic_factor_concentration_scaling_funct_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
            thermodynamic_factor_concentration_scaling_funct_num_.value()));
  }
  return thermodynamic_factor_concentration_scaling_funct_->get();
}



std::shared_ptr<Core::Mat::Material> Mat::PAR::Newman::create_material()
{
  return std::make_shared<Mat::Newman>(this);
}

Mat::NewmanType Mat::NewmanType::instance_;


Core::Communication::ParObject* Mat::NewmanType::create(Core::Communication::UnpackBuffer& buffer)
{
  auto* newman = new Mat::Newman();
  newman->unpack(buffer);
  return newman;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Newman::Newman(Mat::PAR::Newman* params) : ElchSingleMat(params), params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Newman::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  const int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Newman::unpack(Core::Communication::UnpackBuffer& buffer)
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
      {
        params_ = static_cast<Mat::PAR::Newman*>(mat);
        set_elch_single_mat_params(params_);
      }
      else
      {
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
      }
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_transference_number(const double concentration) const
{
  // transference number
  double transference_number = params_->transference_number_;

  // transference_number = transference_number*transference_number(c)
  if (params_->has_transference_number_concentration_scaling())
  {
    transference_number *=
        params_->transference_number_concentration_scaling_funct().evaluate(concentration);
  }

  return transference_number;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_concentration_derivative_of_transference_number(
    const double concentration) const
{
  if (not params_->has_transference_number_concentration_scaling())
  {
    return 0.0;
  }

  // Computation of transference_number*transference_number'(c)
  return params_->transference_number_ *
         params_->transference_number_concentration_scaling_funct().evaluate_derivative(
             concentration);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_thermodynamic_factor(const double concentration) const
{
  // therm_fac
  double therm_fac = params_->thermodynamic_factor_;

  // therm_fac = therm_fac*therm_fac(c)
  if (params_->has_thermodynamic_factor_concentration_scaling())
  {
    therm_fac *=
        params_->thermodynamic_factor_concentration_scaling_funct().evaluate(concentration);
  }

  return therm_fac;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Newman::compute_concentration_derivative_of_thermodynamic_factor(
    const double concentration) const
{
  if (not params_->has_thermodynamic_factor_concentration_scaling())
  {
    return 0.0;
  }

  // Computation of therm_fac*therm_fac'(c)
  return params_->thermodynamic_factor_ *
         params_->thermodynamic_factor_concentration_scaling_funct().evaluate_derivative(
             concentration);
}

FOUR_C_NAMESPACE_CLOSE
