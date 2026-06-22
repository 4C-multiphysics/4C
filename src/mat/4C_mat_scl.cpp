// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scl.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function_of_scalar.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Scl::Scl(const Core::Mat::PAR::Parameter::Data& matdata)
    : ElchSingleMat(matdata),
      valence_(matdata.parameters.get<double>("VALENCE")),
      transference_number_(matdata.parameters.get<double>("TRANSFERENCE_NR")),
      transference_number_concentration_scaling_funct_num_(
          matdata.parameters.get<std::optional<int>>("TRANSFERENCE_NR_CONC_SCALE_FUNCT")),
      cmax_(matdata.parameters.get<double>("MAX_CONC")),
      extrapolation_diffusion_coeff_strategy_(matdata.parameters.get<int>("EXTRAPOL_DIFF")),
      clim_(matdata.parameters.get<double>("LIM_CONC")),
      cbulk_(matdata.parameters.get<double>("BULK_CONC")),
      susceptibility_(matdata.parameters.get<double>("SUSCEPT")),
      delta_nu_(matdata.parameters.get<double>("DELTA_NU")),
      faraday_(Global::Problem::instance()->elch_control_params().get<double>("FARADAY_CONSTANT")),
      R_(Global::Problem::instance()->elch_control_params().get<double>("GAS_CONSTANT")),
      epsilon_0_(Global::Problem::instance()
              ->elch_control_params()
              .sublist("DIFFCOND")
              .get<double>("PERMITTIVITY_VACUUM"))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfScalar&
Mat::PAR::Scl::transference_number_concentration_scaling_funct()
{
  FOUR_C_ASSERT(has_transference_number_concentration_scaling(),
      "You try to access the function defining the optional concentration scaling of the "
      "transference number, but no function number has been set! Check your implementation that "
      "the input file parameter 'TRANSFERENCE_NR_CONC_SCALE_FUNCT' is properly set and that you "
      "only call this function if this optional quantity is set.");

  if (!transference_number_concentration_scaling_funct_.has_value())
  {
    transference_number_concentration_scaling_funct_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfScalar>(
            transference_number_concentration_scaling_funct_num_.value()));
  }
  return transference_number_concentration_scaling_funct_->get();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::Scl::create_material()
{
  return std::make_shared<Mat::Scl>(this);
}

Mat::SclType Mat::SclType::instance_;

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::SclType::create(Core::Communication::UnpackBuffer& buffer)
{
  auto* scl = new Mat::Scl();
  scl->unpack(buffer);
  return scl;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Scl::Scl(Mat::PAR::Scl* params) : ElchSingleMat(params), params_(params) {}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Scl::pack(Core::Communication::PackBuffer& data) const
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
void Mat::Scl::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::Scl*>(mat);
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
double Mat::Scl::compute_transference_number(const double concentration) const
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
double Mat::Scl::compute_concentration_derivative_of_transference_number(
    const double concentration) const
{
  if (not params_->has_transference_number_concentration_scaling())
  {
    return 0.0;
  }

  // Computation of transference_number*transference_number'(c)
  return params_->transference_number_ *
         params_->transference_number_concentration_scaling_funct().evaluate_derivative(
             concentration, 1);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Scl::compute_diffusion_coefficient(
    const double concentration, const double temperature) const
{
  // L indicates the mobility factor corresponding to the linear onsager relation
  const double LRT = params_->R_ * inv_val_valence_faraday_squared() *
                     compute_conductivity(concentration, temperature) *
                     (1.0 / (1.0 - (concentration - params_->cbulk_) * params_->delta_nu_)) *
                     temperature;
  const double c_max = params_->cmax_;
  double diff_coeff = 0.0;

  // The diffusion coefficient within a SCL diverges when a fully depleted or fully occupied state
  // is reached. Extrapolation strategies are used (lin_diff_number != -1) to overcome numerical
  // instabilities and obtain a well converged solution for a wide scope of discretizations in time
  // and space. The model for the diffusion coefficient is based on derivations of Steinberger K. et
  // al. (2021).
  switch (params_->extrapolation_diffusion_coeff_strategy_)
  {
    case -1:
      // no extrapolation
      diff_coeff = LRT * c_max / (concentration * (c_max - concentration));
      break;
    case 0:
    {
      // extrapolation with 0th Taylor approximation
      const double c_lim_min = params_->clim_;
      const double c_lim_max = c_max - c_lim_min;

      // no extrapolation of D(c) is required between c_lim and c_lim_low
      //(L * R * T * c_max / (c_c * (c_max - c_c)));
      if (concentration < c_lim_max && concentration > c_lim_min)
      {
        diff_coeff = LRT * c_max / (concentration * (c_max - concentration));
      }
      // extrapolation in fully depleted region (concentration approaches 0)
      // constant function (dD/dc = 0)
      else if (concentration <= c_lim_min)
      {
        diff_coeff = LRT * c_max / (c_lim_min * (c_max - c_lim_min));
      }
      // extrapolation in fully occupied region (concentration approaches c_max)
      // constant function (dD/dc = 0)
      else
      {
        diff_coeff = LRT * c_max / (c_lim_max * (c_max - c_lim_max));
      }
      break;
    }
    default:
      FOUR_C_THROW("Extrapolation strategy is not available");
  }
  return diff_coeff;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Scl::compute_concentration_derivative_of_diffusion_coefficient(
    const double concentration, const double temperature) const
{
  // L indicates the mobility factor corresponding to the linear onsager relation, which is
  // dependent on cation concentration
  const double LRT =
      params_->R_ * compute_onsager_coefficient(concentration, temperature) * temperature;
  const double LRTderconc =
      params_->R_ *
      compute_concentration_derivative_of_onsager_coefficient(concentration, temperature) *
      temperature;
  const double c_max = params_->cmax_;
  double diff_coeff_der = 0.0;

  switch (params_->extrapolation_diffusion_coeff_strategy_)
  {
    case -1:
      // no extrapolation
      diff_coeff_der = LRT * (-c_max * (c_max - 2.0 * concentration)) /
                           std::pow((concentration * (c_max - concentration)), 2) +
                       LRTderconc * c_max / (concentration * (c_max - concentration));
      break;
    case 0:
    {
      // extrapolation with 0th Taylor approximation ==> dD/dc = 0!
      const double c_lim_min = params_->clim_;
      const double c_lim_max = c_max - c_lim_min;
      // return (L * R * T * c_1 / (c_c * (c_1 - c_c)))
      // no extrapolation of D(c)/Dc is required within c_lim and c_lim_low
      if (concentration < c_lim_min && concentration > c_lim_max)
      {
        diff_coeff_der = LRT * (-c_max * (c_max - 2.0 * concentration)) /
                             std::pow((concentration * (c_max - concentration)), 2) +
                         LRTderconc * c_max / (concentration * (c_max - concentration));
      }
      break;
    }
    default:
      FOUR_C_THROW("Linearization strategy is not available");
  }
  return diff_coeff_der;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Scl::inv_val_valence_faraday_squared() const
{
  return std::pow(valence() * params_->faraday_, -2);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Scl::compute_permittivity() const
{
  const double susceptibility = compute_susceptibility();
  return ((1.0 + susceptibility) * params_->epsilon_0_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Scl::compute_onsager_coefficient(
    const double concentration, const double temperature) const
{
  // onsager coefficient (mobility factor) is derived from the measurable ionic conductivity and is
  // also related to deltanu, the difference between the partial molar volumes of vacancies and
  // cations
  const double conductivity = compute_conductivity(concentration, temperature);
  return inv_val_valence_faraday_squared() * conductivity /
         (1.0 - (concentration - params_->cbulk_) * params_->delta_nu_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Scl::compute_concentration_derivative_of_onsager_coefficient(
    const double concentration, const double temperature) const
{
  // derivative of mobility factor w.r.t concentration depends on the concentration dependence of
  // the conductivity and another factor deltanu, that takes volumetric effects into account
  // (usually, deltanu = 0.0)
  const double conductivity = compute_conductivity(concentration, temperature);
  const double conductivityderconc =
      compute_concentration_derivative_of_conductivity(concentration, temperature);
  const double cbulk = params_->cbulk_;
  const double delta_nu = params_->delta_nu_;

  const double onsagercoeffderconc =
      inv_val_valence_faraday_squared() *
      (conductivity *
              (delta_nu / (std::pow(1.0 + cbulk * delta_nu - concentration * delta_nu, 2))) +
          conductivityderconc / (1.0 - (concentration - cbulk) * delta_nu));
  return onsagercoeffderconc;
}

FOUR_C_NAMESPACE_CLOSE
