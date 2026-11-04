// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluidporo_singlephaselaw.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseLaw::FluidPoroPhaseLaw(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  return;
}


/*----------------------------------------------------------------------*
 *  factory method for phase law                       vuong 08/16      |
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseLaw* Mat::PAR::FluidPoroPhaseLaw::create_phase_law(int phaselawId)
{
  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(phaselawId);

  // phase law
  Mat::PAR::FluidPoroPhaseLaw* phaselaw = nullptr;

  // build the pressure-saturation law
  switch (curmat->type())
  {
    case Core::Materials::m_fluidporo_phaselaw_linear:
    {
      phaselaw = static_cast<Mat::PAR::FluidPoroPhaseLawLinear*>(curmat);
      break;
    }
    case Core::Materials::m_fluidporo_phaselaw_tangent:
    {
      phaselaw = static_cast<Mat::PAR::FluidPoroPhaseLawTangent*>(curmat);
      break;
    }
    case Core::Materials::m_fluidporo_phaselaw_constraint:
    {
      phaselaw = static_cast<Mat::PAR::FluidPoroPhaseLawConstraint*>(curmat);
      break;
    }
    case Core::Materials::m_fluidporo_phaselaw_byfunction:
    {
      phaselaw = static_cast<Mat::PAR::FluidPoroPhaseLawByFunction*>(curmat);
      break;
    }
    default:
      FOUR_C_THROW("invalid pressure-saturation law for material {}", curmat->type());
      break;
  }

  return phaselaw;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseLawLinear::FluidPoroPhaseLawLinear(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroPhaseLaw(matdata),
      numdof_(matdata.parameters.get<int>("NUMDOF")),
      presids_(matdata.parameters.get<std::vector<int>>("PRESCOEFF")),
      reltensions_(matdata.parameters.get<double>("RELTENSION")),
      sat0_(matdata.parameters.get<double>("SATURATION_0"))
{
  // check if sizes fit
  if (numdof_ != (int)presids_.size())
    FOUR_C_THROW(
        "number of dofs {} does not fit to size of dof vector {}", numdof_, presids_.size());

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawLinear::evaluate_saturation(
    const std::span<const double> pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  double presval = std::inner_product(presids_.begin(), presids_.end(), pressure.begin(), 0.0);

  double saturation = sat0_ + reltensions_ * presval;

  return saturation;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawLinear::evaluate_deriv_of_saturation_wrt_pressure(
    int doftoderive, const std::span<const double> pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  if (presids_[doftoderive] == 0) return 0.0;

  double deriv = reltensions_;

  return deriv * presids_[doftoderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawLinear::evaluate_second_deriv_of_saturation_wrt_pressure(
    int firstdoftoderive, int seconddoftoderive, const std::vector<double>& pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  // second derivative is zero
  return 0.0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawLinear::evaluate_deriv_of_pressure_wrt_saturation(
    int doftoderive, double saturation)
{
  if (presids_[doftoderive] == 0) return 0.0;

  double deriv = 1.0 / reltensions_;

  return deriv * presids_[doftoderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawLinear::evaluate_gen_pressure(double saturation)
{
  double presval = 1.0 / reltensions_ * (saturation - sat0_);

  return presval;
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseLawTangent::FluidPoroPhaseLawTangent(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroPhaseLaw(matdata),
      numdof_(matdata.parameters.get<int>("NUMDOF")),
      presids_(matdata.parameters.get<std::vector<int>>("PRESCOEFF")),
      reltensions_(matdata.parameters.get<double>("RELTENSION")),
      exp_(matdata.parameters.get<double>("EXP")),
      sat0_(matdata.parameters.get<double>("SATURATION_0"))
{
  // check if sizes fit
  if (numdof_ != (int)presids_.size())
    FOUR_C_THROW(
        "number of dofs {} does not fit to size of dof vector {}", numdof_, presids_.size());
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawTangent::evaluate_saturation(
    const std::span<const double> pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  double presval = std::inner_product(presids_.begin(), presids_.end(), pressure.begin(), 0.0);

  double saturation =
      sat0_ - std::pow(2 / std::numbers::pi * std::atan(reltensions_ * presval), exp_);

  return saturation;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawTangent::evaluate_deriv_of_saturation_wrt_pressure(
    int doftoderive, const std::span<const double> pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  if (presids_[doftoderive] == 0) return 0.0;

  double presval = std::inner_product(presids_.begin(), presids_.end(), pressure.begin(), 0.0);

  double deriv = -exp_ *
                 std::pow(2 / std::numbers::pi * std::atan(reltensions_ * presval), exp_ - 1.0) *
                 2.0 * reltensions_ /
                 (std::numbers::pi * (1.0 + (reltensions_ * presval) * (reltensions_ * presval)));

  return deriv * presids_[doftoderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawTangent::evaluate_second_deriv_of_saturation_wrt_pressure(
    int firstdoftoderive, int seconddoftoderive, const std::vector<double>& pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  if (presids_[firstdoftoderive] == 0 || presids_[seconddoftoderive] == 0) return 0.0;

  double presval = std::inner_product(presids_.begin(), presids_.end(), pressure.begin(), 0.0);

  double secondderiv = 0.0;
  // avoid division by zero in case of small presval --> second deriv goes to zero for presval --> 0
  if (fabs(presval) > 1.0e-12)
  {
    secondderiv = -exp_ * reltensions_ * reltensions_ *
                  (exp_ - 2.0 * reltensions_ * presval * std::atan(reltensions_ * presval) - 1.0) *
                  std::pow(2.0 / std::numbers::pi * std::atan(reltensions_ * presval), exp_) /
                  ((1.0 + (reltensions_ * presval) * (reltensions_ * presval)) *
                      (1.0 + (reltensions_ * presval) * (reltensions_ * presval))) /
                  (std::atan(reltensions_ * presval) * std::atan(reltensions_ * presval));
  }

  // second derivative
  return secondderiv * presids_[firstdoftoderive] * presids_[seconddoftoderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawTangent::evaluate_deriv_of_pressure_wrt_saturation(
    int doftoderive, double saturation)
{
  if (presids_[doftoderive] == 0) return 0.0;

  double deriv =
      -0.5 * std::numbers::pi / (reltensions_ * exp_) *
      std::pow(sat0_ - saturation, 1.0 / exp_ - 1.0) *
      (1.0 +
          std::pow(std::tan(0.5 * std::numbers::pi * std::pow(sat0_ - saturation, 1.0 / exp_)), 2));

  return deriv * presids_[doftoderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawTangent::evaluate_gen_pressure(double saturation)
{
  double presval = 1.0 / reltensions_ *
                   std::tan(0.5 * std::numbers::pi * std::pow(sat0_ - saturation, 1.0 / exp_));

  return presval;
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroPhaseLawByFunction::FluidPoroPhaseLawByFunction(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroPhaseLaw(matdata),
      numdof_(matdata.parameters.get<int>("NUMDOF")),
      presids_(matdata.parameters.get<std::vector<int>>("PRESCOEFF")),
      functionID_saturation_(matdata.parameters.get<int>("FUNCTSAT")),
      functionID_pressure_(matdata.parameters.get<int>("FUNCTPRES"))
{
  // check if sizes fit
  if (numdof_ != (int)presids_.size())
    FOUR_C_THROW(
        "number of dofs {} does not fit to size of dof vector {}", numdof_, presids_.size());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::FluidPoroPhaseLawByFunction::initialize()
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return initialize_internal<1>();
    case 2:
      return initialize_internal<2>();
    case 3:
      return initialize_internal<3>();
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void Mat::PAR::FluidPoroPhaseLawByFunction::initialize_internal()
{
  if (Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(functionID_saturation_)
          .number_components() != 1)
    FOUR_C_THROW("expected only one component for the saturation evaluation");
  if (Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(functionID_pressure_)
          .number_components() != 1)
    FOUR_C_THROW("expected only one component for the pressure evaluation");


  // initialize pressure vector for function evaluation
  dp_.clear();
  dp_.emplace_back("dp", 0.0);

  // initialize saturation vector for function evaluation
  s_.clear();
  s_.emplace_back("S", 0.0);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_saturation(
    const std::span<const double> pressure)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return evaluate_saturation_internal<1>(pressure);
    case 2:
      return evaluate_saturation_internal<2>(pressure);
    case 3:
      return evaluate_saturation_internal<3>(pressure);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
      return 0.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_saturation_internal(
    const std::span<const double> pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  double presval = std::inner_product(presids_.begin(), presids_.end(), pressure.begin(), 0.0);

  // directly write into entry without checking the name for performance reasons
  dp_[0].second = presval;

  return Global::Problem::instance()
      ->function_by_id<Core::Utils::FunctionOfAnything>(functionID_saturation_)
      .evaluate(dp_, {}, 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_deriv_of_saturation_wrt_pressure(
    int doftoderive, const std::span<const double> pressure)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return evaluate_deriv_of_saturation_wrt_pressure_internal<1>(doftoderive, pressure);
    case 2:
      return evaluate_deriv_of_saturation_wrt_pressure_internal<2>(doftoderive, pressure);
    case 3:
      return evaluate_deriv_of_saturation_wrt_pressure_internal<3>(doftoderive, pressure);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
      return 0.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_deriv_of_saturation_wrt_pressure_internal(
    int doftoderive, const std::span<const double> pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  if (presids_[doftoderive] == 0) return 0.0;

  double presval = std::inner_product(presids_.begin(), presids_.end(), pressure.begin(), 0.0);
  // directly write into entry without checking the name for performance reasons
  dp_[0].second = presval;

  std::vector<double> deriv =
      Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(functionID_saturation_)
          .evaluate_derivative(dp_, {}, 0);

  return deriv[0] * presids_[doftoderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_second_deriv_of_saturation_wrt_pressure(
    int firstdoftoderive, int seconddoftoderive, const std::vector<double>& pressure)
{
  // check if sizes fit
  if (pressure.size() != presids_.size())
    FOUR_C_THROW("number of dofs {} does not fit to size of dof vector {}", pressure.size(),
        presids_.size());

  // TODO: implementation for phaselaw by function --> really necessary???
  return 0.0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_deriv_of_pressure_wrt_saturation(
    int doftoderive, double saturation)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return evaluate_deriv_of_pressure_wrt_saturation_internal<1>(doftoderive, saturation);
    case 2:
      return evaluate_deriv_of_pressure_wrt_saturation_internal<2>(doftoderive, saturation);
    case 3:
      return evaluate_deriv_of_pressure_wrt_saturation_internal<3>(doftoderive, saturation);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
      return 0.0;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_deriv_of_pressure_wrt_saturation_internal(
    int doftoderive, double saturation)
{
  if (presids_[doftoderive] == 0) return 0.0;

  // directly write into entry without checking the name for performance reasons
  s_[0].second = saturation;

  std::vector<double> deriv =
      Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfAnything>(functionID_pressure_)
          .evaluate_derivative(s_, {}, 0);

  return deriv[0] * presids_[doftoderive];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_gen_pressure(double saturation)
{
  switch (Global::Problem::instance()->n_dim())
  {
    case 1:
      return evaluate_gen_pressure_internal<1>(saturation);
    case 2:
      return evaluate_gen_pressure_internal<2>(saturation);
    case 3:
      return evaluate_gen_pressure_internal<3>(saturation);
    default:
      FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
      return 0.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double Mat::PAR::FluidPoroPhaseLawByFunction::evaluate_gen_pressure_internal(double saturation)
{
  // directly write into entry without checking the name for performance reasons
  s_[0].second = saturation;

  return Global::Problem::instance()
      ->function_by_id<Core::Utils::FunctionOfAnything>(functionID_pressure_)
      .evaluate(s_, {}, 0);
}

FOUR_C_NAMESPACE_CLOSE
