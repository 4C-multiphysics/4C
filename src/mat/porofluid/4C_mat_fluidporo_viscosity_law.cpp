// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluidporo_viscosity_law.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroViscosityLaw* Mat::PAR::FluidPoroViscosityLaw::create_viscosity_law(int matID)
{
  // initialize null pointer
  Mat::PAR::FluidPoroViscosityLaw* viscositylaw = nullptr;

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matID);

  switch (curmat->type())
  {
    case Core::Materials::m_fluidporo_viscositylaw_constant:
    {
      viscositylaw = static_cast<Mat::PAR::FluidPoroViscosityLawConstant*>(curmat);
      break;
    }
    case Core::Materials::m_fluidporo_viscositylaw_celladh:
    {
      viscositylaw = static_cast<Mat::PAR::FluidPoroViscosityLawCellAdherence*>(curmat);
      break;
    }
    default:
      FOUR_C_THROW("invalid material for viscosity law {}", curmat->type());
      break;
  }

  return viscositylaw;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroViscosityLawConstant::FluidPoroViscosityLawConstant(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroViscosityLaw(matdata, true), viscosity_(matdata.parameters.get<double>("VALUE"))
{
  return;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroViscosityLawCellAdherence::FluidPoroViscosityLawCellAdherence(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroViscosityLaw(matdata, false),
      visc0_(matdata.parameters.get<double>("VISC_0")),
      xi_(matdata.parameters.get<double>("XI")),
      psi_(matdata.parameters.get<double>("PSI"))

{
  if (visc0_ <= 0.0) FOUR_C_THROW("VISC_0 cannot be smaller or equal to zero!");
  if (xi_ <= 0.0) FOUR_C_THROW("XI cannot be smaller or equal to zero!");
  if (psi_ <= 0.0) FOUR_C_THROW("PSI cannot be smaller or equal to zero!");

  return;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroViscosityLawCellAdherence::get_viscosity(const double abspressgrad) const
{
  // visc = visc0 / ((1 - xi)*(1 - psi / | grad(pressure) |) * heaviside(1 - psi / | grad(pressure)
  // |) + xi)

  if (abspressgrad <= psi_)  // case heaviside(1 - psi / | grad(pressure) |) = 0
    return visc0_ / xi_;
  else  // case heaviside(1 - psi / | grad(pressure) |) = 1
    return visc0_ / ((1.0 - xi_) * (1.0 - psi_ / abspressgrad) + xi_);
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::FluidPoroViscosityLawCellAdherence::get_deriv_of_viscosity_wrt_abs_press_grad(
    const double abspressgrad) const
{
  // visc = visc0 / ((1 - xi)*(1 - psi / | grad(pressure) |) * heaviside(1 - psi / | grad(pressure)
  // |) + xi) case heaviside = 1: dvisc / d|grad(pressure) = visc0 * (xi - 1.0) * psi / ((xi - 1.0)
  // * psi + | grad(pressure) |) ^ 2

  if (abspressgrad <= psi_)  // case heaviside(1 - psi / | grad(pressure) |) = 0
    return 0.0;
  else  // case heaviside(1 - psi / | grad(pressure) |) = 1
    return visc0_ * (xi_ - 1.0) * psi_ / ((xi_ - 1.0) * psi_ + abspressgrad) /
           ((xi_ - 1.0) * psi_ + abspressgrad);
}

FOUR_C_NAMESPACE_CLOSE
