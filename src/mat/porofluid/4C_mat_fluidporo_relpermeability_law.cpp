// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluidporo_relpermeability_law.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroRelPermeabilityLaw*
Mat::PAR::FluidPoroRelPermeabilityLaw::create_rel_permeability_law(int matID)
{
  // initialize null pointer
  Mat::PAR::FluidPoroRelPermeabilityLaw* relpermeabilitylaw = nullptr;

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");
  // yet another safety check
  if (Global::Problem::instance(probinst)->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matID);

  switch (curmat->type())
  {
    case Core::Materials::m_fluidporo_relpermeabilitylaw_constant:
    {
      relpermeabilitylaw = static_cast<Mat::PAR::FluidPoroRelPermeabilityLawConstant*>(curmat);
      break;
    }
    case Core::Materials::m_fluidporo_relpermeabilitylaw_exp:
    {
      relpermeabilitylaw = static_cast<Mat::PAR::FluidPoroRelPermeabilityLawExponent*>(curmat);
      break;
    }
    default:
      FOUR_C_THROW("invalid material for permeability law {}", curmat->type());
      break;
  }

  return relpermeabilitylaw;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroRelPermeabilityLawConstant::FluidPoroRelPermeabilityLawConstant(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroRelPermeabilityLaw(matdata, true),
      relpermeability_(matdata.parameters.get<double>("VALUE"))
{
  if (relpermeability_ > 1.0)
    FOUR_C_THROW(
        "relative permeability (actually the sum of the relative permeabilites) of phase cannot be "
        "greater than 1.0");
  return;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroRelPermeabilityLawExponent::FluidPoroRelPermeabilityLawExponent(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroRelPermeabilityLaw(matdata, false),
      exp_(matdata.parameters.get<double>("EXP")),
      minsat_(matdata.parameters.get<double>("MIN_SAT"))
{
  if (exp_ <= 1.0)
    FOUR_C_THROW("exponent in relative permeability phase law has to be bigger than 1.0");
  // if(minsat_ < 0.0 or minsat_ > 1.0)
  //  FOUR_C_THROW("minimal saturation has to be between 0 and 1");
  return;
}

FOUR_C_NAMESPACE_CLOSE
