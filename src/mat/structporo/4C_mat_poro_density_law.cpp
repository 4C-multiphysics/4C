// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_poro_density_law.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::PoroDensityLaw* Mat::PAR::PoroDensityLaw::create_density_law(int matID)
{
  // initialize null pointer
  Mat::PAR::PoroDensityLaw* densitylaw = nullptr;

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
    case Core::Materials::m_poro_densitylaw_constant:
    {
      densitylaw = static_cast<Mat::PAR::PoroDensityLawConstant*>(curmat);
      break;
    }
    case Core::Materials::m_poro_densitylaw_exp:
    {
      densitylaw = static_cast<Mat::PAR::PoroDensityLawExp*>(curmat);
      break;
    }
    default:
      FOUR_C_THROW("invalid material for density law {}", curmat->type());
      break;
  }

  return densitylaw;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::PoroDensityLawExp::PoroDensityLawExp(const Core::Mat::PAR::Parameter::Data& matdata)
    : PoroDensityLaw(matdata), bulkmodulus_(matdata.parameters.get<double>("BULKMODULUS"))
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::PoroDensityLawExp::create_material()
{
  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::PoroDensityLawExp::compute_cur_density(
    const double& refdensity, const double& press)
{
  return refdensity * exp(press / bulkmodulus_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::PoroDensityLawExp::compute_ref_density_to_cur_density(const double& press)
{
  return exp(-1.0 * press / bulkmodulus_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::PoroDensityLawExp::compute_ref_density_to_cur_density_derivative(
    const double& press)
{
  return -1.0 / bulkmodulus_ * exp(-1.0 * press / bulkmodulus_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::PoroDensityLawExp::compute_ref_density_to_cur_density_second_derivative(
    const double& press)
{
  return 1.0 / (bulkmodulus_ * bulkmodulus_) * exp(-1.0 * press / bulkmodulus_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::PAR::PoroDensityLawExp::compute_cur_density_derivative(
    const double& refdensity, const double& press)
{
  return refdensity / bulkmodulus_ * exp(press / bulkmodulus_);
}

FOUR_C_NAMESPACE_CLOSE
