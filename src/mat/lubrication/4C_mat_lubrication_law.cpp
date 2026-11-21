// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_lubrication_law.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::LubricationLaw::LubricationLaw(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::LubricationLawConstant::LubricationLawConstant(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : LubricationLaw(matdata), viscosity_(matdata.parameters.get<double>("VISCOSITY"))
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::LubricationLawConstant::create_material()
{
  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::LubricationLawConstant::compute_viscosity(const double& press, double& viscosity)
{
  viscosity = viscosity_;
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mat::PAR::LubricationLawConstant::constitutive_derivatives(
    const double& press, const double& viscosity, double& dviscosity_dp)
{
  dviscosity_dp = 0.0;

  return;
}

/*---------------------------------------------------------------------*
 *  Method definitions for Barus viscosity
 *---------------------------------------------------------------------*/

// Standard Constructor
Mat::PAR::LubricationLawBarus::LubricationLawBarus(const Core::Mat::PAR::Parameter::Data& matdata)
    : LubricationLaw(matdata),
      ABSViscosity_(matdata.parameters.get<double>("ABSViscosity")),
      PreVisCoeff_(matdata.parameters.get<double>("PreVisCoeff"))
{
  return;
}

// Create material instance of matching type with my parameters
std::shared_ptr<Core::Mat::Material> Mat::PAR::LubricationLawBarus::create_material()
{
  return nullptr;
}

// Calculate the current viscosity
void Mat::PAR::LubricationLawBarus::compute_viscosity(const double& press, double& viscosity)
{
  viscosity = ABSViscosity_ * (std::exp(PreVisCoeff_ * press));

  return;
}

// Evaluate constitutive relation for viscosity and compute derivatives
void Mat::PAR::LubricationLawBarus::constitutive_derivatives(
    const double& press, const double& viscosity, double& dviscosity_dp)
{
  dviscosity_dp = viscosity * PreVisCoeff_;

  return;
}

/*---------------------------------------------------------------------*
 *  Method definitions for Roeland viscosity
 *---------------------------------------------------------------------*/

// Standard Constructor
Mat::PAR::LubricationLawRoeland::LubricationLawRoeland(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : LubricationLaw(matdata),
      ABSViscosity_(matdata.parameters.get<double>("ABSViscosity")),
      PreVisCoeff_(matdata.parameters.get<double>("PreVisCoeff")),
      RefVisc_(matdata.parameters.get<double>("RefVisc")),
      RefPress_(matdata.parameters.get<double>("RefPress"))
{
  z_ = (PreVisCoeff_ * RefPress_) / (log(ABSViscosity_ / RefVisc_));
  return;
}

// Create material instance of matching type with my parameters
std::shared_ptr<Core::Mat::Material> Mat::PAR::LubricationLawRoeland::create_material()
{
  return nullptr;
}

// Calculate the current viscosity
void Mat::PAR::LubricationLawRoeland::compute_viscosity(const double& press, double& viscosity)
{
  // double z = (PreVisCoeff_ * RefPress_) / (log ( ABSViscosity_ / RefVisc_ ));

  viscosity =
      ABSViscosity_ * exp(log(ABSViscosity_ / RefVisc_) * (pow((1 + press / RefPress_), z_) - 1));

  return;
}

// Evaluate constitutive relation for viscosity and compute derivatives
void Mat::PAR::LubricationLawRoeland::constitutive_derivatives(
    const double& press, const double& viscosity, double& dviscosity_dp)
{
  // double z = (PreVisCoeff_ * RefPress_ ) / (log ( ABSViscosity_ / RefVisc_ ));

  dviscosity_dp = viscosity * log(ABSViscosity_ / RefVisc_) * z_ *
                  pow((1 + press / RefPress_), (z_ - 1)) * (1 / RefPress_);

  return;
}

FOUR_C_NAMESPACE_CLOSE
