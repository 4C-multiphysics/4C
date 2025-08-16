// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_cubic_contactconstitutivelaw.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::CubicConstitutiveLawParams::CubicConstitutiveLawParams(
    const Core::IO::InputParameterContainer& container)
    : CONTACT::CONSTITUTIVELAW::Parameter(container),
      a_(container.get<double>("A")),
      b_(container.get<double>("B")),
      c_(container.get<double>("C")),
      d_(container.get<double>("D"))
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::CubicConstitutiveLaw::CubicConstitutiveLaw(
    CONTACT::CONSTITUTIVELAW::CubicConstitutiveLawParams params)
    : params_(std::move(params))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::CubicConstitutiveLaw::evaluate(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes. ");
  }

  const double negative_gap = -gap;

  const double result = -params_.getdata() * (negative_gap - params_.get_offset()) *
                            (negative_gap - params_.get_offset()) *
                            (negative_gap - params_.get_offset()) -
                        params_.get_b() * (negative_gap - params_.get_offset()) *
                            (negative_gap - params_.get_offset()) -
                        params_.get_c() * (negative_gap - params_.get_offset()) - params_.get_d();

  if (result > 0)
    FOUR_C_THROW(
        "The constitutive function you are using seems to be positive, even though the gap is "
        "negative. Please check your coefficients!");

  return result;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::CubicConstitutiveLaw::evaluate_derivative(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes. ");
  }

  return 3 * params_.getdata() * (-gap - params_.get_offset()) * (-gap - params_.get_offset()) +
         2 * params_.get_b() * (-gap - params_.get_offset()) + params_.get_c();
}

FOUR_C_NAMESPACE_CLOSE
