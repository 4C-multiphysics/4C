// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_ml_surrogate_contactconstitutivelaw.hpp"

#include "4C_contact_rough_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_par_bundle.hpp"

// #ifdef FOUR_C_WITH_ROUGH_SURFACE_ML_SURROGATE

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MLSurrogateConstitutiveLawParams::MLSurrogateConstitutiveLawParams(
    const Core::IO::InputParameterContainer& container)
    : CONTACT::CONSTITUTIVELAW::Parameter(container)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::MLSurrogateConstitutiveLaw::MLSurrogateConstitutiveLaw(
    CONTACT::CONSTITUTIVELAW::MLSurrogateConstitutiveLawParams params)
    : params_(std::move(params))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::MLSurrogateConstitutiveLaw::evaluate(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes. ");
  }

  double pressure = 0.0;

  return (-1 * pressure);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double CONTACT::CONSTITUTIVELAW::MLSurrogateConstitutiveLaw::evaluate_derivative(
    const double gap, CONTACT::Node* cnode)
{
  if (gap + params_.get_offset() > 0.0)
  {
    FOUR_C_THROW("You should not be here. The Evaluate function is only tested for active nodes.");
  }

  double derivative = 0.0;

  return derivative;
}

FOUR_C_NAMESPACE_CLOSE

// #endif
