// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"

#include "4C_contact_constitutivelaw_brokenrational_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_bundle.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_contact_constitutivelaw_cubic_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_linear_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_power_contactconstitutivelaw.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_enum.hpp"

#ifdef FOUR_C_WITH_MIRCO
#include "4C_contact_constitutivelaw_mirco_contactconstitutivelaw.hpp"
#endif

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::unique_ptr<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw>
CONTACT::CONSTITUTIVELAW::ConstitutiveLaw::factory(const int id)
{
  const int probinst =
      Global::Problem::instance()->contact_constitutive_laws()->get_read_from_problem();

  // for the sake of safety
  if (Global::Problem::instance(probinst)->contact_constitutive_laws() == nullptr)
    FOUR_C_THROW("Cannot work out problem instance!");
  // yet another safety check
  if (Global::Problem::instance(probinst)->contact_constitutive_laws()->num() == 0)
    FOUR_C_THROW("Cannot find any contact constitutive law!");

  // retrieve validated input line of material ID in question
  auto& coconstlawdata =
      Global::Problem::instance(probinst)->contact_constitutive_laws()->by_id(id);

  const auto type = coconstlawdata.get<CONTACT::CONSTITUTIVELAW::ConstitutiveLawType>("LAW_TYPE");

  switch (type)
  {
    case CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_cubic:
    {
      CONTACT::CONSTITUTIVELAW::CubicConstitutiveLawParams params(
          coconstlawdata.group("CoConstLaw_cubic"));
      return std::make_unique<CONTACT::CONSTITUTIVELAW::CubicConstitutiveLaw>(params);
    }
    case CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_brokenrational:
    {
      CONTACT::CONSTITUTIVELAW::BrokenRationalConstitutiveLawParams params(
          coconstlawdata.group("CoConstLaw_brokenrational"));
      return std::make_unique<CONTACT::CONSTITUTIVELAW::BrokenRationalConstitutiveLaw>(params);
    }

    case CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_linear:
    {
      CONTACT::CONSTITUTIVELAW::LinearConstitutiveLawParams params(
          coconstlawdata.group("CoConstLaw_linear"));
      return std::make_unique<CONTACT::CONSTITUTIVELAW::LinearConstitutiveLaw>(params);
    }
    case CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_power:
    {
      CONTACT::CONSTITUTIVELAW::PowerConstitutiveLawParams params(
          coconstlawdata.group("CoConstLaw_power"));
      return std::make_unique<CONTACT::CONSTITUTIVELAW::PowerConstitutiveLaw>(params);
    }
    case CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_mirco:
    {
#ifdef FOUR_C_WITH_MIRCO
      CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params(
          coconstlawdata.group("CoConstLaw_mirco"));
      return std::make_unique<CONTACT::CONSTITUTIVELAW::MircoConstitutiveLaw>(params);
#else
      FOUR_C_THROW(
          "You are trying to use MIRCO contact consitutive law with FOUR_C_WITH_MIRCO flag turned "
          "off. Please enable this flag and configure & build 4C again");
#endif
    }
    case CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_none:
    {
      FOUR_C_THROW("No contact constitutive law found\n");
    }
    default:
      FOUR_C_THROW("unknown type of contact constitutive law {}\n", type);
  }
}

FOUR_C_NAMESPACE_CLOSE
