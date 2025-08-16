// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_bundle.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::Bundle::Bundle() : readfromproblem_(0) {}
/*----------------------------------------------------------------------*/
void CONTACT::CONSTITUTIVELAW::Bundle::insert(
    int matid, Core::IO::InputParameterContainer container)
{
  map_.emplace(matid, std::move(container));
}

/*----------------------------------------------------------------------*/
int CONTACT::CONSTITUTIVELAW::Bundle::find(const int id) const
{
  if (map_.find(id) == map_.end())
    return -1;
  else
    return map_.find(id)->first;
}

/*----------------------------------------------------------------------*/
void CONTACT::CONSTITUTIVELAW::Bundle::make_parameters()
{
  for (const auto& [id, law] : map_)
  {
    // indirectly add quick access parameter members as a side effect of construction
    [[maybe_unused]] auto _ = CONTACT::CONSTITUTIVELAW::ConstitutiveLaw::factory(id);
  }
}

/*----------------------------------------------------------------------*/
int CONTACT::CONSTITUTIVELAW::Bundle::num() const { return map_.size(); }

/*----------------------------------------------------------------------*/
Core::IO::InputParameterContainer& CONTACT::CONSTITUTIVELAW::Bundle::by_id(const int id)
{
  if (map_.size() == 0) FOUR_C_THROW("No contact constitutivelaws available, num={}", id);

  auto m = map_.find(id);

  if (m == map_.end())
    FOUR_C_THROW("Contact Constitutive Law 'Law {}' could not be found", id);
  else
    return m->second;
}

FOUR_C_NAMESPACE_CLOSE
