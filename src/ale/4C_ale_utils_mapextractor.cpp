// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_utils_mapextractor.hpp"

#include "4C_fem_condition_selector.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Utils::MapExtractor::setup(const Core::FE::Discretization& dis, bool overlapping)
{
  const int ndim = Global::Problem::instance()->n_dim();
  Core::Conditions::setup_extractor(dis, *this,
      {
          Core::Conditions::Selector("FSICoupling", 0, ndim),
          Core::Conditions::Selector("FREESURFCoupling", 0, ndim),
          Core::Conditions::Selector("StructAleCoupling", 0, ndim),
          Core::Conditions::Selector("AleWear", 0, ndim),
          Core::Conditions::Selector("BioGrCoupling", 0, ndim),
          Core::Conditions::Selector("ALEUPDATECoupling", 0, ndim),
          Core::Conditions::Selector("fpsi_coupling", 0, ndim),
          Core::Conditions::Selector("Mortar", 0, ndim),
      },
      overlapping);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<std::set<int>> ALE::Utils::MapExtractor::conditioned_element_map(
    const Core::FE::Discretization& dis) const
{
  std::shared_ptr<std::set<int>> condelements =
      Core::Conditions::conditioned_element_map(dis, "FSICoupling");
  std::shared_ptr<std::set<int>> condelements2 =
      Core::Conditions::conditioned_element_map(dis, "FREESURFCoupling");
  std::shared_ptr<std::set<int>> condelements3 =
      Core::Conditions::conditioned_element_map(dis, "StructAleCoupling");
  std::shared_ptr<std::set<int>> condelements4 =
      Core::Conditions::conditioned_element_map(dis, "AleWear");
  std::shared_ptr<std::set<int>> condelements5 =
      Core::Conditions::conditioned_element_map(dis, "BioGrCoupling");
  std::shared_ptr<std::set<int>> condelements6 =
      Core::Conditions::conditioned_element_map(dis, "ALEUPDATECoupling");
  std::shared_ptr<std::set<int>> condelements7 =
      Core::Conditions::conditioned_element_map(dis, "fpsi_coupling");
  std::shared_ptr<std::set<int>> condelements8 =
      Core::Conditions::conditioned_element_map(dis, "Mortar");
  std::copy(condelements2->begin(), condelements2->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements3->begin(), condelements3->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements4->begin(), condelements4->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements5->begin(), condelements5->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements6->begin(), condelements6->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements7->begin(), condelements7->end(),
      std::inserter(*condelements, condelements->begin()));
  std::copy(condelements8->begin(), condelements8->end(),
      std::inserter(*condelements, condelements->begin()));
  return condelements;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ALE::Utils::FsiMapExtractor::setup(const Core::FE::Discretization& dis)
{
  const int ndim = Global::Problem::instance()->n_dim();
  Core::Conditions::setup_extractor(
      dis, *this, {Core::Conditions::Selector("FSICoupling", 0, ndim)});
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void ALE::Utils::XFluidFluidMapExtractor::setup(const Core::FE::Discretization& dis)
{
  const int ndim = Global::Problem::instance()->n_dim();
  Core::Conditions::setup_extractor(
      dis, *this, {Core::Conditions::Selector("FluidFluidCoupling", 0, ndim)});
}

FOUR_C_NAMESPACE_CLOSE
