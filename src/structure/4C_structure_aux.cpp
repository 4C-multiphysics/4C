// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_aux.hpp"

#include "4C_fem_condition_selector.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* Calculate vector norm */
double Solid::calculate_vector_norm(const enum Inpar::Solid::VectorNorm norm,
    const Core::LinAlg::Vector<double>& vect, const int numneglect)
{
  // L1 norm
  if (norm == Inpar::Solid::norm_l1)
  {
    double vectnorm;
    vect.norm_1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  else if (norm == Inpar::Solid::norm_l2)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  else if (norm == Inpar::Solid::norm_rms)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm / sqrt((double)(vect.global_length() - numneglect));
  }
  // infinity/maximum norm
  else if (norm == Inpar::Solid::norm_inf)
  {
    double vectnorm;
    vect.norm_inf(&vectnorm);
    return vectnorm;
  }
  else
  {
    FOUR_C_THROW("Cannot handle vector norm");
    return 0;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::MapExtractor::setup(
    const Core::FE::Discretization& dis, const Core::LinAlg::Map& fullmap, bool overlapping)
{
  const int ndim = Global::Problem::instance()->n_dim();
  Core::Conditions::setup_extractor(dis, fullmap, *this,
      {
          Core::Conditions::Selector("FSICoupling", 0, ndim),
          Core::Conditions::Selector("StructAleCoupling", 0, ndim),
          Core::Conditions::Selector("BioGrCoupling", 0, ndim),
          Core::Conditions::Selector("AleWear", 0, ndim),
          Core::Conditions::Selector("fpsi_coupling", 0, ndim),
          Core::Conditions::Selector("IMMERSEDCoupling", 0, ndim),
          Core::Conditions::Selector("ParticleWall", 0, ndim),
      },
      overlapping);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::set<int>> Solid::MapExtractor::conditioned_element_map(
    const Core::FE::Discretization& dis) const
{
  std::shared_ptr<std::set<int>> condelements =
      Core::Conditions::conditioned_element_map(dis, "FSICoupling");
  std::shared_ptr<std::set<int>> condelements2 =
      Core::Conditions::conditioned_element_map(dis, "StructAleCoupling");
  std::shared_ptr<std::set<int>> condelements3 =
      Core::Conditions::conditioned_element_map(dis, "BioGrCoupling");
  std::shared_ptr<std::set<int>> condelements4 =
      Core::Conditions::conditioned_element_map(dis, "AleWear");
  std::shared_ptr<std::set<int>> condelements5 =
      Core::Conditions::conditioned_element_map(dis, "fpsi_coupling");
  std::shared_ptr<std::set<int>> condelements6 =
      Core::Conditions::conditioned_element_map(dis, "IMMERSEDCoupling");
  std::shared_ptr<std::set<int>> condelements7 =
      Core::Conditions::conditioned_element_map(dis, "ParticleWall");


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
  return condelements;
}

FOUR_C_NAMESPACE_CLOSE
