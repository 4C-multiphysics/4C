// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beamcontact_beam3contactvariables.hpp"

#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  constructor (public)                                     meier 01/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
CONTACT::Beam3contactvariables<numnodes, numnodalvalues>::Beam3contactvariables(
    std::pair<TYPE, TYPE>& closestpoint, std::pair<int, int>& segids, std::pair<int, int>& intids,
    const double& pp, TYPE jacobi)
    : closestpoint_(closestpoint),
      segids_(segids),
      intids_(intids),
      jacobi_(jacobi),
      gap_(0.0),
      normal_(Core::LinAlg::Matrix<3, 1, TYPE>(Core::LinAlg::Initialization::zero)),
      pp_(pp),
      ppfac_(0.0),
      dppfac_(0.0),
      fp_(0.0),
      dfp_(0.0),
      energy_(0.0),
      integratedenergy_(0.0),
      angle_(0.0)
{
  return;
}
/*----------------------------------------------------------------------*
 |  end: constructor
 *----------------------------------------------------------------------*/

// Possible template cases: this is necessary for the compiler
template class CONTACT::Beam3contactvariables<2, 1>;
template class CONTACT::Beam3contactvariables<3, 1>;
template class CONTACT::Beam3contactvariables<4, 1>;
template class CONTACT::Beam3contactvariables<5, 1>;
template class CONTACT::Beam3contactvariables<2, 2>;

FOUR_C_NAMESPACE_CLOSE
