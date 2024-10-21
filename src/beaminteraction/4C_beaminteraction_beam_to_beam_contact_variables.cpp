#include "4C_beaminteraction_beam_to_beam_contact_variables.hpp"

#include "4C_beaminteraction_beam_to_beam_contact_pair.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  constructor (public)                                     meier 01/14|
 *----------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
BEAMINTERACTION::BeamToBeamContactVariables<numnodes, numnodalvalues>::BeamToBeamContactVariables(
    std::pair<TYPE, TYPE>& closestpoint, std::pair<int, int>& segids, std::pair<int, int>& intids,
    const double& pp, TYPE jacobi)
    : closestpoint_(closestpoint),
      segids_(segids),
      intids_(intids),
      jacobi_(jacobi),
      gap_(0.0),
      normal_(Core::LinAlg::Matrix<3, 1, TYPE>(true)),
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
template class BEAMINTERACTION::BeamToBeamContactVariables<2, 1>;
template class BEAMINTERACTION::BeamToBeamContactVariables<3, 1>;
template class BEAMINTERACTION::BeamToBeamContactVariables<4, 1>;
template class BEAMINTERACTION::BeamToBeamContactVariables<5, 1>;
template class BEAMINTERACTION::BeamToBeamContactVariables<2, 2>;

FOUR_C_NAMESPACE_CLOSE
