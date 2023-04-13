/*----------------------------------------------------------------------*/
/*! \file
\brief Base class for regularization of optimization problems

\level 3

*/

/*----------------------------------------------------------------------*/
/* headers */
#include "inv_analysis_regularization_base.H"

#include "inv_analysis_initial_guess.H"

#include "linalg_mapextractor.H"
#include <Teuchos_ParameterList.hpp>


/*----------------------------------------------------------------------*/
/* constructor */
INVANA::RegularizationBase::RegularizationBase() : connectivity_(Teuchos::null), weight_(0.0) {}

void INVANA::RegularizationBase::Init(const Teuchos::ParameterList& invp,
    Teuchos::RCP<ConnectivityData> connectivity, Teuchos::RCP<InitialGuess> initguess)
{
  connectivity_ = connectivity;
  initguess_ = initguess;

  weight_ = invp.get<double>("REG_WEIGHT");

  return;
}
