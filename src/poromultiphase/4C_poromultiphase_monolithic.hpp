#ifndef FOUR_C_POROMULTIPHASE_MONOLITHIC_HPP
#define FOUR_C_POROMULTIPHASE_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_poromultiphase_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace POROMULTIPHASE
{
  //! Base class of all solid-scatra algorithms
  class PoroMultiPhaseMonolithic : public PoroMultiPhaseBase
  {
   public:
    /// create using a Epetra_Comm
    PoroMultiPhaseMonolithic(const Epetra_Comm& comm,
        const Teuchos::ParameterList& globaltimeparams);  // Problem builder


  };  // PoroMultiPhaseMonolithic


}  // namespace POROMULTIPHASE



FOUR_C_NAMESPACE_CLOSE

#endif
