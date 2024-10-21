#ifndef FOUR_C_POROMULTIPHASE_SCATRA_MONOLITHIC_HPP
#define FOUR_C_POROMULTIPHASE_SCATRA_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_poromultiphase_scatra_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace PoroMultiPhaseScaTra
{
  //! Base class of all solid-scatra algorithms
  class PoroMultiPhaseScaTraMonolithic : public PoroMultiPhaseScaTraBase
  {
   public:
    /// create using a Epetra_Comm
    PoroMultiPhaseScaTraMonolithic(
        const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams)
        : PoroMultiPhaseScaTraBase(comm, globaltimeparams){};  // Problem builder


  };  // PoroMultiPhaseMonolithic


}  // namespace PoroMultiPhaseScaTra



FOUR_C_NAMESPACE_CLOSE

#endif
