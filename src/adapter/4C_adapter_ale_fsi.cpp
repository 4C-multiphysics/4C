/*----------------------------------------------------------------------------*/
/*! \file

 \brief FSI Wrapper for the ALE time integration

 \level 1


 */
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/* header inclusions */
#include "4C_adapter_ale_fsi.hpp"

#include "4C_ale_utils_mapextractor.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Adapter::AleFsiWrapper::AleFsiWrapper(Teuchos::RCP<Ale> ale) : AleWrapper(ale)
{
  // create the FSI interface
  interface_ = Teuchos::make_rcp<ALE::Utils::MapExtractor>();
  interface_->setup(*discretization());

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<const ALE::Utils::MapExtractor> Adapter::AleFsiWrapper::interface() const
{
  return interface_;
}

FOUR_C_NAMESPACE_CLOSE
