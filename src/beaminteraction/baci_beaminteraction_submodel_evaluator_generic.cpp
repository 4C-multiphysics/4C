/*-----------------------------------------------------------*/
/*! \file

\brief Generic class for all beaminteraction submodel evaluators.


\level 3

*/
/*-----------------------------------------------------------*/


#include "baci_beaminteraction_submodel_evaluator_generic.H"
#include "baci_beaminteraction_str_model_evaluator_datastate.H"
#include "baci_beaminteraction_calc_utils.H"

#include "baci_beaminteraction_periodic_boundingbox.H"
#include "baci_utils_exceptions.H"

#include "baci_beaminteraction_crosslinker_handler.H"



/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::Generic()
    : isinit_(false),
      issetup_(false),
      discret_ptr_(Teuchos::null),
      bindis_ptr_(Teuchos::null),
      gstate_ptr_(Teuchos::null),
      gio_ptr_(Teuchos::null),
      beaminteractiondatastate_(Teuchos::null),
      beam_crosslinker_handler_(Teuchos::null),
      binstrategy_(Teuchos::null),
      periodic_boundingbox_(Teuchos::null),
      eletypeextractor_(Teuchos::null)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Generic::Init(
    Teuchos::RCP<::DRT::Discretization> const& ia_discret,
    Teuchos::RCP<::DRT::Discretization> const& bindis,
    Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> const& gstate,
    Teuchos::RCP<STR::TIMINT::BaseDataIO> const& gio_ptr,
    Teuchos::RCP<STR::MODELEVALUATOR::BeamInteractionDataState> const& ia_gstate_ptr,
    Teuchos::RCP<BEAMINTERACTION::BeamCrosslinkerHandler> const& beamcrosslinkerhandler,
    Teuchos::RCP<BINSTRATEGY::BinningStrategy> binstrategy,
    Teuchos::RCP<CORE::GEO::MESHFREE::BoundingBox> const& periodic_boundingbox,
    Teuchos::RCP<BEAMINTERACTION::UTILS::MapExtractor> const& eletypeextractor)
{
  issetup_ = false;

  discret_ptr_ = ia_discret;
  bindis_ptr_ = bindis;
  gstate_ptr_ = gstate;
  gio_ptr_ = gio_ptr;
  beaminteractiondatastate_ = ia_gstate_ptr;
  beam_crosslinker_handler_ = beamcrosslinkerhandler;
  binstrategy_ = binstrategy;
  periodic_boundingbox_ = periodic_boundingbox;
  eletypeextractor_ = eletypeextractor;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Generic::CheckInitSetup() const
{
  if (!IsInit() or !IsSetup()) dserror("Call Init() and Setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Generic::CheckInit() const
{
  if (not IsInit()) dserror("Call Init() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
DRT::Discretization& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::Discret()
{
  CheckInit();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::DRT::Discretization>& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::DiscretPtr()
{
  CheckInit();
  return discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const ::DRT::Discretization> BEAMINTERACTION::SUBMODELEVALUATOR::Generic::DiscretPtr()
    const
{
  CheckInit();
  return discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
DRT::Discretization const& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::Discret() const
{
  CheckInit();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
DRT::Discretization& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BinDiscret()
{
  CheckInit();
  return *bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<::DRT::Discretization>& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BinDiscretPtr()
{
  CheckInit();
  return bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const ::DRT::Discretization>
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BinDiscretPtr() const
{
  CheckInit();
  return bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
DRT::Discretization const& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BinDiscret() const
{
  CheckInit();
  return *bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::BaseDataGlobalState& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::GState()
{
  CheckInit();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataGlobalState>&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::GStatePtr()
{
  CheckInit();
  return gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::BaseDataGlobalState const& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::GState() const
{
  CheckInit();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::BaseDataIO& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::GInOutput()
{
  CheckInit();
  return *gio_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::BaseDataIO const& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::GInOutput() const
{
  CheckInit();
  return *gio_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::MODELEVALUATOR::BeamInteractionDataState&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BeamInteractionDataState()
{
  CheckInit();
  return *beaminteractiondatastate_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::MODELEVALUATOR::BeamInteractionDataState>&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BeamInteractionDataStatePtr()
{
  CheckInit();
  return beaminteractiondatastate_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::MODELEVALUATOR::BeamInteractionDataState const&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BeamInteractionDataState() const
{
  CheckInit();
  return *beaminteractiondatastate_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::BeamCrosslinkerHandler&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BeamCrosslinkerHandler()
{
  CheckInit();
  return *beam_crosslinker_handler_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<BEAMINTERACTION::BeamCrosslinkerHandler>&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BeamCrosslinkerHandlerPtr()
{
  CheckInit();
  return beam_crosslinker_handler_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BINSTRATEGY::BinningStrategy const& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BinStrategy() const
{
  CheckInit();
  return *binstrategy_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BINSTRATEGY::BinningStrategy& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BinStrategy()
{
  CheckInit();
  return *binstrategy_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<BINSTRATEGY::BinningStrategy>&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BinStrategyPtr()
{
  CheckInit();
  return binstrategy_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::BeamCrosslinkerHandler const&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::BeamCrosslinkerHandler() const
{
  CheckInit();
  return *beam_crosslinker_handler_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CORE::GEO::MESHFREE::BoundingBox& BEAMINTERACTION::SUBMODELEVALUATOR::Generic::PeriodicBoundingBox()
{
  CheckInit();
  return *periodic_boundingbox_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::GEO::MESHFREE::BoundingBox>&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::PeriodicBoundingBoxPtr()
{
  CheckInit();
  return periodic_boundingbox_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
CORE::GEO::MESHFREE::BoundingBox const&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::PeriodicBoundingBox() const
{
  CheckInit();
  return *periodic_boundingbox_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::UTILS::MapExtractor&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::EleTypeMapExtractor()
{
  CheckInit();
  eletypeextractor_->CheckForValidMapExtractor();
  return *eletypeextractor_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<BEAMINTERACTION::UTILS::MapExtractor>&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::EleTypeMapExtractorPtr()
{
  CheckInit();
  eletypeextractor_->CheckForValidMapExtractor();
  return eletypeextractor_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BEAMINTERACTION::UTILS::MapExtractor const&
BEAMINTERACTION::SUBMODELEVALUATOR::Generic::EleTypeMapExtractor() const
{
  CheckInit();
  eletypeextractor_->CheckForValidMapExtractor();
  return *eletypeextractor_;
}
