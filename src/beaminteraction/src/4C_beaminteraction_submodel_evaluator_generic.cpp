// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_submodel_evaluator_generic.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_crosslinker_handler.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::SubmodelEvaluator::Generic::Generic()
    : isinit_(false),
      issetup_(false),
      discret_ptr_(nullptr),
      bindis_ptr_(nullptr),
      gstate_ptr_(nullptr),
      gio_ptr_(nullptr),
      beaminteractiondatastate_(nullptr),
      beam_crosslinker_handler_(nullptr),
      binstrategy_(nullptr),
      periodic_boundingbox_(nullptr),
      eletypeextractor_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Generic::init(
    std::shared_ptr<Core::FE::Discretization> const& ia_discret,
    std::shared_ptr<Core::FE::Discretization> const& bindis,
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> const& gstate,
    std::shared_ptr<Solid::TimeInt::BaseDataIO> const& gio_ptr,
    std::shared_ptr<Solid::ModelEvaluator::BeamInteractionDataState> const& ia_gstate_ptr,
    std::shared_ptr<BeamInteraction::BeamCrosslinkerHandler> const& beamcrosslinkerhandler,
    std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy,
    std::shared_ptr<Core::Geo::MeshFree::BoundingBox> const& periodic_boundingbox,
    std::shared_ptr<BeamInteraction::Utils::MapExtractor> const& eletypeextractor)
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
void BeamInteraction::SubmodelEvaluator::Generic::check_init_setup() const
{
  if (!is_init() or !is_setup()) FOUR_C_THROW("Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::Generic::check_init() const
{
  if (not is_init()) FOUR_C_THROW("Call init() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::Discretization& BeamInteraction::SubmodelEvaluator::Generic::discret()
{
  check_init();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization>&
BeamInteraction::SubmodelEvaluator::Generic::discret_ptr()
{
  check_init();
  return discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::FE::Discretization>
BeamInteraction::SubmodelEvaluator::Generic::discret_ptr() const
{
  check_init();
  return discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::Discretization const& BeamInteraction::SubmodelEvaluator::Generic::discret() const
{
  check_init();
  return *discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::Discretization& BeamInteraction::SubmodelEvaluator::Generic::bin_discret()
{
  check_init();
  return *bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization>&
BeamInteraction::SubmodelEvaluator::Generic::bin_discret_ptr()
{
  check_init();
  return bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::FE::Discretization>
BeamInteraction::SubmodelEvaluator::Generic::bin_discret_ptr() const
{
  check_init();
  return bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::FE::Discretization const& BeamInteraction::SubmodelEvaluator::Generic::bin_discret() const
{
  check_init();
  return *bindis_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataGlobalState& BeamInteraction::SubmodelEvaluator::Generic::g_state()
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>&
BeamInteraction::SubmodelEvaluator::Generic::g_state_ptr()
{
  check_init();
  return gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataGlobalState const& BeamInteraction::SubmodelEvaluator::Generic::g_state()
    const
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataIO& BeamInteraction::SubmodelEvaluator::Generic::g_in_output()
{
  check_init();
  return *gio_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataIO const& BeamInteraction::SubmodelEvaluator::Generic::g_in_output() const
{
  check_init();
  return *gio_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::BeamInteractionDataState&
BeamInteraction::SubmodelEvaluator::Generic::beam_interaction_data_state()
{
  check_init();
  return *beaminteractiondatastate_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::ModelEvaluator::BeamInteractionDataState>&
BeamInteraction::SubmodelEvaluator::Generic::beam_interaction_data_state_ptr()
{
  check_init();
  return beaminteractiondatastate_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const std::shared_ptr<Solid::ModelEvaluator::BeamInteractionDataState>&
BeamInteraction::SubmodelEvaluator::Generic::beam_interaction_data_state_ptr() const
{
  check_init();
  return beaminteractiondatastate_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::BeamInteractionDataState const&
BeamInteraction::SubmodelEvaluator::Generic::beam_interaction_data_state() const
{
  check_init();
  return *beaminteractiondatastate_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamCrosslinkerHandler&
BeamInteraction::SubmodelEvaluator::Generic::beam_crosslinker_handler()
{
  check_init();
  return *beam_crosslinker_handler_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<BeamInteraction::BeamCrosslinkerHandler>&
BeamInteraction::SubmodelEvaluator::Generic::beam_crosslinker_handler_ptr()
{
  check_init();
  return beam_crosslinker_handler_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Binstrategy::BinningStrategy const&
BeamInteraction::SubmodelEvaluator::Generic::bin_strategy() const
{
  check_init();
  return *binstrategy_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Binstrategy::BinningStrategy& BeamInteraction::SubmodelEvaluator::Generic::bin_strategy()
{
  check_init();
  return *binstrategy_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::Binstrategy::BinningStrategy>&
BeamInteraction::SubmodelEvaluator::Generic::bin_strategy_ptr()
{
  check_init();
  return binstrategy_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamCrosslinkerHandler const&
BeamInteraction::SubmodelEvaluator::Generic::beam_crosslinker_handler() const
{
  check_init();
  return *beam_crosslinker_handler_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Geo::MeshFree::BoundingBox&
BeamInteraction::SubmodelEvaluator::Generic::periodic_bounding_box()
{
  check_init();
  return *periodic_boundingbox_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::Geo::MeshFree::BoundingBox>&
BeamInteraction::SubmodelEvaluator::Generic::periodic_bounding_box_ptr()
{
  check_init();
  return periodic_boundingbox_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::Geo::MeshFree::BoundingBox const&
BeamInteraction::SubmodelEvaluator::Generic::periodic_bounding_box() const
{
  check_init();
  return *periodic_boundingbox_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::Utils::MapExtractor&
BeamInteraction::SubmodelEvaluator::Generic::ele_type_map_extractor()
{
  check_init();
  eletypeextractor_->check_for_valid_map_extractor();
  return *eletypeextractor_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<BeamInteraction::Utils::MapExtractor>&
BeamInteraction::SubmodelEvaluator::Generic::ele_type_map_extractor_ptr()
{
  check_init();
  eletypeextractor_->check_for_valid_map_extractor();
  return eletypeextractor_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::Utils::MapExtractor const&
BeamInteraction::SubmodelEvaluator::Generic::ele_type_map_extractor() const
{
  check_init();
  eletypeextractor_->check_for_valid_map_extractor();
  return *eletypeextractor_;
}

FOUR_C_NAMESPACE_CLOSE
