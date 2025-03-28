// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_data.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamInteractionParams::BeamInteractionParams()
    : isinit_(false),
      issetup_(false),
      rep_strategy_(Inpar::BeamInteraction::repstr_adaptive),
      search_strategy_(Inpar::BeamInteraction::SearchStrategy::bruteforce_with_binning)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamInteractionParams::init()
{
  issetup_ = false;

  Teuchos::ParameterList const& params_list =
      Global::Problem::instance()->beam_interaction_params();

  rep_strategy_ = Teuchos::getIntegralValue<Inpar::BeamInteraction::RepartitionStrategy>(
      params_list, "REPARTITIONSTRATEGY");

  search_strategy_ = Teuchos::getIntegralValue<Inpar::BeamInteraction::SearchStrategy>(
      params_list, "SEARCH_STRATEGY");

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamInteractionParams::setup()
{
  check_init();

  // empty for now

  issetup_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::Data::CrosslinkerData::CrosslinkerData()
    : id_(-1), pos_(Core::LinAlg::Initialization::zero), numbond_(0)
{
  bspots_ = std::vector<std::pair<int, int>>(2, std::pair<int, int>(-1, -1));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::CrosslinkerData::pack(Core::Communication::PackBuffer& data) const
{
  // pack id
  add_to_pack(data, id_);
  // pack position
  add_to_pack(data, pos_);
  // pack bspot status
  add_to_pack(data, bspots_);
  // pack number of bonds
  add_to_pack(data, numbond_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::CrosslinkerData::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // id
  extract_from_pack(buffer, id_);
  // position
  extract_from_pack(buffer, pos_);
  // bspot status
  extract_from_pack(buffer, bspots_);
  // number of bonds
  extract_from_pack(buffer, numbond_);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::Data::BeamData::BeamData() : id_(-1)
{
  bspotpos_.clear();
  bspottriad_.clear();
  bspotstatus_.clear();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::BeamData::pack(Core::Communication::PackBuffer& data) const
{
  // pack id
  add_to_pack(data, id_);
  // pack bspotpos
  add_to_pack(data, bspotpos_);
  // pack bspottriad
  add_to_pack(data, bspottriad_);
  // pack bspotstatus_
  add_to_pack(data, bspotstatus_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::BeamData::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // id
  extract_from_pack(buffer, id_);
  // bspotpos
  extract_from_pack(buffer, bspotpos_);
  // bspottriad
  extract_from_pack(buffer, bspottriad_);
  // bspotstatus
  extract_from_pack(buffer, bspotstatus_);



  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::Data::BindEventData::BindEventData()
    : clgid_(-1), elegid_(-1), bspotlocn_(-1), requestproc_(-1), permission_(-1)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::Data::BindEventData::init(
    int clgid, int elegid, int bspotlocn, int requestproc, int permission)
{
  clgid_ = clgid;
  elegid_ = elegid;
  bspotlocn_ = bspotlocn;
  requestproc_ = requestproc;
  permission_ = permission;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::BindEventData::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, clgid_);

  add_to_pack(data, elegid_);

  add_to_pack(data, bspotlocn_);

  add_to_pack(data, requestproc_);

  add_to_pack(data, permission_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::BindEventData::unpack(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, clgid_);

  extract_from_pack(buffer, elegid_);

  extract_from_pack(buffer, bspotlocn_);

  extract_from_pack(buffer, requestproc_);

  extract_from_pack(buffer, permission_);



  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::Data::UnBindEventData::UnBindEventData()
    : clgid_(-1),
      eletoupdate_(std::make_pair(-1, -1)),
      linkertype_(Inpar::BeamInteraction::linkertype_arbitrary)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::UnBindEventData::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, clgid_);

  add_to_pack(data, eletoupdate_);

  add_to_pack(data, linkertype_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::Data::UnBindEventData::unpack(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, clgid_);

  extract_from_pack(buffer, eletoupdate_);

  extract_from_pack(buffer, linkertype_);



  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BeamInteraction::Data::BspotLinkerData::BspotLinkerData()
    : elegid_1_(-1),
      elegid_2_(-1),
      locbspot_1_(-1),
      locbspot_2_(-1),
      type_(-1),
      mat_id_(-1),
      number_of_bonds_1_(-1),
      number_of_bonds_2_(-1)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::Data::BspotLinkerData::same_as(BspotLinkerData bspotlinker)
{
  if (bspotlinker.get_ele_gid1() == elegid_1_ and bspotlinker.get_ele_gid2() == elegid_2_ and
      bspotlinker.get_loc_bspot_id1() == locbspot_1_ and
      bspotlinker.get_loc_bspot_id2() == locbspot_2_ and bspotlinker.get_mat_id() == mat_id_ and
      bspotlinker.get_type() == type_)
    return true;

  return false;
}

FOUR_C_NAMESPACE_CLOSE
