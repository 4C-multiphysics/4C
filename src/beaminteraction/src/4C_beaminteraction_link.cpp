// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_link.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

BeamInteraction::BeamLinkType BeamInteraction::BeamLinkType::instance_;


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamLink::BeamLink()
    : ParObject(),
      isinit_(false),
      issetup_(false),
      id_(-1),
      linkertype_(Inpar::BeamInteraction::linkertype_arbitrary),
      timelinkwasset_(-1.0),
      reflength_(-1.0)
{
  bspot_ids_.clear();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
BeamInteraction::BeamLink::BeamLink(const BeamInteraction::BeamLink& old)
    : ParObject(old),
      isinit_(old.isinit_),
      issetup_(old.issetup_),
      id_(old.id_),
      bspot_ids_(old.bspot_ids_),
      bspotpos1_(old.bspotpos1_),
      bspotpos2_(old.bspotpos2_),
      linkertype_(old.linkertype_),
      timelinkwasset_(old.timelinkwasset_),
      reflength_(old.reflength_)
{
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLink::init(const int id, const std::vector<std::pair<int, int>>& eleids,
    const std::vector<Core::LinAlg::Matrix<3, 1>>& initpos,
    const std::vector<Core::LinAlg::Matrix<3, 3>>& inittriad,
    Inpar::BeamInteraction::CrosslinkerType linkertype, double timelinkwasset)
{
  issetup_ = false;

  id_ = id;
  bspot_ids_ = eleids;

  bspotpos1_ = initpos[0];
  bspotpos2_ = initpos[1];

  linkertype_ = linkertype;

  timelinkwasset_ = timelinkwasset;

  reflength_ = 0.0;
  for (unsigned int i = 0; i < 3; ++i)
    reflength_ += (initpos[1](i) - initpos[0](i)) * (initpos[1](i) - initpos[0](i));
  reflength_ = sqrt(reflength_);

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLink::setup(const int matnum)
{
  check_init();

  // the flag issetup_ will be set in the derived method!
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLink::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // isinit_
  add_to_pack(data, isinit_);
  // issetup
  add_to_pack(data, issetup_);
  // add id
  add_to_pack(data, id_);

  // add eleids_
  add_to_pack(data, bspot_ids_);
  // bspotpos1_
  add_to_pack(data, bspotpos1_);
  // bspotpos2_
  add_to_pack(data, bspotpos2_);
  // linkertype
  add_to_pack(data, linkertype_);
  // timelinkwasset
  add_to_pack(data, timelinkwasset_);
  // reflength
  add_to_pack(data, reflength_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::BeamLink::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // isinit_
  extract_from_pack(buffer, isinit_);
  // issetup
  extract_from_pack(buffer, issetup_);
  // id_
  extract_from_pack(buffer, id_);

  // eleids_
  extract_from_pack(buffer, bspot_ids_);
  // bspotpos1
  extract_from_pack(buffer, bspotpos1_);
  // bspotpos2
  extract_from_pack(buffer, bspotpos2_);
  // linkertype
  extract_from_pack(buffer, linkertype_);
  // timelinkwasset
  extract_from_pack(buffer, timelinkwasset_);
  // reflength
  extract_from_pack(buffer, reflength_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLink::reset_state(std::vector<Core::LinAlg::Matrix<3, 1>>& bspotpos,
    std::vector<Core::LinAlg::Matrix<3, 3>>& bspottriad)
{
  check_init_setup();

  /* the two positions of the linkage element coincide with the positions of the
   * binding spots on the parent elements */
  bspotpos1_ = bspotpos[0];
  bspotpos2_ = bspotpos[1];
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamLink::print(std::ostream& out) const
{
  check_init();

  out << "\nBeamLinkRigidJointed (ID " << id() << "):";
  out << "\nbspotIds_[0] = ";
  out << "EleGID " << get_ele_gid(0) << " locbspotnum " << get_loc_b_spot_num(0);
  out << "\nbspotIds_[1] = ";
  out << "EleGID " << get_ele_gid(1) << " locbspotnum " << get_loc_b_spot_num(1);
  out << "\n";
  out << "\nbspotpos1_ = ";
  get_bind_spot_pos1().print(out);
  out << "\nbspotpos2_ = ";
  get_bind_spot_pos2().print(out);

  out << "\n";
}

FOUR_C_NAMESPACE_CLOSE
