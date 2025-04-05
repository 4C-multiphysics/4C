// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_pair.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_pair.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_params.hpp"
#include "4C_beaminteraction_beam_to_sphere_contact_pair.hpp"
#include "4C_beaminteraction_conditions.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_geometry_pair.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_shell_kl_nurbs.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::BeamContactPair::BeamContactPair()
    : isinit_(false),
      issetup_(false),
      geometry_pair_(nullptr),
      params_(nullptr),
      element1_(nullptr),
      element2_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactPair::init(
    const std::shared_ptr<BeamInteraction::BeamContactParams> params_ptr,
    std::vector<Core::Elements::Element const*> elements)
{
  issetup_ = false;

  params_ = params_ptr;

  element1_ = elements[0];
  element2_ = elements[1];

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactPair::setup()
{
  check_init();

  // the flag issetup_ will be set in the derived method!
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>
BeamInteraction::BeamContactPair::create(
    std::vector<Core::Elements::Element const*> const& ele_ptrs,
    BeamInteraction::BeamInteractionConditions& beam_interaction_conditions_ptr)
{
  std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> created_pairs;

  // Check the type of the second element.
  const bool other_is_beam =
      dynamic_cast<Discret::Elements::Beam3Base const*>(ele_ptrs[1]) != nullptr;
  const bool other_is_solid =
      dynamic_cast<Discret::Elements::Solid const*>(ele_ptrs[1]) != nullptr ||
      dynamic_cast<Discret::Elements::KirchhoffLoveShellNurbs const*>(ele_ptrs[1]) != nullptr;
  const bool other_is_sphere =
      ele_ptrs[1]->element_type() == Discret::Elements::RigidsphereType::instance();

  if (other_is_beam or other_is_solid)
  {
    // Beam-to-beam and beam-to-solid pairs are exclusively created by conditions.
    beam_interaction_conditions_ptr.create_contact_pairs(created_pairs, ele_ptrs);
  }
  else if (other_is_sphere)
  {
    // note: numnodes is to be interpreted as number of nodes used for centerline interpolation.
    // numnodalvalues = 1: only positions as primary nodal DoFs ==> Lagrange interpolation
    // numnodalvalues = 2: positions AND tangents ==> Hermite interpolation

    const Discret::Elements::Beam3Base* beamele1 =
        dynamic_cast<const Discret::Elements::Beam3Base*>(ele_ptrs[0]);

    const unsigned int numnodes_centerline = beamele1->num_centerline_nodes();
    const unsigned int numnodalvalues = beamele1->hermite_centerline_interpolation() ? 2 : 1;

    switch (numnodalvalues)
    {
      case 1:
      {
        switch (numnodes_centerline)
        {
          case 2:
          {
            created_pairs.push_back(
                std::make_shared<BeamInteraction::BeamToSphereContactPair<2, 1>>());
            break;
          }
          case 3:
          {
            created_pairs.push_back(
                std::make_shared<BeamInteraction::BeamToSphereContactPair<3, 1>>());
            break;
          }
          case 4:
          {
            created_pairs.push_back(
                std::make_shared<BeamInteraction::BeamToSphereContactPair<4, 1>>());
            break;
          }
          case 5:
          {
            created_pairs.push_back(
                std::make_shared<BeamInteraction::BeamToSphereContactPair<5, 1>>());
            break;
          }
          default:
          {
            FOUR_C_THROW(
                "{} and {} is no valid template parameter combination for the "
                "number of nodes and number of types of nodal DoFs used for centerline "
                "interpolation!",
                numnodes_centerline, numnodalvalues);
            break;
          }
        }
        break;
      }
      case 2:
      {
        switch (numnodes_centerline)
        {
          case 2:
          {
            created_pairs.push_back(
                std::make_shared<BeamInteraction::BeamToSphereContactPair<2, 2>>());
            break;
          }
          default:
          {
            FOUR_C_THROW(
                "{} and {} is no valid template parameter combination for the "
                "number of nodes and number of types of nodal DoFs used for centerline "
                "interpolation!",
                numnodes_centerline, numnodalvalues);
            break;
          }
        }
        break;
      }
      default:
      {
        FOUR_C_THROW(
            "{} and {} is no valid template parameter combination for the "
            "number of nodes and number of types of nodal DoFs used for centerline "
            "interpolation!",
            numnodes_centerline, numnodalvalues);
        break;
      }
    }
  }
  else
  {
    FOUR_C_THROW("Unknown type of second element in creation of beam contact pair.");
  }
  return created_pairs;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactPair::check_init() const
{
  if (not is_init()) FOUR_C_THROW("Call init() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::BeamContactPair::check_init_setup() const
{
  if (not is_init() or not is_setup()) FOUR_C_THROW("Call init() and setup() first!");
}

FOUR_C_NAMESPACE_CLOSE
