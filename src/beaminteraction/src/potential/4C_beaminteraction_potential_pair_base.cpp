// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_potential_pair_base.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_beaminteraction_potential_pair_beam_to_beam.hpp"
#include "4C_beaminteraction_potential_pair_beam_to_sphere.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
BeamInteraction::BeamPotentialPair::BeamPotentialPair()
    : isinit_(false),
      issetup_(false),
      beam_potential_parameters_(nullptr),
      element1_(nullptr),
      element2_(nullptr)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialPair::init(
    const BeamInteraction::Potential::BeamPotentialParameters* params_ptr,
    const Core::Elements::Element* element1, const Core::Elements::Element* element2)
{
  issetup_ = false;

  beam_potential_parameters_ = params_ptr;

  element1_ = element1;
  element2_ = element2;


  isinit_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialPair::setup()
{
  check_init();

  // the flag issetup_ will be set in the derived method!
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
std::shared_ptr<BeamInteraction::BeamPotentialPair> BeamInteraction::BeamPotentialPair::create(
    std::vector<Core::Elements::Element const*> const& ele_ptrs,
    BeamInteraction::Potential::BeamPotentialParameters const& beam_potential_params)
{
  // note: numnodes is to be interpreted as number of nodes used for centerline interpolation.
  // numnodalvalues = 1: only positions as primary nodal DoFs ==> Lagrange interpolation
  // numnodalvalues = 2: positions AND tangents ==> Hermite interpolation

  const Discret::Elements::Beam3Base* beamele1 =
      dynamic_cast<const Discret::Elements::Beam3Base*>(ele_ptrs[0]);

  // at the moment, both elements of a beam contact pair must be of same type Todo
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
          if (ele_ptrs[1]->element_type() == Discret::Elements::RigidsphereType::instance())
            return std::make_shared<BeamInteraction::BeamToSpherePotentialPair<2, 1>>();
          else
          {
            if (beam_potential_params.automatic_differentiation)
              return std::make_shared<
                  BeamInteraction::BeamToBeamPotentialPair<2, 1, Sacado::Fad::DFad<double>>>();
            else
              return std::make_shared<BeamInteraction::BeamToBeamPotentialPair<2, 1, double>>();
          }
        }
        case 3:
        {
          if (ele_ptrs[1]->element_type() == Discret::Elements::RigidsphereType::instance())
            return std::make_shared<BeamInteraction::BeamToSpherePotentialPair<3, 1>>();
          else
          {
            if (beam_potential_params.automatic_differentiation)
              return std::make_shared<
                  BeamInteraction::BeamToBeamPotentialPair<3, 1, Sacado::Fad::DFad<double>>>();
            else
              return std::make_shared<BeamInteraction::BeamToBeamPotentialPair<3, 1, double>>();
          }
        }
        case 4:
        {
          if (ele_ptrs[1]->element_type() == Discret::Elements::RigidsphereType::instance())
            return std::make_shared<BeamInteraction::BeamToSpherePotentialPair<4, 1>>();
          else
          {
            if (beam_potential_params.automatic_differentiation)
              return std::make_shared<
                  BeamInteraction::BeamToBeamPotentialPair<4, 1, Sacado::Fad::DFad<double>>>();
            else
              return std::make_shared<BeamInteraction::BeamToBeamPotentialPair<4, 1, double>>();
          }
        }
        case 5:
        {
          if (ele_ptrs[1]->element_type() == Discret::Elements::RigidsphereType::instance())
            return std::make_shared<BeamInteraction::BeamToSpherePotentialPair<5, 1>>();
          else
          {
            if (beam_potential_params.automatic_differentiation)
              return std::make_shared<
                  BeamInteraction::BeamToBeamPotentialPair<5, 1, Sacado::Fad::DFad<double>>>();
            else
              return std::make_shared<BeamInteraction::BeamToBeamPotentialPair<5, 1, double>>();
          }
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
          if (ele_ptrs[1]->element_type() == Discret::Elements::RigidsphereType::instance())
            return std::make_shared<BeamInteraction::BeamToSpherePotentialPair<2, 2>>();
          else
          {
            if (beam_potential_params.automatic_differentiation)
              return std::make_shared<
                  BeamInteraction::BeamToBeamPotentialPair<2, 2, Sacado::Fad::DFad<double>>>();
            else
              return std::make_shared<BeamInteraction::BeamToBeamPotentialPair<2, 2, double>>();
          }
        }
        default:
          FOUR_C_THROW(
              "{} and {} is no valid template parameter combination for the "
              "number of nodes and number of types of nodal DoFs used for centerline "
              "interpolation!",
              numnodes_centerline, numnodalvalues);
          break;
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

  return nullptr;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialPair::check_init() const
{
  if (not is_init()) FOUR_C_THROW("Call init() first!");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void BeamInteraction::BeamPotentialPair::check_init_setup() const
{
  if (not is_init() or not is_setup()) FOUR_C_THROW("Call init() and setup() first!");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
Core::FE::GaussRule1D BeamInteraction::BeamPotentialPair::get_gauss_rule() const
{
  switch (params()->n_gauss_points)
  {
    case 5:
    {
      return Core::FE::GaussRule1D::line_5point;
      break;
    }

    case 10:
    {
      return Core::FE::GaussRule1D::line_10point;
      break;
    }

    case 20:
    {
      return Core::FE::GaussRule1D::line_20point;
      break;
    }

    case 32:
    {
      return Core::FE::GaussRule1D::line_32point;
      break;
    }

    case 50:
    {
      return Core::FE::GaussRule1D::line_50point;
      break;
    }

    default:
      FOUR_C_THROW("{} Gauss points are not supported yet!", params()->n_gauss_points);
  }

  return Core::FE::GaussRule1D::undefined;
}

FOUR_C_NAMESPACE_CLOSE
