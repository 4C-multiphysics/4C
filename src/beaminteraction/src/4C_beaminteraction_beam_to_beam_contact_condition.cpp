// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_beam_contact_condition.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_pair.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_pstream.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToBeamContactCondition::BeamToBeamContactCondition(
    const Core::Conditions::Condition& condition_line_1,
    const Core::Conditions::Condition& condition_line_2)
    : BeamInteractionConditionBase(condition_line_1),
      condition_other_(&condition_line_2),
      condition_contact_pairs_(),
      other_line_ids_()
{
  old_cp_normals_ = std::make_shared<std::map<ElementIDKey, Core::LinAlg::Matrix<3, 1, double>>>();
}

/**
 *
 */
void BeamInteraction::BeamToBeamContactCondition::build_id_sets(
    const std::shared_ptr<const Core::FE::Discretization>& discretization)
{
  // Call the parent method to build the line maps.
  BeamInteractionConditionBase::build_id_sets(discretization);

  // Build the other line map.
  std::vector<int> line_ids;
  condition_to_element_ids(*condition_other_, line_ids);
  other_line_ids_ = std::set<int>(line_ids.begin(), line_ids.end());
}

/**
 *
 */
bool BeamInteraction::BeamToBeamContactCondition::ids_in_condition(
    const int id_line, const int id_other) const
{
  if (id_is_in_condition(line_ids_, id_line) and id_is_in_condition(other_line_ids_, id_other))
    return true;
  if (id_is_in_condition(line_ids_, id_other) and id_is_in_condition(other_line_ids_, id_line))
    return true;
  return false;
}

/**
 *
 */
void BeamInteraction::BeamToBeamContactCondition::clear()
{
  BeamInteractionConditionBase::clear();
  condition_contact_pairs_.clear();
}

/**
 *
 */
void BeamInteraction::BeamToBeamContactCondition::setup(
    const std::shared_ptr<const Core::FE::Discretization>& discret)
{
  // hard coded number of pair size and length of normal
  const int pair_size = 2;
  const int normal_size = 3;

  // determine the keys this local proc will need to evaluate the condition
  std::set<ElementIDKey> local_needed_keys;
  for (auto contact_pair : condition_contact_pairs_)
  {
    local_needed_keys.insert({contact_pair->element1()->id(), contact_pair->element2()->id()});
  }

  // remove every key which is already stored locally on this proc
  std::set<ElementIDKey> local_missing_keys = local_needed_keys;
  for (auto const& [key, normal] : *old_cp_normals_)
  {
    local_missing_keys.erase(key);
  }

  // create flat vector for easier communication
  std::vector<int> local_missing_keys_flat;
  local_missing_keys_flat.reserve(pair_size * local_missing_keys.size());
  for (const auto& k : local_missing_keys)
  {
    local_missing_keys_flat.push_back(k.ele1);
    local_missing_keys_flat.push_back(k.ele2);
  }

  // find out who is missing which keys by distributing the concatenation of all ranks
  std::vector<int> global_missing_keys_flat =
      Core::Communication::all_reduce(local_missing_keys_flat, discret->get_comm());

  // build a set of all globally requested keys
  std::set<ElementIDKey> global_missing_keys;
  for (auto it = global_missing_keys_flat.begin(); it != global_missing_keys_flat.end();)
  {
    int e1 = *it++;
    int e2 = *it++;
    global_missing_keys.insert(ElementIDKey{e1, e2});
  }

  // create buffers for sending with same size on each proc
  std::vector<int> global_found_keys;
  std::vector<double> global_found_vals;

  // keep track of local keys which will be distributed to other procs
  std::vector<ElementIDKey> local_keys_to_be_removed;

  // initialize according to element id and normal size
  global_found_keys.reserve(pair_size * global_missing_keys.size());
  global_found_vals.reserve(normal_size * global_missing_keys.size());

  // Search my local normals for globally missing keys
  // and values to global missing array
  for (auto const& [key, normal] : *old_cp_normals_)
  {
    if (global_missing_keys.find(key) != global_missing_keys.end())
    {
      global_found_keys.push_back(key.ele1);
      global_found_keys.push_back(key.ele2);
      global_found_vals.push_back(normal(0));
      global_found_vals.push_back(normal(1));
      global_found_vals.push_back(normal(2));

      if (!local_needed_keys.count(key))
      {
        // eligible for deletion after communication
        local_keys_to_be_removed.push_back(key);
      }
    }
  }

  // Broadcast the concatenated responses from all ranks
  std::vector<int> received_element_ids =
      Core::Communication::all_reduce(global_found_keys, discret->get_comm());
  std::vector<double> received_normal_values =
      Core::Communication::all_reduce(global_found_vals, discret->get_comm());

  std::size_t global_array_size = global_missing_keys.size();

  FOUR_C_ASSERT(received_element_ids.size() == pair_size * global_array_size,
      "received_element_ids must be 2*global_missing_keys.size()");
  FOUR_C_ASSERT(received_normal_values.size() == normal_size * global_array_size,
      "received_normal_values must be 3*global_missing_keys.size()");


  // loop through received global arrays
  for (std::size_t t = 0, ki = 0, vi = 0; t < global_array_size;
      ++t, ki += pair_size, vi += normal_size)
  {
    // create new key based on the received element ids
    ElementIDKey key{received_element_ids[ki], received_element_ids[ki + 1]};

    // Store the received key/value pair
    if (local_missing_keys.erase(key))
    {
      // create corresponding normal matrix
      Core::LinAlg::Matrix<normal_size, 1, double> n(Core::LinAlg::Initialization::zero);
      for (int j = 0; j < normal_size; ++j)
      {
        n(j) = received_normal_values[vi + j];
      }

      old_cp_normals_->emplace(key, n);
    }
  }

  // Remove every local key/normals which were distributed to other procs
  for (auto const& key : local_keys_to_be_removed) old_cp_normals_->erase(key);

  // Print warning on debug if some normals are not found
  if (!local_missing_keys.empty())
  {
    Core::IO::cout(Core::IO::debug)
        << "[Rank " << Core::Communication::my_mpi_rank(discret->get_comm())
        << "] is still missing " << local_missing_keys.size()
        << " contact pair normals after exchange in BeamToBeamContactCondition. Example key: "
        << Core::IO::endl;
    auto ex = *local_missing_keys.begin();
    Core::IO::cout(Core::IO::debug) << "(" << ex.ele1 << "," << ex.ele2 << ")" << Core::IO::endl;
  }

  Core::Communication::barrier(discret->get_comm());
}

/**
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::BeamToBeamContactCondition::create_contact_pair(
    const std::vector<Core::Elements::Element const*>& ele_ptrs)
{
  // Check if the given elements are in this condition.
  if (!ids_in_condition(ele_ptrs[0]->id(), ele_ptrs[1]->id())) return nullptr;

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
          return std::make_shared<BeamInteraction::BeamToBeamContactPair<2, 1>>(old_cp_normals_);
        }
        case 3:
        {
          return std::make_shared<BeamInteraction::BeamToBeamContactPair<3, 1>>(old_cp_normals_);
        }
        case 4:
        {
          return std::make_shared<BeamInteraction::BeamToBeamContactPair<4, 1>>(old_cp_normals_);
        }
        case 5:
        {
          return std::make_shared<BeamInteraction::BeamToBeamContactPair<5, 1>>(old_cp_normals_);
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
          return std::make_shared<BeamInteraction::BeamToBeamContactPair<2, 2>>(old_cp_normals_);
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

  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
