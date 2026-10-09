// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_segments.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::is_identical_segment(const ArteryCouplingPairVector& coupled_ele_pairs,
    const int artery_ele_gid, const double eta_start, const double eta_end,
    int& coupled_ele_pair_id)
{
  for (unsigned i = 0; i < coupled_ele_pairs.size(); i++)
  {
    // first check if the artery element GID is identical
    if (artery_ele_gid == coupled_ele_pairs[i]->artery_ele_gid())
    {
      // check if the integration segment is the same
      if (std::abs(eta_start - coupled_ele_pairs[i]->eta_start()) < 1.0e-9 &&
          std::abs(eta_end - coupled_ele_pairs[i]->eta_end()) < 1.0e-9)
      {
        if constexpr (projection_output) std::cout << "found duplicate integration segment" << '\n';
        coupled_ele_pair_id = static_cast<int>(i);
        return true;
      }
    }
  }

  coupled_ele_pair_id = -1;
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::is_duplicate_segment(const ArteryCouplingPairVector& coupled_ele_pairs,
    const PorofluidElastScatraArteryCouplingPairBase& possible_duplicate)
{
  const double eta_a = possible_duplicate.eta_start();
  const double eta_b = possible_duplicate.eta_end();
  const int ele1_gid = possible_duplicate.artery_ele_gid();
  int ele_pair_id = -1;

  return is_identical_segment(coupled_ele_pairs, ele1_gid, eta_a, eta_b, ele_pair_id);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::fill_gid_to_segment_boundaries(const ArterySegmentContext& context,
    const ArteryCouplingPairVector& coupled_ele_pairs,
    std::map<int, std::vector<double>>& gid_to_segment_boundaries)
{
  // fill the GID-to-segment-boundaries map
  for (const auto& coupled_ele_pair : coupled_ele_pairs)
  {
    const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
    const int homogenized_ele_gid = coupled_ele_pair->homogenized_ele_gid();

    const Core::Elements::Element* homogenized_ele =
        context.homogenized_dis->g_element(homogenized_ele_gid);

    const double eta_start = coupled_ele_pair->eta_start();
    const double eta_end = coupled_ele_pair->eta_end();

    if (homogenized_ele->owner() == context.my_mpi_rank)
    {
      gid_to_segment_boundaries[artery_ele_gid].push_back(eta_start);
      gid_to_segment_boundaries[artery_ele_gid].push_back(eta_end);
    }
    else
    {
      FOUR_C_THROW(
          "Something went wrong here, pair in coupling ele pairs where continuous-discretization "
          "(homogenized 2D/3D) element is not owned by this proc.");
    }
  }

  // communicate it to all procs
  std::vector<int> all_procs(Core::Communication::num_mpi_ranks(context.artery_dis->get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(context.artery_dis->get_comm()); ++i)
    all_procs[i] = i;
  Core::LinAlg::gather<double>(gid_to_segment_boundaries, gid_to_segment_boundaries,
      static_cast<int>(all_procs.size()), all_procs.data(), context.artery_dis->get_comm());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::create_gid_to_segment_boundaries(const ArterySegmentContext& context,
    const ArteryCouplingPairVector& coupled_ele_pairs,
    std::map<int, std::vector<double>>& gid_to_segment_boundaries)
{
  // fill the GID-to-segment-boundaries map
  fill_gid_to_segment_boundaries(context, coupled_ele_pairs, gid_to_segment_boundaries);

  // sort and take care of special cases
  for (int i = 0; i < context.artery_dis->element_col_map()->num_my_elements(); ++i)
  {
    if (const int artery_ele_gid = context.artery_dis->element_col_map()->gid(i);
        gid_to_segment_boundaries[artery_ele_gid].size() > 0)  // check if element projects
    {
      std::ranges::sort(gid_to_segment_boundaries[artery_ele_gid].begin(),
          gid_to_segment_boundaries[artery_ele_gid].end());
      const int end = static_cast<int>(gid_to_segment_boundaries[artery_ele_gid].size());

      // the end of the element lies outside the domain
      if (const double value_at_end = gid_to_segment_boundaries[artery_ele_gid][end - 1];
          std::abs(value_at_end - 1.0) > 1.0e-9)
      {
        gid_to_segment_boundaries[artery_ele_gid].push_back(value_at_end);
        gid_to_segment_boundaries[artery_ele_gid].push_back(1.0);
      }

      // the beginning of the element lies outside the domain
      if (const double value_at_beginning = gid_to_segment_boundaries[artery_ele_gid][0];
          std::abs(value_at_beginning + 1.0) > 1.0e-9)
      {
        gid_to_segment_boundaries[artery_ele_gid].insert(
            gid_to_segment_boundaries[artery_ele_gid].begin(), value_at_beginning);
        gid_to_segment_boundaries[artery_ele_gid].insert(
            gid_to_segment_boundaries[artery_ele_gid].begin(), -1.0);
      }
    }
    // this element does not project
    else
    {
      gid_to_segment_boundaries[artery_ele_gid].push_back(-1.0);
      gid_to_segment_boundaries[artery_ele_gid].push_back(1.0);
    }
  }

  // safety checks
  for (int i = 0; i < context.artery_dis->element_col_map()->num_my_elements(); ++i)
  {
    // 1) check if the artery element has more than MAXNUMSEGPERARTELE segments
    const int artery_ele_gid = context.artery_dis->element_col_map()->gid(i);
    if (static_cast<int>(gid_to_segment_boundaries[artery_ele_gid].size()) >
        2 * context.max_num_segments_per_artery_element)
    {
      FOUR_C_THROW(
          "Artery element {} has {} segments, which is more than the maximum allowed number of "
          "{} "
          "segments per artery element, increase MAXNUMSEGPERARTELE",
          artery_ele_gid, static_cast<int>(gid_to_segment_boundaries[artery_ele_gid].size() / 2),
          context.max_num_segments_per_artery_element);
    }
    // 2) check if the segment has been overlooked
    for (int iseg = 0;
        iseg < static_cast<int>(gid_to_segment_boundaries[artery_ele_gid].size() / 2) - 1; iseg++)
    {
      if (std::abs(gid_to_segment_boundaries[artery_ele_gid][2 * iseg + 1] -
                   gid_to_segment_boundaries[artery_ele_gid][2 * iseg + 2]) > 1.0e-9)
      {
        std::cout << "Problem with segments of artery-element " << artery_ele_gid << ":" << '\n';
        for (int jseg = 0;
            std::cmp_less(jseg, gid_to_segment_boundaries[artery_ele_gid].size() / 2); jseg++)
        {
          std::cout << "[" << gid_to_segment_boundaries[artery_ele_gid][2 * jseg] << ", "
                    << gid_to_segment_boundaries[artery_ele_gid][2 * jseg + 1] << "]" << '\n';
        }
        FOUR_C_THROW(
            "artery element {} has probably not found all possible segments", artery_ele_gid);
      }
    }
  }
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::optional<PoroPressureBased::ArterySegmentLengths>
PoroPressureBased::fill_unaffected_artery_length(const ArterySegmentContext& context,
    ArteryCouplingPairVector& coupled_ele_pairs,
    std::map<int, std::vector<double>>& gid_to_segment_boundaries,
    std::map<int, std::vector<double>>& gid_to_segment_length)
{
  if (context.pure_porofluid_problem)
  {
    for (int i = 0; i < context.artery_dis->element_col_map()->num_my_elements(); ++i)
    {
      const int artery_ele_gid = context.artery_dis->element_col_map()->gid(i);
      Core::Elements::Element* artery_element = context.artery_dis->g_element(artery_ele_gid);

      // TODO: this will not work for higher order artery elements
      const double initial_length = get_max_nodal_distance(artery_element, *context.artery_dis);
      const int num_segments =
          static_cast<int>(gid_to_segment_boundaries[artery_ele_gid].size() / 2);
      gid_to_segment_length[artery_ele_gid].resize(num_segments);
      for (int iseg = 0; iseg < num_segments; iseg++)
      {
        const double etaA = gid_to_segment_boundaries[artery_ele_gid][2 * iseg];
        const double etaB = gid_to_segment_boundaries[artery_ele_gid][2 * iseg + 1];
        gid_to_segment_length[artery_ele_gid][iseg] = initial_length * (etaB - etaA) / 2.0;

        // return also id -> index in coupled_ele_pairs of this segment
        // and set iseg as the segment id of the coupling pairs
        if (int id = -1; is_identical_segment(coupled_ele_pairs, artery_ele_gid, etaA, etaB, id))
          coupled_ele_pairs[id]->set_segment_id(iseg);
      }
    }

    return std::nullopt;
  }

  // no need to do this for a pure porofluid problem

  // The unaffected length is the length of 1D elements not changed by deformation,
  // basically if these elements protrude.
  // For each element, this length is computed as: ele_length - sum_segments seg_length.
  // If the above quantity is bigger than zero, a 1D element protrudes.

  // initialize the unaffected and current lengths
  ArterySegmentLengths lengths{
      .unaffected = std::make_unique<Core::LinAlg::FEVector<double>>(
          *context.artery_dis->dof_row_map(1), true),
      .current =
          std::make_unique<Core::LinAlg::FEVector<double>>(*context.artery_dis->dof_row_map(1)),
  };
  Core::LinAlg::FEVector<double>& unaffected_artery_segment_lengths = *lengths.unaffected;
  Core::LinAlg::FEVector<double>& current_artery_segment_lengths = *lengths.current;

  // set segment ID on coupling pairs and fill the unaffected artery length
  for (int iele = 0; iele < context.artery_dis->element_col_map()->num_my_elements(); ++iele)
  {
    const int artery_ele_gid = context.artery_dis->element_col_map()->gid(iele);
    Core::Elements::Element* current_element = context.artery_dis->g_element(artery_ele_gid);

    // TODO: this will not work for higher order artery elements
    const double initial_length = get_max_nodal_distance(current_element, *context.artery_dis);

    std::vector<double> segment_boundaries = gid_to_segment_boundaries[artery_ele_gid];
    for (unsigned int iseg = 0; iseg < segment_boundaries.size() / 2; iseg++)
    {
      // get EtaA and etaB and calculate initial length
      const double etaA = segment_boundaries[iseg * 2];
      const double etaB = segment_boundaries[iseg * 2 + 1];
      const double segment_length = initial_length * (etaB - etaA) / 2.0;

      // since we use an FE vector
      if (current_element->owner() == context.my_mpi_rank)
      {
        // build the location array
        std::vector<int> segment_length_dofs = context.artery_dis->dof(1, current_element);
        unaffected_artery_segment_lengths.sum_into_global_values(
            1, &segment_length_dofs[iseg], &segment_length);
      }

      // return also id -> index in coupled_ele_pairs of this segment
      // and set iseg as the segment id of the coupling pairs
      if (int id = -1; is_identical_segment(coupled_ele_pairs, artery_ele_gid, etaA, etaB, id))
        coupled_ele_pairs[id]->set_segment_id(static_cast<int>(iseg));
    }
  }

  unaffected_artery_segment_lengths.complete();

  // subtract the segment lengths only if we evaluate in current configuration
  if (!context.evaluate_in_ref_config)
  {
    for (const auto& coupled_ele_pair : coupled_ele_pairs)
    {
      // get the initial lengths
      double initial_segment_length =
          coupled_ele_pair->apply_mesh_movement(true, context.homogenized_dis);
      initial_segment_length *= -1.0;

      const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
      const Core::Elements::Element* current_element =
          context.artery_dis->g_element(artery_ele_gid);

      std::vector<int> segment_length_dofs = context.artery_dis->dof(1, current_element);
      const int segment_id = coupled_ele_pair->get_segment_id();

      unaffected_artery_segment_lengths.sum_into_global_values(
          1, &segment_length_dofs[segment_id], &(initial_segment_length));
    }
    unaffected_artery_segment_lengths.complete();
  }
  // the current length is simply the unaffected length
  else
  {
    current_artery_segment_lengths.update(1.0, unaffected_artery_segment_lengths, 0.0);
  }

  return lengths;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::apply_mesh_movement_for_pairs(const ArterySegmentContext& context,
    ArteryCouplingPairVector& coupled_ele_pairs,
    const Core::LinAlg::FEVector<double>& unaffected_artery_segment_lengths,
    Core::LinAlg::FEVector<double>& current_artery_segment_lengths)
{
  // no need to do this
  if (context.pure_porofluid_problem) return;

  // only if we evaluate in current configuration
  if (!context.evaluate_in_ref_config)
  {
    // safety
    if (!context.homogenized_dis->has_state(1, "dispnp"))
      FOUR_C_THROW("cannot get displacement state");

    // update with unaffected length
    current_artery_segment_lengths.update(1.0, unaffected_artery_segment_lengths, 0.0);

    // apply movement on pairs and fill gid-to-segment-length and current_seg_lengths_artery_
    for (const auto& coupled_ele_pair : coupled_ele_pairs)
    {
      const double new_segment_length =
          coupled_ele_pair->apply_mesh_movement(false, context.homogenized_dis);
      const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
      const int segment_id = coupled_ele_pair->get_segment_id();

      const Core::Elements::Element* artery_element = context.artery_dis->g_element(artery_ele_gid);
      // build the location array
      std::vector<int> segment_length_dofs = context.artery_dis->dof(1, artery_element);

      current_artery_segment_lengths.sum_into_global_values(
          1, &segment_length_dofs[segment_id], &(new_segment_length));
    }

    current_artery_segment_lengths.complete();
  }

  // set state on artery discretization
  context.artery_dis->set_state(
      1, "curr_seg_lengths", Core::LinAlg::Vector<double>(current_artery_segment_lengths));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<double> PoroPressureBased::get_ele_segment_lengths(const ArterySegmentContext& context,
    std::map<int, std::vector<double>>& gid_to_segment_length, const int artery_ele_gid)
{
  if (context.pure_porofluid_problem) return gid_to_segment_length[artery_ele_gid];

  // safety checks
  if (!context.artery_dis->has_state(1, "curr_seg_lengths"))
    FOUR_C_THROW("cannot get state curr_seg_lengths");

  // build the location array
  const Core::Elements::Element* current_element = context.artery_dis->g_element(artery_ele_gid);
  const std::vector<int> segment_length_dof = context.artery_dis->dof(1, current_element);

  const std::shared_ptr<const Core::LinAlg::Vector<double>> current_segment_lengths =
      context.artery_dis->get_state(1, "curr_seg_lengths");

  std::vector<double> segment_lengths =
      Core::FE::extract_values(*current_segment_lengths, segment_length_dof);

  return segment_lengths;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::filter_coupling_pairs_and_resolve_cross_proc_duplicates(
    const ArterySegmentContext& context, ArteryCouplingPairVector& coupled_ele_pairs)
{
  // delete the inactive and duplicated pairs
  ArteryCouplingPairVector active_coupled_ele_pairs;
  for (auto& coupled_ele_pair : coupled_ele_pairs)
  {
    const int homogenized_ele_gid = coupled_ele_pair->homogenized_ele_gid();
    const Core::Elements::Element* homogenized_ele =
        context.homogenized_dis->g_element(homogenized_ele_gid);

    if (coupled_ele_pair->is_active() &&
        !is_duplicate_segment(active_coupled_ele_pairs, *coupled_ele_pair) &&
        homogenized_ele->owner() == context.my_mpi_rank)
      active_coupled_ele_pairs.push_back(coupled_ele_pair);
  }

  // the following case takes care of the special case where the 1D element lies exactly in
  // between two 2D/3D-elements which are owned by different processors

  // fill the GID-to-segment-boundaries map
  std::map<int, std::vector<double>> gid_to_segment_boundaries;
  fill_gid_to_segment_boundaries(context, active_coupled_ele_pairs, gid_to_segment_boundaries);

  // dummy map to collect duplicates in form [ele2gid, eta_a, eta_b, ... ];
  std::map<int, std::vector<double>> duplicates;

  // loop over all artery elements
  for (int i = 0; i < context.artery_dis->element_col_map()->num_my_elements(); ++i)
  {
    if (const int artery_ele_gid = context.artery_dis->element_col_map()->gid(i);
        gid_to_segment_boundaries[artery_ele_gid].size() > 0)  // check if element projects
    {
      // compare all segment with each other if it might be identical
      for (int iseg = 0; std::cmp_less(iseg, gid_to_segment_boundaries[artery_ele_gid].size() / 2);
          iseg++)
      {
        const double eta_a = gid_to_segment_boundaries[artery_ele_gid][2 * iseg];
        const double eta_b = gid_to_segment_boundaries[artery_ele_gid][2 * iseg + 1];
        for (int jseg = iseg + 1;
            std::cmp_less(jseg, gid_to_segment_boundaries[artery_ele_gid].size() / 2); jseg++)
        {
          const double eta_a_jseg = gid_to_segment_boundaries[artery_ele_gid][2 * jseg];
          const double eta_b_jseg = gid_to_segment_boundaries[artery_ele_gid][2 * jseg + 1];
          // identical segment found
          if (std::abs(eta_a - eta_a_jseg) < 1.0e-9 && std::abs(eta_b - eta_b_jseg) < 1.0e-9)
          {
            // we need this to get the GID of the second element
            int id = -1;
            if (is_identical_segment(active_coupled_ele_pairs, artery_ele_gid, eta_a, eta_b, id))
            {
              const int ele2_gid = active_coupled_ele_pairs[id]->homogenized_ele_gid();
              duplicates[artery_ele_gid].push_back((ele2_gid));
              duplicates[artery_ele_gid].push_back(eta_a);
              duplicates[artery_ele_gid].push_back(eta_b);
            }
          }
        }
      }
    }
  }

  // communicate the map to all procs
  std::vector<int> mpi_ranks(Core::Communication::num_mpi_ranks(context.artery_dis->get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(context.artery_dis->get_comm()); ++i)
    mpi_ranks[i] = i;
  Core::LinAlg::gather<double>(duplicates, duplicates, static_cast<int>(mpi_ranks.size()),
      mpi_ranks.data(), context.artery_dis->get_comm());

  // remove duplicate (the one where the 2D/3D element has the large ID)
  for (auto& duplicate : duplicates)
  {
    const int artery_ele_gid = duplicate.first;
    std::vector<double> current_duplicates = duplicate.second;
    // should always be a multiple of six because we should always find exactly two/four, etc.
    // duplicates
    if (current_duplicates.size() % 6 != 0)
      FOUR_C_THROW(
          "duplicate vector has size {}, should be multiple of six", current_duplicates.size());
    // compare the possible duplicates
    for (int idupl = 0; std::cmp_less(idupl, (current_duplicates.size() / 3)); idupl++)
    {
      const double eta_a = current_duplicates[3 * idupl + 1];
      const double eta_b = current_duplicates[3 * idupl + 2];
      for (int jdupl = idupl + 1; std::cmp_less(jdupl, (current_duplicates.size() / 3)); jdupl++)
      {
        const double eta_a_jdupl = current_duplicates[3 * jdupl + 1];
        const double eta_b_jdupl = current_duplicates[3 * jdupl + 2];
        // duplicate found
        if (std::abs(eta_a - eta_a_jdupl) < 1.0e-9 && std::abs(eta_b - eta_b_jdupl) < 1.0e-9)
        {
          const int ele_i = static_cast<int>(current_duplicates[3 * idupl]);
          const int ele_j = static_cast<int>(current_duplicates[3 * jdupl]);
          const int ele_to_be_erased = std::max(ele_i, ele_j);
          int id = -1;
          // delete the duplicate with the larger ele2_gid
          if (is_identical_segment(active_coupled_ele_pairs, artery_ele_gid, eta_a, eta_b, id))
          {
            if (active_coupled_ele_pairs[id]->homogenized_ele_gid() == ele_to_be_erased)
            {
              active_coupled_ele_pairs.erase(active_coupled_ele_pairs.begin() + id);
            }
          }
        }
      }
    }
  }

  // overwrite the coupling pairs
  coupled_ele_pairs = active_coupled_ele_pairs;

  // output
  int num_active_pairs = static_cast<int>(coupled_ele_pairs.size());
  int total_num_active_pairs =
      Core::Communication::sum_all(num_active_pairs, context.artery_dis->get_comm());
  if (context.my_mpi_rank == 0)
  {
    std::cout << "Only " << total_num_active_pairs
              << " Artery-to-PorofluidPressurebasedScatra coupling pairs (segments) are active"
              << '\n';
  }
}

FOUR_C_NAMESPACE_CLOSE
