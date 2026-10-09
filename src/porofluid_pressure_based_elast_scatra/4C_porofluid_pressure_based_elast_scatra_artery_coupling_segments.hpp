// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_SEGMENTS_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_SEGMENTS_HPP

#include "4C_config.hpp"

#include "4C_linalg_fevector.hpp"

#include <map>
#include <memory>
#include <optional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace PoroPressureBased
{
  class PorofluidElastScatraArteryCouplingPairBase;

  using ArteryCouplingPairVector =
      std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>>;

  /*!
   * @brief State of the coupling segments
   */
  struct ArterySegmentContext
  {
    //! 1D artery discretization (column layout, i.e., fully overlapping); not owned by the context
    Core::FE::Discretization* artery_dis;

    //! 2D/3D homogenized discretization; not owned by the context
    Core::FE::Discretization* homogenized_dis;

    //! this processor's MPI rank
    int my_mpi_rank;

    //! input maximum number of segments allowed per artery element (MAXNUMSEGPERARTELE)
    int max_num_segments_per_artery_element;

    //! pure porofluid problem (no elasticity), i.e., the segment lengths do not change
    bool pure_porofluid_problem;

    //! artery elements are evaluated in the reference configuration
    bool evaluate_in_ref_config;
  };

  /*!
   * @brief Artery segment lengths of the coupling pairs (one entry per artery segment)
   */
  struct ArterySegmentLengths
  {
    //! length of the possible protruding parts of the artery segments (not changed by deformation)
    std::unique_ptr<Core::LinAlg::FEVector<double>> unaffected;

    //! current segment lengths (unaffected length + deformed length of the segments due to mesh
    //! movement)
    std::unique_ptr<Core::LinAlg::FEVector<double>> current;
  };

  /*!
   * @brief Check if segments are identical
   * @param[in] coupled_ele_pairs: coupling pairs (artery (discrete) and porous domain
   * (homogenized))
   * @param[in] artery_ele_gid: GID of the artery element
   * @param[in] eta_start: start of the integration segment in artery element coordinates
   * @param[in] eta_end: end of the integration segment in artery element coordinates
   * @param[out] coupled_ele_pair_id: GID of the artery element with duplicated integration segment;
   * -1 if no identical segment was found
   * @return true if duplicate integration segment exists
   */
  bool is_identical_segment(const ArteryCouplingPairVector& coupled_ele_pairs, int artery_ele_gid,
      double eta_start, double eta_end, int& coupled_ele_pair_id);

  /*!
   * @brief Check for duplicate segments. We have to sort out duplicate segments, these might occur
   * if the artery element lies exactly between two different homogenized elements (2D/3D-elements)
   * @param[in] coupled_ele_pairs: coupling pairs to check
   * @param[in] possible_duplicate: pair to be checked against coupled_ele_pairs
   * @return true if @p possible_duplicate describes the same segment as one of the
   * coupled_ele_pairs
   */
  bool is_duplicate_segment(const ArteryCouplingPairVector& coupled_ele_pairs,
      const PorofluidElastScatraArteryCouplingPairBase& possible_duplicate);

  /*!
   * @brief Fill the GID to segment boundaries map and communicate it to all processors
   * @param[in] context: state of the coupling segments
   * @param[in] coupled_ele_pairs: coupling pairs whose segments are collected
   * @param[in,out] gid_to_segment_boundaries: append [eta_a, eta_b] per artery element GID;
   * entries are appended to the existing content and gathered over all processors
   */
  void fill_gid_to_segment_boundaries(const ArterySegmentContext& context,
      const ArteryCouplingPairVector& coupled_ele_pairs,
      std::map<int, std::vector<double>>& gid_to_segment_boundaries);

  /*!
   * @brief Create the GID to segment boundaries map (fill, sort, close gaps at the element ends,
   * safety check)
   * @param[in] context: state of the coupling segments
   * @param[in] coupled_ele_pairs: coupling pairs whose segment boundaries are collected
   * @param[in,out] gid_to_segment_boundaries: [eta_a, eta_b] per artery element GID
   */
  void create_gid_to_segment_boundaries(const ArterySegmentContext& context,
      const ArteryCouplingPairVector& coupled_ele_pairs,
      std::map<int, std::vector<double>>& gid_to_segment_boundaries);

  /*!
   * @brief Fill the artery segment length not changed by deformation and initialize the current
   * length.
   *
   * @param[in] context: state of the coupling segments
   * @param[in,out] coupled_ele_pairs: coupling pairs; the segment ID is set on each pair
   * @param[in] gid_to_segment_boundaries: segment boundaries per artery element GID (see
   * create_gid_to_segment_boundaries())
   * @param[out] gid_to_segment_length: lengths of the segments belonging to the specific artery
   * element GID
   * @return the segment lengths, or std::nullopt for a pure porofluid problem (not needed there)
   */
  std::optional<ArterySegmentLengths> fill_unaffected_artery_length(
      const ArterySegmentContext& context, ArteryCouplingPairVector& coupled_ele_pairs,
      std::map<int, std::vector<double>>& gid_to_segment_boundaries,
      std::map<int, std::vector<double>>& gid_to_segment_length);

  /*!
   * @brief Apply the mesh (solid phase) movement to the artery elements
   * @param[in] context: state of the coupling segments
   * @param[in,out] coupled_ele_pairs: coupling pairs
   * @param[in] unaffected_artery_segment_lengths: length of the protruding parts of the artery
   * segments
   * @param[in,out] current_artery_segment_lengths: current segment lengths; recomputed and
   * stored as the "curr_seg_lengths" state on the artery discretization
   */
  void apply_mesh_movement_for_pairs(const ArterySegmentContext& context,
      ArteryCouplingPairVector& coupled_ele_pairs,
      const Core::LinAlg::FEVector<double>& unaffected_artery_segment_lengths,
      Core::LinAlg::FEVector<double>& current_artery_segment_lengths);

  /*!
   * @brief Get the segment lengths of one artery element
   * @param[in] context: state of the coupling segments
   * @param[in] gid_to_segment_length: initial segment lengths per artery element GID (used for a
   * pure porofluid problem; non-const only because of the map access)
   * @param[in] artery_ele_gid: GID of the artery element
   * @return segment lengths of the artery element
   */
  std::vector<double> get_ele_segment_lengths(const ArterySegmentContext& context,
      std::map<int, std::vector<double>>& gid_to_segment_length, int artery_ele_gid);

  /*!
   * @brief Delete inactive / non-owned / duplicate pairs and resolve the cross-processor duplicate
   * segments that occur when a 1D element lies exactly between two 2D/3D elements owned by
   * different processors
   * @param[in] context: state of the coupling segments
   * @param[in,out] coupled_ele_pairs: coupling pairs; the dropped pairs are removed
   */
  void filter_coupling_pairs_and_resolve_cross_proc_duplicates(
      const ArterySegmentContext& context, ArteryCouplingPairVector& coupled_ele_pairs);
}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
