// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_surfbased.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::
    PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params, const std::string& condition_name)
    : PorofluidElastScatraArteryCouplingNonConformingAlgorithm(
          artery_dis, homogenized_dis, coupling_params, condition_name)
{
  // user info
  if (my_mpi_rank_ == 0)
  {
    std::cout << "<                                                  >" << '\n';
    print_coupling_method();
    std::cout << "<                                                  >" << '\n';
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << '\n';
    std::cout << "\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::
    pre_evaluate_coupling_pairs()
{
  const int num_patches_axial = Global::Problem::instance()
                                    ->porofluid_pressure_based_dynamic_params()
                                    .sublist("artery_coupling")
                                    .sublist("integration_patches")
                                    .get<int>("number_of_patches_axial");
  const int num_patches_radial = Global::Problem::instance()
                                     ->porofluid_pressure_based_dynamic_params()
                                     .sublist("artery_coupling")
                                     .sublist("integration_patches")
                                     .get<int>("number_of_patches_radial");
  const int num_artery_ele = artery_dis_->num_global_elements();
  const int num_gp_per_artery_ele = num_patches_axial * num_patches_radial * 25;
  const int num_gp_desired = num_gp_per_artery_ele * num_artery_ele;

  // this vector keeps track of the Gauss point (GP) evaluation
  const auto gp_vector = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *artery_dis_->element_col_map(), num_gp_per_artery_ele);

  // pre-evaluate
  for (const auto& coupled_ele_pair : coupled_ele_pairs_) coupled_ele_pair->pre_evaluate(gp_vector);

  // delete the inactive pairs
  std::erase_if(coupled_ele_pairs_, [](const auto& pair) { return !pair->is_active(); });

  // The following takes care of a very special case, namely, if a GP on the lateral surface lies
  // exactly in between two or more 3D elements owned by different procs. In that case, the GP is
  // duplicated across all owning procs. We detect such cases and communicate them to all procs
  // below by adapting the "gp_vector", to contain the multiplicity of the respective GP. Finally,
  // the GP weight inside the coupling pair is scaled by the inverse of the multiplicity.

  int duplicates = 0;
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
  {
    std::vector my_gp_vector(num_gp_per_artery_ele, 0);
    std::vector sum_gp_vectors(num_gp_per_artery_ele, 0);
    // loop over all GIDs
    for (int gid = gp_vector->get_map().min_all_gid(); gid <= gp_vector->get_map().max_all_gid();
        gid++)
    {
      // reset
      std::fill_n(sum_gp_vectors.data(), num_gp_per_artery_ele, 0);

      const int my_lid = gp_vector->get_map().lid(gid);
      // if not owned or ghosted fill with zeros
      if (my_lid < 0) std::fill_n(my_gp_vector.data(), num_gp_per_artery_ele, 0);
      // else get the GP vector
      else
        for (int igp = 0; igp < num_gp_per_artery_ele; igp++)
          my_gp_vector[igp] = static_cast<int>((*gp_vector)(igp).local_values_as_span()[my_lid]);

      // communicate to all via summation
      sum_gp_vectors = Core::Communication::sum_all(my_gp_vector, get_comm());

      // This is ok for now, either the GID does not exist or the entire element protrudes.
      // Inform user and continue
      if (*std::max_element(sum_gp_vectors.data(), sum_gp_vectors.data() + num_gp_per_artery_ele) <
          1)
      {
        std::cout << "WARNING! No GP of element  " << gid + 1 << " could be projected!" << '\n';
        continue;
      }

      // if one entry is equal to zero, this GP could not be projected
      if (*std::min_element(sum_gp_vectors.data(), sum_gp_vectors.data() + num_gp_per_artery_ele) <
          1)
        FOUR_C_THROW("It seems as if one GP could not be projected");

      // find number of duplicates
      int sum = 0;
      sum = std::accumulate(
          sum_gp_vectors.data(), sum_gp_vectors.data() + num_gp_per_artery_ele, sum);
      duplicates += sum - num_gp_per_artery_ele;

      // if owned or ghosted by this proc. and if duplicates have been detected, replace entry in
      // gp_vector
      if (my_lid >= 0 && sum > num_gp_per_artery_ele)
      {
        for (int igp = 0; igp < num_gp_per_artery_ele; igp++)
        {
          gp_vector->replace_local_value(my_lid, igp, sum_gp_vectors[igp]);
        }
      }
    }
  }

  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
    coupled_ele_pair->delete_unnecessary_gps(gp_vector);

  int total_num_gp = 0;
  int num_gp = 0;

  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    // segment ID is not needed in this case, just set to zero
    coupled_ele_pair->set_segment_id(0);
    num_gp = num_gp + coupled_ele_pair->num_gp();
  }
  // safety check
  total_num_gp = Core::Communication::sum_all(num_gp, get_comm());
  if (num_gp_desired != total_num_gp - duplicates)
    FOUR_C_THROW("It seems as if some GPs could not be projected");

  // output
  int total_num_active_pairs = 0;
  int num_active_pairs = static_cast<int>(coupled_ele_pairs_.size());
  total_num_active_pairs = Core::Communication::sum_all(num_active_pairs, get_comm());
  if (homogenized_dis_->name() == "porofluid" && my_mpi_rank_ == 0)
    std::cout << "Only " << total_num_active_pairs
              << " Artery-to-PoroMultiphaseScatra coupling pairs are active" << '\n';

  // print summary of pairs
  if (homogenized_dis_->name() == "porofluid" &&
      coupling_params_.get<bool>("print_coupling_pairs_summary"))
  {
    if (my_mpi_rank_ == 0)
      std::cout << "In total " << num_gp_desired << " GPs (" << num_gp_per_artery_ele
                << " per artery element) required for lateral surface coupling" << '\n';
    std::cout << "Proc. " << my_mpi_rank_ << " evaluates " << num_gp << " GPs " << "("
              << static_cast<double>(num_gp) / static_cast<double>(total_num_gp) * 100.0
              << "% of all GPs)" << '\n';
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::setup()
{
  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup();

  // error-checks
  if (has_variable_diameter_)
    FOUR_C_THROW("Variable diameter not yet possible for surface-based coupling");
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not yet possible for surface-based coupling");

  is_setup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::evaluate(
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  if (!is_setup_) FOUR_C_THROW("setup() has not been called");

  if (!porofluid_managers_initialized_)
  {
    // pre-evaluate the pairs
    // --> has to be done here since the radius inside the material is required
    pre_evaluate_coupling_pairs();
  }

  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::evaluate(sysmat, rhs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::setup_system(
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_artery,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
    const std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_homogenized,
    const std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_artery)
{
  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup_system(*sysmat, rhs,
      *sysmat_homogenized, *sysmat_artery, rhs_homogenized, rhs_artery, *dbcmap_homogenized,
      *dbcmap_artery->cond_map(), *dbcmap_artery->cond_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::
    apply_mesh_movement()
{
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not possible for surface-based coupling");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> PoroPressureBased::
    PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::blood_vessel_volume_fraction()
{
  FOUR_C_THROW("Output of vessel volume fraction not possible for surface-based coupling");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingSurfaceBasedAlgorithm::
    print_coupling_method() const
{
  std::cout << "<   surface-based formulation                      >" << '\n';
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::print_coupling_method();
}

FOUR_C_NAMESPACE_CLOSE
