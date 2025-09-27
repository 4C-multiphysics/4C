// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nodetopoint.hpp"

#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::
    PorofluidElastScatraArteryCouplingNodeToPointAlgorithm(
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
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::setup()
{
  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup();

  // pre-evaluate coupling pairs
  pre_evaluate_coupling_pairs();

  // print out summary of pairs
  if (homogenized_dis_->name() == "porofluid" &&
      coupling_params_.get<bool>("print_coupling_pairs_summary"))
    output_coupling_pairs();

  // error-checks
  if (has_variable_diameter_)
    FOUR_C_THROW("Variable diameter not yet possible for node-to-point coupling");
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not yet possible for node-to-point coupling");

  is_setup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::
    pre_evaluate_coupling_pairs()
{
  // pre-evaluate
  for (const auto& coupled_ele_pair : coupled_ele_pairs_) coupled_ele_pair->pre_evaluate(nullptr);

  // delete the inactive pairs
  coupled_ele_pairs_.erase(
      std::remove_if(coupled_ele_pairs_.begin(), coupled_ele_pairs_.end(),
          [](const std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>& coupling_pair)
          { return not coupling_pair->is_active(); }),
      coupled_ele_pairs_.end());

  // output
  int total_num_active_pairs = 0;
  int num_active_pairs = static_cast<int>(coupled_ele_pairs_.size());
  total_num_active_pairs = Core::Communication::sum_all(num_active_pairs, get_comm());
  if (my_mpi_rank_ == 0)
  {
    std::cout << total_num_active_pairs
              << " Artery-to-PoroMultiphaseScatra coupling pairs are active" << '\n';
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::evaluate(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  if (!is_setup_) FOUR_C_THROW("setup() has not been called");

  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::evaluate(sysmat, rhs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::setup_system(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_artery,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_homogenized,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_artery)
{
  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup_system(*sysmat, rhs,
      *sysmat_homogenized, *sysmat_artery, rhs_homogenized, rhs_artery, *dbcmap_homogenized,
      *dbcmap_artery->cond_map(), *dbcmap_artery->cond_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::
    apply_mesh_movement()
{
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not possible for node-to-point coupling");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> PoroPressureBased::
    PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::blood_vessel_volume_fraction()
{
  FOUR_C_THROW("Output of vessel volume fraction not possible for node-to-point coupling");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::
    print_coupling_method() const
{
  std::cout << "<Coupling-Method: 1D node to coincident point in 3D>" << '\n';
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNodeToPointAlgorithm::
    output_coupling_pairs() const
{
  if (my_mpi_rank_ == 0)
  {
    std::cout << "\nSummary of coupling pairs (segments):" << '\n';
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << '\n';
  }
  Core::Communication::barrier(get_comm());
  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    std::cout << "Proc " << std::right << std::setw(2) << my_mpi_rank_ << ": Artery-ele "
              << std::right << std::setw(5) << coupled_ele_pair->artery_ele_gid()
              << ": <---> continuous-ele " << std::right << std::setw(7)
              << coupled_ele_pair->homogenized_ele_gid() << '\n';
  }
  Core::Communication::barrier(get_comm());
  if (my_mpi_rank_ == 0) std::cout << "\n";
}

FOUR_C_NAMESPACE_CLOSE