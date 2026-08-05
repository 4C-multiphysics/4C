// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_dirichlet_bc.hpp"

#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::DirichletBoundaryConditionHandler::DirichletBoundaryConditionHandler(
    const Teuchos::ParameterList& params)
    : params_(params)
{
  // get control parameters for initial/boundary conditions
  const Teuchos::ParameterList& params_conditions =
      params_.sublist("INITIAL AND BOUNDARY CONDITIONS");

  // read parameters relating particle types to values
  ParticleUtils::read_params_types_related_to_values(
      params_conditions, "DIRICHLET_BOUNDARY_CONDITION", dirichlet_bc_type_to_funct_id_);

  // iterate over particle types and insert into set
  for (const auto& type_iter : dirichlet_bc_type_to_funct_id_)
    types_subjected_to_dirichlet_bc_.insert(type_iter.first);

  // read particle types with per-particle Dirichlet function id
  {
    const auto& flagged_types = params_conditions.get<std::optional<std::vector<std::string>>>(
        "DIRICHLET_BOUNDARY_CONDITION_FLAGGED");
    if (flagged_types)
    {
      for (const auto& type_name : *flagged_types)
      {
        const Particle::TypeEnum particle_type = Particle::enum_from_type_name(type_name);
        FOUR_C_ASSERT_ALWAYS(particle_type != Particle::RigidPhase,
            "per-particle Dirichlet BC is not allowed for rigidphase");
        types_with_per_particle_dirichlet_bc_.insert(particle_type);
      }
    }
  }
}

void Particle::DirichletBoundaryConditionHandler::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particle_engine_interface)
{
  // set interface to particle engine
  particle_engine_interface_ = particle_engine_interface;
}

void Particle::DirichletBoundaryConditionHandler::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types subjected to dirichlet boundary conditions
  for (auto& particle_type : types_subjected_to_dirichlet_bc_)
  {
    // insert states for types subjected to dirichlet boundary conditions
    particlestatestotypes[particle_type].insert(Particle::ReferencePosition);
  }

  // iterate over particle types with per-particle Dirichlet function id
  for (auto& particle_type : types_with_per_particle_dirichlet_bc_)
  {
    particlestatestotypes[particle_type].insert(Particle::ReferencePosition);
    particlestatestotypes[particle_type].insert(Particle::DirichletFunctionId);
  }
}

void Particle::DirichletBoundaryConditionHandler::set_particle_reference_position() const
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particle_engine_interface_->get_particle_container_bundle();

  // iterate over particle types subjected to dirichlet boundary conditions
  for (auto& particle_type : types_subjected_to_dirichlet_bc_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particle_type, Particle::Status::Owned);

    // set particle reference position
    container->update_state(0.0, Particle::ReferencePosition, 1.0, Particle::Position);
  }

  // iterate over particle types with per-particle Dirichlet function id (skip if already handled)
  for (auto& particle_type : types_with_per_particle_dirichlet_bc_)
  {
    if (types_subjected_to_dirichlet_bc_.contains(particle_type)) continue;

    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particle_type, Particle::Status::Owned);

    // set particle reference position
    container->update_state(0.0, Particle::ReferencePosition, 1.0, Particle::Position);
  }
}

void Particle::DirichletBoundaryConditionHandler::evaluate_dirichlet_boundary_condition(
    const double& evaltime, const bool evalpos, const bool evalvel, const bool evalacc) const
{
  // degree of maximal function derivative
  int deg = 0;
  if (evalacc)
    deg = 2;
  else if (evalvel)
    deg = 1;

  // get bounding box dimensions
  Core::LinAlg::Matrix<3, 2> boundingbox =
      particle_engine_interface_->domain_bounding_box_corner_positions();

  // get bin size
  const std::array<double, 3> binsize = particle_engine_interface_->bin_size();

  // init vector containing evaluated function and derivatives
  std::vector<double> funct_time_deriv(deg + 1);

  // local lambda to apply Dirichlet BC of a single function to a single particle (usage see below)
  auto apply_dbc_to_particle = [&](const Core::Utils::FunctionOfSpaceTime& function, const int i,
                                   const double* refpos, double* pos, double* vel, double* acc)
  {
    const int statedim = static_cast<int>(function.number_components());
    // iterate over spatial dimension
    for (int dim = 0; dim < statedim; ++dim)
    {
      // evaluate function, first and second time derivative
      funct_time_deriv = function.evaluate_time_derivative(
          std::span(&(refpos[statedim * i]), 3), evaltime, deg, dim);

      // set position state
      if (evalpos)
      {
        // check for periodic boundary condition in current spatial direction
        if (particle_engine_interface_->have_periodic_boundary_conditions_in_spatial_direction(dim))
        {
          // length of binning domain in a spatial direction
          const double binning_domain_length =
              particle_engine_interface_->length_of_binning_domain_in_a_spatial_direction(dim);

          // get displacement from reference position canceling out multiples of periodic length
          // in current spatial direction
          double displacement = std::fmod(funct_time_deriv[0], binning_domain_length);
          double newpos = refpos[statedim * i + dim] + displacement;

          // shift by periodic length if new position is close to the periodic boundary and old
          // position is on other end domain
          if ((newpos > (boundingbox(dim, 1) - binsize[dim])) and
              (std::abs(newpos - pos[statedim * i + dim]) > binsize[dim]))
            pos[statedim * i + dim] = newpos - binning_domain_length;
          else if ((newpos < (boundingbox(dim, 0) + binsize[dim])) and
                   (std::abs(newpos - pos[statedim * i + dim]) > binsize[dim]))
            pos[statedim * i + dim] = newpos + binning_domain_length;
          else
            pos[statedim * i + dim] = newpos;
        }
        // no periodic boundary conditions in current spatial direction
        else
          pos[statedim * i + dim] = refpos[statedim * i + dim] + funct_time_deriv[0];
      }

      // set velocity state
      if (evalvel) vel[statedim * i + dim] = funct_time_deriv[1];

      // set acceleration state
      if (evalacc) acc[statedim * i + dim] = funct_time_deriv[2];
    }
  };

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particle_container_bundle =
      particle_engine_interface_->get_particle_container_bundle();

  // iterate over particle types subjected to dirichlet boundary conditions
  for (const auto& [particle_type, funct_id] : dirichlet_bc_type_to_funct_id_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particle_container_bundle->get_specific_container(particle_type, Particle::Status::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get function to apply
    const auto& function = *per_particle_function_cache_.at(funct_id);

    FOUR_C_ASSERT_ALWAYS(static_cast<std::size_t>(container->get_state_dim(Particle::Position)) ==
                             function.number_components(),
        "dimension of function defining dirichlet boundary condition not correct!");

    // get pointer to particle states
    const double* refpos = container->get_ptr_to_state(Particle::ReferencePosition, 0);
    double* pos = container->get_ptr_to_state_writable(Particle::Position, 0);
    double* vel = container->get_ptr_to_state_writable(Particle::Velocity, 0);
    double* acc = container->get_ptr_to_state_writable(Particle::Acceleration, 0);

    // iterate over owned particles of current type and apply dbc values to particle states
    for (int i = 0; i < particlestored; ++i)
      apply_dbc_to_particle(function, i, refpos, pos, vel, acc);
  }

  // per-particle Dirichlet BC: iterate over types with per-particle function id
  for (auto& particle_type : types_with_per_particle_dirichlet_bc_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particle_container_bundle->get_specific_container(particle_type, Particle::Status::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    FOUR_C_ASSERT_ALWAYS(container->have_stored_state(Particle::DirichletFunctionId),
        "DirichletFunctionId state is missing. This should not happen.");

    const int statedim = container->get_state_dim(Particle::Position);

    // get pointer to particle states
    const double* dirichlet_function_id =
        container->get_ptr_to_state(Particle::DirichletFunctionId, 0);
    const double* refpos = container->get_ptr_to_state(Particle::ReferencePosition, 0);
    double* pos = container->get_ptr_to_state_writable(Particle::Position, 0);
    double* vel = container->get_ptr_to_state_writable(Particle::Velocity, 0);
    double* acc = container->get_ptr_to_state_writable(Particle::Acceleration, 0);

    // iterate over owned particles of current type
    for (int i = 0; i < particlestored; ++i)
    {
      // skip particles without a per-particle Dirichlet function
      const int funct_id = static_cast<int>(dirichlet_function_id[i]);
      if (funct_id <= 0) continue;

      // get function to apply
      const auto& function = *per_particle_function_cache_.at(funct_id);

      FOUR_C_ASSERT_ALWAYS(static_cast<std::size_t>(statedim) == function.number_components(),
          "dimension of per-particle Dirichlet function (FUNCT {}) does not match state "
          "dimension!",
          funct_id);

      // apply dbc values to particle states
      apply_dbc_to_particle(function, i, refpos, pos, vel, acc);
    }
  }
}

void Particle::DirichletBoundaryConditionHandler::build_funct_cache(MPI_Comm comm)
{
  // collect all funct ids used locally on the different procs
  std::set<int> local_funct_ids;
  for (const auto& [particle_type, funct_id] : dirichlet_bc_type_to_funct_id_)
    local_funct_ids.insert(funct_id);

  // collect per-particle funct ids from owned particles
  Particle::ParticleContainerBundleShrdPtr bundle =
      particle_engine_interface_->get_particle_container_bundle();

  for (const auto& particle_type : types_with_per_particle_dirichlet_bc_)
  {
    Particle::ParticleContainer* container =
        bundle->get_specific_container(particle_type, Particle::Status::Owned);

    const int n = container->particles_stored();
    if (n <= 0) continue;

    const double* funct_id_ptr = container->get_ptr_to_state(Particle::DirichletFunctionId, 0);
    for (int i = 0; i < n; ++i)
    {
      const int funct_id = static_cast<int>(funct_id_ptr[i]);
      if (funct_id > 0) local_funct_ids.insert(funct_id);
    }
  }

  // communicate: every proc must know all funct ids used on any proc so that the cache is
  // complete before any particle transfer can bring a previously unseen funct id to this proc
  const int num_procs = Core::Communication::num_mpi_ranks(comm);

  const std::vector<int> local_funct_id_vec(local_funct_ids.begin(), local_funct_ids.end());
  const int local_count = static_cast<int>(local_funct_id_vec.size());

  std::vector<int> all_counts(num_procs);
  MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);

  std::vector<int> displs(num_procs, 0);
  for (int i = 1; i < num_procs; ++i) displs[i] = displs[i - 1] + all_counts[i - 1];

  std::vector<int> all_funct_ids(displs.back() + all_counts.back());
  MPI_Allgatherv(local_funct_id_vec.data(), local_count, MPI_INT, all_funct_ids.data(),
      all_counts.data(), displs.data(), MPI_INT, comm);

  for (const int funct_id : std::set<int>(all_funct_ids.begin(), all_funct_ids.end()))
  {
    if (funct_id > 0)
      per_particle_function_cache_[funct_id] =
          &Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct_id);
  }
}

FOUR_C_NAMESPACE_CLOSE
