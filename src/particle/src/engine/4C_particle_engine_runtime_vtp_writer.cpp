// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_runtime_vtp_writer.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ParticleRuntimeVtpWriter::ParticleRuntimeVtpWriter(MPI_Comm comm)
    : comm_(comm), setuptime_(0.0)
{
  // empty constructor
}

void Particle::ParticleRuntimeVtpWriter::init(
    const ParticleContainerBundleShrdPtr particlecontainerbundle)
{
  // set particle container bundle
  particlecontainerbundle_ = particlecontainerbundle;

  // insert specific particle states in black list
  blackliststates_.insert({DensitySum, DensityDot});
  blackliststates_.insert(TemperatureDot);
  blackliststates_.insert({LastTransferPosition, ReferencePosition});
  blackliststates_.insert({ModifiedVelocity, ModifiedAcceleration});
  blackliststates_.insert({InterfaceNormal, Curvature, WallColorfield, WallInterfaceNormal});
  blackliststates_.insert({LastIterPosition, LastIterVelocity, LastIterAcceleration,
      LastIterAngularVelocity, LastIterAngularAcceleration, LastIterModifiedAcceleration,
      LastIterDensity, LastIterTemperature});
}

void Particle::ParticleRuntimeVtpWriter::setup(bool write_ghosted_particles)
{
  // determine size of vector indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  // allocate memory to hold particle types
  runtime_visualization_managers_.resize(typevectorsize);

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // allocate memory for vtp writer objects of owned and ghosted states
    (runtime_visualization_managers_[type]).resize(2);

    // iterate over particle statuses
    for (const auto& status : {Owned, Ghosted})
    {
      if (status == Ghosted and write_ghosted_particles == false) continue;

      std::ostringstream fieldname;
      fieldname << "particle-" << enum_to_type_name(type) << "-" << enum_to_status_name(status);

      // construct visualiation manager object for current particle type and status
      (runtime_visualization_managers_[type])[status] =
          std::make_shared<Core::IO::VisualizationManager>(
              Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), setuptime_),
              comm_, fieldname.str());
    }
  }
}

void Particle::ParticleRuntimeVtpWriter::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader)
{
  // get restart time
  setuptime_ = reader->read_double("time");
}

void Particle::ParticleRuntimeVtpWriter::set_particle_positions_and_states()
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // iterate over particle statuses
    for (const auto& status : {Owned, Ghosted})
    {
      // check for runtime vtp writer for current particle type and status
      if (not(runtime_visualization_managers_[type])[status]) continue;

      // get container of current particle type and status
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, status);

      // get number of particles stored in container
      const int particlestored = container->particles_stored();

      // get particle states stored in container
      const std::set<ParticleState>& states = container->get_stored_states();

#ifdef FOUR_C_ENABLE_ASSERTIONS
      // safety check
      if (not container->have_stored_state(Position))
        FOUR_C_THROW("particle state '{}' not found!", enum_to_state_name(Position));
#endif

      // iterate over particle states
      for (const auto& state : states)
      {
        // get particle state dimension
        int statedim = container->get_state_dim(state);

        // get name of particle state
        std::string statename = enum_to_state_name(state);

        // get pointer to particle state
        const double* state_ptr =
            (particlestored > 0) ? container->get_ptr_to_state(state, 0) : nullptr;

        if (state == Position)
        {
          // get and prepare storage for position data
          std::vector<double>& positiondata = (runtime_visualization_managers_[type])[status]
                                                  ->get_visualization_data()
                                                  .get_point_coordinates();
          positiondata.clear();
          positiondata.reserve(statedim * particlestored);

          // copy particle position data
          for (int i = 0; i < (statedim * particlestored); ++i)
            positiondata.push_back(state_ptr[i]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
          // safety check
          if (static_cast<int>(positiondata.size()) != statedim * particlestored)
            FOUR_C_THROW("ParticleRuntimeVtpWriter expected {} coordinate values, but got {}!",
                statedim * particlestored, positiondata.size());
#endif
        }
        else if (not blackliststates_.contains(state))
        {
          // prepare particle state data
          std::vector<double> statedata;
          statedata.reserve(statedim * particlestored);

          // copy particle state data
          for (int i = 0; i < (statedim * particlestored); ++i) statedata.push_back(state_ptr[i]);

          // append particle state data to vtp writer
          (runtime_visualization_managers_[type])[status]
              ->get_visualization_data()
              .set_point_data_vector<double>(statename, statedata, statedim);
        }
      }

      // get pointer to global id of particles
      const int* globalids = (particlestored > 0) ? container->get_ptr_to_global_id(0) : nullptr;

      // prepare particle global id data
      std::vector<int> globaliddata;
      globaliddata.reserve(particlestored);

      // copy particle global id data
      for (int i = 0; i < particlestored; ++i) globaliddata.push_back(globalids[i]);

      // append global id of particles to vtp writer
      (runtime_visualization_managers_[type])[status]
          ->get_visualization_data()
          .set_point_data_vector<int>("globalid", globaliddata, 1);

      // set particle owner data
      std::vector<int> ownerdata(particlestored, Core::Communication::my_mpi_rank(comm_));

      // append owner of particles to vtp writer
      (runtime_visualization_managers_[type])[status]
          ->get_visualization_data()
          .set_point_data_vector<int>("owner", ownerdata, 1);
    }
  }
}

void Particle::ParticleRuntimeVtpWriter::write_to_disk(
    const double visualization_time, const int visualization_step)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // iterate over particle statuses
    for (const auto& status : {Owned, Ghosted})
    {
      // check for runtime vtp writer for current particle type and status
      if (not(runtime_visualization_managers_[type])[status]) continue;

      (runtime_visualization_managers_[type])[status]->write_to_disk(
          visualization_time, visualization_step);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
