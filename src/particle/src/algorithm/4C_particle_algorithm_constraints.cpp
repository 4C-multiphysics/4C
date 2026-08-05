// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_constraints.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_input.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

Particle::ConstraintsProjectionBase::ConstraintsProjectionBase(MPI_Comm comm, double tol)
    : comm_(comm), eps_(tol)
{
}

void Particle::ConstraintsProjectionBase::apply(
    Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
    const std::set<Particle::Type>& types_to_integrate, const double time) const
{
  // perform setup if needed
  setup(particle_container_bundle, types_to_integrate);

  // iterate over particle types
  for (auto& particleType : types_to_integrate)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particle_container_bundle->get_specific_container(particleType, Particle::Status::Owned);

    // get number of particles stored in container
    const int n_particle_stored = container->particles_stored();

    // no owned particles of current particle type
    if (n_particle_stored <= 0) continue;

    // get pointer to particle velocity and acceleration
    double* vel = container->get_ptr_to_state_writable(Particle::Velocity, 0);
    double* acc = container->get_ptr_to_state_writable(Particle::Acceleration, 0);
    double* modvel = container->try_get_ptr_to_state_writable(Particle::ModifiedVelocity, 0);
    double* modacc = container->try_get_ptr_to_state_writable(Particle::ModifiedAcceleration, 0);

    // get particle state dimension
    const int pos_state_dim = container->get_state_dim(Particle::Position);

    // project quantities
    project(n_particle_stored, pos_state_dim, vel, acc, modvel, modacc);
  }
}

int Particle::ConstraintsProjectionBase::calc_primary_axis(
    Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
    const std::set<Particle::Type>& types_to_integrate) const
{
  int local_result = calc_primary_axis_local(particle_container_bundle, types_to_integrate);

  int global_result;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX, comm_);

  FOUR_C_ASSERT_ALWAYS(global_result != axis_invalid_, "No rank could determine primary axis");

  FOUR_C_ASSERT_ALWAYS(local_result == axis_invalid_ || local_result == global_result,
      "Primary axis mismatch across ranks");

  return global_result;
}

void Particle::ConstraintsProjectionBase::setup(
    Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
    const std::set<Particle::Type>& types_to_integrate) const
{
  if (axis_ == axis_invalid_)
  {
    const int new_axis = calc_primary_axis(particle_container_bundle, types_to_integrate);
    set_primary_axis(new_axis);
    check_particles(particle_container_bundle, types_to_integrate);
  }
}

void Particle::ConstraintsProjectionBase::check_particles(
    Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
    const std::set<Particle::Type>& types_to_integrate) const
{
  for (auto& particleType : types_to_integrate)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particle_container_bundle->get_specific_container(particleType, Particle::Status::Owned);

    // get number of particles stored in container
    const int n_particle_stored = container->particles_stored();

    // no owned particles of current particle type
    if (n_particle_stored <= 0) continue;

    // get pointer to particle position
    const double* pos = container->get_ptr_to_state(Particle::Position, 0);

    // get pointer to particle radius
    const double* rad = container->get_ptr_to_state(Particle::Radius, 0);

    // get particle state dimension
    const int pos_state_dim = container->get_state_dim(Particle::Position);

    check(n_particle_stored, pos_state_dim, pos, rad);
  }
}

Particle::ConstraintsProjection2D::ConstraintsProjection2D(MPI_Comm comm)
    : ConstraintsProjectionBase(comm)
{
}

int Particle::ConstraintsProjection2D::calc_primary_axis_local(
    Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
    const std::set<Particle::Type>& types_to_integrate) const
{
  int axis_direction = axis_invalid_;

  // working vectors
  double a[3], b[3], c[3];

  // at first find the normal - iterate over particle types
  for (auto& particleType : types_to_integrate)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particle_container_bundle->get_specific_container(particleType, Particle::Status::Owned);

    // get number of particles stored in container
    const int n_particle_stored = container->particles_stored();

    // no owned particles of current particle type or not enough particles at all
    if (n_particle_stored < 3) continue;

    // get pointer to particle position and radius
    const double* pos = container->get_ptr_to_state(Particle::Position, 0);
    const double* rad = container->get_ptr_to_state(Particle::Radius, 0);

    // get particle state dimension
    const int pos_state_dim = container->get_state_dim(Particle::Position);

    // We already assume that there is not coinciding points
    const auto p0 = pos;
    const auto p1 = pos + pos_state_dim;
    const auto r0 = rad;

    ParticleUtils::vec_set(a, p1);
    ParticleUtils::vec_add_scale(a, -1., p0);

    int i2 = 2;
    for (; i2 < n_particle_stored; ++i2)
    {
      const auto p2 = pos + pos_state_dim * i2;
      const auto r2 = rad + i2;

      ParticleUtils::vec_set(b, p2);
      ParticleUtils::vec_add_scale(b, -1., p0);

      ParticleUtils::vec_set_cross(c, a, b);

      if (ParticleUtils::vec_norm_two(c) > eps_ * std::min(r0[0], r2[0])) break;
    }

    // Means we have found the normal
    if (i2 < n_particle_stored)
    {
      // Normalize the normal vector, so the dot product should be close to 1 for the aligned axis
      const auto nrm_c = ParticleUtils::vec_norm_two(c);
      ParticleUtils::vec_scale(c, 1.0 / nrm_c);

      // Identify the direction
      int dim = 0;
      for (; dim < 3; ++dim)
      {
        double axis[3] = {0., 0., 0.};
        axis[dim] = 1.0;

        if (std::abs(ParticleUtils::vec_dot(c, axis)) > (1.0 - eps_))
        {
          axis_direction = dim;
          break;
        }
      }

      FOUR_C_ASSERT_ALWAYS(dim < 3, "Plane normal is not aligned with x, y, or z axis");

      break;
    }
  }

  return axis_direction;
}

void Particle::ConstraintsProjection2D::set_primary_axis(int new_axis) const { axis_ = new_axis; }

void Particle::ConstraintsProjection2D::project(int n_particle_stored, int pos_state_dim,
    double* vel, double* acc, double* modvel, double* modacc) const
{
  // iterate over all owned particles
  for (int i = 0; i < n_particle_stored; ++i)
  {
    const auto idx = pos_state_dim * i + axis_;

    vel[idx] = 0.0;
    acc[idx] = 0.0;
    if (modvel) modvel[idx] = 0.0;
    if (modacc) modacc[idx] = 0.0;
  }
}

void Particle::ConstraintsProjection2D::check(
    int n_particle_stored, int pos_state_dim, const double* pos, const double* rad) const
{
  double curvec[3];
  double normal[3] = {0., 0., 0.};
  normal[axis_] = 1.0;

  const auto p0 = pos;
  const auto r0 = rad;

  // Check coplanarity
  for (int i = 1; i < n_particle_stored; ++i)
  {
    const auto p = pos + pos_state_dim * i;
    const auto r = rad + i;
    ParticleUtils::vec_set(curvec, p);
    ParticleUtils::vec_add_scale(curvec, -1., p0);

    FOUR_C_ASSERT_ALWAYS(ParticleUtils::vec_dot(curvec, normal) <= eps_ * std::min(r0[0], r[0]),
        "Non-coplanar particles detected");
  }
}

Particle::ConstraintsProjection1D::ConstraintsProjection1D(MPI_Comm comm)
    : ConstraintsProjectionBase(comm)
{
}

int Particle::ConstraintsProjection1D::calc_primary_axis_local(
    Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
    const std::set<Particle::Type>& types_to_integrate) const
{
  int axis_direction = axis_invalid_;

  // working vectors
  double axis[3];

  // at first find the axis - iterate over particle types
  for (auto& particleType : types_to_integrate)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particle_container_bundle->get_specific_container(particleType, Particle::Status::Owned);

    // get number of particles stored in container
    const int n_particle_stored = container->particles_stored();

    // no owned particles of current particle type or not enough particles at all
    if (n_particle_stored < 2) continue;

    // get pointer to particle position
    const double* pos = container->get_ptr_to_state(Particle::Position, 0);

    // get particle state dimension
    const int pos_state_dim = container->get_state_dim(Particle::Position);

    // We already assume that there is not coinciding points
    const auto p0 = pos;
    const auto p1 = pos + pos_state_dim;

    ParticleUtils::vec_set(axis, p1);
    ParticleUtils::vec_add_scale(axis, -1., p0);

    // Normalize the normal vector, so the dot product should be close to 1 for the aligned axis
    const auto nrm_axis = ParticleUtils::vec_norm_two(axis);
    ParticleUtils::vec_scale(axis, 1.0 / nrm_axis);

    // Identify the axis direction
    int dim = 0;
    for (; dim < 3; ++dim)
    {
      double axis_try[3] = {0., 0., 0.};
      axis_try[dim] = 1.0;

      if (ParticleUtils::vec_dot(axis_try, axis) > (1.0 - eps_))
      {
        axis_direction = dim;
        break;
      }
    }

    FOUR_C_ASSERT_ALWAYS(dim < 3, "Axis is not aligned with x, y, or z direction");

    break;
  }

  return axis_direction;
}

void Particle::ConstraintsProjection1D::set_primary_axis(int new_axis) const
{
  axis_ = new_axis;

  std::vector<int> indices;
  for (int dim = 0; dim < 3; ++dim)
    if (new_axis != dim) indices.push_back(dim);

  FOUR_C_ASSERT_ALWAYS(
      indices.size() == 2, "Inconsistent number of indices to restrain identified");

  index0_ = indices[0];
  index1_ = indices[1];
}

void Particle::ConstraintsProjection1D::project(int n_particle_stored, int pos_state_dim,
    double* vel, double* acc, double* modvel, double* modacc) const
{
  // iterate over all owned particles
  for (int i = 0; i < n_particle_stored; ++i)
  {
    const auto idx1 = pos_state_dim * i + index0_;
    const auto idx2 = pos_state_dim * i + index1_;

    vel[idx1] = 0.0;
    vel[idx2] = 0.0;
    acc[idx1] = 0.0;
    acc[idx2] = 0.0;

    if (modvel)
    {
      modvel[idx1] = 0.0;
      modvel[idx2] = 0.0;
    }
    if (modacc)
    {
      modacc[idx1] = 0.0;
      modacc[idx2] = 0.0;
    }
  }
}

void Particle::ConstraintsProjection1D::check(
    int n_particle_stored, int pos_state_dim, const double* pos, const double* rad) const
{
  double curvec[3];
  double cproduct[3];
  double normal[3] = {0., 0., 0.};
  normal[axis_] = 1.0;

  const auto p0 = pos;
  const auto r0 = rad;

  // Check collinearity
  for (int i = 1; i < n_particle_stored; ++i)
  {
    const auto p = pos + pos_state_dim * i;
    const auto r = rad + i;
    ParticleUtils::vec_set(curvec, p);
    ParticleUtils::vec_add_scale(curvec, -1., p0);

    ParticleUtils::vec_set_cross(cproduct, curvec, normal);

    FOUR_C_ASSERT_ALWAYS(ParticleUtils::vec_norm_two(cproduct) <= eps_ * std::min(r0[0], r[0]),
        "Non-collinear particles detected");
  }
}

std::unique_ptr<Particle::ConstraintsHandler> Particle::create_constraints(
    const Teuchos::ParameterList& params, MPI_Comm comm)
{
  std::unique_ptr<Particle::ConstraintsHandler> constraints = nullptr;

  const auto constraint_type =
      params.sublist("INITIAL AND BOUNDARY CONDITIONS").get<Particle::Constraint>("CONSTRAINT");

  switch (constraint_type)
  {
    case Particle::NoConstraint:
      break;

    case Particle::Projection1D:
    {
      constraints = std::make_unique<Particle::ConstraintsProjection1D>(comm);
      break;
    }

    case Particle::Projection2D:
    {
      constraints = std::make_unique<Particle::ConstraintsProjection2D>(comm);
      break;
    }
  }

  return constraints;
}

FOUR_C_NAMESPACE_CLOSE
