// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_contact_tangential.hpp"

#include "4C_particle_input.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMContactTangentialBase::DEMContactTangentialBase(
    const Teuchos::ParameterList& params)
    : params_dem_(params), dt_(0.0)
{
  // empty constructor
}

void ParticleInteraction::DEMContactTangentialBase::init()
{
  // nothing to do
}

void ParticleInteraction::DEMContactTangentialBase::setup(const double& k_normal)
{
  // nothing to do
}

void ParticleInteraction::DEMContactTangentialBase::set_current_step_size(
    const double currentstepsize)
{
  dt_ = currentstepsize;
}

ParticleInteraction::DEMContactTangentialLinearSpringDamp::DEMContactTangentialLinearSpringDamp(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactTangentialBase(params),
      e_(params_dem_.get<double>("COEFF_RESTITUTION")),
      nue_(params_dem_.get<double>("POISSON_RATIO")),
      k_tangential_(0.0),
      d_tangential_fac_(0.0)
{
  // empty constructor
}

void ParticleInteraction::DEMContactTangentialLinearSpringDamp::init()
{
  // call base class init
  DEMContactTangentialBase::init();

  // safety checks for contact parameters
  if (nue_ <= -1.0 or nue_ > 0.5)
    FOUR_C_THROW("invalid input parameter POISSON_RATIO (expected in range ]-1.0; 0.5])!");

  if (params_dem_.get<double>("FRICT_COEFF_TANG") <= 0.0)
    FOUR_C_THROW("invalid input parameter FRICT_COEFF_TANG for this kind of contact law!");
}

void ParticleInteraction::DEMContactTangentialLinearSpringDamp::setup(const double& k_normal)
{
  // call base class setup
  DEMContactTangentialBase::setup(k_normal);

  // tangential to normal stiffness ratio
  const double kappa = (1.0 - nue_) / (1.0 - 0.5 * nue_);

  // tangential contact stiffness
  k_tangential_ = kappa * k_normal;

  // determine tangential contact damping factor
  if (e_ > 0.0)
  {
    const double lne = std::log(e_);
    d_tangential_fac_ =
        2.0 * std::abs(lne) *
        std::sqrt(k_normal / (Utils::pow<2>(lne) + Utils::pow<2>(std::numbers::pi)));
  }
  else
    d_tangential_fac_ = 2.0 * std::sqrt(k_normal);
}

void ParticleInteraction::DEMContactTangentialLinearSpringDamp::tangential_contact_force(
    double* gap_tangential, bool& stick_tangential, const double* normal,
    const double* v_rel_tangential, const double& m_eff, const double& mu_tangential,
    const double& normalcontactforce, double* tangentialcontactforce) const
{
  // determine tangential contact damping parameter
  const double d_tangential = d_tangential_fac_ * std::sqrt(m_eff);

  // compute length of tangential gap at time n
  const double old_length = Utils::vec_norm_two(gap_tangential);

  // compute projection of tangential gap onto current normal at time n+1
  Utils::vec_add_scale(gap_tangential, -Utils::vec_dot(normal, gap_tangential), normal);

  // compute length of tangential gap at time n+1
  const double new_length = Utils::vec_norm_two(gap_tangential);

  // maintain length of tangential gap equal to before the projection
  if (new_length > 1.0e-14)
    Utils::vec_set_scale(gap_tangential, old_length / new_length, gap_tangential);

  // update of elastic tangential displacement if stick is true
  if (stick_tangential == true) Utils::vec_add_scale(gap_tangential, dt_, v_rel_tangential);

  // compute tangential contact force (assume stick-case)
  Utils::vec_set_scale(tangentialcontactforce, -k_tangential_, gap_tangential);
  Utils::vec_add_scale(tangentialcontactforce, -d_tangential, v_rel_tangential);

  // compute the norm of the tangential contact force
  const double norm_tangentialcontactforce = Utils::vec_norm_two(tangentialcontactforce);

  // tangential contact force for stick-case
  if (norm_tangentialcontactforce <= (mu_tangential * std::abs(normalcontactforce)))
  {
    stick_tangential = true;

    // tangential contact force already computed
  }
  // tangential contact force for slip-case
  else
  {
    stick_tangential = false;

    // compute tangential contact force
    Utils::vec_set_scale(tangentialcontactforce,
        mu_tangential * std::abs(normalcontactforce) / norm_tangentialcontactforce,
        tangentialcontactforce);

    // compute tangential displacement
    const double inv_k_tangential = 1.0 / k_tangential_;
    Utils::vec_set_scale(gap_tangential, -inv_k_tangential, tangentialcontactforce);
    Utils::vec_add_scale(gap_tangential, -inv_k_tangential * d_tangential, v_rel_tangential);
  }
}

void ParticleInteraction::DEMContactTangentialLinearSpringDamp::tangential_potential_energy(
    const double* gap_tangential, double& tangentialpotentialenergy) const
{
  tangentialpotentialenergy = 0.5 * k_tangential_ * Utils::vec_dot(gap_tangential, gap_tangential);
}

FOUR_C_NAMESPACE_CLOSE
