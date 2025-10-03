// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_contact_rolling.hpp"

#include "4C_particle_input.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMContactRollingBase::DEMContactRollingBase(
    const Teuchos::ParameterList& params)
    : params_dem_(params),
      dt_(0.0),
      e_(params_dem_.get<double>("COEFF_RESTITUTION")),
      nue_(params_dem_.get<double>("POISSON_RATIO")),
      d_rolling_fac_(0.0)
{
  // empty constructor
}

void ParticleInteraction::DEMContactRollingBase::init()
{
  // safety checks for contact parameters
  if (nue_ <= -1.0 or nue_ > 0.5)
    FOUR_C_THROW("invalid input parameter POISSON_RATIO (expected in range ]-1.0; 0.5])!");

  if (params_dem_.get<double>("FRICT_COEFF_ROLL") <= 0.0)
    FOUR_C_THROW("invalid input parameter FRICT_COEFF_ROLL for this kind of contact law!");
}

void ParticleInteraction::DEMContactRollingBase::setup(const double& k_normal)
{
  // nothing to do
}

void ParticleInteraction::DEMContactRollingBase::set_current_step_size(const double currentstepsize)
{
  dt_ = currentstepsize;
}

ParticleInteraction::DEMContactRollingViscous::DEMContactRollingViscous(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactRollingBase(params),
      young_(params_dem_.get<double>("YOUNG_MODULUS")),
      v_max_(params_dem_.get<double>("MAX_VELOCITY"))
{
  // empty constructor
}

void ParticleInteraction::DEMContactRollingViscous::init()
{
  // call base class init
  DEMContactRollingBase::init();

  // safety checks for contact parameters
  if (young_ <= 0.0)
    FOUR_C_THROW("invalid input parameter YOUNG_MODULUS (expected to be positive)!");

  if (v_max_ <= 0.0)
    FOUR_C_THROW("invalid input parameter MAX_VELOCITY (expected to be positive)!");
}

void ParticleInteraction::DEMContactRollingViscous::setup(const double& k_normal)
{
  // call base class setup
  DEMContactRollingBase::setup(k_normal);

  // determine rolling contact damping factor
  const double fac = young_ / (1.0 - Utils::pow<2>(nue_));
  const double c_1 = 1.15344;
  d_rolling_fac_ = (1.0 - e_) / (c_1 * std::pow(fac, 0.4) * std::pow(v_max_, 0.2));
}

void ParticleInteraction::DEMContactRollingViscous::effective_radius_particle(
    const double* radius_i, const double* radius_j, const double& gap, double& r_eff) const
{
  if (radius_j)
    r_eff = (radius_i[0] * radius_j[0]) / (radius_i[0] + radius_j[0]);
  else
    r_eff = radius_i[0];
}

void ParticleInteraction::DEMContactRollingViscous::relative_rolling_velocity(const double& r_eff,
    const double* normal, const double* angvel_i, const double* angvel_j,
    double* v_rel_rolling) const
{
  Utils::vec_set_cross(v_rel_rolling, angvel_i, normal);
  if (angvel_j) Utils::vec_add_cross(v_rel_rolling, normal, angvel_j);
}

void ParticleInteraction::DEMContactRollingViscous::rolling_contact_moment(double* gap_rolling,
    bool& stick_rolling, const double* normal, const double* v_rel_rolling, const double& m_eff,
    const double& r_eff, const double& mu_rolling, const double& normalcontactforce,
    double* rollingcontactmoment) const
{
  // determine rolling contact damping parameter
  const double d_rolling = d_rolling_fac_ * mu_rolling * std::pow(0.5 * r_eff, -0.2);

  // compute rolling contact force
  double rollingcontactforce[3];
  Utils::vec_set_scale(rollingcontactforce, -(d_rolling * normalcontactforce), v_rel_rolling);

  // compute rolling contact moment
  Utils::vec_set_cross(rollingcontactmoment, rollingcontactforce, normal);
  Utils::vec_scale(rollingcontactmoment, r_eff);
}

void ParticleInteraction::DEMContactRollingViscous::rolling_potential_energy(
    const double* gap_rolling, double& rollingpotentialenergy) const
{
  rollingpotentialenergy = 0.0;
}

ParticleInteraction::DEMContactRollingCoulomb::DEMContactRollingCoulomb(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactRollingBase(params), k_rolling_(0.0)
{
  // empty constructor
}

void ParticleInteraction::DEMContactRollingCoulomb::setup(const double& k_normal)
{
  // call base class setup
  DEMContactRollingBase::setup(k_normal);

  // rolling to normal stiffness ratio
  const double kappa = (1.0 - nue_) / (1.0 - 0.5 * nue_);

  // rolling contact stiffness
  k_rolling_ = kappa * k_normal;

  // determine rolling contact damping factor
  if (e_ > 0.0)
  {
    const double lne = std::log(e_);
    d_rolling_fac_ = 2.0 * std::abs(lne) *
                     std::sqrt(k_normal / (Utils::pow<2>(lne) + Utils::pow<2>(std::numbers::pi)));
  }
  else
    d_rolling_fac_ = 2.0 * std::sqrt(k_normal);
}

void ParticleInteraction::DEMContactRollingCoulomb::effective_radius_particle(
    const double* radius_i, const double* radius_j, const double& gap, double& r_eff) const
{
  if (radius_j)
    r_eff =
        ((radius_i[0] + 0.5 * gap) * (radius_j[0] + 0.5 * gap)) / (radius_i[0] + radius_j[0] + gap);
  else
    r_eff = radius_i[0] + gap;
}

void ParticleInteraction::DEMContactRollingCoulomb::relative_rolling_velocity(const double& r_eff,
    const double* normal, const double* angvel_i, const double* angvel_j,
    double* v_rel_rolling) const
{
  Utils::vec_set_cross(v_rel_rolling, normal, angvel_i);
  if (angvel_j) Utils::vec_add_cross(v_rel_rolling, angvel_j, normal);

  Utils::vec_scale(v_rel_rolling, r_eff);
}

void ParticleInteraction::DEMContactRollingCoulomb::rolling_contact_moment(double* gap_rolling,
    bool& stick_rolling, const double* normal, const double* v_rel_rolling, const double& m_eff,
    const double& r_eff, const double& mu_rolling, const double& normalcontactforce,
    double* rollingcontactmoment) const
{
  // determine rolling contact damping parameter
  const double d_rolling = d_rolling_fac_ * std::sqrt(m_eff);

  // compute length of rolling gap at time n
  const double old_length = Utils::vec_norm_two(gap_rolling);

  // compute projection of rolling gap onto current normal at time n+1
  Utils::vec_add_scale(gap_rolling, -Utils::vec_dot(normal, gap_rolling), normal);

  // compute length of rolling gap at time n+1
  const double new_length = Utils::vec_norm_two(gap_rolling);

  // maintain length of rolling gap equal to before the projection
  if (new_length > 1.0e-14) Utils::vec_set_scale(gap_rolling, old_length / new_length, gap_rolling);

  // update of elastic rolling displacement if stick is true
  if (stick_rolling == true) Utils::vec_add_scale(gap_rolling, dt_, v_rel_rolling);

  // compute rolling contact force (assume stick-case)
  double rollingcontactforce[3];
  Utils::vec_set_scale(rollingcontactforce, -k_rolling_, gap_rolling);
  Utils::vec_add_scale(rollingcontactforce, -d_rolling, v_rel_rolling);

  // compute the norm of the rolling contact force
  const double norm_rollingcontactforce = Utils::vec_norm_two(rollingcontactforce);

  // rolling contact force for stick-case
  if (norm_rollingcontactforce <= (mu_rolling * std::abs(normalcontactforce)))
  {
    stick_rolling = true;

    // rolling contact force already computed
  }
  // rolling contact force for slip-case
  else
  {
    stick_rolling = false;

    // compute rolling contact force
    Utils::vec_set_scale(rollingcontactforce,
        mu_rolling * std::abs(normalcontactforce) / norm_rollingcontactforce, rollingcontactforce);

    // compute rolling displacement
    const double inv_k_rolling = 1.0 / k_rolling_;
    Utils::vec_set_scale(gap_rolling, -inv_k_rolling, rollingcontactforce);
    Utils::vec_add_scale(gap_rolling, -inv_k_rolling * d_rolling, v_rel_rolling);
  }

  // compute rolling contact moment
  Utils::vec_set_cross(rollingcontactmoment, rollingcontactforce, normal);
  Utils::vec_scale(rollingcontactmoment, r_eff);
}

void ParticleInteraction::DEMContactRollingCoulomb::rolling_potential_energy(
    const double* gap_rolling, double& rollingpotentialenergy) const
{
  rollingpotentialenergy = 0.5 * k_rolling_ * Utils::vec_dot(gap_rolling, gap_rolling);
}

FOUR_C_NAMESPACE_CLOSE
