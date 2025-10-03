// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_contact_normal.hpp"

#include "4C_particle_input.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMContactNormalBase::DEMContactNormalBase(
    const Teuchos::ParameterList& params)
    : params_dem_(params),
      r_max_(params_dem_.get<double>("MAX_RADIUS")),
      v_max_(params_dem_.get<double>("MAX_VELOCITY")),
      c_(params_dem_.get<double>("REL_PENETRATION")),
      k_normal_(params_dem_.get<double>("NORMAL_STIFF")),
      k_normal_crit_(0.0)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalBase::init()
{
  if (not((c_ <= 0.0 and k_normal_ > 0.0) or (c_ > 0.0 and v_max_ > 0.0 and k_normal_ <= 0.0)))
    FOUR_C_THROW(
        "specify either the relative penetration along with the maximum velocity, or the normal "
        "stiffness, but neither both nor none of them!");
}

void ParticleInteraction::DEMContactNormalBase::setup(const double& dens_max)
{
  // nothing to do
}

ParticleInteraction::DEMContactNormalLinearSpring::DEMContactNormalLinearSpring(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalLinearSpring::setup(const double& dens_max)
{
  // call base class setup
  DEMContactNormalBase::setup(dens_max);

  // calculate normal stiffness from relative penetration and other input parameters
  if (c_ > 0.0)
    k_normal_ = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max * Utils::pow<2>(v_max_) /
                Utils::pow<2>(c_);

  // set critical normal contact stiffness to linear normal contact stiffness
  k_normal_crit_ = k_normal_;
}

void ParticleInteraction::DEMContactNormalLinearSpring::normal_contact_force(const double& gap,
    const double* radius_i, const double* radius_j, const double& v_rel_normal, const double& m_eff,
    double& normalcontactforce) const
{
  normalcontactforce = k_normal_ * gap;
}

void ParticleInteraction::DEMContactNormalLinearSpring::normal_potential_energy(
    const double& gap, double& normalpotentialenergy) const
{
  normalpotentialenergy = 0.5 * k_normal_ * Utils::pow<2>(gap);
}

ParticleInteraction::DEMContactNormalLinearSpringDamp::DEMContactNormalLinearSpringDamp(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalLinearSpring(params),
      e_(params_dem_.get<double>("COEFF_RESTITUTION")),
      damp_reg_fac_(params_dem_.get<double>("DAMP_REG_FAC")),
      d_normal_fac_(0.0)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalLinearSpringDamp::init()
{
  // call base class init
  DEMContactNormalLinearSpring::init();

  // safety checks for contact parameters
  if (e_ < 0.0)
    FOUR_C_THROW("invalid input parameter COEFF_RESTITUTION for this kind of contact law!");
}

void ParticleInteraction::DEMContactNormalLinearSpringDamp::setup(const double& dens_max)
{
  // call base class setup
  DEMContactNormalLinearSpring::setup(dens_max);

  // determine normal contact damping factor
  if (e_ > 0.0)
  {
    const double lne = std::log(e_);
    d_normal_fac_ = 2.0 * std::abs(lne) *
                    std::sqrt(k_normal_ / (Utils::pow<2>(lne) + Utils::pow<2>(std::numbers::pi)));
  }
  else
    d_normal_fac_ = 2.0 * std::sqrt(k_normal_);
}

void ParticleInteraction::DEMContactNormalLinearSpringDamp::normal_contact_force(const double& gap,
    const double* radius_i, const double* radius_j, const double& v_rel_normal, const double& m_eff,
    double& normalcontactforce) const
{
  // determine normal contact damping parameter
  double d_normal = d_normal_fac_ * std::sqrt(m_eff);

  // linear regularization of damping force to reach full amplitude for g = damp_reg_fac_ * r_min
  double reg_fac = 1.0;
  if (damp_reg_fac_ > 0.0)
  {
    double rad_min = std::min(radius_i[0], radius_j[0]);

    if (std::abs(gap) < damp_reg_fac_ * rad_min)
      reg_fac = (std::abs(gap) / (damp_reg_fac_ * rad_min));
  }

  // evaluate normal contact force
  normalcontactforce = (k_normal_ * gap) - (d_normal * v_rel_normal * reg_fac);
}

ParticleInteraction::DEMContactNormalNonlinearBase::DEMContactNormalNonlinearBase(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalNonlinearBase::setup(const double& dens_max)
{
  // call base class setup
  DEMContactNormalBase::setup(dens_max);

  // calculate normal stiffness from relative penetration and other input parameters if necessary
  if (c_ > 0.0)
    k_normal_ = 10.0 / 3.0 * std::numbers::pi * dens_max * Utils::pow<2>(v_max_) *
                std::sqrt(r_max_) / std::sqrt(Utils::pow<5>(2.0 * c_));

  // set critical normal contact stiffness to linear normal contact stiffness
  if (c_ > 0.0)
    k_normal_crit_ = 2.0 / 3.0 * r_max_ * std::numbers::pi * dens_max * Utils::pow<2>(v_max_) /
                     Utils::pow<2>(c_);
  else
    k_normal_crit_ =
        std::pow(2048.0 / 1875.0 * dens_max * Utils::pow<2>(v_max_) * std::numbers::pi *
                     Utils::pow<3>(r_max_) * Utils::pow<4>(k_normal_),
            0.2);
}

void ParticleInteraction::DEMContactNormalNonlinearBase::normal_potential_energy(
    const double& gap, double& normalpotentialenergy) const
{
  normalpotentialenergy = 0.4 * k_normal_ * Utils::pow<2>(gap) * std::sqrt(-gap);
}

ParticleInteraction::DEMContactNormalHertz::DEMContactNormalHertz(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalNonlinearBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalHertz::normal_contact_force(const double& gap,
    const double* radius_i, const double* radius_j, const double& v_rel_normal, const double& m_eff,
    double& normalcontactforce) const
{
  normalcontactforce = -k_normal_ * (-gap) * std::sqrt(-gap);
}

ParticleInteraction::DEMContactNormalNonlinearDampBase::DEMContactNormalNonlinearDampBase(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalNonlinearBase(params),
      d_normal_(params_dem_.get<double>("NORMAL_DAMP"))
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalNonlinearDampBase::init()
{
  // call base class init
  DEMContactNormalNonlinearBase::init();

  // safety checks for contact parameters
  if (d_normal_ < 0.0)
    FOUR_C_THROW("invalid input parameter NORMAL_DAMP for this kind of contact law!");
}

ParticleInteraction::DEMContactNormalLeeHerrmann::DEMContactNormalLeeHerrmann(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalNonlinearDampBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalLeeHerrmann::normal_contact_force(const double& gap,
    const double* radius_i, const double* radius_j, const double& v_rel_normal, const double& m_eff,
    double& normalcontactforce) const
{
  normalcontactforce = -k_normal_ * (-gap) * std::sqrt(-gap) - m_eff * d_normal_ * v_rel_normal;
}

ParticleInteraction::DEMContactNormalKuwabaraKono::DEMContactNormalKuwabaraKono(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalNonlinearDampBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalKuwabaraKono::normal_contact_force(const double& gap,
    const double* radius_i, const double* radius_j, const double& v_rel_normal, const double& m_eff,
    double& normalcontactforce) const
{
  normalcontactforce =
      -k_normal_ * (-gap) * std::sqrt(-gap) - d_normal_ * v_rel_normal * std::sqrt(-gap);
}

ParticleInteraction::DEMContactNormalTsuji::DEMContactNormalTsuji(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMContactNormalNonlinearDampBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMContactNormalTsuji::normal_contact_force(const double& gap,
    const double* radius_i, const double* radius_j, const double& v_rel_normal, const double& m_eff,
    double& normalcontactforce) const
{
  normalcontactforce =
      -k_normal_ * (-gap) * std::sqrt(-gap) - d_normal_ * v_rel_normal * std::sqrt(std::sqrt(-gap));
}

FOUR_C_NAMESPACE_CLOSE
