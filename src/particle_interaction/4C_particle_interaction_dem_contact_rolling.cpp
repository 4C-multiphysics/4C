/*---------------------------------------------------------------------------*/
/*! \file
\brief rolling contact handler for discrete element method (DEM) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_particle_interaction_dem_contact_rolling.hpp"

#include "4C_inpar_particle.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::DEMContactRollingBase::DEMContactRollingBase(
    const Teuchos::ParameterList& params)
    : params_dem_(params),
      dt_(0.0),
      e_(params_dem_.get<double>("COEFF_RESTITUTION")),
      nue_(params_dem_.get<double>("POISSON_RATIO")),
      d_rolling_fac_(0.0)
{
  // empty constructor
}

void PARTICLEINTERACTION::DEMContactRollingBase::Init()
{
  // safety checks for contact parameters
  if (nue_ <= -1.0 or nue_ > 0.5)
    FOUR_C_THROW("invalid input parameter POISSON_RATIO (expected in range ]-1.0; 0.5])!");

  if (params_dem_.get<double>("FRICT_COEFF_ROLL") <= 0.0)
    FOUR_C_THROW("invalid input parameter FRICT_COEFF_ROLL for this kind of contact law!");
}

void PARTICLEINTERACTION::DEMContactRollingBase::Setup(const double& k_normal)
{
  // nothing to do
}

void PARTICLEINTERACTION::DEMContactRollingBase::SetCurrentStepSize(const double currentstepsize)
{
  dt_ = currentstepsize;
}

PARTICLEINTERACTION::DEMContactRollingViscous::DEMContactRollingViscous(
    const Teuchos::ParameterList& params)
    : PARTICLEINTERACTION::DEMContactRollingBase(params),
      young_(params_dem_.get<double>("YOUNG_MODULUS")),
      v_max_(params_dem_.get<double>("MAX_VELOCITY"))
{
  // empty constructor
}

void PARTICLEINTERACTION::DEMContactRollingViscous::Init()
{
  // call base class init
  DEMContactRollingBase::Init();

  // safety checks for contact parameters
  if (young_ <= 0.0)
    FOUR_C_THROW("invalid input parameter YOUNG_MODULUS (expected to be positive)!");

  if (v_max_ <= 0.0)
    FOUR_C_THROW("invalid input parameter MAX_VELOCITY (expected to be positive)!");
}

void PARTICLEINTERACTION::DEMContactRollingViscous::Setup(const double& k_normal)
{
  // call base class setup
  DEMContactRollingBase::Setup(k_normal);

  // determine rolling contact damping factor
  const double fac = young_ / (1.0 - UTILS::Pow<2>(nue_));
  const double c_1 = 1.15344;
  d_rolling_fac_ = (1.0 - e_) / (c_1 * std::pow(fac, 0.4) * std::pow(v_max_, 0.2));
}

void PARTICLEINTERACTION::DEMContactRollingViscous::effective_radius_particle(
    const double* radius_i, const double* radius_j, const double& gap, double& r_eff) const
{
  if (radius_j)
    r_eff = (radius_i[0] * radius_j[0]) / (radius_i[0] + radius_j[0]);
  else
    r_eff = radius_i[0];
}

void PARTICLEINTERACTION::DEMContactRollingViscous::relative_rolling_velocity(const double& r_eff,
    const double* normal, const double* angvel_i, const double* angvel_j,
    double* v_rel_rolling) const
{
  UTILS::VecSetCross(v_rel_rolling, angvel_i, normal);
  if (angvel_j) UTILS::VecAddCross(v_rel_rolling, normal, angvel_j);
}

void PARTICLEINTERACTION::DEMContactRollingViscous::rolling_contact_moment(double* gap_rolling,
    bool& stick_rolling, const double* normal, const double* v_rel_rolling, const double& m_eff,
    const double& r_eff, const double& mu_rolling, const double& normalcontactforce,
    double* rollingcontactmoment) const
{
  // determine rolling contact damping parameter
  const double d_rolling = d_rolling_fac_ * mu_rolling * std::pow(0.5 * r_eff, -0.2);

  // compute rolling contact force
  double rollingcontactforce[3];
  UTILS::VecSetScale(rollingcontactforce, -(d_rolling * normalcontactforce), v_rel_rolling);

  // compute rolling contact moment
  UTILS::VecSetCross(rollingcontactmoment, rollingcontactforce, normal);
  UTILS::VecScale(rollingcontactmoment, r_eff);
}

void PARTICLEINTERACTION::DEMContactRollingViscous::rolling_potential_energy(
    const double* gap_rolling, double& rollingpotentialenergy) const
{
  rollingpotentialenergy = 0.0;
}

PARTICLEINTERACTION::DEMContactRollingCoulomb::DEMContactRollingCoulomb(
    const Teuchos::ParameterList& params)
    : PARTICLEINTERACTION::DEMContactRollingBase(params), k_rolling_(0.0)
{
  // empty constructor
}

void PARTICLEINTERACTION::DEMContactRollingCoulomb::Setup(const double& k_normal)
{
  // call base class setup
  DEMContactRollingBase::Setup(k_normal);

  // rolling to normal stiffness ratio
  const double kappa = (1.0 - nue_) / (1.0 - 0.5 * nue_);

  // rolling contact stiffness
  k_rolling_ = kappa * k_normal;

  // determine rolling contact damping factor
  if (e_ > 0.0)
  {
    const double lne = std::log(e_);
    d_rolling_fac_ =
        2.0 * std::abs(lne) * std::sqrt(k_normal / (UTILS::Pow<2>(lne) + UTILS::Pow<2>(M_PI)));
  }
  else
    d_rolling_fac_ = 2.0 * std::sqrt(k_normal);
}

void PARTICLEINTERACTION::DEMContactRollingCoulomb::effective_radius_particle(
    const double* radius_i, const double* radius_j, const double& gap, double& r_eff) const
{
  if (radius_j)
    r_eff =
        ((radius_i[0] + 0.5 * gap) * (radius_j[0] + 0.5 * gap)) / (radius_i[0] + radius_j[0] + gap);
  else
    r_eff = radius_i[0] + gap;
}

void PARTICLEINTERACTION::DEMContactRollingCoulomb::relative_rolling_velocity(const double& r_eff,
    const double* normal, const double* angvel_i, const double* angvel_j,
    double* v_rel_rolling) const
{
  UTILS::VecSetCross(v_rel_rolling, normal, angvel_i);
  if (angvel_j) UTILS::VecAddCross(v_rel_rolling, angvel_j, normal);

  UTILS::VecScale(v_rel_rolling, r_eff);
}

void PARTICLEINTERACTION::DEMContactRollingCoulomb::rolling_contact_moment(double* gap_rolling,
    bool& stick_rolling, const double* normal, const double* v_rel_rolling, const double& m_eff,
    const double& r_eff, const double& mu_rolling, const double& normalcontactforce,
    double* rollingcontactmoment) const
{
  // determine rolling contact damping parameter
  const double d_rolling = d_rolling_fac_ * std::sqrt(m_eff);

  // compute length of rolling gap at time n
  const double old_length = UTILS::VecNormTwo(gap_rolling);

  // compute projection of rolling gap onto current normal at time n+1
  UTILS::VecAddScale(gap_rolling, -UTILS::VecDot(normal, gap_rolling), normal);

  // compute length of rolling gap at time n+1
  const double new_length = UTILS::VecNormTwo(gap_rolling);

  // maintain length of rolling gap equal to before the projection
  if (new_length > 1.0e-14) UTILS::VecSetScale(gap_rolling, old_length / new_length, gap_rolling);

  // update of elastic rolling displacement if stick is true
  if (stick_rolling == true) UTILS::VecAddScale(gap_rolling, dt_, v_rel_rolling);

  // compute rolling contact force (assume stick-case)
  double rollingcontactforce[3];
  UTILS::VecSetScale(rollingcontactforce, -k_rolling_, gap_rolling);
  UTILS::VecAddScale(rollingcontactforce, -d_rolling, v_rel_rolling);

  // compute the norm of the rolling contact force
  const double norm_rollingcontactforce = UTILS::VecNormTwo(rollingcontactforce);

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
    UTILS::VecSetScale(rollingcontactforce,
        mu_rolling * std::abs(normalcontactforce) / norm_rollingcontactforce, rollingcontactforce);

    // compute rolling displacement
    const double inv_k_rolling = 1.0 / k_rolling_;
    UTILS::VecSetScale(gap_rolling, -inv_k_rolling, rollingcontactforce);
    UTILS::VecAddScale(gap_rolling, -inv_k_rolling * d_rolling, v_rel_rolling);
  }

  // compute rolling contact moment
  UTILS::VecSetCross(rollingcontactmoment, rollingcontactforce, normal);
  UTILS::VecScale(rollingcontactmoment, r_eff);
}

void PARTICLEINTERACTION::DEMContactRollingCoulomb::rolling_potential_energy(
    const double* gap_rolling, double& rollingpotentialenergy) const
{
  rollingpotentialenergy = 0.5 * k_rolling_ * UTILS::VecDot(gap_rolling, gap_rolling);
}

FOUR_C_NAMESPACE_CLOSE
