// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_adhesion_law.hpp"

#include "4C_particle_input.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMAdhesionLawBase::DEMAdhesionLawBase(const Teuchos::ParameterList& params)
    : params_dem_(params),
      adhesion_surface_energy_factor_(params_dem_.get<double>("ADHESION_SURFACE_ENERGY_FACTOR")),
      adhesion_max_contact_pressure_(params_dem_.get<double>("ADHESION_MAX_CONTACT_PRESSURE")),
      adhesion_max_contact_force_(params_dem_.get<double>("ADHESION_MAX_CONTACT_FORCE")),
      adhesion_use_max_contact_force_(params_dem_.get<bool>("ADHESION_USE_MAX_CONTACT_FORCE")),
      adhesion_max_contact_force_fac_(0.0),
      adhesion_vdW_curve_shift_(params_dem_.get<bool>("ADHESION_VDW_CURVE_SHIFT")),
      inv_k_normal_(0.0)
{
  // empty constructor
}

void ParticleInteraction::DEMAdhesionLawBase::init()
{
  // nothing to do
}

void ParticleInteraction::DEMAdhesionLawBase::setup(const double& k_normal)
{
  // set inverse normal contact stiffness
  inv_k_normal_ = 1.0 / k_normal;

  // determine factor for calculation of maximum contact force using maximum contact pressure
  if (not adhesion_use_max_contact_force_)
  {
    // particle Young's modulus
    const double young = params_dem_.get<double>("YOUNG_MODULUS");

    // particle Poisson ratio
    const double nue = params_dem_.get<double>("POISSON_RATIO");

    // safety checks
    if (young <= 0.0) FOUR_C_THROW("invalid input parameter YOUNG_MODULUS (expected positive)!");

    if (nue <= -1.0 or nue > 0.5)
      FOUR_C_THROW("invalid input parameter POISSON_RATIO (expected in range ]-1.0; 0.5])!");

    // determine the effective Young's modulus
    const double young_eff = young / (2 * (1 - Utils::pow<2>(nue)));

    adhesion_max_contact_force_fac_ =
        Utils::pow<3>(std::numbers::pi * adhesion_max_contact_pressure_) /
        (6 * Utils::pow<2>(young_eff));

    // safety check
    if (adhesion_max_contact_pressure_ > 0.0)
      FOUR_C_THROW("positive adhesion maximum contact pressure!");
  }
  // use given maximum contact force
  else
  {
    // safety check
    if (adhesion_max_contact_force_ > 0.0) FOUR_C_THROW("positive adhesion maximum contact force!");
  }
}

ParticleInteraction::DEMAdhesionLawVdWDMT::DEMAdhesionLawVdWDMT(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMAdhesionLawBase(params),
      hamaker_constant_(params_dem_.get<double>("ADHESION_HAMAKER"))
{
  // empty constructor
}

void ParticleInteraction::DEMAdhesionLawVdWDMT::init()
{
  // call base class init
  DEMAdhesionLawBase::init();

  // safety check
  if (hamaker_constant_ <= 0.0) FOUR_C_THROW("negative hamaker constant!");
}

void ParticleInteraction::DEMAdhesionLawVdWDMT::adhesion_force(const double& gap,
    const double& surfaceenergy, const double& r_eff, const double& v_rel_normal,
    const double& m_eff, double& adhesionforce) const
{
  // determine the adhesion maximum contact force
  const double adhesioncontactforce = adhesion_use_max_contact_force_
                                          ? adhesion_max_contact_force_
                                          : adhesion_max_contact_force_fac_ * Utils::pow<2>(r_eff);

  // calculate gap where the maximum pull-off force is achieved
  const double gap_intersect_max = adhesioncontactforce * inv_k_normal_;

  // determine the pull-off force
  const double adhesionforce_pulloff_max = 4.0 * std::numbers::pi * r_eff * surfaceenergy;

  // determine the fraction of the pull-off force for particles in contact
  const double adhesionforce_pulloff = adhesion_surface_energy_factor_ * adhesionforce_pulloff_max;

  // determine linear slope of the adhesive force
  const double slope =
      (gap_intersect_max != 0.0)
          ? (adhesionforce_pulloff - adhesionforce_pulloff_max) / (-gap_intersect_max)
          : 0.0;

  // determine offset gap
  const double gap_offset = std::sqrt(hamaker_constant_ * r_eff / (6.0 * adhesionforce_pulloff));

  double gap_intersect_min = 0.0;
  if (not adhesion_vdW_curve_shift_)
  {
    if (slope == 0)
      gap_intersect_min = std::sqrt(hamaker_constant_ * r_eff / (6.0 * adhesionforce_pulloff));
    else
      calculate_intersection_gap(slope, adhesionforce_pulloff, 0.0,
          -(1.0 / 6.0) * hamaker_constant_ * r_eff, gap_intersect_min);
  }

  // compute adhesion force (assume deformation phase)
  double adhesionforce_temp = 0.0;

  if (gap < gap_intersect_max)
  {
    adhesionforce_temp = adhesionforce_pulloff_max;
  }
  else if (gap >= gap_intersect_max and gap < gap_intersect_min)
  {
    adhesionforce_temp = slope * gap + adhesionforce_pulloff;
  }
  else
  {
    if (adhesion_vdW_curve_shift_)
      adhesionforce_temp = hamaker_constant_ * r_eff / (6.0 * Utils::pow<2>(gap + gap_offset));
    else
      adhesionforce_temp = hamaker_constant_ * r_eff / (6.0 * Utils::pow<2>(gap));
  }

  if (adhesion_vdW_curve_shift_)
  {
    if (adhesionforce_temp > adhesionforce or adhesionforce_temp < adhesionforce_pulloff)
    {
      adhesionforce = adhesionforce_temp;
    }
    else
    {
      const double gap_max =
          std::sqrt(hamaker_constant_ * r_eff / (6.0 * adhesionforce)) - gap_offset;
      if (gap >= gap_max)
        adhesionforce = hamaker_constant_ * r_eff / (6.0 * Utils::pow<2>(gap + gap_offset));
    }
  }
  else
  {
    if (adhesionforce_temp > adhesionforce or
        (adhesionforce_temp < adhesionforce_pulloff and gap > gap_intersect_min))
    {
      adhesionforce = adhesionforce_temp;
    }
    else
    {
      const double gap_max = std::sqrt(hamaker_constant_ * r_eff / (6.0 * adhesionforce));
      if (gap >= gap_max) adhesionforce = hamaker_constant_ * r_eff / (6.0 * Utils::pow<2>(gap));
    }
  }
}

void ParticleInteraction::DEMAdhesionLawVdWDMT::calculate_intersection_gap(
    double a, double b, double c, double d, double& gap_intersect) const
{
  double x1 = 0.0;
  double x2 = 0.0;
  double x3 = 0.0;

  b /= a;
  c /= a;
  d /= a;

  double q = (3.0 * c - Utils::pow<2>(b)) / 9.0;
  double r = -(27.0 * d) + b * (9.0 * c - 2.0 * Utils::pow<2>(b));
  r /= 54.0;
  double disc = Utils::pow<3>(q) + Utils::pow<2>(r);
  double term1 = b / 3.0;
  double r13 = 0.0;

  if (disc > 0.0)
    FOUR_C_THROW("The combination of these input parameters leads to an unreasonable result!");

  // All roots real, at least two are equal
  if (disc == 0.0)
  {
    r13 = (r < 0.0) ? -std::pow(-r, 1.0 / 3.0) : std::pow(r, 1.0 / 3.0);
    x1 = -term1 + 2.0 * r13;
    x2 = -(r13 + term1);
    x3 = -(r13 + term1);

    if (x1 > x2)
      gap_intersect = x1;
    else
      gap_intersect = x2;
  }
  else
  {
    q = -q;
    double dummy = Utils::pow<3>(q);
    dummy = std::acos(r / std::sqrt(dummy));
    r13 = 2.0 * std::sqrt(q);
    x1 = -term1 + r13 * std::cos(dummy / 3.0);
    x2 = -term1 + r13 * std::cos((dummy + 2.0 * std::numbers::pi) / 3.0);
    x3 = -term1 + r13 * std::cos((dummy + 4.0 * std::numbers::pi) / 3.0);

    if (x1 > x2)
    {
      if (x3 > x1)
        gap_intersect = x1;
      else
      {
        if (x3 > x2)
          gap_intersect = x3;
        else
          gap_intersect = x2;
      }
    }
    else
    {
      if (x3 > x2)
        gap_intersect = x2;
      else
      {
        if (x3 > x1)
          gap_intersect = x3;
        else
          gap_intersect = x1;
      }
    }
  }
}

ParticleInteraction::DEMAdhesionLawRegDMT::DEMAdhesionLawRegDMT(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMAdhesionLawBase(params),
      adhesion_distance_(params_dem_.get<double>("ADHESION_DISTANCE"))
{
  // empty constructor
}

void ParticleInteraction::DEMAdhesionLawRegDMT::adhesion_force(const double& gap,
    const double& surfaceenergy, const double& r_eff, const double& v_rel_normal,
    const double& m_eff, double& adhesionforce) const
{
  // determine the adhesion maximum contact force
  const double adhesioncontactforce = adhesion_use_max_contact_force_
                                          ? adhesion_max_contact_force_
                                          : adhesion_max_contact_force_fac_ * Utils::pow<2>(r_eff);

  // calculate gap where the maximum pull-off force is achieved
  const double gap_intersect_max = adhesioncontactforce * inv_k_normal_;

  // determine the pull-off force
  const double adhesionforce_pulloff_max = 4.0 * std::numbers::pi * r_eff * surfaceenergy;

  // determine the fraction of the pull-off force for particles in contact
  const double adhesionforce_pulloff = adhesion_surface_energy_factor_ * adhesionforce_pulloff_max;

  // determine linear slope of the adhesive force
  const double slope =
      (gap_intersect_max != 0.0)
          ? (adhesionforce_pulloff - adhesionforce_pulloff_max) / (-gap_intersect_max)
          : 0.0;

  // regularization with adhesion distance
  const double gap_reg = adhesion_distance_;

  double gap_intersect_min = 0.0;
  if (not adhesion_vdW_curve_shift_)
  {
    gap_intersect_min = (adhesionforce_pulloff_max - adhesionforce_pulloff) /
                        (slope + adhesionforce_pulloff_max / gap_reg);

    // safety check
    if (gap_intersect_min < 0.0)
      FOUR_C_THROW("the combination of these input parameters leads to an unreasonable result!");
  }

  // compute adhesion force (assume deformation phase)
  double adhesionforce_temp = 0.0;

  if (gap < gap_intersect_max)
  {
    adhesionforce_temp = adhesionforce_pulloff_max;
  }
  else if (gap >= gap_intersect_max and gap < gap_intersect_min)
  {
    adhesionforce_temp = slope * gap + adhesionforce_pulloff;
  }
  else if (gap >= gap_intersect_min and gap < gap_reg)
  {
    if (adhesion_vdW_curve_shift_)
      adhesionforce_temp = adhesionforce_pulloff * (1.0 - gap / gap_reg);
    else
      adhesionforce_temp = adhesionforce_pulloff_max * (1.0 - gap / gap_reg);
  }

  if (adhesion_vdW_curve_shift_)
  {
    if (adhesionforce_temp > adhesionforce or adhesionforce_temp < adhesionforce_pulloff)
    {
      adhesionforce = adhesionforce_temp;
    }
    else
    {
      const double gap_max = (1.0 - adhesionforce / adhesionforce_pulloff) * gap_reg;
      if (gap > gap_max) adhesionforce = adhesionforce_pulloff * (1.0 - gap / gap_reg);
    }
  }
  else
  {
    if (adhesionforce_temp > adhesionforce or
        (adhesionforce_temp < adhesionforce_pulloff and gap > gap_intersect_min))
    {
      adhesionforce = adhesionforce_temp;
    }
    else
    {
      const double gap_max = (1.0 - adhesionforce / adhesionforce_pulloff_max) * gap_reg;
      if (gap > gap_max) adhesionforce = adhesionforce_pulloff_max * (1.0 - gap / gap_reg);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
