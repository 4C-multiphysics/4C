// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_kernel.hpp"

#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <numbers>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHKernelBase::SPHKernelBase(const Teuchos::ParameterList& params)
    : kernelspacedim_(
          Teuchos::getIntegralValue<Particle::KernelSpaceDimension>(params, "KERNEL_SPACE_DIM"))
{
  // empty constructor
}

int Particle::SPHKernelBase::kernel_space_dimension() const
{
  switch (kernelspacedim_)
  {
    case Particle::KernelSpaceDimension::Kernel1D:
      return 1;

    case Particle::KernelSpaceDimension::Kernel2D:
      return 2;

    case Particle::KernelSpaceDimension::Kernel3D:
      return 3;

    default:
      FOUR_C_THROW("unknown kernel space dimension!");
  }

  std23::unreachable();
}

void Particle::SPHKernelBase::grad_wij(
    const double& rij, const double& support, const double* eij, double* gradWij) const
{
  ParticleUtils::vec_set_scale(gradWij, this->d_wdrij(rij, support), eij);
}

Particle::SPHKernelCubicSpline::SPHKernelCubicSpline(const Teuchos::ParameterList& params)
    : Particle::SPHKernelBase(params)
{
  // empty constructor
}

double Particle::SPHKernelCubicSpline::smoothing_length(const double& support) const
{
  return (0.5 * support);
}

double Particle::SPHKernelCubicSpline::normalization_constant(const double& inv_h) const
{
  switch (kernelspacedim_)
  {
    case Particle::KernelSpaceDimension::Kernel1D:
    {
      // (2.0 / 3.0) * inv_h
      return 0.6666666666666666 * inv_h;
    }
    case Particle::KernelSpaceDimension::Kernel2D:
    {
      // (10.0 / 7.0) * std::numbers::inv_pi * inv_h * inv_h
      return 0.4547284088339866 * ParticleUtils::pow<2>(inv_h);
    }
    case Particle::KernelSpaceDimension::Kernel3D:
    {
      return std::numbers::inv_pi * ParticleUtils::pow<3>(inv_h);
    }
    default:
    {
      FOUR_C_THROW("unknown kernel space dimension!");
      break;
    }
  }

  return 0.0;
}

double Particle::SPHKernelCubicSpline::w0(const double& support) const
{
  return normalization_constant(2.0 / support);
}

double Particle::SPHKernelCubicSpline::w(const double& rij, const double& support) const
{
  const double inv_h = 2.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (1.0 - 1.5 * ParticleUtils::pow<2>(q) + 0.75 * ParticleUtils::pow<3>(q)) *
           normalization_constant(inv_h);
  else if (q < 2.0)
    return (0.25 * ParticleUtils::pow<3>(2.0 - q)) * normalization_constant(inv_h);
  else
    return 0.0;
}

double Particle::SPHKernelCubicSpline::d_wdrij(const double& rij, const double& support) const
{
  const double inv_h = 2.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (-3.0 * q + 2.25 * ParticleUtils::pow<2>(q)) * inv_h * normalization_constant(inv_h);
  else if (q < 2.0)
    return (-0.75 * ParticleUtils::pow<2>(2.0 - q)) * inv_h * normalization_constant(inv_h);
  else
    return 0.0;
}

double Particle::SPHKernelCubicSpline::d2_wdrij2(const double& rij, const double& support) const
{
  const double inv_h = 2.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (-3.0 + 4.5 * q) * ParticleUtils::pow<2>(inv_h) * normalization_constant(inv_h);
  else if (q < 2.0)
    return (1.5 * (2.0 - q)) * ParticleUtils::pow<2>(inv_h) * normalization_constant(inv_h);
  else
    return 0.0;
}

Particle::SPHKernelQuinticSpline::SPHKernelQuinticSpline(const Teuchos::ParameterList& params)
    : Particle::SPHKernelBase(params)
{
  // empty constructor
}

double Particle::SPHKernelQuinticSpline::smoothing_length(const double& support) const
{
  // (support / 3.0)
  return 0.3333333333333333 * support;
}

double Particle::SPHKernelQuinticSpline::normalization_constant(const double& inv_h) const
{
  switch (kernelspacedim_)
  {
    case Particle::KernelSpaceDimension::Kernel1D:
    {
      // (inv_h / 120.0)
      return 0.0083333333333333 * inv_h;
    }
    case Particle::KernelSpaceDimension::Kernel2D:
    {
      // (7.0 / 478.0) * std::numbers::inv_pi * inv_h * inv_h
      return 0.0046614418478797 * ParticleUtils::pow<2>(inv_h);
    }
    case Particle::KernelSpaceDimension::Kernel3D:
    {
      // (3.0 / 359.0) * std::numbers::inv_pi * inv_h * inv_h * inv_h
      return 0.0026599711937364 * ParticleUtils::pow<3>(inv_h);
    }
    default:
    {
      FOUR_C_THROW("unknown kernel space dimension!");
      break;
    }
  }

  return 0.0;
}

double Particle::SPHKernelQuinticSpline::w0(const double& support) const
{
  return 66.0 * normalization_constant(3.0 / support);
}

double Particle::SPHKernelQuinticSpline::w(const double& rij, const double& support) const
{
  const double inv_h = 3.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (ParticleUtils::pow<5>(3.0 - q) - 6.0 * ParticleUtils::pow<5>(2.0 - q) +
               15.0 * ParticleUtils::pow<5>(1.0 - q)) *
           normalization_constant(inv_h);
  else if (q < 2.0)
    return (ParticleUtils::pow<5>(3.0 - q) - 6.0 * ParticleUtils::pow<5>(2.0 - q)) *
           normalization_constant(inv_h);
  else if (q < 3.0)
    return ParticleUtils::pow<5>(3.0 - q) * normalization_constant(inv_h);
  else
    return 0.0;
}

double Particle::SPHKernelQuinticSpline::d_wdrij(const double& rij, const double& support) const
{
  const double inv_h = 3.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (-5.0 * ParticleUtils::pow<4>(3.0 - q) + 30.0 * ParticleUtils::pow<4>(2.0 - q) -
               75.0 * ParticleUtils::pow<4>(1.0 - q)) *
           inv_h * normalization_constant(inv_h);
  else if (q < 2.0)
    return (-5.0 * ParticleUtils::pow<4>(3.0 - q) + 30.0 * ParticleUtils::pow<4>(2.0 - q)) * inv_h *
           normalization_constant(inv_h);
  else if (q < 3.0)
    return (-5.0 * ParticleUtils::pow<4>(3.0 - q)) * inv_h * normalization_constant(inv_h);
  else
    return 0.0;
}

double Particle::SPHKernelQuinticSpline::d2_wdrij2(const double& rij, const double& support) const
{
  const double inv_h = 3.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (20.0 * ParticleUtils::pow<3>(3.0 - q) - 120.0 * ParticleUtils::pow<3>(2.0 - q) +
               300.0 * ParticleUtils::pow<3>(1.0 - q)) *
           ParticleUtils::pow<2>(inv_h) * normalization_constant(inv_h);
  else if (q < 2.0)
    return (20.0 * ParticleUtils::pow<3>(3.0 - q) - 120.0 * ParticleUtils::pow<3>(2.0 - q)) *
           ParticleUtils::pow<2>(inv_h) * normalization_constant(inv_h);
  else if (q < 3.0)
    return (20.0 * ParticleUtils::pow<3>(3.0 - q)) * ParticleUtils::pow<2>(inv_h) *
           normalization_constant(inv_h);
  else
    return 0.0;
}

FOUR_C_NAMESPACE_CLOSE
