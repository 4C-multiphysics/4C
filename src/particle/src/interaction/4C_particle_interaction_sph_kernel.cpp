// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_kernel.hpp"

#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::SPHKernelBase::SPHKernelBase(const Teuchos::ParameterList& params)
    : kernelspacedim_(
          Teuchos::getIntegralValue<PARTICLE::KernelSpaceDimension>(params, "KERNEL_SPACE_DIM"))
{
  // empty constructor
}

void ParticleInteraction::SPHKernelBase::init()
{
  // nothing to do
}

void ParticleInteraction::SPHKernelBase::setup()
{
  // nothing to do
}

void ParticleInteraction::SPHKernelBase::kernel_space_dimension(int& dim) const
{
  switch (kernelspacedim_)
  {
    case PARTICLE::Kernel1D:
    {
      dim = 1;
      break;
    }
    case PARTICLE::Kernel2D:
    {
      dim = 2;
      break;
    }
    case PARTICLE::Kernel3D:
    {
      dim = 3;
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown kernel space dimension!");
      break;
    }
  }
}

void ParticleInteraction::SPHKernelBase::grad_wij(
    const double& rij, const double& support, const double* eij, double* gradWij) const
{
  Utils::vec_set_scale(gradWij, this->d_wdrij(rij, support), eij);
}

ParticleInteraction::SPHKernelCubicSpline::SPHKernelCubicSpline(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::SPHKernelBase(params)
{
  // empty constructor
}

double ParticleInteraction::SPHKernelCubicSpline::smoothing_length(const double& support) const
{
  return (0.5 * support);
}

double ParticleInteraction::SPHKernelCubicSpline::normalization_constant(const double& inv_h) const
{
  switch (kernelspacedim_)
  {
    case PARTICLE::Kernel1D:
    {
      // (2.0 / 3.0) * inv_h
      return 0.6666666666666666 * inv_h;
    }
    case PARTICLE::Kernel2D:
    {
      // (10.0 / 7.0) * std::numbers::inv_pi * inv_h * inv_h
      return 0.4547284088339866 * Utils::pow<2>(inv_h);
    }
    case PARTICLE::Kernel3D:
    {
      return std::numbers::inv_pi * Utils::pow<3>(inv_h);
    }
    default:
    {
      FOUR_C_THROW("unknown kernel space dimension!");
      break;
    }
  }

  return 0.0;
}

double ParticleInteraction::SPHKernelCubicSpline::w0(const double& support) const
{
  return normalization_constant(2.0 / support);
}

double ParticleInteraction::SPHKernelCubicSpline::w(const double& rij, const double& support) const
{
  const double inv_h = 2.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (1.0 - 1.5 * Utils::pow<2>(q) + 0.75 * Utils::pow<3>(q)) * normalization_constant(inv_h);
  else if (q < 2.0)
    return (0.25 * Utils::pow<3>(2.0 - q)) * normalization_constant(inv_h);
  else
    return 0.0;
}

double ParticleInteraction::SPHKernelCubicSpline::d_wdrij(
    const double& rij, const double& support) const
{
  const double inv_h = 2.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (-3.0 * q + 2.25 * Utils::pow<2>(q)) * inv_h * normalization_constant(inv_h);
  else if (q < 2.0)
    return (-0.75 * Utils::pow<2>(2.0 - q)) * inv_h * normalization_constant(inv_h);
  else
    return 0.0;
}

double ParticleInteraction::SPHKernelCubicSpline::d2_wdrij2(
    const double& rij, const double& support) const
{
  const double inv_h = 2.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (-3.0 + 4.5 * q) * Utils::pow<2>(inv_h) * normalization_constant(inv_h);
  else if (q < 2.0)
    return (1.5 * (2.0 - q)) * Utils::pow<2>(inv_h) * normalization_constant(inv_h);
  else
    return 0.0;
}

ParticleInteraction::SPHKernelQuinticSpline::SPHKernelQuinticSpline(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::SPHKernelBase(params)
{
  // empty constructor
}

double ParticleInteraction::SPHKernelQuinticSpline::smoothing_length(const double& support) const
{
  // (support / 3.0)
  return 0.3333333333333333 * support;
}

double ParticleInteraction::SPHKernelQuinticSpline::normalization_constant(
    const double& inv_h) const
{
  switch (kernelspacedim_)
  {
    case PARTICLE::Kernel1D:
    {
      // (inv_h / 120.0)
      return 0.0083333333333333 * inv_h;
    }
    case PARTICLE::Kernel2D:
    {
      // (7.0 / 478.0) * std::numbers::inv_pi * inv_h * inv_h
      return 0.0046614418478797 * Utils::pow<2>(inv_h);
    }
    case PARTICLE::Kernel3D:
    {
      // (3.0 / 359.0) * std::numbers::inv_pi * inv_h * inv_h * inv_h
      return 0.0026599711937364 * Utils::pow<3>(inv_h);
    }
    default:
    {
      FOUR_C_THROW("unknown kernel space dimension!");
      break;
    }
  }

  return 0.0;
}

double ParticleInteraction::SPHKernelQuinticSpline::w0(const double& support) const
{
  return 66.0 * normalization_constant(3.0 / support);
}

double ParticleInteraction::SPHKernelQuinticSpline::w(
    const double& rij, const double& support) const
{
  const double inv_h = 3.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (Utils::pow<5>(3.0 - q) - 6.0 * Utils::pow<5>(2.0 - q) + 15.0 * Utils::pow<5>(1.0 - q)) *
           normalization_constant(inv_h);
  else if (q < 2.0)
    return (Utils::pow<5>(3.0 - q) - 6.0 * Utils::pow<5>(2.0 - q)) * normalization_constant(inv_h);
  else if (q < 3.0)
    return Utils::pow<5>(3.0 - q) * normalization_constant(inv_h);
  else
    return 0.0;
}

double ParticleInteraction::SPHKernelQuinticSpline::d_wdrij(
    const double& rij, const double& support) const
{
  const double inv_h = 3.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (-5.0 * Utils::pow<4>(3.0 - q) + 30.0 * Utils::pow<4>(2.0 - q) -
               75.0 * Utils::pow<4>(1.0 - q)) *
           inv_h * normalization_constant(inv_h);
  else if (q < 2.0)
    return (-5.0 * Utils::pow<4>(3.0 - q) + 30.0 * Utils::pow<4>(2.0 - q)) * inv_h *
           normalization_constant(inv_h);
  else if (q < 3.0)
    return (-5.0 * Utils::pow<4>(3.0 - q)) * inv_h * normalization_constant(inv_h);
  else
    return 0.0;
}

double ParticleInteraction::SPHKernelQuinticSpline::d2_wdrij2(
    const double& rij, const double& support) const
{
  const double inv_h = 3.0 / support;
  const double q = rij * inv_h;

  if (q < 1.0)
    return (20.0 * Utils::pow<3>(3.0 - q) - 120.0 * Utils::pow<3>(2.0 - q) +
               300.0 * Utils::pow<3>(1.0 - q)) *
           Utils::pow<2>(inv_h) * normalization_constant(inv_h);
  else if (q < 2.0)
    return (20.0 * Utils::pow<3>(3.0 - q) - 120.0 * Utils::pow<3>(2.0 - q)) * Utils::pow<2>(inv_h) *
           normalization_constant(inv_h);
  else if (q < 3.0)
    return (20.0 * Utils::pow<3>(3.0 - q)) * Utils::pow<2>(inv_h) * normalization_constant(inv_h);
  else
    return 0.0;
}

FOUR_C_NAMESPACE_CLOSE
