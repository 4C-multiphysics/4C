// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_KERNEL_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_KERNEL_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHKernelBase
  {
   public:
    //! constructor
    explicit SPHKernelBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHKernelBase() = default;

    //! init kernel handler
    virtual void init();

    //! setup kernel handler
    virtual void setup();

    //! get kernel space dimension
    virtual void kernel_space_dimension(int& dim) const final;

    //! get smoothing length from kernel support radius
    virtual double smoothing_length(const double& support) const = 0;

    //! get normalization constant from inverse smoothing length
    virtual double normalization_constant(const double& inv_h) const = 0;

    //! evaluate kernel (self-interaction)
    virtual double w0(const double& support) const = 0;

    //! evaluate kernel
    virtual double w(const double& rij, const double& support) const = 0;

    //! evaluate first derivative of kernel
    virtual double d_wdrij(const double& rij, const double& support) const = 0;

    //! evaluate second derivative of kernel
    virtual double d2_wdrij2(const double& rij, const double& support) const = 0;

    //! evaluate gradient of kernel
    virtual void grad_wij(
        const double& rij, const double& support, const double* eij, double* gradWij) const final;

   protected:
    // store problem dimension required by weight functions
    PARTICLE::KernelSpaceDimension kernelspacedim_;
  };

  class SPHKernelCubicSpline final : public SPHKernelBase
  {
   public:
    //! constructor
    explicit SPHKernelCubicSpline(const Teuchos::ParameterList& params);

    //! get smoothing length from kernel support radius
    double smoothing_length(const double& support) const override;

    //! get normalization constant from inverse smoothing length
    double normalization_constant(const double& inv_h) const override;

    //! evaluate kernel (self-interaction)
    double w0(const double& support) const override;

    //! evaluate kernel
    double w(const double& rij, const double& support) const override;

    //! evaluate first derivative of kernel
    double d_wdrij(const double& rij, const double& support) const override;

    //! evaluate second derivative of kernel
    double d2_wdrij2(const double& rij, const double& support) const override;
  };

  class SPHKernelQuinticSpline final : public SPHKernelBase
  {
   public:
    //! constructor
    explicit SPHKernelQuinticSpline(const Teuchos::ParameterList& params);

    //! get smoothing length from kernel support radius
    double smoothing_length(const double& support) const override;

    //! get normalization constant from inverse smoothing length
    double normalization_constant(const double& inv_h) const override;

    //! evaluate kernel (self-interaction)
    double w0(const double& support) const override;

    //! evaluate kernel
    double w(const double& rij, const double& support) const override;

    //! evaluate first derivative of kernel
    double d_wdrij(const double& rij, const double& support) const override;

    //! evaluate second derivative of kernel
    double d2_wdrij2(const double& rij, const double& support) const override;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
