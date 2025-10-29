// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_SURFACE_ENERGY_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_SURFACE_ENERGY_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class DEMAdhesionSurfaceEnergyBase
  {
   public:
    //! constructor
    explicit DEMAdhesionSurfaceEnergyBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~DEMAdhesionSurfaceEnergyBase() = default;

    //! init adhesion surface energy handler
    virtual void init();

    //! setup adhesion surface energy handler
    virtual void setup();

    //! calculate adhesion surface energy
    virtual void adhesion_surface_energy(
        const double& mean_surface_energy, double& surface_energy) const = 0;

   protected:
    //! discrete element method parameter list
    const Teuchos::ParameterList& params_dem_;
  };

  class DEMAdhesionSurfaceEnergyConstant : public DEMAdhesionSurfaceEnergyBase
  {
   public:
    //! constructor
    explicit DEMAdhesionSurfaceEnergyConstant(const Teuchos::ParameterList& params);

    //! get adhesion surface energy
    void adhesion_surface_energy(
        const double& mean_surface_energy, double& surface_energy) const override
    {
      surface_energy = mean_surface_energy;
    };
  };

  class DEMAdhesionSurfaceEnergyDistributionBase : public DEMAdhesionSurfaceEnergyBase
  {
   public:
    //! constructor
    explicit DEMAdhesionSurfaceEnergyDistributionBase(const Teuchos::ParameterList& params);

    //! setup adhesion surface energy handler
    void setup() override;

    //! get adhesion surface energy
    void adhesion_surface_energy(
        const double& mean_surface_energy, double& surface_energy) const override = 0;

   protected:
    //! adjust surface energy to allowed bounds
    void adjust_surface_energy_to_allowed_bounds(
        const double& mean_surface_energy, double& surface_energy) const;

    //! variance of adhesion surface energy distribution
    const double variance_;

    //! cutoff factor of adhesion surface energy to determine minimum and maximum value
    const double cutofffactor_;
  };

  class DEMAdhesionSurfaceEnergyDistributionNormal : public DEMAdhesionSurfaceEnergyDistributionBase
  {
   public:
    //! constructor
    explicit DEMAdhesionSurfaceEnergyDistributionNormal(const Teuchos::ParameterList& params);

    //! get adhesion surface energy
    void adhesion_surface_energy(
        const double& mean_surface_energy, double& surface_energy) const override;
  };

  class DEMAdhesionSurfaceEnergyDistributionLogNormal
      : public DEMAdhesionSurfaceEnergyDistributionBase
  {
   public:
    //! constructor
    explicit DEMAdhesionSurfaceEnergyDistributionLogNormal(const Teuchos::ParameterList& params);

    //! get adhesion surface energy
    void adhesion_surface_energy(
        const double& mean_surface_energy, double& surface_energy) const override;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
