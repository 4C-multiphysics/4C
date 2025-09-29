// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_DENSITY_CORRECTION_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_DENSITY_CORRECTION_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHDensityCorrectionBase
  {
   public:
    //! constructor
    explicit SPHDensityCorrectionBase();

    //! virtual destructor
    virtual ~SPHDensityCorrectionBase() = default;

    //! init density correction handler
    virtual void init();

    //! setup density correction handler
    virtual void setup();

    //! density boundary condition is needed
    virtual bool compute_density_bc() const = 0;

    //! set corrected density of interior particles
    virtual void corrected_density_interior(const double* denssum, double* dens) const;

    //! set corrected density of free surface particles
    virtual void corrected_density_free_surface(const double* denssum, const double* colorfield,
        const double* dens_bc, double* dens) const = 0;
  };

  class SPHDensityCorrectionInterior : public SPHDensityCorrectionBase
  {
   public:
    //! constructor
    explicit SPHDensityCorrectionInterior();

    //! density boundary condition is needed
    bool compute_density_bc() const override;

    //! set corrected density of free surface particles
    void corrected_density_free_surface(const double* denssum, const double* colorfield,
        const double* dens_bc, double* dens) const override;
  };

  class SPHDensityCorrectionNormalized : public SPHDensityCorrectionBase
  {
   public:
    //! constructor
    explicit SPHDensityCorrectionNormalized();

    //! density boundary condition is needed
    bool compute_density_bc() const override;

    //! set corrected density of free surface particles
    void corrected_density_free_surface(const double* denssum, const double* colorfield,
        const double* dens_bc, double* dens) const override;
  };

  class SPHDensityCorrectionRandles : public SPHDensityCorrectionBase
  {
   public:
    //! constructor
    explicit SPHDensityCorrectionRandles();

    //! density boundary condition is needed
    bool compute_density_bc() const override;

    //! set corrected density of free surface particles
    void corrected_density_free_surface(const double* denssum, const double* colorfield,
        const double* dens_bc, double* dens) const override;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
