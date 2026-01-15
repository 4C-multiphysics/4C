// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_dem_adhesion_surface_energy.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Particle
{

  class ParticleEngineInterface;
  class ParticleContainerBundle;
  class WallHandlerInterface;
  class InteractionWriter;
  class DEMNeighborPairs;
  class DEMHistoryPairs;
  class DEMAdhesionLawBase;

  /*!
   * Parameters necessary to calculate the adhesive forces in DEM.
   */
  struct DEMAdhesionParams
  {
    double surface_energy;  //!< the mean surface energy of the particles
    DEMAdhesionSurfaceEnergyDistributionParams
        surface_energy_distribution_params;  //!< Parameters for the surface energy distribution
    double adhesion_distance;                //!< Distance to which adhesive forces are evaluated
  };

  /*!
   * Verifies that the parameters are valid and throws an exception otherwise.
   */
  void verify_params_adhesion(const DEMAdhesionParams& params);

  class DEMAdhesion final
  {
   public:
    //! constructor
    explicit DEMAdhesion(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~DEMAdhesion();

    //! setup contact handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
        const std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs,
        const std::shared_ptr<Particle::DEMHistoryPairs> historypairs, const double& k_normal);

    //! get adhesion distance
    inline double get_adhesion_distance() const { return adhesion_params_.adhesion_distance; };

    //! add adhesion contribution to force field
    void add_force_contribution();

   private:
    //! init adhesion law handler
    void init_adhesion_law_handler();

    //! setup particle interaction writer
    void setup_particle_interaction_writer();

    //! evaluate particle adhesion contribution
    void evaluate_particle_adhesion();

    //! evaluate particle-wall adhesion contribution
    void evaluate_particle_wall_adhesion();

    //! discrete element method specific parameter list
    const Teuchos::ParameterList& params_dem_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! particle interaction writer
    std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter_;

    //! neighbor pair handler
    std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs_;

    //! history pair handler
    std::shared_ptr<Particle::DEMHistoryPairs> historypairs_;

    //! adhesion law handler
    std::unique_ptr<Particle::DEMAdhesionLawBase> adhesionlaw_;

    //! Parameters for the adhesion calculation
    const DEMAdhesionParams adhesion_params_;

    //! write particle-wall interaction output
    const bool writeparticlewallinteraction_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
