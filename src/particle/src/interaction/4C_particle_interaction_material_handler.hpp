// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_MATERIAL_HANDLER_HPP
#define FOUR_C_PARTICLE_INTERACTION_MATERIAL_HANDLER_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_mat_particle_base.hpp"
#include "4C_mat_particle_dem.hpp"
#include "4C_mat_particle_sph_boundary.hpp"
#include "4C_mat_particle_sph_fluid.hpp"
#include "4C_mat_particle_thermo.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class MaterialHandler final
  {
   public:
    //! constructor
    explicit MaterialHandler(const Teuchos::ParameterList& params);

    //! return pointer to particle material parameter
    inline const Mat::PAR::ParticleMaterialBase* get_ptr_to_particle_mat_parameter(
        Particle::Type type_i) const
    {
      return phasetypetoparticlematpar_[static_cast<int>(type_i)];
    }

    //! get particle types of stored particle material parameters
    inline std::set<Particle::Type> get_particle_types() const { return storedtypes_; };

   private:
    void initialize_parameters();

    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! relate particle types to particle material parameters
    std::vector<const Mat::PAR::ParticleMaterialBase*> phasetypetoparticlematpar_;

    //! set of particle types of stored particle material parameters
    std::set<Particle::Type> storedtypes_;
  };
}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
