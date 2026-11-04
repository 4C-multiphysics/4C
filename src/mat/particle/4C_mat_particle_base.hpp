// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PARTICLE_BASE_HPP
#define FOUR_C_MAT_PARTICLE_BASE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class definitions                                                         |
 *---------------------------------------------------------------------------*/
namespace Mat
{
  namespace PAR
  {
    class ParticleMaterialBase : virtual public Core::Mat::PAR::Parameter
    {
     public:
      //! constructor
      ParticleMaterialBase(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! initial radius
      const double initRadius_;

      //! initial density
      const double initDensity_;

      //@}

      //! create material instance of matching type with parameters
      std::shared_ptr<Core::Mat::Material> create_material() override = 0;
    };

  }  // namespace PAR

}  // namespace Mat

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
