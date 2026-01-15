// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_WALL_INTERFACE_HPP
#define FOUR_C_PARTICLE_WALL_INTERFACE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_particle_engine_typedefs.hpp"

#include <Teuchos_RCPStdSharedPtrConversions.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Particle
{
  class WallDataState;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief interface to provide restricted access to particle wall handler
   *
   * The particle algorithm holds an instance of the particle wall handler, thus having full access
   * and control over it. This abstract interface class to the particle wall handler provides
   * restricted access to be used in all other classes.
   *
   * \note Methods in this class are documented briefly. Refer to the full documentation of the
   *       particle wall handler class!
   *
   */
  class WallHandlerInterface
  {
   public:
    //! virtual destructor
    virtual ~WallHandlerInterface() = default;

    /*!
     * \brief get wall discretization
     *
     *
     * \return wall discretization
     */
    virtual std::shared_ptr<const Core::FE::Discretization> get_wall_discretization() const = 0;

    /*!
     * \brief get wall data state container
     *
     *
     * \return wall data state container
     */
    virtual std::shared_ptr<Particle::WallDataState> get_wall_data_state() const = 0;

    /*!
     * \brief get reference to potential wall neighbors
     *
     *
     * \return potential particle wall neighbor pairs
     */
    virtual const Particle::PotentialWallNeighbors& get_potential_wall_neighbors() const = 0;

    /*!
     * \brief determine nodal positions of column wall element
     *
     *
     * \param ele[in]             column wall element
     * \param colelenodalpos[out] current nodal position
     */
    virtual void determine_col_wall_ele_nodal_pos(Core::Elements::Element* ele,
        std::map<int, Core::LinAlg::Matrix<3, 1>>& colelenodalpos) const = 0;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
