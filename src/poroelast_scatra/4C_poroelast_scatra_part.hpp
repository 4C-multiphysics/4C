// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROELAST_SCATRA_PART_HPP
#define FOUR_C_POROELAST_SCATRA_PART_HPP

/*----------------------------------------------------------------------*
 | header inclusions                                                     |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_poroelast_scatra_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | forward declarations                                                  |
 *----------------------------------------------------------------------*/
namespace Adapter
{
  class ScaTraBaseAlgorithm;
}

namespace PoroElast
{
  class PoroBase;
}

/*----------------------------------------------------------------------*
 |                                                                       |
 *----------------------------------------------------------------------*/
namespace PoroElastScaTra
{
  /// partitioned algorithm for scalar transport in porous media
  class PoroScatraPart : public PoroScatraBase
  {
   public:
    /// Constructor
    explicit PoroScatraPart(MPI_Comm comm, const Teuchos::ParameterList& timeparams);

    // Methods

    void setup_system() override;

    //! set up a pointer to the contact strategy of the structural field and store it
    void setup_contact_strategy();

    void set_poro_solution() override;

    void set_scatra_solution() override;

    //! solve one time/incremental step of porous media problem (depending on coupling algorithm)
    virtual void do_poro_step() = 0;
    //! solve one time/incremental step of scalar transport problem (depending on coupling
    //! algorithm)
    virtual void do_scatra_step() = 0;

   protected:
    //! store contact nitsche strategy for ssi problems
    std::shared_ptr<CONTACT::NitscheStrategySsi> contact_strategy_nitsche_;
  };
}  // namespace PoroElastScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
