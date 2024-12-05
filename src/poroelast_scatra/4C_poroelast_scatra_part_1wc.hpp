// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROELAST_SCATRA_PART_1WC_HPP
#define FOUR_C_POROELAST_SCATRA_PART_1WC_HPP

#include "4C_config.hpp"

#include "4C_poroelast_scatra_part.hpp"

FOUR_C_NAMESPACE_OPEN

namespace PoroElastScaTra
{
  class PoroScatraPart1WC : public PoroScatraPart
  {
   public:
    explicit PoroScatraPart1WC(MPI_Comm comm, const Teuchos::ParameterList& timeparams)
        : PoroScatraPart(comm, timeparams){};

    //! solve one time step of porous media problem
    void do_poro_step() override;
    //! solve one time step of scalar transport problem
    void do_scatra_step() override;

    //! prepare output
    void prepare_output() override;

    //! update time step
    void update() override;

    //! write output print to screen
    void output() override;
  };

  class PoroScatraPart1WCPoroToScatra : public PoroScatraPart1WC
  {
   public:
    //! constructor
    explicit PoroScatraPart1WCPoroToScatra(MPI_Comm comm, const Teuchos::ParameterList& timeparams);

    //! actual time loop
    void timeloop() override;

    //! increment time and step and print header
    void prepare_time_step(bool printheader = true) override;

    //! perform iteration loop between fields
    void solve() override;

    //! read and set fields needed for restart
    void read_restart(int restart) override;
  };

  class PoroScatraPart1WCScatraToPoro : public PoroScatraPart1WC
  {
   public:
    //! constructor
    explicit PoroScatraPart1WCScatraToPoro(MPI_Comm comm, const Teuchos::ParameterList& timeparams);

    //! actual time loop
    void timeloop() override;

    //! increment time and step and print header
    void prepare_time_step(bool printheader = true) override;

    //! perform iteration loop between fields
    void solve() override;

    //! read and set fields needed for restart
    void read_restart(int restart) override;
  };
}  // namespace PoroElastScaTra


FOUR_C_NAMESPACE_CLOSE

#endif
