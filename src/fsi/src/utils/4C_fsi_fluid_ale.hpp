// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_FLUID_ALE_HPP
#define FOUR_C_FSI_FLUID_ALE_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid_ale.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FSI
{
  /// Fluid on Ale test algorithm
  class FluidAleAlgorithm : public Adapter::FluidMovingBoundaryBaseAlgorithm
  {
   public:
    explicit FluidAleAlgorithm(MPI_Comm comm);

    /// time loop
    void timeloop();

    /// communicator
    MPI_Comm get_comm() const { return comm_; }

    /// read restart data
    virtual void read_restart(int step);

   protected:
    /// time step size
    double dt() const { return dt_; }

    /// step number
    int step() const { return step_; }

    //! @name Time loop building blocks

    /// tests if there are more time steps to do
    bool not_finished() { return step_ < nstep_ and time_ <= maxtime_; }

    /// start a new time step
    void prepare_time_step();

    /// solve ale and fluid fields
    void solve();

    /// take current results for converged and save for next time step
    void update();

    /// write output
    void output();

    //@}

   private:
    /// communication (mainly for screen output)
    MPI_Comm comm_;

    //! @name Time stepping variables
    int step_;
    int nstep_;
    double time_;
    double maxtime_;
    double dt_;
    //@}
  };

}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
