// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_PARTITIONED_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_PARTITIONED_HPP

#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_elast_base.hpp"
#include "4C_porofluid_pressure_based_elast_input.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Global
{
  class Problem;
}  // namespace Global

namespace PoroPressureBased
{
  //! Partitioned solution scheme of porofluid-elasticity problems
  class PorofluidElastPartitionedAlgorithm final : public PorofluidElastAlgorithm
  {
   public:
    PorofluidElastPartitionedAlgorithm(
        Global::Problem& problem, MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /// initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& structparams,
        const Teuchos::ParameterList& fluidparams, const std::string& struct_disname,
        const std::string& fluid_disname, bool isale, int nds_disp, int nds_vel,
        int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    /// setup
    void setup_system() override;

    /// time step of coupled problem
    void time_step() override { outer_loop(); };

    /// read restart
    void read_restart(int restart) override;

    // update
    void update_and_output() override;

    // update
    std::shared_ptr<const Core::LinAlg::Vector<double>> relaxed_fluid_phinp() const override
    {
      return fluidphinp_;
    }

   private:
    //! perform iteration loop between fields
    void outer_loop();

    //! perform iteration step of structure field and set the new disp and vel states in the fluid
    //! field
    void do_struct_step();

    //! perform iteration step of scatra field and set the new phi state in the structure field
    void do_fluid_step();

    //! update the current states in every iteration
    //! states are set to the last solutions obtained
    void iter_update_states();

    //! convergence check of outer loop
    bool convergence_check(int itnum);

    //! perform relaxation
    void perform_relaxation(
        std::shared_ptr<const Core::LinAlg::Vector<double>> phi, const int itnum) override;

    /// set (relaxed) fluid solution on structure field
    void set_relaxed_fluid_solution() override;

    /// perform aitken
    void aitken_relaxation(double& omega, const int itnum);

    //! pressure increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> phiincnp_;
    //! artery pressure increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> arterypressincnp_;
    //! displacement increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> dispincnp_;

    //! fluid primary variable at time n+1, iteration i+1
    std::shared_ptr<Core::LinAlg::Vector<double>> fluidphinp_;
    //! fluid primary variable at time n+1, iteration i
    std::shared_ptr<Core::LinAlg::Vector<double>> fluidphioldnp_;
    //! fluid primary variable increment: phi,n+1^i+1 - phi,n+1^i
    std::shared_ptr<Core::LinAlg::Vector<double>> fluidphiincnp_;
    //! old fluid primary variablee increment: phi,n+1^i+1 - phi,n+1^i
    std::shared_ptr<Core::LinAlg::Vector<double>> fluidphiincnpold_;

    //! convergence tolerance
    double ittol_;
    //! relaxation parameter
    double omega_;
    //! start value for relaxation parameter (or constant value)
    double startomega_;
    //! minimum value for relaxation parameter
    double omegamin_;
    //! maximum value for relaxation parameter
    double omegamax_;
    //! maximum iteration steps
    int itmax_;
    //! current iteration step
    int itnum_;
    //! write restart every n steps
    int writerestartevery_;
    //! is artery coupling active
    bool artery_coupling_active_;

    //! relaxation method
    RelaxationMethods relaxationmethod_;
  };


}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
