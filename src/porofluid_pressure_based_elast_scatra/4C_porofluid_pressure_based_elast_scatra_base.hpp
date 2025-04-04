// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_BASE_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_BASE_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_porofluid_pressure_based_elast_adapter.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Teuchos_Time.hpp>

#include <memory>
#include <set>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class ScaTraBaseAlgorithm;
}  // namespace Adapter

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace ScaTra
{
  class MeshtyingStrategyArtery;
}

namespace PoroMultiPhaseScaTra
{
  //! Base class of all solid-scatra algorithms
  class PoroMultiPhaseScaTraBase : public Adapter::AlgorithmBase
  {
   public:
    PoroMultiPhaseScaTraBase(MPI_Comm comm,
        const Teuchos::ParameterList& globaltimeparams);  // Problem builder

    //! initialization
    virtual void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& poroparams,
        const Teuchos::ParameterList& structparams, const Teuchos::ParameterList& fluidparams,
        const Teuchos::ParameterList& scatraparams, const std::string& struct_disname,
        const std::string& fluid_disname, const std::string& scatra_disname, bool isale,
        int nds_disp, int nds_vel, int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearbyelepairs) = 0;

    /*!
     * @brief Perform all the necessary tasks after initializing the
     * algorithm. Currently, this only calls the post_setup routine of
     * the underlying poroelast multi-phase object.
     */
    void post_init();

    //! read restart
    void read_restart(int restart) override;

    //! create result test for subproblems
    void create_field_test();

    //! setup
    virtual void setup_system() = 0;

    //! setup solver (only needed in monolithic case)
    virtual void setup_solver() = 0;

    //! prepare timeloop of coupled problem
    void prepare_time_loop();

    //! timeloop of coupled problem
    void timeloop();

    //! time step of coupled problem --> here the actual action happens (overwritten by sub-classes)
    virtual void time_step() = 0;

    //! time step of coupled problem
    void prepare_time_step() override { prepare_time_step(false); };

    //! time step of coupled problem
    void prepare_time_step(bool printheader);

    //! update time step and print to screen
    void update_and_output();

    //! apply solution of poro-problem to scatra
    void set_poro_solution();

    //! apply solution of scatra to poro
    void set_scatra_solution();

    //! apply the additional Dirichlet boundary condition for volume fraction species
    void apply_additional_dbc_for_vol_frac_species();

    //! access to poro field
    const std::shared_ptr<POROMULTIPHASE::PoroMultiPhase>& poro_field() { return poromulti_; }

    //! access to fluid field
    const std::shared_ptr<Adapter::ScaTraBaseAlgorithm>& scatra_algo() { return scatra_; }

    //! dof map of vector of unknowns of scatra field
    virtual std::shared_ptr<const Core::LinAlg::Map> scatra_dof_row_map() const;

    //! handle divergence of solver
    void handle_divergence() const;

   private:
    //! underlying poroelast multi phase
    std::shared_ptr<POROMULTIPHASE::PoroMultiPhase> poromulti_;

    //! underlying scatra problem
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_;

    //! flux-reconstruction method
    POROFLUIDMULTIPHASE::FluxReconstructionMethod fluxreconmethod_;

    //! dofset of scatra field on fluid dis
    //! TODO: find a better way to do this. Perhaps this should be moved to the adapter?
    int ndsporofluid_scatra_;

    Teuchos::Time timertimestep_;  //!< timer for measurement of duration of one time-step
    double dttimestep_;            //!< duration of one time step

   protected:
    //! what to do when nonlinear solution fails
    enum PoroMultiPhaseScaTra::DivContAct divcontype_;
    //! do we perform coupling with 1D artery
    const bool artery_coupl_;

    //! additional volume-fraction species Dirichlet conditions
    std::shared_ptr<Core::LinAlg::Map> add_dirichmaps_volfrac_spec_;

    std::shared_ptr<ScaTra::MeshtyingStrategyArtery> scatramsht_;

  };  // PoroMultiPhaseScaTraBase


}  // namespace PoroMultiPhaseScaTra



FOUR_C_NAMESPACE_CLOSE

#endif
