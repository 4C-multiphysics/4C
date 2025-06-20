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
#include "4C_porofluid_pressure_based_elast_base.hpp"
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

namespace PoroPressureBased
{
  //! Base class of all algorithms for porofluid-elasticity problems with scalar transport and
  //! possible coupling to 1D arteries
  class PorofluidElastScatraBaseAlgorithm : public Adapter::AlgorithmBase
  {
   public:
    PorofluidElastScatraBaseAlgorithm(
        MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    //! initialization
    virtual void init(const Teuchos::ParameterList& global_time_params,
        const Teuchos::ParameterList& porofluid_elast_scatra_params,
        const Teuchos::ParameterList& porofluid_elast_params,
        const Teuchos::ParameterList& structure_params,
        const Teuchos::ParameterList& porofluid_params, const Teuchos::ParameterList& scatra_params,
        const std::string& structure_disname, const std::string& porofluid_disname,
        const std::string& scatra_disname, bool isale, int nds_disp, int nds_vel,
        int nds_solidpressure, int nds_porofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) = 0;

    /*!
     * @brief Perform all the necessary tasks after initializing the algorithm. Currently, this only
     * calls the post_setup routine of the underlying porofluid-elasticity algorithm.
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

    //! prepare time loop of coupled problem
    void prepare_time_loop();

    //! time loop of coupled problem
    void time_loop();

    //! time step of coupled problem --> here the actual action happens (overwritten by sub-classes)
    virtual void time_step() = 0;

    //! time step of coupled problem
    void prepare_time_step() override { prepare_time_step(false); };

    //! time step of coupled problem
    void prepare_time_step(bool printheader);

    //! update time step and print to screen
    void update_and_output();

    //! apply solution of porofluid-elasticity-problem to scatra
    void set_porofluid_elast_solution();

    //! apply solution of scatra to porofluid-elasticity-problem
    void set_scatra_solution();

    //! apply the additional Dirichlet boundary condition for volume fraction species
    void apply_additional_dbc_for_vol_frac_species();

    //! access to porofluid-elasticity algorithm
    const std::shared_ptr<PorofluidElastAlgorithm>& porofluid_elast_algo()
    {
      return porofluid_elast_algo_;
    }

    //! access to scatra algorithm
    const std::shared_ptr<Adapter::ScaTraBaseAlgorithm>& scatra_algo() { return scatra_algo_; }

    //! dof map of vector of unknowns of scatra field
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> scatra_dof_row_map() const;

    //! handle divergence of solver
    void handle_divergence() const;

   private:
    //! underlying porofluid algorithm
    std::shared_ptr<PorofluidElastAlgorithm> porofluid_elast_algo_;

    //! underlying scatra problem
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_algo_;

    //! flux-reconstruction active
    bool flux_reconstruction_active_;

    //! dofset of scatra field on fluid dis
    //! TODO: find a better way to do this. Perhaps this should be moved to the adapter?
    int nds_porofluid_scatra_;

    Teuchos::Time timer_timestep_;  //!< timer for measurement of duration of one time-step
    double dt_timestep_;            //!< duration of one time step

   protected:
    //! what to do when nonlinear solution fails
    enum PoroPressureBased::DivergenceAction divergence_action_;
    //! do we perform coupling with 1D artery
    const bool artery_coupling_;

    //! additional volume-fraction species Dirichlet conditions
    std::shared_ptr<Core::LinAlg::Map> add_dirichmaps_volfrac_spec_;

    std::shared_ptr<ScaTra::MeshtyingStrategyArtery> scatra_meshtying_strategy_;
  };

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
