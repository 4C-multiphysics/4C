// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_MESHTYING_STRATEGY_BASE_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_MESHTYING_STRATEGY_BASE_HPP

#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_porofluid_pressure_based_timint_implicit.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class ArtNet;
}

namespace Core::LinAlg
{
  struct SolverParams;
}

namespace PoroPressureBased
{
  class MeshtyingStrategyBase
  {
   public:
    //! constructor
    explicit MeshtyingStrategyBase(PoroPressureBased::TimIntImpl* porofluidmultitimint,
        const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& poroparams)
        : porofluidmultitimint_(porofluidmultitimint),
          params_(probparams),
          poroparams_(poroparams),
          vectornormfres_(Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
              poroparams_, "VECTORNORM_RESF")),
          vectornorminc_(Teuchos::getIntegralValue<PoroPressureBased::VectorNorm>(
              poroparams_, "VECTORNORM_INC"))
    {
      return;
    }

    //! destructor
    virtual ~MeshtyingStrategyBase() = default;

    //! prepare time loop
    virtual void prepare_time_loop() = 0;

    //! prepare time step
    virtual void prepare_time_step() = 0;

    //! update
    virtual void update() = 0;

    //! output
    virtual void output() = 0;

    //! Initialize the linear solver
    virtual void initialize_linear_solver(std::shared_ptr<Core::LinAlg::Solver> solver) = 0;

    //! solve linear system of equations
    virtual void linear_solve(std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Core::LinAlg::SparseOperator> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> increment,
        std::shared_ptr<Core::LinAlg::Vector<double>> residual,
        Core::LinAlg::SolverParams& solver_params) = 0;

    //! calculate norms for convergence checks
    virtual void calculate_norms(std::vector<double>& preresnorm, std::vector<double>& incprenorm,
        std::vector<double>& prenorm,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> increment) = 0;

    //! create the field test
    virtual void create_field_test() = 0;

    //! restart
    virtual void read_restart(const int step) = 0;

    //! evaluate mesh tying
    virtual void evaluate() = 0;

    //! extract increments and update mesh tying
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> extract_and_update_iter(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> inc) = 0;

    // return arterial network time integrator
    virtual std::shared_ptr<Adapter::ArtNet> art_net_tim_int()
    {
      FOUR_C_THROW("ArtNetTimInt() not implemented in base class, wrong mesh tying object?");
      return nullptr;
    }

    //! access dof row map
    virtual std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const
    {
      FOUR_C_THROW("ArteryDofRowMap() not implemented in base class, wrong mesh tying object?");
      return nullptr;
    }

    //! access to block system matrix of artery poro problem
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> artery_porofluid_sysmat() const
    {
      FOUR_C_THROW(
          "artery_porofluid_sysmat() not implemented in base class, wrong mesh tying object?");
      return nullptr;
    }

    //! right-hand side alias the dynamic force residual for coupled system
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> artery_porofluid_rhs() const
    {
      FOUR_C_THROW("ArteryPorofluidRHS() not implemented in base class, wrong mesh tying object?");
      return nullptr;
    }

    //! access to global (combined) increment of coupled problem
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> combined_increment(
        std::shared_ptr<const Core::LinAlg::Vector<double>> inc) const = 0;

    //! check if initial fields on coupled DOFs are equal (only for node-based coupling)
    virtual void check_initial_fields(
        std::shared_ptr<const Core::LinAlg::Vector<double>> vec_cont) const = 0;

    //! set the element pairs that are close as found by search algorithm
    virtual void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearbyelepairs) = 0;

    //! setup the strategy
    virtual void setup() = 0;

    //! apply the mesh movement
    virtual void apply_mesh_movement() const = 0;

    //! return blood vessel volume fraction
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction()
    {
      FOUR_C_THROW(
          "blood_vessel_volume_fraction() not implemented in base class, wrong mesh tying object?");
      return nullptr;
    }

   protected:
    //! porofluid multi time integrator
    PoroPressureBased::TimIntImpl* porofluidmultitimint_;

    //! parameter list of global control problem
    const Teuchos::ParameterList& params_;

    //! parameter list of poro fluid multiphase problem
    const Teuchos::ParameterList& poroparams_;

    // vector norm for residuals
    enum PoroPressureBased::VectorNorm vectornormfres_;

    // vector norm for increments
    enum PoroPressureBased::VectorNorm vectornorminc_;
  };

}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
