// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_BASE_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_BASE_HPP


#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_porofluid_pressure_based_elast_adapter.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class PoroFluidMultiphaseWrapper;
  class Structure;
}  // namespace Adapter

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace POROMULTIPHASE
{
  //! Base class of all solid-scatra algorithms
  class PoroMultiPhaseBase : public Adapter::AlgorithmBase, public PoroMultiPhase
  {
   public:
    PoroMultiPhaseBase(MPI_Comm comm,
        const Teuchos::ParameterList& globaltimeparams);  // Problem builder

    /// initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& structparams,
        const Teuchos::ParameterList& fluidparams, const std::string& struct_disname,
        const std::string& fluid_disname, bool isale, int nds_disp, int nds_vel,
        int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearbyelepairs) override = 0;

    void post_init() override;

    /// read restart
    void read_restart(int restart) override;

    /// test results (if necessary)
    void create_field_test() override;

    /// setup
    void setup_system() override = 0;

    /// prepare timeloop of coupled problem
    void prepare_time_loop() override;

    /// timeloop of coupled problem
    void timeloop() override;

    /// time step of coupled problem
    void time_step() override = 0;

    /// prepare time step of coupled problem
    void prepare_time_step() override;

    //! update fields after convergence
    void update_and_output() override;

    /// dof map of vector of unknowns of structure field
    std::shared_ptr<const Core::LinAlg::Map> struct_dof_row_map() const override;

    /// dof map of vector of unknowns of fluid field
    std::shared_ptr<const Core::LinAlg::Map> fluid_dof_row_map() const override;

    /// dof map of vector of unknowns of artery field
    std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const override;

    /// system matrix of coupled artery porofluid problem
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> artery_porofluid_sysmat() const;

    //! access to structural field
    const std::shared_ptr<Adapter::Structure>& structure_field() override { return structure_; }

    //! access to fluid field
    const std::shared_ptr<Adapter::PoroFluidMultiphaseWrapper>& fluid_field() override
    {
      return fluid_;
    }

    /// set structure solution on scatra field
    void set_struct_solution(std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel) override;

    /// set scatra solution on fluid field
    void set_scatra_solution(
        unsigned nds, std::shared_ptr<const Core::LinAlg::Vector<double>> scalars) override;

    //! setup solver (for monolithic only)
    bool setup_solver() override { return false; };

    /// unknown displacements at \f$t_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> struct_dispnp() const override;

    /// unknown velocity at \f$t_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> struct_velnp() const override;

    /// return fluid flux
    std::shared_ptr<const Core::LinAlg::MultiVector<double>> fluid_flux() const override;

    /// return fluid solution variable
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_phinp() const override;

    /// return relaxed fluid solution variable (partitioned coupling will overwrite this method)
    std::shared_ptr<const Core::LinAlg::Vector<double>> relaxed_fluid_phinp() const override
    {
      return fluid_phinp();
    };

    /// set (relaxed) fluid solution on structure field (partitioned coupling will overwrite this
    /// method)
    void set_relaxed_fluid_solution() override
    {
      FOUR_C_THROW("set_relaxed_fluid_solution() only available for partitioned schemes!");
      return;
    };

    /// return fluid solution variable
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_saturation() const override;

    /// return fluid solution variable
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_pressure() const override;

    /// return fluid solution variable
    std::shared_ptr<const Core::LinAlg::Vector<double>> solid_pressure() const override;

    //! unique map of all dofs that should be constrained with DBC
    std::shared_ptr<const Core::LinAlg::Map> combined_dbc_map() const override
    {
      FOUR_C_THROW("combined_dbc_map() only available for monolithic schemes!");
      return nullptr;
    };

    //! build the block null spaces
    void build_block_null_spaces(std::shared_ptr<Core::LinAlg::Solver>& solver) override
    {
      FOUR_C_THROW("build_block_null_spaces() only available for monolithic schemes!");
      return;
    };

    //! build the block null spaces
    void build_artery_block_null_space(
        std::shared_ptr<Core::LinAlg::Solver>& solver, const int& arteryblocknum) override
    {
      FOUR_C_THROW("build_artery_block_null_space() only available for monolithic schemes!");
      return;
    };

    //! evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fx, const bool firstcall) override
    {
      FOUR_C_THROW("evaluate() only available for monolithic schemes!");
      return;
    };

    //! update all fields after convergence (add increment on displacements and fluid primary
    //! variables)
    void update_fields_after_convergence(std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx) override
    {
      FOUR_C_THROW("update_fields_after_convergence() only available for monolithic schemes!");
      return;
    };

    /// perform relaxaton (only for partitioned schemes)
    void perform_relaxation(
        std::shared_ptr<const Core::LinAlg::Vector<double>> phi, const int itnum) override
    {
      FOUR_C_THROW("PerformRelaxation() only available for partitioned schemes!");
      return;
    };

    //! get monolithic rhs vector
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() const override
    {
      FOUR_C_THROW("RHS() only available for monolithic schemes!");
      return nullptr;
    };

    //! get extractor
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> extractor() const override
    {
      FOUR_C_THROW("Extractor() only available for monolithic schemes!");
      return nullptr;
    };

    //! get monolithic block system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() const override
    {
      FOUR_C_THROW("block_system_matrix() only available for monolithic schemes!");
      return nullptr;
    };

   private:
    /// set structure mesh displacement on fluid field
    void set_mesh_disp(std::shared_ptr<const Core::LinAlg::Vector<double>> disp);

    /// set structure velocity field on fluid field
    void set_velocity_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> vel);

    /// underlying structure of the PoroMultiPhase problem
    std::shared_ptr<Adapter::Structure> structure_;

    /// underlying fluid problem of the PoroMultiPhase problem
    std::shared_ptr<Adapter::PoroFluidMultiphaseWrapper> fluid_;

   protected:
    /// a zero vector of full length of structure dofs
    std::shared_ptr<Core::LinAlg::Vector<double>> struct_zeros_;
    //! here the computation of the structure can be skipped, this is helpful if only fluid-scatra
    //! coupling should be calculated
    bool solve_structure_;

    /// coupling with 1D artery network
    const bool artery_coupl_;

    /// Print user output that structure field is disabled
    void print_structure_disabled_info();

  };  // PoroMultiPhaseBase


}  // namespace POROMULTIPHASE



FOUR_C_NAMESPACE_CLOSE

#endif
