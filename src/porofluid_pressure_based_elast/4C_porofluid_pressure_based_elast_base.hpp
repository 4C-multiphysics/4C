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
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_porofluid_pressure_based_elast_algorithm_dependencies.hpp"
#include "4C_utils_exceptions.hpp"

#include <optional>
#include <utility>

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

namespace PoroPressureBased
{
  //! Base class of porofluid-elasticity algorithms
  class PorofluidElastAlgorithm : public Adapter::AlgorithmBase
  {
   public:
    PorofluidElastAlgorithm(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /// initialization
    virtual void init(const Teuchos::ParameterList& global_time_params,
        const Teuchos::ParameterList& porofluid_elast_params,
        const Teuchos::ParameterList& structure_params,
        const Teuchos::ParameterList& porofluid_params, const std::string& structure_disname,
        const std::string& porofluid_disname, bool isale, int nds_disp, int nds_vel,
        int nds_solidpressure, int nds_porofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) = 0;

    virtual void post_init();

    void set_algorithm_deps(PorofluidElastAlgorithmDeps algorithm_deps)
    {
      algorithm_deps_ = std::move(algorithm_deps);
    }

    /// read restart
    void read_restart(int restart) override;

    /// test results (if necessary)
    virtual void create_field_test();

    /// setup
    virtual void setup_system() = 0;

    /// prepare time loop of coupled problem
    virtual void prepare_time_loop();

    /// time loop of coupled problem
    virtual void time_loop();

    /// time step of coupled problem
    virtual void time_step() = 0;

    /// prepare time step of coupled problem
    void prepare_time_step() override;

    //! update fields after convergence
    virtual void update_and_output();

    /// dof map of vector of unknowns of structure field
    std::shared_ptr<const Core::LinAlg::Map> structure_dof_row_map() const;

    /// dof map of vector of unknowns of fluid field
    std::shared_ptr<const Core::LinAlg::Map> porofluid_dof_row_map() const;

    /// dof map of vector of unknowns of artery field
    std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const;

    /// system matrix of coupled artery porofluid problem
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> artery_porofluid_sysmat() const;

    //! access to structural field
    const std::shared_ptr<Adapter::Structure>& structure_algo() { return structure_algo_; }

    //! access to fluid field
    const std::shared_ptr<Adapter::PoroFluidMultiphaseWrapper>& porofluid_algo()
    {
      return porofluid_algo_;
    }

    /// set structure solution on scatra field
    void set_structure_solution(std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel) const;

    /// set scatra solution on porofluid field
    void set_scatra_solution(
        unsigned nds, std::shared_ptr<const Core::LinAlg::Vector<double>> scalars) const;

    //! setup solver (for monolithic only)
    virtual bool setup_solver() { return false; };

    /// unknown displacements at \f$t_{n+1}\f$
    std::shared_ptr<const Core::LinAlg::Vector<double>> structure_dispnp() const;

    /// return fluid flux
    std::shared_ptr<const Core::LinAlg::MultiVector<double>> fluid_flux() const;

    /// return fluid solution variable
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_phinp() const;

    /// return relaxed fluid solution variable (partitioned coupling will overwrite this method)
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> relaxed_fluid_phinp() const
    {
      return fluid_phinp();
    };

    /// set (relaxed) fluid solution on structure field (partitioned coupling will overwrite this
    /// method)
    virtual void set_relaxed_fluid_solution()
    {
      FOUR_C_THROW("set_relaxed_fluid_solution() only available for partitioned schemes!");
    };

    /// return fluid solution variable
    std::shared_ptr<const Core::LinAlg::Vector<double>> solid_pressure() const;

    //! unique map of all dofs that should be constrained with DBC
    virtual std::shared_ptr<const Core::LinAlg::Map> combined_dbc_map() const
    {
      FOUR_C_THROW("combined_dbc_map() only available for monolithic schemes!");
    };

    //! build the block null spaces
    virtual void build_block_null_spaces(std::shared_ptr<Core::LinAlg::Solver>& solver)
    {
      FOUR_C_THROW("build_block_null_spaces() only available for monolithic schemes!");
    };

    //! build the block null spaces
    virtual void build_artery_block_null_space(
        std::shared_ptr<Core::LinAlg::Solver>& solver, const int& arteryblocknum)
    {
      FOUR_C_THROW("build_artery_block_null_space() only available for monolithic schemes!");
    };

    //! evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fx, const bool firstcall)
    {
      FOUR_C_THROW("evaluate() only available for monolithic schemes!");
    };

    //! update all fields after convergence (add increment on displacements and fluid primary
    //! variables)
    virtual void update_fields_after_convergence(
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx)
    {
      FOUR_C_THROW("update_fields_after_convergence() only available for monolithic schemes!");
    };

    /// perform relaxation (only for partitioned schemes)
    virtual void perform_relaxation(
        std::shared_ptr<const Core::LinAlg::Vector<double>> phi, const int itnum)
    {
      FOUR_C_THROW("PerformRelaxation() only available for partitioned schemes!");
    };

    //! get monolithic rhs vector
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() const
    {
      FOUR_C_THROW("RHS() only available for monolithic schemes!");
    };

    //! get extractor
    virtual std::shared_ptr<const Core::LinAlg::MultiMapExtractor> extractor() const
    {
      FOUR_C_THROW("Extractor() only available for monolithic schemes!");
    };

    //! get monolithic block system matrix
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() const
    {
      FOUR_C_THROW("block_system_matrix() only available for monolithic schemes!");
    };

   private:
    /// set structure mesh displacement on fluid field
    void set_mesh_disp(std::shared_ptr<const Core::LinAlg::Vector<double>> disp) const;

    /// set structure velocity field on fluid field
    void set_velocity_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> vel) const;

    /// underlying structure algorithm
    std::shared_ptr<Adapter::Structure> structure_algo_;

    /// underlying porofluid algorithm
    std::shared_ptr<Adapter::PoroFluidMultiphaseWrapper> porofluid_algo_;

   protected:
    const PorofluidElastAlgorithmDeps& algorithm_deps() const
    {
      FOUR_C_ASSERT_ALWAYS(algorithm_deps_.has_value(),
          "Porofluid-elast algorithm dependencies are not initialized.");
      return *algorithm_deps_;
    }

    std::optional<PorofluidElastAlgorithmDeps> algorithm_deps_;

    /// true if we solve the structure field
    /// (false in case of porofluid-scatra coupling without mesh deformation)
    bool solve_structure_;

    /// Print user output that structure field is disabled
    void print_structure_disabled_info() const;
  };

}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
