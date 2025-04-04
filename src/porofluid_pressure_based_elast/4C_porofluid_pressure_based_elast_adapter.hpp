// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_ADAPTER_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_ADAPTER_HPP

#include "4C_config.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class PoroFluidMultiphaseWrapper;
  class Structure;
}  // namespace Adapter

namespace Core::LinAlg
{
  class Solver;
}

namespace POROMULTIPHASE
{
  class PoroMultiPhase
  {
   public:
    /// constructor
    PoroMultiPhase() {};

    /// virtual destructor
    virtual ~PoroMultiPhase() = default;

    /// initialization
    virtual void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& structparams,
        const Teuchos::ParameterList& fluidparams, const std::string& struct_disname,
        const std::string& fluid_disname, bool isale, int nds_disp, int nds_vel,
        int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearbyelepairs) = 0;

    /*!
     * @brief Perform all necessary tasks after initializing the
     * object. Currently, this only calls the post_setup routine of
     * the structure field.
     */
    virtual void post_init() = 0;


    /// read restart
    virtual void read_restart(int restart) = 0;

    /// test results (if necessary)
    virtual void create_field_test() = 0;

    /// setup
    virtual void setup_system() = 0;

    /// setup the solver (only for monolithic system)
    virtual bool setup_solver() = 0;

    /// perform relaxation (only for partitioned system)
    virtual void perform_relaxation(
        std::shared_ptr<const Core::LinAlg::Vector<double>> phi, const int itnum) = 0;

    /// get relaxed fluid solution (only for partitioned system)
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> relaxed_fluid_phinp() const = 0;

    /// set relaxed fluid solution on structure (only for partitioned system)
    virtual void set_relaxed_fluid_solution() = 0;

    /// prepare timeloop of coupled problem
    virtual void prepare_time_loop() = 0;

    /// timeloop of coupled problem
    virtual void timeloop() = 0;

    /// time step of coupled problem
    virtual void time_step() = 0;

    /// time step of coupled problem
    virtual void prepare_time_step() = 0;

    //! update time step and print to screen
    virtual void update_and_output() = 0;

    /// set structure solution on scatra field
    virtual void set_struct_solution(std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel) = 0;

    /// set scatra solution on fluid field
    virtual void set_scatra_solution(
        unsigned nds, std::shared_ptr<const Core::LinAlg::Vector<double>> scalars) = 0;

    /// dof map of vector of unknowns
    virtual std::shared_ptr<const Core::LinAlg::Map> struct_dof_row_map() const = 0;

    /// unknown displacements at \f$t_{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> struct_dispnp() const = 0;

    /// unknown velocity at \f$t_{n+1}\f$
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> struct_velnp() const = 0;

    /// dof map of vector of unknowns
    virtual std::shared_ptr<const Core::LinAlg::Map> fluid_dof_row_map() const = 0;

    /// dof map of vector of unknowns of artery field
    virtual std::shared_ptr<const Core::LinAlg::Map> artery_dof_row_map() const = 0;

    /// return fluid flux
    virtual std::shared_ptr<const Core::LinAlg::MultiVector<double>> fluid_flux() const = 0;

    /// return fluid solution variable
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_phinp() const = 0;

    /// return fluid solution variable
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_saturation() const = 0;

    /// return fluid solution variable
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_pressure() const = 0;

    /// return fluid solution variable
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> solid_pressure() const = 0;

    //! unique map of all dofs that should be constrained with DBC
    virtual std::shared_ptr<const Core::LinAlg::Map> combined_dbc_map() const = 0;

    //! evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fx, const bool firstcall) = 0;

    //! access to monolithic right-hand side vector
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() const = 0;

    //! update all fields after convergence (add increment on displacements and fluid primary
    //! variables)
    virtual void update_fields_after_convergence(
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx) = 0;

    //! get the extractor
    virtual std::shared_ptr<const Core::LinAlg::MultiMapExtractor> extractor() const = 0;

    //! get the monolithic system matrix
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() const = 0;

    //! get structure field
    virtual const std::shared_ptr<Adapter::Structure>& structure_field() = 0;

    //! get fluid field
    virtual const std::shared_ptr<Adapter::PoroFluidMultiphaseWrapper>& fluid_field() = 0;

    //! build the block null spaces
    virtual void build_block_null_spaces(std::shared_ptr<Core::LinAlg::Solver>& solver) = 0;

    //! build the block null spaces
    virtual void build_artery_block_null_space(
        std::shared_ptr<Core::LinAlg::Solver>& solver, const int& arteryblocknum) = 0;
  };
}  // namespace POROMULTIPHASE


FOUR_C_NAMESPACE_CLOSE

#endif
