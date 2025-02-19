// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_ADAPTER_HPP
#define FOUR_C_THERMO_ADAPTER_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_thermo_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_result_test.hpp"

#include <Epetra_Map.h>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class BlockSparseMatrixBase;
  class MapExtractor;
  class MultiMapExtractor;
}  // namespace Core::LinAlg

namespace Thermo
{
  /// general thermal field interface
  /*!
  The point is to keep T(F)SI as far apart from our field solvers as
  possible. Each thermal field solver we want to use should get its own
  subclass of this. The T(F)SI algorithm should be able to extract all the
  information from the thermal field it needs using this interface.

  All T(F)SI algorithms use this adapter to communicate with the thermal
  field. There are different ways to use this adapter.

  In all cases you need to tell the thermal algorithm about your time
  step. Therefore prepare_time_step(), update() and output() must be called at
  the appropriate position in the TSI algorithm.

  <h3>Dirichlet-Neumann coupled TSI</h3>

  Dirichlet-Neumann coupled TSI will need to Solve() the linear thermal problem
  for each time step after the structure displacements/velocities have been
  applied (ApplyStructVariables()). Solve() will be called many times for each
  time step until the equilibrium is reached. The thermal algorithm has to
  preserve its state until update() is called.

  After each Solve() you get the new temperatures by Tempnp().

  <h3>Monolithic TSI</h3>

  Monolithic TSI is based on evaluate() of elements. This results in a new
  RHS() and a new SysMat(). Together with the initial_guess() these form the
  building blocks for a block based Newton's method.

  \warning Further cleanup is still needed.
  */
  class Adapter
  {
   public:
    /// virtual to get polymorph destruction
    virtual ~Adapter() = default;

    /// @name Vector access
    //@{

    /// initial guess of Newton's method
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> initial_guess() = 0;

    /// RHS of Newton's method
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() = 0;

    /// unknown temperatures at t(n+1)
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> tempnp() = 0;

    /// unknown temperatures at t(n)
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> tempn() = 0;

    //@}

    /// @name Misc
    //@{

    /// DOF map of vector of unknowns
    virtual std::shared_ptr<const Epetra_Map> dof_row_map() = 0;

    /// DOF map of vector of unknowns for multiple dofsets
    virtual std::shared_ptr<const Epetra_Map> dof_row_map(unsigned nds) = 0;

    /// direct access to system matrix
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() = 0;

    /// direct access to discretization
    virtual std::shared_ptr<Core::FE::Discretization> discretization() = 0;

    /// Return MapExtractor for Dirichlet boundary conditions
    virtual std::shared_ptr<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() = 0;

    //@}

    //! @name Time step helpers
    //@{
    //! Return current time \f$t_{n}\f$
    virtual double time_old() const = 0;

    //! Return target time \f$t_{n+1}\f$
    virtual double time() const = 0;

    /// Get upper limit of time range of interest
    virtual double get_time_end() const = 0;

    /// Get time step size \f$\Delta t_n\f$
    virtual double dt() const = 0;

    /// Return current step number $n$
    virtual int step_old() const = 0;

    /// Return current step number $n+1$
    virtual int step() const = 0;

    /// Get number of time steps
    virtual int num_step() const = 0;

    /// Set time step size for the current step
    virtual void set_dt(double timestepsize) = 0;

    //! Sets the target time \f$t_{n+1}\f$ of this time step
    virtual void set_timen(const double time) = 0;

    /// Take the time and integrate (time loop)
    void integrate();

    /// tests if there are more time steps to do
    virtual bool not_finished() const = 0;

    /// start new time step
    virtual void prepare_time_step() = 0;

    /// evaluate residual at given temperature increment
    virtual void evaluate() = 0;

    /// update temperature increment after Newton step
    virtual void update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> tempi) = 0;

    /// update at time step end
    virtual void update() = 0;

    /// print info about finished time step
    virtual void print_step() = 0;

    //! Access to output object
    virtual std::shared_ptr<Core::IO::DiscretizationWriter> disc_writer() = 0;

    /// output results
    virtual void output(bool forced_writerestart = false) = 0;

    /// read restart information for given time step
    virtual void read_restart(const int step) = 0;

    /// reset everything to beginning of time step, for adaptivity
    virtual void reset_step() = 0;

    //@}

    //! @name Solver calls
    //@{

    /**
     * @brief Non-linear solve
     *
     * Do the nonlinear solve, i.e. (multiple) corrector for the time step.
     * All boundary conditions have been set.
     *
     * @return status of the solve, which can be used for adaptivity
     */
    virtual Thermo::ConvergenceStatus solve() = 0;

    //@}

    /// Identify residual
    /// This method does not predict the target solution but
    /// evaluates the residual and the stiffness matrix.
    /// In partitioned solution schemes, it is better to keep the current
    /// solution instead of evaluating the initial guess (as the predictor)
    /// does.
    virtual void prepare_step() = 0;

    /// create result test for encapulated thermo algorithm
    virtual std::shared_ptr<Core::Utils::ResultTest> create_field_test() = 0;
  };


  /// thermo field solver
  class BaseAlgorithm
  {
   public:
    explicit BaseAlgorithm(
        const Teuchos::ParameterList& prbdyn, std::shared_ptr<Core::FE::Discretization> actdis);

    virtual ~BaseAlgorithm() = default;

    /// return thermal field solver
    std::shared_ptr<Adapter> thermo_field() { return thermo_; }

   private:
    /// thermal field solver
    std::shared_ptr<Adapter> thermo_;
  };
}  // namespace Thermo

FOUR_C_NAMESPACE_CLOSE

#endif
