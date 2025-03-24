// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_BASE_HPP
#define FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_BASE_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_scatra_timint_convcheck_strategies.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::IO
{
  class InputControl;
}

namespace Core::LinAlg
{
  class KrylovProjector;
  class Solver;
  struct SolverParams;
  class SparseOperator;
  class MultiMapExtractor;
}  // namespace Core::LinAlg

namespace ScaTra
{
  class ConvCheckStrategyBase;
  class ScaTraTimIntImpl;

  /*!
  \brief Abstract interface for meshtying strategies in scalar transport problems (including
  standard strategy without meshtying)

  To keep the scalar transport time integrator class and derived classes as plain as possible,
  several algorithmic parts have been encapsulated within separate meshtying strategy classes.
  These algorithmic parts include initializing the system matrix and other relevant objects,
  computing meshtying residual terms and their linearizations, and solving the resulting
  linear system of equations. By introducing a hierarchy of strategies for these algorithmic
  parts, a bunch of unhandy if-else selections within the time integrator classes themselves
  can be circumvented. Every specific meshtying strategy, e.g. fluid-fluid meshtying, scatra-
  scatra interface coupling, and also the standard algorithm without meshtying, has to be
  implemented in a subclass derived from this abstract, purely virtual interface class.

  */

  class MeshtyingStrategyBase
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~MeshtyingStrategyBase() = default;

    //! constructor
    explicit MeshtyingStrategyBase(ScaTra::ScaTraTimIntImpl* scatratimint)
        : convcheckstrategy_(nullptr), scatratimint_(scatratimint)
    {
    }

    /*!
     * @brief perform convergence check for Newton-Raphson iteration
     *
     * @param scatratimint  scalar transport time integrator
     * @param actresidual   return maximum current residual value
     * @return bool indicating if convergence is met
     */
    bool abort_nonlin_iter(const ScaTraTimIntImpl& scatratimint, double& actresidual) const
    {
      if (convcheckstrategy_ == nullptr)
        FOUR_C_THROW("Strategy for Newton-Raphson convergence check has not been instantiated!");

      return convcheckstrategy_->abort_nonlin_iter(scatratimint, actresidual);
    }

    /*!
     * @brief perform convergence check for outer iteration in partitioned coupling schemes
     *
     * @param scatratimint  scalar transport time integrator
     * @return bool indicating if convergence is met
     */
    bool abort_outer_iter(const ScaTraTimIntImpl& scatratimint) const
    {
      if (convcheckstrategy_ == nullptr)
        FOUR_C_THROW("Strategy for outer convergence check has not been instantiated!");

      return convcheckstrategy_->abort_outer_iter(scatratimint);
    }

    //! provide global state vectors for element evaluation
    virtual void add_time_integration_specific_vectors() const { return; };


    virtual void equip_extended_solver_with_null_space_info() const
    {
      FOUR_C_THROW(
          "equip_extended_solver_with_null_space_info() is not implemented in "
          "MeshtyingStrategyBase.");
    }

    //! compute time step size
    virtual void compute_time_step_size(double& dt) { return; };

    //! compute time derivatives of discrete state variables
    virtual void compute_time_derivative() const { return; };

    /*!
     * @brief condense global system of equations
     *
     * @param systemmatrix        system matrix
     * @param residual            residual vector
     * @param calcinittimederiv   flag for calculation of initial time derivative
     */
    virtual void condense_mat_and_rhs(
        const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
        const bool calcinittimederiv = false) const {};

    //! return global map of degrees of freedom
    virtual const Core::LinAlg::Map& dof_row_map() const = 0;

    /*!
    \brief Evaluate a given condition

     Evaluate terms of your weak formulation on elements marked with a given condition.

    \param params (in):         List of parameters for use at element level
    \param systemmatrix1 (out): Sparse matrix that may be changed by
                                assembly of boundary element contributions.
                                May not be nullptr.
                                Matrix must be systemmatrix->Filled()==false on input.
    \param systemmatrix2 (out): Sparse matrix that may be changed by
                                assembly of boundary element contributions.
                                May not be nullptr.
                                Matrix must be systemmatrix->Filled()==false on input.
    \param systemvector1 (out): Vector to assemble BCs to.
                                The vector is NOT initialized to zero by this method.
    \param systemvector2 (out): Vector to assemble BCs to.
                                The vector is NOT initialized to zero by this method.
    \param systemvector3 (out): Vector to assemble BCs to.
                                The vector is NOT initialized to zero by this method.
    \param condstring (in):     Name of condition to be evaluated
    \param condid (in):         Condition ID

    \return void

    */
    virtual void evaluate_condition(Teuchos::ParameterList& params,
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
        std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
        std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3, const std::string& condstring,
        const int condid)
    {
      FOUR_C_THROW("evaluate_condition(...) is not implemented in MeshtyingStrategyBase.");
    };

    //! compute meshtying residual terms and their linearizations
    virtual void evaluate_meshtying() = 0;

    //! evaluate coupling between two points/nodes. Does not need to be evaluated on element level
    virtual void evaluate_point_coupling() {};

    //! explicit predictor step to obtain better starting value for Newton-Raphson iteration
    virtual void explicit_predictor() const { return; };

    //! include Dirichlet conditions into condensation
    virtual void include_dirichlet_in_condensation() const { return; };

    //! init meshtying objects
    virtual void init_meshtying() = 0;

    // flag returning whether this meshtying strategy needs to initialize the system matrix due to
    // special requirements
    virtual bool system_matrix_initialization_needed() const = 0;

    //! initialize system matrix
    virtual std::shared_ptr<Core::LinAlg::SparseOperator> init_system_matrix() const = 0;

    //! return interface map extractor
    virtual std::shared_ptr<Core::LinAlg::MultiMapExtractor> interface_maps() const = 0;

    //! collect data that shall be written to output
    virtual void collect_output_data() const {};

    //! output solution for post-processing
    virtual void output() const {};

    //! output restart information
    virtual void write_restart() const {};

    /*!
     * @brief  read restart data
     *
     * @param step  restart step
     * @param input control file manager
     */
    virtual void read_restart(
        const int step, std::shared_ptr<Core::IO::InputControl> input = nullptr) const {};

    //! set general parameters for element evaluation
    virtual void set_element_general_parameters(Teuchos::ParameterList& parameters) const
    {
      return;
    };

    //! compute history vector, i.e., the history part of the right-hand side vector with all
    //! contributions from the previous time step
    virtual void set_old_part_of_rhs() const { return; };

    /*!
    \brief Set state on an discretization hidden in any MeshtyingStrategy.

     This method should encapsulate a standard set_state(...) of the discretizations
     you want to consider. See \ref HeterogeneousReactionStrategy for an example.

    \param nds (in): number of dofset
    \param name (in): Name of data
    \param state (in): vector of some data

    \return void

    */
    virtual void set_state(unsigned nds, const std::string& name,
        std::shared_ptr<const Core::LinAlg::Vector<double>> state)
    {
      FOUR_C_THROW(
          "set_state(...) is not implemented in MeshtyingStrategyBase.\n"
          "set_state(...) allows to set states on discretizations held within any\n"
          "specific MeshtyingStrategy. See e.g. HeterogeneousReactionStrategy for\n"
          "an example implementation.");
    };

    //! setup meshtying objects
    virtual void setup_meshtying() = 0;

    /*!
     * @brief  solve resulting linear system of equations
     *
     * @param solver        solver
     * @param systemmatrix  system matrix
     * @param increment     increment vector
     * @param residual      residual vector
     * @param phinp         state vector at time n+1
     * @param iteration     number of current Newton-Raphson iteration
     * @param projector     Krylov projector
     */
    virtual void solve(const std::shared_ptr<Core::LinAlg::Solver>& solver,
        const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& increment,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& phinp, const int iteration,
        Core::LinAlg::SolverParams& solver_params) const = 0;

    //! return linear solver for global system of linear equations
    virtual const Core::LinAlg::Solver& solver() const = 0;

    //! update solution after convergence of the nonlinear Newton-Raphson iteration
    virtual void update() const { return; };

   protected:
    //! instantiate strategy for Newton-Raphson convergence check
    virtual void init_conv_check_strategy() = 0;

    //! strategy for Newton-Raphson convergence check called by scalar transport time integrator
    std::shared_ptr<ScaTra::ConvCheckStrategyBase> convcheckstrategy_;

    //! scalar transport time integrator
    ScaTra::ScaTraTimIntImpl* scatratimint_;

   private:
    //! copy constructor
    MeshtyingStrategyBase(const MeshtyingStrategyBase& old);
  };  // class MeshtyingStrategyBase
}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
