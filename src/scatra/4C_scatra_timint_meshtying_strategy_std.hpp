// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_STD_HPP
#define FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_STD_HPP

#include "4C_config.hpp"

#include "4C_scatra_timint_meshtying_strategy_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  struct SolverParams;
}

namespace ScaTra
{
  /*!
  \brief Standard solution strategy for standard scalar transport problems (without meshtying)

  To keep the scalar transport time integrator class and derived classes as plain as possible,
  several algorithmic parts have been encapsulated within separate meshtying strategy classes.
  These algorithmic parts include initializing the system matrix and other relevant objects,
  computing meshtying residual terms and their linearizations, and solving the resulting
  linear system of equations. By introducing a hierarchy of strategies for these algorithmic
  parts, a bunch of unhandy if-else selections within the time integrator classes themselves
  can be circumvented. This class contains the standard solution strategy for standard scalar
  transport problems without meshtying.

  */

  class MeshtyingStrategyStd : public MeshtyingStrategyBase
  {
   public:
    //! constructor
    explicit MeshtyingStrategyStd(ScaTra::ScaTraTimIntImpl* scatratimint);

    //! return global map of degrees of freedom
    const Epetra_Map& dof_row_map() const override;

    /*!
    \brief Evaluate a given condition

     Evaluate terms of your weak formulation on elements marked with a given condition.

    \return void
    \date 08/16
    \author rauch
    */
    void evaluate_condition(Teuchos::ParameterList& params,
        Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1,
        Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix2,
        Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector1,
        Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector2,
        Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector3, const std::string& condstring,
        const int condid) override
    {
      FOUR_C_THROW("evaluate_condition(...) is not implemented in MeshtyingStrategyStd.");
    };

    //! compute meshtying residual terms and their linearizations
    void evaluate_meshtying() override;

    //! init meshtying objects
    void init_meshtying() override;

    Teuchos::RCP<Core::LinAlg::MultiMapExtractor> interface_maps() const override
    {
      FOUR_C_THROW("InterfaceMaps() is not implemented in MeshtyingStrategyStd.");
      return Teuchos::null;
    }

    bool system_matrix_initialization_needed() const override { return false; }

    Teuchos::RCP<Core::LinAlg::SparseOperator> init_system_matrix() const override
    {
      FOUR_C_THROW(
          "This meshtying strategy does not need to initialize the system matrix, but relies "
          "instead on the initialization of the field. If this changes, you also need to change "
          "'system_matrix_initialization_needed()' to return true");
      // dummy return
      return Teuchos::null;
    }

    //! setup meshtying objects
    void setup_meshtying() override;

    //! solve resulting linear system of equations
    void solve(const Teuchos::RCP<Core::LinAlg::Solver>& solver,         //!< solver
        const Teuchos::RCP<Core::LinAlg::SparseOperator>& systemmatrix,  //!< system matrix
        const Teuchos::RCP<Core::LinAlg::Vector<double>>& increment,     //!< increment vector
        const Teuchos::RCP<Core::LinAlg::Vector<double>>& residual,      //!< residual vector
        const Teuchos::RCP<Core::LinAlg::Vector<double>>& phinp,  //!< state vector at time n+1
        const int iteration,  //!< number of current Newton-Raphson iteration
        Core::LinAlg::SolverParams& solver_params) const override;

    //! return linear solver for global system of linear equations
    const Core::LinAlg::Solver& solver() const override;

   protected:
    //! instantiate strategy for Newton-Raphson convergence check
    void init_conv_check_strategy() override;

   private:
    //! copy constructor
    MeshtyingStrategyStd(const MeshtyingStrategyStd& old);
  };  // class MeshtyingStrategyStd
}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
