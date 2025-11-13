// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_GLOBALDATA_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_GLOBALDATA_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_constraint_interface_preconditioner.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::LinAlg
{
  class Solver;
}

namespace NOX
{
  namespace Nln
  {
    namespace Interface
    {
      class RequiredBase;
    }  // namespace Interface
    namespace CONSTRAINT
    {
      namespace Interface
      {
        class Required;
      }  // namespace Interface
    }  // namespace CONSTRAINT

    class Scaling;

    class GlobalData
    {
     public:
      /*! CONSTRAINED OPTIMIZATION (standard constructor / most general case)
       *  inclusive the constraint interfaces map
       *  inclusive the pre-conditioner interfaces
       *  inclusive scaling object */
      GlobalData(MPI_Comm comm, Teuchos::ParameterList& noxParams,
          const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& linSolvers,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
          const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
          const OptimizationProblemType& type, const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
          const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
          const std::shared_ptr<NOX::Nln::Scaling>& iscale);

      /*! CONSTRAINED OPTIMIZATION
       * inclusive the constraint interfaces map
       * without any pre-conditioner interfaces */
      GlobalData(MPI_Comm comm, Teuchos::ParameterList& noxParams,
          const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& linSolvers,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
          const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
          const OptimizationProblemType& type,
          const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr);

      /*! UNCONSTRAINED OPTIMIZATION
       *  constructor without the constraint interface map (pure unconstrained optimization)
       *  without a pre-conditioner interface */
      GlobalData(MPI_Comm comm, Teuchos::ParameterList& noxParams,
          const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& linSolvers,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
          const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac);

      //! destructor
      virtual ~GlobalData() = default;

      //! return the nox_utils class
      const ::NOX::Utils& get_nox_utils() const;

      //! return the nox_utils class pointer
      const Teuchos::RCP<::NOX::Utils>& get_nox_utils_ptr() const;

      //! return the nln parameter list
      const Teuchos::ParameterList& get_nln_parameter_list() const;
      Teuchos::ParameterList& get_nln_parameter_list();

      //! return the pointer to the parameter list
      const Teuchos::RCP<Teuchos::ParameterList>& get_nln_parameter_list_ptr();

      //! return underlying discretization MPI_Comm
      MPI_Comm get_comm() const;

      //! return the isConstrained boolean
      //! true if in/equality constrained optimization problem
      //! false if unconstrained optimization problem
      bool is_constrained() const;

      // return linear solver vector
      const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>&
      get_linear_solvers();

      //! return the user-defined required interface
      std::shared_ptr<NOX::Nln::Interface::RequiredBase> get_required_interface();

      //! return the user-defined jacobian interface
      Teuchos::RCP<::NOX::Epetra::Interface::Jacobian> get_jacobian_interface();

      //! return the user-defined constraint interface map
      const NOX::Nln::CONSTRAINT::ReqInterfaceMap& get_constraint_interfaces();

      //! return the user-defined constraint preconditioner interface map
      const NOX::Nln::CONSTRAINT::PrecInterfaceMap& get_constraint_prec_interfaces();

      //! return linear system scaling object
      const std::shared_ptr<NOX::Nln::Scaling>& get_scaling_object();

     private:
      //! setup the nln_utils class
      void setup();

      //! check the constructor input
      void check_input() const;

      /*! \brief set printing parameters
       *
       * translate input file input into NOX input */
      void set_printing_parameters();

      //! set solver option parameters
      void set_solver_option_parameters();

      //! set status test parameters
      void set_status_test_parameters();

     private:
      /// communicator
      MPI_Comm comm_;

      /// complete NOX::NLN parameter list
      Teuchos::RCP<Teuchos::ParameterList> nlnparams_;

      /// optimization problem type (unconstrained, constrained, etc.)
      OptimizationProblemType opt_type_;

      /// map containing all linear solvers
      const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>> lin_solvers_;

      /// required interface pointer
      std::shared_ptr<NOX::Nln::Interface::RequiredBase> i_req_ptr_;

      /// jacobian interface pointer
      Teuchos::RCP<::NOX::Epetra::Interface::Jacobian> i_jac_ptr_;

      /// map of required interface pointer for constrained problems
      NOX::Nln::CONSTRAINT::ReqInterfaceMap i_constr_;

      /// map of preconditioner interface pointer for constrained problems
      NOX::Nln::CONSTRAINT::PrecInterfaceMap i_constr_prec_;

      /// scaling object (for the linear system)
      std::shared_ptr<NOX::Nln::Scaling> i_scale_;

      /// merit function pointer
      Teuchos::RCP<::NOX::MeritFunction::Generic> mrt_fct_ptr_;

      /// user provided direction factory
      Teuchos::RCP<::NOX::Direction::UserDefinedFactory> direction_factory_;

      /// pre/post operator pointer for the NOX::Nln::Solver pre/post operator
      Teuchos::RCP<::NOX::Observer> pre_post_op_ptr_;

      /// True if it is a constrained problem
      bool is_constrained_;

      /// output object
      Teuchos::RCP<::NOX::Utils> nox_utils_;
    };  // namespace GlobalData
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
