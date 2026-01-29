// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_NOXINTERFACE_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_NOXINTERFACE_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"  // (2) base class: jacobian
#include "4C_solver_nonlin_nox_interface_required.hpp"  // (1) base class: rhs, status tests and more

FOUR_C_NAMESPACE_OPEN

// forward declaration ...
namespace NOX
{
  namespace Nln
  {
    enum class CorrectionType : int;
  }  // namespace Nln
}  // namespace NOX
namespace Core::LinAlg
{
  class SparseOperator;
  class SparseMatrix;
}  // namespace Core::LinAlg
namespace Inpar
{
  namespace Solid
  {
    enum ModelType : int;
  }  // namespace Solid
}  // namespace Inpar
namespace Solid
{
  class Dbc;
  class Integrator;
  namespace TimeInt
  {
    class Base;
    class BaseDataGlobalState;
    class NoxInterface : virtual public NOX::Nln::Interface::Required,
                         virtual public NOX::Nln::Interface::Jacobian
    {
     public:
      //! constructor
      NoxInterface();

      //! Init function
      virtual void init(const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
          const std::shared_ptr<Solid::Integrator>& int_ptr,
          const std::shared_ptr<Solid::Dbc>& dbc_ptr,
          const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr);

      virtual void setup();

      //!@{
      /*! compute the right hand side entries
       *  (derived from NOX::Nln::Interface::RequiredBase) */
      bool compute_f(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f,
          NOX::Nln::FillType fill_flag) override;

      /*! compute jacobian
       *  ( derived from NOX::Nln::Interface::JacobianBase) */
      bool compute_jacobian(
          const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) override;

      /*! compute right hand side and jacobian
       *  (derived from NOX::Nln::Interface::Jacobian) */
      bool compute_f_and_jacobian(const Core::LinAlg::Vector<double>& x,
          Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac) override;

      /*! Get the norm of residual rows/entries related to
       *  primary DoFs (derived from NOX::Nln::Interface::Required) */
      double get_primary_rhs_norms(const Core::LinAlg::Vector<double>& F,
          const NOX::Nln::StatusTest::QuantityType& checkquantity,
          const ::NOX::Abstract::Vector::NormType& type = ::NOX::Abstract::Vector::TwoNorm,
          const bool& isscaled = false) const override;

      /*! Get the root mean square of the solution update (vector) entries
       *  (derived from NOX::Nln::Interface::Required) */
      double get_primary_solution_update_rms(const Core::LinAlg::Vector<double>& xnew,
          const Core::LinAlg::Vector<double>& xold, const double& aTol, const double& rTol,
          const NOX::Nln::StatusTest::QuantityType& checkQuantity,
          const bool& disable_implicit_weighting = false) const override;

      /*! Returns the desired norm of the solution update (vector) entries
       *  (derived from NOX::Nln::Interface::Required) */
      double get_primary_solution_update_norms(const Core::LinAlg::Vector<double>& xnew,
          const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::StatusTest::QuantityType& checkquantity,
          const ::NOX::Abstract::Vector::NormType& type = ::NOX::Abstract::Vector::TwoNorm,
          const bool& isscaled = false) const override;

      /*! Return the desired norm of solution of the primary DoF fields from previous iteration
       *  (derived from NOX::Nln::Interface::Required) */
      double get_previous_primary_solution_norms(const Core::LinAlg::Vector<double>& xold,
          const NOX::Nln::StatusTest::QuantityType& checkquantity,
          const ::NOX::Abstract::Vector::NormType& type = ::NOX::Abstract::Vector::TwoNorm,
          const bool& isscaled = false) const override;

      /*! \brief calculate characteristic/reference norms for forces
       *
       *  Necessary for the LinearSystem objects.
       *  (derived from NOX::Nln::Interface::Required) */
      double calc_ref_norm_force() override;

      /// create back-up state of condensed solution variables (e.g. EAS)
      void create_backup_state(const Core::LinAlg::Vector<double>& dir) override;

      /// recover from back-up
      void recover_from_backup_state() override;

      /// fill the sets with DOFs of the desired elements
      void get_dofs_from_elements(
          const std::vector<int>& my_ele_gids, std::set<int>& my_ele_dofs) const override;

      //!@}

      // Get element based scaling operator
      Teuchos::RCP<Core::LinAlg::SparseMatrix>
      calc_jacobian_contributions_from_element_level_for_ptc() override;

      //! Access the implicit integrator
      Solid::Integrator& impl_int();

     protected:
      //! Returns the init state
      inline const bool& is_init() const { return isinit_; };

      //! Returns the setup state
      inline const bool& is_setup() const { return issetup_; };

      //! check if init has been called
      void check_init() const;

      //! check if init and setup have been called
      void check_init_setup() const;

      void find_constraint_models(const ::NOX::Abstract::Group* grp,
          std::vector<Inpar::Solid::ModelType>& constraint_models) const;

     protected:
      //! init flag
      bool isinit_;

      //! setup flag
      bool issetup_;

     private:
      //! global state data container
      std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;

      std::shared_ptr<const Solid::TimeInt::Base> timint_ptr_;

      std::shared_ptr<Solid::Integrator> int_ptr_;

      std::shared_ptr<Solid::Dbc> dbc_ptr_;
    };  // class nox_interface
  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
