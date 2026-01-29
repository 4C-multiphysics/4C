// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_noxinterface.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_constraint_group.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::NoxInterface::NoxInterface()
    : isinit_(false), issetup_(false), gstate_ptr_(nullptr), int_ptr_(nullptr), dbc_ptr_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::init(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
    const std::shared_ptr<Solid::Integrator>& int_ptr, const std::shared_ptr<Solid::Dbc>& dbc_ptr,
    const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr)
{
  // reset the setup flag
  issetup_ = false;

  gstate_ptr_ = gstate_ptr;
  timint_ptr_ = timint_ptr;
  int_ptr_ = int_ptr;
  dbc_ptr_ = dbc_ptr;

  // set the initialization flag
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::setup()
{
  check_init();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::check_init() const
{
  FOUR_C_ASSERT(is_init(), "Call init() first!");
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Integrator& Solid::TimeInt::NoxInterface::impl_int()
{
  check_init_setup();
  return *int_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::NoxInterface::compute_f(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& f, NOX::Nln::FillType fill_flag)
{
  check_init_setup();

  if (not int_ptr_->apply_force(x, f)) return false;

  /* Apply the DBC on the right hand side, since we need the Dirichlet free
   * right hand side inside NOX for the convergence check, etc.               */
  dbc_ptr_->apply_dirichlet_to_rhs(f);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::NoxInterface::compute_jacobian(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();

  if (not int_ptr_->apply_stiff(x, jac)) return false;

  /* We do not consider the jacobian DBC at this point. The Dirichlet conditions
   * are applied inside the NOX::Nln::LinearSystem::apply_jacobian_inverse()
   * routine, instead. See the run_pre_apply_jacobian_inverse() implementation
   * for more information.                               hiermeier 01/15/2016 */

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::NoxInterface::compute_f_and_jacobian(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();

  if (not int_ptr_->apply_force_stiff(x, rhs, jac)) return false;

  /* Apply the DBC on the right hand side, since we need the Dirichlet free
   * right hand side inside NOX for the convergence check, etc.               */
  dbc_ptr_->apply_dirichlet_to_rhs(rhs);

  /* We do not consider the jacobian DBC at this point. The Dirichlet conditions
   * are applied inside the NOX::Nln::LinearSystem::apply_jacobian_inverse()
   * routine, instead. See the run_pre_apply_jacobian_inverse() implementation
   * for more information.                               hiermeier 01/15/2016 */

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::NoxInterface::get_primary_rhs_norms(const Core::LinAlg::Vector<double>& F,
    const NOX::Nln::StatusTest::QuantityType& checkquantity,
    const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const
{
  check_init_setup();
  double rhsnorm = -1.0;

  // convert the given quantity type to a model type
  const Inpar::Solid::ModelType mt = Solid::Nln::convert_quantity_type2_model_type(checkquantity);

  switch (checkquantity)
  {
    case NOX::Nln::StatusTest::quantity_structure:
    case NOX::Nln::StatusTest::quantity_meshtying:
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
    {
      // export the model specific solution if necessary
      auto rhs_ptr = gstate_ptr_->extract_model_entries(mt, F);

      int_ptr_->remove_condensed_contributions_from_rhs(*rhs_ptr);

      rhsnorm = NOX::Nln::Aux::calc_vector_norm(*rhs_ptr, type, isscaled);

      break;
    }
    case NOX::Nln::StatusTest::quantity_pressure:
    case NOX::Nln::StatusTest::quantity_beaminteraction_lm:
    {
      // export the model specific solution if necessary
      auto rhs_ptr = gstate_ptr_->extract_model_entries(mt, F);

      rhsnorm = NOX::Nln::Aux::calc_vector_norm(*rhs_ptr, type, isscaled);

      break;
    }
    default:
    {
      /* Nothing to do. Functionality is supposed to be extended. */
      break;
    }
  }

  return rhsnorm;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::NoxInterface::get_primary_solution_update_rms(
    const Core::LinAlg::Vector<double>& xnew, const Core::LinAlg::Vector<double>& xold,
    const double& atol, const double& rtol, const NOX::Nln::StatusTest::QuantityType& checkquantity,
    const bool& disable_implicit_weighting) const
{
  check_init_setup();

  double rms = -1.0;

  // convert the given quantity type to a model type
  const Inpar::Solid::ModelType mt = Solid::Nln::convert_quantity_type2_model_type(checkquantity);

  switch (checkquantity)
  {
    case NOX::Nln::StatusTest::quantity_structure:
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
    case NOX::Nln::StatusTest::quantity_pressure:
    {
      // export the displacement solution if necessary
      auto model_incr_ptr = gstate_ptr_->extract_model_entries(mt, xold);
      auto model_xnew_ptr = gstate_ptr_->extract_model_entries(mt, xnew);

      model_incr_ptr->update(1.0, *model_xnew_ptr, -1.0);
      rms = NOX::Nln::Aux::root_mean_square_norm(
          atol, rtol, *model_xnew_ptr, *model_incr_ptr, disable_implicit_weighting);

      break;
    }
    case NOX::Nln::StatusTest::quantity_eas:
    {
      rms = int_ptr_->get_condensed_solution_update_rms(checkquantity);
      break;
    }
    default:
    {
      /* Nothing to do. Functionality is supposed to be extended. */
      break;
    }
  }

  return rms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::NoxInterface::get_primary_solution_update_norms(
    const Core::LinAlg::Vector<double>& xnew, const Core::LinAlg::Vector<double>& xold,
    const NOX::Nln::StatusTest::QuantityType& checkquantity,
    const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const
{
  check_init_setup();

  double updatenorm = -1.0;

  // convert the given quantity type to a model type
  const Inpar::Solid::ModelType mt = Solid::Nln::convert_quantity_type2_model_type(checkquantity);

  switch (checkquantity)
  {
    case NOX::Nln::StatusTest::quantity_structure:
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
    case NOX::Nln::StatusTest::quantity_pressure:
    case NOX::Nln::StatusTest::quantity_beaminteraction_lm:
    {
      // export the displacement solution if necessary
      auto model_incr_ptr = gstate_ptr_->extract_model_entries(mt, xold);
      auto model_xnew_ptr = gstate_ptr_->extract_model_entries(mt, xnew);

      model_incr_ptr->update(1.0, *model_xnew_ptr, -1.0);
      updatenorm = NOX::Nln::Aux::calc_vector_norm(*model_incr_ptr, type, isscaled);

      break;
    }
    case NOX::Nln::StatusTest::quantity_eas:
    {
      // get the update norm of the condensed quantities
      updatenorm = int_ptr_->get_condensed_update_norm(checkquantity);
      // do the scaling if desired
      if (isscaled)
      {
        int gdofnumber = int_ptr_->get_condensed_dof_number(checkquantity);
        updatenorm /= static_cast<double>(gdofnumber);
      }
      break;
    }
    default:
    {
      /* Nothing to do. Functionality is supposed to be extended. */
      break;
    }
  }

  return updatenorm;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::NoxInterface::get_previous_primary_solution_norms(
    const Core::LinAlg::Vector<double>& xold,
    const NOX::Nln::StatusTest::QuantityType& checkquantity,
    const ::NOX::Abstract::Vector::NormType& type, const bool& isscaled) const
{
  check_init_setup();

  double xoldnorm = -1.0;

  // convert the given quantity type to a model type
  const Inpar::Solid::ModelType mt = Solid::Nln::convert_quantity_type2_model_type(checkquantity);

  switch (checkquantity)
  {
    case NOX::Nln::StatusTest::quantity_structure:
    case NOX::Nln::StatusTest::quantity_cardiovascular0d:
    case NOX::Nln::StatusTest::quantity_pressure:
    case NOX::Nln::StatusTest::quantity_beaminteraction_lm:
    {
      // export the displacement solution if necessary
      auto model_xold_ptr = gstate_ptr_->extract_model_entries(mt, xold);

      xoldnorm = NOX::Nln::Aux::calc_vector_norm(*model_xold_ptr, type, isscaled);

      break;
    }
    case NOX::Nln::StatusTest::quantity_eas:
    {
      // get the update norm of the condensed quantities
      xoldnorm = int_ptr_->get_condensed_previous_sol_norm(checkquantity);
      if (isscaled)
      {
        int gdofnumber = int_ptr_->get_condensed_dof_number(checkquantity);
        xoldnorm /= static_cast<double>(gdofnumber);
      }
      break;
    }
    default:
    {
      /* Nothing to do. Functionality is supposed to be extended. */
      break;
    }
  }

  return xoldnorm;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::find_constraint_models(const ::NOX::Abstract::Group* grp,
    std::vector<Inpar::Solid::ModelType>& constraint_models) const
{
  const NOX::Nln::CONSTRAINT::Group* constr_grp =
      dynamic_cast<const NOX::Nln::CONSTRAINT::Group*>(grp);

  // direct return if this is no constraint problem
  if (not constr_grp) return;

  // find the constraint model types
  const auto& imap = constr_grp->get_constraint_interfaces();
  constraint_models.reserve(imap.size());

  for (auto cit = imap.begin(); cit != imap.end(); ++cit)
  {
    const NOX::Nln::SolutionType soltype = cit->first;
    const Inpar::Solid::ModelType mtype = Solid::Nln::convert_sol_type2_model_type(soltype);

    constraint_models.push_back(mtype);
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::NoxInterface::calc_ref_norm_force()
{
  check_init_setup();
  const auto& nox_normtype = timint_ptr_->get_data_sdyn().get_nox_norm_type();
  return int_ptr_->calc_ref_norm_force(nox_normtype);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix>
Solid::TimeInt::NoxInterface::calc_jacobian_contributions_from_element_level_for_ptc()
{
  check_init_setup();
  Teuchos::RCP<Core::LinAlg::SparseMatrix> scalingMatrixOpPtr =
      Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(*gstate_ptr_->dof_row_map(), 81, true, true);

  auto scalingMatrixOp = Core::Utils::shared_ptr_from_ref(*scalingMatrixOpPtr);
  int_ptr_->compute_jacobian_contributions_from_element_level_for_ptc(scalingMatrixOp);

  return scalingMatrixOpPtr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::create_backup_state(const Core::LinAlg::Vector<double>& dir)
{
  check_init_setup();
  int_ptr_->create_backup_state(dir);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::recover_from_backup_state()
{
  check_init_setup();
  int_ptr_->recover_from_backup_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::NoxInterface::get_dofs_from_elements(
    const std::vector<int>& my_ele_gids, std::set<int>& my_ele_dofs) const
{
  check_init_setup();

  std::shared_ptr<const Core::FE::Discretization> discret_ptr = gstate_ptr_->get_discret();

  for (int egid : my_ele_gids)
  {
    Core::Elements::Element* ele = discret_ptr->g_element(egid);
    Core::Nodes::Node** nodes = ele->nodes();

    for (int i = 0; i < ele->num_node(); ++i)
    {
      if (nodes[i]->owner() != Core::Communication::my_mpi_rank(gstate_ptr_->get_comm())) continue;

      const std::vector<int> ndofs(discret_ptr->dof(0, nodes[i]));
      my_ele_dofs.insert(ndofs.begin(), ndofs.end());
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
