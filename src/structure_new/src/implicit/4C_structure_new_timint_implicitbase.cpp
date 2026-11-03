// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_implicitbase.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_structure_new_integrator.hpp"

#include <NOX_Abstract_Group.H>
#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::ImplicitBase::ImplicitBase()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> Solid::TimeInt::ImplicitBase::get_f() const
{
  const ::NOX::Abstract::Group& solgrp = get_solution_group();
  const auto& F = dynamic_cast<const NOX::Nln::Vector&>(solgrp.getF());
  return get_data_global_state().extract_displ_entries(F.get_linalg_vector());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Solid::TimeInt::ImplicitBase::freact()
{
  check_init_setup();
  return data_global_state().get_freact_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Solid::TimeInt::ImplicitBase::system_matrix()
{
  check_init_setup();
  return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(data_global_state().get_jacobian());
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
Solid::TimeInt::ImplicitBase::block_system_matrix()
{
  check_init_setup();
  return std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
      data_global_state().get_jacobian());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::ImplicitBase::use_block_matrix(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps)
{
  FOUR_C_THROW("Currently disabled!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::StcScale Solid::TimeInt::ImplicitBase::get_stc_algo()
{
  return data_sdyn().get_stc_algo_type();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Solid::TimeInt::ImplicitBase::get_stc_mat()
{
  FOUR_C_THROW("Not yet implemented!");
  /* See the scaling object in the NOX::Nln::LinearSystem class.
   * The STC matrix has to be implemented as a scaling object or as a
   * preconditioner. Both are part of the linear system. */
  // group->linearsystem->scalingobject
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> Solid::TimeInt::ImplicitBase::initial_guess()
{
  check_init_setup();
  FOUR_C_THROW("Not yet implemented!");
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::ImplicitBase::update(double endtime)
{
  check_init_setup();
  pre_update();
  integrator().apply_element_erosion();
  integrator().update_step_state();
  set_time_np(endtime);
  update_step_time();
  integrator().update_step_element();
  post_update();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::ImplicitBase::print_step()
{
  check_init_setup();

  if (data_global_state().get_my_rank() != 0 or group_id() != 0) return;

  const int stepmax = data_sdyn().get_step_max();
  const int stepn = data_global_state().get_step_n();
  const double& timen = data_global_state().get_time_n();
  const double& dt = data_global_state().get_delta_time()[0];
  const int newtoniter = data_global_state().get_nln_iteration_number(stepn);
  double wct = data_global_state().get_timer()->totalElapsedTime(true);

  // open outstd::stringstream
  std::ostringstream oss;

  /* Output of the following quantities
   * time   : total simulated time
   * dt     : used time step
   * nlniter: number of nonlinear solver steps
   * wct    : wall clock time */
  oss << "Finalised step " << std::setw(1) << stepn;
  oss << " / " << std::setw(1) << stepmax;
  oss << " | time " << std::setw(9) << std::setprecision(3) << std::scientific << timen;
  oss << " | dt " << std::setw(9) << std::setprecision(3) << std::scientific << dt;
  oss << " | nlniter " << std::setw(1) << newtoniter;
  oss << " | wct " << std::setw(8) << std::setprecision(2) << std::scientific << wct;
  oss << "\n--------------------------------------------------------------------------------\n";

  // print to ofile (could be done differently...)
  fprintf(stdout, "%s\n", oss.str().c_str());

  // print it, now
  fflush(stdout);
}

FOUR_C_NAMESPACE_CLOSE
