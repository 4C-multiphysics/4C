// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_dbc.hpp"

#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_linearsystem_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_structure_new_timint_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Dbc::Dbc()
    : isinit_(false),
      issetup_(false),
      islocsys_(false),
      discret_ptr_(nullptr),
      timint_ptr_(nullptr),
      locsysman_ptr_(nullptr),
      zeros_ptr_(nullptr),
      dbcmap_ptr_(nullptr),
      freact_ptr_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::init(const std::shared_ptr<Core::FE::Discretization>& discret_ptr,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& freact_ptr,
    const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr)
{
  // reset the setup indicator
  issetup_ = false;

  discret_ptr_ = discret_ptr;
  freact_ptr_ = freact_ptr.get();
  timint_ptr_ = timint_ptr;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::setup()
{
  check_init();
  // ---------------------------------------------------------------------------
  // Create Dirichlet Boundary Condition map
  // ---------------------------------------------------------------------------
  zeros_ptr_ = std::make_shared<Core::LinAlg::Vector<double>>(*g_state().dof_row_map_view(), true);
  Teuchos::ParameterList p;
  p.set<double>("total time", timint_ptr_->get_data_global_state().get_time_np());
  dbcmap_ptr_ = std::make_shared<Core::LinAlg::MapExtractor>();
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());
  discret_ptr_->evaluate_dirichlet(p, zeros_ptr_, nullptr, nullptr, nullptr, dbcmap_ptr_);
  // clear the system vector of possibly inserted non-zero DBC values
  zeros_ptr_->put_scalar(0.0);

  // ---------------------------------------------------------------------------
  // Create local coordinate system manager
  // ---------------------------------------------------------------------------
  std::vector<const Core::Conditions::Condition*> locsysconditions;
  discret_ptr_->get_condition("Locsys", locsysconditions);
  if (locsysconditions.size())
  {
    locsysman_ptr_ = std::make_shared<Core::Conditions::LocsysManager>(
        *discret_ptr_, Global::Problem::instance()->n_dim());
    // in case we have no time dependent locsys conditions in our problem,
    // this is the only time where the whole setup routine is conducted.
    locsysman_ptr_->update(-1.0, {}, Global::Problem::instance()->function_manager());
    islocsys_ = true;
  }

  // ---------------------------------------------------------------------------
  // Set the new pre/post operator for the nox nln linearsystem in the parameter
  // list
  // ---------------------------------------------------------------------------
  const Teuchos::ParameterList& pnox = timint_ptr_->get_data_sdyn().get_nox_params();
  if (pnox.sublist("Direction").isSublist("Newton"))
  {
    if (pnox.sublist("Direction").sublist("Newton").isSublist("Linear Solver"))
    {
      // get a mutable reference to the linear solver parameter list
      Teuchos::ParameterList& p_linsolver = const_cast<Teuchos::ParameterList&>(
          pnox.sublist("Direction").sublist("Newton").sublist("Linear Solver"));
      NOX::Nln::LinSystem::PrePostOperator::Map& prepostlinsystem_map =
          NOX::Nln::LinSystem::PrePostOp::get_map(p_linsolver);
      // create the new pre/post operator for the nox nln linear system
      Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator> prepostdbc_ptr =
          Teuchos::make_rcp<NOX::Nln::LinSystem::PrePostOp::Dbc>(Teuchos::rcpFromRef(*this));
      // insert/replace the old pointer in the map
      prepostlinsystem_map[NOX::Nln::LinSystem::prepost_dbc] = prepostdbc_ptr;
    }
    else
      FOUR_C_THROW(
          "There is no \"[NOX]->[Direction]->[Newton]->[Linear Solver] "
          "sub-sub-sublist!");
  }
  else
    FOUR_C_THROW("There is no \"[NOX]->[Direction]->[Newton]\" sub-sublist!");

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::check_init() const { FOUR_C_ASSERT(is_init(), "Call init() first!"); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization> Solid::Dbc::discret_ptr()
{
  check_init();
  return discret_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::FE::Discretization> Solid::Dbc::discret_ptr() const
{
  check_init();
  return std::dynamic_pointer_cast<const Core::FE::Discretization>(discret_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::update_loc_sys_manager()
{
  if (!is_loc_sys()) return;

  discret_ptr()->set_state("dispnp", *g_state().get_dis_np());
  locsysman_ptr_->update(
      g_state().get_time_np(), {}, Global::Problem::instance()->function_manager());
  discret_ptr()->clear_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Solid::Dbc::get_dirichlet_increment()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> disn =
      timint_ptr_->get_data_global_state().get_dis_n();
  std::shared_ptr<Core::LinAlg::Vector<double>> dbcincr =
      std::make_shared<Core::LinAlg::Vector<double>>(*disn);
  const double& timenp = g_state().get_time_np();

  // get the new value for the Dirichlet DOFs
  apply_dirichlet_bc(timenp, dbcincr, nullptr, nullptr, false);

  /* Subtract the displacements of the last converged step:
   * --> DBC-DOFs hold increments of current step
   * --> free-DOFs hold zeros. */
  dbcincr->update(-1.0, *disn, 1.0);

  return dbcincr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::apply_dirichlet_bc(const double& time,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::Vector<double>> vel,
    std::shared_ptr<Core::LinAlg::Vector<double>> acc, bool recreatemap) const
{
  check_init_setup();
  // We have to rotate forward ...
  // ---------------------------------------------------------------------------
  if (dis) rotate_global_to_local(*dis, true);
  if (vel) rotate_global_to_local(*vel);
  if (acc) rotate_global_to_local(*acc);

  // Apply DBCs
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList p;
  p.set("total time", time);
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // predicted Dirichlet values
  // \c dis then also holds prescribed new Dirichlet displacements
  discret_ptr_->clear_state();
  if (recreatemap)
    discret_ptr_->evaluate_dirichlet(p, dis, vel, acc, nullptr, dbcmap_ptr_);
  else
    discret_ptr_->evaluate_dirichlet(p, dis, vel, acc, nullptr, nullptr);

  discret_ptr_->clear_state();

  // We have to rotate back into global Cartesian frame
  // ---------------------------------------------------------------------------
  if (dis != nullptr) rotate_local_to_global(*dis, true);
  if (vel != nullptr) rotate_local_to_global(*vel);
  if (acc != nullptr) rotate_local_to_global(*acc);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::apply_dirichlet_to_local_system(
    Core::LinAlg::SparseOperator& A, Core::LinAlg::Vector<double>& b) const
{
  check_init_setup();
  apply_dirichlet_to_local_rhs(b);
  apply_dirichlet_to_local_jacobian(A);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::apply_dirichlet_to_vector(Core::LinAlg::Vector<double>& vec) const
{
  check_init_setup();
  // rotate the coordinate system if desired
  rotate_global_to_local(vec);
  // apply the dbc
  Core::LinAlg::apply_dirichlet_to_system(vec, *zeros_ptr_, *(dbcmap_ptr_->cond_map()));
  // rotate back
  rotate_local_to_global(vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::apply_dirichlet_to_local_rhs(Core::LinAlg::Vector<double>& b) const
{
  check_init_setup();

  // rotate the coordinate system: global --> local
  rotate_global_to_local(b);

  extract_freact(b);
  Core::LinAlg::apply_dirichlet_to_system(b, *zeros_ptr_, *(dbcmap_ptr_->cond_map()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::apply_dirichlet_to_rhs(Core::LinAlg::Vector<double>& b) const
{
  check_init_setup();

  apply_dirichlet_to_local_rhs(b);

  // rotate back: local --> global
  rotate_local_to_global(b);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::apply_dirichlet_to_local_jacobian(Core::LinAlg::SparseOperator& A) const
{
  check_init_setup();
  // don't do it twice...
  /* Note: If the DBCs are applied twice this can lead to an error and strange
   * behavior during the usage of locsys. Furthermore, the consideration of
   * DBCs in an explicit way is a pretty expensive operation.
   *                                                          hiermeier 01/18 */
  if (Core::LinAlg::is_dirichlet_boundary_condition_already_applied(
          A, *dbcmap_ptr_->cond_map(), true, get_loc_sys_trafo()))
    return;
  if (rotate_global_to_local(A))
  {
    std::shared_ptr<std::vector<Core::LinAlg::SparseMatrix*>> mats =
        g_state().extract_displ_row_of_blocks(A);

    for (unsigned i = 0; i < mats->size(); ++i)
    {
      Core::LinAlg::SparseMatrix& mat = *(*mats)[i];

      mat.apply_dirichlet_with_trafo(
          *get_loc_sys_trafo(), *(dbcmap_ptr_->cond_map()), (i == 0), false);
    }

    if (not A.filled()) A.complete();
  }
  else
    A.apply_dirichlet(*(dbcmap_ptr_->cond_map()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Dbc::rotate_global_to_local(Core::LinAlg::Vector<double>& v) const
{
  check_init_setup();
  return rotate_global_to_local(v, false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Dbc::rotate_global_to_local(Core::LinAlg::Vector<double>& v, bool offset) const
{
  check_init_setup();
  if (not is_loc_sys()) return false;

  if (g_state().max_block_number() > 1)
  {
    Core::LinAlg::Vector<double> v_displ(*g_state().dof_row_map_view());
    Core::LinAlg::extract_my_vector(v, v_displ);

    locsysman_ptr_->rotate_global_to_local(v_displ, offset);

    Core::LinAlg::assemble_my_vector(0.0, v, 1.0, v_displ);
  }
  else
    locsysman_ptr_->rotate_global_to_local(v, offset);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Dbc::rotate_global_to_local(Core::LinAlg::SparseOperator& A) const
{
  check_init_setup();
  if (not is_loc_sys()) return false;

  std::shared_ptr<std::vector<Core::LinAlg::SparseMatrix*>> mats =
      g_state().extract_displ_row_of_blocks(A);

  for (unsigned i = 0; i < mats->size(); ++i)
    locsysman_ptr_->rotate_global_to_local(Core::Utils::shared_ptr_from_ref(*(*mats)[i]));

  if (not A.filled()) A.complete();

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Dbc::rotate_local_to_global(Core::LinAlg::Vector<double>& v) const
{
  check_init_setup();
  return rotate_local_to_global(v, false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Dbc::rotate_local_to_global(Core::LinAlg::Vector<double>& v, bool offset) const
{
  check_init_setup();
  if (not is_loc_sys()) return false;

  if (g_state().max_block_number() > 1)
  {
    Core::LinAlg::Vector<double> v_displ(*g_state().dof_row_map_view());
    Core::LinAlg::extract_my_vector(v, v_displ);

    locsysman_ptr_->rotate_local_to_global(v_displ, offset);

    Core::LinAlg::assemble_my_vector(0.0, v, 1.0, v_displ);
  }
  else
    locsysman_ptr_->rotate_local_to_global(v, offset);

  // reset flag
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::SparseMatrix> Solid::Dbc::get_loc_sys_trafo() const
{
  check_init_setup();
  if (not is_loc_sys()) return nullptr;

  return locsysman_ptr_->trafo();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::extract_freact(Core::LinAlg::Vector<double>& b) const
{
  check_init_setup();

  Core::LinAlg::extract_my_vector(b, freact());
  freact().scale(-1.0);

  // put zeros on all non-DBC dofs
  insert_vector_in_non_dbc_dofs(zeros_ptr_, Core::Utils::shared_ptr_from_ref(freact()));

  // turn the reaction forces back to the global coordinate system if necessary
  rotate_local_to_global(freact());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::insert_vector_in_non_dbc_dofs(
    std::shared_ptr<const Core::LinAlg::Vector<double>> source_ptr,
    std::shared_ptr<Core::LinAlg::Vector<double>> target_ptr) const
{
  check_init_setup();
  dbcmap_ptr_->insert_other_vector(*dbcmap_ptr_->extract_other_vector(*source_ptr), *target_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MapExtractor> Solid::Dbc::get_dbc_map_extractor() const
{
  check_init_setup();
  return dbcmap_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::Conditions::LocsysManager> Solid::Dbc::loc_sys_manager_ptr()
{
  check_init_setup();
  return locsysman_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::Dbc::get_zeros() const
{
  check_init_setup();
  return *zeros_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> Solid::Dbc::get_zeros_ptr() const
{
  check_init_setup();
  return zeros_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Vector<double>& Solid::Dbc::freact() const
{
  FOUR_C_ASSERT(freact_ptr_, "nullptr");

  return *freact_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataGlobalState& Solid::Dbc::g_state() const
{
  return timint_ptr_->get_data_global_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::add_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoadd)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(dbcmap_ptr_->cond_map());
  std::shared_ptr<Core::LinAlg::Map> condmerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);
  *dbcmap_ptr_ = Core::LinAlg::MapExtractor(*(discret_ptr_->dof_row_map()), condmerged);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Dbc::remove_dirich_dofs(const std::shared_ptr<const Core::LinAlg::Map> maptoremove)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> othermaps;
  othermaps.push_back(maptoremove);
  othermaps.push_back(dbcmap_ptr_->other_map());
  std::shared_ptr<Core::LinAlg::Map> othermerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(othermaps);
  *dbcmap_ptr_ = Core::LinAlg::MapExtractor(*(discret_ptr_->dof_row_map()), othermerged, false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::PrePostOp::Dbc::Dbc(const Teuchos::RCP<const Solid::Dbc>& dbc_ptr)
    : dbc_ptr_(dbc_ptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::LinSystem::PrePostOp::Dbc::run_pre_apply_jacobian_inverse(
    ::NOX::Abstract::Vector& rhs, Core::LinAlg::SparseOperator& jac,
    const NOX::Nln::LinearSystem& linsys)
{
  auto& rhs_nox = dynamic_cast<NOX::Nln::Vector&>(rhs);
  std::shared_ptr<Core::LinAlg::SparseOperator> jac_ptr = Core::Utils::shared_ptr_from_ref(jac);
  // apply the dirichlet condition and rotate the system if desired
  dbc_ptr_->apply_dirichlet_to_local_system(*jac_ptr, rhs_nox.get_linalg_vector());
}

FOUR_C_NAMESPACE_CLOSE
