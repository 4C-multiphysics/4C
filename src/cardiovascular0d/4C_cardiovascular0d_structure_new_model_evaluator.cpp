// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_structure_new_model_evaluator.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_linalg.hpp"
#include "4C_linalg_block_pod_projector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_io.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

#include <filesystem>


FOUR_C_NAMESPACE_OPEN

namespace
{
  bool is_pod_basis_orthogonal(const Core::LinAlg::MultiVector<double>& M, const double tol = 1e-7)
  {
    const int n = M.num_vectors();

    // calculate V^T * V (should be an nxn identity matrix)
    Core::LinAlg::Map map = Core::LinAlg::Map(n, n, 0, M.get_map().get_comm());
    Core::LinAlg::MultiVector<double> identity = Core::LinAlg::MultiVector<double>(map, n, true);
    identity.multiply('T', 'N', 1.0, M, M, 0.0);

    // subtract one from diagonal
    for (int i = 0; i < n; ++i) identity.sum_into_global_value(i, i, -1.0);

    // inf norm of columns
    std::vector<double> norms(n, 0.0);
    identity.norm_inf(norms.data());

    for (int i = 0; i < n; ++i)
      if (norms[i] > tol) return false;

    return true;
  }

  Core::LinAlg::SparseMatrix make_projection_matrix(
      const std::filesystem::path& pod_matrix_file_name, const Core::LinAlg::Map& full_map)
  {
    Core::LinAlg::MultiVector<double> reduced_basis =
        Core::LinAlg::read_matrix_market_file_as_multi_vector(pod_matrix_file_name, full_map);

    Core::LinAlg::MultiVector<double> projection_matrix(
        full_map, reduced_basis.num_vectors(), true);

    Core::LinAlg::Import dofrowimporter(full_map, reduced_basis.get_map());
    projection_matrix.import(reduced_basis, dofrowimporter, Core::LinAlg::CombineMode::insert);

    // check row dimension
    FOUR_C_ASSERT_ALWAYS(projection_matrix.global_length() == full_map.num_global_elements(),
        "Projection matrix does not match discretization.");

    // check orthogonality
    FOUR_C_ASSERT_ALWAYS(
        is_pod_basis_orthogonal(projection_matrix), "Projection matrix is not orthogonal.");

    const Core::LinAlg::Map reduced_map(projection_matrix.num_vectors(), 0, full_map.get_comm());

    Core::LinAlg::SparseMatrix projection_sparse_matrix{
        full_map, reduced_basis.num_vectors(), false, true};
    Core::LinAlg::multi_vector_to_linalg_sparse_matrix(
        projection_matrix, full_map, reduced_map, projection_sparse_matrix);

    return projection_sparse_matrix;
  }
}  // namespace

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::Cardiovascular0D::Cardiovascular0D()
    : disnp_ptr_(nullptr), stiff_cardio_ptr_(nullptr), fstructcardio_np_ptr_(nullptr)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::setup()
{
  check_init();

  std::shared_ptr<Core::FE::Discretization> dis = discret_ptr();

  // setup the displacement pointer
  disnp_ptr_ = global_state().get_dis_np();

  // contributions of 0D model to structural rhs and stiffness
  fstructcardio_np_ptr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view());
  stiff_cardio_ptr_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *global_state().dof_row_map_view(), 81, true, true);

  Teuchos::ParameterList solvparams;
  Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
      "SOLVER", Core::LinearSolver::SolverType::umfpack, solvparams);
  std::shared_ptr<Core::LinAlg::Solver> dummysolver(new Core::LinAlg::Solver(
      solvparams, disnp_ptr_->get_comm(), nullptr, Core::IO::Verbositylevel::standard));

  // ToDo: we do not want to hand in the structural dynamics parameter list
  // to the manager in the future! -> get rid of it as soon as old
  // time-integration dies ...
  // initialize 0D cardiovascular manager
  cardvasc0dman_ = std::make_shared<Utils::Cardiovascular0DManager>(dis, disnp_ptr_,
      Global::Problem::instance()->structural_dynamic_params(),
      Global::Problem::instance()->cardiovascular0_d_structural_params(), *dummysolver, nullptr);

  const auto& pod_matrix =
      Global::Problem::instance()->mor_params().get<std::optional<std::filesystem::path>>(
          "POD_MATRIX");
  if (pod_matrix.has_value())
  {
    // build block POD projector
    std::vector<std::variant<Core::LinAlg::PODProjectionBlock, Core::LinAlg::NoProjectionBlock>>
        block_projections;
    block_projections.reserve(2);

    block_projections.emplace_back(Core::LinAlg::PODProjectionBlock{
        make_projection_matrix(*pod_matrix, *global_state().dof_row_map())});
    block_projections.emplace_back(
        Core::LinAlg::NoProjectionBlock{.range_map = *cardvasc0dman_->get_cardiovascular0_d_map(),
            .domain_map = *cardvasc0dman_->get_cardiovascular0_d_map()});

    // Add pod projector to the solver parameters
    [[maybe_unused]] Core::LinAlg::Solver& solver =
        *eval_data().sdyn().get_lin_solvers().at(Inpar::Solid::model_cardiovascular0d);
    solver.params().set<std::shared_ptr<Core::LinAlg::LinearSystemProjector>>(
        "Projector", std::make_shared<Core::LinAlg::BlockPODProjector>(block_projections));
  }
  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::reset(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  // update the structural displacement vector
  disnp_ptr_ = global_state().get_dis_np();

  fstructcardio_np_ptr_->put_scalar(0.0);
  stiff_cardio_ptr_->zero();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Cardiovascular0D::evaluate_force()
{
  check_init_setup();

  double time_np = global_state().get_time_np();
  Teuchos::ParameterList pcardvasc0d;
  pcardvasc0d.set("time_step_size", global_state().get_delta_time()[0]);

  // only forces are evaluated!
  cardvasc0dman_->evaluate_force_stiff(
      time_np, disnp_ptr_, fstructcardio_np_ptr_, nullptr, pcardvasc0d);

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Cardiovascular0D::evaluate_stiff()
{
  check_init_setup();

  double time_np = global_state().get_time_np();
  Teuchos::ParameterList pcardvasc0d;
  pcardvasc0d.set("time_step_size", global_state().get_delta_time()[0]);

  // only stiffnesses are evaluated!
  cardvasc0dman_->evaluate_force_stiff(
      time_np, disnp_ptr_, nullptr, stiff_cardio_ptr_, pcardvasc0d);

  if (not stiff_cardio_ptr_->filled()) stiff_cardio_ptr_->complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Cardiovascular0D::evaluate_force_stiff()
{
  check_init_setup();

  double time_np = global_state().get_time_np();
  Teuchos::ParameterList pcardvasc0d;
  pcardvasc0d.set("time_step_size", global_state().get_delta_time()[0]);

  cardvasc0dman_->evaluate_force_stiff(
      time_np, disnp_ptr_, fstructcardio_np_ptr_, stiff_cardio_ptr_, pcardvasc0d);

  if (not stiff_cardio_ptr_->filled()) stiff_cardio_ptr_->complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Cardiovascular0D::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> block_vec_ptr = nullptr;

  // assemble and scale with str time-integrator dependent value
  Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, *fstructcardio_np_ptr_);

  // assemble 0D model rhs - already at the generalized mid-point t_{n+theta} !
  block_vec_ptr = cardvasc0dman_->get_cardiovascular0_drhs();

  if (!block_vec_ptr)
    FOUR_C_THROW(
        "The 0D cardiovascular model vector is a nullptr pointer, although \n"
        "the structural part indicates, that 0D cardiovascular model contributions \n"
        "are present!");

  Core::LinAlg::assemble_my_vector(1.0, f, 1.0, *block_vec_ptr);

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Cardiovascular0D::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> block_ptr = nullptr;

  // --- Kdd - block - scale with str time-integrator dependent value---
  std::shared_ptr<Core::LinAlg::SparseMatrix> jac_dd_ptr = global_state().extract_displ_block(jac);
  Core::LinAlg::matrix_add(*stiff_cardio_ptr_, false, timefac_np, *jac_dd_ptr, 1.0);
  // no need to keep it
  stiff_cardio_ptr_->zero();

  // --- Kdz - block ---------------------------------------------------
  block_ptr = cardvasc0dman_->get_mat_dstruct_dcv0ddof();
  // scale with str time-integrator dependent value
  block_ptr->scale(timefac_np);
  global_state().assign_model_block(jac, *block_ptr, type(), MatBlockType::displ_lm);
  // reset the block pointer, just to be on the safe side
  block_ptr = nullptr;

  // --- Kzd - block - already scaled correctly by 0D model !-----------
  block_ptr = Core::LinAlg::matrix_transpose(*cardvasc0dman_->get_mat_dcardvasc0d_dd());
  global_state().assign_model_block(jac, *block_ptr, type(), MatBlockType::lm_displ);
  // reset the block pointer, just to be on the safe side
  block_ptr = nullptr;

  // --- Kzz - block - already scaled with 0D theta by 0D model !-------
  block_ptr = cardvasc0dman_->get_cardiovascular0_d_stiffness();
  global_state().assign_model_block(jac, *block_ptr, type(), MatBlockType::lm_lm);
  // reset the block pointer, just to be on the safe side
  block_ptr = nullptr;

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  iowriter.write_vector("cv0d_df_np", cardvasc0dman_->get0_d_df_np());
  iowriter.write_vector("cv0d_f_np", cardvasc0dman_->get0_d_f_np());

  iowriter.write_vector("cv0d_dof_np", cardvasc0dman_->get0_d_dof_np());
  iowriter.write_vector("vol_np", cardvasc0dman_->get0_d_vol_np());

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  double time_n = global_state().get_time_n();
  cardvasc0dman_->read_restart(ioreader, time_n);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::run_post_compute_x(
    const Core::LinAlg::Vector<double>& xold, const Core::LinAlg::Vector<double>& dir,
    const Core::LinAlg::Vector<double>& xnew)
{
  check_init_setup();

  std::shared_ptr<Core::LinAlg::Vector<double>> cv0d_incr =
      global_state().extract_model_entries(Inpar::Solid::model_cardiovascular0d, dir);

  cardvasc0dman_->update_cv0_d_dof(*cv0d_incr);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::update_step_state(const double& timefac_n)
{
  // only update 0D model at the end of the time step!
  if (eval_data().get_total_time() == global_state().get_time_np())
    cardvasc0dman_->update_time_step();

  // only print state variables after a finished time step, not when we're
  // in the equilibriate initial state routine
  if (eval_data().get_total_time() == global_state().get_time_np())
    cardvasc0dman_->print_pres_flux(false);

  // add the 0D cardiovascular force contributions to the old structural
  // residual state vector
  if (fstructcardio_np_ptr_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>>& fstructold_ptr =
        global_state().get_fstructure_old();
    fstructold_ptr->update(timefac_n, *fstructcardio_np_ptr_, 1.0);
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::update_step_element()
{
  // nothing to do
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::determine_stress_strain()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::determine_energy()
{
  // nothing to do
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::reset_step_state()
{
  check_init_setup();

  FOUR_C_THROW("Not yet implemented");

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
Solid::ModelEvaluator::Cardiovascular0D::get_block_dof_row_map_ptr() const
{
  check_init_setup();

  return cardvasc0dman_->get_cardiovascular0_d_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::Cardiovascular0D::get_current_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::Cardiovascular0D::get_last_time_step_solution_ptr() const
{
  // there are no model specific solution entries
  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::Cardiovascular0D::post_output()
{
  check_init_setup();
  // empty

  return;
}  // post_output()


FOUR_C_NAMESPACE_CLOSE
