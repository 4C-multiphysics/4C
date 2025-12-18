// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_model_evaluator_structure.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_discretization_runtime_output_params.hpp"
#include "4C_beam3_discretization_runtime_vtu_writer.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_utils_gauss_point_postprocess.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_discretization_runtime_output_params.hpp"
#include "4C_structure_new_error_evaluator.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_predict_generic.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtk_output.hpp"
#include "4C_structure_new_timint_implicit.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Structure::Structure()
    : dt_ele_ptr_(nullptr),
      masslin_type_(Inpar::Solid::MassLin::ml_none),
      stiff_ptr_(nullptr),
      stiff_ptc_ptr_(nullptr),
      dis_incr_ptr_(nullptr),
      vtu_writer_ptr_(nullptr),
      beam_vtu_writer_ptr_(nullptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::setup()
{
  FOUR_C_ASSERT(is_init(), "init() has not been called, yet!");

  // get the global state content
  {
    // structural element evaluation time
    dt_ele_ptr_ = &(global_state().get_element_evaluation_time());
  }

  // displ-displ block
  stiff_ptr_ = dynamic_cast<Core::LinAlg::SparseMatrix*>(
      global_state().create_structural_stiffness_matrix_block());

  // modified stiffness pointer for storing element based scaling operator (PTC)
  stiff_ptc_ptr_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *global_state().dof_row_map_view(), 81, true, true);

  FOUR_C_ASSERT(stiff_ptr_ != nullptr, "Dynamic cast to Core::LinAlg::SparseMatrix failed!");

  // get the structural dynamic content
  {
    // setup important evaluation booleans
    masslin_type_ = tim_int().get_data_sdyn().get_mass_lin_type();
  }
  // setup new variables
  {
    dis_incr_ptr_ = std::make_shared<Core::LinAlg::Vector<double>>(dis_np().get_map(), true);
  }

  // setup output writers
  {
    if (global_in_output().get_runtime_output_params() != nullptr)
    {
      visualization_params_ = Core::IO::visualization_parameters_factory(
          Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
          *Global::Problem::instance()->output_control_file(), global_state().get_time_n());

      // We only want to create the respective writers if they are actually needed. Therefore, we
      // get the global number ob beam and non-beam elements here. Based on that number we know
      // which output writers need to be initialized.
      const auto discretization =
          std::dynamic_pointer_cast<const Core::FE::Discretization>(discret_ptr());
      int number_my_solid_elements = std::ranges::count_if(discretization->my_row_element_range(),
          [](auto row_element)
          {
            return dynamic_cast<const Discret::Elements::Beam3Base*>(row_element.user_element()) ==
                   nullptr;
          });
      int number_my_beam_elements =
          discretization->num_my_row_elements() - number_my_solid_elements;
      int number_global_solid_elements = 0;
      int number_global_beam_elements = 0;
      number_global_solid_elements =
          Core::Communication::max_all(number_my_solid_elements, discretization->get_comm());
      number_global_beam_elements =
          Core::Communication::max_all(number_my_beam_elements, discretization->get_comm());

      if (global_in_output().get_runtime_output_params()->output_structure() &&
          number_global_solid_elements > 0)
        init_output_runtime_structure();

      if (global_in_output().get_runtime_output_params()->output_beams() &&
          number_global_beam_elements > 0)
        init_output_runtime_beams();
    }
  }

  // setup error evaluator parameters
  {
    error_evaluator_parameters_ = ErrorEvaluator::error_evaluator_parameter_factory(
        Global::Problem::instance()->structural_dynamic_params().sublist("ERROR EVALUATION"));
  }

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::reset(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  /* --- reset external forces
   * Please note, that PutScalar is safer (but maybe slower) than
   * put_scalar(0.0), because of possible NaN and inf values! */
  fext_np().put_scalar(0.0);

  /* --- reset internal forces
   * Please note, that PutScalar is safer (but maybe slower) than
   * put_scalar(0.0), because of possible NaN and inf values! */
  fint_np().put_scalar(0.0);

  // reset stiffness matrix
  stiff().zero();

  // reset modified stiffness matrix
  stiff_ptc().zero();


  // set evaluation time back to zero
  *dt_ele_ptr_ = 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::evaluate_force()
{
  check_init_setup();
  bool ok = true;
  // ---------------------------------------
  // (1) EXTERNAL FORCES
  // ---------------------------------------
  ok = apply_force_external();

  // ---------------------------------------
  // (2) INTERNAL FORCES
  // ---------------------------------------
  // ordinary internal force
  ok = (ok ? apply_force_internal() : false);

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::evaluate_stiff()
{
  check_init_setup();
  bool ok = true;

  /* We use the same routines as for the apply_force_stiff case, but we
   * do not update the global force vector, which is used for the
   * solution process in the NOX library.
   * This is meaningful, since the computational overhead, which is
   * generated by evaluating the right hand side is negligible */
  // *********** time measurement ***********
  double dtcpu = global_state().get_timer()->wallTime();
  // *********** time measurement ***********
  // ---------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // ---------------------------------------------------------------------
  ok = apply_force_stiff_external();

  // ---------------------------------------------------------------------
  // (2) INTERNAL FORCES and STIFFNESS ENTRIES
  // ---------------------------------------------------------------------
  // ordinary internal force
  ok = (ok ? apply_force_stiff_internal() : false);

  // *********** time measurement ***********
  *dt_ele_ptr_ += global_state().get_timer()->wallTime() - dtcpu;
  // *********** time measurement ***********

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::evaluate_force_stiff()
{
  check_init_setup();
  bool ok = true;

  // *********** time measurement ***********
  double dtcpu = global_state().get_timer()->wallTime();
  // *********** time measurement ***********
  // ---------------------------------------------------------------------
  // (1) EXTRERNAL FORCES and STIFFNESS ENTRIES
  // ---------------------------------------------------------------------
  ok = apply_force_stiff_external();

  // ---------------------------------------------------------------------
  // (2) INTERNAL FORCES and STIFFNESS ENTRIES
  // ---------------------------------------------------------------------
  // ordinary internal force
  ok = (ok ? apply_force_stiff_internal() : false);

  // *********** time measurement ***********
  *dt_ele_ptr_ += global_state().get_timer()->wallTime() - dtcpu;
  // *********** time measurement ***********

  // that's it
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  Core::LinAlg::assemble_my_vector(1.0, f, -timefac_np, fext_np());
  Core::LinAlg::assemble_my_vector(1.0, f, timefac_np, fint_np());

  // add the scaled force contributions of the old time step
  // structural dofs of the right-hand-side vector at t_{n+timefac_n} (read-only)
  std::shared_ptr<const Core::LinAlg::Vector<double>> fstructold_ptr =
      global_state().get_fstructure_old();
  Core::LinAlg::assemble_my_vector(1.0, f, 1.0, *fstructold_ptr);

  // add the visco and mass contributions
  integrator().add_visco_mass_contributions(f);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::assemble_jacobian(
    Core::LinAlg::SparseOperator& jac, const double& timefac_np) const
{
  stiff().scale(timefac_np);
  global_state().assign_model_block(jac, stiff(), type(), Solid::MatBlockType::displ_displ);

  // add the visco and mass contributions
  integrator().add_visco_mass_contributions(jac);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::initialize_inertia_and_damping()
{
  check_init_setup();

  // currently a fixed number of matrix and vector pointers are supported
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  // create vector with zero entries
  std::shared_ptr<const Core::LinAlg::Vector<double>> zeros =
      integrator().get_dbc().get_zeros_ptr();

  // set vector values needed by elements
  // --> initially zero !!!
  discret().clear_state();
  discret().set_state(0, "residual displacement", *zeros);
  discret().set_state(0, "displacement", *zeros);

  // set action type and evaluation matrix and vector pointers
  static_contributions(eval_mat.data(), eval_vec.data());
  material_damping_contributions(eval_mat.data());
  inertial_contributions(eval_mat.data(), eval_vec.data());

  // evaluate
  evaluate_internal(eval_mat.data(), eval_vec.data());

  // complete stiffness and mass matrix
  fill_complete();

  // assemble the rayleigh damping matrix
  rayleigh_damping_matrix();

  return eval_error_check();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::apply_force_internal()
{
  check_init_setup();

  // currently a fixed number of matrix and vector pointers are supported
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "residual displacement", *dis_incr_ptr_);
  discret().set_state(0, "displacement", *global_state().get_dis_np());
  discret().set_state(0, "velocity", *global_state().get_vel_np());

  // set action type and evaluation matrix and vector pointers
  static_contributions(eval_vec.data());
  material_damping_contributions(eval_mat.data());
  inertial_contributions(eval_vec.data());

  // evaluate ...
  evaluate_internal(eval_mat.data(), eval_vec.data());

  // evaluate inertia and visco forces
  inertial_and_viscous_forces();

  return eval_error_check();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::apply_force_external()
{
  check_init_setup();

  // Set to default value, because it is unnecessary for the
  // evaluate_neumann routine.
  eval_data().set_action_type(Core::Elements::none);
  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state().get_dis_n());
  if (eval_data().get_damping_type() == Inpar::Solid::damp_material)
    discret().set_state(0, "velocity", *global_state().get_vel_n());
  discret().set_state(0, "displacement new", *global_state().get_dis_np());
  evaluate_neumann(*global_state().get_fext_np(), nullptr);

  return eval_error_check();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::apply_force_stiff_external()
{
  check_init_setup();

  if (pre_apply_force_stiff_external(fext_np(), *stiff_ptr_)) return true;

  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state().get_dis_n());

  if (eval_data().get_damping_type() == Inpar::Solid::damp_material)
    discret().set_state(0, "velocity", *global_state().get_vel_n());

  // get load vector
  if (!tim_int().get_data_sdyn().get_load_lin())
    evaluate_neumann(*global_state().get_fext_np(), nullptr);
  else
  {
    discret().set_state(0, "displacement new", *global_state().get_dis_np());

    /* Add the linearization of the external force to the stiffness
     * matrix. */
    evaluate_neumann(*global_state().get_fext_np(), Core::Utils::shared_ptr_from_ref(*stiff_ptr_));
  }

  return eval_error_check();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::pre_apply_force_stiff_external(
    Core::LinAlg::Vector<double>& fextnp, Core::LinAlg::SparseMatrix& stiff) const
{
  check_init_setup();

  const auto* impl_ptr = dynamic_cast<const Solid::TimeInt::Implicit*>(&tim_int());
  if (impl_ptr) return impl_ptr->predictor().pre_apply_force_external(fextnp);

  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::ModelEvaluator::Structure::apply_force_stiff_internal()
{
  check_init_setup();
  // currently a fixed number of matrix and vector pointers are supported
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "residual displacement", *dis_incr_ptr_);
  discret().set_state(0, "displacement", *global_state().get_dis_np());
  discret().set_state(0, "velocity", *global_state().get_vel_np());

  // set action types and evaluate matrices/vectors
  static_contributions(eval_mat.data(), eval_vec.data());
  material_damping_contributions(eval_mat.data());
  if (masslin_type_ != Inpar::Solid::MassLin::ml_none)
    inertial_contributions(eval_mat.data(), eval_vec.data());

  // evaluate
  evaluate_internal(eval_mat.data(), eval_vec.data());

  // complete stiffness and mass matrix
  fill_complete();

  // evaluate inertial and viscous forces
  inertial_and_viscous_forces();

  return eval_error_check();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::static_contributions(
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  // action for elements
  eval_data().set_action_type(Core::Elements::struct_calc_nlnstiff);
  // set default matrix
  eval_mat[0] = Core::Utils::shared_ptr_from_ref(*stiff_ptr_);
  // set default force vector
  eval_vec[0] = global_state().get_fint_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::static_contributions(
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  // action for elements
  eval_data().set_action_type(Core::Elements::struct_calc_internalforce);
  // set default force vector
  eval_vec[0] = global_state().get_fint_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::material_damping_contributions(
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat)
{
  if (eval_data().get_damping_type() != Inpar::Solid::damp_material) return;

  // action for elements
  // (reset the action type to be independent of the calling order)
  eval_data().set_action_type(Core::Elements::struct_calc_nlnstiff);
  // set the discretization state
  discret().set_state(0, "velocity", *global_state().get_vel_np());
  // reset damping matrix
  damp().zero();
  // add the stiffness matrix as well (also for the apply_force case!)
  eval_mat[0] = Core::Utils::shared_ptr_from_ref(*stiff_ptr_);
  // set damping matrix
  eval_mat[1] = global_state().get_damp_matrix();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::inertial_contributions(
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  check_init_setup();

  if (tim_int().get_data_sdyn_ptr()->neglect_inertia()) return;

  // overwrite element action
  if (tim_int().get_data_sdyn().is_mass_lumping())
    eval_data().set_action_type(Core::Elements::struct_calc_nlnstifflmass);
  else
    eval_data().set_action_type(Core::Elements::struct_calc_nlnstiffmass);

  // set the discretization state
  discret().set_state(0, "velocity", *global_state().get_vel_np());
  discret().set_state(0, "acceleration", *global_state().get_acc_np());
  // reset the mass matrix
  mass().zero();
  // set mass matrix
  eval_mat[1] = global_state().get_mass_matrix();
  // set inertial vector if necessary
  eval_vec[1] = get_inertial_force();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::inertial_contributions(
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  check_init_setup();

  if (masslin_type_ == Inpar::Solid::MassLin::ml_none or
      tim_int().get_data_sdyn_ptr()->neglect_inertia())
    return;

  // overwrite element action
  eval_data().set_action_type(Core::Elements::struct_calc_internalinertiaforce);
  // set the discretization state
  discret().set_state(0, "velocity", *global_state().get_vel_np());
  discret().set_state(0, "acceleration", *global_state().get_acc_np());

  // set inertial vector if necessary
  eval_vec[1] = get_inertial_force();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::inertial_and_viscous_forces()
{
  check_init_setup();

  if (masslin_type_ == Inpar::Solid::MassLin::ml_none and
      !tim_int().get_data_sdyn_ptr()->neglect_inertia())
  {
    // calculate the inertial force at t_{n+1}
    mass().multiply(false, *global_state().get_acc_np(), finertial_np());
  }

  // calculate the viscous/damping force at t_{n+1}
  if (eval_data().get_damping_type() != Inpar::Solid::damp_none)
  {
    if (not damp().filled()) damp().complete();
    damp().multiply(false, *global_state().get_vel_np(), fvisco_np());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::fill_complete()
{
  if (not stiff_ptr_->filled()) stiff_ptr_->complete();

  if (not mass().filled()) mass().complete();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::rayleigh_damping_matrix()
{
  if (eval_data().get_damping_type() != Inpar::Solid::damp_rayleigh) return;

  const double dampk = tim_int().get_data_sdyn().get_damping_stiffness_factor();
  const double dampm = tim_int().get_data_sdyn().get_damping_mass_factor();

  // damping matrix with initial stiffness
  damp().add(stiff(), false, dampk, 0.0);
  damp().add(mass(), false, dampm, 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Solid::ModelEvaluator::Structure::get_inertial_force()
{
  switch (masslin_type_)
  {
    case Inpar::Solid::MassLin::ml_rotations:
    {
      finertial_np().put_scalar(0.0);
      // set inertial force
      return global_state().get_finertial_np();
      break;
    }
    case Inpar::Solid::MassLin::ml_none:
      // do nothing
      break;
    default:
      FOUR_C_THROW("Unknown mass linearization type!");
  }

  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::init_output_runtime_structure()
{
  check_init();
  const auto discretization = std::dynamic_pointer_cast<const Core::FE::Discretization>(
      const_cast<Solid::ModelEvaluator::Structure*>(this)->discret_ptr());
  vtu_writer_ptr_ = std::make_shared<Core::IO::DiscretizationVisualizationWriterMesh>(
      discretization, visualization_params_,
      [](const Core::Elements::Element* element)
      {
        // Skip beam elements which live in the same discretization but use a different output
        // mechanism
        return !dynamic_cast<const Discret::Elements::Beam3Base*>(element);
      });

  if (global_in_output()
          .get_runtime_output_params()
          ->get_structure_params()
          ->gauss_point_data_output() != Inpar::Solid::GaussPointDataOutputType::none)
  {
    init_output_runtime_structure_gauss_point_data();
  }
}

void Solid::ModelEvaluator::Structure::init_output_runtime_structure_gauss_point_data()
{
  // Set all parameters in the evaluation data container.
  eval_data().set_action_type(Core::Elements::struct_init_gauss_point_data_output);
  eval_data().set_gauss_point_data_output_manager_ptr(
      std::make_shared<GaussPointDataOutputManager>(global_in_output()
              .get_runtime_output_params()
              ->get_structure_params()
              ->gauss_point_data_output()));
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time(global_state().get_delta_time()[0]);

  // Set vector values needed by elements.
  discret().clear_state();

  // Set dummy evaluation vectors and matrices.
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  evaluate_internal(eval_mat.data(), eval_vec.data());

  eval_data().get_gauss_point_data_output_manager_ptr()->distribute_quantities(
      discret().get_comm());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::write_time_step_output_runtime_structure() const
{
  check_init_setup();

  auto [output_time, output_step] = get_time_and_time_step_index_for_output(
      visualization_params_, global_state().get_time_n(), global_state().get_step_n());
  write_output_runtime_structure(*global_state().get_dis_n(), *global_state().get_vel_n(),
      *global_state().get_acc_n(), output_step, output_time);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::write_iteration_output_runtime_structure() const
{
  check_init_setup();

  auto [output_time, output_step] = get_time_and_time_step_index_for_output(visualization_params_,
      global_state().get_time_n(), global_state().get_step_n(), eval_data().get_nln_iter());
  write_output_runtime_structure(*global_state().get_dis_np(), *global_state().get_vel_np(),
      *global_state().get_acc_np(), output_step, output_time);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::write_output_runtime_structure(
    const Core::LinAlg::Vector<double>& displacement_state_vector,
    const Core::LinAlg::Vector<double>& velocity_state_vector,
    const Core::LinAlg::Vector<double>& acceleration_state_vector, const int timestep_number,
    const double time) const
{
  check_init_setup();

  // get the parameter container object
  const Discret::Elements::StructureRuntimeOutputParams& structure_output_params =
      *global_in_output().get_runtime_output_params()->get_structure_params();

  // reset time and time step of the writer object
  vtu_writer_ptr_->reset();

  // append all desired output data to the writer object's storage

  // append displacement if desired
  if (structure_output_params.output_displacement_state())
  {
    std::vector<std::optional<std::string>> context(global_state().get_dim(), "displacement");
    vtu_writer_ptr_->append_result_data_vector_with_context(
        displacement_state_vector, Core::IO::OutputEntity::dof, context);
  }

  // append velocity if desired
  if (structure_output_params.output_velocity_state())
  {
    std::vector<std::optional<std::string>> context(global_state().get_dim(), "velocity");
    vtu_writer_ptr_->append_result_data_vector_with_context(
        velocity_state_vector, Core::IO::OutputEntity::dof, context);
  }

  // append acceleration if desired
  if (structure_output_params.output_acceleration_state())
  {
    std::vector<std::optional<std::string>> context(global_state().get_dim(), "acceleration");
    vtu_writer_ptr_->append_result_data_vector_with_context(
        acceleration_state_vector, Core::IO::OutputEntity::dof, context);
  }

  // append element owner if desired
  if (structure_output_params.output_element_owner())
    vtu_writer_ptr_->append_element_owner("element_owner");

  // append element GIDs if desired
  if (structure_output_params.output_element_gid())
    vtu_writer_ptr_->append_element_gid("element_gid");

  // append element material IDs if desired
  if (structure_output_params.output_element_material_id())
    vtu_writer_ptr_->append_element_material_id();

  // append element ghosting information if desired
  if (structure_output_params.output_element_ghosting())
    vtu_writer_ptr_->append_element_ghosting_information();

  // append node GIDs if desired
  if (structure_output_params.output_node_gid()) vtu_writer_ptr_->append_node_gid("node_gid");

  // append stress if desired
  if (structure_output_params.output_stress_strain() and
      global_in_output().get_stress_output_type() != Inpar::Solid::stress_none)
  {
    std::string name_nodal = "";
    std::string name_element = "";

    if (global_in_output().get_stress_output_type() == Inpar::Solid::stress_2pk)
    {
      name_nodal = "nodal_2PK_stresses_xyz";
      name_element = "element_2PK_stresses_xyz";
    }
    else if (global_in_output().get_stress_output_type() == Inpar::Solid::stress_cauchy)
    {
      name_nodal = "nodal_cauchy_stresses_xyz";
      name_element = "element_cauchy_stresses_xyz";
    }

    // Write nodal stress data.
    std::vector<std::optional<std::string>> context(6, name_nodal);
    vtu_writer_ptr_->append_result_data_vector_with_context(
        *eval_data().get_stress_data_node_postprocessed(), Core::IO::OutputEntity::node, context);

    // Write element stress data.
    context.assign(6, name_element);
    vtu_writer_ptr_->append_result_data_vector_with_context(
        *eval_data().get_stress_data_element_postprocessed(), Core::IO::OutputEntity::element,
        context);
  }

  // append strain if desired.
  if (structure_output_params.output_stress_strain() and
      global_in_output().get_strain_output_type() != Inpar::Solid::strain_none)
  {
    std::string name_nodal = "";
    std::string name_element = "";

    if (global_in_output().get_strain_output_type() == Inpar::Solid::strain_gl)
    {
      name_nodal = "nodal_GL_strains_xyz";
      name_element = "element_GL_strains_xyz";
    }
    else if (global_in_output().get_strain_output_type() == Inpar::Solid::strain_ea)
    {
      name_nodal = "nodal_EA_strains_xyz";
      name_element = "element_EA_strains_xyz";
    }
    else if (global_in_output().get_strain_output_type() == Inpar::Solid::strain_log)
    {
      name_nodal = "nodal_LOG_strains_xyz";
      name_element = "element_LOG_strains_xyz";
    }

    // Write nodal strain data.
    std::vector<std::optional<std::string>> context(6, name_nodal);
    vtu_writer_ptr_->append_result_data_vector_with_context(
        *eval_data().get_strain_data_node_postprocessed(), Core::IO::OutputEntity::node, context);

    // Write element strain data.
    context.assign(6, name_element);
    vtu_writer_ptr_->append_result_data_vector_with_context(
        *eval_data().get_strain_data_element_postprocessed(), Core::IO::OutputEntity::element,
        context);
  }

  // Add gauss point data if desired
  if (structure_output_params.gauss_point_data_output() !=
      Inpar::Solid::GaussPointDataOutputType::none)
  {
    const GaussPointDataOutputManager& elementDataManager =
        *eval_data().get_gauss_point_data_output_manager_ptr();
    for (const auto& nameAndSize : elementDataManager.get_quantities())
    {
      const std::string& name = nameAndSize.first;
      const int size = nameAndSize.second;

      switch (elementDataManager.get_output_type())
      {
        case Inpar::Solid::GaussPointDataOutputType::element_center:
        {
          std::vector<std::optional<std::string>> context(size, name);
          std::shared_ptr<Core::LinAlg::MultiVector<double>> data =
              elementDataManager.get_element_center_data().at(name);
          vtu_writer_ptr_->append_result_data_vector_with_context(
              *data, Core::IO::OutputEntity::element, context);
          break;
        }
        case Inpar::Solid::GaussPointDataOutputType::gauss_points:
        {
          const std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>>& data_list =
              elementDataManager.get_gauss_point_data().at(name);
          for (std::size_t gp = 0; gp < data_list.size(); ++gp)
          {
            const std::string name_with_gp = name + "_gp_" + std::to_string(gp);
            std::vector<std::optional<std::string>> context(size, name_with_gp);
            vtu_writer_ptr_->append_result_data_vector_with_context(
                *data_list[gp], Core::IO::OutputEntity::element, context);
          }
          break;
        }
        case Inpar::Solid::GaussPointDataOutputType::nodes:
        {
          std::vector<std::optional<std::string>> context(size, name);
          std::shared_ptr<Core::LinAlg::MultiVector<double>> data =
              elementDataManager.get_nodal_data().at(name);
          vtu_writer_ptr_->append_result_data_vector_with_context(
              *data, Core::IO::OutputEntity::node, context);
          break;
        }
        case Inpar::Solid::GaussPointDataOutputType::none:
          FOUR_C_THROW("Gauss point data output type is none");
        default:
          FOUR_C_THROW("Gauss point data output type is not implemented yet");
      }
    }
  }


  // add output for optional quantity
  if (structure_output_params.output_optional_quantity() ==
      Inpar::Solid::optquantity_membranethickness)
  {
    // Write nodal membrane thickness
    std::vector<std::optional<std::string>> context(1, "membrane_thickness");
    vtu_writer_ptr_->append_result_data_vector_with_context(
        eval_data().get_opt_quantity_data_node_postprocessed(), Core::IO::OutputEntity::node,
        context);
  }
  else if (structure_output_params.output_optional_quantity() ==
           Inpar::Solid::optquantity_shell7pthickness)
  {
    // Write nodal shell7p thickness
    std::vector<std::optional<std::string>> context(1, "shell7p_thickness");
    vtu_writer_ptr_->append_result_data_vector_with_context(
        eval_data().get_opt_quantity_data_node_postprocessed(), Core::IO::OutputEntity::node,
        context);
  }
  else if (structure_output_params.output_optional_quantity() ==
           Inpar::Solid::optquantity_shell7pthicknessdirector)
  {
    // Write nodal shell7p thickness director vector
    std::vector<std::optional<std::string>> context(
        Inpar::Solid::get_opt_quantity_num_components(
            Inpar::Solid::optquantity_shell7pthicknessdirector),
        "shell7p_thickness_director");
    vtu_writer_ptr_->append_result_data_vector_with_context(
        eval_data().get_opt_quantity_data_node_postprocessed(), Core::IO::OutputEntity::node,
        context);
  }

  // finalize everything and write all required files to filesystem
  vtu_writer_ptr_->write_to_disk(time, timestep_number);
}

void Solid::ModelEvaluator::Structure::evaluate_analytical_error()
{
  if (error_evaluator_parameters_.evaluate_error_analytical)
  {
    // Set vector values needed by elements
    discret().clear_state();
    discret().set_state(0, "displacement", *global_state().get_dis_np());

    // Call the error evaluator
    Teuchos::ParameterList evaluation_parameters;
    evaluation_parameters.set<std::shared_ptr<Core::Elements::ParamsInterface>>(
        "interface", eval_data_ptr());
    ErrorEvaluator::evaluate_error(
        error_evaluator_parameters_, evaluation_parameters, *discret_ptr());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::output_runtime_structure_postprocess_optional_data()
{
  check_init_setup();

  switch (global_in_output()
          .get_runtime_output_params()
          ->get_structure_params()
          ->output_optional_quantity())
  {
    case Inpar::Solid::optquantity_none:
    {
      // do nothing and return
      return;
    }
    case Inpar::Solid::optquantity_membranethickness:
    {
      // evaluate thickness of membrane finite elements
      eval_data().set_action_type(Core::Elements::struct_calc_thickness);
      break;
    }
    case Inpar::Solid::optquantity_shell7pthickness:
    case Inpar::Solid::optquantity_shell7pthicknessdirector:
    {
      // evaluate thickness of shell7p finite elements
      eval_data().set_action_type(Core::Elements::struct_calc_thickness);
      break;
    }
    default:
      FOUR_C_THROW("Type of optional quantity not implemented yet!");
  }

  // set required parameter in the evaluation data container
  eval_data().set_opt_quantity_data(std::make_shared<std::vector<char>>());

  // set vector values needed by elements
  discret().clear_state();

  // set dummy evaluation vectors and matrices
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  evaluate_internal_specified_elements(
      eval_mat.data(), eval_vec.data(), discret().element_row_map());

  auto evaluate_gauss_point_data = [&](const std::vector<char>& raw_data)
  {
    // Get the values at the Gauss-points.
    std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> mapdata{};

    Core::Communication::UnpackBuffer buffer(raw_data);
    for (int i = 0; i < discret_ptr()->element_row_map()->num_my_elements(); ++i)
    {
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> gpthickness =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>();
      extract_from_pack(buffer, *gpthickness);
      mapdata[discret_ptr()->element_row_map()->gid(i)] = gpthickness;
    }
    return mapdata;
  };

  auto postprocess_gauss_point_data_to_nodes =
      [&](const std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>& map_data,
          Core::LinAlg::MultiVector<double>& assembled_data)
  {
    discret_ptr()->evaluate(
        [&](Core::Elements::Element& ele)
        {
          extrapolate_gauss_point_quantity_to_nodes(
              ele, *map_data.at(ele.id()), discret(), assembled_data);
        });
  };

  // do the actual post processing
  std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> gp_thickness_data =
      evaluate_gauss_point_data(*eval_data().get_opt_quantity_data());

  Core::Communication::Exporter ex(
      *(discret().element_row_map()), *(discret().element_col_map()), discret().get_comm());
  ex.do_export(gp_thickness_data);

  // Determine number of components based on output type
  const int num_components = Inpar::Solid::get_opt_quantity_num_components(global_in_output()
          .get_runtime_output_params()
          ->get_structure_params()
          ->output_optional_quantity());

  Core::LinAlg::MultiVector<double> row_nodal_data(*discret().node_row_map(), num_components, true);
  postprocess_gauss_point_data_to_nodes(gp_thickness_data, row_nodal_data);

  auto opt_quantity = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *discret().node_col_map(), num_components, true);
  export_to(row_nodal_data, *opt_quantity);
  eval_data().set_opt_quantity_data_node_postprocessed(opt_quantity);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::output_runtime_structure_postprocess_stress_strain()
{
  check_init_setup();

  determine_stress_strain();

  auto do_postprocessing_on_element = [](const Core::Elements::Element& ele)
  {
    // If it is not a beam element, we post-process it.
    return dynamic_cast<const Discret::Elements::Beam3Base*>(&ele) == nullptr;
  };

  auto evaluate_gauss_point_data = [&](const std::vector<char>& raw_data)
  {
    // Get the values at the Gauss-points.
    std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> mapdata{};

    Core::Communication::UnpackBuffer buffer(raw_data);
    for (int i = 0; i < discret_ptr()->element_row_map()->num_my_elements(); ++i)
    {
      if (do_postprocessing_on_element(*discret().l_row_element(i)))
      {
        auto gpstress = std::make_shared<Core::LinAlg::SerialDenseMatrix>();
        extract_from_pack(buffer, *gpstress);
        mapdata[discret_ptr()->element_row_map()->gid(i)] = gpstress;
      }
    }
    return mapdata;
  };

  auto postprocess_gauss_point_data_to_nodes =
      [&](const std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>& map_data,
          Core::LinAlg::MultiVector<double>& assembled_data)
  {
    discret_ptr()->evaluate(
        [&](Core::Elements::Element& ele)
        {
          if (do_postprocessing_on_element(ele))
            Core::FE::extrapolate_gauss_point_quantity_to_nodes(
                ele, *map_data.at(ele.id()), discret(), assembled_data);
        });
  };

  auto postprocess_gauss_point_data_to_element_center =
      [&](const std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>& map_data,
          Core::LinAlg::MultiVector<double>& assembled_data)
  {
    discret_ptr()->evaluate(
        [&](Core::Elements::Element& ele)
        {
          if (do_postprocessing_on_element(ele))
            Core::FE::evaluate_gauss_point_quantity_at_element_center(
                ele, *map_data.at(ele.id()), assembled_data);
        });
  };

  if (global_in_output().get_stress_output_type() != Inpar::Solid::stress_none)
  {
    std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> gp_stress_data =
        evaluate_gauss_point_data(*eval_data().get_stress_data());

    Core::Communication::Exporter ex(
        *(discret().element_row_map()), *(discret().element_col_map()), discret().get_comm());
    ex.do_export(gp_stress_data);

    eval_data().get_stress_data_node_postprocessed() =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*discret().node_col_map(), 6, true);
    eval_data().get_stress_data_element_postprocessed() =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*discret().element_row_map(), 6, true);


    Core::LinAlg::MultiVector<double> row_nodal_data(*discret().node_row_map(), 6, true);
    postprocess_gauss_point_data_to_nodes(gp_stress_data, row_nodal_data);
    Core::LinAlg::export_to(row_nodal_data, *eval_data().get_stress_data_node_postprocessed());

    postprocess_gauss_point_data_to_element_center(
        gp_stress_data, *eval_data().get_stress_data_element_postprocessed());
  }
  if (global_in_output().get_strain_output_type() != Inpar::Solid::strain_none)
  {
    std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> gp_strain_data =
        evaluate_gauss_point_data(*eval_data().get_strain_data());

    Core::Communication::Exporter ex(
        *(discret().element_row_map()), *(discret().element_col_map()), discret().get_comm());
    ex.do_export(gp_strain_data);

    eval_data().get_strain_data_node_postprocessed() =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*discret().node_col_map(), 6, true);
    eval_data().get_strain_data_element_postprocessed() =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*discret().element_row_map(), 6, true);

    Core::LinAlg::MultiVector<double> row_nodal_data(*discret().node_row_map(), 6, true);
    postprocess_gauss_point_data_to_nodes(gp_strain_data, row_nodal_data);
    Core::LinAlg::export_to(row_nodal_data, *eval_data().get_strain_data_node_postprocessed());

    postprocess_gauss_point_data_to_element_center(
        gp_strain_data, *eval_data().get_strain_data_element_postprocessed());
  }
}

void Solid::ModelEvaluator::Structure::output_runtime_structure_gauss_point_data()
{
  const Discret::Elements::StructureRuntimeOutputParams& structure_output_params =
      *global_in_output().get_runtime_output_params()->get_structure_params();
  if (structure_output_params.gauss_point_data_output() !=
      Inpar::Solid::GaussPointDataOutputType::none)
  {
    check_init_setup();

    eval_data().set_action_type(Core::Elements::struct_gauss_point_data_output);
    eval_data().set_total_time(global_state().get_time_np());
    eval_data().set_delta_time(global_state().get_delta_time()[0]);

    eval_data().gauss_point_data_output_manager_ptr()->prepare_data(
        *discret().node_col_map(), *discret().element_row_map());

    discret().clear_state();
    discret().set_state(0, "displacement", *global_state().get_dis_np());
    discret().set_state(0, "residual displacement", *dis_incr_ptr_);

    std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
        nullptr, nullptr, nullptr};
    std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

    evaluate_internal(eval_mat.data(), eval_vec.data());

    eval_data().gauss_point_data_output_manager_ptr()->post_evaluate();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::init_output_runtime_beams()
{
  beam_vtu_writer_ptr_ = std::make_shared<BeamDiscretizationRuntimeOutputWriter>(
      visualization_params_, dis_np().get_comm());

  // get the parameter container object
  const Discret::Elements::BeamRuntimeOutputParams& beam_output_params =
      *global_in_output().get_runtime_output_params()->get_beam_params();

  // export displacement state to column format
  std::shared_ptr<Core::LinAlg::Vector<double>> disn_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*discret().dof_col_map(), true);
  Core::LinAlg::export_to(*global_state().get_dis_n(), *disn_col);

  // get bounding box object only if periodic boundaries are active
  std::shared_ptr<Core::Geo::MeshFree::BoundingBox> bounding_box_ptr =
      tim_int().get_data_sdyn_ptr()->get_periodic_bounding_box();

  // initialize the writer object with current displacement state
  beam_vtu_writer_ptr_->initialize(
      const_cast<Solid::ModelEvaluator::Structure*>(this)->discret_ptr(),
      beam_output_params.use_absolute_positions(),
      beam_output_params.get_number_visualization_subsegments(), bounding_box_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::write_time_step_output_runtime_beams() const
{
  check_init_setup();

  // export displacement state to column format
  const auto& discret = dynamic_cast<const Core::FE::Discretization&>(this->discret());
  std::shared_ptr<Core::LinAlg::Vector<double>> disn_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*discret.dof_col_map(), true);
  Core::LinAlg::export_to(*global_state().get_dis_n(), *disn_col);

  auto [output_time, output_step] = Core::IO::get_time_and_time_step_index_for_output(
      visualization_params_, global_state().get_time_n(), global_state().get_step_n());
  write_output_runtime_beams(*disn_col, output_step, output_time);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::write_iteration_output_runtime_beams() const
{
  check_init_setup();

  // export displacement state to column format
  const auto& discret = dynamic_cast<const Core::FE::Discretization&>(this->discret());
  std::shared_ptr<Core::LinAlg::Vector<double>> disnp_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*discret.dof_col_map(), true);
  Core::LinAlg::export_to(*global_state().get_dis_np(), *disnp_col);

  auto [output_time, output_step] =
      Core::IO::get_time_and_time_step_index_for_output(visualization_params_,
          global_state().get_time_n(), global_state().get_step_n(), eval_data().get_nln_iter());
  write_output_runtime_beams(*disnp_col, output_step, output_time);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::write_output_runtime_beams(
    Core::LinAlg::Vector<double>& displacement_state_vector, int timestep_number, double time) const
{
  check_init_setup();

  // get the parameter container object
  const Discret::Elements::BeamRuntimeOutputParams& beam_output_params =
      *global_in_output().get_runtime_output_params()->get_beam_params();

  // set geometry
  beam_vtu_writer_ptr_->set_geometry_from_beam_discretization(displacement_state_vector);

  // append all desired output data to the writer object's storage
  beam_vtu_writer_ptr_->append_element_owning_processor();

  // append beam radius
  beam_vtu_writer_ptr_->append_element_circular_cross_section_radius();

  // append displacement if desired
  if (beam_output_params.is_write_internal_energy_element())
    beam_vtu_writer_ptr_->append_element_internal_energy();

  // append displacement if desired
  if (beam_output_params.is_write_kinetic_energy_element())
    beam_vtu_writer_ptr_->append_element_kinetic_energy();

  // append displacement if desired
  if (beam_output_params.output_displacement_state())
    beam_vtu_writer_ptr_->append_displacement_field(displacement_state_vector);

  // append triads if desired
  if (beam_output_params.is_write_triad_visualization_points())
    beam_vtu_writer_ptr_->append_triad_field(displacement_state_vector);

  // append material cross-section strain resultants if desired
  if (beam_output_params.is_write_material_strains_gauss_points())
    beam_vtu_writer_ptr_->append_gauss_point_material_cross_section_strain_resultants();

  // append material cross-section strain resultants if desired
  if (beam_output_params.is_write_material_strains_continuous())
    beam_vtu_writer_ptr_->append_gauss_point_material_cross_section_strain_resultants_continuous();

  // append material cross-section stress resultants if desired
  if (beam_output_params.is_write_material_stresses_gauss_points())
    beam_vtu_writer_ptr_->append_gauss_point_material_cross_section_stress_resultants();

  // append material cross-section stress resultants if desired
  if (beam_output_params.is_write_material_stress_continuous())
    beam_vtu_writer_ptr_->append_gauss_point_material_cross_section_stress_resultants_continuous();

  // append filament id and type if desired
  if (beam_output_params.is_write_element_filament_condition())
    beam_vtu_writer_ptr_->append_element_filament_id_and_type();

  // append filament id and type if desired
  if (beam_output_params.is_write_orientation_parameter())
    beam_vtu_writer_ptr_->append_element_orientation_parameter(displacement_state_vector);

  // append reference length if desired.
  if (beam_output_params.is_write_ref_length()) beam_vtu_writer_ptr_->append_ref_length();

  // export displacement state to column format
  if (beam_output_params.is_write_rve_crosssection_forces())
    beam_vtu_writer_ptr_->append_rve_crosssection_forces(displacement_state_vector);

  // export beam element IDs
  if (beam_output_params.is_write_element_gid()) beam_vtu_writer_ptr_->append_element_gid();

  // Ghosting information
  if (beam_output_params.is_write_element_ghosting())
    beam_vtu_writer_ptr_->append_element_ghosting_information();

  // finalize everything and write all required VTU files to filesystem
  beam_vtu_writer_ptr_->write_to_disk(time, timestep_number);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::evaluate_internal(
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  pre_evaluate_internal();

  Teuchos::ParameterList p;
  p.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", eval_data_ptr());

  evaluate_internal(p, eval_mat, eval_vec);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::evaluate_internal(Teuchos::ParameterList& p,
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec)
{
  if (p.numParams() > 1)
  {
    FOUR_C_THROW(
        "Please use the Solid::Elements::Interface and its derived "
        "classes to set and get parameters.");
  }
  if (not p.isType<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"))
    FOUR_C_THROW("The given parameter has the wrong type!");

  // FixMe as soon as possible: write data to the parameter list.
  // this is about to go, once the old time integration is deleted
  params_interface2_parameter_list(eval_data_ptr(), p);

  discret().evaluate(p, eval_mat[0], eval_mat[1], eval_vec[0], eval_vec[1], eval_vec[2]);
  discret().clear_state();
}

void Solid::ModelEvaluator::Structure::evaluate_internal_specified_elements(
    std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec,
    const Core::LinAlg::Map* ele_map_to_be_evaluated)
{
  pre_evaluate_internal();

  Teuchos::ParameterList p;
  p.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", eval_data_ptr());

  evaluate_internal_specified_elements(p, eval_mat, eval_vec, ele_map_to_be_evaluated);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::evaluate_internal_specified_elements(
    Teuchos::ParameterList& p, std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
    std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec,
    const Core::LinAlg::Map* ele_map_to_be_evaluated)
{
  if (p.numParams() > 1)
  {
    FOUR_C_THROW(
        "Please use the Solid::Elements::Interface and its derived "
        "classes to set and get parameters.");
  }
  if (not p.isType<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"))
    FOUR_C_THROW("The given parameter has the wrong type!");

  // write data to the parameter list.
  // this is about to go, once the old time integration is deleted
  params_interface2_parameter_list(eval_data_ptr(), p);

  Core::FE::evaluate(*discret_ptr(), p, *eval_mat, *eval_vec, ele_map_to_be_evaluated);

  discret().clear_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::evaluate_neumann(Core::LinAlg::Vector<double>& eval_vec,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& eval_mat)
{
  Teuchos::ParameterList p;
  p.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", eval_data_ptr());
  evaluate_neumann(p, eval_vec, eval_mat);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::evaluate_neumann(Teuchos::ParameterList& p,
    Core::LinAlg::Vector<double>& eval_vec,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& eval_mat)
{
  if (p.numParams() > 1)
  {
    FOUR_C_THROW(
        "Please use the Solid::Elements::Interface and its derived "
        "classes to set and get parameters.");
  }
  if (not p.isType<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"))
    FOUR_C_THROW("The given parameter has the wrong type!");
  discret().evaluate_neumann(p, eval_vec, eval_mat.get());
  discret().clear_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  // write forces
  iowriter.write_vector("fstructure_old", global_state().get_fstructure_old());
  iowriter.write_vector("fint", global_state().get_fint_n());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  // read structural force vector
  ioreader.read_vector(global_state().get_fstructure_old(), "fstructure_old");
  ioreader.read_vector(global_state().get_fint_n(), "fint");
  // read displacement field
  std::shared_ptr<Core::LinAlg::Vector<double>>& disnp = global_state().get_dis_np();
  ioreader.read_vector(disnp, "displacement");
  global_state().get_multi_dis().update_steps(*disnp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::predict(const Inpar::Solid::PredEnum& pred_type)
{
  // set the element action
  eval_data().set_action_type(Core::Elements::struct_calc_predict);
  eval_data().set_predictor_type(pred_type);

  // set the matrix and vector pointers to nullptr
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  evaluate_internal(eval_mat.data(), eval_vec.data());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
    Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp)
{
  check_init_setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::run_recover()
{
  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "residual displacement", *dis_incr_ptr_);
  discret().set_state(0, "displacement", *global_state().get_dis_np());
  // set the element action
  eval_data().set_action_type(Core::Elements::struct_calc_recover);
  // set the matrix and vector pointers to nullptr
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  evaluate_internal(eval_mat.data(), eval_vec.data());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
    const Core::LinAlg::Vector<double>& dir, const Core::LinAlg::Vector<double>& xnew)
{
  check_init_setup();
  reset(xnew);
  /* set the class internal displacement increment vector. Check if it is
   * meaningful/necessary in some cases, like incremental strains etc. */
  dis_incr_ptr_ = global_state().extract_displ_entries(dir);
  run_recover();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::run_post_iterate(const ::NOX::Solver::Generic& solver)
{
  check_init_setup();

  if (vtu_writer_ptr_ != nullptr and
      global_in_output().get_runtime_output_params()->output_every_iteration())
  {
    if (not(global_in_output().get_stress_output_type() == Inpar::Solid::stress_none and
            global_in_output().get_strain_output_type() == Inpar::Solid::strain_none))
    {
      output_runtime_structure_postprocess_stress_strain();
    }

    output_runtime_structure_gauss_point_data();
    output_runtime_structure_postprocess_optional_data();
    write_iteration_output_runtime_structure();
  }

  // write special output for beams if desired
  if (beam_vtu_writer_ptr_ != nullptr and
      global_in_output().get_runtime_output_params()->output_every_iteration())
    write_iteration_output_runtime_beams();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::update_step_state(const double& timefac_n)
{
  check_init_setup();
  // update state
  // new displacements at t_{n+1} -> t_n
  //    D_{n} := D_{n+1}
  global_state().get_multi_dis().update_steps(dis_np());

  // new velocities at t_{n+1} -> t_{n}
  //    V_{n} := V_{n+1}
  global_state().get_multi_vel().update_steps(*global_state().get_vel_np());

  // new at t_{n+1} -> t_n
  //    A_{n} := A_{n+1}
  global_state().get_multi_acc().update_steps(*global_state().get_acc_np());

  // store the old external force
  global_state().get_fext_n()->scale(1.0, fext_np());

  // store the old reaction force
  global_state().get_freact_n()->scale(1.0, *global_state().get_freact_np());

  // store the old internal force
  global_state().get_fint_n()->scale(1.0, fint_np());

  // new at t_{n+1} -> t_{n+timefac_n}
  //    F^{struct}_{n+timefac_n} := timefac_n * F^{struct}_{n+1}
  std::shared_ptr<Core::LinAlg::Vector<double>>& fstructold_ptr =
      global_state().get_fstructure_old();
  fstructold_ptr->update(timefac_n, fint_np(), 1.0);
  fstructold_ptr->update(-timefac_n, fext_np(), 1.0);

  // set the displacement increment back to zero
  dis_incr_ptr_->put_scalar(0.0);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::evaluate_jacobian_contributions_from_element_level_for_ptc()
{
  check_init_setup();
  // currently a fixed number of matrix and vector pointers are supported
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  eval_data().set_action_type(Core::Elements::struct_calc_addjacPTC);

  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state().get_dis_np());

  eval_mat[0] = Core::Utils::shared_ptr_from_ref(*stiff_ptc_ptr_);

  // evaluate
  evaluate_internal(eval_mat.data(), eval_vec.data());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::assemble_jacobian_contributions_from_element_level_for_ptc(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& modjac, const double& timefac_n)
{
  global_state().assign_model_block(*modjac, stiff_ptc(), type(), Solid::MatBlockType::displ_displ);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::update_step_element()
{
  check_init_setup();
  // other parameters that might be needed by the elements
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time(global_state().get_delta_time()[0]);

  const Inpar::Solid::PreStress prestress_type = tim_int().get_data_sdyn().get_pre_stress_type();
  const double prestress_time = tim_int().get_data_sdyn().get_pre_stress_time();
  bool isDuringPrestressing = prestress_type != Inpar::Solid::PreStress::none &&
                              global_state().get_time_n() <= prestress_time + 1.0e-15;

  if (isDuringPrestressing && prestress_type == Inpar::Solid::PreStress::mulf)
  {
    if (Core::Communication::my_mpi_rank(discret().get_comm()) == 0)
      Core::IO::cout << "====== Entering PRESTRESSING update" << Core::IO::endl;

    // Choose special update action for elements in case of MULF
    eval_data().set_action_type(Core::Elements::struct_update_prestress);
  }
  else
  {
    // Call the normal element  update routine
    eval_data().set_action_type(Core::Elements::struct_calc_update_istep);
  }


  // go to elements
  discret().clear_state();
  discret().set_state("displacement", *global_state().get_dis_n());

  // set dummy evaluation vectors and matrices
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};
  evaluate_internal(eval_mat.data(), eval_vec.data());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::update_residual()
{
  check_init_setup();
  dis_incr_ptr_->update(-1.0, *global_state().get_dis_n(), 1.0, *global_state().get_dis_np(), 0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::determine_stress_strain()
{
  check_init_setup();

  if (global_in_output().get_stress_output_type() == Inpar::Solid::stress_none and
      global_in_output().get_strain_output_type() == Inpar::Solid::strain_none and
      global_in_output().get_plastic_strain_output_type() == Inpar::Solid::strain_none)
    return;

  // set all parameters in the evaluation data container
  eval_data().set_action_type(Core::Elements::struct_calc_stress);
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time(global_state().get_delta_time()[0]);
  eval_data().set_stress_data(std::make_shared<std::vector<char>>());
  eval_data().set_strain_data(std::make_shared<std::vector<char>>());
  eval_data().set_plastic_strain_data(std::make_shared<std::vector<char>>());

  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state().get_dis_np());
  discret().set_state(0, "residual displacement", *dis_incr_ptr_);

  // set dummy evaluation vectors and matrices
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  evaluate_internal_specified_elements(
      eval_mat.data(), eval_vec.data(), discret().element_row_map());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::determine_strain_energy(
    const Core::LinAlg::Vector<double>& disnp, const bool global)
{
  check_init_setup();

  // set required parameters in the evaluation data container
  eval_data().set_action_type(Core::Elements::struct_calc_energy);
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time(global_state().get_delta_time()[0]);

  // set state vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "displacement", disnp);
  discret().set_state(0, "residual displacement", *dis_incr_ptr_);

  // set dummy evaluation vectors and matrices
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  pre_evaluate_internal();

  Teuchos::ParameterList p;
  p.set<std::shared_ptr<Core::Elements::ParamsInterface>>("interface", eval_data_ptr());

  // evaluate energy contributions on element level (row elements only)
  evaluate_internal_specified_elements(
      p, eval_mat.data(), eval_vec.data(), discret().element_row_map());

  if (global)
  {
    double my_int_energy = eval_data().get_energy_data(Solid::internal_energy);
    double gsum = 0.0;
    gsum = Core::Communication::sum_all(my_int_energy, discret().get_comm());

    eval_data().set_value_for_energy_type(gsum, Solid::internal_energy);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::determine_energy()
{
  determine_energy(*global_state().get_dis_np(), global_state().get_vel_np().get(), false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::determine_energy(const Core::LinAlg::Vector<double>& disnp,
    const Core::LinAlg::Vector<double>* velnp, const bool global)
{
  determine_strain_energy(disnp, global);

  // global calculation of kinetic energy
  if (masslin_type_ == Inpar::Solid::MassLin::ml_none and velnp != nullptr)
  {
    if (not mass().filled()) mass().complete();

    double kinetic_energy_times2 = 0.0;

    std::shared_ptr<Core::LinAlg::Vector<double>> linear_momentum =
        std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view(), true);

    mass().multiply(false, *velnp, *linear_momentum);

    linear_momentum->dot(*velnp, &kinetic_energy_times2);

    // only add the result on one processor because we sum over all procs later
    if (global or global_state().get_my_rank() == 0)
    {
      eval_data().add_contribution_to_energy_type(
          0.5 * kinetic_energy_times2, Solid::kinetic_energy);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::runtime_pre_output_step_state()
{
  check_init_setup();

  if (vtu_writer_ptr_ != nullptr)
  {
    if (not(global_in_output().get_stress_output_type() == Inpar::Solid::stress_none and
            global_in_output().get_strain_output_type() == Inpar::Solid::strain_none))
    {
      output_runtime_structure_postprocess_stress_strain();
    }
    output_runtime_structure_gauss_point_data();
    output_runtime_structure_postprocess_optional_data();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::runtime_output_step_state() const
{
  check_init_setup();

  if (vtu_writer_ptr_ != nullptr) write_time_step_output_runtime_structure();

  // write special output for beams if desired
  if (beam_vtu_writer_ptr_ != nullptr) write_time_step_output_runtime_beams();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::reset_step_state()
{
  check_init_setup();

  // reset disp, vel, acc state vector
  global_state_ptr()->get_dis_np()->update(1.0, (*global_state_ptr()->get_dis_n()), 0.0);
  global_state_ptr()->get_vel_np()->update(1.0, (*global_state_ptr()->get_vel_n()), 0.0);
  global_state_ptr()->get_acc_np()->update(1.0, (*global_state_ptr()->get_acc_n()), 0.0);

  // other parameters that might be needed by the elements
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time(global_state().get_delta_time()[0]);
  // action for elements
  eval_data().set_action_type(Core::Elements::struct_calc_reset_istep);

  // set dummy evaluation vectors and matrices
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};
  evaluate_internal(eval_mat.data(), eval_vec.data());

  discret_ptr()->clear_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
Solid::ModelEvaluator::Structure::get_block_dof_row_map_ptr() const
{
  check_init_setup();
  return global_state().dof_row_map();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::Structure::get_current_solution_ptr() const
{
  check_init();
  return global_state().get_dis_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::Structure::get_last_time_step_solution_ptr() const
{
  check_init();
  return global_state().get_dis_n();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::post_output()
{
  check_init_setup();
  // empty
}  // post_output()

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fint_np()
{
  check_init();
  FOUR_C_ASSERT(global_state().get_fint_np(), "nullptr!");

  return *global_state().get_fint_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fint_np() const
{
  check_init();
  FOUR_C_ASSERT(global_state().get_fint_np(), "nullptr!");

  return *global_state().get_fint_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fint_n() const
{
  check_init();
  if (global_state().get_fint_n() == nullptr) FOUR_C_THROW("NULL pointer!");

  return *global_state().get_fint_n();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fext_np()
{
  check_init();
  FOUR_C_ASSERT(global_state().get_fext_np(), "nullptr!");

  return *global_state().get_fext_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fext_np() const
{
  check_init();
  FOUR_C_ASSERT(global_state().get_fext_np(), "nullptr!");

  return *global_state().get_fext_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fext_n() const
{
  check_init();
  if (global_state().get_fext_n() == nullptr) FOUR_C_THROW("NULL pointer!");

  return *global_state().get_fext_n();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::finertial_np()
{
  check_init();
  FOUR_C_ASSERT(global_state().get_finertial_np(), "nullptr!");

  return *global_state().get_finertial_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::finertial_np() const
{
  check_init();
  FOUR_C_ASSERT(global_state().get_finertial_np(), "nullptr!");

  return *global_state().get_finertial_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fvisco_np()
{
  check_init();
  FOUR_C_ASSERT(global_state().get_fvisco_np(), "nullptr!");

  return *global_state().get_fvisco_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::fvisco_np() const
{
  check_init();
  FOUR_C_ASSERT(global_state().get_fvisco_np(), "nullptr!");

  return *global_state().get_fvisco_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::dis_np()
{
  check_init();
  FOUR_C_ASSERT(global_state().get_dis_np(), "nullptr!");

  return *global_state().get_dis_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::ModelEvaluator::Structure::dis_np() const
{
  check_init();
  FOUR_C_ASSERT(global_state().get_dis_np(), "nullptr!");

  return *global_state().get_dis_np();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix& Solid::ModelEvaluator::Structure::stiff() const
{
  check_init();
  FOUR_C_ASSERT(stiff_ptr_, "nullptr!");

  return *stiff_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix& Solid::ModelEvaluator::Structure::stiff_ptc() const
{
  check_init();
  FOUR_C_ASSERT(stiff_ptc_ptr_ != nullptr, "nullptr!");

  return *stiff_ptc_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::SparseOperator& Solid::ModelEvaluator::Structure::mass()
{
  check_init();
  FOUR_C_ASSERT(global_state().get_mass_matrix(), "nullptr!");

  return *global_state().get_mass_matrix();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::SparseOperator& Solid::ModelEvaluator::Structure::mass() const
{
  check_init();
  FOUR_C_ASSERT(global_state().get_mass_matrix(), "nullptr!");

  return *global_state().get_mass_matrix();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::LinAlg::SparseOperator& Solid::ModelEvaluator::Structure::damp()
{
  check_init();
  FOUR_C_ASSERT(global_state().get_damp_matrix(), "nullptr!");

  return *global_state().get_damp_matrix();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::SparseOperator& Solid::ModelEvaluator::Structure::damp() const
{
  check_init();
  FOUR_C_ASSERT(global_state().get_damp_matrix(), "nullptr!");

  return *global_state().get_damp_matrix();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::params_interface2_parameter_list(
    std::shared_ptr<Solid::ModelEvaluator::Data> interface_ptr, Teuchos::ParameterList& params)
{
  FOUR_C_ASSERT(interface_ptr != nullptr, "params_interface pointer not set");

  params.set<double>("delta time", interface_ptr->get_delta_time());
  params.set<double>("total time", interface_ptr->get_total_time());
  params.set<double>("timintfac_dis", interface_ptr->get_tim_int_factor_disp());
  params.set<double>("timintfac_vel", interface_ptr->get_tim_int_factor_vel());

  Core::Elements::ActionType act = interface_ptr->get_action_type();
  std::string action;
  switch (act)
  {
    case Core::Elements::struct_calc_linstiff:
      action = "calc_struct_linstiff";
      break;
    case Core::Elements::struct_calc_nlnstiff:
      action = "calc_struct_nlnstiff";
      break;
    case Core::Elements::struct_calc_internalforce:
      action = "calc_struct_internalforce";
      break;
    case Core::Elements::struct_calc_linstiffmass:
      action = "calc_struct_linstiffmass";
      break;
    case Core::Elements::struct_calc_nlnstiffmass:
      action = "calc_struct_nlnstiffmass";
      break;
    case Core::Elements::struct_calc_nlnstifflmass:
      action = "calc_struct_nlnstifflmass";
      break;
    case Core::Elements::struct_calc_stress:
      action = "calc_struct_stress";
      break;
    case Core::Elements::struct_calc_thickness:
      action = "calc_struct_thickness";
      break;
    case Core::Elements::struct_calc_eleload:
      action = "calc_struct_eleload";
      break;
    case Core::Elements::struct_calc_fsiload:
      action = "calc_struct_fsiload";
      break;
    case Core::Elements::struct_calc_update_istep:
      action = "calc_struct_update_istep";
      break;
    case Core::Elements::struct_calc_reset_istep:
      action = "calc_struct_reset_istep";
      break;
    case Core::Elements::struct_calc_energy:
      action = "calc_struct_energy";
      break;
    case Core::Elements::struct_postprocess_thickness:
      action = "postprocess_thickness";
      break;
    case Core::Elements::struct_update_prestress:
      action = "calc_struct_prestress_update";
      break;
    case Core::Elements::struct_calc_global_gpstresses_map:
      action = "calc_global_gpstresses_map";
      break;
    case Core::Elements::struct_calc_recover:
      action = "calc_struct_recover";
      break;
    case Core::Elements::struct_calc_predict:
      action = "calc_struct_predict";
      break;
    case Core::Elements::struct_init_gauss_point_data_output:
      action = "struct_init_gauss_point_data_output";
      break;
    case Core::Elements::struct_gauss_point_data_output:
      action = "struct_gauss_point_data_output";
      break;
    case Core::Elements::struct_poro_calc_fluidcoupling:
      action = "struct_poro_calc_fluidcoupling";
      break;
    case Core::Elements::struct_poro_calc_scatracoupling:
      action = "struct_poro_calc_scatracoupling";
      break;
    default:
      action = "unknown";
      break;
  }
  params.set<std::string>("action", action);

  params.set<std::shared_ptr<std::vector<char>>>("stress", interface_ptr->stress_data_ptr());
  params.set<std::shared_ptr<std::vector<char>>>("strain", interface_ptr->strain_data_ptr());
  params.set<std::shared_ptr<std::vector<char>>>(
      "plstrain", interface_ptr->plastic_strain_data_ptr());
  params.set<Inpar::Solid::StressType>("iostress", interface_ptr->get_stress_output_type());
  params.set<Inpar::Solid::StrainType>("iostrain", interface_ptr->get_strain_output_type());
  params.set<Inpar::Solid::StrainType>(
      "ioplstrain", interface_ptr->get_plastic_strain_output_type());

  params.set<Inpar::Solid::DampKind>("damping", interface_ptr->get_damping_type());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::create_backup_state(const Core::LinAlg::Vector<double>& dir)
{
  check_init_setup();

  // set all parameters in the evaluation data container
  eval_data().set_action_type(Core::Elements::struct_create_backup);

  // set vector values needed by elements
  discret().clear_state();
  discret().set_state(0, "displacement", *global_state().get_dis_np());
  std::shared_ptr<const Core::LinAlg::Vector<double>> dir_displ =
      global_state().extract_displ_entries(dir);
  discret().set_state(0, "residual displacement", *dir_displ);

  // set dummy evaluation vectors and matrices
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  evaluate_internal(eval_mat.data(), eval_vec.data());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::ModelEvaluator::Structure::recover_from_backup_state()
{
  check_init_setup();

  // set all parameters in the evaluation data container
  eval_data().set_action_type(Core::Elements::struct_recover_from_backup);

  // set vector values needed by elements
  discret().clear_state();

  // set dummy evaluation vectors and matrices
  std::array<std::shared_ptr<Core::LinAlg::Vector<double>>, 3> eval_vec = {
      nullptr, nullptr, nullptr};
  std::array<std::shared_ptr<Core::LinAlg::SparseOperator>, 2> eval_mat = {nullptr, nullptr};

  evaluate_internal(eval_mat.data(), eval_vec.data());
}

FOUR_C_NAMESPACE_CLOSE
