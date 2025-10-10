// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_integrator.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_model_evaluator_structure.hpp"
#include "4C_structure_new_monitor_dbc.hpp"
#include "4C_structure_new_nox_nln_str_linearsystem.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_noxinterface.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Integrator::Integrator()
    : isinit_(false),
      issetup_(false),
      modelevaluator_ptr_(nullptr),
      eval_data_ptr_(nullptr),
      sdyn_ptr_(nullptr),
      gstate_ptr_(nullptr),
      io_ptr_(nullptr),
      dbc_ptr_(nullptr),
      timint_ptr_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::init(const std::shared_ptr<Solid::TimeInt::BaseDataSDyn>& sdyn_ptr,
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
    const std::shared_ptr<Solid::TimeInt::BaseDataIO>& io_ptr,
    const std::shared_ptr<Solid::Dbc>& dbc_ptr,
    const std::shared_ptr<const Solid::TimeInt::Base>& timint_ptr)
{
  issetup_ = false;

  sdyn_ptr_ = sdyn_ptr;
  gstate_ptr_ = gstate_ptr;
  io_ptr_ = io_ptr;
  dbc_ptr_ = dbc_ptr;
  timint_ptr_ = timint_ptr;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::setup()
{
  check_init();
  // ---------------------------------------------------------------------------
  // build model evaluator data container
  // ---------------------------------------------------------------------------
  eval_data_ptr_ = std::make_shared<Solid::ModelEvaluator::Data>();
  eval_data_ptr_->init(timint_ptr_);
  eval_data_ptr_->setup();

  // ---------------------------------------------------------------------------
  // build model evaluator
  // ---------------------------------------------------------------------------
  modelevaluator_ptr_ = std::make_shared<Solid::ModelEvaluatorManager>();
  modelevaluator_ptr_->init(eval_data_ptr_, sdyn_ptr_, gstate_ptr_, io_ptr_,
      Core::Utils::shared_ptr_from_ref(*this), timint_ptr_);
  modelevaluator_ptr_->setup();

  // ---------------------------------------------------------------------------
  // build monitor for a tensile test
  // ---------------------------------------------------------------------------
  monitor_dbc_ptr_ = std::make_shared<Solid::MonitorDbc>();
  monitor_dbc_ptr_->init(io_ptr_, *gstate_ptr_->get_discret(), *gstate_ptr_, *dbc_ptr_);
  monitor_dbc_ptr_->setup();

  mt_energy_.setup();

  // the issetup_ flag is not set here!!!
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::set_initial_displacement(
    const Inpar::Solid::InitialDisp init, const int startfuncno)
{
  switch (init)
  {
    case Inpar::Solid::initdisp_zero_disp:
    {
      global_state().get_dis_n()->put_scalar(0.0);
      global_state().get_dis_np()->put_scalar(0.0);

      break;
    }
    case Inpar::Solid::initdisp_disp_by_function:
    {
      const Core::LinAlg::Map* dofrowmap = global_state().get_discret()->dof_row_map();

      // loop all nodes on the processor
      for (int lnodeid = 0; lnodeid < global_state().get_discret()->num_my_row_nodes(); lnodeid++)
      {
        // get the processor local node
        const Core::Nodes::Node* lnode = global_state().get_discret()->l_row_node(lnodeid);

        // the set of degrees of freedom associated with the node
        const std::vector<int> nodedofset = global_state().get_discret()->dof(0, lnode);

        // loop nodal dofs
        for (unsigned int d = 0; d < nodedofset.size(); ++d)
        {
          const int dofgid = nodedofset[d];
          const int doflid = dofrowmap->lid(dofgid);

          // evaluate component k of spatial function
          const double initialval =
              Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                  .evaluate(lnode->x(), global_state().get_time_n(), d);

          const int err = global_state().get_dis_n()->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }

      // initialize also the solution vector
      global_state().get_dis_np()->update(1.0, *global_state().get_dis_n(), 0.0);

      break;
    }
    default:
      FOUR_C_THROW("Unknown option for initial displacement: {}", init);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::check_init() const { FOUR_C_ASSERT(is_init(), "Call init() first!"); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::check_init_setup() const
{
  FOUR_C_ASSERT(is_init() and is_setup(), "Call init() and setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::reset_model_states(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();
  model_eval().reset_states(x);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::compute_mass_matrix_and_init_acc()
{
  check_init();

  // temporary right-hand-side
  Core::LinAlg::Vector<double> rhs_full =
      Core::LinAlg::Vector<double>(global_state().global_problem_map(), /*zero_out=*/true);

  // overwrite initial state vectors with Dirichlet BCs
  // note that we get accelerations resulting from inhomogeneous Dirichlet conditions here
  const double& timen = global_state().get_multi_time()[0];
  std::shared_ptr<Core::LinAlg::Vector<double>> disnp_ptr = global_state().get_dis_np();
  std::shared_ptr<Core::LinAlg::Vector<double>> velnp_ptr = global_state().get_vel_np();
  std::shared_ptr<Core::LinAlg::Vector<double>> accnp_ptr = global_state().get_acc_np();
  dbc().apply_dirichlet_bc(timen, disnp_ptr, velnp_ptr, accnp_ptr, false);


  // ---------------------------------------------------------------------------
  // compute the mass matrix
  // ---------------------------------------------------------------------------
  // set the evaluate parameters of the current base class
  reset_eval_params();
  // !!! evaluate the initial state !!!
  eval_data().set_total_time(gstate_ptr_->get_time_n());

  // initialize the mass matrix and the Rayleigh damping matrix (optional)
  if (not model_eval().initialize_inertia_and_damping(*disnp_ptr))
    FOUR_C_THROW("initialize_inertia_and_damping failed!");

  /* If we are restarting the simulation, we do not have to calculate a
   * consistent acceleration, since we get it anyway from the restart file.
   * Furthermore, we keep the update routines untouched. Actually, the only
   * thing which is necessary is the calculation of the mass matrix. So we are
   * done at this point.                                    hiermeier 06/16 */
  /* However, if we need to restart the initial state, e.g. when starting dynamics
   * analysis from the static one, we should re-calculate the initial state
   * for a warm start-up */
  if (timint_ptr_->is_restarting() && !timint_ptr_->is_restarting_initial_state())
  {
    return;
  }

  // build the entire initial right-hand-side
  if (not model_eval().apply_initial_force(*disnp_ptr, rhs_full))
    FOUR_C_THROW("apply_initial_force failed!");

  // ---------------------------------------------------------------------------
  // build a NOX::Nln::Solid::LinearSystem
  // ---------------------------------------------------------------------------
  // get the structural linear solver
  std::map<enum NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>> str_linsolver;
  str_linsolver[NOX::Nln::sol_structure] = Teuchos::rcpFromRef(
      *tim_int().get_data_sdyn().get_lin_solvers().at(Inpar::Solid::model_structure));

  // copy the nox parameter-list
  Teuchos::ParameterList p_nox = tim_int().get_data_sdyn().get_nox_params();
  NOX::Nln::Aux::set_printing_parameters(p_nox, global_state().get_comm());

  // create a copy of the initial displacement vector
  Core::LinAlg::Vector<double> soln(*global_state().dof_row_map_view(), true);
  // wrap the soln_ptr in a nox_epetra_Vector
  ::NOX::Epetra::Vector nox_soln(
      Teuchos::rcpFromRef(soln.get_ref_of_epetra_vector()), ::NOX::Epetra::Vector::CreateView);

  // Check if we are using a Newton direction
  std::string dir_str = p_nox.sublist("Direction").get<std::string>("Method");
  if (dir_str == "User Defined")
    dir_str = p_nox.sublist("Direction").get<std::string>("User Defined Method");
  if (dir_str != "Newton" and dir_str != "Modified Newton")
    FOUR_C_THROW(
        "The EquilibriateState predictor is currently only working for the "
        "direction-method \"Newton\".");



  // extract the solid part of the rhs
  std::shared_ptr rhs_solid =
      global_state().extract_model_entries(Inpar::Solid::model_structure, rhs_full);

  // add inertial and viscous contributions to rhs
  /* note: this needs to be done 'manually' here because in the RHS evaluation
   * routine of an ordinary time step, these contributions are scaled by weighting
   * factors inside the time integration scheme (e.g. alpha_f/m for GenAlpha) */
  rhs_solid->update(1.0, *global_state().get_finertial_np(), 1.0);
  rhs_solid->update(1.0, *global_state().get_fvisco_np(), 1.0);

  // Note 1: We create a copy of the mass matrix here since the solid linear solver applies the
  // dirichlet conditions to the matrix. We will, however, need the full matrix later on (without
  // the DBCs applied) to calculate the correct reaction forces.
  // Note 2: We only solve for accelerations at non-Dirichlet DoFs. Accelerations at inhomogeneous
  // Dirichlet DoFs were already added above.
  Core::LinAlg::SparseMatrix mass_matrix(*global_state().dof_row_map_view(), 81, true, true);
  mass_matrix.add(*global_state().get_mass_matrix(), false, 1.0, 0.0);
  mass_matrix.complete();

  if (mass_matrix.NormInf() == 0.0) FOUR_C_THROW("You are about to invert a singular matrix!");

  // create the linear system
  // printing parameters
  Teuchos::ParameterList& p_print = p_nox.sublist("Printing", true);
  // linear solver parameters
  Teuchos::ParameterList& p_ls =
      p_nox.sublist("Direction", true).sublist("Newton", true).sublist("Linear Solver", true);

  // (re)set the linear solver parameters
  p_ls.set<int>("Number of Nonlinear Iterations", 0);
  p_ls.set<int>("Current Time Step", global_state().get_step_np());
  // ToDo Get the actual tolerance value
  p_ls.set<double>("Wanted Tolerance", 1.0e-6);

  NOX::Nln::Solid::LinearSystem linsys(p_print, p_ls, str_linsolver, Teuchos::null, Teuchos::null,
      Teuchos::rcpFromRef(mass_matrix), nox_soln);

  linsys.apply_jacobian_inverse(p_ls, rhs_solid->get_ref_of_epetra_vector(), nox_soln);
  nox_soln.scale(-1.0);

  // get the solution vector and add it into the acceleration vector
  accnp_ptr->update(1.0, nox_soln.getEpetraVector(), 1.0);

  // re-build the entire initial right-hand-side with correct accelerations
  model_eval().apply_initial_force(*disnp_ptr, rhs_full);

  // call update routines to copy states from t_{n+1} to t_{n}
  // note that the time step is not incremented
  pre_update();
  update_step_state();
  update_step_element();
  post_update();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Integrator::current_state_is_equilibrium(const double& tol)
{
  check_init();

  // temporary right-hand-side
  Core::LinAlg::Vector<double> rhs_ptr(global_state().global_problem_map(), /*zero_out=*/true);

  // overwrite initial state vectors with Dirichlet BCs
  const double& timen = global_state().get_multi_time()[0];
  std::shared_ptr<Core::LinAlg::Vector<double>> disnp_ptr = global_state().get_dis_np();
  std::shared_ptr<Core::LinAlg::Vector<double>> velnp_ptr = global_state().get_vel_np();
  std::shared_ptr<Core::LinAlg::Vector<double>> accnp_ptr = global_state().get_acc_np();
  dbc().apply_dirichlet_bc(timen, disnp_ptr, velnp_ptr, accnp_ptr, false);

  // set the evaluate parameters of the current base class
  reset_eval_params();
  // !!! evaluate the initial state !!!
  eval_data().set_total_time(gstate_ptr_->get_time_n());

  // build the entire right-hand-side
  model_eval().apply_initial_force(*disnp_ptr, rhs_ptr);

  // add viscous contributions to rhs
  rhs_ptr.update(1.0, *global_state().get_fvisco_np(), 1.0);
  // add inertial contributions to rhs
  rhs_ptr.update(1.0, *global_state().get_finertial_np(), 1.0);

  double resnorm = 0.0;
  rhs_ptr.norm_inf(&resnorm);

  return (resnorm < tol ? true : false);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::determine_stress_strain()
{
  check_init_setup();
  model_eval().determine_stress_strain();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::determine_energy()
{
  check_init_setup();
  model_eval().determine_energy();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::Integrator::get_model_value(const Core::LinAlg::Vector<double>& x)
{
  FOUR_C_THROW(
      "This routine is not supported in the currently active time "
      "integration scheme.");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::Integrator::get_total_mid_time_str_energy(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();
  if (not mt_energy_.is_correctly_configured())
    FOUR_C_THROW(
        "You are trying to compute the mid-time energy in case of a non-static"
        " simulation, but you have not specified the desired energy averaging type."
        " Please add a meaningful MIDTIME_ENERGY_TYPE to the ---STRUCTURAL DYNAMIC"
        " section of your Input file.");

  std::shared_ptr<const Core::LinAlg::Vector<double>> disnp_ptr =
      global_state().extract_displ_entries(x);
  const Core::LinAlg::Vector<double>& disnp = *disnp_ptr;

  set_state(disnp);

  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp_ptr = global_state().get_vel_np();
  const Core::LinAlg::Vector<double>& velnp = *velnp_ptr;

  eval_data().clear_values_for_all_energy_types();
  Solid::ModelEvaluator::Structure& str_model =
      dynamic_cast<Solid::ModelEvaluator::Structure&>(evaluator(Inpar::Solid::model_structure));

  std::shared_ptr<const Core::LinAlg::Vector<double>> dis_avg =
      mt_energy_.average(disnp, *global_state().get_dis_n(), get_int_param());
  std::shared_ptr<const Core::LinAlg::Vector<double>> vel_avg =
      mt_energy_.average(velnp, *global_state().get_vel_n(), get_int_param());

  str_model.determine_energy(*dis_avg, vel_avg.get(), true);
  mt_energy_.int_energy_np_ = eval_data().get_energy_data(Solid::internal_energy);
  mt_energy_.kin_energy_np_ = eval_data().get_energy_data(Solid::kinetic_energy);
  global_state().get_fext_np()->dot(*dis_avg, &mt_energy_.ext_energy_np_);

  Core::IO::cout(Core::IO::debug) << __LINE__ << " -- " << __PRETTY_FUNCTION__ << "\n";
  mt_energy_.print(Core::IO::cout.os(Core::IO::debug));

  return mt_energy_.get_total();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::update_structural_energy()
{
  if (not mt_energy_.store_energy_n()) return;

  get_total_mid_time_str_energy(*global_state().get_dis_np());
  mt_energy_.copy_np_to_n();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::determine_optional_quantity()
{
  check_init_setup();
  model_eval().determine_optional_quantity();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::output_step_state(Core::IO::DiscretizationWriter& iowriter) const
{
  check_init_setup();
  model_eval().output_step_state(iowriter);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::monitor_dbc(Core::IO::DiscretizationWriter& writer) const
{
  monitor_dbc_ptr_->execute(writer);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::runtime_pre_output_step_state()
{
  check_init_setup();
  model_eval().runtime_pre_output_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::runtime_output_step_state() const
{
  check_init_setup();
  model_eval().runtime_output_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::post_output()
{
  check_init_setup();
  model_eval().post_output();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::reset_step_state()
{
  check_init_setup();
  model_eval().reset_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::Integrator::get_condensed_update_norm(
    const enum NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();

  double myupdatenorm = eval_data_ptr_->get_my_update_norm(qtype);
  const enum ::NOX::Abstract::Vector::NormType normtype =
      eval_data_ptr_->get_update_norm_type(qtype);

  return get_condensed_global_norm(qtype, normtype, myupdatenorm);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::Integrator::get_condensed_previous_sol_norm(
    const enum NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();

  double myprevsolnorm = eval_data_ptr_->get_my_previous_sol_norm(qtype);
  const enum ::NOX::Abstract::Vector::NormType normtype =
      eval_data_ptr_->get_update_norm_type(qtype);

  return get_condensed_global_norm(qtype, normtype, myprevsolnorm);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::Integrator::get_condensed_solution_update_rms(
    const enum NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();
  // global relative mean square norm
  double grmsnorm = 0.0;
  // get proc data
  double myrmsnorm = eval_data_ptr_->get_my_rms_norm(qtype);
  // get total dof number
  int gdofnumber = get_condensed_dof_number(qtype);
  // sum over all procs
  grmsnorm = Core::Communication::sum_all(myrmsnorm, gstate_ptr_->get_comm());

  // calculate the root mean square and return the value
  return (std::sqrt(grmsnorm) / static_cast<double>(gdofnumber));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::Integrator::get_condensed_dof_number(
    const enum NOX::Nln::StatusTest::QuantityType& qtype) const
{
  check_init_setup();
  // global dof number of the given quantity
  int gdofnumber = 0.0;
  int mydofnumber = eval_data_ptr_->get_my_dof_number(qtype);
  gdofnumber = Core::Communication::sum_all(mydofnumber, gstate_ptr_->get_comm());
  return gdofnumber;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::Integrator::get_condensed_global_norm(
    const enum NOX::Nln::StatusTest::QuantityType& qtype,
    const enum ::NOX::Abstract::Vector::NormType& normtype, double& mynorm) const
{
  double gnorm = 0;

  switch (normtype)
  {
    case ::NOX::Abstract::Vector::OneNorm:
    {
      gnorm = Core::Communication::sum_all(mynorm, gstate_ptr_->get_comm());
      break;
    }
    case ::NOX::Abstract::Vector::TwoNorm:
    {
      gnorm = Core::Communication::sum_all(mynorm, gstate_ptr_->get_comm());
      gnorm = std::sqrt(gnorm);
      break;
    }
    case ::NOX::Abstract::Vector::MaxNorm:
    {
      gnorm = Core::Communication::max_all(mynorm, gstate_ptr_->get_comm());
      break;
    }
  }
  return gnorm;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluatorManager& Solid::Integrator::model_eval()
{
  check_init();
  return *modelevaluator_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::ModelEvaluatorManager& Solid::Integrator::model_eval() const
{
  check_init();
  return *modelevaluator_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Solid::ModelEvaluatorManager> Solid::Integrator::model_eval_ptr() const
{
  check_init();
  return modelevaluator_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Generic& Solid::Integrator::evaluator(const Inpar::Solid::ModelType& mt)
{
  check_init_setup();
  return model_eval().evaluator(mt);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::ModelEvaluator::Generic& Solid::Integrator::evaluator(
    const Inpar::Solid::ModelType& mt) const
{
  check_init_setup();
  return model_eval().evaluator(mt);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::ModelEvaluator::Data& Solid::Integrator::eval_data() const
{
  check_init();
  return *eval_data_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Data& Solid::Integrator::eval_data()
{
  check_init();
  return *eval_data_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataSDyn& Solid::Integrator::sdyn()
{
  check_init();
  return *sdyn_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataSDyn& Solid::Integrator::s_dyn() const
{
  check_init();
  return *sdyn_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataGlobalState& Solid::Integrator::global_state() const
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::BaseDataGlobalState& Solid::Integrator::global_state()
{
  check_init();
  return *gstate_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Dbc& Solid::Integrator::dbc()
{
  check_init();
  return *dbc_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::Dbc& Solid::Integrator::get_dbc() const
{
  check_init();
  return *dbc_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::Base& Solid::Integrator::tim_int() const
{
  check_init();
  return *timint_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::create_backup_state(const Core::LinAlg::Vector<double>& dir)
{
  check_init_setup();
  model_eval().create_backup_state(dir);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::recover_from_backup_state()
{
  check_init_setup();
  model_eval().recover_from_backup_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::post_time_loop()
{
  check_init_setup();
  model_eval().post_time_loop();

  Solid::ModelEvaluator::Structure& str_model =
      dynamic_cast<Solid::ModelEvaluator::Structure&>(evaluator(Inpar::Solid::model_structure));
  str_model.evaluate_analytical_error();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Integrator::MidTimeEnergy::MidTimeEnergy(const Integrator& integrator)
    : integrator_(integrator), avg_type_(Inpar::Solid::midavg_vague)
{ /* empty */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::MidTimeEnergy::print(std::ostream& os) const
{
  const double time_fac = integrator_.get_int_param();

  os << "--- Contributions of time step n+1 (current)\n";
  os << "strain energy                         = " << int_energy_np_ << "\n";
  os << "external energy (dead load/potential) = " << ext_energy_np_ << "\n";
  os << "kinetic energy                        = " << kin_energy_np_ << "\n";

  if (avg_type_ == Inpar::Solid::midavg_trlike)
  {
    os << "--- Contributions of time step n   (previously accepted)\n";
    os << "strain energy                         = " << int_energy_n_ << "\n";
    os << "external energy (dead load/potential) = " << ext_energy_n_ << "\n";
    os << "kinetic energy                        = " << kin_energy_n_ << "\n";
    os << std::string(40, '-') << "\n";
    os << "tim_int factor at t_n+1 (current)      = " << 1.0 - time_fac << "\n";
    os << "tim_int factor at t_n   (previous)     = " << time_fac << "\n";
  }
  os << std::string(40, '=') << "\n";
  os << "Total structural energy               = " << get_total() << "\n";
  os << std::string(40, '=') << "\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::Integrator::MidTimeEnergy::get_total() const
{
  const double fac_n = integrator_.get_int_param();
  const double fac_np = 1.0 - fac_n;

  double total_energy = 0.0;
  total_energy = int_energy_np_ - kin_energy_np_ - ext_energy_np_;
  if (avg_type_ == Inpar::Solid::midavg_trlike)
  {
    const double energy_n = int_energy_n_ - kin_energy_n_ - ext_energy_n_;
    total_energy = fac_np * total_energy + fac_n * energy_n;
  }

  return total_energy;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> Solid::Integrator::MidTimeEnergy::average(
    const Core::LinAlg::Vector<double>& state_np, const Core::LinAlg::Vector<double>& state_n,
    const double fac_n) const
{
  const double fac_np = 1.0 - fac_n;

  std::shared_ptr<Core::LinAlg::Vector<double>> state_avg =
      std::make_shared<Core::LinAlg::Vector<double>>(state_np);
  switch (avg_type_)
  {
    case Inpar::Solid::midavg_vague:
    case Inpar::Solid::midavg_trlike:
      return state_avg;
    case Inpar::Solid::midavg_imrlike:
    {
      state_avg->update(fac_n, state_n, fac_np);
      return state_avg;
    }
    default:
      FOUR_C_THROW("Don't know what to do for the given MidAvg type.");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::MidTimeEnergy::copy_np_to_n()
{
  kin_energy_n_ = kin_energy_np_;
  int_energy_n_ = int_energy_np_;
  ext_energy_n_ = ext_energy_np_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Integrator::MidTimeEnergy::is_correctly_configured() const
{
  FOUR_C_ASSERT(issetup_, "Call setup() first.");

  if (avg_type_ == Inpar::Solid::midavg_vague)
  {
    if (integrator_.s_dyn().get_dynamic_type() != Inpar::Solid::DynamicType::Statics) return false;
  }
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Integrator::MidTimeEnergy::store_energy_n() const
{
  return avg_type_ == Inpar::Solid::midavg_trlike;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Integrator::MidTimeEnergy::setup()
{
  avg_type_ = integrator_.s_dyn().get_mid_time_energy_type();
  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE
