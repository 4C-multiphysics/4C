// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_base.hpp"

#include "4C_adapter_str_pasiwrapper.hpp"
#include "4C_beaminteraction_str_model_evaluator.hpp"
#include "4C_comm_utils.hpp"
#include "4C_constraint_framework_model_evaluator.hpp"
#include "4C_contact_input.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_structure_new_factory.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_factory.hpp"
#include "4C_structure_new_resulttest.hpp"
#include "4C_structure_new_timint_basedataio_runtime_vtp_output.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::Base::Base()
    : StructureNew(),
      isinit_(false),
      issetup_(false),
      isrestarting_(false),
      state_is_insync_with_noxgroup_(true),
      dataio_(nullptr),
      datasdyn_(nullptr),
      dataglobalstate_(nullptr),
      int_ptr_(nullptr),
      dbc_ptr_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::init(const std::shared_ptr<Solid::TimeInt::BaseDataIO> dataio,
    const std::shared_ptr<Solid::TimeInt::BaseDataSDyn> datasdyn,
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> dataglobalstate)
{
  // ---------------------------------------------------------------------------
  // We need to call setup() after init()
  // ---------------------------------------------------------------------------
  issetup_ = false;

  // ---------------------------------------------------------------------------
  // initialize the data container ptrs
  // ---------------------------------------------------------------------------
  dataio_ = dataio;
  datasdyn_ = datasdyn;
  dataglobalstate_ = dataglobalstate;

  // ---------------------------------------------------------------------------
  // set isInit flag
  // ---------------------------------------------------------------------------
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::setup()
{
  check_init();

  // ---------------------------------------------------------------------------
  // Create the Dirichlet Boundary Condition handler
  // ---------------------------------------------------------------------------
  dbc_ptr_ = Solid::build_dbc(data_sdyn());
  /* FixMe It would be sufficient to use a constant discretization,
   * unfortunately this wasn't considered during the implementation of the
   * discretization routines. Therefore many methods need a slight modification
   * (most times adding a "const" should fix the problem).          hiermeier */
  std::shared_ptr<Core::FE::Discretization> discret_ptr = data_global_state().get_discret();
  dbc_ptr_->init(
      discret_ptr, data_global_state().get_freact_np(), Core::Utils::shared_ptr_from_ref(*this));
  dbc_ptr_->setup();

  // ---------------------------------------------------------------------------
  // Create the explicit/implicit integrator
  // ---------------------------------------------------------------------------
  int_ptr_ = Solid::build_integrator(data_sdyn());
  int_ptr_->init(data_s_dyn_ptr(), data_global_state_ptr(), data_io_ptr(), dbc_ptr_,
      Core::Utils::shared_ptr_from_ref(*this));
  int_ptr_->setup();

  // Initialize the output of system energy
  if (dataio_->get_write_energy_every_n_step())
  {
    select_energy_types_to_be_written();

    if (dataglobalstate_->get_my_rank() == 0) initialize_energy_file_stream_and_write_headers();
  }

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::post_setup()
{
  check_init_setup();
  int_ptr_->post_setup();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::reset()
{
  FOUR_C_THROW(
      "Reset of all class variables is not yet implemented for "
      "the modelevaluator!");
  // ModelEvaluatorManager().reset();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::reset_step()
{
  check_init_setup();

  int_ptr_->reset_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::TimeInt::Base::not_finished() const
{
  check_init_setup();

  // check for early stopping
  const bool early_stop = int_ptr_->early_stopping();
  if (early_stop)
  {
    // Simulation is finished regardless of the simulation time or the time step.
    return false;
  }

  // check the current time
  const double timenp = dataglobalstate_->get_time_np();
  const double timemax = datasdyn_->get_time_max();
  const double dt = dataglobalstate_->get_delta_time()[0];
  // check the step counter
  const int stepnp = dataglobalstate_->get_step_np();
  const int stepmax = datasdyn_->get_step_max();

  return (timenp <= timemax + 1.0e-8 * dt and stepnp <= stepmax);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::set_restart(int stepn, double timen,
    std::shared_ptr<Core::LinAlg::Vector<double>> disn,
    std::shared_ptr<Core::LinAlg::Vector<double>> veln,
    std::shared_ptr<Core::LinAlg::Vector<double>> accn,
    std::shared_ptr<std::vector<char>> elementdata, std::shared_ptr<std::vector<char>> nodedata)
{
  check_init_setup();

  FOUR_C_THROW("SetRestartState() is deprecated, use the read_restart() routine instead!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Map& Solid::TimeInt::Base::get_mass_domain_map() const
{
  check_init_setup();
  return dataglobalstate_->get_mass_matrix()->domain_map();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MapExtractor> Solid::TimeInt::Base::get_dbc_map_extractor()
{
  check_init_setup();
  return dbc_ptr_->get_dbc_map_extractor();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::MapExtractor> Solid::TimeInt::Base::get_dbc_map_extractor()
    const
{
  check_init_setup();
  return dbc_ptr_->get_dbc_map_extractor();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::Conditions::LocsysManager> Solid::TimeInt::Base::locsys_manager()
{
  check_init_setup();
  return dbc_ptr_->loc_sys_manager_ptr();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::ModelEvaluator::Generic& Solid::TimeInt::Base::model_evaluator(
    Inpar::Solid::ModelType mtype) const
{
  return integrator().model_eval().evaluator(mtype);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::ModelEvaluator::Generic& Solid::TimeInt::Base::model_evaluator(Inpar::Solid::ModelType mtype)
{
  return integrator().model_eval().evaluator(mtype);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::Base::tim_int_param() const
{
  check_init_setup();
  return int_ptr_->get_int_param();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::resize_m_step_tim_ada()
{
  check_init_setup();
  // resize time and stepsize fields
  const double& timen = dataglobalstate_->get_time_n();
  dataglobalstate_->get_multi_time().resize(-1, 0, timen);
  const double& dtn = dataglobalstate_->get_delta_time()[0];
  dataglobalstate_->get_delta_time().resize(-1, 0, dtn);

  // resize state vectors, AB2 is a 2-step method, thus we need two
  // past steps at t_{n} and t_{n-1}
  const Core::LinAlg::Map* dofrowmap_ptr = dataglobalstate_->dof_row_map_view();
  dataglobalstate_->get_multi_dis().resize(-1, 0, dofrowmap_ptr, true);
  dataglobalstate_->get_multi_vel().resize(-1, 0, dofrowmap_ptr, true);
  dataglobalstate_->get_multi_acc().resize(-1, 0, dofrowmap_ptr, true);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::update()
{
  check_init_setup();
  int_ptr_->pre_update();
  int_ptr_->update_structural_energy();
  int_ptr_->update_step_state();
  update_step_time();
  set_number_of_nonlinear_iterations();
  int_ptr_->update_step_element();
  int_ptr_->post_update();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::update_step_time()
{
  check_init_setup();
  double& timenp = dataglobalstate_->get_time_np();
  int& stepnp = dataglobalstate_->get_step_np();
  int& stepn = dataglobalstate_->get_step_n();

  // --------------------------------------------------------------------------
  // update old time and step variables
  // --------------------------------------------------------------------------
  dataglobalstate_->get_multi_time().update_steps(timenp);
  stepn = stepnp;

  // --------------------------------------------------------------------------
  // update the new time and step variables
  // --------------------------------------------------------------------------
  // get current time step size
  const double& dtn = dataglobalstate_->get_delta_time()[0];
  dataglobalstate_->get_delta_time().update_steps(dtn);
  timenp += dtn;
  stepnp += 1;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::set_number_of_nonlinear_iterations()
{
  int nlniter = 0;

  if (data_sdyn().get_nox_params().isSublist("Output"))
  {
    const Teuchos::ParameterList& nox_output = data_sdyn().get_nox_params().sublist("Output");
    if (nox_output.isParameter("Nonlinear Iterations"))
      nlniter = nox_output.get<int>("Nonlinear Iterations");
  }

  dataglobalstate_->set_nln_iteration_number(nlniter);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::select_energy_types_to_be_written()
{
  Solid::ModelEvaluator::Data& evaldata = int_ptr_->eval_data();

  // decide which types of energy contributions shall be written separately
  const std::set<Inpar::Solid::ModelType>& mtypes = datasdyn_->get_model_types();

  std::set<Inpar::Solid::ModelType>::const_iterator model_iter;
  for (model_iter = mtypes.begin(); model_iter != mtypes.end(); ++model_iter)
  {
    switch (*model_iter)
    {
      case Inpar::Solid::model_structure:
      {
        evaldata.insert_energy_type_to_be_considered(Solid::internal_energy);
        evaldata.insert_energy_type_to_be_considered(Solid::kinetic_energy);
        break;
      }
      case Inpar::Solid::model_beaminteraction:
      {
        Solid::ModelEvaluator::BeamInteractionModelEvaluator const beaminteraction_evaluator =
            dynamic_cast<Solid::ModelEvaluator::BeamInteractionModelEvaluator const&>(
                int_ptr_->model_eval_ptr()->evaluator(Inpar::Solid::model_beaminteraction));

        if (beaminteraction_evaluator.have_sub_model_type(
                BeamInteraction::SubModelType::submodel_beamcontact))
        {
          evaldata.insert_energy_type_to_be_considered(Solid::beam_contact_penalty_potential);
        }
        if (beaminteraction_evaluator.have_sub_model_type(
                BeamInteraction::SubModelType::submodel_potential))
        {
          evaldata.insert_energy_type_to_be_considered(Solid::beam_interaction_potential);
        }
        if (beaminteraction_evaluator.have_sub_model_type(
                BeamInteraction::SubModelType::submodel_crosslinking))
        {
          evaldata.insert_energy_type_to_be_considered(Solid::beam_to_beam_link_internal_energy);
          evaldata.insert_energy_type_to_be_considered(Solid::beam_to_beam_link_kinetic_energy);
        }
        if (beaminteraction_evaluator.have_sub_model_type(
                BeamInteraction::SubModelType::submodel_spherebeamlink))
        {
          evaldata.insert_energy_type_to_be_considered(Solid::beam_to_sphere_link_internal_energy);
          evaldata.insert_energy_type_to_be_considered(Solid::beam_to_sphere_link_kinetic_energy);
        }
        break;
      }
      case Inpar::Solid::model_constraints:
      {
        Solid::ModelEvaluator::Constraint const constraints_evaluator =
            dynamic_cast<Solid::ModelEvaluator::Constraint const&>(
                int_ptr_->model_eval_ptr()->evaluator(Inpar::Solid::model_constraints));

        if (constraints_evaluator.have_sub_model_type(Constraints::SubModelType::embeddedmesh))
        {
          evaldata.insert_energy_type_to_be_considered(Solid::embedded_mesh_penalty_potential);
        }
      }
      default:
      {
        break;
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::initialize_energy_file_stream_and_write_headers()
{
  auto& evaldata = int_ptr_->eval_data();

  dataio_->setup_energy_output_file();

  // write column headers to file
  dataio_->get_energy_output_stream() << std::setw(12) << "#timestep," << std::setw(24) << "time,";

  for (const auto& energy_data : evaldata.get_energy_data())
  {
    dataio_->get_energy_output_stream()
        << std::setw(36) << Solid::energy_type_to_string(energy_data.first) + ",";
  }

  dataio_->get_energy_output_stream() << std::setw(24) << "total_energy" << std::endl;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> Solid::TimeInt::Base::create_field_test()
{
  check_init_setup();
  std::shared_ptr<Solid::ResultTest> resulttest = std::make_shared<Solid::ResultTest>();
  resulttest->init(get_data_global_state(), integrator().eval_data());
  resulttest->setup();

  return resulttest;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::get_restart_data(std::shared_ptr<int> step, std::shared_ptr<double> time,
    std::shared_ptr<Core::LinAlg::Vector<double>> disnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> accnp,
    std::shared_ptr<std::vector<char>> elementdata, std::shared_ptr<std::vector<char>> nodedata)
{
  check_init_setup();
  // at some point we have to create a copy
  *step = dataglobalstate_->get_step_n();
  *time = dataglobalstate_->get_time_n();
  std::shared_ptr<const Core::FE::Discretization> discret_ptr =
      std::dynamic_pointer_cast<const Core::FE::Discretization>(dataglobalstate_->get_discret());
  *elementdata = *(discret_ptr->pack_my_elements());
  *nodedata = *(discret_ptr->pack_my_nodes());

  // get restart data is only for simple structure problems
  // hence if the model set is larger than one, we throw an error
  if (datasdyn_->get_model_types().size() > 1)
    FOUR_C_THROW("The get_restart_data routine supports the structural model case ONLY!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::prepare_output(bool force_prepare_timestep)
{
  check_init_setup();

  if ((dataio_->is_runtime_output_enabled() && force_prepare_timestep) ||
      dataio_->write_runtime_vtk_results_for_this_step(dataglobalstate_->get_step_np()) ||
      dataio_->write_runtime_vtp_results_for_this_step(dataglobalstate_->get_step_np()))
  {
    int_ptr_->runtime_pre_output_step_state();
  }
  // --- energy calculation ---------------------------------------------------
  if ((dataio_->get_write_energy_every_n_step() and
          (force_prepare_timestep ||
              dataglobalstate_->get_step_np() % dataio_->get_write_energy_every_n_step() == 0)))
  {
    Solid::ModelEvaluator::Data& evaldata = int_ptr_->eval_data();
    evaldata.clear_values_for_all_energy_types();

    int_ptr_->determine_energy();

    // sum processor-local values of all separate contributions into global value
    double energy_local = 0.0;
    double energy_global = 0.0;

    for (const auto& energy_data : evaldata.get_energy_data())
    {
      energy_local = energy_data.second;

      energy_global = Core::Communication::sum_all(energy_local, dataglobalstate_->get_comm());

      evaldata.set_value_for_energy_type(energy_global, energy_data.first);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::output(bool forced_writerestart)
{
  check_init_setup();
  output_step(forced_writerestart);
  // write Gmsh output
  write_gmsh_struct_output_step();
  int_ptr_->post_output();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::output_step(bool forced_writerestart)
{
  check_init_setup();
  // special treatment is necessary when restart is forced
  if (forced_writerestart)
  {
    // reset possible history data on element level
    reset_step();
    // restart has already been written or simulation has just started
    if (dataio_->should_write_restart_for_step(dataglobalstate_->get_step_n()) or
        dataglobalstate_->get_step_n() == Global::Problem::instance()->restart())
      return;
    // TODO: This if statement can be removed once
    // Solid::ModelEvaluator::BeamInteractionModelEvaluator::output_step_state is removed
    // if state already exists, add restart information
    if (dataio_->write_results_for_this_step(dataglobalstate_->get_step_n()))
    {
      add_restart_to_output_state();
      return;
    }
  }

  /* This flag indicates whether some form of output has already been written in the current time
   * step. It is passed along subroutines and prevents repeated initialization of output writer,
   * printing of state vectors, or similar.
   */
  bool datawritten = false;

  // output restart (try this first)
  // write restart step
  if (forced_writerestart || dataio_->should_write_restart_for_step(dataglobalstate_->get_step_n()))
  {
    output_restart(datawritten);
  }

  // output results (not necessary if restart in same step)
  // TODO: This if statement can be removed once
  // Solid::ModelEvaluator::BeamInteractionModelEvaluator::output_step_state is removed
  if (dataio_->is_write_state() and
      dataio_->write_results_for_this_step(dataglobalstate_->get_step_n()) and (not datawritten))
  {
    new_io_step(datawritten);
    output_state();
  }

  // output results during runtime ( not used for restart so far )
  if (dataio_->write_runtime_vtk_results_for_this_step(dataglobalstate_->get_step_n()) or
      dataio_->write_runtime_vtp_results_for_this_step(dataglobalstate_->get_step_n()))
  {
    runtime_output_state();
  }

  // write reaction forces
  if (dataio_->should_write_reaction_forces_for_this_step(dataglobalstate_->get_step_n()))
  {
    output_reaction_forces();
  }

  // output energy
  if (dataio_->should_write_energy_for_this_step(dataglobalstate_->get_step_n()))
  {
    output_energy();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::new_io_step(bool& datawritten)
{
  if (not datawritten)
  {
    // Make new step
    dataio_->get_output_ptr()->new_step(
        dataglobalstate_->get_step_n(), dataglobalstate_->get_time_n());

    datawritten = true;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::output_state()
{
  check_init_setup();
  Core::IO::DiscretizationWriter& iowriter = *(dataio_->get_output_ptr());

  output_state(iowriter, dataio_->is_first_output_of_run());

  dataio_->set_first_output_of_run(false);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::output_state(
    Core::IO::DiscretizationWriter& iowriter, bool write_owner) const
{
  // owner of elements is just written once because it does not change during
  // simulation (so far)
  iowriter.write_element_data(write_owner);
  iowriter.write_node_data(write_owner);

  int_ptr_->output_step_state(iowriter);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::runtime_output_state() const
{
  check_init_setup();
  int_ptr_->runtime_output_step_state();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::output_reaction_forces()
{
  check_init_setup();
  Core::IO::DiscretizationWriter& iowriter = *(dataio_->get_output_ptr());
  int_ptr_->monitor_dbc(iowriter);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::output_energy() const
{
  check_init_setup();

  if (dataglobalstate_->get_my_rank() == 0)
  {
    std::ostream& energy_output_stream = dataio_->get_energy_output_stream();

    energy_output_stream << std::setw(11) << dataglobalstate_->get_step_n() << std::setw(1) << ","
                         << std::scientific << std::setprecision(14) << std::setw(23)
                         << dataglobalstate_->get_time_n() << std::setw(1) << ",";

    Solid::ModelEvaluator::Data& evaldata = int_ptr_->eval_data();

    double total_energy = 0.0;

    for (const auto& energy_data : evaldata.get_energy_data())
    {
      energy_output_stream << std::setw(35) << energy_data.second << std::setw(1) << ",";
      total_energy += energy_data.second;
    }

    energy_output_stream << std::setw(24) << total_energy << std::endl;

    Core::IO::cout(Core::IO::verbose) << "\n\nOutput for energy written to file!" << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::output_restart(bool& datawritten)
{
  check_init_setup();

  std::shared_ptr<Core::IO::DiscretizationWriter> output_ptr = dataio_->get_output_ptr();
  // write restart output, please
  if (dataglobalstate_->get_step_n() != 0)
    output_ptr->write_mesh(dataglobalstate_->get_step_n(), dataglobalstate_->get_time_n());

  new_io_step(datawritten);

  output_ptr->write_element_data(dataio_->is_first_output_of_run());
  output_ptr->write_node_data(dataio_->is_first_output_of_run());
  dataio_->set_first_output_of_run(false);

  // add displacement, velocity and acceleration
  output_ptr->write_vector("displacement", dataglobalstate_->get_dis_n());
  output_ptr->write_vector("velocity", dataglobalstate_->get_vel_n());
  output_ptr->write_vector("acceleration", dataglobalstate_->get_acc_n());

  /* Add the restart information of the different time integrators and model
   * evaluators. */
  int_ptr_->write_restart(*output_ptr);

  // info dedicated to user's eyes staring at standard out
  if ((dataglobalstate_->get_my_rank() == 0) and (dataio_->get_print2_screen_every_n_step() > 0) and
      (step_old() % dataio_->get_print2_screen_every_n_step() == 0))
  {
    Core::IO::cout << "====== Restart for field 'Structure' written in step "
                   << dataglobalstate_->get_step_n() << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::add_restart_to_output_state()
{
  std::shared_ptr<Core::IO::DiscretizationWriter> output_ptr = dataio_->get_output_ptr();

  // output of velocity and acceleration
  output_ptr->write_vector("displacement", dataglobalstate_->get_dis_n());
  output_ptr->write_vector("velocity", dataglobalstate_->get_vel_n());
  output_ptr->write_vector("acceleration", dataglobalstate_->get_acc_n());

  /* Add the restart information of the different time integrators and model
   * evaluators. */
  int_ptr_->write_restart(*output_ptr, true);

  // finally add the missing mesh information, order is important here
  output_ptr->write_mesh(data_global_state().get_step_n(), data_global_state().get_time_n());

  // info dedicated to user's eyes staring at standard out
  if ((dataglobalstate_->get_my_rank() == 0) and (dataio_->get_print2_screen_every_n_step() > 0) and
      (step_old() % dataio_->get_print2_screen_every_n_step() == 0))
  {
    Core::IO::cout << "====== Restart for field 'Structure' written in step "
                   << dataglobalstate_->get_step_n() << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::write_gmsh_struct_output_step()
{
  check_init_setup();
  if (!dataio_->is_gmsh()) return;

  const std::string filename =
      Core::IO::Gmsh::get_file_name("struct", disc_writer()->output().file_name(),
          dataglobalstate_->get_step_np(), false, dataglobalstate_->get_my_rank());
  std::ofstream gmshfilecontent(filename.c_str());

  // add 'View' to Gmsh postprocessing file
  gmshfilecontent << "View \" " << "struct displacement \" {" << std::endl;
  // draw vector field 'struct displacement' for every element
  Core::IO::Gmsh::vector_field_dof_based_to_gmsh(
      *discretization(), dispn(), gmshfilecontent, 0, true);
  gmshfilecontent << "};" << std::endl;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::read_restart(const int stepn)
{
  check_init();
  // that restarting flag
  isrestarting_ = true;

  // create an input/output reader
  Core::IO::DiscretizationReader ioreader(
      *discretization(), Global::Problem::instance()->input_control_file(), stepn);
  dataglobalstate_->get_step_n() = stepn;
  dataglobalstate_->get_step_np() = stepn + 1;
  dataglobalstate_->get_multi_time() =
      TimeStepping::TimIntMStep(0, 0, ioreader.read_double("time"));
  const double& timen = dataglobalstate_->get_time_n();
  const double& dt = dataglobalstate_->get_delta_time()[0];
  dataglobalstate_->get_time_np() = timen + dt;

  // ---------------------------------------------------------------------------
  // The order is important at this point!
  // (0) read element and node data --> new discretization
  // (1) setup() the model evaluator and time integrator
  // (2) read and possibly overwrite the general dynamic state
  // (3) read specific time integrator and model evaluator data
  // ---------------------------------------------------------------------------
  // (0) read element and node data
  ioreader.read_history_data(stepn);

  // (1) setup() the model evaluator and time integrator
  /* Since we call a redistribution on the structural discretization, we have to
   * setup the structural time integration strategy at this point and not as
   * usually during the adapter call.                         hiermeier 05/16 */
  setup();

  // (2) read (or overwrite) the general dynamic state
  std::shared_ptr<Core::LinAlg::Vector<double>>& velnp = dataglobalstate_->get_vel_np();
  ioreader.read_vector(velnp, "velocity");
  dataglobalstate_->get_multi_vel().update_steps(*velnp);
  std::shared_ptr<Core::LinAlg::Vector<double>>& accnp = dataglobalstate_->get_acc_np();
  ioreader.read_vector(accnp, "acceleration");
  dataglobalstate_->get_multi_acc().update_steps(*accnp);

  // (3) read specific time integrator (forces, etc.) and model evaluator data
  int_ptr_->read_restart(ioreader);
  int_ptr_->post_setup();  // compute here the equilibrium system to account for initial
  //  displacement/velocity.

  // short screen output
  if (dataglobalstate_->get_my_rank() == 0)
    Core::IO::cout << "====== Restart of the structural simulation from step " << stepn
                   << Core::IO::endl;

  // end of restarting
  isrestarting_ = false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Discretization> Solid::TimeInt::Base::discretization()
{
  check_init();
  return dataglobalstate_->get_discret();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::set_action_type(const Core::Elements::ActionType& action)
{
  check_init_setup();
  int_ptr_->eval_data().set_action_type(action);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Base::group_id() const
{
  return Global::Problem::instance()->get_communicators().group_id();
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Base::post_update() { int_ptr_->post_update(); }

void Solid::TimeInt::Base::post_time_loop() { int_ptr_->post_time_loop(); }


std::string Solid::TimeInt::Base::method_title() const
{
  return std::string(EnumTools::enum_name(method_name()));
}

FOUR_C_NAMESPACE_CLOSE
