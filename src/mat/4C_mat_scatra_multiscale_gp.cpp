// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_multiscale_gp.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_timint_ost.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

namespace
{
  struct GlobalMicroState
  {
    //! map between number of micro-scale discretization and micro-scale time integrator
    std::map<int, std::shared_ptr<ScaTra::TimIntOneStepTheta>> microdisnum_microtimint_map_;

    //! map between number of micro-scale discretization and number of associated macro-scale
    //! Gauss points
    std::map<int, int> microdisnum_nummacrogp_map_;
  };

  // Manage a global state within a singleton
  GlobalMicroState& global_micro_state()
  {
    static auto global_micro_state =
        Core::Utils::make_singleton_owner([]() { return std::make_unique<GlobalMicroState>(); });

    return *global_micro_state.instance(Core::Utils::SingletonAction::create);
  }

}  // namespace

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::ScatraMultiScaleGP::ScatraMultiScaleGP(
    const int ele_id, const int gp_id, const int microdisnum, const bool is_ale)
    : gp_id_(gp_id),
      ele_id_(ele_id),
      eleowner_(Global::Problem::instance()->get_dis("scatra")->element_row_map()->my_gid(ele_id)),
      microdisnum_(microdisnum),
      step_(0),
      phin_(nullptr),
      phinp_(nullptr),
      phidtn_(nullptr),
      phidtnp_(nullptr),
      hist_(nullptr),
      micro_output_(nullptr),
      restartname_(""),
      det_fn_(1.0),
      det_fnp_(1.0),
      ddet_fdtn_(0.0),
      ddet_fdtnp_(0.0),
      is_ale_(is_ale)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::ScatraMultiScaleGP::~ScatraMultiScaleGP()
{
  // decrement number of macro-scale Gauss points associated with micro-scale time integrator
  --global_micro_state().microdisnum_nummacrogp_map_[microdisnum_];

  // once all macro-scale Gauss point submaterials are removed, destruct micro-scale time integrator
  if (global_micro_state().microdisnum_nummacrogp_map_[microdisnum_] == 0)
    global_micro_state().microdisnum_microtimint_map_[microdisnum_] = nullptr;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::init()
{
  // extract micro-scale problem
  Global::Problem* microproblem = Global::Problem::instance(microdisnum_);

  // extract micro-scale discretization
  std::stringstream microdisname;
  microdisname << "scatra_multiscale_" << microdisnum_;
  std::shared_ptr<Core::FE::Discretization> microdis = microproblem->get_dis(microdisname.str());

  // instantiate and initialize micro-scale state vectors
  phin_ = Core::LinAlg::create_vector(*microdis->dof_row_map(), true);
  phinp_ = Core::LinAlg::create_vector(*microdis->dof_row_map(), true);
  phidtn_ = Core::LinAlg::create_vector(*microdis->dof_row_map(), true);
  phidtnp_ = Core::LinAlg::create_vector(*microdis->dof_row_map(), true);
  hist_ = Core::LinAlg::create_vector(*microdis->dof_row_map(), true);

  // set up micro-scale time integrator for micro-scale problem if not already done
  if (global_micro_state().microdisnum_microtimint_map_.find(microdisnum_) ==
          global_micro_state().microdisnum_microtimint_map_.end() or
      global_micro_state().microdisnum_microtimint_map_[microdisnum_] == nullptr)
  {
    // extract macro-scale parameter list
    const Teuchos::ParameterList& sdyn_macro =
        Global::Problem::instance()->scalar_transport_dynamic_params();

    // extract micro-scale parameter list and create deep copy
    std::shared_ptr<Teuchos::ParameterList> sdyn_micro = std::make_shared<Teuchos::ParameterList>(
        Global::Problem::instance(microdisnum_)->scalar_transport_dynamic_params());

    // preliminary safety check
    if (Global::Problem::instance(microdisnum_)->n_dim() != 1)
    {
      FOUR_C_THROW(
          "Must have one-dimensional micro scale in multi-scale simulations of scalar transport "
          "problems!");
    }
    if (Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(sdyn_macro, "TIMEINTEGR") !=
            Inpar::ScaTra::timeint_one_step_theta or
        Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(
            *sdyn_micro, "TIMEINTEGR") != Inpar::ScaTra::timeint_one_step_theta)
    {
      FOUR_C_THROW(
          "Multi-scale calculations for scalar transport only implemented for one-step-theta time "
          "integration scheme!");
    }
    if (sdyn_macro.get<bool>("SKIPINITDER") != sdyn_micro->get<bool>("SKIPINITDER"))
      FOUR_C_THROW("Flag SKIPINITDER in input file must be equal on macro and micro scales!");
    if (sdyn_macro.get<double>("TIMESTEP") != sdyn_micro->get<double>("TIMESTEP"))
      FOUR_C_THROW("Must have identical time step size on macro and micro scales!");
    if (sdyn_macro.get<int>("NUMSTEP") != sdyn_micro->get<int>("NUMSTEP"))
      FOUR_C_THROW("Must have identical number of time steps on macro and micro scales!");
    if (sdyn_macro.get<double>("THETA") != sdyn_micro->get<double>("THETA"))
      FOUR_C_THROW(
          "Must have identical one-step-theta time integration factor on macro and micro scales!");
    if (microdis->num_global_elements() == 0)
      FOUR_C_THROW("No elements in TRANSPORT ELEMENTS section of micro-scale input file!");

    // Extract all x-coordinates of the microscale discretization, sort them, and check that the
    // first and last x-coordinates are 0 and > 0, respectively.
    {
      // The micro discretization is replicated on all ranks, so this will catch all x-coordinates.
      std::vector<double> xcoords;
      xcoords.reserve(microdis->num_my_row_nodes());
      for (const auto& node : microdis->my_row_node_range())
      {
        xcoords.push_back(node->x()[0]);
      }
      std::ranges::sort(xcoords);

      FOUR_C_ASSERT_ALWAYS(xcoords[0] == 0.0,
          "First x-coordinate of micro-scale discretization must be 0! First x-coordinate is {}",
          xcoords[0]);

      FOUR_C_ASSERT_ALWAYS(xcoords.back() > 0.0,
          "Last x-coordinate of micro-scale discretization must be > 0! Last x-coordinate is {}",
          xcoords.back());
    }

    // extract multi-scale coupling conditions from micro-scale discretization
    std::vector<const Core::Conditions::Condition*> conditions;
    microdis->get_condition("ScatraMultiScaleCoupling", conditions);

    // safety check
    if (conditions.size() == 0)
      FOUR_C_THROW(
          "Couldn't extract multi-scale coupling condition from micro-scale discretization!");

    // loop over all multi-scale coupling conditions
    for (auto& condition : conditions)
    {
      // extract nodal cloud
      const std::vector<int>* const nodeids = condition->get_nodes();
      if (nodeids == nullptr)
        FOUR_C_THROW("Multi-scale coupling condition does not have nodal cloud!");

      // loop over all nodes in nodal cloud
      for (int inode : *nodeids)
      {
        if (microdis->node_row_map()->my_gid(inode))
        {
          // extract node from micro-scale discretization
          Core::Nodes::Node* node = microdis->g_node(inode);

          // safety checks
          if (node == nullptr)
          {
            FOUR_C_THROW(
                "Cannot extract node with global ID {} from micro-scale discretization!", inode);
          }
          else if (node->x()[0] <= 0.0)
            FOUR_C_THROW(
                "Multi-scale coupling condition must be enforced on a node with coordinate > 0!");
        }
      }
    }

    // add proxy of velocity related degrees of freedom to scatra discretization
    std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux =
        std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
            Global::Problem::instance(microdisnum_)->n_dim() + 1, 0, 0, true);
    if (microdis->add_dof_set(dofsetaux) != 1)
      FOUR_C_THROW("Micro-scale discretization has illegal number of dofsets!");

    // finalize discretization
    microdis->fill_complete(true, false, false);

    // get solver number
    const int linsolvernumber = sdyn_micro->get<int>("LINEAR_SOLVER");

    // check solver number
    if (linsolvernumber < 0)
    {
      FOUR_C_THROW(
          "No linear solver defined for scalar field in input file for micro scale! Please set "
          "LINEAR_SOLVER in SCALAR TRANSPORT DYNAMIC to a valid number!");
    }

    // create solver
    std::shared_ptr<Core::LinAlg::Solver> solver = std::make_shared<Core::LinAlg::Solver>(
        Global::Problem::instance(microdisnum_)->solver_params(linsolvernumber),
        microdis->get_comm(), Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));

    // provide solver with null space information if necessary
    microdis->compute_null_space_if_necessary(solver->params());

    // supplementary parameter list
    std::shared_ptr<Teuchos::ParameterList> extraparams =
        std::make_shared<Teuchos::ParameterList>();
    extraparams->set<bool>("isale", false);
    extraparams->sublist("TURBULENT INFLOW") =
        Global::Problem::instance(microdisnum_)->fluid_dynamic_params().sublist("TURBULENT INFLOW");
    extraparams->sublist("TURBULENCE MODEL") =
        Global::Problem::instance(microdisnum_)->fluid_dynamic_params().sublist("TURBULENCE MODEL");

    // instantiate and initialize micro-scale time integrator
    global_micro_state().microdisnum_microtimint_map_[microdisnum_] =
        std::make_shared<ScaTra::TimIntOneStepTheta>(
            microdis, solver, sdyn_micro, extraparams, nullptr, microdisnum_);
    global_micro_state().microdisnum_microtimint_map_[microdisnum_]->init();
    global_micro_state().microdisnum_microtimint_map_[microdisnum_]->set_number_of_dof_set_velocity(
        1);
    global_micro_state().microdisnum_microtimint_map_[microdisnum_]->setup();

    // set initial velocity field
    global_micro_state()
        .microdisnum_microtimint_map_[microdisnum_]
        ->set_velocity_field_from_function();

    // create counter for number of macro-scale Gauss points associated with micro-scale time
    // integrator
    global_micro_state().microdisnum_nummacrogp_map_[microdisnum_] = 0;
  }

  // increment counter
  ++global_micro_state().microdisnum_nummacrogp_map_[microdisnum_];

  // extract initial state vectors from micro-scale time integrator
  phin_->scale(1., *global_micro_state().microdisnum_microtimint_map_[microdisnum_]->phin());
  phinp_->scale(1., *global_micro_state().microdisnum_microtimint_map_[microdisnum_]->phinp());

  // create new result file
  new_result_file();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::prepare_time_step(const std::vector<double>& phinp_macro)
{
  // extract micro-scale time integrator
  const std::shared_ptr<ScaTra::TimIntOneStepTheta>& microtimint =
      global_micro_state().microdisnum_microtimint_map_[microdisnum_];

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
      micro_visualization_writer_, phinp_macro, step_,
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time());

  // prepare time step
  microtimint->prepare_time_step();

  // clear state in micro-scale time integrator
  microtimint->clear_state();

  // increment time step
  ++step_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::evaluate(const std::vector<double>& phinp_macro, double& q_micro,
    std::vector<double>& dq_dphi_micro, const double detFnp, const bool solve)
{
  // extract micro-scale time integrator
  const std::shared_ptr<ScaTra::TimIntOneStepTheta>& microtimint =
      global_micro_state().microdisnum_microtimint_map_[microdisnum_];

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
      micro_visualization_writer_, phinp_macro, step_,
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time());

  if (is_ale_)
  {
    // update determinant of deformation gradient
    det_fnp_ = detFnp;

    // calculate time derivative and pass to micro time integration as reaction coefficient
    calculate_ddet_f_dt(*microtimint);
    microtimint->set_macro_micro_rea_coeff(ddet_fdtnp_);
  }

  if (step_ == 0 or !solve)
  {
    // only evaluate the micro-scale coupling quantities without solving the entire micro-scale
    // problem relevant for truly partitioned multi-scale simulations or for calculation of initial
    // time derivative of macro-scale state vector
    microtimint->evaluate_macro_micro_coupling();
  }
  else
  {
    // solve micro-scale problem
    // note that it is not necessary to transfer the final micro-scale state vectors back to the
    // Gauss-point submaterial due to RCP usage
    microtimint->solve();
  }

  // transfer micro-scale coupling quantities to macro scale
  q_micro = -microtimint->q();
  dq_dphi_micro = microtimint->dq_dphi();
  for (double& dq_dphi_micro_component : dq_dphi_micro) dq_dphi_micro_component *= -1.0;

  // clear state in micro-scale time integrator
  microtimint->clear_state();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::ScatraMultiScaleGP::evaluate_mean_concentration() const
{
  // extract micro-scale discretization
  Core::FE::Discretization& discret =
      *global_micro_state().microdisnum_microtimint_map_[microdisnum_]->discretization();

  // set micro-scale state vector
  discret.clear_state();
  discret.set_state("phinp", *phinp_);

  // set parameters for micro-scale elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_total_and_mean_scalars, eleparams);
  eleparams.set("inverting", false);
  eleparams.set("calc_grad_phi", false);

  // initialize result vector: first component = concentration integral, second component = domain
  // integral
  const std::shared_ptr<Core::LinAlg::SerialDenseVector> integrals =
      std::make_shared<Core::LinAlg::SerialDenseVector>(2);

  // evaluate concentration and domain integrals on micro scale
  discret.evaluate_scalars(eleparams, *integrals);

  // clear discretization
  discret.clear_state();

  // compute and return mean concentration on micro scale
  return (*integrals)[0] / (*integrals)[1];
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
double Mat::ScatraMultiScaleGP::evaluate_mean_concentration_time_derivative() const
{
  // extract micro-scale discretization
  Core::FE::Discretization& discret =
      *global_micro_state().microdisnum_microtimint_map_[microdisnum_]->discretization();

  // set micro-scale state vector
  discret.clear_state();
  discret.set_state("phidtnp", *phidtnp_);

  // set parameters for micro-scale elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_mean_scalar_time_derivatives, eleparams);

  // initialize result vector: first component = integral of concentration time derivative, second
  // component = integral of domain
  const std::shared_ptr<Core::LinAlg::SerialDenseVector> integrals =
      std::make_shared<Core::LinAlg::SerialDenseVector>(2);

  // evaluate integrals of domain and time derivative of concentration on micro scale
  discret.evaluate_scalars(eleparams, *integrals);

  // clear discretization
  discret.clear_state();

  // compute and return mean concentration time derivative on micro scale
  return (*integrals)[0] / (*integrals)[1];
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::update()
{
  if (is_ale_)
  {
    // Update detF
    det_fn_ = det_fnp_;
    ddet_fdtn_ = ddet_fdtnp_;
  }

  // extract micro-scale time integrator
  std::shared_ptr<ScaTra::TimIntOneStepTheta> microtimint =
      global_micro_state().microdisnum_microtimint_map_[microdisnum_];

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
      micro_visualization_writer_, std::vector<double>(0, 0.), step_,
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time());

  // update micro-scale time integrator
  microtimint->update();

  // clear state in micro-scale time integrator
  microtimint->clear_state();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::new_result_file()
{
  // get properties from macro scale
  std::shared_ptr<Core::IO::OutputControl> macrocontrol =
      Global::Problem::instance()->output_control_file();
  std::string microprefix = macrocontrol->restart_name();
  std::string micronewprefix = macrocontrol->new_output_file_name();

  // extract micro-scale problem and discretization
  Global::Problem* microproblem = Global::Problem::instance(microdisnum_);
  std::stringstream microdisname;
  microdisname << "scatra_multiscale_" << microdisnum_;
  std::shared_ptr<Core::FE::Discretization> microdis = microproblem->get_dis(microdisname.str());

  // figure out prefix of micro-scale restart files
  restartname_ = new_result_file_path(microprefix);

  // figure out new prefix for micro-scale output files
  const std::string newfilename = new_result_file_path(micronewprefix);

  if (eleowner_)
  {
    const int ndim = microproblem->n_dim();
    const int restart = Global::Problem::instance()->restart();
    bool adaptname = true;

    // in case of restart, the new output file name has already been adapted
    if (restart) adaptname = false;

    std::shared_ptr<Core::IO::OutputControl> microcontrol =
        std::make_shared<Core::IO::OutputControl>(microdis->get_comm(), "Scalar_Transport",
            microproblem->spatial_approximation_type(), "micro-input-file-not-known", restartname_,
            newfilename, ndim, restart,
            Global::Problem::instance(microdisnum_)->io_params().get<int>("FILESTEPS"),
            Global::Problem::instance(microdisnum_)->io_params().get<bool>("OUTPUT_BIN"),
            adaptname);

    micro_output_ = std::make_shared<Core::IO::DiscretizationWriter>(
        microdis, microcontrol, microproblem->spatial_approximation_type());
    micro_output_->set_output(microcontrol);
    micro_output_->write_mesh(
        step_, Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time());

    micro_visualization_writer_ = std::make_shared<Core::IO::DiscretizationVisualizationWriterMesh>(
        microdis,
        Core::IO::visualization_parameters_factory(
            Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"), *microcontrol,
            Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time()));
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::string Mat::ScatraMultiScaleGP::new_result_file_path(const std::string& newprefix)
{
  std::string newfilename;

  // create path from string to extract only filename prefix
  const std::filesystem::path path(newprefix);
  const std::string newfileprefix = path.filename().string();

  const size_t posn = newfileprefix.rfind('-');
  if (posn != std::string::npos)
  {
    const std::string number = newfileprefix.substr(posn + 1);
    const std::string prefix = newfileprefix.substr(0, posn);

    // recombine path and file
    const std::filesystem::path parent_path(path.parent_path());
    const std::filesystem::path filen_name(prefix);
    const std::filesystem::path recombined_path = parent_path / filen_name;

    std::ostringstream s;
    s << recombined_path.string() << "_microdis" << microdisnum_ << "_el" << ele_id_ << "_gp"
      << gp_id_ << "-" << number;
    newfilename = s.str();
  }
  else
  {
    std::ostringstream s;
    s << newprefix << "_microdis" << microdisnum_ << "_el" << ele_id_ << "_gp" << gp_id_;
    newfilename = s.str();
  }

  return newfilename;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::collect_and_write_output_data()
{
  // skip ghosted macro-scale elements
  if (eleowner_)
  {
    // extract micro-scale time integrator
    std::shared_ptr<ScaTra::TimIntOneStepTheta> microtimint =
        global_micro_state().microdisnum_microtimint_map_[microdisnum_];

    // set current state in micro-scale time integrator
    microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
        micro_visualization_writer_, std::vector<double>(0, 0.0), step_,
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time());

    // output micro-scale results
    if (microtimint->is_result_step()) microtimint->write_runtime_output();

    // clear state in micro-scale time integrator
    microtimint->clear_state();
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::output()
{
  // skip ghosted macro-scale elements
  if (eleowner_)
  {
    // extract micro-scale time integrator
    std::shared_ptr<ScaTra::TimIntOneStepTheta> microtimint =
        global_micro_state().microdisnum_microtimint_map_[microdisnum_];

    // set current state in micro-scale time integrator
    microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
        micro_visualization_writer_, std::vector<double>(0, 0.0), step_,
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time());

    // output micro-scale results
    if (microtimint->is_result_step()) microtimint->write_result();

    // output micro-scale restart information
    if (microtimint->is_restart_step())
    {
      microtimint->write_restart();
      if (is_ale_)
      {
        microtimint->disc_writer()->write_double("detFn", det_fn_);
        microtimint->disc_writer()->write_double("ddetFdtn", ddet_fdtn_);
      }
    }

    // clear state in micro-scale time integrator
    microtimint->clear_state();
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::read_restart()
{
  // extract micro-scale time integrator
  std::shared_ptr<ScaTra::TimIntOneStepTheta> microtimint =
      global_micro_state().microdisnum_microtimint_map_[microdisnum_];

  // extract restart step
  step_ = Global::Problem::instance()->restart();

  // set current state in micro-scale time integrator
  microtimint->set_state(phin_, phinp_, phidtn_, phidtnp_, hist_, micro_output_,
      micro_visualization_writer_, std::vector<double>(0, 0.0), step_,
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time());

  // read restart on micro scale
  auto inputcontrol = std::make_shared<Core::IO::InputControl>(restartname_, true);
  microtimint->read_restart(step_, inputcontrol);

  // safety check
  if (microtimint->step() != step_)
  {
    FOUR_C_THROW("Time step mismatch!");
  }

  std::shared_ptr<Core::IO::DiscretizationReader> reader(nullptr);
  if (inputcontrol == nullptr)
  {
    reader = std::make_shared<Core::IO::DiscretizationReader>(
        microtimint->discretization(), Global::Problem::instance()->input_control_file(), step_);
  }
  else
  {
    reader = std::make_shared<Core::IO::DiscretizationReader>(
        microtimint->discretization(), inputcontrol, step_);
  }

  if (is_ale_)
  {
    det_fn_ = reader->read_double("detFn");
    ddet_fdtn_ = reader->read_double("ddetFdtn");
  }

  // clear state in micro-scale time integrator
  microtimint->clear_state();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::calculate_ddet_f_dt(ScaTra::TimIntOneStepTheta& microtimint)
{
  const double dt = microtimint.dt();

  switch (microtimint.method_name())
  {
    case Inpar::ScaTra::TimeIntegrationScheme::timeint_one_step_theta:
    {
      const double theta = microtimint.scatra_parameter_list()->get<double>("THETA");

      const double part1 = (det_fnp_ - det_fn_) / dt;
      const double part2 = (1.0 - theta) * ddet_fdtn_;
      ddet_fdtnp_ = 1.0 / theta * (part1 - part2);

      break;
    }
    default:
    {
      FOUR_C_THROW("time integration scheme not supported to calculate d detF / d t.");
      break;
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMultiScaleGP::set_time_stepping(const double dt, const double time, const int step)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  FOUR_C_ASSERT(dt > 0.0, "Time step for micro scale must be positive.");
  FOUR_C_ASSERT(time >= 0.0, "Time for micro scale must be positive.");
  FOUR_C_ASSERT(step >= 0, "Number of step for micro scale must be positive.");
#endif

  std::shared_ptr<ScaTra::TimIntOneStepTheta> microtimint =
      global_micro_state().microdisnum_microtimint_map_[microdisnum_];
  microtimint->set_dt(dt);
  microtimint->set_time_step(time, step);
}
FOUR_C_NAMESPACE_CLOSE
