// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_base.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_adapter_porofluid_pressure_based_wrapper.hpp"
#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"
#include "4C_porofluid_pressure_based_elast_utils.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_timint_meshtying_strategy_artery.hpp"
#include "4C_scatra_timint_poromulti.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraBaseAlgorithm::PorofluidElastScatraBaseAlgorithm(
    MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : AlgorithmBase(comm, globaltimeparams),
      porofluid_elast_algo_(nullptr),
      scatra_algo_(nullptr),
      flux_reconstruction_active_(false),
      nds_porofluid_scatra_(-1),
      timer_timestep_("PorofluidElastScatraBaseAlgorithm", true),
      dt_timestep_(0.0),
      divergence_action_(Teuchos::getIntegralValue<PoroPressureBased::DivergenceAction>(
          globaltimeparams, "divergence_action")),
      artery_coupling_(globaltimeparams.get<bool>("artery_coupling_active"))
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::init(
    const Teuchos::ParameterList& global_time_params,
    const Teuchos::ParameterList& porofluid_elast_scatra_params,
    const Teuchos::ParameterList& porofluid_elast_params,
    const Teuchos::ParameterList& structure_params, const Teuchos::ParameterList& porofluid_params,
    const Teuchos::ParameterList& scatra_params, const std::string& structure_disname,
    const std::string& porofluid_disname, const std::string& scatra_disname, bool isale,
    int nds_disp, int nds_vel, int nds_solidpressure, int nds_porofluid_scatra,
    const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  // save the dofset number of the scatra on the fluid dis
  nds_porofluid_scatra_ = nds_porofluid_scatra;

  // access the global problem
  Global::Problem* problem = Global::Problem::instance();

  // Create the two uncoupled subproblems.

  // -------------------------------------------------------------------
  // algorithm construction depending on coupling scheme
  // -------------------------------------------------------------------
  // first of all check for possible couplings
  auto solution_scheme_porofluid_elast = Teuchos::getIntegralValue<SolutionSchemePorofluidElast>(
      porofluid_elast_params, "coupling_scheme");
  auto solution_scheme_porofluid_elast_scatra =
      Teuchos::getIntegralValue<SolutionSchemePorofluidElastScatra>(
          porofluid_elast_scatra_params, "coupling_scheme");

  // partitioned -- monolithic not possible --> error
  if (solution_scheme_porofluid_elast != SolutionSchemePorofluidElast::twoway_monolithic &&
      solution_scheme_porofluid_elast_scatra ==
          SolutionSchemePorofluidElastScatra::twoway_monolithic)
  {
    FOUR_C_THROW(
        "Your requested coupling is not available: possible couplings are:\n"
        "(STRUCTURE <--> FLUID) <--> SCATRA: partitioned -- partitioned_nested\n"
        "                                    monolithic  -- partitioned_nested\n"
        "                                    monolithic  -- monolithic\n"
        "YOUR CHOICE                       : partitioned -- monolithic");
  }

  // monolithic -- partitioned sequential not possible
  if (solution_scheme_porofluid_elast == SolutionSchemePorofluidElast::twoway_monolithic &&
      solution_scheme_porofluid_elast_scatra ==
          SolutionSchemePorofluidElastScatra::twoway_partitioned_sequential)
  {
    FOUR_C_THROW(
        "Your requested coupling is not available: possible couplings are:\n"
        "(STRUCTURE <--> FLUID) <--> SCATRA: partitioned -- partitioned_nested\n"
        "                                    monolithic  -- partitioned_nested\n"
        "                                    monolithic  -- monolithic\n"
        "YOUR CHOICE                       : monolithic  -- partitioned_sequential");
  }

  flux_reconstruction_active_ = porofluid_params.sublist("flux_reconstruction").get<bool>("active");

  if (solution_scheme_porofluid_elast_scatra ==
          SolutionSchemePorofluidElastScatra::twoway_monolithic &&
      flux_reconstruction_active_)
  {
    FOUR_C_THROW(
        "Monolithic porofluid-elasticity-scatra coupling does not work with L2-projection!\n"
        "Set FLUX_PROJ_METHOD to none if you want to use monolithic coupling or use partitioned "
        "approach instead.");
  }

  porofluid_elast_algo_ = PoroPressureBased::create_algorithm_porofluid_elast(
      solution_scheme_porofluid_elast, global_time_params, get_comm());

  // initialize
  porofluid_elast_algo_->init(global_time_params, porofluid_elast_params, structure_params,
      porofluid_params, structure_disname, porofluid_disname, isale, nds_disp, nds_vel,
      nds_solidpressure, nds_porofluid_scatra, nearby_ele_pairs);

  // get the solver number used for ScalarTransport solver
  const int linsolvernumber = scatra_params.get<int>("LINEAR_SOLVER");

  // Translate updated porofluid input format to old scatra format
  Teuchos::ParameterList scatra_global_time_params;
  scatra_global_time_params.set<double>(
      "TIMESTEP", global_time_params.sublist("time_integration").get<double>("time_step_size"));
  scatra_global_time_params.set<double>(
      "MAXTIME", global_time_params.get<double>("total_simulation_time"));
  scatra_global_time_params.set<int>(
      "NUMSTEP", global_time_params.sublist("time_integration").get<int>("number_of_time_steps"));
  scatra_global_time_params.set<int>(
      "RESTARTEVERY", global_time_params.sublist("output").get<int>("restart_data_every"));
  scatra_global_time_params.set<int>(
      "RESULTSEVERY", global_time_params.sublist("output").get<int>("result_data_every"));

  // scatra problem
  scatra_algo_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(scatra_global_time_params,
      scatra_params, problem->solver_params(linsolvernumber), scatra_disname, true);

  // initialize the base algo.
  // scatra time integrator is constructed and initialized inside.
  scatra_algo_->init();
  scatra_algo_->scatra_field()->set_number_of_dof_set_displacement(1);
  scatra_algo_->scatra_field()->set_number_of_dof_set_velocity(1);
  scatra_algo_->scatra_field()->set_number_of_dof_set_pressure(2);

  // do we perform coupling with 1D artery
  if (artery_coupling_)
  {
    // get mesh tying strategy
    scatra_meshtying_strategy_ = std::dynamic_pointer_cast<ScaTra::MeshtyingStrategyArtery>(
        scatra_algo_->scatra_field()->strategy());
    if (scatra_meshtying_strategy_ == nullptr) FOUR_C_THROW("cast to Meshtying strategy failed!");

    scatra_meshtying_strategy_->set_artery_time_integrator(
        porofluid_elast_algo()->porofluid_algo()->art_net_tim_int());
    scatra_meshtying_strategy_->set_nearby_ele_pairs(nearby_ele_pairs);
  }

  // only now we must call setup() on the scatra time integrator.
  // all objects relying on the parallel distribution are
  // created and pointers are set.
  // calls setup() on the scatra time integrator inside.
  scatra_algo_->scatra_field()->setup();

  // do we perform coupling with 1D artery
  if (artery_coupling_)
  {
    // this check can only be performed after calling setup
    scatra_meshtying_strategy_->check_initial_fields();
  }

  std::vector<int> mydirichdofs;
  add_dirichmaps_volfrac_spec_ = std::make_shared<Core::LinAlg::Map>(
      -1, 0, mydirichdofs.data(), 0, scatra_algo()->scatra_field()->discretization()->get_comm());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::post_init()
{
  // call the post_setup routine of the underlying porofluid-elasticity algorithm
  porofluid_elast_algo_->post_init();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::read_restart(int restart)
{
  if (restart)
  {
    // read restart data for structure and porofluid field (will set time and step internally)
    porofluid_elast_algo_->read_restart(restart);

    // read restart data for scatra field (will set time and step internally)
    scatra_algo_->scatra_field()->read_restart(restart);
    if (artery_coupling_) scatra_meshtying_strategy_->art_scatra_field()->read_restart(restart);

    // reset time and step for the global algorithm
    set_time_step(scatra_algo_->scatra_field()->time(), restart);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::time_loop()
{
  prepare_time_loop();

  while (not_finished())
  {
    prepare_time_step();

    // reset timer
    timer_timestep_.reset();
    // *********** time measurement ***********
    double dtcpu = timer_timestep_.wallTime();
    // *********** time measurement ***********
    time_step();
    // *********** time measurement ***********
    double mydttimestep = timer_timestep_.wallTime() - dtcpu;
    dt_timestep_ = Core::Communication::max_all(mydttimestep, get_comm());
    // *********** time measurement ***********

    update_and_output();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::prepare_time_step(bool printheader)
{
  // the global control routine has its own time_ and step_ variables, as well as the single fields
  // keep them in sync!
  increment_time_and_step();

  if (printheader) print_header();

  set_porofluid_elast_solution();
  scatra_algo_->scatra_field()->prepare_time_step();
  if (artery_coupling_) scatra_meshtying_strategy_->art_scatra_field()->prepare_time_step();
  // set structure-based scalar transport values
  set_scatra_solution();

  porofluid_elast_algo_->prepare_time_step();
  set_porofluid_elast_solution();
  apply_additional_dbc_for_vol_frac_species();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::prepare_time_loop()
{
  // set structure-based scalar transport values
  set_scatra_solution();
  porofluid_elast_algo_->prepare_time_loop();
  // initial output for scatra field
  set_porofluid_elast_solution();
  if (scatra_algo_->scatra_field()->has_external_force())
    scatra_algo_->scatra_field()->set_external_force();
  scatra_algo_->scatra_field()->check_and_write_output_and_restart();
  if (artery_coupling_)
    scatra_meshtying_strategy_->art_scatra_field()->check_and_write_output_and_restart();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::update_and_output()
{
  // set scatra on fluid (necessary for possible domain integrals)
  set_scatra_solution();
  porofluid_elast_algo_->update_and_output();

  // scatra field
  scatra_algo_->scatra_field()->update();
  scatra_algo_->scatra_field()->evaluate_error_compared_to_analytical_sol();
  scatra_algo_->scatra_field()->check_and_write_output_and_restart();
  // artery scatra field
  if (artery_coupling_)
  {
    scatra_meshtying_strategy_->art_scatra_field()->update();
    scatra_meshtying_strategy_->art_scatra_field()->evaluate_error_compared_to_analytical_sol();
    scatra_meshtying_strategy_->art_scatra_field()->check_and_write_output_and_restart();
  }
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::cout << "Finished PorofluidElastScatra step " << std::setw(5) << std::setprecision(4)
              << std::scientific << step() << "/" << std::setw(5) << std::setprecision(4)
              << std::scientific << n_step() << ": dtstep = " << dt_timestep_ << '\n';
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::create_field_test()
{
  Global::Problem* problem = Global::Problem::instance();

  porofluid_elast_algo_->create_field_test();
  problem->add_field_test(scatra_algo_->create_scatra_field_test());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::set_porofluid_elast_solution()
{
  // safety check
  std::shared_ptr<ScaTra::ScaTraTimIntPoroMulti> scatra_field =
      std::dynamic_pointer_cast<ScaTra::ScaTraTimIntPoroMulti>(scatra_algo_->scatra_field());
  if (scatra_field == nullptr) FOUR_C_THROW("cast to ScaTraTimIntPoroMulti failed!");

  // set displacements
  scatra_field->apply_mesh_movement(*porofluid_elast_algo_->structure_dispnp());

  // set the porofluid solution
  scatra_field->set_solution_field_of_multi_fluid(porofluid_elast_algo_->relaxed_fluid_phinp(),
      porofluid_elast_algo_->porofluid_algo()->phin());

  // additionally, set nodal flux if L2-projection is desired
  if (flux_reconstruction_active_)
    scatra_field->set_l2_flux_of_multi_fluid(porofluid_elast_algo_->fluid_flux());

  if (artery_coupling_)
  {
    scatra_meshtying_strategy_->set_artery_pressure();
    scatra_meshtying_strategy_->apply_mesh_movement();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::
    apply_additional_dbc_for_vol_frac_species()
{
  // remove the old one
  scatra_algo()->scatra_field()->remove_dirich_cond(add_dirichmaps_volfrac_spec_);

  std::vector<int> dirichlet_dofs;

  // get map and valid dof-vector
  const Core::LinAlg::Map* ele_col_map =
      scatra_algo()->scatra_field()->discretization()->element_col_map();
  std::shared_ptr<const Core::LinAlg::Vector<double>> valid_volfracspec_dofs =
      porofluid_elast_algo()->porofluid_algo()->valid_vol_frac_spec_dofs();

  // we identify the volume fraction species dofs which do not have a physical meaning and set a
  // DBC on them
  for (int iele = 0; iele < ele_col_map->num_my_elements(); ++iele)
  {
    // dynamic_cast necessary because virtual inheritance needs runtime information
    Discret::Elements::Transport* current_ele = dynamic_cast<Discret::Elements::Transport*>(
        scatra_algo()->scatra_field()->discretization()->g_element(ele_col_map->gid(iele)));

    const Core::Mat::Material& material2 = *(current_ele->material(2));

    // check the material
    if (material2.material_type() != Core::Materials::m_fluidporo_multiphase and
        material2.material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
      FOUR_C_THROW("only poro multiphase and poro multiphase reactions material valid");

    // cast fluid material
    const auto& porofluid_mat = static_cast<const Mat::FluidPoroMultiPhase&>(material2);

    const int num_fluid_phases = porofluid_mat.num_fluid_phases();
    const int num_volfrac = porofluid_mat.num_vol_frac();
    const int num_fluid_materials = porofluid_mat.num_mat();


    // this is only necessary if we have volume fractions present
    // TODO: this works only if we have the same number of phases in every element
    if (num_fluid_materials == num_fluid_phases) return;

    const Core::Mat::Material& material = *(current_ele->material());

    // cast scatra material
    const auto& scatra_mat = static_cast<const Mat::MatList&>(material);

    if (scatra_mat.material_type() != Core::Materials::m_matlist &&
        scatra_mat.material_type() != Core::Materials::m_matlist_reactions)
      FOUR_C_THROW("wrong type of ScaTra-Material");

    const int num_scatra_materials = scatra_mat.num_mat();

    Core::Nodes::Node** nodes = current_ele->nodes();

    for (int inode = 0; inode < (current_ele->num_node()); inode++)
    {
      if (nodes[inode]->owner() == Core::Communication::my_mpi_rank(
                                       scatra_algo()->scatra_field()->discretization()->get_comm()))
      {
        std::vector<int> scatra_dofs =
            scatra_algo()->scatra_field()->discretization()->dof(0, nodes[inode]);
        std::vector<int> porofluid_dofs =
            porofluid_elast_algo()->porofluid_algo()->discretization()->dof(0, nodes[inode]);

        for (int idof = 0; idof < num_scatra_materials; ++idof)
        {
          int mat_id = scatra_mat.mat_id(idof);
          std::shared_ptr<Core::Mat::Material> single_material = scatra_mat.material_by_id(mat_id);
          if (single_material->material_type() ==
                  Core::Materials::m_scatra_in_fluid_porofluid_pressure_based ||
              single_material->material_type() ==
                  Core::Materials::m_scatra_in_solid_porofluid_pressure_based ||
              single_material->material_type() ==
                  Core::Materials::m_scatra_as_temperature_porofluid_pressure_based)
          {
            // do nothing
          }
          else if (single_material->material_type() ==
                   Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based)
          {
            if (num_fluid_materials == num_fluid_phases + 2 * num_volfrac)
            {
              const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& scatra_volfrac_material =
                  std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(single_material);

              const int scalar_to_phase_id = scatra_volfrac_material->phase_id();
              // if not already in original dirich map     &&   if it is not a valid volume fraction
              // species dof identified with < 1
              if (scatra_algo()->scatra_field()->dirich_maps()->cond_map()->lid(
                      scatra_dofs[idof]) == -1 &&
                  (int)valid_volfracspec_dofs->local_values_as_span()[porofluid_elast_algo()
                          ->porofluid_algo()
                          ->discretization()
                          ->dof_row_map()
                          ->lid(porofluid_dofs[scalar_to_phase_id + num_volfrac])] < 1)
              {
                dirichlet_dofs.push_back(scatra_dofs[idof]);
                scatra_algo()->scatra_field()->phinp()->replace_global_value(
                    scatra_dofs[idof], 0.0);
              }
            }
            else if (num_fluid_materials == num_fluid_phases + num_volfrac)
            {
              // do nothing in volfrac with closing relation blood lung, all species are valid dofs
            }
            else
            {
              FOUR_C_THROW("Internal error!");
            }
          }
          else
            FOUR_C_THROW("only MAT_scatra_multiporo_(fluid,volfrac,solid,temperature) valid here");
        }
      }
    }
  }

  // build map
  int number_dirichlet_values = dirichlet_dofs.size();
  add_dirichmaps_volfrac_spec_ = std::make_shared<Core::LinAlg::Map>(-1, number_dirichlet_values,
      dirichlet_dofs.data(), 0, scatra_algo()->scatra_field()->discretization()->get_comm());

  // add the condition
  scatra_algo()->scatra_field()->add_dirich_cond(add_dirichmaps_volfrac_spec_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::set_scatra_solution()
{
  porofluid_elast_algo_->set_scatra_solution(
      nds_porofluid_scatra_, scatra_algo_->scatra_field()->phinp());
  if (artery_coupling_)
    porofluid_elast_algo_->porofluid_algo()->art_net_tim_int()->discretization()->set_state(
        2, "one_d_artery_phinp", *scatra_meshtying_strategy_->art_scatra_field()->phinp());
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastScatraBaseAlgorithm::scatra_dof_row_map() const
{
  return scatra_algo_->scatra_field()->dof_row_map();
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraBaseAlgorithm::handle_divergence() const
{
  switch (divergence_action_)
  {
    case DivergenceAction::continue_anyway:
    {
      // warn if itemax is reached without convergence, but proceed to the next timestep
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "+---------------------------------------------------------------+" << '\n';
        std::cout << "|            >>>>>> continuing to next time step!               |" << '\n';
        std::cout << "+---------------------------------------------------------------+" << '\n'
                  << '\n';
      }
      break;
    }
    case DivergenceAction::stop:
    {
      FOUR_C_THROW("PorofluidElastScatra nonlinear solver not converged in ITEMAX steps!");
    }
    default:
      FOUR_C_THROW("Unknown divergence action!");
  }
}

FOUR_C_NAMESPACE_CLOSE
