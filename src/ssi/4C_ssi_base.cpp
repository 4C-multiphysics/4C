// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_base.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_coupling_volmortar.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_global_data_read.hpp"
#include "4C_io_control.hpp"
#include "4C_io_input_file.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_scatra_utils.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_ssi_clonestrategy.hpp"
#include "4C_ssi_coupling.hpp"
#include "4C_ssi_input.hpp"
#include "4C_ssi_resulttest.hpp"
#include "4C_ssi_str_model_evaluator_partitioned.hpp"
#include "4C_ssi_utils.hpp"
#include "4C_structure_new_model_evaluator_contact.hpp"
#include "4C_utils_function_of_time.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSI::SSIBase::SSIBase(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams)
    : AlgorithmBase(comm, globaltimeparams),
      diff_time_step_size_(globaltimeparams.get<bool>("DIFFTIMESTEPSIZE")),
      fieldcoupling_(Teuchos::getIntegralValue<SSI::FieldCoupling>(
          Global::Problem::instance()->ssi_control_params(), "FIELDCOUPLING")),
      is_scatra_manifold_(globaltimeparams.sublist("MANIFOLD").get<bool>("ADD_MANIFOLD")),
      is_manifold_meshtying_(globaltimeparams.sublist("MANIFOLD").get<bool>("MESHTYING_MANIFOLD")),
      is_s2i_kinetic_with_pseudo_contact_(
          check_s2i_kinetics_condition_for_pseudo_contact("structure")),
      macro_scale_(Global::Problem::instance()->materials()->first_id_by_type(
                       Core::Materials::m_scatra_multiscale) != -1 or
                   Global::Problem::instance()->materials()->first_id_by_type(
                       Core::Materials::m_newman_multiscale) != -1),
      ssi_interface_contact_(
          Global::Problem::instance()->get_dis("structure")->has_condition("SSIInterfaceContact")),
      ssi_interface_meshtying_(Global::Problem::instance()
              ->get_dis("structure")
              ->has_condition("ssi_interface_meshtying")),
      temperature_funct_num_(
          Global::Problem::instance()->elch_control_params().get<int>("TEMPERATURE_FROM_FUNCT"))
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SSIBase::init(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, const bool is_ale)
{
  // reset the setup flag
  set_is_setup(false);

  // do discretization specific setup (e.g. clone discr. scatra from structure)
  init_discretizations(
      comm, struct_disname, scatra_disname, globaltimeparams.get<bool>("REDISTRIBUTE_SOLID"));

  init_time_integrators(
      globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, is_ale);

  const RedistributionType redistribution_type = init_field_coupling(struct_disname);

  if (redistribution_type != RedistributionType::none) redistribute(redistribution_type);

  check_ssi_flags();

  check_ssi_interface_conditions(struct_disname);

  // set isinit_ flag true
  set_is_init(true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SSIBase::setup()
{
  // check initialization
  check_is_init();

  // set up helper class for field coupling
  ssicoupling_->setup();

  // in case of an ssi  multi scale formulation we need to set the displacement here
  auto dummy_vec = Core::LinAlg::Vector<double>(
      *Global::Problem::instance()->get_dis("structure")->dof_row_map(), true);
  ssicoupling_->set_mesh_disp(scatra_base_algorithm(), dummy_vec);

  // set up scalar transport field
  scatra_field()->setup();
  if (is_scatra_manifold()) scatra_manifold()->setup();

  // only necessary if adapter base has not already been set up outside
  if (not struct_adapterbase_ptr_->is_setup())
  {
    // set up structural model evaluator
    setup_model_evaluator();

    // pass initial scalar field to structural discretization to correctly compute initial
    // accelerations
    if (Teuchos::getIntegralValue<SSI::SolutionSchemeOverFields>(
            Global::Problem::instance()->ssi_control_params(), "COUPALGO") !=
        SSI::SolutionSchemeOverFields::ssi_OneWay_SolidToScatra)
      ssicoupling_->set_scalar_field(
          *Global::Problem::instance()->get_dis("structure"), scatra_field()->phinp(), 1);

    if (macro_scale_)
    {
      scatra_field()->calc_mean_micro_concentration();
      ssicoupling_->set_scalar_field_micro(
          *Global::Problem::instance()->get_dis("structure"), scatra_field()->phinp_micro(), 2);
    }

    //   temperature is non primary variable. Only set, if function for temperature is given
    if (temperature_funct_num_ != -1)
    {
      temperature_vector_ = std::make_shared<Core::LinAlg::Vector<double>>(
          *Global::Problem::instance()->get_dis("structure")->dof_row_map(2), true);

      temperature_vector_->put_scalar(Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(temperature_funct_num_)
              .evaluate(time()));

      ssicoupling_->set_temperature_field(
          *Global::Problem::instance()->get_dis("structure"), temperature_vector_);
    }

    // set up structural base algorithm
    struct_adapterbase_ptr_->setup();

    // get wrapper and cast it to specific type
    // do not do so, in case the wrapper has already been set from outside
    if (structure_ == nullptr)
      structure_ = std::dynamic_pointer_cast<Adapter::SSIStructureWrapper>(
          struct_adapterbase_ptr_->structure_field());

    if (structure_ == nullptr)
    {
      FOUR_C_THROW(
          "No valid pointer to Adapter::SSIStructureWrapper !\n"
          "Either cast failed, or no valid wrapper was set using set_structure_wrapper(...) !");
    }
  }

  if (is_s2i_kinetics_with_pseudo_contact())
  {
    const auto dummy_stress_state = std::make_shared<Core::LinAlg::Vector<double>>(
        *structure_field()->discretization()->dof_row_map(2), true);
    ssicoupling_->set_mechanical_stress_state(*scatra_field()->discretization(), dummy_stress_state,
        scatra_field()->nds_two_tensor_quantity());
  }

  // check maps from scalar transport and structure discretizations
  if (scatra_field()->dof_row_map()->num_global_elements() == 0)
    FOUR_C_THROW("Scalar transport discretization does not have any degrees of freedom!");
  if (structure_->dof_row_map()->num_global_elements() == 0)
    FOUR_C_THROW("Structure discretization does not have any degrees of freedom!");

  // set up materials
  ssicoupling_->assign_material_pointers(
      structure_->discretization(), scatra_field()->discretization());

  // set up scatra-scatra interface coupling
  if (ssi_interface_meshtying())
  {
    ssi_structure_meshtying_ = std::make_shared<SSI::Utils::SSIMeshTying>(
        "ssi_interface_meshtying", *structure_->discretization(), true, true);

    // extract meshtying strategy for scatra-scatra interface coupling on scatra discretization
    meshtying_strategy_s2i_ =
        std::dynamic_pointer_cast<const ScaTra::MeshtyingStrategyS2I>(scatra_field()->strategy());

    // safety checks
    if (meshtying_strategy_s2i_ == nullptr)
      FOUR_C_THROW("Invalid scatra-scatra interface coupling strategy!");
  }

  // construct vector of zeroes
  zeros_structure_ = std::make_shared<Core::LinAlg::Vector<double>>(*structure_->dof_row_map());

  // set flag
  set_is_setup(true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SSIBase::post_setup() const
{
  check_is_setup();

  // communicate scatra states to structure if necessary
  if (Teuchos::getIntegralValue<SSI::SolutionSchemeOverFields>(
          Global::Problem::instance()->ssi_control_params(), "COUPALGO") !=
      SSI::SolutionSchemeOverFields::ssi_OneWay_SolidToScatra)
  {
    set_scatra_solution(scatra_field()->phinp());
  }

  structure_->post_setup();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SSIBase::init_discretizations(MPI_Comm comm, const std::string& struct_disname,
    const std::string& scatra_disname, const bool redistribute_struct_dis)
{
  Global::Problem* problem = Global::Problem::instance();

  auto structdis = problem->get_dis(struct_disname);
  auto scatradis = problem->get_dis(scatra_disname);

  if (redistribute_struct_dis)
  {
    Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
        binning_params);
    Core::Rebalance::rebalance_discretizations_by_binning(binning_params,
        Global::Problem::instance()->output_control_file(), {structdis}, nullptr, nullptr, false);
  }

  if (scatradis->num_global_nodes() == 0)
  {
    if (fieldcoupling_ != SSI::FieldCoupling::volume_match and
        fieldcoupling_ != SSI::FieldCoupling::volumeboundary_match)
    {
      FOUR_C_THROW(
          "If 'FIELDCOUPLING' is NOT 'volume_matching' or 'volumeboundary_matching' in the SSI "
          "CONTROL section cloning of the scatra discretization from the structure "
          "discretization "
          "is not supported!");
    }

    // fill scatra discretization by cloning structure discretization
    Core::FE::clone_discretization<ScatraStructureCloneStrategy>(
        *structdis, *scatradis, Global::Problem::instance()->cloning_material_map());
    scatradis->fill_complete();

    // create discretization for scatra manifold based on SSISurfaceManifold condition
    if (is_scatra_manifold())
    {
      auto scatra_manifold_dis = problem->get_dis("scatra_manifold");
      Core::FE::clone_discretization_from_condition<SSI::ScatraStructureCloneStrategyManifold>(
          *structdis, *scatra_manifold_dis, "SSISurfaceManifold",
          Global::Problem::instance()->cloning_material_map());

      // clone conditions. Needed this way, as many conditions are cloned from SSISurfaceManifold.
      std::vector<std::map<std::string, std::string>> conditions_to_copy = {
          {std::make_pair("SSISurfaceManifold", "SSISurfaceManifold")},
          {std::make_pair("ScaTraManifoldInitfield", "Initfield")},
          {std::make_pair("ManifoldDirichlet", "Dirichlet")}};

      //! in case of no mesh tying between manifolds: partition manifold domains
      if (!is_manifold_meshtying_)
      {
        std::map<std::string, std::string> temp_map = {
            std::make_pair("SSISurfaceManifold", "ScatraPartitioning")};
        conditions_to_copy.emplace_back(temp_map);
      }

      const auto output_scalar_type = Teuchos::getIntegralValue<Inpar::ScaTra::OutputScalarType>(
          problem->scalar_transport_dynamic_params(), "OUTPUTSCALARS");
      if (output_scalar_type == Inpar::ScaTra::outputscalars_condition or
          output_scalar_type == Inpar::ScaTra::outputscalars_entiredomain_condition)
      {
        std::map<std::string, std::string> tempmap = {
            std::make_pair("SSISurfaceManifold", "TotalAndMeanScalar")};

        conditions_to_copy.emplace_back(tempmap);
      }

      Core::FE::DiscretizationCreatorBase creator;
      for (const auto& condition_to_copy : conditions_to_copy)
        creator.copy_conditions(*structdis, *scatra_manifold_dis, condition_to_copy);

      scatra_manifold_dis->fill_complete();

      //! in case of mesh tying between manifolds: unite manifold domains -> create new
      //! ScatraPartitioning condition
      if (is_manifold_meshtying_)
      {
        // create vector of all node GIDs (all procs) of manifold dis
        int num_my_nodes = scatra_manifold_dis->node_row_map()->num_my_elements();
        std::vector<int> my_node_ids(num_my_nodes);
        for (int lid = 0; lid < num_my_nodes; ++lid)
          my_node_ids[lid] = scatra_manifold_dis->node_row_map()->gid(lid);

        int max_num_nodes = 0;
        max_num_nodes = Core::Communication::max_all(num_my_nodes, get_comm());

        // resize vector and fill with place holders (-1)
        my_node_ids.resize(max_num_nodes, -1);

        std::vector<int> glob_node_ids(
            max_num_nodes * Core::Communication::num_mpi_ranks(get_comm()), -1);
        Core::Communication::gather_all(my_node_ids.data(), glob_node_ids.data(),
            static_cast<int>(my_node_ids.size()), get_comm());

        // remove place holders (-1)
        glob_node_ids.erase(
            std::remove(glob_node_ids.begin(), glob_node_ids.end(), -1), glob_node_ids.end());

        // create new condition
        const int num_conditions =
            static_cast<int>(scatra_manifold_dis->get_all_conditions().size());
        auto cond = std::make_shared<Core::Conditions::Condition>(num_conditions + 1,
            Core::Conditions::ScatraPartitioning, true, Core::Conditions::geometry_type_surface,
            Core::Conditions::EntityType::legacy_id);
        cond->parameters().add("ConditionID", 0);
        cond->set_nodes(glob_node_ids);

        scatra_manifold_dis->set_condition("ScatraPartitioning", cond);

        scatra_manifold_dis->fill_complete();
      }
    }
  }
  else
  {
    if (fieldcoupling_ == SSI::FieldCoupling::volume_match)
    {
      FOUR_C_THROW(
          "Reading a TRANSPORT discretization from the input file for the input parameter "
          "'FIELDCOUPLING volume_matching' in the SSI CONTROL section is not supported! As this "
          "coupling relies on matching node (and sometimes element) IDs, the ScaTra "
          "discretization "
          "is cloned from the structure discretization. Delete the ScaTra discretization from "
          "your "
          "input file.");
    }

    // copy conditions
    // this is actually only needed for copying TRANSPORT DIRICHLET/NEUMANN CONDITIONS
    // as standard DIRICHLET/NEUMANN CONDITIONS
    SSI::ScatraStructureCloneStrategy clonestrategy;
    const auto conditions_to_copy = clonestrategy.conditions_to_copy();
    Core::FE::DiscretizationCreatorBase creator;
    creator.copy_conditions(*scatradis, *scatradis, conditions_to_copy);

    // safety check, since it is not reasonable to have SOLIDSCATRA or SOLIDPOROP1 Elements with a
    // ScaTra::ImplType != 'impltype_undefined' if they are not cloned! Therefore loop over all
    // structure elements and check the impltype
    for (int i = 0; i < structdis->num_my_col_elements(); ++i)
    {
      if (clonestrategy.get_impl_type(structdis->l_col_element(i)) !=
          Inpar::ScaTra::impltype_undefined)
      {
        FOUR_C_THROW(
            "A TRANSPORT discretization is read from the input file, which is fine since the "
            "scatra "
            "discretization is not cloned from the structure discretization. But in the STRUCTURE "
            "ELEMENTS section of the input file an ImplType that is NOT 'Undefined' is prescribed "
            "which does not make sense if you don't want to clone the structure discretization. "
            "Change the ImplType to 'Undefined' or decide to clone the scatra discretization "
            "from "
            "the structure discretization.");
      }
    }
  }
  // read in the micro field, has to be done after cloning of the scatra discretization
  auto input_file_name = problem->output_control_file()->input_file_name();
  read_micro_fields(*problem, std::filesystem::path(input_file_name).parent_path());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
SSI::RedistributionType SSI::SSIBase::init_field_coupling(const std::string& struct_disname)
{
  // initialize return variable
  auto redistribution_required{RedistributionType::none};

  // safety check
  {
    auto scatra_integrator = scatra_base_algorithm()->scatra_field();
    // check for ssi coupling condition
    std::vector<const Core::Conditions::Condition*> ssicoupling;
    scatra_integrator->discretization()->get_condition("SSICoupling", ssicoupling);
    const bool havessicoupling = (ssicoupling.size() > 0);

    if (havessicoupling and (fieldcoupling_ != SSI::FieldCoupling::boundary_nonmatch and
                                fieldcoupling_ != SSI::FieldCoupling::volumeboundary_match))
    {
      FOUR_C_THROW(
          "SSICoupling condition only valid in combination with FIELDCOUPLING set to "
          "'boundary_nonmatching' or 'volumeboundary_matching' in SSI DYNAMIC section. ");
    }

    if (fieldcoupling_ == SSI::FieldCoupling::volume_nonmatch)
    {
      const Teuchos::ParameterList& volmortarparams =
          Global::Problem::instance()->volmortar_params();
      if (Teuchos::getIntegralValue<Coupling::VolMortar::CouplingType>(
              volmortarparams, "COUPLINGTYPE") != Coupling::VolMortar::couplingtype_coninter)
      {
        FOUR_C_THROW(
            "Volmortar coupling only tested for consistent interpolation, i.e. 'COUPLINGTYPE "
            "coninter' in VOLMORTAR COUPLING section. Try other couplings at own risk.");
      }
    }
    if (is_scatra_manifold() and fieldcoupling_ != SSI::FieldCoupling::volumeboundary_match)
      FOUR_C_THROW("Solving manifolds only in combination with matching volumes and boundaries");
  }

  // build SSI coupling class
  switch (fieldcoupling_)
  {
    case SSI::FieldCoupling::volume_match:
      ssicoupling_ = std::make_shared<SSICouplingMatchingVolume>();
      break;
    case SSI::FieldCoupling::volume_nonmatch:
      ssicoupling_ = std::make_shared<SSICouplingNonMatchingVolume>();
      // redistribution is still performed inside
      redistribution_required = RedistributionType::binning;
      break;
    case SSI::FieldCoupling::boundary_nonmatch:
      ssicoupling_ = std::make_shared<SSICouplingNonMatchingBoundary>();
      break;
    case SSI::FieldCoupling::volumeboundary_match:
      ssicoupling_ = std::make_shared<SSICouplingMatchingVolumeAndBoundary>();
      redistribution_required = RedistributionType::match;
      break;
    default:
      FOUR_C_THROW("unknown type of field coupling for SSI!");
  }

  // initialize coupling objects including dof sets
  const Global::Problem* problem = Global::Problem::instance();
  ssicoupling_->init(
      problem->n_dim(), problem->get_dis(struct_disname), Core::Utils::shared_ptr_from_ref(*this));

  return redistribution_required;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SSIBase::read_restart(const int restart)
{
  if (restart)
  {
    structure_->read_restart(restart);

    const Teuchos::ParameterList& ssidyn = Global::Problem::instance()->ssi_control_params();
    const bool restart_from_structure = ssidyn.get<bool>("RESTART_FROM_STRUCTURE");

    if (not restart_from_structure)  // standard restart
    {
      scatra_field()->read_restart(restart);
      if (is_scatra_manifold()) scatra_manifold()->read_restart(restart);
    }
    else  // restart from structure simulation
    {
      // Since there is no restart output for the scatra field available, we only have to fix the
      // time and step counter
      scatra_field()->set_time_step(structure_->time_old(), restart);
      if (is_scatra_manifold()) scatra_manifold()->set_time_step(structure_->time_old(), restart);
    }

    set_time_step(structure_->time_old(), restart);
  }

  // Material pointers to other field were deleted during read_restart().
  // They need to be reset.
  ssicoupling_->assign_material_pointers(
      structure_->discretization(), scatra_field()->discretization());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::test_results(MPI_Comm comm) const
{
  Global::Problem* problem = Global::Problem::instance();

  problem->add_field_test(structure_->create_field_test());
  problem->add_field_test(scatra_base_algorithm()->create_scatra_field_test());
  if (is_scatra_manifold())
    problem->add_field_test(scatra_manifold_base_algorithm()->create_scatra_field_test());
  problem->add_field_test(std::make_shared<SSIResultTest>(Core::Utils::shared_ptr_from_ref(*this)));
  problem->test_all(comm);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_struct_solution(const Core::LinAlg::Vector<double>& disp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel, const bool set_mechanical_stress)
{
  // safety checks
  check_is_init();
  check_is_setup();

  set_mesh_disp(disp);
  set_velocity_fields(vel);

  if (set_mechanical_stress)
    set_mechanical_stress_state(modelevaluator_ssi_base_->get_mechanical_stress_state_n());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_scatra_solution(
    std::shared_ptr<const Core::LinAlg::Vector<double>> phi) const
{
  // safety checks
  check_is_init();
  check_is_setup();

  ssicoupling_->set_scalar_field(*structure_field()->discretization(), phi, 1);

  // set state for contact evaluation
  if (contact_strategy_nitsche_ != nullptr) set_ssi_contact_states(phi);
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::SSIBase::set_ssi_contact_states(
    std::shared_ptr<const Core::LinAlg::Vector<double>> phi) const
{
  contact_strategy_nitsche_->set_state(Mortar::state_scalar, *phi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_micro_scatra_solution(
    std::shared_ptr<const Core::LinAlg::Vector<double>> phi) const
{
  // safety checks
  check_is_init();
  check_is_setup();

  ssicoupling_->set_scalar_field_micro(*structure_field()->discretization(), phi, 2);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::evaluate_and_set_temperature_field()
{
  // temperature is non primary variable. Only set, if function for temperature is given
  if (temperature_funct_num_ != -1)
  {
    // evaluate temperature at current time and put to scalar
    const double temperature =
        Global::Problem::instance()
            ->function_by_id<Core::Utils::FunctionOfTime>(temperature_funct_num_)
            .evaluate(time());
    temperature_vector_->put_scalar(temperature);

    // set temperature vector to structure discretization
    ssicoupling_->set_temperature_field(*structure_->discretization(), temperature_vector_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_velocity_fields(std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  // safety checks
  check_is_init();
  check_is_setup();

  ssicoupling_->set_velocity_fields(scatra_base_algorithm(), zeros_structure_, vel);
  if (is_scatra_manifold())
    ssicoupling_->set_velocity_fields(scatra_manifold_base_algorithm(), zeros_structure_, vel);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_mechanical_stress_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>> mechanical_stress_state) const
{
  check_is_init();
  check_is_setup();

  ssicoupling_->set_mechanical_stress_state(*scatra_field()->discretization(),
      mechanical_stress_state, scatra_field()->nds_two_tensor_quantity());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_mesh_disp(const Core::LinAlg::Vector<double>& disp)
{
  // safety checks
  check_is_init();
  check_is_setup();

  ssicoupling_->set_mesh_disp(scatra_base_algorithm(), disp);
  if (is_scatra_manifold()) ssicoupling_->set_mesh_disp(scatra_manifold_base_algorithm(), disp);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::check_ssi_flags() const
{
  if (scatra_field()->s2i_kinetics())
  {
    if (!(ssi_interface_contact() or ssi_interface_meshtying()))
    {
      FOUR_C_THROW(
          "You defined an 'S2IKinetics' condition in the input-file. However, neither an "
          "'SSIInterfaceContact' condition nor an 'ssi_interface_meshtying' condition defined. "
          "This "
          "is not reasonable!");
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_dt_from_scatra_to_structure() const
{
  structure_field()->set_dt(scatra_field()->dt());
  structure_field()->set_timen(scatra_field()->time());
  structure_field()->post_update();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_dt_from_scatra_to_manifold() const
{
  scatra_manifold()->set_dt(scatra_field()->dt());
  scatra_manifold()->set_time_step(scatra_field()->time(), scatra_field()->step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::set_dt_from_scatra_to_ssi()
{
  // set values for this SSI algorithm
  set_time_step(scatra_field()->time(), step());
  set_dt(scatra_field()->dt());

  // set values for other fields
  set_dt_from_scatra_to_structure();
  if (is_scatra_manifold()) set_dt_from_scatra_to_manifold();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::redistribute(const RedistributionType redistribution_type) const
{
  Global::Problem* problem = Global::Problem::instance();

  auto structdis = problem->get_dis("structure");
  auto scatradis = problem->get_dis("scatra");
  if (redistribution_type == SSI::RedistributionType::match and !is_scatra_manifold())
  {
    // first we bin the scatra discretization
    std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
    dis.push_back(scatradis);
    Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
        binning_params);

    Core::Rebalance::rebalance_discretizations_by_binning(binning_params,
        Global::Problem::instance()->output_control_file(), dis, nullptr, nullptr, false);

    Core::Rebalance::match_element_distribution_of_matching_conditioned_elements(
        *scatradis, *scatradis, "ScatraHeteroReactionMaster", "ScatraHeteroReactionSlave");

    // now we redistribute the structure dis to match the scatra dis
    Core::Rebalance::match_element_distribution_of_matching_discretizations(*scatradis, *structdis);
  }
  else if (redistribution_type == SSI::RedistributionType::binning)
  {
    // create vector of discr.
    std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
    dis.push_back(structdis);
    dis.push_back(scatradis);

    Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
        binning_params);

    Core::Rebalance::rebalance_discretizations_by_binning(binning_params,
        Global::Problem::instance()->output_control_file(), dis, nullptr, nullptr, false);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<ScaTra::ScaTraTimIntImpl> SSI::SSIBase::scatra_field() const
{
  return scatra_base_algorithm_->scatra_field();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<ScaTra::ScaTraTimIntImpl> SSI::SSIBase::scatra_manifold() const
{
  return scatra_manifold_base_algorithm_->scatra_field();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::init_time_integrators(const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, const bool is_ale)
{
  // get the global problem
  auto* problem = Global::Problem::instance();

  // time parameter handling
  // In case of different time stepping, time params have to be read from single field sections.
  // In case of equal time step size for all fields the time params are controlled solely by the
  // problem section (e.g. ---SSI DYNAMIC or ---CELL DYNAMIC).
  const auto* structtimeparams = &globaltimeparams;
  const auto* scatratimeparams = &globaltimeparams;
  if (diff_time_step_size_)
  {
    structtimeparams = &structparams;
    scatratimeparams = &scatraparams;
  }

  // we do not construct a structure, in case it was built externally and handed into this object
  if (struct_adapterbase_ptr_ == nullptr)
  {
    // access the structural discretization
    auto structdis = problem->get_dis(struct_disname);

    // build structure based on new structural time integration
    if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(
            structparams, "INT_STRATEGY") == Inpar::Solid::IntegrationStrategy::int_standard)
    {
      struct_adapterbase_ptr_ = Adapter::build_structure_algorithm(structparams);

      // initialize structure base algorithm
      struct_adapterbase_ptr_->init(
          *structtimeparams, const_cast<Teuchos::ParameterList&>(structparams), structdis);
    }
    // build structure based on old structural time integration
    else if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(
                 structparams, "INT_STRATEGY") == Inpar::Solid::IntegrationStrategy::int_old)
    {
      Adapter::StructureBaseAlgorithm structure(
          *structtimeparams, const_cast<Teuchos::ParameterList&>(structparams), structdis);
      structure_ =
          std::dynamic_pointer_cast<Adapter::SSIStructureWrapper>(structure.structure_field());
      if (structure_ == nullptr)
        FOUR_C_THROW("cast from Adapter::Structure to Adapter::SSIStructureWrapper failed");
    }
    else
    {
      FOUR_C_THROW(
          "Unknown time integration requested!\n"
          "Set parameter INT_STRATEGY to Standard in ---STRUCTURAL DYNAMIC section!\n"
          "If you want to use yet unsupported elements or algorithms,\n"
          "set INT_STRATEGY to Old in ---STRUCTURAL DYNAMIC section!");
    }
  }

  // create and initialize scatra base algorithm.
  // scatra time integrator constructed and initialized inside.
  // mesh is written inside. cloning must happen before!
  scatra_base_algorithm_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(*scatratimeparams,
      SSI::Utils::modify_scatra_params(scatraparams),
      problem->solver_params(scatraparams.get<int>("LINEAR_SOLVER")), scatra_disname, is_ale);

  scatra_base_algorithm()->init();

  // create and initialize scatra base algorithm for manifolds
  if (is_scatra_manifold())
  {
    scatra_manifold_base_algorithm_ =
        std::make_shared<Adapter::ScaTraBaseAlgorithm>(*scatratimeparams,
            SSI::Utils::clone_scatra_manifold_params(
                scatraparams, globaltimeparams.sublist("MANIFOLD")),
            problem->solver_params(globaltimeparams.sublist("MANIFOLD").get<int>("LINEAR_SOLVER")),
            "scatra_manifold", is_ale);

    scatra_manifold_base_algorithm()->init();
  }

  // do checks if adaptive time stepping is activated
  if (globaltimeparams.get<bool>("ADAPTIVE_TIMESTEPPING"))
    check_adaptive_time_stepping(scatraparams, structparams);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool SSI::SSIBase::do_calculate_initial_potential_field() const
{
  const auto ssi_params = Global::Problem::instance()->ssi_control_params();
  const bool init_pot_calc = ssi_params.sublist("ELCH").get<bool>("INITPOTCALC");

  return init_pot_calc and is_elch_scatra_time_int_type();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool SSI::SSIBase::is_elch_scatra_time_int_type() const
{
  const auto ssi_params = Global::Problem::instance()->ssi_control_params();
  const auto scatra_type =
      Teuchos::getIntegralValue<SSI::ScaTraTimIntType>(ssi_params, "SCATRATIMINTTYPE");

  return scatra_type == SSI::ScaTraTimIntType::elch;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool SSI::SSIBase::is_restart() const
{
  // get the global problem
  const auto* problem = Global::Problem::instance();

  const int restartstep = problem->restart();

  return (restartstep > 0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::check_adaptive_time_stepping(
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams)
{
  // safety check: adaptive time stepping in one of the sub problems
  if (!scatraparams.get<bool>("ADAPTIVE_TIMESTEPPING"))
  {
    FOUR_C_THROW(
        "Must provide adaptive time stepping algorithm in one of the sub problems. (Currently "
        "just ScaTra)");
  }
  if (Teuchos::getIntegralValue<Inpar::Solid::TimAdaKind>(
          structparams.sublist("TIMEADAPTIVITY"), "KIND") != Inpar::Solid::timada_kind_none)
    FOUR_C_THROW("Adaptive time stepping in SSI currently just from ScaTra");
  if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structparams, "DYNAMICTYPE") ==
      Inpar::Solid::DynamicType::AdamsBashforth2)
    FOUR_C_THROW("Currently, only one step methods are allowed for adaptive time stepping");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool SSI::SSIBase::check_s2i_kinetics_condition_for_pseudo_contact(
    const std::string& struct_disname) const
{
  bool is_s2i_kinetic_with_pseudo_contact = false;

  auto structdis = Global::Problem::instance()->get_dis(struct_disname);
  // get all s2i kinetics conditions
  std::vector<const Core::Conditions::Condition*> s2ikinetics_conditions;
  structdis->get_condition("S2IKinetics", s2ikinetics_conditions);
  // get all ssi contact conditions
  std::vector<const Core::Conditions::Condition*> ssi_contact_conditions;
  structdis->get_condition("SSIInterfaceContact", ssi_contact_conditions);
  for (auto* s2ikinetics_cond : s2ikinetics_conditions)
  {
    if ((s2ikinetics_cond->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
            Inpar::S2I::side_slave) and
        (s2ikinetics_cond->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") !=
            Inpar::S2I::kinetics_nointerfaceflux) and
        s2ikinetics_cond->parameters().get<bool>("IS_PSEUDO_CONTACT"))
    {
      is_s2i_kinetic_with_pseudo_contact = true;
      const int s2i_kinetics_condition_id = s2ikinetics_cond->parameters().get<int>("ConditionID");

      for (auto* contact_condition : ssi_contact_conditions)
      {
        if (contact_condition->parameters().get<int>("ConditionID") == s2i_kinetics_condition_id)
        {
          FOUR_C_THROW(
              "Pseudo contact formulation of s2i kinetics conditions does not make sense in "
              "combination with resolved contact formulation. Set the respective "
              "IS_PSEUDO_CONTACT "
              "flag to 'False'");
        }
      }
    }
  }

  const bool do_output_cauchy_stress =
      Teuchos::getIntegralValue<Inpar::Solid::StressType>(
          Global::Problem::instance()->io_params(), "STRUCT_STRESS") == Inpar::Solid::stress_cauchy;

  if (is_s2i_kinetic_with_pseudo_contact and !do_output_cauchy_stress)
  {
    FOUR_C_THROW(
        "Consideration of pseudo contact with 'S2IKinetics' condition only possible when Cauchy "
        "stress output is written.");
  }

  return is_s2i_kinetic_with_pseudo_contact;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::check_ssi_interface_conditions(const std::string& struct_disname) const
{
  // access the structural discretization
  auto structdis = Global::Problem::instance()->get_dis(struct_disname);

  if (ssi_interface_meshtying())
    ScaTra::ScaTraUtils::check_consistency_with_s2_i_kinetics_condition(
        "ssi_interface_meshtying", structdis);

  // check scatra-structure-interaction contact condition
  if (ssi_interface_contact())
  {
    // get ssi condition to be tested
    std::vector<const Core::Conditions::Condition*> ssiconditions;
    structdis->get_condition("SSIInterfaceContact", ssiconditions);
    SSI::Utils::check_consistency_of_ssi_interface_contact_condition(ssiconditions, *structdis);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::setup_system()
{
  if (ssi_interface_meshtying_)
    ssi_structure_mesh_tying()->check_slave_side_has_dirichlet_conditions(
        structure_field()->get_dbc_map_extractor()->cond_map());
}

/*---------------------------------------------------------------------------------*
 *---------------------------------------------------------------------------------*/
void SSI::SSIBase::setup_model_evaluator()
{
  // register the model evaluator if s2i condition with pseudo contact is available
  if (is_s2i_kinetics_with_pseudo_contact())
  {
    modelevaluator_ssi_base_ = std::make_shared<Solid::ModelEvaluator::BaseSSI>();
    structure_base_algorithm()->register_model_evaluator(
        "Basic Coupling Model", modelevaluator_ssi_base_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::SSIBase::setup_contact_strategy()
{
  // get the contact solution strategy
  auto contact_solution_type = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(
      Global::Problem::instance()->contact_dynamic_params(), "STRATEGY");

  if (contact_solution_type == CONTACT::SolvingStrategy::nitsche)
  {
    if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(
            Global::Problem::instance()->structural_dynamic_params(), "INT_STRATEGY") !=
        Inpar::Solid::int_standard)
    {
      FOUR_C_THROW("ssi contact only with new structural time integration");
    }

    // get the contact model evaluator and store a pointer to the strategy
    auto& model_evaluator_contact = dynamic_cast<Solid::ModelEvaluator::Contact&>(
        structure_field()->model_evaluator(Inpar::Solid::model_contact));
    contact_strategy_nitsche_ = std::dynamic_pointer_cast<CONTACT::NitscheStrategySsi>(
        model_evaluator_contact.strategy_ptr());
  }
  else
  {
    FOUR_C_THROW("Only Nitsche contact implemented for SSI problems at the moment!");
  }
}

FOUR_C_NAMESPACE_CLOSE
