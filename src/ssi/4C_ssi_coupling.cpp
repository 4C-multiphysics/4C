// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_coupling.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_coupling_volmortar_utils.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_dofset_definedmapping_wrapper.hpp"
#include "4C_fem_dofset_gidbased_wrapper.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_ssi_problem_access.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::init(const int ndim,
    std::shared_ptr<Core::FE::Discretization> structdis, std::shared_ptr<SSI::SSIBase> ssi_base)
{
  Global::Problem* problem = SSI::Utils::problem_from_instance();

  set_is_setup(false);

  int scatra_dofset_counter = ssi_base->scatra_field()->get_max_dof_set_number();
  int structure_dofset_counter = 0;

  auto scatra_integrator = ssi_base->scatra_field();
  auto scatradis = scatra_integrator->discretization();
  // build a proxy of the structure discretization for the scatra field
  std::shared_ptr<Core::DOFSets::DofSetInterface> structdofset = structdis->get_dof_set_proxy();
  // build a proxy of the scatra discretization for the structure field
  std::shared_ptr<Core::DOFSets::DofSetInterface> scatradofset = scatradis->get_dof_set_proxy();

  // add proxy dofssets of other fields to discretizations and check if number of dofsets is correct
  if (scatradis->add_dof_set(structdofset) != ++scatra_dofset_counter)
    FOUR_C_THROW("unexpected dof sets in scatra field");
  scatra_integrator->set_number_of_dof_set_displacement(scatra_dofset_counter);
  scatra_integrator->set_number_of_dof_set_velocity(scatra_dofset_counter);
  if (structdis->add_dof_set(scatradofset) != ++structure_dofset_counter)
    FOUR_C_THROW("unexpected dof sets in structure field");

  if (problem->elch_control_params().get<int>("TEMPERATURE_FROM_FUNCT") != -1)
  {
    const int numDofsPerNodeTemp = 1;  // defined by temperature field

    std::shared_ptr<Core::DOFSets::DofSetInterface> dofsettemp =
        std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(numDofsPerNodeTemp, 0, 0, true);
    if (structdis->add_dof_set(dofsettemp) != ++structure_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in structure field");
  }

  if (problem->materials()->first_id_by_type(Core::Materials::m_scatra_multiscale) != -1 or
      problem->materials()->first_id_by_type(Core::Materials::m_newman_multiscale) != -1)
  {
    auto dofsetmicro = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(1, 0, 0, true);
    if (scatradis->add_dof_set(dofsetmicro) != ++scatra_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in scatra field");
    scatra_integrator->set_number_of_dof_set_micro_scale(scatra_dofset_counter);
    if (structdis->add_dof_set(dofsetmicro) != ++structure_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in structure field");
  }

  if (ssi_base->is_s2i_kinetics_with_pseudo_contact())
  {
    const int numDofsPerNodeStresses = 6;
    std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetstresses =
        std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
            numDofsPerNodeStresses, 0, 0, true);
    if (structdis->add_dof_set(dofsetstresses) != ++structure_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in structure field");
    if (scatradis->add_dof_set(structdis->get_dof_set_proxy(structure_dofset_counter)) !=
        ++scatra_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in scatra field");
    scatra_integrator->set_number_of_dof_set_two_tensor_quantity(scatra_dofset_counter);
  }

  set_is_init(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::setup()
{
  check_is_init();

  set_is_setup(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::assign_material_pointers(
    std::shared_ptr<Core::FE::Discretization> structdis,
    std::shared_ptr<Core::FE::Discretization> scatradis)
{
  const int numelements = scatradis->num_my_col_elements();

  for (int i = 0; i < numelements; ++i)
  {
    Core::Elements::Element* scatratele = scatradis->l_col_element(i);
    const int gid = scatratele->id();

    Core::Elements::Element* structele = structdis->g_element(gid);

    // for coupling we add the source material to the target element and vice versa
    scatratele->add_material(structele->material());
    structele->add_material(scatratele->material());
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::set_mechanical_stress_state(
    Core::FE::Discretization& scatradis, const Core::LinAlg::Vector<double>& stress_state,
    const unsigned nds)
{
  scatradis.set_state(nds, "mechanicalStressState", stress_state);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::set_mesh_disp(
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra, const Core::LinAlg::Vector<double>& disp)
{
  scatra->scatra_field()->apply_mesh_movement(disp);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::set_velocity_fields(
    const std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    const Core::LinAlg::Vector<double>& convvel, const Core::LinAlg::Vector<double>& vel)
{
  scatra->scatra_field()->set_convective_velocity(convvel);
  scatra->scatra_field()->set_velocity_field(vel);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::set_scalar_field(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, const unsigned nds)
{
  dis.set_state(nds, "scalarfield", phi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::set_scalar_field_micro(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, const unsigned nds)
{
  dis.set_state(nds, "MicroCon", phi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolume::set_temperature_field(
    Core::FE::Discretization& structdis, std::shared_ptr<const Core::LinAlg::Vector<double>> temp)
{
  structdis.set_state(2, "temperature", *temp);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::set_temperature_field(
    Core::FE::Discretization& structdis, std::shared_ptr<const Core::LinAlg::Vector<double>> temp)
{
  structdis.set_state(2, "temperature", *temp);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::init(const int ndim,
    std::shared_ptr<Core::FE::Discretization> structdis, std::shared_ptr<SSI::SSIBase> ssi_base)
{
  Global::Problem* problem = SSI::Utils::problem_from_instance();

  set_is_setup(false);

  int scatra_dofset_counter = ssi_base->scatra_field()->get_max_dof_set_number();
  int structure_dofset_counter = 0;

  auto scatra_integrator = ssi_base->scatra_field();

  // set pointers
  structdis_ = structdis;
  scatradis_ = scatra_integrator->discretization();

  // set problem dimension
  problem_dimension_ = ndim;

  // first call fill_complete for single discretizations.
  // This way the physical dofs are numbered successively
  structdis_->fill_complete();
  scatradis_->fill_complete();

  // build auxiliary dofsets, i.e. pseudo dofs on each discretization
  const int ndofpernode_scatra = scatradis_->num_dof(0, scatradis_->l_row_node(0));
  const int ndofperelement_scatra = 0;
  const int ndofpernode_struct = structdis->num_dof(0, structdis->l_row_node(0));
  const int ndofperelement_struct = 0;
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
  dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
      ndofpernode_scatra, ndofperelement_scatra, 0, true);
  if (structdis->add_dof_set(dofsetaux) != ++structure_dofset_counter)
    FOUR_C_THROW("unexpected dof sets in structure field");
  dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
      ndofpernode_struct, ndofperelement_struct, 0, true);
  if (scatradis_->add_dof_set(dofsetaux) != ++scatra_dofset_counter)
    FOUR_C_THROW("unexpected dof sets in scatra field");
  scatra_integrator->set_number_of_dof_set_displacement(scatra_dofset_counter);
  scatra_integrator->set_number_of_dof_set_velocity(scatra_dofset_counter);

  // call assign_degrees_of_freedom also for auxiliary dofsets
  // note: the order of fill_complete() calls determines the gid numbering!
  // 1. structure dofs
  // 2. scatra dofs
  // 3. structure auxiliary dofs
  // 4. scatra auxiliary dofs
  structdis_->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  scatradis_->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  // setup mortar adapter for surface volume coupling
  adaptermeshtying_ = std::make_shared<Coupling::Adapter::CouplingMortar>(problem->n_dim(),
      problem->mortar_coupling_params(), problem->contact_dynamic_params(),
      problem->spatial_approximation_type());

  set_is_init(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::setup()
{
  Global::Problem* problem = SSI::Utils::problem_from_instance();

  check_is_init();

  std::vector<int> coupleddof(problem_dimension_, 1);
  // Setup of meshtying adapter

  adaptermeshtying_->setup(structdis_, scatradis_, nullptr, coupleddof, "SSICoupling",
      structdis_->get_comm(), problem->function_manager(), problem->binning_strategy_params(),
      problem->discretization_map(), problem->output_control_file(),
      problem->spatial_approximation_type(), false, false, 0, 1);


  // extractor for coupled surface of structure discretization with surface scatra
  extractor_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *structdis_->dof_row_map(0), adaptermeshtying_->target_dof_map(), true);

  set_is_setup(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::assign_material_pointers(
    std::shared_ptr<Core::FE::Discretization> structdis,
    std::shared_ptr<Core::FE::Discretization> scatradis)
{
  // nothing to do in this case, since
  // transferring scalar state to structure discretization not implemented for
  // transport on structural boundary. Only SolidToScatra coupling available.
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::set_mesh_disp(
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra, const Core::LinAlg::Vector<double>& disp)
{
  scatra->scatra_field()->apply_mesh_movement(
      *adaptermeshtying_->target_to_source(*extractor_->extract_cond_vector(disp)));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::set_velocity_fields(
    const std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    const Core::LinAlg::Vector<double>& convvel, const Core::LinAlg::Vector<double>& vel)
{
  scatra->scatra_field()->set_convective_velocity(
      *adaptermeshtying_->target_to_source(*extractor_->extract_cond_vector(convvel)));
  scatra->scatra_field()->set_velocity_field(
      *adaptermeshtying_->target_to_source(*extractor_->extract_cond_vector(vel)));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::set_scalar_field(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, unsigned nds)
{
  FOUR_C_THROW(
      "transferring scalar state to structure discretization not implemented for "
      "transport on structural boundary. Only SolidToScatra coupling available.");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingBoundary::set_scalar_field_micro(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, unsigned nds)
{
  FOUR_C_THROW("transferring micro scalar state to structure discretization not implemented.");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::init(const int ndim,
    std::shared_ptr<Core::FE::Discretization> structdis, std::shared_ptr<SSI::SSIBase> ssi_base)
{
  set_is_setup(false);

  int scatra_dofset_counter = ssi_base->scatra_field()->get_max_dof_set_number();
  int structure_dofset_counter = 0;

  auto scatra_integrator = ssi_base->scatra_field();
  auto scatradis = scatra_integrator->discretization();
  // first call fill_complete for single discretizations.
  // This way the physical dofs are numbered successively
  structdis->fill_complete();
  scatradis->fill_complete();

  // build auxiliary dofsets, i.e. pseudo dofs on each discretization
  const int ndofpernode_scatra = scatradis->num_dof(0, scatradis->l_row_node(0));
  const int ndofperelement_scatra = 0;
  const int ndofpernode_struct = structdis->num_dof(0, structdis->l_row_node(0));
  const int ndofperelement_struct = 0;
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
  dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
      ndofpernode_scatra, ndofperelement_scatra, 0, true);
  if (structdis->add_dof_set(dofsetaux) != ++structure_dofset_counter)
    FOUR_C_THROW("unexpected dof sets in structure field");
  dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
      ndofpernode_struct, ndofperelement_struct, 0, true);
  if (scatradis->add_dof_set(dofsetaux) != ++scatra_dofset_counter)
    FOUR_C_THROW("unexpected dof sets in scatra field");
  scatra_integrator->set_number_of_dof_set_displacement(scatra_dofset_counter);
  scatra_integrator->set_number_of_dof_set_velocity(scatra_dofset_counter);

  // call assign_degrees_of_freedom also for auxiliary dofsets
  // note: the order of fill_complete() calls determines the gid numbering!
  // 1. structure dofs
  // 2. scatra dofs
  // 3. structure auxiliary dofs
  // 4. scatra auxiliary dofs
  structdis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  scatradis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  // Scheme: non matching meshes --> volumetric mortar coupling...
  volcoupl_structurescatra_ = std::make_shared<Coupling::Adapter::MortarVolCoupl>();

  // init projection matrices (use default material strategy)
  volcoupl_structurescatra_->init(ndim, structdis, scatradis);

  // parallel redistribution is performed in the global control
  // algorithm. We redistribute between init(...) and setup().
  // volcoupl_structurescatra_->redistribute();

  set_is_init(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::setup()
{
  const auto* problem = SSI::Utils::problem_from_instance();

  check_is_init();

  // setup projection matrices (use default material strategy)
  volcoupl_structurescatra_->setup(problem->volmortar_params(), problem->cut_general_params());

  set_is_setup(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::assign_material_pointers(
    std::shared_ptr<Core::FE::Discretization> structdis,
    std::shared_ptr<Core::FE::Discretization> scatradis)
{
  const auto* problem = SSI::Utils::problem_from_instance();

  volcoupl_structurescatra_->assign_materials(
      structdis, scatradis, problem->volmortar_params(), problem->cut_general_params());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::set_mesh_disp(
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra, const Core::LinAlg::Vector<double>& disp)
{
  scatra->scatra_field()->apply_mesh_movement(
      *volcoupl_structurescatra_->apply_vector_mapping21(disp));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::set_velocity_fields(
    const std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    const Core::LinAlg::Vector<double>& convvel, const Core::LinAlg::Vector<double>& vel)
{
  scatra->scatra_field()->set_convective_velocity(
      *volcoupl_structurescatra_->apply_vector_mapping21(convvel));
  scatra->scatra_field()->set_velocity_field(
      *volcoupl_structurescatra_->apply_vector_mapping21(vel));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::set_scalar_field(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, const unsigned nds)
{
  dis.set_state(nds, "scalarfield", *volcoupl_structurescatra_->apply_vector_mapping12(phi));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingNonMatchingVolume::set_scalar_field_micro(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, unsigned nds)
{
  FOUR_C_THROW("transferring micro scalar state to structure discretization not implemented.");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::init(const int ndim,
    std::shared_ptr<Core::FE::Discretization> structdis, std::shared_ptr<SSI::SSIBase> ssi_base)
{
  const auto* problem = SSI::Utils::problem_from_instance();

  set_is_setup(false);

  int scatra_dofset_counter = ssi_base->scatra_field()->get_max_dof_set_number();
  int structure_dofset_counter = 0;

  auto scatra_integrator = ssi_base->scatra_field();
  auto scatradis = scatra_integrator->discretization();


  // Note : We need to make sure that the parallel distribution of Volume and Boundary
  //        is the same externally! The best thing is if you do this in your *_dyn.cpp,
  //        i.e., your global control algorithm.

  if (ssi_base->scatra_manifold_base_algorithm() == nullptr)
  {
    {
      // get condition which defines the coupling on target discretization
      std::vector<const Core::Conditions::Condition*> conds_struct;
      structdis->get_condition("SSICouplingSolidToScatra", conds_struct);

      // get condition which defines the coupling on source discretization
      std::vector<const Core::Conditions::Condition*> conds_scatra;
      scatradis->get_condition("SSICouplingSolidToScatra", conds_scatra);

      // at least one condition needs to be defined on each discretization
      if (conds_struct.size() == 0 or conds_scatra.size() == 0)
        FOUR_C_THROW(
            "No coupling condition defined on one or both structure or scatra discretization!");

      std::set<int> couplingids;
      for (auto& cond_struct : conds_struct)
        couplingids.insert(cond_struct->parameters().get<int>("coupling_id"));

      std::shared_ptr<Core::DOFSets::DofSetGIDBasedWrapper> structgidmatchingdofset =
          std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
              structdis, structdis->get_dof_set_proxy());

      std::shared_ptr<Core::DOFSets::DofSetDefinedMappingWrapper> newdofset_scatra =
          std::make_shared<Core::DOFSets::DofSetDefinedMappingWrapper>(
              structgidmatchingdofset, structdis, "SSICouplingSolidToScatra", couplingids);

      // add dofset and check if scatra field has 2 dofsets, so that coupling is possible
      if (scatradis->add_dof_set(newdofset_scatra) != ++scatra_dofset_counter)
        FOUR_C_THROW("unexpected dof sets in scatra field");
      scatra_integrator->set_number_of_dof_set_displacement(scatra_dofset_counter);
      scatra_integrator->set_number_of_dof_set_velocity(scatra_dofset_counter);
    }

    {
      // get condition which defines the coupling on target discretization
      std::vector<const Core::Conditions::Condition*> conds_struct;
      structdis->get_condition("SSICouplingScatraToSolid", conds_struct);

      // get condition which defines the coupling on source discretization
      std::vector<const Core::Conditions::Condition*> conds_scatra;
      scatradis->get_condition("SSICouplingScatraToSolid", conds_scatra);

      // at least one condition needs to be defined on each discretization
      if (conds_struct.size() == 0 or conds_scatra.size() == 0)
        FOUR_C_THROW(
            "No coupling condition defined on one or both structure or scatra discretization!");

      std::set<int> couplingids;
      for (auto& cond_struct : conds_struct)
        couplingids.insert(cond_struct->parameters().get<int>("coupling_id"));

      std::shared_ptr<Core::DOFSets::DofSetGIDBasedWrapper> scatragidmatchingdofset =
          std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
              scatradis, scatradis->get_dof_set_proxy());

      for (int coupling_id : couplingids)
      {
        std::set<int> tempest;
        tempest.insert(coupling_id);

        std::shared_ptr<Core::DOFSets::DofSetDefinedMappingWrapper> newdofset_struct =
            std::make_shared<Core::DOFSets::DofSetDefinedMappingWrapper>(
                scatragidmatchingdofset, scatradis, "SSICouplingScatraToSolid", tempest);

        structdis->add_dof_set(newdofset_struct);
      }
    }
  }
  else
  {
    int scatra_manifold_dofset_counter(0);

    auto scatra_manifold_integrator = ssi_base->scatra_manifold();
    auto scatra_manifold_dis = scatra_manifold_integrator->discretization();

    // build a proxy of the structure discretization for the other fields
    auto structdofset = structdis->get_dof_set_proxy();
    // build a proxy of the scatra discretization for the other fields
    auto scatradofset = scatradis->get_dof_set_proxy();

    // add proxy dofssets of other fields to discretizations and check if number of dofsets is
    // correct
    if (scatradis->add_dof_set(structdofset) != ++scatra_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in scatra field");
    scatra_integrator->set_number_of_dof_set_displacement(scatra_dofset_counter);
    scatra_integrator->set_number_of_dof_set_velocity(scatra_dofset_counter);
    if (structdis->add_dof_set(scatradofset) != ++structure_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in structure field");

    // set dummy coupling id, as coupling between scatra_manifold dis and structdis/scatradis should
    // be setup for all conditions
    std::set<int> couplingids;
    couplingids.insert(0);

    auto structgidmatchingdofset = std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
        structdis, structdis->get_dof_set_proxy());

    auto proxy_structure_scatramanifold =
        std::make_shared<Core::DOFSets::DofSetDefinedMappingWrapper>(
            structgidmatchingdofset, scatra_manifold_dis, "SSISurfaceManifold", couplingids);

    if (scatra_manifold_dis->add_dof_set(proxy_structure_scatramanifold) !=
        ++scatra_manifold_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in scatra manifold field");
    scatra_manifold_integrator->set_number_of_dof_set_displacement(scatra_manifold_dofset_counter);
    scatra_manifold_integrator->set_number_of_dof_set_velocity(scatra_manifold_dofset_counter);
  }

  if (problem->elch_control_params().get<int>("TEMPERATURE_FROM_FUNCT") != -1)
  {
    const int numDofsPerNodeTemp = 1;  // defined by temperature field

    std::shared_ptr<Core::DOFSets::DofSetInterface> dofsettemp =
        std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(numDofsPerNodeTemp, 0, 0, true);
    if (structdis->add_dof_set(dofsettemp) != ++structure_dofset_counter)
      FOUR_C_THROW("unexpected dof sets in structure field");
  }

  // exchange material pointers for coupled material formulations
  assign_material_pointers(structdis, scatradis);

  set_is_init(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::setup()
{
  check_is_init();

  set_is_setup(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::assign_material_pointers(
    std::shared_ptr<Core::FE::Discretization> structdis,
    std::shared_ptr<Core::FE::Discretization> scatradis)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::set_mesh_disp(
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra, const Core::LinAlg::Vector<double>& disp)
{
  scatra->scatra_field()->apply_mesh_movement(disp);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::set_velocity_fields(
    const std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra,
    const Core::LinAlg::Vector<double>& convvel, const Core::LinAlg::Vector<double>& vel)
{
  scatra->scatra_field()->set_convective_velocity(convvel);
  scatra->scatra_field()->set_velocity_field(vel);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::set_scalar_field(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, const unsigned nds)
{
  dis.set_state(nds, "scalarfield", phi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSICouplingMatchingVolumeAndBoundary::set_scalar_field_micro(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& phi, unsigned nds)
{
  FOUR_C_THROW("transferring micro scalar state to structure discretization not implemented.");
}

FOUR_C_NAMESPACE_CLOSE
