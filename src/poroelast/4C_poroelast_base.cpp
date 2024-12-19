// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_base.hpp"

#include "4C_adapter_fld_base_algorithm.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_contact_lagrange_strategy_poro.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_dofset_gidbased_wrapper.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_so3_base.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_structure_aux.hpp"
#include "4C_utils_parameter_list.hpp"

#include <cstddef>

FOUR_C_NAMESPACE_OPEN


PoroElast::PoroBase::PoroBase(MPI_Comm comm, const Teuchos::ParameterList& timeparams,
    std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter)
    : AlgorithmBase(comm, timeparams),
      is_part_of_multifield_problem_(false),
      porosity_splitter_(porosity_splitter),
      matchinggrid_(
          Global::Problem::instance()->poroelast_dynamic_params().get<bool>("MATCHINGGRID")),
      oldstructimint_(Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(
                          Global::Problem::instance()->structural_dynamic_params(),
                          "INT_STRATEGY") == Inpar::Solid::int_old)
{
  if (Global::Problem::instance()->get_problem_type() != Core::ProblemType::poroelast)
    is_part_of_multifield_problem_ = true;

  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structdis =
      Global::Problem::instance()->get_dis("structure");

  if (!matchinggrid_)
  {
    std::shared_ptr<Core::FE::Discretization> fluiddis =
        Global::Problem::instance()->get_dis("porofluid");
    // Scheme: non matching meshes --> volumetric mortar coupling...
    volcoupl_ = std::make_shared<Coupling::Adapter::MortarVolCoupl>();

    // build material strategy
    std::shared_ptr<Utils::PoroMaterialStrategy> materialstrategy =
        std::make_shared<Utils::PoroMaterialStrategy>();

    // setup projection matrices
    volcoupl_->init(Global::Problem::instance()->n_dim(), structdis, fluiddis, nullptr, nullptr,
        nullptr, nullptr, materialstrategy);
    Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
        binning_params);

    volcoupl_->redistribute(binning_params, Global::Problem::instance()->output_control_file());
    volcoupl_->setup(Global::Problem::instance()->volmortar_params(),
        Global::Problem::instance()->cut_general_params());
  }

  // access structural dynamic params list which will be possibly modified while creating the time
  // integrator
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();

  // create the structural time integrator (init() called inside)
  // clean up as soon as old time integration is unused!
  if (oldstructimint_)
  {
    Adapter::StructureBaseAlgorithm structure(
        timeparams, const_cast<Teuchos::ParameterList&>(sdyn), structdis);
    structure_ =
        std::dynamic_pointer_cast<Adapter::FPSIStructureWrapper>(structure.structure_field());
    structure_->setup();
  }
  else
  {
    std::shared_ptr<Adapter::StructureBaseAlgorithmNew> adapterbase_ptr =
        Adapter::build_structure_algorithm(sdyn);
    adapterbase_ptr->init(timeparams, const_cast<Teuchos::ParameterList&>(sdyn), structdis);
    adapterbase_ptr->setup();
    structure_ = std::dynamic_pointer_cast<Adapter::FPSIStructureWrapper>(
        adapterbase_ptr->structure_field());
  }

  if (structure_ == nullptr)
    FOUR_C_THROW("cast from Adapter::Structure to Adapter::FPSIStructureWrapper failed");

  // ask base algorithm for the fluid time integrator
  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fluiddynparams = problem->fluid_dynamic_params();
  std::shared_ptr<Adapter::FluidBaseAlgorithm> fluid =
      std::make_shared<Adapter::FluidBaseAlgorithm>(timeparams, fluiddynparams, "porofluid", true);
  fluid_ = std::dynamic_pointer_cast<Adapter::FluidPoro>(fluid->fluid_field());

  if (fluid_ == nullptr)
    FOUR_C_THROW("cast from Adapter::FluidBaseAlgorithm to Adapter::FluidPoro failed");

  // as this is a two way coupled problem, every discretization needs to know the other one.
  // For this we use DofSetProxies and coupling objects which are setup here
  setup_coupling();

  if (submeshes_) replace_dof_sets();

  // extractor for constraints on structure phase
  //
  // when using constraints applied via Lagrange-Multipliers there is a
  // difference between structure_field()->dof_row_map() and structure_field()->dof_row_map(0).
  // structure_field()->dof_row_map(0) returns the dof_row_map
  // known to the discretization (without lagrange multipliers)
  // while structure_field()->dof_row_map() returns the dof_row_map known to
  // the constraint manager (with lagrange multipliers)
  cond_splitter_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *structure_field()->dof_row_map(), structure_field()->dof_row_map(0));

  // look for special poro conditions and set flags
  check_for_poro_conditions();

  // do some checks
  {
    // access the problem-specific parameter lists
    const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();

    std::vector<Core::Conditions::Condition*> porocoupl;
    fluid_field()->discretization()->get_condition("PoroCoupling", porocoupl);
    if (porocoupl.size() == 0)
      FOUR_C_THROW(
          "no Poro Coupling Condition defined for porous media problem. Fix your input file!");

    // check time integration algo -> currently only one-step-theta scheme supported
    auto structtimealgo = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYPE");
    auto fluidtimealgo =
        Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(fdyn, "TIMEINTEGR");

    if (not((structtimealgo == Inpar::Solid::dyna_onesteptheta and
                fluidtimealgo == Inpar::FLUID::timeint_one_step_theta) or
            (structtimealgo == Inpar::Solid::dyna_statics and
                fluidtimealgo == Inpar::FLUID::timeint_stationary) or
            (structtimealgo == Inpar::Solid::dyna_genalpha and
                (fluidtimealgo == Inpar::FLUID::timeint_afgenalpha or
                    fluidtimealgo == Inpar::FLUID::timeint_npgenalpha))))
    {
      FOUR_C_THROW(
          "porous media problem is limited in functionality (only one-step-theta scheme, "
          "stationary and (af)genalpha case possible)");
    }

    if (fluidtimealgo == Inpar::FLUID::timeint_npgenalpha)
    {
      FOUR_C_THROW(
          "npgenalpha time integration for porous fluid is possibly not valid. Either check the "
          "theory or use afgenalpha instead!");
    }

    if (structtimealgo == Inpar::Solid::dyna_onesteptheta and
        fluidtimealgo == Inpar::FLUID::timeint_one_step_theta)
    {
      double theta_struct = sdyn.sublist("ONESTEPTHETA").get<double>("THETA");
      double theta_fluid = fdyn.get<double>("THETA");

      if (theta_struct != theta_fluid)
      {
        FOUR_C_THROW(
            "porous media problem is limited in functionality. Only one-step-theta scheme with "
            "equal theta for both fields possible. Fix your input file.");
      }
    }

    auto damping = Teuchos::getIntegralValue<Inpar::Solid::DampKind>(sdyn, "DAMPING");
    if (damping != Inpar::Solid::DampKind::damp_material &&
        structtimealgo != Inpar::Solid::dyna_statics)
    {
      FOUR_C_THROW(
          "Material damping has to be used for dynamic porous media simulations! Set DAMPING to "
          "'Material' in the STRUCTURAL DYNAMIC section.");
    }

    // access the problem-specific parameter lists
    const Teuchos::ParameterList& pedyn = Global::Problem::instance()->poroelast_dynamic_params();
    auto physicaltype =
        Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(pedyn, "PHYSICAL_TYPE");
    if (porosity_dof_ and physicaltype != Inpar::FLUID::poro_p1)
    {
      FOUR_C_THROW(
          "Poro P1 elements need a special fluid. Set 'PHYSICAL_TYPE' to 'Poro_P1' in the "
          "POROELASTICITY DYNAMIC section!");
    }

    auto transientfluid =
        Teuchos::getIntegralValue<Inpar::PoroElast::TransientEquationsOfPoroFluid>(
            pedyn, "TRANSIENT_TERMS");

    if (fluidtimealgo == Inpar::FLUID::timeint_stationary)
    {
      if (transientfluid != Inpar::PoroElast::transient_none)
      {
        FOUR_C_THROW(
            "Invalid option for stationary fluid! Set 'TRANSIENT_TERMS' in section POROELASTICITY "
            "DYNAMIC to 'none'!");
      }
    }
    else
    {
      if (transientfluid == Inpar::PoroElast::transient_none)
      {
        FOUR_C_THROW(
            "Invalid option for stationary fluid! Set 'TRANSIENT_TERMS' in section POROELASTICITY "
            "DYNAMIC to valid parameter!");
      }
    }

    if (transientfluid == Inpar::PoroElast::transient_momentum_only)
    {
      FOUR_C_THROW(
          "Option 'momentum' for parameter 'TRANSIENT_TERMS' in section POROELASTICITY DYNAMIC is "
          "not working properly! There is probably a bug in the linearization ....");
    }
  }
}

void PoroElast::PoroBase::read_restart(const int step)
{
  if (step)
  {
    if (not oldstructimint_) structure_->setup();

    // apply current velocity and pressures to structure
    set_fluid_solution();
    // apply current structural displacements to fluid
    set_struct_solution();

    fluid_field()->read_restart(step);
    structure_field()->read_restart(step);

    // in case of submeshes, we need to rebuild the subproxies, also (they are reset during restart)
    // for the new timeintegration this cannot be done as it would call evaluate while materials are
    // deleted due to read_restart(). It is however not needed as the subproxies are intact
    if (submeshes_ && oldstructimint_) replace_dof_sets();

    // apply current velocity and pressures to structure
    set_fluid_solution();
    // apply current structural displacements to fluid
    set_struct_solution();

    // second read_restart needed due to the coupling variables
    fluid_field()->read_restart(step);
    structure_field()->read_restart(step);

    // in case of submeshes, we need to rebuild the subproxies, also (they are reset during restart)
    if (submeshes_ && oldstructimint_) replace_dof_sets();

    // set the current time in the algorithm (taken from fluid field)
    set_time_step(fluid_field()->time(), step);

    // Material pointers to other field were deleted during read_restart().
    // They need to be reset.
    if (matchinggrid_)
    {
      PoroElast::Utils::set_material_pointers_matching_grid(
          *structure_field()->discretization(), *fluid_field()->discretization());
    }
    else
    {
      // build material strategy
      std::shared_ptr<Utils::PoroMaterialStrategy> materialstrategy =
          std::make_shared<Utils::PoroMaterialStrategy>();

      volcoupl_->assign_materials(structure_field()->discretization(),
          fluid_field()->discretization(), Global::Problem::instance()->volmortar_params(),
          Global::Problem::instance()->cut_general_params(), materialstrategy);
    }
  }
}

void PoroElast::PoroBase::prepare_time_step()
{
  // counter and print header
  increment_time_and_step();
  if (!is_part_of_multifield_problem_) print_header();

  // set fluid velocities and pressures onto the structure
  set_fluid_solution();

  // call the predictor
  structure_field()->prepare_time_step();

  // set structure displacements onto the fluid
  set_struct_solution();

  // call the predictor
  fluid_field()->prepare_time_step();
}

void PoroElast::PoroBase::update()
{
  structure_field()->update();
  fluid_field()->update();
  // clean up as soon as old time integration is unused!
  if (oldstructimint_)
  {
    if (structure_field()->meshtying_contact_bridge() != nullptr)
    {
      if (structure_field()->meshtying_contact_bridge()->have_contact() && !nit_contact_)
      {
        (static_cast<CONTACT::LagrangeStrategyPoro&>(
             structure_field()->meshtying_contact_bridge()->contact_manager()->get_strategy()))
            .update_poro_contact();
      }
    }
  }
}

void PoroElast::PoroBase::prepare_output(bool force_prepare_timestep)
{
  structure_field()->prepare_output(force_prepare_timestep);
}

void PoroElast::PoroBase::test_results(MPI_Comm comm)
{
  Global::Problem::instance()->add_field_test(structure_field()->create_field_test());
  Global::Problem::instance()->add_field_test(fluid_field()->create_field_test());
  Global::Problem::instance()->test_all(comm);
}

std::shared_ptr<Core::LinAlg::Vector<double>> PoroElast::PoroBase::structure_to_fluid_field(
    const Core::LinAlg::Vector<double>& iv)
{
  if (matchinggrid_)
  {
    if (submeshes_)
      return coupling_fluid_structure_->master_to_slave(*psi_extractor_->extract_cond_vector(iv));
    else
      return coupling_fluid_structure_->master_to_slave(iv);
  }
  else
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> mv = volcoupl_->apply_vector_mapping21(iv);

    std::shared_ptr<Core::LinAlg::Vector<double>> sv =
        Core::LinAlg::create_vector(*(fluid_field()->vel_pres_splitter()->other_map()));

    std::copy(mv->Values(),
        mv->Values() + (static_cast<ptrdiff_t>(mv->MyLength() * mv->NumVectors())), sv->Values());
    return sv;
  }
}

void PoroElast::PoroBase::set_struct_solution()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  // apply current displacements and velocities to the fluid field
  if (structure_field()->have_constraint())
  {
    // displacement vector without lagrange-multipliers
    dispnp = cond_splitter_->extract_cond_vector(*structure_field()->dispnp());
  }
  else
    dispnp = structure_field()->dispnp();

  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = structure_field()->velnp();

  // transfer the current structure displacement to the fluid field
  std::shared_ptr<Core::LinAlg::Vector<double>> structdisp = structure_to_fluid_field(*dispnp);
  fluid_field()->apply_mesh_displacement(structdisp);

  // transfer the current structure velocity to the fluid field
  std::shared_ptr<Core::LinAlg::Vector<double>> structvel = structure_to_fluid_field(*velnp);
  fluid_field()->apply_mesh_velocity(structvel);
}

void PoroElast::PoroBase::set_fluid_solution()
{
  if (matchinggrid_)
  {
    structure_field()->discretization()->set_state(1, "fluidvel", fluid_field()->velnp());
  }
  else
  {
    structure_field()->discretization()->set_state(
        1, "fluidvel", volcoupl_->apply_vector_mapping12(*fluid_field()->velnp()));
  }
}

void PoroElast::PoroBase::time_loop()
{
  while (not_finished())
  {
    // solve one time step
    do_time_step();
  }
}

void PoroElast::PoroBase::output(bool forced_writerestart)
{
  // Note: The order is important here! In here control file entries are
  // written. And these entries define the order in which the filters handle
  // the Discretizations, which in turn defines the dof number ordering of the
  // Discretizations.
  fluid_field()->statistics_and_output();
  structure_field()->output(forced_writerestart);
}

void PoroElast::PoroBase::setup_coupling()
{
  // get discretizations
  std::shared_ptr<Core::FE::Discretization> structdis = structure_field()->discretization();
  std::shared_ptr<Core::FE::Discretization> fluiddis = fluid_field()->discretization();

  // if one discretization is a subset of the other, they will differ in node number (and element
  // number) we assume matching grids for the overlapping part here
  const Epetra_Map* structnoderowmap = structdis->node_row_map();
  const Epetra_Map* fluidnoderowmap = fluiddis->node_row_map();

  const int numglobalstructnodes = structnoderowmap->NumGlobalElements();
  const int numglobalfluidnodes = fluidnoderowmap->NumGlobalElements();

  if (matchinggrid_)
  {
    // check for submeshes
    submeshes_ = (numglobalstructnodes != numglobalfluidnodes);
  }
  else
    submeshes_ = false;

  const int ndim = Global::Problem::instance()->n_dim();
  const int numglobalstructdofs = structdis->dof_row_map()->NumGlobalElements();
  if (numglobalstructdofs == numglobalstructnodes * ndim)
    porosity_dof_ = false;
  else
  {
    porosity_dof_ = true;
    if (!porosity_splitter_)
    {
      porosity_splitter_ =
          PoroElast::Utils::build_poro_splitter(*structure_field()->discretization());
    }
  }

  coupling_fluid_structure_ = std::make_shared<Coupling::Adapter::Coupling>();
  int ndof = ndim;

  // if the porosity is a primary variable, we get one more dof
  if (porosity_dof_) ndof++;

  if (matchinggrid_)
  {
    if (submeshes_)
    {
      // for submeshes we only couple a part of the structure disc. with the fluid disc.
      // we use the fact, that we have matching grids and matching gids
      // The node matching search tree is used to find matching structure and fluid nodes.
      // Note, that the structure discretization must be the bigger one (because it is the
      // masterdis).
      coupling_fluid_structure_->setup_coupling(
          *structdis, *fluiddis, *fluidnoderowmap, *fluidnoderowmap, ndof, false);
    }
    else
    {
      // matching grid case: we rely on that the cloning strategy build the fluid node map with
      // equal node gids as the structure and also identical parallel distribution. Hence, we do not
      // use the node search tree here and use the same fluid node map also as permuted map.
      coupling_fluid_structure_->setup_coupling(
          *structdis, *fluiddis, *structnoderowmap, *fluidnoderowmap, *fluidnoderowmap, ndof);
    }

    fluid_field()->set_mesh_map(coupling_fluid_structure_->slave_dof_map());

    if (submeshes_)
      psi_extractor_ = std::make_shared<Core::LinAlg::MapExtractor>(
          *structure_field()->dof_row_map(), coupling_fluid_structure_->master_dof_map());
  }
  else
  {
    fluid_field()->set_mesh_map(fluid_field()->vel_pres_splitter()->other_map());
  }
}

void PoroElast::PoroBase::replace_dof_sets()
{
  // the problem is two way coupled, thus each discretization must know the other discretization

  // get discretizations
  std::shared_ptr<Core::FE::Discretization> structdis = structure_field()->discretization();
  std::shared_ptr<Core::FE::Discretization> fluiddis = fluid_field()->discretization();

  /* When coupling porous media with a pure structure we will have two discretizations
   * of different size. In this case we need a special proxy, which can handle submeshes.
   */
  if (submeshes_)
  {
    std::shared_ptr<Core::DOFSets::DofSetGIDBasedWrapper> structsubdofset =
        std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
            structdis, structdis->get_dof_set_proxy());
    std::shared_ptr<Core::DOFSets::DofSetGIDBasedWrapper> fluidsubdofset =
        std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
            fluiddis, fluiddis->get_dof_set_proxy());

    fluiddis->replace_dof_set(1, structsubdofset);
    structdis->replace_dof_set(1, fluidsubdofset);
  }
  else
  {
    // build a proxy of the structure discretization for the fluid field
    std::shared_ptr<Core::DOFSets::DofSetInterface> structdofsetproxy =
        structdis->get_dof_set_proxy();
    // build a proxy of the fluid discretization for the structure field
    std::shared_ptr<Core::DOFSets::DofSetInterface> fluiddofsetproxy =
        fluiddis->get_dof_set_proxy();

    fluiddis->replace_dof_set(1, structdofsetproxy);
    structdis->replace_dof_set(1, fluiddofsetproxy);
  }

  fluiddis->fill_complete(true, true, true);
  structdis->fill_complete(true, true, true);

  // for the new time integration setup() has to be called after structdis->fill_complete to make
  // sure all pointers connected to structdis are updated
  if (!oldstructimint_) structure_->setup();
}

void PoroElast::PoroBase::check_for_poro_conditions()
{
  std::vector<Core::Conditions::Condition*> nopencond;
  fluid_field()->discretization()->get_condition("no_penetration", nopencond);
  nopen_handle_ = std::make_shared<PoroElast::NoPenetrationConditionHandle>(nopencond);

  part_int_cond_ = false;
  std::vector<Core::Conditions::Condition*> poroPartInt;
  fluid_field()->discretization()->get_condition("PoroPartInt", poroPartInt);
  if (poroPartInt.size()) part_int_cond_ = true;

  pres_int_cond_ = false;
  std::vector<Core::Conditions::Condition*> poroPresInt;
  fluid_field()->discretization()->get_condition("PoroPresInt", poroPresInt);
  if (poroPresInt.size()) pres_int_cond_ = true;
}

void PoroElast::NoPenetrationConditionHandle::buid_no_penetration_map(
    MPI_Comm comm, std::shared_ptr<const Epetra_Map> dofRowMap)
{
  std::vector<int> condIDs;
  std::set<int>::iterator it;
  for (it = cond_ids_->begin(); it != cond_ids_->end(); it++)
  {
    condIDs.push_back(*it);
  }
  std::shared_ptr<Epetra_Map> nopendofmap = std::make_shared<Epetra_Map>(
      -1, int(condIDs.size()), condIDs.data(), 0, Core::Communication::as_epetra_comm(comm));

  nopenetration_ = std::make_shared<Core::LinAlg::MapExtractor>(*dofRowMap, nopendofmap);
}

void PoroElast::NoPenetrationConditionHandle::apply_cond_rhs(
    Core::LinAlg::Vector<double>& iterinc, Core::LinAlg::Vector<double>& rhs)
{
  if (has_cond_)
  {
    const std::shared_ptr<const Epetra_Map>& nopenetrationmap = nopenetration_->Map(1);
    Core::LinAlg::apply_dirichlet_to_system(iterinc, rhs, *cond_rhs_, *nopenetrationmap);
  }
}

void PoroElast::NoPenetrationConditionHandle::clear(PoroElast::Coupltype coupltype)
{
  if (has_cond_)
  {
    cond_rhs_->PutScalar(0.0);
    cond_ids_->clear();
    switch (coupltype)
    {
      case PoroElast::fluidfluid:
        fluid_fluid_constraint_matrix_->zero();
        cond_dofs_->PutScalar(0.0);
        break;
      case PoroElast::fluidstructure:
        fluid_structure_constraint_matrix_->zero();
        structure_vel_constraint_matrix_->zero();
        break;
      default:
        cond_dofs_->PutScalar(0.0);
        fluid_fluid_constraint_matrix_->zero();
        fluid_structure_constraint_matrix_->zero();
        structure_vel_constraint_matrix_->zero();
        break;
    }
  }
}

void PoroElast::NoPenetrationConditionHandle::setup(
    const Epetra_Map& dofRowMap, const Epetra_Map* dofRowMapFluid)
{
  if (has_cond_)
  {
    cond_rhs_ = std::make_shared<Core::LinAlg::Vector<double>>(dofRowMap, true);

    cond_dofs_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofRowMapFluid, true);

    fluid_fluid_constraint_matrix_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dofRowMapFluid, 81, true, true);

    fluid_structure_constraint_matrix_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dofRowMapFluid, 81, true, true);

    structure_vel_constraint_matrix_ =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dofRowMapFluid, 81, true, true);
  }
}

std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroElast::NoPenetrationConditionHandle::constraint_matrix(PoroElast::Coupltype coupltype)
{
  if (has_cond_)
  {
    if (coupltype == PoroElast::fluidfluid)
      return fluid_fluid_constraint_matrix_;
    else if (coupltype == PoroElast::fluidstructure)
      return fluid_structure_constraint_matrix_;
  }
  return nullptr;
}

std::shared_ptr<Core::LinAlg::SparseMatrix>
PoroElast::NoPenetrationConditionHandle::struct_vel_constraint_matrix(
    PoroElast::Coupltype coupltype)
{
  if (has_cond_)
  {
    if (coupltype == PoroElast::fluidfluid)
      return nullptr;
    else if (coupltype == PoroElast::fluidstructure)
      return structure_vel_constraint_matrix_;
  }
  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
