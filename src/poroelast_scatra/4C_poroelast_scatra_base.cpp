// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_scatra_base.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_coupling_volmortar_utils.hpp"
#include "4C_fem_dofset_gidbased_wrapper.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_poroelast_base.hpp"
#include "4C_poroelast_scatra_utils.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
PoroElastScaTra::PoroScatraBase::PoroScatraBase(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : AlgorithmBase(comm, timeparams),
      matchinggrid_(
          Global::Problem::instance()->poro_scatra_control_params().get<bool>("MATCHINGGRID")),
      volcoupl_structurescatra_(nullptr),
      volcoupl_fluidscatra_(nullptr),
      ssiinterfacecontact_(
          Global::Problem::instance()->get_dis("structure")->get_condition("SSIInterfaceContact") !=
          nullptr)
{
  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& scatradyn = problem->scalar_transport_dynamic_params();

  // do some checks
  {
    auto timealgo =
        Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(scatradyn, "TIMEINTEGR");
    if (timealgo != Inpar::ScaTra::timeint_one_step_theta and
        timealgo != Inpar::ScaTra::timeint_stationary)
      FOUR_C_THROW(
          "scalar transport in porous media is limited in functionality (only one-step-theta "
          "scheme or stationary case possible)");

    auto velfield = scatradyn.get<Inpar::ScaTra::VelocityField>("VELOCITYFIELD");
    if (velfield != Inpar::ScaTra::velocity_Navier_Stokes)
      FOUR_C_THROW(
          "scalar transport is coupled with the porous medium. Set 'VELOCITYFIELD' to "
          "'Navier_Stokes' in the SCALAR TRANSPORT DYNAMIC section! ");
  }

  // the problem is two-way coupled, thus each discretization must know the other discretization
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");
  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis("porofluid");
  std::shared_ptr<Core::FE::Discretization> scatradis = problem->get_dis("scatra");
  setup_coupling(structdis, fluiddis, scatradis);
  // Create the two uncoupled subproblems.
  // 1. poro problem
  poro_ = PoroElast::Utils::create_poro_algorithm(
      timeparams, comm, false, PoroElastScaTra::Utils::build_poro_scatra_splitter(*structdis));

  // get the solver number used for ScalarTransport solver
  const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
  // 2. scatra problem
  scatra_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(
      timeparams, scatradyn, problem->solver_params(linsolvernumber), "scatra", true);

  // now we can call init() on the base algo.
  // time integrator is constructed and initialized inside.
  scatra_->init();
  scatra_->scatra_field()->set_number_of_dof_set_displacement(2);
  scatra_->scatra_field()->set_number_of_dof_set_velocity(2);

  // only now we must call setup() on the scatra time integrator.
  // all objects relying on the parallel distribution are
  // created and pointers are set.
  // calls setup() on the time integrator inside.
  scatra_->setup();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::setup_system() { poro_->setup_system(); }

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::test_results(MPI_Comm comm)
{
  Global::Problem* problem = Global::Problem::instance();

  problem->add_field_test(poro_->structure_field()->create_field_test());
  problem->add_field_test(poro_->fluid_field()->create_field_test());
  problem->add_field_test(scatra_->create_scatra_field_test());
  problem->test_all(comm);
}

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::set_poro_solution()
{
  set_mesh_disp();
  set_velocity_fields();
}

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::set_scatra_solution()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_s = nullptr;
  std::shared_ptr<const Core::LinAlg::Vector<double>> phin_s = nullptr;
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_f = nullptr;
  std::shared_ptr<const Core::LinAlg::Vector<double>> phin_f = nullptr;
  std::shared_ptr<const Core::LinAlg::Vector<double>> phidtnp = nullptr;

  if (matchinggrid_)
  {
    phinp_s = scatra_->scatra_field()->phinp();
    phinp_f = phinp_s;
    phin_s = scatra_->scatra_field()->phin();
    phin_f = phin_s;
    phidtnp = scatra_->scatra_field()->phidtnp();
  }
  else
  {
    phinp_s = volcoupl_structurescatra_->apply_vector_mapping12(*scatra_->scatra_field()->phinp());
    phinp_f = volcoupl_fluidscatra_->apply_vector_mapping12(*scatra_->scatra_field()->phinp());
    phin_s = volcoupl_structurescatra_->apply_vector_mapping12(*scatra_->scatra_field()->phin());
    phin_f = volcoupl_fluidscatra_->apply_vector_mapping12(*scatra_->scatra_field()->phin());
    phidtnp = volcoupl_fluidscatra_->apply_vector_mapping12(*scatra_->scatra_field()->phidtnp());
  }

  // porous structure
  poro_->structure_field()->discretization()->set_state(2, "scalar", phinp_s);
  poro_->structure_field()->discretization()->set_state(2, "scalarn", phin_s);

  // porous fluid
  poro_->fluid_field()->set_iter_scalar_fields(phinp_f, phin_f, phidtnp,
      // scatra_->ScaTraField()->discretization()
      poro_->fluid_field()->discretization(), 2);
}

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::set_velocity_fields()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> convel = nullptr;
  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = nullptr;

  if (matchinggrid_)
  {
    convel = poro_->fluid_field()->convective_vel();
    velnp = poro_->fluid_field()->velnp();
  }
  else
  {
    convel = volcoupl_fluidscatra_->apply_vector_mapping21(*poro_->fluid_field()->convective_vel());
    velnp = volcoupl_fluidscatra_->apply_vector_mapping21(*poro_->fluid_field()->velnp());
  }

  scatra_->scatra_field()->set_velocity_field(convel,  // convective vel.
      nullptr,                                         // acceleration
      velnp,                                           // velocity
      nullptr,                                         // fsvel
      true                                             // set pressure
  );
}

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::set_mesh_disp()
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp = nullptr;

  if (matchinggrid_)
  {
    dispnp = poro_->fluid_field()->dispnp();
  }
  else
  {
    dispnp = volcoupl_fluidscatra_->apply_vector_mapping21(*fluid_field()->dispnp());
  }

  scatra_->scatra_field()->apply_mesh_movement(dispnp);

  std::shared_ptr<const Core::LinAlg::Vector<double>> sdispnp = nullptr;

  if (matchinggrid_)
  {
    sdispnp = structure_field()->dispnp();
  }
  else
  {
    sdispnp = volcoupl_structurescatra_->apply_vector_mapping21(*structure_field()->dispnp());
  }

  scatra_->scatra_field()->discretization()->set_state(1, "displacement", sdispnp);
}

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::replace_dof_sets(
    std::shared_ptr<Core::FE::Discretization> structdis,
    std::shared_ptr<Core::FE::Discretization> fluiddis,
    std::shared_ptr<Core::FE::Discretization> scatradis)
{
  if (matchinggrid_)
  {
    // the problem is two way coupled, thus each discretization must know the other discretization

    if (poro_field()->has_submeshes())
    {
      std::shared_ptr<Core::DOFSets::DofSetGIDBasedWrapper> structsubdofset =
          std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
              structdis, structdis->get_dof_set_proxy());
      std::shared_ptr<Core::DOFSets::DofSetGIDBasedWrapper> fluidsubdofset =
          std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
              fluiddis, fluiddis->get_dof_set_proxy());
      std::shared_ptr<Core::DOFSets::DofSetGIDBasedWrapper> scatrasubdofset =
          std::make_shared<Core::DOFSets::DofSetGIDBasedWrapper>(
              scatradis, scatradis->get_dof_set_proxy());

      scatradis->replace_dof_set(1, structsubdofset);
      scatradis->replace_dof_set(2, fluidsubdofset);
      structdis->replace_dof_set(2, scatrasubdofset);
      fluiddis->replace_dof_set(2, scatrasubdofset);
    }
    else
    {
      // build a proxy of the structure discretization for the scatra field
      std::shared_ptr<Core::DOFSets::DofSetInterface> structdofset = structdis->get_dof_set_proxy();
      // build a proxy of the fluid discretization for the scatra field
      std::shared_ptr<Core::DOFSets::DofSetInterface> fluiddofset = fluiddis->get_dof_set_proxy();
      // build a proxy of the fluid discretization for the structure/fluid field
      std::shared_ptr<Core::DOFSets::DofSetInterface> scatradofset = scatradis->get_dof_set_proxy();

      scatradis->replace_dof_set(1, structdofset);
      scatradis->replace_dof_set(2, fluiddofset);
      structdis->replace_dof_set(2, scatradofset);
      fluiddis->replace_dof_set(2, scatradofset);
    }

    fluiddis->fill_complete();
    scatradis->fill_complete();
    structdis->fill_complete();
  }
  else
  {
    FOUR_C_THROW("restart for non-matching poro-scatra not yet tested. Feel free to try");
  }
}

/*----------------------------------------------------------------------*
 |                                                         vuong 05/13  |
 *----------------------------------------------------------------------*/
void PoroElastScaTra::PoroScatraBase::setup_coupling(
    std::shared_ptr<Core::FE::Discretization> structdis,
    std::shared_ptr<Core::FE::Discretization> fluiddis,
    std::shared_ptr<Core::FE::Discretization> scatradis)
{
  if (not matchinggrid_)
  {
    // Scheme: non matching meshes --> volumetric mortar coupling...
    volcoupl_structurescatra_ = std::make_shared<Coupling::Adapter::MortarVolCoupl>();
    volcoupl_fluidscatra_ = std::make_shared<Coupling::Adapter::MortarVolCoupl>();

    std::pair<int, int> dofsets12_structurescatra = std::pair<int, int>(2, 0);
    std::pair<int, int> dofsets21_structurescatra = std::pair<int, int>(1, 0);
    std::pair<int, int> dofsets12_fluidscatra = std::pair<int, int>(2, 0);
    std::pair<int, int> dofsets21_fluidscatra = std::pair<int, int>(2, 0);

    // setup projection matrices (use default material strategy)
    volcoupl_structurescatra_->init(Global::Problem::instance()->n_dim(), structdis, scatradis,
        nullptr, nullptr, &dofsets12_structurescatra, &dofsets21_structurescatra, nullptr);
    volcoupl_fluidscatra_->init(Global::Problem::instance()->n_dim(), fluiddis, scatradis, nullptr,
        nullptr, &dofsets12_fluidscatra, &dofsets21_fluidscatra, nullptr);

    volcoupl_structurescatra_->setup(Global::Problem::instance()->volmortar_params(),
        Global::Problem::instance()->cut_general_params());
    volcoupl_fluidscatra_->setup(Global::Problem::instance()->volmortar_params(),
        Global::Problem::instance()->cut_general_params());
  }
}

FOUR_C_NAMESPACE_CLOSE
