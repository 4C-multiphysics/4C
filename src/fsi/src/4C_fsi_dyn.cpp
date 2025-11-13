// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_dyn.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_fld_fluid_xfsi.hpp"
#include "4C_adapter_fld_moving_boundary.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_adapter_str_poro_wrapper.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_ale_ale3.hpp"
#include "4C_ale_utils_clonestrategy.hpp"
#include "4C_beam3_base.hpp"
#include "4C_binstrategy.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fbi_input.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_dofset_fixed_size.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fluid_xfluid_fluid.hpp"
#include "4C_fsi_dirichletneumann_disp.hpp"
#include "4C_fsi_dirichletneumann_factory.hpp"
#include "4C_fsi_dirichletneumann_vel.hpp"
#include "4C_fsi_dirichletneumann_volcoupl.hpp"
#include "4C_fsi_dirichletneumannslideale.hpp"
#include "4C_fsi_fluid_ale.hpp"
#include "4C_fsi_fluidfluidmonolithic_fluidsplit.hpp"
#include "4C_fsi_fluidfluidmonolithic_fluidsplit_nonox.hpp"
#include "4C_fsi_fluidfluidmonolithic_structuresplit.hpp"
#include "4C_fsi_fluidfluidmonolithic_structuresplit_nonox.hpp"
#include "4C_fsi_monolithicfluidsplit.hpp"
#include "4C_fsi_monolithicstructuresplit.hpp"
#include "4C_fsi_mortarmonolithic_fluidsplit.hpp"
#include "4C_fsi_mortarmonolithic_fluidsplit_sp.hpp"
#include "4C_fsi_mortarmonolithic_structuresplit.hpp"
#include "4C_fsi_resulttest.hpp"
#include "4C_fsi_slidingmonolithic_fluidsplit.hpp"
#include "4C_fsi_slidingmonolithic_structuresplit.hpp"
#include "4C_fsi_utils.hpp"
#include "4C_fsi_xfem_fluid.hpp"
#include "4C_fsi_xfem_monolithic.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_poroelast_utils_clonestrategy.hpp"
#include "4C_poroelast_utils_setup.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_utils_result_test.hpp"
#include "4C_xfem_discretization.hpp"

#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <set>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
// entry point for Fluid on Ale in discretization management
/*----------------------------------------------------------------------*/
void fluid_ale_drt()
{
  Global::Problem* problem = Global::Problem::instance();

  MPI_Comm comm = problem->get_dis("fluid")->get_comm();

  // make sure the three discretizations are filled in the right order
  // this creates dof numbers with
  //
  //       fluid dof < ale dof
  //
  // We rely on this ordering in certain non-intuitive places!

  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis("fluid");
  // check for xfem discretization
  if (problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
  {
    FLD::XFluid::setup_fluid_discretization();
  }
  else
  {
    fluiddis->fill_complete();
  }

  std::shared_ptr<Core::FE::Discretization> aledis = problem->get_dis("ale");
  aledis->fill_complete();

  // create ale elements if the ale discretization is empty
  if (aledis->num_global_nodes() == 0)
  {
    Core::FE::clone_discretization<ALE::Utils::AleCloneStrategy>(
        *fluiddis, *aledis, Global::Problem::instance()->cloning_material_map());
    aledis->fill_complete();
    // setup material in every ALE element
    Teuchos::ParameterList params;
    params.set<std::string>("action", "setup_material");
    aledis->evaluate(params);
  }
  else  // filled ale discretization
  {
    if (!FSI::Utils::fluid_ale_nodes_disjoint(*fluiddis, *aledis))
      FOUR_C_THROW(
          "Fluid and ALE nodes have the same node numbers. "
          "This it not allowed since it causes problems with Dirichlet BCs. "
          "Use either the ALE cloning functionality or ensure non-overlapping node numbering!");
  }

  std::shared_ptr<FSI::FluidAleAlgorithm> fluid = std::make_shared<FSI::FluidAleAlgorithm>(comm);
  const int restart = problem->restart();
  if (restart)
  {
    // read the restart information, set vectors and variables
    fluid->read_restart(restart);
  }
  fluid->timeloop();

  Global::Problem::instance()->add_field_test(fluid->mb_fluid_field()->create_field_test());
  Global::Problem::instance()->test_all(comm);
}

/*----------------------------------------------------------------------*/
// entry point for Fluid on XFEM in discretization management
/*----------------------------------------------------------------------*/
void fluid_xfem_drt()
{
  MPI_Comm comm = Global::Problem::instance()->get_dis("structure")->get_comm();

  Global::Problem* problem = Global::Problem::instance();

  std::shared_ptr<Core::FE::Discretization> soliddis = problem->get_dis("structure");
  soliddis->fill_complete();

  FLD::XFluid::setup_fluid_discretization();

  const Teuchos::ParameterList xfluid = problem->x_fluid_dynamic_params();
  bool alefluid = xfluid.sublist("GENERAL").get<bool>("ALE_XFluid");

  if (alefluid)  // in ale case
  {
    std::shared_ptr<Core::FE::Discretization> aledis = problem->get_dis("ale");
    aledis->fill_complete();

    // create ale elements if the ale discretization is empty
    if (aledis->num_global_nodes() == 0)
    {
      Core::FE::clone_discretization<ALE::Utils::AleCloneStrategy>(
          *problem->get_dis("fluid"), *aledis, Global::Problem::instance()->cloning_material_map());
      aledis->fill_complete();
      // setup material in every ALE element
      Teuchos::ParameterList params;
      params.set<std::string>("action", "setup_material");
      aledis->evaluate(params);
    }
    else  // filled ale discretization
    {
      if (!FSI::Utils::fluid_ale_nodes_disjoint(*problem->get_dis("fluid"), *aledis))
        FOUR_C_THROW(
            "Fluid and ALE nodes have the same node numbers. "
            "This it not allowed since it causes problems with Dirichlet BCs. "
            "Use either the ALE cloning functionality or ensure non-overlapping node numbering!");
    }
  }

  if (alefluid)
  {
    // create instance of fluid xfem algorithm, for moving interfaces
    FSI::FluidXFEMAlgorithm fluidalgo(comm);

    const int restart = Global::Problem::instance()->restart();
    if (restart)
    {
      // read the restart information, set vectors and variables
      fluidalgo.read_restart(restart);
    }

    // run the simulation
    fluidalgo.timeloop();

    // perform result tests if required
    problem->add_field_test(fluidalgo.mb_fluid_field()->create_field_test());
    problem->test_all(comm);
  }
  else
  {
    //--------------------------------------------------------------
    // create instance of fluid basis algorithm
    const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();

    Adapter::FluidBaseAlgorithm fluidalgo(fdyn, fdyn, "fluid", false);

    //--------------------------------------------------------------
    // restart the simulation
    const int restart = Global::Problem::instance()->restart();
    if (restart)
    {
      // read the restart information, set vectors and variables
      fluidalgo.fluid_field()->read_restart(restart);
    }

    //--------------------------------------------------------------
    // run the simulation
    fluidalgo.fluid_field()->integrate();

    //--------------------------------------------------------------
    // perform result tests if required
    problem->add_field_test(fluidalgo.fluid_field()->create_field_test());
    problem->test_all(comm);
  }
}

/*----------------------------------------------------------------------*/
// entry point for FSI using multidimensional immersed method (FBI)
/*----------------------------------------------------------------------*/
void fsi_immersed_drt()
{
  Global::Problem* problem = Global::Problem::instance();

  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");
  MPI_Comm comm = structdis->get_comm();

  auto correct_node = [](const Core::Nodes::Node& node) -> decltype(auto)
  {
    const Core::Elements::Element* element = node.adjacent_elements()[0].user_element();
    const auto* beamelement = dynamic_cast<const Discret::Elements::Beam3Base*>(element);
    if (beamelement != nullptr && !beamelement->is_centerline_node(node))
      return *element->nodes()[0];
    else
      return node;
  };

  auto determine_relevant_points = [correct_node](const Core::FE::Discretization& discret,
                                       const Core::Elements::Element& ele,
                                       std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)
      -> std::vector<std::array<double, 3>>
  {
    if (dynamic_cast<const Discret::Elements::Beam3Base*>(&ele))
    {
      return Core::Binstrategy::DefaultRelevantPoints{
          .correct_node = correct_node,
      }(discret, ele, disnp);
    }
    else
      return Core::Binstrategy::DefaultRelevantPoints{}(discret, ele, disnp);
  };

  if (structdis->has_condition("PointCoupling"))
  {
    structdis->fill_complete(Core::FE::OptionsFillComplete::none());
    Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
        binning_params);
    Core::Rebalance::rebalance_discretizations_by_binning(binning_params,
        Global::Problem::instance()->output_control_file(), {structdis}, correct_node,
        determine_relevant_points, true);
  }
  else if (not structdis->filled() || not structdis->have_dofs())
  {
    structdis->fill_complete();
  }

  problem->get_dis("fluid")->fill_complete();

  // get discretizations
  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis("fluid");

  // create vector of discr.
  std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
  dis.push_back(fluiddis);
  dis.push_back(structdis);

  // binning strategy is created
  Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
  Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
      "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
      binning_params);


  auto binningstrategy = std::make_shared<Core::Binstrategy::BinningStrategy>(binning_params,
      Global::Problem::instance()->output_control_file(), comm,
      Core::Communication::my_mpi_rank(comm), correct_node, determine_relevant_points, dis);

  const Teuchos::ParameterList& fbidyn = problem->fbi_params();

  FBI::BeamToFluidPreSortStrategy presort_strategy =
      Teuchos::getIntegralValue<FBI::BeamToFluidPreSortStrategy>(fbidyn, "PRESORT_STRATEGY");

  // redistribute discr. with help of binning strategy
  if (presort_strategy == FBI::BeamToFluidPreSortStrategy::binning)
  {
    std::vector<std::shared_ptr<Core::LinAlg::Map>> stdelecolmap;
    std::vector<std::shared_ptr<Core::LinAlg::Map>> stdnodecolmap;
    std::shared_ptr<Core::LinAlg::Map> rowbins =
        binningstrategy
            ->do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
                dis, stdelecolmap, stdnodecolmap);
    binningstrategy->fill_bins_into_bin_discretization(*rowbins);
    binningstrategy->fill_bins_into_bin_discretization(*rowbins);
  }



  const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();

  // Any partitioned algorithm.
  std::shared_ptr<FSI::Partitioned> fsi;

  auto method = Teuchos::getIntegralValue<Inpar::FSI::PartitionedCouplingMethod>(
      fsidyn.sublist("PARTITIONED SOLVER"), "PARTITIONED");
  if (method == Inpar::FSI::DirichletNeumann)
  {
    fsi = FSI::DirichletNeumannFactory::create_algorithm(comm, fsidyn);
    std::dynamic_pointer_cast<FSI::DirichletNeumann>(fsi)->setup();
  }
  else
    FOUR_C_THROW("unsupported partitioned FSI scheme");

  if (presort_strategy == FBI::BeamToFluidPreSortStrategy::binning)
  {
    std::dynamic_pointer_cast<FSI::DirichletNeumannVel>(fsi)->set_binning(binningstrategy);
  }

  const int restart = Global::Problem::instance()->restart();
  if (restart)
  {
    // read the restart information, set vectors and variables
    fsi->read_restart(restart);
  }
  else
  {
    fsi->post_setup();
  }

  fsi->timeloop(fsi);

  // create result tests for single fields
  Global::Problem::instance()->add_field_test(fsi->mb_fluid_field()->create_field_test());
  Global::Problem::instance()->add_field_test(fsi->structure_field()->create_field_test());

  // do the actual testing
  Global::Problem::instance()->test_all(comm);
}
/*----------------------------------------------------------------------*/
// entry point for FSI using ALE in discretization management
/*----------------------------------------------------------------------*/
void fsi_ale_drt()
{
  Global::Problem* problem = Global::Problem::instance();

  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");
  MPI_Comm comm = structdis->get_comm();

  // make sure the three discretizations are filled in the right order
  // this creates dof numbers with
  //
  //       structure dof < fluid dof < ale dof
  //
  // We rely on this ordering in certain non-intuitive places!
  auto correct_node = [](const Core::Nodes::Node& node) -> decltype(auto)
  {
    const Core::Elements::Element* element = node.adjacent_elements()[0].user_element();
    const auto* beamelement = dynamic_cast<const Discret::Elements::Beam3Base*>(element);
    if (beamelement != nullptr && !beamelement->is_centerline_node(node))
      return *element->nodes()[0];
    else
      return node;
  };

  auto determine_relevant_points = [correct_node](const Core::FE::Discretization& discret,
                                       const Core::Elements::Element& ele,
                                       std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)
      -> std::vector<std::array<double, 3>>
  {
    if (dynamic_cast<const Discret::Elements::Beam3Base*>(&ele))
    {
      return Core::Binstrategy::DefaultRelevantPoints{
          .correct_node = correct_node,
      }(discret, ele, disnp);
    }
    else
      return Core::Binstrategy::DefaultRelevantPoints{}(discret, ele, disnp);
  };

  if (structdis->has_condition("PointCoupling"))
  {
    structdis->fill_complete(Core::FE::OptionsFillComplete::none());
    Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
        binning_params);

    Core::Rebalance::rebalance_discretizations_by_binning(binning_params,
        Global::Problem::instance()->output_control_file(), {structdis}, correct_node,
        determine_relevant_points, true);
  }
  else if (not structdis->filled() || not structdis->have_dofs())
  {
    structdis->fill_complete();
  }

  if (problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
  {
    FLD::XFluid::setup_fluid_discretization();
  }
  else
    problem->get_dis("fluid")->fill_complete();

  problem->get_dis("ale")->fill_complete();

  // get discretizations
  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis("fluid");
  std::shared_ptr<Core::FE::Discretization> aledis = problem->get_dis("ale");

  // create ale elements if the ale discretization is empty
  if (aledis->num_global_nodes() == 0)  // empty ale discretization
  {
    Core::FE::clone_discretization<ALE::Utils::AleCloneStrategy>(
        *fluiddis, *aledis, Global::Problem::instance()->cloning_material_map());
    aledis->fill_complete();
    // setup material in every ALE element
    Teuchos::ParameterList params;
    params.set<std::string>("action", "setup_material");
    aledis->evaluate(params);
  }
  else  // filled ale discretization (i.e. read from input file)
  {
    if (!FSI::Utils::fluid_ale_nodes_disjoint(*fluiddis, *aledis))
      FOUR_C_THROW(
          "Fluid and ALE nodes have the same node numbers. "
          "This it not allowed since it causes problems with Dirichlet BCs. "
          "Use either the ALE cloning functionality or ensure non-overlapping node numbering!");

    if ((not problem->fsi_dynamic_params().get<bool>("MATCHGRID_FLUIDALE")) or
        (not problem->fsi_dynamic_params().get<bool>("MATCHGRID_STRUCTALE")))
    {
      // create vector of discr.
      std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
      dis.push_back(structdis);
      dis.push_back(fluiddis);
      dis.push_back(aledis);

      std::vector<std::shared_ptr<Core::LinAlg::Map>> stdelecolmap;
      std::vector<std::shared_ptr<Core::LinAlg::Map>> stdnodecolmap;

      // redistribute discr. with help of binning strategy
      if (Core::Communication::num_mpi_ranks(structdis->get_comm()) > 1)
      {
        // binning strategy is created and parallel redistribution is performed
        Teuchos::ParameterList binning_params =
            Global::Problem::instance()->binning_strategy_params();
        Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
            "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
            binning_params);
        Core::Binstrategy::BinningStrategy binningstrategy(binning_params,
            Global::Problem::instance()->output_control_file(), comm,
            Core::Communication::my_mpi_rank(comm), correct_node, determine_relevant_points, dis);
        binningstrategy
            .do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
                dis, stdelecolmap, stdnodecolmap);
      }
    }
  }

  const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();

  auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");
  switch (coupling)
  {
    case fsi_iter_monolithicfluidsplit:
    case fsi_iter_monolithicstructuresplit:
    case fsi_iter_mortar_monolithicstructuresplit:
    case fsi_iter_mortar_monolithicfluidsplit:
    case fsi_iter_mortar_monolithicfluidsplit_saddlepoint:
    case fsi_iter_fluidfluid_monolithicfluidsplit:
    case fsi_iter_fluidfluid_monolithicstructuresplit:
    case fsi_iter_sliding_monolithicfluidsplit:
    case fsi_iter_sliding_monolithicstructuresplit:
    {
      // monolithic solver settings
      const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");

      std::shared_ptr<FSI::Monolithic> fsi;

      auto linearsolverstrategy =
          Teuchos::getIntegralValue<Inpar::FSI::LinearBlockSolver>(fsimono, "LINEARBLOCKSOLVER");

      // call constructor to initialize the base class
      if (coupling == fsi_iter_monolithicfluidsplit)
      {
        fsi = std::make_shared<FSI::MonolithicFluidSplit>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_monolithicstructuresplit)
      {
        fsi = std::make_shared<FSI::MonolithicStructureSplit>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_mortar_monolithicstructuresplit)
      {
        fsi = std::make_shared<FSI::MortarMonolithicStructureSplit>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_mortar_monolithicfluidsplit)
      {
        fsi = std::make_shared<FSI::MortarMonolithicFluidSplit>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_mortar_monolithicfluidsplit_saddlepoint)
      {
        fsi = std::make_shared<FSI::MortarMonolithicFluidSplitSaddlePoint>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_fluidfluid_monolithicfluidsplit)
      {
        fsi = std::make_shared<FSI::FluidFluidMonolithicFluidSplit>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_fluidfluid_monolithicstructuresplit)
      {
        fsi = std::make_shared<FSI::FluidFluidMonolithicStructureSplit>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_sliding_monolithicfluidsplit)
      {
        fsi = std::make_shared<FSI::SlidingMonolithicFluidSplit>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_sliding_monolithicstructuresplit)
      {
        fsi = std::make_shared<FSI::SlidingMonolithicStructureSplit>(comm, fsidyn);
      }
      else
      {
        FOUR_C_THROW(
            "Cannot find appropriate monolithic solver for coupling {} and linear strategy {}",
            coupling, linearsolverstrategy);
      }

      // read the restart information, set vectors and variables ---
      // be careful, dofmaps might be changed here in a redistribute() call
      const int restart = Global::Problem::instance()->restart();
      if (restart)
      {
        fsi->read_restart(restart);
      }
      else
      {
        fsi->post_setup();
      }

      // now do the coupling setup and create the combined dofmap
      fsi->setup_system();

      // here we go...
      fsi->timeloop(fsi);

      // calculate errors in comparison to analytical solution
      fsi->fluid_field()->calculate_error();

      // create result tests for single fields
      Global::Problem::instance()->add_field_test(fsi->ale_field()->create_field_test());
      Global::Problem::instance()->add_field_test(fsi->fluid_field()->create_field_test());
      Global::Problem::instance()->add_field_test(fsi->structure_field()->create_field_test());

      // create fsi specific result test
      std::shared_ptr<FSI::FSIResultTest> fsitest =
          std::make_shared<FSI::FSIResultTest>(fsi, fsidyn);
      Global::Problem::instance()->add_field_test(fsitest);

      // do the actual testing
      Global::Problem::instance()->test_all(comm);

      break;
    }
    case fsi_iter_fluidfluid_monolithicfluidsplit_nonox:
    case fsi_iter_fluidfluid_monolithicstructuresplit_nonox:
    {
      std::shared_ptr<FSI::MonolithicNoNOX> fsi;
      if (coupling == fsi_iter_fluidfluid_monolithicfluidsplit_nonox)
      {
        fsi = std::make_shared<FSI::FluidFluidMonolithicFluidSplitNoNOX>(comm, fsidyn);
      }
      else if (coupling == fsi_iter_fluidfluid_monolithicstructuresplit_nonox)
      {
        fsi = std::make_shared<FSI::FluidFluidMonolithicStructureSplitNoNOX>(comm, fsidyn);
      }
      else
        FOUR_C_THROW("Unsupported monolithic XFFSI scheme");

      // read the restart information, set vectors and variables ---
      // be careful, dofmaps might be changed here in a redistribute() call
      const int restart = Global::Problem::instance()->restart();
      if (restart)
      {
        fsi->read_restart(restart);
      }
      else
      {
        fsi->post_setup();
      }

      // now do the coupling setup and create the combined dofmap
      fsi->setup_system();

      // here we go...
      fsi->timeloop();

      // calculate errors in comparison to analytical solution
      fsi->fluid_field()->calculate_error();

      // create result tests for single fields
      Global::Problem::instance()->add_field_test(fsi->fluid_field()->create_field_test());
      Global::Problem::instance()->add_field_test(fsi->structure_field()->create_field_test());

      // create fsi specific result test
      std::shared_ptr<FSI::FSIResultTest> fsitest =
          std::make_shared<FSI::FSIResultTest>(fsi, fsidyn);
      Global::Problem::instance()->add_field_test(fsitest);

      // do the actual testing
      Global::Problem::instance()->test_all(comm);

      break;
    }
    default:
    {
      // Any partitioned algorithm.

      std::shared_ptr<FSI::Partitioned> fsi;

      auto method = Teuchos::getIntegralValue<Inpar::FSI::PartitionedCouplingMethod>(
          fsidyn.sublist("PARTITIONED SOLVER"), "PARTITIONED");

      switch (method)
      {
        case Inpar::FSI::DirichletNeumann:
        case Inpar::FSI::DirichletNeumannSlideale:
        case Inpar::FSI::DirichletNeumannVolCoupl:
          fsi = FSI::DirichletNeumannFactory::create_algorithm(comm, fsidyn);
          std::dynamic_pointer_cast<FSI::DirichletNeumann>(fsi)->setup();
          break;
        default:
          FOUR_C_THROW("unsupported partitioned FSI scheme");
          break;
      }
      const int restart = Global::Problem::instance()->restart();
      if (restart)
      {
        // read the restart information, set vectors and variables
        fsi->read_restart(restart);
      }
      else
      {
        // call post setup tasks
        fsi->post_setup();
      }

      fsi->timeloop(fsi);

      // create result tests for single fields
      Global::Problem::instance()->add_field_test(fsi->mb_fluid_field()->create_field_test());
      Global::Problem::instance()->add_field_test(fsi->structure_field()->create_field_test());

      // do the actual testing
      Global::Problem::instance()->test_all(comm);

      break;
    }
  }
}

/*----------------------------------------------------------------------*/
// entry point for FSI using XFEM in discretization management (also for ale case)
/*----------------------------------------------------------------------*/
void xfsi_drt()
{
  MPI_Comm comm = Global::Problem::instance()->get_dis("structure")->get_comm();

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << std::endl;
    std::cout << "       @..@    " << std::endl;
    std::cout << "      (----)      " << std::endl;
    std::cout << "     ( >__< )   " << std::endl;
    std::cout << "     ^^ ~~ ^^  " << std::endl;
    std::cout << "     _     _ _______ _______ _____" << std::endl;
    std::cout << "      \\\\__/  |______ |______   |  " << std::endl;
    std::cout << "     _/  \\\\_ |       ______| __|__" << std::endl;
    std::cout << std::endl << std::endl;
  }

  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();

  std::shared_ptr<Core::FE::Discretization> soliddis = problem->get_dis("structure");
  soliddis->fill_complete();

  FLD::XFluid::setup_fluid_discretization();

  std::shared_ptr<Core::FE::Discretization> fluiddis = Global::Problem::instance()->get_dis(
      "fluid");  // at the moment, 'fluid'-discretization is used for ale!!!

  // CREATE ALE
  const Teuchos::ParameterList& xfdyn = problem->x_fluid_dynamic_params();
  bool ale = xfdyn.sublist("GENERAL").get<bool>("ALE_XFluid");
  std::shared_ptr<Core::FE::Discretization> aledis;
  if (ale)
  {
    aledis = problem->get_dis("ale");
    if (aledis == nullptr) FOUR_C_THROW("XFSI DYNAMIC: ALE discretization empty!!!");

    aledis->fill_complete();

    // Create ALE elements if the ale discretization is empty
    if (aledis->num_global_nodes() == 0)  // ALE discretization still empty
    {
      Core::FE::clone_discretization<ALE::Utils::AleCloneStrategy>(
          *fluiddis, *aledis, Global::Problem::instance()->cloning_material_map());
      aledis->fill_complete();
      // setup material in every ALE element
      Teuchos::ParameterList params;
      params.set<std::string>("action", "setup_material");
      aledis->evaluate(params);
    }
    else  // ALE discretization already filled
    {
      if (!FSI::Utils::fluid_ale_nodes_disjoint(*fluiddis, *aledis))
        FOUR_C_THROW(
            "Fluid and ALE nodes have the same node numbers. "
            "This it not allowed since it causes problems with Dirichlet BCs. "
            "Use the ALE cloning functionality or ensure non-overlapping node numbering!");
    }
  }

  const auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");
  switch (coupling)
  {
    case fsi_iter_xfem_monolithic:
    {
      // monolithic solver settings
      const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");

      auto linearsolverstrategy =
          Teuchos::getIntegralValue<Inpar::FSI::LinearBlockSolver>(fsimono, "LINEARBLOCKSOLVER");

      if (linearsolverstrategy != Inpar::FSI::PreconditionedKrylov)
        FOUR_C_THROW("Only Newton-Krylov scheme with XFEM fluid");

      // create the MonolithicXFEM object that does the whole work
      FSI::MonolithicXFEM fsi(comm, fsidyn);

      // read the restart information, set vectors and variables ---
      // be careful, dofmaps might be changed here in a redistribute() call
      const int restart = Global::Problem::instance()->restart();
      if (restart)
      {
        fsi.read_restart(restart);
      }

      // setup the system (block-DOF-row maps, systemmatrix etc.) for the monolithic XFEM system
      fsi.setup_system();

      // here we go...
      fsi.timeloop();

      Global::Problem::instance()->add_field_test(fsi.fluid_field()->create_field_test());
      fsi.structure_poro()->test_results(Global::Problem::instance());

      //    // create FSI specific result test
      //    std::shared_ptr<FSI::FSIResultTest> fsitest = Teuchos::rcp(new
      //    FSI::FSIResultTest(fsi,fsidyn)); Global::Problem::instance()->AddFieldTest(fsitest);

      // do the actual testing
      Global::Problem::instance()->test_all(comm);

      break;
    }
    case fsi_iter_monolithicfluidsplit:
    case fsi_iter_monolithicstructuresplit:
      FOUR_C_THROW("Unreasonable choice");
      break;
    default:
    {
      // Any partitioned algorithm. Stable of working horses.

      std::shared_ptr<FSI::Partitioned> fsi;

      auto method = Teuchos::getIntegralValue<Inpar::FSI::PartitionedCouplingMethod>(
          fsidyn.sublist("PARTITIONED SOLVER"), "PARTITIONED");

      switch (method)
      {
        case Inpar::FSI::DirichletNeumann:
          fsi = FSI::DirichletNeumannFactory::create_algorithm(comm, fsidyn);
          std::dynamic_pointer_cast<FSI::DirichletNeumann>(fsi)->setup();
          break;
        default:
          FOUR_C_THROW("only Dirichlet-Neumann partitioned schemes with XFEM");
          break;
      }

      const int restart = Global::Problem::instance()->restart();
      if (restart)
      {
        // read the restart information, set vectors and variables
        fsi->read_restart(restart);
      }
      else
      {
        // call post setup tasks
        fsi->post_setup();
      }

      fsi->timeloop(fsi);

      Global::Problem::instance()->add_field_test(fsi->mb_fluid_field()->create_field_test());
      Global::Problem::instance()->add_field_test(fsi->structure_field()->create_field_test());
      Global::Problem::instance()->test_all(comm);

      break;
    }
  }
}

/*----------------------------------------------------------------------*/
// entry point for FPSI using XFEM in discretization management
/*----------------------------------------------------------------------*/
void xfpsi_drt()
{
  MPI_Comm comm = Global::Problem::instance()->get_dis("structure")->get_comm();

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << std::endl;
    std::cout << "       @..@    " << std::endl;
    std::cout << "      (----)      " << std::endl;
    std::cout << "     ( >__< )   " << std::endl;
    std::cout << "     ^^ ~~ ^^  " << std::endl;
    std::cout << "     _     _ _______  ______  _______ _____" << std::endl;
    std::cout << "      \\\\__/  |______ ||____|  |______   |  " << std::endl;
    std::cout << "     _/  \\\\_ |       ||       ______| __|__" << std::endl;
    std::cout << std::endl << std::endl;
  }
  Global::Problem* problem = Global::Problem::instance();

  // 1.-Initialization.
  // setup of the discretizations, including clone strategy
  PoroElast::Utils::setup_poro<PoroElast::Utils::PoroelastCloneStrategy>();

  // setup of discretization for xfluid
  FLD::XFluid::setup_fluid_discretization();
  std::shared_ptr<Core::FE::Discretization> fluiddis = Global::Problem::instance()->get_dis(
      "fluid");  // at the moment, 'fluid'-discretization is used for ale!!!

  std::shared_ptr<Core::FE::Discretization> aledis;
  const Teuchos::ParameterList& xfdyn = problem->x_fluid_dynamic_params();
  bool ale = xfdyn.sublist("GENERAL").get<bool>("ALE_XFluid");
  if (ale)
  {
    aledis = problem->get_dis("ale");
    if (aledis == nullptr) FOUR_C_THROW("Ale discretization empty!");

    aledis->fill_complete();

    // 3.- Create ALE elements if the ale discretization is empty
    if (aledis->num_global_nodes() == 0)  // ALE discretization still empty
    {
      Core::FE::clone_discretization<ALE::Utils::AleCloneStrategy>(
          *fluiddis, *aledis, Global::Problem::instance()->cloning_material_map());
      aledis->fill_complete();
      // setup material in every ALE element
      Teuchos::ParameterList params;
      params.set<std::string>("action", "setup_material");
      aledis->evaluate(params);
    }
    else  // ALE discretization already filled
    {
      if (!FSI::Utils::fluid_ale_nodes_disjoint(*fluiddis, *aledis))
        FOUR_C_THROW(
            "Fluid and ALE nodes have the same node numbers. "
            "This it not allowed since it causes problems with Dirichlet BCs. "
            "Use the ALE cloning functionality or ensure non-overlapping node numbering!");
    }
  }

  // print all dofsets
  fluiddis->get_dof_set_proxy()->print_all_dofsets(fluiddis->get_comm());

  // 2.- Parameter reading
  const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
  const auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");
  switch (coupling)
  {
    case fsi_iter_xfem_monolithic:
    {
      // monolithic solver settings
      const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
      auto linearsolverstrategy =
          Teuchos::getIntegralValue<Inpar::FSI::LinearBlockSolver>(fsimono, "LINEARBLOCKSOLVER");

      if (linearsolverstrategy != Inpar::FSI::PreconditionedKrylov)
        FOUR_C_THROW("Only Newton-Krylov scheme with XFEM fluid");

      FSI::MonolithicXFEM fsi(comm, fsidyn, Adapter::FieldWrapper::type_PoroField);

      // read the restart information, set vectors and variables ---

      // be careful, dofmaps might be changed here in a redistribute() call
      const int restart = Global::Problem::instance()
                              ->restart();  // not adapted at the moment .... Todo check it .. ChrAg
      if (restart)
      {
        fsi.read_restart(restart);
      }

      fsi.setup_system();


      // 3.2.- redistribute the FPSI interface
      // Todo .... fsi->redistribute_interface(); // this is required for parallels fpi-condition
      // (not included in this commit)

      // here we go...
      fsi.timeloop();

      Global::Problem::instance()->add_field_test(fsi.fluid_field()->create_field_test());
      fsi.structure_poro()->test_results(Global::Problem::instance());

      // do the actual testing
      Global::Problem::instance()->test_all(comm);
      break;
    }
    case fsi_iter_monolithicfluidsplit:
    case fsi_iter_monolithicstructuresplit:
      FOUR_C_THROW("Unreasonable choice");
      break;
    default:
    {
      FOUR_C_THROW("FPSI_XFEM: No Partitioned Algorithms implemented !!!");
      break;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
