// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_tsi_algorithm.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_contact_lagrange_strategy.hpp"
#include "4C_contact_lagrange_strategy_tsi.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_contact_strategy_factory.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_coupling_volmortar_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_mortar_multifield_coupling.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_structure_new_model_evaluator_structure.hpp"
#include "4C_thermo_adapter.hpp"
#include "4C_tsi_input.hpp"
#include "4C_tsi_utils.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

//! Note: The order of calling the two BaseAlgorithm-constructors is
//! important here! In here control file entries are written. And these entries
//! define the order in which the filters handle the Discretizations, which in
//! turn defines the dof number ordering of the Discretizations.
/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 12/09 |
 *----------------------------------------------------------------------*/
TSI::Algorithm::Algorithm(MPI_Comm comm)
    : AlgorithmBase(comm, Global::Problem::instance()->tsi_dynamic_params()),
      dispnp_(nullptr),
      tempnp_(nullptr),
      matchinggrid_(Global::Problem::instance()->tsi_dynamic_params().get<bool>("MATCHINGGRID")),
      volcoupl_(nullptr)
{
  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structdis =
      Global::Problem::instance()->get_dis("structure");
  // access the thermo discretization
  std::shared_ptr<Core::FE::Discretization> thermodis =
      Global::Problem::instance()->get_dis("thermo");

  // get the problem instance
  Global::Problem* problem = Global::Problem::instance();
  // get the restart step
  const int restart = problem->restart();

  if (!matchinggrid_)
  {
    // Scheme: non matching meshes --> volumetric mortar coupling...
    volcoupl_ = std::make_shared<Coupling::Adapter::MortarVolCoupl>();

    std::shared_ptr<Coupling::VolMortar::Utils::DefaultMaterialStrategy> materialstrategy =
        std::make_shared<TSI::Utils::TSIMaterialStrategy>();
    // init coupling adapter projection matrices
    volcoupl_->init(Global::Problem::instance()->n_dim(), structdis, thermodis, nullptr, nullptr,
        nullptr, nullptr, materialstrategy);
    // redistribute discretizations to meet needs of volmortar coupling
    Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
    Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
        "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
        binning_params);
    volcoupl_->redistribute(binning_params, Global::Problem::instance()->output_control_file());
    // setup projection matrices
    volcoupl_->setup(Global::Problem::instance()->volmortar_params(),
        Global::Problem::instance()->cut_general_params());
  }

  if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(
          Global::Problem::instance()->structural_dynamic_params(), "INT_STRATEGY") ==
      Inpar::Solid::int_old)
    FOUR_C_THROW("old structural time integration no longer supported in tsi");
  else
  {
    Thermo::BaseAlgorithm thermo(Global::Problem::instance()->tsi_dynamic_params(), thermodis);
    thermo_ = thermo.thermo_field();

    //  // access structural dynamic params list which will be possibly modified while creating the
    //  time integrator
    const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
    std::shared_ptr<Adapter::StructureBaseAlgorithmNew> adapterbase_ptr =
        Adapter::build_structure_algorithm(sdyn);
    adapterbase_ptr->init(Global::Problem::instance()->tsi_dynamic_params(),
        const_cast<Teuchos::ParameterList&>(sdyn), structdis);

    // set the temperature; Monolithic does this in it's own constructor with potentially
    // redistributed discretizations
    if (Teuchos::getIntegralValue<TSI::SolutionSchemeOverFields>(
            Global::Problem::instance()->tsi_dynamic_params(), "COUPALGO") !=
        TSI::SolutionSchemeOverFields::Monolithic)
    {
      if (matchinggrid_)
        structdis->set_state(1, "temperature", *thermo_field()->tempnp());
      else
        structdis->set_state(
            1, "temperature", *volcoupl_->apply_vector_mapping12(*thermo_field()->tempnp()));
    }

    adapterbase_ptr->setup();
    structure_ =
        std::dynamic_pointer_cast<Adapter::StructureWrapper>(adapterbase_ptr->structure_field());

    if (restart && Teuchos::getIntegralValue<TSI::SolutionSchemeOverFields>(
                       Global::Problem::instance()->tsi_dynamic_params(), "COUPALGO") ==
                       TSI::SolutionSchemeOverFields::Monolithic)
      structure_->setup();

    structure_field()->discretization()->clear_state(true);
  }

  // initialise displacement field needed for output()
  // (get noderowmap of discretisation for creating this multivector)
  // TODO: why nds 0 and not 1????
  dispnp_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(thermo_field()->discretization()->node_row_map()), 3, true);
  tempnp_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(structure_field()->discretization()->node_row_map()), 1, true);

  // setup coupling object for matching discretization
  if (matchinggrid_)
  {
    coupST_ = std::make_shared<Coupling::Adapter::Coupling>();
    coupST_->setup_coupling(*structure_field()->discretization(), *thermo_field()->discretization(),
        *structure_field()->discretization()->node_row_map(),
        *thermo_field()->discretization()->node_row_map(), 1, true);
  }

  // setup mortar coupling
  if (Global::Problem::instance()->get_problem_type() == Core::ProblemType::tsi)
  {
    if (structure_field()->discretization()->has_condition("MortarMulti"))
    {
      mortar_coupling_ = std::make_shared<Mortar::MultiFieldCoupling>();
      mortar_coupling_->push_back_coupling(structure_field()->discretization(), 0,
          std::vector<int>(3, 1), Global::Problem::instance()->mortar_coupling_params(),
          Global::Problem::instance()->contact_dynamic_params(),
          Global::Problem::instance()->binning_strategy_params(),
          {{"structure", Global::Problem::instance()->get_dis("structure")},
              {"thermo", Global::Problem::instance()->get_dis("thermo")}},
          Global::Problem::instance()->function_manager(),
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type(),
          Global::Problem::instance()->n_dim());
      mortar_coupling_->push_back_coupling(thermo_field()->discretization(), 0,
          std::vector<int>(1, 1), Global::Problem::instance()->mortar_coupling_params(),
          Global::Problem::instance()->contact_dynamic_params(),
          Global::Problem::instance()->binning_strategy_params(),
          {{"structure", Global::Problem::instance()->get_dis("structure")},
              {"thermo", Global::Problem::instance()->get_dis("thermo")}},
          Global::Problem::instance()->function_manager(),
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type(),
          Global::Problem::instance()->n_dim());
    }
  }

  // reset states
  structure_field()->discretization()->clear_state(true);
  thermo_field()->discretization()->clear_state(true);

  return;
}



/*----------------------------------------------------------------------*
 | update (protected)                                        dano 12/09 |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::update()
{
  apply_thermo_coupling_state(thermo_field()->tempnp());
  structure_field()->update();
  thermo_field()->update();
  if (contact_strategy_lagrange_ != nullptr)
    contact_strategy_lagrange_->update((structure_field()->dispnp()));
  return;
}


/*----------------------------------------------------------------------*
 | output (protected)                                        dano 12/09 |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::output(bool forced_writerestart)
{
  // Note: The order in the output is important here!

  // In here control file entries are written. And these entries define the
  // order in which the filters handle the Discretizations, which in turn
  // defines the dof number ordering of the Discretizations.

  // call the TSI parameter list
  const Teuchos::ParameterList& tsidyn = Global::Problem::instance()->tsi_dynamic_params();
  // Get the parameters for the Newton iteration
  int upres = tsidyn.get<int>("RESULTSEVERY");
  int uprestart = tsidyn.get<int>("RESTARTEVERY");

  //========================
  // output for thermofield:
  //========================
  apply_struct_coupling_state(structure_field()->dispnp(), structure_field()->velnp());
  thermo_field()->output(forced_writerestart);

  // communicate the deformation to the thermal field,
  // current displacements are contained in Dispn()
  if (forced_writerestart == true and
      ((upres != 0 and (step() % upres == 0)) or ((uprestart != 0) and (step() % uprestart == 0))))
  {
    // displacement has already been written into thermo field for this step
  }
  else if ((upres != 0 and (step() % upres == 0)) or
           ((uprestart != 0) and (step() % uprestart == 0)) or forced_writerestart == true)
  {
    if (matchinggrid_)
    {
      output_deformation_in_thermo(
          structure_field()->dispn(), *structure_field()->discretization());

      thermo_field()->disc_writer()->write_multi_vector(
          "displacement", *dispnp_, Core::IO::nodevector);
    }
    else
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dummy =
          volcoupl_->apply_vector_mapping21(*structure_field()->dispnp());

      // determine number of space dimensions
      const int numdim = Global::Problem::instance()->n_dim();

      // loop over all local nodes of thermal discretisation
      for (int lnodeid = 0; lnodeid < (thermo_field()->discretization()->num_my_row_nodes());
          lnodeid++)
      {
        Core::Nodes::Node* thermnode = thermo_field()->discretization()->l_row_node(lnodeid);
        std::vector<int> thermnodedofs_1 = thermo_field()->discretization()->dof(1, thermnode);

        // now we transfer displacement dofs only
        for (int index = 0; index < numdim; ++index)
        {
          // global and processor's local fluid dof ID
          const int sgid = thermnodedofs_1[index];
          const int slid = thermo_field()->discretization()->dof_row_map(1)->lid(sgid);


          // get value of corresponding displacement component
          double disp = (*dummy)[slid];
          // insert velocity value into node-based vector
          dispnp_->replace_local_value(lnodeid, index, disp);
        }

        // for security reasons in 1D or 2D problems:
        // set zeros for all unused velocity components
        for (int index = numdim; index < 3; ++index)
        {
          dispnp_->replace_local_value(lnodeid, index, 0.0);
        }
      }  // for lnodid

      thermo_field()->disc_writer()->write_multi_vector(
          "displacement", *dispnp_, Core::IO::nodevector);
    }
  }


  //===========================
  // output for structurefield:
  //===========================
  apply_thermo_coupling_state(thermo_field()->tempnp());
  structure_field()->output(forced_writerestart);

  // mapped temperatures for structure field
  if ((upres != 0 and (step() % upres == 0)) or ((uprestart != 0) and (step() % uprestart == 0)) or
      forced_writerestart == true)
    if (not matchinggrid_)
    {
      //************************************************************************************
      std::shared_ptr<const Core::LinAlg::Vector<double>> dummy1 =
          volcoupl_->apply_vector_mapping12(*thermo_field()->tempnp());

      // loop over all local nodes of thermal discretisation
      for (int lnodeid = 0; lnodeid < (structure_field()->discretization()->num_my_row_nodes());
          lnodeid++)
      {
        Core::Nodes::Node* structnode = structure_field()->discretization()->l_row_node(lnodeid);
        std::vector<int> structdofs = structure_field()->discretization()->dof(1, structnode);

        // global and processor's local structure dof ID
        const int sgid = structdofs[0];
        const int slid = structure_field()->discretization()->dof_row_map(1)->lid(sgid);

        // get value of corresponding displacement component
        double temp = (*dummy1)[slid];
        // insert velocity value into node-based vector
        tempnp_->replace_local_value(lnodeid, 0, temp);
      }  // for lnodid

      structure_field()->discretization()->writer()->write_multi_vector(
          "struct_temperature", *tempnp_, Core::IO::nodevector);
    }


  // reset states
  structure_field()->discretization()->clear_state(true);
  thermo_field()->discretization()->clear_state(true);
}  // output()


/*----------------------------------------------------------------------*
 | communicate the displacement vector to Thermo field          dano 12/11 |
 | enable visualisation of thermal variables on deformed body           |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::output_deformation_in_thermo(
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp, Core::FE::Discretization& structdis)
{
  if (dispnp == nullptr) FOUR_C_THROW("Got null pointer for displacements");

  // get dofrowmap of structural discretisation
  const Core::LinAlg::Map* structdofrowmap = structdis.dof_row_map(0);

  // loop over all local nodes of thermal discretisation
  for (int lnodeid = 0; lnodeid < (thermo_field()->discretization()->num_my_row_nodes()); lnodeid++)
  {
    // Here we rely on the fact that the thermal discretisation is a clone of
    // the structural mesh.
    // => a thermal node has the same local (and global) ID as its corresponding
    // structural node!

    // get the processor's local structural node with the same lnodeid
    Core::Nodes::Node* structlnode = structdis.l_row_node(lnodeid);
    // get the degrees of freedom associated with this structural node
    std::vector<int> structnodedofs = structdis.dof(0, structlnode);
    // determine number of space dimensions
    const int numdim = Global::Problem::instance()->n_dim();

    // now we transfer displacement dofs only
    for (int index = 0; index < numdim; ++index)
    {
      // global and processor's local fluid dof ID
      const int sgid = structnodedofs[index];
      const int slid = structdofrowmap->lid(sgid);

      // get value of corresponding displacement component
      double disp = (*dispnp)[slid];
      // insert velocity value into node-based vector
      dispnp_->replace_local_value(lnodeid, index, disp);
    }

    // for security reasons in 1D or 2D problems:
    // set zeros for all unused velocity components
    for (int index = numdim; index < 3; ++index)
    {
      dispnp_->replace_local_value(lnodeid, index, 0.0);
    }

  }  // for lnodid

  return;

}  // output_deformation_in_thermo()


/*----------------------------------------------------------------------*
 | calculate velocities                                      dano 12/10 |
 | like interface_velocity(disp) in FSI::DirichletNeumann                |
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> TSI::Algorithm::calc_velocity(
    const Core::LinAlg::Vector<double>& dispnp)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vel = nullptr;
  // copy D_n onto V_n+1
  vel = std::make_shared<Core::LinAlg::Vector<double>>(*(structure_field()->dispn()));
  // calculate velocity with timestep Dt()
  //  V_n+1^k = (D_n+1^k - D_n) / Dt
  vel->update(1. / dt(), dispnp, -1. / dt());

  return vel;
}  // calc_velocity()


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::apply_thermo_coupling_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>> temp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> temp_res)
{
  if (matchinggrid_)
  {
    if (temp != nullptr) structure_field()->discretization()->set_state(1, "temperature", *temp);
    if (temp_res != nullptr)
      structure_field()->discretization()->set_state(1, "residual temperature", *temp_res);
  }
  else
  {
    if (temp != nullptr)
      structure_field()->discretization()->set_state(
          1, "temperature", *volcoupl_->apply_vector_mapping12(*temp));
  }

  // set new temperatures to contact
  {
    if (contact_strategy_lagrange_ != nullptr)
      contact_strategy_lagrange_->set_state(
          Mortar::state_temperature, *coupST_->slave_to_master(*thermo_field()->tempnp()));
  }
}  // apply_thermo_coupling_state()


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void TSI::Algorithm::apply_struct_coupling_state(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  if (matchinggrid_)
  {
    if (disp != nullptr) thermo_field()->discretization()->set_state(1, "displacement", *disp);
    if (vel != nullptr) thermo_field()->discretization()->set_state(1, "velocity", *vel);
  }
  else
  {
    if (disp != nullptr)
      thermo_field()->discretization()->set_state(
          1, "displacement", *volcoupl_->apply_vector_mapping21(*disp));
    if (vel != nullptr)
      thermo_field()->discretization()->set_state(
          1, "velocity", *volcoupl_->apply_vector_mapping21(*vel));
  }
}  // apply_struct_coupling_state()


/*----------------------------------------------------------------------*/
void TSI::Algorithm::prepare_contact_strategy()
{
  auto stype = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(
      Global::Problem::instance()->contact_dynamic_params(), "STRATEGY");

  if (stype == CONTACT::SolvingStrategy::lagmult)
  {
    if (structure_field()->have_model(Inpar::Solid::model_contact))
      FOUR_C_THROW(
          "structure should not have a Lagrange strategy ... as long as condensed"
          "contact formulations are not moved to the new structural time integration");

    std::vector<const Core::Conditions::Condition*> ccond;
    structure_field()->discretization()->get_condition("Contact", ccond);
    if (ccond.size() == 0) return;

    // ---------------------------------------------------------------------
    // create the contact factory
    // ---------------------------------------------------------------------
    CONTACT::STRATEGY::Factory factory;
    factory.init(structure_field()->discretization());
    factory.setup(Global::Problem::instance()->n_dim());

    // check the problem dimension
    factory.check_dimension();

    // create some local variables (later to be stored in strategy)
    std::vector<std::shared_ptr<CONTACT::Interface>> interfaces;
    Teuchos::ParameterList cparams;

    // read and check contact input parameters
    factory.read_and_check_input(cparams);

    // ---------------------------------------------------------------------
    // build the contact interfaces
    // ---------------------------------------------------------------------
    // FixMe Would be great, if we get rid of these poro parameters...
    bool poroslave = false;
    bool poromaster = false;
    factory.build_interfaces(cparams, interfaces, poroslave, poromaster);

    // ---------------------------------------------------------------------
    // build the solver strategy object
    // ---------------------------------------------------------------------
    contact_strategy_lagrange_ = std::dynamic_pointer_cast<CONTACT::LagrangeStrategyTsi>(
        factory.build_strategy(cparams, poroslave, poromaster, 1e8, interfaces));

    // build the search tree
    factory.build_search_tree(interfaces);

    // print final screen output
    factory.print(interfaces, *contact_strategy_lagrange_, cparams);

    // ---------------------------------------------------------------------
    // final touches to the contact strategy
    // ---------------------------------------------------------------------

    contact_strategy_lagrange_->store_dirichlet_status(structure_field()->get_dbc_map_extractor());

    std::shared_ptr<Core::LinAlg::Vector<double>> zero_disp =
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
    contact_strategy_lagrange_->set_state(Mortar::state_new_displacement, *zero_disp);
    contact_strategy_lagrange_->save_reference_state(zero_disp);
    contact_strategy_lagrange_->evaluate_reference_state();
    contact_strategy_lagrange_->inttime_init();
    contact_strategy_lagrange_->set_time_integration_info(structure_field()->tim_int_param(),
        Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(
            Global::Problem::instance()->structural_dynamic_params(), "DYNAMICTYPE"));
    contact_strategy_lagrange_->redistribute_contact(
        structure_field()->dispn(), structure_field()->veln());

    if (contact_strategy_lagrange_ != nullptr)
    {
      contact_strategy_lagrange_->set_alphaf_thermo(
          Global::Problem::instance()->thermal_dynamic_params());
      contact_strategy_lagrange_->set_coupling(coupST_);
    }
  }
}

void TSI::Algorithm::post_setup()
{
  // call the post_setup routine of the structure field
  structure_->post_setup();
}

FOUR_C_NAMESPACE_CLOSE
