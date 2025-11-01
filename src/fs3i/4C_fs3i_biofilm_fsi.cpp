// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fs3i_biofilm_fsi.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_utils_clonestrategy.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fem_geometry_update_reference_config.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fs3i_biofilm_fsi_utils.hpp"
#include "4C_fsi_monolithicfluidsplit.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FS3I::BiofilmFSI::BiofilmFSI(MPI_Comm comm) : PartFS3I1Wc(comm), comm_(comm)
{
  // has to stay empty
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::init()
{
  if (Core::Communication::my_mpi_rank(comm_) == 0)
    std::cout << "\n WARNING ! The implementation of BiofilmFSI is not well tested,\n"
                 " buggy, and introduction of just init(...) and setup() in commit\n"
                 " to revision 22366 led to differing results slightly above the\n"
                 " convergence tolerance. Rework on this problem type is necessary!\n\n"
              << std::endl;

  // call init() in base class
  FS3I::PartFS3I1Wc::init();

  //---------------------------------------------------------------------
  // set up struct ale
  //---------------------------------------------------------------------

  // this algorithm needs an ale discretization also for the structure in order to be able to handle
  // the growth
  Global::Problem* problem = Global::Problem::instance();
  problem->get_dis("structale")->fill_complete();

  // create struct ale elements if not yet existing
  std::shared_ptr<Core::FE::Discretization> structaledis = problem->get_dis("structale");
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");

  // time measurement
  Teuchos::Time time("biofilm_fsi_Init", true);

  if (structaledis->num_global_nodes() == 0)
  {
    Core::FE::DiscretizationCreator<ALE::Utils::AleCloneStrategy> alecreator;
    alecreator.create_matching_discretization(*structdis, *structaledis, 11);
    structaledis->fill_complete();
  }
  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    std::cout << "Created discretization " << (structaledis->name())
              << " as a clone of discretization " << (structdis->name()) << " in...."
              << time.totalElapsedTime(true) << " secs\n\n";
  }

  // ask base algorithm for the ale time integrator
  const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
  std::shared_ptr<Adapter::AleBaseAlgorithm> ale =
      std::make_shared<Adapter::AleBaseAlgorithm>(fsidyn, structaledis);
  ale_ = std::dynamic_pointer_cast<Adapter::AleFsiWrapper>(ale->ale_field());
  if (ale_ == nullptr) FOUR_C_THROW("cast from Adapter::Ale to Adapter::AleFsiWrapper failed");


  //---------------------------------------------------------------------
  // getting and initializing problem-specific parameters
  //---------------------------------------------------------------------

  const Teuchos::ParameterList& biofilmcontrol =
      Global::Problem::instance()->biofilm_control_params();

  // make sure that initial time derivative of concentration is not calculated
  // automatically (i.e. field-wise)
  const Teuchos::ParameterList& scatradyn =
      Global::Problem::instance()->scalar_transport_dynamic_params();
  if (not scatradyn.get<bool>("SKIPINITDER"))
    FOUR_C_THROW(
        "Initial time derivative of phi must not be calculated automatically -> set SKIPINITDER to "
        "false");

  // fsi parameters
  dt_fsi_ = fsidyn.get<double>("TIMESTEP");
  nstep_fsi_ = fsidyn.get<int>("NUMSTEP");
  maxtime_fsi_ = fsidyn.get<double>("MAXTIME");
  step_fsi_ = 0;
  time_fsi_ = 0.;

  // growth parameters
  dt_bio_ = biofilmcontrol.get<double>("BIOTIMESTEP");
  nstep_bio_ = biofilmcontrol.get<int>("BIONUMSTEP");
  fluxcoef_ = biofilmcontrol.get<double>("FLUXCOEF");
  normforceposcoef_ = biofilmcontrol.get<double>("NORMFORCEPOSCOEF");
  normforcenegcoef_ = biofilmcontrol.get<double>("NORMFORCENEGCOEF");
  tangoneforcecoef_ = biofilmcontrol.get<double>("TANGONEFORCECOEF");
  tangtwoforcecoef_ = biofilmcontrol.get<double>("TANGTWOFORCECOEF");
  step_bio_ = 0;
  time_bio_ = 0.;

  // total time
  time_ = 0.;

  // safety checks
  if (volume_fieldcouplings_[0] == Inpar::FS3I::coupling_nonmatch or
      volume_fieldcouplings_[1] == Inpar::FS3I::coupling_nonmatch)
    FOUR_C_THROW("Mortar volume coupling is yet not implemented for biofilm-fs3i.");
  if (!problem->get_dis("scatra1")->has_condition("ScaTraFluxCalc") or
      !problem->get_dis("scatra2")->has_condition("ScaTraFluxCalc"))
    FOUR_C_THROW(
        "Fluid-scatra and solid-scatra discretizations must have boundary conditions for flux "
        "calculation at FSI interface!");

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::setup()
{
  // call setup() in base class
  FS3I::PartFS3I1Wc::setup();

  std::shared_ptr<Core::FE::Discretization> structaledis =
      Global::Problem::instance()->get_dis("structale");

  // create fluid-ALE Dirichlet Map Extractor for FSI step
  ale_->setup_dbc_map_ex(ALE::Utils::MapExtractor::dbc_set_std);

  // create fluid-ALE Dirichlet Map Extractor for growth step
  ale_->setup_dbc_map_ex(ALE::Utils::MapExtractor::dbc_set_biofilm, ale_->interface());

  // create fluid-ALE Dirichlet Map Extractor for growth step
  fsi_->ale_field()->setup_dbc_map_ex(ALE::Utils::MapExtractor::dbc_set_std, nullptr);

  // create fluid-ALE Dirichlet Map Extractor for FSI step
  fsi_->ale_field()->setup_dbc_map_ex(
      ALE::Utils::MapExtractor::dbc_set_biofilm, fsi_->ale_field()->interface());

  //---------------------------------------------------------------------
  // set up couplings
  //---------------------------------------------------------------------

  const std::string condname = "FSICoupling";
  const int ndim = Global::Problem::instance()->n_dim();

  // set up ale-fluid couplings
  icoupfa_ = std::make_shared<Coupling::Adapter::Coupling>();
  icoupfa_->setup_condition_coupling(*(fsi_->fluid_field()->discretization()),
      (fsi_->fluid_field()->interface()->fsi_cond_map()), *(fsi_->ale_field()->discretization()),
      (fsi_->ale_field()->interface()->fsi_cond_map()), condname, ndim);
  // the fluid-ale coupling always matches
  const Core::LinAlg::Map* fluidnodemap = fsi_->fluid_field()->discretization()->node_row_map();
  const Core::LinAlg::Map* fluidalenodemap = fsi_->ale_field()->discretization()->node_row_map();
  coupfa_ = std::make_shared<Coupling::Adapter::Coupling>();
  coupfa_->setup_coupling(*(fsi_->fluid_field()->discretization()),
      *(fsi_->ale_field()->discretization()), *fluidnodemap, *fluidalenodemap, ndim);

  // set up structale-structure couplings
  icoupsa_ = std::make_shared<Coupling::Adapter::Coupling>();
  icoupsa_->setup_condition_coupling(*(fsi_->structure_field()->discretization()),
      fsi_->structure_field()->interface()->fsi_cond_map(), *structaledis,
      ale_->interface()->fsi_cond_map(), condname, ndim);
  // the structure-structale coupling always matches
  const Core::LinAlg::Map* structurenodemap =
      fsi_->structure_field()->discretization()->node_row_map();
  const Core::LinAlg::Map* structalenodemap = structaledis->node_row_map();
  coupsa_ = std::make_shared<Coupling::Adapter::Coupling>();
  coupsa_->setup_coupling(*(fsi_->structure_field()->discretization()), *structaledis,
      *structurenodemap, *structalenodemap, ndim);

  /// do we need this? What's for???
  fsi_->fluid_field()->set_mesh_map(coupfa_->master_dof_map());

  idispn_ = fsi_->fluid_field()->extract_interface_veln();
  idispnp_ = fsi_->fluid_field()->extract_interface_veln();
  iveln_ = fsi_->fluid_field()->extract_interface_veln();

  struidispn_ = fsi_->structure_field()->extract_interface_dispn();
  struidispnp_ = fsi_->structure_field()->extract_interface_dispn();
  struiveln_ = fsi_->structure_field()->extract_interface_dispn();

  struct_growth_disp_ = ale_to_struct_field(ale_->write_access_dispnp());
  fluid_growth_disp_ = ale_to_fluid_field(*fsi_->ale_field()->write_access_dispnp());
  scatra_struct_growth_disp_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(scatravec_[1]->scatra_field()->discretization())->node_row_map(), 3, true);
  scatra_fluid_growth_disp_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *(scatravec_[0]->scatra_field()->discretization())->node_row_map(), 3, true);

  idispn_->put_scalar(0.0);
  idispnp_->put_scalar(0.0);
  iveln_->put_scalar(0.0);

  struidispn_->put_scalar(0.0);
  struidispnp_->put_scalar(0.0);
  struiveln_->put_scalar(0.0);

  struct_growth_disp_->put_scalar(0.0);
  fluid_growth_disp_->put_scalar(0.0);
  scatra_struct_growth_disp_->put_scalar(0.0);
  scatra_fluid_growth_disp_->put_scalar(0.0);

  norminflux_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *(fsi_->structure_field()->discretization()->node_row_map()));
  normtraction_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *(fsi_->structure_field()->discretization()->node_row_map()));
  tangtractionone_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *(fsi_->structure_field()->discretization()->node_row_map()));
  tangtractiontwo_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *(fsi_->structure_field()->discretization()->node_row_map()));

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::timeloop()
{
  check_is_init();
  check_is_setup();


  const Teuchos::ParameterList& biofilmcontrol =
      Global::Problem::instance()->biofilm_control_params();
  const bool biofilmgrowth = biofilmcontrol.get<bool>("BIOFILMGROWTH");
  const bool outputgmsh_ = biofilmcontrol.get<bool>("OUTPUT_GMSH");

  std::cout << std::endl << "--------------SIMULATION PARAMETERS-----------------" << std::endl;
  std::cout << "FSI TIMESTEP = " << dt_fsi_ << "; FSI NUMSTEP = " << nstep_fsi_ << std::endl;
  std::cout << "BIO TIMESTEP = " << dt_bio_ << "; BIO NUMSTEP = " << nstep_bio_ << std::endl;
  std::cout << "FLUXCOEF = " << fluxcoef_ << ";" << std::endl;
  std::cout << "NORMFORCEPOSCOEF = " << normforceposcoef_
            << "; NORMFORCENEGCOEF = " << normforcenegcoef_ << std::endl;
  std::cout << "TANGONEFORCECOEF = " << tangoneforcecoef_
            << "; TANGTWOFORCECOEF = " << tangtwoforcecoef_ << std::endl;
  std::cout << "----------------------------------------------------" << std::endl;

  if (biofilmgrowth)
  {
    // outer loop for biofilm growth
    while (step_bio_ <= nstep_bio_)
    {
      // update step and time
      step_bio_++;
      time_bio_ += dt_bio_;
      time_ = time_bio_ + time_fsi_;

      if (step_bio_ == 1 && outputgmsh_)
      {
        struct_gmsh_output();
        fluid_gmsh_output();
      }

      // inner loop for fsi and scatra
      inner_timeloop();

      // gmsh output only if requested
      if (outputgmsh_)
      {
        struct_gmsh_output();
        fluid_gmsh_output();
      }

      if (Core::Communication::my_mpi_rank(comm()) == 0)
      {
        std::cout << "\n***********************\n     GROWTH STEP \n***********************\n";
        printf(" growth step = %3d   \n", step_bio_);
        printf(" Total time = %3f   \n", time_);
      }

      // compute interface displacement and velocity
      compute_interface_vectors(*idispnp_, *iveln_, struidispnp_, *struiveln_);

      // do all the settings and solve the fluid on a deforming mesh
      fluid_ale_solve();

      // do all the settings and solve the structure on a deforming mesh
      struct_ale_solve();

      fsi_->output();
      scatra_output();

      // reset step and state vectors
      fsi_->structure_field()->reset();
      // fluid reset can be bypassed, in this way the next step starts from a solution closer to the
      // final one fsi_->fluid_field()->reset(false, false, step_bio);
      fsi_->ale_field()->reset();

      fsi_->ale_field()->create_system_matrix(fsi_->ale_field()->interface());
    }
  }

  if (!biofilmgrowth)
  {
    inner_timeloop();

    // gmsh output only if requested
    if (outputgmsh_)
    {
      struct_gmsh_output();
      fluid_gmsh_output();
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::inner_timeloop()
{
  // initialize time and step each time we enter the innerloop
  double t = 0.;
  step_fsi_ = 0;
  // initialize fluxes and tractions each time we enter the innerloop
  norminflux_->put_scalar(0.0);
  normtraction_->put_scalar(0.0);
  tangtractionone_->put_scalar(0.0);
  tangtractiontwo_->put_scalar(0.0);

  // output of initial state
  //  ScatraOutput();

  fsi_->prepare_timeloop();

  // Calculation of growth can be based both on values averaged during the inner timeloop
  // (in this case for the time being it takes in account also the initial transient state!),
  // or only on the last values coming from the fsi-scatra simulation
  const Teuchos::ParameterList& biofilmcontrol =
      Global::Problem::instance()->biofilm_control_params();
  const bool avgrowth = biofilmcontrol.get<bool>("AVGROWTH");
  // in case of averaged values we need temporary variables
  Core::LinAlg::Vector<double> normtempinflux_(
      *(fsi_->structure_field()->discretization()->node_row_map()));
  Core::LinAlg::Vector<double> normtemptraction_(
      *(fsi_->structure_field()->discretization()->node_row_map()));
  Core::LinAlg::Vector<double> tangtemptractionone_(
      *(fsi_->structure_field()->discretization()->node_row_map()));
  Core::LinAlg::Vector<double> tangtemptractiontwo_(
      *(fsi_->structure_field()->discretization()->node_row_map()));
  normtempinflux_.put_scalar(0.0);
  normtemptraction_.put_scalar(0.0);
  tangtemptractionone_.put_scalar(0.0);
  tangtemptractiontwo_.put_scalar(0.0);

  while (step_fsi_ < nstep_fsi_ and t + 1e-10 * dt_fsi_ < maxtime_fsi_)
  {
    step_fsi_++;
    t += dt_fsi_;

    fsi_->prepare_time_step();
    fsi_->time_step(fsi_);

    constexpr bool force_prepare = false;
    fsi_->prepare_output(force_prepare);
    fsi_->update();

    set_fsi_solution();

    if (Core::Communication::my_mpi_rank(comm()) == 0)
    {
      std::cout << "\n***********************\n GAS TRANSPORT SOLVER \n***********************\n";
    }

    // first scatra field is associated with fluid, second scatra field is
    // associated with structure

    bool stopnonliniter = false;
    int itnum = 0;

    prepare_time_step();

    while (stopnonliniter == false)
    {
      scatra_evaluate_solve_iter_update();
      itnum++;
      if (scatra_convergence_check(itnum)) break;
    }

    // calculation of the flux at the interface based on normal influx values before time shift of
    // results is performed in Update
    std::shared_ptr<Core::LinAlg::MultiVector<double>> strufluxn =
        scatravec_[1]->scatra_field()->calc_flux_at_boundary(false);

    update_scatra_fields();

    // this is necessary because we want to write all the steps except the last one
    // the last one will be written only after the calculation of the growth
    // in this way also the displacement due to growth is written
    if (step_fsi_ < nstep_fsi_ and t + 1e-10 * dt_fsi_ < maxtime_fsi_)
    {
      fsi_->output();
      scatra_output();
    }

    // access structure discretization
    std::shared_ptr<Core::FE::Discretization> strudis = fsi_->structure_field()->discretization();

    // recovery of forces at the interface nodes based on lagrange multipliers values
    // lambda_ is defined only at the interface, while lambdafull on the entire fluid/structure
    // field.
    std::shared_ptr<Core::LinAlg::Vector<double>> lambda_;
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull;

    // at the purpose to compute lambdafull, it is necessary to know which coupling algorithm is
    // used however the imposition of a Dirichlet condition on the interface produce wrong lambda_
    // when structuresplit is used
    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
    const auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");
    if (coupling == fsi_iter_monolithicfluidsplit)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> lambdafluid = fsi_->get_lambda();
      lambda_ = fsi_->fluid_to_struct(lambdafluid);
    }
    else if (coupling == fsi_iter_monolithicstructuresplit)
    {
      lambda_ = fsi_->get_lambda();
    }

    lambdafull = fsi_->structure_field()->interface()->insert_fsi_cond_vector(*lambda_);

    // calculate interface normals in deformed configuration
    std::shared_ptr<Core::LinAlg::Vector<double>> nodalnormals =
        std::make_shared<Core::LinAlg::Vector<double>>(*(strudis->dof_row_map()));

    Teuchos::ParameterList eleparams;
    eleparams.set("action", "calc_cur_nodal_normals");
    strudis->clear_state();
    strudis->set_state("displacement", *fsi_->structure_field()->dispnp());
    strudis->evaluate_condition(
        eleparams, nullptr, nullptr, nodalnormals, nullptr, nullptr, "FSICoupling");
    strudis->clear_state();

    const Core::LinAlg::Map* dofrowmap = strudis->dof_row_map();
    const Core::LinAlg::Map* noderowmap = strudis->node_row_map();
    Core::LinAlg::MultiVector<double> lambdanode(*noderowmap, 3, true);

    // lagrange multipliers defined on a nodemap are necessary
    for (int lnodeid = 0; lnodeid < strudis->num_my_row_nodes(); lnodeid++)
    {
      // get the processor local node
      Core::Nodes::Node* lnode = strudis->l_row_node(lnodeid);
      // get the dofs of the node
      //      std::vector<int> dofs= strudis->Dof(lnode);
      // the set of degrees of freedom associated with the node
      const std::vector<int> nodedofset = strudis->dof(0, lnode);

      for (unsigned index = 0; index < nodedofset.size(); ++index)
      {
        // get global id of the dof
        int gid = nodedofset[index];
        // get local id of the dof
        int lid = dofrowmap->lid(gid);

        //        int lnodeid=lnode->LID();

        double lambdai = lambdafull->local_values_as_span()[lid];
        int lnodeid = noderowmap->lid(lnode->id());
        (lambdanode)(index).replace_local_value(lnodeid, lambdai);
      }
    }
    // loop over all local interface nodes of structure discretization
    std::shared_ptr<Core::LinAlg::Map> condnodemap =
        Core::Conditions::condition_node_row_map(*strudis, "FSICoupling");
    for (int nodei = 0; nodei < condnodemap->num_my_elements(); nodei++)
    {
      // Here we rely on the fact that the structure scatra discretization is a clone of the
      // structure mesh

      // get the processor's local node with the same lnodeid
      int gnodeid = condnodemap->gid(nodei);
      Core::Nodes::Node* strulnode = strudis->g_node(gnodeid);
      // get the degrees of freedom associated with this node
      std::vector<int> strunodedofs = strudis->dof(0, strulnode);
      // determine number of space dimensions
      const int numdim = ((int)strunodedofs.size());

      std::vector<int> doflids(numdim);
      double temp = 0.;
      std::vector<double> unitnormal(3);
      for (int i = 0; i < numdim; ++i)
      {
        doflids[i] = strudis->dof_row_map()->lid(strunodedofs[i]);
        unitnormal[i] = nodalnormals->local_values_as_span()[doflids[i]];
        temp += unitnormal[i] * unitnormal[i];
      }
      double unitnormalabsval = sqrt(temp);
      int lnodeid = strudis->node_row_map()->lid(gnodeid);

      // compute average unit nodal normal
      std::vector<double> Values(numdim);
      for (int j = 0; j < numdim; ++j)
      {
        unitnormal[j] /= unitnormalabsval;
      }

      // compute tangents
      std::vector<double> unittangentone(3);
      std::vector<double> unittangenttwo(3);

      // take care of special case
      double TOL = 1e-11;
      if (abs(unitnormal[0]) < TOL && abs(unitnormal[1]) < TOL)
      {
        unittangentone[0] = 1.0;
        unittangentone[1] = 0.0;
        unittangentone[2] = 0.0;

        unittangenttwo[0] = 0.0;
        unittangenttwo[1] = 1.0;
        unittangenttwo[2] = 0.0;
      }
      else
      {
        // first unit tangent
        unittangentone[0] = -unitnormal[1];
        unittangentone[1] = unitnormal[0];
        unittangentone[2] = 0.0;
        temp = 0.;
        for (int i = 0; i < numdim; ++i)
        {
          temp += unittangentone[i] * unittangentone[i];
        }
        double unittangentoneabsval = sqrt(temp);
        for (int j = 0; j < numdim; ++j)
        {
          unittangentone[j] /= unittangentoneabsval;
        }

        // second unit tangent
        unittangenttwo[0] = -unitnormal[0] * unitnormal[2];
        unittangenttwo[1] = -unitnormal[1] * unitnormal[2];
        unittangenttwo[2] = unitnormal[0] * unitnormal[0] + unitnormal[1] * unitnormal[1];
        temp = 0.;
        for (int i = 0; i < numdim; ++i)
        {
          temp += unittangenttwo[i] * unittangenttwo[i];
        }
        double unittangenttwoabsval = sqrt(temp);
        for (int j = 0; j < numdim; ++j)
        {
          unittangenttwo[j] /= unittangenttwoabsval;
        }
      }

      double tempflux = 0.0;
      double tempnormtrac = 0.0;
      double temptangtracone = 0.0;
      double temptangtractwo = 0.0;
      for (int index = 0; index < numdim; ++index)
      {
        double fluxcomp = (*strufluxn)(index).local_values_as_span()[lnodeid];
        tempflux += fluxcomp * unitnormal[index];
        // for the calculation of the growth and erosion both the tangential and the normal
        // components of the forces acting on the interface are important.
        // Since probably they will have a different effect on the biofilm growth,
        // they are calculated separately and different coefficients can be used.
        double traccomp = lambdanode(index).local_values_as_span()[lnodeid];
        tempnormtrac += traccomp * unitnormal[index];
        temptangtracone += traccomp * unittangentone[index];
        temptangtractwo += traccomp * unittangenttwo[index];
      }

      if (avgrowth)
      {
        normtempinflux_.get_values()[lnodeid] += tempflux;
        normtemptraction_.get_values()[lnodeid] += abs(tempnormtrac);
        tangtemptractionone_.get_values()[lnodeid] += abs(temptangtracone);
        tangtemptractiontwo_.get_values()[lnodeid] += abs(temptangtractwo);
      }
      else
      {
        (*norminflux_).get_values()[lnodeid] = tempflux;
        (*normtraction_).get_values()[lnodeid] = abs(tempnormtrac);
        (*tangtractionone_).get_values()[lnodeid] = abs(temptangtracone);
        (*tangtractiontwo_).get_values()[lnodeid] = abs(temptangtractwo);
      }
    }
  }

  // here is the averaging of variables needed for biofilm growth, in case the average way was
  // chosen
  if (avgrowth)
  {
    std::shared_ptr<Core::FE::Discretization> strudis = fsi_->structure_field()->discretization();

    // loop over all local interface nodes of structure discretization
    std::shared_ptr<Core::LinAlg::Map> condnodemap =
        Core::Conditions::condition_node_row_map(*strudis, "FSICoupling");
    for (int i = 0; i < condnodemap->num_my_elements(); i++)
    {
      // get the processor's local node with the same lnodeid
      int gnodeid = condnodemap->gid(i);
      int lnodeid = strudis->node_row_map()->lid(gnodeid);

      (*norminflux_).get_values()[lnodeid] =
          normtempinflux_.local_values_as_span()[lnodeid] / step_fsi_;
      (*normtraction_).get_values()[lnodeid] =
          normtemptraction_.local_values_as_span()[lnodeid] / step_fsi_;
      (*tangtractionone_).get_values()[lnodeid] =
          tangtemptractionone_.local_values_as_span()[lnodeid] / step_fsi_;
      (*tangtractiontwo_).get_values()[lnodeid] =
          tangtemptractiontwo_.local_values_as_span()[lnodeid] / step_fsi_;
    }
  }

  time_fsi_ += t;
}

/*----------------------------------------------------------------------*
 | write FSI solutions into scatra discretisation             Thon 11/14|
 *----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::set_fsi_solution()
{
  set_mesh_disp();
  set_velocity_fields();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::compute_interface_vectors(Core::LinAlg::Vector<double>& idispnp,
    Core::LinAlg::Vector<double>& iveln, std::shared_ptr<Core::LinAlg::Vector<double>> struidispnp,
    Core::LinAlg::Vector<double>& struiveln)
{
  // initialize structure interface displacement at time t^{n+1}
  // shouldn't that be zeroed?
  struidispnp->put_scalar(0.0);

  // select biofilm growth boundaries
  std::string biogrcondname = "BioGrCoupling";

  // set action for elements: compute normal vectors at nodes (for reference configuration)
  std::shared_ptr<Core::FE::Discretization> strudis = fsi_->structure_field()->discretization();
  std::shared_ptr<Core::LinAlg::Vector<double>> nodalnormals =
      std::make_shared<Core::LinAlg::Vector<double>>(*(strudis->dof_row_map()));
  Teuchos::ParameterList eleparams;
  eleparams.set("action", "calc_ref_nodal_normals");
  strudis->evaluate_condition(
      eleparams, nullptr, nullptr, nodalnormals, nullptr, nullptr, biogrcondname);

  // select row map with nodes from condition
  std::shared_ptr<Core::LinAlg::Map> condnodemap =
      Core::Conditions::condition_node_row_map(*strudis, biogrcondname);

  // loop all conditioned nodes
  for (int i = 0; i < condnodemap->num_my_elements(); ++i)
  {
    int nodegid = condnodemap->gid(i);
    if (strudis->have_global_node(nodegid) == false) FOUR_C_THROW("node not found on this proc");
    Core::Nodes::Node* actnode = strudis->g_node(nodegid);
    std::vector<int> globaldofs = strudis->dof(0, actnode);
    const int numdim = (int)(globaldofs.size());

    // extract averaged nodal normal and compute its absolute value
    std::vector<double> unitnormal(numdim);
    double temp = 0.;
    for (int j = 0; j < numdim; ++j)
    {
      unitnormal[j] =
          nodalnormals->local_values_as_span()[strudis->dof_row_map()->lid(globaldofs[j])];
      temp += unitnormal[j] * unitnormal[j];
    }
    double unitnormalabsval = sqrt(temp);
    int lnodeid = strudis->node_row_map()->lid(nodegid);
    double influx = (*norminflux_).get_values()[lnodeid];
    double normforces = (*normtraction_).get_values()[lnodeid];
    double tangoneforce = (*tangtractionone_).get_values()[lnodeid];
    double tangtwoforce = (*tangtractiontwo_).get_values()[lnodeid];

    // compute average unit nodal normal and "interface velocity"
    std::vector<double> Values(numdim, 0);

    for (int j = 0; j < numdim; ++j)
    {
      unitnormal[j] /= unitnormalabsval;

      // Traction and compression probably have different effect on biofilm growth -->
      // different coefficients can be used
      double normforcecoef_;
      if (normforces > 0)
        normforcecoef_ = normforceposcoef_;
      else
        normforcecoef_ = normforcenegcoef_;

      // explanation of signs present in the following phenomenological laws
      // influx<0     --> growth  -->  - fluxcoef_
      // normforces>0 --> erosion -->  - normforcecoef_
      // tangforces>0 --> erosion -->  - tangforcecoef_
      // for pseudo-3D problems the second tangent should not be taken in account
      Values[j] = -fluxcoef_ * influx * unitnormal[j] -
                  normforcecoef_ * normforces * unitnormal[j] -
                  tangoneforcecoef_ * tangoneforce * unitnormal[j] -
                  tangtwoforcecoef_ * tangtwoforce * unitnormal[j];
    }

    struiveln_->replace_global_values(numdim, Values.data(), globaldofs.data());
  }

  struidispnp->update(dt_bio_, *struiveln_, 0.0);

  std::shared_ptr<Core::LinAlg::Vector<double>> fluididisp = fsi_->struct_to_fluid(struidispnp);
  idispnp.update(1.0, *fluididisp, 0.0);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::fluid_ale_solve()
{
  std::shared_ptr<Core::FE::Discretization> fluidaledis =
      fsi_->ale_field()->write_access_discretization();

  // if we have values at the fluid interface we need to apply them
  if (idispnp_ != nullptr)
  {
    fsi_->ale_field()->apply_interface_displacements(fluid_to_ale(*idispnp_));
  }

  fsi_->ale_field()->create_system_matrix(nullptr);
  fsi_->ale_field()->evaluate(nullptr, ALE::Utils::MapExtractor::dbc_set_biofilm);
  int error = fsi_->ale_field()->solve();
  if (error == 1) FOUR_C_THROW("Could not solve fluid ALE in biofilm FS3I!");
  fsi_->ale_field()->update_iter();

  // change nodes reference position of the fluid field
  std::shared_ptr<Core::LinAlg::Vector<double>> fluiddisp =
      ale_to_fluid_field(*fsi_->ale_field()->write_access_dispnp());
  std::shared_ptr<Core::FE::Discretization> fluiddis = fsi_->fluid_field()->discretization();
  Core::Geo::update_reference_config_with_disp(*fluiddis, *fluiddisp);



  // change nodes reference position also for the fluid ale field
  std::shared_ptr<Core::LinAlg::Vector<double>> fluidaledisp =
      fsi_->ale_field()->write_access_dispnp();
  Core::Geo::update_reference_config_with_disp(*fluidaledis, *fluidaledisp);

  // change nodes reference position also for scatra fluid field
  std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[0];
  std::shared_ptr<Core::FE::Discretization> scatradis = scatra->scatra_field()->discretization();
  FS3I::BioFilm::Utils::scatra_change_config(*scatradis, *fluiddis, *fluiddisp);

  // set the total displacement due to growth for output reasons
  // fluid
  fluid_growth_disp_->update(1.0, *fluiddisp, 1.0);
  fsi_->fluid_field()->set_fld_gr_disp(fluid_growth_disp_);
  // fluid scatra
  vec_to_scatravec(*scatradis, *fluid_growth_disp_, *scatra_fluid_growth_disp_);
  scatra->scatra_field()->set_sc_fld_gr_disp(scatra_fluid_growth_disp_);

  // computation of fluid solution
  // fluid_->Solve();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::struct_ale_solve()
{
  std::shared_ptr<Core::FE::Discretization> structaledis = ale_->write_access_discretization();

  // if we have values at the structure interface we need to apply them
  if (struidispnp_ != nullptr)
  {
    ale_->apply_interface_displacements(struct_to_ale(struidispnp_));
  }

  ale_->create_system_matrix(nullptr);
  ale_->evaluate(nullptr, ALE::Utils::MapExtractor::dbc_set_biofilm);
  int error = ale_->solve();
  if (error == 1) FOUR_C_THROW("Could not solve fluid ALE in biofilm FS3I!");
  ale_->update_iter();

  // change nodes reference position of the structure field
  std::shared_ptr<Core::LinAlg::Vector<double>> structdisp =
      ale_to_struct_field(ale_->write_access_dispnp());
  std::shared_ptr<Core::FE::Discretization> structdis = fsi_->structure_field()->discretization();
  Core::Geo::update_reference_config_with_disp(*structdis, *structdisp);
  structdis->fill_complete({
      .assign_degrees_of_freedom = false,
      .init_elements = true,
      .do_boundary_conditions = true,
  });

  // change nodes reference position also for the struct ale field
  Core::Geo::update_reference_config_with_disp(*structaledis, *ale_->write_access_dispnp());

  // change nodes reference position also for scatra structure field
  std::shared_ptr<Adapter::ScaTraBaseAlgorithm> struscatra = scatravec_[1];
  std::shared_ptr<Core::FE::Discretization> struscatradis =
      struscatra->scatra_field()->discretization();
  FS3I::BioFilm::Utils::scatra_change_config(*struscatradis, *structdis, *structdisp);

  // set the total displacement due to growth for output reasons
  // structure
  struct_growth_disp_->update(1.0, *structdisp, 1.0);
  fsi_->structure_field()->set_str_gr_disp(struct_growth_disp_);
  // structure scatra
  vec_to_scatravec(*struscatradis, *struct_growth_disp_, *scatra_struct_growth_disp_);
  struscatra->scatra_field()->set_sc_str_gr_disp(scatra_struct_growth_disp_);

  // computation of structure solution
  // structure_->Solve();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::BiofilmFSI::fluid_to_ale(
    Core::LinAlg::Vector<double>& iv) const
{
  return icoupfa_->master_to_slave(iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::BiofilmFSI::ale_to_fluid_field(
    Core::LinAlg::Vector<double>& iv) const
{
  return coupfa_->slave_to_master(iv);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::BiofilmFSI::ale_to_struct_field(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv) const
{
  return coupsa_->slave_to_master(*iv);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::BiofilmFSI::ale_to_struct_field(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupsa_->slave_to_master(*iv);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::BiofilmFSI::struct_to_ale(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv) const
{
  return icoupsa_->master_to_slave(*iv);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FS3I::BiofilmFSI::struct_to_ale(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return icoupsa_->master_to_slave(*iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::vec_to_scatravec(Core::FE::Discretization& scatradis,
    Core::LinAlg::Vector<double>& vec, Core::LinAlg::MultiVector<double>& scatravec)
{
  // loop over all local nodes of scatra discretization
  for (int lnodeid = 0; lnodeid < scatradis.num_my_row_nodes(); lnodeid++)
  {
    // determine number of space dimensions
    const int numdim = Global::Problem::instance()->n_dim();

    for (int index = 0; index < numdim; ++index)
    {
      double vecval = vec.local_values_as_span()[index + numdim * lnodeid];

      // insert value into node-based vector
      scatravec.replace_local_value(lnodeid, index, vecval);
    }

    // for 1- and 2-D problems: set all unused vector components to zero
    for (int index = numdim; index < 3; ++index)
    {
      scatravec.replace_local_value(lnodeid, index, 0.0);
    }
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::struct_gmsh_output()
{
  const std::shared_ptr<Core::FE::Discretization> structdis =
      fsi_->structure_field()->discretization();
  const std::shared_ptr<Core::FE::Discretization> structaledis =
      ale_->write_access_discretization();
  std::shared_ptr<Core::FE::Discretization> struscatradis =
      scatravec_[1]->scatra_field()->discretization();

  const std::string filename = Core::IO::Gmsh::get_new_file_name_and_delete_old_files("struct",
      structdis->writer()->output().file_name(), step_bio_, 701, false,
      Core::Communication::my_mpi_rank(structdis->get_comm()));
  std::ofstream gmshfilecontent(filename.c_str());

  std::shared_ptr<const Core::LinAlg::Vector<double>> structdisp = fsi_->structure_field()->dispn();
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "struct displacement \" {" << std::endl;
    // draw vector field 'struct displacement' for every element
    Core::IO::Gmsh::vector_field_dof_based_to_gmsh(*structdis, structdisp, gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  std::shared_ptr<const Core::LinAlg::Vector<double>> structaledisp = ale_->dispnp();
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "struct ale displacement \" {" << std::endl;
    // draw vector field 'struct ale displacement' for every element
    Core::IO::Gmsh::vector_field_dof_based_to_gmsh(*structaledis, structaledisp, gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  std::shared_ptr<const Core::LinAlg::Vector<double>> structphi =
      scatravec_[1]->scatra_field()->phinp();
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "struct phi \" {" << std::endl;
    // draw vector field 'struct phi' for every element
    Core::IO::Gmsh::scalar_field_to_gmsh(*struscatradis, structphi, gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  gmshfilecontent.close();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BiofilmFSI::fluid_gmsh_output()
{
  const std::shared_ptr<Core::FE::Discretization> fluiddis = fsi_->fluid_field()->discretization();
  const std::shared_ptr<Core::FE::Discretization> fluidaledis =
      fsi_->ale_field()->write_access_discretization();
  std::shared_ptr<Core::FE::Discretization> fluidscatradis =
      scatravec_[0]->scatra_field()->discretization();

  const std::string filenamefluid = Core::IO::Gmsh::get_new_file_name_and_delete_old_files("fluid",
      fluiddis->writer()->output().file_name(), step_bio_, 701, false,
      Core::Communication::my_mpi_rank(fluiddis->get_comm()));
  std::ofstream gmshfilecontent(filenamefluid.c_str());

  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidvel = fsi_->fluid_field()->velnp();
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "fluid velocity \" {" << std::endl;
    // draw vector field 'fluid velocity' for every element
    Core::IO::Gmsh::vector_field_dof_based_to_gmsh(*fluiddis, fluidvel, gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> fluidaledisp =
      fsi_->ale_field()->write_access_dispnp();
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "fluid ale displacement \" {" << std::endl;
    // draw vector field 'fluid ale displacement' for every element
    Core::IO::Gmsh::vector_field_dof_based_to_gmsh(*fluidaledis, fluidaledisp, gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> fluidphi = scatravec_[0]->scatra_field()->phinp();
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "fluid phi \" {" << std::endl;
    // draw vector field 'fluid phi' for every element
    Core::IO::Gmsh::scalar_field_to_gmsh(*fluidscatradis, fluidphi, gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  gmshfilecontent.close();

  return;
}

FOUR_C_NAMESPACE_CLOSE
