/*----------------------------------------------------------------------*/
/*! \file

\brief partitioned immersed fsi algorithm for neumann-(dirichlet-neumann) like coupling

\level 2


*----------------------------------------------------------------------*/
#include "4C_immersed_problem_immersed_partitioned_fsi_dirichletneumann.hpp"

#include "4C_adapter_fld_fluid_immersed.hpp"
#include "4C_adapter_str_fsiwrapper_immersed.hpp"
#include "4C_fem_geometry_searchtree.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_global_data.hpp"
#include "4C_immersed_problem_fsi_partitioned_immersed.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_immersed.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

Immersed::ImmersedPartitionedFSIDirichletNeumann::ImmersedPartitionedFSIDirichletNeumann(
    const Epetra_Comm& comm)
    : ImmersedBase(),
      FSI::PartitionedImmersed(comm),
      struct_bdry_traction_(Teuchos::null),
      fluid_artificial_velocity_(Teuchos::null),
      dbcmap_immersed_(Teuchos::null),
      fluid_SearchTree_(Teuchos::null),
      structure_SearchTree_(Teuchos::null),
      myrank_(comm.MyPID()),
      numproc_(comm.NumProc()),
      globalproblem_(nullptr),
      displacementcoupling_(false),
      multibodysimulation_(false),
      output_evry_nlniter_(false),
      is_relaxation_(false),
      correct_boundary_velocities_(0),
      degree_gp_fluid_bound_(0),
      artificial_velocity_isvalid_(false),
      boundary_traction_isvalid_(false),
      immersed_info_isvalid_(false),
      fluiddis_(Teuchos::null),
      structdis_(Teuchos::null),
      immersedstructure_(Teuchos::null)

{
  // empty constructor
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Immersed::ImmersedPartitionedFSIDirichletNeumann::init(const Teuchos::ParameterList& params)
{
  // reset the setup flag
  set_is_setup(false);

  // do all init stuff here

  // set isinit_ flag true
  set_is_init(true);

  return 0;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::read_restart(int step)
{
  FSI::Partitioned::read_restart(step);

  setup_structural_discretization();

  if (not displacementcoupling_)
  {
    calc_artificial_velocity();
    calc_fluid_tractions_on_structure();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::setup()
{
  // make sure init(...) was called first
  check_is_init();

  // call setup of base class
  FSI::PartitionedImmersed::setup();

  // get pointer to global problem
  globalproblem_ = Global::Problem::instance();

  // get pointer to discretizations
  fluiddis_ = globalproblem_->get_dis("fluid");
  structdis_ = globalproblem_->get_dis("structure");

  // cast to specialized adapter
  immersedstructure_ =
      Teuchos::rcp_dynamic_cast<Adapter::FSIStructureWrapperImmersed>(structure_field());

  // get coupling variable
  displacementcoupling_ =
      globalproblem_->fsi_dynamic_params()
          .sublist("PARTITIONED SOLVER")
          .get<Inpar::FSI::CoupVarPart>("COUPVARIABLE") == Inpar::FSI::CoupVarPart::disp;
  if (displacementcoupling_ and myrank_ == 0)
    std::cout << "\n Coupling variable for partitioned FSI scheme :  Displacements " << std::endl;
  else if (!displacementcoupling_ and myrank_ == 0)
    std::cout << "\n Coupling variable for partitioned FSI scheme :  Force " << std::endl;

  // set switch for interface velocity correction
  correct_boundary_velocities_ =
      globalproblem_->immersed_method_params().get<bool>("CORRECT_BOUNDARY_VELOCITIES");

  // set switch for output in every nln. iteration (for debugging)
  output_evry_nlniter_ = globalproblem_->immersed_method_params().get<bool>("OUTPUT_EVRY_NLNITER");

  // print acceleration method
  if (globalproblem_->fsi_dynamic_params().get<FsiCoupling>("COUPALGO") ==
      FsiCoupling::fsi_iter_stagg_fixed_rel_param)
  {
    is_relaxation_ = false;
    if (myrank_ == 0) std::cout << "\n Using FIXED relaxation parameter. " << std::endl;
  }
  else if (globalproblem_->fsi_dynamic_params().get<FsiCoupling>("COUPALGO") ==
           FsiCoupling::fsi_iter_stagg_AITKEN_rel_param)
  {
    is_relaxation_ = true;
    if (myrank_ == 0) std::cout << "\n Using AITKEN relaxation parameter. " << std::endl;
  }
  else
    FOUR_C_THROW("Unknown definition of COUPALGO in FSI DYNAMIC section for Immersed FSI.");

  // check for unfeasible combination
  if (correct_boundary_velocities_ and displacementcoupling_ and is_relaxation_)
    FOUR_C_THROW(
        "Interface velocity correction is not possible with displacement coupled Immersed FSI in "
        "combination with relaxation.");

  // get integration rule for fluid elements cut by structural boundary
  int num_gp_fluid_bound = globalproblem_->immersed_method_params().get<int>("NUM_GP_FLUID_BOUND");
  if (num_gp_fluid_bound == 8)
    degree_gp_fluid_bound_ = 3;
  else if (num_gp_fluid_bound == 64)
    degree_gp_fluid_bound_ = 7;
  else if (num_gp_fluid_bound == 125)
    degree_gp_fluid_bound_ = 9;
  else if (num_gp_fluid_bound == 343)
    degree_gp_fluid_bound_ = 13;
  else if (num_gp_fluid_bound == 729)
    degree_gp_fluid_bound_ = 17;
  else if (num_gp_fluid_bound == 1000)
    degree_gp_fluid_bound_ = 19;
  else
    FOUR_C_THROW(
        "Invalid value for parameter NUM_GP_FLUID_BOUND (valid parameters are 8, 64, 125, 343, "
        "729, and 1000).");

  // Decide whether multiple structural bodies or not.
  // Bodies need to be labeled with "ImmersedSearchbox" condition.
  std::vector<Core::Conditions::Condition*> conditions;
  structdis_->get_condition("ImmersedSearchbox", conditions);
  if ((int)conditions.size() > 0)
  {
    if (myrank_ == 0)
      std::cout << " MULTIBODY SIMULATION   Number of bodies: " << (int)conditions.size()
                << std::endl;
    multibodysimulation_ = true;
  }
  else
    multibodysimulation_ = false;

  // vector of fluid stresses interpolated to structural bdry. int. points and integrated over
  // structural surface
  struct_bdry_traction_ =
      Teuchos::make_rcp<Core::LinAlg::Vector<double>>(*(immersedstructure_->dof_row_map()), true);

  // vector with fluid velocities interpolated from structure
  fluid_artificial_velocity_ = Teuchos::make_rcp<Core::LinAlg::Vector<double>>(
      *(mb_fluid_field()->fluid_field()->dof_row_map()), true);

  // build 3D search tree for fluid domain
  fluid_SearchTree_ = Teuchos::make_rcp<Core::Geo::SearchTree>(5);

  // find positions of the background fluid discretization
  for (int lid = 0; lid < fluiddis_->num_my_col_nodes(); ++lid)
  {
    const Core::Nodes::Node* node = fluiddis_->l_col_node(lid);
    Core::LinAlg::Matrix<3, 1> currpos;

    currpos(0) = node->x()[0];
    currpos(1) = node->x()[1];
    currpos(2) = node->x()[2];

    currpositions_fluid_[node->id()] = currpos;
  }

  // find the bounding box of the elements and initialize the search tree
  const Core::LinAlg::Matrix<3, 2> rootBox =
      Core::Geo::get_xaab_bof_positions(currpositions_fluid_);
  fluid_SearchTree_->initialize_tree(rootBox, *fluiddis_, Core::Geo::TreeType(Core::Geo::OCTTREE));

  if (myrank_ == 0) std::cout << "\n Build Fluid SearchTree ... " << std::endl;

  // construct 3D search tree for structural domain
  // initialized in setup_structural_discretization()
  structure_SearchTree_ = Teuchos::make_rcp<Core::Geo::SearchTree>(5);

  // Validation flag for velocity in artificial domain. After each structure solve the velocity
  // becomes invalid and needs to be projected again.
  artificial_velocity_isvalid_ = false;

  // Validation flag for bdry. traction on structure. After each fluid solve the traction becomes
  // invalid and needs to be integrated again.
  boundary_traction_isvalid_ = false;

  // Validation flag for immersed information. After each structure solve, we have to assess
  // again, which fluid elements are covered by the structure and which fluid eles are cut by
  // the interface.
  // todo: NOTE: There is little inconsistency right now in this method.
  //             Fluid elements are labeled isboundarimmersed_ if:
  //             1) at least one but not all nodes are covered, or
  //             2) if a gp of the structural surface lies in a fluid element.
  //
  //             The first criterion is checked in calc_artificial_velocity(). This is
  //             done at the same time as struct vel. projection is performed.
  //             The second criterion is checked in calc_fluid_tractions_on_structure().
  //             This is done at the same time as the bdry. traction is integrated.
  //
  //             Since, the fluid field decides which gps of the fluid are compressible,
  //             based on IsImmersed() and IsBoundaryImmersed() information, it might
  //             happen, that after performing calc_artificial_velocity(), the nodal criterion
  //             is updated correctly, but since the structure has moved, the gp criterion
  //             is invalid. It is only updated after the next struct solve. So the fluid
  //             might be solved with incorrect compressible gps.
  //
  //             To fix this, one would have to split the information update and the projections.
  //
  //             However:
  //             1) This would be more expensive.
  //             2) I assume the error we make by the procedure explained above is very small, since
  //                the deformations between iteration steps should be small. Especially, when we
  //                are close to convergence, the difference in structure deformation should be so
  //                small, that no new gps should have changed their states inbetween the last two
  //                structure solves.
  immersed_info_isvalid_ = false;

  // wait for all processors to arrive here
  get_comm().Barrier();

  // set flag issetup true
  set_is_setup(true);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::fsi_op(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& F, const FillType fillFlag)
{
  check_is_init();
  check_is_setup();

  if (displacementcoupling_)
  {
    // get the current artificial velocity state
    const Teuchos::RCP<Core::LinAlg::Vector<double>> artificial_velocity_n =
        Teuchos::make_rcp<Core::LinAlg::Vector<double>>(x);

    ////////////////////
    // CALL FluidOp
    ////////////////////
    fluid_op(artificial_velocity_n, fillFlag);  //!< solve the fluid

    ////////////////////
    // CALL StructOp
    ////////////////////
    calc_fluid_tractions_on_structure();  //!< calculate new fluid tractions interpolated to
                                          //!< structural surface
    struct_op(immersedstructure_->interface()->extract_immersed_cond_vector(*struct_bdry_traction_),
        fillFlag);                 //!< solve the structure
    reset_immersed_information();  //!< structure moved; immersed info are invalid -> reset
    const Teuchos::RCP<Core::LinAlg::Vector<double>> artificial_velocity_np =
        calc_artificial_velocity();  //!< calc new projected velocities and update immersed
                                     //!< information

    int err = calc_residual(F, artificial_velocity_np, artificial_velocity_n);
    if (err != 0) FOUR_C_THROW("Vector update of FSI-residual returned err=%d", err);
  }
  else if (!displacementcoupling_)  // FORCE COUPLING
  {
    // get the current interface force state
    const Teuchos::RCP<Core::LinAlg::Vector<double>> iforcen =
        Teuchos::make_rcp<Core::LinAlg::Vector<double>>(x);

    ////////////////////
    // CALL StructOp
    ////////////////////
    struct_op(iforcen, fillFlag);  //!< solve the structure
    reset_immersed_information();  //!< structure moved; immersed info are invalid -> reset

    ////////////////////
    // CALL FluidOp
    ////////////////////
    calc_artificial_velocity();  //!< calc the new velocity in the artificial fluid domain, immersed
                                 //!< info are set inside
    fluid_op(fluid_artificial_velocity_, fillFlag);  //!< solve the fluid
    calc_fluid_tractions_on_structure();  //!< calculate new fluid tractions integrated over
                                          //!< structural surface

    int err = calc_residual(F, struct_bdry_traction_, iforcen);
    if (err != 0) FOUR_C_THROW("Vector update of FSI-residual returned err=%d", err);
  }

  // write output after every solve of fluid and structure
  // current limitations:
  // max 100 partitioned iterations and max 100 timesteps in total
  if (output_evry_nlniter_)
  {
    int iter = ((FSI::Partitioned::iteration_counter())[0]);
    constexpr bool force_prepare = false;
    structure_field()->prepare_output(force_prepare);

    Teuchos::rcp_dynamic_cast<Adapter::FSIStructureWrapperImmersed>(structure_field())
        ->output(false, (step() * 100) + (iter - 1), time() - dt() * ((100 - iter) / 100.0));
  }

  // perform n steps max; then set converged
  const bool nlnsolver_continue =
      globalproblem_->immersed_method_params().get<Inpar::Immersed::ImmersedNlnsolver>(
          "DIVERCONT") == Inpar::Immersed::ImmersedNlnsolver::nlnsolver_continue;
  int itemax =
      globalproblem_->fsi_dynamic_params().sublist("PARTITIONED SOLVER").get<int>("ITEMAX");
  if ((FSI::Partitioned::iteration_counter())[0] == itemax and nlnsolver_continue)
  {
    if (myrank_ == 0)
      std::cout << "\n  Continue with next time step after ITEMAX = "
                << (FSI::Partitioned::iteration_counter())[0] << " iterations. \n"
                << std::endl;

    // !!! EXPERIMENTAL !!!
    // set F to zero to tell NOX that this timestep is converged
    Teuchos::RCP<Core::LinAlg::Vector<double>> zeros =
        Teuchos::make_rcp<Core::LinAlg::Vector<double>>(F.Map(), true);
    F.Update(1.0, *zeros, 0.0);
    // !!! EXPERIMENTAL !!!

    // clear states after time step was set converged
    immersedstructure_->discretization()->clear_state();
    mb_fluid_field()->discretization()->clear_state();
  }

  if (globalproblem_->immersed_method_params().get<std::string>("TIMESTATS") == "everyiter")
  {
    Teuchos::TimeMonitor::summarize();
    Teuchos::TimeMonitor::zeroOutTimers();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Vector<double>>
Immersed::ImmersedPartitionedFSIDirichletNeumann::fluid_op(
    Teuchos::RCP<Core::LinAlg::Vector<double>> fluid_artificial_velocity, const FillType fillFlag)
{
  // print
  FSI::Partitioned::fluid_op(fluid_artificial_velocity, fillFlag);

  if (fillFlag == User)
  {
    FOUR_C_THROW("fillFlag == User : not yet implemented");
  }
  else
  {
    // get maximum number of Newton iterations
    const int itemax = mb_fluid_field()->itemax();

    // apply the given artificial velocity to the fluid field
    apply_immersed_dirichlet(fluid_artificial_velocity);

    // solve fluid
    solve_fluid();

    // remove immersed dirichlets from dbcmap of fluid (may be different in next iteration)
    remove_dirich_cond();

    // correct the quality of the interface solution
    correct_interface_velocity();

    // set max number of Newton iterations
    mb_fluid_field()->set_itemax(itemax);

    // we just invalidated the boundary tractions
    boundary_traction_isvalid_ = false;

  }  // fillflag is not User

  return Teuchos::null;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Vector<double>>
Immersed::ImmersedPartitionedFSIDirichletNeumann::struct_op(
    Teuchos::RCP<Core::LinAlg::Vector<double>> struct_bdry_traction, const FillType fillFlag)
{
  FSI::Partitioned::struct_op(struct_bdry_traction, fillFlag);

  if (fillFlag == User)
  {
    FOUR_C_THROW("fillFlag == User : not yet implemented");
    return Teuchos::null;
  }
  else
  {
    // prescribe neumann values at structural boundary dofs
    apply_interface_forces(struct_bdry_traction);

    // solve
    solve_struct();

    // structure moved; we just invalidated the artificial velocity
    artificial_velocity_isvalid_ = false;

    // we also invalidated the immersed info
    immersed_info_isvalid_ = false;


    return extract_interface_dispnp();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Vector<double>>
Immersed::ImmersedPartitionedFSIDirichletNeumann::initial_guess()
{
  if (myrank_ == 0) std::cout << "\n Do Initial Guess." << std::endl;

  if (displacementcoupling_)
    return calc_artificial_velocity();
  else
    return immersedstructure_->interface()->extract_immersed_cond_vector(*struct_bdry_traction_);

  return Teuchos::null;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::build_immersed_dirich_map(
    Teuchos::RCP<Core::FE::Discretization> dis, Teuchos::RCP<Epetra_Map>& dirichmap,
    const Teuchos::RCP<const Epetra_Map>& dirichmap_original)
{
  const Epetra_Map* elecolmap = dis->element_col_map();
  std::vector<int> mydirichdofs(0);

  for (int i = 0; i < elecolmap->NumMyElements(); ++i)
  {
    // dynamic_cast necessary because virtual inheritance needs runtime information
    Discret::ELEMENTS::FluidImmersedBase* immersedele =
        dynamic_cast<Discret::ELEMENTS::FluidImmersedBase*>(dis->g_element(elecolmap->GID(i)));
    if (immersedele->has_projected_dirichlet())
    {
      Core::Nodes::Node** nodes = immersedele->nodes();
      for (int inode = 0; inode < (immersedele->num_node()); inode++)
      {
        if (static_cast<Core::Nodes::ImmersedNode*>(nodes[inode])->is_matched() and
            nodes[inode]->owner() == myrank_)
        {
          std::vector<int> dofs = dis->dof(nodes[inode]);

          for (int dim = 0; dim < 3; ++dim)
          {
            // if not already in original dirich map
            if (dirichmap_original->LID(dofs[dim]) == -1) mydirichdofs.push_back(dofs[dim]);
          }
          // include also pressure dof if node does not belong to a boundary background element
          // if((nodes[inode]->IsBoundaryImmersed())==0)
          //  mydirichdofs.push_back(dofs[3]);
        }
      }
    }
  }

  int nummydirichvals = mydirichdofs.size();
  dirichmap =
      Teuchos::make_rcp<Epetra_Map>(-1, nummydirichvals, mydirichdofs.data(), 0, dis->get_comm());

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::do_immersed_dirichlet_cond(
    Teuchos::RCP<Core::LinAlg::Vector<double>> statevector,
    Teuchos::RCP<Core::LinAlg::Vector<double>> dirichvals, Teuchos::RCP<Epetra_Map> dbcmap)
{
  int mynumvals = dbcmap->NumMyElements();
  double* myvals = dirichvals->Values();

  for (int i = 0; i < mynumvals; ++i)
  {
    int gid = dbcmap->GID(i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    int err = -2;
    int lid = dirichvals->Map().LID(gid);
    err = statevector->ReplaceGlobalValue(gid, 0, myvals[lid]);
    if (err == -1)
      FOUR_C_THROW("VectorIndex >= NumVectors()");
    else if (err == 1)
      FOUR_C_THROW("GlobalRow not associated with calling processor");
    else if (err != -1 and err != 1 and err != 0)
      FOUR_C_THROW("Trouble using ReplaceGlobalValue on fluid state vector. ErrorCode = %d", err);
#else
    int lid = dirichvals->Map().LID(gid);
    statevector->ReplaceGlobalValue(gid, 0, myvals[lid]);
#endif
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::setup_structural_discretization()
{
  // find positions of the immersed structural discretization
  std::map<int, Core::LinAlg::Matrix<3, 1>> my_currpositions_struct;
  for (int lid = 0; lid < structdis_->num_my_row_nodes(); ++lid)
  {
    const Core::Nodes::Node* node = structdis_->l_row_node(lid);
    Core::LinAlg::Matrix<3, 1> currpos;

    currpos(0) = node->x()[0];
    currpos(1) = node->x()[1];
    currpos(2) = node->x()[2];

    my_currpositions_struct[node->id()] = currpos;
  }
  // Communicate local currpositions:
  // map with current structural positions should be same on all procs
  // to make use of the advantages of ghosting the structure redundantly
  // on all procs.
  std::vector<int> procs(numproc_);
  for (int i = 0; i < numproc_; i++) procs[i] = i;
  Core::LinAlg::gather<int, Core::LinAlg::Matrix<3, 1>>(
      my_currpositions_struct, currpositions_struct_, numproc_, procs.data(), get_comm());

  // find the bounding box of the elements and initialize the search tree
  const Core::LinAlg::Matrix<3, 2> rootBox2 =
      Core::Geo::get_xaab_bof_dis(*structdis_, currpositions_struct_);
  structure_SearchTree_->initialize_tree(
      rootBox2, *structdis_, Core::Geo::TreeType(Core::Geo::OCTTREE));

  if (myrank_ == 0) std::cout << "\n Build Structure SearchTree ... " << std::endl;

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::set_states_fluid_op()
{
  // for FluidOP
  structdis_->set_state(0, "displacement", immersedstructure_->dispnp());
  structdis_->set_state(0, "velocity", immersedstructure_->velnp());

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::set_states_velocity_correction()
{
  structdis_->set_state(0, "displacement", immersedstructure_->dispnp());
  structdis_->set_state(0, "velocity", immersedstructure_->velnp());
  fluiddis_->set_state(0, "velnp", mb_fluid_field()->fluid_field()->velnp());

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::set_states_struct_op()
{
  structdis_->set_state(0, "displacement", immersedstructure_->dispnp());
  fluiddis_->set_state(0, "velnp",
      Teuchos::rcp_dynamic_cast<Adapter::FluidImmersed>(mb_fluid_field())->fluid_field()->velnp());
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::solve_fluid()
{
  mb_fluid_field()->nonlinear_solve(Teuchos::null, Teuchos::null);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::solve_struct()
{
  immersedstructure_->solve();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::prepare_fluid_op()
{
  TEUCHOS_FUNC_TIME_MONITOR("Immersed::prepare_fluid_op()");

  // search radius factor around center of structure bounding box (fac*diagonal of bounding box)
  const double structsearchradiusfac =
      globalproblem_->immersed_method_params().get<double>("STRCT_SRCHRADIUS_FAC");

  // determine subset of fluid discretization which is potentially underlying the structural
  // discretization
  //
  // get state
  Teuchos::RCP<const Core::LinAlg::Vector<double>> displacements = immersedstructure_->dispnp();

  // find current positions for immersed structural discretization
  std::map<int, Core::LinAlg::Matrix<3, 1>> my_currpositions_struct;
  for (int lid = 0; lid < structdis_->num_my_row_nodes(); ++lid)
  {
    const Core::Nodes::Node* node = structdis_->l_row_node(lid);
    Core::LinAlg::Matrix<3, 1> currpos;
    std::vector<int> dofstoextract(3);
    std::vector<double> mydisp(3);

    // get the current displacement
    structdis_->dof(node, 0, dofstoextract);
    Core::FE::extract_my_values(*displacements, mydisp, dofstoextract);

    currpos(0) = node->x()[0] + mydisp.at(0);
    currpos(1) = node->x()[1] + mydisp.at(1);
    currpos(2) = node->x()[2] + mydisp.at(2);

    my_currpositions_struct[node->id()] = currpos;
  }

  // Communicate local currpositions:
  // map with current structural positions should be same on all procs
  // to make use of the advantages of ghosting the structure redundantly
  // on all procs.
  std::vector<int> procs(numproc_);
  for (int i = 0; i < numproc_; i++) procs[i] = i;
  Core::LinAlg::gather<int, Core::LinAlg::Matrix<3, 1>>(
      my_currpositions_struct, currpositions_struct_, numproc_, procs.data(), get_comm());

  // take special care in case of multibody simulations
  if (multibodysimulation_ == false)
  {
    // get bounding box of current configuration of structural dis
    const Core::LinAlg::Matrix<3, 2> structBox =
        Core::Geo::get_xaab_bof_dis(*structdis_, currpositions_struct_);
    double max_radius =
        sqrt(pow(structBox(0, 0) - structBox(0, 1), 2) + pow(structBox(1, 0) - structBox(1, 1), 2) +
             pow(structBox(2, 0) - structBox(2, 1), 2));
    // search for background elements within a certain radius around the center of the immersed
    // bounding box
    Core::LinAlg::Matrix<3, 1> boundingboxcenter;
    boundingboxcenter(0) = structBox(0, 0) + (structBox(0, 1) - structBox(0, 0)) * 0.5;
    boundingboxcenter(1) = structBox(1, 0) + (structBox(1, 1) - structBox(1, 0)) * 0.5;
    boundingboxcenter(2) = structBox(2, 0) + (structBox(2, 1) - structBox(2, 0)) * 0.5;

#ifdef FOUR_C_ENABLE_ASSERTIONS
    std::cout << "Bounding Box of Structure: " << structBox << " on PROC " << myrank_ << std::endl;
    std::cout << "Bounding Box Center of Structure: " << boundingboxcenter << " on PROC " << myrank_
              << std::endl;
    std::cout << "Search Radius Around Center: " << structsearchradiusfac * max_radius
              << " on PROC " << myrank_ << std::endl;
    std::cout << "Length of Dispnp()=" << displacements->MyLength() << " on PROC " << myrank_
              << std::endl;
    std::cout << "Size of currpositions_struct_=" << currpositions_struct_.size() << " on PROC "
              << myrank_ << std::endl;
    std::cout << "My dof_row_map Size=" << structdis_->dof_row_map()->NumMyElements()
              << "  My DofColMap Size=" << structdis_->dof_col_map()->NumMyElements() << " on PROC "
              << myrank_ << std::endl;
    std::cout << "Dis Structure NumColEles: " << structdis_->num_my_col_elements() << " on PROC "
              << myrank_ << std::endl;
#endif

    search_potentially_covered_backgrd_elements(&curr_subset_of_fluiddis_, fluid_SearchTree_,
        *fluiddis_, currpositions_fluid_, boundingboxcenter, structsearchradiusfac * max_radius, 0);

    if (curr_subset_of_fluiddis_.empty() == false)
      std::cout << "\nPrepareFluidOp returns " << curr_subset_of_fluiddis_.begin()->second.size()
                << " background elements on Proc " << myrank_ << std::endl;
  }
  else
  {
    // get searchbox conditions on bodies
    std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator curr;
    std::vector<Core::Conditions::Condition*> conditions;
    structdis_->get_condition("ImmersedSearchbox", conditions);

    // build element list
    std::map<int, std::set<int>> elementList;
    std::set<int> settoinsert;
    for (int i = 0; i < (int)conditions.size(); ++i)
    {
      for (curr = conditions[i]->geometry().begin(); curr != conditions[i]->geometry().end();
           ++curr)
      {
        settoinsert.insert(curr->second->id());
      }
      elementList.insert(std::pair<int, std::set<int>>(i, settoinsert));
      settoinsert.clear();
    }

    // get bounding boxes of the bodies
    std::vector<Core::LinAlg::Matrix<3, 2>> structboxes =
        Core::Geo::compute_xaabb_for_labeled_structures(
            *structdis_, currpositions_struct_, elementList);

    double max_radius;

    // search for background elements within a certain radius around the center of the immersed
    // bounding box
    for (int i = 0; i < (int)structboxes.size(); ++i)
    {
      max_radius = sqrt(pow(structboxes[i](0, 0) - structboxes[i](0, 1), 2) +
                        pow(structboxes[i](1, 0) - structboxes[i](1, 1), 2) +
                        pow(structboxes[i](2, 0) - structboxes[i](2, 1), 2));

      Core::LinAlg::Matrix<3, 1> boundingboxcenter;
      boundingboxcenter(0) =
          structboxes[i](0, 0) + (structboxes[i](0, 1) - structboxes[i](0, 0)) * 0.5;
      boundingboxcenter(1) =
          structboxes[i](1, 0) + (structboxes[i](1, 1) - structboxes[i](1, 0)) * 0.5;
      boundingboxcenter(2) =
          structboxes[i](2, 0) + (structboxes[i](2, 1) - structboxes[i](2, 0)) * 0.5;

      std::map<int, std::set<int>> tempmap = fluid_SearchTree_->search_elements_in_radius(
          *fluiddis_, currpositions_fluid_, boundingboxcenter, 0.5 * max_radius, 0);
      curr_subset_of_fluiddis_.insert(std::pair<int, std::set<int>>(i, (tempmap.begin()->second)));
    }

    for (int i = 0; i < (int)structboxes.size(); ++i)
      std::cout << "\nPrepareFluidOp returns " << curr_subset_of_fluiddis_.at(i).size()
                << " background elements for body " << i << std::endl;
  }  //

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Vector<double>>
Immersed::ImmersedPartitionedFSIDirichletNeumann::extract_interface_dispnp()
{
  return immersedstructure_->extract_immersed_interface_dispnp();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::apply_interface_forces(
    Teuchos::RCP<Core::LinAlg::Vector<double>> full_traction_vec)
{
  double normorstructbdrytraction;
  full_traction_vec->Norm2(&normorstructbdrytraction);
  if (myrank_ == 0)
  {
    std::cout << "#################################################################################"
                 "###############"
              << std::endl;
    std::cout << "###   Norm of Boundary Traction:   " << normorstructbdrytraction << std::endl;
    std::cout << "#################################################################################"
                 "###############"
              << std::endl;
  }
  immersedstructure_->apply_immersed_interface_forces(Teuchos::null, full_traction_vec);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::add_dirich_cond()
{
  Teuchos::rcp_dynamic_cast<Adapter::FluidImmersed>(mb_fluid_field())
      ->add_dirich_cond(dbcmap_immersed_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::remove_dirich_cond()
{
  Teuchos::rcp_dynamic_cast<Adapter::FluidImmersed>(mb_fluid_field())
      ->remove_dirich_cond(dbcmap_immersed_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Immersed::ImmersedPartitionedFSIDirichletNeumann::calc_residual(Core::LinAlg::Vector<double>& F,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> newstate,
    const Teuchos::RCP<Core::LinAlg::Vector<double>> oldstate)
{
  int err = -1234;

  if (!displacementcoupling_)
    err = F.Update(1.0, *(immersedstructure_->interface()->extract_immersed_cond_vector(*newstate)),
        -1.0, *oldstate, 0.0);
  else
    err = F.Update(1.0, *newstate, -1.0, *oldstate, 0.0);

  return err;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::calc_fluid_tractions_on_structure()
{
  // sanity check
  if (boundary_traction_isvalid_)
    FOUR_C_THROW(
        "Boundary traction from fluid onto immersed structure is still valid!\n"
        "If you really need to calc them anew, invalidate flag boundary_traction_isvalid_ at the "
        "proper position.");

  // reinitialize the transfer vector
  struct_bdry_traction_->PutScalar(0.0);

  // declare and fill parameter list
  Teuchos::ParameterList params;
  params.set<std::string>("action", "calc_fluid_traction");
  params.set<std::string>("backgrddisname", "fluid");
  params.set<std::string>("immerseddisname", "structure");

  // set the states needed for evaluation
  set_states_struct_op();

  Core::FE::AssembleStrategy struct_bdry_strategy(0,  // struct dofset for row
      0,                                              // struct dofset for column
      Teuchos::null,                                  // matrix 1
      Teuchos::null,                                  //
      struct_bdry_traction_,                          // vector 1
      Teuchos::null,                                  //
      Teuchos::null                                   //
  );
  if (myrank_ == 0)
  {
    std::cout << "#################################################################################"
                 "###############"
              << std::endl;
    std::cout << "###   Interpolate fluid stresses to structural surface and calculate traction "
                 "...              "
              << std::endl;
    std::cout << "#################################################################################"
                 "###############"
              << std::endl;
  }
  evaluate_interpolation_condition(
      immersedstructure_->discretization(), params, struct_bdry_strategy, "IMMERSEDCoupling", -1);

  // we just validate the boundary tractions
  boundary_traction_isvalid_ = true;

  // we just validated the immersed info again.
  // technically this is not entirely true.
  // see remark in constructor of this class.
  // here we additionally validated the IsBoundaryImmersed
  // information based on the struct. bdry. int. points.
  immersed_info_isvalid_ = true;

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Vector<double>>
Immersed::ImmersedPartitionedFSIDirichletNeumann::calc_artificial_velocity()
{
  if (not artificial_velocity_isvalid_)
  {
    // reinitialize the transfer vector
    fluid_artificial_velocity_->PutScalar(0.0);

    // declare parameter list
    Teuchos::ParameterList params;

    // provide number of integration points in fluid elements cut by boundary
    params.set<int>("intpoints_fluid_bound", degree_gp_fluid_bound_);
    // provide name of immersed discretization
    params.set<std::string>("immerseddisname", "structure");

    // set the states needed for evaluation
    set_states_fluid_op();
    // update search trees, etc. ...
    prepare_fluid_op();

    // calc the fluid velocity from the structural displacements
    Core::FE::AssembleStrategy fluid_vol_strategy(0,  // struct dofset for row
        0,                                            // struct dofset for column
        Teuchos::null,                                // matrix 1
        Teuchos::null,                                //
        fluid_artificial_velocity_,                   // vector 1
        Teuchos::null,                                //
        Teuchos::null                                 //
    );

    if (myrank_ == 0)
    {
      std::cout << "###############################################################################"
                   "#################"
                << std::endl;
      std::cout << "###   Interpolate Dirichlet Values from immersed elements which overlap the "
                << mb_fluid_field()->discretization()->name() << " nodes ..." << std::endl;
      std::cout << "###############################################################################"
                   "#################"
                << std::endl;
    }

    evaluate_immersed(params, mb_fluid_field()->discretization(), &fluid_vol_strategy,
        &curr_subset_of_fluiddis_, structure_SearchTree_, &currpositions_struct_,
        FLD::interpolate_velocity_to_given_point_immersed, false);

    // we just validated the artificial velocity
    artificial_velocity_isvalid_ = true;

    // we just validated the immersed info again.
    // technically this is not entirely true.
    // see remark in constructor of this class.
    immersed_info_isvalid_ = true;
  }

  return fluid_artificial_velocity_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::apply_immersed_dirichlet(
    Teuchos::RCP<Core::LinAlg::Vector<double>> artificial_velocity)
{
  build_immersed_dirich_map(mb_fluid_field()->discretization(), dbcmap_immersed_,
      mb_fluid_field()->fluid_field()->get_dbc_map_extractor()->cond_map());
  add_dirich_cond();

  // apply immersed dirichlets
  do_immersed_dirichlet_cond(
      mb_fluid_field()->fluid_field()->write_access_velnp(), artificial_velocity, dbcmap_immersed_);
  double normofvelocities = -1234.0;
  mb_fluid_field()
      ->fluid_field()
      ->extract_velocity_part(artificial_velocity)
      ->Norm2(&normofvelocities);

  if (myrank_ == 0)
  {
    std::cout << "#################################################################################"
                 "###############"
              << std::endl;
    std::cout << "###   Norm of Dirichlet Values:   " << std::setprecision(7) << normofvelocities
              << std::endl;
    std::cout << "#################################################################################"
                 "###############"
              << std::endl;
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::correct_interface_velocity()
{
  //***********************************************************************************
  // Correct velocity at nodes of fluid elements being cut by the structural surface
  // (fluid is solved a second time with different Dirichlet values)
  //***********************************************************************************
  if (correct_boundary_velocities_)
  {
    // declare parameter list
    Teuchos::ParameterList params;

    // provide number of integration points in fluid elements cut by boundary
    params.set<int>("intpoints_fluid_bound", degree_gp_fluid_bound_);

    // calc the fluid velocity from the structural displacements
    Core::FE::AssembleStrategy fluid_vol_strategy(0,  // struct dofset for row
        0,                                            // struct dofset for column
        Teuchos::null,                                // matrix 1
        Teuchos::null,                                //
        fluid_artificial_velocity_,                   // vector 1
        Teuchos::null,                                //
        Teuchos::null                                 //
    );

    set_states_velocity_correction();

    if (myrank_ == 0)
    {
      std::cout << "\nCorrection step " << std::endl;
      std::cout << "###############################################################################"
                   "#################"
                << std::endl;
      std::cout << "###   Correct Velocity in fluid boundary elements " << std::endl;
    }

    // calculate new dirichlet velocities for fluid elements cut by structure
    evaluate_immersed(params, mb_fluid_field()->discretization(), &fluid_vol_strategy,
        &curr_subset_of_fluiddis_, structure_SearchTree_, &currpositions_struct_,
        FLD::correct_immersed_fluid_bound_vel, true);

    // Build new dirich map
    build_immersed_dirich_map(mb_fluid_field()->discretization(), dbcmap_immersed_,
        mb_fluid_field()->fluid_field()->get_dbc_map_extractor()->cond_map());
    add_dirich_cond();

    // apply new dirichlets after velocity correction
    do_immersed_dirichlet_cond(mb_fluid_field()->fluid_field()->write_access_velnp(),
        fluid_artificial_velocity_, dbcmap_immersed_);
    double normofnewvelocities;
    mb_fluid_field()
        ->fluid_field()
        ->extract_velocity_part(fluid_artificial_velocity_)
        ->Norm2(&normofnewvelocities);

    if (myrank_ == 0)
    {
      std::cout << "###   Norm of new Dirichlet Values:   " << std::setprecision(7)
                << normofnewvelocities << std::endl;
      std::cout << "###############################################################################"
                   "#################"
                << std::endl;
    }

    // solve fluid again with new dirichlet values
    solve_fluid();

    // remove immersed dirichlets from dbcmap of fluid (may be different in next iteration step)
    remove_dirich_cond();

  }  // correct_boundary_velocities_ finished

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedPartitionedFSIDirichletNeumann::reset_immersed_information()
{
  if (immersed_info_isvalid_)
    FOUR_C_THROW(
        "Immersed information are valid! Reconsider your call to reset_immersed_information().\n"
        "Did you forget to invalidate the flag immersed_info_isvalid_?");

  if (myrank_ == 0) std::cout << "\nReset Immersed Information ...\n" << std::endl;

  // reset element and node information about immersed method
  Teuchos::ParameterList params;
  params.set<FLD::Action>("action", FLD::reset_immersed_ele);
  params.set<int>("intpoints_fluid_bound", degree_gp_fluid_bound_);
  evaluate_subset_elements(
      params, fluiddis_, curr_subset_of_fluiddis_, (int)FLD::reset_immersed_ele);

  return;
}

FOUR_C_NAMESPACE_CLOSE
