// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ehl_base.hpp"

#include "4C_adapter_coupling_ehl_mortar.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_node.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_ehl_utils.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_lubrication_adapter.hpp"
#include "4C_lubrication_timint_implicit.hpp"
#include "4C_mat_lubrication_mat.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                     (public) wirtz 12/15 |
 *----------------------------------------------------------------------*/
EHL::Base::Base(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& lubricationparams, const Teuchos::ParameterList& structparams,
    const std::string struct_disname, const std::string lubrication_disname)
    : AlgorithmBase(comm, globaltimeparams),
      structure_(nullptr),
      lubrication_(nullptr),
      fieldcoupling_(Teuchos::getIntegralValue<EHL::FieldCoupling>(
          Global::Problem::instance()->elasto_hydro_dynamic_params(), "FIELDCOUPLING")),
      dry_contact_(
          Global::Problem::instance()->elasto_hydro_dynamic_params().get<bool>("DRY_CONTACT_MODEL"))
{
  Global::Problem* problem = Global::Problem::instance();

  // get the solver number used for Lubrication solver
  const int linsolvernumber = lubricationparams.get<int>("LINEAR_SOLVER");

  // 2.- Setup discretizations and coupling.
  setup_discretizations(comm, struct_disname, lubrication_disname);

  setup_field_coupling(struct_disname, lubrication_disname);

  // 3.- Create the two uncoupled subproblems.
  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structdis =
      Global::Problem::instance()->get_dis(struct_disname);

  // set moving grid
  bool isale = true;

  // determine which time params to use to build the single fields
  // in case of time stepping time params have to be read from single field sections
  // in case of equal timestep size for all fields the time params are controlled solely
  // by the problem section (e.g. ehl or cell dynamic)
  const Teuchos::ParameterList* structtimeparams = &globaltimeparams;
  const Teuchos::ParameterList* lubricationtimeparams = &globaltimeparams;
  if (Global::Problem::instance()->elasto_hydro_dynamic_params().get<bool>("DIFFTIMESTEPSIZE"))
  {
    structtimeparams = &structparams;
    lubricationtimeparams = &lubricationparams;
  }

  std::shared_ptr<Adapter::StructureBaseAlgorithm> structure =
      std::make_shared<Adapter::StructureBaseAlgorithm>(
          *structtimeparams, const_cast<Teuchos::ParameterList&>(structparams), structdis);
  structure_ = std::dynamic_pointer_cast<Adapter::Structure>(structure->structure_field());
  structure_->setup();
  lubrication_ = std::make_shared<Lubrication::LubricationBaseAlgorithm>();
  lubrication_->setup(*lubricationtimeparams, lubricationparams,
      problem->solver_params(linsolvernumber), lubrication_disname, isale);
  mortaradapter_->store_dirichlet_status(*structure_field()->get_dbc_map_extractor());

  // Structure displacement at the lubricated interface
  std::shared_ptr<Core::LinAlg::Vector<double>> disp =
      Core::LinAlg::create_vector(*(structdis->dof_row_map()), true);

  mortaradapter_->integrate(disp, dt());
  // the film thickness initialization for very first time step
  heightold_ = mortaradapter_->nodal_gap();
}

/*----------------------------------------------------------------------*
 | read restart information for given time step   (public) wirtz 12/15  |
 *----------------------------------------------------------------------*/
void EHL::Base::read_restart(int restart)
{
  if (restart)
  {
    lubrication_->lubrication_field()->read_restart(restart);
    structure_->read_restart(restart);
    set_time_step(structure_->time_old(), restart);

    mortaradapter_->interface()->set_state(Mortar::state_old_displacement, *structure_->dispn());
    mortaradapter_->interface()->set_state(Mortar::state_new_displacement, *structure_->dispn());
    mortaradapter_->interface()->evaluate_nodal_normals();
    mortaradapter_->interface()->export_nodal_normals();
    mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::n_old);
    mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::dm);
    mortaradapter_->integrate(structure_->dispnp(), dt());
    heightold_ = mortaradapter_->nodal_gap();

    Core::IO::DiscretizationReader reader(*lubrication_->lubrication_field()->discretization(),
        Global::Problem::instance()->input_control_file(), restart);
    mortaradapter_->read_restart(reader);
  }
}

/*----------------------------------------------------------------------*
 | calculate velocities by a FD approximation               wirtz 12/15 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> EHL::Base::calc_velocity(
    const Core::LinAlg::Vector<double>& dispnp)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vel = nullptr;
  // copy D_n onto V_n+1
  vel = std::make_shared<Core::LinAlg::Vector<double>>(*(structure_->dispn()));
  // calculate velocity with timestep Dt()
  //  V_n+1^k = (D_n+1^k - D_n) / Dt
  vel->update(1. / dt(), dispnp, -1. / dt());

  return vel;
}  // calc_velocity()

/*----------------------------------------------------------------------*
 | test results (if necessary)                     (public) wirtz 12/15 |
 *----------------------------------------------------------------------*/
void EHL::Base::test_results(MPI_Comm comm)
{
  Global::Problem* problem = Global::Problem::instance();

  problem->add_field_test(structure_->create_field_test());
  problem->add_field_test(lubrication_->create_lubrication_field_test());
  problem->test_all(comm);
}

/*----------------------------------------------------------------------*
 | setup discretizations and dofsets                        wirtz 12/15 |
 *----------------------------------------------------------------------*/
void EHL::Base::setup_discretizations(
    MPI_Comm comm, const std::string struct_disname, const std::string lubrication_disname)
{
  // Scheme   : the structure discretization is received from the input. Then, an ale-lubrication
  // disc. is cloned.

  Global::Problem* problem = Global::Problem::instance();

  // 1.-Initialization.
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis(struct_disname);
  std::shared_ptr<Core::FE::Discretization> lubricationdis = problem->get_dis(lubrication_disname);
  if (!structdis->filled()) structdis->fill_complete();
  if (!lubricationdis->filled()) lubricationdis->fill_complete();

  // first call fill_complete for single discretizations.
  // This way the physical dofs are numbered successively
  structdis->fill_complete();
  lubricationdis->fill_complete();

  // build auxiliary dofsets, i.e. pseudo dofs on each discretization
  const int ndofpernode_lubrication = lubricationdis->num_dof(0, lubricationdis->l_row_node(0));
  const int ndofperelement_lubrication = 0;
  const int ndofpernode_struct = structdis->num_dof(0, structdis->l_row_node(0));
  const int ndofperelement_struct = 0;

  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux_lubrication =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
          ndofpernode_lubrication, ndofperelement_lubrication, 0, true);
  if (structdis->add_dof_set(dofsetaux_lubrication) != 1)
    FOUR_C_THROW("unexpected dof sets in structure field");

  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux_struct =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
          ndofpernode_struct, ndofperelement_struct, 0, true);
  if (lubricationdis->add_dof_set(dofsetaux_struct) != 1)
    FOUR_C_THROW("unexpected dof sets in lubrication field");

  // call assign_degrees_of_freedom also for auxiliary dofsets
  // note: the order of fill_complete() calls determines the gid numbering!
  // 1. structure dofs
  // 2. lubrication dofs
  // 3. structure auxiliary dofs
  // 4. lubrication auxiliary dofs
  structdis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
  lubricationdis->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
}

/*----------------------------------------------------------------------*
 | set structure solution on lubrication field              wirtz 12/15 |
 *----------------------------------------------------------------------*/
void EHL::Base::set_struct_solution(std::shared_ptr<const Core::LinAlg::Vector<double>> disp)
{
  //---------------------------------------------------------
  // 1. Update the Mortar Coupling
  //---------------------------------------------------------

  // Reevaluate the mortar martices D and M
  mortaradapter_->integrate(disp, dt());

  // Displace the mesh of the lubrication field in accordance with the slave-side interface
  set_mesh_disp(*disp);

  // Calculate the average tangential fractions of the structure velocities at the interface and
  // provide them to the lubrication field
  set_average_velocity_field();

  // Calculate the relative tangential fractions of the structure velocities at the interface and
  // provide them to the lubrication field
  set_relative_velocity_field();

  // Provide the gap at the interface
  set_height_field();

  // provide the heightdot (time derivative of the gap)
  set_height_dot();

  // Create DBC map for unprojectable nodes
  setup_unprojectable_dbc();

  return;
}

/*----------------------------------------------------------------------*
 | calc tractions, resulting from fluid (pressure and viscous) seitz 01/18 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> EHL::Base::evaluate_fluid_force(
    const Core::LinAlg::Vector<double>& pressure)
{
  // safety: unprojectable nodes to zero pressure
  if (inf_gap_toggle_lub_ != nullptr)
    for (int i = 0; i < lubrication_->lubrication_field()->prenp()->get_map().num_my_elements();
        ++i)
    {
      if (abs(inf_gap_toggle_lub_->local_values_as_span()[(inf_gap_toggle_lub_->get_map().lid(
                  lubrication_->lubrication_field()->prenp()->get_map().gid(i)))] -
              1) < 1.e-2)
        lubrication_->lubrication_field()->prenp()->get_values()[i] = 0.;
    }

  // Forces on the interfaces due to the fluid traction
  Core::LinAlg::Vector<double> slaveiforce(mortaradapter_->get_mortar_matrix_d()->domain_map());
  Core::LinAlg::Vector<double> masteriforce(mortaradapter_->get_mortar_matrix_m()->domain_map());

  stritraction_D_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*ada_strDisp_to_lubDisp_->master_dof_map());
  stritraction_M_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*ada_strDisp_to_lubDisp_->master_dof_map());

  // add pressure force
  add_pressure_force(slaveiforce, masteriforce);
  // add poiseuille flow force
  add_poiseuille_force(slaveiforce, masteriforce);
  // add couette flow force
  add_couette_force(slaveiforce, masteriforce);

  // External force vector (global)
  std::shared_ptr<Core::LinAlg::Vector<double>> strforce =
      std::make_shared<Core::LinAlg::Vector<double>>(*(structure_->dof_row_map()));

  // Insert both interface forces into the global force vector
  slaverowmapextr_->insert_vector(slaveiforce, 0, *strforce);
  masterrowmapextr_->insert_vector(masteriforce, 0, *strforce);

  return strforce;
}

/*----------------------------------------------------------------------*
 | set tractions, resulting from lubrication pressure       seitz 01/18 |
 *----------------------------------------------------------------------*/
void EHL::Base::set_lubrication_solution(
    std::shared_ptr<const Core::LinAlg::Vector<double>> pressure)
{
  // Provide the structure field with the force vector
  // Note that the mid-point values (gen-alpha) of the interface forces are evaluated in
  // Solid::TimIntGenAlpha::evaluate_force_residual()
  structure_->set_force_interface(evaluate_fluid_force(*pressure)->as_multi_vector());
}

void EHL::Base::add_pressure_force(
    Core::LinAlg::Vector<double>& slaveiforce, Core::LinAlg::Vector<double>& masteriforce)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> stritraction;

  std::shared_ptr<Core::LinAlg::Vector<double>> p_full =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *lubrication_->lubrication_field()->dof_row_map(1));
  if (lubrimaptransform_->Apply(*lubrication_->lubrication_field()->prenp(), *p_full))
    FOUR_C_THROW("apply failed");
  std::shared_ptr<Core::LinAlg::Vector<double>> p_exp =
      std::make_shared<Core::LinAlg::Vector<double>>(*mortaradapter_->slave_dof_map());
  p_exp = ada_strDisp_to_lubDisp_->slave_to_master(*p_full);
  stritraction = std::make_shared<Core::LinAlg::Vector<double>>(*mortaradapter_->normals());
  stritraction->multiply(-1., *mortaradapter_->normals(), *p_exp, 0.);

  // Get the Mortar D and M Matrix
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortard = mortaradapter_->get_mortar_matrix_d();
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarm = mortaradapter_->get_mortar_matrix_m();

  // f_slave = D^T*t
  mortard->multiply(true, *stritraction, slaveiforce);
  stritraction_D_->update(1., *stritraction, 1.);

  // f_master = -M^T*t
  mortarm->multiply(true, *stritraction, masteriforce);
  masteriforce.scale(-1.0);
  stritraction_M_->update(-1., *stritraction, 1.);
}

void EHL::Base::add_poiseuille_force(
    Core::LinAlg::Vector<double>& slaveiforce, Core::LinAlg::Vector<double>& masteriforce)
{
  // poiseuille flow forces
  std::shared_ptr<Core::LinAlg::Vector<double>> p_int =
      ada_strDisp_to_lubPres_->slave_to_master(*lubrication_->lubrication_field()->prenp());
  Core::LinAlg::Vector<double> p_int_full(*mortaradapter_->slave_dof_map());
  Core::LinAlg::export_to(*p_int, p_int_full);

  Core::LinAlg::Vector<double> nodal_gap(*mortaradapter_->slave_dof_map());
  slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), nodal_gap);

  Core::LinAlg::SparseMatrix m(*mortaradapter_->surf_grad_matrix());

  m.left_scale(nodal_gap);
  m.scale(-.5);

  Core::LinAlg::Vector<double> poiseuille_force(*mortaradapter_->slave_dof_map());
  m.Apply(p_int_full, poiseuille_force);

  Core::LinAlg::Vector<double> slave_psl(mortaradapter_->get_mortar_matrix_d()->domain_map());
  Core::LinAlg::Vector<double> master_psl(mortaradapter_->get_mortar_matrix_m()->domain_map());

  // f_slave = D^T*t
  mortaradapter_->get_mortar_matrix_d()->multiply(true, poiseuille_force, slave_psl);
  stritraction_D_->update(1., poiseuille_force, 1.);

  // f_master = +M^T*t // attention: no minus sign here: poiseuille points in same direction on
  // slave and master side
  mortaradapter_->get_mortar_matrix_m()->multiply(true, poiseuille_force, master_psl);
  stritraction_M_->update(1., poiseuille_force, 1.);

  // add the contribution
  slaveiforce.update(1., slave_psl, 1.);
  masteriforce.update(1., master_psl, 1.);
}


void EHL::Base::add_couette_force(
    Core::LinAlg::Vector<double>& slaveiforce, Core::LinAlg::Vector<double>& masteriforce)
{
  const int ndim = Global::Problem::instance()->n_dim();
  const std::shared_ptr<const Core::LinAlg::Vector<double>> relVel = mortaradapter_->rel_tang_vel();
  Core::LinAlg::Vector<double> height(*mortaradapter_->slave_dof_map());
  slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), height);
  Core::LinAlg::Vector<double> h_inv(*mortaradapter_->slave_dof_map());
  h_inv.reciprocal(height);
  Core::LinAlg::Vector<double> hinv_relV(*mortaradapter_->slave_dof_map());
  hinv_relV.multiply(1., h_inv, *relVel, 0.);

  Core::FE::Discretization& lub_dis = *lubrication_->lubrication_field()->discretization();
  Core::LinAlg::Vector<double> visc_vec(*lubrication_->lubrication_field()->dof_row_map(1));
  for (int i = 0; i < lub_dis.node_row_map()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* lnode = lub_dis.l_row_node(i);
    if (!lnode) FOUR_C_THROW("node not found");
    const double p = lubrication_->lubrication_field()->prenp()->local_values_as_span()[(
        lubrication_->lubrication_field()->prenp()->get_map().lid(lub_dis.dof(0, lnode, 0)))];

    std::shared_ptr<Core::Mat::Material> mat =
        lnode->adjacent_elements()[0].user_element()->material(0);
    if (!mat) FOUR_C_THROW("null pointer");
    std::shared_ptr<Mat::LubricationMat> lmat = std::dynamic_pointer_cast<Mat::LubricationMat>(mat);
    const double visc = lmat->compute_viscosity(p);

    for (int d = 0; d < ndim; ++d) visc_vec.replace_global_value(lub_dis.dof(1, lnode, d), visc);
  }
  std::shared_ptr<Core::LinAlg::Vector<double>> visc_vec_str =
      ada_strDisp_to_lubDisp_->slave_to_master(visc_vec);
  Core::LinAlg::Vector<double> couette_force(*mortaradapter_->slave_dof_map());
  couette_force.multiply(-1., *visc_vec_str, hinv_relV, 0.);

  Core::LinAlg::Vector<double> slave_cou(mortaradapter_->get_mortar_matrix_d()->domain_map());
  Core::LinAlg::Vector<double> master_cou(mortaradapter_->get_mortar_matrix_m()->domain_map());
  // f_slave = D^T*t
  mortaradapter_->get_mortar_matrix_d()->multiply(true, couette_force, slave_cou);
  stritraction_D_->update(1., couette_force, 1.);

  // f_master = -M^T*t
  mortaradapter_->get_mortar_matrix_m()->multiply(true, couette_force, master_cou);
  stritraction_M_->update(-1., couette_force, 1.);

  // add the contribution
  slaveiforce.update(1., slave_cou, 1.);
  masteriforce.update(-1., master_cou, 1.);
}

/*----------------------------------------------------------------------*
 | set structure velocity fields on lubrication field       seitz 12/17 |
 *----------------------------------------------------------------------*/
void EHL::Base::set_average_velocity_field()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> avVelLub =
      ada_strDisp_to_lubDisp_->master_to_slave(*mortaradapter_->av_tang_vel());
  lubrication_->lubrication_field()->set_average_velocity_field(1, avVelLub);
}

/*----------------------------------------------------------------------*
 | set structure relative velocity fields on lub. field     faraji 02/19 |
 *----------------------------------------------------------------------*/
void EHL::Base::set_relative_velocity_field()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> relVelLub =
      ada_strDisp_to_lubDisp_->master_to_slave(*mortaradapter_->rel_tang_vel());
  lubrication_->lubrication_field()->set_relative_velocity_field(1, relVelLub);
}

/*----------------------------------------------------------------------*
 | set film height on lubrication field                      wirtz 01/15 |
 *----------------------------------------------------------------------*/
void EHL::Base::set_height_field()
{
  //  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortardinv =
  //  mortaradapter_->GetDinvMatrix();
  std::shared_ptr<Core::LinAlg::Vector<double>> discretegap =
      Core::LinAlg::create_vector(*(slaverowmapextr_->map(0)), true);

  // get the weighted gap and store it in slave dof map (for each node, the scalar value is stored
  // in the 0th dof)
  slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), *discretegap);

  // store discrete gap in lubrication disp dof map (its the film height)
  std::shared_ptr<Core::LinAlg::Vector<double>> height =
      ada_strDisp_to_lubDisp_->master_to_slave(*discretegap);

  // provide film height to lubrication discretization
  lubrication_->lubrication_field()->set_height_field(1, height);
}

/*----------------------------------------------------------------------*
 | set time derivative of film height on lubrication field   Faraji 03/18|
 *----------------------------------------------------------------------*/
void EHL::Base::set_height_dot()
{
  Core::LinAlg::Vector<double> heightdot(*(mortaradapter_->nodal_gap()));
  std::shared_ptr<const Core::LinAlg::Vector<double>> heightnp = mortaradapter_->nodal_gap();

  heightdot.update(-1.0 / dt(), *heightold_, 1.0 / dt());

  std::shared_ptr<Core::LinAlg::Vector<double>> discretegap =
      Core::LinAlg::create_vector(*(slaverowmapextr_->map(0)), true);
  // get the weighted heightdot and store it in slave dof map (for each node, the scalar value is
  // stored in the 0th dof)
  slavemaptransform_->multiply(false, heightdot, *discretegap);
  // store discrete heightDot in lubrication disp dof map (its the film height time derivative)
  std::shared_ptr<Core::LinAlg::Vector<double>> heightdotSet =
      ada_strDisp_to_lubDisp_->master_to_slave(*discretegap);

  // provide film height time derivative to lubrication discretization
  lubrication_->lubrication_field()->set_height_dot_field(1, heightdotSet);
}

/*----------------------------------------------------------------------*
 | set structure mesh displacement on lubrication field     wirtz 03/15 |
 *----------------------------------------------------------------------*/
void EHL::Base::set_mesh_disp(const Core::LinAlg::Vector<double>& disp)
{
  // Extract the structure displacement at the slave-side interface
  std::shared_ptr<Core::LinAlg::Vector<double>> slaveidisp = Core::LinAlg::create_vector(
      *(slaverowmapextr_->map(0)), true);  // Structure displacement at the lubricated interface
  slaverowmapextr_->extract_vector(disp, 0, *slaveidisp);

  // Transfer the displacement vector onto the lubrication field
  std::shared_ptr<Core::LinAlg::Vector<double>> lubridisp =
      ada_strDisp_to_lubDisp_->master_to_slave(*slaveidisp);

  // Provide the lubrication discretization with the displacement
  lubrication_->lubrication_field()->apply_mesh_movement(lubridisp, 1);
}


/*----------------------------------------------------------------------*
 | Create DBC toggle for unprojectable nodes                seitz 01/18 |
 *----------------------------------------------------------------------*/
void EHL::Base::setup_unprojectable_dbc()
{
  if (not Global::Problem::instance()->elasto_hydro_dynamic_params().get<bool>("UNPROJ_ZERO_DBC"))
    return;

  Core::LinAlg::FEVector<double> inf_gap_toggle(*mortaradapter_->slave_dof_map(), true);
  for (int i = 0; i < mortaradapter_->interface()->slave_row_nodes()->num_my_elements(); ++i)
  {
    Core::Nodes::Node* node = mortaradapter_->interface()->discret().g_node(
        mortaradapter_->interface()->slave_row_nodes()->gid(i));
    if (!node) FOUR_C_THROW("gnode returned nullptr");
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("dynamic cast failed");
    if (cnode->data().getg() > 1.e11)
    {
      for (int e = 0; e < cnode->num_element(); ++e)
      {
        Core::Elements::Element* ele = cnode->adjacent_elements()[e].user_element();
        for (int nn = 0; nn < ele->num_node(); ++nn)
        {
          CONTACT::Node* cnn = dynamic_cast<CONTACT::Node*>(ele->nodes()[nn]);
          if (!cnn) FOUR_C_THROW("cast failed");
          for (int j = 0; j < 3; ++j)
          {
            const int row = cnn->dofs()[j];
            const double one = 1.;
            inf_gap_toggle.sum_into_global_values(1, &row, &one, 0);
          }
        }
      }
    }
  }

  inf_gap_toggle.complete(Epetra_Max, false);

  for (int i = 0; i < inf_gap_toggle.get_map().num_my_elements(); ++i)
    if (inf_gap_toggle.get_ref_of_epetra_fevector().operator()(0)->operator[](i) > 0.5)
      inf_gap_toggle.get_ref_of_epetra_fevector().operator()(0)->operator[](i) = 1.;

  std::shared_ptr<Core::LinAlg::Vector<double>> exp =
      std::make_shared<Core::LinAlg::Vector<double>>(*ada_strDisp_to_lubPres_->master_dof_map());
  Core::LinAlg::View inf_gap_toggle_view(inf_gap_toggle.get_ref_of_epetra_fevector());
  Core::LinAlg::export_to(
      inf_gap_toggle_view.underlying().as_multi_vector(), exp->as_multi_vector());
  inf_gap_toggle_lub_ = ada_strDisp_to_lubPres_->master_to_slave(*exp);

  static std::shared_ptr<Core::LinAlg::Vector<double>> old_toggle = nullptr;
  if (old_toggle != nullptr)
  {
    for (int i = 0; i < inf_gap_toggle_lub_->get_map().num_my_elements(); ++i)
      if (abs(inf_gap_toggle_lub_->local_values_as_span()[(i)] -
              old_toggle->local_values_as_span()[(i)]) > 1.e-12)
      {
        if (!Core::Communication::my_mpi_rank(get_comm()))
          std::cout << "dbc of unprojectable nodes changed boundary condition" << std::endl;
        break;
      }
  }
  else
  {
    double d = 0.;
    inf_gap_toggle_lub_->max_value(&d);

    if (!Core::Communication::my_mpi_rank(get_comm()))
      std::cout << "dbc of unprojectable nodes changed boundary condition" << std::endl;
  }
  old_toggle = std::make_shared<Core::LinAlg::Vector<double>>(*inf_gap_toggle_lub_);

  lubrication_->lubrication_field()->inf_gap_toggle() =
      std::make_shared<Core::LinAlg::Vector<double>>(*inf_gap_toggle_lub_);
}

/*----------------------------------------------------------------------*
 | setup adapters for EHL on boundary                       wirtz 01/15 |
 *----------------------------------------------------------------------*/
void EHL::Base::setup_field_coupling(
    const std::string struct_disname, const std::string lubrication_disname)
{
  Global::Problem* problem = Global::Problem::instance();
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis(struct_disname);
  std::shared_ptr<Core::FE::Discretization> lubricationdis = problem->get_dis(lubrication_disname);

  if (!structdis) FOUR_C_THROW("structure dis does not exist");
  if (!lubricationdis) FOUR_C_THROW("lubrication dis does not exist");

  const int ndim = Global::Problem::instance()->n_dim();

  //------------------------------------------------------------------
  // 1. Mortar coupling: Slave-side structure <-> Master-side structure
  //------------------------------------------------------------------

  // A mortar coupling adapter, using the "EHL Coupling Condition" is set up. The Coupling is
  // between the slave- and the master-side interface of the structure. Dofs, which, need to be
  // transferred from the master-side to the lubrication field, need to be mortar-projected to the
  // slave-side interface and then transferred by a matching-node coupling,  and vice versa. The
  // matching node coupling is defined below.

  std::vector<int> coupleddof(ndim, 1);
  mortaradapter_ = std::make_shared<Adapter::CouplingEhlMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());
  mortaradapter_->setup(structdis, structdis, coupleddof, "EHLCoupling");

  if (Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(
          mortaradapter_->interface()->interface_params(), "STRATEGY") !=
      CONTACT::SolvingStrategy::ehl)
    FOUR_C_THROW("you need to set ---CONTACT DYNAMIC: STRATEGY   Ehl");

  std::shared_ptr<Core::LinAlg::Vector<double>> idisp = Core::LinAlg::create_vector(
      *(structdis->dof_row_map()), true);  // Structure displacement at the lubricated interface
  mortaradapter_->interface()->initialize();
  mortaradapter_->interface()->set_state(Mortar::state_old_displacement, *idisp);
  mortaradapter_->interface()->set_state(Mortar::state_new_displacement, *idisp);
  mortaradapter_->interface()->evaluate_nodal_normals();
  mortaradapter_->interface()->export_nodal_normals();
  mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::n_old);
  mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::dm);
  mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::activeold);
  mortaradapter_->integrate(idisp, dt());

  // Maps of the interface dofs
  std::shared_ptr<const Core::LinAlg::Map> masterdofrowmap =
      mortaradapter_->interface()->master_row_dofs();
  std::shared_ptr<const Core::LinAlg::Map> slavedofrowmap =
      mortaradapter_->interface()->slave_row_dofs();
  std::shared_ptr<Core::LinAlg::Map> mergeddofrowmap =
      Core::LinAlg::merge_map(masterdofrowmap, slavedofrowmap, false);

  // Map extractors with the structure dofs as full maps and local interface maps
  slaverowmapextr_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *(structdis->dof_row_map()), slavedofrowmap, false);
  masterrowmapextr_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *(structdis->dof_row_map()), masterdofrowmap, false);
  mergedrowmapextr_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *(structdis->dof_row_map()), mergeddofrowmap, false);


  //----------------------------------------------------------
  // 2. build coupling adapters
  //----------------------------------------------------------
  std::shared_ptr<const Core::LinAlg::Map> strucnodes =
      mortaradapter_->interface()->slave_row_nodes();
  const Core::LinAlg::Map* lubrinodes = lubricationdis->node_row_map();
  ada_strDisp_to_lubDisp_ = std::make_shared<Coupling::Adapter::Coupling>();
  ada_strDisp_to_lubDisp_->setup_coupling(
      *structdis, *lubricationdis, *strucnodes, *lubrinodes, ndim, true, 1.e-8, 0, 1);

  ada_lubPres_to_lubDisp_ = std::make_shared<Coupling::Adapter::Coupling>();
  ada_lubPres_to_lubDisp_->setup_coupling(*lubricationdis, *lubricationdis,
      *lubricationdis->node_row_map(), *lubricationdis->node_row_map(), 1, true, 1.e-8, 0, 1);

  ada_strDisp_to_lubPres_ = std::make_shared<Coupling::Adapter::Coupling>();
  ada_strDisp_to_lubPres_->setup_coupling(mortaradapter_->interface()->discret(), *lubricationdis,
      *mortaradapter_->interface()->slave_row_nodes(), *lubricationdis->node_row_map(), 1, true,
      1.e-3);

  // Setup of transformation matrix: slave node map <-> slave disp dof map
  slavemaptransform_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*slavedofrowmap, 81, false, false);
  for (int i = 0; i < mortaradapter_->interface()->slave_row_nodes()->num_my_elements(); ++i)
  {
    int gid = mortaradapter_->interface()->slave_row_nodes()->gid(i);
    Core::Nodes::Node* node = structdis->g_node(gid);
    std::vector<int> dofs = structdis->dof(0, node);
    // slavemaptransform_->Assemble(1.0,dofs[0],gid);
    for (unsigned int idim = 0; idim < dofs.size(); idim++)
    {
      int row = dofs[idim];
      slavemaptransform_->assemble(1.0, row, gid);
    }
  }
  slavemaptransform_->complete(*(mortaradapter_->interface()->slave_row_nodes()), *slavedofrowmap);

  // Setup of transformation matrix: lubrication pre dof map <-> lubrication disp dof map
  lubrimaptransform_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(lubricationdis->dof_row_map(1)), 81, false, false);
  for (int inode = 0; inode < lubricationdis->num_my_row_nodes(); ++inode)
  {
    Core::Nodes::Node* node = lubricationdis->l_row_node(inode);
    std::vector<int> nodepredof = lubricationdis->dof(0, node);
    std::vector<int> nodedispdofs = lubricationdis->dof(1, node);
    for (unsigned int idim = 0; idim < nodedispdofs.size(); idim++)
      lubrimaptransform_->assemble(1.0, nodedispdofs[idim], nodepredof[0]);
  }
  lubrimaptransform_->complete(
      *(lubricationdis->dof_row_map(0)), *(lubricationdis->dof_row_map(1)));
}


/*----------------------------------------------------------------------*
 | update (protected)                                       wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Base::update()
{
  heightold_ = mortaradapter_->nodal_gap();
  mortaradapter_->interface()->set_state(
      Mortar::state_old_displacement, *structure_field()->dispnp());
  mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::n_old);
  mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::dm);
  mortaradapter_->interface()->store_to_old(Mortar::StrategyBase::activeold);
  structure_field()->update();
  lubrication_->lubrication_field()->update();

  return;
}

/*----------------------------------------------------------------------*
 | output (protected)                                       wirtz 01/16 |
 *----------------------------------------------------------------------*/
void EHL::Base::output(bool forced_writerestart)
{
  // Note: The order in the output is important here!

  // In here control file entries are written. And these entries define the
  // order in which the filters handle the Discretizations, which in turn
  // defines the dof number ordering of the Discretizations.

  //===========================
  // output for structurefield:
  //===========================
  //  ApplyLubricationCouplingState(lubrication_->LubricationField()->Prenp());
  structure_field()->output(forced_writerestart);

  // Additional output on structure field
  structure_field()->disc_writer()->write_vector("fluid_force",
      evaluate_fluid_force(*lubrication_->lubrication_field()->prenp()), Core::IO::dofvector);

  if (dry_contact_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> active_toggle, slip_toggle;
    mortaradapter_->create_active_slip_toggle(&active_toggle, &slip_toggle);
    for (int i = 0; i < active_toggle->get_map().num_my_elements(); ++i)
      slip_toggle->get_values()[i] += active_toggle->local_values_as_span()[(i)];
    std::shared_ptr<Core::LinAlg::Vector<double>> active =
        std::make_shared<Core::LinAlg::Vector<double>>(
            *structure_field()->discretization()->node_row_map());
    std::shared_ptr<Core::LinAlg::Vector<double>> slip =
        std::make_shared<Core::LinAlg::Vector<double>>(
            *structure_field()->discretization()->node_row_map());
    Core::LinAlg::export_to(*active_toggle, *active);
    Core::LinAlg::export_to(*slip_toggle, *slip);
    structure_field()->disc_writer()->write_vector("active", active, Core::IO::dofvector);
    structure_field()->disc_writer()->write_vector("slip", slip, Core::IO::dofvector);
  }
  if (dry_contact_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> n, t;
    mortaradapter_->create_force_vec(n, t);
    std::shared_ptr<Core::LinAlg::Vector<double>> ne =
        std::make_shared<Core::LinAlg::Vector<double>>(
            *structure_field()->discretization()->dof_row_map());
    std::shared_ptr<Core::LinAlg::Vector<double>> te =
        std::make_shared<Core::LinAlg::Vector<double>>(
            *structure_field()->discretization()->dof_row_map());
    Core::LinAlg::export_to(*n, *ne);
    Core::LinAlg::export_to(*t, *te);
    structure_field()->disc_writer()->write_vector("normal_contact", ne, Core::IO::dofvector);
    structure_field()->disc_writer()->write_vector("tangential_contact", te, Core::IO::dofvector);
  }

  //=============================
  // output for lubricationfield:
  //=============================
  set_mesh_disp(*structure_field()->dispnp());
  lubrication_->lubrication_field()->output(forced_writerestart);

  // ============================
  // output for mortar interface
  // ============================
  mortaradapter_->write_restart(*lubrication_->lubrication_field()->disc_writer());

  // Additional output on the lubrication field
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> discretegap =
        Core::LinAlg::create_vector(*(slaverowmapextr_->map(0)), true);

    // get the weighted gap and store it in slave dof map (for each node, the scalar value is stored
    // in the 0th dof)
    slavemaptransform_->multiply(false, *mortaradapter_->nodal_gap(), *discretegap);

    // store discrete gap in lubrication disp dof map (its the film height)
    std::shared_ptr<Core::LinAlg::Vector<double>> height =
        ada_strDisp_to_lubDisp_->master_to_slave(*discretegap);

    Core::LinAlg::Vector<double> height_ex(*ada_lubPres_to_lubDisp_->slave_dof_map());
    Core::LinAlg::export_to(*height, height_ex);
    std::shared_ptr<Core::LinAlg::Vector<double>> h1 =
        ada_lubPres_to_lubDisp_->slave_to_master(height_ex);
    lubrication_->lubrication_field()->disc_writer()->write_vector(
        "height", h1, Core::IO::dofvector);

    if (inf_gap_toggle_lub_ != nullptr)
      lubrication_->lubrication_field()->disc_writer()->write_vector(
          "no_gap_DBC", inf_gap_toggle_lub_, Core::IO::dofvector);

    // output for viscosity

    const int ndim = Global::Problem::instance()->n_dim();
    Core::LinAlg::Vector<double> visc_vec(*lubrication_->lubrication_field()->dof_row_map(1));
    for (int i = 0;
        i < lubrication_->lubrication_field()->discretization()->node_row_map()->num_my_elements();
        ++i)
    {
      Core::Nodes::Node* lnode = lubrication_->lubrication_field()->discretization()->l_row_node(i);
      if (!lnode) FOUR_C_THROW("node not found");
      const double p = lubrication_->lubrication_field()->prenp()->local_values_as_span()[(
          lubrication_->lubrication_field()->prenp()->get_map().lid(
              lubrication_->lubrication_field()->discretization()->dof(0, lnode, 0)))];
      std::shared_ptr<Core::Mat::Material> mat =
          lnode->adjacent_elements()[0].user_element()->material(0);
      if (!mat) FOUR_C_THROW("null pointer");
      std::shared_ptr<Mat::LubricationMat> lmat =
          std::dynamic_pointer_cast<Mat::LubricationMat>(mat);
      const double visc = lmat->compute_viscosity(p);

      for (int d = 0; d < ndim; ++d)
        visc_vec.replace_global_value(
            lubrication_->lubrication_field()->discretization()->dof(1, lnode, d), visc);
    }

    Core::LinAlg::Vector<double> visc_vec_ex(*ada_lubPres_to_lubDisp_->slave_dof_map());

    Core::LinAlg::export_to(visc_vec, visc_vec_ex);

    std::shared_ptr<Core::LinAlg::Vector<double>> v1 =
        ada_lubPres_to_lubDisp_->slave_to_master(visc_vec_ex);

    lubrication_->lubrication_field()->disc_writer()->write_vector(
        "viscosity", v1, Core::IO::dofvector);
  }

  // reset states
  structure_field()->discretization()->clear_state(true);
  lubrication_->lubrication_field()->discretization()->clear_state(true);
}  // output()

FOUR_C_NAMESPACE_CLOSE
