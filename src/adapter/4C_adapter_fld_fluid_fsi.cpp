// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_fld_fluid_fsi.hpp"

#include "4C_adapter_fld_fluid.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>
#include <set>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*======================================================================*/
/* constructor */
Adapter::FluidFSI::FluidFSI(std::shared_ptr<Fluid> fluid,
    std::shared_ptr<Core::FE::Discretization> dis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Core::IO::DiscretizationWriter> output, bool isale, bool dirichletcond)
    : FluidWrapper(fluid),
      dis_(dis),
      params_(params),
      output_(output),
      dirichletcond_(dirichletcond),
      interface_(std::make_shared<FLD::Utils::MapExtractor>()),
      meshmap_(std::make_shared<Core::LinAlg::MapExtractor>()),
      locerrvelnp_(nullptr),
      auxintegrator_(Inpar::FSI::timada_fld_none),
      numfsidbcdofs_(0),
      methodadapt_(ada_none)
{
  // make sure
  if (fluid_ == nullptr) FOUR_C_THROW("Failed to create the underlying fluid adapter");
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::init()
{
  // call base class init
  FluidWrapper::init();

  // cast fluid to fluidimplicit
  if (fluidimpl_ == nullptr)
    fluidimpl_ = std::dynamic_pointer_cast<FLD::FluidImplicitTimeInt>(fluid_);

  if (fluidimpl_ == nullptr)
    FOUR_C_THROW("Failed to cast Adapter::Fluid to FLD::FluidImplicitTimeInt.");

  // default dofset for coupling
  int nds_master = 0;

  // set nds_master = 2 in case of HDG discretization
  // (nds = 0 used for trace values, nds = 1 used for interior values)
  if (Global::Problem::instance()->spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg)
  {
    nds_master = 2;
  }

  // create fluid map extractor
  setup_interface(nds_master);

  fluidimpl_->set_surface_splitter(&(*interface_));

  // create map of inner velocity dof (no FSI or Dirichlet conditions)
  build_inner_vel_map();

  if (dirichletcond_)
  {
    // mark all interface velocities as dirichlet values
    fluidimpl_->add_dirich_cond(interface()->fsi_cond_map());
  }

  interfaceforcen_ = std::make_shared<Core::LinAlg::Vector<double>>(*(interface()->fsi_cond_map()));

  // time step size adaptivity in monolithic FSI
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const bool timeadapton = fsidyn.sublist("TIMEADAPTIVITY").get<bool>("TIMEADAPTON");
  if (timeadapton)
  {
    // extract the type of auxiliary integrator from the input parameter list
    auxintegrator_ = Teuchos::getIntegralValue<Inpar::FSI::FluidMethod>(
        fsidyn.sublist("TIMEADAPTIVITY"), "AUXINTEGRATORFLUID");

    if (auxintegrator_ != Inpar::FSI::timada_fld_none)
    {
      // determine type of adaptivity
      if (aux_method_order_of_accuracy() > fluidimpl_->method_order_of_accuracy())
        methodadapt_ = ada_upward;
      else if (aux_method_order_of_accuracy() < fluidimpl_->method_order_of_accuracy())
        methodadapt_ = ada_downward;
      else
        methodadapt_ = ada_orderequal;
    }

    //----------------------------------------------------------------------------
    // Handling of Dirichlet BCs in error estimation
    //----------------------------------------------------------------------------
    // Create intersection of fluid DOFs that hold a Dirichlet boundary condition
    // and are located at the FSI interface.
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmaps;
    intersectionmaps.push_back(get_dbc_map_extractor()->cond_map());
    intersectionmaps.push_back(interface()->fsi_cond_map());
    std::shared_ptr<Core::LinAlg::Map> intersectionmap =
        Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

    // store number of interface DOFs subject to Dirichlet BCs on structure and fluid side of the
    // interface
    numfsidbcdofs_ = intersectionmap->num_global_elements();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Adapter::FluidFSI::dof_row_map() { return dof_row_map(0); }


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Adapter::FluidFSI::dof_row_map(unsigned nds)
{
  const Core::LinAlg::Map* dofrowmap = dis_->dof_row_map(nds);
  return Core::Utils::shared_ptr_from_ref(*dofrowmap);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Adapter::FluidFSI::time_scaling() const
{
  if (params_->get<bool>("interface second order"))
    return 2. / dt();
  else
    return 1. / dt();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::update()
{
  if (Global::Problem::instance()->spatial_approximation_type() !=
      Core::FE::ShapeFunctionType::hdg)  // TODO also fix this!
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> interfaceforcem =
        interface()->extract_fsi_cond_vector(*true_residual());

    interfaceforcen_ = fluidimpl_->extrapolate_end_point(interfaceforcen_, interfaceforcem);
  }

  FluidWrapper::update();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidFSI::relaxation_solve(
    std::shared_ptr<Core::LinAlg::Vector<double>> ivel)
{
  const Core::LinAlg::Map* dofrowmap = discretization()->dof_row_map();
  std::shared_ptr<Core::LinAlg::Vector<double>> relax =
      Core::LinAlg::create_vector(*dofrowmap, true);
  interface()->insert_fsi_cond_vector(*ivel, *relax);
  fluidimpl_->linear_relaxation_solve(relax);
  return extract_interface_forces();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Adapter::FluidFSI::inner_velocity_row_map()
{
  return innervelmap_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidFSI::extract_interface_forces()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> interfaceforcem =
      interface()->extract_fsi_cond_vector(*true_residual());

  return fluidimpl_->extrapolate_end_point(interfaceforcen_, interfaceforcem);
}

/*----------------------------------------------------------------------*
 | Return interface velocity at new time level n+1                      |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidFSI::extract_interface_velnp()
{
  return interface()->extract_fsi_cond_vector(*velnp());
}


/*----------------------------------------------------------------------*
 | Return interface velocity at old time level n                        |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidFSI::extract_interface_veln()
{
  return interface()->extract_fsi_cond_vector(*veln());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::apply_interface_velocities(
    std::shared_ptr<Core::LinAlg::Vector<double>> ivel)
{
  // apply the interface velocities
  interface()->insert_fsi_cond_vector(*ivel, *fluidimpl_->write_access_velnp());

  const Teuchos::ParameterList& fsipart =
      Global::Problem::instance()->fsi_dynamic_params().sublist("PARTITIONED SOLVER");
  if (fsipart.get<bool>("DIVPROJECTION"))
  {
    // project the velocity field into a divergence free subspace
    // (might enhance the linear solver, but we are still not sure.)
    proj_vel_to_div_zero();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::apply_initial_mesh_displacement(
    std::shared_ptr<const Core::LinAlg::Vector<double>> initfluiddisp)
{
  // cast fluid to fluidimplicit
  if (fluidimpl_ == nullptr)
    fluidimpl_ = std::dynamic_pointer_cast<FLD::FluidImplicitTimeInt>(fluid_);

  if (fluidimpl_ == nullptr)
    FOUR_C_THROW("Failed to cast Adapter::Fluid to FLD::FluidImplicitTimeInt.");

  meshmap_->insert_cond_vector(*initfluiddisp, *fluidimpl_->create_dispn());
  meshmap_->insert_cond_vector(*initfluiddisp, *fluidimpl_->create_dispnp());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::apply_mesh_displacement(
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddisp)
{
  meshmap_->insert_cond_vector(*fluiddisp, *fluidimpl_->write_access_dispnp());

  // new grid velocity
  fluidimpl_->update_gridv();
}

/*----------------------------------------------------------------------*
 | Update fluid griv velocity via FD approximation           Thon 12/14 |
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::update_gridv()
{
  // new grid velocity via FD approximation
  fluidimpl_->update_gridv();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::apply_mesh_velocity(
    std::shared_ptr<const Core::LinAlg::Vector<double>> gridvel)
{
  meshmap_->insert_cond_vector(*gridvel, *fluidimpl_->write_access_grid_vel());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::set_mesh_map(
    std::shared_ptr<const Core::LinAlg::Map> mm, const int nds_master)
{
  meshmap_->setup(*dis_->dof_row_map(nds_master), mm,
      Core::LinAlg::split_map(*dis_->dof_row_map(nds_master), *mm));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::displacement_to_velocity(std::shared_ptr<Core::LinAlg::Vector<double>> fcx)
{
  // get interface velocity at t(n)
  const std::shared_ptr<const Core::LinAlg::Vector<double>> veln_vector =
      interface()->extract_fsi_cond_vector(*veln());

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check, whether maps are the same
  if (!fcx->get_map().point_same_as(veln_vector->get_map()))
  {
    FOUR_C_THROW("Maps do not match, but they have to.");
  }
#endif

  /*
   * Delta u(n+1,i+1) = fac * (Delta d(n+1,i+1) - dt * u(n))
   *
   *             / = 2 / dt   if interface time integration is second order
   * with fac = |
   *             \ = 1 / dt   if interface time integration is first order
   */
  const double timescale = time_scaling();
  fcx->update(-timescale * dt(), *veln_vector, timescale);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidFSI::velocity_to_displacement(std::shared_ptr<Core::LinAlg::Vector<double>> fcx)
{
  // get interface velocity at t(n)
  const std::shared_ptr<const Core::LinAlg::Vector<double>> veln_vector =
      interface()->extract_fsi_cond_vector(*veln());

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check, whether maps are the same
  if (!fcx->get_map().point_same_as(veln_vector->get_map()))
  {
    FOUR_C_THROW("Maps do not match, but they have to.");
  }
#endif

  /*
   * Delta d(n+1,i+1) = tau * Delta u(n+1,i+1) + dt * u(n)]
   *
   *             / = dt / 2   if interface time integration is second order
   * with tau = |
   *             \ = dt       if interface time integration is first order
   */
  const double tau = 1. / time_scaling();
  fcx->update(dt(), *veln_vector, tau);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidFSI::integrate_interface_shape()
{
  return interface()->extract_fsi_cond_vector(
      *fluidimpl_->integrate_interface_shape("FSICoupling"));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::use_block_matrix(bool splitmatrix)
{
  std::shared_ptr<std::set<int>> condelements =
      interface()->conditioned_element_map(*discretization());
  fluidimpl_->use_block_matrix(condelements, *interface(), *interface(), splitmatrix);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Adapter::FluidFSI::linear_solver()
{
  return FluidWrapper::linear_solver();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::proj_vel_to_div_zero()
{
  // This projection affects also the inner DOFs. Unfortunately, the matrix
  // does not look nice. Hence, the inversion of B^T*B is quite costly and
  // we are not sure yet whether it is worth the effort.

  //   get maps with Dirichlet DOFs and fsi interface DOFs
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> dbcfsimaps;
  dbcfsimaps.push_back(get_dbc_map_extractor()->cond_map());
  dbcfsimaps.push_back(interface()->fsi_cond_map());

  // create a map with all DOFs that have either a Dirichlet boundary condition
  // or are located on the fsi interface
  std::shared_ptr<Core::LinAlg::Map> dbcfsimap =
      Core::LinAlg::MultiMapExtractor::merge_maps(dbcfsimaps);

  // create an element map with offset
  const int numallele = discretization()->num_global_elements();
  const int mapoffset =
      dbcfsimap->max_all_gid() + discretization()->element_row_map()->min_all_gid() + 1;
  std::shared_ptr<Core::LinAlg::Map> elemap =
      std::make_shared<Core::LinAlg::Map>(numallele, mapoffset, discretization()->get_comm());

  // create the combination of dbcfsimap and elemap
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> domainmaps;
  domainmaps.push_back(dbcfsimap);
  domainmaps.push_back(elemap);
  std::shared_ptr<Core::LinAlg::Map> domainmap =
      Core::LinAlg::MultiMapExtractor::merge_maps(domainmaps);

  // build the corresponding map extractor
  Core::LinAlg::MapExtractor domainmapex(*domainmap, dbcfsimap);

  const int numofrowentries = 82;
  std::shared_ptr<Core::LinAlg::SparseMatrix> B =
      std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map(), numofrowentries, false);

  // define element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  discretization()->clear_state();
  discretization()->set_state("dispnp", *dispnp());

  // loop over all fluid elements
  for (int lid = 0; lid < discretization()->num_my_col_elements(); lid++)
  {
    // get pointer to current element
    Core::Elements::Element* actele = discretization()->l_col_element(lid);

    // get element location vector and ownerships
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    actele->location_vector(*discretization(), lm, lmowner, lmstride);

    // get dimension of element matrices and vectors
    const int eledim = (int)lm.size();

    // Reshape element matrices and vectors and initialize to zero
    elevector1.size(eledim);

    // set action in order to calculate the integrated divergence operator via an evaluate()-call
    Teuchos::ParameterList params;
    params.set<FLD::Action>("action", FLD::calc_divop);

    // call the element specific evaluate method
    actele->evaluate(
        params, *discretization(), lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);

    // assembly
    std::vector<int> lmcol(1);
    lmcol[0] = actele->id() + dbcfsimap->max_all_gid() + 1;
    B->assemble(actele->id(), lmstride, elevector1, lm, lmowner, lmcol);
  }  // end of loop over all fluid elements

  discretization()->clear_state();

  // insert '1's for all DBC and interface DOFs
  for (int i = 0; i < dbcfsimap->num_my_elements(); i++)
  {
    int rowid = dbcfsimap->gid(i);
    int colid = dbcfsimap->gid(i);
    B->assemble(1.0, rowid, colid);
  }

  B->complete(*domainmap, *dof_row_map());

  // Compute the projection operator
  std::shared_ptr<Core::LinAlg::SparseMatrix> BTB =
      Core::LinAlg::matrix_multiply(*B, true, *B, false, true);

  std::shared_ptr<Core::LinAlg::Vector<double>> BTvR =
      std::make_shared<Core::LinAlg::Vector<double>>(*domainmap);
  B->multiply(true, *velnp(), *BTvR);
  Core::LinAlg::Vector<double> zeros(*dbcfsimap, true);

  domainmapex.insert_cond_vector(zeros, *BTvR);

  std::shared_ptr<Core::LinAlg::Vector<double>> x =
      std::make_shared<Core::LinAlg::Vector<double>>(*domainmap);

  const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();
  const int simplersolvernumber = fdyn.get<int>("LINEAR_SOLVER");
  if (simplersolvernumber == (-1))
    FOUR_C_THROW(
        "no simpler solver, that is used to solve this system, defined for fluid pressure problem. "
        "\nPlease set LINEAR_SOLVER in FLUID DYNAMIC to a valid number!");

  std::shared_ptr<Core::LinAlg::Solver> solver = std::make_shared<Core::LinAlg::Solver>(
      Global::Problem::instance()->solver_params(simplersolvernumber), discretization()->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  if (solver->params().isSublist("ML Parameters"))
  {
    std::shared_ptr<Core::LinAlg::MultiVector<double>> pressure_nullspace =
        std::make_shared<Core::LinAlg::MultiVector<double>>(*(dis_->dof_row_map()), 1);
    pressure_nullspace->PutScalar(1.0);

    solver->params().sublist("ML Parameters").set("PDE equations", 1);
    solver->params().sublist("ML Parameters").set("null space: dimension", 1);
    solver->params()
        .sublist("ML Parameters")
        .set("null space: vectors", pressure_nullspace->Values());
    solver->params().sublist("ML Parameters").remove("nullspace", false);  // necessary?
    solver->params()
        .sublist("Michael's secret vault")
        .set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
            "pressure nullspace", pressure_nullspace);
  }

  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver->solve(BTB, x, BTvR, solver_params);

  std::shared_ptr<Core::LinAlg::Vector<double>> vmod =
      std::make_shared<Core::LinAlg::Vector<double>>(velnp()->get_map(), true);
  B->Apply(*x, *vmod);
  write_access_velnp()->update(-1.0, *vmod, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::reset(bool completeReset, int numsteps, int iter)

{
  FluidWrapper::reset(completeReset, numsteps, iter);
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::calculate_error()

{
  fluidimpl_->evaluate_error_compared_to_analytical_sol();
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::time_step_auxiliary()
{
  // current state
  const Core::LinAlg::Vector<double> veln_vector(*veln());
  const Core::LinAlg::Vector<double> accn_vector(*accn());

  // prepare vector for solution of auxiliary time step
  locerrvelnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_->dof_row_map(), true);

  // ---------------------------------------------------------------------------

  // calculate time step with auxiliary time integrator, i.e. the extrapolated solution
  switch (auxintegrator_)
  {
    case Inpar::FSI::timada_fld_none:
    {
      break;
    }
    case Inpar::FSI::timada_fld_expleuler:
    {
      explicit_euler(veln_vector, accn_vector, *locerrvelnp_);

      break;
    }
    case Inpar::FSI::timada_fld_adamsbashforth2:
    {
      if (step() >= 1)  // adams_bashforth2 only if at least second time step
      {
        // Acceleration from previous time step
        Core::LinAlg::Vector<double> accnm_vector(*extract_velocity_part(accnm()));

        adams_bashforth2(veln_vector, accn_vector, accnm_vector, *locerrvelnp_);
      }
      else  // explicit_euler as starting algorithm
      {
        explicit_euler(veln_vector, accn_vector, *locerrvelnp_);
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown auxiliary time integration scheme for fluid field.");
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::explicit_euler(const Core::LinAlg::Vector<double>& veln,
    const Core::LinAlg::Vector<double>& accn, Core::LinAlg::Vector<double>& velnp) const
{
  // Do a single explicit Euler step
  velnp.update(1.0, veln, dt(), accn, 0.0);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::adams_bashforth2(const Core::LinAlg::Vector<double>& veln,
    const Core::LinAlg::Vector<double>& accn, const Core::LinAlg::Vector<double>& accnm,
    Core::LinAlg::Vector<double>& velnp) const
{
  // time step sizes of current and previous time step
  const double current_dt = dt();
  const double dto = fluidimpl_->dt_previous();

  // Do a single Adams-Bashforth 2 step
  velnp.update(1.0, veln, 0.0);
  velnp.update((2.0 * current_dt * dto + current_dt * current_dt) / (2 * dto), accn,
      -current_dt * current_dt / (2.0 * dto), accnm, 1.0);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::indicate_error_norms(double& err, double& errcond, double& errother,
    double& errinf, double& errinfcond, double& errinfother)
{
  // compute estimation of local discretization error
  if (methodadapt_ == ada_orderequal)
  {
    const double coeffmarch = fluidimpl_->method_lin_err_coeff_vel();
    const double coeffaux = aux_method_lin_err_coeff_vel();
    locerrvelnp_->update(-1.0, *velnp(), 1.0);
    locerrvelnp_->scale(coeffmarch / (coeffaux - coeffmarch));
  }
  else
  {
    // schemes do not have the same order of accuracy
    locerrvelnp_->update(-1.0, *velnp(), 1.0);
  }

  // set '0' on all pressure DOFs
  auto zeros = std::make_shared<Core::LinAlg::Vector<double>>(locerrvelnp_->get_map(), true);
  Core::LinAlg::apply_dirichlet_to_system(*locerrvelnp_, *zeros, *pressure_row_map());
  // TODO: Do not misuse apply_dirichlet_to_system()...works for this purpose here: writes zeros
  // into all pressure DoFs

  // set '0' on Dirichlet DOFs
  zeros = std::make_shared<Core::LinAlg::Vector<double>>(locerrvelnp_->get_map(), true);
  Core::LinAlg::apply_dirichlet_to_system(
      *locerrvelnp_, *zeros, *(get_dbc_map_extractor()->cond_map()));

  // extract the condition part of the full error vector (i.e. only interface velocity DOFs)
  Core::LinAlg::Vector<double> errorcond(*interface()->extract_fsi_cond_vector(*locerrvelnp_));

  /* in case of structure split: extract the other part of the full error vector
   * (i.e. interior velocity and all pressure DOFs) */
  std::shared_ptr<Core::LinAlg::Vector<double>> errorother =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *interface()->extract_other_vector(*locerrvelnp_));

  // calculate L2-norms of different subsets of temporal discretization error vector
  // (neglect Dirichlet and pressure DOFs for length scaling)
  err = calculate_error_norm(
      *locerrvelnp_, get_dbc_map_extractor()->cond_map()->num_global_elements() +
                         pressure_row_map()->num_global_elements());
  errcond = calculate_error_norm(errorcond, numfsidbcdofs_);
  errother = calculate_error_norm(*errorother,
      pressure_row_map()->num_global_elements() +
          (get_dbc_map_extractor()->cond_map()->num_global_elements() - numfsidbcdofs_));

  // calculate L-inf-norms of temporal discretization errors
  locerrvelnp_->norm_inf(&errinf);
  errorcond.norm_inf(&errinfcond);
  errorother->norm_inf(&errinfother);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Adapter::FluidFSI::calculate_wall_shear_stresses()
{
  // get inputs
  std::shared_ptr<const Core::LinAlg::Vector<double>> trueresidual = fluidimpl_->true_residual();
  double dt = fluidimpl_->dt();

  // Get WSSManager
  std::shared_ptr<FLD::Utils::StressManager> stressmanager = fluidimpl_->stress_manager();

  // Since the WSS Manager cannot be initialized in the FluidImplicitTimeInt::init()
  // it is not so sure if the WSSManager is jet initialized. So let's be safe here..
  if (stressmanager == nullptr) FOUR_C_THROW("Call of StressManager failed!");
  if (not stressmanager->is_init()) FOUR_C_THROW("StressManager has not been initialized jet!");

  // Call StressManager to calculate WSS from residual
  std::shared_ptr<Core::LinAlg::Vector<double>> wss =
      stressmanager->get_wall_shear_stresses(*trueresidual, dt);

  return wss;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Adapter::FluidFSI::calculate_error_norm(
    const Core::LinAlg::Vector<double>& vec, const int numneglect) const
{
  double norm = 1.0e+12;

  vec.norm_2(&norm);

  if (vec.global_length() - numneglect > 0.0)
    norm /= sqrt((double)(vec.global_length() - numneglect));
  else
    norm = 0.0;

  return norm;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Adapter::FluidFSI::aux_method_order_of_accuracy() const
{
  if (auxintegrator_ == Inpar::FSI::timada_fld_none)
    return 0;
  else if (auxintegrator_ == Inpar::FSI::timada_fld_expleuler)
    return 1;
  else if (auxintegrator_ == Inpar::FSI::timada_fld_adamsbashforth2)
    return 2;
  else
  {
    FOUR_C_THROW("Unknown auxiliary time integration scheme for fluid field.");
    return 0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Adapter::FluidFSI::aux_method_lin_err_coeff_vel() const
{
  if (auxintegrator_ == Inpar::FSI::timada_fld_none)
    return 0.0;
  else if (auxintegrator_ == Inpar::FSI::timada_fld_expleuler)
    return 0.5;
  else if (auxintegrator_ == Inpar::FSI::timada_fld_adamsbashforth2)
  {
    // time step sizes of current and previous time step
    const double dtc = dt();
    const double dto = fluidimpl_->dt_previous();

    // leading error coefficient
    return (2 * dtc + 3 * dto) / (12 * dtc);
  }
  else
  {
    FOUR_C_THROW("Unknown auxiliary time integration scheme for fluid field.");
    return 0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Adapter::FluidFSI::get_tim_ada_err_order() const
{
  if (auxintegrator_ != Inpar::FSI::timada_fld_none)
  {
    if (methodadapt_ == ada_upward)
      return fluidimpl_->method_order_of_accuracy_vel();
    else
      return aux_method_order_of_accuracy();
  }
  else
  {
    FOUR_C_THROW(
        "Cannot return error order for adaptive time integration, since"
        "no auxiliary scheme has been chosen for the fluid field.");
    return 0.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::string Adapter::FluidFSI::get_tim_ada_method_name() const
{
  switch (auxintegrator_)
  {
    case Inpar::FSI::timada_fld_none:
    {
      return "none";
      break;
    }
    case Inpar::FSI::timada_fld_expleuler:
    {
      return "ExplicitEuler";
      break;
    }
    case Inpar::FSI::timada_fld_adamsbashforth2:
    {
      return "AdamsBashforth2";
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown auxiliary time integration scheme for fluid field.");
      return "";
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::setup_interface(const int nds_master)
{
  interface_->setup(*dis_, false, false, nds_master);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::build_inner_vel_map()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  maps.push_back(FluidWrapper::velocity_row_map());
  maps.push_back(interface()->other_map());
  maps.push_back(get_dbc_map_extractor()->other_map());
  innervelmap_ = Core::LinAlg::MultiMapExtractor::intersect_maps(maps);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Adapter::FluidFSI::update_slave_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& f)
{
  fluidimpl_->update_slave_dof(*f);
}

FOUR_C_NAMESPACE_CLOSE
