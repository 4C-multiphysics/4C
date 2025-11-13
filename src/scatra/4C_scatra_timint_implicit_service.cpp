// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_superconvergent_patch_recovery.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fluid_rotsym_periodicbc_utils.hpp"
#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_turbulence.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_turbulence_hit_scalar_forcing.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::calc_flux(const bool writetofile)
{
  switch (calcflux_domain_)
  {
    case Inpar::ScaTra::flux_diffusive:
    case Inpar::ScaTra::flux_total:
    {
      flux_domain_ = calc_flux_in_domain();
      break;
    }

    case Inpar::ScaTra::flux_none:
    {
      // do nothing
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid option for flux calculation inside domain!");
      break;
    }
  }

  switch (calcflux_boundary_)
  {
    case Inpar::ScaTra::flux_convective:
    case Inpar::ScaTra::flux_diffusive:
    case Inpar::ScaTra::flux_total:
    {
      // calculate normal flux vector field only for the user-defined boundary conditions:
      flux_boundary_ = calc_flux_at_boundary(writetofile);

      break;
    }

    case Inpar::ScaTra::flux_none:
    {
      // do nothing
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid option for flux calculation on boundary!");
      break;
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> ScaTra::ScaTraTimIntImpl::calc_flux_in_domain()
    const
{
  // extract dofrowmap from discretization
  const Core::LinAlg::Map& dofrowmap = *discret_->dof_row_map();

  // initialize global flux vectors
  auto flux = std::make_shared<Core::LinAlg::MultiVector<double>>(dofrowmap, 3);
  auto flux_projected = std::make_shared<Core::LinAlg::MultiVector<double>>(dofrowmap, 3);

  // parameter list for element evaluation
  Teuchos::ParameterList params;

  // set action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_flux_domain, params);

  // provide discretization with state vector
  discret_->set_state("phinp", *phinp_);

  // evaluate flux vector field inside the whole computational domain (e.g., for visualization of
  // particle path lines)
  {
    auto f0 = std::make_shared<Core::LinAlg::Vector<double>>(dofrowmap);
    auto f1 = std::make_shared<Core::LinAlg::Vector<double>>(dofrowmap);
    auto f2 = std::make_shared<Core::LinAlg::Vector<double>>(dofrowmap);
    discret_->evaluate(params, nullptr, nullptr, f0, f1, f2);
    (*flux)(0).update(1., *f0, 0.);
    (*flux)(1).update(1., *f1, 0.);
    (*flux)(2).update(1., *f2, 0.);
  }

  if (calcflux_domain_lumped_)
  {
    // vector for integrated shape functions
    std::shared_ptr<Core::LinAlg::Vector<double>> integrated_shape_functions =
        Core::LinAlg::create_vector(dofrowmap);

    // overwrite action for elements
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::integrate_shape_functions, params);

    // integrate shape functions
    auto dofids = std::make_shared<Core::LinAlg::IntSerialDenseVector>(num_dof_per_node());
    dofids->putScalar(1);  // integrate shape functions for all dofs
    params.set("dofids", dofids);
    discret_->evaluate(params, nullptr, nullptr, integrated_shape_functions, nullptr, nullptr);

    // insert values into final flux vector for visualization
    // We do not solve a global, linear system of equations (exact L2 projection with good
    // consistency), but perform mass matrix lumping, i.e., we divide by the values of the
    // integrated shape functions

    // the mass matrix lumping was previously conducted via the
    // ReciprocalMultiply. This realized a Hadamard product of two MultiVectors:
    // this = ScalarThis * this + ScalarAB * B / A (Important: element-wise)

    // start by creating the inverse
    Core::LinAlg::MultiVector<double> inverse_of_integrated_shape_functions(
        integrated_shape_functions->get_map(), 1, true);
    {
      const double* values = integrated_shape_functions->as_multi_vector().get_values();

      // perform a zero-safe reciprocal of the local vector
      for (int i = 0; i < integrated_shape_functions->local_length(); ++i)
        inverse_of_integrated_shape_functions.get_values()[i] =
            (std::abs(values[i]) > std::numeric_limits<double>::epsilon())
                ? (1.0 / values[i])
                : std::numeric_limits<double>::max();
    }

    // Broadcast inverse_of_integrated_shape_functions across column to match the flux
    Core::LinAlg::MultiVector<double> inverse_of_integrated_shape_functions_bcast(
        inverse_of_integrated_shape_functions.get_map(), flux->num_vectors(), false);
    for (int j = 0; j < flux->num_vectors(); ++j)
    {
      // copy column 0 into j-th column
      inverse_of_integrated_shape_functions_bcast(j).update(
          1.0, inverse_of_integrated_shape_functions(0), 0.0);
    }

    // now apply the elementwise-multiplication
    flux_projected->multiply(1.0, *flux, inverse_of_integrated_shape_functions_bcast, 0.0);
  }

  else
  {
    // solve global, linear system of equations without lumping global mass matrix
    // overwrite action for elements
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::calc_mass_matrix, params);

    // initialize global mass matrix
    auto massmatrix = std::make_shared<Core::LinAlg::SparseMatrix>(dofrowmap, 27, false);

    // call loop over elements
    discret_->evaluate(params, massmatrix, nullptr);

    // finalize global mass matrix
    massmatrix->complete();

    // compute flux vector field
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver_->solve_with_multi_vector(massmatrix, flux_projected, flux, solver_params);

    // reset solver
    solver_->reset();
  }

  return flux_projected;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> ScaTra::ScaTraTimIntImpl::calc_flux_at_boundary(
    const bool writetofile)
{
  // The normal flux calculation is based on the idea proposed in
  // GRESHO ET AL.,
  // "THE CONSISTENT GALERKIN FEM FOR COMPUTING DERIVED BOUNDARY
  // QUANTITIES IN THERMAL AND/OR FLUIDS PROBLEMS",
  // INTERNATIONAL JOURNAL FOR NUMERICAL METHODS IN FLUIDS, VOL. 7, 371-394 (1987)

  // empty vector for (normal) mass or heat flux vectors (always 3D)
  std::shared_ptr<Core::LinAlg::MultiVector<double>> flux =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*discret_->dof_row_map(), 3, true);

  // determine the averaged normal vector field for indicated boundaries
  // used for the output of the normal flux as a vector field
  // is computed only once; for ALE formulation recalculation is necessary
  if ((normals_ == nullptr) or isale_)
    normals_ = compute_normal_vectors(std::vector<std::string>(1, "ScaTraFluxCalc"));

  if (calcflux_boundary_ == Inpar::ScaTra::flux_convective)
  {
    // zero out trueresidual vector -> we do not need this info
    trueresidual_->put_scalar(0.0);
  }
  else
  {
    // was the residual already prepared?
    if ((solvtype_ != Inpar::ScaTra::solvertype_nonlinear) and (lastfluxoutputstep_ != step_))
    {
      lastfluxoutputstep_ = step_;

      // For nonlinear problems we already have the actual residual vector
      // from the last convergence test!
      // For linear problems we have to compute this information first, since
      // the residual (w.o. Neumann boundary) has not been computed after the last solve!

      // zero out matrix entries
      sysmat_->zero();

      // zero out residual vector
      residual_->put_scalar(0.0);

      Teuchos::ParameterList eleparams;
      // action for elements
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
          "action", ScaTra::Action::calc_mat_and_rhs, eleparams);

      // other parameters that might be needed by the elements
      // the resulting system has to be solved incrementally
      set_element_time_parameter(true);

      // AVM3 separation for incremental solver: get fine-scale part of scalar
      if (incremental_ and (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
                               turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales))
        avm3_separation();

      // add element parameters according to time-integration scheme
      // we want to have an incremental solver!!
      add_time_integration_specific_vectors(true);

      // add element parameters according to specific problem
      add_problem_specific_parameters_and_vectors(eleparams);

      {
        // call standard loop over elements
        discret_->evaluate(eleparams, sysmat_, nullptr, residual_, nullptr, nullptr);
      }

      // scaling to get true residual vector for all time integration schemes
      trueresidual_->update(residual_scaling(), *residual_, 0.0);

      // undo potential changes
      set_element_time_parameter();

    }  // if ((solvtype_!=Inpar::ScaTra::solvertype_nonlinear) && (lastfluxoutputstep_ != step_))
  }  // if (writeflux_==Inpar::ScaTra::flux_convective_boundary)

  // if desired add the convective flux contribution
  // to the trueresidual_ now.
  if (calcflux_boundary_ == Inpar::ScaTra::flux_total or
      calcflux_boundary_ == Inpar::ScaTra::flux_convective)
  {
    std::vector<const Core::Conditions::Condition*> cond;
    discret_->get_condition("ScaTraFluxCalc", cond);

    Teuchos::ParameterList params;
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
        "action", ScaTra::BoundaryAction::add_convective_mass_flux, params);

    // add element parameters according to time-integration scheme
    add_time_integration_specific_vectors();

    // call loop over boundary elements and add integrated fluxes to trueresidual_
    discret_->evaluate_condition(params, trueresidual_, "ScaTraFluxCalc");
  }

  // vector for effective flux over all defined boundary conditions
  // maximal number of fluxes  (numscal+1 -> ionic species + potential) is generated
  // for OUTPUT standard -> last entry is not used
  std::vector<double> normfluxsum(num_dof_per_node());

  std::vector<const Core::Conditions::Condition*> cond;
  discret_->get_condition("ScaTraFluxCalc", cond);

  // safety check
  if (!cond.size())
    FOUR_C_THROW("Flux output requested without corresponding boundary condition specification!");

  if (myrank_ == 0)
  {
    std::cout << '\n';
    std::cout << "Normal fluxes at boundary 'ScaTraFluxCalc' on discretization '"
              << discret_->name() << "':" << '\n';
    std::cout
        << "+----+-----+-------------------------+------------------+--------------------------+"
        << '\n';
    std::cout
        << "| ID | DOF | Integral of normal flux | Area of boundary | Mean normal flux density |"
        << '\n';
  }

  // first, add to all conditions of interest a ConditionID
  for (int condid = 0; condid < (int)cond.size(); condid++)
  {
    // is there already a ConditionID?
    const auto* CondID = cond[condid]->parameters().get_if<int>("ConditionID");
    if (CondID)
    {
      if ((*CondID) != condid)
        FOUR_C_THROW("Condition 'ScaTraFluxCalc' has non-matching ConditionID!");
    }
    else
    {
      // let's add a ConditionID
      const_cast<Core::Conditions::Condition*>(cond[condid])
          ->parameters()
          .add("ConditionID", condid);
    }
  }

  // now we evaluate the conditions and separate via ConditionID
  for (int icond = 0; icond < static_cast<int>(cond.size()); ++icond)
  {
    // extract dofrowmap associated with current boundary segment
    const Core::LinAlg::Map& dofrowmap = *flux_boundary_maps_->map(icond + 1);

    // extract part of true residual vector associated with current boundary segment
    const std::shared_ptr<Core::LinAlg::Vector<double>> trueresidual_boundary =
        flux_boundary_maps_->extract_vector(*trueresidual_, icond + 1);

    // initialize vector for nodal values of normal boundary fluxes
    std::shared_ptr<Core::LinAlg::Vector<double>> normalfluxes =
        Core::LinAlg::create_vector(dofrowmap);

    // create parameter list for boundary elements
    Teuchos::ParameterList params;

    // initialize variable for value of boundary integral
    double boundaryint(-1.);

    // compute nodal values of normal boundary fluxes
    if (calcflux_boundary_lumped_)
    {
      // lump boundary mass matrix instead of solving a small, linear system of equations
      // calculate integral of shape functions over indicated boundary and its area
      params.set("area", 0.0);
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
          "action", ScaTra::BoundaryAction::integrate_shape_functions, params);

      // create vector (+ initialization with zeros)
      const std::shared_ptr<Core::LinAlg::Vector<double>> integratedshapefunc =
          Core::LinAlg::create_vector(dofrowmap);

      // call loop over elements
      discret_->evaluate_condition(params, integratedshapefunc, "ScaTraFluxCalc", icond);

      // compute normal boundary fluxes
      for (int idof = 0; idof < dofrowmap.num_my_elements(); ++idof)
        (*normalfluxes).get_values()[idof] = trueresidual_boundary->local_values_as_span()[idof] /
                                             integratedshapefunc->local_values_as_span()[idof];

      // get value of boundary integral on this processor
      boundaryint = params.get<double>("area");
    }

    else
    {
      // solve small, linear system of equations without lumping boundary mass matrix
      // add action to parameter list
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
          "action", ScaTra::BoundaryAction::calc_mass_matrix, params);

      // initialize boundary mass matrix
      std::shared_ptr<Core::LinAlg::SparseMatrix> massmatrix_boundary =
          std::make_shared<Core::LinAlg::SparseMatrix>(dofrowmap, 27, false);

      // call loop over elements
      discret_->evaluate_condition(
          params, massmatrix_boundary, nullptr, nullptr, nullptr, nullptr, "ScaTraFluxCalc", icond);

      // finalize boundary mass matrix
      massmatrix_boundary->complete();

      // compute normal boundary fluxes
      Core::LinAlg::SolverParams solver_params;
      solver_params.refactor = true;
      solver_params.reset = true;
      solver_->solve(massmatrix_boundary, normalfluxes, trueresidual_boundary, solver_params);

      // reset solver
      solver_->reset();

      // overwrite action in parameter list
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
          "action", ScaTra::BoundaryAction::calc_boundary_integral, params);

      // initialize one-component result vector for value of boundary integral
      Core::LinAlg::SerialDenseVector boundaryint_vector(1);

      // compute value of boundary integral
      discret_->evaluate_scalars(params, boundaryint_vector, "ScaTraFluxCalc", icond);

      // extract value of boundary integral
      boundaryint = (boundaryint_vector)(0);
    }

    // maximal number of fluxes
    std::vector<double> normfluxintegral(num_dof_per_node());

    // insert values into final flux vector for visualization
    for (int lnodid = 0; lnodid < discret_->num_my_row_nodes(); ++lnodid)
    {
      Core::Nodes::Node* actnode = discret_->l_row_node(lnodid);
      for (int idof = 0; idof < discret_->num_dof(0, actnode); ++idof)
      {
        const int dofgid = discret_->dof(0, actnode, idof);
        if (dofrowmap.my_gid(dofgid))
        {
          const int doflid = dofrowmap.lid(dofgid);

          // compute integral value for every degree of freedom
          normfluxintegral[idof] += trueresidual_boundary->local_values_as_span()[doflid];

          // care for the slave nodes of rotationally symm. periodic boundary conditions
          double rotangle(0.0);
          bool havetorotate = FLD::is_slave_node_of_rot_sym_pbc(*discret_, actnode, rotangle);

          // do not insert slave node values here, since they would overwrite the
          // master node values owning the same dof
          // (rotation of slave node vectors is performed later during output)
          if (not havetorotate)
          {
            // for visualization, we plot the normal flux with
            // outward pointing normal vector
            for (int idim = 0; idim < 3; idim++)
            {
              auto& normalcomp = (*normals_)(idim);
              double normalveccomp = normalcomp.local_values_as_span()[lnodid];
              flux->replace_global_value(
                  dofgid, idim, normalfluxes->local_values_as_span()[doflid] * normalveccomp);
            }
          }
        }
      }
    }

    // care for the parallel case
    std::vector<double> parnormfluxintegral(num_dof_per_node());
    parnormfluxintegral = Core::Communication::sum_all(normfluxintegral, discret_->get_comm());
    double parboundaryint = 0.0;
    parboundaryint = Core::Communication::sum_all(boundaryint, discret_->get_comm());

    for (int idof = 0; idof < num_dof_per_node(); ++idof)
    {
      // print out results
      if (myrank_ == 0)
      {
        printf("| %2d | %2d  |       %+10.4E       |    %10.4E    |        %+10.4E       |\n",
            icond, idof, parnormfluxintegral[idof], parboundaryint,
            parnormfluxintegral[idof] / parboundaryint);
      }
      normfluxsum[idof] += parnormfluxintegral[idof];
    }

    // statistics section for normfluxintegral
    if (do_boundary_flux_statistics())
    {
      // add current flux value to the sum!
      (*sumnormfluxintegral_)[icond] += parnormfluxintegral[0];  // only first scalar!
      int samstep = step_ - samstart_ + 1;

      // dump every dumperiod steps (i.e., write to screen)
      bool dumpit(false);
      if (dumperiod_ == 0)
      {
        dumpit = true;
      }
      else
      {
        if (samstep % dumperiod_ == 0) dumpit = true;
      }

      if (dumpit)
      {
        double meannormfluxintegral = (*sumnormfluxintegral_)[icond] / samstep;
        // dump statistical results
        if (myrank_ == 0)
        {
          printf("| %2d | Mean normal-flux integral (step %5d -- step %5d) :   %12.5E |\n", icond,
              samstart_, step_, meannormfluxintegral);
        }
      }
    }

    // print out results to file as well (only if really desired)
    if ((myrank_ == 0) and writetofile)
    {
      const std::string fname = problem_->output_control_file()->file_name() +
                                ".boundaryflux_ScaTraFluxCalc_" + std::to_string(icond) +
                                discret_->name() + ".txt";

      std::ofstream f;
      if (step() <= 1)
      {
        f.open(fname.c_str(), std::fstream::trunc);
        f << "#| ID | Step | Time | Area of boundary |";
        for (int idof = 0; idof < num_dof_per_node(); ++idof)
        {
          f << " Integral of normal flux " << idof << " | Mean normal flux density " << idof
            << " |";
        }
        f << "\n";
      }
      else
      {
        f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
      }

      f << icond << " " << step() << " " << time() << " " << parboundaryint << " ";
      for (int idof = 0; idof < num_dof_per_node(); ++idof)
      {
        f << parnormfluxintegral[idof] << " " << parnormfluxintegral[idof] / parboundaryint << " ";
      }
      f << "\n";
      f.flush();
      f.close();
    }  // write to file

  }  // loop over condid

  if (myrank_ == 0)
  {
    std::cout
        << "+----+-----+-------------------------+------------------+--------------------------+"
        << '\n';
  }

  // print out the accumulated normal flux over all indicated boundaries
  // the accumulated normal flux over all indicated boundaries is not printed for numscal+1
  if (myrank_ == 0)
  {
    for (int idof = 0; idof < num_scal(); ++idof)
    {
      printf("| Sum of all normal flux boundary integrals for scalar %d: %+10.5E             |\n",
          idof, normfluxsum[idof]);
    }
    std::cout
        << "+----------------------------------------------------------------------------------+"
        << '\n';
  }

  return flux;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::calc_initial_time_derivative()
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + calculate initial time derivative");

  // initial screen output
  if (myrank_ == 0)
  {
    std::cout
        << "SCATRA: calculating initial time derivative of state variables on discretization \""
        << discret_->name().c_str() << "\" (step " << step() << ", time " << time() << ") ... ... ";
  }

  // evaluate Dirichlet and Neumann boundary conditions at time t = 0 to ensure consistent
  // computation of initial time derivative vector Dirichlet boundary conditions should be
  // consistent with initial field
  apply_dirichlet_bc(time_, phinp_, nullptr);
  compute_intermediate_values();
  apply_neumann_bc(neumann_loads_);

  // In most cases, the history vector is entirely zero at this point, since this routine is called
  // before the first time step. However, in levelset simulations with reinitialization, this
  // routine might be called before every single time step. For this reason, the history vector
  // needs to be manually set to zero here and restored at the end of this routine.
  std::shared_ptr<Core::LinAlg::Vector<double>> hist = hist_;
  hist_ = zeros_;

  // In a first step, we assemble the standard global system of equations.
  assemble_mat_and_rhs();

  // In a second step, we need to modify the assembled system of equations, since we want to solve
  // M phidt^0 = f^n - K\phi^n - C(u_n)\phi^n
  // In particular, we need to replace the global system matrix by a global mass matrix,
  // and we need to remove all transient contributions associated with time discretization from the
  // global residual vector.

  // reset global system matrix
  sysmat_->zero();

  // create and fill parameter list for elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_initial_time_deriv, eleparams);
  add_problem_specific_parameters_and_vectors(eleparams);

  // add state vectors according to time integration scheme
  add_time_integration_specific_vectors();

  // modify global system of equations as explained above
  discret_->evaluate(eleparams, sysmat_, residual_);

  // apply condensation to global system of equations if necessary
  strategy_->condense_mat_and_rhs(sysmat_, residual_, true);

  // finalize assembly of global mass matrix
  sysmat_->complete();

  // apply artificial Dirichlet boundary conditions to system of equations to hold initial values of
  // auxiliary degrees of freedom (= non-transported scalars) constant when solving for initial time
  // derivatives of primary degrees of freedom (= transported scalars)
  if (splitter_ != nullptr)
    Core::LinAlg::apply_dirichlet_to_system(
        *sysmat_, *phidtnp_, *residual_, *zeros_, *(splitter_->cond_map()));

  // solve global system of equations for initial time derivative of state variables
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(sysmat_, phidtnp_, residual_, solver_params);

  // ToDo: Impose initial time derivatives resulting from Dirichlet boundary conditions.
  // At the moment, the initial time derivatives do not take account of time-dependent
  // Dirichlet boundary conditions. So for example, if you consider a simple setup with
  // one single finite element exhibiting a zero initial scalar field, and if you then
  // ramp up the scalar inside the entire element from zero to one linearly in time via
  // a time-dependent Dirichlet boundary condition, you will currently get zero initial
  // time derivatives (although they should be positive) and, as a consequence, there will
  // be unphysical temporal oscillations in the time derivatives from time step to time
  // step. The following, currently commented line of code would solve this problem, but
  // also cause many existing test cases to fail. There are intricate cases where the
  // imposed initial scalar field is not consistent with Dirichlet boundary conditions,
  // and it is still not clear how to handle these cases. One should have a closer look
  // at this problem and try to activate the following line of code at some point.

  // apply_dirichlet_bc(time_,nullptr,phidtnp_);

  // copy solution
  phidtn_->update(1.0, *phidtnp_, 0.0);

  // reset global system matrix and its graph, since we solved a very special problem with a special
  // sparsity pattern
  sysmat_->reset();

  // reset solver
  solver_->reset();

  // reset true residual vector computed during assembly of the standard global system of equations,
  // since not yet needed
  trueresidual_->put_scalar(0.0);

  // restore history vector as explained above
  hist_ = hist;

  // final screen output
  if (myrank_ == 0) std::cout << "done!" << '\n';
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::compute_density()
{
  // loop over all nodes owned by current processor
  for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); ++lnodeid)
  {
    // get current node
    Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);

    // get associated degrees of freedom
    std::vector<int> nodedofs = discret_->dof(0, lnode);
    const int numdof = static_cast<int>(nodedofs.size());

    // initialize nodal density value
    double density = 1.;

    // loop over all transported scalars
    for (int k = 0; k < num_scal(); ++k)
    {
      /*
        //                  k=NumScal()-1
        //          /       ----                         \
        //         |        \                            |
        // rho_0 * | 1 +    /      alpha_k * (c_k - c_0) |
        //         |        ----                         |
        //          \       k=0                          /
        //
        // For use of molar mass M_k:  alpha_k = M_k/rho_0
       */

      // global and local dof ID
      const int globaldofid = nodedofs[k];
      const int localdofid = phiafnp()->get_map().lid(globaldofid);
      if (localdofid < 0) FOUR_C_THROW("Local dof ID not found in dof map!");

      // add contribution of scalar k to nodal density value
      density += densific_[k] * ((*phiafnp()).local_values_as_span()[localdofid] - c0_[k]);
    }

    // insert nodal density value into global density vector
    // note that the density vector has been initialized with the dofrowmap of the discretization
    // in case there is more than one dof per node, the nodal density value is inserted into the
    // position of the last dof this way, all nodal density values will be correctly extracted in
    // the fluid algorithm
    const int globaldofid = nodedofs[numdof - 1];
    const int localdofid = phiafnp()->get_map().lid(globaldofid);
    if (localdofid < 0) FOUR_C_THROW("Local dof ID not found in dof map!");

    densafnp_->replace_local_value(localdofid, density);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::compute_time_derivative()
{
  // compute time derivatives of discrete state variables associated with meshtying strategy
  strategy_->compute_time_derivative();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::output_total_and_mean_scalars(const int num)
{
  if (outputscalars_ != Inpar::ScaTra::outputscalars_none)
  {
    if (outputscalarstrategy_ == nullptr) FOUR_C_THROW("output strategy was not initialized!");
    outputscalarstrategy_->output_total_and_mean_scalars(this, num);
  }  // if(outputscalars_)
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::output_domain_or_boundary_integrals(const std::string& condstring)
{
  // check whether output is applicable
  if ((computeintegrals_ == Inpar::ScaTra::computeintegrals_initial and step() == 0) or
      computeintegrals_ == Inpar::ScaTra::computeintegrals_repeated)
  {
    if (outputdomainintegralstrategy_ == nullptr)
      FOUR_C_THROW("output strategy was not initialized!");

    outputdomainintegralstrategy_->evaluate_integrals_and_print_results(this, condstring);
  }  // check whether output is applicable
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::surface_permeability(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // time measurement: evaluate condition 'SurfacePermeability'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ScaTraCoupling'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_fs3i_surface_permeability, condparams);

  // add element parameters according to time-integration scheme
  add_time_integration_specific_vectors();

  // replace phinp by the mean of phinp if this has been set
  if (mean_conc_ != nullptr)
  {
    // TODO: (thon) this is not a nice way of using the mean concentration instead of phinp!
    // Note: meanconc_ is not cleared by calling 'discret_->ClearState()', hence if you
    // don't want to do this replacement here any more call 'ClearMeanConcentration()'
    discret_->set_state("phinp", *mean_conc_);

    if (myrank_ == 0)
      std::cout << "Replacing 'phinp' by 'meanconc_' in the evaluation of the surface permeability"
                << '\n';
  }

  if (membrane_conc_ == nullptr)
    FOUR_C_THROW("Membrane concentration must already been saved before calling this function!");
  discret_->set_state("MembraneConcentration", *membrane_conc_);

  // test if all necessary ingredients had been set
  if (not discret_->has_state(nds_wall_shear_stress(), "WallShearStress"))
    FOUR_C_THROW(
        "WSS must already been set into one of the secondary dofset before calling this function!");

  // Evaluate condition
  discret_->evaluate_condition(
      condparams, matrix, nullptr, rhs, nullptr, nullptr, "ScaTraCoupling");

  matrix->complete();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::kedem_katchalsky(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // time measurement: evaluate condition 'SurfacePermeability'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'ScaTraCoupling'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_fps3i_surface_permeability, condparams);

  // add element parameters according to time-integration scheme
  add_time_integration_specific_vectors();

  // test if all necessary ingredients for the second Kedem-Katchalsky equations had been set
  if (not discret_->has_state(nds_wall_shear_stress(), "WallShearStress"))
    FOUR_C_THROW(
        "WSS must already been set into one of the secondary dofset before calling this function!");

  if (not discret_->has_state(nds_pressure(), "Pressure"))
  {
    FOUR_C_THROW(
        "Pressure must already been set into one of the secondary dofset before calling this "
        "function!");
  }

  if (membrane_conc_ == nullptr)
    FOUR_C_THROW("Membrane concentration must already been saved before calling this function!");
  discret_->set_state("MembraneConcentration", *membrane_conc_);

  // Evaluate condition
  discret_->evaluate_condition(
      condparams, matrix, nullptr, rhs, nullptr, nullptr, "ScaTraCoupling");

  matrix->complete();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::add_flux_approx_to_parameter_list(Teuchos::ParameterList& p) const
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> flux = calc_flux_in_domain();

  // post_ensight does not support multivectors based on the dofmap
  // for now, I create single vectors that can be handled by the filters

  // get the noderowmap
  const Core::LinAlg::Map* noderowmap = discret_->node_row_map();
  Core::LinAlg::MultiVector<double> fluxk(*noderowmap, 3, true);
  for (int k = 0; k < num_scal(); ++k)
  {
    const std::string name = "flux_phi_" + std::to_string(k);
    for (int i = 0; i < fluxk.local_length(); ++i)
    {
      Core::Nodes::Node* actnode = discret_->l_row_node(i);
      int dofgid = discret_->dof(0, actnode, k);
      fluxk.replace_local_value(
          i, 0, ((*flux)(0)).local_values_as_span()[(flux->get_map()).lid(dofgid)]);
      fluxk.replace_local_value(
          i, 1, ((*flux)(1)).local_values_as_span()[(flux->get_map()).lid(dofgid)]);
      fluxk.replace_local_value(
          i, 2, ((*flux)(2)).local_values_as_span()[(flux->get_map()).lid(dofgid)]);
    }

    auto tmp = std::make_shared<Core::LinAlg::MultiVector<double>>(
        *discret_->node_col_map(), fluxk.num_vectors());
    Core::LinAlg::export_to(fluxk, *tmp);
    p.set(name, tmp);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> ScaTra::ScaTraTimIntImpl::compute_normal_vectors(
    const std::vector<std::string>& condnames) const
{
  // create vectors for x,y and z component of average normal vector field
  // get noderowmap of discretization
  const Core::LinAlg::Map* noderowmap = discret_->node_row_map();
  std::shared_ptr<Core::LinAlg::MultiVector<double>> normal =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*noderowmap, 3, true);

  // set action for elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_normal_vectors, eleparams);
  eleparams.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("normal vectors", normal);

  // loop over all intended types of conditions
  for (const auto& condname : condnames)
  {
    discret_->evaluate_condition(eleparams, condname);
  }

  // the normal vector field is not properly scaled up to now. We do this here
  int numrownodes = discret_->num_my_row_nodes();
  auto& xcomp = (*normal)(0);
  auto& ycomp = (*normal)(1);
  auto& zcomp = (*normal)(2);
  for (int i = 0; i < numrownodes; ++i)
  {
    double x = xcomp.local_values_as_span()[i];
    double y = ycomp.local_values_as_span()[i];
    double z = zcomp.local_values_as_span()[i];
    double norm = sqrt(x * x + y * y + z * z);
    // form the unit normal vector
    if (norm > 1e-15)
    {
      normal->replace_local_value(i, 0, x / norm);
      normal->replace_local_value(i, 1, y / norm);
      normal->replace_local_value(i, 2, z / norm);
    }
  }

  return normal;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::compute_null_space_if_necessary() const
{
  // extract solver parameters
  Teuchos::ParameterList& solverparams = solver_->params();

  // compute point-based null space information if applicable
  if (params_->get<bool>("NULLSPACE_POINTBASED"))
  {
    // MueLu preconditioner
    if (solverparams.isSublist("MueLu Parameters"))
    {
      // extract and fill parameter list for MueLu preconditioner
      Teuchos::ParameterList& mllist = solverparams.sublist("MueLu Parameters", true);
      mllist.set("PDE equations", 1);

      std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
          std::make_shared<Core::LinAlg::MultiVector<double>>(*(discret_->dof_row_map()), 1, true);
      nullspace->put_scalar(1.0);

      mllist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspace);

      std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates =
          extract_node_coordinates(*discret_);

      mllist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates", coordinates);
    }

    else
    {
      FOUR_C_THROW(
          "Point-based null space calculation currently only implemented for MueLu "
          "preconditioner!");
    }
  }

  else
  {
    // compute standard, vector-based null space information if applicable
    Core::FE::compute_null_space_if_necessary(*discret_, solverparams, true);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::compute_neumann_inflow(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // time measurement: evaluate condition 'Neumann inflow'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'TransportNeumannInflow'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_Neumann_inflow, condparams);

  // add element parameters and vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // set thermodynamic pressure in case of loma
  add_problem_specific_parameters_and_vectors(condparams);

  std::string condstring("TransportNeumannInflow");
  discret_->evaluate_condition(condparams, matrix, nullptr, rhs, nullptr, nullptr, condstring);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::evaluate_convective_heat_transfer(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // time measurement: evaluate condition 'TransportThermoConvections'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'TransportThermoConvections'");

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_convective_heat_transfer, condparams);

  // add element parameters and vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  std::string condstring("TransportThermoConvections");
  discret_->evaluate_condition(condparams, matrix, nullptr, rhs, nullptr, nullptr, condstring);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::output_lin_solver_stats(const Core::LinAlg::Solver& solver,
    const double& time, const int& step, const int& iteration, const int& size)
{
  // extract communicator
  MPI_Comm comm = solver.get_comm();

  // write performance statistics to file
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    // set file name
    std::string filename(
        Global::Problem::instance()->output_control_file()->file_name() + ".lin_solver_stats.csv");

    // open file in appropriate mode and write header at beginning
    std::ofstream file;
    if (step == 1 and iteration == 1)
    {
      file.open(filename.c_str(), std::fstream::trunc);
      file << "Step,NewtonRaphsonIteration,SystemSize,SolverIterations,SolverTime" << '\n';
    }
    else
    {
      file.open(filename.c_str(), std::fstream::app);
    }

    // write results
    file << step << "," << iteration << "," << size << "," << solver.get_num_iters() << ","
         << std::setprecision(16) << std::scientific << time << '\n';

    // close file
    file.close();
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::output_nonlin_solver_stats(
    const int& iterations, const double& time, const int& step, MPI_Comm comm)
{
  // write performance statistics to file
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    // set file name
    std::string filename(Global::Problem::instance()->output_control_file()->file_name() +
                         ".nonlin_solver_stats.csv");

    // open file in appropriate mode and write header at beginning
    std::ofstream file;
    if (step == 1)
    {
      file.open(filename.c_str(), std::fstream::trunc);
      file << "Step,NonlinSolverIterations,NonlinSolverTime" << '\n';
    }
    else
    {
      file.open(filename.c_str(), std::fstream::app);
    }

    // write results
    file << step << "," << iterations << "," << std::setprecision(16) << std::scientific << time
         << '\n';

    // close file
    file.close();
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::output_to_gmsh(const int step, const double time) const
{
  // turn on/off screen output for writing process of Gmsh postprocessing file
  const bool screen_out = true;

  // create Gmsh postprocessing file
  const std::string filename = Core::IO::Gmsh::get_new_file_name_and_delete_old_files(
      "solution_field_scalar", disc_writer()->output().file_name(), step, 500, screen_out,
      Core::Communication::my_mpi_rank(discret_->get_comm()));
  std::ofstream gmshfilecontent(filename.c_str());
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "Phinp \" {" << '\n';
    // draw scalar field 'Phinp' for every element
    Core::IO::Gmsh::scalar_field_to_gmsh(*discret_, phinp_, gmshfilecontent);
    gmshfilecontent << "};" << '\n';
  }
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "Convective Velocity \" {" << '\n';

    // extract convective velocity from discretization
    std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
        discret_->get_state(nds_vel(), "convective velocity field");
    if (convel == nullptr)
      FOUR_C_THROW("Cannot extract convective velocity field from discretization");

    // draw vector field 'Convective Velocity' for every element
    Core::IO::Gmsh::vector_field_dof_based_to_gmsh(*discret_, convel, gmshfilecontent, nds_vel());
    gmshfilecontent << "};" << '\n';
  }
  gmshfilecontent.close();
  if (screen_out) std::cout << " done" << '\n';
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::write_restart() const
{
  // step number and time (only after that data output is possible)
  output_->new_step(step_, time_);

  // solution
  output_->write_vector("phinp", phinp_);

  // output restart information associated with mesh tying strategy
  strategy_->write_restart();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::collect_output_flux_data(
    std::shared_ptr<Core::LinAlg::MultiVector<double>> flux, const std::string& fluxtype)
{
  // safety checks
  if (flux == nullptr)
    FOUR_C_THROW("Null pointer for flux vector output. Output() called before Update() ??");
  if (fluxtype != "domain" and fluxtype != "boundary")
    FOUR_C_THROW("Unknown flux type. Must be either 'domain' or 'boundary'!");

  // WORK-AROUND FOR NURBS DISCRETIZATIONS
  // using noderowmap is problematic. Thus, we do not add normal vectors
  // to the scalar result field (scalar information is enough anyway)
  auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*discret_));
  if (nurbsdis != nullptr)
  {
    auto normalflux = (*flux)(0);
    std::vector<std::optional<std::string>> context(nsd_, "normalflux");
    visualization_writer_->append_result_data_vector_with_context(
        normalflux, Core::IO::OutputEntity::dof, context);
    return;  // leave here
  }

  // get the noderowmap
  const Core::LinAlg::Map* noderowmap = discret_->node_row_map();
  auto fluxk = Core::LinAlg::MultiVector<double>(*noderowmap, 3, true);
  for (int writefluxid : *writefluxids_)
  {
    std::string name = "flux_" + fluxtype + "_phi_" + std::to_string(writefluxid);
    for (int i = 0; i < fluxk.local_length(); ++i)
    {
      Core::Nodes::Node* actnode = discret_->l_row_node(i);
      int dofgid = discret_->dof(0, actnode, writefluxid - 1);
      // get value for each component of flux vector
      double xvalue = (*flux)(0).local_values_as_span()[(flux->get_map()).lid(dofgid)];
      double yvalue = (*flux)(1).local_values_as_span()[(flux->get_map()).lid(dofgid)];
      double zvalue = (*flux)(2).local_values_as_span()[(flux->get_map()).lid(dofgid)];
      // care for the slave nodes of rotationally symm. periodic boundary conditions
      double rotangle(0.0);  // already converted to radians
      bool havetorotate = FLD::is_slave_node_of_rot_sym_pbc(*discret_, actnode, rotangle);
      if (havetorotate)
      {
        double xvalue_rot = (xvalue * cos(rotangle)) - (yvalue * sin(rotangle));
        double yvalue_rot = (xvalue * sin(rotangle)) + (yvalue * (cos(rotangle)));
        xvalue = xvalue_rot;
        yvalue = yvalue_rot;
      }
      // insert values
      fluxk.replace_local_value(i, 0, xvalue);
      fluxk.replace_local_value(i, 1, yvalue);
      fluxk.replace_local_value(i, 2, zvalue);
    }
    std::vector<std::optional<std::string>> context(fluxk.num_vectors(), name);
    visualization_writer_->append_result_data_vector_with_context(
        fluxk, Core::IO::OutputEntity::node, context);
  }
}

// TODO: SCATRA_ELE_CLEANING: BIOFILM
// This action is not called at element level!!
// Can it be integrated in CalcFlux_domain?
/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::output_integr_reac(const int num) const
{
  if (outintegrreac_)
  {
    if (!discret_->filled()) FOUR_C_THROW("fill_complete() was not called");
    if (!discret_->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

    // set scalar values needed by elements
    discret_->set_state("phinp", *phinp_);
    // set action for elements
    Teuchos::ParameterList eleparams;
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::calc_integr_reaction, eleparams);
    std::shared_ptr<std::vector<double>> myreacnp =
        std::make_shared<std::vector<double>>(num_scal(), 0.0);
    eleparams.set<std::shared_ptr<std::vector<double>>>("local reaction integral", myreacnp);

    discret_->evaluate(eleparams);

    myreacnp = eleparams.get<std::shared_ptr<std::vector<double>>>("local reaction integral");
    // global integral of reaction terms
    std::vector<double> intreacterm(num_scal(), 0.0);
    for (int k = 0; k < num_scal(); ++k)
      intreacterm[k] = Core::Communication::sum_all(((*myreacnp)[k]), phinp_->get_map().get_comm());

    // print out values
    if (myrank_ == 0)
    {
      for (int k = 0; k < num_scal(); k++)
      {
        std::cout << "Total reaction (r_" << k << "): " << std::setprecision(9) << intreacterm[k]
                  << '\n';
      }
    }

    // print out results to file as well
    if (myrank_ == 0)
    {
      std::stringstream number;
      number << num;
      const std::string fname =
          problem_->output_control_file()->file_name() + number.str() + ".integrreacvalues.txt";

      std::ofstream f;
      if (step() <= 1)
      {
        f.open(fname.c_str(), std::fstream::trunc);
        f << "#| Step | Time ";
        for (int k = 0; k < num_scal(); k++)
        {
          f << "| Total reaction (r_" << k << ") ";
        }
        f << "\n";
      }
      else
      {
        f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
      }

      f << step() << " " << time() << " ";
      for (int k = 0; k < num_scal(); k++)
      {
        f << std::setprecision(9) << intreacterm[k] << " ";
      }
      f << "\n";
      f.flush();
      f.close();
    }

  }  // if(outintegrreac_)
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::avm3_preparation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // create normalized all-scale subgrid-diffusivity matrix
  sysmat_sd_->zero();

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements, time factor and stationary flag
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_subgrid_diffusivity_matrix, eleparams);

  // add element parameters according to time-integration scheme
  add_time_integration_specific_vectors();

  // call loop over elements
  discret_->evaluate(eleparams, sysmat_sd_, residual_);

  // finalize the normalized all-scale subgrid-diffusivity matrix
  sysmat_sd_->complete();

  // apply DBC to normalized all-scale subgrid-diffusivity matrix
  Core::LinAlg::apply_dirichlet_to_system(
      *sysmat_sd_, *phinp_, *residual_, *phinp_, *(dbcmaps_->cond_map()));

  // get normalized fine-scale subgrid-diffusivity matrix
  {
    Teuchos::ParameterList params;
    Core::LinearSolver::Parameters::compute_solver_parameters(*discret_, params);

    // get nullspace parameters
    auto nullspace = params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace");

    // get plain aggregation Ptent
    Core::LinAlg::SparseMatrix Ptent =
        Core::LinAlg::create_interpolation_matrix(*sysmat_sd_, *nullspace, params);

    // compute scale-separation matrix: S = I - Ptent*Ptent^T
    Sep_ = Core::LinAlg::matrix_multiply(Ptent, false, Ptent, true);
    Sep_->scale(-1.0);
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
        Core::LinAlg::create_vector(Sep_->row_map(), false);
    tmp->put_scalar(1.0);
    std::shared_ptr<Core::LinAlg::Vector<double>> diag =
        Core::LinAlg::create_vector(Sep_->row_map(), false);
    Sep_->extract_diagonal_copy(*diag);
    diag->update(1.0, *tmp, 1.0);
    Sep_->replace_diagonal_values(*diag);

    // complete scale-separation matrix and check maps
    Sep_->complete(Sep_->domain_map(), Sep_->range_map());
    if (!Sep_->row_map().same_as(sysmat_sd_->row_map())) FOUR_C_THROW("rowmap not equal");
    if (!Sep_->range_map().same_as(sysmat_sd_->range_map())) FOUR_C_THROW("rangemap not equal");
    if (!Sep_->domain_map().same_as(sysmat_sd_->domain_map())) FOUR_C_THROW("domainmap not equal");

    // precomputation of unscaled diffusivity matrix:
    // either two-sided S^T*M*S: multiply M by S from left- and right-hand side
    // or one-sided M*S: only multiply M by S from left-hand side
    if (not incremental_)
    {
      Mnsv_ = Core::LinAlg::matrix_multiply(*sysmat_sd_, false, *Sep_, false);
      // Mnsv_ = Core::LinAlg::Multiply(*Sep_,true,*Mnsv_,false);
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::avm3_scaling(Teuchos::ParameterList& eleparams)
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // some necessary definitions
  double* sgvsqrt = nullptr;
  int length = subgrdiff_->local_length();

  // square-root of subgrid-viscosity-scaling vector for left and right scaling
  sgvsqrt = (double*)subgrdiff_->get_values();
  for (int i = 0; i < length; ++i)
  {
    sgvsqrt[i] = sqrt(sgvsqrt[i]);
    subgrdiff_->replace_local_value(i, sgvsqrt[i]);
  }

  // get unscaled S^T*M*S from Sep
  sysmat_sd_ = std::make_shared<Core::LinAlg::SparseMatrix>(*Mnsv_);

  // left and right scaling of normalized fine-scale subgrid-viscosity matrix
  sysmat_sd_->left_scale(*subgrdiff_);
  sysmat_sd_->right_scale(*subgrdiff_);

  // add the subgrid-viscosity-scaled fine-scale matrix to obtain complete matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat = system_matrix();
  sysmat->add(*sysmat_sd_, false, 1.0, 1.0);

  // set subgrid-diffusivity vector to zero after scaling procedure
  subgrdiff_->put_scalar(0.0);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> ScaTra::ScaTraTimIntImpl::dirichlet_toggle()
    const
{
  if (dbcmaps_ == nullptr) FOUR_C_THROW("Dirichlet map has not been allocated");
  std::shared_ptr<Core::LinAlg::Vector<double>> dirichones =
      Core::LinAlg::create_vector(*(dbcmaps_->cond_map()), false);
  dirichones->put_scalar(1.0);
  std::shared_ptr<Core::LinAlg::Vector<double>> dirichtoggle =
      Core::LinAlg::create_vector(*(discret_->dof_row_map()), true);
  dbcmaps_->insert_cond_vector(*dirichones, *dirichtoggle);
  return dirichtoggle;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::access_dyn_smag_filter(std::shared_ptr<FLD::DynSmagFilter> dynSmag)
{
  DynSmag_ = Core::Utils::shared_ptr_from_ref(*dynSmag.get());

  // access to the dynamic Smagorinsky class is provided
  // by the fluid scatra coupling algorithm
  // therefore, we only have to add some scatra specific parameters here
  if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
  {
    if (myrank_ == 0)
    {
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
      std::cout << "Dynamic Smagorinsky model: provided access for ScaTra       " << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }
    DynSmag_->add_scatra(discret_);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::access_vreman(std::shared_ptr<FLD::Vreman> vrem)
{
  Vrem_ = Core::Utils::shared_ptr_from_ref(*vrem.get());

  // access to the dynamic Vreman class is provided
  // by the fluid scatra coupling algorithm
  // therefore, we only have to add some scatra specific parameters here
  if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    if (myrank_ == 0)
    {
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n';
      std::cout << "Dynamic Vreman model: provided access for ScaTra            " << '\n';
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << '\n';
    }
    Vrem_->add_scatra(discret_);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::read_restart(
    const int step, std::shared_ptr<Core::IO::InputControl> input)
{
  // read restart data associated with meshtying strategy
  strategy_->read_restart(step, std::move(input));
}  // ScaTra::ScaTraTimIntImpl::read_restart()

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::recompute_mean_csgs_b()
{
  if (extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES").get<bool>("ADAPT_CSGS_PHI"))
  {
    // mean Cai
    double meanCai = 0.0;

    // variables required for calculation
    // local sums
    double local_sumCai = 0.0;
    double local_sumVol = 0.0;
    // global sums
    double global_sumCai = 0.0;
    double global_sumVol = 0.0;

    // define element matrices and vectors --- dummies
    Core::LinAlg::SerialDenseMatrix emat1;
    Core::LinAlg::SerialDenseMatrix emat2;
    Core::LinAlg::SerialDenseVector evec1;
    Core::LinAlg::SerialDenseVector evec2;
    Core::LinAlg::SerialDenseVector evec3;

    // generate a parameterlist for communication and control
    Teuchos::ParameterList myparams;

    // action for elements
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::calc_mean_Cai, myparams);

    // add element parameters according to time-integration scheme
    add_time_integration_specific_vectors();

    // set thermodynamic pressure in case of loma
    add_problem_specific_parameters_and_vectors(myparams);

    // loop all elements on this proc (excluding ghosted ones)
    for (int nele = 0; nele < discret_->num_my_row_elements(); ++nele)
    {
      // get the element
      Core::Elements::Element* ele = discret_->l_row_element(nele);

      // get element location vector, dirichlet flags and ownerships
      Core::Elements::LocationArray la(discret_->num_dof_sets());
      ele->location_vector(*discret_, la);

      // call the element evaluate method to integrate functions
      int err = ele->evaluate(myparams, *discret_, la, emat1, emat2, evec1, evec2, evec2);
      if (err) FOUR_C_THROW("Proc {}: Element {} returned err={}", myrank_, ele->id(), err);

      // get contributions of this element and add it up
      local_sumCai += myparams.get<double>("Cai_int");
      local_sumVol += myparams.get<double>("ele_vol");
    }

    // gather contibutions of all procs
    global_sumCai = Core::Communication::sum_all(local_sumCai, discret_->get_comm());
    global_sumVol = Core::Communication::sum_all(local_sumVol, discret_->get_comm());

    // calculate mean Cai
    meanCai = global_sumCai / global_sumVol;

    if (myrank_ == 0)
    {
      std::cout << "\n+----------------------------------------------------------------------------"
                   "----------------+"
                << '\n';
      std::cout << "Multifractal subgrid scales: adaption of CsgsD from near-wall limit of CsgsB:  "
                << std::setprecision(8) << meanCai << '\n';
      std::cout << "+------------------------------------------------------------------------------"
                   "--------------+\n"
                << '\n';
    }

    Discret::Elements::ScaTraEleParameterTurbulence::instance(discret_->name())
        ->set_csgs_phi(meanCai);
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::calc_intermediate_solution()
{
  if (special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" and
      extraparams_->sublist("TURBULENCE MODEL").get<std::string>("SCALAR_FORCING") ==
          "isotropic" and
      Teuchos::getIntegralValue<Inpar::FLUID::ForcingType>(
          extraparams_->sublist("TURBULENCE MODEL"), "FORCING_TYPE") ==
          Inpar::FLUID::linear_compensation_from_intermediate_spectrum)
  {
    bool activate = true;

    if (activate)
    {
      if (homisoturb_forcing_ == nullptr) FOUR_C_THROW("Forcing expected!");

      if (myrank_ == 0)
      {
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|     calculate intermediate solution\n";
        std::cout << "|" << '\n';
      }

      // turn off forcing in nonlinear_solve()
      homisoturb_forcing_->activate_forcing(false);

      // temporary store velnp_ since it will be modified in nonlinear_solve()
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          Core::LinAlg::create_vector(*dofrowmap, true);
      tmp->update(1.0, *phinp_, 0.0);

      // compute intermediate solution without forcing
      forcing_->put_scalar(0.0);  // just to be sure
      nonlinear_solve();

      // calculate required forcing
      homisoturb_forcing_->calculate_forcing(step_);

      // reset velnp_
      phinp_->update(1.0, *tmp, 0.0);

      // recompute intermediate values, since they have been likewise overwritten
      // only for gen.-alpha
      compute_intermediate_values();

      homisoturb_forcing_->activate_forcing(true);

      if (myrank_ == 0)
      {
        std::cout << "|\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|" << '\n';
      }
    }
    else
    {
      // set force to zero
      forcing_->put_scalar(0.0);
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_sc_fld_gr_disp(
    std::shared_ptr<Core::LinAlg::MultiVector<double>> scatra_fluid_growth_disp)
{
  scfldgrdisp_ = std::move(scatra_fluid_growth_disp);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_sc_str_gr_disp(
    std::shared_ptr<Core::LinAlg::MultiVector<double>> scatra_struct_growth_disp)
{
  scstrgrdisp_ = std::move(scatra_struct_growth_disp);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>>
ScaTra::ScaTraTimIntImpl::compute_superconvergent_patch_recovery(
    const Core::LinAlg::Vector<double>& state, const std::string& statename, const int numvec,
    Teuchos::ParameterList& eleparams, const int dim) const
{
  // Warning, this is only tested so far for 1 scalar field!!!

  // dependent on the desired projection, just remove this line
  if (not state.get_map().same_as(*discret_->dof_row_map()))
    FOUR_C_THROW("input map is not a dof row map of the fluid");

  // set given state for element evaluation
  discret_->set_state(statename, state);

  switch (dim)
  {
    case 3:
      return Core::FE::compute_superconvergent_patch_recovery<3>(
          *discret_, state, statename, numvec, eleparams);
      break;
    case 2:
      return Core::FE::compute_superconvergent_patch_recovery<2>(
          *discret_, state, statename, numvec, eleparams);
      break;
    case 1:
      return Core::FE::compute_superconvergent_patch_recovery<1>(
          *discret_, state, statename, numvec, eleparams);
      break;
    default:
      FOUR_C_THROW("only 1/2/3D implementation available for superconvergent patch recovery");
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
bool ScaTra::ScaTraTimIntImpl::convergence_check(int itnum, int itmax, const double ittol) const
{
  // declare bool variable for potentially stopping nonlinear iteration
  bool stopnonliniter = false;

  // declare L2-norm of (two) residual, incremental scalar and scalar
  double res1norm_L2(0.0);
  double res2norm_L2(0.0);
  double phi1incnorm_L2(0.0);
  double phi2incnorm_L2(0.0);
  double phi1norm_L2(0.0);
  double phi2norm_L2(0.0);

  // distinguish whether one or two scalars are considered
  if (num_scal() == 2)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> vec1 =
        splitter_->extract_other_vector(*residual_);
    std::shared_ptr<Core::LinAlg::Vector<double>> vec2 = splitter_->extract_cond_vector(*residual_);
    vec1->norm_2(&res1norm_L2);
    vec2->norm_2(&res2norm_L2);

    vec1 = splitter_->extract_other_vector(*increment_);
    vec2 = splitter_->extract_cond_vector(*increment_);
    vec1->norm_2(&phi1incnorm_L2);
    vec2->norm_2(&phi2incnorm_L2);

    vec1 = splitter_->extract_other_vector(*phinp_);
    vec2 = splitter_->extract_cond_vector(*phinp_);
    vec1->norm_2(&phi1norm_L2);
    vec2->norm_2(&phi2norm_L2);

    // check for any INF's and NaN's
    if (std::isnan(res1norm_L2) or std::isnan(phi1incnorm_L2) or std::isnan(phi1norm_L2) or
        std::isnan(res2norm_L2) or std::isnan(phi2incnorm_L2) or std::isnan(phi2norm_L2))
      FOUR_C_THROW("At least one of the calculated vector norms is NaN.");

    if (std::isinf(res1norm_L2) or std::isinf(phi1incnorm_L2) or std::isinf(phi1norm_L2) or
        std::isinf(res2norm_L2) or std::isinf(phi2incnorm_L2) or std::isinf(phi2norm_L2))
      FOUR_C_THROW("At least one of the calculated vector norms is INF.");

    // for scalar norm being (close to) zero, set to one
    if (phi1norm_L2 < 1e-5) phi1norm_L2 = 1.0;
    if (phi2norm_L2 < 1e-5) phi2norm_L2 = 1.0;

    if (myrank_ == 0)
    {
      printf(
          "+------------+-------------------+---------------+---------------+---------------+------"
          "---------+\n");
      printf(
          "|- step/max -|- tol      [norm] -|- residual1   -|- scalar1-inc -|- residual2   -|- "
          "scalar2-inc -|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  |  %10.3E   |  %10.3E   |  %10.3E   |  %10.3E   |",
          itnum, itmax, ittol, res1norm_L2, phi1incnorm_L2 / phi1norm_L2, res2norm_L2,
          phi2incnorm_L2 / phi2norm_L2);
      printf("\n");
      printf(
          "+------------+-------------------+---------------+---------------+---------------+------"
          "---------+\n");
    }

    if ((res1norm_L2 <= ittol) and (phi1incnorm_L2 / phi1norm_L2 <= ittol) and
        (res2norm_L2 <= ittol) and (phi2incnorm_L2 / phi2norm_L2 <= ittol))
      stopnonliniter = true;

    // warn if itemax is reached without convergence, but proceed to next timestep
    if ((itnum == itmax) and ((res1norm_L2 > ittol) or (phi1incnorm_L2 / phi1norm_L2 > ittol) or
                                 (res2norm_L2 > ittol) or (phi2incnorm_L2 / phi2norm_L2 > ittol)))
    {
      stopnonliniter = true;
      if (myrank_ == 0)
      {
        printf("|            >>>>>> not converged in itemax steps!             |\n");
        printf("+--------------------------------------------------------------+\n");
      }
    }
  }
  else if (num_scal() == 1)
  {
    // compute L2-norm of residual, incremental scalar and scalar
    residual_->norm_2(&res1norm_L2);
    increment_->norm_2(&phi1incnorm_L2);
    phinp_->norm_2(&phi1norm_L2);

    // check for any INF's and NaN's
    if (std::isnan(res1norm_L2) or std::isnan(phi1incnorm_L2) or std::isnan(phi1norm_L2))
      FOUR_C_THROW("At least one of the calculated vector norms is NaN.");

    if (std::isinf(res1norm_L2) or std::isinf(phi1incnorm_L2) or std::isinf(phi1norm_L2))
      FOUR_C_THROW("At least one of the calculated vector norms is INF.");

    // for scalar norm being (close to) zero, set to one
    if (phi1norm_L2 < 1e-5) phi1norm_L2 = 1.0;

    if (myrank_ == 0)
    {
      printf("+------------+-------------------+--------------+--------------+\n");
      printf("|- step/max -|- tol      [norm] -|- residual   -|- scalar-inc -|\n");
      printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E   |", itnum, itmax, ittol,
          res1norm_L2, phi1incnorm_L2 / phi1norm_L2);
      printf("\n");
      printf("+------------+-------------------+--------------+--------------+\n");
    }

    if ((res1norm_L2 <= ittol) and (phi1incnorm_L2 / phi1norm_L2 <= ittol)) stopnonliniter = true;

    // warn if itemax is reached without convergence, but proceed to next timestep
    if ((itnum == itmax) and ((res1norm_L2 > ittol) or (phi1incnorm_L2 / phi1norm_L2 > ittol)))
    {
      stopnonliniter = true;
      if (myrank_ == 0)
      {
        printf("|            >>>>>> not converged in itemax steps!             |\n");
        printf("+--------------------------------------------------------------+\n");
      }
    }
  }
  else
  {
    FOUR_C_THROW(
        "ScaTra convergence check for number of scalars other than one or two not yet supported!");
  }

  return stopnonliniter;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::fd_check()
{
  // initial screen output
  if (myrank_ == 0) std::cout << '\n' << "FINITE DIFFERENCE CHECK FOR SCATRA SYSTEM MATRIX" << '\n';

  // make a copy of state variables to undo perturbations later
  Core::LinAlg::Vector<double> phinp_original(*phinp_);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_original = nullptr;
  if (std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_) != nullptr)
  {
    sysmat_original = std::make_shared<Core::LinAlg::SparseMatrix>(
        *(std::static_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_)));
  }
  else if (std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_) != nullptr)
  {
    sysmat_original = std::make_shared<Core::LinAlg::SparseMatrix>(
        *(std::static_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_)->merge()));
  }
  else
  {
    FOUR_C_THROW("Type of system matrix unknown!");
  }
  sysmat_original->complete();

  // make a copy of system right-hand side vector
  Core::LinAlg::Vector<double> rhs_original(*residual_);

  // initialize counter for system matrix entries with failing finite difference check
  int counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  for (int colgid = 0; colgid <= sysmat_original->col_map().max_all_gid(); ++colgid)
  {
    // check whether current column index is a valid global column index and continue loop if not
    int collid(sysmat_original->col_map().lid(colgid));
    int maxcollid(-1);
    maxcollid = Core::Communication::max_all(collid, discret_->get_comm());
    if (maxcollid < 0) continue;

    // fill state vector with original state variables
    phinp_->update(1., phinp_original, 0.);

    // impose perturbation
    if (phinp_->get_map().my_gid(colgid))
    {
      phinp_->sum_into_global_value(colgid, fdcheckeps_);
    }

    // carry perturbation over to state vectors at intermediate time stages if necessary
    compute_intermediate_values();

    // calculate element right-hand side vector for perturbed state
    assemble_mat_and_rhs();

    // Now we compare the difference between the current entries in the system matrix
    // and their finite difference approximations according to
    // entries ?= (residual_perturbed - residual_original) / epsilon

    // Note that the residual_ vector actually denotes the right-hand side of the linear
    // system of equations, i.e., the negative system residual.
    // To account for errors due to numerical cancellation, we additionally consider
    // entries + residual_original / epsilon ?= residual_perturbed / epsilon

    // Note that we still need to evaluate the first comparison as well. For small entries in the
    // system matrix, the second comparison might yield good agreement in spite of the entries being
    // wrong!
    for (int rowlid = 0; rowlid < discret_->dof_row_map()->num_my_elements(); ++rowlid)
    {
      // get global index of current matrix row
      const int rowgid = sysmat_original->col_map().gid(rowlid);
      if (rowgid < 0) FOUR_C_THROW("Invalid global ID of matrix row!");

      // get relevant entry in current row of original system matrix
      double entry(0.);
      int length = sysmat_original->num_my_entries(rowlid);
      int numentries;
      std::vector<double> values(length);
      std::vector<int> indices(length);
      sysmat_original->extract_my_row_copy(
          rowlid, length, numentries, values.data(), indices.data());
      for (int ientry = 0; ientry < length; ++ientry)
      {
        if (sysmat_original->col_map().gid(indices[ientry]) == colgid)
        {
          entry = values[ientry];
          break;
        }
      }

      // finite difference suggestion (first divide by epsilon and then add for better conditioning)
      const double fdval = -residual_->local_values_as_span()[rowlid] / fdcheckeps_ +
                           rhs_original.local_values_as_span()[rowlid] / fdcheckeps_;

      // confirm accuracy of first comparison
      if (abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
        FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

      // absolute and relative errors in first comparison
      const double abserr1 = entry - fdval;
      if (abs(abserr1) > maxabserr) maxabserr = abs(abserr1);
      double relerr1(0.);
      if (abs(entry) > 1.e-17)
        relerr1 = abserr1 / abs(entry);
      else if (abs(fdval) > 1.e-17)
        relerr1 = abserr1 / abs(fdval);
      if (abs(relerr1) > maxrelerr) maxrelerr = abs(relerr1);

      // evaluate first comparison
      if (abs(relerr1) > fdchecktol_)
      {
        std::cout << "sysmat[" << rowgid << "," << colgid << "]:  " << entry << "   ";
        std::cout << "finite difference suggestion:  " << fdval << "   ";
        std::cout << "absolute error:  " << abserr1 << "   ";
        std::cout << "relative error:  " << relerr1 << '\n';

        counter++;
      }

      // first comparison OK
      else
      {
        // left-hand side in second comparison
        const double left = entry - rhs_original.local_values_as_span()[rowlid] / fdcheckeps_;

        // right-hand side in second comparison
        const double right = -residual_->local_values_as_span()[rowlid] / fdcheckeps_;

        // confirm accuracy of second comparison
        if (abs(right) > 1.e-17 and abs(right) < 1.e-15)
          FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

        // absolute and relative errors in second comparison
        const double abserr2 = left - right;
        if (abs(abserr2) > maxabserr) maxabserr = abs(abserr2);
        double relerr2(0.);
        if (abs(left) > 1.e-17)
          relerr2 = abserr2 / abs(left);
        else if (abs(right) > 1.e-17)
          relerr2 = abserr2 / abs(right);
        if (abs(relerr2) > maxrelerr) maxrelerr = abs(relerr2);

        // evaluate second comparison
        if (abs(relerr2) > fdchecktol_)
        {
          std::cout << "sysmat[" << rowgid << "," << colgid << "]-rhs[" << rowgid
                    << "]/eps:  " << left << "   ";
          std::cout << "-rhs_perturbed[" << rowgid << "]/eps:  " << right << "   ";
          std::cout << "absolute error:  " << abserr2 << "   ";
          std::cout << "relative error:  " << relerr2 << '\n';

          counter++;
        }
      }
    }
  }

  // communicate tracking variables
  int counterglobal(0);
  counterglobal = Core::Communication::sum_all(counter, discret_->get_comm());
  double maxabserrglobal(0.);
  maxabserrglobal = Core::Communication::max_all(maxabserr, discret_->get_comm());
  double maxrelerrglobal(0.);
  maxrelerrglobal = Core::Communication::max_all(maxrelerr, discret_->get_comm());

  // final screen output
  if (myrank_ == 0)
  {
    if (counterglobal)
    {
      printf(
          "--> FAILED AS LISTED ABOVE WITH %d CRITICAL MATRIX ENTRIES IN TOTAL\n\n", counterglobal);
      FOUR_C_THROW("Finite difference check failed for scalar transport system matrix!");
    }
    else
    {
      printf(
          "--> PASSED WITH MAXIMUM ABSOLUTE ERROR %+12.5e AND MAXIMUM RELATIVE ERROR %+12.5e\n\n",
          maxabserrglobal, maxrelerrglobal);
    }
  }

  // undo perturbations of state variables
  phinp_->update(1., phinp_original, 0.);
  compute_intermediate_values();

  // recompute system matrix and right-hand side vector based on original state variables
  assemble_mat_and_rhs();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::evaluate_error_compared_to_analytical_sol()
{
  switch (calcerror_)
  {
    case Inpar::ScaTra::calcerror_byfunction:
    case Inpar::ScaTra::calcerror_spherediffusion:
    {
      // create the parameters for the error calculation
      Teuchos::ParameterList eleparams;
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
          "action", ScaTra::Action::calc_error, eleparams);
      eleparams.set<Inpar::ScaTra::CalcError>("calcerrorflag", calcerror_);

      if (calcerror_ == Inpar::ScaTra::calcerror_byfunction)
      {
        const int errorfunctnumber = params_->get<int>("CALCERRORNO");
        if (errorfunctnumber < 1)
          FOUR_C_THROW(
              "Invalid value of parameter CALCERRORNO for error "
              "function evaluation!");

        eleparams.set<int>("error function number", errorfunctnumber);
      }

      // set vector values needed by elements
      discret_->set_state("phinp", *phinp_);

      // get (squared) error values
      Core::LinAlg::SerialDenseVector errors(4 * num_dof_per_node());
      discret_->evaluate_scalars(eleparams, errors);

      for (int k = 0; k < num_dof_per_node(); ++k)
      {
        if (std::abs((errors)[k * 4 + 2]) > 1e-14)
          (*relerrors_)[k * 2] = sqrt((errors)[k * 4]) / sqrt((errors)[k * 4 + 2]);
        else
          FOUR_C_THROW("Can't compute relative L2 error due to numerical roundoff sensitivity!");
        if (std::abs((errors)[k * 4 + 3]) > 1e-14)
          (*relerrors_)[k * 2 + 1] = sqrt((errors)[k * 4 + 1]) / sqrt((errors)[k * 4 + 3]);
        else
          FOUR_C_THROW("Can't compute relative H1 error due to numerical roundoff sensitivity!");

        if (myrank_ == 0)
        {
          const std::string simulation = problem_->output_control_file()->file_name();
          const std::string fname = simulation + "_c" + std::to_string(k) + "_time.relerror";
          std::ofstream f;

          // create new error file and write initial error
          if (step_ == 0)
          {
            f.open(fname.c_str());
            f << "| Step | Time | rel. L2-error | rel. H1-error |" << '\n';
          }

          // append error of the last time step to the error file
          else
          {
            f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          }

          f << step_ << " " << time_ << " " << std::setprecision(6) << (*relerrors_)[k * 2] << " "
            << (*relerrors_)[k * 2 + 1] << '\n';

          f.flush();
          f.close();
        }
      }

      break;
    }

    case Inpar::ScaTra::calcerror_bycondition:
    {
      // extract conditions from discretization
      std::vector<const Core::Conditions::Condition*> relerrorconditions;
      discret_->get_condition("ScatraRelError", relerrorconditions);
      const unsigned ncond = relerrorconditions.size();

      // loop over all conditions
      for (unsigned icond = 0; icond < ncond; ++icond)
      {
        // extract condition ID
        const int condid = relerrorconditions[icond]->parameters().get<int>("ConditionID");

        // create element parameter list for error calculation
        Teuchos::ParameterList eleparams;
        Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
            "action", ScaTra::Action::calc_error, eleparams);
        eleparams.set<Inpar::ScaTra::CalcError>(
            "calcerrorflag", Inpar::ScaTra::calcerror_byfunction);
        const int errorfunctnumber = relerrorconditions[icond]->parameters().get<int>("Function");
        if (errorfunctnumber < 1) FOUR_C_THROW("Invalid function number for error calculation!");
        eleparams.set<int>("error function number", errorfunctnumber);

        // set state vector needed by elements
        discret_->set_state("phinp", *phinp_);

        // get (squared) error values
        Core::LinAlg::SerialDenseVector errors(4 * num_dof_per_node());
        discret_->evaluate_scalars(eleparams, errors, "ScatraRelError", condid);

        // compute index offset
        const int offset = condid * num_dof_per_node() * 2;

        // loop over all degrees of freedom
        for (int k = 0; k < num_dof_per_node(); ++k)
        {
          // compute relative L2 error
          if (std::abs((errors)[k * 4 + 2]) > 1e-14)
            (*relerrors_)[offset + k * 2] = sqrt((errors)[k * 4]) / sqrt((errors)[k * 4 + 2]);
          else
            FOUR_C_THROW("Can't compute relative L2 error due to numerical roundoff sensitivity!");

          // compute relative H1 error
          if (std::abs((errors)[k * 4 + 3]) > 1e-14)
          {
            (*relerrors_)[offset + k * 2 + 1] =
                sqrt((errors)[k * 4 + 1]) / sqrt((errors)[k * 4 + 3]);
          }
          else
          {
            FOUR_C_THROW("Can't compute relative H1 error due to numerical roundoff sensitivity!");
          }
        }
      }

      // print errors to files
      if (myrank_ == 0)
      {
        // loop over all degrees of freedom
        for (int k = 0; k < num_dof_per_node(); ++k)
        {
          // determine name of file associated with current degree of freedom
          const std::string fname = problem_->output_control_file()->file_name() + "_dof_" +
                                    std::to_string(k) + ".relerror";

          // initialize output file stream
          std::ofstream f;

          // create new error file and write initial error
          if (step_ == 0)
          {
            f.open(fname.c_str());
            f << "| Step | Time |";

            // loop over all conditions
            for (unsigned icond = 0; icond < ncond; ++icond)
            {
              // extract condition ID
              const int condid = relerrorconditions[icond]->parameters().get<int>("ConditionID");

              // extend headline
              f << " rel. L2-error in domain " << condid << " | rel. H1-error in domain " << condid
                << " |";
            }

            // break line
            f << '\n';
          }

          // append error of the last time step to the error file
          else
          {
            f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          }

          // write error
          f << step_ << " " << time_ << std::scientific << std::setprecision(6);

          // loop over all conditions
          for (unsigned icond = 0; icond < ncond; ++icond)
          {
            // extract condition ID
            const int condid = relerrorconditions[icond]->parameters().get<int>("ConditionID");

            // extend error line
            f << " " << (*relerrors_)[condid * num_dof_per_node() * 2 + k * 2] << " "
              << (*relerrors_)[condid * num_dof_per_node() * 2 + k * 2 + 1];
          }

          // finalize
          f << '\n';
          f.flush();
          f.close();
        }
      }

      break;
    }

    case Inpar::ScaTra::calcerror_no:
    {
      // do nothing
      break;
    }

    default:
    {
      FOUR_C_THROW("Cannot calculate error. Unknown type of analytical test problem!");
      break;
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::explicit_predictor() const
{
  // predict discrete solution variables associated with meshtying strategy
  strategy_->explicit_predictor();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::perform_aitken_relaxation(
    Core::LinAlg::Vector<double>& phinp, const Core::LinAlg::Vector<double>& phinp_inc_diff)
{
  if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken)
  {
    // compute L2 norm of difference between current and previous increments of macro-scale state
    // vector
    double phinp_inc_diff_L2(0.);
    phinp_inc_diff.norm_2(&phinp_inc_diff_L2);

    // compute dot product between increment of macro-scale state vector and difference between
    // current and previous increments of macro-scale state vector
    double phinp_inc_dot_phinp_inc_diff(0.);
    phinp_inc_diff.dot(*phinp_inc_, &phinp_inc_dot_phinp_inc_diff);

    // compute Aitken relaxation factor
    if (iternum_outer_ > 1 and phinp_inc_diff_L2 > 1.e-12)
      omega_[0] *= 1 - phinp_inc_dot_phinp_inc_diff / (phinp_inc_diff_L2 * phinp_inc_diff_L2);

    // perform Aitken relaxation
    phinp.update(omega_[0], *phinp_inc_, 1.);
  }

  else
  {
    FOUR_C_THROW("Invalid Aitken method!");
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::apply_bc_to_system()
{
  apply_dirichlet_bc(time_, phinp_, nullptr);
  compute_intermediate_values();
  apply_neumann_bc(neumann_loads_);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::evaluate_initial_time_derivative(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // create and fill parameter list for elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_initial_time_deriv, eleparams);
  add_problem_specific_parameters_and_vectors(eleparams);

  // add state vectors according to time integration scheme
  add_time_integration_specific_vectors();
  // modify global system of equations as explained above
  discret_->evaluate(eleparams, matrix, rhs);

  // finalize assembly of global mass matrix
  matrix->complete();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyBase::prepare_evaluate(
    const ScaTraTimIntImpl* const scatratimint, Teuchos::ParameterList& eleparams) const
{
  const std::shared_ptr<Core::FE::Discretization>& discret = scatratimint->discret_;

  // add state vector to discretization
  discret->set_state("phinp", *scatratimint->phinp_);

  // set action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_total_and_mean_scalars, eleparams);
  eleparams.set("inverting", false);
  eleparams.set("calc_grad_phi", output_mean_grad_);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyBase::print_header_to_screen(const std::string& dis_name) const
{
  if (myrank_ == 0)
  {
    // screen output
    std::cout << '\n';
    std::cout << "Total and mean values of transported scalars from discretization " << dis_name
              << ":" << '\n';
    std::cout
        << "+-----------+-----------+--------------------+-----------------+-------------------+"
        << '\n';
    std::cout
        << "| domain ID | scalar ID | total scalar value | domain integral | mean scalar value |"
        << '\n';
    std::cout
        << "+-----------+-----------+--------------------+-----------------+-------------------+"
        << '\n';
  }
}

/*----------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyBase::finalize_screen_output() const
{
  if (myrank_ == 0)
  {
    std::cout
        << "+-----------+-----------+--------------------+-----------------+-------------------+"
        << '\n';
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyBase::output_total_and_mean_scalars(
    const ScaTraTimIntImpl* const scatratimint, const int num)
{
  // evaluate domain/condition integrals
  evaluate_integrals(scatratimint);

  // print evaluated data to screen
  print_header_to_screen(scatratimint->discret_->name());
  print_to_screen();
  finalize_screen_output();

  // write evaluated data to file
  const auto output_data = prepare_csv_output();

  runtime_csvwriter_->write_data_to_file(scatratimint->time(), scatratimint->step(), output_data);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
ScaTra::OutputScalarsStrategyBase::OutputScalarsStrategyBase(const ScaTraTimIntImpl* scatratimint)
{
  myrank_ = scatratimint->myrank_;

  output_mean_grad_ = scatratimint->scatra_parameter_list()->get<bool>("OUTPUTSCALARSMEANGRAD");

  output_micro_dis_ = scatratimint->macro_scale() and scatratimint->nds_micro();

  std::string filename("");
  if (scatratimint->scatra_parameter_list()->isParameter("output_file_name_discretization") and
      scatratimint->scatra_parameter_list()->get<bool>("output_file_name_discretization"))
    filename = scatratimint->discretization()->name() + "_scalarvalues";
  else
    filename = "scalarvalues";

  runtime_csvwriter_.emplace(myrank_, scatratimint->disc_writer()->output(), filename);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
ScaTra::OutputScalarsStrategyDomain::OutputScalarsStrategyDomain(
    const ScaTraTimIntImpl* scatratimint)
    : OutputScalarsStrategyBase(scatratimint), dummy_domain_id_(-1)
{
  if (not scatratimint->scalarhandler_->equal_num_dof())
  {
    FOUR_C_THROW(
        "Output of scalars for entire domain not valid for different numbers of DOFs per node in "
        "ScaTra discretization. \n"
        "Use option `by_condition' for the parameter 'OUTPUTSCALARS' in the SCALAR TRANSPORT "
        "DYNAMIC section instead!");
  }

  numdofpernode_ = scatratimint->scalarhandler_->num_dof_per_node();
  numscal_ = scatratimint->scalarhandler_->num_scal();

  // initialize result vectors associated with entire domain
  totalscalars_[dummy_domain_id_].resize(numdofpernode_, 0.0);
  meanscalars_[dummy_domain_id_].resize(numdofpernode_, 0.0);
  meangradients_[dummy_domain_id_].resize(numscal_, 0.0);
  domainintegral_[dummy_domain_id_] = 0.0;

  FOUR_C_ASSERT(runtime_csvwriter_.has_value(), "internal error: runtime csv writer not created.");

  // register output in csv writer
  runtime_csvwriter_->register_data_vector("Integral of entire domain", 1, 16);
  for (int k = 0; k < numdofpernode_; ++k)
  {
    runtime_csvwriter_->register_data_vector(
        "Total value of scalar " + std::to_string(k + 1) + " in entire domain", 1, 16);
    runtime_csvwriter_->register_data_vector(
        "Mean value of scalar " + std::to_string(k + 1) + " in entire domain", 1, 16);
  }
  if (output_mean_grad_)
  {
    for (int k = 0; k < numscal_; ++k)
    {
      runtime_csvwriter_->register_data_vector(
          "Mean value of gradient of scalar " + std::to_string(k + 1) + " in entire domain", 1, 16);
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyDomain::print_to_screen()
{
  if (myrank_ == 0)
  {
    std::cout
        << "+-----------+-----------+--------------------+-----------------+-------------------+"
        << '\n';

    for (int k = 0; k < numdofpernode_; ++k)
    {
      std::cout << "|  overall  |    " << std::setw(2) << k + 1 << "     |    " << std::scientific
                << std::setprecision(6) << totalscalars_[dummy_domain_id_][k] << "    |   "
                << domainintegral_[dummy_domain_id_] << "  |    "
                << meanscalars_[dummy_domain_id_][k] << "   |" << '\n';
    }
    if (output_mean_grad_)
    {
      for (int k = 0; k < numscal_; ++k)
      {
        std::cout << "|  overall  |  grad. " << k + 1 << "  |        ----        |   "
                  << domainintegral_[dummy_domain_id_] << "  |    "
                  << meangradients_[dummy_domain_id_][k] << "   |" << '\n';
      }
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::map<std::string, std::vector<double>> ScaTra::OutputScalarsStrategyDomain::prepare_csv_output()
{
  FOUR_C_ASSERT(runtime_csvwriter_.has_value(), "internal error: runtime csv writer not created.");
  std::map<std::string, std::vector<double>> output_data;
  output_data["Integral of entire domain"] = {domainintegral_[dummy_domain_id_]};

  for (int k = 0; k < numdofpernode_; ++k)
  {
    output_data["Total value of scalar " + std::to_string(k + 1) + " in entire domain"] = {
        totalscalars_[dummy_domain_id_][k]};
    output_data["Mean value of scalar " + std::to_string(k + 1) + " in entire domain"] = {
        meanscalars_[dummy_domain_id_][k]};
  }
  if (output_mean_grad_)
  {
    for (int k = 0; k < numscal_; ++k)
    {
      output_data["Mean value of gradient of scalar " + std::to_string(k + 1) +
                  " in entire domain"] = {meangradients_[dummy_domain_id_][k]};
    }
  }
  return output_data;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyDomain::evaluate_integrals(
    const ScaTraTimIntImpl* const scatratimint)
{
  Teuchos::ParameterList eleparams;

  // fill parameter list and set states in discretization
  prepare_evaluate(scatratimint, eleparams);

  // initialize result vector
  // first components = scalar integrals, last component = domain integral
  const int num_active_scalars = output_mean_grad_ ? numscal_ : 0;
  FourC::Core::LinAlg::SerialDenseVector scalars(numdofpernode_ + 1 + num_active_scalars);

  // perform integration
  scatratimint->discret_->evaluate_scalars(eleparams, scalars);

  // extract domain integral
  domainintegral_[dummy_domain_id_] = (scalars)[numdofpernode_];

  // compute results
  for (int k = 0; k < numdofpernode_; ++k)
  {
    totalscalars_[dummy_domain_id_][k] = (scalars)[k];
    meanscalars_[dummy_domain_id_][k] = (scalars)[k] / domainintegral_[dummy_domain_id_];
  }
  if (output_mean_grad_)
  {
    for (int k = 0; k < numscal_; ++k)
    {
      const double mean_gradient =
          (scalars)[numdofpernode_ + 1 + k] / domainintegral_[dummy_domain_id_];

      meangradients_[dummy_domain_id_][k] = std::abs(mean_gradient) < 1.0e-10 ? 0.0 : mean_gradient;
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
ScaTra::OutputScalarsStrategyCondition::OutputScalarsStrategyCondition(
    const ScaTraTimIntImpl* scatratimint)
    : OutputScalarsStrategyBase(scatratimint)
{
  // extract conditions for calculation of total and mean values of transported scalars
  scatratimint->discret_->get_condition("TotalAndMeanScalar", conditions_);

  if (conditions_.empty())
  {
    FOUR_C_THROW(
        "No 'TotalAndMeanScalar' conditions defined for scalar output by condition!\n"
        "Change the parameter 'OUTPUTSCALARS' in the SCALAR TRANSPORT DYNAMIC section \n"
        "or include 'DESIGN TOTAL AND MEAN SCALAR' conditions!");
  }

  numdofpernodepercondition_.clear();
  numscalpercondition_.clear();

  // loop over all conditions
  for (const auto* condition : conditions_)
  {
    // extract condition ID
    const int condid = condition->parameters().get<int>("ConditionID");

    // determine the number of dofs on the current condition
    const int numdofpernode = scatratimint->scalarhandler_->num_dof_per_node_in_condition(
        *condition, *scatratimint->discret_);
    const int numscalpernode = scatratimint->scalarhandler_->num_scal();

    // save the number of dofs on the current condition
    numdofpernodepercondition_.insert(std::make_pair(condid, numdofpernode));
    numscalpercondition_.insert(std::make_pair(condid, numscalpernode));

    // initialize result vectors associated with current condition
    totalscalars_[condid].resize(numdofpernodepercondition_[condid], 0.0);
    meanscalars_[condid].resize(numdofpernodepercondition_[condid], 0.0);
    meangradients_[condid].resize(numscalpercondition_[condid], 0.0);
    domainintegral_[condid] = 0.0;

    // micro dis has only one dof
    if (output_micro_dis_) micromeanscalars_[condid].resize(1, 0.0);

    FOUR_C_ASSERT(
        runtime_csvwriter_.has_value(), "internal error: runtime csv writer not created.");

    // register all data vectors
    runtime_csvwriter_->register_data_vector("Integral of domain " + std::to_string(condid), 1, 16);
    for (int k = 0; k < numdofpernodepercondition_[condid]; ++k)
    {
      runtime_csvwriter_->register_data_vector(
          "Total value of scalar " + std::to_string(k + 1) + " in domain " + std::to_string(condid),
          1, 16);
      runtime_csvwriter_->register_data_vector(
          "Mean value of scalar " + std::to_string(k + 1) + " in domain " + std::to_string(condid),
          1, 16);
    }
    if (output_mean_grad_)
    {
      for (int k = 0; k < numscalpercondition_[condid]; ++k)
      {
        runtime_csvwriter_->register_data_vector("Mean value of gradient of scalar " +
                                                     std::to_string(k + 1) + " in domain " +
                                                     std::to_string(condid),
            1, 16);
      }
    }
    if (output_micro_dis_)
    {
      runtime_csvwriter_->register_data_vector(
          "Mean value of micro scalar in domain " + std::to_string(condid), 1, 16);
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
ScaTra::OutputDomainIntegralStrategy::OutputDomainIntegralStrategy(
    const ScaTraTimIntImpl* scatratimint)
{
  // extract conditions for calculation of total and mean values of transported scalars
  scatratimint->discretization()->get_condition("DomainIntegral", conditionsdomain_);
  scatratimint->discretization()->get_condition("BoundaryIntegral", conditionsboundary_);

  if (conditionsdomain_.empty() && conditionsboundary_.empty())
  {
    FOUR_C_THROW(
        "No 'DomainIntegral' or 'BoundaryIntegral' conditions defined for scalar output by "
        "condition!\n"
        "Change the parameter 'COMPUTEINTEGRALS' in the SCALAR TRANSPORT DYNAMIC section \n"
        "or include 'DESIGN DOMAIN INTEGRAL * CONDITIONS' conditions!");
  }

  domainintegralvalues_.resize(conditionsdomain_.size());
  boundaryintegralvalues_.resize(conditionsboundary_.size());
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyCondition::print_to_screen()
{
  if (myrank_ == 0)
  {
    for (const auto* condition : conditions_)
    {
      // extract condition ID
      const int condid = condition->parameters().get<int>("ConditionID");

      // determine the number of dofs on the current condition
      const int numdofpernode = numdofpernodepercondition_[condid];

      std::ostringstream condid_string;
      condid_string << "|    " << std::setw(2) << condid << "     |";
      for (int k = 0; k < numdofpernode; ++k)
      {
        std::cout << condid_string.str() << "    " << std::setw(2) << k + 1 << "     |    "
                  << std::scientific << std::setprecision(6) << totalscalars_[condid][k]
                  << "    |   " << domainintegral_[condid] << "  |    " << meanscalars_[condid][k]
                  << "   |" << '\n';
      }
      if (output_mean_grad_)
      {
        for (int k = 0; k < numscalpercondition_[condid]; ++k)
        {
          std::cout << condid_string.str() << "  grad. " << k + 1 << "  |        ----        |   "
                    << domainintegral_[condid] << "  |    " << meangradients_[condid][k] << "   |"
                    << '\n';
        }
      }
    }
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::map<std::string, std::vector<double>>
ScaTra::OutputScalarsStrategyCondition::prepare_csv_output()
{
  std::map<std::string, std::vector<double>> output_data;

  for (const auto* condition : conditions_)
  {
    // extract condition ID
    const int condid = condition->parameters().get<int>("ConditionID");

    FOUR_C_ASSERT(
        runtime_csvwriter_.has_value(), "internal error: runtime csv writer not created.");

    output_data["Integral of domain " + std::to_string(condid)] = {domainintegral_[condid]};

    for (int k = 0; k < numdofpernodepercondition_[condid]; ++k)
    {
      output_data["Total value of scalar " + std::to_string(k + 1) + " in domain " +
                  std::to_string(condid)] = {totalscalars_[condid][k]};
      output_data["Mean value of scalar " + std::to_string(k + 1) + " in domain " +
                  std::to_string(condid)] = {meanscalars_[condid][k]};
    }
    if (output_mean_grad_)
    {
      for (int k = 0; k < numscalpercondition_[condid]; ++k)
      {
        output_data["Mean value of gradient of scalar " + std::to_string(k + 1) + " in domain " +
                    std::to_string(condid)] = {meangradients_[condid][k]};
      }
    }
    if (output_micro_dis_)
    {
      output_data["Mean value of micro scalar in domain " + std::to_string(condid)] = {
          micromeanscalars_[condid][0]};
    }
  }
  return output_data;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputDomainIntegralStrategy::evaluate_integrals_and_print_results(
    const ScaTraTimIntImpl* const scatratimint, const std::string& condstring)
{
  // initialize label
  std::string label;

  // create parameter list
  Teuchos::ParameterList condparams;

  // determine label and element action depending on whether domain or boundary integrals are
  // relevant
  if (condstring == "BoundaryIntegral")
  {
    label = "Boundary";
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
        "action", ScaTra::BoundaryAction::calc_boundary_integral, condparams);
  }
  else if (condstring == "DomainIntegral")
  {
    label = "Domain";
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::calc_domain_integral, condparams);
  }
  else
  {
    FOUR_C_THROW("Invalid condition name!");
  }

  // extract conditions for computation of domain or boundary integrals
  std::vector<const Core::Conditions::Condition*> conditions;
  std::shared_ptr<Core::FE::Discretization> discret = scatratimint->discretization();
  const int myrank = Core::Communication::my_mpi_rank(discret->get_comm());
  discret->get_condition(condstring, conditions);

  // print header
  if (conditions.size() > 0 and myrank == 0)
  {
    std::cout << '\n' << label + " integrals:" << '\n';
    std::cout << "+----+-------------------------+" << '\n';
    std::cout << "| ID | value of integral       |" << '\n';
  }

  // loop over all conditions
  for (int condid = 0; condid < static_cast<int>(conditions.size()); ++condid)
  {
    // initialize one-component result vector for value of current domain or boundary integral
    Core::LinAlg::SerialDenseVector integralvalue(1);

    // compute value of current domain or boundary integral
    discret->evaluate_scalars(condparams, integralvalue, condstring, condid);

    // output results to screen and file
    if (myrank == 0)
    {
      // print results to screen
      std::cout << "| " << std::setw(2) << condid << " |        " << std::setw(6) << std::scientific
                << std::setprecision(3) << (integralvalue)(0) << "        |" << '\n';

      // set file name
      const std::string filename(Global::Problem::instance()->output_control_file()->file_name() +
                                 "." + label + "_integrals.csv");

      // open file in appropriate mode and write header at beginning
      std::ofstream file;
      if (scatratimint->step() == 0 and condid == 0)
      {
        file.open(filename.c_str(), std::fstream::trunc);
        file << "Step,Time," << label << "ID," << label << "Integral" << '\n';
      }
      else
      {
        file.open(filename.c_str(), std::fstream::app);
      }

      // write value of current domain or boundary integral to file
      file << scatratimint->step() << "," << scatratimint->time() << "," << condid << ','
           << std::scientific << std::setprecision(6) << (integralvalue)(0) << '\n';

      // close file
      file.close();
    }  // if(myrank_ == 0)

    if (condstring == "BoundaryIntegral")
      boundaryintegralvalues_[condid] = (integralvalue)(0);
    else if (condstring == "DomainIntegral")
      domainintegralvalues_[condid] = (integralvalue)(0);
    else
      FOUR_C_THROW("Invalid condition name!");
  }  // loop over all conditions

  // print finish line to screen
  if (conditions.size() and myrank == 0) std::cout << "+----+-------------------------+" << '\n';
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyCondition::evaluate_integrals(
    const ScaTraTimIntImpl* const scatratimint)
{
  Teuchos::ParameterList eleparams;

  // fill parameter list and set states in discretization
  prepare_evaluate(scatratimint, eleparams);

  // evaluate scalar integrals and domain integral for each subdomain
  for (const auto* condition : conditions_)
  {
    // extract condition ID
    const int condid = condition->parameters().get<int>("ConditionID");

    // determine the number of dofs on the current condition
    const int numdofpernode = numdofpernodepercondition_[condid];
    const int numscalpernode = numscalpercondition_[condid];

    // initialize result vector
    // first components = scalar integrals, last component = domain integral
    const int num_active_scalars = output_mean_grad_ ? numscalpernode : 0;
    const int num_micro_dis = output_micro_dis_ ? 1 : 0;
    Core::LinAlg::SerialDenseVector scalars(numdofpernode + 1 + num_active_scalars + num_micro_dis);

    // perform integration
    scatratimint->discret_->evaluate_scalars(eleparams, scalars, "TotalAndMeanScalar", condid);

    // extract domain integral
    domainintegral_[condid] = (scalars)[numdofpernode];

    // compute results
    for (int k = 0; k < numdofpernode; ++k)
    {
      totalscalars_[condid][k] = (scalars)[k];
      meanscalars_[condid][k] = (scalars)[k] / domainintegral_[condid];
    }
    if (output_mean_grad_)
    {
      for (int k = 0; k < numscalpercondition_[condid]; ++k)
      {
        const double mean_gradient = (scalars)[numdofpernode + 1 + k] / domainintegral_[condid];
        meangradients_[condid][k] = std::abs(mean_gradient) < 1.0e-10 ? 0.0 : mean_gradient;
      }
    }
    if (output_micro_dis_)
      micromeanscalars_[condid][0] = (scalars)[scalars.length() - 1] / domainintegral_[condid];
  }
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
ScaTra::OutputScalarsStrategyDomainAndCondition::OutputScalarsStrategyDomainAndCondition(
    const ScaTraTimIntImpl* scatratimint)
    : OutputScalarsStrategyBase(scatratimint),
      OutputScalarsStrategyDomain(scatratimint),
      OutputScalarsStrategyCondition(scatratimint)
{
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyDomainAndCondition::print_to_screen()
{
  OutputScalarsStrategyCondition::print_to_screen();
  OutputScalarsStrategyDomain::print_to_screen();
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
std::map<std::string, std::vector<double>>
ScaTra::OutputScalarsStrategyDomainAndCondition::prepare_csv_output()
{
  std::map<std::string, std::vector<double>> output_data;
  auto output_data_condition = OutputScalarsStrategyCondition::prepare_csv_output();
  auto output_data_domain = OutputScalarsStrategyDomain::prepare_csv_output();

  // merge both maps
  output_data.merge(output_data_condition);
  output_data.merge(output_data_domain);

  return output_data;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::OutputScalarsStrategyDomainAndCondition::evaluate_integrals(
    const ScaTraTimIntImpl* const scatratimint)
{
  // call base classes
  OutputScalarsStrategyCondition::evaluate_integrals(scatratimint);
  OutputScalarsStrategyDomain::evaluate_integrals(scatratimint);
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScalarHandler::setup(const ScaTraTimIntImpl* const scatratimint)
{
  // save reference to discretization for convenience
  const std::shared_ptr<Core::FE::Discretization>& discret = scatratimint->discretization();

  // initialize set of all number of dofs per node on this proc
  std::set<int> mynumdofpernode;

  // collect number of DoFs for each node on this proc
  for (int i = 0; i < discret->num_my_row_nodes(); i++)
    mynumdofpernode.insert(discret->num_dof(0, discret->l_row_node(i)));

  // number of different numbers of dofs on all procs
  int maxsize = 0;
  // number of different numbers of dofs on this procs
  int mysize = static_cast<int>(mynumdofpernode.size());
  // communicate
  maxsize = Core::Communication::max_all(mysize, discret->get_comm());

  // copy mynumdofpernode into std::vector for communication
  std::vector<int> vecmynumdofpernode(mynumdofpernode.begin(), mynumdofpernode.end());
  vecmynumdofpernode.resize(maxsize, 0);

  // initialize std::vector for communication
  std::vector<int> vecnumdofpernode(
      maxsize * Core::Communication::num_mpi_ranks(discret->get_comm()), 0);

  // communicate
  Core::Communication::gather_all(vecmynumdofpernode.data(), vecnumdofpernode.data(),
      static_cast<int>(vecmynumdofpernode.size()), discret->get_comm());

  // copy back into set
  for (int& ndofpernode : vecnumdofpernode)
    if (ndofpernode != 0) numdofpernode_.insert(ndofpernode);

  // check for equal number of Dofs on all nodes in whole discretization
  equalnumdof_ = (numdofpernode_.size() == 1);

  // done
  issetup_ = true;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
int ScaTra::ScalarHandler::max_num_dof_per_node() const { return *numdofpernode_.rbegin(); }

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
int ScaTra::ScalarHandler::num_dof_per_node_in_condition(
    const Core::Conditions::Condition& condition, const Core::FE::Discretization& discret) const
{
  check_is_setup();

  // get all nodes in condition
  const std::vector<int>* nodegids = condition.get_nodes();

  // initialize set of all number of dofs per node on all procs
  std::set<int> numdofpernode;
  // initialize set of all number of dofs per node on this proc
  std::set<int> mynumdofpernode;

  // determine all number of dofs per node on this proc
  for (int nodegid : *nodegids)
  {
    // if on this proc
    if (discret.node_row_map()->my_gid(nodegid))
    {
      // get node
      Core::Nodes::Node* curnode = discret.g_node(nodegid);
      // save number of dofs in set
      mynumdofpernode.insert(discret.num_dof(0, curnode));
    }
  }

  // number of different numbers of dofs on all procs
  int maxsize = 0;
  // number of different numbers of dofs on this procs
  int mysize = static_cast<int>(mynumdofpernode.size());
  // communicate
  maxsize = Core::Communication::max_all(mysize, discret.get_comm());

  // copy mynumdofpernode into std::vector for communication
  std::vector<int> vecmynumdofpernode(mynumdofpernode.begin(), mynumdofpernode.end());
  vecmynumdofpernode.resize(maxsize, 0);

  // initialize std::vector for communication
  std::vector<int> vecnumdofpernode(
      maxsize * Core::Communication::num_mpi_ranks(discret.get_comm()), 0);

  // communicate
  Core::Communication::gather_all(vecmynumdofpernode.data(), vecnumdofpernode.data(),
      static_cast<int>(vecmynumdofpernode.size()), discret.get_comm());

  // copy back into set
  for (int& ndofpernode : vecnumdofpernode)
    if (ndofpernode != 0) numdofpernode.insert(ndofpernode);

  // this is the maximum number of dofs per node within the discretization on all procs
  int maxnumdofpernode = *numdofpernode.rbegin();

  if (numdofpernode.size() != 1)
  {
    FOUR_C_THROW(
        "Different number of DOFs within condition. This is not supported. Split the condition "
        "in your input file!");
  }

  return maxnumdofpernode;
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
int ScaTra::ScalarHandler::num_dof_per_node() const
{
  check_is_setup();

  if (not equalnumdof_)
  {
    FOUR_C_THROW(
        "Number of DOFs per node is not equal for all nodes within the ScaTra discretization!\n"
        "Calling this method is not valid in this case!");
  }
  return *(numdofpernode_.rbegin());
}

/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
void ScaTra::ScalarHandler::check_is_setup() const
{
  if (not issetup_) FOUR_C_THROW("ScalarHandler is not set up. Call setup() first.");
}

FOUR_C_NAMESPACE_CLOSE
