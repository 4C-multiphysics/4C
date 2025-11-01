// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_timint_impl.hpp"

#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_mat_fourier.hpp"
#include "4C_thermo_aux.hpp"
#include "4C_thermo_ele_action.hpp"
#include "4C_thermo_timint.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <sstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Thermo::TimIntImpl::TimIntImpl(const Teuchos::ParameterList& ioparams,
    const Teuchos::ParameterList& tdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : TimInt(ioparams, tdynparams, xparams, actdis, solver, output),
      pred_(Teuchos::getIntegralValue<Thermo::PredEnum>(tdynparams, "PREDICT")),
      itertype_(Teuchos::getIntegralValue<Thermo::NonlinSolTech>(tdynparams, "NLNSOL")),
      normtypetempi_(Teuchos::getIntegralValue<Thermo::ConvNorm>(tdynparams, "NORM_TEMP")),
      normtypefres_(Teuchos::getIntegralValue<Thermo::ConvNorm>(tdynparams, "NORM_RESF")),
      combtempifres_(Teuchos::getIntegralValue<Thermo::BinaryOp>(tdynparams, "NORMCOMBI_RESFTEMP")),
      iternorm_(Teuchos::getIntegralValue<Thermo::VectorNorm>(tdynparams, "ITERNORM")),
      itermax_(tdynparams.get<int>("MAXITER")),
      itermin_(tdynparams.get<int>("MINITER")),
      divcontype_(Teuchos::getIntegralValue<Thermo::DivContAct>(tdynparams, "DIVERCONT")),
      divcontrefinelevel_(0),
      divcontfinesteps_(0),
      toltempi_(tdynparams.get<double>("TOLTEMP")),
      tolfres_(tdynparams.get<double>("TOLRES")),
      iter_(-1),
      resetiter_(0),
      normcharforce_(0.0),
      normchartemp_(0.0),
      normfres_(0.0),
      normtempi_(1e6),
      tempi_(nullptr),
      timer_("", true),
      fres_(nullptr),
      freact_(nullptr)
{
  // create empty residual force vector
  fres_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), false);

  // create empty reaction force vector of full length
  freact_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), false);

  // iterative temperature increments IncT_{n+1}
  // also known as residual temperatures
  tempi_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

  // We assume the material is not time dependent, thus can be assembled here once and for all!
  // Also assume the same bulk material at every element, we pre-initialize with the conductivity
  // size.
  // Here we just grab the first local element we can find and initialize our data structures
  auto material = std::dynamic_pointer_cast<Mat::Fourier>(discret_->l_row_element(0)->material());
  const size_t columns = material->conductivity(discret_->l_row_element(0)->id()).size();
  Core::LinAlg::FEVector<double> overlapping_element_material_vector =
      Core::LinAlg::FEVector<double>(*discret_->element_col_map(), columns, true);

  auto get_element_material_vector = [&](Core::Elements::Element& ele)
  {
    auto thermo_material = std::dynamic_pointer_cast<Mat::Fourier>(ele.material());
    const std::vector<double>& conductivity = thermo_material->conductivity(ele.id());

    FOUR_C_ASSERT(columns == conductivity.size(),
        "Number of material vectors has to be the same size as conductivity values given.");

    for (size_t col = 0; col < conductivity.size(); col++)
      overlapping_element_material_vector.replace_global_value(ele.id(), col, conductivity[col]);
  };
  discret_->evaluate(get_element_material_vector);
  overlapping_element_material_vector.complete();

  conductivity_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      Core::LinAlg::MultiVector<double>(*discret_->node_row_map(), columns, true));

  for (const auto& node : discret_->my_row_node_range())
  {
    for (size_t col = 0; col < columns; col++)
    {
      Core::LinAlg::Vector<double> element_material(
          *overlapping_element_material_vector.get_ref_of_epetra_fevector()(col));
      double nodal_material = 0.0;

      for (auto ele : node.adjacent_elements())
      {
        const int global_element_id = ele.global_id();
        const int local_element_id = element_material.get_map().lid(global_element_id);
        nodal_material = nodal_material + element_material.local_values_as_span()[local_element_id];
      }

      const int num_elements = node.adjacent_elements().size();
      conductivity_->replace_global_value(node.global_id(), col, nodal_material / num_elements);
    }
  }

  // setup mortar coupling
  if (Global::Problem::instance()->get_problem_type() == Core::ProblemType::thermo)
  {
    if (actdis->has_condition("Mortar"))
    {
      adaptermeshtying_ =
          std::make_shared<Coupling::Adapter::CouplingMortar>(Global::Problem::instance()->n_dim(),
              Global::Problem::instance()->mortar_coupling_params(),
              Global::Problem::instance()->contact_dynamic_params(),
              Global::Problem::instance()->spatial_approximation_type());

      std::vector<int> coupleddof(1, 1);
      adaptermeshtying_->setup(actdis, actdis, nullptr, coupleddof, "Mortar", actdis->get_comm(),
          Global::Problem::instance()->function_manager(),
          Global::Problem::instance()->binning_strategy_params(),
          Global::Problem::instance()->discretization_map(),
          Global::Problem::instance()->output_control_file(),
          Global::Problem::instance()->spatial_approximation_type(), false, false, 0, 0);
      adaptermeshtying_->evaluate();
    }
  }
}


/*----------------------------------------------------------------------*
 | build linear system tangent matrix, rhs/force residual    dano 02/11 |
 | Monolithic TSI accesses the linearised thermo problem                |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::evaluate()
{
  // builds tangent, residual and applies DBC
  evaluate_rhs_tang_residual();
  prepare_system_for_newton_solve();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::prepare_time_step()
{
  predict_step();
  prepare_step();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::predict_step()
{
  switch (pred_)
  {
    case Thermo::pred_consttemp:
    {
      predict_const_temp_consist_rate();
      break;
    }
    case Thermo::pred_consttemprate:
    {
      predict_const_temp_rate();
      break;
    }
    case Thermo::pred_tangtemp:
    {
      predict_tang_temp_consist_rate();
      break;
    }
    default:
      FOUR_C_THROW("Trouble in determining predictor {}.", pred_);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::prepare_step()
{
  // set iteration step to 0
  iter_ = 0;

  // apply Dirichlet BCs
  apply_dirichlet_bc(timen_, tempn_, raten_, false);

  // compute residual forces fres_ and stiffness tang_
  evaluate_rhs_tang_residual();

  // extract reaction forces
  // reactions are negative to balance residual on DBC
  freact_->update(-1.0, *fres_, 0.0);
  dbcmaps_->insert_other_vector(*dbcmaps_->extract_other_vector(*zeros_), *freact_);

  // blank residual at DOFs on Dirichlet BC
  dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *fres_);

  // split norms
  // build residual force norm
  normfres_ = Thermo::Aux::calculate_vector_norm(iternorm_, *fres_);

  // determine characteristic norms
  // we set the minimum of calc_ref_norm_force() and #tolfres_, because
  // we want to prevent the case of a zero characteristic fnorm
  normcharforce_ = calc_ref_norm_force();
  if (normcharforce_ == 0.0) normcharforce_ = tolfres_;
  normchartemp_ = calc_ref_norm_temperature();
  if (normchartemp_ == 0.0) normchartemp_ = toltempi_;

  print_predictor();
}


/*----------------------------------------------------------------------* |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::predict_const_temp_rate()
{
  // constant predictor
  tempn_->update(1.0, *temp_(0), 0.0);
  raten_->update(1.0, *rate_(0), 0.0);
}


/*----------------------------------------------------------------------* |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::predict_tang_temp_consist_rate()
{
  // initialise
  tempn_->update(1.0, *temp_(0), 0.0);
  raten_->update(1.0, *rate_(0), 0.0);
  tempi_->put_scalar(0.0);

  // for temperature increments on Dirichlet boundary
  std::shared_ptr<Core::LinAlg::Vector<double>> dbcinc =
      Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

  // copy last converged temperatures
  dbcinc->update(1.0, *temp_(0), 0.0);

  // get Dirichlet values at t_{n+1}
  apply_dirichlet_bc(timen_, dbcinc, nullptr, false);

  // subtract the temperatures of the last converged step
  // DBC-DOFs hold increments of current step
  // free-DOFs hold zeros
  dbcinc->update(-1.0, *temp_(0), 1.0);

  // compute residual forces fres_ and tangent tang_
  // at tempn_, etc which are unchanged
  evaluate_rhs_tang_residual();

  // add linear reaction forces to residual
  {
    // linear reactions
    std::shared_ptr<Core::LinAlg::Vector<double>> freact =
        Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
    tang_->multiply(false, *dbcinc, *freact);

    // add linear reaction forces due to prescribed Dirichlet BCs
    fres_->update(1.0, *freact, 1.0);
  }

  // extract reaction forces
  freact_->update(-1.0, *fres_, 0.0);  // reactions are negative
  dbcmaps_->insert_other_vector(*dbcmaps_->extract_other_vector(*zeros_), *freact_);

  // blank residual at DOFs on Dirichlet BC
  dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *fres_);

  // make negative residual
  // K . DT = -fres = -(fint - fext)
  fres_->scale(-1.0);

  // apply Dirichlet BCs to system of equations
  tempi_->put_scalar(0.0);
  tang_->complete();
  Core::LinAlg::apply_dirichlet_to_system(
      *tang_, *tempi_, *fres_, *zeros_, *(dbcmaps_->cond_map()));

  // solve for tempi_
  // Solve K_Teffdyn . IncT = -R  ===>  IncT_{n+1}
  solver_->reset();
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(tang_, tempi_, fres_, solver_params);
  solver_->reset();

  // build residual temperature norm
  normtempi_ = Thermo::Aux::calculate_vector_norm(iternorm_, *tempi_);

  // set Dirichlet increments in temperature increments
  tempi_->update(1.0, *dbcinc, 1.0);

  // update end-point temperatures etc
  update_iter_incrementally();
  // tempn_->Update(1.0, *tempi_, 1.0);

  // MARK:
  // temperature rates unset on Dirichlet boundary

  // reset to zero
  tempi_->put_scalar(0.0);

  // reset anything that needs to be reset at the element level
  {
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    p.set<Thermo::Action>("action", Thermo::calc_thermo_reset_istep);
    // set the total time
    p.set("total time", time_[0]);
    // go to elements
    discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
    discret_->clear_state();
  }
}


/*----------------------------------------------------------------------*
 | converged                                                bborn 08/09 |
 *----------------------------------------------------------------------*/
bool Thermo::TimIntImpl::converged()
{
  // verify: #normcharforce_ has been delivered strictly larger than zero
  if (normcharforce_ <= 0.0)
  {
    FOUR_C_THROW("Characteristic force norm {} must be strictly larger than 0", normcharforce_);
  }
  // verify: #normchartemp_ has been delivered strictly larger than zero
  if (normchartemp_ <= 0.0)
  {
    FOUR_C_THROW(
        "Characteristic temperature norm {} must be strictly larger than 0", normchartemp_);
  }

  // check for single norms
  bool convtemp = false;
  bool convfres = false;

  // residual forces
  switch (normtypefres_)
  {
    case Thermo::convnorm_abs:
      convfres = normfres_ < tolfres_;
      break;
    case Thermo::convnorm_rel:
      convfres = normfres_ < std::max(normcharforce_ * tolfres_, 1e-15);
      break;
    case Thermo::convnorm_mix:
      convfres =
          ((normfres_ < tolfres_) or (normfres_ < std::max(normcharforce_ * tolfres_, 1e-15)));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }

  // residual temperature
  switch (normtypetempi_)
  {
    case Thermo::convnorm_abs:
      convtemp = normtempi_ < toltempi_;
      break;
    case Thermo::convnorm_rel:
      convtemp = normtempi_ < std::max(normchartemp_ * toltempi_, 1e-15);
      break;
    case Thermo::convnorm_mix:
      convtemp =
          ((normtempi_ < toltempi_) or (normtempi_ < std::max(normchartemp_ * toltempi_, 1e-15)));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual temperatures!");
      break;
  }

  // combine temperature-like and force-like residuals
  bool conv = false;
  if (combtempifres_ == Thermo::bop_and)
    conv = convtemp and convfres;
  else if (combtempifres_ == Thermo::bop_or)
    conv = convtemp or convfres;
  else
    FOUR_C_THROW("Something went terribly wrong with binary operator!");

  // return things
  return conv;
}

/*----------------------------------------------------------------------*
 | solve equilibrium                                        bborn 08/09 |
 *----------------------------------------------------------------------*/
Thermo::ConvergenceStatus Thermo::TimIntImpl::solve()
{
  // choose solution technique in accordance with user's will
  switch (itertype_)
  {
    case Thermo::soltech_newtonfull:
      return newton_full();
    // catch problems
    default:
      FOUR_C_THROW("Solution technique \"{}\" is not implemented", itertype_);
      return Thermo::conv_nonlin_fail;  // compiler happiness
  }
}

/*----------------------------------------------------------------------*
 | solution with full Newton-Raphson iteration              bborn 08/09 |
 *----------------------------------------------------------------------*/
Thermo::ConvergenceStatus Thermo::TimIntImpl::newton_full()
{
  // we do a Newton-Raphson iteration here.
  // the specific time integration has set the following
  // --> On #fres_ is the positive force residuum
  // --> On #tang_ is the effective dynamic tangent matrix

  // check whether we have a sanely filled tangent matrix
  if (not tang_->filled())
  {
    FOUR_C_THROW("Effective tangent matrix must be filled here");
  }

  // initialise equilibrium loop
  iter_ = 1;
  normfres_ = calc_ref_norm_force();
  // normtempi_ was already set in predictor; this is strictly >0
  timer_.reset();

  // Do mortar condensation
  if (adaptermeshtying_ != nullptr) adaptermeshtying_->mortar_condensation(tang_, *fres_);

  // equilibrium iteration loop
  while (((not converged()) and (iter_ <= itermax_)) or (iter_ <= itermin_))
  {
    // make negative residual
    fres_->scale(-1.0);

    // apply Dirichlet BCs to system of equations
    tempi_->put_scalar(0.0);  // Useful? depends on solver and more
    Core::LinAlg::apply_dirichlet_to_system(
        *tang_, *tempi_, *fres_, *zeros_, *(dbcmaps_->cond_map()));

    // Solve for tempi_
    // Solve K_Teffdyn . IncT = -R  ===>  IncT_{n+1}

    Core::LinearSolver::Parameters::compute_solver_parameters(*discret_, solver_->params());
    solver_->params().set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
        "Material", conductivity_);

    Core::LinAlg::SolverParams solver_params;
    if (solveradapttol_ and (iter_ > 1))
    {
      solver_params.nonlin_tolerance = tolfres_;
      solver_params.nonlin_residual = normfres_;
      solver_params.lin_tol_better = solveradaptolbetter_;
    }
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(tang_, tempi_, fres_, solver_params);
    solver_->reset_tolerance();

    // recover condensed variables
    if (adaptermeshtying_ != nullptr) adaptermeshtying_->mortar_recover(*tang_, *tempi_);

    // update end-point temperatures etc
    update_iter(iter_);

    // compute residual forces #fres_ and tangent #tang_
    // whose components are globally oriented
    evaluate_rhs_tang_residual();

    blank_dirichlet_and_calc_norms();

    // print stuff
    print_newton_iter();

    // increment equilibrium loop index
    iter_ += 1;
  }  // end equilibrium loop

  // correct iteration counter
  iter_ -= 1;

  return newton_full_error_check();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::blank_dirichlet_and_calc_norms()
{
  // extract reaction forces
  // reactions are negative to balance residual on DBC
  freact_->update(-1.0, *fres_, 0.0);
  // copy the dbc onto freact_,
  // everything that is not DBC node ("OtherVector") is blanked
  dbcmaps_->insert_other_vector(*dbcmaps_->extract_other_vector(*zeros_), *freact_);

  // blank residual at DOFs on Dirichlet BC
  // DBC node do not enter the residual, because values are known at the nodes
  dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *fres_);

  // do mortar condensation
  if (adaptermeshtying_ != nullptr) adaptermeshtying_->mortar_condensation(tang_, *fres_);

  // build residual force norm
  normfres_ = Thermo::Aux::calculate_vector_norm(iternorm_, *fres_);
  // build residual temperature norm
  normtempi_ = Thermo::Aux::calculate_vector_norm(iternorm_, *tempi_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Thermo::ConvergenceStatus Thermo::TimIntImpl::newton_full_error_check()
{
  // do some error checks
  if ((iter_ >= itermax_) and (divcontype_ == Thermo::divcont_stop))
  {
    // write restart output of last converged step before stopping
    output(true);

    FOUR_C_THROW("Newton unconverged in {} iterations", iter_);
    return Thermo::conv_nonlin_fail;
  }
  else if ((iter_ >= itermax_) and (divcontype_ == Thermo::divcont_continue))
  {
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      Core::IO::cout << "Newton unconverged in " << iter_ << " iterations, continuing"
                     << Core::IO::endl;
    return Thermo::conv_success;
  }
  else if ((iter_ >= itermax_) and divcontype_ == Thermo::divcont_halve_step)
  {
    halve_time_step();
    return Thermo::conv_fail_repeat;
  }
  else if (divcontype_ == Thermo::divcont_repeat_step or
           divcontype_ == Thermo::divcont_repeat_simulation)
  {
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      FOUR_C_THROW(
          "Fatal failure in newton_full_error_check()! divcont_repeat_step and "
          "divcont_repeat_simulation not implemented for Thermo");
    return Thermo::conv_nonlin_fail;
  }
  // if everything is fine print to screen and return
  if (converged())
  {
    check_for_time_step_increase();
    return Thermo::conv_success;
  }
  else
    return Thermo::conv_nonlin_fail;

}  // NewtonFull()


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::halve_time_step()
{
  const double old_dt = dt();
  const double new_dt = old_dt * 0.5;
  const int endstep = num_step() + (num_step() - step()) + 1;
  set_dt(new_dt);
  set_timen(time_old() + new_dt);
  set_num_step(endstep);
  reset_step();
  // TODO limit the maximum number of refinement levels?
  // go down one refinement level
  divcontrefinelevel_++;
  divcontfinesteps_ = 0;

  // remember number of iterations
  resetiter_ += iter_;
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    Core::IO::cout << "Nonlinear solver failed to converge in step " << step()
                   << ". Divide timestep in half. "
                   << "Old time step: " << old_dt << Core::IO::endl
                   << "New time step: " << new_dt << Core::IO::endl
                   << Core::IO::endl;
}

/*-----------------------------------------------------------------------------*
 * check, if according to divercont flag                            proell 09/18
 * time step size can be increased
 *-----------------------------------------------------------------------------*/
void Thermo::TimIntImpl::check_for_time_step_increase()
{
  const int maxnumfinestep = 4;

  if (divcontype_ != Thermo::divcont_halve_step)
    return;
  else if (divcontrefinelevel_ != 0)
  {
    // increment for the current, converged step
    divcontfinesteps_++;
    if (divcontfinesteps_ >= maxnumfinestep)
    {
      // increase the step size if the remaining number of steps is a even number
      if (((num_step() - step()) % 2) == 0 and num_step() != step())
      {
        if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
          Core::IO::cout << "Nonlinear solver successful. Double timestep size!" << Core::IO::endl;

        // step up one refinement level
        divcontrefinelevel_--;
        divcontfinesteps_ = 0;
        // update total number of steps and next time step
        const int endstep = num_step() - (num_step() - step()) / 2;
        set_num_step(endstep);
        set_dt(dt() * 2.0);
      }
    }
  }
}


/*----------------------------------------------------------------------*
 | Prepare system for solving with Newton's method          bborn 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::prepare_system_for_newton_solve()
{
  // extract reaction forces
  // reactions are negative to balance residual on DBC
  freact_->update(-1.0, *fres_, 0.0);
  dbcmaps_->insert_other_vector(*dbcmaps_->extract_other_vector(*zeros_), *freact_);

  // make the residual negative
  fres_->scale(-1.0);
  // blank residual at DOFs on Dirichlet BCs, fres_=0 at nodes with DBC
  dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *fres_);

  // apply Dirichlet BCs to system of equations
  tempi_->put_scalar(0.0);  // Useful? depends on solver and more
  // at dofs with DBC change tang_:
  // blank all off-diagonal terms and put 1s at diagonal terms of tang_
  Core::LinAlg::apply_dirichlet_to_system(
      *tang_, *tempi_, *fres_, *zeros_, *(dbcmaps_->cond_map()));

  // final sip
  return;
}  // prepare_system_for_newton_solve()


/*----------------------------------------------------------------------*
 | Update iteration                                         bborn 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::update_iter(const int iter  //!< iteration counter
)
{
  // we need to do an incremental update (expensive)
  // in the very first iteration (i.e. predictor) of a Newton loop
  // to protect the Dirichlet BCs and to achieve consistent
  // behaviour across all predictors
  // HINT: Sorry, this comment was added delayed and might be inaccurate.
  if (iter <= 1)
  {
    update_iter_incrementally();
  }
  else
  {
    update_iter_iteratively();
  }

  // morning is broken
  return;
}  // UpdateIter()


/*----------------------------------------------------------------------*
 | Update iteration incrementally with prescribed           bborn 08/09 |
 | residual temperatures                                                |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::update_iter_incrementally(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>
        tempi  //!< input residual temperatures
)
{
  // select residual temperatures
  if (tempi != nullptr)
    // tempi_ = \f$\Delta{T}^{<k>}_{n+1}\f$
    tempi_->update(1.0, *tempi, 0.0);  // set the new solution we just got
  else
    tempi_->put_scalar(0.0);

  // Update using #tempi_
  update_iter_incrementally();

  // leave this place
  return;
}  // update_iter_incrementally()


/*----------------------------------------------------------------------*
 | update time step                                         bborn 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::update()
{
  // update temperature and temperature rate
  // after this call we will have tempn_ == temp_ (temp_{n+1} == temp_n), etc.
  update_step_state();
  // update everything on the element level
  update_step_element();
  // update time and step
  update_step_time();
  // correct iteration counter by adding all reset iterations
  iter_ += resetiter_;
  resetiter_ = 0;
  return;

}  // update()


/*----------------------------------------------------------------------*
 | update Newton step                                        dano 02/11 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::update_newton(std::shared_ptr<const Core::LinAlg::Vector<double>> tempi)
{
  // Yes, this is complicated. But we have to be very careful
  // here. The field solver always expects an increment only. And
  // there are Dirichlet conditions that need to be preserved. So take
  // the sum of increments we get from NOX and apply the latest
  // increment only.
  update_iter_incrementally(tempi);
  return;

}  // UpdateNewton()


/*----------------------------------------------------------------------*
 | print to screen                                          bborn 08/09 |
 | originally by lw 12/07                                               |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::print_predictor()
{
  // only master processor
  if ((Core::Communication::my_mpi_rank(discret_->get_comm()) == 0) and printscreen_ and
      (step_old() % printscreen_ == 0))
  {
    // relative check of force residual
    if (normtypefres_ == Thermo::convnorm_rel)
    {
      std::cout << "Predictor thermo scaled res-norm " << normfres_ / normcharforce_ << std::endl;
    }
    // absolute check of force residual
    else if (normtypefres_ == Thermo::convnorm_abs)
    {
      std::cout << "Predictor thermo absolute res-norm " << normfres_ << std::endl;
    }
    // mixed absolute-relative check of force residual
    else if (normtypefres_ == Thermo::convnorm_mix)
    {
      std::cout << "Predictor thermo mixed res-norm "
                << std::min(normfres_, normfres_ / normcharforce_) << std::endl;
    }
    // default
    else
    {
      FOUR_C_THROW("You should not turn up here.");
    }
    // print it, now
    fflush(stdout);
  }

  // leave your hat on
  return;

}  // print_predictor()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen and error file  bborn 08/09 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::print_newton_iter()
{
  // print to standard out
  if ((Core::Communication::my_mpi_rank(discret_->get_comm()) == 0) and printscreen_ and
      (step_old() % printscreen_ == 0))
  {
    if (iter_ == 1) print_newton_iter_header(stdout);
    print_newton_iter_text(stdout);
  }

}  // print_newton_iter()


/*----------------------------------------------------------------------*
 | print header                                             bborn 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::print_newton_iter_header(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter converged state etc
  oss << std::setw(6) << "numiter";

  // different style due relative or absolute error checking
  // temperature
  switch (normtypefres_)
  {
    case Thermo::convnorm_rel:
      oss << std::setw(18) << "rel-res-norm";
      break;
    case Thermo::convnorm_abs:
      oss << std::setw(18) << "abs-res-norm";
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(18) << "mix-res-norm";
      break;
    default:
      FOUR_C_THROW("Unknown type of convergence check for residual forces.");
      break;
  }

  switch (normtypetempi_)
  {
    case Thermo::convnorm_rel:
      oss << std::setw(18) << "rel-temp-norm";
      break;
    case Thermo::convnorm_abs:
      oss << std::setw(18) << "abs-temp-norm";
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(18) << "mix-temp-norm";
      break;
    default:
      FOUR_C_THROW("Unknown type of convergence check for residual temperatures.");
      break;
  }

  // add solution time
  oss << std::setw(14) << "wct";

  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);

  // nice to have met you
  return;
}  // print_newton_iter_header()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen                 bborn 08/09 |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::print_newton_iter_text(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter converged state etc
  oss << std::setw(7) << iter_;

  // different style due relative or absolute error checking
  // temperature
  switch (normtypefres_)
  {
    case Thermo::convnorm_rel:
      oss << std::setw(18) << std::setprecision(5) << std::scientific << normfres_ / normcharforce_;
      break;
    case Thermo::convnorm_abs:
      oss << std::setw(18) << std::setprecision(5) << std::scientific << normfres_;
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << std::min(normfres_, normfres_ / normcharforce_);
      break;
    default:
      FOUR_C_THROW("Unknown type of convergence check for residual forces.");
      break;
  }

  switch (normtypetempi_)
  {
    case Thermo::convnorm_rel:
      oss << std::setw(18) << std::setprecision(5) << std::scientific << normtempi_ / normchartemp_;
      break;
    case Thermo::convnorm_abs:
      oss << std::setw(18) << std::setprecision(5) << std::scientific << normtempi_;
      break;
    case Thermo::convnorm_mix:
      oss << std::setw(18) << std::setprecision(5) << std::scientific
          << std::min(normtempi_, normtempi_ / normchartemp_);
      break;
    default:
      FOUR_C_THROW("Unknown type of convergence check for residual temperatures.");
      break;
  }

  // add solution time
  oss << std::setw(14) << std::setprecision(2) << std::scientific << timer_.totalElapsedTime(true);

  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);

  // nice to have met you
  return;
}  // print_newton_iter_text()


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::TimIntImpl::print_step()
{
  if ((Core::Communication::my_mpi_rank(discret_->get_comm()) == 0) and printscreen_ and
      (step_old() % printscreen_ == 0))
  {
    std::cout << "Finalised: step " << step_ << " | nstep " << stepmax_ << " | time " << time_[0]
              << " | dt " << dt_[0] << " | numiter " << iter_ + resetiter_ << std::endl;
    std::cout << "------------------------------------------------------------------------------\n";
  }
}

FOUR_C_NAMESPACE_CLOSE
