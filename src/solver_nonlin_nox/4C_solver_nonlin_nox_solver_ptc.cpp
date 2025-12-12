// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_solver_ptc.hpp"  // class definition

#include "4C_fem_geometry_intersection_math.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_direction_factory.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_linearsystem_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_linesearch_factory.hpp"
#include "4C_solver_nonlin_nox_statustest_normf.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Direction_Generic.H>
#include <NOX_GlobalData.H>
#include <NOX_LineSearch_Generic.H>
#include <NOX_MeritFunction_Generic.H>
#include <NOX_Solver_SolverUtils.H>
#include <NOX_StatusTest_Generic.H>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::Solver::PseudoTransient::PseudoTransient(const Teuchos::RCP<::NOX::Abstract::Group>& grp,
    const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
    const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests,
    const Teuchos::RCP<Teuchos::ParameterList>& params)
    : NOX::Nln::Solver::LineSearchBased(grp, outerTests, innerTests, params),
      iTestPtr_(innerTests),
      xDot_(Teuchos::null),
      usePseudoTransientResidual_(false),
      calcDeltaInit_(false),
      isScalingOperator_(false),
      modelReductionRatio_(0.0)
{
  init();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::init()
{
  checkType = ::NOX::Solver::parseStatusTestCheckType(paramsPtr->sublist("Solver Options"));

  /* We use our own factories at this point to generate the line search
   * and the direction object. */
  LineSearchBased::init(iTestPtr_);

  const Teuchos::ParameterList& p_ptc = paramsPtr->sublist("Pseudo Transient");
  deltaInit_ = p_ptc.get<double>("deltaInit");
  if (deltaInit_ == 0.0)
    throw_error("init()", "The initial pseudo time step is not allowed to be equal to 0.0!");
  else if (deltaInit_ < 0.0)
    calcDeltaInit_ = true;
  delta_ = deltaInit_;
  invDelta_ = 1.0 / delta_;
  deltaMax_ = p_ptc.get<double>("deltaMax");
  deltaMin_ = p_ptc.get<double>("deltaMin");
  SER_alpha_ = p_ptc.get<double>("SER_alpha");
  scaleFactor_ = p_ptc.get<double>("ScalingFactor");
  pseudoTime_ = 0.0;

  maxPseudoTransientIterations_ = p_ptc.get<int>("Max Number of PTC Iterations");

  // get the time step control type
  const std::string& control_str = p_ptc.get<std::string>("Time Step Control");
  tscType_ = string_to_tsc_type(control_str);
  const std::string& norm_str = p_ptc.get<std::string>("Norm Type for TSC");
  normType_ = NOX::Nln::Aux::string_to_norm_type(norm_str);

  // get the scaling operator type
  const std::string& scaleop_str = p_ptc.get<std::string>("Scaling Type");
  scaleOpType_ = string_to_scale_op_type(scaleop_str);

  const std::string& build_scaling_op = p_ptc.get<std::string>("Build scaling operator");
  build_scaling_op_ = string_to_build_op_type(build_scaling_op);

  // create the scaling operator
  create_scaling_operator();

  // create the linear system pre/post operator
  create_lin_system_pre_post_operator();

  // create the group pre/post operator
  create_group_pre_post_operator();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::reset(const ::NOX::Abstract::Vector& initialGuess,
    const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
    const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests)
{
  solnPtr->setX(initialGuess);
  if (not outerTests.is_null()) testPtr = outerTests;
  if (not innerTests.is_null()) iTestPtr_ = innerTests;

  init();

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::reset(const ::NOX::Abstract::Vector& initialGuess,
    const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests)
{
  solnPtr->setX(initialGuess);
  if (not outerTests.is_null()) testPtr = outerTests;

  init();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::create_scaling_operator()
{
  if (is_scaling_operator()) return;

  switch (scaleOpType_)
  {
    case NOX::Nln::Solver::PseudoTransient::scale_op_identity:
    {
      // identity matrix
      const auto& epetraXPtr = dynamic_cast<const NOX::Nln::Vector&>(solnPtr->getX());
      scalingDiagOpPtr_ = Teuchos::make_rcp<Core::LinAlg::Vector<double>>(
          epetraXPtr.get_linalg_vector().get_map(), false);

      scalingDiagOpPtr_->put_scalar(1.0);

      isScalingOperator_ = true;

      break;
    }
    // get element based scaling operator
    case NOX::Nln::Solver::PseudoTransient::scale_op_element_based:
    {
      // Get element based contributions and assemble into SparseMatrix
      scalingMatrixOpPtr_ = Teuchos::rcp_dynamic_cast<NOX::Nln::Group>(solnPtr)
                                ->get_contributions_from_element_level();

      if (build_scaling_op_ == NOX::Nln::Solver::PseudoTransient::build_op_everyiter)
        isScalingOperator_ = false;
      else
        isScalingOperator_ = true;

      break;
    }
    case NOX::Nln::Solver::PseudoTransient::scale_op_cfl_diagonal:
    {
      FOUR_C_THROW("Not yet implemented!");
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown/unsupported scaling operator type!");
      break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::create_lin_system_pre_post_operator()
{
  prePostLinSysPtr_ = Teuchos::make_rcp<NOX::Nln::LinSystem::PrePostOp::PseudoTransient>(
      scalingDiagOpPtr_, scalingMatrixOpPtr_, *this);

  // The name of the direction method and the corresponding sublist has to match!
  const std::string dir_str(NOX::Nln::Aux::get_direction_method_list_name(*paramsPtr));
  if (paramsPtr->sublist("Direction").isSublist(dir_str))
  {
    // set the new pre/post operator for the linear system in the parameter list
    Teuchos::ParameterList& p_linsolver =
        paramsPtr->sublist("Direction").sublist(dir_str).sublist("Linear Solver");
    // get the current map. If there is no map, return a new empty one. (reference)
    NOX::Nln::LinSystem::PrePostOperator::Map& prePostLinSystemMap =
        NOX::Nln::LinSystem::PrePostOp::get_map(p_linsolver);
    // insert/replace the old pointer in the map
    prePostLinSystemMap[NOX::Nln::LinSystem::prepost_ptc] = prePostLinSysPtr_;
    /* Now the last thing to do is, that we have to reset the pre/post-operator in
     * the nln linear system class with the ptc pre/post-operator. */
    // Get the linear system
    Teuchos::RCP<NOX::Nln::Group> nlnSolnPtr = Teuchos::rcp_dynamic_cast<NOX::Nln::Group>(solnPtr);
    if (nlnSolnPtr.is_null())
      throw_error("create_lin_system_pre_post_operator", "The group cast failed!");
    /* Call the reset function of the nox nln linear system to reset the
     * corresponding pre/post operator map. */
    nlnSolnPtr->reset_lin_sys_pre_post_operator(p_linsolver, true);
  }
  else
  {
    std::ostringstream msg;
    msg << "The name of the \"Method\" in the \"Direction\" sublist has to match the name "
        << "of the corresponding sub-sublist! There is no \"Direction->" << dir_str
        << "\" sub-sublist!" << std::endl;
    throw_error("create_lin_system_pre_post_operator()", msg.str());
  }
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::create_group_pre_post_operator()
{
  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln group in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_grpOpt = paramsPtr->sublist("Group Options");
  // get the current map. If there is no map, return a new empty one. (reference)
  NOX::Nln::GROUP::PrePostOperator::Map& prePostGroupMap =
      NOX::Nln::GROUP::PrePostOp::get_map(p_grpOpt);
  // insert or replace the old pointer in the map
  prePostGroupPtr_ = Teuchos::make_rcp<NOX::Nln::GROUP::PrePostOp::PseudoTransient>(
      scalingDiagOpPtr_, scalingMatrixOpPtr_, *this);

  // set the modified map
  prePostGroupMap[NOX::Nln::GROUP::prepost_ptc] = prePostGroupPtr_;
  /* Now the last thing to do is, that we have to reset the pre/post-operator in
   * the nln group class with the ptc pre/post-operator. */
  // Get the nln group
  Teuchos::RCP<NOX::Nln::Group> nlnSoln = Teuchos::rcp_dynamic_cast<NOX::Nln::Group>(solnPtr);
  if (nlnSoln.is_null()) throw_error("create_group_pre_post_operator", "The group cast failed!");
  /* Call the reset function of the nox nln group to reset the corresponding
   * pre/post operator map. */
  nlnSoln->reset_pre_post_operator(p_grpOpt, true);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::StatusTest::StatusType NOX::Nln::Solver::PseudoTransient::step()
{
  observer->runPreIterate(*this);

  // On the first step do some initializations
  if (nIter == 0)
  {
    // Compute F of initial guess
    ::NOX::Abstract::Group::ReturnType rtype = solnPtr->computeF();
    if (rtype != ::NOX::Abstract::Group::Ok)
    {
      utilsPtr->out() << "::NOX::Solver::PseudoTransient::init - " << "Unable to compute F"
                      << std::endl;
      throw "NOX Error";
    }

    // Test the initial guess
    status = testPtr->checkStatus(*this, checkType);
    if ((status == ::NOX::StatusTest::Converged) and (utilsPtr->isPrintType(::NOX::Utils::Warning)))
    {
      utilsPtr->out() << "Warning: ::NOX::Solver::PseudoTransient::init() - "
                      << "The solution passed into the solver (either "
                      << "through constructor or reset method) "
                      << "is already converged! The solver will not "
                      << "attempt to solve this system since status is " << "flagged as converged."
                      << std::endl;
    }
    printUpdate();
  }
  else
  {
    // eventually create new scaling operator each iteration step
    create_scaling_operator();

    /* Adjust the pseudo time step, such it fits to the chosen line search
     * step length. */
    adjust_pseudo_time_step();
  }

  // First check status
  if (status != ::NOX::StatusTest::Unconverged)
  {
    observer->runPostIterate(*this);

    printUpdate();
    return status;
  }

  // Copy pointers into temporary references
  ::NOX::Abstract::Group& soln = *solnPtr;
  ::NOX::StatusTest::Generic& otest = *testPtr;

  // Update the pseudo time step size.
  update_pseudo_time_step();

  // Only necessary for the optional CFL scaling option
  compute_pseudo_velocity();

  /* Compute the direction for the update vector at the current
   * solution. Steady-state F is already computed so the only thing
   * to compute is J. The necessary non-linear solver dependent changes
   * to the Jacobian are done by using the run_post_compute_jacobian() routine of the
   * pre/post-operator object of the NOX::Nln::LinearSystem class and
   * its derived classes.
   *
   * See the NOX::Nln::LinSystem::PrePostOP::PseudoTransient class for
   * more information. */
  bool ok = directionPtr->compute(*dirPtr, soln, *this);

  if (not ok)
  {
    utilsPtr->out() << "NOX::Nln::Solver::PseudoTransient::step - unable to calculate direction"
                    << std::endl;
    status = ::NOX::StatusTest::Failed;

    observer->runPostIterate(*this);
    printUpdate();
    return status;
  }

  // Update iteration count.
  nIter++;

  // Copy current soln to the old soln
  // (Changes the owner of the linear system. New owner is the oldSolnPtr_)
  *oldSolnPtr = *solnPtr;

  /* Additional line search (optional).
   * You have to specify an inner status test to use it, otherwise
   * it will do nothing, but updating x with the specified default
   * step length (same behavior as for the "Full Step" case). */
  // Use the pseudo transient residual during the line search procedure.
  usePseudoTransientResidual_ = true;
  // Do line search and compute new soln.
  ok = lineSearchPtr->compute(soln, stepSize, *dirPtr, *this);
  /* call the computeF routine again, to be sure that it has been evaluated!
   * (this becomes necessary for a Full Step method call!) */
  solnPtr->computeF();
  usePseudoTransientResidual_ = false;
  // evaluate the model reduction ratio if desired
  if (ok) eval_model_reduction_ratio();

  if (!ok)
  {
    if (stepSize == 0.0)
    {
      utilsPtr->out() << "NOX::Nln::Solver::PseudoTransient::step - line search failed"
                      << std::endl;
      status = ::NOX::StatusTest::Failed;

      observer->runPostIterate(*this);
      printUpdate();
      return status;
    }
    else if (utilsPtr->isPrintType(::NOX::Utils::Warning))
      utilsPtr->out()
          << "NOX::Nln::Solver::PseudoTransient::step - using recovery step for line search"
          << std::endl;
  }

  // Compute F for new current solution.
  ::NOX::Abstract::Group::ReturnType rtype = soln.computeF();
  if (rtype != ::NOX::Abstract::Group::Ok)
  {
    utilsPtr->out() << "::NOX::Solver::PseudoTransient::iterate - unable to compute F" << std::endl;
    status = ::NOX::StatusTest::Failed;

    observer->runPostIterate(*this);
    printUpdate();
    return status;
  }

  // Evaluate the current status.
  status = otest.checkStatus(*this, checkType);

  observer->runPostIterate(*this);
  printUpdate();
  return status;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::eval_model_reduction_ratio()
{
  if (tscType_ != tsc_mrr) return;

  double fref = globalDataPtr->getMeritFunction()->computef(*oldSolnPtr);
  double fnew = globalDataPtr->getMeritFunction()->computef(*solnPtr);
  Teuchos::RCP<::NOX::Abstract::Vector> sdirPtr = dirPtr->clone(::NOX::DeepCopy);
  sdirPtr->scale(stepSize);
  double modelnew = globalDataPtr->getMeritFunction()->computeQuadraticModel(*sdirPtr, *oldSolnPtr);

  modelReductionRatio_ = (fref - fnew) / (fref - modelnew);
  if (utilsPtr->isPrintType(::NOX::Utils::Details))
  {
    utilsPtr->out() << "(" << fref << " - " << fnew << ") / (" << fref << " - " << modelnew << ")"
                    << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
::NOX::StatusTest::StatusType NOX::Nln::Solver::PseudoTransient::solve()
{
  isPtcSolve_ = true;
  observer->runPreSolve(*this);

  // Iterate until converged or failed
  while (status == ::NOX::StatusTest::Unconverged) step();

  Teuchos::ParameterList& outputParams = paramsPtr->sublist("Output");
  outputParams.set("Nonlinear Iterations", nIter);
  outputParams.set("2-Norm of Residual", solnPtr->getNormF());

  observer->runPostSolve(*this);
  isPtcSolve_ = false;
  return status;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::compute_pseudo_velocity()
{
  if (nIter >= maxPseudoTransientIterations_ or scaleOpType_ == scale_op_identity) return;

  if (xDot_.is_null()) xDot_ = solnPtr->getX().clone(::NOX::ShapeCopy);

  // computing xDot_ using finite differences
  if (nIter == 0)
    xDot_->init(0.0);
  else
    xDot_->update(invDelta_, solnPtr->getX(), -invDelta_, oldSolnPtr->getX(), 0.0);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::adjust_pseudo_time_step()
{
  /* Check the step-length. If the step-length is not equal 1.0,
   * we adjust the pseudo time step size by using a least squares
   * approximation. */
  if (stepSize == 1.0 or nIter >= maxPseudoTransientIterations_) return;

  if (utilsPtr->isPrintType(::NOX::Utils::Details))
  {
    utilsPtr->out() << "*--- Adjust Pseudo Time Step ---*" << std::endl
                    << "| Previous dt_ptc:    " << std::setw(9) << std::setprecision(3)
                    << std::scientific << delta_ << " |" << std::endl;
  }

  Teuchos::RCP<NOX::Nln::Group> oldNlnSolnPtr =
      Teuchos::rcp_dynamic_cast<NOX::Nln::Group>(oldSolnPtr);
  if (oldNlnSolnPtr.is_null())
    throw_error("update_pseudo_time_step", "Dynamic cast to NOX::Nln::Group failed!");
  oldNlnSolnPtr->adjust_pseudo_time_step(delta_, stepSize, *dirPtr, *this);

  if (utilsPtr->isPrintType(::NOX::Utils::Details))
  {
    utilsPtr->out() << "| Corrected dt_ptc:   " << std::setw(9) << std::setprecision(3)
                    << std::scientific << delta_ << " |" << std::endl
                    << "*-------------------------------*" << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::update_pseudo_time_step()
{
  if (nIter < maxPseudoTransientIterations_)
  {
    deltaOld_ = delta_;

    if (nIter == 0)
    {
      if (calcDeltaInit_)
      {
        double normF = 0.0;
        if (scaleOpType_ == scale_op_identity || scaleOpType_ == scale_op_element_based)
        {
          normF = solnPtr->getF().norm(normType_);
        }
        else
        {
          Teuchos::RCP<::NOX::Abstract::Vector> scaledRHS = solnPtr->getF().clone(::NOX::DeepCopy);
          auto& epetraScaledRHS = dynamic_cast<NOX::Nln::Vector&>(*scaledRHS);
          epetraScaledRHS.get_linalg_vector().reciprocal_multiply(
              1.0, *scalingDiagOpPtr_, epetraScaledRHS.get_linalg_vector(), 0.0);
          normF = epetraScaledRHS.norm(normType_);
        }
        if (normF > Core::Geo::TOL12)
          deltaInit_ = 1.0 / (normF * normF);
        else
          deltaInit_ = 1.0;
      }
      delta_ = deltaInit_;
    }
    else
    {
      // -----------------------------------------------------------------
      // desired time stepping control method
      // -----------------------------------------------------------------
      switch (tscType_)
      {
        // ---------------------------------------------------------------
        // switched evolution relaxation
        // ---------------------------------------------------------------
        case tsc_set:
        {
          if (normType_ == ::NOX::Abstract::Vector::TwoNorm)
            delta_ =
                deltaOld_ * std::pow((oldSolnPtr->getNormF() / solnPtr->getNormF()), SER_alpha_);
          else
            delta_ = deltaOld_ *
                     std::pow(oldSolnPtr->getF().norm(normType_) / solnPtr->getF().norm(normType_),
                         SER_alpha_);
          break;
        }
        // ---------------------------------------------------------------
        // temporal truncation error
        // ---------------------------------------------------------------
        case tsc_tte:
        {
          throw_error("UpdatePseudoTimeStep()",
              "The \"Temporal Truncation Error\" method is not yet implemented!");
          break;
        }
        // ---------------------------------------------------------------
        // model reduction ratio
        // ---------------------------------------------------------------
        case tsc_mrr:
        {
          double normFOld = oldSolnPtr->getF().norm(normType_);
          double normF = solnPtr->getF().norm(normType_);
          double ratioF = normFOld / normF;
          // calculated the corrected old inverse mu
          double muinv = deltaOld_ * normFOld;

          /* If the normF Status test is already converged and no additional
           * step length modification has been necessary, we will just increase
           * the pseudo time step.
           * This is meant to speed up the convergence close to the solution. */
          if (stepSize == 1.0 and
              get_status<NOX::Nln::StatusTest::NormF>() == ::NOX::StatusTest::Converged)
            muinv *= 4.0;
          /* if the model performed badly: reduce the pseudo time step by a
           * factor tau_red, where 0.25 <= tau_red <= 0.8.
           * (ToDo supposed to become an input parameter) */
          else if (modelReductionRatio_ < 0.2)
            muinv *= std::min(std::max(0.25, ratioF), 0.8);
          /* if the model performed well: increase the pseudo time step by a
           * factor tau_inc, where 1.25 <= tau_inc <= 4.0.
           * (ToDo supposed to become an input parameter) */
          else if (modelReductionRatio_ > 0.8)
            muinv *= std::max(std::min(4.0, ratioF), 1.25);

          // update the pseudo time step
          delta_ = muinv / normF;
          if (utilsPtr->isPrintType(::NOX::Utils::Details))
          {
            utilsPtr->out() << "*--- Model Reduction Ratio -----------------------------*"
                            << std::endl
                            << "| MRR: " << std::setw(9) << std::setprecision(3) << std::scientific
                            << modelReductionRatio_ << std::setw(41) << "|" << std::endl
                            << "| dt_ptc(k-1) --> (    muinv *  normFinv) --> dt_ptc(k) |"
                            << std::endl
                            << "| " << std::setw(11) << std::setprecision(3) << std::scientific
                            << deltaOld_ << " --> (" << std::setw(9) << std::setprecision(3)
                            << std::scientific << muinv << " * " << std::setw(9)
                            << std::setprecision(3) << std::scientific << 1.0 / normF << ") --> "
                            << std::setw(9) << std::setprecision(3) << std::scientific << delta_
                            << " |" << std::endl
                            << "*-------------------------------------------------------*"
                            << std::endl;
          }
          break;
        }
        default:
        {
          throw_error("UpdatePseudoTimeStep()", "Unknown time step control type.");
          break;
        }
      }
    }

    // invert delta_ and check possible thresholds
    if (delta_ > deltaMax_)
      invDelta_ = 0.0;
    else
      invDelta_ = 1.0 / delta_;

    if (delta_ < deltaMin_)
    {
      delta_ = deltaMin_;
      invDelta_ = 1.0 / delta_;
    }

    // increase the pseudoTime_ counter
    pseudoTime_ += delta_;
  }
  /* if the maximum PTC iteration number is reached, we switch PTC off and
   * use a standard line search based solution procedure. */
  else
  {
    delta_ = std::numeric_limits<double>::infinity();
    invDelta_ = 0.0;
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const double& NOX::Nln::Solver::PseudoTransient::get_inverse_pseudo_time_step() const
{
  return invDelta_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const double& NOX::Nln::Solver::PseudoTransient::get_scaling_factor() const { return scaleFactor_; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const enum NOX::Nln::Solver::PseudoTransient::ScaleOpType&
NOX::Nln::Solver::PseudoTransient::get_scaling_operator_type() const
{
  return scaleOpType_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& NOX::Nln::Solver::PseudoTransient::get_scaling_diag_operator()
    const
{
  if (scalingDiagOpPtr_.is_null())
    throw_error("get_scaling_diag_operator", "The scaling operator is not initialized!");

  return *scalingDiagOpPtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::Solver::PseudoTransient::use_pseudo_transient_residual() const
{
  return usePseudoTransientResidual_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::Nln::Solver::PseudoTransient::is_ptc_solve() const { return isPtcSolve_; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::printUpdate()
{
  double normSoln = 0;
  double normStep = 0;

  // Print the status test parameters at each iteration if requested
  if ((status == ::NOX::StatusTest::Unconverged) &&
      (utilsPtr->isPrintType(::NOX::Utils::OuterIterationStatusTest)))
  {
    utilsPtr->out() << ::NOX::Utils::fill(82) << "\n";
    utilsPtr->out() << "-- Status Test Results --\n";
    testPtr->print(utilsPtr->out());
    utilsPtr->out() << ::NOX::Utils::fill(82) << "\n";
  }

  // All processes participate in the computation of these norms...
  if (utilsPtr->isPrintType(::NOX::Utils::OuterIteration))
  {
    normSoln = solnPtr->getNormF();
    normStep = (nIter > 0) ? dirPtr->norm() : 0;
  }

  // ...But only the print process actually prints the result.
  // ------ standard output ------------------------------------------
  if (utilsPtr->isPrintType(::NOX::Utils::OuterIteration) and
      (utilsPtr->isPrintType(::NOX::Utils::OuterIterationStatusTest) or
          utilsPtr->isPrintType(::NOX::Utils::InnerIteration)))
  {
    utilsPtr->out() << "\n" << ::NOX::Utils::fill(82) << "\n";
    utilsPtr->out() << "-- Nonlinear Solver Step " << nIter << " -- \n";
    utilsPtr->out() << "||F|| = " << utilsPtr->sciformat(normSoln);
    utilsPtr->out() << "  step = " << utilsPtr->sciformat(stepSize);
    utilsPtr->out() << "  dx = " << utilsPtr->sciformat(normStep);
    if (delta_ == -1.0)
      utilsPtr->out() << "  dt_ptc = auto";
    else
      utilsPtr->out() << "  dt_ptc = " << utilsPtr->sciformat(delta_);
    if (status == ::NOX::StatusTest::Converged) utilsPtr->out() << " (Converged!)";
    if (status == ::NOX::StatusTest::Failed) utilsPtr->out() << " (Failed!)";
    utilsPtr->out() << "\n" << ::NOX::Utils::fill(82) << "\n" << std::endl;
  }
  // ------ short output ---------------------------------------------
  else if (utilsPtr->isPrintType(::NOX::Utils::OuterIteration))
  {
    // print the head line
    if (nIter == 0)
    {
      utilsPtr->out() << std::setw(4) << "#It" << std::setw(13) << "||F||_2" << std::setw(13)
                      << "step" << std::setw(13) << "||dx||_2" << std::setw(13) << "dt_ptc\n";
      utilsPtr->out() << ::NOX::Utils::fill(60, '^') << "\n";
    }
    utilsPtr->out() << std::setw(4) << nIter;
    utilsPtr->out() << "  " << utilsPtr->sciformat(normSoln);
    utilsPtr->out() << "  " << utilsPtr->sciformat(stepSize);
    utilsPtr->out() << "  " << utilsPtr->sciformat(normStep);
    if (delta_ == -1.0)
      utilsPtr->out() << std::setw(13) << "auto";
    else
      utilsPtr->out() << "  " << utilsPtr->sciformat(delta_);
    if (status == ::NOX::StatusTest::Converged) utilsPtr->out() << " (Converged!)";
    if (status == ::NOX::StatusTest::Failed) utilsPtr->out() << " (Failed!)";
    utilsPtr->out() << std::endl;
  }

  // Print the final parameter values of the status test
  if ((status != ::NOX::StatusTest::Unconverged) &&
      (utilsPtr->isPrintType(::NOX::Utils::OuterIterationStatusTest)))
  {
    utilsPtr->out() << ::NOX::Utils::fill(82) << "\n";
    utilsPtr->out() << "-- Final Status Test Results --\n";
    testPtr->print(utilsPtr->out());
    utilsPtr->out() << ::NOX::Utils::fill(82) << "\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::Solver::PseudoTransient::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utilsPtr->isPrintType(::NOX::Utils::Error))
  {
    std::ostringstream msg;
    msg << "ERROR - NOX::Nln::Solver::PseudoTransient::" << functionName << " - " << errorMsg
        << std::endl;
    FOUR_C_THROW("{}", msg.str());
  }
  throw "NOX Error";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LinSystem::PrePostOp::PseudoTransient::PseudoTransient(
    Teuchos::RCP<Core::LinAlg::Vector<double>>& scalingDiagOpPtr,
    Teuchos::RCP<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr,
    const NOX::Nln::Solver::PseudoTransient& ptcsolver)
    : ptcsolver_(ptcsolver),
      scaling_diag_op_ptr_(scalingDiagOpPtr),
      scaling_matrix_op_ptr_(scalingMatrixOpPtr)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinSystem::PrePostOp::PseudoTransient::run_post_compute_jacobian(
    Core::LinAlg::SparseOperator& jac, const Core::LinAlg::Vector<double>& x,
    const NOX::Nln::LinearSystem& linsys)
{
  if (not ptcsolver_.is_ptc_solve()) return;

  // get the type of the jacobian
  const NOX::Nln::LinSystem::OperatorType& jactype = linsys.get_jacobian_operator_type();

  switch (jactype)
  {
    case NOX::Nln::LinSystem::LinalgSparseMatrix:
    {
      // First cast the Core::LinAlg::SparseOperator and do an additional sanity
      // check.
      Core::LinAlg::SparseMatrix* jacPtr = dynamic_cast<Core::LinAlg::SparseMatrix*>(&jac);
      if (jacPtr == nullptr)
        FOUR_C_THROW(
            "Something strange happened: The jacobian has not the "
            "operator type defined in the linear system object!");

      modify_jacobian(*jacPtr);

      break;
    }
    default:
    {
      FOUR_C_THROW("Unsupported jacobian operator type: {}",
          NOX::Nln::LinSystem::operator_type_to_string(jactype));
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinSystem::PrePostOp::PseudoTransient::run_post_compute_f_and_jacobian(
    Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac,
    const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys)
{
  run_post_compute_jacobian(jac, x, linsys);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::LinSystem::PrePostOp::PseudoTransient::modify_jacobian(
    Core::LinAlg::SparseMatrix& jac)
{
  // get the inverse pseudo time step
  const double& deltaInv = ptcsolver_.get_inverse_pseudo_time_step();
  const NOX::Nln::Solver::PseudoTransient::ScaleOpType& scaleoptype =
      ptcsolver_.get_scaling_operator_type();
  const double& scaleFactor = ptcsolver_.get_scaling_factor();

  switch (scaleoptype)
  {
    case NOX::Nln::Solver::PseudoTransient::scale_op_identity:
    {
      /* Build the scaling operator V and multiply it with the inverse
       * pseudo time step. Finally, we modify the jacobian.
       *
       *        (\delta^{-1} \boldsymbol{I} + \boldsymbol{J}) */
      Core::LinAlg::Vector<double> v(*scaling_diag_op_ptr_);
      // Scale v with scaling factor
      v.scale(deltaInv * scaleFactor);
      // get the diagonal terms of the jacobian
      auto diag = std::make_shared<Core::LinAlg::Vector<double>>(jac.row_map(), false);
      jac.extract_diagonal_copy(*diag);
      diag->update(1.0, v, 1.0);
      // Finally modify the jacobian
      jac.replace_diagonal_values(*diag);
      break;
    }
    case NOX::Nln::Solver::PseudoTransient::scale_op_element_based:
    {
      /* Build the scaling operator V and multiply it with the inverse
       * pseudo time step. Finally, we modify the jacobian.
       *
       *        (\delta^{-1} \boldsymbol{V} + \boldsymbol{J}) */

      scaling_matrix_op_ptr_->complete();
      Core::LinAlg::matrix_add(*scaling_matrix_op_ptr_, false, scaleFactor * deltaInv, jac, 1.0);
      jac.complete();

      break;
    }
    case NOX::Nln::Solver::PseudoTransient::scale_op_cfl_diagonal:
    {
      FOUR_C_THROW("Not yet implemented!");
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown/unsupported scaling operator type!");
      break;
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::GROUP::PrePostOp::PseudoTransient::PseudoTransient(
    Teuchos::RCP<Core::LinAlg::Vector<double>>& scalingDiagOpPtr,
    Teuchos::RCP<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr,
    const NOX::Nln::Solver::PseudoTransient& ptcsolver)
    : ptcsolver_(ptcsolver),
      scaling_diag_op_ptr_(scalingDiagOpPtr),
      scaling_matrix_op_ptr_(scalingMatrixOpPtr),
      is_pseudo_transient_residual_(false)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::Vector NOX::Nln::GROUP::PrePostOp::PseudoTransient::eval_pseudo_transient_f_update(
    const NOX::Nln::Group& grp)
{
  // get the current trial point
  NOX::Nln::Vector xUpdate(grp.get_x(), ::NOX::DeepCopy);

  // get the old x vector
  const ::NOX::Abstract::Group& oldGrp = ptcsolver_.getPreviousSolutionGroup();
  const auto& xOld = dynamic_cast<const NOX::Nln::Vector&>(oldGrp.getX());

  /* Calculate the difference between the old and the new solution vector.
   * This is equivalent to the search direction scaled with the
   * step size. */
  xUpdate.update(-1.0, xOld, 1.0);

  const NOX::Nln::Solver::PseudoTransient::ScaleOpType& scaleoptype =
      ptcsolver_.get_scaling_operator_type();
  switch (scaleoptype)
  {
    case NOX::Nln::Solver::PseudoTransient::scale_op_identity:
    {
      NOX::Nln::Vector v(*scaling_diag_op_ptr_);
      v.scale(ptcsolver_.get_inverse_pseudo_time_step());
      xUpdate.scale(v);

      break;
    }
    case NOX::Nln::Solver::PseudoTransient::scale_op_element_based:
    {
      scaling_matrix_op_ptr_->multiply(
          false, xUpdate.get_linalg_vector(), xUpdate.get_linalg_vector());
      xUpdate.scale(ptcsolver_.get_inverse_pseudo_time_step());

      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown scaling operator type!");
      break;
    }
  }
  return xUpdate;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::GROUP::PrePostOp::PseudoTransient::run_post_compute_f(
    Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp)
{
  if (not ptcsolver_.is_ptc_solve()) return;

  const bool use_pseudo_transient_residual = ptcsolver_.use_pseudo_transient_residual();

  /* If we need no pseudo transient residual or if the transient residual
   * has already been added, we can skip this function. */
  if (not use_pseudo_transient_residual or is_pseudo_transient_residual_) return;

  NOX::Nln::Vector fUpdate(eval_pseudo_transient_f_update(grp));

  // add the transient part
  F.update(1.0, fUpdate.get_linalg_vector(), 1.0);

  // set the flag
  is_pseudo_transient_residual_ = true;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::GROUP::PrePostOp::PseudoTransient::run_pre_compute_f(
    Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp)
{
  if (not ptcsolver_.is_ptc_solve()) return;

  // If the current rhs has not been calculated, yet.
  if (!grp.isF())
  {
    is_pseudo_transient_residual_ = false;
    return;
  }

  /* Recalculate the static residual, if the current right hand side
   * has already been modified, though we need the static residual. */
  if (ptcsolver_.use_pseudo_transient_residual() or !is_pseudo_transient_residual_) return;

  NOX::Nln::Vector fUpdate(eval_pseudo_transient_f_update(grp));

  // subtract the transient part
  F.update(-1.0, fUpdate.get_linalg_vector(), 1.0);

  // set flag
  is_pseudo_transient_residual_ = false;

  return;
}

FOUR_C_NAMESPACE_CLOSE
