// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timint_statics.hpp"

#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN


/*======================================================================*/
/* constructor */
Solid::TimIntStatics::TimIntStatics(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams, std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::LinAlg::Solver> contactsolver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : TimIntImpl(timeparams, ioparams, sdynparams, xparams, actdis, solver, contactsolver, output),
      fint_(nullptr),
      fintn_(nullptr),
      fext_(nullptr),
      fextn_(nullptr)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
  return;
}

/*----------------------------------------------------------------------------------------------*
 * Initialize this class                                                            rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntStatics::init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver)
{
  // call init() in base class
  Solid::TimIntImpl::init(timeparams, sdynparams, xparams, actdis, solver);

  auto dyntype = Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdynparams, "DYNAMICTYPE");
  const Inpar::Solid::PreStress pre_stress_type =
      Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
          Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");

  if (pre_stress_type != Inpar::Solid::PreStress::none &&
      dyntype != Inpar::Solid::DynamicType::Statics)
  {
    FOUR_C_THROW(
        "Paranoia Error: PRESTRESS is only allowed in combinations with DYNAMICTYPE Statics!!");
  }

  // info to user
  if (myrank_ == 0 && bool(printscreen_))
  {
    // check if we are in prestressing mode
    if (pre_stress_type == Inpar::Solid::PreStress::mulf)
      Core::IO::cout << "with static MULF prestress" << Core::IO::endl;
    else
      Core::IO::cout << "with statics" << Core::IO::endl;
  }

  // have a nice day
  return;
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntStatics::setup()
{
  // call setup() in base class
  Solid::TimIntImpl::setup();

  // create force vectors

  // internal force vector F_{int;n+1} at new time
  fintn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // external force vector F_{n+1} at new time
  fextn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // internal force vector F_{int;n} at new time
  fint_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // external force vector F_{n} at new time
  fext_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  return;
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant displacements
 * and consistent velocities and displacements */
void Solid::TimIntStatics::predict_const_dis_consist_vel_acc()
{
  // constant predictor : displacement in domain
  disn_->update(1.0, *(*dis_)(0), 0.0);

  // new end-point velocities, these stay zero in static calculation
  veln_->put_scalar(0.0);

  // new end-point accelerations, these stay zero in static calculation
  accn_->put_scalar(0.0);

  // reset the residual displacement
  disi_->put_scalar(0.0);

  // watch out
  return;
}

/*----------------------------------------------------------------------*/
/* linear extrapolation of displacement field */
void Solid::TimIntStatics::predict_const_vel_consist_acc()
{
  // for the first step we don't have any history to do
  // an extrapolation. Hence, we do TangDis
  if (step_ == 0)
  {
    predict_tang_dis_consist_vel_acc();
    return;
  }
  else
  {
    // Displacement increment over last time step
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_inc =
        std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
    disp_inc->update((*dt_)[0], *(*vel_)(0), 0.);
    Core::LinAlg::apply_dirichlet_to_system(*disp_inc, *zeros_, *(dbcmaps_->cond_map()));
    disn_->update(1.0, *(*dis_)(0), 0.0);
    disn_->update(1., *disp_inc, 1.);
    veln_->update(1.0, *(*vel_)(0), 0.0);
    accn_->update(1.0, *(*acc_)(0), 0.0);
    disi_->put_scalar(0.0);
    return;
  }
}

/*----------------------------------------------------------------------*/
/* quadratic extrapolation of displacement field */
void Solid::TimIntStatics::predict_const_acc()
{
  // for the first step we don't have any history to do
  // an extrapolation. Hence, we do TangDis
  if (step_ == 0)
  {
    predict_tang_dis_consist_vel_acc();
    return;
  }
  else if (step_ == 1)
  {
    predict_const_vel_consist_acc();
    return;
  }
  else
  {
    // Displacement increment over last time step
    std::shared_ptr<Core::LinAlg::Vector<double>> disp_inc =
        std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
    disp_inc->update((*dt_)[0], *(*vel_)(0), 0.);
    disp_inc->update(.5 * (*dt_)[0] * (*dt_)[0], *(*acc_)(0), 1.);
    Core::LinAlg::apply_dirichlet_to_system(*disp_inc, *zeros_, *(dbcmaps_->cond_map()));
    disn_->update(1.0, *(*dis_)(0), 0.0);
    disn_->update(1., *disp_inc, 1.);
    veln_->update(1.0, *(*vel_)(0), 0.0);
    accn_->update(1.0, *(*acc_)(0), 0.0);
    disi_->put_scalar(0.0);
    return;
  }
}

/*----------------------------------------------------------------------*/
/* evaluate residual force and its stiffness, ie derivative
 * with respect to end-point displacements \f$D_{n+1}\f$ */
void Solid::TimIntStatics::evaluate_force_stiff_residual(Teuchos::ParameterList& params)
{
  // get info about prediction step from parameter list
  bool predict = false;
  if (params.isParameter("predict")) predict = params.get<bool>("predict");

  // initialize stiffness matrix to zero
  stiff_->zero();

  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->put_scalar(0.0);
  apply_force_stiff_external(timen_, (*dis_)(0), *disn_, *(*vel_)(0), *fextn_, stiff_);

  // additional external forces are added (e.g. interface forces)
  fextn_->update(1.0, *fifc_, 1.0);

  // ************************** (2) INTERNAL FORCES ***************************

  // initialize internal forces
  fintn_->put_scalar(0.0);

  // ordinary internal force and stiffness
  apply_force_stiff_internal(timen_, (*dt_)[0], disn_, disi_, *veln_, fintn_, stiff_, params);

  // apply forces and stiffness due to constraints
  Teuchos::ParameterList pcon;  // apply empty parameterlist, no scaling necessary
  apply_force_stiff_constraint(timen_, (*dis_)(0), disn_, fintn_, stiff_, pcon);

  // add forces and stiffness due to Cardiovascular0D bcs
  Teuchos::ParameterList pwindk;
  pwindk.set("time_step_size", (*dt_)[0]);
  apply_force_stiff_cardiovascular0_d(timen_, disn_, fintn_, stiff_, pwindk);

  // add forces and stiffness due to spring dashpot condition
  Teuchos::ParameterList psprdash;
  apply_force_stiff_spring_dashpot(stiff_, fintn_, disn_, *veln_, predict, psprdash);

  // ************************** (3) INERTIAL FORCES ***************************
  // This is statics, so there are no inertial forces.

  // ************************** (4) DAMPING FORCES ****************************
  // This is statics, so there are no viscous damping forces.

  // ******************** Finally, put everything together ********************

  // build residual  Res = F_{int;n+1}
  //                     - F_{ext;n+1}
  fres_->update(-1.0, *fextn_, 0.0);
  fres_->update(1.0, *fintn_, 1.0);

  // build pure structural residual (only LS with EAS)
  if (fresn_str_ != nullptr)
  {
    fresn_str_->update(1., *fintn_str_, 0.);
    fresn_str_->update(-1., *fextn_, 1.);
    Core::LinAlg::apply_dirichlet_to_system(*fresn_str_, *zeros_, *(dbcmaps_->cond_map()));
  }

  // build tangent matrix : effective dynamic stiffness matrix
  //    K_{Teffdyn} = K_{T}
  // i.e. do nothing here

  // apply forces and stiffness due to contact / meshtying
  apply_force_stiff_contact_meshtying(stiff_, fres_, disn_, predict);

  // close stiffness matrix
  stiff_->complete();

  return;
}

/*----------------------------------------------------------------------*/
/* Evaluate/define the residual force vector #fres_ for
 * relaxation solution with solve_relaxation_linear */
void Solid::TimIntStatics::evaluate_force_stiff_residual_relax(Teuchos::ParameterList& params)
{
  // compute residual forces #fres_ and stiffness #stiff_
  evaluate_force_stiff_residual(params);

  // overwrite the residual forces #fres_ with interface load
  fres_->update(-1.0, *fifc_, 0.0);
}

/*----------------------------------------------------------------------*/
/* Evaluate residual */
void Solid::TimIntStatics::evaluate_force_residual()
{
  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->put_scalar(0.0);
  apply_force_external(timen_, (*dis_)(0), disn_, *(*vel_)(0), *fextn_);

  // additional external forces are added (e.g. interface forces)
  fextn_->update(1.0, *fifc_, 1.0);

  // ************************** (2) INTERNAL FORCES ***************************

  // initialize internal forces
  fintn_->put_scalar(0.0);

  // ordinary internal force and stiffness
  apply_force_internal(timen_, (*dt_)[0], disn_, disi_, *veln_, fintn_);

  // ************************** (3) INERTIAL FORCES ***************************
  // This is statics, so there are no inertial forces.

  // ************************** (4) DAMPING FORCES ****************************
  // This is statics, so there are no viscous damping forces.

  // ******************** Finally, put everything together ********************

  // build residual  Res = F_{int;n+1}
  //                     - F_{ext;n+1}
  fres_->update(-1.0, *fextn_, 0.0);
  fres_->update(1.0, *fintn_, 1.0);

  // build pure structural residual (only LS with EAS)
  if (fresn_str_ != nullptr)
  {
    fresn_str_->update(1., *fintn_str_, 0.);
    fresn_str_->update(-1., *fextn_, 1.);
    Core::LinAlg::apply_dirichlet_to_system(*fresn_str_, *zeros_, *(dbcmaps_->cond_map()));
  }

  return;
}

/*----------------------------------------------------------------------*/
/* calculate characteristic/reference norms for forces
 * originally by lw */
double Solid::TimIntStatics::calc_ref_norm_force()
{
  // The reference norms are used to scale the calculated iterative
  // displacement norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  // norm of the internal forces
  double fintnorm = 0.0;
  fintnorm = Solid::calculate_vector_norm(iternorm_, *fintn_);

  // norm of the external forces
  double fextnorm = 0.0;
  fextnorm = Solid::calculate_vector_norm(iternorm_, *fextn_);

  // norm of reaction forces
  double freactnorm = 0.0;
  freactnorm = Solid::calculate_vector_norm(iternorm_, *freact_);

  // return char norm
  return std::max(fintnorm, std::max(fextnorm, freactnorm));
}

/*----------------------------------------------------------------------*/
void Solid::TimIntStatics::update_iter_incrementally()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->update(1.0, *disi_, 1.0);
}

/*----------------------------------------------------------------------*/
/* iterative iteration update of state */
void Solid::TimIntStatics::update_iter_iteratively()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->update(1.0, *disi_, 1.0);

  // bye
  return;
}

/*----------------------------------------------------------------------*/
/* update after time step */
void Solid::TimIntStatics::update_step_state()
{
  // calculate pseudo velocity and acceleration for predictor and/or binning
  // of the contact interface before updates
  if (pred_ == Inpar::Solid::pred_constvel || pred_ == Inpar::Solid::pred_constacc ||
      have_contact_meshtying())
    veln_->update(1. / (*(*dt_)(0)), *disn_, -1. / (*(*dt_)(0)), *(*dis_)(0), 0.);
  if (pred_ == Inpar::Solid::pred_constacc)
    accn_->update(1. / (*(*dt_)(0)), *veln_, -1. / (*(*dt_)(0)), *(*vel_)(0), 0.);

  // update state
  // new displacements at t_{n+1} -> t_n
  //    D_{n} := D_{n+1}
  dis_->update_steps(*disn_);

  // new velocities at t_{n+1} -> t_n
  //    V_{n} := V_{n+1}
  vel_->update_steps(*veln_);
  // new accelerations at t_{n+1} -> t_n
  //    A_{n} := A_{n+1}
  acc_->update_steps(*accn_);

  // update constraints
  update_step_constraint();

  // update Cardiovascular0D
  update_step_cardiovascular0_d();

  // update constraints
  update_step_spring_dashpot();

  // update contact / meshtying
  update_step_contact_meshtying();

  // update new external force
  //    F_{ext;n} := F_{ext;n+1}
  fext_->update(1.0, *fextn_, 0.0);

  // update new internal force
  //    F_{int;n} := F_{int;n+1}
  fint_->update(1.0, *fintn_, 0.0);

  // look out
  return;
}

/*----------------------------------------------------------------------*/
/* update after time step after output on element level*/
// update anything that needs to be updated at the element level
void Solid::TimIntStatics::update_step_element()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // other parameters that might be needed by the elements
  p.set("total time", timen_);
  p.set("delta time", (*dt_)[0]);
  // p.set("alpha f", theta_);
  // action for elements
  p.set("action", "calc_struct_update_istep");
  // go to elements
  discret_->clear_state();
  discret_->set_state("displacement", *(*dis_)(0));

  discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
  discret_->clear_state();
}

/*----------------------------------------------------------------------*/
/* read restart forces */
void Solid::TimIntStatics::read_restart_force() { return; }

/*----------------------------------------------------------------------*/
/* write internal and external forces for restart */
void Solid::TimIntStatics::write_restart_force(
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
{
  output->write_vector("fexternal", fext_);
  output->write_vector("fint", fint_);

  // This restart output is needed in case of a static pre-simulation has to be restartet with
  // dynamic time integration
  output->write_vector("finert", zeros_);
}

/*---------------------------------------------------------------*/
/* Apply Dirichlet boundary conditions on provided state vectors */
void Solid::TimIntStatics::apply_dirichlet_bc(const double time,
    std::shared_ptr<Core::LinAlg::Vector<double>> dis,
    std::shared_ptr<Core::LinAlg::Vector<double>> vel,
    std::shared_ptr<Core::LinAlg::Vector<double>> acc, bool recreatemap)
{
  // call base apply_dirichlet_bc
  Solid::TimInt::apply_dirichlet_bc(time, dis, vel, acc, recreatemap);

  // statics: set velocities and accelerations to zero
  if (vel != nullptr) vel->put_scalar(0.0);
  if (acc != nullptr) acc->put_scalar(0.0);

  return;
}


FOUR_C_NAMESPACE_CLOSE
