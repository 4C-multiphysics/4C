// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timint_ab2.hpp"

#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_mortar_strategy_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* Constructor */
Solid::TimIntAB2::TimIntAB2(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams,
    // const Teuchos::ParameterList& ab2params,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::LinAlg::Solver> contactsolver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : TimIntExpl(timeparams, ioparams, sdynparams, xparams, actdis, solver, contactsolver, output),
      fextn_(nullptr),
      fintn_(nullptr),
      fviscn_(nullptr),
      fcmtn_(nullptr),
      frimpn_(nullptr)
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
void Solid::TimIntAB2::init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver)
{
  // call init() in base class
  Solid::TimIntExpl::init(timeparams, sdynparams, xparams, actdis, solver);


  // info to user
  if (myrank_ == 0)
  {
    std::cout << "with Adams-Bashforth 2nd order" << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntAB2::setup()
{
  // call setup() in base class
  Solid::TimIntExpl::setup();


  // determine mass, damping and initial accelerations
  determine_mass_damp_consist_accel();

  // resize of multi-step quantities
  resize_m_step();

  // allocate force vectors
  fextn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  fintn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  fviscn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  fcmtn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);
  frimpn_ = Core::LinAlg::create_vector(*dof_row_map_view(), true);

  return;
}


/*----------------------------------------------------------------------*/
/* Resizing of multi-step quantities */
void Solid::TimIntAB2::resize_m_step()
{
  // resize time and step size fields
  time_->resize(-1, 0, (*time_)[0]);
  dt_->resize(-1, 0, (*dt_)[0]);

  // resize state vectors, AB2 is a 2-step method, thus we need two
  // past steps at t_{n} and t_{n-1}
  dis_->resize(-1, 0, dof_row_map_view(), true);
  vel_->resize(-1, 0, dof_row_map_view(), true);
  acc_->resize(-1, 0, dof_row_map_view(), true);

  return;
}

/*----------------------------------------------------------------------*/
/* Integrate step */
int Solid::TimIntAB2::integrate_step()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // things to be done before integrating
  pre_solve();

  // time this step
  timer_->reset();

  const double dt = (*dt_)[0];    // \f$\Delta t_{n}\f$
  const double dto = (*dt_)[-1];  // \f$\Delta t_{n-1}\f$

  // new displacements \f$D_{n+}\f$
  disn_->update(1.0, *(*dis_)(0), 0.0);
  disn_->update((2.0 * dt * dto + dt * dt) / (2.0 * dto), *(*vel_)(0), -(dt * dt) / (2.0 * dto),
      *(*vel_)(-1), 1.0);

  // new velocities \f$V_{n+1}\f$
  veln_->update(1.0, *(*vel_)(0), 0.0);
  veln_->update((2.0 * dt * dto + dt * dt) / (2.0 * dto), *(*acc_)(0), -(dt * dt) / (2.0 * dto),
      *(*acc_)(-1), 1.0);

  // *********** time measurement ***********
  double dtcpu = timer_->wallTime();
  // *********** time measurement ***********

  // apply Dirichlet BCs
  apply_dirichlet_bc(timen_, disn_, veln_, nullptr, false);

  // initialise stiffness matrix to zero
  stiff_->zero();

  // build new external forces
  fextn_->put_scalar(0.0);
  apply_force_external(timen_, disn_, *veln_, *fextn_);

  // TIMING
  // double dtcpu = timer_->wallTime();

  // initialise internal forces
  fintn_->put_scalar(0.0);

  // ordinary internal force and stiffness
  {
    // displacement increment in step
    Core::LinAlg::Vector<double> disinc = Core::LinAlg::Vector<double>(*disn_);
    disinc.update(-1.0, *(*dis_)(0), 1.0);
    // internal force
    apply_force_internal(
        timen_, dt, disn_, Core::Utils::shared_ptr_from_ref(disinc), *veln_, fintn_);
  }

  // *********** time measurement ***********
  dtele_ = timer_->wallTime() - dtcpu;
  // *********** time measurement ***********

  // viscous forces due Rayleigh damping
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    damp_->multiply(false, *veln_, *fviscn_);
  }

  // *********** time measurement ***********
  dtcpu = timer_->wallTime();
  // *********** time measurement ***********

  // contact or meshtying forces
  if (have_contact_meshtying())
  {
    fcmtn_->put_scalar(0.0);

    if (cmtbridge_->have_meshtying())
      cmtbridge_->mt_manager()->get_strategy().apply_force_stiff_cmt(
          disn_, stiff_, fcmtn_, stepn_, 0, false);
    if (cmtbridge_->have_contact())
      cmtbridge_->contact_manager()->get_strategy().apply_force_stiff_cmt(
          disn_, stiff_, fcmtn_, stepn_, 0, false);
  }

  // *********** time measurement ***********
  dtcmt_ = timer_->wallTime() - dtcpu;
  // *********** time measurement ***********

  // determine time derivative of linear momentum vector,
  // ie \f$\dot{P} = M \dot{V}_{n=1}\f$
  frimpn_->update(1.0, *fextn_, -1.0, *fintn_, 0.0);

  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    frimpn_->update(-1.0, *fviscn_, 1.0);
  }

  if (have_contact_meshtying())
  {
    frimpn_->update(1.0, *fcmtn_, 1.0);
  }

  // *********** time measurement ***********
  dtcpu = timer_->wallTime();
  // *********** time measurement ***********

  // obtain new accelerations \f$A_{n+1}\f$
  {
    FOUR_C_ASSERT(mass_->filled(), "Mass matrix has to be completed");
    // blank linear momentum zero on DOFs subjected to DBCs
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *frimpn_);
    // get accelerations
    accn_->put_scalar(0.0);

    // in case of no lumping or if mass matrix is a BlockSparseMatrix, use solver
    if (lumpmass_ == false ||
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(mass_) == nullptr)
    {
      // linear solver call
      // refactor==false: This is not necessary, because we always
      // use the same constant mass matrix, which was firstly factorised
      // in TimInt::determine_mass_damp_consist_accel
      Core::LinAlg::SolverParams solver_params;
      solver_params.reset = true;
      solver_->solve(mass_, accn_, frimpn_, solver_params);
    }

    // direct inversion based on lumped mass matrix
    else
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> massmatrix =
          std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(mass_);
      std::shared_ptr<Core::LinAlg::Vector<double>> diagonal =
          Core::LinAlg::create_vector(*dof_row_map_view(), true);
      massmatrix->extract_diagonal_copy(*diagonal);
      accn_->reciprocal_multiply(1.0, *diagonal, *frimpn_, 0.0);
    }
  }

  // apply Dirichlet BCs on accelerations
  apply_dirichlet_bc(timen_, nullptr, nullptr, accn_, false);

  // *********** time measurement ***********
  dtsolve_ = timer_->wallTime() - dtcpu;
  // *********** time measurement ***********

  return 0;
}

/*----------------------------------------------------------------------*/
/* Update step */
void Solid::TimIntAB2::update_step_state()
{
  // new displacements at t_{n+1} -> t_n
  //    D_{n} := D_{n+1}, D_{n-1} := D_{n}
  dis_->update_steps(*disn_);
  // new velocities at t_{n+1} -> t_n
  //    V_{n} := V_{n+1}, V_{n-1} := V_{n}
  vel_->update_steps(*veln_);
  // new accelerations at t_{n+1} -> t_n
  //    A_{n} := A_{n+1}, A_{n-1} := A_{n}
  acc_->update_steps(*accn_);

  // update contact and meshtying
  update_step_contact_meshtying();

  return;
}

/*----------------------------------------------------------------------*/
/* update after time step after output on element level*/
// update anything that needs to be updated at the element level
void Solid::TimIntAB2::update_step_element()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // other parameters that might be needed by the elements
  p.set("total time", timen_);
  p.set("delta time", (*dt_)[0]);
  // action for elements
  p.set("action", "calc_struct_update_istep");
  // go to elements
  discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);

  return;
}

/*----------------------------------------------------------------------*/
/* read restart forces */
void Solid::TimIntAB2::read_restart_force()
{
  FOUR_C_THROW("No restart ability for Adams-Bashforth 2nd order time integrator!");
  return;
}

/*----------------------------------------------------------------------*/
/* write internal and external forces for restart */
void Solid::TimIntAB2::write_restart_force(std::shared_ptr<Core::IO::DiscretizationWriter> output)
{
  return;
}

FOUR_C_NAMESPACE_CLOSE
