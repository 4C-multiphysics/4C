/*----------------------------------------------------------------------*/
/*! \file
\brief Explicit time integration for structural dynamics

\level 1

*/

/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_structure_timint_expl.hpp"

#include "4C_cardiovascular0d_manager.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_constraint_springdashpot_manager.hpp"
#include "4C_contact_meshtying_contact_bridge.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_mortar_strategy_base.hpp"
#include "4C_structure_aux.hpp"
#include "4C_structure_timint.hpp"

#include <sstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* constructor */
Solid::TimIntExpl::TimIntExpl(const Teuchos::ParameterList& timeparams,  //! time parameters
    const Teuchos::ParameterList& ioparams,                              //!< ioflags
    const Teuchos::ParameterList& sdynparams,                            //!< input parameters
    const Teuchos::ParameterList& xparams,                               //!< extra flags
    Teuchos::RCP<Core::FE::Discretization> actdis,                       //!< current discretisation
    Teuchos::RCP<Core::LinAlg::Solver> solver,                           //!< the solver
    Teuchos::RCP<Core::LinAlg::Solver> contactsolver,    //!< the solver for contact meshtying
    Teuchos::RCP<Core::IO::DiscretizationWriter> output  //!< the output
    )
    : TimInt(timeparams, ioparams, sdynparams, xparams, actdis, solver, contactsolver, output)
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
void Solid::TimIntExpl::init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    Teuchos::RCP<Core::FE::Discretization> actdis, Teuchos::RCP<Core::LinAlg::Solver> solver)
{
  // call init() in base class
  Solid::TimInt::init(timeparams, sdynparams, xparams, actdis, solver);

  // get away
  return;
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntExpl::setup()
{
  // call setup() in base class
  Solid::TimInt::setup();

  // explicit time integrators cannot handle constraints
  if (conman_->have_constraint())
    FOUR_C_THROW("Currently, constraints cannot be done with explicit time integration.");

  // explicit time integrators can only handle penalty contact / meshtying
  if (have_contact_meshtying())
  {
    Inpar::CONTACT::SolvingStrategy soltype =
        Core::UTILS::integral_value<Inpar::CONTACT::SolvingStrategy>(
            cmtbridge_->get_strategy().params(), "STRATEGY");
    if (soltype != Inpar::CONTACT::solution_penalty &&
        (soltype != Inpar::CONTACT::solution_multiscale))
      FOUR_C_THROW(
          "Currently, only penalty or multi-scale contact / meshtying can be done with explicit "
          "time integration schemes.");
  }

  // cannot handle rotated DOFs
  if (locsysman_ != Teuchos::null)
    FOUR_C_THROW("Explicit time integration schemes cannot handle local co-ordinate systems");

  // explicit time integrators cannot handle nonlinear inertia forces
  if (have_nonlinear_mass())
    FOUR_C_THROW(
        "Explicit time integration schemes cannot handle nonlinear inertia forces (flag: MASSLIN)");

  return;
}

/*----------------------------------------------------------------------*/
/* evaluate external forces at t_{n+1} */
void Solid::TimIntExpl::apply_force_external(const double time,  //!< evaluation time
    const Teuchos::RCP<Epetra_Vector> dis,                       //!< displacement state
    const Teuchos::RCP<Epetra_Vector> vel,                       //!< velocity state
    Teuchos::RCP<Epetra_Vector>& fext                            //!< external force
)
{
  Teuchos::ParameterList p;
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state(0, "displacement", dis);
  discret_->set_state(0, "displacement new", dis);

  if (damping_ == Inpar::Solid::damp_material) discret_->set_state(0, "velocity", vel);
  // get load vector
  discret_->evaluate_neumann(p, *fext);

  // go away
  return;
}

/*----------------------------------------------------------------------*/
/* print step summary */
void Solid::TimIntExpl::print_step()
{
  // print out
  if ((myrank_ == 0) and printscreen_ and (step_old() % printscreen_ == 0))
  {
    print_step_text(stdout);
  }
}

/*----------------------------------------------------------------------*/
/* print step summary */
void Solid::TimIntExpl::print_step_text(FILE* ofile)
{
  fprintf(ofile,
      "Finalised: step %6d"
      " | nstep %6d"
      " | time %-14.8E"
      " | dt %-14.8E"
      " | numiter %3d"
      " | wct %-14.8E\n",
      step_, stepmax_, (*time_)[0], (*dt_)[0], 0, timer_->totalElapsedTime(true));
  fprintf(ofile,
      "                      "
      " ( ts %-14.8E"
      " | te %-14.8E"
      " | tc %-14.8E)\n",
      dtsolve_, dtele_, dtcmt_);
  // print a beautiful line made exactly of 80 dashes
  fprintf(ofile,
      "--------------------------------------------------------------"
      "------------------\n");
  // do it, print now!
  fflush(ofile);

  // fall asleep
  return;
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
