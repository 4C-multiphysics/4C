// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timint_prestress.hpp"

#include "4C_constraint_springdashpot_manager.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

/*======================================================================*/
/* constructor */
Solid::TimIntPrestress::TimIntPrestress(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams, const Teuchos::RCP<Core::FE::Discretization>& actdis,
    const Teuchos::RCP<Core::LinAlg::Solver>& solver,
    const Teuchos::RCP<Core::LinAlg::Solver>& contactsolver,
    const Teuchos::RCP<Core::IO::DiscretizationWriter>& output)
    : TimIntStatics(
          timeparams, ioparams, sdynparams, xparams, actdis, solver, contactsolver, output)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
  return;
}

void Solid::TimIntPrestress::setup()
{
  Solid::TimIntStatics::setup();
  // Check for compatible prestressing algorithms
  const auto pre_stress = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
      Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");
  switch (pre_stress)
  {
    case Inpar::Solid::PreStress::mulf:
      break;
    default:
      FOUR_C_THROW(
          "Your prestressing algorithm is not implemented in the old structural time integration "
          "framework. Possibly you have to use the new structural time integration framework.");
  }
}

/*----------------------------------------------------------------------*/
/* update after time step after output on element level*/
// update anything that needs to be updated at the element level
void Solid::TimIntPrestress::update_step_element()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;

  const auto pre_stress = Teuchos::getIntegralValue<Inpar::Solid::PreStress>(
      Global::Problem::instance()->structural_dynamic_params(), "PRESTRESS");
  const double pstime =
      Global::Problem::instance()->structural_dynamic_params().get<double>("PRESTRESSTIME");
  // MULF, Material iterative prestressing
  if (pre_stress == Inpar::Solid::PreStress::mulf)
  {
    if ((*time_)[0] <= pstime + 1e-15)
    {
      if (!discret_->get_comm().MyPID())
        Core::IO::cout << "====== Entering MULF update" << Core::IO::endl;
      // action for elements
      p.set("action", "calc_struct_prestress_update");
      discret_->clear_state();
      discret_->set_state(0, "residual displacement", zeros_);
    }
    else
    {
      // action for elements
      p.set("action", "calc_struct_update_istep");
      discret_->clear_state();
    }
  }

  // params for MULF
  p.set("total time", (*time_)[0]);
  p.set("delta time", (*dt_)[0]);

  // go to elements
  discret_->set_state("displacement", (*dis_)(0));
  discret_->evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);


  if (pre_stress == Inpar::Solid::PreStress::mulf && (*time_)[0] <= pstime + 1e-15)
  {
    // prestressing for spring in spring dashpot - corresponds to storage of deformation gradient
    // in material law (mhv 12/2015) pass current displacement state to spring at end of MULF step
    if (springman_->have_spring_dashpot())
    {
      springman_->reset_prestress(*disn_);
    }
    // only for MULF prestressing mode:
    dis_->update_steps(*zeros_);
    vel_->update_steps(*zeros_);  // this simply copies zero vectors
    acc_->update_steps(*zeros_);  // this simply copies zero vectors
  }
}
/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
