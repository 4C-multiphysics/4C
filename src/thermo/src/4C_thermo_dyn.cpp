// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_dyn.hpp"

#include "4C_global_data.hpp"
#include "4C_thermo_adapter.hpp"
#include "4C_thermo_resulttest.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void thermo_dyn_drt()
{
  std::shared_ptr<Core::FE::Discretization> discretization =
      Global::Problem::instance()->get_dis("thermo");
  const Teuchos::ParameterList& parameters = Global::Problem::instance()->thermal_dynamic_params();

  std::shared_ptr<Thermo::BaseAlgorithm> algorithm =
      std::make_shared<Thermo::BaseAlgorithm>(parameters, discretization);

  // do restart if demanded from input file
  const int restart = Global::Problem::instance()->restart();
  const int write_initial_state =
      Global::Problem::instance()->io_params().get<bool>("WRITE_INITIAL_STATE");
  if (restart)
  {
    algorithm->thermo_field()->read_restart(restart);
  }
  else if (write_initial_state)
  {
    algorithm->thermo_field()->output();
  }

  // enter time loop to solve problem
  algorithm->thermo_field()->integrate();

  // perform testing if required
  Global::Problem::instance()->add_field_test(algorithm->thermo_field()->create_field_test());
  Global::Problem::instance()->test_all(discretization->get_comm());
}

FOUR_C_NAMESPACE_CLOSE
