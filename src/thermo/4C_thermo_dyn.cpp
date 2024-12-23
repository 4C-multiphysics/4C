// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_dyn.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_thermo_adapter.hpp"
#include "4C_thermo_resulttest.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | main control routine for (in)stationary heat conduction              |
 *----------------------------------------------------------------------*/
void thermo_dyn_drt()
{
  // access the discretization
  std::shared_ptr<Core::FE::Discretization> thermodis = nullptr;
  thermodis = Global::Problem::instance()->get_dis("thermo");

  // set degrees of freedom in the discretization
  if (not thermodis->filled()) thermodis->fill_complete();

  const Teuchos::ParameterList& tdyn = Global::Problem::instance()->thermal_dynamic_params();

  // create instance of thermo basis algorithm (no structure discretization)
  std::shared_ptr<Thermo::BaseAlgorithm> thermoonly =
      std::make_shared<Thermo::BaseAlgorithm>(tdyn, thermodis);

  // do restart if demanded from input file
  const int restart = Global::Problem::instance()->restart();
  if (restart)
  {
    thermoonly->thermo_field().read_restart(restart);
  }

  // enter time loop to solve problem
  (thermoonly->thermo_field()).integrate();

  // perform the result test if required
  Global::Problem::instance()->add_field_test(thermoonly->thermo_field().create_field_test());
  Global::Problem::instance()->test_all(thermodis->get_comm());

  // done
  return;
}


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
