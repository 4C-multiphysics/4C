// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMINT_CREATE_HPP
#define FOUR_C_STRUCTURE_TIMINT_CREATE_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class Solver;
}

namespace Core::IO
{
  class DiscretizationWriter;
}

/*----------------------------------------------------------------------*/
namespace Solid
{
  // forward declarations
  class TimInt;
  class TimIntImpl;
  class TimIntExpl;

  /*====================================================================*/
  //! Create marching time integrator convenience routine
  //!
  std::shared_ptr<Solid::TimInt> tim_int_create(
      const Teuchos::ParameterList& timeparams,                //!< time parameters
      const Teuchos::ParameterList& ioflags,                   //!< input-output-flags
      const Teuchos::ParameterList& sdyn,                      //!< structural dynamic flags
      const Teuchos::ParameterList& xparams,                   //!< extra flags
      std::shared_ptr<Core::FE::Discretization>& actdis,       //!< discretisation
      std::shared_ptr<Core::LinAlg::Solver>& solver,           //!< the solver
      std::shared_ptr<Core::LinAlg::Solver>& contactsolver,    //!< the solver for contact/meshtying
      std::shared_ptr<Core::IO::DiscretizationWriter>& output  //!< output writer
  );

  /*====================================================================*/
  //! Create \b implicit marching time integrator convenience routine
  //!
  std::shared_ptr<Solid::TimIntImpl> tim_int_impl_create(
      const Teuchos::ParameterList& timeparams,                //!< time parameters
      const Teuchos::ParameterList& ioflags,                   //!< input-output-flags
      const Teuchos::ParameterList& sdyn,                      //!< structural dynamic flags
      const Teuchos::ParameterList& xparams,                   //!< extra flags
      std::shared_ptr<Core::FE::Discretization>& actdis,       //!< discretisation
      std::shared_ptr<Core::LinAlg::Solver>& solver,           //!< the solver
      std::shared_ptr<Core::LinAlg::Solver>& contactsolver,    //!< the contact solver
      std::shared_ptr<Core::IO::DiscretizationWriter>& output  //!< output writer
  );

  /*====================================================================*/
  //! Create \b explicit marching time integrator convenience routine
  //!
  std::shared_ptr<Solid::TimIntExpl> tim_int_expl_create(
      const Teuchos::ParameterList& timeparams,                //!< time parameters
      const Teuchos::ParameterList& ioflags,                   //!< input-output-flags
      const Teuchos::ParameterList& sdyn,                      //!< structural dynamic flags
      const Teuchos::ParameterList& xparams,                   //!< extra flags
      std::shared_ptr<Core::FE::Discretization>& actdis,       //!< discretisation
      std::shared_ptr<Core::LinAlg::Solver>& solver,           //!< the solver
      std::shared_ptr<Core::LinAlg::Solver>& contactsolver,    //!< the solver for contact/meshtying
      std::shared_ptr<Core::IO::DiscretizationWriter>& output  //!< output writer
  );

}  // namespace Solid

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
