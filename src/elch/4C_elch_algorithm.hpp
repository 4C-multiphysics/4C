// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ELCH_ALGORITHM_HPP
#define FOUR_C_ELCH_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_scatra_algorithm.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ElCh
{
  /// ELCH algorithm base
  /*!

    Base class of ELCH algorithms. Derives from ScaTraAlgorithm.

   */
  class Algorithm : public ScaTra::ScaTraAlgorithm
  {
   public:
    /// constructor
    explicit Algorithm(Global::Problem& problem,    ///< global problem
        MPI_Comm comm,                              ///< communicator
        const Teuchos::ParameterList& elchcontrol,  ///< elch parameter list
        const Teuchos::ParameterList& scatradyn,    ///< scatra parameter list
        const Teuchos::ParameterList& fdyn,         ///< fluid parameter list
        const Teuchos::ParameterList& solverparams  ///< solver parameter list
    );


   protected:
    /// provide information about initial field
    void prepare_time_loop() override;

    /// print scatra solver type to screen
    void print_scatra_solver() override;

    /// convergence check for natural convection solver
    bool convergence_check(int natconvitnum, int natconvitmax, double natconvittol) override;
  };
}  // namespace ElCh

FOUR_C_NAMESPACE_CLOSE

#endif
