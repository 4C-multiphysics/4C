/*----------------------------------------------------------------------*/
/*! \file

\brief Provides enum of actions for electromagnetics elements

\level 2

*---------------------------------------------------------------------------*/

#ifndef FOUR_C_ELEMAG_ELE_ACTION_HPP
#define FOUR_C_ELEMAG_ELE_ACTION_HPP

#include "baci_config.hpp"

BACI_NAMESPACE_OPEN

namespace ELEMAG
{
  /*!
    \brief Enum that provides all possible elemag actions
  */
  enum Action
  {
    /// no action
    none,
    /// Compute system matrix and rhs
    calc_systemmat_and_residual,
    /// Compute Absorbing Boundary Conditions
    calc_abc,
    /// Project field (initialization)
    project_field,
    /// Project electric field from a scatra field (initialization)
    project_electric_from_scatra_field,
    /// Update secondary solution (local solver)
    update_secondary_solution,
    /// Interpolate the discontinous solution to mesh nodes (output)
    interpolate_hdg_to_node,
    /// Update secondary solution and compute RHS
    update_secondary_solution_and_calc_residual,
    /// Initialize elements
    ele_init,
    /// Fill the restart vectors
    fill_restart_vecs,
    /// Initilize elements from a given state
    ele_init_from_restart,
    /// Project Dirichlet boundary conditions
    project_dirich_field,
    /// Integrate boundary integrals
    bd_integrate,
    /// Obtain coordinates of Gauss points
    get_gauss_points,
    /// Project local field (testing purposes)
    project_field_test,
    /// Project global field (testing purposes)
    project_field_test_trace,
    /// Compute error wrt analitical solution
    compute_error
  };  // enum Action

}  // namespace ELEMAG


BACI_NAMESPACE_CLOSE

#endif  // ELEMAG_ELE_ACTION_H
