#ifndef FOUR_C_LUBRICATION_ELE_ACTION_HPP
#define FOUR_C_LUBRICATION_ELE_ACTION_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace LUBRICATION
{
  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible lubrication actions
   *///                                                            wirtz 11/15
  /*--------------------------------------------------------------------------*/
  enum Action
  {
    // domain action
    set_general_lubrication_parameter,  // set general parameters for element evaluation
    set_time_parameter,                 // set time-integration parameters for element evaluation
    calc_mat_and_rhs,                   // calc_condif_systemmat_and_residual,
    calc_mean_pressures,                // calc_mean_pressures,
    calc_error,                         // calc_error
    calc_lubrication_coupltang          // calculate off-diagonal tangent matrix term
  };                                    // enum Action

  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible lubrication actions on a boundary
   *///                                                            wirtz 11/15
  /*--------------------------------------------------------------------------*/
  enum BoundaryAction
  {
    // new action
    bd_calc_weak_Dirichlet,  // weak_dirichlet,
    bd_calc_Neumann,         // n/a
  };                         // enum LUBRICATION::BoundaryAction
}  // namespace LUBRICATION

FOUR_C_NAMESPACE_CLOSE

#endif
