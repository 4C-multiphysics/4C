#ifndef FOUR_C_ART_NET_ARTERY_ELE_ACTION_HPP
#define FOUR_C_ART_NET_ARTERY_ELE_ACTION_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Arteries
{
  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible artery actions
   *///                                                   kremheller 03/18
  /*--------------------------------------------------------------------------*/
  enum Action
  {
    none,
    calc_sys_matrix_rhs,
    calc_flow_pressurebased,
    get_initial_artery_state,
    solve_riemann_problem,
    set_term_bc,
    set_scatra_term_bc,
    set_scatra_bc,
    calc_postpro_vals,
    calc_scatra_sys_matrix_rhs,
    calc_scatra_from_scatra_fb,
    evaluate_wf_wb,
    evaluate_scatra_analytically
  };

}  // namespace Arteries

FOUR_C_NAMESPACE_CLOSE

#endif
