#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_list.hpp"
#include "4C_red_airways_airway_impl.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_red_airways_implicitintegration.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 |evaluate the element (public)                            ismail 01/10|
 *---------------------------------------------------------------------*/
int Discret::ELEMENTS::RedAirway::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  Discret::ELEMENTS::RedAirway::ActionType act = RedAirway::none;

  // get the action required
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_sys_matrix_rhs")
    act = RedAirway::calc_sys_matrix_rhs;
  else if (action == "get_initial_state")
    act = RedAirway::get_initial_state;
  else if (action == "set_bc")
    act = RedAirway::set_bc;
  else if (action == "calc_flow_rates")
    act = RedAirway::calc_flow_rates;
  else if (action == "get_coupled_values")
    act = RedAirway::get_coupled_values;
  else if (action == "calc_sys_matrix_rhs_iad")
    act = RedAirway::calc_sys_matrix_rhs_iad;
  else if (action == "get_junction_volume_mix")
    act = RedAirway::get_junction_volume_mix;
  else if (action == "solve_scatra")
    act = RedAirway::solve_scatra;
  else if (action == "solve_junction_scatra")
    act = RedAirway::solve_junction_scatra;
  else if (action == "calc_cfl")
    act = RedAirway::calc_cfl;
  else if (action == "eval_nodal_essential_values")
    act = RedAirway::eval_nodal_ess_vals;
  else if (action == "solve_blood_air_transport")
    act = RedAirway::solve_blood_air_transport;
  else if (action == "update_scatra")
    act = RedAirway::update_scatra;
  else if (action == "update_elem12_scatra")
    act = RedAirway::update_elem12_scatra;
  else if (action == "eval_PO2_from_concentration")
    act = RedAirway::eval_PO2_from_concentration;
  else if (action == "calc_elem_volumes")
    act = RedAirway::calc_elem_volumes;
  else
  {
    FOUR_C_THROW("Unknown type of action (%s) for reduced dimensional airway", action.c_str());
  }

  /*
  Here one must add the steps for evaluating an element
  */
  Teuchos::RCP<Core::Mat::Material> mat = material();

  switch (act)
  {
    case calc_sys_matrix_rhs:
    {
      return Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->evaluate(
          this, params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3, mat);
    }
    break;
    case calc_sys_matrix_rhs_iad:
    {
    }
    break;
    case get_initial_state:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->initial(
          this, params, discretization, lm, elevec1, elevec2, mat);
    }
    break;
    case set_bc:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->evaluate_terminal_bc(
          this, params, discretization, lm, elevec1, mat);
    }
    break;
    case calc_flow_rates:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->calc_flow_rates(
          this, params, discretization, lm, mat);
    }
    break;
    case get_coupled_values:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->get_coupled_values(
          this, params, discretization, lm, mat);
    }
    break;
    case get_junction_volume_mix:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->get_junction_volume_mix(
          this, params, discretization, elevec1, lm, mat);
    }
    break;
    case calc_cfl:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->calc_cfl(
          this, params, discretization, lm, mat);
    }
    break;
    case solve_blood_air_transport:
    {
      // do nothing
    }
    break;
    case eval_nodal_ess_vals:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->eval_nodal_essential_values(
          this, params, discretization, elevec1, elevec2, elevec3, lm, mat);
    }
    break;
    case update_scatra:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->update_scatra(
          this, params, discretization, lm, mat);
    }
    break;
    case update_elem12_scatra:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->update_elem12_scatra(
          this, params, discretization, lm, mat);
    }
    break;
    case calc_elem_volumes:
    {
      Discret::ELEMENTS::RedAirwayImplInterface::impl(this)->calc_elem_volume(
          this, params, discretization, lm, mat);
    }
    break;
    default:
      FOUR_C_THROW("Unknown type of action for reduced dimensional airways");
      break;
  }  // end of switch(act)

  return 0;
}  // end of Discret::ELEMENTS::RedAirway::Evaluate


/*----------------------------------------------------------------------*
 |  do nothing (public)                                     ismail 01/10|
 |                                                                      |
 |  The function is just a dummy.                                       |
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::RedAirway::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  return 0;
}


/*----------------------------------------------------------------------*
 |  do nothing (public)                                     ismail 01/10|
 |                                                                      |
 |  The function is just a dummy.                                       |
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::RedAirway::evaluate_dirichlet(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1)
{
  return 0;
}


/*----------------------------------------------------------------------*
 | get optimal gaussrule for discretisation type                        |
 |                                                                      |
 *----------------------------------------------------------------------*/
Core::FE::GaussRule1D Discret::ELEMENTS::RedAirway::get_optimal_gaussrule(
    const Core::FE::CellType& distype)
{
  Core::FE::GaussRule1D rule = Core::FE::GaussRule1D::undefined;
  switch (distype)
  {
    case Core::FE::CellType::line2:
      rule = Core::FE::GaussRule1D::line_2point;
      break;
    case Core::FE::CellType::line3:
      rule = Core::FE::GaussRule1D::line_3point;
      break;
    default:
      FOUR_C_THROW("unknown number of nodes for gaussrule initialization");
      break;
  }
  return rule;
}


/*----------------------------------------------------------------------*
 | Check, whether higher order derivatives for shape functions          |
 | (dxdx, dxdy, ...) are necessary|                                     |
 *----------------------------------------------------------------------*/
bool Discret::ELEMENTS::RedAirway::is_higher_order_element(const Core::FE::CellType distype) const
{
  bool hoel = true;
  switch (distype)
  {
    case Core::FE::CellType::line3:
      hoel = true;
      break;
    case Core::FE::CellType::line2:
      hoel = false;
      break;
    default:
      FOUR_C_THROW("distype unknown!");
      break;
  }
  return hoel;
}

FOUR_C_NAMESPACE_CLOSE
