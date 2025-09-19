// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele_calc.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_gder2.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_porofluid_pressure_based_ele_evaluator.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_porofluid_pressure_based_ele_phasemanager.hpp"
#include "4C_porofluid_pressure_based_ele_variablemanager.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | protected constructor for singletons                      vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::PoroFluidMultiPhaseEleCalc(
    const int numdofpernode, const std::string& disname)
    : ele_(nullptr),
      totalnumdofpernode_(numdofpernode),
      numfluidphases_(0),
      para_(Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(
          disname)),  // standard parameter list
      xsi_(),
      xyze0_(),
      funct_(),
      deriv_(),
      deriv2_(),
      derxy_(),
      derxy2_(),
      xjm_(),
      xij_(),
      det_(0.0),
      j_(0.0),
      phasemanager_(nullptr)
{
  return;
}

/*----------------------------------------------------------------------*
 | singleton access method                                   vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>*
Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::instance(
    const int numdofpernode, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::pair<std::string, int>>(
      [](const int numdofpernode, const std::string& disname)
      {
        return std::unique_ptr<PoroFluidMultiPhaseEleCalc<distype>>(
            new PoroFluidMultiPhaseEleCalc<distype>(numdofpernode, disname));
      });

  return singleton_map[std::make_pair(disname, numdofpernode)].instance(
      Core::Utils::SingletonAction::create, numdofpernode, disname);
}


/*----------------------------------------------------------------------*
 | evaluate  routine                                         vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::evaluate(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec)
{
  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<PoroPressureBased::Action>(params, "action");

  // setup
  if (setup_calc(ele, discretization, action) == -1) return -1;

  // extract element based or nodal values
  extract_element_and_node_values(ele, params, discretization, la);

  // evaluate action
  evaluate_action(ele, params, discretization, action, la, elemat, elevec);

  return 0;
}

/*----------------------------------------------------------------------*
 | evaluate action                                           vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::evaluate_action(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const PoroPressureBased::Action& action,
    Core::Elements::LocationArray& la, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec)
{
  // determine and evaluate action
  switch (action)
  {
    case PoroPressureBased::calc_mat_and_rhs:
    case PoroPressureBased::calc_initial_time_deriv:
    case PoroPressureBased::recon_flux_at_nodes:
    case PoroPressureBased::calc_domain_integrals:
    {
      // loop over gauss points and evaluate terms (standard call)
      gauss_point_loop(ele, elemat, elevec, discretization, la);
      break;
    }
    case PoroPressureBased::calc_fluid_struct_coupl_mat:
    {
      // loop over gauss points and evaluate off-diagonal terms
      gauss_point_loop_od_struct(ele, elemat, elevec, discretization, la);
      break;
    }
    case PoroPressureBased::calc_fluid_scatra_coupl_mat:
    {
      // loop over gauss points and evaluate off-diagonal terms
      gauss_point_loop_od_scatra(ele, elemat, elevec, discretization, la);
      break;
    }
    case PoroPressureBased::calc_pres_and_sat:
    {
      // loop over nodes and evaluate terms
      node_loop(ele, elemat, elevec, discretization, la,
          false  // no Jacobian at the node needed
      );
      break;
    }
    case PoroPressureBased::calc_porosity:
    case PoroPressureBased::calc_solidpressure:
    case PoroPressureBased::calc_determinant_of_deformationgradient:
    case PoroPressureBased::calc_volfrac_blood_lung:
    {
      // loop over nodes and evaluate terms
      node_loop(ele, elemat, elevec, discretization, la,
          true  // Jacobian at the node needed since porosity and (pressure in case of volume
                // fractions )depends on J
      );
      break;
    }
    case PoroPressureBased::calc_valid_dofs:
    {
      // evaluate the element
      evaluate_only_element(ele, elemat, elevec, discretization, la);
      break;
    }
    case PoroPressureBased::calc_phase_velocities:
    {
      // loop over gauss points and average
      gauss_point_loop_average(ele, elemat, elevec, discretization, la);
      break;
    }
    default:
    {
      FOUR_C_THROW("Not acting on this action. Forgot implementation?");
      break;
    }
  }  // switch(action)

  return 0;
}


/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                 vuong 08/16|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::gauss_point_loop(
    Core::Elements::Element* ele, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // prepare gauss point evaluation
  prepare_gauss_point_loop(ele);

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_> intpoints(
      PoroPressureBased::ElementUtils::DisTypeToOptGaussRule<distype>::rule);

  // start loop over gauss points
  gauss_point_loop(intpoints, ele, elemat, elevec, discretization, la);

  return;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::gauss_point_loop_average(
    Core::Elements::Element* ele, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  prepare_gauss_point_loop(ele);

  const Core::FE::IntPointsAndWeights<nsd_> intpoints(
      PoroPressureBased::ElementUtils::DisTypeToOptGaussRule<distype>::rule);

  gauss_point_loop(intpoints, ele, elemat, elevec, discretization, la);

  const auto number_gauss_points = 1.0 / (double)intpoints.ip().nquad;

  (*elemat[0]).scale(number_gauss_points);
  (*elemat[1]).scale(number_gauss_points);
  (*elevec[0]).scale(number_gauss_points);
  (*elevec[1]).scale(number_gauss_points);
  (*elevec[2]).scale(number_gauss_points);
}
/*----------------------------------------------------------------------*
| calculate off-diagonal fluid-struct-coupling matrix  kremheller 03/17 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::gauss_point_loop_od_struct(
    Core::Elements::Element* ele, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // prepare gauss point evaluation
  prepare_gauss_point_loop(ele);

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_> intpoints(
      PoroPressureBased::ElementUtils::DisTypeToOptGaussRule<distype>::rule);

  // start loop over gauss points
  gauss_point_loop_od_struct(intpoints, ele, elemat, elevec, discretization, la);

  return;
}

/*----------------------------------------------------------------------*
| calculate off-diagonal fluid-scatra-coupling matrix  kremheller 06/17 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::gauss_point_loop_od_scatra(
    Core::Elements::Element* ele, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // prepare gauss point evaluation
  prepare_gauss_point_loop(ele);

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_> intpoints(
      PoroPressureBased::ElementUtils::DisTypeToOptGaussRule<distype>::rule);

  // start loop over gauss points
  gauss_point_loop_od_scatra(intpoints, ele, elemat, elevec, discretization, la);

  return;
}

/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                 vuong 08/16|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::prepare_gauss_point_loop(
    Core::Elements::Element* ele)
{
  return;
}

/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                 vuong 08/16|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::gauss_point_loop(
    const Core::FE::IntPointsAndWeights<nsd_>& intpoints, Core::Elements::Element* ele,
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // start the loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    variablemanager_->evaluate_gp_variables(funct_, derxy_);

    // set the current gauss point state in the manager
    phasemanager_->evaluate_gp_state(j_, *variablemanager_);

    // stabilization parameter and integration factors
    // const double taufac     = tau[k]*fac;
    const double timefacfac = para_->time_fac() * fac;
    // const double timetaufac = para_->TimeFac()*taufac;

    double rhsfac = para_->time_fac_rhs() * fac;

    //----------------------------------------------------------------
    // 1) element matrix
    //----------------------------------------------------------------
    evaluator_->evaluate_matrix(elemat, funct_, derxy_, totalnumdofpernode_, *phasemanager_,
        *variablemanager_, timefacfac, fac);

    //----------------------------------------------------------------
    // 2) element right hand side
    //----------------------------------------------------------------

    evaluator_->evaluate_vector(elevec, funct_, derxy_, xyze_, totalnumdofpernode_, *phasemanager_,
        *variablemanager_, rhsfac, fac);

    // clear current gauss point data for safety
    phasemanager_->clear_gp_state();
  }

  return;
}

/*----------------------------------------------------------------------*
| calculate off-diagonal fluid-struct-coupling matrix  kremheller 03/17 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::gauss_point_loop_od_struct(
    const Core::FE::IntPointsAndWeights<nsd_>& intpoints, Core::Elements::Element* ele,
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // start the loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    variablemanager_->evaluate_gp_variables(funct_, derxy_);

    // set the current gauss point state in the manager
    phasemanager_->evaluate_gp_state(j_, *variablemanager_);

    // stabilization parameter and integration factors
    // const double taufac     = tau[k]*fac;
    double rhsfac = para_->time_fac_rhs() * fac;
    // const double timetaufac = para_->TimeFac()*taufac;

    //----------------------------------------------------------------
    // 1) element off-diagonal matrix
    //----------------------------------------------------------------
    evaluator_->evaluate_matrix_od_struct(elemat, funct_, deriv_, derxy_, xjm_, totalnumdofpernode_,
        *phasemanager_, *variablemanager_, rhsfac, fac, det_);

    // clear current gauss point data for safety
    phasemanager_->clear_gp_state();
  }

  return;
}

/*----------------------------------------------------------------------*
| calculate off-diagonal fluid-scatra-coupling matrix  kremheller 03/17 |
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::gauss_point_loop_od_scatra(
    const Core::FE::IntPointsAndWeights<nsd_>& intpoints, Core::Elements::Element* ele,
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // start the loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    variablemanager_->evaluate_gp_variables(funct_, derxy_);

    // set the current gauss point state in the manager
    phasemanager_->evaluate_gp_state(j_, *variablemanager_);

    // stabilization parameter and integration factors
    // const double taufac     = tau[k]*fac;
    double rhsfac = para_->time_fac_rhs() * fac;

    // const double timetaufac = para_->TimeFac()*taufac;

    //----------------------------------------------------------------
    // 1) element off-diagonal matrix
    //----------------------------------------------------------------
    evaluator_->evaluate_matrix_od_scatra(elemat, funct_, derxy_, totalnumdofpernode_,
        *phasemanager_, *variablemanager_, rhsfac, fac);

    // clear current gauss point data for safety
    phasemanager_->clear_gp_state();
  }

  return;
}

/*-----------------------------------------------------------------------------*
 | loop over nodes for evaluation                                  vuong 08/16 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::node_loop(Core::Elements::Element* ele,
    std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, const bool jacobian_needed)
{
  // Note: this is an evaluation at every node (e.g. for post processing)
  //       Hence, the shape function matrix funct and the derivative matrix derxy
  //       are set in such a way that the standard routines can be used

  // dummy derivative matrix as zero
  derxy_.clear();

  for (int inode = 0; inode < nen_; ++inode)
  {
    // dummy shape function matrix (set to zero)
    funct_.clear();
    // set value at current node to 1
    funct_(inode) = 1.0;

    if (jacobian_needed)
    {
      compute_jacobian_at_node(inode);
    }
    else
      j_ = 1.0;

    // evaluate needed variables
    variablemanager_->evaluate_gp_variables(funct_, derxy_);

    // set the current node state as GP state in the manager
    phasemanager_->evaluate_gp_state(j_, *variablemanager_);

    //----------------------------------------------------------------
    // 1) element matrix
    //----------------------------------------------------------------
    evaluator_->evaluate_matrix(
        elemat, funct_, derxy_, totalnumdofpernode_, *phasemanager_, *variablemanager_, 1.0, 1.0);

    //----------------------------------------------------------------
    // 2) element vector
    //----------------------------------------------------------------
    evaluator_->evaluate_vector(elevec, funct_, derxy_, xyze_, totalnumdofpernode_, *phasemanager_,
        *variablemanager_, 1.0, 1.0);

    // clear current gauss point data for safety
    phasemanager_->clear_gp_state();
  }
}

/*-----------------------------------------------------------------------------*
 | loop over nodes for evaluation                                  vuong 08/16 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::evaluate_only_element(
    Core::Elements::Element* ele, std::vector<Core::LinAlg::SerialDenseMatrix*>& elemat,
    std::vector<Core::LinAlg::SerialDenseVector*>& elevec, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // Note: this is an evaluation of the element where no Gauss point data and function values (from
  //       variable and phase-managers are required

  // dummy derivative matrix as zero
  derxy_.clear();
  funct_.clear();

  //----------------------------------------------------------------
  // 1) element matrix
  //----------------------------------------------------------------
  evaluator_->evaluate_matrix(
      elemat, funct_, derxy_, totalnumdofpernode_, *phasemanager_, *variablemanager_, 1.0, 1.0);

  //----------------------------------------------------------------
  // 2) element vector
  //----------------------------------------------------------------
  evaluator_->evaluate_vector(elevec, funct_, derxy_, xyze_, totalnumdofpernode_, *phasemanager_,
      *variablemanager_, 1.0, 1.0);

  return;
}

/*----------------------------------------------------------------------*
 | setup element evaluation                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::setup_calc(Core::Elements::Element* ele,
    Core::FE::Discretization& discretization, const PoroPressureBased::Action& action)
{
  // get element coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze0_);

  // set current coordinates to initial coordinates
  // the displacements will be added later in extract_element_and_node_values() for the moving mesh
  // case
  xyze_ = xyze0_;

  // set element
  ele_ = ele;

  std::shared_ptr<Core::Mat::Material> mat = ele->material();
  if (mat->material_type() != Core::Materials::m_fluidporo_multiphase and
      mat->material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
    FOUR_C_THROW(
        "PoroFluidMultiPhase element got unsupported material type {}", mat->material_type());

  std::shared_ptr<Mat::FluidPoroMultiPhase> actmat =
      std::static_pointer_cast<Mat::FluidPoroMultiPhase>(mat);
  if (actmat == nullptr) FOUR_C_THROW("cast failed");
  numfluidphases_ = actmat->num_fluid_phases();

  // Note:
  // here the phase manager, the variable manager and the evaluator classes are
  // rebuild here, allocating new memory. This is actually not necessary.
  // If this proves to be a performance issue, then the Create*** methods
  // could be adapted to have static members, which are only built once.

  // build the phase manager
  phasemanager_ = Discret::Elements::PoroFluidManager::PhaseManagerInterface::create_phase_manager(
      *para_, nsd_, ele->material()->material_type(), action, totalnumdofpernode_, numfluidphases_,
      actmat->num_vol_frac());
  // setup the manager
  phasemanager_->setup(ele);

  // rebuild the phase manager
  variablemanager_ = Discret::Elements::PoroFluidManager::VariableManagerInterface<nsd_,
      nen_>::create_variable_manager(*para_, action, actmat, totalnumdofpernode_, numfluidphases_,
      actmat->num_vol_frac());

  // build the evaluator
  evaluator_ =
      Discret::Elements::PoroFluidEvaluator::EvaluatorInterface<nsd_, nen_>::create_evaluator(
          *para_, action, totalnumdofpernode_, numfluidphases_, *phasemanager_);

  return 0;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::extract_element_and_node_values(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  variablemanager_->extract_element_and_node_values(*ele, discretization, la, xyze_);
  return;
}


/*------------------------------------------------------------------------*
 | evaluate shape functions and derivatives at int. point     vuong 08/16 |
 *------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double
Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::eval_shape_func_and_derivs_at_int_point(
    const Core::FE::IntPointsAndWeights<nsd_>& intpoints,  ///< integration points
    const int iquad                                        ///< id of current Gauss point
)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.ip().qxg)[iquad];
  for (int idim = 0; idim < nsd_; idim++) xsi_(idim) = gpcoord[idim];

  det_ = eval_shape_func_and_derivs_in_parameter_space();

  if (det_ < 1E-16)
    FOUR_C_THROW(
        "GLOBAL ELEMENT NO. {} \nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", ele_->id(), det_);

  // compute global spatial derivatives
  derxy_.multiply(xij_, deriv_);

  // compute second global derivatives (if needed)
  if (use2ndderiv_)
  {
    // get global second derivatives
    Core::FE::gder2<distype, nen_>(xjm_, derxy_, deriv2_, xyze_, derxy2_);
  }
  else
    derxy2_.clear();

  //------------------------get determinant of Jacobian dX / ds
  // transposed jacobian "dX/ds"
  Core::LinAlg::Matrix<nsd_, nsd_> xjm0;
  xjm0.multiply_nt(deriv_, xyze0_);

  // inverse of transposed jacobian "ds/dX"
  const double det0 = xjm0.determinant();

  // determinant of deformationgradient det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds)
  // )^-1
  j_ = det_ / det0;

  // set integration factor: fac = Gauss weight * det(J)
  const double fac = intpoints.ip().qwgt[iquad] * det_;

  // return integration factor for current GP: fac = Gauss weight * det(J)
  return fac;

}  // PoroFluidMultiPhaseEleCalc::eval_shape_func_and_derivs_at_int_point

/*--------------------------------------------------------------------------*
 | evaluate shape functions and derivatives in parameter space   vuong 08/16 |
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Discret::Elements::PoroFluidMultiPhaseEleCalc<
    distype>::eval_shape_func_and_derivs_in_parameter_space()
{
  double det = 0.0;

  // shape functions and their first derivatives
  Core::FE::shape_function<distype>(xsi_, funct_);
  Core::FE::shape_function_deriv1<distype>(xsi_, deriv_);
  if (use2ndderiv_)
  {
    // get the second derivatives of standard element at current GP
    Core::FE::shape_function_deriv2<distype>(xsi_, deriv2_);
  }

  // compute Jacobian matrix and determinant
  // actually compute its transpose....
  /*
    +-            -+ T      +-            -+
    | dx   dx   dx |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dr   dr   dr |
    |              |        |              |
    | dy   dy   dy |        | dx   dy   dz |
    | --   --   -- |   =    | --   --   -- |
    | dr   ds   dt |        | ds   ds   ds |
    |              |        |              |
    | dz   dz   dz |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dt   dt   dt |
    +-            -+        +-            -+
  */

  xjm_.multiply_nt(deriv_, xyze_);
  det = xij_.invert(xjm_);

  return det;
}

/*--------------------------------------------------------------------------*
 | Compute Jacobian at node 'inode'                        kremheller 04/17 |
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::PoroFluidMultiPhaseEleCalc<distype>::compute_jacobian_at_node(
    const int inode)
{
  // get parameter space coordinates of current node
  Core::LinAlg::Matrix<3, 1> myXi = Core::FE::get_node_coordinates(inode, distype);
  for (int idim = 0; idim < nsd_; idim++) xsi_(idim) = myXi(idim);

  det_ = eval_shape_func_and_derivs_in_parameter_space();

  if (det_ < 1E-16)
    FOUR_C_THROW(
        "GLOBAL ELEMENT NO. {} \nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", ele_->id(), det_);

  // compute global spatial derivatives
  derxy_.multiply(xij_, deriv_);

  //------------------------get determinant of Jacobian dX / ds
  // transposed jacobian "dX/ds"
  Core::LinAlg::Matrix<nsd_, nsd_> xjm0;
  xjm0.multiply_nt(deriv_, xyze0_);

  // inverse of transposed jacobian "ds/dX"
  const double det0 = xjm0.determinant();

  // determinant of deformationgradient det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds)
  // )^-1
  j_ = det_ / det0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes

// 1D elements
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::line2>;
// template class
// Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::line2,2>; template
// class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::line2,3>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::line3>;
// template class
// Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::line3,2>; template
// class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::line3,3>;

// 2D elements
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::tri3>;
// template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::tri6>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::quad4>;
// template class
// Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::quad8>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::quad9>;
// template class
// Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::quad9,3>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::hex8>;
// template class
// Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::hex20>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::hex27>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::tet4>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::tet10>;
// template class
// Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::wedge6>;
template class Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::pyramid5>;
// template class
// Discret::Elements::PoroFluidMultiPhaseEleCalc<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
