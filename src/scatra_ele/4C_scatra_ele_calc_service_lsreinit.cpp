// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc_lsreinit.hpp"
#include "4C_scatra_ele_parameter_lsreinit.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
int Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::evaluate_action(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const ScaTra::Action& action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  const std::vector<int>& lm = la[0].lm_;

  // determine and evaluate action
  switch (action)
  {
    case ScaTra::Action::calc_mat_and_rhs_lsreinit_correction_step:
    {
      // extract local values from the global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> phizero =
          discretization.get_state("phizero");
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phizero == nullptr or phinp == nullptr)
        FOUR_C_THROW("Cannot get state vector 'phizero' and/ or 'phinp'!");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, lm);
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phizero, ephizero_, lm);

      //------------------------------------------------------
      // Step 1: precompute element penalty parameter
      //------------------------------------------------------

      // for correction a penalty parameter
      //                /          (phinp - phi_0)
      //               | H'(phi_0) ---------------- dOmega
      //              /                  time
      //  lambda = - ------------------------------------------
      //                /           2
      //               | (H'(phi_0))  dOmega
      //              /
      // is defined for each element

      // penalty parameter to be computed
      double penalty = 0.0;
      // calculate penalty parameter for element
      calc_ele_penalty_parameter(penalty);

      //------------------------------------------------------
      // Step 2: calculate matrix and rhs
      //------------------------------------------------------

      sysmat_correction(penalty, elemat1_epetra, elevec1_epetra);

      break;
    }
    case ScaTra::Action::calc_node_based_reinit_velocity:
    {
      // extract local values from the global vectors
      std::shared_ptr<const Core::LinAlg::Vector<double>> phizero =
          discretization.get_state("phizero");
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phizero == nullptr or phinp == nullptr)
        FOUR_C_THROW("Cannot get state vector 'phizero' and/ or 'phinp'!");
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, lm);
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phizero, ephizero_, lm);

      // get current direction
      const int dir = params.get<int>("direction");

      sysmat_nodal_vel(dir, elemat1_epetra, elevec1_epetra);

      break;
    }
    default:
    {
      my::evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
    }
  }  // switch(action)

  return 0;
}


/*----------------------------------------------------------------------*
 | setup element evaluation                                  fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
int Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::setup_calc(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization)
{
  // reset all managers to their default values (I feel better this way)
  diff_manager()->reset();
  var_manager()->reset();

  // clear all unused variables
  my::edispnp_.clear();
  my::weights_.clear();
  my::evelnp_.clear();
  my::eaccnp_.clear();
  my::eprenp_.clear();

  // call base class routine
  return my::setup_calc(ele, discretization);
}


/*----------------------------------------------------------------------*
 | calculate system matrix and rhs for correction step  rasthofer 12/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::sysmat_correction(
    const double penalty,                   ///< element penalty parameter
    Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs   ///< element rhs to calculate
)
{
  //----------------------------------------------------------------------
  // calculation of element volume for characteristic element length
  //----------------------------------------------------------------------
  // use one-point Gauss rule to do calculations at the element center
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints_tau(
      ScaTra::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = my::eval_shape_func_and_derivs_at_int_point(intpoints_tau, 0);

  //----------------------------------------------------------------------
  // calculation of characteristic element length
  //----------------------------------------------------------------------

  // get gradient of initial phi at element center
  Core::LinAlg::Matrix<nsd_, 1> gradphizero(Core::LinAlg::Initialization::zero);
  gradphizero.multiply(my::derxy_, ephizero_[0]);

  // get characteristic element length
  const double charelelength = calc_char_ele_length_reinit(vol, gradphizero);

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // initial phi at Gauss point
    double phizero = 0.0;
    phizero = my::funct_.dot(ephizero_[0]);
    // and corresponding gradient
    gradphizero.clear();
    gradphizero.multiply(my::derxy_, ephizero_[0]);
    double norm_gradphizero = gradphizero.norm2();

    // derivative of sign function at Gauss point
    double deriv_sign = 0.0;
    deriv_sign_function(deriv_sign, charelelength, phizero);

    // scalar at integration point at time step n+1
    const double phinp = my::funct_.dot(my::ephinp_[0]);
    std::dynamic_pointer_cast<
        Discret::Elements::ScaTraEleInternalVariableManagerLsReinit<nsd_, nen_>>(
        my::scatravarmanager_)
        ->set_phinp(0, phinp);
    std::dynamic_pointer_cast<
        Discret::Elements::ScaTraEleInternalVariableManagerLsReinit<nsd_, nen_>>(
        my::scatravarmanager_)
        ->set_hist(0, 0.0);

    //------------------------------------------------
    // element matrix
    //------------------------------------------------

    my::calc_mat_mass(emat, 0, fac, 1.0);

    //------------------------------------------------
    // element rhs
    //------------------------------------------------

    // predictor for phinp
    // caution: this function can be used here, since gen-alpha is excluded
    if (my::scatraparatimint_->is_gen_alpha())
      FOUR_C_THROW("Not supported by this implementation!");
    // note: this function computes phinp at integration point
    my::calc_rhs_lin_mass(erhs, 0, 0.0, -fac, 1.0, 1.0);  // sign has to be changed!!!!

    // penalty term
    calc_rhs_penalty(erhs, fac, penalty, deriv_sign, norm_gradphizero);

  }  // end: loop all Gauss points

  return;
}


/*-------------------------------------------------------------------------------*
 | calculation of element-wise denominator of penalty parameter  rasthofer 12/13 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::calc_ele_penalty_parameter(
    double& penalty)
{
  // safety check
  if (lsreinitparams_->sign_type() != Inpar::ScaTra::signtype_SussmanFatemi1999)
    FOUR_C_THROW("Penalty method only for smoothed sign function: SussmanFatemi1999!");

  // denominator
  double ele_dom = 0.0;
  // nominator
  double ele_nom = 0.0;

  //----------------------------------------------------------------------
  // calculation of element volume for characteristic element length
  //----------------------------------------------------------------------
  // use one-point Gauss rule to do calculations at the element center
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints_tau(
      ScaTra::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = my::eval_shape_func_and_derivs_at_int_point(intpoints_tau, 0);

  //----------------------------------------------------------------------
  // calculation of characteristic element length
  //----------------------------------------------------------------------

  // get gradient of initial phi at element center
  Core::LinAlg::Matrix<nsd_, 1> gradphizero(Core::LinAlg::Initialization::zero);
  gradphizero.multiply(my::derxy_, ephizero_[0]);

  // get characteristic element length
  const double charelelength = calc_char_ele_length_reinit(vol, gradphizero);

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // initial phi at Gauss point
    double phizero = 0.0;
    phizero = my::funct_.dot(ephizero_[0]);
    // and corresponding gradient
    gradphizero.clear();
    gradphizero.multiply(my::derxy_, ephizero_[0]);
    double norm_gradphizero = gradphizero.norm2();

    // derivative of sign function at Gauss point
    double deriv_sign = 0.0;
    deriv_sign_function(deriv_sign, charelelength, phizero);

    // current phi at Gauss point
    double phinp = 0.0;
    phinp = my::funct_.dot(my::ephinp_[0]);

    // get sign function
    double signphi = 0.0;
    // gradient of current scalar
    Core::LinAlg::Matrix<nsd_, 1> gradphi(Core::LinAlg::Initialization::zero);
    gradphi.multiply(my::derxy_, my::ephinp_[0]);
    // get norm
    const double gradphi_norm = gradphi.norm2();
    sign_function(signphi, charelelength, phizero, gradphizero, phinp, gradphi);

    // get velocity at element center
    Core::LinAlg::Matrix<nsd_, 1> convelint(Core::LinAlg::Initialization::zero);
    if (gradphi_norm > 1e-8) convelint.update(signphi / gradphi_norm, gradphi);
    // convective term
    //    double conv_phi = convelint.Dot(gradphi);

    // add Gauss point contribution to denominator
    // TODO: mit norm_gradphizero Sussman-Style
    ele_dom += (fac * deriv_sign * deriv_sign * norm_gradphizero);
    ele_nom -= (fac * deriv_sign * (phinp - phizero) / my::scatraparatimint_->time());
    //    ele_nom -= (fac * deriv_sign * (-conv_phi + signphi)); // gecheckt
  }  // end: loop all Gauss points

  // compute penalty parameter
  if (std::abs(ele_dom) > 1.0e-9) penalty = ele_nom / ele_dom;

  return;
}


/*------------------------------------------------------------------- *
 |  calculation of penalty term on rhs                rasthofer 12/13 |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::calc_rhs_penalty(
    Core::LinAlg::SerialDenseVector& erhs, const double fac, const double penalty,
    const double deriv_sign, const double norm_gradphizero)
{
  double vpenalty = fac * my::scatraparatimint_->dt() * penalty * deriv_sign * norm_gradphizero;

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_;

    erhs[fvi] += vpenalty * my::funct_(vi);
  }

  return;
}


/*-------------------------------------------------------------------------*
 | calculate system matrix and rhs for velocity projection rasthofer 12/13 |
 *-------------------------------------------------------------------------*/
template <Core::FE::CellType distype, unsigned prob_dim>
void Discret::Elements::ScaTraEleCalcLsReinit<distype, prob_dim>::sysmat_nodal_vel(
    const int dir,                          ///< current spatial direction
    Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs   ///< element rhs to calculate
)
{
  //----------------------------------------------------------------------
  // calculation of element volume for characteristic element length
  //----------------------------------------------------------------------
  // use one-point Gauss rule to do calculations at the element center
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints_center(
      ScaTra::DisTypeToStabGaussRule<distype>::rule);

  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = my::eval_shape_func_and_derivs_at_int_point(intpoints_center, 0);

  //----------------------------------------------------------------------
  // calculation of characteristic element length
  //----------------------------------------------------------------------

  // get gradient of initial phi at element center
  Core::LinAlg::Matrix<nsd_, 1> gradphizero(Core::LinAlg::Initialization::zero);
  gradphizero.multiply(my::derxy_, ephizero_[0]);

  // get characteristic element length
  const double charelelength = calc_char_ele_length_reinit(vol, gradphizero);

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // initial phi at Gauss point
    double phizero = 0.0;
    phizero = my::funct_.dot(ephizero_[0]);
    // and corresponding gradient
    gradphizero.clear();
    gradphizero.multiply(my::derxy_, ephizero_[0]);

    // current phi at Gauss point
    double phinp = 0.0;
    phinp = my::funct_.dot(my::ephinp_[0]);
    // gradient of current scalar
    Core::LinAlg::Matrix<nsd_, 1> gradphi(Core::LinAlg::Initialization::zero);
    gradphi.multiply(my::derxy_, my::ephinp_[0]);
    // get norm
    const double gradphi_norm = gradphi.norm2();

    // TODO: remove
    //    if (std::abs(my::ephinp_[0](0,0)-my::ephinp_[0](1,0))>1.0e-10 or
    //        std::abs(my::ephinp_[0](2,0)-my::ephinp_[0](3,0))>1.0e-10 or
    //        std::abs(my::ephinp_[0](4,0)-my::ephinp_[0](5,0))>1.0e-10 or
    //        std::abs(my::ephinp_[0](6,0)-my::ephinp_[0](7,0))>1.0e-10)
    //    {
    //        std::cout << my::ephinp_[0] << std::endl;
    //        FOUR_C_THROW("END");
    //    }

    // get velocity at element center
    Core::LinAlg::Matrix<nsd_, 1> convelint(Core::LinAlg::Initialization::zero);
    if (lsreinitparams_->reinit_type() == Inpar::ScaTra::reinitaction_sussman)
    {
      // get sign function
      double signphi = 0.0;
      sign_function(signphi, charelelength, phizero, gradphizero, phinp, gradphi);

      if (gradphi_norm > 1e-8) convelint.update(signphi / gradphi_norm, gradphi);
    }

    //------------------------------------------------
    // element matrix
    //------------------------------------------------
    my::calc_mat_mass(emat, 0, fac, 1.0);

    //------------------------------------------------
    // add dissipation for smooth fields
    //------------------------------------------------
    // should not be used together with lumping
    // prevented by FOUR_C_THROW in lsreinit parameters
    if (lsreinitparams_->project_diff() > 0.0)
    {
      const double diff = (lsreinitparams_->project_diff()) * charelelength * charelelength;
      my::diffmanager_->set_isotropic_diff(diff, 0);
      my::calc_mat_diff(emat, 0, fac);
    }

    //------------------------------------------------
    // element rhs
    //------------------------------------------------
    // distinguish reinitialization
    switch (lsreinitparams_->reinit_type())
    {
      case Inpar::ScaTra::reinitaction_sussman:
      {
        my::calc_rhs_hist_and_source(erhs, 0, fac, convelint(dir, 0));
        break;
      }
      case Inpar::ScaTra::reinitaction_ellipticeq:
      {
        my::calc_rhs_hist_and_source(erhs, 0, fac, gradphi(dir, 0));
        break;
      }
      default:
        break;
    }
  }  // loop Gauss points

  // do lumping: row sum
  if (lsreinitparams_->lumping())
  {
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * my::numdofpernode_;

      double sum = 0.0;
      // loop all columns
      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const int fui = ui * my::numdofpernode_;
        sum += emat(fvi, fui);
        // reset
        emat(fvi, fui) = 0.0;
      }

      emat(fvi, fvi) = sum;
    }
  }

  return;
}


// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad4, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad8,2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::hex8, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::hex20,3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::tet10, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::wedge6,3>;
template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::pyramid5, 3>;
// template class Discret::Elements::ScaTraEleCalcLsReinit<Core::FE::CellType::nurbs27,3>;

FOUR_C_NAMESPACE_CLOSE
