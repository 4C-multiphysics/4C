// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_elch.hpp"

#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_utils_elch.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcElch<distype, probdim>::ScaTraEleCalcElch(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype, probdim>::ScaTraEleCalc(
          numdofpernode, numscal, disname),
      elchparams_(Discret::Elements::ScaTraEleParameterElch::instance(
          disname)),  // parameter class for electrochemistry problems
      utils_(
          Discret::Elements::ScaTraEleUtilsElch<distype>::instance(numdofpernode, numscal, disname))
{
  // replace standard scatra diffusion manager by elch diffusion manager
  my::diffmanager_ = std::make_shared<ScaTraEleDiffManagerElch>(my::numscal_);

  // safety check
  if (not my::scatraparatimint_->is_incremental())
    FOUR_C_THROW(
        "Since the ion-transport equations are non-linear, it can be solved only incrementally!!");
}


/*----------------------------------------------------------------------*
 | Action type: Evaluate                                     ehrl 01/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcElch<distype, probdim>::evaluate(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // call base class routine
  my::evaluate(ele, params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra,
      elevec2_epetra, elevec3_epetra);

  // for certain ELCH problem formulations we have to provide
  // additional flux terms / currents across Dirichlet boundaries
  if (elchparams_->boundary_flux_coupling())
    correction_for_flux_across_dc(discretization, la[0].lm_, elemat1_epetra, elevec1_epetra);

  return 0;
}


/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                 ehrl  08/08|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::sysmat(
    Core::Elements::Element* ele,               ///< the element whose matrix is calculated
    Core::LinAlg::SerialDenseMatrix& emat,      ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs,      ///< element rhs to calculate
    Core::LinAlg::SerialDenseVector& subgrdiff  ///< subgrid-diff.-scaling vector
)
{
  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  const double vol = my::eval_shape_func_and_derivs_at_ele_center();

  //-----------------------------------------------------------------------------------------------
  // calculate material and stabilization parameters (one per transported scalar) at element center
  //-----------------------------------------------------------------------------------------------
  // density at t_(n) (one per transported scalar)
  std::vector<double> densn(my::numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F) (one per transported scalar)
  std::vector<double> densnp(my::numscal_, 1.0);
  // density at t_(n+alpha_M) (one per transported scalar)
  std::vector<double> densam(my::numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // stabilization variables
  std::vector<double> tau(my::numscal_, 0.);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> tauderpot(
      my::numscal_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));

  if (not my::scatrapara_->mat_gp() or not my::scatrapara_->tau_gp())
  {
    // set internal variables at element center
    set_internal_variables_for_mat_and_rhs();

    // material parameters at element center
    get_material_params(ele, densn, densnp, densam, visc);

    if (not my::scatrapara_->tau_gp()) prepare_stabilization(tau, tauderpot, densnp, vol);
  }

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    set_internal_variables_for_mat_and_rhs();

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (my::scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc, iquad);

    //-------------------------------------------------------------------------------------
    // calculate stabilization parameters (one per transported scalar) at integration point
    //-------------------------------------------------------------------------------------
    if (my::scatrapara_->tau_gp()) prepare_stabilization(tau, tauderpot, densnp, vol);

    //-----------------------------------------------------------------------------------
    // calculate contributions to element matrix and right-hand side at integration point
    //-----------------------------------------------------------------------------------
    const double timefacfac = my::scatraparatimint_->time_fac() * fac;
    const double rhsfac = my::scatraparatimint_->time_fac_rhs() * fac;

    // loop all scalars
    // deal with a system of transported scalars
    for (int k = 0; k < my::numscal_; ++k)
    {
      const double taufac = tau[k] * fac;
      const double timetaufac = my::scatraparatimint_->time_fac() * taufac;
      const double rhstaufac = my::scatraparatimint_->time_fac_rhs_tau() * taufac;

      // compute rhs containing bodyforce (divided by specific heat capacity) and,
      // for temperature equation, the time derivative of thermodynamic pressure,
      // if not constant, and for temperature equation of a reactive
      // equation system, the reaction-rate term
      double rhsint(0.0);
      my::get_rhs_int(rhsint, densnp[k], k);

      // Compute element matrix and rhs
      calc_mat_and_rhs(emat, erhs, k, fac, timefacfac, rhsfac, taufac, timetaufac, rhstaufac,
          tauderpot[k], rhsint);
    }  // end loop over scalar

    // Compute element matrix and rhs
    calc_mat_and_rhs_outside_scalar_loop(emat, erhs, fac, timefacfac, rhsfac);
  }
}


/*----------------------------------------------------------------------------------*
|  CalcMat: Potential equation ENC                                       ehrl  02/14|
*-----------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::calc_mat_pot_equ_enc(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                            //!< index of current scalar
    const double fac,                       //!< domain-integration factor
    const double alphaf                     //!< time factor for ENC
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      // linearization of the transference number in the conduction term (transport equation)
      //
      // (w, sum(z_k c_k))
      //
      // electroneutrality condition (only derivative w.r.t. concentration c_k)
      emat(vi * my::numdofpernode_ + my::numscal_, ui * my::numdofpernode_ + k) +=
          alphaf * diff_manager()->get_valence(k) * fac * my::funct_(vi) * my::funct_(ui);
    }
  }
}


/*-------------------------------------------------------------------------------------*
 |  CalcRhs: Potential equation ENC                                         ehrl 11/13 |
 *-------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcElch<distype, probdim>::calc_rhs_pot_equ_enc(
    Core::LinAlg::SerialDenseVector& erhs,  //!< element vector to be filled
    const int k,                            //!< index of current scalar
    const double fac,                       //!< domain-integration factor
    const double conint                     //!< concentration at GP
)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    // electroneutrality condition
    // for incremental formulation, there is the residuum on the rhs! : 0-sum(z_k c_k)
    erhs[vi * my::numdofpernode_ + my::numscal_] -=
        diff_manager()->get_valence(k) * fac * my::funct_(vi) * conint;
  }
}


// template classes
// 1D elements
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad4, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::hex8, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::tet10, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::pyramid5, 3>;
// template class Discret::Elements::ScaTraEleCalcElch<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
