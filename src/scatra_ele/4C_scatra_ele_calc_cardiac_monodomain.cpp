// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_cardiac_monodomain.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_cardiac_monodomain.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_myocard.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>::ScaTraEleCalcCardiacMonodomain(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype, probdim>::ScaTraEleCalc(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcAniso<distype, probdim>::ScaTraEleCalcAniso(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcAdvReac<distype, probdim>::ScaTraEleCalcAdvReac(
          numdofpernode, numscal, disname)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>*
Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcCardiacMonodomain<distype, probdim>>(
            new ScaTraEleCalcCardiacMonodomain<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    ljag 06/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>::materials(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point

)
{
  // safety check
  if (material->material_type() != Core::Materials::m_myocard)
    FOUR_C_THROW("Material type is not supported");

  // safety check
  std::shared_ptr<Mat::Myocard> actmat = std::dynamic_pointer_cast<Mat::Myocard>(
      std::const_pointer_cast<Core::Mat::Material>(material));
  if (actmat->get_number_of_gp() != 1 and not my::scatrapara_->mat_gp())
  {
    actmat->set_gp(1);
    actmat->resize_internal_state_variables();
  }
  mat_myocard(material, k, densn, densnp, densam, visc, iquad);

  return;
}


/*----------------------------------------------------------------------*
 |  Material ScaTra                                          ljag 06/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>::mat_myocard(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point (default = -1)
)
{
  const std::shared_ptr<const Mat::Myocard>& actmat =
      std::dynamic_pointer_cast<const Mat::Myocard>(material);

  // dynamic cast to Advanced_Reaction-specific reaction manager
  std::shared_ptr<ScaTraEleReaManagerAdvReac> advreamanager =
      std::dynamic_pointer_cast<ScaTraEleReaManagerAdvReac>(my::reamanager_);

  // dynamic cast to anisotropic diffusion manager
  std::shared_ptr<ScaTraEleDiffManagerAniso<nsd_>> diffmanageraniso =
      std::dynamic_pointer_cast<ScaTraEleDiffManagerAniso<nsd_>>(my::diffmanager_);

  // get constant diffusivity
  Core::LinAlg::Matrix<nsd_, nsd_> difftensor(Core::LinAlg::Initialization::zero);
  actmat->diffusivity(difftensor);

  diffmanageraniso->set_anisotropic_diff(difftensor, k);

  // clear
  advreamanager->clear(my::numscal_);

  if (my::scatrapara_->semi_implicit())
  {
    // get membrane potential at n at integration point
    const double phin = my::scatravarmanager_->phin(k);
    const double phinp = my::scatravarmanager_->phinp(k);
    // get reaction coefficient
    double react = -actmat->rea_coeff_n(phin, my::scatraparatimint_->dt(), iquad);
    if (my::scatraparatimint_->is_gen_alpha())
      react *= my::scatraparatimint_->dt() / my::scatraparatimint_->time_fac();
    advreamanager->add_to_rea_body_force(react, k);
    advreamanager->add_to_rea_body_force_deriv_matrix(0.0, k, k);
    actmat->rea_coeff(phinp, my::scatraparatimint_->dt(), iquad);
  }
  else
  {
    // get membrane potential at n+1 or n+alpha_F at integration point
    const double phinp = my::scatravarmanager_->phinp(k);
    // get reaction coefficient
    advreamanager->add_to_rea_body_force(
        -actmat->rea_coeff(phinp, my::scatraparatimint_->dt(), iquad), k);
    advreamanager->add_to_rea_body_force_deriv_matrix(
        -actmat->rea_coeff_deriv(phinp, my::scatraparatimint_->dt(), iquad), k, k);
  }

  return;
}  // ScaTraEleCalcCardiacMonodomain<distype>::MatMyocard


/*----------------------------------------------------------------------*
|  calculate system matrix and rhs for ep                 hoermann 06/16|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype, probdim>::sysmat(
    Core::Elements::Element* ele,               ///< the element whose matrix is calculated
    Core::LinAlg::SerialDenseMatrix& emat,      ///< element matrix to calculate
    Core::LinAlg::SerialDenseVector& erhs,      ///< element rhs to calculate
    Core::LinAlg::SerialDenseVector& subgrdiff  ///< subgrid-diff.-scaling vector
)
{
  // density at t_(n) (one per transported scalar)
  std::vector<double> densn(my::numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F) (one per transported scalar)
  std::vector<double> densnp(my::numscal_, 1.0);
  // density at t_(n+alpha_M) (one per transported scalar)
  std::vector<double> densam(my::numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // calculation of material parameter at element center
  if (not my::scatrapara_->mat_gp())
  {
    advreac::eval_shape_func_and_derivs_at_ele_center();
    // set Gauss point variables needed for evaluation of mat and rhs
    my::set_internal_variables_for_mat_and_rhs();
    advreac::get_material_params(ele, densn, densnp, densam, visc);
  }
  // calculation of material at integration points (different number of integration points possible)
  else
  {
    int deg = 0;
    if (ele->degree() == 1)
      deg = 4 * ele->degree();
    else
      deg = 3 * ele->degree();
    const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
        ScaTra::DisTypeToMatGaussRule<distype>::get_gauss_rule(deg));

    // loop over integration points
    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

      // set gauss point variables needed for evaluation of mat and rhs
      my::set_internal_variables_for_mat_and_rhs();

      // get material parameters (evaluation at integration point)
      advreac::get_material_params(ele, densn, densnp, densam, visc, iquad);

      // loop all scalars
      for (int k = 0; k < my::numscal_; ++k)  // deal with a system of transported scalars
      {
        double rhsint(0.0);
        advreac::get_rhs_int(rhsint, densnp[k], k);

        Core::LinAlg::Matrix<nen_, 1> dummy(Core::LinAlg::Initialization::zero);
        const double timefacfac = my::scatraparatimint_->time_fac() * fac;

        // reactive terms on integration point on rhs
        my::compute_rhs_int(rhsint, densam[k], densnp[k], my::scatravarmanager_->hist(k));

        // standard Galerkin transient, old part of rhs and bodyforce term
        my::calc_rhs_hist_and_source(erhs, k, fac, rhsint);

        // element matrix: reactive term
        advreac::calc_mat_react(emat, k, timefacfac, 0., 0., densnp[k], dummy, dummy);
      }
    }
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

    // set gauss point variables needed for evaluation of mat and rhs
    my::set_internal_variables_for_mat_and_rhs();

    // loop all scalars
    for (int k = 0; k < my::numscal_; ++k)  // deal with a system of transported scalars
    {
      // compute rhs containing bodyforce
      double rhsint(0.0);
      advreac::get_rhs_int(rhsint, densnp[k], k);

      // integration factors
      const double timefacfac = my::scatraparatimint_->time_fac() * fac;

      //----------------------------------------------------------------
      // 1) element matrix: stationary terms
      //----------------------------------------------------------------

      // calculation of diffusive element matrix
      aniso::calc_mat_diff(emat, k, timefacfac);

      //----------------------------------------------------------------
      // 2) element matrix: instationary terms
      //----------------------------------------------------------------

      if (not my::scatraparatimint_->is_stationary()) my::calc_mat_mass(emat, k, fac, densam[k]);

      //----------------------------------------------------------------
      // 3) element matrix: reactive term
      //----------------------------------------------------------------

      Core::LinAlg::Matrix<nen_, 1> dummy(Core::LinAlg::Initialization::zero);
      if (not my::scatrapara_->mat_gp())
        advreac::calc_mat_react(emat, k, timefacfac, 0., 0., densnp[k], dummy, dummy);

      //----------------------------------------------------------------
      // 5) element right hand side
      //----------------------------------------------------------------
      //----------------------------------------------------------------
      // computation of bodyforce (and potentially history) term,
      // residual, integration factors and standard Galerkin transient
      // term (if required) on right hand side depending on respective
      // (non-)incremental stationary or time-integration scheme
      //----------------------------------------------------------------
      double rhsfac = my::scatraparatimint_->time_fac_rhs() * fac;

      if (my::scatraparatimint_->is_incremental() and not my::scatraparatimint_->is_stationary())
        my::calc_rhs_lin_mass(erhs, k, rhsfac, fac, densam[k], densnp[k]);


      if (not my::scatrapara_->mat_gp())
      {
        my::compute_rhs_int(rhsint, densam[k], densnp[k], my::scatravarmanager_->hist(k));
        // standard Galerkin transient, old part of rhs and bodyforce term
        my::calc_rhs_hist_and_source(erhs, k, fac, rhsint);
      }

      //----------------------------------------------------------------
      // standard Galerkin terms on right hand side
      //----------------------------------------------------------------

      // diffusive term
      aniso::calc_rhs_diff(erhs, k, rhsfac);

    }  // end loop all scalars
  }  // end loop Gauss points

  return;
}


/*----------------------------------------------------------------------*
 | extract element based or nodal values                 hoermann 06/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcCardiacMonodomain<distype,
    probdim>::extract_element_and_node_values(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  my::extract_element_and_node_values(ele, params, discretization, la);

  // extract additional local values from global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> phin = discretization.get_state("phin");
  if (phin == nullptr) FOUR_C_THROW("Cannot get state vector 'phin'");
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phin, my::ephin_, la[0].lm_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes
// 1D elements
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad4, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::hex8, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::tet10, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::pyramid5, 3>;
// template class
// Discret::Elements::ScaTraEleCalcCardiacMonodomain<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
