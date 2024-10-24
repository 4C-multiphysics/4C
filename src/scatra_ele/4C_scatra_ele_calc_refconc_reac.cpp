// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_refconc_reac.hpp"

#include "4C_mat_list_reactions.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcRefConcReac<distype>::ScaTraEleCalcRefConcReac(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcAdvReac<distype>::ScaTraEleCalcAdvReac(
          numdofpernode, numscal, disname),
      j_(1.0),
      c_inv_(true),
      d_jd_x_(true)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcRefConcReac<distype>*
Discret::Elements::ScaTraEleCalcRefConcReac<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcRefConcReac<distype>>(
            new ScaTraEleCalcRefConcReac<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}

//!
/*----------------------------------------------------------------------------*
 |  Set reac. body force, reaction coefficient and derivatives     thon 02/16 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcRefConcReac<distype>::set_advanced_reaction_terms(
    const int k,                                            //!< index of current scalar
    const Teuchos::RCP<Mat::MatListReactions> matreaclist,  //!< index of current scalar
    const double* gpcoord                                   //!< current Gauss-point coordinates
)
{
  const Teuchos::RCP<ScaTraEleReaManagerAdvReac> remanager = advreac::rea_manager();

  remanager->add_to_rea_body_force(
      matreaclist->calc_rea_body_force_term(k, my::scatravarmanager_->phinp(), gpcoord, 1.0 / j_) *
          j_,
      k);

  matreaclist->calc_rea_body_force_deriv_matrix(
      k, remanager->get_rea_body_force_deriv_vector(k), my::scatravarmanager_->phinp(), gpcoord);
}

/*------------------------------------------------------------------------------------------*
 |  calculation of convective element matrix: add conservative contributions     thon 02/16 |
 *------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcRefConcReac<distype>::calc_mat_conv_add_cons(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac, const double vdiv,
    const double densnp)
{
  FOUR_C_THROW(
      "If you want to calculate the reference concentrations the CONVFORM must be 'convective'!");
}

/*------------------------------------------------------------------------------*
 | set internal variables                                           thon 02/16  |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcRefConcReac<distype>::set_internal_variables_for_mat_and_rhs()
{
  // do the usual and...
  advreac::set_internal_variables_for_mat_and_rhs();

  /////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////
  // spatial node coordinates
  Core::LinAlg::Matrix<nsd_, nen_> xyze(my::xyze_);
  xyze += my::edispnp_;

  //! transposed jacobian "dx/ds"
  Core::LinAlg::Matrix<nsd_, nsd_> dxds(true);
  dxds.multiply_nt(my::deriv_, xyze);

  // deformation gradtient dx/dX = dx/ds * ds/dX = dx/ds * (dX/ds)^(-1)
  Core::LinAlg::Matrix<nsd_, nsd_> F(true);
  F.multiply_tt(dxds, my::xij_);

  // inverse of jacobian "dx/dX"
  Core::LinAlg::Matrix<nsd_, nsd_> F_inv(true);
  j_ = F_inv.invert(F);

  // calculate inverse of cauchy-green stress tensor
  c_inv_.multiply_nt(F_inv, F_inv);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // calculate derivative dJ/dX by finite differences
  ////////////////////////////////////////////////////////////////////////////////////////////////
  const double epsilon = 1.0e-8;

  for (unsigned i = 0; i < 3; i++)
  {
    Core::LinAlg::Matrix<nsd_, nen_> xyze_epsilon(my::xyze_);
    for (unsigned j = 0; j < nen_; ++j) xyze_epsilon(i, j) = xyze_epsilon(i, j) + epsilon;

    Core::LinAlg::Matrix<nsd_, nsd_> xjm_epsilon(true);
    xjm_epsilon.multiply_nt(my::deriv_, xyze_epsilon);

    Core::LinAlg::Matrix<nsd_, nsd_> xij_epsilon(true);
    xij_epsilon.invert(xjm_epsilon);

    // dx/dX = dx/ds * ds/dX = dx/ds * (dX/ds)^(-1)
    Core::LinAlg::Matrix<nsd_, nsd_> F_epsilon(true);
    F_epsilon.multiply_tt(dxds, xij_epsilon);

    // inverse of transposed jacobian "ds/dX"
    const double J_epsilon = F_epsilon.determinant();
    const double dJdX_i = (J_epsilon - j_) / epsilon;

    d_jd_x_(i, 0) = dJdX_i;
  }

  return;
}

/*------------------------------------------------------------------- *
 |  calculation of diffusive element matrix                thon 02/16 |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcRefConcReac<distype>::calc_mat_diff(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac)
{
  Core::LinAlg::Matrix<nsd_, nsd_> Diff_tens(c_inv_);
  Diff_tens.scale(my::diffmanager_->get_isotropic_diff(k));

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * my::numdofpernode_ + k;

      double laplawf = 0.0;
      //      get_laplacian_weak_form(laplawf,Diff_tens,vi,ui);
      for (unsigned j = 0; j < nsd_; j++)
      {
        for (unsigned i = 0; i < nsd_; i++)
        {
          laplawf += my::derxy_(j, vi) * Diff_tens(j, i) * my::derxy_(i, ui);
        }
      }

      emat(fvi, fui) += timefacfac * laplawf;
    }
  }



  Core::LinAlg::Matrix<nsd_, nsd_> Diff_tens2(c_inv_);
  Diff_tens2.scale(my::diffmanager_->get_isotropic_diff(k) / j_);

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_ + k;

    double laplawf2 = 0.0;
    for (unsigned j = 0; j < nsd_; j++)
    {
      for (unsigned i = 0; i < nsd_; i++)
      {
        laplawf2 += my::derxy_(j, vi) * Diff_tens2(j, i) * d_jd_x_(i);
      }
    }

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const int fui = ui * my::numdofpernode_ + k;

      emat(fvi, fui) -= timefacfac * laplawf2 * my::funct_(ui);
    }
  }

  return;
}

/*-------------------------------------------------------------------- *
 |  standard Galerkin diffusive term on right hand side     ehrl 11/13 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcRefConcReac<distype>::calc_rhs_diff(
    Core::LinAlg::SerialDenseVector& erhs, const int k, const double rhsfac)
{
  /////////////////////////////////////////////////////////////////////
  // \D* \grad c_0 \times \grad \phi ...
  /////////////////////////////////////////////////////////////////////
  Core::LinAlg::Matrix<nsd_, nsd_> Diff_tens(c_inv_);
  Diff_tens.scale(my::diffmanager_->get_isotropic_diff(k));

  const Core::LinAlg::Matrix<nsd_, 1>& gradphi = my::scatravarmanager_->grad_phi(k);

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_ + k;

    double laplawf(0.0);
    //    get_laplacian_weak_form_rhs(laplawf,Diff_tens,gradphi,vi);
    for (unsigned j = 0; j < nsd_; j++)
    {
      for (unsigned i = 0; i < nsd_; i++)
      {
        laplawf += my::derxy_(j, vi) * Diff_tens(j, i) * gradphi(i);
      }
    }

    erhs[fvi] -= rhsfac * laplawf;
  }

  /////////////////////////////////////////////////////////////////////
  // ... + \D* c_0/J * \grad J \times \grad \phi
  /////////////////////////////////////////////////////////////////////
  Core::LinAlg::Matrix<nsd_, nsd_> Diff_tens2(c_inv_);
  Diff_tens2.scale(my::diffmanager_->get_isotropic_diff(k) / j_ * my::scatravarmanager_->phinp(k));

  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const int fvi = vi * my::numdofpernode_ + k;

    double laplawf2(0.0);
    //    get_laplacian_weak_form_rhs(laplawf2,Diff_tens2,dJdX_,vi);
    for (unsigned j = 0; j < nsd_; j++)
    {
      for (unsigned i = 0; i < nsd_; i++)
      {
        laplawf2 += my::derxy_(j, vi) * Diff_tens2(j, i) * d_jd_x_(i);
      }
    }

    erhs[fvi] += rhsfac * laplawf2;
  }

  return;
}



// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::quad9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::nurbs9>;
// template class
// Discret::Elements::ScaTraEleCalcRefConcReac<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
