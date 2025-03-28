// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_sti_elch.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"

FOUR_C_NAMESPACE_OPEN

/*-------------------------------------------------------------------------------------------------------------------------------------*
 | element matrix and right-hand side vector contributions arising from thermal source terms in
 discrete thermo residuals   fang 11/15 |
 *-------------------------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleSTIElch<distype>::calc_mat_and_rhs_source(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
    const double& timefacfac,  //!< domain integration factor times time integration factor
    const double& rhsfac       //!< domain integration factor times time integration factor for
                               //!< right-hand side vector
)
{
  // matrix and vector contributions arising from Joule's heat
  calc_mat_and_rhs_joule(emat, erhs, timefacfac, rhsfac);

  // matrix and vector contributions arising from heat of mixing
  calc_mat_and_rhs_mixing(emat, erhs, timefacfac, rhsfac);

  // matrix and vector contributions arising from Soret effect
  calc_mat_and_rhs_soret(emat, erhs, timefacfac, rhsfac);

  return;
};


/*-------------------------------------------------------------------------------------------------------------------------*
 | provide element matrix with linearizations of source terms in discrete thermo residuals w.r.t.
 scatra dofs   fang 11/15 |
 *-------------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleSTIElch<distype>::calc_mat_source_od(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const double& timefacfac  //!< domain integration factor times time integration factor
)
{
  // matrix contributions arising from Joule's heat
  calc_mat_joule_od(emat, timefacfac);

  // matrix contributions arising from heat of mixing
  calc_mat_mixing_od(emat, timefacfac);

  // matrix contributions arising from Soret effect
  calc_mat_soret_od(emat, timefacfac);

  return;
};


/*----------------------------------------------------------------------*
 | extract quantities for element evaluation                 fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleSTIElch<distype>::extract_element_and_node_values(
    Core::Elements::Element* ele,              //!< current element
    Teuchos::ParameterList& params,            //!< parameter list
    Core::FE::Discretization& discretization,  //!< discretization
    Core::Elements::LocationArray& la          //!< location array
)
{
  // extract electrochemistry state vector from discretization
  const std::shared_ptr<const Core::LinAlg::Vector<double>> elchnp =
      discretization.get_state(2, "scatra");
  if (elchnp == nullptr)
    FOUR_C_THROW("Cannot extract electrochemistry state vector from discretization!");

  // extract local nodal values of concentration and electric potential from global state vector
  const std::vector<int>& lm = la[2].lm_;
  std::vector<double> myelchnp = Core::FE::extract_values(*elchnp, lm);
  for (int inode = 0; inode < nen_; ++inode)
  {
    econcnp_(inode) = myelchnp[inode * 2];
    epotnp_(inode) = myelchnp[inode * 2 + 1];
  }

  return;
}


/*----------------------------------------------------------------------*
 | protected constructor for singletons                      fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleSTIElch<distype>::ScaTraEleSTIElch(
    const int numdofpernode, const int numscal, const std::string& disname)
    : econcnp_(Core::LinAlg::Initialization::zero), epotnp_(Core::LinAlg::Initialization::zero)
{
  return;
}


// template classes
// 1D elements
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::quad9>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::pyramid5>;
// template class Discret::Elements::ScaTraEleSTIElch<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
