// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_chemo_reac.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_list_chemoreac.hpp"
#include "4C_mat_list_chemotaxis.hpp"
#include "4C_mat_list_reactions.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcChemoReac<distype, probdim>::ScaTraEleCalcChemoReac(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype, probdim>::ScaTraEleCalc(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcChemo<distype, probdim>::ScaTraEleCalcChemo(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcAdvReac<distype, probdim>::ScaTraEleCalcAdvReac(
          numdofpernode, numscal, disname)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcChemoReac<distype, probdim>*
Discret::Elements::ScaTraEleCalcChemoReac<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcChemoReac<distype, probdim>>(
            new ScaTraEleCalcChemoReac<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 |  get the material constants  (private)                    thon 06/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcChemoReac<distype, probdim>::get_material_params(
    const Core::Elements::Element* ele,  //!< the element we are dealing with
    std::vector<double>& densn,          //!< density at t_(n)
    std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,         //!< density at t_(n+alpha_M)
    double& visc,                        //!< fluid viscosity
    const int iquad                      //!< id of current gauss point (default = -1)
)
{
  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  // We may have some reactive and some non-reactive elements in one discretisation.
  // But since the calculation classes are singleton, we have to reset all reactive stuff in case
  // of non-reactive elements:
  advreac::rea_manager()->clear(my::numscal_);

  // We may have some chemotactic and some non-chemotactic discretisation.
  // But since the calculation classes are singleton, we have to reset all chemotaxis stuff each
  // time
  chemo::clear_chemotaxis_terms();

  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<const Mat::MatList> actmat =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (actmat->num_mat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      my::materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }

  else if (material->material_type() == Core::Materials::m_matlist_reactions)
  {
    const std::shared_ptr<Mat::MatListReactions> actmat =
        std::dynamic_pointer_cast<Mat::MatListReactions>(material);
    if (actmat->num_mat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      // Note: order is important here!!
      advreac::materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);

      advreac::set_advanced_reaction_terms(k, actmat,
          advreac::get_gp_coord(), 1.0, ele->id());  // every reaction calculation stuff happens in here!!
      // by default scale is set to 1.0 (cf 4C_mat_list_reactions)
    }
  }

  else if (material->material_type() == Core::Materials::m_matlist_chemotaxis)
  {
    const std::shared_ptr<Mat::MatListChemotaxis> actmat =
        std::dynamic_pointer_cast<Mat::MatListChemotaxis>(material);
    if (actmat->num_mat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    chemo::get_chemotaxis_coefficients(
        material);  // read all chemotaxis input from material and copy it into local variables

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      my::materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }

  else if (material->material_type() == Core::Materials::m_matlist_chemoreac)
  {
    const std::shared_ptr<Mat::MatListReactions> actmat =
        std::dynamic_pointer_cast<Mat::MatListReactions>(material);
    if (actmat->num_mat() != my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    chemo::get_chemotaxis_coefficients(
        material);  // read all chemotaxis input from material and copy it into local variables

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      // Note: order is important here!!
      my::materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
      advreac::set_advanced_reaction_terms(k, actmat,
          advreac::get_gp_coord(), 1.0, ele->id());  // every reaction calculation stuff happens in here!!
      // by default scale is set to 1.0 (cf 4C_mat_list_reactions)
    }
  }

  else
  {
    advreac::materials(material, 0, densn[0], densnp[0], densam[0], visc, iquad);
  }

  return;
}  // ScaTraEleCalc::get_material_params


// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::quad9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::nurbs9>;
// template class Discret::Elements::ScaTraEleCalcChemoReac<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
