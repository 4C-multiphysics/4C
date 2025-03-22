// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_poro_reac.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_mat_structporo_reaction_ecm.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcPoroReac<distype>::ScaTraEleCalcPoroReac(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcPoro<distype>::ScaTraEleCalcPoro(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcAdvReac<distype>::ScaTraEleCalcAdvReac(
          numdofpernode, numscal, disname)
{
  // safety check
  if (not my::scatrapara_->tau_gp())
    FOUR_C_THROW("For poro reactions, tau needs to be evaluated by integration-point evaluations!");

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcPoroReac<distype>*
Discret::Elements::ScaTraEleCalcPoroReac<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcPoroReac<distype>>(
            new ScaTraEleCalcPoroReac<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    vuong 10/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoroReac<distype>::get_material_params(
    const Core::Elements::Element* ele,  //!< the element we are dealing with
    std::vector<double>& densn,          //!< density at t_(n)
    std::vector<double>& densnp,         //!< density at t_(n+1) or t_(n+alpha_F)
    std::vector<double>& densam,         //!< density at t_(n+alpha_M)
    double& visc,                        //!< fluid viscosity
    const int iquad                      //!< id of current gauss point
)
{
  // call poro base class to compute porosity
  poro::compute_porosity(ele);

  // call advreac base class
  advreac::get_material_params(ele, densn, densnp, densam, visc, iquad);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                    thon 02/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoroReac<distype>::materials(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  switch (material->material_type())
  {
    case Core::Materials::m_scatra:
      mat_scatra(material, k, densn, densnp, densam, visc, iquad);
      break;
    default:
      FOUR_C_THROW("Material type {} is not supported", material->material_type());
      break;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Material ScaTra                                          thon 02/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoroReac<distype>::mat_scatra(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  poro::mat_scatra(material, k, densn, densnp, densam, visc, iquad);

  return;
}  // ScaTraEleCalcPoroReac<distype>::MatScaTra

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     vuong 04/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcPoroReac<distype>::extract_element_and_node_values(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // call base class routine
  poro::extract_element_and_node_values(ele, params, discretization, la);

  return;
}

// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::quad9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::nurbs9>;
// template class Discret::Elements::ScaTraEleCalcPoroReac<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
