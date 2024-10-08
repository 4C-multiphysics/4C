/*! \file

\brief A library of free functions for a default solid element

\level 1
*/

#ifndef FOUR_C_SOLID_3D_ELE_CALC_LIB_INTEGRATION_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_LIB_INTEGRATION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN
namespace Discret::ELEMENTS
{
  /*!
   * @brief Compare two Gauss integration rules for equality
   */
  inline bool compare_gauss_integration(const Core::FE::GaussIntegration& integration_a,
      const Core::FE::GaussIntegration& integration_b)
  {
    // currently this simple check is sufficient as we only use the same type of gauss integrations.
    return integration_a.num_points() == integration_b.num_points();
  }

  /*!
   * @brief Get the default Gauss integration rules for different Cell types.
   *
   * @note It follows the rules defined in Discret::ELEMENTS::DisTypeToOptGaussRule<celltype>::rule,
   * except for the stiffness matrix of tetrahedral elements.
   *
   */
  /// @{

  template <Core::FE::CellType celltype>
  constexpr auto get_gauss_rule_mass_matrix()
  {
    return Discret::ELEMENTS::DisTypeToOptGaussRule<celltype>::rule;
  }

  template <Core::FE::CellType celltype>
  constexpr auto get_gauss_rule_stiffness_matrix()
  {
    return Discret::ELEMENTS::DisTypeToOptGaussRule<celltype>::rule;
  }

  template <>
  constexpr auto get_gauss_rule_stiffness_matrix<Core::FE::CellType::tet10>()
  {
    return Core::FE::GaussRule3D::tet_4point;
  }

  template <>
  constexpr auto get_gauss_rule_stiffness_matrix<Core::FE::CellType::tet4>()
  {
    return Core::FE::GaussRule3D::tet_1point;
  }

  /*!
   * @brief Create a Gauss integration interface from a given Gauss rule type
   */
  template <Core::FE::CellType celltype, typename GaussRuleType>
  Core::FE::GaussIntegration create_gauss_integration(GaussRuleType rule)
  {
    // setup default integration
    Core::FE::IntPointsAndWeights<Core::FE::dim<celltype>> intpoints(rule);

    // format as Discret::UTILS::GaussIntegration
    Teuchos::RCP<Core::FE::CollectedGaussPoints> gp =
        Teuchos::make_rcp<Core::FE::CollectedGaussPoints>();

    std::array<double, 3> xi = {0., 0., 0.};
    for (int i = 0; i < intpoints.ip().nquad; ++i)
    {
      for (int d = 0; d < Core::FE::dim<celltype>; ++d) xi[d] = intpoints.ip().qxg[i][d];
      gp->append(xi[0], xi[1], xi[2], intpoints.ip().qwgt[i]);
    }

    return Core::FE::GaussIntegration(gp);
  }
  /// @}
}  // namespace Discret::ELEMENTS
FOUR_C_NAMESPACE_CLOSE

#endif