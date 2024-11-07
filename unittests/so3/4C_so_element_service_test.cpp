// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_utils_gauss_point_extrapolation.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_so3_element_service.hpp"

#include <vector>

namespace
{
  using namespace FourC;

  TEST(ElementServiceTest, TestProjectNodalQuantityToXiHex8)
  {
    Core::LinAlg::Matrix<3, 1> xi(true);
    xi(0) = 0.01, xi(1) = 0.25, xi(2) = 0.115;
    std::vector<double> nodal_quantity = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
    std::vector<double> ref_val{1.397875};
    auto test_val = Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::hex8>(
        xi, nodal_quantity);
    for (std::size_t i = 0; i < ref_val.size(); ++i) EXPECT_NEAR(test_val[i], ref_val[i], 1.0e-10);
  }

  TEST(ElementServiceTest, TestProjectNodalQuantityToXiHex27)
  {
    Core::LinAlg::Matrix<3, 1> xi(true);
    xi(0) = 0.01, xi(1) = 0.25, xi(2) = 0.115;
    std::vector<double> nodal_quantity = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6};
    std::vector<double> ref_val = {3.623649611383};
    auto test_val = Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::hex27>(
        xi, nodal_quantity);
    for (std::size_t i = 0; i < ref_val.size(); ++i) EXPECT_NEAR(test_val[i], ref_val[i], 1.0e-10);
  }

  TEST(ElementServiceTest, TestProjectNodalQuantityToXiTet4)
  {
    Core::LinAlg::Matrix<3, 1> xi(true);
    xi(0) = 0.01, xi(1) = 0.25, xi(2) = 0.115;
    std::vector<double> nodal_quantity = {1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 1.3, 2.3};
    std::vector<double> ref_val = {1.0855, 2.0855};
    auto test_val = Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::tet4>(
        xi, nodal_quantity);
    for (std::size_t i = 0; i < ref_val.size(); ++i) EXPECT_NEAR(test_val[i], ref_val[i], 1.0e-10);
  }

  TEST(ElementServiceTest, TestProjectNodalQuantityToXiTet10)
  {
    Core::LinAlg::Matrix<3, 1> xi(true);
    xi(0) = 0.01, xi(1) = 0.25, xi(2) = 0.115;
    std::vector<double> nodal_quantity = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};
    std::vector<double> ref_val = {1.645885};
    auto test_val = Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::tet10>(
        xi, nodal_quantity);
    for (std::size_t i = 0; i < ref_val.size(); ++i) EXPECT_NEAR(test_val[i], ref_val[i], 1.0e-10);
  }

  TEST(ElementServiceTest, TestProjectNodalQuantityToXiWedge6)
  {
    Core::LinAlg::Matrix<3, 1> xi(true);
    xi(0) = 0.01, xi(1) = 0.25, xi(2) = 0.115;
    std::vector<double> nodal_quantity = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
    std::vector<double> ref_val = {1.34025};
    auto test_val = Discret::Elements::project_nodal_quantity_to_xi<Core::FE::CellType::wedge6>(
        xi, nodal_quantity);
    for (std::size_t i = 0; i < ref_val.size(); ++i) EXPECT_NEAR(test_val[i], ref_val[i], 1.0e-10);
  }

  TEST(ElementServiceTest, TestGaussPointProjectionMatrixHex8)
  {
    constexpr Core::FE::CellType distype = Core::FE::CellType::hex8;
    constexpr int nsd = 3;

    Core::FE::IntPointsAndWeights<nsd> intpoints(
        Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

    // format as Discret::Utils::GaussIntegration
    std::shared_ptr<Core::FE::CollectedGaussPoints> gp =
        std::make_shared<Core::FE::CollectedGaussPoints>();

    std::array<double, nsd> xi{};
    for (int i = 0; i < intpoints.ip().nquad; ++i)
    {
      for (int d = 0; d < nsd; ++d) xi[d] = intpoints.ip().qxg[i][d];
      gp->append(xi[0], xi[1], xi[2], intpoints.ip().qwgt[i]);
    }

    // save default integration rule
    Core::FE::GaussIntegration integration(gp);
    Core::LinAlg::SerialDenseMatrix m =
        Core::FE::evaluate_gauss_points_to_nodes_extrapolation_matrix<distype>(integration);
  }

}  // namespace
