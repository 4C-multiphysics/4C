// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_meshintersection.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_tetmeshintersection.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cut_test_utils.hpp"

void test_alex52()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1543893689e-01;
    tri3_xyze(1, 0) = 5.8055238132e-02;
    tri3_xyze(2, 0) = 2.2476988705e-01;
    tri3_xyze(0, 1) = 9.1544731933e-01;
    tri3_xyze(1, 1) = 5.8036481457e-02;
    tri3_xyze(2, 1) = 1.9255072753e-01;
    tri3_xyze(0, 2) = 9.1787432923e-01;
    tri3_xyze(1, 2) = 7.2708142399e-02;
    tri3_xyze(2, 2) = 2.0866775263e-01;
    nids.clear();
    nids.push_back(537);
    nids.push_back(541);
    nids.push_back(572);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1544731933e-01;
    tri3_xyze(1, 0) = 5.8036481457e-02;
    tri3_xyze(2, 0) = 1.9255072753e-01;
    tri3_xyze(0, 1) = 9.2031850428e-01;
    tri3_xyze(1, 1) = 8.7365486164e-02;
    tri3_xyze(2, 1) = 1.9255679313e-01;
    tri3_xyze(0, 2) = 9.1787432923e-01;
    tri3_xyze(1, 2) = 7.2708142399e-02;
    tri3_xyze(2, 2) = 2.0866775263e-01;
    nids.clear();
    nids.push_back(541);
    nids.push_back(571);
    nids.push_back(572);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.2031850428e-01;
    tri3_xyze(1, 0) = 8.7365486164e-02;
    tri3_xyze(2, 0) = 1.9255679313e-01;
    tri3_xyze(0, 1) = 9.1544731933e-01;
    tri3_xyze(1, 1) = 5.8036481457e-02;
    tri3_xyze(2, 1) = 1.9255072753e-01;
    tri3_xyze(0, 2) = 9.1786796209e-01;
    tri3_xyze(1, 2) = 7.2704172565e-02;
    tri3_xyze(2, 2) = 1.7647355445e-01;
    nids.clear();
    nids.push_back(571);
    nids.push_back(541);
    nids.push_back(574);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1229823863e-01;
    tri3_xyze(1, 0) = 2.8863203307e-02;
    tri3_xyze(2, 0) = 1.9253258983e-01;
    tri3_xyze(0, 1) = 9.1544731933e-01;
    tri3_xyze(1, 1) = 5.8036481457e-02;
    tri3_xyze(2, 1) = 1.9255072753e-01;
    tri3_xyze(0, 2) = 9.1387082135e-01;
    tri3_xyze(1, 2) = 4.3459312180e-02;
    tri3_xyze(2, 2) = 2.0864097695e-01;
    nids.clear();
    nids.push_back(539);
    nids.push_back(541);
    nids.push_back(542);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1541827194e-01;
    tri3_xyze(1, 0) = 5.8040009143e-02;
    tri3_xyze(2, 0) = 1.6039611391e-01;
    tri3_xyze(0, 1) = 9.1544731933e-01;
    tri3_xyze(1, 1) = 5.8036481457e-02;
    tri3_xyze(2, 1) = 1.9255072753e-01;
    tri3_xyze(0, 2) = 9.1386102652e-01;
    tri3_xyze(1, 2) = 4.3451325888e-02;
    tri3_xyze(2, 2) = 1.7646938018e-01;
    nids.clear();
    nids.push_back(545);
    nids.push_back(541);
    nids.push_back(546);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1544731933e-01;
    tri3_xyze(1, 0) = 5.8036481457e-02;
    tri3_xyze(2, 0) = 1.9255072753e-01;
    tri3_xyze(0, 1) = 9.1229823863e-01;
    tri3_xyze(1, 1) = 2.8863203307e-02;
    tri3_xyze(2, 1) = 1.9253258983e-01;
    tri3_xyze(0, 2) = 9.1386102652e-01;
    tri3_xyze(1, 2) = 4.3451325888e-02;
    tri3_xyze(2, 2) = 1.7646938018e-01;
    nids.clear();
    nids.push_back(541);
    nids.push_back(539);
    nids.push_back(546);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1229823863e-01;
    tri3_xyze(1, 0) = 2.8863203307e-02;
    tri3_xyze(2, 0) = 1.9253258983e-01;
    tri3_xyze(0, 1) = 9.1228027620e-01;
    tri3_xyze(1, 1) = 2.8865609647e-02;
    tri3_xyze(2, 1) = 1.6039808947e-01;
    tri3_xyze(0, 2) = 9.1386102652e-01;
    tri3_xyze(1, 2) = 4.3451325888e-02;
    tri3_xyze(2, 2) = 1.7646938018e-01;
    nids.clear();
    nids.push_back(539);
    nids.push_back(543);
    nids.push_back(546);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1544731933e-01;
    tri3_xyze(1, 0) = 5.8036481457e-02;
    tri3_xyze(2, 0) = 1.9255072753e-01;
    tri3_xyze(0, 1) = 9.1541827194e-01;
    tri3_xyze(1, 1) = 5.8040009143e-02;
    tri3_xyze(2, 1) = 1.6039611391e-01;
    tri3_xyze(0, 2) = 9.1786796209e-01;
    tri3_xyze(1, 2) = 7.2704172565e-02;
    tri3_xyze(2, 2) = 1.7647355445e-01;
    nids.clear();
    nids.push_back(541);
    nids.push_back(545);
    nids.push_back(574);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1229879053e-01;
    tri3_xyze(1, 0) = 2.8882325825e-02;
    tri3_xyze(2, 0) = 2.2471070339e-01;
    tri3_xyze(0, 1) = 9.1229823863e-01;
    tri3_xyze(1, 1) = 2.8863203307e-02;
    tri3_xyze(2, 1) = 1.9253258983e-01;
    tri3_xyze(0, 2) = 9.1387082135e-01;
    tri3_xyze(1, 2) = 4.3459312180e-02;
    tri3_xyze(2, 2) = 2.0864097695e-01;
    nids.clear();
    nids.push_back(535);
    nids.push_back(539);
    nids.push_back(542);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.1544731933e-01;
    tri3_xyze(1, 0) = 5.8036481457e-02;
    tri3_xyze(2, 0) = 1.9255072753e-01;
    tri3_xyze(0, 1) = 9.1543893689e-01;
    tri3_xyze(1, 1) = 5.8055238132e-02;
    tri3_xyze(2, 1) = 2.2476988705e-01;
    tri3_xyze(0, 2) = 9.1387082135e-01;
    tri3_xyze(1, 2) = 4.3459312180e-02;
    tri3_xyze(2, 2) = 2.0864097695e-01;
    nids.clear();
    nids.push_back(541);
    nids.push_back(537);
    nids.push_back(542);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

  hex8_xyze(0, 0) = 9.2537313433e-01;
  hex8_xyze(1, 0) = 2.9629629630e-02;
  hex8_xyze(2, 0) = 1.7647058824e-01;
  hex8_xyze(0, 1) = 9.2537313433e-01;
  hex8_xyze(1, 1) = 5.9259259259e-02;
  hex8_xyze(2, 1) = 1.7647058824e-01;
  hex8_xyze(0, 2) = 8.9552238806e-01;
  hex8_xyze(1, 2) = 5.9259259259e-02;
  hex8_xyze(2, 2) = 1.7647058824e-01;
  hex8_xyze(0, 3) = 8.9552238806e-01;
  hex8_xyze(1, 3) = 2.9629629630e-02;
  hex8_xyze(2, 3) = 1.7647058824e-01;
  hex8_xyze(0, 4) = 9.2537313433e-01;
  hex8_xyze(1, 4) = 2.9629629630e-02;
  hex8_xyze(2, 4) = 2.0588235294e-01;
  hex8_xyze(0, 5) = 9.2537313433e-01;
  hex8_xyze(1, 5) = 5.9259259259e-02;
  hex8_xyze(2, 5) = 2.0588235294e-01;
  hex8_xyze(0, 6) = 8.9552238806e-01;
  hex8_xyze(1, 6) = 5.9259259259e-02;
  hex8_xyze(2, 6) = 2.0588235294e-01;
  hex8_xyze(0, 7) = 8.9552238806e-01;
  hex8_xyze(1, 7) = 2.9629629630e-02;
  hex8_xyze(2, 7) = 2.0588235294e-01;

  nids.clear();
  for (int i = 0; i < 8; ++i) nids.push_back(i);

  intersection.add_element(1, nids, hex8_xyze, Core::FE::CellType::hex8);
  intersection.cut_test_cut(true, Cut::VCellGaussPts_DirectDivergence);
}
