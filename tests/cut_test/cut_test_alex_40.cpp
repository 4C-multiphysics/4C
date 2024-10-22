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
#include <list>
#include <map>
#include <string>
#include <vector>

#include "cut_test_utils.hpp"

void test_alex40()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.588142e-01;
    tri3_xyze(1, 0) = 1.585063e-01;
    tri3_xyze(2, 0) = 3.208616e-01;
    tri3_xyze(0, 1) = 8.855745e-01;
    tri3_xyze(1, 1) = 1.579358e-01;
    tri3_xyze(2, 1) = 3.209260e-01;
    tri3_xyze(0, 2) = 8.725808e-01;
    tri3_xyze(1, 2) = 1.740499e-01;
    tri3_xyze(2, 2) = 3.208935e-01;
    nids.clear();
    nids.push_back(174);
    nids.push_back(176);
    nids.push_back(206);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.855745e-01;
    tri3_xyze(1, 0) = 1.579358e-01;
    tri3_xyze(2, 0) = 3.209260e-01;
    tri3_xyze(0, 1) = 8.588142e-01;
    tri3_xyze(1, 1) = 1.585063e-01;
    tri3_xyze(2, 1) = 3.208616e-01;
    tri3_xyze(0, 2) = 8.718493e-01;
    tri3_xyze(1, 2) = 1.423912e-01;
    tri3_xyze(2, 2) = 3.208941e-01;
    nids.clear();
    nids.push_back(176);
    nids.push_back(174);
    nids.push_back(177);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.581235e-01;
    tri3_xyze(1, 0) = 1.268069e-01;
    tri3_xyze(2, 0) = 3.208561e-01;
    tri3_xyze(0, 1) = 8.575415e-01;
    tri3_xyze(1, 1) = 9.509566e-02;
    tri3_xyze(2, 1) = 3.208522e-01;
    tri3_xyze(0, 2) = 8.712137e-01;
    tri3_xyze(1, 2) = 1.107303e-01;
    tri3_xyze(2, 2) = 3.208949e-01;
    nids.clear();
    nids.push_back(145);
    nids.push_back(116);
    nids.push_back(148);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.595878e-01;
    tri3_xyze(1, 0) = 1.901952e-01;
    tri3_xyze(2, 0) = 3.208677e-01;
    tri3_xyze(0, 1) = 8.588142e-01;
    tri3_xyze(1, 1) = 1.585063e-01;
    tri3_xyze(2, 1) = 3.208616e-01;
    tri3_xyze(0, 2) = 8.725808e-01;
    tri3_xyze(1, 2) = 1.740499e-01;
    tri3_xyze(2, 2) = 3.208935e-01;
    nids.clear();
    nids.push_back(203);
    nids.push_back(174);
    nids.push_back(206);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 9.123533e-01;
    tri3_xyze(1, 0) = 1.573435e-01;
    tri3_xyze(2, 0) = 3.209881e-01;
    tri3_xyze(0, 1) = 8.855745e-01;
    tri3_xyze(1, 1) = 1.579358e-01;
    tri3_xyze(2, 1) = 3.209260e-01;
    tri3_xyze(0, 2) = 8.986208e-01;
    tri3_xyze(1, 2) = 1.418496e-01;
    tri3_xyze(2, 2) = 3.209635e-01;
    nids.clear();
    nids.push_back(476);
    nids.push_back(176);
    nids.push_back(477);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.855745e-01;
    tri3_xyze(1, 0) = 1.579358e-01;
    tri3_xyze(2, 0) = 3.209260e-01;
    tri3_xyze(0, 1) = 8.848852e-01;
    tri3_xyze(1, 1) = 1.263158e-01;
    tri3_xyze(2, 1) = 3.209329e-01;
    tri3_xyze(0, 2) = 8.986208e-01;
    tri3_xyze(1, 2) = 1.418496e-01;
    tri3_xyze(2, 2) = 3.209635e-01;
    nids.clear();
    nids.push_back(176);
    nids.push_back(147);
    nids.push_back(477);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.863468e-01;
    tri3_xyze(1, 0) = 1.895624e-01;
    tri3_xyze(2, 0) = 3.209188e-01;
    tri3_xyze(0, 1) = 8.855745e-01;
    tri3_xyze(1, 1) = 1.579358e-01;
    tri3_xyze(2, 1) = 3.209260e-01;
    tri3_xyze(0, 2) = 8.993486e-01;
    tri3_xyze(1, 2) = 1.734376e-01;
    tri3_xyze(2, 2) = 3.209501e-01;
    nids.clear();
    nids.push_back(205);
    nids.push_back(176);
    nids.push_back(481);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.313855e-01;
    tri3_xyze(1, 0) = 1.273181e-01;
    tri3_xyze(2, 0) = 3.207810e-01;
    tri3_xyze(0, 1) = 8.581235e-01;
    tri3_xyze(1, 1) = 1.268069e-01;
    tri3_xyze(2, 1) = 3.208561e-01;
    tri3_xyze(0, 2) = 8.450990e-01;
    tri3_xyze(1, 2) = 1.429322e-01;
    tri3_xyze(2, 2) = 3.208242e-01;
    nids.clear();
    nids.push_back(140);
    nids.push_back(145);
    nids.push_back(175);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.588142e-01;
    tri3_xyze(1, 0) = 1.585063e-01;
    tri3_xyze(2, 0) = 3.208616e-01;
    tri3_xyze(0, 1) = 8.320729e-01;
    tri3_xyze(1, 1) = 1.590976e-01;
    tri3_xyze(2, 1) = 3.207982e-01;
    tri3_xyze(0, 2) = 8.450990e-01;
    tri3_xyze(1, 2) = 1.429322e-01;
    tri3_xyze(2, 2) = 3.208242e-01;
    nids.clear();
    nids.push_back(174);
    nids.push_back(169);
    nids.push_back(175);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.588142e-01;
    tri3_xyze(1, 0) = 1.585063e-01;
    tri3_xyze(2, 0) = 3.208616e-01;
    tri3_xyze(0, 1) = 8.595878e-01;
    tri3_xyze(1, 1) = 1.901952e-01;
    tri3_xyze(2, 1) = 3.208677e-01;
    tri3_xyze(0, 2) = 8.458296e-01;
    tri3_xyze(1, 2) = 1.746618e-01;
    tri3_xyze(2, 2) = 3.208363e-01;
    nids.clear();
    nids.push_back(174);
    nids.push_back(203);
    nids.push_back(204);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.855745e-01;
    tri3_xyze(1, 0) = 1.579358e-01;
    tri3_xyze(2, 0) = 3.209260e-01;
    tri3_xyze(0, 1) = 9.123533e-01;
    tri3_xyze(1, 1) = 1.573435e-01;
    tri3_xyze(2, 1) = 3.209881e-01;
    tri3_xyze(0, 2) = 8.993486e-01;
    tri3_xyze(1, 2) = 1.734376e-01;
    tri3_xyze(2, 2) = 3.209501e-01;
    nids.clear();
    nids.push_back(176);
    nids.push_back(476);
    nids.push_back(481);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.588142e-01;
    tri3_xyze(1, 0) = 1.585063e-01;
    tri3_xyze(2, 0) = 3.208616e-01;
    tri3_xyze(0, 1) = 8.581235e-01;
    tri3_xyze(1, 1) = 1.268069e-01;
    tri3_xyze(2, 1) = 3.208561e-01;
    tri3_xyze(0, 2) = 8.718493e-01;
    tri3_xyze(1, 2) = 1.423912e-01;
    tri3_xyze(2, 2) = 3.208941e-01;
    nids.clear();
    nids.push_back(174);
    nids.push_back(145);
    nids.push_back(177);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.848852e-01;
    tri3_xyze(1, 0) = 1.263158e-01;
    tri3_xyze(2, 0) = 3.209329e-01;
    tri3_xyze(0, 1) = 8.855745e-01;
    tri3_xyze(1, 1) = 1.579358e-01;
    tri3_xyze(2, 1) = 3.209260e-01;
    tri3_xyze(0, 2) = 8.718493e-01;
    tri3_xyze(1, 2) = 1.423912e-01;
    tri3_xyze(2, 2) = 3.208941e-01;
    nids.clear();
    nids.push_back(147);
    nids.push_back(176);
    nids.push_back(177);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.320729e-01;
    tri3_xyze(1, 0) = 1.590976e-01;
    tri3_xyze(2, 0) = 3.207982e-01;
    tri3_xyze(0, 1) = 8.588142e-01;
    tri3_xyze(1, 1) = 1.585063e-01;
    tri3_xyze(2, 1) = 3.208616e-01;
    tri3_xyze(0, 2) = 8.458296e-01;
    tri3_xyze(1, 2) = 1.746618e-01;
    tri3_xyze(2, 2) = 3.208363e-01;
    nids.clear();
    nids.push_back(169);
    nids.push_back(174);
    nids.push_back(204);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.848852e-01;
    tri3_xyze(1, 0) = 1.263158e-01;
    tri3_xyze(2, 0) = 3.209329e-01;
    tri3_xyze(0, 1) = 8.581235e-01;
    tri3_xyze(1, 1) = 1.268069e-01;
    tri3_xyze(2, 1) = 3.208561e-01;
    tri3_xyze(0, 2) = 8.712137e-01;
    tri3_xyze(1, 2) = 1.107303e-01;
    tri3_xyze(2, 2) = 3.208949e-01;
    nids.clear();
    nids.push_back(147);
    nids.push_back(145);
    nids.push_back(148);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.581235e-01;
    tri3_xyze(1, 0) = 1.268069e-01;
    tri3_xyze(2, 0) = 3.208561e-01;
    tri3_xyze(0, 1) = 8.313855e-01;
    tri3_xyze(1, 1) = 1.273181e-01;
    tri3_xyze(2, 1) = 3.207810e-01;
    tri3_xyze(0, 2) = 8.444644e-01;
    tri3_xyze(1, 2) = 1.111820e-01;
    tri3_xyze(2, 2) = 3.208143e-01;
    nids.clear();
    nids.push_back(145);
    nids.push_back(140);
    nids.push_back(146);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.581235e-01;
    tri3_xyze(1, 0) = 1.268069e-01;
    tri3_xyze(2, 0) = 3.208561e-01;
    tri3_xyze(0, 1) = 8.588142e-01;
    tri3_xyze(1, 1) = 1.585063e-01;
    tri3_xyze(2, 1) = 3.208616e-01;
    tri3_xyze(0, 2) = 8.450990e-01;
    tri3_xyze(1, 2) = 1.429322e-01;
    tri3_xyze(2, 2) = 3.208242e-01;
    nids.clear();
    nids.push_back(145);
    nids.push_back(174);
    nids.push_back(175);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.581235e-01;
    tri3_xyze(1, 0) = 1.268069e-01;
    tri3_xyze(2, 0) = 3.208561e-01;
    tri3_xyze(0, 1) = 8.848852e-01;
    tri3_xyze(1, 1) = 1.263158e-01;
    tri3_xyze(2, 1) = 3.209329e-01;
    tri3_xyze(0, 2) = 8.718493e-01;
    tri3_xyze(1, 2) = 1.423912e-01;
    tri3_xyze(2, 2) = 3.208941e-01;
    nids.clear();
    nids.push_back(145);
    nids.push_back(147);
    nids.push_back(177);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.575415e-01;
    tri3_xyze(1, 0) = 9.509566e-02;
    tri3_xyze(2, 0) = 3.208522e-01;
    tri3_xyze(0, 1) = 8.581235e-01;
    tri3_xyze(1, 1) = 1.268069e-01;
    tri3_xyze(2, 1) = 3.208561e-01;
    tri3_xyze(0, 2) = 8.444644e-01;
    tri3_xyze(1, 2) = 1.111820e-01;
    tri3_xyze(2, 2) = 3.208143e-01;
    nids.clear();
    nids.push_back(116);
    nids.push_back(145);
    nids.push_back(146);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.855745e-01;
    tri3_xyze(1, 0) = 1.579358e-01;
    tri3_xyze(2, 0) = 3.209260e-01;
    tri3_xyze(0, 1) = 8.863468e-01;
    tri3_xyze(1, 1) = 1.895624e-01;
    tri3_xyze(2, 1) = 3.209188e-01;
    tri3_xyze(0, 2) = 8.725808e-01;
    tri3_xyze(1, 2) = 1.740499e-01;
    tri3_xyze(2, 2) = 3.208935e-01;
    nids.clear();
    nids.push_back(176);
    nids.push_back(205);
    nids.push_back(206);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 8.848852e-01;
    tri3_xyze(1, 0) = 1.263158e-01;
    tri3_xyze(2, 0) = 3.209329e-01;
    tri3_xyze(0, 1) = 9.116703e-01;
    tri3_xyze(1, 1) = 1.258034e-01;
    tri3_xyze(2, 1) = 3.210069e-01;
    tri3_xyze(0, 2) = 8.986208e-01;
    tri3_xyze(1, 2) = 1.418496e-01;
    tri3_xyze(2, 2) = 3.209635e-01;
    nids.clear();
    nids.push_back(147);
    nids.push_back(472);
    nids.push_back(477);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

  hex8_xyze(0, 0) = 8.936170e-01;
  hex8_xyze(1, 0) = 1.684211e-01;
  hex8_xyze(2, 0) = 3.437500e-01;
  hex8_xyze(0, 1) = 8.936170e-01;
  hex8_xyze(1, 1) = 1.263158e-01;
  hex8_xyze(2, 1) = 3.437500e-01;
  hex8_xyze(0, 2) = 8.510638e-01;
  hex8_xyze(1, 2) = 1.263158e-01;
  hex8_xyze(2, 2) = 3.437500e-01;
  hex8_xyze(0, 3) = 8.510638e-01;
  hex8_xyze(1, 3) = 1.684211e-01;
  hex8_xyze(2, 3) = 3.437500e-01;
  hex8_xyze(0, 4) = 8.936170e-01;
  hex8_xyze(1, 4) = 1.684211e-01;
  hex8_xyze(2, 4) = 3.125000e-01;
  hex8_xyze(0, 5) = 8.936170e-01;
  hex8_xyze(1, 5) = 1.263158e-01;
  hex8_xyze(2, 5) = 3.125000e-01;
  hex8_xyze(0, 6) = 8.510638e-01;
  hex8_xyze(1, 6) = 1.263158e-01;
  hex8_xyze(2, 6) = 3.125000e-01;
  hex8_xyze(0, 7) = 8.510638e-01;
  hex8_xyze(1, 7) = 1.684211e-01;
  hex8_xyze(2, 7) = 3.125000e-01;

  nids.clear();
  for (int i = 0; i < 8; ++i) nids.push_back(i);

  intersection.add_element(1, nids, hex8_xyze, Core::FE::CellType::hex8);

  intersection.cut_test_cut(true, Cut::VCellGaussPts_DirectDivergence);
}
