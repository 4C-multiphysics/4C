// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_combintersection.hpp"
#include "4C_cut_levelsetintersection.hpp"
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

void test_generated_1858()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2707);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = -0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2687);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-84);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2707);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = -0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 2) = 0.0485458;
    tri3_xyze(1, 2) = -0.1172;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-85);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = -0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = -0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2689);
    tri3_xyze(0, 2) = 0.0485458;
    tri3_xyze(1, 2) = -0.1172;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-85);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = -0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2687);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2707);
    tri3_xyze(0, 2) = 0.0485458;
    tri3_xyze(1, 2) = -0.1172;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-85);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = -0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = -0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 2) = 0.0550999;
    tri3_xyze(1, 2) = -0.133023;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-86);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = -0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 1) = 0.0388229;
    tri3_xyze(1, 1) = -0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2691);
    tri3_xyze(0, 2) = 0.0550999;
    tri3_xyze(1, 2) = -0.133023;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-86);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0388229;
    tri3_xyze(1, 0) = -0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2691);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = -0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2689);
    tri3_xyze(0, 2) = 0.0550999;
    tri3_xyze(1, 2) = -0.133023;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-86);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = -0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2689);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = -0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 2) = 0.0550999;
    tri3_xyze(1, 2) = -0.133023;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-86);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = -0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = -0.121634;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2713);
    tri3_xyze(0, 2) = 0.0550999;
    tri3_xyze(1, 2) = -0.133023;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-87);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0388229;
    tri3_xyze(1, 0) = -0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2691);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = -0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 2) = 0.0550999;
    tri3_xyze(1, 2) = -0.133023;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-87);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816361;
    tri3_xyze(1, 0) = -0.0816361;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2727);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-95);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = -0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-95);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = -0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2707);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-95);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2707);
    tri3_xyze(0, 1) = 0.0816361;
    tri3_xyze(1, 1) = -0.0816361;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2727);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-95);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 1) = 0.106066;
    tri3_xyze(1, 1) = -0.106066;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-96);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.106066;
    tri3_xyze(1, 0) = -0.106066;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = -0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-96);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = -0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = -0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-96);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = -0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2709);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-96);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.106066;
    tri3_xyze(1, 0) = -0.106066;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-97);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = -0.121634;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2713);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = -0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-97);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = -0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2711);
    tri3_xyze(0, 1) = 0.106066;
    tri3_xyze(1, 1) = -0.106066;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-97);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 1) = 0.0816361;
    tri3_xyze(1, 1) = -0.0816361;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2727);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816361;
    tri3_xyze(1, 0) = -0.0816361;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2727);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 1) = 0.129904;
    tri3_xyze(1, 1) = -0.075;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.129904;
    tri3_xyze(1, 0) = -0.075;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 1) = 0.106066;
    tri3_xyze(1, 1) = -0.106066;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.106066;
    tri3_xyze(1, 0) = -0.106066;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2729);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.129904;
    tri3_xyze(1, 0) = -0.075;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 1) = 0.106066;
    tri3_xyze(1, 1) = -0.106066;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.106066;
    tri3_xyze(1, 0) = -0.106066;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 1) = 0.129904;
    tri3_xyze(1, 1) = -0.075;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2767);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-114);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.135665;
    tri3_xyze(1, 0) = -0.0363514;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2769);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 2) = 0.1172;
    tri3_xyze(1, 2) = -0.0485458;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-115);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 2) = 0.1172;
    tri3_xyze(1, 2) = -0.0485458;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-115);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2767);
    tri3_xyze(0, 2) = 0.1172;
    tri3_xyze(1, 2) = -0.0485458;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-115);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.135665;
    tri3_xyze(1, 0) = -0.0363514;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2769);
    tri3_xyze(0, 1) = 0.144889;
    tri3_xyze(1, 1) = -0.0388229;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2771);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-116);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.144889;
    tri3_xyze(1, 0) = -0.0388229;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2771);
    tri3_xyze(0, 1) = 0.129904;
    tri3_xyze(1, 1) = -0.075;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-116);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.129904;
    tri3_xyze(1, 0) = -0.075;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-116);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2749);
    tri3_xyze(0, 1) = 0.135665;
    tri3_xyze(1, 1) = -0.0363514;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2769);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-116);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 1) = 0.129904;
    tri3_xyze(1, 1) = -0.075;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-117);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.129904;
    tri3_xyze(1, 0) = -0.075;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 1) = 0.144889;
    tri3_xyze(1, 1) = -0.0388229;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2771);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-117);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.2;
    hex8_xyze(1, 0) = -0.15;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1820);
    hex8_xyze(0, 1) = 0.2;
    hex8_xyze(1, 1) = -0.1;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1822);
    hex8_xyze(0, 2) = 0.15;
    hex8_xyze(1, 2) = -0.1;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1840);
    hex8_xyze(0, 3) = 0.15;
    hex8_xyze(1, 3) = -0.15;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1839);
    hex8_xyze(0, 4) = 0.2;
    hex8_xyze(1, 4) = -0.15;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1941);
    hex8_xyze(0, 5) = 0.2;
    hex8_xyze(1, 5) = -0.1;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1943);
    hex8_xyze(0, 6) = 0.15;
    hex8_xyze(1, 6) = -0.1;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1961);
    hex8_xyze(0, 7) = 0.15;
    hex8_xyze(1, 7) = -0.15;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1960);

    intersection.add_element(1848, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.15;
    hex8_xyze(1, 0) = -0.2;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1837);
    hex8_xyze(0, 1) = 0.15;
    hex8_xyze(1, 1) = -0.15;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1839);
    hex8_xyze(0, 2) = 0.1;
    hex8_xyze(1, 2) = -0.15;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1850);
    hex8_xyze(0, 3) = 0.1;
    hex8_xyze(1, 3) = -0.2;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1848);
    hex8_xyze(0, 4) = 0.15;
    hex8_xyze(1, 4) = -0.2;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1958);
    hex8_xyze(0, 5) = 0.15;
    hex8_xyze(1, 5) = -0.15;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1960);
    hex8_xyze(0, 6) = 0.1;
    hex8_xyze(1, 6) = -0.15;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1971);
    hex8_xyze(0, 7) = 0.1;
    hex8_xyze(1, 7) = -0.2;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1969);

    intersection.add_element(1857, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.15;
    hex8_xyze(1, 0) = -0.15;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1839);
    hex8_xyze(0, 1) = 0.15;
    hex8_xyze(1, 1) = -0.1;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1840);
    hex8_xyze(0, 2) = 0.1;
    hex8_xyze(1, 2) = -0.1;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1851);
    hex8_xyze(0, 3) = 0.1;
    hex8_xyze(1, 3) = -0.15;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1850);
    hex8_xyze(0, 4) = 0.15;
    hex8_xyze(1, 4) = -0.15;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1960);
    hex8_xyze(0, 5) = 0.15;
    hex8_xyze(1, 5) = -0.1;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1961);
    hex8_xyze(0, 6) = 0.1;
    hex8_xyze(1, 6) = -0.1;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1972);
    hex8_xyze(0, 7) = 0.1;
    hex8_xyze(1, 7) = -0.15;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1971);

    intersection.add_element(1858, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.15;
    hex8_xyze(1, 0) = -0.1;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1840);
    hex8_xyze(0, 1) = 0.15;
    hex8_xyze(1, 1) = -0.05;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1841);
    hex8_xyze(0, 2) = 0.1;
    hex8_xyze(1, 2) = -0.05;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1852);
    hex8_xyze(0, 3) = 0.1;
    hex8_xyze(1, 3) = -0.1;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1851);
    hex8_xyze(0, 4) = 0.15;
    hex8_xyze(1, 4) = -0.1;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1961);
    hex8_xyze(0, 5) = 0.15;
    hex8_xyze(1, 5) = -0.05;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1962);
    hex8_xyze(0, 6) = 0.1;
    hex8_xyze(1, 6) = -0.05;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1973);
    hex8_xyze(0, 7) = 0.1;
    hex8_xyze(1, 7) = -0.1;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1972);

    intersection.add_element(1859, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.1;
    hex8_xyze(1, 0) = -0.15;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1850);
    hex8_xyze(0, 1) = 0.1;
    hex8_xyze(1, 1) = -0.1;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1851);
    hex8_xyze(0, 2) = 0.05;
    hex8_xyze(1, 2) = -0.1;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1862);
    hex8_xyze(0, 3) = 0.05;
    hex8_xyze(1, 3) = -0.15;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1861);
    hex8_xyze(0, 4) = 0.1;
    hex8_xyze(1, 4) = -0.15;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1971);
    hex8_xyze(0, 5) = 0.1;
    hex8_xyze(1, 5) = -0.1;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1972);
    hex8_xyze(0, 6) = 0.05;
    hex8_xyze(1, 6) = -0.1;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1983);
    hex8_xyze(0, 7) = 0.05;
    hex8_xyze(1, 7) = -0.15;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1982);

    intersection.add_element(1868, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.15;
    hex8_xyze(1, 0) = -0.15;
    hex8_xyze(2, 0) = 0.8;
    nids.push_back(1960);
    hex8_xyze(0, 1) = 0.15;
    hex8_xyze(1, 1) = -0.1;
    hex8_xyze(2, 1) = 0.8;
    nids.push_back(1961);
    hex8_xyze(0, 2) = 0.1;
    hex8_xyze(1, 2) = -0.1;
    hex8_xyze(2, 2) = 0.8;
    nids.push_back(1972);
    hex8_xyze(0, 3) = 0.1;
    hex8_xyze(1, 3) = -0.15;
    hex8_xyze(2, 3) = 0.8;
    nids.push_back(1971);
    hex8_xyze(0, 4) = 0.15;
    hex8_xyze(1, 4) = -0.15;
    hex8_xyze(2, 4) = 0.85;
    nids.push_back(2081);
    hex8_xyze(0, 5) = 0.15;
    hex8_xyze(1, 5) = -0.1;
    hex8_xyze(2, 5) = 0.85;
    nids.push_back(2082);
    hex8_xyze(0, 6) = 0.1;
    hex8_xyze(1, 6) = -0.1;
    hex8_xyze(2, 6) = 0.85;
    nids.push_back(2093);
    hex8_xyze(0, 7) = 0.1;
    hex8_xyze(1, 7) = -0.15;
    hex8_xyze(2, 7) = 0.85;
    nids.push_back(2092);

    intersection.add_element(1958, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
