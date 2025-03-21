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

void test_generated_6920()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0503943;
    tri3_xyze(1, 1) = -1.96873e-17;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(2546);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0503943;
    tri3_xyze(1, 0) = -1.96873e-17;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(2546);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 1) = -0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 2) = -0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-5);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.0515708;
    tri3_xyze(1, 1) = -3.90641e-17;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(2551);
    tri3_xyze(0, 2) = -0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-5);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0503943;
    tri3_xyze(1, 0) = -1.96873e-17;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(2546);
    tri3_xyze(0, 1) = -0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-5);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 1) = -0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 2) = -0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 1) = -0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 2) = -0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0503943;
    tri3_xyze(1, 1) = -1.96873e-17;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(2546);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-7);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5791);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-7);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 1) = -0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 2) = -0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(2737);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-3);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-3);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0503943;
    tri3_xyze(1, 1) = 1.96873e-17;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(2739);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-3);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(3139);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 2) = -0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-4);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-4);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(2737);
    tri3_xyze(0, 2) = -0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-4);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0503943;
    tri3_xyze(1, 0) = 1.96873e-17;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(2739);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-8);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5791);
    tri3_xyze(0, 2) = -0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-8);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(2553);
    tri3_xyze(0, 2) = -0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-10);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515708;
    tri3_xyze(1, 0) = -3.90641e-17;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(2551);
    tri3_xyze(0, 1) = -0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-10);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 1) = -0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 2) = -0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 1) = -0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(2553);
    tri3_xyze(0, 2) = -0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(2553);
    tri3_xyze(0, 1) = -0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 2) = -0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 1) = -0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 2) = -0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-14);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 1) = -0.0542702;
    tri3_xyze(1, 1) = -0.0145417;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(2557);
    tri3_xyze(0, 2) = -0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-14);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(2553);
    tri3_xyze(0, 1) = -0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 2) = -0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-14);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3051);
    tri3_xyze(0, 2) = -0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-17);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0542702;
    tri3_xyze(1, 0) = -0.0145417;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(2557);
    tri3_xyze(0, 1) = -0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 2) = -0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-17);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 1) = -0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 2) = -0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 1) = -0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 2) = -0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 2) = -0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 2) = -0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(3389);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 2) = -0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 2) = -0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3041);
    tri3_xyze(0, 1) = -0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(3139);
    tri3_xyze(0, 2) = -0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 1) = -0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 2) = -0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 1) = -0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 2) = -0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 1) = -0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 2) = -0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3042);
    tri3_xyze(0, 1) = -0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 2) = -0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 1) = -0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 2) = -0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 1) = -0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 2) = -0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 1) = -0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 2) = -0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3045);
    tri3_xyze(0, 1) = -0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 2) = -0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 1) = -0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 2) = -0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-155);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 1) = -0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 2) = -0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-155);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 1) = -0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 2) = -0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-155);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3047);
    tri3_xyze(0, 1) = -0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 2) = -0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-155);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 2) = -0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-156);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3051);
    tri3_xyze(0, 2) = -0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-156);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3051);
    tri3_xyze(0, 1) = -0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 2) = -0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-156);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3049);
    tri3_xyze(0, 1) = -0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 2) = -0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-156);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 1) = -0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 2) = -0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-157);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 1) = -0.0550373;
    tri3_xyze(1, 1) = -0.0317758;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3053);
    tri3_xyze(0, 2) = -0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-157);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3053);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3051);
    tri3_xyze(0, 2) = -0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-157);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3051);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 2) = -0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-157);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 1) = -0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 2) = -0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-158);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 1) = -0.0590013;
    tri3_xyze(1, 1) = -0.0340644;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(3055);
    tri3_xyze(0, 2) = -0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-158);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3053);
    tri3_xyze(0, 1) = -0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 2) = -0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-158);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 1) = -0.0517663;
    tri3_xyze(1, 1) = -0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(3307);
    tri3_xyze(0, 2) = -0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-159);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0590013;
    tri3_xyze(1, 0) = -0.0340644;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(3055);
    tri3_xyze(0, 1) = -0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 2) = -0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-159);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 1) = -0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 2) = -0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-201);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 1) = -0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 2) = -0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-201);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 2) = -0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-201);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 2) = -0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-201);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(3639);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 2) = -0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-202);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 2) = -0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-202);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3291);
    tri3_xyze(0, 1) = -0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(3389);
    tri3_xyze(0, 2) = -0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-202);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 1) = -0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 2) = -0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-203);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 1) = -0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 2) = -0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-203);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 1) = -0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 2) = -0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-203);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3292);
    tri3_xyze(0, 1) = -0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 2) = -0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-203);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 1) = -0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 2) = -0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-204);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 1) = -0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 2) = -0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-204);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 1) = -0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 2) = -0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-204);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3295);
    tri3_xyze(0, 1) = -0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 2) = -0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-204);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 1) = -0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 2) = -0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-205);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 1) = -0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 2) = -0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-205);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 1) = -0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 2) = -0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-205);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3297);
    tri3_xyze(0, 1) = -0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 2) = -0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-205);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3551);
    tri3_xyze(0, 2) = -0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-206);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3551);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 2) = -0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-206);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 1) = -0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 2) = -0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-206);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3299);
    tri3_xyze(0, 1) = -0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 2) = -0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-206);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3551);
    tri3_xyze(0, 1) = -0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3553);
    tri3_xyze(0, 2) = -0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-207);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0317758;
    tri3_xyze(1, 0) = -0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3553);
    tri3_xyze(0, 1) = -0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 2) = -0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-207);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 2) = -0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-207);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3301);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3551);
    tri3_xyze(0, 2) = -0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-207);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0340644;
    tri3_xyze(1, 0) = -0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(3555);
    tri3_xyze(0, 1) = -0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 2) = -0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-208);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 1) = -0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 2) = -0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-208);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3303);
    tri3_xyze(0, 1) = -0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3553);
    tri3_xyze(0, 2) = -0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-208);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0517663;
    tri3_xyze(1, 0) = -0.0517663;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(3307);
    tri3_xyze(0, 1) = -0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 2) = -0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-209);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(3305);
    tri3_xyze(0, 1) = -0.0340644;
    tri3_xyze(1, 1) = -0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(3555);
    tri3_xyze(0, 2) = -0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-209);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 1) = -0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 2) = -0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-251);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 1) = -0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 2) = -0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-251);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 2) = -0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-251);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 2) = -0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-251);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(3889);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 2) = -0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-252);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 2) = -0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-252);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3541);
    tri3_xyze(0, 1) = -0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(3639);
    tri3_xyze(0, 2) = -0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-252);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 1) = -0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 2) = -0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-253);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 1) = -0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 2) = -0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-253);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 1) = -0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 2) = -0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-253);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3542);
    tri3_xyze(0, 1) = -0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 2) = -0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-253);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 1) = -0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3797);
    tri3_xyze(0, 2) = -0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-254);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3797);
    tri3_xyze(0, 1) = -0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 2) = -0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-254);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 1) = -0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 2) = -0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-254);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3545);
    tri3_xyze(0, 1) = -0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 2) = -0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-254);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0145417;
    tri3_xyze(1, 0) = -0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3799);
    tri3_xyze(0, 1) = -0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 2) = -0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-255);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 1) = -0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 2) = -0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-255);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3547);
    tri3_xyze(0, 1) = -0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3797);
    tri3_xyze(0, 2) = -0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-255);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3551);
    tri3_xyze(0, 1) = -0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 2) = -0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-256);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3549);
    tri3_xyze(0, 1) = -0.0145417;
    tri3_xyze(1, 1) = -0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3799);
    tri3_xyze(0, 2) = -0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-256);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 1) = 1.02547e-15;
    tri3_xyze(1, 1) = -0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-301);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02547e-15;
    tri3_xyze(1, 0) = -0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 1) = -0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-301);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-301);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-301);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02547e-15;
    tri3_xyze(1, 0) = -0.0503943;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4139);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-302);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-302);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3791);
    tri3_xyze(0, 1) = -0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(3889);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-302);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02547e-15;
    tri3_xyze(1, 0) = -0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 1) = 1.02606e-15;
    tri3_xyze(1, 1) = -0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 2) = -0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-303);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02606e-15;
    tri3_xyze(1, 0) = -0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 1) = -0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 2) = -0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-303);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 1) = -0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 2) = -0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-303);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(3792);
    tri3_xyze(0, 1) = 1.02547e-15;
    tri3_xyze(1, 1) = -0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 2) = -0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-303);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02606e-15;
    tri3_xyze(1, 0) = -0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 1) = 1.01516e-15;
    tri3_xyze(1, 1) = -0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 2) = -0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-304);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01516e-15;
    tri3_xyze(1, 0) = -0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 1) = -0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(3797);
    tri3_xyze(0, 2) = -0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-304);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3797);
    tri3_xyze(0, 1) = -0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 2) = -0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-304);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(3795);
    tri3_xyze(0, 1) = 1.02606e-15;
    tri3_xyze(1, 1) = -0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 2) = -0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-304);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01516e-15;
    tri3_xyze(1, 0) = -0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 1) = 1.02839e-15;
    tri3_xyze(1, 1) = -0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 2) = -0.00709784;
    tri3_xyze(1, 2) = -0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-305);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02839e-15;
    tri3_xyze(1, 0) = -0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 1) = -0.0145417;
    tri3_xyze(1, 1) = -0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(3799);
    tri3_xyze(0, 2) = -0.00709784;
    tri3_xyze(1, 2) = -0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-305);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(3797);
    tri3_xyze(0, 1) = 1.01516e-15;
    tri3_xyze(1, 1) = -0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 2) = -0.00709784;
    tri3_xyze(1, 2) = -0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-305);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02839e-15;
    tri3_xyze(1, 0) = -0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 2) = -0.00748853;
    tri3_xyze(1, 2) = -0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-306);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3801);
    tri3_xyze(0, 2) = -0.00748853;
    tri3_xyze(1, 2) = -0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-306);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0145417;
    tri3_xyze(1, 0) = -0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(3799);
    tri3_xyze(0, 1) = 1.02839e-15;
    tri3_xyze(1, 1) = -0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 2) = -0.00748853;
    tri3_xyze(1, 2) = -0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-306);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 1) = 1.03211e-15;
    tri3_xyze(1, 1) = -0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 2) = -0.0079652;
    tri3_xyze(1, 2) = -0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-307);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03211e-15;
    tri3_xyze(1, 0) = -0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 1) = -0.0164484;
    tri3_xyze(1, 1) = -0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(3803);
    tri3_xyze(0, 2) = -0.0079652;
    tri3_xyze(1, 2) = -0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-307);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3801);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 2) = -0.0079652;
    tri3_xyze(1, 2) = -0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-307);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03211e-15;
    tri3_xyze(1, 0) = -0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 1) = 1.0193e-15;
    tri3_xyze(1, 1) = -0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 2) = -0.00852035;
    tri3_xyze(1, 2) = -0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-308);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0193e-15;
    tri3_xyze(1, 0) = -0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 1) = -0.017633;
    tri3_xyze(1, 1) = -0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(3805);
    tri3_xyze(0, 2) = -0.00852035;
    tri3_xyze(1, 2) = -0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-308);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0164484;
    tri3_xyze(1, 0) = -0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(3803);
    tri3_xyze(0, 1) = 1.03211e-15;
    tri3_xyze(1, 1) = -0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 2) = -0.00852035;
    tri3_xyze(1, 2) = -0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-308);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0193e-15;
    tri3_xyze(1, 0) = -0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 1) = 1.03699e-15;
    tri3_xyze(1, 1) = -0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 2) = -0.00914521;
    tri3_xyze(1, 2) = -0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-309);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03699e-15;
    tri3_xyze(1, 0) = -0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 1) = -0.0189478;
    tri3_xyze(1, 1) = -0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(3807);
    tri3_xyze(0, 2) = -0.00914521;
    tri3_xyze(1, 2) = -0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-309);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.017633;
    tri3_xyze(1, 0) = -0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(3805);
    tri3_xyze(0, 1) = 1.0193e-15;
    tri3_xyze(1, 1) = -0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 2) = -0.00914521;
    tri3_xyze(1, 2) = -0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-309);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03699e-15;
    tri3_xyze(1, 0) = -0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 1) = 1.03977e-15;
    tri3_xyze(1, 1) = -0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 2) = -0.00982993;
    tri3_xyze(1, 2) = -0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-310);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03977e-15;
    tri3_xyze(1, 0) = -0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 1) = -0.0203719;
    tri3_xyze(1, 1) = -0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(3809);
    tri3_xyze(0, 2) = -0.00982993;
    tri3_xyze(1, 2) = -0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-310);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0189478;
    tri3_xyze(1, 0) = -0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(3807);
    tri3_xyze(0, 1) = 1.03699e-15;
    tri3_xyze(1, 1) = -0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 2) = -0.00982993;
    tri3_xyze(1, 2) = -0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-310);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03977e-15;
    tri3_xyze(1, 0) = -0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 2) = -0.0105637;
    tri3_xyze(1, 2) = -0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-311);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 1) = -0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(3811);
    tri3_xyze(0, 2) = -0.0105637;
    tri3_xyze(1, 2) = -0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-311);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0203719;
    tri3_xyze(1, 0) = -0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(3809);
    tri3_xyze(0, 1) = 1.03977e-15;
    tri3_xyze(1, 1) = -0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 2) = -0.0105637;
    tri3_xyze(1, 2) = -0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-311);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 1) = 1.06592e-15;
    tri3_xyze(1, 1) = -0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 2) = -0.011335;
    tri3_xyze(1, 2) = -0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-312);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06592e-15;
    tri3_xyze(1, 0) = -0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 1) = -0.023457;
    tri3_xyze(1, 1) = -0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(3813);
    tri3_xyze(0, 2) = -0.011335;
    tri3_xyze(1, 2) = -0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-312);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(3811);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 2) = -0.011335;
    tri3_xyze(1, 2) = -0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-312);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06592e-15;
    tri3_xyze(1, 0) = -0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 1) = 1.02744e-15;
    tri3_xyze(1, 1) = -0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 2) = -0.0121316;
    tri3_xyze(1, 2) = -0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-313);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02744e-15;
    tri3_xyze(1, 0) = -0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 1) = -0.0250693;
    tri3_xyze(1, 1) = -0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(3815);
    tri3_xyze(0, 2) = -0.0121316;
    tri3_xyze(1, 2) = -0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-313);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.023457;
    tri3_xyze(1, 0) = -0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(3813);
    tri3_xyze(0, 1) = 1.06592e-15;
    tri3_xyze(1, 1) = -0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 2) = -0.0121316;
    tri3_xyze(1, 2) = -0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-313);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02744e-15;
    tri3_xyze(1, 0) = -0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 1) = 1.05212e-15;
    tri3_xyze(1, 1) = -0.10314;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4067);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-314);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0250693;
    tri3_xyze(1, 0) = -0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(3815);
    tri3_xyze(0, 1) = 1.02744e-15;
    tri3_xyze(1, 1) = -0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-314);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-351);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 1) = 1.02547e-15;
    tri3_xyze(1, 1) = -0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-351);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02547e-15;
    tri3_xyze(1, 0) = -0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-351);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-351);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-352);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-352);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4041);
    tri3_xyze(0, 1) = 1.02547e-15;
    tri3_xyze(1, 1) = -0.0503943;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4139);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-352);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02547e-15;
    tri3_xyze(1, 0) = -0.0503943;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4139);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = -0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-352);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-353);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 1) = 1.02606e-15;
    tri3_xyze(1, 1) = -0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-353);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02606e-15;
    tri3_xyze(1, 0) = -0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 1) = 1.02547e-15;
    tri3_xyze(1, 1) = -0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-353);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02547e-15;
    tri3_xyze(1, 0) = -0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4042);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-353);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-354);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 1) = 1.01516e-15;
    tri3_xyze(1, 1) = -0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-354);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01516e-15;
    tri3_xyze(1, 0) = -0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 1) = 1.02606e-15;
    tri3_xyze(1, 1) = -0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-354);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02606e-15;
    tri3_xyze(1, 0) = -0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4045);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-354);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = -0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = -0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-355);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = -0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 1) = 1.02839e-15;
    tri3_xyze(1, 1) = -0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = -0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-355);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02839e-15;
    tri3_xyze(1, 0) = -0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 1) = 1.01516e-15;
    tri3_xyze(1, 1) = -0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = -0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-355);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01516e-15;
    tri3_xyze(1, 0) = -0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4047);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = -0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-355);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = -0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = -0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-356);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = -0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-356);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 1) = 1.02839e-15;
    tri3_xyze(1, 1) = -0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = -0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-356);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02839e-15;
    tri3_xyze(1, 0) = -0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4049);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = -0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = -0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-356);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = -0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = -0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-357);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = -0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 1) = 1.03211e-15;
    tri3_xyze(1, 1) = -0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = -0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-357);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03211e-15;
    tri3_xyze(1, 0) = -0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = -0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-357);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4051);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = -0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-357);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = -0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = -0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = -0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-358);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = -0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 1) = 1.0193e-15;
    tri3_xyze(1, 1) = -0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = -0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-358);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0193e-15;
    tri3_xyze(1, 0) = -0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 1) = 1.03211e-15;
    tri3_xyze(1, 1) = -0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = -0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-358);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03211e-15;
    tri3_xyze(1, 0) = -0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4053);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = -0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = -0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-358);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = -0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = -0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = -0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-359);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = -0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 1) = 1.03699e-15;
    tri3_xyze(1, 1) = -0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = -0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-359);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03699e-15;
    tri3_xyze(1, 0) = -0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 1) = 1.0193e-15;
    tri3_xyze(1, 1) = -0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = -0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-359);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0193e-15;
    tri3_xyze(1, 0) = -0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4055);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = -0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = -0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-359);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = -0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = -0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = -0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-360);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = -0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 1) = 1.03977e-15;
    tri3_xyze(1, 1) = -0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = -0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-360);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03977e-15;
    tri3_xyze(1, 0) = -0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 1) = 1.03699e-15;
    tri3_xyze(1, 1) = -0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = -0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-360);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03699e-15;
    tri3_xyze(1, 0) = -0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4057);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = -0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = -0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-360);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = -0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = -0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-361);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = -0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-361);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 1) = 1.03977e-15;
    tri3_xyze(1, 1) = -0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = -0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-361);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03977e-15;
    tri3_xyze(1, 0) = -0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4059);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = -0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = -0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-361);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = -0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = -0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-362);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = -0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 1) = 1.06592e-15;
    tri3_xyze(1, 1) = -0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = -0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-362);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06592e-15;
    tri3_xyze(1, 0) = -0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = -0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-362);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4061);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = -0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-362);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = -0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = -0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = -0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-363);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = -0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 1) = 1.02744e-15;
    tri3_xyze(1, 1) = -0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = -0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-363);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02744e-15;
    tri3_xyze(1, 0) = -0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 1) = 1.06592e-15;
    tri3_xyze(1, 1) = -0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = -0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-363);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06592e-15;
    tri3_xyze(1, 0) = -0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4063);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = -0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = -0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-363);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = -0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = -0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-364);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = -0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 1) = 1.05212e-15;
    tri3_xyze(1, 1) = -0.10314;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4067);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-364);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05212e-15;
    tri3_xyze(1, 0) = -0.10314;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4067);
    tri3_xyze(0, 1) = 1.02744e-15;
    tri3_xyze(1, 1) = -0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-364);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02744e-15;
    tri3_xyze(1, 0) = -0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4065);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = -0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-364);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = -0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = -0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4319);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = -0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-365);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05212e-15;
    tri3_xyze(1, 0) = -0.10314;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4067);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = -0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = -0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-365);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4385);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-399);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 1) = 1.02606e-15;
    tri3_xyze(1, 1) = -0.0515708;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4137);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = -0.0516459;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-399);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-400);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 1) = 1.02547e-15;
    tri3_xyze(1, 1) = -0.0503943;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4139);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-400);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.02606e-15;
    tri3_xyze(1, 0) = -0.0515708;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4137);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = -0.050114;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-400);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-401);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-401);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-401);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-401);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-402);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-402);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4291);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-402);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = -0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-402);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-403);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-403);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-403);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4292);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-403);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-404);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-404);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-404);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4295);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-404);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-405);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = -0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-405);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = -0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-405);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4297);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-405);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-406);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-406);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = -0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-406);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = -0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4299);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-406);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = -0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-407);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = -0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = -0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = -0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-407);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = -0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = -0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-407);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4301);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = -0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-407);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = -0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = -0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = -0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-408);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = -0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = -0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = -0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-408);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = -0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = -0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = -0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-408);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = -0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4303);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = -0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-408);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = -0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = -0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = -0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-409);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = -0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = -0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = -0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-409);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = -0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = -0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = -0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-409);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = -0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4305);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = -0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = -0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-409);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = -0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = -0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = -0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-410);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = -0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = -0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = -0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-410);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = -0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = -0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = -0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-410);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = -0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4307);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = -0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = -0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-410);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = -0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = -0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-411);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = -0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-411);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = -0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = -0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-411);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = -0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4309);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = -0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = -0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-411);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = -0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = -0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-412);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = -0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = -0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = -0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-412);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = -0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = -0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-412);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4311);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = -0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-412);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = -0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = -0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = -0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-413);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = -0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = -0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = -0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-413);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = -0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = -0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = -0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-413);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = -0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4313);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = -0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = -0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-413);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = -0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 1) = 0.0515698;
    tri3_xyze(1, 1) = -0.0893214;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4567);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-414);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515698;
    tri3_xyze(1, 0) = -0.0893214;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4567);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = -0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-414);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = -0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = -0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-414);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = -0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4315);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = -0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-414);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515698;
    tri3_xyze(1, 0) = -0.0893214;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4567);
    tri3_xyze(0, 1) = 0.0546845;
    tri3_xyze(1, 1) = -0.0947164;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4569);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = -0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-415);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0546845;
    tri3_xyze(1, 0) = -0.0947164;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4569);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = -0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4319);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = -0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-415);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = -0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4319);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = -0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = -0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-415);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = -0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4317);
    tri3_xyze(0, 1) = 0.0515698;
    tri3_xyze(1, 1) = -0.0893214;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4567);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = -0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-415);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0546845;
    tri3_xyze(1, 0) = -0.0947164;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4569);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4571);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = -0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-416);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = -0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4319);
    tri3_xyze(0, 1) = 0.0546845;
    tri3_xyze(1, 1) = -0.0947164;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4569);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = -0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-416);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(4631);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-447);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = -0.0542702;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4383);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = -0.0530047;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-447);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-448);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4385);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-448);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = -0.0542702;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4383);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = -0.0502394;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-448);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = -0.0516878;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4385);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = -0.0516878;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4385);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = -0.0481263;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-450);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = -0.0486771;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-450);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = -0.0486771;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4389);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = -0.0498136;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-450);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = -0.0498136;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4387);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = -0.0466988;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-450);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-451);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-451);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-451);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-451);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-452);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-452);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4541);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-452);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = -0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-452);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-453);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-453);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-453);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4542);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-453);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-454);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-454);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-454);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4545);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-454);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-455);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-455);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-455);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4547);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-455);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-456);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-456);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-456);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4549);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-456);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-457);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-457);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = -0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-457);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4551);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-457);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-458);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = -0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-458);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = -0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-458);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = -0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4553);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-458);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = -0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-459);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0517663;
    tri3_xyze(1, 0) = -0.0517663;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = -0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-459);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = -0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = -0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-459);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = -0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4555);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-459);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0517663;
    tri3_xyze(1, 0) = -0.0517663;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 1) = 0.0556571;
    tri3_xyze(1, 1) = -0.0556571;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = -0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-460);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0556571;
    tri3_xyze(1, 0) = -0.0556571;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = -0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = -0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-460);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = -0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = -0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = -0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-460);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = -0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4557);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = -0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = -0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-460);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0556571;
    tri3_xyze(1, 0) = -0.0556571;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4811);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = -0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-461);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4811);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = -0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-461);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = -0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = -0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-461);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = -0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4559);
    tri3_xyze(0, 1) = 0.0556571;
    tri3_xyze(1, 1) = -0.0556571;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = -0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-461);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0640857;
    tri3_xyze(1, 0) = -0.0640857;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4813);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = -0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 2) = 0.0528653;
    tri3_xyze(1, 2) = -0.0688954;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-462);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = -0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 2) = 0.0528653;
    tri3_xyze(1, 2) = -0.0688954;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-462);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4561);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4811);
    tri3_xyze(0, 2) = 0.0528653;
    tri3_xyze(1, 2) = -0.0688954;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-462);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0684907;
    tri3_xyze(1, 0) = -0.0684907;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4815);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = -0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 2) = 0.0565805;
    tri3_xyze(1, 2) = -0.0737372;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-463);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = -0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = -0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 2) = 0.0565805;
    tri3_xyze(1, 2) = -0.0737372;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-463);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = -0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4563);
    tri3_xyze(0, 1) = 0.0640857;
    tri3_xyze(1, 1) = -0.0640857;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4813);
    tri3_xyze(0, 2) = 0.0565805;
    tri3_xyze(1, 2) = -0.0737372;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-463);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515698;
    tri3_xyze(1, 0) = -0.0893214;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4567);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = -0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-464);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = -0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4565);
    tri3_xyze(0, 1) = 0.0684907;
    tri3_xyze(1, 1) = -0.0684907;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4815);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-464);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = 0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = 0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1001);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = 0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = 0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = 0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1001);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = 0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = 0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1001);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = 0.0459793;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1001);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = 0.0486771;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7389);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = 0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1002);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = 0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1002);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = 0.0436427;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7639);
    tri3_xyze(0, 2) = 0.0190453;
    tri3_xyze(1, 2) = 0.0459793;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1002);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = 0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = 0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = 0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1003);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0517663;
    tri3_xyze(1, 0) = -0.0517663;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(4875);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-494);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = -0.0590013;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(4627);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = -0.0555856;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-494);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-495);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(4629);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-495);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = -0.0590013;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(4627);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = -0.0517877;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-495);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-496);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(4631);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-496);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(4631);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = -0.0550373;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(4629);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-496);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = -0.0550373;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(4629);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = -0.0484134;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-496);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-497);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-497);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(4631);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-497);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(4631);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = -0.0455161;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-497);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-498);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-498);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = -0.0486573;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-498);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = -0.0486573;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4633);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = -0.0431415;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-498);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-499);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-499);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = -0.046342;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-499);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = -0.046342;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4635);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = -0.041327;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-499);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-500);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = -0.0436427;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-500);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = -0.0436427;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4639);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = -0.0446617;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-500);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = -0.0446617;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4637);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = -0.0401011;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-500);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-501);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-501);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-501);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-501);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-502);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-502);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(4791);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-502);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = -0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-502);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-503);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-503);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-503);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(4792);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-503);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-504);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-504);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-504);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(4795);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-504);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-505);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-505);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-505);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(4797);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-505);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-506);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-506);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-506);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(4799);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-506);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 1) = 0.0550373;
    tri3_xyze(1, 1) = -0.0317758;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-507);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-507);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-507);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(4801);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-507);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 1) = 0.0590013;
    tri3_xyze(1, 1) = -0.0340644;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-508);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0590013;
    tri3_xyze(1, 0) = -0.0340644;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-508);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-508);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(4803);
    tri3_xyze(0, 1) = 0.0550373;
    tri3_xyze(1, 1) = -0.0317758;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-508);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0590013;
    tri3_xyze(1, 0) = -0.0340644;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 1) = 0.0634006;
    tri3_xyze(1, 1) = -0.0366043;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-509);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0634006;
    tri3_xyze(1, 0) = -0.0366043;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = -0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-509);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0517663;
    tri3_xyze(1, 0) = -0.0517663;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-509);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(4805);
    tri3_xyze(0, 1) = 0.0590013;
    tri3_xyze(1, 1) = -0.0340644;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-509);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0634006;
    tri3_xyze(1, 0) = -0.0366043;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 1) = 0.0681658;
    tri3_xyze(1, 1) = -0.0393555;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 2) = 0.0597474;
    tri3_xyze(1, 2) = -0.0458458;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681658;
    tri3_xyze(1, 0) = -0.0393555;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 1) = 0.0556571;
    tri3_xyze(1, 1) = -0.0556571;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 2) = 0.0597474;
    tri3_xyze(1, 2) = -0.0458458;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0556571;
    tri3_xyze(1, 0) = -0.0556571;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = -0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 2) = 0.0597474;
    tri3_xyze(1, 2) = -0.0458458;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0517663;
    tri3_xyze(1, 0) = -0.0517663;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(4807);
    tri3_xyze(0, 1) = 0.0634006;
    tri3_xyze(1, 1) = -0.0366043;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 2) = 0.0597474;
    tri3_xyze(1, 2) = -0.0458458;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681658;
    tri3_xyze(1, 0) = -0.0393555;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 2) = 0.0642075;
    tri3_xyze(1, 2) = -0.0492681;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(4811);
    tri3_xyze(0, 2) = 0.0642075;
    tri3_xyze(1, 2) = -0.0492681;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4811);
    tri3_xyze(0, 1) = 0.0556571;
    tri3_xyze(1, 1) = -0.0556571;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 2) = 0.0642075;
    tri3_xyze(1, 2) = -0.0492681;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0556571;
    tri3_xyze(1, 0) = -0.0556571;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(4809);
    tri3_xyze(0, 1) = 0.0681658;
    tri3_xyze(1, 1) = -0.0393555;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 2) = 0.0642075;
    tri3_xyze(1, 2) = -0.0492681;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 1) = 0.0784887;
    tri3_xyze(1, 1) = -0.0453155;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 2) = 0.0688954;
    tri3_xyze(1, 2) = -0.0528653;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-512);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0784887;
    tri3_xyze(1, 0) = -0.0453155;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 1) = 0.0640857;
    tri3_xyze(1, 1) = -0.0640857;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(4813);
    tri3_xyze(0, 2) = 0.0688954;
    tri3_xyze(1, 2) = -0.0528653;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-512);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(4811);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 2) = 0.0688954;
    tri3_xyze(1, 2) = -0.0528653;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-512);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0784887;
    tri3_xyze(1, 0) = -0.0453155;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 1) = 0.0838836;
    tri3_xyze(1, 1) = -0.0484302;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 2) = 0.0737372;
    tri3_xyze(1, 2) = -0.0565805;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-513);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0838836;
    tri3_xyze(1, 0) = -0.0484302;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 1) = 0.0684907;
    tri3_xyze(1, 1) = -0.0684907;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(4815);
    tri3_xyze(0, 2) = 0.0737372;
    tri3_xyze(1, 2) = -0.0565805;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-513);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0640857;
    tri3_xyze(1, 0) = -0.0640857;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(4813);
    tri3_xyze(0, 1) = 0.0784887;
    tri3_xyze(1, 1) = -0.0453155;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 2) = 0.0737372;
    tri3_xyze(1, 2) = -0.0565805;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-513);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0838836;
    tri3_xyze(1, 0) = -0.0484302;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 1) = 0.0893214;
    tri3_xyze(1, 1) = -0.0515698;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5067);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-514);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0684907;
    tri3_xyze(1, 0) = -0.0684907;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(4815);
    tri3_xyze(0, 1) = 0.0838836;
    tri3_xyze(1, 1) = -0.0484302;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-514);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0590013;
    tri3_xyze(1, 0) = -0.0340644;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(5127);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-544);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = -0.0517663;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(4875);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = -0.0426524;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-544);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(5129);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-545);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = -0.0481743;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-545);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = -0.0481743;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(4877);
    tri3_xyze(0, 1) = 0.0590013;
    tri3_xyze(1, 1) = -0.0340644;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(5127);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = -0.0397381;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-545);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(5129);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(5131);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-546);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(5131);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-546);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = -0.0449377;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-546);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = -0.0449377;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(4879);
    tri3_xyze(0, 1) = 0.0550373;
    tri3_xyze(1, 1) = -0.0317758;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(5129);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = -0.0371489;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-546);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(5131);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-547);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-547);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-547);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(4881);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(5131);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = -0.0349258;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-547);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-548);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-548);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = -0.0397286;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-548);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = -0.0397286;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(4883);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = -0.0331036;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-548);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-549);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-549);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = -0.0378381;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-549);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = -0.0378381;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(4885);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = -0.0317113;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-549);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-550);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = -0.0356341;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-550);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = -0.0356341;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(4889);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = -0.0364661;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-550);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = -0.0364661;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(4887);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = -0.0307707;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-550);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-551);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-551);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-551);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-551);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-552);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-552);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5041);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-552);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = -0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-552);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-553);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-553);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-553);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5042);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-553);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-554);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-554);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-554);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5045);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-554);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 1) = 0.0542702;
    tri3_xyze(1, 1) = -0.0145417;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-555);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0542702;
    tri3_xyze(1, 0) = -0.0145417;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-555);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-555);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5047);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-555);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0542702;
    tri3_xyze(1, 0) = -0.0145417;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-556);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-556);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-556);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5049);
    tri3_xyze(0, 1) = 0.0542702;
    tri3_xyze(1, 1) = -0.0145417;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-556);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 1) = 0.0613861;
    tri3_xyze(1, 1) = -0.0164484;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 2) = 0.0563786;
    tri3_xyze(1, 2) = -0.0233528;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-557);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0613861;
    tri3_xyze(1, 0) = -0.0164484;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 1) = 0.0550373;
    tri3_xyze(1, 1) = -0.0317758;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 2) = 0.0563786;
    tri3_xyze(1, 2) = -0.0233528;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-557);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 2) = 0.0563786;
    tri3_xyze(1, 2) = -0.0233528;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-557);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5051);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 2) = 0.0563786;
    tri3_xyze(1, 2) = -0.0233528;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-557);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0613861;
    tri3_xyze(1, 0) = -0.0164484;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 1) = 0.0658074;
    tri3_xyze(1, 1) = -0.017633;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 2) = 0.060308;
    tri3_xyze(1, 2) = -0.0249804;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-558);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0658074;
    tri3_xyze(1, 0) = -0.017633;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 1) = 0.0590013;
    tri3_xyze(1, 1) = -0.0340644;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 2) = 0.060308;
    tri3_xyze(1, 2) = -0.0249804;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-558);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0590013;
    tri3_xyze(1, 0) = -0.0340644;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 1) = 0.0550373;
    tri3_xyze(1, 1) = -0.0317758;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 2) = 0.060308;
    tri3_xyze(1, 2) = -0.0249804;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-558);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = -0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5053);
    tri3_xyze(0, 1) = 0.0613861;
    tri3_xyze(1, 1) = -0.0164484;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 2) = 0.060308;
    tri3_xyze(1, 2) = -0.0249804;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-558);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0658074;
    tri3_xyze(1, 0) = -0.017633;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 1) = 0.0707141;
    tri3_xyze(1, 1) = -0.0189478;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 2) = 0.0647308;
    tri3_xyze(1, 2) = -0.0268124;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-559);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0707141;
    tri3_xyze(1, 0) = -0.0189478;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 1) = 0.0634006;
    tri3_xyze(1, 1) = -0.0366043;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 2) = 0.0647308;
    tri3_xyze(1, 2) = -0.0268124;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-559);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0634006;
    tri3_xyze(1, 0) = -0.0366043;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 1) = 0.0590013;
    tri3_xyze(1, 1) = -0.0340644;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 2) = 0.0647308;
    tri3_xyze(1, 2) = -0.0268124;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-559);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0590013;
    tri3_xyze(1, 0) = -0.0340644;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5055);
    tri3_xyze(0, 1) = 0.0658074;
    tri3_xyze(1, 1) = -0.017633;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 2) = 0.0647308;
    tri3_xyze(1, 2) = -0.0268124;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-559);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0707141;
    tri3_xyze(1, 0) = -0.0189478;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 1) = 0.076029;
    tri3_xyze(1, 1) = -0.0203719;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 2) = 0.0695774;
    tri3_xyze(1, 2) = -0.0288199;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-560);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.076029;
    tri3_xyze(1, 0) = -0.0203719;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 1) = 0.0681658;
    tri3_xyze(1, 1) = -0.0393555;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 2) = 0.0695774;
    tri3_xyze(1, 2) = -0.0288199;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-560);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681658;
    tri3_xyze(1, 0) = -0.0393555;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 1) = 0.0634006;
    tri3_xyze(1, 1) = -0.0366043;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 2) = 0.0695774;
    tri3_xyze(1, 2) = -0.0288199;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-560);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0634006;
    tri3_xyze(1, 0) = -0.0366043;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5057);
    tri3_xyze(0, 1) = 0.0707141;
    tri3_xyze(1, 1) = -0.0189478;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 2) = 0.0695774;
    tri3_xyze(1, 2) = -0.0288199;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-560);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.076029;
    tri3_xyze(1, 0) = -0.0203719;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 2) = 0.0747712;
    tri3_xyze(1, 2) = -0.0309712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-561);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 2) = 0.0747712;
    tri3_xyze(1, 2) = -0.0309712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-561);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 1) = 0.0681658;
    tri3_xyze(1, 1) = -0.0393555;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 2) = 0.0747712;
    tri3_xyze(1, 2) = -0.0309712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-561);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681658;
    tri3_xyze(1, 0) = -0.0393555;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5059);
    tri3_xyze(0, 1) = 0.076029;
    tri3_xyze(1, 1) = -0.0203719;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 2) = 0.0747712;
    tri3_xyze(1, 2) = -0.0309712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-561);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 1) = 0.0875428;
    tri3_xyze(1, 1) = -0.023457;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 2) = 0.0802303;
    tri3_xyze(1, 2) = -0.0332325;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-562);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0875428;
    tri3_xyze(1, 0) = -0.023457;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 1) = 0.0784887;
    tri3_xyze(1, 1) = -0.0453155;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 2) = 0.0802303;
    tri3_xyze(1, 2) = -0.0332325;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-562);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0784887;
    tri3_xyze(1, 0) = -0.0453155;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 2) = 0.0802303;
    tri3_xyze(1, 2) = -0.0332325;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-562);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5061);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 2) = 0.0802303;
    tri3_xyze(1, 2) = -0.0332325;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-562);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0875428;
    tri3_xyze(1, 0) = -0.023457;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 1) = 0.09356;
    tri3_xyze(1, 1) = -0.0250693;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 2) = 0.0858688;
    tri3_xyze(1, 2) = -0.035568;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-563);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.09356;
    tri3_xyze(1, 0) = -0.0250693;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 1) = 0.0838836;
    tri3_xyze(1, 1) = -0.0484302;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 2) = 0.0858688;
    tri3_xyze(1, 2) = -0.035568;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-563);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0838836;
    tri3_xyze(1, 0) = -0.0484302;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 1) = 0.0784887;
    tri3_xyze(1, 1) = -0.0453155;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 2) = 0.0858688;
    tri3_xyze(1, 2) = -0.035568;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-563);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0784887;
    tri3_xyze(1, 0) = -0.0453155;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5063);
    tri3_xyze(0, 1) = 0.0875428;
    tri3_xyze(1, 1) = -0.023457;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 2) = 0.0858688;
    tri3_xyze(1, 2) = -0.035568;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-563);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.09356;
    tri3_xyze(1, 0) = -0.0250693;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 1) = 0.0996251;
    tri3_xyze(1, 1) = -0.0266945;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-564);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0996251;
    tri3_xyze(1, 0) = -0.0266945;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 1) = 0.0893214;
    tri3_xyze(1, 1) = -0.0515698;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5067);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-564);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0893214;
    tri3_xyze(1, 0) = -0.0515698;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5067);
    tri3_xyze(0, 1) = 0.0838836;
    tri3_xyze(1, 1) = -0.0484302;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-564);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0838836;
    tri3_xyze(1, 0) = -0.0484302;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5065);
    tri3_xyze(0, 1) = 0.09356;
    tri3_xyze(1, 1) = -0.0250693;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-564);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0996251;
    tri3_xyze(1, 0) = -0.0266945;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 1) = 0.105642;
    tri3_xyze(1, 1) = -0.0283068;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5319);
    tri3_xyze(0, 2) = 0.0973263;
    tri3_xyze(1, 2) = -0.0403139;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-565);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.105642;
    tri3_xyze(1, 0) = -0.0283068;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5319);
    tri3_xyze(0, 1) = 0.0947164;
    tri3_xyze(1, 1) = -0.0546845;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5069);
    tri3_xyze(0, 2) = 0.0973263;
    tri3_xyze(1, 2) = -0.0403139;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-565);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0947164;
    tri3_xyze(1, 0) = -0.0546845;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5069);
    tri3_xyze(0, 1) = 0.0893214;
    tri3_xyze(1, 1) = -0.0515698;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5067);
    tri3_xyze(0, 2) = 0.0973263;
    tri3_xyze(1, 2) = -0.0403139;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-565);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0893214;
    tri3_xyze(1, 0) = -0.0515698;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5067);
    tri3_xyze(0, 1) = 0.0996251;
    tri3_xyze(1, 1) = -0.0266945;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 2) = 0.0973263;
    tri3_xyze(1, 2) = -0.0403139;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-565);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5071);
    tri3_xyze(0, 1) = 0.0947164;
    tri3_xyze(1, 1) = -0.0546845;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5069);
    tri3_xyze(0, 2) = 0.102965;
    tri3_xyze(1, 2) = -0.0426494;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-566);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0947164;
    tri3_xyze(1, 0) = -0.0546845;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5069);
    tri3_xyze(0, 1) = 0.105642;
    tri3_xyze(1, 1) = -0.0283068;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5319);
    tri3_xyze(0, 2) = 0.102965;
    tri3_xyze(1, 2) = -0.0426494;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-566);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0542702;
    tri3_xyze(1, 0) = -0.0145417;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(5383);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-597);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(5131);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = -0.0219553;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-597);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(5385);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-598);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = -0.0280923;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-598);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = -0.0280923;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(5133);
    tri3_xyze(0, 1) = 0.0542702;
    tri3_xyze(1, 1) = -0.0145417;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(5383);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = -0.0208098;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-598);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(5385);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-599);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-599);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = -0.0267556;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-599);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = -0.0267556;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(5135);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(5385);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = -0.0199346;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-599);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-600);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = -0.0251971;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-600);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = -0.0251971;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5139);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = -0.0257854;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-600);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = -0.0257854;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5137);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = -0.0193433;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-600);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 1) = 0.0503943;
    tri3_xyze(1, 1) = -1.96873e-17;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0503943;
    tri3_xyze(1, 0) = -1.96873e-17;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0503943;
    tri3_xyze(1, 0) = 1.96873e-17;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5639);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5291);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 1) = 0.0503943;
    tri3_xyze(1, 1) = 1.96873e-17;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5639);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = -0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0503943;
    tri3_xyze(1, 0) = -1.96873e-17;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 1) = 0.0515708;
    tri3_xyze(1, 1) = -3.90641e-17;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515708;
    tri3_xyze(1, 0) = -3.90641e-17;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5292);
    tri3_xyze(0, 1) = 0.0503943;
    tri3_xyze(1, 1) = -1.96873e-17;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515708;
    tri3_xyze(1, 0) = -3.90641e-17;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 1) = 0.0535112;
    tri3_xyze(1, 1) = -5.78249e-17;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0535112;
    tri3_xyze(1, 0) = -5.78249e-17;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5295);
    tri3_xyze(0, 1) = 0.0515708;
    tri3_xyze(1, 1) = -3.90641e-17;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0535112;
    tri3_xyze(1, 0) = -5.78249e-17;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 1) = 0.0561847;
    tri3_xyze(1, 1) = -7.56737e-17;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 2) = 0.0539135;
    tri3_xyze(1, 2) = -0.00709784;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0561847;
    tri3_xyze(1, 0) = -7.56737e-17;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 1) = 0.0542702;
    tri3_xyze(1, 1) = -0.0145417;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 2) = 0.0539135;
    tri3_xyze(1, 2) = -0.00709784;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0542702;
    tri3_xyze(1, 0) = -0.0145417;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 2) = 0.0539135;
    tri3_xyze(1, 2) = -0.00709784;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = -0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5297);
    tri3_xyze(0, 1) = 0.0535112;
    tri3_xyze(1, 1) = -5.78249e-17;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 2) = 0.0539135;
    tri3_xyze(1, 2) = -0.00709784;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0561847;
    tri3_xyze(1, 0) = -7.56737e-17;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-17;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 2) = 0.056881;
    tri3_xyze(1, 2) = -0.00748853;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-17;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 2) = 0.056881;
    tri3_xyze(1, 2) = -0.00748853;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 1) = 0.0542702;
    tri3_xyze(1, 1) = -0.0145417;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 2) = 0.056881;
    tri3_xyze(1, 2) = -0.00748853;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0542702;
    tri3_xyze(1, 0) = -0.0145417;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5299);
    tri3_xyze(0, 1) = 0.0561847;
    tri3_xyze(1, 1) = -7.56737e-17;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 2) = 0.056881;
    tri3_xyze(1, 2) = -0.00748853;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-17;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 1) = 0.0635516;
    tri3_xyze(1, 1) = -1.07528e-16;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 2) = 0.0605017;
    tri3_xyze(1, 2) = -0.0079652;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-607);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0635516;
    tri3_xyze(1, 0) = -1.07528e-16;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 1) = 0.0613861;
    tri3_xyze(1, 1) = -0.0164484;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 2) = 0.0605017;
    tri3_xyze(1, 2) = -0.0079652;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-607);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0613861;
    tri3_xyze(1, 0) = -0.0164484;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 2) = 0.0605017;
    tri3_xyze(1, 2) = -0.0079652;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-607);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5301);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-17;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 2) = 0.0605017;
    tri3_xyze(1, 2) = -0.0079652;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-607);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0635516;
    tri3_xyze(1, 0) = -1.07528e-16;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 1) = 0.0681288;
    tri3_xyze(1, 1) = -1.21032e-16;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 2) = 0.0647185;
    tri3_xyze(1, 2) = -0.00852035;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-608);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681288;
    tri3_xyze(1, 0) = -1.21032e-16;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 1) = 0.0658074;
    tri3_xyze(1, 1) = -0.017633;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 2) = 0.0647185;
    tri3_xyze(1, 2) = -0.00852035;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-608);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0658074;
    tri3_xyze(1, 0) = -0.017633;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 1) = 0.0613861;
    tri3_xyze(1, 1) = -0.0164484;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 2) = 0.0647185;
    tri3_xyze(1, 2) = -0.00852035;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-608);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0613861;
    tri3_xyze(1, 0) = -0.0164484;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5303);
    tri3_xyze(0, 1) = 0.0635516;
    tri3_xyze(1, 1) = -1.07528e-16;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 2) = 0.0647185;
    tri3_xyze(1, 2) = -0.00852035;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-608);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681288;
    tri3_xyze(1, 0) = -1.21032e-16;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 1) = 0.0732087;
    tri3_xyze(1, 1) = -1.32627e-16;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 2) = 0.0694647;
    tri3_xyze(1, 2) = -0.00914521;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-609);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732087;
    tri3_xyze(1, 0) = -1.32627e-16;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 1) = 0.0707141;
    tri3_xyze(1, 1) = -0.0189478;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 2) = 0.0694647;
    tri3_xyze(1, 2) = -0.00914521;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-609);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0707141;
    tri3_xyze(1, 0) = -0.0189478;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 1) = 0.0658074;
    tri3_xyze(1, 1) = -0.017633;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 2) = 0.0694647;
    tri3_xyze(1, 2) = -0.00914521;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-609);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0658074;
    tri3_xyze(1, 0) = -0.017633;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5305);
    tri3_xyze(0, 1) = 0.0681288;
    tri3_xyze(1, 1) = -1.21032e-16;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 2) = 0.0694647;
    tri3_xyze(1, 2) = -0.00914521;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-609);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732087;
    tri3_xyze(1, 0) = -1.32627e-16;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 1) = 0.078711;
    tri3_xyze(1, 1) = -1.4213e-16;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 2) = 0.0746657;
    tri3_xyze(1, 2) = -0.00982993;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-610);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.078711;
    tri3_xyze(1, 0) = -1.4213e-16;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 1) = 0.076029;
    tri3_xyze(1, 1) = -0.0203719;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 2) = 0.0746657;
    tri3_xyze(1, 2) = -0.00982993;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-610);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.076029;
    tri3_xyze(1, 0) = -0.0203719;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 1) = 0.0707141;
    tri3_xyze(1, 1) = -0.0189478;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 2) = 0.0746657;
    tri3_xyze(1, 2) = -0.00982993;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-610);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0707141;
    tri3_xyze(1, 0) = -0.0189478;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5307);
    tri3_xyze(0, 1) = 0.0732087;
    tri3_xyze(1, 1) = -1.32627e-16;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 2) = 0.0746657;
    tri3_xyze(1, 2) = -0.00982993;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-610);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.078711;
    tri3_xyze(1, 0) = -1.4213e-16;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-16;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 2) = 0.0802394;
    tri3_xyze(1, 2) = -0.0105637;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-16;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 2) = 0.0802394;
    tri3_xyze(1, 2) = -0.0105637;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 1) = 0.076029;
    tri3_xyze(1, 1) = -0.0203719;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 2) = 0.0802394;
    tri3_xyze(1, 2) = -0.0105637;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.076029;
    tri3_xyze(1, 0) = -0.0203719;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5309);
    tri3_xyze(0, 1) = 0.078711;
    tri3_xyze(1, 1) = -1.4213e-16;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 2) = 0.0802394;
    tri3_xyze(1, 2) = -0.0105637;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-16;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 1) = 0.0906309;
    tri3_xyze(1, 1) = -1.54297e-16;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 2) = 0.0860978;
    tri3_xyze(1, 2) = -0.011335;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0906309;
    tri3_xyze(1, 0) = -1.54297e-16;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 1) = 0.0875428;
    tri3_xyze(1, 1) = -0.023457;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 2) = 0.0860978;
    tri3_xyze(1, 2) = -0.011335;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0875428;
    tri3_xyze(1, 0) = -0.023457;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 2) = 0.0860978;
    tri3_xyze(1, 2) = -0.011335;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5311);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-16;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 2) = 0.0860978;
    tri3_xyze(1, 2) = -0.011335;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0906309;
    tri3_xyze(1, 0) = -1.54297e-16;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 1) = 0.0968605;
    tri3_xyze(1, 1) = -1.5677e-16;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 2) = 0.0921486;
    tri3_xyze(1, 2) = -0.0121316;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-613);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0968605;
    tri3_xyze(1, 0) = -1.5677e-16;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 1) = 0.09356;
    tri3_xyze(1, 1) = -0.0250693;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 2) = 0.0921486;
    tri3_xyze(1, 2) = -0.0121316;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-613);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.09356;
    tri3_xyze(1, 0) = -0.0250693;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 1) = 0.0875428;
    tri3_xyze(1, 1) = -0.023457;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 2) = 0.0921486;
    tri3_xyze(1, 2) = -0.0121316;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-613);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0875428;
    tri3_xyze(1, 0) = -0.023457;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5313);
    tri3_xyze(0, 1) = 0.0906309;
    tri3_xyze(1, 1) = -1.54297e-16;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 2) = 0.0921486;
    tri3_xyze(1, 2) = -0.0121316;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-613);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0968605;
    tri3_xyze(1, 0) = -1.5677e-16;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 1) = 0.10314;
    tri3_xyze(1, 1) = -1.5677e-16;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5567);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-614);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.10314;
    tri3_xyze(1, 0) = -1.5677e-16;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5567);
    tri3_xyze(0, 1) = 0.0996251;
    tri3_xyze(1, 1) = -0.0266945;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-614);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0996251;
    tri3_xyze(1, 0) = -0.0266945;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 1) = 0.09356;
    tri3_xyze(1, 1) = -0.0250693;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-614);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.09356;
    tri3_xyze(1, 0) = -0.0250693;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5315);
    tri3_xyze(0, 1) = 0.0968605;
    tri3_xyze(1, 1) = -1.5677e-16;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-614);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.105642;
    tri3_xyze(1, 0) = -0.0283068;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5319);
    tri3_xyze(0, 1) = 0.0996251;
    tri3_xyze(1, 1) = -0.0266945;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 2) = 0.104444;
    tri3_xyze(1, 2) = -0.0137503;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-615);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0996251;
    tri3_xyze(1, 0) = -0.0266945;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5317);
    tri3_xyze(0, 1) = 0.10314;
    tri3_xyze(1, 1) = -1.5677e-16;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5567);
    tri3_xyze(0, 2) = 0.104444;
    tri3_xyze(1, 2) = -0.0137503;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-615);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515708;
    tri3_xyze(1, 0) = 3.90641e-17;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5637);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-649);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = -0.0138497;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(5385);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = -0.00679931;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-649);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0503943;
    tri3_xyze(1, 0) = 1.96873e-17;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5639);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = -0.013043;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-650);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = -0.013043;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(5389);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = -0.0133475;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-650);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = -0.0133475;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(5387);
    tri3_xyze(0, 1) = 0.0515708;
    tri3_xyze(1, 1) = 3.90641e-17;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(5637);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = -0.00659763;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-650);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = 0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-651);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = 0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 1) = 0.0503943;
    tri3_xyze(1, 1) = -1.96873e-17;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-651);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0503943;
    tri3_xyze(1, 0) = -1.96873e-17;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-651);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-651);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = 0.013043;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(8389);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-652);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-652);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(5541);
    tri3_xyze(0, 1) = 0.0503943;
    tri3_xyze(1, 1) = 1.96873e-17;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(5639);
    tri3_xyze(0, 2) = 0.0493419;
    tri3_xyze(1, 2) = 0.00649599;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-652);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = 0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = 0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = 0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-653);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = 0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 1) = 0.0515708;
    tri3_xyze(1, 1) = -3.90641e-17;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = 0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-653);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515708;
    tri3_xyze(1, 0) = -3.90641e-17;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 1) = 0.0503943;
    tri3_xyze(1, 1) = -1.96873e-17;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = 0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-653);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0503943;
    tri3_xyze(1, 0) = -1.96873e-17;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(5542);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = 0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 2) = 0.050114;
    tri3_xyze(1, 2) = 0.00659763;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-653);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = 0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = 0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(8297);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = 0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-654);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = 0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(8297);
    tri3_xyze(0, 1) = 0.0535112;
    tri3_xyze(1, 1) = -5.78249e-17;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = 0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-654);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0535112;
    tri3_xyze(1, 0) = -5.78249e-17;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 1) = 0.0515708;
    tri3_xyze(1, 1) = -3.90641e-17;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = 0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-654);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515708;
    tri3_xyze(1, 0) = -3.90641e-17;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(5545);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = 0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 2) = 0.0516459;
    tri3_xyze(1, 2) = 0.00679931;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-654);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0542702;
    tri3_xyze(1, 0) = 0.0145417;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(8299);
    tri3_xyze(0, 1) = 0.0561847;
    tri3_xyze(1, 1) = -7.56737e-17;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 2) = 0.0539135;
    tri3_xyze(1, 2) = 0.00709784;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-655);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0561847;
    tri3_xyze(1, 0) = -7.56737e-17;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 1) = 0.0535112;
    tri3_xyze(1, 1) = -5.78249e-17;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 2) = 0.0539135;
    tri3_xyze(1, 2) = 0.00709784;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-655);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0535112;
    tri3_xyze(1, 0) = -5.78249e-17;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(5547);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = 0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(8297);
    tri3_xyze(0, 2) = 0.0539135;
    tri3_xyze(1, 2) = 0.00709784;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-655);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(8301);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-17;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 2) = 0.056881;
    tri3_xyze(1, 2) = 0.00748853;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-656);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-17;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 1) = 0.0561847;
    tri3_xyze(1, 1) = -7.56737e-17;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 2) = 0.056881;
    tri3_xyze(1, 2) = 0.00748853;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-656);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0561847;
    tri3_xyze(1, 0) = -7.56737e-17;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(5549);
    tri3_xyze(0, 1) = 0.0542702;
    tri3_xyze(1, 1) = 0.0145417;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(8299);
    tri3_xyze(0, 2) = 0.056881;
    tri3_xyze(1, 2) = 0.00748853;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-656);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0613861;
    tri3_xyze(1, 0) = 0.0164484;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(8303);
    tri3_xyze(0, 1) = 0.0635516;
    tri3_xyze(1, 1) = -1.07528e-16;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 2) = 0.0605017;
    tri3_xyze(1, 2) = 0.0079652;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-657);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0635516;
    tri3_xyze(1, 0) = -1.07528e-16;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-17;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 2) = 0.0605017;
    tri3_xyze(1, 2) = 0.0079652;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-657);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-17;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(5551);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(8301);
    tri3_xyze(0, 2) = 0.0605017;
    tri3_xyze(1, 2) = 0.0079652;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-657);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0658074;
    tri3_xyze(1, 0) = 0.017633;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(8305);
    tri3_xyze(0, 1) = 0.0681288;
    tri3_xyze(1, 1) = -1.21032e-16;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 2) = 0.0647185;
    tri3_xyze(1, 2) = 0.00852035;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-658);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681288;
    tri3_xyze(1, 0) = -1.21032e-16;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 1) = 0.0635516;
    tri3_xyze(1, 1) = -1.07528e-16;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 2) = 0.0647185;
    tri3_xyze(1, 2) = 0.00852035;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-658);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0635516;
    tri3_xyze(1, 0) = -1.07528e-16;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(5553);
    tri3_xyze(0, 1) = 0.0613861;
    tri3_xyze(1, 1) = 0.0164484;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(8303);
    tri3_xyze(0, 2) = 0.0647185;
    tri3_xyze(1, 2) = 0.00852035;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-658);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0707141;
    tri3_xyze(1, 0) = 0.0189478;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(8307);
    tri3_xyze(0, 1) = 0.0732087;
    tri3_xyze(1, 1) = -1.32627e-16;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 2) = 0.0694647;
    tri3_xyze(1, 2) = 0.00914521;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-659);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732087;
    tri3_xyze(1, 0) = -1.32627e-16;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 1) = 0.0681288;
    tri3_xyze(1, 1) = -1.21032e-16;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 2) = 0.0694647;
    tri3_xyze(1, 2) = 0.00914521;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-659);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0681288;
    tri3_xyze(1, 0) = -1.21032e-16;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(5555);
    tri3_xyze(0, 1) = 0.0658074;
    tri3_xyze(1, 1) = 0.017633;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(8305);
    tri3_xyze(0, 2) = 0.0694647;
    tri3_xyze(1, 2) = 0.00914521;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-659);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.076029;
    tri3_xyze(1, 0) = 0.0203719;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(8309);
    tri3_xyze(0, 1) = 0.078711;
    tri3_xyze(1, 1) = -1.4213e-16;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 2) = 0.0746657;
    tri3_xyze(1, 2) = 0.00982993;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-660);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.078711;
    tri3_xyze(1, 0) = -1.4213e-16;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 1) = 0.0732087;
    tri3_xyze(1, 1) = -1.32627e-16;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 2) = 0.0746657;
    tri3_xyze(1, 2) = 0.00982993;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-660);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732087;
    tri3_xyze(1, 0) = -1.32627e-16;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(5557);
    tri3_xyze(0, 1) = 0.0707141;
    tri3_xyze(1, 1) = 0.0189478;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(8307);
    tri3_xyze(0, 2) = 0.0746657;
    tri3_xyze(1, 2) = 0.00982993;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-660);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = 0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(8311);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-16;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 2) = 0.0802394;
    tri3_xyze(1, 2) = 0.0105637;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-661);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-16;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 1) = 0.078711;
    tri3_xyze(1, 1) = -1.4213e-16;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 2) = 0.0802394;
    tri3_xyze(1, 2) = 0.0105637;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-661);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.078711;
    tri3_xyze(1, 0) = -1.4213e-16;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(5559);
    tri3_xyze(0, 1) = 0.076029;
    tri3_xyze(1, 1) = 0.0203719;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(8309);
    tri3_xyze(0, 2) = 0.0802394;
    tri3_xyze(1, 2) = 0.0105637;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-661);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0875428;
    tri3_xyze(1, 0) = 0.023457;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(8313);
    tri3_xyze(0, 1) = 0.0906309;
    tri3_xyze(1, 1) = -1.54297e-16;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 2) = 0.0860978;
    tri3_xyze(1, 2) = 0.011335;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-662);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0906309;
    tri3_xyze(1, 0) = -1.54297e-16;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-16;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 2) = 0.0860978;
    tri3_xyze(1, 2) = 0.011335;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-662);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-16;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(5561);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = 0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(8311);
    tri3_xyze(0, 2) = 0.0860978;
    tri3_xyze(1, 2) = 0.011335;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-662);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.09356;
    tri3_xyze(1, 0) = 0.0250693;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(8315);
    tri3_xyze(0, 1) = 0.0968605;
    tri3_xyze(1, 1) = -1.5677e-16;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 2) = 0.0921486;
    tri3_xyze(1, 2) = 0.0121316;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-663);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0968605;
    tri3_xyze(1, 0) = -1.5677e-16;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 1) = 0.0906309;
    tri3_xyze(1, 1) = -1.54297e-16;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 2) = 0.0921486;
    tri3_xyze(1, 2) = 0.0121316;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-663);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0906309;
    tri3_xyze(1, 0) = -1.54297e-16;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(5563);
    tri3_xyze(0, 1) = 0.0875428;
    tri3_xyze(1, 1) = 0.023457;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(8313);
    tri3_xyze(0, 2) = 0.0921486;
    tri3_xyze(1, 2) = 0.0121316;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-663);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.10314;
    tri3_xyze(1, 0) = -1.5677e-16;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5567);
    tri3_xyze(0, 1) = 0.0968605;
    tri3_xyze(1, 1) = -1.5677e-16;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-664);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0968605;
    tri3_xyze(1, 0) = -1.5677e-16;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(5565);
    tri3_xyze(0, 1) = 0.09356;
    tri3_xyze(1, 1) = 0.0250693;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(8315);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-664);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00309e-15;
    tri3_xyze(1, 0) = 0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-901);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(6791);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-901);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(6791);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-902);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 1) = 1.00309e-15;
    tri3_xyze(1, 1) = 0.0503943;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7139);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-902);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 1) = 1.00309e-15;
    tri3_xyze(1, 1) = 0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-951);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00309e-15;
    tri3_xyze(1, 0) = 0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = 0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-951);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = 0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-951);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-951);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00309e-15;
    tri3_xyze(1, 0) = 0.0503943;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7139);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-952);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7041);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-952);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7291);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = 0.0486771;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7389);
    tri3_xyze(0, 2) = 0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-952);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00316e-15;
    tri3_xyze(1, 0) = 0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = 0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = 0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-953);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = 0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = 0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = 0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-953);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.013043;
    tri3_xyze(1, 0) = 0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 1) = 1.00309e-15;
    tri3_xyze(1, 1) = 0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = 0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-953);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = 0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = 0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = 0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-954);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = 0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 1) = 1.00316e-15;
    tri3_xyze(1, 1) = 0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = 0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-954);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = 0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = 0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = 0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1003);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = 0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = 0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = 0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1003);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = 0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 1) = 0.013043;
    tri3_xyze(1, 1) = 0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7292);
    tri3_xyze(0, 2) = 0.0193433;
    tri3_xyze(1, 2) = 0.0466988;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1003);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0133475;
    tri3_xyze(1, 0) = 0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = 0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = 0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1004);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = 0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = 0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = 0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1004);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = 0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = 0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = 0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1004);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = 0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 1) = 0.0133475;
    tri3_xyze(1, 1) = 0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7295);
    tri3_xyze(0, 2) = 0.0199346;
    tri3_xyze(1, 2) = 0.0481263;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1004);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = 0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = 0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = 0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1005);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = 0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = 0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = 0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1005);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = 0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = 0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = 0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1005);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = 0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = 0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = 0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1006);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = 0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = 0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = 0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1006);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = 0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = 0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1051);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = 0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = 0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = 0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1051);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = 0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = 0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1051);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = 0.0394834;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1051);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = 0.0436427;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7639);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = 0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1052);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7541);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = 0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1052);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = 0.0356341;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7889);
    tri3_xyze(0, 2) = 0.0302966;
    tri3_xyze(1, 2) = 0.0394834;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1052);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0251971;
    tri3_xyze(1, 0) = 0.0436427;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = 0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = 0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1053);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = 0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = 0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = 0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1053);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = 0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = 0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = 0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1053);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = 0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 1) = 0.0251971;
    tri3_xyze(1, 1) = 0.0436427;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7542);
    tri3_xyze(0, 2) = 0.0307707;
    tri3_xyze(1, 2) = 0.0401011;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1053);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0257854;
    tri3_xyze(1, 0) = 0.0446617;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = 0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = 0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1054);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = 0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = 0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = 0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1054);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = 0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = 0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = 0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1054);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = 0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 1) = 0.0257854;
    tri3_xyze(1, 1) = 0.0446617;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7545);
    tri3_xyze(0, 2) = 0.0317113;
    tri3_xyze(1, 2) = 0.041327;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1054);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0267556;
    tri3_xyze(1, 0) = 0.046342;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = 0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = 0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1055);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = 0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = 0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = 0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1055);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = 0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = 0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = 0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1055);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = 0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 1) = 0.0267556;
    tri3_xyze(1, 1) = 0.046342;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7547);
    tri3_xyze(0, 2) = 0.0331036;
    tri3_xyze(1, 2) = 0.0431415;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1055);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0280923;
    tri3_xyze(1, 0) = 0.0486573;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = 0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = 0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1056);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = 0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = 0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1056);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = 0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = 0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1056);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = 0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 1) = 0.0280923;
    tri3_xyze(1, 1) = 0.0486573;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7549);
    tri3_xyze(0, 2) = 0.0349258;
    tri3_xyze(1, 2) = 0.0455161;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1056);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = 0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = 0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = 0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1057);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = 0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = 0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = 0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1057);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = 0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = 0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1057);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = 0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 2) = 0.0371489;
    tri3_xyze(1, 2) = 0.0484134;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1057);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = 0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = 0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = 0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1058);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = 0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = 0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = 0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1058);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = 0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = 0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = 0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1058);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0517663;
    tri3_xyze(1, 0) = 0.0517663;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7807);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = 0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = 0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1059);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = 0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = 0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = 0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1059);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = 0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = 0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = 0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = 0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = 0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = 0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = 0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = 0.0302966;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = 0.0356341;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7889);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = 0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7791);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = 0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = 0.0251971;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(8139);
    tri3_xyze(0, 2) = 0.0394834;
    tri3_xyze(1, 2) = 0.0302966;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0356341;
    tri3_xyze(1, 0) = 0.0356341;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = 0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = 0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = 0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = 0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = 0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = 0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = 0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = 0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = 0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 1) = 0.0356341;
    tri3_xyze(1, 1) = 0.0356341;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7792);
    tri3_xyze(0, 2) = 0.0401011;
    tri3_xyze(1, 2) = 0.0307707;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0364661;
    tri3_xyze(1, 0) = 0.0364661;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = 0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = 0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1104);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = 0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = 0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = 0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1104);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = 0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = 0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = 0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1104);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = 0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 1) = 0.0364661;
    tri3_xyze(1, 1) = 0.0364661;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7795);
    tri3_xyze(0, 2) = 0.041327;
    tri3_xyze(1, 2) = 0.0317113;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1104);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0378381;
    tri3_xyze(1, 0) = 0.0378381;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = 0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = 0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = 0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = 0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = 0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = 0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = 0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = 0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = 0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 1) = 0.0378381;
    tri3_xyze(1, 1) = 0.0378381;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7797);
    tri3_xyze(0, 2) = 0.0431415;
    tri3_xyze(1, 2) = 0.0331036;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1105);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0397286;
    tri3_xyze(1, 0) = 0.0397286;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = 0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(8051);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = 0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(8051);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = 0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = 0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = 0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 1) = 0.0397286;
    tri3_xyze(1, 1) = 0.0397286;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7799);
    tri3_xyze(0, 2) = 0.0455161;
    tri3_xyze(1, 2) = 0.0349258;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1106);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = 0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = 0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = 0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 1) = 0.0550373;
    tri3_xyze(1, 1) = 0.0317758;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(8053);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = 0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = 0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(8053);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(8051);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = 0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(8051);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7801);
    tri3_xyze(0, 2) = 0.0484134;
    tri3_xyze(1, 2) = 0.0371489;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0449377;
    tri3_xyze(1, 0) = 0.0449377;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = 0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = 0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1108);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = 0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 1) = 0.0590013;
    tri3_xyze(1, 1) = 0.0340644;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(8055);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = 0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1108);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0550373;
    tri3_xyze(1, 0) = 0.0317758;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(8053);
    tri3_xyze(0, 1) = 0.0449377;
    tri3_xyze(1, 1) = 0.0449377;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7803);
    tri3_xyze(0, 2) = 0.0517877;
    tri3_xyze(1, 2) = 0.0397381;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1108);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0481743;
    tri3_xyze(1, 0) = 0.0481743;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = 0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7807);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = 0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1109);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0590013;
    tri3_xyze(1, 0) = 0.0340644;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(8055);
    tri3_xyze(0, 1) = 0.0481743;
    tri3_xyze(1, 1) = 0.0481743;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7805);
    tri3_xyze(0, 2) = 0.0555856;
    tri3_xyze(1, 2) = 0.0426524;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1109);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = 0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = 0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = 0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = 0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = 0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = 0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = 0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = 0.0190453;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = 0.0251971;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(8139);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = 0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8041);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = 0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(8291);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = 0.013043;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(8389);
    tri3_xyze(0, 2) = 0.0459793;
    tri3_xyze(1, 2) = 0.0190453;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0436427;
    tri3_xyze(1, 0) = 0.0251971;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = 0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = 0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = 0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = 0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = 0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = 0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 1) = 0.0486771;
    tri3_xyze(1, 1) = 0.013043;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = 0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486771;
    tri3_xyze(1, 0) = 0.013043;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(8292);
    tri3_xyze(0, 1) = 0.0436427;
    tri3_xyze(1, 1) = 0.0251971;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(8042);
    tri3_xyze(0, 2) = 0.0466988;
    tri3_xyze(1, 2) = 0.0193433;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0446617;
    tri3_xyze(1, 0) = 0.0257854;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = 0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = 0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = 0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 1) = 0.0516878;
    tri3_xyze(1, 1) = 0.0138497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(8297);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = 0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = 0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(8297);
    tri3_xyze(0, 1) = 0.0498136;
    tri3_xyze(1, 1) = 0.0133475;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = 0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0498136;
    tri3_xyze(1, 0) = 0.0133475;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(8295);
    tri3_xyze(0, 1) = 0.0446617;
    tri3_xyze(1, 1) = 0.0257854;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(8045);
    tri3_xyze(0, 2) = 0.0481263;
    tri3_xyze(1, 2) = 0.0199346;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1154);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.046342;
    tri3_xyze(1, 0) = 0.0267556;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = 0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = 0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1155);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = 0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 1) = 0.0542702;
    tri3_xyze(1, 1) = 0.0145417;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(8299);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = 0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1155);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0516878;
    tri3_xyze(1, 0) = 0.0138497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(8297);
    tri3_xyze(0, 1) = 0.046342;
    tri3_xyze(1, 1) = 0.0267556;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(8047);
    tri3_xyze(0, 2) = 0.0502394;
    tri3_xyze(1, 2) = 0.0208098;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1155);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0486573;
    tri3_xyze(1, 0) = 0.0280923;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(8051);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = 0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1156);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0542702;
    tri3_xyze(1, 0) = 0.0145417;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(8299);
    tri3_xyze(0, 1) = 0.0486573;
    tri3_xyze(1, 1) = 0.0280923;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(8049);
    tri3_xyze(0, 2) = 0.0530047;
    tri3_xyze(1, 2) = 0.0219553;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1156);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.1;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1852);
    hex8_xyze(0, 1) = 0.1;
    hex8_xyze(1, 1) = 1.73472e-18;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1853);
    hex8_xyze(0, 2) = 0.05;
    hex8_xyze(1, 2) = -1.73472e-18;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1864);
    hex8_xyze(0, 3) = 0.05;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1863);
    hex8_xyze(0, 4) = 0.1;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1973);
    hex8_xyze(0, 5) = 0.1;
    hex8_xyze(1, 5) = 2.08167e-18;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1974);
    hex8_xyze(0, 6) = 0.05;
    hex8_xyze(1, 6) = -2.08167e-18;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1985);
    hex8_xyze(0, 7) = 0.05;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1984);

    intersection.add_element(6910, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = -0.1;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1862);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = -0.05;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1863);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = -0.05;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1874);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = -0.1;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1873);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = -0.1;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1983);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = -0.05;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1984);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = -0.05;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1995);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = -0.1;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1994);

    intersection.add_element(6919, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1863);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = -1.73472e-18;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1864);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1875);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1874);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1984);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = -2.08167e-18;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1985);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1996);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1995);

    intersection.add_element(6920, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = -1.73472e-18;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1864);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = 0.05;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1865);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0.05;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1876);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = 0;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1875);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = -2.08167e-18;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1985);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = 0.05;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1986);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0.05;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1997);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = 0;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1996);

    intersection.add_element(6921, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1874);
    hex8_xyze(0, 1) = 0;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1875);
    hex8_xyze(0, 2) = -0.05;
    hex8_xyze(1, 2) = 0;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1886);
    hex8_xyze(0, 3) = -0.05;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1885);
    hex8_xyze(0, 4) = 0;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1995);
    hex8_xyze(0, 5) = 0;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1996);
    hex8_xyze(0, 6) = -0.05;
    hex8_xyze(1, 6) = 0;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(2007);
    hex8_xyze(0, 7) = -0.05;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(2006);

    intersection.add_element(6930, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.8;
    nids.push_back(1984);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = -2.08167e-18;
    hex8_xyze(2, 1) = 0.8;
    nids.push_back(1985);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0;
    hex8_xyze(2, 2) = 0.8;
    nids.push_back(1996);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.8;
    nids.push_back(1995);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.85;
    nids.push_back(2105);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = -2.42861e-18;
    hex8_xyze(2, 5) = 0.85;
    nids.push_back(2106);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0;
    hex8_xyze(2, 6) = 0.85;
    nids.push_back(2117);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.85;
    nids.push_back(2116);

    intersection.add_element(7020, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
