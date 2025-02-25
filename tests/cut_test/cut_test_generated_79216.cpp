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

void test_generated_79216()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52004836105039309313;
    tri3_xyze(1, 0) = 0.017472065563509005248;
    tri3_xyze(2, 0) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 1) = 0.52703808292901843657;
    tri3_xyze(1, 1) = 0.017521739375091063828;
    tri3_xyze(2, 1) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 2) = 0.52284945967201235106;
    tri3_xyze(1, 2) = 0.026236537704978871166;
    tri3_xyze(2, 2) = 0.29985116843686548949;
    nids.push_back(-2614);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68880);
    tri3_xyze(0, 2) = 0.49948176897042095845;
    tri3_xyze(1, 2) = 0.0084127679258171050858;
    tri3_xyze(2, 2) = 0.29245331965377791006;
    nids.push_back(-781);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68880);
    tri3_xyze(0, 1) = 0.49897108020437830334;
    tri3_xyze(1, 1) = 0.016870489445534233436;
    tri3_xyze(2, 1) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 2) = 0.49948176897042095845;
    tri3_xyze(1, 2) = 0.0084127679258171050858;
    tri3_xyze(2, 2) = 0.29245331965377791006;
    nids.push_back(-781);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49897108020437830334;
    tri3_xyze(1, 0) = 0.016870489445534233436;
    tri3_xyze(2, 0) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 1) = 0.49895599567730553048;
    tri3_xyze(1, 1) = 0.016780582257734183438;
    tri3_xyze(2, 1) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 2) = 0.49948176897042095845;
    tri3_xyze(1, 2) = 0.0084127679258171050858;
    tri3_xyze(2, 2) = 0.29245331965377791006;
    nids.push_back(-781);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49895599567730553048;
    tri3_xyze(1, 0) = 0.016780582257734183438;
    tri3_xyze(2, 0) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 2) = 0.49948176897042095845;
    tri3_xyze(1, 2) = 0.0084127679258171050858;
    tri3_xyze(2, 2) = 0.29245331965377791006;
    nids.push_back(-781);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49895599567730553048;
    tri3_xyze(1, 0) = 0.016780582257734183438;
    tri3_xyze(2, 0) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 1) = 0.49897108020437830334;
    tri3_xyze(1, 1) = 0.016870489445534233436;
    tri3_xyze(2, 1) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 2) = 0.4983554349761136204;
    tri3_xyze(1, 2) = 0.025355575985797778465;
    tri3_xyze(2, 2) = 0.29214113724612911227;
    nids.push_back(-782);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49897108020437830334;
    tri3_xyze(1, 0) = 0.016870489445534233436;
    tri3_xyze(2, 0) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 1) = 0.49763623912943577565;
    tri3_xyze(1, 1) = 0.033973962005409537313;
    tri3_xyze(2, 1) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 2) = 0.4983554349761136204;
    tri3_xyze(1, 2) = 0.025355575985797778465;
    tri3_xyze(2, 2) = 0.29214113724612911227;
    nids.push_back(-782);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49763623912943577565;
    tri3_xyze(1, 0) = 0.033973962005409537313;
    tri3_xyze(2, 0) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 1) = 0.49785842489333492766;
    tri3_xyze(1, 1) = 0.033797270234513163145;
    tri3_xyze(2, 1) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 2) = 0.4983554349761136204;
    tri3_xyze(1, 2) = 0.025355575985797778465;
    tri3_xyze(2, 2) = 0.29214113724612911227;
    nids.push_back(-782);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49785842489333492766;
    tri3_xyze(1, 0) = 0.033797270234513163145;
    tri3_xyze(2, 0) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 1) = 0.49895599567730553048;
    tri3_xyze(1, 1) = 0.016780582257734183438;
    tri3_xyze(2, 1) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 2) = 0.4983554349761136204;
    tri3_xyze(1, 2) = 0.025355575985797778465;
    tri3_xyze(2, 2) = 0.29214113724612911227;
    nids.push_back(-782);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49785842489333492766;
    tri3_xyze(1, 0) = 0.033797270234513163145;
    tri3_xyze(2, 0) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 1) = 0.49763623912943577565;
    tri3_xyze(1, 1) = 0.033973962005409537313;
    tri3_xyze(2, 1) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 2) = 0.49704350575433026149;
    tri3_xyze(1, 2) = 0.042544875937232079499;
    tri3_xyze(2, 2) = 0.29154925010791188367;
    nids.push_back(-783);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49763623912943577565;
    tri3_xyze(1, 0) = 0.033973962005409537313;
    tri3_xyze(2, 0) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 1) = 0.49609463704686668217;
    tri3_xyze(1, 1) = 0.05130803379321603741;
    tri3_xyze(2, 1) = 0.29872066659990398341;
    nids.push_back(69006);
    tri3_xyze(0, 2) = 0.49704350575433026149;
    tri3_xyze(1, 2) = 0.042544875937232079499;
    tri3_xyze(2, 2) = 0.29154925010791188367;
    nids.push_back(-783);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49658472194768382701;
    tri3_xyze(1, 0) = 0.051100237715789566251;
    tri3_xyze(2, 0) = 0.28372506346233872243;
    nids.push_back(67284);
    tri3_xyze(0, 1) = 0.49785842489333492766;
    tri3_xyze(1, 1) = 0.033797270234513163145;
    tri3_xyze(2, 1) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 2) = 0.49704350575433026149;
    tri3_xyze(1, 2) = 0.042544875937232079499;
    tri3_xyze(2, 2) = 0.29154925010791188367;
    nids.push_back(-783);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65444);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 2) = 0.52449999999999996625;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1984);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 2) = 0.52449999999999996625;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1984);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65442);
    tri3_xyze(0, 2) = 0.52449999999999996625;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1984);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65446);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 2) = 0.53149999999999997247;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1985);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 2) = 0.53149999999999997247;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1985);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65444);
    tri3_xyze(0, 2) = 0.53149999999999997247;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1985);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.54199999999999992628;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65448);
    tri3_xyze(0, 1) = 0.54199999999999992628;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67170);
    tri3_xyze(0, 2) = 0.53849999999999997868;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1986);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.54199999999999992628;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67170);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 2) = 0.53849999999999997868;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1986);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65446);
    tri3_xyze(0, 2) = 0.53849999999999997868;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1986);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1991);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1991);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68880);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1991);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68880);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1991);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1992);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1992);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1992);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1992);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 2) = 0.51750000000000007105;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1993);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 2) = 0.51750000000000007105;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1993);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 2) = 0.51750000000000007105;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1993);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 2) = 0.51750000000000007105;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1993);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 2) = 0.52449999999999996625;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1994);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 2) = 0.52449999999999996625;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1994);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 2) = 0.52449999999999996625;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1994);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 2) = 0.52449999999999996625;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1994);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 2) = 0.53149999999999997247;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1995);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 2) = 0.53149999999999997247;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1995);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 2) = 0.53149999999999997247;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1995);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67166);
    tri3_xyze(0, 2) = 0.53149999999999997247;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1995);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 1) = 0.54199999999999992628;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67170);
    tri3_xyze(0, 2) = 0.53849999999999997868;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1996);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.54199999999999992628;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67170);
    tri3_xyze(0, 1) = 0.54199999999999992628;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68892);
    tri3_xyze(0, 2) = 0.53849999999999997868;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1996);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.54199999999999992628;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68892);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 2) = 0.53849999999999997868;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1996);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67168);
    tri3_xyze(0, 2) = 0.53849999999999997868;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.29249999999999998224;
    nids.push_back(-1996);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 2) = 0.49950377967816955715;
    tri3_xyze(1, 2) = 0.0084045910693904875982;
    tri3_xyze(2, 2) = 0.27736780416159245721;
    nids.push_back(-761);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 1) = 0.49895599567730553048;
    tri3_xyze(1, 1) = 0.016780582257734183438;
    tri3_xyze(2, 1) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 2) = 0.49950377967816955715;
    tri3_xyze(1, 2) = 0.0084045910693904875982;
    tri3_xyze(2, 2) = 0.27736780416159245721;
    nids.push_back(-761);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49895599567730553048;
    tri3_xyze(1, 0) = 0.016780582257734183438;
    tri3_xyze(2, 0) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 1) = 0.49905912303537275365;
    tri3_xyze(1, 1) = 0.016837782019827766955;
    tri3_xyze(2, 1) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 2) = 0.49950377967816955715;
    tri3_xyze(1, 2) = 0.0084045910693904875982;
    tri3_xyze(2, 2) = 0.27736780416159245721;
    nids.push_back(-761);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49905912303537275365;
    tri3_xyze(1, 0) = 0.016837782019827766955;
    tri3_xyze(2, 0) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 2) = 0.49950377967816955715;
    tri3_xyze(1, 2) = 0.0084045910693904875982;
    tri3_xyze(2, 2) = 0.27736780416159245721;
    nids.push_back(-761);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49905912303537275365;
    tri3_xyze(1, 0) = 0.016837782019827766955;
    tri3_xyze(2, 0) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 1) = 0.49895599567730553048;
    tri3_xyze(1, 1) = 0.016780582257734183438;
    tri3_xyze(2, 1) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 2) = 0.49850133387602196811;
    tri3_xyze(1, 2) = 0.025329134754319485023;
    tri3_xyze(2, 2) = 0.27701486996738489132;
    nids.push_back(-762);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49895599567730553048;
    tri3_xyze(1, 0) = 0.016780582257734183438;
    tri3_xyze(2, 0) = 0.28480320746087511852;
    nids.push_back(67200);
    tri3_xyze(0, 1) = 0.49785842489333492766;
    tri3_xyze(1, 1) = 0.033797270234513163145;
    tri3_xyze(2, 1) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 2) = 0.49850133387602196811;
    tri3_xyze(1, 2) = 0.025329134754319485023;
    tri3_xyze(2, 2) = 0.27701486996738489132;
    nids.push_back(-762);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49785842489333492766;
    tri3_xyze(1, 0) = 0.033797270234513163145;
    tri3_xyze(2, 0) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 1) = 0.49813179189807477165;
    tri3_xyze(1, 1) = 0.033900904505202826555;
    tri3_xyze(2, 1) = 0.26925185413339275398;
    nids.push_back(65520);
    tri3_xyze(0, 2) = 0.49850133387602196811;
    tri3_xyze(1, 2) = 0.025329134754319485023;
    tri3_xyze(2, 2) = 0.27701486996738489132;
    nids.push_back(-762);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49813179189807477165;
    tri3_xyze(1, 0) = 0.033900904505202826555;
    tri3_xyze(2, 0) = 0.26925185413339275398;
    nids.push_back(65520);
    tri3_xyze(0, 1) = 0.49905912303537275365;
    tri3_xyze(1, 1) = 0.016837782019827766955;
    tri3_xyze(2, 1) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 2) = 0.49850133387602196811;
    tri3_xyze(1, 2) = 0.025329134754319485023;
    tri3_xyze(2, 2) = 0.27701486996738489132;
    nids.push_back(-762);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49813179189807477165;
    tri3_xyze(1, 0) = 0.033900904505202826555;
    tri3_xyze(2, 0) = 0.26925185413339275398;
    nids.push_back(65520);
    tri3_xyze(0, 1) = 0.49785842489333492766;
    tri3_xyze(1, 1) = 0.033797270234513163145;
    tri3_xyze(2, 1) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 2) = 0.49741160651184246344;
    tri3_xyze(1, 2) = 0.042502867829028331825;
    tri3_xyze(2, 2) = 0.27650915621496258145;
    nids.push_back(-763);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49785842489333492766;
    tri3_xyze(1, 0) = 0.033797270234513163145;
    tri3_xyze(2, 0) = 0.28433640908977664274;
    nids.push_back(67242);
    tri3_xyze(0, 1) = 0.49658472194768382701;
    tri3_xyze(1, 1) = 0.051100237715789566251;
    tri3_xyze(2, 1) = 0.28372506346233872243;
    nids.push_back(67284);
    tri3_xyze(0, 2) = 0.49741160651184246344;
    tri3_xyze(1, 2) = 0.042502867829028331825;
    tri3_xyze(2, 2) = 0.27650915621496258145;
    nids.push_back(-763);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68880);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 2) = 0.50300183414704568996;
    tri3_xyze(1, 2) = 0.0085258161850635325041;
    tri3_xyze(2, 2) = 0.29999301787959364862;
    nids.push_back(-2601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 1) = 0.50603625638380445029;
    tri3_xyze(1, 1) = 0.017232775294719893111;
    tri3_xyze(2, 1) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 2) = 0.50300183414704568996;
    tri3_xyze(1, 2) = 0.0085258161850635325041;
    tri3_xyze(2, 2) = 0.29999301787959364862;
    nids.push_back(-2601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50603625638380445029;
    tri3_xyze(1, 0) = 0.017232775294719893111;
    tri3_xyze(2, 0) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 1) = 0.49897108020437830334;
    tri3_xyze(1, 1) = 0.016870489445534233436;
    tri3_xyze(2, 1) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 2) = 0.50300183414704568996;
    tri3_xyze(1, 2) = 0.0085258161850635325041;
    tri3_xyze(2, 2) = 0.29999301787959364862;
    nids.push_back(-2601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49897108020437830334;
    tri3_xyze(1, 0) = 0.016870489445534233436;
    tri3_xyze(2, 0) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68880);
    tri3_xyze(0, 2) = 0.50300183414704568996;
    tri3_xyze(1, 2) = 0.0085258161850635325041;
    tri3_xyze(2, 2) = 0.29999301787959364862;
    nids.push_back(-2601);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 2) = 0.51002230776534507317;
    tri3_xyze(1, 2) = 0.0086553643219371319273;
    tri3_xyze(2, 2) = 0.29997705490724324573;
    nids.push_back(-2602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 1) = 0.51305297467757604579;
    tri3_xyze(1, 1) = 0.017388681993028638068;
    tri3_xyze(2, 1) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 2) = 0.51002230776534507317;
    tri3_xyze(1, 2) = 0.0086553643219371319273;
    tri3_xyze(2, 2) = 0.29997705490724324573;
    nids.push_back(-2602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51305297467757604579;
    tri3_xyze(1, 0) = 0.017388681993028638068;
    tri3_xyze(2, 0) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 1) = 0.50603625638380445029;
    tri3_xyze(1, 1) = 0.017232775294719893111;
    tri3_xyze(2, 1) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 2) = 0.51002230776534507317;
    tri3_xyze(1, 2) = 0.0086553643219371319273;
    tri3_xyze(2, 2) = 0.29997705490724324573;
    nids.push_back(-2602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50603625638380445029;
    tri3_xyze(1, 0) = 0.017232775294719893111;
    tri3_xyze(2, 0) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68882);
    tri3_xyze(0, 2) = 0.51002230776534507317;
    tri3_xyze(1, 2) = 0.0086553643219371319273;
    tri3_xyze(2, 2) = 0.29997705490724324573;
    nids.push_back(-2602);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 2) = 0.51702533393199234801;
    tri3_xyze(1, 2) = 0.0087151868891344116963;
    tri3_xyze(2, 2) = 0.29997208147138304524;
    nids.push_back(-2603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 1) = 0.52004836105039309313;
    tri3_xyze(1, 1) = 0.017472065563509005248;
    tri3_xyze(2, 1) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 2) = 0.51702533393199234801;
    tri3_xyze(1, 2) = 0.0087151868891344116963;
    tri3_xyze(2, 2) = 0.29997208147138304524;
    nids.push_back(-2603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52004836105039309313;
    tri3_xyze(1, 0) = 0.017472065563509005248;
    tri3_xyze(2, 0) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 1) = 0.51305297467757604579;
    tri3_xyze(1, 1) = 0.017388681993028638068;
    tri3_xyze(2, 1) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 2) = 0.51702533393199234801;
    tri3_xyze(1, 2) = 0.0087151868891344116963;
    tri3_xyze(2, 2) = 0.29997208147138304524;
    nids.push_back(-2603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51305297467757604579;
    tri3_xyze(1, 0) = 0.017388681993028638068;
    tri3_xyze(2, 0) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68884);
    tri3_xyze(0, 2) = 0.51702533393199234801;
    tri3_xyze(1, 2) = 0.0087151868891344116963;
    tri3_xyze(2, 2) = 0.29997208147138304524;
    nids.push_back(-2603);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 2) = 0.52402161099485289331;
    tri3_xyze(1, 2) = 0.0087484512346500172691;
    tri3_xyze(2, 2) = 0.29997173187543335615;
    nids.push_back(-2604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 1) = 0.52703808292901843657;
    tri3_xyze(1, 1) = 0.017521739375091063828;
    tri3_xyze(2, 1) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 2) = 0.52402161099485289331;
    tri3_xyze(1, 2) = 0.0087484512346500172691;
    tri3_xyze(2, 2) = 0.29997173187543335615;
    nids.push_back(-2604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52703808292901843657;
    tri3_xyze(1, 0) = 0.017521739375091063828;
    tri3_xyze(2, 0) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 1) = 0.52004836105039309313;
    tri3_xyze(1, 1) = 0.017472065563509005248;
    tri3_xyze(2, 1) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 2) = 0.52402161099485289331;
    tri3_xyze(1, 2) = 0.0087484512346500172691;
    tri3_xyze(2, 2) = 0.29997173187543335615;
    nids.push_back(-2604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52004836105039309313;
    tri3_xyze(1, 0) = 0.017472065563509005248;
    tri3_xyze(2, 0) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68886);
    tri3_xyze(0, 2) = 0.52402161099485289331;
    tri3_xyze(1, 2) = 0.0087484512346500172691;
    tri3_xyze(2, 2) = 0.29997173187543335615;
    nids.push_back(-2604);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52800000000000002487;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 2) = 0.53101664731801723196;
    tri3_xyze(1, 2) = 0.0087700104067684053755;
    tri3_xyze(2, 2) = 0.29997400246955246983;
    nids.push_back(-2605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 1) = 0.53402850634305065736;
    tri3_xyze(1, 1) = 0.017558302251982561143;
    tri3_xyze(2, 1) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 2) = 0.53101664731801723196;
    tri3_xyze(1, 2) = 0.0087700104067684053755;
    tri3_xyze(2, 2) = 0.29997400246955246983;
    nids.push_back(-2605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53402850634305065736;
    tri3_xyze(1, 0) = 0.017558302251982561143;
    tri3_xyze(2, 0) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 1) = 0.52703808292901843657;
    tri3_xyze(1, 1) = 0.017521739375091063828;
    tri3_xyze(2, 1) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 2) = 0.53101664731801723196;
    tri3_xyze(1, 2) = 0.0087700104067684053755;
    tri3_xyze(2, 2) = 0.29997400246955246983;
    nids.push_back(-2605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52703808292901843657;
    tri3_xyze(1, 0) = 0.017521739375091063828;
    tri3_xyze(2, 0) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 1) = 0.52800000000000002487;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68888);
    tri3_xyze(0, 2) = 0.53101664731801723196;
    tri3_xyze(1, 2) = 0.0087700104067684053755;
    tri3_xyze(2, 2) = 0.29997400246955246983;
    nids.push_back(-2605);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53499999999999992006;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 1) = 0.54199999999999992628;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68892);
    tri3_xyze(0, 2) = 0.53801227497011294698;
    tri3_xyze(1, 2) = 0.008788735143553876028;
    tri3_xyze(2, 2) = 0.29997737725429968192;
    nids.push_back(-2606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.54199999999999992628;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(68892);
    tri3_xyze(0, 1) = 0.54102059353740095116;
    tri3_xyze(1, 1) = 0.017596638322232946439;
    tri3_xyze(2, 1) = 0.29995832002002476013;
    nids.push_back(68934);
    tri3_xyze(0, 2) = 0.53801227497011294698;
    tri3_xyze(1, 2) = 0.008788735143553876028;
    tri3_xyze(2, 2) = 0.29997737725429968192;
    nids.push_back(-2606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.54102059353740095116;
    tri3_xyze(1, 0) = 0.017596638322232946439;
    tri3_xyze(2, 0) = 0.29995832002002476013;
    nids.push_back(68934);
    tri3_xyze(0, 1) = 0.53402850634305065736;
    tri3_xyze(1, 1) = 0.017558302251982561143;
    tri3_xyze(2, 1) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 2) = 0.53801227497011294698;
    tri3_xyze(1, 2) = 0.008788735143553876028;
    tri3_xyze(2, 2) = 0.29997737725429968192;
    nids.push_back(-2606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53402850634305065736;
    tri3_xyze(1, 0) = 0.017558302251982561143;
    tri3_xyze(2, 0) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 1) = 0.53499999999999992006;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(68890);
    tri3_xyze(0, 2) = 0.53801227497011294698;
    tri3_xyze(1, 2) = 0.008788735143553876028;
    tri3_xyze(2, 2) = 0.29997737725429968192;
    nids.push_back(-2606);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49897108020437830334;
    tri3_xyze(1, 0) = 0.016870489445534233436;
    tri3_xyze(2, 0) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 1) = 0.50603625638380445029;
    tri3_xyze(1, 1) = 0.017232775294719893111;
    tri3_xyze(2, 1) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 2) = 0.50182082120568960448;
    tri3_xyze(1, 2) = 0.025631682175490895503;
    tri3_xyze(2, 2) = 0.29972440613942485932;
    nids.push_back(-2611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50603625638380445029;
    tri3_xyze(1, 0) = 0.017232775294719893111;
    tri3_xyze(2, 0) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 1) = 0.50463970910513999968;
    tri3_xyze(1, 1) = 0.034449501956299914684;
    tri3_xyze(2, 1) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 2) = 0.50182082120568960448;
    tri3_xyze(1, 2) = 0.025631682175490895503;
    tri3_xyze(2, 2) = 0.29972440613942485932;
    nids.push_back(-2611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50463970910513999968;
    tri3_xyze(1, 0) = 0.034449501956299914684;
    tri3_xyze(2, 0) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 1) = 0.49763623912943577565;
    tri3_xyze(1, 1) = 0.033973962005409537313;
    tri3_xyze(2, 1) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 2) = 0.50182082120568960448;
    tri3_xyze(1, 2) = 0.025631682175490895503;
    tri3_xyze(2, 2) = 0.29972440613942485932;
    nids.push_back(-2611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49763623912943577565;
    tri3_xyze(1, 0) = 0.033973962005409537313;
    tri3_xyze(2, 0) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 1) = 0.49897108020437830334;
    tri3_xyze(1, 1) = 0.016870489445534233436;
    tri3_xyze(2, 1) = 0.30001007115423650173;
    nids.push_back(68922);
    tri3_xyze(0, 2) = 0.50182082120568960448;
    tri3_xyze(1, 2) = 0.025631682175490895503;
    tri3_xyze(2, 2) = 0.29972440613942485932;
    nids.push_back(-2611);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50603625638380445029;
    tri3_xyze(1, 0) = 0.017232775294719893111;
    tri3_xyze(2, 0) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 1) = 0.51305297467757604579;
    tri3_xyze(1, 1) = 0.017388681993028638068;
    tri3_xyze(2, 1) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 2) = 0.50884520732417048983;
    tri3_xyze(1, 2) = 0.025953841810643980825;
    tri3_xyze(2, 2) = 0.29975619146992216191;
    nids.push_back(-2612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51305297467757604579;
    tri3_xyze(1, 0) = 0.017388681993028638068;
    tri3_xyze(2, 0) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 1) = 0.51165188913016157457;
    tri3_xyze(1, 1) = 0.03474440799852746703;
    tri3_xyze(2, 1) = 0.29960585449101889699;
    nids.push_back(68968);
    tri3_xyze(0, 2) = 0.50884520732417048983;
    tri3_xyze(1, 2) = 0.025953841810643980825;
    tri3_xyze(2, 2) = 0.29975619146992216191;
    nids.push_back(-2612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51165188913016157457;
    tri3_xyze(1, 0) = 0.03474440799852746703;
    tri3_xyze(2, 0) = 0.29960585449101889699;
    nids.push_back(68968);
    tri3_xyze(0, 1) = 0.50463970910513999968;
    tri3_xyze(1, 1) = 0.034449501956299914684;
    tri3_xyze(2, 1) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 2) = 0.50884520732417048983;
    tri3_xyze(1, 2) = 0.025953841810643980825;
    tri3_xyze(2, 2) = 0.29975619146992216191;
    nids.push_back(-2612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50463970910513999968;
    tri3_xyze(1, 0) = 0.034449501956299914684;
    tri3_xyze(2, 0) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 1) = 0.50603625638380445029;
    tri3_xyze(1, 1) = 0.017232775294719893111;
    tri3_xyze(2, 1) = 0.29996200036413822598;
    nids.push_back(68924);
    tri3_xyze(0, 2) = 0.50884520732417048983;
    tri3_xyze(1, 2) = 0.025953841810643980825;
    tri3_xyze(2, 2) = 0.29975619146992216191;
    nids.push_back(-2612);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51305297467757604579;
    tri3_xyze(1, 0) = 0.017388681993028638068;
    tri3_xyze(2, 0) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 1) = 0.52004836105039309313;
    tri3_xyze(1, 1) = 0.017472065563509005248;
    tri3_xyze(2, 1) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 2) = 0.51585255654870332265;
    tri3_xyze(1, 2) = 0.026131316124275202895;
    tri3_xyze(2, 2) = 0.2998002351047138192;
    nids.push_back(-2613);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52004836105039309313;
    tri3_xyze(1, 0) = 0.017472065563509005248;
    tri3_xyze(2, 0) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 1) = 0.51865700133668224403;
    tri3_xyze(1, 1) = 0.034920108942035690824;
    tri3_xyze(2, 1) = 0.29970676004230406564;
    nids.push_back(68970);
    tri3_xyze(0, 2) = 0.51585255654870332265;
    tri3_xyze(1, 2) = 0.026131316124275202895;
    tri3_xyze(2, 2) = 0.2998002351047138192;
    nids.push_back(-2613);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51165188913016157457;
    tri3_xyze(1, 0) = 0.03474440799852746703;
    tri3_xyze(2, 0) = 0.29960585449101889699;
    nids.push_back(68968);
    tri3_xyze(0, 1) = 0.51305297467757604579;
    tri3_xyze(1, 1) = 0.017388681993028638068;
    tri3_xyze(2, 1) = 0.29994621926483494567;
    nids.push_back(68926);
    tri3_xyze(0, 2) = 0.51585255654870332265;
    tri3_xyze(1, 2) = 0.026131316124275202895;
    tri3_xyze(2, 2) = 0.2998002351047138192;
    nids.push_back(-2613);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52703808292901843657;
    tri3_xyze(1, 0) = 0.017521739375091063828;
    tri3_xyze(2, 0) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 1) = 0.52565439337195518643;
    tri3_xyze(1, 1) = 0.035032236939279724763;
    tri3_xyze(2, 1) = 0.29981098620342461203;
    nids.push_back(68972);
    tri3_xyze(0, 2) = 0.52284945967201235106;
    tri3_xyze(1, 2) = 0.026236537704978871166;
    tri3_xyze(2, 2) = 0.29985116843686548949;
    nids.push_back(-2614);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51865700133668224403;
    tri3_xyze(1, 0) = 0.034920108942035690824;
    tri3_xyze(2, 0) = 0.29970676004230406564;
    nids.push_back(68970);
    tri3_xyze(0, 1) = 0.52004836105039309313;
    tri3_xyze(1, 1) = 0.017472065563509005248;
    tri3_xyze(2, 1) = 0.29994210662069731299;
    nids.push_back(68928);
    tri3_xyze(0, 2) = 0.52284945967201235106;
    tri3_xyze(1, 2) = 0.026236537704978871166;
    tri3_xyze(2, 2) = 0.29985116843686548949;
    nids.push_back(-2614);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52703808292901843657;
    tri3_xyze(1, 0) = 0.017521739375091063828;
    tri3_xyze(2, 0) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 1) = 0.53402850634305065736;
    tri3_xyze(1, 1) = 0.017558302251982561143;
    tri3_xyze(2, 1) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 2) = 0.52984168926180008619;
    tri3_xyze(1, 2) = 0.026307416092189318813;
    tri3_xyze(2, 2) = 0.29990600082100837831;
    nids.push_back(-2615);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53402850634305065736;
    tri3_xyze(1, 0) = 0.017558302251982561143;
    tri3_xyze(2, 0) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 1) = 0.53264577440317606438;
    tri3_xyze(1, 1) = 0.035117385802403935924;
    tri3_xyze(2, 1) = 0.29991700720239877764;
    nids.push_back(68974);
    tri3_xyze(0, 2) = 0.52984168926180008619;
    tri3_xyze(1, 2) = 0.026307416092189318813;
    tri3_xyze(2, 2) = 0.29990600082100837831;
    nids.push_back(-2615);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52565439337195518643;
    tri3_xyze(1, 0) = 0.035032236939279724763;
    tri3_xyze(2, 0) = 0.29981098620342461203;
    nids.push_back(68972);
    tri3_xyze(0, 1) = 0.52703808292901843657;
    tri3_xyze(1, 1) = 0.017521739375091063828;
    tri3_xyze(2, 1) = 0.2999448208810360228;
    nids.push_back(68930);
    tri3_xyze(0, 2) = 0.52984168926180008619;
    tri3_xyze(1, 2) = 0.026307416092189318813;
    tri3_xyze(2, 2) = 0.29990600082100837831;
    nids.push_back(-2615);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53402850634305065736;
    tri3_xyze(1, 0) = 0.017558302251982561143;
    tri3_xyze(2, 0) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 1) = 0.54102059353740095116;
    tri3_xyze(1, 1) = 0.017596638322232946439;
    tri3_xyze(2, 1) = 0.29995832002002476013;
    nids.push_back(68934);
    tri3_xyze(0, 2) = 0.5368314846705304344;
    tri3_xyze(1, 2) = 0.026369283298627508444;
    tri3_xyze(2, 2) = 0.29996251189192779663;
    nids.push_back(-2616);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.54102059353740095116;
    tri3_xyze(1, 0) = 0.017596638322232946439;
    tri3_xyze(2, 0) = 0.29995832002002476013;
    nids.push_back(68934);
    tri3_xyze(0, 1) = 0.53963106439849417573;
    tri3_xyze(1, 1) = 0.03520480681789059374;
    tri3_xyze(2, 1) = 0.30002353134811393653;
    nids.push_back(68976);
    tri3_xyze(0, 2) = 0.5368314846705304344;
    tri3_xyze(1, 2) = 0.026369283298627508444;
    tri3_xyze(2, 2) = 0.29996251189192779663;
    nids.push_back(-2616);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.53264577440317606438;
    tri3_xyze(1, 0) = 0.035117385802403935924;
    tri3_xyze(2, 0) = 0.29991700720239877764;
    nids.push_back(68974);
    tri3_xyze(0, 1) = 0.53402850634305065736;
    tri3_xyze(1, 1) = 0.017558302251982561143;
    tri3_xyze(2, 1) = 0.29995118899717393424;
    nids.push_back(68932);
    tri3_xyze(0, 2) = 0.5368314846705304344;
    tri3_xyze(1, 2) = 0.026369283298627508444;
    tri3_xyze(2, 2) = 0.29996251189192779663;
    nids.push_back(-2616);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49763623912943577565;
    tri3_xyze(1, 0) = 0.033973962005409537313;
    tri3_xyze(2, 0) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 1) = 0.50463970910513999968;
    tri3_xyze(1, 1) = 0.034449501956299914684;
    tri3_xyze(2, 1) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 2) = 0.50035979183841750029;
    tri3_xyze(1, 2) = 0.042887989687440078446;
    tri3_xyze(2, 2) = 0.29914621550005854322;
    nids.push_back(-2621);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50463970910513999968;
    tri3_xyze(1, 0) = 0.034449501956299914684;
    tri3_xyze(2, 0) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 1) = 0.50306858207222748813;
    tri3_xyze(1, 1) = 0.051820460994834845192;
    tri3_xyze(2, 1) = 0.29893864236100553544;
    nids.push_back(69008);
    tri3_xyze(0, 2) = 0.50035979183841750029;
    tri3_xyze(1, 2) = 0.042887989687440078446;
    tri3_xyze(2, 2) = 0.29914621550005854322;
    nids.push_back(-2621);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.25500000000000000444;
    nids.push_back(63714);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 2) = 0.49956274721146110007;
    tri3_xyze(1, 2) = 0.0084426273436703693637;
    tri3_xyze(2, 2) = 0.26231746261470645365;
    nids.push_back(-741);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 1) = 0.49905912303537275365;
    tri3_xyze(1, 1) = 0.016837782019827766955;
    tri3_xyze(2, 1) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 2) = 0.49956274721146110007;
    tri3_xyze(1, 2) = 0.0084426273436703693637;
    tri3_xyze(2, 2) = 0.26231746261470645365;
    nids.push_back(-741);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49905912303537275365;
    tri3_xyze(1, 0) = 0.016837782019827766955;
    tri3_xyze(2, 0) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 1) = 0.49919186581047170215;
    tri3_xyze(1, 1) = 0.0169327273548537105;
    tri3_xyze(2, 1) = 0.25460184127333085335;
    nids.push_back(63756);
    tri3_xyze(0, 2) = 0.49956274721146110007;
    tri3_xyze(1, 2) = 0.0084426273436703693637;
    tri3_xyze(2, 2) = 0.26231746261470645365;
    nids.push_back(-741);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49919186581047170215;
    tri3_xyze(1, 0) = 0.0169327273548537105;
    tri3_xyze(2, 0) = 0.25460184127333085335;
    nids.push_back(63756);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.25500000000000000444;
    nids.push_back(63714);
    tri3_xyze(0, 2) = 0.49956274721146110007;
    tri3_xyze(1, 2) = 0.0084426273436703693637;
    tri3_xyze(2, 2) = 0.26231746261470645365;
    nids.push_back(-741);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49919186581047170215;
    tri3_xyze(1, 0) = 0.0169327273548537105;
    tri3_xyze(2, 0) = 0.25460184127333085335;
    nids.push_back(63756);
    tri3_xyze(0, 1) = 0.49905912303537275365;
    tri3_xyze(1, 1) = 0.016837782019827766955;
    tri3_xyze(2, 1) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 2) = 0.49870140737636137196;
    tri3_xyze(1, 2) = 0.025435761906038446833;
    tri3_xyze(2, 2) = 0.26193192756903449503;
    nids.push_back(-742);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49905912303537275365;
    tri3_xyze(1, 0) = 0.016837782019827766955;
    tri3_xyze(2, 0) = 0.26966800918549493904;
    nids.push_back(65478);
    tri3_xyze(0, 1) = 0.49813179189807477165;
    tri3_xyze(1, 1) = 0.033900904505202826555;
    tri3_xyze(2, 1) = 0.26925185413339275398;
    nids.push_back(65520);
    tri3_xyze(0, 2) = 0.49870140737636137196;
    tri3_xyze(1, 2) = 0.025435761906038446833;
    tri3_xyze(2, 2) = 0.26193192756903449503;
    nids.push_back(-742);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.49609463704686668217;
    tri3_xyze(1, 0) = 0.05130803379321603741;
    tri3_xyze(2, 0) = 0.29872066659990398341;
    nids.push_back(69006);
    tri3_xyze(0, 1) = 0.49763623912943577565;
    tri3_xyze(1, 1) = 0.033973962005409537313;
    tri3_xyze(2, 1) = 0.29941486127962807506;
    nids.push_back(68964);
    tri3_xyze(0, 2) = 0.50035979183841750029;
    tri3_xyze(1, 2) = 0.042887989687440078446;
    tri3_xyze(2, 2) = 0.29914621550005854322;
    nids.push_back(-2621);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50463970910513999968;
    tri3_xyze(1, 0) = 0.034449501956299914684;
    tri3_xyze(2, 0) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 1) = 0.51165188913016157457;
    tri3_xyze(1, 1) = 0.03474440799852746703;
    tri3_xyze(2, 1) = 0.29960585449101889699;
    nids.push_back(68968);
    tri3_xyze(0, 2) = 0.50735362138744655169;
    tri3_xyze(1, 2) = 0.043292676686574035894;
    tri3_xyze(2, 2) = 0.29930503017945142563;
    nids.push_back(-2622);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51165188913016157457;
    tri3_xyze(1, 0) = 0.03474440799852746703;
    tri3_xyze(2, 0) = 0.29960585449101889699;
    nids.push_back(68968);
    tri3_xyze(0, 1) = 0.51005430524225714439;
    tri3_xyze(1, 1) = 0.052156335796633916668;
    tri3_xyze(2, 1) = 0.29916493210608452458;
    nids.push_back(69010);
    tri3_xyze(0, 2) = 0.50735362138744655169;
    tri3_xyze(1, 2) = 0.043292676686574035894;
    tri3_xyze(2, 2) = 0.29930503017945142563;
    nids.push_back(-2622);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50306858207222748813;
    tri3_xyze(1, 0) = 0.051820460994834845192;
    tri3_xyze(2, 0) = 0.29893864236100553544;
    nids.push_back(69008);
    tri3_xyze(0, 1) = 0.50463970910513999968;
    tri3_xyze(1, 1) = 0.034449501956299914684;
    tri3_xyze(2, 1) = 0.29951069175969669001;
    nids.push_back(68966);
    tri3_xyze(0, 2) = 0.50735362138744655169;
    tri3_xyze(1, 2) = 0.043292676686574035894;
    tri3_xyze(2, 2) = 0.29930503017945142563;
    nids.push_back(-2622);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.25500000000000000444;
    nids.push_back(63714);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.25500000000000000444;
    nids.push_back(63716);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.2625000000000000111;
    nids.push_back(-1971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.25500000000000000444;
    nids.push_back(63716);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.2625000000000000111;
    nids.push_back(-1971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.2625000000000000111;
    nids.push_back(-1971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.25500000000000000444;
    nids.push_back(63714);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.2625000000000000111;
    nids.push_back(-1971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.25500000000000000444;
    nids.push_back(63716);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.25500000000000000444;
    nids.push_back(63718);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.2625000000000000111;
    nids.push_back(-1972);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65440);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.2625000000000000111;
    nids.push_back(-1972);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.25500000000000000444;
    nids.push_back(63716);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.2625000000000000111;
    nids.push_back(-1972);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.5;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67158);
    tri3_xyze(0, 1) = 0.5;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65436);
    tri3_xyze(0, 2) = 0.50350000000000005862;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65440);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65440);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.50700000000000000622;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67160);
    tri3_xyze(0, 1) = 0.50700000000000000622;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65438);
    tri3_xyze(0, 2) = 0.51049999999999995381;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.26999999999999996225;
    nids.push_back(65442);
    tri3_xyze(0, 1) = 0.52100000000000001865;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 2) = 0.51750000000000007105;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1983);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.52100000000000001865;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67164);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 2) = 0.51750000000000007105;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1983);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.51400000000000001243;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.28499999999999997558;
    nids.push_back(67162);
    tri3_xyze(0, 1) = 0.51400000000000001243;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.26999999999999996225;
    nids.push_back(65440);
    tri3_xyze(0, 2) = 0.51750000000000007105;
    tri3_xyze(1, 2) = 0;
    tri3_xyze(2, 2) = 0.27749999999999996891;
    nids.push_back(-1983);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.47999999999999998224;
    hex8_xyze(1, 0) = 0;
    hex8_xyze(2, 0) = 0.26000000000000000888;
    nids.push_back(704539);
    hex8_xyze(0, 1) = 0.51000000000000000888;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.26000000000000000888;
    nids.push_back(704541);
    hex8_xyze(0, 2) = 0.51000000000000000888;
    hex8_xyze(1, 2) = 0.020000000000000000416;
    hex8_xyze(2, 2) = 0.26000000000000000888;
    nids.push_back(704783);
    hex8_xyze(0, 3) = 0.47999999999999998224;
    hex8_xyze(1, 3) = 0.020000000000000000416;
    hex8_xyze(2, 3) = 0.26000000000000000888;
    nids.push_back(704781);
    hex8_xyze(0, 4) = 0.47999999999999998224;
    hex8_xyze(1, 4) = 0;
    hex8_xyze(2, 4) = 0.27999999999999991562;
    nids.push_back(719301);
    hex8_xyze(0, 5) = 0.51000000000000000888;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.27999999999999991562;
    nids.push_back(719303);
    hex8_xyze(0, 6) = 0.51000000000000000888;
    hex8_xyze(1, 6) = 0.020000000000000000416;
    hex8_xyze(2, 6) = 0.27999999999999991562;
    nids.push_back(719545);
    hex8_xyze(0, 7) = 0.47999999999999998224;
    hex8_xyze(1, 7) = 0.020000000000000000416;
    hex8_xyze(2, 7) = 0.27999999999999991562;
    nids.push_back(719543);

    intersection.add_element(77416, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.51000000000000000888;
    hex8_xyze(1, 0) = 0;
    hex8_xyze(2, 0) = 0.27999999999999991562;
    nids.push_back(719303);
    hex8_xyze(0, 1) = 0.54000000000000003553;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.27999999999999991562;
    nids.push_back(719305);
    hex8_xyze(0, 2) = 0.54000000000000003553;
    hex8_xyze(1, 2) = 0.020000000000000000416;
    hex8_xyze(2, 2) = 0.27999999999999991562;
    nids.push_back(719547);
    hex8_xyze(0, 3) = 0.51000000000000000888;
    hex8_xyze(1, 3) = 0.020000000000000000416;
    hex8_xyze(2, 3) = 0.27999999999999991562;
    nids.push_back(719545);
    hex8_xyze(0, 4) = 0.51000000000000000888;
    hex8_xyze(1, 4) = 0;
    hex8_xyze(2, 4) = 0.2999999999999999889;
    nids.push_back(734065);
    hex8_xyze(0, 5) = 0.54000000000000003553;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.2999999999999999889;
    nids.push_back(734067);
    hex8_xyze(0, 6) = 0.54000000000000003553;
    hex8_xyze(1, 6) = 0.020000000000000000416;
    hex8_xyze(2, 6) = 0.2999999999999999889;
    nids.push_back(734309);
    hex8_xyze(0, 7) = 0.51000000000000000888;
    hex8_xyze(1, 7) = 0.020000000000000000416;
    hex8_xyze(2, 7) = 0.2999999999999999889;
    nids.push_back(734307);

    intersection.add_element(79217, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.47999999999999998224;
    hex8_xyze(1, 0) = 0.020000000000000000416;
    hex8_xyze(2, 0) = 0.27999999999999991562;
    nids.push_back(719543);
    hex8_xyze(0, 1) = 0.51000000000000000888;
    hex8_xyze(1, 1) = 0.020000000000000000416;
    hex8_xyze(2, 1) = 0.27999999999999991562;
    nids.push_back(719545);
    hex8_xyze(0, 2) = 0.51000000000000000888;
    hex8_xyze(1, 2) = 0.040000000000000000833;
    hex8_xyze(2, 2) = 0.27999999999999991562;
    nids.push_back(719787);
    hex8_xyze(0, 3) = 0.47999999999999998224;
    hex8_xyze(1, 3) = 0.040000000000000000833;
    hex8_xyze(2, 3) = 0.27999999999999991562;
    nids.push_back(719785);
    hex8_xyze(0, 4) = 0.47999999999999998224;
    hex8_xyze(1, 4) = 0.020000000000000000416;
    hex8_xyze(2, 4) = 0.2999999999999999889;
    nids.push_back(734305);
    hex8_xyze(0, 5) = 0.51000000000000000888;
    hex8_xyze(1, 5) = 0.020000000000000000416;
    hex8_xyze(2, 5) = 0.2999999999999999889;
    nids.push_back(734307);
    hex8_xyze(0, 6) = 0.51000000000000000888;
    hex8_xyze(1, 6) = 0.040000000000000000833;
    hex8_xyze(2, 6) = 0.2999999999999999889;
    nids.push_back(734549);
    hex8_xyze(0, 7) = 0.47999999999999998224;
    hex8_xyze(1, 7) = 0.040000000000000000833;
    hex8_xyze(2, 7) = 0.2999999999999999889;
    nids.push_back(734547);

    intersection.add_element(79276, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.47999999999999998224;
    hex8_xyze(1, 0) = 0;
    hex8_xyze(2, 0) = 0.2999999999999999889;
    nids.push_back(734063);
    hex8_xyze(0, 1) = 0.51000000000000000888;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.2999999999999999889;
    nids.push_back(734065);
    hex8_xyze(0, 2) = 0.51000000000000000888;
    hex8_xyze(1, 2) = 0.020000000000000000416;
    hex8_xyze(2, 2) = 0.2999999999999999889;
    nids.push_back(734307);
    hex8_xyze(0, 3) = 0.47999999999999998224;
    hex8_xyze(1, 3) = 0.020000000000000000416;
    hex8_xyze(2, 3) = 0.2999999999999999889;
    nids.push_back(734305);
    hex8_xyze(0, 4) = 0.47999999999999998224;
    hex8_xyze(1, 4) = 0;
    hex8_xyze(2, 4) = 0.32000000000000006217;
    nids.push_back(748825);
    hex8_xyze(0, 5) = 0.51000000000000000888;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.32000000000000006217;
    nids.push_back(748827);
    hex8_xyze(0, 6) = 0.51000000000000000888;
    hex8_xyze(1, 6) = 0.020000000000000000416;
    hex8_xyze(2, 6) = 0.32000000000000006217;
    nids.push_back(749069);
    hex8_xyze(0, 7) = 0.47999999999999998224;
    hex8_xyze(1, 7) = 0.020000000000000000416;
    hex8_xyze(2, 7) = 0.32000000000000006217;
    nids.push_back(749067);

    intersection.add_element(81016, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.47999999999999998224;
    hex8_xyze(1, 0) = 0;
    hex8_xyze(2, 0) = 0.27999999999999991562;
    nids.push_back(719301);
    hex8_xyze(0, 1) = 0.51000000000000000888;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.27999999999999991562;
    nids.push_back(719303);
    hex8_xyze(0, 2) = 0.51000000000000000888;
    hex8_xyze(1, 2) = 0.020000000000000000416;
    hex8_xyze(2, 2) = 0.27999999999999991562;
    nids.push_back(719545);
    hex8_xyze(0, 3) = 0.47999999999999998224;
    hex8_xyze(1, 3) = 0.020000000000000000416;
    hex8_xyze(2, 3) = 0.27999999999999991562;
    nids.push_back(719543);
    hex8_xyze(0, 4) = 0.47999999999999998224;
    hex8_xyze(1, 4) = 0;
    hex8_xyze(2, 4) = 0.2999999999999999889;
    nids.push_back(734063);
    hex8_xyze(0, 5) = 0.51000000000000000888;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.2999999999999999889;
    nids.push_back(734065);
    hex8_xyze(0, 6) = 0.51000000000000000888;
    hex8_xyze(1, 6) = 0.020000000000000000416;
    hex8_xyze(2, 6) = 0.2999999999999999889;
    nids.push_back(734307);
    hex8_xyze(0, 7) = 0.47999999999999998224;
    hex8_xyze(1, 7) = 0.020000000000000000416;
    hex8_xyze(2, 7) = 0.2999999999999999889;
    nids.push_back(734305);

    intersection.add_element(79216, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
