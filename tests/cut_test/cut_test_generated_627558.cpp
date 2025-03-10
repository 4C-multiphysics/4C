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

void test_generated_627558()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11320);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826084253;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69025);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = 0.15860869565217394772;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11196);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(68289);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = 0.15860869565217394772;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11196);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(68289);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11198);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 1) = 1.4891304347826088694;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69057);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11198);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826084253;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69025);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69777);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11318);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69777);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11318);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11318);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 1) = 1.4891304347826084253;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69025);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11318);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69809);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11320);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826088694;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69057);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69041);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.020000000000000000416;
    nids.push_back(-11320);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608689104;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69760);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608689104;
    tri3_xyze(2, 1) = 0.014666666666666664659;
    nids.push_back(69762);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11892);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608689104;
    tri3_xyze(2, 0) = 0.014666666666666664659;
    nids.push_back(69762);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347829917;
    tri3_xyze(2, 1) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11892);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347829917;
    tri3_xyze(2, 0) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69776);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11892);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69776);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608689104;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69760);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11892);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608689104;
    tri3_xyze(2, 0) = 0.014666666666666664659;
    nids.push_back(69762);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608694655;
    tri3_xyze(2, 1) = 0.01200000000000000025;
    nids.push_back(69763);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.013333333333333332454;
    nids.push_back(-11893);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608694655;
    tri3_xyze(2, 0) = 0.01200000000000000025;
    nids.push_back(69763);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.1531739130434782159;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.013333333333333332454;
    nids.push_back(-11893);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.1531739130434782159;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347829917;
    tri3_xyze(2, 1) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.013333333333333332454;
    nids.push_back(-11893);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347829917;
    tri3_xyze(2, 0) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608689104;
    tri3_xyze(2, 1) = 0.014666666666666664659;
    nids.push_back(69762);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.013333333333333332454;
    nids.push_back(-11893);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608694655;
    tri3_xyze(2, 0) = 0.01200000000000000025;
    nids.push_back(69763);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608689104;
    tri3_xyze(2, 1) = 0.0093333333333333323711;
    nids.push_back(69764);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.010666666666666668045;
    nids.push_back(-11894);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608689104;
    tri3_xyze(2, 0) = 0.0093333333333333323711;
    nids.push_back(69764);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347824366;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.010666666666666668045;
    nids.push_back(-11894);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347824366;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.1531739130434782159;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.010666666666666668045;
    nids.push_back(-11894);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.1531739130434782159;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608694655;
    tri3_xyze(2, 1) = 0.01200000000000000025;
    nids.push_back(69763);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.010666666666666668045;
    nids.push_back(-11894);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608689104;
    tri3_xyze(2, 0) = 0.0093333333333333323711;
    nids.push_back(69764);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608691879;
    tri3_xyze(2, 1) = 0.0066666666666666679619;
    nids.push_back(69765);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11895);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608691879;
    tri3_xyze(2, 0) = 0.0066666666666666679619;
    nids.push_back(69765);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11895);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347824366;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11895);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347824366;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608689104;
    tri3_xyze(2, 1) = 0.0093333333333333323711;
    nids.push_back(69764);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11895);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.14230434782608691879;
    tri3_xyze(2, 0) = 0.0066666666666666679619;
    nids.push_back(69765);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608694655;
    tri3_xyze(2, 1) = 0.0039999999999999983485;
    nids.push_back(69766);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.0053333333333333322879;
    nids.push_back(-11896);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0039999999999999983485;
    nids.push_back(69782);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.0053333333333333322879;
    nids.push_back(-11896);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.14230434782608691879;
    tri3_xyze(2, 1) = 0.0066666666666666679619;
    nids.push_back(69765);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.1477391304347825951;
    tri3_xyze(2, 2) = 0.0053333333333333322879;
    nids.push_back(-11896);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69777);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69776);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.018666666666666668212;
    nids.push_back(-11906);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69776);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.018666666666666668212;
    nids.push_back(-11906);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.018666666666666668212;
    nids.push_back(-11906);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69777);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.018666666666666668212;
    nids.push_back(-11906);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69776);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347829917;
    tri3_xyze(2, 1) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11907);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347829917;
    tri3_xyze(2, 0) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11907);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11907);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69776);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11907);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347829917;
    tri3_xyze(2, 0) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.1531739130434782159;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.013333333333333334189;
    nids.push_back(-11908);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.1531739130434782159;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.013333333333333334189;
    nids.push_back(-11908);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.013333333333333334189;
    nids.push_back(-11908);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347829917;
    tri3_xyze(2, 1) = 0.014666666666666662924;
    nids.push_back(69778);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.013333333333333334189;
    nids.push_back(-11908);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.1531739130434782159;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347824366;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11909);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347824366;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11909);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11909);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.1531739130434782159;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69779);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11909);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347824366;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11910);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11910);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11910);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347824366;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69780);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11910);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0039999999999999983485;
    nids.push_back(69782);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11911);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0039999999999999983485;
    nids.push_back(69782);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11911);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11911);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69781);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217389221;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11911);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0039999999999999983485;
    nids.push_back(69782);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0013333333333333356741;
    nids.push_back(69783);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0026666666666666670113;
    nids.push_back(-11912);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0013333333333333356741;
    nids.push_back(69783);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0026666666666666670113;
    nids.push_back(-11912);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0026666666666666670113;
    nids.push_back(-11912);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0039999999999999983485;
    nids.push_back(69782);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 0.0026666666666666670113;
    nids.push_back(-11912);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = 0.0013333333333333356741;
    nids.push_back(69783);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = -0.0013333333333333339393;
    nids.push_back(69784);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 4.336808689942017736e-19;
    nids.push_back(-11913);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = -0.0013333333333333339393;
    nids.push_back(69784);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 4.336808689942017736e-19;
    nids.push_back(-11913);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 4.336808689942017736e-19;
    nids.push_back(-11913);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = 0.0013333333333333356741;
    nids.push_back(69783);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = 4.336808689942017736e-19;
    nids.push_back(-11913);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347827141;
    tri3_xyze(2, 0) = -0.0013333333333333339393;
    nids.push_back(69784);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347824366;
    tri3_xyze(2, 1) = -0.0040000000000000035527;
    nids.push_back(69785);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.002666666666666667445;
    nids.push_back(-11914);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347824366;
    tri3_xyze(2, 0) = -0.0040000000000000035527;
    nids.push_back(69785);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.002666666666666667445;
    nids.push_back(-11914);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.002666666666666667445;
    nids.push_back(-11914);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347827141;
    tri3_xyze(2, 1) = -0.0013333333333333339393;
    nids.push_back(69784);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.002666666666666667445;
    nids.push_back(-11914);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347824366;
    tri3_xyze(2, 0) = -0.0040000000000000035527;
    nids.push_back(69785);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347829917;
    tri3_xyze(2, 1) = -0.0066666666666666662272;
    nids.push_back(69786);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.0053333333333333322879;
    nids.push_back(-11915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.15317391304347829917;
    tri3_xyze(2, 0) = -0.0066666666666666662272;
    nids.push_back(69786);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0066666666666666644925;
    nids.push_back(69802);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.0053333333333333322879;
    nids.push_back(-11915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0066666666666666644925;
    nids.push_back(69802);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.0053333333333333322879;
    nids.push_back(-11915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.15317391304347824366;
    tri3_xyze(2, 1) = -0.0040000000000000035527;
    nids.push_back(69785);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.15860869565217391997;
    tri3_xyze(2, 2) = -0.0053333333333333322879;
    nids.push_back(-11915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.018666666666666671681;
    nids.push_back(-11921);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826083787;
    tri3_xyze(2, 1) = 0.017333333333333342946;
    nids.push_back(69808);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.018666666666666671681;
    nids.push_back(-11921);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.020000000000000000416;
    nids.push_back(69809);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.020000000000000000416;
    nids.push_back(69793);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.018666666666666671681;
    nids.push_back(-11921);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.016000000000000003803;
    nids.push_back(-11922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.016000000000000003803;
    nids.push_back(-11922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826083787;
    tri3_xyze(2, 1) = 0.017333333333333342946;
    nids.push_back(69808);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.016000000000000003803;
    nids.push_back(-11922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826083787;
    tri3_xyze(2, 0) = 0.017333333333333342946;
    nids.push_back(69808);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.017333333333333336007;
    nids.push_back(69792);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.016000000000000003803;
    nids.push_back(-11922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086959627;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.013333333333333335924;
    nids.push_back(-11923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.013333333333333335924;
    nids.push_back(-11923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.013333333333333335924;
    nids.push_back(-11923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086959627;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69794);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.013333333333333335924;
    nids.push_back(-11923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.012000000000000003719;
    nids.push_back(69795);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.01066666666666666978;
    nids.push_back(-11924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0080000000000000019013;
    nids.push_back(-11925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0080000000000000019013;
    nids.push_back(-11925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0080000000000000019013;
    nids.push_back(-11925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69796);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0080000000000000019013;
    nids.push_back(-11925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0040000000000000000833;
    nids.push_back(69814);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0040000000000000000833;
    nids.push_back(69814);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0066666666666666696967;
    nids.push_back(69797);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0053333333333333340226;
    nids.push_back(-11926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086954076;
    tri3_xyze(2, 0) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0026666666666666670113;
    nids.push_back(-11927);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0013333333333333339393;
    nids.push_back(69815);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0026666666666666670113;
    nids.push_back(-11927);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0040000000000000000833;
    nids.push_back(69814);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086954076;
    tri3_xyze(2, 1) = 0.0040000000000000000833;
    nids.push_back(69798);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956521707;
    tri3_xyze(2, 2) = 0.0026666666666666670113;
    nids.push_back(-11927);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956518932;
    tri3_xyze(2, 2) = 4.336808689942017736e-19;
    nids.push_back(-11928);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826081012;
    tri3_xyze(2, 1) = -0.0013333333333333322046;
    nids.push_back(69816);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956518932;
    tri3_xyze(2, 2) = 4.336808689942017736e-19;
    nids.push_back(-11928);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0013333333333333339393;
    nids.push_back(69815);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = 0.0013333333333333339393;
    nids.push_back(69799);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956518932;
    tri3_xyze(2, 2) = 4.336808689942017736e-19;
    nids.push_back(-11928);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = -0.0026666666666666657103;
    nids.push_back(-11929);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826092114;
    tri3_xyze(2, 1) = -0.0039999999999999983485;
    nids.push_back(69817);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = -0.0026666666666666657103;
    nids.push_back(-11929);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826081012;
    tri3_xyze(2, 0) = -0.0013333333333333322046;
    nids.push_back(69816);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0013333333333333339393;
    nids.push_back(69800);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = -0.0026666666666666657103;
    nids.push_back(-11929);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0066666666666666644925;
    nids.push_back(69802);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = -0.0053333333333333322879;
    nids.push_back(-11930);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.16404347826086956852;
    tri3_xyze(2, 0) = -0.0066666666666666644925;
    nids.push_back(69802);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826083787;
    tri3_xyze(2, 1) = -0.0066666666666666679619;
    nids.push_back(69818);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = -0.0053333333333333322879;
    nids.push_back(-11930);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826092114;
    tri3_xyze(2, 0) = -0.0039999999999999983485;
    nids.push_back(69817);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.16404347826086956852;
    tri3_xyze(2, 1) = -0.0039999999999999983485;
    nids.push_back(69801);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.16947826086956524483;
    tri3_xyze(2, 2) = -0.0053333333333333322879;
    nids.push_back(-11930);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826083787;
    tri3_xyze(2, 0) = 0.017333333333333342946;
    nids.push_back(69808);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11937);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.18578260869565216273;
    tri3_xyze(2, 1) = 0.014666666666666664659;
    nids.push_back(69826);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11937);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.18578260869565219049;
    tri3_xyze(2, 0) = 0.017333333333333332538;
    nids.push_back(69824);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826083787;
    tri3_xyze(2, 1) = 0.017333333333333342946;
    nids.push_back(69808);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.016000000000000000333;
    nids.push_back(-11937);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.013333333333333332454;
    nids.push_back(-11938);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.18578260869565219049;
    tri3_xyze(2, 1) = 0.01200000000000000025;
    nids.push_back(69827);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.013333333333333332454;
    nids.push_back(-11938);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.18578260869565216273;
    tri3_xyze(2, 0) = 0.014666666666666664659;
    nids.push_back(69826);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.014666666666666666394;
    nids.push_back(69810);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.013333333333333332454;
    nids.push_back(-11938);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.010666666666666668045;
    nids.push_back(-11939);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.18578260869565221824;
    tri3_xyze(2, 1) = 0.0093333333333333358406;
    nids.push_back(69828);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.010666666666666668045;
    nids.push_back(-11939);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.18578260869565219049;
    tri3_xyze(2, 0) = 0.01200000000000000025;
    nids.push_back(69827);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.012000000000000001985;
    nids.push_back(69811);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.010666666666666668045;
    nids.push_back(-11939);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11940);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.18578260869565216273;
    tri3_xyze(2, 1) = 0.0066666666666666679619;
    nids.push_back(69829);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11940);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.18578260869565221824;
    tri3_xyze(2, 0) = 0.0093333333333333358406;
    nids.push_back(69828);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0093333333333333341059;
    nids.push_back(69812);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.0080000000000000001665;
    nids.push_back(-11940);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.17491304347826086563;
    tri3_xyze(2, 0) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0040000000000000000833;
    nids.push_back(69814);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.0053333333333333331552;
    nids.push_back(-11941);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = 0.18578260869565216273;
    tri3_xyze(2, 0) = 0.0066666666666666679619;
    nids.push_back(69829);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = 0.17491304347826086563;
    tri3_xyze(2, 1) = 0.0066666666666666662272;
    nids.push_back(69813);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = 0.18034782608695654194;
    tri3_xyze(2, 2) = 0.0053333333333333331552;
    nids.push_back(-11941);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888844382403501054;
    hex8_xyze(1, 0) = 0.15555537019860399273;
    hex8_xyze(2, 0) = 0.027775224160459001388;
    nids.push_back(1659166);
    hex8_xyze(0, 1) = 1.4888843227231900457;
    hex8_xyze(1, 1) = 0.15555536058671501354;
    hex8_xyze(2, 1) = 0.016665065396155899491;
    nids.push_back(1659167);
    hex8_xyze(0, 2) = 1.4888842240022399643;
    hex8_xyze(1, 2) = 0.16666626305307200018;
    hex8_xyze(2, 2) = 0.016665064871666700891;
    nids.push_back(1659170);
    hex8_xyze(0, 3) = 1.4888843213410800637;
    hex8_xyze(1, 3) = 0.16666627402986999851;
    hex8_xyze(2, 3) = 0.027775221498795001074;
    nids.push_back(1659169);
    hex8_xyze(0, 4) = 1.5000004897729199982;
    hex8_xyze(1, 4) = 0.15555522844023400575;
    hex8_xyze(2, 4) = 0.027775157512175699392;
    nids.push_back(1659175);
    hex8_xyze(0, 5) = 1.5000001424968198993;
    hex8_xyze(1, 5) = 0.15555522493156401231;
    hex8_xyze(2, 5) = 0.016665059489050899899;
    nids.push_back(1659176);
    hex8_xyze(0, 6) = 1.5000001282995700791;
    hex8_xyze(1, 6) = 0.16666603816894198786;
    hex8_xyze(2, 6) = 0.016665058991174600683;
    nids.push_back(1659179);
    hex8_xyze(0, 7) = 1.5000004189006899136;
    hex8_xyze(1, 7) = 0.16666604639274298916;
    hex8_xyze(2, 7) = 0.027775153442394000247;
    nids.push_back(1659178);

    intersection.add_element(627533, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4777664532429199973;
    hex8_xyze(1, 0) = 0.15555544151139200082;
    hex8_xyze(2, 0) = 0.016665084338472798547;
    nids.push_back(1659158);
    hex8_xyze(0, 1) = 1.4777665041507099808;
    hex8_xyze(1, 1) = 0.15555542440888900368;
    hex8_xyze(2, 1) = 0.0055550086758738900089;
    nids.push_back(1659183);
    hex8_xyze(0, 2) = 1.4777663833769199009;
    hex8_xyze(1, 2) = 0.16666640851046099492;
    hex8_xyze(2, 2) = 0.0055550106126051202599;
    nids.push_back(1659186);
    hex8_xyze(0, 3) = 1.4777663331356800658;
    hex8_xyze(1, 3) = 0.16666642353435900947;
    hex8_xyze(2, 3) = 0.016665083910939598733;
    nids.push_back(1659161);
    hex8_xyze(0, 4) = 1.4888843227231900457;
    hex8_xyze(1, 4) = 0.15555536058671501354;
    hex8_xyze(2, 4) = 0.016665065396155899491;
    nids.push_back(1659167);
    hex8_xyze(0, 5) = 1.4888842941263600306;
    hex8_xyze(1, 5) = 0.1555553419713419927;
    hex8_xyze(2, 5) = 0.0055550089165411703843;
    nids.push_back(1659192);
    hex8_xyze(0, 6) = 1.4888841782854700391;
    hex8_xyze(1, 6) = 0.16666624615229999606;
    hex8_xyze(2, 6) = 0.0055550105485112303422;
    nids.push_back(1659195);
    hex8_xyze(0, 7) = 1.4888842240022399643;
    hex8_xyze(1, 7) = 0.16666626305307200018;
    hex8_xyze(2, 7) = 0.016665064871666700891;
    nids.push_back(1659170);

    intersection.add_element(627549, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888844108284700063;
    hex8_xyze(1, 0) = 0.14444432852221800179;
    hex8_xyze(2, 0) = 0.016665065679285200745;
    nids.push_back(1659164);
    hex8_xyze(0, 1) = 1.4888843821355000063;
    hex8_xyze(1, 1) = 0.14444431303200699901;
    hex8_xyze(2, 1) = 0.0055550091745071600996;
    nids.push_back(1659189);
    hex8_xyze(0, 2) = 1.4888842941263600306;
    hex8_xyze(1, 2) = 0.1555553419713419927;
    hex8_xyze(2, 2) = 0.0055550089165411703843;
    nids.push_back(1659192);
    hex8_xyze(0, 3) = 1.4888843227231900457;
    hex8_xyze(1, 3) = 0.15555536058671501354;
    hex8_xyze(2, 3) = 0.016665065396155899491;
    nids.push_back(1659167);
    hex8_xyze(0, 4) = 1.5000002679713100306;
    hex8_xyze(1, 4) = 0.14444424099927699601;
    hex8_xyze(2, 4) = 0.016665060064730799483;
    nids.push_back(1659173);
    hex8_xyze(0, 5) = 1.5000001113807999165;
    hex8_xyze(1, 5) = 0.14444422943338700027;
    hex8_xyze(2, 5) = 0.0055550180097395703699;
    nids.push_back(1659198);
    hex8_xyze(0, 6) = 1.4999999860720900635;
    hex8_xyze(1, 6) = 0.15555520379053400237;
    hex8_xyze(2, 6) = 0.0055550175608654397963;
    nids.push_back(1659201);
    hex8_xyze(0, 7) = 1.5000001424968198993;
    hex8_xyze(1, 7) = 0.15555522493156401231;
    hex8_xyze(2, 7) = 0.016665059489050899899;
    nids.push_back(1659176);

    intersection.add_element(627555, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888843227231900457;
    hex8_xyze(1, 0) = 0.15555536058671501354;
    hex8_xyze(2, 0) = 0.016665065396155899491;
    nids.push_back(1659167);
    hex8_xyze(0, 1) = 1.4888842941263600306;
    hex8_xyze(1, 1) = 0.1555553419713419927;
    hex8_xyze(2, 1) = 0.0055550089165411703843;
    nids.push_back(1659192);
    hex8_xyze(0, 2) = 1.4888841782854700391;
    hex8_xyze(1, 2) = 0.16666624615229999606;
    hex8_xyze(2, 2) = 0.0055550105485112303422;
    nids.push_back(1659195);
    hex8_xyze(0, 3) = 1.4888842240022399643;
    hex8_xyze(1, 3) = 0.16666626305307200018;
    hex8_xyze(2, 3) = 0.016665064871666700891;
    nids.push_back(1659170);
    hex8_xyze(0, 4) = 1.5000001424968198993;
    hex8_xyze(1, 4) = 0.15555522493156401231;
    hex8_xyze(2, 4) = 0.016665059489050899899;
    nids.push_back(1659176);
    hex8_xyze(0, 5) = 1.4999999860720900635;
    hex8_xyze(1, 5) = 0.15555520379053400237;
    hex8_xyze(2, 5) = 0.0055550175608654397963;
    nids.push_back(1659201);
    hex8_xyze(0, 6) = 1.4999999158749999228;
    hex8_xyze(1, 6) = 0.16666602179153500174;
    hex8_xyze(2, 6) = 0.0055550205723179897979;
    nids.push_back(1659204);
    hex8_xyze(0, 7) = 1.5000001282995700791;
    hex8_xyze(1, 7) = 0.16666603816894198786;
    hex8_xyze(2, 7) = 0.016665058991174600683;
    nids.push_back(1659179);

    intersection.add_element(627558, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888842941263600306;
    hex8_xyze(1, 0) = 0.1555553419713419927;
    hex8_xyze(2, 0) = 0.0055550089165411703843;
    nids.push_back(1659192);
    hex8_xyze(0, 1) = 1.4888842941636299955;
    hex8_xyze(1, 1) = 0.15555532285765499845;
    hex8_xyze(2, 1) = -0.0055550092980582096991;
    nids.push_back(1659193);
    hex8_xyze(0, 2) = 1.4888841783508199867;
    hex8_xyze(1, 2) = 0.16666622711676701285;
    hex8_xyze(2, 2) = -0.0055550113929668199983;
    nids.push_back(1659196);
    hex8_xyze(0, 3) = 1.4888841782854700391;
    hex8_xyze(1, 3) = 0.16666624615229999606;
    hex8_xyze(2, 3) = 0.0055550105485112303422;
    nids.push_back(1659195);
    hex8_xyze(0, 4) = 1.4999999860720900635;
    hex8_xyze(1, 4) = 0.15555520379053400237;
    hex8_xyze(2, 4) = 0.0055550175608654397963;
    nids.push_back(1659201);
    hex8_xyze(0, 5) = 1.49999998622225994;
    hex8_xyze(1, 5) = 0.15555518428953399312;
    hex8_xyze(2, 5) = -0.0055550182451922797827;
    nids.push_back(1659202);
    hex8_xyze(0, 6) = 1.4999999161410799697;
    hex8_xyze(1, 6) = 0.16666600246681201325;
    hex8_xyze(2, 6) = -0.0055550222066562803674;
    nids.push_back(1659205);
    hex8_xyze(0, 7) = 1.4999999158749999228;
    hex8_xyze(1, 7) = 0.16666602179153500174;
    hex8_xyze(2, 7) = 0.0055550205723179897979;
    nids.push_back(1659204);

    intersection.add_element(627559, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888842240022399643;
    hex8_xyze(1, 0) = 0.16666626305307200018;
    hex8_xyze(2, 0) = 0.016665064871666700891;
    nids.push_back(1659170);
    hex8_xyze(0, 1) = 1.4888841782854700391;
    hex8_xyze(1, 1) = 0.16666624615229999606;
    hex8_xyze(2, 1) = 0.0055550105485112303422;
    nids.push_back(1659195);
    hex8_xyze(0, 2) = 1.4888839159546700674;
    hex8_xyze(1, 2) = 0.17777705213192498968;
    hex8_xyze(2, 2) = 0.0055550081988082400039;
    nids.push_back(1659603);
    hex8_xyze(0, 3) = 1.4888839436887000289;
    hex8_xyze(1, 3) = 0.1777770673452669925;
    hex8_xyze(2, 3) = 0.016665064173765401639;
    nids.push_back(1659578);
    hex8_xyze(0, 4) = 1.5000001282995700791;
    hex8_xyze(1, 4) = 0.16666603816894198786;
    hex8_xyze(2, 4) = 0.016665058991174600683;
    nids.push_back(1659179);
    hex8_xyze(0, 5) = 1.4999999158749999228;
    hex8_xyze(1, 5) = 0.16666602179153500174;
    hex8_xyze(2, 5) = 0.0055550205723179897979;
    nids.push_back(1659204);
    hex8_xyze(0, 6) = 1.4999994811792800586;
    hex8_xyze(1, 6) = 0.17777671894665300623;
    hex8_xyze(2, 6) = 0.0055550165728729802503;
    nids.push_back(1659612);
    hex8_xyze(0, 7) = 1.4999996367108900941;
    hex8_xyze(1, 7) = 0.17777673040163100016;
    hex8_xyze(2, 7) = 0.016665058339944201216;
    nids.push_back(1659587);

    intersection.add_element(627957, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.5000001424968198993;
    hex8_xyze(1, 0) = 0.15555522493156401231;
    hex8_xyze(2, 0) = 0.016665059489050899899;
    nids.push_back(1659176);
    hex8_xyze(0, 1) = 1.4999999860720900635;
    hex8_xyze(1, 1) = 0.15555520379053400237;
    hex8_xyze(2, 1) = 0.0055550175608654397963;
    nids.push_back(1659201);
    hex8_xyze(0, 2) = 1.4999999158749999228;
    hex8_xyze(1, 2) = 0.16666602179153500174;
    hex8_xyze(2, 2) = 0.0055550205723179897979;
    nids.push_back(1659204);
    hex8_xyze(0, 3) = 1.5000001282995700791;
    hex8_xyze(1, 3) = 0.16666603816894198786;
    hex8_xyze(2, 3) = 0.016665058991174600683;
    nids.push_back(1659179);
    hex8_xyze(0, 4) = 1.5111175273932899721;
    hex8_xyze(1, 4) = 0.15555493170109899181;
    hex8_xyze(2, 4) = 0.016665101232082601274;
    nids.push_back(1671716);
    hex8_xyze(0, 5) = 1.5111171325680099464;
    hex8_xyze(1, 5) = 0.15555496743220700862;
    hex8_xyze(2, 5) = 0.0055550494019330201242;
    nids.push_back(1671741);
    hex8_xyze(0, 6) = 1.511116998590170013;
    hex8_xyze(1, 6) = 0.16666573023434899659;
    hex8_xyze(2, 6) = 0.0055549944925532800111;
    nids.push_back(1671744);
    hex8_xyze(0, 7) = 1.511117261834499903;
    hex8_xyze(1, 7) = 0.16666574291770899774;
    hex8_xyze(2, 7) = 0.016665100819606998989;
    nids.push_back(1671719);

    intersection.add_element(639690, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
