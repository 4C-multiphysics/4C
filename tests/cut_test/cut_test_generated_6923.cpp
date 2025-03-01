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

void test_generated_6923()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 1) = 0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 2) = 0.0748649;
    tri3_xyze(1, 2) = 0.0975658;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1068);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 1) = 0.089655;
    tri3_xyze(1, 1) = 0.089655;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7825);
    tri3_xyze(0, 2) = 0.0748649;
    tri3_xyze(1, 2) = 0.0975658;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1068);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0546845;
    tri3_xyze(1, 0) = 0.0947164;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(6569);
    tri3_xyze(0, 1) = -0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 2) = -0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-865);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 1) = -0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(6817);
    tri3_xyze(0, 2) = -0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-865);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0546845;
    tri3_xyze(1, 0) = 0.0947164;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(6569);
    tri3_xyze(0, 1) = -0.0577254;
    tri3_xyze(1, 1) = 0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(6571);
    tri3_xyze(0, 2) = -0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-866);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0577254;
    tri3_xyze(1, 0) = 0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(6571);
    tri3_xyze(0, 1) = -0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 2) = -0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-866);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 1) = -0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 2) = -0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-866);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 1) = -0.0546845;
    tri3_xyze(1, 1) = 0.0947164;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(6569);
    tri3_xyze(0, 2) = -0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-866);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0577254;
    tri3_xyze(1, 0) = 0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(6571);
    tri3_xyze(0, 1) = -0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(6573);
    tri3_xyze(0, 2) = -0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-867);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(6573);
    tri3_xyze(0, 1) = -0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 2) = -0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-867);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 1) = -0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 2) = -0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-867);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 1) = -0.0577254;
    tri3_xyze(1, 1) = 0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(6571);
    tri3_xyze(0, 2) = -0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-867);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(6573);
    tri3_xyze(0, 1) = -0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(6575);
    tri3_xyze(0, 2) = -0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-868);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(6575);
    tri3_xyze(0, 1) = -0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 2) = -0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-868);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 1) = -0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 2) = -0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-868);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 1) = -0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(6573);
    tri3_xyze(0, 2) = -0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-868);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(6575);
    tri3_xyze(0, 1) = -0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(6577);
    tri3_xyze(0, 2) = -0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-869);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0659356;
    tri3_xyze(1, 0) = 0.114204;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(6577);
    tri3_xyze(0, 1) = -0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 2) = -0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-869);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 1) = -0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 2) = -0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-869);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 1) = -0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(6575);
    tri3_xyze(0, 2) = -0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-869);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0682242;
    tri3_xyze(1, 0) = 0.118168;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(6579);
    tri3_xyze(0, 1) = -0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 2) = -0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-870);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 1) = -0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 2) = -0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-870);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 1) = -0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(6577);
    tri3_xyze(0, 2) = -0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-870);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0702254;
    tri3_xyze(1, 0) = 0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(6581);
    tri3_xyze(0, 1) = -0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 2) = -0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-871);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 1) = -0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 2) = -0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-871);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 1) = -0.0682242;
    tri3_xyze(1, 1) = 0.118168;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(6579);
    tri3_xyze(0, 2) = -0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-871);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0719077;
    tri3_xyze(1, 0) = 0.124548;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(6583);
    tri3_xyze(0, 1) = -0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 2) = -0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-872);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 1) = -0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 2) = -0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-872);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 1) = -0.0702254;
    tri3_xyze(1, 1) = 0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(6581);
    tri3_xyze(0, 2) = -0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-872);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0732444;
    tri3_xyze(1, 0) = 0.126863;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(6585);
    tri3_xyze(0, 1) = -0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 2) = -0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-873);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 1) = -0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 2) = -0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-873);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 1) = -0.0719077;
    tri3_xyze(1, 1) = 0.124548;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(6583);
    tri3_xyze(0, 2) = -0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-873);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0742146;
    tri3_xyze(1, 0) = 0.128543;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(6587);
    tri3_xyze(0, 1) = -0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 2) = -0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-874);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 1) = -0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 2) = -0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-874);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 1) = -0.0732444;
    tri3_xyze(1, 1) = 0.126863;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(6585);
    tri3_xyze(0, 2) = -0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-874);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0748029;
    tri3_xyze(1, 0) = 0.129562;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(6589);
    tri3_xyze(0, 1) = -0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 2) = -0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-875);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 1) = -0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 2) = -0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-875);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 1) = -0.0742146;
    tri3_xyze(1, 1) = 0.128543;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(6587);
    tri3_xyze(0, 2) = -0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-875);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.075;
    tri3_xyze(1, 0) = 0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(6591);
    tri3_xyze(0, 1) = -0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 2) = -0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-876);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 1) = -0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 2) = -0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-876);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 1) = -0.0748029;
    tri3_xyze(1, 1) = 0.129562;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(6589);
    tri3_xyze(0, 2) = -0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-876);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(6843);
    tri3_xyze(0, 1) = -0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 2) = -0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-877);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 1) = -0.075;
    tri3_xyze(1, 1) = 0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(6591);
    tri3_xyze(0, 2) = -0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-877);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.013043;
    tri3_xyze(1, 0) = 0.0486771;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(6792);
    tri3_xyze(0, 1) = 1.00309e-15;
    tri3_xyze(1, 1) = 0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 2) = -0.00649599;
    tri3_xyze(1, 2) = 0.0493419;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-901);
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
    tri3_xyze(0, 0) = -0.0133475;
    tri3_xyze(1, 0) = 0.0498136;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(6795);
    tri3_xyze(0, 1) = 1.00316e-15;
    tri3_xyze(1, 1) = 0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 2) = -0.00659763;
    tri3_xyze(1, 2) = 0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-903);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00316e-15;
    tri3_xyze(1, 0) = 0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 1) = 1.00309e-15;
    tri3_xyze(1, 1) = 0.0503943;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 2) = -0.00659763;
    tri3_xyze(1, 2) = 0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-903);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00309e-15;
    tri3_xyze(1, 0) = 0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 1) = -0.013043;
    tri3_xyze(1, 1) = 0.0486771;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(6792);
    tri3_xyze(0, 2) = -0.00659763;
    tri3_xyze(1, 2) = 0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-903);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0138497;
    tri3_xyze(1, 0) = 0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(6797);
    tri3_xyze(0, 1) = 9.55749e-16;
    tri3_xyze(1, 1) = 0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 2) = -0.00679931;
    tri3_xyze(1, 2) = 0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-904);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.55749e-16;
    tri3_xyze(1, 0) = 0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 1) = 1.00316e-15;
    tri3_xyze(1, 1) = 0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 2) = -0.00679931;
    tri3_xyze(1, 2) = 0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-904);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00316e-15;
    tri3_xyze(1, 0) = 0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 1) = -0.0133475;
    tri3_xyze(1, 1) = 0.0498136;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(6795);
    tri3_xyze(0, 2) = -0.00679931;
    tri3_xyze(1, 2) = 0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-904);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0145417;
    tri3_xyze(1, 0) = 0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(6799);
    tri3_xyze(0, 1) = 1.05334e-15;
    tri3_xyze(1, 1) = 0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 2) = -0.00709784;
    tri3_xyze(1, 2) = 0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-905);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05334e-15;
    tri3_xyze(1, 0) = 0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 1) = 9.55749e-16;
    tri3_xyze(1, 1) = 0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 2) = -0.00709784;
    tri3_xyze(1, 2) = 0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-905);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.55749e-16;
    tri3_xyze(1, 0) = 0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 1) = -0.0138497;
    tri3_xyze(1, 1) = 0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(6797);
    tri3_xyze(0, 2) = -0.00709784;
    tri3_xyze(1, 2) = 0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-905);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = 0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(6801);
    tri3_xyze(0, 1) = 1.05654e-15;
    tri3_xyze(1, 1) = 0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 2) = -0.00748853;
    tri3_xyze(1, 2) = 0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-906);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05654e-15;
    tri3_xyze(1, 0) = 0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 1) = 1.05334e-15;
    tri3_xyze(1, 1) = 0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 2) = -0.00748853;
    tri3_xyze(1, 2) = 0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-906);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05334e-15;
    tri3_xyze(1, 0) = 0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 1) = -0.0145417;
    tri3_xyze(1, 1) = 0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(6799);
    tri3_xyze(0, 2) = -0.00748853;
    tri3_xyze(1, 2) = 0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-906);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0164484;
    tri3_xyze(1, 0) = 0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(6803);
    tri3_xyze(0, 1) = 1.06034e-15;
    tri3_xyze(1, 1) = 0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 2) = -0.0079652;
    tri3_xyze(1, 2) = 0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-907);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06034e-15;
    tri3_xyze(1, 0) = 0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 1) = 1.05654e-15;
    tri3_xyze(1, 1) = 0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 2) = -0.0079652;
    tri3_xyze(1, 2) = 0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-907);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05654e-15;
    tri3_xyze(1, 0) = 0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = 0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(6801);
    tri3_xyze(0, 2) = -0.0079652;
    tri3_xyze(1, 2) = 0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-907);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.017633;
    tri3_xyze(1, 0) = 0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(6805);
    tri3_xyze(0, 1) = 1.00417e-15;
    tri3_xyze(1, 1) = 0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 2) = -0.00852035;
    tri3_xyze(1, 2) = 0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-908);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00417e-15;
    tri3_xyze(1, 0) = 0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 1) = 1.06034e-15;
    tri3_xyze(1, 1) = 0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 2) = -0.00852035;
    tri3_xyze(1, 2) = 0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-908);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06034e-15;
    tri3_xyze(1, 0) = 0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 1) = -0.0164484;
    tri3_xyze(1, 1) = 0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(6803);
    tri3_xyze(0, 2) = -0.00852035;
    tri3_xyze(1, 2) = 0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-908);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0189478;
    tri3_xyze(1, 0) = 0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(6807);
    tri3_xyze(0, 1) = 1.00448e-15;
    tri3_xyze(1, 1) = 0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 2) = -0.00914521;
    tri3_xyze(1, 2) = 0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-909);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00448e-15;
    tri3_xyze(1, 0) = 0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 1) = 1.00417e-15;
    tri3_xyze(1, 1) = 0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 2) = -0.00914521;
    tri3_xyze(1, 2) = 0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-909);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00417e-15;
    tri3_xyze(1, 0) = 0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 1) = -0.017633;
    tri3_xyze(1, 1) = 0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(6805);
    tri3_xyze(0, 2) = -0.00914521;
    tri3_xyze(1, 2) = 0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-909);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0203719;
    tri3_xyze(1, 0) = 0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(6809);
    tri3_xyze(0, 1) = 1.00482e-15;
    tri3_xyze(1, 1) = 0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 2) = -0.00982993;
    tri3_xyze(1, 2) = 0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-910);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00482e-15;
    tri3_xyze(1, 0) = 0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 1) = 1.00448e-15;
    tri3_xyze(1, 1) = 0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 2) = -0.00982993;
    tri3_xyze(1, 2) = 0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-910);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00448e-15;
    tri3_xyze(1, 0) = 0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 1) = -0.0189478;
    tri3_xyze(1, 1) = 0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(6807);
    tri3_xyze(0, 2) = -0.00982993;
    tri3_xyze(1, 2) = 0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-910);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0218829;
    tri3_xyze(1, 0) = 0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(6811);
    tri3_xyze(0, 1) = 1.00518e-15;
    tri3_xyze(1, 1) = 0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 2) = -0.0105637;
    tri3_xyze(1, 2) = 0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-911);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00518e-15;
    tri3_xyze(1, 0) = 0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 1) = 1.00482e-15;
    tri3_xyze(1, 1) = 0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 2) = -0.0105637;
    tri3_xyze(1, 2) = 0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-911);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00482e-15;
    tri3_xyze(1, 0) = 0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 1) = -0.0203719;
    tri3_xyze(1, 1) = 0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(6809);
    tri3_xyze(0, 2) = -0.0105637;
    tri3_xyze(1, 2) = 0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-911);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.023457;
    tri3_xyze(1, 0) = 0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(6813);
    tri3_xyze(0, 1) = 1.00555e-15;
    tri3_xyze(1, 1) = 0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 2) = -0.011335;
    tri3_xyze(1, 2) = 0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-912);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00555e-15;
    tri3_xyze(1, 0) = 0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 1) = 1.00518e-15;
    tri3_xyze(1, 1) = 0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 2) = -0.011335;
    tri3_xyze(1, 2) = 0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-912);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00518e-15;
    tri3_xyze(1, 0) = 0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 1) = -0.0218829;
    tri3_xyze(1, 1) = 0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(6811);
    tri3_xyze(0, 2) = -0.011335;
    tri3_xyze(1, 2) = 0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-912);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0250693;
    tri3_xyze(1, 0) = 0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(6815);
    tri3_xyze(0, 1) = 1.00593e-15;
    tri3_xyze(1, 1) = 0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 2) = -0.0121316;
    tri3_xyze(1, 2) = 0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-913);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00593e-15;
    tri3_xyze(1, 0) = 0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 1) = 1.00555e-15;
    tri3_xyze(1, 1) = 0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 2) = -0.0121316;
    tri3_xyze(1, 2) = 0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-913);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00555e-15;
    tri3_xyze(1, 0) = 0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 1) = -0.023457;
    tri3_xyze(1, 1) = 0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(6813);
    tri3_xyze(0, 2) = -0.0121316;
    tri3_xyze(1, 2) = 0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-913);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(6817);
    tri3_xyze(0, 1) = 1.00632e-15;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-914);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00632e-15;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 1) = 1.00593e-15;
    tri3_xyze(1, 1) = 0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-914);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00593e-15;
    tri3_xyze(1, 0) = 0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 1) = -0.0250693;
    tri3_xyze(1, 1) = 0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(6815);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-914);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(6817);
    tri3_xyze(0, 1) = -0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 2) = -0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 1) = 1.0067e-15;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 2) = -0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0067e-15;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 1) = 1.00632e-15;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 2) = -0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00632e-15;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 1) = -0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(6817);
    tri3_xyze(0, 2) = -0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-915);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 1) = -0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 2) = -0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-916);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 1) = 9.04528e-16;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 2) = -0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-916);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.04528e-16;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 1) = 1.0067e-15;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 2) = -0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-916);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0067e-15;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 1) = -0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(6819);
    tri3_xyze(0, 2) = -0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-916);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 1) = -0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 2) = -0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-917);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 1) = 7.91974e-16;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 2) = -0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-917);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.91974e-16;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 1) = 9.04528e-16;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 2) = -0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-917);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.04528e-16;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 1) = -0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(6821);
    tri3_xyze(0, 2) = -0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-917);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 1) = -0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 2) = -0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-918);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 2) = -0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-918);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 1) = 7.91974e-16;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 2) = -0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-918);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.91974e-16;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 1) = -0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(6823);
    tri3_xyze(0, 2) = -0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-918);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 1) = -0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 2) = -0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-919);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 2) = -0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-919);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 2) = -0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-919);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 1) = -0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(6825);
    tri3_xyze(0, 2) = -0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-919);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 1) = -0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 2) = -0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-920);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 2) = -0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-920);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 2) = -0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-920);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 1) = -0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(6827);
    tri3_xyze(0, 2) = -0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-920);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 1) = -0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 2) = -0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-921);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 2) = -0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-921);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 2) = -0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-921);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 1) = -0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(6829);
    tri3_xyze(0, 2) = -0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-921);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 1) = -0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 2) = -0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 2) = -0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 2) = -0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 1) = -0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(6831);
    tri3_xyze(0, 2) = -0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-922);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 1) = -0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 2) = -0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 2) = -0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 2) = -0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 1) = -0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(6833);
    tri3_xyze(0, 2) = -0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-923);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 1) = -0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 2) = -0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 1) = 1.27275e-15;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 2) = -0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27275e-15;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 2) = -0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 1) = -0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(6835);
    tri3_xyze(0, 2) = -0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-924);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 1) = -0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 2) = -0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 1) = 1.27491e-15;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 2) = -0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27491e-15;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 1) = 1.27275e-15;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 2) = -0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27275e-15;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 1) = -0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(6837);
    tri3_xyze(0, 2) = -0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-925);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 1) = -0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 1) = 8.75958e-16;
    tri3_xyze(1, 1) = 0.15;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.75958e-16;
    tri3_xyze(1, 0) = 0.15;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 1) = 1.27491e-15;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27491e-15;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 1) = -0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(6839);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-926);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 1) = -0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(6843);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-927);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(6843);
    tri3_xyze(0, 1) = 7.43408e-16;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-927);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.43408e-16;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 1) = 8.75958e-16;
    tri3_xyze(1, 1) = 0.15;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-927);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.75958e-16;
    tri3_xyze(1, 0) = 0.15;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 1) = -0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(6841);
    tri3_xyze(0, 2) = -0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-927);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(6845);
    tri3_xyze(0, 1) = 7.45426e-16;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 2) = -0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-928);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.45426e-16;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 1) = 7.43408e-16;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 2) = -0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-928);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.43408e-16;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 1) = -0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(6843);
    tri3_xyze(0, 2) = -0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-928);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(6847);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 2) = -0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-929);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 1) = 7.45426e-16;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 2) = -0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-929);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.45426e-16;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 1) = -0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(6845);
    tri3_xyze(0, 2) = -0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-929);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(6849);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 2) = -0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-930);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 2) = -0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-930);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 1) = -0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(6847);
    tri3_xyze(0, 2) = -0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-930);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(6851);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 2) = -0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-931);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 2) = -0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-931);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 1) = -0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(6849);
    tri3_xyze(0, 2) = -0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-931);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(6853);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 2) = -0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-932);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 2) = -0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-932);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 1) = -0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(6851);
    tri3_xyze(0, 2) = -0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-932);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(6855);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 2) = -0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-933);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 2) = -0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-933);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 1) = -0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(6853);
    tri3_xyze(0, 2) = -0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-933);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(6857);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 2) = -0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-934);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 2) = -0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-934);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 1) = -0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(6855);
    tri3_xyze(0, 2) = -0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-934);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(6859);
    tri3_xyze(0, 1) = 1.22288e-15;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 2) = -0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-935);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.22288e-15;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 2) = -0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-935);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 1) = -0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(6857);
    tri3_xyze(0, 2) = -0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-935);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(6861);
    tri3_xyze(0, 1) = 1.10961e-15;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 2) = -0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-936);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.10961e-15;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 1) = 1.22288e-15;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 2) = -0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-936);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.22288e-15;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 1) = -0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(6859);
    tri3_xyze(0, 2) = -0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-936);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(6863);
    tri3_xyze(0, 1) = 8.12418e-16;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 2) = -0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-937);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.12418e-16;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 1) = 1.10961e-15;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 2) = -0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-937);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.10961e-15;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 1) = -0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(6861);
    tri3_xyze(0, 2) = -0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-937);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(6865);
    tri3_xyze(0, 1) = 9.14709e-16;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 2) = -0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-938);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.14709e-16;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 1) = 8.12418e-16;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 2) = -0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-938);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.12418e-16;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 1) = -0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(6863);
    tri3_xyze(0, 2) = -0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-938);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00593e-15;
    tri3_xyze(1, 0) = 0.0968605;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(7117);
    tri3_xyze(0, 1) = 9.14709e-16;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.849901;
    nids.push_back(-939);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.14709e-16;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 1) = -0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(6865);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.849901;
    nids.push_back(-939);
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
    tri3_xyze(0, 0) = 1.00309e-15;
    tri3_xyze(1, 0) = 0.0503943;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7042);
    tri3_xyze(0, 1) = 1.00316e-15;
    tri3_xyze(1, 1) = 0.0515708;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 2) = 0.00659763;
    tri3_xyze(1, 2) = 0.050114;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-953);
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
    tri3_xyze(0, 0) = 1.00316e-15;
    tri3_xyze(1, 0) = 0.0515708;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7045);
    tri3_xyze(0, 1) = 9.55749e-16;
    tri3_xyze(1, 1) = 0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = 0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-954);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.55749e-16;
    tri3_xyze(1, 0) = 0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = 0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 2) = 0.00679931;
    tri3_xyze(1, 2) = 0.0516459;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-954);
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
    tri3_xyze(0, 0) = 9.55749e-16;
    tri3_xyze(1, 0) = 0.0535112;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 1) = 1.05334e-15;
    tri3_xyze(1, 1) = 0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = 0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-955);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05334e-15;
    tri3_xyze(1, 0) = 0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = 0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = 0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-955);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = 0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 1) = 0.0138497;
    tri3_xyze(1, 1) = 0.0516878;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = 0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-955);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = 0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 1) = 9.55749e-16;
    tri3_xyze(1, 1) = 0.0535112;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7047);
    tri3_xyze(0, 2) = 0.00709784;
    tri3_xyze(1, 2) = 0.0539135;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-955);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05334e-15;
    tri3_xyze(1, 0) = 0.0561847;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 1) = 1.05654e-15;
    tri3_xyze(1, 1) = 0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = 0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-956);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05654e-15;
    tri3_xyze(1, 0) = 0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = 0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = 0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-956);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = 0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = 0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = 0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-956);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = 0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 1) = 1.05334e-15;
    tri3_xyze(1, 1) = 0.0561847;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7049);
    tri3_xyze(0, 2) = 0.00748853;
    tri3_xyze(1, 2) = 0.056881;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-956);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05654e-15;
    tri3_xyze(1, 0) = 0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 1) = 1.06034e-15;
    tri3_xyze(1, 1) = 0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = 0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-957);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06034e-15;
    tri3_xyze(1, 0) = 0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = 0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = 0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-957);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = 0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = 0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = 0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-957);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = 0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 1) = 1.05654e-15;
    tri3_xyze(1, 1) = 0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7051);
    tri3_xyze(0, 2) = 0.0079652;
    tri3_xyze(1, 2) = 0.0605017;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-957);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.06034e-15;
    tri3_xyze(1, 0) = 0.0635516;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 1) = 1.00417e-15;
    tri3_xyze(1, 1) = 0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = 0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-958);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00417e-15;
    tri3_xyze(1, 0) = 0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = 0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = 0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-958);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = 0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = 0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = 0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-958);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = 0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 1) = 1.06034e-15;
    tri3_xyze(1, 1) = 0.0635516;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7053);
    tri3_xyze(0, 2) = 0.00852035;
    tri3_xyze(1, 2) = 0.0647185;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-958);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00417e-15;
    tri3_xyze(1, 0) = 0.0681288;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 1) = 1.00448e-15;
    tri3_xyze(1, 1) = 0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = 0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-959);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00448e-15;
    tri3_xyze(1, 0) = 0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = 0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = 0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-959);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = 0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = 0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = 0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-959);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = 0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 1) = 1.00417e-15;
    tri3_xyze(1, 1) = 0.0681288;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7055);
    tri3_xyze(0, 2) = 0.00914521;
    tri3_xyze(1, 2) = 0.0694647;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-959);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00448e-15;
    tri3_xyze(1, 0) = 0.0732087;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 1) = 1.00482e-15;
    tri3_xyze(1, 1) = 0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = 0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-960);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00482e-15;
    tri3_xyze(1, 0) = 0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = 0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = 0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-960);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = 0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = 0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = 0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-960);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = 0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 1) = 1.00448e-15;
    tri3_xyze(1, 1) = 0.0732087;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7057);
    tri3_xyze(0, 2) = 0.00982993;
    tri3_xyze(1, 2) = 0.0746657;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-960);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00482e-15;
    tri3_xyze(1, 0) = 0.078711;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 1) = 1.00518e-15;
    tri3_xyze(1, 1) = 0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = 0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-961);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00518e-15;
    tri3_xyze(1, 0) = 0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = 0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = 0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-961);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = 0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = 0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = 0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-961);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = 0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 1) = 1.00482e-15;
    tri3_xyze(1, 1) = 0.078711;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7059);
    tri3_xyze(0, 2) = 0.0105637;
    tri3_xyze(1, 2) = 0.0802394;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-961);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00518e-15;
    tri3_xyze(1, 0) = 0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 1) = 1.00555e-15;
    tri3_xyze(1, 1) = 0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = 0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-962);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00555e-15;
    tri3_xyze(1, 0) = 0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = 0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = 0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-962);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = 0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = 0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = 0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-962);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = 0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 1) = 1.00518e-15;
    tri3_xyze(1, 1) = 0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7061);
    tri3_xyze(0, 2) = 0.011335;
    tri3_xyze(1, 2) = 0.0860978;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-962);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00555e-15;
    tri3_xyze(1, 0) = 0.0906309;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 1) = 1.00593e-15;
    tri3_xyze(1, 1) = 0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = 0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-963);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00593e-15;
    tri3_xyze(1, 0) = 0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = 0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = 0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-963);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = 0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = 0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = 0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-963);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = 0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 1) = 1.00555e-15;
    tri3_xyze(1, 1) = 0.0906309;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7063);
    tri3_xyze(0, 2) = 0.0121316;
    tri3_xyze(1, 2) = 0.0921486;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-963);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00593e-15;
    tri3_xyze(1, 0) = 0.0968605;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 1) = 1.00632e-15;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-964);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00632e-15;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-964);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = 0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-964);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = 0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 1) = 1.00593e-15;
    tri3_xyze(1, 1) = 0.0968605;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7065);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-964);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00632e-15;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 1) = 1.0067e-15;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-965);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0067e-15;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-965);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-965);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 1) = 1.00632e-15;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7067);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-965);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0067e-15;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 1) = 9.04528e-16;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-966);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.04528e-16;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-966);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-966);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 1) = 1.0067e-15;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7069);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-966);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.04528e-16;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 1) = 7.91974e-16;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-967);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.91974e-16;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-967);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-967);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 1) = 9.04528e-16;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7071);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-967);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.91974e-16;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-968);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-968);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-968);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 1) = 7.91974e-16;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7073);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-968);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-969);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-969);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-969);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7075);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-969);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-970);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-970);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-970);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7077);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-970);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7079);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-971);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-972);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-972);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-972);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7081);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-972);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-973);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-973);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-973);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7083);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-973);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 1) = 1.27275e-15;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-974);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27275e-15;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-974);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-974);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7085);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-974);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27275e-15;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 1) = 1.27491e-15;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-975);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27491e-15;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-975);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-975);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 1) = 1.27275e-15;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7087);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-975);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.27491e-15;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 1) = 8.75958e-16;
    tri3_xyze(1, 1) = 0.15;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-976);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.75958e-16;
    tri3_xyze(1, 0) = 0.15;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 1) = 0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-976);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-976);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 1) = 1.27491e-15;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7089);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-976);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.75958e-16;
    tri3_xyze(1, 0) = 0.15;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 1) = 7.43408e-16;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-977);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.43408e-16;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-977);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 1) = 0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-977);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 1) = 8.75958e-16;
    tri3_xyze(1, 1) = 0.15;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7091);
    tri3_xyze(0, 2) = 0.0193859;
    tri3_xyze(1, 2) = 0.147251;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-977);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.43408e-16;
    tri3_xyze(1, 0) = 0.149606;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 1) = 7.45426e-16;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-978);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.45426e-16;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-978);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-978);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 1) = 7.43408e-16;
    tri3_xyze(1, 1) = 0.149606;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7093);
    tri3_xyze(0, 2) = 0.0192843;
    tri3_xyze(1, 2) = 0.146479;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-978);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 7.45426e-16;
    tri3_xyze(1, 0) = 0.148429;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-979);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-979);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-979);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 1) = 7.45426e-16;
    tri3_xyze(1, 1) = 0.148429;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7095);
    tri3_xyze(0, 2) = 0.0190826;
    tri3_xyze(1, 2) = 0.144947;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-979);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00897e-15;
    tri3_xyze(1, 0) = 0.146489;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-980);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-980);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-980);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 1) = 1.00897e-15;
    tri3_xyze(1, 1) = 0.146489;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7097);
    tri3_xyze(0, 2) = 0.0187841;
    tri3_xyze(1, 2) = 0.142679;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-980);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00881e-15;
    tri3_xyze(1, 0) = 0.143815;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 1) = 1.00881e-15;
    tri3_xyze(1, 1) = 0.143815;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7099);
    tri3_xyze(0, 2) = 0.0183934;
    tri3_xyze(1, 2) = 0.139712;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-981);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0086e-15;
    tri3_xyze(1, 0) = 0.140451;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 1) = 1.0086e-15;
    tri3_xyze(1, 1) = 0.140451;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7101);
    tri3_xyze(0, 2) = 0.0179167;
    tri3_xyze(1, 2) = 0.136091;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-982);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00836e-15;
    tri3_xyze(1, 0) = 0.136448;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-983);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-983);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-983);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 1) = 1.00836e-15;
    tri3_xyze(1, 1) = 0.136448;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7103);
    tri3_xyze(0, 2) = 0.0173616;
    tri3_xyze(1, 2) = 0.131874;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-983);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00807e-15;
    tri3_xyze(1, 0) = 0.131871;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-984);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-984);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-984);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 1) = 1.00807e-15;
    tri3_xyze(1, 1) = 0.131871;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7105);
    tri3_xyze(0, 2) = 0.0167367;
    tri3_xyze(1, 2) = 0.127128;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-984);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.00776e-15;
    tri3_xyze(1, 0) = 0.126791;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 1) = 1.22288e-15;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-985);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.22288e-15;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-985);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-985);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 1) = 1.00776e-15;
    tri3_xyze(1, 1) = 0.126791;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7107);
    tri3_xyze(0, 2) = 0.016052;
    tri3_xyze(1, 2) = 0.121927;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-985);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.22288e-15;
    tri3_xyze(1, 0) = 0.121289;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 1) = 1.10961e-15;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-986);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.10961e-15;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-986);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-986);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 1) = 1.22288e-15;
    tri3_xyze(1, 1) = 0.121289;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7109);
    tri3_xyze(0, 2) = 0.0153182;
    tri3_xyze(1, 2) = 0.116353;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-986);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.10961e-15;
    tri3_xyze(1, 0) = 0.115451;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 1) = 8.12418e-16;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-987);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.12418e-16;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-987);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-987);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 1) = 1.10961e-15;
    tri3_xyze(1, 1) = 0.115451;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7111);
    tri3_xyze(0, 2) = 0.0145469;
    tri3_xyze(1, 2) = 0.110495;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-987);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 8.12418e-16;
    tri3_xyze(1, 0) = 0.109369;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 1) = 9.14709e-16;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-988);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.14709e-16;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(7365);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-988);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(7365);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-988);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 1) = 8.12418e-16;
    tri3_xyze(1, 1) = 0.109369;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7113);
    tri3_xyze(0, 2) = 0.0137503;
    tri3_xyze(1, 2) = 0.104444;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-988);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.14709e-16;
    tri3_xyze(1, 0) = 0.10314;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 1) = 1.00593e-15;
    tri3_xyze(1, 1) = 0.0968605;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(7117);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.849901;
    nids.push_back(-989);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.849901;
    nids.push_back(7365);
    tri3_xyze(0, 1) = 9.14709e-16;
    tri3_xyze(1, 1) = 0.10314;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(7115);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = 0.0982963;
    tri3_xyze(2, 2) = 0.849901;
    nids.push_back(-989);
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
    tri3_xyze(0, 0) = 0.0138497;
    tri3_xyze(1, 0) = 0.0516878;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7297);
    tri3_xyze(0, 1) = 0.0145417;
    tri3_xyze(1, 1) = 0.0542702;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 2) = 0.0208098;
    tri3_xyze(1, 2) = 0.0502394;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1005);
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
    tri3_xyze(0, 0) = 0.0145417;
    tri3_xyze(1, 0) = 0.0542702;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7299);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = 0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = 0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1006);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = 0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = 0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 2) = 0.0219553;
    tri3_xyze(1, 2) = 0.0530047;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1006);
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
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = 0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = 0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = 0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1007);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = 0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = 0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = 0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1007);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = 0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = 0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = 0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1007);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = 0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7551);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = 0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7301);
    tri3_xyze(0, 2) = 0.0233528;
    tri3_xyze(1, 2) = 0.0563786;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1007);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0164484;
    tri3_xyze(1, 0) = 0.0613861;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = 0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = 0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1008);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = 0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = 0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = 0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1008);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = 0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 1) = 0.0317758;
    tri3_xyze(1, 1) = 0.0550373;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = 0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1008);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = 0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 1) = 0.0164484;
    tri3_xyze(1, 1) = 0.0613861;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7303);
    tri3_xyze(0, 2) = 0.0249804;
    tri3_xyze(1, 2) = 0.060308;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1008);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.017633;
    tri3_xyze(1, 0) = 0.0658074;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = 0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = 0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1009);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = 0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = 0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = 0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1009);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = 0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = 0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = 0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1009);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = 0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 1) = 0.017633;
    tri3_xyze(1, 1) = 0.0658074;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7305);
    tri3_xyze(0, 2) = 0.0268124;
    tri3_xyze(1, 2) = 0.0647308;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1009);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0189478;
    tri3_xyze(1, 0) = 0.0707141;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = 0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = 0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1010);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = 0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = 0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = 0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1010);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = 0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = 0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = 0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1010);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = 0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 1) = 0.0189478;
    tri3_xyze(1, 1) = 0.0707141;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7307);
    tri3_xyze(0, 2) = 0.0288199;
    tri3_xyze(1, 2) = 0.0695774;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1010);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0203719;
    tri3_xyze(1, 0) = 0.076029;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = 0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = 0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1011);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = 0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = 0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = 0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1011);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = 0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = 0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = 0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1011);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = 0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 1) = 0.0203719;
    tri3_xyze(1, 1) = 0.076029;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7309);
    tri3_xyze(0, 2) = 0.0309712;
    tri3_xyze(1, 2) = 0.0747712;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1011);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = 0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = 0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = 0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1012);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = 0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = 0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = 0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1012);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = 0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = 0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = 0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1012);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = 0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = 0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7311);
    tri3_xyze(0, 2) = 0.0332325;
    tri3_xyze(1, 2) = 0.0802303;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1012);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.023457;
    tri3_xyze(1, 0) = 0.0875428;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = 0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = 0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1013);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = 0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = 0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = 0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1013);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = 0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = 0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = 0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1013);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = 0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 1) = 0.023457;
    tri3_xyze(1, 1) = 0.0875428;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7313);
    tri3_xyze(0, 2) = 0.035568;
    tri3_xyze(1, 2) = 0.0858688;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1013);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0250693;
    tri3_xyze(1, 0) = 0.09356;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = 0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-1014);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 1) = 0.0515698;
    tri3_xyze(1, 1) = 0.0893214;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7567);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = 0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-1014);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515698;
    tri3_xyze(1, 0) = 0.0893214;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7567);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = 0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = 0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-1014);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = 0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 1) = 0.0250693;
    tri3_xyze(1, 1) = 0.09356;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7315);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = 0.0915976;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-1014);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0266945;
    tri3_xyze(1, 0) = 0.0996251;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1015);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 1) = 0.0546845;
    tri3_xyze(1, 1) = 0.0947164;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7569);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1015);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0546845;
    tri3_xyze(1, 0) = 0.0947164;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7569);
    tri3_xyze(0, 1) = 0.0515698;
    tri3_xyze(1, 1) = 0.0893214;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7567);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1015);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515698;
    tri3_xyze(1, 0) = 0.0893214;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7567);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7317);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1015);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1016);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = 0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7571);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1016);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = 0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7571);
    tri3_xyze(0, 1) = 0.0546845;
    tri3_xyze(1, 1) = 0.0947164;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7569);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1016);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0546845;
    tri3_xyze(1, 0) = 0.0947164;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7569);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7319);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1016);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1017);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 1) = 0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1017);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = 0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7571);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1017);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = 0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7571);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7321);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1017);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1018);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 1) = 0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1018);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 1) = 0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1018);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7323);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1018);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1019);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 1) = 0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1019);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0659356;
    tri3_xyze(1, 0) = 0.114204;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 1) = 0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1019);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7325);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1019);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 2) = 0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1020);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 1) = 0.0682242;
    tri3_xyze(1, 1) = 0.118168;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 2) = 0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1020);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0682242;
    tri3_xyze(1, 0) = 0.118168;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 1) = 0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 2) = 0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1020);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0659356;
    tri3_xyze(1, 0) = 0.114204;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7327);
    tri3_xyze(0, 2) = 0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1020);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 2) = 0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1021);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = 0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 2) = 0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1021);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = 0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 1) = 0.0682242;
    tri3_xyze(1, 1) = 0.118168;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 2) = 0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1021);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0682242;
    tri3_xyze(1, 0) = 0.118168;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7329);
    tri3_xyze(0, 2) = 0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1021);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 2) = 0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1022);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 1) = 0.0719077;
    tri3_xyze(1, 1) = 0.124548;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 2) = 0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1022);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0719077;
    tri3_xyze(1, 0) = 0.124548;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = 0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 2) = 0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1022);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = 0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7331);
    tri3_xyze(0, 2) = 0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1022);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 2) = 0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1023);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 1) = 0.0732444;
    tri3_xyze(1, 1) = 0.126863;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 2) = 0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1023);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732444;
    tri3_xyze(1, 0) = 0.126863;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 1) = 0.0719077;
    tri3_xyze(1, 1) = 0.124548;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 2) = 0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1023);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0719077;
    tri3_xyze(1, 0) = 0.124548;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7333);
    tri3_xyze(0, 2) = 0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1023);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 2) = 0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1024);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 1) = 0.0742146;
    tri3_xyze(1, 1) = 0.128543;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 2) = 0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1024);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0742146;
    tri3_xyze(1, 0) = 0.128543;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 1) = 0.0732444;
    tri3_xyze(1, 1) = 0.126863;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 2) = 0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1024);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732444;
    tri3_xyze(1, 0) = 0.126863;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7335);
    tri3_xyze(0, 2) = 0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1024);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 2) = 0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1025);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 1) = 0.0748029;
    tri3_xyze(1, 1) = 0.129562;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 2) = 0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1025);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0748029;
    tri3_xyze(1, 0) = 0.129562;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 1) = 0.0742146;
    tri3_xyze(1, 1) = 0.128543;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 2) = 0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1025);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0742146;
    tri3_xyze(1, 0) = 0.128543;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7337);
    tri3_xyze(0, 2) = 0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1025);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 1) = 0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1026);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = 0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1026);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = 0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 1) = 0.0748029;
    tri3_xyze(1, 1) = 0.129562;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1026);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0748029;
    tri3_xyze(1, 0) = 0.129562;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7339);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1026);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0388229;
    tri3_xyze(1, 0) = 0.144889;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1027);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 1) = 0.0748029;
    tri3_xyze(1, 1) = 0.129562;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7593);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1027);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0748029;
    tri3_xyze(1, 0) = 0.129562;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7593);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = 0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1027);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = 0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 1) = 0.0388229;
    tri3_xyze(1, 1) = 0.144889;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7341);
    tri3_xyze(0, 2) = 0.0568366;
    tri3_xyze(1, 2) = 0.137216;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1027);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0387208;
    tri3_xyze(1, 0) = 0.144508;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 2) = 0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-1028);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 1) = 0.0742146;
    tri3_xyze(1, 1) = 0.128543;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7595);
    tri3_xyze(0, 2) = 0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-1028);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0748029;
    tri3_xyze(1, 0) = 0.129562;
    tri3_xyze(2, 0) = 0.806267;
    nids.push_back(7593);
    tri3_xyze(0, 1) = 0.0387208;
    tri3_xyze(1, 1) = 0.144508;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7343);
    tri3_xyze(0, 2) = 0.0565386;
    tri3_xyze(1, 2) = 0.136496;
    tri3_xyze(2, 2) = 0.809351;
    nids.push_back(-1028);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0384163;
    tri3_xyze(1, 0) = 0.143372;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 2) = 0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-1029);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 1) = 0.0732444;
    tri3_xyze(1, 1) = 0.126863;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7597);
    tri3_xyze(0, 2) = 0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-1029);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0742146;
    tri3_xyze(1, 0) = 0.128543;
    tri3_xyze(2, 0) = 0.812434;
    nids.push_back(7595);
    tri3_xyze(0, 1) = 0.0384163;
    tri3_xyze(1, 1) = 0.143372;
    tri3_xyze(2, 1) = 0.812434;
    nids.push_back(7345);
    tri3_xyze(0, 2) = 0.0559473;
    tri3_xyze(1, 2) = 0.135069;
    tri3_xyze(2, 2) = 0.81542;
    nids.push_back(-1029);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0379141;
    tri3_xyze(1, 0) = 0.141497;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 2) = 0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-1030);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 1) = 0.0719077;
    tri3_xyze(1, 1) = 0.124548;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7599);
    tri3_xyze(0, 2) = 0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-1030);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732444;
    tri3_xyze(1, 0) = 0.126863;
    tri3_xyze(2, 0) = 0.818406;
    nids.push_back(7597);
    tri3_xyze(0, 1) = 0.0379141;
    tri3_xyze(1, 1) = 0.141497;
    tri3_xyze(2, 1) = 0.818406;
    nids.push_back(7347);
    tri3_xyze(0, 2) = 0.0550721;
    tri3_xyze(1, 2) = 0.132956;
    tri3_xyze(2, 2) = 0.821247;
    nids.push_back(-1030);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0372221;
    tri3_xyze(1, 0) = 0.138915;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 2) = 0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-1031);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = 0.121634;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7601);
    tri3_xyze(0, 2) = 0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-1031);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0719077;
    tri3_xyze(1, 0) = 0.124548;
    tri3_xyze(2, 0) = 0.824088;
    nids.push_back(7599);
    tri3_xyze(0, 1) = 0.0372221;
    tri3_xyze(1, 1) = 0.138915;
    tri3_xyze(2, 1) = 0.824088;
    nids.push_back(7349);
    tri3_xyze(0, 2) = 0.0539266;
    tri3_xyze(1, 2) = 0.13019;
    tri3_xyze(2, 2) = 0.826738;
    nids.push_back(-1031);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0363514;
    tri3_xyze(1, 0) = 0.135665;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 2) = 0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-1032);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 1) = 0.0682242;
    tri3_xyze(1, 1) = 0.118168;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7603);
    tri3_xyze(0, 2) = 0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-1032);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = 0.121634;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(7601);
    tri3_xyze(0, 1) = 0.0363514;
    tri3_xyze(1, 1) = 0.135665;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(7351);
    tri3_xyze(0, 2) = 0.0525291;
    tri3_xyze(1, 2) = 0.126816;
    tri3_xyze(2, 2) = 0.831808;
    nids.push_back(-1032);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353155;
    tri3_xyze(1, 0) = 0.131799;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 2) = 0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-1033);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 1) = 0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7605);
    tri3_xyze(0, 2) = 0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-1033);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0682242;
    tri3_xyze(1, 0) = 0.118168;
    tri3_xyze(2, 0) = 0.834227;
    nids.push_back(7603);
    tri3_xyze(0, 1) = 0.0353155;
    tri3_xyze(1, 1) = 0.131799;
    tri3_xyze(2, 1) = 0.834227;
    nids.push_back(7353);
    tri3_xyze(0, 2) = 0.0509015;
    tri3_xyze(1, 2) = 0.122887;
    tri3_xyze(2, 2) = 0.836377;
    nids.push_back(-1033);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0341308;
    tri3_xyze(1, 0) = 0.127378;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-1034);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 1) = 0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7607);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-1034);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7607);
    tri3_xyze(0, 1) = 0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7605);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-1034);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0659356;
    tri3_xyze(1, 0) = 0.114204;
    tri3_xyze(2, 0) = 0.838526;
    nids.push_back(7605);
    tri3_xyze(0, 1) = 0.0341308;
    tri3_xyze(1, 1) = 0.127378;
    tri3_xyze(2, 1) = 0.838526;
    nids.push_back(7355);
    tri3_xyze(0, 2) = 0.0490695;
    tri3_xyze(1, 2) = 0.118464;
    tri3_xyze(2, 2) = 0.840371;
    nids.push_back(-1034);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.032816;
    tri3_xyze(1, 0) = 0.122471;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-1035);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 1) = 0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7609);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-1035);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7609);
    tri3_xyze(0, 1) = 0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7607);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-1035);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.842216;
    nids.push_back(7607);
    tri3_xyze(0, 1) = 0.032816;
    tri3_xyze(1, 1) = 0.122471;
    tri3_xyze(2, 1) = 0.842216;
    nids.push_back(7357);
    tri3_xyze(0, 2) = 0.047062;
    tri3_xyze(1, 2) = 0.113618;
    tri3_xyze(2, 2) = 0.843729;
    nids.push_back(-1035);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0313919;
    tri3_xyze(1, 0) = 0.117156;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-1036);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = 0.0999834;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7611);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-1036);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = 0.0999834;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7611);
    tri3_xyze(0, 1) = 0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7609);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-1036);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.845241;
    nids.push_back(7609);
    tri3_xyze(0, 1) = 0.0313919;
    tri3_xyze(1, 1) = 0.117156;
    tri3_xyze(2, 1) = 0.845241;
    nids.push_back(7359);
    tri3_xyze(0, 2) = 0.0449107;
    tri3_xyze(1, 2) = 0.108424;
    tri3_xyze(2, 2) = 0.846397;
    nids.push_back(-1036);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = 0.111517;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-1037);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 1) = 0.0546845;
    tri3_xyze(1, 1) = 0.0947164;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7613);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-1037);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0546845;
    tri3_xyze(1, 0) = 0.0947164;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7613);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = 0.0999834;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7611);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-1037);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = 0.0999834;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(7611);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = 0.111517;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(7361);
    tri3_xyze(0, 2) = 0.0426494;
    tri3_xyze(1, 2) = 0.102965;
    tri3_xyze(2, 2) = 0.848334;
    nids.push_back(-1037);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0283068;
    tri3_xyze(1, 0) = 0.105642;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 1) = 0.0266945;
    tri3_xyze(1, 1) = 0.0996251;
    tri3_xyze(2, 1) = 0.849901;
    nids.push_back(7365);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-1038);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0546845;
    tri3_xyze(1, 0) = 0.0947164;
    tri3_xyze(2, 0) = 0.849114;
    nids.push_back(7613);
    tri3_xyze(0, 1) = 0.0283068;
    tri3_xyze(1, 1) = 0.105642;
    tri3_xyze(2, 1) = 0.849114;
    nids.push_back(7363);
    tri3_xyze(0, 2) = 0.0403139;
    tri3_xyze(1, 2) = 0.0973263;
    tri3_xyze(2, 2) = 0.849508;
    nids.push_back(-1038);
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
    tri3_xyze(0, 0) = 0.0317758;
    tri3_xyze(1, 0) = 0.0550373;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7553);
    tri3_xyze(0, 1) = 0.0340644;
    tri3_xyze(1, 1) = 0.0590013;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 2) = 0.0397381;
    tri3_xyze(1, 2) = 0.0517877;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1058);
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
    tri3_xyze(0, 0) = 0.0340644;
    tri3_xyze(1, 0) = 0.0590013;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7555);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = 0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = 0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1059);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = 0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = 0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7807);
    tri3_xyze(0, 2) = 0.0426524;
    tri3_xyze(1, 2) = 0.0555856;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1059);
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
    tri3_xyze(0, 0) = 0.0366043;
    tri3_xyze(1, 0) = 0.0634006;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = 0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = 0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1060);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = 0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 1) = 0.0556571;
    tri3_xyze(1, 1) = 0.0556571;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7809);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = 0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1060);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0556571;
    tri3_xyze(1, 0) = 0.0556571;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7809);
    tri3_xyze(0, 1) = 0.0517663;
    tri3_xyze(1, 1) = 0.0517663;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7807);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = 0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1060);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0517663;
    tri3_xyze(1, 0) = 0.0517663;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7807);
    tri3_xyze(0, 1) = 0.0366043;
    tri3_xyze(1, 1) = 0.0634006;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7557);
    tri3_xyze(0, 2) = 0.0458458;
    tri3_xyze(1, 2) = 0.0597474;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1060);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0393555;
    tri3_xyze(1, 0) = 0.0681658;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = 0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = 0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1061);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = 0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = 0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7811);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = 0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1061);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = 0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7811);
    tri3_xyze(0, 1) = 0.0556571;
    tri3_xyze(1, 1) = 0.0556571;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7809);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = 0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1061);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0556571;
    tri3_xyze(1, 0) = 0.0556571;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7809);
    tri3_xyze(0, 1) = 0.0393555;
    tri3_xyze(1, 1) = 0.0681658;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7559);
    tri3_xyze(0, 2) = 0.0492681;
    tri3_xyze(1, 2) = 0.0642075;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1061);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = 0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = 0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 2) = 0.0528653;
    tri3_xyze(1, 2) = 0.0688954;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1062);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = 0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 1) = 0.0640857;
    tri3_xyze(1, 1) = 0.0640857;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7813);
    tri3_xyze(0, 2) = 0.0528653;
    tri3_xyze(1, 2) = 0.0688954;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1062);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = 0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7811);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = 0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(7561);
    tri3_xyze(0, 2) = 0.0528653;
    tri3_xyze(1, 2) = 0.0688954;
    tri3_xyze(2, 2) = 0.751666;
    nids.push_back(-1062);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0453155;
    tri3_xyze(1, 0) = 0.0784887;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = 0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 2) = 0.0565805;
    tri3_xyze(1, 2) = 0.0737372;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1063);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = 0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 1) = 0.0684907;
    tri3_xyze(1, 1) = 0.0684907;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7815);
    tri3_xyze(0, 2) = 0.0565805;
    tri3_xyze(1, 2) = 0.0737372;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1063);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0640857;
    tri3_xyze(1, 0) = 0.0640857;
    tri3_xyze(2, 0) = 0.750886;
    nids.push_back(7813);
    tri3_xyze(0, 1) = 0.0453155;
    tri3_xyze(1, 1) = 0.0784887;
    tri3_xyze(2, 1) = 0.750886;
    nids.push_back(7563);
    tri3_xyze(0, 2) = 0.0565805;
    tri3_xyze(1, 2) = 0.0737372;
    tri3_xyze(2, 2) = 0.750492;
    nids.push_back(-1063);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0484302;
    tri3_xyze(1, 0) = 0.0838836;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 1) = 0.0515698;
    tri3_xyze(1, 1) = 0.0893214;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7567);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = 0.0786566;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-1064);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0684907;
    tri3_xyze(1, 0) = 0.0684907;
    tri3_xyze(2, 0) = 0.750099;
    nids.push_back(7815);
    tri3_xyze(0, 1) = 0.0484302;
    tri3_xyze(1, 1) = 0.0838836;
    tri3_xyze(2, 1) = 0.750099;
    nids.push_back(7565);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = 0.0786566;
    tri3_xyze(2, 2) = 0.750099;
    nids.push_back(-1064);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = 0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(7571);
    tri3_xyze(0, 1) = 0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 2) = 0.0714426;
    tri3_xyze(1, 2) = 0.0931058;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1067);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0606445;
    tri3_xyze(1, 0) = 0.105039;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 1) = 0.0857642;
    tri3_xyze(1, 1) = 0.0857642;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7823);
    tri3_xyze(0, 2) = 0.0714426;
    tri3_xyze(1, 2) = 0.0931058;
    tri3_xyze(2, 2) = 0.753603;
    nids.push_back(-1067);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0857642;
    tri3_xyze(1, 0) = 0.0857642;
    tri3_xyze(2, 0) = 0.754759;
    nids.push_back(7823);
    tri3_xyze(0, 1) = 0.0606445;
    tri3_xyze(1, 1) = 0.105039;
    tri3_xyze(2, 1) = 0.754759;
    nids.push_back(7573);
    tri3_xyze(0, 2) = 0.0748649;
    tri3_xyze(1, 2) = 0.0975658;
    tri3_xyze(2, 2) = 0.756271;
    nids.push_back(-1068);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0633957;
    tri3_xyze(1, 0) = 0.109805;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 1) = 0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 2) = 0.0780583;
    tri3_xyze(1, 2) = 0.101728;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1069);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0659356;
    tri3_xyze(1, 0) = 0.114204;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 1) = 0.093247;
    tri3_xyze(1, 1) = 0.093247;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7827);
    tri3_xyze(0, 2) = 0.0780583;
    tri3_xyze(1, 2) = 0.101728;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1069);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.093247;
    tri3_xyze(1, 0) = 0.093247;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7827);
    tri3_xyze(0, 1) = 0.089655;
    tri3_xyze(1, 1) = 0.089655;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7825);
    tri3_xyze(0, 2) = 0.0780583;
    tri3_xyze(1, 2) = 0.101728;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1069);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.089655;
    tri3_xyze(1, 0) = 0.089655;
    tri3_xyze(2, 0) = 0.757784;
    nids.push_back(7825);
    tri3_xyze(0, 1) = 0.0633957;
    tri3_xyze(1, 1) = 0.109805;
    tri3_xyze(2, 1) = 0.757784;
    nids.push_back(7575);
    tri3_xyze(0, 2) = 0.0780583;
    tri3_xyze(1, 2) = 0.101728;
    tri3_xyze(2, 2) = 0.759629;
    nids.push_back(-1069);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0659356;
    tri3_xyze(1, 0) = 0.114204;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 1) = 0.0682242;
    tri3_xyze(1, 1) = 0.118168;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 2) = 0.0809726;
    tri3_xyze(1, 2) = 0.105526;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1070);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0682242;
    tri3_xyze(1, 0) = 0.118168;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 1) = 0.0964836;
    tri3_xyze(1, 1) = 0.0964836;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7829);
    tri3_xyze(0, 2) = 0.0809726;
    tri3_xyze(1, 2) = 0.105526;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1070);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0964836;
    tri3_xyze(1, 0) = 0.0964836;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7829);
    tri3_xyze(0, 1) = 0.093247;
    tri3_xyze(1, 1) = 0.093247;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7827);
    tri3_xyze(0, 2) = 0.0809726;
    tri3_xyze(1, 2) = 0.105526;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1070);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.093247;
    tri3_xyze(1, 0) = 0.093247;
    tri3_xyze(2, 0) = 0.761474;
    nids.push_back(7827);
    tri3_xyze(0, 1) = 0.0659356;
    tri3_xyze(1, 1) = 0.114204;
    tri3_xyze(2, 1) = 0.761474;
    nids.push_back(7577);
    tri3_xyze(0, 2) = 0.0809726;
    tri3_xyze(1, 2) = 0.105526;
    tri3_xyze(2, 2) = 0.763623;
    nids.push_back(-1070);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0682242;
    tri3_xyze(1, 0) = 0.118168;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = 0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 2) = 0.0835617;
    tri3_xyze(1, 2) = 0.1089;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1071);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = 0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = 0.0993137;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7831);
    tri3_xyze(0, 2) = 0.0835617;
    tri3_xyze(1, 2) = 0.1089;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1071);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = 0.0993137;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7831);
    tri3_xyze(0, 1) = 0.0964836;
    tri3_xyze(1, 1) = 0.0964836;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7829);
    tri3_xyze(0, 2) = 0.0835617;
    tri3_xyze(1, 2) = 0.1089;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1071);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0964836;
    tri3_xyze(1, 0) = 0.0964836;
    tri3_xyze(2, 0) = 0.765773;
    nids.push_back(7829);
    tri3_xyze(0, 1) = 0.0682242;
    tri3_xyze(1, 1) = 0.118168;
    tri3_xyze(2, 1) = 0.765773;
    nids.push_back(7579);
    tri3_xyze(0, 2) = 0.0835617;
    tri3_xyze(1, 2) = 0.1089;
    tri3_xyze(2, 2) = 0.768192;
    nids.push_back(-1071);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = 0.121634;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 1) = 0.0719077;
    tri3_xyze(1, 1) = 0.124548;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 2) = 0.0857849;
    tri3_xyze(1, 2) = 0.111797;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1072);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0719077;
    tri3_xyze(1, 0) = 0.124548;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 1) = 0.101693;
    tri3_xyze(1, 1) = 0.101693;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7833);
    tri3_xyze(0, 2) = 0.0857849;
    tri3_xyze(1, 2) = 0.111797;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1072);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.101693;
    tri3_xyze(1, 0) = 0.101693;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7833);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = 0.0993137;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7831);
    tri3_xyze(0, 2) = 0.0857849;
    tri3_xyze(1, 2) = 0.111797;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1072);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = 0.0993137;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7831);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = 0.121634;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(7581);
    tri3_xyze(0, 2) = 0.0857849;
    tri3_xyze(1, 2) = 0.111797;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1072);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0719077;
    tri3_xyze(1, 0) = 0.124548;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 1) = 0.0732444;
    tri3_xyze(1, 1) = 0.126863;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 2) = 0.087607;
    tri3_xyze(1, 2) = 0.114172;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1073);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732444;
    tri3_xyze(1, 0) = 0.126863;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 1) = 0.103583;
    tri3_xyze(1, 1) = 0.103583;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7835);
    tri3_xyze(0, 2) = 0.087607;
    tri3_xyze(1, 2) = 0.114172;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1073);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.103583;
    tri3_xyze(1, 0) = 0.103583;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7835);
    tri3_xyze(0, 1) = 0.101693;
    tri3_xyze(1, 1) = 0.101693;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7833);
    tri3_xyze(0, 2) = 0.087607;
    tri3_xyze(1, 2) = 0.114172;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1073);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.101693;
    tri3_xyze(1, 0) = 0.101693;
    tri3_xyze(2, 0) = 0.775912;
    nids.push_back(7833);
    tri3_xyze(0, 1) = 0.0719077;
    tri3_xyze(1, 1) = 0.124548;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7583);
    tri3_xyze(0, 2) = 0.087607;
    tri3_xyze(1, 2) = 0.114172;
    tri3_xyze(2, 2) = 0.778753;
    nids.push_back(-1073);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732444;
    tri3_xyze(1, 0) = 0.126863;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 1) = 0.0742146;
    tri3_xyze(1, 1) = 0.128543;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 2) = 0.0889994;
    tri3_xyze(1, 2) = 0.115986;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1074);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0742146;
    tri3_xyze(1, 0) = 0.128543;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 1) = 0.104955;
    tri3_xyze(1, 1) = 0.104955;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7837);
    tri3_xyze(0, 2) = 0.0889994;
    tri3_xyze(1, 2) = 0.115986;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1074);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.104955;
    tri3_xyze(1, 0) = 0.104955;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7837);
    tri3_xyze(0, 1) = 0.103583;
    tri3_xyze(1, 1) = 0.103583;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7835);
    tri3_xyze(0, 2) = 0.0889994;
    tri3_xyze(1, 2) = 0.115986;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1074);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.103583;
    tri3_xyze(1, 0) = 0.103583;
    tri3_xyze(2, 0) = 0.781594;
    nids.push_back(7835);
    tri3_xyze(0, 1) = 0.0732444;
    tri3_xyze(1, 1) = 0.126863;
    tri3_xyze(2, 1) = 0.781594;
    nids.push_back(7585);
    tri3_xyze(0, 2) = 0.0889994;
    tri3_xyze(1, 2) = 0.115986;
    tri3_xyze(2, 2) = 0.78458;
    nids.push_back(-1074);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0742146;
    tri3_xyze(1, 0) = 0.128543;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 1) = 0.0748029;
    tri3_xyze(1, 1) = 0.129562;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 2) = 0.08994;
    tri3_xyze(1, 2) = 0.117212;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1075);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0748029;
    tri3_xyze(1, 0) = 0.129562;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 1) = 0.105787;
    tri3_xyze(1, 1) = 0.105787;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7839);
    tri3_xyze(0, 2) = 0.08994;
    tri3_xyze(1, 2) = 0.117212;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1075);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.105787;
    tri3_xyze(1, 0) = 0.105787;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7839);
    tri3_xyze(0, 1) = 0.104955;
    tri3_xyze(1, 1) = 0.104955;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7837);
    tri3_xyze(0, 2) = 0.08994;
    tri3_xyze(1, 2) = 0.117212;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1075);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.104955;
    tri3_xyze(1, 0) = 0.104955;
    tri3_xyze(2, 0) = 0.787566;
    nids.push_back(7837);
    tri3_xyze(0, 1) = 0.0742146;
    tri3_xyze(1, 1) = 0.128543;
    tri3_xyze(2, 1) = 0.787566;
    nids.push_back(7587);
    tri3_xyze(0, 2) = 0.08994;
    tri3_xyze(1, 2) = 0.117212;
    tri3_xyze(2, 2) = 0.790649;
    nids.push_back(-1075);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0748029;
    tri3_xyze(1, 0) = 0.129562;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = 0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 2) = 0.090414;
    tri3_xyze(1, 2) = 0.11783;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1076);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = 0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 1) = 0.106066;
    tri3_xyze(1, 1) = 0.106066;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7841);
    tri3_xyze(0, 2) = 0.090414;
    tri3_xyze(1, 2) = 0.11783;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1076);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.106066;
    tri3_xyze(1, 0) = 0.106066;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7841);
    tri3_xyze(0, 1) = 0.105787;
    tri3_xyze(1, 1) = 0.105787;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7839);
    tri3_xyze(0, 2) = 0.090414;
    tri3_xyze(1, 2) = 0.11783;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1076);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.105787;
    tri3_xyze(1, 0) = 0.105787;
    tri3_xyze(2, 0) = 0.793733;
    nids.push_back(7839);
    tri3_xyze(0, 1) = 0.0748029;
    tri3_xyze(1, 1) = 0.129562;
    tri3_xyze(2, 1) = 0.793733;
    nids.push_back(7589);
    tri3_xyze(0, 2) = 0.090414;
    tri3_xyze(1, 2) = 0.11783;
    tri3_xyze(2, 2) = 0.796867;
    nids.push_back(-1076);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.075;
    tri3_xyze(1, 0) = 0.129904;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 1) = 0.0748029;
    tri3_xyze(1, 1) = 0.129562;
    tri3_xyze(2, 1) = 0.806267;
    nids.push_back(7593);
    tri3_xyze(0, 2) = 0.090414;
    tri3_xyze(1, 2) = 0.11783;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1077);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.106066;
    tri3_xyze(1, 0) = 0.106066;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(7841);
    tri3_xyze(0, 1) = 0.075;
    tri3_xyze(1, 1) = 0.129904;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(7591);
    tri3_xyze(0, 2) = 0.090414;
    tri3_xyze(1, 2) = 0.11783;
    tri3_xyze(2, 2) = 0.803133;
    nids.push_back(-1077);
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
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = 0.0993137;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(7831);
    tri3_xyze(0, 1) = 0.101693;
    tri3_xyze(1, 1) = 0.101693;
    tri3_xyze(2, 1) = 0.775912;
    nids.push_back(7833);
    tri3_xyze(0, 2) = 0.111797;
    tri3_xyze(1, 2) = 0.0857849;
    tri3_xyze(2, 2) = 0.773262;
    nids.push_back(-1122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.1;
    hex8_xyze(1, 0) = 0.1;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1855);
    hex8_xyze(0, 1) = 0.1;
    hex8_xyze(1, 1) = 0.15;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1856);
    hex8_xyze(0, 2) = 0.05;
    hex8_xyze(1, 2) = 0.15;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1867);
    hex8_xyze(0, 3) = 0.05;
    hex8_xyze(1, 3) = 0.1;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1866);
    hex8_xyze(0, 4) = 0.1;
    hex8_xyze(1, 4) = 0.1;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1976);
    hex8_xyze(0, 5) = 0.1;
    hex8_xyze(1, 5) = 0.15;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1977);
    hex8_xyze(0, 6) = 0.05;
    hex8_xyze(1, 6) = 0.15;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1988);
    hex8_xyze(0, 7) = 0.05;
    hex8_xyze(1, 7) = 0.1;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1987);

    intersection.add_element(6913, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = 0.05;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1865);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = 0.1;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1866);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0.1;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1877);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = 0.05;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1876);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = 0.05;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1986);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = 0.1;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1987);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0.1;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1998);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = 0.05;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1997);

    intersection.add_element(6922, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = 0.1;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1866);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = 0.15;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1867);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0.15;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1878);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = 0.1;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1877);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = 0.1;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1987);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = 0.15;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1988);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0.15;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1999);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = 0.1;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1998);

    intersection.add_element(6923, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = 0.15;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1867);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = 0.2;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1868);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0.2;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1879);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = 0.15;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1878);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = 0.15;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1988);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = 0.2;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1989);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0.2;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(2000);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = 0.15;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1999);

    intersection.add_element(6924, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0;
    hex8_xyze(1, 0) = 0.1;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1877);
    hex8_xyze(0, 1) = 0;
    hex8_xyze(1, 1) = 0.15;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1878);
    hex8_xyze(0, 2) = -0.05;
    hex8_xyze(1, 2) = 0.15;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1889);
    hex8_xyze(0, 3) = -0.05;
    hex8_xyze(1, 3) = 0.1;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1888);
    hex8_xyze(0, 4) = 0;
    hex8_xyze(1, 4) = 0.1;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1998);
    hex8_xyze(0, 5) = 0;
    hex8_xyze(1, 5) = 0.15;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1999);
    hex8_xyze(0, 6) = -0.05;
    hex8_xyze(1, 6) = 0.15;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(2010);
    hex8_xyze(0, 7) = -0.05;
    hex8_xyze(1, 7) = 0.1;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(2009);

    intersection.add_element(6933, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = 0.1;
    hex8_xyze(2, 0) = 0.8;
    nids.push_back(1987);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = 0.15;
    hex8_xyze(2, 1) = 0.8;
    nids.push_back(1988);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0.15;
    hex8_xyze(2, 2) = 0.8;
    nids.push_back(1999);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = 0.1;
    hex8_xyze(2, 3) = 0.8;
    nids.push_back(1998);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = 0.1;
    hex8_xyze(2, 4) = 0.85;
    nids.push_back(2108);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = 0.15;
    hex8_xyze(2, 5) = 0.85;
    nids.push_back(2109);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0.15;
    hex8_xyze(2, 6) = 0.85;
    nids.push_back(2120);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = 0.1;
    hex8_xyze(2, 7) = 0.85;
    nids.push_back(2119);

    intersection.add_element(7023, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
