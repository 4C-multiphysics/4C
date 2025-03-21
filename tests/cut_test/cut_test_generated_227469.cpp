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

void test_generated_227469()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39923325095054695844;
    tri3_xyze(1, 0) = 0.12498513649226991595;
    tri3_xyze(2, 0) = -0.068666081750516971827;
    nids.push_back(43663);
    tri3_xyze(0, 1) = 0.39915485709905629275;
    tri3_xyze(1, 1) = 0.1250964988895283958;
    tri3_xyze(2, 1) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 2) = 0.3986821010616874994;
    tri3_xyze(1, 2) = 0.13027243866924168025;
    tri3_xyze(2, 2) = -0.064640500655890095749;
    nids.push_back(-7292);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39915485709905629275;
    tri3_xyze(1, 0) = 0.1250964988895283958;
    tri3_xyze(2, 0) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 1) = 0.39813320825728015784;
    tri3_xyze(1, 1) = 0.13555839597567348465;
    tri3_xyze(2, 1) = -0.060617933766494866421;
    nids.push_back(43825);
    tri3_xyze(0, 2) = 0.3986821010616874994;
    tri3_xyze(1, 2) = 0.13027243866924168025;
    tri3_xyze(2, 2) = -0.064640500655890095749;
    nids.push_back(-7292);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39813320825728015784;
    tri3_xyze(1, 0) = 0.13555839597567348465;
    tri3_xyze(2, 0) = -0.060617933766494866421;
    nids.push_back(43825);
    tri3_xyze(0, 1) = 0.39820708793986664409;
    tri3_xyze(1, 1) = 0.13544972331949495237;
    tri3_xyze(2, 1) = -0.068627140607473172129;
    nids.push_back(43659);
    tri3_xyze(0, 2) = 0.3986821010616874994;
    tri3_xyze(1, 2) = 0.13027243866924168025;
    tri3_xyze(2, 2) = -0.064640500655890095749;
    nids.push_back(-7292);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40037153502305483643;
    tri3_xyze(1, 0) = 0.11444113470552699785;
    tri3_xyze(2, 0) = -0.076725147700746246238;
    nids.push_back(43666);
    tri3_xyze(0, 1) = 0.40018604020638171015;
    tri3_xyze(1, 1) = 0.11453152966691711179;
    tri3_xyze(2, 1) = -0.068703503338040552983;
    nids.push_back(43667);
    tri3_xyze(0, 2) = 0.39979974054018335705;
    tri3_xyze(1, 2) = 0.11971383039320199204;
    tri3_xyze(2, 2) = -0.072693577896014036077;
    nids.push_back(-7295);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40018604020638171015;
    tri3_xyze(1, 0) = 0.11453152966691711179;
    tri3_xyze(2, 0) = -0.068703503338040552983;
    nids.push_back(43667);
    tri3_xyze(0, 1) = 0.39923325095054695844;
    tri3_xyze(1, 1) = 0.12498513649226991595;
    tri3_xyze(2, 1) = -0.068666081750516971827;
    nids.push_back(43663);
    tri3_xyze(0, 2) = 0.39979974054018335705;
    tri3_xyze(1, 2) = 0.11971383039320199204;
    tri3_xyze(2, 2) = -0.072693577896014036077;
    nids.push_back(-7295);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39923325095054695844;
    tri3_xyze(1, 0) = 0.12498513649226991595;
    tri3_xyze(2, 0) = -0.068666081750516971827;
    nids.push_back(43663);
    tri3_xyze(0, 1) = 0.39940813598075003421;
    tri3_xyze(1, 1) = 0.12489752070809394258;
    tri3_xyze(2, 1) = -0.076679578794752373261;
    nids.push_back(43662);
    tri3_xyze(0, 2) = 0.39979974054018335705;
    tri3_xyze(1, 2) = 0.11971383039320199204;
    tri3_xyze(2, 2) = -0.072693577896014036077;
    nids.push_back(-7295);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39923325095054695844;
    tri3_xyze(1, 0) = 0.12498513649226991595;
    tri3_xyze(2, 0) = -0.068666081750516971827;
    nids.push_back(43663);
    tri3_xyze(0, 1) = 0.40018604020638171015;
    tri3_xyze(1, 1) = 0.11453152966691711179;
    tri3_xyze(2, 1) = -0.068703503338040552983;
    nids.push_back(43667);
    tri3_xyze(0, 2) = 0.39966924525659142109;
    tri3_xyze(1, 2) = 0.11981461760862495425;
    tri3_xyze(2, 2) = -0.064675342517410694398;
    nids.push_back(-7296);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40018604020638171015;
    tri3_xyze(1, 0) = 0.11453152966691711179;
    tri3_xyze(2, 0) = -0.068703503338040552983;
    nids.push_back(43667);
    tri3_xyze(0, 1) = 0.400102832770380612;
    tri3_xyze(1, 1) = 0.11464530538578435181;
    tri3_xyze(2, 1) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 2) = 0.39966924525659142109;
    tri3_xyze(1, 2) = 0.11981461760862495425;
    tri3_xyze(2, 2) = -0.064675342517410694398;
    nids.push_back(-7296);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.400102832770380612;
    tri3_xyze(1, 0) = 0.11464530538578435181;
    tri3_xyze(2, 0) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 1) = 0.39915485709905629275;
    tri3_xyze(1, 1) = 0.1250964988895283958;
    tri3_xyze(2, 1) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 2) = 0.39966924525659142109;
    tri3_xyze(1, 2) = 0.11981461760862495425;
    tri3_xyze(2, 2) = -0.064675342517410694398;
    nids.push_back(-7296);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39915485709905629275;
    tri3_xyze(1, 0) = 0.1250964988895283958;
    tri3_xyze(2, 0) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 1) = 0.39923325095054695844;
    tri3_xyze(1, 1) = 0.12498513649226991595;
    tri3_xyze(2, 1) = -0.068666081750516971827;
    nids.push_back(43663);
    tri3_xyze(0, 2) = 0.39966924525659142109;
    tri3_xyze(1, 2) = 0.11981461760862495425;
    tri3_xyze(2, 2) = -0.064675342517410694398;
    nids.push_back(-7296);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40097530024400784843;
    tri3_xyze(1, 0) = 0.10420353847874216924;
    tri3_xyze(2, 0) = -0.060705981549868566483;
    nids.push_back(43831);
    tri3_xyze(0, 1) = 0.400102832770380612;
    tri3_xyze(1, 1) = 0.11464530538578435181;
    tri3_xyze(2, 1) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 2) = 0.40058155937148925974;
    tri3_xyze(1, 2) = 0.10936687853235196877;
    tri3_xyze(2, 2) = -0.064706428328296206054;
    nids.push_back(-7300);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.400102832770380612;
    tri3_xyze(1, 0) = 0.11464530538578435181;
    tri3_xyze(2, 0) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 1) = 0.40018604020638171015;
    tri3_xyze(1, 1) = 0.11453152966691711179;
    tri3_xyze(2, 1) = -0.068703503338040552983;
    nids.push_back(43667);
    tri3_xyze(0, 2) = 0.40058155937148925974;
    tri3_xyze(1, 2) = 0.10936687853235196877;
    tri3_xyze(2, 2) = -0.064706428328296206054;
    nids.push_back(-7300);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39813320825728015784;
    tri3_xyze(1, 0) = 0.13555839597567348465;
    tri3_xyze(2, 0) = -0.060617933766494866421;
    nids.push_back(43825);
    tri3_xyze(0, 1) = 0.39915485709905629275;
    tri3_xyze(1, 1) = 0.1250964988895283958;
    tri3_xyze(2, 1) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 2) = 0.39862395996947302956;
    tri3_xyze(1, 2) = 0.13038699885142771007;
    tri3_xyze(2, 2) = -0.056630195917963623009;
    nids.push_back(-7448);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39915485709905629275;
    tri3_xyze(1, 0) = 0.1250964988895283958;
    tri3_xyze(2, 0) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 1) = 0.39911337699962468051;
    tri3_xyze(1, 1) = 0.12521692887577934306;
    tri3_xyze(2, 1) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 2) = 0.39862395996947302956;
    tri3_xyze(1, 2) = 0.13038699885142771007;
    tri3_xyze(2, 2) = -0.056630195917963623009;
    nids.push_back(-7448);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39911337699962468051;
    tri3_xyze(1, 0) = 0.12521692887577934306;
    tri3_xyze(2, 0) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 1) = 0.39809439752193104267;
    tri3_xyze(1, 1) = 0.13567617166472964452;
    tri3_xyze(2, 1) = -0.052612388930145463639;
    nids.push_back(43919);
    tri3_xyze(0, 2) = 0.39862395996947302956;
    tri3_xyze(1, 2) = 0.13038699885142771007;
    tri3_xyze(2, 2) = -0.056630195917963623009;
    nids.push_back(-7448);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39809439752193104267;
    tri3_xyze(1, 0) = 0.13567617166472964452;
    tri3_xyze(2, 0) = -0.052612388930145463639;
    nids.push_back(43919);
    tri3_xyze(0, 1) = 0.39813320825728015784;
    tri3_xyze(1, 1) = 0.13555839597567348465;
    tri3_xyze(2, 1) = -0.060617933766494866421;
    nids.push_back(43825);
    tri3_xyze(0, 2) = 0.39862395996947302956;
    tri3_xyze(1, 2) = 0.13038699885142771007;
    tri3_xyze(2, 2) = -0.056630195917963623009;
    nids.push_back(-7448);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39915485709905629275;
    tri3_xyze(1, 0) = 0.1250964988895283958;
    tri3_xyze(2, 0) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 1) = 0.400102832770380612;
    tri3_xyze(1, 1) = 0.11464530538578435181;
    tri3_xyze(2, 1) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 2) = 0.39960741043895364744;
    tri3_xyze(1, 2) = 0.11993170226773146314;
    tri3_xyze(2, 2) = -0.056658716190387758971;
    nids.push_back(-7449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.400102832770380612;
    tri3_xyze(1, 0) = 0.11464530538578435181;
    tri3_xyze(2, 0) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 1) = 0.40005857488675306;
    tri3_xyze(1, 1) = 0.11476807591983374801;
    tri3_xyze(2, 1) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 2) = 0.39960741043895364744;
    tri3_xyze(1, 2) = 0.11993170226773146314;
    tri3_xyze(2, 2) = -0.056658716190387758971;
    nids.push_back(-7449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40005857488675306;
    tri3_xyze(1, 0) = 0.11476807591983374801;
    tri3_xyze(2, 0) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 1) = 0.39911337699962468051;
    tri3_xyze(1, 1) = 0.12521692887577934306;
    tri3_xyze(2, 1) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 2) = 0.39960741043895364744;
    tri3_xyze(1, 2) = 0.11993170226773146314;
    tri3_xyze(2, 2) = -0.056658716190387758971;
    nids.push_back(-7449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39911337699962468051;
    tri3_xyze(1, 0) = 0.12521692887577934306;
    tri3_xyze(2, 0) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 1) = 0.39915485709905629275;
    tri3_xyze(1, 1) = 0.1250964988895283958;
    tri3_xyze(2, 1) = -0.06065084649907537262;
    nids.push_back(43827);
    tri3_xyze(0, 2) = 0.39960741043895364744;
    tri3_xyze(1, 2) = 0.11993170226773146314;
    tri3_xyze(2, 2) = -0.056658716190387758971;
    nids.push_back(-7449);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.400102832770380612;
    tri3_xyze(1, 0) = 0.11464530538578435181;
    tri3_xyze(2, 0) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 1) = 0.40097530024400784843;
    tri3_xyze(1, 1) = 0.10420353847874216924;
    tri3_xyze(2, 1) = -0.060705981549868566483;
    nids.push_back(43831);
    tri3_xyze(0, 2) = 0.40051633270656356034;
    tri3_xyze(1, 2) = 0.10948634222953099182;
    tri3_xyze(2, 2) = -0.056683227688008659684;
    nids.push_back(-7450);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40092862292511266542;
    tri3_xyze(1, 0) = 0.10432844913376372598;
    tri3_xyze(2, 0) = -0.052682525415829177529;
    nids.push_back(43925);
    tri3_xyze(0, 1) = 0.40005857488675306;
    tri3_xyze(1, 1) = 0.11476807591983374801;
    tri3_xyze(2, 1) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 2) = 0.40051633270656356034;
    tri3_xyze(1, 2) = 0.10948634222953099182;
    tri3_xyze(2, 2) = -0.056683227688008659684;
    nids.push_back(-7450);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40005857488675306;
    tri3_xyze(1, 0) = 0.11476807591983374801;
    tri3_xyze(2, 0) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 1) = 0.400102832770380612;
    tri3_xyze(1, 1) = 0.11464530538578435181;
    tri3_xyze(2, 1) = -0.060680938482009928736;
    nids.push_back(43829);
    tri3_xyze(0, 2) = 0.40051633270656356034;
    tri3_xyze(1, 2) = 0.10948634222953099182;
    tri3_xyze(2, 2) = -0.056683227688008659684;
    nids.push_back(-7450);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39809439752193104267;
    tri3_xyze(1, 0) = 0.13567617166472964452;
    tri3_xyze(2, 0) = -0.052612388930145463639;
    nids.push_back(43919);
    tri3_xyze(0, 1) = 0.39911337699962468051;
    tri3_xyze(1, 1) = 0.12521692887577934306;
    tri3_xyze(2, 1) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 2) = 0.39859461469321760818;
    tri3_xyze(1, 2) = 0.13050886921475512992;
    tri3_xyze(2, 2) = -0.048624306974148108484;
    nids.push_back(-7496);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39911337699962468051;
    tri3_xyze(1, 0) = 0.12521692887577934306;
    tri3_xyze(2, 0) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 1) = 0.39909405407954939848;
    tri3_xyze(1, 1) = 0.1253427488037150983;
    tri3_xyze(2, 1) = -0.044633506273407334841;
    nids.push_back(44015);
    tri3_xyze(0, 2) = 0.39859461469321760818;
    tri3_xyze(1, 2) = 0.13050886921475512992;
    tri3_xyze(2, 2) = -0.048624306974148108484;
    nids.push_back(-7496);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39807663017176525555;
    tri3_xyze(1, 0) = 0.13579962751479640604;
    tri3_xyze(2, 0) = -0.04461171821690085304;
    nids.push_back(44013);
    tri3_xyze(0, 1) = 0.39809439752193104267;
    tri3_xyze(1, 1) = 0.13567617166472964452;
    tri3_xyze(2, 1) = -0.052612388930145463639;
    nids.push_back(43919);
    tri3_xyze(0, 2) = 0.39859461469321760818;
    tri3_xyze(1, 2) = 0.13050886921475512992;
    tri3_xyze(2, 2) = -0.048624306974148108484;
    nids.push_back(-7496);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39911337699962468051;
    tri3_xyze(1, 0) = 0.12521692887577934306;
    tri3_xyze(2, 0) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 1) = 0.40005857488675306;
    tri3_xyze(1, 1) = 0.11476807591983374801;
    tri3_xyze(2, 1) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 2) = 0.39957594726116790751;
    tri3_xyze(1, 2) = 0.12005593335467884542;
    tri3_xyze(2, 2) = -0.048647122743231956121;
    nids.push_back(-7497);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40005857488675306;
    tri3_xyze(1, 0) = 0.11476807591983374801;
    tri3_xyze(2, 0) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 1) = 0.40003778307874449105;
    tri3_xyze(1, 1) = 0.11489597981938723392;
    tri3_xyze(2, 1) = -0.0446519049190547343;
    nids.push_back(44017);
    tri3_xyze(0, 2) = 0.39957594726116790751;
    tri3_xyze(1, 2) = 0.12005593335467884542;
    tri3_xyze(2, 2) = -0.048647122743231956121;
    nids.push_back(-7497);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.39909405407954939848;
    tri3_xyze(1, 0) = 0.1253427488037150983;
    tri3_xyze(2, 0) = -0.044633506273407334841;
    nids.push_back(44015);
    tri3_xyze(0, 1) = 0.39911337699962468051;
    tri3_xyze(1, 1) = 0.12521692887577934306;
    tri3_xyze(2, 1) = -0.052639614476138775478;
    nids.push_back(43921);
    tri3_xyze(0, 2) = 0.39957594726116790751;
    tri3_xyze(1, 2) = 0.12005593335467884542;
    tri3_xyze(2, 2) = -0.048647122743231956121;
    nids.push_back(-7497);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40005857488675306;
    tri3_xyze(1, 0) = 0.11476807591983374801;
    tri3_xyze(2, 0) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 1) = 0.40092862292511266542;
    tri3_xyze(1, 1) = 0.10432844913376372598;
    tri3_xyze(2, 1) = -0.052682525415829177529;
    nids.push_back(43925);
    tri3_xyze(0, 2) = 0.4004828638209200764;
    tri3_xyze(1, 2) = 0.109612669285296227;
    tri3_xyze(2, 2) = -0.048665964493397384505;
    nids.push_back(-7498);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.40003778307874449105;
    tri3_xyze(1, 0) = 0.11489597981938723392;
    tri3_xyze(2, 0) = -0.0446519049190547343;
    nids.push_back(44017);
    tri3_xyze(0, 1) = 0.40005857488675306;
    tri3_xyze(1, 1) = 0.11476807591983374801;
    tri3_xyze(2, 1) = -0.052663465304326965988;
    nids.push_back(43923);
    tri3_xyze(0, 2) = 0.4004828638209200764;
    tri3_xyze(1, 2) = 0.109612669285296227;
    tri3_xyze(2, 2) = -0.048665964493397384505;
    nids.push_back(-7498);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.37777777777777798995;
    hex8_xyze(1, 0) = 0.11111111111111099392;
    hex8_xyze(2, 0) = -0.050000000000000002776;
    nids.push_back(1245398);
    hex8_xyze(0, 1) = 0.37777777777777798995;
    hex8_xyze(1, 1) = 0.11111111111111099392;
    hex8_xyze(2, 1) = -0.061111111111111102168;
    nids.push_back(1245720);
    hex8_xyze(0, 2) = 0.37777777777777798995;
    hex8_xyze(1, 2) = 0.12222222222222199617;
    hex8_xyze(2, 2) = -0.061111111111111102168;
    nids.push_back(1245723);
    hex8_xyze(0, 3) = 0.37777777777777798995;
    hex8_xyze(1, 3) = 0.12222222222222199617;
    hex8_xyze(2, 3) = -0.050000000000000002776;
    nids.push_back(1245401);
    hex8_xyze(0, 4) = 0.38888888888888900608;
    hex8_xyze(1, 4) = 0.11111111111111099392;
    hex8_xyze(2, 4) = -0.050000000000000002776;
    nids.push_back(1245407);
    hex8_xyze(0, 5) = 0.38888888888888900608;
    hex8_xyze(1, 5) = 0.11111111111111099392;
    hex8_xyze(2, 5) = -0.061111111111111102168;
    nids.push_back(1245729);
    hex8_xyze(0, 6) = 0.38888888888888900608;
    hex8_xyze(1, 6) = 0.12222222222222199617;
    hex8_xyze(2, 6) = -0.061111111111111102168;
    nids.push_back(1245732);
    hex8_xyze(0, 7) = 0.38888888888888900608;
    hex8_xyze(1, 7) = 0.12222222222222199617;
    hex8_xyze(2, 7) = -0.050000000000000002776;
    nids.push_back(1245410);

    intersection.add_element(227460, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.38888888888888900608;
    hex8_xyze(1, 0) = 0.10000000000000000555;
    hex8_xyze(2, 0) = -0.050000000000000002776;
    nids.push_back(1242496);
    hex8_xyze(0, 1) = 0.38888888888888900608;
    hex8_xyze(1, 1) = 0.10000000000000000555;
    hex8_xyze(2, 1) = -0.061111111111111102168;
    nids.push_back(1242495);
    hex8_xyze(0, 2) = 0.38888888888888900608;
    hex8_xyze(1, 2) = 0.11111111111111099392;
    hex8_xyze(2, 2) = -0.061111111111111102168;
    nids.push_back(1245729);
    hex8_xyze(0, 3) = 0.38888888888888900608;
    hex8_xyze(1, 3) = 0.11111111111111099392;
    hex8_xyze(2, 3) = -0.050000000000000002776;
    nids.push_back(1245407);
    hex8_xyze(0, 4) = 0.4000000000000000222;
    hex8_xyze(1, 4) = 0.10000000000000000555;
    hex8_xyze(2, 4) = -0.050000000000000002776;
    nids.push_back(1242505);
    hex8_xyze(0, 5) = 0.4000000000000000222;
    hex8_xyze(1, 5) = 0.10000000000000000555;
    hex8_xyze(2, 5) = -0.061111111111111102168;
    nids.push_back(1242504);
    hex8_xyze(0, 6) = 0.4000000000000000222;
    hex8_xyze(1, 6) = 0.11111111111111099392;
    hex8_xyze(2, 6) = -0.061111111111111102168;
    nids.push_back(1245738);
    hex8_xyze(0, 7) = 0.4000000000000000222;
    hex8_xyze(1, 7) = 0.11111111111111099392;
    hex8_xyze(2, 7) = -0.050000000000000002776;
    nids.push_back(1245416);

    intersection.add_element(227466, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.38888888888888900608;
    hex8_xyze(1, 0) = 0.11111111111111099392;
    hex8_xyze(2, 0) = -0.050000000000000002776;
    nids.push_back(1245407);
    hex8_xyze(0, 1) = 0.38888888888888900608;
    hex8_xyze(1, 1) = 0.11111111111111099392;
    hex8_xyze(2, 1) = -0.061111111111111102168;
    nids.push_back(1245729);
    hex8_xyze(0, 2) = 0.38888888888888900608;
    hex8_xyze(1, 2) = 0.12222222222222199617;
    hex8_xyze(2, 2) = -0.061111111111111102168;
    nids.push_back(1245732);
    hex8_xyze(0, 3) = 0.38888888888888900608;
    hex8_xyze(1, 3) = 0.12222222222222199617;
    hex8_xyze(2, 3) = -0.050000000000000002776;
    nids.push_back(1245410);
    hex8_xyze(0, 4) = 0.4000000000000000222;
    hex8_xyze(1, 4) = 0.11111111111111099392;
    hex8_xyze(2, 4) = -0.050000000000000002776;
    nids.push_back(1245416);
    hex8_xyze(0, 5) = 0.4000000000000000222;
    hex8_xyze(1, 5) = 0.11111111111111099392;
    hex8_xyze(2, 5) = -0.061111111111111102168;
    nids.push_back(1245738);
    hex8_xyze(0, 6) = 0.4000000000000000222;
    hex8_xyze(1, 6) = 0.12222222222222199617;
    hex8_xyze(2, 6) = -0.061111111111111102168;
    nids.push_back(1245741);
    hex8_xyze(0, 7) = 0.4000000000000000222;
    hex8_xyze(1, 7) = 0.12222222222222199617;
    hex8_xyze(2, 7) = -0.050000000000000002776;
    nids.push_back(1245419);

    intersection.add_element(227469, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.38888888888888900608;
    hex8_xyze(1, 0) = 0.11111111111111099392;
    hex8_xyze(2, 0) = -0.061111111111111102168;
    nids.push_back(1245729);
    hex8_xyze(0, 1) = 0.38888888888888900608;
    hex8_xyze(1, 1) = 0.11111111111111099392;
    hex8_xyze(2, 1) = -0.07222222222222220156;
    nids.push_back(1245730);
    hex8_xyze(0, 2) = 0.38888888888888900608;
    hex8_xyze(1, 2) = 0.12222222222222199617;
    hex8_xyze(2, 2) = -0.07222222222222220156;
    nids.push_back(1245733);
    hex8_xyze(0, 3) = 0.38888888888888900608;
    hex8_xyze(1, 3) = 0.12222222222222199617;
    hex8_xyze(2, 3) = -0.061111111111111102168;
    nids.push_back(1245732);
    hex8_xyze(0, 4) = 0.4000000000000000222;
    hex8_xyze(1, 4) = 0.11111111111111099392;
    hex8_xyze(2, 4) = -0.061111111111111102168;
    nids.push_back(1245738);
    hex8_xyze(0, 5) = 0.4000000000000000222;
    hex8_xyze(1, 5) = 0.11111111111111099392;
    hex8_xyze(2, 5) = -0.07222222222222220156;
    nids.push_back(1245739);
    hex8_xyze(0, 6) = 0.4000000000000000222;
    hex8_xyze(1, 6) = 0.12222222222222199617;
    hex8_xyze(2, 6) = -0.07222222222222220156;
    nids.push_back(1245742);
    hex8_xyze(0, 7) = 0.4000000000000000222;
    hex8_xyze(1, 7) = 0.12222222222222199617;
    hex8_xyze(2, 7) = -0.061111111111111102168;
    nids.push_back(1245741);

    intersection.add_element(227470, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.38888888888888900608;
    hex8_xyze(1, 0) = 0.12222222222222199617;
    hex8_xyze(2, 0) = -0.050000000000000002776;
    nids.push_back(1245410);
    hex8_xyze(0, 1) = 0.38888888888888900608;
    hex8_xyze(1, 1) = 0.12222222222222199617;
    hex8_xyze(2, 1) = -0.061111111111111102168;
    nids.push_back(1245732);
    hex8_xyze(0, 2) = 0.38888888888888900608;
    hex8_xyze(1, 2) = 0.13333333333333299842;
    hex8_xyze(2, 2) = -0.061111111111111102168;
    nids.push_back(1245735);
    hex8_xyze(0, 3) = 0.38888888888888900608;
    hex8_xyze(1, 3) = 0.13333333333333299842;
    hex8_xyze(2, 3) = -0.050000000000000002776;
    nids.push_back(1245413);
    hex8_xyze(0, 4) = 0.4000000000000000222;
    hex8_xyze(1, 4) = 0.12222222222222199617;
    hex8_xyze(2, 4) = -0.050000000000000002776;
    nids.push_back(1245419);
    hex8_xyze(0, 5) = 0.4000000000000000222;
    hex8_xyze(1, 5) = 0.12222222222222199617;
    hex8_xyze(2, 5) = -0.061111111111111102168;
    nids.push_back(1245741);
    hex8_xyze(0, 6) = 0.4000000000000000222;
    hex8_xyze(1, 6) = 0.13333333333333299842;
    hex8_xyze(2, 6) = -0.061111111111111102168;
    nids.push_back(1245744);
    hex8_xyze(0, 7) = 0.4000000000000000222;
    hex8_xyze(1, 7) = 0.13333333333333299842;
    hex8_xyze(2, 7) = -0.050000000000000002776;
    nids.push_back(1245422);

    intersection.add_element(227472, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.4000000000000000222;
    hex8_xyze(1, 0) = 0.11111111111111099392;
    hex8_xyze(2, 0) = -0.050000000000000002776;
    nids.push_back(1245416);
    hex8_xyze(0, 1) = 0.4000000000000000222;
    hex8_xyze(1, 1) = 0.11111111111111099392;
    hex8_xyze(2, 1) = -0.061111111111111102168;
    nids.push_back(1245738);
    hex8_xyze(0, 2) = 0.4000000000000000222;
    hex8_xyze(1, 2) = 0.12222222222222199617;
    hex8_xyze(2, 2) = -0.061111111111111102168;
    nids.push_back(1245741);
    hex8_xyze(0, 3) = 0.4000000000000000222;
    hex8_xyze(1, 3) = 0.12222222222222199617;
    hex8_xyze(2, 3) = -0.050000000000000002776;
    nids.push_back(1245419);
    hex8_xyze(0, 4) = 0.41111111111111098282;
    hex8_xyze(1, 4) = 0.11111111111111099392;
    hex8_xyze(2, 4) = -0.050000000000000002776;
    nids.push_back(1255867);
    hex8_xyze(0, 5) = 0.41111111111111098282;
    hex8_xyze(1, 5) = 0.11111111111111099392;
    hex8_xyze(2, 5) = -0.061111111111111102168;
    nids.push_back(1255866);
    hex8_xyze(0, 6) = 0.41111111111111098282;
    hex8_xyze(1, 6) = 0.12222222222222199617;
    hex8_xyze(2, 6) = -0.061111111111111102168;
    nids.push_back(1255869);
    hex8_xyze(0, 7) = 0.41111111111111098282;
    hex8_xyze(1, 7) = 0.12222222222222199617;
    hex8_xyze(2, 7) = -0.050000000000000002776;
    nids.push_back(1255870);

    intersection.add_element(237306, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
