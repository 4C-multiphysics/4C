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

void test_generated_238425()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36151953585418855619;
    tri3_xyze(1, 0) = 0.097835809677823321051;
    tri3_xyze(2, 0) = -0.14021469133447883593;
    nids.push_back(4540);
    tri3_xyze(0, 1) = 0.36732117713104794898;
    tri3_xyze(1, 1) = 0.071201318112217720779;
    tri3_xyze(2, 1) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 2) = 0.36231437071790884019;
    tri3_xyze(1, 2) = 0.09235380867398007565;
    tri3_xyze(2, 2) = -0.14613613177488360417;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35867965647971034038;
    tri3_xyze(1, 0) = 0.086547825841667952451;
    tri3_xyze(2, 0) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 1) = 0.35864631567444449356;
    tri3_xyze(1, 1) = 0.070338644266236510783;
    tri3_xyze(2, 1) = -0.11714920409628661047;
    nids.push_back(3808);
    tri3_xyze(0, 2) = 0.36089144603821898816;
    tri3_xyze(1, 2) = 0.065264653431961777708;
    tri3_xyze(2, 2) = -0.1230037775776276765;
    nids.push_back(-426);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36766575565349585153;
    tri3_xyze(1, 0) = 0.049689854500496240253;
    tri3_xyze(2, 0) = -0.1406225233660632068;
    nids.push_back(4534);
    tri3_xyze(0, 1) = 0.36700989847585469006;
    tri3_xyze(1, 1) = 0.065882444294840442067;
    tri3_xyze(2, 1) = -0.14047368202652865676;
    nids.push_back(4536);
    tri3_xyze(0, 2) = 0.36490254019335516267;
    tri3_xyze(1, 2) = 0.071104972491011914082;
    tri3_xyze(2, 2) = -0.13458097589134637717;
    nids.push_back(-468);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36732117713104794898;
    tri3_xyze(1, 0) = 0.071201318112217720779;
    tri3_xyze(2, 0) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 1) = 0.37117286399149956866;
    tri3_xyze(1, 1) = 0.044231869327032463657;
    tri3_xyze(2, 1) = -0.1640896261976290682;
    nids.push_back(5262);
    tri3_xyze(0, 2) = 0.36770962389572892093;
    tri3_xyze(1, 2) = 0.065675144425834386386;
    tri3_xyze(2, 2) = -0.15808150887200428381;
    nids.push_back(-551);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35716769828443428736;
    tri3_xyze(1, 0) = 0.10270244013223353563;
    tri3_xyze(2, 0) = -0.11675867213855899152;
    nids.push_back(3812);
    tri3_xyze(0, 1) = 0.36337584731373412383;
    tri3_xyze(1, 1) = 0.07636468016047247287;
    tri3_xyze(2, 1) = -0.12869587765356355069;
    nids.push_back(4173);
    tri3_xyze(0, 2) = 0.35907138816336187093;
    tri3_xyze(1, 2) = 0.097552914801020684799;
    tri3_xyze(2, 2) = -0.122638462570951845;
    nids.push_back(-428);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36700989847585469006;
    tri3_xyze(1, 0) = 0.065882444294840442067;
    tri3_xyze(2, 0) = -0.14047368202652865676;
    nids.push_back(4536);
    tri3_xyze(0, 1) = 0.36155865933033615178;
    tri3_xyze(1, 1) = 0.092482911008238480322;
    tri3_xyze(2, 1) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 2) = 0.36490254019335516267;
    tri3_xyze(1, 2) = 0.071104972491011914082;
    tri3_xyze(2, 2) = -0.13458097589134637717;
    nids.push_back(-468);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35674115084445812141;
    tri3_xyze(1, 0) = 0.11336270719330492074;
    tri3_xyze(2, 0) = -0.14009992908238597109;
    nids.push_back(4542);
    tri3_xyze(0, 1) = 0.36151953585418855619;
    tri3_xyze(1, 1) = 0.097835809677823321051;
    tri3_xyze(2, 1) = -0.14021469133447883593;
    nids.push_back(4540);
    tri3_xyze(0, 2) = 0.36231437071790884019;
    tri3_xyze(1, 2) = 0.09235380867398007565;
    tri3_xyze(2, 2) = -0.14613613177488360417;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35867965647971034038;
    tri3_xyze(1, 0) = 0.086547825841667952451;
    tri3_xyze(2, 0) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 1) = 0.35168675901983048604;
    tri3_xyze(1, 1) = 0.11249096262694177617;
    tri3_xyze(2, 1) = -0.10498612605683477206;
    nids.push_back(3449);
    tri3_xyze(0, 2) = 0.35545941613502329837;
    tri3_xyze(1, 2) = 0.091422182473595847707;
    tri3_xyze(2, 2) = -0.11107495297579038362;
    nids.push_back(-386);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35674115084445812141;
    tri3_xyze(1, 0) = 0.11336270719330492074;
    tri3_xyze(2, 0) = -0.14009992908238597109;
    nids.push_back(4542);
    tri3_xyze(0, 1) = 0.3636756190419407897;
    tri3_xyze(1, 1) = 0.087015399712574326152;
    tri3_xyze(2, 1) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 2) = 0.357430460289674512;
    tri3_xyze(1, 2) = 0.10781719777968096219;
    tri3_xyze(2, 2) = -0.14603387174099907719;
    nids.push_back(-512);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36155865933033615178;
    tri3_xyze(1, 0) = 0.092482911008238480322;
    tri3_xyze(2, 0) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 1) = 0.36700989847585469006;
    tri3_xyze(1, 1) = 0.065882444294840442067;
    tri3_xyze(2, 1) = -0.14047368202652865676;
    nids.push_back(4536);
    tri3_xyze(0, 2) = 0.36296497247517628404;
    tri3_xyze(1, 2) = 0.087183247475846908925;
    tri3_xyze(2, 2) = -0.1344297289775739368;
    nids.push_back(-469);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.3636756190419407897;
    tri3_xyze(1, 0) = 0.087015399712574326152;
    tri3_xyze(2, 0) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 1) = 0.3586717955127640689;
    tri3_xyze(1, 1) = 0.10246340565472714101;
    tri3_xyze(2, 1) = -0.15197911111175060883;
    nids.push_back(4975);
    tri3_xyze(0, 2) = 0.357430460289674512;
    tri3_xyze(1, 2) = 0.10781719777968096219;
    tri3_xyze(2, 2) = -0.14603387174099907719;
    nids.push_back(-512);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36494793121117607981;
    tri3_xyze(1, 0) = 0.081969348260157490205;
    tri3_xyze(2, 0) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 1) = 0.36700989847585469006;
    tri3_xyze(1, 1) = 0.065882444294840442067;
    tri3_xyze(2, 1) = -0.14047368202652865676;
    nids.push_back(4536);
    tri3_xyze(0, 2) = 0.36801616396620173699;
    tri3_xyze(1, 2) = 0.06049050195940270519;
    tri3_xyze(2, 2) = -0.14636895130397126197;
    nids.push_back(-509);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.3586717955127640689;
    tri3_xyze(1, 0) = 0.10246340565472714101;
    tri3_xyze(2, 0) = -0.15197911111175060883;
    nids.push_back(4975);
    tri3_xyze(0, 1) = 0.3636756190419407897;
    tri3_xyze(1, 1) = 0.087015399712574326152;
    tri3_xyze(2, 1) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 2) = 0.36395017674232726934;
    tri3_xyze(1, 2) = 0.081433580217876083696;
    tri3_xyze(2, 2) = -0.15799636055992646866;
    nids.push_back(-552);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.3636756190419407897;
    tri3_xyze(1, 0) = 0.087015399712574326152;
    tri3_xyze(2, 0) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 1) = 0.36866883541842737637;
    tri3_xyze(1, 1) = 0.060251990551513041894;
    tri3_xyze(2, 1) = -0.16400650260771842959;
    nids.push_back(5264);
    tri3_xyze(0, 2) = 0.36395017674232726934;
    tri3_xyze(1, 2) = 0.081433580217876083696;
    tri3_xyze(2, 2) = -0.15799636055992646866;
    nids.push_back(-552);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35834340088333810348;
    tri3_xyze(1, 0) = 0.10839828634015123698;
    tri3_xyze(2, 0) = -0.12837626552624967213;
    nids.push_back(4177);
    tri3_xyze(0, 1) = 0.36155865933033615178;
    tri3_xyze(1, 1) = 0.092482911008238480322;
    tri3_xyze(2, 1) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 2) = 0.36296497247517628404;
    tri3_xyze(1, 2) = 0.087183247475846908925;
    tri3_xyze(2, 2) = -0.1344297289775739368;
    nids.push_back(-469);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36494793121117607981;
    tri3_xyze(1, 0) = 0.081969348260157490205;
    tri3_xyze(2, 0) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 1) = 0.35834340088333810348;
    tri3_xyze(1, 1) = 0.10839828634015123698;
    tri3_xyze(2, 1) = -0.12837626552624967213;
    nids.push_back(4177);
    tri3_xyze(0, 2) = 0.36296497247517628404;
    tri3_xyze(1, 2) = 0.087183247475846908925;
    tri3_xyze(2, 2) = -0.1344297289775739368;
    nids.push_back(-469);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36337584731373412383;
    tri3_xyze(1, 0) = 0.07636468016047247287;
    tri3_xyze(2, 0) = -0.12869587765356355069;
    nids.push_back(4173);
    tri3_xyze(0, 1) = 0.35716769828443428736;
    tri3_xyze(1, 1) = 0.10270244013223353563;
    tri3_xyze(2, 1) = -0.11675867213855899152;
    nids.push_back(3812);
    tri3_xyze(0, 2) = 0.3607352129917399397;
    tri3_xyze(1, 2) = 0.081443925491652330306;
    tri3_xyze(2, 2) = -0.12281767770409744711;
    nids.push_back(-427);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35867965647971034038;
    tri3_xyze(1, 0) = 0.086547825841667952451;
    tri3_xyze(2, 0) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 1) = 0.35716769828443428736;
    tri3_xyze(1, 1) = 0.10270244013223353563;
    tri3_xyze(2, 1) = -0.11675867213855899152;
    nids.push_back(3812);
    tri3_xyze(0, 2) = 0.35413478596238573415;
    tri3_xyze(1, 2) = 0.10756240723399365655;
    tri3_xyze(2, 2) = -0.11086414529547067298;
    nids.push_back(-387);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36732117713104794898;
    tri3_xyze(1, 0) = 0.071201318112217720779;
    tri3_xyze(2, 0) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 1) = 0.3636756190419407897;
    tri3_xyze(1, 1) = 0.087015399712574326152;
    tri3_xyze(2, 1) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 2) = 0.36231437071790884019;
    tri3_xyze(1, 2) = 0.09235380867398007565;
    tri3_xyze(2, 2) = -0.14613613177488360417;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35864631567444449356;
    tri3_xyze(1, 0) = 0.070338644266236510783;
    tri3_xyze(2, 0) = -0.11714920409628661047;
    nids.push_back(3808);
    tri3_xyze(0, 1) = 0.35867965647971034038;
    tri3_xyze(1, 1) = 0.086547825841667952451;
    tri3_xyze(2, 1) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 2) = 0.35545941613502329837;
    tri3_xyze(1, 2) = 0.091422182473595847707;
    tri3_xyze(2, 2) = -0.11107495297579038362;
    nids.push_back(-386);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35867965647971034038;
    tri3_xyze(1, 0) = 0.086547825841667952451;
    tri3_xyze(2, 0) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 1) = 0.36371764988908100724;
    tri3_xyze(1, 1) = 0.060160755832235339458;
    tri3_xyze(2, 1) = -0.12886620717799654456;
    nids.push_back(4171);
    tri3_xyze(0, 2) = 0.3607352129917399397;
    tri3_xyze(1, 2) = 0.081443925491652330306;
    tri3_xyze(2, 2) = -0.12281767770409744711;
    nids.push_back(-427);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35834340088333810348;
    tri3_xyze(1, 0) = 0.10839828634015123698;
    tri3_xyze(2, 0) = -0.12837626552624967213;
    nids.push_back(4177);
    tri3_xyze(0, 1) = 0.36494793121117607981;
    tri3_xyze(1, 1) = 0.081969348260157490205;
    tri3_xyze(2, 1) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 2) = 0.35964679179035397016;
    tri3_xyze(1, 2) = 0.10305042192291621883;
    tri3_xyze(2, 2) = -0.134289798927130416;
    nids.push_back(-470);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36732117713104794898;
    tri3_xyze(1, 0) = 0.071201318112217720779;
    tri3_xyze(2, 0) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 1) = 0.36960062161953771698;
    tri3_xyze(1, 1) = 0.055145388941402063987;
    tri3_xyze(2, 1) = -0.15226974595110145949;
    nids.push_back(4899);
    tri3_xyze(0, 2) = 0.37009793015004444072;
    tri3_xyze(1, 2) = 0.049661386421037470584;
    tri3_xyze(2, 2) = -0.15817721435045564715;
    nids.push_back(-550);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36337584731373412383;
    tri3_xyze(1, 0) = 0.07636468016047247287;
    tri3_xyze(2, 0) = -0.12869587765356355069;
    nids.push_back(4173);
    tri3_xyze(0, 1) = 0.36155865933033615178;
    tri3_xyze(1, 1) = 0.092482911008238480322;
    tri3_xyze(2, 1) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 2) = 0.35907138816336187093;
    tri3_xyze(1, 2) = 0.097552914801020684799;
    tri3_xyze(2, 2) = -0.122638462570951845;
    nids.push_back(-428);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36151953585418855619;
    tri3_xyze(1, 0) = 0.097835809677823321051;
    tri3_xyze(2, 0) = -0.14021469133447883593;
    nids.push_back(4540);
    tri3_xyze(0, 1) = 0.36494793121117607981;
    tri3_xyze(1, 1) = 0.081969348260157490205;
    tri3_xyze(2, 1) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 2) = 0.36584731645398754774;
    tri3_xyze(1, 2) = 0.076537966247900143801;
    tri3_xyze(2, 2) = -0.1462460202988880853;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36155865933033615178;
    tri3_xyze(1, 0) = 0.092482911008238480322;
    tri3_xyze(2, 0) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 1) = 0.35418334772494297624;
    tri3_xyze(1, 1) = 0.11866162790313820874;
    tri3_xyze(2, 1) = -0.1165674799724547156;
    nids.push_back(3814);
    tri3_xyze(0, 2) = 0.35907138816336187093;
    tri3_xyze(1, 2) = 0.097552914801020684799;
    tri3_xyze(2, 2) = -0.122638462570951845;
    nids.push_back(-428);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36478445699617684239;
    tri3_xyze(1, 0) = 0.076003524952689804906;
    tri3_xyze(2, 0) = -0.16393241790925203172;
    nids.push_back(5336);
    tri3_xyze(0, 1) = 0.3586717955127640689;
    tri3_xyze(1, 1) = 0.10246340565472714101;
    tri3_xyze(2, 1) = -0.15197911111175060883;
    nids.push_back(4975);
    tri3_xyze(0, 2) = 0.36395017674232726934;
    tri3_xyze(1, 2) = 0.081433580217876083696;
    tri3_xyze(2, 2) = -0.15799636055992646866;
    nids.push_back(-552);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36155865933033615178;
    tri3_xyze(1, 0) = 0.092482911008238480322;
    tri3_xyze(2, 0) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 1) = 0.35834340088333810348;
    tri3_xyze(1, 1) = 0.10839828634015123698;
    tri3_xyze(2, 1) = -0.12837626552624967213;
    nids.push_back(4177);
    tri3_xyze(0, 2) = 0.35597532155278432953;
    tri3_xyze(1, 2) = 0.11346482973867598465;
    tri3_xyze(2, 2) = -0.12246371395126497139;
    nids.push_back(-429);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36155865933033615178;
    tri3_xyze(1, 0) = 0.092482911008238480322;
    tri3_xyze(2, 0) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 1) = 0.36337584731373412383;
    tri3_xyze(1, 1) = 0.07636468016047247287;
    tri3_xyze(2, 1) = -0.12869587765356355069;
    nids.push_back(4173);
    tri3_xyze(0, 2) = 0.36490254019335516267;
    tri3_xyze(1, 2) = 0.071104972491011914082;
    tri3_xyze(2, 2) = -0.13458097589134637717;
    nids.push_back(-468);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.3636756190419407897;
    tri3_xyze(1, 0) = 0.087015399712574326152;
    tri3_xyze(2, 0) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 1) = 0.35674115084445812141;
    tri3_xyze(1, 1) = 0.11336270719330492074;
    tri3_xyze(2, 1) = -0.14009992908238597109;
    nids.push_back(4542);
    tri3_xyze(0, 2) = 0.36231437071790884019;
    tri3_xyze(1, 2) = 0.09235380867398007565;
    tri3_xyze(2, 2) = -0.14613613177488360417;
    nids.push_back(-511);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.37117286399149956866;
    tri3_xyze(1, 0) = 0.044231869327032463657;
    tri3_xyze(2, 0) = -0.1640896261976290682;
    nids.push_back(5262);
    tri3_xyze(0, 1) = 0.36732117713104794898;
    tri3_xyze(1, 1) = 0.071201318112217720779;
    tri3_xyze(2, 1) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 2) = 0.37009793015004444072;
    tri3_xyze(1, 2) = 0.049661386421037470584;
    tri3_xyze(2, 2) = -0.15817721435045564715;
    nids.push_back(-550);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36960062161953771698;
    tri3_xyze(1, 0) = 0.055145388941402063987;
    tri3_xyze(2, 0) = -0.15226974595110145949;
    nids.push_back(4899);
    tri3_xyze(0, 1) = 0.36732117713104794898;
    tri3_xyze(1, 1) = 0.071201318112217720779;
    tri3_xyze(2, 1) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 2) = 0.36584731645398754774;
    tri3_xyze(1, 2) = 0.076537966247900143801;
    tri3_xyze(2, 2) = -0.1462460202988880853;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.3636756190419407897;
    tri3_xyze(1, 0) = 0.087015399712574326152;
    tri3_xyze(2, 0) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 1) = 0.36732117713104794898;
    tri3_xyze(1, 1) = 0.071201318112217720779;
    tri3_xyze(2, 1) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 2) = 0.36770962389572892093;
    tri3_xyze(1, 2) = 0.065675144425834386386;
    tri3_xyze(2, 2) = -0.15808150887200428381;
    nids.push_back(-551);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36866883541842737637;
    tri3_xyze(1, 0) = 0.060251990551513041894;
    tri3_xyze(2, 0) = -0.16400650260771842959;
    nids.push_back(5264);
    tri3_xyze(0, 1) = 0.3636756190419407897;
    tri3_xyze(1, 1) = 0.087015399712574326152;
    tri3_xyze(2, 1) = -0.1520674106109848045;
    nids.push_back(4903);
    tri3_xyze(0, 2) = 0.36770962389572892093;
    tri3_xyze(1, 2) = 0.065675144425834386386;
    tri3_xyze(2, 2) = -0.15808150887200428381;
    nids.push_back(-551);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36371764988908100724;
    tri3_xyze(1, 0) = 0.060160755832235339458;
    tri3_xyze(2, 0) = -0.12886620717799654456;
    nids.push_back(4171);
    tri3_xyze(0, 1) = 0.35867965647971034038;
    tri3_xyze(1, 1) = 0.086547825841667952451;
    tri3_xyze(2, 1) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 2) = 0.36089144603821898816;
    tri3_xyze(1, 2) = 0.065264653431961777708;
    tri3_xyze(2, 2) = -0.1230037775776276765;
    nids.push_back(-426);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36494793121117607981;
    tri3_xyze(1, 0) = 0.081969348260157490205;
    tri3_xyze(2, 0) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 1) = 0.36151953585418855619;
    tri3_xyze(1, 1) = 0.097835809677823321051;
    tri3_xyze(2, 1) = -0.14021469133447883593;
    nids.push_back(4540);
    tri3_xyze(0, 2) = 0.35964679179035397016;
    tri3_xyze(1, 2) = 0.10305042192291621883;
    tri3_xyze(2, 2) = -0.134289798927130416;
    nids.push_back(-470);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36371764988908100724;
    tri3_xyze(1, 0) = 0.060160755832235339458;
    tri3_xyze(2, 0) = -0.12886620717799654456;
    nids.push_back(4171);
    tri3_xyze(0, 1) = 0.36337584731373412383;
    tri3_xyze(1, 1) = 0.07636468016047247287;
    tri3_xyze(2, 1) = -0.12869587765356355069;
    nids.push_back(4173);
    tri3_xyze(0, 2) = 0.3607352129917399397;
    tri3_xyze(1, 2) = 0.081443925491652330306;
    tri3_xyze(2, 2) = -0.12281767770409744711;
    nids.push_back(-427);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36732117713104794898;
    tri3_xyze(1, 0) = 0.071201318112217720779;
    tri3_xyze(2, 0) = -0.15216249607168483293;
    nids.push_back(4901);
    tri3_xyze(0, 1) = 0.36151953585418855619;
    tri3_xyze(1, 1) = 0.097835809677823321051;
    tri3_xyze(2, 1) = -0.14021469133447883593;
    nids.push_back(4540);
    tri3_xyze(0, 2) = 0.36584731645398754774;
    tri3_xyze(1, 2) = 0.076537966247900143801;
    tri3_xyze(2, 2) = -0.1462460202988880853;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35418334772494297624;
    tri3_xyze(1, 0) = 0.11866162790313820874;
    tri3_xyze(2, 0) = -0.1165674799724547156;
    nids.push_back(3814);
    tri3_xyze(0, 1) = 0.36155865933033615178;
    tri3_xyze(1, 1) = 0.092482911008238480322;
    tri3_xyze(2, 1) = -0.12853182051923012219;
    nids.push_back(4175);
    tri3_xyze(0, 2) = 0.35597532155278432953;
    tri3_xyze(1, 2) = 0.11346482973867598465;
    tri3_xyze(2, 2) = -0.12246371395126497139;
    nids.push_back(-429);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35168675901983048604;
    tri3_xyze(1, 0) = 0.11249096262694177617;
    tri3_xyze(2, 0) = -0.10498612605683477206;
    nids.push_back(3449);
    tri3_xyze(0, 1) = 0.35867965647971034038;
    tri3_xyze(1, 1) = 0.086547825841667952451;
    tri3_xyze(2, 1) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 2) = 0.35413478596238573415;
    tri3_xyze(1, 2) = 0.10756240723399365655;
    tri3_xyze(2, 2) = -0.11086414529547067298;
    nids.push_back(-387);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36960062161953771698;
    tri3_xyze(1, 0) = 0.055145388941402063987;
    tri3_xyze(2, 0) = -0.15226974595110145949;
    nids.push_back(4899);
    tri3_xyze(0, 1) = 0.36494793121117607981;
    tri3_xyze(1, 1) = 0.081969348260157490205;
    tri3_xyze(2, 1) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 2) = 0.36801616396620173699;
    tri3_xyze(1, 2) = 0.06049050195940270519;
    tri3_xyze(2, 2) = -0.14636895130397126197;
    nids.push_back(-509);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36700989847585469006;
    tri3_xyze(1, 0) = 0.065882444294840442067;
    tri3_xyze(2, 0) = -0.14047368202652865676;
    nids.push_back(4536);
    tri3_xyze(0, 1) = 0.36494793121117607981;
    tri3_xyze(1, 1) = 0.081969348260157490205;
    tri3_xyze(2, 1) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 2) = 0.36296497247517628404;
    tri3_xyze(1, 2) = 0.087183247475846908925;
    tri3_xyze(2, 2) = -0.1344297289775739368;
    nids.push_back(-469);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.36494793121117607981;
    tri3_xyze(1, 0) = 0.081969348260157490205;
    tri3_xyze(2, 0) = -0.14033714783828721284;
    nids.push_back(4538);
    tri3_xyze(0, 1) = 0.36960062161953771698;
    tri3_xyze(1, 1) = 0.055145388941402063987;
    tri3_xyze(2, 1) = -0.15226974595110145949;
    nids.push_back(4899);
    tri3_xyze(0, 2) = 0.36584731645398754774;
    tri3_xyze(1, 2) = 0.076537966247900143801;
    tri3_xyze(2, 2) = -0.1462460202988880853;
    nids.push_back(-510);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.35716769828443428736;
    tri3_xyze(1, 0) = 0.10270244013223353563;
    tri3_xyze(2, 0) = -0.11675867213855899152;
    nids.push_back(3812);
    tri3_xyze(0, 1) = 0.35867965647971034038;
    tri3_xyze(1, 1) = 0.086547825841667952451;
    tri3_xyze(2, 1) = -0.11694995384627070167;
    nids.push_back(3810);
    tri3_xyze(0, 2) = 0.3607352129917399397;
    tri3_xyze(1, 2) = 0.081443925491652330306;
    tri3_xyze(2, 2) = -0.12281767770409744711;
    nids.push_back(-427);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.36525608783557250714;
    hex8_xyze(1, 0) = 0.08039894463892298393;
    hex8_xyze(2, 0) = -0.1560526315789479701;
    nids.push_back(259046);
    hex8_xyze(0, 1) = 0.36211537978535868199;
    hex8_xyze(1, 1) = 0.093533158414037573247;
    hex8_xyze(2, 1) = -0.1560526315789479701;
    nids.push_back(259047);
    hex8_xyze(0, 2) = 0.35501507822093991695;
    hex8_xyze(1, 2) = 0.091699174915723102863;
    hex8_xyze(2, 2) = -0.1560526315789479701;
    nids.push_back(259077);
    hex8_xyze(0, 3) = 0.35809420376036515954;
    hex8_xyze(1, 3) = 0.078822494744042140291;
    hex8_xyze(2, 3) = -0.1560526315789479701;
    nids.push_back(259076);
    hex8_xyze(0, 4) = 0.36525608783557250714;
    hex8_xyze(1, 4) = 0.080398944638922942296;
    hex8_xyze(2, 4) = -0.13631578947368463983;
    nids.push_back(247258);
    hex8_xyze(0, 5) = 0.36211537978535868199;
    hex8_xyze(1, 5) = 0.093533158414037531614;
    hex8_xyze(2, 5) = -0.13631578947368463983;
    nids.push_back(247259);
    hex8_xyze(0, 6) = 0.35501507822093991695;
    hex8_xyze(1, 6) = 0.09169917491572306123;
    hex8_xyze(2, 6) = -0.13631578947368463983;
    nids.push_back(247289);
    hex8_xyze(0, 7) = 0.35809420376036515954;
    hex8_xyze(1, 7) = 0.078822494744042098658;
    hex8_xyze(2, 7) = -0.13631578947368463983;
    nids.push_back(247288);

    intersection.add_element(238425, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.36792057163056907099;
    hex8_xyze(1, 0) = 0.067159905978457987152;
    hex8_xyze(2, 0) = -0.1560526315789479701;
    nids.push_back(259045);
    hex8_xyze(0, 1) = 0.36525608783557250714;
    hex8_xyze(1, 1) = 0.08039894463892298393;
    hex8_xyze(2, 1) = -0.1560526315789479701;
    nids.push_back(259046);
    hex8_xyze(0, 2) = 0.35809420376036515954;
    hex8_xyze(1, 2) = 0.078822494744042140291;
    hex8_xyze(2, 2) = -0.1560526315789479701;
    nids.push_back(259076);
    hex8_xyze(0, 3) = 0.36070644277506769271;
    hex8_xyze(1, 3) = 0.065843045076919584635;
    hex8_xyze(2, 3) = -0.1560526315789479701;
    nids.push_back(259075);
    hex8_xyze(0, 4) = 0.36792057163056907099;
    hex8_xyze(1, 4) = 0.067159905978457945519;
    hex8_xyze(2, 4) = -0.13631578947368463983;
    nids.push_back(247257);
    hex8_xyze(0, 5) = 0.36525608783557250714;
    hex8_xyze(1, 5) = 0.080398944638922942296;
    hex8_xyze(2, 5) = -0.13631578947368463983;
    nids.push_back(247258);
    hex8_xyze(0, 6) = 0.35809420376036515954;
    hex8_xyze(1, 6) = 0.078822494744042098658;
    hex8_xyze(2, 6) = -0.13631578947368463983;
    nids.push_back(247288);
    hex8_xyze(0, 7) = 0.36070644277506769271;
    hex8_xyze(1, 7) = 0.065843045076919543002;
    hex8_xyze(2, 7) = -0.13631578947368463983;
    nids.push_back(247287);

    intersection.add_element(238424, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.35809420376036515954;
    hex8_xyze(1, 0) = 0.078822494744042140291;
    hex8_xyze(2, 0) = -0.1560526315789479701;
    nids.push_back(259076);
    hex8_xyze(0, 1) = 0.35501507822093991695;
    hex8_xyze(1, 1) = 0.091699174915723102863;
    hex8_xyze(2, 1) = -0.1560526315789479701;
    nids.push_back(259077);
    hex8_xyze(0, 2) = 0.34791477665652109641;
    hex8_xyze(1, 2) = 0.089865191417408646357;
    hex8_xyze(2, 2) = -0.1560526315789479701;
    nids.push_back(259107);
    hex8_xyze(0, 3) = 0.35093231968515792296;
    hex8_xyze(1, 3) = 0.07724604484916131053;
    hex8_xyze(2, 3) = -0.1560526315789479701;
    nids.push_back(259106);
    hex8_xyze(0, 4) = 0.35809420376036515954;
    hex8_xyze(1, 4) = 0.078822494744042098658;
    hex8_xyze(2, 4) = -0.13631578947368463983;
    nids.push_back(247288);
    hex8_xyze(0, 5) = 0.35501507822093991695;
    hex8_xyze(1, 5) = 0.09169917491572306123;
    hex8_xyze(2, 5) = -0.13631578947368463983;
    nids.push_back(247289);
    hex8_xyze(0, 6) = 0.34791477665652109641;
    hex8_xyze(1, 6) = 0.089865191417408604724;
    hex8_xyze(2, 6) = -0.13631578947368463983;
    nids.push_back(247319);
    hex8_xyze(0, 7) = 0.35093231968515792296;
    hex8_xyze(1, 7) = 0.077246044849161268897;
    hex8_xyze(2, 7) = -0.13631578947368463983;
    nids.push_back(247318);

    intersection.add_element(238454, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.36525608783557250714;
    hex8_xyze(1, 0) = 0.080398944638922942296;
    hex8_xyze(2, 0) = -0.13631578947368463983;
    nids.push_back(247258);
    hex8_xyze(0, 1) = 0.36211537978535868199;
    hex8_xyze(1, 1) = 0.093533158414037531614;
    hex8_xyze(2, 1) = -0.13631578947368463983;
    nids.push_back(247259);
    hex8_xyze(0, 2) = 0.35501507822093991695;
    hex8_xyze(1, 2) = 0.09169917491572306123;
    hex8_xyze(2, 2) = -0.13631578947368463983;
    nids.push_back(247289);
    hex8_xyze(0, 3) = 0.35809420376036515954;
    hex8_xyze(1, 3) = 0.078822494744042098658;
    hex8_xyze(2, 3) = -0.13631578947368463983;
    nids.push_back(247288);
    hex8_xyze(0, 4) = 0.36525608783557250714;
    hex8_xyze(1, 4) = 0.080398944638922942296;
    hex8_xyze(2, 4) = -0.11657894736842105976;
    nids.push_back(235470);
    hex8_xyze(0, 5) = 0.36211537978535868199;
    hex8_xyze(1, 5) = 0.093533158414037531614;
    hex8_xyze(2, 5) = -0.11657894736842105976;
    nids.push_back(235471);
    hex8_xyze(0, 6) = 0.35501507822093991695;
    hex8_xyze(1, 6) = 0.09169917491572306123;
    hex8_xyze(2, 6) = -0.11657894736842105976;
    nids.push_back(235501);
    hex8_xyze(0, 7) = 0.35809420376036515954;
    hex8_xyze(1, 7) = 0.078822494744042098658;
    hex8_xyze(2, 7) = -0.11657894736842105976;
    nids.push_back(235500);

    intersection.add_element(226725, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
