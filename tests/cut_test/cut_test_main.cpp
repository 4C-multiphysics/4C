
#include "../../src/drt_cut/cut_mesh.H"
#include "../../src/drt_cut/cut_element.H"
#include "../../src/drt_fem_general/drt_utils_gausspoints.H"
#include "cut_test_utils.H"

//#include <boost/program_options.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <map>
#include <string>
#include <sstream>

#include <fenv.h>

#include <mpi.h>

void test_hex8_simple();
void test_tet4_simple();
void test_pyramid5_simple();
void test_wedge6_simple();
void test_hex8_fullside();
void test_hex8_diagonal();
void test_hex8_tet4();
void test_hex8_hex8();
void test_hex8_touch();
void test_hex8_touch2();
void test_hex8_schraeg();
void test_hex8_quad4_woelbung();
void test_hex8_tet4_touch();
void test_hex8_tet4_touch2();
void test_hex8_mesh();
void test_hex8_double();
void test_hex8_multiple();
void test_hex8_bad1();
void test_hex8_bad2();
void test_hex8_bad3();
void test_hex8_bad4();
void test_hex8_wedge6();
void test_hex8_quad4_touch();
void test_hex8_quad4_touch2();
void test_hex8_quad4_touch3();
void test_hex8_quad4_cut();
void test_hex8_quad4_gedreht();
void test_hex8_hex8_durchstoss();
void test_hex8_hex8_onside();
void test_hex8_hex8_internal();
void test_hex8_hex8_sideintersection();
void test_hex8_hex8_inside();
void test_hex8_quad4_schnitt();
void test_hex8_quad4_touch4();
void test_hex8_quad4_touch5();
void test_hex8_quad4_touch6();
void test_hex8_quad4_touch7();
void test_quad4_quad4_simple();
void test_hex8_quad4_mesh();
void test_position2d();
void test_facet_split();


void test_quad4_line2();
void test_hex8_quad4_qhull1();
void test_hex8_quad4_alex1();
void test_hex8_quad4_alex2();
void test_hex8_quad4_alex3();
void test_hex8_quad4_alex4();
void test_hex8_quad4_alex5();
void test_hex8_quad4_alex6();
void test_hex8_quad4_alex7();
void test_hex8_quad4_alex8();
void test_tet4_quad4_alex9();
void test_tet4_quad4_alex10();
void test_tet4_quad4_alex11();
void test_hex8_quad4_alex12();
void test_hex8_quad4_alex13();
void test_hex8_quad4_alex14();
void test_hex8_quad4_alex15();
void test_tet4_quad4_alex16();
void test_hex8_quad4_alex17();
void test_hex8_quad4_alex18();
void test_hex8_quad4_alex19();
void test_hex8_quad4_alex20();
void test_hex8_quad4_alex21();
void test_hex8_quad4_alex22();
void test_hex8_quad4_alex23();
void test_hex8_quad4_alex24();
void test_hex8_quad4_alex25();
void test_hex8_quad4_alex26();
void test_hex8_quad4_alex27();
void test_hex8_quad4_alex28();
void test_hex8_quad4_alex29();
void test_hex8_quad4_alex30();
void test_hex8_quad4_alex31();
void test_hex8_quad4_alex32();
void test_hex8_quad4_alex33();
void test_hex8_quad4_alex34();
void test_hex8_quad4_alex35();
void test_hex8_quad4_alex36();
void test_hex8_quad4_alex37();
void test_hex8_quad4_alex38();
void test_hex8_twintri();
void test_hex8_twinQuad();
void test_hex8_chairCut();
void test_hex8_VCut();
void test_alex39();
void test_alex40();
void test_alex41();
void test_alex42();
void test_alex43();
void test_alex44();
void test_alex45();
void test_alex46();
void test_alex47();
void test_alex48();
void test_alex49();
void test_alex50();
void test_alex51();
void test_alex52();
void test_alex53();
void test_alex54();
void test_alex55();
void test_alex56();
void test_alex57();
void test_alex58();
void test_alex59();
void test_alex60();
void test_alex61();
void test_alex62();
void test_hex8_quad4_axel1();
void test_hex8_quad4_axel2();
void test_hex8_quad4_axel3();
void test_hex8_quad4_axel4();
void test_hex8_quad4_axel5();
void test_hex8_quad4_axel6();
void test_hex8_quad4_axel7();
void test_axel8();
void test_axel9();
void test_axel10();
void test_hex8_quad4_shadan1();
void test_hex8_quad4_shadan2();
void test_hex8_quad4_shadan3();
void test_hex8_quad4_shadan4();
void test_hex8_quad4_shadan5();
void test_shadan6();
void test_hex8_tri3_ursula1();
void test_hex8_quad4_mesh_many();
void test_hex8_quad4_mesh_edgecut();
void test_hex8_quad4_mesh_edgecut2();
void test_hex8_quad4_mesh_inner();
void test_hex27_quad9_simple();
void test_hex20_quad9_simple();
void test_hex20_quad9_moved();
void test_tet10_quad9_simple();
void test_tet10_quad9_moved();
void test_tet4_quad4_double();
void test_tet4_tri3_double();
void test_benedikt1();

void test_ls_hex8_florian1();
void test_ls_hex8_florian2();
void test_ls_hex8_florian3();
void test_ls_hex8_florian4();
void test_ls_hex8_florian5();
void test_ls_hex8_florian6();
void test_ls_hex8_florian7();
void test_ls_hex8_florian8();
void test_ls_hex8_florian9();
void test_ls_hex8_florian10();
void test_ls_hex8_florian11();
void test_ls_hex8_florian12();
void test_ls_hex8_florian13();
void test_ls_hex8_ursula1();
void test_ls_hex8_ursula2();
void test_ls_hex8_ursula3();
void test_ls_hex8_ursula4();
void test_ls_hex8_ursula5();
void test_ls_hex8_ursula6();
void test_ls_hex8_simple();
void test_ls_hex8_simple2();
void test_ls_hex8_simple3();
void test_ls_hex8_simple4();
void test_ls_hex8_simple5();
void test_ls_hex8_simple6();
void test_ls_hex8_simple7();
void test_ls_hex8_touch();
void test_ls_hex8_between();
void test_ls_hex8_experiment();

void test_quad4_surface_mesh_cut();
void test_hex8_quad4_double_cut();

void test_unit_intersection_touch();

void test_facets_corner_points();

void test_colored_graph();
void test_colored_graph2();
void test_graph();
void test_graph2();

void test_geometry();

void test_cut_volumes();
void test_cut_volumes2();
void test_cut_volumes3();

void test_fluidfluid();
void test_fluidfluid2();

typedef void ( *testfunct )();


int runtests( char ** argv, const std::map<std::string, testfunct> & functable, std::string testname )
{
  if ( testname == "(all)" )
  {
    std::vector<std::string> failures;
    std::vector<std::string> msgs;

    for ( std::map<std::string, testfunct>::const_iterator i=functable.begin(); i!=functable.end(); ++i )
    {
      std::cout << "Testing " << i->first << " ...\n";
      try
      {
        ( *i->second )();
      }
      catch ( std::runtime_error & err )
      {
        std::cout << "FAILED: " << err.what() << "\n";
        failures.push_back( i->first );
        msgs.push_back( err.what() );
      }
    }

    if ( failures.size() > 0 )
    {
      std::cout << "\n" << failures.size() << " out of " << functable.size() << " tests failed.\n";
      for ( std::vector<std::string>::iterator i=failures.begin(); i!=failures.end(); ++i )
      {
        std::string & txt = *i;
        std::cout << "    " << txt;
        for ( unsigned j=0; j<40-txt.length(); ++j )
          std::cout << " ";
        std::cout << "(" << msgs[i-failures.begin()] << ")"
                  << "\n";
      }
    }
    else
    {
      std::cout << "\nall " << functable.size() << " tests succeeded.\n";
    }
    return failures.size();
  }
  else
  {
    std::map<std::string, testfunct>::const_iterator i = functable.find( testname );
    if ( i==functable.end() )
    {
      std::cerr << argv[0] << ": test '" << testname << "' not found\n";
      return 1;
    }
    else
    {
      ( *i->second )();
      return 0;
    }
  }
}

int main( int argc, char ** argv )
{
  MPI_Init( &argc, &argv );
  //MPI::Init( argc, argv );

  //feenableexcept( FE_INVALID | FE_DIVBYZERO );

  std::map<std::string, testfunct> functable;

  functable["hex8_simple"] = test_hex8_simple;
  functable["tet4_simple"] = test_tet4_simple;
  functable["pyramid5_simple"] = test_pyramid5_simple;
  functable["wedge6_simple"] = test_wedge6_simple;
  functable["hex8_diagonal"] = test_hex8_diagonal;
  functable["hex8_fullside"] = test_hex8_fullside;
  functable["hex8_hex8"] = test_hex8_hex8;
  functable["hex8_tet4"] = test_hex8_tet4;
  functable["hex8_touch"] = test_hex8_touch;
  functable["hex8_touch2"] = test_hex8_touch2;
  functable["hex8_schraeg"] = test_hex8_schraeg;
  functable["hex8_quad4_woelbung"] = test_hex8_quad4_woelbung;
  functable["hex8_tet4_touch"] = test_hex8_tet4_touch;
  functable["hex8_tet4_touch2"] = test_hex8_tet4_touch2;
  functable["hex8_mesh"] = test_hex8_mesh;
  functable["hex8_double"] = test_hex8_double;
  functable["hex8_multiple"] = test_hex8_multiple;
  functable["hex8_bad1"] = test_hex8_bad1;
  functable["hex8_bad2"] = test_hex8_bad2;
  functable["hex8_bad3"] = test_hex8_bad3;
  functable["hex8_bad4"] = test_hex8_bad4;
  functable["hex8_wedge6"] = test_hex8_wedge6;
  functable["hex8_quad4_touch"] = test_hex8_quad4_touch;
  functable["hex8_quad4_touch2"] = test_hex8_quad4_touch2;
  functable["hex8_quad4_touch3"] = test_hex8_quad4_touch3;
  functable["hex8_quad4_cut"] = test_hex8_quad4_cut;
  functable["hex8_quad4_gedreht"] = test_hex8_quad4_gedreht;
  functable["hex8_hex8_durchstoss"] = test_hex8_hex8_durchstoss;
  functable["hex8_hex8_onside"] = test_hex8_hex8_onside;
  functable["hex8_hex8_internal"] = test_hex8_hex8_internal;
  functable["hex8_hex8_sideintersection"] = test_hex8_hex8_sideintersection;
  functable["facet_split"] = test_facet_split;

  // Cells within cells without contact to any surface are not supported.
  //
  //functable["hex8_hex8_inside"] = test_hex8_hex8_inside;

  //functable["hex8_quad4_schnitt"] = test_hex8_quad4_schnitt;
  functable["hex8_quad4_touch4"] = test_hex8_quad4_touch4;
  functable["hex8_quad4_touch5"] = test_hex8_quad4_touch5;
  functable["hex8_quad4_touch6"] = test_hex8_quad4_touch6;
  //functable["hex8_quad4_touch7"] = test_hex8_quad4_touch7;
  functable["hex8_quad4_mesh"] = test_hex8_quad4_mesh;
  functable["position2d"] = test_position2d;

  functable["quad4_line2"] = test_quad4_line2;
  functable["hex8_quad4_qhull1"] = test_hex8_quad4_qhull1;
  functable["hex8_quad4_alex1"] = test_hex8_quad4_alex1;
  functable["hex8_quad4_alex2"] = test_hex8_quad4_alex2;
  functable["hex8_quad4_alex3"] = test_hex8_quad4_alex3;
  functable["hex8_quad4_alex4"] = test_hex8_quad4_alex4;
  functable["hex8_quad4_alex5"] = test_hex8_quad4_alex5;
  functable["hex8_quad4_alex6"] = test_hex8_quad4_alex6;
  functable["hex8_quad4_alex7"] = test_hex8_quad4_alex7;
  functable["hex8_quad4_alex8"] = test_hex8_quad4_alex8;
  functable["tet4_quad4_alex9"] = test_tet4_quad4_alex9;
  functable["tet4_quad4_alex10"] = test_tet4_quad4_alex10;
  //functable["tet4_quad4_alex11"] = test_tet4_quad4_alex11;
  functable["hex8_quad4_alex12"] = test_hex8_quad4_alex12;
  functable["hex8_quad4_alex13"] = test_hex8_quad4_alex13;
  functable["hex8_quad4_alex14"] = test_hex8_quad4_alex14;
  functable["hex8_quad4_alex15"] = test_hex8_quad4_alex15;
  functable["tet4_quad4_alex16"] = test_tet4_quad4_alex16;
  functable["hex8_quad4_alex17"] = test_hex8_quad4_alex17;
  functable["hex8_quad4_alex18"] = test_hex8_quad4_alex18;
  functable["hex8_quad4_alex19"] = test_hex8_quad4_alex19;
  functable["hex8_quad4_alex20"] = test_hex8_quad4_alex20;
  functable["hex8_quad4_alex21"] = test_hex8_quad4_alex21;
  functable["hex8_quad4_alex22"] = test_hex8_quad4_alex22;
  functable["hex8_quad4_alex23"] = test_hex8_quad4_alex23;
  functable["hex8_quad4_alex24"] = test_hex8_quad4_alex24;
  functable["hex8_quad4_alex25"] = test_hex8_quad4_alex25;
  functable["hex8_quad4_alex26"] = test_hex8_quad4_alex26;
  functable["hex8_quad4_alex27"] = test_hex8_quad4_alex27;
  functable["hex8_quad4_alex28"] = test_hex8_quad4_alex28;
  functable["hex8_quad4_alex29"] = test_hex8_quad4_alex29;
  functable["hex8_quad4_alex30"] = test_hex8_quad4_alex30;
  functable["hex8_quad4_alex31"] = test_hex8_quad4_alex31;
  functable["hex8_quad4_alex32"] = test_hex8_quad4_alex32;
  functable["hex8_quad4_alex33"] = test_hex8_quad4_alex33;
  functable["hex8_quad4_alex34"] = test_hex8_quad4_alex34;
  functable["hex8_quad4_alex35"] = test_hex8_quad4_alex35;
  functable["hex8_quad4_alex36"] = test_hex8_quad4_alex36;
  functable["hex8_quad4_alex37"] = test_hex8_quad4_alex37;
  functable["hex8_quad4_alex38"] = test_hex8_quad4_alex38;
  functable["hex8_twintri"] = test_hex8_twintri;
  functable["hex8_twinQuad"] = test_hex8_twinQuad;
  functable["hex8_chairCut"] = test_hex8_chairCut;
  functable["hex8_VCut"] = test_hex8_VCut;
  functable["alex39"] = test_alex39;
  functable["alex40"] = test_alex40;
  functable["alex41"] = test_alex41;
  functable["alex42"] = test_alex42;
  functable["alex43"] = test_alex43;
  functable["alex44"] = test_alex44;
  functable["alex45"] = test_alex45;
  functable["alex46"] = test_alex46;
  functable["alex47"] = test_alex47;
  functable["alex48"] = test_alex48;
  functable["alex49"] = test_alex49;
  functable["alex50"] = test_alex50;
  functable["alex51"] = test_alex51;
  functable["alex52"] = test_alex52;
  functable["alex53"] = test_alex53;
  functable["alex54"] = test_alex54;
  functable["alex55"] = test_alex55;
  functable["alex56"] = test_alex56;
  functable["alex57"] = test_alex57;
  functable["alex58"] = test_alex58;
  functable["alex59"] = test_alex59;
  functable["alex60"] = test_alex60;
  functable["alex61"] = test_alex61;
  functable["alex62"] = test_alex62;
  functable["hex8_quad4_axel1"] = test_hex8_quad4_axel1;
  functable["hex8_quad4_axel2"] = test_hex8_quad4_axel2;
  functable["hex8_quad4_axel3"] = test_hex8_quad4_axel3;
  functable["hex8_quad4_axel4"] = test_hex8_quad4_axel4;
  functable["hex8_quad4_axel5"] = test_hex8_quad4_axel5;
  functable["hex8_quad4_axel6"] = test_hex8_quad4_axel6;
  functable["hex8_quad4_axel7"] = test_hex8_quad4_axel7;
  functable["axel8"] = test_axel8;
  functable["axel9"] = test_axel9;
  functable["axel10"] = test_axel10;
  functable["hex8_quad4_shadan1"] = test_hex8_quad4_shadan1;
  functable["hex8_quad4_shadan2"] = test_hex8_quad4_shadan2;
  functable["hex8_quad4_shadan3"] = test_hex8_quad4_shadan3;
  functable["hex8_quad4_shadan4"] = test_hex8_quad4_shadan4;
  functable["hex8_quad4_shadan5"] = test_hex8_quad4_shadan5;
  functable["shadan6"] = test_shadan6;
  //functable["hex8_tri3_ursula1"] = test_hex8_tri3_ursula1;
  functable["hex8_quad4_mesh_edgecut"] = test_hex8_quad4_mesh_edgecut;
  functable["hex8_quad4_mesh_edgecut2"] = test_hex8_quad4_mesh_edgecut2;
  //functable["hex8_quad4_mesh_inner"] = test_hex8_quad4_mesh_inner;
  functable["hex8_quad4_mesh_many"] = test_hex8_quad4_mesh_many;
  functable["hex27_quad9_simple"] = test_hex27_quad9_simple;
  functable["hex20_quad9_simple"] = test_hex20_quad9_simple;
  functable["hex20_quad9_moved"] = test_hex20_quad9_moved;
  functable["tet10_quad9_simple"] = test_tet10_quad9_simple;
  functable["tet10_quad9_moved"] = test_tet10_quad9_moved;
  functable["tet4_quad4_double"] = test_tet4_quad4_double;
  functable["tet4_tri3_double"] = test_tet4_tri3_double;
  functable["benedikt1"] = test_benedikt1;


  functable["ls_hex8_florian1"] = test_ls_hex8_florian1;
  functable["ls_hex8_florian2"] = test_ls_hex8_florian2;
  functable["ls_hex8_florian3"] = test_ls_hex8_florian3;
  functable["ls_hex8_florian4"] = test_ls_hex8_florian4;
  functable["ls_hex8_florian5"] = test_ls_hex8_florian5;
  functable["ls_hex8_florian6"] = test_ls_hex8_florian6;
  functable["ls_hex8_florian7"] = test_ls_hex8_florian7;
  functable["ls_hex8_florian8"] = test_ls_hex8_florian8;
  functable["ls_hex8_florian9"] = test_ls_hex8_florian9;
  functable["ls_hex8_florian10"] = test_ls_hex8_florian10;
  functable["ls_hex8_florian11"] = test_ls_hex8_florian11;
  functable["ls_hex8_florian12"] = test_ls_hex8_florian12;
  functable["ls_hex8_florian13"] = test_ls_hex8_florian13;
  functable["ls_hex8_ursula1"] = test_ls_hex8_ursula1;
  functable["ls_hex8_ursula2"] = test_ls_hex8_ursula2;
  functable["ls_hex8_ursula3"] = test_ls_hex8_ursula3;
  functable["ls_hex8_ursula4"] = test_ls_hex8_ursula4;
  functable["ls_hex8_ursula5"] = test_ls_hex8_ursula5;
  functable["ls_hex8_ursula6"] = test_ls_hex8_ursula6;
  functable["ls_hex8_simple"] = test_ls_hex8_simple;
  functable["ls_hex8_simple2"] = test_ls_hex8_simple2;
  functable["ls_hex8_simple3"] = test_ls_hex8_simple3;
  //functable["ls_hex8_simple4"] = test_ls_hex8_simple4;
  functable["ls_hex8_simple5"] = test_ls_hex8_simple5;
  functable["ls_hex8_simple6"] = test_ls_hex8_simple6;
  functable["ls_hex8_simple7"] = test_ls_hex8_simple7;
  functable["ls_hex8_touch"] = test_ls_hex8_touch;
  functable["ls_hex8_between"] = test_ls_hex8_between;
  functable["ls_hex8_experiment"] = test_ls_hex8_experiment;

  functable["quad4_surface_mesh_cut"] = test_quad4_surface_mesh_cut;
  functable["hex8_quad4_double_cut"] = test_hex8_quad4_double_cut;

  functable["unit_intersection_touch"] = test_unit_intersection_touch;

  functable["facets_corner_points"] = test_facets_corner_points;

  functable["colored_graph"] = test_colored_graph;
  functable["colored_graph2"] = test_colored_graph2;
  functable["graph"] = test_graph;
  functable["graph2"] = test_graph2;

  functable["geometry"] = test_geometry;

  functable["cut_volumes"] = test_cut_volumes;
#if 0
  // Does not work with current volume cell construction
  // algorithms. FacetGraph fails here.
  functable["cut_volumes2"] = test_cut_volumes2;
  functable["cut_volumes3"] = test_cut_volumes3;
#endif

  functable["fluidfluid"] = test_fluidfluid;
  functable["fluidfluid2"] = test_fluidfluid2;

  Teuchos::CommandLineProcessor clp( false );

  std::string indent = "\t\t\t\t\t";
  std::stringstream doc;
  doc << "Available tests:\n"
      << indent << "(all)\n";
  for ( std::map<std::string, testfunct>::iterator i=functable.begin(); i!=functable.end(); ++i )
  {
    const std::string & name = i->first;
    doc << indent << name << "\n";
  }

  std::string testname = "(all)";
  clp.setOption( "test", &testname, doc.str().c_str() );

  switch ( clp.parse( argc, argv ) )
  {
  case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
    break;
  case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
    return 0;
  case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
    std::cerr << argv[0] << ": unrecognized option\n";
    MPI_Finalize();
    return 1;
  }

  int result = runtests( argv, functable, testname );
  DRT::UTILS::GaussPointCache::Instance().Done();
  MPI_Finalize();
  return result;
}
