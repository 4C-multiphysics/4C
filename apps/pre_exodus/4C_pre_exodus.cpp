// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_inpar_validconditions.hpp"
#include "4C_inpar_validmaterials.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_pre_exodus_readbc.hpp"
#include "4C_pre_exodus_reader.hpp"
#include "4C_pre_exodus_validate.hpp"
#include "4C_pre_exodus_writedat.hpp"
#include "4C_utils_result_test.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Epetra_MpiComm.h>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <memory>


using namespace FourC;

namespace
{
  /*----------------------------------------------------------------------*/
  /* create default bc file                                               */
  /*----------------------------------------------------------------------*/
  int create_default_bc_file(EXODUS::Mesh& mymesh)
  {
    using namespace FourC;

    std::string defaultbcfilename = "default.bc";
    std::cout << "found no BC specification file --> creating " << defaultbcfilename << std::endl;

    // open default bc specification file
    std::ofstream defaultbc(defaultbcfilename.c_str());
    if (!defaultbc) FOUR_C_THROW("failed to open file: %s", defaultbcfilename.c_str());

    // write mesh verbosely
    defaultbc << "----------- Mesh contents -----------" << std::endl << std::endl;
    mymesh.print(defaultbc, false);

    // give examples for element and boundary condition syntax
    defaultbc << "---------- Syntax examples ----------" << std::endl
              << std::endl
              << "Element Block, named: " << std::endl
              << "of Shape: TET4" << std::endl
              << "has 9417816 Elements" << std::endl
              << "'*eb0=\"ELEMENT\"'" << std::endl
              << "sectionname=\"FLUID\"" << std::endl
              << "description=\"MAT 1 NA Euler\"" << std::endl
              << "elementname=\"FLUID\" \n"
              << std::endl
              << "Element Block, named: " << std::endl
              << "of Shape: HEX8" << std::endl
              << "has 9417816 Elements" << std::endl
              << "'*eb0=\"ELEMENT\"'" << std::endl
              << "sectionname=\"STRUCTURE\"" << std::endl
              << "description=\"MAT 1 KINEM nonlinear\"" << std::endl
              << "elementname=\"SOLID\" \n"
              << std::endl
              << "Node Set, named:" << std::endl
              << "Property Name: INFLOW" << std::endl
              << "has 45107 Nodes" << std::endl
              << "'*ns0=\"CONDITION\"'" << std::endl
              << "sectionname=\"DESIGN SURF DIRICH CONDITIONS\"" << std::endl
              << "description=\"NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 2.0 0.0 0.0 0.0 0.0 0.0 FUNCT 1 0 0 "
                 "0 0 0\""
              << std::endl
              << std::endl;

    defaultbc << "MIND that you can specify a condition also on an ElementBlock, just replace "
                 "'ELEMENT' with 'CONDITION'"
              << std::endl;
    defaultbc << "The 'E num' in the dat-file depends on the order of the specification below"
              << std::endl;
    defaultbc << "------------------------------------------------BCSPECS" << std::endl
              << std::endl;

    // write ElementBlocks with specification proposal
    const std::map<int, std::shared_ptr<EXODUS::ElementBlock>> myblocks =
        mymesh.get_element_blocks();
    std::map<int, std::shared_ptr<EXODUS::ElementBlock>>::const_iterator it;
    for (it = myblocks.begin(); it != myblocks.end(); ++it)
    {
      it->second->print(defaultbc);
      defaultbc << "*eb" << it->first << "=\"ELEMENT\"" << std::endl
                << "sectionname=\"\"" << std::endl
                << "description=\"\"" << std::endl
                << "elementname=\"\"" << std::endl
                << std::endl;
    }

    // write NodeSets with specification proposal
    const std::map<int, EXODUS::NodeSet> mynodesets = mymesh.get_node_sets();
    std::map<int, EXODUS::NodeSet>::const_iterator ins;
    for (ins = mynodesets.begin(); ins != mynodesets.end(); ++ins)
    {
      ins->second.print(defaultbc);
      defaultbc << "*ns" << ins->first << "=\"CONDITION\"" << std::endl
                << "sectionname=\"\"" << std::endl
                << "description=\"\"" << std::endl
                << std::endl;
    }

    // write SideSets with specification proposal
    const std::map<int, EXODUS::SideSet> mysidesets = mymesh.get_side_sets();
    std::map<int, EXODUS::SideSet>::const_iterator iss;
    for (iss = mysidesets.begin(); iss != mysidesets.end(); ++iss)
    {
      iss->second.print(defaultbc);
      defaultbc << "*ss" << iss->first << "=\"CONDITION\"" << std::endl
                << "sectionname=\"\"" << std::endl
                << "description=\"\"" << std::endl
                << std::endl;
    }

    // print validconditions as proposal
    defaultbc << "-----------------------------------------VALIDCONDITIONS" << std::endl;
    std::shared_ptr<std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>> condlist =
        Input::valid_conditions();
    Input::print_empty_condition_definitions(defaultbc, *condlist);

    // print valid element lines as proposal (parobjects have to be registered for doing this!)
    defaultbc << std::endl << std::endl;
    Core::Elements::ElementDefinition ed;
    ed.print_element_dat_header_to_stream(defaultbc);

    // close default bc specification file
    if (defaultbc.is_open()) defaultbc.close();

    return 0;
  }
}  // namespace

/**
 *
 * Pre_exodus contains classes to open and preprocess exodusII files into the
 * discretization of 4C. It uses the "valid-parameters"-list defined in 4C for preparing
 * a up-to-date 4C header and another file specifying element and boundary
 * specifications. As result either a preliminary input file set is suggestioned,
 * or the well-known .dat file is created.
 *
 */
int main(int argc, char** argv)
{
  using namespace FourC;
  Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;

  // communication
  MPI_Init(&argc, &argv);

  global_legacy_module_callbacks().RegisterParObjectTypes();

  // create default communicators
  std::shared_ptr<Core::Communication::Communicators> communicators =
      Core::Communication::create_comm({});
  Global::Problem::instance()->set_communicators(communicators);
  MPI_Comm comm = communicators->global_comm();

  try
  {
    if ((Core::Communication::num_mpi_ranks(comm) > 1))
      FOUR_C_THROW("Using more than one processor is not supported.");

    std::string exofile;
    std::string bcfile;
    std::string headfile;
    std::string datfile;
    std::string cline;

    bool twodim = false;

    // related to quad->tri conversion
    bool quadtri = false;

    Teuchos::CommandLineProcessor My_CLP;
    My_CLP.setDocString(
        "This preprocessor converts Exodus2 files to the proprietary FourC format\n");
    My_CLP.throwExceptions(false);
    My_CLP.setOption("exo", &exofile, "exodus file to open");
    My_CLP.setOption("bc", &bcfile, "bc's and ele's file to open");
    My_CLP.setOption("head", &headfile, "4C header file to open");
    My_CLP.setOption("dat", &datfile, "output .dat file name [defaults to exodus file name]");

    // switch for generating a 2d .dat - file
    My_CLP.setOption("d2", "d3", &twodim, "space dimensions in .dat-file: d2: 2D, d3: 3D");

    // check for quad->tri conversion
    My_CLP.setOption(
        "quadtri", "noquadtri", &quadtri, "transform quads to tris by cutting in two halves");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc, argv);

    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
    {
      return 0;
    }
    if (parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      FOUR_C_THROW("CommandLineProcessor reported an error");
    }

    if (datfile != "")
    {
      const std::string basename = datfile.substr(0, datfile.find_last_of(".")) + "_pre";
      Core::IO::cout.setup(true, false, false, Core::IO::standard, comm, 0, 0,
          basename);  // necessary setup of Core::IO::cout
    }
    else
    {
      Core::IO::cout.setup(true, false, false, Core::IO::standard, comm, 0, 0,
          "xxx_pre");  // necessary setup of Core::IO::cout
    }


    /**************************************************************************
     * Start with the preprocessing
     **************************************************************************/
    if (exofile == "")
    {
      if (datfile != "")
      {
        // just validate a given 4C input file
        EXODUS::validate_input_file(comm, datfile);
        return 0;
      }
      else
      {
        My_CLP.printHelpMessage(argv[0], std::cout);
        FOUR_C_THROW("No Exodus II file was found");
      }
    }

    // create mesh object based on given exodus II file
    EXODUS::Mesh mymesh(exofile);
    // print infos to std::cout
    mymesh.print(std::cout);

    /**************************************************************************
     * Edit a existing Mesh, e.g. extrusion of surface
     **************************************************************************/

    // transform quads->tris
    if (quadtri)
    {
      EXODUS::Mesh trimesh = EXODUS::quadto_tri(mymesh);
      trimesh.write_mesh("tri_" + exofile);
      exit(0);
    }

    /**************************************************************************
     * Read ControlFile for Boundary and Element descriptions
     **************************************************************************/

    // declare empty vectors for holding "boundary" conditions
    std::vector<EXODUS::ElemDef> eledefs;
    std::vector<EXODUS::CondDef> condefs;

    if (bcfile == "")
    {
      int error = create_default_bc_file(mymesh);
      if (error != 0) FOUR_C_THROW("Creation of default bc-file not successful.");
    }
    else
    {
      // read provided bc-file
      EXODUS::read_bc_file(bcfile, eledefs, condefs);

      int sum =
          mymesh.get_num_element_blocks() + mymesh.get_num_node_sets() + mymesh.get_num_side_sets();
      int test = eledefs.size() + condefs.size();
      if (test != sum)
        std::cout
            << "Your " << test << " definitions do not match the " << sum
            << " entities in your mesh!" << std::endl
            << "(This is OK, if more than one BC is applied to an entity, e.g in FSI simulations)"
            << std::endl;
    }

    /**************************************************************************
     * Read HeaderFile for 'header' parameters, e.g. solver, dynamic, material
     * or create a default HeaderFile
     **************************************************************************/
    if (headfile == "")
    {
      const std::string defaultheadfilename = "default.head";
      std::cout << "found no header file           --> creating " << defaultheadfilename
                << std::endl;

      // open default header file
      std::ofstream defaulthead(defaultheadfilename.c_str());
      if (!defaulthead) FOUR_C_THROW("failed to open file: %s", defaultheadfilename.c_str());

      // get valid input parameters
      std::shared_ptr<const Teuchos::ParameterList> list = Input::valid_parameters();

      // write default .dat header into file
      std::stringstream prelimhead;
      Core::IO::InputFileUtils::print_dat(prelimhead, *list);
      std::string headstring = prelimhead.str();
      size_t size_section =
          headstring.find("-------------------------------------------------------PROBLEM SIZE");
      if (size_section != std::string::npos)
      {
        size_t typ_section =
            headstring.find("--------------------------------------------------------PROBLEM TYPE");
        headstring.erase(size_section, typ_section - size_section);
      }
      defaulthead << headstring;

      // get valid input materials
      {
        std::shared_ptr<std::vector<std::shared_ptr<Mat::MaterialDefinition>>> mlist =
            Input::valid_materials();
        Input::print_empty_material_definitions(defaulthead, *mlist);
      }

      // print cloning material map default lines (right after the materials)
      const auto lines = Core::FE::valid_cloning_material_map_lines();
      Core::IO::InputFileUtils::print_section(defaulthead, "CLONING MATERIAL MAP", lines);

      // print spatial functions
      defaulthead << "-------------------------------------------------------------FUNCT1"
                  << std::endl
                  << "-------------------------------------------------------------FUNCT2"
                  << std::endl
                  << "-------------------------------------------------------------FUNCT3"
                  << std::endl
                  << "-------------------------------------------------------------FUNCT4"
                  << std::endl;
      {
        std::stringstream tmp;
        Core::Utils::FunctionManager functionmanager;
        global_legacy_module_callbacks().AttachFunctionDefinitions(functionmanager);
        const std::vector<Input::LineDefinition> flines = functionmanager.valid_function_lines();
        Core::IO::InputFileUtils::print_section(tmp, "FUNCT", flines);
        std::string tmpstring = tmp.str();
        std::string removeit =
            "--------------------------------------------------------------FUNCT\n";
        size_t pos = tmpstring.find(removeit);
        if (pos != std::string::npos)
        {
          tmpstring.erase(pos, removeit.length());
        }
        defaulthead << tmpstring;
      }

      // default result-test lines
      {
        const auto lines = global_legacy_module_callbacks().valid_result_description_lines();
        Core::IO::InputFileUtils::print_section(defaulthead, "RESULT DESCRIPTION", lines);
      }

      // close default header file
      if (defaulthead.is_open()) defaulthead.close();
    }

    /**************************************************************************
     * Finally, create and validate the 4C input file
     **************************************************************************/
    if ((headfile != "") && (bcfile != "") && (exofile != ""))
    {
      // set default dat-file name if needed
      if (datfile == "")
      {
        const std::string exofilebasename = exofile.substr(0, exofile.find_last_of("."));
        datfile = exofilebasename + ".dat";
      }

      // screen info
      std::cout << "creating and checking 4C input file       --> " << datfile << std::endl;
      auto timer = Teuchos::TimeMonitor::getNewTimer("pre-exodus timer");

      // check for positive Element-Center-Jacobians and otherwise rewind them
      {
        std::cout << "...Ensure positive element jacobians";
        timer->start();
        validate_mesh_element_jacobians(mymesh);
        timer->stop();
        std::cout << "        in...." << timer->totalElapsedTime(true) << " secs" << std::endl;
        timer->reset();
      }

      // in case of periodic boundary conditions :
      // ensure that the two coordinates of two matching nodes,
      // which should be the same are exactly the same
      // in order to keep the Krylov norm below 1e-6 :-)
      // only supported for angle 0.0
      {
        if (periodic_boundary_conditions_found(condefs))
        {
          std::cout << "...Ensure high quality p.b.c.";
          timer->start();
          correct_nodal_coordinates_for_periodic_boundary_conditions(mymesh, condefs);
          timer->stop();
          std::cout << "               in...." << timer->totalElapsedTime(true) << " secs"
                    << std::endl;
          timer->reset();
        }
      }

      // write the 4C input file
      {
        if (twodim) mymesh.set_nsd(2);
        std::cout << "...Writing dat-file";
        timer->start();
        EXODUS::write_dat_file(datfile, mymesh, headfile, eledefs, condefs);
        timer->stop();
        std::cout << "                         in...." << timer->totalElapsedTime(true) << " secs"
                  << std::endl;
        timer->reset();
      }

      // validate the generated 4C input file
      EXODUS::validate_input_file(comm, datfile);
    }
  }
  catch (Core::Exception& err)
  {
    char line[] = "=========================================================================\n";
    std::cout << "\n\n" << line << err.what_with_stacktrace() << "\n" << line << "\n" << std::endl;

#ifdef FOUR_C_ENABLE_CORE_DUMP
    abort();
#endif

    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  MPI_Finalize();
  return 0;
}
