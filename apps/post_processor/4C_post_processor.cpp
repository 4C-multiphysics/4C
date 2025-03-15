// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_post_common.hpp"
#include "4C_post_processor_single_field_writers.hpp"
#include "4C_post_writer_base.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"


using namespace FourC;
namespace
{

  void run_ensight_vtu_filter(PostProblem& problem)
  {
    // each problem type is different and writes different results
    switch (problem.problemtype())
    {
      case Core::ProblemType::fsi:
      case Core::ProblemType::fsi_redmodels:
      {
        std::string basename = problem.outname();
        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* fluidfield = problem.get_discretization(1);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        PostField* alefield = problem.get_discretization(2);
        AleFilter alewriter(alefield, basename);
        alewriter.write_files();
        // 1d artery
        if (problem.num_discr() == 4)
        {
          PostField* field = problem.get_discretization(2);
          StructureFilter writer(field, basename, problem.stresstype(), problem.straintype());
          writer.write_files();
        }
        if (problem.num_discr() > 2 and problem.get_discretization(2)->name() == "xfluid")
        {
          std::string basename = problem.outname();
          PostField* fluidfield = problem.get_discretization(2);
          FluidFilter xfluidwriter(fluidfield, basename);
          xfluidwriter.write_files();
        }
        break;
      }
      case Core::ProblemType::gas_fsi:
      case Core::ProblemType::thermo_fsi:
      {
        std::string basename = problem.outname();
        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* fluidfield = problem.get_discretization(1);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        break;
      }
      case Core::ProblemType::biofilm_fsi:
      {
        std::string basename = problem.outname();
        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* fluidfield = problem.get_discretization(1);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        break;
      }
      case Core::ProblemType::structure:
      {
        // Regular solid/structure output
        {
          PostField* structurefield = problem.get_discretization(0);
          StructureFilter structwriter(
              structurefield, problem.outname(), problem.stresstype(), problem.straintype());
          structwriter.write_files();
        }

        // Deal with contact / meshtying problems
        if (problem.do_mortar_interfaces())
        {
          /* Loop over all mortar interfaces and process each interface individually
           *
           * Start at i = 1 since discretization '0' is the structure discretization.
           *
           * Note: We assume that there is only one structure discretization and
           * that all other discretizations are mortar interface discretizations.
           */
          for (int i = 1; i < problem.num_discr(); ++i)
          {
            PostField* mortarfield = problem.get_discretization(i);
            MortarFilter mortarwriter(mortarfield, problem.outname());
            mortarwriter.write_files();
          }
        }

        break;
      }
      case Core::ProblemType::polymernetwork:
      {
        int numdiscr = problem.num_discr();
        for (int i = 0; i < numdiscr; ++i)
        {
          std::string disname = problem.get_discretization(i)->name();
          if (disname == "structure")
          {
            PostField* structure = problem.get_discretization(i);
            StructureFilter writer(structure, problem.outname());
            writer.write_files();
          }
          else if (disname == "ia_structure")
          {
            PostField* ia_structure = problem.get_discretization(i);
            StructureFilter iawriter(ia_structure, problem.outname());
            iawriter.write_files();
          }
          else if (disname == "boundingbox")
          {
            PostField* boxdiscret = problem.get_discretization(i);
            StructureFilter boxwriter(boxdiscret, problem.outname());
            boxwriter.write_files();
          }
          else if (disname == "bins")
          {
            PostField* visualizebins = problem.get_discretization(i);
            StructureFilter binwriter(visualizebins, problem.outname());
            binwriter.write_files();
          }
          else
          {
            FOUR_C_THROW("unknown discretization for postprocessing of polymer network problem!");
          }
        }
        break;
      }
      case Core::ProblemType::fluid:
      {
        if (problem.num_discr() == 2)
        {
          if (problem.get_discretization(1)->name() == "xfluid")
          {
            std::string basename = problem.outname();
            PostField* fluidfield = problem.get_discretization(1);
            XFluidFilter xfluidwriter(fluidfield, basename);
            xfluidwriter.write_files();
          }
        }
        [[fallthrough]];
      }
      case Core::ProblemType::fluid_redmodels:
      {
        if (problem.num_discr() == 2)
        {
          // 1d artery
          std::string basename = problem.outname();
          PostField* field = problem.get_discretization(1);
          StructureFilter writer(field, basename, problem.stresstype(), problem.straintype());
          writer.write_files();
          if (problem.get_discretization(1)->name() == "xfluid")
          {
            std::string basename = problem.outname();
            PostField* fluidfield = problem.get_discretization(1);
            XFluidFilter xfluidwriter(fluidfield, basename);
            xfluidwriter.write_files();
          }
        }
        [[fallthrough]];
      }
      case Core::ProblemType::fluid_ale:
      {
        PostField* field = problem.get_discretization(0);
        FluidFilter writer(field, problem.outname());
        writer.write_files();
        if (problem.num_discr() > 1 and problem.get_discretization(1)->name() == "xfluid")
        {
          std::string basename = problem.outname();
          PostField* fluidfield = problem.get_discretization(1);
          FluidFilter xfluidwriter(fluidfield, basename);
          xfluidwriter.write_files();
        }
        break;
      }
      case Core::ProblemType::particle:
      case Core::ProblemType::pasi:
      {
        int numdiscr = problem.num_discr();
        for (int i = 0; i < numdiscr; ++i)
        {
          std::string disname = problem.get_discretization(i)->name();
          if (disname == "bins")
          {
            PostField* visualizebins = problem.get_discretization(i);
            StructureFilter binwriter(visualizebins, problem.outname());
            binwriter.write_files();
          }
          else if (disname == "structure")
          {
            PostField* structure = problem.get_discretization(i);
            StructureFilter writer(
                structure, problem.outname(), problem.stresstype(), problem.straintype());
            writer.write_files();
          }
          else
          {
            FOUR_C_THROW("Particle problem has illegal discretization name!");
          }
        }
        break;
      }
      case Core::ProblemType::redairways_tissue:
      {
        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, problem.outname(), problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* fluidfield = problem.get_discretization(1);
        StructureFilter fluidwriter(
            fluidfield, problem.outname(), problem.stresstype(), problem.straintype());
        fluidwriter.write_files();
        break;
      }
      case Core::ProblemType::ale:
      {
        PostField* field = problem.get_discretization(0);
        AleFilter writer(field, problem.outname());
        writer.write_files();
        break;
      }
      case Core::ProblemType::lubrication:
      {
        PostField* lubricationfield = problem.get_discretization(0);
        LubricationFilter lubricationwriter(lubricationfield, problem.outname());
        lubricationwriter.write_files();
        break;
      }
      case Core::ProblemType::cardiac_monodomain:
      case Core::ProblemType::scatra:
      {
        std::string basename = problem.outname();
        // do we have a fluid discretization?
        int numfields = problem.num_discr();
        if (numfields == 2)
        {
          PostField* fluidfield = problem.get_discretization(0);
          FluidFilter fluidwriter(fluidfield, basename);
          fluidwriter.write_files();
        }
        else if (numfields == 1)
        {
          // runtime output is used for scatra
        }
        else
          FOUR_C_THROW("number of fields does not match: got %d", numfields);

        break;
      }
      case Core::ProblemType::fsi_xfem:
      case Core::ProblemType::fpsi_xfem:
      case Core::ProblemType::fluid_xfem_ls:
      {
        std::cout
            << "|=============================================================================|"
            << std::endl;
        std::cout << "|==  Output for General Problem" << std::endl;

        int numfield = problem.num_discr();

        std::cout << "|==  Number of discretizations: " << numfield << std::endl;

        std::string basename = problem.outname();

        std::cout << "\n|==  Start postprocessing for discretizations:" << std::endl;

        for (int idx_int = 0; idx_int < numfield; idx_int++)
        {
          std::cout
              << "\n|=============================================================================|"
              << std::endl;
          std::string disname = problem.get_discretization(idx_int)->name();
          PostField* field = problem.get_discretization(idx_int);
          if (disname == "structure")
          {
            std::cout << "|==  Structural Field ( " << disname << " )" << std::endl;
            StructureFilter structwriter(
                field, basename, problem.stresstype(), problem.straintype());
            structwriter.write_files();
          }
          else if (disname == "fluid" or disname == "xfluid" or disname == "porofluid")
          {
            std::cout << "|==    Fluid Field ( " << disname << " )" << std::endl;
            FluidFilter fluidwriter(field, basename);
            fluidwriter.write_files();
          }
          else if (disname == "scatra")
          {
            // runtime output is used for scatra
          }
          else if (disname == "ale")
          {
            std::cout << "|==    Ale Field ( " << disname << " )" << std::endl;
            //          AleFilter alewriter(field, basename);
            //          alewriter.WriteFiles();
          }
          else if (disname.compare(1, 12, "boundary_of_"))
          {
            std::cout << "|==    Interface Field ( " << disname << " )" << std::endl;
            InterfaceFilter ifacewriter(field, basename);
            ifacewriter.write_files();
          }
          else
            FOUR_C_THROW(
                "You try to postprocess a discretization with name %s, maybe you should add it "
                "here?",
                disname.c_str());
        }
        std::cout
            << "|=============================================================================|"
            << std::endl;
        break;
      }
      case Core::ProblemType::fluid_xfem:
      {
        std::cout << "Output FLUID-XFEM Problem" << std::endl;

        int numfield = problem.num_discr();

        std::cout << "Number of discretizations: " << numfield << std::endl;
        for (int i = 0; i < numfield; i++)
        {
          std::cout << "dis-name i=" << i << ": " << problem.get_discretization(i)->name()
                    << std::endl;
        }

        if (numfield == 0) FOUR_C_THROW("we expect at least a fluid field, numfield=%i", numfield);
        std::string basename = problem.outname();

        // XFluid in the standard case, embedded fluid for XFF
        std::cout << "  Fluid Field" << std::endl;
        PostField* fluidfield = problem.get_discretization(0);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        // start index for interface discretizations
        int idx_int = 1;
        if (problem.num_discr() > 1 and problem.get_discretization(1)->name() == "xfluid")
        {
          // XFluid for XFF
          std::cout << "  XFluid Field" << std::endl;
          PostField* xfluidfield = problem.get_discretization(1);
          FluidFilter xfluidwriter(xfluidfield, basename);
          xfluidwriter.write_files();
          idx_int += 1;
        }

        // all other fields are interface fields
        for (int i = idx_int; i < numfield; i++)
        {
          std::cout << "  Interface Field ( " << problem.get_discretization(i)->name() << " )"
                    << std::endl;
          PostField* ifacefield = problem.get_discretization(i);
          InterfaceFilter ifacewriter(ifacefield, basename);
          ifacewriter.write_files();
        }
        break;
      }
      case Core::ProblemType::loma:
      {
        std::string basename = problem.outname();

        PostField* fluidfield = problem.get_discretization(0);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        break;
      }
      case Core::ProblemType::elch:
      {
        std::string basename = problem.outname();
        int numfield = problem.num_discr();

        if (numfield == 3)
        {
          // Fluid, ScaTra and ALE fields are present
          PostField* fluidfield = problem.get_discretization(0);
          FluidFilter fluidwriter(fluidfield, basename);
          fluidwriter.write_files();

          PostField* alefield = problem.get_discretization(2);
          AleFilter alewriter(alefield, basename);
          alewriter.write_files();
        }
        else if (numfield == 2)
        {
          auto* const dis0 = problem.get_discretization(0);
          auto* const dis1 = problem.get_discretization(1);

          if (dis0->name() == "scatra" and dis1->name() == "scatra_micro")
          {
            // runtime output is used for scatra
          }
          else
          {
            // Fluid and ScaTra fields are present
            PostField* fluidfield = dis0;
            FluidFilter fluidwriter(fluidfield, basename);
            fluidwriter.write_files();
          }
          break;
        }
        else if (numfield == 1)
        {
          // runtime output is used for scatra
        }
        else
          FOUR_C_THROW("number of fields does not match: got %d", numfield);
        break;
      }
      case Core::ProblemType::art_net:
      {
        std::string basename = problem.outname();
        PostField* field = problem.get_discretization(0);
        // AnyFilter writer(field, problem.outname());
        StructureFilter writer(field, basename, problem.stresstype(), problem.straintype());
        writer.write_files();

        break;
      }
      case Core::ProblemType::thermo:
      {
        PostField* field = problem.get_discretization(0);
        ThermoFilter writer(
            field, problem.outname(), problem.heatfluxtype(), problem.tempgradtype());
        writer.write_files();
        break;
      }
      case Core::ProblemType::tsi:
      {
        std::cout << "Output TSI Problem" << std::endl;

        std::string basename = problem.outname();

        PostField* thermfield = problem.get_discretization(0);
        ThermoFilter thermwriter(
            thermfield, basename, problem.heatfluxtype(), problem.tempgradtype());
        thermwriter.write_files();

        PostField* structfield = problem.get_discretization(1);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();
        break;
      }
      case Core::ProblemType::red_airways:
      {
        std::string basename = problem.outname();
        PostField* field = problem.get_discretization(0);
        // AnyFilter writer(field, problem.outname());
        StructureFilter writer(field, basename, problem.stresstype(), problem.straintype());
        writer.write_files();
        //      writer.WriteFiles();

        break;
      }
      case Core::ProblemType::poroelast:
      {
        std::string basename = problem.outname();

        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* fluidfield = problem.get_discretization(1);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();
        break;
      }
      case Core::ProblemType::poroscatra:
      {
        std::string basename = problem.outname();

        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* fluidfield = problem.get_discretization(1);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        break;
      }
      case Core::ProblemType::fpsi:
      {
        std::string basename = problem.outname();

        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* porofluidfield = problem.get_discretization(1);
        FluidFilter porofluidwriter(porofluidfield, basename);
        porofluidwriter.write_files();

        PostField* fluidfield = problem.get_discretization(2);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        break;
      }
      case Core::ProblemType::fbi:
      {
        std::string basename = problem.outname();

        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* fluidfield = problem.get_discretization(1);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        break;
      }
      case Core::ProblemType::fps3i:
      {
        std::string basename = problem.outname();

        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* porofluidfield = problem.get_discretization(1);
        FluidFilter porofluidwriter(porofluidfield, basename);
        porofluidwriter.write_files();

        PostField* fluidfield = problem.get_discretization(2);
        FluidFilter fluidwriter(fluidfield, basename);
        fluidwriter.write_files();

        break;
      }
      case Core::ProblemType::ehl:
      {
        std::string basename = problem.outname();

        PostField* structfield = problem.get_discretization(0);
        StructureFilter structwriter(
            structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.write_files();

        PostField* lubricationfield = problem.get_discretization(1);
        LubricationFilter lubricationwriter(lubricationfield, basename);
        lubricationwriter.write_files();

        break;
      }
      case Core::ProblemType::none:
      {
        // Special problem type that contains one discretization and any number
        // of vectors. We just want to see whatever there is.
        PostField* field = problem.get_discretization(0);
        AnyFilter writer(field, problem.outname());
        writer.write_files();
        break;
      }
      default:
        FOUR_C_THROW("problem type %d not yet supported", problem.problemtype());
        break;
    }
  }

  std::string get_filter(int argc, char** argv)
  {
    Teuchos::CommandLineProcessor clp(false, false, false);
    std::string filter("ensight");
    clp.setOption("filter", &filter, "filter to run [ensight, vtu, vti]");
    // ignore warnings about unrecognized options.
    std::ostringstream warning;
    clp.parse(argc, argv, &warning);
    return filter;
  }
}  // namespace


/*!
 \brief post-processor main routine

 Select the appropriate filter and run!

 */
int main(int argc, char** argv)
{
  Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  try
  {
    std::string filter = get_filter(argc, argv);
    Teuchos::CommandLineProcessor My_CLP;
    My_CLP.setDocString("Main 4C post-processor\n");

    PostProblem problem(My_CLP, argc, argv);

    if (filter == "ensight" || filter == "vtu" || filter == "vtu_node_based" || filter == "vti")
      run_ensight_vtu_filter(problem);
    else
      FOUR_C_THROW("Unknown filter %s given, supported filters: [ensight|vtu|vti]", filter.c_str());

  }  // try
  catch (Core::Exception& err)
  {
    char line[] = "=========================================================================\n";
    std::cout << "\n\n" << line << err.what_with_stacktrace() << "\n" << line << "\n" << std::endl;

#ifdef FOUR_C_ENABLE_CORE_DUMP
    abort();
#endif

    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }  // catch

  return 0;
}
