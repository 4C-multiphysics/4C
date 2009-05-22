/*!
 \file post_drt_ensight.cpp

 \brief main routine of the Ensight filter

 <pre>
 Maintainer: Ulrich Kuettler
 kuettler@lnm.mw.tum.de
 http://www.lnm.mw.tum.de/Members/kuettler
 089 - 289-15238
 </pre>

 */

#ifdef CCADISCRET

#include "post_drt_ensight_writer.H"
#include "post_drt_ensight_single_field_writers.H"

/*!
 \brief filter main routine

 Write binary ensight format.

 The ens_checker that is part of ensight can be used to verify the
 files generated here.

 \author u.kue
 \date 03/07
 */
int main(
        int argc,
        char** argv)
{
    Teuchos::CommandLineProcessor My_CLP;
    My_CLP.setDocString("Post DRT ensight Filter\n");

    PostProblem problem(My_CLP, argc, argv);

#if 0
    for (int i = 0; i<problem.num_discr(); ++i)
    {
        PostField* field = problem.get_discretization(i);
        StructureEnsightWriter writer(field, problem.outname());
        writer.WriteFiles();
    }
#endif

    // each problem type is different and writes different results
    switch (problem.Problemtype())
    {
    case prb_fsi:
    {
        string basename = problem.outname();
        PostField* structfield = problem.get_discretization(0);
        StructureEnsightWriter structwriter(structfield, basename, problem.stresstype(), problem.straintype());
        structwriter.WriteFiles();

        PostField* fluidfield = problem.get_discretization(1);
        FluidEnsightWriter fluidwriter(fluidfield, basename);
        fluidwriter.WriteFiles();

        //PostField* alefield = problem.get_discretization(2);
        //AleEnsightWriter alewriter(alefield, basename);
        //alewriter.WriteFiles();
        break;
    }
    case prb_structure:
    case prb_struct_multi:
    {
        PostField* field = problem.get_discretization(0);
        StructureEnsightWriter writer(field, problem.outname(), problem.stresstype(), problem.straintype());
        writer.WriteFiles();
        break;
    }
    case prb_fluid:
    case prb_fluid_ale:
    case prb_freesurf:
    {
        PostField* field = problem.get_discretization(0);
        FluidEnsightWriter writer(field, problem.outname());
        writer.WriteFiles();
        break;
    }
    case prb_fluid_dgfem:
    {
        PostField* field = problem.get_discretization(0);
        DGFEMFluidEnsightWriter writer(field, problem.outname());
        writer.WriteFiles();
        break;
    }
    case prb_ale:
    {
        PostField* field = problem.get_discretization(0);
        AleEnsightWriter writer(field, problem.outname());
        writer.WriteFiles();
        break;
    }
    case prb_scatra:
    {
        string basename = problem.outname();
        // do we have a fluid discretization?
        int numfield = problem.num_discr();
        if(numfield==2)
        {
          PostField* fluidfield = problem.get_discretization(0);
          FluidEnsightWriter fluidwriter(fluidfield, basename);
          fluidwriter.WriteFiles();

          PostField* scatrafield = problem.get_discretization(1);
          ScaTraEnsightWriter scatrawriter(scatrafield, basename);
          scatrawriter.WriteFiles();
        }
        else if (numfield==1)
        {
          PostField* scatrafield = problem.get_discretization(0);
          ScaTraEnsightWriter scatrawriter(scatrafield, basename);
          scatrawriter.WriteFiles();
        }
        else
          dserror("number of fields does not match: got %d",numfield);

        break;
    }
    case prb_fluid_xfem: case prb_fsi_xfem:
    {
        cout << "Output XFEM Problem" << endl;

        cout << "  Structural Field" << endl;
        string basename = problem.outname();
        PostField* structfield = problem.get_discretization(0);
        StructureEnsightWriter structwriter(structfield, problem.outname(), problem.stresstype(), problem.straintype());
        structwriter.WriteFiles();

        cout << "  Fluid Field" << endl;
        PostField* fluidfield = problem.get_discretization(1);
        FluidEnsightWriter fluidwriter(fluidfield, basename);
        fluidwriter.WriteFiles();
        
        // in the future, we might also write the interface
        // but at the moment, some procs might have no row elements
        // and the HDF5 writing process can not handle this 
//        cout << "  Interface Field" << endl;
//        PostField* ifacefield = problem.get_discretization(2);
//        InterfaceEnsightWriter ifacewriter(ifacefield, basename);
//        ifacewriter.WriteFiles();
        break;
    }
    case prb_loma:
    {
        string basename = problem.outname();

        PostField* fluidfield = problem.get_discretization(0);
        FluidEnsightWriter fluidwriter(fluidfield, basename);
        fluidwriter.WriteFiles();

        PostField* scatrafield = problem.get_discretization(1);
        ScaTraEnsightWriter scatrawriter(scatrafield, basename);
        scatrawriter.WriteFiles();
        break;
    }
    case prb_elch:
    {
      string basename = problem.outname();
      int numfield = problem.num_discr();
      if(numfield==3)
      {
        // Fluid, ScaTra and ALE fields are present
        PostField* fluidfield = problem.get_discretization(0);
        FluidEnsightWriter fluidwriter(fluidfield, basename);
        fluidwriter.WriteFiles();

        PostField* scatrafield = problem.get_discretization(1);
        ElchEnsightWriter elchwriter(scatrafield, basename);
        elchwriter.WriteFiles();

        PostField* alefield = problem.get_discretization(2);
        AleEnsightWriter alewriter(alefield, basename);
        alewriter.WriteFiles();
      }
      else if(numfield==2)
      {
        // Fluid and ScaTra fields are present
        PostField* fluidfield = problem.get_discretization(0);
        FluidEnsightWriter fluidwriter(fluidfield, basename);
        fluidwriter.WriteFiles();

        PostField* scatrafield = problem.get_discretization(1);
        ElchEnsightWriter elchwriter(scatrafield, basename);
        elchwriter.WriteFiles();
        break;
      }
      else if (numfield==1)
      {
        // only a ScaTra field is present
        PostField* scatrafield = problem.get_discretization(0);
        ElchEnsightWriter elchwriter(scatrafield, basename);
        elchwriter.WriteFiles();
      }
      else
        dserror("number of fields does not match: got %d",numfield);
      break;
    }
    case prb_combust:
    {
        string basename = problem.outname();

        PostField* fluidfield = problem.get_discretization(0);
        FluidEnsightWriter fluidwriter(fluidfield, basename);
        fluidwriter.WriteFiles();

        PostField* scatrafield = problem.get_discretization(1);
        ScaTraEnsightWriter scatrawriter(scatrafield, basename);
        scatrawriter.WriteFiles();
        break;
    }
    case prb_none:
    {
      // Special problem type that contains one discretization and any number
      // of vectors. We just want to see whatever there is.
      PostField* field = problem.get_discretization(0);
      AnyEnsightWriter writer(field, problem.outname());
      writer.WriteFiles();
      break;
    }
    default:
        dserror("problem type %d not yet supported", problem.Problemtype());
    }

    return 0;
}

#endif
