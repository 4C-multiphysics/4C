/*----------------------------------------------------------------------*/
/*! \file

\brief input parameters for VTK output of structural problem at runtime

\level 2

*/
/*----------------------------------------------------------------------*/

#include "baci_inpar_IO_runtime_vtk_output.H"

#include "baci_inpar_validparameters.H"
#include "baci_inpar.H"
#include "baci_inpar_parameterlist_utils.H"

#include <Teuchos_ParameterList.hpp>

namespace INPAR
{
  namespace IO_RUNTIME_VTK
  {
    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    void SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
    {
      using namespace DRT::INPUT;
      using Teuchos::setStringToIntegralParameter;
      using Teuchos::tuple;

      // related sublist
      Teuchos::ParameterList& sublist_IO = list->sublist("IO", false, "");
      Teuchos::ParameterList& sublist_IO_VTK_structure =
          sublist_IO.sublist("RUNTIME VTK OUTPUT", false, "");


      // output interval regarding steps: write output every INTERVAL_STEPS steps
      IntParameter("INTERVAL_STEPS", -1, "write VTK output at runtime every INTERVAL_STEPS steps",
          &sublist_IO_VTK_structure);


      IntParameter("STEP_OFFSET", 0,
          "An offset added to the current step to shift the steps to be written.",
          &sublist_IO_VTK_structure);


      // data format for written numeric data
      setStringToIntegralParameter<int>("OUTPUT_DATA_FORMAT", "binary",
          "data format for written numeric data",
          tuple<std::string>("binary", "Binary", "ascii", "ASCII"),
          tuple<int>(INPAR::IO_RUNTIME_VTK::binary, INPAR::IO_RUNTIME_VTK::binary,
              INPAR::IO_RUNTIME_VTK::ascii, INPAR::IO_RUNTIME_VTK::ascii),
          &sublist_IO_VTK_structure);


      // whether to write output in every iteration of the nonlinear solver
      BoolParameter("EVERY_ITERATION", "No",
          "write output in every iteration of the nonlinear solver", &sublist_IO_VTK_structure);
    }


  }  // namespace IO_RUNTIME_VTK
}  // namespace INPAR