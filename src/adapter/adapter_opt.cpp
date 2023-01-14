/*----------------------------------------------------------------------*/
/*! \file

\brief topology optimization adapter

\level 3


 *------------------------------------------------------------------------------------------------*/


#include "adapter_opt.H"
#include "io.H"
#include "io_control.H"
#include "lib_discret.H"
#include "lib_globalproblem.H"
#include "opti_topopt_optimizer.H"
#include "opti_topopt_utils.H"
#include "inpar_parameterlist_utils.H"


/// constructor
ADAPTER::TopOptBaseAlgorithm::TopOptBaseAlgorithm(
    const Teuchos::ParameterList& prbdyn,  ///< problem-dependent parameters
    const std::string disname  ///< optimization field discretization name(default: "opti")
)
{
  DRT::Problem* problem = DRT::Problem::Instance();

  // -------------------------------------------------------------------
  // access the fluid and the optimization discretization
  // -------------------------------------------------------------------
  Teuchos::RCP<DRT::Discretization> optidis = Teuchos::null;
  optidis = problem->GetDis(disname);
  Teuchos::RCP<DRT::Discretization> fluiddis = Teuchos::null;
  fluiddis = problem->GetDis("fluid");

  // -------------------------------------------------------------------
  // check degrees of freedom in the discretization
  // -------------------------------------------------------------------
  if (!optidis->Filled()) dserror("optimization discretization should be filled before");
  if (!fluiddis->Filled()) dserror("fluid discretization should be filled before");

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  // output control for optimization field
  // equal to output for fluid equations except for the filename
  // and the - not necessary - input file name
  Teuchos::RCP<IO::OutputControl> optioutput =
      Teuchos::rcp(new IO::OutputControl(optidis->Comm(), problem->ProblemName(),
          problem->SpatialApproximationType(), problem->OutputControlFile()->InputFileName(),
          TOPOPT::modifyFilename(problem->OutputControlFile()->FileName(), "xxx_opti_",
              (bool)DRT::Problem::Instance()->Restart(), true),
          problem->NDim(), problem->Restart(), problem->OutputControlFile()->FileSteps(),
          DRT::INPUT::IntegralValue<int>(problem->IOParams(), "OUTPUT_BIN")));

  Teuchos::RCP<IO::DiscretizationWriter> output = optidis->Writer();
  output->SetOutput(optioutput);
  output->WriteMesh(0, 0.0);

  // -------------------------------------------------------------------
  // create instance of the optimization class (call the constructor)
  // -------------------------------------------------------------------
  optimizer_ = Teuchos::rcp(new TOPOPT::Optimizer(optidis, fluiddis, prbdyn, output));

  return;
}
