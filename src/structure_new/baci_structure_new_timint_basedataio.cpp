/*-----------------------------------------------------------*/
/*! \file

\brief Input/output data container for the structural (time) integration


\level 3

*/
/*-----------------------------------------------------------*/


#include "baci_structure_new_timint_basedataio.H"

#include "baci_io_control.H"
#include "baci_io_every_iteration_writer.H"
#include "baci_lib_globalproblem.H"
#include "baci_solver_nonlin_nox_aux.H"
#include "baci_solver_nonlin_nox_linesearch_generic.H"
#include "baci_solver_nonlin_nox_linesearch_prepostoperator.H"
#include "baci_structure_new_timint_basedataio_monitor_dbc.H"
#include "baci_structure_new_timint_basedataio_runtime_vtk_output.H"
#include "baci_structure_new_timint_basedataio_runtime_vtp_output.H"

#include <NOX_Solver_Generic.H>

namespace
{
  inline bool DetermineWriteOutput(int step, int offset, int write_every)
  {
    return (step + offset) % write_every == 0;
  }
}  // namespace

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::BaseDataIO::BaseDataIO()
    : isinit_(false),
      issetup_(false),
      output_(Teuchos::null),
      writer_every_iter_(Teuchos::null),
      params_runtime_vtk_output_(Teuchos::null),
      params_runtime_vtp_output_(Teuchos::null),
      params_monitor_dbc_(Teuchos::null),
      energyfile_(Teuchos::null),
      errfile_(nullptr),
      gmsh_out_(false),
      printlogo_(false),
      printerrfile_(false),
      printiter_(false),
      outputeveryiter_(false),
      writesurfactant_(false),
      writestate_(false),
      writevelacc_(false),
      writejac2matlab_(false),
      firstoutputofrun_(false),
      printscreen_(-1),
      outputcounter_(-1),
      writerestartevery_(-1),
      writeresultsevery_(-1),
      writeenergyevery_(-1),
      writestress_(INPAR::STR::stress_none),
      writecouplstress_(INPAR::STR::stress_none),
      writestrain_(INPAR::STR::strain_none),
      writeplstrain_(INPAR::STR::strain_none),
      writeoptquantity_(INPAR::STR::optquantity_none),
      conditionnumbertype_(INPAR::STR::ConditionNumber::none)
{
  // empty constructor
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataIO::Init(const Teuchos::ParameterList& ioparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    Teuchos::RCP<IO::DiscretizationWriter> output)
{
  // We have to call Setup() after Init()
  issetup_ = false;

  // ---------------------------------------------------------------------------
  // initialize the printing and output parameters
  // ---------------------------------------------------------------------------
  {
    output_ = output;
    printscreen_ = ioparams.get<int>("STDOUTEVRY");
    printlogo_ = printscreen_ > 0;
    errfile_ = xparams.get<FILE*>("err file");
    gmsh_out_ = (bool)DRT::INPUT::IntegralValue<int>(ioparams, "OUTPUT_GMSH");
    printerrfile_ = errfile_ != nullptr;
    printiter_ = true;
    p_io_every_iteration_ =
        Teuchos::rcp(new Teuchos::ParameterList(ioparams.sublist("EVERY ITERATION")));
    outputeveryiter_ = DRT::INPUT::IntegralValue<bool>(*p_io_every_iteration_, "OUTPUT_EVERY_ITER");
    writerestartevery_ = sdynparams.get<int>("RESTARTEVRY");
    writetimestepoffset_ = sdynparams.get<int>("OUTPUT_STEP_OFFSET");
    writestate_ = (bool)DRT::INPUT::IntegralValue<int>(ioparams, "STRUCT_DISP");
    writevelacc_ = (bool)DRT::INPUT::IntegralValue<int>(ioparams, "STRUCT_VEL_ACC");
    writejac2matlab_ = (bool)DRT::INPUT::IntegralValue<int>(ioparams, "STRUCT_JACOBIAN_MATLAB");
    conditionnumbertype_ =
        Teuchos::getIntegralValue<INPAR::STR::ConditionNumber>(ioparams, "STRUCT_CONDITION_NUMBER");
    firstoutputofrun_ = true;
    writeresultsevery_ = sdynparams.get<int>("RESULTSEVRY");
    writecurrentelevolume_ =
        (bool)DRT::INPUT::IntegralValue<int>(ioparams, "STRUCT_CURRENT_VOLUME");
    writestress_ = DRT::INPUT::IntegralValue<INPAR::STR::StressType>(ioparams, "STRUCT_STRESS");
    writecouplstress_ =
        DRT::INPUT::IntegralValue<INPAR::STR::StressType>(ioparams, "STRUCT_COUPLING_STRESS");
    writestrain_ = DRT::INPUT::IntegralValue<INPAR::STR::StrainType>(ioparams, "STRUCT_STRAIN");
    writeplstrain_ =
        DRT::INPUT::IntegralValue<INPAR::STR::StrainType>(ioparams, "STRUCT_PLASTIC_STRAIN");
    writeenergyevery_ = sdynparams.get<int>("RESEVRYERGY");
    writesurfactant_ = (bool)DRT::INPUT::IntegralValue<int>(ioparams, "STRUCT_SURFACTANT");
    writeoptquantity_ = DRT::INPUT::IntegralValue<INPAR::STR::OptQuantityType>(
        ioparams, "STRUCT_OPTIONAL_QUANTITY");

    // build params container for monitoring reaction forces
    params_monitor_dbc_ = Teuchos::rcp(new ParamsMonitorDBC());
    params_monitor_dbc_->Init(ioparams.sublist("MONITOR STRUCTURE DBC"));
    params_monitor_dbc_->Setup();

    // check whether VTK output at runtime is desired
    if (ioparams.sublist("RUNTIME VTK OUTPUT").get<int>("INTERVAL_STEPS") != -1)
    {
      params_runtime_vtk_output_ = Teuchos::rcp(new ParamsRuntimeVtkOutput());

      params_runtime_vtk_output_->Init(ioparams.sublist("RUNTIME VTK OUTPUT"));
      params_runtime_vtk_output_->Setup();
    }

    // check whether VTP output at runtime is desired
    if (ioparams.sublist("RUNTIME VTP OUTPUT STRUCTURE").get<int>("INTERVAL_STEPS") != -1)
    {
      params_runtime_vtp_output_ = Teuchos::rcp(new ParamsRuntimeVtpOutput());

      params_runtime_vtp_output_->Init(ioparams.sublist("RUNTIME VTP OUTPUT STRUCTURE"));
      params_runtime_vtp_output_->Setup();
    }
  }

  isinit_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataIO::Setup()
{
  // safety check
  if (!IsInit()) dserror("Init() has not been called, yet!");

  if (outputeveryiter_) writer_every_iter_ = Teuchos::rcp(new IO::EveryIterationWriter());

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataIO::CheckInitSetup() const
{
  if (!IsInit() or !IsSetup()) dserror("Call Init() and Setup() first!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataIO::InitSetupEveryIterationWriter(
    IO::EveryIterationWriterInterface* interface, Teuchos::ParameterList& p_nox)
{
  if (not outputeveryiter_) return;

  writer_every_iter_->Init(output_.get(), interface, *p_io_every_iteration_);
  writer_every_iter_->Setup();

  // insert the every_iter output writer as ppo for the solver object
  Teuchos::ParameterList& p_sol_opt = p_nox.sublist("Solver Options");

  Teuchos::RCP<NOX::Observer> prepost_solver_ptr = Teuchos::rcp(
      new NOX::NLN::Solver::PrePostOp::TIMINT::WriteOutputEveryIteration(*writer_every_iter_));

  NOX::NLN::AUX::AddToPrePostOpVector(p_sol_opt, prepost_solver_ptr);

  // insert the every_iter output writer as ppo for the linesearch object
  Teuchos::ParameterList& p_linesearch = p_nox.sublist("Line Search");

  // Get the current map. If there is no map, return a new empty one. (reference)
  NOX::NLN::LineSearch::PrePostOperator::map& prepostls_map =
      NOX::NLN::LineSearch::PrePostOperator::GetMutableMap(p_linesearch);

  // insert/replace the old pointer in the map
  prepostls_map[NOX::NLN::LineSearch::prepost_output_every_iter] =
      Teuchos::rcp_dynamic_cast<NOX::NLN::Abstract::PrePostOperator>(prepost_solver_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::TIMINT::BaseDataIO::SetupEnergyOutputFile()
{
  if (energyfile_.is_null())
  {
    std::string energy_file_name =
        DRT::Problem::Instance()->OutputControlFile()->FileName() + "_energy.csv";

    energyfile_ = Teuchos::rcp(new std::ofstream(energy_file_name.c_str()));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::TIMINT::BaseDataIO::WriteResultsForThisStep(const int step) const
{
  if (step < 0) dserror("The variable step is not allowed to be negative.");
  return IsWriteResultsEnabled() and
         DetermineWriteOutput(step, GetWriteTimestepOffset(), GetWriteResultsEveryNStep());
}

bool STR::TIMINT::BaseDataIO::IsWriteResultsEnabled() const
{
  return GetWriteResultsEveryNStep() > 0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::TIMINT::BaseDataIO::WriteRuntimeVtkResultsForThisStep(const int step) const
{
  if (step < 0) dserror("The variable step is not allowed to be negative.");
  return (IsRuntimeVtkOutputEnabled() &&
          DetermineWriteOutput(step, GetRuntimeVtkOutputParams()->OutputStepOffset(),
              GetRuntimeVtkOutputParams()->OutputIntervalInSteps()));
}

bool STR::TIMINT::BaseDataIO::IsRuntimeVtkOutputEnabled() const
{
  return GetRuntimeVtkOutputParams() != Teuchos::null;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::TIMINT::BaseDataIO::WriteRuntimeVtpResultsForThisStep(const int step) const
{
  if (step < 0) dserror("The variable step is not allowed to be negative.");
  return (GetRuntimeVtpOutputParams() != Teuchos::null &&
          DetermineWriteOutput(step, GetRuntimeVtkOutputParams()->OutputStepOffset(),
              GetRuntimeVtkOutputParams()->OutputIntervalInSteps()));
}


bool STR::TIMINT::BaseDataIO::ShouldWriteRestartForStep(const int step) const
{
  return GetWriteRestartEveryNStep() &&
         DetermineWriteOutput(step, GetWriteTimestepOffset(), GetWriteRestartEveryNStep()) &&
         step != 0;
}


bool STR::TIMINT::BaseDataIO::ShouldWriteReactionForcesForThisStep(const int step) const
{
  return GetMonitorDBCParams()->OutputIntervalInSteps() > 0 &&
         DetermineWriteOutput(
             step, GetWriteTimestepOffset(), GetMonitorDBCParams()->OutputIntervalInSteps());
}


bool STR::TIMINT::BaseDataIO::ShouldWriteStressStrainForThisStep(const int step) const
{
  return WriteResultsForThisStep(step) &&
         ((GetStressOutputType() != INPAR::STR::stress_none) ||
             (GetCouplingStressOutputType() != INPAR::STR::stress_none) ||
             (GetStrainOutputType() != INPAR::STR::strain_none) ||
             (GetPlasticStrainOutputType() != INPAR::STR::strain_none));
}

bool STR::TIMINT::BaseDataIO::ShouldWriteEnergyForThisStep(const int step) const
{
  return GetWriteEnergyEveryNStep() > 0 &&
         DetermineWriteOutput(step, GetWriteTimestepOffset(), GetWriteEnergyEveryNStep());
}

int STR::TIMINT::BaseDataIO::GetLastWrittenResults() const { return lastwrittenresultsstep_; }

void STR::TIMINT::BaseDataIO::SetLastWrittenResults(const int step)
{
  lastwrittenresultsstep_ = step;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::Solver::PrePostOp::TIMINT::WriteOutputEveryIteration::WriteOutputEveryIteration(
    IO::EveryIterationWriter& every_iter_writer)
    : every_iter_writer_(every_iter_writer)
{
  /* empty */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::Solver::PrePostOp::TIMINT::WriteOutputEveryIteration::runPreSolve(
    const NOX::Solver::Generic& solver)
{
  every_iter_writer_.InitNewtonIteration();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::Solver::PrePostOp::TIMINT::WriteOutputEveryIteration::runPostIterate(
    const NOX::Solver::Generic& solver)
{
  const int newton_iteration = solver.getNumIterations();
  every_iter_writer_.AddNewtonIteration(newton_iteration);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::Solver::PrePostOp::TIMINT::WriteOutputEveryIteration::runPreModifyStepLength(
    const NOX::Solver::Generic& solver, const NOX::LineSearch::Generic& linesearch)
{
  const int newton_iteration = solver.getNumIterations();
  const int ls_iteration =
      dynamic_cast<const NOX::NLN::LineSearch::Generic&>(linesearch).GetNumIterations();
  every_iter_writer_.AddLineSearchIteration(newton_iteration, ls_iteration);
}
