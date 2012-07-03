/*----------------------------------------------------------------------*/
/*!
 * \file ale.cpp
 *
\brief ALE base implementation

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/



#include "ale.H"
#include "ale_lin.H"
#include "ale_laplace.H"
#include "ale_springs.H"
#include "ale_springs_fixed_ref.H"
#include "ale_resulttest.H"
#include "ale_utils_mapextractor.H"

// further includes for AleBaseAlgorithm:
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_sparseoperator.H"
#include "../linalg/linalg_sparsematrix.H"
#include "../linalg/linalg_blocksparsematrix.H"
#include "../linalg/linalg_solver.H"
#include "../drt_io/io.H"
#include "../drt_io/io_control.H"
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include "../drt_inpar/inpar_ale.H"
#include "../drt_inpar/inpar_fsi.H"
#include "../drt_fluid/drt_periodicbc.H"

using namespace std;
using namespace Teuchos;


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ALE::Ale::Ale(RCP<DRT::Discretization> actdis,
              Teuchos::RCP<LINALG::Solver> solver,
              Teuchos::RCP<ParameterList> params,
              Teuchos::RCP<IO::DiscretizationWriter> output,
              bool dirichletcond)
  : discret_(actdis),
    solver_ (solver),
    params_ (params),
    output_ (output),
    step_(0),
    time_(0.0),
    uprestart_(params->get("write restart every", -1)),
    sysmat_(null)
{
  numstep_ = params_->get<int>("numstep");
  maxtime_ = params_->get<double>("maxtime");
  dt_      = params_->get<double>("dt");

  const Epetra_Map* dofrowmap = discret_->DofRowMap();

  dispn_          = LINALG::CreateVector(*dofrowmap,true);
  dispnp_         = LINALG::CreateVector(*dofrowmap,true);
  residual_       = LINALG::CreateVector(*dofrowmap,true);

  interface_ = Teuchos::rcp(new ALE::UTILS::MapExtractor);
  interface_->Setup(*actdis);

  SetupDBCMapEx(dirichletcond);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> ALE::Ale::DofRowMap()
{
  return Teuchos::rcp(discret_->DofRowMap(),false);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix> ALE::Ale::SystemMatrix()
{
  return Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(sysmat_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::BlockSparseMatrixBase> ALE::Ale::BlockSystemMatrix()
{
  return Teuchos::rcp_dynamic_cast<LINALG::BlockSparseMatrixBase>(sysmat_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ALE::Ale::Integrate()
{
  while (step_ < numstep_-1 and time_ <= maxtime_)
  {
    PrepareTimeStep();
    Solve();
    Update();
    Output();
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ALE::Ale::ReadRestart(int step)
{
  IO::DiscretizationReader reader(discret_,step);
  time_ = reader.ReadDouble("time");
  step_ = reader.ReadInt("step");

  reader.ReadVector(dispnp_, "dispnp");
  reader.ReadVector(dispn_,  "dispn");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ALE::Ale::PrepareTimeStep()
{
  step_ += 1;
  time_ += dt_;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ALE::Ale::SetupDBCMapEx(bool dirichletcond)
{
  // set fixed nodes (conditions != 0 are not supported right now)
  ParameterList eleparams;
  eleparams.set("total time", time_);
  eleparams.set("delta time", dt_);
  dbcmaps_ = Teuchos::rcp(new LINALG::MapExtractor());
  discret_->EvaluateDirichlet(eleparams,dispnp_,null,null,null,dbcmaps_);

  if (dirichletcond)
  {
    // for partitioned FSI the interface becomes a Dirichlet boundary
    // also for structural Lagrangian simulations with contact and wear
    // followed by an Eulerian step to take wear into account, the interface
    // becomes a dirichlet
    std::vector<Teuchos::RCP<const Epetra_Map> > condmaps;
    condmaps.push_back(interface_->FSICondMap());
    condmaps.push_back(interface_->AleWearCondMap());
    condmaps.push_back(dbcmaps_->CondMap());
    Teuchos::RCP<Epetra_Map> condmerged = LINALG::MultiMapExtractor::MergeMaps(condmaps);
    *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), condmerged);
  }

  if (dirichletcond and interface_->FSCondRelevant())
  {
    // for partitioned solves the free surface becomes a Dirichlet boundary
    std::vector<Teuchos::RCP<const Epetra_Map> > condmaps;
    condmaps.push_back(interface_->FSCondMap());
    condmaps.push_back(dbcmaps_->CondMap());
    Teuchos::RCP<Epetra_Map> condmerged = LINALG::MultiMapExtractor::MergeMaps(condmaps);
    *dbcmaps_ = LINALG::MapExtractor(*(discret_->DofRowMap()), condmerged);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<DRT::ResultTest> ALE::Ale::CreateFieldTest()
{
  return Teuchos::rcp(new ALE::AleResultTest(*this));
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ALE::Ale::ApplyFreeSurfaceDisplacements(Teuchos::RCP<Epetra_Vector> fsdisp)
{
  interface_->InsertFSCondVector(fsdisp,dispnp_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ALE::Ale::ApplyInterfaceDisplacements(Teuchos::RCP<Epetra_Vector> idisp)
{
  // applying interface displacements
  if(DRT::Problem::Instance()->ProblemName()!="structure_ale")
    interface_->InsertFSICondVector(idisp,dispnp_);
  else
    interface_->InsertAleWearCondVector(idisp,dispnp_);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ALE::Ale::ExtractDisplacement() const
{
  // We know that the ale dofs are coupled with their original map. So
  // we just return them here.
  return dispnp_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ALE::AleBaseAlgorithm::AleBaseAlgorithm(const Teuchos::ParameterList& prbdyn, int disnum)
{
  SetupAle(prbdyn,disnum);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ALE::AleBaseAlgorithm::~AleBaseAlgorithm()
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ALE::AleBaseAlgorithm::SetupAle(const Teuchos::ParameterList& prbdyn, int disnum)
{
  Teuchos::RCP<Teuchos::Time> t = Teuchos::TimeMonitor::getNewTimer("ALE::AleBaseAlgorithm::SetupAle");
  Teuchos::TimeMonitor monitor(*t);

  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  RCP<DRT::Discretization> actdis = null;
  if (disnum > 0) dserror("Disnum > 0");
  actdis = DRT::Problem::Instance()->GetDis("ale");

  // -------------------------------------------------------------------
  // set degrees of freedom in the discretization
  // -------------------------------------------------------------------
  if (!actdis->Filled()) actdis->FillComplete();

  // -------------------------------------------------------------------
  // connect degrees of freedom for coupled nodes
  // -------------------------------------------------------------------
  PeriodicBoundaryConditions pbc(actdis);
  pbc.UpdateDofsForPeriodicBoundaryConditions();

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  RCP<IO::DiscretizationWriter> output =
    rcp(new IO::DiscretizationWriter(actdis));
  output->WriteMesh(0,0.0);

  // -------------------------------------------------------------------
  // set some pointers and variables
  // -------------------------------------------------------------------
  const Teuchos::ParameterList& adyn     = DRT::Problem::Instance()->AleDynamicParams();

  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  // get the solver number used for ALE problems
    const int linsolvernumber = adyn.get<int>("LINEAR_SOLVER");
    // check if the TSI solver has a valid solver number
    if (linsolvernumber == (-1))
      dserror("no linear solver defined for ALE problems. Please set LINEAR_SOLVER in ALE DYNAMIC to a valid number!");

  RCP<LINALG::Solver> solver =
    rcp(new LINALG::Solver(DRT::Problem::Instance()->SolverParams(linsolvernumber),
                           actdis->Comm(),
                           DRT::Problem::Instance()->ErrorFile()->Handle()));
  actdis->ComputeNullSpaceIfNecessary(solver->Params());

  RCP<ParameterList> params = rcp(new ParameterList());
  params->set<int>("numstep",    prbdyn.get<int>("NUMSTEP"));
  params->set<double>("maxtime", prbdyn.get<double>("MAXTIME"));
  params->set<double>("dt",      prbdyn.get<double>("TIMESTEP"));

  // ----------------------------------------------- restart and output
  // restart
  params->set<int>("write restart every", prbdyn.get<int>("RESTARTEVRY"));

  params->set<int>("ALE_TYPE",DRT::INPUT::IntegralValue<int>(adyn,"ALE_TYPE"));


  bool dirichletcond = true;
  // what's the current problem type?
  PROBLEM_TYP probtype = DRT::Problem::Instance()->ProblemType();
  if (probtype == prb_fsi or
      probtype == prb_fsi_lung or
      probtype == prb_gas_fsi or
      probtype == prb_thermo_fsi or
      probtype == prb_biofilm_fsi or
      probtype == prb_fluid_fluid_fsi)
  {
    // FSI input parameters
    const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
    int coupling = DRT::INPUT::IntegralValue<int>(fsidyn,"COUPALGO");
    if (coupling == fsi_iter_monolithicfluidsplit or
        coupling == fsi_iter_monolithicstructuresplit or
        coupling == fsi_iter_constr_monolithicfluidsplit or
        coupling == fsi_iter_constr_monolithicstructuresplit or
        coupling == fsi_iter_lung_monolithicfluidsplit or
        coupling == fsi_iter_lung_monolithicstructuresplit or
        coupling == fsi_iter_mortar_monolithicstructuresplit or
        coupling == fsi_iter_mortar_monolithicfluidsplit or
        coupling == fsi_iter_fluidfluid_monolithicstructuresplit)
    {
        dirichletcond = false;
    }
  }

  if (probtype == prb_freesurf)
  {
    // FSI input parameters
    const Teuchos::ParameterList& fsidyn = DRT::Problem::Instance()->FSIDynamicParams();
    int coupling = DRT::INPUT::IntegralValue<int>(fsidyn,"COUPALGO");
    if (coupling == fsi_iter_monolithicfluidsplit or
         coupling == fsi_iter_monolithicstructuresplit or
         coupling == fsi_iter_constr_monolithicfluidsplit or
         coupling == fsi_iter_constr_monolithicstructuresplit or
         coupling == fsi_iter_lung_monolithicfluidsplit or
         coupling == fsi_iter_lung_monolithicstructuresplit or
         coupling == fsi_iter_mortar_monolithicstructuresplit or
         coupling == fsi_iter_mortar_monolithicfluidsplit)
    {
      dirichletcond = false;
    }
  }

  int aletype = DRT::INPUT::IntegralValue<int>(adyn,"ALE_TYPE");
  if (aletype==INPAR::ALE::classic_lin)
    ale_ = rcp(new AleLinear(actdis, solver, params, output, false, dirichletcond));
  else if (aletype==INPAR::ALE::incr_lin)
    ale_ = rcp(new AleLinear(actdis, solver, params, output, true , dirichletcond));
  else if (aletype==INPAR::ALE::laplace)
    ale_ = rcp(new AleLaplace(actdis, solver, params, output, true, dirichletcond));
  else if (aletype==INPAR::ALE::springs)
    ale_ = rcp(new AleSprings(actdis, solver, params, output, dirichletcond));
  else if (aletype==INPAR::ALE::springs_fixed_ref)
    ale_ = rcp(new AleSpringsFixedRef(actdis, solver, params, output, true, dirichletcond));
  else
    dserror("ale type '%s' unsupported",adyn.get<std::string>("ALE_TYPE").c_str());
}


