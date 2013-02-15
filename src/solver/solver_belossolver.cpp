/*
 * solver_belossolver.cpp
 *
 *  Created on: Jul 5, 2011
 *      Author: wiesner
 */

#ifdef HAVE_MueLu
#ifdef HAVE_EXPERIMENTAL_MueLu

#include <MueLu_ConfigDefs.hpp>

#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>

#include <MueLu.hpp>
#include <MueLu_FactoryBase.hpp>
#include <MueLu_PermutationFactory.hpp>
#include <MueLu_SmootherPrototype.hpp>
#include <MueLu_SmootherFactory.hpp>
#include <MueLu_DirectSolver.hpp>    // remove me
#include <MueLu_HierarchyHelpers.hpp>
#include <MueLu_VerboseObject.hpp>
#endif // HAVE_EXPERIMENTAL_MueLu
#endif // HAVE_MueLu

// Belos headers
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosEpetraAdapter.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosBlockGmresSolMgr.hpp"

// BACI headers
#include "solver_belossolver.H"
#include "solver_pointpreconditioner.H"
#include "solver_blockpreconditioners.H"
#include "solver_krylovprojectionpreconditioner.H"
#include "solver_ifpackpreconditioner.H"
#include "solver_mlpreconditioner.H"

#include "../linalg/linalg_solver.H"

// Read a parameter value from a paraeter list and copy it into a new parameter list (with another parameter name)
#define LINALG_COPY_PARAM(paramList, paramStr, varType, defaultValue, outParamList, outParamStr) \
  if (paramList.isParameter(paramStr))                                  \
    outParamList.set<varType>(outParamStr, paramList.get<varType>(paramStr)); \
  else outParamList.set<varType>(outParamStr, defaultValue);            \

void LINALG::Solver::BuildBelosSolver(const Epetra_Comm & comm, Teuchos::ParameterList & params, FILE * outfile)
{
  solver_ = Teuchos::rcp( new LINALG::SOLVER::BelosSolver( comm, params, outfile));
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
LINALG::SOLVER::BelosSolver::BelosSolver( const Epetra_Comm & comm,
                                            Teuchos::ParameterList & params,
                                            FILE * outfile )
 : KrylovSolver(comm,params,outfile)
{
  ncall_ = 0;
  preconditioner_ = Teuchos::null;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
LINALG::SOLVER::BelosSolver::~BelosSolver()
{
  preconditioner_ = Teuchos::null;
  A_ = Teuchos::null;
  x_ = Teuchos::null;
  b_ = Teuchos::null;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void LINALG::SOLVER::BelosSolver::Setup(  Teuchos::RCP<Epetra_Operator>     matrix            ,
                                          Teuchos::RCP<Epetra_MultiVector>       x            ,
                                          Teuchos::RCP<Epetra_MultiVector>       b            ,
                                          bool                             refactor           ,
                                          bool                             reset              ,
                                          Teuchos::RCP<Epetra_MultiVector>  weighted_basis_mean,
                                          Teuchos::RCP<Epetra_MultiVector>  kernel_c           ,
                                          bool                             project)
{
  // see whether operator is a Epetra_CrsMatrix
  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>( matrix );

  if (!Params().isSublist("Belos Parameters"))
    dserror("Do not have belos parameter list");
  Teuchos::ParameterList& belist = Params().sublist("Belos Parameters");

#ifdef HAVE_MueLu
#ifdef HAVE_EXPERIMENTAL_MueLu
  bAllowPermutation_ = belist.get<bool>("allow permutation",false);
#endif
#endif

  int reuse = belist.get("reuse",0); // TODO: fix me!
  bool create = not Ncall() or not reuse or (Ncall() % reuse ) == 0;
  if (create)
  {
    ncall_ = 0;
    CreatePreconditioner(belist, A!=Teuchos::null, weighted_basis_mean, kernel_c, project );
  }

  // feed preconditioner with more information about linear system using
  // the "Linear System properties" sublist in the preconditioner's
  // paramter list
  if (Preconditioner() != NULL) {

    const std::string precondParamListName = Preconditioner()->getParameterListName();
    if(Params().isSublist(precondParamListName)) {
      Teuchos::ParameterList & precondParams = Params().sublist(precondParamListName);
      Teuchos::ParameterList & linSystemProps = precondParams.sublist("Linear System properties");

      LINALG_COPY_PARAM(Params().sublist("Belos Parameters").sublist("Linear System properties"),
          "contact slaveDofMap", Teuchos::RCP<Epetra_Map>, Teuchos::null,
          linSystemProps, "contact slaveDofMap");
      LINALG_COPY_PARAM(Params().sublist("Belos Parameters").sublist("Linear System properties"),
          "contact masterDofMap", Teuchos::RCP<Epetra_Map>, Teuchos::null,
          linSystemProps, "contact masterDofMap");
      LINALG_COPY_PARAM(Params().sublist("Belos Parameters").sublist("Linear System properties"),
          "contact innerDofMap", Teuchos::RCP<Epetra_Map>, Teuchos::null,
          linSystemProps, "contact innerDofMap");
      LINALG_COPY_PARAM(Params().sublist("Belos Parameters").sublist("Linear System properties"),
          "contact activeDofMap", Teuchos::RCP<Epetra_Map>, Teuchos::null,
          linSystemProps, "contact activeDofMap");

    }
  }

  ////////////////////////////////////// permutation stuff
#ifdef HAVE_MueLu
#ifdef HAVE_EXPERIMENTAL_MueLu
  if(bAllowPermutation_) {
    // extract (user-given) additional information about linear system from
    // "Aztec Parameters" -> "Linear System properties"
    Teuchos::RCP<Epetra_Map> epSlaveDofMap = ExtractPermutationMap("Belos Parameters", "contact slaveDofMap");

    // build permutation operators
    // permP, permQT and A = permQ^T A permP
    // all variables and information is stored in data_
    // note: we only allow permutations for rows in epSlaveDofMap
    //       the idea is not to disturb the matrix in regions which are
    //       known to work perfectly (no contact)
    BuildPermutationOperator(A,epSlaveDofMap);

    // decide whether to permute linear system or not.
    // set all information corresponding to the decision.
    bPermuteLinearSystem_ = DecideAboutPermutation(A);
  }

  // set linear system
  if(bAllowPermutation_ && bPermuteLinearSystem_) {
    // set
    // b_ = permP * b;
    // A_ = permQ^T * A * permP
    PermuteLinearSystem(A,b);

    // calculate (permQT)^T * b_f where b_f is the fine level null space (multi)vector
    //PermuteNullSpace(A);  // TODO think about this
    // do not permute null space to preserve pattern of null space for transfer operators
    // important e.g. for one pt aggregates?
  } else {
#endif // HAVE_EXPERIMENTAL_MueLu
#endif // HAVE_MueLu
    b_ = b;
    A_ = matrix; // we cannot use A here, since it could be Teuchos::null (for blocked operators);
#ifdef HAVE_MueLu
#ifdef HAVE_EXPERIMENTAL_MueLu
  }
#endif
#endif
  x_ = x;


  // call setup of preconditioner
  preconditioner_->Setup(create,&*A_, &*x_,&*b_);

}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void LINALG::SOLVER::BelosSolver::Solve()
{
  Teuchos::ParameterList& belist = Params().sublist("Belos Parameters");

  typedef double                            ST;
  typedef Teuchos::ScalarTraits<ST>        SCT;
  typedef SCT::magnitudeType                MT;
  typedef Epetra_MultiVector                MV;
  typedef Epetra_Operator                   OP;
  typedef Belos::MultiVecTraits<ST,MV>     MVT;
  typedef Belos::OperatorTraits<ST,MV,OP>  OPT;

  // build Belos linear problem
  Teuchos::RCP<Belos::LinearProblem<double, MV, OP> > problem = Teuchos::rcp(new Belos::LinearProblem<double,MV,OP>(A_, x_, b_) );
  // TODO support for left preconditioner?
  if (preconditioner_ != Teuchos::null)
  {
    // prepare preconditioner in preconditioner_->PrecOperator() for Belos
    Teuchos::RCP<Belos::EpetraPrecOp> belosPrec = Teuchos::rcp(new Belos::EpetraPrecOp(Teuchos::rcp(preconditioner_->PrecOperator(),false)));
    problem->setRightPrec(belosPrec);
  }
  bool set = problem->setProblem();
  if (set == false)
  {
    std::cout << std::endl << "ERROR: Belos::LinearProblem failed to set up correctly!" << std::endl;
  }

  // create iterative solver manager
  Teuchos::RCP<Belos::SolverManager<double,MV,OP> > newSolver;
  std::string solverType = belist.get<std::string>("Solver Type");
  if(solverType=="GMRES")
     newSolver = Teuchos::rcp(new Belos::BlockGmresSolMgr<double,MV,OP>(problem,Teuchos::rcp(&belist,false)));
  else if (solverType=="CG")
     newSolver = Teuchos::rcp(new Belos::BlockCGSolMgr<double,MV,OP>(problem,Teuchos::rcp(&belist,false)));
  else
    dserror("unknown solver type for Belos");

  //
  // Perform solve
  //
  Belos::ReturnType ret = newSolver->solve();

  // TODO: check me -> access solution x from linear problem???
  if(preconditioner_!=Teuchos::null)
    preconditioner_->Finish( &*A_, &*x_, &*b_ );

  if (ret!=Belos::Converged)
  {
    std::cout << std::endl << "WARNING: Belos did not converge!" << std::endl;
  }

#ifdef HAVE_MueLu
#ifdef HAVE_EXPERIMENTAL_MueLu
  GlobalOrdinal rowperm  = 0;
  GlobalOrdinal colperm  = 0;
  GlobalOrdinal lrowperm = 0;
  GlobalOrdinal lcolperm = 0;
  int nonDiagDomRows     = 0;
  int nonPermutedZeros   = 0;
  int PermutedZeros      = 0;
  int PermutedNearZeros  = 0;
  int NonPermutedNearZeros=0;


  if(bAllowPermutation_ && bPermuteLinearSystem_) {
    // repermutate solution vector
    this->ReTransformSolution();
    rowperm = data_->Get<GlobalOrdinal>("#RowPermutations", PermFact_.get());
    colperm = data_->Get<GlobalOrdinal>("#ColPermutations", PermFact_.get());
    lrowperm = data_->Get<GlobalOrdinal>("#WideRangeRowPermutations", PermFact_.get());
    lcolperm = data_->Get<GlobalOrdinal>("#WideRangeColPermutations", PermFact_.get());
  }
  if(data_->IsAvailable("nonDiagDomRows"))             nonDiagDomRows = data_->Get<int>("nonDiagDomRows");
  if(data_->IsAvailable("NonPermutedZerosOnDiagonal")) nonPermutedZeros = data_->Get<int>("NonPermutedZerosOnDiagonal");
  if(data_->IsAvailable("PermutedZerosOnDiagonal"))    PermutedZeros    = data_->Get<int>("PermutedZerosOnDiagonal");
  if(data_->IsAvailable("PermutedNearZeros"))          PermutedNearZeros = data_->Get<int>("PermutedNearZeros");
  if(data_->IsAvailable("NonPermutedNearZeros"))       NonPermutedNearZeros = data_->Get<int>("NonPermutedNearZeros");

  // print some output if desired
  if (comm_.MyPID()==0 && outfile_)
  {
    fprintf(outfile_,"Belos: unknowns/iterations/time/rowpermutations/colpermutations/lrowperm/lcolperm/nonDiagDomRows %d  %d  %f %d %d %d %d %d NonPermutedZeros/PermutedZeros %d %d bPermuted %d nonPermNearZeros/PermNearZeros %d %d\n",
            A_->OperatorRangeMap().NumGlobalElements(),(int)newSolver->getNumIters(),-1.0,rowperm,colperm,lrowperm,lcolperm,nonDiagDomRows,nonPermutedZeros,PermutedZeros,bPermuteLinearSystem_ ? 1 : 0,NonPermutedNearZeros,PermutedNearZeros);
    fflush(outfile_);
  }
#endif // HAVE_EXPERIMENTAL_MueLu
#endif // HAVE_MueLu

  ncall_ += 1; // increment counter of solver calls
}
