/*!----------------------------------------------------------------------
\file linalg_solver.cpp
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#define WRITEOUTSTATISTICS
#ifdef WRITEOUTSTATISTICS
#include "Teuchos_Time.hpp"
#endif

#ifdef PARALLEL
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "../drt_inpar/inpar_solver.H"
#include "linalg_solver.H"
#include "linalg_mlapi_operator.H"
#include "simpler_operator.H"
#include "linalg_downwindmatrix.H"
#include "linalg_sparsematrix.H"
#include "standardtypes_cpp.H"
#include "linalg_krylov_projector.H"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <EpetraExt_Transpose_RowMatrix.h>

#include "saddlepointpreconditioner.H"

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 02/07|
 *----------------------------------------------------------------------*/
LINALG::Solver::Solver(RefCountPtr<ParameterList> params,
                       const Epetra_Comm& comm, FILE* outfile) :
comm_(comm),
params_(params),
outfile_(outfile),
factored_(false),
ncall_(0)
{
  Setup();
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 03/08|
 *----------------------------------------------------------------------*/
LINALG::Solver::Solver(const Epetra_Comm& comm, FILE* outfile) :
comm_(comm),
params_(rcp(new ParameterList())),
outfile_(outfile),
factored_(false),
ncall_(0)
{
  // set the default solver
  Params().set("solver","klu");
  Params().set("symmetric",false);

  // set-up
  Setup();

  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                                  11/08|
 *----------------------------------------------------------------------*/
LINALG::Solver::Solver(const Teuchos::ParameterList& inparams,
                       const Epetra_Comm& comm,
                       FILE* outfile) :
comm_(comm),
params_(rcp(new ParameterList())),
outfile_(outfile),
factored_(false),
ncall_(0)
{
  // set solver parameters
  *params_ = TranslateSolverParameters(inparams);

  // set-up
  Setup();

  return;
}

/*----------------------------------------------------------------------*
 |  set-up of stuff common to all constructors                     11/08|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Setup()
{
  // create an empty linear problem
  lp_ = rcp(new Epetra_LinearProblem());

#ifdef PARALLEL
#ifdef SPOOLES_PACKAGE
  frontmtx_      =NULL;
  newA_          =NULL;
  newY_          =NULL;
  frontETree_    =NULL;
  mtxmanager_    =NULL;
  newToOldIV_    =NULL;
  oldToNewIV_    =NULL;
  ownersIV_      =NULL;
  vtxmapIV_      =NULL;
  ownedColumnsIV_=NULL;
  solvemap_      =NULL;
  symbfacIVL_    =NULL;
  graph_         =NULL;
  mtxY_          =NULL;
  mtxX_          =NULL;
  mtxA_          =NULL;
#endif
#endif

  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 02/07|
 *----------------------------------------------------------------------*/
LINALG::Solver::~Solver()
{
#ifdef PARALLEL
#ifdef SPOOLES_PACKAGE
  if (frontmtx_)       FrontMtx_free(frontmtx_);        frontmtx_      =NULL;
  if (newA_)           InpMtx_free(newA_);              newA_          =NULL;
  if (newY_)           DenseMtx_free(newY_);            newY_          =NULL;
  if (frontETree_)     ETree_free(frontETree_);         frontETree_    =NULL;
  if (mtxmanager_)     SubMtxManager_free(mtxmanager_); mtxmanager_    =NULL;
  if (newToOldIV_)     IV_free(newToOldIV_);            newToOldIV_    =NULL;
  if (oldToNewIV_)     IV_free(oldToNewIV_);            oldToNewIV_    =NULL;
  if (ownersIV_)       IV_free(ownersIV_);              ownersIV_      =NULL;
  if (vtxmapIV_)       IV_free(vtxmapIV_);              vtxmapIV_      =NULL;
  if (ownedColumnsIV_) IV_free(ownedColumnsIV_);        ownedColumnsIV_=NULL;
  if (solvemap_)       SolveMap_free(solvemap_);        solvemap_      =NULL;
  if (graph_)          Graph_free(graph_);              graph_         =NULL;
  if (mtxY_)           DenseMtx_free(mtxY_);            mtxY_          =NULL;
  if (mtxX_)           DenseMtx_free(mtxX_);            mtxX_          =NULL;
  if (mtxA_)           InpMtx_free(mtxA_);              mtxA_          =NULL;
  if (symbfacIVL_)     IVL_free(symbfacIVL_);           symbfacIVL_    =NULL;
#endif
#endif

  // destroy in the right order
  Reset();

  return;
}

/*----------------------------------------------------------------------*
 |  reset solver (public)                                    mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Reset()
{
  lp_       = rcp(new Epetra_LinearProblem());
  amesos_   = null;
  aztec_    = null;
  P_        = null;
  Pmatrix_  = null;
  A_        = null;
  Aplus_    = null;
  x_        = null;
  b_        = null;
  factored_ = false;
  ncall_    = 0;
#ifdef PARALLEL
#ifdef SPOOLES_PACKAGE
  if (frontmtx_)       FrontMtx_free(frontmtx_);        frontmtx_      =NULL;
  if (newA_)           InpMtx_free(newA_);              newA_          =NULL;
  if (newY_)           DenseMtx_free(newY_);            newY_          =NULL;
  if (frontETree_)     ETree_free(frontETree_);         frontETree_    =NULL;
  if (mtxmanager_)     SubMtxManager_free(mtxmanager_); mtxmanager_    =NULL;
  if (newToOldIV_)     IV_free(newToOldIV_);            newToOldIV_    =NULL;
  if (oldToNewIV_)     IV_free(oldToNewIV_);            oldToNewIV_    =NULL;
  if (ownersIV_)       IV_free(ownersIV_);              ownersIV_      =NULL;
  if (vtxmapIV_)       IV_free(vtxmapIV_);              vtxmapIV_      =NULL;
  if (ownedColumnsIV_) IV_free(ownedColumnsIV_);        ownedColumnsIV_=NULL;
  if (solvemap_)       SolveMap_free(solvemap_);        solvemap_      =NULL;
  if (graph_)          Graph_free(graph_);              graph_         =NULL;
  if (mtxY_)           DenseMtx_free(mtxY_);            mtxY_          =NULL;
  if (mtxX_)           DenseMtx_free(mtxX_);            mtxX_          =NULL;
  if (mtxA_)           InpMtx_free(mtxA_);              mtxA_          =NULL;
  if (symbfacIVL_)     IVL_free(symbfacIVL_);           symbfacIVL_    =NULL;
#endif
#endif
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 02/07|
 *----------------------------------------------------------------------*/
ostream& operator << (ostream& os, const LINALG::Solver& solver)
{
  solver.Print(os);
  return os;
}

/*----------------------------------------------------------------------*
 |  print solver (public)                                    mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Print(ostream& os) const
{
  if (Comm().MyPID()==0)
  {
    os << "============================LINALG::Solver Parameter List\n";
    os << *params_;
    os << "========================end LINALG::Solver Parameter List\n";
  }
  return;
}

/*----------------------------------------------------------------------*
 |  adapt tolerance (public)                                 mwgee 02/08|
 *----------------------------------------------------------------------*/
void LINALG::Solver::AdaptTolerance(const double desirednlnres,
                                    const double currentnlnres,
                                    const double better)
{
  if (!Params().isSublist("Aztec Parameters")) return;
  const int myrank = Comm().MyPID();
  ParameterList& azlist = Params().sublist("Aztec Parameters");
  int output   = azlist.get<int>("AZ_output",1);
  int convtest = azlist.get<int>("AZ_conv",AZ_noscaled);
  if (convtest != AZ_r0) dserror("Using convergence adaptivity: Use AZ_r0 in input file");
  bool havesavedvalue = azlist.isParameter("AZ_tol save");
  if (!havesavedvalue)
  {
    if (!azlist.isParameter("AZ_tol"))
    {
      cout << azlist;
      dserror("No Aztec tolerance in ParameterList");
    }
    azlist.set<double>("AZ_tol save",azlist.get<double>("AZ_tol",1.e-8));
  }
  double tol = azlist.get<double>("AZ_tol save",1.e-8);
  if (!myrank && output)
    printf("                --- Aztec input   relative tolerance %10.3E\n",tol);
  if (currentnlnres*tol < desirednlnres)
  {
    double tolnew = desirednlnres*better/currentnlnres;
    if (tolnew<tol) tolnew = tol;
    if (!myrank && output && tolnew > tol)
      printf("                *** Aztec adapted relative tolerance %10.3E\n",tolnew);
    azlist.set<double>("AZ_tol",tolnew);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  adapt tolerance (public)                                 mwgee 02/08|
 *----------------------------------------------------------------------*/
void LINALG::Solver::ResetTolerance()
{
  if (!Params().isSublist("Aztec Parameters")) return;
  ParameterList& azlist = Params().sublist("Aztec Parameters");
  bool havesavedvalue = azlist.isParameter("AZ_tol save");
  if (!havesavedvalue) return;
  azlist.set<double>("AZ_tol",azlist.get<double>("AZ_tol save",1.e-8));
  return;
}

/*----------------------------------------------------------------------*
 |  solve (public)                                           mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Solve(
  RefCountPtr<Epetra_Operator>     matrix             ,
  RefCountPtr<Epetra_Vector>       x                  ,
  RefCountPtr<Epetra_Vector>       b                  ,
  bool                             refactor           ,
  bool                             reset              ,
  RefCountPtr<Epetra_MultiVector>  weighted_basis_mean,
  RefCountPtr<Epetra_MultiVector>  kernel_c           ,
  bool                             project            )
{
  // reset data flags on demand
  if (reset)
  {
    Reset();
    refactor = true;
  }

  // is projection desired?
  RCP<LINALG::KrylovProjector> projector = Teuchos::null;
  if(project)
  {
    projector = rcp(new LINALG::KrylovProjector(project,weighted_basis_mean,kernel_c,matrix));
  }

  // set the data passed to the method
  if (refactor)
  {
    A_ = rcp(new LINALG::LinalgProjectedOperator::LinalgProjectedOperator(matrix,project,projector));
  }

  x_ = x;
  b_ = b;

  // set flag indicating that problem should be refactorized
  if (refactor) factored_ = false;

  // fill the linear problem
  lp_->SetRHS(b_.get());
  lp_->SetLHS(x_.get());
  if(project)
  {
    lp_->SetOperator(A_.get());
  }
  else
  {
    lp_->SetOperator(A_->UnprojectedOperator().get());
  }

  // decide what solver to use
  string solvertype = Params().get("solver","none");

  if("aztec"!=solvertype && project)
  {
    dserror("a projection of Krylov space basis vectors is possible only for aztec type iterative solvers\n");
  }

  if ("aztec"  ==solvertype)
  Solve_aztec(reset,project,projector);
  else if ("klu"    ==solvertype)
    Solve_klu(reset);
  else if ("umfpack"==solvertype)
    Solve_umfpack(reset);
#ifdef PARALLEL
  else if ("superlu"==solvertype)
    Solve_superlu(reset);
#endif
#ifdef PARALLEL
#ifdef SPOOLES_PACKAGE
  else if ("spooles"==solvertype)
    Solve_spooles(reset);
#endif
#endif
  else if ("lapack" ==solvertype)
    Solve_lapack(reset);
  else if ("none"   ==solvertype)
    dserror("Unknown type of solver");

  factored_ = true;
  ncall_++;

  return;
}

/*----------------------------------------------------------------------*
 |  solve (protected)                                        mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Solve_aztec(
  const bool                                  reset    ,
  const bool                                  project  ,
  const Teuchos::RCP<LINALG::KrylovProjector> projector
  )
{

#ifdef WRITEOUTSTATISTICS
  double dtimeprecondsetup = 0.;
  Epetra_Time ttt(Comm());       // time measurement for whole Solve_aztec routine
  Epetra_Time tttcreate(Comm()); // time measurement for creation of preconditioner
  ttt.ResetStartTime();
#endif

  if (!Params().isSublist("Aztec Parameters"))
    dserror("Do not have aztec parameter list");
  ParameterList& azlist = Params().sublist("Aztec Parameters");
  int azoutput = azlist.get<int>("AZ_output",0);

  // see whether unprojected Operator is a Epetra_CrsMatrix
  Epetra_CrsMatrix* A = dynamic_cast<Epetra_CrsMatrix*>((A_->UnprojectedOperator()).get());

  if (!A && project)
  {
    dserror("Projection out nullspaces is only possible for Epetra_CrsMatrix-type Operators\n");
  }

  // For singular systems, forcing (i.e. right hand side, residual etc)
  // must be orthogonal to matrix kernel. Make sure this is true.
  if(project)
  {
    projector->ApplyPT(*b_);
  }

  // decide whether we recreate preconditioners
  bool create = false;
  int  reuse  = azlist.get("reuse",0);
  if      (reset)            create = true;
  else if (!Ncall())         create = true;
  else if (!reuse)           create = true;
  else if (Ncall()%reuse==0) create = true;

  // Allocate an aztec solver with default parameters
  // We do this every time because reusing the solver object
  // does lead to crashes that are not understood
  {
    // create an aztec solver
    aztec_ = Teuchos::null;
    aztec_ = rcp(new AztecOO());
    aztec_->SetAztecDefaults();
    // tell aztec to which stream to write
    aztec_->SetOutputStream(std::cout);
    aztec_->SetErrorStream(std::cerr);
  }

  // decide whether we do what kind of scaling
  bool scaling_infnorm = false;
  bool scaling_symdiag = false;
  string scaling = azlist.get("scaling","none");
  if (scaling=="none");
  else if (scaling=="infnorm")
  {
    scaling_infnorm = true;
    scaling_symdiag = false;
  }
  else if (scaling=="symmetric")
  {
    scaling_infnorm = false;
    scaling_symdiag = true;
  }
  else dserror("Unknown type of scaling found in parameter list");

  if (!A)
  {
    scaling_infnorm = false;
    scaling_symdiag = false;
  }

  // do infnorm scaling
  RefCountPtr<Epetra_Vector> rowsum;
  RefCountPtr<Epetra_Vector> colsum;
  if (scaling_infnorm)
  {
    rowsum = rcp(new Epetra_Vector(A->RowMap(),false));
    colsum = rcp(new Epetra_Vector(A->RowMap(),false));
    A->InvRowSums(*rowsum);
    A->InvColSums(*colsum);
    lp_->LeftScale(*rowsum);
    lp_->RightScale(*colsum);
  }

  // do symmetric diagonal scaling
  RefCountPtr<Epetra_Vector> diag;
  if (scaling_symdiag)
  {
    Epetra_Vector invdiag(A->RowMap(),false);
    diag = rcp(new Epetra_Vector(A->RowMap(),false));
    A->ExtractDiagonalCopy(*diag);
    invdiag.Reciprocal(*diag);
    lp_->LeftScale(invdiag);
    lp_->RightScale(invdiag);
  }

  // get type of preconditioner and build either Ifpack or ML
  // if we have an ifpack parameter list, we do ifpack
  // if we have an ml parameter list we do ml
  // if we have a downwinding flag we downwind the linear problem
  bool   doifpack  = Params().isSublist("IFPACK Parameters");
  bool   doml      = Params().isSublist("ML Parameters");
  bool   dwind     = azlist.get<bool>("downwinding",false);
  bool   dosimpler = Params().isSublist("SIMPLER");
  bool   doamgbs   = Params().isSublist("AMGBS"); // TODO: check  AZPREC directly

  if (!A || dosimpler || doamgbs)
  {
    doifpack = false;
    doml     = false;
    dwind    = false; // we can do downwinding inside SIMPLER if desired
  }

  if (create && dwind)
  {
    double tau  = azlist.get<double>("downwinding tau",1.0);
    int    nv   = azlist.get<int>("downwinding nv",1);
    int    np   = azlist.get<int>("downwinding np",0);
    RCP<Epetra_CrsMatrix> fool = rcp(A,false);
    dwind_ = rcp(new LINALG::DownwindMatrix(fool,nv,np,tau,azoutput));
  }
  if (dwind && dwind_==null) dserror("Do not have downwinding matrix");

  // pass linear problem to aztec
  RCP<Epetra_LinearProblem> dwproblem;
  RCP<Epetra_CrsMatrix>     dwA;
  RCP<Epetra_MultiVector>   dwx;
  RCP<Epetra_MultiVector>   dwb;
  if (!dwind) aztec_->SetProblem(*lp_);
  else
  {
    dwA = dwind_->Permute(A);
    dwx = dwind_->Permute(lp_->GetLHS());
    dwb = dwind_->Permute(lp_->GetRHS());
    dwproblem = rcp(new Epetra_LinearProblem());
    dwproblem->SetOperator(dwA.get());
    dwproblem->SetLHS(dwx.get());
    dwproblem->SetRHS(dwb.get());
    aztec_->SetProblem(*dwproblem);
    A = dwA.get();
  }

  // Don't want linear problem to alter our aztec parameters (idiot feature!)
  // this is why we set our list here AFTER the linear problem has been set
  //
  // We don't want to use Aztec's scaling capabilities as we prefer to do
  // the scaling ourselves (so we precisely know what happens)
  // Therefore set scaling parameter to none and reset it after aztec has made
  // its internal copy of the parameter list
  azlist.set("scaling","none");
  aztec_->SetParameters(azlist,false);
  azlist.set("scaling",scaling);

  // create preconditioner if necessary
  if(create)
  {
#ifdef WRITEOUTSTATISTICS
  tttcreate.ResetStartTime();
#endif

    // dump old preconditioner
    P_ = Teuchos::null;

    // do ifpack if desired
    if (doifpack)
    {
      // parameter list (ifpack parameters)
      ParameterList&  ifpacklist = Params().sublist("IFPACK Parameters");

      // free old matrix to avoid usage of (temporary) memory during copy procedure
      Pmatrix_ = Teuchos::null;

      // create a copy of the scaled matrix
      // so we can reuse the preconditioner
      Pmatrix_ = rcp(new Epetra_CrsMatrix(*A));

      // get the type of ifpack preconditioner from aztec
      string prectype = azlist.get("preconditioner","ILU");
      int    overlap  = azlist.get("AZ_overlap",0);
      Ifpack Factory;
      Ifpack_Preconditioner* prec = Factory.Create(prectype,Pmatrix_.get(),overlap);
      prec->SetParameters(ifpacklist);
      prec->Initialize();
      prec->Compute();

      P_ = rcp(new LINALG::LinalgPrecondOperator::LinalgPrecondOperator(rcp(prec),project,projector));
    }

    // do ml if desired
    if(doml)
    {
      // parameter list (ml parameters)
      ParameterList&  mllist = Params().sublist("ML Parameters");

      // free old matrix to avoid usage of (temporary) memory during copy procedure
      Pmatrix_ = Teuchos::null;

      // create a copy of the scaled matrix
      // so we can reuse the preconditioner
      Pmatrix_ = rcp(new Epetra_CrsMatrix(*A));

      // see whether we use standard ml or our own mlapi operator
      const bool domlapioperator = mllist.get<bool>("LINALG::AMG_Operator",false);
      if (domlapioperator)
      {
        Teuchos::RCP<LINALG::AMG_Operator> linalgAMG
          = rcp(new LINALG::AMG_Operator(Pmatrix_,mllist,true));

        P_ = rcp(new LINALG::LinalgPrecondOperator::LinalgPrecondOperator(linalgAMG,project,projector));
      }
      else
      {
        Teuchos::RCP<ML_Epetra::MultiLevelPreconditioner> linalgML
          = rcp(new ML_Epetra::MultiLevelPreconditioner(*Pmatrix_,mllist,true));

        P_ = rcp(new LINALG::LinalgPrecondOperator::LinalgPrecondOperator(linalgML,project,projector));
        // for debugging ML
        //dynamic_cast<ML_Epetra::MultiLevelPreconditioner&>(*P_).PrintUnused(0);
      }
    }

    // do simpler if desired
    if(dosimpler)
    {
      // SIMPLER does not need copy of preconditioning matrix to live
      // SIMPLER does not use the downwinding installed here, it does
      // its own downwinding inside if desired

      Teuchos::RCP<LINALG::SIMPLER_Operator> SimplerOperator
          = rcp(new LINALG::SIMPLER_Operator(A_->UnprojectedOperator(),Params(),
                                             Params().sublist("SIMPLER"),
                                             outfile_));

      P_ = rcp(new LINALG::LinalgPrecondOperator::LinalgPrecondOperator(SimplerOperator,project,projector));

      Pmatrix_ = null;
    }

    if(doamgbs)
    {
      if(!Params().isSublist("AMGBS Parameters")) dserror("set AZPREC to ML for AMG(Braess-Sarazin) in FLUID SOLVER block");

      // Params().sublist("AMGBS") just contains the Fluid Pressure Solver block from the dat file
      Teuchos::RCP<LINALG::SaddlePointPreconditioner> SaddlePointPrec
        = rcp(new LINALG::SaddlePointPreconditioner(A_->UnprojectedOperator(),Params(),Params().sublist("AMGBS"),outfile_));

      P_ = rcp(new LINALG::LinalgPrecondOperator::LinalgPrecondOperator(SaddlePointPrec,project,projector));
      Pmatrix_ = null;
    }

#ifdef WRITEOUTSTATISTICS
    dtimeprecondsetup = tttcreate.ElapsedTime();
#endif
  }

  // set the preconditioner
  if (doifpack || doml || dosimpler || doamgbs)
  {
    aztec_->SetPrecOperator(P_.get());
  }

  // iterate on the solution
  int iter = azlist.get("AZ_max_iter",500);
  double tol = azlist.get("AZ_tol",1.0e-6);

  // create an aztec convergence test as combination of
  // L2-norm and Inf-Norm to be both satisfied where we demand
  // L2 < tol and Linf < 10*tol
  {
    Epetra_Operator* op  = aztec_->GetProblem()->GetOperator();
    Epetra_Vector*   rhs = static_cast<Epetra_Vector*>(aztec_->GetProblem()->GetRHS());
    Epetra_Vector*   lhs = static_cast<Epetra_Vector*>(aztec_->GetProblem()->GetLHS());
    // max iterations
    aztest_maxiter_ = rcp(new AztecOO_StatusTestMaxIters(iter));
    // L2 norm
    aztest_norm2_ = rcp(new AztecOO_StatusTestResNorm(*op,*lhs,*rhs,tol));
    aztest_norm2_->DefineResForm(AztecOO_StatusTestResNorm::Implicit,
                                 AztecOO_StatusTestResNorm::TwoNorm);
    aztest_norm2_->DefineScaleForm(AztecOO_StatusTestResNorm::NormOfInitRes,
                                   AztecOO_StatusTestResNorm::TwoNorm);
    // Linf norm (demanded to be 10 times L2-norm now, to become an input parameter)
    aztest_norminf_ = rcp(new AztecOO_StatusTestResNorm(*op,*lhs,*rhs,1.0*tol));
    aztest_norminf_->DefineResForm(AztecOO_StatusTestResNorm::Implicit,
                                   AztecOO_StatusTestResNorm::InfNorm);
    aztest_norminf_->DefineScaleForm(AztecOO_StatusTestResNorm::NormOfInitRes,
                                     AztecOO_StatusTestResNorm::InfNorm);
    // L2 AND Linf
    aztest_combo1_ = rcp(new AztecOO_StatusTestCombo(AztecOO_StatusTestCombo::SEQ));
    // maxiters OR (L2 AND Linf)
    aztest_combo2_ = rcp(new AztecOO_StatusTestCombo(AztecOO_StatusTestCombo::OR));
    aztest_combo1_->AddStatusTest(*aztest_norm2_);
    aztest_combo1_->AddStatusTest(*aztest_norminf_);
    aztest_combo2_->AddStatusTest(*aztest_maxiter_);
    aztest_combo2_->AddStatusTest(*aztest_combo1_);
    // set status test
    aztec_->SetStatusTest(aztest_combo2_.get());
  }

  // if you want to get some information on eigenvalues of the Hessenberg matrix/the
  // estimated condition number of the preconditioned system, uncomment the following
  // line and set AZOUTPUT>0 in your .dat-file
  // aztec_->SetAztecOption(AZ_solver,AZ_gmres_condnum);

  //------------------------------- just do it----------------------------------------
  aztec_->Iterate(iter,tol);
  //----------------------------------------------------------------------------------

  // undo downwinding
  if (dwind)
  {
    // undo reordering of lhs, don't care for rhs
    dwind_->InvPermute(dwproblem->GetLHS(),lp_->GetLHS());
    // undo reordering of matrix (by pointing to original matrix)
    if (A) A = dynamic_cast<Epetra_CrsMatrix*>(A_->UnprojectedOperator().get());
    // trash temporary data
    dwproblem = null;
    dwA = null;
    dwx = null;
    dwb = null;
  }

  // check status of solution process
  const double* status = aztec_->GetAztecStatus();
  if (status[AZ_why] != AZ_normal)
  {
    bool resolve = false;
    if (status[AZ_why] == AZ_breakdown)
    {
      if (Comm().MyPID()==0)
        printf("Numerical breakdown in AztecOO\n");
      resolve = true;
    }
    else if (status[AZ_why] == AZ_ill_cond)
    {
      if (Comm().MyPID()==0)
        printf("Problem is near singular in AztecOO\n");
      resolve = true;
    }
    else if (status[AZ_why] == AZ_maxits)
    {
      if (Comm().MyPID()==0)
        printf("Max iterations reached in AztecOO\n");
      resolve = true;
    }
  } // if (status[AZ_why] != AZ_normal)

  // undo scaling
  if (scaling_infnorm)
  {
    Epetra_Vector invrowsum(A->RowMap(),false);
    invrowsum.Reciprocal(*rowsum);
    rowsum = null;
    Epetra_Vector invcolsum(A->RowMap(),false);
    invcolsum.Reciprocal(*colsum);
    colsum = null;
    lp_->LeftScale(invrowsum);
    lp_->RightScale(invcolsum);
  }
  if (scaling_symdiag)
  {
    lp_->LeftScale(*diag);
    lp_->RightScale(*diag);
    diag = null;
  }

#ifdef WRITEOUTSTATISTICS
    if(outfile_)
    {
      fprintf(outfile_,"LinIter %i\tNumGlobalElements %i\tAZ_solve_time %f\tAztecSolveTime %f\tAztecPrecondSetup %f\t\n",(int)status[AZ_its],A_->OperatorRangeMap().NumGlobalElements(),status[AZ_solve_time],ttt.ElapsedTime(),dtimeprecondsetup);
      fflush(outfile_);
    }
#endif

  // print some output if desired
  if (Comm().MyPID()==0 && outfile_)
  {
    fprintf(outfile_,"AztecOO: unknowns/iterations/time %d  %d  %f\n",
            A_->OperatorRangeMap().NumGlobalElements(),(int)status[AZ_its],status[AZ_solve_time]);
    fflush(outfile_);
  }

  return;
}



/*----------------------------------------------------------------------*
 |  solve (protected)                                        mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Solve_superlu(const bool reset)
{

#ifndef HAVENOT_SUPERLU
#ifdef PARALLEL
  if (reset || !IsFactored())
  {
    reindexer_ = rcp(new EpetraExt::LinearProblem_Reindex(NULL));
    amesos_ = rcp(new Amesos_Superludist((*reindexer_)(*lp_)));
  }

  if (amesos_==null) dserror("No solver allocated");

  // Problem has not been factorized before
  if (!IsFactored())
  {
    int err = amesos_->SymbolicFactorization();
    if (err) dserror("Amesos::SymbolicFactorization returned an err");
    err = amesos_->NumericFactorization();
    if (err) dserror("Amesos::NumericFactorization returned an err");
  }

  int err = amesos_->Solve();
  if (err) dserror("Amesos::Solve returned an err");
#else
  dserror("Distributed SuperLU only in parallel");
#endif    //! system of equations
#endif
  RefCountPtr<Epetra_CrsMatrix>     A_;

  return;
}

/*----------------------------------------------------------------------*
 |  solve (protected)                                        mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Solve_umfpack(const bool reset)
{
#ifndef HAVENOT_UMFPACK
  if (reset || !IsFactored())
  {
    reindexer_ = rcp(new EpetraExt::LinearProblem_Reindex(NULL));
    amesos_ = rcp(new Amesos_Umfpack((*reindexer_)(*lp_)));
  }

  if (amesos_==null) dserror("No solver allocated");

  // Problem has not been factorized before
  if (!IsFactored())
  {
    bool symmetric = Params().get("symmetric",false);
    amesos_->SetUseTranspose(symmetric);
    int err = amesos_->SymbolicFactorization();
    if (err) dserror("Amesos::SymbolicFactorization returned an err");
    err = amesos_->NumericFactorization();
    if (err) dserror("Amesos::NumericFactorization returned an err");
  }

  int err = amesos_->Solve();
  if (err) dserror("Amesos::Solve returned an err");
#else
#endif
  return;
}

/*----------------------------------------------------------------------*
 |  solve (protected)                                        mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Solve_klu(const bool reset)
{
  if (reset || !IsFactored())
  {
    reindexer_ = rcp(new EpetraExt::LinearProblem_Reindex(NULL));
    amesos_ = rcp(new Amesos_Klu((*reindexer_)(*lp_)));
  }

  if (amesos_==null) dserror("No solver allocated");

  // Problem has not been factorized before
  if (!IsFactored())
  {
    bool symmetric = Params().get("symmetric",false);
    amesos_->SetUseTranspose(symmetric);
    int err = amesos_->SymbolicFactorization();
    if (err) dserror("Amesos::SymbolicFactorization returned an err");
    err = amesos_->NumericFactorization();
    if (err) dserror("Amesos::NumericFactorization returned an err");
  }

  int err = amesos_->Solve();
  if (err) dserror("Amesos::Solve returned an err");

  return;
}

/*----------------------------------------------------------------------*
 |  solve (protected)                                        mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::Solver::Solve_lapack(const bool reset)
{
  if (reset || !IsFactored())
  {
    reindexer_ = rcp(new EpetraExt::LinearProblem_Reindex(NULL));
    amesos_ = rcp(new Amesos_Lapack((*reindexer_)(*lp_)));
  }

  if (amesos_==null) dserror("No solver allocated");

  // Problem has not been factorized before
  if (!IsFactored())
  {
    int err = amesos_->SymbolicFactorization();
    if (err) dserror("Amesos::SymbolicFactorization returned an err");
    err = amesos_->NumericFactorization();
    if (err) dserror("Amesos::NumericFactorization returned an err");
  }

  int err = amesos_->Solve();
  if (err) dserror("Amesos::Solve returned an err");

  return;
}

/*----------------------------------------------------------------------*
 |  translate solver parameters (public)               mwgee 02/07,11/08|
 *----------------------------------------------------------------------*/
const Teuchos::ParameterList LINALG::Solver::TranslateSolverParameters(const ParameterList& inparams)
{
  // HINT:
  // input parameter inparams.get<int>("AZGRAPH") is not retrieved

  // make empty output parameters
  Teuchos::ParameterList outparams;

  // switch type of solver
  switch (Teuchos::getIntegralValue<INPAR::SOLVER::SolverType>(inparams,"SOLVER"))
  {
#ifdef PARALLEL
  case INPAR::SOLVER::superlu://============================== superlu solver (parallel only)
    outparams.set("solver","superlu");
    outparams.set("symmetric",false);
  break;
#endif
  case INPAR::SOLVER::amesos_klu_sym://====================================== Tim Davis' KLU
    outparams.set("solver","klu");
    outparams.set("symmetric",true);
  break;
  case INPAR::SOLVER::amesos_klu_nonsym://=================================== Tim Davis' KLU
    outparams.set("solver","klu");
    outparams.set("symmetric",false);
  break;
  case INPAR::SOLVER::umfpack://========================================= Tim Davis' Umfpack
    outparams.set("solver","umfpack");
    outparams.set("symmetric",false);
  break;
  case INPAR::SOLVER::lapack_sym://================================================== Lapack
    outparams.set("solver","lapack");
    outparams.set("symmetric",true);
  break;
  case INPAR::SOLVER::lapack_nonsym://=============================================== Lapack
    outparams.set("solver","lapack");
    outparams.set("symmetric",false);
  break;
  case INPAR::SOLVER::aztec_msr://================================================= AztecOO
  {
    outparams.set("solver","aztec");
    outparams.set("symmetric",false);
    ParameterList& azlist = outparams.sublist("Aztec Parameters");
    //--------------------------------- set scaling of linear problem
    const int azscal = Teuchos::getIntegralValue<int>(inparams,"AZSCAL");
    if (azscal==1)
      azlist.set("scaling","symmetric");
    else if (azscal==2)
      azlist.set("scaling","infnorm");
    else
      azlist.set("scaling","none");
    //--------------------------------------------- set type of solver
    switch (Teuchos::getIntegralValue<INPAR::SOLVER::AzSolverType>(inparams,"AZSOLVE"))
    {
    case INPAR::SOLVER::azsolv_CG:       azlist.set("AZ_solver",AZ_cg);       break;
    case INPAR::SOLVER::azsolv_GMRES:    azlist.set("AZ_solver",AZ_gmres);    break;
    case INPAR::SOLVER::azsolv_CGS:      azlist.set("AZ_solver",AZ_cgs);      break;
    case INPAR::SOLVER::azsolv_BiCGSTAB: azlist.set("AZ_solver",AZ_bicgstab); break;
    case INPAR::SOLVER::azsolv_LU:       azlist.set("AZ_solver",AZ_lu);       break;
    case INPAR::SOLVER::azsolv_TFQMR:    azlist.set("AZ_solver",AZ_tfqmr);    break;
    default: dserror("Unknown solver for AztecOO");            break;
    }
    //------------------------------------- set type of preconditioner
    const int azprectyp = Teuchos::getIntegralValue<INPAR::SOLVER::AzPrecType>(inparams,"AZPREC");
    switch (azprectyp)
    {
    case INPAR::SOLVER::azprec_none:
      azlist.set("AZ_precond",AZ_none);
      azlist.set("AZ_subdomain_solve",AZ_none);
      azlist.set("preconditioner",AZ_none);
    break;
    case INPAR::SOLVER::azprec_ILUT:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","ILUT");
    break;
    case INPAR::SOLVER::azprec_ILU:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","ILU");
    break;
    case INPAR::SOLVER::azprec_Neumann:
      azlist.set("AZ_precond",AZ_Neumann);
    break;
    case INPAR::SOLVER::azprec_Least_Squares:
      azlist.set("AZ_precond",AZ_ls);
    break;
    case INPAR::SOLVER::azprec_Jacobi:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","point relaxation");
    break;
    case INPAR::SOLVER::azprec_SymmGaussSeidel:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","point relaxation");
    break;
    case INPAR::SOLVER::azprec_GaussSeidel:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","point relaxation");
    break;
    case INPAR::SOLVER::azprec_DownwindGaussSeidel:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","point relaxation");
      azlist.set<bool>("downwinding",true);
      azlist.set<double>("downwinding tau",inparams.get<double>("DWINDTAU"));
    break;
    case INPAR::SOLVER::azprec_LU:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","Amesos");
    break;
    case INPAR::SOLVER::azprec_RILU:
      azlist.set("AZ_precond",AZ_dom_decomp);
      azlist.set("AZ_subdomain_solve",AZ_rilu);
      azlist.set("AZ_graph_fill",inparams.get<int>("IFPACKGFILL"));
    break;
    case INPAR::SOLVER::azprec_ICC:
      // using ifpack
      azlist.set("AZ_precond",AZ_user_precond);
      azlist.set("preconditioner","IC");
    break;
    case INPAR::SOLVER::azprec_ML:
    case INPAR::SOLVER::azprec_MLfluid:
    case INPAR::SOLVER::azprec_MLAPI:
    case INPAR::SOLVER::azprec_MLfluid2:
    case INPAR::SOLVER::azprec_AMGBS:
      azlist.set("AZ_precond",AZ_user_precond);
    break;
    default:
      dserror("Unknown preconditioner for AztecOO");
    break;
    }
    //------------------------------------- set other aztec parameters
    azlist.set("AZ_kspace",inparams.get<int>("AZSUB"));
    azlist.set("AZ_max_iter",inparams.get<int>("AZITER"));
    azlist.set("AZ_overlap",inparams.get<int>("IFPACKOVERLAP"));
    azlist.set("AZ_type_overlap",AZ_symmetric);
    azlist.set("AZ_poly_ord",inparams.get<int>("AZPOLY"));
    const int azoutput = inparams.get<int>("AZOUTPUT");
    if (!azoutput)
      azlist.set("AZ_output",AZ_none);             // AZ_none AZ_all AZ_warnings AZ_last 10
    else
      azlist.set("AZ_output",azoutput);
    azlist.set("AZ_diagnostics",inparams.get<int>("AZBDIAG"));          // AZ_none AZ_all
    azlist.set("AZ_conv",Teuchos::getIntegralValue<int>(inparams,"AZCONV"));
    azlist.set("AZ_tol",inparams.get<double>("AZTOL"));
    azlist.set("AZ_drop",inparams.get<double>("AZDROP"));
    azlist.set("AZ_scaling",AZ_none);
    azlist.set("AZ_keep_info",0);
    // set reuse parameters
    azlist.set("ncall",0);                         // counting number of solver calls
    azlist.set("reuse",inparams.get<int>("AZREUSE"));            // reuse info for n solver calls
    //-------------------------------- set parameters for Ifpack if used
    if (azprectyp == INPAR::SOLVER::azprec_ILU  ||
        azprectyp == INPAR::SOLVER::azprec_ILUT ||
        azprectyp == INPAR::SOLVER::azprec_ICC  ||
        azprectyp == INPAR::SOLVER::azprec_LU   ||
        azprectyp == INPAR::SOLVER::azprec_SymmGaussSeidel ||
        azprectyp == INPAR::SOLVER::azprec_GaussSeidel ||
        azprectyp == INPAR::SOLVER::azprec_DownwindGaussSeidel ||
        azprectyp == INPAR::SOLVER::azprec_Jacobi)
    {
      ParameterList& ifpacklist = outparams.sublist("IFPACK Parameters");
      ifpacklist.set("relaxation: damping factor",inparams.get<double>("AZOMEGA"));
      ifpacklist.set("fact: drop tolerance",inparams.get<double>("AZDROP"));
      ifpacklist.set("fact: level-of-fill",inparams.get<int>("IFPACKGFILL"));
      ifpacklist.set("fact: ilut level-of-fill",inparams.get<double>("IFPACKFILL"));
      ifpacklist.set("partitioner: overlap",inparams.get<int>("IFPACKOVERLAP"));
      ifpacklist.set("schwarz: combine mode",inparams.get<string>("IFPACKCOMBINE")); // can be "Zero", "Add", "Insert"
      ifpacklist.set("schwarz: reordering type","rcm"); // "rcm" or "metis" or "amd"
      ifpacklist.set("amesos: solver type", "Amesos_Klu"); // can be "Amesos_Klu", "Amesos_Umfpack", "Amesos_Superlu"
      if (azprectyp == INPAR::SOLVER::azprec_SymmGaussSeidel)
      {
        ifpacklist.set("relaxation: type","symmetric Gauss-Seidel");
        ifpacklist.set("relaxation: sweeps",inparams.get<int>("IFPACKGFILL"));
        ifpacklist.set("relaxation: damping factor",inparams.get<double>("AZOMEGA"));
      }
      if (azprectyp == INPAR::SOLVER::azprec_GaussSeidel)
      {
        ifpacklist.set("relaxation: type","Gauss-Seidel");
        ifpacklist.set("relaxation: sweeps",inparams.get<int>("IFPACKGFILL"));
        ifpacklist.set("relaxation: damping factor",inparams.get<double>("AZOMEGA"));
      }
      if (azprectyp == INPAR::SOLVER::azprec_DownwindGaussSeidel)
      {
        // in case of downwinding prevent ifpack from again reordering
        ifpacklist.set("schwarz: reordering type","none");
        ifpacklist.set("relaxation: type","Gauss-Seidel");
        ifpacklist.set("relaxation: sweeps",inparams.get<int>("IFPACKGFILL"));
        ifpacklist.set("relaxation: damping factor",inparams.get<double>("AZOMEGA"));
      }
      if (azprectyp == INPAR::SOLVER::azprec_Jacobi)
      {
        ifpacklist.set("relaxation: type","Jacobi");
        ifpacklist.set("relaxation: sweeps",inparams.get<int>("IFPACKGFILL"));
        ifpacklist.set("relaxation: damping factor",inparams.get<double>("AZOMEGA"));
      }
    }
    //------------------------------------- set parameters for ML if used
    if (azprectyp == INPAR::SOLVER::azprec_ML       ||
        azprectyp == INPAR::SOLVER::azprec_MLfluid  ||
        azprectyp == INPAR::SOLVER::azprec_MLfluid2 ||
        azprectyp == INPAR::SOLVER::azprec_MLAPI )
    {
      ParameterList& mllist = outparams.sublist("ML Parameters");
      ML_Epetra::SetDefaults("SA",mllist);
      switch (azprectyp)
      {
      case INPAR::SOLVER::azprec_ML: // do nothing, this is standard
      break;
      case INPAR::SOLVER::azprec_MLAPI: // set flag to use mlapi operator
        mllist.set<bool>("LINALG::AMG_Operator",true);
      break;
      case INPAR::SOLVER::azprec_MLfluid: // unsymmetric, unsmoothed restruction
        mllist.set("aggregation: use tentative restriction",true);
      break;
      case INPAR::SOLVER::azprec_MLfluid2: // full Pretrov-Galerkin unsymmetric smoothed
        mllist.set("energy minimization: enable",true);
        mllist.set("energy minimization: type",3); // 1,2,3 cheap -> expensive
        mllist.set("aggregation: block scaling",false);
      break;
      default: dserror("Unknown type of ml preconditioner");
      }
      mllist.set("output"                          ,inparams.get<int>("ML_PRINT"));
      if (inparams.get<int>("ML_PRINT")==10)
        mllist.set("print unused"                  ,1);
      else
        mllist.set("print unused"                  ,-2);
      mllist.set("increasing or decreasing"        ,"increasing");
      mllist.set("coarse: max size"                ,inparams.get<int>("ML_MAXCOARSESIZE"));
      mllist.set("max levels"                      ,inparams.get<int>("ML_MAXLEVEL"));
      mllist.set("smoother: pre or post"           ,"both");
      mllist.set("aggregation: threshold"          ,inparams.get<double>("ML_PROLONG_THRES"));
      mllist.set("aggregation: damping factor"     ,inparams.get<double>("ML_PROLONG_SMO"));
      mllist.set("aggregation: nodes per aggregate",inparams.get<int>("ML_AGG_SIZE"));
      // override the default sweeps=2 with a default sweeps=1
      // individual level sweeps are set below
      mllist.set("smoother: sweeps",1);
      switch (Teuchos::getIntegralValue<int>(inparams,"ML_COARSEN"))
      {
        case 0:  mllist.set("aggregation: type","Uncoupled");  break;
        case 1:  mllist.set("aggregation: type","METIS");      break;
        case 2:  mllist.set("aggregation: type","VBMETIS");    break;
        case 3:  mllist.set("aggregation: type","MIS");        break;
        default: dserror("Unknown type of coarsening for ML"); break;
      }

      // set ml smoothers
      const int mlmaxlevel = inparams.get<int>("ML_MAXLEVEL");
      // create vector of integers containing smoothing steps/polynomial order of level
      std::vector<int> mlsmotimessteps;
      {
        std::istringstream mlsmotimes(Teuchos::getNumericStringParameter(inparams,"ML_SMOTIMES"));
        std::string word;
        while (mlsmotimes >> word)
          mlsmotimessteps.push_back(std::atoi(word.c_str()));
      }

      if ((int)mlsmotimessteps.size() < mlmaxlevel)
        dserror("Not enough smoothing steps ML_SMOTIMES=%d, must be larger/equal than ML_MAXLEVEL=%d\n",
                mlsmotimessteps.size(),mlmaxlevel);

      for (int i=0; i<mlmaxlevel-1; ++i)
      {
        char levelstr[11];
        sprintf(levelstr,"(level %d)",i);
        ParameterList& smolevelsublist = mllist.sublist("smoother: list "+(string)levelstr);
        int type;
        double damp;
        if (i==0)
        {
          type = Teuchos::getIntegralValue<int>(inparams,"ML_SMOOTHERFINE");
          damp = inparams.get<double>("ML_DAMPFINE");
        }
        else if (i < mlmaxlevel-1)
        {
          type = Teuchos::getIntegralValue<int>(inparams,"ML_SMOOTHERMED");
          damp = inparams.get<double>("ML_DAMPMED");
        }
        else
        {
          type = Teuchos::getIntegralValue<int>(inparams,"ML_SMOOTHERCOARSE");
          damp = inparams.get<double>("ML_DAMPCOARSE");
        }
        switch (type)
        {
        case 0: // SGS
          smolevelsublist.set("smoother: type"                        ,"symmetric Gauss-Seidel");
          smolevelsublist.set("smoother: sweeps"                      ,mlsmotimessteps[i]);
          smolevelsublist.set("smoother: damping factor"              ,damp);
        break;
        case 7: // GS
          smolevelsublist.set("smoother: type"                        ,"Gauss-Seidel");
          smolevelsublist.set("smoother: sweeps"                      ,mlsmotimessteps[i]);
          smolevelsublist.set("smoother: damping factor"              ,damp);
        break;
        case 8: // DGS
          smolevelsublist.set("smoother: type"                        ,"Gauss-Seidel");
          smolevelsublist.set("smoother: sweeps"                      ,mlsmotimessteps[i]);
          smolevelsublist.set("smoother: damping factor"              ,damp);
          azlist.set<bool>("downwinding",true);
          azlist.set<double>("downwinding tau",inparams.get<double>("DWINDTAU"));
          {
            ParameterList& ifpacklist = mllist.sublist("smoother: ifpack list");
            ifpacklist.set("schwarz: reordering type","true");
          }
        break;
        case 1: // Jacobi
          smolevelsublist.set("smoother: type"                        ,"Jacobi");
          smolevelsublist.set("smoother: sweeps"                      ,mlsmotimessteps[i]);
          smolevelsublist.set("smoother: damping factor"              ,damp);
        break;
        case 2: // Chebychev
          smolevelsublist.set("smoother: type"                        ,"MLS");
          smolevelsublist.set("smoother: MLS polynomial order"        ,mlsmotimessteps[i]);
        break;
        case 3: // MLS
          smolevelsublist.set("smoother: type"                        ,"MLS");
          smolevelsublist.set("smoother: MLS polynomial order"        ,-mlsmotimessteps[i]);
        break;
        case 4: // Ifpack's ILU
        {
          smolevelsublist.set("smoother: type"                        ,"IFPACK");
          smolevelsublist.set("smoother: ifpack type"                 ,"ILU");
          smolevelsublist.set("smoother: ifpack overlap"              ,inparams.get<int>("IFPACKOVERLAP"));
          smolevelsublist.set<double>("smoother: ifpack level-of-fill",(double)mlsmotimessteps[i]);
          ParameterList& ifpacklist = mllist.sublist("smoother: ifpack list");
          ifpacklist.set("schwarz: reordering type","rcm"); // "rcm" or "metis" or "amd" or "true"
          ifpacklist.set("schwarz: combine mode",inparams.get<string>("IFPACKCOMBINE")); // can be "Zero", "Insert", "Add"
          ifpacklist.set("partitioner: overlap",inparams.get<int>("IFPACKOVERLAP"));
        }
        break;
        case 5: // Amesos' KLU
          smolevelsublist.set("smoother: type"                        ,"Amesos-KLU");
        break;
        case 9: // Amesos' Umfpack
          smolevelsublist.set("smoother: type"                        ,"Amesos-UMFPACK");
        break;
#ifdef PARALLEL
        case 6: // Amesos' SuperLU_Dist
          smolevelsublist.set("smoother: type"                        ,"Amesos-Superludist");
        break;
#endif
        default: dserror("Unknown type of smoother for ML: tuple %d",type); break;
        } // switch (type)
      } // for (int i=0; i<azvar->mlmaxlevel-1; ++i)

      // set coarse grid solver
      const int coarse = mlmaxlevel-1;
      switch (Teuchos::getIntegralValue<int>(inparams,"ML_SMOOTHERCOARSE"))
      {
        case 0:
          mllist.set("coarse: type"          ,"symmetric Gauss-Seidel");
          mllist.set("coarse: sweeps"        ,mlsmotimessteps[coarse]);
          mllist.set("coarse: damping factor",inparams.get<double>("ML_DAMPCOARSE"));
        break;
        case 7:
          mllist.set("coarse: type"          ,"Gauss-Seidel");
          mllist.set("coarse: sweeps"        ,mlsmotimessteps[coarse]);
          mllist.set("coarse: damping factor",inparams.get<double>("ML_DAMPCOARSE"));
        break;
        case 8:
          mllist.set("coarse: type"          ,"Gauss-Seidel");
          mllist.set("coarse: sweeps"        ,mlsmotimessteps[coarse]);
          mllist.set("coarse: damping factor",inparams.get<double>("ML_DAMPCOARSE"));
          azlist.set<bool>("downwinding",true);
          azlist.set<double>("downwinding tau",inparams.get<double>("DWINDTAU"));
          {
            ParameterList& ifpacklist = mllist.sublist("smoother: ifpack list");
            ifpacklist.set("schwarz: reordering type","true");
          }
        break;
        case 1:
          mllist.set("coarse: type"          ,"Jacobi");
          mllist.set("coarse: sweeps"        ,mlsmotimessteps[coarse]);
          mllist.set("coarse: damping factor",inparams.get<double>("ML_DAMPCOARSE"));
        break;
        case 2:
          mllist.set("coarse: type"                ,"MLS");
          mllist.set("coarse: MLS polynomial order",mlsmotimessteps[coarse]);
        break;
        case 3:
          mllist.set("coarse: type"                ,"MLS");
          mllist.set("coarse: MLS polynomial order",-mlsmotimessteps[coarse]);
        break;
        case 4:
        {
          mllist.set("coarse: type"          ,"IFPACK");
          mllist.set("coarse: ifpack type"   ,"ILU");
          mllist.set("coarse: ifpack overlap",0);
          mllist.set<double>("coarse: ifpack level-of-fill",(double)mlsmotimessteps[coarse]);
          ParameterList& ifpacklist = mllist.sublist("coarse: ifpack list");
          ifpacklist.set<int>("fact: level-of-fill",mlsmotimessteps[coarse]);
          ifpacklist.set("schwarz: reordering type","rcm");
          ifpacklist.set("schwarz: combine mode",inparams.get<string>("IFPACKCOMBINE")); // can be "Zero", "Insert", "Add"
          ifpacklist.set("partitioner: overlap",inparams.get<int>("IFPACKOVERLAP"));
        }
        break;
        case 5:
          mllist.set("coarse: type","Amesos-KLU");
        break;
        case 9:
          mllist.set("coarse: type","Amesos-UMFPACK");
        break;
        case 6:
          mllist.set("coarse: type","Amesos-Superludist");
        break;
        default: dserror("Unknown type of coarse solver for ML"); break;
      } // switch (azvar->mlsmotype_coarse)
      // default values for nullspace
      mllist.set("PDE equations",1);
      mllist.set("null space: dimension",1);
      mllist.set("null space: type","pre-computed");
      mllist.set("null space: add default vectors",false);
      mllist.set<double*>("null space: vectors",NULL);
#if defined(PARALLEL) && defined(PARMETIS)
      mllist.set("repartition: enable",1);
      mllist.set("repartition: partitioner","ParMETIS");
      mllist.set("repartition: max min ratio",1.3);
      mllist.set("repartition: min per proc",3000);
#endif
      //cout << mllist << endl << endl << endl; fflush(stdout);
    } // if ml preconditioner
    //------------------------------------- set parameters for AMGBS if used
    if (azprectyp == INPAR::SOLVER::azprec_AMGBS      )
    {
      //ParameterList& mllist = outparams.sublist("ML Parameters");   // dummy ML Parameter List for ComuteNullSpaceIfNecessary
      ParameterList& amglist = outparams.sublist("AMGBS Parameters");
      ML_Epetra::SetDefaults("SA",amglist);
      amglist.set("amgbs: smoother: pre or post"    ,"both");
      amglist.set("amgbs: prolongator smoother (vel)",inparams.get<string>("AMGBS_PSMOOTHER_VEL"));
      amglist.set("amgbs: prolongator smoother (pre)",inparams.get<string>("AMGBS_PSMOOTHER_PRE"));

      amglist.set("output"                          ,inparams.get<int>("ML_PRINT"));
      amglist.set("coarse: max size"                ,inparams.get<int>("ML_MAXCOARSESIZE"));
      amglist.set("max levels"                      ,inparams.get<int>("ML_MAXLEVEL"));
      amglist.set("aggregation: threshold"          ,inparams.get<double>("ML_PROLONG_THRES"));
      amglist.set("aggregation: damping factor"     ,inparams.get<double>("ML_PROLONG_SMO"));
      amglist.set("aggregation: nodes per aggregate",inparams.get<int>("ML_AGG_SIZE"));
      // override the default sweeps=2 with a default sweeps=1
      // individual level sweeps are set below
      amglist.set("smoother: sweeps",1);
      switch (Teuchos::getIntegralValue<int>(inparams,"ML_COARSEN"))
      {
        case 0:  amglist.set("aggregation: type","Uncoupled");  break;
        case 1:  amglist.set("aggregation: type","METIS");      break;
        case 2:  amglist.set("aggregation: type","VBMETIS");    break;
        case 3:  amglist.set("aggregation: type","MIS");        break;
        default: dserror("Unknown type of coarsening for ML"); break;
      }

      //////////////////// set braess-sarazin smoothers
      const int mlmaxlevel = inparams.get<int>("ML_MAXLEVEL");
      // create vector of integers containing smoothing steps/polynomial order of level
      std::vector<int> mlsmotimessteps;
      {
        std::istringstream mlsmotimes(Teuchos::getNumericStringParameter(inparams,"ML_SMOTIMES"));
        std::string word;
        while (mlsmotimes >> word)
          mlsmotimessteps.push_back(std::atoi(word.c_str()));
      }

      if ((int)mlsmotimessteps.size() < mlmaxlevel)
        dserror("Not enough smoothing steps ML_SMOTIMES=%d, must be larger/equal than ML_MAXLEVEL=%d\n",
                mlsmotimessteps.size(),mlmaxlevel);

      //////////////////// read in damping parameters for Braess Sarazin
      std::vector<double> bsdamping;   // damping parameters for Braess-Sarazin
      {
        double word;
        std::istringstream bsdampingstream(Teuchos::getNumericStringParameter(inparams,"AMGBS_BS_DAMPING"));
        while (bsdampingstream >> word)
          bsdamping.push_back(word);
      }
      if ((int)bsdamping.size() < mlmaxlevel)
        dserror("Not enough damping factors AMGBS_BS_DAMPING=%d, must be larger/equal than ML_MAXLEVEL=%d\n",
                bsdamping.size(),mlmaxlevel);


      for (int i=0; i<mlmaxlevel; ++i)
      {
        char levelstr[11];
        sprintf(levelstr,"(level %d)",i);
        ParameterList& smolevelsublist = amglist.sublist("braess-sarazin: list "+(string)levelstr);

          smolevelsublist.set("braess-sarazin: sweeps"                      ,mlsmotimessteps[i]);
          smolevelsublist.set("braess-sarazin: damping factor"              ,bsdamping[i]);
          smolevelsublist.set("pressure correction approx: type"      ,"IFPACK");   // TODO choose IFPACK or ML or UMFPACK

          switch (Teuchos::getIntegralValue<int>(inparams,"AMGBS_BS_PCCOARSE"))
          {
            case 0:
            {
              smolevelsublist.set("coarse: type","Umfpack");
            }
            break;
            case 1:
            {
              smolevelsublist.set("coarse: type","KLU");
            }
            break;
            case 2:
            {
              smolevelsublist.set("coarse: type"          ,"IFPACK");
              smolevelsublist.set("coarse: ifpack type"   ,"ILU");
              //smolevelsublist.set("coarse: ifpack overlap",0);
              //smolevelsublist.set<double>("coarse: ifpack level-of-fill",(double)mlsmotimessteps[coarse]);
              //ParameterList& ifpacklist = smolevelsublist.sublist("coarse: ifpack list");
              //ifpacklist.set<int>("fact: level-of-fill",mlsmotimessteps[coarse]);
              //ifpacklist.set("schwarz: reordering type","rcm");
            }
            break;
            case 3:
              smolevelsublist.set("coarse: type","ML");
              break;
            default: dserror("Unknown type of coarse solver for pressure correction equation"); break;
          } // switch (azvar->mlsmotype_coarse)

          switch (Teuchos::getIntegralValue<int>(inparams,"AMGBS_BS_PCMEDIUM"))
          {
            case 0:
            {
              smolevelsublist.set("medium: type","Umfpack");
            }
            break;
            case 1:
            {
              smolevelsublist.set("medium: type","KLU");
            }
            break;
            case 2:
            {
              smolevelsublist.set("medium: type"          ,"IFPACK");
              smolevelsublist.set("medium: ifpack type"   ,"ILU");
            }
            break;
            case 3:
              smolevelsublist.set("medium: type","ML");
              break;
            default: dserror("Unknown type of medium level solver for pressure correction equation"); break;
          }

          switch (Teuchos::getIntegralValue<int>(inparams,"AMGBS_BS_PCFINE"))
          {
            case 0:
            {
              smolevelsublist.set("fine: type","Umfpack");
            }
            break;
            case 1:
            {
              smolevelsublist.set("fine: type","KLU");
            }
            break;
            case 2:
            {
              smolevelsublist.set("fine: type"          ,"IFPACK");
              smolevelsublist.set("fine: ifpack type"   ,"ILU");
            }
            break;
            case 3:
              smolevelsublist.set("fine: type","ML");
              break;
            default: dserror("Unknown type of coarse solver for pressure correction equation"); break;
          } // switch (azvar->mlsmotype_coarse)


      } // for (int i=0; i<azvar->mlmaxlevel-1; ++i)



      amglist.set("PDE equations",1);
      amglist.set("null space: dimension",1);
      amglist.set("null space: type","pre-computed");
      amglist.set("null space: add default vectors",false);
      amglist.set<double*>("null space: vectors",NULL);



      cout << amglist << endl;
    } // if AMGBS preconditioner
  }
  break;
#ifdef PARALLEL
  case INPAR::SOLVER::SPOOLES_sym://================================== Spooles (parallel only)
  case INPAR::SOLVER::SPOOLES_nonsym:
    outparams.set("solver","spooles");
    outparams.set("symmetric",false);
  break;
#endif
  default:
    dserror("Unsupported type of solver");
  break;
  }

  //================================================================== deliver
  return outparams;
}


/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 02/08|
 *----------------------------------------------------------------------*/
RCP<LINALG::SparseMatrix> LINALG::MLMultiply(const LINALG::SparseMatrix& A,
                                             const LINALG::SparseMatrix& B,
                                             bool complete)
{
  return MLMultiply(*A.EpetraMatrix(),*B.EpetraMatrix(),
                    A.explicitdirichlet_,A.savegraph_,complete);
}

/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 02/08|
 *----------------------------------------------------------------------*/
RCP<LINALG::SparseMatrix> LINALG::MLMultiply(const LINALG::SparseMatrix& A,
                                             const LINALG::SparseMatrix& B,
                                             bool explicitdirichlet,
                                             bool savegraph,
                                             bool complete)
{
  return MLMultiply(*A.EpetraMatrix(),*B.EpetraMatrix(),
                    explicitdirichlet,savegraph,complete);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix> LINALG::MLMultiply(const LINALG::SparseMatrix& A,
                                                      bool transA,
                                                      const LINALG::SparseMatrix& B,
                                                      bool transB,
                                                      bool explicitdirichlet,
                                                      bool savegraph,
                                                      bool completeoutput)
{
  // make sure FillComplete was called on the matrices
  if (!A.Filled()) dserror("A has to be FillComplete");
  if (!B.Filled()) dserror("B has to be FillComplete");

  EpetraExt::RowMatrix_Transpose transposera(true,NULL,false);
  EpetraExt::RowMatrix_Transpose transposerb(true,NULL,false);
  Epetra_CrsMatrix* Atrans = NULL;
  Epetra_CrsMatrix* Btrans = NULL;
  if (transA)
    Atrans = &(dynamic_cast<Epetra_CrsMatrix&>(transposera(*A.EpetraMatrix())));
  else 
    Atrans = A.EpetraMatrix().get();
  if (transB)
    Btrans = &(dynamic_cast<Epetra_CrsMatrix&>(transposerb(*B.EpetraMatrix())));
  else 
    Btrans = B.EpetraMatrix().get();
  
  Teuchos::RCP<LINALG::SparseMatrix> C;
  C = LINALG::MLMultiply(*Atrans,*Btrans,explicitdirichlet,savegraph,completeoutput);

  return C;
}

/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 02/08|
 *----------------------------------------------------------------------*/
//static void CopySortDeleteZeros(const Epetra_CrsMatrix& A, Epetra_CrsMatrix& As);
RCP<LINALG::SparseMatrix> LINALG::MLMultiply(const Epetra_CrsMatrix& Aorig,
                                             const Epetra_CrsMatrix& Borig,
                                             bool explicitdirichlet,
                                             bool savegraph,
                                             bool complete)
{
  EpetraExt::CrsMatrix_SolverMap Atransform;
  EpetraExt::CrsMatrix_SolverMap Btransform;
  const Epetra_CrsMatrix& A = Atransform(const_cast<Epetra_CrsMatrix&>(Aorig));
  const Epetra_CrsMatrix& B = Btransform(const_cast<Epetra_CrsMatrix&>(Borig));

  // make sure FillComplete was called on the matrices
  if (!A.Filled()) dserror("A has to be FillComplete");
  if (!B.Filled()) dserror("B has to be FillComplete");

  // For debugging, it might be helpful when all columns are
  // sorted and all zero values are wiped from the input:
  //RCP<Epetra_CrsMatrix> As = CreateMatrix(A.RowMap(),A.MaxNumEntries());
  //RCP<Epetra_CrsMatrix> Bs = CreateMatrix(B.RowMap(),B.MaxNumEntries());
  //CopySortDeleteZeros(A,*As);
  //CopySortDeleteZeros(B,*Bs);
  ML_Operator* ml_As = ML_Operator_Create(GetML_Comm());
  ML_Operator* ml_Bs = ML_Operator_Create(GetML_Comm());
  //ML_Operator_WrapEpetraMatrix(As.get(),ml_As);
  //ML_Operator_WrapEpetraMatrix(Bs.get(),ml_Bs);
  ML_Operator_WrapEpetraMatrix(const_cast<Epetra_CrsMatrix*>(&A),ml_As);
  ML_Operator_WrapEpetraMatrix(const_cast<Epetra_CrsMatrix*>(&B),ml_Bs);
  ML_Operator* ml_AtimesB = ML_Operator_Create(GetML_Comm());
  ML_2matmult(ml_As,ml_Bs,ml_AtimesB,ML_CSR_MATRIX); // do NOT use ML_EpetraCRS_MATRIX !!
  ML_Operator_Destroy(&ml_As);
  ML_Operator_Destroy(&ml_Bs);
  // For ml_AtimesB we have to reconstruct the column map in global indexing,
  // The following is going down to the salt-mines of ML ...
  int N_local = ml_AtimesB->invec_leng;
  ML_CommInfoOP* getrow_comm = ml_AtimesB->getrow->pre_comm;
  if (!getrow_comm) dserror("ML_Operator does not have CommInfo");
  ML_Comm* comm = ml_AtimesB->comm;
  if (N_local != B.DomainMap().NumMyElements())
    dserror("Mismatch in local row dimension between ML and Epetra");
  int N_rcvd  = 0;
  int N_send  = 0;
  int flag    = 0;
  for (int i=0; i<getrow_comm->N_neighbors; i++)
  {
    N_rcvd += (getrow_comm->neighbors)[i].N_rcv;
    N_send += (getrow_comm->neighbors)[i].N_send;
    if (  ((getrow_comm->neighbors)[i].N_rcv != 0) &&
       ((getrow_comm->neighbors)[i].rcv_list != NULL) )  flag = 1;
  }
  // For some unknown reason, ML likes to have stuff one larger than
  // neccessary...
  vector<double> dtemp(N_local+N_rcvd+1);
  vector<int>    cmap(N_local+N_rcvd+1);
  for (int i=0; i<N_local; ++i)
  {
    cmap[i] = B.DomainMap().GID(i);
    dtemp[i] = (double)cmap[i];
  }
  ML_cheap_exchange_bdry(&dtemp[0],getrow_comm,N_local,N_send,comm);
  if (flag)
  {
    int count = N_local;
    const int neighbors = getrow_comm->N_neighbors;
    for (int i=0; i<neighbors; i++)
    {
      const int nrcv = getrow_comm->neighbors[i].N_rcv;
      for (int j=0; j<nrcv; j++)
        cmap[getrow_comm->neighbors[i].rcv_list[j]] = (int)dtemp[count++];
    }
  }
  else
    for (int i=0; i<N_local+N_rcvd; ++i) cmap[i] = (int)dtemp[i];
  dtemp.clear();

  // we can now determine a matching column map for the result
  Epetra_Map gcmap(-1,N_local+N_rcvd,&cmap[0],0,A.Comm());

  int allocated=0;
  int rowlength;
  double* val=NULL;
  int* bindx=NULL;
  const int myrowlength = A.RowMap().NumMyElements();
  const Epetra_Map& rowmap = A.RowMap();

  // determine the maximum bandwith for the result matrix.
  // replaces the old, very(!) memory-consuming guess:
  // int guessnpr = A.MaxNumEntries()*B.MaxNumEntries();
  int educatedguess = 0;
  for (int i=0; i<myrowlength; ++i)
  {
    // get local row
    ML_get_matrix_row(ml_AtimesB,1,&i,&allocated,&bindx,&val,&rowlength,0);
    if (rowlength>educatedguess) educatedguess = rowlength;
  }

  // allocate our result matrix and fill it
  RCP<Epetra_CrsMatrix> result
    = rcp(new Epetra_CrsMatrix(Copy,A.RangeMap(),gcmap,educatedguess,false));

  vector<int> gcid(educatedguess);
  for (int i=0; i<myrowlength; ++i)
  {
    const int grid = rowmap.GID(i);
    // get local row
    ML_get_matrix_row(ml_AtimesB,1,&i,&allocated,&bindx,&val,&rowlength,0);
    if (!rowlength) continue;
    if ((int)gcid.size() < rowlength) gcid.resize(rowlength);
    for (int j=0; j<rowlength; ++j)
    {
      gcid[j] = gcmap.GID(bindx[j]);
#ifdef DEBUG
      if (gcid[j]<0) dserror("This is really bad... cannot find gcid");
#endif
    }
#ifdef DEBUG
    int err = result->InsertGlobalValues(grid,rowlength,val,&gcid[0]);
    if (err!=0 && err!=1) dserror("Epetra_CrsMatrix::InsertGlobalValues returned err=%d",err);
#else
    result->InsertGlobalValues(grid,rowlength,val,&gcid[0]);
#endif
  }
  if (bindx) ML_free(bindx);
  if (val) ML_free(val);
  ML_Operator_Destroy(&ml_AtimesB);
  if (complete)
  {
    int err = result->FillComplete(B.DomainMap(),A.RangeMap());
    if (err) dserror("Epetra_CrsMatrix::FillComplete returned err=%d",err);

#if 0 // the current status is that we don't need this (mwgee)
    EpetraExt::CrsMatrix_SolverMap ABtransform;
    const Epetra_CrsMatrix& tmp = ABtransform(*result);
    RCP<Epetra_CrsMatrix> finalresult = rcp(new Epetra_CrsMatrix(*result));
    if (!finalresult->Filled()) 
    {
      finalresult->FillComplete(B.DomainMap(),A.RangeMap());
      finalresult->OptimizeStorage();
    }
    result = null;
    return rcp(new SparseMatrix(finalresult,explicitdirichlet,savegraph));
#endif
  }
  return rcp(new SparseMatrix(result,explicitdirichlet,savegraph));
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
/*
static void CopySortDeleteZeros(const Epetra_CrsMatrix& A, Epetra_CrsMatrix& As)
{
  vector<int>    scindices(A.MaxNumEntries());
  vector<double> scvalues(A.MaxNumEntries());
  for (int i=0; i<A.NumMyRows(); ++i)
  {
    int grid = A.RowMap().GID(i);
    int numentries;
    double* values;
    int* indices;
    A.ExtractMyRowView(i,numentries,values,indices);
    int snumentries=0;
    for (int j=0; j<numentries; ++j)
    {
      if (values[j]==0.0) continue;
      scindices[snumentries] = A.ColMap().GID(indices[j]);
      scvalues[snumentries] = values[j];
      snumentries++;
    }
    ML_az_sort(&scindices[0],snumentries,NULL,&scvalues[0]);
    int err = As.InsertGlobalValues(grid,snumentries,&scvalues[0],&scindices[0]);
    if (err) dserror("Epetra_CrsMatrix::InsertGlobalValues returned err=%d",err);
  }
  if (A.Filled()) As.FillComplete(A.DomainMap(),A.RangeMap(),true);
  return;
}
*/


#endif  // #ifdef CCADISCRET
