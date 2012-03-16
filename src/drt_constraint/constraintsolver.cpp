/*!----------------------------------------------------------------------
\file constraintsolver.cpp

\brief Class containing uzawa algorithm to solve linear system.

<pre>
Maintainer: Thomas Kloeppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/kloeppel
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <stdio.h>
#include <iostream>

#include "constraintsolver.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_utils.H"


/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 11/07|
 *----------------------------------------------------------------------*/
UTILS::ConstraintSolver::ConstraintSolver
(
  RCP<DRT::Discretization> discr,
  LINALG::Solver& solver,
  RCP<Epetra_Vector> dirichtoggle,
  RCP<Epetra_Vector> invtoggle,
  ParameterList params
):
actdisc_(discr),
maxIter_(params.get<int>   ("UZAWAMAXITER", 50)),
dirichtoggle_(dirichtoggle),
dbcmaps_(Teuchos::null)
{
  dbcmaps_ = LINALG::ConvertDirichletToggleVectorToMaps(dirichtoggle);
  Setup(discr,solver,dbcmaps_,params);
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 11/07|
 *----------------------------------------------------------------------*/
UTILS::ConstraintSolver::ConstraintSolver
(
  RCP<DRT::Discretization> discr,
  LINALG::Solver& solver,
  RCP<LINALG::MapExtractor> dbcmaps,
  ParameterList params
):
actdisc_(discr),
maxIter_(params.get<int>   ("UZAWAMAXITER", 50)),
dirichtoggle_(Teuchos::null),
dbcmaps_(dbcmaps)
{
  Setup(discr,solver,dbcmaps,params);
}

/*----------------------------------------------------------------------*
 |  set-up (public)                                             tk 11/07|
 *----------------------------------------------------------------------*/
void UTILS::ConstraintSolver::Setup
(
  RCP<DRT::Discretization> discr,
  LINALG::Solver& solver,
  RCP<LINALG::MapExtractor> dbcmaps,
  ParameterList params
)
{

  solver_ = rcp(&solver,false);

  algochoice_ = DRT::INPUT::IntegralValue<INPAR::STR::ConSolveAlgo>(params,"UZAWAALGO");

  // different setup for #adapttol_
  isadapttol_ = true;
  isadapttol_ = (DRT::INPUT::IntegralValue<int>(params,"ADAPTCONV") == 1);
  
  // simple parameters
  adaptolbetter_ = params.get<double>("ADAPTCONV_BETTER", 0.01);
  iterationparam_ = params.get<double>("UZAWAPARAM", 1);
  minparam_ = iterationparam_*1E-3;
  iterationtol_ = params.get<double>("UZAWATOL", 1E-8);


  counter_ = 0;
  return;
}



/*----------------------------------------------------------------------*
|(public)                                                               |
|Solve linear constrained system                                        |
*-----------------------------------------------------------------------*/
void UTILS::ConstraintSolver::Solve
(
  RCP<LINALG::SparseMatrix> stiff,
  RCP<LINALG::SparseMatrix> constr,
  RCP<LINALG::SparseMatrix> constrT,
  RCP<Epetra_Vector> dispinc,
  RCP<Epetra_Vector> lagrinc,
  const RCP<Epetra_Vector> rhsstand,
  const RCP<Epetra_Vector> rhsconstr
)
{
  switch (algochoice_)
  {
    case INPAR::STR::consolve_uzawa:
      SolveUzawa(stiff,constr,constrT,dispinc,lagrinc,rhsstand,rhsconstr);
    break;
    case INPAR::STR::consolve_direct:
      SolveDirect(stiff,constr,constrT,dispinc,lagrinc,rhsstand,rhsconstr);
    break;
    case INPAR::STR::consolve_simple:
      SolveSimple(stiff,constr,constrT,dispinc,lagrinc,rhsstand,rhsconstr);
    break;
    default :
      dserror("Unknown constraint solution technique!");
  }
  return;
}

/*----------------------------------------------------------------------*
|(public)                                                               |
|Solve linear constrained system by iterative Uzawa algorithm           |
*-----------------------------------------------------------------------*/
void UTILS::ConstraintSolver::SolveUzawa
(
  RCP<LINALG::SparseMatrix> stiff,
  RCP<LINALG::SparseMatrix> constr,
  RCP<LINALG::SparseMatrix> constrT,
  RCP<Epetra_Vector> dispinc,
  RCP<Epetra_Vector> lagrinc,
  const RCP<Epetra_Vector> rhsstand,
  const RCP<Epetra_Vector> rhsconstr
)
{
  const int myrank=(actdisc_->Comm().MyPID());
  // For every iteration step an uzawa algorithm is used to solve the linear system.
  //Preparation of uzawa method to solve the linear system.
  double norm_uzawa;
  double norm_uzawa_old;
  double quotient;
  double norm_constr_uzawa;
  int numiter_uzawa = 0;
  //counter used for adaptivity
  const int adaptstep = 2;
  const int minstep = 1;
  int count_paramadapt = 1;

  const double computol = 1E-8;

  RCP<Epetra_Vector> constrTLagrInc = rcp(new Epetra_Vector(rhsstand->Map()));
  RCP<Epetra_Vector> constrTDispInc = rcp(new Epetra_Vector(rhsconstr->Map()));
  //LINALG::SparseMatrix constrT = *(Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(constr));

  // ONLY compatability
  // dirichtoggle_ changed and we need to rebuild associated DBC maps
  if (dirichtoggle_ != Teuchos::null)
    dbcmaps_ = LINALG::ConvertDirichletToggleVectorToMaps(dirichtoggle_);

  RCP<Epetra_Vector> zeros = rcp(new Epetra_Vector(rhsstand->Map(),true));
  RCP<Epetra_Vector> dirichzeros = dbcmaps_->ExtractCondVector(zeros);

  // Compute residual of the uzawa algorithm
  RCP<Epetra_Vector> fresmcopy=rcp(new Epetra_Vector(*rhsstand));
  Epetra_Vector uzawa_res(*fresmcopy);
  (*stiff).Multiply(false,*dispinc,uzawa_res);
  uzawa_res.Update(1.0,*fresmcopy,-1.0);

  // blank residual DOFs which are on Dirichlet BC
  dbcmaps_->InsertCondVector(dirichzeros, Teuchos::rcp(&uzawa_res,false));

  uzawa_res.Norm2(&norm_uzawa);
  Epetra_Vector constr_res(lagrinc->Map());

  constr_res.Update(1.0,*(rhsconstr),0.0);
  constr_res.Norm2(&norm_constr_uzawa);
  quotient =1;
  //Solve one iteration step with augmented lagrange
  //Since we calculate displacement norm as well, at least one step has to be taken
  while (((norm_uzawa > iterationtol_ or norm_constr_uzawa > iterationtol_) and numiter_uzawa < maxIter_)
      or numiter_uzawa < minstep)
  {
    //LINALG::ApplyDirichlettoSystem(dispinc,fresmcopy,zeros,*(dbcmaps_->CondMap()));
//    constr->ApplyDirichlet(*(dbcmaps_->CondMap()),false);

    #if 0
    const double cond_number = LINALG::Condest(static_cast<LINALG::SparseMatrix&>(*stiff),Ifpack_GMRES, 1000);
    // computation of significant digits might be completely bogus, so don't take it serious
    const double tmp = std::abs(std::log10(cond_number*1.11022e-16));
    const int sign_digits = (int)floor(tmp);
    if (!myrank)
      cout << " cond est: " << scientific << cond_number << ", max.sign.digits: " << sign_digits;
#endif

    // solve for disi
    // Solve K . IncD = -R  ===>  IncD_{n+1}
    if (isadapttol_ && counter_ && numiter_uzawa)
    {
      double worst = norm_uzawa;
      double wanted = tolres_/10.0;
      solver_->AdaptTolerance(wanted,worst,adaptolbetter_);
    }
    solver_->Solve(stiff->EpetraMatrix(),dispinc,fresmcopy,true,numiter_uzawa==0 && counter_==0);
    solver_->ResetTolerance();

    //compute Lagrange multiplier increment
    constrTDispInc->PutScalar(0.0);
    constrT->Multiply(true,*dispinc,*constrTDispInc) ;
    lagrinc->Update(iterationparam_,*constrTDispInc,iterationparam_,*rhsconstr,1.0);

    //Compute residual of the uzawa algorithm
    constr->Multiply(false,*lagrinc,*constrTLagrInc);

    fresmcopy->Update(-1.0,*constrTLagrInc,1.0,*rhsstand,0.0);
    Epetra_Vector uzawa_res(*fresmcopy);
    (*stiff).Multiply(false,*dispinc,uzawa_res);
    uzawa_res.Update(1.0,*fresmcopy,-1.0);
    
    // blank residual DOFs which are on Dirichlet BC
    dbcmaps_->InsertCondVector(dirichzeros, Teuchos::rcp(&uzawa_res,false));
    norm_uzawa_old=norm_uzawa;
    uzawa_res.Norm2(&norm_uzawa);
    Epetra_Vector constr_res(lagrinc->Map());

    constr_res.Update(1.0,*constrTDispInc,1.0,*rhsconstr,0.0);
    constr_res.Norm2(&norm_constr_uzawa);
    //-------------Adapt Uzawa parameter--------------
    // For a constant parameter the quotient of two successive residual norms
    // stays nearly constant during the computation. So this quotient seems to be a good
    // measure for the parameter choice
    // Adaptivity only takes place every second step. Otherwise the quotient is not significant.
    if (count_paramadapt>=adaptstep)
    {
      double quotient_new=norm_uzawa/norm_uzawa_old;
      // In case of divergence the parameter must be too high
      if (quotient_new>(1.+computol))
      {
        if (iterationparam_>2.*minparam_)
          iterationparam_ = iterationparam_/2.;
        quotient=1;
      }
      else
      {
        // In case the newly computed quotient is better than the one obtained from the
        // previous parameter, the parameter is increased by a factor (1+quotient_new)
        if (quotient>=quotient_new)
        {
          iterationparam_=iterationparam_*(1.+quotient_new);
          quotient=quotient_new;
        }
        // In case the newly computed quotient is worse than the one obtained from the
        // previous parameter, the parameter is decreased by a factor 1/(1+quotient_new)
        else
        {
          if (iterationparam_>2.*minparam_)
            iterationparam_=iterationparam_/(1.+quotient_new);
          quotient=quotient_new;
        }
      }

      if (iterationparam_<=minparam_)
      {
        if (!myrank)
          cout<<"leaving uzawa loop since Uzawa parameter is too low"<<endl;
        iterationparam_*=1E2;
        break;
      }
      count_paramadapt=0;
    }
    count_paramadapt++;
    numiter_uzawa++;
  } //Uzawa loop

  if (!myrank)
  {
     cout<<"Uzawa steps "<<numiter_uzawa<<", Uzawa parameter: "<< iterationparam_;
     cout<<", residual norms for linear system: "<< norm_constr_uzawa<<" and "<<norm_uzawa<<endl;
  }
  counter_++;
  return;
}

/*----------------------------------------------------------------------*
|(public)                                                               |
|Solve linear constrained system by iterative Uzawa algorithm           |
*-----------------------------------------------------------------------*/
void UTILS::ConstraintSolver::SolveDirect
(
  RCP<LINALG::SparseMatrix> stiff,
  RCP<LINALG::SparseMatrix> constr,
  RCP<LINALG::SparseMatrix> constrT,
  RCP<Epetra_Vector> dispinc,
  RCP<Epetra_Vector> lagrinc,
  const RCP<Epetra_Vector> rhsstand,
  const RCP<Epetra_Vector> rhsconstr
)
{
  // define maps of standard dofs and additional lagrange multipliers
  RCP<Epetra_Map> standrowmap = rcp(new Epetra_Map(stiff->RowMap()));
  RCP<Epetra_Map> conrowmap = rcp(new Epetra_Map(constr->DomainMap()));
  // merge maps to one large map
  RCP<Epetra_Map> mergedmap = LINALG::MergeMap(standrowmap,conrowmap,false);
  // define MapExtractor
  LINALG::MapExtractor mapext(*mergedmap,standrowmap,conrowmap);

  // initialize large Sparse Matrix and Epetra_Vectors
  RCP<LINALG::SparseMatrix> mergedmatrix = rcp(new LINALG::SparseMatrix(*mergedmap,81));
  RCP<Epetra_Vector> mergedrhs = rcp(new Epetra_Vector(*mergedmap));
  RCP<Epetra_Vector> mergedsol = rcp(new Epetra_Vector(*mergedmap));
  // ONLY compatability
  // dirichtoggle_ changed and we need to rebuild associated DBC maps
  if (dirichtoggle_ != Teuchos::null)
    dbcmaps_ = LINALG::ConvertDirichletToggleVectorToMaps(dirichtoggle_);
  // fill merged matrix using Add
  mergedmatrix -> Add(*stiff,false,1.0,1.0);
  mergedmatrix -> Add(*constr,false,1.0,1.0);
  mergedmatrix -> Add(*constrT,true,1.0,1.0);
  mergedmatrix -> Complete(*mergedmap,*mergedmap);
  // fill merged vectors using Export
  LINALG::Export(*rhsconstr,*mergedrhs);
  mergedrhs -> Scale(-1.0);
  LINALG::Export(*rhsstand,*mergedrhs);

#if 0
    const int myrank=(actdisc_->Comm().MyPID());
    const double cond_number = LINALG::Condest(static_cast<LINALG::SparseMatrix&>(*mergedmatrix),Ifpack_GMRES, 100);
    // computation of significant digits might be completely bogus, so don't take it serious
    const double tmp = std::abs(std::log10(cond_number*1.11022e-16));
    const int sign_digits = (int)floor(tmp);
    if (!myrank)
      cout << " cond est: " << scientific << cond_number << ", max.sign.digits: " << sign_digits<<endl;
#endif

  // solve
  solver_->Solve(mergedmatrix->EpetraMatrix(),mergedsol,mergedrhs,true,counter_==0);
  solver_->ResetTolerance();
  // store results in smaller vectors
  mapext.ExtractCondVector(mergedsol,dispinc);
  mapext.ExtractOtherVector(mergedsol,lagrinc);

  counter_++;
  return;
}

void UTILS::ConstraintSolver::SolveSimple
(
  RCP<LINALG::SparseMatrix> stiff,
  RCP<LINALG::SparseMatrix> constr,
  RCP<LINALG::SparseMatrix> constrT,
  RCP<Epetra_Vector> dispinc,
  RCP<Epetra_Vector> lagrinc,
  const RCP<Epetra_Vector> rhsstand,
  const RCP<Epetra_Vector> rhsconstr
)
{
  // row maps (assumed to equal to range map) and extractor
  RCP<Epetra_Map> standrowmap = rcp(new Epetra_Map(stiff->RowMap()));
  RCP<Epetra_Map> conrowmap = rcp(new Epetra_Map(constr->DomainMap()));
  RCP<Epetra_Map> mergedrowmap = LINALG::MergeMap(standrowmap,conrowmap,false);
  LINALG::MapExtractor rowmapext(*mergedrowmap,conrowmap,standrowmap);
  
  // domain maps and extractor
  RCP<Epetra_Map> standdommap = rcp(new Epetra_Map(stiff->DomainMap()));
  RCP<Epetra_Map> condommap = rcp(new Epetra_Map(constr->DomainMap()));
  RCP<Epetra_Map> mergeddommap = LINALG::MergeMap(standdommap,condommap,false);
  LINALG::MapExtractor dommapext(*mergeddommap,condommap,standdommap);
  
  // cast constraint operators to matrices and save transpose of constraint matrix
  LINALG::SparseMatrix constrTrans (*conrowmap,81,false,true);
  constrTrans.Add(*constrT,true,1.0,0.0);
  constrTrans.Complete(constrT->RangeMap(),constrT->DomainMap());
  
  // ONLY compatability
  // dirichtoggle_ changed and we need to rebuild associated DBC maps
  if (dirichtoggle_ != Teuchos::null)
    dbcmaps_ = LINALG::ConvertDirichletToggleVectorToMaps(dirichtoggle_);

  // stuff needed for Dirichlet BCs
  RCP<Epetra_Vector> zeros = rcp(new Epetra_Vector(rhsstand->Map(),true));
  RCP<Epetra_Vector> dirichzeros = dbcmaps_->ExtractCondVector(zeros);
  RCP<Epetra_Vector> rhscopy=rcp(new Epetra_Vector(*rhsstand));
  
  //make solver CheapSIMPLE-ready
  Teuchos::ParameterList sfparams = solver_->Params();  // save copy of original solver parameter list
  solver_->Params() = LINALG::Solver::TranslateSolverParameters(DRT::Problem::Instance()->ContactSolverParams());
  if(!solver_->Params().isSublist("Aztec Parameters") &&
     !solver_->Params().isSublist("Belos Parameters"))
  {
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    cout << "You need a \'CONTACT SOLVER\' block within your dat file with either \'Aztec_MSR\' or \'Belos\' as SOLVER." << endl;
    cout << "The \'STRUCT SOLVER\' block is then used for the primary inverse within CheapSIMPLE and the \'FLUID PRESSURE SOLVER\' " << endl;
    cout << "block for the constraint block" << endl;
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    dserror("Please edit your dat file");
  }
  solver_->Params().set<bool>("CONSTRAINT",true);      // handling of constraint null space within Simple type preconditioners
  solver_->Params().sublist("CheapSIMPLE Parameters"); // this automatically sets preconditioner to CheapSIMPLE!
  solver_->Params().sublist("Inverse1") = sfparams;
  solver_->PutSolverParamsToSubParams("Inverse2",
      DRT::Problem::Instance()->FluidPressureSolverParams());

  //build block matrix for SIMPLE
  Teuchos::RCP<LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy> > mat=
      rcp(new LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>(dommapext,rowmapext,81,false,false));
  mat->Assign(0,0,View,*stiff);
  mat->Assign(0,1,View,*constr);
  mat->Assign(1,0,View,constrTrans);
  mat->Complete();
  
  // merged rhs using Export
  RCP<Epetra_Vector> mergedrhs = rcp(new Epetra_Vector(*mergedrowmap));
  LINALG::Export(*rhsconstr,*mergedrhs);
  mergedrhs -> Scale(-1.0);
  LINALG::Export(*rhscopy,*mergedrhs);
  
  // solution vector
  RCP<Epetra_Vector> mergedsol = rcp(new Epetra_Vector(*mergedrowmap));

  // solve
  solver_->Solve(mat->EpetraOperator(),mergedsol,mergedrhs,true,counter_==0);
  solver_->ResetTolerance();
  solver_->Params() = sfparams; // store back original parameter list

  // store results in smaller vectors
  rowmapext.ExtractCondVector(mergedsol,lagrinc);
  rowmapext.ExtractOtherVector(mergedsol,dispinc);
  
  counter_++;
  return;  
}


#endif
