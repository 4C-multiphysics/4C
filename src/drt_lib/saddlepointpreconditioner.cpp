/*
 * saddlepointpreconditioner.cpp
 *
 *  Created on: Feb 16, 2010
 *      Author: wiesner
 */

#ifdef CCADISCRET


#define WRITEOUTSTATISTICS
#undef WRITEOUTSYMMETRY // write out || A - A^T ||_F
#define WRITEOUTAGGREGATES

#include <Epetra_LocalMap.h>

#include "linalg_sparsematrix.H"

#include "saddlepointpreconditioner.H"
#include "transfer_operator.H"
#include "transfer_operator_tentative.H"
#include "transfer_operator_saamg.H"
#include "transfer_operator_pgamg.H"
#include "braesssarazin_smoother.H"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_TimeMonitor.hpp"


// includes for MLAPI functions
#include "ml_common.h"
#include "ml_include.h"
#include "ml_aggregate.h"
#include "ml_agg_METIS.h"
#include "Teuchos_RefCountPtr.hpp"
#include "MLAPI_Error.h"
#include "MLAPI_Expressions.h"
#include "MLAPI_Space.h"
#include "MLAPI_Operator.h"
#include "MLAPI_Workspace.h"
#include "MLAPI_Aggregation.h"
#include "MLAPI.h"

#include "float.h" // for DBL_MAX and DBL_MIN

LINALG::SaddlePointPreconditioner::SaddlePointPreconditioner(RCP<Epetra_Operator> A, const ParameterList& params, const ParameterList& pressurelist, FILE* outfile)
: params_(params),
pressureparams_(pressurelist),
outfile_(outfile),
nVerbose_(0)
{
  Setup(A,params,pressurelist);
}

LINALG::SaddlePointPreconditioner::~SaddlePointPreconditioner()
{

}


int LINALG::SaddlePointPreconditioner::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  // VCycle

  // note: Aztec might pass X and Y as physically identical objects,
  // so we better deep copy here

  RCP<LINALG::ANA::Vector> Xv = rcp(new LINALG::ANA::Vector(*mmex_.Map(0),false));
  RCP<LINALG::ANA::Vector> Xp = rcp(new LINALG::ANA::Vector(*mmex_.Map(1),false));

  RCP<LINALG::ANA::Vector> Yv = rcp(new LINALG::ANA::Vector(*mmex_.Map(0),false));
  RCP<LINALG::ANA::Vector> Yp = rcp(new LINALG::ANA::Vector(*mmex_.Map(1),false));


  // split vector using mmex_
  mmex_.ExtractVector(X,0,*Xv);
  mmex_.ExtractVector(X,1,*Xp);

  VCycle(*Xv,*Xp,*Yv,*Yp,0);

  mmex_.InsertVector(*Yv,0,Y);
  mmex_.InsertVector(*Yp,1,Y);

  return 0;
}

int LINALG::SaddlePointPreconditioner::VCycle(const Epetra_MultiVector& Xvel, const Epetra_MultiVector& Xpre, Epetra_MultiVector& Yvel, Epetra_MultiVector& Ypre, const int level) const
{
  // Y = A_^{-1} * X => solve A*Y = X

  if (level == nlevels_)
  {
    // coarsest level
    coarsestSmoother_->ApplyInverse(Xvel,Xpre,Yvel,Ypre);

    return 0;
  }

  // vectors for presmoothed solution
  RCP<Epetra_MultiVector> Zvel = rcp(new Epetra_MultiVector(Yvel.Map(),1,true));
  RCP<Epetra_MultiVector> Zpre = rcp(new Epetra_MultiVector(Ypre.Map(),1,true));

  // presmoothing
  if(bPresmoothing_) preS_[level]->ApplyInverse(Xvel,Xpre,*Zvel,*Zpre);  // rhs X is fix, initial solution Z = 0 (per definition, see above)
                                                      // note: ApplyInverse expects the "solution" and no solution increment "Delta Z"

  // calculate residual (fine grid)
  RCP<Epetra_Vector> velres = rcp(new Epetra_Vector(Yvel.Map(),true));
  RCP<Epetra_Vector> preres = rcp(new Epetra_Vector(Ypre.Map(),true));

  RCP<Epetra_Vector> vtemp = rcp(new Epetra_Vector(Yvel.Map(),true));
  RCP<Epetra_Vector> ptemp = rcp(new Epetra_Vector(Ypre.Map(),true));

  A11_[level]->Apply(*Zvel,*vtemp);
  A12_[level]->Apply(*Zpre,*velres);
  velres->Update(1.0,*vtemp,1.0);  // velres = + F Zvel + G Zpre
  velres->Update(1.0,Xvel,-1.0); // velres = Xvel - F Zvel - G Zpre

  A21_[level]->Apply(*Zvel,*ptemp);
  A22_[level]->Apply(*Zpre,*preres);
  preres->Update(1.0,*ptemp,1.0); // preres = + D Zvel + Z Zpre
  preres->Update(1.0,Xpre,-1.0); // preres = Xpre - D Zvel - Z Zpre

  // calculate coarse residual
  RCP<Epetra_Vector> velres_coarse = rcp(new Epetra_Vector(Tvel_[level]->R().RowMap(),true));
  RCP<Epetra_Vector> preres_coarse = rcp(new Epetra_Vector(Tpre_[level]->R().RowMap(),true));
  Tvel_[level]->R().Apply(*velres,*velres_coarse);
  Tpre_[level]->R().Apply(*preres,*preres_coarse);

  // define vector for coarse level solution
  RCP<Epetra_Vector> velsol_coarse = rcp(new Epetra_Vector(A11_[level+1]->RowMap(),true));
  RCP<Epetra_Vector> presol_coarse = rcp(new Epetra_Vector(A22_[level+1]->RowMap(),true));

  // call Vcycle recursively
  VCycle(*velres_coarse,*preres_coarse,*velsol_coarse,*presol_coarse,level+1);

  // define vectors for prolongated solution
  RCP<Epetra_Vector> velsol_prolongated = rcp(new Epetra_Vector(A11_[level]->RowMap(),true));
  RCP<Epetra_Vector> presol_prolongated = rcp(new Epetra_Vector(A22_[level]->RowMap(),true));

  // prolongate solution
  Tvel_[level]->P().Apply(*velsol_coarse,*velsol_prolongated);
  Tpre_[level]->P().Apply(*presol_coarse,*presol_prolongated);

  // update solution Zvel and Zpre for postsmoother
  Zvel->Update(1.0,*velsol_prolongated,1.0);
  Zpre->Update(1.0,*presol_prolongated,1.0);

  // postsmoothing
  if (bPostsmoothing_) postS_[level]->ApplyInverse(Xvel,Xpre,*Zvel,*Zpre); // rhs the same as for presmoothing, but better initial solution (Z)

  // write out solution
  Yvel.Update(1.0,*Zvel,0.0);
  Ypre.Update(1.0,*Zpre,0.0);

  return 0;
}

void LINALG::SaddlePointPreconditioner::Setup(RCP<Epetra_Operator> A,const ParameterList& origlist,const ParameterList& origplist)
{

#ifdef WRITEOUTSTATISTICS
  Epetra_Time ttt(A->Comm());
  ttt.ResetStartTime();
#endif

  // SETUP with SparseMatrix base class
  //////////////////// define some variables
  //const int myrank = A->Comm().MyPID();
  Epetra_Time time(A->Comm());
  const Epetra_Map& fullmap = A->OperatorRangeMap();
  const int         length  = fullmap.NumMyElements();
  nVerbose_ = 0;      // level of verbosity
  int ndofpernode = 0;// number of dofs per node
  int nv = 0;         // number of velocity dofs per node
  int np = 0;         // number of pressure dofs per node (1)
  int nlnode;         // number of nodes (local)


  Teuchos::RCP<Epetra_MultiVector> curvelNS = null;   // variables for null space
  Teuchos::RCP<Epetra_MultiVector> nextvelNS = null;
  Teuchos::RCP<Epetra_MultiVector> curpreNS = null;
  Teuchos::RCP<Epetra_MultiVector> nextpreNS = null;

  ///////////////// set parameter list
  RCP<ParameterList> spparams = rcp(new ParameterList());     // all paramaters
  RCP<ParameterList> velparams = rcp(new ParameterList());    // parameters (velocity specific)
  RCP<ParameterList> preparams = rcp(new ParameterList());    // parameters (pressure specific)

  // obtain common ML parameters from FLUID SOLVER block for coarsening from the dat file
  // we need at least "ML Parameters"."PDE equations" and "nullspace" information
  spparams->sublist("AMGBS Parameters") = params_.sublist("AMGBS Parameters"); // copy common parameters

  // first and most important: we need the number of PDE equations
  // we extract this from the Aztec Parameters and the downwind nv parameter there
  if(!params_.isSublist("Aztec Parameters")) dserror ("we expect Aztec Parameters, but there are none" );

  // this is somewhat suspicious
  int nPDE = params_.sublist("Aztec Parameters").get("downwinding nv",3);
  if(nPDE == 2) nPDE = 3;   // i don't know why downwinding nv is 2 in 2d and 4 in 3d, thats somewhat stupid. i think nv=3 in 3d would be better?? (it's meant to be the number of PDEs per dof)
  spparams->sublist("AMGBS Parameters").set("PDE equations",nPDE);
  spparams->sublist("AMGBS Parameters").set("null space: dimension",params_.sublist("AMGBS Parameters").get("PDE equations",3)); // copy the PDE equations as nullspace dimension

  spparams->sublist("AMGBS Parameters").set("null space: add default vectors",params_.sublist("ML Parameters").get("null space: add default vectors",false));


  spparams->sublist("AMGBS Parameters").set("ML output",spparams->sublist("AMGBS Parameters").get("output",0)); // set ML output
  spparams->sublist("AMGBS Parameters").remove("output");
  spparams->sublist("AMGBS Parameters").remove("smoother: type");  // we're using Braess-Sarazin only

  params_.remove("ML Parameters",false);  // now we don't need the ML Parameters any more


  /////////////////// prepare variables
  nmaxlevels_ = spparams->sublist("AMGBS Parameters").get("max levels",6) - 1;
  nlevels_ = 0;       // no levels defined
  bPresmoothing_ = false;   // get flags for pre- and postsmoothing
  bPostsmoothing_ = false;
  if(spparams->sublist("AMGBS Parameters").get("amgbs: smoother: pre or post","both") == "both" ||
     spparams->sublist("AMGBS Parameters").get("amgbs: smoother: pre or post","both") == "pre")
    bPresmoothing_ = true;
  if(spparams->sublist("AMGBS Parameters").get("amgbs: smoother: pre or post","both") == "both" ||
     spparams->sublist("AMGBS Parameters").get("amgbs: smoother: pre or post","both") == "post")
    bPostsmoothing_ = true;
  A11_.resize(nmaxlevels_+1);
  A12_.resize(nmaxlevels_+1);
  A21_.resize(nmaxlevels_+1);
  A22_.resize(nmaxlevels_+1);
  preS_.resize(nmaxlevels_);    // smoothers
  postS_.resize(nmaxlevels_);
  Tvel_.resize(nmaxlevels_);    // transfer operators
  Tpre_.resize(nmaxlevels_);

  int nmaxcoarsedim = spparams->sublist("AMGBS Parameters").get("max coarse dimension",20);
  nVerbose_ = spparams->sublist("AMGBS Parameters").get("ML output",0);
  ndofpernode = spparams->sublist("AMGBS Parameters").get<int>("PDE equations",0);
  if(ndofpernode == 0) dserror("dof per node is zero -> error");

  nv       = ndofpernode-1;
  np       = 1;
  nlnode   = length / ndofpernode;



  /////////////////// transform Input matrix
  Ainput_ = rcp_dynamic_cast<BlockSparseMatrixBase>(A);
  if(Ainput_ != null)
  {
    mmex_ = Ainput_->RangeExtractor();
  }
  else
  {
    // get # dofs per node from params_ list and split row map
    time.ResetStartTime();
    vector<int> vgid(nlnode*nv);
    vector<int> pgid(nlnode);
    int vcount=0;
    for (int i=0; i<nlnode; ++i)
    {
      for (int j=0; j<ndofpernode-1; ++j)
        vgid[vcount++] = fullmap.GID(i*ndofpernode+j);
      pgid[i] = fullmap.GID(i*ndofpernode+ndofpernode-1);
    }
    vector<RCP<const Epetra_Map> > maps(2);
    maps[0] = rcp(new Epetra_Map(-1,nlnode*nv,&vgid[0],0,fullmap.Comm()));
    maps[1] = rcp(new Epetra_Map(-1,nlnode,&pgid[0],0,fullmap.Comm()));
    vgid.clear(); pgid.clear();
    mmex_.Setup(fullmap,maps);
    //if (!myrank /*&& SIMPLER_TIMING*/) printf("--- Time to split map       %10.3E\n",time.ElapsedTime());
    time.ResetStartTime();
    // wrap matrix in SparseMatrix and split it into 2x2 BlockMatrix
    {
      SparseMatrix fullmatrix(rcp_dynamic_cast<Epetra_CrsMatrix>(A));
      Ainput_ = fullmatrix.Split<LINALG::DefaultBlockMatrixStrategy>(mmex_,mmex_);
      //if (!myrank /*&& SIMPLER_TIMING*/) printf("--- Time to split matrix    %10.3E\n",time.ElapsedTime());
      time.ResetStartTime();
      Ainput_->Complete();
      //if (!myrank /*&& SIMPLER_TIMING*/) printf("--- Time to complete matrix %10.3E\n",time.ElapsedTime());
      time.ResetStartTime();
    }
  }

  /////////////////// prepare null space for finest level (splitted into velocity and pressure part)

  // velocity part: fill in parameter list
  velparams->sublist("AMGBS Parameters") = spparams->sublist("AMGBS Parameters"); // copy common parameters
  velparams->sublist("AMGBS Parameters").set("PDE equations",nv);             // adapt nPDE (only velocity dofs)
  velparams->sublist("AMGBS Parameters").set("null space: dimension",nv);
  const int vlength = (*Ainput_)(0,0).RowMap().NumMyElements();
  RCP<vector<double> > vnewns = rcp(new vector<double>(nv*vlength,0.0));
  for (int i=0; i<nlnode; ++i)
  {
    (*vnewns)[i*nv] = 1.0;
    (*vnewns)[vlength+i*nv+1] = 1.0;
    if (nv>2) (*vnewns)[2*vlength+i*nv+2] = 1.0;
  }
  velparams->sublist("AMGBS Parameters").set("null space: vectors",&((*vnewns)[0])); // adapt default null space
  velparams->sublist("AMGBS Parameters").remove("nullspace",false);

  curvelNS = rcp(new Epetra_MultiVector(View,(*Ainput_)(0,0).RowMap(),&((*vnewns)[0]),(*Ainput_)(0,0).EpetraMatrix()->RowMatrixRowMap().NumMyElements(),nv));


  // pressure part: fill parameter list
  preparams->sublist("AMGBS Parameters") = spparams->sublist("AMGBS Parameters");
  preparams->sublist("AMGBS Parameters").set("PDE equations",1);               // adapt nPDE (only one pressure dof)
  preparams->sublist("AMGBS Parameters").set("null space: dimension", 1);
  const int plength = (*Ainput_)(1,1).RowMap().NumMyElements();
  RCP<vector<double> > pnewns = rcp(new vector<double>(plength,1.0));
  preparams->sublist("AMGBS Parameters").set("null space: vectors",&((*pnewns)[0]));
  preparams->sublist("AMGBS Parameters").remove("nullspace",false);

  curpreNS = rcp(new Epetra_MultiVector(View,(*Ainput_)(1,1).RowMap(),&((*pnewns)[0]),(*Ainput_)(1,1).EpetraMatrix()->RowMatrixRowMap().NumMyElements(),1));

  ////////////////// store level 0 matrices (finest level)
  int curlevel = 0;

  A11_[curlevel] = rcp(new SparseMatrix(Ainput_->Matrix(0,0),Copy));    // check me: copy or view only??
  A12_[curlevel] = rcp(new SparseMatrix(Ainput_->Matrix(0,1),Copy));
  A21_[curlevel] = rcp(new SparseMatrix(Ainput_->Matrix(1,0),Copy));
  A22_[curlevel] = rcp(new SparseMatrix(Ainput_->Matrix(1,1),Copy));

  MLAPI::Init();

  for (curlevel = 0; curlevel < nmaxlevels_; ++curlevel)
  {
    /////////////////////////////////////////////////////////
    /////////////////////// AGGREGATION PROCESS
    RCP<Epetra_IntVector> velaggs = rcp(new Epetra_IntVector(A11_[curlevel]->RowMap(),true));
    RCP<Epetra_IntVector> preaggs = rcp(new Epetra_IntVector(A22_[curlevel]->RowMap(),true));

    ////////////// determine aggregates using the velocity block matrix A11_[curlevel]
    int naggregates_local = 0;
    int naggregates = GetGlobalAggregates(*A11_[curlevel],velparams->sublist("AMGBS Parameters"),*curvelNS,*velaggs,naggregates_local);

    ////////////// transform vector with velocity aggregates to pressure block
    for(int i=0; i < preaggs->MyLength(); i++)
    {
      (*preaggs)[i] = (*velaggs)[i*nv];
    }

    /////////////////////////////////////////////////////////
    /////////////////////// CALCULATE TRANSFER OPERATORS

    ///////////// velocity transfer operators
    string velProlongSmoother = velparams->sublist("AMGBS Parameters").get("amgbs: prolongator smoother (vel)","PA-AMG");
    Tvel_[curlevel] = TransferOperatorFactory::Create(velProlongSmoother,A11_[curlevel],NULL); /* outfile */
    nextvelNS = Tvel_[curlevel]->buildTransferOperators(velaggs,naggregates_local,velparams->sublist("AMGBS Parameters"),curvelNS,0);



    //////////// pressure transfer operators
    string preProlongSmoother = preparams->sublist("AMGBS Parameters").get("amgbs: prolongator smoother (pre)","PA-AMG");
    Tpre_[curlevel] = TransferOperatorFactory::Create(preProlongSmoother,A22_[curlevel],NULL); /* outfile */
    nextpreNS = Tpre_[curlevel]->buildTransferOperators(preaggs,naggregates_local,preparams->sublist("AMGBS Parameters"),curpreNS,naggregates*nv);

    if(nVerbose_ > 4) // be verbose
    {
      cout << "Pvel[" << curlevel << "]: " << Tvel_[curlevel]->Prolongator()->EpetraMatrix()->NumGlobalRows() << " x " << Tvel_[curlevel]->Prolongator()->EpetraMatrix()->NumGlobalCols() << " (" << Tvel_[curlevel]->Prolongator()->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;
      cout << "Ppre[" << curlevel << "]: " << Tpre_[curlevel]->Prolongator()->EpetraMatrix()->NumGlobalRows() << " x " << Tpre_[curlevel]->Prolongator()->EpetraMatrix()->NumGlobalCols() << " (" << Tpre_[curlevel]->Prolongator()->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;

      cout << "Rvel[" << curlevel << "]: " << Tvel_[curlevel]->Restrictor()->EpetraMatrix()->NumGlobalRows() << " x " << Tvel_[curlevel]->Restrictor()->EpetraMatrix()->NumGlobalCols() << " (" << Tvel_[curlevel]->Restrictor()->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;
      cout << "Rpre[" << curlevel << "]: " << Tpre_[curlevel]->Restrictor()->EpetraMatrix()->NumGlobalRows() << " x " << Tpre_[curlevel]->Restrictor()->EpetraMatrix()->NumGlobalCols() << " (" << Tpre_[curlevel]->Restrictor()->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;
    }

#ifdef WRITEOUTAGGREGATES
//    std::ofstream fileout;
//    std::stringstream fileoutstream;
//    fileoutstream << "/home/wiesner/fluid/cubits/dc/aggregates/dc/aggregates" << curlevel << ".vel";
//    fileout.open(fileoutstream.str().c_str(),ios_base::out);
//    velaggs->Print(fileout);
//    fileout.flush();
//    fileout.close();
//
//    std::stringstream fileoutstreamp;
//    fileoutstreamp << "/home/wiesner/fluid/cubits/dc/aggregates/dc/aggregates" << curlevel << ".pre";
//    fileout.open(fileoutstreamp.str().c_str(),ios_base::out);
//    preaggs->Print(fileout);
//    fileout.flush();
//    fileout.close();


 /*   std::ofstream fileout2;
    std::stringstream fileoutstream2;
    fileoutstream2 << "/home/wiesner/Amat" << curlevel << ".txt";
    fileout2.open(fileoutstream2.str().c_str(),ios_base::out);
    fileout2 << *A11_[curlevel] << endl;
    fileout2.flush();
    fileout2.close();*/
#endif



    /////////////////////////// calc RAP product for next level
    A11_[curlevel+1] = Multiply(Tvel_[curlevel]->R(),*A11_[curlevel],Tvel_[curlevel]->P());
    A12_[curlevel+1] = Multiply(Tvel_[curlevel]->R(),*A12_[curlevel],Tpre_[curlevel]->P());
    A21_[curlevel+1] = Multiply(Tpre_[curlevel]->R(),*A21_[curlevel],Tvel_[curlevel]->P());
    A22_[curlevel+1] = Multiply(Tpre_[curlevel]->R(),*A22_[curlevel],Tpre_[curlevel]->P());

    if(nVerbose_ > 4) // be verbose
    {
      cout << "A11[" << curlevel+1 << "]: " << A11_[curlevel+1]->EpetraMatrix()->NumGlobalRows() << " x " << A11_[curlevel+1]->EpetraMatrix()->NumGlobalCols() << " (" << A11_[curlevel+1]->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;
      cout << "A12[" << curlevel+1 << "]: " << A12_[curlevel+1]->EpetraMatrix()->NumGlobalRows() << " x " << A12_[curlevel+1]->EpetraMatrix()->NumGlobalCols() << " (" << A12_[curlevel+1]->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;
      cout << "A21[" << curlevel+1 << "]: " << A21_[curlevel+1]->EpetraMatrix()->NumGlobalRows() << " x " << A21_[curlevel+1]->EpetraMatrix()->NumGlobalCols() << " (" << A21_[curlevel+1]->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;
      cout << "A22[" << curlevel+1 << "]: " << A22_[curlevel+1]->EpetraMatrix()->NumGlobalRows() << " x " << A22_[curlevel+1]->EpetraMatrix()->NumGlobalCols() << " (" << A22_[curlevel+1]->EpetraMatrix()->NumGlobalNonzeros() << ")" << endl;
    }

    //////////////////// create pre- and postsmoothers
    std::stringstream stream;
    stream << "braess-sarazin: list (level " << curlevel << ")";
    ParameterList& subparams = spparams->sublist("AMGBS Parameters").sublist(stream.str());

    // copy ML Parameters or IFPACK Parameters from FLUID PRESSURE SOLVER block
    if (pressureparams_.isSublist("IFPACK Parameters"))
      subparams.sublist("IFPACK Parameters") = pressureparams_.sublist("IFPACK Parameters");
    else if(pressureparams_.isSublist("ML Parameters"))
      subparams.sublist("ML Parameters") = pressureparams_.sublist("ML Parameters");
    else
      dserror("SaddlePointPreconditioner::Setup: no IFPACK or ML ParameterList found in FLUD PRESSURE SOLVER block -> cannot be!");

    if(curlevel==0)
    {
      subparams.set("pressure correction approx: type",subparams.get("fine: type","IFPACK"));
      if(subparams.isParameter("fine: ifpack type"))  subparams.set("pressure correction approx: ifpack type",subparams.get("fine: ifpack type","ILU"));
      subparams.remove("fine: type",false);
      subparams.remove("fine: ifpack type",false);
    }
    else
    {
      subparams.set("pressure correction approx: type",subparams.get("medium: type","IFPACK"));
      if(subparams.isParameter("medium: ifpack type"))  subparams.set("pressure correction approx: ifpack type",subparams.get("medium: ifpack type","ILU"));
      subparams.remove("medium: type",false);
      subparams.remove("medium: ifpack type",false);
    }

    if(nVerbose_ > 8)
    {
      cout << "Braess-Sarazin smoother (level " << curlevel << ")" << endl << "parameters:" << endl << subparams << endl << endl;
    }

    preS_[curlevel]  = rcp(new BraessSarazin_Smoother(A11_[curlevel],A12_[curlevel],A21_[curlevel],A22_[curlevel],subparams));
    postS_[curlevel] = preS_[curlevel];//rcp(new BraessSarazin_Smoother(A11_[curlevel],A12_[curlevel],A21_[curlevel],A22_[curlevel],subparams));

    //////////////////// prepare variables for next aggregation level
    curvelNS = nextvelNS;
    curpreNS = nextpreNS;

    nlevels_ = curlevel + 1;

    //////////////////// check if aggregation is complete
    if ((A11_[curlevel+1]->EpetraMatrix()->NumGlobalRows() + A22_[curlevel+1]->EpetraMatrix()->NumGlobalRows()) < nmaxcoarsedim)
    {
      if(nVerbose_ > 4) cout << "dim A[" << curlevel+1 << "] < " << nmaxcoarsedim << ". -> end aggregation process" << endl;
      break;
    }
  }

  //////////////////// setup coarsest smoother
  std::stringstream stream;
  stream << "braess-sarazin: list (level " << nlevels_ << ")";
  ParameterList& subparams = spparams->sublist("AMGBS Parameters").sublist(stream.str());

  // copy ML Parameters or IFPACK Parameters from FLUID PRESSURE SOLVER block
  if (pressureparams_.isSublist("IFPACK Parameters"))
    subparams.sublist("IFPACK Parameters") = pressureparams_.sublist("IFPACK Parameters");
  else if(pressureparams_.isSublist("ML Parameters"))
    subparams.sublist("ML Parameters") = pressureparams_.sublist("ML Parameters");
  else
    dserror("SaddlePointPreconditioner::Setup: no IFPACK or ML ParameterList found in FLUD PRESSURE SOLVER block -> cannot be!");

  subparams.set("pressure correction approx: type",subparams.get("coarse: type","IFPACK"));
  if(subparams.isParameter("coarse: ifpack type"))  subparams.set("pressure correction approx: ifpack type",subparams.get("coarse: ifpack type","ILU"));
  subparams.remove("coarse: type",false);
  subparams.remove("coarse: ifpack type",false);

  if(nVerbose_ > 8)
  {
    cout << "Braess-Sarazin smoother (level " << nlevels_ << ")" << endl << "parameters:" << endl << subparams << endl << endl;
  }

  coarsestSmoother_ = rcp(new BraessSarazin_Smoother(A11_[nlevels_],A12_[nlevels_],A21_[nlevels_],A22_[nlevels_],subparams));

  if(nVerbose_ > 2)
  {
    cout << "setup phase complete:" << endl;
    cout << "nlevels/maxlevels: " << nlevels_+1 << "/" << nmaxlevels_+1 << endl;
  }

  MLAPI::Finalize();


#ifdef WRITEOUTSTATISTICS
  if(outfile_)
  {
    fprintf(outfile_,"saddlepointPrecSetupTime %f\tsaddlepointPrecLevels %i\t",ttt.ElapsedTime(),nlevels_);
  }

#ifdef WRITEOUTSYMMETRY
  RCP<SparseMatrix> tmpmtx = rcp(new SparseMatrix(*Ainput_->Merge(),Copy));
  tmpmtx->Add(*Ainput_->Merge(),true,-1.0,1.0);
  fprintf(outfile_,"NormFrobenius %f\t",tmpmtx->NormFrobenius());
#endif
#endif


}

///////////////////////////////////////////////////////////
int LINALG::SaddlePointPreconditioner::GetGlobalAggregates(SparseMatrix& A, ParameterList& List, const Epetra_MultiVector& ThisNS, Epetra_IntVector& aggrinfo, int& naggregates_local)
{
  int naggregates = GetAggregates(A,List,ThisNS,aggrinfo);

  const Epetra_Comm& comm = A.Comm();
  vector<int> local(comm.NumProc());
  vector<int> global(comm.NumProc());
  for (int i=0; i<comm.NumProc(); ++i) local[i] = 0;  // zero out local vector
  local[comm.MyPID()] = naggregates;                  // fill in local aggregates
  comm.SumAll(&local[0],&global[0],comm.NumProc());   // now all aggregates are in global
  int offset = 0;
  for (int i=0; i<comm.MyPID(); ++i) offset += global[i];
  for (int i=0; i<aggrinfo.MyLength(); ++i)
    if (aggrinfo[i] < naggregates) aggrinfo[i] += offset; // shift "local" agg id to "global" agg id
    else                           aggrinfo[i] = -1;      // set agg info of all non local dofs to -1

  int naggregatesglobal = 0;
  for (int i=0; i<comm.NumProc(); ++i)    // sum up all number of aggregates over all processors
  {
    naggregatesglobal += global[i];
  }

  naggregates_local = naggregates;  // return local number of aggregates for current processor as reference
  return naggregatesglobal;
}

int LINALG::SaddlePointPreconditioner::GetAggregates(SparseMatrix& A, ParameterList& List, const Epetra_MultiVector& ThisNS, Epetra_IntVector& aggrinfo)
{
  if(!A.RowMap().SameAs(aggrinfo.Map())) dserror ("map of aggrinfo must match row map of operator");

  string CoarsenType    = List.get("aggregation: type","Uncoupled");
  double Threshold    = List.get("aggregation: threshold", 0.0);
  int NumPDEEquations   = List.get("PDE equations",1);
  int nsdim         = List.get("null space: dimension", -1);
  if (nsdim==-1)  cout << "dimension of null space not set" << endl;
  int size = A.RowMap().NumMyElements();

  // create ML objects
  ML_Aggregate* agg_object;
  ML_Aggregate_Create(&agg_object);
  ML_Aggregate_KeepInfo(agg_object,1);
  ML_Aggregate_Set_MaxLevels(agg_object,2);
  ML_Aggregate_Set_StartLevel(agg_object,0);
  ML_Aggregate_Set_Threshold(agg_object,Threshold);

  ML_Set_PrintLevel(List.get("ML output", 0));

  // create ML operator
  ML_Operator* ML_Ptent = 0;
  ML_Ptent = ML_Operator_Create(MLAPI::GetML_Comm());

  //if(!thisns) cout << "error: null space is NULL" << endl;
  if (ThisNS.NumVectors() == 0) dserror("error: zero-dimension null space");

  int ns_size = ThisNS.MyLength();

  double* null_vect = 0;
  ML_memory_alloc((void **)(&null_vect), sizeof(double) * ns_size * ThisNS.NumVectors(), "ns");

  int incr = 1;
  for (int v = 0 ; v < ThisNS.NumVectors() ; ++v)
    DCOPY_F77(&ns_size, (double*)ThisNS[v], &incr,
        null_vect + v * ThisNS.MyLength(), &incr);

  ML_Aggregate_Set_NullSpace(agg_object,NumPDEEquations,nsdim,null_vect,size);

  // set coarsening type
  if(CoarsenType == "Uncoupled")
    agg_object->coarsen_scheme = ML_AGGR_UNCOUPLED;
  else if (CoarsenType == "Uncoupled-MIS")
    agg_object->coarsen_scheme = ML_AGGR_HYBRIDUM;
  else if(CoarsenType == "MIS")
  { // needed for MIS, otherwise it sets the number of equations to the null space dimension
    //agg_object->max_levels = -7; // i don't understand this
    agg_object->coarsen_scheme = ML_AGGR_MIS;
  }
  else if(CoarsenType == "METIS")
    agg_object->coarsen_scheme = ML_AGGR_METIS;
  else
  {
    dserror(std::string("error: requested aggregation scheme (" + CoarsenType + ") not recognized"));
  }

  // create ML_Operator for A
  ML_Operator* ML_A = ML_Operator_Create(MLAPI::GetML_Comm());
  ML_Operator_WrapEpetraMatrix(A.EpetraMatrix().get(),ML_A);

  // run coarsening process
  int NextSize = ML_Aggregate_Coarsen(agg_object, ML_A, &ML_Ptent, MLAPI::GetML_Comm());

  int* aggrmap = NULL;
  ML_Aggregate_Get_AggrMap(agg_object,0,&aggrmap);
  if (!aggrmap) dserror("aggr_info not available");

#if 0 // debugging
  fflush(stdout);
  for (int proc=0; proc<A.GetRowMatrix()->Comm().NumProc(); ++proc)
  {
    if (A.GetRowMatrix()->Comm().MyPID()==proc)
    {
      cout << "Proc " << proc << ":" << endl;
      cout << "aggrcount " << aggrcount << endl;
      cout << "NextSize " << NextSize << endl;
      for (int i=0; i<size; ++i)
        cout << "aggrmap[" << i << "] = " << aggrmap[i] << endl;
      fflush(stdout);
    }
    A.GetRowMatrix()->Comm().Barrier();
  }
#endif

  assert (NextSize * nsdim != 0);
  for (int i=0; i<size; ++i) aggrinfo[i] = aggrmap[i];

  ML_Aggregate_Destroy(&agg_object);

  ////////////////////////////////
  // -> i think, Michael forgot this
  // since we're only interested in the aggregates we can free the ML_Operators
  // now valgrind isn't complaining any more
  // but there are still two reachable blocks for Uncoupled coarsening scheme (in ml_qr_fix 15 and 20, called by ML_Aggregate_CoarsenUncoupled in line 629, ml_agg_uncoupled.c)
  // i think it is as ML_qr_fix_setNumDeadNod(numDeadNod); is never called???
  ML_Operator_Destroy(&ML_Ptent);
  ML_Operator_Destroy(&ML_A);
  ML_Ptent = NULL;
  ML_A = NULL;
  ML_qr_fix_Destroy();   // <- ok, this is missing in ML_Aggregate_CoarsenUncoupled in line 629, ml_agg_uncoupled.c

  ML_memory_free((void**)(&null_vect));  // temporary vector with null space data
  null_vect = NULL;
  ////////////////////////////////

  return (NextSize/nsdim);
}

///////////////////////////////////////////////////////////////////
RCP<LINALG::SparseMatrix> LINALG::SaddlePointPreconditioner::Multiply(const SparseMatrix& A, const SparseMatrix& B, const SparseMatrix& C, bool bComplete)
{
  TEUCHOS_FUNC_TIME_MONITOR("SaddlePoint_Preconditioner::Multiply (with MLMultiply)");

  RCP<SparseMatrix> tmp = LINALG::MLMultiply(B,C,true);
  return LINALG::MLMultiply(A,*tmp,bComplete);
}


#endif // CCADISCRET

