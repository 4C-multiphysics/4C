/*!----------------------------------------------------------------------
\file linalg_precond.cpp

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "linalg_precond.H"
#include "linalg_solver.H"

#include "linalg_mlapi_operator.H"
#include "simpler_operator.H"
#include "../drt_lib/drt_node.H"

#include "../drt_f2/fluid2.H"
#include "../drt_f2/fluid2_nurbs.H"
#include "../drt_f3/fluid3.H"
#include "../drt_f3/fluid3_nurbs.H"
#include "../drt_f3/xfluid3.H"
#include "../drt_combust/combust3.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
LINALG::Preconditioner::Preconditioner(Teuchos::RCP<Solver> solver)
  : solver_(solver),
    ncall_(0)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::Preconditioner::Setup(Teuchos::RCP<Epetra_Operator>      matrix,
                                   Teuchos::RCP<LINALG::MapExtractor> fsidofmapex,
                                   Teuchos::RCP<DRT::Discretization>  fdis,
                                   Teuchos::RCP<Epetra_Map>           inodes,
                                   bool                               structuresplit)
{
  std::string solvertype = solver_->Params().get("solver","none");
  if (solvertype=="aztec")
  {
    Teuchos::ParameterList& azlist = solver_->Params().sublist("Aztec Parameters");
    // see whether Operator is a Epetra_CrsMatrix
    Epetra_CrsMatrix* A = dynamic_cast<Epetra_CrsMatrix*>(&*matrix);

    // get type of preconditioner and build either Ifpack or ML
    // if we have an ifpack parameter list, we do ifpack
    // if we have an ml parameter list we do ml
    bool   doifpack  = solver_->Params().isSublist("IFPACK Parameters");
    bool   doml      = solver_->Params().isSublist("ML Parameters");
    bool   dosimpler = solver_->Params().isSublist("SIMPLER");
    if (!A || dosimpler)
    {
      doifpack = false;
      doml     = false;
    }

    // do ifpack if desired
    if (doifpack)
    {
      Teuchos::ParameterList& ifpacklist = solver_->Params().sublist("IFPACK Parameters");
      ifpacklist.set<bool>("relaxation: zero starting solution",true);
      // create a copy of the scaled matrix
      // so we can reuse the preconditioner
      Pmatrix_ = rcp(new Epetra_CrsMatrix(*A));
      // get the type of ifpack preconditioner from aztec
      std::string prectype = azlist.get("preconditioner","ILU");
      int    overlap  = azlist.get("AZ_overlap",0);
      Ifpack Factory;
      Ifpack_Preconditioner* prec = Factory.Create(prectype,Pmatrix_.get(),overlap);
      prec->SetParameters(ifpacklist);
      prec->Initialize();
      prec->Compute();
      prec_ = rcp(prec);
    }

    // do ml if desired
    if (doml)
    {
      // in case of monolithic fsi with structure split enrich the fluid nullspace
      // by rotations on the interface
      if (structuresplit && fsidofmapex != Teuchos::null)
        EnrichFluidNullSpace(solver_->Params().sublist("ML Parameters"),
                             fsidofmapex,
                             inodes,
                             fdis);

      Teuchos::ParameterList& mllist = solver_->Params().sublist("ML Parameters");
      // see whether we use standard ml or our own mlapi operator
      const bool domlapioperator = mllist.get<bool>("LINALG::AMG_Operator",false);
      if (domlapioperator)
      {
        // create a copy of the scaled matrix
        // so we can reuse the preconditioner several times
        prec_ = Teuchos::null;
        Pmatrix_ = rcp(new Epetra_CrsMatrix(*A));
        prec_ = rcp(new LINALG::AMG_Operator(Pmatrix_,mllist,true));
      }
      else
      {
        // create a copy of the scaled (and downwinded) matrix
        // so we can reuse the preconditioner several times
        prec_ = Teuchos::null;
        Pmatrix_ = rcp(new Epetra_CrsMatrix(*A));
        prec_ = rcp(new ML_Epetra::MultiLevelPreconditioner(*Pmatrix_,mllist,true));
        // for debugging ML
        //dynamic_cast<ML_Epetra::MultiLevelPreconditioner&>(*P_).PrintUnused(0);
      }
    }

#if 0
    if (dosimpler)
    {
      // SIMPLER does not need copy of preconditioning matrix to live
      // SIMPLER does not use the downwinding installed here, it does
      // its own downwinding inside if desired
      prec_ = rcp(new LINALG::SIMPLER_Operator(matrix,Params(),
                                               solver_->Params().sublist("SIMPLER"),
                                               outfile_));
      Pmatrix_ = null;
    }
#endif
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::Preconditioner::Solve(Teuchos::RCP<Epetra_Operator>  matrix,
                                   Teuchos::RCP<Epetra_Vector>    x,
                                   Teuchos::RCP<Epetra_Vector>    b,
                                   bool refactor,
                                   bool reset)
{
  std::string solvertype = solver_->Params().get("solver","none");
  if (solvertype=="aztec")
  {
    // do just the preconditioner from iterative solver

#if 0
    bool   doifpack  = solver_->Params().isSublist("IFPACK Parameters");
    if (doifpack)
    {
      Teuchos::ParameterList& ifpacklist = solver_->Params().sublist("IFPACK Parameters");
      ifpacklist.set<bool>("relaxation: zero starting solution",false);
    }
#endif

    // apply the preconditioner
    // This is were the work happens.
    ApplyInverse(*b,*x);
  }
  else
  {
    // direct solves are done by the solver itself.
    solver_->Solve(matrix,x,b,refactor,reset);
  }

  ncall_ += 1;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::ParameterList& LINALG::Preconditioner::Params()
{
  return solver_->Params();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::Preconditioner::EnrichFluidNullSpace(
                                      Teuchos::ParameterList&            mllist,
                                      Teuchos::RCP<LINALG::MapExtractor> fsidofmapex,
                                      Teuchos::RCP<Epetra_Map>           inodes,
                                      Teuchos::RCP<DRT::Discretization>  fdis)
{
  // map of fluid problem
  const Teuchos::RCP<const Epetra_Map>& fluidmap = fsidofmapex->FullMap();

  // increase number of null space vectors
  int nsdim = mllist.get("null space: dimension",0);
  if (!nsdim) dserror("null space: dimension does not exist in ML parameter list");
  int newnsdim = nsdim;

  DRT::Element* ele = fdis->lRowElement(0);

  // this is not the way it should be done
  const DRT::ElementType & eot = ele->ElementType();
  bool is3d = false;
  if ( eot==DRT::ELEMENTS::Fluid3Type::Instance() or
       eot==DRT::ELEMENTS::NURBS::Fluid3NurbsType::Instance() or
       eot==DRT::ELEMENTS::XFluid3Type::Instance() or
       eot==DRT::ELEMENTS::Combust3Type::Instance() )
  {
    is3d = true;
  }
  else if ( eot==DRT::ELEMENTS::Fluid2Type::Instance() or
            eot==DRT::ELEMENTS::NURBS::Fluid2NurbsType::Instance() )
  {
    is3d = false;
  }
  else
  {
    dserror("Element type not supported by ML");
  }

  if (nsdim==3 && !is3d)      // add one rotation in case of 2D
    newnsdim += 1;
  else if (nsdim==4 && is3d)  // add three rotations in case of 3D
    newnsdim += 3;
  else if (nsdim==7 && is3d)  // ns has been previously enriched, do nothing
    return;
  else if (nsdim==4 && !is3d) // ns has been previously enriched, do nothing
    return;
  else
    dserror("Unexpected nullspace dimension %d",nsdim);

  // row length of fluid problem
  const int size = fluidmap->NumMyElements();

  // old nullspace
  Teuchos::RCP<vector<double> > ns = mllist.get<Teuchos::RCP<vector<double> > >("nullspace",Teuchos::null);
  if (ns == Teuchos::null) dserror("there is no nullspace in ml list");

  // new nullspace
  Teuchos::RCP<vector<double> > newns = Teuchos::rcp(new vector<double>(newnsdim*size,0.0));

  // use old nullspace as the first modes of new nullspace
  copy((*ns).begin(),(*ns).end(),(*newns).begin());

  // point to the modes to be added
  double* rot1 = NULL;
  double* rot2 = NULL;
  double* rot3 = NULL;
  rot1 = &((*newns)[nsdim*size]);
  if (is3d)
  {
    rot2 = rot1 + size;
    rot3 = rot2 + size;
  }

  // nodal center of the interface nodes
  double x0[3];
  {
    double x0send[3] = {0.0,0.0,0.0};
    int counts=0;
    int count=0;
    for (int i=0; i<fdis->NumMyRowNodes(); ++i)
      if (inodes->MyGID(fdis->lRowNode(i)->Id()))
      {
        for (int j=0; j<3; ++j) x0send[j] += fdis->lRowNode(i)->X()[j];
        ++counts;
      }
    fdis->Comm().SumAll(x0send,x0,3);
    fdis->Comm().SumAll(&counts,&count,1);
    for (int i=0; i<3; ++i) x0[i] /= count;
  }

  for (int i=0; i<inodes->NumMyElements(); ++i)
  {
    DRT::Node* actnode = fdis->gNode(inodes->GID(i));
    const double* x    = actnode->X();
    vector<int>   dofs = fdis->Dof(actnode);
    if ((int)dofs.size() != nsdim) dserror("dof <-> nullspace dimension mismatch");
    // skip the pressure dof
    const int ndof = (int)dofs.size()-1;
    for (int j=0; j<ndof; ++j)
    {
      const int dof = dofs[j];
      const int lid = fluidmap->LID(dof);
      if (lid<0) dserror("Cannot find dof");
      if (is3d) // 3d case
      {
        switch (j) // j is degree of freedom
        {
          case 0: // x-direction, add rotational components
            rot1[lid] = 0.0;
            rot2[lid] = x[2] - x0[2];
            rot3[lid] = -x[1] + x0[1];
          break;
          case 1: // y-direction, add rotational components
            rot1[lid] = -x[2] + x0[2];
            rot2[lid] = 0.0;
            rot3[lid] = x[0] - x0[0];
          break;
          case 2: // z-direction, add rotational components
            rot1[lid] = x[1] - x0[1];
            rot2[lid] = -x[0] + x0[0];
            rot3[lid] = 0.0;
          break;
          default:
            dserror("dof out of bound");
          break;
        }
      }
      else // 2D case
      {
        switch (j) // j is degree of freedom
        {
          case 0: // x-direction, add rotational components
            rot1[lid] = -x[1] + x0[1];
          break;
          case 1: // y-direction, add rotational components
            rot1[lid] = x[0] - x0[0];
          break;
          default:
            dserror("dof out of bound");
          break;
        }
      }
    } // for (int j=0; j<ndof; ++j)
  } // for (int i=0; i<inodes->NumMyElements(); ++i)

#if 0 // debug output
  for (int i=0; i<size; ++i)
  {
    printf("%10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e \n",
           (*newns)[i],(*newns)[i+size],(*newns)[i+size*2],(*newns)[i+size*3],(*newns)[i+size*4],
           (*newns)[i+size*5],(*newns)[i+size*6]);
  }
#endif

  // put new nullspace and its dimension in mllist
  mllist.set("null space: dimension",newnsdim);
  mllist.set<Teuchos::RCP<vector<double> > >("nullspace",newns);
  mllist.set("null space: vectors",&(*newns)[0]);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int LINALG::Preconditioner::SetUseTranspose(bool UseTranspose)
{
  return prec_->SetUseTranspose(UseTranspose);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int LINALG::Preconditioner::Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const
{
  return prec_->Apply(X,Y);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int LINALG::Preconditioner::ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const
{
  return prec_->ApplyInverse(X,Y);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double LINALG::Preconditioner::NormInf() const
{
  return prec_->NormInf();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const char* LINALG::Preconditioner::Label() const
{
  return "LINALG::Preconditioner";
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool LINALG::Preconditioner::UseTranspose() const
{
  return prec_->UseTranspose();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool LINALG::Preconditioner::HasNormInf() const
{
  return prec_->HasNormInf();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Comm& LINALG::Preconditioner::Comm() const
{
  return prec_->Comm();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map& LINALG::Preconditioner::OperatorDomainMap() const
{
  return prec_->OperatorDomainMap();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map& LINALG::Preconditioner::OperatorRangeMap() const
{
  return prec_->OperatorRangeMap();
}


#endif
