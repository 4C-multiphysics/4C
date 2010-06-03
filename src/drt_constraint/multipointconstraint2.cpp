/*!----------------------------------------------------------------------
\file multipointconstraint2.cpp

\brief Basic constraint class, dealing with multi point constraints
<pre>
Maintainer: Thomas Kloeppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/kloeppel
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET


#include "multipointconstraint2.H"

#include "../drt_lib/drt_discret.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_sparsematrix.H"
#include <iostream>
#include "../drt_lib/drt_dofset_transparent.H"
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_globalproblem.H"



/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
UTILS::MPConstraint2::MPConstraint2(RCP<DRT::Discretization> discr,
        const string& conditionname,
        int& minID,
        int& maxID)
: MPConstraint(discr,
     conditionname,
     minID,
     maxID)
{
  if (constrcond_.size())
  {
    int dummy=0;
    //create constraint discretization and store it with label 0, within the map
    constraintdis_=CreateDiscretizationFromCondition(actdisc_,constrcond_,"ConstrDisc","CONSTRELE2", dummy);
    RCP<Epetra_Map> newcolnodemap = DRT::UTILS::ComputeNodeColMap(actdisc_, constraintdis_.find(0)->second);
    actdisc_->Redistribute(*(actdisc_->NodeRowMap()), *newcolnodemap);
    RCP<DRT::DofSet> newdofset = rcp(new DRT::TransparentDofSet(actdisc_));
    (constraintdis_.find(0)->second)->ReplaceDofSet(newdofset);
    newdofset = null;
    (constraintdis_.find(0)->second)->FillComplete();
  }

  return;
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine activates conditions (restart)                    |
*------------------------------------------------------------------------*/
void UTILS::MPConstraint2::Initialize
(
  const double& time
)
{
  for (unsigned int i = 0; i < constrcond_.size(); ++i)
  {
    DRT::Condition& cond = *(constrcond_[i]);

    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID=cond.GetInt("ConditionID");

    // if current time (at) is larger than activation time of the condition, activate it
    if((inittimes_.find(condID)->second < time) && (!activecons_.find(condID)->second))
    {
      activecons_.find(condID)->second=true;
      if (actdisc_->Comm().MyPID()==0)
      {
        cout << "Encountered another active condition (Id = " << condID << ")  for restart time t = "<< time << endl;
      }
    }
  }
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void UTILS::MPConstraint2::Initialize(
    ParameterList&        params,
    RCP<Epetra_Vector>    systemvector)
{
  const double time = params.get("total time",-1.0);
  // in case init is set to true we want to set systemvector1 to the amplitudes defined
  // in the input file
  // allocate vectors for amplitudes and IDs


  vector<double> amplit(constrcond_.size());
  vector<int> IDs(constrcond_.size());
  // read data of the input files
  for (unsigned int i=0;i<constrcond_.size();i++)
  {
    DRT::Condition& cond = *(constrcond_[i]);
    int condID=cond.GetInt("ConditionID");
    if(inittimes_.find(condID)->second<=time)
    {
      const int    MPCcondID  = constrcond_[i]->GetInt("ConditionID");
      amplit[i] = constrcond_[i]->GetDouble("amplitude");
      const int mid=params.get("OffsetID",0);
      IDs[i] = MPCcondID-mid;
      // remember next time, that this condition is already initialized, i.e. active
      activecons_.find(condID)->second=true;
      if (actdisc_->Comm().MyPID()==0)
      {
        cout << "Encountered a new active condition (Id = " << condID << ")  at time t = "<< time << endl;
      }
    }
  }
  // replace systemvector by the given amplitude values
  // systemvector is supposed to be the vector with initial values of the constraints
  if (actdisc_->Comm().MyPID()==0)
  {
    systemvector->ReplaceGlobalValues(amplit.size(),&(amplit[0]),&(IDs[0]));
  }

  return;
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void UTILS::MPConstraint2::Evaluate(
    ParameterList&        params,
    RCP<LINALG::SparseOperator> systemmatrix1,
    RCP<LINALG::SparseOperator> systemmatrix2,
    RCP<Epetra_Vector>    systemvector1,
    RCP<Epetra_Vector>    systemvector2,
    RCP<Epetra_Vector>    systemvector3)
{

  switch (Type())
  {
    case mpcnodeonline2d:
      params.set("action","calc_MPC_stiff");
    break;
    case none:
      return;
    default:
      dserror("Constraint/monitor is not an multi point constraint!");
  }
  EvaluateConstraint(constraintdis_.find(0)->second,params,systemmatrix1,systemmatrix2,systemvector1,systemvector2,systemvector3);

  return;
}

/*------------------------------------------------------------------------*
 |(private)                                                   tk 04/08    |
 |subroutine creating a new discretization containing constraint elements |
 *------------------------------------------------------------------------*/
map<int,RCP<DRT::Discretization> > UTILS::MPConstraint2::CreateDiscretizationFromCondition
(
  RCP<DRT::Discretization> actdisc,
  vector< DRT::Condition* >      constrcondvec,
  const string&             discret_name,
  const string&             element_name,
  int& startID
)
{
  RCP<Epetra_Comm> com = rcp(actdisc->Comm().Clone());

  RCP<DRT::Discretization> newdis = rcp(new DRT::Discretization(discret_name,com));

  if (!actdisc->Filled())
  {
    actdisc->FillComplete();
  }

  const int myrank = newdis->Comm().MyPID();

  if(constrcondvec.size()==0)
      dserror("number of multi point constraint conditions = 0 --> cannot create constraint discretization");

  set<int> rownodeset;
  set<int> colnodeset;
  const Epetra_Map* actnoderowmap = actdisc->NodeRowMap();

  // Loop all conditions in constrcondvec
  for (unsigned int j=0;j<constrcondvec.size();j++)
  {
    vector<int> ngid=*(constrcondvec[j]->Nodes());
    const int numnodes=ngid.size();
    // We sort the global node ids according to the definition of the boundary condition
    ReorderConstraintNodes(ngid, constrcondvec[j]);

    remove_copy_if(&ngid[0], &ngid[0]+numnodes,
                     inserter(rownodeset, rownodeset.begin()),
                     not1(DRT::UTILS::MyGID(actnoderowmap)));
    // copy node ids specified in condition to colnodeset
    copy(&ngid[0], &ngid[0]+numnodes,
          inserter(colnodeset, colnodeset.begin()));

    // construct boundary nodes, which use the same global id as the cutter nodes
    for (int i=0; i<actnoderowmap->NumMyElements(); ++i)
    {
      const int gid = actnoderowmap->GID(i);
      if (rownodeset.find(gid)!=rownodeset.end())
      {
        const DRT::Node* standardnode = actdisc->lRowNode(i);
        newdis->AddNode(rcp(new DRT::Node(gid, standardnode->X(), myrank)));
      }
    }

    if (myrank == 0)
    {
      RCP<DRT::Element> constraintele = DRT::UTILS::Factory(element_name,"Polynomial", j, myrank);
      // set the same global node ids to the ale element
      constraintele->SetNodeIds(ngid.size(), &(ngid[0]));

      // add constraint element
      newdis->AddElement(constraintele);

    }
    // now care about the parallel distribution and ghosting.
    // So far every processor only knows about his nodes
  }

  //build unique node row map
  vector<int> boundarynoderowvec(rownodeset.begin(), rownodeset.end());
  rownodeset.clear();
  RCP<Epetra_Map> constraintnoderowmap = rcp(new Epetra_Map(-1,
                                                             boundarynoderowvec.size(),
                                                             &boundarynoderowvec[0],
                                                             0,
                                                             newdis->Comm()));
  boundarynoderowvec.clear();

  //build overlapping node column map
  vector<int> constraintnodecolvec(colnodeset.begin(), colnodeset.end());
  colnodeset.clear();
  RCP<Epetra_Map> constraintnodecolmap = rcp(new Epetra_Map(-1,
                                                             constraintnodecolvec.size(),
                                                             &constraintnodecolvec[0],
                                                             0,
                                                             newdis->Comm()));

  constraintnodecolvec.clear();

  newdis->Redistribute(*constraintnoderowmap,*constraintnodecolmap);

  map<int,RCP<DRT::Discretization> > newdismap;
  newdismap[startID]=newdis;
  return newdismap;
}

/*----------------------------------------------------------------------*
 |(private)                                                 tk 04/08    |
 |reorder MPC nodes based on condition input                            |
 *----------------------------------------------------------------------*/
void UTILS::MPConstraint2::ReorderConstraintNodes
(
  vector<int>& nodeids,
  const DRT::Condition* cond
)
{
  // get this condition's nodes
  vector<int> temp=nodeids;
  if (nodeids.size()==3)
  {
    nodeids[0]=temp[cond->GetInt("constrNode 1")-1];
    nodeids[1]=temp[cond->GetInt("constrNode 2")-1];
    nodeids[2]=temp[cond->GetInt("constrNode 3")-1];
  }
  else
  {
    dserror("strange number of nodes for an MPC! Should be 3 in 2D.");
  }
  return;
}

/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void UTILS::MPConstraint2::EvaluateConstraint(RCP<DRT::Discretization> disc,
    ParameterList&        params,
    RCP<LINALG::SparseOperator> systemmatrix1,
    RCP<LINALG::SparseOperator> systemmatrix2,
    RCP<Epetra_Vector>    systemvector1,
    RCP<Epetra_Vector>    systemvector2,
    RCP<Epetra_Vector>    systemvector3)
{

  if (!(disc->Filled())) dserror("FillComplete() was not called");
  if (!(disc->HaveDofs())) dserror("AssignDegreesOfFreedom() was not called");

  // see what we have for input
  bool assemblemat1 = systemmatrix1!=Teuchos::null;
  bool assemblemat2 = systemmatrix2!=Teuchos::null;
  bool assemblevec1 = systemvector1!=Teuchos::null;
  bool assemblevec2 = systemvector2!=Teuchos::null;
  bool assemblevec3 = systemvector3!=Teuchos::null;

  // define element matrices and vectors
  Epetra_SerialDenseMatrix elematrix1;
  Epetra_SerialDenseMatrix elematrix2;
  Epetra_SerialDenseVector elevector1;
  Epetra_SerialDenseVector elevector2;
  Epetra_SerialDenseVector elevector3;


  const double time = params.get("total time",-1.0);
  const int numcolele = disc->NumMyColElements();

  // get values from time integrator to scale matrices with
  double scStiff = params.get("scaleStiffEntries",1.0);
  double scConMat = params.get("scaleConstrMat",1.0);

  // loop over column elements
  for (int i=0; i<numcolele; ++i)
  {
    DRT::Element* actele = disc->lColElement(i);
    DRT::Condition& cond = *(constrcond_[actele->Id()]);
    int condID=cond.GetInt("ConditionID");

    // computation only if time is larger or equal than initialization time for constraint
    if(inittimes_.find(condID)->second<=time)
    {
      // initialize if it is the first time condition is evaluated
      if(activecons_.find(condID)->second==false)
      {
        const string action = params.get<string>("action");
        Initialize(params,systemvector2);
        params.set("action",action);
      }

      //define global and local index of this bc in redundant vectors
      const int offsetID = params.get<int>("OffsetID");
      int gindex = condID-offsetID;
      const int lindex = (systemvector3->Map()).LID(gindex);

      // Get the current lagrange multiplier value for this condition
      const RCP<Epetra_Vector> lagramul = params.get<RCP<Epetra_Vector> >("LagrMultVector");
      const double lagraval = (*lagramul)[lindex];

      // get element location vector, dirichlet flags and ownerships
      vector<int> lm;
      vector<int> lmowner;
      actele->LocationVector(*disc,lm,lmowner);
      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();
      if (assemblemat1) elematrix1.Shape(eledim,eledim);
      if (assemblemat2) elematrix2.Shape(eledim,eledim);
      if (assemblevec1) elevector1.Size(eledim);
      if (assemblevec2) elevector2.Size(eledim);
      if (assemblevec3) elevector3.Size(1); // elevector3 always contains a scalar

      params.set("ConditionID",condID);
      params.set<RefCountPtr<DRT::Condition> >("condition", rcp(&cond,false));
      // call the element evaluate method
      int err = actele->Evaluate(params,*disc,lm,elematrix1,elematrix2,
                                 elevector1,elevector2,elevector3);
      if (err) dserror("Proc %d: Element %d returned err=%d",disc->Comm().MyPID(),actele->Id(),err);

      int eid = actele->Id();

      // Assembly
      if (assemblemat1)
      {
        // scale with time integrator dependent value
        elematrix1.Scale(scStiff*lagraval);
        systemmatrix1->Assemble(eid,elematrix1,lm,lmowner);
      }
      if (assemblemat2)
      {
        vector<int> colvec(1);
        colvec[0]=gindex;
        elevector2.Scale(scConMat);
        systemmatrix2->Assemble(eid,elevector2,lm,lmowner,colvec);
      }
      if (assemblevec1)
      {
        elevector1.Scale(lagraval);
        LINALG::Assemble(*systemvector1,elevector1,lm,lmowner);
      }
      if (assemblevec3)
      {
        vector<int> constrlm;
        vector<int> constrowner;
        constrlm.push_back(gindex);
        constrowner.push_back(actele->Owner());
        LINALG::Assemble(*systemvector3,elevector3,constrlm,constrowner);
      }

      // Load curve business
      const vector<int>*    curve  = cond.Get<vector<int> >("curve");
      int curvenum = -1;
      if (curve) curvenum = (*curve)[0];
      double curvefac = 1.0;
      bool usetime = true;
      if (time<0.0) usetime = false;
      if (curvenum>=0 && usetime)
        curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
      RCP<Epetra_Vector> timefact = params.get<RCP<Epetra_Vector> >("vector curve factors");
      timefact->ReplaceGlobalValues(1,&curvefac,&gindex);
    }
  }
  return;
} // end of EvaluateCondition

#endif
