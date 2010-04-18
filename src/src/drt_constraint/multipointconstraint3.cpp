/*!----------------------------------------------------------------------
\file multipointconstraint3.cpp

\brief Basic constraint class, dealing with multi point constraints
<pre>
Maintainer: Thomas Kloeppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/kloeppel
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET


#include "multipointconstraint3.H"
#include "mpcdofset.H"
#include "constraint_element3.H"

#include "../drt_lib/drt_discret.H"
#include "../drt_lib/linalg_utils.H"
#include "../drt_lib/linalg_sparsematrix.H"
#include <iostream>
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_globalproblem.H"

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
UTILS::MPConstraint3::MPConstraint3
(
  RCP<DRT::Discretization> discr,
  const string& conditionname,
  int& offsetID,
  int& maxID
):
MPConstraint
(
  discr,
  conditionname
)
{
  if (constrcond_.size())
  {
    maxID++;
    // control the constraint by absolute or relative values
    vector<DRT::Condition*>::iterator conditer;
    for (conditer=constrcond_.begin();conditer!=constrcond_.end();conditer++)
    {
      const int condID = (*conditer)->GetInt("ConditionID");
      if (offsetID>maxID) 
        offsetID=maxID;
//      if (Type()==mpcnormalcomp3d)
//        absconstraint_[condID]=true;
//      else
//      {
        const string* type = (*conditer)-> Get<string>("control");
        if (*type == "abs")
          absconstraint_[condID]=true;
        else
          absconstraint_[condID]=false;
//      }
    }

    constraintdis_=CreateDiscretizationFromCondition(actdisc_,constrcond_,"ConstrDisc","CONSTRELE3",maxID);

    map<int, RCP<DRT::Discretization> > ::iterator discriter;
    for (discriter=constraintdis_.begin(); discriter!=constraintdis_.end(); discriter++)
    {
      //ReplaceNumDof(actdisc_,discriter->second);
      RCP<Epetra_Map> newcolnodemap = ComputeNodeColMap(actdisc_, discriter->second);
      actdisc_->Redistribute(*(actdisc_->NodeRowMap()), *newcolnodemap);
      RCP<DRT::DofSet> newdofset=rcp(new MPCDofSet(actdisc_));
      (discriter->second)->ReplaceDofSet(newdofset);
      newdofset=null;
      (discriter->second)->FillComplete();
    }
  }

  return;
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine activates conditions (restart)                    |
*------------------------------------------------------------------------*/
void UTILS::MPConstraint3::Initialize
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
void UTILS::MPConstraint3::Initialize(
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
    if(inittimes_.find(condID)->second<=time&& (!(activecons_.find(condID)->second)))
    {
      // control absolute values
      if (absconstraint_.find(condID)->second)
      {
        int  MPCcondID  = constrcond_[i]->GetInt("ConditionID");
        //in case of a mpcnormalcomp3d-condition amplitude is always 0
//        if (Type()==mpcnormalcomp3d)
//          amplit[i]=0.0;
//        else
//        {
          double    MPCampl  = constrcond_[i]->GetDouble("amplitude");
          amplit[i]=MPCampl;
//        }
        const int mid=params.get("OffsetID",0);
        IDs[i]=MPCcondID-mid;
      }
      // control relative values
      else
      {
        switch (Type())
        {
          
          case mpcnormalcomp3d:
          case mpcnodeonplane3d:
            params.set("action","calc_MPC_state");
          break;
          case none:
            return;
          default:
            dserror("Constraint is not an multi point constraint!");
        }
        InitializeConstraint(constraintdis_.find(condID)->second,params,systemvector);
      }
      activecons_.find(condID)->second=true;
      if (actdisc_->Comm().MyPID()==0)
      {
        cout << "Encountered a new active condition (Id = " << condID << ")  at time t = "<< time << endl;
      }
    }
  }

  if (actdisc_->Comm().MyPID()==0) systemvector->SumIntoGlobalValues(amplit.size(),&(amplit[0]),&(IDs[0]));

  return;
}

/*-----------------------------------------------------------------------*
|(public)                                                        tk 07/08|
|Evaluate Constraints, choose the right action based on type             |
*-----------------------------------------------------------------------*/
void UTILS::MPConstraint3::Evaluate(
    ParameterList&        params,
    RCP<LINALG::SparseOperator> systemmatrix1,
    RCP<LINALG::SparseOperator> systemmatrix2,
    RCP<Epetra_Vector>    systemvector1,
    RCP<Epetra_Vector>    systemvector2,
    RCP<Epetra_Vector>    systemvector3)
{

  switch (Type())
  {
    case mpcnodeonplane3d:
    case mpcnormalcomp3d:
      params.set("action","calc_MPC_stiff");
    break;
    case none:
      return;
    default:
      dserror("Constraint/monitor is not an multi point constraint!");
  }
  map<int, RCP<DRT::Discretization> > ::iterator discriter;
  for (discriter=constraintdis_.begin(); discriter!=constraintdis_.end(); discriter++)
    EvaluateConstraint(discriter->second,params,systemmatrix1,systemmatrix2,systemvector1,systemvector2,systemvector3);

  return;
}

/*------------------------------------------------------------------------*
 |(private)                                                   tk 04/08    |
 |subroutine creating a new discretization containing constraint elements |
 *------------------------------------------------------------------------*/
map<int,RCP<DRT::Discretization> > UTILS::MPConstraint3::CreateDiscretizationFromCondition
(
  RCP<DRT::Discretization> actdisc,
  vector< DRT::Condition* >      constrcondvec,
  const string&             discret_name,
  const string&             element_name,
  int& startID
)
{
  // start with empty map
  map<int,RCP<DRT::Discretization> > newdiscmap;

   if (!actdisc->Filled())
  {
    actdisc->FillComplete();
  }

  if(constrcondvec.size()==0)
      dserror("number of multi point constraint conditions = 0 --> cannot create constraint discretization");


  // Loop all conditions in constrcondvec and build discretization for any condition ID

  int index=0; // counter for the index of condition in vector
  vector<DRT::Condition*>::iterator conditer;
  for (conditer=constrcondvec.begin();conditer!=constrcondvec.end();conditer++)
  {
    // initialize a new discretization
    RCP<Epetra_Comm> com = rcp(actdisc->Comm().Clone());
    RCP<DRT::Discretization> newdis = rcp(new DRT::Discretization(discret_name,com));
    const int myrank = newdis->Comm().MyPID();
    set<int> rownodeset;
    set<int> colnodeset;
    const Epetra_Map* actnoderowmap = actdisc->NodeRowMap();
    //get node IDs, this vector will only contain FREE nodes in the end
    vector<int> ngid=*((*conditer)->Nodes());
    vector<int> defnv;
    switch (Type())
    {
    case mpcnodeonplane3d:
    {
      // take three nodes defining plane as specified by user and put them into a set
      const vector<int>*  defnvp = (*conditer)->Get<vector<int> > ("planeNodes");
      defnv = *defnvp;
    }
    break;
    case mpcnormalcomp3d:
    {
      // take master node
      const int defn = (*conditer)->GetInt("masterNode");
      defnv.push_back(defn);
    }
    break;
    default: dserror ("not good!");
    }
    set<int> defns (defnv.begin(),defnv.end());
    set<int>::iterator nsit;
    // safe gids of definition nodes in a vector
    vector<int> defnodeIDs;

    int counter=1;//counter is used to keep track of deleted node ids from the vector, input starts with 1

    for (nsit=defns.begin(); nsit!=defns.end();++nsit)
    {
      defnodeIDs.push_back(ngid.at((*nsit)-counter));
      ngid.erase(ngid.begin()+(*nsit)-counter);
      counter++;
    }

    unsigned int nodeiter;
    // loop over all free nodes of condition
    for (nodeiter=0; nodeiter<ngid.size();nodeiter++)
    {
      vector<int> ngid_ele = defnodeIDs;
      ngid_ele.push_back(ngid[nodeiter]);
      const int numnodes=ngid_ele.size();

      remove_copy_if(&ngid_ele[0], &ngid_ele[0]+numnodes,
                       inserter(rownodeset, rownodeset.begin()),
                       not1(DRT::UTILS::MyGID(actnoderowmap)));
      // copy node ids specified in condition to colnodeset
      copy(&ngid_ele[0], &ngid_ele[0]+numnodes,
            inserter(colnodeset, colnodeset.begin()));

      // construct constraint nodes, which use the same global id as the standard nodes
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
        RCP<DRT::Element> constraintele = DRT::UTILS::Factory(element_name,"Polynomial", nodeiter+startID, myrank);
        // set the same global node ids to the ale element
        constraintele->SetNodeIds(ngid_ele.size(), &(ngid_ele[0]));
        // add constraint element
        newdis->AddElement(constraintele);
      }
      // save the connection between element and condition
      eletocondID_[nodeiter+startID]=(*conditer)->GetInt("ConditionID");
      eletocondvecindex_[nodeiter+startID]=index;
    }
    //adjust starting ID for next condition, in this case nodeiter=ngid.size(), hence the counter is larger than the ID
    // of the last element
    startID+=nodeiter;

    // now care about the parallel distribution and ghosting.
    // So far every processor only knows about his nodes

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
    //put new discretization into the map
    newdiscmap[(*conditer)->GetInt("ConditionID")]=newdis;
    // increase counter
    index++;
  }

  startID--; // set counter back to ID of the last element
  return newdiscmap;
}



/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void UTILS::MPConstraint3::EvaluateConstraint(
    RCP<DRT::Discretization> disc,
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
    // some useful data for computation
    DRT::Element* actele = disc->lColElement(i);
    int eid=actele->Id();
    int condID = eletocondID_.find(eid)->second;
    DRT::Condition* cond=constrcond_[eletocondvecindex_.find(eid)->second];
    params.set< RCP<DRT::Condition> >("condition", rcp(cond,false));

    // computation only if time is larger or equal than initialization time for constraint
    if(inittimes_.find(condID)->second<=time)
    {
      // initialize if it is the first time condition is evaluated
      if(activecons_.find(condID)->second==false)
      {
        const string action = params.get<string>("action");
        RCP<Epetra_Vector> displast=params.get<RCP<Epetra_Vector> >("old disp");
        SetConstrState("displacement",displast);
        // last converged step is used reference
        Initialize(params,systemvector2);
        RCP<Epetra_Vector> disp=params.get<RCP<Epetra_Vector> >("new disp");
        SetConstrState("displacement",disp);
        params.set("action",action);
      }

      //define global and local index of this bc in redundant vectors
      const int offsetID = params.get<int>("OffsetID");
      int gindex = eid-offsetID;
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
      if (assemblevec3) elevector3.Size(1);
      params.set("ConditionID",eid);

      // call the element evaluate method
      int err = actele->Evaluate(params,*disc,lm,elematrix1,elematrix2,
                                 elevector1,elevector2,elevector3);
      if (err) dserror("Proc %d: Element %d returned err=%d",disc->Comm().MyPID(),eid,err);

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

      //loadcurve business
      const vector<int>*    curve  = cond->Get<vector<int> >("curve");
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

/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void UTILS::MPConstraint3::InitializeConstraint(RCP<DRT::Discretization> disc,
    ParameterList&        params,
    RCP<Epetra_Vector>    systemvector)
{
  if (!(disc->Filled())) dserror("FillComplete() was not called");
  if (!(disc->HaveDofs())) dserror("AssignDegreesOfFreedom() was not called");

  // define element matrices and vectors
  Epetra_SerialDenseMatrix elematrix1;
  Epetra_SerialDenseMatrix elematrix2;
  Epetra_SerialDenseVector elevector1;
  Epetra_SerialDenseVector elevector2;
  Epetra_SerialDenseVector elevector3;

  // loop over column elements
  const double time = params.get("total time",-1.0);
  const int numcolele = disc->NumMyColElements();
  for (int i=0; i<numcolele; ++i)
  {
    // some useful data for computation
    DRT::Element* actele = disc->lColElement(i);
    int eid=actele->Id();
    int condID = eletocondID_.find(eid)->second;
    DRT::Condition* cond=constrcond_[eletocondvecindex_.find(eid)->second];
    params.set< RCP<DRT::Condition> >("condition", rcp(cond,false));

    // get element location vector, dirichlet flags and ownerships
    vector<int> lm;
    vector<int> lmowner;
    actele->LocationVector(*disc,lm,lmowner);
    // get dimension of element matrices and vectors
    // Reshape element matrices and vectors and init to zero
    const int eledim = (int)lm.size();
    elematrix1.Shape(eledim,eledim);
    elematrix2.Shape(eledim,eledim);
    elevector1.Size(eledim);
    elevector2.Size(eledim);
    elevector3.Size(1);
    // call the element evaluate method
    int err = actele->Evaluate(params,*disc,lm,elematrix1,elematrix2,
                               elevector1,elevector2,elevector3);
    if (err) dserror("Proc %d: Element %d returned err=%d",disc->Comm().MyPID(),actele->Id(),err);

    //assembly
    vector<int> constrlm;
    vector<int> constrowner;
    int offsetID = params.get<int>("OffsetID");
    constrlm.push_back(eid-offsetID);
    constrowner.push_back(actele->Owner());
    LINALG::Assemble(*systemvector,elevector3,constrlm,constrowner);

    // loadcurve business
    const vector<int>*    curve  = cond->Get<vector<int> >("curve");
    int curvenum = -1;
    if (curve) curvenum = (*curve)[0];
    double curvefac = 1.0;
    bool usetime = true;
    if (time<0.0) usetime = false;
    if (curvenum>=0 && usetime)
      curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);

    // Get ConditionID of current condition if defined and write value in parameterlist
    char factorname[30];
    sprintf(factorname,"LoadCurveFactor %d",condID);
    params.set(factorname,curvefac);

  }
  return;
} // end of InitializeConstraint

#endif
