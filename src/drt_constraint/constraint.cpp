/*!----------------------------------------------------------------------
\file constraint.cpp

\brief Basic constraint class, dealing with constraints living on boundaries
<pre>
Maintainer: Thomas Kloeppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/kloeppel
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "constraint.H"
#include "iostream"
#include "../drt_lib/drt_condition_utils.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_timecurve.H"


/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
UTILS::Constraint::Constraint(RCP<DRT::Discretization> discr,
        const string& conditionname,
        int& offsetID,
        int& maxID):
actdisc_(discr)
{
  actdisc_->GetCondition(conditionname,constrcond_);
  if (constrcond_.size())
  {
    constrtype_=GetConstrType(conditionname);
    for (unsigned int i=0; i<constrcond_.size();i++)
    {
      //constrcond_[i]->Print(cout);
      int condID=(*(constrcond_[i]->Get<vector<int> >("ConditionID")))[0];
      if (condID>maxID)
      {
        maxID=condID;
      }
      if (condID<offsetID)
      {
        offsetID=condID;
      }
      
      vector<double> myinittime=*(constrcond_[i]->Get<vector<double> >("activTime"));
      if (myinittime.size())
      {
        inittimes_.insert(pair<int,double>(condID,myinittime[0]));
        activecons_.insert(pair<int,bool>(condID,false));
      }
      else
      {
        inittimes_.insert(pair<int,double>(condID,0.0));
        activecons_.insert(pair<int,bool>(condID,false));
      }
    }
  }
  else
  {
    constrtype_=none;
  }  
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
UTILS::Constraint::Constraint(RCP<DRT::Discretization> discr,
        const string& conditionname):
actdisc_(discr)
{
  actdisc_->GetCondition(conditionname,constrcond_);
  if (constrcond_.size())
  {
    constrtype_=GetConstrType(conditionname);
    for (unsigned int i=0; i<constrcond_.size();i++)
    {
      vector<double> myinittime=*(constrcond_[i]->Get<vector<double> >("activTime"));
      int condID=constrcond_[i]->GetInt("ConditionID");
      if (myinittime.size())
      {
        inittimes_.insert(pair<int,double>(condID,myinittime[0]));
        activecons_.insert(pair<int,bool>(condID,false));
      }
      else
      {
        inittimes_.insert(pair<int,double>(condID,0.0));
        activecons_.insert(pair<int,bool>(condID,false));
      }
    }
  }
  else
  {
    constrtype_=none;
  }  
}

/*-----------------------------------------------------------------------*
|(private)                                                       tk 07/08|
*-----------------------------------------------------------------------*/
UTILS::Constraint::ConstrType UTILS::Constraint::GetConstrType(const string& name)
{
  if (name=="VolumeConstraint_3D")
    return volconstr3d;
  else if (name=="AreaConstraint_3D")
    return areaconstr3d;
  else if (name=="AreaConstraint_2D")
    return areaconstr2d;
  else if (name=="MPC_NodeOnPlane_3D")
    return mpcnodeonplane3d;
  else if (name=="MPC_NodeOnLine_2D")
    return mpcnodeonline2d;
  else if (name=="MPC_NormalComponent_3D")
    return mpcnormalcomp3d;
  return none;
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine computes ref base values and activates conditions |
*------------------------------------------------------------------------*/
void UTILS::Constraint::Initialize(
    ParameterList&        params,
    RCP<Epetra_Vector>    systemvector3)
{
  // choose action
  switch (constrtype_)
  {
    case volconstr3d: 
      params.set("action","calc_struct_constrvol");
    break;
    case areaconstr3d:
      params.set("action","calc_struct_constrarea");
    break;
    case areaconstr2d:
      params.set("action","calc_struct_constrarea");
    break;
    case none:
      return;
    default:
      dserror("Unknown constraint/monitor type to be evaluated in Constraint class!");
  }
  // start computing 
  InitializeConstraint(params,systemvector3);
  return;
}

/*------------------------------------------------------------------------*
|(public)                                                       tk 08/08  |
|Initialization routine activates conditions (restart)                    |
*------------------------------------------------------------------------*/
void UTILS::Constraint::Initialize
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
    if((inittimes_.find(condID)->second<=time) && (activecons_.find(condID)->second==false)) 
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
void UTILS::Constraint::Evaluate(
    ParameterList&        params,
    RCP<LINALG::SparseMatrix> systemmatrix1,
    RCP<LINALG::SparseMatrix> systemmatrix2,
    RCP<Epetra_Vector>    systemvector1,
    RCP<Epetra_Vector>    systemvector2,
    RCP<Epetra_Vector>    systemvector3)
{
  switch (constrtype_)
  {
    case volconstr3d: 
      params.set("action","calc_struct_volconstrstiff");
    break;
    case areaconstr3d:
      params.set("action","calc_struct_areaconstrstiff");
    break;
    case areaconstr2d:
      params.set("action","calc_struct_areaconstrstiff");
    break;
    case none:
      return;  
    default:
      dserror("Wrong constraint type to evaluate systemvector!");
  }
  EvaluateConstraint(params,systemmatrix1,systemmatrix2,systemvector1,systemvector2,systemvector3);
  return;
}

/*-----------------------------------------------------------------------*
 |(private)                                                     tk 07/08 |
 |Evaluate method, calling element evaluates of a condition and          |
 |assembing results based on this conditions                             |
 *----------------------------------------------------------------------*/
void UTILS::Constraint::EvaluateConstraint(
    ParameterList&        params,
    RCP<LINALG::SparseMatrix> systemmatrix1,
    RCP<LINALG::SparseMatrix> systemmatrix2,
    RCP<Epetra_Vector>    systemvector1,
    RCP<Epetra_Vector>    systemvector2,
    RCP<Epetra_Vector>    systemvector3)
{  
  if (!(actdisc_->Filled())) dserror("FillComplete() was not called");
  if (!actdisc_->HaveDofs()) dserror("AssignDegreesOfFreedom() was not called");
  // get the current time
  const double time = params.get("total time",-1.0);

  const bool assemblemat1 = systemmatrix1!=Teuchos::null;
  const bool assemblemat2 = systemmatrix2!=Teuchos::null;
  const bool assemblevec1 = systemvector1!=Teuchos::null;
  const bool assemblevec2 = systemvector2!=Teuchos::null;
  const bool assemblevec3 = systemvector3!=Teuchos::null;

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (unsigned int i = 0; i < constrcond_.size(); ++i)
  {
    DRT::Condition& cond = *(constrcond_[i]);
    
    // get values from time integrator to scale matrices with 
    double scStiff = params.get("scaleStiffEntries",1.0);
    double scConMat = params.get("scaleConstrMat",1.0);

    // Get ConditionID of current condition if defined and write value in parameterlist
    int condID=cond.GetInt("ConditionID");
    params.set("ConditionID",condID);
    
    // is conditions supposed to be active?
    if(inittimes_.find(condID)->second<=time)
    {
      // is conditions already labeled as active?
      if(activecons_.find(condID)->second==false)
      {
        const string action = params.get<string>("action"); 
        RCP<Epetra_Vector> displast=params.get<RCP<Epetra_Vector> >("old disp");
        actdisc_->SetState("displacement",displast);
        Initialize(params,systemvector2);
        RCP<Epetra_Vector> disp=params.get<RCP<Epetra_Vector> >("new disp");
        actdisc_->SetState("displacement",disp);
        params.set("action",action);
      }
      
      // Evaluate loadcurve if defined. Put current load factor in parameterlist
      const vector<int>*    curve  = cond.Get<vector<int> >("curve");
      int curvenum = -1;
      if (curve) curvenum = (*curve)[0];
      double curvefac = 1.0;
      if (curvenum>=0 )
        curvefac = DRT::UTILS::TimeCurveManager::Instance().Curve(curvenum).f(time);
      
      // global and local ID of this bc in the redundant vectors
      const int offsetID = params.get<int>("OffsetID");
      int gindex = condID-offsetID;
      const int lindex = (systemvector3->Map()).LID(gindex);
      
      // store loadcurve values
      RCP<Epetra_Vector> timefact = params.get<RCP<Epetra_Vector> >("vector curve factors");
      timefact->ReplaceGlobalValues(1,&curvefac,&gindex);
      
      // Get the current lagrange multiplier value for this condition  
      const RCP<Epetra_Vector> lagramul = params.get<RCP<Epetra_Vector> >("LagrMultVector");
      const double lagraval = (*lagramul)[lindex];
      
      // elements might need condition 
      params.set<RefCountPtr<DRT::Condition> >("condition", rcp(&cond,false));

      // define element matrices and vectors
      Epetra_SerialDenseMatrix elematrix1;
      Epetra_SerialDenseMatrix elematrix2;
      Epetra_SerialDenseVector elevector1;
      Epetra_SerialDenseVector elevector2;
      Epetra_SerialDenseVector elevector3;

      map<int,RefCountPtr<DRT::Element> >& geom = cond.Geometry();
      // if (geom.empty()) dserror("evaluation of condition with empty geometry");
      // no check for empty geometry here since in parallel computations
      // can exist processors which do not own a portion of the elements belonging
      // to the condition geometry
      map<int,RefCountPtr<DRT::Element> >::iterator curr;
      for (curr=geom.begin(); curr!=geom.end(); ++curr)
      {
        // get element location vector and ownerships
        vector<int> lm;
        vector<int> lmowner;
        curr->second->LocationVector(*actdisc_,lm,lmowner);

        // get dimension of element matrices and vectors
        // Reshape element matrices and vectors and init to zero
        const int eledim = (int)lm.size();
        if (assemblemat1) elematrix1.Shape(eledim,eledim);
        if (assemblemat2) elematrix2.Shape(eledim,eledim);
        if (assemblevec1) elevector1.Size(eledim);
        if (assemblevec2) elevector2.Size(eledim);
        if (assemblevec3) elevector3.Size(1);

        // call the element specific evaluate method
        int err = curr->second->Evaluate(params,*actdisc_,lm,elematrix1,elematrix2,
            elevector1,elevector2,elevector3);
        if (err) dserror("error while evaluating elements");

        // assembly
        int eid = curr->second->Id();
        if (assemblemat1) 
        { 
          // scale with time integrator dependent value
          elematrix1.Scale(scStiff*lagraval);
          systemmatrix1->Assemble(eid,elematrix1,lm,lmowner);
        }
        if (assemblemat2)
        {
          // assemble to rectangular matrix. The column corresponds to the constraint ID.
          // scale with time integrator dependent value
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
          constrowner.push_back(curr->second->Owner());
          LINALG::Assemble(*systemvector3,elevector3,constrlm,constrowner);
        }
      }
    }
  }
  return;
} // end of EvaluateCondition

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void UTILS::Constraint::InitializeConstraint(
    ParameterList&        params,
    RCP<Epetra_Vector>    systemvector)
{
  if (!(actdisc_->Filled())) dserror("FillComplete() was not called");
  if (!actdisc_->HaveDofs()) dserror("AssignDegreesOfFreedom() was not called");
  // get the current time
  const double time = params.get("total time",-1.0);

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (unsigned int i = 0; i < constrcond_.size(); ++i)
  {
    DRT::Condition& cond = *(constrcond_[i]);

    // Get ConditionID of current condition if defined and write value in parameterlist

    int condID=cond.GetInt("ConditionID");
    params.set("ConditionID",condID);
    
    // if current time is larger than initialization time of the condition, start computing
    if((inittimes_.find(condID)->second<=time) && (!(activecons_.find(condID)->second)))
    {
      params.set<RefCountPtr<DRT::Condition> >("condition", rcp(&cond,false));
  
      // define element matrices and vectors
      Epetra_SerialDenseMatrix elematrix1;
      Epetra_SerialDenseMatrix elematrix2;
      Epetra_SerialDenseVector elevector1;
      Epetra_SerialDenseVector elevector2;
      Epetra_SerialDenseVector elevector3;
  
      map<int,RefCountPtr<DRT::Element> >& geom = cond.Geometry();
      // no check for empty geometry here since in parallel computations
      // can exist processors which do not own a portion of the elements belonging
      // to the condition geometry
      map<int,RefCountPtr<DRT::Element> >::iterator curr;
      for (curr=geom.begin(); curr!=geom.end(); ++curr)
      {
        // get element location vector and ownerships
        vector<int> lm;
        vector<int> lmowner;
        curr->second->LocationVector(*actdisc_,lm,lmowner);
  
        // get dimension of element matrices and vectors
        // Reshape element matrices and vectors and init to zero
        elevector3.Size(1);
  
        // call the element specific evaluate method
        int err = curr->second->Evaluate(params,*actdisc_,lm,elematrix1,elematrix2,
                                         elevector1,elevector2,elevector3);
        if (err) dserror("error while evaluating elements");
  
        // assembly
          
        vector<int> constrlm;
        vector<int> constrowner;
        int offsetID = params.get<int> ("OffsetID");
        constrlm.push_back(condID-offsetID);
        constrowner.push_back(curr->second->Owner());
        LINALG::Assemble(*systemvector,elevector3,constrlm,constrowner);
      }
      // remember next time, that this condition is already initialized, i.e. active
      activecons_.find(condID)->second=true;

      if (actdisc_->Comm().MyPID()==0)
      {
        cout << "Encountered a new active condition (Id = " << condID << ")  at time t = "<< time << endl;
      }
    }
    
  }
  return;
} // end of Initialize Constraint

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
vector<int> UTILS::Constraint::GetActiveCondID()
{
  vector<int> condID;
  map<int,bool>::const_iterator mapit;
  for(mapit = activecons_.begin();mapit!=activecons_.end();mapit++)
  {
    if (mapit->second)
      condID.push_back(mapit->first);
  }
  return condID;
}
#endif
