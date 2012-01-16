/*!------------------------------------------------------------------------------------------------*
\file startvalues.cpp

\brief provides the Extrapolation class

<pre>
Maintainer: Martin Winklmaier
            winklmaier@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15241
</pre>
 *------------------------------------------------------------------------------------------------*/

#ifdef CCADISCRET


#include "timeInt_std_extrapolation.H"


/*------------------------------------------------------------------------------------------------*
 * Extrapolation constructor                                                     winklmaier 11/11 *
 *------------------------------------------------------------------------------------------------*/
XFEM::Extrapolation::Extrapolation(
    XFEM::TIMEINT& timeInt,
    INPAR::COMBUST::XFEMTimeIntegration timeIntType,
    const RCP<Epetra_Vector> veln,
    const double& dt,
    const RCP<COMBUST::FlameFront> flamefront,
    bool initialize
) : STD(timeInt,timeIntType,veln,dt,flamefront,initialize)
{
  return;
} // end constructor



/*------------------------------------------------------------------------------------------------*
 * call the computation based on an extrapolation                                winklmaier 11/11 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::Extrapolation::compute(
    vector<RCP<Epetra_Vector> > newRowVectorsn,
    vector<RCP<Epetra_Vector> > newRowVectorsnp
)
{
  if (FGIType_==FRSNot1_)
    return;

  handleVectors(newRowVectorsn,newRowVectorsnp);

  resetState(TimeIntData::basicStd_,TimeIntData::extrapolateStd_);

  for (vector<TimeIntData>::iterator data=timeIntData_->begin();
      data!=timeIntData_->end(); data++)
  {
    extrapolationMain(&*data);
  }

#ifdef PARALLEL
  exportFinalData();
#endif
  setFinalData();
} // end compute



/*------------------------------------------------------------------------------------------------*
 * call the computation based on an extrapolation                                winklmaier 11/11 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::Extrapolation::compute(vector<RCP<Epetra_Vector> > newRowVectors)
{
  if (oldVectors_.size() != newRowVectors.size())
  {
    cout << "sizes are " << oldVectors_.size() << " and " << newRowVectors.size() << endl;
    dserror("Number of state-vectors at new and old discretization are different!");
  }

  newVectors_ = newRowVectors;

  for (vector<TimeIntData>::iterator data=timeIntData_->begin();
      data!=timeIntData_->end(); data++)
  {
    extrapolationMain(&*data);
  }

#ifdef PARALLEL
  exportFinalData();
#endif
  setFinalData();
}// end compute



/*------------------------------------------------------------------------------------------------*
 * extrapolation of values to data-requiring nodes:                              winklmaier 10/11 *
 * a straight line through the data-requiring node and the nearest node on the correct interface  *
 * side is set up, two appropriate points on this line are used for extrapolation                 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::Extrapolation::extrapolationMain(
    TimeIntData* data
)
{
  if (data->state_!=TimeIntData::extrapolateStd_)
    dserror("extrapolation must be called with according state");

  const int nsd = 3;

  DRT::Element* startele = NULL; // element of startpoint
  DRT::Element* midele = NULL; // element of midpoint

  LINALG::Matrix<nsd,1> startpoint(true); // global coordinates of startpoint
  LINALG::Matrix<nsd,1> xistartpoint(true); // local coordinates of startpoint

  LINALG::Matrix<nsd,1> midpoint(true); // global coordinates of midpoint
  LINALG::Matrix<nsd,1> ximidpoint(true); // local coordinates of midpoint

  LINALG::Matrix<nsd,1> endpoint(data->node_.X());

//    cout << "searching node = endpoint = " << endpoint << endl;
  // identify final element and local coordinates of startpoint and midpoint for extrapolation
  bisection(data,startele,startpoint,xistartpoint,midele,midpoint,ximidpoint);

//    cout << endl << "startpoint is " << startpoint;
//    cout << "midpoint is " << midpoint;

  // compute the constants for the extrapolation:
  // value = c1*valuestartpoint + c2*valuemidpoint
  LINALG::Matrix<nsd,1> dist1;
  LINALG::Matrix<nsd,1> dist2;

  dist1.Update(1.0,midpoint,-1.0,startpoint);
  dist2.Update(1.0,endpoint,-1.0,midpoint);
//  cout << "distance from startpoint to midpoint is " << dist1;
//  cout << "distance from midpoint to endpoint is " << dist2;

  if (dist1.Norm2()<1e-14 or dist2.Norm2()<1e-14)
    dserror("something wrong in bisection");

  double c1 = - dist2.Norm2()/dist1.Norm2(); // 1 + dist2/dist1
  double c2 = 1.0 + dist2.Norm2()/dist1.Norm2(); // dist2/dist1

  // get the velocities and the pressures at the start- and midpoint
  vector<LINALG::Matrix<nsd,1> > velstartpoint(oldVectors_.size(),LINALG::Matrix<nsd,1>(true));
  vector<LINALG::Matrix<nsd,1> > velmidpoint(oldVectors_.size(),LINALG::Matrix<nsd,1>(true));

  vector<double> presstartpoint(oldVectors_.size(),0.0);
  vector<double> presmidpoint(oldVectors_.size(),0.0);

  callInterpolation(startele,xistartpoint,velstartpoint,presstartpoint);
  callInterpolation(midele,ximidpoint,velmidpoint,presmidpoint);

  //  cout << "pres at startpoint is " << presstartpoint[0];
  //  cout << "pres at midpoint is " << presmidpoint[0];

  // compute the final velocities and pressure due to the extrapolation
  vector<LINALG::Matrix<nsd,1> > velendpoint(oldVectors_.size(),LINALG::Matrix<nsd,1>(true));
  vector<double> presendpoint(oldVectors_.size(),0.0);

  for (size_t index=0;index<oldVectors_.size();index++)
  {
    velendpoint[index].Update(c1,velstartpoint[index],c2,velmidpoint[index]);
    presendpoint[index] = c1*presstartpoint[index] + c2*presmidpoint[index];

    //	if (index == 0)
    //		cout << "final pressure is " << presendpoint[index] <<
    //				" and final velocity is " << velendpoint[index];

  } // loop over vectors to be set

  data->startOwner_ = vector<int>(1,myrank_);
  data->velValues_ = velendpoint;
  data->presValues_ = presendpoint;
  data->state_ = TimeIntData::doneStd_;
}



/*------------------------------------------------------------------------------------------------*
 * perform a bisection on the line startpoint-endpoint                           winklmaier 11/11 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::Extrapolation::bisection(
    TimeIntData* data,
    DRT::Element*& startele,
    LINALG::Matrix<3,1>& startpoint,
    LINALG::Matrix<3,1>& xistartpoint,
    DRT::Element*& midele,
    LINALG::Matrix<3,1>& midpoint,
    LINALG::Matrix<3,1>& ximidpoint
)
{
  // general initialization
  const int nsd = 3;

  DRT::Node* node = discret_->gNode(data->startGid_[0]); // startpoint node on current proc
  LINALG::Matrix<nsd,1> nodecoords(node->X());

  vector<const DRT::Element*> eles;
  addPBCelements(node,eles);
  const int numele=eles.size();

  startpoint = nodecoords; // first point used for extrapolation
  midpoint = nodecoords; // second point used for extrapolation
  LINALG::Matrix<nsd,1> endpoint(data->node_.X()); // coordinates of data-requiring node
//    cout << endl << "initial startpoint and midpoint is " << nodecoords;

  LINALG::Matrix<nsd,1> pointTmp; // temporarily point used for computations and for bisection
  pointTmp.Update(0.5,startpoint,0.5,endpoint); // midpoint of startpoint and endpoint

  LINALG::Matrix<nsd,1> dist;
  dist.Update(1.0,endpoint,-1.0,startpoint);

  LINALG::Matrix<nsd,1> xipointTmp(true);

  int iter = 0;
  const int crit_max_iter = 100; // maximal iteration number for critical cases
  int curr_max_iter = 10; // currently used iteration number
  int std_max_iter = 10; // iteration number usually sufficient
  bool elefound = false;

  DRT::Element* eletmp = NULL;
  bool elefoundtmp = false;

  // prework: search for the element around "startpoint" into "endpoint"-direction
  while(true)
  {
//    cout << "potential midpoint is " << pointTmp;
    iter++;
    for (int i=0; i<numele; i++)
    {
      eletmp = (DRT::Element*)eles[i];
      callXToXiCoords(eletmp,pointTmp,xipointTmp,elefoundtmp);
      if (elefoundtmp) // usually loop can stop here, just in special cases of critical cuts problems might happen
      {
        midele = eletmp;
        elefound = true;
        if (intersectionStatus(midele)==XFEM::TIMEINT::cut_) // really cut -> ok
          break;
        else if (intersectionStatus(midele)==XFEM::TIMEINT::uncut_) // really uncut -> ok
          break;
        else // special cases -> try to take another element
          ; // do not break, but potential element is saved
      }
    }

    if (elefound)
      break;
    else // corresponds to CFL > 1 -> not very good, but possible
      pointTmp.Update(-pow(0.5,iter+1),dist,1.0);

    if (iter==crit_max_iter) break;
  }

  curr_max_iter = crit_max_iter;

  // search for midpoint with bisection of the straight line:
  //	distances p1-p2 and p2-pend as even as possible
  //	-> p2 near midpoint of p1 and pend as far as possible
  //	gives back either a point between pend and p1 or p2 = p1
  for (int i=iter;i<=curr_max_iter;i++)
  {
//    cout << "potential midpoint is " << pointTmp;
    callXToXiCoords(midele,pointTmp,xipointTmp,elefound);

    if (!elefound)
      dserror("two points of a line segment in one element "
          "-> also any point between them should be in the same element.");

    if (interfaceSideCompare(midele,pointTmp,0,data->phiValue_) == true) // Lagrangian origin and original node on different interface sides
    {
//      cout << "current midpoint is " << pointTmp;
      curr_max_iter = std_max_iter; // usable point found
      midpoint = pointTmp; // possible point
      ximidpoint = xipointTmp;
      pointTmp.Update(pow(0.5,i+1),dist,1.0); // nearer to endpoint
    }
    else
      pointTmp.Update(-pow(0.5,i+1),dist,1.0); // nearer to startpoint
  }

  // if element is (nearly) touched, the above bisection might fail
  if (midpoint == nodecoords) // nothing was changed above
  {
    cout << endl << endl << "WARNING: this case should no more happen!" << endl << endl;
    midele = (DRT::Element*)eles[0];
    callXToXiCoords(midele,midpoint,ximidpoint,elefound);
  }
//  cout << "final midpoint is " << midpoint << " in " << *midele;

  // get current distances of the three points
  LINALG::Matrix<nsd,1> dist1; // distance from startpoint to midpoint
  LINALG::Matrix<nsd,1> dist2; // distance from midpoint to endpoint

  dist1.Update(1.0,midpoint,-1.0,startpoint);
  dist2.Update(1.0,endpoint,-1.0,midpoint);

  curr_max_iter = crit_max_iter;

  // compute potentially final startpoint
  if ((dist1.Norm2()/dist.Norm2() > 1.0/3.0) and (dist2.Norm2()/dist.Norm2() > 1.0/3.0)) // distances are ok
  {
    startele = midele;
    callXToXiCoords(startele,startpoint,xistartpoint,elefound);

    if (!elefound)
      dserror("extrapolation should be done within one convex element. Thus this should work");

    if (interfaceSideCompare(startele,startpoint,0,data->phiValue_) == false) // Lagrangian origin and original node on different interface sides
      dserror("node which was evaluated to be the nearest node on correct side is on wrong side???");
  }
  else if (dist2.Norm2()/dist.Norm2() <= 1.0/3.0) // midpoint much nearer at endpoint than at startpoint
  {
    startpoint.Update(1.0,endpoint,-2.0,dist2); // same, moderate distances
    startele = midele;
    callXToXiCoords(startele,startpoint,xistartpoint,elefound);

    if (!elefound)
      dserror("extrapolation should be done within one convex element. Thus this should work");

    if (interfaceSideCompare(startele,startpoint,0,data->phiValue_) == false) // Lagrangian origin and original node on different interface sides
    {
      LINALG::Matrix<nsd,1> startpointLeft = startpoint;
      LINALG::Matrix<nsd,1> xistartpointLeft(true);
      LINALG::Matrix<nsd,1> startpointRight = startpoint;
      LINALG::Matrix<nsd,1> xistartpointRight(true);

      // get the potential startpoint on the "node"-side of the current, not usable startpoint
      dist.Update(1.0,startpoint,-1.0,nodecoords);
      pointTmp.Update(0.5,startpoint,0.5,nodecoords);
      for (int i=1;i<=curr_max_iter;i++)
      {
        callXToXiCoords(startele,pointTmp,xipointTmp,elefound);

        if (interfaceSideCompare(startele,pointTmp,0,data->phiValue_) == true) // Lagrangian origin and original node on different interface sides
        {
          curr_max_iter = std_max_iter;
          startpointLeft = pointTmp; // possible point
          pointTmp.Update(pow(0.5,i+1),dist,1.0); // nearer to optimal startpoint
        }
        else
          pointTmp.Update(-pow(0.5,i+1),dist,1.0); // nearer to not-optimal, but possible node
      }

      curr_max_iter = crit_max_iter;
      // get the potential startpoint on the "midpoint"-side of the current, not usable startpoint
      dist.Update(1.0,midpoint,-1.0,startpoint);
      pointTmp.Update(0.5,midpoint,0.5,startpoint);
      for (int i=1;i<=curr_max_iter;i++)
      {
        callXToXiCoords(startele,pointTmp,xipointTmp,elefound);

        if (interfaceSideCompare(startele,pointTmp,0,data->phiValue_) == true) // Lagrangian origin and original node on different interface sides
        {
          curr_max_iter = std_max_iter;
          startpointRight = pointTmp; // possible point
          pointTmp.Update(-pow(0.5,i+1),dist,1.0); // nearer to original startpoint
        }
        else
          pointTmp.Update(+pow(0.5,i+1),dist,1.0); // nearer to midpoint, worse ratio, but possible
      }

      // check if startpoints changed
      if (startpointLeft == startpoint) // left possible startpoint didn't change
        startpointLeft = nodecoords; // possible, but not very good point due to high ratio
      if (startpointRight == startpoint) // right possible startpoint didn't change
        startpointRight = nodecoords; // possible, but not very good point due to high ratio

      // check which startpoint is the better one
      dist.Update(1.0,midpoint,-1.0,startpointLeft);
      double ratioLeft = dist1.Norm2()/dist.Norm2(); // dist is the greater entry in this case

      dist.Update(1.0,midpoint,-1.0,startpointRight);
      double ratioRight =  dist.Norm2()/dist1.Norm2(); // dist1 is the greater entry in this case

      if (ratioLeft > ratioRight) // nearer-to-one ratio = better
        startpoint = startpointLeft;
      else
        startpoint = startpointRight;

      // compute final local coordinates of the startpoint
      callXToXiCoords(startele,startpoint,xistartpoint,elefound);
    }
  }
  else // distance startpoint - midpoint much small than distance midpoint - endpoint
  {
    pointTmp.Update(1.0,endpoint,-1.5,dist2); // dist2 = 2*dist1 -> moderate factor and moderate total lenght
    dist.Update(1.0,midpoint,-1.0,pointTmp);

    iter = 0;
    // possibly the startpoint is too far away from the "node"
    // -> reduce distance until startpoint is in element adjacent to the "node"
    while(true)
    {
      iter++;
      for (int i=0; i<numele; i++)
      {
        startele = (DRT::Element*)eles[i];
        callXToXiCoords(startele,pointTmp,xipointTmp,elefound);
        if (elefound) break;
      }

      if (elefound) break;
      if (iter==curr_max_iter) break;

      pointTmp.Update(+pow(0.5,iter),dist,1.0);
    }

    if (!elefound)
      dserror("extrapolation should be done within one convex element. Thus this should work");

    // get the potential startpoint on the "node"-side of the current, not usable startpoint
    for (int i=iter;i<=curr_max_iter+1;i++)
    {
      if (!elefound)
        pointTmp.Update(+pow(0.5,i),dist,1.0); // nearer to midpoint
      else
      {
        if (interfaceSideCompare(startele,pointTmp,0,data->phiValue_) == true) // Lagrangian origin and original node on different interface sides
        {
          curr_max_iter = std_max_iter;
          startpoint = pointTmp; // possible point
          xistartpoint = xipointTmp;
          pointTmp.Update(-pow(0.5,i),dist,1.0); // nearer to original startpoint
        }
        else
          pointTmp.Update(+pow(0.5,i),dist,1.0); // nearer to midpoint
      }

      if (i<=curr_max_iter)
      {
        callXToXiCoords(startele,pointTmp,xipointTmp,elefound);
        if (!elefound)
        {
          for (int i=0; i<numele; i++)
          {
            startele = (DRT::Element*)eles[i];
            callXToXiCoords(startele,pointTmp,xipointTmp,elefound);
            if (elefound) break;
          }
        }
      }
    }
  }
//  cout << "final startpoint is " << startpoint << " in " << *startele;

} // end bisection



/*------------------------------------------------------------------------------------------------*
 * call the interpolation                                                        winklmaier 06/10 *
 *------------------------------------------------------------------------------------------------*/
void XFEM::Extrapolation::callInterpolation(
    DRT::Element* ele,
    LINALG::Matrix<3,1>& xi,
    vector<LINALG::Matrix<3,1> >& velValues,
    vector<double>& presValues
)
{
  switch (ele->Shape())
  {
  case DRT::Element::hex8:
  {
    const int numnode = DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex8>::numNodePerElement;
    interpolation<numnode,DRT::Element::hex8>(ele,xi,velValues,presValues);
  }
  break;
  case DRT::Element::hex20:
  {
    const int numnode = DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex20>::numNodePerElement;
    interpolation<numnode,DRT::Element::hex20>(ele,xi,velValues,presValues);
  }
  break;
  default:
    dserror("xfem assembly type not yet implemented in time integration");
  }
}



/*------------------------------------------------------------------------------------------------*
 * perform the interpolation                                                     winklmaier 06/10 *
 *------------------------------------------------------------------------------------------------*/
template<const int numnode, DRT::Element::DiscretizationType DISTYPE>
void XFEM::Extrapolation::interpolation(
    DRT::Element* ele,
    LINALG::Matrix<3,1>& xi,
    vector<LINALG::Matrix<3,1> >& velValues,
    vector<double>& presValues
)
{
  const int nsd = 3;

  // node velocities of the element nodes for the data that should be changed
  vector<LINALG::Matrix<nsd,2*numnode> > nodeveldata(oldVectors_.size(),LINALG::Matrix<nsd,2*numnode>(true));
  // node pressures of the element nodes for the data that should be changed
  vector<LINALG::Matrix<1,2*numnode> > nodepresdata(oldVectors_.size(),LINALG::Matrix<1,2*numnode>(true));
#ifdef COMBUST_NORMAL_ENRICHMENT
  vector<LINALG::Matrix<1,numnode> > nodevelenrdata(oldVectors_.size(),LINALG::Matrix<1,numnode>(true));
  LINALG::Matrix<1,numnode> nodevelenr(true);
#endif

  // required enriched shape functions
  LINALG::Matrix<2*numnode,1> enrShapeFcnVel(true);
  LINALG::Matrix<2*numnode,1> enrShapeFcnPres(true);

  // dummies for function call
  LINALG::Matrix<nsd,nsd> dummy_jacobi(true);
  LINALG::Matrix<numnode,1> dummy_shpFcn(true);
  LINALG::Matrix<nsd,2*numnode> dummy_enrShapeXYVelDeriv1(true);
  LINALG::Matrix<nsd,2*numnode> dummy_enrShapeXYPresDeriv1(true);

  // evaluate data for the given point
#ifdef COMBUST_NORMAL_ENRICHMENT
#ifdef COLLAPSE_FLAME_NORMAL
  LINALG::Matrix<nsd,1> normal(node->X());
  normal(2) = 0.0;
  normal.Scale(1.0/normal.Norm2());
#endif
  ApproxFuncNormalVector<2,2*numnode> shp;
  pointdataXFEMNormal<numnode,DISTYPE>(
      ele,
#ifdef COLLAPSE_FLAME_NORMAL
      normal,
#endif
      xi,
      xji,
      shapeFcn,
      enrShapeFcnPres,
      enrShapeXYPresDeriv1,
      shp,
      false
  );
#else
  pointdataXFEM<numnode,DISTYPE>(
      ele,
      xi,
      dummy_jacobi,
      dummy_shpFcn,
      enrShapeFcnVel,
      enrShapeFcnPres,
      dummy_enrShapeXYVelDeriv1,
      dummy_enrShapeXYPresDeriv1,
      false
  );
#endif

  const int* elenodeids = ele->NodeIds();

  int dofcounterVelx = 0;
  int dofcounterVely = 0;
  int dofcounterVelz = 0;
  int dofcounterPres = 0;
#ifdef COMBUST_NORMAL_ENRICHMENT
  int dofcounterVeln = 0;
#endif

  for (int nodeid=0;nodeid<ele->NumNode();nodeid++) // loop over element nodes
  {
    // get nodal velocities and pressures with help of the field set of node
    const std::set<XFEM::FieldEnr>& fieldEnrSet(olddofman_->getNodeDofSet(elenodeids[nodeid]));
    for (set<XFEM::FieldEnr>::const_iterator fieldenr = fieldEnrSet.begin();
        fieldenr != fieldEnrSet.end();++fieldenr)
    {
      const DofKey<onNode> olddofkey(elenodeids[nodeid], *fieldenr);
      const int olddofpos = oldNodalDofColDistrib_.find(olddofkey)->second;
      switch (fieldenr->getEnrichment().Type())
      {
      case XFEM::Enrichment::typeStandard :
      case XFEM::Enrichment::typeJump :
      case XFEM::Enrichment::typeVoidFSI :
      case XFEM::Enrichment::typeVoid :
      case XFEM::Enrichment::typeKink :
      {
        if (fieldenr->getField() == XFEM::PHYSICS::Velx)
        {
          for (size_t index=0;index<oldVectors_.size();index++)
            nodeveldata[index](0,dofcounterVelx) = (*oldVectors_[index])[olddofcolmap_.LID(olddofpos)];
          dofcounterVelx++;
        }
        else if (fieldenr->getField() == XFEM::PHYSICS::Vely)
        {
          for (size_t index=0;index<oldVectors_.size();index++)
            nodeveldata[index](1,dofcounterVely) = (*oldVectors_[index])[olddofcolmap_.LID(olddofpos)];
          dofcounterVely++;
        }
        else if (fieldenr->getField() == XFEM::PHYSICS::Velz)
        {
          for (size_t index=0;index<oldVectors_.size();index++)
            nodeveldata[index](2,dofcounterVelz) = (*oldVectors_[index])[olddofcolmap_.LID(olddofpos)];
          dofcounterVelz++;
        }
#ifdef COMBUST_NORMAL_ENRICHMENT
        else if (fieldenr->getField() == XFEM::PHYSICS::Veln)
        {
          for (size_t index=0;index<oldVectors_.size();index++)
            nodevelenrdata[index](0,dofcounterVeln) = (*oldVectors_[index])[olddofcolmap_.LID(olddofpos)];
          dofcounterVeln++;
        }
#endif
        else if (fieldenr->getField() == XFEM::PHYSICS::Pres)
        {
          for (size_t index=0;index<oldVectors_.size();index++)
            nodepresdata[index](0,dofcounterPres) = (*oldVectors_[index])[olddofcolmap_.LID(olddofpos)];
          dofcounterPres++;
        }
        else
        {
          cout << XFEM::PHYSICS::physVarToString(fieldenr->getField()) << endl;
          dserror("not implemented physical field!");
        }
        break;
      }
      case XFEM::Enrichment::typeUndefined :
      default :
      {
        cout << fieldenr->getEnrichment().enrTypeToString(fieldenr->getEnrichment().Type()) << endl;
        dserror("unknown enrichment type");
        break;
      }
      } // end switch enrichment
    } // end loop over fieldenr
  } // end loop over element nodes
  // compute shape functions

  LINALG::Matrix<nsd,1> vel(true);
  LINALG::Matrix<1,1> pres(true);

  for (size_t index=0;index<oldVectors_.size();index++)
  {
    vel.Multiply(nodeveldata[index],enrShapeFcnVel); // v = theta*Dv^n+1/Dx*v^n+1+(1-theta)*Dv^n/Dx*v^n
    velValues[index]=vel;

    pres.Multiply(nodepresdata[index],enrShapeFcnPres); // p = p_n + dt*(theta*Dp^n+1/Dx*v^n+1+(1-theta)*Dp^n/Dx*v^n)
    presValues[index] = pres(0);
  } // loop over vectors to be set
}


#endif // CCADISCRET



