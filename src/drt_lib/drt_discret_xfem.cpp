/*!----------------------------------------------------------------------
\file drt_discret_xfem.cpp

\brief a class to manage one discretization

<pre>
-------------------------------------------------------------------------
                 BACI finite element library subsystem
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library may solemnly used in conjunction with the BACI contact library
for purposes described in the above mentioned contract.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/

#include "drt_discret_xfem.H"
#include "drt_globalproblem.H"
#include "../linalg/linalg_mapextractor.H"
#include "../drt_combust/combust_defines.H"

/*!
\brief this is a modified copy of the original version DoDirchletCondition() for XFEM problems

- A severe problem occurs if an enriched XFEM element touches a Dirichlet boundary.
  Then values have to be prescribed on the dofs. Therefore all dofs of a node (standard and enriched
  ones!) are looped. Values, time curve factors and function factors are assigned according to the
  input file. However, this causes the code to crash with a segmentation fault. The reason is that
  vectors "curve", "funct", "onoff" and "val" only have length 6. But an enriched node has 8 dofs
  (4 standard fluid ones and 3-4 enrichred ones)!

- For Axels XFSI problems this never caused trouble, because he only has the (void) enriched dofs
  (that is 4!) - the standard dofs have been kicked out since they are 0 anyway. Values prescribed
  in the input file thus have a different meaning. They act directly upon the enriched dofs!

- For two-phase flow and combustion problems, however, standard dofs and enriched dofs exist at the
  same time for an enriched node. The number of dofs per node exceeds 6 and there is no way to get
  around this!

- The current hack does not circumvent the problem in a general way - let alone in an elegant way!
  It works for no penetration BCs at e.g. channel walls. The same Dirichlet value is prescribed for
  the enriched dof as for the standard dof (that is 0.0!). That means that there is no velocity and
  no discontinuity at a channel wall. This does not work for e.g. inflow profiles, where velocity
  values != 0.0 are prescribed. Enforcing the same values on enriched dofs would introduce
  discontinuities in the inflow profile!

- The hack consists in modifying the access to vectors "val", "funct", "onoff" and "curve" in the
  loop over all dofs j. Instead of accessing the corresponding vector component for every dof
  (e.g.(*curve)[j]) (causes a crash!), the component belonging to the standard dof is accessed:

  -> replace "j" by "truncate(j/2)" wherever a vector component is accessed (e.g. (*curve)[j])

     the "truncate" operation is performed by a (dirty) cast operation from double to int:

     int truncj = j/2; // the closest smaller integer number is taken

  example: 3 velocity components enriched, pressure not enriched -> 7 dofs per node (counter j)
  for one node i:

  counter j    meaning of dof         e.g. (*curve)[truncj]
  dof 0        velx standard    ->    vector component 0
  dof 1        velx enriched    ->    vector component 0
  dof 2        vely standard    ->    vector component 1
  dof 3        vely enriched    ->    vector component 1
  dof 4        velz standard    ->    vector component 2
  dof 5        velz enriched    ->    vector component 2
  dof 6        pres standard    ->    vector component 3
  dof 7        pres enriched    ->    vector component 3

\author henke 07/09
*/
static void DoDirichletCondition(DRT::Condition&             cond,
                                 DRT::DiscretizationXFEM&        dis,
                                 const bool                  usetime,
                                 const double                time,
                                 Teuchos::RCP<Epetra_Vector> systemvector,
                                 Teuchos::RCP<Epetra_Vector> systemvectord,
                                 Teuchos::RCP<Epetra_Vector> systemvectordd,
                                 Teuchos::RCP<Epetra_Vector> toggle,
                                 Teuchos::RCP<std::set<int> > dbcgids
                                 );


/*----------------------------------------------------------------------*
 | set Dirichlet conditions for XFEM problems               henke 07/09 |
 | remark: read documentation at top of this file!                      |
 *----------------------------------------------------------------------*/
void DoDirichletCondition(DRT::Condition&             cond,
                          DRT::DiscretizationXFEM&        dis,
                          const bool                  usetime,
                          const double                time,
                          Teuchos::RCP<Epetra_Vector> systemvector,
                          Teuchos::RCP<Epetra_Vector> systemvectord,
                          Teuchos::RCP<Epetra_Vector> systemvectordd,
                          Teuchos::RCP<Epetra_Vector> toggle,
                          Teuchos::RCP<std::set<int> > dbcgids)
{
  const vector<int>* nodeids = cond.Nodes();
  if (!nodeids) dserror("Dirichlet condition does not have nodal cloud");
  const int nnode = (*nodeids).size();
  const vector<int>*    curve  = cond.Get<vector<int> >("curve");
  const vector<int>*    funct  = cond.Get<vector<int> >("funct");
  const vector<int>*    onoff  = cond.Get<vector<int> >("onoff");
  const vector<double>* val    = cond.Get<vector<double> >("val");

  // determine highest degree of time derivative
  // and first existent system vector to apply DBC to
  unsigned deg = 0;  // highest degree of requested time derivative
  Teuchos::RCP<Epetra_Vector> systemvectoraux = Teuchos::null;  // auxiliar system vector
  if (systemvector != Teuchos::null)
  {
    deg = 0;
    systemvectoraux = systemvector;
  }
  if (systemvectord != Teuchos::null)
  {
    deg = 1;
    if (systemvectoraux == Teuchos::null)
      systemvectoraux = systemvectord;
  }
  if (systemvectordd != Teuchos::null)
  {
    deg = 2;
    if (systemvectoraux == Teuchos::null)
      systemvectoraux = systemvectordd;
  }
  dsassert(systemvectoraux!=Teuchos::null, "At least one vector must be unequal to null");

  // loop nodes to identify and evaluate load curves and spatial distributions
  // of Dirichlet boundary conditions
  for (int i=0; i<nnode; ++i)
  {
    // do only nodes in my row map
    if (!dis.NodeRowMap()->MyGID((*nodeids)[i])) continue;
    DRT::Node* actnode = dis.gNode((*nodeids)[i]);
    if (!actnode) dserror("Cannot find global node %d",(*nodeids)[i]);
    vector<int> dofs = dis.Dof(actnode);
    const unsigned numdf = dofs.size();

    if (numdf==4)//StandardFEM
    {
       for (unsigned j=0; j<numdf; ++j)
       {
         if ((*onoff)[j]==0)// if Dirichlet value is turned off in input file (0)
         {
           const int lid = (*systemvectoraux).Map().LID(dofs[j]);
           if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
           if (toggle!=Teuchos::null)
              (*toggle)[lid] = 0.0;
           // get rid of entry in DBC map - if it exists
           if (dbcgids != Teuchos::null)
            (*dbcgids).erase(dofs[j]);
           continue; // for loop over dofs is advanced by 1 (++j)
         }
         const int gid = dofs[j];
         vector<double> value(deg+1,(*val)[j]);

         // factor given by time curve
         std::vector<double> curvefac(deg+1, 1.0);
         int curvenum = -1;
         if (curve) curvenum = (*curve)[j];
           if (curvenum>=0 && usetime)
             curvefac = DRT::Problem::Instance()->Curve(curvenum).FctDer(time,deg);
         else
           for (unsigned i=1; i<(deg+1); ++i) curvefac[i] = 0.0;

         // factor given by spatial function
         double functfac = 1.0;
         int funct_num = -1;
         if (funct) funct_num = (*funct)[j];
         {
           if (funct_num>0)
             functfac = DRT::Problem::Instance()->Funct(funct_num-1).Evaluate(j,
                                                                     actnode->X(),
                                                                     time,
                                                                     &dis);
         }

        // apply factors to Dirichlet value
        for (unsigned i=0; i<deg+1; ++i)
        {
          value[i] *= functfac * curvefac[i];
        }

        // assign value
        const int lid = (*systemvectoraux).Map().LID(gid);
        if (lid<0) dserror("Global id %d not on this proc in system vector",gid);
        if (systemvector != Teuchos::null)
           (*systemvector)[lid] = value[0];
        if (systemvectord != Teuchos::null)
           (*systemvectord)[lid] = value[1];
        if (systemvectordd != Teuchos::null)
           (*systemvectordd)[lid] = value[2];
        // set toggle vector
        if (toggle != Teuchos::null)
           (*toggle)[lid] = 1.0;
        // amend vector of DOF-IDs which are Dirichlet BCs
        if (dbcgids != Teuchos::null)
           (*dbcgids).insert(gid);
      }  // loop over nodal DOFs
    }
    else if(numdf==8) //ExtendedFEM
    {
      for (unsigned j=0; j<numdf; ++j) // loop over all dofs (Std + Enr)
      {
        // this is a hack to truncate a double!
        int truncj = j/2; // the closest smaller integer number is taken

        if ((*onoff)[truncj]==0) // if Dirichlet value is turned off in input file (0)
        {
          const int lid = (*systemvectoraux).Map().LID(dofs[j]);
          if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
          if (toggle!=Teuchos::null)
            (*toggle)[lid] = 0.0;
          // get rid of entry in DBC map - if it exists
          if (dbcgids != Teuchos::null)
            (*dbcgids).erase(dofs[j]);
          continue; // for loop over dofs is advanced by 1 (++j)
        }
        const int gid = dofs[j];
        vector<double> value(deg+1,(*val)[truncj]);

        // factor given by time curve
        std::vector<double> curvefac(deg+1, 1.0);
        int curvenum = -1;

        if (curve) curvenum = (*curve)[truncj];
        if (curvenum>=0 && usetime)
          curvefac = DRT::Problem::Instance()->Curve(curvenum).FctDer(time,deg);
        else
          for (unsigned i=1; i<(deg+1); ++i) curvefac[i] = 0.0;

        // factor given by spatial function
        double functfac = 1.0;
        int funct_num = -1;
        if (funct) funct_num = (*funct)[truncj];
        {
           if (funct_num>0)
             functfac = DRT::Problem::Instance()->Funct(funct_num-1).Evaluate(j,
                                                                     actnode->X(),
                                                                     time,
                                                                     &dis);
        }

        // apply factors to Dirichlet value
        for (unsigned i=0; i<deg+1; ++i)
        {
          value[i] *= functfac * curvefac[i];
        }

        // overwrite Dirichlet values for XFEM dofs
        dsassert((*onoff)[truncj]!=0,"there should be a Dirichlet condition assigned to this dof!");
        if (j%2!=0) // if XFEM dof
        {
          //cout << "/!\\ warning === Dirichlet value of enriched dof " << j << " is set to 0.0 for node " << actnode->Id() << endl;
          for (unsigned i=0; i<deg+1; ++i)
          {
            value[i] = 0.0; // previously assigned DBC value is overwritten by 0.0
          }
        }

#ifdef COMBUST_TESTCOUETTEFLOW
        //--------------------------------------------
        // ugly hack for decouled Couette flow example
        //--------------------------------------------
        if ((*onoff)[truncj]!=0 and j==1) // if Dirichlet value is turned on in input file (1)
        {
//          // domain with 10 elements in x-direction
//          if (actnode->Id() == 26 or
//              actnode->Id() == 27 or
//              actnode->Id() == 40 or
//              actnode->Id() == 41)
          // domain with 20 elements in x-direction
          if (actnode->Id() == 96 or
              actnode->Id() == 97 or
              actnode->Id() == 110 or
              actnode->Id() == 111)
          {
            cout << "/!\\ warning === Dirichlet value of enriched x-velocity dof is modified manually for node " << actnode->Id() << endl;
            cout << *actnode << endl;
            for (unsigned i=0; i<deg+1; ++i)
            {
#ifdef COMBUST_TESTCOUETTEFLOWDECOUPLED
              value[i] = -6.5;
#else
              value[i] = 1.0;
#endif
            }
          }
//          // domain with 10 elements in x-direction
//          if (actnode->Id() == 98 or
//              actnode->Id() == 99 or
//              actnode->Id() == 112 or
//              actnode->Id() == 113)
          // domain with 20 elements in x-direction
          else if (actnode->Id() == 168 or
                   actnode->Id() == 169 or
                   actnode->Id() == 182 or
                   actnode->Id() == 183)
          {
            cout << "/!\\ warning === Dirichlet value of enriched x-velocity dof is modified manually for node " << actnode->Id() << endl;
            cout << *actnode << endl;
            for (unsigned i=0; i<deg+1; ++i)
            {
#ifdef COMBUST_TESTCOUETTEFLOWDECOUPLED
              value[i] = -2.5;
#else
              value[i] = 1.0;
#endif
            }
          }
          else // free x-velocity dof for all other Dirichlet nodes
          {
            cout << "/!\\ warning === no Dirichlet value set for enriched x-velocity dof of node " << actnode->Id() << endl;
            const int lid = (*systemvectoraux).Map().LID(dofs[j]);
            if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
            if (toggle!=Teuchos::null)
              (*toggle)[lid] = 0.0;
            // get rid of entry in DBC map - if it exists
            if (dbcgids != Teuchos::null)
              (*dbcgids).erase(dofs[j]);
            continue; // for loop over dofs is advanced by 1 (++j)
          }
        }
#endif

#ifdef COMBUST_XVELFREE
        if ((*onoff)[truncj]!=0 and j==1) // if Dirichlet value is turned on in input file (1)
        {
          cout << "/!\\ warning === no Dirichlet value set for enriched x-velocity dof of node " << actnode->Id() << endl;
          const int lid = (*systemvectoraux).Map().LID(dofs[j]);
          if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
          if (toggle!=Teuchos::null)
            (*toggle)[lid] = 0.0;
          // get rid of entry in DBC map - if it exists
          if (dbcgids != Teuchos::null)
            (*dbcgids).erase(dofs[j]);
          continue; // for loop over dofs is advanced by 1 (++j)
        }
#endif
#ifdef COMBUST_YVELFREE
        if ((*onoff)[truncj]!=0 and j==3) // if Dirichlet value is turned on in input file (1)
        {
          cout << "/!\\ warning === no Dirichlet value set for enriched y-velocity dof of node " << actnode->Id() << endl;
          const int lid = (*systemvectoraux).Map().LID(dofs[j]);
          if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
          if (toggle!=Teuchos::null)
            (*toggle)[lid] = 0.0;
          // get rid of entry in DBC map - if it exists
          if (dbcgids != Teuchos::null)
            (*dbcgids).erase(dofs[j]);
          continue; // for loop over dofs is advanced by 1 (++j)
        }
#endif
#ifdef COMBUST_PRESFREE
        if ((*onoff)[truncj]!=0 and j==7) // if Dirichlet value is turned on in input file (1)
        {
          cout << "/!\\ warning === no Dirichlet value set for enriched pressure dof of node " << actnode->Id() << endl;
          const int lid = (*systemvectoraux).Map().LID(dofs[j]);
          if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
          if (toggle!=Teuchos::null)
            (*toggle)[lid] = 0.0;
          // get rid of entry in DBC map - if it exists
          if (dbcgids != Teuchos::null)
            (*dbcgids).erase(dofs[j]);
          continue; // for loop over dofs is advanced by 1 (++j)
        }
#endif
        // assign value
        const int lid = (*systemvectoraux).Map().LID(gid);
        if (lid<0) dserror("Global id %d not on this proc in system vector",gid);
        if (systemvector != Teuchos::null)
          (*systemvector)[lid] = value[0];
        if (systemvectord != Teuchos::null)
          (*systemvectord)[lid] = value[1];
        if (systemvectordd != Teuchos::null)
          (*systemvectordd)[lid] = value[2];
        // set toggle vector
        if (toggle != Teuchos::null)
          (*toggle)[lid] = 1.0;
        // amend vector of DOF-IDs which are Dirichlet BCs
        if (dbcgids != Teuchos::null)
          (*dbcgids).insert(gid);
      }  // loop over nodal DOFs
    }
    else if(numdf==6)
    {
      for (unsigned j=0; j<numdf; ++j)
      {
        int xfemj = j;
        if(j==0) xfemj = 0;
        if(j==1) xfemj = 1;
        if(j==2) xfemj = 2;
        if(j==3) xfemj = 2;
        if(j==4) xfemj = 3;
        if(j==5) xfemj = 3;
        //-------------------------
        // ich bin ein Standard dof
        //-------------------------
        if(j!=3 && j!=5)
        {
          //---------------------------
          // ich bin kein Dirichlet dof
          //---------------------------
          if ((*onoff)[xfemj]==0)// if Dirichlet value is turned off in input file (0)
          {
            const int lid = (*systemvectoraux).Map().LID(dofs[j]);
            if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
            if (toggle!=Teuchos::null)
              (*toggle)[lid] = 0.0;
            // get rid of entry in DBC map - if it exists
            if (dbcgids != Teuchos::null)
              (*dbcgids).erase(dofs[j]);
            continue; // for loop over dofs is advanced by 1 (++j)
          }
          //--------------------------
          // ich bin ein Dirichlet dof
          //--------------------------
          const int gid = dofs[j];
          vector<double> value(deg+1,(*val)[xfemj]);

          // factor given by time curve
          std::vector<double> curvefac(deg+1, 1.0);
          int curvenum = -1;
          if (curve) curvenum = (*curve)[xfemj];
          if (curvenum>=0 && usetime)
            curvefac = DRT::Problem::Instance()->Curve(curvenum).FctDer(time,deg);
          else
            for (unsigned i=1; i<(deg+1); ++i) curvefac[i] = 0.0;

          // factor given by spatial function
          double functfac = 1.0;
          int funct_num = -1;
          if (funct) funct_num = (*funct)[xfemj];
          {
            if (funct_num>0)
              functfac = DRT::Problem::Instance()->Funct(funct_num-1).Evaluate(j,
                  actnode->X(),
                  time,
                  &dis);
          }
          // apply factors to Dirichlet value
          for (unsigned i=0; i<deg+1; ++i)
          {
            value[i] *= functfac * curvefac[i];
          }
          // assign value
          const int lid = (*systemvectoraux).Map().LID(gid);
          if (lid<0) dserror("Global id %d not on this proc in system vector",gid);
          if (systemvector != Teuchos::null)
            (*systemvector)[lid] = value[0];
          if (systemvectord != Teuchos::null)
            (*systemvectord)[lid] = value[1];
          if (systemvectordd != Teuchos::null)
            (*systemvectordd)[lid] = value[2];
          // set toggle vector
          if (toggle != Teuchos::null)
            (*toggle)[lid] = 1.0;
          // amend vector of DOF-IDs which are Dirichlet BCs
          if (dbcgids != Teuchos::null)
            (*dbcgids).insert(gid);
        }
        //-----------------------------------
        // ich bin der angereicherte veln dof
        //-----------------------------------
        else if(j==3 && (*onoff)[0]!=0 && (*onoff)[1]!=0 && (*onoff)[2]!=0 )
        {
          //--------------------------
          // ich bin ein Dirichlet dof
          //--------------------------
          const int gid = dofs[j];
          vector<double> value(deg+1,(*val)[xfemj]);

          //cout << "/!\\ warning === Dirichlet value of enriched dof " << j << " (veln) is set to 0.0 for node " << actnode->Id() << endl;
          // apply factors to Dirichlet value
          for (unsigned i=0; i<deg+1; ++i)
          {
            value[i] = 0.0;
          }
#ifdef COMBUST_TESTCOUETTEFLOW
          //--------------------------------------------
          // ugly hack for decoupled Couette flow example
          //--------------------------------------------

          // domain with 10 elements in x-direction
          //if (actnode->Id() == 26 or
          //    actnode->Id() == 27 or
          //    actnode->Id() == 40 or
          //    actnode->Id() == 41)
          // domain with 20 elements in x-direction
          if (actnode->Id() == 96 or
              actnode->Id() == 97 or
              actnode->Id() == 110 or
              actnode->Id() == 111)
          {
            cout << "/!\\ warning === Dirichlet value of enriched x-velocity dof is modified manually for node " << actnode->Id() << endl;
            cout << *actnode << endl;
            for (unsigned i=0; i<deg+1; ++i)
            {
#ifdef COMBUST_TESTCOUETTEFLOWDECOUPLED
              value[i] = 6.5*sqrt(2.0);
#else
              value[i] = -1.0*sqrt(2.0);
#endif
            }
          }
          // domain with 10 elements in x-direction
          //if (actnode->Id() == 98 or
          //    actnode->Id() == 99 or
          //    actnode->Id() == 112 or
          //    actnode->Id() == 113)
          // domain with 20 elements in x-direction
          else if (actnode->Id() == 168 or
              actnode->Id() == 169 or
              actnode->Id() == 182 or
              actnode->Id() == 183)
          {
            cout << "/!\\ warning === Dirichlet value of enriched x-velocity dof is modified manually for node " << actnode->Id() << endl;
            cout << *actnode << endl;
            for (unsigned i=0; i<deg+1; ++i)
            {
#ifdef COMBUST_TESTCOUETTEFLOWDECOUPLED
              value[i] = 2.5*sqrt(2.0);
#else
              value[i] = -1.0*sqrt(2.0);
#endif
            }
          }
          else // free x-velocity dof for all other Dirichlet nodes
          {
            cout << "/!\\ warning === no Dirichlet value set for enriched x-velocity dof of node " << actnode->Id() << endl;
            const int lid = (*systemvectoraux).Map().LID(dofs[j]);
            if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
            if (toggle!=Teuchos::null)
              (*toggle)[lid] = 0.0;
            // get rid of entry in DBC map - if it exists
            if (dbcgids != Teuchos::null)
              (*dbcgids).erase(dofs[j]);
            continue; // for loop over dofs is advanced by 1 (++j)
          }
#endif
            // assign value
            const int lid = (*systemvectoraux).Map().LID(gid);
            if (lid<0) dserror("Global id %d not on this proc in system vector",gid);
            if (systemvector != Teuchos::null)
              (*systemvector)[lid] = value[0];
            if (systemvectord != Teuchos::null)
              (*systemvectord)[lid] = value[1];
            if (systemvectordd != Teuchos::null)
              (*systemvectordd)[lid] = value[2];
            // set toggle vector
            if (toggle != Teuchos::null)
              (*toggle)[lid] = 1.0;
            // amend vector of DOF-IDs which are Dirichlet BCs
            if (dbcgids != Teuchos::null)
              (*dbcgids).insert(gid);
        }
        //-----------------------------------
        // ich bin der angereicherte pres dof
        //-----------------------------------
        else if( j==5 && (*onoff)[4]!=0 )
        {
            //--------------------------
            // ich bin ein Dirichlet dof
            //--------------------------
            const int gid = dofs[j];
            vector<double> value(deg+1,(*val)[xfemj]);

            //cout << "/!\\ warning === Dirichlet value of enriched dof " << j << " (pres) is set to 0.0 for node " << actnode->Id() << endl;
            // apply factors to Dirichlet value
            for (unsigned i=0; i<deg+1; ++i)
            {
              value[i] = 0.0;
            }
            // assign value
            const int lid = (*systemvectoraux).Map().LID(gid);
            if (lid<0) dserror("Global id %d not on this proc in system vector",gid);
            if (systemvector != Teuchos::null)
              (*systemvector)[lid] = value[0];
            if (systemvectord != Teuchos::null)
              (*systemvectord)[lid] = value[1];
            if (systemvectordd != Teuchos::null)
              (*systemvectordd)[lid] = value[2];
            // set toggle vector
            if (toggle != Teuchos::null)
              (*toggle)[lid] = 1.0;
            // amend vector of DOF-IDs which are Dirichlet BCs
            if (dbcgids != Teuchos::null)
              (*dbcgids).insert(gid);
        }
        else
        {
          //cout << "/!\\ warning === no Dirichlet value set for dof " << j << " of node " << actnode->Id() << endl;
          const int lid = (*systemvectoraux).Map().LID(dofs[j]);
          if (lid<0) dserror("Global id %d not on this proc in system vector",dofs[j]);
          if (toggle!=Teuchos::null)
            (*toggle)[lid] = 0.0;
          // get rid of entry in DBC map - if it exists
          if (dbcgids != Teuchos::null)
            (*dbcgids).erase(dofs[j]);
          continue;
        }
      }  // loop over nodal DOFs
    }
    else if (numdf==0) //relevant for xfsi and fxf-Coupling (void enrichment)
    {

    }
    else
    {
      dserror("So viele dofs gibts doch gar nicht an einem Knoten!");
    }
  }  // loop over nodes
  return;
}


/*----------------------------------------------------------------------*
 | evaluate Dirichlet conditions (public)                   henke 07/09 |
 *----------------------------------------------------------------------*/
void DRT::DiscretizationXFEM::EvaluateDirichlet(ParameterList& params,
                                            Teuchos::RCP<Epetra_Vector> systemvector,
                                            Teuchos::RCP<Epetra_Vector> systemvectord,
                                            Teuchos::RCP<Epetra_Vector> systemvectordd,
                                            Teuchos::RCP<Epetra_Vector> toggle,
                                            Teuchos::RCP<LINALG::MapExtractor> dbcmapextractor)
{
  if (!Filled()) dserror("FillComplete() was not called");
  if (!HaveDofs()) dserror("AssignDegreesOfFreedom() was not called");

  // get the current time
  bool usetime = true;
  const double time = params.get("total time",-1.0);
  if (time<0.0) usetime = false;

  // vector of DOF-IDs which are Dirichlet BCs
  Teuchos::RCP<std::set<int> > dbcgids = Teuchos::null;
  if (dbcmapextractor != Teuchos::null) dbcgids = Teuchos::rcp(new std::set<int>());

  multimap<string,RCP<Condition> >::iterator fool;
  //--------------------------------------------------------
  // loop through Dirichlet conditions and evaluate them
  //--------------------------------------------------------
  // Note that this method does not sum up but 'sets' values in systemvector.
  // For this reason, Dirichlet BCs are evaluated hierarchical meaning
  // in this order:
  //                VolumeDirichlet
  //                SurfaceDirichlet
  //                LineDirichlet
  //                PointDirichlet
  // This way, lower entities override higher ones which is
  // equivalent to inheritance of dirichlet BCs as done in the old
  // ccarat discretization with design          (mgee 1/07)

  // Do VolumeDirichlet first
  for (fool=condition_.begin(); fool!=condition_.end(); ++fool)
  {
    if (fool->first != "Dirichlet") continue;
    if (fool->second->Type() != DRT::Condition::VolumeDirichlet) continue;
    DoDirichletCondition(*(fool->second),*this,usetime,time,
                         systemvector,systemvectord,systemvectordd,
                         toggle,dbcgids);
  }
  // Do SurfaceDirichlet
  for (fool=condition_.begin(); fool!=condition_.end(); ++fool)
  {
    if (fool->first != "Dirichlet") continue;
    if (fool->second->Type() != DRT::Condition::SurfaceDirichlet) continue;
    DoDirichletCondition(*(fool->second),*this,usetime,time,
                         systemvector,systemvectord,systemvectordd,
                         toggle,dbcgids);
  }
  // Do LineDirichlet
  for (fool=condition_.begin(); fool!=condition_.end(); ++fool)
  {
    if (fool->first != "Dirichlet") continue;
    if (fool->second->Type() != DRT::Condition::LineDirichlet) continue;
    DoDirichletCondition(*(fool->second),*this,usetime,time,
                         systemvector,systemvectord,systemvectordd,
                         toggle,dbcgids);
  }
  // Do PointDirichlet
  for (fool=condition_.begin(); fool!=condition_.end(); ++fool)
  {
    if (fool->first != "Dirichlet") continue;
    if (fool->second->Type() != DRT::Condition::PointDirichlet) continue;
    DoDirichletCondition(*(fool->second),*this,usetime,time,
                         systemvector,systemvectord,systemvectordd,
                         toggle,dbcgids);
  }

  // create DBC and free map and build their common extractor
  if (dbcmapextractor != Teuchos::null)
  {
    // build map of Dirichlet DOFs
    int nummyelements = 0;
    int* myglobalelements = NULL;
    std::vector<int> dbcgidsv;
    if (dbcgids->size() > 0)
    {
      dbcgidsv.reserve(dbcgids->size());
      dbcgidsv.assign(dbcgids->begin(),dbcgids->end());
      nummyelements = dbcgidsv.size();
      myglobalelements = &(dbcgidsv[0]);
    }
    Teuchos::RCP<Epetra_Map> dbcmap
      = Teuchos::rcp(new Epetra_Map(-1, nummyelements, myglobalelements, DofRowMap()->IndexBase(), DofRowMap()->Comm()));
    // build the map extractor of Dirichlet-conditioned and free DOFs
    *dbcmapextractor = LINALG::MapExtractor(*(DofRowMap()), dbcmap);
  }

  return;
}
