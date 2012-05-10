/*!------------------------------------------------------------------------------------------------*
\file topopt_fluidAdjointResulttest.cpp

\brief 

<pre>
Maintainer: Martin Winklmaier
            winklmaier@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15241
</pre>
 *------------------------------------------------------------------------------------------------*/


#include <string>

#include "topopt_fluidAdjointResulttest.H"
#include "topopt_fluidAdjointImplTimeIntegration.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_linedefinition.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
TOPOPT::ADJOINT::FluidAdjointResultTest::FluidAdjointResultTest(
    const ImplicitTimeInt& adjointfluid
)
: fluiddis_(adjointfluid.Discretization()),
  mysol_(adjointfluid.Velnp())
{
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void TOPOPT::ADJOINT::FluidAdjointResultTest::TestNode(
    DRT::INPUT::LineDefinition& res,
    int& nerr,
    int& test_count
)
{
  int dis;
  res.ExtractInt("DIS",dis);
  if (dis != 1)
    dserror("fix me: only one fluid discretization supported for testing");

  int nodeGid;
  res.ExtractInt("NODE",nodeGid);
  nodeGid -= 1;

  if (fluiddis_->HaveGlobalNode(nodeGid))
  {
    const DRT::Node* node = fluiddis_->gNode(nodeGid);

    // Test only, if actnode is a row node
    if (node->Owner() != fluiddis_->Comm().MyPID())
      return;

    double result = 0.;

    const Epetra_BlockMap& velnpmap = mysol_->Map();

    const int numdim = DRT::Problem::Instance()->NDim();

    std::string position;
    res.ExtractString("POSITION",position);
    if (position=="velx")
    {
      result = (*mysol_)[velnpmap.LID(fluiddis_->Dof(0,node,0))];
    }
    else if (position=="vely")
    {
      result = (*mysol_)[velnpmap.LID(fluiddis_->Dof(0,node,1))];
    }
    else if (position=="velz")
    {
      if (numdim==2)
        dserror("Cannot test result for velz in 2D case.");
      result = (*mysol_)[velnpmap.LID(fluiddis_->Dof(0,node,2))];
    }
    else if (position=="pressure")
    {
      if (numdim==2)
      {
        if (fluiddis_->NumDof(0,node)<3)
          dserror("too few dofs at node %d for pressure testing",node->Id());
        result = (*mysol_)[velnpmap.LID(fluiddis_->Dof(0,node,2))];
      }
      else
      {
        if (fluiddis_->NumDof(0,node)<4)
          dserror("too few dofs at node %d for pressure testing",node->Id());
        result = (*mysol_)[velnpmap.LID(fluiddis_->Dof(0,node,3))];
      }
    }
    else
    {
      dserror("position '%s' not supported in fluid testing", position.c_str());
    }

    nerr += CompareValues(result, res);
    test_count++;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool TOPOPT::ADJOINT::FluidAdjointResultTest::Match(DRT::INPUT::LineDefinition& res)
{
  return res.HaveNamed("ADJOINT");
}

