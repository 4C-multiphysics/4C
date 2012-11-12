/*----------------------------------------------------------------------*/
/*!
\file artery_resulttest.cpp

\brief testing of artnet calculation results

<pre>
Maintainer: Mahmoud Ismail
            ismail@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15268
</pre>
*/
/*----------------------------------------------------------------------*/

#include "artnetexplicitintegration.H"
#include "artery_resulttest.H"
#include "../drt_lib/drt_linedefinition.H"
#include "../drt_lib/drt_discret.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ART::ArteryResultTest::ArteryResultTest(ArtNetExplicitTimeInt& art_net)
{
  dis_    = art_net.Discretization();
  mysol_  = art_net.QAnp();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ART::ArteryResultTest::TestNode(DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis;
  res.ExtractString("DIS",dis);
  if (dis != dis_->Name())
    return;

  int node;
  res.ExtractInt("NODE",node);
  node -= 1;

  int havenode(dis_->HaveGlobalNode(node));
  int isnodeofanybody(0);
  dis_->Comm().SumAll(&havenode,&isnodeofanybody,1);

  if (isnodeofanybody==0)
  {
    dserror("Node %d does not belong to discretization %s",node+1,dis_->Name().c_str());
  }
  else
  {
    if (dis_->HaveGlobalNode(node))
    {
      DRT::Node* actnode = dis_->gNode(node);

      // Strange! It seems we might actually have a global node around
      // even if it does not belong to us. But here we are just
      // interested in our nodes!
      if (actnode->Owner() != dis_->Comm().MyPID())
        return;

      double result = 0.;
      const Epetra_BlockMap& pnpmap = mysol_->Map();
      std::string position;
      res.ExtractString("QUANTITY",position);

      // test result value of single scalar field
      if (position=="area")
        result = (*mysol_)[pnpmap.LID(dis_->Dof(actnode,0))];
      else if (position=="flowrate")
        result = (*mysol_)[pnpmap.LID(dis_->Dof(actnode,1))];
      // test result values for a system of scalars
      else
      {
        dserror("Quantity '%s' not supported in result-test of red_airway transport problems", position.c_str());
      }

      nerr += CompareValues(result, res);
      test_count++;
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool ART::ArteryResultTest::Match(DRT::INPUT::LineDefinition& res)
{
  return res.HaveNamed("ARTNET");
}


