/*----------------------------------------------------------------------*/
/*! \file

\brief testing of electromagnetic calculation results

\level 2

*/
/*----------------------------------------------------------------------*/


#include "4C_elemag_resulttest.hpp"

#include "4C_elemag_timeint.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_lib_discret.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                     berardocco 10/18 |
 *----------------------------------------------------------------------*/
ELEMAG::ElemagResultTest::ElemagResultTest(ElemagTimeInt& elemagalgo)
    : DRT::ResultTest("ELECTROMAGNETIC")
{
  dis_ = elemagalgo.Discretization();
  // mysol_ = CORE::LINALG::CreateVector(*(dis_->NodeRowMap()), true);
  error_ = Teuchos::rcp(new CORE::LINALG::SerialDenseVector(4));
  error_ = elemagalgo.ComputeError();
  // elemagalgo.NodalPressureField(mysol_);
}

/*----------------------------------------------------------------------*
 |                                                     berardocco 10/18 |
 *----------------------------------------------------------------------*/
void ELEMAG::ElemagResultTest::TestNode(INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis;
  res.ExtractString("DIS", dis);
  if (dis != dis_->Name()) return;

  int node;
  res.ExtractInt("NODE", node);
  node -= 1;

  int havenode(dis_->HaveGlobalNode(node));
  int isnodeofanybody(0);
  dis_->Comm().SumAll(&havenode, &isnodeofanybody, 1);

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW("Node %d does not belong to discretization %s", node + 1, dis_->Name().c_str());
  }
  else
  {
    if (dis_->HaveGlobalNode(node))
    {
      DRT::Node* actnode = dis_->gNode(node);

      // Here, we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->Owner() != dis_->Comm().MyPID()) return;

      double result = 0.;
      // const Epetra_BlockMap& map = mysol_->Map();
      std::string variable;
      res.ExtractString("QUANTITY", variable);

      if (variable == "electric")
      {
        std::cout << "This error function has not yet been implemented." << std::endl;
        // result = (*mysol_)[map.LID(actnode->Id())];  // Here, we got a node map, not a dof map
        // with dofs based on nodes!
      }
      else if (variable == "L2electric")
      {
        result = std::sqrt((*error_)[0]);
      }
      else if (variable == "L2magnetic")
      {
        result = std::sqrt((*error_)[2]);
      }
      else if (variable == "L2electric-rel")
      {
        if ((*error_)[1] > 0.0)
          result = std::sqrt((*error_)[0] / (*error_)[1]);
        else
          FOUR_C_THROW(
              "Impossible to compute the electric relative error. The L2-norm of the "
              "analytical solution is zero, resulting in a division by zero.");
      }
      else if (variable == "L2magnetic-rel")
      {
        if ((*error_)[3] > 0.0)
          result = std::sqrt((*error_)[2] / (*error_)[3]);
        else
          FOUR_C_THROW(
              "Impossible to compute the magnetic relative error. The L2-norm of the "
              "analytical solution is zero, resulting in a division by zero.");
      }
      else if (variable == "Hdiv-electric")
      {
        result = std::sqrt((*error_)[4]);
      }
      else if (variable == "Hdiv-magnetic")
      {
        result = std::sqrt((*error_)[5]);
      }
      else if (variable == "Hcurl-electric")
      {
        result = std::sqrt((*error_)[6]);
      }
      else if (variable == "Hcurl-magnetic")
      {
        result = std::sqrt((*error_)[7]);
      }
      else if (variable == "L2electric-post")
      {
        result = std::sqrt((*error_)[8]);
      }
      else if (variable == "L2electric-rel-post")
      {
        if ((*error_)[1] > 0.0)
          result = std::sqrt((*error_)[8] / (*error_)[1]);
        else
          FOUR_C_THROW(
              "Impossible to compute the post-processed electric relative error. The L2-norm of "
              "the analytical solution is zero, resulting in a division by zero.");
      }
      else if (variable == "Hdiv-electric-post")
      {
        result = std::sqrt((*error_)[9]);
      }
      else if (variable == "Hcurl-electric-post")
      {
        result = std::sqrt((*error_)[10]);
      }
      else
      {
        FOUR_C_THROW("Quantity '%s' not supported in result-test of elemagstic transport problems",
            variable.c_str());
      }

      nerr += CompareValues(result, "NODE", res);
      test_count++;
    }
  }
  return;
}
FOUR_C_NAMESPACE_CLOSE
