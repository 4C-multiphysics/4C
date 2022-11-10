/*----------------------------------------------------------------------*/
/*! \file
 \brief result test for multiphase porous flow

   \level 3

 *----------------------------------------------------------------------*/


#include "porofluidmultiphase_resulttest.H"

#include "drt_linedefinition.H"
#include "drt_discret.H"
#include "porofluidmultiphase_timint_implicit.H"
#include "porofluidmultiphase_meshtying_strategy_base.H"


/*----------------------------------------------------------------------*
 | ctor                                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
POROFLUIDMULTIPHASE::ResultTest::ResultTest(TimIntImpl& porotimint)
    : DRT::ResultTest("POROFLUIDMULTIPHASE"), porotimint_(porotimint)
{
  return;
}


/*----------------------------------------------------------------------*
 | test node                                                vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::ResultTest::TestNode(
    DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis;
  res.ExtractString("DIS", dis);
  if (dis != porotimint_.Discretization()->Name()) return;

  int node;
  res.ExtractInt("NODE", node);
  node -= 1;

  int havenode(porotimint_.Discretization()->HaveGlobalNode(node));
  int isnodeofanybody(0);
  porotimint_.Discretization()->Comm().SumAll(&havenode, &isnodeofanybody, 1);

  if (isnodeofanybody == 0)
  {
    dserror("Node %d does not belong to discretization %s", node + 1,
        porotimint_.Discretization()->Name().c_str());
  }
  else
  {
    if (porotimint_.Discretization()->HaveGlobalNode(node))
    {
      DRT::Node* actnode = porotimint_.Discretization()->gNode(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->Owner() != porotimint_.Discretization()->Comm().MyPID()) return;

      // extract name of quantity to be tested
      std::string quantity;
      res.ExtractString("QUANTITY", quantity);

      // get result to be tested
      const double result = ResultNode(quantity, actnode);

      nerr += CompareValues(result, "NODE", res);
      test_count++;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | test element                                        kremheller 10/19 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::ResultTest::TestElement(
    DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis;
  res.ExtractString("DIS", dis);

  if (dis != porotimint_.Discretization()->Name()) return;

  int element;
  res.ExtractInt("ELEMENT", element);
  element -= 1;

  int haveelement(porotimint_.Discretization()->HaveGlobalElement(element));
  int iselementofanybody(0);
  porotimint_.Discretization()->Comm().SumAll(&haveelement, &iselementofanybody, 1);

  if (iselementofanybody == 0)
  {
    dserror("Element %d does not belong to discretization %s", element + 1,
        porotimint_.Discretization()->Name().c_str());
  }
  else
  {
    if (porotimint_.Discretization()->HaveGlobalElement(element))
    {
      const DRT::Element* actelement = porotimint_.Discretization()->gElement(element);

      // Here we are just interested in the elements that we own (i.e. a row element)!
      if (actelement->Owner() != porotimint_.Discretization()->Comm().MyPID()) return;

      // extract name of quantity to be tested
      std::string quantity;
      res.ExtractString("QUANTITY", quantity);

      // get result to be tested
      const double result = ResultElement(quantity, actelement);

      nerr += CompareValues(result, "ELEMENT", res);
      test_count++;
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | get nodal result to be tested                            vuong 08/16 |
 *----------------------------------------------------------------------*/
double POROFLUIDMULTIPHASE::ResultTest::ResultNode(
    const std::string quantity, DRT::Node* node) const
{
  // initialize variable for result
  double result(0.);

  // extract row map from solution vector
  const Epetra_BlockMap& phinpmap = porotimint_.Phinp()->Map();

  // test result value of phi field
  if (quantity == "phi")
    result = (*porotimint_.Phinp())[phinpmap.LID(porotimint_.Discretization()->Dof(0, node, 0))];

  // test result value for a system of scalars
  else if (!quantity.compare(0, 3, "phi"))
  {
    // read species ID
    std::string k_string = quantity.substr(3);
    char* locator(NULL);
    int k = strtol(k_string.c_str(), &locator, 10) - 1;
    if (locator == k_string.c_str()) dserror("Couldn't read species ID!");

    if (porotimint_.Discretization()->NumDof(0, node) <= k)
      dserror("Species ID is larger than number of DOFs of node!");

    // extract result
    result = (*porotimint_.Phinp())[phinpmap.LID(porotimint_.Discretization()->Dof(0, node, k))];
  }

  // test result value of phi field
  else if (quantity == "pressure")
    result = (*porotimint_.Pressure())[phinpmap.LID(porotimint_.Discretization()->Dof(0, node, 0))];

  // test result value for a system of scalars
  else if (!quantity.compare(0, 8, "pressure"))
  {
    // read species ID
    std::string k_string = quantity.substr(8);
    char* locator(NULL);
    int k = strtol(k_string.c_str(), &locator, 10) - 1;
    if (locator == k_string.c_str()) dserror("Couldn't read pressure ID!");

    if (porotimint_.Discretization()->NumDof(0, node) <= k)
      dserror("Pressure ID is larger than number of DOFs of node!");

    // extract result
    result = (*porotimint_.Pressure())[phinpmap.LID(porotimint_.Discretization()->Dof(0, node, k))];
  }

  // test result value of phi field
  else if (quantity == "saturation")
    result =
        (*porotimint_.Saturation())[phinpmap.LID(porotimint_.Discretization()->Dof(0, node, 0))];

  // test result value for a system of scalars
  else if (!quantity.compare(0, 10, "saturation"))
  {
    // read species ID
    std::string k_string = quantity.substr(10);
    char* locator(NULL);
    int k = strtol(k_string.c_str(), &locator, 10) - 1;
    if (locator == k_string.c_str()) dserror("Couldn't read saturation ID!");

    if (porotimint_.Discretization()->NumDof(0, node) <= k)
      dserror("Saturation ID is larger than number of DOFs of node!");

    // extract result
    result =
        (*porotimint_.Saturation())[phinpmap.LID(porotimint_.Discretization()->Dof(0, node, k))];
  }

  // catch unknown quantity strings
  else
    dserror("Quantity '%s' not supported in result test!", quantity.c_str());

  return result;
}  // POROFLUIDMULTIPHASE::ResultTest::ResultNode

/*----------------------------------------------------------------------*
 | get element result to be tested                     kremheller 10/19 |
 *----------------------------------------------------------------------*/
double POROFLUIDMULTIPHASE::ResultTest::ResultElement(
    const std::string quantity, const DRT::Element* element) const
{
  // initialize variable for result
  double result(0.);

  if (quantity == "bloodvesselvolfrac")
  {
    result =
        (*porotimint_.MeshTyingStrategy()
                ->BloodVesselVolumeFraction())[porotimint_.Discretization()->ElementRowMap()->LID(
            element->Id())];
  }
  // catch unknown quantity strings
  else
    dserror("Quantity '%s' not supported in result test!", quantity.c_str());

  return result;
}

/*-------------------------------------------------------------------------------------*
 | test special quantity not associated with a particular element or node  vuong 08/16 |
 *-------------------------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::ResultTest::TestSpecial(
    DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  // make sure that quantity is tested only once
  if (porotimint_.Discretization()->Comm().MyPID() == 0)
  {
    // extract name of quantity to be tested
    std::string quantity;
    res.ExtractString("QUANTITY", quantity);

    // get result to be tested
    const double result = ResultSpecial(quantity);

    // compare values
    const int err = CompareValues(result, "SPECIAL", res);
    nerr += err;
    test_count++;
  }

  return;
}


/*----------------------------------------------------------------------*
 | get special result to be tested                          vuong 08/16 |
 *----------------------------------------------------------------------*/
double POROFLUIDMULTIPHASE::ResultTest::ResultSpecial(
    const std::string quantity  //! name of quantity to be tested
) const
{
  // initialize variable for result
  double result(0.);

  if (quantity == "numiterlastnewton") result = (double)porotimint_.IterNum();
  // result test of domain integrals
  else if (!quantity.compare(0, 22, "domain_integral_value_"))
  {
    // get the index of the value which should be checked
    std::string suffix = quantity.substr(22);
    int idx = -1;
    try
    {
      idx = std::stoi(suffix);
    }
    catch (const std::invalid_argument& e)
    {
      dserror(
          "You provided the wrong format for output of domain_integral_values. The integer number "
          "must be at the very last position of the name, separated by an underscore.\n"
          "The correct format is: domain_integral_value_<number>");
    }

    // index should be in range [0, number_functions - 1]
    if (idx < 0 || idx >= porotimint_.NumDomainIntFunctions())
      dserror("detected wrong index %i, index should be in range [0,%i]", idx,
          porotimint_.NumDomainIntFunctions() - 1);

    // return the result
    result = (*porotimint_.DomainIntValues())[idx];
  }
  // catch unknown quantity strings
  else
    dserror("Quantity '%s' not supported in result test!", quantity.c_str());

  return result;
}  // POROFLUIDMULTIPHASE::ResultTest::ResultSpecial
