/*----------------------------------------------------------------------*/
/*!
\file drt_resulttest.cpp

\brief general result test framework

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/kuettler
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/

#include "drt_resulttest.H"
#include "drt_dserror.H"
#include "../drt_lib/drt_colors.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_io/io_control.H"
#include "drt_linedefinition.H"
#include "drt_inputreader.H"


DRT::ResultTest::ResultTest()
{
}

DRT::ResultTest::~ResultTest()
{
}

void DRT::ResultTest::TestElement(DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  dserror("no element test available");
}

void DRT::ResultTest::TestNode(DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  dserror("no node test available");
}

void DRT::ResultTest::TestSpecial(DRT::INPUT::LineDefinition& res, int& nerr, int& test_count)
{
  dserror("no special case test available");
}


/*----------------------------------------------------------------------*/
/*!
 \brief Compare \a actresult with \a givenresult and return 0 if they are
 considered to be equal.

 Compare \a actresult with \a givenresult and return 0 if they are
 considered to be equal.

 \param err        the file where to document both values
 \param res        the describtion of the expected result including name and tolerance

 \author uk
 \date 06/04
 */
/*----------------------------------------------------------------------*/
int DRT::ResultTest::CompareValues(double actresult, DRT::INPUT::LineDefinition& res)
{
  FILE *err = DRT::Problem::Instance()->ErrorFile()->Handle();
  int ret = 0;
  std::string name;
  double givenresult;
  double tolerance;
  res.ExtractDouble("VALUE",givenresult);
  res.ExtractDouble("TOLERANCE",tolerance);
  res.ExtractString("NAME",name);

  fprintf(err,"actual = %.17e, given = %.17e, diff = %.17e\n",
          actresult, givenresult, actresult-givenresult);
  if (!(fabs(fabs(actresult-givenresult)-fabs(actresult-givenresult)) < tolerance) )
  {
    printf("RESULTCHECK: %s is NAN!\n", name.c_str());
    ret = 1;
  }
  else if (fabs(actresult-givenresult) > tolerance)
  {
    printf("RESULTCHECK: %s not correct. actresult=%.17e, givenresult=%.17e\n",
           name.c_str(), actresult, givenresult);
    ret = 1;
  }
  return ret;
}


///////////////////////////////////////////////////////////////////


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::ResultTestManager::AddFieldTest(Teuchos::RCP<ResultTest> test)
{
  fieldtest_.push_back(test);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::ResultTestManager::TestAll(const Epetra_Comm& comm)
{
  FILE *err = DRT::Problem::Instance()->ErrorFile()->Handle();
  int nerr = 0;
  int test_count = 0;

  if (comm.MyPID()==0)
    cout << "\nChecking results ...\n";

  for (unsigned i=0; i<results_.size(); ++i)
  {
    DRT::INPUT::LineDefinition& res = *results_[i];

    for (unsigned j=0; j<fieldtest_.size(); ++j)
    {
      if (fieldtest_[j]->Match(res))
      {
        if (res.HaveNamed("ELEMENT"))
          fieldtest_[j]->TestElement(res,nerr,test_count);
        else if (res.HaveNamed("NODE"))
          fieldtest_[j]->TestNode(res,nerr,test_count);
        else
          fieldtest_[j]->TestSpecial(res,nerr,test_count);
      }
    }
  }

  int numerr;
  comm.SumAll(&nerr,&numerr,1);

  int size = results_.size();

  if (numerr > 0)
  {
    dserror("Result check failed with %d errors out of %d tests", numerr, size);
  }
  else
    fprintf(err,"===========================================\n");

  /* test_count == -1 means we had a special test routine. It's thus
   * illegal to use both a special routine and single tests. But who
   * wants that? */
  int count;
  if (test_count > -1)
  {
    comm.SumAll(&test_count,&count,1);

    /* It's indeed possible to count more tests than expected if you
     * happen to test values of a boundary element. We don't want this
     * dserror to go off in that case. */
    if (count < size)
    {
      dserror("expected %d tests but performed %d", size, count);
    }
  }

  if (comm.MyPID()==0)
    cout << "\n" GREEN_LIGHT "OK (" << count << ")" END_COLOR "\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<DRT::INPUT::Lines> DRT::ResultTestManager::ValidResultLines()
{
  DRT::INPUT::LineDefinition structure;
  structure
    .AddTag("STRUCTURE")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  DRT::INPUT::LineDefinition fluid;
  fluid
    .AddTag("FLUID")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  DRT::INPUT::LineDefinition ale;
  ale
    .AddTag("ALE")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  DRT::INPUT::LineDefinition thermal;
  thermal
    .AddTag("THERMAL")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  DRT::INPUT::LineDefinition scatra;
  scatra
    .AddTag("SCATRA")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  DRT::INPUT::LineDefinition red_airway;
  red_airway
    .AddTag("RED_AIRWAY")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  DRT::INPUT::LineDefinition art_net;
  art_net
    .AddTag("ARTNET")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  DRT::INPUT::LineDefinition fld_adj;
  fld_adj
    .AddTag("ADJOINT")
    .AddNamedInt("DIS")
    .AddNamedInt("NODE")
    .AddNamedString("POSITION")
    .AddNamedString("NAME")
    .AddNamedDouble("VALUE")
    .AddNamedDouble("TOLERANCE")
    ;

  Teuchos::RCP<DRT::INPUT::Lines> lines = Teuchos::rcp(new DRT::INPUT::Lines("RESULT DESCRIPTION"));
  lines->Add(structure);
  lines->Add(fluid);
  lines->Add(ale);
  lines->Add(thermal);
  lines->Add(scatra);
  lines->Add(red_airway);
  lines->Add(art_net);
  lines->Add(fld_adj);
  return lines;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::ResultTestManager::ReadInput(DRT::INPUT::DatFileReader& reader)
{
  Teuchos::RCP<DRT::INPUT::Lines> lines = ValidResultLines();
  results_ = lines->Read(reader);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void PrintResultDescrDatHeader()
{
  DRT::ResultTestManager resulttestmanager;
  Teuchos::RCP<DRT::INPUT::Lines> lines = resulttestmanager.ValidResultLines();

  lines->Print(std::cout);
}


