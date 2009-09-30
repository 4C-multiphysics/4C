/*----------------------------------------------------------------------*/
/*!
\file pre_exodus.cpp

\brief preprocessor for exodusII format

<pre>
Maintainer: Moritz & Georg
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/frenzel
            089 - 289-15240
</pre>

Pre_exodus contains classes to open and preprocess exodusII files into the
drt of Baci. It uses the "valid-parameters"-list defined in Baci for preparing
a up-to-date Baci header and another file specifying element and boundary
specifications based on "valid-conditions". As result either a preliminary
input file set is suggestioned, or the well-known .dat file is created.
Addionally, specify an already existing BACI input file in order to validate
its parameters and conditions.

*/
/*----------------------------------------------------------------------*/
#ifdef D_EXODUS
#ifdef CCADISCRET
#include "pre_exodus.H"
#include <Teuchos_RefCountPtr.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include "Epetra_Time.h"
#include "Teuchos_TimeMonitor.hpp"
#include "../drt_inpar/drt_validparameters.H"
#include "../drt_inpar/drt_validmaterials.H"
#include "../drt_inpar/drt_validconditions.H"
#include "../drt_lib/drt_conditiondefinition.H"
#include "../drt_lib/drt_elementdefinition.H"
#include "pre_exodus_reader.H"
#include "pre_exodus_soshextrusion.H"
#include "pre_exodus_writedat.H"
#include "pre_exodus_readbc.H"
#include "pre_exodus_validate.H"
#include "pre_exodus_centerline.H"


using namespace std;
using namespace Teuchos;


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int main(
        int argc,
        char** argv)
{

// communication
#ifdef PARALLEL
  MPI_Init(&argc,&argv);

  int myrank = 0;
  int nproc  = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if ((nproc>1) && (myrank==0)) dserror("Using more than one processor is not supported.");
  RefCountPtr<Epetra_Comm> comm = rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
#else
  RefCountPtr<Epetra_Comm> comm = rcp(new Epetra_SerialComm());
#endif

  string exofile;
  string bcfile;
  string headfile;
  string datfile;
  string cline;

  // related to solid shell extrusion
  double soshthickness = 0.0;
  int soshnumlayer = 1;
  int soshseedid = 0;
  int soshgmsh = -1;
  int concat2loose = 0;
  int diveblocks = 0;

  // related to centerline
  vector<double> cline_coordcorr(3);
  double clinedx = 0.0;
  double clinedy = 0.0;
  double clinedz = 0.0;

  // related to quad->tri conversion
  bool quadtri = false;

  Teuchos::CommandLineProcessor My_CLP;
  My_CLP.setDocString("Preprocessor Exodus2Baci \n");
  My_CLP.throwExceptions(false);
  My_CLP.setOption("exo",&exofile,"exodus file to open");
  My_CLP.setOption("bc",&bcfile,"bc's and ele's file to open");
  My_CLP.setOption("head",&headfile,"baci header file to open");
  My_CLP.setOption("dat",&datfile,"output .dat file name [defaults to exodus file name]");
  // here options related to solid shell extrusion are defined
  My_CLP.setOption("gensosh",&soshthickness,"generate solid-shell body with given thickness");
  My_CLP.setOption("numlayer",&soshnumlayer,"number of layers of generated solid-shell body");
  My_CLP.setOption("diveblocks",&diveblocks,"if larger 0 the xxx inner elements of generated layers are put into first eblock, the rest into second");
  My_CLP.setOption("seedid",&soshseedid,"id where to start extrusion, default is first");
  My_CLP.setOption("gmsh",&soshgmsh,"gmsh output of xxx elements, default off, 0 all eles");
  My_CLP.setOption("concf",&concat2loose,"concatenate extruded volume with base, however loose every xxx'th node, default 0=off=fsi");

  // centerline related
  My_CLP.setOption("cline",&cline,"generate local element coordinate systems based on centerline file, or mesh line (set to 'mesh'");
  My_CLP.setOption("clinedx",&clinedx,"move centerline coords to align with mesh: delta x");
  My_CLP.setOption("clinedy",&clinedy,"move centerline coords to align with mesh: delta y");
  My_CLP.setOption("clinedz",&clinedz,"move centerline coords to align with mesh: delta z");
  map<int,map<int,vector<vector<double> > > >elecenterlineinfo;

  // check for quad->tri conversion
  My_CLP.setOption("quadtri","noquadtri",&quadtri,"transform quads to tris by cutting in two halves");

  CommandLineProcessor::EParseCommandLineReturn
    parseReturn = My_CLP.parse(argc,argv);

  if (parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED)
  {
    exit(0);
  }
  if (parseReturn != CommandLineProcessor::PARSE_SUCCESSFUL)
  {
    dserror("CommandLineProcessor reported an error");
  }

  // create a problem instance
  Teuchos::RCP<DRT::Problem> problem = DRT::Problem::Instance();

  // create error files
  // call this one rather early, since ReadConditions etc
  // underlying methods may try to write to allfiles.out_err
  // this (old-style) global variable is (indirectly) set as well
  problem->OpenErrorFile(*comm, datfile);

  // centerline related: transfer separate doubles into vector
  cline_coordcorr[0] = clinedx; cline_coordcorr[1] = clinedy; cline_coordcorr[2] = clinedz;

  /**************************************************************************
   * Start with the preprocessing
   **************************************************************************/
  if (exofile=="")
  {
    if (datfile!="")
    {
      // just validate a given BACI input file
      EXODUS::ValidateInputFile(comm, datfile);
      return 0;
    }
    else
    {
      cout<<"No Exodus II file was found"<<endl;
      My_CLP.printHelpMessage(argv[0],cout);
      exit(1);
    }
  }

  // create mesh object based on given exodus II file
  EXODUS::Mesh mymesh(exofile);
  // print infos to cout
  mymesh.Print(cout);

  /**************************************************************************
   * Edit a existing Mesh, e.g. extrusion of surface
   **************************************************************************/

  // generate solid shell extrusion based on exodus file
  if (soshthickness!=0.0){
    if (exofile=="") dserror("no exofile specified for extrusion");
    if (soshnumlayer <= diveblocks) dserror("number of layers and inner-layer elements mismatch, check if numlayer>diveblocks!");
    EXODUS::Mesh mysosh = EXODUS::SolidShellExtrusion(mymesh, soshthickness, soshnumlayer, soshseedid,
            soshgmsh, concat2loose, diveblocks, cline, cline_coordcorr);
    mysosh.WriteMesh("extr_" + exofile);

    exit(0);
  }

  // generate local element coordinate systems based on centerline
  if (cline!=""){
    elecenterlineinfo = EleCenterlineInfo(cline,mymesh,cline_coordcorr);
  }

  // transform quads->tris
  if (quadtri){
    EXODUS::Mesh trimesh = EXODUS::QuadtoTri(mymesh);
    trimesh.WriteMesh("tri_" + exofile);
    exit(0);
  }

  /**************************************************************************
   * Read ControlFile for Boundary and Element descriptions
   **************************************************************************/

  // declare empty vectors for holding "boundary" conditions
  vector<EXODUS::elem_def> eledefs;
  vector<EXODUS::cond_def> condefs;

  if (bcfile=="")
  {
    int error = EXODUS::CreateDefaultBCFile(mymesh);
    if (error!=0) dserror("Creation of default bc-file not successful.");
  }
  else
  {
    // read provided bc-file
    EXODUS::ReadBCFile(bcfile,eledefs,condefs);

    int sum = mymesh.GetNumElementBlocks() + mymesh.GetNumNodeSets() + mymesh.GetNumSideSets();
    int test = eledefs.size() + condefs.size();
    if (test != sum) cout << "Your " << test << " definitions do not match the " << sum << " entities in your mesh!" <<endl << "(This is OK, if more than one BC is applied to an entity, e.g in FSI simulations)" << endl;
  }

  /**************************************************************************
   * Read HeaderFile for 'header' parameters, e.g. solver, dynamic, material
   * or create a default HeaderFile
   **************************************************************************/
  if (headfile=="")
  {
    const string defaultheadfilename = "default.head";
    cout << "found no header file           --> creating "<<defaultheadfilename<< endl;

    // open default header file
    ofstream defaulthead(defaultheadfilename.c_str());
    if (!defaulthead) dserror("failed to open file: %s", defaultheadfilename.c_str());

    // get valid input parameters
    Teuchos::RCP<const Teuchos::ParameterList> list = DRT::INPUT::ValidParameters();

    // write default .dat header into file
    stringstream prelimhead;
    DRT::INPUT::PrintDatHeader(prelimhead,*list);
    string headstring = prelimhead.str();
    size_t size_section = headstring.find("-------------------------------------------------------PROBLEM SIZE");
    if (size_section!=string::npos){
      size_t typ_section = headstring.find("--------------------------------------------------------PROBLEM TYP");
      headstring.erase(size_section,typ_section-size_section);
    }
    defaulthead << headstring;

    // get valid input materials
    {
      Teuchos::RCP<std::vector<Teuchos::RCP<DRT::INPUT::MaterialDefinition> > > mlist
       = DRT::INPUT::ValidMaterials();
      DRT::INPUT::PrintEmptyMaterialDefinitions(defaulthead, *mlist,false);
    }

    defaulthead <<
    "-------------------------------------------------------LOAD CURVES"<<endl<<
    "------------------------------------------------------------CURVE1"<<endl<<
    "------------------------------------------------------------CURVE2"<<endl<<
    "------------------------------------------------------------CURVE3"<<endl<<
    "------------------------------------------------------------CURVE4"<<endl<<
    "------------------------------------------------------------FUNCT1"<<endl<<
    "------------------------------------------------------------FUNCT2"<<endl<<
    "------------------------------------------------------------FUNCT3"<<endl<<
    "------------------------------------------------------------FUNCT4"<<endl;

    // default result-test lines
    {
    DRT::ResultTestManager resulttestmanager;
    Teuchos::RCP<DRT::INPUT::Lines> lines = resulttestmanager.ValidResultLines();
    lines->Print(defaulthead);
    }

    // close default header file
    if (defaulthead.is_open()) defaulthead.close();
  }

  /**************************************************************************
   * Finally, create and validate the BACI input file
   **************************************************************************/
  if ((headfile!="") && (bcfile!="") && (exofile!=""))
  {
    // set default dat-file name if needed
    if (datfile=="")
    {
      const string exofilebasename = exofile.substr(0,exofile.find_last_of("."));
      datfile=exofilebasename+".dat";
    }

    // screen info
    cout << "creating and checking BACI input file       --> " << datfile << endl;
    RCP<Time> timer = TimeMonitor::getNewTimer("pre-exodus timer");

    // check for positive Element-Center-Jacobians and otherwise rewind them
    {
      timer->start();
      ValidateMeshElementJacobians(mymesh);
      timer->stop();
      cout << "...Ensure positive element jacobians";
      cout << "        in...." << timer->totalElapsedTime() <<" secs" << endl;
      timer->reset();
    }

    // write the BACI input file
    {
      timer->start();
      EXODUS::WriteDatFile(datfile, mymesh, headfile, eledefs, condefs,elecenterlineinfo);
      timer->stop();
      cout << "...Writing dat-file";
      cout << "                         in...." << timer->totalElapsedTime() << " secs" << endl;
      timer->reset();
    }

    //validate the generated BACI input file
    EXODUS::ValidateInputFile(comm, datfile);
  }

  // free the global problem instance
  problem->Done();

#ifdef PARALLEL
  MPI_Finalize();
#endif

  return 0;

} //main.cpp


/*----------------------------------------------------------------------*/
/* create default bc file                                               */
/*----------------------------------------------------------------------*/
int EXODUS::CreateDefaultBCFile(EXODUS::Mesh& mymesh)
{
  string defaultbcfilename = "default.bc";
  cout << "found no BC specification file --> creating " <<defaultbcfilename<< endl;

  // open default bc specification file
  ofstream defaultbc(defaultbcfilename.c_str());
  if (!defaultbc)
    dserror("failed to open file: %s", defaultbcfilename.c_str());

  // write mesh verbosely
  defaultbc<<"----------- Mesh contents -----------"<<endl<<endl;
  mymesh.Print(defaultbc, false);

  // give examples for element and boundary condition syntax
  defaultbc<<"---------- Syntax examples ----------"<<endl<<endl<<
  "Element Block, named: "<<endl<<
  "of Shape: TET4"<<endl<<
  "has 9417816 Elements"<<endl<<
  "'*eb0=\"ELEMENT\"'"<<endl<<
  "sectionname=\"FLUID\""<<endl<<
  "description=\"MAT 1 NA Euler GP_TET 4 GP_ALT standard\""<<endl<<
  "elementname=\"FLUID3\" \n"<<endl<<
  "Element Block, named: "<<endl<<
  "of Shape: HEX8"<<endl<<
  "has 9417816 Elements"<<endl<<
  "'*eb0=\"ELEMENT\"'"<<endl<<
  "sectionname=\"STRUCTURE\""<<endl<<
  "description=\"MAT 1 EAS mild\""<<endl<<
  "elementname=\"SOLIDH8\" \n"<<endl<<
  "Node Set, named:"<<endl<<
  "Property Name: INFLOW"<<endl<<
  "has 45107 Nodes"<<endl<<
  "'*ns0=\"CONDITION\"'"<<endl<<
  "sectionname=\"DESIGN SURF DIRICH CONDITIONS\""<<endl<<
  "description=\"1 1 1 0 0 0 2.0 0.0 0.0 0.0 0.0 0.0  1 none none none none none  1 0 0 0 0 0\""
  <<endl<<endl;

  defaultbc << "MIND that you can specify a condition also on an ElementBlock, just replace 'ELEMENT' with 'CONDITION'"<<endl;
  defaultbc << "The 'E num' in the dat-file depends on the order of the specification below" << endl;
  defaultbc<<"------------------------------------------------BCSPECS"<<endl<<endl;

  // write ElementBlocks with specification proposal
  const map<int,RCP<EXODUS::ElementBlock> > myblocks = mymesh.GetElementBlocks();
  map<int,RCP<EXODUS::ElementBlock> >::const_iterator it;
  for (it = myblocks.begin(); it != myblocks.end(); ++it){
    it->second->Print(defaultbc);
    defaultbc<<"*eb"<< it->first << "=\"ELEMENT\""<<endl
    <<"sectionname=\"\""<<endl
    <<"description=\"\""<<endl
    <<"elementname=\"\""<<endl
    //<<"elementshape=\""
    //<< DRT::DistypeToString(PreShapeToDrt(it->second.GetShape()))<<"\""<<endl
    <<endl;
  }

  // write NodeSets with specification proposal
  const map<int,EXODUS::NodeSet> mynodesets = mymesh.GetNodeSets();
  map<int,EXODUS::NodeSet>::const_iterator ins;
  for (ins =mynodesets.begin(); ins != mynodesets.end(); ++ins){
    ins->second.Print(defaultbc);
    defaultbc<<"*ns"<< ins->first << "=\"CONDITION\""<<endl
    <<"sectionname=\"\""<<endl
    <<"description=\"\""<<endl
    <<endl;
  }

  // write SideSets with specification proposal
  const map<int,EXODUS::SideSet> mysidesets = mymesh.GetSideSets();
  map<int,EXODUS::SideSet>::const_iterator iss;
  for (iss = mysidesets.begin(); iss!=mysidesets.end(); ++iss){
    iss->second.Print(defaultbc);
    defaultbc<<"*ss"<< iss->first << "=\"CONDITION\""<<endl
    <<"sectionname=\"\""<<endl
    <<"description=\"\""<<endl
    <<endl;
  }

  // print validconditions as proposal
  defaultbc << "-----------------------------------------VALIDCONDITIONS"<< endl;
  Teuchos::RCP<std::vector<Teuchos::RCP<DRT::INPUT::ConditionDefinition> > > condlist = DRT::INPUT::ValidConditions();
  DRT::INPUT::PrintEmptyConditionDefinitions(defaultbc,*condlist,false);

  // print valid element lines as proposal
  defaultbc << endl << endl;
  DRT::INPUT::ElementDefinition ed;
  ed.PrintElementDatHeaderToStream(defaultbc);

  // close default bc specification file
  if (defaultbc.is_open())
    defaultbc.close();

  return 0;
}


#endif
#endif
