/*----------------------------------------------------------------------*/
/*!
\file pre_exodus_reader.cpp

\brief preprocessor reader for exodusII format 

<pre>
Maintainer: Moritz & Georg
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/frenzel
            089 - 289-15240
</pre>

Here everything related with the exodus format and the accessible data
is handed to a c++ object mesh.
*/
/*----------------------------------------------------------------------*/
#ifdef D_EXODUS
#include "pre_exodus_reader.H"
#include "Epetra_SerialComm.h"
#include "Epetra_Time.h"
#include "Teuchos_TimeMonitor.hpp"
#include "../drt_lib/drt_utils_local_connectivity_matrices.H"

#ifdef PARALLEL
#include <mpi.h>
#endif

using namespace std;
using namespace Teuchos;

/*----------------------------------------------------------------------*
 |  ctor (public)                                              maf 12/07|
 *----------------------------------------------------------------------*/
EXODUS::Mesh::Mesh(string exofilename)
{
  int error;
  int CPU_word_size,IO_word_size;
  float exoversion;                   /* version of exodus */
  CPU_word_size = sizeof(float);      /* float or double */
  IO_word_size = 0;                   /* use what is stored in file */

  //cout << "meshfilename: " << exofilename.c_str() << endl;

  const char *exofilenamechar;
  exofilenamechar=exofilename.c_str();

  // open EXODUS II file
  exoid_ = ex_open(exofilenamechar,EX_READ,&CPU_word_size,&IO_word_size,&exoversion);
  if (exoid_<=0)
	  dserror("Error while opening EXODUS II file %s",exofilenamechar);

  // print version
  cout<<exofilename<<" was created with EXODUS II library version "<<exoversion<<endl;

  // read database parameters
  int num_elem_blk,num_node_sets,num_side_sets,num_nodes;
  char title[MAX_LINE_LENGTH+1];
  error = ex_get_init (exoid_, title, &num_dim_, &num_nodes,&num_elem_, &num_elem_blk, &num_node_sets, &num_side_sets);
  string stitle(title,int(MAX_LINE_LENGTH+1));
  title_ = stitle;
  if (num_dim_ != 3) dserror("only 3 dimensions for mesh, yet");

  // get nodal coordinates
  vector<float> x(num_nodes);
  vector<float> y(num_nodes);
  vector<float> z(num_nodes);
  error = ex_get_coord(exoid_,&x[0],&y[0],&z[0]);
  if (error != 0) dserror("exo error returned");
  
  // store nodes in map
  for (int i = 0; i < num_nodes; ++i) {
    vector<double> coords;
    coords.push_back(x[i]);
    coords.push_back(y[i]);
    coords.push_back(z[i]);
    nodes_.insert(std::pair<int,vector<double> >(i,coords));
  }

  // Get all ElementBlocks
  vector<int> epropID(num_elem_blk);
  vector<int> ebids(num_elem_blk);
  error = ex_get_elem_blk_ids(exoid_,&(ebids[0]));
  error = ex_get_prop_array(exoid_, EX_ELEM_BLOCK, "ID", &(epropID[0]));
  for (int i = 0; i < num_elem_blk; ++i) 
  {
	  // Read Element Blocks into Map
    char mychar[MAX_STR_LENGTH+1];
    int num_el_in_blk,num_nod_per_elem,num_attr;
    //error = ex_get_elem_block (exoid_, epropID[i], mychar, &num_el_in_blk, &num_nod_per_elem, &num_attr);
    error = ex_get_elem_block (exoid_, ebids[i], mychar, &num_el_in_blk, &num_nod_per_elem, &num_attr);
    // prefer string to store element type
    string ele_type(mychar,int (MAX_STR_LENGTH));
    
    // get ElementBlock name
    error = ex_get_name (exoid_, EX_ELEM_BLOCK, ebids[i], mychar);
    // prefer string to store name
    string blockname(mychar,int (MAX_STR_LENGTH));
    
    // get element connectivity
    vector<int> allconn(num_nod_per_elem*num_el_in_blk);
    error = ex_get_elem_conn(exoid_,ebids[i],&allconn[0]);
    if (error != 0) dserror("exo error returned");
	  map<int,vector<int> > eleconn;
	  for (int j = 0; j < num_el_in_blk; ++j) {
	    vector<int> actconn;
	    for (int k = 0; k < num_nod_per_elem; ++k){
        actconn.push_back(allconn[k + j*num_nod_per_elem]);
	    }
	    eleconn.insert(std::pair<int,vector<int> >(j,actconn));
    }
	  ElementBlock actEleBlock(StringToShape(ele_type), eleconn, blockname);
	  
	  // Add this ElementBlock into Mesh map
	  elementBlocks_.insert(std::pair<int,ElementBlock>(i,actEleBlock));
  }
  
  // get all NodeSets
  map<int,NodeSet> prelimNodeSets;   // prelim due to possible prop names
  vector<int> npropID(num_node_sets);
  error = ex_get_prop_array(exoid_, EX_NODE_SET, "ID", &(npropID[0]));
  for (int i = 0; i < num_node_sets; ++i) {
    // Read NodeSet params
    int num_nodes_in_set,num_df_in_set;
    error = ex_get_node_set_param (exoid_, npropID[i], &num_nodes_in_set,&num_df_in_set);
    
    // get NodeSet name
    char mychar[MAX_STR_LENGTH+1];
    error = ex_get_name (exoid_, EX_NODE_SET, npropID[i], mychar);
    // prefer string to store name
    string nodesetname(mychar, int(MAX_STR_LENGTH));

    // get nodes in node set
    vector<int> node_set_node_list(num_nodes_in_set);
    error = ex_get_node_set (exoid_, npropID[i], &(node_set_node_list[0]));
    if (error != 0) dserror("error reading node set");
    set<int> nodes_in_set;
    for (int j = 0; j < num_nodes_in_set; ++j) nodes_in_set.insert(node_set_node_list[j]);
    NodeSet actNodeSet(nodes_in_set,nodesetname,"none");
    
    // Add this NodeSet into Mesh map (here prelim due to pro names)
    prelimNodeSets.insert(std::pair<int,NodeSet>(i,actNodeSet));
  }
  
  /* Read NodeSet property names ***********************************************
   * They are assigned by ICEM and provide recognition */
  int num_props;
  float fdum;
  char *cdum;
  error = ex_inquire (exoid_, EX_INQ_NS_PROP, &num_props, &fdum, cdum);
  // allocate memory for NodeSet property names
  char** prop_names = new char*[num_props];
  for (int i=0; i<num_props; ++i)
  {
     prop_names[i] = new char[MAX_STR_LENGTH+1];
  }
  // get prop names of node sets  
  error = ex_get_prop_names(exoid_,EX_NODE_SET,prop_names);
  
  // Add prop names to final Mesh NodeSet if available
  if ((num_props-1) == num_node_sets){
    for (int i = 1; i < num_props; ++i) {
      string propname(prop_names[i], int(MAX_STR_LENGTH));
      
      map<int,NodeSet>::const_iterator blub = prelimNodeSets.find(i-1);
      if (blub == prelimNodeSets.end())
        dserror("impossible");
      const NodeSet actNodeSet = blub->second;
      //NodeSet actNodeSet = nodeSets_[i-1];
      const NodeSet newNodeSet(actNodeSet.GetNodeSet(),actNodeSet.GetName(),propname);
      //nodeSets_[i-1] = newNodeSet;
      nodeSets_.insert(std::pair<int,NodeSet>(blub->first,newNodeSet));
    }
  } else {
    // this is the standard case without prop names
    nodeSets_ = prelimNodeSets;
  }
  // ***************************************************************************
  
  // get all SideSets
  vector<int> spropID(num_side_sets);
  error = ex_get_prop_array(exoid_, EX_SIDE_SET, "ID", &(spropID[0]));
  for (int i = 0; i < num_side_sets; ++i) {
    // get SideSet name
    char mychar[MAX_STR_LENGTH+1];
    error = ex_get_name (exoid_, EX_SIDE_SET, spropID[i], mychar);
    // prefer string to store name
    string sidesetname(mychar, int(MAX_STR_LENGTH));

    // Read SideSet params
    int num_side_in_set,num_dist_fact_in_set;
    error = ex_get_side_set_param (exoid_, spropID[i], &num_side_in_set,&num_dist_fact_in_set);
    
    // get SideSet
    vector<int> side_set_elem_list(num_side_in_set);
    vector<int> side_set_side_list(num_side_in_set);
    error = ex_get_side_set (exoid_, spropID[i], &(side_set_elem_list[0]),&(side_set_side_list[0]));
    if (error != 0) dserror("error reading side set");
    map<int,vector<int> > sides_in_set;
    for (int j = 0; j < num_side_in_set; ++j){
      vector<int> side(2); // first entry is element, second side
      side[0] = side_set_elem_list[j];
      side[1] = side_set_side_list[j];
      sides_in_set.insert(pair<int,vector<int> >(j,side));
    }
    
    SideSet actSideSet(sides_in_set,sidesetname);
    
    // Add this SideSet into Mesh map
    sideSets_.insert(std::pair<int,SideSet>(i,actSideSet));
  }
  
  // close ExoFile
  CloseExo();

  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                              maf 12/07|
 *----------------------------------------------------------------------*/
EXODUS::Mesh::~Mesh()
{
  //CloseExo();
  return;
}

/*----------------------------------------------------------------------*
 |  Extension constructor (public)                             maf 01/08|
 *----------------------------------------------------------------------*/
EXODUS::Mesh::Mesh(const EXODUS::Mesh basemesh,
                   const map<int,vector<double> >extNodes,
                   const map<int,ElementBlock> extBlocks,
                   const map<int,NodeSet> extNodesets,
                   const map<int,SideSet> extSidesets,
                   const string newtitle)
{
  // get all data from basemesh
  int basedim     = basemesh.GetNumDim();
  int basenumele  = basemesh.GetNumEle();
  map<int,vector<double> > baseNodes = basemesh.GetNodes();
  map<int,ElementBlock>    baseEblocks = basemesh.GetElementBlocks();
  map<int,NodeSet>         baseNodesets = basemesh.GetNodeSets();
  map<int,SideSet>         baseSidesets = basemesh.GetSideSets();
  
//  // get infos from extension
//  int extnumele = extBlocks.size();
  
  /********************* merge everything into new mesh ***********************/
  title_ = newtitle;
  num_dim_ = basedim;
  int total_num_elem = basenumele;
//  num_elem_ = basenumele; // + extnumele;
  exoid_ = basemesh.GetExoId(); //basefile still used for writing minor infos, e.g. qa record or coordnames
  
  // merge nodes
  map<int,vector<double> >::const_iterator i_node;
  for(i_node = baseNodes.begin(); i_node != baseNodes.end(); ++i_node){
    nodes_.insert(pair<int,vector<double> >(i_node->first,i_node->second));
  }
  for(i_node = extNodes.begin(); i_node != extNodes.end(); ++i_node){
    pair< map<int,vector<double> >::iterator, bool > check;
    check = nodes_.insert(pair<int,vector<double> >(i_node->first,i_node->second));
    if (check.second == false) dserror("Extension node already exists!");
  }
  
  // merge ElementBlocks
  map<int,ElementBlock>::const_iterator i_block;
  for(i_block = baseEblocks.begin(); i_block != baseEblocks.end(); ++i_block){
    elementBlocks_.insert(pair<int,ElementBlock>(i_block->first,i_block->second));
  }
  for(i_block = extBlocks.begin(); i_block != extBlocks.end(); ++i_block){
    pair< map<int,ElementBlock>::iterator, bool > check;
    check = elementBlocks_.insert(pair<int,ElementBlock>(i_block->first,i_block->second));
    if (check.second == false) dserror("Extension ElementBlock already exists!");
    else total_num_elem += i_block->second.GetNumEle();
  }
  num_elem_ = total_num_elem;
  
  // merge NodeSets
  map<int,NodeSet>::const_iterator i_ns;
  for(i_ns = baseNodesets.begin(); i_ns != baseNodesets.end(); ++i_ns){
    nodeSets_.insert(pair<int,NodeSet>(i_ns->first,i_ns->second));
  }
  for(i_ns = extNodesets.begin(); i_ns != extNodesets.end(); ++i_ns){
    pair< map<int,NodeSet>::iterator, bool > check;
    check = nodeSets_.insert(pair<int,NodeSet>(i_ns->first,i_ns->second));
    if (check.second == false) dserror("Extension NodeSet already exists!");
  }

  // merge SideSets
  map<int,SideSet>::const_iterator i_ss;
  for(i_ss = baseSidesets.begin(); i_ss != baseSidesets.end(); ++i_ss){
    sideSets_.insert(pair<int,SideSet>(i_ss->first,i_ss->second));
  }
  for(i_ss = extSidesets.begin(); i_ss != extSidesets.end(); ++i_ss){
    pair< map<int,SideSet>::iterator, bool > check;
    check = sideSets_.insert(pair<int,SideSet>(i_ss->first,i_ss->second));
    if (check.second == false) dserror("Extension SideSet already exists!");
  }
}


/*----------------------------------------------------------------------*
 |  Close corresponding Exofile(public)                        maf 12/07|
 *----------------------------------------------------------------------*/
void EXODUS::Mesh::CloseExo()
{
  // close exodus II file
  int exoid = GetExoId();
  int error = ex_close(exoid);
  if (error < 0)
    dserror("error while closing exodus II file");
}


/*----------------------------------------------------------------------*
 |  Print method (public)                                      maf 12/07|
 *----------------------------------------------------------------------*/
void EXODUS::Mesh::Print(ostream & os, bool verbose) const
{
  os << "Mesh consists of ";
  os << GetNumNodes() << " Nodes, ";
  os << num_elem_ << " Elements, organized in " << endl;
  os << GetNumElementBlocks() << " ElementBlocks, ";
  os << GetNumNodeSets() << " NodeSets, ";
  os << GetNumSideSets() << " SideSets ";
  os << endl << endl;
  if (verbose){
    os << "ElementBlocks" << endl;
    map<int,ElementBlock>::const_iterator it;
    map<int,ElementBlock> eleBlocks = GetElementBlocks();
    for (it=eleBlocks.begin(); it != eleBlocks.end(); it++){
      os << it->first << ": ";
      it->second.Print(os);
    }
    os << endl << "NodeSets" << endl;
    map<int,NodeSet>::const_iterator it2;
    map<int,NodeSet> nodeSets = GetNodeSets();
    for (it2=nodeSets.begin(); it2 != nodeSets.end(); it2++){
      os << "NodeSet " << it2->first << ": ";
      it2->second.Print(os);
    }
    os << endl << "SideSets" << endl;
    os << "Warning: SideSets are not yet fully supported by PreExodus!" << endl;
    map<int,SideSet>::const_iterator it3;
    map<int,SideSet> sideSets = GetSideSets();
    for (it3=sideSets.begin(); it3 != sideSets.end(); it3++){
      os << "SideSet " << it3->first << ": ";
      it3->second.Print(os);
    }
  }
}

EXODUS::ElementBlock EXODUS::Mesh::GetElementBlock(const int id) const
{
  return (elementBlocks_.find(id))->second;
}

EXODUS::NodeSet EXODUS::Mesh::GetNodeSet(const int id) const
{
  return (nodeSets_.find(id))->second;
}

EXODUS::SideSet EXODUS::Mesh::GetSideSet(const int id) const
{
  return (sideSets_.find(id))->second;
}

void EXODUS::Mesh::PrintNodes(ostream& os, bool storeid) const
{
  map<int,vector<double> >::const_iterator it;
  for (it=nodes_.begin(); it != nodes_.end(); it++){
    if (storeid) os << "MapID: " << it->first;
    int exoid = it->first + 1;
    os << " ExoID: " << exoid << " : ";
    const vector<double> mycoords = it->second;
    for (int i=0; i < signed(mycoords.size()); i++ ){
      os << mycoords[i] << ",";
    }
    os << endl;
  }
}

vector<double> EXODUS::Mesh::GetNodeExo(const int ExoNodeID) const
{
  int mapID = ExoNodeID - 1;
  map<int,vector<double> >::const_iterator  it = nodes_.find(mapID);
  return it->second;
}

vector<double> EXODUS::Mesh::GetNodeMap(const int MapNodeID) const
{
  map<int,vector<double> >::const_iterator  it = nodes_.find(MapNodeID);
  return it->second;
}

map<int,vector<int> > EXODUS::Mesh::GetSideSetConn(const SideSet sideset) const
{
  cout << "Creating SideSet Connectivity... " << endl;
  fflush(stdout);
  
  Epetra_SerialComm Comm;
  Epetra_Time time(Comm);
  RCP<Time> timetot;
  timetot= TimeMonitor::getNewTimer("Side Set Connect total");
  RCP<Time> time1 = TimeMonitor::getNewTimer("One Side Set");
  RCP<Time> time2 = TimeMonitor::getNewTimer("Get one Element Block");
  RCP<Time> time3 = TimeMonitor::getNewTimer("Get Ele Conn");
  RCP<Time> time4 = TimeMonitor::getNewTimer("Get one Ele");
  RCP<Time> time5 = TimeMonitor::getNewTimer("Build one Side Conn");
  RCP<Time> time6 = TimeMonitor::getNewTimer("One Side Set");
  RCP<Time> time7 = TimeMonitor::getNewTimer("Get all Eblocks and Econns");
  RCP<TimeMonitor> tm_total = rcp(new TimeMonitor(*timetot));

  map<int,vector<int> > conn;
  map <int,vector<int> > mysides = sideset.GetSideSet();
  map<int,vector<int> >::iterator i_side;
  
  map<int,EXODUS::ElementBlock> ebs = GetElementBlocks();
  map<int,EXODUS::ElementBlock>::const_iterator i_ebs;
  
  // Range Vector for global eleID identification in SideSet
  vector<int> glob_eb_erange(1,0);
  int rangebreak = 0;
  // Also we once get all EBlocks and EConns to enable quick access
  vector<EXODUS::ElementBlock> eblocks;
  vector<map<int,vector<int> > > econns;
  RCP<TimeMonitor> tm7 = rcp(new TimeMonitor(*time7));
  for (i_ebs = ebs.begin(); i_ebs != ebs.end(); ++i_ebs ){
    rangebreak += i_ebs->second.GetNumEle();
    glob_eb_erange.push_back(rangebreak);
    eblocks.push_back(i_ebs->second);
    econns.push_back(i_ebs->second.GetEleConn());
  }
  tm7 = null;
  
  // fill SideSet Connectivity
  //int perc = 1;
  int tetc=0,hexc=0,pyrc=0,wedgc=0;
  for (i_side = mysides.begin(); i_side != mysides.end(); ++i_side){
    RCP<TimeMonitor> tm1 = rcp(new TimeMonitor(*time1));
    int actele = i_side->second.at(0) -1;   //ExoIds start from 1, but we from 0
    int actface = i_side->second.at(1) -1;  //ExoIds start from 1, but we from 0
    // find actual EBlock where actele lies in
    int actebid;
    for(unsigned int i=0; i<glob_eb_erange.size(); ++i) 
      if (actele < glob_eb_erange[i]){ actebid = i-1; break;}
    RCP<TimeMonitor> tm2 = rcp(new TimeMonitor(*time2));
    //EXODUS::ElementBlock acteb = ebs.find(actebid)->second;
    tm2 = null;
    //EXODUS::ElementBlock::Shape actshape = acteb.GetShape();
    EXODUS::ElementBlock::Shape actshape = eblocks[actebid].GetShape();
    RCP<TimeMonitor> tm3 = rcp(new TimeMonitor(*time3));
    //map<int,vector<int> > acteconn = acteb.GetEleConn();
    tm3 = null;
    // get act parent ele from actual Side
    int parent_ele_id = actele - glob_eb_erange[actebid]; 
    RCP<TimeMonitor> tm4 = rcp(new TimeMonitor(*time4));
    //vector<int> parent_ele = acteconn.find(parent_ele_id)->second;
    vector<int> parent_ele = econns[actebid].find(parent_ele_id)->second;
    tm4 = null;
    // Face to ElementNode Map
    //// **** temporary hex map due to conflicts between side numbering exo<->baci
    RCP<TimeMonitor> tm5 = rcp(new TimeMonitor(*time5));
    switch (actshape){
    case ElementBlock::tet4: {actface = actface; tetc++; break;}
    case ElementBlock::hex8: {actface = HexSideNumberExoToBaci(actface); hexc++; break;}
    case ElementBlock::pyramid5: {
//      vector<vector<int> > test = DRT::UTILS::getEleNodeNumberingSurfaces(PreShapeToDrt(actshape));
//      for(unsigned int j=0; j<test.size(); ++j) PrintVec(cout,test[j]);
      actface = PyrSideNumberExoToBaci(actface); pyrc++; break;}
    case ElementBlock::wedge6: {
//      vector<vector<int> > test = DRT::UTILS::getEleNodeNumberingSurfaces(PreShapeToDrt(actshape));
//      for(unsigned int j=0; j<test.size(); ++j) PrintVec(cout,test[j]);
      actface = actface; wedgc++; break;
      }
    default: {cout << ShapeToString(actshape) << ":" << endl; dserror("Parent Element Type not supported");}
    }
    vector<int> childmap = DRT::UTILS::getEleNodeNumberingSurfaces(PreShapeToDrt(actshape))[actface];
    // child gets its node ids
    vector<int> child;
    for(unsigned int j=0; j<childmap.size(); ++j)
      child.push_back(parent_ele[childmap[j]]);
//    PrintVec(cout,childmap);
//    PrintVec(cout,child);
    // some checking
    if ((child.size() != 3) && (child.size() != 4)){
      PrintVec(cout,child);
      PrintVec(cout,childmap);
      PrintVec(cout,parent_ele);
      cout << ShapeToString(actshape) << ",Face: " << actface << ",childsize:" << child.size() << endl;
      dserror("Child Ele error");
    }
    // insert child into SideSet Connectivity
    conn.insert(pair<int,vector<int> >(i_side->first,child));
    tm5 = null;
    
//    // progress output
//    tm1 = null;
//    if (signed(i_side->first) == perc * signed(mysides.size())/100){
//      TimeMonitor::summarize();
//      cout << perc << " % of " << mysides.size() << " Sides done" << endl;
//      perc ++;
//      fflush(stdout);
//    }
  }
  tm_total = null;
//  TimeMonitor::summarize();
  cout << "...done" << endl;
  fflush(stdout);

  return conn;
}

map<int,vector<int> > EXODUS::Mesh::GetSideSetConn(const SideSet sideset, bool checkoutside) const
{
  cout << "Creating SideSet Connectivity with outside-check... " << endl;
  fflush(stdout);
  
  Epetra_SerialComm Comm;
  Epetra_Time time(Comm);
  RCP<Time> timetot;
  timetot= TimeMonitor::getNewTimer("Side Set Connect total");
  RCP<Time> time1 = TimeMonitor::getNewTimer("One Side Set");
  RCP<TimeMonitor> tm_total = rcp(new TimeMonitor(*timetot));

  map<int,vector<int> > conn;
  map <int,vector<int> > mysides = sideset.GetSideSet();
  map<int,vector<int> >::iterator i_side;
  
  map<int,EXODUS::ElementBlock> ebs = GetElementBlocks();
  map<int,EXODUS::ElementBlock>::const_iterator i_ebs;
  
  // Range Vector for global eleID identification in SideSet
  vector<int> glob_eb_erange(1,0);
  int rangebreak = 0;
  // Also we once get all EBlocks and EConns to enable quick access
  vector<EXODUS::ElementBlock> eblocks;
  vector<map<int,vector<int> > > econns;
  for (i_ebs = ebs.begin(); i_ebs != ebs.end(); ++i_ebs ){
    rangebreak += i_ebs->second.GetNumEle();
    glob_eb_erange.push_back(rangebreak);
    eblocks.push_back(i_ebs->second);
    econns.push_back(i_ebs->second.GetEleConn());
  }
  
  // fill SideSet Connectivity
  int tetc=0,hexc=0,pyrc=0,wedgc=0;
  for (i_side = mysides.begin(); i_side != mysides.end(); ++i_side){
    RCP<TimeMonitor> tm1 = rcp(new TimeMonitor(*time1));
    int actele = i_side->second.at(0) -1;   //ExoIds start from 1, but we from 0
    int actface = i_side->second.at(1) -1;  //ExoIds start from 1, but we from 0
    // find actual EBlock where actele lies in
    int actebid;
    for(unsigned int i=0; i<glob_eb_erange.size(); ++i) 
      if (actele < glob_eb_erange[i]){ actebid = i-1; break;}
    EXODUS::ElementBlock::Shape actshape = eblocks[actebid].GetShape();
    
    // get act parent ele from actual Side
    int parent_ele_id = actele - glob_eb_erange[actebid]; 
    vector<int> parent_ele = econns[actebid].find(parent_ele_id)->second;
    
    // Face to ElementNode Map
    //// **** temporary hex map due to conflicts between side numbering exo<->baci
    switch (actshape){
    case ElementBlock::tet4: {actface = actface; tetc++; break;}
    case ElementBlock::hex8: {actface = HexSideNumberExoToBaci(actface); hexc++; break;}
    case ElementBlock::pyramid5: {actface = PyrSideNumberExoToBaci(actface); pyrc++; break;}
    case ElementBlock::wedge6: {actface = actface; wedgc++; break;}
    default: {cout << ShapeToString(actshape) << ":" << endl; dserror("Parent Element Type not supported");}
    }
    vector<int> childmap = DRT::UTILS::getEleNodeNumberingSurfaces(PreShapeToDrt(actshape))[actface];
    
    vector<int> child;
    if(checkoutside) child = OutsideOrientedSide(parent_ele,childmap);
    else for(unsigned int j=0; j<childmap.size(); ++j) child.push_back(parent_ele[childmap[j]]);
    
    // some checking
    if ((child.size() != 3) && (child.size() != 4)){
      PrintVec(cout,child);
      PrintVec(cout,childmap);
      PrintVec(cout,parent_ele);
      cout << ShapeToString(actshape) << ",Face: " << actface << ",childsize:" << child.size() << endl;
      dserror("Child Ele error");
    }
    
    // insert child into SideSet Connectivity
    conn.insert(pair<int,vector<int> >(i_side->first,child));
    tm1 = null;    
  }
  tm_total = null;
  //  TimeMonitor::summarize();
  cout << "...done" << endl;
  fflush(stdout);

  return conn;
}

vector<int> EXODUS::Mesh::OutsideOrientedSide(const vector<int> parentele, const vector<int> sidemap) const
{
  // first guess of child 
  vector<int> child;
  // set simplifies later inverse search
  set<int> childset;
  for(unsigned int j=0; j<sidemap.size(); ++j){
    child.push_back(parentele[sidemap[j]]);
    childset.insert(parentele[sidemap[j]]);
  }
  
  // set of parentele for later inverse search
  set<int> parentset;
  for(unsigned int i=0; i<parentele.size(); ++i)
    parentset.insert(parentele.at(i));
  
  // find parentele node not within side
  int insidenode;
  set<int>::iterator it;
  for(it = parentset.begin(); it != parentset.end(); ++it){
    if ( childset.find(*it) == childset.end() ){ insidenode = *it; break;}
  }
  
  // build normal at first side node
  vector<double> sidenormal = Normal(child.back(), child.front(), child.at(1));
  // build vector from first side node to inside element node
  vector<double> insidevec = NodeVec(child.front(), insidenode);
  
  // scalar-product
  double scp = sidenormal[0]*insidevec[0] + sidenormal[1]*insidevec[1] + sidenormal[2]*insidevec[2];
  
  vector<int> out_side;
  if(scp<0){
    vector<int> reversechild;
    vector<int>::reverse_iterator rit;
    for (rit=child.rbegin(); rit<child.rend(); ++rit) out_side.push_back(*rit); 
  } else out_side = child;
  
  return out_side;
}

vector<double> EXODUS::Mesh::Normal(const int head1,const int origin,const int head2) const
{
  vector<double> normal(3);
  vector<double> h1 = GetNodeExo(head1);
  vector<double> h2 = GetNodeExo(head2);
  vector<double> o  = GetNodeExo(origin);
  
  normal[0] =   ((h1[1]-o[1])*(h2[2]-o[2]) - (h1[2]-o[2])*(h2[1]-o[1]));
  normal[1] = - ((h1[0]-o[0])*(h2[2]-o[2]) - (h1[2]-o[2])*(h2[0]-o[0]));
  normal[2] =   ((h1[0]-o[0])*(h2[1]-o[1]) - (h1[1]-o[1])*(h2[0]-o[0]));
  
  double length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
  normal[0] = normal[0]/length;
  normal[1] = normal[1]/length;
  normal[2] = normal[2]/length;
  
  return normal;
}

vector<double> EXODUS::Mesh::NodeVec(const int tail, const int head) const
{
  vector<double> nv(3);
  vector<double> t = GetNodeExo(tail);
  vector<double> h = GetNodeExo(head);
  nv[0] = h[0] - t[0]; nv[1] = h[1] - t[1]; nv[2] = h[2] - t[2];
  double length = sqrt(nv[0]*nv[0] + nv[1]*nv[1] + nv[2]*nv[2]);
  nv[0] = nv[0]/length; nv[1] = nv[1]/length; nv[2] = nv[2]/length;
  return nv;
}


vector<EXODUS::ElementBlock> EXODUS::Mesh::SideSetToEBlocks(const EXODUS::SideSet& sideset, const map<int,vector<int> >& sideconn) const
{
  vector<ElementBlock> eblocks;
  //map<int,vector<int> > sideconn = sideset.GetSideSet();
  map<int,vector<int> >::const_iterator i_ele;
  map<int,vector<int> > quadconn;
  int quadcounter = 0;
  map<int,vector<int> > triconn;
  int tricounter = 0;
  for (i_ele = sideconn.begin(); i_ele != sideconn.end(); ++i_ele){
    int numnodes = i_ele->second.size();
    if (numnodes == 4){
      quadconn.insert(pair<int,vector<int> >(quadcounter,i_ele->second));
      quadcounter ++;
    }
    else if (numnodes == 3){
      triconn.insert(pair<int,vector<int> >(tricounter,i_ele->second));
      tricounter ++;
    }
    else dserror("Number of basenodes for conversion from SideSet to EBlock not supported");
  }
  if (quadcounter>0){
    std::ostringstream quadblockname;
    quadblockname << sideset.GetName() << "quad";
    EXODUS::ElementBlock neweblock(ElementBlock::quad4,quadconn,quadblockname.str());
    eblocks.push_back(neweblock);
  }
  if (tricounter>0){
    std::ostringstream triblockname;
    triblockname << sideset.GetName() << "tri";
    EXODUS::ElementBlock neweblock(ElementBlock::tri3,triconn,triblockname.str());
    eblocks.push_back(neweblock);
  }
  
  return eblocks;
}

EXODUS::NodeSet EXODUS::Mesh::SideSetToNodeSet(const EXODUS::SideSet& sideset, const map<int,vector<int> >& sideconn) const
{
  map<int,vector<int> >::const_iterator i_side;
  vector<int>::const_iterator i_node;
  set<int> nodes;
  for(i_side = sideconn.begin(); i_side != sideconn.end(); ++i_side)
    for(i_node = i_side->second.begin(); i_node != i_side->second.end(); ++i_node)
      nodes.insert(*i_node); //nodes.insert(i_side->second.at(i_node));
  std::ostringstream nodesetname;
  nodesetname << "nodes";//sideset.GetName() << "nodes";
  string propname = "";
  EXODUS::NodeSet nodeset(nodes,nodesetname.str(),propname);
 
  return nodeset;
}

set<int> EXODUS::Mesh::GetSideSetNodes(const EXODUS::SideSet& sideset, const map<int,vector<int> >& sideconn) const
{
  map<int,vector<int> >::const_iterator i_side;
  vector<int>::const_iterator i_node;
  set<int> nodes;
  for(i_side = sideconn.begin(); i_side != sideconn.end(); ++i_side)
    for(i_node = i_side->second.begin(); i_node != i_side->second.end(); ++i_node)
      nodes.insert(*i_node); //nodes.insert(i_side->second.at(i_node));
  return nodes;
}

/*----------------------------------------------------------------------*
 |  Write Mesh into exodus file (public)                                maf 01/08|
 *----------------------------------------------------------------------*/
void EXODUS::Mesh::WriteMesh(string newexofilename)
{
  cout << "Writing Mesh into: " << newexofilename.c_str() << endl;
  //string newexofile(newexofilename);
  //newexofile += ".exo";
  const char *newexofilechar;
  newexofilechar = newexofilename.c_str();
  
  int CPU_word_size, IO_word_size, exoid, error;
  CPU_word_size = sizeof(float); /* use float or double*/
  IO_word_size = 8; /* store variables as doubles */
  
  /* create EXODUS II file */
  exoid = ex_create (newexofilechar, /* filename path */
                     EX_CLOBBER,     /* create mode */
                     &CPU_word_size, /* CPU float word size in bytes */
                     &IO_word_size); /* I/O float word size in bytes */
  
  int num_elem_blk = GetNumElementBlocks();
  int num_node_sets = GetNumNodeSets();
  int num_side_sets = GetNumSideSets();
  int num_nodes = GetNumNodes();
  /* initialize file with parameters */
  const char* title = title_.c_str();
  error = ex_put_init (exoid, title, num_dim_, num_nodes, num_elem_,num_elem_blk, num_node_sets, num_side_sets);
  if (error!=0) dserror("error in exfile init");
  
  /* Write QA record based on original exofile */
  int num_qa_rec;
  char* qa_record[MAX_STR_LENGTH][4]; // should be MAX_QA_REC][4], but this is nowhere defined!;
  char *cdum;
  float fdum;
  /* read QA records */
  ex_inquire (exoid_, EX_INQ_QA, &num_qa_rec, &fdum, cdum);/* write QA records */
  for (int i=0; i<num_qa_rec; i++)
    for (int j=0; j<4; j++) qa_record[i][j] = (char *) calloc ((MAX_STR_LENGTH+1), sizeof(char));
    //for (int j=0; j<4; j++) qa_record[i][j] = new (char)[MAX_STR_LENGTH+1];
  error = ex_get_qa (exoid_, qa_record);
  error = ex_put_qa (exoid, num_qa_rec, qa_record);
  for (int i=0; i<num_qa_rec; i++)
    for (int j=0; j<4; j++)
      //delete [] qa_record[i][j];
      free(qa_record[i][j]);
  
  // Write coord names based on original exofile
  char *coord_names[3];
  for (int i=0; i<num_dim_; i++) coord_names[i] = (char *) calloc ((MAX_STR_LENGTH+1), sizeof(char));
  error = ex_get_coord_names (exoid_, coord_names);
  error = ex_put_coord_names (exoid, coord_names);
  for (int i=0; i<num_dim_; i++) free(coord_names[i]);
  
  // Write nodal coordinates
  vector<float> xc(num_nodes);
  vector<float> yc(num_nodes);
  vector<float> zc(num_nodes);
  map<int,vector<double> >::const_iterator it;
  map<int,vector<double> > nodes = GetNodes();
  for(it=nodes.begin(); it != nodes.end(); ++it){
    xc[it->first] = it->second[0];
    yc[it->first] = it->second[1];
    zc[it->first] = it->second[2];
  }
  error = ex_put_coord (exoid, &xc[0], &yc[0], &zc[0]);

  // Write NodeSets ************************************************************
  map<int,NodeSet>::const_iterator ins;
  const map<int,NodeSet> nss = GetNodeSets();
  for (ins=nss.begin(); ins != nss.end(); ++ins){
    const int nsID = ins->first + 1;   // exodus starts with 1
    const NodeSet ns = ins->second;
    const int num_nodes_in_set = ns.GetNumNodes();
    const string name = ns.GetName();
    const char* nsname = name.c_str();
    const string propname = ns.GetPropName();
    const set<int> nodes = ns.GetNodeSet();
    error = ex_put_node_set_param(exoid,              // of write file
                                  nsID,               // node set id
                                  num_nodes_in_set,
                                  0);                 // yet no distribution factors
    if (error!=0) dserror("error writing node set params");
    vector<int> nodelist(num_nodes_in_set);
    ns.FillNodelistArray(&nodelist[0]);
    error = ex_put_node_set(exoid,nsID,&nodelist[0]);
    if (error!=0) dserror("error writing node set");
    error = ex_put_name (exoid, EX_NODE_SET, nsID, nsname);
    if (error!=0) dserror("error writing node set name");
  }
  
  // Write ElementBlocks  ******************************************************
  map<int,ElementBlock>::const_iterator iebs;
  const map<int,ElementBlock> ebs = GetElementBlocks();
  for (iebs=ebs.begin(); iebs != ebs.end(); iebs++){
    const int blockID = iebs->first + 1;  // exodus starts with 1
    const ElementBlock eb = iebs->second;
    const ElementBlock::Shape shape = eb.GetShape();
    const string shapestring = ShapeToString(shape);
    const vector<int> exampleconn = eb.GetEleNodes(0); //iebs->first);
    const int num_nod_per_elem = exampleconn.size();
    const int numele = eb.GetNumEle();
    const char* elem_type = shapestring.c_str();
    error = ex_put_elem_block(exoid,                  //of write file
                              blockID,                //element block id 
                              elem_type,              //its name
                              numele,                 //number of element in block
                              num_nod_per_elem,       //num of nodes per ele
                              1);                     //num of attributes, not supported yet ->1
    if (error!=0) dserror("error writing element block");
    // Write Element Connectivity
    vector<int> conn(num_nod_per_elem*numele);
    eb.FillEconnArray(&conn[0]);
    error = ex_put_elem_conn(exoid,blockID,&conn[0]);
    if (error!=0) dserror("error writing element block conns");
    // write block name
    const string bname = eb.GetName();
    const char* blockname = bname.c_str();
    error = ex_put_name (exoid, EX_ELEM_BLOCK, blockID, blockname);
    if (error!=0) dserror("error writing element block name");
  }
  
  // Write SideSets ************************************************************
  map<int,SideSet>::const_iterator iss;
  const map<int,SideSet> sss = GetSideSets();
  for (iss=sss.begin(); iss != sss.end(); ++iss){
    const int ssID = iss->first + 1;   // exodus starts with 1
    const SideSet ss = iss->second;
    const int num_side_in_set = ss.GetNumSides();
    const string name = ss.GetName();
    const char* ssname = name.c_str();
    error = ex_put_side_set_param(exoid,              // of write file
                                  ssID,               // side set id
                                  num_side_in_set,
                                  0);                 // yet no distribution factors
    if (error!=0) dserror("error writing side set params");
    vector<int> side_set_elem_list(num_side_in_set);
    vector<int> side_set_side_list(num_side_in_set);
    ss.FillSideLists(&side_set_elem_list[0],&side_set_side_list[0]);
    error = ex_put_side_set(exoid,ssID,&side_set_elem_list[0],&side_set_side_list[0]);
    if (error!=0) dserror("error writing side set");
    error = ex_put_name (exoid, EX_SIDE_SET, ssID, ssname);
    if (error!=0) dserror("error writing sideset name");
  }
  
  // ***************************************************************************
  
  // close file
  error = ex_close (exoid);
  if (error!=0) dserror("error closing exodus file");
  cout << ".. finished" << endl;
}
/*----------------------------------------------------------------------*
 |  Add Element Block to mesh(public)                          maf 01/08|
 *----------------------------------------------------------------------*/
void EXODUS::Mesh::AddElementBlock(const EXODUS::ElementBlock eblock) const
{
  map<int,ElementBlock> eblocks = GetElementBlocks();
  eblocks.insert(std::pair<int,ElementBlock>(GetNumElementBlocks()+1,eblock));
}


EXODUS::ElementBlock::ElementBlock(ElementBlock::Shape Distype, map<int,vector<int> > &eleconn, string name)
{
  distype_ = Distype;
  eleconn_ = eleconn;
  name_ = name;
  return;
}
EXODUS::ElementBlock::~ElementBlock()
{
  return;
}

vector<int> EXODUS::ElementBlock::GetEleNodes(int i) const
{
  map<int,vector<int> >::const_iterator  it = eleconn_.find(i);
  return it->second;
}

int EXODUS::ElementBlock::GetEleNode(int ele, int node) const
{
  map<int,vector<int> >::const_iterator  it = eleconn_.find(ele);
  if (it == eleconn_.end()) dserror("Element not found");
  vector<int> elenodes = GetEleNodes(ele);
  return elenodes[node];
}

void EXODUS::ElementBlock::FillEconnArray(int *connarray) const
{
  const map<int,vector<int> >::const_iterator iele;
  int numele = eleconn_.size();
  for (int i = 0; i < numele; ++i) {
    vector<int> ele = GetEleNodes(i);
    int num_nod_per_elem = ele.size();
    for (int j = 0; j < num_nod_per_elem; ++j) {
      connarray[i*num_nod_per_elem + j] = GetEleNode(i,j);
    }
  }
}


void EXODUS::ElementBlock::Print(ostream& os, bool verbose) const
{
  os << "Element Block, named: " << name_.c_str() << endl
  << "of Shape: " << ShapeToString(distype_) << endl
  << "has " << GetNumEle() << " Elements" << endl;
  if (verbose){
    map<int,vector<int> >::const_iterator it;
    for (it=eleconn_.begin(); it != eleconn_.end(); it++){
      os << "Ele " << it->first << ": ";
      const vector<int> myconn = it->second; //GetEleNodes(int(it));
      for (int i=0; i < signed(myconn.size()); i++ ){
        os << myconn[i] << ",";
      }
      os << endl;
    }
  }
}


EXODUS::NodeSet::NodeSet(set<int> nodeids, string name, string propname)
{
  nodeids_ = nodeids;
  name_ = name;
  propname_ = propname;
}

EXODUS::NodeSet::~NodeSet()
{
  return;
}

void EXODUS::NodeSet::Print(ostream& os, bool verbose) const
{
  os << "Node Set, named: " << name_.c_str() << endl
  << "Property Name: " << propname_.c_str() << endl
  << "has " << GetNumNodes() << " Nodes" << endl;
  if (verbose){
    os << "Contains Nodes:" << endl;
    set<int>::iterator it;
    for (it=nodeids_.begin(); it != nodeids_.end(); it++) os << *it << ",";
    os << endl;
  }
}

void EXODUS::NodeSet::FillNodelistArray(int* nodelist) const
{
  set<int> nlist = GetNodeSet();
  set<int>::iterator it;
  int i=0;
  for (it=nlist.begin(); it != nlist.end(); ++it){
    nodelist[i] = (*it);
    ++i;
  }
}


EXODUS::SideSet::SideSet(map<int,vector<int> >sides, string name)
{
  sides_ = sides;
  name_ = name;
}

void EXODUS::SideSet::FillSideLists(int* elemlist, int* sidelist) const
{
  map<int,vector<int> > sides = GetSideSet();
  map<int,vector<int> >::iterator it;
  int i=0;
  for (it=sides.begin(); it != sides.end(); ++it){
    elemlist[i] = it->second[0];
    sidelist[i] = it->second[1];
    ++i;
  }
}

void EXODUS::SideSet::Print(ostream& os, bool verbose) const{
  os << "SideSet, named: " << name_.c_str() << endl
  << "has " << GetNumSides() << " Sides" << endl;
  if (verbose){
    map<int,vector<int> >::const_iterator it;
    for (it=sides_.begin(); it != sides_.end(); it++){
      os << "Side " << it->first << ": ";
      os << "Ele: " << it->second.at(0) << ", Side: " << it->second.at(1) << endl;
    }
  }
}

EXODUS::SideSet::~SideSet()
{
  return;
}

void EXODUS::PrintMap(ostream& os,const map<int,vector<int> > mymap)
{
  map<int,vector<int> >::const_iterator iter;
  for(iter = mymap.begin(); iter != mymap.end(); ++iter)
  {
      os << iter->first << ": ";
      vector<int> actvec = iter->second;
      vector<int>::iterator i;
      for (i=actvec.begin(); i<actvec.end(); ++i) {
        os << *i << ",";
      }
      os << endl;
  }
}

void EXODUS::PrintMap(ostream& os,const map<int,set<int> > mymap)
{
  map<int,set<int> >::const_iterator iter;
  for(iter = mymap.begin(); iter != mymap.end(); ++iter)
  {
      os << iter->first << ": ";
      set<int> actset = iter->second;
      set<int>::iterator i;
      for (i=actset.begin(); i != actset.end(); ++i) {
        os << *i << ",";
      }
      os << endl;
  }
}

void EXODUS::PrintMap(ostream& os,const map<int,vector<double> > mymap)
{
  map<int,vector<double> >::const_iterator iter;
  for(iter = mymap.begin(); iter != mymap.end(); ++iter)
  {
      os << iter->first << ": ";
      vector<double> actvec = iter->second;
      vector<double>::iterator i;
      for (i=actvec.begin(); i<actvec.end(); ++i) {
        os << *i << ",";
      }
      os << endl;
  }
}

void EXODUS::PrintVec(ostream& os, const vector<int> actvec)
{
  vector<int>::const_iterator i;
  for (i=actvec.begin(); i<actvec.end(); ++i) {
    os << *i << ",";
  }
  os << endl;
}

void EXODUS::PrintVec(ostream& os, const vector<double> actvec)
{
  vector<double>::const_iterator i;
  for (i=actvec.begin(); i<actvec.end(); ++i) {
    os << *i << ",";
  }
  os << endl;
}

void EXODUS::PrintSet(ostream& os, const set<int> actset)
{
  set<int>::iterator i;
  for (i=actset.begin(); i != actset.end(); ++i) {
    os << *i << ",";
  }
  os << endl;
  
}

int EXODUS::HexSideNumberExoToBaci(const int exoface)
{
  const int map[6] = {1,2,3,4,0,5};
  return map[exoface];
}

int EXODUS::PyrSideNumberExoToBaci(const int exoface)
{
  const int map[5] = {1,2,3,4,0};
  return map[exoface];
}


#endif
