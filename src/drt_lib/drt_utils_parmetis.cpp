
#ifdef CCADISCRET

#if defined(PARALLEL) && defined(PARMETIS)

#include "drt_inputreader.H"
#include "drt_utils.H"
#include "../linalg/linalg_utils.H"
#include "standardtypes_cpp.H"

#include <Epetra_Time.h>
#include <Isorropia_EpetraPartitioner.hpp>

#include <iterator>

namespace DRT {
namespace UTILS {

void PackLocalConnectivity(map<int,set<int> >& lcon, vector<char>& sblock);
void UnpackLocalConnectivity(map<int,set<int> >& lcon, vector<char>& rblock);

}
}

typedef int idxtype;
extern "C"
{
  void ParMETIS_V3_PartKway(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
                            idxtype *adjwgt, int *wgtflag, int *numflag, int *ncon, int *nparts,
                            float *tpwgts, float *ubvec, int *options, int *edgecut, idxtype *part,
                            MPI_Comm *comm);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::PackLocalConnectivity(
  map<int,set<int> > & lcon,
  vector<char>       & sblock
  )
{
  map<int,set<int> >::iterator gidinlcon;
  set<int>::iterator           adjacentgid;

  sblock.clear();
  // size (number of nodes we have a connectivity for)
  int size=lcon.size();
  DRT::ParObject::AddtoPack(sblock,size);

  for(gidinlcon=lcon.begin();gidinlcon!=lcon.end();++gidinlcon)
  {
    // add node gid we store the connectivity for
    DRT::ParObject::AddtoPack<int>(sblock,(int)(gidinlcon->first));

    // add number of nodes adjacent to this one
    DRT::ParObject::AddtoPack<int>(sblock,(int)(gidinlcon->second).size());

    // add list of neighbours to this node
    for(adjacentgid =(gidinlcon->second).begin();
        adjacentgid!=(gidinlcon->second).end();
        ++adjacentgid)
    {
      DRT::ParObject::AddtoPack(sblock,*adjacentgid);
    }
  }

  lcon.clear();

  return;
} // end Pack_lcon


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::UnpackLocalConnectivity(
  map<int,set<int> > & lcon  ,
  vector<char>       & rblock
  )
{
  if(rblock.empty())
  {
    dserror("trying to extract information from an empty receive block\n");
  }

  lcon.clear();

  // position to extract
  vector<char>::size_type position = 0;

  // extract size (number of nodes we have a connectivity for)
  int size=0;
  DRT::ParObject::ExtractfromPack<int>(position,rblock,size);

  for(int i=0;i<size;++i)
  {
    // extract node gid we store the connectivity for
    int gid=-1;
    DRT::ParObject::ExtractfromPack<int>(position,rblock,gid);

    if(gid<0)
    {
      dserror("Unable to unpack a proper gid");
    }

    // extract number of adjacent nodes
    int numnb=0;
    DRT::ParObject::ExtractfromPack<int>(position,rblock,numnb);

    if(numnb<1)
    {
      dserror("Everybody should have at least one unpackable neighbour (%d given)",numnb);
    }

    set<int> neighbourset;

    // extract all adjacent nodes and feed them into the set
    for(int j=0;j<numnb;++j)
    {
      int nbgid=0;
      DRT::ParObject::ExtractfromPack(position,rblock,nbgid);
      neighbourset.insert(nbgid);
    }

    // add this node connectivity to local connectivity map
    lcon.insert(pair<int,set<int> >(gid,neighbourset));
  }

  // trash receive block
  rblock.clear();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::PartUsingParMetis(RCP<DRT::Discretization> dis,
                                   RCP<Epetra_Map> roweles,
                                   RCP<Epetra_Map>& rownodes,
                                   RCP<Epetra_Map>& colnodes,
                                   vector<int>& nids,
                                   int nblock,
                                   int ntarget,
                                   RCP<Epetra_Comm> comm,
                                   Epetra_Time& time,
                                   bool outflag)
{
    const int myrank = comm->MyPID();
    const int numproc = comm->NumProc();
    Epetra_Time timer(dis->Comm());
    double t1 = timer.ElapsedTime();
    if (!myrank && outflag)
    {                  
      printf("parmetis:\n");
      fflush(stdout);
    }
    // construct reverse lookup from gids to vertex ids. Again,
    // this is a global, fully redundant map!
    // We need this lookup for the construction of the parmetis
    // adjacency array
    //
    //
    //                         gidtoidx
    //                      ------------->
    //                gid_i                vertexid
    //                      <-------------
    //                           nids
    //
    map<int,int> gidtoidx;
    int numnodes = static_cast<int>(nids.size());
    for (int i=0;i<numnodes;++i)
    {
      gidtoidx[nids[i]]=i;
    }


#if 0 // don't need this anymore
    if (myrank==0)
    {
      // Should definitely not modify the time object that came in from outside!
      //time.ResetStartTime();
      if (outflag)
      {
        cout << "Build initial node graph, call PARMETIS  in....\n";
        fflush(stdout);
      }
    }
#endif      

    // --------------------------------------------------
    // - define distributed nodal row map

    // vertices and nodes --- the parmetis call will be based on a
    // consecutive numbering of vertices. We will use a lookup
    // for global node ids which will map the gid to its vertex id.
    // We need this for the construction of the adjacency array.
    //
    //         rownode gid   | (row)vertex id            vtxdist
    //                       |
    //      -----------------+----------------------------+---+
    //   +-        gid0      |      0.....................| 0 |
    //   |         gid1      |      1                     +---+
    //  p|         gid2      |      2                       ^
    //  r|         gid3      |      3                       |
    //  o|         gid4      |      4                       |
    //  c|         gid5      |      5                       |
    //  0|         gid6      |      6                       |
    //   |         gid7      |      7                       |
    //   |         gid8      |      8                       |
    //   +-        gid9      |      9                       v
    //      -----------------+----------------------------+---+
    //  .+-       gid10      |     10.....................| 10|
    //  p|        gid11      |     11                     +---+
    //  r|        gid12      |     12                       ^
    //  o|        gid13      |     13                       |
    //  c|        gid14      |     14                       |
    //  1|        gid15      |     15                       |
    //   +-       gid16      |     16                       v
    //      -----------------+----------------------------+---+
    //  p+-       gid17      |     17.....................| 17|
    //  r|        gid18      |     18                     +---+
    //  o|        gid19      |     19                       ^
    //  c|        gid20      |     20                       |
    //  2|        gid21      |     21                       |
    //   +-       gid22      |     22                       v
    //      ----------------------------------------------+---+
    //      ..............................................| 23|
    //                                                    +---+
    //

    // nblock is not neccessarily equal to number of processors
    nblock = numproc;
    // number of node id junks
    int nbsize = numnodes/nblock;

    // create a simple (pseudo linear) map for nodes
    int mynsize = nbsize;
    if (myrank==numproc-1)
      mynsize = numnodes-(numproc-1)*nbsize;

    // construct the initial linear node rowmap
    RCP<Epetra_Map> lin_noderowmap = rcp(new Epetra_Map(-1,mynsize,&nids[myrank*nbsize],0,*comm));

    // remember my vertex distribution for the later parmetis call
    vector<int> vtxdist(numproc+1);
    for(int np=0;np<numproc;++np)
    {
      vtxdist[np]=np*nbsize;
    }
    vtxdist[numproc]=numnodes;

    // --------------------------------------------------
    // - determine adjacency array (i.e. the infos for the node graph)
    //   using the nodal row distribution and a round robin communication
    //   of element connectivity information

    // this is a set of gids of all nodes for which we have a
    // connectivity information on this proc
    set<int> procnodes;

    // loop all eles on this proc and determine all gids for
    // which we have some connectivity information
    for(int lid=0;lid<roweles->NumMyElements();++lid)
    {
      int gid=roweles->GID(lid);

      DRT::Element* ele=dis->gElement(gid);

      // get the node ids of this element
      const int  numnode = ele->NumNode();
      const int* nodeids = ele->NodeIds();

      // all node gids of this element are inserted into a set of
      // node ids
      copy(nodeids, nodeids+numnode, inserter(procnodes,procnodes.begin()));
    }

    // ---------------------
    // build a processor local node connectivity


    //     node gid contained in one of the elements
    //                    on this proc
    //                         |
    //                         | lcon
    //                         |
    //                         v
    //    set of all gids of adjacent nodes on this proc
    //
    map<int,set<int> > lcon;

    // construct empty local map
    set<int>::iterator procnode;

    for(procnode=procnodes.begin();procnode!=procnodes.end();++procnode)
    {
      lcon.insert(pair<int, set<int> >(*procnode,set<int> ()));
    }

    // loop all eles on this proc and construct the local
    // connectivity information
    map<int,set<int> >::iterator gidinlcon;

    for(int lid=0;lid<roweles->NumMyElements();++lid)
    {
      int gid=roweles->GID(lid);

      DRT::Element* ele=dis->gElement(gid);

      // get the node ids of this element
      const int  numnode = ele->NumNode();
      const int* nodeids = ele->NodeIds();

      // loop nodeids
      for(int rr=0;rr<numnode;++rr)
      {
        // find gid of rr in map
        gidinlcon=lcon.find(nodeids[rr]);

        if(gidinlcon==lcon.end())
        {
          dserror("GID %d should be already contained in the map",nodeids[rr]);
        }

        // add nodeids to local connectivity
        copy(nodeids,
             nodeids+numnode,
             inserter((gidinlcon->second),(gidinlcon->second).begin()));
      }
    }

    //-------------------------
    // (incomplete) round robin loop to build complete node
    //  connectivity for local nodes across all processors

    //       node gid of one row node on this proc
    //                         |
    //                         | gcon
    //                         |
    //                         v
    //    set of all gids of adjacent nodes on all procs

    // prepare empty map
    map<int,set<int> > gcon;
    for(int j=0;j<lin_noderowmap->NumMyElements();++j)
    {
      int gid=lin_noderowmap->GID(j);

      gcon.insert(pair<int, set<int> >(gid,set<int> ()));
    }

    {
#ifdef PARALLEL
      // create an exporter for point to point communication
      DRT::Exporter exporter(dis->Comm());

      // necessary variables
      MPI_Request request;

      int         tag    =-1;
      int         frompid=-1;
      int         topid  =-1;
      int         length =-1;

      // define send and receive blocks
      vector<char> sblock;
      vector<char> rblock;

#endif

      for (int np=0;np<numproc;++np)
      {
        // in the first step, we cannot receive anything
        if(np >0)
        {
#ifdef PARALLEL
          //----------------------
          // Unpack local graph from the receive block from the
          // last proc

          // make sure that you do not think you received something if
          // you didn't
          if(rblock.empty()==false)
          {
            dserror("rblock not empty");
          }

          // receive from predecessor
          frompid=(myrank+numproc-1)%numproc;
          exporter.ReceiveAny(frompid,tag,rblock,length);

          if(tag!=(myrank+numproc-1)%numproc)
          {
            dserror("received wrong message (ReceiveAny)");
          }

          exporter.Wait(request);

          // for safety
          exporter.Comm().Barrier();
#endif

          // Unpack received block
          UnpackLocalConnectivity(lcon,rblock);

        }

        // -----------------------
        // add local connectivity passing by to global
        // connectivity on this proc

        // loop this procs global connectivity
        map<int,set<int> >::iterator gidingcon;

        for(gidingcon=gcon.begin();gidingcon!=gcon.end();++gidingcon)
        {

          // search in (other) procs local connectivity
          gidinlcon=lcon.find(gidingcon->first);

          if(gidinlcon!=lcon.end())
          {
            // in this case we do have some additional
            // connectivity info from the proc owning lcon

            copy((gidinlcon->second).begin(),
                 (gidinlcon->second).end(),
                 inserter((gidingcon->second),(gidingcon->second).begin()));
          }
        }

        // in the last step, we keep everything on this proc
        if(np < numproc-1)
        {
          //-------------------------
          // Pack local graph into block to send
          PackLocalConnectivity(lcon,sblock);

#ifdef PARALLEL
          // Send block to next proc.

          tag    =myrank;
          frompid=myrank;
          topid  =(myrank+1)%numproc;

          exporter.ISend(frompid,topid,
                         &(sblock[0]),sblock.size(),
                         tag,request);
#endif
        }
      }
    }

    lcon.clear();

    // --------------------------------------------------
    // - do partitioning using parmetis

    //    gid(i)           gid(i+1)                        index gids
    //      |^                 |^
    //  nids||             nids||
    //      ||gidtoidxd        ||gidtoidx
    //      ||                 ||
    //      v|                 v|
    //   nodegid(i)       nodegid(i+1)                      node gids
    //      ^                 ^
    //      |                 |
    //      | lin_noderowmap  | lin_noderowmap
    //      |                 |
    //      v                 v
    //      i                i+1                        local equivalent indices
    //      |                 |
    //      | xadj            | xadj
    //      |                 |
    //      v                 v
    //     +-+-+-+-+-+-+-+-+-+-+                -+-+-+
    //     | | | | | | | | | | | ............... | | |      adjncy
    //     +-+-+-+-+-+-+-+-+-+-+                -+-+-+
    //
    //     |     gid(i)'s    |   gid(i+1)'s
    //     |    neighbours   |   neighbours           (numbered by global equivalent
    //                                                 indices, mapping by nids vector
    //                                                 to global ids)
    //

    // xadj points from vertex index i to the index of the
    // first adjacent vertex.
    vector<int> xadj(lin_noderowmap->NumMyElements()+1);

    // a list of adjacent nodes, adressed using xadj
    vector<int> adjncy;

    int count=0;
    xadj[0] = 0;

    for (int idx=0;idx<lin_noderowmap->NumMyElements();++idx)
    {
      // get global node id of rownode
      int  growid =lin_noderowmap->GID(idx);

      map<int,set<int> >::iterator gidingcon;
      set<int>::iterator           nbgid;

      gidingcon=gcon.find(growid);

      if(gidingcon!=gcon.end())
      {

        for(nbgid=(gidingcon->second).begin();nbgid!=(gidingcon->second).end();++nbgid)
        {

          if(*nbgid!=growid)
          {
            // reverse lookup to determine neighbours index
            int nbidx=gidtoidx[*nbgid];

            adjncy.push_back(nbidx);
            ++count;
          }
        }
      }
      else
      {
        dserror("GID %d not found in global connectivity during setup of adjncy",growid);
      }
      xadj[idx+1] = count;
    }

#if 0
    // define a vector of nodeweights
    // at the moment, we use a constant weight distribution at this point
    vector<int> vwgt(lin_noderowmap->NumMyElements(),1);

    for (int i=0; i<lin_noderowmap->NumMyElements(); ++i)
    {
      vwgt[i] = 1;
    }
#endif

    /*
      This is used to indicate if the graph is weighted.
      wgtflag can take one of four values:
      0  No weights (vwgt and adjwgt are both NULL).
      1  Weights on the edges only (vwgt is NULL).
      2  Weights on the vertices only (adjwgt is NULL).
      3  Weights on both the vertices and edges.
    */
    int wgtflag = 0; // int wgtflag=2;
    /*
      This is used to indicate the numbering scheme that
      is used for the vtxdist, xadj, adjncy, and part
      arrays. numflag can take one of two values:
      0 C-style numbering that starts from 0.
      1 Fortran-style numbering that starts from 1.
    */
    int numflag=0;
    /*
      This is used to specify the number of weights that
      each vertex has. It is also the number of balance
      constraints that must be satisfied.
    */
    int ncon=1;
    /*
      This is used to specify the number of sub-domains
      that are desired. Note that the number of sub-domains
      is independent of the number of processors that call
      this routine.
    */
    int npart=ntarget;
    /*
      This is an array of integers that is used to pass
      additional parameters for the routine. If options[0]=0,
      then the default values are used. If options[0]=1,
      then the remaining two elements of options are
      interpreted as follows:
      options[1]     This specifies the level of information
                     to be returned during the execution of
                     the algorithm. Timing information can be
                     obtained by setting this to 1. Additional
                     options for this parameter can be obtained
                     by looking at the the file defs.h in the
                     ParMETIS-Lib directory. The numerical values
                     there should be added to obtain the correct
                     value. The default value is 0.
      options[2]     This is the random number seed for the routine.
                     The default value is 15.
    */
    int options[3] = { 0,0,15 };
    /*
      Upon successful completion, the number of edges that are cut
      by the partitioning is written to this parameter.
    */
    int edgecut=0;
    /*
      This is an array of size equal to the number of locally-stored
      vertices. Upon successful completion the partition vector of
      the locally-stored vertices is written to this array.
      Note that this vector will not be redistributed by metis, it
      will just contain the information how to redistribute.
    */
    vector<int> part(lin_noderowmap->NumMyElements());
    /*
      An array of size ncon that is used to specify the imbalance
      tolerance for each vertex weight, with 1 being perfect balance
      and nparts being perfect imbalance. A value of 1.05 for each
      of the ncon weights is recommended.
    */
    float ubvec =1.05;
    /*
      An array of size ncon x nparts that is used to specify
      the fraction of vertex weight that should be distributed
      to each sub-domain for each balance constraint. If all
      of the sub-domains are to be of the same size for every
      vertex weight, then each of the ncon x nparts elements
      should be set to a value of 1/nparts. If ncon is greater
      than one, the target sub-domain weights for each sub-domain
      are stored contiguously (similar to the vwgt array). Note
      that the sum of all of the tpwgts for a give vertex weight
      should be one.
    */
    vector<float> tpwgts(npart,1.0/(double)npart);
    /*
      This is a pointer to the MPI communicator of the processes that
      call PARMETIS.
    */
    MPI_Comm mpicomm=(dynamic_cast<const Epetra_MpiComm*>(&(dis->Comm())))->Comm();

    double t2 = timer.ElapsedTime();
    if (!myrank && outflag)
    {                  
      printf("parmetis setup    %10.5e secs\n",t2-t1);
      fflush(stdout);
    }

    ParMETIS_V3_PartKway(
      &(vtxdist[0]),
      &(xadj   [0]),
      &(adjncy [0]),
      NULL, // &(vwgt   [0]), no weights, because they are 1 anyway
      NULL         ,
      &wgtflag     ,
      &numflag     ,
      &ncon        ,
      &npart       ,
      &(tpwgts[0]) ,
      &ubvec       ,
      &(options[0]),
      &edgecut     ,
      &(part[0])   ,
      &mpicomm);

    double t3 = timer.ElapsedTime();
    if (!myrank && outflag)
    {                  
      printf("parmetis call     %10.5e secs\n",t3-t2);
      fflush(stdout);
    }

    // for each vertex j, the proc number to which this vertex
    // belongs was written to part[j]. Note that PARMETIS does
    // not redistribute the graph according to the new partitioning,
    // it simply computes the partitioning and writes it to
    // the part array.

    // construct epetra graph on linear noderowmap
    Teuchos::RCP<Epetra_CrsGraph> graph = rcp(new Epetra_CrsGraph(Copy,*lin_noderowmap,108,false));

    for (map<int,set<int> >::iterator gid=gcon.begin();
         gid!=gcon.end();
         ++gid
      )
    {
      set<int>& rowset = gid->second;
      vector<int> row;
      row.reserve(rowset.size());
      row.assign(rowset.begin(),rowset.end());
      rowset.clear();

      int err = graph->InsertGlobalIndices(gid->first,row.size(),&row[0]);
      if (err<0) dserror("graph->InsertGlobalIndices returned %d",err);
    }

    // trash the stl based graph
    gcon.clear();

    // fill graph and optimize storage
    graph->FillComplete();
    graph->OptimizeStorage();

    // redundant, global vectors to broadcast partition results
    vector<int> lglobalpart(numnodes,0);
    vector<int> globalpart (numnodes,0);

    // insert proc ids to distribute to into global vector
    for (unsigned i=0;i<part.size();++i)
    {
      lglobalpart[gidtoidx[lin_noderowmap->GID(i)]]=part[i];
    }

    // finally broadcast partition results
    comm->SumAll(&(lglobalpart[0]),&(globalpart[0]),numnodes);

    lglobalpart.clear();

    // --------------------------------------------------
    // - build final nodal row map

    // resize part array to new size
    count=0;
    for (unsigned i=0; i<globalpart.size(); ++i)
    {
      if (globalpart[i]==myrank)
      {
        ++count;
      }
    }
    part.resize(count);

    // fill it with the new distribution from the partitioning results
    count=0;
    for (unsigned i=0; i<globalpart.size(); ++i)
    {
      if (globalpart[i]==myrank)
      {
        part[count] = nids[i];
        ++count;
      }
    }

    nids.clear();

    // create map with new layout
    Epetra_Map newmap(globalpart.size(),count,&part[0],0,*comm);

    // create the output graph and export to it
    RCP<Epetra_CrsGraph> outgraph =
      rcp(new Epetra_CrsGraph(Copy,newmap,108,false));
    Epetra_Export exporter(graph->RowMap(),newmap);
    int err = outgraph->Export(*graph,exporter,Add);
    if (err<0) dserror("Graph export returned err=%d",err);

    //trash old graph
    graph=null;

    // call fill complete and optimize storage
    outgraph->FillComplete();
    outgraph->OptimizeStorage();

    // replace rownodes, colnodes with row and column maps from the graph
    // do stupid conversion from Epetra_BlockMap to Epetra_Map
    const Epetra_BlockMap& brow = outgraph->RowMap();
    const Epetra_BlockMap& bcol = outgraph->ColMap();
    rownodes = rcp(new Epetra_Map(brow.NumGlobalElements(),
                                   brow.NumMyElements(),
                                   brow.MyGlobalElements(),
                                   0,
                                   *comm));
    colnodes = rcp(new Epetra_Map(bcol.NumGlobalElements(),
                                   bcol.NumMyElements(),
                                   bcol.MyGlobalElements(),
                                   0,
                                   *comm));

    double t4 = timer.ElapsedTime();
    if (!myrank && outflag)
    {                  
      printf("parmetis cleanup  %10.5e secs\n",t4-t3);
      fflush(stdout);
    }

}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::PartUsingParMetis(RCP<DRT::Discretization> dis,
                                   RCP<Epetra_Map> roweles,
                                   RCP<Epetra_Map>& rownodes,
                                   RCP<Epetra_Map>& colnodes,
                                   RCP<Epetra_Comm> comm,
                                   bool outflag)
{
  const int myrank = comm->MyPID();
  const int numproc = comm->NumProc();
  Epetra_Time timer(dis->Comm());
  double t1 = timer.ElapsedTime();
  if (!myrank && outflag)
  {                  
    printf("parmetis:\n");
    fflush(stdout);
  }

  // create a set of all nodes that I have
  set<int> mynodes;
  for (int lid=0;lid<roweles->NumMyElements();++lid)
  {
    DRT::Element* ele=dis->gElement(roweles->GID(lid));
    const int  numnode = ele->NumNode();
    const int* nodeids = ele->NodeIds();
    copy(nodeids,nodeids+numnode,inserter(mynodes,mynodes.begin()));
  }
  
  // build a unique row map from the overlapping sets
  for (int proc=0; proc<numproc; ++proc)
  {
    int size = 0;
    vector<int> recvnodes;
    if (proc==myrank)
    {
      recvnodes.clear();
      set<int>::iterator fool;
      for (fool = mynodes.begin(); fool != mynodes.end(); ++fool)
        recvnodes.push_back(*fool);
      size=(int)recvnodes.size();
    }
    comm->Broadcast(&size,1,proc);
    if (proc!=myrank) recvnodes.resize(size);
    comm->Broadcast(&recvnodes[0],size,proc);
    if (proc!=myrank)
    {
      for (int i=0; i<size; ++i)
      {
        set<int>::iterator fool = mynodes.find(recvnodes[i]);
        if (fool==mynodes.end()) continue;
        else                     mynodes.erase(fool);
      }
    }
    comm->Barrier();
  } 


  // copy the set to a vector
  {
    vector<int> nodes;
    set<int>::iterator fool;
    for (fool = mynodes.begin(); fool != mynodes.end(); ++fool)
      nodes.push_back(*fool);
    mynodes.clear();
    // create a non-overlapping row map
    rownodes = rcp(new Epetra_Map(-1,(int)nodes.size(),&nodes[0],0,*comm));
  }

  
  // start building the graph object
  map<int,set<int> > locals;
  map<int,set<int> > remotes;
  for (int lid=0;lid<roweles->NumMyElements();++lid)
  {
    DRT::Element* ele=dis->gElement(roweles->GID(lid));
    const int  numnode = ele->NumNode();
    const int* nodeids = ele->NodeIds();
    for (int i=0; i<numnode; ++i)
    {
      const int lid = rownodes->LID(nodeids[i]); // am I owner of this gid?
      map<int,set<int> >* insertmap = NULL;
      if (lid != -1) insertmap = &locals;
      else           insertmap = &remotes;
      // see whether we already have an entry for nodeids[i]
      map<int,set<int> >::iterator fool = (*insertmap).find(nodeids[i]);
      if (fool==(*insertmap).end()) // no entry in that row yet
      {
        set<int> tmp;
        copy(nodeids,nodeids+numnode,inserter(tmp,tmp.begin()));
        (*insertmap)[nodeids[i]] = tmp;
      }
      else
      {
        set<int>& imap = fool->second;
        copy(nodeids,nodeids+numnode,inserter(imap,imap.begin()));
      }
    } // for (int i=0; i<numnode; ++i)
  } // for (int lid=0;lid<roweles->NumMyElements();++lid)
  // run through locals and remotes to find the max bandwith
  int maxband = 0;
  {
    int smaxband = 0;
    map<int,set<int> >::iterator fool;
    for (fool = locals.begin(); fool != locals.end(); ++fool)
      if (smaxband < (int)fool->second.size()) smaxband = (int)fool->second.size();
    for (fool = remotes.begin(); fool != remotes.end(); ++fool)
      if (smaxband < (int)fool->second.size()) smaxband = (int)fool->second.size();
    comm->MaxAll(&smaxband,&maxband,1);
  }
  if (!myrank && outflag)
  {                  
    printf("parmetis max nodal bandwith %d\n",maxband);
    fflush(stdout);
  }
  
  Teuchos::RCP<Epetra_CrsGraph> graph = rcp(new Epetra_CrsGraph(Copy,*rownodes,maxband,false));


  // fill all local entries into the graph
  {
    map<int,set<int> >::iterator fool = locals.begin();
    for (; fool != locals.end(); ++fool)
    {
      const int grid = fool->first;
      vector<int> cols(0,0);
      set<int>::iterator setfool = fool->second.begin();
      for (; setfool != fool->second.end(); ++setfool) cols.push_back(*setfool);
      int err = graph->InsertGlobalIndices(grid,(int)cols.size(),&cols[0]);
      if (err<0) dserror("Epetra_CrsGraph::InsertGlobalIndices returned %d for global row %d",err,grid);
    }
    locals.clear();
  }
  

  // now we need to communicate and add the remote entries
  for (int proc=0; proc<numproc; ++proc)
  {
    vector<int> recvnodes;
    int size = 0;
    if (proc==myrank)
    {
      recvnodes.clear();
      map<int,set<int> >::iterator mapfool = remotes.begin();
      for (; mapfool != remotes.end(); ++mapfool)
      {
        recvnodes.push_back((int)mapfool->second.size()+1); // length of this entry
        recvnodes.push_back(mapfool->first); // global row id
        set<int>::iterator fool = mapfool->second.begin();
        for (; fool!=mapfool->second.end(); ++fool) // global col ids
          recvnodes.push_back(*fool);
      }
      size = (int)recvnodes.size();
    }
    comm->Broadcast(&size,1,proc);
    if (proc!=myrank) recvnodes.resize(size);
    comm->Broadcast(&recvnodes[0],size,proc);
    if (proc!=myrank && size)
    {
      int* ptr = &recvnodes[0];
      while (ptr < &recvnodes[size-1])
      {
        int num  = *ptr; 
        int grid = *(ptr+1);
        // see whether I have grid in my row map
        if (rownodes->LID(grid) != -1) // I have it, put stuff in my graph
        {
          int err = graph->InsertGlobalIndices(grid,num-1,(ptr+2));
          if (err<0) dserror("Epetra_CrsGraph::InsertGlobalIndices returned %d",err);
          ptr += (num+1);
        }
        else // I don't have it so I don't care for entries of this row, goto next row
          ptr += (num+1);
      }
    }
    comm->Barrier();
  } //  for (int proc=0; proc<numproc; ++proc)
  remotes.clear();
  
  // finish graph
  graph->FillComplete();
  graph->OptimizeStorage();

  // prepare parmetis input and call parmetis to partition the graph
  vector<int> vtxdist(numproc+1,0);
  vector<int> xadj(graph->RowMap().NumMyElements()+1);
  vector<int> adjncy(0,0);
  {
    Epetra_IntVector ridtoidx(graph->RowMap(),false);
    Epetra_IntVector cidtoidx(graph->ColMap(),false);
    
    xadj[0] = 0;
    
    // create vtxdist
    vector<int> svtxdist(numproc+1,0);
    svtxdist[myrank+1] = graph->RowMap().NumMyElements();
    comm->SumAll(&svtxdist[0],&vtxdist[0],numproc+1);
    for (int i=0; i<numproc; ++i) vtxdist[i+1] += vtxdist[i];

    // create a Epetra_IntVector in rowmap with new numbering starting from zero and being linear
    for (int i=0; i<ridtoidx.Map().NumMyElements(); ++i) ridtoidx[i] = i + vtxdist[myrank];
    Epetra_Import importer(graph->ColMap(),graph->RowMap());
    int err = cidtoidx.Import(ridtoidx,importer,Insert);
    if (err) dserror("Import using importer returned %d",err);
    
    // create xadj and adjncy;
    for (int lrid=0; lrid<graph->RowMap().NumMyElements(); ++lrid)
    {
      int metisrid = ridtoidx[lrid];
      int  numindices;
      int* indices;
      int err = graph->ExtractMyRowView(lrid,numindices,indices);
      if (err) dserror("Epetra_CrsGraph::GetMyRowView returned %d",err);
      // switch column indices to parmetis indices and add to vector of columns
      vector<int> metisindices(0);
      for (int i=0; i<numindices; ++i) 
      {
        if (metisrid == cidtoidx[indices[i]]) continue; // must not contain main diagonal entry
        metisindices.push_back(cidtoidx[indices[i]]);
      }
      //sort(metisindices.begin(),metisindices.end());
      // beginning of new row
      xadj[lrid+1] = xadj[lrid] + (int)metisindices.size();
      for (int i=0; i<(int)metisindices.size(); ++i) 
        adjncy.push_back(metisindices[i]);
    }
  }
  
  double t2 = timer.ElapsedTime();
  if (!myrank && outflag)
  {                  
    printf("parmetis setup    %10.5e secs\n",t2-t1);
    fflush(stdout);
  }

  vector<int> part(graph->RowMap().NumMyElements()); // output
  {
    int wgtflag = 0;             // graph is not weighted
    int numflag = 0;             // numbering start from zero
    int ncon = 1;                // number of weights on each node
    int npart = numproc;         // number of partitions desired
    int options[3] = { 0,0,15 }; // use default metis parameters
    int edgecut = 0;             // output, number of edges cut in partitioning
    float ubvec = 1.05;
    vector<float> tpwgts(npart,1.0/(double)npart);
    MPI_Comm mpicomm=(dynamic_cast<const Epetra_MpiComm*>(&(dis->Comm())))->Comm();
    
    ParMETIS_V3_PartKway(&(vtxdist[0]),&(xadj[0]),&(adjncy[0]),
                         NULL,NULL,&wgtflag,&numflag,&ncon,&npart,&(tpwgts[0]),&ubvec,
                         &(options[0]),&edgecut,&(part[0]),&mpicomm);
    
  }
  
  double t3 = timer.ElapsedTime();
  if (!myrank && outflag)
  {                  
    printf("parmetis call     %10.5e secs\n",t3-t2);
    fflush(stdout);
  }

#if 0  
  for (int proc=0; proc<numproc; ++proc)
  {
    if (myrank==proc)
    {
      for (int i=0; i<graph->RowMap().NumMyElements(); ++i)
        printf("Proc %d part[%4d] = %d\n",myrank,i,part[i]);
      fflush(stdout);
    }
    comm->Barrier();
  }
#endif

  // build new row map according to part
  {
    vector<int> mygids;
    for (int proc=0; proc<numproc; ++proc)
    {
      int size = 0;
      vector<int> sendpart;
      vector<int> sendgid;
      if (proc==myrank)
      {
        sendpart = part;
        size = (int)sendpart.size();
        sendgid.resize(size);
        for (int i=0; i<(int)sendgid.size(); ++i) sendgid[i] = graph->RowMap().GID(i);
      }
      comm->Broadcast(&size,1,proc);
      if (proc!=myrank) 
      {
        sendpart.resize(size);
        sendgid.resize(size);
      }
      comm->Broadcast(&sendpart[0],size,proc);
      comm->Broadcast(&sendgid[0],size,proc);
      for (int i=0; i<(int)sendpart.size(); ++i)
      {
        if (sendpart[i] != myrank) continue;
        mygids.push_back(sendgid[i]);
      }
      comm->Barrier();
    }
    rownodes = rcp(new Epetra_Map(-1,(int)mygids.size(),&mygids[0],0,*comm)); 
  }

  // export the graph to the new row map to obtain the new column map
  {
    int bandwith = graph->GlobalMaxNumNonzeros();
    Epetra_CrsGraph finalgraph(Copy,*rownodes,bandwith,false);
    Epetra_Export exporter(graph->RowMap(),*rownodes);
    int err = finalgraph.Export(*graph,exporter,Insert);
    if (err<0) dserror("Graph export returned err=%d",err);
    graph = null;
    finalgraph.FillComplete();
    finalgraph.OptimizeStorage();
    colnodes = rcp(new Epetra_Map(-1,//finalgraph.ColMap().NumGlobalElements(),
                                  finalgraph.ColMap().NumMyElements(),
                                  finalgraph.ColMap().MyGlobalElements(),0,*comm));
  }


  double t4 = timer.ElapsedTime();
  if (!myrank && outflag)
  {                  
    printf("parmetis cleanup  %10.5e secs\n",t4-t3);
    fflush(stdout);
  }


  return;
}
#endif
#endif



















