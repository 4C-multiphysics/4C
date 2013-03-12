


#include "drt_discret.H"
#include "drt_nodereader.H"
#include "drt_globalproblem.H"
#include "../drt_nurbs_discret/drt_control_point.H"
#include "../drt_meshfree_discret/drt_meshfree_discret.H"
#include "../drt_meshfree_discret/drt_meshfree_node.H"

#include <Epetra_Time.h>

namespace DRT
{
namespace INPUT
{

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NodeReader::NodeReader(const DRT::INPUT::DatFileReader& reader, std::string sectionname)
  : reader_(reader), comm_(reader.Comm()), sectionname_(sectionname)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Discretization> > NodeReader::FindDisNode(int nodeid)
{
  std::vector<Teuchos::RCP<DRT::Discretization> > v;
  for (unsigned i=0; i<ereader_.size(); ++i)
  {
    if (ereader_[i]->HasNode(nodeid))
    {
      v.push_back(ereader_[i]->MyDis());
    }
  }
  return v;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void NodeReader::Read()
{
  const int myrank  = comm_->MyPID();
  const int numproc = comm_->NumProc();
  std::string inputfile_name = reader_.MyInputfileName();

  int numnodes = reader_.ExcludedSectionLength(sectionname_);

  if( (numnodes < numproc) && (numnodes != 0) )
    dserror("Bad idea: Simulation with %d procs for problem with %d nodes", numproc,numnodes);

  for (unsigned i=0; i<ereader_.size(); ++i)
  {
    ereader_[i]->Partition();
  }

  // Debug
#if 0
  if (numnodes!=DRT::Problem::Instance()->ProblemSizeParams().get<int>("NODES"))
    dserror("expect %d nodes but got %d",
            DRT::Problem::Instance()->ProblemSizeParams().get<int>("NODES"),
            numnodes);
#endif

  Epetra_Time time(*comm_);

  if (!myrank && !reader_.MyOutputFlag())
  {
    cout << "Read, create and partition nodes\n" << flush;
#if defined(PARALLEL) && defined(PARMETIS)
//    cout << "block " << flush;
#else
//    cout << "        " << flush;
#endif
  }

  // We will read the nodes block wise. we will use one block per processor
  // so the number of blocks is numproc
  // determine a rough blocksize
  int nblock = numproc;
  int bsize = std::max(numnodes/nblock, 1);

  // an upper limit for bsize
  int maxblocksize = 200000;

  if (bsize > maxblocksize)
  {
    // without an additional increase of nblock by 1 the last block size
    // could reach a maximum value of (2*maxblocksize)-1, potentially
    // violating the intended upper limit!
    nblock = 1+ static_cast<int>(numnodes/maxblocksize);
    bsize = maxblocksize;
  }

  // open input file at the right position
  // note that stream is valid on proc 0 only!
  std::ifstream file;
  if (myrank==0)
  {
    file.open(inputfile_name.c_str());
    file.seekg(reader_.ExcludedSectionPosition(sectionname_));
  }
  std::string tmp;

  if (!myrank && !reader_.MyOutputFlag())
  {
    printf("numnode %d nblock %d bsize %d\n",numnodes,nblock,bsize);
    fflush(stdout);
  }


  // note that the last block is special....
  int filecount=0;
  for (int block=0; block<nblock; ++block)
  {
    double t1 = time.ElapsedTime();
    if (0==myrank)
    {
#if defined(PARALLEL) && defined(PARMETIS)
      if (!reader_.MyOutputFlag())
        printf("block %d ",block);
#endif

      int bcount=0;
      for (; file; ++filecount)
      {
        file >> tmp;

        if (tmp=="NODE")
        {
          double coords[3];
          int nodeid;
          file >> nodeid >> tmp >> coords[0] >> coords[1] >> coords[2];
          nodeid--;
          if (tmp!="COORD") dserror("failed to read node %d",nodeid);
          std::vector<Teuchos::RCP<DRT::Discretization> > diss = FindDisNode(nodeid);
          for (unsigned i=0; i<diss.size(); ++i)
          {
            // create node and add to discretization
            Teuchos::RCP<DRT::Node> node = Teuchos::rcp(new DRT::Node(nodeid,coords,myrank));
            diss[i]->AddNode(node);
          }
          ++bcount;
          if (block != nblock-1) // last block takes all the rest
            if (bcount==bsize)   // block is full
            {
              ++filecount;
              break;
            }
        }
        // this node is a meshfree knot
        else if (tmp=="KNOT")
        {
          double coords[3];
          int nodeid;
          file >> nodeid >> tmp >> coords[0] >> coords[1] >> coords[2];
          nodeid--;
          if (tmp!="COORD") dserror("failed to read node %d",nodeid);
          std::vector<Teuchos::RCP<DRT::Discretization> > diss = FindDisNode(nodeid);
          for (unsigned i=0; i<diss.size(); ++i)
          {
            // create node and add to discretization
            Teuchos::RCP<DRT::Node> node = Teuchos::rcp(new DRT::Node(nodeid,coords,myrank));
            Teuchos::RCP<DRT::MESHFREE::MeshfreeNode> knot = Teuchos::rcp(new DRT::MESHFREE::MeshfreeNode(nodeid,coords,myrank));
            diss[i]->AddNode(node);
            // hyper hack ?
            Teuchos::rcp_dynamic_cast<DRT::MESHFREE::MeshfreeDiscretization>(diss[i])->AddKnot(knot);
          }
          ++bcount;
          if (block != nblock-1) // last block takes all the rest
            if (bcount==bsize)   // block is full
            {
              ++filecount;
              break;
            }
        }
        // this node is a Nurbs control point
        else if (tmp=="CP")
        {
          // read control points for isogeometric analysis (Nurbs)
          double coords[3];
          double weight;

          int cpid;
          file >> cpid >> tmp >> coords[0] >> coords[1] >> coords[2] >> weight;
          cpid--;
          if (cpid != filecount)
            dserror("Reading of control points failed: They must be numbered consecutive!!");
          if (tmp!="COORD")
            dserror("failed to read control point %d",cpid);
          std::vector<Teuchos::RCP<DRT::Discretization> > diss = FindDisNode(cpid);
          for (unsigned i=0; i<diss.size(); ++i)
          {
            Teuchos::RCP<DRT::Discretization> dis = diss[i];
            // create node/control point and add to discretization
            Teuchos::RCP<DRT::NURBS::ControlPoint> node = Teuchos::rcp(new DRT::NURBS::ControlPoint(cpid,coords,weight,myrank));
            dis->AddNode(node);
          }
          ++bcount;
          if (block != nblock-1) // last block takes all the rest
            if (bcount==bsize)   // block is full
            {
              ++filecount;
              break;
            }
        }
        // this node is a particle
        else if (tmp=="PARTICLE")
        {
          double coords[3];
          int nodeid;
          file >> nodeid >> tmp >> coords[0] >> coords[1] >> coords[2];
          nodeid--;
          if (tmp!="COORD") dserror("failed to read node %d",nodeid);
          Teuchos::RCP<DRT::Discretization> diss = DRT::Problem::Instance()->GetDis("particle");

          // create particle and add to discretization
          Teuchos::RCP<DRT::Node> particle = Teuchos::rcp(new DRT::Node(nodeid,coords,myrank));
          diss->AddNode(particle);

          ++bcount;
          if (block != nblock-1) // last block takes all the rest
            if (bcount==bsize)   // block is full
            {
              ++filecount;
              break;
            }
        }
        else if (tmp.find("--")==0)
          break;
        else
          dserror("unexpected word '%s'",tmp.c_str());
      } // for (filecount; file; ++filecount)
    } // if (0==myrank)

    double t2 = time.ElapsedTime();
    if (!myrank && !reader_.MyOutputFlag())
      printf("reading %10.5e secs",t2-t1);

    // export block of nodes to other processors as reflected in rownodes,
    // changes ownership of nodes
    for (unsigned i=0; i<ereader_.size(); ++i)
    {
      ereader_[i]->dis_->ProcZeroDistributeNodesToAll(*ereader_[i]->rownodes_);
      // this does the same job but slower
      //ereader_[i]->dis_->ExportRowNodes(*ereader_[i]->rownodes_);
    }
    double t3 = time.ElapsedTime();
    if (!myrank && !reader_.MyOutputFlag())
    {
      printf(" / distrib %10.5e secs\n",t3-t2);
      fflush(stdout);
    }

  } // for (int block=0; block<nblock; ++block)


  // last thing to do here is to produce nodal ghosting/overlap
  for (unsigned i=0; i<ereader_.size(); ++i)
  {
    ereader_[i]->dis_->ExportColumnNodes(*ereader_[i]->colnodes_);
  }

  if (!myrank && !reader_.MyOutputFlag())
    printf("in............................................. %10.5e secs\n",time.ElapsedTime());

  for (unsigned i=0; i<ereader_.size(); ++i)
  {
    ereader_[i]->Complete();
  }
} // NodeReader::Read

}
}

