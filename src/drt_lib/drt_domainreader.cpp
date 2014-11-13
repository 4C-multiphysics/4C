/*----------------------------------------------------------------------*/
/*!
\file drt_domainreader.cpp

\brief Read domain sections of dat files.

<pre>
Maintainer: Karl-Robert Wichmann
            wichmann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/
/*----------------------------------------------------------------------*/


#include "drt_domainreader.H"
#include "standardtypes_cpp.H"
#include "drt_elementdefinition.H"
#include "drt_utils_parmetis.H"
#include "drt_utils_factory.H"
#include "drt_utils_parallel.H"
#include "drt_discret.H"
#include "drt_parobject.H"
#include "../drt_particle/particle_node.H"
#include "../drt_io/io_pstream.H"

#include <Epetra_Time.h>

#if (BOOST_MAJOR_VERSION == 1) && (BOOST_MINOR_VERSION >= 47)
#include <boost/algorithm/string.hpp>
#endif

namespace DRT
{
namespace INPUT
{

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DomainReader::DomainReader(Teuchos::RCP<Discretization> dis,
                             const DRT::INPUT::DatFileReader& reader,
                             std::string sectionname)
  : name_(dis->Name()),
    reader_(reader),
    comm_(reader.Comm()),
    sectionname_(sectionname),
    dis_(dis),
    elementtype_(""),
    distype_(""),
    elearguments_("")
{
  for(size_t i = 0; i<sizeof(lower_bound_)/sizeof(lower_bound_[0]); ++i)
  {
    lower_bound_[i] = 0.0;
    upper_bound_[i] = 0.0;
    interval_[i] = 0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DomainReader::DomainReader(Teuchos::RCP<Discretization> dis,
                             const DRT::INPUT::DatFileReader& reader,
                             std::string sectionname,
                             std::string elementtype)
  : name_(dis->Name()),
    reader_(reader),
    comm_(reader.Comm()),
    sectionname_(sectionname),
    dis_(dis),
    elementtype_(""),
    distype_(""),
    elearguments_("")
{
  elementtypes_.insert(elementtype);
  for(size_t i = 0; i<sizeof(lower_bound_)/sizeof(lower_bound_[0]); ++i)
  {
    lower_bound_[i] = 0.0;
    upper_bound_[i] = 0.0;
    interval_[i] = 0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DomainReader::DomainReader(Teuchos::RCP<Discretization> dis,
                             const DRT::INPUT::DatFileReader& reader,
                             std::string sectionname,
                             const std::set<std::string> & elementtypes)
  : name_(dis->Name()),
    reader_(reader),
    comm_(reader.Comm()),
    sectionname_(sectionname),
    dis_(dis),
    elementtype_(""),
    distype_(""),
    elearguments_("")
{
  std::copy(elementtypes.begin(),elementtypes.end(),
            std::inserter(elementtypes_,elementtypes_.begin()));
  for(size_t i = 0; i<sizeof(lower_bound_)/sizeof(lower_bound_[0]); ++i)
  {
    lower_bound_[i] = 0.0;
    upper_bound_[i] = 0.0;
    interval_[i] = 0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DomainReader::Partition(int* nodeoffset)
{
  const int myrank  = comm_->MyPID();
  const int numproc = comm_->NumProc();

  bool autopartition = true;

  DRT::INPUT::ElementDefinition ed;
  ed.SetupValidElementLines();

  Epetra_Time time(*comm_);

  std::vector<int> eids;
  std::string inputfile_name = reader_.MyInputfileName();

  // all reading is done on proc 0
  if (myrank==0)
  {
    if (!reader_.MyOutputFlag())
      IO::cout << "Entering domain generation mode for " << name_
          << " discretization ...\nCreate and partition elements      in...."
          << IO::endl;

    // open input file at the right position
    std::ifstream file(inputfile_name.c_str());
    std::ifstream::pos_type pos = reader_.ExcludedSectionPosition(sectionname_);
    if (pos!=std::ifstream::pos_type(-1))
    {
      file.seekg(pos);

      // read domain info
      std::string line;
      for (int i=0; getline(file, line); ++i)
      {
        if (line.find("--")==0)
        {
          break;
        }
        else
        {
          std::istringstream t;
          t.str(line);
          std::string key;
          t >> key;
          if (key == "LOWER_BOUND")
            t >> lower_bound_[0] >> lower_bound_[1] >> lower_bound_[2];
          else if (key == "UPPER_BOUND")
            t >> upper_bound_[0] >> upper_bound_[1] >> upper_bound_[2];
          else if (key == "INTERVALS")
            t >> interval_[0] >> interval_[1] >> interval_[2];
          else if (key == "ELEMENTS")
          {
            t >> elementtype_ >> distype_;
            getline(t, elearguments_);
          }
          else if (key == "PARTITION")
          {
            std::string tmp;
            t >> tmp;
#if (BOOST_MAJOR_VERSION == 1) && (BOOST_MINOR_VERSION >= 47)
            boost::algorithm::to_lower(tmp);
#endif
            if (tmp == "auto")
              autopartition = true;
            else if (tmp == "structured")
              autopartition = false;
            else
              dserror("Invalid argument for PARTITION in DOMAIN reader. Valid options are \"auto\" and \"structured\".");
          }
          else
            dserror("Unknown Key in DOMAIN section");
        }
      }
    }
    else
    {
      dserror("No DOMAIN specified but box geometry selected!");
    }
  }

  // broadcast
  if (numproc > 1)
  {
    comm_->Broadcast(lower_bound_, sizeof(lower_bound_)/sizeof(lower_bound_[0]), 0);
    comm_->Broadcast(upper_bound_, sizeof(upper_bound_)/sizeof(upper_bound_[0]), 0);
    comm_->Broadcast(   interval_,       sizeof(interval_)/sizeof(interval_[0]), 0);
    {
      int tmp = autopartition;
      comm_->Broadcast(&tmp, 1, 0);
      autopartition = tmp;
    }

    std::vector<char> data;
    if (myrank == 0)
    {
      DRT::PackBuffer buffer;
      DRT::ParObject::AddtoPack(buffer, elementtype_);
      DRT::ParObject::AddtoPack(buffer, distype_);
      DRT::ParObject::AddtoPack(buffer, elearguments_);
      buffer.StartPacking();
      DRT::ParObject::AddtoPack(buffer, elementtype_);
      DRT::ParObject::AddtoPack(buffer, distype_);
      DRT::ParObject::AddtoPack(buffer, elearguments_);
      std::swap(data, buffer());
    }

    ssize_t data_size = data.size();
    comm_->Broadcast(&data_size,1,0);
    if (myrank != 0)
      data.resize(data_size,0);
    comm_->Broadcast(&(data[0]), data.size(), 0);

    if (myrank != 0)
    {
      size_t pos = 0;
      DRT::ParObject::ExtractfromPack(pos, data, elementtype_);
      DRT::ParObject::ExtractfromPack(pos, data, distype_);
      DRT::ParObject::ExtractfromPack(pos, data, elearguments_);
    }
  }

  // safety checks
  for (int i=0; i<3; ++i)
  {
    if (lower_bound_[i] >= upper_bound_[i])
      dserror("lower bound in domain reader must be smaller than upper bound");

    if (interval_[i] <= 0)
      dserror("intervals in domain reader must be greater than zero");
  }

  // Create initial (or final) map of row elements
  int numnewele = interval_[0]*interval_[1]*interval_[2];
  if (autopartition) // linear map
  {
    roweles_ = Teuchos::rcp(new Epetra_Map(numnewele,0,*comm_));
  }
  else // fancy final box map
  {
    std::vector<int> factors;
    int nproc = numproc;
    for (int fac = 2; fac < nproc+1;)
    {
      if (nproc % fac == 0)
      {
        factors.push_back(fac);
        nproc /= fac;
      }
      else
      {
        fac++;
      }
    }
    if (nproc != 1)
      dserror("Could not split numproc.");

    unsigned int subdivisions[] = {1,1,1};
    const double dinterval[]    = {static_cast<double>(interval_[0]), static_cast<double>(interval_[1]), static_cast<double>(interval_[2])};
    for (std::vector<int>::const_reverse_iterator fac = factors.rbegin(); fac != factors.rend(); ++fac)
    {
      const double ratios[] = {dinterval[0]/subdivisions[0], dinterval[1]/subdivisions[1], dinterval[2]/subdivisions[2]};
      if (ratios[0] >= ratios[1] && ratios[0] >= ratios[2])
        subdivisions[0] *= *fac;
      else if (ratios[1] >= ratios[0] && ratios[1] >= ratios[2])
        subdivisions[1] *= *fac;
      else if (ratios[2] >= ratios[0] && ratios[2] >= ratios[1])
        subdivisions[2] *= *fac;
    }

    if (myrank == 0)
      IO::cout << "Determined domain subdivision to: " << subdivisions[0] << "x" << subdivisions[1] << "x" << subdivisions[2] << "\n";

    unsigned int xranges[subdivisions[0]+1ul];
    for (size_t i=0; i<subdivisions[0]+1ul; ++i)
      xranges[i] = std::max(0, std::min(interval_[0], static_cast<int>(round(i*dinterval[0]/subdivisions[0])) ));

    unsigned int yranges[subdivisions[1]+1ul];
    for (size_t i=0; i<subdivisions[1]+1ul; ++i)
      yranges[i] = std::max(0, std::min(interval_[1], static_cast<int>(round(i*dinterval[1]/subdivisions[1])) ));

    unsigned int zranges[subdivisions[2]+1ul];
    for (size_t i=0; i<subdivisions[2]+1ul; ++i)
      zranges[i] = std::max(0, std::min(interval_[2], static_cast<int>(round(i*dinterval[2]/subdivisions[2])) ));

    const unsigned int mysection[] = {myrank%subdivisions[0], (myrank/subdivisions[0])%subdivisions[1], myrank/(subdivisions[0]*subdivisions[1])};
    const int nummynewele          = (xranges[mysection[0]+1]-xranges[mysection[0]])*(yranges[mysection[1]+1]-yranges[mysection[1]])*(zranges[mysection[2]+1]-zranges[mysection[2]]);
    int mynewele[nummynewele];

    size_t idx = 0;
    for (size_t iz=zranges[mysection[2]]; iz<zranges[mysection[2]+1]; ++iz)
      for (size_t iy=yranges[mysection[1]]; iy<yranges[mysection[1]+1]; ++iy)
        for (size_t ix=xranges[mysection[0]]; ix<xranges[mysection[0]+1]; ++ix)
          mynewele[idx++] = (iz*interval_[1]+iy)*interval_[0]+ix;

    roweles_ = Teuchos::rcp(new Epetra_Map(-1,nummynewele,mynewele,0,*comm_));
  }

  // Create the actual elements according to the row map
  for (int lid = 0; lid < roweles_->NumMyElements(); ++lid)
  {
    int eleid = roweles_->GID(lid);
    dsassert(eleid >= 0, "Missing gid");

    // let the factory create a matching empty element
    Teuchos::RCP<DRT::Element> ele = DRT::UTILS::Factory(elementtype_,distype_,eleid,myrank);

    // For the time being we support old and new input facilities. To
    // smooth transition.

    DRT::INPUT::LineDefinition* linedef = ed.ElementLines(elementtype_,distype_);
    if (linedef == NULL)
      dserror("a matching line definition is needed for %s %s", elementtype_.c_str(), distype_.c_str());

    std::istringstream eleargstream(elearguments_);
    if (not linedef->Read(eleargstream, &distype_))
    {
      IO::cout << "\n"
                << eleid << " "
                << elementtype_ << " "
                << distype_ << " ";
      linedef->Print(IO::cout.cout_replacement());
      IO::cout << "\n";
      dserror("failed to read element %d %s %s",eleid,elementtype_.c_str(),distype_.c_str());
    }

    // this depends on the distype
    std::vector<int> nodeids(DRT::UTILS::getNumberOfElementNodes(DRT::StringToDistype(distype_)));
    if (nodeids.size() == 8)
    {
      // current element position
      const size_t ex =  eleid%interval_[0];
      const size_t ey = (eleid/interval_[0])%interval_[1];
      const size_t ez =  eleid/(interval_[0]*interval_[1]);

      // number of nodes per direction
      const size_t nx = interval_[0]+1;
      const size_t ny = interval_[1]+1;

      nodeids[0] = *nodeoffset+(ez*ny+ey)*nx+ex;
      nodeids[1] = *nodeoffset+(ez*ny+ey)*nx+ex+1;
      nodeids[2] = *nodeoffset+(ez*ny+ey+1)*nx+ex+1;
      nodeids[3] = *nodeoffset+(ez*ny+ey+1)*nx+ex;
      nodeids[4] = *nodeoffset+((ez+1)*ny+ey)*nx+ex;
      nodeids[5] = *nodeoffset+((ez+1)*ny+ey)*nx+ex+1;
      nodeids[6] = *nodeoffset+((ez+1)*ny+ey+1)*nx+ex+1;
      nodeids[7] = *nodeoffset+((ez+1)*ny+ey+1)*nx+ex;
    }
    else
    {
      dserror("Not implemented: Currently only elements with 8 nodes are implemented for the box geometry generation.");
    }
    ele->SetNodeIds(nodeids.size(),&(nodeids[0]));
    ele->ReadElement(elementtype_,distype_,linedef);

    // add element to discretization
    dis_->AddElement(ele);
  }

  // redistribute the elements
  if (autopartition)
  {
#if defined(PARMETIS)
    rownodes_ = Teuchos::null;
    colnodes_ = Teuchos::null;
    DRT::UTILS::PartUsingParMetis(dis_,roweles_,rownodes_,colnodes_,comm_,!reader_.MyOutputFlag());
#else
    dserror("We need parmetis.");
#endif
  }
  else // do not destroy our manual partitioning
  {
    Teuchos::RCP<const Epetra_CrsGraph> graph = DRT::UTILS::BuildGraph(dis_, roweles_,
        rownodes_, comm_, !reader_.MyOutputFlag());
    colnodes_ = Teuchos::rcp(new Epetra_Map(-1, graph->ColMap().NumMyElements(),
        graph->ColMap().MyGlobalElements(),0,*comm_));
  }


  // now we have all elements in a linear map roweles
  // build reasonable maps for elements from the
  // already valid and final node maps
  // note that nothing is actually redistributed in here
  dis_->BuildElementRowColumn(*rownodes_,*colnodes_,roweles_,coleles_);

  // we can now export elements to resonable row element distribution
  dis_->ExportRowElements(*roweles_);

  // export to the column map / create ghosting of elements
  dis_->ExportColumnElements(*coleles_);

  // Create the nodes according to their elements
  // number of nodes per direction
  const size_t nx = interval_[0]+1;
  const size_t ny = interval_[1]+1;
  int maxgid = -1;

  // as we are using the redistributed row node map, the nodes are directly created on the
  // correct processors
  for (int lid = 0; lid < rownodes_->NumMyElements(); ++lid)
  {
    const int gid = rownodes_->GID(lid);
    maxgid = std::max(gid, maxgid);

    const int posid = gid - *nodeoffset;
    dsassert(posid >= 0, "Tried to access a node gid that was not on this proc");
    size_t i = posid%nx;
    size_t j = (posid/nx)%ny;
    size_t k = posid/(nx*ny);

    double coords[3];
    coords[0] = static_cast<double>(i)/interval_[0]*(upper_bound_[0]-lower_bound_[0])+lower_bound_[0];
    coords[1] = static_cast<double>(j)/interval_[1]*(upper_bound_[1]-lower_bound_[1])+lower_bound_[1];
    coords[2] = static_cast<double>(k)/interval_[2]*(upper_bound_[2]-lower_bound_[2])+lower_bound_[2];

    Teuchos::RCP<DRT::Node> node = Teuchos::rcp(new DRT::Node(gid,coords,myrank));
    dis_->AddNode(node);
  }
  dis_->ExportColumnNodes(*colnodes_);

  comm_->MaxAll(&maxgid, nodeoffset, 1);

  if (!myrank && reader_.MyOutputFlag() == 0)
    IO::cout << "............................................... "
        << std::setw(10) << std::setprecision(5) << std::scientific
        << time.ElapsedTime() << " secs" << IO::endl;

  return;
} // end Partition()


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DomainReader::Complete()
{
  const int myrank  = comm_->MyPID();

  Epetra_Time time(*comm_);

  if (!myrank && !reader_.MyOutputFlag())
    IO::cout << "Complete discretization " <<  std::left << std::setw(16)
             << name_ << " in...." << IO::flush;

  int err = dis_->FillComplete(false,false,false);
  if (err)
    dserror("dis_->FillComplete() returned %d",err);

  if (!myrank && !reader_.MyOutputFlag())
    IO::cout << time.ElapsedTime() << " secs" << IO::endl;

  DRT::UTILS::PrintParallelDistribution(*dis_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ParticleDomainReader::ParticleDomainReader(Teuchos::RCP<Discretization> dis,
                               const DRT::INPUT::DatFileReader& reader,
                               std::string sectionname)
  : DomainReader(dis, reader, sectionname)
{}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ParticleDomainReader::Partition(int* nodeoffset)
{
  const int myrank  = comm_->MyPID();
  const int numproc = comm_->NumProc();

  Epetra_Time time(*comm_);

  std::string inputfile_name = reader_.MyInputfileName();

  bool haveparticledomain = true;
  // all reading is done on proc 0
  if (myrank==0)
  {
    if (!reader_.MyOutputFlag())
      IO::cout << "Entering domain generation mode for " << name_
          << " discretization ...\nCreate and partition elements      in...."
          << IO::endl;

    // open input file at the right position
    std::ifstream file(inputfile_name.c_str());
    std::ifstream::pos_type pos = reader_.ExcludedSectionPosition(sectionname_);
    if (pos!=std::ifstream::pos_type(-1))
    {
      file.seekg(pos);

      // read domain info
      std::string line;
      for (int i=0; getline(file, line); ++i)
      {
        if (line.find("--")==0)
        {
          break;
        }
        else
        {
          std::istringstream t;
          t.str(line);
          std::string key;
          t >> key;
          if (key == "LOWER_BOUND")
            t >> lower_bound_[0] >> lower_bound_[1] >> lower_bound_[2];
          else if (key == "UPPER_BOUND")
            t >> upper_bound_[0] >> upper_bound_[1] >> upper_bound_[2];
          else if (key == "INTERVALS")
            t >> interval_[0] >> interval_[1] >> interval_[2];
          else
            dserror("Unknown Key in DOMAIN section");
        }
      }
    }
    else
    {
      haveparticledomain = false;
    }
  }

  // check whether a particle domain is specified
  int tmp = haveparticledomain;
  comm_->Broadcast(&tmp, 1, 0);
  if(!tmp)
    return;

  // broadcast
  if (numproc > 1)
  {
    comm_->Broadcast(lower_bound_, sizeof(lower_bound_)/sizeof(lower_bound_[0]), 0);
    comm_->Broadcast(upper_bound_, sizeof(upper_bound_)/sizeof(upper_bound_[0]), 0);
    comm_->Broadcast(   interval_,       sizeof(interval_)/sizeof(interval_[0]), 0);
  }

  // safety checks
  for (int i=0; i<3; ++i)
  {
    if (lower_bound_[i] >= upper_bound_[i])
      dserror("lower bound in domain reader must be smaller than upper bound");

    if (interval_[i] <= 0)
      dserror("intervals in domain reader must be greater than zero");
  }

  // split node ids
  int numnewnodes = interval_[0]*interval_[1]*interval_[2];
  rownodes_ = Teuchos::rcp(new Epetra_Map(numnewnodes,*nodeoffset,*comm_));

  // number of nodes per direction
  const size_t nx = interval_[0];
  const size_t ny = interval_[1];
  int maxgid = -1;

  // as we are using a distributed row node map, the nodes are directly created on the
  // correct processors
  for (int lid = 0; lid < rownodes_->NumMyElements(); ++lid)
  {
    const int gid = rownodes_->GID(lid);
    maxgid = std::max(gid, maxgid);

    const int posid = gid - *nodeoffset;
    dsassert(posid >= 0, "Tried to access a node gid that was not on this proc");
    size_t i = posid%nx;
    size_t j = (posid/nx)%ny;
    size_t k = posid/(nx*ny);

    double coords[3];
    coords[0] = static_cast<double>(i)/(interval_[0]-1)*(upper_bound_[0]-lower_bound_[0])+lower_bound_[0];
    coords[1] = static_cast<double>(j)/(interval_[1]-1)*(upper_bound_[1]-lower_bound_[1])+lower_bound_[1];
    coords[2] = static_cast<double>(k)/(interval_[2]-1)*(upper_bound_[2]-lower_bound_[2])+lower_bound_[2];

    Teuchos::RCP<DRT::Node> node = Teuchos::rcp(new DRT::Node(gid,coords,myrank));

    // create particle and add to discretization
    Teuchos::RCP<DRT::Node> particle = Teuchos::rcp(new PARTICLE::ParticleNode(gid,coords,myrank));
    dis_->AddNode(particle);
  }

  comm_->MaxAll(&maxgid, nodeoffset, 1);

  if (!myrank && reader_.MyOutputFlag() == 0)
    IO::cout << "............................................... "
        << std::setw(10) << std::setprecision(5) << std::scientific
        << time.ElapsedTime() << " secs" << IO::endl;

  return;
}

}
}

