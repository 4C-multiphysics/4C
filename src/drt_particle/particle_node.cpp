/*----------------------------------------------------------------------*/
/*!
\file particle_node.cpp

\brief A particle is a DRT::Node with additional knowledge of its collision status

<pre>
Maintainer: Georg Hammerl
            hammerl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-152537
</pre>
*----------------------------------------------------------------------*/
#include "particle_node.H"
#include "../drt_lib/drt_dserror.H"


PARTICLE::ParticleNodeType PARTICLE::ParticleNodeType::instance_;


DRT::ParObject* PARTICLE::ParticleNodeType::Create( const std::vector<char> & data )
{
  double dummycoord[3] = {999.,999.,999.};
  DRT::Node* particle = new PARTICLE::ParticleNode(-1,dummycoord,-1);
  particle->Unpack(data);
  return particle;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            ghamm 06/13|
 *----------------------------------------------------------------------*/
PARTICLE::ParticleNode::ParticleNode(
  int id,
  const double* coords,
  const int owner
) :
DRT::Node(id,coords,owner)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       ghamm 06/13|
 *----------------------------------------------------------------------*/
PARTICLE::ParticleNode::ParticleNode(const PARTICLE::ParticleNode& old) :
DRT::Node(old),
history_particle_(old.history_particle_),
history_wall_(old.history_wall_)
{

  // not yet used and thus not necessarily consistent
  dserror("ERROR: ParticleNode copy-ctor not yet implemented");

  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance and return pointer to it (public)           |
 |                                                           ghamm 06/13|
 *----------------------------------------------------------------------*/
PARTICLE::ParticleNode* PARTICLE::ParticleNode::Clone() const
{
  PARTICLE::ParticleNode* newnode = new PARTICLE::ParticleNode(*this);
  return newnode;
}

/*----------------------------------------------------------------------*
 |  << operator                                              ghamm 06/13|
 *----------------------------------------------------------------------*/
ostream& operator << (ostream& os, const PARTICLE::ParticleNode& particle)
{
  particle.Print(os);
  return os;
}

/*----------------------------------------------------------------------*
 |  print this ParticleNode (public)                         ghamm 06/13|
 *----------------------------------------------------------------------*/
void PARTICLE::ParticleNode::Print(ostream& os) const
{
  // Print id and coordinates
  os << "Particle ";
  DRT::Node::Print(os);

  if(history_particle_.size())
    os << "in contact with particles (Id): ";
  for(std::map<int, Collision>::const_iterator i=history_particle_.begin(); i!=history_particle_.end(); ++i)
  {
    os << i->first << "  ";
  }

  if(history_particle_.size())
    os << "in contact with walls (Id): ";
  for(std::map<int, Collision>::const_iterator i=history_particle_.begin(); i!=history_particle_.end(); ++i)
  {
    os << i->first << "  ";
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           ghamm 06/13|
 *----------------------------------------------------------------------*/
void PARTICLE::ParticleNode::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class DRT::Node
  DRT::Node::Pack(data);

  // add size of history_particle_
  AddtoPack(data,(int)history_particle_.size());
  for(std::map<int, Collision>::const_iterator i=history_particle_.begin(); i!=history_particle_.end(); ++i)
  {
    // one collision event with another particle
    AddtoPack(data,i->first);
    AddtoPack(data,i->second.stick);
    AddtoPack(data,i->second.g_t,3*sizeof(double));
  }

  // add size of history_wall_
  AddtoPack(data,(int)history_wall_.size());
  for(std::map<int, Collision>::const_iterator i=history_wall_.begin(); i!=history_wall_.end(); ++i)
  {
    // one collision event with a wall element
    AddtoPack(data,i->first);
    AddtoPack(data,i->second.stick);
    AddtoPack(data,i->second.g_t,3*sizeof(double));
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           ghamm 06/13|
 *----------------------------------------------------------------------*/
void PARTICLE::ParticleNode::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class DRT::Node
  std::vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  DRT::Node::Unpack(basedata);

  // extract size of history_particle_
  int num_particle_collisions;
  ExtractfromPack(position,data,num_particle_collisions);
  for(int i=0; i<num_particle_collisions; ++i)
  {
    // one collision event with another particle
    Collision coll;
    // collision id
    int collid = -1;
    ExtractfromPack(position,data,collid);
    // coll.stick
    coll.stick = ExtractInt(position,data);
    // coll.g_t
    ExtractfromPack(position,data,coll.g_t,3*sizeof(double));
    history_particle_[collid] = coll;
  }

  // extract size of history_wall_
  int num_wall_collisions;
  ExtractfromPack(position,data,num_wall_collisions);
  for(int i=0; i<num_wall_collisions; ++i)
  {
    // one collision event with a wall element
    Collision coll;
    // collision id
    int collid = -1;
    ExtractfromPack(position,data,collid);
    // coll.stick
    coll.stick = ExtractInt(position,data);
    // coll.g_t
    ExtractfromPack(position,data,coll.g_t,3*sizeof(double));
    history_wall_[collid] = coll;
  }

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}
