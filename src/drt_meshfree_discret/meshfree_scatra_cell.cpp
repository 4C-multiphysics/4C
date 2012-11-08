/*!----------------------------------------------------------------------
\file meshfree_scatra_cell.cpp

\brief scatra cell for meshfree discretisations

<pre>
Maintainer: Keijo Nissen
            nissen@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15253
</pre>

*----------------------------------------------------------------------*/

#include "meshfree_scatra_cell.H"
#include "../drt_lib/drt_utils_factory.H"
#include "../drt_lib/drt_utils_nullspace.H"
//#include "../drt_lib/drt_utils.H"
#include "../drt_mat/matlist.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_linedefinition.H"

/*==========================================================================*\
 *                                                                          *
 * class MeshfreeTransportType                                              *
 *                                                                          *
\*==========================================================================*/

DRT::ELEMENTS::MeshfreeTransportType DRT::ELEMENTS::MeshfreeTransportType::instance_;

DRT::ParObject* DRT::ELEMENTS::MeshfreeTransportType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::MeshfreeTransport* object =
    new DRT::ELEMENTS::MeshfreeTransport(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::MeshfreeTransportType::Create( const string eletype,
                                                                         const string eledistype,
                                                                         const int id,
                                                                         const int owner )
{
  if (eletype=="METRANSP")
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::MeshfreeTransport(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::MeshfreeTransportType::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::MeshfreeTransport(id,owner));
  return ele;
}


void DRT::ELEMENTS::MeshfreeTransportType::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = dwele->NumDofPerNode(*(dwele->Nodes()[0]));
  dimns = numdf;
  nv = numdf;

  if (DRT::Problem::Instance(0)->ProblemName() == "elch")
  {
    if (nv > 1) // only when we have more than 1 dof per node!
    {
      nv -= 1; // ion concentrations
      np = 1;  // electric potential
    }
  }
}

void DRT::ELEMENTS::MeshfreeTransportType::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeFluidDNullSpace( dis, ns, x0, numdf, dimns );
}

void DRT::ELEMENTS::MeshfreeTransportType::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["METRANSP"];

  defs["HEX8"]
    .AddIntVector("HEX8",8)
    .AddNamedInt("MAT")
    ;

  defs["TET4"]
    .AddIntVector("TET4",4)
    .AddNamedInt("MAT")
    ;

  defs["QUAD4"]
    .AddIntVector("QUAD4",4)
    .AddNamedInt("MAT")
    ;

  defs["TRI3"]
    .AddIntVector("TRI3",3)
    .AddNamedInt("MAT")
    ;

  defs["LINE2"]
    .AddIntVector("LINE2",2)
    .AddNamedInt("MAT")
    ;

  defs["POINT1"]
    .AddIntVector("POINT1",1)
    .AddNamedInt("MAT")
    ;

//  defs["VOROH8"]
//    .AddIntVector("VOROH8",8)
//    .AddNamedInt("MAT")
//    ;
//
//  defs["VOROT4"]
//    .AddIntVector("VOROT4",4)
//    .AddNamedInt("MAT")
//    ;
//
//  defs["VOROQ4"]
//    .AddIntVector("VOROQ4",4)
//    .AddNamedInt("MAT")
//    ;
//
//  defs["VOROT3"]
//    .AddIntVector("VOROT3",3)
//    .AddNamedInt("MAT")
//    ;
//
//  defs["VOROL2"]
//    .AddIntVector("VOROL2",2)
//    .AddNamedInt("MAT")
//    ;
//
//  defs["VOROP1"]
//    .AddIntVector("VOROP1",1)
//    .AddNamedInt("MAT")
//    ;
}

/*==========================================================================*\
 *                                                                          *
 * class MeshfreeTransport                                                  *
 *                                                                          *
\*==========================================================================*/

/*--------------------------------------------------------------------------*
 |  ctor                                                 (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
DRT::ELEMENTS::MeshfreeTransport::MeshfreeTransport(int id, int owner) :
DRT::MESHFREE::Cell(id,owner),
data_(),
numdofpernode_(-1),
distype_(DRT::Element::dis_none)
{
    return;
}

/*--------------------------------------------------------------------------*
 |  copy-ctor                                            (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
DRT::ELEMENTS::MeshfreeTransport::MeshfreeTransport(const DRT::ELEMENTS::MeshfreeTransport& old) :
DRT::MESHFREE::Cell(old),
data_(old.data_),
numdofpernode_(old.numdofpernode_),
distype_(old.distype_)
{
    return;
}

/*--------------------------------------------------------------------------*
 | returns pointer to deep copy of MeshfreeTransport     (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::MeshfreeTransport::Clone() const
{
  DRT::ELEMENTS::MeshfreeTransport* newelement = new DRT::ELEMENTS::MeshfreeTransport(*this);
  return newelement;
}


/*--------------------------------------------------------------------------*
 |  create material class                                (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransport::SetMaterial(int matnum)
{
  // the standard part:
  //mat_ = MAT::Material::Factory(matnum);  // not allowed since mat_ is private
  DRT::Element::SetMaterial(matnum);

  // the special part:
  // now the element knows its material, and we can use it to determine numdofpernode
  RefCountPtr<MAT::Material> mat = Material();
  if(mat->MaterialType() == INPAR::MAT::m_scatra or
     mat->MaterialType() == INPAR::MAT::m_mixfrac or
     mat->MaterialType() == INPAR::MAT::m_sutherland or
     mat->MaterialType() == INPAR::MAT::m_arrhenius_pv or
     mat->MaterialType() == INPAR::MAT::m_ferech_pv or
     mat->MaterialType() == INPAR::MAT::m_ion or
     mat->MaterialType() == INPAR::MAT::m_biofilm or
     mat->MaterialType() == INPAR::MAT::m_th_fourier_iso or
     mat->MaterialType() == INPAR::MAT::m_yoghurt
     )
  {
    numdofpernode_=1; // we only have a single scalar
  }
  else if (mat->MaterialType() == INPAR::MAT::m_matlist) // we have a system of scalars
  {
    const MAT::MatList* actmat = static_cast<const MAT::MatList*>(mat.get());
    numdofpernode_=actmat->NumMat();

    // for problem type ELCH we have one additional degree of freedom per node
    // for the electric potential
    if (DRT::Problem::Instance()->ProblemName()=="elch")
    {
      numdofpernode_ += 1;
      dsassert(numdofpernode_>2,"numdofpernode_ is not > 2 for ELCH problem");
    }

  }
  else
    dserror("MeshfreeTransport element got unsupported material type %d", mat->MaterialType());

  return;
}


/*--------------------------------------------------------------------------*
 |  Return the shape of a MeshfreeTransport element      (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::MeshfreeTransport::Shape() const
{
  return distype_;
}

/*--------------------------------------------------------------------------*
 |  Return number of lines of this element               (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
int DRT::ELEMENTS::MeshfreeTransport::NumLine() const
{
  return DRT::UTILS::getNumberOfElementLines(distype_);
}


/*--------------------------------------------------------------------------*
 |  Return number of surfaces of this element            (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
int DRT::ELEMENTS::MeshfreeTransport::NumSurface() const
{
  return DRT::UTILS::getNumberOfElementSurfaces(distype_);
}


/*--------------------------------------------------------------------------*
 | Return number of volumes of this element              (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
int DRT::ELEMENTS::MeshfreeTransport::NumVolume() const
{
  return DRT::UTILS::getNumberOfElementVolumes(distype_);
}


/*--------------------------------------------------------------------------*
 |  dtor                                                 (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
DRT::ELEMENTS::MeshfreeTransport::~MeshfreeTransport()
{
  return;
}


/*--------------------------------------------------------------------------*
 |  print this element                                   (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransport::Print(ostream& os) const
{
  os << "MeshfreeTransportCell ";
  Cell::Print(os);
  cout << "DisType "<< DRT::DistypeToString(distype_) << " ";
  cout << "NumDofPerNode " << numdofpernode_ << " ";
  cout << data_;
  return;
}


/*--------------------------------------------------------------------------*
 |  get vector of lines                                  (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::MeshfreeTransport::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  if (NumLine() > 1) // 3D and 2D
    return DRT::UTILS::ElementBoundaryFactory<MeshfreeTransportBoundary,MeshfreeTransport>(DRT::UTILS::buildLines,this);
  else
  {
    // 1D (we return the element itself)
    vector<RCP<Element> > lines(1);
    lines[0]= rcp(this, false);
    return lines;
  }

}


/*--------------------------------------------------------------------------*
 |  get vector of surfaces                               (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::MeshfreeTransport::Surfaces()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new surface elements:
  if (NumSurface() > 1) // 3D
    return DRT::UTILS::ElementBoundaryFactory<MeshfreeTransportBoundary,MeshfreeTransport>(DRT::UTILS::buildSurfaces,this);
  else if (NumSurface() == 1)
  {
    // 2D (we return the element itself)
    vector<RCP<Element> > surfaces(1);
    surfaces[0]= rcp(this, false);
    return surfaces;
  }
  else
  {
    // 1D
    dserror("Surfaces() for 1D-MeshfreeTransport element not implemented");
    return DRT::Element::Surfaces();
  }
}


/*--------------------------------------------------------------------------*
 |  get vector of volumes (length 1)                     (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::MeshfreeTransport::Volumes()
{
  if (NumVolume() == 1)
  {
    vector<RCP<Element> > volumes(1);
    volumes[0]= rcp(this, false);
    return volumes;
  }
  else
  {
    dserror("Volumes() for 1D-/2D-MeshfreeTransport element not implemented");
    return DRT::Element::Volumes();
  }
}


/*--------------------------------------------------------------------------*
 |  Pack data (public)                                            nis Jan12 |
 *--------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransport::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Cell
  DRT::MESHFREE::Cell::Pack(data);
  // numdofpernode
  AddtoPack(data,numdofpernode_);
  // distype
  AddtoPack(data,distype_);

  // data_
  AddtoPack(data,data_);

  return;
}


/*--------------------------------------------------------------------------*
 |  Unpack data                                           (public)nis Jan12 |
 *--------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransport::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  dsassert(type == UniqueParObjectId(), "wrong instance type data");
  // extract base class Cell
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  DRT::MESHFREE::Cell::Unpack(basedata);
  // numdofpernode
  ExtractfromPack(position,data,numdofpernode_);
  // distype
  distype_ = static_cast<DiscretizationType>( ExtractInt(position,data) );

  vector<char> tmp(0);
  ExtractfromPack(position,data,tmp);
  data_.Unpack(tmp);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}

/*--------------------------------------------------------------------------*
 |  Return names of visualization data                   (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransport::VisNames(map<string,int>& names)
{
  // Put the owner of this element into the file (use base class method for this)
  DRT::Element::VisNames(names);

  // see whether we have additional data for visualization in our container
  for (int k = 0 ;k<numdofpernode_; k++)
  {
    ostringstream temp;
    temp << k;

    // element Peclet number
    string name = "Pe_"+temp.str();
    const vector<double>* Pe = data_.Get<vector<double> >(name);
    if (Pe) names.insert(pair<string,int>(name,1));

    // element Peclet number (only migration term)
    name = "Pe_mig_"+temp.str();
    const vector<double>* Pe_mig = data_.Get<vector<double> >(name);
    if (Pe_mig) names.insert(pair<string,int>(name,1));

    //characteristic element length
    name = "hk_"+temp.str();
    const vector<double>* hk = data_.Get<vector<double> >(name);
    if (hk) names.insert(pair<string,int>(name,1));

    // Stabilization parameter at element center
    name = "tau_"+temp.str();
    const vector<double>* tau = data_.Get<vector<double> >(name);
    if (tau) names.insert(pair<string,int>(name,1));

  } // loop over transported scalars

  return;
}


/*--------------------------------------------------------------------------*
 |  Return visualization data                            (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
bool DRT::ELEMENTS::MeshfreeTransport::VisData(const string& name, vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if(DRT::Element::VisData(name,data))
    return true;

  for (int k = 0 ;k<numdofpernode_; k++)
  {
    ostringstream temp;
    temp << k;
    if (   (name == "Pe_"+temp.str()    )
        || (name == "Pe_mig_"+temp.str())
        || (name == "hk_"+temp.str()    )
        || (name == "tau_"+temp.str()   )
    )
    {
      if ((int)data.size()!=1) dserror("size mismatch");
      const double value = data_.GetDouble(name);
      data[0] = value;
      return true;
    }
  } // loop over transported scalars

  return false;
}

/*--------------------------------------------------------------------------*
 |  Read input for this element                          (public) nis Jan12 |
 *--------------------------------------------------------------------------*/
bool DRT::ELEMENTS::MeshfreeTransport::ReadElement(
  const std::string& eletype,
  const std::string& distype,
  DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT",material);
  SetMaterial(material);

  SetDisType(DRT::StringToDistype(distype));

  return true;
}

/*==========================================================================*\
 *                                                                          *
 * class MeshfreeTransportBoundaryType                                      *
 *                                                                          *
\*==========================================================================*/

/*--------------------------------------------------------------------------*
 | self-instantiation as parallel object type (?TODO: understand) nis Mar12 |
 *--------------------------------------------------------------------------*/
DRT::ELEMENTS::MeshfreeTransportBoundaryType DRT::ELEMENTS::MeshfreeTransportBoundaryType::instance_;

/*--------------------------------------------------------------------------*
 | creates meshfree node (?not really TODO: understand)  (public) nis Mar12 |
 *--------------------------------------------------------------------------*/
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::MeshfreeTransportBoundaryType::Create( const int id, const int owner )
{
  return Teuchos::null;
}

/*==========================================================================*\
 *                                                                          *
 * class MeshfreeTransportBoundary                                          *
 *                                                                          *
\*==========================================================================*/

/*---------------------------------------------------------------------------*
 |  ctor                                                  (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
DRT::ELEMENTS::MeshfreeTransportBoundary::MeshfreeTransportBoundary(int id, int owner,
                              int nnode, const int* nodeids,
                              DRT::Node** nodes,
                              DRT::ELEMENTS::MeshfreeTransport* parent,
                              const int lbeleid) :
DRT::MESHFREE::Cell(id,owner),
parent_(parent),
lbeleid_(lbeleid)
{
  SetNodeIds(nnode,nodeids);
  BuildNodalPointers(nodes);
  return;
}

/*---------------------------------------------------------------------------*
 |  copy-ctor                                             (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
DRT::ELEMENTS::MeshfreeTransportBoundary::MeshfreeTransportBoundary(const DRT::ELEMENTS::MeshfreeTransportBoundary& old) :
DRT::MESHFREE::Cell(old),
parent_(old.parent_),
lbeleid_(old.lbeleid_)
{
  return;
}

/*---------------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it          (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::MeshfreeTransportBoundary::Clone() const
{
  DRT::ELEMENTS::MeshfreeTransportBoundary* newelement = new DRT::ELEMENTS::MeshfreeTransportBoundary(*this);
  return newelement;
}

/*---------------------------------------------------------------------------*
 |  dtor                                                  (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
DRT::ELEMENTS::MeshfreeTransportBoundary::~MeshfreeTransportBoundary()
{
  return;
}

/*---------------------------------------------------------------------------*
 |  Return shape of this cell                             (public) nis Feb12 |
 *---------------------------------------------------------------------------*/
inline DRT::Element::DiscretizationType DRT::ELEMENTS::MeshfreeTransportBoundary::Shape() const
{
  return DRT::UTILS::getShapeOfBoundaryElement(NumNode(), parent_->Shape());
}

/*---------------------------------------------------------------------------*
 |  Return number of lines of boundary cell               (public) nis Feb12 |
 *---------------------------------------------------------------------------*/
inline int DRT::ELEMENTS::MeshfreeTransportBoundary::NumLine() const
{
  return DRT::UTILS::getNumberOfElementLines(Shape());
}

/*--------------------------------------------------------------------------*
 |  Return number of surfaces of boundary cell            (public) nis Feb12 |
 *---------------------------------------------------------------------------*/
inline int DRT::ELEMENTS::MeshfreeTransportBoundary::NumSurface() const
{
  return DRT::UTILS::getNumberOfElementSurfaces(Shape());
}

/*---------------------------------------------------------------------------*
 |  get vector of lines                                   (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::MeshfreeTransportBoundary::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  dserror("Lines of MeshfreeTransportBoundary not implemented");
  vector<RCP<DRT::Element> > lines(0);
  return lines;
}

/*---------------------------------------------------------------------------*
 |  get vector of lines                                   (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::MeshfreeTransportBoundary::Surfaces()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new surface elements:
  dserror("Surfaces of MeshfreeTransportBoundary not implemented");
  vector<RCP<DRT::Element> > surfaces(0);
  return surfaces;
}

/*---------------------------------------------------------------------------*
 |  Pack data (public)                                    (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransportBoundary::Pack(DRT::PackBuffer& data) const
{
  dserror("This TransportBoundary element does not support communication");

  return;
}

/*---------------------------------------------------------------------------*
 |  Unpack data (public)                                  (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransportBoundary::Unpack(const vector<char>& data)
{
  dserror("This TransportBoundary element does not support communication");
  return;
}

/*---------------------------------------------------------------------------*
 |  Return unique ParObject id                            (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
inline int DRT::ELEMENTS::MeshfreeTransportBoundary::UniqueParObjectId() const
{
  return MeshfreeTransportBoundaryType::Instance().UniqueParObjectId();
}

/*---------------------------------------------------------------------------*
 |  print this element                                    (public) nis Jan12 |
 *---------------------------------------------------------------------------*/
void DRT::ELEMENTS::MeshfreeTransportBoundary::Print(ostream& os) const
{
  os << "MeshfreeTransportBoundary element";
  Element::Print(os);
  cout << endl;
  cout << "DiscretizationType:  "<<Shape()<<endl;
  cout << endl;
  return;
}
