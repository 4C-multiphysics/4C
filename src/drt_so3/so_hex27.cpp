/*!----------------------------------------------------------------------
\file so_hex27.cpp
\brief

<pre>
Maintainer: Thomas Kloeppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET

#include "so_hex27.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_mat/contchainnetw.H"
#include "../drt_mat/artwallremod.H"
#include "../drt_mat/viscoanisotropic.H"
#include "../drt_mat/anisotropic_balzani.H"
#include "../drt_mat/holzapfelcardiovascular.H"
#include "../drt_mat/humphreycardiovascular.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"


DRT::ELEMENTS::So_hex27Type DRT::ELEMENTS::So_hex27Type::instance_;


DRT::ParObject* DRT::ELEMENTS::So_hex27Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::So_hex27* object = new DRT::ELEMENTS::So_hex27(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex27Type::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="SOLIDH27" )
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So_hex27(id,owner));
    return ele;
  }
  return Teuchos::null;
}

DRT::ELEMENTS::Soh27RegisterType DRT::ELEMENTS::Soh27RegisterType::instance_;


DRT::ParObject* DRT::ELEMENTS::Soh27RegisterType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::Soh27Register* object =
    new DRT::ELEMENTS::Soh27Register(DRT::Element::element_so_hex27);
  object->Unpack(data);
  return object;
}



/*----------------------------------------------------------------------*
 |  ctor (public)                                                       |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex27::So_hex27(int id, int owner) :
DRT::Element(id,element_so_hex27,owner),
data_()
{
  kintype_ = soh27_totlag;
  invJ_.resize(NUMGPT_SOH27, LINALG::Matrix<NUMDIM_SOH27,NUMDIM_SOH27>(true));
  detJ_.resize(NUMGPT_SOH27, 0.0);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                                  |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex27::So_hex27(const DRT::ELEMENTS::So_hex27& old) :
DRT::Element(old),
kintype_(old.kintype_),
data_(old.data_),
detJ_(old.detJ_)
{
  invJ_.resize(old.invJ_.size());
  for (int i=0; i<(int)invJ_.size(); ++i)
  {
    // can this size be anything but NUMDIM_SOH27 x NUMDIM_SOH27?
    invJ_[i] = old.invJ_[i];
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::So_hex27::Clone() const
{
  DRT::ELEMENTS::So_hex27* newelement = new DRT::ELEMENTS::So_hex27(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::So_hex27::Shape() const
{
  return hex27;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex27::Pack(vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  vector<char> basedata(0);
  Element::Pack(basedata);
  AddtoPack(data,basedata);
  // kintype_
  AddtoPack(data,kintype_);
  // data_
  vector<char> tmp(0);
  data_.Pack(tmp);
  AddtoPack(data,tmp);

  // detJ_
  AddtoPack(data,detJ_);

  // invJ_
  const int size = (int)invJ_.size();
  AddtoPack(data,size);
  for (int i=0; i<size; ++i)
    AddtoPack(data,invJ_[i]);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex27::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);
  // kintype_
  ExtractfromPack(position,data,kintype_);
  // data_
  vector<char> tmp(0);
  ExtractfromPack(position,data,tmp);
  data_.Unpack(tmp);

  // detJ_
  ExtractfromPack(position,data,detJ_);
  // invJ_
  int size = 0;
  ExtractfromPack(position,data,size);
  invJ_.resize(size,LINALG::Matrix<NUMDIM_SOH27,NUMDIM_SOH27>(true) );
  for (int i=0; i<size; ++i)
    ExtractfromPack(position,data,invJ_[i]);


  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*
 |  dtor (public)                                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex27::~So_hex27()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                                         |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex27::Print(ostream& os) const
{
  os << "So_hex27 ";
  Element::Print(os);
  cout << endl;
  cout << data_;
  return;
}


/*----------------------------------------------------------------------*
 |  extrapolation of quantities at the GPs to the nodes      popp 12/09 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex27::soh27_expol
(
    LINALG::Matrix<NUMGPT_SOH27,NUMSTR_SOH27>& stresses,
    LINALG::Matrix<NUMNOD_SOH27,NUMSTR_SOH27>& nodalstresses
)
{
  static LINALG::Matrix<NUMNOD_SOH27,NUMGPT_SOH27> expol;
  static bool isfilled;

  if (isfilled==true)
  {
    //nodalstresses.Multiply('N','N',1.0,expol,stresses,0.0);
    nodalstresses.Multiply(expol, stresses);
  }
  else
  {
    // get gaussian points
    const DRT::UTILS::IntegrationPoints3D intpoints(DRT::UTILS::intrule_hex_27point);

    // loop over all nodes
    for (int ip=0; ip<NUMNOD_SOH27; ++ip)
    {
      // gaussian coordinates
      const double e1 = intpoints.qxg[ip][0];
      const double e2 = intpoints.qxg[ip][1];
      const double e3 = intpoints.qxg[ip][2];

      // coordinates of node in the fictitious GP element
      double e1expol;
      double e2expol;
      double e3expol;

      if (e1!=0) e1expol = 1/e1;
      else       e1expol = 0;
      if (e2!=0) e2expol = 1/e2;
      else       e2expol = 0;
      if (e3!=0) e3expol = 1/e3;
      else       e3expol = 0;

      // shape functions for the extrapolated coordinates
      LINALG::Matrix<NUMGPT_SOH27,1> funct;
      DRT::UTILS::shape_function_3D(funct,e1expol,e2expol,e3expol,hex27);

      // extrapolation matrix
      for (int i=0;i<NUMGPT_SOH27;++i) expol(ip,i) = funct(i);
    }

    // do extrapolation
    nodalstresses.Multiply(expol, stresses);

    isfilled = true;
  }
}

/*----------------------------------------------------------------------*
 |  allocate and return So_hex27Register (public)                       |
 *----------------------------------------------------------------------*/
RefCountPtr<DRT::ElementRegister> DRT::ELEMENTS::So_hex27::ElementRegister() const
{
  return rcp(new DRT::ELEMENTS::Soh27Register(Type()));
}


/*----------------------------------------------------------------------*
 |  get vector of volumes (length 1) (public)                           |
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::So_hex27::Volumes()
{
  vector<RCP<Element> > volumes(1);
  volumes[0]= rcp(this, false);
  return volumes;
}

 /*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                                      |
 |  surface normals always point outward                                 |
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::So_hex27::Surfaces()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new surface elements:
  return DRT::UTILS::ElementBoundaryFactory<StructuralSurface,DRT::Element>(DRT::UTILS::buildSurfaces,this);
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                                        |
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::So_hex27::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<StructuralLine,DRT::Element>(DRT::UTILS::buildLines,this);
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                         |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex27::VisNames(map<string,int>& names)
{
  // Put the owner of this element into the file (use base class method for this)
  DRT::Element::VisNames(names);

  if (Material()->MaterialType() == INPAR::MAT::m_contchainnetw){
    string fiber = "Fiber1";
    names[fiber] = 3; // 3-dim vector
    fiber = "Fiber2";
    names[fiber] = 3; // 3-dim vector
    fiber = "Fiber3";
    names[fiber] = 3; // 3-dim vector
    fiber = "Fiber4";
    names[fiber] = 3; // 3-dim vector
    fiber = "FiberCell1";
    names[fiber] = 3; // 3-dim vector
    fiber = "FiberCell2";
    names[fiber] = 3; // 3-dim vector
    fiber = "FiberCell3";
    names[fiber] = 3; // 3-dim vector
    fiber = "l1";
    names[fiber] = 1;
    fiber = "l2";
    names[fiber] = 1;
    fiber = "l3";
    names[fiber] = 1;
//    fiber = "l1_0";
//    names[fiber] = 1;
//    fiber = "l2_0";
//    names[fiber] = 1;
//    fiber = "l3_0";
//    names[fiber] = 1;
  }
  if ((Material()->MaterialType() == INPAR::MAT::m_artwallremod) ||
      (Material()->MaterialType() == INPAR::MAT::m_viscoanisotropic) ||
      (Material()->MaterialType() == INPAR::MAT::m_holzapfelcardiovascular))
  {
    string fiber = "Fiber1";
    names[fiber] = 3; // 3-dim vector
    fiber = "Fiber2";
    names[fiber] = 3; // 3-dim vector
  }
  if (Material()->MaterialType() == INPAR::MAT::m_anisotropic_balzani){
    string fiber = "Fiber1";
    names[fiber] = 3; // 3-dim vector
    fiber = "Fiber2";
    names[fiber] = 3; // 3-dim vector
  }
  if (Material()->MaterialType() == INPAR::MAT::m_humphreycardiovascular)
  {
    string fiber = "Fiber1";
    names[fiber] = 3; // 3-dim vector
    fiber = "Fiber2";
    names[fiber] = 3;
    fiber = "Fiber3";
    names[fiber] = 3;
    fiber = "Fiber4";
    names[fiber] = 3;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                                  |
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::So_hex27::VisData(const string& name, vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if(DRT::Element::VisData(name,data))
    return true;

  if (Material()->MaterialType() == INPAR::MAT::m_contchainnetw){
    RefCountPtr<MAT::Material> mat = Material();
    MAT::ContChainNetw* chain = static_cast <MAT::ContChainNetw*>(mat.get());
    if (!chain->Initialized()){
      data[0] = 0.0; data[1] = 0.0; data[2] = 0.0;
    } else {
      RCP<vector<vector<double> > > gplis = chain->Getli();
      RCP<vector<vector<double> > > gpli0s = chain->Getli0();
      RCP<vector<LINALG::Matrix<3,3> > > gpnis = chain->Getni();

      vector<double> centerli (3,0.0);
      vector<double> centerli_0 (3,0.0);
      for (int i = 0; i < (int)gplis->size(); ++i) {
        LINALG::Matrix<3,1> loc(&(gplis->at(i)[0]));
        //Epetra_SerialDenseVector loc(CV,&(gplis->at(i)[0]),3);
        LINALG::Matrix<3,1> glo;
        //glo.Multiply('N','N',1.0,gpnis->at(i),loc,0.0);
        glo.Multiply(gpnis->at(i),loc);
        // Unfortunately gpnis is a vector of Epetras, to change this
        // I must begin at a deeper level...
        centerli[0] += glo(0);
        centerli[1] += glo(1);
        centerli[2] += glo(2);

//        centerli[0] += gplis->at(i)[0];
//        centerli[1] += gplis->at(i)[1];
//        centerli[2] += gplis->at(i)[2];
//
        centerli_0[0] += gplis->at(i)[0];
        centerli_0[1] += gplis->at(i)[1];
        centerli_0[2] += gplis->at(i)[2];
      }
      centerli[0] /= gplis->size();
      centerli[1] /= gplis->size();
      centerli[2] /= gplis->size();

      centerli_0[0] /= gplis->size();
      centerli_0[1] /= gplis->size();
      centerli_0[2] /= gplis->size();

      // just the unit cell of the first gp
      int gp = 0;
//      Epetra_SerialDenseVector loc(CV,&(gplis->at(gp)[0]),3);
//      Epetra_SerialDenseVector glo(3);
//      glo.Multiply('N','N',1.0,gpnis->at(gp),loc,0.0);
      LINALG::Matrix<3,3> T(gpnis->at(gp).A(),true);
      vector<double> gpli =  chain->Getli()->at(gp);

      if (name == "Fiber1"){
        if ((int)data.size()!=3) dserror("size mismatch");
        data[0] = centerli[0]; data[1] = -centerli[1]; data[2] = -centerli[2];
      } else if (name == "Fiber2"){
        data[0] = centerli[0]; data[1] = centerli[1]; data[2] = -centerli[2];
      } else if (name == "Fiber3"){
        data[0] = centerli[0]; data[1] = centerli[1]; data[2] = centerli[2];
      } else if (name == "Fiber4"){
        data[0] = -centerli[0]; data[1] = -centerli[1]; data[2] = centerli[2];
      } else if (name == "FiberCell1"){
        LINALG::Matrix<3,1> e(true);
        e(0) = gpli[0];
        LINALG::Matrix<3,1> glo;
        //glo.Multiply('N','N',1.0,T,e,0.0);
        glo.Multiply(T, e);
        data[0] = glo(0); data[1] = glo(1); data[2] = glo(2);
      } else if (name == "FiberCell2"){
        LINALG::Matrix<3,1> e(true);
        e(1) = gpli[1];
        LINALG::Matrix<3,1> glo;
        //glo.Multiply('N','N',1.0,T,e,0.0);
        glo.Multiply(T, e);
        data[0] = glo(0); data[1] = glo(1); data[2] = glo(2);
      } else if (name == "FiberCell3"){
        LINALG::Matrix<3,1> e(true);
        e(2) = gpli[2];
        LINALG::Matrix<3,1> glo;
        //glo.Multiply('N','N',1.0,T,e,0.0);
        glo.Multiply(T, e);
        data[0] = glo(0); data[1] = glo(1); data[2] = glo(2);
      } else if (name == "l1"){
        data[0] = centerli_0[0];
      } else if (name == "l2"){
        data[0] = centerli_0[1];
      } else if (name == "l3"){
        data[0] = centerli_0[2];
//      } else if (name == "l1_0"){
//        data[0] = centerli_0[0];
//      } else if (name == "l2_0"){
//        data[0] = centerli_0[1];
//      } else if (name == "l3_0"){
//        data[0] = centerli_0[2];
      } else {
        return false;
      }
    }
  }
  if (Material()->MaterialType() == INPAR::MAT::m_artwallremod){
    MAT::ArtWallRemod* art = static_cast <MAT::ArtWallRemod*>(Material().get());
    vector<double> a1 = art->Geta1()->at(0);  // get a1 of first gp
    vector<double> a2 = art->Geta2()->at(0);  // get a2 of first gp
    if (name == "Fiber1"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a1[0]; data[1] = a1[1]; data[2] = a1[2];
    } else if (name == "Fiber2"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a2[0]; data[1] = a2[1]; data[2] = a2[2];
    } else {
      return false;
    }
  }
  if (Material()->MaterialType() == INPAR::MAT::m_viscoanisotropic){
    MAT::ViscoAnisotropic* art = static_cast <MAT::ViscoAnisotropic*>(Material().get());
    vector<double> a1 = art->Geta1()->at(0);  // get a1 of first gp
    vector<double> a2 = art->Geta2()->at(0);  // get a2 of first gp
    if (name == "Fiber1"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a1[0]; data[1] = a1[1]; data[2] = a1[2];
    } else if (name == "Fiber2"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a2[0]; data[1] = a2[1]; data[2] = a2[2];
    } else {
      return false;
    }
  }
  if (Material()->MaterialType() == INPAR::MAT::m_anisotropic_balzani){
    MAT::AnisotropicBalzani* balz = static_cast <MAT::AnisotropicBalzani*>(Material().get());
    if (name == "Fiber1"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = balz->Geta1().at(0); data[1] = balz->Geta1().at(1); data[2] = balz->Geta1().at(2);
    } else if (name == "Fiber2"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = balz->Geta2().at(0); data[1] = balz->Geta2().at(1); data[2] = balz->Geta2().at(2);
    } else {
      return false;
    }
  }
  if (Material()->MaterialType() == INPAR::MAT::m_holzapfelcardiovascular){
    MAT::HolzapfelCardio* art = static_cast <MAT::HolzapfelCardio*>(Material().get());
    vector<double> a1 = art->Geta1()->at(0);  // get a1 of first gp
    vector<double> a2 = art->Geta2()->at(0);  // get a2 of first gp
    if (name == "Fiber1"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a1[0]; data[1] = a1[1]; data[2] = a1[2];
    } else if (name == "Fiber2"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a2[0]; data[1] = a2[1]; data[2] = a2[2];
    } else {
      return false;
    }
  }
  if (Material()->MaterialType() == INPAR::MAT::m_humphreycardiovascular){
    MAT::HumphreyCardio* art = static_cast <MAT::HumphreyCardio*>(Material().get());
    vector<double> a1 = art->Geta1()->at(0);  // get a1 of first gp
    vector<double> a2 = art->Geta2()->at(0);  // get a2 of first gp
    vector<double> a3 = art->Geta3()->at(0);  // get a3 of first gp
    vector<double> a4 = art->Geta4()->at(0);  // get a4 of first gp
    if (name == "Fiber1"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a1[0]; data[1] = a1[1]; data[2] = a1[2];
    } else if (name == "Fiber2"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a2[0]; data[1] = a2[1]; data[2] = a2[2];
    } else if (name == "Fiber3"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a3[0]; data[1] = a3[1]; data[2] = a3[2];
    } else if (name == "Fiber4"){
      if ((int)data.size()!=3) dserror("size mismatch");
      data[0] = a4[0]; data[1] = a4[1]; data[2] = a4[2];
    } else {
      return false;
    }
  }

  return true;
}



//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================

/*----------------------------------------------------------------------*
 |  ctor (public)                                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Soh27Register::Soh27Register(DRT::Element::ElementType etype) :
ElementRegister(etype)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                                  |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Soh27Register::Soh27Register(
                               const DRT::ELEMENTS::Soh27Register& old) :
ElementRegister(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Soh27Register* DRT::ELEMENTS::Soh27Register::Clone() const
{
  return new DRT::ELEMENTS::Soh27Register(*this);
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Soh27Register::Pack(vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class ElementRegister
  vector<char> basedata(0);
  ElementRegister::Pack(basedata);
  AddtoPack(data,basedata);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Soh27Register::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // base class ElementRegister
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  ElementRegister::Unpack(basedata);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*
 |  dtor (public)                                                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Soh27Register::~Soh27Register()
{
  return;
}

/*----------------------------------------------------------------------*
 |  print (public)                                                      |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Soh27Register::Print(ostream& os) const
{
  os << "Soh27Register ";
  ElementRegister::Print(os);
  return;
}

#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_SOLID3

