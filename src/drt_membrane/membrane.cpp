/*----------------------------------------------------------------------*/
/*! \file
\brief

\level 3


\brief Nonlinear Membrane Finite Element

*----------------------------------------------------------------------*/
#include "membrane.H"

#include "drt_utils_factory.H"
#include "str_elements_paramsinterface.H"
#include "so3_material.H"


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Membrane_line2Type::Create(const int id, const int owner)
{
  // return Teuchos::rcp( new MembraneLine( id, owner ) );
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Membrane_line3Type::Create(const int id, const int owner)
{
  // return Teuchos::rcp( new MembraneLine( id, owner ) );
  return Teuchos::null;
}

/*-----------------------------------------------------------------------------*
 |  constructor (public)                                          fbraeu 06/16 |
 |  id          (in)  this element's global id                                 |
 *-----------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::Membrane<distype>::Membrane(int id, int owner)
    : DRT::Element(id, owner),
      thickness_(0.0),
      cur_thickness_(0),
      data_(),
      planetype_(plane_stress),
      intpoints_(DRT::UTILS::GaussRule2D::tri_3point)
{
  switch (distype)
  {
    case tri3:
    {
      DRT::UTILS::GaussRule2D gaussrule = DRT::UTILS::GaussRule2D::tri_3point;
      // get gauss integration points
      intpoints_ = DRT::UTILS::IntegrationPoints2D(gaussrule);
      break;
    }
    case tri6:
    {
      DRT::UTILS::GaussRule2D gaussrule = DRT::UTILS::GaussRule2D::tri_6point;
      // get gauss integration points
      intpoints_ = DRT::UTILS::IntegrationPoints2D(gaussrule);
      break;
    }
    case quad4:
    {
      DRT::UTILS::GaussRule2D gaussrule = DRT::UTILS::GaussRule2D::quad_4point;
      // get gauss integration points
      intpoints_ = DRT::UTILS::IntegrationPoints2D(gaussrule);
      break;
    }
    case quad9:
    {
      DRT::UTILS::GaussRule2D gaussrule = DRT::UTILS::GaussRule2D::quad_9point;
      // get gauss integration points
      intpoints_ = DRT::UTILS::IntegrationPoints2D(gaussrule);
      break;
    }
    default:
      dserror("shape type unknown!\n");
      break;
  }
  cur_thickness_.resize(intpoints_.nquad, thickness_);
  return;
}

/*-----------------------------------------------------------------------------*
 |  copy-constructor (public)                                     fbraeu 06/16 |
 |  id               (in)  this element's global id                            |
 *-----------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::Membrane<distype>::Membrane(const DRT::ELEMENTS::Membrane<distype>& old)
    : DRT::Element(old),
      thickness_(old.thickness_),
      cur_thickness_(old.cur_thickness_),
      data_(old.data_),
      planetype_(old.planetype_),
      intpoints_(old.intpoints_)
{
  cur_thickness_.resize(intpoints_.nquad, thickness_);
  return;
}

/*------------------------------------------------------------------------*
 |  Deep copy this instance of Membrane and return pointer to it (public) |
 |                                                           fbraeu 06/16 |
 *------------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::Element* DRT::ELEMENTS::Membrane<distype>::Clone() const
{
  DRT::ELEMENTS::Membrane<distype>* newelement = new DRT::ELEMENTS::Membrane<distype>(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::Element::DiscretizationType DRT::ELEMENTS::Membrane<distype>::Shape() const
{
  return distype;
}

/*----------------------------------------------------------------------*
 |  Return number of lines of this element                     (public) |
 |                                                         fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::Membrane<distype>::NumLine() const
{
  return DRT::UTILS::getNumberOfElementLines(distype);
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::Membrane<distype>::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // add base class Element
  Element::Pack(data);

  // thickness_
  AddtoPack(data, thickness_);

  // current thickness_
  AddtoPack(data, cur_thickness_);

  // data_
  AddtoPack(data, data_);


  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::Membrane<distype>::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  Element::Unpack(basedata);
  // thickness_
  ExtractfromPack(position, data, thickness_);
  // current thickness_
  ExtractfromPack(position, data, cur_thickness_);
  // data_
  std::vector<char> tmp(0);
  ExtractfromPack(position, data, tmp);
  data_.Unpack(tmp);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}

/*----------------------------------------------------------------------*
 |  destructor (public)                                    fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::Membrane<distype>::~Membrane()
{
  return;
}

/*----------------------------------------------------------------------*
 |  return solid material (public)                         sfuchs 05/17 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
Teuchos::RCP<MAT::So3Material> DRT::ELEMENTS::Membrane<distype>::SolidMaterial(int nummat) const
{
  return Teuchos::rcp_dynamic_cast<MAT::So3Material>(DRT::Element::Material(nummat), true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::Membrane<distype>::SetParamsInterfacePtr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
    interface_ptr_ = p.get<Teuchos::RCP<DRT::ELEMENTS::ParamsInterface>>("interface");
  else
    interface_ptr_ = Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
Teuchos::RCP<DRT::ELEMENTS::ParamsInterface> DRT::ELEMENTS::Membrane<distype>::ParamsInterfacePtr()
{
  return interface_ptr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
STR::ELEMENTS::ParamsInterface& DRT::ELEMENTS::Membrane<distype>::StrParamsInterface()
{
  if (not IsParamsInterface()) dserror("The interface ptr is not set!");
  return *(Teuchos::rcp_dynamic_cast<STR::ELEMENTS::ParamsInterface>(interface_ptr_, true));
}

/*----------------------------------------------------------------------*
 |  print this element (public)                            fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::Membrane<distype>::Print(std::ostream& os) const
{
  os << "Membrane ";
  os << " Discretization type: " << DRT::DistypeToString(distype).c_str();
  Element::Print(os);
  std::cout << std::endl;
  std::cout << data_;
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                           fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::Membrane<distype>::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<MembraneLine<distype>, Membrane<distype>>(
      DRT::UTILS::buildLines, this);
}

/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                        fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::Membrane<distype>::Surfaces()
{
  std::vector<Teuchos::RCP<Element>> surfaces(1);
  surfaces[0] = Teuchos::rcp(this, false);
  return surfaces;
}

template class DRT::ELEMENTS::Membrane<DRT::Element::tri3>;
template class DRT::ELEMENTS::Membrane<DRT::Element::tri6>;
template class DRT::ELEMENTS::Membrane<DRT::Element::quad4>;
template class DRT::ELEMENTS::Membrane<DRT::Element::quad9>;
