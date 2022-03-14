/*----------------------------------------------------------------------*/
/*! \file
\brief A 2D constraint element with no physics attached
\level 2


*----------------------------------------------------------------------*/

#include "constraint_element2.H"


DRT::ELEMENTS::ConstraintElement2Type DRT::ELEMENTS::ConstraintElement2Type::instance_;


DRT::ELEMENTS::ConstraintElement2Type& DRT::ELEMENTS::ConstraintElement2Type::Instance()
{
  return instance_;
}


DRT::ParObject* DRT::ELEMENTS::ConstraintElement2Type::Create(const std::vector<char>& data)
{
  DRT::ELEMENTS::ConstraintElement2* object = new DRT::ELEMENTS::ConstraintElement2(-1, -1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::ConstraintElement2Type::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "CONSTRELE2")
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::ConstraintElement2(id, owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::ConstraintElement2Type::Create(
    const int id, const int owner)
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::ConstraintElement2(id, owner));
  return ele;
}


void DRT::ELEMENTS::ConstraintElement2Type::NodalBlockInformation(
    DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
}

Epetra_SerialDenseMatrix DRT::ELEMENTS::ConstraintElement2Type::ComputeNullSpace(
    DRT::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Epetra_SerialDenseMatrix nullspace;
  dserror("method ComputeNullSpace not implemented!");
  return nullspace;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ConstraintElement2::ConstraintElement2(int id, int owner)
    : DRT::Element(id, owner), data_()
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ConstraintElement2::ConstraintElement2(const DRT::ELEMENTS::ConstraintElement2& old)
    : DRT::Element(old), data_(old.data_)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::ConstraintElement2::Clone() const
{
  DRT::ELEMENTS::ConstraintElement2* newelement = new DRT::ELEMENTS::ConstraintElement2(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ConstraintElement2::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);
  // add base class Element
  Element::Pack(data);

  // data_
  AddtoPack(data, data_);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ConstraintElement2::Unpack(const std::vector<char>& data)
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

  // data_
  std::vector<char> tmp(0);
  ExtractfromPack(position, data, tmp);
  data_.Unpack(tmp);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ConstraintElement2::~ConstraintElement2() { return; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ConstraintElement2::Print(std::ostream& os) const
{
  os << "ConstraintElement2 ";
  Element::Print(os);
  std::cout << std::endl;
  std::cout << data_;
  return;
}
