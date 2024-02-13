/*-----------------------------------------------------------------------*/
/*! \file
\brief A class for a frictional contact node
\level 2
*/
/*-----------------------------------------------------------------------*/

#include "baci_contact_friction_node.hpp"

#include "baci_contact_defines.hpp"
#include "baci_contact_element.hpp"
#include "baci_utils_exceptions.hpp"

BACI_NAMESPACE_OPEN

CONTACT::FriNodeType CONTACT::FriNodeType::instance_;

CORE::COMM::ParObject* CONTACT::FriNodeType::Create(const std::vector<char>& data)
{
  std::vector<double> x(3, 0.0);
  std::vector<int> dofs(0);

  // TODO: friplus = true for all nodes!!! change this with pack/unpack
  CONTACT::FriNode* node = new CONTACT::FriNode(0, x, 0, dofs, false, false, true);
  node->Unpack(data);

  return node;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             mgit 01/10|
 *----------------------------------------------------------------------*/
CONTACT::FriNodeDataContainer::FriNodeDataContainer()
    : slip_(false), slipold_(false), drowsold_(0), drowsoldLTL_(0)
{
  for (int i = 0; i < 3; ++i)
  {
    jump()[i] = 0.0;
    traction()[i] = 0.0;
    tractionold()[i] = 0.0;
    tractionLTL()[i] = 0.0;
    tractionoldLTL()[i] = 0.0;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            mgit 01/10|
 *----------------------------------------------------------------------*/
void CONTACT::FriNodeDataContainer::Pack(CORE::COMM::PackBuffer& data) const
{
  // add jump_
  CORE::COMM::ParObject::AddtoPack(data, jump_, 3 * sizeof(double));
  // add slip_
  CORE::COMM::ParObject::AddtoPack(data, slip_);
  // add slip_
  CORE::COMM::ParObject::AddtoPack(data, slipold_);
  // add traction_
  CORE::COMM::ParObject::AddtoPack(data, traction_, 3 * sizeof(double));
  // add tractionold_
  CORE::COMM::ParObject::AddtoPack(data, tractionold_, 3 * sizeof(double));
  // add traction_
  CORE::COMM::ParObject::AddtoPack(data, tractionLTL_, 3 * sizeof(double));
  // add tractionold_
  CORE::COMM::ParObject::AddtoPack(data, tractionoldLTL_, 3 * sizeof(double));

  // add drowsold_,mrowsold_,mnodesold_
  int hasdata = drowsold_.size();

  CORE::COMM::ParObject::AddtoPack(data, hasdata);

  if (hasdata != 0)
  {
    int dentries = (int)drowsold_.size();
    CORE::COMM::ParObject::AddtoPack(data, dentries);
    CORE::COMM::ParObject::AddtoPack(data, drowsold_);
    CORE::COMM::ParObject::AddtoPack(data, mrowsold_);
    CORE::COMM::ParObject::AddtoPack(data, mnodesold_);
  }

  int hasdata2 = drowsoldLTL_.size();
  CORE::COMM::ParObject::AddtoPack(data, hasdata2);
  if (hasdata2 != 0)
  {
    int dentries = (int)drowsoldLTL_.size();
    CORE::COMM::ParObject::AddtoPack(data, dentries);
    CORE::COMM::ParObject::AddtoPack(data, drowsoldLTL_);
    CORE::COMM::ParObject::AddtoPack(data, mrowsoldLTL_);
  }
  // add derivjump
  int hasdataderivjump = derivjump_.size();
  CORE::COMM::ParObject::AddtoPack(data, hasdataderivjump);

  if (hasdataderivjump != 0)
  {
    for (int i = 0; i < hasdataderivjump; i++)
      CORE::COMM::ParObject::AddtoPack(data, (derivjump_[i]));
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            mgit 01/10|
 *----------------------------------------------------------------------*/
void CONTACT::FriNodeDataContainer::Unpack(
    std::vector<char>::size_type& position, const std::vector<char>& data)
{
  // jump_
  CORE::COMM::ParObject::ExtractfromPack(position, data, jump_, 3 * sizeof(double));
  // slip_
  slip_ = CORE::COMM::ParObject::ExtractInt(position, data);
  // slipold_
  slipold_ = CORE::COMM::ParObject::ExtractInt(position, data);
  // traction_
  CORE::COMM::ParObject::ExtractfromPack(position, data, traction_, 3 * sizeof(double));
  // tractionold_
  CORE::COMM::ParObject::ExtractfromPack(position, data, tractionold_, 3 * sizeof(double));
  // traction_
  CORE::COMM::ParObject::ExtractfromPack(position, data, tractionLTL_, 3 * sizeof(double));
  // tractionold_
  CORE::COMM::ParObject::ExtractfromPack(position, data, tractionoldLTL_, 3 * sizeof(double));

  // drowsold_,mrowsold_,mnodesold_
  int hasdata;
  CORE::COMM::ParObject::ExtractfromPack(position, data, hasdata);

  if (hasdata != 0)
  {
    int dentries = CORE::COMM::ParObject::ExtractInt(position, data);

    drowsold_.resize(dentries);
    CORE::COMM::ParObject::ExtractfromPack(position, data, drowsold_);
    CORE::COMM::ParObject::ExtractfromPack(position, data, mrowsold_);
    CORE::COMM::ParObject::ExtractfromPack(position, data, mnodesold_);
  }

  // drowsold_,mrowsold_,mnodesold_
  int hasdata2;
  CORE::COMM::ParObject::ExtractfromPack(position, data, hasdata2);

  if (hasdata2 != 0)
  {
    int dentries = CORE::COMM::ParObject::ExtractInt(position, data);

    drowsoldLTL_.resize(dentries);
    CORE::COMM::ParObject::ExtractfromPack(position, data, drowsoldLTL_);
    CORE::COMM::ParObject::ExtractfromPack(position, data, mrowsoldLTL_);
  }

  // and derivjump_
  int hasdataderivjump;
  CORE::COMM::ParObject::ExtractfromPack(position, data, hasdataderivjump);

  if (hasdataderivjump != 0)
  {
    derivjump_.resize(hasdataderivjump);
    for (int i = 0; i < hasdataderivjump; i++)
    {
      CORE::COMM::ParObject::ExtractfromPack(position, data, derivjump_[i]);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             mgit 07/11|
 *----------------------------------------------------------------------*/
CONTACT::FriNodeWearDataContainer::FriNodeWearDataContainer()
    : weightedwear_(0.0), deltaweightedwear_(0.0)
{
  wcurr_[0] = 0.0;
  wold_[0] = 0.0;
  waccu_[0] = 0.0;
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           farah 10/13|
 *----------------------------------------------------------------------*/
void CONTACT::FriNodeWearDataContainer::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::ParObject::AddtoPack(data, weightedwear_);
  CORE::COMM::ParObject::AddtoPack(data, deltaweightedwear_);

  // add d2row
  int hasdata = d2rows_.size();
  CORE::COMM::ParObject::AddtoPack(data, hasdata);

  if (hasdata != 0)
  {
    for (int i = 0; i < hasdata; i++)
    {
      CORE::COMM::ParObject::AddtoPack(data, (d2rows_[i]));
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           farah 10/13|
 *----------------------------------------------------------------------*/
void CONTACT::FriNodeWearDataContainer::Unpack(
    std::vector<char>::size_type& position, const std::vector<char>& data)
{
  CORE::COMM::ParObject::ExtractfromPack(position, data, weightedwear_);
  CORE::COMM::ParObject::ExtractfromPack(position, data, deltaweightedwear_);

  // d2rows_
  int hasdata;
  CORE::COMM::ParObject::ExtractfromPack(position, data, hasdata);

  if (hasdata != 0)
  {
    d2rows_.resize(hasdata);
    for (int i = 0; i < hasdata; i++)
    {
      CORE::COMM::ParObject::ExtractfromPack(position, data, d2rows_[i]);
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             mgit 02/10|
 *----------------------------------------------------------------------*/
CONTACT::FriNode::FriNode(int id, const std::vector<double>& coords, const int owner,
    const std::vector<int>& dofs, const bool isslave, const bool initactive, const bool friplus)
    : CONTACT::Node(id, coords, owner, dofs, isslave, initactive), wear_(friplus)
{
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        mgit 02/10|
 *----------------------------------------------------------------------*/
CONTACT::FriNode::FriNode(const CONTACT::FriNode& old) : CONTACT::Node(old), wear_(old.wear_)
{
  // not yet used and thus not necessarily consistent
  dserror("FriNode copy-ctor not yet implemented");

  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of FriNode and return pointer to it (public)|
 |                                                            mgit 02/10|
 *----------------------------------------------------------------------*/
CONTACT::FriNode* CONTACT::FriNode::Clone() const
{
  CONTACT::FriNode* newnode = new CONTACT::FriNode(*this);
  return newnode;
}

/*----------------------------------------------------------------------*
 |  << operator                                               mgit 02/10|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const CONTACT::FriNode& frinode)
{
  frinode.Print(os);
  return os;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                               mgit 02/10|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::Print(std::ostream& os) const
{
  // Print id and coordinates
  os << "Contact ";
  CONTACT::Node::Print(os);
  if (IsSlave())
    if (IsInitActive()) os << " InitActive ";

  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            mgit 02/10|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // add base class MORTAR::Node
  CONTACT::Node::Pack(data);

  // add data_
  bool hasdata = (fridata_ != Teuchos::null);
  AddtoPack(data, hasdata);
  if (hasdata) fridata_->Pack(data);

  bool hasweardata = (weardata_ != Teuchos::null);
  AddtoPack(data, hasweardata);
  if (hasweardata) weardata_->Pack(data);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            mgit 02/10|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class CONTACT::Node
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  CONTACT::Node::Unpack(basedata);

  // **************************
  // FriData
  bool hasdata = ExtractInt(position, data);
  if (hasdata)
  {
    fridata_ = Teuchos::rcp(new CONTACT::FriNodeDataContainer());
    fridata_->Unpack(position, data);
  }
  else
    fridata_ = Teuchos::null;

  // **************************
  // FriDataPlus
  bool hasdataplus = ExtractInt(position, data);
  if (hasdataplus)
  {
    weardata_ = Teuchos::rcp(new CONTACT::FriNodeWearDataContainer());
    weardata_->Unpack(position, data);
  }
  else
    weardata_ = Teuchos::null;

  // Check
  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d", (int)data.size(), position);
  return;
}

/*----------------------------------------------------------------------*
 | calculate the apparent coefficient of friction            seitz 11/15|
 *----------------------------------------------------------------------*/
double CONTACT::FriNode::FrCoeff(const double& frcoeff_in)
{
  // return the friction coefficient, if we do not have a TSI problem
  if (cTSIdata_ == Teuchos::null) return frcoeff_in;

  // in TSI case, the friction coefficient is temperature dependent
  else
  {
    double maxT = std::max(TSIData().Temp(), TSIData().TempMaster());
    return frcoeff_in * (maxT - cTSIdata_->Temp_Dam()) * (maxT - cTSIdata_->Temp_Dam()) /
           ((cTSIdata_->Temp_Dam() - cTSIdata_->Temp_Ref()) *
               (cTSIdata_->Temp_Dam() - cTSIdata_->Temp_Ref()));
  }
}

/*----------------------------------------------------------------------*
 | calculate derivative of apparent coefficient of friction  seitz 11/15|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::derivFrCoeffTemp(
    const double& frcoeff_in, std::map<int, double>& derivT, std::map<int, double>& derivDisp)
{
  derivT.clear();
  derivDisp.clear();

  // if we do not have a TSI problem, the friction coefficient is constant
  if (cTSIdata_ == Teuchos::null) return;

  double T_dam = cTSIdata_->Temp_Dam();
  double T_ref = cTSIdata_->Temp_Ref();
  if (cTSIdata_->Temp() > cTSIdata_->TempMaster())
  {
    double maxT = TSIData().Temp();
    derivT[Dofs()[0]] += 2. * frcoeff_in * (maxT - T_dam) / ((T_dam - T_ref) * (T_dam - T_ref));
    derivDisp.clear();
  }
  else
  {
    double maxT = TSIData().TempMaster();
    for (std::map<int, double>::const_iterator i = TSIData().DerivTempMasterTemp().begin();
         i != TSIData().DerivTempMasterTemp().end(); ++i)
      derivT[i->first] +=
          2. * frcoeff_in * (maxT - T_dam) / ((T_dam - T_ref) * (T_dam - T_ref)) * i->second;
    for (std::map<int, double>::const_iterator i = TSIData().DerivTempMasterDisp().begin();
         i != TSIData().DerivTempMasterDisp().end(); ++i)
      derivDisp[i->first] +=
          2. * frcoeff_in * (maxT - T_dam) / ((T_dam - T_ref) * (T_dam - T_ref)) * i->second;
  }
}

/*----------------------------------------------------------------------*
 |  Add a value to the 'SNodes' set                        gitterle 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::AddSNode(int node)
{
  FriData().GetSNodes().insert(node);

  return;
}

/*----------------------------------------------------------------------*
 |  Add a value to the 'MNodes' set                        gitterle 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::AddMNode(int node)
{
  FriData().GetMNodes().insert(node);

  return;
}

/*----------------------------------------------------------------------*
 |  Add a value to the 'D2' map                              farah 06/13|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::AddD2Value(int& row, int& col, double& val)
{
  // check if this is a master node or slave boundary node
  if (IsSlave() == true) dserror("AddD2Value: function called for slave node %i", Id());

  // check if this has been called before
  if ((int)WearData().GetD2().size() == 0) WearData().GetD2().resize(NumDof());

  // check row index input
  if ((int)WearData().GetD2().size() <= row)
    dserror("AddD2Value: tried to access invalid row index!");

  // add the pair (col,val) to the given row
  std::map<int, double>& d2map = WearData().GetD2()[row];
  d2map[col] += val;

  return;
}

/*----------------------------------------------------------------------*
 |  Add a value to the 'DerivJump' map                     gitterle 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::AddDerivJumpValue(int& row, const int& col, double val)
{
  // check if this is a master node or slave boundary node
  if (IsSlave() == false) dserror("AddJumpValue: function called for master node %i", Id());
  if (IsOnBound() == true) dserror("AddJumpValue: function called for boundary node %i", Id());

  // check if this has been called before
  if ((int)FriData().GetDerivJump().size() == 0) FriData().GetDerivJump().resize(NumDof());

  // check row index input
  if ((int)FriData().GetDerivJump().size() <= row)
    dserror("AddDerivJumpValue: tried to access invalid row index!");

  // add the pair (col,val) to the given row
  std::map<int, double>& zmap = FriData().GetDerivJump()[row];
  zmap[col] += val;

  return;
}

/*----------------------------------------------------------------------*
|  Add jump value from gp-wise integration of slip          farah 08/13|
*----------------------------------------------------------------------*/
void CONTACT::FriNode::AddJumpValue(double val, int k)
{
  // check if this is a master node or slave boundary node
  if (IsSlave() == false) dserror("AddJumpValue: function called for master node %i", Id());
  if (IsOnBound() == true) dserror("AddJumpValue: function called for boundary node %i", Id());

  FriData().jump_var()[k] += val;

  return;
}

/*----------------------------------------------------------------------*
 |  Add a value to the 'T' map                               farah 09/13|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::AddTValue(int& row, int& col, double& val)
{
  // check if this is a master node or slave boundary node
  //  if (IsSlave()==false)
  //    dserror("AddTValue: function called for master node %i", Id());

  // check if this has been called before
  if ((int)WearData().GetT().size() == 0) WearData().GetT().resize(NumDof());

  // check row index input
  if ((int)WearData().GetT().size() <= row)
    dserror("AddTValue: tried to access invalid row index!");

  // add the pair (col,val) to the given row
  std::map<int, double>& tmap = WearData().GetT()[row];
  tmap[col] += val;

  return;
}

/*----------------------------------------------------------------------*
 |  Add a value to the 'E' map                               farah 09/13|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::AddEValue(int& row, int& col, double& val)
{
  // check if this is a master node or slave boundary node
  //  if (IsSlave()==false)
  //    dserror("AddEValue: function called for master node %i", Id());

  // check if this has been called before
  if ((int)WearData().GetE().size() == 0) WearData().GetE().resize(NumDof());

  // check row index input
  if ((int)WearData().GetE().size() <= row)
    dserror("AddEValue: tried to access invalid row index!");

  // add the pair (col,val) to the given row
  std::map<int, double>& emap = WearData().GetE()[row];
  emap[col] += val;

  return;
}

/*----------------------------------------------------------------------*
 |  Store nodal entries of D and M to old ones             gitterle 12/08|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::StoreDMOld()
{
  // clear and zero nodal vectors
  FriData().GetDOld().clear();
  FriData().GetMOld().clear();
  FriData().GetDOldLTL().clear();
  FriData().GetMOldLTL().clear();

  // write drows_ to drowsold_
  FriData().GetDOld() = MoData().GetD();
  FriData().GetMOld() = MoData().GetM();

  // write drows_ to drowsold_
  FriData().GetDOldLTL() = MoData().GetDltl();
  FriData().GetMOldLTL() = MoData().GetMltl();

  // also vectors containing the according master nodes
  FriData().GetMNodesOld().clear();
  FriData().GetMNodesOld() = FriData().GetMNodes();

  return;
}

/*----------------------------------------------------------------------*
 |  Store nodal entries penalty tractions to old ones      gitterle 10/09|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::StoreTracOld()
{
  // write entries to old ones
  for (int j = 0; j < 3; ++j) FriData().tractionold()[j] = FriData().traction()[j];

  for (int j = 0; j < 3; ++j) FriData().tractionoldLTL()[j] = FriData().tractionLTL()[j];

  return;
}

/*-----------------------------------------------------------------------*
 |  Set the value of deltawear                             gitterle 12/10|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::AddDeltaWeightedWearValue(double& val)
{
  // add given value to deltawear_
  WearData().DeltaWeightedWear() += val;
}

/*----------------------------------------------------------------------*
 |  Initialize data container                             gitterle 10/09|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::InitializeDataContainer()
{
  // get maximum size of lin vectors
  linsize_ = 0;
  for (int i = 0; i < NumElement(); ++i)
    for (int j = 0; j < Elements()[i]->NumNode(); ++j)
      linsize_ += Elements()[i]->NumDofPerNode(*(Elements()[i]->Nodes()[j]));

  // get maximum size of lin vectors
  dentries_ = 0;
  for (int i = 0; i < NumElement(); ++i)
    for (int j = 0; j < Elements()[i]->NumNode(); ++j)
      dentries_ += Elements()[i]->NumDofPerNode(*(Elements()[i]->Nodes()[j]));

  // only initialize if not yet done
  if (modata_ == Teuchos::null && codata_ == Teuchos::null && fridata_ == Teuchos::null)
  {
    modata_ = Teuchos::rcp(new MORTAR::NodeDataContainer());
    codata_ = Teuchos::rcp(new CONTACT::NodeDataContainer());
    fridata_ = Teuchos::rcp(new CONTACT::FriNodeDataContainer());
  }

  // initialize data container for wear and tsi problems
  if (wear_ == true)
  {
    if (weardata_ == Teuchos::null)
      weardata_ = Teuchos::rcp(new CONTACT::FriNodeWearDataContainer());
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Reset data container                                      popp 09/10|
 *----------------------------------------------------------------------*/
void CONTACT::FriNode::ResetDataContainer()
{
  // reset to Teuchos::null
  fridata_ = Teuchos::null;
  weardata_ = Teuchos::null;
  codata_ = Teuchos::null;
  modata_ = Teuchos::null;

  return;
}

BACI_NAMESPACE_CLOSE
