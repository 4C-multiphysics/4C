// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_node.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

CONTACT::NodeType CONTACT::NodeType::instance_;


Core::Communication::ParObject* CONTACT::NodeType::create(Core::Communication::UnpackBuffer& buffer)
{
  std::vector<double> x(3, 0.0);
  std::vector<int> dofs;
  auto* node = new CONTACT::Node(0, x, 0, dofs, false, false);
  node->unpack(buffer);
  return node;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mgit 02/10 |
 *----------------------------------------------------------------------*/
CONTACT::NodeDataContainer::NodeDataContainer()
    : grow_(1.0e12),
      gnts_(1.0e12),
      glts_(1.0e12),
      activeold_(false),
      derivn_(0, 0),     // init deriv normal to length 0 with 0 entries per direction
      derivtxi_(0, 0),   // init deriv txi    to length 0 with 0 entries per direction
      derivteta_(0, 0),  // init deriv teta   to length 0 with 0 entries per direction
      alpha_(0),
      kappa_(1.0)
{
  // set all tangent entries to 0.0
  for (int i = 0; i < 3; ++i)
  {
    txi()[i] = 0.0;
    teta()[i] = 0.0;
    getgltl()[i] = 1.0e12;
    getjumpltl()[i] = 1.0e12;
  }
  get_deriv_gltl().resize(3);
  get_deriv_jumpltl().resize(3);
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            mgit 02/10|
 *----------------------------------------------------------------------*/
void CONTACT::NodeDataContainer::pack(Core::Communication::PackBuffer& data) const
{
  // add txi_
  add_to_pack(data, txi_, 3 * sizeof(double));
  // add teta_
  add_to_pack(data, teta_, 3 * sizeof(double));
  // add grow_
  add_to_pack(data, grow_);
  // add kappa_
  add_to_pack(data, kappa_);
  // add activeold_
  add_to_pack(data, activeold_);
  // add n_old_
  add_to_pack(data, n_old_, 3 * sizeof(double));

  // no need to pack derivs_
  // (these will evaluated anew anyway)

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            mgit 02/10|
 *----------------------------------------------------------------------*/
void CONTACT::NodeDataContainer::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // txi_
  extract_from_pack(buffer, txi_, 3 * sizeof(double));
  // teta_
  extract_from_pack(buffer, teta_, 3 * sizeof(double));
  // grow_
  extract_from_pack(buffer, grow_);
  // kappa_
  extract_from_pack(buffer, kappa_);
  // activeold_
  extract_from_pack(buffer, activeold_);
  // n_old_
  extract_from_pack(buffer, n_old_, 3 * sizeof(double));

  return;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                             ager 08/14|
 *----------------------------------------------------------------------*/
CONTACT::NodePoroDataContainer::NodePoroDataContainer()
{
  ncouprow_ = 0.0;
  for (int i = 0; i < 3; ++i)
  {
    fvel()[i] = 0.0;
    svel()[i] = 0.0;
    poro_lm()[i] = 0.0;
  }
  *fpres() = 0.0;
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            ager 08/14|
 *----------------------------------------------------------------------*/
void CONTACT::NodePoroDataContainer::pack(Core::Communication::PackBuffer& data) const
{
  // add fvel
  add_to_pack(data, fvel_, 3 * sizeof(double));
  // add fpres
  add_to_pack(data, fpres_);
  // add svel
  add_to_pack(data, svel_, 3 * sizeof(double));
  // add poroLM
  add_to_pack(data, porolm_, 3 * sizeof(double));
  // add ncoup
  add_to_pack(data, ncouprow_);
  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            ager 08/14|
 *----------------------------------------------------------------------*/
void CONTACT::NodePoroDataContainer::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // fvel
  extract_from_pack(buffer, fvel_, 3 * sizeof(double));
  // fpres
  extract_from_pack(buffer, fpres_);
  // svel
  extract_from_pack(buffer, svel_, 3 * sizeof(double));
  // poroLM
  extract_from_pack(buffer, porolm_, 3 * sizeof(double));
  // ncoup
  extract_from_pack(buffer, ncouprow_);
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            seitz 08/15|
 *----------------------------------------------------------------------*/
CONTACT::NodeTSIDataContainer::NodeTSIDataContainer(double t_ref, double t_dam)
    : temp_(-1.e12), t_ref_(t_ref), t_dam_(t_dam), thermo_lm_(0.), temp_target_(-1.e12)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           seitz 08/15|
 *----------------------------------------------------------------------*/
void CONTACT::NodeTSIDataContainer::pack(Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, temp_target_);
  add_to_pack(data, t_ref_);
  add_to_pack(data, t_dam_);
  add_to_pack(data, derivTempTargetDisp_);
  add_to_pack(data, derivTempTargetTemp_);
  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           seitz 08/15|
 *----------------------------------------------------------------------*/
void CONTACT::NodeTSIDataContainer::unpack(Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, temp_target_);
  extract_from_pack(buffer, t_ref_);
  extract_from_pack(buffer, t_dam_);
  extract_from_pack(buffer, derivTempTargetDisp_);
  extract_from_pack(buffer, derivTempTargetTemp_);
  return;
}

/*----------------------------------------------------------------------*
 |  clear data                                                 (public) |
 |                                                           seitz 08/15|
 *----------------------------------------------------------------------*/
void CONTACT::NodeTSIDataContainer::clear()
{
  temp_target_ = -1.e12;
  derivTempTargetDisp_.clear();
  derivTempTargetTemp_.clear();
  return;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::Node::Node(int id, std::span<const double> coords, const int owner,
    const std::vector<int>& dofs, const bool issource, const bool initactive)
    : Mortar::Node(id, coords, owner, dofs, issource),
      active_(false),
      initactive_(initactive),
      involvedm_(false),
      linsize_(0),  // length of linearization
      codata_(nullptr),
      coporodata_(nullptr),
      cTSIdata_(nullptr)
{
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::Node::Node(const CONTACT::Node& old)
    : Mortar::Node(old),
      active_(old.active_),
      initactive_(old.initactive_),
      involvedm_(false),
      linsize_(0),
      codata_(nullptr),
      coporodata_(nullptr),
      cTSIdata_(nullptr)
{
  // not yet used and thus not necessarily consistent
  FOUR_C_THROW("Node copy-ctor not yet implemented");

  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Node and return pointer to it (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::Node* CONTACT::Node::clone() const
{
  CONTACT::Node* newnode = new CONTACT::Node(*this);
  return newnode;
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const CONTACT::Node& cnode)
{
  cnode.print(os);
  return os;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                              mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Node::print(std::ostream& os) const
{
  // Print id and coordinates
  os << "Contact ";
  Mortar::Node::print(os);
  if (is_source())
    if (is_init_active()) os << " InitActive ";
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Node::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Mortar::Node
  Mortar::Node::pack(data);

  // add active_
  add_to_pack(data, active_);
  // add initactive_
  add_to_pack(data, initactive_);
  // add involved
  add_to_pack(data, involvedm_);
  // add linsize_
  add_to_pack(data, linsize_);
  // add data_
  bool hasdata = (codata_ != nullptr);
  add_to_pack(data, hasdata);
  if (hasdata) codata_->pack(data);

  // add porodata_
  bool hasdataporo = (coporodata_ != nullptr);
  add_to_pack(data, hasdataporo);
  if (hasdataporo) coporodata_->pack(data);

  // add tsidata
  bool hasTSIdata = (cTSIdata_ != nullptr);
  add_to_pack(data, hasTSIdata);
  if (hasTSIdata) cTSIdata_->pack(data);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Node::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Mortar::Node
  Mortar::Node::unpack(buffer);

  // active_
  extract_from_pack(buffer, active_);
  // issource_
  extract_from_pack(buffer, initactive_);
  // issource_
  extract_from_pack(buffer, involvedm_);
  // issource_
  extract_from_pack(buffer, linsize_);

  // data_
  bool hasdata;
  extract_from_pack(buffer, hasdata);
  if (hasdata)
  {
    codata_ = std::make_shared<CONTACT::NodeDataContainer>();
    codata_->unpack(buffer);
  }
  else
  {
    codata_ = nullptr;
  }

  // porodata_
  bool hasdataporo;
  extract_from_pack(buffer, hasdataporo);
  if (hasdataporo)
  {
    coporodata_ = std::make_shared<CONTACT::NodePoroDataContainer>();
    coporodata_->unpack(buffer);
  }
  else
  {
    coporodata_ = nullptr;
  }

  // TSI data
  bool hasTSIdata;
  extract_from_pack(buffer, hasTSIdata);
  if (hasTSIdata)
  {
    cTSIdata_ = std::make_shared<CONTACT::NodeTSIDataContainer>();
    cTSIdata_->unpack(buffer);
  }
  else
    cTSIdata_ = nullptr;


  return;
}

/*----------------------------------------------------------------------*
 |  Add a value to the weighted gap                           popp 01/08|
 *----------------------------------------------------------------------*/
void CONTACT::Node::addg_value(const double val)
{
  // check if this is a target node or source boundary node
  if (is_source() == false) FOUR_C_THROW("AddgValue: function called for target node {}", id());
  if (is_on_bound() == true) FOUR_C_THROW("AddgValue: function called for boundary node {}", id());

  // initialize if called for the first time
  if (data().getg() == 1.0e12) data().getg() = 0.0;

  // add given value to grow_
  data().getg() += val;
}


/*----------------------------------------------------------------------*
 |  Add a value to the nts gap                               farah 01/16|
 *----------------------------------------------------------------------*/
void CONTACT::Node::addnts_gap_value(const double val)
{
  // check if this is a target node or source boundary node
  // initialize if called for the first time
  if (data().getgnts() == 1.0e12) data().getgnts() = 0;

  // add given value to wGap_
  data().getgnts() += val;
}

/*----------------------------------------------------------------------*
 |  Add a value to the lts gap                               farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Node::addlts_gap_value(const double val)
{
  // initialize if called for the first time
  if (data().getglts() == 1.0e12) data().getglts() = 0;

  // add given value to wGap_
  data().getglts() += val;
}

/*----------------------------------------------------------------------*
 |  Add a value to the ltl gap                               farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Node::addltl_gap_value(const double* val)
{
  // check if this is a target node or source boundary node
  if (is_source() == false) FOUR_C_THROW("function called for target node {}", id());
  if (!is_on_edge()) FOUR_C_THROW("function call for non edge node! {}", id());

  // initialize if called for the first time
  if (data().getgltl()[0] == 1.0e12 or data().getgltl()[1] == 1.0e12 or
      data().getgltl()[2] == 1.0e12)
  {
    data().getgltl()[0] = 0.0;
    data().getgltl()[1] = 0.0;
    data().getgltl()[2] = 0.0;
  }

  // add given value to wGap_
  data().getgltl()[0] += val[0];
  data().getgltl()[1] += val[1];
  data().getgltl()[2] += val[2];
}

/*----------------------------------------------------------------------*
 |  Add a value to the ltl jump                               farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::Node::addltl_jump_value(const double* val)
{
  // check if this is a target node or source boundary node
  if (is_source() == false) FOUR_C_THROW("function called for target node {}", id());
  if (!is_on_edge()) FOUR_C_THROW("function call for non edge node! {}", id());

  // initialize if called for the first time
  if (data().getjumpltl()[0] == 1.0e12 or data().getjumpltl()[1] == 1.0e12 or
      data().getjumpltl()[2] == 1.0e12)
  {
    data().getjumpltl()[0] = 0.0;
    data().getjumpltl()[1] = 0.0;
    data().getjumpltl()[2] = 0.0;
  }

  // add given value to wGap_
  data().getjumpltl()[0] += val[0];
  data().getjumpltl()[1] += val[1];
  data().getjumpltl()[2] += val[2];
}


/*----------------------------------------------------------------------*
 |  Add a value to the 'DerivZ' map                           popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::Node::add_deriv_z_value(const int row, const int col, const double val)
{
  // check if this is a target node or source boundary node
  if (is_source() == false) FOUR_C_THROW("AddZValue: function called for target node {}", id());
  if (is_on_bound() == true) FOUR_C_THROW("AddZValue: function called for boundary node {}", id());

  // check if this has been called before
  if ((int)data().get_deriv_z().size() == 0) data().get_deriv_z().resize(num_dof());

  // check row index input
  if ((int)data().get_deriv_z().size() <= row)
    FOUR_C_THROW("AddDerivZValue: tried to access invalid row index!");

  // add the pair (col,val) to the given row
  std::map<int, double>& zmap = data().get_deriv_z()[row];
  zmap[col] += val;
}

/*----------------------------------------------------------------------*
 |  Initialize data container                             gitterle 02/10|
 *----------------------------------------------------------------------*/
void CONTACT::Node::initialize_data_container()
{
  // get maximum size of lin vectors
  linsize_ = 0;
  // get maximum size of nodal D-entries
  dentries_ = 0;
  std::set<int> sIdCheck;
  std::pair<std::set<int>::iterator, bool> check;
  for (auto ele : adjacent_elements())
  {
    for (auto node : ele.nodes())
    {
      const int numdof = ele.user_element()->num_dof_per_node(*(node.user_node()));

      check = sIdCheck.insert(node.global_id());
      if (check.second)
      {
        dentries_ += numdof;
      }
      linsize_ += numdof;
    }
  }
  // set a minimal number for pure ghost nodes
  if (num_element() == 0)
  {
    dentries_ = 3;
    linsize_ = 3;
  }

  // only initialize if not yet done
  if (modata_ == nullptr && codata_ == nullptr)
  {
    codata_ = std::make_shared<CONTACT::NodeDataContainer>();
    modata_ = std::make_shared<Mortar::NodeDataContainer>();
  }

  return;
}

/*-----------------------------------------------------------------------*
 |  Initialize poro data container                             ager 07/14|
 *----------------------------------------------------------------------*/
void CONTACT::Node::initialize_poro_data_container()
{
  // only initialize if not yet done

  if (coporodata_ == nullptr)
  {
    coporodata_ = std::make_shared<CONTACT::NodePoroDataContainer>();
  }

  return;
}

/*-----------------------------------------------------------------------*
 |  Initialize ehl data container                             seitz 11/17|
 *----------------------------------------------------------------------*/
void CONTACT::Node::initialize_ehl_data_container()
{
  // only initialize if not yet done

  if (cEHLdata_ == nullptr)
  {
    cEHLdata_ = std::make_shared<CONTACT::NodeEhlDataContainer>();
  }

  return;
}

/*-----------------------------------------------------------------------*
 |  Initialize TSI data container                             seitz 08/15|
 *----------------------------------------------------------------------*/
void CONTACT::Node::initialize_tsi_data_container(double t_ref, double t_dam)
{
  // only initialize if not yet done

  if (cTSIdata_ == nullptr)
    cTSIdata_ = std::make_shared<CONTACT::NodeTSIDataContainer>(t_ref, t_dam);

  return;
}

/*----------------------------------------------------------------------*
 |  Reset data container                                      popp 09/10|
 *----------------------------------------------------------------------*/
void CONTACT::Node::reset_data_container()
{
  // reset to nullptr
  codata_ = nullptr;
  modata_ = nullptr;
  coporodata_ = nullptr;

  return;
}

/*----------------------------------------------------------------------*
 |  Build averaged nodal edge tangents                       farah 11/16|
 *----------------------------------------------------------------------*/
void CONTACT::Node::build_averaged_edge_tangent()
{
  for (int j = 0; j < 3; ++j)
  {
    mo_data().edge_tangent()[j] = 0.0;
  }

  //**************************************************
  //              CALCULATE EDGES
  //**************************************************
  // empty vector of source element pointers
  std::vector<std::shared_ptr<Mortar::Element>> lineElementsS;
  std::set<std::pair<int, int>> donebefore;

  // loop over all surface elements
  for (auto ele : adjacent_elements())
  {
    auto* cele = ele.user_element();

    if (cele->shape() == Core::FE::CellType::quad4)
    {
      for (int j = 0; j < 4; ++j)
      {
        int nodeIds[2] = {0, 0};
        int nodeLIds[2] = {0, 0};

        if (j == 0)
        {
          nodeIds[0] = cele->node_ids()[0];
          nodeIds[1] = cele->node_ids()[1];

          nodeLIds[0] = 0;
          nodeLIds[1] = 1;
        }
        else if (j == 1)
        {
          nodeIds[0] = cele->node_ids()[1];
          nodeIds[1] = cele->node_ids()[2];

          nodeLIds[0] = 1;
          nodeLIds[1] = 2;
        }
        else if (j == 2)
        {
          nodeIds[0] = cele->node_ids()[2];
          nodeIds[1] = cele->node_ids()[3];

          nodeLIds[0] = 2;
          nodeLIds[1] = 3;
        }
        else if (j == 3)
        {
          nodeIds[0] = cele->node_ids()[3];
          nodeIds[1] = cele->node_ids()[0];

          nodeLIds[0] = 3;
          nodeLIds[1] = 0;
        }

        // check if both nodes on edge geometry
        bool node0Edge = dynamic_cast<Mortar::Node*>(cele->nodes()[nodeLIds[0]])->is_on_edge();
        bool node1Edge = dynamic_cast<Mortar::Node*>(cele->nodes()[nodeLIds[1]])->is_on_edge();

        if (!node0Edge or !node1Edge) continue;

        // create pair
        std::pair<int, int> actIDs = std::pair<int, int>(nodeIds[0], nodeIds[1]);
        std::pair<int, int> actIDstw = std::pair<int, int>(nodeIds[1], nodeIds[0]);

        // check if processed before
        std::set<std::pair<int, int>>::iterator iter = donebefore.find(actIDs);
        std::set<std::pair<int, int>>::iterator itertw = donebefore.find(actIDstw);

        // if not then create ele
        if (iter == donebefore.end() and itertw == donebefore.end())
        {
          // add to set of processed nodes
          donebefore.insert(actIDs);
          donebefore.insert(actIDstw);

          // create line ele:
          std::shared_ptr<Mortar::Element> lineEle = std::make_shared<Mortar::Element>(
              j, cele->owner(), Core::FE::CellType::line2, 2, nodeIds, false);

          // get nodes
          std::array<Core::Nodes::Node*, 2> nodes = {
              cele->nodes()[nodeLIds[0]], cele->nodes()[nodeLIds[1]]};
          lineEle->build_nodal_pointers(nodes.data());

          // init data container for dual shapes
          lineEle->initialize_data_container();

          // push back into vector
          lineElementsS.push_back(lineEle);
        }
      }  // end edge loop
    }
    else
      FOUR_C_THROW("only quad4!");
  }  // end surfele loop


  //**************************************************
  //      GET EDGES WHICH ARE CONNECTED TO NODE
  //**************************************************
  std::vector<int> dummy;
  for (int edges = 0; edges < (int)lineElementsS.size(); ++edges)
  {
    for (int count = 0; count < lineElementsS[edges]->num_node(); ++count)
    {
      if (lineElementsS[edges]->nodes()[count]->id() == id()) dummy.push_back(edges);
    }
  }

  if (dummy.size() > 2) std::cout << "WARNING: multiple edge definitions possible!" << std::endl;

  if (dummy.size() < 1) FOUR_C_THROW("ERROR!");

  //**************************************************
  //      CALC ADJACENT TANGENTS
  //**************************************************
  Node* n1 = nullptr;
  Node* n2 = nullptr;

  if (lineElementsS[dummy[0]]->nodes()[0]->id() != id())
    n1 = dynamic_cast<Node*>(lineElementsS[dummy[0]]->nodes()[0]);
  else if (lineElementsS[dummy[0]]->nodes()[1]->id() != id())
    n1 = dynamic_cast<Node*>(lineElementsS[dummy[0]]->nodes()[1]);
  else
    FOUR_C_THROW("ERROR");

  if (dummy.size() < 2)
  {
    // get node itself if no adjacent edge elements could be found
    n2 = this;
  }
  else
  {
    if (lineElementsS[dummy[1]]->nodes()[0]->id() != id())
      n2 = dynamic_cast<Node*>(lineElementsS[dummy[0]]->nodes()[0]);
    else if (lineElementsS[dummy[1]]->nodes()[1]->id() != id())
      n2 = dynamic_cast<Node*>(lineElementsS[dummy[0]]->nodes()[1]);
    else
      FOUR_C_THROW("ERROR");
  }


  std::array<double, 3> tmp1 = {0.0, 0.0, 0.0};
  // difference
  tmp1[0] = n1->xspatial()[0] - n2->xspatial()[0];
  tmp1[1] = n1->xspatial()[1] - n2->xspatial()[1];
  tmp1[2] = n1->xspatial()[2] - n2->xspatial()[2];
  const double length = sqrt(tmp1[0] * tmp1[0] + tmp1[1] * tmp1[1] + tmp1[2] * tmp1[2]);
  if (length < 1e-12) FOUR_C_THROW("ERROR");
  mo_data().edge_tangent()[0] = tmp1[0] / length;
  mo_data().edge_tangent()[1] = tmp1[1] / length;
  mo_data().edge_tangent()[2] = tmp1[2] / length;

  //**************************************************
  //      LINEARIZATION
  //**************************************************
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  for (int j = 0; j < (int)((data().get_deriv_tangent()).size()); ++j)
    (data().get_deriv_tangent())[j].clear();
  (data().get_deriv_tangent()).resize(0, 0);
  if ((int)data().get_deriv_tangent().size() == 0) data().get_deriv_tangent().resize(3, 2 * 100);

  std::vector<Core::Gen::Pairedvector<int, double>> lint(3, 100);  // added all sizes
  if (n1 != nullptr)
  {
    lint[0][n1->dofs()[0]] += 1;
    lint[1][n1->dofs()[1]] += 1;
    lint[2][n1->dofs()[2]] += 1;
  }
  if (n2 != nullptr)
  {
    lint[0][n2->dofs()[0]] -= 1;
    lint[1][n2->dofs()[1]] -= 1;
    lint[2][n2->dofs()[2]] -= 1;
  }
  // first part
  std::vector<Core::Gen::Pairedvector<int, double>> Lin1(3, 100);  // added all sizes
  for (CI p = lint[0].begin(); p != lint[0].end(); ++p) Lin1[0][p->first] += p->second / length;
  for (CI p = lint[1].begin(); p != lint[1].end(); ++p) Lin1[1][p->first] += p->second / length;
  for (CI p = lint[2].begin(); p != lint[2].end(); ++p) Lin1[2][p->first] += p->second / length;

  Core::Gen::Pairedvector<int, double> Lin2(100);  // added all sizes
  for (CI p = lint[0].begin(); p != lint[0].end(); ++p)
    Lin2[p->first] += p->second * mo_data().edge_tangent()[0];
  for (CI p = lint[1].begin(); p != lint[1].end(); ++p)
    Lin2[p->first] += p->second * mo_data().edge_tangent()[1];
  for (CI p = lint[2].begin(); p != lint[2].end(); ++p)
    Lin2[p->first] += p->second * mo_data().edge_tangent()[2];

  std::vector<Core::Gen::Pairedvector<int, double>> Lin3(3, 100);  // added all sizes
  for (CI p = Lin2.begin(); p != Lin2.end(); ++p)
    Lin3[0][p->first] += p->second * mo_data().edge_tangent()[0] / (length * length * length);
  for (CI p = Lin2.begin(); p != Lin2.end(); ++p)
    Lin3[1][p->first] += p->second * mo_data().edge_tangent()[1] / (length * length * length);
  for (CI p = Lin2.begin(); p != Lin2.end(); ++p)
    Lin3[2][p->first] += p->second * mo_data().edge_tangent()[2] / (length * length * length);

  for (CI p = Lin1[0].begin(); p != Lin1[0].end(); ++p)
    data().get_deriv_tangent()[0][p->first] += p->second;
  for (CI p = Lin1[1].begin(); p != Lin1[1].end(); ++p)
    data().get_deriv_tangent()[1][p->first] += p->second;
  for (CI p = Lin1[2].begin(); p != Lin1[2].end(); ++p)
    data().get_deriv_tangent()[2][p->first] += p->second;

  for (CI p = Lin3[0].begin(); p != Lin3[0].end(); ++p)
    data().get_deriv_tangent()[0][p->first] -= p->second;
  for (CI p = Lin3[1].begin(); p != Lin3[1].end(); ++p)
    data().get_deriv_tangent()[1][p->first] -= p->second;
  for (CI p = Lin3[2].begin(); p != Lin3[2].end(); ++p)
    data().get_deriv_tangent()[2][p->first] -= p->second;

  // std::cout << "tangent = " << MoData().EdgeTangent()[0] << "  " << MoData().EdgeTangent()[1] <<
  // "  " << MoData().EdgeTangent()[2] << std::endl;

  return;
}

/*----------------------------------------------------------------------*
 |  Build averaged nodal normal + tangents                    popp 12/07|
 *----------------------------------------------------------------------*/
void CONTACT::Node::build_averaged_normal()
{
  // reset normal and tangents when this method is called
  for (int j = 0; j < 3; ++j)
  {
    mo_data().n()[j] = 0.0;
    data().txi()[j] = 0.0;
    data().teta()[j] = 0.0;
  }

  int nseg = num_element();

  // temporary vector to store nodal normal
  std::array<double, 3> n_tmp = {0., 0., 0.};
  Core::LinAlg::SerialDenseMatrix elens(6, nseg);

  // we need to store some stuff here
  //**********************************************************************
  // elens(0,i): x-coord of element normal
  // elens(1,i): y-coord of element normal
  // elens(2,i): z-coord of element normal
  // elens(3,i): id of adjacent element i
  // elens(4,i): length of element normal
  // elens(5,i): length/area of element itself
  //**********************************************************************

  // loop over all adjacent elements
  int i = 0;
  for (auto ele : adjacent_elements())
  {
    auto* adjcele = dynamic_cast<Element*>(ele.user_element());

    // build element normal at current node
    // (we have to pass in the index i to be able to store the
    // normal and other information at the right place in elens)
    adjcele->build_normal_at_node(id(), i, elens);

    // add (weighted) element normal to nodal normal n
    for (int j = 0; j < 3; ++j) n_tmp[j] += elens(j, i) / elens(4, i);
    ++i;
  }

  // modify normal in case of symmetry condition
  for (int i = 0; i < 3; i++)
    if (dbc_dofs()[i]) n_tmp[i] = 0.;

  // create unit normal vector
  double length = sqrt(n_tmp[0] * n_tmp[0] + n_tmp[1] * n_tmp[1] + n_tmp[2] * n_tmp[2]);
  if (length < 1e-12)
  {
    std::cout << "normal zero: node source= " << is_source() << "  length= " << length << std::endl;
    FOUR_C_THROW("Nodal normal length 0, node ID {}", id());
  }
  else
  {
    for (int j = 0; j < 3; ++j)
    {
      mo_data().n()[j] = n_tmp[j] / length;
    }
  }

  // create unit tangent vectors
  // (note that this definition is not unique in 3D!)
  double ltxi = 1.0;

  if (num_dof() == 2)
  {
    // simple definition for txi
    data().txi()[0] = -mo_data().n()[1];
    data().txi()[1] = mo_data().n()[0];
    data().txi()[2] = 0.0;

    // teta is z-axis
    data().teta()[0] = 0.0;
    data().teta()[1] = 0.0;
    data().teta()[2] = 1.0;
  }
  else if (num_dof() == 3)
  {
    if (abs(mo_data().n()[0]) > 1.0e-4 || abs(mo_data().n()[1]) > 1.0e-4)
    {
      data().txi()[0] = -mo_data().n()[1];
      data().txi()[1] = mo_data().n()[0];
      data().txi()[2] = 0.0;
    }
    else
    {
      data().txi()[0] = 0.0;
      data().txi()[1] = -mo_data().n()[2];
      data().txi()[2] = mo_data().n()[1];
    }

    ltxi = sqrt(data().txi()[0] * data().txi()[0] + data().txi()[1] * data().txi()[1] +
                data().txi()[2] * data().txi()[2]);
    if (ltxi < 1e-12)
    {
      std::cout << "tangent 1 zero: node source= " << is_source() << "  length= " << ltxi
                << std::endl;
      FOUR_C_THROW("Nodal tangent length 0, node ID {}", id());
    }
    else
    {
      for (int j = 0; j < 3; ++j) data().txi()[j] /= ltxi;
    }



    // teta follows from corkscrew rule (teta = n x txi)
    data().teta()[0] = mo_data().n()[1] * data().txi()[2] - mo_data().n()[2] * data().txi()[1];
    data().teta()[1] = mo_data().n()[2] * data().txi()[0] - mo_data().n()[0] * data().txi()[2];
    data().teta()[2] = mo_data().n()[0] * data().txi()[1] - mo_data().n()[1] * data().txi()[0];
  }
  else
    FOUR_C_THROW("Contact problems must be either 2D or 3D");

  // build linearization of averaged nodal normal and tangents
  deriv_averaged_normal(elens, length, ltxi);

  return;
}

/*----------------------------------------------------------------------*
 |  Build directional deriv. of nodal normal + tangents       popp 09/08|
 *----------------------------------------------------------------------*/
void CONTACT::Node::deriv_averaged_normal(
    Core::LinAlg::SerialDenseMatrix& elens, double length, double ltxi)
{
  // prepare nodal storage maps for derivative
  if ((int)data().get_deriv_n().size() == 0) data().get_deriv_n().resize(3, linsize_);
  if ((int)data().get_deriv_txi().size() == 0) data().get_deriv_txi().resize(3, linsize_);
  if ((int)data().get_deriv_teta().size() == 0) data().get_deriv_teta().resize(3, linsize_);

  // loop over all adjacent elements
  int i = 0;
  for (auto ele : adjacent_elements())
  {
    Element* adjcele = dynamic_cast<Element*>(ele.user_element());

    // build element normal derivative at current node
    adjcele->deriv_normal_at_node(id(), i, elens, data().get_deriv_n());
    ++i;
  }

  // modify normal in case of symmetry condition
  for (int i = 0; i < 3; i++)
    if (dbc_dofs()[i]) data().get_deriv_n()[i].clear();

  // normalize directional derivative
  // (length differs for weighted/unweighted case but not the procedure!)
  // (be careful with reference / copy of derivative maps!)
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;
  Core::Gen::Pairedvector<int, double>& derivnx = data().get_deriv_n()[0];
  Core::Gen::Pairedvector<int, double>& derivny = data().get_deriv_n()[1];
  Core::Gen::Pairedvector<int, double>& derivnz = data().get_deriv_n()[2];
  Core::Gen::Pairedvector<int, double> cderivnx = data().get_deriv_n()[0];
  Core::Gen::Pairedvector<int, double> cderivny = data().get_deriv_n()[1];
  Core::Gen::Pairedvector<int, double> cderivnz = data().get_deriv_n()[2];
  const double nxnx = mo_data().n()[0] * mo_data().n()[0];
  const double nxny = mo_data().n()[0] * mo_data().n()[1];
  const double nxnz = mo_data().n()[0] * mo_data().n()[2];
  const double nyny = mo_data().n()[1] * mo_data().n()[1];
  const double nynz = mo_data().n()[1] * mo_data().n()[2];
  const double nznz = mo_data().n()[2] * mo_data().n()[2];

  // build a vector with all keys from x,y,z maps
  // (we need this in order not to miss any entry!)
  std::vector<int> allkeysn;
  for (CI p = derivnx.begin(); p != derivnx.end(); ++p)
  {
    bool found = false;
    for (int j = 0; j < (int)allkeysn.size(); ++j)
      if ((p->first) == allkeysn[j]) found = true;
    if (!found) allkeysn.push_back(p->first);
  }
  for (CI p = derivny.begin(); p != derivny.end(); ++p)
  {
    bool found = false;
    for (int j = 0; j < (int)allkeysn.size(); ++j)
      if ((p->first) == allkeysn[j]) found = true;
    if (!found) allkeysn.push_back(p->first);
  }
  for (CI p = derivnz.begin(); p != derivnz.end(); ++p)
  {
    bool found = false;
    for (int j = 0; j < (int)allkeysn.size(); ++j)
      if ((p->first) == allkeysn[j]) found = true;
    if (!found) allkeysn.push_back(p->first);
  }

  // normalize x-components
  for (int j = 0; j < (int)allkeysn.size(); ++j)
  {
    double val = cderivnx[allkeysn[j]];
    derivnx[allkeysn[j]] =
        (val - nxnx * val - nxny * cderivny[allkeysn[j]] - nxnz * cderivnz[allkeysn[j]]) / length;
  }

  // normalize y-components
  for (int j = 0; j < (int)allkeysn.size(); ++j)
  {
    double val = cderivny[allkeysn[j]];
    derivny[allkeysn[j]] =
        (val - nxny * cderivnx[allkeysn[j]] - nyny * val - nynz * cderivnz[allkeysn[j]]) / length;
  }

  // normalize z-components
  for (int j = 0; j < (int)allkeysn.size(); ++j)
  {
    double val = cderivnz[allkeysn[j]];
    derivnz[allkeysn[j]] =
        (val - nxnz * cderivnx[allkeysn[j]] - nynz * cderivny[allkeysn[j]] - nznz * val) / length;
  }

  //**********************************************************************
  // tangent derivatives 2D
  //**********************************************************************
  if (num_dof() == 2)
  {
    // get directional derivative of nodal tangent txi "for free"
    // (we just have to use the orthogonality of n and t)
    // the directional derivative of nodal tangent teta is 0
    Core::Gen::Pairedvector<int, double>& derivtxix = data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double>& derivtxiy = data().get_deriv_txi()[1];

    for (CI p = derivny.begin(); p != derivny.end(); ++p) derivtxix[p->first] = -(p->second);
    for (CI p = derivnx.begin(); p != derivnx.end(); ++p) derivtxiy[p->first] = (p->second);
  }

  //**********************************************************************
  // tangent derivatives 3D
  //**********************************************************************
  else
  {
    // unnormalized tangent derivative txi
    // use definitions for txi from BuildAveragedNormal()
    if (abs(mo_data().n()[0]) > 1.0e-4 || abs(mo_data().n()[1]) > 1.0e-4)
    {
      Core::Gen::Pairedvector<int, double>& derivtxix = data().get_deriv_txi()[0];
      Core::Gen::Pairedvector<int, double>& derivtxiy = data().get_deriv_txi()[1];

      for (CI p = derivny.begin(); p != derivny.end(); ++p) derivtxix[p->first] -= (p->second);

      for (CI p = derivnx.begin(); p != derivnx.end(); ++p) derivtxiy[p->first] += (p->second);
    }
    else
    {
      Core::Gen::Pairedvector<int, double>& derivtxiy = data().get_deriv_txi()[1];
      Core::Gen::Pairedvector<int, double>& derivtxiz = data().get_deriv_txi()[2];

      for (CI p = derivnz.begin(); p != derivnz.end(); ++p) derivtxiy[p->first] -= (p->second);

      for (CI p = derivny.begin(); p != derivny.end(); ++p) derivtxiz[p->first] += (p->second);
    }

    // normalize txi directional derivative
    // (identical to normalization of normal derivative)
    using CI = Core::Gen::Pairedvector<int, double>::const_iterator;
    Core::Gen::Pairedvector<int, double>& derivtxix = data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double>& derivtxiy = data().get_deriv_txi()[1];
    Core::Gen::Pairedvector<int, double>& derivtxiz = data().get_deriv_txi()[2];
    Core::Gen::Pairedvector<int, double> cderivtxix = data().get_deriv_txi()[0];
    Core::Gen::Pairedvector<int, double> cderivtxiy = data().get_deriv_txi()[1];
    Core::Gen::Pairedvector<int, double> cderivtxiz = data().get_deriv_txi()[2];
    const double txtx = data().txi()[0] * data().txi()[0];
    const double txty = data().txi()[0] * data().txi()[1];
    const double txtz = data().txi()[0] * data().txi()[2];
    const double tyty = data().txi()[1] * data().txi()[1];
    const double tytz = data().txi()[1] * data().txi()[2];
    const double tztz = data().txi()[2] * data().txi()[2];

    // build a vector with all keys from x,y,z maps
    // (we need this in order not to miss any entry!)
    std::vector<int> allkeyst;
    for (CI p = derivtxix.begin(); p != derivtxix.end(); ++p)
    {
      bool found = false;
      for (int j = 0; j < (int)allkeyst.size(); ++j)
        if ((p->first) == allkeyst[j]) found = true;
      if (!found) allkeyst.push_back(p->first);
    }
    for (CI p = derivtxiy.begin(); p != derivtxiy.end(); ++p)
    {
      bool found = false;
      for (int j = 0; j < (int)allkeyst.size(); ++j)
        if ((p->first) == allkeyst[j]) found = true;
      if (!found) allkeyst.push_back(p->first);
    }
    for (CI p = derivtxiz.begin(); p != derivtxiz.end(); ++p)
    {
      bool found = false;
      for (int j = 0; j < (int)allkeyst.size(); ++j)
        if ((p->first) == allkeyst[j]) found = true;
      if (!found) allkeyst.push_back(p->first);
    }

    // normalize x-components
    for (int j = 0; j < (int)allkeyst.size(); ++j)
    {
      double val = cderivtxix[allkeyst[j]];
      derivtxix[allkeyst[j]] =
          (val - txtx * val - txty * cderivtxiy[allkeyst[j]] - txtz * cderivtxiz[allkeyst[j]]) /
          ltxi;
    }

    // normalize y-components
    for (int j = 0; j < (int)allkeyst.size(); ++j)
    {
      double val = cderivtxiy[allkeyst[j]];
      derivtxiy[allkeyst[j]] =
          (val - txty * cderivtxix[allkeyst[j]] - tyty * val - tytz * cderivtxiz[allkeyst[j]]) /
          ltxi;
    }

    // normalize z-components
    for (int j = 0; j < (int)allkeyst.size(); ++j)
    {
      double val = cderivtxiz[allkeyst[j]];
      derivtxiz[allkeyst[j]] =
          (val - txtz * cderivtxix[allkeyst[j]] - tytz * cderivtxiy[allkeyst[j]] - tztz * val) /
          ltxi;
    }

    // get normalized tangent derivative teta
    // use corkscrew rule from BuildAveragedNormal()
    Core::Gen::Pairedvector<int, double>& derivtetax = data().get_deriv_teta()[0];
    Core::Gen::Pairedvector<int, double>& derivtetay = data().get_deriv_teta()[1];
    Core::Gen::Pairedvector<int, double>& derivtetaz = data().get_deriv_teta()[2];

    for (CI p = derivnx.begin(); p != derivnx.end(); ++p)
    {
      derivtetay[p->first] -= data().txi()[2] * (p->second);
      derivtetaz[p->first] += data().txi()[1] * (p->second);
    }
    for (CI p = derivny.begin(); p != derivny.end(); ++p)
    {
      derivtetax[p->first] += data().txi()[2] * (p->second);
      derivtetaz[p->first] -= data().txi()[0] * (p->second);
    }
    for (CI p = derivnz.begin(); p != derivnz.end(); ++p)
    {
      derivtetax[p->first] -= data().txi()[1] * (p->second);
      derivtetay[p->first] += data().txi()[0] * (p->second);
    }
    for (CI p = derivtxix.begin(); p != derivtxix.end(); ++p)
    {
      derivtetay[p->first] += mo_data().n()[2] * (p->second);
      derivtetaz[p->first] -= mo_data().n()[1] * (p->second);
    }
    for (CI p = derivtxiy.begin(); p != derivtxiy.end(); ++p)
    {
      derivtetax[p->first] -= mo_data().n()[2] * (p->second);
      derivtetaz[p->first] += mo_data().n()[0] * (p->second);
    }
    for (CI p = derivtxiz.begin(); p != derivtxiz.end(); ++p)
    {
      derivtetax[p->first] += mo_data().n()[1] * (p->second);
      derivtetay[p->first] -= mo_data().n()[0] * (p->second);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Add a value to the NCoup of this node                      ager 06/14|
 *----------------------------------------------------------------------*/
void CONTACT::Node::add_ncoup_value(const double val)
{
  // check if this is a target node or source boundary node
  if (is_source() == false) FOUR_C_THROW("AddNcoupValue: function called for target node {}", id());
  if (is_on_bound() == true)
    FOUR_C_THROW("AddNcoupValue: function called for boundary node {}", id());

  // add given value to ncoup
  poro_data().getn_coup() += val;
}

/*----------------------------------------------------------------------*
 |  Store nodal normals to old ones                         seitz 05/17 |
 *----------------------------------------------------------------------*/
void CONTACT::Node::store_old_normal()
{
  // write entries to old ones
  for (int j = 0; j < 3; ++j) data().normal_old()[j] = mo_data().n()[j];

  return;
}

FOUR_C_NAMESPACE_CLOSE
