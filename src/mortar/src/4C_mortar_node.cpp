// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_node.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_mortar_element.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


Mortar::NodeType Mortar::NodeType::instance_;


Core::Communication::ParObject* Mortar::NodeType::create(Core::Communication::UnpackBuffer& buffer)
{
  std::vector<double> x(3, 0.0);
  std::vector<int> dofs;
  auto* node = new Mortar::Node(0, x, 0, dofs, false);
  node->unpack(buffer);
  return node;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mortar::NodeDataContainer::NodeDataContainer()
    : drows_(0), drows_nts_(0), drows_lts_(0), drows_ltl_(0)
{
  for (int i = 0; i < 3; ++i)
  {
    n()[i] = 0.0;
    edge_tangent()[i] = 0.0;
    lm()[i] = 0.0;
    lmold()[i] = 0.0;
    lmuzawa()[i] = 0.0;
    get_dscale() = 0.0;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::NodeDataContainer::pack(Core::Communication::PackBuffer& data) const
{
  // add n_
  add_to_pack(data, n_, 3 * sizeof(double));
  // add edgetangent_
  add_to_pack(data, edgeTangent_, 3 * sizeof(double));
  // add lm_
  add_to_pack(data, lm_, 3 * sizeof(double));
  // add lmold_
  add_to_pack(data, lmold_, 3 * sizeof(double));
  // add lmuzawa_
  add_to_pack(data, lmuzawa_, 3 * sizeof(double));

  // no need to pack drows_, mrows_ and mmodrows_
  // (these will evaluated anew anyway)
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::NodeDataContainer::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // n_
  extract_from_pack(buffer, n_, 3 * sizeof(double));
  // edgetangent_
  extract_from_pack(buffer, edgeTangent_, 3 * sizeof(double));
  // lm_
  extract_from_pack(buffer, lm_, 3 * sizeof(double));
  // lmold_
  extract_from_pack(buffer, lmold_, 3 * sizeof(double));
  // lmuzawa_
  extract_from_pack(buffer, lmuzawa_, 3 * sizeof(double));
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mortar::Node::Node(int id, std::span<const double> coords, const int owner,
    const std::vector<int>& dofs, const bool isslave)
    : Core::Nodes::Node(id, coords, owner),
      isslave_(isslave),
      istiedslave_(isslave),
      isonbound_(false),
      isonedge_(false),
      isoncorner_(false),
      isdbc_(false),
      dofs_(dofs),
      hasproj_(false),
      hassegment_(false),
      detected_(false),
      dentries_(0),
      modata_(nullptr),
      nurbsw_(-1.0)
{
  for (std::size_t i = 0; i < coords.size(); ++i)
  {
    uold()[i] = 0.0;
    xspatial()[i] = x()[i];
    dbcdofs_[i] = false;
  }
  // For 2D problems, initialize the unused third coordinate to zero
  for (std::size_t i = coords.size(); i < 3; ++i)
  {
    uold()[i] = 0.0;
    xspatial()[i] = 0.0;
    dbcdofs_[i] = false;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mortar::Node::Node(const Mortar::Node& old)
    : Core::Nodes::Node(old),
      isslave_(old.isslave_),
      istiedslave_(old.istiedslave_),
      isonbound_(old.isonbound_),
      isdbc_(old.isdbc_),
      dofs_(old.dofs_),
      hasproj_(old.hasproj_),
      hassegment_(old.hassegment_),
      detected_(false)
{
  for (int i = 0; i < 3; ++i)
  {
    uold()[i] = old.uold_[i];
    xspatial()[i] = old.xspatial_[i];
    dbcdofs_[i] = false;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mortar::Node* Mortar::Node::clone() const
{
  auto* newnode = new Mortar::Node(*this);
  return newnode;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const Mortar::Node& mrtrnode)
{
  mrtrnode.print(os);
  return os;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::print(std::ostream& os) const
{
  // Print id and coordinates
  os << "Mortar ";
  Core::Nodes::Node::print(os);

  if (is_slave())
    os << " Slave  ";
  else
    os << " Master ";

  if (is_on_bound())
    os << " Boundary ";
  else
    os << " Interior ";
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Core::Nodes::Node
  Core::Nodes::Node::pack(data);
  // add isslave_
  add_to_pack(data, isslave_);
  // add istiedslave_
  add_to_pack(data, istiedslave_);
  // add isonbound_
  add_to_pack(data, isonbound_);
  // add isonbound_
  add_to_pack(data, isonedge_);
  // add isonbound_
  add_to_pack(data, isoncorner_);
  // add isdbc_
  add_to_pack(data, isdbc_);
  // add dbcdofs_
  add_to_pack(data, dbcdofs_[0]);
  add_to_pack(data, dbcdofs_[1]);
  add_to_pack(data, dbcdofs_[2]);
  // dentries_
  add_to_pack(data, dentries_);
  // add dofs_
  add_to_pack(data, dofs_);
  // add xspatial_
  add_to_pack(data, xspatial_, 3 * sizeof(double));
  // add uold_
  add_to_pack(data, uold_, 3 * sizeof(double));
  // add hasproj_
  add_to_pack(data, hasproj_);
  // add hassegment_
  add_to_pack(data, hassegment_);
  // add nurbsw_
  add_to_pack(data, nurbsw_);

  // add data_
  bool hasdata = (modata_ != nullptr);
  add_to_pack(data, hasdata);
  if (hasdata) modata_->pack(data);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Core::Nodes::Node
  Core::Nodes::Node::unpack(buffer);
  // isslave_
  extract_from_pack(buffer, isslave_);
  // istiedslave_
  extract_from_pack(buffer, istiedslave_);
  // isonbound_
  extract_from_pack(buffer, isonbound_);
  // isonedge_
  extract_from_pack(buffer, isonedge_);
  // isoncorner_
  extract_from_pack(buffer, isoncorner_);
  // isdbc_
  extract_from_pack(buffer, isdbc_);
  // dbcdofs_
  extract_from_pack(buffer, dbcdofs_[0]);
  extract_from_pack(buffer, dbcdofs_[1]);
  extract_from_pack(buffer, dbcdofs_[2]);
  // dentries_
  extract_from_pack(buffer, dentries_);
  // dofs_
  extract_from_pack(buffer, dofs_);
  // xspatial_
  extract_from_pack(buffer, xspatial_, 3 * sizeof(double));
  // uold_
  extract_from_pack(buffer, uold_, 3 * sizeof(double));
  // hasproj_
  extract_from_pack(buffer, hasproj_);
  // hassegment_
  extract_from_pack(buffer, hassegment_);
  // nurbsw_
  extract_from_pack(buffer, nurbsw_);

  // data_
  bool hasdata;
  extract_from_pack(buffer, hasdata);
  if (hasdata)
  {
    modata_ = std::make_shared<Mortar::NodeDataContainer>();
    modata_->unpack(buffer);
  }
  else
  {
    modata_ = nullptr;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_d_value(const int& colnode, const double& val)
{
  // check if this is a master node or slave boundary node
  if (not is_slave()) FOUR_C_THROW("AddDValue: function called for master node {}", id());
  if (is_on_bound()) FOUR_C_THROW("AddDValue: function called for boundary node {}", id());

  // check if this has been called before
  if ((int)mo_data().get_d().size() == 0) mo_data().get_d().resize(dentries_);

  // add the pair (col,val) to the given row
  mo_data().get_d()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_dnts_value(const int& colnode, const double& val)
{
  // check if this has been called before
  if ((int)mo_data().get_dnts().size() == 0) mo_data().get_dnts().resize(dentries_);

  // add the pair (col,val) to the given row
  mo_data().get_dnts()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_dlts_value(const int& colnode, const double& val)
{
  // check if this has been called before
  if ((int)mo_data().get_dlts().size() == 0) mo_data().get_dlts().resize(dentries_);

  // add the pair (col,val) to the given row
  mo_data().get_dlts()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_dltl_value(const int& colnode, const double& val)
{
  // check if this is a master node or slave boundary node
  if (not is_slave()) FOUR_C_THROW("AddDValue: function called for master node {}", id());
  if (not is_on_edge()) FOUR_C_THROW("function called for non-edge node {}", id());

  // check if this has been called before
  if (static_cast<int>(mo_data().get_dltl().size()) == 0) mo_data().get_dltl().resize(dentries_);

  // add the pair (col,val) to the given row
  mo_data().get_dltl()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_m_value(const int& colnode, const double& val)
{
  // check if this is a master node or slave boundary node
  if (not is_slave()) FOUR_C_THROW("AddMValue: function called for master node {}", id());
  if (is_on_boundor_ce()) FOUR_C_THROW("AddMValue: function called for boundary node {}", id());

  // add the pair (col,val) to the given row
  mo_data().get_m()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_mnts_value(const int& colnode, const double& val)
{
  // add the pair (col,val) to the given row
  mo_data().get_mnts()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_mlts_value(const int& colnode, const double& val)
{
  // add the pair (col,val) to the given row
  mo_data().get_mlts()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_mltl_value(const int& colnode, const double& val)
{
  // check if this is a master node or slave boundary node
  if (not is_slave()) FOUR_C_THROW("AddMValue: function called for master node {}", id());
  if (not is_on_edge()) FOUR_C_THROW("function called for non-edge node {}", id());

  // add the pair (col,val) to the given row
  mo_data().get_mltl()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::add_mmod_value(const int& colnode, const double& val)
{
  // check if this is a master node or slave boundary node
  if (not is_slave()) FOUR_C_THROW("AddMmodValue: function called for master node {}", id());
  if (is_on_bound()) FOUR_C_THROW("AddMmodValue: function called for boundary node {}", id());

  // add the pair (col,val) to the given row
  mo_data().get_mmod()[colnode] += val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::initialize_data_container()
{
  // get maximum size of nodal D-entries
  dentries_ = 0;
  std::set<int> sIdCheck;
  for (auto ele : adjacent_elements())
  {
    for (auto node : ele.nodes())
    {
      auto check = sIdCheck.insert(node.global_id());
      if (check.second) dentries_ += ele.user_element()->num_dof_per_node(*(node.user_node()));
    }
  }

  // only initialize if not yet done
  if (modata_ == nullptr) modata_ = std::make_shared<Mortar::NodeDataContainer>();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::reset_data_container()
{
  // reset to nullptr
  modata_ = nullptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Node::build_averaged_normal()
{
  // reset normal and tangents when this method is called
  for (int j = 0; j < 3; ++j) mo_data().n()[j] = 0.0;

  const int nseg = num_element();

  // we need to store some stuff here
  //**********************************************************************
  // elens(0,i): x-coord of element normal
  // elens(1,i): y-coord of element normal
  // elens(2,i): z-coord of element normal
  // elens(3,i): id of adjacent element i
  // elens(4,i): length of element normal
  // elens(5,i): length/area of element itself
  //**********************************************************************
  Core::LinAlg::SerialDenseMatrix elens(6, nseg);

  // loop over all adjacent elements
  int i = 0;
  for (auto ele : adjacent_elements())
  {
    auto* adjmrtrele = dynamic_cast<Mortar::Element*>(ele.user_element());

    // build element normal at current node
    // (we have to pass in the index i to be able to store the
    // normal and other information at the right place in elens)
    adjmrtrele->build_normal_at_node(id(), i, elens);

    // add (weighted) element normal to nodal normal n
    for (int j = 0; j < 3; ++j) mo_data().n()[j] += elens(j, i) / elens(4, i);
    ++i;
  }

  // create unit normal vector
  double length = sqrt(mo_data().n()[0] * mo_data().n()[0] + mo_data().n()[1] * mo_data().n()[1] +
                       mo_data().n()[2] * mo_data().n()[2]);
  if (length == 0.0)
    FOUR_C_THROW("Nodal normal length 0, node ID {}", id());
  else
    for (int j = 0; j < 3; ++j) mo_data().n()[j] /= length;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mortar::Node* Mortar::Node::find_closest_node(
    const std::shared_ptr<Core::FE::Discretization> intdis,
    const std::shared_ptr<Core::LinAlg::Map> nodesearchmap, double& mindist)
{
  Node* closestnode = nullptr;

  // loop over all nodes of the Core::FE::Discretization that are
  // included in the given Core::LinAlg::Map ("brute force" search)
  for (int i = 0; i < nodesearchmap->num_my_elements(); ++i)
  {
    int gid = nodesearchmap->gid(i);
    Core::Nodes::Node* node = intdis->g_node(gid);
    if (!node) FOUR_C_THROW("FindClosestNode: Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    // build distance between the two nodes
    double dist = 0.0;
    const double* p1 = xspatial();
    const double* p2 = mrtrnode->xspatial();

    for (int j = 0; j < 3; ++j) dist += (p1[j] - p2[j]) * (p1[j] - p2[j]);
    dist = sqrt(dist);

    // new closest node found, update
    if (dist <= mindist)
    {
      mindist = dist;
      closestnode = mrtrnode;
    }
  }

  if (!closestnode) FOUR_C_THROW("FindClosestNode: No closest node found at all!");

  return closestnode;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Mortar::Node::check_mesh_distortion(double& relocation, const double& limit)
{
  // initialize return parameter
  bool ok = true;

  // loop over all adjacent elements of this node
  for (auto ele : adjacent_elements())
  {
    // get the current element
    Core::Elements::Element* eleptr = ele.user_element();
    if (!eleptr) FOUR_C_THROW("Cannot find element with lid {}", ele.local_id());
    auto* mrtrele = dynamic_cast<Mortar::Element*>(eleptr);

    // minimal edge size of the current element
    const double minedgesize = mrtrele->min_edge_size();

    // check whether relocation is not too large
    if (relocation > limit * minedgesize)
    {
      // print information to screen
      std::cout << "\n*****************WARNING***********************" << '\n';
      std::cout << "Checking distortion for CNode:     " << id() << '\n';
      std::cout << "Relocation distance:               " << relocation << '\n';
      std::cout << "AdjEle: " << mrtrele->id() << "\tLimit*MinEdgeSize: " << limit * minedgesize
                << '\n';
      std::cout << "*****************WARNING***********************" << '\n';

      // set return parameter and stop
      ok = false;
      break;
    }
  }

  return ok;
}

FOUR_C_NAMESPACE_CLOSE
