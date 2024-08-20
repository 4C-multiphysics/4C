/*---------------------------------------------------------------------*/
/*! \file

\brief Sidehandle represents a side original loaded into the cut, internal it can be split into
subsides

\level 3


*----------------------------------------------------------------------*/

#include "4C_cut_sidehandle.hpp"

#include "4C_cut_mesh.hpp"
#include "4C_cut_position.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Tri6SideHandle::Tri6SideHandle(Mesh& mesh, int sid, const std::vector<int>& node_ids)
{
  subsides_.reserve(4);

  nodes_.reserve(4);
  for (int i = 0; i < 4; ++i)
  {
    Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
    nodes_.push_back(n);
  }

  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Triangle<3>>();

  std::vector<int> nids(3);

  nids[0] = node_ids[0];
  nids[1] = node_ids[3];
  nids[2] = node_ids[5];
  subsides_.push_back(mesh.get_side(sid, nids, top_data));

  nids[0] = node_ids[3];
  nids[1] = node_ids[1];
  nids[2] = node_ids[4];
  subsides_.push_back(mesh.get_side(sid, nids, top_data));

  nids[0] = node_ids[3];
  nids[1] = node_ids[4];
  nids[2] = node_ids[5];
  subsides_.push_back(mesh.get_side(sid, nids, top_data));

  nids[0] = node_ids[5];
  nids[1] = node_ids[4];
  nids[2] = node_ids[2];
  subsides_.push_back(mesh.get_side(sid, nids, top_data));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Quad4SideHandle::Quad4SideHandle(Mesh& mesh, int sid, const std::vector<int>& node_ids)
{
  subsides_.reserve(4);

  Core::LinAlg::Matrix<3, 4> xyze;
  nodes_.reserve(4);
  for (int i = 0; i < 4; ++i)
  {
    Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
    nodes_.push_back(n);
    n->coordinates(&xyze(0, i));
  }

  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Triangle<3>>();

  // create middle node

  Core::LinAlg::Matrix<4, 1> funct;
  Core::FE::shape_function_2d(funct, 0.0, 0.0, Core::FE::CellType::quad4);

  Core::LinAlg::Matrix<3, 1> xyz;
  xyz.multiply(xyze, funct);

  plain_int_set node_nids;
  node_nids.insert(node_ids.begin(), node_ids.end());
  Node* middle = mesh.get_node(node_nids, xyz.data());
  int middle_id = middle->id();

  std::vector<int> nids(3);

  nids[0] = node_ids[0];
  nids[1] = node_ids[1];
  nids[2] = middle_id;
  subsides_.push_back(mesh.get_side(sid, nids, top_data));

  nids[0] = node_ids[1];
  nids[1] = node_ids[2];
  nids[2] = middle_id;
  subsides_.push_back(mesh.get_side(sid, nids, top_data));

  nids[0] = node_ids[2];
  nids[1] = node_ids[3];
  nids[2] = middle_id;
  subsides_.push_back(mesh.get_side(sid, nids, top_data));

  nids[0] = node_ids[3];
  nids[1] = node_ids[0];
  nids[2] = middle_id;
  subsides_.push_back(mesh.get_side(sid, nids, top_data));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Quad8SideHandle::Quad8SideHandle(
    Mesh& mesh, int sid, const std::vector<int>& node_ids, bool iscutside)
{
  if (iscutside)
  {
    subsides_.reserve(6);
    nodes_.reserve(8);
    for (int i = 0; i < 8; ++i)
    {
      Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
      nodes_.push_back(n);
    }
    const CellTopologyData* top_data = shards::getCellTopologyData<shards::Triangle<3>>();
    std::vector<int> nids(3);
    nids[0] = node_ids[7];
    nids[1] = node_ids[0];
    nids[2] = node_ids[4];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[4];
    nids[1] = node_ids[1];
    nids[2] = node_ids[5];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[5];
    nids[1] = node_ids[2];
    nids[2] = node_ids[6];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[6];
    nids[1] = node_ids[3];
    nids[2] = node_ids[7];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[4];
    nids[1] = node_ids[5];
    nids[2] = node_ids[6];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[6];
    nids[1] = node_ids[7];
    nids[2] = node_ids[4];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
  }
  else
  {
    subsides_.reserve(4);

    Core::LinAlg::Matrix<3, 8> xyze;
    nodes_.reserve(8);
    for (int i = 0; i < 8; ++i)
    {
      Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
      nodes_.push_back(n);
      n->coordinates(&xyze(0, i));
    }

    const CellTopologyData* top_data = shards::getCellTopologyData<shards::Quadrilateral<4>>();

    // create middle node

    Core::LinAlg::Matrix<8, 1> funct;
    Core::FE::shape_function_2d(funct, 0.0, 0.0, Core::FE::CellType::quad8);

    Core::LinAlg::Matrix<3, 1> xyz;
    xyz.multiply(xyze, funct);

    plain_int_set node_nids;
    std::copy(node_ids.begin(), node_ids.end(), std::inserter(node_nids, node_nids.begin()));
    Node* middle = mesh.get_node(node_nids, xyz.data());
    int middle_id = middle->id();

    std::vector<int> nids(4);

    nids[0] = node_ids[0];
    nids[1] = node_ids[4];
    nids[2] = middle_id;
    nids[3] = node_ids[7];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[4];
    nids[1] = node_ids[1];
    nids[2] = node_ids[5];
    nids[3] = middle_id;
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = middle_id;
    nids[1] = node_ids[5];
    nids[2] = node_ids[2];
    nids[3] = node_ids[6];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[7];
    nids[1] = middle_id;
    nids[2] = node_ids[6];
    nids[3] = node_ids[3];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Quad9SideHandle::Quad9SideHandle(
    Mesh& mesh, int sid, const std::vector<int>& node_ids, bool iscutside)
{
  if (iscutside)
  {
    subsides_.reserve(8);
    nodes_.reserve(9);
    for (int i = 0; i < 9; ++i)
    {
      Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
      nodes_.push_back(n);
    }
    const CellTopologyData* top_data = shards::getCellTopologyData<shards::Triangle<3>>();
    std::vector<int> nids(3);
    nids[0] = node_ids[7];
    nids[1] = node_ids[0];
    nids[2] = node_ids[4];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
    nids[0] = node_ids[4];
    nids[1] = node_ids[8];
    nids[2] = node_ids[7];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
    nids[0] = node_ids[4];
    nids[1] = node_ids[1];
    nids[2] = node_ids[5];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
    nids[0] = node_ids[5];
    nids[1] = node_ids[8];
    nids[2] = node_ids[4];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
    nids[0] = node_ids[5];
    nids[1] = node_ids[2];
    nids[2] = node_ids[6];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
    nids[0] = node_ids[6];
    nids[1] = node_ids[8];
    nids[2] = node_ids[5];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
    nids[0] = node_ids[6];
    nids[1] = node_ids[3];
    nids[2] = node_ids[7];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
    nids[0] = node_ids[7];
    nids[1] = node_ids[8];
    nids[2] = node_ids[6];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
  }
  else
  {
    subsides_.reserve(4);

    nodes_.reserve(9);
    for (int i = 0; i < 9; ++i)
    {
      Node* n = mesh.get_node(node_ids[i], static_cast<double*>(nullptr));
      nodes_.push_back(n);
    }

    const CellTopologyData* top_data = shards::getCellTopologyData<shards::Quadrilateral<4>>();

    std::vector<int> nids(4);

    nids[0] = node_ids[0];
    nids[1] = node_ids[4];
    nids[2] = node_ids[8];
    nids[3] = node_ids[7];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[4];
    nids[1] = node_ids[1];
    nids[2] = node_ids[5];
    nids[3] = node_ids[8];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[8];
    nids[1] = node_ids[5];
    nids[2] = node_ids[2];
    nids[3] = node_ids[6];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));

    nids[0] = node_ids[7];
    nids[1] = node_ids[8];
    nids[2] = node_ids[6];
    nids[3] = node_ids[3];
    subsides_.push_back(mesh.get_side(sid, nids, top_data));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Tri6SideHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<2, 1>& rst)
{
  Core::LinAlg::Matrix<3, 6> xyze;

  for (int i = 0; i < 6; ++i)
  {
    Node* n = nodes_[i];
    n->coordinates(&xyze(0, i));
  }

  Teuchos::RCP<Position> pos =
      PositionFactory::build_position<3, Core::FE::CellType::tri6>(xyze, xyz);
  bool success = pos->compute();
  if (not success)
  {
  }
  pos->local_coordinates(rst);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Quad4SideHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<2, 1>& rst)
{
  Core::LinAlg::Matrix<3, 4> xyze;

  for (int i = 0; i < 4; ++i)
  {
    Node* n = nodes_[i];
    n->coordinates(&xyze(0, i));
  }

  Teuchos::RCP<Position> pos =
      PositionFactory::build_position<3, Core::FE::CellType::quad4>(xyze, xyz);
  bool success = pos->compute();
  if (not success)
  {
  }
  pos->local_coordinates(rst);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Quad8SideHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<2, 1>& rst)
{
  Core::LinAlg::Matrix<3, 8> xyze;

  for (int i = 0; i < 8; ++i)
  {
    Node* n = nodes_[i];
    n->coordinates(&xyze(0, i));
  }

  Teuchos::RCP<Position> pos =
      PositionFactory::build_position<3, Core::FE::CellType::quad8>(xyze, xyz);
  bool success = pos->compute();
  if (not success)
  {
  }
  pos->local_coordinates(rst);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Quad9SideHandle::local_coordinates(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<2, 1>& rst)
{
  Core::LinAlg::Matrix<3, 9> xyze;

  for (int i = 0; i < 9; ++i)
  {
    Node* n = nodes_[i];
    n->coordinates(&xyze(0, i));
  }

  Teuchos::RCP<Position> pos =
      PositionFactory::build_position<3, Core::FE::CellType::quad9>(xyze, xyz);
  bool success = pos->compute();
  if (not success)
  {
  }
  pos->local_coordinates(rst);
}

FOUR_C_NAMESPACE_CLOSE
