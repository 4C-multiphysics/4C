/*----------------------------------------------------------------------*/
/*! \file

 \brief provides geometry methods for a search tree

 \level 1

 */

#include "4C_fem_geometry_searchtree_service.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_element_coordtrafo.hpp"
#include "4C_fem_geometry_intersection_service.hpp"
#include "4C_fem_geometry_intersection_service_templates.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_geometry_searchtree_nearestobject.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | delivers a axis-aligned bounding box for a given          peder 07/08|
 | discretization                                                       |
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 2> Core::Geo::getXAABBofDis(const Core::FE::Discretization& dis,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions)
{
  Core::LinAlg::Matrix<3, 2> XAABB(true);
  if (dis.num_global_elements() == 0) return XAABB;

  if (dis.num_my_col_elements() == 0) return XAABB;

  // initialize XAABB as rectangle around the first point of dis
  const int nodeid = dis.l_col_element(0)->nodes()[0]->id();
  const Core::LinAlg::Matrix<3, 1> pos = currentpositions.find(nodeid)->second;
  for (int dim = 0; dim < 3; ++dim)
  {
    XAABB(dim, 0) = pos(dim) - Core::Geo::TOL7;
    XAABB(dim, 1) = pos(dim) + Core::Geo::TOL7;
  }

  // TODO make more efficient by cahcking nodes
  // loop over elements and merge XAABB with their eXtendedAxisAlignedBoundingBox
  for (int j = 0; j < dis.num_my_col_elements(); ++j)
  {
    const Core::Elements::Element* element = dis.l_col_element(j);
    const Core::LinAlg::SerialDenseMatrix xyze_element(
        Core::Geo::getCurrentNodalPositions(element, currentpositions));
    Core::Geo::EleGeoType eleGeoType(Core::Geo::HIGHERORDER);
    Core::Geo::checkRoughGeoType(element, xyze_element, eleGeoType);

    const Core::LinAlg::Matrix<3, 2> xaabbEle =
        Core::Geo::computeFastXAABB(element->shape(), xyze_element, eleGeoType);
    XAABB = mergeAABB(XAABB, xaabbEle);
  }
  return XAABB;
}

/*----------------------------------------------------------------------*
 | delivers a slightly enlarged axis-aligned bounding box for u.may09/09|
 | given elements with their current postions for sliding ALE           |
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 2> Core::Geo::getXAABBofEles(
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions)
{
  Core::LinAlg::Matrix<3, 2> XAABB(true);
  if (elements.begin() == elements.end()) return XAABB;

  // initialize XAABB as rectangle around the first point of the first element
  const int nodeid = elements.begin()->second->nodes()[0]->id();
  const Core::LinAlg::Matrix<3, 1> pos = currentpositions.find(nodeid)->second;
  for (int dim = 0; dim < 3; ++dim)
  {
    XAABB(dim, 0) = pos(dim) - Core::Geo::TOL7;
    XAABB(dim, 1) = pos(dim) + Core::Geo::TOL7;
  }

  std::map<int, Teuchos::RCP<Core::Elements::Element>>::const_iterator elemiter;
  for (elemiter = elements.begin(); elemiter != elements.end(); ++elemiter)
  {
    Teuchos::RCP<Core::Elements::Element> currelement = elemiter->second;
    const Core::LinAlg::SerialDenseMatrix xyze_element(
        Core::Geo::getCurrentNodalPositions(currelement, currentpositions));
    Core::Geo::EleGeoType eleGeoType(Core::Geo::HIGHERORDER);
    Core::Geo::checkRoughGeoType(currelement, xyze_element, eleGeoType);
    const Core::LinAlg::Matrix<3, 2> xaabbEle =
        Core::Geo::computeFastXAABB(currelement->shape(), xyze_element, eleGeoType);
    XAABB = mergeAABB(XAABB, xaabbEle);
  }

  return XAABB;
}

/*----------------------------------------------------------------------*
 | delivers a axis-aligned bounding box for a given          peder 07/08|
 | discretization in reference configuration                            |
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 2> Core::Geo::getXAABBofDis(const Core::FE::Discretization& dis)
{
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions;

  for (int lid = 0; lid < dis.num_my_col_nodes(); ++lid)
  {
    const Core::Nodes::Node* node = dis.l_col_node(lid);
    Core::LinAlg::Matrix<3, 1> currpos;
    currpos(0) = node->x()[0];
    currpos(1) = node->x()[1];
    currpos(2) = node->x()[2];
    currentpositions[node->id()] = currpos;
  }

  return getXAABBofDis(dis, currentpositions);
}


/*----------------------------------------------------------------------*
 | delivers an axis-aligned bounding box for coords          ghamm 11/12|
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 2> Core::Geo::getXAABBofPositions(
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions)
{
  Core::LinAlg::Matrix<3, 2> XAABB(true);

  if (currentpositions.size() == 0) FOUR_C_THROW("map with current positions is emtpy");

  // initialize XAABB as rectangle around the first node
  const Core::LinAlg::Matrix<3, 1> initcurrentpos = currentpositions.begin()->second;
  for (int dim = 0; dim < 3; ++dim)
  {
    XAABB(dim, 0) = initcurrentpos(dim) - Core::Geo::TOL7;
    XAABB(dim, 1) = initcurrentpos(dim) + Core::Geo::TOL7;
  }

  // loop over remaining entries and merge XAABB with their eXtendedAxisAlignedBoundingBox
  std::map<int, Core::LinAlg::Matrix<3, 1>>::const_iterator iter;
  for (iter = currentpositions.begin(); iter != currentpositions.end(); ++iter)
  {
    const Core::LinAlg::Matrix<3, 1> currentpos = iter->second;
    for (int dim = 0; dim < 3; dim++)
    {
      XAABB(dim, 0) = std::min(XAABB(dim, 0), currentpos(dim) - TOL7);
      XAABB(dim, 1) = std::max(XAABB(dim, 1), currentpos(dim) + TOL7);
    }
  }
  return XAABB;
}

/*----------------------------------------------------------------------*
 | compute AABB s for all labeled strutcures in the          u.may 08/08|
 | element list                                                         |
 *----------------------------------------------------------------------*/
std::vector<Core::LinAlg::Matrix<3, 2>> Core::Geo::computeXAABBForLabeledStructures(
    const Core::FE::Discretization& dis,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const std::map<int, std::set<int>>& elementList)
{
  std::vector<Core::LinAlg::Matrix<3, 2>> XAABBs;

  for (std::map<int, std::set<int>>::const_iterator labelIter = elementList.begin();
       labelIter != elementList.end(); labelIter++)
  {
    Core::LinAlg::Matrix<3, 2> xaabb_label;
    // initialize xaabb_label with box around first point
    const int eleId = *((labelIter->second).begin());
    const int nodeId = dis.g_element(eleId)->nodes()[0]->id();
    const Core::LinAlg::Matrix<3, 1> pos = currentpositions.find(nodeId)->second;
    for (int dim = 0; dim < 3; ++dim)
    {
      xaabb_label(dim, 0) = pos(dim) - Core::Geo::TOL7;
      xaabb_label(dim, 1) = pos(dim) + Core::Geo::TOL7;
    }
    // run over set elements
    for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
         eleIter != (labelIter->second).end(); eleIter++)
    {
      const Core::Elements::Element* element = dis.g_element(*eleIter);
      const Core::LinAlg::SerialDenseMatrix xyze_element(
          Core::Geo::getCurrentNodalPositions(element, currentpositions));
      Core::Geo::EleGeoType eleGeoType(Core::Geo::HIGHERORDER);
      Core::Geo::checkRoughGeoType(element, xyze_element, eleGeoType);
      Core::LinAlg::Matrix<3, 2> xaabbEle =
          Core::Geo::computeFastXAABB(element->shape(), xyze_element, eleGeoType);
      xaabb_label = mergeAABB(xaabb_label, xaabbEle);
    }
    XAABBs.push_back(xaabb_label);
  }
  return XAABBs;
}

/*----------------------------------------------------------------------*
 | a set of nodes in a given radius                          u.may 07/08|
 | from a query point                                                   |
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>> Core::Geo::getElementsInRadius(const Core::FE::Discretization& dis,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Matrix<3, 1>& querypoint, const double radius, const int label,
    std::map<int, std::set<int>>& elementList)
{
  std::map<int, std::set<int>> nodeList;
  std::map<int, std::set<int>> elementMap;

  // collect all nodes with different label
  for (std::map<int, std::set<int>>::const_iterator labelIter = elementList.begin();
       labelIter != elementList.end(); labelIter++)
  {
    if (label != labelIter->first)  // don't collect nodes which belong to the same label
    {
      for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
           eleIter != (labelIter->second).end(); eleIter++)
      {
        Core::Elements::Element* element = dis.g_element(*eleIter);
        for (int i = 0; i < Core::FE::getNumberOfElementCornerNodes(element->shape()); i++)
          nodeList[labelIter->first].insert(element->node_ids()[i]);
      }
    }
  }

  for (std::map<int, std::set<int>>::const_iterator labelIter = nodeList.begin();
       labelIter != nodeList.end(); labelIter++)
    for (std::set<int>::const_iterator nodeIter = (labelIter->second).begin();
         nodeIter != (labelIter->second).end(); nodeIter++)
    {
      double distance = std::numeric_limits<double>::max();
      const Core::Nodes::Node* node = dis.g_node(*nodeIter);
      Core::Geo::getDistanceToPoint(node, currentpositions, querypoint, distance);

      if (distance < (radius + Core::Geo::TOL7))
      {
        for (int i = 0; i < dis.g_node(*nodeIter)->num_element(); i++)
          elementMap[labelIter->first].insert(dis.g_node(*nodeIter)->elements()[i]->id());
      }
    }

  return elementMap;
}

/*----------------------------------------------------------------------*
 | a vector of intersection elements                         u.may 02/09|
 | for a given query element (CONTACT)                                  | |
 *----------------------------------------------------------------------*/
void Core::Geo::searchCollisions(const std::map<int, Core::LinAlg::Matrix<3, 2>>& currentBVs,
    const Core::LinAlg::Matrix<3, 2>& queryBV, const int label,
    const std::map<int, std::set<int>>& elementList, std::set<int>& collisions)
{
  // loop over all entries of elementList (= intersection candidates) with different label
  // run over global ids
  for (std::map<int, std::set<int>>::const_iterator labelIter = elementList.begin();
       labelIter != elementList.end(); labelIter++)
  {
    if (label != labelIter->first)  // don t collect nodes which belong to the same label
    {
      for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
           eleIter != (labelIter->second).end(); eleIter++)
      {
        if (intersectionOfBVs(queryBV, currentBVs.find(*eleIter)->second))
        {
          collisions.insert(*eleIter);
        }
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | a vector of intersection elements                         u.may 02/09|
 | for a given query element (CONTACT)                                  | |
 *----------------------------------------------------------------------*/
void Core::Geo::searchCollisions(const std::map<int, Core::LinAlg::Matrix<9, 2>>& currentKDOPs,
    const Core::LinAlg::Matrix<9, 2>& queryKDOP, const int label,
    const std::map<int, std::set<int>>& elementList, std::set<int>& contactEleIds)
{
  // loop over all entries of elementList (= intersection candidates) with different label
  // run over global ids
  for (std::map<int, std::set<int>>::const_iterator labelIter = elementList.begin();
       labelIter != elementList.end(); labelIter++)
  {
    if (label != labelIter->first)  // don t collect nodes which belong to the same label
    {
      for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
           eleIter != (labelIter->second).end(); eleIter++)
      {
        if (intersectionOfKDOPs(queryKDOP, currentKDOPs.find(*eleIter)->second))
          contactEleIds.insert(*eleIter);
      }
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 | gives the coords of the nearest point on or in an object in  tk 01/10|
 | tree node; object is either a node or a line                         |
 *----------------------------------------------------------------------*/
void Core::Geo::nearest2DObjectInNode(const Teuchos::RCP<Core::FE::Discretization> dis,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const std::map<int, std::set<int>>& elementList, const Core::LinAlg::Matrix<3, 1>& point,
    Core::LinAlg::Matrix<3, 1>& minDistCoords)
{
  Core::Geo::NearestObject nearestObject;
  bool pointFound = false;
  double min_distance = std::numeric_limits<double>::max();
  double distance = std::numeric_limits<double>::max();
  Core::LinAlg::Matrix<3, 1> normal(true);
  Core::LinAlg::Matrix<3, 1> x_surface(true);
  std::map<int, std::set<int>> nodeList;

  // run over all line elements
  for (std::map<int, std::set<int>>::const_iterator labelIter = elementList.begin();
       labelIter != elementList.end(); labelIter++)
    for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
         eleIter != (labelIter->second).end(); eleIter++)
    {
      // not const because otherwise no lines can be obtained
      Core::Elements::Element* element = elements[*eleIter].get();
      pointFound =
          Core::Geo::getDistanceToLine(element, currentpositions, point, x_surface, distance);
      if (pointFound && distance < min_distance)
      {
        pointFound = false;
        nearestObject.set_line_object_type(1, *eleIter, labelIter->first, x_surface);
        min_distance = distance;
      }

      // collect nodes
      for (int i = 0; i < Core::FE::getNumberOfElementCornerNodes(element->shape()); i++)
        nodeList[labelIter->first].insert(element->node_ids()[i]);
    }

  // run over all nodes
  for (std::map<int, std::set<int>>::const_iterator labelIter = nodeList.begin();
       labelIter != nodeList.end(); labelIter++)
    for (std::set<int>::const_iterator nodeIter = (labelIter->second).begin();
         nodeIter != (labelIter->second).end(); nodeIter++)
    {
      const Core::Nodes::Node* node = dis->g_node(*nodeIter);
      Core::Geo::getDistanceToPoint(node, currentpositions, point, distance);
      if (distance < min_distance)
      {
        min_distance = distance;
        nearestObject.set_node_object_type(
            *nodeIter, labelIter->first, currentpositions.find(node->id())->second);
      }
    }

  if (nearestObject.get_object_type() == Core::Geo::NOTYPE_OBJECT)
    FOUR_C_THROW("no nearest object obtained");

  // save projection point
  minDistCoords = nearestObject.get_phys_coord();

  return;
}

/*----------------------------------------------------------------------*
 | gives the coords of the nearest point on or in an object in  tk 01/10|
 | tree node; object is either a node, line or surface element;         |
 | also surface id of nearest object is returned, in case of a          |
 | line or node, a random adjacent surface id is returned               |
 *----------------------------------------------------------------------*/
int Core::Geo::nearest3DObjectInNode(const Teuchos::RCP<Core::FE::Discretization> dis,
    std::map<int, Teuchos::RCP<Core::Elements::Element>>& elements,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const std::map<int, std::set<int>>& elementList, const Core::LinAlg::Matrix<3, 1>& point,
    Core::LinAlg::Matrix<3, 1>& minDistCoords)
{
  Core::Geo::NearestObject nearestObject;
  bool pointFound = false;
  double min_distance = std::numeric_limits<double>::max();
  double distance = std::numeric_limits<double>::max();
  Core::LinAlg::Matrix<3, 1> normal(true);
  Core::LinAlg::Matrix<3, 1> x_surface(true);
  std::map<int, std::set<int>> nodeList;
  int surfid = -1;

  // run over all surface elements
  for (std::map<int, std::set<int>>::const_iterator labelIter = elementList.begin();
       labelIter != elementList.end(); labelIter++)
    for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
         eleIter != (labelIter->second).end(); eleIter++)
    {
      // not const because otherwise no lines can be obtained
      Core::Elements::Element* element = elements[*eleIter].get();
      pointFound =
          Core::Geo::getDistanceToSurface(element, currentpositions, point, x_surface, distance);
      if (pointFound && distance < min_distance)
      {
        pointFound = false;
        min_distance = distance;
        nearestObject.set_surface_object_type(*eleIter, labelIter->first, x_surface);
        surfid = element->id();
      }

      // run over all line elements
      const std::vector<Teuchos::RCP<Core::Elements::Element>> eleLines = element->lines();
      for (int i = 0; i < element->num_line(); i++)
      {
        pointFound = Core::Geo::getDistanceToLine(
            eleLines[i].get(), currentpositions, point, x_surface, distance);
        if (pointFound && distance < min_distance)
        {
          pointFound = false;
          min_distance = distance;
          nearestObject.set_line_object_type(i, *eleIter, labelIter->first, x_surface);
          surfid = element->id();
        }
      }
      // collect nodes
      for (int i = 0; i < Core::FE::getNumberOfElementCornerNodes(element->shape()); i++)
        nodeList[labelIter->first].insert(element->node_ids()[i]);
    }

  // run over all nodes
  for (std::map<int, std::set<int>>::const_iterator labelIter = nodeList.begin();
       labelIter != nodeList.end(); labelIter++)
    for (std::set<int>::const_iterator nodeIter = (labelIter->second).begin();
         nodeIter != (labelIter->second).end(); nodeIter++)
    {
      const Core::Nodes::Node* node = dis->g_node(*nodeIter);
      Core::Geo::getDistanceToPoint(node, currentpositions, point, distance);
      if (distance < min_distance)
      {
        min_distance = distance;
        nearestObject.set_node_object_type(
            *nodeIter, labelIter->first, currentpositions.find(node->id())->second);
        surfid = node->elements()[0]->id();  // surf id of any of the adjacent elements
      }
    }

  if (nearestObject.get_object_type() == Core::Geo::NOTYPE_OBJECT)
    FOUR_C_THROW("no nearest object obtained");

  // save projection point
  minDistCoords = nearestObject.get_phys_coord();

  return surfid;
}

/*----------------------------------------------------------------------*
 | gives the coords of the nearest point on a surface        ghamm 09/13|
 | element and return type of nearest object                            |
 *----------------------------------------------------------------------*/
Core::Geo::ObjectType Core::Geo::nearest3DObjectOnElement(Core::Elements::Element* surfaceelement,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Matrix<3, 1>& point, Core::LinAlg::Matrix<3, 1>& minDistCoords)
{
  Core::Geo::NearestObject nearestObject;
  bool pointFound = false;
  double min_distance = std::numeric_limits<double>::max();
  double distance = std::numeric_limits<double>::max();
  Core::LinAlg::Matrix<3, 1> x_surface(true);

  pointFound =
      Core::Geo::getDistanceToSurface(surfaceelement, currentpositions, point, x_surface, distance);
  if (pointFound && distance <= min_distance)
  {
    pointFound = false;
    min_distance = distance;
    nearestObject.set_surface_object_type(surfaceelement->id(), -1, x_surface);
  }

  // run over all line elements
  const std::vector<Teuchos::RCP<Core::Elements::Element>> eleLines = surfaceelement->lines();
  for (int i = 0; i < surfaceelement->num_line(); i++)
  {
    pointFound = Core::Geo::getDistanceToLine(
        eleLines[i].get(), currentpositions, point, x_surface, distance);
    if (pointFound && distance <= min_distance)
    {
      pointFound = false;
      min_distance = distance;
      nearestObject.set_line_object_type(i, surfaceelement->id(), -1, x_surface);
    }
  }

  // run over all nodes
  for (std::map<int, Core::LinAlg::Matrix<3, 1>>::const_iterator nodeIter =
           currentpositions.begin();
       nodeIter != currentpositions.end(); nodeIter++)
  {
    Core::LinAlg::Matrix<3, 1> distance_vector;
    // vector pointing away from the node towards physCoord
    distance_vector.update(1.0, point, -1.0, nodeIter->second);

    // absolute distance between point and node
    distance = distance_vector.norm2();

    if (distance <= min_distance)
    {
      min_distance = distance;
      nearestObject.set_node_object_type(nodeIter->first, -1, nodeIter->second);
    }
  }

  if (nearestObject.get_object_type() == Core::Geo::NOTYPE_OBJECT)
    FOUR_C_THROW("no nearest object obtained");

  // save projection point
  minDistCoords = nearestObject.get_phys_coord();

  return nearestObject.get_object_type();
}

/*----------------------------------------------------------------------*
 |  computes the normal distance from a point to a           u.may 07/08|
 |  surface element, if it exits                                        |
 *----------------------------------------------------------------------*/
bool Core::Geo::getDistanceToSurface(const Core::Elements::Element* surfaceElement,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Matrix<3, 1>& point, Core::LinAlg::Matrix<3, 1>& x_surface_phys,
    double& distance)
{
  bool pointFound = false;
  double min_distance = std::numeric_limits<double>::max();
  Core::LinAlg::Matrix<3, 1> distance_vector(true);
  Core::LinAlg::Matrix<2, 1> elecoord(true);  // starting value at element center

  const Core::LinAlg::SerialDenseMatrix xyze_surfaceElement(
      Core::Geo::getCurrentNodalPositions(surfaceElement, currentpositions));
  Core::Geo::CurrentToSurfaceElementCoordinates(
      surfaceElement->shape(), xyze_surfaceElement, point, elecoord);

  if (Core::Geo::checkPositionWithinElementParameterSpace(elecoord, surfaceElement->shape()))
  {
    Core::Geo::elementToCurrentCoordinates(
        surfaceElement->shape(), xyze_surfaceElement, elecoord, x_surface_phys);
    // normal pointing away from the surface towards point
    distance_vector.update(1.0, point, -1.0, x_surface_phys);
    distance = distance_vector.norm2();
    min_distance = distance;
    pointFound = true;
  }

  // if the element is curved
  Core::Geo::EleGeoType eleGeoType = Core::Geo::HIGHERORDER;
  checkRoughGeoType(surfaceElement, xyze_surfaceElement, eleGeoType);

  // TODO fix check if deformed in the linear case
  if (eleGeoType == Core::Geo::HIGHERORDER)
  {
    Core::LinAlg::SerialDenseMatrix eleCoordMatrix =
        Core::FE::getEleNodeNumbering_nodes_paramspace(surfaceElement->shape());
    for (int i = 0; i < surfaceElement->num_node(); i++)
    {
      // use nodes as starting values
      for (int j = 0; j < 2; j++) elecoord(j) = eleCoordMatrix(j, i);

      Core::Geo::CurrentToSurfaceElementCoordinates(
          surfaceElement->shape(), xyze_surfaceElement, point, elecoord);

      if (Core::Geo::checkPositionWithinElementParameterSpace(elecoord, surfaceElement->shape()))
      {
        Core::LinAlg::Matrix<3, 1> physcoord(true);
        Core::Geo::elementToCurrentCoordinates(
            surfaceElement->shape(), xyze_surfaceElement, elecoord, physcoord);
        // normal pointing away from the surface towards point
        distance_vector.update(1.0, point, -1.0, physcoord);
        distance = distance_vector.norm2();
        if (distance < min_distance)
        {
          x_surface_phys = physcoord;
          min_distance = distance;
          pointFound = true;
        }
      }
    }
  }
  return pointFound;
}

/*----------------------------------------------------------------------*
 |  computes the normal distance from a point to a           u.may 07/08|
 |  line element, if it exits                                           |
 *----------------------------------------------------------------------*/
bool Core::Geo::getDistanceToLine(const Core::Elements::Element* lineElement,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Matrix<3, 1>& point, Core::LinAlg::Matrix<3, 1>& x_line_phys,
    double& distance)
{
  bool pointFound = false;
  double min_distance = std::numeric_limits<double>::max();
  Core::LinAlg::Matrix<3, 1> distance_vector(true);
  Core::LinAlg::Matrix<1, 1> elecoord(true);  // starting value at element center

  const Core::LinAlg::SerialDenseMatrix xyze_lineElement(
      Core::Geo::getCurrentNodalPositions(lineElement, currentpositions));
  Core::Geo::CurrentToLineElementCoordinates(
      lineElement->shape(), xyze_lineElement, point, elecoord);

  if (Core::Geo::checkPositionWithinElementParameterSpace(elecoord, lineElement->shape()))
  {
    Core::Geo::elementToCurrentCoordinates(
        lineElement->shape(), xyze_lineElement, elecoord, x_line_phys);
    // normal pointing away from the line towards point
    distance_vector.update(1.0, point, -1.0, x_line_phys);
    distance = distance_vector.norm2();
    min_distance = distance;
    pointFound = true;
  }

  // if the element is curved
  Core::Geo::EleGeoType eleGeoType = Core::Geo::HIGHERORDER;
  checkRoughGeoType(lineElement, xyze_lineElement, eleGeoType);

  if (eleGeoType == Core::Geo::HIGHERORDER)
  {
    Core::LinAlg::SerialDenseMatrix eleCoordMatrix =
        Core::FE::getEleNodeNumbering_nodes_paramspace(lineElement->shape());
    // use end nodes as starting values in addition
    for (int i = 0; i < 2; i++)
    {
      // use end nodes as starting values in addition
      elecoord(0) = eleCoordMatrix(0, i);

      Core::Geo::CurrentToLineElementCoordinates(
          lineElement->shape(), xyze_lineElement, point, elecoord);

      if (Core::Geo::checkPositionWithinElementParameterSpace(elecoord, lineElement->shape()))
      {
        Core::LinAlg::Matrix<3, 1> physcoord(true);
        Core::Geo::elementToCurrentCoordinates(
            lineElement->shape(), xyze_lineElement, elecoord, physcoord);
        // normal pointing away from the line towards point
        distance_vector.update(1.0, point, -1.0, physcoord);
        distance = distance_vector.norm2();
        if (distance < min_distance)
        {
          x_line_phys = physcoord;
          min_distance = distance;
          pointFound = true;
        }
      }
    }
  }
  return pointFound;
}


/*----------------------------------------------------------------------*
 |  computes the distance from a point to a node             u.may 07/08|
 |  of an element                                                       |
 *----------------------------------------------------------------------*/
void Core::Geo::getDistanceToPoint(const Core::Nodes::Node* node,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Matrix<3, 1>& point, double& distance)
{
  // node position in physical coordinates
  const Core::LinAlg::Matrix<3, 1> x_node = currentpositions.find(node->id())->second;

  Core::LinAlg::Matrix<3, 1> distance_vector;
  // vector pointing away from the node towards physCoord
  distance_vector.update(1.0, point, -1.0, x_node);

  // absolute distance between point and node
  distance = distance_vector.norm2();
}

/*----------------------------------------------------------------------*
 | check s if nearest point found on nearest object          u.may 09/08|
 | lies in a tree node specified by its box                             |
 *----------------------------------------------------------------------*/
bool Core::Geo::pointInTreeNode(
    const Core::LinAlg::Matrix<3, 1>& point, const Core::LinAlg::Matrix<3, 2>& nodeBox)
{
  for (int dim = 0; dim < 3; dim++)
    if ((point(dim) < (nodeBox(dim, 0) - Core::Geo::TOL6)) ||
        (point(dim) > (nodeBox(dim, 1) + Core::Geo::TOL6)))
      return false;

  return true;
}

/*----------------------------------------------------------------------*
 | merge two AABB and deliver the resulting AABB's           peder 07/08|
 *----------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 2> Core::Geo::mergeAABB(
    const Core::LinAlg::Matrix<3, 2>& AABB1, const Core::LinAlg::Matrix<3, 2>& AABB2)
{
  Core::LinAlg::Matrix<3, 2> mergedAABB;

  for (int dim = 0; dim < 3; dim++)
  {
    mergedAABB(dim, 0) = std::min(AABB1(dim, 0), AABB2(dim, 0));
    mergedAABB(dim, 1) = std::max(AABB1(dim, 1), AABB2(dim, 1));
  }
  return mergedAABB;
}

/*----------------------------------------------------------------------*
 | determines the geometry type of an element                peder 07/08|
 | not for Cartesian elements cjecked because no improvements are       |
 | expected                                                             |
 *----------------------------------------------------------------------*/
void Core::Geo::checkRoughGeoType(const Core::Elements::Element* element,
    const Core::LinAlg::SerialDenseMatrix xyze_element, Core::Geo::EleGeoType& eleGeoType)
{
  const int order = Core::FE::getOrder(element->shape());

  if (order == 1)
    eleGeoType = Core::Geo::LINEAR;  // TODO check for bilinear elements in the tree they count
                                     // as higerorder fix it
  else if (order == 2)
    eleGeoType = Core::Geo::HIGHERORDER;
  else
    FOUR_C_THROW("order of element is not correct");

  // TODO check if a higher order element is linear
  // by checking if the higher order node is on the line
}

/*----------------------------------------------------------------------*
 | determines the geometry type of an element                 u.may 09/09|
 |  -->needed for Teuchos::RCP on element                                |
 *----------------------------------------------------------------------*/
void Core::Geo::checkRoughGeoType(const Teuchos::RCP<Core::Elements::Element> element,
    const Core::LinAlg::SerialDenseMatrix xyze_element, Core::Geo::EleGeoType& eleGeoType)
{
  const int order = Core::FE::getOrder(element->shape());

  if (order == 1)
    eleGeoType = Core::Geo::LINEAR;  // TODO check for bilinear elements in the tree they count
                                     // as higerorder fix it
  else if (order == 2)
    eleGeoType = Core::Geo::HIGHERORDER;
  else
    FOUR_C_THROW("order of element is not correct");

  // TODO check if a higher order element is linear
  // by checking if the higher order node is on the line
}

FOUR_C_NAMESPACE_CLOSE
