// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_MESH_HPP
#define FOUR_C_CUT_MESH_HPP

#include "4C_config.hpp"

#include "4C_cut_boundingbox.hpp"
#include "4C_cut_edge.hpp"
#include "4C_cut_facet.hpp"
#include "4C_cut_node.hpp"

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopologyTraits.hpp>

#include <list>

FOUR_C_NAMESPACE_OPEN

namespace Core::Geo
{
  class SearchTree;
}

namespace Cut
{
  class Node;
  class Edge;
  class Side;
  class Element;

  class BoundingBox;

  class PointPool;
  class Options;

  class Point;
  class Line;
  class Facet;
  class VolumeCell;

  class BoundaryCell;
  class Point1BoundaryCell;
  class Line2BoundaryCell;
  class Tri3BoundaryCell;
  class Quad4BoundaryCell;
  class ArbitraryBoundaryCell;

  class IntegrationCell;
  class Line2IntegrationCell;
  class Tri3IntegrationCell;
  class Quad4IntegrationCell;
  class Hex8IntegrationCell;
  class Tet4IntegrationCell;
  class Wedge6IntegrationCell;
  class Pyramid5IntegrationCell;



  /*!
  \brief All the geometrical entities (mesh, volume, cut surface, nodes etc.) are contained in
  this.

    There is one background mesh and one mesh for the cut surface. These meshes
    have different objects but share the same points via the point pool.

    Mesh does the memory management for the whole thing. Therefore, all
    creation of cut library objects is done via the mesh.
   */
  class Mesh
  {
   public:
    /// constructor
    Mesh(Options& options, double norm = 1, std::shared_ptr<PointPool> pp = nullptr,
        bool cutmesh = false, int myrank = -1);

    /*========================================================================*/
    //! @name general create-routines for elements and sides
    /*========================================================================*/

    /// creates a new element, dependent on distype
    Element* create_element(int eid, const std::vector<int>& nids, Core::FE::CellType distype);

    /// creates a new side, dependent on distype
    Side* create_side(int sid, const std::vector<int>& nids, Core::FE::CellType distype);

    /*========================================================================*/
    //! @name Create-routines for elements
    /*========================================================================*/
    /// @{
    /// creates a new line2 element based on element id and node ids
    Element* create_line2(int eid, const std::vector<int>& nids);

    /// creates a new tri3 element based on element id and node ids
    Element* create_tri3(int eid, const std::vector<int>& nids);

    /// creates a new quad4 element based on element id and node ids
    Element* create_quad4(int eid, const std::vector<int>& nids);

    /// creates a new tet4 element based on element id and node ids
    Element* create_tet4(int eid, const std::vector<int>& nids);

    /// creates a new pyramid5 element based on element id and node ids
    Element* create_pyramid5(int eid, const std::vector<int>& nids);

    /// creates a new wedge6 element based on element id and node ids
    Element* create_wedge6(int eid, const std::vector<int>& nids);

    /// creates a new hex8 element based on element id and node ids
    Element* create_hex8(int eid, const std::vector<int>& nids);
    /// @}

    /*========================================================================*/
    //! @name Create-routines for sides
    /*========================================================================*/
    /// @{
    /// creates a new tri3 side based on side id and node ids
    Side* create_tri3_side(int sid, const std::vector<int>& nids);

    /// creates a new quad4 side based on side id and node ids
    Side* create_quad4_side(int sid, const std::vector<int>& nids);
    /// @}

    /*========================================================================*/
    //! @name Create-routines for points, lines, facets and volumecells
    /*========================================================================*/

    /// creates a new point, optional information about cut-edge and cut-side, and whether this is
    /// happening during
    // loading of the mesh
    Point* new_point(
        const double* x, Edge* cut_edge, Side* cut_side, double tolerance, double tol_scale = 1.0);

    /// creates a new line
    void new_line(Point* p1, Point* p2, Side* cut_side1, Side* cut_side2, Element* cut_element,
        std::vector<Line*>* newlines = nullptr);

    /// ?
    bool new_lines_between(const std::vector<Point*>& line, Side* cut_side1, Side* cut_side2,
        Element* cut_element, std::vector<Line*>* newlines = nullptr);

    /// creates a new facet, consists of points, additional bool if it is a facet on a cutsurface
    Facet* new_facet(const std::vector<Point*>& points, Side* side, bool cutsurface);

    /// creates a new volumecell, consists of facets
    VolumeCell* new_volume_cell(const plain_facet_set& facets,
        const std::map<std::pair<Point*, Point*>, plain_facet_set>& volume_lines, Element* element);

    /*========================================================================*/
    //! @name Create-routines for 0D boundary cells
    /*========================================================================*/

    /// creates a new point1 boundary cell
    Point1BoundaryCell* new_point1_cell(
        VolumeCell* volume, Facet* facet, const std::vector<Point*>& points);

    /*========================================================================*/
    //! @name Create-routines for 1D boundary cells
    /*========================================================================*/

    Line2BoundaryCell* new_line2_cell(
        VolumeCell* volume, Facet* facet, const std::vector<Point*>& points);

    /*========================================================================*/
    //! @name Create-routines for 2D boundary cells
    /*========================================================================*/

    /// creates a new tri3 boundary cell
    Tri3BoundaryCell* new_tri3_cell(
        VolumeCell* volume, Facet* facet, const std::vector<Point*>& points);

    /// creates a new quad4 boundary cell
    Quad4BoundaryCell* new_quad4_cell(
        VolumeCell* volume, Facet* facet, const std::vector<Point*>& points);

    /// creates a new ??? boundary cell
    ArbitraryBoundaryCell* new_arbitrary_cell(VolumeCell* volume, Facet* facet,
        const std::vector<Point*>& points, const Core::FE::GaussIntegration& gaussRule,
        const Core::LinAlg::Matrix<3, 1>& normal);


    /*========================================================================*/
    //! @name Create-routines for 1D integration cells
    /*========================================================================*/

    /// creates a new line2 integration cell
    Line2IntegrationCell* new_line2_cell(
        Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell);

    /*========================================================================*/
    //! @name Create-routines for 2D integration cells
    /*========================================================================*/

    /// creates a new tri3 integration cell
    Tri3IntegrationCell* new_tri3_cell(
        Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell);

    /// creates a new tri3 integration cell
    Quad4IntegrationCell* new_quad4_cell(
        Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell);


    /*========================================================================*/
    //! @name Create-routines for 3D integration cells
    /*========================================================================*/

    /// creates a new hex8 integration cell
    Hex8IntegrationCell* new_hex8_cell(
        Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell);

    /// creates a new tet4 integration cell, based on points
    Tet4IntegrationCell* new_tet4_cell(
        Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell);

    /// creates a new hex8 integration cell, based on xyz coordinates
    Tet4IntegrationCell* new_tet4_cell(Point::PointPosition position,
        const Core::LinAlg::SerialDenseMatrix& xyz, VolumeCell* cell);

    /// creates a new wedge6 integration cell
    Wedge6IntegrationCell* new_wedge6_cell(
        Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell);

    /// creates a new pyramid5 integration cell
    Pyramid5IntegrationCell* new_pyramid5_cell(
        Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell);


    /*========================================================================*/
    //! @name Basic routines to run the collision detection
    /*========================================================================*/

    /// build the static search tree for the collision detection in the self cut
    void build_self_cut_tree();

    /// build the static search tree for the collision detection
    void build_static_search_tree();

    /// detects if a side of the cut mesh possibly collides with an element of the background mesh
    void search_collisions(Mesh& cutmesh);


    /*========================================================================*/
    //! @name Basic routines to run the cut, creates a cloud of cut points
    /*========================================================================*/

    /// Cuts the background elements of the mesh with all the cut sides
    void cut(Mesh& mesh, plain_element_set& elements_done);

    /// Cuts the background elements with this considered side
    void cut(Side& side, const plain_element_set& done, plain_element_set& elements_done);

    /// Cuts the background elements with this levelset side
    void cut(Side& side);

    /// Used in TetMeshIntersection, however, unclear for what use??!
    void rectify_cut_numerics();

    /// finds intersections between sides and edges
    void find_cut_points();

    /*========================================================================*/
    //! @name Basic routines to create lines, facets and volumecells
    /*========================================================================*/

    /// create cut lines based on the point cloud
    void make_cut_lines();

    /// create facets based on the cut lines
    void make_facets();

    /// create volumecells based on created facets
    void make_volume_cells();


    /*========================================================================*/
    //! @name Basic routines to determine positions of nodes, facets, volumecells
    /*========================================================================*/

    /// find node positions and propagate the positions to facets, points and volumecells
    void find_node_positions();

    /// ?
    void find_ls_node_positions();

    /// find facet positions for remaining facets, points, volumecells that have not been found
    /// using FindNodePositions()
    void find_facet_positions();

    /// Check if there are nodes whose position is undecided (often the case in parallel), return
    /// whether undecided node positions available
    bool check_for_undecided_node_positions(
        std::map<int, int>& undecided_node, std::map<plain_int_set, int>& undecided_shadow_nodes);


    /*========================================================================*/
    //! @name Basic routines to determine nodal dofsets
    /*========================================================================*/

    /// still used???
    void find_nodal_dof_sets(bool include_inner);


    /*========================================================================*/
    //! @name Basic routines to create integration cells and/or integration points
    /*========================================================================*/

    /// Execute Tessellation for each element to generate integrationcells
    void create_integration_cells(int count, bool tetcellsonly = false);

    /// Call the moment fitting method for each element to generate the Gaussian integration rule
    void moment_fit_gauss_weights(bool include_inner, Cut::BCellGaussPts Bcellgausstype);

    /// Call the DirectDivergence method for each element to generate the Gaussian integration
    /// rule
    void direct_divergence_gauss_rule(bool include_inner, Cut::BCellGaussPts Bcellgausstype);


    /*========================================================================*/
    //! @name others ?
    /*========================================================================*/

    /// ?
    void remove_empty_volume_cells();

    /// test if for all elements the element volume is equal to the volume of all integration
    /// cells
    void test_element_volume(bool fatal, VCellGaussPts VCellGP = VCellGaussPts_Tessellation);

    /*!
    \brief Find the difference between the volume of background element and the sum of volume of
    all integration cells. There should be no difference between these two
     */
    void test_element_volume(
        Core::FE::CellType shape, Element& e, bool fatal, VCellGaussPts VCellGP);


    //! Return the elements of this mesh
    std::map<int, std::shared_ptr<Element>> get_mesh_elements() { return elements_; }

    /*========================================================================*/
    //! @name print statistics
    /*========================================================================*/

    /// ???
    void print_cell_stats();

    /// print all facets
    void print_facets();


    /*========================================================================*/
    //! @name GMSH output routines
    /*========================================================================*/

    /// Write full Gmsh Output
    void dump_gmsh(std::string name);

    /*!
    \brief Output information about the volume cell.
    If the cut has a level set side. Also write output for level set values and gradients.
     */
    void dump_gmsh_volume_cells(std::string name, bool include_inner);

    /// ?
    void dump_gmsh_integration_cells(std::string name);

    /* writer boundary cells belonging to a volume cell with "pos" position
     * and their normals to the given file */
    void dump_gmsh_boundary_cells(std::ofstream& file, Point::PointPosition pos);

    /// ?
    void dump_gmsh_volume_cells(std::string name);

    /// DebugDump to call before runtime error!!!
    void debug_dump(Cut::Element* ele, std::string file = "", int line = -1);


    /*========================================================================*/
    //! @name Get routines for nodes, sides and elements
    /*========================================================================*/

    /// ? -> used in cut_tetmeshintersection
    void new_nodes_from_points(std::map<Point*, Node*>& nodemap);

    /// get a map of node id and the pointer to the node
    void get_node_map(std::map<int, Node*>& nodemap);

    /// Returns the node with given id
    Node* get_node(int nid) const;

    /*!
    \brief Returns the unique shadow node
    identified by given nids of a quad8 boundary side or all nodes of hex20 element
    for the inner shadow node
     */
    Node* get_node(const plain_int_set& nids) const;

    /*!
    \brief If node with the given id exists return the node, else create a new node with
    given coordinates and levelset value
     */
    Node* get_node(int nid, const double* xyz, double lsv = 0.0, double tolerance = 0.0);

    /// ?
    Node* get_node(const plain_int_set& nids, const double* xyz, double lsv = 0.0);

    /// get the edge with begin node and end node
    Edge* get_edge(Node* begin, Node* end);

    /// get the side that contains the nodes with the following node ids
    Side* get_side(std::vector<int>& nids) const;

    /// ???
    Side* get_side(int sid, const std::vector<int>& nids, const CellTopologyData* top_data);

    /// ???
    Side* get_side(int sid, const std::vector<Node*>& nodes, const CellTopologyData* top_data);

    /// Returns the element with given id
    Element* get_element(int eid);

    /*!
    \brief  If element with the given id exists return the element, else create a new element
    with given node ids and details given in cell topology data
     */
    Element* get_element(int eid, const std::vector<int>& nids, const CellTopologyData& top_data,
        bool active = true);

    /*! \brief Create a new element 1D/2D/3D element with given nodes.
     *
     *  All details of the element are in cell topology data. */
    Cut::Element* get_element(
        int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active);

    /*! \brief Create a new element with desired element dimension
     *
     *  dim = 1: Create a new element 1D/line element with given nodes.
     *
     *  All details of the element are in cell topology data.
     *  This routine uses the elements (1-D) as edges (1-D) and sides (1-D).
     *
     *  dim = 1: Create a new element 2D/surface element with given nodes.
     *
     *  All details of the element are in cell topology data.
     *  This routine creates new sides (1-D) and uses them also as edges (1-D).
     *
     *  dim = 3: Create a new element 3D element with given nodes.
     *
     *  All details of the element (3-D) are in cell topology data.
     *  This routine creates also the corresponding edges (1-D) and sides (2-D) for
     *  3-dimensional elements. */
    template <unsigned dim>
    Element* get_element(int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data,
        bool active = true);

    /*========================================================================*/
    //! @name Get routines for points, edges, volumecells
    /*========================================================================*/

    /// get the octTree based PointPool that contains all points of the current mesh
    std::shared_ptr<PointPool> points() { return pp_; }

    /// get a list of all volumecells
    const std::list<std::shared_ptr<VolumeCell>>& volume_cells() const { return cells_; }

    /// ???
    const std::map<plain_int_set, std::shared_ptr<Edge>>& edges() const { return edges_; }


    /*========================================================================*/
    //! @name others
    /*========================================================================*/

    /// check if xyz-coordinates lie within the mesh's bounding box
    bool within_bb(const Core::LinAlg::SerialDenseMatrix& xyz);

    /// check if the element lies within the bounding box
    bool within_bb(Element& element);

    //     Only used in cut_test_volume.cpp
    void create_side_ids_cut_test(int lastid = 0);

    // only used for testing
    int create_side_ids_all_cut_test(int lastid = 0);

    //     Only used in cut_test_volume.cpp
    void assign_other_volume_cells_cut_test(const Mesh& other);

    /// return the options
    Options& create_options() { return options_; }

    /// return the options
    Options& get_options() const { return options_; }

    /// ???
    void test_volume_surface();

    /// Test if the area of a cut facet is covered by the same area of boundary cells on both
    /// sides.
    void test_facet_area(bool istetmeshintersection = false);


    /*========================================================================*/
    //! @name Routines which provides the interface to the selfcut
    /*========================================================================*/

    /// Returns all sides of the cutmesh
    const std::map<plain_int_set, std::shared_ptr<Side>>& sides() const { return sides_; }

    /// Returns of search tree all sides of the cutmesh
    const std::shared_ptr<Core::Geo::SearchTree>& self_cut_tree() const { return selfcuttree_; }

    /// Returns the bounding volumes of all sides of the cutmesh
    const std::map<int, Core::LinAlg::Matrix<3, 2>>& self_cut_bvs() const { return selfcutbvs_; }

    /// Returns the map of all sides of the cutmesh
    const std::map<int, Side*>& shadow_sides() const { return shadow_sides_; }

    /// Returns all nodes of the cutmesh
    const std::map<int, std::shared_ptr<Node>>& nodes() const { return nodes_; }

    /// Creates a new node in the cutmesh
    void get_node(int nid, Node* node) { nodes_[nid] = std::shared_ptr<Node>(node); }

    /// Creates a new edge in the cutmesh
    void get_edge(plain_int_set eid, const std::shared_ptr<Edge>& edge) { edges_[eid] = edge; }

    /// Erases a side of the cutmesh
    void erase_side(plain_int_set sid) { sides_.erase(sid); }

    /// Erases a edge of the cutmesh
    void erase_edge(plain_int_set eid) { edges_.erase(eid); }

    /// Erases a node of the cutmesh
    void erase_node(int nid) { nodes_.erase(nid); }

    /// Move this Side from to the Storage container of the Mesh
    void move_sideto_storage(plain_int_set sid)
    {
      storagecontainer_sides_[sid] = sides_[sid];
      sides_.erase(sid);
    }

    /// Move this Node from to the Storage container of the Mesh
    void move_nodeto_storage(int nid)
    {
      storagecontainer_nodes_[nid] = nodes_[nid];
      nodes_.erase(nid);
    }

   private:
    /*========================================================================*/
    //! @name private member functions
    /*========================================================================*/

    /// ?
    Edge* get_edge(const plain_int_set& nids, const std::vector<Node*>& nodes,
        const CellTopologyData& edge_topology);

    /// ?
    Side* get_side(int sid, const plain_int_set& nids, const std::vector<Node*>& nodes,
        const std::vector<Edge*>& edges, const CellTopologyData& side_topology);

    /// Create new line between the two given cut points that are in given two cut sides
    Cut::Line* new_line_internal(
        Point* p1, Point* p2, Side* cut_side1, Side* cut_side2, Element* cut_element);


    /*========================================================================*/
    //! @name private member variables
    /*========================================================================*/

    /// options container
    Options& options_;

    /// mesh dependent point lookup norm
    double norm_;

    /// shared point storage with geometry based access (octtree)
    std::shared_ptr<PointPool> pp_;

    /// bounding box of this mesh
    std::shared_ptr<BoundingBox> bb_;

    /// (output) flag for cut mesh
    ///  TODO: Remove this!
    //     Only used in cut_test_volume.cpp
    bool cutmesh_;

    /// the spatial partitioning octree for a fast collision detection in the self cut
    std::shared_ptr<Core::Geo::SearchTree> selfcuttree_;

    /// the bounding volumes for a fast collision detection in the self cut
    std::map<int, Core::LinAlg::Matrix<3, 2>> selfcutbvs_;

    /// the spatial partitioning octree for a fast collision detection
    std::shared_ptr<Core::Geo::SearchTree> searchtree_;

    /// the bounding volumes for a fast collision detection
    std::map<int, Core::LinAlg::Matrix<3, 2>> boundingvolumes_;

    /*========================================================================*/
    //! @name Containers that hold all those mesh objects
    /*========================================================================*/

    /// Plain pointers are used within the library. Memory management is done here.
    std::list<std::shared_ptr<Line>> lines_;
    std::list<std::shared_ptr<Facet>> facets_;
    std::list<std::shared_ptr<VolumeCell>> cells_;
    std::list<std::shared_ptr<BoundaryCell>> boundarycells_;
    std::list<std::shared_ptr<IntegrationCell>> integrationcells_;

    /// nodes by unique id, contains also shadow nodes with negative node-Ids
    /// Remark: the negative nids of shadow nodes are not unique over processors!
    std::map<int, std::shared_ptr<Node>> nodes_;

    /// edges by set of nodal ids
    std::map<plain_int_set, std::shared_ptr<Edge>> edges_;

    /// sides by set of nodal ids
    std::map<plain_int_set, std::shared_ptr<Side>> sides_;

    /// elements by unique id
    std::map<int, std::shared_ptr<Element>> elements_;

    /// internally generated nodes by nodal ids of element nodes
    /// there is at most one unique shadow node for each 2D side element (no for quad9 side, one
    /// for quad8 side, no for tri6 side, no for linear elements and so on) it is the center node
    /// of a quad8 side in case of a hex20 element, the eight nodes are used as key for the
    /// boundary shadow node the inner center node of hex20 element is a (kind of) unique inner!
    /// shadow node and is stored using all 20 nodes of the hex20 element as key for the map
    /// warning: in the context of the selfcut, nodes are erased for the cut mesh
    /// these nodes are not erased in shadow_nodes_
    /// so don't use the shadow_nodes_ of the cut mesh
    std::map<plain_int_set, Node*> shadow_nodes_;

    // unique map between int ( < 0 ) and element; int is not eid and not parentid
    // we need this for the search tree in the self cut
    // is not erased during self cut!!!
    std::map<int, Side*> shadow_sides_;

    /// new: unique map between int ( < 0 ) and element; int is not eid and not parentid
    std::map<int, std::shared_ptr<Element>> shadow_elements_;

    /// A storage container for sides which should not interact with the CUT anymore (If we want
    /// to access these geometric objects still from outside)
    std::map<plain_int_set, std::shared_ptr<Side>> storagecontainer_sides_;

    /// A storage container for nodes which should not interact with the CUT anymore (If we want
    /// to access these geometric objects still from outside)
    std::map<int, std::shared_ptr<Node>> storagecontainer_nodes_;

    /// processor id --> required just for output!
    int myrank_;

    //@}
  };

  // instantiation of the function template specializations
  template <>
  Element* Mesh::get_element<1>(
      int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active);
  template <>
  Element* Mesh::get_element<2>(
      int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active);
  template <>
  Element* Mesh::get_element<3>(
      int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active);


}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
