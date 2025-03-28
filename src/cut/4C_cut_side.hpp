// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_SIDE_HPP
#define FOUR_C_CUT_SIDE_HPP

#include "4C_config.hpp"

#include "4C_cut_edge.hpp"
#include "4C_cut_element.hpp"
#include "4C_cut_tolerance.hpp"

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopologyTraits.hpp>

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class Facet;
  class LineSegment;
  class BoundaryCell;
  class LineSegmentList;
  class Cycle;
  class BoundingBox;

  enum MarkedActions
  {
    mark_and_do_nothing = -1,
    mark_and_create_boundarycells = 0
  };

  /*! \brief Base class for dealing sides in cut algorithm
   *
   *  The maximal allowed dimension of sides is 2! */
  class Side
  {
   public:
    /** create a new level set side
     *
     *  Create a new level set side using the current problem dimension
     *
     *  \param sid (in) : side id
     *
     */
    static Side* create_level_set_side(const int& sid);

    /// Create a concrete side (using dis-type for identification)
    static Side* create(const Core::FE::CellType& sidetype, const int& sid,
        const std::vector<Node*>& nodes, const std::vector<Edge*>& edges);

    /// Create a concrete side (using shards-key for identification)
    static Side* create(const unsigned& shardskey, const int& sid, const std::vector<Node*>& nodes,
        const std::vector<Edge*>& edges);

    /** \brief constructor
     *
     *  side  (in) : side id
     *  nodes (in) : nodes of this side
     *  edges (in) : edges of this side */
    Side(int sid, const std::vector<Node*>& nodes, const std::vector<Edge*>& edges);

    /// destructor
    virtual ~Side() = default;
    /// \brief Returns the ID of the side
    int id() const { return sid_; }

    /// returns the geometric shape of this side
    virtual Core::FE::CellType shape() const = 0;

    /// element dimension
    virtual unsigned n_dim() const = 0;

    /// problem dimension
    virtual unsigned n_prob_dim() const = 0;

    /// number of nodes
    virtual unsigned num_nodes() const = 0;

    /// \brief Returns the topology data for the side from Shards library
    virtual const CellTopologyData* topology() const = 0;

    /// \brief Set the side ID to the input value
    void set_id(int sid) { sid_ = sid; }

    /// \brief Set the side ID to the input value
    void set_marked_side_properties(int markedid, enum MarkedActions markedaction)
    {
      // No combination of cut-sides and marked sides!!
      if (sid_ > -1) FOUR_C_THROW("Currently a marked side and a cut side can not co-exist");
      // Number of markings on the surface should not be more than 1
      if (markedsidemap_.size() > 0)
        FOUR_C_THROW("Currently more than one mark on a side is NOT possible.");

      markedsidemap_.insert(
          std::pair<enum MarkedActions, int>(Cut::mark_and_create_boundarycells, markedid));
    }

    std::map<enum MarkedActions, int>& get_markedsidemap() { return markedsidemap_; }

    /// \brief Returns true if this is a cut side
    bool is_cut_side() { return (sid_ > -1); }

    /// \brief Returns true if this is a cut side
    bool is_boundary_cell_side() { return (is_cut_side() or is_marked_background_side()); }

    /// \brief Returns true if this is a marked side
    bool is_marked_background_side() { return (!is_cut_side() and (markedsidemap_.size() > 0)); }

    /// \brief Register this element in which the side is a part of
    void register_entity(Element* element) { elements_.insert(element); }

    /*void CollectElements( plain_element_set & elements )
     {
     std::copy( elements_.begin(), elements_.end(), std::inserter( elements, elements.begin() ) );
     }*/

    /*! \brief is this side closer to the start-point as the other side?
     *
     *  Set is_closer and return TRUE if check was successful. */
    template <class T>
    bool is_closer_side(const T& startpoint_xyz, Cut::Side* other, bool& is_closer)
    {
      if (startpoint_xyz.m() != n_prob_dim())
        FOUR_C_THROW("The dimension of startpoint_xyz is wrong! (probdim = {})", n_prob_dim());
      return is_closer_side(startpoint_xyz.data(), other, is_closer);
    }

    /*! \brief Calculates the points at which the side is cut by this edge */
    virtual bool cut(Mesh& mesh, Edge& edge, PointSet& cut_points) = 0;

    virtual bool just_parallel_cut(Mesh& mesh, Edge& edge, PointSet& cut_points, int idx = -1)
    {
      FOUR_C_THROW("JustParallelCut is not implemented for the levelset!");
      return false;
    }

    /*! \brief get all edges adjacent to given local coordinates */
    template <class T1>
    void edge_at(const T1& rs, std::vector<Edge*>& edges)
    {
      if (static_cast<unsigned>(rs.m()) != n_dim())
        FOUR_C_THROW("The dimension of rs is wrong! (dim = {})", n_dim());

      EdgeAt(rs.data(), edges);
    }

    /*! \brief get the global coordinates on side at given local coordinates */
    template <class T1, class T2>
    void point_at(const T1& rs, T2& xyz)
    {
      if (static_cast<unsigned>(xyz.m()) < n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());
      if (static_cast<unsigned>(rs.m()) != n_dim())
        FOUR_C_THROW("The dimension of rs is wrong! (dim = {})", n_dim());

      PointAt(rs.data(), xyz.data());

      // fill remaining entries with zeros
      std::fill(xyz.data() + n_prob_dim(), xyz.data() + xyz.m(), 0.0);
    }

    /*! \brief get global coordinates of the center of the side */
    template <class T1>
    void side_center(T1& midpoint)
    {
      if (midpoint.m() != n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());
      SideCenter(midpoint.data());
    }

    /*! \brief Calculates the local coordinates (\c rsd) with respect to the element shape
     *  from its global coordinates (xyz), return \c TRUE if successful. The last coordinate
     *  of \c rsd is the distance of the n-dimensional point \c xyz to the embedded
     *  side. */
    template <class T1, class T2>
    bool local_coordinates(
        const T1& xyz, T2& rsd, bool allow_dist = false, double tol = POSITIONTOL)
    {
      if (static_cast<unsigned>(xyz.m()) < n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());
      if (static_cast<unsigned>(rsd.m()) < n_prob_dim())
        FOUR_C_THROW("The dimension of rsd is wrong! (dim = {})", n_dim());

      const bool check = local_coordinates(xyz.data(), rsd.data(), allow_dist, tol);

      std::fill(rsd.data() + n_prob_dim(), rsd.data() + rsd.m(), 0.0);

      return check;
    }

    /*! \brief get local coordinates (rst) with respect to the element shape
     * for all the corner points */
    virtual void local_corner_coordinates(double* rst_corners) = 0;

    /*! \brief lies point with given coordinates within this side? */
    template <class T1, class T2>
    bool within_side(const T1& xyz, T2& rs, double& dist)
    {
      if (static_cast<unsigned>(xyz.m()) != n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());
      if (static_cast<unsigned>(rs.m()) != n_dim())
        FOUR_C_THROW("The dimension of rs is wrong! (dim = {})", n_dim());

      return WithinSide(xyz.data(), rs.data(), dist);
    }

    /* \brief compute the cut of a ray through two points with the 2D space defined by
     * the side */
    template <class T1, class T2>
    bool ray_cut(const T1& p1_xyz, const T1& p2_xyz, T2& rs, double& line_xi)
    {
      if (p1_xyz.m() != n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());
      if (rs.m() != n_dim()) FOUR_C_THROW("The dimension of rs is wrong! (dim = {})", n_dim());
      return ray_cut(p1_xyz.data(), p2_xyz.data(), rs.data(), line_xi);
    }

    /*! \brief Calculates the normal vector with respect to the element shape
     *  at local coordinates \c rs */
    template <class T1, class T2>
    void normal(const T1& rs, T2& n, bool unitnormal = true)
    {
      if (static_cast<unsigned>(n.m()) != n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());
      if (static_cast<unsigned>(rs.m()) != n_dim())
        FOUR_C_THROW("The dimension of rs is wrong! (dim = {})", n_dim());

      normal(rs.data(), n.data(), unitnormal);
    }

    /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
     * the normal vector with respect to the element shape at local coordinates rs,
     * basis vectors have norm=1. */
    template <class T1, class T2>
    void basis(const T1& rs, T2& t1, T2& t2, T2& n)
    {
      if (static_cast<unsigned>(t1.m()) != n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());
      if (static_cast<unsigned>(rs.m()) != n_dim())
        FOUR_C_THROW("The dimension of rs is wrong! (dim = {})", n_dim());

      Basis(rs.data(), t1.data(), t2.data(), n.data());
    }

    /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
     * the normal vector with respect to the element shape at local coordinates rs,
     * basis vectors have norm=1 */
    template <class T>
    void basis_at_center(T& t1, T& t2, T& n)
    {
      if (static_cast<unsigned>(t1.m()) != n_prob_dim())
        FOUR_C_THROW("The dimension of xyz is wrong! (probdim = {})", n_prob_dim());

      basis_at_center(t1.data(), t2.data(), n.data());
    }

    /*! \brief Returns the global coordinates of the nodes of this side */
    virtual void coordinates(double* xyze) const = 0;

   private:
    /*! \brief Fixes the matrix shape and returns the global coordinates of the
     *  nodes of this side
     *
     *  Be aware that this routine is more expensive than the calling function
     *  because we have to copy the data in the end. So if it's possible to give
     *  the matrix with the correct shape, always do it!
     *
     *  */
    template <class T>
    void fix_shape_and_get_coordinates(T& xyze) const
    {
      Core::LinAlg::SerialDenseMatrix xyze_corrected(n_prob_dim(), num_nodes());

      coordinates(xyze_corrected.values());

      // copy the result back into the given matrix
      for (unsigned c = 0; c < num_nodes(); ++c)
      {
        Core::LinAlg::SerialDenseMatrix xyz(
            Teuchos::View, &xyze_corrected(0, c), n_prob_dim(), n_prob_dim(), 1);
        std::copy(xyz.values(), xyz.values() + n_prob_dim(), &xyze(0, c));
        // fill the rows out of range with zeros
        std::fill(&xyze(0, c) + n_prob_dim(), &xyze(0, c) + xyze.num_rows(), 0.0);
      }

      // fill columns out of range with zeros
      for (unsigned c = num_nodes(); c < static_cast<unsigned>(xyze.num_cols()); ++c)
        std::fill(&xyze(0, c), &xyze(0, c) + xyze.num_rows(), 0.0);
    }

   public:
    template <class T>
    void coordinates(T& xyze) const
    {
      if (static_cast<unsigned>(xyze.num_rows()) < n_prob_dim())
        FOUR_C_THROW("The row dimension of xyze is wrong! (probdim = {})", n_prob_dim());
      if (static_cast<unsigned>(xyze.num_cols()) < num_nodes())
        FOUR_C_THROW("The col dimension of xyze is wrong! (numNodesSide = {})", num_nodes());

      // if the matrix
      if (static_cast<unsigned>(xyze.num_rows()) > n_prob_dim() or
          static_cast<unsigned>(xyze.num_cols()) > num_nodes())
      {
        fix_shape_and_get_coordinates(xyze);
      }
      else
      {
        coordinates(xyze.values());
      }
    }

    /*! \brief Returns true if this is a cut side */
    virtual bool is_level_set_side() { return false; };

    /** create facets on the background sides of the element
     *
     *  For all these facets, parent side is an element side */
    virtual void make_owned_side_facets(Mesh& mesh, Element* element, plain_facet_set& facets);

    /** \brief create facets on the cut sides of the element
     *
     *  For all these facets, parent side is a cut side.
     *
     *  \note See also the derived LevelSet version for information.  */
    virtual void make_internal_facets(Mesh& mesh, Element* element, plain_facet_set& facets);

    void make_internal_facets(
        Mesh& mesh, Element* element, const Cycle& points, plain_facet_set& facets);

    virtual bool is_cut();

    //   virtual bool DoTriangulation() { return true; }

    // bool FullSideCut() { return cut_lines_.size()==edges_.size() and facets_.size()==1; }

    bool on_side(const PointSet& points);

    bool on_edge(Point* point);

    bool on_edge(Line* line);

    /// All points of this side are in the other side (based on the Id())
    bool all_points_common(Side& side);

    bool have_common_node(Side& side);

    // Finds if this sides is touched by the other side at the point "p"
    bool is_touched(Side& other, Point* p);

    // Finds if this side is topologically touched by the other side in the point "p"
    // Purely based on the location of the point might not even be in the node of other side
    bool is_touched_at(Side* other, Point* p);

    bool have_common_edge(Side& side);

    Element* common_element(Side* other);

    //   virtual void ExchangeFacetSide( Side * side, bool cutsurface ) = 0;

    void add_point(Point* cut_point);

    void add_line(Line* cut_line);

    Facet* find_facet(const std::vector<Point*>& facet_points);

    /*! \brief Find Cut Lines for two Cut Sides, which have more than two common cut points!
     *  (This happens if the cutsides are in the same plane !) */
    bool find_touching_cut_lines(Mesh& mesh, Element* element, Side& side, const PointSet& cut);

    /*! \brief Find Cut Lines for two Cut Sides specially based on a discretization,
     *  which have more than two common cut points!
     *
     *  (This happens if the cutsides are in the same plane or due to numerical tolerances! */
    virtual bool find_ambiguous_cut_lines(
        Mesh& mesh, Element* element, Side& side, const PointSet& cut);

    // Find part of this side (cut points on the cut_edges), that lies on the other side
    // (parallel)
    virtual bool find_parallel_intersection(
        Mesh& mesh, Element* element, Side& side, const PointSet& cut, point_line_set& new_lines);


    // create parallel cut surface between two sides
    virtual bool create_parallel_cut_surface(Mesh& mesh, Element* element, Side& other,
        const PointSet& cut, std::vector<Point*>* cut_point_for_lines_out = nullptr);

    void get_boundary_cells(plain_boundarycell_set& bcells);

    void print();

    template <class T>
    Node* on_node(const T& x)
    {
      if (x.m() != n_prob_dim())
        FOUR_C_THROW("x has the wrong dimension! (probDim = {})", n_prob_dim());

      T nx;
      for (std::vector<Node*>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
      {
        Node* n = *i;
        n->coordinates(nx.data());
        nx.update(-1, x, 1);
        if (nx.norm2() <= (x.norm_inf() * POSITIONTOL + n->point()->tolerance()))
        {
          return n;
        }
      }
      return nullptr;
    }

    const std::vector<Edge*>& edges() const { return edges_; }

    /*!
     brief Returns the edge of this side with given begin and end points
     */
    Edge* find_edge(Point* begin, Point* end);

    const std::vector<Node*>& nodes() const { return nodes_; }

    virtual bool find_cut_points_dispatch(Mesh& mesh, Element* element, Side& side, Edge& e);

    /*!
     \brief Calculate the points at which the other side intersects with this considered side
     */
    virtual bool find_cut_points(Mesh& mesh, Element* element, Side& other);

    /*!
     \brief Draw cut lines between the cut points of this edge and "other"
     */
    bool find_cut_lines(Mesh& mesh, Element* element, Side& other);

    /*!
     \brief Get (not calculate) all the cut points between this side and "other"
     */
    void get_cut_points(Element* element, Side& other, PointSet& cuts);

    /*!
     \brief Get all the cut points that are produced by this edge
     */
    const PointSet& cut_points() const { return cut_points_; }

    const std::vector<Line*>& cut_lines() const { return cut_lines_; }

    const plain_element_set& elements() const { return elements_; }

    const std::vector<Facet*>& facets() const { return facets_; }

    /// returns true if the hole is inside the facet
    bool hole_of_facet(Facet& facet, const std::vector<Cycle>& hole);

    /// Gets a cutting side of this cutside
    void get_cutting_side(Side* cuttingside) { cutting_sides_.insert(cuttingside); }

    /// Gets selfcutpoints of this cutside
    void get_self_cut_points(PointSet& selfcutpoints)
    {
      for (PointSet::iterator i = selfcutpoints.begin(); i != selfcutpoints.end(); ++i)
      {
        Point* selfcutpoint = *i;
        self_cut_points_.insert(selfcutpoint);
      }
    }

    /// Gets a selfcutnode of this cutside
    void get_self_cut_node(Node* selfcutnode) { self_cut_nodes_.insert(selfcutnode); }

    /// Gets a selfcutedge of this cutside
    void get_self_cut_edge(Edge* selfcutedge) { self_cut_edges_.insert(selfcutedge); }

    /// Gets a selfcuttriangle of this cutside
    void get_self_cut_triangle(std::vector<Cut::Point*> selfcuttriangle)
    {
      self_cut_triangles_.push_back(selfcuttriangle);
    }

    /// Gets the selfcutposition of this cutside and spreads the positional information
    void get_self_cut_position(Point::PointPosition p);

    /// Changes the selfcutposition of this cutside and spreads the positional information
    void change_self_cut_position(Point::PointPosition p);

    /// Erase a cuttingside from this cutside because the bounding box found too much
    void erase_cutting_side(Side* nocuttingside) { cutting_sides_.erase(nocuttingside); }

    /// Returns all cutting sides of this cutside
    const plain_side_set& cutting_sides() const { return cutting_sides_; }

    /// Returns all selfcutpoints of this cutside
    const PointSet& self_cut_points() const { return self_cut_points_; }

    /// Returns all selfcutnodes of this cutside
    const plain_node_set& self_cut_nodes() const { return self_cut_nodes_; }

    /// Returns all selfcutedges of this cutside
    const plain_edge_set& self_cut_edges() const { return self_cut_edges_; }

    /// Returns all selfcuttriangles of this cutside
    const std::vector<std::vector<Cut::Point*>>& self_cut_triangles() const
    {
      return self_cut_triangles_;
    }

    /// Returns the selfcutposition of this cutside
    Point::PointPosition self_cut_position() { return selfcutposition_; }

    /// replace "nod" associated with this side by the new node "replwith"
    void replace_nodes(Node* nod, Node* replwith);

    /// get bounding volume
    const BoundingBox& get_bounding_volume() const { return *boundingvolume_; }

    /// Remove point p from this side
    void remove_point(Point* p) { cut_points_.erase(p); };

    /// Parallel Cut Surfaces on this side (represented by the points)
    std::set<std::set<Point*>>& get_parallel_cut_surfaces() { return parallel_cut_surfaces_; }


   protected:
    bool all_on_nodes(const PointSet& points);

    /** @name All these functions have to be implemented in the derived classes!
     *
     *  \remark Please note, that none of these functions has any inherent
     *          dimension checks! Be careful if you access them directly. Each of
     *          these functions has a public alternative which checks the dimensions
     *          and is therefore much safer.                      hiermeier 07/16 */
    /// @{

    /*! \brief is this side closer to the starting-point as the other side?
     *
     *  Set is_closer and return TRUE if check was successful. */
    virtual bool is_closer_side(
        const double* startpoint_xyz, Cut::Side* other, bool& is_closer) = 0;

    /*! \brief get all edges adjacent to given local coordinates */
    virtual void edge_at(const double* rs, std::vector<Edge*>& edges) = 0;

    /*! \brief get the global coordinates on side at given local coordinates */
    virtual void point_at(const double* rs, double* xyz) = 0;

    /*! \brief get global coordinates of the center of the side */
    virtual void side_center(double* midpoint) = 0;

    /*! \brief Calculates the local coordinates (rsd) with respect to the element shape
     *  from its global coordinates (xyz), return TRUE if successful. The last coordinate
     *  of \c rsd is the distance of the n-dimensional point \c xyz to the embedded
     *  side. */
    virtual bool local_coordinates(const double* xyz, double* rsd, bool allow_dist, double tol) = 0;

    /*! \brief Does the point with given coordinates lie within this side? */
    virtual bool within_side(const double* xyz, double* rs, double& dist) = 0;

    /* \brief compute the cut of a ray through two points with the 2D space defined by the side */
    virtual bool ray_cut(
        const double* p1_xyz, const double* p2_xyz, double* rs, double& line_xi) = 0;

    /*! \brief Calculates the normal vector with respect to the element shape at local coordinates
     * rs */
    virtual void normal(const double* rs, double* normal, bool unitnormal) = 0;

    /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
     * the normal vector with respect to the element shape at local coordinates rs,
     * basis vectors have norm=1. */
    virtual void basis(const double* rs, double* t1, double* t2, double* n) = 0;

    /* \brief Calculates a Basis of two tangential vectors (non-orthogonal!) and
     * the normal vector with respect to the element shape at local coordinates rs,
     * basis vectors have norm=1 */
    virtual void basis_at_center(double* t1, double* t2, double* n) = 0;

    /// @}

   private:
    /** \brief Get the default uncut facet number per side
     *
     *  The number is dependent on the dimension of the parent element.
     *
     *  */
    unsigned uncut_facet_number_per_side() const;

    /// Simplifies topological connection in the case, when side contains two parallel cut
    /// surfaces, which are self-intersecting.
    void simplify_mixed_parallel_cut_surface(Mesh& mesh, Element* element, Side& other,
        std::set<Point*>& new_surface, std::vector<Point*>& cut_points_for_lines);

    /// Does this Side have another parallel cut surface than this?
    bool has_mixed_parallel_cut_surface(const std::set<Point*>& surface)
    {
      return (parallel_cut_surfaces_.size() > 0 &&
              parallel_cut_surfaces_.find(surface) == parallel_cut_surfaces_.end());
    }

   private:
    int sid_;

    // Marked side additions:
    std::map<Cut::MarkedActions, int> markedsidemap_;
    // -----------------------

    std::vector<Node*> nodes_;

    std::vector<Edge*> edges_;

    plain_element_set elements_;

    std::vector<Line*> cut_lines_;

    PointSet cut_points_;

    std::vector<Facet*> facets_;

    /// in some extreme cases surface can be partially
    /// parallel to the other side
    std::set<std::set<Point*>> parallel_cut_surfaces_;

    /// all sides which are cutting this cutside
    plain_side_set cutting_sides_;

    /// all selfcutpoints of this cutside
    PointSet self_cut_points_;

    /// all selfcutnodes of this cutside
    plain_node_set self_cut_nodes_;

    /// all selfcutedges of this cutside
    plain_edge_set self_cut_edges_;

    /// all selfcuttriangles of this cutside
    std::vector<std::vector<Cut::Point*>> self_cut_triangles_;

    /// the selfcutposition of this cutside shows if it is inside or outside the other structure
    /// body
    Point::PointPosition selfcutposition_;

    /// the bounding volume of the side
    std::shared_ptr<BoundingBox> boundingvolume_;

  };  // class side

  /*! \brief Implementation of the concrete side element
   *
   *  The class is a template on the problem dimension \c probdim,
   *                             the side type \c sidetype
   *                             the number of nodes per side \c numNodesSide
   *                             the side dimension \c dim
   *
   *  */
  template <unsigned probdim, Core::FE::CellType sidetype,
      unsigned num_nodes_side = Core::FE::num_nodes<sidetype>,
      unsigned dim = Core::FE::dim<sidetype>>
  class ConcreteSide : public Side, public ConcreteElement<probdim, sidetype>
  {
   public:
    /// constructor
    ConcreteSide(int sid, const std::vector<Node*>& nodes, const std::vector<Edge*>& edges)
        : Side(sid, nodes, edges),
          ConcreteElement<probdim, sidetype>(-1, std::vector<Side*>(0), nodes, false)
    {
      // sanity check
      if (dim > 2)
        FOUR_C_THROW(
            "The element dimension of sides is not allowed to be greater"
            " than 2!");
    }

    /// Returns the geometrical shape of this side
    Core::FE::CellType shape() const override { return sidetype; }

    /// element dimension
    unsigned n_dim() const override { return dim; }

    /// problem dimension
    unsigned n_prob_dim() const override { return probdim; }

    unsigned num_nodes() const override { return num_nodes_side; }

    /// Returns the topology data for the side from Shards library
    const CellTopologyData* topology() const override
    {
      switch (sidetype)
      {
        case Core::FE::CellType::tri3:
          return shards::getCellTopologyData<shards::Triangle<3>>();
          break;
        case Core::FE::CellType::quad4:
          return shards::getCellTopologyData<shards::Quadrilateral<4>>();
          break;
        case Core::FE::CellType::line2:
          return shards::getCellTopologyData<shards::Line<2>>();
          break;
        default:
          FOUR_C_THROW("Unknown sidetype! ({})\n", Core::FE::cell_type_to_string(sidetype));
          break;
      }
      exit(EXIT_FAILURE);
    }

    /// Calculates the points at which the side is cut by this edge
    bool cut(Mesh& mesh, Edge& edge, PointSet& cut_points) override
    {
      return edge.cut(mesh, *this, cut_points);
    }

    bool just_parallel_cut(Mesh& mesh, Edge& edge, PointSet& cut_points, int idx = -1) override
    {
      return edge.just_parallel_cut(mesh, *this, cut_points, idx);
    }

    /** \brief Is this side closer to the start-point as the other side?
     *
     *  check based on ray-tracing technique set is_closer and return
     *  \TRUE if check was successful */
    bool is_closer_side(
        const Core::LinAlg::Matrix<probdim, 1>& startpoint_xyz, Cut::Side* other, bool& is_closer);

    /// get all edges adjacent to given local coordinates
    void edge_at(const Core::LinAlg::Matrix<dim, 1>& rs, std::vector<Edge*>& edges)
    {
      switch (sidetype)
      {
        case Core::FE::CellType::tri3:
        {
          if (fabs(rs(1)) < REFERENCETOL) edges.push_back(Side::edges()[0]);
          if (fabs(rs(0) + rs(1) - 1) < REFERENCETOL) edges.push_back(Side::edges()[1]);
          if (fabs(rs(0)) < REFERENCETOL) edges.push_back(Side::edges()[2]);
          break;
        }
        case Core::FE::CellType::quad4:
        {
          if (fabs(rs(1) + 1) < REFERENCETOL) edges.push_back(Side::edges()[0]);
          if (fabs(rs(0) - 1) < REFERENCETOL) edges.push_back(Side::edges()[1]);
          if (fabs(rs(1) - 1) < REFERENCETOL) edges.push_back(Side::edges()[2]);
          if (fabs(rs(0) + 1) < REFERENCETOL) edges.push_back(Side::edges()[3]);
          break;
        }
        case Core::FE::CellType::line2:
        {
          FOUR_C_THROW("If we need this, the edges will degenerate to nodes!");
          break;
        }
        default:
        {
          throw "Unknown/unsupported side type! \n";
          break;
        }
      }  // switch (sidetype)
    }

    /** \brief get the global coordinates on side at given local coordinates
     *
     *  \param rs  (in)  : parameter space coordinates
     *  \param xyz (out) : corresponding spatial coordinates */
    void point_at(const Core::LinAlg::Matrix<dim, 1>& rs, Core::LinAlg::Matrix<probdim, 1>& xyz)
    {
      ConcreteElement<probdim, sidetype>::point_at(rs, xyz);
    }

    /** \brief get global coordinates of the center of the side
     *
     *  \param midpoint (out) : mid-point spatial coordinates */
    void side_center(Core::LinAlg::Matrix<probdim, 1>& midpoint)
    {
      ConcreteElement<probdim, sidetype>::element_center(midpoint);
    }

    ///  lies point with given coordinates within this side?
    bool within_side(const Core::LinAlg::Matrix<probdim, 1>& xyz, Core::LinAlg::Matrix<dim, 1>& rs,
        double& dist);

    /// compute the cut of a ray through two points with the 2D space defined by the side
    bool ray_cut(const Core::LinAlg::Matrix<probdim, 1>& p1_xyz,
        const Core::LinAlg::Matrix<probdim, 1>& p2_xyz, Core::LinAlg::Matrix<dim, 1>& rs,
        double& line_xi);

    /** \brief Calculates the local coordinates (rst) with respect to the element shape from its
     *  global coordinates (xyz), return TRUE if successful
     *
     *  \remark The last coordinate of the variable rsd holds the distance to the side. */
    bool local_coordinates(const Core::LinAlg::Matrix<probdim, 1>& xyz,
        Core::LinAlg::Matrix<probdim, 1>& rsd, bool allow_dist = false, double tol = POSITIONTOL);

    /// get local coordinates (rst) with respect to the element shape for all the corner points
    void local_corner_coordinates(double* rst_corners) override
    {
      switch (sidetype)
      {
        case Core::FE::CellType::tri3:
        {
          const double rs[6] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
          std::copy(rs, rs + 6, rst_corners);
          break;
        }
        case Core::FE::CellType::quad4:
        {
          const double rs[8] = {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0};
          std::copy(rs, rs + 8, rst_corners);
          break;
        }
        case Core::FE::CellType::line2:
        {
          const double r[2] = {-1.0, 1.0};
          std::copy(r, r + 2, rst_corners);
          break;
        }
        default:
        {
          FOUR_C_THROW("Unknown/unsupported side type!");
          exit(EXIT_FAILURE);
        }
      }
    }

    /// Calculates the normal vector with respect to the element shape at local coordinates xsi
    void normal(const Core::LinAlg::Matrix<dim, 1>& xsi, Core::LinAlg::Matrix<probdim, 1>& normal,
        bool unitnormal = true)
    {
      // get derivatives at pos
      Core::LinAlg::Matrix<probdim, num_nodes_side> side_xyze(Core::LinAlg::Initialization::zero);
      this->coordinates(side_xyze);

      Core::LinAlg::Matrix<dim, num_nodes_side> deriv(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<dim, probdim> A(Core::LinAlg::Initialization::zero);

      Core::FE::shape_function_deriv1<sidetype>(xsi, deriv);
      A.multiply_nt(deriv, side_xyze);

      switch (dim)
      {
        // 1-dimensional side-element embedded in 2-dimensional space
        case 1:
        {
          if (probdim == 3)
            FOUR_C_THROW(
                "Dimension mismatch. A 1-dimensional element in a 3-dimensional space is "
                "not a side, but a edge!");

          normal(0) = A(0, 1);   //   dy/dxi
          normal(1) = -A(0, 0);  // - dx/dxi

          break;
        }
        // 2-dimensional side element embedded in 3-dimensional space
        case 2:
        {
          FOUR_C_ASSERT_ALWAYS(probdim == 3, "Internal error: dimension mismatch!");
          // cross product to get the normal at the point
          normal(0) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
          normal(1) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
          normal(2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

          break;
        }
        default:
          FOUR_C_THROW("Unsupported element dimension!");
          break;
      }
      // force unit length
      if (unitnormal)
      {
        double norm = normal.norm2();
        normal.scale(1. / norm);
      }
    }

    /** Calculates a Basis of two tangential vectors (non-orthogonal!) and
     *  the normal vector with respect to the element shape at center of the side.
     *  All basis vectors are of unit length */
    void basis_at_center(Core::LinAlg::Matrix<probdim, 1>& t1, Core::LinAlg::Matrix<probdim, 1>& t2,
        Core::LinAlg::Matrix<probdim, 1>& n)
    {
      Core::LinAlg::Matrix<dim, 1> center_rs(Core::FE::get_local_center_position<dim>(sidetype));
      basis(center_rs, t1, t2, n);
    }

    /** Calculates a Basis of two tangential vectors (non-orthogonal!) and
     * the normal vector with respect to the element shape at local coordinates xsi.
     * All basis vectors are of unit length. */
    void basis(const Core::LinAlg::Matrix<dim, 1>& xsi, Core::LinAlg::Matrix<probdim, 1>& t1,
        Core::LinAlg::Matrix<probdim, 1>& t2, Core::LinAlg::Matrix<probdim, 1>& n)
    {
      // get derivatives at pos
      Core::LinAlg::Matrix<probdim, num_nodes_side> side_xyze(Core::LinAlg::Initialization::zero);
      this->coordinates(side_xyze);

      Core::LinAlg::Matrix<dim, num_nodes_side> deriv(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<dim, probdim> A(Core::LinAlg::Initialization::zero);

      Core::FE::shape_function_deriv1<sidetype>(xsi, deriv);
      A.multiply_nt(deriv, side_xyze);

      // set the first tangential vector
      t1(0) = A(0, 0);
      t1(1) = A(0, 1);
      t1(2) = A(0, 2);

      t1.scale(1. / t1.norm2());


      switch (dim)
      {
        case 1:
        {
          // there is no 2nd tangential vector in the 1-D case
          t2 = 0;
          break;
        }
        case 2:
        {
          FOUR_C_ASSERT_ALWAYS(probdim == 3, "Internal error: dimension mismatch!");
          // set the second tangential vector
          t2(0) = A(1, 0);
          t2(1) = A(1, 1);
          t2(2) = A(1, 2);

          t2.scale(1. / t2.norm2());
          break;
        }
      }
      // calculate the normal
      normal(xsi, n);
    }

    /// get coordinates of side
    void coordinates(Core::LinAlg::Matrix<probdim, num_nodes_side>& xyze_surfaceElement) const
    {
      coordinates(xyze_surfaceElement.data());
    }

    /*! \brief Returns the global coordinates of the nodes of this side */
    void coordinates(double* xyze) const override
    {
      ConcreteElement<probdim, sidetype>::coordinates(xyze);
    }

   protected:
    /// derived
    bool is_closer_side(const double* startpoint_xyz, Cut::Side* other, bool& is_closer) override
    {
      const Core::LinAlg::Matrix<probdim, 1> startpoint_xyz_mat(startpoint_xyz, true);
      return is_closer_side(startpoint_xyz_mat, other, is_closer);
    }

    /// derived
    void edge_at(const double* rs, std::vector<Edge*>& edges) override
    {
      const Core::LinAlg::Matrix<dim, 1> rs_mat(rs, true);  // create view
      edge_at(rs_mat, edges);
    }

    /// derived
    void point_at(const double* rs, double* xyz) override
    {
      const Core::LinAlg::Matrix<dim, 1> rs_mat(rs, true);  // create view
      Core::LinAlg::Matrix<probdim, 1> xyz_mat(xyz, true);  // create view
      point_at(rs_mat, xyz_mat);
    }

    /// derived
    void side_center(double* midpoint) override
    {
      Core::LinAlg::Matrix<probdim, 1> midpoint_mat(midpoint, true);  // create view
      side_center(midpoint_mat);
    }

    /// derived
    bool local_coordinates(
        const double* xyz, double* rsd, bool allow_dist = false, double tol = POSITIONTOL) override
    {
      const Core::LinAlg::Matrix<probdim, 1> xyz_mat(xyz, true);  // create view
      Core::LinAlg::Matrix<probdim, 1> rsd_mat(rsd, true);        // create view
      return local_coordinates(xyz_mat, rsd_mat, allow_dist, tol);
    }

    /// derived
    bool within_side(const double* xyz, double* rs, double& dist) override
    {
      const Core::LinAlg::Matrix<probdim, 1> xyz_mat(xyz, true);  // create view
      Core::LinAlg::Matrix<dim, 1> rs_mat(rs, true);              // create view
      return within_side(xyz_mat, rs_mat, dist);
    }

    /// derived
    bool ray_cut(const double* p1_xyz, const double* p2_xyz, double* rs, double& line_xi) override
    {
      const Core::LinAlg::Matrix<probdim, 1> p1_xyz_mat(p1_xyz, true);  // create view
      const Core::LinAlg::Matrix<probdim, 1> p2_xyz_mat(p2_xyz, true);  // create view
      Core::LinAlg::Matrix<dim, 1> rs_mat(rs, true);                    // create view
      return ray_cut(p1_xyz_mat, p2_xyz_mat, rs_mat, line_xi);
    }

    /// derived
    void normal(const double* rs, double* normal_v, bool unitnormal = true) override
    {
      const Core::LinAlg::Matrix<dim, 1> rs_mat(rs, true);          // create view
      Core::LinAlg::Matrix<probdim, 1> normal_mat(normal_v, true);  // create view
      normal(rs_mat, normal_mat, unitnormal);
    }

    /// derived
    void basis(const double* rs, double* t1, double* t2, double* n) override
    {
      const Core::LinAlg::Matrix<dim, 1> rs_mat(rs, true);  // create view
      Core::LinAlg::Matrix<probdim, 1> t1_mat(t1, true);    // create view
      Core::LinAlg::Matrix<probdim, 1> t2_mat(t2, true);    // create view
      Core::LinAlg::Matrix<probdim, 1> n_mat(n, true);      // create view
      basis(rs_mat, t1_mat, t2_mat, n_mat);
    }

    /// derived
    void basis_at_center(double* t1, double* t2, double* n) override
    {
      Core::LinAlg::Matrix<probdim, 1> t1_mat(t1, true);  // create view
      Core::LinAlg::Matrix<probdim, 1> t2_mat(t2, true);  // create view
      Core::LinAlg::Matrix<probdim, 1> n_mat(n, true);    // create view
      basis_at_center(t1_mat, t2_mat, n_mat);
    }
  };  // class ConcreteSide

  /*--------------------------------------------------------------------------*/
  class SideFactory
  {
   public:
    SideFactory() {};

    virtual ~SideFactory() = default;

    Side* create_side(Core::FE::CellType sidetype, int sid, const std::vector<Node*>& nodes,
        const std::vector<Edge*>& edges) const;

   private:
    template <Core::FE::CellType sidetype>
    Cut::Side* create_concrete_side(int sid, const std::vector<Node*>& nodes,
        const std::vector<Edge*>& edges, int probdim) const
    {
      Side* s = nullptr;
      // sanity check
      if (probdim < Core::FE::dim<sidetype>)
        FOUR_C_THROW("Problem dimension is smaller than the side dimension!");

      switch (probdim)
      {
        case 2:
          s = new ConcreteSide<2, sidetype>(sid, nodes, edges);
          break;
        case 3:
          s = new ConcreteSide<3, sidetype>(sid, nodes, edges);
          break;
        default:
          FOUR_C_THROW("Unsupported problem dimension! (probdim = {})", probdim);
          break;
      }
      return s;
    }
  };  // class SideFactory

}  // namespace Cut


std::ostream& operator<<(std::ostream& stream, Cut::Side& s);

FOUR_C_NAMESPACE_CLOSE

#endif
