// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRIC_SEARCH_MATCHINGOCTREE_HPP
#define FOUR_C_GEOMETRIC_SEARCH_MATCHINGOCTREE_HPP

#include "4C_config.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>
#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
  class ParObject;
}  // namespace Core::Communication

namespace Core::GeometricSearch
{
  class OctreeElement;

  //! Base class for parallel octree to establish neighborhood relations between entities
  class MatchingOctree
  {
   public:
    /// Data structure to hold entity id and position regardless of entity type
    struct Entity
    {
      int gid;
      std::array<double, 3> position;
    };

    //! Constructor
    MatchingOctree();

    //! Destructor
    virtual ~MatchingOctree() = default;

    /*! \brief initialize this class (public)


    Initialize processor local octree                               rauch 09/16

    On input a list of so called "target"-entities and a discretisation which
    contains these "target"-entities. Make sure, that you provide the octree
    only with those entity gids, you want in the processor local octree.
    You may want to match only entities, which are owned, ghosted, or both.

    The number maxeleperleaf specifies the upper bound for the number of
    eleids attached to a octree leaf. If the number would be higher, the
    octree element would be split into two children (which could be leaves
    on their own)

    If an entity id is associated with a child of an octree element, it is
    checked whether its coordinate is inside the bounding box of the
    octree element. This is done only with a tolerance tol to ensure that
    each node is guaranteed to be owned by one of the children and could
    be found later on. The same tolerance is used for finding the smallest
    distance between two nodes. Thus, it should be of the order or magnitude
    of the smallest element size.


    \param   actdis         (i) discretisation
    \param   target_node_ids   (i) list of target entity ids
    \param   maxnodeperleaf  (i) parameter for octree
    \param   tol            (i) tolerance for octree

    \return void  */
    virtual int init(const Core::FE::Discretization& actdis,
        const std::vector<int>& target_node_ids, const int maxnodeperleaf = 150,
        const double tol = 1e-08);

    //! setup this class
    virtual int setup();

   protected:
    //! @name Methods to create processor local node matching

    /*! \brief Declaration


    Search closest entity to given coordinate x in local octree.

    returns false if entity is not in bounding box of local octree

    gammi 05/07


    \param  x                  (i) coordinate of source_node
    \param  idofclosestpoint   (o) the node id of the closest point
    \param  distofclosestpoint (o) the distance to the closest point

    \return  bool        false if node is not in bounding box of
                         local octree  */
    virtual bool search_closest_entity_on_this_proc(const std::vector<double>& x,
        int& idofclosestpoint, double& distofclosestpoint, bool searchsecond = false);

    //@}


    //! @name Pure virtual methods which have to be overridden by specialized octrees

    //! calc unique coordinate of entity
    virtual void calc_point_coordinate(
        const Core::FE::Discretization* dis, const int id, double* coord) = 0;

    //! calc unique coordinate of entity
    virtual void calc_point_coordinate(Core::Communication::ParObject* entity, double* coord) = 0;

    //! check if entity is on calling proc
    virtual bool check_have_entity(const Core::FE::Discretization* dis, const int id) = 0;

    //! returns true if entity is owned by calling proc
    virtual bool check_entity_owner(const Core::FE::Discretization* dis, const int id) = 0;

    //! Write data as Entity to PackPuffer
    virtual void pack_entity(Core::Communication::PackBuffer& data,
        const Core::FE::Discretization* dis, const int id) = 0;

    //! check if unpacked type is correct
    virtual int check_valid_entity_type(std::shared_ptr<Core::Communication::ParObject> o) = 0;

    //! create an octree element
    virtual std::shared_ptr<OctreeElement> create_octree_element(std::vector<int>& nodeidstoadd,
        Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer) = 0;

    //@}

   public:
    //! @name Methods to create global node matching

    /*! \brief Search for closest (source) nodes on all procs


    Search for closest (source) entities on all processors to given (target)
    entity set (only in bounding box of target entities)        gammi 05/07

     1) each proc gets a list of its source entities (source_node_ids)

     2) the list is communicated in a round robin pattern to all the
        other procs.

     3) the proc checks the package from each proc and calcs the min
        distance on each --- the result is kept if distance is smaller
        than on the preceding processors

    Again, dofsforpbcplane is used to overwrite the coordinate of the
    source perpendicular to the plane of the periodic boundary condition
    by the coordinate of the target_plane (so to speak, the closest node
    matching is "exact")

    On output, we get a map from target_ids on this proc to source_node_ids
    (on arbitrary procs).


    \param  source_node_ids     (i) list of source_node_ids
    \param  dofsforpbcplane  (i) specification of the plane containing
                                 parallel to the boundary condition
                                 (used to "shift" nodes in the normal
                                  direction)
    \param  rotangle         (i) angle (RAD) for rotation of source plane
    \param  midtosid         (o) map from target to source_nodes

    \return void  */
    virtual void create_global_entity_matching(const std::vector<int>& source_node_ids,
        const std::vector<int>& dofsforpbcplane, const double rotangle,
        std::map<int, std::vector<int>>& midtosid);

    /*! \brief find pairs of nearest nodes

      Find the pairs of target node and source node with the smallest
      distance. The sets of nodes might come from different
      discretisations, however the communicators of both discretisations
      need to span the same processors. For communication the target
      discretisation communicator is used.

      Internally the search is based on the octtree spanned by the
      target nodes. The source nodes are gathered on their owning
      processors and send in a round robin fashion to each processor
      once.

      \param source_dis     (i) discretization the source nodes belong to
      \param source_node_ids (i) gids of nodes to match
      \param coupling     (o) target node gid to (source node gid, distance) */
    virtual void find_match(const Core::FE::Discretization& source_dis,
        const std::vector<int>& source_node_ids, std::map<int, std::pair<int, double>>& coupling);

    /*! \brief find pairs of nearest nodes

      Basically, the algorithm performed in this method is equal to the one
      described in the documentation to \ref FindMatch .

      \note On exit, the map 'coupling' maps the source entity gids (key) to
            the matching target entity gids, distance, and the information,
            whether the current proc owns the matching target entity or not.
            This is different from the method \ref FindMatch.

      \param source_dis     (i) discretization the source entity belongs to
      \param source_node_ids (i) gids of entity to match
      \param coupling     (o) maps source entity gid to (target entity gid, distance, owned) */
    virtual void fill_source_to_target_gid_mapping(const Core::FE::Discretization& source_dis,
        const std::vector<int>& source_node_ids, std::map<int, std::vector<double>>& coupling);

    //@}

   protected:
    //! \brief all nodes in leaves are nodes of this discretization
    const Core::FE::Discretization* discret_;
    //! \brief order of magnitude of smallest element size (used for tolerances)
    double tol_;
    //! \brief root of the local search tree
    std::shared_ptr<OctreeElement> octreeroot_;
    //! \brief coordinate of one point in the target plane
    std::vector<double> target_plane_coords_;
    //! \brief ids of entities to be coupled (e.g. nodes in \ref NodeMatchingOctree )
    const std::vector<int>* target_entity_ids_;
    //! maximum number of tree nodes per leaf
    int maxtreenodesperleaf_;

   private:
    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    bool is_setup() { return issetup_; };

    //! returns true if init(..) was called and is still valid
    bool is_init() { return isinit_; };

    //! check if \ref setup() was called
    void check_is_setup()
    {
      if (not is_setup()) FOUR_C_THROW("setup() was not called.");
    };

    //! check if \ref init() was called
    void check_is_init()
    {
      if (not is_init()) FOUR_C_THROW("init(...) was not called.");
    };

   private:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; };

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; };

  };  // MatchingOctree


  //! Parallel octree to establish neighborhood relations between two sets of nodes
  class NodeMatchingOctree : public MatchingOctree
  {
   public:
    //! Constructor
    NodeMatchingOctree();

   protected:
    //! calc unique coordinate, associated to node
    void calc_point_coordinate(
        const Core::FE::Discretization* dis, const int id, double* coord) override;

    //! calc unique coordinate of Node
    void calc_point_coordinate(Core::Communication::ParObject* entity, double* coord) override;

    //! check if node with gid = id is on calling proc
    bool check_have_entity(const Core::FE::Discretization* dis, const int id) override;

    //! returns true if node with gid = id is owned by calling proc
    bool check_entity_owner(const Core::FE::Discretization* dis, const int id) override;

    //! pack node to PackPuffer
    void pack_entity(Core::Communication::PackBuffer& data, const Core::FE::Discretization* dis,
        const int id) override;

    //! check if unpacked type is correct
    int check_valid_entity_type(std::shared_ptr<Core::Communication::ParObject> o) override;

    //! create an octree element
    std::shared_ptr<OctreeElement> create_octree_element(std::vector<int>& nodeidstoadd,
        Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer) override;

  };  // class NodeMatchingOctree


  //! Parallel octree to establish neighborhood relations between two sets of elements
  class ElementMatchingOctree : public MatchingOctree
  {
   public:
    //! Constructor
    ElementMatchingOctree();

   protected:
    //! calc unique coordinate, associated to element
    void calc_point_coordinate(
        const Core::FE::Discretization* dis, const int id, double* coord) override;

    //! calc unique coordinate of entity
    void calc_point_coordinate(Core::Communication::ParObject* entity, double* coord) override;

    //! check if element with gid = id is on calling proc
    bool check_have_entity(const Core::FE::Discretization* dis, const int id) override;

    //! returns true if element is owned by calling proc
    bool check_entity_owner(const Core::FE::Discretization* dis, const int id) override;

    //! pack element to PackPuffer
    void pack_entity(Core::Communication::PackBuffer& data, const Core::FE::Discretization* dis,
        const int id) override;

    //! check if unpacked type is correct
    int check_valid_entity_type(std::shared_ptr<Core::Communication::ParObject> o) override;

    //! create an octree element
    std::shared_ptr<OctreeElement> create_octree_element(std::vector<int>& nodeidstoadd,
        Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer) override;

   private:
    //! \brief Map that stores id and RCP to flying nodes
    //!
    //! The flying nodes are sent around in the round robin loop
    //! together with the corresponding element.
    //!
    //! \note We need this map as member, because in \ref unpack_entity
    //!       we Extract the nodes and fill this map. Then, we set the
    //!       nodal pointers in the communicated element by providing
    //!       the element with this map. The RCPs stored in this map have
    //!       to survive until they re used in \ref calc_point_coordinate .
    //!       Therefore, we need to have a global temporary storage for
    //!       the nodal pointers, which is provided by this member.
    //!

    std::map<int, std::shared_ptr<Core::Nodes::Node>> nodes_;

  };  // class ElementMatchingOctree


  //! Leaf in parallel octree
  class OctreeElement
  {
   public:
    //! Standard Constructor
    OctreeElement();

    //! Destructor
    virtual ~OctreeElement() = default;

    /*! \brief initialize this class


    Initialize one element in octree                              rauch 09/16

    nodeids is a list of tree-nodes out of discretisation actdis belonging to
    the tree-element or its children. bounding box specifies the geometry of
    the tree-element. layer is the number of subdivisions after the octree
    root. maxnodeperleaf specifies the max number of nodes contained by
    one leaf element. tol is a tolerance used to assign the tree-nodes to the
    tree-elements by checking if they are in the bounding box or not.


    \param  actdis          (i) the discretisation
    \param  nodeids         (i) tree-nodes attached to tree-element
    \param  boundingbox     (i) bounding box of element
    \param  layer           (i) depth in tree
    \param  maxnodeperleaf  (i) how many nodes does
                                one leaf contain?
    \param  tol             (i) Tolerance for octree

    \return  int  */
    virtual int init(const Core::FE::Discretization& actdis, std::vector<int>& nodeids,
        const Core::LinAlg::SerialDenseMatrix& boundingbox, const int layer,
        const int maxnodeperleaf, const double tol);

    //! setup this class
    virtual int setup();

    /*! \brief Find closest point in leaf


    leaf contains still maxnodeperleaf nodes. This function
    determines the closest point of all nodes attached to this leaf.

                                                     gammi 05/07


    \param   x                  (i) coordinate of point
    \param   idofclosestpoint   (o) global id of closest point
    \param   distofclosestpoint (o) distance of closest point
    \param   elesize            (i) tolerance for node matching
    \param   searchsecond       (i) flag for search of second match

    \return void  */
    virtual void search_closest_node_in_leaf(const std::vector<double>& x, int& idofclosestpoint,
        double& distofclosestpoint, const double& elesize, bool searchsecond);

    //! @name Octree element functions

    /*! \brief Question if coordinate is in bounding box


    Is x in the bounding box of the element?              gammi 05/07


    \param x         (i) coordinate of point

    \return  bool true if point in bounding box  */
    bool is_point_in_bounding_box(const std::vector<double>& x);

    /*! \brief Question if octree element is leaf


    is the octree element a leaf, i.e. does it possess a vector
    of nodes attached to it?                         gammi 05/07


    \return  bool true if element is leaf  */
    bool is_leaf();

    /*! \brief Return child containing point


    Determine which child contains the coordinate x. In some cases
    the return value may not be the unique answer (overlap!)

                                                     gammi 05/07


    \param   x   (i) coordinate

    \return  std::shared_ptr<OctreeElement> child  */
    std::shared_ptr<OctreeElement> return_child_containing_point(const std::vector<double>& x);

    /*! \brief Print some information on the octree leaf


    std::cout the node coordinates associated to a leaf
                                                     gammi 05/07


    \param  os (i)

    \return void  */
    void print(std::ostream& os) const;

    //@}

   protected:
    //! calc unique coordinate of entity
    virtual void calc_point_coordinate(
        const Core::FE::Discretization* dis, const int id, double* coord) = 0;

    //! create an octree element
    virtual std::shared_ptr<OctreeElement> create_octree_element(std::vector<int>& nodeidstoadd,
        Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer) = 0;

   protected:
    //! all nodes belong to this discretisation
    const Core::FE::Discretization* discret_;
    //! the bounding box of the element
    Core::LinAlg::SerialDenseMatrix boundingbox_;
    //! the nodeids (needs to be a copy !)
    std::vector<int> nodeids_;
    //! \brief depth of the octree element in the tree
    int layer_;
    //! maximum number of tree nodes per leaf
    int maxtreenodesperleaf_;
    //! tolerance for octree
    double tol_;
    //! pointer to first child
    std::shared_ptr<OctreeElement> octreechild1_;
    //! pointer to second child
    std::shared_ptr<OctreeElement> octreechild2_;

   private:
    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    bool is_setup() { return issetup_; };

    //! returns true if init(..) was called and is still valid
    bool is_init() { return isinit_; };

    //! check if \ref setup() was called
    void check_is_setup()
    {
      if (not is_setup()) FOUR_C_THROW("setup() was not called.");
    };

    //! check if \ref init() was called
    void check_is_init()
    {
      if (not is_init()) FOUR_C_THROW("init(...) was not called.");
    };

   private:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; };

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; };

  };  // class OctreeElement


  //! Leaf in parallel octree for FEM nodes
  class OctreeNodalElement : public OctreeElement
  {
   public:
    //! Constructor
    OctreeNodalElement();

   protected:
    //! calc unique coordinate of node
    void calc_point_coordinate(
        const Core::FE::Discretization* dis, const int id, double* coord) override;

    //! create an octree element
    std::shared_ptr<OctreeElement> create_octree_element(std::vector<int>& nodeidstoadd,
        Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer) override;

  };  // OctreeNodalElement


  //! Leaf in parallel octree for FEM elements
  class OctreeElementElement : public OctreeElement
  {
   public:
    //! Constructor
    OctreeElementElement();

   protected:
    //! calc unique coordinate of element
    void calc_point_coordinate(
        const Core::FE::Discretization* dis, const int id, double* coord) override;

    //! create an octree element
    std::shared_ptr<OctreeElement> create_octree_element(std::vector<int>& nodeidstoadd,
        Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer) override;

  };  // OctreeNodalElement

}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE

#endif
