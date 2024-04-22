/*---------------------------------------------------------------------*/
/*! \file

\brief PointPool, stores a points in the cut and decides if points are merged or new points are
created

\level 3


*----------------------------------------------------------------------*/

#ifndef FOUR_C_CUT_POINTPOOL_HPP
#define FOUR_C_CUT_POINTPOOL_HPP

#include "baci_config.hpp"

#include "baci_cut_boundingbox.hpp"
#include "baci_cut_point.hpp"

#include <Teuchos_RCP.hpp>

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace CORE::GEO
{
  namespace CUT
  {
    class Point;
    class Edge;
    class Side;

    ///< Specifies the merge Strategy of the Pointpool
    enum PointpoolMergeStrategy
    {
      InitialLoad,
      SelfCutLoad,
      NormalCutLoad
    };

    /// OctTreeNode class that implements a single node of an OctTree
    /*!
     * Might need some speedup. (Uli)
     * this class is an OcTree-based pool that stores all points in tree-structure
     * Each node of this tree is represented by a bounding box and a RCPPointSet (a set of pointer
     * to nodes) that lie within this bounding box, and has 8 children nodes. Pointers of nodes are
     * contained on more levels of this tree, also the leafs of this tree is a node with the
     * additional info IsLeaf (no children) Points between 2 or more bounding boxes are strictly
     * related to one of these boxes
     */
    class OctTreeNode
    {
     public:
      /// empty constructor
      OctTreeNode(double norm) : bb_(Teuchos::rcp(BoundingBox::Create())), norm_(norm) {}

      /*========================================================================*/
      //! @name create and get points
      /*========================================================================*/

      /// If a point with the coordinates "x" does not exists, it creates a new point
      /// correspondingly
      Point* NewPoint(const double* x, Edge* cut_edge, Side* cut_side, double tolerance,
          PointpoolMergeStrategy merge_strategy);

      /// Get the point with the specified coordinates "x" from the pointpool
      Point* GetPoint(const double* x, Edge* cut_edge, Side* cut_side, double tolerance,
          PointpoolMergeStrategy merge_strategy);

      /// Simply insert p into the pointpool and correspondingly modify the boundingbox size
      void AddPoint(const double* x, Teuchos::RCP<Point> p);


      /*========================================================================*/
      //! @name get info about the tree
      /*========================================================================*/

      /// how many points are contained in the current bounding box
      unsigned size() const { return points_.size(); }

      /// is the current tree-node a leaf of the tree?
      bool IsLeaf() const { return nodes_[0] == Teuchos::null; }

      /*========================================================================*/
      //! @name collect adjacent objects
      /*========================================================================*/

      /// collect all edges
      void CollectEdges(const BoundingBox& edgebox, plain_edge_set& edges);

      /// collect all sides
      void CollectSides(const BoundingBox& sidebox, plain_side_set& sides);

      /*!
      \brief collect all elements near the sidebox
             (unused, does not work properly when there is no point adjacent to elements in a tree's
      leaf, e.g. when side lies within an element)
      */
      void CollectElements(const BoundingBox& sidebox, plain_element_set& elements);


      /*========================================================================*/
      //! @name print routine and others
      /*========================================================================*/

      /// reset the Point::Position of outside points
      void ResetOutsidePoints();

      /// print the tree at a given level
      void Print(int level, std::ostream& stream);

      /// Get Points
      const RCPPointSet& GetPoints() { return points_; }

      // get the node where point with specified coordinate lies
      OctTreeNode* FindNode(const double* coord, Point* p);

     private:
      /*========================================================================*/
      //! @name private routines
      /*========================================================================*/

      /// Create a point with the specified ID
      Teuchos::RCP<Point> CreatePoint(
          unsigned newid, const double* x, Edge* cut_edge, Side* cut_side, double tolerance);

      /// get the leaf where the point with the given coordinates lies in
      OctTreeNode* Leaf(const double* x);

      /// split the current boounding box (tree-node)
      void Split(int level);


      /*========================================================================*/
      //! @name private class variables
      /*========================================================================*/

      /// children of the current OctTreeNode
      Teuchos::RCP<OctTreeNode> nodes_[8];

      /// set of points contained in the bounding box related to this tree-node
      RCPPointSet points_;

      /// bounding box related to this tree-node
      Teuchos::RCP<BoundingBox> bb_;

      /// point coordinates where the current bounding box is split into 8 children bboxes
      CORE::LINALG::Matrix<3, 1> splitpoint_;

      /// norm
      double norm_;
    };



    /*!
    \brief OctTree based storage container (the PointPool class)
     */
    class PointPool
    {
     public:
      /// constructor
      explicit PointPool(double norm = 1);

      /// create a new point in the point PointPool
      Point* NewPoint(const double* x, Edge* cut_edge, Side* cut_side, double tolerance)
      {
        switch (probdim_)
        {
          case 2:
          {
            // extend the point dimension
            double x_ext[3] = {0.0, 0.0, 0.0};
            std::copy(x, x + 2, x_ext);

            return tree_.NewPoint(
                x_ext, cut_edge, cut_side, GetTolerance(x_ext, tolerance), merge_strategy_);
          }
          case 3:
            return tree_.NewPoint(
                x, cut_edge, cut_side, GetTolerance(x, tolerance), merge_strategy_);
          default:
            FOUR_C_THROW("Unsupported problem dimension! (probdim = %d)", probdim_);
            exit(EXIT_FAILURE);
        }

        FOUR_C_THROW("Impossible to reach this line!");
        exit(EXIT_FAILURE);
      }

      /// get the pointer to a stored point if possible
      Point* GetPoint(const double* x, Edge* cut_edge, Side* cut_side, double tolerance)
      {
        return tree_.GetPoint(x, cut_edge, cut_side, GetTolerance(x, tolerance), merge_strategy_);
      }

      /// get default tolerance for the case that the tolerance of a point is set to 0!!!
      double GetTolerance(const double* x, double tolerance)
      {
        if (tolerance == 0.0)
        {
          CORE::LINALG::Matrix<3, 1> px(x);
          return (px.NormInf() * POSITIONTOL);
        }
        else
          return tolerance;
      }

      /// get the size of the OctTree (total number of stored points)
      unsigned size() const { return tree_.size(); }

      /// Collects the edges that are fully or partially contained in the edgebox
      void CollectEdges(const BoundingBox& edgebox, plain_edge_set& edges)
      {
        tree_.CollectEdges(edgebox, edges);
      }

      /// Collects the sides that are fully or partially contained in the sidebox
      void CollectSides(const BoundingBox& sidebox, plain_side_set& sides)
      {
        tree_.CollectSides(sidebox, sides);
      }

      /// Collects the elements that are fully or partially contained in the sidebox
      void CollectElements(const BoundingBox& sidebox, plain_element_set& elements)
      {
        tree_.CollectElements(sidebox, elements);
      }

      /// reset Point::Position of all stored outside points
      void ResetOutsidePoints() { tree_.ResetOutsidePoints(); }

      /// print the OctTree
      void Print(std::ostream& stream) { tree_.Print(0, stream); }

      /// Get Points
      const RCPPointSet& GetPoints() { return tree_.GetPoints(); }

      void SetMergeStrategy(PointpoolMergeStrategy strategy) { merge_strategy_ = strategy; }

     private:
      ///< Current Merge Strategy of the Pointpool
      PointpoolMergeStrategy merge_strategy_;

      ///< the PointPool class is represented by the first node in the OctTree
      OctTreeNode tree_;

      /// problem dimension
      const unsigned probdim_;
    };
  }  // namespace CUT
}  // namespace CORE::GEO

FOUR_C_NAMESPACE_CLOSE

#endif
