// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_BOUNDINGBOX_HPP
#define FOUR_C_CUT_BOUNDINGBOX_HPP

#include "4C_config.hpp"

#include "4C_cut_tolerance.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class Node;
  class Edge;
  class Side;
  class Facet;
  class VolumeCell;
  class Element;

  /*! \brief Construction of boundingbox over the considered geometrical element
   *  for fast overlap detection */
  class BoundingBox
  {
   public:
    /// @name Different Create methods
    /// @{

    static BoundingBox* create();
    static BoundingBox* create(Node& node);
    static BoundingBox* create(Edge& edge);
    static BoundingBox* create(Side& side);
    static BoundingBox* create(VolumeCell& volcell);
    static BoundingBox* create(VolumeCell& volcell, Element* elem1);
    static BoundingBox* create(Element& element);

    /// @}
   protected:
    /// empty constructor
    BoundingBox() : empty_(true) {};

    /// Initialize the BoundingBox by using information of a node
    void init(Node& node);

    /// Initialize the BoundingBox by using information of an edge
    void init(Edge& edge);

    /// Initialize the BoundingBox by using information of a side
    void init(Side& side);

    /// Initialize the BoundingBox by using information of a volume cell
    void init(VolumeCell& volcell);

    /** \brief initializes bounding box over the volumecell in local
     *  coordinates (rst) with respect to elem1 */
    void init(VolumeCell& volcell, Element* elem1);

    /// Initialize the BoundingBox by using the information of an element
    void init(Element& element);

   public:
    /// destructor
    virtual ~BoundingBox() = default;

    /*!
     \brief Make the boundingbox larger to include this edge within the box
     */
    void assign(Edge& edge);

    /*! \brief Make the boundingbox larger to include this side within the
     *  box */
    void assign(Side& side);

    /*! \brief Make the boundingbox larger to include this element
     * within the box */
    void assign(Element& element);

    double operator()(int i, int j) const { return box_(i, j); }

    /*!  \brief If necessary make the boundingbox larger to include
     * this point as one of the corners of the box */
    virtual void add_point(const double* x) = 0;

    /*! \brief If necessary make the boundingbox larger to include
     *  this point as one of the corners of the box */
    void add_point(const Core::LinAlg::Matrix<3, 1>& p) { add_point(p.data()); }

    /*! \brief If necessary make the boundingbox larger to include
     *  all these nodes as one of the corners of the box */
    void add_points(const std::vector<Node*>& nodes);

    /*! \brief Check whether "b" is within this boundingbox */
    bool within(double norm, const BoundingBox& b) const;

    /*! \brief Check the point is within this boundingbox */
    virtual bool within(double norm, const double* x) const;

    /*! \brief Check these points are within this boundingbox */
    bool within(double norm, const Core::LinAlg::SerialDenseMatrix& xyz) const;

    /*! \brief Check the element is within this boundingbox */
    bool within(double norm, Element& element) const;

    /*! \brief Print the corner points of boundingbox on the screen */
    void print();

    double minx() const { return box_(0, 0); }
    double miny() const { return box_(1, 0); }
    double minz() const { return box_(2, 0); }

    double maxx() const { return box_(0, 1); }
    double maxy() const { return box_(1, 1); }
    double maxz() const { return box_(2, 1); }

    /*! \brief Get the out-most point of the boundingbox */
    void corner_point(int i, double* x);

    /*! \brief Get the boundingbox */
    const Core::LinAlg::Matrix<3, 2>& get_bounding_volume() const { return box_; }

    /// Compute and return the diagonal length of the BoundingBox
    double diagonal()
    {
      return sqrt((maxx() - minx()) * (maxx() - minx()) + (maxy() - miny()) * (maxy() - miny()) +
                  (maxz() - minz()) * (maxz() - minz()));
    }

   protected:
    bool in_between(const double& norm, const double& smin, const double& smax, const double& omin,
        const double& omax) const;

    bool empty_;
    Core::LinAlg::Matrix<3, 2> box_;
  };  // class BoundingBox

  /*--------------------------------------------------------------------------*/
  /** Simple modification of the BoundingBox implementation, such that
   *  a correct and deterministic evaluation is ensured for all problem
   *  dimensions.
   *
   *  \author hiermeier \date 11/16 */
  template <unsigned probdim>
  class ConcreteBoundingBox : public BoundingBox
  {
   public:
    /// empty constructor
    ConcreteBoundingBox() : BoundingBox() {};

    /** add a new point under consideration of the problem dimension. If a
     *  2-dimensional problem is observed, we ignore the z-coordinate and keep
     *  a value of zero. */
    void add_point(const double* x) override;

    /** \brief [derived]
     *
     *  special treatment of the 2-D case if necessary. */
    bool within(double norm, const double* x) const override;
  };  // class ConcreteBoundingBox

}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
