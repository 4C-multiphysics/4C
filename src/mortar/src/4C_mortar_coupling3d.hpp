// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_COUPLING3D_HPP
#define FOUR_C_MORTAR_COUPLING3D_HPP

#include "4C_config.hpp"

#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_input.hpp"
#include "4C_utils_pairedvector.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Mortar
{
  // forward declarations
  class Element;
  class ParamsInterface;

  /*!
   \brief A class representing the framework for mortar coupling of ONE
   slave element and ONE master element of a mortar interface in
   3D. Concretely, this class controls projection, overlap detection
   and finally integration of the mortar coupling matrices D and M
   and possibly the weighted gap vector g~.
   Note that 3D Coupling can EITHER be done in physical space (this is
   the case when an auxiliary plane is used) or in the slave element
   parameter space (this is the case when everything is done directly
   on the slave surface without any auxiliary plane). The boolean class
   variable auxplane_ decides about this (true = auxiliary plane).

   */

  class Coupling3d
  {
   public:
    /*!
     \brief Constructor with shape function specification

     Constructs an instance of this class and enables custom shape function types.<br>
     Note that this is \b not a collective call as coupling is
     performed in parallel by individual processes.

     */
    Coupling3d(Core::FE::Discretization& idiscret, int dim, bool quad,
        Teuchos::ParameterList& params, Mortar::Element& sele, Mortar::Element& mele);

    /*!
     \brief Destructor

     */
    virtual ~Coupling3d() = default;
    //! @name Access methods

    /*!
     \brief Get interface discretization

     */
    virtual Core::FE::Discretization& discret() const { return idiscret_; }

    /*!
     \brief Get communicator

     */
    virtual MPI_Comm get_comm() const;

    /*!
     \brief Get problem dimension (here: 3D)

     */
    virtual int n_dim() const { return dim_; };

    /*!
     \brief Get coupling / FE ansatz type (true = quadratic)

     */
    virtual bool quad() const { return quad_; };

    /*!
     \brief Get coupling slave element

     */
    virtual Mortar::Element& source_element() const { return sele_; }

    /*!
     \brief Get coupling master element

     */
    virtual Mortar::Element& target_element() const { return mele_; }

    /*!
     \brief Get coupling slave integration element

     Note that (here) for linear ansatz functions in 3D, this is IDENTICAL to
     the SlaveElement() as no splitting of the Mortar::Elements is performed.
     For the 3D quadratic case with the use of auxiliary planes, this
     method is overloaded by the derived class Mortar::Coupling3dQuad!

     */
    virtual Mortar::Element& source_int_element() const { return sele_; }

    /*!
     \brief Get coupling master integration element

     Note that (here) for linear ansatz functions in 3D, this is IDENTICAL to
     the SlaveElement() as no splitting of the Mortar::Elements is performed.
     For the 3D quadratic case with the use of auxiliary planes, this
     method is overloaded by the derived class Mortar::Coupling3dQuad!

     */
    virtual Mortar::Element& target_int_element() const { return mele_; }

    /*!
     \brief Return center of auxiliary plane

     */
    virtual double* auxc() { return auxc_; }
    const double* auxc() const { return auxc_; }

    /*!
     \brief Return normal of auxiliary plane

     */
    virtual double* auxn() { return auxn_; }
    const double* auxn() const { return auxn_; }

    /*!
     \brief Return length of Auxn() before normalization

     */
    virtual double& lauxn() { return lauxn_; }

    /*!
     \brief Return vector of (projected) slave node vertex objects

     */
    virtual std::vector<Vertex>& source_vertices() { return svertices_; }
    const std::vector<Vertex>& source_vertices() const { return svertices_; }

    /*!
     \brief Return vector of projected master node vertex objects

     */
    virtual std::vector<Vertex>& target_vertices() { return mvertices_; }
    const std::vector<Vertex>& target_vertices() const { return mvertices_; }

    /*!
     \brief Return vector of clip polygon vertex objects

     */
    virtual std::vector<Vertex>& clip() { return clip_; }
    const std::vector<Vertex>& clip() const { return clip_; }

    /*!
     \brief Return vector of integration cells

     */
    virtual std::vector<std::shared_ptr<IntCell>>& cells() { return cells_; }
    const std::vector<std::shared_ptr<IntCell>>& cells() const { return cells_; }

    /*!
     \brief Return the 'DerivAuxn' map (vector) of this coupling pair

     */
    virtual std::vector<Core::Gen::Pairedvector<int, double>>& get_deriv_auxn()
    {
      return derivauxn_;
    }
    const std::vector<Core::Gen::Pairedvector<int, double>>& get_deriv_auxn() const
    {
      return derivauxn_;
    }

    /*!
     \brief Return the LM interpolation / testing type for quadratic FE

     */
    virtual Mortar::LagMultQuad lag_mult_quad() const { return lmquadtype_; };

    /*!
     \brief Get interface contact parameter list

     */
    virtual Teuchos::ParameterList& interface_params() { return imortar_; };

    /*!
     \brief Return the LM shape fcn type

     */
    Mortar::ShapeFcn shape_fcn() const { return shapefcn_; };

    //@}

    //! @name Evlauation methods

    /*!
     \brief Evaluate coupling (3D)

     */
    virtual bool evaluate_coupling();

    /*!
     \brief Checks roughly whether the two elements are near (3D)

     This methods computes the distance of the two element centers.
     If they are not close, then coupling is stopped for the pair.

     */
    virtual bool rough_check_centers();

    /*!
     \brief Checks roughly whether the two elements are near (3D)

     This methods checks the orientation of slave and master element.
     Projection and overlap detection only make sense if the scalar
     product of the two center normals is negative (i.e. if the two
     elements form an angle smaller than 90 degrees).

     */
    virtual bool rough_check_orient();

    /*!
     \brief Integrate the integration cells (3D)

     This method creates an integrator object for the cell triangles,
     then projects the Gauss points back onto slave and master elements
     (1st case, aux. plane) or only back onto the master element (2nd case)
     in order to evaluate the respective shape function there. Then
     entries of the mortar matrix M and the weighted gap g are integrated
     and assembled into the slave element nodes.

     */
    virtual bool integrate_cells(const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr);

    //@}

    //! @name Visualization and Debugging methods

    /*!
     \brief Visualize integration cells with gmsh

     */
    virtual void gmsh_output_cells(int lid) const;

    //@}

    //! @name Linearization methods (for mortar contact only!)

    /*!
     \brief Linearization of clip vertex coordinates (3D)

     */
    virtual bool vertex_linearization(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex,
        std::map<int, double>& projpar, bool printderiv = false) const
    {
      return true;
    }

    /*!
     \brief Linearization of clip vertex coordinates (3D)

     */
    virtual bool center_linearization(
        const std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex,
        std::vector<Core::Gen::Pairedvector<int, double>>& lincenter) const
    {
      return true;
    }

    //@}
   private:
    /*!
     \brief Build auxiliary plane from slave element (3D)

     This method builds an auxiliary plane based on the possibly
     warped slave element of this coupling class. This plane is
     defined by the slave normal at the slave element center.

     */
    virtual bool auxiliary_plane();

    /*!
     \brief Triangulation of clip polygon (3D)

     In a first step, this method computes linearizations of all clip
     polygon vertices and stores them into local maps. Then, one of the
     two available triangulation algorithms (Delaunay, Center) is called
     to do the actual triangulation of the convex clip polygon.

     delaunay_triangulation() is more efficient, as it always generates N-2 triangles
     for a clip polygon with N vertices. Moreover, the triangle shape is optimized
     according to the Delaunay criterion, which maximizes the smallest internal
     angle. Thus, delaunay_triangulation() is used BY DEFAULT here.

     center_triangulation() is an old version that was based on first finding the
     center of the clip polygon by computing the centroid / geometric center
     (see M. Puso, A 3D mortar method for solid mechanics, IJNME, 2004).
     The clip polygon is triangulated into integration cells all containing two
     vertices and the center point. Thus, we always generate N triangles here
     for a clip polygon with N vertices (except for the special case N=3).
     Thus, center_triangulation() is NOT used anymore.

     */
    virtual bool triangulation(std::map<int, double>& projpar, double tol);
    virtual bool delaunay_triangulation(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex, double tol);
    virtual bool center_triangulation(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex, double tol);

    /*!
     \brief Check / set projection status of slave nodes (3D)

     This method checks for all slave nodes if they are part of the clip
     polygon (equal to any vertex). If so the HasProj status is set true!

     */
    virtual bool has_proj_status();

    /*!
     \brief Compute and return area of clipping polygon (3D)

     This method automatically returns the relevant(!) area of the
     current clip polygon, depending on whether coupling is performed
     in the auxiliary plane or in the slave parameter space.

     */
    virtual double polygon_area();

    /*!
     \brief Clipping of slave and master element (3D) (NEW VERSION based on convex hull)

     Note that PolygonClipping3D can EITHER be done in physical space (this
     is the case when an auxiliary plane is used for 3D coupling) or in the
     slave element parameter space (this is the case when 3D coupling is
     performed on the slave surface without any auxiliary plane).

     This method applies a polygon clipping algorithm to find the
     polygon defined by the projection of the slave and master element
     in the auxiliary plane (1st case) or defined by the slave element
     itself and the projection of the master element into the slave
     parameter space (2nd case). Of course, in the 2nd case the clipping
     plane has the normal (0,0,1). As input variables the clipping algorithm
     requires two sets of vertices, which define sele_ / mele_ respectively.
     The clipping algorithm is based on finding the convex hull of three
     sets of vertices: sele_ vertices, mele_ vertices, line intersections,
     using an algorithm quite similar to the well-known "Graham Scan".
     This makes the algorithm FAR MORE ROBUST than the old approach above.
     Returns a vector of vertices defining the clipped polygon.

     In some cases, projection of the master element may lead to a non-convex
     input polygon poly2list, which leads to a FOUR_C_THROW. Before the error is
     thrown, this method (similarly to RoughCheck()) again checks if the two
     elements to the clipped are close to each other at all and returns a
     boolean flag. If they are not close, then coupling is stopped for the pair.

     \param poly1list (in): vector of vertex objects for input polygon 1 (slave)
     \param poly2list (in): vector of vertex objects for input polygon 2 (master)
     \param respoly (out): vector of vertex objects for result polygon
     \param tol (in): clipping tolerance for close vertices detection

     */
    virtual bool polygon_clipping_convex_hull(std::vector<Vertex>& poly1list,
        std::vector<Vertex>& poly2list, std::vector<Vertex>& respoly, double& tol);

    /*!
     \brief Projection of slave element onto aux. plane (3D)

     This method projects the nodes of the given slave CElement
     onto the auxiliary plane derived before.

     */
    virtual bool project_slave();

    /*!
     \brief Projection of master element onto aux. plane (3D)

     This method projects the nodes of the current master CElement
     onto the auxiliary plane derived from the slave CElement before.

     */
    virtual bool project_master();

    /*!
     \brief Checks roughly whether the two elements are near (3D)

     This methods computes the distance of all master nodes to the
     slave element (auxiliary plane). If they are not close, then
     coupling is stopped for the pair.

     */
    virtual bool rough_check_nodes();

   protected:
    /*!
     \brief Compute and return area of slave element (3D)

     This method automatically returns the relevant(!) area of the
     current slave element, depending on whether coupling is performed
     in the auxiliary plane or in the slave parameter space.

     */
    virtual double source_element_area() const;

    // don't want = operator and cctor
    Coupling3d operator=(const Coupling3d& old);
    Coupling3d(const Coupling3d& old);

    Core::FE::Discretization& idiscret_;  // discretization of the contact interface
    int dim_;                             // problem dimension (here: 3D)
    Mortar::ShapeFcn shapefcn_;           // lm shape function type
    bool quad_;                           // flag indicating coupling type (true = quadratic)
    Mortar::LagMultQuad lmquadtype_;      // type of quadratic lm interpolation

    Mortar::Element& sele_;            // slave element to perform coupling for
    Mortar::Element& mele_;            // master element to perform coupling for
    Teuchos::ParameterList& imortar_;  // containing contact input parameters
    double auxc_[3];                   // center of auxiliary plane
    double auxn_[3];                   // normal of auxiliary plane
    double lauxn_;                     // length of interpolated Auxn() before normalization
    std::vector<Vertex> svertices_;    // slave node vertex objects
    std::vector<Vertex> mvertices_;    // master node vertex objects
    std::vector<Vertex> clip_;         // clipped polygon vertex objects

    std::vector<std::shared_ptr<IntCell>> cells_;  // vector of integration cells
    std::vector<Core::Gen::Pairedvector<int, double>>
        derivauxn_;  // derivatives of auxiliary plane normal
  };
  // class Coupling3d

  /*!
   \brief A class representing the framework for mortar coupling of ONE
   slave element and ONE master element of a mortar interface in
   3D. Concretely, this class controls projection, overlap
   detection and finally integration of the mortar coupling matrices
   D and M and possibly the weighted gap vector g~.

   This is a special derived class for 3D quadratic mortar coupling
   with the use of auxiliary planes. This approach is based on
   "Puso, M.A., Laursen, T.A., Solberg, J., A segment-to-segment
   mortar contact method for quadratic elements and large deformations,
   CMAME, 197, 2008, pp. 555-566". For this type of formulation, a
   quadratic Mortar::Element is split into several linear IntElements,
   on which the geometrical coupling is performed. Thus, we additionally
   hand in in two IntElements to Coupling3dQuad.

   */

  class Coupling3dQuad : public Coupling3d
  {
   public:
    /*!
     \brief Constructor with shape function specification

     Constructs an instance of this class and enables custom shape function types.<br>
     Note that this is \b not a collective call as coupling is
     performed in parallel by individual processes.

     */
    Coupling3dQuad(Core::FE::Discretization& idiscret, int dim, bool quad,
        Teuchos::ParameterList& params, Mortar::Element& sele, Mortar::Element& mele,
        Mortar::IntElement& sintele, Mortar::IntElement& mintele);


    //! @name Access methods

    /*!
     \brief Get coupling slave integration element

     */
    Mortar::IntElement& source_int_element() const override { return sintele_; }

    /*!
     \brief Get coupling master integration element

     */
    Mortar::IntElement& target_int_element() const override { return mintele_; }

   protected:
    // don't want = operator and cctor
    Coupling3dQuad operator=(const Coupling3dQuad& old);
    Coupling3dQuad(const Coupling3dQuad& old);

    Mortar::IntElement& sintele_;  // slave sub-integration element
    Mortar::IntElement& mintele_;  // slave sub-integration element
  };
  // class Coupling3dQuad

  /*!
   \brief A class representing the framework for mortar coupling of ONE
   slave element and SEVERAL master elements of a mortar interface in
   3D. Concretely, this class simply stores several Coupling3d objects.

   */

  class Coupling3dManager
  {
   public:
    /*!
     \brief Constructor with shape function specification

     Constructs an instance of this class and enables custom shape function types.<br>
     Note that this is \b not a collective call as coupling is
     performed in parallel by individual processes.

     */
    Coupling3dManager(Core::FE::Discretization& idiscret, int dim, bool quad,
        Teuchos::ParameterList& params, Mortar::Element* sele, std::vector<Mortar::Element*> mele);

    /*!
     \brief Destructor

     */
    virtual ~Coupling3dManager() = default;
    /*!
     \brief Get coupling slave element

     */
    virtual Mortar::Element& source_element() const { return *sele_; }

    /*!
     \brief Get one specific coupling master element

     */
    virtual Mortar::Element& target_element(int k) const { return *(mele_[k]); }

    /*!
     \brief Get all coupling master elements

     */
    virtual std::vector<Mortar::Element*> target_elements() const { return mele_; }

    /*!
     \brief Get coupling pairs

     */
    virtual std::vector<std::shared_ptr<Mortar::Coupling3d>>& coupling() { return coup_; }

    /*!
     \brief Get type of integration scheme

     */
    virtual Mortar::IntType int_type() { return integrationtype_; }

    /*!
     \brief Get coupling type

     */
    virtual const bool& quad() { return quad_; };

    /*!
     \brief Get flag indicating dual consistent lm

     */
    virtual bool lm_dual_consistent() { return lmdualconsistent_; }

    /*!
     \brief Get communicator

     */
    virtual MPI_Comm get_comm() const;

    /*!
     \brief Evaluate coupling pairs

     */
    virtual bool evaluate_coupling(std::shared_ptr<Mortar::ParamsInterface> mparams_ptr);

    /*!
     \brief Return the LM interpolation / testing type for quadratic FE

     */
    virtual Mortar::LagMultQuad lag_mult_quad() { return lmquadtype_; };

    /*!
     \brief Return the LM shape fcn type

     */
    virtual Mortar::ShapeFcn shape_fcn() { return shapefcn_; }
    //@}

   protected:
    /*!
     \brief Evaluate mortar-coupling pairs

     */
    virtual void integrate_coupling(const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr);

    /*!
     \brief Calculate consistent dual shape functions in boundary elements

     */
    virtual void consist_dual_shape();

    // don't want = operator and cctor
    Coupling3dManager operator=(const Coupling3dManager& old);
    Coupling3dManager(const Coupling3dManager& old);

    Core::FE::Discretization& idiscret_;           // discretization of the contact interface
    int dim_;                                      // problem dimension (here: 3D)
    Mortar::IntType integrationtype_;              // integration type
    Mortar::ShapeFcn shapefcn_;                    // lm shape function type
    Mortar::LagMultQuad lmquadtype_;               // type of quadratic lm interpolation
    Mortar::ConsistentDualType lmdualconsistent_;  // flag indicating dual consistent lm
    bool quad_;                           // flag indicating coupling type (true = quadratic)
    Teuchos::ParameterList& imortar_;     // containing contact input parameters
    Mortar::Element* sele_;               // slave element
    std::vector<Mortar::Element*> mele_;  // master elements
    std::vector<std::shared_ptr<Mortar::Coupling3d>> coup_;  // coupling pairs
  };
  // class Coupling3dManager

  /*!
   \brief A class representing the framework for mortar coupling of ONE
   slave element and SEVERAL master elements of a mortar interface in
   3D with QUADRATIC ELEMENTS. Concretely, this class simply sub-divides
   elements and then stores several Coupling3d objects.

  */

  class Coupling3dQuadManager : public Coupling3dManager
  {
   public:
    /*!
     \brief Constructor
     */
    Coupling3dQuadManager(Core::FE::Discretization& idiscret, int dim, bool quad,
        Teuchos::ParameterList& params, Mortar::Element* sele, std::vector<Mortar::Element*> mele);


    Mortar::LagMultQuad lag_mult_quad() override
    {
      return Teuchos::getIntegralValue<Mortar::LagMultQuad>(imortar_, "LM_QUAD");
    }

    /*!
     \brief Get coupling slave element

     */
    Mortar::Element& source_element() const override { return *sele_; }

    /*!
     \brief Get one specific coupling master element

     */
    Mortar::Element& target_element(int k) const override { return *(mele_[k]); }

    /*!
     \brief Get all coupling master elements

     */
    std::vector<Mortar::Element*> target_elements() const override { return mele_; }

    /*!
     \brief Get integration type

     */
    Mortar::IntType int_type() override { return integrationtype_; }

    /*!
     \brief Get coupling type

     */
    const bool& quad() override { return quad_; }

    // @

   protected:
    /*!
     \brief Evaluate mortar-coupling pairs

     */
    void integrate_coupling(const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr) override;

    /*!
     \brief Split Mortar::Elements into IntElements for 3D quadratic coupling

     */
    virtual bool split_int_elements(
        Mortar::Element& ele, std::vector<std::shared_ptr<Mortar::IntElement>>& auxele);

    // don't want = operator and cctor
    Coupling3dQuadManager operator=(const Coupling3dQuadManager& old);
    Coupling3dQuadManager(const Coupling3dQuadManager& old);
  };

}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
