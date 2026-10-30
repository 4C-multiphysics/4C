// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_COUPLING3D_CLASSES_HPP
#define FOUR_C_MORTAR_COUPLING3D_CLASSES_HPP

#include "4C_config.hpp"

#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mortar
{
  // forward declarations
  // namespace Utils
  //{
  // template<typename _Key, typename _Tp> class fixedsizemap;
  //}
  /*!
  \brief A class representing a special mortar element for 3D quadratic
         mortar coupling with the use auf auxiliary planes. This approach
         is based on "Puso, M.A., Laursen, T.A., Solberg, J., A segment-to-
         segment mortar contact method for quadratic elements and large
         deformations, CMAME, 197, 2008, pp. 555-566". For this type of
         quadratic formulation, a quadratic Mortar::Element is split into several
         linear IntElements, on which the geometrical coupling is performed.
         Note that this is a derived class from Mortar::Element, with the
         only difference that we explicitly set the node pointers here!

  */

  class IntElement : public Mortar::Element
  {
   public:
    /*!
    \brief Standard constructor

    \param lid   (in):    A locally unique element id
    \param id    (in):    Parent element global id
    \param owner (in):    owner processor of the element
    \param parshape (in): shape of parent element
    \param shape (in):    shape of this element
    \param numnode (in):  Number of nodes to this element
    \param nodeids (in):  ids of nodes adjacent to this element
    \param nodes (in):    pointers to nodes adjacent to this element
    \param isslave (in):  flag indicating whether element is slave or master side
    */
    IntElement(int lid, int id, int owner, Mortar::Element* parele, const Core::FE::CellType& shape,
        const int numnode, const int* nodeids, std::vector<Core::Nodes::Node*> nodes,
        const bool isslave, const bool rewind);


    /*!
    \brief Return local id of this IntElement

    */
    int lid() const override { return lid_; }

    /*!
    \brief Get shape type of parent element

    */
    Core::FE::CellType par_shape() const { return parele_->shape(); }

    /*!
    \brief Get shape type of parent element

    */
    Mortar::Element* par_ele() const { return parele_; }

    /*!
    \brief Affine map of IntElement coordinates to parent element

    */
    virtual bool map_to_parent(const double* xi, double* parxi);

    /*!
    \brief Affine map of IntElement coordinate derivatives to parent element

    */
    virtual bool map_to_parent(const std::vector<Core::Gen::Pairedvector<int, double>>& dxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& dparxi);

    /*!
    \brief Get the linearization of the spatial position of the Nodes for this IntEle.
           For Lagrange finite elements, this is trivial, since the nodes are interpolatory.
           For NURBS elements, we generated pseudo-nodes as the corners of the parameter
           space onto the actual surface. Those nodes now depend on all CPs and shape functions
           of the parent NURBS element.

           Returns a vector of vector of maps. Outer vector for the (pseudo-)nodes,
           inner vector for the spatial dimensions, map for the derivatives.
    */
    void node_linearization(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& nodelin) const override;

   protected:
    // don't want = operator and cctor
    IntElement operator=(const IntElement& old);
    IntElement(const IntElement& old);

    int lid_;  // local IntElement id

    // the integration element has its own vector of nodes
    // for Lagrange elements, those are copies of the original mortar nodes
    // for NURBS elements, those are newly created pseudo-nodes, which are
    // located on the physical surface. The "nodes" in IGA are already used
    // with control points
    // Be careful here: the pseudo-nodes need an Id!
    // They are given the Id of the closest control point. However, be aware
    // that this Id is now used twice: once for the control point (as a part
    // of the discretization) and once for the pseudo-node, which is only
    // stored here temporarily.
    std::vector<Mortar::Node> nodes_;
    std::vector<Core::Nodes::Node*> nodes_ptr_;
    const bool rewind_;  // if the parameter space of the int element has been rewinded

    Mortar::Element* parele_;

  };  // class IntElement


  /*!
  \brief A class representing one Integration Cell after triangulation
         of the clip polygon of slave and master element from the Coupling
         class. This class provides some basic functionality a Mortar::Element would
         also provide (coords, shape functions and derivatives, Jacobian, ...).
         Note that an IntCell can EITHER live in physical space (this is the
         case when an auxiliary plane is used for 3D coupling) or in the slave
         element parameter space (this is the case when 3D coupling is performed
         on the slave surface without any auxiliary plane). Of course, in the
         2nd case the third coordinate of all intcell points is zero!

  */

  class IntCell
  {
   public:
    /*!
    \brief Standard constructor

    Constructs an instance of this class.<br>
    Note that this is \b not a collective call as coupling is
    performed in parallel by individual processes.

    */
    IntCell(int id, int nvertices, Core::LinAlg::Matrix<3, 3>& coords, double* auxn,
        const Core::FE::CellType& shape, std::vector<Core::Gen::Pairedvector<int, double>>& linv1,
        std::vector<Core::Gen::Pairedvector<int, double>>& linv2,
        std::vector<Core::Gen::Pairedvector<int, double>>& linv3,
        std::vector<Core::Gen::Pairedvector<int, double>>& linauxn);

    /*!
    \brief Destructor

    */
    virtual ~IntCell() = default;
    //! @name Access methods

    /*!
    \brief Return ID of this intcell

    */
    const int& id() const { return id_; }

    /*!
    \brief Set slave element ID of this intcell

    */
    void set_slave_id(const int slaveid)
    {
      slaveId_ = slaveid;
      return;
    }

    /*!
    \brief Return slave element ID of this intcell

    */
    const int& get_slave_id() const
    {
      if (slaveId_ < 0) FOUR_C_THROW("Invalid slave element ID for this integration cell!");

      return slaveId_;
    }

    /*!
    \brief Set master element ID of this intcell

    */
    void set_master_id(const int masterid)
    {
      masterId_ = masterid;
      return;
    }
    /*!
    \brief Return master element ID of this intcell

    */
    const int& get_master_id() const
    {
      if (masterId_ < 0) FOUR_C_THROW("Invalid master element ID for this integration cell!");

      return masterId_;
    }

    /*!
    \brief Return number of vertices of this intcell

    */
    int num_vertices() const { return nvertices_; }

    /*!
    \brief Return current area

    */
    virtual double& area() { return area_; }

    /*!
    \brief Return coordinates of intcell vertices

    */
    const Core::LinAlg::Matrix<3, 3>& coords() { return coords_; }

    /*!
    \brief Return normal of auxiliary plane of this intcell

    */
    virtual double* auxn() { return auxn_; }

    /*!
    \brief Get shape type of element

    */
    virtual Core::FE::CellType shape() const { return shape_; }

    /*!
    \brief Return one of the three 'DerivVertex' maps (vectors) of this node

    These maps contain the directional derivatives of the intcell's vertex
    coordinates with respect to the slave and master displacements. A vector
    is used because the coordinates themselves are a vector (2 components for
    the 'coupling in slave parameter space' case). The respective vertex (1,2,3)
    is addressed by an int-variable and checked internally.

    */
    virtual std::vector<Core::Gen::Pairedvector<int, double>>& get_deriv_vertex(int i)
    {
      if (shape_ == Core::FE::CellType::line2)
      {
        if (i < 0 || i > 1) FOUR_C_THROW("int_line has 2 vertex linearizations only!");
        return linvertex_[i];
      }
      else
      {
        if (i < 0 || i > 2) FOUR_C_THROW("IntCell has 3 vertex linearizations only!");
        return linvertex_[i];
      }
    }

    /*!
    \brief Return the 'DerivAuxn' map (vector) of this intcell

    */
    virtual std::vector<Core::Gen::Pairedvector<int, double>>& get_deriv_auxn() { return linauxn_; }

    //@}

    //! @name Evaluation methods

    /*!
    \brief Interpolate global coordinates for given local intcell coordinates

    This method interpolates global coordinates for a given local intcell
    coordinate variable using the intcell vertex coordinates. For interpolation
    one can choose between shape functions or shape function derivatives!

    \param xi (in)        : local intcell coordinates
    \param inttype (in)   : set to 0 for shape function usage,
                            set to 1 for derivative xi usage
                            set to 2 for derivative eta usage
    \param globccord (out): interpolated global coordinates
    */
    virtual bool local_to_global(const double* xi, double* globcoord, int inttype);

    // output functionality
    virtual void print();

    /*!
    \brief Evaluate shape functions and derivatives

    */
    virtual bool evaluate_shape(
        const double* xi, Core::LinAlg::Matrix<3, 1>& val, Core::LinAlg::Matrix<3, 2>& deriv);

    /*!
    \brief Evaluate Jacobian determinant for parameter space integration

    */
    virtual double jacobian();

    /*!
    \brief Compute Jacobian determinant derivative
           Note that this is a linearization with respect to the intcell
           vertices, which themselves have to be linearized later, of course!
    */
    virtual void deriv_jacobian(Core::Gen::Pairedvector<int, double>& derivjac);

    //@}

   protected:
    // don't want = operator and cctor
    IntCell operator=(const IntCell& old);
    IntCell(const IntCell& old);

    int id_;                             // local ID of this intcell
    int slaveId_;                        // id of slave element
    int masterId_;                       // id of master element
    int nvertices_;                      // number of vertices (always 3)
    double area_;                        // integration cell area
    Core::LinAlg::Matrix<3, 3> coords_;  // coords of cell vertices
    double auxn_[3];                     // normal of auxiliary plane (3D)

    bool auxplane_;             // flag indicating coupling strategy (true = auxplane)
    Core::FE::CellType shape_;  // shape of this element (always tri3)
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>
        linvertex_;  // derivatives of the 3 vertices
    std::vector<Core::Gen::Pairedvector<int, double>>
        linauxn_;  // derivatives of auxiliary plane normal

  };  // class IntCell

  /*!
  \brief A class representing one Vertex during the polygon clipping of
         slave and master element from the Coupling class. Besides the
         vertex coordinates this class provides a pointer to the next
         vertex on the polygon.
         Note that a Vertex can EITHER live in physical space (this is the
         case when an auxiliary plane is used for 3D coupling) or in the slave
         element parameter space (this is the case when 3D coupling is performed
         on the slave surface without any auxiliary plane). Of course, in the
         2nd case the third coordinate of all vertices is zero!

  */

  class Vertex
  {
   public:
    //! @name Enums and Friends
    enum VType  // vertex types recognized by Vertex
    {
      source,      // source node
      projtarget,  // projected target node
      lineclip,    // clipping point of two lines
      target,      // target node (LTS)
      projsource   // projected source node (LTS)
    };

    //@}

    /*!
    \brief Standard constructor

    Constructs an instance of this class.<br>
    Note that this is \b not a collective call as coupling is
    performed in parallel by individual processes.

    */
    Vertex(std::vector<double> coord, Vertex::VType type, std::vector<int> nodeids, Vertex* next);

    /*!
    \brief Copy Constructor

    Makes a deep copy of a Vertex

    */
    Vertex(const Vertex& old);

    /*!
    \brief Destructor

    */
    virtual ~Vertex() = default;
    //! @name Access methods

    /*!
    \brief Return vector of vertex coordinates (length 3)

    */
    virtual std::vector<double>& coord() { return coord_; }
    const std::vector<double>& coord() const { return coord_; }

    /*!
    \brief Return vertex type (slave, projmaster or lineclip)

    */
    virtual Vertex::VType v_type() const { return type_; }

    /*!
    \brief Return pointer to next vertex on polygon

    */
    virtual Vertex* next() { return next_; }

    /*!
    \brief Assign pointer to next vertex on polygon

    */
    virtual void assign_next(Vertex* assign) { next_ = assign; }

    /*!
    \brief Return vector of relevant node ids (length 1 or 4)
    Note that for a slave or projmaster type vertex only the
    respective node id is relevant, therefore length 1. For a
    lineclip type vertex we need the 4 node ids of both the
    corresponding slave and master lines which intersect!

    */
    virtual std::vector<int>& nodeids()
    {  // if(type_==Vertex::slave && nodeids_.size()!=1) FOUR_C_THROW("Error: Vertex Ids");
      // if(type_==Vertex::projmaster && nodeids_.size()!=1) FOUR_C_THROW("Error: Vertex Ids");
      // if(type_==Vertex::lineclip && nodeids_.size()!=4) FOUR_C_THROW("Error: Vertex Ids");
      return nodeids_;
    }
    const std::vector<int>& nodeids() const
    {  // if(type_==Vertex::slave && nodeids_.size()!=1) FOUR_C_THROW("Error: Vertex Ids");
      // if(type_==Vertex::projmaster && nodeids_.size()!=1) FOUR_C_THROW("Error: Vertex Ids");
      // if(type_==Vertex::lineclip && nodeids_.size()!=4) FOUR_C_THROW("Error: Vertex Ids");
      return nodeids_;
    }

    //@}


   protected:
    std::vector<double> coord_;  // vertex coordinates (length 3)
    Vertex::VType type_;         // vertex type (slave,projmaster,lineclip)
    std::vector<int> nodeids_;   // relevant ids (1 if slave or master, 4 if lineclip)
    Vertex* next_;               // pointer to next vertex on polygon

  };  // class Vertex

}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
