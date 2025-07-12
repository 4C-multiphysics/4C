// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_ELEMENT_HPP
#define FOUR_C_FEM_GENERAL_ELEMENT_HPP


#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputParameterContainer;
}

namespace Core::Mat
{
  class Material;
}

namespace Core::GeometricSearch
{
  struct BoundingVolume;
  class GeometricSearchParams;
}  // namespace Core::GeometricSearch

namespace Core::FE::MeshFree
{
  template <typename>
  class MeshfreeBin;
}


namespace Core::FE
{
  class Discretization;
  class DiscretizationHDG;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SerialDenseMatrix;
  class SerialDenseVector;
}  // namespace Core::LinAlg

namespace Core::Nodes
{
  class Node;
}

namespace Core::Conditions
{
  class Condition;
}

namespace Core::Elements
{
  class ElementType;
  class FaceElement;

  /// Location data for one dof set
  /*!
    A helper that manages location vectors. Required since there can be an
    arbitrary number of location vectors, matching the number of dofsets in
    the discretization.

   */
  class LocationData
  {
   public:
    /// clear all vectors
    void clear()
    {
      lm_.clear();
      lmowner_.clear();
      stride_.clear();
    }

    /// return number of dofs collected
    int size() const { return lm_.size(); }

    /// global dof numbers of elemental dofs
    std::vector<int> lm_;

    /// Owner of dof (that is owner of node or element the dof belongs to)
    std::vector<int> lmowner_;

    /// Nodal stride (that is, how many dofs are guaranteed to be contiguous in the system matrix)
    std::vector<int> stride_;
  };


  /// Location data for all dof sets
  /*!
    A helper that manages location vectors. Required since there can be an
    arbitrary number of location vectors, matching the number of dofsets in
    the discretization.

   */
  class LocationArray
  {
   public:
    /// constructed with number of dofsets in discretization
    explicit LocationArray(int size) : data_(size) {}

    /// clear all location entries
    void clear()
    {
      for (unsigned i = 0; i < data_.size(); ++i) data_[i].clear();
    }

    /// access location entry
    LocationData& operator[](int i) { return data_[i]; }

    /// access location entry
    const LocationData& operator[](int i) const { return data_[i]; }

    /// number of location entries, that is number of dofsets in discretization
    int size() const { return data_.size(); }

   private:
    std::vector<LocationData> data_;
  };

  /*!
  \brief A virtual class all elements used in the discretization management have to implement

  This is the pure virtual base class for all finite elements to be used with
  the Core::FE::Discretization. Every element (and boundary condition) to be used
  with the discretization management module has to implement this class.
  It implements various basic element methods and
  stores basic information such as element to node connectivity.

  For elements that are attached to another 'parent' element, we use the derived
  class FaceElement. All boundary element types as well as interior face element
  classes are of type FaceElement. A FaceElement behaves as a usual Element most
  of the time, but it does have additional query methods for the parent element
  and other connectivity information.

  */
  class Element : public Core::Communication::ParObject
  {
   public:
    template <typename>
    friend class Core::FE::MeshFree::MeshfreeBin;

    //! @name Constructors and destructors and related methods

    /*!
    \brief Standard Constructor

    \param id    (in): A globally unique element id
    \param owner (in): owner processor of the element
    */
    Element(int id, int owner);

    /*!
    \brief Copy Constructor

    Makes a deep copy of a Element

    */
    Element(const Element& old);

    Element& operator=(const Element&) = default;


    /*!
    \brief Deep copy the derived class and return pointer to it

    This method is sort of a copy constructor for a class derived from Element.
    It allows to copy construct the derived class without knowing what it
    actually is using the base class Element.

    */
    virtual Element* clone() const = 0;


    /*!
    \brief Return unique ParObject id

    Every class implementing ParObject needs a unique id defined at the
    top of parobject.H
    */
    int unique_par_object_id() const override = 0;

    /*!
    \brief Pack this class so it can be communicated

    \ref pack and \ref unpack are used to communicate this element

    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
    \brief Unpack data from a char vector into this class

    \ref pack and \ref unpack are used to communicate this element

    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    /// return ElementType instance
    virtual Core::Elements::ElementType& element_type() const = 0;

    //@}

    //! @name Query methods

    /*!
    \brief Return global id of this element
    */
    int id() const { return id_; }

    /*!
    \brief Return processor local col map id
    */
    virtual int lid() const { return lid_; }

    /*!
    \brief Return owner of this element
    */
    virtual int owner() const { return owner_; }

    /*!
    \brief Get shape type of element
    */
    virtual Core::FE::CellType shape() const = 0;

    /*!
    \brief Return number of nodes to this element
    */
    virtual int num_node() const { return nodeid_.size(); }

    /*!
     * \brief Return number of points forming the geometry of this element
     *
     * For finite elements, this is equal to NumNodes(). Needed for meshfree
     * discretization where the integration cell (i.e. the "element") is not
     * formed by the nodes and many more nodes can have influence.
     */
    virtual int num_point() const { return nodeid_.size(); }

    /*!
    \brief Return number of lines to this element
    */
    virtual int num_line() const { return 0; }

    /*!
    \brief Return number of surfaces to this element
    */
    virtual int num_surface() const { return 0; }

    /*!
    \brief Return number of volumes to this element
    */
    virtual int num_volume() const { return 0; }

    /*!
     * \brief Return the number of faces to the element.
     *
     * As opposed to NumLine(), NumSurface(), etc which have fixed dimension,
     * this is always the object one dimension less than the element dimension.
     */
    virtual int num_face() const;

    /*!
    \brief Return id's of nodes adjacent to this element
    */
    virtual const int* node_ids() const
    {
      if (nodeid_.size())
        return nodeid_.data();
      else
        return nullptr;
    }

    /*!
    \brief Get vector of ptrs to nodes

    \warning The pointers to the nodes are build in
             Core::FE::Discretization::fill_complete. A standalone
             element that has not been added to a discretization
             (or the discretization has not been called fill_complete)
             does not have pointers to nodes. In this case, the method returns
             nullptr.
    \return Ptr to pointers to nodes of the element in local nodal ordering.
            Returns nullptr if pointers to not exist.
    */
    virtual Core::Nodes::Node** nodes()
    {
      if (node_.size())
        return node_.data();
      else
        return nullptr;
    }

    // mgee: change to this (as const and !const):
    // virtual std::vector<Core::Nodes::Node*>& Nodes()
    //{ return node_; }

    /*!
    \brief Get const vector of ptrs to nodes

    \warning The pointers to the nodes are build in
             Core::FE::Discretization::fill_complete. A standalone
             element that has not been added to a discretization
             (or the discretization has not been called fill_complete)
             does not have pointers to nodes. In this case, the method returns
             nullptr.
    \return Ptr to pointers to nodes of the element in local nodal ordering.
            Returns nullptr if pointers to not exist.
    */
    virtual const Core::Nodes::Node* const* nodes() const
    {
      if (node_.size())
        return (const Core::Nodes::Node* const*)(node_.data());
      else
        return nullptr;
    }

    /*!
     * \brief Return id's of points forming the geometry of this element
     *
     * For finite elements, this is equal to NodeIds(). Needed for meshfree
     * discretization where the integration cell (i.e. the "element") is not
     * formed by the nodes and many more nodes can have influence.
     */
    virtual const int* point_ids() const { return node_ids(); }

    /*!
     * \brief Get vector of ptrs to points forming the geometry of this element
     *
     * For finite elements, this is equal to Nodes(). Needed for meshfree
     * discretization where the integration cell (i.e. the "element") is not
     * formed by the nodes and many more nodes can have influence.
     */
    virtual Core::Nodes::Node** points() { return nodes(); }

    /*!
     * \brief Get const vector of ptrs to points forming the geometry of this element
     *
     * For finite elements, this is equal to Nodes() const. Needed for meshfree
     * discretization where the integration cell (i.e. the "element") is not
     * formed by the nodes and many more nodes can have influence.
     */
    virtual const Core::Nodes::Node* const* points() const { return nodes(); }


    /*!
    \brief Get vector of std::shared_ptrs to the lines of this element

    This is a base class dummy routine that always returns nullptr.
    The derived element class is expected to allocate and store
    a vector of elements that represent the edges of this element.
    These edges are then used to create and evaluate boundary conditions.

    \note A 1D type of element (e.g. a beam) may return
         a vector of length one pointing to itself. It then does not need
         to be able to spin of an explicit separate line element.
         In this case, the element itself must be able to evaluate the
         Neumann boundary conditions on a line itself.

    \note Do not store line elements inside parent elements since the nodes
might become invalid after a redistribution of the discretization.
    */
    virtual std::vector<std::shared_ptr<Element>> lines()
    {
      return std::vector<std::shared_ptr<Element>>(0);
    }

    // virtual const Element*const* Lines() const { FOUR_C_THROW("unexpected base method called.");
    // return nullptr; }

    /*!
    \brief Get vector of std::shared_ptrs to the surfaces of this element

    This is a base class dummy routine that always returns nullptr.
    The derived element class is expected to allocate and store
    a vector of elements that represent the surfaces of this element.
    These surfaces are then used to create and evaluate boundary conditions.

    \note A 2D type of element (e.g. a shell, wall, fluid2 etc) may return
         a vector of length one pointing to itself. It then does not need
         to be able to spin of an explicit separate surface element.
         In this case, the element itself must be able to evaluate the
         Neumann boundary conditions on a surface itself.

    \note Do not store surface elements inside parent elements since the nodes
might become invalid after a redistribution of the discretization.
    */
    virtual std::vector<std::shared_ptr<Element>> surfaces()
    {
      return std::vector<std::shared_ptr<Element>>(0);
    }

    /*!
    \brief Returns whether the given element actually is a face element with degrees of freedom
    living there
    */
    virtual bool is_face_element() const { return false; }

    /*!
    \brief Get vector of std::shared_ptrs to the faces of this element (as opposed to the Lines or
    Surfaces)
    */
    std::shared_ptr<FaceElement>* faces() { return face_.empty() ? nullptr : face_.data(); }

    /*!
    \brief Get vector of std::shared_ptrs to the faces of this element (as opposed to the Lines or
    Surfaces)
    */
    std::shared_ptr<FaceElement> const* faces() const
    {
      return face_.empty() ? nullptr : face_.data();
    }

    /*!
    \brief Get a pointer to the neighboring element behind the given face. Returns 0 if at boundary
    or faces are not created
    */
    Element* neighbor(const int face) const;

    /*!
    \brief Construct a face element between this element and the given slave element

    This is a base class routine that constructs a plain element without association to physics,
    containing only the geometry and topology information. A surface/line element is created
    that represents the face between this element (master element) and parent_slave element (in case
    it exists). The derived element class is expected to provide physics information when allocate
    and store an surface/line element that represents the face between this element (master element)
    and parent_slave element.

    This element is e.g. used to create and evaluate edge stabilizations or for HDG discretizations
    */
    virtual std::shared_ptr<Element> create_face_element(
        Element* parent_slave,                 //!< parent slave element
        int nnode,                             //!< number of nodes
        const int* nodeids,                    //!< node ids
        Core::Nodes::Node** nodes,             //!< nodeids
        const int lsurface_master,             //!< local index of surface w.r.t master element
        const int lsurface_slave,              //!< local index of surface w.r.t slave element
        const std::vector<int>& localtrafomap  //!< local trafo map
    )
    {
      std::shared_ptr<Element> face;
      return face;
    }


    /*!
    \brief Get nodal connectivity and weights for nodes

    The method is used to build the connectivity between all the nodes adjacent
    to this element and how expensive its evaluation is

    \param edgeweights (out): A Core::LinAlg::SerialDenseMatrix containing weights of all connected
    nodes \param nodeweights (out): A Core::LinAlg::SerialDenseVector containing weights of all
    nodes
    */
    virtual void nodal_connectivity(
        Core::LinAlg::SerialDenseMatrix& edgeweights, Core::LinAlg::SerialDenseVector& nodeweights);

    /*!
    \brief Return value how expensive it is to evaluate this element

    \param double (out): cost to evaluate this element
    */
    virtual double evaluation_cost() { return 10.0; }

    /*!
    \brief Get number of degrees of freedom of a certain node

    The element decides how many degrees of freedom its nodes must have.
    As this may vary along a simulation, the element can redecide the
    number of degrees of freedom per node along the way for each of it's nodes
    separately.<br>
    This method is used by the Core::FE::Discretization to determine how many
    degrees of freedom should be assigned to each node. The discretization will
    assign a node the max number of dofs the adjacent elements demand by this
    method.

    */
    virtual int num_dof_per_node(const Core::Nodes::Node& node) const
    {
      FOUR_C_THROW("not implemented");
      return -1;
    }

    /*!
    \brief Get number of degrees of freedom per face

    The element decides how many element degrees of freedom it has.
    It can redecide along the way of a simulation.

    Usually, a standard finite element would not have face degrees of
    freedom and would therefore return zero here.

    \note Face degrees of freedom mentioned here are dofs that are supposed
          to be visible at the global system level. Purely internal
          element dofs that are condensed internally should NOT be considered.
          The Core::FE::Discretization will use this method to determine how many degrees
          of freedom it should include in the degree of freedom row and column map
          for the global system of equations.
    */
    virtual int num_dof_per_face(const unsigned /* face */) const { return 0; }

    /*!
    \brief Get number of degrees of freedom per component

    This is especially important for face elements in HDG discretizations: Since
    the trace field can be in terms of a scalar quantity, e.g. the pressure, or in terms
    of a vector quantity, e.g. the velocity, it is important to know, which case actually
    occurs. With this function, the number of scalar dofs may be asked. Standard elements
    return the number of nodes, where the number of dofs per node specifies the number of
    components. HDG face elements overwrite this function and return the number of scalar
    dofs.
    */
    virtual int num_dof_per_component(const unsigned /* face */) const { return num_node(); }

    /*!
    \brief Get the degree of an element

    The element can have variable degree, in case of HDG discretizations.
    HDG elements implement this functions, standard elements return the degree
    depending on their discretization type and assuming isoparametric concept.
    */
    virtual int degree() const;


    /*!
    \brief Get number of degrees of freedom per element

    The element decides how many element degrees of freedom it has.
    It can redecide along the way of a simulation.<br>
    Usually, a standard finite element would not have element degrees of
    freedom and would therefore return zero here.

    \note Element degrees of freedom mentioned here are dofs that are supposed
          to be visible at the global system level. Purely internal
          element dofs that are condensed internally should NOT be considered.
          The Core::FE::Discretization will use this method to determine how many degrees
          of freedom it should include in the degree of freedom row and column map
          for the global system of equations.
    */
    virtual int num_dof_per_element() const
    {
      FOUR_C_THROW("not implemented");
      return -1;
    }

    /*!
    \brief Print this element

    Prints basic information about this element to ostream. This method would
    usually be called by the print method of a derived class.
    */
    virtual void print(std::ostream& os) const;

    /*!
    \brief Return the material of this element

    Note: The input parameter nummat is not the material number from input file
          as in set_material(int matnum), but the number of the material within
          the vector of materials the element holds

    \param nummat (in): number of requested material
    */
    virtual std::shared_ptr<Core::Mat::Material> material(int nummat = 0) const
    {
      FOUR_C_ASSERT(nummat < (int)mat_.size(), "invalid material number");
      return mat_[nummat];
    }

    /*!
    \brief Check whether the element has only ghost nodes

    Note: If the element has only ghost nodes, the element will not be allowed
          to assemble in any global vectors or matrixes, and, hence, can be
          skipped during evaluate(). The only reason why it is ghosted on this
          proc is to provide access to its data. This might be necessary for volumetric
          coupling of non conforming meshes.

    \param mypid (in): id of calling processor
    */
    virtual bool has_only_ghost_nodes(const int mypid) const;

    /*!
    \brief Query names of element data to be visualized using BINIO

    This method is to be overloaded by a derived class.
    The element is supposed to fill the provided map with key names of
    visualization data the element wants to visualize AT THE CENTER
    of the element geometry. The values is supposed to be dimension of the
    data to be visualized. It can either be 1 (scalar), 3 (vector), 6 (sym. tensor)
    or 9 (nonsym. tensor)

    Example:
    \code
      // Name of data is 'StressesXYZ', dimension is 6 (sym. tensor value)
      names.insert(std::pair<string,int>("StressesXYZ",6));
    \endcode

    \param names (out): On return, the derived class has filled names with
                        key names of data it wants to visualize and with int dimensions
                        of that data.
    */
    virtual void vis_names(std::map<std::string, int>& names) { return; }

    /*!
    \brief Visuzalize the owner of the element using BINIO

    \param names (out): Owner is added to the key names
    */
    virtual void vis_owner(std::map<std::string, int>& names)
    {
      names.insert(std::pair<std::string, int>("Owner", 1));
      // names.insert(std::pair<string,int>("EleGId",1));
      return;
    }

    /*!
    \brief Query data to be visualized using BINIO of a given name

    This method is to be overloaded by a derived method.
    The derived method is supposed to call this base method to visualize the owner of
    the element.
    If the derived method recognizes a supported data name, it shall fill it
    with corresponding data.
    If it does NOT recognizes the name, it shall do nothing.

    \warning The method must not change size of variable data

    \param name (in):   Name of data that is currently processed for visualization
    \param data (out):  data to be filled by element if it recognizes the name
    */
    virtual bool vis_data(const std::string& name, std::vector<double>& data)
    {
      if (name == "Owner")
      {
        if ((int)data.size() < 1) FOUR_C_THROW("Size mismatch");
        data[0] = owner();
        return true;
      }
      if (name == "EleGId")
      {
        if ((int)data.size() < 1) FOUR_C_THROW("Size mismatch");
        data[0] = id();
        return true;
      }
      return false;
    }

    //@}

    //! @name Construction

    /*!
    \brief Set global id of this element
    */
    void set_id(const int id) { id_ = id; }

    /*!
    \brief Read input for this element
    */
    virtual bool read_element(const std::string& eletype, const std::string& distype,
        const Core::IO::InputParameterContainer& container);

    /*!
      \brief Set processor local col id

      \param lid: processor local col id
     */
    void set_lid(int lid) { lid_ = lid; }

    /*!
    \brief Set ownership

    This method is used by the Discretization to change the ownership of
    an element that got communicated from one processor to another.

    \param owner: Proc owning this node

    \warning You should be very careful with changing the ownership
             of an element by hand as this might significantly confuse
             the Core::FE::Discretization the element is stored in.

    */
    void set_owner(const int owner) { owner_ = owner; }

    /*!
    \brief Set a list of node ids this element is connected to

    Sets the nodal ids of the nodes adjacent to this element and the number
    of nodes. This method is used in the construction phase of a discretization.
    It allows, that elements and nodes are created separately and be combined later
    on the way.

    \param nnode : number of nodes
    \param nodes : list of unique global nodal ids

    */
    void set_node_ids(const int nnode, const int* nodes);

    /**
     * Set the list of node ids this element is connected to. Here, the index
     * starts at 1, not 0. This is used for the legacy element input.
     */
    void set_node_ids_one_based_index(std::span<const int> node_ids);

    /*!
    \brief Set a the face with index faceindex this element is connected to

    Sets the face pointer of the face adjacent to this element, using NumFace() as
    number of faces. This method is used in the construction phase of a discretization.
    It allows, that elements and faces are created separately and be combined later
    on the way.

    Faces, that are set via this method are created as non-owning (weak) RCPs in
    the std::vector face_.

    \param faceindex   : index of the given face
    \param faceelement : face object
    */
    void set_face(const int faceindex, FaceElement* faceelement);

    /*!
    \brief Set a the face with index faceindex this element is connected to

    Sets the face pointer of the face adjacent to this element, using NumFace() as
    number of faces.

    Faces, that are set via this method copy the incoming RCP so strong and weak
    RCPs in the std::vector face_ can be created.

    \param faceindex   : index of the given face
    \param faceelement : face object
    */
    void set_face(const int faceindex, std::shared_ptr<FaceElement> faceelement);

    /// @brief Set specific element material
    /*!
      Store Material object in element's list of materials at given index.

      @note This method enables using the exact same Material instance in multiple elements.

      @param index index in material list
      @param mat pointer to the Material instance to set
     */
    virtual void set_material(const int index, std::shared_ptr<Core::Mat::Material> mat);

    /// Add element material
    /*!
      In case of volume coupled problems the element needs information
      from the material of the other field. Therefore, it is possible to
      add pointers on other materials from other elements.

      \param mat: material to be added
      \param nummat (out):  number of materials the element holds
     */
    int add_material(std::shared_ptr<Core::Mat::Material> mat);

    /// Number of materials of the element
    /*!
      In case of volume coupled problems the element needs information
      from the material of the other field. Therefore, it is possible to
      add pointers on other materials from other elements.

      \param nummat (out):  number of materials the element holds
     */
    int num_material() const { return mat_.size(); };

    //@}

    //! @name Evaluation methods

    /*!
    \brief Return the location vector of this element

    Extended version that features a nodal dof set (nds) array. Each node might
    have multiple sets of (physical) dofs. That is the actual number of dofs at
    a node represent a (small) number of independent physical dofs. This is not
    to be confused with multiple Core::DOFSets::DofSet objects within one
    Core::FE::Discretization.

    The case of multiple set of dofs occurs in xfem without enrichments. E.g. a
    node of a cut fluid element might own two or more velocity-pressure
    pairs. In that case we need to choose one set of physical dofs for each
    node. The nds array gives the dof set number of each node.

    \note The degrees of freedom returned are not necessarily only nodal dofs.
          Depending on the element implementation, output might also include
          element dofs.

    \param dis (in)      : the discretization this element belongs to
    \param nds           : dof set number of each node
    \param la (out)      : location data for all dofsets of the discretization

    */
    virtual void location_vector(
        const Core::FE::Discretization& dis, const std::vector<int>& nds, LocationArray& la) const;

    /*!
    \brief Return the location vector of this element

    The method computes degrees of freedom this element addresses.
    Degree of freedom ordering is as follows:<br>
    First all degrees of freedom of adjacent nodes are numbered in
    local nodal order, then the element internal degrees of freedom are
    given if present.<br>
    If a derived element has to use a different ordering scheme,
    it is welcome to overload this method as the assembly routines actually
    don't care as long as matrices and vectors evaluated by the element
    match the ordering, which is implicitly assumed.<br>
    Length of the output vector matches number of degrees of freedom
    exactly.<br>

    \note The degrees of freedom returned are not necessarily only nodal dofs.
          Depending on the element implementation, output might also include
          element dofs.

    \param dis (in)      : the discretization this element belongs to
    \param la (out)      : location data for all dofsets of the discretization

    */
    virtual void location_vector(const Core::FE::Discretization& dis, LocationArray& la) const;


    /*!
    \brief Return the location vector of this element

    The method computes degrees of freedom this element addresses.
    Degree of freedom ordering is as follows:<br>
    First all degrees of freedom of adjacent nodes are numbered in
    local nodal order, then the element internal degrees of freedom are
    given if present.<br>
    If a derived element has to use a different ordering scheme,
    it is welcome to overload this method as the assembly routines actually
    don't care as long as matrices and vectors evaluated by the element
    match the ordering, which is implicitly assumed.<br>
    Length of the output vector matches number of degrees of freedom
    exactly.<br>
    This version is intended to fill the LocationArray with the dofs
    the element will assemble into. In the standard case these dofs are
    the dofs of the element itself. For some special conditions (e.g.
    the weak dirichlet boundary condition) a surface element will assemble
    into the dofs of a volume element. These elements need to overwrite this
    method.<br>

    \note The degrees of freedom returned are not necessarily only nodal dofs.
          Depending on the element implementation, output might also include
          element dofs.

    \param dis (in)      : the discretization this element belongs to
    \param la (out)      : location data for all dofsets of the discretization
    \param condstring (in): Name of condition to be evaluated
    \param condstring (in):  List of parameters for use at element level
    */
    virtual void location_vector(const Core::FE::Discretization& dis, LocationArray& la,
        const std::string& condstring, Teuchos::ParameterList& params) const;

    /*!
    \brief Return the location vector of this element

    The method computes degrees of freedom this element addresses.
    Degree of freedom ordering is as follows:<br>
    First all degrees of freedom of adjacent nodes are numbered in
    local nodal order, then the element internal degrees of freedom are
    given if present.<br>
    If a derived element has to use a different ordering scheme,
    it is welcome to overload this method as the assembly routines actually
    don't care as long as matrices and vectors evaluated by the element
    match the ordering, which is implicitly assumed.<br>
    Length of the output vector matches number of degrees of freedom
    exactly.<br>

    \note The degrees of freedom returned are not necessarily only nodal dofs.
          Depending on the element implementation, output might also include
          element dofs.

    \param dis (in)      : the discretization this element belongs to
    \param lm (out)      : vector of degrees of freedom addressed by this element
    \param lmowner (out) : vector of proc numbers indicating which dofs are owned
                           by which procs in a dof row map. Ordering
                           matches dofs in lm.

    */
    virtual void location_vector(const Core::FE::Discretization& dis, std::vector<int>& lm,
        std::vector<int>& lmowner, std::vector<int>& lmstride) const;

    /*!
    \brief Evaluate an element

    An element derived from this class uses the Evaluate method to receive commands
    and parameters from some control routine in params and evaluates element matrices and
    vectors according to the command in params.

    \note This class implements a dummy of this method that prints a warning and
          returns false.

    \param params (in/out)    : ParameterList for communication between control routine
                                and elements
    \param discretization (in): A reference to the underlying discretization
    \param la (in)            : location data for all dofsets of the discretization
    \param elemat1 (out)      : matrix to be filled by element depending on commands
                                given in params
    \param elemat2 (out)      : matrix to be filled by element depending on commands
                                given in params
    \param elevec1 (out)      : vector to be filled by element depending on commands
                                given in params
    \param elevec2 (out)      : vector to be filled by element depending on commands
                                given in params
    \param elevec3 (out)      : vector to be filled by element depending on commands
                                given in params
    \return 0 if successful, negative otherwise
    */
    virtual int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3);

    /*!
    \brief Evaluate an element

    An element derived from this class uses the Evaluate method to receive commands
    and parameters from some control routine in params and evaluates element matrices and
    vectors according to the command in params.

    \note This class implements a dummy of this method that prints a warning and
          returns false.

    \param params (in/out)    : ParameterList for communication between control routine
                                and elements
    \param discretization (in): A reference to the underlying discretization
    \param lm (in)            : location vector of this element
    \param elemat1 (out)      : matrix to be filled by element depending on commands
                                given in params
    \param elemat2 (out)      : matrix to be filled by element depending on commands
                                given in params
    \param elevec1 (out)      : vector to be filled by element depending on commands
                                given in params
    \param elevec2 (out)      : vector to be filled by element depending on commands
                                given in params
    \param elevec3 (out)      : vector to be filled by element depending on commands
                                given in params
    \return 0 if successful, negative otherwise
    */
    virtual int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3);

    /*!
    \brief Evaluate a Neumann boundary condition

    An element derived from this class uses the evaluate_neumann method to receive commands
    and parameters from some control routine in params and evaluates a Neumann boundary condition
    given in condition

    \note This class implements a dummy of this method that prints a warning and
          returns false.

    \param params (in/out)    : ParameterList for communication between control routine
                                and elements
    \param discretization (in): A reference to the underlying discretization
    \param condition (in)     : The condition to be evaluated
    \param lm (in)            : location vector of this element
    \param elevec1 (out)      : Force vector to be filled by element

    \return 0 if successful, negative otherwise
    */
    virtual int evaluate_neumann(Teuchos::ParameterList& params,
        Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
        std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) = 0;


    //@}


    //! @name Public methods to be used by Core::FE::Discretization only

    /*!
    \brief Build pointer vector from map of nodes

    \warning (public, but to be used by Core::FE::Discretization ONLY!)

    The method is used to build the variable node_ in this element. It is called from
    Core::FE::Discretization in Core::FE::Discretization::fill_complete() to
    create the pointers from elements to nodes (and nodes to elements)

    \param nodes (in): A map of all nodes of a discretization
    */
    virtual bool build_nodal_pointers(std::map<int, std::shared_ptr<Core::Nodes::Node>>& nodes);

    /*!
    \brief Build pointer vector from vector of nodal pointers

    \warning (public, but to be used by Core::FE::Discretization ONLY!)

    The method is used to build the variable node_ in this element. It can be used
    to explicitly pass the nodal pointers to the element.

    \param nodes (in): Pointer to array of pointers to nodes. The array of pointers
                       is implicitly expected to be of length num_node() and contain pointers
                       to nodes in the correct element local ordering scheme.
    */
    virtual bool build_nodal_pointers(Core::Nodes::Node** nodes);

    /*!
    \brief Build pointer vector from map of elements

    \warning (public, but to be used by Core::FE::Discretization ONLY!)

    The method is used to build the element connectivity in this element. For standard elements this
    procedure returns true and does nothing. For interface-elements a connection is made between the
    interface and it's left and right element. It is called from Core::FE::Discretization
    in Core::FE::Discretization::fill_complete() to create the pointers from elements to
    elements (next to the node-element and nodes-elements connectivity).

    \param elements (in): A map of all elements of a discretization
    */
    virtual bool build_element_pointers(std::map<int, std::shared_ptr<Element>>& elements)
    {
      return true;
    }

    //@}

    /// @name Parameter interface access and management functions
    /// @{

    /*!
    \brief set the interface ptr for the elements

    \param p (in): Parameter list coming from the time integrator.
     */
    virtual void set_params_interface_ptr(const Teuchos::ParameterList& p) {
      /* This is a dummy function. Please implement the function in the derived classes, if
         necessary. */
    };

    /*!
    \brief returns true if the interface is defined and initialized, otherwise false
    */
    virtual inline bool is_params_interface() const
    {
      // dummy implementation
      return false;
    }

    /*!
    \brief get access to the interface pointer
    */
    virtual std::shared_ptr<Core::Elements::ParamsInterface> params_interface_ptr()
    {
      FOUR_C_THROW(
          "This is a dummy function. Please implement the function in the derived classes, if "
          "necessary.");
      return nullptr;
    }

    /**
     * \brief Add the element geometry visualization to the vtu data vectors
     *
     * For standard Lagrangian elements, each element node is added as a vtu point. A vtu cell
     * corresponding to the element type connects the added nodes and represents the element in
     * the output. Note that each cell adds all of its geometry information independently of other
     * cells. This implies that nodes shared by multiple elements are duplicated in the output.
     *
     * @param discret (in) discretization
     * @param cell_types (in/out) cell type data vector
     * @param point_coordinates (in/out) point coordinates for the representation of this element
     * @return Number of added points
     */
    virtual unsigned int append_visualization_geometry(const Core::FE::Discretization& discret,
        std::vector<uint8_t>& cell_types, std::vector<double>& point_coordinates) const;

    /**
     * \brief Add dof based results to point data vector, such that the data matches the geometry
     * created in append_visualization_geometry.
     *
     * @param discret (in) discretization
     * @param result_data_dofbased (in) Global vector with results
     * @param result_num_dofs_per_node (in) Number of scalar values per point.
     * @param read_result_data_from_dofindex (in) Starting DOF index for the nodal DOFs. This is
     * used if not all nodal DOFs should be output, e.g., velocity or pressure in fluid.
     * @param vtu_point_result_data (in/out) Result data vector.
     * @return Number of points added by this element.
     */
    virtual unsigned int append_visualization_dof_based_result_data_vector(
        const Core::FE::Discretization& discret,
        const Core::LinAlg::Vector<double>& result_data_dofbased,
        unsigned int result_num_dofs_per_node, unsigned int read_result_data_from_dofindex,
        std::vector<double>& vtu_point_result_data) const;

    /*!
     * \brief Add node based results to point data vector, such that the data matches the geometry
     * created in append_visualization_geometry.
     *
     * @param discret (in) discretization
     * @param result_data_nodebased (in) Global node-based vector with results.
     * @param result_num_components_per_node (in) Number of scalar values per point.
     * @param point_result_data (in/out) Result data vector.
     * @return Number of points added by this element.
     */
    virtual unsigned int append_visualization_node_based_result_data_vector(
        const Core::FE::Discretization& discret,
        const Core::LinAlg::MultiVector<double>& result_data_nodebased,
        int result_num_components_per_node, std::vector<double>& point_result_data) const;

    /**
     * \brief Add the current position of all nodes of the element to a boundary volume.
     */
    virtual Core::GeometricSearch::BoundingVolume get_bounding_volume(
        const Core::FE::Discretization& discret,
        const Core::LinAlg::Vector<double>& result_data_dofbased,
        const Core::GeometricSearch::GeometricSearchParams& params) const;
    /// @}

   private:
    //! \brief A unique global element id
    int id_;

    //! local col map id
    int lid_;

    //! \brief owner processor of this element
    int owner_;

    //! \brief List of my nodal ids, length num_node()
    std::vector<int> nodeid_;

    //! \brief Pointers to adjacent nodes in element local ordering
    std::vector<Core::Nodes::Node*> node_;

    //! \brief List of my faces, length NumFace(). Only filled if face elements are created, when
    //! using DiscretizationFaces
    std::vector<std::shared_ptr<FaceElement>> face_;

    //! vector of material objects of element
    std::vector<std::shared_ptr<Core::Mat::Material>> mat_;
  };  // class Element


  /*!
  \brief A virtual class all face elements used in the discretization management module have to
  implement

  This is the base class for all face elements. Every element that is attached to another
  (parent) element needs to implement this class, in particular boundary elements used for
  boundary conditions.

  A face element can be attached to one (for boundary faces) or two
  elements (for interior faces). In the latter case, we distinguish between the so-called
  parent master element and the parent slave element. The parent master element is
  the element that shares the node numbering with the face element. Boundary elements
  have only one parent and the methods parent_element() and parent_master_element()
  return the same element. The parent slave element is the element on the other
  side that usually has a different orientation of nodes.

  */

  class FaceElement : public Element
  {
   public:
    /*!
    \brief Standard Constructor

    \param id    (in): A globally unique element id
    \param owner (in): owner processor of the element
    */
    FaceElement(int id, int owner);

    /*!
    \brief Copy Constructor
    */
    FaceElement(const FaceElement& old);

    /*!
    \brief Pack this class so it can be communicated

    \ref pack and \ref unpack are used to communicate this face element

    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
    \brief Unpack data from a char vector into this class

    \ref pack and \ref unpack are used to communicate this face element

    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    /*!
    \brief Returns whether the given element actually is a face element with degrees of freedom
    living there
    */
    bool is_face_element() const override { return true; }

    /*!
    \brief Return the parent element id the face element is connected to (for interior faces, the
    parent master element).

    If no parent master has been assigned, -1 is returned (e.g. on non-face discretizations)

    This Id is also available, if the calling processor is not owner of the parent_element()!
    */
    int parent_element_id() const { return parent_id_; }

    /*!
    \brief Return the parent element the face element is connected to (for interior faces, the
    parent master element).

    If no parent master has been assigned, nullptr is returned (e.g. on non-face discretizations)
    */
    virtual Element* parent_element() const { return parent_master_; }

    /*!
    \brief Return the master element the face element is connected to

    If no parent master has been assigned, nullptr is returned (e.g. on non-face discretizations)
    */
    Element* parent_master_element() const { return parent_master_; }

    /*!
    \brief Return the slave element the face element is connected to

    If no parent slave has been assigned, nullptr is returned (non-face discretizations, boundary
    faces)
    */
    Element* parent_slave_element() const { return parent_slave_; }

    /*!
     \brief Get the index of a face element within the parent element
     */
    int face_parent_number() const
    {
      FOUR_C_ASSERT(lface_master_ != -1,
          "Face information has not been filled or this is not a face element");
      return lface_master_;
    }

    /*!
     \brief Get the index of a face element within the parent master element
     */
    int face_master_number() const
    {
      FOUR_C_ASSERT(lface_master_ != -1,
          "Face information has not been filled or this is not a face element");
      return lface_master_;
    }

    /*!
     \brief Get the index of a face element within the parent slave element
     */
    int face_slave_number() const
    {
      FOUR_C_ASSERT(lface_slave_ != -1,
          "Face information has not been filled or this is not a face element with slave parent");
      return lface_slave_;
    }

    /*!
    \brief return a transformation for the face's nodes between the local coordinate systems of the
           face w.r.t the master parent element's face's coordinate system and the slave element's
           face's coordinate system

    Only filled for interior faces, otherwise zero length-vector (empty).
    */
    const std::vector<int>& get_local_trafo_map() const { return localtrafomap_; }

    /*!
    \brief Set the master element this face element is connected to

    Sets the pointers of the main parent element adjacent to the current face.
    This method is used in the construction phase of a discretization.
    It allows that elements and faces are created separately and be combined later
    on the way.

    \param master: parent element which shares the orientation of faces
    \param lface_master: local number of face within master element
    */
    void set_parent_master_element(Element* master, const int lface_master)
    {
      parent_master_ = master;
      lface_master_ = lface_master;
      if (master != nullptr) parent_id_ = master->id();
    }

    /*!
    \brief Set the slave element this face element is connected to

    Sets the pointers of the second parent element adjacent to the current face.
    This method is used in the construction phase of a discretization.
    It allows that elements and faces are created separately and be combined later
    on the way.

    \param slave: second parent element when there already is a master element
    \param lface_slave: local number of face within slave element
    */
    void set_parent_slave_element(Element* slave, const int lface_slave)
    {
      parent_slave_ = slave;
      lface_slave_ = lface_slave;
    }


   protected:
    /*!
    \brief Set the slave element this face element is connected to

    Sets the pointers of the second parent element adjacent to the current face.
    This method is used in the construction phase of a discretization.
    It allows that elements and faces are created separately and be combined later
    on the way.

    \param trace: transformation between
    */
    void set_local_trafo_map(const std::vector<int>& trafo);

   private:
    //! \brief The master parent element of face element. nullptr for usual elements
    Element* parent_master_;

    //! \brief The master parent element of face element. nullptr for usual elements
    Element* parent_slave_;

    //! The local surface number of this surface w.r.t to the parent_master_ element
    int lface_master_;

    //! The local surface number of this surface w.r.t to the parent_slave_ element
    int lface_slave_;

    /*!
     \brief map for the face's nodes between the local coordinate systems of the face w.r.t the
     master parent element's face's coordinate system and the slave element's face's coordinate
     system
     */
    std::vector<int> localtrafomap_;

    //! The parent element id
    int parent_id_;

    //! Friend declaration
    friend class Core::FE::DiscretizationHDG;
  };

  /// translate shards::key to enum
  Core::FE::CellType shards_key_to_dis_type(const unsigned& key);
}  // namespace Core::Elements


// << operator
std::ostream& operator<<(std::ostream& os, const Core::Elements::Element& ele);



FOUR_C_NAMESPACE_CLOSE

#endif
