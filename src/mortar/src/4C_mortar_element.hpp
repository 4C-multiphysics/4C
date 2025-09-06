// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_ELEMENT_HPP
#define FOUR_C_MORTAR_ELEMENT_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mortar_node.hpp"
#include "4C_utils_pairedvector.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace Mortar
{
  class ElementNitscheContainer;
};

namespace Mortar
{
  /*!
  \brief A subclass of Core::Elements::ElementType that adds mortar element type specific methods

  */
  class ElementType : public Core::Elements::ElementType
  {
   public:
    std::string name() const override { return "Mortar::ElementType"; }

    static ElementType& instance();

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

   private:
    static ElementType instance_;
  };

  /*!
  \brief A class containing additional data for mortar elements

  This class contains additional information for mortar elements which is
  only needed for contact evaluation. Thus, in order to save memory, it is
  sufficient to have this data available only on the slave element column map.

  */
  class MortarEleDataContainer
  {
   public:
    //! @name Constructors and destructors and related methods

    /*!
    \brief Standard Constructor

    */
    MortarEleDataContainer();

    /*!
    \brief Destructor

    */
    virtual ~MortarEleDataContainer() = default;
    /*!
    \brief Pack this class so that it can be communicated

    This function packs the datacontainer. This is only called
    when the class has been initialized and the pointer to this
    class exists.

    */
    virtual void pack(Core::Communication::PackBuffer& data) const;

    /*!
    \brief Unpack data from a vector into this class

    This function unpacks the datacontainer. This is only called
    when the class has been initialized and the pointer to this
    class exists.

    */
    virtual void unpack(Core::Communication::UnpackBuffer& buffer);

    //@}

    //! @name Access methods

    /*!
    \brief Return current area

    */
    virtual double& area() { return area_; }
    double area() const { return area_; }

    /*!
    \brief Return number of potentially contacting elements

    */
    virtual int num_search_elements() const { return (int)searchelements_.size(); }

    /*!
    \brief Return global ids of potentially contacting elements

    */
    virtual std::vector<int>& search_elements() { return searchelements_; }
    const std::vector<int>& search_elements() const { return searchelements_; }

    /*!
    \brief Return matrix of dual shape function coefficients

    */
    virtual std::shared_ptr<Core::LinAlg::SerialDenseMatrix>& dual_shape()
    {
      return dualshapecoeff_;
    }
    std::shared_ptr<const Core::LinAlg::SerialDenseMatrix> dual_shape() const
    {
      return dualshapecoeff_;
    }

    /*!
    \brief Return trafo matrix for boundary modification

    */
    virtual std::shared_ptr<Core::LinAlg::SerialDenseMatrix>& trafo() { return trafocoeff_; }
    std::shared_ptr<const Core::LinAlg::SerialDenseMatrix> trafo() const { return trafocoeff_; }

    /*!
    \brief Return directional derivative of matrix of dual shape function coefficients

    */
    virtual std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>&
    deriv_dual_shape()
    {
      return derivdualshapecoeff_;
    }

    /*!
    \brief Return directional derivative of matrix of dual shape function coefficients
     (const version)

    */
    std::shared_ptr<const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>
    get_deriv_dual_shape() const
    {
      return derivdualshapecoeff_;
    }

    /*!
    \brief Reset matrix of dual shape function coefficients and free memory

    */
    virtual void reset_dual_shape()
    {
      dualshapecoeff_ = nullptr;
      return;
    }

    /*!
    \brief Reset directional derivative of matrix of dual shape function coefficients and free
    memory

    */
    virtual void reset_deriv_dual_shape()
    {
      derivdualshapecoeff_ = nullptr;
      return;
    }

    /*!
    \brief Return Parent displacement vector

    */
    virtual std::vector<double>& parent_disp() { return parentdisp_; }

    /*!
    \brief Return Parent velocity vector

    */
    virtual std::vector<double>& parent_vel() { return parentvel_; }

    /*!
    \brief Return Parent element degrees of freedom

    */
    virtual std::vector<int>& parent_dof() { return parentdofs_; }

    /*!
    \brief Return Parent scalar vector

    */
    virtual std::vector<double>& parent_scalar() { return parentscalar_; }

    /*!
    \brief Return Parent Scalar element degrees of freedom

    */
    virtual std::vector<int>& parent_scalar_dof() { return parentscalardofs_; }

    /*!
    \brief Return Parent temperature vector

    */
    virtual std::vector<double>& parent_temp() { return parenttemp_; }

    /*!
    \brief Return Parent element temperature degrees of freedom
           (To not have to use the thermo-discretization in contact,
            we use the first displacement dof)

    */
    virtual std::vector<int>& parent_temp_dof() { return parenttempdofs_; }

    /*!
    \brief Return Parent poro pressure vector

    */
    virtual std::vector<double>& parent_pf_pres() { return parentpfpres_; }

    /*!
    \brief Return Parent poro velocity vector

    */
    virtual std::vector<double>& parent_pf_vel() { return parentpfvel_; }

    /*!
    \brief Return Parent Poro Fluid element degrees of freedom

    */
    virtual std::vector<int>& parent_pf_dof() { return parentpfdofs_; }

    //@}

   protected:
    // don't want = operator and cctor
    MortarEleDataContainer operator=(const MortarEleDataContainer& old);
    MortarEleDataContainer(const MortarEleDataContainer& old);

    double area_;                      // element length/area in current configuration
    std::vector<int> searchelements_;  // global ids of potentially contacting elements

    // coefficient matrix for dual shape functions
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> dualshapecoeff_;

    // derivative of coefficient matrix for dual shape functions
    std::shared_ptr<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>
        derivdualshapecoeff_;

    // coefficient matrix for boundary trafo
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> trafocoeff_;

    // Displacement of Parent Element
    std::vector<double> parentdisp_;

    // Velocity of Parent Element
    std::vector<double> parentvel_;

    // Displacement Parent element degrees of freedom
    std::vector<int> parentdofs_;

    /// Scalar of Parent Element
    std::vector<double> parentscalar_;

    /// Scalar Parent element degrees of freedom
    std::vector<int> parentscalardofs_;

    // Temperature of Parent Element
    std::vector<double> parenttemp_;

    // Temperature Parent element degrees of freedom
    // (To not have to use the thermo-discretization in contact,
    //  we use the first displacement dof)
    std::vector<int> parenttempdofs_;

    // Poro Pressure of Parent Element
    std::vector<double> parentpfpres_;

    // Poro Vel of Parent Element
    std::vector<double> parentpfvel_;

    // Poro Fluid Parent element degrees of freedom
    std::vector<int> parentpfdofs_;

  };  // class MortarEleDataContainer


  /*!
  \brief A mortar coupling element

  */
  class Element : public Core::Elements::FaceElement
  {
   public:
    //! @name Enums and Friends

    /*!
    \brief Enum for shape function types recognized by Elements

    */
    enum ShapeType
    {
      p0,             // displacements / LM constant per element
      lin1D,          // displacements / std LM linear 1D
      quad1D,         // displacements / std LM quadratic 1D
      lin2D,          // displacements / std LM linear 2D
      bilin2D,        // displacements / std LM bilinear 2D
      quad2D,         // displacements / std LM quadratic 2D
      serendipity2D,  // displacements / std LM serendipity 2D
      biquad2D,       // displacements / std LM biquadratic 2D

      lindual1D,          // dual LM linear 1D
      quaddual1D,         // dual LM quadratic 1D
      lindual2D,          // dual LM linear 2D
      bilindual2D,        // dual LM bilinear 2D
      quaddual2D,         // dual LM quadratic 2D
      serendipitydual2D,  // dual LM serendipity 2D
      biquaddual2D,       // dual LM biquadratic 2D

      lin1D_edge0,            // crosspoint LM modification 1D
      lin1D_edge1,            // crosspoint LM modification 1D
      lindual1D_edge0,        // crosspoint LM modification 1D
      lindual1D_edge1,        // crosspoint LM modification 1D
      dual1D_base_for_edge0,  // crosspoint LM modification 1D
      dual1D_base_for_edge1,  // crosspoint LM modification 1D
      quad1D_edge0,           // crosspoint LM modification 1D
      quad1D_edge1,           // crosspoint LM modification 1D
      quaddual1D_edge0,       // crosspoint LM modification 1D
      quaddual1D_edge1,       // crosspoint LM modification 1D

      quad1D_only_lin,         // quad->lin standard LM modification 1D
      quad2D_only_lin,         // quad->lin standard LM modification 2D
      serendipity2D_only_lin,  // quad->lin standard LM modification 2D
      biquad2D_only_lin,       // quad->lin standard LM modification 2D

      quaddual1D_only_lin,         // quad->lin dual LM modification 1D (not yet impl.)
      quaddual2D_only_lin,         // quad->lin dual LM modification 2D
      serendipitydual2D_only_lin,  // quad->lin dual LM modification 2D
      biquaddual2D_only_lin,       // quad->lin dual LM modification 2D

      quad1D_modified,         // displacement modification for dual LM quadratic 1D (not yet impl.)
      quad2D_modified,         // displacement modification for dual LM quadratic 2D
      serendipity2D_modified,  // displacement modification for dual LM serendipity 2D
      biquad2D_modified,       // displacement modification for dual LM biquadratic 2D

      quad1D_hierarchical,  // displacement modification for quad->lin dual LM quadratic 1D (not yet
                            // impl.)
      quad2D_hierarchical,  // displacement modification for quad->lin dual LM quadratic 2D
      serendipity2D_hierarchical,  // displacement modification for quad->lin dual LM serendipity 2D
      biquad2D_hierarchical        // displacement modification for quad->lin dual LM biquadratic 2D
    };

    // physical type of mortar element
    // (Scoped enumeration: allows static_cast since C++11)
    //  enum struct PhysicalType : int
    enum PhysicalType
    {
      poro = 0,       // poroelastic: (porofluid exists and must be considered in contact/meshtying)
      structure = 1,  // structure
      other = 2       // this should not happen
    };

    //@}

    //! @name Constructors and destructors and related methods

    /*!
    \brief Standard Constructor

    \param id    (in): A globally unique element id
    \param etype (in): Type of element
    \param owner (in): owner processor of the element
    \param shape (in): shape of this element
    \param numnode (in): Number of nodes to this element
    \param nodeids (in): ids of nodes adjacent to this element
    \param isslave (in): flag indicating whether element is slave or master side
    \param isnurbs (in): flag indicating whether element is nurbs element or not
    */
    Element(int id, int owner, const Core::FE::CellType& shape, const int numnode,
        const int* nodeids, const bool isslave, bool isnurbs = false);

    /*!
    \brief Copy Constructor

    Makes a deep copy of this class

    */
    Element(const Mortar::Element& old);



    /*!
    \brief Deep copy the derived class and return pointer to it

    */
    Core::Elements::Element* clone() const override;

    /*!
    \brief Return unique ParObject id

    Every class implementing ParObject needs a unique id defined at the
    top of parobject.H

    */
    int unique_par_object_id() const override
    {
      return ElementType::instance().unique_par_object_id();
    }

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

    Core::Elements::ElementType& element_type() const override { return ElementType::instance(); }

    //@}

    //! @name Query methods

    /*!
    \brief Get shape type of element

    */
    Core::FE::CellType shape() const override { return shape_; }

    /*!
    \brief Return number of lines to this element

    */
    int num_line() const override { return 0; }

    /*!
    \brief Return number of surfaces to this element

    */
    int num_surface() const override { return 0; }

    /*!
    \brief Get vector of std::shared_ptrs to the lines of this element

    */
    std::vector<std::shared_ptr<Core::Elements::Element>> lines() override
    {
      std::vector<std::shared_ptr<Core::Elements::Element>> lines;
      return lines;
    }

    /*!
    \brief Get vector of std::shared_ptrs to the surfaces of this element

    */
    std::vector<std::shared_ptr<Core::Elements::Element>> surfaces() override
    {
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces;
      return surfaces;
    }

    /*!
    \brief Get number of degrees of freedom of a certain node

    This Mortar::Element is picky: It cooperates only with Mortar::Node(s),
    not with standard Node objects!

    */
    int num_dof_per_node(const Core::Nodes::Node& node) const override;

    /*!
    \brief Get number of degrees of freedom per element

    For now mortar coupling elements do not have degrees of freedom
    independent of the nodes.

    */
    int num_dof_per_element() const override { return 0; }

    /*!
    \brief Print this element

    */
    void print(std::ostream& os) const override;

    /*!
    \brief Return slave (true) or master status

    */
    virtual bool is_slave() const { return isslave_; }

    /*!
    \brief Return attached status

    */
    virtual bool is_attached() const { return attached_; }

    /*!
    \brief Change slave (true) or master status

    This changing of contact topology becomes necessary for self contact
    simulations, where slave and master status are assigned dynamically

    */
    virtual bool& set_slave() { return isslave_; }

    /*!
    \brief Set attached status

    */
    virtual bool& set_attached() { return attached_; }

    /*!
    \brief Return ansatz type (true = quadratic) of element

    */
    virtual bool is_quad() const
    {
      bool isquad = false;
      switch (shape())
      {
        case Core::FE::CellType::line2:
        case Core::FE::CellType::nurbs2:
        case Core::FE::CellType::tri3:
        case Core::FE::CellType::quad4:
        {
          // do nothing
          break;
        }
        case Core::FE::CellType::line3:
        case Core::FE::CellType::nurbs3:
        case Core::FE::CellType::quad8:
        case Core::FE::CellType::quad9:
        case Core::FE::CellType::nurbs9:
        case Core::FE::CellType::tri6:
        {
          isquad = true;
          break;
        }
        default:
          FOUR_C_THROW("Unknown mortar element type identifier");
      }
      return isquad;
    }

    /*!
    \brief Return spatial dimension

    */
    virtual int n_dim() const
    {
      switch (shape())
      {
        case Core::FE::CellType::line2:
        case Core::FE::CellType::nurbs2:
        case Core::FE::CellType::line3:
        case Core::FE::CellType::nurbs3:
        {
          // this is a 2-D problem
          return 2;
          break;
        }
        case Core::FE::CellType::tri3:
        case Core::FE::CellType::quad4:
        case Core::FE::CellType::quad8:
        case Core::FE::CellType::quad9:
        case Core::FE::CellType::nurbs9:
        case Core::FE::CellType::tri6:
        {
          // this is a 3-D problem
          return 3;
          break;
        }
        default:
        {
          FOUR_C_THROW("Unknown mortar element type identifier");
          return 0;
        }
      }
    }

    /*!
    \brief Return nurbs (true) or not-nurbs (false) status

    */
    virtual bool is_nurbs() const { return nurbs_; }

    /*!
    \brief Return data container of this element

    This method returns the data container of this mortar element where additional
    mortar specific quantities/information are stored.

    */
    inline Mortar::MortarEleDataContainer& mo_data()
    {
      FOUR_C_ASSERT(modata_ != nullptr, "Mortar data container not set");
      return *modata_;
    }

    inline const Mortar::MortarEleDataContainer& mo_data() const
    {
      FOUR_C_ASSERT(modata_ != nullptr, "Mortar data container not set");
      return *modata_;
    }

    //@}

    //! @name Evaluation methods

    /*!
    \brief Evaluate an element

    An element derived from this class uses the Evaluate method to receive commands
    and parameters from some control routine in params and evaluates element matrices and
    vectors according to the command in params.

    \note This class implements a dummy of this method that prints a FOUR_C_THROW and
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
    int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

    /*!
    \brief Evaluate a Neumann boundary condition dummy

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
    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override
    {
      return 0;
    }

    /*!
    \brief Get local coordinates for local node id
    */
    bool local_coordinates_of_node(int lid, double* xi) const;

    /*!
    \brief Get local numbering for global node id
    */
    int get_local_node_id(int nid) const;

    /*!
    \brief Build element normal at node passed in
    */
    virtual void build_normal_at_node(
        int nid, int& i, Core::LinAlg::SerialDenseMatrix& elens) const;

    /*!
    \brief Compute element normal at local coordinate xi
           Caution: This function cannot be called stand-alone! It is
           integrated into the whole nodal normal calculation process.
    */
    virtual void compute_normal_at_xi(
        const double* xi, int& i, Core::LinAlg::SerialDenseMatrix& elens) const;

    /*!
    \brief Compute averaged nodal normal at local coordinate xi
    */
    virtual double compute_averaged_unit_normal_at_xi(const double* xi, double* n) const;

    /*!
    \brief Compute unit element normal at local coordinate xi
           This function is a real stand-alone function to be called
           for a CElement in order to compute a unit normal at any point.
           Returns the length of the non-unit interpolated normal at xi.
    */
    virtual double compute_unit_normal_at_xi(const double* xi, double* n) const;

    /*!
    \brief Compute element unit normal derivative at local coordinate xi
           This function is a real stand-alone function to be called
           for a Element in order to compute a unit normal derivative at any point.
    */
    virtual void deriv_unit_normal_at_xi(
        const double* xi, std::vector<Core::Gen::Pairedvector<int, double>>& derivn) const;

    /*!
    \brief Get nodal reference / spatial coords of current element

    */
    virtual void get_nodal_coords(Core::LinAlg::SerialDenseMatrix& coord) const;

    /*! \brief Get nodal reference / spatial coords of current element
     *
     *  */
    template <unsigned elenumnode>
    inline void get_nodal_coords(Core::LinAlg::Matrix<3, elenumnode>& coord) const
    {
      Core::LinAlg::SerialDenseMatrix sdm_coord(Teuchos::View, coord.data(), 3, 3, elenumnode);
      get_nodal_coords(sdm_coord);
    }

    double inline get_nodal_coords(const int direction, const int node) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      const Node* mymrtrnode = dynamic_cast<const Node*>(points()[node]);
      if (!mymrtrnode) FOUR_C_THROW("GetNodalCoords: Null pointer!");
      return mymrtrnode->xspatial()[direction];
#else
      return static_cast<const Node*>(points()[node])->xspatial()[direction];
#endif
    }

    /*!
    \brief Get nodal spatial coords from previous time step of current element

    \param isinit (in): true if called for reference coords
    */
    virtual void get_nodal_coords_old(
        Core::LinAlg::SerialDenseMatrix& coord, bool isinit = false) const;

    double inline get_nodal_coords_old(const int direction, const int node) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      const Node* mymrtrnode = dynamic_cast<const Node*>(points()[node]);
      if (!mymrtrnode) FOUR_C_THROW("GetNodalCoords: Null pointer!");
      return mymrtrnode->x()[direction] + mymrtrnode->uold()[direction];
#else
      return (static_cast<const Node*>(points()[node])->x()[direction] +
              static_cast<const Node*>(points()[node])->uold()[direction]);
#endif
    }

    /*!
    \brief Get nodal spatial coords from previous time step of current element

    \param isinit (in): true if called for reference coords
    */
    virtual void get_nodal_lag_mult(
        Core::LinAlg::SerialDenseMatrix& lagmult, bool isinit = false) const;

    /*!
    \brief Evaluate element metrics (local basis vectors)
    */
    virtual void metrics(const double* xi, double* gxi, double* geta) const;

    /*!
    \brief Evaluate Jacobian determinant for parameter space integration
    */
    virtual double jacobian(const double* xi) const;

    /*!
    \brief Compute Jacobian determinant derivative
    */
    virtual void deriv_jacobian(
        const double* xi, Core::Gen::Pairedvector<int, double>& derivjac) const;

    /*!
    \brief Compute length/area of the element
    */
    virtual double compute_area() const;

    /*!
    \brief Compute length/area of the element and its derivative
    */
    virtual double compute_area_deriv(Core::Gen::Pairedvector<int, double>& area_deriv) const;

    /*!
    \brief A repository for all kinds of 1D/2D shape functions
    */
    virtual void shape_functions(Element::ShapeType shape, const double* xi,
        Core::LinAlg::SerialDenseVector& val, Core::LinAlg::SerialDenseMatrix& deriv) const;

    /*!
    \brief A repository for 1D/2D shape function linearizations

    \param derivdual (in): derivative maps to be filled
                           (= derivatives of the dual coefficient matrix Ae)
    Remark: this function cannot be marked const since it changes the underlying array for
        dual shape function derivatives calculation
    */
    void shape_function_linearizations(Element::ShapeType shape,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivdual);

    /*!
    \brief Evaluate displacement shape functions and derivatives
    */
    virtual bool evaluate_shape(const double* xi, Core::LinAlg::SerialDenseVector& val,
        Core::LinAlg::SerialDenseMatrix& deriv, const int valdim, bool dualquad3d = false) const;

    /*! \brief Evaluate displacement shape functions and derivatives
     *
     *  */
    template <unsigned elenumnode, unsigned eledim>
    inline bool evaluate_shape(const double* xi, Core::LinAlg::Matrix<elenumnode, 1>& val,
        Core::LinAlg::Matrix<elenumnode, eledim>& deriv, unsigned valdim = elenumnode,
        bool dualquad3d = false) const
    {
      Core::LinAlg::SerialDenseVector sdv_val(Teuchos::View, val.data(), elenumnode);
      Core::LinAlg::SerialDenseMatrix sdm_deriv(
          Teuchos::View, deriv.data(), elenumnode, elenumnode, eledim);
      return evaluate_shape(xi, sdv_val, sdm_deriv, valdim, dualquad3d);
    }

    /*!
    \brief Evaluate Lagrange multiplier shape functions and derivatives
    */
    virtual bool evaluate_shape_lag_mult(const Inpar::Mortar::ShapeFcn& lmtype, const double* xi,
        Core::LinAlg::SerialDenseVector& val, Core::LinAlg::SerialDenseMatrix& deriv,
        const int valdim, bool boundtrafo = true) const;

    /*! \brief Evaluate Lagrange multiplier shape functions and derivatives
     *
     *  */
    template <unsigned elenumnode, unsigned eledim>
    inline bool evaluate_shape_lag_mult(Inpar::Mortar::ShapeFcn lmtype, const double* xi,
        Core::LinAlg::Matrix<elenumnode, 1>& val, Core::LinAlg::Matrix<elenumnode, eledim>& deriv,
        unsigned valdim, bool boundtrafo) const
    {
      Core::LinAlg::SerialDenseVector sdv_val(Teuchos::View, val.data(), elenumnode);
      Core::LinAlg::SerialDenseMatrix sdm_deriv(
          Teuchos::View, deriv.data(), elenumnode, elenumnode, eledim);
      return evaluate_shape_lag_mult(lmtype, xi, sdv_val, sdm_deriv, valdim, boundtrafo);
    }

    /*!
    \brief Evaluate Lagrange multiplier shape functions and derivatives
    (special version for 3D quadratic mortar with linear Lagrange multipliers)
    */
    virtual bool evaluate_shape_lag_mult_lin(const Inpar::Mortar::ShapeFcn& lmtype,
        const double* xi, Core::LinAlg::SerialDenseVector& val,
        Core::LinAlg::SerialDenseMatrix& deriv, const int valdim) const;

    /*!
    \brief Evaluate Lagrange multiplier shape functions and derivatives
    (special version for quadratic mortar with element-wise constant Lagrange multipliers)
    */
    virtual bool evaluate_shape_lag_mult_const(const Inpar::Mortar::ShapeFcn& lmtype,
        const double* xi, Core::LinAlg::SerialDenseVector& val,
        Core::LinAlg::SerialDenseMatrix& deriv, const int valdim) const;

    /*! \brief Evaluate Lagrange multiplier shape functions and derivatives
     *  (special version for 3D quadratic mortar with linear Lagrange multipliers)
     *
     *  */
    template <unsigned elenumnode, unsigned eledim>
    inline bool evaluate_shape_lag_mult_lin(Inpar::Mortar::ShapeFcn lmtype, const double* xi,
        Core::LinAlg::Matrix<elenumnode, 1>& val, Core::LinAlg::Matrix<elenumnode, eledim>& deriv,
        int valdim)
    {
      Core::LinAlg::SerialDenseVector sdv_val(Teuchos::View, val.data(), elenumnode);
      Core::LinAlg::SerialDenseMatrix sdm_deriv(
          Teuchos::View, deriv.data(), elenumnode, elenumnode, eledim);
      return evaluate_shape_lag_mult_lin(lmtype, xi, sdv_val, sdm_deriv, valdim);
    }

    /*!
    \brief Evaluate 2nd derivative of shape functions
    */
    virtual bool evaluate2nd_deriv_shape(
        const double* xi, Core::LinAlg::SerialDenseMatrix& secderiv, const int& valdim) const;

    template <unsigned elenumnode>
    inline bool evaluate2nd_deriv_shape(
        const double* xi, Core::LinAlg::Matrix<elenumnode, 3>& secderiv, const int& valdim) const
    {
      Core::LinAlg::SerialDenseMatrix sdm_secderiv(
          Teuchos::View, secderiv.data(), elenumnode, elenumnode, 3);
      return evaluate2nd_deriv_shape(xi, sdm_secderiv, valdim);
    }

    /*!
    \brief Compute directional derivative of dual shape functions

    \param derivdual (in): derivative maps to be filled
                           (= derivatives of the dual coefficient matrix Ae)
    */
    virtual bool deriv_shape_dual(
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivdual);

    /*!
    \brief Interpolate global coordinates for given local element coordinates

    This method interpolates global coordinates for a given local element
    coordinate variable using the element node coordinates. For interpolation
    one can choose between shape functions or shape function derivatives!

    \param xi (in)        : local element coordinates
    \param inttype (in)   : set to 0 for shape function usage,
                            set to 1 for derivative xi usage
                            set to 2 for derivative eta usage (3D only)
    \param globccord (out): interpolated global coordinates
    */
    virtual bool local_to_global(const double* xi, double* globcoord, int inttype) const;

    /*!
    \brief Evaluate minimal edge size of this element


    \return Approximation of minimum geometric dimension of this element
    */
    virtual double min_edge_size() const;

    /*!
    \brief Evaluate maximal edge size of this element

    \note We only care about geometric dimensions. Hence, all elements are treated like linear
    elements here.

    \return Approximation of maximum geometric dimension of this element
    */
    virtual double max_edge_size() const;

    /*!
    \brief Add one Mortar::Element to this Mortar::Element's potential contact partners

    This is for the element-based brute-force search and for the new
    binary search tree. We do NOT have to additionally check, if the
    given Mortar::Element has already been added to this MortarCElement's
    potential contact partners before. This cannot happen by construction!

    */
    virtual bool add_search_elements(const int& gid);

    /*!
    \brief Initializes the data container of the element

    With this function, the container with mortar specific quantities/information
    is initialized.

    */
    virtual void initialize_data_container();

    /*!
    \brief delete all found meles for this element

    */
    virtual void delete_search_elements();

    /*!
    \brief Resets the data container of the element

    With this function, the container with mortar specific quantities/information
    is deleted / reset to nullptr pointer

    */
    virtual void reset_data_container();

    // nurbs specific:

    /*!
    \brief bool for integration: if true: no integration for this element

    this is only the case if there are more than polynom degree + 1 multiple knot entries

    */
    virtual bool& zero_sized() { return zero_sized_; }
    bool zero_sized() const { return zero_sized_; }

    /*!
    \brief factor for normal calculation (default 1.0)

    */
    double& normal_fac() { return normalfac_; }
    double normal_fac() const { return normalfac_; }

    /*!
    \brief get knot vectors for this mortar element

    */
    virtual std::vector<Core::LinAlg::SerialDenseVector>& knots() { return mortarknots_; }
    const std::vector<Core::LinAlg::SerialDenseVector>& knots() const { return mortarknots_; }

    /*!
    \brief Get the linearization of the spatial position of the Nodes for this Ele.

           Returns a vector of vector of maps. Outer vector for the nodes,
           inner vector for the spatial dimensions, map for the derivatives.

           Needed to be overloaded by IntElement
    */
    virtual void node_linearization(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& nodelin) const;

    // h.Willmann return physical type of the mortar element
    PhysicalType& phys_type() { return physicaltype_; }
    PhysicalType phys_type() const { return physicaltype_; }

    /*!
    \brief Estimate mesh size and stiffness parameter h/E via Eigenvalues of the trace inequality.
           For Nitsche contact formulations

    \note It has been decided that Nitsche's method for contact problems will only be done in three
    dimensions. Hence, we check for the number of spatial dimensions and throw an error if the
    problem at hand is not 3D.
      */
    void estimate_nitsche_trace_max_eigenvalue();

    /*!
    \brief Estimated mesh size and stiffness parameter h/E via Eigenvalues of the trace inequality.
           For Nitsche contact formulations
      */
    virtual double& trace_he() { return traceHE_; }

    /*!
    \brief Estimated mesh size and thermal conductivity h/K via Eigenvalues of the trace inequality.
           For Nitsche contact formulations
      */
    virtual double& trace_h_cond() { return traceHCond_; }

    /*!
    \brief Get Nitsche data container
      */
    virtual Mortar::ElementNitscheContainer& get_nitsche_container();
    //@}

   protected:
    Core::FE::CellType shape_;  // shape of this element
    bool isslave_;              // indicating slave or master side
    bool attached_;             // bool whether an element contributes to M

    std::shared_ptr<Mortar::MortarEleDataContainer> modata_;  // additional information

    // nurbs specific:
    bool nurbs_;
    std::vector<Core::LinAlg::SerialDenseVector> mortarknots_;  // mortar element knot vector
    double normalfac_;                                          // factor for normal orientation
    bool zero_sized_;  // zero-sized element: if true: no integration for this element

    // h.Willmann
    PhysicalType physicaltype_;  // physical type

    // approximation of mesh size and stiffness from inverse trace inequality (h/E)
    double traceHE_;
    // approximation of mesh size and stiffness from inverse trace inequality (h/conductivity)
    double traceHCond_;

    // data container for element matrices in Nitsche contact
    std::shared_ptr<Mortar::ElementNitscheContainer> nitsche_container_;

    /*!
    \brief Protected constructor for use in derived classes that expect standard element
    constructors

    \param id    (in): A globally unique element id
    \param owner (in): owner processor of the element
    */
    Element(int id, int owner);

  };  // class Element

  /*!
  \brief A class to perform Gaussian integration on a mortar element

  */

  class ElementIntegrator
  {
   public:
    /*!
    \brief Standard constructor

    */
    ElementIntegrator(Core::FE::CellType eletype);

    /*!
    \brief Destructor

    */
    virtual ~ElementIntegrator() = default;
    //! @name Access methods

    /*!
    \brief Return number of Gauss points

    */
    int& n_gp() { return ngp_; }

    /*!
    \brief Return coordinates of a specific GP

    */
    double& coordinate(int& gp, int dir) { return coords_(gp, dir); }

    /*!
    \brief Return weight of a specific GP in 1D/2D CElement

    */
    double& weight(int& gp) { return weights_[gp]; }

    //@}

   protected:
    // don't want = operator and cctor
    ElementIntegrator operator=(const ElementIntegrator& old);
    ElementIntegrator(const ElementIntegrator& old);

    int ngp_;                                 // number of Gauss points
    Core::LinAlg::SerialDenseMatrix coords_;  // Gauss point coordinates
    std::vector<double> weights_;             // Gauss point weights

  };  // class ElementIntegrator
}  // namespace Mortar

// << operator
std::ostream& operator<<(std::ostream& os, const Mortar::Element& ele);

FOUR_C_NAMESPACE_CLOSE

#endif
