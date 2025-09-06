// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_EULER_BERNOULLI_HPP
#define FOUR_C_BEAM3_EULER_BERNOULLI_HPP


#include "4C_config.hpp"

#include "4C_beam3_base.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_serialdensevector.hpp"

// #define SIMPLECALC ;          // simplified residuum and stiffness calculation for the small
//  tension case Default: Off
#define NODALDOFS \
  2  // With this flag it can be switched between third order (2 nodal dofs) and fifth order
     // hermitian polynomials (3 nodal dofs). So the valid values of NODALDOFS are 2 and 3
// Default: 2

#define ANS_BEAM3EB  // Flag, to apply Assumed Natural Strain approach for axial tension epsilon
// Default: On
#define ANSVALUES 3  // Decide, how many ANS points are applied (2 or 3 possible)
// Default: 3

#define mygaussruleeb \
  Core::FE::GaussRule1D::line_4point  // define gauss rule; intrule_line_1point -
                                      // intrule_line_10point
                                      // is implemented
// Default: 4

// #define BEAM3EBAUTOMATICDIFF      // is Sacado used or not?
//  Default: off

// #define CONSISTENTANSBEAM3EB   //Decide whether the variational correct or the simplified ANS
//  approach should be applied Default: off

// #define INEXTENSIBLE 1.0        //apply inextensibility constraint. only possible in combination
//  with FAD and ANS_BEAM3EB and ANSVALUES=3 as well as NODALDOFS=2
//  Default: off

// #define SWITCHINEXTENSIBLEON
//  Default: off

// #define ORTHOPRESSURE 3.1415926535897938
//  Default: off

// #define BEAM3EBROTPTC          //switch from standard PTC scheme to angular velocity based PTC
// for
//  tangential DoFs Default: off #define BEAM3EBCONSTSTOCHFORCE       //Flag in order to hold
//  stochastic forces constant over the element length Default: off                  //(needed in
//  order to compare with Reissner beams)

// #define BEAM3EBDISCRETELINENEUMANN 0.0 //Flag in order to interpret line Neumann condition as
//  discrete force at given element parameter position DISCRETELINENEUMANN Default: off //for all
//  elements contained in the corresponding design line!

#if (defined(CONSISTENTANSBEAM3EB) or defined(INEXTENSIBLE) or defined(ORTHOPRESSURE)) and \
    !defined(BEAM3EBAUTOMATICDIFF)
FOUR_C_THROW(
    "CONSISTENTANSBEAM3EB, INEXTENSIBLE and ORTHOPRESSURE only work in combination with "
    "BEAM3EBAUTOMATICDIFF!");
#endif

#if defined(INEXTENSIBLE) and !defined(ANS_BEAM3EB)
FOUR_C_THROW(Flag INEXTENSIBLE only possible in combination with flag ANS_BEAM3EB !);
#endif

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  class Beam3ebType : public Core::Elements::ElementType
  {
   public:
    std::string name() const override { return "Beam3ebType"; }

    static Beam3ebType& instance();

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

    std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
        const std::string eledistype, const int id, const int owner) override;

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    int initialize(Core::FE::Discretization& dis) override;

    void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

    void setup_element_definition(
        std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
        override;

   private:
    static Beam3ebType instance_;
  };

  /*!
  \brief 3D nonlinear Euler-Bernoulli-like beam element based on chapter 5 of the diploma thesis
  "Development of a finite element for nonlinear beams based on the formulas of Frenet-Serret" by
  Christoph Meier. The current formulation is only able to display axial tension and bending
  curvature based on the curve describing the centerline of an initially (i.e. stress free)
  straight beam. There is no shear deformation and no torsion. Refer to 'beam3k' for a full
  Kirchhoff type element.

  */
  class Beam3eb : public Beam3Base
  {
   public:
    //! @name Friends
    friend class Beam3ebType;


    //! @name Constructors and destructors and related methods

    /*!
    \brief Standard Constructor

    \param id    (in): A globally unique element id
    \param etype (in): Type of element
    \param owner (in): owner processor of the element
    */
    Beam3eb(int id, int owner);

    /*!
    \brief Copy Constructor

    Makes a deep copy of a Element
    */
    Beam3eb(const Beam3eb& old);

    /*!
    \brief Deep copy this instance of Beam3eb and return pointer to the copy

    The clone() method is used by the virtual base class Element in cases
    where the type of the derived class is unknown and a copy-ctor is needed
  .
    */
    Core::Elements::Element* clone() const override;

    //! \brief Get shape type of element
    Core::FE::CellType shape() const override;

    /*!
    \brief Return unique ParObject id

    Every class implementing ParObject needs a unique id defined at the
    top of parobject.H
    */
    int unique_par_object_id() const override
    {
      return Beam3ebType::instance().unique_par_object_id();
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

    Core::Elements::ElementType& element_type() const override { return Beam3ebType::instance(); }

    //@}

    //! \brief Return number of lines to this element
    int num_line() const override { return 1; }

    //! \brief get number of nodes used for centerline interpolation
    inline int num_centerline_nodes() const override { return 2; }

    //! \brief find out whether given node is used for centerline interpolation
    inline bool is_centerline_node(const Core::Nodes::Node& node) const override { return true; }

    //! \brief Get jacobian this element
    const double& get_jacobi() const { return jacobi_; }

    //! \brief get access to the reference length
    inline double ref_length() const override { return 2.0 * jacobi_; }

    /** \brief get unit tangent vector in reference configuration at i-th node of beam element
     * (element-internal numbering)
     */
    inline void get_ref_tangent_at_node(
        Core::LinAlg::Matrix<3, 1>& Tref_i, const int& i) const override
    {
      Tref_i = tref()[i];
    }

    //! \brief Get maximal bending curvature occurring in this element
    const double& get_kappa_max() const { return kappa_max_; }

    //! \brief Get material cross-section deformation measures, i.e. strain resultants
    inline void get_material_strain_resultants_at_all_gps(std::vector<double>& axial_strain_GPs,
        std::vector<double>& shear_strain_2_GPs, std::vector<double>& shear_strain_3_GPs,
        std::vector<double>& twist_GPs, std::vector<double>& curvature_2_GPs,
        std::vector<double>& curvature_3_GPs) const override
    {
      axial_strain_GPs = axial_strain_gp_;
      // shear deformations are zero by definition for Kirchhoff beam formulation
      shear_strain_2_GPs.clear();
      shear_strain_3_GPs.clear();

      // twist deformation cannot be represented by this reduced Kirchhoff beam formulation
      twist_GPs.clear();
      // only one curvature component due to isotropic formulation of this reduced Kirchhoff beam
      curvature_2_GPs = curvature_gp_;
      curvature_3_GPs.clear();
    }

    //! \brief Get material cross-section stress resultants
    inline void get_material_stress_resultants_at_all_gps(std::vector<double>& axial_force_GPs,
        std::vector<double>& shear_force_2_GPs, std::vector<double>& shear_force_3_GPs,
        std::vector<double>& torque_GPs, std::vector<double>& bending_moment_2_GPs,
        std::vector<double>& bending_moment_3_GPs) const override
    {
      axial_force_GPs = axial_force_gp_;
      // note: shear deformations are zero by definition for Kirchhoff beam formulation
      shear_force_2_GPs.clear();
      shear_force_3_GPs.clear();

      // torsion moment cannot be represented by this reduced Kirchhoff beam formulation
      torque_GPs.clear();
      // only one bending moment component due to isotropic formulation of this reduced Kirchhoff
      // beam
      bending_moment_2_GPs = bending_moment_gp_;
      bending_moment_3_GPs.clear();
    }

    //! \brief get centerline position at xi \in [-1,1] (element parameter space)
    void get_pos_at_xi(Core::LinAlg::Matrix<3, 1>& pos, const double& xi,
        const std::vector<double>& disp) const override;

    //! \brief get triad at xi \in [-1,1] (element parameter space)
    void get_triad_at_xi(Core::LinAlg::Matrix<3, 3>& triad, const double& xi,
        const std::vector<double>& disp) const override;

    //! \brief Get axial strain at xi for given nodal displacements
    double get_axial_strain(double& xi, const Core::LinAlg::Matrix<12, 1>& disp_totlag) const;

    //! get internal (elastic) energy of element
    double get_internal_energy() const override { return eint_; };

    //! get kinetic energy of element
    double get_kinetic_energy() const override { return ekin_; };

    //! \brief Get vector of std::shared_ptrs to the lines of this element
    std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

    //! \brief Get number of degrees of freedom of a single node
    int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
/*note: this is not necessarily the number of DOF assigned to this node by the discretization
 *finally, but only the number of DOF requested for this node by this element; the discretization
 *will finally assign the maximal number of DOF to this node requested by any element connected to
 *this node*/
#ifndef INEXTENSIBLE
      return 3 * NODALDOFS;
#else
      if (node.Id() == this->Nodes()[0]->Id() or node.Id() == this->Nodes()[1]->Id())
      {
        return 7;
      }
      else
      {
        return 1;
      }
#endif
    }

    //! \brief Get number of degrees of freedom per element not including nodal degrees of freedom
    int num_dof_per_element() const override { return 0; }

    //! \brief Print this element
    void print(std::ostream& os) const override;

    //! \brief get reference triad i.e. Tref_
    std::vector<Core::LinAlg::Matrix<3, 1>> tref() const;

    //! \brief get jacobi factor jacobi_
    double jacobi() const;

    //! \brief get Jacobi factor ds/dxi(xi) at xi \in [-1;1]
    inline double get_jacobi_fac_at_xi(const double& xi) const override
    {
      /* beam3eb elements are restricted to initially straight beams, for which the jacobi factor
       * is constant along the entire element length */
      return jacobi();
    }

    //! @name Construction

    /*!
    \brief Read input for this element

    This class implements a dummy of this method that prints a warning and
    returns false. A derived class would read one line from the input file and
    store all necessary information.

    */
    // virtual bool read_element();

    //! \brief Read input for this element
    bool read_element(const std::string& eletype, const std::string& distype,
        const Core::IO::InputParameterContainer& container) override;

    //@}

    //! @name Evaluation methods


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
    int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

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
    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

    /*!
    \brief Evaluate PTC addition to stiffness for free Brownian motion

    An element derived from this class uses the Evaluate method to receive commands
    and parameters from some control routine in params and evaluates a  statistical Neumann
    boundary condition as used in the problem type STATISTICAL MECHANICS

    \param params (in/out)       : ParameterList for communication between control routine and
    elements \param vector<double> mydisp : current nodal displacement \param elemat1 (out) :
    artificial damping matrix to be filled by element

    \return 0 if successful, negative otherwise
    */
    template <int nnode>
    void evaluate_ptc(Teuchos::ParameterList& params, Core::LinAlg::SerialDenseMatrix& elemat1);

    //@}


    //@}
    //! sets up from current nodal position all geometric parameters (considering current position
    //! as reference configuration)
    void set_up_reference_geometry(const std::vector<double>& xrefe, const bool secondinit = false);

    /* computes the number of different random numbers required in each time step for generation
     of stochastic forces                                                                    */
    int how_many_random_numbers_i_need() const override;

    //! \brief add indices of those DOFs of a given node that are positions
    inline void position_dof_indices(
        std::vector<int>& posdofs, const Core::Nodes::Node& node) const override
    {
      posdofs.push_back(0);
      posdofs.push_back(1);
      posdofs.push_back(2);
    }

    /** \brief add indices of those DOFs of a given node that are tangents (in the case of Hermite
     * interpolation)
     */
    inline void tangent_dof_indices(
        std::vector<int>& tangdofs, const Core::Nodes::Node& node) const override
    {
      tangdofs.push_back(3);
      tangdofs.push_back(4);
      tangdofs.push_back(5);
    }

    /** \brief add indices of those DOFs of a given node that are rotation DOFs (non-additive
     * rotation vectors)
     */
    inline void rotation_vec_dof_indices(
        std::vector<int>& rotvecdofs, const Core::Nodes::Node& node) const override
    {
    }

    /** \brief add indices of those DOFs of a given node that are 1D rotation DOFs
     *         (planar rotations are additive, e.g. in case of relative twist DOF of beam3k with
     *          rotvec=false)
     */
    inline void rotation_1d_dof_indices(
        std::vector<int>& twistdofs, const Core::Nodes::Node& node) const override
    {
    }

    /** \brief add indices of those DOFs of a given node that represent norm of tangent vector
     *         (additive, e.g. in case of beam3k with rotvec=true)
     */
    inline void tangent_length_dof_indices(
        std::vector<int>& tangnormdofs, const Core::Nodes::Node& node) const override
    {
    }

    //! \brief get element local indices of those Dofs that are used for centerline interpolation
    inline void centerline_dof_indices_of_element(
        std::vector<unsigned int>& centerlinedofindices) const override
    {
      centerlinedofindices.resize(12, 0);

      for (unsigned int idof = 0; idof < 12; ++idof) centerlinedofindices[idof] = idof;
    }

    /** \brief extract values for those Dofs relevant for centerline-interpolation from total
     * state vector
     */
    inline void extract_centerline_dof_values_from_element_state_vector(
        const std::vector<double>& dofvec, std::vector<double>& dofvec_centerline,
        bool add_reference_values = false) const override
    {
      if (dofvec.size() != 12)
        FOUR_C_THROW(
            "size mismatch: expected 12 values for element state vector and got {}", dofvec.size());

      dofvec_centerline.resize(12, 0.0);
      std::copy(dofvec.begin(), dofvec.end(), dofvec_centerline.begin());

      if (add_reference_values)
      {
        for (unsigned int dim = 0; dim < 3; ++dim)
          for (unsigned int node = 0; node < 2; ++node)
          {
            dofvec_centerline[6 * node + dim] += nodes()[node]->x()[dim];

            // Hermite interpolation: add reference values for tangent DOFs as well
            dofvec_centerline[6 * node + 3 + dim] += Tref_[node](dim);
          }
      }
    }

   private:
    //! vector storing the internal force vector
    Core::LinAlg::SerialDenseVector internalforces_;

    //! variable saving whether element has already been initialized (then isinit_ == true)
    bool isinit_;

    //! Vector holding value of Jacobi determinant jacobi for each Gauss point for
    //! underintegration
    double jacobi_;

    //! bool recognizing first element call
    bool firstcall_;

    //! kinetic energy
    double ekin_;

    //! internal energy
    double eint_;
    //! internal energy stemming from axial tension
    double eint_axial_;

    //! angular momentum of the element
    Core::LinAlg::Matrix<3, 1> l_;
    //! linear momentum of the element
    Core::LinAlg::Matrix<3, 1> p_;
    //! nodal tangents of last time step (necessary for PTC scheme)
    Core::LinAlg::Matrix<3, 2> t0_;
    //! nodal tangents of current time step (necessary for PTC scheme)
    Core::LinAlg::Matrix<3, 2> t_;
    //! norm of maximal curvature occurring in this element
    double kappa_max_;
    //! norm of maximal axial tension occurring in this element
    double epsilon_max_;

    //! strain resultant values at GPs
    std::vector<double> axial_strain_gp_;
    std::vector<double> curvature_gp_;

    //! stress resultant values at GPs
    std::vector<double> axial_force_gp_;
    std::vector<double> bending_moment_gp_;

#if NODALDOFS == 3
    //! Matrix holding the derivatives of the tangents at each node in reference configuration
    std::vector<Core::LinAlg::Matrix<3, 1>> Kref_;
#endif

    //! @name Internal calculation methods

    //! calculation of nonlinear stiffness and mass matrix
    void calc_internal_and_inertia_forces_and_stiff(Teuchos::ParameterList& params,
        std::vector<double>& vel, std::vector<double>& disp,
        Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
        Core::LinAlg::SerialDenseVector* force);

    template <unsigned int nnode, unsigned int dofpn>
    void update_disp_totlag(
        const std::vector<double>& disp, Core::LinAlg::Matrix<dofpn * nnode, 1>& disp_totlag) const;

    //! lump mass matrix
    void lumpmass(Core::LinAlg::SerialDenseMatrix* emass);

    //! computes translational damping forces and stiffness
    template <unsigned int nnode, unsigned int vpernode, int ndim>
    void evaluate_translational_damping(Teuchos::ParameterList& params,  //!< parameter list
        const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1>& vel,
        const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1>& disp_totlag,
        Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
        Core::LinAlg::SerialDenseVector* force);       //!< element internal force vector


    //! computes stochastic forces and resulting stiffness
    template <unsigned int nnode, unsigned int vpernode, unsigned int ndim,
        unsigned int randompergauss>
    void evaluate_stochastic_forces(Teuchos::ParameterList& params,  //!< parameter list
        const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1>& disp_totlag,
        Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
        Core::LinAlg::SerialDenseVector* force);       //!< element internal force vector


    /*! \brief Assemble stochastic and viscous forces and respective stiffness according to
     *         fluctuation dissipation
     */
    template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
    void calc_brownian_forces_and_stiff(Teuchos::ParameterList& params,
        std::vector<double>& vel,                      //!< element velocity vector
        std::vector<double>& disp,                     //!< element displacement vector
        Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
        Core::LinAlg::SerialDenseVector* force);       //!< element internal force vector

    // don't want = operator
    Beam3eb& operator=(const Beam3eb& old) = delete;
  };

  // << operator
  std::ostream& operator<<(std::ostream& os, const Core::Elements::Element& ele);

}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif
