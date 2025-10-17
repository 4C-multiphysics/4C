// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_KIRCHHOFF_HPP
#define FOUR_C_BEAM3_KIRCHHOFF_HPP


// Todo @grill: get rid of header in header inclusions
#include "4C_config.hpp"

#include "4C_beam3_base.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

using FAD = Sacado::Fad::DFad<double>;

#define REFERENCE_NODE \
  2  // number of reference-node for the calculation of the rotation of the other triads (local
     // numbering: 0 2 3 4 1)
// DEFAULT: REFERENCE_NODE 2   //REFERENCE_NODE=2 represents the midpoint node in the case of 3
// BEAM3K_COLLOCATION_POINTS -> standard choice!

#if defined(REFERENCE_NODE) && (REFERENCE_NODE != 2)
FOUR_C_THROW(
    "Beam3k REFERENCE_NODE: Only the value 2 is covered by tests and has therefore been "
    "cultivated in subsequent modifications to the code; carefully check correctness of code "
    "before using other values than 2!");
// note: especially look into the code for triad interpolation based on local rotation vectors
//       (as applied in WK case); reference triad according to Crisfield/Jelenic1999 is chosen as
//       triad at the 'mid'-node for uneven number of nodes
//       => REFERENCE_NODE=2 for BEAM3K_COLLOCATION_POINTS=3
//       generalize the class for triad interpolation if other choices of the reference nodes are
//       desired here in Beam3k
#endif

#define MYGAUSSRULEBEAM3K \
  Core::FE::GaussRule1D::line_4point  // define gauss rule; intrule_line_1point -
                                      // intrule_line_10point is implemented.
// DEFAULT: intrule_line_4point

#define BEAM3K_COLLOCATION_POINTS \
  3  // defines type of element. 2,3,4 are supported. A value of 3 or 4 means that further inner
     // nodes are introduced
// DEFAULT: BEAM3K_COLLOCATION_POINTS 3   //to interpolate the torsional degree of freedom alpha_.
// Furthermore, it specifies the number of collocation points defining the number of material triads
// used to interpolate the triad field.

#if defined(BEAM3K_COLLOCATION_POINTS) && (BEAM3K_COLLOCATION_POINTS != 3)
FOUR_C_THROW(
    "BEAM3K_COLLOCATION_POINTS: Only the value 3 is covered by tests and has therefore been "
    "cultivated in subsequent modifications to the code; carefully check correctness of code "
    "before using other values than 3!");
#endif

FOUR_C_NAMESPACE_OPEN

// #define CONSISTENTSPINSK        //Apply variationally consistent variant of first spin vector
//  component as test function for strong Kirchhoff
//-> this interpolation enables exact energy and angular momentum balances

// forward declaration
namespace LargeRotations
{
  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors;
}

namespace Discret
{
  namespace Elements
  {
    /*
     * @brief Flag for enforcement of Kirchhoff constraint.
     */
    enum class Beam3KirchhoffConstraintType
    {
      strong,
      weak
    };

    /*
     * @brief Flag for type of beam parametrization.
     */
    enum class Beam3KirchhoffParametrizationType
    {
      tan,
      rot
    };

    class Beam3kType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Beam3kType"; }

      static Beam3kType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& actnode, std::span<const double> x0, const int numdof) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static Beam3kType instance_;
    };

    /*!
    \brief 3D nonlinear Kirchhoff-like beam element that can display initially curved beams.

    */
    class Beam3k : public Beam3Base
    {
     public:
      //! @name Friends
      friend class Beam3kType;

      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id    (in): A globally unique element id
      \param etype (in): Type of element
      \param owner (in): owner processor of the element
      */
      Beam3k(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      Beam3k(const Beam3k& old);



      /*!
      \brief Deep copy this instance of Beam3k and return pointer to the copy

      The clone() method is used by the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed
    .
      */
      Core::Elements::Element* clone() const override;

      /*!
     \brief Get shape type of element
     */
      Core::FE::CellType shape() const override;


      /*!
      \brief Return unique ParObject id

      Every class implementing ParObject needs a unique id defined at the
      top of parobject.H
      */
      int unique_par_object_id() const override
      {
        return Beam3kType::instance().unique_par_object_id();
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

      Core::Elements::ElementType& element_type() const override { return Beam3kType::instance(); }

      //@}

      /*!
       \brief Get bool inidcating usage of rotation vectors
       */
      const bool& rot_vec() const { return rotvec_; }

      /*!
      \brief get reference rotation vector i.e. theta0_
      */
      std::vector<Core::LinAlg::Matrix<3, 1>> theta0() const { return theta0_; }

      /*!
       \brief Get (non-unit) tangent vectors at the two boundary nodes
      */
      std::vector<Core::LinAlg::Matrix<3, 1>> get_nodal_tangents() const { return t_; }

      /** \brief get unit tangent vector in reference configuration at i-th node of beam element
       * (element-internal numbering)
       *
       */
      inline void get_ref_tangent_at_node(
          Core::LinAlg::Matrix<3, 1>& Tref_i, const int& i) const override
      {
        if (not((unsigned)i < tref().size()))
          FOUR_C_THROW("asked for tangent at node index {}, but only {} centerline nodes existing",
              i, tref().size());
        Tref_i = tref()[i];
      }

      /** \brief get centerline position at xi \in [-1,1] (element parameter space)
       *
       */
      void get_pos_at_xi(Core::LinAlg::Matrix<3, 1>& pos, const double& xi,
          const std::vector<double>& disp) const override;

      /** \brief get triad at xi \in [-1,1] (element parameter space)
       *
       */
      void get_triad_at_xi(Core::LinAlg::Matrix<3, 3>& triad, const double& xi,
          const std::vector<double>& disp) const override;

      /** \brief Get scaled base vectors describing the cross-section orientation and size at given
       * parameter coordinate xi
       *
       *  note: this method is only used for visualization so far and limited to a rectangular (?)
       *        cross-section shape; the length of the base vectors indicates the size of the
       * cross-section in the direction of the base vector
       *
       */
      void get_scaled_second_and_third_base_vector_at_xi(const double& xi,
          const std::vector<double>& disp, Core::LinAlg::Matrix<3, 2>& scaledbasevectors) const;

      /** \brief get generalized interpolation matrix which yields the variation of the position and
       *         orientation at xi \in [-1,1] if multiplied with the vector of primary DoF
       * variations
       *
       */
      void get_generalized_interpolation_matrix_variations_at_xi(
          Core::LinAlg::SerialDenseMatrix& Ivar, const double& xi,
          const std::vector<double>& disp) const override;

      /** \brief get linearization of the product of (generalized interpolation matrix for
       * variations (see above) and applied force vector) with respect to the primary DoFs of this
       * element
       *
       */
      void get_stiffmat_resulting_from_generalized_interpolation_matrix_at_xi(
          Core::LinAlg::SerialDenseMatrix& stiffmat, const double& xi,
          const std::vector<double>& disp,
          const Core::LinAlg::SerialDenseVector& force) const override;

      /** \brief get generalized interpolation matrix which yields the increments of the position
       * and orientation at xi \in [-1,1] if multiplied with the vector of primary DoF increments
       *
       */
      void get_generalized_interpolation_matrix_increments_at_xi(
          Core::LinAlg::SerialDenseMatrix& Iinc, const double& xi,
          const std::vector<double>& disp) const override;

      /** \brief get access to the reference length
       *
       */
      inline double ref_length() const override { return length_; }

      /*!
      \brief Get jacobi factor of first Gauss point
      */
      const double& get_jacobi() const { return jacobi_[0]; }

      /** \brief get Jacobi factor ds/dxi(xi) at xi \in [-1;1]
       *
       */
      double get_jacobi_fac_at_xi(const double& xi) const override;

      /** \brief Get material cross-section deformation measures, i.e. strain resultants
       *
       */
      inline void get_material_strain_resultants_at_all_gps(std::vector<double>& axial_strain_GPs,
          std::vector<double>& shear_strain_2_GPs, std::vector<double>& shear_strain_3_GPs,
          std::vector<double>& twist_GPs, std::vector<double>& curvature_2_GPs,
          std::vector<double>& curvature_3_GPs) const override
      {
        axial_strain_GPs = axial_strain_gp_;
        // note: shear deformations are zero by definition for Kirchhoff beam formulation
        shear_strain_2_GPs.clear();
        shear_strain_3_GPs.clear();

        twist_GPs = twist_gp_;
        curvature_2_GPs = curvature_2_gp_;
        curvature_3_GPs = curvature_3_gp_;
      }

      /** \brief Get material cross-section stress resultants
       *
       */
      inline void get_material_stress_resultants_at_all_gps(std::vector<double>& axial_force_GPs,
          std::vector<double>& shear_force_2_GPs, std::vector<double>& shear_force_3_GPs,
          std::vector<double>& torque_GPs, std::vector<double>& bending_moment_2_GPs,
          std::vector<double>& bending_moment_3_GPs) const override
      {
        axial_force_GPs = axial_force_gp_;
        // note: shear deformations are zero by definition for Kirchhoff beam formulation
        shear_force_2_GPs.clear();
        shear_force_3_GPs.clear();

        torque_GPs = torque_gp_;
        bending_moment_2_GPs = bending_moment_2_gp_;
        bending_moment_3_GPs = bending_moment_3_gp_;
      }

      //! get internal (elastic) energy of element
      double get_internal_energy() const override { return eint_; };

      //! get kinetic energy of element
      double get_kinetic_energy() const override { return ekin_; };

      /** \brief get number of nodes used for centerline interpolation
       *
       */
      inline int num_centerline_nodes() const override { return 2; }

      /** \brief find out whether given node is used for centerline interpolation
       *
       */
      inline bool is_centerline_node(const Core::Nodes::Node& node) const override
      {
        if (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id())
          return true;
        else
          return false;
      }

      /*!
      \brief Return number of lines to this element
      */
      int num_line() const override { return 1; }


      /*!
      \brief Get vector of RCPs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;


      /*!
      \brief Get number of degrees of freedom of a single node
      */
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        /*note: this is not necessarily the number of DOF assigned to this node by the
         *discretization finally, but only the number of DOF requested for this node by this
         *element; the discretization will finally assign the maximal number of DOF to this node
         *requested by any element connected to this node*/
        int dofpn = 0;

        if (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id())
          dofpn = 7;
        else
          dofpn = 1;

        return dofpn;
      }

      /*!
      \brief Get number of degrees of freedom per element not including nodal degrees of freedom
      */
      int num_dof_per_element() const override { return 0; }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

      //@}

      //! @name Construction


      /*!
      \brief Read input for this element

      This class implements a dummy of this method that prints a warning and
      returns false. A derived class would read one line from the input file and
      store all necessary information.

      */
      // virtual bool read_element();

      /*!
      \brief Read input for this element
      */
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
          Core::LinAlg::SerialDenseMatrix* elemat1) override;

      /*!
      \brief Set rotations in the initial configuration

       Set the initial rotations based on nodal rotation (pseudo) vectors.
       The nodal rotation vectors are independent of the subsequent used
       rotational interpolation method.
      */
      void set_up_initial_rotations(const std::vector<double>& nodal_thetas);


      //! sets up from current nodal position all geometric parameters (considering current position
      //! as reference configuration)
      void set_up_reference_geometry(
          const std::vector<Core::LinAlg::Matrix<3, 1>>& xrefe, const bool secondinit = false);

      //! computes the artificial damping contributions for element based PTC
      void calc_stiff_contributions_ptc(Core::LinAlg::SerialDenseMatrix& elemat1);

      /*!
      \brief get (non-unit) tangent vectors at the two boundary nodes in the initial configuration
      */
      std::vector<Core::LinAlg::Matrix<3, 1>> tref() const { return Tref_; }
      //@}

      /** \brief add indices of those DOFs of a given node that are positions
       *
       */
      inline void position_dof_indices(
          std::vector<int>& posdofs, const Core::Nodes::Node& node) const override
      {
        if (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id())
        {
          posdofs.push_back(0);
          posdofs.push_back(1);
          posdofs.push_back(2);
        }
        return;
      }

      /** \brief add indices of those DOFs of a given node that are tangents (in the case of Hermite
       * interpolation)
       *
       */
      inline void tangent_dof_indices(
          std::vector<int>& tangdofs, const Core::Nodes::Node& node) const override
      {
        if ((not rotvec_) and
            (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id()))
        {
          tangdofs.push_back(3);
          tangdofs.push_back(4);
          tangdofs.push_back(5);
        }
        return;
      }

      /** \brief add indices of those DOFs of a given node that are rotation DOFs (non-additive
       * rotation vectors)
       *
       */
      inline void rotation_vec_dof_indices(
          std::vector<int>& rotvecdofs, const Core::Nodes::Node& node) const override
      {
        if ((rotvec_) and
            (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id()))
        {
          rotvecdofs.push_back(3);
          rotvecdofs.push_back(4);
          rotvecdofs.push_back(5);
        }
        return;
      }

      /** \brief add indices of those DOFs of a given node that are 1D rotation DOFs
       *         (planar rotations are additive, e.g. in case of relative twist DOF of beam3k with
       * rotvec=false)
       *
       */
      inline void rotation_1d_dof_indices(
          std::vector<int>& twistdofs, const Core::Nodes::Node& node) const override
      {
        if ((not rotvec_) and
            (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id()))
        {
          twistdofs.push_back(6);
        }
        else if (node.id() == this->nodes()[2]->id())
        {
          twistdofs.push_back(0);
        }

        return;
      }

      /** \brief add indices of those DOFs of a given node that represent norm of tangent vector
       *         (additive, e.g. in case of beam3k with rotvec=true)
       *
       */
      inline void tangent_length_dof_indices(
          std::vector<int>& tangnormdofs, const Core::Nodes::Node& node) const override
      {
        if (rotvec_)
        {
          if (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id())
          {
            tangnormdofs.push_back(6);
          }
        }
        return;
      }

      /** \brief get element local indices of those Dofs that are used for centerline interpolation
       *
       */
      inline void centerline_dof_indices_of_element(
          std::vector<unsigned int>& centerlinedofindices) const override
      {
        if (rotvec_)
          FOUR_C_THROW(
              "The logic of this implementation does not apply for Beam3k with rotation vector "
              "Dofs! "
              "Be careful and find a solution to convert force/stiffness contributions to tangent "
              "Dofs "
              "automatically and apply them consistently to rotvec Dofs");

        const unsigned int nnodecl = this->num_centerline_nodes();
        centerlinedofindices.resize(6 * nnodecl, 0);

        for (unsigned int inodecl = 0; inodecl < nnodecl; ++inodecl)
          for (unsigned int idof = 0; idof < 6; ++idof)
            centerlinedofindices[6 * inodecl + idof] = 7 * inodecl + idof;
      }

      /** \brief extract values for those Dofs relevant for centerline-interpolation from total
       * state vector
       *
       */
      void extract_centerline_dof_values_from_element_state_vector(
          const std::vector<double>& dofvec, std::vector<double>& dofvec_centerline,
          bool add_reference_values = false) const override;

      //! computes the number of different random numbers required in each time step for generation
      //! of stochastic forces
      int how_many_random_numbers_i_need() const override;

     private:
      // Functions

      //! @name methods for initialization of the element

      //! sets up from current nodal position all geometric parameters (considering current position
      //! as reference configuration) in case of a Weak Kirchhoff Constraint
      void set_up_reference_geometry_wk(
          const std::vector<Core::LinAlg::Matrix<3, 1>>& xrefe, const bool secondinit);

      //! sets up from current nodal position all geometric parameters (considering current position
      //! as reference configuration) in case of a Strong Kirchhoff Constraint
      void set_up_reference_geometry_sk(
          const std::vector<Core::LinAlg::Matrix<3, 1>>& xrefe, const bool secondinit);

      //@}

      //! @name auxiliary computation methods for non-additive, large rotation variables

      //@}

      //! @name Internal calculation methods

      /** \brief Calculate internal forces and stiffness matrix
       *
       */
      void calc_internal_and_inertia_forces_and_stiff(Teuchos::ParameterList& params,
          std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::SerialDenseMatrix* massmatrix, Core::LinAlg::SerialDenseVector* force,
          Core::LinAlg::SerialDenseVector* force_inert);

      /** \brief Calculate internal forces and stiffness matrix in case of a Weak Kirchhoff
       * Constraint
       *
       */
      template <unsigned int nnodecl, typename T>
      void calculate_internal_forces_and_stiff_wk(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline,
          const std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& internal_force,
          std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>>&
              v_theta_gp,
          std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>>&
              lin_theta_gp,
          std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_gp);

      /** \brief Calculate internal forces and stiffness matrix in case of a Strong Kirchhoff
       * Constraint
       *
       */
      template <unsigned int nnodecl>
      void calculate_internal_forces_and_stiff_sk(Teuchos::ParameterList& params,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>&
              disp_totlag_centerline,
          const std::vector<Core::LinAlg::Matrix<3, 3, FAD>>& triad_mat_cp,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>& internal_force,
          std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD>>&
              v_theta_gp,
          std::vector<Core::LinAlg::Matrix<3, 3, FAD>>& triad_mat_gp);

      /** \brief Calculate contributions to the stiffness matrix at a Gauss point analytically
       *         in case of weak Kirchhoff constraint
       *
       */
      template <unsigned int nnodecl>
      void calculate_stiffmat_contributions_analytic_wk(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
              disp_totlag_centerline,
          const LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              double>& triad_intpol,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double>&
              v_theta_s_bar,
          const std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              double>>& lin_theta_cp,
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta_bar,
          const std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>& lin_v_epsilon_cp,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
              v_epsilon_bar,
          double axial_force_bar, const Core::LinAlg::Matrix<3, 1, double>& moment_resultant,
          double axial_rigidity,
          const Core::LinAlg::Matrix<3, 3, double>& constitutive_matrix_moment_material,
          const Core::LinAlg::Matrix<3, 1, double>& theta_gp,
          const Core::LinAlg::Matrix<3, 1, double>& theta_s_gp,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat_gp, double xi_gp, double jacobifac_gp,
          double GPwgt) const;

      template <unsigned int nnodecl>
      void calculate_stiffmat_contributions_analytic_wk(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>&
              disp_totlag_centerline,
          const LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              FAD>& triad_intpol,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD>&
              v_theta_s_bar,
          const std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>>&
              lin_theta_cp,
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>& lin_theta_bar,
          const std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>>& lin_v_epsilon_cp,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>&
              v_epsilon_bar,
          FAD axial_force_bar, const Core::LinAlg::Matrix<3, 1, FAD>& moment_resultant,
          FAD axial_rigidity,
          const Core::LinAlg::Matrix<3, 3, FAD>& constitutive_matrix_moment_material,
          const Core::LinAlg::Matrix<3, 1, FAD>& theta_gp,
          const Core::LinAlg::Matrix<3, 1, FAD>& theta_s_gp,
          const Core::LinAlg::Matrix<3, 3, FAD>& triad_mat_gp, double xi_gp, double jacobifac_gp,
          double GPwgt) const
      {
        // this is a dummy because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }

      /** \brief pre-compute quantities required for analytic computation of stiffness matrix
       *         in case of weak Kirchhoff constraint
       *
       */
      template <unsigned int nnodecl>
      void pre_compute_terms_at_cp_for_stiffmat_contributions_analytic_wk(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_epsilon,
          const Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_s, double abs_r_s,
          const Core::LinAlg::Matrix<4, 1, double>& Qref_conv) const;

      template <unsigned int nnodecl>
      void pre_compute_terms_at_cp_for_stiffmat_contributions_analytic_wk(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>& lin_theta,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>& lin_v_epsilon,
          const Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>& L,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>& N_s,
          const Core::LinAlg::Matrix<3, 1, FAD>& r_s, FAD abs_r_s,
          const Core::LinAlg::Matrix<4, 1, double>& Qref_conv) const
      {
        // this is empty because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }


      /** \brief Calculate inertia forces and mass matrix
       *
       */
      template <unsigned int nnodecl, typename T>
      void calculate_inertia_forces_and_mass_matrix(Teuchos::ParameterList& params,
          const std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_gp,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline,
          const std::vector<Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>>&
              v_theta_gp,
          const std::vector<Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>>&
              lin_theta_gp,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& f_inert,
          Core::LinAlg::SerialDenseMatrix* massmatrix);

      /** \brief Calculate analytic linearization of inertia forces, i.e. mass matrix
       *
       */
      template <unsigned int nnodecl>
      void calculate_mass_matrix_contributions_analytic_wk(
          Core::LinAlg::SerialDenseMatrix& massmatrix,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
              disp_totlag_centerline,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, double>&
              v_theta_bar,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>&
              lin_theta_bar,
          const Core::LinAlg::Matrix<3, 1, double>& moment_rho,
          const Core::LinAlg::Matrix<3, 1, double>& deltatheta,
          const Core::LinAlg::Matrix<3, 1, double>& angular_velocity_material,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N,
          double mass_inertia_translational,
          const Core::LinAlg::Matrix<3, 3, double>& tensor_mass_moment_of_inertia,
          double lin_prefactor_acc, double lin_prefactor_vel, double xi_gp, double jacobifac_gp,
          double GPwgt) const;

      template <unsigned int nnodecl>
      void calculate_mass_matrix_contributions_analytic_wk(
          Core::LinAlg::SerialDenseMatrix& massmatrix,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>&
              disp_totlag_centerline,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, FAD>& v_theta_bar,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>&
              lin_theta_bar,
          const Core::LinAlg::Matrix<3, 1, FAD>& moment_rho,
          const Core::LinAlg::Matrix<3, 1, FAD>& deltatheta,
          const Core::LinAlg::Matrix<3, 1, FAD>& angular_velocity_material,
          const Core::LinAlg::Matrix<3, 3, FAD>& triad_mat,
          const Core::LinAlg::Matrix<3, 3, FAD>& triad_mat_conv,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>& N,
          double density, const Core::LinAlg::Matrix<3, 3, FAD>& tensor_mass_moment_of_inertia,
          double lin_prefactor_acc, double lin_prefactor_vel, double xi_gp, double jacobifac_gp,
          double GPwgt) const
      {
        // this is empty because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic mass
        // matrix
      }

      /** \brief evaluate contributions to element residual vector and stiffmat from point Neumann
       *         condition
       *         note: we need to evaluate this on element level because point moments need to be
       *         linearized in case of tangent-based formulation (rotvec_=false)
       *
       */
      template <unsigned int nnodecl>
      void evaluate_point_neumann_eb(Core::LinAlg::SerialDenseVector& forcevec,
          Core::LinAlg::SerialDenseMatrix* stiffmat,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
              disp_totlag,
          const Core::LinAlg::Matrix<6, 1, double>& load_vector_neumann, int node) const;

      /** \brief evaluate contributions to element residual vector from point Neumann moment
       *
       */
      template <unsigned int nnodecl, typename T>
      void evaluate_residual_from_point_neumann_moment(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& force_ext,
          const Core::LinAlg::Matrix<3, 1, T>& moment_ext, const Core::LinAlg::Matrix<3, 1, T>& r_s,
          T abs_r_s, int node) const;

      /** \brief evaluate contributions to element stiffness matrix from point Neumann moment
       *
       */
      template <unsigned int nnodecl>
      void evaluate_stiff_matrix_analytic_from_point_neumann_moment(
          Core::LinAlg::SerialDenseMatrix& stiffmat,
          const Core::LinAlg::Matrix<3, 1, double>& moment_ext,
          const Core::LinAlg::Matrix<3, 1, double>& r_s, double abs_r_s, int node) const;

      /** \brief evaluate contributions to element residual vector and stiffmat from line Neumann
       *         condition
       *
       */
      template <unsigned int nnodecl>
      void evaluate_line_neumann(Core::LinAlg::SerialDenseVector& forcevec,
          Core::LinAlg::SerialDenseMatrix* stiffmat,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, double>&
              disp_totlag,
          const Core::LinAlg::Matrix<6, 1, double>& load_vector_neumann,
          const std::vector<std::optional<int>>& function_numbers, double time) const;

      /** \brief evaluate contributions to element residual vector from line Neumann condition
       *
       */
      template <unsigned int nnodecl, typename T>
      void evaluate_line_neumann_forces(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& force_ext,
          const Core::LinAlg::Matrix<6, 1, double>& load_vector_neumann,
          const std::vector<std::optional<int>>& function_numbers, double time) const;

      /** \brief evaluate all contributions from brownian dynamics (thermal & viscous
       * forces/moments)
       *
       */
      template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void calc_brownian_forces_and_stiff(Teuchos::ParameterList& params,
          std::vector<double>& vel,                      //!< element velocity vector
          std::vector<double>& disp,                     //!< element displacement vector
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::SerialDenseVector* force);       //!< element internal force vector

      /** \brief evaluate all contributions from translational damping forces
       *
       */
      template <typename T, unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void evaluate_translational_damping(Teuchos::ParameterList& params,  //!< parameter list
          const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1, double>& vel,
          const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1, T>& disp_totlag,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>&
              f_int);  //!< element internal force vector

      /** \brief evaluate contributions to element stiffness matrix from translational damping
       * forces
       *
       */
      template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void evaluate_analytic_stiffmat_contributions_from_translational_damping(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<ndim, ndim, double>& damping_matrix,
          const Core::LinAlg::Matrix<ndim, 1, double>& r_s,
          const Core::LinAlg::Matrix<ndim, 1, double>& vel_rel,
          const Core::LinAlg::Matrix<ndim, 1, double>& gamma,
          const Core::LinAlg::Matrix<ndim, ndim, double>& velbackgroundgrad,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
          double gp_weight) const;

      template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void evaluate_analytic_stiffmat_contributions_from_translational_damping(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<ndim, ndim, FAD>& damping_matrix,
          const Core::LinAlg::Matrix<ndim, 1, FAD>& r_s,
          const Core::LinAlg::Matrix<ndim, 1, FAD>& vel_rel,
          const Core::LinAlg::Matrix<ndim, 1, double>& gamma,
          const Core::LinAlg::Matrix<ndim, ndim, FAD>& velbackgroundgrad,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
          double gp_weight) const
      {
        // this is empty because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }

      /** \brief evaluate all contributions from thermal/stochastic forces
       *
       */
      template <typename T, unsigned int nnode, unsigned int vpernode, unsigned int ndim,
          unsigned int randompergauss>
      void evaluate_stochastic_forces(
          const Core::LinAlg::Matrix<ndim * vpernode * nnode, 1, T>& disp_totlag,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>&
              f_int);  //!< element internal force vector


      /** \brief evaluate contributions to element stiffness matrix from thermal/stochastic forces
       *
       */
      template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void evaluate_analytic_stiffmat_contributions_from_stochastic_forces(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<ndim, 1, double>& r_s,
          const Core::LinAlg::Matrix<ndim, 1, double>& randnumvec,
          const Core::LinAlg::Matrix<ndim, 1, double>& gamma,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
          double gp_weight) const;

      template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void evaluate_analytic_stiffmat_contributions_from_stochastic_forces(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<ndim, 1, FAD>& r_s,
          const Core::LinAlg::Matrix<ndim, 1, double>& randnumvec,
          const Core::LinAlg::Matrix<ndim, 1, double>& gamma,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i,
          const Core::LinAlg::Matrix<1, nnode * vpernode, double>& N_i_xi, double jacobifactor,
          double gp_weight) const
      {
        // this is empty because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }

      /** \brief evaluate all contributions from rotational damping moment/torque
       *
       */
      template <typename T, unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void evaluate_rotational_damping(
          const Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline,
          const std::vector<Core::LinAlg::Matrix<ndim, ndim, T>>& triad_mat_cp,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::Matrix<ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int);

      /** \brief evaluate contributions to element stiffness matrix from rotational damping moment
       *
       */
      template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
      void evaluate_analytic_stiffmat_contributions_from_rotational_damping(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, 1,
              double>& disp_totlag_centerline,
          const LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              double>& triad_intpol,
          const Core::LinAlg::Matrix<3, 1, double> theta_gp,
          const Core::LinAlg::Matrix<3, 1, double>& deltatheta_gp,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat_gp,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv_gp,
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, ndim,
              double>& v_theta_par_bar,
          const std::vector<Core::LinAlg::Matrix<ndim,
              ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, double>>& lin_theta_cp,
          const Core::LinAlg::Matrix<3, 1, double> moment_viscous, double gamma_polar, double dt,
          double xi_gp, double jacobifac_GPwgt) const;

      template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
      void evaluate_analytic_stiffmat_contributions_from_rotational_damping(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>&
              disp_totlag_centerline,
          const LargeRotations::TriadInterpolationLocalRotationVectors<BEAM3K_COLLOCATION_POINTS,
              FAD>& triad_intpol,
          const Core::LinAlg::Matrix<3, 1, FAD> theta_gp,
          const Core::LinAlg::Matrix<3, 1, FAD>& deltatheta_gp,
          const Core::LinAlg::Matrix<3, 3, FAD>& triad_mat_gp,
          const Core::LinAlg::Matrix<3, 3, FAD>& triad_mat_conv_gp,
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, ndim,
              FAD>& v_theta_par_bar,
          const std::vector<Core::LinAlg::Matrix<ndim,
              ndim * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, FAD>>& lin_theta_cp,
          const Core::LinAlg::Matrix<3, 1, FAD> moment_viscous, double gamma_polar, double dt,
          double xi_gp, double jacobifac_GPwgt) const
      {
        // this is empty because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }

      /** \brief pre-compute quantities required for linearization of rotational damping moment
       *
       */
      template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void pre_compute_terms_at_cp_for_analytic_stiffmat_contributions_from_rotational_damping(
          Core::LinAlg::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, double>&
              lin_theta,
          const Core::LinAlg::Matrix<1, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS,
              double>& L,
          const Core::LinAlg::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS,
              double>& N_s,
          const Core::LinAlg::Matrix<ndim, 1, double>& r_s, double abs_r_s,
          const Core::LinAlg::Matrix<4, 1, double>& Qref_conv) const;

      template <unsigned int nnode, unsigned int vpernode, unsigned int ndim>
      void pre_compute_terms_at_cp_for_analytic_stiffmat_contributions_from_rotational_damping(
          Core::LinAlg::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, FAD>&
              lin_theta,
          const Core::LinAlg::Matrix<1, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS, FAD>&
              L,
          const Core::LinAlg::Matrix<ndim, ndim * vpernode * nnode + BEAM3K_COLLOCATION_POINTS,
              FAD>& N_s,
          const Core::LinAlg::Matrix<ndim, 1, FAD>& r_s, FAD abs_r_s,
          const Core::LinAlg::Matrix<4, 1, double>& Qref_conv) const
      {
        // this is empty because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }

      //! compute (material) strain K
      template <typename T>
      void computestrain(const Core::LinAlg::Matrix<3, 1, T>& theta,
          const Core::LinAlg::Matrix<3, 1, T>& theta_deriv, Core::LinAlg::Matrix<3, 1, T>& K) const
      {
        Core::LinAlg::Matrix<3, 3, T> Tinv = Core::LargeRotations::tinvmatrix(theta);

        K.clear();
        K.multiply_tn(Tinv, theta_deriv);
      }

      //! calculate material stress resultants M,N from material strain resultants K, epsilon
      template <typename T>
      void straintostress(const Core::LinAlg::Matrix<3, 1, T>& Omega, const T& epsilon,
          const Core::LinAlg::Matrix<3, 3, T>& Cn, const Core::LinAlg::Matrix<3, 3, T>& Cm,
          Core::LinAlg::Matrix<3, 1, T>& M, T& f_par) const;

      //! Compute the material triad in case of the strong Kirchhoff (SK)beam formulation
      template <typename T>
      void compute_triad_sk(const T& phi, const Core::LinAlg::Matrix<3, 1, T>& r_s,
          const Core::LinAlg::Matrix<3, 3, T>& triad_ref,
          Core::LinAlg::Matrix<3, 3, T>& triad) const
      {
        Core::LinAlg::Matrix<3, 3, T> triad_bar(Core::LinAlg::Initialization::zero);

        // Compute triad_bar via SR mapping from triad_ref onto r_s
        Core::LargeRotations::calculate_sr_triads<T>(r_s, triad_ref, triad_bar);

        // Compute triad via relative rotation of triad_bar
        Core::LargeRotations::rotate_triad<T>(triad_bar, phi, triad);
      }

      template <typename T1, typename T2>
      void assemble_shapefunctions_l(Core::LinAlg::Matrix<1, BEAM3K_COLLOCATION_POINTS, T1>& L_i,
          Core::LinAlg::Matrix<1, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& L) const;

      template <typename T1, typename T2>
      void assemble_shapefunctions_nss(Core::LinAlg::Matrix<1, 4, T1>& N_i_xi,
          Core::LinAlg::Matrix<1, 4, T1>& N_i_xixi, double jacobi, double jacobi2,
          Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N_ss) const;

      template <typename T1, typename T2>
      void assemble_shapefunctions_ns(Core::LinAlg::Matrix<1, 4, T1>& N_i_xi, double jacobi,
          Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N_s) const;

      template <typename T1, typename T2>
      void assemble_shapefunctions_n(Core::LinAlg::Matrix<1, 4, T1>& N_i,
          Core::LinAlg::Matrix<3, 2 * 6 + BEAM3K_COLLOCATION_POINTS, T2>& N) const;

      //! update absolute values for primary Dof vector based on the given displacement vector
      template <unsigned int nnodecl, typename T>
      void update_disp_totlag(const std::vector<double>& disp,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag) const;

      //! update positions vectors and tangents at boundary nodes and triads at all CPs
      //  based on the given element displacement vector
      template <unsigned int nnodecl, typename T>
      void update_nodal_variables(
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline,
          std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
          std::vector<Core::LinAlg::Matrix<4, 1>>& Qref_new) const;

      //! extract those Dofs relevant for centerline-interpolation from total state vector
      template <unsigned int nnodecl, unsigned int vpernode, typename T>
      void extract_centerline_dof_values_from_element_state_vector(
          const Core::LinAlg::Matrix<3 * vpernode * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              dofvec,
          Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, T>& dofvec_centerline,
          bool add_reference_values = false) const;

      //! "add" reference values to displacement state vector (multiplicative in case of rotation
      //! pseudo vectors)
      template <unsigned int nnodecl, typename T>
      void add_ref_values_disp(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& dofvec) const;

      //! set positions at boundary nodes
      template <unsigned int nnodecl, typename T>
      void set_positions_at_boundary_nodes(
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline) const;

      //! set tangents and triads and reference triads in quaternion form at boundary nodes
      template <unsigned int nnodecl, typename T>
      void set_tangents_and_triads_and_reference_triads_at_boundary_nodes(
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline,
          std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
          std::vector<Core::LinAlg::Matrix<4, 1>>& Qref_new) const;

      //! set triads and reference triads in quaternion form at all CPs except boundary nodes
      template <unsigned int nnodecl, typename T>
      void set_triads_and_reference_triads_at_remaining_collocation_points(
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline,
          std::vector<Core::LinAlg::Matrix<3, 3, T>>& triad_mat_cp,
          std::vector<Core::LinAlg::Matrix<4, 1>>& Qref_new) const;

      //! set differentiation variables for automatic differentiation via FAD
      template <unsigned int nnodecl>
      void set_automatic_differentiation_variables(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, FAD>& disp_totlag) const;

      //! Pre-multiply trafo matrix if rotvec_==true: \tilde{\vec{f}_int}=\mat{T}^T*\vec{f}_int
      template <unsigned int nnodecl, typename T>
      void apply_rot_vec_trafo(
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>&
              disp_totlag_centerline,
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& f_int) const;

      //! Transform stiffness matrix in order to solve for multiplicative rotation vector increments
      template <unsigned int nnodecl, typename T>
      void transform_stiff_matrix_multiplicative(Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          const Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 1, T>& disp_totlag)
          const;

      //! lump mass matrix
      void lumpmass(Core::LinAlg::SerialDenseMatrix* emass);

      template <typename T>
      void calculate_clcurvature(Core::LinAlg::Matrix<3, 1, T>& r_s,
          Core::LinAlg::Matrix<3, 1, T>& r_ss, Core::LinAlg::Matrix<3, 1, T>& kappacl) const
      {
        // spinmatrix Sr' = r'x
        Core::LinAlg::Matrix<3, 3, T> Srs(Core::LinAlg::Initialization::zero);
        Core::LargeRotations::computespin(Srs, r_s);

        // cross-product r'xr''
        Core::LinAlg::Matrix<3, 1, T> Srsrss(Core::LinAlg::Initialization::zero);
        Srsrss.multiply(Srs, r_ss);
        T rsTrs = 0.0;

        for (int i = 0; i < 3; i++) rsTrs += r_s(i) * r_s(i);

        for (int i = 0; i < 3; i++)
        {
          kappacl(i) = Srsrss(i) / rsTrs;
        }

        return;
      }

      template <typename T>
      void computestrain_sk(const T& phi_s, const Core::LinAlg::Matrix<3, 1, T>& kappacl,
          const Core::LinAlg::Matrix<3, 3, T>& triadref,
          const Core::LinAlg::Matrix<3, 3, T>& triad_mat, Core::LinAlg::Matrix<3, 1, T>& K) const
      {
        Core::LinAlg::Matrix<1, 1, T> scalar_aux(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<3, 1, T> g1(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<3, 1, T> g2(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<3, 1, T> g3(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<3, 1, T> gref1(Core::LinAlg::Initialization::zero);
        T KR1 = 0.0;

        for (int i = 0; i < 3; i++)
        {
          g1(i) = triad_mat(i, 0);
          g2(i) = triad_mat(i, 1);
          g3(i) = triad_mat(i, 2);
          gref1(i) = triadref(i, 0);
        }

        scalar_aux.multiply_tn(kappacl, gref1);
        KR1 = -scalar_aux(0, 0);
        scalar_aux.clear();
        scalar_aux.multiply_tn(g1, gref1);
        KR1 = KR1 / (1.0 + scalar_aux(0, 0));
        K(0) = KR1 + phi_s;

        scalar_aux.clear();
        scalar_aux.multiply_tn(kappacl, g2);
        K(1) = scalar_aux(0, 0);

        scalar_aux.clear();
        scalar_aux.multiply_tn(kappacl, g3);
        K(2) = scalar_aux(0, 0);

        return;
      }

      void resize_class_variables(const int& n)
      {
        qrefconv_.resize(BEAM3K_COLLOCATION_POINTS);
        qrefnew_.resize(BEAM3K_COLLOCATION_POINTS);
        k0_.resize(n);
        jacobi_.resize(n);
        jacobi2_.resize(n);
        jacobi_cp_.resize(BEAM3K_COLLOCATION_POINTS);
        jacobi2_cp_.resize(BEAM3K_COLLOCATION_POINTS);
        qconvmass_.resize(n);
        qnewmass_.resize(n);
        wconvmass_.resize(n);
        wnewmass_.resize(n);
        aconvmass_.resize(n);
        anewmass_.resize(n);
        amodconvmass_.resize(n);
        amodnewmass_.resize(n);
        rttconvmass_.resize(n);
        rttnewmass_.resize(n);
        rttmodconvmass_.resize(n);
        rttmodnewmass_.resize(n);
        rtconvmass_.resize(n);
        rtnewmass_.resize(n);
        rconvmass_.resize(n);
        rnewmass_.resize(n);



        return;
      }

      void set_initial_dynamic_class_variables(const int& num,
          const Core::LinAlg::Matrix<3, 3>& triad_mat, const Core::LinAlg::Matrix<3, 1>& r)
      {
        qconvmass_[num].clear();
        qnewmass_[num].clear();
        rconvmass_[num].clear();
        rnewmass_[num].clear();
        wconvmass_[num].clear();
        wnewmass_[num].clear();
        aconvmass_[num].clear();
        anewmass_[num].clear();
        amodconvmass_[num].clear();
        amodnewmass_[num].clear();
        rtconvmass_[num].clear();
        rtnewmass_[num].clear();
        rttconvmass_[num].clear();
        rttnewmass_[num].clear();
        rttmodconvmass_[num].clear();
        rttmodnewmass_[num].clear();

        Core::LargeRotations::triadtoquaternion(triad_mat, qconvmass_[num]);
        qnewmass_[num] = qconvmass_[num];
        rconvmass_[num] = r;
        rnewmass_[num] = r;

        return;
      }

      // don't want = operator
      Beam3k& operator=(const Beam3k& old);

      template <int dim>
      void compute_triple_product(Core::LinAlg::Matrix<3, dim, FAD>& mat1,
          Core::LinAlg::Matrix<3, 1, FAD>& vec1, Core::LinAlg::Matrix<3, 1, FAD>& vec2,
          Core::LinAlg::Matrix<dim, 1, FAD>& vec_out) const
      {
        Core::LinAlg::Matrix<3, 3, FAD> auxmatrix1(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<3, 1, FAD> auxvec1(Core::LinAlg::Initialization::zero);
        Core::LargeRotations::computespin(auxmatrix1, vec1);
        auxvec1.multiply(auxmatrix1, vec2);
        vec_out.multiply_tn(mat1, auxvec1);

        return;
      }

      /** \brief compute interpolated velocity vector from element state vector
       *
       */
      template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
      void calc_velocity(
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& velocity_dofvec,
          const Core::LinAlg::Matrix<1, vpernode * nnodecl, double>& N_i,
          Core::LinAlg::Matrix<ndim, 1, double>& velocity,
          const Core::LinAlg::Matrix<ndim, 1, double>& position, int gausspoint_index) const;

      /** \brief compute interpolated velocity vector if Fad is used
       *
       */
      template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim>
      void calc_velocity(
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& velocity_dofvec,
          const Core::LinAlg::Matrix<1, vpernode * nnodecl, double>& N_i,
          Core::LinAlg::Matrix<ndim, 1, FAD>& velocity,
          const Core::LinAlg::Matrix<ndim, 1, FAD>& position, int gausspoint_index);

      /** \brief compute discrete strain variations v_thetaperp
       *
       */
      template <unsigned int nnodecl, typename T>
      void calc_v_thetaperp(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>& v_thetaperp,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>& N_s,
          const Core::LinAlg::Matrix<3, 1, T>& r_s, T abs_r_s) const;

      /** \brief compute discrete strain variations v_thetapartheta
       *
       */
      template <unsigned int nnodecl, typename T>
      void calc_v_thetapartheta(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS, 3, T>& v_thetapartheta,
          const Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, T>& L,
          const Core::LinAlg::Matrix<3, 1, T>& r_s, T abs_r_s) const;

      /** \brief compute discrete strain increments v_lin_thetaperp
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_thetaperp(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_thetaperp,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_s, double abs_r_s) const;

      /** \brief compute discrete strain increments v_lin_thetapar
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_thetapar(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_thetapar,
          const Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& g_1,
          const Core::LinAlg::Matrix<3, 1, double>& g_1_bar, double abs_r_s) const;

      /** \brief compute linearization of scaled tangent vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_tangent_tilde(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>&
              lin_tangent_tilde,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s) const;

      /** \brief compute linearization of first arc-length derivative of scaled tangent vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_tangent_tilde_s(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>&
              lin_tangent_tilde_s,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
          const Core::LinAlg::Matrix<3, 1, double>& g_1,
          const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_ss, double abs_r_s) const;

      /** \brief compute linearization of first base vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_g_1(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_g_1,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s) const;

      /** \brief compute linearization of first arc-length derivative of first base vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_g_1_s(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_g_1_s,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
          const Core::LinAlg::Matrix<3, 1, double>& g_1,
          const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_ss, double abs_r_s) const;

      /** \brief compute linearization of v_epsilon
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_v_epsilon(Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
                                  6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_epsilon,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s) const;

      /** \brief compute linearization of moment resultant
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_moment_resultant(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>&
              lin_moment_resultant,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>&
              lin_theta_s,
          const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment,
          const Core::LinAlg::Matrix<3, 3, double>& cm) const;

      /** \brief compute linearization of inertia moment
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_moment_inertia(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>&
              lin_moment_inertia,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv,
          const Core::LinAlg::Matrix<3, 1, double>& deltatheta,
          const Core::LinAlg::Matrix<3, 1, double>& angular_velocity_material,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
          const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment,
          const Core::LinAlg::Matrix<3, 3, double>& C_rho, double lin_prefactor_acc,
          double lin_prefactor_vel) const;

      /** \brief compute linearization of moment from rotational damping
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_moment_viscous(
          Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>&
              lin_moment_viscous,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat,
          const Core::LinAlg::Matrix<3, 3, double>& triad_mat_conv,
          const Core::LinAlg::Matrix<3, 1, double>& deltatheta,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_theta,
          const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment, double gamma_polar,
          double dt) const;

      /** \brief compute linearization of v_theta_perp multiplied with moment vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_v_thetaperp_moment(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetaperp_moment,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s,
          const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment) const;

      /** \brief compute linearization of v_theta_perp_s multiplied with moment vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_v_thetaperp_s_moment(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetaperp_s_moment,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
          const Core::LinAlg::Matrix<3, 1, double>& g_1,
          const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_ss, double abs_r_s,
          const Core::LinAlg::Matrix<3, 3, double>& spinmatrix_of_moment) const;

      /** \brief compute linearization of v_theta_par multiplied with moment vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_v_thetapar_moment(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetapar_moment,
          Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 1, double>& g_1, double abs_r_s,
          const Core::LinAlg::Matrix<3, 1, double>& moment) const;

      /** \brief compute linearization of v_theta_par_s multiplied with moment vector
       *
       */
      template <unsigned int nnodecl>
      void calc_lin_v_thetapar_s_moment(
          Core::LinAlg::Matrix<6 * nnodecl + BEAM3K_COLLOCATION_POINTS,
              6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& lin_v_thetapar_s_moment,
          Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L,
          Core::LinAlg::Matrix<1, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& L_s,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_s,
          const Core::LinAlg::Matrix<3, 6 * nnodecl + BEAM3K_COLLOCATION_POINTS, double>& N_ss,
          const Core::LinAlg::Matrix<3, 1, double>& g_1,
          const Core::LinAlg::Matrix<3, 1, double>& g_1_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_s,
          const Core::LinAlg::Matrix<3, 1, double>& r_ss, double abs_r_s,
          const Core::LinAlg::Matrix<3, 1, double>& moment) const;

     private:
      // Variables

      //! bool storing whether automatic differentiation shall be used for this element evaluation
      bool use_fad_;

      //! variable saving whether element has already been initialized (then isinit_ == true)
      bool isinit_;

      //! current (non-unit) tangent vectors at the two boundary nodes
      std::vector<Core::LinAlg::Matrix<3, 1>> t_;
      // Variables needed by all local triads
      //! Matrix holding pseudo rotation vectors describing the material triads in the initial
      //! configuration at each node
      std::vector<Core::LinAlg::Matrix<3, 1>> theta0_;
      //! quaternion describing the nodal reference triads (for the case BEAM3EK_ROT ==false) of the
      //! converged configuration of the last time step
      std::vector<Core::LinAlg::Matrix<4, 1>> qrefconv_;
      //! quaternion describing the nodal reference triads (for the case BEAM3EK_ROT ==false) of the
      //! current configuration
      std::vector<Core::LinAlg::Matrix<4, 1>> qrefnew_;
      //! Matrix with the material curvature in the initial configuration at each gp
      std::vector<Core::LinAlg::Matrix<3, 1>> k0_;

      //! Length of the element
      double length_;
      //! Jacobi determinant of the element at the Gauss points
      std::vector<double> jacobi_;
      //! Additional jacobi factor appearing in the second derivatives (required in case of strong
      //! Kirchhoff constraint)
      std::vector<double> jacobi2_;
      //! Jacobi determinant at collocation points
      std::vector<double> jacobi_cp_;
      //! Additional jacobi factor appearing in the second derivatives (required in linearization)
      std::vector<double> jacobi2_cp_;
      //! bool indicating, if the DoFs at the element boundary are described by rotation vectors or
      //! tangent vectors plus relative angle
      bool rotvec_;
      //! bool indicating, if Kirchhoff constraint is enforced in a strong (weakkirchhoff_==false)
      //! or weak manner (weakkirchhoff_==true)
      bool weakkirchhoff_;
      //! internal energy
      double eint_;
      //! kinetic energy
      double ekin_;
      //! temporarily storing the rot_damp_stiffness matrix for use in the PTC scaling operator
      Core::LinAlg::SerialDenseMatrix stiff_ptc_;


      //******************************Begin: Class variables required for time
      // integration**************************************************//
      //! triads at Gauss points for exact integration in quaternion at the end of the preceding
      //! time step (required for computation of angular velocity)
      std::vector<Core::LinAlg::Matrix<4, 1>> qconvmass_;
      //! current triads at Gauss points for exact integration in quaternion (required for
      //! computation of angular velocity)
      std::vector<Core::LinAlg::Matrix<4, 1>> qnewmass_;
      //! spatial angular velocity vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> wconvmass_;
      //! current spatial angular velocity vector at Gauss points for exact integration (required
      //! for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> wnewmass_;
      //! spatial angular acceleration vector at Gauss points for exact integration at the end of
      //! the preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> aconvmass_;
      //! current spatial angular acceleration vector at Gauss points for exact integration
      //! (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> anewmass_;
      //! modified spatial angular acceleration vector (according to gen-alpha time integration) at
      //! Gauss points for exact integration at the end of the preceding time step (required for
      //! computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> amodconvmass_;
      //! current modified spatial angular acceleration vector (according to gen-alpha time
      //! integration) at Gauss points for exact integration (required for computation of inertia
      //! terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> amodnewmass_;
      //! translational acceleration vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttconvmass_;
      //! current translational acceleration vector at Gauss points for exact integration (required
      //! for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttnewmass_;
      //! modified translational acceleration vector at Gauss points for exact integration at the
      //! end of the preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttmodconvmass_;
      //! modified current translational acceleration vector at Gauss points for exact integration
      //! (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttmodnewmass_;
      //! translational velocity vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rtconvmass_;
      //! current translational velocity vector at Gauss points for exact integration (required for
      //! computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rtnewmass_;
      //! translational displacement vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rconvmass_;
      //! current translational displacement vector at Gauss points for exact integration (required
      //! for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rnewmass_;
      //******************************End: Class variables required for time
      // integration**************************************************//


      //! strain resultant values at GPs
      std::vector<double> axial_strain_gp_;

      std::vector<double> twist_gp_;
      std::vector<double> curvature_2_gp_;
      std::vector<double> curvature_3_gp_;

      //! stress resultant values at GPs
      std::vector<double> axial_force_gp_;

      std::vector<double> torque_gp_;
      std::vector<double> bending_moment_2_gp_;
      std::vector<double> bending_moment_3_gp_;
    };

    // << operator
    std::ostream& operator<<(std::ostream& os, const Core::Elements::Element& ele);

  }  // namespace Elements

}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
