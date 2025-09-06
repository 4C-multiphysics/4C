// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_REISSNER_HPP
#define FOUR_C_BEAM3_REISSNER_HPP


#include "4C_config.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_spatial_discretization_utils.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

using FAD = Sacado::Fad::DFad<double>;

// forward declaration ...
namespace Core::LinAlg
{
  class SerialDenseMatrix;
}

namespace LargeRotations
{
  template <unsigned int numnodes, typename T>
  class TriadInterpolationLocalRotationVectors;
}

// #define BEAM3RCONSTSTOCHFORCE  //Flag in order to hold stochastic forces constant over the
// element
//  length and to only provide random numbers for the 3 translational DoFs (needed in order to
//  compare with beam3eb)

namespace Discret
{
  namespace Elements
  {
    class Beam3rType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "Beam3rType"; }

      static Beam3rType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(Core::Nodes::Node& actnode,
          const double* x0, const int numdof, const int dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static Beam3rType instance_;
    };

    /*!
    \brief 3D nonlinear Reissner beam element implemented according to the following sources:
    Jelenic, Crisfield, 1999, "Geometrically exact 3D beam theory: implementations of a
    strain-invariant finite element for statics and dynamics", Crisfield, Jelenic, 1999,
    "Objectivity of strain measures in the geometrically exact three dimensional beam theory and its
    finite element implementation", Romero, 2004, "The interpolation of rotations and its
    application to finite element models of geometrically exact rods", Crisfield, 2003, "Non-linear
    Finite Element Analysis of Solids and Structures", Volume 2

    */
    class Beam3r : public Beam3Base
    {
     public:
      //! purposes of numerical integration
      enum IntegrationPurpose
      {
        res_elastic_force,
        res_elastic_moment,
        res_inertia,
        res_damp_stoch,
        neumann_lineload
      };

     public:
      //! @name Friends
      friend class Beam3rType;

      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id    (in): A globally unique element id
      \param etype (in): Type of element
      \param owner (in): owner processor of the element
      */
      Beam3r(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      Beam3r(const Beam3r& old);



      // don't want = operator
      Beam3r& operator=(const Beam3r& old);

      /*!
      \brief Deep copy this instance of Beam3r and return pointer to the copy

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
        return Beam3rType::instance().unique_par_object_id();
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

      Core::Elements::ElementType& element_type() const override { return Beam3rType::instance(); }

      //@}

      /*!
      \brief Return number of lines to this element
      */
      int num_line() const override { return 1; }

      /*!
      \brief Get vector of std::shared_ptrs to the lines of this element
      */
      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      /** \brief Get number of nodes used for centerline interpolation
       *
       */
      inline int num_centerline_nodes() const override
      {
        return centerline_hermite_ ? 2 : this->num_node();
      }

      /** \brief find out whether given node is used for centerline interpolation
       *
       */
      inline bool is_centerline_node(const Core::Nodes::Node& node) const override
      {
        if (!centerline_hermite_ or node.id() == this->nodes()[0]->id() or
            node.id() == this->nodes()[1]->id())
          return true;
        else
          return false;
      }

      /*!
      \brief Get number of degrees of freedom of a single node
      */
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        /* note: this is not necessarily the number of DOF assigned to this node by the
         * discretization finally, but only the number of DOF requested for this node by this
         * element; the discretization will finally assign the maximal number of DOF to this node
         * requested by any element connected to this node*/
        if (!centerline_hermite_)
          return 6;
        else
        {
          /* in case of Hermite centerline interpolation (so far always 3rd order = 2nodes), we have
           * 6 translational DOFs for the first two nodes and additionally 3 rotational DOFs for
           * each node */
          int dofpn_aux = 0;

          if (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id())
          {
            dofpn_aux = 9;
          }
          else
          {
            dofpn_aux = 3;
          }

          const int dofpn = dofpn_aux;
          return dofpn;
        }
      }

      /*!
      \brief Get number of degrees of freedom per element not including nodal degrees of freedom
      */
      int num_dof_per_element() const override { return 0; }

      /*!
      \brief Print this element
      */
      void print(std::ostream& os) const override;

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
          const Core::LinAlg::SerialDenseVector& force) const override
      {
        const unsigned int vpernode = this->hermite_centerline_interpolation() ? 2 : 1;
        const unsigned int nnodecl = this->num_centerline_nodes();
        const unsigned int nnodetriad = this->num_node();

        // safety check
        if ((unsigned int)stiffmat.numRows() != 3 * vpernode * nnodecl + 3 * nnodetriad or
            (unsigned int) stiffmat.numCols() != 3 * vpernode * nnodecl + 3 * nnodetriad)
          FOUR_C_THROW("size mismatch! expected {}x{} matrix and got {}x{}",
              3 * vpernode * nnodecl + 3 * nnodetriad, 3 * vpernode * nnodecl + 3 * nnodetriad,
              stiffmat.numRows(), stiffmat.numCols());

        // nothing to do here since this term vanishes for Beam3r
        stiffmat.putScalar(0.0);
      }

      /** \brief get generalized interpolation matrix which yields the increments of the position
       * and orientation at xi \in [-1,1] if multiplied with the vector of primary DoF increments
       *
       */
      void get_generalized_interpolation_matrix_increments_at_xi(
          Core::LinAlg::SerialDenseMatrix& Iinc, const double& xi,
          const std::vector<double>& disp) const override;

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

      /*!
      \brief get tangent of centerline at all nodes in reference configuration
      */
      std::vector<Core::LinAlg::Matrix<3, 1>> tref() const { return Tref_; }

      /*!
      \brief Get jacobiGP_ factor of first Gauss point for under-integration (constant over element
      length for linear Lagrange interpolation)
      */
      const double& get_jacobi() const { return jacobi_gp_elastf_[0]; }

      /** \brief get Jacobi factor ds/dxi(xi) at xi \in [-1;1]
       *
       */
      double get_jacobi_fac_at_xi(const double& xi) const override;

      /*!
      \brief Get maximal bending curvature occurring in this element
      */
      const double& get_kappa_max() const { return kmax_; }

      /** \brief Get material cross-section deformation measures, i.e. strain resultants
       *
       */
      inline void get_material_strain_resultants_at_all_gps(std::vector<double>& axial_strain_GPs,
          std::vector<double>& shear_strain_2_GPs, std::vector<double>& shear_strain_3_GPs,
          std::vector<double>& twist_GPs, std::vector<double>& curvature_2_GPs,
          std::vector<double>& curvature_3_GPs) const override
      {
        axial_strain_GPs = axial_strain_gp_elastf_;
        shear_strain_2_GPs = shear_strain_2_gp_elastf_;
        shear_strain_3_GPs = shear_strain_3_gp_elastf_;

        twist_GPs = twist_gp_elastm_;
        curvature_2_GPs = curvature_2_gp_elastm_;
        curvature_3_GPs = curvature_3_gp_elastm_;
      }

      /** \brief Get spatial cross-section stress resultants
       *
       */
      inline void get_spatial_stress_resultants_at_all_gps(
          std::vector<double>& spatial_axial_force_GPs,
          std::vector<double>& spatial_shear_force_2_GPs,
          std::vector<double>& spatial_shear_force_3_GPs, std::vector<double>& spatial_torque_GPs,
          std::vector<double>& spatial_bending_moment_2_GPs,
          std::vector<double>& spatial_bending_moment_3_GPs) const override
      {
        get_spatial_forces_at_all_gps(
            spatial_axial_force_GPs, spatial_shear_force_2_GPs, spatial_shear_force_3_GPs);

        get_spatial_moments_at_all_gps(
            spatial_torque_GPs, spatial_bending_moment_2_GPs, spatial_bending_moment_3_GPs);
      }

      /** \brief Get spatial cross-section stress resultants
       *
       */
      inline void get_spatial_forces_at_all_gps(std::vector<double>& spatial_axial_force_GPs,
          std::vector<double>& spatial_shear_force_2_GPs,
          std::vector<double>& spatial_shear_force_3_GPs) const override
      {
        spatial_axial_force_GPs = spatial_x_force_gp_elastf_;
        spatial_shear_force_2_GPs = spatial_y_force_2_gp_elastf_;
        spatial_shear_force_3_GPs = spatial_z_force_3_gp_elastf_;
      }

      /** \brief Get spatial cross-section stress resultants
       *
       */
      inline void get_spatial_moments_at_all_gps(std::vector<double>& spatial_torque_GPs,
          std::vector<double>& spatial_bending_moment_2_GPs,
          std::vector<double>& spatial_bending_moment_3_GPs) const override
      {
        spatial_torque_GPs = spatial_x_moment_gp_elastm_;
        spatial_bending_moment_2_GPs = spatial_y_moment_2_gp_elastm_;
        spatial_bending_moment_3_GPs = spatial_z_moment_3_gp_elastm_;
      }

      /** \brief Get material cross-section stress resultants
       *
       */
      inline void get_material_stress_resultants_at_all_gps(
          std::vector<double>& material_axial_force_GPs,
          std::vector<double>& material_shear_force_2_GPs,
          std::vector<double>& material_shear_force_3_GPs, std::vector<double>& material_torque_GPs,
          std::vector<double>& material_bending_moment_2_GPs,
          std::vector<double>& material_bending_moment_3_GPs) const override
      {
        material_axial_force_GPs = material_axial_force_gp_elastf_;
        material_shear_force_2_GPs = material_shear_force_2_gp_elastf_;
        material_shear_force_3_GPs = material_shear_force_3_gp_elastf_;

        material_torque_GPs = material_torque_gp_elastm_;
        material_bending_moment_2_GPs = material_bending_moment_2_gp_elastm_;
        material_bending_moment_3_GPs = material_bending_moment_3_gp_elastm_;
      }

      /** \brief get access to the reference length
       *
       */
      inline double ref_length() const override { return reflength_; }

      /*!
      \brief Get initial nodal rotation vectors
      */
      const std::vector<Core::LinAlg::Matrix<3, 1>>& initial_nodal_rot_vecs() const
      {
        return theta0node_;
      }

      /*!
      \brief Set bool indicating Hermite centerline interpolation
      */
      void set_centerline_hermite(const bool yesno);

      //! computes the number of different random numbers required in each time step for generation
      //! of stochastic forces
      int how_many_random_numbers_i_need() const override;

      //! sets up all geometric parameters (considering current position as reference configuration)
      /* nnodetriad: number of nodes used for interpolation of triad field
       * nnodecl: number of nodes used for interpolation of centerline
       * vpernode: interpolated values per centerline node (1: value (i.e. Lagrange), 2: value +
       * derivative of value (i.e. Hermite))*/
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void set_up_reference_geometry(
          const std::vector<double>& xrefe, const std::vector<double>& rotrefe);

      //@}

      //! @name Construction

      /*!
      \brief Read input for this element
      */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;

      //@}


      //! @name Evaluation methods


      /*!
      \brief Evaluate this element

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
      template <unsigned int nnode>
      void evaluate_ptc(Teuchos::ParameterList& params, Core::LinAlg::SerialDenseMatrix& elemat1);

      //! calculation of nonlinear stiffness and mass matrix
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void calc_internal_and_inertia_forces_and_stiff(
          Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
          std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
          Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseVector* inertia_force);

      //@}

      /** \brief add indices of those DOFs of a given node that are positions
       *
       */
      inline void position_dof_indices(
          std::vector<int>& posdofs, const Core::Nodes::Node& node) const override
      {
        if ((not centerline_hermite_) or
            (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id()))
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
        if (centerline_hermite_ and
            (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id()))
        {
          tangdofs.push_back(6);
          tangdofs.push_back(7);
          tangdofs.push_back(8);
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
        if ((not centerline_hermite_) or
            (node.id() == this->nodes()[0]->id() or node.id() == this->nodes()[1]->id()))
        {
          rotvecdofs.push_back(3);
          rotvecdofs.push_back(4);
          rotvecdofs.push_back(5);
        }
        else
        {
          rotvecdofs.push_back(0);
          rotvecdofs.push_back(1);
          rotvecdofs.push_back(2);
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
        return;
      }

      /** \brief add indices of those DOFs of a given node that represent norm of tangent vector
       *         (additive, e.g. in case of beam3k with rotvec=true)
       *
       */
      inline void tangent_length_dof_indices(
          std::vector<int>& tangnormdofs, const Core::Nodes::Node& node) const override
      {
        return;
      }

      /** \brief get element local indices of those Dofs that are used for centerline interpolation
       *
       */
      inline void centerline_dof_indices_of_element(
          std::vector<unsigned int>& centerlinedofindices) const override
      {
        // nnodecl: number of nodes used for interpolation of centerline
        // vpernode: number of interpolated values per centerline node (1: value (i.e. Lagrange), 2:
        // value + derivative of value (i.e. Hermite))
        const unsigned int vpernode = this->hermite_centerline_interpolation() ? 2 : 1;
        const unsigned int nnodecl = this->num_centerline_nodes();

        const unsigned int dofperclnode = 3 * vpernode;
        const unsigned int dofpertriadnode = 3;
        const unsigned int dofpercombinode =
            dofperclnode +
            dofpertriadnode;  // if node is used for centerline and triad interpolation

        centerlinedofindices.resize(dofperclnode * nnodecl, 0);

        for (unsigned int inodecl = 0; inodecl < nnodecl; ++inodecl)
        {
          // position Dofs: always node-local indices 0,1,2
          for (unsigned int idof = 0; idof < 3; ++idof)
            centerlinedofindices[dofperclnode * inodecl + idof] = dofpercombinode * inodecl + idof;
          // tangent Dofs (if needed): node-local indices 6,7,8
          for (unsigned int idof = 6; idof < dofpercombinode; ++idof)
            centerlinedofindices[dofperclnode * inodecl + idof - 3] =
                dofpercombinode * inodecl + idof;
        }
      }

      /** \brief extract values for those Dofs relevant for centerline-interpolation from total
       * state vector
       *
       */
      void extract_centerline_dof_values_from_element_state_vector(
          const std::vector<double>& dofvec, std::vector<double>& dofvec_centerline,
          bool add_reference_values = false) const override;

      //! get internal (elastic) energy of element
      double get_internal_energy() const override { return eint_; };

      //! get kinetic energy of element
      double get_kinetic_energy() const override { return ekin_; };

     private:
      //! @name Internal calculation methods

      /** \brief compute internal (i.e. elastic) force, stiffness matrix, inertia force and mass
       * matrix
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void calc_internal_and_inertia_forces_and_stiff(Teuchos::ParameterList& params,
          std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::SerialDenseMatrix* massmatrix, Core::LinAlg::SerialDenseVector* force,
          Core::LinAlg::SerialDenseVector* inertia_force);

      /** \brief compute internal (i.e. elastic) force and stiffness matrix
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode, typename T>
      void calc_internal_force_and_stiff(
          const Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, T>& disp_totlag_centerline,
          const std::vector<Core::LinAlg::Matrix<4, 1, T>>& Qnode,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::Matrix<3 * vpernode * nnodecl + 3 * nnodetriad, 1, T>& internal_force);

      /** \brief calculate inertia force and mass matrix
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void calc_inertia_force_and_mass_matrix(
          const Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
          const std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
          Core::LinAlg::SerialDenseMatrix* massmatrix,
          Core::LinAlg::SerialDenseVector* inertia_force);

      /** \brief get Jacobi factor ds/dxi(xi) at xi \in [-1;1]
       *
       */
      template <unsigned int nnodecl, unsigned int vpernode>
      double get_jacobi_fac_at_xi(const double& xi) const
      {
        /* ||dr_0/ds(xi)||=1 because s is arc-length parameter => ||dr_0/dxi(xi)|| * dxi/ds(xi) = 1
         * => JacobiFac(xi) = ds/dxi(xi) = ||dr_0/dxi(xi)|| */
        Core::LinAlg::Matrix<3, 1> r0_xi;
        Core::LinAlg::Matrix<1, vpernode * nnodecl, double> N_i_xi;
        Core::LinAlg::Matrix<3 * nnodecl * vpernode, 1> disp_centerline_ref;

        // fill disp_centerline_ref with reference nodal centerline positions and tangents
        for (unsigned int node = 0; node < nnodecl; node++)
        {
          for (unsigned int dim = 0; dim < 3; ++dim)
          {
            disp_centerline_ref(3 * vpernode * node + dim) = nodes()[node]->x()[dim];
            if (vpernode == 2)
              disp_centerline_ref(3 * vpernode * node + 3 + dim) = (Tref_[node])(dim);
          }
        }

        Discret::Utils::Beam::evaluate_shape_function_derivs_at_xi<nnodecl, vpernode>(
            xi, N_i_xi, this->shape(), this->ref_length());
        this->calc_r_xi<nnodecl, vpernode, double>(disp_centerline_ref, N_i_xi, r0_xi);

        return r0_xi.norm2();
      }

      /*!
       \brief Get triad (three unit base vectors) at given parameter coordinate xi
      */
      template <unsigned int nnodetriad, typename T>
      void get_triad_at_xi(Core::LinAlg::Matrix<3, 3, T>& triad, const double& xi,
          const std::vector<Core::LinAlg::Matrix<4, 1, T>>& Qnode) const
      {
        // create object of triad interpolation scheme
        std::shared_ptr<LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, T>>
            triad_interpolation_scheme_ptr = std::make_shared<
                LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, T>>();

        // reset scheme with nodal quaternions
        triad_interpolation_scheme_ptr->reset(Qnode);

        triad_interpolation_scheme_ptr->get_interpolated_triad_at_xi(triad, xi);
      }

      /** \brief get generalized interpolation matrix which yields the variation of the position and
       *         orientation at xi \in [-1,1] if multiplied with the vector of primary DoF
       * variations
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void get_generalized_interpolation_matrix_variations_at_xi(
          Core::LinAlg::Matrix<6, 3 * vpernode * nnodecl + 3 * nnodetriad, double>& Ivar,
          const double& xi) const;

      /** \brief get generalized interpolation matrix which yields the increments of the position
       * and orientation at xi \in [-1,1] if multiplied with the vector of primary DoF increments
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void get_generalized_interpolation_matrix_increments_at_xi(
          Core::LinAlg::Matrix<6, 3 * vpernode * nnodecl + 3 * nnodetriad, double>& Iinc,
          const double& xi, const std::vector<double>& disp) const;

      //! compute material curvature at certain Gauss point according to Crisfield 1999, eq. (4.9)
      template <typename T>
      void compute_k(const Core::LinAlg::Matrix<3, 1, T>& Psi_l,
          const Core::LinAlg::Matrix<3, 1, T>& Psi_l_s,
          const Core::LinAlg::Matrix<3, 1, double>& Kref, Core::LinAlg::Matrix<3, 1, T>& K) const
      {
        K.clear();

        /* Calculation of material curvature vector according to Crisfield 1999, eq. (4.2) (this
         * equation has been derived for a different beam element formulation but is also valid for
         * the element type considered here),
         * or Jelenic 1999, paragraph on page 153 between NOTE 5 and NOTE 6*/
        Core::LinAlg::Matrix<3, 3, T> Tinv(Core::LinAlg::Initialization::zero);
        Tinv = Core::LargeRotations::tinvmatrix<T>(Psi_l);
        // It is important to use the transposed matrix Tinv^T instead of Tinv (these two only
        // differ in one of three terms)
        K.multiply_tn(Tinv, Psi_l_s);

        // mechanically relevant curvature is current curvature minus curvature in reference
        // position
        for (int i = 0; i < 3; ++i) K(i) -= Kref(i);
      }

      //! compute convected strain at certain Gauss point with according to Crisfield 1999, eq.
      //! (3.4)
      template <typename T>
      void compute_gamma(const Core::LinAlg::Matrix<3, 1, T>& r_s,
          const Core::LinAlg::Matrix<3, 3, T>& Lambda,
          const Core::LinAlg::Matrix<3, 1, double>& Gammaref,
          Core::LinAlg::Matrix<3, 1, T>& Gamma) const
      {
        Gamma.clear();

        // convected strain gamma according to Crisfield 1999, eq. (3.4)
        Gamma.multiply_tn(Lambda, r_s);

        /* In contrary to Crisfield 1999, eq. (3.4), the current implementation allows for initial
         * values of the vector gammaref which has also a second and a third component, i.e. it
         * allows for initial shear deformation. This is the case, when the initial triad at the
         * evaluation point is not parallel to the centerline tangent vector at this point. The
         * geometrically exact beam theory does in general allow for such initial triads if they are
         * considered consistently in the reference strains. While it is standard to assume
         * vanishing initial shear strains in the space-continuous setting, the possibility of
         * initial shear strains might be advantageous for the spatially discretized problem: For
         * curved initial geometries, the nodal triad had to be determined such that the resulting
         * interpolated triad at the Gauss point would be tangential to the centerline tangent at
         * this point resulting from the centerline interpolation. In order to avoid this additional
         * effort and to allow for an independent prescription of the nodal triads (e.g. prescribed
         * by an analytical geometry definition), the approach of considering arbitrary initial
         * shear angles at the Gauss points is applied here.*/
        for (int i = 0; i < 3; ++i) Gamma(i) -= Gammaref(i);
      }

      //! push forward material stress vector and constitutive matrix to their spatial counterparts
      //! by rotation matrix Lambda
      template <typename T>
      void pushforward(const Core::LinAlg::Matrix<3, 3, T>& Lambda,
          const Core::LinAlg::Matrix<3, 1, T>& stress_mat,
          const Core::LinAlg::Matrix<3, 3, T>& C_mat, Core::LinAlg::Matrix<3, 1, T>& stress_spatial,
          Core::LinAlg::Matrix<3, 3, T>& c_spatial) const;

      /** \brief compute analytic linearization (i.e. stiffness matrix) of element force vector
       *         resulting from internal elastic forces at a certain Gauss point
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void calc_stiffmat_analytic_force_contributions(Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<3, 1, double>& stressn,
          const Core::LinAlg::Matrix<3, 3, double>& cn,
          const Core::LinAlg::Matrix<3, 3, double>& r_s_hat,
          const LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>&
              triad_intpol,
          const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i,
          const Core::LinAlg::Matrix<1, vpernode * nnodecl, double>& H_i_xi, const double wgt,
          const double jacobifactor) const;

      /** \brief compute analytic linearization (i.e. stiffness matrix) of element force vector
       *         resulting from internal elastic forces at a certain Gauss point
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      inline void calc_stiffmat_analytic_force_contributions(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>& stressn,
          const Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>& cn,
          const Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>& r_s_hat,
          const LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad,
              Sacado::Fad::DFad<double>>& triad_intpol,
          const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i,
          const Core::LinAlg::Matrix<1, vpernode * nnodecl, double>& H_i_xi, const double wgt,
          const double jacobifactor) const
      {
        // this is a dummy because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }

      /** \brief compute analytic linearization (i.e. stiffness matrix) of element force vector
       *         resulting from internal elastic moments at a certain Gauss point
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void calc_stiffmat_analytic_moment_contributions(Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<3, 1, double>& stressm,
          const Core::LinAlg::Matrix<3, 3, double>& cm,
          const LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad, double>&
              triad_intpol,
          const Core::LinAlg::Matrix<3, 1, double>& Psi_l,
          const Core::LinAlg::Matrix<3, 1, double>& Psi_l_s,
          const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i,
          const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i_xi, const double wgt,
          const double jacobifactor) const;

      /** \brief compute analytic linearization (i.e. stiffness matrix) of element force vector
       *         resulting from internal elastic moments at a certain Gauss point
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      inline void calc_stiffmat_analytic_moment_contributions(
          Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>& stressm,
          const Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>& cm,
          const LargeRotations::TriadInterpolationLocalRotationVectors<nnodetriad,
              Sacado::Fad::DFad<double>>& triad_intpol,
          const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>& Psi_l,
          const Core::LinAlg::Matrix<3, 1, Sacado::Fad::DFad<double>>& Psi_l_s,
          const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i,
          const Core::LinAlg::Matrix<1, nnodetriad, double>& I_i_xi, const double wgt,
          const double jacobifactor) const
      {
        // this is a dummy because in case that the pre-calculated values are of type Fad,
        // we use automatic differentiation and consequently there is no need for analytic stiffmat
      }

      /** \brief compute linearization (i.e. stiffness matrix) of a given force vector
       *         using automatic differentiation based on Sacado::Fad package
       *
       */
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void calc_stiffmat_automatic_differentiation(Core::LinAlg::SerialDenseMatrix& stiffmatrix,
          const std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
          Core::LinAlg::Matrix<3 * vpernode * nnodecl + 3 * nnodetriad, 1,
              Sacado::Fad::DFad<double>>
              forcevec) const;

      //! calculation of thermal (i.e. stochastic) and damping forces according to Brownian dynamics
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void calc_brownian_forces_and_stiff(Teuchos::ParameterList& params, std::vector<double>& vel,
          std::vector<double>& disp, Core::LinAlg::SerialDenseMatrix* stiffmatrix,
          Core::LinAlg::SerialDenseVector* force);

      //! update (total) displacement vector and set nodal triads (as quaternions)
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode, typename T>
      void update_disp_tot_lag_and_nodal_triads(const std::vector<double>& disp,
          Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, T>& disp_totlag_centerline,
          std::vector<Core::LinAlg::Matrix<4, 1, T>>& Q_i);

      //! set differentiation variables for automatic differentiation via FAD
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode>
      void set_automatic_differentiation_variables(
          Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, FAD>& disp_totlag_centerline,
          std::vector<Core::LinAlg::Matrix<4, 1, FAD>>& Q_i) const;

      //! extract those Dofs relevant for centerline-interpolation from total state vector
      template <unsigned int nnodecl, unsigned int vpernode, typename T>
      void extract_centerline_dof_values_from_element_state_vector(
          const std::vector<double>& dofvec,
          Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, T>& dofvec_centerline,
          bool add_reference_values = false) const;

      //! extract those Dofs relevant for triad interpolation from total state vector
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode, typename T>
      void extract_rot_vec_dof_values(const std::vector<double>& dofvec,
          std::vector<Core::LinAlg::Matrix<3, 1, T>>& dofvec_rotvec) const;

      //! extract those Dofs relevant for triad interpolation from total state vector
      void extract_rot_vec_dof_values(const std::vector<double>& dofvec,
          std::vector<Core::LinAlg::Matrix<3, 1, double>>& dofvec_rotvec) const;

      //! compute nodal triads from current nodal rotation vectors ("displacement", i.e. relative
      //! rotations)
      template <unsigned int nnodetriad, typename T>
      void get_nodal_triads_from_disp_theta(
          const std::vector<Core::LinAlg::Matrix<3, 1, double>>& disptheta,
          std::vector<Core::LinAlg::Matrix<4, 1, T>>& Qnode) const;

     public:
      //! compute nodal triads either from current nodal rotation vectors ("displacement", i.e.
      //! relative rotations)
      //  or from full displacement state vector of the element (decision based on size of given STL
      //  vector)
      template <unsigned int nnodetriad, typename T>
      void get_nodal_triads_from_full_disp_vec_or_from_disp_theta(
          const std::vector<T>& dispvec, std::vector<Core::LinAlg::Matrix<4, 1, T>>& Qnode) const;

     private:
      //! compute vector with nnodetriad elements, who represent the 3x3-matrix-shaped interpolation
      //  function \tilde{I}^nnode at a certain point xi according to (3.18), Jelenic 1999
      template <unsigned int nnodetriad, typename T>
      void compute_generalized_nodal_rotation_interpolation_matrix_from_nodal_triads(
          const std::vector<Core::LinAlg::Matrix<4, 1, T>>& Qnode, const double xi,
          std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde) const;

      //! lump mass matrix
      template <unsigned int nnode>
      void lumpmass(Core::LinAlg::SerialDenseMatrix* massmatrix);

     public:
      //! determine Gauss rule from required type of integration and parameter list
      Core::FE::GaussRule1D my_gauss_rule(const IntegrationPurpose intpurpose) const;

      //@}

     private:
      //! @name Methods for Brownian dynamics simulations

      //! computes rotational damping forces and stiffness
      template <unsigned int nnodetriad, unsigned int nnodecl, unsigned int vpernode,
          unsigned int ndim>
      void evaluate_rotational_damping(Teuchos::ParameterList& params,  //!< parameter list
          const std::vector<Core::LinAlg::Matrix<4, 1, double>>& Qnode,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
          Core::LinAlg::SerialDenseVector* force);       //!< element internal force vector

      //! computes translational damping forces and stiffness
      template <unsigned int nnodecl, unsigned int vpernode,
          unsigned int ndim>  // number of nodes, number of dimensions of embedding space, number of
                              // degrees of freedom per node
      void evaluate_translational_damping(Teuchos::ParameterList& params,  //!< parameter list
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& vel_centerline,
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,   //!< element stiffness matrix
          Core::LinAlg::SerialDenseVector* force) const;  //!< element internal force vector

      //! computes stochastic translational forces and resulting stiffness
      template <unsigned int nnodecl, unsigned int vpernode, unsigned int ndim,
          unsigned int randompergauss>
      void evaluate_stochastic_forces(Teuchos::ParameterList& params,  //!< parameter list
          const Core::LinAlg::Matrix<ndim * vpernode * nnodecl, 1, double>& disp_totlag_centerline,
          Core::LinAlg::SerialDenseMatrix* stiffmatrix,   //!< element stiffness matrix
          Core::LinAlg::SerialDenseVector* force) const;  //!< element internal force vector

      //@}
      //! computes modified Jacobian for PTC
      void calc_stiff_contributions_ptc(Core::LinAlg::SerialDenseMatrix& elemat1);

     private:
      //! storing temporary stiffness matrix for element based scaling operator in PTC method
      Core::LinAlg::SerialDenseMatrix stiff_ptc_;

      //! bool storing whether automatic differentiation shall be used for this element evaluation
      bool use_fad_;

      //! variable saving whether element has already been initialized (then isinit_ == true)
      bool isinit_;


      //! initial length of the element
      double reflength_;

      //! rotational pseudovectors at nodes in reference configuration
      std::vector<Core::LinAlg::Matrix<3, 1>> theta0node_;

      //! vector holding current tangent at the centerline nodes Todo needed?
      std::vector<Core::LinAlg::Matrix<3, 1>> tcurrnode_;

      //! initial material curvature at Gauss points for elasticity (corresponding to \Lambda_0^t
      //! \Labmda'_0 in eq. (3.5), Crisfield 1999
      std::vector<Core::LinAlg::Matrix<3, 1>> kref_gp_;

      //! initial axial tension (always zero) and shear deformation at Gauss points for elasticity
      //! (corresponding to \Lambda_0^t rprime_0 - (1,0,0) )
      std::vector<Core::LinAlg::Matrix<3, 1>> gammaref_gp_;


      //! Vector holding value of Jacobi determinant for each Gauss point of integration purpose
      //! res_elastic_force
      std::vector<double> jacobi_gp_elastf_;

      //! Vector holding value of Jacobi determinant for each Gauss point of integration purpose
      //! res_elastic_moment
      std::vector<double> jacobi_gp_elastm_;

      //! Vector holding value of Jacobi determinant for each Gauss point of integration purpose
      //! res_inertia
      std::vector<double> jacobi_gp_mass_;

      //! Vector holding value of Jacobi determinant for each Gauss point of integration purpose
      //! res_damp_stoch
      std::vector<double> jacobi_gp_dampstoch_;

      //! Vector holding value of Jacobi determinant for each Gauss point of integration purpose
      //! neumann_lineload
      std::vector<double> jacobi_gp_neumannline_;


      //  //! nodal triads in quaternion form at the end of the preceding time step
      std::vector<Core::LinAlg::Matrix<4, 1>> qconvnode_;
      //! nodal triads in quaternion during the current iteration step
      std::vector<Core::LinAlg::Matrix<4, 1>> qnewnode_;

      //************** begin: Class variables required for element-based Lie-group time integration
      //*******************************
      //! triads at Gauss points for exact integration in quaternion at the end of the preceding
      //! time step (required for computation of angular velocity)
      std::vector<Core::LinAlg::Matrix<4, 1>> qconv_gp_mass_;
      //! current triads at Gauss points for exact integration in quaternion (required for
      //! computation of angular velocity)
      std::vector<Core::LinAlg::Matrix<4, 1>> qnew_gp_mass_;
      //! spatial angular velocity vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> wconv_gp_mass_;
      //! current spatial angular velocity vector at Gauss points for exact integration (required
      //! for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> wnew_gp_mass_;
      //! spatial angular acceleration vector at Gauss points for exact integration at the end of
      //! the preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> aconv_gp_mass_;
      //! current spatial angular acceleration vector at Gauss points for exact integration
      //! (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> anew_gp_mass_;
      //! modified spatial angular acceleration vector (according to gen-alpha time integration) at
      //! Gauss points for exact integration at the end of the preceding time step (required for
      //! computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> amodconv_gp_mass_;
      //! current modified spatial angular acceleration vector (according to gen-alpha time
      //! integration) at Gauss points for exact integration (required for computation of inertia
      //! terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> amodnew_gp_mass_;
      //! translational acceleration vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttconv_gp_mass_;
      //! current translational acceleration vector at Gauss points for exact integration (required
      //! for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttnew_gp_mass_;
      //! modified translational acceleration vector at Gauss points for exact integration at the
      //! end of the preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttmodconv_gp_mass_;
      //! modified current translational acceleration vector at Gauss points for exact integration
      //! (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rttmodnew_gp_mass_;
      //! translational velocity vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rtconv_gp_mass_;
      //! current translational velocity vector at Gauss points for exact integration (required for
      //! computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rtnew_gp_mass_;
      //! translational displacement vector at Gauss points for exact integration at the end of the
      //! preceding time step (required for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rconv_gp_mass_;
      //! current translational displacement vector at Gauss points for exact integration (required
      //! for computation of inertia terms)
      std::vector<Core::LinAlg::Matrix<3, 1>> rnew_gp_mass_;
      //************** end: Class variables required for element-based Lie-group time integration
      //*******************************//

      //! triads at Gauss points for integration of damping/stochastic forces in quaternion form at
      //! the end of the preceding time step (required for computation of angular velocity)
      std::vector<Core::LinAlg::Matrix<4, 1>> qconv_gp_dampstoch_;
      //! current triads at Gauss points for integration of damping/stochastic forces in quaternion
      //! form (required for computation of angular velocity)
      std::vector<Core::LinAlg::Matrix<4, 1>> qnew_gp_dampstoch_;

      //!@name variables only needed/used for output purposes. Note: No need to pack and unpack
      //! @{

      //! internal (elastic) energy of element
      double eint_;

      //! kinetic energy of element
      double ekin_;

      //! kinetic energy from rotational dofs part1
      double ekintorsion_;

      //! kinetic energy from rotational dofs part2
      double ekinbending_;

      //! kinetic energy from translational dofs
      double ekintrans_;

      //! angular momentum of the element
      Core::LinAlg::Matrix<3, 1> l_;

      //! linear momentum of the element
      Core::LinAlg::Matrix<3, 1> p_;

      //! norm of maximal bending curvature occurring in this element Todo obsolete?
      double kmax_;

      //! strain resultant values at GPs
      std::vector<double> axial_strain_gp_elastf_;
      std::vector<double> shear_strain_2_gp_elastf_;
      std::vector<double> shear_strain_3_gp_elastf_;

      std::vector<double> twist_gp_elastm_;
      std::vector<double> curvature_2_gp_elastm_;
      std::vector<double> curvature_3_gp_elastm_;

      //! material stress resultant values at GPs
      std::vector<double> material_axial_force_gp_elastf_;
      std::vector<double> material_shear_force_2_gp_elastf_;
      std::vector<double> material_shear_force_3_gp_elastf_;

      std::vector<double> material_torque_gp_elastm_;
      std::vector<double> material_bending_moment_2_gp_elastm_;
      std::vector<double> material_bending_moment_3_gp_elastm_;

      //! spatial stress resultant values at GPs
      std::vector<double> spatial_x_force_gp_elastf_;
      std::vector<double> spatial_y_force_2_gp_elastf_;
      std::vector<double> spatial_z_force_3_gp_elastf_;

      std::vector<double> spatial_x_moment_gp_elastm_;
      std::vector<double> spatial_y_moment_2_gp_elastm_;
      std::vector<double> spatial_z_moment_3_gp_elastm_;

      //! @}
    };

    // << operator
    std::ostream& operator<<(std::ostream& os, const Core::Elements::Element& ele);


  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
