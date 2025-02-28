// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_BASE_HPP
#define FOUR_C_BEAM3_BASE_HPP

#include "4C_config.hpp"

#include "4C_beam3_spatial_discretization_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_vector.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN


// forward declaration ...
namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace Core::Geo::MeshFree
{
  class BoundingBox;
}

namespace Solid
{
  namespace Elements
  {
    class ParamsInterface;
  }
}  // namespace Solid

namespace BrownianDynamics
{
  class ParamsInterface;
}

namespace Mat
{
  class BeamMaterial;
  template <typename T>
  class BeamMaterialTemplated;
}  // namespace Mat

namespace Discret
{
  namespace Elements
  {
    //! base class for all beam elements
    class Beam3Base : public Core::Elements::Element
    {
     public:
      /*!
      \brief Standard Constructor

      \param id    (in): A globally unique element id
      \param etype (in): Type of element
      \param owner (in): owner processor of the element
      */
      Beam3Base(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      Beam3Base(const Beam3Base& old);

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

      /** \brief set the parameter interface ptr for the solid elements
       *
       *  \param p (in): Parameter list coming from the time integrator.
       *
       *  \date 04/16 */
      void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

      virtual void set_brownian_dyn_params_interface_ptr();

      /** \brief returns true if the parameter interface is defined and initialized, otherwise
       * false
       *
       *  \date 04/16 */
      inline bool is_params_interface() const override { return (interface_ptr_ != nullptr); }

      /** \brief get access to the parameter interface pointer
       *
       *  \date 04/16 */
      std::shared_ptr<Core::Elements::ParamsInterface> params_interface_ptr() override;
      virtual std::shared_ptr<BrownianDynamics::ParamsInterface> brownian_dyn_params_interface_ptr()
          const;

      //! computes the number of different random numbers required in each time step for
      //! generation of stochastic forces
      virtual int how_many_random_numbers_i_need() const = 0;

      /** \brief get access to the element reference length
       *        (i.e. arc-length in stress-free configuration)
       *
       *  \date 05/16 */
      virtual double ref_length() const = 0;

      /** \brief get the radius of the element which is used for interactions (contact, viscous,
       *         potential-based, ...)
       *         - if needed, extend this to other than circular cross-section shapes and
       * dimensions to be specified via input file
       *         - allow for different assumed shapes for different interaction types if needed
       *
       *  \date 02/17 */
      double get_circular_cross_section_radius_for_interactions() const;

      /** \brief get number of nodes used for centerline interpolation
       *
       *  \date 05/16 */
      virtual int num_centerline_nodes() const = 0;

      /** \brief find out whether given node is used for centerline interpolation
       *
       *  \date 10/16 */
      virtual bool is_centerline_node(const Core::Nodes::Node& node) const = 0;

      /** \brief return GIDs of all additive DoFs for a given node
       *
       *  \date 07/16 */
      std::vector<int> get_additive_dof_gids(
          const Core::FE::Discretization& discret, const Core::Nodes::Node& node) const;

      /** \brief return GIDs of all non-additive, i.e. rotation pseudo vector DoFs for a given
       * node
       *
       *  \date 07/16 */
      std::vector<int> get_rot_vec_dof_gids(
          const Core::FE::Discretization& discret, const Core::Nodes::Node& node) const;

      /** \brief add indices of those DOFs of a given node that are positions
       *
       *  \date 07/16 */
      virtual void position_dof_indices(
          std::vector<int>& posdofs, const Core::Nodes::Node& node) const = 0;

      /** \brief add indices of those DOFs of a given node that are tangents (in the case of
       * Hermite interpolation)
       *
       *  \date 07/16 */
      virtual void tangent_dof_indices(
          std::vector<int>& tangdofs, const Core::Nodes::Node& node) const = 0;

      /** \brief add indices of those DOFs of a given node that are rotation DOFs (non-additive
       * rotation vectors)
       *
       *  \date 07/16 */
      virtual void rotation_vec_dof_indices(
          std::vector<int>& rotvecdofs, const Core::Nodes::Node& node) const = 0;

      /** \brief add indices of those DOFs of a given node that are 1D rotation DOFs
       *         (planar rotations are additive, e.g. in case of relative twist DOF of beam3k with
       * rotvec=false)
       *
       *  \date 07/16 */
      virtual void rotation_1d_dof_indices(
          std::vector<int>& twistdofs, const Core::Nodes::Node& node) const = 0;

      /** \brief add indices of those DOFs of a given node that represent norm of tangent vector
       *         (additive, e.g. in case of beam3k with rotvec=true)
       *
       *  \date 07/16 */
      virtual void tangent_length_dof_indices(
          std::vector<int>& tangnormdofs, const Core::Nodes::Node& node) const = 0;

      /** \brief get element local indices of those Dofs that are used for centerline
       * interpolation
       *
       *  \date 12/16 */
      virtual void centerline_dof_indices_of_element(
          std::vector<unsigned int>& centerlinedofindices) const = 0;

      /** \brief get Jacobi factor ds/dxi(xi) at xi \in [-1;1]
       *
       *  \date 06/16 */
      virtual double get_jacobi_fac_at_xi(const double& xi) const = 0;

      /** \brief Get material cross-section deformation measures, i.e. strain resultants
       *
       *  \date 04/17 */
      virtual inline void get_material_strain_resultants_at_all_gps(
          std::vector<double>& axial_strain_GPs, std::vector<double>& shear_strain_2_GPs,
          std::vector<double>& shear_strain_3_GPs, std::vector<double>& twist_GPs,
          std::vector<double>& curvature_2_GPs, std::vector<double>& curvature_3_GPs) const
      {
        FOUR_C_THROW("not implemented");
      }

      /** \brief Get spatial cross-section stress resultants
       *
       *  \date 05/17 */
      virtual inline void get_spatial_stress_resultants_at_all_gps(
          std::vector<double>& spatial_axial_force_GPs,
          std::vector<double>& spatial_shear_force_2_GPs,
          std::vector<double>& spatial_shear_force_3_GPs, std::vector<double>& spatial_torque_GPs,
          std::vector<double>& spatial_bending_moment_2_GPs,
          std::vector<double>& spatial_bending_moment_3_GPs) const
      {
        FOUR_C_THROW("not implemented");
      }

      /** \brief Get spatial cross-section stress resultants
       *
       *  \date 05/17 */
      virtual inline void get_spatial_forces_at_all_gps(
          std::vector<double>& spatial_axial_force_GPs,
          std::vector<double>& spatial_shear_force_2_GPs,
          std::vector<double>& spatial_shear_force_3_GPs) const
      {
        FOUR_C_THROW("not implemented");
      }

      /** \brief Get spatial cross-section stress resultants
       *
       *  \date 05/17 */
      virtual inline void get_spatial_moments_at_all_gps(std::vector<double>& spatial_torque_GPs,
          std::vector<double>& spatial_bending_moment_2_GPs,
          std::vector<double>& spatial_bending_moment_3_GPs) const
      {
        FOUR_C_THROW("not implemented");
      }

      /** \brief Get material cross-section stress resultants
       *
       *  \date 04/17 */
      virtual inline void get_material_stress_resultants_at_all_gps(
          std::vector<double>& material_axial_force_GPs,
          std::vector<double>& material_shear_force_2_GPs,
          std::vector<double>& material_shear_force_3_GPs, std::vector<double>& material_torque_GPs,
          std::vector<double>& material_bending_moment_2_GPs,
          std::vector<double>& material_bending_moment_3_GPs) const
      {
        FOUR_C_THROW("not implemented");
      }

      /** \brief Get number of degrees of freedom of a single node
       *
       *  \date 08/16 */
      int num_dof_per_node(const Core::Nodes::Node& node) const override
      {
        FOUR_C_THROW("not implemented");
        return -1;
      }

      /** \brief get centerline position at xi \in [-1,1] (element parameter space) in stress-free
       * reference configuration
       *
       *  \date 06/16 */
      void get_ref_pos_at_xi(Core::LinAlg::Matrix<3, 1>& refpos, const double& xi) const;

      /** \brief get unit tangent vector in reference configuration at i-th node of beam element
       * (element-internal numbering)
       *
       *  \date 06/16 */
      virtual void get_ref_tangent_at_node(
          Core::LinAlg::Matrix<3, 1>& Tref_i, const int& i) const = 0;

      /** \brief get centerline position at xi \in [-1,1] (element parameter space) from
       * displacement state vector
       *
       *  \date 06/16 */
      virtual void get_pos_at_xi(Core::LinAlg::Matrix<3, 1>& pos, const double& xi,
          const std::vector<double>& disp) const = 0;

      /** \brief get triad at xi \in [-1,1] (element parameter space)
       *
       *  \date 07/16 */
      virtual void get_triad_at_xi(Core::LinAlg::Matrix<3, 3>& triad, const double& xi,
          const std::vector<double>& disp) const
      {
        // ToDo make pure virtual and add/generalize implementations in beam eles
        FOUR_C_THROW("not implemented");
      }

      /** \brief get generalized interpolation matrix which yields the variation of the position
       * and orientation at xi \in [-1,1] if multiplied with the vector of primary DoF variations
       *
       *  \date 11/16 */
      virtual void get_generalized_interpolation_matrix_variations_at_xi(
          Core::LinAlg::SerialDenseMatrix& Ivar, const double& xi,
          const std::vector<double>& disp) const
      {
        FOUR_C_THROW("not implemented");
      }

      /** \brief get linearization of the product of (generalized interpolation matrix for
       * variations (see above) and applied force vector) with respect to the primary DoFs of this
       * element
       *
       *  \date 01/17 */
      virtual void get_stiffmat_resulting_from_generalized_interpolation_matrix_at_xi(
          Core::LinAlg::SerialDenseMatrix& stiffmat, const double& xi,
          const std::vector<double>& disp, const Core::LinAlg::SerialDenseVector& force) const
      {
        FOUR_C_THROW("not implemented");
      }

      /** \brief get generalized interpolation matrix which yields the increments of the position
       * and orientation at xi \in [-1,1] if multiplied with the vector of primary DoF increments
       *
       *  \date 11/16 */
      virtual void get_generalized_interpolation_matrix_increments_at_xi(
          Core::LinAlg::SerialDenseMatrix& Iinc, const double& xi,
          const std::vector<double>& disp) const
      {
        FOUR_C_THROW("not implemented");
      }

      //! get internal (elastic) energy of element
      virtual double get_internal_energy() const = 0;

      //! get kinetic energy of element
      virtual double get_kinetic_energy() const = 0;

      //! shifts nodes so that proper evaluation is possible even in case of periodic boundary
      //! conditions
      virtual void un_shift_node_position(std::vector<double>& disp,  //!< element disp vector
          Core::Geo::MeshFree::BoundingBox const& periodic_boundingbox) const;

      //! get directions in which element might be cut by a periodic boundary
      virtual void get_directions_of_shifts(std::vector<double>& disp,
          Core::Geo::MeshFree::BoundingBox const& periodic_boundingbox,
          std::vector<bool>& shift_in_dim) const;

      /** \brief extract values for those Dofs relevant for centerline-interpolation from total
       * state vector
       *
       *  \date 11/16 */
      virtual void extract_centerline_dof_values_from_element_state_vector(
          const std::vector<double>& dofvec, std::vector<double>& dofvec_centerline,
          bool add_reference_values = false) const = 0;

      /** \brief return flag whether Hermite polynomials are applied for centerline interpolation
       */
      inline bool hermite_centerline_interpolation() const { return centerline_hermite_; }

     protected:
      //! vector holding reference tangent at the centerline nodes
      std::vector<Core::LinAlg::Matrix<3, 1>> Tref_;

      //! bool storing whether Hermite interpolation of centerline is applied (false: Lagrange
      //! interpolation)
      bool centerline_hermite_;

      /** \brief get access to the interface
       *
       *  \date 04/16 */
      inline Solid::Elements::ParamsInterface& params_interface() const
      {
        if (not is_params_interface()) FOUR_C_THROW("The interface ptr is not set!");
        return *interface_ptr_;
      }

      inline BrownianDynamics::ParamsInterface& brownian_dyn_params_interface() const
      {
        return *browndyn_interface_ptr_;
      }

      /** \brief add reference positions and tangents to (centerline) displacement state vector
       *
       * @tparam nnode number of nodes
       * @tparam vpernode values per nodal direction
       *
       * @param pos_ref_centerline Vector containing the centerline reference position values
       */
      template <unsigned int nnodecl, unsigned int vpernode, typename T>
      void add_ref_values_disp_centerline(
          Core::LinAlg::Matrix<3 * vpernode * nnodecl, 1, T>& pos_ref_centerline) const
      {
        for (unsigned int dim = 0; dim < 3; ++dim)
          for (unsigned int node = 0; node < nnodecl; ++node)
          {
            pos_ref_centerline(3 * vpernode * node + dim) += nodes()[node]->x()[dim];
            if (hermite_centerline_interpolation())
              pos_ref_centerline(3 * vpernode * node + 3 + dim) += Tref_[node](dim);
          }
      }

      /** \brief calculates the element length in reference configuration
       *
       * @tparam nnode number of nodes
       * @tparam vpernode values per nodal direction
       *
       * For Lagrange centerline interpolation the difference between both boundary points is used
       * as reference length. In the case of Hermite centerline interpolation the value is used as
       * start for a newton iteration.
       *
       * see "Meier, C.: Geometrically exact finite element formulations for slender beams and
       * their contact interaction, Technical University of Munich, Germany, 2016",
       * chapter 3.2.2.2
       */
      template <unsigned int nnode, unsigned int vpernode>
      double calc_reflength(
          const Core::LinAlg::Matrix<3 * vpernode * nnode, 1, double>& disp_refe_centerline);

      /*! \brief Get centerline position at given parameter coordinate xi
       *
       * @tparam nnode number of nodes
       * @tparam vpernode values per nodal direction
       *
       * The parameter disp_totlag has to contain the absolute (total Lagrange) values of the
       * centerline degrees of freedom, i.e., the reference values + the current displacement
       * values (\ref update_disp_totlag or \ref add_ref_values_disp_centerline).
       */
      template <unsigned int nnode, unsigned int vpernode, typename T>
      void get_pos_at_xi(Core::LinAlg::Matrix<3, 1, T>& r, const double& xi,
          const Core::LinAlg::Matrix<3 * nnode * vpernode, 1, T>& disp_totlag) const
      {
        Core::LinAlg::Matrix<1, vpernode * nnode, T> N_i;

        Discret::Utils::Beam::evaluate_shape_functions_at_xi<nnode, vpernode>(
            xi, N_i, shape(), ref_length());
        calc_r<nnode, vpernode, T>(disp_totlag, N_i, r);
      }

      /** \brief compute beam centerline position vector at position \xi in element parameter
       * space
       * [-1,1] via interpolation of nodal DoFs based on given shape function values
       *
       *  \date 03/16 */
      template <unsigned int nnode, unsigned int vpernode, typename T>
      void calc_r(const Core::LinAlg::Matrix<3 * vpernode * nnode, 1, T>& disp_totlag_centerline,
          const Core::LinAlg::Matrix<1, vpernode * nnode, double>& funct,
          Core::LinAlg::Matrix<3, 1, T>& r) const
      {
        Discret::Utils::Beam::calc_interpolation<nnode, vpernode, 3, T>(
            disp_totlag_centerline, funct, r);
      }

      /** \brief compute derivative of beam centerline (i.e. tangent vector) at position \xi in
       *         element parameter space [-1,1] with respect to \xi via interpolation of nodal
       * DoFs based on given shape function derivative values
       *
       *  \date 03/16 */
      template <unsigned int nnode, unsigned int vpernode, typename T>
      void calc_r_xi(const Core::LinAlg::Matrix<3 * vpernode * nnode, 1, T>& disp_totlag_centerline,
          const Core::LinAlg::Matrix<1, vpernode * nnode, double>& deriv,
          Core::LinAlg::Matrix<3, 1, T>& r_xi) const
      {
        Discret::Utils::Beam::calc_interpolation<nnode, vpernode, 3, T>(
            disp_totlag_centerline, deriv, r_xi);
      }

      /** \brief compute derivative of beam centerline (i.e. tangent vector) at position \xi in
       *         element parameter space [-1,1] with respect to arc-length parameter s in
       * reference configuration via interpolation of nodal DoFs based on given shape function
       * derivative values
       *
       *  \date 03/16 */
      template <unsigned int nnode, unsigned int vpernode, typename T>
      void calc_r_s(const Core::LinAlg::Matrix<3 * vpernode * nnode, 1, T>& disp_totlag_centerline,
          const Core::LinAlg::Matrix<1, vpernode * nnode, double>& deriv, const double& jacobi,
          Core::LinAlg::Matrix<3, 1, T>& r_s) const
      {
        calc_r_xi<nnode, vpernode, T>(disp_totlag_centerline, deriv, r_s);

        /* at this point we have computed derivative with respect to the element parameter \xi \in
         * [-1;1]; as we want derivative with respect to the reference arc-length parameter s, we
         * have to divide it by the Jacobi determinant at the respective point*/
        r_s.scale(1.0 / jacobi);
      }

      /** \brief get applied beam material law object
       */
      Mat::BeamMaterial& get_beam_material() const;


      /** \brief get elasto(plastic) beam material law object
       *
       */
      template <typename T>
      Mat::BeamMaterialTemplated<T>& get_templated_beam_material() const;

      /** \brief setup constitutive matrices from material law
       *
       *  \date 03/16 */
      template <typename T>
      void get_constitutive_matrices(
          Core::LinAlg::Matrix<3, 3, T>& CN, Core::LinAlg::Matrix<3, 3, T>& CM) const;

      /** \brief setup mass inertia tensors from material law
       *
       *  \date 03/16 */
      template <typename T>
      void get_translational_and_rotational_mass_inertia_tensor(
          double& mass_inertia_translational, Core::LinAlg::Matrix<3, 3, T>& J) const;

      /** \brief setup only translational mass inertia factor from material law
       *      this method is called by reduced beam formulation which don't include
       *      rotational mass inertia
       *
       *  \date 03/17 */
      void get_translational_mass_inertia_factor(double& mass_inertia_translational) const;

      //! @name Methods and variables for Brownian dynamics or beaminteraction simulations
      //! @{
      //! computes damping coefficients
      void get_damping_coefficients(Core::LinAlg::Matrix<3, 1>& gamma) const;

      //! computes velocity of background fluid and gradient of that velocity at a certain
      //! evaluation point in the physical space and adds respective terms to internal forces and
      //! damping matrix
      template <unsigned int ndim, typename T>  // number of dimensions of embedding space
      void get_background_velocity(Teuchos::ParameterList& params,  //!< parameter list
          const Core::LinAlg::Matrix<ndim, 1, T>&
              evaluationpoint,  //!< point at which background velocity and its gradient has to be
                                //!< computed
          Core::LinAlg::Matrix<ndim, 1, T>& velbackground,  //!< velocity of background fluid
          Core::LinAlg::Matrix<ndim, ndim, T>& velbackgroundgrad)
          const;  //!< gradient of velocity of background fluid

     public:
      //! get centerline pos at binding spot with locn x stored in element parameter space
      //! coordinates \in [-1,1] from displacement state vector
      void get_pos_of_binding_spot(Core::LinAlg::Matrix<3, 1>& pos, std::vector<double>& disp,
          Inpar::BeamInteraction::CrosslinkerType linkertype, int bspotlocn,
          Core::Geo::MeshFree::BoundingBox const& periodic_boundingbox) const;

      //! get triad at binding spot with locn x stored in element parameter space coordinates \in
      //! [-1,1] from displacement state vector
      void get_triad_of_binding_spot(Core::LinAlg::Matrix<3, 3>& triad, std::vector<double>& disp,
          Inpar::BeamInteraction::CrosslinkerType linkertype, int bspotlocn) const;

      /** \brief get entire binding spot information of element
       *
       *  \date 06/17 */
      std::map<Inpar::BeamInteraction::CrosslinkerType, std::vector<double>> const&
      get_binding_spots() const
      {
        return bspotposxi_;
      }

      /** \brief get number of binding spot types on this element
       *
       *  \date 06/17 */
      unsigned int get_number_of_binding_spot_types() const { return bspotposxi_.size(); }

      /** \brief get number of binding spots of certain binding spot type on this element
       *
       *  \date 06/17 */
      unsigned int get_number_of_binding_spots(
          Inpar::BeamInteraction::CrosslinkerType linkertype) const
      {
        return bspotposxi_.at(linkertype).size();
      }

      /** \brief get binding spot positions xi
       *
       *  \date 03/17 */
      double get_binding_spot_xi(
          Inpar::BeamInteraction::CrosslinkerType linkertype, unsigned int bspotlocn) const
      {
        if (bspotlocn > bspotposxi_.at(linkertype).size())
          FOUR_C_THROW("number of requested binding spot exceeds total number of binding spots");

        return bspotposxi_.at(linkertype)[bspotlocn];
      }

      /** \brief set binding spot positions and status in crosslinker simulation
       *
       *  \date 03/17 */
      void set_binding_spots(
          std::map<Inpar::BeamInteraction::CrosslinkerType, std::vector<double>> bspotposxi)
      {
        bspotposxi_.clear();
        bspotposxi_ = bspotposxi;
      }

      /** \brief set binding spot positions and status in crosslinker simulation
       *
       *  \date 03/17 */
      void set_positions_of_binding_spot_type(
          Inpar::BeamInteraction::CrosslinkerType linkertype, std::vector<double> const& bspotposxi)
      {
        bspotposxi_[linkertype] = bspotposxi;
      }

      /** \brief set/get type of filament the element is part of
       *
       *  \date 03/17 */
      void set_filament_type(Inpar::BeamInteraction::FilamentType filamenttype)
      {
        filamenttype_ = filamenttype;
      }

      Inpar::BeamInteraction::FilamentType get_filament_type() const { return filamenttype_; }

      /**
       * \brief Get the bounding volume of the element for geometric search
       *
       * @param discret discretization of the respective field
       * @param result_data_dofbased Result data vector used for extracting positions
       * @return bounding volume of the respective element
       */
      Core::GeometricSearch::BoundingVolume get_bounding_volume(
          const Core::FE::Discretization& discret,
          const Core::LinAlg::Vector<double>& result_data_dofbased,
          const Core::GeometricSearch::GeometricSearchParams& params) const override;

     private:
      //! position of binding spots on beam element in local coordinate system
      //! size of vector equals number of binding spots on this element
      std::map<Inpar::BeamInteraction::CrosslinkerType, std::vector<double>> bspotposxi_;

      //! type of filament element belongs to
      Inpar::BeamInteraction::FilamentType filamenttype_;

      //! @}

      /** \brief interface ptr
       *
       *  data exchange between the element and the time integrator. */
      std::shared_ptr<Solid::Elements::ParamsInterface> interface_ptr_;

      std::shared_ptr<BrownianDynamics::ParamsInterface> browndyn_interface_ptr_;

      /*!
      \brief Default Constructor must not be called
      */
      Beam3Base();
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
