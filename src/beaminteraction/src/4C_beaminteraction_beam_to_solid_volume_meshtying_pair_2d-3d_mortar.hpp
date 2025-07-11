// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_2D_3D_MORTAR_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_VOLUME_MESHTYING_PAIR_2D_3D_MORTAR_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_base.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_shape_functions.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_linalg_vector.hpp"


FOUR_C_NAMESPACE_OPEN


namespace GeometryPair
{
  /**
   * \brief Base class for Fourier shape functions defined along a 1D curve in 3D space.
   *
   * @tparam discretization Type of 1D shape function.
   * @tparam n_fourier_modes Number of Fourier modes.
   * @tparam spatial_dim Number of spatial dimensions. This affects the number of degrees of freedom
   * of the element
   */
  template <typename CurveDiscretization, unsigned int n_fourier_modes,
      unsigned int spatial_dim = 3>
  class FourierDiscretization
  {
   public:
    // Discretization of the 1D part
    using curve_discretization_ = CurveDiscretization;

    //! Type of shape function that will be used when evaluating the shape functions.
    static constexpr Core::FE::CellType discretization_ = curve_discretization_::discretization_;

    //! Dimension of element (curve=1, surface=2, volume=3).
    static constexpr unsigned int element_dim_ = Core::FE::dim<discretization_>;

    //! Number of Fourier modes.
    static constexpr unsigned int n_fourier_modes_ = n_fourier_modes;

    //! Number of values per node.
    static constexpr unsigned int n_val_ = 1 + 2 * n_fourier_modes_;

    //! Number of nodes for this element.
    static constexpr unsigned int n_nodes_ = Core::FE::num_nodes(discretization_);

    //! Number of spatial dimensions.
    static const unsigned int spatial_dim_ = spatial_dim;

    //! Number of unknowns for this element.
    static constexpr unsigned int n_dof_ = spatial_dim_ * n_val_ * n_nodes_;

    //! Geometry type of the element.
    static constexpr GeometryPair::DiscretizationTypeGeometry geometry_type_ =
        ElementDiscretizationToGeometryType<discretization_>::geometry_type_;
  };

  //! Fourier types
  using t_line2_fourier_0 = FourierDiscretization<t_line2_scalar, 0>;
  using t_line2_fourier_1 = FourierDiscretization<t_line2_scalar, 1>;
  using t_line2_fourier_2 = FourierDiscretization<t_line2_scalar, 2>;
  using t_line2_fourier_3 = FourierDiscretization<t_line2_scalar, 3>;


  /**
   * \brief Compile time struct to check if an element type is based on Fourier shape functions
   */
  template <typename ElementType>
  struct IsFourierElement
  {
    static const bool value_ = false;
  };

  template <>
  struct IsFourierElement<t_line2_fourier_0>
  {
    static const bool value_ = true;
  };
  template <>
  struct IsFourierElement<t_line2_fourier_1>
  {
    static const bool value_ = true;
  };
  template <>
  struct IsFourierElement<t_line2_fourier_2>
  {
    static const bool value_ = true;
  };
  template <>
  struct IsFourierElement<t_line2_fourier_3>
  {
    static const bool value_ = true;
  };


  /**
   * \brief Specialization for Fourier elements
   */
  template <typename ElementType>
  struct EvaluateShapeFunction<ElementType,
      typename std::enable_if<IsFourierElement<ElementType>::value_>::type>
  {
    /**
     * \brief Evaluate the shape functions of the element.
     *
     * xi(0) in [-1, 1] is the coordinate along the 1D curve
     * xi(1) in [0,2pi] is the coordinate in circumferential direction
     *
     * @param N (out) shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T, typename... NotNeededArgumentType>
    static void evaluate(V& N, const T& xi, const NotNeededArgumentType&... not_needed_argument)
    {
      if constexpr (ElementType::element_dim_ == 1)
      {
        Core::LinAlg::Matrix<ElementType::n_nodes_, 1, typename T::scalar_type> N_1d;
        Core::FE::shape_function_1d(N_1d, xi(0), ElementType::discretization_);

        Core::LinAlg::Matrix<ElementType::n_val_, 1, typename T::scalar_type> N_fourier;
        N_fourier(0) = 1.0;
        for (unsigned int i_fourier_mode = 0; i_fourier_mode < ElementType::n_fourier_modes_;
            i_fourier_mode++)
        {
          N_fourier(1 + i_fourier_mode * 2) = cos((1 + i_fourier_mode) * xi(1));
          N_fourier(1 + i_fourier_mode * 2 + 1) = sin((1 + i_fourier_mode) * xi(1));
        }

        for (unsigned int i_node = 0; i_node < ElementType::n_nodes_; i_node++)
        {
          for (unsigned int i_val = 0; i_val < ElementType::n_val_; i_val++)
          {
            N(i_node * ElementType::n_val_ + i_val) = N_1d(i_node) * N_fourier(i_val);
          }
        }
      }
      else
      {
        FOUR_C_THROW("Got unexpected element dimension {}", ElementType::element_dim_);
      }
    }
  };
}  // namespace GeometryPair


namespace BeamInteraction
{
  // Forward declarations
  class BeamToSolidMortarManager;

  /**
   * \brief Class for full 2D-3D beam-to-solid volume mesh tying based on a Simo-Reissner beam
   * element with mortar constraint discretization.
   *
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @param solid Type from GeometryPair::ElementDiscretization... representing the solid.
   * @param solid Mortar shape function.
   */
  template <typename Beam, typename Solid, typename Mortar>
  class BeamToSolidVolumeMeshtyingPair2D3DMortar
      : public BeamToSolidVolumeMeshtyingPair2D3DBase<double, Beam, Solid>
  {
   private:
    //! Shortcut to the base class.
    using base_class = BeamToSolidVolumeMeshtyingPair2D3DBase<double, Beam, Solid>;

    //! Rotation representation. Each node has a rotation vector with 3 components.
    static constexpr unsigned int n_nodes_rot_ = 3;
    static constexpr unsigned int n_dof_rot_ = n_nodes_rot_ * 3;

    //! Number of DOFs for the pair.
    static constexpr unsigned int n_dof_pair_ = Beam::n_dof_ + Solid::n_dof_ + n_dof_rot_;

    //! Number of dependent variables for the pair. The ordering is as follows: first the beam DOFs,
    //! then the solid DOFs, then the components of the cross section rotation vector.
    static constexpr unsigned int n_dof_fad_ = Beam::n_dof_ + Solid::n_dof_ + n_dof_rot_ + 3;

    //! FAD type to evaluate the rotational coupling terms and derive them w.r.t. psi.
    using scalar_type_rotation_vector = typename Sacado::Fad::SLFad<double, 3>;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToSolidVolumeMeshtyingPair2D3DMortar() = default;

    /**
     * \brief This pair enforces constraints via a mortar-type method, which requires an own
     * assembly method (provided by the mortar manager).
     */
    inline bool is_assembly_direct() const override { return false; };

    /*!
     *\brief things that need to be done in a separate loop before the actual evaluation loop
     *      over all contact pairs
     */
    void pre_evaluate() override;

    /**
     * \brief Evaluate the global matrices and vectors resulting from mortar coupling. (derived)
     */
    void evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
        Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
        Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
        Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
        Core::LinAlg::FEVector<double>& global_constraint,
        Core::LinAlg::FEVector<double>& global_kappa,
        Core::LinAlg::SparseMatrix& global_kappa_lin_beam,
        Core::LinAlg::SparseMatrix& global_kappa_lin_solid,
        Core::LinAlg::FEVector<double>& global_lambda_active,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector) override;

    /**
     * \brief Evaluate the terms that directly assemble it into the global force vector and
     * stiffness matrix (derived).
     */
    void evaluate_and_assemble(const Core::FE::Discretization& discret,
        const BeamToSolidMortarManager* mortar_manager,
        const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
        const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
        const Core::LinAlg::Vector<double>& global_lambda,
        const Core::LinAlg::Vector<double>& displacement_vector) override;

    /**
     * \brief Update state of rotational DoFs of both elements
     */
    void reset_rotation_state(const Core::FE::Discretization& discret,
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& ia_discolnp) override;

    /**
     * \brief Add the visualization of this pair to the beam to solid visualization output writer
     * (derived)
     */
    void get_pair_visualization(
        std::shared_ptr<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
        Teuchos::ParameterList& visualization_params) const override;

   protected:
    /**
     * \brief Get the triad of the beam at the parameter coordinate xi (derived)
     */
    void get_triad_at_xi_double(const double xi, Core::LinAlg::Matrix<3, 3, double>& triad,
        const bool reference) const override;

   private:
    //! Reference triad interpolation in the beam element
    LargeRotations::TriadInterpolationLocalRotationVectors<3, double>
        triad_interpolation_scheme_ref_;

    //! Current triad interpolation in the beam element
    LargeRotations::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme_;

    //! Data evaluated along side the mortar matrices, but required for direct stiffness
    //! contributions (due to rotational coupling terms)
    std::array<std::array<std::array<double, n_dof_rot_>, Mortar::n_dof_>, n_dof_rot_>
        lagrange_shape_times_skew_times_mortar_shape_lin_psi_times_t_times_itilde_;
  };

  /**
   * \brief Factory for the mortar cross section pair
   */
  std::shared_ptr<BeamContactPair> create_beam_to_solid_volume_pair_mortar_cross_section(
      const Core::FE::CellType shape,
      const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function,
      const int n_fourier_modes);

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
