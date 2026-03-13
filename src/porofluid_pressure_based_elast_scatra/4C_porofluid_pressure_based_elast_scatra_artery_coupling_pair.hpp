// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_PAIR_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ARTERY_COUPLING_PAIR_HPP

#include "4C_config.hpp"

#include "4C_art_net_input.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_porofluid_pressure_based_ele_phasemanager.hpp"
#include "4C_porofluid_pressure_based_ele_variablemanager.hpp"

#include <Sacado.hpp>

#include <functional>
#include <memory>

// define Fad object for evaluation
using FAD = Sacado::Fad::DFad<double>;

// forward declaration

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Mat
{
  class MatList;
  class Cnst1dArt;
}  // namespace Mat
namespace Core
{
  namespace LinAlg
  {
    class SerialDenseVector;
    class SerialDenseMatrix;
  }  // namespace LinAlg
  namespace Utils
  {
    class FunctionOfAnything;
  }
}  // namespace Core

namespace PoroPressureBased
{
  //! Tolerance for check if two elements are collinear (default value: 1.0e-8)
  constexpr double tol_collinear = 1.0e-8;

  //! Tolerance for Newton loop in the projection algorithm (default value: 1.0e-10)
  constexpr double tol_projection = 1.0e-10;

  //! Print projection information to the console for debugging purposes (default value: false)
  constexpr bool projection_output = false;

  //! Maximum number of iterations for the projection algorithm (default value: 10)
  constexpr int projection_max_iter = 10;

  class PorofluidElastScatraArteryCouplingPairBase
  {
   public:
    //! constructor
    PorofluidElastScatraArteryCouplingPairBase() = default;

    //! destructor
    virtual ~PorofluidElastScatraArteryCouplingPairBase() = default;

    //! Init
    virtual void init(std::vector<Core::Elements::Element const*> elements,
        const Teuchos::ParameterList& coupling_params,
        const Teuchos::ParameterList& porofluid_coupling_params,
        const std::vector<int>& coupled_dofs_homogenized,
        const std::vector<int>& coupled_dofs_artery,
        const std::vector<std::vector<int>>& scale_vector,
        const std::vector<std::vector<int>>& funct_vector, std::string condition_names,
        double penalty_parameter, std::string coupling_type = "", int eta_ntp = 0,
        const std::function<const Core::Utils::FunctionOfAnything&(int)>&
            function_of_anything_by_id = {},
        int my_mpi_rank = -1) = 0;

    //! query if pair active
    virtual bool is_active() = 0;

    //! things that need to be done in a separate loop before the actual evaluation loop
    //! over all coupling pairs
    virtual void pre_evaluate(std::shared_ptr<Core::LinAlg::MultiVector<double>> gp_vector) = 0;

    //! things that need to be done in a separate loop before the actual evaluation loop
    //! over all coupling pairs
    virtual void delete_unnecessary_gps(
        std::shared_ptr<Core::LinAlg::MultiVector<double>> gp_vector) = 0;

    /*!
     * @brief Evaluate this pair
     *
     * @returns integral of diameter of the segment
     */
    virtual double evaluate(Core::LinAlg::SerialDenseVector* ele_rhs_artery,
        Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized,
        Core::LinAlg::SerialDenseMatrix* D_ele, Core::LinAlg::SerialDenseMatrix* M_ele,
        Core::LinAlg::SerialDenseVector* Kappa_ele, const std::vector<double>& segment_lengths) = 0;

    //! evaluate additional linearization of (integrated) element diameter dependent terms
    //! (Hagen-Poiseuille)
    virtual void evaluate_additional_linearization_of_integrated_diameter(
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized) = 0;

    //! flag if diameter function is active, i.e., variable diameter linearization needs to be
    //! calculated
    virtual bool variable_diameter_active() = 0;

    //! reset state
    virtual void reset_state(std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        std::shared_ptr<Core::FE::Discretization> artery_dis) = 0;

    /**
     * Set up the porofluid-managers and the materials for later evaluation
     * @param[in] dis_name: name of homogenized discretization
     * @param[in] timefacrhs_artery: right-hand side factor for artery time integration
     * @param[in] timefacrhs_homogenized: right-hand side factor for time integration of homogenized
     * 2D/3D discretization
     */
    virtual void setup_fluid_managers_and_materials(std::string dis_name,
        const double& timefacrhs_artery, const double& timefacrhs_homogenized) = 0;

    //! start of the integration segment
    virtual double eta_start() const = 0;
    //! end of the integration segment
    virtual double eta_end() const = 0;

    //! element 1 (= artery) GID
    virtual int artery_ele_gid() const = 0;
    //! element 2 (= homogenized) GID
    virtual int homogenized_ele_gid() const = 0;

    //! apply mesh movement to the artery element
    virtual double apply_mesh_movement(
        bool first_call, std::shared_ptr<Core::FE::Discretization> homogenized_dis) = 0;

    //! set segment id
    virtual void set_segment_id(const int& segment_id) = 0;
    //! get segment id
    virtual int get_segment_id() const = 0;

    //! get the volume of the homogenized 2D/3D element
    virtual double calculate_volume_homogenized_element() const = 0;

    //! get number of Gauss points
    virtual int num_gp() const = 0;
  };

  //! type of coupling pair
  enum class CouplingType
  {
    undefined,
    porofluid,
    scatra
  };

  //! the coupling pair
  template <Core::FE::CellType dis_type_artery, Core::FE::CellType dis_type_homogenized, int dim>
  class PorofluidElastScatraArteryCouplingPair final
      : public PorofluidElastScatraArteryCouplingPairBase
  {
   public:
    //! constructor
    PorofluidElastScatraArteryCouplingPair();

    //! Init
    void init(std::vector<Core::Elements::Element const*> elements,
        const Teuchos::ParameterList& coupling_params,
        const Teuchos::ParameterList& porofluid_coupling_params,
        const std::vector<int>& coupled_dofs_homogenized,
        const std::vector<int>& coupled_dofs_artery,
        const std::vector<std::vector<int>>& scale_vector,
        const std::vector<std::vector<int>>& function_id_vector, std::string condition_name,
        double penalty_parameter, std::string coupling_type = "", int eta_ntp = 0,
        const std::function<const Core::Utils::FunctionOfAnything&(int)>&
            function_of_anything_by_id = {},
        int my_mpi_rank = -1) override;

    //! query if pair active
    bool is_active() override { return is_active_; }

    //! things that need to be done in a separate loop before the actual evaluation loop
    //! over all coupling pairs
    void pre_evaluate(std::shared_ptr<Core::LinAlg::MultiVector<double>> gp_vector) override;

    //! things that need to be done in a separate loop before the actual evaluation loop
    //! over all coupling pairs
    void delete_unnecessary_gps(
        std::shared_ptr<Core::LinAlg::MultiVector<double>> gp_vector) override;

    //! flag if diameter function is active, i.e., variable diameter linearization needs to be
    //! calculated
    bool variable_diameter_active() override { return variable_diameter_active_; }

    //! reset state
    void reset_state(std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        std::shared_ptr<Core::FE::Discretization> artery_dis) override;

    /**
     * Set up the porofluid-managers and the materials for later evaluation
     * @param[in] dis_name: name of homogenized discretization
     * @param[in] timefacrhs_artery: right-hand side factor for artery time integration
     * @param[in] timefacrhs_homogenized: right-hand side factor for time integration of homogenized
     * 2D/3D discretization
     */
    void setup_fluid_managers_and_materials(std::string dis_name, const double& timefacrhs_artery,
        const double& timefacrhs_homogenized) override;

    /*!
     * @brief Evaluate this pair
     *
     * @returns integral of diameter of the segment
     */
    double evaluate(Core::LinAlg::SerialDenseVector* ele_rhs_artery,
        Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized,
        Core::LinAlg::SerialDenseMatrix* D_ele, Core::LinAlg::SerialDenseMatrix* M_ele,
        Core::LinAlg::SerialDenseVector* Kappa_ele,
        const std::vector<double>& segment_lengths) override;

    //! evaluate additional linearization of (integrated) element diameter dependent terms
    //! (Hagen-Poiseuille)
    void evaluate_additional_linearization_of_integrated_diameter(
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized) override;

    //! beginning and end of integration segment
    double eta_start() const override { return artery_segment_start_; }
    double eta_end() const override { return artery_segment_end_; }

    //! element 1 (= artery) GID
    int artery_ele_gid() const override;
    //! element 2 (= homogenized) GID
    int homogenized_ele_gid() const override;

    //! number of GP
    int num_gp() const override { return num_gp_; };

    //! apply mesh movement to the artery element
    double apply_mesh_movement(
        bool first_call, std::shared_ptr<Core::FE::Discretization> homogenized_dis) override;

    //! set segment id
    void set_segment_id(const int& segment_id) override;
    //! get segment id
    int get_segment_id() const override;

    //! get the volume of the homogenized 2D/3D element
    double calculate_volume_homogenized_element() const override;

   private:
    //! number of nodes of 1D artery element
    static constexpr unsigned num_nodes_artery_ = Core::FE::num_nodes(dis_type_artery);
    //! number of nodes of the homogenized 2D/3D element
    static constexpr unsigned num_nodes_homogenized_ = Core::FE::num_nodes(dis_type_homogenized);
    //! number of spatial dimensions
    static constexpr unsigned num_dim_ = Core::FE::dim<dis_type_homogenized>;

    //! set time factor needed for evaluation of right-hand side (function coupling) terms
    void set_time_fac_rhs(const double& artery_density,
        const Mat::MatList* scatra_material_homogenized, const double& timefacrhs_artery,
        const double& timefacrhs_homogenized);

    //! pre-evaluate for lateral surface coupling
    void pre_evaluate_lateral_surface_coupling(
        Core::LinAlg::MultiVector<double>& gauss_point_vector);

    //! pre-evaluate for centerline coupling
    void pre_evaluate_centerline_coupling();

    //! pre-evaluate for node-to-point coupling
    void pre_evaluate_node_to_point_coupling();

    //! extract velocity of solid phase
    void extract_velocity_solid_phase(const Core::FE::Discretization& homogenized_dis);

    //! recompute if deformable arteries are assumed
    void recompute_gp_coords_in_deformed_configuration(const std::vector<double>& segment_lengths,
        std::vector<double>& gp_coords_artery,
        std::vector<std::vector<double>>& gp_coords_homogenized, double& artery_coords_start,
        double& artery_coords_end);

    /**
     * \brief create segment [eta_a, eta_b]
     *
     * \note: the following algorithm works only for linear 1D elements and linear 2D/3D
     * elements where always 0,1 or 2 intersections can be found. For higher order elements with
     * special cases, it has to be re-thought. For instance, if we find two
     * intersections, it is always assumed that the integration segment lies between these two
     * intersections --> for a higher order 1D element this may not be the case
     */
    void create_integration_segment();

    //! get all intersections of artery element with 2D/3D element
    std::vector<double> get_all_intersections();

    //! project a Gauss point on 1D element into 2D/3D element
    template <typename T>
    void projection(Core::LinAlg::Matrix<num_dim_, 1, T>& coords_artery_ref,
        std::vector<T>& coords_homogenized, bool& projection_valid);

    //! Check for duplicate projections
    static bool projection_not_yet_found(
        const std::vector<double>& intersections, const double& eta);

    //! Intersect the artery element with edges (2D) or surfaces (3D) of the homogenized element
    void intersect_with_homogenized_element(std::vector<double>& coords_homogenized,
        double& coords_artery, const int& fixed_coord_idx, const double& fixed_at,
        bool& projection_valid);

    //! get artery 1D shape-functions at coordinate
    template <typename T>
    void get_artery_shape_functions(Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function,
        Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function_deriv, const T& coordinate);

    //! get homogenized 2D/3D shape-functions at xi1, xi2 (, xi3)
    template <typename T>
    void get_homogenized_shape_functions(
        Core::LinAlg::Matrix<1, num_nodes_homogenized_, T>& shape_function,
        Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, T>& shape_function_deriv,
        const std::vector<T>& coordinate);

    //! compute artery coordinates and derivatives in reference configuration
    template <typename T>
    void compute_artery_coords_and_derivs_ref(Core::LinAlg::Matrix<num_dim_, 1, T>& coordinates_ref,
        Core::LinAlg::Matrix<num_dim_, 1, T>& coordinates_deriv_ref,
        const Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function,
        const Core::LinAlg::Matrix<1, num_nodes_artery_, T>& shape_function_deriv);

    //! compute 2D/3D coordinates and derivatives in reference configuration
    template <typename T>
    void compute_homogenized_coords_and_derivs_ref(
        Core::LinAlg::Matrix<num_dim_, 1, T>& coordinates_ref,
        Core::LinAlg::Matrix<num_dim_, num_dim_, T>& coordinates_deriv_ref,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_, T>& shape_function,
        const Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_, T>& shape_function_deriv);

    //! evaluate the function coupling (return integral of diameter of the segment)
    void evaluate_function_coupling(const std::vector<double>& gp_coords_artery,
        const std::vector<std::vector<double>>& gp_coords_homogenized,
        const std::vector<double>& segment_lengths, Core::LinAlg::SerialDenseVector* ele_rhs_artery,
        Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized,
        double& integrated_diameter);

    /**
     * evaluate derivative of 1D shape function times solid velocity (only porofluid has this term)
     * @param[in] gp_coords_artery: GP coordinates in artery element parameter space
     * @param[in] gp_coords_homogenized: GP coordinates in porofluid element parameter space
     * @param[in] ele_rhs_artery: rhs-vector to assemble into
     * @param[in] artery_segment_start: beginning of segment in artery element parameter space
     * @param[in] artery_segment_end: end of segment in artery element parameter space
     */
    void evaluate_nds_solid_velocity(const std::vector<double>& gp_coords_artery,
        const std::vector<std::vector<double>>& gp_coords_homogenized,
        Core::LinAlg::SerialDenseVector& ele_rhs_artery, const double& artery_segment_start,
        const double& artery_segment_end);

    //! evaluate contribution to element matrix for the Gauss-point-to-segment case
    void evaluate_gpts_element_matrix(const double& gp_weight,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        const double& jacobian_matrix, const double& penalty_parameter);

    //! evaluate contribution to element matrix for the node-to-point case
    void evaluate_ntp_element_matrix(
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        const double& penalty_parameter);

    //! evaluate mortar coupling matrices D and M
    void evaluate_mortar_matrices_and_vector(const double& gp_weight,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        const double& jacobian_matrix);

    //! evaluate Gauss-point-to-segment coupling
    void evaluate_gpts(const std::vector<double>& gp_coords_artery,
        const std::vector<std::vector<double>>& gp_coords_homogenized,
        const std::vector<double>& segment_lengths, Core::LinAlg::SerialDenseVector* ele_rhs_artery,
        Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized);

    //! evaluate node-to-point coupling
    void evaluate_ntp(const std::vector<double>& gp_coords_artery,
        const std::vector<std::vector<double>>& gp_coords_homogenized,
        Core::LinAlg::SerialDenseVector* ele_rhs_artery,
        Core::LinAlg::SerialDenseVector* ele_rhs_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix* ele_matrix_homogenized_homogenized);

    //! evaluate mortar coupling matrices D and M
    void evaluate_mortar_coupling(const std::vector<double>& gp_coords_artery,
        const std::vector<std::vector<double>>& gp_coords_homogenized,
        const std::vector<double>& segment_lengths,
        Core::LinAlg::SerialDenseMatrix* mortar_matrix_d,
        Core::LinAlg::SerialDenseMatrix* mortar_matrix_m,
        Core::LinAlg::SerialDenseVector* mortar_vector_kappa);

    //! evaluate the function coupling
    void evaluate_function_coupling(const double& gp_weight,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery_deriv,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        const double& jacobi, Core::LinAlg::SerialDenseVector& ele_rhs_artery,
        Core::LinAlg::SerialDenseVector& ele_rhs_homogenized,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized,
        double& integrated_diameter);

    //! evaluate the diameter function and derivative (for coupling type porofluid)
    void evaluate_diameter_function_and_deriv(const double artery_pressure_np_at_gp,
        const double& gp_weight,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        const double& jacobian_determinant);

    //! integrate in deformed configuration from artery_coords_start to deformed_coords
    FAD integrate_length_to_deformed_coords(const FAD& deformed_coords);

    //! get values of artery at GP
    void get_artery_values_at_gp(
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery_deriv,
        double& artery_pressure, double& artery_pressure_gradient,
        std::vector<double>& artery_scalars);

    //! get scalar values of homogenized discretization at GP
    void get_homogenized_scalar_values_at_gp(
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        std::vector<double>& scalars_homogenized_np);

    //! assemble the function coupling into element matrix (artery-part)
    void assemble_function_coupling_into_ele_matrix_rhs_artery(const int& i_art,
        const double& gp_weight,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        const double& jacobian_determinant, const int& scale, const double& function_value,
        const std::vector<double>& artery_derivs, const std::vector<double>& homogenized_derivs,
        Core::LinAlg::SerialDenseVector& ele_rhs_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized);

    //! assemble the function coupling into element matrix (2D/3D-part)
    void assemble_function_coupling_into_ele_matrix_rhs_homogenized(
        const std::vector<int>& assemble_into, const double& gp_weight,
        const Core::LinAlg::Matrix<1, num_nodes_artery_>& shape_functions_artery,
        const Core::LinAlg::Matrix<1, num_nodes_homogenized_>& shape_functions_homogenized,
        const double& jacobian_determinant, const int& scale, const double& timefacrhs_homogenized,
        const double& function_value, const std::vector<double>& artery_derivs,
        const std::vector<double>& homogenized_derivs,
        Core::LinAlg::SerialDenseVector& ele_rhs_homogenized,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized);

    //! evaluate function and its derivative
    void evaluate_function_and_deriv(const Core::Utils::FunctionOfAnything& function,
        const double& artery_pressure, const double& artery_element_flow_rate,
        const std::vector<double>& artery_scalars, const std::vector<double>& homogenized_scalars,
        double& function_value, std::vector<double>& artery_derivs,
        std::vector<double>& homogenized_derivs);

    //! evaluate artery flow rate based on Hagen-Poiseuille law
    double evaluate_artery_flow_rate(double artery_pressure_gradient) const;

    //! set scalar as constants into function
    void set_scalar_values_as_constants(std::vector<std::pair<std::string, double>>& constants,
        const std::vector<double>& artery_scalars, const std::vector<double>& homogenized_scalars);

    //! set fluid as variables into function
    void set_fluid_values_as_variables(
        std::vector<std::pair<std::string, double>>& variables, const double& artery_pressure);

    //! set fluid as constants into function
    void set_fluid_values_as_constants(
        std::vector<std::pair<std::string, double>>& constants, const double& artery_pressure);

    //! set scalar as variables into function
    void set_scalar_values_as_variables(std::vector<std::pair<std::string, double>>& variables,
        const std::vector<double>& artery_scalars, const std::vector<double>& homogenized_scalars);

    //! evaluate derivatives w.r.t. fluid of function
    void evaluate_fluid_derivs(std::vector<double>& artery_derivs,
        std::vector<double>& homogenized_derivs, const std::vector<double>& function_derivs) const;

    //! evaluate derivatives w.r.t. scalar of function
    void evaluate_scalar_derivs(std::vector<double>& artery_derivs,
        std::vector<double>& homogenized_derivs, const std::vector<double>& function_derivs) const;

    //! evaluate contribution to element rhs for Gauss-point-to-segment or node-to-point case
    void evaluate_gpts_ntp_ele_rhs(Core::LinAlg::SerialDenseVector& ele_rhs_artery,
        Core::LinAlg::SerialDenseVector& ele_rhs_homogenized,
        const Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
        const Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
        const Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
        const Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized) const;

    //! update the element matrix for Gauss-point-to-segment or node-to-point
    void update_gpts_ntp_element_matrix(Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized) const;

    /**
     * \brief coupling to additional porous network is only possible if we also have an
     * element with a valid volume fraction pressure, i.e., if we also have a smeared representation
     * of the neovasculature at this point if not ---> corresponding matrices are set to zero
     */
    void check_valid_volume_fraction_pressure_coupling(
        Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_artery_homogenized,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_artery,
        Core::LinAlg::SerialDenseMatrix& ele_matrix_homogenized_homogenized);

    //! update the D, M and Kappa for mortar-penalty coupling
    void update_mortar_matrices_and_vector(Core::LinAlg::SerialDenseMatrix& mortar_matrix_d,
        Core::LinAlg::SerialDenseMatrix& mortar_matrix_m,
        Core::LinAlg::SerialDenseVector& mortar_vector_kappa);

    //! fill the function vector
    void fill_function_vector(std::vector<const Core::Utils::FunctionOfAnything*>& function_vector,
        const std::vector<int>& function_id_vector, const std::vector<int>& scale_vector);


    //! initialize a function
    static void initialize_function(const Core::Utils::FunctionOfAnything& funct);

    //! initialize names used in functions
    void initialize_function_names();

    //! initialize vector where to assemble homogenized DOF functions into
    void initialize_assemble_into_homogenized_dof_vector();

    //! coupling type
    CouplingType coupling_type_;

    //! coupling method (either Gauss-point-to-segment or mortar-penalty coupling)
    ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod coupling_method_;

    //! name of the condition
    std::string condition_name_;

    //! indicates if the init() function has been called
    bool is_init_;

    //! indicates if the pre_evaluate() function has been called
    bool is_pre_evaluated_;

    //! indicates if mesh tying is active, i.e., if projection possible
    bool is_active_;

    //! indicates if function coupling is active, i.e., if functions are defined
    bool function_coupling_active_;

    //! indicates if diameter function is active, i.e, if diameter function is defined
    bool variable_diameter_active_;

    //! So far, it is assumed that artery elements always follow the deformation of the
    //! underlying porous medium. Hence, we actually have to evaluate them in the current
    //! configuration if this flag is set to true. Artery elements will not move and are
    //! evaluated in reference configuration.
    bool evaluate_in_ref_config_;

    //! evaluate 1D-3D coupling on lateral surface?
    bool evaluate_on_lateral_surface_;

    //! first element of the interacting pair (artery or scatra element)
    const Core::Elements::Element* artery_element_;

    //! second element of the interacting pair (2D/3D element)
    const Core::Elements::Element* homogenized_element_;

    //! reference nodal positions of the 1D artery element
    Core::LinAlg::Matrix<num_dim_ * num_nodes_artery_, 1> nodal_coords_artery_ele_ref_;
    //! reference nodal positions of the 2D/3D homogenized element
    Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> nodal_coords_homogenized_ele_ref_;

    //! current position of the homogenized element
    Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> nodal_coords_homogenized_ele_;
    //! current velocity of the homogenized element
    Core::LinAlg::Matrix<num_dim_, num_nodes_homogenized_> nodal_velocity_homogenized_ele_;

    //! reference diameter of the artery element (constant across the element)
    double artery_diameter_ref_;
    //! current diameter of the artery element at the GP
    double artery_diameter_at_gp_;
    //! derivatives of the diameter function
    std::vector<double> diameter_derivs_;

    //! number of dofs of 2D/3D element
    int num_dof_homogenized_;
    //! number of dofs of the artery element
    int num_dof_artery_;
    //! number of dofs * number of nodes of the artery element
    int dim_artery_;
    //! number of dofs * number of nodes of the homogenized 2D/3D element
    int dim_homogenized_;

    //! coupled dofs of homogenized (2D/3D) element
    std::vector<int> coupled_dofs_homogenized_;
    //! coupled dofs of artery (1D) element
    std::vector<int> coupled_dofs_artery_;
    //! number of coupled dofs
    int num_coupled_dofs_;

    //! the id of the volume fraction pressure phase
    std::vector<int> volfrac_pressure_id_;

    //! number of fluid phases in the porofluid (homogenized) problem
    int num_fluid_phases_;
    //! number of volume fractions in the porofluid (homogenized) problem
    int num_volfracs_;

    //! number of scalars (homogenized)
    int num_scalars_homogenized_;
    //! number of scalars (artery)
    int num_scalars_artery_;

    //! dof-set number of porofluid (either 0 or 2)
    int nds_porofluid_;

    //! number of Gauss points of the element
    int num_gp_;

    //! number of Gauss points of the element per patch
    // (only required for surface-based formulation)
    int num_gp_per_patch_;

    //! artery element length in reference configuration
    double artery_ele_length_ref_;

    //! artery element length in current configuration
    double artery_ele_length_;

    //! initial orientation of the artery element
    Core::LinAlg::Matrix<num_dim_, 1> initial_artery_orientation_;

    //! Jacobian determinant for integration segment = L/2.0*(eta_a - eta_b)/2.0
    double jacobian_determinant_;

    //! Gauss points in the homogenized element
    std::vector<std::vector<double>> gp_coords_homogenized_;

    //! Gauss points in the artery element
    std::vector<double> gp_coords_artery_;
    //! Gauss point weights in the artery element
    std::vector<double> gp_weights_;

    //! Gauss point coordinates in the deformed artery element from last converged value
    std::vector<double> previous_gp_coords_deformed_;

    //! primary variables of the homogenized element
    std::vector<double> phi_np_homogenized_ele_;
    //! primary variables of the artery element
    std::vector<double> phi_np_artery_ele_;

    //! nodal artery pressure values
    Core::LinAlg::Matrix<num_nodes_artery_, 1> nodal_artery_pressure_np_;

    //! nodal artery-scalar values
    std::vector<Core::LinAlg::Matrix<num_nodes_artery_, 1>> nodal_artery_scalar_np_;

    //! nodal homogenized scalar values for scatra coupling
    std::vector<Core::LinAlg::Matrix<num_nodes_homogenized_, 1>> nodal_homogenized_scalar_np_;

    //! penalty parameter
    double penalty_parameter_;

    //! start of the integration segment
    double artery_segment_start_;
    //! end of the integration segment
    double artery_segment_end_;

    //! length of integration segment int current configuration
    double current_segment_length_;

    //! check if constant part (i.e., Gauss-point-to-segment and MP part) has already been evaluated
    // if integration in reference configuration is performed
    bool constant_part_evaluated_;

    //! 1D coupling element type (can be ARTERY or AIRWAY)
    std::string coupling_element_type_;

    //! Gauss-point-to-segment/node-to-point element matrix (artery-artery contribution)
    Core::LinAlg::SerialDenseMatrix gpts_ntp_ele_matrix_artery_artery_;
    //! Gauss-point-to-segment/node-to-point element matrix (artery-homogenized contribution)
    Core::LinAlg::SerialDenseMatrix gpts_ntp_ele_matrix_artery_homogenized_;
    //! Gauss-point-to-segment/node-to-point element matrix (homogenized-artery contribution)
    Core::LinAlg::SerialDenseMatrix gpts_ntp_ele_matrix_homogenized_artery_;
    //! Gauss-point-to-segment/node-to-point element matrix (homogenized-homogenized contribution)
    Core::LinAlg::SerialDenseMatrix gpts_ntp_ele_matrix_homogenized_homogenized_;

    //! (variable) diameter element matrix (artery-artery contribution)
    Core::LinAlg::SerialDenseMatrix diameter_ele_matrix_artery_artery_;
    //! (variable) diameter element matrix (artery-homogenized contribution)
    Core::LinAlg::SerialDenseMatrix diameter_ele_matrix_artery_homogenized_;

    //! mortar coupling matrix D
    Core::LinAlg::SerialDenseMatrix mortar_matrix_d_;
    //! mortar coupling matrix M
    Core::LinAlg::SerialDenseMatrix mortar_matrix_m_;
    //! mortar coupling vector kappa
    Core::LinAlg::SerialDenseVector mortar_vector_kappa_;

    //! (dX/dxi)^-1
    std::vector<Core::LinAlg::Matrix<num_dim_, num_dim_>> inverse_jacobian_matrix_;

    //! phase manager of the porofluid problem
    std::shared_ptr<Discret::Elements::PoroFluidManager::PhaseManagerInterface> phase_manager_;

    //! variable manager of the porofluid problem
    std::shared_ptr<Discret::Elements::PoroFluidManager::VariableManagerInterface<num_dim_,
        num_nodes_homogenized_>>
        variable_manager_;

    //! scale vector
    std::vector<std::vector<int>> scale_vector_;
    //! function vector
    std::vector<std::vector<const Core::Utils::FunctionOfAnything*>> function_vector_;

    //! diameter function
    const Core::Utils::FunctionOfAnything* artery_diameter_funct_;

    //! string name used for scalars in function parser
    std::vector<std::string> scalar_names_;
    //! string name used for pressure in function parser
    std::vector<std::string> pressure_names_;
    //! string name used for saturation in function parser
    std::vector<std::string> saturation_names_;
    //! string name used for porosity in function parser
    const std::string porosity_name_;
    //! string name used for artery-pressure in function parser
    const std::string artery_pressure_name_;
    //! string name used for artery-scalars in function parser
    std::vector<std::string> artery_scalar_names_;
    //! string name used for volume fractions in function parser
    std::vector<std::string> volfrac_names_;
    //! string name used for volume fraction pressures in function parser
    std::vector<std::string> volfrac_pressure_names_;

    //! dofset of artery pressure in scatra discretization
    // TODO: find a better way to do this
    const int nds_scatra_artery_ = 2;

    //! dofset of scatra primary variable in artery discretization
    // TODO: find a better way to do this
    const int nds_artery_scatra_ = 2;

    //! segment id
    int segment_id_;

    //! number of integration patches in axial direction (for surface-based coupling)
    int num_patches_axial_;
    //! number of integration patches in radial direction (for surface-based coupling)
    int num_patches_radial_;

    //! right-hand side factor for artery time integration scaled with inverse density
    double timefacrhs_artery_density_;

    //! right-hand side factor for time integration of 2D/3D discretization
    // scaled with inverse density of specific phase or species
    std::vector<double> timefacrhs_homogenized_density_;

    //! right-hand side factor for artery time integration
    double timefacrhs_artery_;
    //! right-hand side factor for time integration of 2D/3D discretization
    double timefacrhs_homogenized_;

    //! vector where to assemble rhs-(function) coupling into
    //! summed up phase requires special treatment
    std::vector<std::vector<int>> homogenized_dofs_to_assemble_functions_into_;

    //! the artery material
    std::shared_ptr<Mat::Cnst1dArt> artery_material_;

    //! callback to retrieve function definitions by id
    std::function<const Core::Utils::FunctionOfAnything&(int)> function_of_anything_by_id_;

    //! rank of the process evaluating this pair
    int my_mpi_rank_;
  };

}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif
