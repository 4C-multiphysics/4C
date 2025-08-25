// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_POROFLUID_PRESSURE_BASED_HPP
#define FOUR_C_SCATRA_ELE_CALC_POROFLUID_PRESSURE_BASED_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_fluidporo_multiphase_reactions.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_porofluid_pressure_based_ele_calc.hpp"
#include "4C_porofluid_pressure_based_ele_calc_utils.hpp"
#include "4C_porofluid_pressure_based_ele_evaluator.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_porofluid_pressure_based_ele_phasemanager.hpp"
#include "4C_porofluid_pressure_based_ele_variablemanager.hpp"
#include "4C_scatra_ele_calc_poro_reac.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Mat
{
  class ScatraMat;
}

namespace Discret::Elements
{
  template <int nsd, int nen>
  class ScaTraEleInternalVariableManagerPorofluidPressureBased;

  template <Core::FE::CellType discretization_type>
  class ScaTraEleCalcPorofluidPressureBased final
      : public ScaTraEleCalcPoroReac<discretization_type>
  {
    /// private constructor, since we are a Singleton.
    ScaTraEleCalcPorofluidPressureBased(
        int number_of_dofs_per_node, int number_of_scalars, const std::string& discretization_name);

    using scatra_ele_calc = ScaTraEleCalc<discretization_type>;
    using poro_reaction = ScaTraEleCalcPoroReac<discretization_type>;
    using poro = ScaTraEleCalcPoro<discretization_type>;
    using advanced_reaction = ScaTraEleCalcAdvReac<discretization_type>;
    using scatra_ele_calc::nen_;
    using scatra_ele_calc::nsd_;

   public:
    /// Singleton access method
    static ScaTraEleCalcPorofluidPressureBased* instance(
        int number_of_dofs_per_node, int number_of_scalars, const std::string& discretization_name);

    /// Setup element evaluation
    int setup_calc(
        Core::Elements::Element* element, Core::FE::Discretization& discretization) override;

   protected:
    //! extract element based or nodal values
    void extract_element_and_node_values(Core::Elements::Element* element,
        Teuchos::ParameterList& parameters, Core::FE::Discretization& discretization,
        Core::Elements::LocationArray& location_array) override;

    //! extract element based or nodal values
    //! --> L2-projection case: called within extract_element_and_node_values
    void extract_nodal_flux(const Core::FE::Discretization& discretization,
        Core::Elements::LocationArray& location_array, int number_of_fluid_phases);

    //! set internal variables
    void set_internal_variables_for_mat_and_rhs() override;

    //! evaluate material
    void materials(
        std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
        int scalar_id,                                        //!< id of current scalar
        double& densn,                                        //!< density at t_(n)
        double& densnp,          //!< density at t_(n+1) or t_(n+alpha_F)
        double& densam,          //!< density at t_(n+alpha_M)
        double& viscosity,       //!< fluid viscosity
        int gauss_point_id = -1  //!< id of current gauss point (default = -1)
        ) override;

    //! material for scalar transport in a fluid phase
    void material_scatra_in_fluid(
        std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
        int scalar_id,                                        //!< id of current scalar
        double& densn,                                        //!< density at t_(n)
        double& densnp,          //!< density at t_(n+1) or t_(n+alpha_F)
        double& densam,          //!< density at t_(n+alpha_M)
        int gauss_point_id = -1  //!< id of current gauss point (default = -1)
    );

    //! material for scalar transport in a volume fraction
    void material_scatra_in_volfrac(
        std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
        int scalar_id,                                        //!< id of current scalar
        double& densn,                                        //!< density at t_(n)
        double& densnp,          //!< density at t_(n+1) or t_(n+alpha_F)
        double& densam,          //!< density at t_(n+alpha_M)
        int gauss_point_id = -1  //!< id of current gauss point (default = -1)
    );

    //! material for scalar transport in the solid phase
    void material_scatra_in_solid(
        std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
        int scalar_id,                                        //!< id of current scalar
        double& densn,                                        //!< density at t_(n)
        double& densnp,          //!< density at t_(n+1) or t_(n+alpha_F)
        double& densam,          //!< density at t_(n+alpha_M)
        int gauss_point_id = -1  //!< id of current gauss point (default = -1)
    );

    //! material for scalar transport of temperature (energy balance)
    void material_scatra_as_temperature(
        std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
        int scalar_id,                                        //!< id of current scalar
        double& densn,                                        //!< density at t_(n)
        double& densnp,          //!< density at t_(n+1) or t_(n+alpha_F)
        double& densam,          //!< density at t_(n+alpha_M)
        int gauss_point_id = -1  //!< id of current gauss point (default = -1)
    );

    //! Set advanced reaction terms and derivatives
    void set_advanced_reaction_terms(int scalar_id,                     //!< index of current scalar
        std::shared_ptr<Mat::MatListReactions> material_reaction_list,  //!< material reaction list
        const double* gauss_point_coordinates  //!< current Gauss-point coordinates
        ) override;

    //! compute pore pressure
    double compute_pore_pressure() override;

    //! get internal variable manager for pressure-based porofluid
    std::shared_ptr<ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd_, nen_>>
    variable_manager()
    {
      return std::static_pointer_cast<
          ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd_, nen_>>(
          scatra_ele_calc::scatravarmanager_);
    }

    //! calculation of convective element matrix in convective form
    //! the only difference to the base class version is, that there is no scaling with the
    //! density
    void calc_mat_conv(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        int scalar_id,                                    //!< index of current scalar
        double timefacfac,  //!< domain-integration factor times time-integration factor
        double densnp,      //!< density at time_(n+1)
        const Core::LinAlg::Matrix<nen_, 1>& sgconv  //!< subgrid-scale convective operator
        ) override;

    //! adaption of convective term for rhs
    //! the only difference to the base class version is, that there is no scaling with the
    //! density
    void recompute_conv_phi_for_rhs(int scalar_id,      //!< index of current scalar
        const Core::LinAlg::Matrix<nsd_, 1>& sgvelint,  //!< subgrid-scale velocity at Gauss point
        double densnp,                                  //!< density at time_(n+1)
        double densn,                                   //!< density at time_(n)
        double vdiv                                     //!< velocity divergence
        ) override;

    //! calculation of convective element matrix: add conservative contributions
    //! the only difference to the base class version is, that there is no scaling with the
    //! density
    void calc_mat_conv_add_cons(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        int scalar_id,                                    //!< index of current scalar
        double timefacfac,  //!< domain-integration factor times time-integration factor
        double vdiv,        //!< velocity divergence
        double densnp       //!< density at time_(n+1)
        ) override;

    //! calculation of mass element matrix (standard shape functions)
    void calc_mat_mass(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        const int& scalar_id,                             //!< index of current scalar
        const double& fac,                                //!< domain-integration factor
        const double& densam                              //!< density at time_(n+am)
        ) override;

    //! calculation of convective element matrix (OD term structure coupling)
    //! difference to base class: linearization of mesh motion + shapederivatives pressure
    //! gradient have to be included
    void calc_conv_od_mesh(Core::LinAlg::SerialDenseMatrix& element_matrix, int scalar_id,
        int number_of_dofs_per_node_mesh, double fac, double rhsfac, double densnp, double J,
        const Core::LinAlg::Matrix<nsd_, 1>& gradphi,
        const Core::LinAlg::Matrix<nsd_, 1>& convelint) override;

    //! calculation of linearized mass (off-diagonal/shapederivative term mesh)
    //! difference to base class: linearization of porosity is included
    void calc_lin_mass_od_mesh(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of ale element
        double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
        double fac,     //!< domain-integration factor
        double densam,  //!< density at time_(n+am)
        double densnp,  //!< density at time_(n+1)
        double phinp,   //!< scalar at time_(n+1)
        double hist,    //!< history of time integration
        double J,       //!< determinant of Jacobian det(dx/ds)
        const Core::LinAlg::Matrix<1, nsd_ * nen_>&
            dJ_dmesh  //!< derivative of det(dx/ds) w.r.t. mesh displacement
        ) override;

    //! standard Galerkin transient, old part of rhs and source term (off-diagonal/shapederivative
    //! term mesh) difference to base class: linearization of porosity and advanced reaction terms
    //! are included
    void calc_hist_and_source_od_mesh(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element current to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of ale element
        double fac,                        //!< domain-integration factor
        double rhsint,                     //!< rhs at Gauss point
        double J,                          //!< determinant of Jacobian det(dx/ds)
        const Core::LinAlg::Matrix<1, nsd_ * nen_>&
            dJ_dmesh,  //!< derivative of det(dx/ds) w.r.t. mesh displacement
        double densnp  //!< density
        ) override;

    //! standard Galerkin diffusive term (off-diagonal/shapederivative term mesh)
    //! difference to base class: linearization of porosity and effective diffusivity
    void calc_diff_od_mesh(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element current to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of ale element
        double diffcoeff,                  //!< diffusion coefficient
        double fac,                        //!< domain-integration factor
        double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
        double J,       //!< determinant of Jacobian det(dx/ds)
        const Core::LinAlg::Matrix<nsd_, 1>& gradphi,    //!< scalar gradient at Gauss point
        const Core::LinAlg::Matrix<nsd_, 1>& convelint,  //!< convective velocity
        const Core::LinAlg::Matrix<1, nsd_ * nen_>&
            dJ_dmesh  //!< derivative of det(dx/ds) w.r.t. mesh displacement
        ) override;

    //! reactive terms (standard Galerkin) (off-diagonal/shapederivative term mesh)
    //! difference to base class: linearization of porosity
    void calc_react_od_mesh(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element current to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of ale element
        double rhsfac,   //!< time-integration factor for rhs times domain-integration factor
        double rea_phi,  //!< reactive term
        double J,        //!< determinant of Jacobian det(dx/ds)
        const Core::LinAlg::Matrix<1, nsd_ * nen_>&
            dJ_dmesh  //!< derivative of det(dx/ds) w.r.t. mesh displacement
        ) override;

    //! calculation of convective element matrix in convective form (off-diagonal term fluid)
    void calc_mat_conv_od_fluid(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of fluid element // only a
                                           //!< dummy variable
        double rhsfac,  //!< domain-integration factor times time-integration factor
        double densnp,  //!< density at time_(n+1)
        const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient
        ) override;

    //! calculation of convective element matrix in convective form -- additional conservative
    //! contributions (off-diagonal term fluid) not yet implemented --> FOUR_C_THROW
    void calc_mat_conv_add_cons_od_fluid(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of fluid element
        double timefacfac,  //!< domain-integration factor times time-integration factor
        double densnp,      //!< density at time_(n+1)
        double phinp        //!< scalar at time_(n+1)
        ) override;

    //! calculation of linearized mass (off-diagonal porofluid terms)
    //! linearization of porosity * saturation * vtrans
    void calc_lin_mass_od_fluid(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of fluid element // only a
                                           //!< dummy variable
        double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
        double fac,     //!< domain-integration factor
        double densam,  //!< density at time_(n+am)
        double densnp,  //!< density at time_(n+1)
        double phinp,   //!< scalar at time_(n+1)
        double hist     //!< history of time integration
        ) override;

    //! calculate linearization of a mass matrix type matrix (off-diagonal porofluid terms)
    void calc_mass_matrix_type_od_porofluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
        int scalar_id, const std::vector<double>* per_factor_mass_matrix_OD_porofluid,
        int total_number_of_porofluid_dofs, double pre_factor);

    //! standard Galerkin transient, old part of rhs and source term (off-diagonal porofluid terms)
    //! linearization of porosity * saturation * vtrans + advanced reaction terms
    void calc_hist_and_source_od_fluid(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element current to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of ale element
        double fac,                        //!< domain-integration factor
        double rhsint,                     //!< rhs at Gauss point
        double densnp                      //!< density
        ) override;

    //! standard Galerkin reactive term (off-diagonal porofluid terms)
    //! linearization of porosity * saturation * vreact
    void calc_react_od_fluid(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element current to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of ale element
        double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
        double rea_phi  //!< rhs at Gauss point
        ) override;

    //! standard Galerkin diffusive term (off-diagonal porofluid terms)
    //! linearization of porosity*saturation*d_eff
    void calc_diff_od_fluid(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element current to be filled
        int scalar_id,                                    //!< index of current scalar
        int number_of_dofs_per_node_mesh,  //!< number of dofs per node of ale element
        double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
        const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient at Gauss point
        ) override;

    //! fill coupling vector and add variables to reaction in order to compute reaction values and
    //! derivatives
    void fill_coupling_vector_and_add_variables(Mat::MatListReactions& material_list_reactions,
        ScaTraEleReaManagerAdvReac& reaction_manager);

    //! Get right hand side including reaction body force term
    void get_rhs_int(double& rhsint,  //!< rhs containing body force at Gauss point
        double densnp,                //!< density at t_(n+1)
        int scalar_id                 //!< index of current scalar
        ) override;

    //! calculation of reactive element matrix
    void calc_mat_react(
        Core::LinAlg::SerialDenseMatrix& element_matrix,  //!< element matrix to be filled
        int scalar_id,                                    //!< index of current scalar
        double timefacfac,  //!< domain-integration factor times time-integration factor
        double timetaufac,  //!< domain-integration factor times time-integration factor times tau
        double taufac,      //!< domain-integration factor times tau
        double densnp,      //!< density at time_(n+1)
        const Core::LinAlg::Matrix<nen_, 1>& sgconv,  //!< subgrid-scale convective operator
        const Core::LinAlg::Matrix<nen_, 1>& diff     //!< laplace term
        ) override;

    //! "shape derivatives" pressure gradient
    void apply_shape_derivs_pressure_grad(Core::LinAlg::SerialDenseMatrix& element_matrix,
        int scalar_id, double vrhs, const Core::LinAlg::Matrix<nsd_, 1>& gradient_phi,
        Core::LinAlg::Matrix<nsd_, 1> pressure_gradient_reference);

    //! nodal flux values at t_(n+1)
    std::vector<Core::LinAlg::Matrix<nsd_, nen_>> element_flux_np_;

    //! a vector containing all quantities, the equation is coupled with
    //! (i.e. pressures, saturations and porosity)
    std::vector<std::pair<std::string, double>> coupling_values_;

    //! penalty factor to avoid very small "densities"
    const double penalty_ = 1.0;

    //! do we use L2-projection or evaluation at GP
    bool enable_L2_projection_ = false;
  };


  template <int nsd, int nen>
  class ScaTraEleInternalVariableManagerPorofluidPressureBased final
      : public ScaTraEleInternalVariableManager<nsd, nen>
  {
    using scatra_variable_manager = ScaTraEleInternalVariableManager<nsd, nen>;

   public:
    explicit ScaTraEleInternalVariableManagerPorofluidPressureBased(int number_of_scalars)
        : ScaTraEleInternalVariableManager<nsd, nen>(number_of_scalars),
          pressure_(0),
          saturation_(0),
          density_(0),
          solid_pressure_(0.0),
          delta_(number_of_scalars, 0.0),
          relative_mobility_funct_id_(number_of_scalars),
          heat_capacity_(0),
          thermal_diffusivity_(0),
          effective_heat_capacity_(0.0),
          minimal_value_of_phase_(number_of_scalars, 0.0),
          evaluate_scalar_(number_of_scalars, true),
          scalar_to_phase_map_(
              number_of_scalars, {-1, Mat::ScaTraMatMultiPoro::SpeciesType::species_undefined}),
          material_is_set_(false),
          action_(ScaTra::Action::calc_mat_and_rhs)
    {
    }

    //! compute and set internal variables -- no L2-projection but evaluation at GP
    void set_internal_variables_porofluid_pressure_based(
        const Core::LinAlg::Matrix<nen, 1>& shape_functions,  //! array for shape functions
        const Core::LinAlg::Matrix<nsd, nen>&
            shape_functions_deriv_xyz,  //! global derivatives of shape functions w.r.t x,y,z
        const Core::LinAlg::Matrix<nsd, nen>&
            shape_functions_deriv,  //! global derivatives of shape functions w.r.t r,s,t
        const Core::LinAlg::Matrix<nsd, nsd>& jacobian_matrix,
        const Core::LinAlg::Matrix<nsd, nen>& node_coordinates_reference,
        const std::vector<Core::LinAlg::Matrix<nen, 1>>&
            element_phi_np,  //! scalar at t_(n+1) or t_(n+alpha_F)
        const std::vector<Core::LinAlg::Matrix<nen, 1>>& element_phi_n,  //! scalar at t_(n)
        const std::vector<Core::LinAlg::Matrix<nen, 1>>&
            element_hist,  //! history vector of transported scalars
        const Core::LinAlg::Matrix<nsd, nen>&
            element_force_velocity  //! nodal velocity due to external force
    );

    //! adapt convective term in case of L2-projection
    void adapt_convective_term_for_l2(
        const Core::LinAlg::Matrix<nen, 1>& shape_functions,  //! array for shape functions
        const Core::LinAlg::Matrix<nsd, nen>&
            shape_functions_deriv_xyz,  //! global derivatives of shape functions w.r.t x,y,z
        const std::vector<Core::LinAlg::Matrix<nsd, nen>>&
            element_flux_np  //! nodal flux values at t_(n+1) or t_(n+alpha_F)
    );

    //! Set the porofluid-material in the scatra variable manager
    void set_porofluid_material(const Core::Elements::Element* element)
    {
      // check if we actually have three materials
      if (element->num_material() < 3) FOUR_C_THROW("no third material available");

      // requires PoroMultiPhase material to be the third material
      porofluid_material_ = std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(
          element->material(nds_scatra_porofluid_));
      if (porofluid_material_ == nullptr) FOUR_C_THROW("cast to Mat::FluidPoroMultiPhase failed!");

      material_is_set_ = true;
    }

    //! get pressure associated with scalar
    [[nodiscard]] double get_pressure(const int scalar_id) const
    {
      switch (scalar_to_phase_map_[scalar_id].species_type)
      {
        case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          return pressure_[scalar_to_phase_map_[scalar_id].phaseID];

        default:
          FOUR_C_THROW("scalar_to_phase_id = {} for species {}",
              scalar_to_phase_map_[scalar_id].phaseID, scalar_id);
      }
    }

    //! get pressures
    [[nodiscard]] const std::vector<double>& get_pressure() const { return pressure_; }

    //! get saturation associated with scalar
    [[nodiscard]] double get_saturation(const int scalar_id) const
    {
      switch (scalar_to_phase_map_[scalar_id].species_type)
      {
        case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
          return saturation_[scalar_to_phase_map_[scalar_id].phaseID];

        default:
          FOUR_C_THROW("scalar_to_phase_id = {} for species {}",
              scalar_to_phase_map_[scalar_id].phaseID, scalar_id);
      }
    }

    //! get saturation associated with scalar
    [[nodiscard]] const std::vector<double>& get_saturation() const { return saturation_; }

    //! get density associated with scalar
    [[nodiscard]] double get_density(const int scalar_id) const
    {
      switch (scalar_to_phase_map_[scalar_id].species_type)
      {
        case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
        case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          return density_[scalar_to_phase_map_[scalar_id].phaseID];

        case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
          return phase_manager_->solid_density();

        case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
          // set to 1.0 because densities are included in the effective heat capacity
          return 1.0;

        default:
          FOUR_C_THROW("Unknown species type {} for species {}.",
              scalar_to_phase_map_[scalar_id].species_type, scalar_id);
      }
    }

    //! return density vector
    [[nodiscard]] const std::vector<double>& get_density() const { return density_; }

    //! return volfrac vector
    [[nodiscard]] const std::vector<double>& get_volume_fraction() const
    {
      return volume_fractions_;
    }

    //! get volume fraction associated with scalar
    [[nodiscard]] double get_volume_fraction(const int scalar_id) const
    {
      switch (scalar_to_phase_map_[scalar_id].species_type)
      {
        case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
          return volume_fractions_[get_phase_id(scalar_id) - phase_manager_->num_fluid_phases()];

        default:
          FOUR_C_THROW("scalar_to_phase_id = {} for species {}",
              scalar_to_phase_map_[scalar_id].phaseID, scalar_id);
      }
    }

    //! get volfrac pressure vector
    [[nodiscard]] const std::vector<double>& get_volume_fractions_pressure() const
    {
      return volume_fractions_pressure_;
    }

    //! get solid pressure
    [[nodiscard]] double get_solid_pressure() const { return solid_pressure_; };

    //! set scalar ID to phase ID mapping and species type
    void set_phase_id_and_species_type(const int scalar_id, const int phase_id,
        const Mat::ScaTraMatMultiPoro::SpeciesType& species_type)
    {
      scalar_to_phase_map_[scalar_id].phaseID = phase_id;
      scalar_to_phase_map_[scalar_id].species_type = species_type;
    }

    //! get phase ID from scalar ID
    [[nodiscard]] int get_phase_id(const int scalar_id) const
    {
      if (scalar_to_phase_map_[scalar_id].phaseID < 0)
        FOUR_C_THROW("scalar_to_phase_id = {} for species {}",
            scalar_to_phase_map_[scalar_id].phaseID, scalar_id);

      return scalar_to_phase_map_[scalar_id].phaseID;
    }

    //! get species type of scalar
    [[nodiscard]] Mat::ScaTraMatMultiPoro::SpeciesType get_species_type(const int scalar_id) const
    {
      return scalar_to_phase_map_[scalar_id].species_type;
    }

    //! set delta for evaluation of effective diffusivity
    void set_delta(const double delta, const int scalar_id) { delta_[scalar_id] = delta; }

    //! set relative mobility function ID
    void set_relative_mobility_function_id(
        const int relative_mobility_funct_id, const int scalar_id)
    {
      relative_mobility_funct_id_[scalar_id] = relative_mobility_funct_id;
    }

    //! set heat capacity
    void set_heat_capacity(const std::vector<double>& heat_capacity)
    {
      heat_capacity_ = heat_capacity;
    }

    //! set effective heat capacity
    void set_effective_heat_capacity(const double effective_heat_capacity)
    {
      effective_heat_capacity_ = effective_heat_capacity;
    }

    //! set thermal diffusivity
    void set_thermal_diffusivity(const std::vector<double>& kappa) { thermal_diffusivity_ = kappa; }

    //! set minimum value of corresponding phase under which we assume that mass fraction is zero
    void set_minimal_value_of_phase(const double minimal_value_of_phase, const int scalar_id)
    {
      minimal_value_of_phase_[scalar_id] = minimal_value_of_phase;
    }

    //! set action
    void set_action(const ScaTra::Action action) { action_ = action; };

    //! get delta
    [[nodiscard]] double get_delta(const int phase_id) const { return delta_[phase_id]; }

    //! get heat capacity: order [ <fluid>  <volfrac>  <solid> ]
    [[nodiscard]] double get_heat_capacity(const int phase_id) const
    {
      return heat_capacity_[phase_id];
    }

    //! get effective heat capacity
    [[nodiscard]] double get_effective_heat_capacity() const { return effective_heat_capacity_; }

    //! get thermal diffusivity: order [ <fluid>  <volfrac>  <solid> ]
    [[nodiscard]] double get_thermal_diffusivity(const int phase_id) const
    {
      return thermal_diffusivity_[phase_id];
    }

    //! get evaluate scalar flag
    bool evaluate_scalar(const int scalar_id) { return evaluate_scalar_[scalar_id]; }

    //! get minimum value of corresponding phase under which we assume that mass fraction is zero
    [[nodiscard]] double get_minimal_value_of_phase(const int scalar_id) const
    {
      return minimal_value_of_phase_[scalar_id];
    }

    //! get pre-factor needed for off-diagonal mesh-linearization of mass matrix
    double get_pre_factor_mass_matrix_od_mesh(int scalar_id, double fac);

    //! get pre-factor needed for off-diagonal porofluid-linearization of mass matrix
    void get_pre_factor_mass_matrix_od_porofluid(
        int scalar_id, std::vector<double>* pre_factor_mass_matrix_od_porofluid);

    //! evaluate relative mobility
    auto evaluate_relative_mobility(int current_scalar) const;

    //! get pre-factor needed for off-diagonal fluid-linearization of convective term
    void get_pre_factor_convection_od_porofluid(unsigned ui,
        std::vector<double>* pre_factor_convection_od_porofluid,
        const Core::LinAlg::Matrix<nsd, 1>& grad_phi,
        const Core::LinAlg::Matrix<1, nsd>& grad_phi_transpose_diffusion_tensor,
        const Core::LinAlg::Matrix<nen, 1>& shape_functions,
        const Core::LinAlg::Matrix<nsd, nen>& shape_functions_deriv_xyz, int phase_id);

    //! get pre-factor needed for off-diagonal mesh-linearization of diffusive term
    double get_pre_factor_diffusion_od_mesh(
        int scalar_id, double rhsfac, double diffusion_coefficient);

    //! get pre-factor needed for off-diagonal porofluid linearization of diffusive term
    void get_pre_factor_diffusion_od_porofluid(int scalar_id, double rhsfac,
        double diffusion_coefficient, std::vector<double>* pre_factor_diffusion_OD_porofluid);

    //! get diffusion tensor of fluid phases (permeability * relative permeability)/viscosity
    void get_diffusion_tensor_fluid(
        int scalar_id, Core::LinAlg::Matrix<nsd, nsd>& diffusion_tensor, int phase_id);

    //! compute the pressure gradient in reference configuration
    void get_pressure_gradient_reference(const Core::LinAlg::Matrix<nsd, nsd>& jacobian_matrix,
        Core::LinAlg::Matrix<nsd, 1>& pressure_gradient_reference, int phase_id);

    //! set flag if we actually evaluate the scalar or set it to zero
    void set_evaluate_scalar_flag(int scalar_id);

    //! get pre-factor needed for off-diagonal mesh-linearization of hist and source term
    double get_pre_factor_hist_and_source_od_mesh(
        int scalar_id, double fac, double densnp, double hist, double rhsint);

    //! get action
    [[nodiscard]] ScaTra::Action get_action() const { return action_; };

    //! return pressure gradient of current phase
    const Core::LinAlg::Matrix<nsd, 1>& get_pressure_gradient(const int current_phase) const
    {
      return pressure_gradient_[current_phase];
    }

    //! return absolute value of pressure gradient of current phase
    [[nodiscard]] double get_pressure_gradient_norm(const int current_phase) const
    {
      return pressure_gradient_norm_[current_phase];
    }

    //! return porofluid material
    std::shared_ptr<Mat::FluidPoroMultiPhase> porofluid_material()
    {
      if (!material_is_set_)
        FOUR_C_THROW(
            "Pressure-based porofluid material has not yet been set in the variable manager.");

      return porofluid_material_;
    }

    //! Setup phase manager and variable manager of fluid
    void setup_porofluid_managers(const Core::Elements::Element* element,
        const Core::FE::Discretization& discretization, int number_of_fluid_phases,
        int total_number_of_porofluid_dofs, const int number_volfracs);

    //! extract the element and node values of the porofluid from variable manager
    void extract_element_and_node_values_of_porofluid(Core::Elements::Element* element,
        Core::FE::Discretization& discretization, Core::Elements::LocationArray& location_array,
        Core::LinAlg::Matrix<nsd, nen>& node_coordinates);

    //! get the phase manager of the fluid
    std::shared_ptr<PoroFluidManager::PhaseManagerInterface> porofluid_phase_manager()
    {
      return phase_manager_;
    }

   private:
    //! phase pressure
    std::vector<double> pressure_;
    //! phase saturation
    std::vector<double> saturation_;
    //! phase density
    std::vector<double> density_;
    //! solid pressure
    double solid_pressure_;
    //! pressure gradient
    std::vector<Core::LinAlg::Matrix<nsd, 1>> pressure_gradient_;
    //! diffusion tensor of porofluid
    std::vector<Core::LinAlg::Matrix<nsd, nsd>> diffusion_tensor_porofluid_;
    //! norm of pressure-gradient
    std::vector<double> pressure_gradient_norm_;
    //! volume fraction
    std::vector<double> volume_fractions_;
    //! volume fraction pressure
    std::vector<double> volume_fractions_pressure_;

    //! pressure-based porofluid material
    std::shared_ptr<Mat::FluidPoroMultiPhase> porofluid_material_;

    //! delta for effective diffusivity
    std::vector<double> delta_;

    //! function IDs of relative mobility functions
    std::vector<int> relative_mobility_funct_id_;

    //! heat capacity: order [ <fluid>  <volfrac>  <solid> ]
    std::vector<double> heat_capacity_;
    //! thermal diffusivity: order [ <fluid>  <volfrac>  <solid> ]
    std::vector<double> thermal_diffusivity_;
    //! effective heat capacity
    double effective_heat_capacity_;

    //! minimum saturation under which the corresponding mass fraction is assumed to be zero
    std::vector<double> minimal_value_of_phase_;
    //! flag to check if we have to evaluate the species equation in this element
    std::vector<bool> evaluate_scalar_;
    //! scalar to phase map
    std::vector<Mat::ScaTraMatMultiPoro::ScalarToPhaseMap> scalar_to_phase_map_;
    //! check if multiphase material has been set
    bool material_is_set_;

    ScaTra::Action action_;

    //! phase manager of the fluid
    std::shared_ptr<PoroFluidManager::PhaseManagerInterface> phase_manager_;

    //! variable manager of the fluid
    std::shared_ptr<PoroFluidManager::VariableManagerInterface<nsd, nen>> variable_manager_;

    //! dofset of fluid field on scatra discretization
    // TODO: find a better way to do this
    const int nds_scatra_porofluid_ = 2;
  };

}  // namespace Discret::Elements


FOUR_C_NAMESPACE_CLOSE

#endif
