// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_porofluid_pressure_based.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_list_reactions.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_scatra_multiporo.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::ScaTraEleCalcPorofluidPressureBased(const int number_of_dofs_per_node,
    const int number_of_scalars, const std::string& discretization_name)
    : ScaTraEleCalc<discretization_type>::ScaTraEleCalc(
          number_of_dofs_per_node, number_of_scalars, discretization_name),
      ScaTraEleCalcPoro<discretization_type>::ScaTraEleCalcPoro(
          number_of_dofs_per_node, number_of_scalars, discretization_name),
      ScaTraEleCalcAdvReac<discretization_type>::ScaTraEleCalcAdvReac(
          number_of_dofs_per_node, number_of_scalars, discretization_name),
      ScaTraEleCalcPoroReac<discretization_type>::ScaTraEleCalcPoroReac(
          number_of_dofs_per_node, number_of_scalars, discretization_name),
      element_flux_np_(0)
{
  // replace internal variable manager by internal variable manager for pressure-based porofluid
  scatra_ele_calc::scatravarmanager_ =
      std::make_shared<ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd_, nen_>>(
          scatra_ele_calc::numscal_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>*
Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::instance(
    const int number_of_dofs_per_node, const int number_of_scalars,
    const std::string& discretization_name)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [number_of_dofs_per_node, number_of_scalars, &discretization_name]()
      {
        return std::unique_ptr<ScaTraEleCalcPorofluidPressureBased>(
            new ScaTraEleCalcPorofluidPressureBased(
                number_of_dofs_per_node, number_of_scalars, discretization_name));
      });

  return singleton_map[discretization_name].instance(Core::Utils::SingletonAction::create);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
int Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::setup_calc(
    Core::Elements::Element* element, Core::FE::Discretization& discretization)
{
  poro_reaction::setup_calc(element, discretization);

  // get the material
  std::shared_ptr<Core::Mat::Material> material = element->material();

  // set the porofluid material in the element
  variable_manager()->set_porofluid_material(element);

  if (material->material_type() == Core::Materials::m_matlist or
      material->material_type() == Core::Materials::m_matlist_reactions)
  {
    const std::shared_ptr<const Mat::MatList>& material_list =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (material_list->num_mat() < scatra_ele_calc::numdofpernode_)
      FOUR_C_THROW("Not enough materials in MatList.");

    for (int scalar_id = 0; scalar_id < scatra_ele_calc::numdofpernode_; ++scalar_id)
    {
      int material_id = material_list->mat_id(scalar_id);
      std::shared_ptr<Core::Mat::Material> single_material =
          material_list->material_by_id(material_id);

      switch (single_material->material_type())
      {
        case Core::Materials::m_scatra_in_fluid_porofluid_pressure_based:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& scatra_material =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(single_material);

          // smaller zero or greater equal to the number of fluid phases
          if (scatra_material->phase_id() < 0 or
              scatra_material->phase_id() >=
                  variable_manager()->porofluid_material()->num_fluid_phases())
          {
            FOUR_C_THROW(
                "Invalid phase ID {} for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)",
                scatra_material->phase_id(), scalar_id);
          }

          const int porofluid_material_id =
              variable_manager()->porofluid_material()->mat_id(scatra_material->phase_id());
          std::shared_ptr<Core::Mat::Material> porofluid_material =
              variable_manager()->porofluid_material()->material_by_id(porofluid_material_id);

          if (porofluid_material->material_type() != Core::Materials::m_fluidporo_singlephase)
          {
            FOUR_C_THROW(
                "Invalid phase ID for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)",
                scalar_id);
          }

          variable_manager()->set_phase_id_and_species_type(scalar_id, scatra_material->phase_id(),
              Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid);
          // set delta in the variablemanager
          variable_manager()->set_delta(scatra_material->delta(), scalar_id);
          // set minimum saturation in the variablemanager
          variable_manager()->set_minimal_value_of_phase(scatra_material->min_sat(), scalar_id);
          // set reacts to external force
          variable_manager()->set_reacts_to_force(
              scatra_material->reacts_to_external_force(), scalar_id);
          // set relative mobility function ID
          variable_manager()->set_relative_mobility_function_id(
              scatra_material->relative_mobility_funct_id(), scalar_id);
          break;
        }
        case Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& scatra_material =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(single_material);

          // smaller zero or greater equal the number of fluid phases + number of volume fractions
          if (scatra_material->phase_id() <
                  variable_manager()->porofluid_material()->num_fluid_phases() or
              scatra_material->phase_id() >=
                  variable_manager()->porofluid_material()->num_fluid_phases() +
                      variable_manager()->porofluid_material()->num_vol_frac())
          {
            FOUR_C_THROW(
                "Invalid phase ID {} for scalar {} (species in volume fraction = "
                "MAT_scatra_multiporo_volfrac)",
                scatra_material->phase_id(), scalar_id);
          }

          const int porofluid_material_id =
              variable_manager()->porofluid_material()->mat_id(scatra_material->phase_id());
          std::shared_ptr<Core::Mat::Material> porofluid_material =
              variable_manager()->porofluid_material()->material_by_id(porofluid_material_id);

          if (porofluid_material->material_type() != Core::Materials::m_fluidporo_singlevolfrac &&
              porofluid_material->material_type() !=
                  Core::Materials::m_fluidporo_volfrac_pressure_blood_lung)
          {
            FOUR_C_THROW(
                "Invalid phase ID for scalar {} (species in volume fraction = "
                "MAT_scatra_multiporo_volfrac)",
                scalar_id);
          }

          variable_manager()->set_phase_id_and_species_type(scalar_id, scatra_material->phase_id(),
              Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac);
          // set delta in the variablemanager
          variable_manager()->set_delta(scatra_material->delta(), scalar_id);
          // set reacts to external force
          variable_manager()->set_reacts_to_force(
              scatra_material->reacts_to_external_force(), scalar_id);
          // set relative mobility function ID
          variable_manager()->set_relative_mobility_function_id(
              scatra_material->relative_mobility_funct_id(), scalar_id);

          break;
        }
        case Core::Materials::m_scatra_in_solid_porofluid_pressure_based:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroSolid>& scatra_material =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroSolid>(single_material);

          // set delta in the variablemanager
          variable_manager()->set_delta(scatra_material->delta(), scalar_id);

          // dummy value -1000 for phaseID because species in solid do not have a phaseID
          variable_manager()->set_phase_id_and_species_type(
              scalar_id, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid);

          break;
        }
        case Core::Materials::m_scatra_as_temperature_porofluid_pressure_based:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroTemperature>& scatra_material =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroTemperature>(single_material);

          // assemble heat capacities of fluid phases, volume fractions and solid phase
          // cp order [ <fluid>  <volfrac>  <solid> ]
          std::vector<double> heat_capacity;
          std::vector heat_capacity_fluid(scatra_material->cp_fluid());
          std::vector heat_capacity_volfrac(scatra_material->cp_volfrac());

          heat_capacity.insert(
              heat_capacity.begin(), heat_capacity_fluid.begin(), heat_capacity_fluid.end());
          heat_capacity.insert(
              heat_capacity.end(), heat_capacity_volfrac.begin(), heat_capacity_volfrac.end());
          heat_capacity.insert(heat_capacity.end(), scatra_material->cp_solid());

          variable_manager()->set_heat_capacity(heat_capacity);

          // assemble thermal diffusivity of fluid phases, volume fractions and solid phase
          // kappa order [ <fluid>  <volfrac>  <solid> ]
          std::vector<double> thermal_diffusivity;
          std::vector thermal_diffusivity_fluid(scatra_material->kappa_fluid());
          std::vector thermal_diffusivity_volfrac(scatra_material->kappa_volfrac());

          thermal_diffusivity.insert(thermal_diffusivity.begin(), thermal_diffusivity_fluid.begin(),
              thermal_diffusivity_fluid.end());
          thermal_diffusivity.insert(thermal_diffusivity.end(), thermal_diffusivity_volfrac.begin(),
              thermal_diffusivity_volfrac.end());
          thermal_diffusivity.insert(thermal_diffusivity.end(), scatra_material->kappa_solid());

          variable_manager()->set_thermal_diffusivity(thermal_diffusivity);

          // dummy value -1000 for phaseID because temperature does not have a phaseID
          variable_manager()->set_phase_id_and_species_type(
              scalar_id, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature);

          break;
        }
        default:
          FOUR_C_THROW(
              "Material type {} is not supported for multiphase flow through porous media!",
              single_material->material_type());
      }
    }
  }
  else
  {
    switch (material->material_type())
    {
      case Core::Materials::m_scatra_in_fluid_porofluid_pressure_based:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& scatra_material =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(material);

        // smaller zero or greater equal the number of fluid phases
        if (scatra_material->phase_id() < 0 or
            scatra_material->phase_id() >=
                variable_manager()->porofluid_material()->num_fluid_phases())
        {
          FOUR_C_THROW(
              "Invalid phase ID {} for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)",
              scatra_material->phase_id(), 0);
        }

        const int porofluid_material_id =
            variable_manager()->porofluid_material()->mat_id(scatra_material->phase_id());
        std::shared_ptr<Core::Mat::Material> porofluid_material =
            variable_manager()->porofluid_material()->material_by_id(porofluid_material_id);

        if (porofluid_material->material_type() != Core::Materials::m_fluidporo_singlephase)
          FOUR_C_THROW(
              "Invalid phase ID for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)", 0);

        variable_manager()->set_phase_id_and_species_type(
            0, scatra_material->phase_id(), Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid);
        // set delta in the variablemanager
        variable_manager()->set_delta(scatra_material->delta(), 0);
        // set minimum saturation in the variablemanager
        variable_manager()->set_minimal_value_of_phase(scatra_material->min_sat(), 0);
        // set reacts to external force
        variable_manager()->set_reacts_to_force(scatra_material->reacts_to_external_force(), 0);
        // set relative mobility function ID
        variable_manager()->set_relative_mobility_function_id(
            scatra_material->relative_mobility_funct_id(), 0);
        break;
      }
      case Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(material);

        // smaller zero or greater equal the number of fluid phases + number of volume fractions
        if (poromat->phase_id() < 0 or
            poromat->phase_id() >= variable_manager()->porofluid_material()->num_fluid_phases() +
                                       variable_manager()->porofluid_material()->num_vol_frac())
        {
          FOUR_C_THROW(
              "Invalid phase ID {} for scalar {} (species in volume fraction = "
              "MAT_scatra_multiporo_volfrac)",
              poromat->phase_id(), 0);
        }

        const int porofluid_material_id =
            variable_manager()->porofluid_material()->mat_id(poromat->phase_id());
        std::shared_ptr<Core::Mat::Material> porofluid_material =
            variable_manager()->porofluid_material()->material_by_id(porofluid_material_id);

        if (porofluid_material->material_type() != Core::Materials::m_fluidporo_singlevolfrac &&
            porofluid_material->material_type() !=
                Core::Materials::m_fluidporo_volfrac_pressure_blood_lung)
        {
          FOUR_C_THROW(
              "Invalid phase ID for scalar {} (species in volume fraction = "
              "MAT_scatra_multiporo_volfrac)",
              0);
        }

        variable_manager()->set_phase_id_and_species_type(
            0, poromat->phase_id(), Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac);
        // set delta in the variablemanager
        variable_manager()->set_delta(poromat->delta(), 0);
        // set reacts to external force
        variable_manager()->set_reacts_to_force(poromat->reacts_to_external_force(), 0);
        // set relative mobility function ID
        variable_manager()->set_relative_mobility_function_id(
            poromat->relative_mobility_funct_id(), 0);

        break;
      }
      case Core::Materials::m_scatra_in_solid_porofluid_pressure_based:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroSolid>& scatra_material =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroSolid>(material);

        // set delta in the variablemanager
        variable_manager()->set_delta(scatra_material->delta(), 0);

        // dummy value -1000 for phaseID because species in solid do not have a phaseID
        variable_manager()->set_phase_id_and_species_type(
            0, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid);

        break;
      }
      case Core::Materials::m_scatra_as_temperature_porofluid_pressure_based:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroTemperature>& scatra_material =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroTemperature>(material);

        // assemble heat capacities of fluid phases, volume fractions and solid phase
        std::vector<double> heat_capacity;
        std::vector heat_capacity_fluid(scatra_material->cp_fluid());
        std::vector heat_capacity_volfrac(scatra_material->cp_volfrac());

        heat_capacity.insert(
            heat_capacity.begin(), heat_capacity_fluid.begin(), heat_capacity_fluid.end());
        heat_capacity.insert(
            heat_capacity.end(), heat_capacity_volfrac.begin(), heat_capacity_volfrac.end());
        heat_capacity.insert(heat_capacity.end(), scatra_material->cp_solid());

        variable_manager()->set_heat_capacity(heat_capacity);

        // assemble thermal diffusivity of fluid phases, volume fractions and solid phase
        std::vector<double> thermal_diffusivity;
        std::vector thermal_diffusivity_fluid(scatra_material->kappa_fluid());
        std::vector thermal_diffusivity_volfrac(scatra_material->kappa_volfrac());

        thermal_diffusivity.insert(thermal_diffusivity.begin(), thermal_diffusivity_fluid.begin(),
            thermal_diffusivity_fluid.end());
        thermal_diffusivity.insert(thermal_diffusivity.end(), thermal_diffusivity_volfrac.begin(),
            thermal_diffusivity_volfrac.end());
        thermal_diffusivity.insert(thermal_diffusivity.end(), scatra_material->kappa_solid());

        variable_manager()->set_thermal_diffusivity(thermal_diffusivity);

        // dummy value -1000 for phaseID because temperature does not have a phaseID
        variable_manager()->set_phase_id_and_species_type(
            0, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature);

        break;
      }
      default:
        FOUR_C_THROW("Material type {} is not supported for multiphase flow through porous media!",
            material->material_type());
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::extract_element_and_node_values(Core::Elements::Element* element,
    Teuchos::ParameterList& parameters, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& location_array)
{
  // extract action parameter
  const auto action = Teuchos::getIntegralValue<ScaTra::Action>(parameters, "action");
  variable_manager()->set_action(action);

  //---------------------------------------------------------------------------------------------
  //                                 STRUCTURE
  //---------------------------------------------------------------------------------------------

  // get additional state vector for ALE case: grid displacement
  if (scatra_ele_calc::scatrapara_->is_ale())
  {
    // get number of dofset associated with displacement related dofs
    const int nds_displacement = scatra_ele_calc::scatrapara_->nds_disp();

    const std::shared_ptr<const Core::LinAlg::Vector<double>> displacement_np =
        discretization.get_state(nds_displacement, "dispnp");
    if (displacement_np == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int number_of_displacement_dofs_per_node =
        location_array[nds_displacement].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> location_vector(nsd_ * nen_, -1);
    for (unsigned inode = 0; inode < nen_; ++inode)
    {
      for (unsigned idim = 0; idim < nsd_; ++idim)
      {
        location_vector[inode * nsd_ + idim] =
            location_array[nds_displacement]
                .lm_[inode * number_of_displacement_dofs_per_node + idim];
      }
    }

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(
        *displacement_np, scatra_ele_calc::edispnp_, location_vector);

    // add nodal displacements to point coordinates
    scatra_ele_calc::update_node_coordinates();
  }
  else
  {
    scatra_ele_calc::edispnp_.clear();
  }

  //---------------------------------------------------------------------------------------------
  //                                 SCATRA
  //---------------------------------------------------------------------------------------------

  // extract local values from the global vectors
  const std::shared_ptr<const Core::LinAlg::Vector<double>> hist = discretization.get_state("hist");
  const std::shared_ptr<const Core::LinAlg::Vector<double>> phi_np =
      discretization.get_state("phinp");
  if (hist == nullptr || phi_np == nullptr)
    FOUR_C_THROW("Cannot get state vector 'hist' and/or 'phinp'");

  // values of scatra field are always in first dofset
  const std::vector<int>& global_dof_ids = location_array[0].lm_;
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(
      *hist, scatra_ele_calc::ehist_, global_dof_ids);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(
      *phi_np, scatra_ele_calc::ephinp_, global_dof_ids);

  if (scatra_ele_calc::scatraparatimint_->is_gen_alpha() and
      not scatra_ele_calc::scatraparatimint_->is_incremental())
  {
    // extract additional local values from global vector
    const std::shared_ptr<const Core::LinAlg::Vector<double>> phi_n =
        discretization.get_state("phin");
    if (phi_n == nullptr) FOUR_C_THROW("Cannot get state vector 'phin'");
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(
        *phi_n, scatra_ele_calc::ephin_, global_dof_ids);
  }

  if (scatra_ele_calc::scatrapara_->has_external_force())
  {
    // get number of dofset associated with velocity related dofs
    const auto nds_velocity = scatra_ele_calc::scatrapara_->nds_vel();
    const auto force_velocity = discretization.get_state(nds_velocity, "force_velocity");

    const int number_of_dofs_per_node =
        location_array[nds_velocity].lm_.size() / scatra_ele_calc::nen_;
    std::vector<int> location_vector(scatra_ele_calc::nsd_ * scatra_ele_calc::nen_, -1);
    for (unsigned inode = 0; inode < scatra_ele_calc::nen_; ++inode)
    {
      for (unsigned idim = 0; idim < scatra_ele_calc::nsd_; ++idim)
        location_vector[inode * scatra_ele_calc::nsd_ + idim] =
            location_array[nds_velocity].lm_[inode * number_of_dofs_per_node + idim];
    }

    Core::FE::extract_my_values<Core::LinAlg::Matrix<scatra_ele_calc::nsd_, scatra_ele_calc::nen_>>(
        *force_velocity, scatra_ele_calc::eforcevelocity_, location_vector);
  }

  //---------------------------------------------------------------------------------------------
  //                                 FLUID
  //---------------------------------------------------------------------------------------------

  // get number of dofset associated with pressure/fluid related dofs
  const int nds_pressure = scatra_ele_calc::scatrapara_->nds_pres();

  // determine number of velocity related dofs per node (= number of phases)
  const int number_of_fluid_phases = variable_manager()->porofluid_material()->num_fluid_phases();
  const int total_number_of_porofluid_dofs = variable_manager()->porofluid_material()->num_mat();

  // extract element and node values of the porofluid
  if (discretization.has_state(nds_pressure, "phinp_fluid"))
  {
    variable_manager()->setup_porofluid_managers(element, discretization, number_of_fluid_phases,
        total_number_of_porofluid_dofs, variable_manager()->porofluid_material()->num_vol_frac());
    variable_manager()->extract_element_and_node_values_of_porofluid(
        element, discretization, location_array, scatra_ele_calc::xyze_);
    enable_L2_projection_ = parameters.get<bool>("L2-projection");
    // extract the nodal flux
    if (enable_L2_projection_)
    {
      extract_nodal_flux(discretization, location_array, number_of_fluid_phases);
    }
  }
  else
  {
    FOUR_C_THROW("Something went wrong here, scatra-dis does not have fluid primary variable");
  }

  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes
  // (time n+alpha_F for generalized-alpha scheme, at time n+1 otherwise)
  // ---------------------------------------------------------------------
  scatra_ele_calc::body_force(element);
  //--------------------------------------------------------------------------------
  // further node-based source terms not given via Neumann volume condition
  // i.e., set special body force for homogeneous isotropic turbulence
  //--------------------------------------------------------------------------------
  scatra_ele_calc::other_node_based_source_terms(global_dof_ids, discretization, parameters);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::extract_nodal_flux(const Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& location_array, const int number_of_fluid_phases)
{
  // resize state vectors based on number of phases
  element_flux_np_.resize(number_of_fluid_phases);

  // get number of dofset associated with velocity related dofs
  const int nds_velocity = scatra_ele_calc::scatrapara_->nds_vel();

  const std::string state_prefix = "flux";
  for (int phase_id = 0; phase_id < number_of_fluid_phases; phase_id++)
  {
    std::stringstream state_name;
    state_name << state_prefix << phase_id;

    // get convective (velocity - mesh displacement) velocity at nodes
    std::shared_ptr<const Core::LinAlg::Vector<double>> nodal_convective_velocity =
        discretization.get_state(nds_velocity, state_name.str());
    if (nodal_convective_velocity == nullptr)
      FOUR_C_THROW("Cannot get state vector {}", state_name.str());

    // extract local values of convective velocity field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(
        *nodal_convective_velocity, element_flux_np_[phase_id], location_array[nds_velocity].lm_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
double
Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::compute_pore_pressure()
{
  return variable_manager()->get_solid_pressure();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::materials(
    const std::shared_ptr<const Core::Mat::Material> material, const int scalar_id, double& densn,
    double& densnp, double& densam, double& viscosity, const int gauss_point_id)
{
  switch (material->material_type())
  {
    case Core::Materials::m_scatra_in_fluid_porofluid_pressure_based:
    {
      material_scatra_in_fluid(material, scalar_id, densn, densnp, densam, gauss_point_id);
      break;
    }
    case Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based:
    {
      material_scatra_in_volfrac(material, scalar_id, densn, densnp, densam, gauss_point_id);
      break;
    }
    case Core::Materials::m_scatra_in_solid_porofluid_pressure_based:
    {
      material_scatra_in_solid(material, scalar_id, densn, densnp, densam, gauss_point_id);
      break;
    }
    case Core::Materials::m_scatra_as_temperature_porofluid_pressure_based:
    {
      material_scatra_as_temperature(material, scalar_id, densn, densnp, densam, gauss_point_id);
      break;
    }
    default:
      FOUR_C_THROW("Material type {} is not supported for multiphase flow through porous media!",
          material->material_type());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    material_scatra_in_fluid(const std::shared_ptr<const Core::Mat::Material> material,
        const int scalar_id, double& densn, double& densnp, double& densam,
        const int gauss_point_id)
{
  if (gauss_point_id == -1)
    FOUR_C_THROW(
        "No gauss point given for evaluation of MatMultiPoro material. Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& scatra_material =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(material);

  // volume fraction of fluid phase: volfrac_fluid = porosity * saturation_fluid
  double volfrac_fluid = 0.0;
  // d_eff = d_0 * (porosity * saturation(scalar_id))^delta
  double d_eff = 0.0;

  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    volfrac_fluid = variable_manager()->porofluid_phase_manager()->porosity() *
                    variable_manager()->get_saturation(scalar_id);
    d_eff = std::pow(variable_manager()->porofluid_phase_manager()->porosity() *
                         variable_manager()->get_saturation(scalar_id),
        scatra_material->delta());
  }

  {
    // set diffusivity (scaled with volfrac_fluid)
    poro::set_diffusivity(scatra_material, scalar_id, volfrac_fluid * d_eff);

    // set densities (scaled with volfrac_fluid)
    poro::set_densities(volfrac_fluid, densn, densnp, densam);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    material_scatra_in_volfrac(const std::shared_ptr<const Core::Mat::Material> material,
        const int scalar_id, double& densn, double& densnp, double& densam,
        const int gauss_point_id)
{
  if (gauss_point_id == -1)
    FOUR_C_THROW(
        "No gauss point given for evaluation of MatMultiPoro material. Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& scatra_material =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(material);

  // volume fraction
  double volfrac = 0.0;
  // d_eff = d_0 * (porosity * saturation(scalar_id))^delta
  double d_eff = 0.0;

  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    volfrac = variable_manager()->get_volume_fraction(scalar_id);
    d_eff = std::pow(variable_manager()->get_volume_fraction(scalar_id), scatra_material->delta());
  }

  {
    // set diffusivity (scaled with volfrac)
    poro::set_diffusivity(scatra_material, scalar_id, volfrac * d_eff);

    // set densities (scaled with volfrac)
    poro::set_densities(volfrac, densn, densnp, densam);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    material_scatra_in_solid(const std::shared_ptr<const Core::Mat::Material> material,
        const int scalar_id, double& densn, double& densnp, double& densam,
        const int gauss_point_id)
{
  if (gauss_point_id == -1)
    FOUR_C_THROW(
        "No gauss point given for evaluation of MatMultiPoro material. Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroSolid>& scatra_material =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroSolid>(material);

  // volume fraction of solid phase: volfrac_solid_phase = (1 - porosity - sumaddvolfrac)
  double volfrac_solid_phase = 0.0;
  // d_eff = d_0 * (porosity * saturation(scalar_id))^delta
  double d_eff = 0.0;

  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    volfrac_solid_phase = (1 - variable_manager()->porofluid_phase_manager()->porosity() -
                           variable_manager()->porofluid_phase_manager()->sum_add_vol_frac());
    d_eff = std::pow((1 - variable_manager()->porofluid_phase_manager()->porosity() -
                         variable_manager()->porofluid_phase_manager()->sum_add_vol_frac()),
        scatra_material->delta());
  }

  {
    // set diffusivity (scaled with volfrac_solid_phase)
    poro::set_diffusivity(scatra_material, scalar_id, volfrac_solid_phase * d_eff);

    // set densities (scaled with volfrac_solid_phase)
    poro::set_densities(volfrac_solid_phase, densn, densnp, densam);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    material_scatra_as_temperature(const std::shared_ptr<const Core::Mat::Material> material,
        const int scalar_id, double& densn, double& densnp, double& densam,
        const int gauss_point_id)
{
  if (gauss_point_id == -1)
    FOUR_C_THROW(
        "No gauss point given for evaluation of MatMultiPoro material. Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroTemperature>& scatra_material =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroTemperature>(material);

  double effective_heat_capacity = 0.0;
  double effective_thermal_conductivity = 0.0;

  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    const int number_of_fluid_phases =
        variable_manager()->porofluid_phase_manager()->num_fluid_phases();
    const int number_of_volume_fractions =
        variable_manager()->porofluid_phase_manager()->num_vol_frac();

    effective_heat_capacity =
        (1 - variable_manager()->porofluid_phase_manager()->porosity() -
            variable_manager()->porofluid_phase_manager()->sum_add_vol_frac()) *
        variable_manager()->porofluid_phase_manager()->solid_density() *
        scatra_material->cp_solid();

    // kappa_eff = kappa_s*poro_s + kappa_fluids*poro*saturation_fluids + kappa_volfrac*poro_volfrac
    effective_thermal_conductivity =
        (1 - variable_manager()->porofluid_phase_manager()->porosity() -
            variable_manager()->porofluid_phase_manager()->sum_add_vol_frac()) *
        scatra_material->kappa_solid();

    for (int phase = 0; phase < number_of_fluid_phases; ++phase)
    {
      effective_heat_capacity += scatra_material->cp_fluid(phase) *
                                 variable_manager()->porofluid_phase_manager()->porosity() *
                                 variable_manager()->porofluid_phase_manager()->saturation(phase) *
                                 variable_manager()->porofluid_phase_manager()->density(phase);

      effective_thermal_conductivity +=
          scatra_material->kappa_fluid(phase) *
          variable_manager()->porofluid_phase_manager()->porosity() *
          variable_manager()->porofluid_phase_manager()->saturation(phase);
    }

    for (int phase = 0; phase < number_of_volume_fractions; ++phase)
    {
      effective_heat_capacity +=
          scatra_material->cp_volfrac(phase) *
          variable_manager()->porofluid_phase_manager()->vol_frac(phase) *
          variable_manager()->porofluid_phase_manager()->vol_frac_density(phase);

      effective_thermal_conductivity +=
          scatra_material->kappa_volfrac(phase) *
          variable_manager()->porofluid_phase_manager()->vol_frac(phase);
    }
  }

  {
    variable_manager()->set_effective_heat_capacity(effective_heat_capacity);

    // set diffusivity
    poro::set_diffusivity(scatra_material, scalar_id, effective_thermal_conductivity);

    // set densities
    poro::set_densities(effective_heat_capacity, densn, densnp, densam);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::set_internal_variables_for_mat_and_rhs()
{
  variable_manager()->set_internal_variables_porofluid_pressure_based(scatra_ele_calc::funct_,
      scatra_ele_calc::derxy_, scatra_ele_calc::deriv_, scatra_ele_calc::xjm_,
      poro_reaction::xyze0_, scatra_ele_calc::ephinp_, scatra_ele_calc::ephin_,
      scatra_ele_calc::ehist_, scatra_ele_calc::eforcevelocity_);

  if (enable_L2_projection_)
  {
    variable_manager()->adapt_convective_term_for_l2(
        scatra_ele_calc::funct_, scatra_ele_calc::derxy_, element_flux_np_);
  }
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::set_advanced_reaction_terms(const int scalar_id,
    const std::shared_ptr<Mat::MatListReactions> material_reaction_list,
    const double* gauss_point_coordinates)
{
  const std::shared_ptr<ScaTraEleReaManagerAdvReac> reaction_manager =
      advanced_reaction::rea_manager();

  fill_coupling_vector_and_add_variables(*material_reaction_list, *reaction_manager);

  const ScaTra::Action action = variable_manager()->get_action();

  auto time = scatra_ele_calc::scatraparatimint_->time();
  // note: we always need the reaction term to calculate rhsint, which is needed also for OD-terms
  reaction_manager->add_to_rea_body_force(
      material_reaction_list->calc_rea_body_force_term(scalar_id,
          scatra_ele_calc::scatravarmanager_->phinp(), coupling_values_, gauss_point_coordinates,
          time),
      scalar_id);

  std::vector<std::pair<std::string, double>> empty_constants;

  switch (action)
  {
    case ScaTra::Action::calc_mat_and_rhs:
    case ScaTra::Action::calc_initial_time_deriv:
    {
      material_reaction_list->calc_rea_body_force_deriv_matrix(scalar_id,
          reaction_manager->get_rea_body_force_deriv_vector(scalar_id),
          scatra_ele_calc::scatravarmanager_->phinp(), coupling_values_, gauss_point_coordinates,
          time);

      break;
    }
    case ScaTra::Action::calc_scatra_mono_odblock_fluid:
    {
      material_reaction_list->calc_rea_body_force_deriv_matrix_add_variables(scalar_id,
          reaction_manager->get_rea_body_force_deriv_vector_add_variables(scalar_id),
          scatra_ele_calc::scatravarmanager_->phinp(), coupling_values_, empty_constants,
          gauss_point_coordinates, time);

      break;
    }
    case ScaTra::Action::calc_scatra_mono_odblock_mesh:
    {
      if (variable_manager()->porofluid_phase_manager()->porosity_depends_on_struct())
      {
        material_reaction_list->calc_rea_body_force_deriv_matrix_add_variables(scalar_id,
            reaction_manager->get_rea_body_force_deriv_vector_add_variables(scalar_id),
            scatra_ele_calc::scatravarmanager_->phinp(), coupling_values_, empty_constants,
            gauss_point_coordinates, time);
      }
      break;
    }
    default:
      FOUR_C_THROW("Wrong action type in variable_manager(), action type is {}", action);
  }
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::get_rhs_int(
    double& rhsint, const double densnp, const int scalar_id)
{
  // only difference is the inverse scaling with density
  advanced_reaction::get_rhs_int(
      rhsint, 1.0 / variable_manager()->get_density(scalar_id), scalar_id);
}

/*------------------------------------------------------------------------------ *
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::calc_mat_react(
    Core::LinAlg::SerialDenseMatrix& element_matrix, const int scalar_id, const double timefacfac,
    const double timetaufac, const double taufac, const double densnp,
    const Core::LinAlg::Matrix<nen_, 1>& sgconv, const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  // only difference is the inverse scaling with density
  advanced_reaction::calc_mat_react(element_matrix, scalar_id, timefacfac, timetaufac, taufac,
      1.0 / variable_manager()->get_density(scalar_id), sgconv, diff);
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    fill_coupling_vector_and_add_variables(Mat::MatListReactions& material_list_reactions,
        ScaTraEleReaManagerAdvReac& reaction_manager)
{
  // if it is empty rebuilt it
  if (coupling_values_.empty())
  {
    // pressures
    const std::vector<double>& pressures = variable_manager()->get_pressure();
    const int number_of_fluid_phases = variable_manager()->porofluid_material()->num_fluid_phases();
    for (int phase_id = 0; phase_id < number_of_fluid_phases; ++phase_id)
    {
      coupling_values_.emplace_back("p" + std::to_string(phase_id + 1), pressures[phase_id]);
    }

    // saturation
    const std::vector<double>& saturations = variable_manager()->get_saturation();
    for (int phase_id = 0; phase_id < number_of_fluid_phases; ++phase_id)
    {
      coupling_values_.emplace_back("S" + std::to_string(phase_id + 1), saturations[phase_id]);
    }
    // porosity
    coupling_values_.push_back(std::pair<std::string, double>(
        "porosity", variable_manager()->porofluid_phase_manager()->porosity()));

    // additional volume fractions
    const std::vector<double>& volume_fractions = variable_manager()->get_volume_fraction();
    const int number_of_volume_fractions =
        variable_manager()->porofluid_phase_manager()->num_vol_frac();
    for (int phase_id = 0; phase_id < number_of_volume_fractions; ++phase_id)
    {
      coupling_values_.emplace_back(
          "VF" + std::to_string(phase_id + 1), volume_fractions[phase_id]);
    }

    // additional volume fraction pressures
    const std::vector<double>& volume_fraction_pressures =
        variable_manager()->get_volume_fractions_pressure();
    for (int phase_id = 0; phase_id < number_of_volume_fractions; phase_id++)
    {
      coupling_values_.emplace_back(
          "VFP" + std::to_string(phase_id + 1), volume_fraction_pressures[phase_id]);
    }

    // initialize and add the variables to the reaction manager --> has to be done only once
    reaction_manager.initialize_rea_body_force_deriv_vector_add_variables(
        scatra_ele_calc::numdofpernode_, coupling_values_.size());
    // error will be thrown if reaction != by-reaction coupling is chosen
    for (int dof_id = 0; dof_id < scatra_ele_calc::numdofpernode_; dof_id++)
      material_list_reactions.add_additional_variables(dof_id, coupling_values_);
  }
  // directly copy values (rely on order for performance reasons)
  else
  {
    // pressures
    const std::vector<double>& pressures = variable_manager()->get_pressure();
    const int number_of_fluid_phases = variable_manager()->porofluid_material()->num_fluid_phases();
    for (int phase_id = 0; phase_id < number_of_fluid_phases; phase_id++)
    {
      coupling_values_[phase_id].second = pressures[phase_id];
    }
    // saturation
    const std::vector<double>& saturations = variable_manager()->get_saturation();
    for (int phase_id = 0; phase_id < number_of_fluid_phases; phase_id++)
    {
      coupling_values_[number_of_fluid_phases + phase_id].second = saturations[phase_id];
    }
    // porosity
    coupling_values_[2 * number_of_fluid_phases].second =
        variable_manager()->porofluid_phase_manager()->porosity();

    // additional volume fractions
    const std::vector<double>& volume_fractions = variable_manager()->get_volume_fraction();
    const std::vector<double>& volume_fraction_pressures =
        variable_manager()->get_volume_fractions_pressure();
    const int number_of_volume_fractions =
        variable_manager()->porofluid_phase_manager()->num_vol_frac();
    for (int i = 0; i < number_of_volume_fractions; i++)
    {
      coupling_values_[2 * number_of_fluid_phases + 1 + i].second = volume_fractions[i];
      coupling_values_[2 * number_of_fluid_phases + number_of_volume_fractions + 1 + i].second =
          volume_fraction_pressures[i];
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::calc_mat_conv(
    Core::LinAlg::SerialDenseMatrix& element_matrix, const int scalar_id, const double timefacfac,
    const double densnp, const Core::LinAlg::Matrix<nen_, 1>& sgconv)
{
  // case of zero saturation/volfrac
  // no convective term for species in solid
  if (variable_manager()->evaluate_scalar(scalar_id) &&
      variable_manager()->get_species_type(scalar_id) !=
          Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
  {
    // the only difference to the base class version is that there is no scaling with the density
    poro_reaction::calc_mat_conv(element_matrix, scalar_id, timefacfac, 1.0, sgconv);
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::calc_mat_mass(
    Core::LinAlg::SerialDenseMatrix& element_matrix, const int& scalar_id, const double& fac,
    const double& densam)
{
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    // the only difference to the base class version is that there is no scaling with the density
    poro_reaction::calc_mat_mass(element_matrix, scalar_id, fac, densam);
  }
  else
  {
    if (variable_manager()->get_species_type(scalar_id) ==
        Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
    {
      // If we have zero "densities" (porosity * saturation(scalar_id)), which mostly happens for
      // tumor cells, the whole equation will be equal to zero since it is scaled with the density.
      // In that case, the mass fraction of the species (necrotic tumor cells) also has to be zero.
      // --> Here, we explicitly force it to be zero through a "Dirichlet" boundary condition.
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
        element_matrix(fvi, fvi) += penalty_;
      }
    }
  }
}


/*------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::calc_mat_conv_add_cons(Core::LinAlg::SerialDenseMatrix& element_matrix,
    const int scalar_id, const double timefacfac, const double vdiv, const double densnp)
{
  // the only difference to the base class version is that there is no scaling with the density
  poro_reaction::calc_mat_conv_add_cons(element_matrix, scalar_id, timefacfac, vdiv, 1.0);
}

/*------------------------------------------------------------------- *
 *--------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::recompute_conv_phi_for_rhs(const int scalar_id,
    const Core::LinAlg::Matrix<nsd_, 1>& sgvelint, const double densnp, const double densn,
    const double vdiv)
{
  // the only difference to the base class version is that there is no scaling with the density
  poro_reaction::recompute_conv_phi_for_rhs(scalar_id, sgvelint, 1.0, 1.0, vdiv);
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::calc_conv_od_mesh(
    Core::LinAlg::SerialDenseMatrix& element_matrix, const int scalar_id,
    const int number_of_dofs_per_node_mesh, const double fac, const double rhsfac,
    const double densnp, const double J, const Core::LinAlg::Matrix<nsd_, 1>& gradphi,
    const Core::LinAlg::Matrix<nsd_, 1>& convelint)
{
  // case of zero saturation/volfrac
  // no convective term for species in solid
  if (variable_manager()->evaluate_scalar(scalar_id) &&
      variable_manager()->get_species_type(scalar_id) !=
          Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
  {
    static Core::LinAlg::Matrix<nsd_, nsd_> diffusion_tensor;
    static Core::LinAlg::Matrix<nsd_, 1> pressure_gradient_reference;

    // linearization of mesh motion
    // dJ/dd = dJ/dF : dF/dd = J * F^-T . N_{\psi} = J * N_x
    // J denotes the determinant of the Jacobian of the mapping between current and parameter space,
    // i.e., det(dx/ds) in our case: rhsfac = J * dt * theta --> d(rhsfac)/dd = rhsfac * N_x
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
      const double v =
          rhsfac * scatra_ele_calc::funct_(vi) * (-1.0) * variable_manager()->conv_phi(scalar_id);

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        for (unsigned idim = 0; idim < nsd_; ++idim)
        {
          const int fui = ui * nsd_ + idim;
          element_matrix(fvi, fui) += v * scatra_ele_calc::derxy_(idim, ui);
        }
      }
    }

    if (variable_manager()->get_species_type(scalar_id) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid ||
        variable_manager()->get_species_type(scalar_id) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac)
    {
      variable_manager()->get_diffusion_tensor_fluid(
          scalar_id, diffusion_tensor, variable_manager()->get_phase_id(scalar_id));
      variable_manager()->get_pressure_gradient_reference(scatra_ele_calc::xjm_,
          pressure_gradient_reference, variable_manager()->get_phase_id(scalar_id));
      const double vrhs = rhsfac * 1.0 / J * diffusion_tensor(0, 0) * (-1.0);

      // linearization of pressure gradient
      // standard Galerkin terms  -- "shapederivatives" pressure gradient
      apply_shape_derivs_pressure_grad(
          element_matrix, scalar_id, vrhs, gradphi, pressure_gradient_reference);

      // linearization of gradphi
      // standard Galerkin terms  -- "shapederivatives" gradphi
      poro_reaction::apply_shape_derivs_conv(
          element_matrix, scalar_id, rhsfac, 1.0, J, gradphi, convelint);
    }

    else if (variable_manager()->get_species_type(scalar_id) ==
             Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature)
    {
      const int number_of_fluid_phases =
          variable_manager()->porofluid_phase_manager()->num_fluid_phases();
      const int number_of_volume_fractions =
          variable_manager()->porofluid_phase_manager()->num_vol_frac();

      for (int phase = 0; phase < number_of_fluid_phases; ++phase)
      {
        variable_manager()->get_diffusion_tensor_fluid(scalar_id, diffusion_tensor, phase);
        variable_manager()->get_pressure_gradient_reference(
            scatra_ele_calc::xjm_, pressure_gradient_reference, phase);

        const double vrhs = rhsfac * 1.0 / J * diffusion_tensor(0, 0) * (-1.0) *
                            variable_manager()->porofluid_phase_manager()->density(phase) *
                            variable_manager()->get_heat_capacity(phase);

        // linearization of pressure gradient
        // standard Galerkin terms  -- "shapederivatives" pressure gradient
        apply_shape_derivs_pressure_grad(
            element_matrix, scalar_id, vrhs, gradphi, pressure_gradient_reference);
      }

      for (int phase_id = number_of_fluid_phases;
          phase_id < number_of_fluid_phases + number_of_volume_fractions; ++phase_id)
      {
        variable_manager()->get_diffusion_tensor_fluid(scalar_id, diffusion_tensor, phase_id);
        variable_manager()->get_pressure_gradient_reference(
            scatra_ele_calc::xjm_, pressure_gradient_reference, phase_id);

        const double vrhs = rhsfac * 1.0 / J * diffusion_tensor(0, 0) * (-1.0) *
                            variable_manager()->porofluid_phase_manager()->vol_frac_density(
                                phase_id - number_of_fluid_phases) *
                            variable_manager()->get_heat_capacity(phase_id);

        // linearization of pressure gradient
        // standard Galerkin terms  -- "shapederivatives" pressure gradient
        apply_shape_derivs_pressure_grad(
            element_matrix, scalar_id, vrhs, gradphi, pressure_gradient_reference);
      }

      // linearization of gradphi
      // standard Galerkin terms  -- "shapederivatives" gradphi
      poro_reaction::apply_shape_derivs_conv(
          element_matrix, scalar_id, rhsfac, 1.0, J, gradphi, convelint);
    }
    else
    {
      FOUR_C_THROW("Species type no valid!");
    }
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::calc_lin_mass_od_mesh(Core::LinAlg::SerialDenseMatrix& element_matrix,
    const int scalar_id, const int number_of_dofs_per_node_mesh, const double rhsfac,
    const double fac, const double densam, const double densnp, const double phinp,
    const double hist, const double J, const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    // get pre-factor for this scalar
    const double pre_factor =
        variable_manager()->get_pre_factor_mass_matrix_od_mesh(scalar_id, fac);

    scatra_ele_calc::calc_lin_mass_od_mesh(element_matrix, scalar_id, number_of_dofs_per_node_mesh,
        rhsfac, pre_factor, densam, densnp, phinp, hist, J, dJ_dmesh);
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    calc_hist_and_source_od_mesh(Core::LinAlg::SerialDenseMatrix& element_matrix,
        const int scalar_id, const int number_of_dofs_per_node_mesh, const double fac,
        const double rhsint, const double J, const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh,
        const double densnp)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    const int number_of_fluid_phases =
        variable_manager()->porofluid_phase_manager()->num_fluid_phases();

    // get pre-factor for this scalar
    const double pre_factor = variable_manager()->get_pre_factor_hist_and_source_od_mesh(
        scalar_id, fac, densnp, scatra_ele_calc::scatravarmanager_->hist(scalar_id), rhsint);

    // linearization of mesh motion: call base class with correct pre-factor
    scatra_ele_calc::calc_hist_and_source_od_mesh(element_matrix, scalar_id,
        number_of_dofs_per_node_mesh, pre_factor, 1.0, J, dJ_dmesh, densnp);

    // linearization of advanced reaction terms
    const std::shared_ptr<ScaTraEleReaManagerAdvReac> reaction_manager =
        advanced_reaction::rea_manager();
    if (reaction_manager->active() &&
        variable_manager()->porofluid_phase_manager()->porosity_depends_on_struct())
    {
      const std::vector<double> reaction_derivatives =
          reaction_manager->get_rea_body_force_deriv_vector_add_variables(scalar_id);

      // porosity deriv at [2* number_of_fluid_phases]:
      // d reac / d d = d reac / d poro * d poro / d d
      // with
      // dporo/dd = dporo/dJ * dJ/dd = dporosity/dJ * J * N_x
      // J denotes the determinant of the deformation gradient, i.e.
      // det F = det ( d x / d X ) = det(dx/ds) * ( det(dX/ds) )^-1

      const double porosity_derivative =
          reaction_derivatives[2 * number_of_fluid_phases] *
          variable_manager()->porofluid_phase_manager()->jacobian_def_grad() *
          variable_manager()->porofluid_phase_manager()->porosity_deriv_wrt_jacobian_def_grad();

      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
        // TODO: gen-alpha
        const double v = scatra_ele_calc::funct_(vi) * porosity_derivative *
                         scatra_ele_calc::scatraparatimint_->time_fac() * fac * (-1.0) /
                         variable_manager()->get_density(scalar_id);

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)
          {
            const int fui = ui * nsd_ + idim;

            element_matrix(fvi, fui) += v * scatra_ele_calc::derxy_(idim, ui);
          }
        }
      }
    }
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::calc_diff_od_mesh(
    Core::LinAlg::SerialDenseMatrix& element_matrix, const int scalar_id,
    const int number_of_dofs_per_node_mesh, const double diffcoeff, const double fac,
    const double rhsfac, const double J, const Core::LinAlg::Matrix<nsd_, 1>& gradphi,
    const Core::LinAlg::Matrix<nsd_, 1>& convelint,
    const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    // call base class
    scatra_ele_calc::calc_diff_od_mesh(element_matrix, scalar_id, number_of_dofs_per_node_mesh,
        diffcoeff, fac, rhsfac, J, gradphi, convelint, dJ_dmesh);

    // get pre-factor for this scalar
    const double pre_factor =
        variable_manager()->get_pre_factor_diffusion_od_mesh(scalar_id, rhsfac, diffcoeff);

    if (fabs(pre_factor) > 1.0e-12)
    {
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;

        double laplacian(0.0);
        scatra_ele_calc::get_laplacian_weak_form_rhs(laplacian, gradphi, vi);
        const double v = pre_factor * laplacian;
        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)
          {
            const int fui = ui * nsd_ + idim;

            element_matrix(fvi, fui) += v * scatra_ele_calc::derxy_(idim, ui);
          }
        }
      }
    }
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::calc_react_od_mesh(Core::LinAlg::SerialDenseMatrix& element_matrix,
    const int scalar_id, const int number_of_dofs_per_node_mesh, const double rhsfac,
    const double rea_phi, const double J, const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    // get pre-factor for this scalar
    const double pre_factor =
        variable_manager()->get_pre_factor_mass_matrix_od_mesh(scalar_id, rhsfac);

    scatra_ele_calc::calc_react_od_mesh(
        element_matrix, scalar_id, number_of_dofs_per_node_mesh, pre_factor, rea_phi, J, dJ_dmesh);
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::calc_mat_conv_od_fluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
    const int scalar_id, const int number_of_dofs_per_node_mesh, const double rhsfac,
    const double densnp, const Core::LinAlg::Matrix<nsd_, 1>& gradphi)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    if (variable_manager()->get_species_type(scalar_id) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid ||
        variable_manager()->get_species_type(scalar_id) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac)
    {
      const int total_number_of_porofluid_dofs =
          variable_manager()->porofluid_material()->num_mat();

      static Core::LinAlg::Matrix<nsd_, nsd_> diffusion_tensor(Core::LinAlg::Initialization::zero);
      variable_manager()->get_diffusion_tensor_fluid(
          scalar_id, diffusion_tensor, variable_manager()->get_phase_id(scalar_id));

      // gradphi^T * diffusion_tensor
      // TODO: not sure if this works for anisotropic fluid diffusion tensor
      static Core::LinAlg::Matrix<1, nsd_> gradient_phi_transpose_diffusion_tensor;
      gradient_phi_transpose_diffusion_tensor.multiply_tn(gradphi, diffusion_tensor);

      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
        const double v = rhsfac * scatra_ele_calc::funct_(vi) * (-1.0);

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          // get pre-factor vector for this scalar
          static std::vector pre_factor_convection_OD_porofluid(
              total_number_of_porofluid_dofs, 0.0);
          variable_manager()->get_pre_factor_convection_od_porofluid(ui,
              &pre_factor_convection_OD_porofluid, gradphi, gradient_phi_transpose_diffusion_tensor,
              scatra_ele_calc::funct_, scatra_ele_calc::derxy_,
              variable_manager()->get_phase_id(scalar_id));

          for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
          {
            const int fui = ui * total_number_of_porofluid_dofs + dof_id;
            element_matrix(fvi, fui) += v * pre_factor_convection_OD_porofluid[dof_id];
          }
        }
      }

      // TODO: linearization of dynamic viscosity w.r.t. dof, necessary? FD check does not fail
    }

    else if (variable_manager()->get_species_type(scalar_id) ==
             Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature)
    {
      const int number_of_fluid_phases =
          variable_manager()->porofluid_phase_manager()->num_fluid_phases();
      const int number_of_volume_fractions =
          variable_manager()->porofluid_phase_manager()->num_vol_frac();

      for (int phase = 0; phase < number_of_fluid_phases + number_of_volume_fractions; ++phase)
      {
        const int total_number_of_porofluid_dofs =
            variable_manager()->porofluid_material()->num_mat();

        static Core::LinAlg::Matrix<nsd_, nsd_> diffusion_tensor;
        variable_manager()->get_diffusion_tensor_fluid(scalar_id, diffusion_tensor, phase);

        // gradphi^T * diffusion_tensor
        static Core::LinAlg::Matrix<1, nsd_> gradient_phi_transpose_diffusion_tensor;
        gradient_phi_transpose_diffusion_tensor.multiply_tn(gradphi, diffusion_tensor);

        // calculate density * heat capacity
        double density_heat_capacity =
            variable_manager()->get_heat_capacity(phase) * variable_manager()->get_density()[phase];

        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
          const double v = rhsfac * scatra_ele_calc::funct_(vi) * (-1.0) * density_heat_capacity;

          for (unsigned ui = 0; ui < nen_; ++ui)
          {
            // get pre-fac vector for this scalar
            static std::vector pre_factor_convection_OD_porofluid(
                total_number_of_porofluid_dofs, 0.0);
            variable_manager()->get_pre_factor_convection_od_porofluid(ui,
                &pre_factor_convection_OD_porofluid, gradphi,
                gradient_phi_transpose_diffusion_tensor, scatra_ele_calc::funct_,
                scatra_ele_calc::derxy_, phase);

            for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
            {
              const int fui = ui * total_number_of_porofluid_dofs + dof_id;
              element_matrix(fvi, fui) += v * pre_factor_convection_OD_porofluid[dof_id];
            }
          }
        }
      }
    }
  }
}

/*-------------------------------------------------------------------------- *
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    calc_mat_conv_add_cons_od_fluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
        const int scalar_id, const int number_of_dofs_per_node_mesh, const double timefacfac,
        const double densnp, const double phinp)
{
  FOUR_C_THROW(
      "calc_mat_conv_add_cons_od_fluid not yet available for "
      "scatra_ele_calc_porofluid_pressure_based.");
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::calc_lin_mass_od_fluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
    const int scalar_id, const int number_of_dofs_per_node_mesh, const double rhsfac,
    const double fac, const double densam, const double densnp, const double phinp,
    const double hist)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    if (scatra_ele_calc::scatraparatimint_->is_gen_alpha())
      FOUR_C_THROW("GenAlpha is not implemented for scatra_ele_calc_porofluid_pressure_based.");

    double vtrans = 0.0;
    vtrans = fac * densnp * phinp;
    const int total_number_of_porofluid_dofs = variable_manager()->porofluid_material()->num_mat();

    // get pre-fac vector for this scalar
    std::vector pre_factor_mass_matrix_OD_porofluid(total_number_of_porofluid_dofs, 0.0);
    variable_manager()->get_pre_factor_mass_matrix_od_porofluid(
        scalar_id, &pre_factor_mass_matrix_OD_porofluid);

    calc_mass_matrix_type_od_porofluid(element_matrix, scalar_id,
        &pre_factor_mass_matrix_OD_porofluid, total_number_of_porofluid_dofs, vtrans);
  }
}

/*---------------------------------------------------------------------- *
 *-----------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    calc_mass_matrix_type_od_porofluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
        const int scalar_id, const std::vector<double>* per_factor_mass_matrix_OD_porofluid,
        const int total_number_of_porofluid_dofs, double pre_factor)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = pre_factor * scatra_ele_calc::funct_(vi);
    const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const double vfunct = v * scatra_ele_calc::funct_(ui);
      for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
      {
        const int fui = ui * total_number_of_porofluid_dofs + dof_id;

        element_matrix(fvi, fui) += vfunct * (*per_factor_mass_matrix_OD_porofluid)[dof_id];
      }
    }
  }
}

/*----------------------------------------------------------------------------- *
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    calc_hist_and_source_od_fluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
        const int scalar_id, const int number_of_dofs_per_node_mesh, const double fac,
        const double rhsint, const double densnp)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    const int number_of_fluid_phases =
        variable_manager()->porofluid_phase_manager()->num_fluid_phases();
    const int number_of_volume_fractions =
        variable_manager()->porofluid_phase_manager()->num_vol_frac();
    // total number of porofluid dofs additionally includes volume fraction pressures
    const int total_number_of_porofluid_dofs = variable_manager()->porofluid_material()->num_mat();

    // linearization of history: pre-factor is densnp = porosity * rho * S
    // --> porosity and saturation have to be linearized
    const double vrhs = -1.0 * fac * scatra_ele_calc::scatravarmanager_->hist(scalar_id) * densnp;

    // get pre-factor vector for this scalar
    static std::vector pre_factor_mass_matrix_OD_porofluid(total_number_of_porofluid_dofs, 0.0);
    variable_manager()->get_pre_factor_mass_matrix_od_porofluid(
        scalar_id, &pre_factor_mass_matrix_OD_porofluid);

    calc_mass_matrix_type_od_porofluid(element_matrix, scalar_id,
        &pre_factor_mass_matrix_OD_porofluid, total_number_of_porofluid_dofs, vrhs);

    // linearization of advanced reaction terms
    const std::shared_ptr<ScaTraEleReaManagerAdvReac> reaction_manager =
        advanced_reaction::rea_manager();
    if (reaction_manager->active())
    {
      const std::vector<double> reaction_derivatives =
          reaction_manager->get_rea_body_force_deriv_vector_add_variables(scalar_id);

      // derivatives after primary variables of fluid
      std::vector phi_derivatives(total_number_of_porofluid_dofs, 0.0);

      for (int phase_id = 0; phase_id < number_of_fluid_phases; phase_id++)
      {
        // porosity deriv at 2 * number_of_fluid_phases:
        // d reac / d phi_i = (d reac / d poro) * (d poro / d phi_i)
        phi_derivatives[phase_id] +=
            reaction_derivatives[2 * number_of_fluid_phases] *
            variable_manager()->porofluid_phase_manager()->porosity_deriv(phase_id);
        for (int j = 0; j < number_of_fluid_phases; j++)
        {
          // pressure derivs at [0 ... number_of_fluid_phases]:
          // d reac / d phi_i = (d reac / d pressure_j) * (d pressure_j / d phi_i)
          // saturation derivs at [number_of_fluid_phases ... (2*number_of_fluid_phases-1)]:
          // d reac / d phi_i = (d reac / d sat_j) * (d sat_j / d phi_i)
          phi_derivatives[phase_id] +=
              reaction_derivatives[j + number_of_fluid_phases] *
                  variable_manager()->porofluid_phase_manager()->saturation_deriv(j, phase_id) +
              reaction_derivatives[j] *
                  variable_manager()->porofluid_phase_manager()->pressure_deriv(j, phase_id);
        }
      }

      for (int volume_fraction_id = 0; volume_fraction_id < number_of_volume_fractions;
          volume_fraction_id++)
      {
        if (total_number_of_porofluid_dofs ==
            number_of_fluid_phases +
                2 * number_of_volume_fractions)  // volfrac with closing relation
                                                 // "homogenized_vasculature_tumor"
        {
          // derivatives after volume fractions at [2*number_of_fluid_phases + 1 +
          // volume_fraction_id]
          phi_derivatives[volume_fraction_id + number_of_fluid_phases] +=
              reaction_derivatives[2 * number_of_fluid_phases + 1 + volume_fraction_id] +
              reaction_derivatives[2 * number_of_fluid_phases] *
                  variable_manager()->porofluid_phase_manager()->porosity_deriv(
                      volume_fraction_id + number_of_fluid_phases);
          // derivatives after volume fraction pressures at
          // [2*number_of_fluid_phases + number_of_volume_fractions+ 1 + volume_fraction_id]
          phi_derivatives[volume_fraction_id + number_of_fluid_phases +
                          number_of_volume_fractions] +=
              reaction_derivatives[2 * number_of_fluid_phases + number_of_volume_fractions + 1 +
                                   volume_fraction_id];
        }
        else  // volfrac with closing relation "blood lung"
        {
          // derivatives after volume fraction pressures at
          // [2*number_of_fluid_phases + 1 + volume_fraction_id]
          phi_derivatives[volume_fraction_id + number_of_fluid_phases] +=
              reaction_derivatives[2 * number_of_fluid_phases + 1 + volume_fraction_id];
        }
      }

      // fill matrix
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
        // TODO: gen-alpha?
        const double v = scatra_ele_calc::funct_(vi) *
                         scatra_ele_calc::scatraparatimint_->time_fac() * fac * (-1.0) /
                         variable_manager()->get_density(scalar_id);

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const double vfunct = v * scatra_ele_calc::funct_(ui);

          for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
          {
            const int fui = ui * total_number_of_porofluid_dofs + dof_id;

            element_matrix(fvi, fui) += vfunct * phi_derivatives[dof_id];
          }
        }
      }
    }
  }
}

/*-------------------------------------------------------------------- *
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::calc_react_od_fluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
    const int scalar_id, const int number_of_dofs_per_node_mesh, const double rhsfac,
    const double rea_phi)
{
  if (scatra_ele_calc::reamanager_->active() && variable_manager()->evaluate_scalar(scalar_id))
  {
    // total number of porofluid dofs additionally includes volume fraction pressures
    const int total_number_of_porofluid_dofs = variable_manager()->porofluid_material()->num_mat();

    const double vrhs = rhsfac * rea_phi;

    // get pre-factor vector for this scalar
    static std::vector pre_factor_mass_matrix_OD_porofluid(total_number_of_porofluid_dofs, 0.0);
    variable_manager()->get_pre_factor_mass_matrix_od_porofluid(
        scalar_id, &pre_factor_mass_matrix_OD_porofluid);

    calc_mass_matrix_type_od_porofluid(element_matrix, scalar_id,
        &pre_factor_mass_matrix_OD_porofluid, total_number_of_porofluid_dofs, vrhs);
  }
}

/*------------------------------------------------------------------ *
 *----------------------------------------------------------------   */
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<
    discretization_type>::calc_diff_od_fluid(Core::LinAlg::SerialDenseMatrix& element_matrix,
    const int scalar_id, const int number_of_dofs_per_node_mesh, const double rhsfac,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi)
{
  // case of zero saturation/volfrac
  if (variable_manager()->evaluate_scalar(scalar_id))
  {
    // total number of porofluid dofs additionally includes volume fraction pressures
    const int total_number_of_porofluid_dofs = variable_manager()->porofluid_material()->num_mat();

    // get pre-fac vector for this scalar
    static std::vector pre_factor_diffusion_OD_porofluid(total_number_of_porofluid_dofs, 0.0);
    variable_manager()->get_pre_factor_diffusion_od_porofluid(scalar_id, rhsfac,
        scatra_ele_calc::diffmanager_->get_isotropic_diff(scalar_id),
        &pre_factor_diffusion_OD_porofluid);

    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;

      double laplacian(0.0);
      scatra_ele_calc::get_laplacian_weak_form_rhs(laplacian, gradphi, vi);

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const double laplacian_shape_function = laplacian * scatra_ele_calc::funct_(ui);

        // derivative w.r.t. porofluid variables
        for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
        {
          const int fui = ui * total_number_of_porofluid_dofs + dof_id;

          element_matrix(fvi, fui) +=
              laplacian_shape_function * pre_factor_diffusion_OD_porofluid[dof_id];
        }
      }
    }
  }
}

/*---------------------------------------------------------------------*
 *---------------------------------------------------------------------*/
template <Core::FE::CellType discretization_type>
void Discret::Elements::ScaTraEleCalcPorofluidPressureBased<discretization_type>::
    apply_shape_derivs_pressure_grad(Core::LinAlg::SerialDenseMatrix& element_matrix,
        const int scalar_id, const double vrhs, const Core::LinAlg::Matrix<nsd_, 1>& gradient_phi,
        const Core::LinAlg::Matrix<nsd_, 1> pressure_gradient_reference)
{
  if (nsd_ == 3)
  {
    const double xjm_0_0 = scatra_ele_calc::xjm_(0, 0);
    const double xjm_0_1 = scatra_ele_calc::xjm_(0, 1);
    const double xjm_0_2 = scatra_ele_calc::xjm_(0, 2);
    const double xjm_1_0 = scatra_ele_calc::xjm_(1, 0);
    const double xjm_1_1 = scatra_ele_calc::xjm_(1, 1);
    const double xjm_1_2 = scatra_ele_calc::xjm_(1, 2);
    const double xjm_2_0 = scatra_ele_calc::xjm_(2, 0);
    const double xjm_2_1 = scatra_ele_calc::xjm_(2, 1);
    const double xjm_2_2 = scatra_ele_calc::xjm_(2, 2);

    const double refgradpres_0 = pressure_gradient_reference(0);
    const double refgradpres_1 = pressure_gradient_reference(1);
    const double refgradpres_2 = pressure_gradient_reference(2);

    const double gradphi_0 = gradient_phi(0);
    const double gradphi_1 = gradient_phi(1);
    const double gradphi_2 = gradient_phi(2);

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const double v00 =
          +gradphi_1 * (refgradpres_0 * (scatra_ele_calc::deriv_(2, ui) * xjm_1_2 -
                                            scatra_ele_calc::deriv_(1, ui) * xjm_2_2) +
                           refgradpres_1 * (scatra_ele_calc::deriv_(0, ui) * xjm_2_2 -
                                               scatra_ele_calc::deriv_(2, ui) * xjm_0_2) +
                           refgradpres_2 * (scatra_ele_calc::deriv_(1, ui) * xjm_0_2 -
                                               scatra_ele_calc::deriv_(0, ui) * xjm_1_2)) +
          gradphi_2 * (refgradpres_0 * (scatra_ele_calc::deriv_(1, ui) * xjm_2_1 -
                                           scatra_ele_calc::deriv_(2, ui) * xjm_1_1) +
                          refgradpres_1 * (scatra_ele_calc::deriv_(2, ui) * xjm_0_1 -
                                              scatra_ele_calc::deriv_(0, ui) * xjm_2_1) +
                          refgradpres_2 * (scatra_ele_calc::deriv_(0, ui) * xjm_1_1 -
                                              scatra_ele_calc::deriv_(1, ui) * xjm_0_1));
      const double v01 =
          +gradphi_0 * (refgradpres_0 * (scatra_ele_calc::deriv_(1, ui) * xjm_2_2 -
                                            scatra_ele_calc::deriv_(2, ui) * xjm_1_2) +
                           refgradpres_1 * (scatra_ele_calc::deriv_(2, ui) * xjm_0_2 -
                                               scatra_ele_calc::deriv_(0, ui) * xjm_2_2) +
                           refgradpres_2 * (scatra_ele_calc::deriv_(0, ui) * xjm_1_2 -
                                               scatra_ele_calc::deriv_(1, ui) * xjm_0_2)) +
          gradphi_2 * (refgradpres_0 * (scatra_ele_calc::deriv_(2, ui) * xjm_1_0 -
                                           scatra_ele_calc::deriv_(1, ui) * xjm_2_0) +
                          refgradpres_1 * (scatra_ele_calc::deriv_(0, ui) * xjm_2_0 -
                                              scatra_ele_calc::deriv_(2, ui) * xjm_0_0) +
                          refgradpres_2 * (scatra_ele_calc::deriv_(1, ui) * xjm_0_0 -
                                              scatra_ele_calc::deriv_(0, ui) * xjm_1_0));
      const double v02 =
          +gradphi_0 * (refgradpres_0 * (scatra_ele_calc::deriv_(2, ui) * xjm_1_1 -
                                            scatra_ele_calc::deriv_(1, ui) * xjm_2_1) +
                           refgradpres_1 * (scatra_ele_calc::deriv_(0, ui) * xjm_2_1 -
                                               scatra_ele_calc::deriv_(2, ui) * xjm_0_1) +
                           refgradpres_2 * (scatra_ele_calc::deriv_(1, ui) * xjm_0_1 -
                                               scatra_ele_calc::deriv_(0, ui) * xjm_1_1)) +
          gradphi_1 * (refgradpres_0 * (scatra_ele_calc::deriv_(1, ui) * xjm_2_0 -
                                           scatra_ele_calc::deriv_(2, ui) * xjm_1_0) +
                          refgradpres_1 * (scatra_ele_calc::deriv_(2, ui) * xjm_0_0 -
                                              scatra_ele_calc::deriv_(0, ui) * xjm_2_0) +
                          refgradpres_2 * (scatra_ele_calc::deriv_(0, ui) * xjm_1_0 -
                                              scatra_ele_calc::deriv_(1, ui) * xjm_0_0));

      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
        const double v = vrhs * scatra_ele_calc::funct_(vi);

        element_matrix(fvi, ui * 3 + 0) += v * v00;
        element_matrix(fvi, ui * 3 + 1) += v * v01;
        element_matrix(fvi, ui * 3 + 2) += v * v02;
      }
    }
  }
  else if (nsd_ == 2)
  {
    const double refgradpres_0 = pressure_gradient_reference(0);
    const double refgradpres_1 = pressure_gradient_reference(1);

    const double gradphi_0 = gradient_phi(0);
    const double gradphi_1 = gradient_phi(1);

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const double v00 = +gradphi_1 * (-refgradpres_0 * scatra_ele_calc::deriv_(1, ui) +
                                          refgradpres_1 * scatra_ele_calc::deriv_(0, ui));
      const double v01 = +gradphi_0 * (refgradpres_0 * scatra_ele_calc::deriv_(1, ui) -
                                          refgradpres_1 * scatra_ele_calc::deriv_(0, ui));

      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * scatra_ele_calc::numdofpernode_ + scalar_id;
        const double v = vrhs * scatra_ele_calc::funct_(vi);

        element_matrix(fvi, ui * 2 + 0) += v * v00;
        element_matrix(fvi, ui * 2 + 1) += v * v01;
      }
    }
  }
  else
  {
    FOUR_C_THROW("shapederivatives not implemented for 1D!");
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::set_internal_variables_porofluid_pressure_based(const Core::LinAlg::Matrix<nen, 1>&
                                                              shape_functions,
    const Core::LinAlg::Matrix<nsd, nen>& shape_functions_deriv_xyz,
    const Core::LinAlg::Matrix<nsd, nen>& shape_functions_deriv,
    const Core::LinAlg::Matrix<nsd, nsd>& jacobian_matrix,
    const Core::LinAlg::Matrix<nsd, nen>& node_coordinates_reference,
    const std::vector<Core::LinAlg::Matrix<nen, 1>>& element_phi_np,
    const std::vector<Core::LinAlg::Matrix<nen, 1>>& element_phi_n,
    const std::vector<Core::LinAlg::Matrix<nen, 1>>& element_hist,
    const Core::LinAlg::Matrix<nsd, nen>& element_force_velocity)
{
  // call base class (scatra) with dummy variable
  const Core::LinAlg::Matrix<nsd, nen> dummy_element_conv;
  scatra_variable_manager::set_internal_variables(shape_functions, shape_functions_deriv_xyz,
      element_phi_np, element_phi_n, dummy_element_conv, element_hist, dummy_element_conv);

  // velocity due to the external force
  Core::LinAlg::Matrix<nsd, 1> force_velocity;
  force_velocity.multiply(element_force_velocity, shape_functions);

  // get determinant of Jacobian dX / ds
  // transposed jacobian "dX/ds"
  Core::LinAlg::Matrix<nsd, nsd> jacobian_matrix_reference;
  jacobian_matrix_reference.multiply_nt(shape_functions_deriv, node_coordinates_reference);

  // inverse of transposed jacobian "ds/dX"
  const double jacobian_determinant_reference = jacobian_matrix_reference.determinant();
  const double jacobian_determinant = jacobian_matrix.determinant();

  // determinant of deformation gradient
  // det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds))^-1
  const double determinant_deformation_gradient =
      jacobian_determinant / jacobian_determinant_reference;

  // clear current gauss point data for safety
  phase_manager_->clear_gp_state();
  variable_manager_->evaluate_gp_variables(shape_functions, shape_functions_deriv_xyz);
  // access from outside to the phase manager:
  // scatra-discretization has fluid-discretization on dofset 2
  phase_manager_->evaluate_gp_state(
      determinant_deformation_gradient, *variable_manager_, nds_scatra_porofluid_);

  const int number_of_fluid_phases = phase_manager_->num_fluid_phases();
  const int number_of_volume_fractions = phase_manager_->num_vol_frac();
  const int number_of_fluid_dofs = phase_manager_->total_num_dof();

  // resize all phase related vectors
  pressure_.resize(number_of_fluid_phases);
  saturation_.resize(number_of_fluid_phases);
  density_.resize(number_of_fluid_phases + number_of_volume_fractions);
  heat_capacity_.resize(number_of_fluid_phases + number_of_volume_fractions + 1);
  thermal_diffusivity_.resize(number_of_fluid_phases + number_of_volume_fractions + 1);
  pressure_gradient_.resize(number_of_fluid_phases + number_of_volume_fractions);
  diffusion_tensor_porofluid_.resize(number_of_fluid_phases + number_of_volume_fractions);
  pressure_gradient_norm_.resize(number_of_fluid_phases);
  volume_fractions_.resize(number_of_volume_fractions);
  volume_fractions_pressure_.resize(number_of_volume_fractions);
  relative_mobility_funct_id_.resize(scatra_variable_manager::numscal_);

  const std::vector<Core::LinAlg::Matrix<nsd, 1>>& fluid_gradient_phi =
      *(variable_manager_->grad_phinp());

  volume_fractions_ = phase_manager_->vol_frac();
  volume_fractions_pressure_ = phase_manager_->vol_frac_pressure();

  // set closing relation type volfrac
  auto closing_relation = std::invoke(
      [&]()
      {
        if (number_of_fluid_dofs == number_of_fluid_phases + number_of_volume_fractions)
          return Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung;
        else if (number_of_fluid_dofs == number_of_fluid_phases + 2 * number_of_volume_fractions)
          return Mat::PAR::PoroFluidPressureBased::ClosingRelation::
              evolutionequation_homogenized_vasculature_tumor;
        else
          FOUR_C_THROW("Internal error!");
      });

  //! convective velocity
  std::vector<Core::LinAlg::Matrix<nsd, 1>> convective_phase_velocity(0.0);
  convective_phase_velocity.resize(number_of_fluid_phases + number_of_volume_fractions);
  //! convective part in convective form: (u_x * N,x) + (u_y * N,y)
  std::vector<Core::LinAlg::Matrix<nen, 1>> convective_phase_velocity_convective_form(0.0);
  convective_phase_velocity_convective_form.resize(
      number_of_fluid_phases + number_of_volume_fractions);

  //! temperature convective velocity
  Core::LinAlg::Matrix<nsd, 1> convective_temperature_velocity;
  //! temperature convective part in convective form
  Core::LinAlg::Matrix<nen, 1> convective_temperature_velocity_convective_form;

  for (int phase_id = 0; phase_id < number_of_fluid_phases; ++phase_id)
  {
    // current pressure gradient
    pressure_gradient_[phase_id].clear();

    // phase density
    density_[phase_id] = phase_manager_->density(phase_id);

    // compute the pressure gradient from the phi gradients
    for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
      pressure_gradient_[phase_id].update(
          phase_manager_->pressure_deriv(phase_id, dof_id), fluid_gradient_phi[dof_id], 1.0);

    // compute the absolute value of the pressure gradient from the phi gradients
    pressure_gradient_norm_[phase_id] = 0.0;
    for (int i = 0; i < nsd; i++)
      pressure_gradient_norm_[phase_id] +=
          pressure_gradient_[phase_id](i) * pressure_gradient_[phase_id](i);
    pressure_gradient_norm_[phase_id] = sqrt(pressure_gradient_norm_[phase_id]);

    // diffusion tensor
    diffusion_tensor_porofluid_[phase_id].clear();
    phase_manager_->permeability_tensor(phase_id, diffusion_tensor_porofluid_[phase_id]);
    diffusion_tensor_porofluid_[phase_id].scale(
        phase_manager_->rel_permeability(phase_id) / phase_manager_->dyn_viscosity(phase_id,
                                                         pressure_gradient_norm_[phase_id],
                                                         nds_scatra_porofluid_));

    // Insert Darcy's law:
    // porosity * S^phase * (v^phase - v^s) = - permeability/viscosity * grad p
    convective_phase_velocity[phase_id].multiply(
        -1.0, diffusion_tensor_porofluid_[phase_id], pressure_gradient_[phase_id]);
    convective_temperature_velocity.update(
        heat_capacity_[phase_id] * density_[phase_id], convective_phase_velocity[phase_id], 1.0);
    // in convective form: (u_x * N,x) + (u_y * N,y)
    convective_phase_velocity_convective_form[phase_id].multiply_tn(
        shape_functions_deriv_xyz, convective_phase_velocity[phase_id]);
    convective_temperature_velocity_convective_form.update(
        heat_capacity_[phase_id] * density_[phase_id],
        convective_phase_velocity_convective_form[phase_id], 1.0);

    // phase pressure
    pressure_[phase_id] = phase_manager_->pressure(phase_id);
    // phase saturation
    saturation_[phase_id] = phase_manager_->saturation(phase_id);
  }

  for (int volume_fraction_id = number_of_fluid_phases;
      volume_fraction_id < number_of_fluid_phases + number_of_volume_fractions;
      ++volume_fraction_id)
  {
    if (closing_relation ==
        Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung)
    {
      // current pressure gradient
      pressure_gradient_[volume_fraction_id].update(
          1.0, fluid_gradient_phi[volume_fraction_id], 0.0);

      // vol frac density
      density_[volume_fraction_id] =
          phase_manager_->vol_frac_density(volume_fraction_id - number_of_fluid_phases);

      // diffusion tensor
      diffusion_tensor_porofluid_[volume_fraction_id].clear();
      phase_manager_->permeability_tensor_vol_frac_pressure(
          volume_fraction_id - number_of_fluid_phases,
          diffusion_tensor_porofluid_[volume_fraction_id]);
      diffusion_tensor_porofluid_[volume_fraction_id].scale(
          1.0 / phase_manager_->dyn_viscosity_vol_frac_pressure_blood_lung(
                    volume_fraction_id - number_of_fluid_phases, -1.0,
                    nds_scatra_porofluid_));  // -1.0 --> don't need abspressgrad

      // Insert Darcy's law: porosity*(v^\pi - v_s) = - k/\mu * grad p
      convective_phase_velocity[volume_fraction_id].multiply(-1.0,
          diffusion_tensor_porofluid_[volume_fraction_id], pressure_gradient_[volume_fraction_id]);
      convective_temperature_velocity.update(
          heat_capacity_[volume_fraction_id] * density_[volume_fraction_id],
          convective_phase_velocity[volume_fraction_id], 1.0);
      // in convective form: u_x*N,x + u_y*N,y
      convective_phase_velocity_convective_form[volume_fraction_id].multiply_tn(
          shape_functions_deriv_xyz, convective_phase_velocity[volume_fraction_id]);
      convective_temperature_velocity_convective_form.update(
          heat_capacity_[volume_fraction_id] * density_[volume_fraction_id],
          convective_phase_velocity_convective_form[volume_fraction_id], 1.0);
    }
    else
    {
      // current pressure gradient
      pressure_gradient_[volume_fraction_id].update(
          1.0, fluid_gradient_phi[volume_fraction_id + number_of_volume_fractions], 0.0);

      // density of the volume fraction
      density_[volume_fraction_id] =
          phase_manager_->vol_frac_density(volume_fraction_id - number_of_fluid_phases);

      // diffusion tensor
      diffusion_tensor_porofluid_[volume_fraction_id].clear();
      phase_manager_->permeability_tensor_vol_frac_pressure(
          volume_fraction_id - number_of_fluid_phases,
          diffusion_tensor_porofluid_[volume_fraction_id]);
      // -1.0 --> don't need pressure gradient norm
      diffusion_tensor_porofluid_[volume_fraction_id].scale(
          1.0 / phase_manager_->dyn_viscosity_vol_frac_pressure(
                    volume_fraction_id - number_of_fluid_phases, -1.0, nds_scatra_porofluid_));

      // Insert Darcy's law: volume fraction * (v^phase - v^s) = - permeability/viscosity * grad p
      convective_phase_velocity[volume_fraction_id].multiply(-1.0,
          diffusion_tensor_porofluid_[volume_fraction_id], pressure_gradient_[volume_fraction_id]);
      convective_temperature_velocity.update(
          heat_capacity_[volume_fraction_id] * density_[volume_fraction_id],
          convective_phase_velocity[volume_fraction_id], 1.0);
      // in convective form: (u_x * N,x) + (u_y * N,y)
      convective_phase_velocity_convective_form[volume_fraction_id].multiply_tn(
          shape_functions_deriv_xyz, convective_phase_velocity[volume_fraction_id]);
      convective_temperature_velocity_convective_form.update(
          heat_capacity_[volume_fraction_id] * density_[volume_fraction_id],
          convective_phase_velocity_convective_form[volume_fraction_id], 1.0);
    }
  }

  // solid pressure
  solid_pressure_ = phase_manager_->solid_pressure();

  for (int scalar_id = 0; scalar_id < scatra_variable_manager::numscal_; ++scalar_id)
  {
    // overwrite convective term: - rho * k/\mu*grad p * grad phi
    switch (scalar_to_phase_map_[scalar_id].species_type)
    {
      case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
      case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
        scatra_variable_manager::convelint_[scalar_id] =
            convective_phase_velocity[scalar_to_phase_map_[scalar_id].phaseID];
        // if the scalar reacts to the external force, add the velocity due to the external
        // force scaled with the relative mobility and the porosity * saturation
        if (scatra_variable_manager::reacts_to_force_[scalar_id])
        {
          const auto prefactor =
              evaluate_relative_mobility(scalar_id) * phase_manager_->porosity() *
              phase_manager_->saturation(scalar_to_phase_map_[scalar_id].phaseID);
          scatra_variable_manager::convelint_[scalar_id].update(prefactor, force_velocity, 1.0);
        }
        scatra_variable_manager::conv_[scalar_id].multiply_tn(
            shape_functions_deriv_xyz, scatra_variable_manager::convelint_[scalar_id]);
        scatra_variable_manager::conv_phi_[scalar_id] =
            scatra_variable_manager::convelint_[scalar_id].dot(
                scatra_variable_manager::gradphi_[scalar_id]);
        break;

      case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
        scatra_variable_manager::convelint_[scalar_id] =
            Core::LinAlg::Matrix<nsd, 1>(Core::LinAlg::Initialization::zero);
        scatra_variable_manager::conv_[scalar_id] =
            Core::LinAlg::Matrix<nen, 1>(Core::LinAlg::Initialization::zero);
        scatra_variable_manager::conv_phi_[scalar_id] = 0;
        break;

      case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
        scatra_variable_manager::convelint_[scalar_id] = convective_temperature_velocity;
        scatra_variable_manager::conv_[scalar_id] = convective_temperature_velocity_convective_form;
        scatra_variable_manager::conv_phi_[scalar_id] =
            convective_temperature_velocity.dot(scatra_variable_manager::gradphi_[scalar_id]);
        break;

      default:
        FOUR_C_THROW("Unknown species type {} for species {}.",
            scalar_to_phase_map_[scalar_id].species_type, scalar_id);
    }
    // set flag if we actually have to evaluate the species
    set_evaluate_scalar_flag(scalar_id);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::adapt_convective_term_for_l2(const Core::LinAlg::Matrix<nen, 1>& shape_functions,
    const Core::LinAlg::Matrix<nsd, nen>& shape_functions_deriv_xyz,
    const std::vector<Core::LinAlg::Matrix<nsd, nen>>& element_flux_np)
{
  const int number_of_fluid_phases = element_flux_np.size();

  std::vector<Core::LinAlg::Matrix<nsd, 1>> flux(0.0);
  flux.resize(number_of_fluid_phases);

  // in convective form: q_x*N,x + q_y*N,y
  std::vector<Core::LinAlg::Matrix<nen, 1>> flux_conv(0.0);
  flux_conv.resize(number_of_fluid_phases);

  for (int phase_id = 0; phase_id < number_of_fluid_phases; ++phase_id)
  {
    flux[phase_id].multiply(1.0, element_flux_np[phase_id], shape_functions);
    flux_conv[phase_id].multiply_tn(shape_functions_deriv_xyz, flux[phase_id]);
  }

  for (const auto& [phase_id, _] : scalar_to_phase_map_)
  {
    if (phase_id < 0 || phase_id >= number_of_fluid_phases)
      FOUR_C_THROW("Invalid phase ID {}", phase_id);
  }

  // set convective term
  for (int scalar_id = 0; scalar_id < scatra_variable_manager::numscal_; ++scalar_id)
  {
    switch (scalar_to_phase_map_[scalar_id].species_type)
    {
      case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
      case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
        scatra_variable_manager::conv_phi_[scalar_id] =
            flux[scalar_to_phase_map_[scalar_id].phaseID].dot(
                scatra_variable_manager::gradphi_[scalar_id]);
        break;

      case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
      case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
        scatra_variable_manager::conv_phi_[scalar_id] = 0;
        break;

      default:
        FOUR_C_THROW("Unknown species type {} for species {}.",
            scalar_to_phase_map_[scalar_id].species_type, scalar_id);
    }
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
double Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_pre_factor_mass_matrix_od_mesh(const int scalar_id, const double fac)
{
  double pre_factor = fac;

  switch (scalar_to_phase_map_[scalar_id].species_type)
  {
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
    {
      // linearization of porosity
      //
      // dporosity/dd = dporosity/dJ * dJ/dd = dporosity/dJ * J * N_x
      // J denotes the determinant of the deformation gradient, i.e.,
      // det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
      // in our case: pre_factor is scaled with density, i.e., porosity
      // --> scale it with 1.0/porosity here
      //
      // We will pass fac/porosity * J * dporosity/dJ into CalcMatMassODMesh, where it will
      // internally be scaled correctly.
      if (phase_manager_->porosity_depends_on_struct())
        pre_factor += fac / (phase_manager_->porosity()) * phase_manager_->jacobian_def_grad() *
                      phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
    {
      // do nothing: correct pre_factor = fac
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
    {
      if (phase_manager_->porosity_depends_on_struct())
      {
        pre_factor -= fac / (1 - phase_manager_->porosity() - phase_manager_->sum_add_vol_frac()) *
                      phase_manager_->jacobian_def_grad() *
                      phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
    {
      if (phase_manager_->porosity_depends_on_struct())
      {
        // evaluate the effective heat capacity factor
        const int total_number_of_porofluid_phases =
            phase_manager_->num_fluid_phases() + phase_manager_->num_vol_frac();
        const int number_of_fluid_phases = phase_manager_->num_fluid_phases();

        const auto effective_heat_capacity = get_effective_heat_capacity();

        pre_factor += fac / effective_heat_capacity * phase_manager_->solid_density() *
                      get_heat_capacity(total_number_of_porofluid_phases) * (-1.0) *
                      phase_manager_->jacobian_def_grad() *
                      phase_manager_->porosity_deriv_wrt_jacobian_def_grad();

        for (int phase = 0; phase < number_of_fluid_phases; ++phase)
        {
          pre_factor += fac / effective_heat_capacity * phase_manager_->density(phase) *
                        get_heat_capacity(phase) * phase_manager_->jacobian_def_grad() *
                        phase_manager_->saturation(phase) *
                        phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
        }
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown species type {} for species {}.",
          scalar_to_phase_map_[scalar_id].species_type, scalar_id);
  }
  return pre_factor;
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_pre_factor_mass_matrix_od_porofluid(const int scalar_id,
    std::vector<double>* pre_factor_mass_matrix_od_porofluid)
{
  // reset to zero
  std::ranges::fill(pre_factor_mass_matrix_od_porofluid->begin(),
      pre_factor_mass_matrix_od_porofluid->end(), 0.0);

  const int number_of_fluid_phases = phase_manager_->num_fluid_phases();

  switch (scalar_to_phase_map_[scalar_id].species_type)
  {
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
    {
      const int phase_id = get_phase_id(scalar_id);
      const int total_number_of_porofluid_dofs = porofluid_material()->num_mat();

      // d poro / d S_j = SaturationDeriv
      // --> scaling with 1.0/S since term is re-scaled with densnp = porosity * S * rho
      for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
      {
        (*pre_factor_mass_matrix_od_porofluid)[dof_id] +=
            phase_manager_->saturation_deriv(phase_id, dof_id) /
            phase_manager_->saturation(phase_id);
      }

      // linearization of porosity (only if porosity is pressure-dependent)
      // d poro / d psi_j = porosityDeriv
      // --> scaling with 1.0/porosity  since term is re-scaled with densnp = porosity * S *rho
      if (phase_manager_->porosity_depends_on_fluid())
      {
        for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
          (*pre_factor_mass_matrix_od_porofluid)[dof_id] +=
              phase_manager_->porosity_deriv(dof_id) / phase_manager_->porosity();
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
    {
      const int phase_id = get_phase_id(scalar_id);
      // only necessary in volfrac closing relation "homogenized_vasculature_tumor", because
      // otherwise volfrac is no primary variable d volfrac_j / d volfrac_j = 1.0
      // --> scaling with 1.0/volfrac since term is re-scaled with densnp = rho * volfrac
      if (porofluid_material_->num_mat() ==
          (number_of_fluid_phases + 2 * phase_manager_->num_vol_frac()))
      {
        (*pre_factor_mass_matrix_od_porofluid)[phase_id] += 1.0 / get_volume_fraction(scalar_id);
      }

      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
    {
      // porosity of solid epsilon_s does not depend on porofluid primary variables
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
    {
      // evaluate the effective heat capacity factor
      const int total_number_of_porofluid_phases =
          phase_manager_->num_fluid_phases() + phase_manager_->num_vol_frac();
      // total number of porofluid dofs additionally includes volume fraction pressures
      const int total_number_of_porofluid_dofs = porofluid_material()->num_mat();

      const auto effective_heat_capacity = get_effective_heat_capacity();

      for (int phase = 0; phase < number_of_fluid_phases; ++phase)
      {
        for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
        {
          (*pre_factor_mass_matrix_od_porofluid)[dof_id] +=
              phase_manager_->saturation_deriv(phase, dof_id) / effective_heat_capacity *
              phase_manager_->density(phase) * get_heat_capacity(phase) *
              phase_manager_->porosity();
        }

        if (phase_manager_->porosity_depends_on_fluid())
        {
          for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
          {
            (*pre_factor_mass_matrix_od_porofluid)[dof_id] +=
                phase_manager_->porosity_deriv(dof_id) / effective_heat_capacity *
                phase_manager_->density(phase) * get_heat_capacity(phase) *
                phase_manager_->saturation(phase);
          }
        }
      }

      for (int phase = number_of_fluid_phases; phase < total_number_of_porofluid_phases; ++phase)
      {
        (*pre_factor_mass_matrix_od_porofluid)[phase] +=
            get_heat_capacity(phase) *
            phase_manager_->vol_frac_density(phase - number_of_fluid_phases) /
            effective_heat_capacity;
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown species type {} for species {}.",
          scalar_to_phase_map_[scalar_id].species_type, scalar_id);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
auto Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::evaluate_relative_mobility(const int current_scalar) const
{
  std::vector<std::pair<std::string, double>> varfunction_variables;
  std::vector<std::pair<std::string, double>> varfunction_constants;
  const auto number_of_fluid_phases = phase_manager_->num_fluid_phases();

  for (int phase_id = 0; phase_id < number_of_fluid_phases; phase_id++)
  {
    varfunction_variables.emplace_back(
        "S" + std::to_string(phase_id + 1), phase_manager_->saturation(phase_id));
    varfunction_variables.emplace_back(
        "p" + std::to_string(phase_id + 1), phase_manager_->pressure(phase_id));
  }
  varfunction_variables.emplace_back("porosity", phase_manager_->porosity());

  const auto relative_mobility = Global::Problem::instance()
                                     ->function_by_id<Core::Utils::FunctionOfAnything>(
                                         relative_mobility_funct_id_[current_scalar])
                                     .evaluate(varfunction_variables, varfunction_constants, 0);

  return relative_mobility;
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_pre_factor_convection_od_porofluid(const unsigned ui,
    std::vector<double>* pre_factor_convection_od_porofluid,
    const Core::LinAlg::Matrix<nsd, 1>& grad_phi,
    const Core::LinAlg::Matrix<1, nsd>& grad_phi_transpose_diffusion_tensor,
    const Core::LinAlg::Matrix<nen, 1>& shape_functions,
    const Core::LinAlg::Matrix<nsd, nen>& shape_functions_deriv_xyz, const int phase_id)
{
  // reset to zero
  std::ranges::fill(
      pre_factor_convection_od_porofluid->begin(), pre_factor_convection_od_porofluid->end(), 0.0);

  // get correct factor
  double laplacian(0.0);
  for (int dim = 0; dim < nsd; dim++)
    laplacian += shape_functions_deriv_xyz(dim, ui) * grad_phi_transpose_diffusion_tensor(0, dim);

  const int number_of_fluid_phases = phase_manager_->num_fluid_phases();
  const int number_of_volume_fractions = phase_manager_->num_vol_frac();
  const int number_of_fluid_dofs = phase_manager_->total_num_dof();

  // fluid phase
  if (phase_id >= 0 && phase_id < number_of_fluid_phases)
  {
    // derivative after fluid pressures
    for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
      (*pre_factor_convection_od_porofluid)[dof_id] +=
          laplacian * phase_manager_->pressure_deriv(phase_id, dof_id);

    // derivative after relative permeability
    if (not phase_manager_->has_constant_rel_permeability(phase_id))
    {
      const Core::LinAlg::Matrix<nsd, 1> pressure_gradient = get_pressure_gradient(phase_id);
      const double pressure_gradient_norm = get_pressure_gradient_norm(phase_id);

      static Core::LinAlg::Matrix<nsd, nsd> diffusion_tensor;
      phase_manager_->permeability_tensor(phase_id, diffusion_tensor);
      diffusion_tensor.scale(phase_manager_->rel_permeability_deriv(phase_id) /
                             phase_manager_->dyn_viscosity(phase_id, pressure_gradient_norm, 2));

      static Core::LinAlg::Matrix<1, nsd> phase_grad_phi_transpose_diffusion_tensor;
      phase_grad_phi_transpose_diffusion_tensor.multiply_tn(grad_phi, diffusion_tensor);

      double phase_laplacian(0.0);
      for (unsigned j = 0; j < nsd; j++)
        phase_laplacian += pressure_gradient(j) * phase_grad_phi_transpose_diffusion_tensor(0, j);

      for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
      {
        (*pre_factor_convection_od_porofluid)[dof_id] +=
            phase_laplacian * shape_functions(ui) *
            phase_manager_->saturation_deriv(phase_id, dof_id);
      }
    }
  }
  // volume fraction (in additional porous network)
  else if (phase_id < number_of_fluid_phases + number_of_volume_fractions)
  {
    // derivative after volfrac-pressure
    // for volfrac with closing relation "homogenized_vasculature_tumor"
    if (number_of_fluid_dofs == number_of_fluid_phases + 2 * number_of_volume_fractions)
    {
      (*pre_factor_convection_od_porofluid)[phase_id + number_of_volume_fractions] += laplacian;
    }
    else  // derivative after volfrac-pressure for volfrac with closing relation "blood lung"
    {
      (*pre_factor_convection_od_porofluid)[phase_id] += laplacian;
    }
  }
  else
  {
    FOUR_C_THROW("get_pre_factor_convection_od_porofluid has been called with phase {}", phase_id);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
double Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_pre_factor_diffusion_od_mesh(const int scalar_id, const double rhsfac,
    const double diffusion_coefficient)
{
  double pre_factor = 0.0;

  switch (scalar_to_phase_map_[scalar_id].species_type)
  {
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
    {
      if (phase_manager_->porosity_depends_on_struct())
      {
        const double delta = get_delta(scalar_id);

        // linearization of porosity
        // dporosity/dd = dporosity/dJ * dJ/dd = dporosity/dJ * J * N_x
        // J denotes the determinant of the deformation gradient, i.e.,
        // det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
        //
        // in our case: diffusivity is scaled with porosity^(delta + 1)
        // --> scale it with 1.0/porosity^(delta + 1) here and build derivative
        // d diff/d porosity = (delta + 1) * porosity^delta
        pre_factor = rhsfac * diffusion_coefficient /
                     std::pow(phase_manager_->porosity(), delta + 1.0) * (delta + 1.0) *
                     std::pow(phase_manager_->porosity(), delta) *
                     phase_manager_->jacobian_def_grad() *
                     phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
    {
      // do nothing: linearization performed in base class
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
    {
      if (phase_manager_->porosity_depends_on_struct())
      {
        const double delta = get_delta(scalar_id);

        // linearization of porosity
        // dporosity/dd = dporosity/dJ * dJ/dd = dporosity/dJ * J * N_x
        // J denotes the determinant of the deformation gradient, i.e.,
        // det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
        //
        // in our case: diffusivity is scaled with (1-porosity-sum_volfrac)^(delta + 1)
        // --> scale it with 1.0/(1-porosity-sum_volfrac)^(delta + 1) here and build
        // derivative
        // d diff/d porosity = (delta + 1) * (1-porosity-sum_volfrac)^delta
        pre_factor =
            -1.0 * rhsfac * diffusion_coefficient /
            std::pow(1.0 - phase_manager_->porosity() - phase_manager_->sum_add_vol_frac(),
                delta + 1.0) *
            (delta + 1.0) *
            std::pow(1.0 - phase_manager_->porosity() - phase_manager_->sum_add_vol_frac(), delta) *
            phase_manager_->jacobian_def_grad() *
            phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
    {
      if (phase_manager_->porosity_depends_on_struct())
      {
        const int total_number_of_porofluid_phases =
            phase_manager_->num_fluid_phases() + phase_manager_->num_vol_frac();
        const int number_of_fluid_phases = phase_manager_->num_fluid_phases();

        pre_factor = rhsfac * get_thermal_diffusivity(total_number_of_porofluid_phases) * (-1.0) *
                     phase_manager_->jacobian_def_grad() *
                     phase_manager_->porosity_deriv_wrt_jacobian_def_grad();

        for (int phase = 0; phase < number_of_fluid_phases; ++phase)
        {
          pre_factor += rhsfac * get_thermal_diffusivity(phase) *
                        phase_manager_->jacobian_def_grad() * phase_manager_->saturation(phase) *
                        phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
        }
      }
      break;
    }
    default:
      FOUR_C_THROW("Unknown species type {} for species {}.",
          scalar_to_phase_map_[scalar_id].species_type, scalar_id);
  }
  return pre_factor;
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_pre_factor_diffusion_od_porofluid(const int scalar_id, const double rhsfac,
    const double diffusion_coefficient, std::vector<double>* pre_factor_diffusion_OD_porofluid)
{
  // reset to zero
  std::ranges::fill(
      pre_factor_diffusion_OD_porofluid->begin(), pre_factor_diffusion_OD_porofluid->end(), 0.0);

  switch (scalar_to_phase_map_[scalar_id].species_type)
  {
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
    {
      const int phase_id = get_phase_id(scalar_id);
      const int number_of_fluid_phases = phase_manager_->num_fluid_phases();
      const int total_number_of_porofluid_dofs = porofluid_material()->num_mat();

      const double delta = get_delta(scalar_id);

      // linearization of saturation * porosity * d_eff = (saturation * porosity)^(delta+1)
      // w.r.t saturation
      //
      // in our case: diffusivity is scaled with saturation^(delta + 1)
      // --> scale it with 1.0/saturation^(delta + 1) here and build derivative
      // d diff/d saturation = (delta + 1) * saturation^delta
      const double vrhs_sat = rhsfac * diffusion_coefficient /
                              std::pow(phase_manager_->saturation(phase_id), delta + 1.0) *
                              (delta + 1.0) * std::pow(phase_manager_->saturation(phase_id), delta);

      // linearization of saturation * porosity * d_eff = (saturation * porosity)^(delta+1)
      // w.r.t porosity
      //
      // in our case: diffusivity is scaled with porosity^(delta + 1)
      // --> scale it with 1.0/porosity^(delta + 1) here and build derivative
      // d diff/d porosity = (delta + 1) * porosity^delta
      const double vrhs_poro = rhsfac * diffusion_coefficient /
                               std::pow(phase_manager_->porosity(), delta + 1.0) * (delta + 1.0) *
                               std::pow(phase_manager_->porosity(), delta);

      for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
        (*pre_factor_diffusion_OD_porofluid)[dof_id] +=
            vrhs_sat * phase_manager_->saturation_deriv(phase_id, dof_id);

      // linearization of porosity (only if porosity is pressure-dependent)
      if (phase_manager_->porosity_depends_on_fluid())
      {
        for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
          (*pre_factor_diffusion_OD_porofluid)[dof_id] +=
              vrhs_poro * phase_manager_->porosity_deriv(dof_id);
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
    {
      const int phase_id = get_phase_id(scalar_id);
      const double delta = get_delta(scalar_id);
      // diffusivity is scaled with volfrac^(delta + 1)
      // --> scale it with 1.0/volfrac^(delta + 1) here and build derivative
      // d diff/d volfrac = (delta + 1) * volfrac^delta
      (*pre_factor_diffusion_OD_porofluid)[phase_id] +=
          rhsfac * diffusion_coefficient / std::pow(get_volume_fraction(scalar_id), delta + 1.0) *
          (delta + 1.0) * std::pow(get_volume_fraction(scalar_id), delta);
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
    {
      // porosity of solid epsilon_s does not depend on fluid primary variables
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
    {
      const int number_of_fluid_phases = phase_manager_->num_fluid_phases();
      const int total_number_of_porofluid_phases =
          number_of_fluid_phases + phase_manager_->num_vol_frac();
      // total number of porofluid dofs additionally includes volume fraction pressures
      const int total_number_of_porofluid_dofs = porofluid_material()->num_mat();

      for (int phase_id = 0; phase_id < number_of_fluid_phases; ++phase_id)
      {
        for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
        {
          (*pre_factor_diffusion_OD_porofluid)[dof_id] +=
              rhsfac * phase_manager_->saturation_deriv(phase_id, dof_id) *
              get_thermal_diffusivity(phase_id) * phase_manager_->porosity();
        }

        if (phase_manager_->porosity_depends_on_fluid())
        {
          for (int dof_id = 0; dof_id < total_number_of_porofluid_dofs; ++dof_id)
          {
            (*pre_factor_diffusion_OD_porofluid)[dof_id] +=
                rhsfac * phase_manager_->porosity_deriv(dof_id) *
                get_thermal_diffusivity(phase_id) * phase_manager_->saturation(phase_id);
          }
        }
      }

      for (int phase_id = number_of_fluid_phases; phase_id < total_number_of_porofluid_phases;
          ++phase_id)
        (*pre_factor_diffusion_OD_porofluid)[phase_id] +=
            rhsfac * get_thermal_diffusivity(phase_id);
      break;
    }
    default:
      FOUR_C_THROW("Unknown species type {} for species {}.",
          scalar_to_phase_map_[scalar_id].species_type, scalar_id);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_diffusion_tensor_fluid(const int scalar_id,
    Core::LinAlg::Matrix<nsd, nsd>& diffusion_tensor, const int phase_id)
{
  switch (scalar_to_phase_map_[scalar_id].species_type)
  {
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
    {
      diffusion_tensor = diffusion_tensor_porofluid_[phase_id];
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
    default:
      FOUR_C_THROW("Unknown species type {} for species {}.",
          scalar_to_phase_map_[scalar_id].species_type, scalar_id);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_pressure_gradient_reference(const Core::LinAlg::Matrix<nsd, nsd>& jacobian_matrix,
    Core::LinAlg::Matrix<nsd, 1>& pressure_gradient_reference, const int phase_id)
{
  pressure_gradient_reference.clear();

  const int number_of_fluid_phases = phase_manager_->num_fluid_phases();
  const int number_of_volume_fractions = phase_manager_->num_vol_frac();

  // fluid phase
  if (phase_id >= 0 && phase_id < number_of_fluid_phases)
  {
    const std::vector<Core::LinAlg::Matrix<nsd, 1>>& gradient_fluid_phi =
        *(variable_manager_->grad_phinp());

    // gradient of phi w.r.t. reference coordinates
    std::vector<Core::LinAlg::Matrix<nsd, 1>> gradient_fluid_phi_reference(
        number_of_fluid_phases, Core::LinAlg::Matrix<nsd, 1>(Core::LinAlg::Initialization::zero));
    for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
      gradient_fluid_phi_reference[dof_id].multiply(jacobian_matrix, gradient_fluid_phi[dof_id]);

    // compute the pressure gradient from the phi gradients
    for (int dof_id = 0; dof_id < number_of_fluid_phases; ++dof_id)
      pressure_gradient_reference.update(phase_manager_->pressure_deriv(phase_id, dof_id),
          gradient_fluid_phi_reference[dof_id], 1.0);
  }

  // volume fraction (in additional porous network)
  else if (phase_id < number_of_fluid_phases + number_of_volume_fractions)
  {
    pressure_gradient_reference.multiply(jacobian_matrix, pressure_gradient_[phase_id]);
  }

  else
  {
    FOUR_C_THROW("GetRefGradPres has been called with phase {}", phase_id);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::set_evaluate_scalar_flag(const int scalar_id)
{
  const int number_of_fluid_phases = phase_manager_->num_fluid_phases();

  switch (scalar_to_phase_map_[scalar_id].species_type)
  {
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
    {
      // we do not evaluate if smaller than threshold (at GP)
      evaluate_scalar_[scalar_id] = (fabs(saturation_[scalar_to_phase_map_[scalar_id].phaseID]) >
                                     minimal_value_of_phase_[scalar_id]);
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
    {
      // we do not evaluate if smaller than minimum nodal volume fraction in element
      if (phase_manager_->total_num_dof() ==
          phase_manager_->num_fluid_phases() + 2 * phase_manager_->num_vol_frac())
      {
        evaluate_scalar_[scalar_id] = variable_manager_->element_has_valid_vol_frac_species(
            scalar_to_phase_map_[scalar_id].phaseID - number_of_fluid_phases);
      }
      else  // blood lung
      {
        evaluate_scalar_[scalar_id] = true;
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
    {
      evaluate_scalar_[scalar_id] = true;
      break;
    }
    default:
      FOUR_C_THROW("Unknown species type {} for species {}.",
          scalar_to_phase_map_[scalar_id].species_type, scalar_id);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
double Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::get_pre_factor_hist_and_source_od_mesh(const int scalar_id, const double fac,
    const double densnp, const double hist, const double rhsint)
{
  // linearization of mesh motion: call base class with fac * rhsint
  double pre_factor = fac * rhsint;

  switch (scalar_to_phase_map_[scalar_id].species_type)
  {
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid:
    {
      // linearization of history: prefactor is densnp = porosity * rho * S
      // --> porosity has to be linearized with linearization of porosity
      //
      // dporosity/dd = dporosity/dJ * dJ/dd = dporosity/dJ * J * N_x
      // J denotes the determinant of the deformation gradient, i.e.,
      // det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
      // in our case: pre_factor is scaled with density, i.e., porosity
      // --> scale it with 1.0/porosity here

      if (phase_manager_->porosity_depends_on_struct())
      {
        pre_factor += 1.0 * fac * hist * densnp / (phase_manager_->porosity()) *
                      phase_manager_->jacobian_def_grad() *
                      phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac:
    {
      // do nothing: correct pre_factor = fac * rhsint
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid:
    {
      if (phase_manager_->porosity_depends_on_struct())
      {
        pre_factor -= fac * hist * densnp /
                      (1 - phase_manager_->porosity() - phase_manager_->sum_add_vol_frac()) *
                      phase_manager_->jacobian_def_grad() *
                      phase_manager_->porosity_deriv_wrt_jacobian_def_grad();
      }
      break;
    }
    case Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature:
    {
      // evaluate the effective heat capacity factor
      const int number_of_dofs =
          phase_manager_->num_fluid_phases() + phase_manager_->num_vol_frac();
      const int number_of_fluid_phases = phase_manager_->num_fluid_phases();

      const auto effective_heat_capacity = get_effective_heat_capacity();

      pre_factor += fac * hist * densnp / effective_heat_capacity *
                    phase_manager_->solid_density() * get_heat_capacity(number_of_dofs) * (-1.0) *
                    phase_manager_->jacobian_def_grad() *
                    phase_manager_->porosity_deriv_wrt_jacobian_def_grad();

      for (int phase_id = 0; phase_id < number_of_fluid_phases; ++phase_id)
      {
        pre_factor += fac * hist * densnp / effective_heat_capacity *
                      phase_manager_->density(phase_id) * get_heat_capacity(phase_id) *
                      phase_manager_->saturation(phase_id) *
                      phase_manager_->porosity_deriv_wrt_jacobian_def_grad() *
                      phase_manager_->jacobian_def_grad();
      }
      break;
    }

    default:
      FOUR_C_THROW("Unknown species type {} for species {}.",
          scalar_to_phase_map_[scalar_id].species_type, scalar_id);
  }

  return pre_factor;
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::setup_porofluid_managers(const Core::Elements::Element* element,
    const Core::FE::Discretization& discretization, const int number_of_fluid_phases,
    const int total_number_of_porofluid_dofs, const int number_volfracs)
{
  PoroFluidMultiPhaseEleParameter* parameters =
      PoroFluidMultiPhaseEleParameter::instance(discretization.name());

  phase_manager_ = PoroFluidManager::PhaseManagerInterface::create_phase_manager(*parameters, nsd,
      porofluid_material()->material_type(), PoroPressureBased::Action::get_access_from_scatra,
      total_number_of_porofluid_dofs, number_of_fluid_phases, number_volfracs);

  // access from outside to the phase manager:
  // scatra-discretization has fluid-discretization on dofset 2
  phase_manager_->setup(element, nds_scatra_porofluid_);

  variable_manager_ = PoroFluidManager::VariableManagerInterface<nsd, nen>::create_variable_manager(
      *parameters, PoroPressureBased::Action::get_access_from_scatra, porofluid_material(),
      total_number_of_porofluid_dofs, number_of_fluid_phases, number_volfracs);
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::ScaTraEleInternalVariableManagerPorofluidPressureBased<nsd,
    nen>::extract_element_and_node_values_of_porofluid(Core::Elements::Element* element,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& location_array,
    Core::LinAlg::Matrix<nsd, nen>& node_coordinates)
{
  // access from outside to the variable manager:
  // scatra-discretization has fluid-discretization on dofset 2
  variable_manager_->extract_element_and_node_values(
      *element, discretization, location_array, node_coordinates, nds_scatra_porofluid_);
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/

// 1D elements
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::quad4>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::quad9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::hex8>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::tet10>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ScaTraEleCalcPorofluidPressureBased<Core::FE::CellType::nurbs9>;

FOUR_C_NAMESPACE_CLOSE
