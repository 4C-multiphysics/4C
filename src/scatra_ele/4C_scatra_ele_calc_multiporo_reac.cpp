// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_multiporo_reac.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
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
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::ScaTraEleCalcMultiPoroReac(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcPoro<distype>::ScaTraEleCalcPoro(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcAdvReac<distype>::ScaTraEleCalcAdvReac(
          numdofpernode, numscal, disname),
      Discret::Elements::ScaTraEleCalcPoroReac<distype>::ScaTraEleCalcPoroReac(
          numdofpernode, numscal, disname),
      efluxnp_(0)
{
  // replace internal variable manager by internal variable manager for muliporo
  my::scatravarmanager_ =
      std::make_shared<ScaTraEleInternalVariableManagerMultiPoro<nsd_, nen_>>(my::numscal_);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>*
Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcMultiPoroReac<distype>>(
            new ScaTraEleCalcMultiPoroReac<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 | setup element evaluation                                 vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::setup_calc(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization)
{
  pororeac::setup_calc(ele, discretization);

  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  // set the fluid material in the element
  var_manager()->set_fluid_poromultiphase_material(ele);

  if (material->material_type() == Core::Materials::m_matlist or
      material->material_type() == Core::Materials::m_matlist_reactions)
  {
    const std::shared_ptr<const Mat::MatList>& actmat =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (actmat->num_mat() < my::numdofpernode_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numdofpernode_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      switch (singlemat->material_type())
      {
        case Core::Materials::m_scatra_multiporo_fluid:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(singlemat);

          // smaller zero or greater equal numfluidphases
          if (poromat->phase_id() < 0 or
              poromat->phase_id() >= var_manager()->multiphase_mat()->num_fluid_phases())
            FOUR_C_THROW(
                "Invalid phase ID {} for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)",
                poromat->phase_id(), k);

          const int singlephasematid = var_manager()->multiphase_mat()->mat_id(poromat->phase_id());
          std::shared_ptr<Core::Mat::Material> singlemat =
              var_manager()->multiphase_mat()->material_by_id(singlephasematid);

          if (singlemat->material_type() != Core::Materials::m_fluidporo_singlephase)
            FOUR_C_THROW(
                "Invalid phase ID for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)",
                k);

          var_manager()->set_phase_id_and_species_type(
              k, poromat->phase_id(), Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid);
          // set delta in the variablemanager
          var_manager()->set_delta(poromat->delta(), k);
          // set minimum saturation in the variablemanager
          var_manager()->set_min_val_of_phase(poromat->min_sat(), k);
          // set reacts to external force
          var_manager()->set_reacts_to_force(poromat->reacts_to_external_force(), k);
          // set relative mobility function ID
          var_manager()->set_relative_mobility_function_id(
              poromat->relative_mobility_funct_id(), k);
          break;
        }
        case Core::Materials::m_scatra_multiporo_volfrac:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(singlemat);

          // smaller zero or greater equal numfluidphases
          if (poromat->phase_id() < var_manager()->multiphase_mat()->num_fluid_phases() or
              poromat->phase_id() >= var_manager()->multiphase_mat()->num_fluid_phases() +
                                         var_manager()->multiphase_mat()->num_vol_frac())
            FOUR_C_THROW(
                "Invalid phase ID {} for scalar {} (species in volume fraction = "
                "MAT_scatra_multiporo_volfrac)",
                poromat->phase_id(), k);

          const int singlephasematid = var_manager()->multiphase_mat()->mat_id(poromat->phase_id());
          std::shared_ptr<Core::Mat::Material> singlemat =
              var_manager()->multiphase_mat()->material_by_id(singlephasematid);

          if (singlemat->material_type() != Core::Materials::m_fluidporo_singlevolfrac)
            FOUR_C_THROW(
                "Invalid phase ID for scalar {} (species in volume fraction = "
                "MAT_scatra_multiporo_volfrac)",
                k);

          var_manager()->set_phase_id_and_species_type(
              k, poromat->phase_id(), Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac);
          // set delta in the variablemanager
          var_manager()->set_delta(poromat->delta(), k);
          // set reacts to external force
          var_manager()->set_reacts_to_force(poromat->reacts_to_external_force(), k);
          // set relative mobility function ID
          var_manager()->set_relative_mobility_function_id(
              poromat->relative_mobility_funct_id(), k);

          break;
        }
        case Core::Materials::m_scatra_multiporo_solid:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroSolid>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroSolid>(singlemat);

          // set delta in the variablemanager
          var_manager()->set_delta(poromat->delta(), k);

          // dummy value -1000 for phaseID because species in solid do not have a phaseID
          var_manager()->set_phase_id_and_species_type(
              k, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid);

          break;
        }
        case Core::Materials::m_scatra_multiporo_temperature:
        {
          const std::shared_ptr<const Mat::ScatraMatMultiPoroTemperature>& poromat =
              std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroTemperature>(singlemat);

          // assemble heat capacities of fluid phases, volfracs and solid phase
          // cp order [ <fluid>  <volfrac>  <solid> ]
          std::vector<double> cp;
          std::vector<double> cp_fluid(poromat->cp_fluid());
          std::vector<double> cp_volfrac(poromat->cp_volfrac());

          cp.insert(cp.begin(), cp_fluid.begin(), cp_fluid.end());
          cp.insert(cp.end(), cp_volfrac.begin(), cp_volfrac.end());
          cp.insert(cp.end(), poromat->cp_solid());

          var_manager()->set_heat_capacity(cp);

          // assemble thermal diffusivities of fluid phases, volfracs and solid phase
          // kappa order [ <fluid>  <volfrac>  <solid> ]
          std::vector<double> kappa;
          std::vector<double> kappa_fluid(poromat->kappa_fluid());
          std::vector<double> kappa_volfrac(poromat->kappa_volfrac());

          kappa.insert(kappa.begin(), kappa_fluid.begin(), kappa_fluid.end());
          kappa.insert(kappa.end(), kappa_volfrac.begin(), kappa_volfrac.end());
          kappa.insert(kappa.end(), poromat->kappa_solid());

          var_manager()->set_thermal_diffusivity(kappa);

          // dummy value -1000 for phaseID because temperature does not have a phaseID
          var_manager()->set_phase_id_and_species_type(
              k, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature);

          break;
        }

        default:
        {
          FOUR_C_THROW(
              "Material type {} is not supported for multiphase flow through porous media!",
              singlemat->material_type());
          break;
        }
      }
    }
  }
  else
  {
    switch (material->material_type())
    {
      case Core::Materials::m_scatra_multiporo_fluid:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& poromat =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(material);

        // smaller zero or greater equal numfluidphases
        if (poromat->phase_id() < 0 or
            poromat->phase_id() >= var_manager()->multiphase_mat()->num_fluid_phases())
          FOUR_C_THROW(
              "Invalid phase ID {} for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)",
              poromat->phase_id(), 0);

        const int singlephasematid = var_manager()->multiphase_mat()->mat_id(poromat->phase_id());
        std::shared_ptr<Core::Mat::Material> singlemat =
            var_manager()->multiphase_mat()->material_by_id(singlephasematid);

        if (singlemat->material_type() != Core::Materials::m_fluidporo_singlephase)
          FOUR_C_THROW(
              "Invalid phase ID for scalar {} (species in fluid = MAT_scatra_multiporo_fluid)", 0);

        var_manager()->set_phase_id_and_species_type(
            0, poromat->phase_id(), Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid);
        // set delta in the variablemanager
        var_manager()->set_delta(poromat->delta(), 0);
        // set minimum saturation in the variablemanager
        var_manager()->set_min_val_of_phase(poromat->min_sat(), 0);
        // set reacts to external force
        var_manager()->set_reacts_to_force(poromat->reacts_to_external_force(), 0);
        // set relative mobility function ID
        var_manager()->set_relative_mobility_function_id(poromat->relative_mobility_funct_id(), 0);
        break;
      }
      case Core::Materials::m_scatra_multiporo_volfrac:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& poromat =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(material);

        // smaller zero or greater equal numfluidphases
        if (poromat->phase_id() < 0 or
            poromat->phase_id() >= var_manager()->multiphase_mat()->num_fluid_phases() +
                                       var_manager()->multiphase_mat()->num_vol_frac())
          FOUR_C_THROW(
              "Invalid phase ID {} for scalar {} (species in volume fraction = "
              "MAT_scatra_multiporo_volfrac)",
              poromat->phase_id(), 0);

        const int singlephasematid = var_manager()->multiphase_mat()->mat_id(poromat->phase_id());
        std::shared_ptr<Core::Mat::Material> singlemat =
            var_manager()->multiphase_mat()->material_by_id(singlephasematid);

        if (singlemat->material_type() != Core::Materials::m_fluidporo_singlevolfrac)
          FOUR_C_THROW(
              "Invalid phase ID for scalar {} (species in volume fraction = "
              "MAT_scatra_multiporo_volfrac)",
              0);

        var_manager()->set_phase_id_and_species_type(
            0, poromat->phase_id(), Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac);
        // set delta in the variablemanager
        var_manager()->set_delta(poromat->delta(), 0);
        // set reacts to external force
        var_manager()->set_reacts_to_force(poromat->reacts_to_external_force(), 0);
        // set relative mobility function ID
        var_manager()->set_relative_mobility_function_id(poromat->relative_mobility_funct_id(), 0);

        break;
      }
      case Core::Materials::m_scatra_multiporo_solid:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroSolid>& poromat =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroSolid>(material);

        // set delta in the variablemanager
        var_manager()->set_delta(poromat->delta(), 0);

        // dummy value -1000 for phaseID because species in solid do not have a phaseID
        var_manager()->set_phase_id_and_species_type(
            0, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid);

        break;
      }
      case Core::Materials::m_scatra_multiporo_temperature:
      {
        const std::shared_ptr<const Mat::ScatraMatMultiPoroTemperature>& poromat =
            std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroTemperature>(material);

        // assemble heat capacities of fluid phases, volfracs and solid phase
        std::vector<double> cp;
        std::vector<double> cp_fluid(poromat->cp_fluid());
        std::vector<double> cp_volfrac(poromat->cp_volfrac());

        cp.insert(cp.begin(), cp_fluid.begin(), cp_fluid.end());
        cp.insert(cp.end(), cp_volfrac.begin(), cp_volfrac.end());
        cp.insert(cp.end(), poromat->cp_solid());

        var_manager()->set_heat_capacity(cp);

        // assemble thermal diffusivities of fluid phases, volfracs and solid phase
        std::vector<double> kappa;
        std::vector<double> kappa_fluid(poromat->kappa_fluid());
        std::vector<double> kappa_volfrac(poromat->kappa_volfrac());

        kappa.insert(kappa.begin(), kappa_fluid.begin(), kappa_fluid.end());
        kappa.insert(kappa.end(), kappa_volfrac.begin(), kappa_volfrac.end());
        kappa.insert(kappa.end(), poromat->kappa_solid());

        var_manager()->set_thermal_diffusivity(kappa);

        // dummy value -1000 for phaseID because temperature does not have a phaseID
        var_manager()->set_phase_id_and_species_type(
            0, -1000, Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature);

        break;
      }

      default:
      {
        FOUR_C_THROW("Material type {} is not supported for multiphase flow through porous media!",
            material->material_type());
        break;
      }
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                   vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::extract_element_and_node_values(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // extract action parameter
  const auto action = Teuchos::getIntegralValue<ScaTra::Action>(params, "action");
  var_manager()->set_action(action);

  //---------------------------------------------------------------------------------------------
  //                                 STRUCTURE
  //---------------------------------------------------------------------------------------------

  // get additional state vector for ALE case: grid displacement
  if (my::scatrapara_->is_ale())
  {
    // get number of dofset associated with displacement related dofs
    const int ndsdisp = my::scatrapara_->nds_disp();

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discretization.get_state(ndsdisp, "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp(nsd_ * nen_, -1);
    for (unsigned inode = 0; inode < nen_; ++inode)
      for (unsigned idim = 0; idim < nsd_; ++idim)
        lmdisp[inode * nsd_ + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*dispnp, my::edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    my::update_node_coordinates();
  }
  else
  {
    my::edispnp_.clear();
  }

  //---------------------------------------------------------------------------------------------
  //                                 SCATRA
  //---------------------------------------------------------------------------------------------

  // extract local values from the global vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> hist = discretization.get_state("hist");
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (hist == nullptr || phinp == nullptr)
    FOUR_C_THROW("Cannot get state vector 'hist' and/or 'phinp'");

  // values of scatra field are always in first dofset
  const std::vector<int>& lm = la[0].lm_;
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*hist, my::ehist_, lm);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, my::ephinp_, lm);

  if (my::scatraparatimint_->is_gen_alpha() and not my::scatraparatimint_->is_incremental())
  {
    // extract additional local values from global vector
    std::shared_ptr<const Core::LinAlg::Vector<double>> phin = discretization.get_state("phin");
    if (phin == nullptr) FOUR_C_THROW("Cannot get state vector 'phin'");
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phin, my::ephin_, lm);
  }

  if (my::scatrapara_->has_external_force())
  {
    // get number of dofset associated with velocity related dofs
    const auto ndsvel = my::scatrapara_->nds_vel();
    const auto force_velocity = discretization.get_state(ndsvel, "force_velocity");

    const int number_dof_per_node = la[ndsvel].lm_.size() / my::nen_;
    std::vector<int> location_vector(my::nsd_ * my::nen_, -1);
    for (unsigned inode = 0; inode < my::nen_; ++inode)
    {
      for (unsigned idim = 0; idim < my::nsd_; ++idim)
        location_vector[inode * my::nsd_ + idim] =
            la[ndsvel].lm_[inode * number_dof_per_node + idim];
    }

    Core::FE::extract_my_values<Core::LinAlg::Matrix<my::nsd_, my::nen_>>(
        *force_velocity, my::eforcevelocity_, location_vector);
  }

  //---------------------------------------------------------------------------------------------
  //                                 FLUID
  //---------------------------------------------------------------------------------------------

  // get number of dofset associated with pressure/fluid related dofs
  const int ndspres = my::scatrapara_->nds_pres();

  // determine number of velocity related dofs per node (= number of phases)
  const int numfluidphases = var_manager()->multiphase_mat()->num_fluid_phases();
  const int totalnummultiphasedofpernode = var_manager()->multiphase_mat()->num_mat();

  // extract element and node values of the porofluid
  if (discretization.has_state(ndspres, "phinp_fluid"))
  {
    var_manager()->setup_poro_fluid_managers(
        ele, params, discretization, la, numfluidphases, totalnummultiphasedofpernode);
    var_manager()->extract_element_and_node_values_of_poro_fluid(
        ele, discretization, la, my::xyze_);
    L2_projection_ = params.get<bool>("L2-projection");
    // extract the nodal flux
    if (L2_projection_)
    {
      extract_nodal_flux(ele, params, discretization, la, numfluidphases);
    }
  }
  else
    FOUR_C_THROW("Something went wrong here, scatra-dis does not have fluid primary variable");

  // ---------------------------------------------------------------------
  // call routine for calculation of body force in element nodes
  // (time n+alpha_F for generalized-alpha scheme, at time n+1 otherwise)
  // ---------------------------------------------------------------------
  my::body_force(ele);
  //--------------------------------------------------------------------------------
  // further node-based source terms not given via Neumann volume condition
  // i.e., set special body force for homogeneous isotropic turbulence
  //--------------------------------------------------------------------------------
  my::other_node_based_source_terms(lm, discretization, params);

  return;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values (L2-projection)    vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::extract_nodal_flux(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    const int numfluidphases)
{
  // resize state vectors based on number of phases
  efluxnp_.resize(numfluidphases);

  // get number of dofset associated with velocity related dofs
  const int ndsvel = my::scatrapara_->nds_vel();

  std::string stateprefix = "flux";
  for (int curphase = 0; curphase < numfluidphases; curphase++)
  {
    std::stringstream statename;
    statename << stateprefix << curphase;

    // get convective (velocity - mesh displacement) velocity at nodes
    std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
        discretization.get_state(ndsvel, statename.str());
    if (convel == nullptr) FOUR_C_THROW("Cannot get state vector {}", statename.str().c_str());

    // extract local values of convective velocity field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(
        *convel, efluxnp_[curphase], la[ndsvel].lm_);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  compute the solid pressure at gauss point  (protected)    vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::compute_pore_pressure()
{
  return var_manager()->solid_pressure();
}

/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                   vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::materials(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point

)
{
  switch (material->material_type())
  {
    case Core::Materials::m_scatra_multiporo_fluid:
    {
      mat_multi_poro_fluid(material, k, densn, densnp, densam, visc, iquad);
      break;
    }
    case Core::Materials::m_scatra_multiporo_volfrac:
    {
      mat_multi_poro_vol_frac(material, k, densn, densnp, densam, visc, iquad);
      break;
    }

    case Core::Materials::m_scatra_multiporo_solid:
    {
      mat_multi_poro_solid(material, k, densn, densnp, densam, visc, iquad);
      break;
    }

    case Core::Materials::m_scatra_multiporo_temperature:
    {
      mat_multi_poro_temperature(material, k, densn, densnp, densam, visc, iquad);
      break;
    }

    default:
    {
      FOUR_C_THROW("Material type {} is not supported for multiphase flow through porous media!",
          material->material_type());
      break;
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |                                                           vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::mat_multi_poro_fluid(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  if (iquad == -1)
    FOUR_C_THROW(
        "no gauss point given for evaluation of MatMultiPoro material. Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroFluid>& actmat =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroFluid>(material);

  // volume fraction of fluid phase: volfrac_fluid = porosity * saturation_fluid
  double volfrac_fluid = 0.0;
  // d_eff = d_0 * (porosity * saturation(k))^delta
  double d_eff = 0.0;

  if (var_manager()->evaluate_scalar(k))
  {
    volfrac_fluid = var_manager()->fluid_phase_manager()->porosity() * var_manager()->saturation(k);
    d_eff =
        std::pow(var_manager()->fluid_phase_manager()->porosity() * var_manager()->saturation(k),
            actmat->delta());
  }

  {
    // set diffusivity (scaled with volfrac_fluid)
    poro::set_diffusivity(actmat, k, volfrac_fluid * d_eff);

    // set densities (scaled with volfrac_fluid)
    poro::set_densities(volfrac_fluid, densn, densnp, densam);
  }

  return;
}  // ScaTraEleCalcMultiPoroReac<distype>::mat_multi_poro_fluid


/*----------------------------------------------------------------------*
 |                                                     kremheller 02/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::mat_multi_poro_vol_frac(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  if (iquad == -1)
    FOUR_C_THROW(
        "no gauss point given for evaluation of MatMultiPoro material. Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroVolFrac>& actmat =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroVolFrac>(material);

  // volume fraction
  double volfrac = 0.0;
  // d_eff = d_0 * (porosity * saturation(k))^delta
  double d_eff = 0.0;

  if (var_manager()->evaluate_scalar(k))
  {
    volfrac = var_manager()->vol_frac(k);
    d_eff = std::pow(var_manager()->vol_frac(k), actmat->delta());
  }

  {
    // set diffusivity (scaled with volfrac)
    poro::set_diffusivity(actmat, k, volfrac * d_eff);

    // set densities (scaled with volfrac)
    poro::set_densities(volfrac, densn, densnp, densam);
  }

  return;
}  // ScaTraEleCalcMultiPoroReac<distype>::MatMultiPoro

/*----------------------------------------------------------------------*
 | Species in solid                                        wirthl 12/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::mat_multi_poro_solid(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  if (iquad == -1)
    FOUR_C_THROW(
        "no gauss point given for evaluation of MatMultiPoro material. "
        "Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroSolid>& actmat =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroSolid>(material);

  // volume fraction of solid phase: volfrac_solid_phase = (1 - porosity - sumaddvolfrac)
  double volfrac_solid_phase = 0.0;
  // d_eff = d_0 * (porosity * saturation(k))^delta
  double d_eff = 0.0;

  if (var_manager()->evaluate_scalar(k))
  {
    volfrac_solid_phase = (1 - var_manager()->fluid_phase_manager()->porosity() -
                           var_manager()->fluid_phase_manager()->sum_add_vol_frac());
    d_eff = std::pow((1 - var_manager()->fluid_phase_manager()->porosity() -
                         var_manager()->fluid_phase_manager()->sum_add_vol_frac()),
        actmat->delta());
  }

  {
    // set diffusivity (scaled with volfrac_solid_phase)
    poro::set_diffusivity(actmat, k, volfrac_solid_phase * d_eff);

    // set densities (scaled with volfrac_solid_phase)
    poro::set_densities(volfrac_solid_phase, densn, densnp, densam);
  }

  return;
}  // ScaTraEleCalcMultiPoroReac<distype>::MatMultiPoro

/*----------------------------------------------------------------------*
 | Species temperature                                     wirthl 12/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::mat_multi_poro_temperature(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    double& densn,                                              //!< density at t_(n)
    double& densnp,  //!< density at t_(n+1) or t_(n+alpha_F)
    double& densam,  //!< density at t_(n+alpha_M)
    double& visc,    //!< fluid viscosity
    const int iquad  //!< id of current gauss point
)
{
  if (iquad == -1)
    FOUR_C_THROW(
        "no gauss point given for evaluation of MatMultiPoro material. "
        "Check your input file.");

  const std::shared_ptr<const Mat::ScatraMatMultiPoroTemperature>& actmat =
      std::dynamic_pointer_cast<const Mat::ScatraMatMultiPoroTemperature>(material);

  // read the heat capacity
  double cp_eff = 0.0;
  // kappa_eff = kappa_s*poro_s + kappa_fluids*poro*saturation_fluids + kappa_volfrac*poro_volfrac
  double kappa_eff = 0.0;

  if (var_manager()->evaluate_scalar(k))
  {
    const int numfluidphases = var_manager()->fluid_phase_manager()->num_fluid_phases();
    const int numvolfracs = var_manager()->fluid_phase_manager()->num_vol_frac();

    cp_eff = (1 - var_manager()->fluid_phase_manager()->porosity() -
                 var_manager()->fluid_phase_manager()->sum_add_vol_frac()) *
             var_manager()->fluid_phase_manager()->solid_density() * actmat->cp_solid();

    kappa_eff = (1 - var_manager()->fluid_phase_manager()->porosity() -
                    var_manager()->fluid_phase_manager()->sum_add_vol_frac()) *
                actmat->kappa_solid();

    for (int phase = 0; phase < numfluidphases; ++phase)
    {
      cp_eff += actmat->cp_fluid(phase) * var_manager()->fluid_phase_manager()->porosity() *
                var_manager()->fluid_phase_manager()->saturation(phase) *
                var_manager()->fluid_phase_manager()->density(phase);

      kappa_eff += actmat->kappa_fluid(phase) * var_manager()->fluid_phase_manager()->porosity() *
                   var_manager()->fluid_phase_manager()->saturation(phase);
    }

    for (int phase = 0; phase < numvolfracs; ++phase)
    {
      cp_eff += actmat->cp_volfrac(phase) * var_manager()->fluid_phase_manager()->vol_frac(phase) *
                var_manager()->fluid_phase_manager()->vol_frac_density(phase);

      kappa_eff +=
          actmat->kappa_volfrac(phase) * var_manager()->fluid_phase_manager()->vol_frac(phase);
    }
  }

  {
    var_manager()->set_effective_heat_capacity(cp_eff);

    // set diffusivity
    poro::set_diffusivity(actmat, k, kappa_eff);

    // set densities
    poro::set_densities(cp_eff, densn, densnp, densam);
  }

  return;
}  // ScaTraEleCalcMultiPoroReac<distype>::MatMultiPoro

/*------------------------------------------------------------------------------*
 | set internal variables                                           vuong 08/16 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<
    distype>::set_internal_variables_for_mat_and_rhs()
{
  var_manager()->set_internal_variables_multi_poro(my::funct_, my::derxy_, my::deriv_, my::xjm_,
      pororeac::xyze0_, my::ephinp_, my::ephin_, my::ehist_, my::eforcevelocity_);

  if (L2_projection_)
  {
    var_manager()->adapt_convective_term_for_l2(my::funct_, my::derxy_, efluxnp_);
  }

  return;
}

/*-------------------------------------------------------------------------------*
 |  Set advanced reaction terms and derivatives                      vuong 08/16 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::set_advanced_reaction_terms(
    const int k,                                               //!< index of current scalar
    const std::shared_ptr<Mat::MatListReactions> matreaclist,  //!< index of current scalar
    const double* gpcoord                                      //!< current Gauss-point coordinates
)
{
  const std::shared_ptr<ScaTraEleReaManagerAdvReac> remanager = advreac::rea_manager();

  fill_coupling_vector_and_add_variables(k, *matreaclist, *remanager);

  const ScaTra::Action act = var_manager()->get_action();

  auto time = my::scatraparatimint_->time();
  // note: we always need the reaction term to calculate rhsint, which is needed also for OD-terms
  remanager->add_to_rea_body_force(
      matreaclist->calc_rea_body_force_term(
          k, my::scatravarmanager_->phinp(), couplingvalues_, gpcoord, time),
      k);

  std::vector<std::pair<std::string, double>> emptyconstants;

  switch (act)
  {
    case ScaTra::Action::calc_mat_and_rhs:
    case ScaTra::Action::calc_initial_time_deriv:
    {
      matreaclist->calc_rea_body_force_deriv_matrix(k,
          remanager->get_rea_body_force_deriv_vector(k), my::scatravarmanager_->phinp(),
          couplingvalues_, gpcoord, time);

      break;
    }
    case ScaTra::Action::calc_scatra_mono_odblock_fluid:
    {
      matreaclist->calc_rea_body_force_deriv_matrix_add_variables(k,
          remanager->get_rea_body_force_deriv_vector_add_variables(k),
          my::scatravarmanager_->phinp(), couplingvalues_, emptyconstants, gpcoord, time);

      break;
    }
    case ScaTra::Action::calc_scatra_mono_odblock_mesh:
    {
      if (var_manager()->fluid_phase_manager()->porosity_depends_on_struct())
      {
        matreaclist->calc_rea_body_force_deriv_matrix_add_variables(k,
            remanager->get_rea_body_force_deriv_vector_add_variables(k),
            my::scatravarmanager_->phinp(), couplingvalues_, emptyconstants, gpcoord, time);
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("Wrong action type in var_manager(), action type is {}", act);
    }
  }
}

/*-------------------------------------------------------------------------------*
 |  Get right hand side including reaction bodyforce term       kremheller 02/18 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::get_rhs_int(
    double& rhsint,       //!< rhs containing bodyforce at Gauss point
    const double densnp,  //!< density at t_(n+1)
    const int k           //!< index of current scalar
)
{
  // only difference is the inverse scaling with density
  advreac::get_rhs_int(rhsint, 1.0 / var_manager()->density(k), k);
}

/*------------------------------------------------------------------------------ *
 | calculation of reactive element matrix for coupled reactions kremheller 02/18 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_mat_react(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac,
    const double timetaufac, const double taufac, const double densnp,
    const Core::LinAlg::Matrix<nen_, 1>& sgconv, const Core::LinAlg::Matrix<nen_, 1>& diff)
{
  // only difference is the inverse scaling with density
  advreac::calc_mat_react(
      emat, k, timefacfac, timetaufac, taufac, 1.0 / var_manager()->density(k), sgconv, diff);
}

/*-------------------------------------------------------------------------------*
 |  fill the coupling vector and add variables to reactions          vuong 08/16 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::fill_coupling_vector_and_add_variables(
    const int k, Mat::MatListReactions& matreaclist, ScaTraEleReaManagerAdvReac& remanager)
{
  // if it is empty rebuilt it
  if (couplingvalues_.empty())
  {
    // pressures
    const std::vector<double>& pressures = var_manager()->pressure();
    const int numfluidphases = var_manager()->multiphase_mat()->num_fluid_phases();
    for (int i = 0; i < numfluidphases; i++)
    {
      std::ostringstream temp;
      temp << i + 1;
      couplingvalues_.push_back(std::pair<std::string, double>("p" + temp.str(), pressures[i]));
    }
    // saturation
    const std::vector<double>& saturations = var_manager()->saturation();
    for (int i = 0; i < numfluidphases; i++)
    {
      std::ostringstream temp;
      temp << i + 1;

      couplingvalues_.push_back(std::pair<std::string, double>("S" + temp.str(), saturations[i]));
    }
    // porosity
    couplingvalues_.push_back(std::pair<std::string, double>(
        "porosity", var_manager()->fluid_phase_manager()->porosity()));
    // additional volume fractions
    const std::vector<double>& volfracs = var_manager()->vol_frac();
    const int numvolfrac = var_manager()->fluid_phase_manager()->num_vol_frac();
    for (int i = 0; i < numvolfrac; i++)
    {
      std::ostringstream temp;
      temp << i + 1;

      couplingvalues_.push_back(std::pair<std::string, double>("VF" + temp.str(), volfracs[i]));
    }
    // additional volume fraction pressures
    const std::vector<double>& volfracpressures = var_manager()->vol_frac_pressure();
    for (int i = 0; i < numvolfrac; i++)
    {
      std::ostringstream temp;
      temp << i + 1;

      couplingvalues_.push_back(
          std::pair<std::string, double>("VFP" + temp.str(), volfracpressures[i]));
    }

    // initialize and add the variables to the reaction manager --> has to be done only once
    remanager.initialize_rea_body_force_deriv_vector_add_variables(
        my::numdofpernode_, couplingvalues_.size());
    // error will be thrown if reaction != by-reaction coupling is chosen
    for (int j = 0; j < my::numdofpernode_; j++)
      matreaclist.add_additional_variables(j, couplingvalues_);
  }
  // directly copy values (rely on order for performance reasons)
  else
  {
    // pressures
    const std::vector<double>& pressures = var_manager()->pressure();
    const int numfluidphases = var_manager()->multiphase_mat()->num_fluid_phases();
    for (int i = 0; i < numfluidphases; i++)
    {
      // std::cout<<"pressure "<<i<<": "<<pressures[i]<<std::endl;
      couplingvalues_[i].second = pressures[i];
    }
    // saturation
    const std::vector<double>& saturations = var_manager()->saturation();
    for (int i = 0; i < numfluidphases; i++)
    {
      //   std::cout<<"saturation "<<i<<": "<<saturations[i]<<std::endl;
      couplingvalues_[numfluidphases + i].second = saturations[i];
    }
    // porosity
    couplingvalues_[2 * numfluidphases].second = var_manager()->fluid_phase_manager()->porosity();
    // additional volume fractions
    const std::vector<double>& volfracs = var_manager()->vol_frac();
    const std::vector<double>& volfracpressures = var_manager()->vol_frac_pressure();
    const int numvolfrac = var_manager()->fluid_phase_manager()->num_vol_frac();
    for (int i = 0; i < numvolfrac; i++)
    {
      couplingvalues_[2 * numfluidphases + 1 + i].second = volfracs[i];
      couplingvalues_[2 * numfluidphases + numvolfrac + 1 + i].second = volfracpressures[i];
    }
  }
}

/*-----------------------------------------------------------------------------*
 |  calculation of convective element matrix in convective form    vuong 08/16 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_mat_conv(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac,
    const double densnp, const Core::LinAlg::Matrix<nen_, 1>& sgconv)
{
  // case of zero saturation/volfrac
  // no convective term for species in solid
  if (var_manager()->evaluate_scalar(k) &&
      var_manager()->get_species_type(k) != Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
  {
    // the only difference to the base class version is, that there is no scaling with the density
    pororeac::calc_mat_conv(emat, k, timefacfac, 1.0, sgconv);
  }

  return;
}  // ScaTraEleCalc<distype>::CalcMatConv

/*-----------------------------------------------------------------------------*
 |  calculation of convective element matrix in convective form    vuong 08/16 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_mat_mass(
    Core::LinAlg::SerialDenseMatrix& emat, const int& k, const double& fac, const double& densam)
{
  if (var_manager()->evaluate_scalar(k))
  {
    // the only difference to the base class version is, that there is no scaling with the density
    pororeac::calc_mat_mass(emat, k, fac, densam);
  }
  else
  {
    if (var_manager()->get_species_type(k) ==
        Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid)
    {
      // If we have zero "densities" (porosity*saturation(k)), which mostly happens for tumor
      // cells, the whole equation will be equal to zero since it is scaled with the density
      // In that case also the mass fraction of the species (necrotic tumor cells) has to be zero
      // --> here we explicitly force it to be zero through a "Dirichlet" boundary condition
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * my::numdofpernode_ + k;
        emat(fvi, fvi) += penalty_;
      }
    }
  }
  return;
}  // ScaTraEleCalc<distype>::CalcMatConv


/*------------------------------------------------------------------------------------------*
 |  calculation of convective element matrix: add conservative contributions   vuong 08/16 |
 *------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_mat_conv_add_cons(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double timefacfac, const double vdiv,
    const double densnp)
{
  // the only difference to the base class version is, that there is no scaling with the density
  pororeac::calc_mat_conv_add_cons(emat, k, timefacfac, vdiv, 1.0);

  return;
}

/*------------------------------------------------------------------- *
 | adaption of convective term for rhs                     vuong 08/16 |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::recompute_conv_phi_for_rhs(const int k,
    const Core::LinAlg::Matrix<nsd_, 1>& sgvelint, const double densnp, const double densn,
    const double vdiv)
{
  // the only difference to the base class version is, that there is no scaling with the density
  pororeac::recompute_conv_phi_for_rhs(k, sgvelint, 1.0, 1.0, vdiv);
  return;
}

/*-------------------------------------------------------------------- *
 |  standard Galerkin convective term (OD mesh)       kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_conv_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh, const double fac,
    const double rhsfac, const double densnp, const double J,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi, const Core::LinAlg::Matrix<nsd_, 1>& convelint)
{
  // case of zero saturation/volfrac
  // no convective term for species in solid
  if (var_manager()->evaluate_scalar(k) &&
      var_manager()->get_species_type(k) != Mat::ScaTraMatMultiPoro::SpeciesType::species_in_solid)
  {
    static Core::LinAlg::Matrix<nsd_, nsd_> difftensor(Core::LinAlg::Initialization::set_zero);
    static Core::LinAlg::Matrix<nsd_, 1> refgradpres(Core::LinAlg::Initialization::set_zero);

    // linearization of mesh motion
    //--------------------------------------dJ/dd = dJ/dF : dF/dd = J * F^-T . N_{\psi} = J * N_x
    // J denotes the determinant of the Jacobian of the mapping between current and parameter space,
    // i.e. det(dx/ds) in our case: rhsfac = J * dt * theta --> d(rhsfac)/dd = rhsfac * N_x
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * my::numdofpernode_ + k;
      const double v = rhsfac * my::funct_(vi) * (-1.0) * var_manager()->conv_phi(k);

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        for (unsigned idim = 0; idim < nsd_; ++idim)
        {
          const int fui = ui * nsd_ + idim;
          emat(fvi, fui) += v * my::derxy_(idim, ui);
        }
      }
    }

    if (var_manager()->get_species_type(k) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid ||
        var_manager()->get_species_type(k) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac)
    {
      var_manager()->get_diff_tensor_fluid(k, difftensor, var_manager()->get_phase_id(k));
      var_manager()->get_ref_grad_pres(k, my::xjm_, refgradpres, var_manager()->get_phase_id(k));
      const double vrhs = rhsfac * 1.0 / J * difftensor(0, 0) * (-1.0);

      // linearization of prassure gradient
      // standard Galerkin terms  -- "shapederivatives" pressure gradient
      apply_shape_derivs_pressure_grad(emat, k, vrhs, gradphi, refgradpres);

      // linearization of gradphi
      // standard Galerkin terms  -- "shapederivatives" gradphi
      pororeac::apply_shape_derivs_conv(emat, k, rhsfac, 1.0, J, gradphi, convelint);
    }

    else if (var_manager()->get_species_type(k) ==
             Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature)
    {
      const int numfluidphases = var_manager()->fluid_phase_manager()->num_fluid_phases();
      const int numvolfracs = var_manager()->fluid_phase_manager()->num_vol_frac();

      for (int phase = 0; phase < numfluidphases; ++phase)
      {
        var_manager()->get_diff_tensor_fluid(k, difftensor, phase);
        var_manager()->get_ref_grad_pres(k, my::xjm_, refgradpres, phase);

        const double vrhs = rhsfac * 1.0 / J * difftensor(0, 0) * (-1.0) *
                            var_manager()->fluid_phase_manager()->density(phase) *
                            var_manager()->get_heat_capacity(phase);

        // linearization of prassure gradient
        // standard Galerkin terms  -- "shapederivatives" pressure gradient
        apply_shape_derivs_pressure_grad(emat, k, vrhs, gradphi, refgradpres);
      }

      for (int phase = numfluidphases; phase < numfluidphases + numvolfracs; ++phase)
      {
        var_manager()->get_diff_tensor_fluid(k, difftensor, phase);
        var_manager()->get_ref_grad_pres(k, my::xjm_, refgradpres, phase);

        const double vrhs =
            rhsfac * 1.0 / J * difftensor(0, 0) * (-1.0) *
            var_manager()->fluid_phase_manager()->vol_frac_density(phase - numfluidphases) *
            var_manager()->get_heat_capacity(phase);

        // linearization of prassure gradient
        // standard Galerkin terms  -- "shapederivatives" pressure gradient
        apply_shape_derivs_pressure_grad(emat, k, vrhs, gradphi, refgradpres);
      }

      // linearization of gradphi
      // standard Galerkin terms  -- "shapederivatives" gradphi
      pororeac::apply_shape_derivs_conv(emat, k, rhsfac, 1.0, J, gradphi, convelint);
    }
    else
      FOUR_C_THROW("Species type no valid!");
  }
  return;
}

/*-------------------------------------------------------------------- *
 |  standard Galerkin temporal term (OD mesh)         kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_lin_mass_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double rhsfac, const double fac, const double densam, const double densnp,
    const double phinp, const double hist, const double J,
    const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    // get pre-factor for this scalar
    const double myfac = var_manager()->get_pre_factor_for_mass_matrix_od_mesh(k, fac);

    // call base class
    my::calc_lin_mass_od_mesh(
        emat, k, ndofpernodemesh, rhsfac, myfac, densam, densnp, phinp, hist, J, dJ_dmesh);
  }

  return;
}

/*-------------------------------------------------------------------- *
 | hist and source term (OD mesh)                     kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_hist_and_source_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh, const double fac,
    const double rhsint, const double J, const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh,
    const double densnp)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    // const int curphase = var_manager()->GetPhaseID(k);
    const int numfluidphases = var_manager()->fluid_phase_manager()->num_fluid_phases();

    // get pre-factor for this scalar
    const double myfac = var_manager()->get_pre_factor_for_hist_and_source_od_mesh(
        k, fac, densnp, my::scatravarmanager_->hist(k), rhsint);

    // 1) linearization of mesh motion:
    //    call base class with correct factor
    my::calc_hist_and_source_od_mesh(emat, k, ndofpernodemesh, myfac, 1.0, J, dJ_dmesh, densnp);

    // 2) linearization of advanced reaction terms
    const std::shared_ptr<ScaTraEleReaManagerAdvReac> remanager = advreac::rea_manager();
    if (remanager->active() && var_manager()->fluid_phase_manager()->porosity_depends_on_struct())
    {
      const std::vector<double> myderivs =
          remanager->get_rea_body_force_deriv_vector_add_variables(k);

      // porosity deriv at [2* numfluidphases]: d reac / d d = d reac / d poro * d poro / d d
      // with
      // dporo/dd = dporo/dJ * dJ/dd = dporosity/dJ * J * N_x
      // J denotes the determinant of the deformation gradient, i.e. det F = det ( d x / d X ) = det
      // (dx/ds) * ( det(dX/ds) )^-1

      const double poroderiv =
          myderivs[2 * numfluidphases] * var_manager()->fluid_phase_manager()->jacobian_def_grad() *
          var_manager()->fluid_phase_manager()->porosity_deriv_wrt_jacobian_def_grad();

      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * my::numdofpernode_ + k;
        // TODO: gen-alpha
        const double v = my::funct_(vi) * poroderiv * my::scatraparatimint_->time_fac() * fac *
                         (-1.0) / var_manager()->density(k);

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)
          {
            const int fui = ui * nsd_ + idim;

            emat(fvi, fui) += v * my::derxy_(idim, ui);
          }
        }
      }
    }
  }

  return;
}

/*-------------------------------------------------------------------- *
 | diffusive term (OD mesh)                           kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_diff_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double diffcoeff, const double fac, const double rhsfac, const double J,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi, const Core::LinAlg::Matrix<nsd_, 1>& convelint,
    const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    // call base class
    my::calc_diff_od_mesh(
        emat, k, ndofpernodemesh, diffcoeff, fac, rhsfac, J, gradphi, convelint, dJ_dmesh);

    // get pre-factor for this scalar
    const double myfac =
        var_manager()->get_pre_factor_for_diff_matrix_od_mesh(k, rhsfac, diffcoeff);

    if (fabs(myfac) > 1.0e-12)
    {
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * my::numdofpernode_ + k;

        double laplawf(0.0);
        my::get_laplacian_weak_form_rhs(laplawf, gradphi, vi);
        const double v = myfac * laplawf;
        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          for (unsigned idim = 0; idim < nsd_; ++idim)
          {
            const int fui = ui * nsd_ + idim;

            emat(fvi, fui) += v * my::derxy_(idim, ui);
          }
        }
      }
    }
  }
  return;
}

/*-------------------------------------------------------------------- *
 | reactive term (OD mesh)                            kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_react_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double rhsfac, const double rea_phi, const double J,
    const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    // get pre-factor for this scalar
    const double myfac = var_manager()->get_pre_factor_for_mass_matrix_od_mesh(k, rhsfac);
    // call base class
    my::calc_react_od_mesh(emat, k, ndofpernodemesh, myfac, rea_phi, J, dJ_dmesh);
  }

  return;
}

/*-------------------------------------------------------------------- *
 | convective term (OD fluid)                         kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_mat_conv_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                            //!< index of current scalar
    const int ndofpernodefluid,             //!< number of dofs per node of fluid element
    const double rhsfac,  //!< domain-integration factor times time-integration factor
    const double densnp,  //!< density at time_(n+1)
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient
)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    if (var_manager()->get_species_type(k) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_fluid ||
        var_manager()->get_species_type(k) ==
            Mat::ScaTraMatMultiPoro::SpeciesType::species_in_volfrac)
    {
      const int totalnummultiphasedofpernode = var_manager()->multiphase_mat()->num_mat();

      static Core::LinAlg::Matrix<nsd_, nsd_> difftensor(Core::LinAlg::Initialization::set_zero);
      var_manager()->get_diff_tensor_fluid(k, difftensor, var_manager()->get_phase_id(k));

      // gradphi^T * difftensor
      // TODO: not sure if this works for anisotropic fluid difftensor
      static Core::LinAlg::Matrix<1, nsd_> gradphiTdifftensor(
          Core::LinAlg::Initialization::set_zero);
      gradphiTdifftensor.multiply_tn(gradphi, difftensor);

      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * my::numdofpernode_ + k;
        const double v = rhsfac * my::funct_(vi) * (-1.0);

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          // get pre-fac vector for this scalar
          static std::vector<double> prefaclinconvodfluid(totalnummultiphasedofpernode, 0.0);
          var_manager()->get_pre_fac_lin_conv_od_fluid(k, ui, &prefaclinconvodfluid, gradphi,
              gradphiTdifftensor, my::funct_, my::derxy_, var_manager()->get_phase_id(k));

          for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
          {
            const int fui = ui * totalnummultiphasedofpernode + idof;
            emat(fvi, fui) += v * prefaclinconvodfluid[idof];
          }
        }
      }

      //----------------------------------------------------------------
      // linearization of dynamic viscosity w.r.t. dof --> TODO
      // however, I believe this is not necessary: FD-check does not fail
      //----------------------------------------------------------------
    }

    else if (var_manager()->get_species_type(k) ==
             Mat::ScaTraMatMultiPoro::SpeciesType::species_temperature)
    {
      const int numfluid = var_manager()->fluid_phase_manager()->num_fluid_phases();
      const int numvolfracs = var_manager()->fluid_phase_manager()->num_vol_frac();

      for (int phase = 0; phase < numfluid + numvolfracs; ++phase)
      {
        const int totalnummultiphasedofpernode = var_manager()->multiphase_mat()->num_mat();

        static Core::LinAlg::Matrix<nsd_, nsd_> difftensor(Core::LinAlg::Initialization::set_zero);
        var_manager()->get_diff_tensor_fluid(k, difftensor, phase);

        // gradphi^T * difftensor
        static Core::LinAlg::Matrix<1, nsd_> gradphiTdifftensor(
            Core::LinAlg::Initialization::set_zero);
        gradphiTdifftensor.multiply_tn(gradphi, difftensor);

        // calculate density*heatcapacity
        double densheatcapacity =
            var_manager()->get_heat_capacity(phase) * var_manager()->density()[phase];

        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * my::numdofpernode_ + k;
          const double v = rhsfac * my::funct_(vi) * (-1.0) * densheatcapacity;

          for (unsigned ui = 0; ui < nen_; ++ui)
          {
            // get pre-fac vector for this scalar
            static std::vector<double> prefaclinconvodfluid(totalnummultiphasedofpernode, 0.0);

            var_manager()->get_pre_fac_lin_conv_od_fluid(k, ui, &prefaclinconvodfluid, gradphi,
                gradphiTdifftensor, my::funct_, my::derxy_, phase);

            for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
            {
              const int fui = ui * totalnummultiphasedofpernode + idof;
              emat(fvi, fui) += v * prefaclinconvodfluid[idof];
            }
          }
        }
      }
    }
  }
  return;
}

/*-------------------------------------------------------------------------- *
 | convective term -- conservative contributions (OD fluid) kremheller 07/17 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_mat_conv_add_cons_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                            //!< index of current scalar
    const int ndofpernodefluid,             //!< number of dofs per node of fluid element
    const double timefacfac,  //!< domain-integration factor times time-integration factor
    const double densnp,      //!< density at time_(n+1)
    const double phinp        //!< scalar at time_(n+1)
)
{
  FOUR_C_THROW("calc_mat_conv_add_cons_od_fluid not yet available for scatre ele calc multiporo");
}

/*-------------------------------------------------------------------- *
 | temporal term (OD fluid)                           kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_lin_mass_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix to be filled
    const int k,                            //!< index of current scalar
    const int
        ndofpernodemesh,  //!< number of dofs per node of fluid element // only a dummy variable
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const double fac,     //!< domain-integration factor
    const double densam,  //!< density at time_(n+am)
    const double densnp,  //!< density at time_(n+1)
    const double phinp,   //!< scalar at time_(n+1)
    const double hist     //!< history of time integartion
)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    const int totalnummultiphasedofpernode = var_manager()->multiphase_mat()->num_mat();

    double vtrans = 0.0;

    if (my::scatraparatimint_->is_gen_alpha())
      FOUR_C_THROW("not implemented");
    else
    {
      // compute scalar at integration point
      vtrans = fac * densnp * phinp;
    }

    // get pre-fac vector for this scalar
    static std::vector<double> prefaclinmassodfluid(totalnummultiphasedofpernode, 0.0);
    var_manager()->get_pre_fac_lin_mass_od_fluid(k, &prefaclinmassodfluid);

    calc_lin_mass_matrix_type_od_fluid(
        emat, k, &prefaclinmassodfluid, totalnummultiphasedofpernode, vtrans);
  }

  return;
}

/*---------------------------------------------------------------------- *
 | generic linearization of mass matrix type (OD fluid) kremheller 03/18 |
 | correct pre-factor has to be passed                                   |
 *-----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_lin_mass_matrix_type_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k,
    const std::vector<double>* prefaclinmassodfluid, const int totalnummultiphasedofpernode,
    double prefac)
{
  for (unsigned vi = 0; vi < nen_; ++vi)
  {
    const double v = prefac * my::funct_(vi);
    const int fvi = vi * my::numdofpernode_ + k;

    for (unsigned ui = 0; ui < nen_; ++ui)
    {
      const double vfunct = v * my::funct_(ui);
      for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
      {
        const int fui = ui * totalnummultiphasedofpernode + idof;

        emat(fvi, fui) += vfunct * (*prefaclinmassodfluid)[idof];
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------------- *
 |  standard Galerkin transient, old part of rhs and source term (OD fluid)     |
 |  + advanced reaction terms if reactionmanager is active     kremheller 07/17 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_hist_and_source_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh, const double fac,
    const double rhsint, const double densnp)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    const int numfluidphases = var_manager()->fluid_phase_manager()->num_fluid_phases();
    const int totalnummultiphasedofpernode = var_manager()->multiphase_mat()->num_mat();
    const int numvolfrac = var_manager()->fluid_phase_manager()->num_vol_frac();

    // 1) linearization of history:
    //    prefactor is densnp = porosity * rho * S
    //    --> porosity and saturation have to be linearized
    double vrhs = -1.0 * fac * my::scatravarmanager_->hist(k) * densnp;

    // get pre-fac vector for this scalar
    static std::vector<double> prefaclinmassodfluid(totalnummultiphasedofpernode, 0.0);
    var_manager()->get_pre_fac_lin_mass_od_fluid(k, &prefaclinmassodfluid);

    calc_lin_mass_matrix_type_od_fluid(
        emat, k, &prefaclinmassodfluid, totalnummultiphasedofpernode, vrhs);

    // 2) linearization of advanced reaction terms
    const std::shared_ptr<ScaTraEleReaManagerAdvReac> remanager = advreac::rea_manager();
    if (remanager->active())
    {
      const std::vector<double> myderivs =
          remanager->get_rea_body_force_deriv_vector_add_variables(k);

      // derivatives after primary variables of fluid
      std::vector<double> phiderivs(totalnummultiphasedofpernode, 0.0);

      for (int i = 0; i < numfluidphases; i++)
      {
        // porosity deriv at 2*numfluidphases: d reac / d phi_i = d reac / d poro * d poro / d phi_i
        phiderivs[i] +=
            myderivs[2 * numfluidphases] * var_manager()->fluid_phase_manager()->porosity_deriv(i);
        for (int j = 0; j < numfluidphases; j++)
        {
          // pressure derivs at       [0..numfluidphases]: d reac / d phi_i = d reac / d pres_j * d
          // pres_j / d phi_i saturation derivs at  [numflph..2*numflph-1]: d reac / d phi_i = d
          // reac /  d sat_j *  d sat_j / d phi_i
          phiderivs[i] += myderivs[j + numfluidphases] *
                              var_manager()->fluid_phase_manager()->saturation_deriv(j, i) +
                          myderivs[j] * var_manager()->fluid_phase_manager()->pressure_deriv(j, i);
        }
      }

      for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
      {
        // derivatives after volume fractions at [2*numfluidphases+1+ivolfrac]
        phiderivs[ivolfrac + numfluidphases] +=
            myderivs[2 * numfluidphases + 1 + ivolfrac] +
            myderivs[2 * numfluidphases] *
                var_manager()->fluid_phase_manager()->porosity_deriv(ivolfrac + numfluidphases);
        // derivatives after volume fraction pressures at [2*numfluidphases+numvolfrac+1+ivolfrac]
        phiderivs[ivolfrac + numfluidphases + numvolfrac] +=
            myderivs[2 * numfluidphases + numvolfrac + 1 + ivolfrac];
      }

      // fill matrix
      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * my::numdofpernode_ + k;
        // TODO: gen-alpha?
        const double v = my::funct_(vi) * my::scatraparatimint_->time_fac() * fac * (-1.0) /
                         var_manager()->density(k);

        for (unsigned ui = 0; ui < nen_; ++ui)
        {
          const double vfunct = v * my::funct_(ui);

          for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
          {
            const int fui = ui * totalnummultiphasedofpernode + idof;

            emat(fvi, fui) += vfunct * phiderivs[idof];
          }
        }
      }
    }
  }

  return;
}

/*-------------------------------------------------------------------- *
 | reactive term (OD fluid)                           kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_react_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double rhsfac, const double rea_phi)
{
  if (my::reamanager_->active() && var_manager()->evaluate_scalar(k))
  {
    const int totalnummultiphasedofpernode = var_manager()->multiphase_mat()->num_mat();

    double vrhs = rhsfac * rea_phi;

    // get pre-fac vector for this scalar
    static std::vector<double> prefaclinmassodfluid(totalnummultiphasedofpernode, 0.0);
    var_manager()->get_pre_fac_lin_mass_od_fluid(k, &prefaclinmassodfluid);

    calc_lin_mass_matrix_type_od_fluid(
        emat, k, &prefaclinmassodfluid, totalnummultiphasedofpernode, vrhs);
  }

  return;
}

/*------------------------------------------------------------------ *
 |  standard Galerkin diffusive term (OD fluid)     kremheller 07/17 |
 *----------------------------------------------------------------   */
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::calc_diff_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
    const int k,                            //!< index of current scalar
    const int ndofpernodemesh,              //!< number of dofs per node of ale element
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient at Gauss point
)
{
  // case of zero saturation/volfrac
  if (var_manager()->evaluate_scalar(k))
  {
    const int totalnummultiphasedofpernode = var_manager()->multiphase_mat()->num_mat();

    // get pre-fac vector for this scalar
    static std::vector<double> prefacdiffodfluid(totalnummultiphasedofpernode, 0.0);
    var_manager()->get_pre_fac_diff_od_fluid(
        k, rhsfac, my::diffmanager_->get_isotropic_diff(k), &prefacdiffodfluid);


    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * my::numdofpernode_ + k;

      double laplawf(0.0);
      my::get_laplacian_weak_form_rhs(laplawf, gradphi, vi);

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const double functlaplawf = laplawf * my::funct_(ui);

        // derivative w.r.t. fluid phases
        for (int idof = 0; idof < totalnummultiphasedofpernode; ++idof)
        {
          const int fui = ui * totalnummultiphasedofpernode + idof;

          emat(fvi, fui) += functlaplawf * prefacdiffodfluid[idof];
        }
      }
    }
  }
  return;
}
/*---------------------------------------------------------------------*
 | standard Galerkin terms  -- "shapederivatives" pressure gradient    |
 | gradient of pressure w.r.t. reference coordinates       vuong 08/14 |
 | put it into its own function                           wirthl 12/18 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcMultiPoroReac<distype>::apply_shape_derivs_pressure_grad(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double vrhs,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi, const Core::LinAlg::Matrix<nsd_, 1> refgradpres)
{
  if (nsd_ == 3)
  {
    const double xjm_0_0 = my::xjm_(0, 0);
    const double xjm_0_1 = my::xjm_(0, 1);
    const double xjm_0_2 = my::xjm_(0, 2);
    const double xjm_1_0 = my::xjm_(1, 0);
    const double xjm_1_1 = my::xjm_(1, 1);
    const double xjm_1_2 = my::xjm_(1, 2);
    const double xjm_2_0 = my::xjm_(2, 0);
    const double xjm_2_1 = my::xjm_(2, 1);
    const double xjm_2_2 = my::xjm_(2, 2);

    {
      const double refgradpres_0 = refgradpres(0);
      const double refgradpres_1 = refgradpres(1);
      const double refgradpres_2 = refgradpres(2);

      const double gradphi_0 = gradphi(0);
      const double gradphi_1 = gradphi(1);
      const double gradphi_2 = gradphi(2);

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const double v00 =
            +gradphi_1 *
                (refgradpres_0 * (my::deriv_(2, ui) * xjm_1_2 - my::deriv_(1, ui) * xjm_2_2) +
                    refgradpres_1 * (my::deriv_(0, ui) * xjm_2_2 - my::deriv_(2, ui) * xjm_0_2) +
                    refgradpres_2 * (my::deriv_(1, ui) * xjm_0_2 - my::deriv_(0, ui) * xjm_1_2)) +
            gradphi_2 *
                (refgradpres_0 * (my::deriv_(1, ui) * xjm_2_1 - my::deriv_(2, ui) * xjm_1_1) +
                    refgradpres_1 * (my::deriv_(2, ui) * xjm_0_1 - my::deriv_(0, ui) * xjm_2_1) +
                    refgradpres_2 * (my::deriv_(0, ui) * xjm_1_1 - my::deriv_(1, ui) * xjm_0_1));
        const double v01 =
            +gradphi_0 *
                (refgradpres_0 * (my::deriv_(1, ui) * xjm_2_2 - my::deriv_(2, ui) * xjm_1_2) +
                    refgradpres_1 * (my::deriv_(2, ui) * xjm_0_2 - my::deriv_(0, ui) * xjm_2_2) +
                    refgradpres_2 * (my::deriv_(0, ui) * xjm_1_2 - my::deriv_(1, ui) * xjm_0_2)) +
            gradphi_2 *
                (refgradpres_0 * (my::deriv_(2, ui) * xjm_1_0 - my::deriv_(1, ui) * xjm_2_0) +
                    refgradpres_1 * (my::deriv_(0, ui) * xjm_2_0 - my::deriv_(2, ui) * xjm_0_0) +
                    refgradpres_2 * (my::deriv_(1, ui) * xjm_0_0 - my::deriv_(0, ui) * xjm_1_0));
        const double v02 =
            +gradphi_0 *
                (refgradpres_0 * (my::deriv_(2, ui) * xjm_1_1 - my::deriv_(1, ui) * xjm_2_1) +
                    refgradpres_1 * (my::deriv_(0, ui) * xjm_2_1 - my::deriv_(2, ui) * xjm_0_1) +
                    refgradpres_2 * (my::deriv_(1, ui) * xjm_0_1 - my::deriv_(0, ui) * xjm_1_1)) +
            gradphi_1 *
                (refgradpres_0 * (my::deriv_(1, ui) * xjm_2_0 - my::deriv_(2, ui) * xjm_1_0) +
                    refgradpres_1 * (my::deriv_(2, ui) * xjm_0_0 - my::deriv_(0, ui) * xjm_2_0) +
                    refgradpres_2 * (my::deriv_(0, ui) * xjm_1_0 - my::deriv_(1, ui) * xjm_0_0));

        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * my::numdofpernode_ + k;
          const double v = vrhs * my::funct_(vi);

          emat(fvi, ui * 3 + 0) += v * v00;
          emat(fvi, ui * 3 + 1) += v * v01;
          emat(fvi, ui * 3 + 2) += v * v02;
        }
      }
    }
  }
  else if (nsd_ == 2)
  {
    {
      const double refgradpres_0 = refgradpres(0);
      const double refgradpres_1 = refgradpres(1);

      const double gradphi_0 = gradphi(0);
      const double gradphi_1 = gradphi(1);

      for (unsigned ui = 0; ui < nen_; ++ui)
      {
        const double v00 =
            +gradphi_1 * (-refgradpres_0 * my::deriv_(1, ui) + refgradpres_1 * my::deriv_(0, ui));
        const double v01 =
            +gradphi_0 * (refgradpres_0 * my::deriv_(1, ui) - refgradpres_1 * my::deriv_(0, ui));

        for (unsigned vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * my::numdofpernode_ + k;
          const double v = vrhs * my::funct_(vi);

          emat(fvi, ui * 2 + 0) += v * v00;
          emat(fvi, ui * 2 + 1) += v * v01;
        }
      }
    }
  }
  else
    FOUR_C_THROW("shapederivatives not implemented for 1D!");

  return;
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::quad4>;
// template class
// Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::quad9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::hex8>;
// template class
// Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::tet10>;
// template class
// Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::nurbs9>;
// template class
// Discret::Elements::ScaTraEleCalcMultiPoroReac<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
