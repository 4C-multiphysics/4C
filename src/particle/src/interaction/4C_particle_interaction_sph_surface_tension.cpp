// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_surface_tension.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_particle_interaction_sph_kernel.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_sph_surface_tension_barrier_force.hpp"
#include "4C_particle_interaction_sph_surface_tension_interface_viscosity.hpp"
#include "4C_particle_interaction_sph_surface_tension_recoilpressure_evaporation.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::SPHSurfaceTension::SPHSurfaceTension(const Teuchos::ParameterList& params)
    : params_sph_(params),
      liquidtype_(PARTICLEENGINE::Phase1),
      gastype_(PARTICLEENGINE::Phase2),
      time_(0.0),
      timerampfct_(params.get<int>("SURFACETENSION_RAMP_FUNCT")),
      alpha0_(params_sph_.get<double>("SURFACETENSIONCOEFFICIENT")),
      alphamin_(params_sph_.get<double>("SURFACETENSIONMINIMUM")),
      alpha_t_(params_sph_.get<double>("SURFACETENSIONTEMPFAC")),
      surf_ref_temp_(params_sph_.get<double>("SURFACETENSIONREFTEMP")),
      staticcontactangle_(params_sph_.get<double>("STATICCONTACTANGLE")),
      tpn_corr_cf_low_(params_sph_.get<double>("TRIPLEPOINTNORMAL_CORR_CF_LOW")),
      tpn_corr_cf_up_(params_sph_.get<double>("TRIPLEPOINTNORMAL_CORR_CF_UP")),
      trans_ref_temp_(params_sph_.get<double>("TRANS_REF_TEMPERATURE")),
      trans_d_t_surf_(params_sph_.get<double>("TRANS_DT_SURFACETENSION")),
      trans_d_t_mara_(params_sph_.get<double>("TRANS_DT_MARANGONI")),
      trans_d_t_curv_(params_sph_.get<double>("TRANS_DT_CURVATURE")),
      trans_d_t_wet_(params_sph_.get<double>("TRANS_DT_WETTING"))
{
  // empty constructor
}

ParticleInteraction::SPHSurfaceTension::~SPHSurfaceTension() = default;

void ParticleInteraction::SPHSurfaceTension::init()
{
  // init interface viscosity handler
  init_interface_viscosity_handler();

  // init evaporation induced recoil pressure handler
  init_recoil_pressure_evaporation_handler();

  // init barrier force handler
  init_barrier_force_handler();

  // init fluid particle types
  fluidtypes_ = {liquidtype_, gastype_};

  // init with potential boundary particle types
  boundarytypes_ = {PARTICLEENGINE::BoundaryPhase, PARTICLEENGINE::RigidPhase};

  // safety check
  if (not(alpha0_ > 0.0))
    FOUR_C_THROW("constant factor of surface tension coefficient not positive!");

  if (not(alpha0_ > alphamin_))
    FOUR_C_THROW("constant part smaller than minimum surface tension coefficient!");

  if (alpha_t_ != 0.0)
  {
    if (Teuchos::getIntegralValue<PARTICLE::TemperatureEvaluationScheme>(
            params_sph_, "TEMPERATUREEVALUATION") == PARTICLE::NoTemperatureEvaluation)
      FOUR_C_THROW("temperature evaluation needed for temperature dependent surface tension!");

    if (not params_sph_.get<bool>("TEMPERATUREGRADIENT"))
      FOUR_C_THROW(
          "temperature gradient evaluation needed for temperature dependent surface tension!");
  }

  if (trans_d_t_surf_ > 0.0 or trans_d_t_mara_ > 0.0 or trans_d_t_curv_ > 0.0 or
      trans_d_t_wet_ > 0.0)
  {
    if (Teuchos::getIntegralValue<PARTICLE::TemperatureEvaluationScheme>(
            params_sph_, "TEMPERATUREEVALUATION") == PARTICLE::NoTemperatureEvaluation)
      FOUR_C_THROW("temperature evaluation needed for linear transition of surface tension!");
  }
}

void ParticleInteraction::SPHSurfaceTension::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel,
    const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
    const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle,
    const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set kernel handler
  kernel_ = kernel;

  // set particle material handler
  particlematerial_ = particlematerial;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // setup interface viscosity handler
  if (interfaceviscosity_)
    interfaceviscosity_->setup(
        particleengineinterface, kernel, *particlematerial, equationofstatebundle, neighborpairs);

  // setup evaporation induced recoil pressure handler
  if (recoilpressureevaporation_) recoilpressureevaporation_->setup(particleengineinterface);

  // setup barrier force handler
  if (barrierforce_) barrierforce_->setup(particleengineinterface, neighborpairs);

  // safety check
  for (const auto& type_i : fluidtypes_)
    if (not particlecontainerbundle_->get_particle_types().count(type_i))
      FOUR_C_THROW("no particle container for particle type '{}' found!",
          PARTICLEENGINE::enum_to_type_name(type_i));

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->get_particle_types().count(type_i))
      boundarytypes_.erase(type_i);

  // setup interface normal of ghosted particles to refresh
  {
    std::vector<PARTICLEENGINE::StateEnum> states{PARTICLEENGINE::InterfaceNormal};

    // iterate over fluid particle types
    for (const auto& type_i : fluidtypes_)
      intnormtorefresh_.push_back(std::make_pair(type_i, states));
  }
}

void ParticleInteraction::SPHSurfaceTension::set_current_time(const double currenttime)
{
  time_ = currenttime;
}

void ParticleInteraction::SPHSurfaceTension::insert_particle_states_of_particle_types(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
    const
{
  bool haveboundarytypes = false;

  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum type = typeIt.first;

    if (boundarytypes_.count(type)) haveboundarytypes = true;
  }

  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum type = typeIt.first;

    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    // current particle type is not a fluid particle type
    if (not fluidtypes_.count(type)) continue;

    // states for surface tension evaluation scheme
    particlestates.insert({PARTICLEENGINE::ColorfieldGradient, PARTICLEENGINE::InterfaceNormal,
        PARTICLEENGINE::Curvature});

    if (haveboundarytypes)
      particlestates.insert({PARTICLEENGINE::WallColorfield, PARTICLEENGINE::WallInterfaceNormal});
  }
}

void ParticleInteraction::SPHSurfaceTension::compute_interface_quantities()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::SPHSurfaceTension::compute_interface_quantities");

  // compute colorfield gradient
  compute_colorfield_gradient();

  // compute interface normal
  compute_interface_normal();

  if (not boundarytypes_.empty())
  {
    // compute wall colorfield and wall interface normal
    compute_wall_colorfield_and_wall_interface_normal();

    // correct normal vector of particles close to triple point
    correct_triple_point_normal();
  }

  // refresh interface normal
  particleengineinterface_->refresh_particles_of_specific_states_and_types(intnormtorefresh_);
}

void ParticleInteraction::SPHSurfaceTension::add_acceleration_contribution()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::SPHSurfaceTension::add_acceleration_contribution");

  // compute curvature
  compute_curvature();

  // compute surface tension contribution
  compute_surface_tension_contribution();

  // compute temperature gradient driven contribution
  if (alpha_t_ != 0.0) compute_temp_grad_driven_contribution();

  // compute interface viscosity contribution
  if (interfaceviscosity_) interfaceviscosity_->compute_interface_viscosity_contribution();

  // compute evaporation induced recoil pressure contribution
  if (recoilpressureevaporation_)
    recoilpressureevaporation_->compute_recoil_pressure_contribution();

  // compute barrier force contribution
  if (barrierforce_) barrierforce_->compute_barrier_force_contribution();
}

void ParticleInteraction::SPHSurfaceTension::init_interface_viscosity_handler()
{
  // create interface viscosity handler
  if (params_sph_.get<bool>("INTERFACE_VISCOSITY"))
    interfaceviscosity_ = std::unique_ptr<ParticleInteraction::SPHInterfaceViscosity>(
        new ParticleInteraction::SPHInterfaceViscosity(params_sph_));

  // init interface viscosity handler
  if (interfaceviscosity_) interfaceviscosity_->init();
}

void ParticleInteraction::SPHSurfaceTension::init_recoil_pressure_evaporation_handler()
{
  // create evaporation induced recoil pressure handler
  if (params_sph_.get<bool>("VAPOR_RECOIL"))
    recoilpressureevaporation_ = std::unique_ptr<ParticleInteraction::SPHRecoilPressureEvaporation>(
        new ParticleInteraction::SPHRecoilPressureEvaporation(params_sph_));

  // init evaporation induced recoil pressure handler
  if (recoilpressureevaporation_) recoilpressureevaporation_->init();
}

void ParticleInteraction::SPHSurfaceTension::init_barrier_force_handler()
{
  // create barrier force handler
  if (params_sph_.get<bool>("BARRIER_FORCE"))
    barrierforce_ = std::unique_ptr<ParticleInteraction::SPHBarrierForce>(
        new ParticleInteraction::SPHBarrierForce(params_sph_));

  // init barrier force handler
  if (barrierforce_) barrierforce_->init();
}

void ParticleInteraction::SPHSurfaceTension::compute_colorfield_gradient() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // clear colorfield gradient state
    container_i->clear_state(PARTICLEENGINE::ColorfieldGradient);
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_equal_combination(fluidtypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // no evaluation for particles of same type
    if (type_i == type_j) continue;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
    double* cfg_i = container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);
    const double* dens_j = container_j->get_ptr_to_state(PARTICLEENGINE::Density, particle_j);
    double* cfg_j = container_j->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_j);

    // (current) volume of particle i and j
    const double V_i = mass_i[0] / dens_i[0];
    const double V_j = mass_j[0] / dens_j[0];

    const double fac = (Utils::pow<2>(V_i) + Utils::pow<2>(V_j)) / (dens_i[0] + dens_j[0]);

    // sum contribution of neighboring particle j
    Utils::vec_add_scale(cfg_i, dens_i[0] / V_i * fac * particlepair.dWdrij_, particlepair.e_ij_);

    // sum contribution of neighboring particle i
    if (status_j == PARTICLEENGINE::Owned)
      Utils::vec_add_scale(
          cfg_j, -dens_j[0] / V_j * fac * particlepair.dWdrji_, particlepair.e_ij_);
  }

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle state
      const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
      double* cfg_i = container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);

      // norm of colorfield gradient
      const double cfg_i_norm = Utils::vec_norm_two(cfg_i);

      // clear colorfield gradient
      if (not(cfg_i_norm > (1.0e-10 * rad_i[0]))) Utils::vec_clear(cfg_i);
    }
  }
}

void ParticleInteraction::SPHSurfaceTension::compute_interface_normal() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // clear interface normal state
    container_i->clear_state(PARTICLEENGINE::InterfaceNormal);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
      const double* cfg_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);
      double* ifn_i = container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);

      // norm of colorfield gradient
      const double cfg_i_norm = Utils::vec_norm_two(cfg_i);

      // set interface normal
      if (cfg_i_norm > (1.0e-10 * rad_i[0])) Utils::vec_set_scale(ifn_i, 1.0 / cfg_i_norm, cfg_i);
    }
  }
}

void ParticleInteraction::SPHSurfaceTension::compute_wall_colorfield_and_wall_interface_normal()
    const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // clear wall colorfield state
    container_i->clear_state(PARTICLEENGINE::WallColorfield);

    // clear wall interface normal state
    container_i->clear_state(PARTICLEENGINE::WallInterfaceNormal);
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      boundarytypes_, fluidtypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // evaluate contribution of neighboring boundary particle j
    if (fluidtypes_.count(type_i))
    {
      // get material for current particle type
      const Mat::PAR::ParticleMaterialBase* material_j =
          particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

      // get pointer to particle states
      const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
      const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
      double* wallcf_i = container_i->get_ptr_to_state(PARTICLEENGINE::WallColorfield, particle_i);
      double* wallifn_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::WallInterfaceNormal, particle_i);

      const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);
      const double* temp_j =
          container_j->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_j);

      // (current) volume of particle i
      const double V_i = mass_i[0] / dens_i[0];

      // (initial) volume of boundary particle j
      const double V_j = mass_j[0] / material_j->initDensity_;

      const double fac = (Utils::pow<2>(V_i) + Utils::pow<2>(V_j)) * dens_i[0] /
                         (V_i * (dens_i[0] + material_j->initDensity_));


      // evaluate transition factor below reference temperature
      double tempfac = 1.0;
      if (trans_d_t_wet_ > 0.0)
        tempfac =
            Utils::comp_lin_trans(temp_j[0], trans_ref_temp_ - trans_d_t_wet_, trans_ref_temp_);

      // sum contribution of neighboring boundary particle j
      wallcf_i[0] += tempfac * fac * particlepair.Wij_;
      Utils::vec_add_scale(wallifn_i, tempfac * fac * particlepair.dWdrij_, particlepair.e_ij_);
    }

    // evaluate contribution of neighboring boundary particle i
    if (fluidtypes_.count(type_j) and status_j == PARTICLEENGINE::Owned)
    {
      // get material for current particle type
      const Mat::PAR::ParticleMaterialBase* material_i =
          particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

      // get pointer to particle states
      const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);
      const double* dens_j = container_j->get_ptr_to_state(PARTICLEENGINE::Density, particle_j);
      double* wallcf_j = container_j->get_ptr_to_state(PARTICLEENGINE::WallColorfield, particle_j);
      double* wallifn_j =
          container_j->get_ptr_to_state(PARTICLEENGINE::WallInterfaceNormal, particle_j);

      const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
      const double* temp_i =
          container_i->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);

      // (initial) volume of boundary particle i
      const double V_i = mass_i[0] / material_i->initDensity_;

      // (current) volume of particle j
      const double V_j = mass_j[0] / dens_j[0];

      const double fac = (Utils::pow<2>(V_i) + Utils::pow<2>(V_j)) * dens_j[0] /
                         (V_j * (material_i->initDensity_ + dens_j[0]));

      // evaluate transition factor below reference temperature
      double tempfac = 1.0;
      if (trans_d_t_wet_ > 0.0)
        tempfac =
            Utils::comp_lin_trans(temp_i[0], trans_ref_temp_ - trans_d_t_wet_, trans_ref_temp_);

      // sum contribution of neighboring boundary particle i
      wallcf_j[0] += tempfac * fac * particlepair.Wji_;
      Utils::vec_add_scale(wallifn_j, -tempfac * fac * particlepair.dWdrji_, particlepair.e_ij_);
    }
  }

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle state
      const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
      double* wallifn_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::WallInterfaceNormal, particle_i);

      // norm of wall interface normal
      const double wallifn_i_norm = Utils::vec_norm_two(wallifn_i);

      // scale or clear wall interface normal
      if (wallifn_i_norm > (1.0e-10 * rad_i[0]))
        Utils::vec_set_scale(wallifn_i, 1.0 / wallifn_i_norm, wallifn_i);
      else
        Utils::vec_clear(wallifn_i);
    }
  }
}

void ParticleInteraction::SPHSurfaceTension::correct_triple_point_normal() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // static contact angle with respect to liquid particle type
    const double staticcontactangle =
        (type_i == liquidtype_) ? staticcontactangle_ : (180 - staticcontactangle_);

    // convert static contact angle in radians
    const double theta_0 = staticcontactangle * std::numbers::pi / 180.0;

    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
      const double* wallifn_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::WallInterfaceNormal, particle_i);
      const double* wallcf_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::WallColorfield, particle_i);
      double* ifn_i = container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);

      // evaluation only for non-zero wall interface normal
      if (not(Utils::vec_norm_two(wallifn_i) > 0.0)) continue;

      // evaluation only for non-zero interface normal
      if (not(Utils::vec_norm_two(ifn_i) > 0.0)) continue;

      // determine correction factor
      double f_i = Utils::comp_lin_trans(wallcf_i[0], tpn_corr_cf_low_, tpn_corr_cf_up_);

      // determine wall interface tangential
      double wallift_i[3];
      Utils::vec_set(wallift_i, ifn_i);
      Utils::vec_add_scale(wallift_i, -Utils::vec_dot(ifn_i, wallifn_i), wallifn_i);

      // norm of wall interface tangential
      const double wallift_i_norm = Utils::vec_norm_two(wallift_i);

      // scale or clear wall interface tangential
      if (wallift_i_norm > (1.0e-10 * rad_i[0]))
        Utils::vec_set_scale(wallift_i, 1.0 / wallift_i_norm, wallift_i);
      else
        Utils::vec_clear(wallift_i);

      // determine triple point normal
      double tpn_i[3];
      Utils::vec_set_scale(tpn_i, std::sin(theta_0), wallift_i);
      Utils::vec_add_scale(tpn_i, -std::cos(theta_0), wallifn_i);

      // determine corrected interface normal
      double corifn_i[3];
      Utils::vec_set_scale(corifn_i, f_i, ifn_i);
      Utils::vec_add_scale(corifn_i, (1.0 - f_i), tpn_i);

      // norm of corrected interface normal
      const double corifn_i_norm = Utils::vec_norm_two(corifn_i);

      // scale or clear interface normal
      if (corifn_i_norm > (1.0e-10 * rad_i[0]))
        Utils::vec_set_scale(ifn_i, 1.0 / corifn_i_norm, corifn_i);
      else
        Utils::vec_clear(ifn_i);
    }
  }
}

void ParticleInteraction::SPHSurfaceTension::compute_curvature() const
{
  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  std::vector<std::vector<double>> sumj_nij_Vj_eij_dWij(typevectorsize);
  std::vector<std::vector<double>> sumj_Vj_Wij(typevectorsize);

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // clear curvature state
    container_i->clear_state(PARTICLEENGINE::Curvature);

    // get number of particles stored in container
    const int particlestored = container_i->particles_stored();

    // allocate memory
    sumj_nij_Vj_eij_dWij[type_i].assign(particlestored, 0.0);
    sumj_Vj_Wij[type_i].assign(particlestored, 0.0);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
      const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
      const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
      const double* ifn_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);
      const double* temp_i =
          container_i->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);

      // evaluation only for non-zero interface normal
      if (not(Utils::vec_norm_two(ifn_i) > 0.0)) continue;

      // evaluate kernel
      const double Wii = kernel_->w0(rad_i[0]);

      // (current) volume of particle i
      const double V_i = mass_i[0] / dens_i[0];

      // evaluate transition factor above reference temperature
      double tempfac = 1.0;
      if (trans_d_t_curv_ > 0.0)
        tempfac = Utils::lin_trans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_curv_);

      // add self-interaction
      sumj_Vj_Wij[type_i][particle_i] += tempfac * V_i * Wii;
    }
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_equal_combination(fluidtypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
    const double* ifn_i =
        container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);
    const double* temp_i =
        container_i->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);
    const double* dens_j = container_j->get_ptr_to_state(PARTICLEENGINE::Density, particle_j);
    const double* ifn_j =
        container_j->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_j);
    const double* temp_j =
        container_j->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_j);

    // evaluation only for non-zero interface normals
    if (not(Utils::vec_norm_two(ifn_i) > 0.0) or not(Utils::vec_norm_two(ifn_j) > 0.0)) continue;

    // change sign of interface normal for different particle types
    double signfac = (type_i == type_j) ? 1.0 : -1.0;

    double n_ij[3];
    Utils::vec_set(n_ij, ifn_i);
    Utils::vec_add_scale(n_ij, -signfac, ifn_j);

    const double nij_eij = Utils::vec_dot(n_ij, particlepair.e_ij_);

    // evaluate contribution of neighboring particle j
    {
      // (current) volume of particle j
      const double V_j = mass_j[0] / dens_j[0];

      // evaluate transition factor above reference temperature
      double tempfac = 1.0;
      if (trans_d_t_curv_ > 0.0)
        tempfac = Utils::lin_trans(temp_j[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_curv_);

      // sum contribution of neighboring particle j
      sumj_nij_Vj_eij_dWij[type_i][particle_i] += tempfac * V_j * nij_eij * particlepair.dWdrij_;
      sumj_Vj_Wij[type_i][particle_i] += tempfac * V_j * particlepair.Wij_;
    }

    // evaluate contribution of neighboring particle i
    if (status_j == PARTICLEENGINE::Owned)
    {
      // (current) volume of particle i
      const double V_i = mass_i[0] / dens_i[0];

      // evaluate transition factor above reference temperature
      double tempfac = 1.0;
      if (trans_d_t_curv_ > 0.0)
        tempfac = Utils::lin_trans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_curv_);

      // sum contribution of neighboring particle i
      sumj_nij_Vj_eij_dWij[type_j][particle_j] +=
          signfac * tempfac * V_i * nij_eij * particlepair.dWdrji_;
      sumj_Vj_Wij[type_j][particle_j] += tempfac * V_i * particlepair.Wji_;
    }
  }

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
      const double* ifn_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);
      double* curv_i = container_i->get_ptr_to_state(PARTICLEENGINE::Curvature, particle_i);

      // evaluation only for non-zero interface normal
      if (not(Utils::vec_norm_two(ifn_i) > 0.0)) continue;

      // evaluation only for meaningful contributions
      if (not(std::abs(sumj_Vj_Wij[type_i][particle_i]) > (1.0e-10 * rad_i[0]))) continue;

      // compute curvature
      curv_i[0] = -sumj_nij_Vj_eij_dWij[type_i][particle_i] / sumj_Vj_Wij[type_i][particle_i];
    }
  }
}

void ParticleInteraction::SPHSurfaceTension::compute_surface_tension_contribution() const
{
  // evaluate surface tension time ramp function
  double timefac = 1.0;
  if (timerampfct_ > 0)
    timefac = Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfTime>(timerampfct_)
                  .evaluate(time_);

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
      const double* curv_i = container_i->get_ptr_to_state(PARTICLEENGINE::Curvature, particle_i);
      const double* cfg_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);
      const double* ifn_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);
      const double* temp_i =
          container_i->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);
      double* acc_i = container_i->get_ptr_to_state(PARTICLEENGINE::Acceleration, particle_i);

      // evaluation only for non-zero interface normal
      if (not(Utils::vec_norm_two(ifn_i) > 0.0)) continue;

      // evaluate surface tension coefficient
      double alpha = alpha0_;
      if (alpha_t_ != 0.0)
      {
        alpha += alpha_t_ * (temp_i[0] - surf_ref_temp_);
        alpha = std::max(alpha, alphamin_);
      }

      // evaluate transition factor above reference temperature
      double tempfac = 1.0;
      if (trans_d_t_surf_ > 0.0)
        tempfac = Utils::lin_trans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_surf_);

      // add contribution to acceleration
      Utils::vec_add_scale(acc_i, -timefac * tempfac * alpha * curv_i[0] / dens_i[0], cfg_i);
    }
  }
}

void ParticleInteraction::SPHSurfaceTension::compute_temp_grad_driven_contribution() const
{
  // evaluate surface tension time ramp function
  double timefac = 1.0;
  if (timerampfct_ > 0)
    timefac = Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfTime>(timerampfct_)
                  .evaluate(time_);

  // temperature in transition from linear to constant regime of surface tension coefficient
  const double transitiontemp = surf_ref_temp_ + (alphamin_ - alpha0_) / alpha_t_;

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
      const double* cfg_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);
      const double* ifn_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);
      const double* temp_i = container_i->get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);
      const double* tempgrad_i =
          container_i->get_ptr_to_state(PARTICLEENGINE::temperature_gradient, particle_i);
      double* acc_i = container_i->get_ptr_to_state(PARTICLEENGINE::Acceleration, particle_i);

      // evaluation only for non-zero interface normal
      if (not(Utils::vec_norm_two(ifn_i) > 0.0)) continue;

      // no evaluation in the regime of constant surface tension coefficient
      if (temp_i[0] > transitiontemp) continue;

      // projection of temperature gradient onto tangential plane defined by interface normal
      double tempgrad_i_proj[3];
      Utils::vec_set(tempgrad_i_proj, tempgrad_i);
      Utils::vec_add_scale(tempgrad_i_proj, -Utils::vec_dot(tempgrad_i, ifn_i), ifn_i);

      // evaluate transition factor above reference temperature
      double tempfac = 1.0;
      if (trans_d_t_mara_ > 0.0)
        tempfac = Utils::lin_trans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_mara_);

      // add contribution to acceleration
      Utils::vec_add_scale(acc_i,
          timefac * tempfac * alpha_t_ * Utils::vec_norm_two(cfg_i) / dens_i[0], tempgrad_i_proj);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
