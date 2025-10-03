// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_surface_tension_interface_viscosity.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_artificialviscosity.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_particle_interaction_sph_kernel.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::SPHInterfaceViscosity::SPHInterfaceViscosity(
    const Teuchos::ParameterList& params)
    : params_sph_(params),
      liquidtype_(PARTICLEENGINE::Phase1),
      gastype_(PARTICLEENGINE::Phase2),
      artvisc_lg_int_(params_sph_.get<double>("INTERFACE_VISCOSITY_LIQUIDGAS")),
      artvisc_sl_int_(params_sph_.get<double>("INTERFACE_VISCOSITY_SOLIDLIQUID")),
      trans_ref_temp_(params_sph_.get<double>("TRANS_REF_TEMPERATURE")),
      trans_d_t_intvisc_(params_sph_.get<double>("TRANS_DT_INTVISC"))
{
  // empty constructor
}

ParticleInteraction::SPHInterfaceViscosity::~SPHInterfaceViscosity() = default;

void ParticleInteraction::SPHInterfaceViscosity::init()
{
  // init artificial viscosity handler
  init_artificial_viscosity_handler();

  // init fluid particle types
  fluidtypes_ = {liquidtype_, gastype_};

  // init with potential boundary particle types
  boundarytypes_ = {PARTICLEENGINE::BoundaryPhase, PARTICLEENGINE::RigidPhase};

  // safety check
  if (trans_d_t_intvisc_ > 0.0)
  {
    if (Teuchos::getIntegralValue<PARTICLE::TemperatureEvaluationScheme>(
            params_sph_, "TEMPERATUREEVALUATION") == PARTICLE::NoTemperatureEvaluation)
      FOUR_C_THROW("temperature evaluation needed for linear transition of interface viscosity!");
  }
}

void ParticleInteraction::SPHInterfaceViscosity::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel,
    ParticleInteraction::MaterialHandler& particlematerial,
    const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle,
    const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set kernel handler
  kernel_ = kernel;

  // set equation of state handler
  equationofstatebundle_ = equationofstatebundle;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // setup artificial viscosity handler
  artificialviscosity_->setup();

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

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  // allocate memory to hold particle types
  fluidmaterial_.resize(typevectorsize);

  // iterate over all fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    fluidmaterial_[type_i] = dynamic_cast<const Mat::PAR::ParticleMaterialSPHFluid*>(
        particlematerial.get_ptr_to_particle_mat_parameter(type_i));
  }
}

void ParticleInteraction::SPHInterfaceViscosity::compute_interface_viscosity_contribution() const
{
  // compute interface viscosity contribution (particle contribution)
  compute_interface_viscosity_particle_contribution();

  // compute interface viscosity contribution (particle-boundary contribution)
  compute_interface_viscosity_particle_boundary_contribution();
}

void ParticleInteraction::SPHInterfaceViscosity::init_artificial_viscosity_handler()
{
  // create artificial viscosity handler
  artificialviscosity_ = std::unique_ptr<ParticleInteraction::SPHArtificialViscosity>(
      new ParticleInteraction::SPHArtificialViscosity());

  // init artificial viscosity handler
  artificialviscosity_->init();
}

void ParticleInteraction::SPHInterfaceViscosity::compute_interface_viscosity_particle_contribution()
    const
{
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

    // get material for particle types
    const Mat::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[type_i];
    const Mat::PAR::ParticleMaterialSPHFluid* material_j = fluidmaterial_[type_j];

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(PARTICLEENGINE::Velocity, particle_i);
    const double* cfg_i =
        container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);
    const double* temp_i =
        container_i->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);
    double* acc_i = container_i->get_ptr_to_state(PARTICLEENGINE::Acceleration, particle_i);

    const double* rad_j = container_j->get_ptr_to_state(PARTICLEENGINE::Radius, particle_j);
    const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);
    const double* dens_j = container_j->get_ptr_to_state(PARTICLEENGINE::Density, particle_j);
    const double* vel_j = container_j->get_ptr_to_state(PARTICLEENGINE::Velocity, particle_j);
    const double* cfg_j =
        container_j->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_j);
    const double* temp_j =
        container_j->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_j);
    double* acc_j = container_j->get_ptr_to_state(PARTICLEENGINE::Acceleration, particle_j);

    // get smoothing length
    const double h_i = kernel_->smoothing_length(rad_i[0]);
    const double h_j = (rad_i[0] == rad_j[0]) ? h_i : kernel_->smoothing_length(rad_j[0]);

    // evaluate transition factor above reference temperature
    double tempfac_i = 1.0;
    double tempfac_j = 1.0;
    if (trans_d_t_intvisc_ > 0.0)
    {
      tempfac_i =
          Utils::comp_lin_trans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_intvisc_);
      tempfac_j =
          Utils::comp_lin_trans(temp_j[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_intvisc_);
    }

    // compute artificial viscosity
    const double artvisc_i =
        Utils::vec_norm_two(cfg_i) * h_i * artvisc_lg_int_ + tempfac_i * artvisc_sl_int_;
    const double artvisc_j =
        Utils::vec_norm_two(cfg_j) * h_j * artvisc_lg_int_ + tempfac_j * artvisc_sl_int_;

    // evaluate artificial viscosity
    if (artvisc_i > 0.0 or artvisc_j > 0.0)
    {
      // particle averaged smoothing length
      const double h_ij = 0.5 * (h_i + h_j);

      // get speed of sound
      const double c_i = material_i->speed_of_sound();
      const double c_j = (type_i == type_j) ? c_i : material_j->speed_of_sound();

      // particle averaged speed of sound
      const double c_ij = 0.5 * (c_i + c_j);

      // particle averaged density
      const double dens_ij = 0.5 * (dens_i[0] + dens_j[0]);

      // evaluate artificial viscosity
      artificialviscosity_->artificial_viscosity(vel_i, vel_j, mass_i, mass_j, artvisc_i, artvisc_j,
          particlepair.dWdrij_, particlepair.dWdrji_, dens_ij, h_ij, c_ij, particlepair.absdist_,
          particlepair.e_ij_, acc_i, acc_j);
    }
  }
}

void ParticleInteraction::SPHInterfaceViscosity::
    compute_interface_viscosity_particle_boundary_contribution() const
{
  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      fluidtypes_, boundarytypes_, relindices);

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

    // swap fluid particle and boundary particle
    const bool swapparticles = boundarytypes_.count(type_i);
    if (swapparticles)
    {
      std::tie(type_i, status_i, particle_i) = particlepair.tuple_j_;
      std::tie(type_j, status_j, particle_j) = particlepair.tuple_i_;
    }

    // absolute distance between particles
    const double absdist = particlepair.absdist_;

    // versor from particle j to i
    double e_ij[3];
    Utils::vec_set(e_ij, particlepair.e_ij_);
    if (swapparticles) Utils::vec_scale(e_ij, -1.0);

    // first derivative of kernel
    const double dWdrij = (swapparticles) ? particlepair.dWdrji_ : particlepair.dWdrij_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get material for particle types
    const Mat::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[type_i];

    // get equation of state for particle types
    const ParticleInteraction::SPHEquationOfStateBase* equationofstate_i =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(PARTICLEENGINE::Velocity, particle_i);
    const double* cfg_i =
        container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);
    const double* temp_i =
        container_i->cond_get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);

    double* acc_i = nullptr;
    if (status_i == PARTICLEENGINE::Owned)
      acc_i = container_i->get_ptr_to_state(PARTICLEENGINE::Acceleration, particle_i);

    // get pointer to boundary particle states
    const double* mass_j = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
    const double* press_j =
        container_j->get_ptr_to_state(PARTICLEENGINE::BoundaryPressure, particle_j);
    const double* vel_j =
        container_j->get_ptr_to_state(PARTICLEENGINE::BoundaryVelocity, particle_j);

    double temp_dens(0.0);
    temp_dens = equationofstate_i->pressure_to_density(press_j[0], material_i->initDensity_);
    const double* dens_j = &temp_dens;

    // get smoothing length
    const double h_i = kernel_->smoothing_length(rad_i[0]);

    // evaluate transition factor above reference temperature
    double tempfac_i = 1.0;
    if (trans_d_t_intvisc_ > 0.0)
      tempfac_i =
          Utils::comp_lin_trans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_intvisc_);

    // compute artificial viscosity
    const double artvisc_i =
        Utils::vec_norm_two(cfg_i) * h_i * artvisc_lg_int_ + tempfac_i * artvisc_sl_int_;

    // evaluate artificial viscosity
    if (artvisc_i > 0.0)
    {
      // get speed of sound
      const double c_i = material_i->speed_of_sound();

      // particle averaged density
      const double dens_ij = 0.5 * (dens_i[0] + dens_j[0]);

      // evaluate artificial viscosity
      artificialviscosity_->artificial_viscosity(vel_i, vel_j, mass_i, mass_j, artvisc_i, 0.0,
          dWdrij, 0.0, dens_ij, h_i, c_i, absdist, e_ij, acc_i, nullptr);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
