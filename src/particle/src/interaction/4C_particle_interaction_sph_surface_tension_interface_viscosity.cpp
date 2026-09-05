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

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHInterfaceViscosity::SPHInterfaceViscosity(const Teuchos::ParameterList& params)
    : params_sph_(params),
      liquidtype_(ParticleType::Phase1),
      gastype_(ParticleType::Phase2),
      fluidtypes_({liquidtype_, gastype_}),
      boundarytypes_(
          {ParticleType::BoundaryPhase, ParticleType::RigidPhase, ParticleType::PDPhase}),
      artvisc_lg_int_(params_sph_.get<double>("INTERFACE_VISCOSITY_LIQUIDGAS")),
      artvisc_sl_int_(params_sph_.get<double>("INTERFACE_VISCOSITY_SOLIDLIQUID")),
      trans_ref_temp_(params_sph_.get<double>("TRANS_REF_TEMPERATURE")),
      trans_d_t_intvisc_(params_sph_.get<double>("TRANS_DT_INTVISC"))
{
  init_artificial_viscosity_handler();

  if (trans_d_t_intvisc_ > 0.0)
  {
    if (Teuchos::getIntegralValue<Particle::TemperatureEvaluationScheme>(
            params_sph_, "TEMPERATUREEVALUATION") == Particle::NoTemperatureEvaluation)
      FOUR_C_THROW("temperature evaluation needed for linear transition of interface viscosity!");
  }
}

Particle::SPHInterfaceViscosity::~SPHInterfaceViscosity() = default;

void Particle::SPHInterfaceViscosity::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::SPHKernelBase> kernel,
    Particle::MaterialHandler& particlematerial,
    const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs)
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

  // safety check
  for (const auto& type_i : fluidtypes_)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      FOUR_C_THROW("no particle container for particle type '{}' found!",
          Particle::enum_to_type_name(type_i));

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      boundarytypes_.erase(type_i);

  // determine size of vectors indexed by particle types
  const int typevectorsize =
      static_cast<int>(*(--particlecontainerbundle_->get_particle_types().end())) + 1;

  // allocate memory to hold particle types
  fluidmaterial_.resize(typevectorsize);

  // iterate over all fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    fluidmaterial_[static_cast<int>(type_i)] =
        dynamic_cast<const Mat::PAR::ParticleMaterialSPHFluid*>(
            particlematerial.get_ptr_to_particle_mat_parameter(type_i));
  }
}

void Particle::SPHInterfaceViscosity::compute_interface_viscosity_contribution() const
{
  // compute interface viscosity contribution (particle contribution)
  compute_interface_viscosity_particle_contribution();

  // compute interface viscosity contribution (particle-boundary contribution)
  compute_interface_viscosity_particle_boundary_contribution();
}

void Particle::SPHInterfaceViscosity::init_artificial_viscosity_handler()
{
  // create artificial viscosity handler
  artificialviscosity_ = std::make_unique<Particle::SPHArtificialViscosity>();
}

void Particle::SPHInterfaceViscosity::compute_interface_viscosity_particle_contribution() const
{
  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_equal_combination(fluidtypes_, relindices);

  // get pointers to particle states
  const int statedim = Particle::enum_to_state_dim(ParticleState::Position);
  ConstParticleContainerBundleStatePtrs& rad =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Radius);
  ConstParticleContainerBundleStatePtrs& mass =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Mass);
  ConstParticleContainerBundleStatePtrs& dens =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Density);
  ConstParticleContainerBundleStatePtrs& vel =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Velocity);
  ConstParticleContainerBundleStatePtrs& cfg =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::ColorfieldGradient);
  ConstParticleContainerBundleStatePtrs& temp =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Temperature);
  ParticleContainerBundleStatePtrs& acc =
      particlecontainerbundle_->try_get_ptrs_to_state_writable(ParticleState::Acceleration);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    ParticleType type_i;
    ParticleStatus status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    ParticleType type_j;
    ParticleStatus status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get material for particle types
    const Mat::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[static_cast<int>(type_i)];
    const Mat::PAR::ParticleMaterialSPHFluid* material_j = fluidmaterial_[static_cast<int>(type_j)];

    // get pointers to particle states
    const double* rad_i = Particle::bundle_state_ptrs_index(rad, type_i, status_i, particle_i);
    const double* mass_i = Particle::bundle_state_ptrs_index(mass, type_i, status_i, particle_i);
    const double* dens_i = Particle::bundle_state_ptrs_index(dens, type_i, status_i, particle_i);
    const double* vel_i =
        Particle::bundle_state_ptrs_index(vel, type_i, status_i, particle_i, statedim);
    const double* cfg_i =
        Particle::bundle_state_ptrs_index(cfg, type_i, status_i, particle_i, statedim);
    const double* temp_i =
        Particle::bundle_state_ptrs_index(temp, nullptr, type_i, status_i, particle_i);
    double* acc_i = Particle::bundle_state_ptrs_index(acc, type_i, status_i, particle_i, statedim);

    const double* rad_j = Particle::bundle_state_ptrs_index(rad, type_j, status_j, particle_j);
    const double* mass_j = Particle::bundle_state_ptrs_index(mass, type_j, status_j, particle_j);
    const double* dens_j = Particle::bundle_state_ptrs_index(dens, type_j, status_j, particle_j);
    const double* vel_j =
        Particle::bundle_state_ptrs_index(vel, type_j, status_j, particle_j, statedim);
    const double* cfg_j =
        Particle::bundle_state_ptrs_index(cfg, type_j, status_j, particle_j, statedim);
    const double* temp_j =
        Particle::bundle_state_ptrs_index(temp, nullptr, type_j, status_j, particle_j);
    double* acc_j = Particle::bundle_state_ptrs_index(acc, type_j, status_j, particle_j, statedim);

    // get smoothing length
    const double h_i = kernel_->smoothing_length(rad_i[0]);
    const double h_j = (rad_i[0] == rad_j[0]) ? h_i : kernel_->smoothing_length(rad_j[0]);

    // evaluate transition factor above reference temperature
    double tempfac_i = 1.0;
    double tempfac_j = 1.0;
    if (trans_d_t_intvisc_ > 0.0)
    {
      tempfac_i = ParticleUtils::comp_lin_trans(
          temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_intvisc_);
      tempfac_j = ParticleUtils::comp_lin_trans(
          temp_j[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_intvisc_);
    }

    // compute artificial viscosity
    const double artvisc_i =
        ParticleUtils::vec_norm_two(cfg_i) * h_i * artvisc_lg_int_ + tempfac_i * artvisc_sl_int_;
    const double artvisc_j =
        ParticleUtils::vec_norm_two(cfg_j) * h_j * artvisc_lg_int_ + tempfac_j * artvisc_sl_int_;

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

void Particle::SPHInterfaceViscosity::compute_interface_viscosity_particle_boundary_contribution()
    const
{
  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      fluidtypes_, boundarytypes_, relindices);

  // get pointers to particle states
  const int statedim = Particle::enum_to_state_dim(ParticleState::Position);
  ConstParticleContainerBundleStatePtrs& rad =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Radius);
  ConstParticleContainerBundleStatePtrs& mass =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Mass);
  ConstParticleContainerBundleStatePtrs& dens =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Density);
  ConstParticleContainerBundleStatePtrs& vel =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Velocity);
  ConstParticleContainerBundleStatePtrs& cfg =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::ColorfieldGradient);
  ConstParticleContainerBundleStatePtrs& temp =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Temperature);
  ConstParticleContainerBundleStatePtrs& bpress =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::BoundaryPressure);
  ConstParticleContainerBundleStatePtrs& bvel =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::BoundaryVelocity);
  ParticleContainerBundleStatePtrs& acc = particlecontainerbundle_->try_get_ptrs_to_state_writable(
      ParticleState::Acceleration, ParticleStatus::Owned);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    ParticleType type_i;
    ParticleStatus status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    ParticleType type_j;
    ParticleStatus status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // swap fluid particle and boundary particle
    const bool swapparticles = boundarytypes_.contains(type_i);
    if (swapparticles)
    {
      std::tie(type_i, status_i, particle_i) = particlepair.tuple_j_;
      std::tie(type_j, status_j, particle_j) = particlepair.tuple_i_;
    }

    // absolute distance between particles
    const double absdist = particlepair.absdist_;

    // versor from particle j to i
    double e_ij[3];
    ParticleUtils::vec_set(e_ij, particlepair.e_ij_);
    if (swapparticles) ParticleUtils::vec_scale(e_ij, -1.0);

    // first derivative of kernel
    const double dWdrij = (swapparticles) ? particlepair.dWdrji_ : particlepair.dWdrij_;

    // get material for particle types
    const Mat::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[static_cast<int>(type_i)];

    // get equation of state for particle types
    const Particle::SPHEquationOfStateBase* equationofstate_i =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // get pointers to particle states
    const double* rad_i = Particle::bundle_state_ptrs_index(rad, type_i, status_i, particle_i);
    const double* mass_i = Particle::bundle_state_ptrs_index(mass, type_i, status_i, particle_i);
    const double* dens_i = Particle::bundle_state_ptrs_index(dens, type_i, status_i, particle_i);
    const double* vel_i =
        Particle::bundle_state_ptrs_index(vel, type_i, status_i, particle_i, statedim);
    const double* cfg_i =
        Particle::bundle_state_ptrs_index(cfg, type_i, status_i, particle_i, statedim);
    const double* temp_i =
        Particle::bundle_state_ptrs_index(temp, nullptr, type_i, status_i, particle_i);

    double* acc_i = nullptr;
    if (status_i == ParticleStatus::Owned)
      acc_i = Particle::bundle_state_ptrs_index(acc, type_i, status_i, particle_i, statedim);

    // get pointers to boundary particle states
    const double* mass_j = Particle::bundle_state_ptrs_index(mass, type_j, status_j, particle_j);
    const double* press_j = Particle::bundle_state_ptrs_index(bpress, type_j, status_j, particle_j);
    const double* vel_j =
        Particle::bundle_state_ptrs_index(bvel, type_j, status_j, particle_j, statedim);

    double temp_dens(0.0);
    temp_dens = equationofstate_i->pressure_to_density(press_j[0], material_i->initDensity_);
    const double* dens_j = &temp_dens;

    // get smoothing length
    const double h_i = kernel_->smoothing_length(rad_i[0]);

    // evaluate transition factor above reference temperature
    double tempfac_i = 1.0;
    if (trans_d_t_intvisc_ > 0.0)
      tempfac_i = ParticleUtils::comp_lin_trans(
          temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_d_t_intvisc_);

    // compute artificial viscosity
    const double artvisc_i =
        ParticleUtils::vec_norm_two(cfg_i) * h_i * artvisc_lg_int_ + tempfac_i * artvisc_sl_int_;

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
