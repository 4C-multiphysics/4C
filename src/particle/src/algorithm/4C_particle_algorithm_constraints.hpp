// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_CONSTRAINTS_HPP
#define FOUR_C_PARTICLE_ALGORITHM_CONSTRAINTS_HPP

#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <mpi.h>

FOUR_C_NAMESPACE_OPEN

namespace Particle
{
  /*!
   * \brief constraints handler for particle simulations
   *
   */
  class ConstraintsHandler
  {
   public:
    virtual ~ConstraintsHandler() {};

    /*!
     * \brief Apply constraints to all the particles of the provided types in a bundle.
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    virtual void apply(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate, const double time) const = 0;
  };

  class ConstraintsProjectionBase : public ConstraintsHandler
  {
   protected:
    /*!
     * \brief Protected ctor, such that only children could call it.
     *
     *
     * \param[in] comm MPI communicator
     * \param[in] tol zero threshold
     */
    ConstraintsProjectionBase(MPI_Comm comm, double tol = 1e-12);

   public:
    /*!
     * \brief Apply constraints to all the particles of the provided types in a bundle.
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    void apply(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate, const double time) const override;

   private:
    /*!
     * \brief Perform setup step. Basically, we identify the primary axis for the projection
     * (applies to both 2D and 1D cases) and then verify if all the relevant particles considered
     * for integration respect this projection.
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    void setup(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate) const;

    /*!
     * \brief Identify the primary axis for the projection (applies to both 2D and 1D cases). Calls
     * the local function for each MPI rank and then collects and verifies information globally.
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    int calc_primary_axis(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate) const;

    /*!
     * \brief Check if all the particles are properly positioned in the initial configuration with
     * the respect to the imposed projection constraint (2D or 1D).
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    void check_particles(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate) const;

    /*!
     * \brief Identify the primary axis for the projection within an MPI rank.
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    virtual int calc_primary_axis_local(
        Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate) const = 0;

    /*!
     * \brief Set the primary axis.
     *
     *
     * \param[in] new_axis axis index (0, 1 or 2)
     */
    virtual void set_primary_axis(int new_axis) const = 0;

    /*!
     * \brief Restrain the particle state with respect to the defined constraint.
     *
     *
     * \param[in] n_particle_stored number of particles stored in a bundle
     * \param[in] pos_state_dim offset for the state vectors
     * \param[out] vel pointer to the velocity state vector
     * \param[out] acc pointer to the acceleration state vector
     * \param[out] modvel pointer to the modified velocity state vector
     * \param[out] modacc pointer to the modified acceleration state vector
     */
    virtual void project(int n_particle_stored, int pos_state_dim, double* vel, double* acc,
        double* modvel, double* modacc) const = 0;

    /*!
     * \brief Check the particle positions with respect to the defined projection constraint.
     *
     *
     * \param[in] n_particle_stored number of particles stored in a bundle
     * \param[in] pos_state_dim offset for the position state vector
     * \param[in] pos pointer to the position state vector
     * \param[in] rad pointer to the radius vector
     */
    virtual void check(
        int n_particle_stored, int pos_state_dim, const double* pos, const double* rad) const = 0;

   protected:
    MPI_Comm comm_;

    const double eps_;

    static constexpr int axis_invalid_ = -1;

    mutable int axis_ = axis_invalid_;
  };

  class ConstraintsProjection2D : public ConstraintsProjectionBase
  {
   public:
    /*!
     * \brief Constructor to create the 2D projection constraint onto a plane.
     *
     *
     * \param[in] comm MPI communicator
     */
    ConstraintsProjection2D(MPI_Comm comm);

   private:
    /*!
     * \brief Identify the primary axis for the projection within an MPI rank.
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    int calc_primary_axis_local(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate) const override;

    /*!
     * \brief Set the primary axis. The provided axis index is used then to nullify the
     * corresponding components of the state vectors when the 2D projection constraint is imposed.
     *
     *
     * \param[in] new_axis axis index (0, 1 or 2)
     */
    void set_primary_axis(int new_axis) const override;

    /*!
     * \brief Restrain the particle state by nullifying one of the X,Y,Z components according to the
     * identified primary axis index.
     *
     *
     * \param[in] n_particle_stored number of particles stored in a bundle
     * \param[in] pos_state_dim offset for the state vectors
     * \param[out] vel pointer to the velocity state vector
     * \param[out] acc pointer to the acceleration state vector
     * \param[out] modvel pointer to the modified velocity state vector
     * \param[out] modacc pointer to the modified acceleration state vector
     */
    void project(int n_particle_stored, int pos_state_dim, double* vel, double* acc, double* modvel,
        double* modacc) const override;

    /*!
     * \brief Check if all the particle are positioned within the same plane whose normal is defined
     * by the primary axis.
     *
     *
     * \param[in] n_particle_stored number of particles stored in a bundle
     * \param[in] pos_state_dim offset for the position state vector
     * \param[in] pos pointer to the position state vector
     * \param[in] rad pointer to the radius vector
     */
    void check(int n_particle_stored, int pos_state_dim, const double* pos,
        const double* rad) const override;
  };

  class ConstraintsProjection1D : public ConstraintsProjectionBase
  {
   public:
    /*!
     * \brief Constructor to create the 1D projection constraint onto an axis.
     *
     *
     * \param[in] comm MPI communicator
     */
    ConstraintsProjection1D(MPI_Comm comm);

   private:
    /*!
     * \brief Identify the primary axis for the projection within an MPI rank.
     *
     *
     * \param[in] particle_container_bundle particle container bundle
     * \param[in] types_to_integrate types to integrate
     */
    int calc_primary_axis_local(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::Type>& types_to_integrate) const override;

    /*!
     * \brief Set the primary axis. This method gets the index of the new axis and stores the
     * remaining two indices which will nullify the corresponding state quantities when the 1D
     * projection constraint is imposed.
     *
     *
     * \param[in] new_axis axis index (0, 1 or 2)
     */
    void set_primary_axis(int new_axis) const override;

    /*!
     * \brief Restrain the particle state by nullifying two of the X,Y,Z components according to the
     * identified primary axis index.
     *
     *
     * \param[in] n_particle_stored number of particles stored in a bundle
     * \param[in] pos_state_dim offset for the state vectors
     * \param[out] vel pointer to the velocity state vector
     * \param[out] acc pointer to the acceleration state vector
     * \param[out] modvel pointer to the modified velocity state vector
     * \param[out] modacc pointer to the modified acceleration state vector
     */
    void project(int n_particle_stored, int pos_state_dim, double* vel, double* acc, double* modvel,
        double* modacc) const override;

    /*!
     * \brief Check if all the particle are positioned on the primary axis.
     *
     *
     * \param[in] n_particle_stored number of particles stored in a bundle
     * \param[in] pos_state_dim offset for the position state vector
     * \param[in] pos pointer to the position state vector
     * \param[in] rad pointer to the radius vector
     */
    void check(int n_particle_stored, int pos_state_dim, const double* pos,
        const double* rad) const override;

    mutable int index0_;
    mutable int index1_;
  };

  /*!
   * \brief Factory method to create constraints.
   *
   *
   * \param[in] params input parameters which defines the selection of constraints
   * \param[in] comm MPI comunicator
   */
  std::unique_ptr<Particle::ConstraintsHandler> create_constraints(
      const Teuchos::ParameterList& params, MPI_Comm comm);

}  // namespace Particle

FOUR_C_NAMESPACE_CLOSE

#endif
