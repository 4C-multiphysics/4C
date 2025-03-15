// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PASI_PARTITIONED_HPP
#define FOUR_C_PASI_PARTITIONED_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Adapter
{
  class StructureBaseAlgorithmNew;
  class PASIStructureWrapper;
}  // namespace Adapter

namespace PARTICLEALGORITHM
{
  class ParticleAlgorithm;
}

namespace Solid
{
  class MapExtractor;
}  // namespace Solid

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace PaSI
{
  /*!
   * \brief partitioned algorithm for particle structure interaction
   *
   * An abstract base class for partitioned particle structure interaction problems defining
   * methods and holding members to be used in derived algorithms.
   *
   */
  class PartitionedAlgo : public Adapter::AlgorithmBase
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle structure interaction parameter list
     */
    explicit PartitionedAlgo(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief init pasi algorithm
     *
     */
    virtual void init();

    /*!
     * \brief setup pasi algorithm
     *
     */
    virtual void setup();

    /*!
     * \brief Perform all necessary tasks after initializing the PaSI
     * algorithm. Currently, this only calls the post_setup routine of the
     * structural field.
     *
     */
    void post_init();

    /*!
     * \brief read restart information for given time step
     *
     *
     * \param[in] restartstep restart step
     */
    void read_restart(int restartstep) override;

    /*!
     * \brief timeloop of coupled problem
     *
     */
    virtual void timeloop() = 0;

    /*!
     * \brief perform result tests
     *
     *
     * \param[in] comm communicator
     */
    void test_results(MPI_Comm comm);

    //! get initialization status
    bool is_init() { return isinit_; };

    //! get setup status
    bool is_setup() { return issetup_; };

   protected:
    /*!
     * \brief prepare time step
     *
     *
     * \param[in] printheader flag to control output of time step header
     */
    void prepare_time_step(bool printheader = true);

    /*!
     * \brief pre evaluate time step
     *
     */
    void pre_evaluate_time_step();

    /*!
     * \brief structural time step
     *
     */
    void struct_step();

    /*!
     * \brief particle time step
     *
     */
    void particle_step();

    /*!
     * \brief post evaluate time step
     *
     */
    void post_evaluate_time_step();

    /*!
     * \brief extract interface states
     *
     * Extract the interface states displacement, velocity, and acceleration from the structural
     * states.
     *
     */
    void extract_interface_states();

    /*!
     * \brief set interface states
     *
     * Set the interface states displacement, velocity, and acceleration as handed in to the
     * particle wall handler. This includes communication, since the structural discretization and
     * the particle wall discretization are in general distributed independently of each other to
     * all processors.
     *
     *
     * \param[in] intfdispnp interface displacement
     * \param[in] intfvelnp  interface velocity
     * \param[in] intfaccnp  interface acceleration
     */
    void set_interface_states(std::shared_ptr<const Core::LinAlg::Vector<double>> intfdispnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> intfvelnp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> intfaccnp);

    /*!
     * \brief output of structure field
     *
     */
    void struct_output();

    /*!
     * \brief output of particle field
     *
     */
    void particle_output();

    //! check correct setup
    void check_is_setup()
    {
      if (not is_setup()) FOUR_C_THROW("pasi algorithm not setup correctly!");
    };

    //! check correct initialization
    void check_is_init()
    {
      if (not is_init()) FOUR_C_THROW("pasi algorithm not initialized correctly!");
    };

    //! structural field
    std::shared_ptr<Adapter::PASIStructureWrapper> structurefield_;

    //! particle algorithm
    std::shared_ptr<PARTICLEALGORITHM::ParticleAlgorithm> particlealgorithm_;

    //! communication object at the interface
    std::shared_ptr<const Solid::MapExtractor> interface_;

    //! interface displacement
    std::shared_ptr<Core::LinAlg::Vector<double>> intfdispnp_;

    //! interface velocity
    std::shared_ptr<Core::LinAlg::Vector<double>> intfvelnp_;

    //! interface acceleration
    std::shared_ptr<Core::LinAlg::Vector<double>> intfaccnp_;

   private:
    /*!
     * \brief init structure field
     *
     */
    void init_structure_field();

    /*!
     * \brief init particle algorithm
     *
     */
    void init_particle_algorithm();

    /*!
     * \brief build and register structure model evaluator
     *
     */
    void build_structure_model_evaluator();

    //! ptr to the underlying structure problem base algorithm
    std::shared_ptr<Adapter::StructureBaseAlgorithmNew> struct_adapterbase_ptr_;

    //! flag indicating correct initialization
    bool isinit_;

    //! flag indicating correct setup
    bool issetup_;

    //! set flag indicating correct initialization
    void set_is_init(bool isinit) { isinit_ = isinit; };

    //! set flag indicating correct setup
    void set_is_setup(bool issetup) { issetup_ = issetup; };
  };

}  // namespace PaSI

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
