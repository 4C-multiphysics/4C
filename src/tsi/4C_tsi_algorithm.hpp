// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_TSI_ALGORITHM_HPP
#define FOUR_C_TSI_ALGORITHM_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 12/09 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | forward declarations                                      dano 02/12 |
 *----------------------------------------------------------------------*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Global
{
  class Problem;
}

namespace CONTACT
{
  class LagrangeStrategyTsi;
  class NitscheStrategyTsi;
}  // namespace CONTACT

namespace Adapter
{
  class Structure;
  class MortarVolCoupl;
}  // namespace Adapter

namespace Thermo
{
  class Adapter;
}

namespace Mortar
{
  class MultiFieldCoupling;
}


/*----------------------------------------------------------------------*
 |                                                           dano 12/09 |
 *----------------------------------------------------------------------*/
namespace TSI
{
  //! TSI algorithm base
  //!
  //!
  //!  Base class of TSI algorithms. Derives from structure_base_algorithm and
  //!  Thermo::BaseAlgorithm with temperature field.
  //!  There can (and will) be different subclasses that implement different
  //!  coupling schemes.
  //!
  //!  \warning The order of calling the two BaseAlgorithm-constructors (that
  //!  is the order in which we list the base classes) is important here! In the
  //!  constructors control file entries are written. And these entries define
  //!  the order in which the filters handle the Discretizations, which in turn
  //!  defines the dof number ordering of the Discretizations... Don't get
  //!  confused. Just always list structure, thermo. In that order.
  class Algorithm : public Adapter::AlgorithmBase
  {
   public:
    explicit Algorithm(MPI_Comm comm);


    //! outer level time loop
    void time_loop();

    /// initialise TSI system
    virtual void setup_system() = 0;

    /*!
     * @brief Perform all necessary tasks after setting up the TSI
     * algorithm. Currently, this only calls the post_setup method of the
     * structure_field.
     */
    void post_setup();

    /// non-linear solve, i.e. (multiple) corrector
    virtual void solve() = 0;

    //! read restart data
    void read_restart(int step  //!< step number where the calculation is continued
        ) override = 0;

    //! access to structural field
    const std::shared_ptr<Adapter::Structure>& structure_field() { return structure_; }

    //! access to thermal field
    const std::shared_ptr<Thermo::Adapter>& thermo_field() { return thermo_; }

    //! calculate stresses, strains, energies
    virtual void prepare_output(bool force_prepare) = 0;

    //! calculate stresses, strains, energies without forcing runtime preparation
    void prepare_output() { prepare_output(false); }

    //! write output
    virtual void output(bool forced_writerestart = false);

   protected:
    //! @name Time loop building blocks

    //! start a new time step
    void prepare_time_step() override = 0;

    //! take current results for converged and save for next time step
    void update() override = 0;

    //@}

    //! @name Transfer methods

    //! apply temperature state on structure discretization
    void apply_thermo_coupling_state(std::shared_ptr<const Core::LinAlg::Vector<double>> temp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> temp_res = nullptr);

    //! apply structural displacements and velocities on thermo discretization
    void apply_struct_coupling_state(std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel);

    //! Prepare a ptr to the contact strategy from the structural field,
    //! store it in tsi and hand it to the thermal field
    void prepare_contact_strategy();

    /*! \brief Deactivate the thermo elements of inactive structural elements
     *
     *  Call this after the update of the structural field (where structural elements are eroded)
     *  and before the update of the thermal field. Thermo nodes which are only connected to
     *  inactive elements are constrained. After a time step (@p initialization is false), the
     *  thermal field is evaluated again at the current state if thermo elements have been
     *  deactivated, such that their contributions are removed before the thermal update. At the
     *  start or after a restart (@p initialization is true), the thermal field is adapted to the
     *  inactive elements in any case.
     */
    void adapt_to_structure_active_flags(bool initialization);

    //! Access to the dof coupling for matching grid TSI
    Coupling::Adapter::Coupling& structure_thermo_coupling() { return *structure_thermo_coupling_; }
    //@}

    //! @name Access methods

    //! velocity calculation given the displacements (like in FSI)
    std::shared_ptr<const Core::LinAlg::Vector<double>> calc_velocity(
        const Core::LinAlg::Vector<double>& dispnp);

    //! displacements at time n+1 for thermal output
    std::shared_ptr<Core::LinAlg::MultiVector<double>> dispnp_;

    //! temperatures at time n+1 for structure output
    //! introduced for non-matching discretizations
    std::shared_ptr<Core::LinAlg::MultiVector<double>> tempnp_;

    //@}


    //! @name Underlying fields

    //! underlying structure of the FSI problem
    std::shared_ptr<Adapter::Structure> structure_;

    //! underlying fluid of the FSI problem
    std::shared_ptr<Thermo::Adapter> thermo_;

    //! contact strategies
    std::shared_ptr<CONTACT::LagrangeStrategyTsi> contact_strategy_lagrange_;
    //@}

    //! @name Volume Mortar stuff

    Global::Problem* problem_;

    //! flag for matchinggrid
    const bool matchinggrid_;

    //! whether structural elements can be eroded
    bool can_deactivate_elements_ = false;
    //! volume coupling (using mortar) adapter
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> volcoupl_;

    std::shared_ptr<Coupling::Adapter::Coupling>
        structure_thermo_coupling_;  //! structure (target), thermo (source)
    //@}


    //! @name Surface Mortar stuff
    std::shared_ptr<Mortar::MultiFieldCoupling> mortar_coupling_;

    //@}
  };  // Algorithm
}  // namespace TSI

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE

#endif
