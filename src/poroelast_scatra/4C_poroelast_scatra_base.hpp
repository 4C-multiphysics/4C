// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROELAST_SCATRA_BASE_HPP
#define FOUR_C_POROELAST_SCATRA_BASE_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_poroelast_base.hpp"
#include "4C_scatra_algorithm.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | forward declarations                                                  |
 *----------------------------------------------------------------------*/
namespace Adapter
{
  class ScaTraBaseAlgorithm;
  class MortarVolCoupl;
  class FluidPoro;
  class FPSIStructureWrapper;
}  // namespace Adapter

namespace CONTACT
{
  class NitscheStrategySsi;
}

// namespace PoroElast
//{
//  class PoroBase;
//}

/*----------------------------------------------------------------------*
 |                                                                       |
 *----------------------------------------------------------------------*/
namespace PoroElastScaTra
{
  /// base class of algorithms for scalar transport in porous media
  class PoroScatraBase : public Adapter::AlgorithmBase
  {
   public:
    explicit PoroScatraBase(MPI_Comm comm,
        const Teuchos::ParameterList& timeparams);  // Problem builder

    //! Main time loop.
    virtual void timeloop() = 0;

    //! prepare time step for single fields
    virtual void prepare_time_step(bool printheader = true)
    {
      FOUR_C_THROW("not implemented in base class. override in subclass.");
    };

    //! perform iteration loop between fields
    virtual void solve() { FOUR_C_THROW("not implemented in base class. override in subclass."); };

    //! prepare output
    virtual void prepare_output()
    {
      FOUR_C_THROW("not implemented in base class. override in subclass.");
    };

    //! update time step
    void update() override
    {
      FOUR_C_THROW("not implemented in base class. override in subclass.");
    };

    //! write output print to screen
    void output() override
    {
      FOUR_C_THROW("not implemented in base class. override in subclass.");
    };

    //! read and set fields needed for restart
    void read_restart(int restart) override = 0;

    //! setup for single fields
    virtual void setup_system();

    //! Build the combined dirichlet map of the monolithic poro problem
    virtual void build_combined_dbc_map() { poro_->build_combined_dbc_map(); };

    //! perform result test
    void test_results(MPI_Comm comm);

    //! apply solution of poro-problem to scatra
    virtual void set_poro_solution();

    //! apply solution of scatra to poro
    virtual void set_scatra_solution();

    //! return pointer to porous medium problem
    const std::shared_ptr<PoroElast::PoroBase>& poro_field() { return poro_; };

    //! return pointer to interstitial fluid
    const std::shared_ptr<Adapter::FluidPoro>& fluid_field() { return poro_->fluid_field(); };

    //! return pointer to porous structure
    const std::shared_ptr<Adapter::FPSIStructureWrapper>& structure_field()
    {
      return poro_->structure_field();
    };

    //! return pointer to scalar transport problem
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra_field() { return scatra_->scatra_field(); };

    //! return pointer to scalar problem adapter base class
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_field_base() { return scatra_; };

    //! setup solver (for monolithic only)
    virtual bool setup_solver() { return true; };

    //! get bool indicating if we have at least one ssi interface contact condition
    bool ssi_interface_contact() const { return ssiinterfacecontact_; }

   protected:
    //! setup up of dofsets for two way coupling
    void replace_dof_sets(std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<Core::FE::Discretization> fluiddis,
        std::shared_ptr<Core::FE::Discretization> scatradis);

    //! setup up coupling objects if necessary
    void setup_coupling(std::shared_ptr<Core::FE::Discretization> structdis,
        std::shared_ptr<Core::FE::Discretization> fluiddis,
        std::shared_ptr<Core::FE::Discretization> scatradis);

    //! Pointer to the porous media problem. (poroelastic)
    std::shared_ptr<PoroElast::PoroBase> poro_;
    //! Pointer to the ScaTra problem.     (scatra)
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_;

    //! @name Volume Mortar stuff

    //! flag for matchinggrid
    const bool matchinggrid_;

    //! volume coupling (using mortar) adapter
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> volcoupl_structurescatra_;
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> volcoupl_fluidscatra_;
    //@}

   private:
    //! apply displacement fields to scatra
    void set_mesh_disp();
    //! apply velocity fields to scatra
    void set_velocity_fields();

    //! bool indicating if we have at least one ssi interface contact condition
    const bool ssiinterfacecontact_;
  };
}  // namespace PoroElastScaTra


FOUR_C_NAMESPACE_CLOSE

#endif
