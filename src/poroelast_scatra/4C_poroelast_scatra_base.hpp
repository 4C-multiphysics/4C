/*----------------------------------------------------------------------*/
/*! \file

 \brief  base class for all poroelasticity scalar transport interaction algorithms

\level 2

 *----------------------------------------------------------------------*/


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
namespace ADAPTER
{
  class ScaTraBaseAlgorithm;
  class MortarVolCoupl;
  class FluidPoro;
  class FPSIStructureWrapper;
}  // namespace ADAPTER

// namespace POROELAST
//{
//  class PoroBase;
//}

/*----------------------------------------------------------------------*
 |                                                                       |
 *----------------------------------------------------------------------*/
namespace POROELASTSCATRA
{
  /// base class of algorithms for scalar transport in porous media
  class PoroScatraBase : public ADAPTER::AlgorithmBase
  {
   public:
    /// create using a Epetra_Comm
    explicit PoroScatraBase(const Epetra_Comm& comm,
        const Teuchos::ParameterList& timeparams);  // Problem builder

    //! Main time loop.
    virtual void Timeloop() = 0;

    //! prepare time step for single fields
    virtual void prepare_time_step(bool printheader = true)
    {
      FOUR_C_THROW("not implemented in base class. override in subclass.");
    };

    //! perform iteration loop between fields
    virtual void Solve() { FOUR_C_THROW("not implemented in base class. override in subclass."); };

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
    virtual void SetupSystem();

    //! Build the combined dirichlet map of the monolithic poro problem
    virtual void build_combined_dbc_map() { poro_->build_combined_dbc_map(); };

    //! perform result test
    void TestResults(const Epetra_Comm& comm);

    //! apply solution of poro-problem to scatra
    void SetPoroSolution();

    //! apply solution of scatra to poro
    void SetScatraSolution();

    //! return pointer to porous medium problem
    const Teuchos::RCP<POROELAST::PoroBase>& poro_field() { return poro_; };

    //! return pointer to interstitial fluid
    const Teuchos::RCP<ADAPTER::FluidPoro>& fluid_field() { return poro_->fluid_field(); };

    //! return pointer to porous structure
    const Teuchos::RCP<ADAPTER::FPSIStructureWrapper>& structure_field()
    {
      return poro_->structure_field();
    };

    //! return pointer to scalar transport problem
    Teuchos::RCP<SCATRA::ScaTraTimIntImpl> ScaTraField() { return scatra_->ScaTraField(); };

    //! return pointer to scalar problem adapter base class
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> ScaTraFieldBase() { return scatra_; };

    //! setup solver (for monolithic only)
    virtual bool SetupSolver() { return true; };

   protected:
    //! setup up of dofsets for two way coupling
    void replace_dof_sets(Teuchos::RCP<DRT::Discretization> structdis,
        Teuchos::RCP<DRT::Discretization> fluiddis, Teuchos::RCP<DRT::Discretization> scatradis);

    //! setup up coupling objects if necessary
    void setup_coupling(Teuchos::RCP<DRT::Discretization> structdis,
        Teuchos::RCP<DRT::Discretization> fluiddis, Teuchos::RCP<DRT::Discretization> scatradis);

    //! Pointer to the porous media problem. (poroelastic)
    Teuchos::RCP<POROELAST::PoroBase> poro_;
    //! Pointer to the ScaTra problem.     (scatra)
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra_;

    //! @name Volume Mortar stuff

    //! flag for matchinggrid
    const bool matchinggrid_;

    //! volume coupling (using mortar) adapter
    Teuchos::RCP<CORE::ADAPTER::MortarVolCoupl> volcoupl_structurescatra_;
    Teuchos::RCP<CORE::ADAPTER::MortarVolCoupl> volcoupl_fluidscatra_;
    //@}

   private:
    //! apply displacement fields to scatra
    void set_mesh_disp();
    //! apply velocity fields to scatra
    void set_velocity_fields();
  };
}  // namespace POROELASTSCATRA


FOUR_C_NAMESPACE_CLOSE

#endif