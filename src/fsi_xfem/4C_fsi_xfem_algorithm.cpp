/*----------------------------------------------------------------------*/
/*! \file

\brief Basis of monolithic XFSI algorithm that performs a coupling between the
       structural field equation and XFEM fluid field equations

\level 2

*/


#include "4C_fsi_xfem_algorithm.hpp"

#include "4C_adapter_ale.hpp"
#include "4C_adapter_ale_fpsi.hpp"
#include "4C_adapter_fld_base_algorithm.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_adapter_str_poro_wrapper.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_poroelast_monolithic.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN



//! Note: The order of calling the two BaseAlgorithm-constructors is
//! important here! In here control file entries are written. And these entries
//! define the order in which the filters handle the Discretizations, which in
//! turn defines the dof number ordering of the Discretizations.
/*----------------------------------------------------------------------*
 | constructor (public)                                    schott 08/14 |
 *----------------------------------------------------------------------*/
FSI::AlgorithmXFEM::AlgorithmXFEM(const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams,
    const Adapter::FieldWrapper::Fieldtype type)
    : AlgorithmBase(comm, timeparams),
      num_fields_(0),
      structp_block_(-1),
      fluid_block_(-1),
      fluidp_block_(-1),
      ale_i_block_(-1)
{
  // access structural dynamic params list which will be possibly modified while creating the time
  // integrator
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();
  const Teuchos::ParameterList& xfdyn = Global::Problem::instance()->x_fluid_dynamic_params();
  bool ale = xfdyn.sublist("GENERAL").get<bool>("ALE_XFluid");

  num_fields_ += 2;
  structp_block_ = 0;
  fluid_block_ = 1;

  if (type == Adapter::StructurePoroWrapper::type_StructureField)
  {
    // ask base algorithm for the structural time integrator
    // access the structural discretization
    Teuchos::RCP<Core::FE::Discretization> structdis =
        Global::Problem::instance()->get_dis("structure");
    Adapter::StructureBaseAlgorithm structure(
        timeparams, const_cast<Teuchos::ParameterList&>(sdyn), structdis);
    structureporo_ = Teuchos::make_rcp<Adapter::StructurePoroWrapper>(
        structure.structure_field(), Adapter::StructurePoroWrapper::type_StructureField, true);
  }
  else if (type == Adapter::StructurePoroWrapper::type_PoroField)
  {
    num_fields_ += 1;
    fluidp_block_ = 2;

    Global::Problem* problem = Global::Problem::instance();
    const Teuchos::ParameterList& poroelastdyn =
        problem->poroelast_dynamic_params();  // access the problem-specific parameter list
    Teuchos::RCP<PoroElast::Monolithic> poro = Teuchos::rcp_dynamic_cast<PoroElast::Monolithic>(
        PoroElast::Utils::create_poro_algorithm(poroelastdyn, comm, false));
    if (poro == Teuchos::null)  // safety check
      FOUR_C_THROW(
          "Couldn't cast poro to PoroElast::Monolithic --> check your COUPALGO in the "
          "POROELASTICITY DYNAMIC section!");
    if (Teuchos::getIntegralValue<Inpar::PoroElast::SolutionSchemeOverFields>(
            poroelastdyn, "COUPALGO") != Inpar::PoroElast::Monolithic)
      FOUR_C_THROW(
          "You created a different poroelast algorithm than monolithic (not combineable with xfpsi "
          "at the moment)--> check your COUPALGO in the POROELASTICITY DYNAMIC section!");
    structureporo_ = Teuchos::make_rcp<Adapter::StructurePoroWrapper>(
        poro, Adapter::StructurePoroWrapper::type_PoroField, true);
  }
  else
    FOUR_C_THROW("AlgorithmXFEM cannot handle this Fieldtype for structure!");

  if (ale)
  {
    num_fields_ += 1;
    ale_i_block_ = num_fields_ - 1;
    Global::Problem* problem = Global::Problem::instance();
    const Teuchos::ParameterList& fsidynparams = problem->fsi_dynamic_params();
    // ask base algorithm for the ale time integrator
    Adapter::AleBaseAlgorithm ale(fsidynparams, Global::Problem::instance()->get_dis("ale"));
    ale_ = Teuchos::rcp_dynamic_cast<Adapter::AleFpsiWrapper>(ale.ale_field());
    if (ale_ == Teuchos::null)
      FOUR_C_THROW("Cast from Adapter::Ale to Adapter::AleFpsiWrapper failed");
  }
  else
  {
    ale_ = Teuchos::null;
  }


  //--------------------------------------------
  // ask base algorithm for the fluid time integrator
  // do not init in ale case!!! (will be done in MonolithicAFSI_XFEM::Setup System())
  Teuchos::RCP<Adapter::FluidBaseAlgorithm> fluid =
      Teuchos::make_rcp<Adapter::FluidBaseAlgorithm>(timeparams, fdyn, "fluid", ale, false);
  fluid_ = Teuchos::rcp_dynamic_cast<FLD::XFluid>(fluid->fluid_field());
  if (fluid_ == Teuchos::null)
    FOUR_C_THROW("Cast of Fluid to XFluid failed! - Everything fine in setup_fluid()?");
  fluid_->init(false);

  // Do setup of the fields here
  structureporo_->setup();
  return;
}



/*----------------------------------------------------------------------*
 | setup (public)                                            ager 12/16 |
 *----------------------------------------------------------------------*/
void FSI::AlgorithmXFEM::setup()
{
  // Do setup of the fields here
  // structureporo_->setup();
}

/*----------------------------------------------------------------------*
 | update (protected)                                      schott 08/14 |
 *----------------------------------------------------------------------*/
void FSI::AlgorithmXFEM::update()
{
  FOUR_C_THROW("currently unused");

  structure_poro()->update();
  fluid_field()->update();
  if (have_ale()) ale_field()->update();

  return;
}



/*----------------------------------------------------------------------*
 | calculate stresses, strains, energies                   schott 08/14 |
 *----------------------------------------------------------------------*/
void FSI::AlgorithmXFEM::prepare_output(bool force_prepare)
{
  structure_poro()->prepare_output(force_prepare);
}

FOUR_C_NAMESPACE_CLOSE
