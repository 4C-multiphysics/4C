/*----------------------------------------------------------------------*/
/*! \file

\brief Evaluation of general fluid parameter for fluid in poroelast problem

FluidEleParameter::SetElementPoroParameter(Teuchos::ParameterList& params)
set all general porofluid parameter once for all elements.

\level 2


*/
/*----------------------------------------------------------------------*/

#include "fluid_ele_parameter_poro.H"

DRT::ELEMENTS::FluidEleParameterPoro* DRT::ELEMENTS::FluidEleParameterPoro::Instance(
    CORE::UTILS::SingletonAction action)
{
  static auto singleton_owner = CORE::UTILS::MakeSingletonOwner(
      []()
      {
        return std::unique_ptr<DRT::ELEMENTS::FluidEleParameterPoro>(
            new DRT::ELEMENTS::FluidEleParameterPoro());
      });

  return singleton_owner.Instance(action);
}

//----------------------------------------------------------------------*/
//    constructor
//----------------------------------------------------------------------*/
DRT::ELEMENTS::FluidEleParameterPoro::FluidEleParameterPoro()
    : DRT::ELEMENTS::FluidEleParameter::FluidEleParameter(),
      set_fluid_parameter_poro_(false),
      poro_conti_partint_(false),
      stab_biot_(false),
      stab_biot_scaling_(0.0),
      transient_terms_(INPAR::POROELAST::transient_all)
{
}

//----------------------------------------------------------------------*
//  set poro parameters                                      vuong 11/12|
//---------------------------------------------------------------------*/
void DRT::ELEMENTS::FluidEleParameterPoro::SetElementPoroParameter(
    Teuchos::ParameterList& params, int myrank)
{
  SetElementGeneralFluidParameter(params, myrank);

  set_fluid_parameter_poro_ = true;
  poro_conti_partint_ = params.get<bool>("conti partial integration", false);
  reaction_ = true;
  transient_terms_ = DRT::INPUT::get<INPAR::POROELAST::TransientEquationsOfPoroFluid>(
      params, "Transient Terms Poro Fluid");

  // ---------------------------------------------------------------------
  // get control parameters for stabilization and higher-order elements
  //----------------------------------------------------------------------
  if (stabtype_ == INPAR::FLUID::stabtype_residualbased)
  {
    Teuchos::ParameterList& stablist = params.sublist("RESIDUAL-BASED STABILIZATION");
    stab_biot_ = DRT::INPUT::IntegralValue<int>(stablist, "STAB_BIOT");
    stab_biot_scaling_ = stablist.get<double>("STAB_BIOT_SCALING");
  }
  else if (stabtype_ == INPAR::FLUID::stabtype_nostab)
  {
    stab_biot_ = false;
    stab_biot_scaling_ = 0.0;
  }
}

//----------------------------------------------------------------------*/
// print fluid parameter to screen                          rauch 11/13 |
//----------------------------------------------------------------------*/
void DRT::ELEMENTS::FluidEleParameterPoro::PrintFluidParameterPoro() const
{
  std::cout << std::endl
            << "|-----------------------------------------------------------------------------"
            << std::endl;
  std::cout << "|  Poro Fluid parameter: " << std::endl;
  std::cout << "|-----------------------------------------------------------------------------"
            << std::endl;
  // flag SetGeneralParameter was called
  std::cout << "|    method SetElementParameterPoro was called:    " << set_fluid_parameter_poro_
            << std::endl;
  // flag to (de)activate stationary formulation
  std::cout << "|    Partial integration of conti equation:    " << poro_conti_partint_
            << std::endl;
  // type of handling transient terms
  std::cout << "|   type of handling transient terms:  " << transient_terms_ << std::endl;
  // flag to (de)activate Newton linearization
  std::cout << "|    Type of stabilization:    " << stabtype_ << std::endl;

  std::cout << "|---------------------------------------------------------------------------"
            << std::endl;
}
