/*----------------------------------------------------------------------*/
/*! \file

\brief Fluid field adapter for xfem-ale-fluids with moving boundaries

\level 2

*/
/*----------------------------------------------------------------------*/


#ifndef FOUR_C_ADAPTER_FLD_FLUID_ALE_XFEM_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_ALE_XFEM_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_fluid_ale.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{

  /// fluid with moving interfaces implemented by the XFEM
  class FluidAleXFEM : public FluidAle
  {
   public:
    /// constructor
    explicit FluidAleXFEM(const Teuchos::ParameterList& prbdyn, std::string condname);

    /*========================================================================*/
    //! @name Misc
    /*========================================================================*/

    /// return the boundary discretization that matches the structure discretization
    Teuchos::RCP<Core::FE::Discretization> boundary_discretization();

    /// communication object at the struct interface
    virtual Teuchos::RCP<FLD::UTILS::MapExtractor> const& struct_interface();

    //@}

    /*========================================================================*/
    //! @name Solver calls
    /*========================================================================*/

    /// nonlinear solve
    void nonlinear_solve(
        Teuchos::RCP<Core::LinAlg::Vector> idisp, Teuchos::RCP<Core::LinAlg::Vector> ivel) override;

    /// relaxation solve
    Teuchos::RCP<Core::LinAlg::Vector> relaxation_solve(
        Teuchos::RCP<Core::LinAlg::Vector> idisp, double dt) override;
    //@}

    /*========================================================================*/
    //! @name Extract interface forces
    /*========================================================================*/

    /// After the fluid solve we need the forces at the FSI interface.
    Teuchos::RCP<Core::LinAlg::Vector> extract_interface_forces() override;
    //@}

    /*========================================================================*/
    //! @name extract helpers
    /*========================================================================*/

    /// extract the interface velocity at time t^(n+1)
    Teuchos::RCP<Core::LinAlg::Vector> extract_interface_velnp() override;

    /// extract the interface velocity at time t^n
    Teuchos::RCP<Core::LinAlg::Vector> extract_interface_veln() override;
    //@}
    //@}
  };

}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
