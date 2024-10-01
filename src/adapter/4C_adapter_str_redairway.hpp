/*----------------------------------------------------------------------*/
/*! \file

\brief Structure field adapter for coupling with reduced-D airway trees


\level 3

*/

/*----------------------------------------------------------------------*/
/* macros */


#ifndef FOUR_C_ADAPTER_STR_REDAIRWAY_HPP
#define FOUR_C_ADAPTER_STR_REDAIRWAY_HPP
/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_adapter_str_wrapper.hpp"
#include "4C_fem_condition.hpp"
#include "4C_linalg_vector.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Adapter
{
  class StructureRedAirway : public StructureWrapper
  {
   public:
    /// Constructor
    StructureRedAirway(Teuchos::RCP<Structure> stru);

    /// set pressure calculated from reduced-d airway tree
    void set_pressure(Teuchos::RCP<Core::LinAlg::Vector<double>> couppres);

    /// calculate outlet fluxes for reduced-d airway tree
    void calc_flux(Teuchos::RCP<Core::LinAlg::Vector<double>> coupflux,
        Teuchos::RCP<Core::LinAlg::Vector<double>> coupvol, double dt);

    /// calculate volume
    void calc_vol(std::map<int, double>& V);

    /// calculate initial volume
    void init_vol();

    //! (derived)
    void update() override;

   private:
    /// map between coupling ID and conditions on structure
    std::map<int, Core::Conditions::Condition*> coupcond_;

    /// map of coupling IDs
    Teuchos::RCP<Epetra_Map> coupmap_;

    std::map<int, double> vn_;
    std::map<int, double> vnp_;
  };

}  // namespace Adapter
FOUR_C_NAMESPACE_CLOSE

#endif
