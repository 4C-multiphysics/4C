/*----------------------------------------------------------------------*/
/*! \file
\brief Interface for all scatra adapters.
\level 1
 */
/*----------------------------------------------------------------------*/


#ifndef FOUR_C_ADAPTER_SCATRA_INTERFACE_HPP
#define FOUR_C_ADAPTER_SCATRA_INTERFACE_HPP


#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace ScaTra
{
  class MeshtyingStrategyBase;
}


namespace Adapter
{
  /*! \brief General pure virtual interface for all scatra time integrators and scatra adapters.
   *
   *  The point is to keep coupled problems as far apart from our field solvers as
   *  possible. Each scatra field solver we want to use should get its own subclass
   *  of this. The coupled algorithm should be able to extract all the information
   *  from the scatra field it needs using this interface.
   *
   * \sa ScaTraTimIntImpl
   * \date 12/2016
   */
  class ScatraInterface
  {
   public:
    //! constructor
    ScatraInterface(){};

    //! virtual to get polymorph destruction
    virtual ~ScatraInterface() = default;

    //! return discretization
    virtual Teuchos::RCP<Core::FE::Discretization> discretization() const = 0;

    //! add parameters specific for time-integration scheme
    virtual void add_time_integration_specific_vectors(bool forcedincrementalsolver = false) = 0;

    //! return number of dofset associated with displacement dofs
    virtual int nds_disp() const = 0;

    //! return rcp ptr to neumann loads vector
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> get_neumann_loads_ptr() = 0;

    //! return meshtying strategy (includes standard case without meshtying)
    virtual const Teuchos::RCP<ScaTra::MeshtyingStrategyBase>& strategy() const = 0;

    //! return scalar field phi at time n
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> phin() = 0;

  };  // class ScatraInterface
}  // namespace Adapter


FOUR_C_NAMESPACE_CLOSE

#endif
