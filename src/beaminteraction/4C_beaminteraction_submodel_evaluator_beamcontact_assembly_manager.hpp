/*-----------------------------------------------------------*/
/*! \file

\brief Class to assemble pair based contributions into global matrices.


\level 3

*/
/*-----------------------------------------------------------*/


#ifndef FOUR_C_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_ASSEMBLY_MANAGER_HPP
#define FOUR_C_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_ASSEMBLY_MANAGER_HPP


#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_FEVector.h>
#include <Teuchos_RCP.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
}
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace BEAMINTERACTION
{
  class BeamContactPair;
}
namespace Solid
{
  namespace ModelEvaluator
  {
    class BeamInteractionDataState;
  }
}  // namespace Solid


namespace BEAMINTERACTION
{
  namespace SUBMODELEVALUATOR
  {
    /**
     * \brief This class assembles the contribution of beam contact pairs into the global force
     * vector and stiffness matrix. The method evaluate_force_stiff has to be overloaded in the
     * derived classes to implement the correct assembly method.
     */
    class BeamContactAssemblyManager
    {
     public:
      /**
       * \brief Constructor.
       */
      BeamContactAssemblyManager();

      /**
       * \brief Destructor.
       */
      virtual ~BeamContactAssemblyManager() = default;

      /**
       * \brief Evaluate all force and stiffness terms and add them to the global matrices.
       * @param discret (in) Pointer to the disretization.
       * @param data_state (in) Beam interaction data state.
       * @param fe_sysvec (out) Global force vector.
       * @param fe_sysmat (out) Global stiffness matrix.
       */
      virtual void evaluate_force_stiff(Teuchos::RCP<Core::FE::Discretization> discret,
          const Teuchos::RCP<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
          Teuchos::RCP<Epetra_FEVector> fe_sysvec,
          Teuchos::RCP<Core::LinAlg::SparseMatrix> fe_sysmat)
      {
        FOUR_C_THROW("Not implemented!");
      }

      virtual double get_energy(const Teuchos::RCP<const Core::LinAlg::Vector<double>>& disp) const
      {
        return 0.0;
      }
    };

  }  // namespace SUBMODELEVALUATOR
}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
