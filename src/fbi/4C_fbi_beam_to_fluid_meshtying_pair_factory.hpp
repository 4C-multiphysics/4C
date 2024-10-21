#ifndef FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_FACTORY_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_FACTORY_HPP

#include "4C_config.hpp"

#include <Teuchos_RCP.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  class Element;
}

namespace BEAMINTERACTION
{
  class BeamContactPair;
}  // namespace BEAMINTERACTION
namespace FBI
{
  class BeamToFluidMeshtyingParams;
  /**
   *  \brief Factory that creates the appropriate beam to fluid meshtying pair for the desired
   * discretization
   *
   */
  class PairFactory
  {
   private:
    /// constructor
    PairFactory() = delete;

   public:
    /**
     *  \brief Creates the appropriate beam to fluid meshtying pair for the desired
     * discretization
     *
     * This function is static so that it can be called without creating a factory object first.
     * It can be called directly.
     *
     * \param[in] params_ptr Container containing the Fluid beam interaction parameters
     * \param[in] ele_ptrs Pointer containing the elements which make up a pair
     *
     * \return Beam contact pair
     */
    static Teuchos::RCP<BEAMINTERACTION::BeamContactPair> create_pair(
        std::vector<Core::Elements::Element const*> const& ele_ptrs,
        FBI::BeamToFluidMeshtyingParams& params_ptr);
  };
}  // namespace FBI

FOUR_C_NAMESPACE_CLOSE

#endif
