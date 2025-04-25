// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_BEAM_TO_FLUID_ASSEMBLY_MANAGER_FACTORY_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_ASSEMBLY_MANAGER_FACTORY_HPP

#include "4C_config.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace FBI
{
  class BeamToFluidMeshtyingParams;
  namespace Utils
  {
    class FBIAssemblyStrategy;
  }
}  // namespace FBI
namespace BeamInteraction
{
  class BeamContactPair;

  namespace SubmodelEvaluator
  {
    class PartitionedBeamInteractionAssemblyManager;
  }
  // namespace BeamInteraction
  /**
   *  \brief Factory that creates the appropriate beam to fluid meshtying assembly manager for the
   * desired discretization
   *
   */
  class BeamToFluidAssemblyManagerFactory
  {
   private:
    /// constructor
    BeamToFluidAssemblyManagerFactory() = delete;

   public:
    /**
     *  \brief Creates the appropriate beam to fluid meshtying assembly manager for the desired
     * discretizations
     *
     * This function is static so that it can be called without creating a factory object first.
     * It can be called directly.
     *
     * \param[in] params_ptr Container containing the Fluid beam interaction parameters
     * \param[in] interaction_pairs Vector of possible fluid beam interaction pairs
     * \param[in] assemblystrategy object handling the assembly into the global fluid matrix
     *
     * \return beam interaction assembly manager
     */
    static std::shared_ptr<
        BeamInteraction::SubmodelEvaluator::PartitionedBeamInteractionAssemblyManager>
    create_assembly_manager(std::shared_ptr<const Core::FE::Discretization> discretization1,
        std::shared_ptr<const Core::FE::Discretization> discretization2,
        std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> interaction_pairs,
        const std::shared_ptr<FBI::BeamToFluidMeshtyingParams> params_ptr,
        std::shared_ptr<FBI::Utils::FBIAssemblyStrategy> assemblystrategy);
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
