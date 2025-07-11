// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_INDIRECT_HPP
#define FOUR_C_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_INDIRECT_HPP


#include "4C_config.hpp"

#include "4C_fbi_partitioned_penaltycoupling_assembly_manager.hpp"

FOUR_C_NAMESPACE_OPEN


// Forward declaration.
namespace BeamInteraction
{
  class BeamToFluidMortarManager;
}  // namespace BeamInteraction

namespace FBI
{
  class BeamToFluidMeshtyingParams;
}
namespace BeamInteraction
{
  namespace SubmodelEvaluator
  {
    /**
     * \brief This class collects local mortar coupling terms of the pairs (D and M) and assembles
     * them into the global coupling matrices. Those global coupling matrices are then multiplied
     * with each other and added to the global force vector and stiffness matrix.
     */
    class PartitionedBeamInteractionAssemblyManagerIndirect
        : public PartitionedBeamInteractionAssemblyManager
    {
     public:
      /**
       * \brief Constructor.
       * @param assembly_contact_elepairs (in) Vector with element pairs to be evaluated by this
       * class.
       */
      PartitionedBeamInteractionAssemblyManagerIndirect(
          std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& assembly_contact_elepairs,
          std::shared_ptr<const Core::FE::Discretization>& discretization1,
          std::shared_ptr<const Core::FE::Discretization>& discretization2,
          std::shared_ptr<FBI::BeamToFluidMeshtyingParams> beam_contact_params_ptr);

      /**
       * \brief Evaluate all force and stiffness terms and add them to the global matrices.
       * @param discret (in) Pointer to the disretization.
       * @param fe_sysvec (out) Global force vector.
       * @param fe_sysmat (out) Global stiffness matrix.
       * @param disp (in) Current displacement vector.
       */
      void evaluate_force_stiff(const Core::FE::Discretization& discretization1,
          const Core::FE::Discretization& discretization2,
          std::shared_ptr<Core::LinAlg::FEVector<double>>& ff,
          std::shared_ptr<Core::LinAlg::FEVector<double>>& fb,
          std::shared_ptr<Core::LinAlg::SparseOperator> cff,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& cbb,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& cfb,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& cbf,
          std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vel,
          std::shared_ptr<const Core::LinAlg::Vector<double>> beam_vel) override;

      /**
       * \brief Return a const pointer to the mortar manager.
       */
      inline std::shared_ptr<const BeamInteraction::BeamToFluidMortarManager> get_mortar_manager()
          const
      {
        return mortar_manager_;
      }

     private:
      //! Pointer to the mortar manager. This object stores the relevant mortar matrices.
      std::shared_ptr<BeamInteraction::BeamToFluidMortarManager> mortar_manager_;
    };

  }  // namespace SubmodelEvaluator
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
