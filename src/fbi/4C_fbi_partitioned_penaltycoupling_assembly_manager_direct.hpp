// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_DIRECT_HPP
#define FOUR_C_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_DIRECT_HPP

#include "4C_config.hpp"

#include "4C_fbi_partitioned_penaltycoupling_assembly_manager.hpp"
#include "4C_linalg_fevector.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FBI
{
  namespace Utils
  {
    class FBIAssemblyStrategy;
  }
}  // namespace FBI

namespace BeamInteraction
{
  namespace SubmodelEvaluator
  {
    /**
     * \brief This class collects local force and stiffness terms of the pairs and adds them
     * directly into the global force vector and stiffness matrix.
     */
    class PartitionedBeamInteractionAssemblyManagerDirect
        : public PartitionedBeamInteractionAssemblyManager
    {
     public:
      /**
       * \brief Constructor.
       * \param[in] assembly_contact_elepairs Vector with element pairs to be evaluated by this
       * class.
       * \param[in] assemblystrategy Object determining how the local matrices are assembled into
       * the global one
       */
      PartitionedBeamInteractionAssemblyManagerDirect(
          const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>
              assembly_contact_elepairs,
          std::shared_ptr<FBI::Utils::FBIAssemblyStrategy> assemblystrategy);

      /**
       * \brief Evaluate all force and stiffness terms and add them to the global matrices.
       * \param[in] fluid_dis (in) Pointer to the fluid disretization
       * \param[in] solid_dis (in) Pointer to the solid disretization
       * \param[inout] ff Global force vector acting on the fluid
       * \param[inout] fb Global force vector acting on the beam
       * \param[inout] cff  Global stiffness matrix coupling fluid to fluid DOFs
       * \param[inout] cbb  Global stiffness matrix coupling beam to beam DOFs
       * \param[inout] cfb  Global stiffness matrix coupling beam to fluid DOFs
       * \param[inout] cbf  Global stiffness matrix coupling fluid to beam DOFs
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

     protected:
      /// Object determining how the local matrices are assembled into the global one
      std::shared_ptr<FBI::Utils::FBIAssemblyStrategy> assemblystrategy_;
    };
  }  // namespace SubmodelEvaluator
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
