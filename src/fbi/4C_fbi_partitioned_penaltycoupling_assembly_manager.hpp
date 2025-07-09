// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_HPP
#define FOUR_C_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_linalg_fevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace BeamInteraction
{
  class BeamContactPair;
}

namespace BeamInteraction
{
  namespace SubmodelEvaluator
  {
    /**
     * \brief This class assembles the contribution of beam contact pairs into the global force
     * vector and stiffness matrix for partitioned algorithms. The method evaluate_force_stiff has
     * to be overloaded in the derived classes to implement the correct assembly method.
     */
    class PartitionedBeamInteractionAssemblyManager
    {
     public:
      /**
       * \brief Constructor.
       * \param[in] assembly_contact_elepairs Vector with element pairs to be evaluated by this
       * class.
       */
      PartitionedBeamInteractionAssemblyManager(
          std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>&
              assembly_contact_elepairs);

      /**
       * \brief Destructor.
       */
      virtual ~PartitionedBeamInteractionAssemblyManager() = default;

      /**
       * \brief Evaluate all force and stiffness terms and add them to the global matrices.
       * \param[in] discret (in) Pointer to the disretization.
       * \param[inout] ff Global force vector acting on the fluid
       * \param[inout] fb Global force vector acting on the beam
       * \param[inout] cff  Global stiffness matrix coupling fluid to fluid DOFs
       * \param[inout] cbb  Global stiffness matrix coupling beam to beam DOFs
       * \param[inout] cfb  Global stiffness matrix coupling beam to fluid DOFs
       * \param[inout] cbf  Global stiffness matrix coupling fluid to beam DOFs
       */
      virtual void evaluate_force_stiff(const Core::FE::Discretization& discretization1,
          const Core::FE::Discretization& discretization2,
          std::shared_ptr<Core::LinAlg::FEVector<double>>& ff,
          std::shared_ptr<Core::LinAlg::FEVector<double>>& fb,
          std::shared_ptr<Core::LinAlg::SparseOperator> cff,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& cbb,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& cfb,
          std::shared_ptr<Core::LinAlg::SparseMatrix>& cbf,
          std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vel,
          std::shared_ptr<const Core::LinAlg::Vector<double>> beam_vel) = 0;

     protected:
      //! Vector of pairs to be evaluated by this class.
      std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> assembly_contact_elepairs_;
    };

  }  // namespace SubmodelEvaluator
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
