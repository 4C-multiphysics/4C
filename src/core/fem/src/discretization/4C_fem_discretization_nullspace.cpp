// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization_nullspace.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"

#include <Teuchos_ArrayRCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> compute_null_space(
      const Core::FE::Discretization& dis, const int numdf, const int dimns,
      const Core::LinAlg::Map& dofmap)
  {
    if (dimns > 10) FOUR_C_THROW("Nullspace size only up to 10 supported!");

    std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
        std::make_shared<Core::LinAlg::MultiVector<double>>(dofmap, dimns, true);

    if (dimns == 1 && numdf == 1)
    {
      // compute nullspace for simple case: vector of ones
      nullspace->PutScalar(1.0);
    }
    else
    {
      // for rigid body rotations compute nodal center of the discretization
      std::array<double, 3> x0send = {0.0, 0.0, 0.0};
      for (int i = 0; i < dis.num_my_row_nodes(); ++i)
        for (int j = 0; j < 3; ++j) x0send[j] += dis.l_row_node(i)->x()[j];

      std::array<double, 3> x0;
      Core::Communication::sum_all(x0send.data(), x0.data(), 3, dis.get_comm());

      for (int i = 0; i < 3; ++i) x0[i] /= dis.num_global_nodes();

      // assembly process of the nodalNullspace into the actual nullspace
      for (int node = 0; node < dis.num_my_row_nodes(); ++node)
      {
        Core::Nodes::Node* actnode = dis.l_row_node(node);
        std::vector<int> dofs = dis.dof(0, actnode);
        const int localLength = dofs.size();

        // check if degrees of freedom are zero
        if (localLength == 0) continue;

        // check if dof is existing as index
        if (dofmap.lid(dofs[0]) == -1) continue;

        // check size of degrees of freedom
        if (localLength != numdf)
        {
          std::cout << "Warning: At local node " << node << " : nullspace degrees of freedom ( "
                    << numdf << " ) "
                    << "and rowmap degrees of freedom ( " << localLength << " ) are not consistent"
                    << std::endl;
        }

        // Here we check the first element type of the node. One node can be owned by several
        // elements we restrict the routine, that a node is only owned by elements with the same
        // physics
        if (actnode->num_element() > 1)
        {
          for (int i = 0; i < actnode->num_element() - 1; i++)
          {
            // if element types are different, check nullspace dimension and dofs
            if (actnode->elements()[i + 1]->element_type() !=
                actnode->elements()[i]->element_type())
            {
              int numdof1, dimnsp1;
              actnode->elements()[i]->element_type().nodal_block_information(
                  actnode->elements()[i], numdof1, dimnsp1);
              int numdof2, dimnsp2;
              actnode->elements()[i + 1]->element_type().nodal_block_information(
                  actnode->elements()[i + 1], numdof2, dimnsp2);

              if (numdof1 != numdof2 || dimnsp1 != dimnsp2)
                FOUR_C_THROW(
                    "Node is owned by different element types, nullspace calculation aborted!");
            }
          }
        }

        Core::LinAlg::SerialDenseMatrix nodalNullspace =
            actnode->elements()[0]->element_type().compute_null_space(
                *actnode, x0.data(), localLength, dimns);

        for (int dim = 0; dim < dimns; ++dim)
        {
          double** arrayOfPointers;
          nullspace->ExtractView(&arrayOfPointers);
          double* data = arrayOfPointers[dim];
          Teuchos::ArrayRCP<double> dataVector(data, dofmap.lid(dofs[0]), localLength, false);

          for (int j = 0; j < localLength; ++j)
          {
            const int lid = dofmap.lid(dofs[j]);
            dataVector[lid] = nodalNullspace(j, dim);
          }
        }
      }
    }

    return nullspace;
  }
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE
