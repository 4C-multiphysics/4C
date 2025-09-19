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
#include "4C_io_pstream.hpp"

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
      for (auto node : dis.my_row_node_range())
      {
        auto x = node.x();
        for (size_t j = 0; j < x.size(); ++j) x0send[j] += x[j];
      }

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
          Core::IO::cout(Core::IO::debug)
              << "Warning: At local node " << node << " : nullspace degrees of freedom ( " << numdf
              << " ) and rowmap degrees of freedom ( " << localLength << " ) are not consistent.\n";
        }

        // Here we check the first element type of the node. One node can be owned by several
        // elements we restrict the routine, that a node is only owned by elements with the same
        // physics
        auto adjacent_elements = actnode->adjacent_elements();
        if (adjacent_elements.size() > 1)
        {
          // strip of the first element and compare the rest with it
          ElementRef first = *adjacent_elements.begin();
          auto& type_first = first.user_element()->element_type();
          int numdof_first;
          int dimnsp_first;
          type_first.nodal_block_information(first.user_element(), numdof_first, dimnsp_first);

          for (auto ele : adjacent_elements | std::views::drop(1))
          {
            auto* other = ele.user_element();
            auto& type_other = other->element_type();
            if (type_first != type_other)
            {
              int numdof_other;
              int dimnsp_other;
              type_other.nodal_block_information(other, numdof_other, dimnsp_other);
              if (numdof_first != numdof_other || dimnsp_first != dimnsp_other)
                FOUR_C_THROW(
                    "Node is owned by different element types, nullspace calculation aborted!");
            }
          }
        }

        Core::LinAlg::SerialDenseMatrix nodalNullspace =
            adjacent_elements[0].user_element()->element_type().compute_null_space(
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
