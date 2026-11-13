// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_UTILS_HPP
#define FOUR_C_MAT_ANISOTROPY_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor.hpp"

#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication

namespace Core::IO
{
  class InputParameterContainer;
}

namespace Mat
{
  // forward declaration
  namespace Elastic
  {
    class StructuralTensorStrategyBase;
  }
  /*!
   * Reads a fiber with a specification from the input file definition
   *
   * @param container (in): Input parameter container of the element
   * @param specifier (in) : Identifier of the fiber
   * @param fiber_vector (out) : Fiber vector
   */
  void read_anisotropy_fiber(const Core::IO::InputParameterContainer& container,
      std::string specifier, Core::LinAlg::Tensor<double, 3>& fiber_vector);

  /*!
   * \brief Compute structural tensors of a 2D vector of fibers with the structural tensor
   * strategy
   *
   * \tparam T Output type of the structural tensor (either matrix notation or stress-like Voigt
   * notation)
   *
   * \tparam numfib number of fibers
   *
   * \param fibers 2D vector of fibers (3x1 matrices)
   * \param structural_tensor 2D vector of structural tensors (3x3 or 6x1 matrices)
   * \param strategy Reference to the structural tensor strategy
   */
  template <typename T, unsigned int numfib>
  void compute_structural_tensors(
      std::vector<std::array<Core::LinAlg::Tensor<double, 3>, numfib>>& fibers,
      std::vector<std::array<T, numfib>>& structural_tensor,
      const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& strategy);
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
