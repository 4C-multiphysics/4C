// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_FIBER_PROVIDER_HPP
#define FOUR_C_MAT_ANISOTROPY_FIBER_PROVIDER_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  /*!
   * @brief Pure abstract class that defines the interface of a fiber holder
   */
  class FiberProvider
  {
   public:
    virtual ~FiberProvider() = default;

    /// @name Getter methods for the fibers
    //@{
    /**
     * \brief Returns the i-th fiber vector at the Integration point
     *
     * \note Use gp=#GPDEFAULT if element fibers are used
     *
     * @param gp (in) : Id of the integration point (use #GPDEFAULT for Element fibers)
     * @param i (in) : Id of the fiber
     * @return Reference to the vector of the fiber
     */
    virtual const Core::LinAlg::Tensor<double, 3>& get_fiber(int gp, int i) const = 0;

    /**
     * \brief Returns the i-th structural tensor at the Integration point
     *
     * \note Use gp=#GPDEFAULT if element fibers are used
     *
     * @param gp (in) : Id of the integration point (use #GPDEFAULT for Element fibers)
     * @param i (in) : Id of the fiber
     */
    virtual const Core::LinAlg::SymmetricTensor<double, 3, 3>& get_structural_tensor(
        int gp, int i) const = 0;
    //@}
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
