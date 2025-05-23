// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_TRIAD_INTERPOLATION_HPP
#define FOUR_C_BEAM3_TRIAD_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace LargeRotations
{
  /**
   * \brief abstract base class for a triad interpolation scheme
   */
  template <typename T>
  class TriadInterpolation
  {
   public:
    //! @name Constructors and destructors and related methods

    /** \brief Standard Constructor
     *
     */
    TriadInterpolation();

    /** \brief Destructor
     *
     */
    virtual ~TriadInterpolation() = default;

    /** \brief return appropriate derived (templated) class (acts as a simple factory)
     *
     */
    static std::shared_ptr<TriadInterpolation<T>> create(unsigned int numnodes);
    //@}


    //! @name Public evaluation methods

    /** \brief reset interpolation scheme with nodal quaternions
     *
     */
    virtual void reset(std::vector<Core::LinAlg::Matrix<4, 1, T>> const& nodal_quaternions) = 0;

    /** \brief reset interpolation scheme with nodal triads
     *
     */
    virtual void reset(std::vector<Core::LinAlg::Matrix<3, 3, T>> const& nodal_triads) = 0;

    /** \brief compute the interpolated triad at any point \xi \in [-1,1] in parameter space
     *
     */
    virtual void get_interpolated_triad_at_xi(
        Core::LinAlg::Matrix<3, 3, T>& triad, const double xi) const = 0;

    /** \brief compute the interpolated quaternion at any point \xi \in [-1,1] in parameter space
     *
     */
    virtual void get_interpolated_quaternion_at_xi(
        Core::LinAlg::Matrix<4, 1, T>& quaternion, const double xi) const = 0;
    //@}

   private:
    //! @name Private evaluation methods

    //@}

   private:
    //! @name member variables

    //@}
  };

}  // namespace LargeRotations

FOUR_C_NAMESPACE_CLOSE

#endif
