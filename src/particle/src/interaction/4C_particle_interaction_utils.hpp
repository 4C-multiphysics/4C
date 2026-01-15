// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_UTILS_HPP
#define FOUR_C_PARTICLE_INTERACTION_UTILS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/

namespace Particle::ParticleUtils
{
  /**
   *  \brief provide an efficient method to determine the power with integer exponents
   */
  template <class T, int n>
  struct Helper
  {
    static_assert(n >= 0, "The exponent must be positive!");
    static constexpr T pow(const T x)
    {
      return ((n % 2) == 0 ? 1 : x) * Helper<T, (n / 2)>::pow(x * x);
    }
  };

  template <class T>
  struct Helper<T, 0>
  {
    static constexpr T pow(const T x) { return 1; }
  };

  /**
   *  \brief helper function
   *
   *  when you use this helper function there will be no need to explicitly insert the class type
   */
  template <int n, class T>
  T constexpr pow(T const x)
  {
    return Helper<T, n>::pow(x);
  }

  //! @name collection of three dimensional vector operations
  //@{

  /**
   *  \brief clear vector c
   */
  template <class T>
  inline void vec_clear(T* c)
  {
    c[0] = 0.0;
    c[1] = 0.0;
    c[2] = 0.0;
  }

  /**
   *  \brief set vector a to vector c
   */
  template <class T>
  inline void vec_set(T* c, const T* a)
  {
    c[0] = a[0];
    c[1] = a[1];
    c[2] = a[2];
  }

  /**
   *  \brief add vector a to vector c
   */
  template <class T>
  inline void vec_add(T* c, const T* a)
  {
    c[0] += a[0];
    c[1] += a[1];
    c[2] += a[2];
  }

  /**
   *  \brief subtract vector a from vector c
   */
  template <class T>
  inline void vec_sub(T* c, const T* a)
  {
    c[0] -= a[0];
    c[1] -= a[1];
    c[2] -= a[2];
  }

  /**
   *  \brief scale vector c
   */
  template <class T>
  inline void vec_scale(T* c, const T fac)
  {
    c[0] *= fac;
    c[1] *= fac;
    c[2] *= fac;
  }

  /**
   *  \brief scale vector a and set to vector c
   */
  template <class T>
  inline void vec_set_scale(T* c, const T fac, const T* a)
  {
    c[0] = fac * a[0];
    c[1] = fac * a[1];
    c[2] = fac * a[2];
  }

  /**
   *  \brief scale vector a and add to vector c
   */
  template <class T>
  inline void vec_add_scale(T* c, const T fac, const T* a)
  {
    c[0] += fac * a[0];
    c[1] += fac * a[1];
    c[2] += fac * a[2];
  }

  /**
   *  \brief set cross product of vector a and vector b to vector c
   */
  template <class T>
  inline void vec_set_cross(T* c, const T* a, const T* b)
  {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
  }

  /**
   *  \brief add cross product of vector a and vector b to vector c
   */
  template <class T>
  inline void vec_add_cross(T* c, const T* a, const T* b)
  {
    c[0] += a[1] * b[2] - a[2] * b[1];
    c[1] += a[2] * b[0] - a[0] * b[2];
    c[2] += a[0] * b[1] - a[1] * b[0];
  }

  /**
   *  \brief return scalar product of vector a and vector b
   */
  template <class T>
  inline T vec_dot(const T* a, const T* b)
  {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  /**
   *  \brief return 2-norm of vector a
   */
  template <class T>
  inline T vec_norm_two(const T* a)
  {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  }

  //@}

  //! @name methods for construction of three dimensional vector space
  //@{

  /**
   *  \brief construct orthogonal unit surface tangent vectors from given unit surface normal
   */
  template <class T>
  inline void unit_surface_tangents(const T* n, T* t1, T* t2)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (std::abs(1.0 - vec_norm_two(n)) > 1.0e-14)
      FOUR_C_THROW("given unit surface normal not normalized!");
#endif

    if ((std::abs(n[0]) <= std::abs(n[1])) and (std::abs(n[0]) <= std::abs(n[2])))
    {
      t1[0] = 0.0;
      t1[1] = -n[2];
      t1[2] = n[1];
    }
    else if (std::abs(n[1]) <= std::abs(n[2]))
    {
      t1[0] = -n[2];
      t1[1] = 0.0;
      t1[2] = n[0];
    }
    else
    {
      t1[0] = -n[1];
      t1[1] = n[0];
      t1[2] = 0.0;
    }

    vec_scale(t1, 1.0 / vec_norm_two(t1));

    vec_set_cross(t2, n, t1);
  }

  //@}

  //! @name methods for linear transition in a given interval
  //@{

  /**
   *  \brief linear transition function
   */
  inline double lin_trans(const double x, const double x1, const double x2)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (not(std::abs(x2 - x1) > 1.0e-14)) FOUR_C_THROW("danger of division by zero!");
#endif

    if (x < x1) return 0.0;
    if (x > x2) return 1.0;
    return (x - x1) / (x2 - x1);
  }

  /**
   *  \brief complementary linear transition function
   */
  inline double comp_lin_trans(const double x, const double x1, const double x2)
  {
    return 1.0 - lin_trans(x, x1, x2);
  }

  //@}

}  // namespace Particle::ParticleUtils

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
