// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_INTERSECTION_MATH_HPP
#define FOUR_C_FEM_GEOMETRY_INTERSECTION_MATH_HPP


#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Core::Geo
{
  //! named tolerance for easy search/grep
  const double TOL14 = 1e-14;
  //! named tolerance for easy search/grep
  const double TOL13 = 1e-13;
  //! named tolerance for easy search/grep
  const double TOL12 = 1e-12;
  //! named tolerance for easy search/grep
  const double TOL8 = 1e-8;
  //! named tolerance for easy search/grep
  const double TOL7 = 1e-7;
  //! named tolerance for easy search/grep
  const double TOL6 = 1e-6;
  //! named tolerance for easy search/grep
  const double TOLPLUS8 = 1e8;
}  // namespace Core::Geo


FOUR_C_NAMESPACE_CLOSE

#endif
