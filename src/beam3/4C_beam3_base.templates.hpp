// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_BASE_TEMPLATES_HPP
#define FOUR_C_BEAM3_BASE_TEMPLATES_HPP

#include "4C_config.hpp"

#include "4C_beam3_base.hpp"
#include "4C_utils_local_newton.hpp"

FOUR_C_NAMESPACE_OPEN

template <unsigned int nnode, unsigned int vpernode>
double Discret::Elements::Beam3Base::calc_reflength(
    const Core::LinAlg::Matrix<3 * vpernode * nnode, 1, double>& disp_refe_centerline)
{
  Core::LinAlg::Matrix<3, 1> tempvec(Core::LinAlg::Initialization::zero);

  for (int dim = 0; dim < 3; dim++)
    tempvec(dim) = disp_refe_centerline(3 * vpernode * 1 + dim) - disp_refe_centerline(dim);
  double reflength = tempvec.norm2();

  if (hermite_centerline_interpolation())
  {
    const auto gausspoints = Core::FE::IntegrationPoints1D(Core::FE::GaussRule1D::line_10point);
    const auto distype = this->shape();

    auto lengthEquationAndDerivative = [&gausspoints, &disp_refe_centerline, &distype](
                                           double reflength)
    {
      return Discret::Utils::Beam::integrate_centerline_arc_length_and_arc_length_derivative<nnode,
          vpernode>(gausspoints, disp_refe_centerline, distype, reflength);
    };

    const double newton_tolerance = 1e-12;
    const int max_iterations = 200;
    reflength = Core::Utils::solve_local_newton(
        lengthEquationAndDerivative, reflength, newton_tolerance, max_iterations);
  }

  return reflength;
}

FOUR_C_NAMESPACE_CLOSE

#endif
