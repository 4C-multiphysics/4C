// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_aux.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | Calculate vector norm                                    bborn 08/09 |
 *----------------------------------------------------------------------*/
double Thermo::Aux::calculate_vector_norm(
    const enum Thermo::VectorNorm norm, Core::LinAlg::Vector<double>& vect)
{
  // L1 norm
  if (norm == Thermo::norm_l1)
  {
    double vectnorm;
    vect.Norm1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  else if (norm == Thermo::norm_l2)
  {
    double vectnorm;
    vect.Norm2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  else if (norm == Thermo::norm_rms)
  {
    double vectnorm;
    vect.Norm2(&vectnorm);
    return vectnorm / std::sqrt((double)vect.GlobalLength());
  }
  // infinity/maximum norm
  else if (norm == Thermo::norm_inf)
  {
    double vectnorm;
    vect.NormInf(&vectnorm);
    return vectnorm;
  }
  else
  {
    FOUR_C_THROW("Cannot handle vector norm");
    return 0;
  }
}  // calculate_vector_norm()


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
