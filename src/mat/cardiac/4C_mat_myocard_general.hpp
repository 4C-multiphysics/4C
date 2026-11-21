// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MYOCARD_GENERAL_HPP
#define FOUR_C_MAT_MYOCARD_GENERAL_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

class MyocardGeneral

{
 public:
  /// construct empty material object
  MyocardGeneral() {};

  /// construct empty material object
  explicit MyocardGeneral(const double eps_deriv_myocard, const std::string tissue);

  /// destructor
  virtual ~MyocardGeneral() = default;
  /// compute reaction coefficient for multiple Gauss points
  virtual double rea_coeff(const double phi, const double dt, int gp)
  {
    if (gp > 0)
    {
      FOUR_C_THROW("Multiple Gauss points only implemented for MV and FHN model");
      return 0.;
    }

    else
      return rea_coeff(phi, dt);
  };

  /// compute reaction coefficient
  virtual double rea_coeff(const double phi, const double dt) = 0;

  /// compute reaction coefficient at timestep n
  virtual double rea_coeff_n(const double phi, const double dt) { return 0; };

  /// compute reaction coefficient for multiple Gauss points at timestep n
  virtual double rea_coeff_n(const double phi, const double dt, int gp)
  {
    if (gp > 0)
    {
      FOUR_C_THROW("Multiple Gauss points only implemented for MV and FHN model");
      return 0.;
    }

    else
      return rea_coeff_n(phi, dt);
  };

  ///  returns number of internal state variables of the material
  virtual int get_number_of_internal_state_variables() const = 0;

  ///  returns current internal state of the material
  virtual double get_internal_state(const int k) const = 0;

  ///  returns current internal state of the material
  virtual double get_internal_state(const int k, int gp) const
  {
    if (gp > 0)
    {
      FOUR_C_THROW("Multiple Gauss points only implemented for MV and FHN model");
      return 0.;
    }
    else
      return get_internal_state(k);
  };

  ///  set internal state of the material
  virtual void set_internal_state(const int k, const double val) = 0;

  ///  set internal state of the material for multiple Gauss points
  virtual void set_internal_state(const int k, const double val, int gp)
  {
    if (gp > 0)
      FOUR_C_THROW("Multiple Gauss points only implemented for MV and FHN model");
    else
      set_internal_state(k, val);

    return;
  };

  ///  return number of ionic currents
  virtual int get_number_of_ionic_currents() const = 0;

  ///  return ionic currents
  virtual double get_ionic_currents(const int k) const = 0;

  ///  return ionic currents
  virtual double get_ionic_currents(const int k, int gp) const
  {
    FOUR_C_THROW("Multiple Gauss Points only implemented for MV and FHN model");
    return 0.;
  };

  // resize internal state variables if number of Gauss point changes
  virtual void resize_internal_state_variables(int gp)
  {
    FOUR_C_THROW("Multiple Gauss Points only implemented for MV and FHN model");
    return;
  };

  /// get number of Gauss points
  virtual int get_number_of_gp() const { return 1; };

  /// time update for this material
  virtual void update(const double phi, const double dt) = 0;

};  // Myocard_general


FOUR_C_NAMESPACE_CLOSE

#endif
