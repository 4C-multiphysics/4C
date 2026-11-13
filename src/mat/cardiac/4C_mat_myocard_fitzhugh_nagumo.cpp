// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_myocard_fitzhugh_nagumo.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardFitzhughNagumo::MyocardFitzhughNagumo() {}


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardFitzhughNagumo::MyocardFitzhughNagumo(
    const double eps_deriv_myocard, const std::string tissue, int num_gp)
    : tools_(), r0_(num_gp), r_(num_gp), j1_(num_gp), j2_(num_gp), mechanical_activation_(num_gp)
{
  // Initial condition
  for (int i = 0; i < num_gp; ++i)
  {
    r0_[i] = 0.0;
    r_[i] = r0_[i];
    mechanical_activation_[i] = 0.0;  // to store the variable for activation
  }


  eps_deriv_ = eps_deriv_myocard;

  // initialization of the material parameters
  a_ = 0.13;
  b_ = 0.013;
  c1_ = 0.26;
  c2_ = 0.1;
  d_ = 1.0;


  // Variables for electromechanical coupling
  act_thres_ = 0.2;  // activation threshold (so that activation = 1.0 if mechanical_activation_ >=
                     // act_thres_)
}

double MyocardFitzhughNagumo::rea_coeff(const double phi, const double dt)
{
  return MyocardFitzhughNagumo::rea_coeff(phi, dt, 0);
}

double MyocardFitzhughNagumo::rea_coeff(const double phi, const double dt, int gp)
{
  double reacoeff;
  r_[gp] = tools_.gating_var_calc(dt, r0_[gp], phi / d_, 1.0 / (b_ * d_));
  j1_[gp] = c1_ * phi * (phi - a_) * (phi - 1.0);
  j2_[gp] = c2_ * phi * r_[gp];
  reacoeff = j1_[gp] + j2_[gp];

  // For electromechanics
  mechanical_activation_[gp] = phi;

  return reacoeff;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardFitzhughNagumo::get_number_of_internal_state_variables() const { return 1; }

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardFitzhughNagumo::get_internal_state(const int k) const
{
  return get_internal_state(k, 0);
}

/*----------------------------------------------------------------------*
 |  returns current internal state of the material       hoermann 09/19 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double MyocardFitzhughNagumo::get_internal_state(const int k, int gp) const
{
  double val = 0.0;
  switch (k)
  {
    case -1:
    {
      val = mechanical_activation_[gp];
      break;
    }
    case 0:
    {
      val = r_[gp];
      break;
    }
  }
  return val;
}

/*----------------------------------------------------------------------*
 |  set  internal state of the material                     cbert 08/13 |
 *----------------------------------------------------------------------*/
void MyocardFitzhughNagumo::set_internal_state(const int k, const double val)
{
  set_internal_state(k, val, 0);
  return;
}

/*----------------------------------------------------------------------*
 |  set  internal state of the material                  hoermann 09/16 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
void MyocardFitzhughNagumo::set_internal_state(const int k, const double val, int gp)
{
  switch (k)
  {
    case -1:
    {
      mechanical_activation_[gp] = val;
      break;
    }
    case 0:
    {
      r0_[gp] = val;
      r_[gp] = val;
      break;
    }
    default:
    {
      FOUR_C_THROW("There are only 1 internal variables in this material!");
      break;
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardFitzhughNagumo::get_number_of_ionic_currents() const { return 2; }

/*----------------------------------------------------------------------*
 |  returns current internal currents                       cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardFitzhughNagumo::get_ionic_currents(const int k) const
{
  return get_ionic_currents(k, 0);
}

/*----------------------------------------------------------------------*
 |  returns current internal currents                    hoermann 09/16 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double MyocardFitzhughNagumo::get_ionic_currents(const int k, int gp) const
{
  double val = 0.0;
  switch (k)
  {
    case 0:
    {
      val = j1_[gp];
      break;
    }
    case 1:
    {
      val = j2_[gp];
      break;
    }
  }
  return val;
}


/*----------------------------------------------------------------------*
 |  update of material at the end of a time step             ljag 07/12 |
 *----------------------------------------------------------------------*/
void MyocardFitzhughNagumo::update(const double phi, const double dt)
{
  // update initial values for next time step
  r0_ = r_;

  return;
}

FOUR_C_NAMESPACE_CLOSE
