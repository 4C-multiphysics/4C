// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc_elch_electrode_utils.hpp"

#include "4C_scatra_s2i_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::calculate_butler_volmer_elch_linearizations(const int kineticmodel,
    const double j0, const double frt, const double epdderiv, const double alphaa,
    const double alphac, const double resistance, const double expterm1, const double expterm2,
    const double kr, const double faraday, const double emasterphiint, const double eslavephiint,
    const double cmax, const double eta, double& dj_dc_slave, double& dj_dc_master,
    double& dj_dpot_slave, double& dj_dpot_master)
{
  const double expterm = expterm1 - expterm2;
  // core linearizations associated with Butler-Volmer mass flux density
  switch (kineticmodel)
  {
    case S2I::kinetics_butlervolmerreduced:
    case S2I::kinetics_butlervolmerreducedthermoresistance:
    case S2I::kinetics_butlervolmerreducedcapacitance:
    {
      dj_dc_slave = j0 * frt * epdderiv * (-alphaa * expterm1 - alphac * expterm2);
      dj_dc_master = 0.0;
      dj_dpot_slave = j0 * (alphaa * frt * expterm1 + alphac * frt * expterm2);
      dj_dpot_master = -dj_dpot_slave;
      break;
    }
    case S2I::kinetics_butlervolmerreducedlinearized:
    {
      dj_dc_slave = -j0 * frt * epdderiv;
      dj_dc_master = 0.0;
      dj_dpot_slave = j0 * frt;
      dj_dpot_master = -dj_dpot_slave;
      break;
    }
    case S2I::kinetics_butlervolmer:
    case S2I::kinetics_butlervolmerpeltier:
    {
      dj_dc_slave =
          (kr * std::pow(emasterphiint, alphaa) * std::pow(cmax - eslavephiint, alphaa - 1.0) *
                  std::pow(eslavephiint, alphac - 1.0) *
                  (-alphaa * eslavephiint + alphac * (cmax - eslavephiint)) * expterm +
              j0 * frt * epdderiv * (-alphaa * expterm1 - alphac * expterm2));
      dj_dc_master = j0 * alphaa / emasterphiint * expterm;
      dj_dpot_slave = j0 * (alphaa * frt * expterm1 + alphac * frt * expterm2);
      dj_dpot_master = -dj_dpot_slave;
      break;
    }
    case S2I::kinetics_butlervolmerlinearized:
    {
      dj_dc_slave =
          (kr * std::pow(emasterphiint, alphaa) * std::pow(cmax - eslavephiint, alphaa - 1.0) *
                  std::pow(eslavephiint, alphac - 1.0) *
                  (-alphaa * eslavephiint + alphac * (cmax - eslavephiint)) * frt * eta -
              j0 * frt * epdderiv);
      dj_dc_master = j0 * alphaa / emasterphiint * frt * eta;
      dj_dpot_slave = j0 * frt;
      dj_dpot_master = -dj_dpot_slave;
      break;
    }
    case S2I::kinetics_butlervolmerresistance:
    {
      // core linearizations associated with Butler-Volmer current density according to MA Schmidt
      // 2016 via implicit differentiation where F(x,i) = i - i0 * expterm
      const double dF_di_inverse =
          1.0 / (1.0 + j0 * faraday * resistance * frt * (alphaa * expterm1 + alphac * expterm2));
      const double dF_dc_slave =
          (-kr * std::pow(emasterphiint, alphaa) * std::pow(cmax - eslavephiint, alphaa - 1.0) *
                  std::pow(eslavephiint, alphac - 1.0) *
                  (-alphaa * eslavephiint + alphac * (cmax - eslavephiint)) * expterm +
              j0 * frt * epdderiv * (alphaa * expterm1 + alphac * expterm2));
      const double dF_dc_master = -j0 * alphaa / emasterphiint * expterm;
      const double dF_dpot_slave = -j0 * frt * (alphaa * expterm1 + alphac * expterm2);
      const double dF_dpot_master = -dF_dpot_slave;

      // rule of implicit differentiation
      dj_dc_slave = -dF_dc_slave * dF_di_inverse;
      dj_dc_master = -dF_dc_master * dF_di_inverse;
      dj_dpot_slave = -dF_dpot_slave * dF_di_inverse;
      dj_dpot_master = -dF_dpot_master * dF_di_inverse;
      break;
    }  // case S2I::kinetics_butlervolmerresistance
    case S2I::kinetics_butlervolmerreducedresistance:
    {
      // core linearizations associated with Butler-Volmer current density according to MA Schmidt
      // 2016 via implicit differentiation where F(x,i) = i - i0 * expterm
      const double dF_di_inverse =
          1.0 / (1.0 + j0 * faraday * resistance * frt * (alphaa * expterm1 + alphac * expterm2));
      const double dF_dc_slave = j0 * frt * epdderiv * (alphaa * expterm1 + alphac * expterm2);
      const double dF_dc_master = 0.0;
      const double dF_dpot_slave = -j0 * frt * (alphaa * expterm1 + alphac * expterm2);
      const double dF_dpot_master = -dF_dpot_slave;

      // rule of implicit differentiation
      dj_dc_slave = -dF_dc_slave * dF_di_inverse;
      dj_dc_master = -dF_dc_master * dF_di_inverse;
      dj_dpot_slave = -dF_dpot_slave * dF_di_inverse;
      dj_dpot_master = -dF_dpot_master * dF_di_inverse;
      break;
    }  // case S2I::kinetics_butlervolmerreducedresistance
    default:
    {
      FOUR_C_THROW("Unknown scatra-scatra interface kinetic model: {}", kineticmodel);
    }
  }  // switch(kineticmodel)
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::calculate_butler_volmer_temp_linearizations(const double alphaa,
    const double alphac, const double depddT, const double eta, const double etempint,
    const double faraday, const double frt, const double gasconstant, const double j0,
    double& dj_dT_slave)
{
  // exponential Butler-Volmer terms
  const double exptermA = std::exp(alphaa * frt * eta);
  const double exptermB = std::exp(-alphac * frt * eta);

  // Butler-Volmer:
  // j = j0 * (exp(A)-exp(B)), A = a eta/T, B = b eta/T
  const double a = alphaa * faraday / gasconstant;
  const double b = -1.0 * alphac * faraday / gasconstant;

  // C = d(eta/T)/dT
  const double C = (depddT / etempint) - (eta / (etempint * etempint));

  // derivative of A and B w.r.t. temperature
  const double dAdT = a * C;
  const double dBdT = b * C;

  // derivative of flux w.r.t. temperature
  const double djdT = j0 * (exptermA * dAdT - exptermB * dBdT);

  // derivative of flux w.r.t. slave side temperature (T = 0.5 * [T_slave + T_master])
  dj_dT_slave = djdT * 0.5;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::calculate_butler_volmer_disp_linearizations(const int kineticmodel,
    const double alphaa, const double alphac, const double frt, const double j0, const double eta,
    const double depd_ddetF, double& dj_dsqrtdetg, double& dj_ddetF)
{
  double dj_depd;

  if (is_butler_volmer_linearized(kineticmodel))
  {
    dj_dsqrtdetg = j0 * frt * eta;
    dj_depd = -j0 * frt;
  }
  else
  {
    // exponential Butler-Volmer terms
    const double expterm1 = std::exp(alphaa * frt * eta);
    const double expterm2 = std::exp(-alphac * frt * eta);
    const double expterm = expterm1 - expterm2;

    dj_dsqrtdetg = j0 * expterm;
    dj_depd = -j0 * frt * (alphaa * expterm1 + alphac * expterm2);
  }

  dj_ddetF = dj_depd * depd_ddetF;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Discret::Elements::calculate_butler_volmer_exchange_mass_flux_density(const double kr,
    const double alpha_a, const double alpha_c, const double c_max, const double c_ed,
    const double c_el, const int kinetic_model,
    const Core::Conditions::ConditionType& s2i_condition_type)
{
  FOUR_C_ASSERT(s2i_condition_type == Core::Conditions::S2IKinetics,
      "This method is called with the wrong condition type. Check the implementation!");

  if (is_reduced_butler_volmer(kinetic_model))
  {
    return kr;
  }
  else
  {
    return kr * std::pow(c_el, alpha_a) * std::pow(c_max - c_ed, alpha_a) * std::pow(c_ed, alpha_c);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Discret::Elements::calculate_modified_butler_volmer_mass_flux_density(const double j0,
    const double alphaa, const double alphac, const double frt, const double pot_ed,
    const double pot_el, const double epd, const double resistance, const double itemax,
    const double convtol, const double faraday)
{
  // Iterations are conducted over current density i which is scaled down to mass flux
  // density j by j = i / faraday at the end of the function in order to reduce the effect
  // of numerical error introduced into global problem since i is roughly 10^5 times bigger than j

  // initialize Butler-Volmer current density
  double i(0.0);

  const double i0 = j0 * faraday;
  // compute Butler-Volmer current density in case of physically reasonable half-cell open-circuit
  // potential
  if (not std::isinf(epd))
  {
    // initialize Newton-Raphson iteration counter
    unsigned iternum(0);

    // apply Newton-Raphson method to compute Butler-Volmer current density, involving overpotential
    // due to scatra-scatra interface layer resistance
    while (true)
    {
      // increment counter
      ++iternum;

      // compute current Newton-Raphson residual
      const double eta = pot_ed - pot_el - epd - resistance * i;
      const double expterm1 = std::exp(alphaa * frt * eta);
      const double expterm2 = std::exp(-alphac * frt * eta);
      const double residual = i0 * (expterm1 - expterm2) - i;

      // convergence check
      if (std::abs(residual) < convtol)
      {
        break;
      }
      else if (iternum == itemax)
        FOUR_C_THROW(
            "Local Newton-Raphson iteration for Butler-Volmer current density did not converge!");

      // compute linearization of current Newton-Raphson residual w.r.t. Butler-Volmer current
      // density
      const double linearization =
          -i0 * resistance * frt * (alphaa * expterm1 + alphac * expterm2) - 1.0;

      // update Butler-Volmer current density
      i -= residual / linearization;
    }
  }
  // final scaling
  return i / faraday;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Discret::Elements::is_butler_volmer_linearized(const int kineticmodel)
{
  return (kineticmodel == S2I::kinetics_butlervolmerlinearized or
          kineticmodel == S2I::kinetics_butlervolmerreducedlinearized);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Discret::Elements::is_reduced_butler_volmer(const int kineticmodel)
{
  return (kineticmodel == S2I::kinetics_butlervolmerreduced or
          kineticmodel == S2I::kinetics_butlervolmerreducedlinearized or
          kineticmodel == S2I::kinetics_butlervolmerreducedresistance or
          kineticmodel == S2I::kinetics_butlervolmerreducedthermoresistance or
          kineticmodel == S2I::kinetics_butlervolmerreducedcapacitance);
}
FOUR_C_NAMESPACE_CLOSE
