// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_RED_AIRWAYS_ELEM_PARAMS_HPP
#define FOUR_C_RED_AIRWAYS_ELEM_PARAMS_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::ReducedLung
{
  struct ElemParams
  {
    double qout_np;
    double qout_n;
    double qout_nm;
    double qin_np;
    double qin_n;
    double qin_nm;
    double volnp;
    double voln;
    double acin_vnp;
    double acin_vn;
    double lungVolume_np;
    double lungVolume_n;
    double lungVolume_nm;
    double x_np;
    double x_n;
    double p_extn;
    double p_extnp;
    double open;
  };

  struct AirwayParams
  {
    // viscoelastic RLC airway parameters
    double power_velocity_profile{2.0};
    double wall_elasticity{10000.0};
    double poisson_ratio{0.49};
    double wall_thickness{0.5};
    double area{1.0};
    double viscous_Ts{2.0};
    double viscous_phase_shift{0.13};
    double branch_length{-1.0};
    int generation{0};

    // extended parameters for collapsible airways
    double airway_coll{0.0};
    double s_close{0.0};
    double s_open{0.0};
    double p_crit_open{0.0};
    double p_crit_close{0.0};
    double open_init{1.0};
  };

  struct AcinusParams
  {
    double volume_relaxed{1.0};
    double alveolar_duct_volume{0.0337};
    double volume_init{1.0};
    int generation{-1};
  };
}  // namespace Discret::ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
