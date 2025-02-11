// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LEGACY_ENUM_DEFINITIONS_PROBLEM_TYPE_HPP
#define FOUR_C_LEGACY_ENUM_DEFINITIONS_PROBLEM_TYPE_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  /**
   * A global definition of available problem types.
   */
  enum class ProblemType
  {
    none,                // not a problem at all
    ale,                 // pure ale problem
    art_net,             // arterial network problem _1D_ARTERY_
    biofilm_fsi,         // biofilm growth problem
    cardiac_monodomain,  // Cardiac electrophsiology problem
    ehl,        // elastohydrodynamic lubrication problem (or lubrication structure interaction)
    elch,       // electrochemical problem
    fluid,      // fluid problem
    fluid_ale,  // fluid on an ale mesh (no structure)
    fbi,        // 3D fluid interacting with a 1D beam
    fluid_redmodels,  // fluid_redairways problem
    fluid_xfem,       // fluid problem including XFEM interfaces
    fluid_xfem_ls,    // xfluid calculations using levelset for cut.
    fps3i,            // fluid porous structure scatra scatra interaction
    fpsi,             // fluid porous structure interaction problem
    fpsi_xfem,       // fluid poro structure interaction problem including XFEM interfaces (atm just
                     // for FSI Interface!)
    fsi,             // fluid structure interaction problem
    fsi_redmodels,   // fluid structure interaction problem
    fsi_xfem,        // fluid structure interaction problem including XFEM interfaces
    gas_fsi,         // fsi with gas transport
    level_set,       // level-set problem
    loma,            // low-Mach-number flow problem
    lubrication,     // lubrication problem (reduced fluid model for elastohydrodynamic lubrication)
    np_support,      // supporting procs for nested parallelism
    particle,        // particle simulation
    pasi,            // particle structure interaction
    polymernetwork,  // polymer network
    poroelast,       // poroelasticity
    poroscatra,      // passive scalar transport in porous media
    porofluidmultiphase,   // multiphase flow in porous media
    poromultiphase,        // multiphase flow in elastic porous media
    poromultiphasescatra,  // multiphase flow in elastic porous media with transport of species
    red_airways,           // reduced dimensional airways
    redairways_tissue,     // coupling of reduced-dimensional airways with 3D tissue model
    reduced_lung,          // New implementation of red_airways with Newton-iteration
    scatra,                // scalar transport problem (e.g. convection-diffusion)
    ssi,                   // scalar structure interaction
    ssti,                  // scalar structure thermo interaction
    sti,                   // scalar-thermo interaction
    structure,             // structural problem
    thermo,                // thermal problem
    thermo_fsi,            // thermo-fluid-structure-interaction problem
    tsi,                   // thermal structure interaction
  };
}  // namespace Core

FOUR_C_NAMESPACE_CLOSE

#endif
