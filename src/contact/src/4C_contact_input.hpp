// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_INPUT_HPP
#define FOUR_C_CONTACT_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace CONTACT
{
  /// Type of contact friction law
  /// (this enum represents the input file parameter FRICTION)
  enum class FrictionType
  {
    none,    ///< no friction
    stick,   ///< perfect stick
    tresca,  ///< Tresca friction law
    coulomb  ///< Coulomb friction law
  };

  /// Type of contact adhesion law
  /// (this enum represents the input file parameter ADHESION)
  enum class AdhesionType
  {
    none,    ///< no adhesion
    bounded  ///< fix bounded adhesion
  };

  /// Type of employed solving strategy
  /// (this enum represents the input file parameter STRATEGY)
  enum class SolvingStrategy
  {
    vague,      ///< no solving strategy defined
    lagmult,    ///< method of Lagrange multipliers
    augmented,  ///< augmented Lagrange approach
    penalty,    ///< penalty approach
    uzawa,      ///< Uzawa augmented Lagrange approach
    nitsche,    ///< Nitsche contact solution method
    ehl,        ///< method for elasto-hydrodynamic lubrication
    multiscale  ///< method for contact of rough surfaces with a multi scale approach
  };

  inline std::string solving_strategy_to_string(SolvingStrategy stype)
  {
    switch (stype)
    {
      case SolvingStrategy::vague:
        return "CONTACT::SolvingStrategy::vague";
      case SolvingStrategy::lagmult:
        return "CONTACT::SolvingStrategy::lagmult";
      case SolvingStrategy::augmented:
        return "CONTACT::SolvingStrategy::augmented";
      case SolvingStrategy::penalty:
        return "CONTACT::SolvingStrategy::penalty";
      case SolvingStrategy::uzawa:
        return "CONTACT::SolvingStrategy::uzawa";
      case SolvingStrategy::nitsche:
        return "CONTACT::SolvingStrategy::nitsche";
      default:
        return "Invalid CONTACT::SolvingStrategy";
    }
  }

  /// Type of linear system setup and solution
  /// (this enum represents the input file parameter SYSTEM)
  enum class SystemType
  {
    none,               ///< no system defined
    condensed,          ///< condensed system
    condensed_lagmult,  ///< system with condensed lagrange multiplier (differs just in
                        ///< meshtying case)
    saddlepoint         ///< saddle point system
  };

  /// Type of formulation of constraint equations
  /// (this enum represents the input file parameter CONSTRAINT_DIRECTIONS)
  enum class ConstraintDirection
  {
    vague,  ///< no constraint directions defined
    ntt,    ///< local normal and tangential coordinates
    xyz     ///< global Cartesian coordinates
  };

  /// Local definition of problemtype to avoid use of globalproblem.H
  enum class Problemtype
  {
    structure,      ///< structural contact problem
    tsi,            ///< coupled TSI problem with contact
    structalewear,  ///< wear problem including ALE shape changes
    poroelast,      ///< poroelasticity problem with contact
    poroscatra,     ///< poroscatra problem with contact
    ehl,            ///< elasto-hydrodymanic lubrication
    fsi,            ///< coupled FSI problem with contact
    fpi,            ///< coupled FPI problem with contact
    ssi,            ///< coupled SSI problem with contact
    ssi_elch,       ///< coupled SSI elch problem with contact
    other           ///< other problemtypes
  };

  /// weighting in Nitsche contact
  enum class NitscheWeighting
  {
    slave,
    master,
    harmonic,
    physical
  };

  /// Contact coupling mode for multiphysics solver
  enum class CouplingScheme : int
  {
    unknown,      ///< unspecified
    monolithic,   ///< monolithic approach
    partitioning  ///< partitioning approach
  };

  /// contact parameters
  Core::IO::InputSpec valid_parameters();

}  // namespace CONTACT

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
