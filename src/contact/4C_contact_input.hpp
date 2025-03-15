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

#include <map>
#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace CONTACT
{
  /// Type of contact friction law
  /// (this enum represents the input file parameter FRICTION)
  enum FrictionType
  {
    friction_none = 0,  ///< no friction
    friction_stick,     ///< perfect stick
    friction_tresca,    ///< Tresca friction law
    friction_coulomb    ///< Coulomb friction law
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
  enum SolvingStrategy : int
  {
    solution_vague,      ///< no solving strategy defined
    solution_lagmult,    ///< method of Lagrange multipliers
    solution_penalty,    ///< penalty approach
    solution_uzawa,      ///< Uzawa augmented Lagrange approach
    solution_nitsche,    ///< Nitsche contact solution method
    solution_ehl,        ///< method for elasto-hydrodynamic lubrication
    solution_multiscale  ///< method for contact of rough surfaces with a multi scale approach
  };

  inline std::string solving_strategy_to_string(enum SolvingStrategy stype)
  {
    switch (stype)
    {
      case solution_vague:
        return "solution_vague";
      case solution_lagmult:
        return "solution_lagmult";
      case solution_penalty:
        return "solution_penalty";
      case solution_uzawa:
        return "solution_uzawa";
      case solution_nitsche:
        return "solution_nitsche";
      default:
        return "INVALID SolvingStrategy";
    }
  }

  /// Type of linear system setup and solution
  /// (this enum represents the input file parameter SYSTEM)
  enum SystemType : int
  {
    system_none,               ///< no system defined
    system_condensed,          ///< condensed system
    system_condensed_lagmult,  ///< system with condensed lagrange multiplier (differs just in
                               ///< meshtying case)
    system_saddlepoint         ///< saddle point system
  };

  /// Type of formulation of constraint equations
  /// (this enum represents the input file parameter CONSTRAINT_DIRECTIONS)
  enum ConstraintDirection
  {
    constr_vague,  ///< no constraint directions defined
    constr_ntt,    ///< local normal and tangential coordinates
    constr_xyz     ///< global Cartesian coordinates
  };

  /// Local definition of problemtype to avoid use of globalproblem.H
  enum Problemtype
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

  /// Assemble strategy for the augmented Lagrangian framework
  enum AssembleStrategy : int
  {
    assemble_none,
    assemble_node_based  ///< assemble based on nodal data containers
  };

  /// convert assemble strategy to string
  inline std::string assemble_strategy_to_string(enum AssembleStrategy assemble_type)
  {
    switch (assemble_type)
    {
      case assemble_none:
        return "assemble_none";
      case assemble_node_based:
        return "assemble_node_based";
      default:
        return "INVALID assemble strategy";
    }
  }

  /// Contact coupling mode for multiphysics solver
  enum class CouplingScheme : int
  {
    unknown,      ///< unspecified
    monolithic,   ///< monolithic approach
    partitioning  ///< partitioning approach
  };

  /// set the contact parameters
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

}  // namespace CONTACT

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
