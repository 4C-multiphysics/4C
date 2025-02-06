// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_constraint_framework.hpp"

#include "4C_utils_parameter_list.hpp"


FOUR_C_NAMESPACE_OPEN

/**
 *
 */
void Inpar::CONSTRAINTS::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::tuple;

  Teuchos::ParameterList& embeddedmeshcoupling = list.sublist("EMBEDDED MESH COUPLING", false, "");
  {
    Core::Utils::string_to_integral_parameter<EmbeddedMeshCouplingStrategy>("COUPLING_STRATEGY",
        "none", "Strategy to couple background and overlapping mesh",
        tuple<std::string>("none", "mortar"),
        tuple<EmbeddedMeshCouplingStrategy>(
            EmbeddedMeshCouplingStrategy::none, EmbeddedMeshCouplingStrategy::mortar),
        &embeddedmeshcoupling);

    Core::Utils::string_to_integral_parameter<SolidToSolidMortarShapefunctions>(
        "MORTAR_SHAPE_FUNCTION", "none",
        "Shape functions that should be use in case of coupling using the Mortar/Lagrange "
        "Multiplier method",
        tuple<std::string>("none", "quad4", "quad9", "nurbs9"),
        tuple<SolidToSolidMortarShapefunctions>(SolidToSolidMortarShapefunctions::none,
            SolidToSolidMortarShapefunctions::quad4, SolidToSolidMortarShapefunctions::quad9,
            SolidToSolidMortarShapefunctions::nurbs9),
        &embeddedmeshcoupling);

    Core::Utils::string_to_integral_parameter<EmbeddedMeshConstraintEnforcement>(
        "CONSTRAINT_ENFORCEMENT", "none",
        "Apply a constraint enforcement in the embedded mesh coupling strategy",
        tuple<std::string>("none", "penalty"),
        tuple<EmbeddedMeshConstraintEnforcement>(
            EmbeddedMeshConstraintEnforcement::none, EmbeddedMeshConstraintEnforcement::penalty),
        &embeddedmeshcoupling);

    Core::Utils::double_parameter("CONSTRAINT_ENFORCEMENT_PENALTYPARAM", 0.0,
        "Penalty parameter for the constraint enforcement in embedded mesh coupling",
        &embeddedmeshcoupling);
  }
}

FOUR_C_NAMESPACE_CLOSE