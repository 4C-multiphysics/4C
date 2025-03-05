// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_volmortar.hpp"

#include "4C_coupling_volmortar.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::VolMortar::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /* parameters for volmortar */
  Core::Utils::SectionSpecs volmortar{"VOLMORTAR COUPLING"};

  Core::Utils::string_to_integral_parameter<Coupling::VolMortar::IntType>("INTTYPE", "Elements",
      "Type of numerical integration scheme",
      tuple<std::string>("Elements", "elements", "Segments", "segments"),
      tuple<Coupling::VolMortar::IntType>(Coupling::VolMortar::inttype_elements,
          Coupling::VolMortar::inttype_elements, Coupling::VolMortar::inttype_segments,
          Coupling::VolMortar::inttype_segments),
      volmortar);

  Core::Utils::string_to_integral_parameter<Coupling::VolMortar::CouplingType>("COUPLINGTYPE",
      "Volmortar", "Type of coupling",
      tuple<std::string>("Volmortar", "volmortar", "consistentinterpolation", "consint"),
      tuple<Coupling::VolMortar::CouplingType>(Coupling::VolMortar::couplingtype_volmortar,
          Coupling::VolMortar::couplingtype_volmortar, Coupling::VolMortar::couplingtype_coninter,
          Coupling::VolMortar::couplingtype_coninter),
      volmortar);

  Core::Utils::string_to_integral_parameter<Coupling::VolMortar::Shapefcn>("SHAPEFCN", "Dual",
      "Type of employed set of shape functions",
      tuple<std::string>("Dual", "dual", "Standard", "standard", "std"),
      tuple<Coupling::VolMortar::Shapefcn>(Coupling::VolMortar::shape_dual,
          Coupling::VolMortar::shape_dual, Coupling::VolMortar::shape_std,
          Coupling::VolMortar::shape_std, Coupling::VolMortar::shape_std),
      volmortar);

  Core::Utils::string_to_integral_parameter<Coupling::VolMortar::CutType>("CUTTYPE", "dd",
      "Type of cut procedure/ integration point calculation",
      tuple<std::string>(
          "dd", "directdivergence", "DirectDivergence", "tessellation", "t", "Tessellation"),
      tuple<Coupling::VolMortar::CutType>(Coupling::VolMortar::cuttype_directdivergence,
          Coupling::VolMortar::cuttype_directdivergence,
          Coupling::VolMortar::cuttype_directdivergence, Coupling::VolMortar::cuttype_tessellation,
          Coupling::VolMortar::cuttype_tessellation, Coupling::VolMortar::cuttype_tessellation),
      volmortar);

  Core::Utils::string_to_integral_parameter<Coupling::VolMortar::DualQuad>("DUALQUAD", "nomod",
      "Type of dual shape function for weighting function for quadr. problems",
      tuple<std::string>("nm", "nomod", "lm", "lin_mod", "qm", "quad_mod"),
      tuple<Coupling::VolMortar::DualQuad>(Coupling::VolMortar::dualquad_no_mod,
          Coupling::VolMortar::dualquad_no_mod, Coupling::VolMortar::dualquad_lin_mod,
          Coupling::VolMortar::dualquad_lin_mod, Coupling::VolMortar::dualquad_quad_mod,
          Coupling::VolMortar::dualquad_quad_mod),
      volmortar);

  volmortar.specs.emplace_back(parameter<bool>(
      "MESH_INIT", {.description = "If chosen, mesh initialization procedure is performed",
                       .default_value = false}));

  volmortar.specs.emplace_back(parameter<bool>("KEEP_EXTENDEDGHOSTING",
      {.description = "If chosen, extended ghosting is kept for simulation",
          .default_value = true}));

  volmortar.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
