// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_element_definition.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
//! Print function
/*----------------------------------------------------------------------*/
void print_element_dat_header()
{
  Core::Elements::ElementDefinition ed;
  ed.print_element_dat_header_to_stream(std::cout);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Elements::ElementDefinition::print_element_dat_header_to_stream(std::ostream& stream)
{
  setup_valid_element_lines();

  print_section_header(stream, "STRUCTURE ELEMENTS");

  print_element_lines(stream, "BEAM3R");
  print_element_lines(stream, "BEAM3EB");
  print_element_lines(stream, "BEAM3K");
  print_element_lines(stream, "BELE3_3");
  print_element_lines(stream, "RIGIDSPHERE");
  print_element_lines(stream, "SHELL7P");
  print_element_lines(stream, "SHELL7PSCATRA");
  print_element_lines(stream, "SHELL_KIRCHHOFF_LOVE_NURBS");
  print_element_lines(stream, "SOLID");
  print_element_lines(stream, "SOLIDPORO_PRESSURE_BASED");
  print_element_lines(stream, "SOLIDPORO_PRESSURE_VELOCITY_BASED");
  print_element_lines(stream, "SOLIDSCATRA");
  print_element_lines(stream, "SOLIDH27_DEPRECATED");
  print_element_lines(stream, "SOLIDH27PORO");
  print_element_lines(stream, "SOLIDH8_DEPRECATED");
  print_element_lines(stream, "MEMBRANE3");
  print_element_lines(stream, "SOLIDH8PORO");
  print_element_lines(stream, "SOLIDH8POROSCATRA");
  print_element_lines(stream, "SOLIDH8POROP1");
  print_element_lines(stream, "SOLIDH8POROP1SCATRA");
  print_element_lines(stream, "SOLIDSH8");
  print_element_lines(stream, "SOLIDT10_DEPRECATED");
  print_element_lines(stream, "SOLIDT4_DEPRECATED");
  print_element_lines(stream, "SOLIDT4PORO");
  print_element_lines(stream, "TORSION3");
  print_element_lines(stream, "TRUSS3");
  print_element_lines(stream, "TRUSS3SCATRA");
  print_element_lines(stream, "WALL");
  print_element_lines(stream, "WALLNURBS");
  print_element_lines(stream, "WALLSCATRA");
  print_element_lines(stream, "WALLQ4PORO");
  print_element_lines(stream, "WALLQ4POROSCATRA");
  print_element_lines(stream, "WALLQ4POROP1");
  print_element_lines(stream, "WALLQ4POROP1SCATRA");
  print_element_lines(stream, "WALLQ9PORO");


  print_section_header(stream, "FLUID ELEMENTS");
  print_element_lines(stream, "FLUID");
  print_element_lines(stream, "FLUIDXW");
  print_element_lines(stream, "FLUIDHDG");
  print_element_lines(stream, "FLUIDHDGWEAKCOMP");
  print_element_lines(stream, "FLUIDIMMERSED");

  print_section_header(stream, "LUBRICATION ELEMENTS");
  print_element_lines(stream, "LUBRICATION");

  print_section_header(stream, "TRANSPORT ELEMENTS");
  print_element_lines(stream, "TRANSP");

  print_section_header(stream, "TRANSPORT2 ELEMENTS");
  print_element_lines(stream, "TRANSP");

  print_section_header(stream, "ALE ELEMENTS");
  print_element_lines(stream, "ALE2");
  print_element_lines(stream, "ALE3");

  print_section_header(stream, "THERMO ELEMENTS");
  print_element_lines(stream, "THERMO");

  print_section_header(stream, "ARTERY ELEMENTS");
  print_element_lines(stream, "ART");

  print_section_header(stream, "REDUCED D AIRWAYS ELEMENTS");
  print_element_lines(stream, "RED_AIRWAY");
  print_element_lines(stream, "RED_ACINUS");
  print_element_lines(stream, "RED_ACINAR_INTER_DEP");

  print_section_header(stream, "ELECTROMAGNETIC ELEMENTS");
  print_element_lines(stream, "ELECTROMAGNETIC");
  print_element_lines(stream, "ELECTROMAGNETICDIFF");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Elements::ElementDefinition::print_section_header(std::ostream& stream, std::string name)
{
  unsigned l = name.length();
  stream << "--";
  for (int i = 0; i < std::max<int>(65 - l, 0); ++i) stream << '-';
  stream << name << '\n';
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Elements::ElementDefinition::print_element_lines(std::ostream& stream, std::string name)
{
  FOUR_C_ASSERT(definitions_.contains(name), "Element type not found: " + name);
  auto& defs = definitions_[name];

  const Core::IO::InputParameterContainer container;
  for (const auto& [cell_type, spec] : defs)
  {
    stream << "// 0 " << name << " ";
    spec.print_as_dat(stream, container);
    stream << '\n';
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::Elements::ElementDefinition::setup_valid_element_lines()
{
  Core::Communication::ParObjectFactory::instance().setup_element_definition(definitions_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::IO::InputSpec& Core::Elements::ElementDefinition::element_lines(
    std::string name, std::string cell_type)
{
  FOUR_C_ASSERT(definitions_.contains(name), "Element type not found: " + name);
  auto& defs = definitions_.at(name);
  FOUR_C_ASSERT(defs.contains(cell_type), "Cell type not found: " + cell_type);
  return defs.at(cell_type);
}

FOUR_C_NAMESPACE_CLOSE
