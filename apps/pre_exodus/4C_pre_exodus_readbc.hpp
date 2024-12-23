// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PRE_EXODUS_READBC_HPP
#define FOUR_C_PRE_EXODUS_READBC_HPP

#include "4C_config.hpp"

#include "4C_legacy_enum_definitions_conditions.hpp"
#include "4C_pre_exodus_reader.hpp"
#include "4C_utils_exceptions.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace EXODUS
{
  //! differentiate between underlying mesh entity
  enum MeshEntity
  {
    bceb,
    bcns,
    bcss
  };

  //! differentiate between corresponding condition_type
  enum CondType
  {
    element,
    dvol,
    dsurf,
    dline,
    dpoint,
    empty,
    invalid
  };

  //! this is what fully defines a 4C element
  struct ElemDef
  {
    int id;             ///< referring to mesh_entity id of eb,ns,ss
    MeshEntity me;      ///< referring to underlying mesh entity
    std::string sec;    ///< FLUID,STRUCTURE,ALE,etc.
    std::string desc;   ///< like "MAT 1 EAS full"
    std::string ename;  ///< FLUID,SOLIDSH8,etc
  };

  //! this is what fully defines a 4C condition
  struct CondDef
  {
    int id;            ///< referring to mesh_entity id of eb,ns,ss
    MeshEntity me;     ///< referring to underlying mesh entity
    std::string sec;   ///< see valid_condition 'sectionname'
    std::string desc;  ///< see valid_condition 'description'
    int e_id;          ///< refers to datfile 'E num -'
    Core::Conditions::GeometryType gtype;
  };

  void read_bc_file(const std::string& bcfile, std::vector<EXODUS::ElemDef>& eledefs,
      std::vector<EXODUS::CondDef>& condefs);

  EXODUS::ElemDef read_edef(
      const std::string& mesh_entity, const int id, const std::string& actcond);

  EXODUS::CondDef read_cdef(
      const std::string& mesh_entity, const int id, const std::string& actcond);

  //! Read bc_entity specifications
  std::vector<std::string> read_bc_entity(const std::string actcond);

  //! Check condition type against valid types
  inline EXODUS::CondType check_cond_type(const std::string buffer);

  //! Conversion
  inline std::string cond_type_to_string(const EXODUS::CondType);

  //! Print bc_entity
  void print_bc_def(std::ostream& os, const EXODUS::ElemDef& def);
  void print_bc_def(std::ostream& os, const EXODUS::CondDef& def);

  // ! Check if periodic boundary conditions are defined
  bool periodic_boundary_conditions_found(std::vector<EXODUS::CondDef> condefs);

  // ! Correct nodal coordinates for periodic boundary conditions
  void correct_nodal_coordinates_for_periodic_boundary_conditions(
      EXODUS::Mesh& mesh, std::vector<EXODUS::CondDef> condefs);

  // ! Correct nodal coordinates in the YZ plane for periodic boundary conditions
  void correct_yz_plane_for_periodic_boundary_conditions(
      EXODUS::Mesh& mesh, const std::vector<EXODUS::CondDef>& condefs);

  // ! Correct nodal coordinates in the XZ plane for periodic boundary conditions
  void correct_xz_plane_for_periodic_boundary_conditions(
      EXODUS::Mesh& mesh, const std::vector<EXODUS::CondDef>& condefs);

  // ! Correct nodal coordinates in the XY plane for periodic boundary conditions
  void correct_xy_plane_for_periodic_boundary_conditions(
      EXODUS::Mesh& mesh, const std::vector<EXODUS::CondDef>& condefs);

}  // namespace EXODUS

FOUR_C_NAMESPACE_CLOSE

#endif
