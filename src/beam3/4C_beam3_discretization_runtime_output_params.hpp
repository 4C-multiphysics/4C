// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAM3_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP
#define FOUR_C_BEAM3_DISCRETIZATION_RUNTIME_OUTPUT_PARAMS_HPP


#include "4C_config.hpp"

#include "4C_beam3_discretization_runtime_output_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    /** \brief Input data container for output at runtime for beams
     *
     * */
    class BeamRuntimeOutputParams
    {
     public:
      /// constructor
      BeamRuntimeOutputParams();

      /// destructor
      virtual ~BeamRuntimeOutputParams() = default;

      /// initialize the class variables
      void init(const Teuchos::ParameterList& IO_vtk_structure_beams_paramslist);

      /// setup new class variables
      void setup();


      /// whether to write displacements
      bool output_displacement_state() const
      {
        check_init_setup();
        return output_displacement_state_;
      };

      /// whether to write triads at the Gauss points
      bool is_write_triad_visualization_points() const
      {
        check_init_setup();
        return write_triads_visualizationpoints_;
      };

      /// whether to use absolute positions or initial positions for the vtu geometry definition
      /// (i.e. for the visualization point coordinates)
      bool use_absolute_positions() const
      {
        check_init_setup();
        return use_absolute_positions_visualizationpoint_coordinates_;
      };

      /// whether to write material cross-section strains at the Gauss points
      bool is_write_internal_energy_element() const
      {
        check_init_setup();
        return write_internal_energy_element_;
      };

      /// whether to write material cross-section strains at the Gauss points
      bool is_write_kinetic_energy_element() const
      {
        check_init_setup();
        return write_kinetic_energy_element_;
      };

      /// whether to write material cross-section strains at the Gauss points
      bool is_write_material_strains_gauss_points() const
      {
        check_init_setup();
        return write_material_crosssection_strains_gausspoints_;
      };

      /// whether to write material cross-section strains at the visualization points
      bool is_write_material_strains_continuous() const
      {
        check_init_setup();
        return write_material_crosssection_strains_continuous_;
      };

      /// whether to write material cross-section stresses at the Gauss points
      bool is_write_material_stresses_gauss_points() const
      {
        check_init_setup();
        return write_material_crosssection_stresses_gausspoints_;
      };

      /// whether to write material cross-section stresses at the visualization points
      bool is_write_material_stress_continuous() const
      {
        check_init_setup();
        return write_material_crosssection_strains_continuous_;
      };

      /// whether to write material cross-section stresses at the Gauss points
      bool is_write_spatial_stresses_gauss_points() const
      {
        check_init_setup();
        return write_spatial_crosssection_stresses_gausspoints_;
      };

      /// whether to write material cross-section stresses at the Gauss points
      bool is_write_element_filament_condition() const
      {
        check_init_setup();
        return write_filament_condition_;
      };

      /// whether to write element and network orientation parameter
      bool is_write_orientation_parameter() const
      {
        check_init_setup();
        return write_orientation_parameter_;
      };

      /// whether to write crosssection forces of periodic rve in x, y, and z direction
      bool is_write_rve_crosssection_forces() const
      {
        check_init_setup();
        return write_rve_crosssection_forces_;
      };

      /// whether to write the element reference length
      bool is_write_ref_length() const
      {
        check_init_setup();
        return write_ref_length_;
      };

      /// whether to write beam element GIDs.
      bool is_write_element_gid() const
      {
        check_init_setup();
        return write_element_gid_;
      };

      /// write ghosting information
      bool is_write_element_ghosting() const
      {
        check_init_setup();
        return write_element_ghosting_;
      };

      /// number of visualization subsegments.
      unsigned int get_number_visualization_subsegments() const
      {
        check_init_setup();
        return n_subsegments_;
      };

     private:
      /// get the init indicator status
      const bool& is_init() const { return isinit_; };

      /// get the setup indicator status
      const bool& is_setup() const { return issetup_; };

      /// Check if init() and setup() have been called, yet.
      void check_init_setup() const;


     private:
      /// @name variables for internal use only
      /// @{
      ///
      bool isinit_;

      bool issetup_;
      /// @}

      /// @name variables controlling output
      /// @{

      /// whether to write displacement output
      bool output_displacement_state_;

      /// whether to use absolute positions or initial positions for the vtu geometry definition
      /// (i.e. for the visualization point coordinates)
      bool use_absolute_positions_visualizationpoint_coordinates_;

      /// whether to write internal (elastic) energy for each element
      bool write_internal_energy_element_;

      /// whether to write kinetic energy for each element
      bool write_kinetic_energy_element_;

      /// whether to write triads at the visualization points
      bool write_triads_visualizationpoints_;

      /// whether to write material cross-section strains at the Gauss points
      bool write_material_crosssection_strains_gausspoints_;

      /// whether to write material cross-section strains at the visualization points
      bool write_material_crosssection_strains_continuous_;

      /// whether to write material cross-section stresses at the Gauss points
      bool write_material_crosssection_stresses_gausspoints_;

      /// whether to write spatial cross-section stresses at the Gauss points
      bool write_spatial_crosssection_stresses_gausspoints_;

      /// whether to write beam filament condition (id, type)
      bool write_filament_condition_;

      /// whether to write element and network orientation parameter
      bool write_orientation_parameter_;

      /// whether to write crosssection forces of periodic rve in x, y, and z direction
      bool write_rve_crosssection_forces_;

      /// whether to write the element GIDs.
      bool write_ref_length_;

      /// whether to write the element GIDs.
      bool write_element_gid_;

      /// whether to write the element ghosting information.
      bool write_element_ghosting_;

      /// number of visualization subsegments
      unsigned int n_subsegments_;

      //@}
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
