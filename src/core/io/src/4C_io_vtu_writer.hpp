// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VTU_WRITER_HPP
#define FOUR_C_IO_VTU_WRITER_HPP


#include "4C_config.hpp"

#include "4C_io_vtk_writer_base.hpp"

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*
 \brief class for VTU output generation

 The VtuWriter writes the VTU content into output streams provided by the caller. The caller owns
 the streams (e.g. per-rank files or a buffer that is later written collectively via MPI-IO).

*/
class VtuWriter : public VtkWriterBase
{
 public:
  //! constructor
  VtuWriter(unsigned int myrank, unsigned int num_processors,
      unsigned int max_number_timesteps_to_be_written,
      const std::string& path_existing_working_directory,
      const std::string& name_new_vtk_subdirectory, const std::string& geometry_name,
      const std::string& restart_name, double restart_time, bool write_binary_output,
      LibB64::CompressionLevel compression_level);

  //! set up the base file name (geometry name and current time step) for the current output
  void initialize_current_time_step_output_file_name();

  //! full file name (path + base name + processor id + suffix) of this processor's output file
  std::string output_file_name_this_processor() const;

  //! full file name (path + base name + suffix) of the parallel (master) output file
  std::string output_file_name_master() const;

  //! append the current master file and time value to the pvd collection content (only proc 0)
  void append_master_file_and_time_to_collection_file_mid_section_content();

  //! write the prologue of the vtk files into the given streams
  void write_vtk_headers(std::ostream& filestream, std::ostream& masterfilestream);

  //! write the geometry defining this unstructured grid
  void write_geometry_unstructured_grid(std::ostream& filestream, std::ostream& masterfilestream,
      const std::vector<double>& point_coordinates,
      const std::vector<Core::IO::index_type>& point_cell_connectivity,
      const std::vector<Core::IO::index_type>& cell_offset, const std::vector<uint8_t>& cell_types,
      const std::vector<Core::IO::index_type>& face_connectivity,
      const std::vector<Core::IO::index_type>& face_offset) const;


  //! write a data vector with num_component values of type T per point
  void write_point_data_vector(std::ostream& filestream, std::ostream& masterfilestream,
      const Core::IO::visualization_vector_type_variant& data,
      unsigned int num_components_per_point, const std::string& name);

  //! write a data vector with num_component values of type T per cell
  void write_cell_data_vector(std::ostream& filestream, std::ostream& masterfilestream,
      const Core::IO::visualization_vector_type_variant& data, unsigned int num_components_per_cell,
      const std::string& name);

  //! write field data array into the given file stream
  void write_vtk_field_data_and_or_time_and_or_cycle(std::ostream& filestream,
      const std::map<std::string, Core::IO::visualization_vector_type_variant>& field_data_map);

  //! write field data array for time and cycle into the given file stream [for restart
  //! information]
  void write_vtk_time_and_or_cycle(std::ostream& filestream);

  //! write the epilogue of the vtk files into the given streams
  void write_vtk_footers(std::ostream& filestream, std::ostream& masterfilestream);


 protected:
  //! write a data vector as DataArray to the given streams
  // Todo template <typename T>
  void write_data_array(std::ostream& filestream, std::ostream& masterfilestream,
      const Core::IO::visualization_vector_type_variant& data, const int num_components,
      const std::string& name);

  //! Return the opening xml tag for this writer type
  const std::string& writer_opening_tag() const override;

  //! Return the parallel opening xml tag for this writer type
  const std::string& writer_p_opening_tag() const override;

  //! Return a vector of parallel piece tags for each file
  const std::vector<std::string>& writer_p_piece_tags() const override;

  //! Return the parallel file suffix including the dot for this file type
  const std::string& writer_p_suffix() const override;

  //! Return the string of this writer type
  const std::string& writer_string() const override;

  //! Return the file suffix including the dot for this file type
  const std::string& writer_suffix() const override;

 private:
  //! write prologue of the VTK master file (handled by proc 0)
  void write_vtk_header_master_file(
      std::ostream& masterfilestream, const std::string& byteorder) const;

  //! write prologue of the VTK file on this processor
  void write_vtk_header_this_processor(
      std::ostream& filestream, const std::string& byteorder) const;

  //! write field data array into the given file stream
  template <typename T>
  void write_field_data_array(
      std::ostream& filestream, const std::string& name, const std::vector<T>& field_data);

  //! write the required information about the DataArray to master file
  // Todo template <typename T>
  void write_data_array_master_file(std::ostream& masterfilestream, const int num_components,
      const std::string& name, const std::string& data_type_name) const;

  //! write the data array into the given file stream
  template <typename T>
  void write_data_array_this_processor(std::ostream& filestream, const std::vector<T>& data,
      const int num_components, const std::string& name);

  //! write epilogue of of the VTK master file (handled by proc 0)
  void write_vtk_footer_master_file(std::ostream& masterfilestream) const;

  //! write epilogue of the VTK file on this processor
  void write_vtk_footer_this_processor(std::ostream& filestream) const;
};

FOUR_C_NAMESPACE_CLOSE

#endif