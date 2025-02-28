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

#include <map>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*
 \brief class for VTU output generation
 \date 03/14, 03/17
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

  //! write the geometry defining this unstructured grid
  void write_geometry_unstructured_grid(const std::vector<double>& point_coordinates,
      const std::vector<Core::IO::index_type>& point_cell_connectivity,
      const std::vector<Core::IO::index_type>& cell_offset, const std::vector<uint8_t>& cell_types,
      const std::vector<Core::IO::index_type>& face_connectivity,
      const std::vector<Core::IO::index_type>& face_offset);


  //! write a data vector with num_component values of type T per point
  void write_point_data_vector(const Core::IO::visualization_vector_type_variant& data,
      unsigned int num_components_per_point, const std::string& name);

  //! write a data vector with num_component values of type T per cell
  void write_cell_data_vector(const Core::IO::visualization_vector_type_variant& data,
      unsigned int num_components_per_cell, const std::string& name);


 protected:
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
};

FOUR_C_NAMESPACE_CLOSE

#endif
