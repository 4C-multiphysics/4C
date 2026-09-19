// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_visualization_writer_vtu_per_rank.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_visualization_data.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
Core::IO::VisualizationWriterVtuPerRank::VisualizationWriterVtuPerRank(
    const Core::IO::VisualizationParameters& parameters, MPI_Comm comm,
    std::string visualization_data_name)
    : VisualizationWriterBase(parameters, comm, std::move(visualization_data_name)),
      vtu_writer_(Core::Communication::my_mpi_rank(comm), Core::Communication::num_mpi_ranks(comm),
          std::pow(10, Core::IO::get_total_digits_to_reserve_in_time_step(parameters)),
          parameters.directory_name_, (parameters.file_name_prefix_ + "-vtk-files"),
          visualization_data_name_, parameters.restart_from_name_, parameters.restart_time_,
          parameters.data_format_ == OutputDataFormat::binary, parameters.compression_level_)
{
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuPerRank::initialize_time_step(
    const double visualization_time, const int visualization_step)
{
  vtu_writer_.reset_time_and_time_step(visualization_time, visualization_step);

  vtu_writer_.initialize_current_time_step_output_file_name();

  rank_file_.open(vtu_writer_.output_file_name_this_processor());

  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    master_file_.open(vtu_writer_.output_file_name_master());

    vtu_writer_.append_master_file_and_time_to_collection_file_mid_section_content();
  }

  vtu_writer_.write_vtk_headers(rank_file_, master_file_);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuPerRank::write_field_data_to_disk(
    const std::map<std::string, visualization_vector_type_variant>& field_data_map)
{
  vtu_writer_.write_vtk_field_data_and_or_time_and_or_cycle(rank_file_, field_data_map);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuPerRank::write_geometry_to_disk(
    const std::vector<double>& point_coordinates,
    const std::vector<Core::IO::index_type>& point_cell_connectivity,
    const std::vector<Core::IO::index_type>& cell_offset, const std::vector<uint8_t>& cell_types,
    const std::vector<Core::IO::index_type>& face_connectivity,
    const std::vector<Core::IO::index_type>& face_offset)
{
  vtu_writer_.write_geometry_unstructured_grid(rank_file_, master_file_, point_coordinates,
      point_cell_connectivity, cell_offset, cell_types, face_connectivity, face_offset);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuPerRank::write_point_data_vector_to_disk(
    const visualization_vector_type_variant& data, unsigned int num_components_per_point,
    const std::string& name)
{
  vtu_writer_.write_point_data_vector(
      rank_file_, master_file_, data, num_components_per_point, name);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuPerRank::write_cell_data_vector_to_disk(
    const visualization_vector_type_variant& data, unsigned int num_components_per_point,
    const std::string& name)
{
  vtu_writer_.write_cell_data_vector(
      rank_file_, master_file_, data, num_components_per_point, name);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuPerRank::finalize_time_step()
{
  vtu_writer_.write_vtk_footers(rank_file_, master_file_);

  rank_file_.close();

  if (Core::Communication::my_mpi_rank(comm_) == 0) master_file_.close();

  // Write a collection file summarizing all previously written files
  vtu_writer_.write_vtk_collection_file_for_all_written_master_files(
      parameters_.file_name_prefix_ + "-" + visualization_data_name_);
}
FOUR_C_NAMESPACE_CLOSE
