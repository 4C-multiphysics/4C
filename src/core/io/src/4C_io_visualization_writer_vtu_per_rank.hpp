// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VISUALIZATION_WRITER_VTU_PER_RANK_HPP
#define FOUR_C_IO_VISUALIZATION_WRITER_VTU_PER_RANK_HPP

#include "4C_config.hpp"

#include "4C_io_visualization_writer_base.hpp"
#include "4C_io_vtu_writer.hpp"

#include <fstream>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class VisualizationWriterVtuPerRank : public VisualizationWriterBase
  {
   public:
    /**
     * @brief Default constructor
     */
    VisualizationWriterVtuPerRank(const Core::IO::VisualizationParameters& parameters,
        MPI_Comm comm, std::string visualization_data_name);

    /**
     * @brief Default destructor
     */
    ~VisualizationWriterVtuPerRank() override = default;

    /**
     * @brief Initialize the current time step (derived)
     */
    void initialize_time_step(
        const double visualization_time, const int visualization_step) override;

    /**
     * @brief Write all fields contained in the field data map to disk (derived)
     */
    void write_field_data_to_disk(
        const std::map<std::string, visualization_vector_type_variant>& field_data_map) override;

    /**
     * @brief Write the full geometry, i.e., points, cells, faces and the respective connectivity to
     * disk (derived)
     */
    void write_geometry_to_disk(const std::vector<double>& point_coordinates,
        const std::vector<Core::IO::index_type>& point_cell_connectivity,
        const std::vector<Core::IO::index_type>& cell_offset,
        const std::vector<uint8_t>& cell_types,
        const std::vector<Core::IO::index_type>& face_connectivity,
        const std::vector<Core::IO::index_type>& face_offset) override;

    /**
     * @brief Write a single point data vector to disk (derived)
     */
    void write_point_data_vector_to_disk(const visualization_vector_type_variant& data,
        unsigned int num_components_per_point, const std::string& name) override;

    /**
     * @brief Write a single cell data vector to disk (derived)
     */
    void write_cell_data_vector_to_disk(const visualization_vector_type_variant& data,
        unsigned int num_components_per_point, const std::string& name) override;

    /**
     * @brief Finalize the write operations for the current time step (derived)
     */
    void finalize_time_step() override;

    //! VtuWriter used for the VTU/PVTU format serialization
    VtuWriter vtu_writer_;

   private:
    //! Output stream for the file of this processor (one .vtu per rank)
    std::ofstream rank_file_;

    //! Output stream for the parallel (master) file (only proc 0, .pvtu)
    std::ofstream master_file_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif