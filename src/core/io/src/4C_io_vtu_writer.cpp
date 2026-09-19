// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_vtu_writer.hpp"

#include "4C_io_pstream.hpp"
#include "4C_io_vtk_writer_base.hpp"
#include "4C_utils_exceptions.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <variant>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
VtuWriter::VtuWriter(unsigned int myrank, unsigned int num_processors,
    unsigned int max_number_timesteps_to_be_written,
    const std::string& path_existing_working_directory,
    const std::string& name_new_vtk_subdirectory, const std::string& geometry_name,
    const std::string& restart_name, double restart_time, bool write_binary_output,
    LibB64::CompressionLevel compression_level)
    : VtkWriterBase(myrank, num_processors, max_number_timesteps_to_be_written,
          path_existing_working_directory, name_new_vtk_subdirectory, geometry_name, restart_name,
          restart_time, write_binary_output, compression_level)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::initialize_current_time_step_output_file_name()
{
  std::ostringstream tmpstream;
  tmpstream << geometry_name_ << "-" << std::setfill('0') << std::setw(num_timestep_digits_)
            << timestep_;

  filename_base_ = tmpstream.str();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::string VtuWriter::output_file_name_this_processor() const
{
  std::ostringstream tmpstream;

  tmpstream << working_directory_full_path_ << "/" << filename_base_
            << get_part_of_file_name_indicating_processor_id(myrank_) << writer_suffix();

  return tmpstream.str();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::string VtuWriter::output_file_name_master() const
{
  return working_directory_full_path_ + "/" + filename_base_ + writer_p_suffix();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::append_master_file_and_time_to_collection_file_mid_section_content()
{
  if (myrank_ != 0) return;

  // append this new master file to the stream of all written files and times
  // for later use as vtk collection file ('.pvd')
  VtkWriterBase::append_master_file_and_time_to_collection_file_mid_section_content(
      filename_base_ + writer_p_suffix(),
      determine_vtk_subdirectory_name_from_full_vtk_working_path(), time_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_headers(std::ostream& filestream, std::ostream& masterfilestream)
{
  // Todo: might need BigEndian on some systems
  const std::string byteorder = "LittleEndian";

  // Todo: specify xml version, vtk DataFile Version, ... if needed

  // start master file on processor 0
  if (myrank_ == 0) write_vtk_header_master_file(masterfilestream, byteorder);

  // start file on each individual processor
  write_vtk_header_this_processor(filestream, byteorder);

  currentPhase_ = INIT;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_field_data_and_or_time_and_or_cycle(std::ostream& filestream,
    const std::map<std::string, Core::IO::visualization_vector_type_variant>& field_data_map)
{
  throw_error_if_invalid_file_stream(filestream);

  // Initialize field data section.
  filestream << "    <FieldData>\n";

  // If previously set add time and cycle to field data.
  if (time_ != std::numeric_limits<double>::min() || cycle_ != std::numeric_limits<int>::max())
  {
    if (time_ != std::numeric_limits<double>::min())
    {
      std::vector<double> temp_vector;
      temp_vector.resize(1);
      temp_vector[0] = time_;
      write_field_data_array(filestream, "TIME", temp_vector);
    }

    if (cycle_ != std::numeric_limits<int>::max())
    {
      std::vector<int> temp_vector;
      temp_vector.resize(1);
      temp_vector[0] = cycle_;
      write_field_data_array(filestream, "CYCLE", temp_vector);
    }
  }

  // Write every field data array.
  for (const auto& [field_name, field_data] : field_data_map)
    std::visit(
        [&](const auto& vec) { write_field_data_array(filestream, field_name, vec); }, field_data);

  // Finalize field data section.
  filestream << "    </FieldData>\n\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_time_and_or_cycle(std::ostream& filestream)
{
  std::map<std::string, Core::IO::visualization_vector_type_variant> empty_map;
  empty_map.clear();
  write_vtk_field_data_and_or_time_and_or_cycle(filestream, empty_map);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_header_master_file(
    std::ostream& masterfilestream, const std::string& byteorder) const
{
  throw_error_if_invalid_file_stream(masterfilestream);

  masterfilestream << "<?xml version=\"1.0\" ?> \n";
  masterfilestream << "<!-- \n";
  masterfilestream << "# vtk DataFile Version 3.0\n";
  masterfilestream << "-->\n";
  masterfilestream << "<VTKFile type=\"P" << this->writer_string() << R"(" version="0.1")";
  masterfilestream << " byte_order=\"" << byteorder << "\"";
  masterfilestream << ">\n";
  masterfilestream << "  " << this->writer_p_opening_tag() << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_header_this_processor(
    std::ostream& filestream, const std::string& byteorder) const
{
  throw_error_if_invalid_file_stream(filestream);

  filestream << "<?xml version=\"1.0\" ?> \n";
  filestream << "<!-- \n";
  filestream << "# vtk DataFile Version 3.0\n";
  filestream << "-->\n";
  filestream << "<VTKFile type=\"" << this->writer_string() << R"(" version="0.1")";
  filestream << " compressor=\"vtkZLibDataCompressor\"";
  filestream << " byte_order=\"" << byteorder << "\"";
  filestream << ">\n";
  filestream << "  " << this->writer_opening_tag() << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <typename T>
void VtuWriter::write_field_data_array(
    std::ostream& filestream, const std::string& name, const std::vector<T>& field_data)
{
  const unsigned int n_data = field_data.size();

  // If the array is empty it is skipped.
  if (n_data == 0) return;

  // Set the header for the current field data array.
  filestream << "      <DataArray type=\"";
  filestream << scalar_type_to_vtk_type<T>();
  filestream << "\" Name=\"";
  filestream << name;
  filestream << R"(" NumberOfTuples="1")";
  if (n_data > 1) filestream << " NumberOfComponents=\"" << n_data << "\"";
  filestream << " format=\"ascii\">\n";

  // Add the field data.
  filestream << std::setprecision(15) << std::scientific;
  for (unsigned int i = 0; i < n_data; i++) filestream << field_data[i] << " ";
  filestream << std::resetiosflags(std::ios::scientific);

  // Finish the current field data array.
  filestream << "\n      </DataArray>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_data_array(std::ostream& filestream, std::ostream& masterfilestream,
    const Core::IO::visualization_vector_type_variant& data, const int num_components,
    const std::string& name)
{
  std::string vtk_type_name = "";
  if (std::holds_alternative<std::vector<double>>(data))
  {
    write_data_array_this_processor(
        filestream, std::get<std::vector<double>>(data), num_components, name);
    vtk_type_name = scalar_type_to_vtk_type<double>();
  }
  else if (std::holds_alternative<std::vector<int>>(data))
  {
    write_data_array_this_processor(
        filestream, std::get<std::vector<int>>(data), num_components, name);
    vtk_type_name = scalar_type_to_vtk_type<int>();
  }
  else
  {
    FOUR_C_THROW("Got unexpected vector type");
  }

  if (myrank_ == 0)
    write_data_array_master_file(masterfilestream, num_components, name, vtk_type_name);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_data_array_master_file(std::ostream& masterfilestream,
    const int num_components, const std::string& name, const std::string& data_type_name) const
{
  throw_error_if_invalid_file_stream(masterfilestream);


  masterfilestream << "      <PDataArray type=\"" << data_type_name.c_str() << "\" Name=\"" << name
                   << "\"";

  if (num_components > 1) masterfilestream << " NumberOfComponents=\"" << num_components << "\"";

  masterfilestream << " format=\"ascii\"/>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <typename T>
void VtuWriter::write_data_array_this_processor(std::ostream& filestream,
    const std::vector<T>& data, const int num_components, const std::string& name)
{
  throw_error_if_invalid_file_stream(filestream);

  filestream << "        <DataArray type=\"" << scalar_type_to_vtk_type<T>() << "\" Name=\"" << name
             << "\"";

  if (num_components > 1) filestream << " NumberOfComponents=\"" << num_components << "\"";

  if (write_binary_output_)
  {
    filestream << " format=\"binary\">\n";

    LibB64::write_compressed_block(data, filestream, compression_level_);
  }
  else
  {
    filestream << " format=\"ascii\">\n";

    int counter = 1;
    for (auto it = data.begin(); it != data.end(); ++it)
    {
      filestream << std::setprecision(15) << std::scientific << *it;

      if (counter % num_components != 0)
        filestream << " ";
      else
        filestream << '\n';

      counter++;
    }

    filestream << std::resetiosflags(std::ios::scientific);
  }

  filestream << "        </DataArray>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_footers(std::ostream& filestream, std::ostream& masterfilestream)
{
  throw_error_if_invalid_file_stream(filestream);
  if (myrank_ == 0) throw_error_if_invalid_file_stream(masterfilestream);

  // end the scalar fields
  switch (currentPhase_)
  {
    case POINTS:
    {
      filestream << "      </PointData>\n\n";

      if (myrank_ == 0) masterfilestream << "    </PPointData>\n";

      currentPhase_ = FINAL;

      break;
    }

    case CELLS:
    {
      filestream << "      </CellData>\n\n";

      if (myrank_ == 0) masterfilestream << "    </PCellData>\n";

      currentPhase_ = FINAL;

      break;
    }

    default:
    {
      FOUR_C_THROW("No data was written or writer was already in final phase.");

      break;
    }
  }

  if (myrank_ == 0) write_vtk_footer_master_file(masterfilestream);

  write_vtk_footer_this_processor(filestream);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_footer_master_file(std::ostream& masterfilestream) const
{
  throw_error_if_invalid_file_stream(masterfilestream);

  // generate information about 'pieces' (piece = part that is written by individual processor)
  using pptags_type = std::vector<std::string>;
  const pptags_type& ppiecetags = this->writer_p_piece_tags();

  if (numproc_ != ppiecetags.size()) FOUR_C_THROW("Incorrect number of Pieces.");

  for (const auto& ppiecetag : ppiecetags) masterfilestream << "    " << ppiecetag << "\n";

  masterfilestream << "  </P" << this->writer_string() << ">\n";
  masterfilestream << "</VTKFile>\n";
  masterfilestream << std::flush;

  if (myrank_ == 0)
  {
    Core::IO::cout(Core::IO::verbose)
        << "\nVtk Files '" << filename_base_ << "' written. Time: " << std::scientific
        << std::setprecision(std::numeric_limits<double>::digits10 - 1) << time_ << Core::IO::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_vtk_footer_this_processor(std::ostream& filestream) const
{
  throw_error_if_invalid_file_stream(filestream);

  filestream << "    </Piece>\n";
  filestream << "  </" << this->writer_string() << ">\n";
  filestream << "</VTKFile>\n";
  filestream << std::flush;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::string& VtuWriter::writer_string() const
{
  static std::string name("UnstructuredGrid");
  return name;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::string& VtuWriter::writer_opening_tag() const
{
  static std::string tag("<UnstructuredGrid>");
  return tag;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::string& VtuWriter::writer_p_opening_tag() const
{
  static std::string tag("<PUnstructuredGrid GhostLevel=\"0\">");
  return tag;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::vector<std::string>& VtuWriter::writer_p_piece_tags() const
{
  static std::vector<std::string> tags;
  tags.clear();

  for (size_t iproc = 0; iproc < numproc_; ++iproc)
  {
    std::stringstream stream;
    stream << "<Piece Source=\"" << filename_base_
           << get_part_of_file_name_indicating_processor_id(iproc) << ".vtu\"/>";
    tags.push_back(std::string(stream.str()));
  }
  return tags;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::string& VtuWriter::writer_suffix() const
{
  static std::string name(".vtu");
  return name;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::string& VtuWriter::writer_p_suffix() const
{
  static std::string name(".pvtu");
  return name;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_geometry_unstructured_grid(std::ostream& filestream,
    std::ostream& masterfilestream, const std::vector<double>& point_coordinates,
    const std::vector<Core::IO::index_type>& point_cell_connectivity,
    const std::vector<Core::IO::index_type>& cell_offset, const std::vector<uint8_t>& cell_types,
    const std::vector<Core::IO::index_type>& face_connectivity,
    const std::vector<Core::IO::index_type>& face_offset) const
{
  // always assume 3D for now Todo maybe use this as template to allow for 2D case
  const unsigned int num_spatial_dimensions = 3;

  const unsigned int num_points = point_coordinates.size() / num_spatial_dimensions;

  const unsigned int num_cells = cell_types.size();

  // some sanity checks
  if (point_coordinates.size() % num_spatial_dimensions != 0)
    FOUR_C_THROW("VtuWriter assumes 3D point coordinates here! Extend to 2D if needed");

  if (cell_offset.size() != cell_types.size())
    FOUR_C_THROW(
        "VtuWriter: number of specified cell types does not equal number of "
        "specified index offsets");


  // step 0: tell the master file that we specify point coordinates
  /*----------------------------------------------------------------------*/
  if (myrank_ == 0)
  {
    throw_error_if_invalid_file_stream(masterfilestream);

    masterfilestream << "    <PPoints>\n";
    masterfilestream << R"(      <PDataArray type="Float64" NumberOfComponents=")"
                     << num_spatial_dimensions << "\"/>\n";
    masterfilestream << "    </PPoints>\n";
  }


  // step 1: write point coordinates into file
  /*----------------------------------------------------------------------*/
  throw_error_if_invalid_file_stream(filestream);

  filestream << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\"" << num_cells
             << "\" >\n"
             << "      <Points>\n"
             << R"(        <DataArray type="Float64" NumberOfComponents=")"
             << num_spatial_dimensions << "\"";

  if (write_binary_output_)
  {
    filestream << " format=\"binary\">\n";
    LibB64::write_compressed_block(point_coordinates, filestream, compression_level_);
  }
  else
  {
    filestream << " format=\"ascii\">\n";

    int counter = 1;
    for (const double point_coordinate : point_coordinates)
    {
      filestream << std::setprecision(15) << std::scientific << point_coordinate;

      // single space between dimensions, new line upon completion of a point
      if (counter % num_spatial_dimensions != 0)
        filestream << " ";
      else
        filestream << '\n';

      counter++;
    }

    filestream << std::resetiosflags(std::ios::scientific);
  }


  filestream << "        </DataArray>\n"
             << "      </Points>\n\n";



  // step 2: write mesh-point topology into file
  /*----------------------------------------------------------------------*/
  filestream << "      <Cells>\n"
             << R"(        <DataArray type="Int32" Name="connectivity")";

  if (write_binary_output_)
  {
    filestream << " format=\"binary\">\n";
    LibB64::write_compressed_block(point_cell_connectivity, filestream, compression_level_);
  }
  else
  {
    filestream << " format=\"ascii\">\n";

    for (const int it : point_cell_connectivity) filestream << it << " ";
  }

  filestream << "\n        </DataArray>\n";



  // step 3: write indices where individual cells begin
  /*----------------------------------------------------------------------*/
  filestream << R"(        <DataArray type="Int32" Name="offsets")";

  if (write_binary_output_)
  {
    filestream << " format=\"binary\">\n";
    LibB64::write_compressed_block(cell_offset, filestream, compression_level_);
  }
  else
  {
    filestream << " format=\"ascii\">\n";
    for (const int it : cell_offset) filestream << it << " ";
  }

  filestream << "\n        </DataArray>\n";



  // step 4: write cell types
  /*----------------------------------------------------------------------*/
  filestream << R"(        <DataArray type="UInt8" Name="types")";
  if (write_binary_output_)
  {
    filestream << " format=\"binary\">\n";
    LibB64::write_compressed_block(cell_types, filestream, compression_level_);
  }
  else
  {
    filestream << " format=\"ascii\">\n";
    for (const unsigned char cell_type : cell_types)
      filestream << static_cast<unsigned int>(cell_type) << " ";
  }
  filestream << "\n        </DataArray>\n";

  // step 5: write face data if required
  if (face_offset.size() > 0)
  {
    // Face connectivity
    filestream << R"(        <DataArray type="Int32" Name="faces")";
    if (write_binary_output_)
    {
      filestream << " format=\"binary\">\n";
      LibB64::write_compressed_block(face_connectivity, filestream, compression_level_);
    }
    else
    {
      filestream << " format=\"ascii\">\n";
      for (const int value : face_connectivity) filestream << value << " ";
    }
    filestream << "\n        </DataArray>\n";

    // Face offsets
    filestream << R"(        <DataArray type="Int32" Name="faceoffsets")";
    if (write_binary_output_)
    {
      filestream << " format=\"binary\">\n";
      LibB64::write_compressed_block(face_offset, filestream, compression_level_);
    }
    else
    {
      filestream << " format=\"ascii\">\n";
      for (const int value : face_offset) filestream << value << " ";
    }
    filestream << "\n        </DataArray>\n";
  }

  filestream << "      </Cells>\n\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_point_data_vector(std::ostream& filestream, std::ostream& masterfilestream,
    const Core::IO::visualization_vector_type_variant& data, unsigned int num_components_per_point,
    const std::string& name)
{
  // start the point data section that will be written subsequently
  if (currentPhase_ == INIT)
  {
    throw_error_if_invalid_file_stream(filestream);
    filestream << "  <PointData>\n";

    if (myrank_ == 0)
    {
      throw_error_if_invalid_file_stream(masterfilestream);
      masterfilestream << "    <PPointData>\n";
    }

    currentPhase_ = POINTS;
  }

  if (currentPhase_ != POINTS)
    FOUR_C_THROW(
        "VtuWriter cannot write point data at this stage. Most likely, cell and "
        "point data fields are mixed. First, all point data needs to be written, "
        "then all cell data!");

  this->write_data_array(filestream, masterfilestream, data, num_components_per_point, name);

  if (myrank_ == 0)
    Core::IO::cout(Core::IO::debug)
        << "\nVtuWriter: point data " << name << " written." << Core::IO::endl;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtuWriter::write_cell_data_vector(std::ostream& filestream, std::ostream& masterfilestream,
    const Core::IO::visualization_vector_type_variant& data, unsigned int num_components_per_cell,
    const std::string& name)
{
  // if required, end the point data section
  if (currentPhase_ == POINTS)
  {
    throw_error_if_invalid_file_stream(filestream);
    filestream << "  </PointData>\n";

    if (myrank_ == 0)
    {
      throw_error_if_invalid_file_stream(masterfilestream);
      masterfilestream << "    </PPointData>\n";
    }
  }

  // start the cell data section that will be written subsequently
  if (currentPhase_ == INIT || currentPhase_ == POINTS)
  {
    throw_error_if_invalid_file_stream(filestream);
    filestream << "  <CellData>\n";

    if (myrank_ == 0)
    {
      throw_error_if_invalid_file_stream(masterfilestream);
      masterfilestream << "    <PCellData>\n";
    }

    currentPhase_ = CELLS;
  }

  if (currentPhase_ != CELLS)
    FOUR_C_THROW(
        "VtuWriter cannot write cell data at this stage. Most likely, cell and "
        "point data fields are mixed. First, all point data needs to be written, "
        "then all cell data!");

  this->write_data_array(filestream, masterfilestream, data, num_components_per_cell, name);

  if (myrank_ == 0)
    Core::IO::cout(Core::IO::debug)
        << "\nVtuWriter: cell data " << name << " written." << Core::IO::endl;
}

FOUR_C_NAMESPACE_CLOSE