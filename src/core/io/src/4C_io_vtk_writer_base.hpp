// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VTK_WRITER_BASE_HPP
#define FOUR_C_IO_VTK_WRITER_BASE_HPP


#include "4C_config.hpp"

#include "4C_io_visualization_data.hpp"
#include "4C_utils_exceptions.hpp"

#include <stdint.h>
#include <zlib.h>

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace LibB64
{
  /**
   * Do a base64 encoding of the given data.
   *
   * The function allocates memory as necessary and returns a pointer to
   * it. The calling function must release this memory again.
   */
  char* encode_block(const char* data, const int data_size);

  /**
   * This enum class encodes the compression levels available in zlib. Usually, you want to use
   * best_speed for compression since it offers nearly as good compression as best_compression while
   * being a lot faster.
   */
  enum class CompressionLevel
  {
    best_speed = Z_BEST_SPEED,
    best_compression = Z_BEST_COMPRESSION,
    no_compression = Z_NO_COMPRESSION
  };

  template <typename T>
  void write_compressed_block(const std::vector<T>& data, std::ostream& out,
      const CompressionLevel compression_level = CompressionLevel::best_speed)
  {
    uLongf compressed_data_length = compressBound(data.size() * sizeof(T));
    char* compressed_data = new char[compressed_data_length];
    int err = compress2((Bytef*)compressed_data, &compressed_data_length, (const Bytef*)data.data(),
        data.size() * sizeof(T),
        static_cast<std::underlying_type_t<CompressionLevel>>(compression_level));
    if (err != Z_OK) FOUR_C_THROW("zlib compression failed");

    // now encode the compression header
    const uint32_t compression_header[4] = {1, /* number of blocks */
        (uint32_t)(data.size() * sizeof(T)),   /* size of block */
        (uint32_t)(data.size() * sizeof(T)),   /* size of last block */
        (uint32_t)compressed_data_length};     /* list of compressed sizes of blocks */

    char* encoded_header =
        encode_block((char*)compression_header, 4 * sizeof(compression_header[0]));
    out << encoded_header;
    delete[] encoded_header;

    // next do the compressed
    // data encoding in base64
    char* encoded_data = encode_block(compressed_data, compressed_data_length);
    delete[] compressed_data;

    out << encoded_data;
    out << std::endl;
    delete[] encoded_data;
  }

  /**
   \brief Helper function to determine output file string from time step number
   */
  std::string int2string(const unsigned int i, const unsigned int digits);

  /**
   \brief Helper function to determine number of digits for a given integer value
   */
  inline unsigned int ndigits(unsigned int number)
  {
    // start numbering from 0, so need count digits based on number one less
    if (number > 1) number -= 1;
    unsigned int digits = 0;
    while (number > 0)
    {
      digits++;
      number /= 10;
    }
    return digits;
  }

}  // namespace LibB64


/*
 \brief Base class for VTK output generation

*/
class VtkWriterBase
{
 public:
  //! constructor
  VtkWriterBase(unsigned int myrank, unsigned int num_processors,
      unsigned int max_number_timesteps_to_be_written,
      const std::string& path_existing_working_directory,
      const std::string& name_new_vtk_subdirectory, const std::string& geometry_name,
      const std::string& restart_name, double restart_time, bool write_binary_output,
      LibB64::CompressionLevel compression_level);

  //! destructor
  virtual ~VtkWriterBase() = default;

  /** \brief set class variable storing working directory and create it if not existing
   *
   */
  void set_and_create_vtk_working_directory(const std::string& path_existing_working_directory,
      const std::string& name_vtk_subdirectory_to_be_created);

  /** \brief reset current simulation time and time step number
   *
   */
  void reset_time_and_time_step(double time, unsigned int timestepnumber);

  //! write a VTK collection file that summarizes paths to all written files (e.g. for all time
  //! steps) note: this only includes files written during the 'lifetime' of this writer object
  void write_vtk_collection_file_for_all_written_master_files(
      const std::string& collectionfilename) const;


 protected:
  //! generate the part of the filename that expresses the processor ID
  const std::string& get_part_of_file_name_indicating_processor_id(unsigned int processor_id) const;


  //! Return the opening xml tag for this writer type
  virtual const std::string& writer_opening_tag() const = 0;

  //! Return the parallel opening xml tag for this writer type
  virtual const std::string& writer_p_opening_tag() const = 0;

  //! Return a vector of parallel piece tags for each file
  virtual const std::vector<std::string>& writer_p_piece_tags() const = 0;

  //! Return the parallel file suffix including the dot for this file type
  virtual const std::string& writer_p_suffix() const = 0;

  //! Return the string of this writer type
  virtual const std::string& writer_string() const = 0;

  //! Return the file suffix including the dot for this file type
  virtual const std::string& writer_suffix() const = 0;


  //! throw error if file stream not ready to write into
  void throw_error_if_invalid_file_stream(const std::ostream& ostream) const;

  //! append current master file name and time to collection file content
  void append_master_file_and_time_to_collection_file_mid_section_content(
      const std::string& master_file_name, const std::string& master_file_directory_name,
      double time);

  //! determine the name of the subdirectory by extracting the part after the last '/'
  //! in the full path of the vtk working directory
  std::string determine_vtk_subdirectory_name_from_full_vtk_working_path() const;

 private:
  //! write a VTK collection file that summarizes paths to all the given files
  void write_vtk_collection_file_for_given_list_of_master_files(
      const std::string& collectionfilename,
      const std::vector<std::pair<double, std::string>>& masterfiles_time_and_name) const;

  /** write a VTK collection file that summarizes paths to all the given files
   *
   */
  void create_restarted_initial_collection_file_mid_section(
      const std::string& geometryname, const std::string& restartfilename, double restart_time);

  //! construct full path and name for VTK collection file stream from given file name
  std::string get_vtk_collection_file_full_path_and_name(
      const std::string& collectionfilename) const;

  //! write the header into a given VTK collection file stream
  void write_header_into_given_vtk_collection_file_stream(
      std::ofstream& collectionfilestream) const;

  //! write the footer into a given VTK collection file stream
  void write_footer_into_given_vtk_collection_file_stream(
      std::ofstream& collectionfilestream) const;

  //! write given master file and time value into given collection file stream
  inline void write_master_file_and_time_value_into_given_vtk_collection_file_stream(
      std::ostream& collectionfilestream, const std::string& master_file_name,
      const std::string& master_file_directory_name, double time) const
  {
    write_master_file_and_time_value_into_given_vtk_collection_file_stream(
        collectionfilestream, master_file_directory_name + "/" + master_file_name, time);
  }

  //! write given master file and time value into given collection file stream
  void write_master_file_and_time_value_into_given_vtk_collection_file_stream(
      std::ostream& collectionfilestream, const std::string& master_file_name_full_path,
      double time) const;


  std::string get_xml_option_value(const std::string& line, const std::string& name)
  {
    std::size_t start = line.find(name + "=\"");
    if (start == std::string::npos)
    {
      FOUR_C_THROW("Could not find parameter {} in line {}", name, line);
    }
    start += name.length() + 2;
    std::size_t end = line.find('"', start + 1);
    if (end == std::string::npos)
    {
      FOUR_C_THROW("Syntax error in line {}", line);
    }
    return line.substr(start, end - start);
  }

 protected:
  //! subsequent phases in the process of writing data to VTK files:
  //  header -> point data -> cell data -> footer
  enum Phase
  {
    INIT,
    POINTS,
    CELLS,
    FINAL,
    VAGUE
  };

  // current phase
  Phase currentPhase_;

  //! stream for midsection of the vtk collection file ('.pvd')
  //! containing [time value and filename] of all yet written master files
  std::stringstream collection_file_midsection_cumulated_content_;

  //! full path of the existing working directory where the vtk subdirectory is created.
  //! relative or absolute depending on output specification
  std::string path_existing_working_directory_;

  //! full path of the working directory for all vtk files to be written into
  std::string working_directory_full_path_;

  //! Part of the current filename w/ timestep w/out processor id
  //!  processor-specific files and master file share this part of the filename
  std::string filename_base_;


  //! number of digits (i.e. field width) which is reserved for time step number in file names
  const unsigned int num_timestep_digits_;

  //! number of digits (i.e. field width) which is reserved for number of processors in file names
  const unsigned int num_processor_digits_;

  //! name of the geometry
  const std::string geometry_name_;

  //! Time value for the current time step
  double time_;

  //! output step
  unsigned int timestep_;

  //! flag indicating if simulation is restarted
  const bool is_restart_;

  //! Current cycle step (e.g. in nonlinear iteration, not used yet)
  int cycle_;

  //! toggle between ascii and binary output
  const bool write_binary_output_;

  //! Level of compression used when writing output.
  const LibB64::CompressionLevel compression_level_;

  //! global processor id of this processor
  const unsigned int myrank_;

  //! number of involved processors
  const unsigned int numproc_;
};


/**
 * \brief Function to get the vtk type string for a c++ scalar type. This function has to be
 * specialized for all used scalar types.
 * @return Vtk type string.
 */
template <typename T>
std::string scalar_type_to_vtk_type() = delete;

template <>
inline std::string scalar_type_to_vtk_type<int>()
{
  return "Int32";
}
template <>
inline std::string scalar_type_to_vtk_type<double>()
{
  return "Float64";
}

FOUR_C_NAMESPACE_CLOSE

#endif
