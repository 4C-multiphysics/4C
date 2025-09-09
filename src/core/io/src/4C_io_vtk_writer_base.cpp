// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_vtk_writer_base.hpp"

#include "4C_io_legacy_table.hpp"
#include "4C_io_pstream.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>

FOUR_C_NAMESPACE_OPEN

namespace LibB64
{
  // functions taken from the libb64 project, http://sourceforge.net/projects/libb64
  //
  // libb64 is in the public domain
  namespace Base64
  {
    typedef enum
    {
      step_A,
      step_B,
      step_C
    } base64_encodestep;

    typedef struct
    {
      base64_encodestep step;
      char result;
    } base64_encodestate;

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    void base64_init_encodestate(base64_encodestate* state_in)
    {
      state_in->step = step_A;
      state_in->result = 0;
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    inline char base64_encode_value(char value_in)
    {
      static const char* encoding =
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      if (value_in > 63) return '=';
      return encoding[(int)value_in];
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    int base64_encode_block(
        const char* plaintext_in, int length_in, char* code_out, base64_encodestate* state_in)
    {
      const char* plainchar = plaintext_in;
      const char* const plaintextend = plaintext_in + length_in;
      char* codechar = code_out;
      char result;
      char fragment;

      result = state_in->result;

      switch (state_in->step)
      {
        while (1)
        {
          case step_A:
            if (plainchar == plaintextend)
            {
              state_in->result = result;
              state_in->step = step_A;
              return codechar - code_out;
            }
            fragment = *plainchar++;
            result = (fragment & 0x0fc) >> 2;
            *codechar++ = base64_encode_value(result);
            result = (fragment & 0x003) << 4;
            [[fallthrough]];
          case step_B:
            if (plainchar == plaintextend)
            {
              state_in->result = result;
              state_in->step = step_B;
              return codechar - code_out;
            }
            fragment = *plainchar++;
            result |= (fragment & 0x0f0) >> 4;
            *codechar++ = base64_encode_value(result);
            result = (fragment & 0x00f) << 2;
            [[fallthrough]];
          case step_C:
            if (plainchar == plaintextend)
            {
              state_in->result = result;
              state_in->step = step_C;
              return codechar - code_out;
            }
            fragment = *plainchar++;
            result |= (fragment & 0x0c0) >> 6;
            *codechar++ = base64_encode_value(result);
            result = (fragment & 0x03f) >> 0;
            *codechar++ = base64_encode_value(result);
        }
      }
      /* control should not reach here */
      return codechar - code_out;
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    int base64_encode_blockend(char* code_out, base64_encodestate* state_in)
    {
      char* codechar = code_out;

      switch (state_in->step)
      {
        case step_B:
          *codechar++ = base64_encode_value(state_in->result);
          *codechar++ = '=';
          *codechar++ = '=';
          break;
        case step_C:
          *codechar++ = base64_encode_value(state_in->result);
          *codechar++ = '=';
          break;
        case step_A:
          break;
      }
      *codechar++ = '\0';

      return codechar - code_out;
    }
  }  // namespace Base64

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  char* encode_block(const char* data, const int data_size)
  {
    Base64::base64_encodestate state;
    Base64::base64_init_encodestate(&state);

    char* encoded_data = new char[2 * data_size + 1];

    const int encoded_length_data =
        Base64::base64_encode_block(data, data_size, encoded_data, &state);
    Base64::base64_encode_blockend(encoded_data + encoded_length_data, &state);

    return encoded_data;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  std::string int2string(const unsigned int i, const unsigned int digits)
  {
    FOUR_C_ASSERT(i < std::pow(10, digits), "Invalid digits information");
    if (digits == 0 || digits > 9) return "invalid_digit";

    std::string digitstring(digits, '0');
    unsigned int divisor = 1;
    for (unsigned int d = 0; d < digits; ++d, divisor *= 10)
      digitstring[digits - 1 - d] = '0' + (i % (divisor * 10)) / divisor;
    return digitstring;
  }

}  // namespace LibB64



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
VtkWriterBase::VtkWriterBase(unsigned int myrank, unsigned int num_processors,
    unsigned int max_number_timesteps_to_be_written,
    const std::string& path_existing_working_directory,
    const std::string& name_new_vtk_subdirectory, const std::string& geometry_name,
    const std::string& restart_name, double restart_time, bool write_binary_output,
    LibB64::CompressionLevel compression_level)
    : currentPhase_(VAGUE),
      num_timestep_digits_(LibB64::ndigits(max_number_timesteps_to_be_written)),
      num_processor_digits_(LibB64::ndigits(num_processors)),
      geometry_name_(geometry_name),
      time_(restart_time),
      timestep_(std::numeric_limits<unsigned int>::min()),
      is_restart_(restart_time > 0.0),
      cycle_(std::numeric_limits<int>::max()),
      write_binary_output_(write_binary_output),
      compression_level_(compression_level),
      myrank_(myrank),
      numproc_(num_processors)
{
  set_and_create_vtk_working_directory(path_existing_working_directory, name_new_vtk_subdirectory);

  create_restarted_initial_collection_file_mid_section(geometry_name, restart_name, restart_time);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::set_and_create_vtk_working_directory(
    const std::string& path_existing_working_directory,
    const std::string& name_vtk_subdirectory_to_be_created)
{
  // Note: path_existing_working_directory is allowed to be an empty string,
  //       if the VTK working directory shall be created in current working
  //       directory of executable
  if (name_vtk_subdirectory_to_be_created.empty())
    FOUR_C_THROW("VtkWriterBase: name for VTK working directory must not be empty!");

  working_directory_full_path_ =
      std::filesystem::path(path_existing_working_directory) / name_vtk_subdirectory_to_be_created;

  std::filesystem::create_directories(working_directory_full_path_);

  if (!std::filesystem::is_directory(working_directory_full_path_))
    FOUR_C_THROW("VtkWriterBase failed to create working (sub)directory!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::reset_time_and_time_step(double time, unsigned int timestepnumber)
{
  time_ = time;
  timestep_ = timestepnumber;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::initialize_vtk_file_streams_for_new_geometry_and_or_time_step()
{
  {
    std::ostringstream tmpstream;
    tmpstream << geometry_name_ << "-" << std::setfill('0') << std::setw(num_timestep_digits_)
              << timestep_;

    filename_base_ = tmpstream.str();
  }


  initialize_vtk_file_stream_this_processor();


  if (myrank_ == 0)
  {
    initialize_vtk_master_file_stream();


    // append this new master file to the stream of all written files and times
    // for later use as vtk collection file ('.pvd')
    append_master_file_and_time_to_collection_file_mid_section_content(
        filename_base_ + this->writer_p_suffix(),
        determine_vtk_subdirectory_name_from_full_vtk_working_path(), time_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::initialize_vtk_file_stream_this_processor()
{
  std::ostringstream tmpstream;

  tmpstream << working_directory_full_path_ << "/" << filename_base_
            << get_part_of_file_name_indicating_processor_id(myrank_) << this->writer_suffix();

  currentout_.close();
  currentout_.open(tmpstream.str().c_str());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::initialize_vtk_master_file_stream()
{
  currentmasterout_.close();
  currentmasterout_.open(
      (working_directory_full_path_ + "/" + filename_base_ + this->writer_p_suffix()).c_str());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::append_master_file_and_time_to_collection_file_mid_section_content(
    const std::string& master_file_name, const std::string& master_file_directory_name, double time)
{
  collection_file_midsection_cumulated_content_
      << "    <DataSet timestep=\"" << std::scientific
      << std::setprecision(std::numeric_limits<double>::digits10 - 1) << time
      << "\" group=\"\" part=\"0\" file=\"" << master_file_directory_name << "/" << master_file_name
      << "\"/>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_headers()
{
  // Todo: might need BigEndian on some systems
  const std::string byteorder = "LittleEndian";

  // Todo: specify xml version, vtk DataFile Version, ... if needed


  // start master file on processor 0
  if (myrank_ == 0) write_vtk_header_master_file(byteorder);

  // start file on each individual processor
  write_vtk_header_this_processor(byteorder);

  currentPhase_ = INIT;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_field_data_and_or_time_and_or_cycle(
    const std::map<std::string, Core::IO::visualization_vector_type_variant>& field_data_map)
{
  throw_error_if_invalid_file_stream(currentout_);

  // Initialize field data section.
  currentout_ << "    <FieldData>\n";

  // If previously set add time and cycle to field data.
  if (time_ != std::numeric_limits<double>::min() || cycle_ != std::numeric_limits<int>::max())
  {
    if (time_ != std::numeric_limits<double>::min())
    {
      std::vector<double> temp_vector;
      temp_vector.resize(1);
      temp_vector[0] = time_;
      write_field_data_array("TIME", temp_vector);
    }

    if (cycle_ != std::numeric_limits<int>::max())
    {
      std::vector<int> temp_vector;
      temp_vector.resize(1);
      temp_vector[0] = cycle_;
      write_field_data_array("CYCLE", temp_vector);
    }
  }

  // Write every field data array.
  for (const auto& field_data_iterator : field_data_map)
    write_field_data_array(
        field_data_iterator.first, std::get<std::vector<double>>(field_data_iterator.second));

  // Finalize field data section.
  currentout_ << "    </FieldData>\n\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_time_and_or_cycle()
{
  std::map<std::string, Core::IO::visualization_vector_type_variant> empty_map;
  empty_map.clear();
  write_vtk_field_data_and_or_time_and_or_cycle(empty_map);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_header_master_file(const std::string& byteorder)
{
  throw_error_if_invalid_file_stream(currentmasterout_);

  currentmasterout_ << "<?xml version=\"1.0\" ?> \n";
  currentmasterout_ << "<!-- \n";
  currentmasterout_ << "# vtk DataFile Version 3.0\n";
  currentmasterout_ << "-->\n";
  currentmasterout_ << "<VTKFile type=\"P" << this->writer_string() << "\" version=\"0.1\"";
  currentmasterout_ << " byte_order=\"" << byteorder << "\"";
  currentmasterout_ << ">\n";
  currentmasterout_ << "  " << this->writer_p_opening_tag() << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_header_this_processor(const std::string& byteorder)
{
  throw_error_if_invalid_file_stream(currentout_);

  currentout_ << "<?xml version=\"1.0\" ?> \n";
  currentout_ << "<!-- \n";
  currentout_ << "# vtk DataFile Version 3.0\n";
  currentout_ << "-->\n";
  currentout_ << "<VTKFile type=\"" << this->writer_string() << "\" version=\"0.1\"";
  currentout_ << " compressor=\"vtkZLibDataCompressor\"";
  currentout_ << " byte_order=\"" << byteorder << "\"";
  currentout_ << ">\n";
  currentout_ << "  " << this->writer_opening_tag() << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <typename T>
void VtkWriterBase::write_field_data_array(
    const std::string& name, const std::vector<T>& field_data)
{
  const unsigned int n_data = field_data.size();

  // If the array is empty it is skipped.
  if (n_data == 0) return;

  // Set the header for the current field data array.
  currentout_ << "      <DataArray type=\"";
  currentout_ << scalar_type_to_vtk_type<T>();
  currentout_ << "\" Name=\"";
  currentout_ << name;
  currentout_ << "\" NumberOfTuples=\"1\"";
  if (n_data > 1) currentout_ << " NumberOfComponents=\"" << n_data << "\"";
  currentout_ << " format=\"ascii\">\n";

  // Add the field data.
  currentout_ << std::setprecision(15) << std::scientific;
  for (unsigned int i = 0; i < n_data; i++) currentout_ << field_data[i] << " ";
  currentout_ << std::resetiosflags(std::ios::scientific);

  // Finish the current field data array.
  currentout_ << "\n      </DataArray>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_data_array(const Core::IO::visualization_vector_type_variant& data,
    const int num_components, const std::string& name)
{
  std::string vtk_type_name = "";
  if (std::holds_alternative<std::vector<double>>(data))
  {
    write_data_array_this_processor(std::get<std::vector<double>>(data), num_components, name);
    vtk_type_name = scalar_type_to_vtk_type<double>();
  }
  else if (std::holds_alternative<std::vector<int>>(data))
  {
    write_data_array_this_processor(std::get<std::vector<int>>(data), num_components, name);
    vtk_type_name = scalar_type_to_vtk_type<int>();
  }
  else
    FOUR_C_THROW("Got unexpected vector type");

  if (myrank_ == 0) write_data_array_master_file(num_components, name, vtk_type_name);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::string& VtkWriterBase::get_part_of_file_name_indicating_processor_id(
    unsigned int processor_id) const
{
  static std::string filename_part("");

  std::stringstream filename_part_stream;

  filename_part_stream << "-" << std::setfill('0') << std::setw(num_processor_digits_)
                       << processor_id;

  filename_part = filename_part_stream.str();

  return filename_part;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_data_array_master_file(
    const int num_components, const std::string& name, const std::string& data_type_name)
{
  std::ofstream& masterfilestream = currentmasterout_;
  throw_error_if_invalid_file_stream(masterfilestream);


  masterfilestream << "      <PDataArray type=\"" << data_type_name.c_str() << "\" Name=\"" << name
                   << "\"";

  if (num_components > 1) masterfilestream << " NumberOfComponents=\"" << num_components << "\"";

  masterfilestream << " format=\"ascii\"/>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <typename T>
void VtkWriterBase::write_data_array_this_processor(
    const std::vector<T>& data, const int num_components, const std::string& name)
{
  std::ofstream& filestream = currentout_;
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
    for (typename std::vector<T>::const_iterator it = data.begin(); it != data.end(); ++it)
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
void VtkWriterBase::write_vtk_footers()
{
  throw_error_if_invalid_file_stream(currentout_);
  throw_error_if_invalid_file_stream(currentmasterout_);

  // end the scalar fields
  switch (currentPhase_)
  {
    case POINTS:
    {
      currentout_ << "      </PointData>\n\n";

      if (myrank_ == 0) currentmasterout_ << "    </PPointData>\n";

      currentPhase_ = FINAL;

      break;
    }

    case CELLS:
    {
      currentout_ << "      </CellData>\n\n";

      if (myrank_ == 0) currentmasterout_ << "    </PCellData>\n";

      currentPhase_ = FINAL;

      break;
    }

    default:
    {
      FOUR_C_THROW("No data was written or writer was already in final phase.");

      break;
    }
  }


  if (myrank_ == 0) write_vtk_footer_master_file();

  write_vtk_footer_this_processor();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_footer_master_file()
{
  throw_error_if_invalid_file_stream(currentmasterout_);


  // generate information about 'pieces' (piece = part that is written by individual processor)
  using pptags_type = std::vector<std::string>;
  const pptags_type& ppiecetags = this->writer_p_piece_tags();

  if (numproc_ != ppiecetags.size()) FOUR_C_THROW("Incorrect number of Pieces.");

  for (pptags_type::const_iterator it = ppiecetags.begin(); it != ppiecetags.end(); ++it)
    currentmasterout_ << "    " << *it << "\n";

  currentmasterout_ << "  </P" << this->writer_string() << ">\n";
  currentmasterout_ << "</VTKFile>\n";

  currentmasterout_ << std::flush;

  if (myrank_ == 0)
    Core::IO::cout(Core::IO::verbose)
        << "\nVtk Files '" << filename_base_ << "' written. Time: " << std::scientific
        << std::setprecision(std::numeric_limits<double>::digits10 - 1) << time_ << Core::IO::endl;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_footer_this_processor()
{
  throw_error_if_invalid_file_stream(currentout_);


  currentout_ << "    </Piece>\n";

  currentout_ << "  </" << this->writer_string() << ">\n";
  currentout_ << "</VTKFile>\n";

  currentout_ << std::flush;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_collection_file_for_all_written_master_files(
    const std::string& collectionfilename) const
{
  /* The file mentioned here is the collection file ('.pvd') which contains
   * references (full path) to a set of written master files. */

  /* Note:
   * This collection file is not necessarily required because Paraview allows
   * us to open/load a series of masterfiles (e.g. one file per timestep) at
   * once if they are named consistently
   * (e.g. 'fancysimulation-structure-[num_timestep].pvtu') */

  /* However, it turns out to be more convenient to open/load this collection
   * file. Some advantages based on first experiences:
   * 1) Paraview handles the time information correctly (as displayed in the
   *    toolbar 'Current time controls')
   * 2) Having restarted (once or multiple times), we can create a collection
   *    file for each restart which only contains the output files of relevant
   *    time steps and discards the outpur files of redundantly computed time
   *    steps. */

  if (myrank_ == 0)
  {
    // initialize the output filestream for the new collection file
    std::ofstream collectionfilestream(
        get_vtk_collection_file_full_path_and_name(collectionfilename).c_str());

    write_header_into_given_vtk_collection_file_stream(collectionfilestream);

    collectionfilestream << collection_file_midsection_cumulated_content_.str();

    write_footer_into_given_vtk_collection_file_stream(collectionfilestream);


    collectionfilestream.flush();
    collectionfilestream.close();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_vtk_collection_file_for_given_list_of_master_files(
    const std::string& collectionfilename,
    const std::vector<std::pair<double, std::string>>& masterfiles_time_and_name) const
{
  /* The file mentioned here is the collection file ('.pvd') which contains
   * references (full path) to a set of written master files. */

  //! Todo currently unused, re-activate and use only after thorough testing

  if (myrank_ == 0)
  {
    // initialize the output filestream for the new collection file
    std::ofstream collectionfilestream(
        get_vtk_collection_file_full_path_and_name(collectionfilename).c_str());


    write_header_into_given_vtk_collection_file_stream(collectionfilestream);

    // determine the name of the subdirectory where all files have been written into

    /* This is necessary because we only want to collect RELATIVE paths of the
     * individual master files. Otherwise, the collection file would not work
     * as expected after copying/moving the simulation output data */
    const std::string vtk_subdirectory_name =
        determine_vtk_subdirectory_name_from_full_vtk_working_path();


    for (unsigned int ifile = 0; ifile < masterfiles_time_and_name.size(); ++ifile)
    {
      write_master_file_and_time_value_into_given_vtk_collection_file_stream(collectionfilestream,
          masterfiles_time_and_name[ifile].second, vtk_subdirectory_name,
          masterfiles_time_and_name[ifile].first);
    }


    write_footer_into_given_vtk_collection_file_stream(collectionfilestream);

    collectionfilestream.flush();
    collectionfilestream.close();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void VtkWriterBase::create_restarted_initial_collection_file_mid_section(
    const std::string& geometryname, const std::string& restartfilename, const double restart_time)
{
  if (myrank_ != 0 or not is_restart_) return;

  // get name and path of restarted collection file
  std::string restartcollectionfilename(restartfilename + "-" + geometryname + ".pvd");

  // open collection file
  std::ifstream restart_collection_file;
  restart_collection_file.open(restartcollectionfilename.c_str(), std::ios::out);

  // check if file was found
  if (not restart_collection_file) FOUR_C_THROW(" restart collection file could not be found");

  // loop over lines of restarted collection file
  std::string line;
  while (std::getline(restart_collection_file, line))
  {
    // found line with timestep
    if (line.find("timestep=", 0) != std::string::npos)
    {
      double readtime = std::atof(get_xml_option_value(line, "timestep").c_str());

      if (readtime <= (restart_time + 1e-12))
      {
        std::filesystem::path p_restart(restartfilename);
        std::filesystem::path p_filename(get_xml_option_value(line, "file"));

        // Choose base directory: on absolute restart select its parent directory
        // otherwise keep relative with ".")
        const std::filesystem::path base_dir =
            p_restart.is_absolute() ? p_restart.parent_path()
                                    : (std::filesystem::path(".") / p_restart.parent_path());

        // Resolve final path: select absolute file paths, otherwise anchor at base directory
        const std::filesystem::path resolved =
            p_filename.is_absolute() ? p_filename : (base_dir / p_filename);

        write_master_file_and_time_value_into_given_vtk_collection_file_stream(
            collection_file_midsection_cumulated_content_,
            resolved.lexically_normal().generic_string(), readtime);
      }
      else
        break;
    }
  }

  // close file
  restart_collection_file.close();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::string VtkWriterBase::get_vtk_collection_file_full_path_and_name(
    const std::string& collectionfilename) const
{
  // initialize the output filestream for the new collection file
  return (working_directory_full_path_ + "/../" + collectionfilename + ".pvd");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_header_into_given_vtk_collection_file_stream(
    std::ofstream& collectionfilestream) const
{
  throw_error_if_invalid_file_stream(collectionfilestream);

  // Todo specify byte order, xml version, vtk DataFile Version, ... in a central place

  collectionfilestream << "<?xml version=\"1.0\"?>\n";

  collectionfilestream << "<!--\n";
  collectionfilestream << "# vtk DataFile Version 3.0\n";
  collectionfilestream << "-->\n";

  collectionfilestream
      << "<VTKFile type=\"Collection\" version=\"0.1\" ByteOrder=\"LittleEndian\">\n";
  collectionfilestream << "  <Collection>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_footer_into_given_vtk_collection_file_stream(
    std::ofstream& collectionfilestream) const
{
  throw_error_if_invalid_file_stream(collectionfilestream);

  collectionfilestream << "  </Collection>\n";
  collectionfilestream << "</VTKFile>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::string VtkWriterBase::determine_vtk_subdirectory_name_from_full_vtk_working_path() const
{
  // this extracts the substring starting from the last '/' in the full
  // path of the vtk working directory

  size_t extractor_start_position = working_directory_full_path_.find_last_of("/");

  // if we can't find a '/', the given path was already a relative one and
  // we set the start position for substring extraction to the beginning (0ul)
  if (extractor_start_position == working_directory_full_path_.npos) extractor_start_position = 0ul;
  // otherwise, we start from the subsequent character
  else
    extractor_start_position++;

  return working_directory_full_path_.substr(extractor_start_position);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::write_master_file_and_time_value_into_given_vtk_collection_file_stream(
    std::ostream& collectionfilestream, const std::string& master_file_name, double time) const
{
  throw_error_if_invalid_file_stream(collectionfilestream);

  collectionfilestream << "    <DataSet timestep=\"" << std::scientific
                       << std::setprecision(std::numeric_limits<double>::digits10 - 1) << time
                       << R"(" group="" part="0" file=")" << master_file_name << "\"/>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::throw_error_if_invalid_file_stream(const std::ostream& ostream) const
{
  if (not ostream) FOUR_C_THROW("VtkWriterBase: trying to write to invalid output stream!");
}

FOUR_C_NAMESPACE_CLOSE
