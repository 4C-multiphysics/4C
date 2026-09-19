// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_vtk_writer_base.hpp"

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
      path_existing_working_directory_(path_existing_working_directory),
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
  restart_collection_file.open(restartcollectionfilename.c_str(), std::ios::in);

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
        // otherwise form relative path from current working directory to the restart directory
        const std::filesystem::path restart_parent =
            p_restart.parent_path().empty() ? std::filesystem::path(".") : p_restart.parent_path();
        const std::filesystem::path working_dir =
            path_existing_working_directory_.empty()
                ? std::filesystem::path(".")
                : std::filesystem::path(path_existing_working_directory_);
        const std::filesystem::path base_dir =
            p_restart.is_absolute()
                ? p_restart.parent_path()
                : std::filesystem::relative(std::filesystem::absolute(restart_parent),
                      std::filesystem::absolute(working_dir));

        // Resolve final path: select absolute file paths, otherwise anchor at base directory
        const std::filesystem::path resolved =
            p_filename.is_absolute() ? p_filename : (base_dir / p_filename);

        write_master_file_and_time_value_into_given_vtk_collection_file_stream(
            collection_file_midsection_cumulated_content_,
            resolved.lexically_normal().generic_string(), readtime);
      }
      else
      {
        break;
      }
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
