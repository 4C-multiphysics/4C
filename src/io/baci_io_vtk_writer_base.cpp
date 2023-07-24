/*----------------------------------------------------------------------*/
/*! \file

\brief VTK writer base class

\level 2

*/
/*----------------------------------------------------------------------*/

#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <filesystem>

#include "baci_io_legacy_table_cpp.h"

#include "baci_utils_exceptions.H"
#include "baci_lib_globalproblem.H"

#include "baci_io_pstream.H"

#include "baci_io_vtk_writer_base.H"


namespace LIBB64
{
  // functions taken from the libb64 project, http://sourceforge.net/projects/libb64
  //
  // libb64 is in the public domain
  namespace base64
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
  }  // namespace base64

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  char* encode_block(const char* data, const int data_size)
  {
    base64::base64_encodestate state;
    base64::base64_init_encodestate(&state);

    char* encoded_data = new char[2 * data_size + 1];

    const int encoded_length_data =
        base64::base64_encode_block(data, data_size, encoded_data, &state);
    base64::base64_encode_blockend(encoded_data + encoded_length_data, &state);

    return encoded_data;
  }

  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  std::string int2string(const unsigned int i, const unsigned int digits)
  {
    dsassert(i < std::pow(10, digits), "Invalid digits information");
    if (digits == 0 || digits > 9) return "invalid_digit";

    std::string digitstring(digits, '0');
    unsigned int divisor = 1;
    for (unsigned int d = 0; d < digits; ++d, divisor *= 10)
      digitstring[digits - 1 - d] = '0' + (i % (divisor * 10)) / divisor;
    return digitstring;
  }

}  // namespace LIBB64



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
VtkWriterBase::VtkWriterBase()
    : currentPhase_(VAGUE),
      num_timestep_digits_(0),
      num_processor_digits_(0),
      time_(std::numeric_limits<double>::min()),
      timestep_(-1),
      restart_(DRT::Problem::Instance()->Restart()),
      cycle_(std::numeric_limits<int>::max()),
      write_binary_output_(true),
      myrank_(0),
      numproc_(1)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::Initialize(unsigned int myrank, unsigned int num_processors,
    unsigned int max_number_timesteps_to_be_written,
    const std::string& path_existing_working_directory,
    const std::string& name_new_vtk_subdirectory, const std::string& geometry_name,
    const std::string& restart_name, double restart_time, bool write_binary_output)
{
  myrank_ = myrank;
  numproc_ = num_processors;

  num_timestep_digits_ = LIBB64::ndigits(max_number_timesteps_to_be_written);
  num_processor_digits_ = LIBB64::ndigits(num_processors);

  write_binary_output_ = write_binary_output;

  geometry_name_ = geometry_name;

  SetAndCreateVtkWorkingDirectory(path_existing_working_directory, name_new_vtk_subdirectory);

  CreateRestartedInitialCollectionFileMidSection(geometry_name, restart_name, restart_time);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::SetAndCreateVtkWorkingDirectory(
    const std::string& path_existing_working_directory,
    const std::string& name_vtk_subdirectory_to_be_created)
{
  // Note: path_existing_working_directory is allowed to be an empty string,
  //       if the VTK working directory shall be created in current working
  //       directory of executable
  if (name_vtk_subdirectory_to_be_created.empty())
    dserror("VtkWriterBase: name for VTK working directory must not be empty!");

  working_directory_full_path_ =
      path_existing_working_directory + name_vtk_subdirectory_to_be_created;

  std::filesystem::create_directories(working_directory_full_path_);

  if (!std::filesystem::is_directory(working_directory_full_path_))
    dserror("VtkWriterBase failed to create working (sub)directory!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::ResetTimeAndTimeStep(double time, unsigned int timestepnumber)
{
  time_ = time;
  timestep_ = timestepnumber;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::ResetGeometryName(const std::string& geometryname)
{
  geometry_name_ = geometryname;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::InitializeVtkFileStreamsForNewGeometryAndOrTimeStep()
{
  {
    std::ostringstream tmpstream;
    tmpstream << geometry_name_ << "-" << std::setfill('0') << std::setw(num_timestep_digits_)
              << timestep_;

    filename_base_ = tmpstream.str();
  }


  InitializeVtkFileStreamThisProcessor();


  if (myrank_ == 0)
  {
    InitializeVtkMasterFileStream();


    // append this new master file to the stream of all written files and times
    // for later use as vtk collection file ('.pvd')
    AppendMasterFileAndTimeToCollectionFileMidSectionContent(filename_base_ + this->WriterPSuffix(),
        DetermineVtkSubdirectoryNameFromFullVtkWorkingPath(), time_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::InitializeVtkFileStreamThisProcessor()
{
  std::ostringstream tmpstream;

  tmpstream << working_directory_full_path_ << "/" << filename_base_
            << GetPartOfFileNameIndicatingProcessorId(myrank_) << this->WriterSuffix();

  currentout_.close();
  currentout_.open(tmpstream.str().c_str());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::InitializeVtkMasterFileStream()
{
  currentmasterout_.close();
  currentmasterout_.open(
      (working_directory_full_path_ + "/" + filename_base_ + this->WriterPSuffix()).c_str());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::AppendMasterFileAndTimeToCollectionFileMidSectionContent(
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
void VtkWriterBase::WriteVtkHeaders()
{
  // Todo: might need BigEndian on some systems
  const std::string byteorder = "LittleEndian";

  // Todo: specify xml version, vtk DataFile Version, ... if needed


  // start master file on processor 0
  if (myrank_ == 0) WriteVtkHeaderMasterFile(byteorder);

  // start file on each individual processor
  WriteVtkHeaderThisProcessor(byteorder);

  currentPhase_ = INIT;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkFieldDataAndOrTimeAndOrCycle(
    const std::map<std::string, std::vector<double>>& field_data_map)
{
  ThrowErrorIfInvalidFileStream(currentout_);

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
      WriteFieldDataArray("TIME", temp_vector);
    }

    if (cycle_ != std::numeric_limits<int>::max())
    {
      std::vector<int> temp_vector;
      temp_vector.resize(1);
      temp_vector[0] = cycle_;
      WriteFieldDataArray("CYCLE", temp_vector);
    }
  }

  // Write every field data array.
  for (const auto& field_data_iterator : field_data_map)
    WriteFieldDataArray(field_data_iterator.first, field_data_iterator.second);

  // Finalize field data section.
  currentout_ << "    </FieldData>\n\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkTimeAndOrCycle()
{
  std::map<std::string, std::vector<double>> empty_map;
  empty_map.clear();
  WriteVtkFieldDataAndOrTimeAndOrCycle(empty_map);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkHeaderMasterFile(const std::string& byteorder)
{
  ThrowErrorIfInvalidFileStream(currentmasterout_);

  currentmasterout_ << "<?xml version=\"1.0\" ?> \n";
  currentmasterout_ << "<!-- \n";
  currentmasterout_ << "# vtk DataFile Version 3.0\n";
  currentmasterout_ << "-->\n";
  currentmasterout_ << "<VTKFile type=\"P" << this->WriterString() << "\" version=\"0.1\"";
  currentmasterout_ << " byte_order=\"" << byteorder << "\"";
  currentmasterout_ << ">\n";
  currentmasterout_ << "  " << this->WriterPOpeningTag() << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkHeaderThisProcessor(const std::string& byteorder)
{
  ThrowErrorIfInvalidFileStream(currentout_);

  currentout_ << "<?xml version=\"1.0\" ?> \n";
  currentout_ << "<!-- \n";
  currentout_ << "# vtk DataFile Version 3.0\n";
  currentout_ << "-->\n";
  currentout_ << "<VTKFile type=\"" << this->WriterString() << "\" version=\"0.1\"";
  currentout_ << " compressor=\"vtkZLibDataCompressor\"";
  currentout_ << " byte_order=\"" << byteorder << "\"";
  currentout_ << ">\n";
  currentout_ << "  " << this->WriterOpeningTag() << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <typename T>
void VtkWriterBase::WriteFieldDataArray(const std::string& name, const std::vector<T>& field_data)
{
  const unsigned int n_data = field_data.size();

  // If the array is empty it is skipped.
  if (n_data == 0) return;

  // Set the header for the current field data array.
  currentout_ << "      <DataArray type=\"";
  currentout_ << ScalarTypeToVtkType<T>();
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
void VtkWriterBase::WriteDataArray(
    const std::vector<double>& data, const int num_components, const std::string& name)
{
  if (myrank_ == 0) WriteDataArrayMasterFile(num_components, name);

  WriteDataArrayThisProcessor(data, num_components, name);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const std::string& VtkWriterBase::GetPartOfFileNameIndicatingProcessorId(
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
void VtkWriterBase::WriteDataArrayMasterFile(const int num_components, const std::string& name)
{
  std::ofstream& masterfilestream = currentmasterout_;
  ThrowErrorIfInvalidFileStream(masterfilestream);


  masterfilestream << "      <PDataArray type=\"Float64\" Name=\"" << name << "\"";

  if (num_components > 1) masterfilestream << " NumberOfComponents=\"" << num_components << "\"";

  masterfilestream << " format=\"ascii\"/>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteDataArrayThisProcessor(
    const std::vector<double>& data, const int num_components, const std::string& name)
{
  std::ofstream& filestream = currentout_;
  ThrowErrorIfInvalidFileStream(filestream);


  filestream << "        <DataArray type=\"Float64\" Name=\"" << name << "\"";

  if (num_components > 1) filestream << " NumberOfComponents=\"" << num_components << "\"";

  if (write_binary_output_)
  {
    filestream << " format=\"binary\">\n";

    LIBB64::writeCompressedBlock(data, filestream);
  }
  else
  {
    filestream << " format=\"ascii\">\n";

    int counter = 1;
    for (std::vector<double>::const_iterator it = data.begin(); it != data.end(); ++it)
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
void VtkWriterBase::WriteVtkFooters()
{
  ThrowErrorIfInvalidFileStream(currentout_);
  ThrowErrorIfInvalidFileStream(currentmasterout_);

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
      dserror("No data was written or writer was already in final phase.");

      break;
    }
  }


  if (myrank_ == 0) WriteVtkFooterMasterFile();

  WriteVtkFooterThisProcessor();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkFooterMasterFile()
{
  ThrowErrorIfInvalidFileStream(currentmasterout_);


  // generate information about 'pieces' (piece = part that is written by individual processor)
  typedef std::vector<std::string> pptags_type;
  const pptags_type& ppiecetags = this->WriterPPieceTags();

  if (numproc_ != ppiecetags.size()) dserror("Incorrect number of Pieces.");

  for (pptags_type::const_iterator it = ppiecetags.begin(); it != ppiecetags.end(); ++it)
    currentmasterout_ << "    " << *it << "\n";

  currentmasterout_ << "  </P" << this->WriterString() << ">\n";
  currentmasterout_ << "</VTKFile>\n";

  currentmasterout_ << std::flush;

  if (myrank_ == 0)
    IO::cout(IO::verbose) << "\nVtk Files '" << filename_base_
                          << "' written. Time: " << std::scientific
                          << std::setprecision(std::numeric_limits<double>::digits10 - 1) << time_
                          << IO::endl;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkFooterThisProcessor()
{
  ThrowErrorIfInvalidFileStream(currentout_);


  currentout_ << "    </Piece>\n";

  currentout_ << "  </" << this->WriterString() << ">\n";
  currentout_ << "</VTKFile>\n";

  currentout_ << std::flush;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkCollectionFileForAllWrittenMasterFiles(
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
        GetVtkCollectionFileFullPathAndName(collectionfilename).c_str());

    WriteHeaderIntoGivenVtkCollectionFileStream(collectionfilestream);

    collectionfilestream << collection_file_midsection_cumulated_content_.str();

    WriteFooterIntoGivenVtkCollectionFileStream(collectionfilestream);


    collectionfilestream.flush();
    collectionfilestream.close();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteVtkCollectionFileForGivenListOfMasterFiles(
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
        GetVtkCollectionFileFullPathAndName(collectionfilename).c_str());


    WriteHeaderIntoGivenVtkCollectionFileStream(collectionfilestream);

    // determine the name of the subdirectory where all files have been written into

    /* This is necessary because we only want to collect RELATIVE paths of the
     * individual master files. Otherwise, the collection file would not work
     * as expected after copying/moving the simulation output data */
    const std::string vtk_subdirectory_name = DetermineVtkSubdirectoryNameFromFullVtkWorkingPath();


    for (unsigned int ifile = 0; ifile < masterfiles_time_and_name.size(); ++ifile)
    {
      WriteMasterFileAndTimeValueIntoGivenVtkCollectionFileStream(collectionfilestream,
          masterfiles_time_and_name[ifile].second, vtk_subdirectory_name,
          masterfiles_time_and_name[ifile].first);
    }


    WriteFooterIntoGivenVtkCollectionFileStream(collectionfilestream);

    collectionfilestream.flush();
    collectionfilestream.close();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void VtkWriterBase::CreateRestartedInitialCollectionFileMidSection(
    const std::string& geometryname, const std::string& restartfilename, double restart_time)
{
  if (myrank_ != 0 or not restart_) return;

  // get name and path of restarted collection file
  std::string restartcollectionfilename(restartfilename + "-" + geometryname + ".pvd");

  // open collection file
  std::ifstream restart_collection_file;
  restart_collection_file.open(restartcollectionfilename.c_str(), std::ios::out);

  // check if file was found
  if (not restart_collection_file) dserror(" restart collection file could not be found");

  // loop over lines of restarted collection file
  std::string line;
  while (std::getline(restart_collection_file, line))
  {
    // found line with timestep
    if (line.find("timestep=", 0) != std::string::npos)
    {
      double readtime = std::atof(GetXmlOptionValue(line, "timestep").c_str());

      if (readtime <= (restart_time + 1e-12))
      {
        std::string filename = GetXmlOptionValue(line, "file");
        std::filesystem::path p_restart(restartfilename);

        if (p_restart.is_absolute())
        {
          // Do not modify the path if the restart is given with an absolute path (we don't want to
          // have absolute paths in the collection file)
          WriteMasterFileAndTimeValueIntoGivenVtkCollectionFileStream(
              collection_file_midsection_cumulated_content_, filename, readtime);
        }
        else
        {
          std::filesystem::path p_new_filename = "." / p_restart.parent_path() / filename;
          WriteMasterFileAndTimeValueIntoGivenVtkCollectionFileStream(
              collection_file_midsection_cumulated_content_, p_new_filename.string(), readtime);
        }
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
std::string VtkWriterBase::GetVtkCollectionFileFullPathAndName(
    const std::string& collectionfilename) const
{
  // initialize the output filestream for the new collection file
  return (working_directory_full_path_ + "/../" + collectionfilename + ".pvd");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::WriteHeaderIntoGivenVtkCollectionFileStream(
    std::ofstream& collectionfilestream) const
{
  ThrowErrorIfInvalidFileStream(collectionfilestream);

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
void VtkWriterBase::WriteFooterIntoGivenVtkCollectionFileStream(
    std::ofstream& collectionfilestream) const
{
  ThrowErrorIfInvalidFileStream(collectionfilestream);

  collectionfilestream << "  </Collection>\n";
  collectionfilestream << "</VTKFile>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::string VtkWriterBase::DetermineVtkSubdirectoryNameFromFullVtkWorkingPath() const
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
void VtkWriterBase::WriteMasterFileAndTimeValueIntoGivenVtkCollectionFileStream(
    std::ostream& collectionfilestream, const std::string& master_file_name, double time) const
{
  ThrowErrorIfInvalidFileStream(collectionfilestream);

  collectionfilestream << "    <DataSet timestep=\"" << std::scientific
                       << std::setprecision(std::numeric_limits<double>::digits10 - 1) << time
                       << R"(" group="" part="0" file=")" << master_file_name << "\"/>\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void VtkWriterBase::ThrowErrorIfInvalidFileStream(const std::ostream& ostream) const
{
  if (not ostream) dserror("VtkWriterBase: trying to write to invalid output stream!");
}
