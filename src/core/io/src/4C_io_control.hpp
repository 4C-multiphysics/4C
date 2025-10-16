// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_CONTROL_HPP
#define FOUR_C_IO_CONTROL_HPP

#include "4C_config.hpp"

#include "4C_fem_general_shape_function_type.hpp"
#include "4C_io_legacy_types.hpp"

#include <mpi.h>

#include <fstream>
#include <memory>
#include <string>


FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  namespace Internal
  {
    class ControlFileWriterImpl;
  }

  /**
   * Wrap functionality related to writing control files.
   */
  class ControlFileWriter
  {
   public:
    /**
     * Initialize the writer object. If @p do_write is false, no write will be performed.
     * This is useful for non-root MPI ranks.
     */
    ControlFileWriter(bool do_write, std::ostream& stream);

    /**
     * Delete the writer object. Flush any pending writes.
     */
    ~ControlFileWriter();

    ControlFileWriter& operator=(const ControlFileWriter&) = delete;
    ControlFileWriter(const ControlFileWriter&) = delete;
    ControlFileWriter(ControlFileWriter&&) = delete;
    ControlFileWriter& operator=(ControlFileWriter&&) = delete;

    /**
     * Write metadata to the header.
     */
    void write_metadata_header();

    /**
     * End all groups and flush pending writes. This is automatically called when the last group is
     * closed or the object is destroyed.
     */
    void end_all_groups_and_flush();

    /**
     * Write a key-value pair to the control file. The value can be a string, integer, or double.
     * The value is added to the current group with the current indentation.
     */
    ControlFileWriter& write(std::string_view key, const std::string_view& value);
    ControlFileWriter& write(std::string_view key, int value);
    ControlFileWriter& write(std::string_view key, double value);

    /**
     * Start a new group in the control file with increased indentation.
     */
    ControlFileWriter& start_group(const std::string& group_name);

    /**
     * End the previously opened group in the control file and decrease indentation.
     */
    ControlFileWriter& end_group();

    /**
     * End the previously opened group if one is open. Otherwise, do nothing.
     */
    ControlFileWriter& try_end_group();

   private:
    //! Pointer-to-implementation to hide implementation details.
    std::unique_ptr<Internal::ControlFileWriterImpl> pimpl_;
  };

  /// control class to manage a control file for output
  class OutputControl
  {
   public:
    /*!
     * @brief construct output control object
     *
     * @param[in] comm                    communicator
     * @param[in] problemtype             problem type
     * @param[in] type_of_spatial_approx  spatial approximation type of the fe discretization
     * @param[in] inputfile               file name of input file
     * @param[in] restartname          file name prefix for restart
     * @param[in] outputname           output file name prefix
     * @param[in] ndim                 number of space dimensions
     * @param[in] restart_step         step from which restart is performed
     * @param[in] filesteps            number of output steps per binary file
     * @param[in] write_binary_output  flag indicating if output is written in binary format
     * @param[in] adaptname            flag indicating if output name is adapted
     */
    OutputControl(MPI_Comm comm, std::string problemtype,
        Core::FE::ShapeFunctionType type_of_spatial_approx, std::string inputfile,
        const std::string& restartname, std::string outputname, int ndim, int restart_step,
        int filesteps, bool write_binary_output, bool adaptname = true);

    /*!
     * @brief output prefix we write to
     *
     * In case of restart this will be different from the read prefix.
     * \note might contain path
     */
    std::string file_name() const { return filename_; }

    /**
     * @brief Return the file name prefix, i.e., the file name that is given to the 4C call
     */
    std::string file_name_only_prefix() const;

    /**
     * @brief Base output directory
     */
    std::string directory_name() const;

    /*!
     * @brief original prefix as given
     *
     * In case of restart this prefix specifies the control file we read.
     * \note might contain path
     */
    std::string restart_name() const { return restartname_; }

    std::string new_output_file_name() const { return filename_; }

    /// open control file
    ControlFileWriter& control_file() { return control_file_; }

    /// number of output steps per binary file
    int file_steps() const { return filesteps_; }

    /// time step the simulation is restarted from
    int restart_step() const { return restart_step_; }

    // modify the number of output steps per binary file
    // (necessary for the structural debugging option "OUTPUTEVERYITER")
    void set_file_steps(int filesteps) { filesteps_ = filesteps; }

    /// input filename
    std::string input_file_name() const { return inputfile_; }

    bool write_binary_output() const { return write_binary_output_; }

    /// return my processor ID
    inline int my_rank() const { return myrank_; };

   private:
    std::string problemtype_;
    std::string inputfile_;  ///< input file name
    const int ndim_;
    const int myrank_;
    std::string filename_;  ///< prefix of outputfiles (might contain path)
    std::string restartname_;
    std::ofstream control_file_stream_;
    ControlFileWriter control_file_;
    int filesteps_;
    const int restart_step_;
    const bool write_binary_output_;
  };


  /// control class to manage a control file for input
  class InputControl
  {
   public:
    InputControl(const std::string& filename, const bool serial = false);
    InputControl(const std::string& filename, MPI_Comm comm);
    ~InputControl();

    MAP* control_file() { return &table_; }

    std::string file_name() const { return filename_; }

    /// find control file entry to given time step
    /*!
      The control file entry with the given group_name those field and step match
      my discretization and step. From that we need a backward search to find
      the entry that links to the binary files that cover our entry.
     */
    void find_group(int step, const std::string& discretization_name, const char* group_name,
        const char* filestring, MAP*& result_info, MAP*& file_info);

   private:
    InputControl(const InputControl&);
    InputControl& operator=(const InputControl&);

    std::string filename_;
    MAP table_;
  };


  /// find the last possible restart step in the control file
  int get_last_possible_restart_step(Core::IO::InputControl& inputcontrol);
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
