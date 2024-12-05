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
     * @param[in] outputname           output file name prefix
     * @param[in] ndim                 number of space dimensions
     * @param[in] restart_step         step from which restart is performed
     * @param[in] filesteps            number of output steps per binary file
     * @param[in] write_binary_output  flag indicating if output is written in binary format
     */
    OutputControl(MPI_Comm comm, std::string problemtype,
        Core::FE::ShapeFunctionType type_of_spatial_approx, std::string inputfile,
        const std::string& outputname, int ndim, int restart_step, int filesteps,
        bool write_binary_output);

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
     * @brief copy constructor
     *
     * @param[in] ocontrol   Copy this object of same type
     * @param[in] new_prefix Add the prefix to restart and file names (optional)
     */
    OutputControl(const OutputControl& ocontrol, const char* new_prefix = nullptr);

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
    std::fstream& control_file() { return controlfile_; }

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

    /// overwrites result files
    void overwrite_result_file(const Core::FE::ShapeFunctionType& spatial_approx);

    /// creates new result files
    void new_result_file(int numb_run, const Core::FE::ShapeFunctionType& spatial_approx);

    /// creates new result files for the mlmc
    void new_result_file(const std::string& name_appendix, int numb_run,
        const Core::FE::ShapeFunctionType& spatial_approx);

    /// creates new result files
    void new_result_file(std::string name, const Core::FE::ShapeFunctionType& spatial_approx);

    /// return my processor ID
    inline int my_rank() const { return myrank_; };

   private:
    void write_header(
        const std::string& control_file_name, const Core::FE::ShapeFunctionType& spatial_approx);

    void insert_restart_back_reference(int restart, const std::string& outputname);

   private:
    std::string problemtype_;
    std::string inputfile_;  ///< input file name
    const int ndim_;
    std::string filename_;  ///< prefix of outputfiles (might contain path)
    std::string restartname_;
    std::fstream controlfile_;
    int filesteps_;
    const int restart_step_;
    const int myrank_;
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


   private:
    InputControl(const InputControl&);
    InputControl& operator=(const InputControl&);

    std::string filename_;
    MAP table_;
  };


  /// find position of restart number in filename (if existing):
  /// for "outname-5" will return position of the "-"
  /// returns std::string::npos if not found
  size_t restart_finder(const std::string& filename);

  /// find the last possible restart step in the control file
  int get_last_possible_restart_step(Core::IO::InputControl& inputcontrol);
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
