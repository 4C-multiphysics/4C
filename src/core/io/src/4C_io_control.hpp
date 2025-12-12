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

#include <mpi.h>

#include <fstream>
#include <memory>
#include <optional>
#include <string>


FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputControl;

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
    struct Impl;
    //! Pointer-to-implementation to hide implementation details.
    std::unique_ptr<Impl> pimpl_;
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

  template <typename T>
  concept ControlFileEntrySupportedType =
      std::is_same_v<T, std::string> || std::is_same_v<T, int> || std::is_same_v<T, double>;

  /**
   * An entry in the control file. This is either a value or a group that holds values or groups.
   * This is a light-weight accessor object that refers to an entry in the control file held by
   * InputControl.
   *
   * Any invalid access fails gracefully by either returning an invalid ControlFileEntry or, when
   * accessing the value, an empty optional. Take the following example:
   *
   * @code
   *   auto value = entry["some_group"]["some_key"].as<int>();
   * @endcode
   *
   * Traversing through the call chain, may fail for any of the following reasons:
   *  - `entry` is already invalid
   *  - `entry` does not contain a group with key `some_group`
   *  - what is stored under key `some_group` is not a group
   *  - `some_group` does not contain a key `some_key`
   *  - `some_key` is not an integer
   *
   * These failures are all handled gracefully and `value` will be an empty optional, should any of
   * the above occur. Usually, we are not interested where exactly the failure occurred, only that
   * it did.
   */
  class ControlFileEntry
  {
   public:
    /**
     * Creates an invalid entry.
     */
    ControlFileEntry() = default;

    /**
     * Construct an entry that wraps the given control file and index. There is no reason
     * to call this constructor directly. ControlFileEntry objects are created by InputControl and
     * other ControlFileEntry objects.
     */
    ControlFileEntry(const InputControl* control, size_t index);

    /**
     * Return true if the entry is of the given type. False if it is of a different type, a group,
     * or invalid.
     */
    template <ControlFileEntrySupportedType T>
    [[nodiscard]] bool is_a() const;

    /**
     * Return true if the entry is a group. False if it is a value or invalid.
     */
    [[nodiscard]] bool is_group() const;

    /**
     * Return the value of the entry as the given type, if possible. If the entry is not of the
     * given type, is a group, or is invalid, return an empty optional.
     */
    template <ControlFileEntrySupportedType T>
    [[nodiscard]] std::optional<T> as() const;

    /**
     * If the entry is a group, return the child entry with the given key. Otherwise, return an
     * Invalid ControlFileEntry.
     */
    [[nodiscard]] ControlFileEntry operator[](const std::string& key) const;

    /**
     * Return true if this object is valid and refers to an entry in the control file.
     */
    [[nodiscard]] bool is_valid() const { return control_ != nullptr; }

   private:
    //! Refer to the input control that holds the control file details.
    const InputControl* control_{nullptr};
    //! Opaque index of this entry in the control file.
    size_t index_{};
  };

  /**
   * Manages reading from a control file.
   */
  class InputControl
  {
   public:
    explicit InputControl(const std::string& filename, const bool serial = false);
    InputControl(const std::string& filename, MPI_Comm comm);

    ~InputControl();

    InputControl(const InputControl&) = delete;
    InputControl& operator=(const InputControl&) = delete;
    InputControl(InputControl&&) = delete;
    InputControl& operator=(InputControl&&) = delete;

    /**
     * Return the name of the control file.
     */
    [[nodiscard]] const std::string& file_name() const;

    /**
     * A control file contains a list of entries which may contain the same key. This function
     * returns the nth last entry (i.e., the nth most recent one) with the given key. If no such
     * entry exists, an invalid ControlFileEntry is returned.
     * Defaults to the very last entry (n=1).
     */
    [[nodiscard]] ControlFileEntry nth_last_entry(
        std::optional<std::string> key, size_t n = 1) const;

    /**
     * A control file contains a list of entries. This function returns the last key in the control
     * file. If no last key exists, an empty optional is returned.
     */
    [[nodiscard]] std::optional<std::string> last_key() const;

    /**
     * Return the last entry in the control file.
     */
    // [[nodiscard]] ControlFileEntry last_entry() const;

    /**
     * Find the control file entries for the given discretization and step.
     */
    std::pair<ControlFileEntry, ControlFileEntry> find_group(int step,
        const std::string& discretization_name, const std::string& group_name,
        const std::string& file_marker) const;

   private:
    struct Impl;
    //! Pointer-to-implementation to hide implementation details.
    std::unique_ptr<Impl> pimpl_;

    //! ControlFileEntry is a light-weight accessor into the control file structure.
    friend class ControlFileEntry;
  };


  /// find the last possible restart step in the control file
  int get_last_possible_restart_step(Core::IO::InputControl& inputcontrol);
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
