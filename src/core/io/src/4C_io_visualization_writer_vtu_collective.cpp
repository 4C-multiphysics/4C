// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_visualization_writer_vtu_collective.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_visualization_data.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  //! name stem of the transient probe file used to verify cross-rank visibility of the output dir
  constexpr auto shared_fs_probe_file_name = ".4c_vtu_shared_fs_probe";

  /**
   * @brief Temporarily change the process working directory and restore it on scope exit
   *
   * The collective writer must not pass ever-long output file paths into the MPI layer: the
   * shared file pointer component of OpenMPI <= 4.1.6 ('sharedfp_lockedfile') stores the file
   * name in a fixed-size 256 byte stack buffer (sprintf(filename, "%s%s%d", ...)) during
   * MPI_File_open and aborts the run with a buffer-overflow error if the path is longer. 4C
   * output directories can easily exceed this limit (e.g. long test names). By switching to the
   * file's directory and handing MPI_File_open only the short relative file name the issue is
   * avoided while all MPI-IO calls stay collective.
   *
   * Changing the working directory is process-local and the guard restores the previous
   * directory immediately, so the rest of the code (which uses absolute paths for output files)
   * is not affected. A possible exception in the guarded block restores the directory before it
   * propagates.
   */
  class ScopedWorkingDirectoryChange
  {
   public:
    explicit ScopedWorkingDirectoryChange(const std::string& new_working_directory)
        : previous_working_directory_(std::filesystem::current_path())
    {
      std::filesystem::current_path(new_working_directory);
    }

    ~ScopedWorkingDirectoryChange() { std::filesystem::current_path(previous_working_directory_); }

    //! copying is not allowed since the destructor restores a unique previous directory
    ScopedWorkingDirectoryChange(const ScopedWorkingDirectoryChange&) = delete;
    ScopedWorkingDirectoryChange& operator=(const ScopedWorkingDirectoryChange&) = delete;
    ScopedWorkingDirectoryChange(ScopedWorkingDirectoryChange&&) = delete;
    ScopedWorkingDirectoryChange& operator=(ScopedWorkingDirectoryChange&&) = delete;

   private:
    //! the working directory that has to be restored in the destructor
    std::filesystem::path previous_working_directory_;
  };
}  // namespace

/**
 *
 */
Core::IO::VisualizationWriterVtuCollective::VisualizationWriterVtuCollective(
    const Core::IO::VisualizationParameters& parameters, const MPI_Comm comm,
    std::string visualization_data_name)
    : VisualizationWriterBase(parameters, comm, std::move(visualization_data_name)),
      vtu_writer_(Core::Communication::my_mpi_rank(comm), Core::Communication::num_mpi_ranks(comm),
          std::pow(10, Core::IO::get_total_digits_to_reserve_in_time_step(parameters)),
          parameters.directory_name_, (parameters.file_name_prefix_ + "-vtk-files"),
          visualization_data_name_, parameters.restart_from_name_, parameters.restart_time_,
          parameters.data_format_ == OutputDataFormat::binary, parameters.compression_level_)
{
  // The VTU working directory is created by the VtuWriter base constructor above. Verify once,
  // right at construction, that all ranks access the same physical output directory; otherwise
  // the collective MPI-IO calls would deadlock when output is written later.
  const std::string probe_dir = (std::filesystem::path(parameters_.directory_name_) /
                                 (parameters_.file_name_prefix_ + "-vtk-files"))
                                    .string();
  check_all_ranks_have_access_to_same_file(probe_dir);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::check_all_ranks_have_access_to_same_file(
    const std::string& output_directory) const
{
  const int myrank = Core::Communication::my_mpi_rank(comm_);
  const int numproc = Core::Communication::num_mpi_ranks(comm_);

  // Build a unique probe token on rank 0 and distribute it to all ranks. MPI_Bcast does not
  // depend on the file being shared (unlike MPI-IO), so this cannot deadlock if the ranks do not
  // access the same physical probe file within the output directory.
  std::string token;
  if (myrank == 0)
  {
    token = "4C-VTU-PROBE-" + std::to_string(numproc) + "-" + std::to_string(MPI_Wtime()) + "\n";
  }
  int token_size = static_cast<int>(token.size());
  MPI_Bcast(&token_size, 1, MPI_INT, 0, comm_);
  token.resize(static_cast<std::size_t>(token_size));
  MPI_Bcast(token.data(), token_size, MPI_CHAR, 0, comm_);

  // Rank 0 generates a unique, runtime-only probe file name and distributes it to all ranks. The
  // file is created exclusively below (O_EXCL | O_NOFOLLOW), so a pre-existing entry at this path
  // -- e.g., a symlink planted by another user into a shared output directory -- is rejected
  // instead of being followed, truncated or unlinked.
  std::string probe_file_path;
  if (myrank == 0)
  {
    probe_file_path = (std::filesystem::path(output_directory) /
                       (std::string(shared_fs_probe_file_name) + "-" + std::to_string(::getpid()) +
                           "-" + std::to_string(MPI_Wtime())))
                          .string();
  }
  int probe_path_size = static_cast<int>(probe_file_path.size());
  MPI_Bcast(&probe_path_size, 1, MPI_INT, 0, comm_);
  probe_file_path.resize(static_cast<std::size_t>(probe_path_size));
  MPI_Bcast(probe_file_path.data(), probe_path_size, MPI_CHAR, 0, comm_);

  // Rank 0 creates the probe file exclusively, writes the token and makes it durable with fsync.
  // O_EXCL guarantees that no other entry existed at this path before (so no stale or foreign file
  // or symlink is followed or truncated); O_NOFOLLOW protects against a symlink being placed there.
  int write_failed = 0;
  int probe_created = 0;
  int probe_create_errno = 0;
  if (myrank == 0)
  {
    const int fd = ::open(probe_file_path.c_str(), O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY, 0644);
    if (fd < 0)
    {
      probe_create_errno = errno;
      write_failed = 1;
    }
    else
    {
      probe_created = 1;
      if (::write(fd, token.data(), token.size()) != static_cast<std::streamsize>(token_size) or
          ::fsync(fd) != 0)
        write_failed = 1;
      ::close(fd);
    }
  }
  MPI_Bcast(&probe_create_errno, 1, MPI_INT, 0, comm_);

  // Give the write time to become visible on all nodes before the ranks probe the file.
  MPI_Barrier(comm_);

  // Every rank re-opens the file independently and verifies that the token written by rank 0 is
  // visible. A node-local file either does not exist on the other nodes or still holds stale
  // content, so the probe fails; on a shared mount every rank reads the token regardless of local
  // device/inode numbering.
  std::vector<int> read_failed(numproc, 0);
  if (myrank == 0) read_failed[0] = write_failed;
  {
    const int fd = ::open(probe_file_path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (fd < 0)
      read_failed[myrank] = 1;
    else
    {
      std::string content(static_cast<std::size_t>(token_size), '\0');
      if (::read(fd, content.data(), token_size) != token_size or content != token)
        read_failed[myrank] = 1;
      ::close(fd);
    }
  }
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, read_failed.data(), 1, MPI_INT, comm_);

  // The probe file is transient; remove it again. Only rank 0 ever creates the file, so only rank
  // 0 unlinks it -- this avoids that several ranks race with each other on a shared filesystem and
  // that a late rank removes a replacement that appeared at the probe path after the first unlink.
  // A pre-existing entry at the probe path is never unlinked (probe_created stays 0 then).
  if (myrank == 0 && probe_created != 0) ::unlink(probe_file_path.c_str());

  // determine whether any rank failed to write or observe the token
  bool any_read_failed = false;
  for (const int failed : read_failed) any_read_failed = any_read_failed || (failed != 0);

  if (any_read_failed)
  {
    std::string message =
        "The collective VTU writer requires that all ranks access the same physical output file, "
        "i.e., the output directory has to be located on a node-shared filesystem. ";
    if (read_failed[0] != 0)
    {
      message += "Failed to create or fill the probe file '" + probe_file_path + "'";
      if (probe_create_errno == EEXIST)
      {
        message +=
            " because an entry already exists at this path (pre-existing entries are never "
            "touched)";
      }
      else
      {
        message += " (errno " + std::to_string(probe_create_errno) + ")";
      }
      message += ". ";
    }
    else
    {
      message += "The probe token written by rank 0 is not visible on rank(s):";
      for (int r = 0; r < numproc; ++r)
        if (read_failed[r] != 0) message += " " + std::to_string(r);
      message += '.';
    }
    message +=
        " Either use OUTPUT_WRITER vtu_per_rank or relocate the output directory to a "
        "node-shared filesystem.";
    FOUR_C_THROW("{}", message);
  }
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::initialize_time_step(
    const double visualization_time, const int visualization_step)
{
  vtu_writer_.reset_time_and_time_step(visualization_time, visualization_step);
  vtu_writer_.initialize_current_time_step_output_file_name();

  const std::string byteorder = "LittleEndian";

  // reset the buffers for the current time step
  piece_buffer_buf_.clear();
  piece_buffer_.clear();
  header_buffer_buf_.clear();
  header_buffer_.clear();
  dummy_master_.str("");
  dummy_master_.clear();
  piece_opened_ = false;
  written_point_data_arrays_.clear();
  written_cell_data_arrays_.clear();

  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    // write the VTU file header (including the <FieldData> section) on proc 0 only
    vtu_writer_.write_vtk_file_header(header_buffer_, byteorder);

    vtu_writer_.append_master_file_and_time_to_collection_file_mid_section_content(
        std::filesystem::path(vtu_writer_.output_file_name_shared()).filename().string());
  }
  else
  {
    // set the writer into the initial phase without contributing bytes to the file
    vtu_writer_.write_vtk_file_header(dummy_master_, byteorder);
  }
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_field_data_to_disk(
    const std::map<std::string, visualization_vector_type_variant>& field_data_map)
{
  if (Core::Communication::my_mpi_rank(comm_) == 0)
    vtu_writer_.write_vtk_field_data_and_or_time_and_or_cycle(header_buffer_, field_data_map);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_geometry_to_disk(
    const std::vector<double>& point_coordinates,
    const std::vector<Core::IO::index_type>& point_cell_connectivity,
    const std::vector<Core::IO::index_type>& cell_offset, const std::vector<uint8_t>& cell_types,
    const std::vector<Core::IO::index_type>& face_connectivity,
    const std::vector<Core::IO::index_type>& face_offset)
{
  const bool has_points = !point_coordinates.empty();
  const bool has_cells = !cell_types.empty();

  piece_opened_ = has_points || has_cells;

  if (not piece_opened_) return;

  vtu_writer_.write_geometry_unstructured_grid(piece_buffer_, dummy_master_, point_coordinates,
      point_cell_connectivity, cell_offset, cell_types, face_connectivity, face_offset);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_point_data_vector_to_disk(
    const visualization_vector_type_variant& data, const unsigned int num_components_per_point,
    const std::string& name)
{
  if (not piece_opened_) return;

  vtu_writer_.write_point_data_vector(
      piece_buffer_, dummy_master_, data, num_components_per_point, name);

  track_written_array(written_point_data_arrays_, name, vtk_type_string(data),
      static_cast<int>(num_components_per_point));
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_cell_data_vector_to_disk(
    const visualization_vector_type_variant& data, const unsigned int num_components_per_point,
    const std::string& name)
{
  if (not piece_opened_) return;

  vtu_writer_.write_cell_data_vector(
      piece_buffer_, dummy_master_, data, num_components_per_point, name);

  track_written_array(written_cell_data_arrays_, name, vtk_type_string(data),
      static_cast<int>(num_components_per_point));
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::finalize_time_step()
{
  // VTK reads the array layout from the first <Piece> of the file. Since this rank only writes the
  // arrays for which it has piece content, ranks without any content could otherwise produce a
  // <Piece> that misses PointData/CellData arrays entirely, breaking the read for the whole file.
  // Gather the union of all arrays written by any rank so that every <Piece> can declare the same,
  // complete array layout (with empty arrays on ranks without content).
  const std::vector<VtuArrayMetadata> all_point_data_arrays =
      gather_written_arrays(written_point_data_arrays_);
  const std::vector<VtuArrayMetadata> all_cell_data_arrays =
      gather_written_arrays(written_cell_data_arrays_);

  // VTK reads the array layout of a shared, multi-piece VTU file from the first <Piece>. Empty
  // ranks are handled by write_empty_piece() below, which builds the complete layout from the
  // union; a non-empty rank, in contrast, keeps the arrays it was streamed with. If such a rank
  // lacks an optional point/cell data array that another rank writes (and the first <Piece> in
  // particular omits it), its <Piece> silently diverges from the layout of the first <Piece> and
  // breaks the read for the whole file. Detect this here instead of writing a corrupt file.
  const auto find_arrays_missing_from_union =
      [](const std::vector<VtuArrayMetadata>& union_arrays,
          const std::vector<VtuArrayMetadata>& written_arrays)
  {
    std::vector<std::string> missing;
    for (const auto& array : union_arrays)
    {
      const bool written_with_matching_metadata = std::ranges::any_of(written_arrays,
          [&array](const VtuArrayMetadata& candidate)
          {
            return candidate.name == array.name and candidate.type == array.type and
                   candidate.num_components == array.num_components;
          });
      if (not written_with_matching_metadata) missing.push_back(array.name);
    }
    return missing;
  };

  // Determine whether any non-empty piece lacks an array of the union. The error is coordinated
  // across all ranks (MPI_MAX) before throwing so that no rank leaves a collective call while its
  // peers are still waiting inside it.
  std::vector<std::string> missing_point_data;
  std::vector<std::string> missing_cell_data;
  int layout_inconsistent = 0;
  if (piece_opened_)
  {
    missing_point_data =
        find_arrays_missing_from_union(all_point_data_arrays, written_point_data_arrays_);
    missing_cell_data =
        find_arrays_missing_from_union(all_cell_data_arrays, written_cell_data_arrays_);

    if (not missing_point_data.empty() or not missing_cell_data.empty()) layout_inconsistent = 1;
  }
  int any_layout_inconsistent = 0;
  MPI_Allreduce(&layout_inconsistent, &any_layout_inconsistent, 1, MPI_INT, MPI_MAX, comm_);
  if (any_layout_inconsistent)
  {
    std::string message = "The layout of the shared VTU file is inconsistent: at least one rank ";
    if (layout_inconsistent)
    {
      message +=
          "writes a non-empty <Piece> that does not declare the complete array layout and "
          "would therefore break the file (VTK reads the layout from the first <Piece>). "
          "Missing ";
      if (not missing_point_data.empty())
      {
        message += "point data array(s):";
        for (const auto& name : missing_point_data) message += " '" + name + "'";
      }
      if (not missing_cell_data.empty())
      {
        if (not missing_point_data.empty()) message += ", and";
        message += " cell data array(s):";
        for (const auto& name : missing_cell_data) message += " '" + name + "'";
      }
      message +=
          ". These arrays are written by other ranks; register them on every rank (empty arrays "
          "are allowed) so that all pieces declare the same layout.";
    }
    else
    {
      message +=
          "does not declare the complete array layout (see the error message on that "
          "rank).";
    }
    FOUR_C_THROW("{}", message);
  }

  // close the current piece (or write a well-formed, but empty piece if this rank has no content)
  if (piece_opened_)
    vtu_writer_.write_vtk_piece_footer(piece_buffer_, dummy_master_);
  else
    write_empty_piece(piece_buffer_, all_point_data_arrays, all_cell_data_arrays);

  // serialize the VTU file footer into a separate buffer (only the last rank writes it)
  std::ostringstream file_footer_buffer;
  vtu_writer_.write_vtk_file_footer(file_footer_buffer);
  const std::string file_footer = file_footer_buffer.str();

  const int myrank = Core::Communication::my_mpi_rank(comm_);
  const int numproc = Core::Communication::num_mpi_ranks(comm_);

  // determine the size of the header written by proc 0
  uint64_t header_size = header_buffer_buf_.buffer().size();
  MPI_Bcast(&header_size, 1, MPI_UINT64_T, 0, comm_);

  // determine the size of this processor's piece and its offset in the file
  const std::string piece = piece_buffer_buf_.release();
  uint64_t piece_size = piece.size();
  uint64_t piece_offset = 0;
  MPI_Exscan(&piece_size, &piece_offset, 1, MPI_UINT64_T, MPI_SUM, comm_);
  if (myrank == 0) piece_offset = 0;

  // total size of all pieces
  uint64_t total_piece_size = 0;
  MPI_Allreduce(&piece_size, &total_piece_size, 1, MPI_UINT64_T, MPI_SUM, comm_);

  {
    // Check every MPI-IO result and coordinate the error across all ranks before throwing so that
    // no rank leaves a collective call while its peers are still waiting inside it.
    const auto throw_if_mpi_io_failed = [this](const int ierr, const std::string& message)
    {
      int max_ierr = ierr;
      MPI_Allreduce(MPI_IN_PLACE, &max_ierr, 1, MPI_INT, MPI_MAX, comm_);
      if (max_ierr != MPI_SUCCESS) FOUR_C_THROW("{}", message);
    };

    // Switch to the output directory and open the shared file by its short relative name. Passing
    // the full (potentially very long) absolute path into MPI_File_open would overflow a fixed-size
    // 256 byte buffer in the shared file pointer component of OpenMPI <= 4.1.6 and abort the run
    // (see ScopedWorkingDirectoryChange). The working directory is restored again right after the
    // file has been closed.
    const std::filesystem::path shared_file_path = vtu_writer_.output_file_name_shared();
    const std::string shared_file_name = shared_file_path.filename().string();
    const ScopedWorkingDirectoryChange change_working_directory(
        shared_file_path.parent_path().string());

    // open the shared file for collective writing
    MPI_File file_handle = MPI_FILE_NULL;
    const int ierr = MPI_File_open(comm_, shared_file_name.c_str(),
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_handle);
    throw_if_mpi_io_failed(ierr, "MPI_File_open failed for the shared VTU file.");

    // truncate a possibly existing file from a previous run
    throw_if_mpi_io_failed(
        MPI_File_set_size(file_handle, 0), "Truncating the shared VTU file failed.");
    throw_if_mpi_io_failed(MPI_Barrier(comm_), "Synchronizing the ranks before writing failed.");

    if (header_size > std::numeric_limits<int>::max())
      FOUR_C_THROW("VTU header exceeds the maximum supported size.");

    // proc 0 writes the header (with the field data) at the beginning of the file
    int write_err = MPI_SUCCESS;
    if (myrank == 0)
    {
      write_err = MPI_File_write_at(file_handle, 0, header_buffer_buf_.buffer().data(),
          static_cast<int>(header_size), MPI_BYTE, MPI_STATUS_IGNORE);
    }
    throw_if_mpi_io_failed(write_err, "Writing the VTU file header failed.");

    // determine the maximum piece size across all ranks so every rank takes the same branch
    uint64_t max_piece_size = 0;
    MPI_Allreduce(&piece_size, &max_piece_size, 1, MPI_UINT64_T, MPI_MAX, comm_);
    if (max_piece_size > std::numeric_limits<int>::max())
      FOUR_C_THROW("VTU piece exceeds the maximum supported size.");

    // all processors write their piece content at the correct offset
    const MPI_Offset write_offset = header_size + piece_offset;
    throw_if_mpi_io_failed(MPI_File_write_at_all(file_handle, write_offset, piece.data(),
                               static_cast<int>(piece_size), MPI_BYTE, MPI_STATUS_IGNORE),
        "Writing the collective VTU piece to the shared file failed.");
    if (file_footer.size() > std::numeric_limits<int>::max())
      FOUR_C_THROW("VTU footer exceeds the maximum supported size.");

    // the last processor appends the file footer
    write_err = MPI_SUCCESS;
    if (myrank == numproc - 1)
    {
      const MPI_Offset footer_offset = header_size + total_piece_size;
      write_err = MPI_File_write_at(file_handle, footer_offset, file_footer.data(),
          static_cast<int>(file_footer.size()), MPI_BYTE, MPI_STATUS_IGNORE);
    }
    throw_if_mpi_io_failed(write_err, "Writing the VTU file footer failed.");

    throw_if_mpi_io_failed(MPI_File_sync(file_handle), "Synchronizing the shared VTU file failed.");
    throw_if_mpi_io_failed(MPI_File_close(&file_handle), "Closing the shared VTU file failed.");
  }

  // Write a collection file summarizing all previously written files
  vtu_writer_.write_vtk_collection_file_for_all_written_master_files(
      parameters_.file_name_prefix_ + "-" + visualization_data_name_);
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::track_written_array(
    std::vector<VtuArrayMetadata>& written_arrays, const std::string& name, const std::string& type,
    const int num_components)
{
  // only the first occurrence of an array name is remembered
  const auto existing = std::ranges::find_if(
      written_arrays, [&name](const VtuArrayMetadata& array) { return array.name == name; });

  if (existing == written_arrays.end())
    written_arrays.push_back({.name = name, .type = type, .num_components = num_components});
}

/**
 *
 */
std::vector<Core::IO::VisualizationWriterVtuCollective::VtuArrayMetadata>
Core::IO::VisualizationWriterVtuCollective::gather_written_arrays(
    const std::vector<VtuArrayMetadata>& local_arrays) const
{
  const int numproc = Core::Communication::num_mpi_ranks(comm_);

  // serialize the local array list: each entry consists of the name, a terminating null character,
  // the VTK type string, a terminating null character and four bytes for the number of components
  std::string serialized;
  for (const VtuArrayMetadata& array : local_arrays)
  {
    serialized.append(array.name);
    serialized.push_back('\0');
    serialized.append(array.type);
    serialized.push_back('\0');
    const int num_components = array.num_components;
    serialized.append(reinterpret_cast<const char*>(&num_components), sizeof(num_components));
  }

  // gather the serialized lists of all ranks
  std::vector<int> sizes(numproc, 0);
  const int local_size = static_cast<int>(serialized.size());
  MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm_);

  std::vector<int> offsets(numproc, 0);
  int total_size = 0;
  for (int rank = 0; rank < numproc; ++rank)
  {
    offsets[rank] = total_size;
    total_size += sizes[rank];
  }

  std::string all_serialized(static_cast<std::size_t>(total_size), '\0');
  MPI_Allgatherv(serialized.data(), local_size, MPI_BYTE, all_serialized.data(), sizes.data(),
      offsets.data(), MPI_BYTE, comm_);

  // merge into the union of all arrays, keeping the order of first occurrence so that the array
  // layout is deterministic across time steps
  std::vector<VtuArrayMetadata> union_arrays;
  for (int rank = 0; rank < numproc; ++rank)
  {
    std::size_t position = 0;
    const auto end = static_cast<std::size_t>(sizes[rank]);
    while (position < end)
    {
      const std::string name(all_serialized.data() + offsets[rank] + position);
      position += name.size() + 1;
      const std::string type(all_serialized.data() + offsets[rank] + position);
      position += type.size() + 1;
      int num_components = 0;
      std::memcpy(&num_components, all_serialized.data() + offsets[rank] + position,
          sizeof(num_components));
      position += sizeof(num_components);

      const auto existing = std::ranges::find_if(
          union_arrays, [&name](const VtuArrayMetadata& array) { return array.name == name; });

      if (existing == union_arrays.end())
        union_arrays.push_back({.name = name, .type = type, .num_components = num_components});
    }
  }
  return union_arrays;
}

/**
 *
 */
std::string Core::IO::VisualizationWriterVtuCollective::vtk_type_string(
    const visualization_vector_type_variant& data)
{
  if (std::holds_alternative<std::vector<double>>(data)) return "Float64";
  if (std::holds_alternative<std::vector<int>>(data)) return "Int32";
  FOUR_C_THROW("Got unexpected vector type");
}

/**
 *
 */
void Core::IO::VisualizationWriterVtuCollective::write_empty_piece(std::ostream& piece_buffer,
    const std::vector<VtuArrayMetadata>& point_data_arrays,
    const std::vector<VtuArrayMetadata>& cell_data_arrays) const
{
  // This rank has no piece content (no points and no cells), but it still has to contribute a
  // well-formed <Piece> to the shared file: VTK requires every <Piece> to contain a present Cells
  // element with at least one nested array, and it uses the array layout of the first <Piece> for
  // the whole file. The empty arrays are written with the standard zlib-compressed empty block so
  // that VTK parses them without warnings (see LibB64::write_compressed_block).
  const bool binary = parameters_.data_format_ == OutputDataFormat::binary;

  const auto write_empty_data_array =
      [&piece_buffer, &binary, compression = parameters_.compression_level_](
          const std::string& type, const std::string& name, const int num_components)
  {
    piece_buffer << "        <DataArray type=\"" << type << "\"";
    if (not name.empty()) piece_buffer << " Name=\"" << name << "\"";
    if (num_components > 1) piece_buffer << " NumberOfComponents=\"" << num_components << "\"";
    piece_buffer << " format=\"" << (binary ? "binary" : "ascii") << "\"";
    if (binary)
    {
      piece_buffer << ">\n";
      LibB64::write_compressed_block(std::vector<int>{}, piece_buffer, compression);
      piece_buffer << "        </DataArray>\n";
    }
    else
    {
      piece_buffer << "/>\n";
    }
  };

  piece_buffer << "    <Piece NumberOfPoints=\"0\" NumberOfCells=\"0\">\n";
  piece_buffer << "      <Points>\n";
  write_empty_data_array("Float64", "", 3);
  piece_buffer << "      </Points>\n";

  piece_buffer << "      <Cells>\n";
  write_empty_data_array("Int32", "connectivity", 1);
  write_empty_data_array("Int32", "offsets", 1);
  write_empty_data_array("UInt8", "types", 1);
  piece_buffer << "      </Cells>\n";

  piece_buffer << "      <PointData>\n";
  for (const VtuArrayMetadata& array : point_data_arrays)
    write_empty_data_array(array.type, array.name, array.num_components);
  piece_buffer << "      </PointData>\n";

  piece_buffer << "      <CellData>\n";
  for (const VtuArrayMetadata& array : cell_data_arrays)
    write_empty_data_array(array.type, array.name, array.num_components);
  piece_buffer << "      </CellData>\n";

  piece_buffer << "    </Piece>\n";
}

FOUR_C_NAMESPACE_CLOSE