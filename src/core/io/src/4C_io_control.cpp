// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"
#include "4C_config_revision.hpp"

#include "4C_io_control.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_legacy_table.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_yaml.hpp"

#include <pwd.h>
#include <unistd.h>

#include <array>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/// find position of restart number in filename (if existing):
/// for "outname-5" will return position of the "-"
/// returns std::string::npos if not found
static size_t restart_finder(const std::string& filename)
{
  size_t pos;
  for (pos = filename.size(); pos > 0; --pos)
  {
    if (filename[pos - 1] == '-') return pos - 1;

    if (not std::isdigit(filename[pos - 1]) or filename[pos - 1] == '/') return std::string::npos;
  }
  return std::string::npos;
}

namespace Core::IO::Internal
{
  class ControlFileWriterImpl
  {
   public:
    ControlFileWriterImpl(std::ostream& out) : out_(out)
    {
      tree_ = init_yaml_tree_with_exceptions();
      current_node_id_ = tree_.rootref().id();
      tree_.rootref() |= ryml::SEQ;
    }

    // Flush the current tree to the output stream and reset it.
    // Ends all open groups.
    void flush_and_reset_tree()
    {
      // Only write if there is anything to write. Otherwise, we get an ugly "[]" in the output.
      if (tree_.rootref().num_children() > 0) out_ << tree_ << "\n" << std::flush;

      // Clear the tree and reset the current node to the root
      tree_.clear();
      tree_.clear_arena();
      current_node_id_ = tree_.rootref().id();
      group_level = 0;
      tree_.rootref() |= ryml::SEQ;
    }

    template <typename T>
    void write(std::string_view key, const T& value)
    {
      FOUR_C_ASSERT_ALWAYS(current().is_map(), "Internal error: appending to non-map node");
      auto node = current().append_child();
      const auto key_str = ryml::csubstr(key.data(), key.size());
      node << ryml::key(key_str);

      emit_value_as_yaml(YamlNodeRef{node, ""}, value);
    }

    void start_group(std::string_view key)
    {
      ++group_level;

      ryml::NodeRef outer = (current().is_seq()) ? current().append_child() : current();
      outer |= ryml::MAP;

      auto group_node = outer.append_child();
      group_node |= ryml::MAP;
      const auto key_str = ryml::csubstr(key.data(), key.size());
      group_node << ryml::key(key_str);
      current_node_id_ = group_node.id();
    }

    void end_group()
    {
      FOUR_C_ASSERT(group_level > 0, "Internal error: unmatched end_group()");
      --group_level;
      current_node_id_ = current().parent().id();

      FOUR_C_ASSERT(current().num_children() > 0, "Internal error: empty group");

      // Whenever we close the outermost group, flush the tree to file
      if (group_level == 0) flush_and_reset_tree();
    }

    ryml::NodeRef current() { return ryml::NodeRef{&tree_, current_node_id_}; }

    //! A temporary tree and node that can be used to build the YAML structure
    //! The data is written to file periodically to avoid excessive memory usage
    ryml::Tree tree_;
    ryml::id_type current_node_id_;

    //! Remember how many groups were opened.
    size_t group_level{0};

    //! output stream for the control file
    std::ostream& out_;
  };
}  // namespace Core::IO::Internal



Core::IO::ControlFileWriter::ControlFileWriter(bool do_write, std::ostream& stream)
    : pimpl_(do_write ? std::make_unique<Internal::ControlFileWriterImpl>(stream) : nullptr)
{
}

Core::IO::ControlFileWriter::~ControlFileWriter() { end_all_groups_and_flush(); }


void Core::IO::ControlFileWriter::write_metadata_header()
{
  if (!pimpl_) return;

  time_t time_value = std::time(nullptr);
  auto local_time = std::localtime(&time_value);
  std::ostringstream time_formst_stream;
  time_formst_stream << std::put_time(local_time, "%d-%m-%Y %H-%M-%S");

  std::array<char, 256> hostname;
  struct passwd* user_entry;
  user_entry = getpwuid(getuid());
  gethostname(hostname.data(), 256);

  start_group("metadata")
      .write("created_by", user_entry->pw_name)
      .write("host", hostname.data())
      .write("time", time_formst_stream.str())
      .write("sha", VersionControl::git_hash)
      .write("version", FOUR_C_VERSION_FULL)
      .end_group();
}

void Core::IO::ControlFileWriter::end_all_groups_and_flush()
{
  if (!pimpl_) return;

  pimpl_->flush_and_reset_tree();
}

Core::IO::ControlFileWriter& Core::IO::ControlFileWriter::write(
    std::string_view key, const std::string_view& value)
{
  if (!pimpl_) return *this;
  pimpl_->write(key, value);
  return *this;
}

Core::IO::ControlFileWriter& Core::IO::ControlFileWriter::write(std::string_view key, int value)
{
  if (!pimpl_) return *this;
  pimpl_->write(key, value);
  return *this;
}

Core::IO::ControlFileWriter& Core::IO::ControlFileWriter::write(std::string_view key, double value)
{
  if (!pimpl_) return *this;
  pimpl_->write(key, value);
  return *this;
}

Core::IO::ControlFileWriter& Core::IO::ControlFileWriter::start_group(const std::string& group_name)
{
  if (!pimpl_) return *this;
  pimpl_->start_group(group_name);
  return *this;
}

Core::IO::ControlFileWriter& Core::IO::ControlFileWriter::end_group()
{
  if (!pimpl_) return *this;
  pimpl_->end_group();
  return *this;
}

Core::IO::ControlFileWriter& Core::IO::ControlFileWriter::try_end_group()
{
  if (!pimpl_) return *this;

  if (pimpl_->group_level > 0) return end_group();
  return *this;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::OutputControl::OutputControl(MPI_Comm comm, std::string problemtype,
    const Core::FE::ShapeFunctionType type_of_spatial_approx, std::string inputfile,
    const std::string& restartname, std::string outputname, const int ndim, const int restart_step,
    const int filesteps, const bool write_binary_output, const bool adaptname)
    : problemtype_(std::move(problemtype)),
      inputfile_(std::move(inputfile)),
      ndim_(ndim),
      myrank_(Core::Communication::my_mpi_rank(comm)),
      filename_(std::move(outputname)),
      restartname_(restartname),
      control_file_stream_(std::invoke([&]() { return std::ofstream(filename_ + ".control"); })),
      control_file_(myrank_ == 0, control_file_stream_),
      filesteps_(filesteps),
      restart_step_(restart_step),
      write_binary_output_(write_binary_output)
{
  if (restart_step)
  {
    if (myrank_ == 0 && adaptname)
    {
      // check whether filename_ includes a dash and in case separate the number at the end
      int number = 0;
      size_t pos = restart_finder(filename_);
      if (pos != std::string::npos)
      {
        number = atoi(filename_.substr(pos + 1).c_str());
        filename_ = filename_.substr(0, pos);
      }

      // either add or increase the number in the end or just set the new name for the control file
      for (;;)
      {
        // if no number is found and the control file name does not yet exist -> create it
        if (number == 0)
        {
          std::stringstream name;
          name << filename_ << ".control";
          std::ifstream file(name.str().c_str());
          if (not file)
          {
            std::cout << "restart with new output file: " << filename_ << '\n';
            break;
          }
        }
        // a number was found or the file does already exist -> set number correctly and add it
        number += 1;
        std::stringstream name;
        name << filename_ << "-" << number << ".control";
        std::ifstream file(name.str().c_str());
        if (not file)
        {
          filename_ = name.str();
          filename_ = filename_.substr(0, filename_.length() - 8);
          std::cout << "restart with new output file: " << filename_ << '\n';
          break;
        }
      }
    }

    if (Core::Communication::num_mpi_ranks(comm) > 1)
    {
      int length = static_cast<int>(filename_.length());
      std::vector<int> name(filename_.begin(), filename_.end());
      Core::Communication::broadcast(&length, 1, 0, comm);
      name.resize(length);
      Core::Communication::broadcast(name.data(), length, 0, comm);
      filename_.assign(name.begin(), name.end());
    }
  }

  if (write_binary_output_)
  {
    control_file().write_metadata_header();

    control_file().start_group("general");
    control_file().write("input_file", inputfile_);
    control_file().write("problem_type", problemtype_);
    control_file().write(
        "spatial_approximation", Core::FE::shape_function_type_to_string(type_of_spatial_approx));
    control_file().write("ndim", ndim_);

    // insert back reference
    if (restart_step)
    {
      size_t pos = outputname.rfind('/');
      control_file().write(
          "restarted_run", ((pos != std::string::npos) ? outputname.substr(pos + 1) : outputname));
    }
    control_file().end_group();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string Core::IO::OutputControl::file_name_only_prefix() const
{
  std::string filenameonlyprefix = filename_;

  size_t pos = filename_.rfind('/');
  if (pos != std::string::npos)
  {
    filenameonlyprefix = filename_.substr(pos + 1);
  }

  return filenameonlyprefix;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string Core::IO::OutputControl::directory_name() const
{
  std::filesystem::path path(filename_);
  return path.parent_path();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::InputControl::InputControl(const std::string& filename, const bool serial)
    : filename_(filename)
{
  std::stringstream name;
  name << filename << ".control";

  if (!serial)
    parse_control_file(&table_, name.str().c_str(), MPI_COMM_WORLD);
  else
    parse_control_file_serial(&table_, name.str().c_str());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::IO::InputControl::InputControl(const std::string& filename, MPI_Comm comm)
    : filename_(filename)
{
  std::stringstream name;
  name << filename << ".control";

  parse_control_file(&table_, name.str().c_str(), comm);
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::InputControl::~InputControl() { destroy_map(&table_); }



void Core::IO::InputControl::find_group(int step, const std::string& discretization_name,
    const char* group_name, const char* filestring, MAP*& result_info, MAP*& file_info)
{
  /* Iterate all symbols under the name "result" and get the one that
   * matches the given step. Note that this iteration starts from the
   * last result group and goes backward. */

  SYMBOL* symbol = map_find_symbol(&table_, group_name);
  while (symbol != nullptr)
  {
    if (symbol_is_map(symbol))
    {
      MAP* map;
      symbol_get_map(symbol, &map);
      if (map_has_string(map, "field", discretization_name.c_str()) and
          map_has_int(map, "step", step))
      {
        result_info = map;
        break;
      }
    }
    symbol = symbol->next;
  }
  if (symbol == nullptr)
  {
    FOUR_C_THROW(
        "No restart entry for discretization '{}' step {} in symbol table. "
        "Control file corrupt?\n\nLooking for control file at: {}",
        discretization_name, step, filename_);
  }

  /*--------------------------------------------------------------------*/
  /* open file to read */

  /* We have a symbol and its map that corresponds to the step we are
   * interested in. Now we need to continue our search to find the
   * step that defines the output file used for our step. */

  while (symbol != nullptr)
  {
    if (symbol_is_map(symbol))
    {
      MAP* map;
      symbol_get_map(symbol, &map);
      if (map_has_string(map, "field", discretization_name.c_str()))
      {
        /*
         * If one of these files is here the other one has to be
         * here, too. If it's not, it's a bug in the input. */
        if (map_symbol_count(map, filestring) > 0)
        {
          file_info = map;
          break;
        }
      }
    }
    symbol = symbol->next;
  }

  /* No restart files defined? */
  if (symbol == nullptr)
  {
    FOUR_C_THROW("no restart file definitions found in control file");
  }
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::IO::get_last_possible_restart_step(Core::IO::InputControl& inputcontrol)
{
  /* Go to the first symbol under the name "field" and get the
   * corresponding step. Note that it will find the last "field"
   * group starting from the end of the file and looking backwards. */

  SYMBOL* symbol = map_find_symbol(inputcontrol.control_file(), "field");
  if (symbol != nullptr && symbol_is_map(symbol))
  {
    MAP* map;
    symbol_get_map(symbol, &map);
    return map_read_int(map, "step");
  }

  FOUR_C_THROW(
      "No restart entry in symbol table. "
      "Control file corrupt?\n\nLooking for control file at: {}",
      inputcontrol.file_name().c_str());

  return 0;
}

FOUR_C_NAMESPACE_CLOSE
