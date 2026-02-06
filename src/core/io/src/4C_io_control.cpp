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
#include "4C_io_pstream.hpp"
#include "4C_io_yaml.hpp"
#include "4C_utils_exceptions.hpp"

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

struct Core::IO::ControlFileWriter::Impl
{
 public:
  explicit Impl(std::ostream& out) : out_(out)
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



Core::IO::ControlFileWriter::ControlFileWriter(bool do_write, std::ostream& stream)
    : pimpl_(do_write ? std::make_unique<Impl>(stream) : nullptr)
{
}

Core::IO::ControlFileWriter::~ControlFileWriter() { end_all_groups_and_flush(); }


void Core::IO::ControlFileWriter::write_metadata_header()
{
  if (!pimpl_) return;

  time_t time_value = std::time(nullptr);
  auto local_time = std::localtime(&time_value);
  std::ostringstream time_format_stream;
  time_format_stream << std::put_time(local_time, "%d-%m-%Y %H-%M-%S");

  std::array<char, 256> hostname;
  passwd* user_entry = getpwuid(getuid());
  gethostname(hostname.data(), 256);

  start_group("metadata")
      .write("created_by", user_entry->pw_name)
      .write("host", hostname.data())
      .write("time", time_format_stream.str())
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

  if (write_binary_output_ && myrank_ == 0)
  {
    control_file_stream_.open(filename_ + ".control", std::ios::out);
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
      control_file().write("restarted_from_step", restart_step);
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

static std::string read_file(const std::string& filename, MPI_Comm comm)
{
  int myrank = (comm != MPI_COMM_NULL) ? Core::Communication::my_mpi_rank(comm) : 0;
  int nprocs = (comm != MPI_COMM_NULL) ? Core::Communication::num_mpi_ranks(comm) : 1;

  std::string file_content;
  if (myrank == 0)
  {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
      FOUR_C_THROW("Cannot read file '{}'", filename);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    file_content.resize(file_size);
    if (!file.read(file_content.data(), file_size))
    {
      FOUR_C_THROW("Failed to read file {}", filename);
    }
  }

  if (nprocs > 1)
  {
    Core::Communication::broadcast(file_content, 0, comm);
  }

  return file_content;
}

struct Core::IO::InputControl::Impl
{
  ryml::Tree tree;
  std::string filename;
  std::string file_buffer;

  ryml::ConstNodeRef node_at_index(size_t index) const { return tree.ref(index); }
};


Core::IO::ControlFileEntry::ControlFileEntry(const InputControl* control, size_t index)
    : control_(control), index_(index)
{
}


template <Core::IO::ControlFileEntrySupportedType T>
bool Core::IO::ControlFileEntry::is_a() const
{
  if (!is_valid()) return false;
  ryml::ConstNodeRef node = control_->pimpl_->node_at_index(index_);
  T val;
  auto status = read_value_from_yaml(ConstYamlNodeRef{node, ""}, val);
  return status == YamlReadStatus::success;
}

template <Core::IO::ControlFileEntrySupportedType T>
std::optional<T> Core::IO::ControlFileEntry::as() const
{
  if (!is_valid()) return std::nullopt;
  ryml::ConstNodeRef node = control_->pimpl_->node_at_index(index_);
  T val;
  auto status = read_value_from_yaml(ConstYamlNodeRef{node, ""}, val);
  if (status != YamlReadStatus::success) return std::nullopt;
  return val;
}



bool Core::IO::ControlFileEntry::is_group() const
{
  if (!is_valid()) return false;
  ryml::ConstNodeRef node = control_->pimpl_->node_at_index(index_);
  return node.is_map();
}


Core::IO::ControlFileEntry Core::IO::ControlFileEntry::operator[](const std::string& key) const
{
  if (!is_valid()) return {};

  ryml::ConstNodeRef node = control_->pimpl_->node_at_index(index_);
  FOUR_C_ASSERT(node.is_map(), "Internal error: expected a map node");

  if (!node.has_child(ryml::to_csubstr(key))) return {};
  ryml::ConstNodeRef child_node = node[ryml::to_csubstr(key)];
  return {control_, child_node.id()};
}

Core::IO::InputControl::InputControl(const std::string& filename, const bool serial)
    : InputControl(filename, serial ? MPI_COMM_NULL : MPI_COMM_WORLD)
{
}



Core::IO::InputControl::InputControl(const std::string& filename, MPI_Comm comm)
    : pimpl_(std::make_unique<Impl>())
{
  pimpl_->filename = filename;
  std::stringstream name;
  name << filename << ".control";
  pimpl_->file_buffer = read_file(name.str(), comm);
  ryml::parse_in_place(ryml::to_substr(pimpl_->file_buffer), &pimpl_->tree);
}



Core::IO::InputControl::~InputControl() = default;



const std::string& Core::IO::InputControl::file_name() const { return pimpl_->filename; }

Core::IO::ControlFileEntry Core::IO::InputControl::nth_last_entry(
    std::optional<std::string> key, size_t n) const
{
  auto root = pimpl_->tree.rootref();
  FOUR_C_ASSERT(root.is_seq(), "Internal error: expected top-level sequence in control file");

  size_t count = 0;
  for (int i = root.num_children() - 1; i >= 0; --i)
  {
    ryml::NodeRef entry = root[i];
    FOUR_C_ASSERT(entry.is_map(), "Internal error: expected sequence items to be maps");
    FOUR_C_ASSERT(entry.num_children() == 1, "Internal error: expected exactly one child");

    ryml::NodeRef symbol_node = entry.child(0);
    FOUR_C_ASSERT(symbol_node.is_map(), "Internal error: expected a map");

    if (!key || symbol_node.key() == ryml::to_csubstr(*key))
    {
      ++count;
      if (count == n)
      {
        return {this, symbol_node.id()};
      }
    }
  }

  return {};  // not found
}


std::optional<std::string> Core::IO::InputControl::last_key() const
{
  auto root = pimpl_->tree.rootref();
  FOUR_C_ASSERT(root.is_seq(), "Internal error: expected top-level sequence in control file");

  if (root.num_children() == 0) return std::nullopt;

  // last entry in the sequence
  ryml::ConstNodeRef entry = root[root.num_children() - 1];
  FOUR_C_ASSERT(entry.is_map(), "Internal error: sequence items must be maps");
  FOUR_C_ASSERT(
      entry.num_children() == 1, "Internal error: each entry must have exactly one child");

  ryml::ConstNodeRef symbol_node = entry.child(0);
  FOUR_C_ASSERT(symbol_node.is_map(), "Internal error: entry child must be a map");

  auto key = symbol_node.key();
  return std::string(key.data(), key.size());
}



std::pair<Core::IO::ControlFileEntry, Core::IO::ControlFileEntry>
Core::IO::InputControl::find_group(int step, const std::string& discretization_name,
    const std::string& group_name, const std::string& file_marker) const
{
  ControlFileEntry result_info;
  ControlFileEntry file_info;

  auto root = pimpl_->tree.rootref();
  FOUR_C_ASSERT(root.is_seq(), "Internal error: expected top-level sequence in control file");

  for (int i = root.num_children() - 1; i >= 0; --i)
  {
    ryml::NodeRef entry = root[i];
    FOUR_C_ASSERT(entry.is_map(), "Internal error: expected sequence items to be maps");
    FOUR_C_ASSERT(entry.num_children() == 1, "Internal error: expected exactly one child");
    ryml::NodeRef symbol_node = entry.child(0);
    FOUR_C_ASSERT(symbol_node.is_map(), "Internal error: expected a map");
    auto symbol_key = symbol_node.key();
    if (symbol_key != ryml::to_csubstr(group_name)) continue;

    const auto has_entry_with_value =
        []<typename T>(const ControlFileEntry& group, const std::string& key, const T& val)
    {
      auto entry = group[key].as<T>();
      if (entry) return (*entry == val);
      return false;
    };

    ControlFileEntry group = ControlFileEntry(this, symbol_node.id());

    if (!result_info.is_valid())
    {
      if (!has_entry_with_value(group, "field", discretization_name)) continue;
      if (!has_entry_with_value(group, "step", step)) continue;
      // Found the correct group
      result_info = group;
    }

    if (result_info.is_valid())
    {
      if (!has_entry_with_value(group, "field", discretization_name)) continue;

      if (!group[file_marker].is_a<std::string>()) continue;

      // Found the correct file info --> stop searching
      file_info = group;
      break;
    }
  }

  FOUR_C_ASSERT_ALWAYS(result_info.is_valid() && file_info.is_valid(),
      "No restart entry for discretization '{}' step {} in control file. "
      "Control file corrupt?\n\nLooking in control file at: {}",
      discretization_name, step, file_name());

  return {result_info, file_info};
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::IO::get_last_possible_restart_step(Core::IO::InputControl& inputcontrol)
{
  // field entries indicate a step with restart information
  auto last_possible = inputcontrol.nth_last_entry("field")["step"].as<int>();

  if (!last_possible.has_value())  // throw if no field entry exists
  {
    FOUR_C_THROW(
        "Cannot find restart entry in the control file at: {}\n"
        "Control file corrupt?",
        inputcontrol.file_name());
  }

  // If the very last entry is not a field entry, return the last possible restart step directly.
  // Otherwise there is an accompanying result description missing.
  if (inputcontrol.last_key() != "field")
  {
    return last_possible.value();
  }
  else  // try to find the second last possible restart step
  {
    // only on first processor print warning
    if (Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Warning: Restart information at step " << last_possible.value()
                << " is incomplete. Attempting to find the second last restart step...\n";
    }
    // Start looking at the second last field entry
    int counter = 2;
    while (true)
    {
      auto second_last_possible = inputcontrol.nth_last_entry("field", counter)["step"].as<int>();
      // throw if no more field entries exist
      if (!second_last_possible.has_value())
      {
        FOUR_C_THROW("Cannot find complete restart entry in the control file at: {}\n",
            inputcontrol.file_name());
      }
      // Return the step of the first field entry with a step smaller than the previously found one.
      // This ensures that restart information is complete at that step (for all fields).
      if (second_last_possible.value() < last_possible.value())
      {
        if (Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 0)
        {
          std::cout << "Attempting to restart from the second last restart step "
                    << second_last_possible.value() << " instead.\n";
        }
        return second_last_possible.value();
      }
      // increment counter
      ++counter;
    }
  }
}

template bool Core::IO::ControlFileEntry::is_a<int>() const;
template bool Core::IO::ControlFileEntry::is_a<double>() const;
template bool Core::IO::ControlFileEntry::is_a<std::string>() const;

template std::optional<int> Core::IO::ControlFileEntry::as() const;
template std::optional<double> Core::IO::ControlFileEntry::as() const;
template std::optional<std::string> Core::IO::ControlFileEntry::as() const;

FOUR_C_NAMESPACE_CLOSE
