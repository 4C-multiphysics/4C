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

#include <pwd.h>
#include <unistd.h>

#include <array>
#include <ctime>
#include <filesystem>
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
      control_file_(myrank_ == 0),
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
    std::stringstream name;
    name << filename_ << ".control";

    control_file().open_and_write_header(name.str());

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
  }
}



Core::IO::ControlFile::ControlFile(bool do_write) : do_write_(do_write) {}


void Core::IO::ControlFile::open_and_write_header(const std::string& control_file_name)
{
  if (!do_write_) return;

  FOUR_C_ASSERT(!file_.is_open(), "Internal error: control file already opened");

  file_.open(control_file_name.c_str(), std::ios_base::out);
  if (not file_) FOUR_C_THROW("Could not open control file '{}' for writing", control_file_name);

  time_t time_value;
  time_value = time(nullptr);

  std::array<char, 256> hostname;
  struct passwd* user_entry;
  user_entry = getpwuid(getuid());
  gethostname(hostname.data(), 256);

  file_ << "# 4C output control file\n"
        << "# created by " << user_entry->pw_name << " on " << hostname.data() << " at "
        << ctime(&time_value) << "# using code version (git SHA1) " << VersionControl::git_hash
        << " \n\n";
}

Core::IO::ControlFile& Core::IO::ControlFile::write(
    std::string_view key, const std::string_view& value)
{
  if (!do_write_) return *this;

  file_ << indent() << key << " = \"" << value << "\"\n";
  file_ << std::flush;
  return *this;
}

Core::IO::ControlFile& Core::IO::ControlFile::write(std::string_view key, int value)
{
  if (!do_write_) return *this;

  file_ << indent() << key << " = " << value << "\n";
  file_ << std::flush;
  return *this;
}

Core::IO::ControlFile& Core::IO::ControlFile::write(std::string_view key, double value)
{
  if (!do_write_) return *this;

  file_ << indent() << key << " = " << std::scientific << std::setprecision(16) << value << "\n";
  file_ << std::flush;
  return *this;
}

Core::IO::ControlFile& Core::IO::ControlFile::start_group(const std::string& group_name)
{
  if (!do_write_) return *this;

  file_ << indent() << group_name << ":\n";
  indent_ += indent_increment_;
  return *this;
}

Core::IO::ControlFile& Core::IO::ControlFile::end_group()
{
  if (!do_write_) return *this;

  FOUR_C_ASSERT(
      indent_ >= indent_increment_, "Internal error: end_group() is missing a start_group()");
  indent_ -= indent_increment_;
  // Print an extra newline after ending a group for better readability
  file_ << "\n";
  file_ << std::flush;
  return *this;
}

Core::IO::ControlFile& Core::IO::ControlFile::try_end_group()
{
  if (!do_write_) return *this;

  if (indent_ >= indent_increment_) return end_group();
  return *this;
}

std::string Core::IO::ControlFile::indent() const { return std::string(indent_, ' '); }

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
