// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_every_iteration_writer.hpp"

#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"

#include <Teuchos_ParameterList.hpp>

#include <filesystem>
#include <iterator>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::IO::EveryIterationWriter::EveryIterationWriter()
    : isinit_(false),
      issetup_(false),
      isnewton_initialized_(false),
      myrank_(-1),
      run_number_(-1),
      write_only_this_step_(-1),
      write_owner_each_newton_iteration_(false),
      base_filename_(),
      parent_writer_(nullptr),
      interface_(nullptr),
      every_iter_writer_(nullptr)
{ /* empty */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::init(const Core::IO::DiscretizationWriter* parent_writer,
    Core::IO::EveryIterationWriterInterface* interface, const Teuchos::ParameterList& params)
{
  issetup_ = false;

  parent_writer_ = parent_writer;
  interface_ = interface;


  myrank_ = parent_writer_->output()->my_rank();

  run_number_ = params.get<int>("RUN_NUMBER");
  write_only_this_step_ = params.get<int>("STEP_NP_NUMBER");
  write_owner_each_newton_iteration_ = params.get<bool>("WRITE_OWNER_EACH_NEWTON_ITER");

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::setup()
{
  throw_if_not_initialized(__LINE__);

  /* Remove the restart counter from the folder name. Note that the restart
   * counter stays a part of the final file name of the corresponding step. */
  const std::string filename_without_restart = remove_restart_step_from_file_name(
      parent_writer().output()->file_name_only_prefix(), parent_writer().output()->restart_step());

  const std::string dir_name(filename_without_restart + "_every_iter");

  std::string file_dir_path = extract_path(parent_writer().output()->file_name());
  file_dir_path += dir_name;
  create_directory(file_dir_path);

  // modify the prefix and copy possible restart number
  std::string prefix;
  prefix = dir_name + "/" + create_run_directory(file_dir_path);

  std::shared_ptr<Core::IO::OutputControl> control_iteration = nullptr;
  control_iteration =
      std::make_shared<Core::IO::OutputControl>(*parent_writer().output(), prefix.c_str());

  // adjust steps per file
  adjust_steps_per_file(*control_iteration);

  // create new output writer object
  every_iter_writer_ = std::make_shared<Core::IO::DiscretizationWriter>(
      parent_writer(), control_iteration, Core::IO::CopyType::shape);

  // save base file name
  base_filename_ = every_iter_writer_->output()->file_name();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string Core::IO::EveryIterationWriter::create_run_directory(
    const std::string& file_dir_path) const
{
  if (run_number_ < 0) return "";

  std::ostringstream run_dir;
  run_dir << "run_" << run_number_;

  std::string full_dir_path = file_dir_path + "/" + run_dir.str();
  create_directory(full_dir_path);

  return run_dir.str() + "/";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::create_directory(const std::string& dir_path) const
{
  Core::IO::create_directory(dir_path, myrank_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::create_directory(const std::string& dir_path, const int myrank)
{
  if (myrank != 0) return;

  std::filesystem::path dir(dir_path);
  if (!std::filesystem::is_directory(dir))
    if (!std::filesystem::create_directory(dir))
      FOUR_C_THROW("The directory \"%s\" could not be created!", dir_path.c_str());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string Core::IO::extract_path(const std::string& full_filename)
{
  std::string filename_path;
  size_t pos = full_filename.rfind('/');

  if (pos != std::string::npos) filename_path = full_filename.substr(0, pos + 1);

  return filename_path;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string Core::IO::extract_file_name(const std::string& full_filename)
{
  std::string filenameonly = full_filename;

  size_t pos = full_filename.rfind('/');
  if (pos != std::string::npos)
  {
    filenameonly = full_filename.substr(pos + 1);
  }

  return filenameonly;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string Core::IO::remove_restart_step_from_file_name(
    const std::string& filename, const int restart_step)
{
  if (restart_step == 0) return filename;

  size_t pos = filename.rfind('-');
  if (pos == std::string::npos) return filename;

  // potential restart counter
  const char rc = filename.at(pos + 1);

  // check if it is really an integer
  const int irc = rc - '0';
  if (irc >= 0 and irc <= 9)
    return filename.substr(0, pos);
  else
    return filename;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Core::IO::EveryIterationWriter::write_this_step() const
{
  return (write_only_this_step_ < 0 or write_only_this_step_ == interface_->get_step_np());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::adjust_steps_per_file(Core::IO::OutputControl& control) const
{
  int new_file_steps = control.file_steps() * MAX_NUMBER_LINE_SEARCH_ITERATIONS_;
  if (new_file_steps > std::numeric_limits<int>::max())
    new_file_steps = std::numeric_limits<int>::max();

  control.set_file_steps(new_file_steps);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::print_path2_screen(const std::string& path) const
{
  Core::IO::cout << "every iteration output path: \"" << path << "\"" << Core::IO::endl;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::init_newton_iteration()
{
  throw_if_not_setup(__LINE__);

  if (not write_this_step()) return;

  const int curr_step = interface_->get_step_np();

  // create new result and mesh files for each step
  std::ostringstream result_name;
  result_name << base_filename_ << "_step_" << curr_step;

  print_path2_screen(result_name.str());

  every_iter_writer_->new_result_file(result_name.str());
  every_iter_writer_->write_mesh(0, 0.0);

  every_iter_writer_->new_step(0, 0.0);

  constexpr bool force_prepare = false;
  interface_->prepare_output(force_prepare);
  interface_->output_debug_state(*every_iter_writer_, true);

  isnewton_initialized_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::add_newton_iteration(const int newton_iteration)
{
  throw_if_not_setup(__LINE__);

  if (not write_this_step()) return;

  if (not isnewton_initialized_) FOUR_C_THROW("Call InitNewtonIteration() first!");

  const int counter = MAX_NUMBER_LINE_SEARCH_ITERATIONS_ * newton_iteration;
  every_iter_writer_->write_mesh(counter, counter);
  every_iter_writer_->new_step(counter, counter);

  constexpr bool force_prepare = false;
  interface_->prepare_output(force_prepare);
  interface_->output_debug_state(*every_iter_writer_, write_owner_each_newton_iteration_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::IO::EveryIterationWriter::add_line_search_iteration(
    const int newton_iteration, const int linesearch_iteration)
{
  throw_if_not_setup(__LINE__);

  if (not write_this_step()) return;

  if (not isnewton_initialized_) FOUR_C_THROW("Call InitNewtonIteration() first!");

  if (linesearch_iteration >= static_cast<int>(MAX_NUMBER_LINE_SEARCH_ITERATIONS_))
    FOUR_C_THROW(
        "The EveryIterationWriter does not support more than %d line search"
        " steps. If this number is exceeded, the counters will get mixed up.",
        MAX_NUMBER_LINE_SEARCH_ITERATIONS_ - 1);

  const int counter = MAX_NUMBER_LINE_SEARCH_ITERATIONS_ * newton_iteration + linesearch_iteration;
  every_iter_writer_->write_mesh(counter, counter);
  every_iter_writer_->new_step(counter, counter);

  constexpr bool force_prepare = false;
  interface_->prepare_output(force_prepare);
  interface_->output_debug_state(*every_iter_writer_, write_owner_each_newton_iteration_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Core::IO::count_lines_in_file(const std::string& filepath)
{
  std::ifstream myfile(filepath);

  if (not myfile.is_open()) return -1;

  // new lines will be skipped unless we stop it from happening:
  myfile.unsetf(std::ios_base::skipws);

  // count the newlines with an algorithm specialized for counting:
  int line_count =
      std::count(std::istream_iterator<char>(myfile), std::istream_iterator<char>(), '\n');

  myfile.close();

  return line_count;
}

FOUR_C_NAMESPACE_CLOSE
