// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_pstream.hpp"

#include "4C_comm_mpi_utils.hpp"

#include <mpi.h>
#include <Teuchos_oblackholestream.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /// this is the Core::IO::cout that everyone can refer to
  Pstream cout;
}  // namespace Core::IO


/*----------------------------------------------------------------------*
 * empty constructor                                          wic 11/12 *
 *----------------------------------------------------------------------*/
Core::IO::Pstream::Pstream()
    : is_initialized_(false),
      comm_(MPI_COMM_NULL),
      targetpid_(-2),
      writetoscreen_(false),
      writetofile_(false),
      outfile_(nullptr),
      prefixgroup_id_(false),
      group_id_(-2),
      buffer_(std::string()),
      requestedoutputlevel_(undef),
      level_(new Level(this))
{
}


/*----------------------------------------------------------------------*
 * destructor                                                 wic 09/16 *
 *----------------------------------------------------------------------*/
Core::IO::Pstream::~Pstream()
{
  if (level_) delete level_;
  level_ = nullptr;

  if (blackholestream_) delete blackholestream_;
  blackholestream_ = nullptr;

  mystream_ = nullptr;

  this->close();
}

/*----------------------------------------------------------------------*
 * configure the output                                       wic 11/12 *
 *----------------------------------------------------------------------*/
void Core::IO::Pstream::setup(const bool writetoscreen, const bool writetofile,
    const bool prefixgroupID, const Core::IO::Verbositylevel level, MPI_Comm comm,
    const int targetpid, const int groupID, const std::string fileprefix)
{
  // make sure that setup is called only once or we get unpredictable behavior
  if (is_initialized_) FOUR_C_THROW("Thou shalt not call setup on the output twice!");
  is_initialized_ = true;

  requestedoutputlevel_ = level;
  comm_ = comm;
  targetpid_ = targetpid;
  writetoscreen_ = writetoscreen;
  writetofile_ = writetofile;
  outfile_ = nullptr;
  prefixgroup_id_ = prefixgroupID;
  group_id_ = groupID;

  // make sure the target processor exists
  if (targetpid_ >= Core::Communication::num_mpi_ranks(comm_))
    FOUR_C_THROW("Chosen target processor does not exist.");

  // prepare the file handle
  if (on_pid() and writetofile_)
  {
    std::stringstream fname;
    fname << fileprefix << ".p" << std::setfill('0') << std::setw(2)
          << Core::Communication::my_mpi_rank(comm_) << ".log";
    outfile_ = new std::ofstream(fname.str().c_str());
    if (!outfile_) FOUR_C_THROW("could not open output file");
  }

  // prepare the very first line of output
  if (on_pid() and prefixgroup_id_) buffer_ << group_id_ << ": ";

  // setup mystream
  blackholestream_ = new Teuchos::oblackholestream;
  if (writetoscreen_ and (Core::Communication::my_mpi_rank(comm_) == targetpid_ or targetpid_ < 0))
    mystream_ = &std::cout;
  else
    mystream_ = blackholestream_;
}

/*-----------------------------------------------------------------------*
 * return a std::ostream following the restrictions      hiermeier 12/17 *
 *-----------------------------------------------------------------------*/
std::ostream& Core::IO::Pstream::os(const Verbositylevel level) const
{
  if (not is_initialized_) FOUR_C_THROW("Setup the output before you use it!");

  if (level <= requested_output_level())
  {
    if (prefixgroup_id_) *mystream_ << prefixgroup_id_ << ": ";
    return *mystream_;
  }
  else
    return *blackholestream_;
}

/*----------------------------------------------------------------------*
 * close open file handles and reset pstream                  wic 11/12 *
 *----------------------------------------------------------------------*/
void Core::IO::Pstream::close()
{
  if (not is_initialized_) return;

  is_initialized_ = false;
  comm_ = MPI_COMM_NULL;
  targetpid_ = -2;
  writetoscreen_ = false;
  writetofile_ = false;

  // close file
  if (outfile_)
  {
    outfile_->close();
    delete outfile_;
  }
  outfile_ = nullptr;

  if (blackholestream_)
  {
    blackholestream_->flush();
    delete blackholestream_;
    blackholestream_ = nullptr;
  }

  prefixgroup_id_ = false;
  group_id_ = -2;

  // flush the buffer
  if (writetoscreen_ and on_pid() and buffer_.str().size() > 0)
    std::cout << buffer_.str() << std::flush;
  buffer_.str(std::string());
}


/*----------------------------------------------------------------------*
 * writes the buffer to screen                                wic 11/12 *
 *----------------------------------------------------------------------*/
void Core::IO::Pstream::flush()
{
  if (not is_initialized_) FOUR_C_THROW("Setup the output before you use it!");

  if (on_pid() and writetoscreen_ and buffer_.str().size() > 0)
  {
    std::cout << buffer_.str();
    std::flush(std::cout);
    buffer_.str(std::string());
  }

  return;
}


/*----------------------------------------------------------------------*
 * return whether this is a target processor                  wic 11/12 *
 *----------------------------------------------------------------------*/
bool Core::IO::Pstream::on_pid()
{
  if (targetpid_ < 0) return true;
  return (Core::Communication::my_mpi_rank(comm_) == targetpid_);
}


/*----------------------------------------------------------------------*
 * set output level                                           wic 09/16 *
 *----------------------------------------------------------------------*/
Core::IO::Level& Core::IO::Pstream::operator()(const Verbositylevel level)
{
  return level_->set_level(level);
}


/*----------------------------------------------------------------------*
 * Imitate the std::endl behavior w/out the flush             wic 11/12 *
 *----------------------------------------------------------------------*/
Core::IO::Pstream& Core::IO::endl(Core::IO::Pstream& out)
{
  out << "\n";
  return out;
}

/*----------------------------------------------------------------------*
 * Imitate the std::endl behavior w/out the flush             wic 09/16 *
 *----------------------------------------------------------------------*/
Core::IO::Level& Core::IO::endl(Core::IO::Level& out)
{
  out << "\n";
  return out;
}

/*----------------------------------------------------------------------*
 * Imitate the std::flush behavior                            wic 11/12 *
 *----------------------------------------------------------------------*/
Core::IO::Pstream& Core::IO::flush(Core::IO::Pstream& out)
{
  out.flush();
  return out;
}


/*----------------------------------------------------------------------*
 * Imitate the std::flush behavior                            wic 11/12 *
 *----------------------------------------------------------------------*/
Core::IO::Level& Core::IO::flush(Core::IO::Level& out)
{
  out.flush();
  return out;
}


/*----------------------------------------------------------------------*
 * writes the buffer to screen                                wic 09/16 *
 *----------------------------------------------------------------------*/
void Core::IO::Level::flush()
{
  if (level_ <= pstream_->requested_output_level()) pstream_->flush();

  return;
}


/*----------------------------------------------------------------------*
 * Handle special manipulators                                wic 11/12 *
 *----------------------------------------------------------------------*/
Core::IO::Pstream& Core::IO::operator<<(
    Core::IO::Pstream& out, Core::IO::Pstream& (*pf)(Core::IO::Pstream&))
{
  return pf(out);
}

/*----------------------------------------------------------------------*
 * Handle special manipulators                                wic 09/16 *
 *----------------------------------------------------------------------*/
Core::IO::Level& Core::IO::operator<<(
    Core::IO::Level& out, Core::IO::Level& (*pf)(Core::IO::Level&))
{
  return pf(out);
}

FOUR_C_NAMESPACE_CLOSE
