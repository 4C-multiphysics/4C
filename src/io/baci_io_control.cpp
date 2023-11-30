/*----------------------------------------------------------------------*/
/*! \file
 * \brief output control
\level 0
*/
/*----------------------------------------------------------------------*/


#include "baci_config.H"
#include "baci_config_revision.H"

#include "baci_io_control.H"

#include "baci_inpar_problemtype.H"
#include "baci_io_legacy_table_cpp.h"  // access to legacy parser module
#include "baci_io_pstream.H"

#include <Epetra_MpiComm.h>
#include <pwd.h>
#include <unistd.h>

#include <ctime>
#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
IO::OutputControl::OutputControl(const Epetra_Comm& comm, std::string problemtype,
    ShapeFunctionType spatial_approx, std::string inputfile, const std::string& outputname,
    int ndim, int restart, int filesteps, int create_controlfile)
    : problemtype_(std::move(problemtype)),
      inputfile_(std::move(inputfile)),
      ndim_(ndim),
      filename_(outputname),
      restartname_(outputname),
      filesteps_(filesteps),
      create_controlfile_(create_controlfile),
      myrank_(comm.MyPID())
{
  if (restart)
  {
    if (myrank_ == 0)
    {
      int number = 0;
      size_t pos = RestartFinder(filename_);
      if (pos != std::string::npos)
      {
        number = atoi(filename_.substr(pos + 1).c_str());
        filename_ = filename_.substr(0, pos);
      }

      for (;;)
      {
        number += 1;
        std::stringstream name;
        name << filename_ << "-" << number << ".control";
        std::ifstream file(name.str().c_str());
        if (not file)
        {
          filename_ = name.str();
          filename_ = filename_.substr(0, filename_.length() - 8);
          std::cout << "restart with new output file: " << filename_ << "\n";
          break;
        }
      }
    }

    if (comm.NumProc() > 1)
    {
      int length = static_cast<int>(filename_.length());
      std::vector<int> name(filename_.begin(), filename_.end());
      int err = comm.Broadcast(&length, 1, 0);
      if (err) dserror("communication error");
      name.resize(length);
      err = comm.Broadcast(name.data(), length, 0);
      if (err) dserror("communication error");
      filename_.assign(name.begin(), name.end());
    }
  }

  std::stringstream name;
  name << filename_ << ".control";
  WriteHeader(name.str(), spatial_approx);

  InsertRestartBackReference(restart, outputname);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
IO::OutputControl::OutputControl(const Epetra_Comm& comm, std::string problemtype,
    ShapeFunctionType spatial_approx, std::string inputfile, const std::string& restartname,
    std::string outputname, int ndim, int restart, int filesteps, int create_controlfile,
    bool adaptname)
    : problemtype_(std::move(problemtype)),
      inputfile_(std::move(inputfile)),
      ndim_(ndim),
      filename_(std::move(outputname)),
      restartname_(restartname),
      filesteps_(filesteps),
      create_controlfile_(create_controlfile),
      myrank_(comm.MyPID())
{
  if (restart)
  {
    if (myrank_ == 0 && adaptname)
    {
      // check whether filename_ includes a dash and in case separate the number at the end
      int number = 0;
      size_t pos = RestartFinder(filename_);
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
            std::cout << "restart with new output file: " << filename_ << std::endl;
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
          std::cout << "restart with new output file: " << filename_ << std::endl;
          break;
        }
      }
    }

    if (comm.NumProc() > 1)
    {
      int length = static_cast<int>(filename_.length());
      std::vector<int> name(filename_.begin(), filename_.end());
      int err = comm.Broadcast(&length, 1, 0);
      if (err) dserror("communication error");
      name.resize(length);
      err = comm.Broadcast(name.data(), length, 0);
      if (err) dserror("communication error");
      filename_.assign(name.begin(), name.end());
    }
  }

  if (create_controlfile_)
  {
    std::stringstream name;
    name << filename_ << ".control";

    WriteHeader(name.str(), spatial_approx);

    InsertRestartBackReference(restart, restartname);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
IO::OutputControl::OutputControl(const OutputControl& ocontrol, const char* new_prefix)
    : problemtype_(ocontrol.problemtype_),
      inputfile_(ocontrol.inputfile_),
      ndim_(ocontrol.ndim_),
      filename_(ocontrol.filename_),
      restartname_(ocontrol.restartname_),
      filesteps_(ocontrol.filesteps_),
      create_controlfile_(ocontrol.create_controlfile_),
      myrank_(ocontrol.myrank_)
{
  // replace file names if provided
  if (new_prefix)
  {
    // modify file name
    {
      std::string filename_path;
      std::string filename_suffix;
      size_t pos = filename_.rfind('/');

      if (pos != std::string::npos) filename_path = filename_.substr(0, pos + 1);

      filename_suffix = filename_.substr(pos + 1);
      filename_ = filename_path + new_prefix + filename_suffix;
    }

    // modify restart name
    {
      std::string restartname_path;
      std::string restartname_suffix;
      size_t pos = restartname_.rfind('/');

      if (pos != std::string::npos) restartname_path = restartname_.substr(0, pos + 1);

      restartname_suffix = restartname_.substr(pos + 1);
      restartname_ = restartname_path + new_prefix + restartname_suffix;
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IO::OutputControl::OverwriteResultFile(const ShapeFunctionType& spatial_approx)
{
  std::stringstream name;
  name << filename_ << ".control";

  WriteHeader(name.str(), spatial_approx);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IO::OutputControl::NewResultFile(int numb_run, const ShapeFunctionType& spatial_approx)
{
  if (filename_.rfind("_run_") != std::string::npos)
  {
    size_t pos = filename_.rfind("_run_");
    if (pos == std::string::npos) dserror("inconsistent file name");
    filename_ = filename_.substr(0, pos);
  }

  std::stringstream name;
  name << filename_ << "_run_" << numb_run;
  filename_ = name.str();
  name << ".control";


  WriteHeader(name.str(), spatial_approx);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IO::OutputControl::NewResultFile(
    const std::string& name_appendix, int numb_run, const ShapeFunctionType& spatial_approx)
{
  std::stringstream name;
  name << name_appendix;
  name << "_run_" << numb_run;

  NewResultFile(name.str(), spatial_approx);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IO::OutputControl::NewResultFile(std::string name, const ShapeFunctionType& spatial_approx)
{
  filename_ = name;
  name += ".control";

  WriteHeader(name, spatial_approx);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void IO::OutputControl::WriteHeader(
    const std::string& control_file_name, const ShapeFunctionType& spatial_approx)
{
  if (myrank_ == 0)
  {
    if (controlfile_.is_open()) controlfile_.close();

    controlfile_.open(control_file_name.c_str(), std::ios_base::out);
    if (not controlfile_)
      dserror("Could not open control file '%s' for writing", control_file_name.c_str());

    time_t time_value;
    time_value = time(nullptr);

    std::array<char, 256> hostname;
    struct passwd* user_entry;
    user_entry = getpwuid(getuid());
    gethostname(hostname.data(), 256);

    controlfile_ << "# baci output control file\n"
                 << "# created by " << user_entry->pw_name << " on " << hostname.data() << " at "
                 << ctime(&time_value) << "# using code version (git SHA1) " << BaciGitHash
                 << " \n\n"
                 << "input_file = \"" << inputfile_ << "\"\n"
                 << "problem_type = \"" << problemtype_ << "\"\n"
                 << "spatial_approximation = \""
                 << INPAR::PROBLEMTYPE::ShapeFunctionTypeToString(spatial_approx) << "\"\n"
                 << "ndim = " << ndim_ << "\n"
                 << "\n";

    controlfile_ << std::flush;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void IO::OutputControl::InsertRestartBackReference(int restart, const std::string& outputname)
{
  if (myrank_ != 0) return;

  // insert back reference
  if (restart)
  {
    size_t pos = outputname.rfind('/');
    controlfile_ << "restarted_run = \""
                 << ((pos != std::string::npos) ? outputname.substr(pos + 1) : outputname)
                 << "\"\n\n";

    controlfile_ << std::flush;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::string IO::OutputControl::FileNameOnlyPrefix() const
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
std::string IO::OutputControl::DirectoryName() const
{
  std::filesystem::path path(filename_);
  return path.parent_path();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
IO::InputControl::InputControl(const std::string& filename, const bool serial) : filename_(filename)
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
IO::InputControl::InputControl(const std::string& filename, const Epetra_Comm& comm)
    : filename_(filename)
{
  std::stringstream name;
  name << filename << ".control";

  // works for parallel, as well as serial applications because we only have an Epetra_MpiComm
  const auto* epetrampicomm = dynamic_cast<const Epetra_MpiComm*>(&comm);
  if (!epetrampicomm)
    dserror("ERROR: casting Epetra_Comm -> Epetra_MpiComm failed");
  else
  {
    const MPI_Comm lcomm = epetrampicomm->GetMpiComm();
    parse_control_file(&table_, name.str().c_str(), lcomm);
  }
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
IO::InputControl::~InputControl() { destroy_map(&table_); }

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
IO::ErrorFileControl::ErrorFileControl(
    const Epetra_Comm& comm, std::string outputname, int restart, int create_errorfiles)
    : filename_(std::move(outputname)), errfile_(nullptr)
{
  // create error file name
  {
    // standard mode, no restart
    std::ostringstream mypid;
    mypid << comm.MyPID();
    errname_ = filename_ + mypid.str() + ".err";

    if (restart)
    {
      // check whether filename_ includes a dash and in case separate the number at the end
      int number = 0;
      size_t pos = RestartFinder(filename_);
      if (pos != std::string::npos)
      {
        number = atoi(filename_.substr(pos + 1).c_str());
        filename_ = filename_.substr(0, pos);
      }

      // either add or increase the number in the end or just set the new name for the error file
      for (;;)
      {
        // if no number is found and the error file name does not yet exist -> create it
        if (number == 0)
        {
          errname_ = filename_ + mypid.str() + ".err";
          std::ifstream file(errname_.c_str());
          if (not file)
          {
            break;
          }
        }
        // a number was found or the file does already exist -> set number correctly and add it
        number += 1;
        std::stringstream name;
        name << "-" << number << "_";
        errname_ = filename_ + name.str() + mypid.str() + ".err";
        std::ifstream file(errname_.c_str());
        if (not file)
        {
          break;
        }
      }
    }
  }

  // open error files (one per processor)
  // int create_errorfiles =0;
  if (create_errorfiles)
  {
    errfile_ = fopen(errname_.c_str(), "w");
    if (errfile_ == nullptr) dserror("Opening of output file %s failed\n", errname_.c_str());
  }

  // inform user
  if (comm.MyPID() == 0) IO::cout << "errors are reported to " << errname_.c_str() << IO::endl;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
IO::ErrorFileControl::~ErrorFileControl()
{
  if (errfile_ != nullptr) fclose(errfile_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
size_t IO::RestartFinder(const std::string& filename)
{
  size_t pos;
  for (pos = filename.size(); pos > 0; --pos)
  {
    if (filename[pos - 1] == '-') return pos - 1;

    if (not std::isdigit(filename[pos - 1]) or filename[pos - 1] == '/') return std::string::npos;
  }
  return std::string::npos;
}
